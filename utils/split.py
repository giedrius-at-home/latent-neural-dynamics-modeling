from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import yaml

from utils.config import Config
from utils.logger import get_logger


# Precomputed split YAMLs live here. Produce with training/precompute_splits.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_CONFIG_DIR = PROJECT_ROOT / "configs" / "splits"


class SplitConfigError(RuntimeError):
    """Raised when the precomputed split YAML is missing or malformed."""


def _find_split_yaml(participant: str, session: str) -> Path:
    path = SPLITS_CONFIG_DIR / f"{participant}_S{session}.yaml"
    if not path.exists():
        raise SplitConfigError(
            f"Precomputed split YAML not found: {path}. "
            f"Run `python training/precompute_splits.py --participant {participant} "
            f"--session {session}` first."
        )
    return path


def _load_block_assignment(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        payload = yaml.safe_load(f)
    for key in ("train_blocks", "val_blocks", "test_blocks"):
        if key not in payload:
            raise SplitConfigError(f"{yaml_path} missing '{key}'")
    return {
        "train": set(payload["train_blocks"]),
        "val": set(payload["val_blocks"]),
        "test": set(payload["test_blocks"]),
    }


def create_splits(
    recordings: pl.DataFrame, split_params: Config, results_config: Config
):
    """Apply precomputed block-level split to the trial-level recordings DF.

    Reads ``configs/splits/{participant}_S{session}.yaml`` (produced by
    ``training/precompute_splits.py``) and partitions trials by block
    membership into train / val / test parquets under ``<save_dir>/split/``.

    All fold computation happens ahead of time in the precompute script — so
    every framework (PSID, DPAD, VARMA) and every variant sees byte-identical
    splits by construction, regardless of pre-filtering (e.g. ``dbs_condition``
    set to "on" / "off" still reuses the same per-session block lists; the
    intersection with the dbs-filtered recordings yields a single-condition
    subset of the same blocks).
    """
    logger = get_logger()

    assert split_params.within_session_split, "Only within-session split is supported"

    participants = recordings["participant_id"].unique().to_list()
    sessions = recordings["session"].unique().to_list()
    if len(participants) != 1 or len(sessions) != 1:
        raise SplitConfigError(
            f"Expected single (participant, session); got participants={participants}, "
            f"sessions={sessions}. Precomputed splits are per-session."
        )
    participant = str(participants[0])
    session = str(sessions[0])

    yaml_path = _find_split_yaml(participant, session)
    assignment = _load_block_assignment(yaml_path)
    logger.info(f"Loaded split YAML: {yaml_path.relative_to(PROJECT_ROOT)}")

    all_assigned_blocks = assignment["train"] | assignment["val"] | assignment["test"]
    present_blocks = set(recordings["block"].unique().to_list())

    extra_in_yaml = all_assigned_blocks - present_blocks
    missing_from_yaml = present_blocks - all_assigned_blocks
    if missing_from_yaml:
        logger.warning(
            f"Blocks present in data but not in split YAML: {sorted(missing_from_yaml)} — "
            f"they will be dropped from all folds."
        )
    if extra_in_yaml:
        logger.info(
            f"Blocks in YAML but not in current data (likely filtered by dbs_condition): "
            f"{sorted(extra_in_yaml)}"
        )

    train_trials = recordings.filter(pl.col("block").is_in(list(assignment["train"])))
    val_trials = recordings.filter(pl.col("block").is_in(list(assignment["val"])))
    test_trials = recordings.filter(pl.col("block").is_in(list(assignment["test"])))

    def _summarize(name: str, df: pl.DataFrame) -> str:
        if df.height == 0:
            return f"{name}=0 trials"
        ep = int(df.select(pl.col("n_epochs").sum()).item()) if "n_epochs" in df.columns else df.height
        if "stim" in df.columns:
            dbs_cnt = (
                df.group_by("stim").agg(pl.len().alias("n")).sort("stim")
            )
            dbs_str = ", ".join(
                f"{r['stim']}:{r['n']}" for r in dbs_cnt.iter_rows(named=True)
            )
            return f"{name}={df.height} trials ({ep} epochs) [{dbs_str}]"
        return f"{name}={df.height} trials ({ep} epochs)"

    logger.info(
        "Resulting splits: "
        + " | ".join(
            _summarize(n, d)
            for n, d in (
                ("train", train_trials),
                ("val", val_trials),
                ("test", test_trials),
            )
        )
    )

    save_dir = Path(results_config.save_dir) / "split"
    save_dir.mkdir(parents=True, exist_ok=True)
    train_trials.write_parquet(save_dir / "train.parquet")
    val_trials.write_parquet(save_dir / "val.parquet")
    test_trials.write_parquet(save_dir / "test.parquet")
