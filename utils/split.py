from __future__ import annotations

from pathlib import Path

import polars as pl

from utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SplitConfigError(RuntimeError):
    """Raised when the precomputed split parquet is missing or has no rows for a session."""


def _splits_dir(data_root: str | Path) -> Path:
    """Split parquets live at ``<data_root>/../splits/`` alongside the recordings."""
    return Path(data_root).parent / "splits"


def _load_block_set(
    split: str, participant: str, session: str, data_root: str | Path
) -> set[int]:
    """Return the set of block IDs assigned to ``split`` for (participant, session)."""
    path = _splits_dir(data_root) / f"{split}.parquet"
    if not path.exists():
        raise SplitConfigError(
            f"Split parquet not found: {path}. "
            f"Run `python training/precompute_splits.py --participant {participant} "
            f"--session {session}` first."
        )
    df = pl.read_parquet(path).filter(
        (pl.col("participant_id") == participant) & (pl.col("session") == int(session))
    )
    if df.height == 0:
        raise SplitConfigError(
            f"No rows for participant={participant} session={session} in {path}. "
            f"Run `python training/precompute_splits.py --participant {participant} "
            f"--session {session}` first."
        )
    return set(df["block"].unique().to_list())


def create_splits(
    recordings: pl.DataFrame,
    save_dir: Path,
    data_root: str | Path,
):
    """Apply precomputed block-level split to the trial-level recordings DataFrame.

    Reads ``<data_root>/../splits/{train,val,test}.parquet`` (produced by
    ``training/precompute_splits.py``) and partitions trials by block
    membership into train / val / test parquets under ``save_dir/``.

    Every framework (PSID, DPAD, VARMA) and every variant sees identical
    block assignments by construction, regardless of DBS filtering.
    """
    logger = get_logger()

    participants = recordings["participant_id"].unique().to_list()
    sessions = recordings["session"].unique().to_list()
    if len(participants) != 1 or len(sessions) != 1:
        raise SplitConfigError(
            f"Expected single (participant, session); got participants={participants}, "
            f"sessions={sessions}."
        )
    participant = str(participants[0])
    session = str(sessions[0])

    assignment = {
        split: _load_block_set(split, participant, session, data_root)
        for split in ("train", "val", "test")
    }
    logger.info(
        f"Loaded splits for {participant}_S{session}: "
        + " | ".join(f"{k}={sorted(v)}" for k, v in assignment.items())
    )

    all_assigned = assignment["train"] | assignment["val"] | assignment["test"]
    present = set(recordings["block"].unique().to_list())
    missing = present - all_assigned
    if missing:
        logger.warning(
            f"Blocks in data but not in split parquets: {sorted(missing)} — dropped."
        )

    subsets = {
        name: recordings.filter(pl.col("block").is_in(list(blocks)))
        for name, blocks in assignment.items()
    }

    def _summarize(name: str, df: pl.DataFrame) -> str:
        if df.height == 0:
            return f"{name}=0 trials"
        ep = int(df["n_epochs"].sum()) if "n_epochs" in df.columns else df.height
        if "stim" in df.columns:
            dbs_cnt = df.group_by("stim").agg(pl.len().alias("n")).sort("stim")
            dbs_str = ", ".join(
                f"{r['stim']}:{r['n']}" for r in dbs_cnt.iter_rows(named=True)
            )
            return f"{name}={df.height} trials ({ep} epochs) [{dbs_str}]"
        return f"{name}={df.height} trials ({ep} epochs)"

    logger.info(
        "Resulting splits: "
        + " | ".join(_summarize(n, subsets[n]) for n in ("train", "val", "test"))
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, df in subsets.items():
        df.write_parquet(save_dir / f"{name}.parquet")
