#!/usr/bin/env python3
"""Precompute balanced block-chronological splits for every session.

Writes three shared parquets co-located with the recordings:

    <data_root>/../splits/train.parquet
    <data_root>/../splits/val.parquet
    <data_root>/../splits/test.parquet

Each parquet has columns: ``participant_id``, ``session``, ``block``,
``trial``, ``dbs_state``. Running for a new session upserts into the
existing files (removes stale rows for that participant+session, appends
fresh rows).

Algorithm:
  * Group blocks by DBS condition, sort chronologically.
  * min_n = min(#off_blocks, #on_blocks).
  * Per condition: first round(train_frac * min_n) -> train,
                   last  round(test_frac  * min_n) -> test,
                   middle (+surplus) -> val.

Usage:
    python training/precompute_splits.py
    python training/precompute_splits.py --participant PDI4 --session 3
    python training/precompute_splits.py --train 0.5 --test 0.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

_DEFAULT_DATA_ROOT = "resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band"

_SPLIT_NAMES = ("train", "val", "test")


def splits_dir(data_root: Path) -> Path:
    """Split parquets live alongside the recordings: ``<data_root>/../splits/``."""
    return data_root.parent / "splits"


def load_trial_index(data_root: Path, participant: str, session: str) -> pl.DataFrame:
    """Return one row per trial: block, trial, dbs_state."""
    sess = data_root / f"participant_id={participant}" / f"session={session}"
    rows = []
    for b_dir in sorted(sess.glob("block=*")):
        pq_files = list(b_dir.glob("*.parquet"))
        if not pq_files:
            continue
        block_id = int(b_dir.name.split("=")[1])
        df = pl.read_parquet(pq_files[0])
        trial_col = "trial" if "trial" in df.columns else "trial_number"
        stim_col = "stim" if "stim" in df.columns else None
        for row in df.select([trial_col] + ([stim_col] if stim_col else [])).iter_rows(
            named=True
        ):
            trial_id = row[trial_col]
            stim_raw = (
                str(row.get(stim_col, "unknown")).lower() if stim_col else "unknown"
            )
            dbs = (
                "on"
                if stim_raw in ("on", "true", "1")
                else ("off" if stim_raw in ("off", "false", "0") else stim_raw)
            )
            rows.append({"block": block_id, "trial": trial_id, "dbs_state": dbs})
    return pl.DataFrame(
        rows, schema={"block": pl.Int64, "trial": pl.Int64, "dbs_state": pl.Utf8}
    ).sort(["block", "trial"])


def allocate_blocks(
    trial_index: pl.DataFrame, train_frac: float, test_frac: float
) -> dict[int, str]:
    """Return {block_id: split_name} using balanced block-chronological logic."""
    block_df = (
        trial_index.group_by("block").agg(pl.col("dbs_state").first()).sort("block")
    )
    conds = block_df["dbs_state"].unique().to_list()
    fold_map: dict[int, str] = {}

    if len(conds) == 1:
        blocks = block_df["block"].to_list()
        n = len(blocks)
        n_train = round(train_frac * n)
        n_test = round(test_frac * n)
        for i, b in enumerate(blocks):
            if i < n_train:
                fold_map[b] = "train"
            elif i >= n - n_test:
                fold_map[b] = "test"
            else:
                fold_map[b] = "val"
    else:
        per_cond = {
            c: block_df.filter(pl.col("dbs_state") == c)["block"].to_list()
            for c in conds
        }
        min_n = min(len(v) for v in per_cond.values())
        n_train = round(train_frac * min_n)
        n_test = round(test_frac * min_n)
        for blocks in per_cond.values():
            n = len(blocks)
            for i, b in enumerate(blocks):
                if i < n_train:
                    fold_map[b] = "train"
                elif i >= n - n_test:
                    fold_map[b] = "test"
                else:
                    fold_map[b] = "val"
    return fold_map


def upsert_split_parquets(
    out_dir: Path,
    participant: str,
    session: str,
    trial_index: pl.DataFrame,
    block_assignment: dict[int, str],
) -> dict[str, int]:
    """Upsert rows for (participant, session) into the three split parquets.

    Returns {split_name: n_trials} for logging.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for split in _SPLIT_NAMES:
        blocks_in_split = {b for b, s in block_assignment.items() if s == split}
        new_rows = (
            trial_index.filter(pl.col("block").is_in(list(blocks_in_split)))
            .with_columns(
                [
                    pl.lit(participant).alias("participant_id"),
                    pl.lit(int(session)).alias("session"),
                ]
            )
            .select(["participant_id", "session", "block", "trial", "dbs_state"])
        )
        counts[split] = new_rows.height

        out_path = out_dir / f"{split}.parquet"
        if out_path.exists():
            existing = pl.read_parquet(out_path).filter(
                ~(
                    (pl.col("participant_id") == participant)
                    & (pl.col("session") == int(session))
                )
            )
            combined = pl.concat([existing, new_rows], how="diagonal_relaxed")
        else:
            combined = new_rows

        combined.sort(["participant_id", "session", "block", "trial"]).write_parquet(
            out_path
        )

    return counts


def discover_sessions(data_root: Path) -> list[tuple[str, str]]:
    out = []
    for p_dir in sorted(data_root.glob("participant_id=*")):
        p = p_dir.name.split("=")[1]
        for s_dir in sorted(p_dir.glob("session=*")):
            s = s_dir.name.split("=")[1]
            if list(s_dir.glob("block=*")):
                out.append((p, s))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Precompute train/val/test splits as shared parquets."
    )
    ap.add_argument("--participant", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--train", type=float, default=0.5)
    ap.add_argument("--test", type=float, default=0.4)
    ap.add_argument(
        "--data-root",
        default=None,
        help=f"Override data dir. Default: {_DEFAULT_DATA_ROOT}",
    )
    args = ap.parse_args()

    raw = args.data_root or _DEFAULT_DATA_ROOT
    data_root = Path(raw) if Path(raw).is_absolute() else PROJECT_ROOT / raw
    if not data_root.exists():
        print(f"[error] data root not found: {data_root}")
        sys.exit(2)
    print(f"[run] data root: {data_root.relative_to(PROJECT_ROOT)}")

    out_dir = splits_dir(data_root)
    sessions = (
        [(args.participant, args.session)]
        if args.participant and args.session
        else discover_sessions(data_root)
    )

    for p, s in sessions:
        try:
            trial_index = load_trial_index(data_root, p, s)
        except Exception as e:
            print(f"[skip] {p}_S{s}: {e}")
            continue
        if trial_index.height == 0:
            print(f"[skip] {p}_S{s}: no trials found")
            continue

        block_assignment = allocate_blocks(trial_index, args.train, args.test)
        counts = upsert_split_parquets(out_dir, p, s, trial_index, block_assignment)

        train_b = sorted(b for b, sp in block_assignment.items() if sp == "train")
        val_b = sorted(b for b, sp in block_assignment.items() if sp == "val")
        test_b = sorted(b for b, sp in block_assignment.items() if sp == "test")
        print(
            f"[ok]  {p}_S{s}: "
            f"train_blocks={train_b} ({counts['train']} trials)  "
            f"val_blocks={val_b} ({counts['val']} trials)  "
            f"test_blocks={test_b} ({counts['test']} trials)"
        )
    print(
        f"[done] splits -> {out_dir.relative_to(PROJECT_ROOT)}/{{train,val,test}}.parquet"
    )


if __name__ == "__main__":
    main()
