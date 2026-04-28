#!/usr/bin/env python3
"""Precompute balanced block-chronological splits for every session and
write per-session YAMLs under ``configs/splits/{participant}_S{session}.yaml``.

Each YAML carries train / val / test block-id lists. ``utils/split.py:
create_splits`` loads the matching file at training time and assigns trials
by block membership — so every framework (PSID, DPAD, VARMA) and every
variant sees byte-identical block assignments by construction.

Algorithm (same as create_splits):
  * group blocks by DBS condition ('stim'), sort chronologically.
  * min_n = min(#off_blocks, #on_blocks).
  * per-cond: first round(train*min_n) → train,
              last  round(test *min_n) → test,
              middle (+surplus) → val.

Usage:
    python training/precompute_splits.py
    python training/precompute_splits.py --participant PDI4 --session 3
    python training/precompute_splits.py --train 0.5 --test 0.4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "resampled_recordings" / "participants_at_200Hz_scaled_1e6_narrow_band"
OUT_DIR = PROJECT_ROOT / "configs" / "splits"


def block_metadata(participant: str, session: str) -> pl.DataFrame:
    sess = DATA_ROOT / f"participant_id={participant}" / f"session={session}"
    rows = []
    for b in sorted(sess.glob("block=*")):
        pq_files = list(b.glob("*.parquet"))
        if not pq_files:
            continue
        df = pl.read_parquet(pq_files[0])
        block_id = int(b.name.split("=")[1])
        stim_vals = [str(v).lower() for v in df["stim"].unique().to_list()]
        dbs = "mixed" if len(set(stim_vals)) > 1 else ("on" if stim_vals[0] in ("on", "true", "1") else "off")
        trial_col = "trial" if "trial" in df.columns else "trial_number"
        rows.append({"block": block_id, "dbs": dbs, "n_trials": df[trial_col].n_unique()})
    return pl.DataFrame(rows).sort("block")


def allocate(blocks_df: pl.DataFrame, train_frac: float, test_frac: float) -> dict:
    conds = blocks_df["dbs"].unique().to_list()

    fold_map: dict = {}

    if len(conds) == 1:
        blocks = blocks_df["block"].to_list()
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
            c: blocks_df.filter(pl.col("dbs") == c)["block"].to_list()
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

    out = {"train_blocks": [], "val_blocks": [], "test_blocks": []}
    for b in sorted(fold_map):
        out[f"{fold_map[b]}_blocks"].append(b)
    return out


def discover_sessions() -> list[tuple[str, str]]:
    out = []
    for p_dir in sorted(DATA_ROOT.glob("participant_id=*")):
        p = p_dir.name.split("=")[1]
        for s_dir in sorted(p_dir.glob("session=*")):
            s = s_dir.name.split("=")[1]
            if list(s_dir.glob("block=*")):
                out.append((p, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--train", type=float, default=0.5)
    ap.add_argument("--test", type=float, default=0.4)
    args = ap.parse_args()

    if args.participant and args.session:
        sessions = [(args.participant, args.session)]
    else:
        sessions = discover_sessions()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p, s in sessions:
        try:
            meta = block_metadata(p, s)
        except Exception as e:
            print(f"[skip] {p}_S{s}: {e}")
            continue
        if meta.height == 0:
            print(f"[skip] {p}_S{s}: no blocks found")
            continue

        alloc = allocate(meta, train_frac=args.train, test_frac=args.test)

        off_n = meta.filter(pl.col("dbs") == "off").height
        on_n = meta.filter(pl.col("dbs") == "on").height
        dbs_of = meta.with_columns(pl.col("block").cast(pl.Int64))
        cond_by_block = {
            r["block"]: r["dbs"] for r in dbs_of.iter_rows(named=True)
        }
        trials_by_block = {
            r["block"]: r["n_trials"] for r in meta.iter_rows(named=True)
        }

        def _fold_stats(blocks: list[int]) -> dict:
            off = sum(trials_by_block[b] for b in blocks if cond_by_block[b] == "off")
            on = sum(trials_by_block[b] for b in blocks if cond_by_block[b] == "on")
            return {"n_blocks": len(blocks), "n_trials_off": off, "n_trials_on": on}

        payload = {
            "participant": p,
            "session": int(s),
            "strategy": "balanced_block_chronological",
            "train_frac": args.train,
            "test_frac": args.test,
            "n_blocks_total": meta.height,
            "n_blocks_off": off_n,
            "n_blocks_on": on_n,
            "train_blocks": alloc["train_blocks"],
            "val_blocks": alloc["val_blocks"],
            "test_blocks": alloc["test_blocks"],
            "summary": {
                "train": _fold_stats(alloc["train_blocks"]),
                "val": _fold_stats(alloc["val_blocks"]),
                "test": _fold_stats(alloc["test_blocks"]),
            },
        }

        out_path = OUT_DIR / f"{p}_S{s}.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        s_sum = payload["summary"]
        print(
            f"[ok]   {p}_S{s}: "
            f"train={alloc['train_blocks']} "
            f"val={alloc['val_blocks']} "
            f"test={alloc['test_blocks']} "
            f"trials off/on = "
            f"train {s_sum['train']['n_trials_off']}/{s_sum['train']['n_trials_on']}, "
            f"val {s_sum['val']['n_trials_off']}/{s_sum['val']['n_trials_on']}, "
            f"test {s_sum['test']['n_trials_off']}/{s_sum['test']['n_trials_on']}"
        )
        print(f"       → {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
