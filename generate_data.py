#!/usr/bin/env python3
# Writes fake recordings in the shape package_recordings consumes, at the real
# dimensions, so the pipeline can be run without the real data. See README.md.

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FAKE_ROOT = PROJECT_ROOT / "fake_data"
RAW_ROOT = FAKE_ROOT / "raw"
PREPROCESSED_ROOT = FAKE_ROOT / "participants_fake_200Hz"

PARTICIPANT = "FAKE1"
SESSION = 1
RAW_FS = 1000  # recordings are 1000 Hz before preprocessing
OUT_FS = 200  # and 200 Hz after

CHANNELS = (
    [f"LFP_{i}" for i in range(1, 17)]
    + [f"ECOG_{i}" for i in range(1, 5)]
    + [f"EOG_{i}" for i in range(1, 5)]
)

# Taken from a real session (PDI1_S2): 12 blocks x 12 trials, 9 s per trial,
# onsets ~18 s apart, ~277 s of recording per block, ~1069 pen samples per trial.
DEF_BLOCKS = 12
DEF_TRIALS = 12
DEF_TRIAL_SECONDS = 9.0
DEF_ONSET_SPACING = 18.0
DEF_TAIL_SECONDS = 55.0
MOTION_FS = 119

def log(msg: str) -> None:
    print(f"[fake-data] {msg}", flush=True)


def generate(
    blocks: int,
    trials_per_block: int,
    trial_seconds: float,
    onset_spacing: float,
    seed: int,
) -> None:
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(seed)
    if RAW_ROOT.exists():
        shutil.rmtree(RAW_ROOT)

    session_dir = RAW_ROOT / f"sub-{PARTICIPANT}" / f"ses-{SESSION}"
    motion_dir = session_dir / "motion"
    ieeg_dir = RAW_ROOT / "resampled"
    motion_dir.mkdir(parents=True, exist_ok=True)
    ieeg_dir.mkdir(parents=True, exist_ok=True)

    onsets = [
        float(onset_spacing * (t + 1)) for t in range(trials_per_block)
    ]
    block_seconds = onsets[-1] + trial_seconds + DEF_TAIL_SECONDS
    n_raw = int(block_seconds * RAW_FS)
    n_motion = int(trial_seconds * MOTION_FS)

    for block in range(1, blocks + 1):
        # One DBS state per block, alternating — stimulation never switches
        # mid-block in the real recordings either.
        dbs_on = block % 2 == 0

        sig = rng.normal(scale=50.0, size=(n_raw, len(CHANNELS))).astype("float32")
        ieeg = {name: sig[:, i] for i, name in enumerate(CHANNELS)}
        ieeg["sfreq"] = np.full(n_raw, RAW_FS, dtype="float32")
        ieeg_path = (
            ieeg_dir
            / f"sub-{PARTICIPANT}_ses-{SESSION}_task-copydraw_run-{block}_ieeg.parquet"
        )
        pl.DataFrame(ieeg).write_parquet(ieeg_path)

        for trial in range(1, trials_per_block + 1):
            # Pen trace: random walk, integer coordinates as in the real files.
            walk = np.cumsum(rng.normal(scale=3.0, size=(n_motion, 2)), axis=0)
            xy = walk.round().astype(int)
            tsv = motion_dir / (
                f"sub-{PARTICIPANT}_ses-{SESSION}_task-copydraw"
                f"_run-{block:02d}_chunk-{trial:02d}_tracksys-wacom_motion.tsv"
            )
            tsv.write_text("\n".join(["x\ty"] + [f"{x}\t{y}" for x, y in xy]) + "\n")

        part_dir = (
            RAW_ROOT
            / "participants_2"
            / f"participant_id={PARTICIPANT}"
            / f"session={SESSION}"
            / f"block={block}"
        )
        part_dir.mkdir(parents=True, exist_ok=True)
        # Column names and dtypes as in the real data/participants_2 table.
        pl.DataFrame(
            {
                "participant_id": [PARTICIPANT],
                "session": pl.Series([SESSION], dtype=pl.UInt32),
                "block": pl.Series([block], dtype=pl.UInt64),
                "trials": pl.Series(
                    [list(range(1, trials_per_block + 1))], dtype=pl.List(pl.UInt64)
                ),
                "onsets": pl.Series([onsets], dtype=pl.List(pl.Float64)),
                "trial_time": pl.Series([float(trial_seconds)], dtype=pl.Float64),
                "is_fragmented": [False],
                "stim": ["on" if dbs_on else "off"],
                "ieeg_parquet": [str(ieeg_path)],
                "session_path": [str(session_dir)],
            }
        ).write_parquet(part_dir / "0.parquet")

    log(
        f"{blocks} blocks x {trials_per_block} trials, {trial_seconds:g}s each, "
        f"{block_seconds:.0f}s of {RAW_FS} Hz recording per block "
        f"-> {RAW_ROOT.relative_to(PROJECT_ROOT)}"
    )


def status() -> None:
    def count(root: Path, pattern: str) -> int:
        return len(list(root.glob(pattern))) if root.exists() else 0

    print()
    log("on disk:")
    print(
        f"  raw            {RAW_ROOT.relative_to(PROJECT_ROOT)}  "
        f"({count(RAW_ROOT, 'participants_2/**/block=*')} blocks, "
        f"{count(RAW_ROOT, 'resampled/*.parquet')} iEEG parquets, "
        f"{count(RAW_ROOT, '**/motion/*.tsv')} motion files)"
    )
    print(
        f"  preprocessed   {PREPROCESSED_ROOT.relative_to(PROJECT_ROOT)}  "
        f"({count(PREPROCESSED_ROOT, '**/block=*')} blocks)"
    )
    for split in ("train", "val", "test"):
        p = FAKE_ROOT / "splits" / f"{split}.parquet"
        print(f"  split {split:5s}    {'present' if p.exists() else '-'}")
    print()


def clean() -> None:
    if FAKE_ROOT.exists():
        shutil.rmtree(FAKE_ROOT)
        log(f"removed {FAKE_ROOT.relative_to(PROJECT_ROOT)}")
    else:
        log("nothing to clean")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate fake raw recordings for package_recordings to preprocess."
    )
    ap.add_argument("--blocks", type=int, default=DEF_BLOCKS, help="alternating DBS on/off")
    ap.add_argument("--trials-per-block", type=int, default=DEF_TRIALS)
    ap.add_argument("--trial-seconds", type=float, default=DEF_TRIAL_SECONDS)
    ap.add_argument("--onset-spacing", type=float, default=DEF_ONSET_SPACING)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", action="store_true", help="show what exists, then exit")
    ap.add_argument("--clean", action="store_true", help="delete fake_data/, then exit")
    args = ap.parse_args()

    if args.clean:
        clean()
        return 0
    if args.status:
        status()
        return 0

    generate(
        args.blocks,
        args.trials_per_block,
        args.trial_seconds,
        args.onset_spacing,
        args.seed,
    )
    status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
