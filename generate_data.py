#!/usr/bin/env python3
"""Generate fake recordings at the very start of the chain, so the whole
pipeline can be run locally without the real data.

What it writes is the input ``preprocessing.package_recordings`` consumes: the
intermediate ``participants_2`` table, one 1000 Hz iEEG parquet per block, and
the motion tsv/json sidecars. Preprocessing turns that into the 200 Hz trial
table, and training reads that in turn — so a local run exercises every stage in
order, on data shaped like the real recordings.

    fake_data/
    ├── preprocessing_fake.yaml                      config pointed at this tree
    └── raw/
        ├── participants_2/participant_id=FAKE1/session=1/block=<b>/0.parquet
        └── FAKE1/ses-1/
            ├── ieeg/block-<b>_ieeg.parquet          LFP_1..16, ECOG_1..4, EOG_1..4, sfreq
            └── motion/
                ├── sub-FAKE1_ses-1_task-copydraw_run-<b>_chunk-<t>_motion.tsv
                └── sub-FAKE1_ses-1_task-copydraw_run-<b>_motion.json

Usage:
    python generate_data.py                     # write the fake raw tree
    python generate_data.py --blocks 10 --trials-per-block 3
    python generate_data.py --status
    python generate_data.py --clean

Then, in the same order as on the compute host:

    python -m preprocessing.package_recordings --config fake_data/preprocessing_fake.yaml
    python training/precompute_splits.py --data-root fake_data/participants_fake_200Hz \\
        --participant FAKE1 --session 1
    python -m training.pipeline --config <run YAML pointed at that data root>

The fake participant is FAKE1, session 1, so nothing can collide with real data.
This script writes data only — it changes no code and no real config.

Local environment notes (all verified by running the chain above end to end):

* Python 3.11. The code uses `X | None` annotations, so 3.9 will not import it.
* `polars==1.34`, the version on the compute host. From roughly 1.4x polars
  names partition files `00000000.parquet` instead of `0.parquet`, and the
  trainer globs `block=*/0.parquet` — so a newer polars silently yields "no
  valid data found" at training time even though preprocessing succeeded.
* `training/pipeline.py` imports `dpad_modal` and `psid_diagnostic` at module
  scope, so running any framework locally also needs `modal`, `feature-engine`
  and `mrmr-selection` installed, even for a plain PSID run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FAKE_ROOT = PROJECT_ROOT / "fake_data"
RAW_ROOT = FAKE_ROOT / "raw"
CONFIG_PATH = FAKE_ROOT / "preprocessing_fake.yaml"
PREPROCESSED_ROOT = FAKE_ROOT / "participants_fake_200Hz"

PARTICIPANT = "FAKE1"
SESSION = 1
RAW_FS = 1000  # the real recordings are 1000 Hz before preprocessing
OUT_FS = 200  # and 200 Hz after

ECOG_CHANNELS = [f"ECOG_{i}" for i in range(1, 5)]
LFP_CHANNELS = [f"LFP_{i}" for i in range(1, 17)]
EOG_CHANNELS = [f"EOG_{i}" for i in range(1, 5)]

# Four bands instead of the real seventeen, so the generated files stay small.
BANDS = {
    "theta_4_8": (4, 8),
    "beta_18_23": (18, 23),
    "gamma_58_63": (58, 63),
    "gamma_88_93": (88, 93),
}


def log(msg: str) -> None:
    print(f"[fake-data] {msg}", flush=True)


def _latent(rng, n: int, fs: int, dbs_on: bool, n_latent: int = 4):
    """Stable rotating AR(1) latent state; DBS-ON rotates a little faster.

    Driving every channel from a shared latent gives the state-space models real
    structure to identify, rather than white noise.
    """
    import numpy as np

    theta = (0.6 if dbs_on else 0.45) * 2 * np.pi / fs
    radius = 0.999
    A = np.zeros((n_latent, n_latent))
    for i in range(n_latent // 2):
        c, s = np.cos(theta), np.sin(theta)
        A[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = radius * np.array([[c, -s], [s, c]])
    X = np.zeros((n, n_latent))
    x = rng.normal(size=n_latent)
    for t in range(n):
        x = A @ x + rng.normal(scale=0.05, size=n_latent)
        X[t] = x
    return X


def generate(
    blocks: int,
    trials_per_block: int,
    trial_seconds: float,
    gap_seconds: float,
    seed: int,
) -> None:
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(seed)
    if RAW_ROOT.exists():
        shutil.rmtree(RAW_ROOT)

    session_dir = RAW_ROOT / PARTICIPANT / f"ses-{SESSION}"
    ieeg_dir = session_dir / "ieeg"
    motion_dir = session_dir / "motion"
    ieeg_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)

    names = ECOG_CHANNELS + LFP_CHANNELS + EOG_CHANNELS
    mixing = rng.normal(size=(len(names), 4))

    n_trials = 0
    with np.errstate(all="ignore"):  # macOS BLAS raises spurious matmul warnings
        for block in range(1, blocks + 1):
            # One DBS state per block, alternating — stimulation never switches
            # mid-block in the real recordings either.
            dbs_on = block % 2 == 0

            block_seconds = trials_per_block * (trial_seconds + gap_seconds) + gap_seconds
            n_raw = int(block_seconds * RAW_FS)
            X = _latent(rng, n_raw, RAW_FS, dbs_on)
            sig = X @ mixing.T + rng.normal(scale=0.4, size=(n_raw, len(names)))
            if dbs_on:
                # Broadband gain on the cortical contacts; after band-passing it
                # becomes the beta / high-gamma envelope difference the
                # classification sweep looks for.
                for i, name in enumerate(names):
                    if name.startswith("ECOG"):
                        sig[:, i] *= 1.4
            if not np.isfinite(sig).all():
                raise RuntimeError(f"non-finite samples generated in block {block}")

            ieeg = {name: sig[:, i].astype("float32") for i, name in enumerate(names)}
            ieeg["sfreq"] = np.full(n_raw, RAW_FS, dtype="float32")
            ieeg_path = ieeg_dir / f"block-{block}_ieeg.parquet"
            pl.DataFrame(ieeg).write_parquet(ieeg_path)

            onsets = [
                float(gap_seconds + t * (trial_seconds + gap_seconds))
                for t in range(trials_per_block)
            ]
            for trial in range(1, trials_per_block + 1):
                # Pen trace: smooth random walk, integer coordinates as in the
                # real tsv files.
                n_motion = int(trial_seconds * 100)
                walk = np.cumsum(rng.normal(scale=3.0, size=(n_motion, 2)), axis=0)
                xy = (walk + 500).round().astype(int)
                tsv = motion_dir / (
                    f"sub-{PARTICIPANT}_ses-{SESSION}_task-copydraw"
                    f"_run-{block}_chunk-{trial}_motion.tsv"
                )
                tsv.write_text("\n".join(["x\ty"] + [f"{x}\t{y}" for x, y in xy]) + "\n")
                n_trials += 1

            (
                motion_dir
                / f"sub-{PARTICIPANT}_ses-{SESSION}_task-copydraw_run-{block}_motion.json"
            ).write_text(json.dumps({"dbs_stim": "on" if dbs_on else "off"}))

            part_dir = (
                RAW_ROOT
                / "participants_2"
                / f"participant_id={PARTICIPANT}"
                / f"session={SESSION}"
                / f"block={block}"
            )
            part_dir.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "participant_id": [PARTICIPANT],
                    "session": pl.Series([SESSION], dtype=pl.UInt64),
                    "block": pl.Series([block], dtype=pl.UInt64),
                    "trials": pl.Series(
                        [list(range(1, trials_per_block + 1))],
                        dtype=pl.List(pl.UInt64),
                    ),
                    "onsets": [onsets],
                    # Per-trial duration in seconds. _chunk_recordings uses it to
                    # size each trial window, so it has to be one value per row.
                    "trial_time": [float(trial_seconds)],
                    "ieeg_parquet": [str(ieeg_path)],
                    "session_path": [str(session_dir)],
                    "stim": ["on" if dbs_on else "off"],
                    "is_fragmented": [False],
                }
            ).write_parquet(part_dir / "0.parquet")

    write_config()
    log(
        f"{blocks} blocks x {trials_per_block} trials ({n_trials} total) at {RAW_FS} Hz "
        f"-> {RAW_ROOT.relative_to(PROJECT_ROOT)}"
    )
    log(f"preprocessing config -> {CONFIG_PATH.relative_to(PROJECT_ROOT)}")


def write_config() -> Path:
    """Preprocessing YAML pointed at the fake tree.

    The shipped configs carry absolute paths on the compute host, so a local run
    needs its own. Four bands instead of seventeen; sampling rate, margins, CAR
    and notches match the real `raw_envelope` config.
    """
    import yaml

    FAKE_ROOT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "name": "package_recordings_fake",
        "root_directory": str(PROJECT_ROOT),
        "data_directory": str(RAW_ROOT),
        "save_directory": str(FAKE_ROOT),
        "logger_directory": "{root_directory}/logs",
        "participants_table_name": "participants.tsv",
        "participants_intermediate_table_name": "participants_2",
        "output_participants_table_name": PREPROCESSED_ROOT.name,
        "ieeg_process": {
            "resampled_dir": "{save_directory}/resampled_mat_recordings",
            "chunk_margin": 2,
            "resampled_freq": OUT_FS,
            "notch_freqs": [50, 100],
            "scale_factor": 1.0,
            "apply_car": True,
            "drop_lfp": False,
            "max_pause_seconds": 2.0,
            "raw_bands": {f"{b}_raw": list(r) for b, r in BANDS.items()},
            "envelope_bands": {f"{b}_env": list(r) for b, r in BANDS.items()},
        },
    }
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return CONFIG_PATH


def status() -> None:
    def count(root: Path, pattern: str) -> int:
        return len(list(root.glob(pattern))) if root.exists() else 0

    print()
    log("on disk:")
    print(
        f"  raw            {RAW_ROOT.relative_to(PROJECT_ROOT)}  "
        f"({count(RAW_ROOT, 'participants_2/**/block=*')} blocks, "
        f"{count(RAW_ROOT, '**/ieeg/*.parquet')} iEEG parquets, "
        f"{count(RAW_ROOT, '**/motion/*.tsv')} motion files)"
    )
    print(
        f"  config         {CONFIG_PATH.relative_to(PROJECT_ROOT)}  "
        f"{'present' if CONFIG_PATH.exists() else '-'}"
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
        description="Generate fake raw recordings for a local end-to-end pipeline run."
    )
    ap.add_argument("--blocks", type=int, default=10, help="blocks, alternating DBS on/off")
    ap.add_argument("--trials-per-block", type=int, default=3)
    ap.add_argument("--trial-seconds", type=float, default=6.0)
    ap.add_argument("--gap-seconds", type=float, default=3.0, help="pause between trials")
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
        args.gap_seconds,
        args.seed,
    )
    status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
