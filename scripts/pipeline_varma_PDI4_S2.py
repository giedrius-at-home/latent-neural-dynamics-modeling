#!/usr/bin/env python3
"""
VARMA pipeline for PDI4 Session 3 (200 Hz narrow-band data).

Uses PSID channel importance to select top-N neural channels for VARMA.
Reuses train/val/test splits from the PSID run.

Phases:
  1. Extract channel importance from trained PSID model (Cy, eigenvalues)
  2. Train VARMA with selected channels (dbs_both/on/off)
  3. Cross-condition eval (on→off, off→on)
  4. Update specs.py + regenerate thesis HTML

Usage:
    python scripts/pipeline_varma_PDI4_S2.py 2>&1 | tee varma_PDI4_S2.log
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH = PROJECT_ROOT / "varma_PDI4_S2.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger("pipeline")

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════
PARTICIPANT = "PDI4"
SESSION = "2"
DATA_ROOT = "resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band"
SAMPLING_FREQ = 200
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3

# VARMA hyperparameters
P = 30          # AR lags
Q = 1           # MA lags
LONG_AR_LAGS = 30

# Channel selection
TOP_N_CHANNELS = 20  # top channels from PSID importance ranking
IMPORTANCE_METHOD = "weighted"  # eigenvalue-weighted ||Cy @ diag(|lambda|)||

# PSID model to extract channel importance from
PSID_NX = 30
PSID_N1 = 6
PSID_I = 50
PSID_BOTH_VARIANT = f"psid_behavioral_{PARTICIPANT}_{SESSION}_nx_{PSID_NX}_n{PSID_N1}_i{PSID_I}_dbs_both_200Hz_narrow_band"

BANDS = [
    "theta_4_8_raw", "alpha_8_12_raw",
    "beta_12_17_raw", "beta_17_22_raw", "beta_22_27_raw", "beta_27_30_raw",
    "gamma_30_35_raw", "gamma_35_40_raw", "gamma_40_45_raw", "gamma_45_50_raw",
    "gamma_50_55_raw", "gamma_55_60_raw", "gamma_60_65_raw", "gamma_65_70_raw",
    "gamma_75_80_raw",
]
ALL_CHANNELS = [f"ECOG_{e}_{b}" for e in range(1, 5) for b in BANDS]
BEHAVIORAL_OUTPUTS = ["tracing_velocity_x", "tracing_acceleration_magnitude"]

# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _split_block():
    return {
        "within_session_split": True,
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
        "min_train_epochs": 50,
    }


def _data_block(dbs_condition: str, channels: list[str]):
    return {
        "root": DATA_ROOT,
        "participant": PARTICIPANT,
        "session": SESSION,
        "blocks": "all",
        "channels": {
            "neural_input": channels,
            "output_type": "behavioral",
            "output": BEHAVIORAL_OUTPUTS,
        },
        "split": _split_block(),
        "batch_size": 32,
        "sampling_frequency": SAMPLING_FREQ,
        "dbs_condition": dbs_condition,
    }


def write_yaml(path: str | Path, data: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return p


def run_cmd(args: list[str], label: str, timeout: int = 7200) -> subprocess.CompletedProcess:
    log.info(f"  RUN: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-10:]:
            log.info(f"    {line}")
    if result.returncode != 0:
        log.error(f"  FAILED ({label}): {result.stderr[-500:] if result.stderr else 'no stderr'}")
    return result


def get_latest_model_ts(variant_dir: Path) -> str:
    pkls = sorted(variant_dir.glob("model_*.pkl"))
    pkls = [p for p in pkls if "_metadata" not in p.name]
    if not pkls:
        raise FileNotFoundError(f"No model_*.pkl in {variant_dir}")
    return pkls[-1].stem.replace("model_", "")


def variant_name(dbs: str, top_n: int = TOP_N_CHANNELS) -> str:
    return f"varma_behavioral_{PARTICIPANT}_{SESSION}_p{P}_q{Q}_top{top_n}_dbs_{dbs}_200Hz_narrow_band"


def variant_config_path(dbs: str, top_n: int = TOP_N_CHANNELS) -> Path:
    name = variant_name(dbs, top_n)
    subdir = "both" if dbs == "both" else dbs
    return Path(f"training/setups/varma/narrow_band_200Hz/{subdir}/{name}.yaml")


def _copy_splits_from_psid(dbs: str, selected_channels: list[str]):
    """Copy splits from PSID results dir, then re-split to match VARMA channel set."""
    psid_variant = f"psid_behavioral_{PARTICIPANT}_{SESSION}_nx_{PSID_NX}_n{PSID_N1}_i{PSID_I}_dbs_{dbs}_200Hz_narrow_band"
    if dbs != "both":
        # on/off don't have their own splits — they use the 'both' split filtered by stim
        psid_variant = PSID_BOTH_VARIANT
    psid_split_dir = PROJECT_ROOT / "results" / psid_variant / "split"
    varma_var = variant_name(dbs)
    varma_split_dir = PROJECT_ROOT / "results" / varma_var / "split"

    if varma_split_dir.exists() and (varma_split_dir / "train.parquet").exists():
        log.info("  Splits already exist for %s — skipping copy", varma_var)
        return True

    if not psid_split_dir.exists():
        log.warning("  PSID splits not found at %s", psid_split_dir)
        return False

    varma_split_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("train.parquet", "val.parquet", "test.parquet"):
        src = psid_split_dir / fname
        dst = varma_split_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            log.info("  Copied %s → %s", src, dst)
    return True


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Extract channel importance from PSID
# ═════════════════════════════════════════════════════════════════════════════

def phase1_channel_selection() -> list[str]:
    log.info("=" * 60)
    log.info("PHASE 1: Extract channel importance from PSID model")
    log.info("=" * 60)

    from scripts.extract_psid_channel_importance import (
        get_top_behavioral_and_neural,
    )

    psid_dir = PROJECT_ROOT / "results" / PSID_BOTH_VARIANT
    model_ts = get_latest_model_ts(psid_dir)
    model_path = psid_dir / f"model_{model_ts}.pkl"

    log.info("  PSID model: %s", model_path)
    log.info("  n1=%d, top_n=%d", PSID_N1, TOP_N_CHANNELS)

    result = get_top_behavioral_and_neural(
        model_path, n1=PSID_N1, channel_names=ALL_CHANNELS, top_n=TOP_N_CHANNELS,
    )
    imp = result["importance"]
    br = result["behavioral_relevance"]

    # Log eigenvalue info
    log.info("  A spectral radius: %.4f", np.max(np.abs(imp["eigenvalues"])))
    log.info("  A11 spectral radius: %.4f", np.max(np.abs(imp["eig_x1"])))
    log.info("  A22 spectral radius: %.4f", np.max(np.abs(imp["eig_x2"])))

    # Log top channels for both criteria
    log.info("  Top %d channels — BEHAVIORAL (||Cz @ Cy[i,:]||):", TOP_N_CHANNELS)
    for i, (ch, sc) in enumerate(zip(result["top_behavioral"], result["top_behavioral_scores"])):
        log.info("    %3d. %-40s  score=%.6f", i + 1, ch, sc)

    log.info("  Top %d channels — NEURAL (eigenvalue-weighted ||Cy||):", TOP_N_CHANNELS)
    for i, (ch, sc) in enumerate(zip(result["top_neural"], result["top_neural_scores"])):
        log.info("    %3d. %-40s  score=%.6f", i + 1, ch, sc)

    # Union of both sets (no duplicates, preserve order from behavioral first)
    seen = set()
    selected_channels = []
    for ch in result["top_behavioral"] + result["top_neural"]:
        if ch not in seen:
            seen.add(ch)
            selected_channels.append(ch)
    log.info("  Combined unique channels: %d (from %d behavioral + %d neural)",
             len(selected_channels), TOP_N_CHANNELS, TOP_N_CHANNELS)
    for i, ch in enumerate(selected_channels):
        log.info("    %3d. %s", i + 1, ch)

    # Save channel selection to JSON for reproducibility
    selection_path = PROJECT_ROOT / "results" / f"varma_channel_selection_{PARTICIPANT}_{SESSION}.json"
    selection_data = {
        "psid_model": str(model_path),
        "psid_variant": PSID_BOTH_VARIANT,
        "top_n_per_criterion": TOP_N_CHANNELS,
        "top_behavioral": result["top_behavioral"],
        "top_behavioral_scores": result["top_behavioral_scores"],
        "top_neural": result["top_neural"],
        "top_neural_scores": result["top_neural_scores"],
        "selected_channels": selected_channels,
        "all_rankings": {
            "behavioral": [ALL_CHANNELS[i] for i in br["rank_behavioral"]],
            "behavioral_x1": [ALL_CHANNELS[i] for i in br["rank_behavioral_x1"]],
            "weighted": [ALL_CHANNELS[i] for i in imp["rank_weighted"]],
            "raw": [ALL_CHANNELS[i] for i in imp["rank_raw"]],
        },
    }
    with open(selection_path, "w") as f:
        json.dump(selection_data, f, indent=2)
    log.info("  Saved channel selection → %s", selection_path)

    return selected_channels


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Train VARMA (3 DBS conditions)
# ═════════════════════════════════════════════════════════════════════════════

def _make_varma_config(dbs: str, channels: list[str]) -> Path:
    name = variant_name(dbs)
    cfg = {
        "model": {
            "name": name,
            "p": P,
            "q": Q,
            "long_ar_lags": LONG_AR_LAGS,
            "trial_edge_taper_sec": 0.1,
            "time_first": True,
            "fast": False,
            "reuse_splits": True,
            "forecast": {"m": 2, "history": 5},
        },
        "data": _data_block(dbs, channels),
        "results": {
            "save_dir": f"results/{name}",
            "log_dir": f"results/{name}/logs",
            "checkpoint_dir": "checkpoints",
        },
    }
    return write_yaml(variant_config_path(dbs), cfg)


def phase2_train(channels: list[str]) -> dict[str, str]:
    log.info("=" * 60)
    log.info("PHASE 2: Train VARMA (p=%d, q=%d) with %d channels", P, Q, len(channels))
    log.info("=" * 60)

    timestamps: dict[str, str] = {}

    for dbs in ("both", "on", "off"):
        name = variant_name(dbs)
        result_dir = PROJECT_ROOT / "results" / name

        # Check if model already exists
        existing_models = [p for p in result_dir.glob("model_*.pkl")
                           if "_metadata" not in p.name] if result_dir.exists() else []
        if existing_models:
            ts = sorted(existing_models)[-1].stem.replace("model_", "")
            log.info("  SKIP training %s — model exists (ts=%s)", name, ts)
            timestamps[dbs] = ts
            continue

        # Copy splits from PSID
        _copy_splits_from_psid(dbs, channels)

        cfg = _make_varma_config(dbs, channels)
        log.info("  Training %s …", name)
        result = run_cmd(
            [sys.executable, "-m", "training.train", "--config", str(cfg)],
            label=f"train_{dbs}",
            timeout=7200,
        )
        model_files = list((PROJECT_ROOT / "results" / name).glob("model_*.pkl"))
        model_files = [p for p in model_files if "_metadata" not in p.name]
        if model_files:
            ts = sorted(model_files)[-1].stem.replace("model_", "")
            log.info("    → ts=%s", ts)
            timestamps[dbs] = ts
        else:
            log.error("  Training FAILED for %s!", name)
            sys.exit(1)

        # Run tester to generate test-split results
        log.info("  Testing %s …", name)
        run_cmd(
            [sys.executable, "-m", "training.test", "--config", str(cfg), "--run", timestamps[dbs]],
            label=f"test_{dbs}",
            timeout=7200,
        )

    return timestamps


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Cross-condition eval
# ═════════════════════════════════════════════════════════════════════════════

def phase3_cross_eval(timestamps: dict[str, str]):
    log.info("=" * 60)
    log.info("PHASE 3: Cross-condition evaluation")
    log.info("=" * 60)

    on_cfg = variant_config_path("on")
    off_cfg = variant_config_path("off")

    log.info("  on → eval_off …")
    run_cmd(
        [sys.executable, "scripts/run_cross_condition_eval.py",
         "--config", str(on_cfg), "--target", "off", "--run", timestamps["on"]],
        label="cross_eval_on_off",
    )

    log.info("  off → eval_on …")
    run_cmd(
        [sys.executable, "scripts/run_cross_condition_eval.py",
         "--config", str(off_cfg), "--target", "on", "--run", timestamps["off"]],
        label="cross_eval_off_on",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Update specs.py & generate thesis HTML
# ═════════════════════════════════════════════════════════════════════════════

def phase4_thesis(timestamps: dict[str, str]):
    log.info("=" * 60)
    log.info("PHASE 5: Update specs.py & generate thesis HTML")
    log.info("=" * 60)

    both_var = variant_name("both")
    ts_both = timestamps["both"]

    specs_path = PROJECT_ROOT / "dashboard" / "thesis" / "specs.py"
    text = specs_path.read_text()

    # Update VARMA variant in the triplet for PDI4 S3
    text = re.sub(
        r'(_TRIPLET_PDI4_S3\s*=\s*AlignedTriplet\([^)]*?varma_variant=")[^"]*(")',
        rf'\g<1>{both_var}\2', text, flags=re.DOTALL,
    )
    text = re.sub(
        r'(_TRIPLET_PDI4_S3\s*=\s*AlignedTriplet\([^)]*?varma_run_ts=")[^"]*(")',
        rf'\g<1>{ts_both}\2', text, flags=re.DOTALL,
    )

    specs_path.write_text(text)
    log.info("Updated specs.py → varma_variant=%s", both_var)

    log.info("Generating thesis HTML …")
    run_cmd(
        [sys.executable, "scripts/generate_thesis_html.py"],
        label="thesis_html",
        timeout=600,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-phase", type=int, default=1,
                        help="Phase to start from (1-4)")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Override number of top channels to select")
    args = parser.parse_args()

    global TOP_N_CHANNELS
    if args.top_n is not None:
        TOP_N_CHANNELS = args.top_n

    log.info("=" * 60)
    log.info("VARMA pipeline — %s S%s, 200 Hz narrow band", PARTICIPANT, SESSION)
    log.info("  p=%d, q=%d, top_n=%d channels (from PSID %s)", P, Q, TOP_N_CHANNELS, IMPORTANCE_METHOD)
    log.info("  PSID source: %s", PSID_BOTH_VARIANT)
    log.info("  Project root: %s", PROJECT_ROOT)
    if args.start_phase > 1:
        log.info("  Resuming from Phase %d", args.start_phase)
    log.info("=" * 60)

    # Ensure directories exist
    for d in ["training/setups/varma/narrow_band_200Hz/both",
              "training/setups/varma/narrow_band_200Hz/on",
              "training/setups/varma/narrow_band_200Hz/off",
              "classification/setups",
              "logs/classification"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    if args.start_phase <= 1:
        selected_channels = phase1_channel_selection()
    else:
        # Load from saved selection
        selection_path = PROJECT_ROOT / "results" / f"varma_channel_selection_{PARTICIPANT}_{SESSION}.json"
        with open(selection_path) as f:
            sel = json.load(f)
        selected_channels = sel["selected_channels"]
        log.info("Loaded %d channels from %s", len(selected_channels), selection_path)

    if args.start_phase <= 2:
        timestamps = phase2_train(selected_channels)
    else:
        timestamps = {}
        for dbs in ("both", "on", "off"):
            name = variant_name(dbs)
            ts = get_latest_model_ts(PROJECT_ROOT / "results" / name)
            timestamps[dbs] = ts

    if args.start_phase <= 3:
        phase3_cross_eval(timestamps)
    if args.start_phase <= 4:
        phase4_thesis(timestamps)

    log.info("")
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("  Selected channels: %d (method=%s)", len(selected_channels), IMPORTANCE_METHOD)
    log.info("  Timestamps: both=%s  on=%s  off=%s",
             timestamps["both"], timestamps["on"], timestamps["off"])
    log.info("  Thesis HTML: thesis_results.html")
    log.info("  Full log: %s", LOG_PATH)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
