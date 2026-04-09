#!/usr/bin/env python3
"""
DPAD pipeline for PDI4 Session 3 (200 Hz narrow-band data).
Equivalent to PSID pipeline but using deep learning (DPAD).
Reuses PSID data splits and latent dimensions (no grid search).

Phases:
  1. Train models (dbs_both/on/off) — reusing PSID splits
  2. Cross-condition eval (on→off, off→on)
  3. Classification (2-step: grid search → permutation test)
  4. Update specs.py + regenerate thesis HTML

Usage:
    python scripts/pipeline_dpad_PDI4_S3.py 2>&1 | tee dpad_PDI4_S3.log
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

import yaml

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH = PROJECT_ROOT / "dpad_PDI4_S3.log"
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
SESSION = "3"
DATA_ROOT = "resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band"
SAMPLING_FREQ = 200
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3

# DPAD-specific
EPOCHS = 3000
METHOD_CODE = "DPAD_uAKCzCy2HL32U"
CHECKPOINT_EVERY = 100

# Latent dimensions — reuse from PSID best (no grid search)
NX = 25
N1 = 6

# PSID variant to copy splits from
PSID_I = 50
PSID_BOTH_VARIANT = f"psid_behavioral_{PARTICIPANT}_{SESSION}_nx_{NX}_n{N1}_i{PSID_I}_dbs_both_200Hz_narrow_band"

BANDS = [
    "theta_4_8_raw", "alpha_8_12_raw",
    "beta_12_17_raw", "beta_17_22_raw", "beta_22_27_raw", "beta_27_30_raw",
    "gamma_30_35_raw", "gamma_35_40_raw", "gamma_40_45_raw", "gamma_45_50_raw",
    "gamma_50_55_raw", "gamma_55_60_raw", "gamma_60_65_raw", "gamma_65_70_raw",
    "gamma_75_80_raw",
]
CHANNELS = [f"ECOG_{e}_{b}" for e in range(1, 5) for b in BANDS]
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


def _data_block(dbs_condition: str = "both"):
    return {
        "root": DATA_ROOT,
        "participant": PARTICIPANT,
        "session": SESSION,
        "blocks": "all",
        "channels": {
            "neural_input": CHANNELS,
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


def run_cmd(args: list[str], label: str, timeout: int = 14400) -> subprocess.CompletedProcess:
    log.info(f"  RUN: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-10:]:
            log.info(f"    {line}")
    if result.returncode != 0:
        log.error(f"  FAILED ({label}): {result.stderr[-500:] if result.stderr else 'no stderr'}")
    return result


def get_latest_model_ts(variant_dir: Path) -> str:
    """Get the latest model timestamp from a results variant directory."""
    pkls = sorted(variant_dir.glob("model_*.pkl"))
    pkls = [p for p in pkls if "_metadata" not in p.name]
    if not pkls:
        raise FileNotFoundError(f"No model_*.pkl in {variant_dir}")
    return pkls[-1].stem.replace("model_", "")


def variant_name(dbs: str) -> str:
    return f"dpad_behavioral_{PARTICIPANT}_{SESSION}_nx_{NX}_n{N1}_e{EPOCHS}_dbs_{dbs}_200Hz_narrow_band"


def variant_config_path(dbs: str) -> Path:
    subdir = "both" if dbs == "both" else dbs
    return Path(f"training/setups/dpad/narrow_band_200Hz/{subdir}/{variant_name(dbs)}.yaml")


def _copy_splits_from_psid(dbs: str):
    """Copy train/val/test splits from PSID results to DPAD results dir."""
    psid_var = f"psid_behavioral_{PARTICIPANT}_{SESSION}_nx_{NX}_n{N1}_i{PSID_I}_dbs_{dbs}_200Hz_narrow_band"
    psid_split = PROJECT_ROOT / "results" / psid_var / "split"
    dpad_split = PROJECT_ROOT / "results" / variant_name(dbs) / "split"

    if not psid_split.exists():
        log.warning("  PSID split dir not found: %s", psid_split)
        return False

    dpad_split.mkdir(parents=True, exist_ok=True)
    for fname in ("train.parquet", "val.parquet", "test.parquet"):
        src = psid_split / fname
        dst = dpad_split / fname
        if src.exists():
            shutil.copy2(src, dst)
    log.info("  Copied splits from %s → %s", psid_split, dpad_split)
    return True


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Train models (3 DBS conditions)
# ═════════════════════════════════════════════════════════════════════════════

def _make_training_config(dbs: str) -> Path:
    name = variant_name(dbs)
    cfg = {
        "model": {
            "name": name,
            "nx": NX,
            "n1": N1,
            "method_code": METHOD_CODE,
            "epochs": EPOCHS,
            "fast": False,
            "reuse_splits": True,
            "checkpoint_every": CHECKPOINT_EVERY,
            "steps_ahead": [1],
            "steps_ahead_loss_weights": [1.0],
            "forecast": {"m": 2, "history": 5},
        },
        "data": _data_block(dbs),
        "results": {
            "save_dir": f"results/{name}",
            "log_dir": f"results/{name}/logs",
            "checkpoint_dir": "checkpoints",
        },
    }
    return write_yaml(variant_config_path(dbs), cfg)


def phase1_train() -> dict[str, str]:
    """Train 3 variants (dbs_both, dbs_on, dbs_off).
    Returns {dbs_condition: run_timestamp}."""
    log.info("=" * 60)
    log.info("PHASE 1: Training  nx=%d, n1=%d, epochs=%d", NX, N1, EPOCHS)
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
            log.info("  SKIP training %s — model already exists (ts=%s)", name, ts)
            timestamps[dbs] = ts
            continue

        # Copy splits from PSID
        _copy_splits_from_psid(dbs)

        cfg = _make_training_config(dbs)
        log.info("  Training %s (epochs=%d) …", name, EPOCHS)
        result = run_cmd(
            [sys.executable, "-m", "training.train", "--config", str(cfg)],
            label=f"train_{dbs}",
            timeout=86400,  # 24h — DPAD can be slow
        )

        model_files = [p for p in (PROJECT_ROOT / "results" / name).glob("model_*.pkl")
                       if "_metadata" not in p.name]
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
            [sys.executable, "-m", "training.test", "--config", str(cfg), "--run", ts],
            label=f"test_{dbs}",
            timeout=7200,
        )

    return timestamps


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Cross-condition eval
# ═════════════════════════════════════════════════════════════════════════════

def phase2_cross_eval(timestamps: dict[str, str]):
    log.info("=" * 60)
    log.info("PHASE 2: Cross-condition evaluation")
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
# PHASE 3 — Classification
# ═════════════════════════════════════════════════════════════════════════════

def _find_best_hm(results_dir: Path, run_ts: str, feat_label: str) -> tuple[float, float, float]:
    """Find the best h/m pair by balanced accuracy from forecast pickles."""
    best_ba, best_h, best_m = -1.0, 0.5, 0.5
    for hm_dir in sorted((results_dir / run_ts).glob("h*_m*")):
        pkl = hm_dir / f"LDA_{feat_label}_forecast.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        ba = d.get("balanced_accuracy", 0.0)
        parts = hm_dir.name.split("_")
        h = float(parts[0][1:])
        m = float(parts[1][1:])
        if ba > best_ba:
            best_ba, best_h, best_m = ba, h, m
    return best_h, best_m, best_ba


def phase3_classification(timestamps: dict[str, str]):
    log.info("=" * 60)
    log.info("PHASE 3: Classification (2-step: grid search → permutation test)")
    log.info("=" * 60)

    both_var = variant_name("both")
    on_var = variant_name("on")
    off_var = variant_name("off")
    ts_both, ts_on, ts_off = timestamps["both"], timestamps["on"], timestamps["off"]

    H_GRID = [0.5, 1.0, 1.5, 2.5, 4.5]
    M_GRID = [0.5, 1.0, 2.0]

    base_cls = {
        "n_splits": 5,
        "epoch_length": 0.5,
        "epoch_overlap": 0.25,
        "sampling_freq": SAMPLING_FREQ,
        "h": H_GRID,
        "m": M_GRID,
        "n1": N1,
        "nx": NX,
        "permutation_test": False,
        "n_permutations": 0,
        "param_grid": {
            "LDA": {
                "classifier__solver": ["lsqr"],
                "classifier__shrinkage": ["auto"],
            }
        },
    }

    feature_sources = [("", "Xp"), ("_xp_1", "Xp_1"),
                       ("_xp_2", "Xp_2"), ("_xp_with_dbs", "Xp_with_dbs")]

    # ── Step 1: Run all h×m combos WITHOUT permutation test ──
    log.info("  Step 1: Classification grid search (no permutations)")
    configs_step1: list[Path] = []

    for suffix, feat in feature_sources:
        cfg = {
            "run": {"variant": both_var, "run_ts": ts_both},
            "classification": {
                **base_cls,
                "flipped": False,
                "prediction_feature_source": feat,
                "forecast_feature_source": feat,
            },
            "results": {
                "project_root": str(PROJECT_ROOT),
                "results_dir": f"{PROJECT_ROOT}/results/classification/{both_var}",
                "log_dir": f"{PROJECT_ROOT}/logs/classification",
            },
        }
        p = write_yaml(f"classification/setups/{both_var}{suffix}.yaml", cfg)
        configs_step1.append(p)

    # Flipped: Xp (also without permutations)
    flipped_cfg = {
        "name": f"{both_var}_flipped",
        "run": {
            "dbs_on":   {"variant": on_var,   "run_ts": ts_on},
            "dbs_off":  {"variant": off_var,  "run_ts": ts_off},
            "dbs_both": {"variant": both_var, "run_ts": ts_both},
        },
        "classification": {
            **base_cls,
            "flipped": True,
            "h": H_GRID,
            "m": M_GRID,
            "prediction_feature_source": "Xp",
            "forecast_feature_source": "Xp",
        },
        "results": {
            "project_root": str(PROJECT_ROOT),
            "results_dir": f"{PROJECT_ROOT}/results/classification/{both_var}_flipped",
            "log_dir": f"{PROJECT_ROOT}/logs/classification",
        },
    }
    p = write_yaml(f"classification/setups/{both_var}_flipped.yaml", flipped_cfg)
    configs_step1.append(p)

    for cfg_path in configs_step1:
        log.info("  Classifying %s …", cfg_path.stem)
        run_cmd(
            [sys.executable, "classification/compute.py", "--config", str(cfg_path)],
            label=f"cls_{cfg_path.stem}",
            timeout=3600,
        )

    # ── Step 2: Find best h/m per feature source, rerun with permutations ──
    log.info("  Step 2: Permutation tests on best h/m per feature source")
    cls_base = PROJECT_ROOT / "results" / "classification" / both_var

    for suffix, feat in feature_sources:
        best_h, best_m, best_ba = _find_best_hm(cls_base, ts_both, feat)
        log.info("    %s best: h=%.1f m=%.1f (bal_acc=%.4f)", feat, best_h, best_m, best_ba)

        cfg = {
            "run": {"variant": both_var, "run_ts": ts_both},
            "classification": {
                **base_cls,
                "flipped": False,
                "h": [best_h],
                "m": [best_m],
                "prediction_feature_source": feat,
                "forecast_feature_source": feat,
                "permutation_test": True,
                "n_permutations": 100,
            },
            "results": {
                "project_root": str(PROJECT_ROOT),
                "results_dir": f"{PROJECT_ROOT}/results/classification/{both_var}",
                "log_dir": f"{PROJECT_ROOT}/logs/classification",
            },
        }
        p = write_yaml(f"classification/setups/{both_var}{suffix}_perm.yaml", cfg)
        log.info("    Running permutation test for %s (h=%.1f, m=%.1f) …", feat, best_h, best_m)
        run_cmd(
            [sys.executable, "classification/compute.py", "--config", str(p)],
            label=f"cls_perm_{feat}",
            timeout=3600,
        )

    # Flipped permutation test
    flipped_cls_base = PROJECT_ROOT / "results" / "classification" / f"{both_var}_flipped"
    best_h, best_m, best_ba = -1.0, 0.5, 0.5
    flipped_ts_dir = None
    for ts_dir in sorted(flipped_cls_base.glob("*")):
        if ts_dir.is_dir():
            flipped_ts_dir = ts_dir.name
            for hm_dir in sorted(ts_dir.glob("h*_m*")):
                pkl = hm_dir / "LDA_Xp_flipped.pkl"
                if not pkl.exists():
                    continue
                with open(pkl, "rb") as f:
                    d = pickle.load(f)
                ba = d.get("balanced_accuracy", 0.0)
                parts = hm_dir.name.split("_")
                h = float(parts[0][1:])
                m = float(parts[1][1:])
                if ba > best_ba:
                    best_ba, best_h, best_m = ba, h, m

    if flipped_ts_dir:
        log.info("    flipped best: h=%.1f m=%.1f (bal_acc=%.4f)", best_h, best_m, best_ba)
        flipped_perm_cfg = {
            "name": f"{both_var}_flipped",
            "run": {
                "dbs_on":   {"variant": on_var,   "run_ts": ts_on},
                "dbs_off":  {"variant": off_var,  "run_ts": ts_off},
                "dbs_both": {"variant": both_var, "run_ts": ts_both},
            },
            "classification": {
                **base_cls,
                "flipped": True,
                "h": [best_h],
                "m": [best_m],
                "prediction_feature_source": "Xp",
                "forecast_feature_source": "Xp",
                "permutation_test": True,
                "n_permutations": 100,
            },
            "results": {
                "project_root": str(PROJECT_ROOT),
                "results_dir": f"{PROJECT_ROOT}/results/classification/{both_var}_flipped",
                "log_dir": f"{PROJECT_ROOT}/logs/classification",
            },
        }
        p = write_yaml(f"classification/setups/{both_var}_flipped_perm.yaml", flipped_perm_cfg)
        log.info("    Running permutation test for flipped (h=%.1f, m=%.1f) …", best_h, best_m)
        run_cmd(
            [sys.executable, "classification/compute.py", "--config", str(p)],
            label="cls_perm_flipped",
            timeout=3600,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Update specs.py & generate thesis HTML
# ═════════════════════════════════════════════════════════════════════════════

def phase4_thesis(timestamps: dict[str, str]):
    log.info("=" * 60)
    log.info("PHASE 4: Update specs.py & generate thesis HTML")
    log.info("=" * 60)

    new_variant = variant_name("both")
    ts_both, ts_on, ts_off = timestamps["both"], timestamps["on"], timestamps["off"]

    specs_path = PROJECT_ROOT / "dashboard" / "thesis" / "specs.py"
    text = specs_path.read_text()

    # ── Helper to update one triplet block ──
    def update_triplet(text: str, triplet_name: str) -> str:
        # dpad_variant
        text = re.sub(
            rf'({triplet_name}\s*=\s*AlignedTriplet\([^)]*?dpad_variant=")[^"]*(")',
            rf'\g<1>{new_variant}\2', text, flags=re.DOTALL,
        )
        # dpad_run_ts
        text = re.sub(
            rf'({triplet_name}\s*=\s*AlignedTriplet\([^)]*?dpad_run_ts=")[^"]*(")',
            rf'\g<1>{ts_both}\2', text, flags=re.DOTALL,
        )
        text = re.sub(
            rf'({triplet_name}\s*=\s*AlignedTriplet\([^)]*?dpad_run_ts_off=")[^"]*(")',
            rf'\g<1>{ts_off}\2', text, flags=re.DOTALL,
        )
        text = re.sub(
            rf'({triplet_name}\s*=\s*AlignedTriplet\([^)]*?dpad_run_ts_on=")[^"]*(")',
            rf'\g<1>{ts_on}\2', text, flags=re.DOTALL,
        )
        return text

    text = update_triplet(text, "_TRIPLET_PDI4_S3")
    text = update_triplet(text, "_NEURAL_BAND_TRIPLET_PDI4_S3")

    specs_path.write_text(text)
    log.info("Updated specs.py → dpad_variant=%s", new_variant)

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
                        help="Phase to start from (1-4). Phases before this are skipped.")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("DPAD pipeline — %s S%s, 200 Hz narrow band", PARTICIPANT, SESSION)
    log.info("  nx=%d, n1=%d, epochs=%d, method=%s", NX, N1, EPOCHS, METHOD_CODE)
    log.info("  Splits from PSID variant: %s", PSID_BOTH_VARIANT)
    log.info("  Project root: %s", PROJECT_ROOT)
    if args.start_phase > 1:
        log.info("Resuming from Phase %d", args.start_phase)
    log.info("=" * 60)

    # Ensure directories exist
    for d in ["training/setups/dpad/narrow_band_200Hz/both",
              "training/setups/dpad/narrow_band_200Hz/on",
              "training/setups/dpad/narrow_band_200Hz/off",
              "classification/setups",
              "logs/classification"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    if args.start_phase <= 1:
        timestamps = phase1_train()
    else:
        timestamps = {}
        for dbs in ("both", "on", "off"):
            name = variant_name(dbs)
            ts = get_latest_model_ts(PROJECT_ROOT / "results" / name)
            timestamps[dbs] = ts

    if args.start_phase <= 2:
        phase2_cross_eval(timestamps)
    if args.start_phase <= 3:
        phase3_classification(timestamps)
    if args.start_phase <= 4:
        phase4_thesis(timestamps)

    log.info("")
    log.info("=" * 60)
    log.info("DPAD PIPELINE COMPLETE")
    log.info("  Timestamps: both=%s  on=%s  off=%s",
             timestamps["both"], timestamps["on"], timestamps["off"])
    log.info("  Thesis HTML: thesis_results.html")
    log.info("  Full log: %s", LOG_PATH)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
