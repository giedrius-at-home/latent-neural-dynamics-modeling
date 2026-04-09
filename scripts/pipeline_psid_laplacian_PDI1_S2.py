#!/usr/bin/env python3
"""
PSID pipeline for PDI1 Session 2 — Laplacian LFP (200 Hz narrow-band).
ECoG input → LAPLACIAN_14-16 output (neural-to-neural PSID).

Phases:
  1. Grid search over nx × n1 (dbs_both, non-vanilla)
  2. Classify each grid-search run → pick best nx/n1 by balanced accuracy
  3. Train full models (dbs_both/on/off non-vanilla + vanilla dbs_both)
  4. Cross-condition eval (on→off, off→on)
  5. Classification (2-step: grid search → permutation test)
  6. Update specs.py + regenerate thesis HTML

Usage:
    python scripts/pipeline_psid_laplacian_PDI1_S2.py 2>&1 | tee psid_lap_PDI1_S2.log
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
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

# ── Resolve project root ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_PATH = PROJECT_ROOT / "psid_lap_PDI1_S2.log"
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
PARTICIPANT = "PDI1"
SESSION = "2"
DATA_ROOT = "resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band"
SAMPLING_FREQ = 200
TRAIN_RATIO = 0.6
VAL_RATIO = 0.1
TEST_RATIO = 0.3

GS_I = 30           # grid-search PSID iterations
FULL_I = 50          # full-training PSID iterations
MAX_EIGENVALUE = 0.999
GS_WORKERS = 1

# Higher dimensions for neural-to-neural PSID
NX_GRID = [15, 25, 40, 55, 65, 80]
N1_GRID = [4, 8, 12]

GS_BASE = f"psid_gs_lap_{PARTICIPANT}_S{SESSION}_200Hz_narrow_band"

# Laplacian pair to predict (last pair)
LAP_PAIR = "LAPLACIAN_14-16"

# ECoG input channels (same as behavioral pipeline)
BANDS = [
    "theta_4_8_raw", "alpha_8_12_raw",
    "beta_12_17_raw", "beta_17_22_raw", "beta_22_27_raw", "beta_27_30_raw",
    "gamma_30_35_raw", "gamma_35_40_raw", "gamma_40_45_raw", "gamma_45_50_raw",
    "gamma_50_55_raw", "gamma_55_60_raw", "gamma_60_65_raw", "gamma_65_70_raw",
    "gamma_75_80_raw",
]
CHANNELS = [f"ECOG_{e}_{b}" for e in range(1, 5) for b in BANDS]

# Laplacian output channels (15 bands for LAPLACIAN_14-16)
LAP_OUTPUTS = [f"{LAP_PAIR}_LFP_{b}" for b in BANDS]

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
            "output_type": "neural",
            "output": LAP_OUTPUTS,
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


def variant_name(nx: int, n1: int, dbs: str, vanilla: bool = False, i: int | None = None) -> str:
    tag = "vanilla_" if vanilla else ""
    ii = i if i is not None else FULL_I
    return f"psid_laplacian_{PARTICIPANT}_{SESSION}_nx_{nx}_n{n1}_i{ii}_{tag}dbs_{dbs}_200Hz_narrow_band"


def variant_config_path(nx: int, n1: int, dbs: str, vanilla: bool = False, i: int | None = None) -> Path:
    name = variant_name(nx, n1, dbs, vanilla, i=i)
    subdir = "both" if dbs == "both" else dbs
    return Path(f"training/setups/psid/laplacian_200Hz/{subdir}/{name}.yaml")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Grid search
# ═════════════════════════════════════════════════════════════════════════════

def phase1_grid_search() -> Path:
    log.info("=" * 60)
    log.info("PHASE 1: Grid search  nx=%s × n1=%s (Laplacian %s)", NX_GRID, N1_GRID, LAP_PAIR)
    log.info("=" * 60)

    gs_config = {
        "model": {
            "name": GS_BASE,
            "i": GS_I,
            "time_first": True,
            "reuse_splits": True,
            "backward_kalman": True,
            "rescale_states": True,
            "max_eigenvalue": MAX_EIGENVALUE,
            "forecast": {"m": 2, "history": 5},
            "grid_search": {
                "nx": NX_GRID,
                "n1": N1_GRID,
            },
        },
        "data": _data_block("both"),
        "results": {
            "save_dir": f"results/{GS_BASE}",
            "log_dir": f"results/{GS_BASE}/logs",
            "checkpoint_dir": "checkpoints",
        },
    }

    cfg_path = write_yaml(f"training/setups/{GS_BASE}.yaml", gs_config)
    log.info("Grid search config → %s", cfg_path)

    run_cmd(
        [sys.executable, "-m", "training.psid_grid_search",
         "--config", str(cfg_path), "--workers", str(GS_WORKERS)],
        label="grid_search",
        timeout=14400,
    )

    gs_dir = PROJECT_ROOT / "results" / GS_BASE
    log.info("Grid search results → %s", gs_dir)
    return gs_dir


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Classify grid-search runs, pick best nx / n1
# ═════════════════════════════════════════════════════════════════════════════

def phase2_find_best(gs_dir: Path) -> tuple[int, int, float]:
    log.info("=" * 60)
    log.info("PHASE 2: Classify grid-search runs → best nx / n1")
    log.info("=" * 60)

    run_dirs = sorted(gs_dir.glob(f"{GS_BASE}_run*"))
    log.info("Found %d grid-search run directories", len(run_dirs))

    best_ba, best_nx, best_n1 = -1.0, -1, -1

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue

        model_files = sorted(
            [p for p in run_dir.glob("model_*.pkl") if "_metadata" not in p.name]
        )
        if not model_files:
            log.info("  SKIP %s: no model", run_dir.name)
            continue
        run_ts = model_files[-1].stem.replace("model_", "")

        meta_files = sorted(run_dir.glob("model_*_metadata.json"))
        if not meta_files:
            log.info("  SKIP %s: no metadata", run_dir.name)
            continue
        with open(meta_files[-1]) as f:
            meta = json.load(f)
        nx = int(meta.get("nx", 0))
        n1 = int(meta.get("n1", 0))

        rel_variant = f"{GS_BASE}/{run_dir.name}"
        cls_results_dir = str(PROJECT_ROOT / "results" / "classification" / "gs_lap_200Hz" / run_dir.name)

        cls_cfg = {
            "run": {"variant": rel_variant, "run_ts": run_ts},
            "classification": {
                "flipped": False,
                "n_splits": 5,
                "epoch_length": 0.5,
                "epoch_overlap": 0.25,
                "sampling_freq": SAMPLING_FREQ,
                "h": [],
                "m": [],
                "n1": n1,
                "nx": nx,
                "prediction_feature_source": "Xp",
                "forecast_feature_source": "Xp",
                "permutation_test": False,
                "n_permutations": 0,
                "param_grid": {
                    "LDA": {
                        "classifier__solver": ["lsqr"],
                        "classifier__shrinkage": ["auto"],
                    }
                },
            },
            "results": {
                "project_root": str(PROJECT_ROOT),
                "results_dir": cls_results_dir,
                "log_dir": str(PROJECT_ROOT / "logs" / "classification"),
            },
        }

        cfg_path = PROJECT_ROOT / "classification" / "setups" / f"_tmp_gs_lap_{run_dir.name}.yaml"
        write_yaml(cfg_path, cls_cfg)

        log.info("  Classifying %s (nx=%d, n1=%d) …", run_dir.name, nx, n1)
        res = run_cmd(
            [sys.executable, "classification/compute.py", "--config", str(cfg_path)],
            label=f"gs_cls_{run_dir.name}",
            timeout=1800,
        )
        cfg_path.unlink(missing_ok=True)

        if res.returncode != 0:
            continue

        pkl_path = Path(cls_results_dir) / run_ts / "LDA_Xp_prediction.pkl"
        if not pkl_path.exists():
            log.info("    MISSING pickle: %s", pkl_path)
            continue

        with open(pkl_path, "rb") as f:
            cls_res = pickle.load(f)

        test_ba = cls_res.get("test_results", {}).get("balanced_accuracy", -1)
        cv_ba = -1.0
        if "fold_results" in cls_res:
            cv_bas = [fr["balanced_accuracy"] for fr in cls_res["fold_results"]
                      if "balanced_accuracy" in fr]
            if cv_bas:
                cv_ba = sum(cv_bas) / len(cv_bas)

        ba = test_ba if test_ba > 0 else cv_ba
        log.info("    balanced_accuracy=%.4f  (test=%.4f, cv=%.4f)", ba, test_ba, cv_ba)

        if ba > best_ba:
            best_ba, best_nx, best_n1 = ba, nx, n1

    if best_nx < 0:
        log.error("No successful classification runs!")
        sys.exit(1)

    log.info(">>> BEST: nx=%d, n1=%d, balanced_accuracy=%.4f", best_nx, best_n1, best_ba)
    return best_nx, best_n1, best_ba


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Full training (4 variants)
# ═════════════════════════════════════════════════════════════════════════════

def _make_training_config(nx: int, n1: int, dbs: str,
                          backward_kalman: bool, rescale_states: bool,
                          vanilla: bool = False, i: int | None = None) -> Path:
    ii = i if i is not None else FULL_I
    name = variant_name(nx, n1, dbs, vanilla, i=ii)
    cfg = {
        "model": {
            "name": name,
            "nx": nx,
            "n1": n1,
            "i": ii,
            "time_first": True,
            "fast": False,
            "reuse_splits": False,
            "backward_kalman": backward_kalman,
            "rescale_states": rescale_states,
            "max_eigenvalue": MAX_EIGENVALUE,
            "forecast": {"m": 2, "history": 5},
        },
        "data": _data_block(dbs),
        "results": {
            "save_dir": f"results/{name}",
            "log_dir": f"results/{name}/logs",
            "checkpoint_dir": "checkpoints",
        },
    }
    return write_yaml(variant_config_path(nx, n1, dbs, vanilla, i=ii), cfg)


def _train_with_retry(nx: int, n1: int, dbs: str,
                      backward_kalman: bool, rescale_states: bool,
                      vanilla: bool = False) -> tuple[str, int]:
    """Try training with decreasing i values until it succeeds."""
    i_values = [FULL_I] + [i for i in [45, 40, 35, 30] if i < FULL_I]
    for ii in i_values:
        name = variant_name(nx, n1, dbs, vanilla, i=ii)
        result_dir = PROJECT_ROOT / "results" / name
        existing_models = [p for p in result_dir.glob("model_*.pkl")
                           if "_metadata" not in p.name] if result_dir.exists() else []
        if existing_models:
            ts = sorted(existing_models)[-1].stem.replace("model_", "")
            log.info("  SKIP training %s — model already exists (ts=%s)", name, ts)
            return ts, ii
        if result_dir.exists():
            shutil.rmtree(result_dir)

        cfg = _make_training_config(nx, n1, dbs, backward_kalman, rescale_states, vanilla, i=ii)
        log.info("  Training %s (i=%d) …", name, ii)
        result = run_cmd(
            [sys.executable, "-m", "training.train", "--config", str(cfg)],
            label=f"train_{dbs}_i{ii}",
            timeout=7200,
        )
        model_files = list((PROJECT_ROOT / "results" / name).glob("model_*.pkl"))
        model_files = [p for p in model_files if "_metadata" not in p.name]
        if model_files:
            ts = sorted(model_files)[-1].stem.replace("model_", "")
            log.info("    → ts=%s (trained with i=%d)", ts, ii)
            return ts, ii
        else:
            log.warning("  Training failed with i=%d, trying lower …", ii)

    log.error("  All i values failed for %s!", variant_name(nx, n1, dbs, vanilla))
    sys.exit(1)


def phase3_train(nx: int, n1: int) -> tuple[dict[str, str], dict[str, int]]:
    log.info("=" * 60)
    log.info("PHASE 3: Full training  nx=%d, n1=%d, i=%d (with auto-retry)", nx, n1, FULL_I)
    log.info("=" * 60)

    timestamps: dict[str, str] = {}
    i_used: dict[str, int] = {}

    for dbs in ("both", "on", "off"):
        ts, ii = _train_with_retry(nx, n1, dbs, backward_kalman=True, rescale_states=True)
        timestamps[dbs] = ts
        i_used[dbs] = ii
        name = variant_name(nx, n1, dbs, i=ii)

        cfg = variant_config_path(nx, n1, dbs, i=ii)
        log.info("  Testing %s …", name)
        run_cmd(
            [sys.executable, "-m", "training.test", "--config", str(cfg), "--run", ts],
            label=f"test_{dbs}",
            timeout=7200,
        )

    # Vanilla dbs_both
    ts, ii = _train_with_retry(nx, n1, "both", backward_kalman=False, rescale_states=False, vanilla=True)
    timestamps["vanilla"] = ts
    i_used["vanilla"] = ii
    v_name = variant_name(nx, n1, "both", vanilla=True, i=ii)

    cfg = variant_config_path(nx, n1, "both", vanilla=True, i=ii)
    log.info("  Testing %s …", v_name)
    run_cmd(
        [sys.executable, "-m", "training.test", "--config", str(cfg), "--run", timestamps["vanilla"]],
        label="test_vanilla",
        timeout=7200,
    )

    log.info("  Actual i values: %s", i_used)
    return timestamps, i_used


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Cross-condition eval
# ═════════════════════════════════════════════════════════════════════════════

def phase4_cross_eval(nx: int, n1: int, timestamps: dict[str, str],
                      i_used: dict[str, int] | None = None):
    log.info("=" * 60)
    log.info("PHASE 4: Cross-condition evaluation")
    log.info("=" * 60)

    i_on = i_used["on"] if i_used else FULL_I
    i_off = i_used["off"] if i_used else FULL_I
    on_cfg = variant_config_path(nx, n1, "on", i=i_on)
    off_cfg = variant_config_path(nx, n1, "off", i=i_off)

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
# PHASE 5 — Classification
# ═════════════════════════════════════════════════════════════════════════════

def _find_best_hm(results_dir: Path, run_ts: str, feat_label: str) -> tuple[float, float, float]:
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


def phase5_classification(nx: int, n1: int, timestamps: dict[str, str],
                          i_used: dict[str, int] | None = None):
    log.info("=" * 60)
    log.info("PHASE 5: Classification (2-step: grid search → permutation test)")
    log.info("=" * 60)

    i_both = i_used["both"] if i_used else FULL_I
    i_on = i_used["on"] if i_used else FULL_I
    i_off = i_used["off"] if i_used else FULL_I
    both_var = variant_name(nx, n1, "both", i=i_both)
    on_var = variant_name(nx, n1, "on", i=i_on)
    off_var = variant_name(nx, n1, "off", i=i_off)
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
        "n1": n1,
        "nx": nx,
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

    # Flipped
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
# PHASE 6 — Update specs.py & generate thesis HTML
# ═════════════════════════════════════════════════════════════════════════════

def phase6_thesis(nx: int, n1: int, timestamps: dict[str, str],
                  i_used: dict[str, int] | None = None):
    log.info("=" * 60)
    log.info("PHASE 6: Update specs.py & generate thesis HTML")
    log.info("=" * 60)

    # For now just regenerate HTML — Laplacian triplet names TBD
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
                        help="Phase to start from (1-6). Phases before this are skipped.")
    parser.add_argument("--best-nx", type=int, default=None)
    parser.add_argument("--best-n1", type=int, default=None)
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("PSID Laplacian pipeline — %s S%s, 200 Hz narrow band", PARTICIPANT, SESSION)
    log.info("  Output: %s (15 bands, neural-to-neural)", LAP_PAIR)
    log.info("  GS: nx=%s × n1=%s, i=%d", NX_GRID, N1_GRID, GS_I)
    log.info("  Full: i=%d (with auto-retry)", FULL_I)
    log.info("  Project root: %s", PROJECT_ROOT)
    if args.start_phase > 1:
        log.info("Resuming from Phase %d", args.start_phase)
    log.info("=" * 60)

    for d in ["training/setups/psid/laplacian_200Hz/both",
              "training/setups/psid/laplacian_200Hz/on",
              "training/setups/psid/laplacian_200Hz/off",
              "classification/setups",
              "logs/classification"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    best_ba = 0.0
    if args.start_phase <= 1:
        gs_dir = phase1_grid_search()
    else:
        gs_dir = PROJECT_ROOT / "results" / GS_BASE

    if args.start_phase <= 2:
        best_nx, best_n1, best_ba = phase2_find_best(gs_dir)
    else:
        best_nx = args.best_nx
        best_n1 = args.best_n1
        if best_nx is None or best_n1 is None:
            raise ValueError("--best-nx and --best-n1 required when --start-phase > 2")
        log.info("Using provided best params: nx=%d, n1=%d", best_nx, best_n1)

    i_used = None
    if args.start_phase <= 3:
        timestamps, i_used = phase3_train(best_nx, best_n1)
    else:
        timestamps = {}
        for tag, dbs in [("", "both"), ("", "on"), ("", "off"), ("vanilla_", "both")]:
            key = "vanilla" if tag == "vanilla_" else dbs
            name = variant_name(best_nx, best_n1, dbs, vanilla=(tag == "vanilla_"))
            ts = get_latest_model_ts(PROJECT_ROOT / "results" / name)
            timestamps[key] = ts

    if args.start_phase <= 4:
        phase4_cross_eval(best_nx, best_n1, timestamps, i_used=i_used)
    if args.start_phase <= 5:
        phase5_classification(best_nx, best_n1, timestamps, i_used=i_used)
    if args.start_phase <= 6:
        phase6_thesis(best_nx, best_n1, timestamps, i_used=i_used)

    log.info("")
    log.info("=" * 60)
    log.info("LAPLACIAN PIPELINE COMPLETE")
    log.info("  Best grid-search: nx=%d, n1=%d (bal_acc=%.4f)", best_nx, best_n1, best_ba)
    if i_used:
        log.info("  Actual i values: %s", i_used)
    log.info("  Timestamps: both=%s  on=%s  off=%s  vanilla=%s",
             timestamps["both"], timestamps["on"], timestamps["off"], timestamps["vanilla"])
    log.info("  Full log: %s", LOG_PATH)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
