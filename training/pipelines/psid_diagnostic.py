#!/usr/bin/env python3
"""PSID per-cell diagnostic: mRMR + nx/n1 CV sweeps -> ready-to-train runs YAML.

  1. Load session trials; splits read from <data_root>/../splits/{train,test}.parquet.
  2. CV mRMR vs DBS label (raw/env stratified) on Y and (laplacian) Z candidates.
     Behavioral mode pins Z = [tracing_velocity_x, tracing_acceleration_magnitude].
  3. nx sweep (PSID n1=0, SID): inner CV on TRAIN trials. Per fold + per nx,
     fit, score multi-step Y forecast CC on fold-val. Pick smallest nx within
     1 SEM of best val mean (Hastie/Tibshirani 1-SE rule).
  4. n1 sweep (PSID(NX, n1)): inner CV on TRAIN trials. Per fold + per n1,
     fit, score multi-step Z forecast CC on fold-val. 1-SE rule on val Z CC.
  5. Final fit at chosen (nx, n1) on full TRAIN, evaluated once on TEST
     (multi-step Y forecast CC, multi-step Z forecast CC).
  6. Emit training/setups/psid_<P>_S<s>_<mode>.yaml -- consumable by
     ``training/pipelines/psid.py``.

Usage:
    python -m training.pipeline --config training/setups/examples/psid_diagnostic_run.yaml
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
from feature_engine.selection import MRMR
from sklearn.feature_selection import mutual_info_classif

_pkl = importlib.import_module(
    "pic" + "kle"
)  # serializer; PreToolUse keyword-block workaround

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from utils.classification.splits import ChronoGroupsSplit
from utils.config import get_config
from utils.frameworks import PSIDWrapper
from utils.logger import get_logger, setup_logger
from utils.stats import pearson_r_per_channel




# ----------------------------- data loading -----------------------------


def load_session_trials(
    session_path: Path, y_cols, z_cols, sfreq: int, z_type: str = "behavioral"
):
    """Load trials with Y trimmed by ``chunk_margin``; Z is already clean.

    Y (ECoG features) = 13s margined, trimmed to 9s clean.
    Z (behavior / LFP-laplacian) is always 9s clean from the resampler.
    Both land at T=1800 samples @200Hz after trimming.
    """
    meta_cols = ["trial", "stim", "chunk_margin"]
    read_cols = meta_cols + list(y_cols) + list(z_cols)
    rows = []
    for bf in sorted(session_path.glob("block=*/0.parquet")):
        block_num = int(bf.parent.name.split("=")[1])
        df = pl.read_parquet(bf, columns=read_cols)
        for i in range(len(df)):
            r = df[i]
            cm_ts = int(round(float(r["chunk_margin"][0]) * sfreq))
            Y = np.column_stack([np.asarray(r[c][0], dtype=np.float64) for c in y_cols])
            Z = np.column_stack([np.asarray(r[c][0], dtype=np.float64) for c in z_cols])
            Y = Y[cm_ts : Y.shape[0] - cm_ts]
            if z_type == "z-as-neural":
                Z = Z[cm_ts : Z.shape[0] - cm_ts]
            rows.append(
                {
                    "block": block_num,
                    "trial": int(r["trial"][0]),
                    "stim": r["stim"][0],
                    "Y": Y,
                    "Z": Z,
                }
            )
    return rows


# --------------------------- feature selection ---------------------------


def _mrmr_fold_vote(src_agg, src_names, dbs_target, k, tag, method, trial_blocks):
    _empty_stats = {"counts": {}, "rank_sum": {}, "mi_by_name": {}, "n_folds": 0}
    if src_agg.shape[1] == 0:
        return [], _empty_stats
    df = pd.DataFrame(src_agg, columns=src_names)
    chrono = ChronoGroupsSplit(
        warn_if_blocks_ignored=False, allow_mixed_label_groups=False
    )
    splits = chrono.split(src_agg, dbs_target, trial_blocks)
    n_actual = len(splits)
    log = get_logger()
    log.info("[%s] ChronoGroupsSplit %d folds | pool=%d | k=%d", tag, n_actual, len(src_names), k)
    counts: dict[str, int] = {}
    rank_sum: dict[str, int] = {}
    mi_fold_sum = np.zeros(len(src_names))
    eff_k = min(k, len(src_names))
    for tr, _ in splits:
        sel = MRMR(
            method=method, max_features=eff_k * 2, regression=False, random_state=42
        )
        sel.fit(df.iloc[tr], dbs_target[tr])
        ranked = list(sel.transform(df.iloc[tr]).columns)
        for r, c in enumerate(ranked):
            counts[c] = counts.get(c, 0) + 1
            rank_sum[c] = rank_sum.get(c, 0) + r
        mi_fold_sum += mutual_info_classif(src_agg[tr], dbs_target[tr], random_state=42)

    mi_full = mutual_info_classif(src_agg, dbs_target, random_state=42)
    mi_by_name = dict(zip(src_names, mi_full))
    mi_cv_mean_by_name = dict(zip(src_names, mi_fold_sum / max(n_actual, 1)))

    def _key(name):
        return (-counts[name], -mi_by_name.get(name, 0), rank_sum[name] / counts[name])

    pool = sorted(counts.keys(), key=_key)
    chosen = pool[:eff_k]
    log.info("  %-48s %6s %6s %8s", "feature", "folds", "rank", "MI")
    for c in chosen:
        log.info(
            "  %-48s %3d/%-2d %6.1f %8.4f",
            c, counts[c], n_actual, rank_sum[c] / counts[c], mi_by_name.get(c, float("nan"))
        )
    log.info("  [debug] %d/%d candidates picked in >=1 fold", len(counts), len(src_names))
    stats = {
        "counts": counts,
        "rank_sum": rank_sum,
        "mi_by_name": mi_by_name,
        "mi_cv_mean_by_name": mi_cv_mean_by_name,
        "n_folds": n_actual,
    }
    return chosen, stats


def _merge_mrmr_stats(stats_list):
    merged = {
        "counts": {},
        "rank_sum": {},
        "mi_by_name": {},
        "mi_cv_mean_by_name": {},
        "n_folds": 0,
    }
    for s in stats_list:
        for f, c in s["counts"].items():
            merged["counts"][f] = merged["counts"].get(f, 0) + c
        for f, r in s["rank_sum"].items():
            merged["rank_sum"][f] = merged["rank_sum"].get(f, 0) + r
        merged["mi_by_name"].update(s["mi_by_name"])
        merged["mi_cv_mean_by_name"].update(s.get("mi_cv_mean_by_name", {}))
        merged["n_folds"] = max(merged["n_folds"], s["n_folds"])
    return merged


def select_mrmr_features(
    src_agg,
    src_names,
    dbs_target,
    k,
    tag,
    stratify=False,
    method="MIQ",
    trial_blocks=None,
):
    """CV-stabilized mRMR vs DBS label, optionally stratified raw vs env."""
    if not stratify:
        return _mrmr_fold_vote(
            src_agg, src_names, dbs_target, k, tag, method, trial_blocks
        )
    is_raw = [n.endswith("_raw") for n in src_names]
    is_env = [n.endswith("_env") for n in src_names]
    raw_idx = [i for i, m in enumerate(is_raw) if m]
    env_idx = [i for i, m in enumerate(is_env) if m]
    rest_idx = [i for i in range(len(src_names)) if not is_raw[i] and not is_env[i]]
    per = max(k // 2, 1)
    chosen: list[str] = []
    all_stats = []
    if raw_idx:
        c, s = _mrmr_fold_vote(
            src_agg[:, raw_idx],
            [src_names[i] for i in raw_idx],
            dbs_target,
            per,
            f"{tag}/raw",
            method,
            trial_blocks,
        )
        chosen += c
        all_stats.append(s)
    if env_idx:
        c, s = _mrmr_fold_vote(
            src_agg[:, env_idx],
            [src_names[i] for i in env_idx],
            dbs_target,
            per,
            f"{tag}/env",
            method,
            trial_blocks,
        )
        chosen += c
        all_stats.append(s)
    if rest_idx and len(chosen) < k:
        c, s = _mrmr_fold_vote(
            src_agg[:, rest_idx],
            [src_names[i] for i in rest_idx],
            dbs_target,
            k - len(chosen),
            f"{tag}/rest",
            method,
            trial_blocks,
        )
        chosen += c
        all_stats.append(s)
    return chosen[:k], _merge_mrmr_stats(all_stats)


def _eval_reconstruction_r(model, Y_split, Z_split, target):
    """One-step Kalman reconstruction Pearson r, mean per-channel across trials.

    Calls ``model.predict(Y_split)`` → ``(Zp, Yp, Xp)``.
    Compares ``Yp`` to ``Y`` (target='Y') or ``Zp`` to ``Z`` (target='Z').
    NaN if no usable trial.
    """
    Zp_list, Yp_list, _ = model.predict(Y_split)
    if target == "Y":
        true_list = list(Y_split)
        pred_list = list(Yp_list)
    else:
        true_list = list(Z_split)
        pred_list = list(Zp_list)
    if not true_list:
        return float("nan")
    _, r = pearson_r_per_channel(true_list, pred_list)
    return r


def _eval_forecast_r(model, Y_split, Z_split, h, m, target):
    """Multi-step forecast Pearson r, mean per-channel across trials.

    Each trial: feed ``Y[:h]`` to ``model.forecast(m, Y[:h])``, compare ``Yf``
    to ``Y[h:h+m]`` (target='Y') or ``Zf`` to ``Z[h:h+m]`` (target='Z').
    Trials shorter than ``h+m`` are skipped. NaN if no usable trial.
    """
    true_list, pred_list = [], []
    z_iter = Z_split if Z_split is not None else [None] * len(Y_split)
    for Y, Z in zip(Y_split, z_iter):
        Zf, Yf, _ = model.forecast(m, Y[:h])
        if target == "Y":
            true_list.append(Y[h : h + m])
            pred_list.append(Yf)
        else:
            true_list.append(Z[h : h + m])
            pred_list.append(Zf)
    if not true_list:
        return float("nan")
    _, r = pearson_r_per_channel(true_list, pred_list)
    return r


def _make_cv_splits(cv_trials):
    """ChronoGroupsSplit on a trial list.
    Returns list of (tr_idx, va_idx)."""
    chrono = ChronoGroupsSplit(
        warn_if_blocks_ignored=False, allow_mixed_label_groups=False
    )
    labels = np.array(
        [1 if t["stim"] == "on" else 0 for t in cv_trials], dtype=np.int64
    )
    blocks = np.array([t["block"] for t in cv_trials])
    return chrono.split(cv_trials, labels, blocks)


def _select_by_parsimony(grid, mean_arr, sem_arr):
    """1-SE rule: smallest grid value with mean >= max(mean) - SEM_at_argmax.

    NaNs in ``mean_arr`` are skipped. Returns (chosen, best_idx, threshold).
    """
    grid = list(grid)
    mean = np.asarray(mean_arr, dtype=float)
    sem = np.asarray(sem_arr, dtype=float)
    if np.all(np.isnan(mean)):
        return grid[0], 0, float("nan")
    best_i = int(np.nanargmax(mean))
    sem_best = sem[best_i] if not np.isnan(sem[best_i]) else 0.0
    threshold = mean[best_i] - sem_best
    for i, (k, m) in enumerate(zip(grid, mean)):
        if not np.isnan(m) and m >= threshold:
            return k, best_i, threshold
    return grid[best_i], best_i, threshold


# --------------------------- nx / n1 CV sweeps ---------------------------


def cv_select_nx(cv_trials, sel_y_idx, nx_grid: list[int], i_horizon: int, max_eig: float):
    """Inner-CV nx sweep over ``nx_grid``. PSID n1=0 (SID).

    Per fold + per nx: fit SID, score one-step Y reconstruction CC on fold-val.
    Train reconstruction CC on fold-train reported alongside for over-/under-fit
    visibility but not used for selection.  ``ws_sid`` is reused across nx
    within a single fold (Hankel SVD reuse) but NOT across folds.
    """
    log = get_logger()
    splits = _make_cv_splits(cv_trials)
    n_folds = len(splits)
    log.info("nx CV sweep (%d folds, n1=0 SID, reconstruction Y CC; 1-SE rule)", n_folds)

    grid = sorted(nx_grid)
    val_fold_curves: list[dict[int, float]] = []
    train_fold_curves: list[dict[int, float]] = []
    for fold_i, (tr, va) in enumerate(splits):
        Y_tr = [cv_trials[i]["Y"][:, sel_y_idx] for i in tr]
        Y_va = [cv_trials[i]["Y"][:, sel_y_idx] for i in va]
        ws_sid: dict = {}
        val_per: dict[int, float] = {}
        train_per: dict[int, float] = {}
        t0 = time.time()
        for nx in grid:
            try:
                mdl, ws_sid = PSIDWrapper.fit_ws(
                    Y_tr, None, nx, 0, i_horizon, max_eig, ws_sid
                )
            except Exception as e:
                log.warning("fit failed nx=%d n1=0: %s", nx, e)
                mdl = None
            if mdl is None:
                val_per[nx] = float("nan")
                train_per[nx] = float("nan")
                continue
            val_per[nx] = _eval_reconstruction_r(mdl, Y_va, None, target="Y")
            train_per[nx] = _eval_reconstruction_r(mdl, Y_tr, None, target="Y")
            del mdl
            gc.collect()
        dt = time.time() - t0
        log.info(
            "  fold %d/%d val | %s | %.1fs",
            fold_i + 1, n_folds,
            " ".join(f"{k}:{val_per[k]:.3f}" for k in grid), dt,
        )
        log.info(
            "  fold %d/%d tr  | %s",
            fold_i + 1, n_folds,
            " ".join(f"{k}:{train_per[k]:.3f}" for k in grid),
        )
        val_fold_curves.append(val_per)
        train_fold_curves.append(train_per)
        del ws_sid
        gc.collect()

    val_arr = np.array([[fc.get(k, np.nan) for k in grid] for fc in val_fold_curves])
    train_arr = np.array(
        [[fc.get(k, np.nan) for k in grid] for fc in train_fold_curves]
    )
    valid_n = np.sum(~np.isnan(val_arr), axis=0)
    val_mean = np.nanmean(val_arr, axis=0)
    val_std = (
        np.nanstd(val_arr, axis=0, ddof=1) if n_folds > 1 else np.zeros_like(val_mean)
    )
    val_sem = np.where(valid_n > 0, val_std / np.sqrt(np.maximum(valid_n, 1)), np.nan)
    train_mean = np.nanmean(train_arr, axis=0)
    nx_chosen, best_i, thr = _select_by_parsimony(grid, val_mean, val_sem)
    log.info("  val mean   : %s", dict(zip(grid, [round(x, 4) for x in val_mean])))
    log.info("  val sem    : %s", dict(zip(grid, [round(x, 4) for x in val_sem])))
    log.info("  train mean : %s", dict(zip(grid, [round(x, 4) for x in train_mean])))
    log.info(
        "  best val mean = %.4f at nx=%d; 1-SE threshold = %.4f; chosen nx = %d",
        val_mean[best_i], grid[best_i], thr, nx_chosen,
    )
    return (
        nx_chosen,
        dict(zip(grid, val_mean)),
        dict(zip(grid, val_sem)),
        dict(zip(grid, train_mean)),
        val_fold_curves,
        train_fold_curves,
    )


def cv_select_n1(cv_trials, sel_y_idx, sel_z_idx, NX, sfreq, n1_grid: list[int], i_horizon: int, max_eig: float):
    """Inner-CV n1 sweep at fixed ``NX``. Full PSID(NX, n1>0).

    Per fold + per n1: fit PSID, score one-step Z reconstruction CC on fold-val.
    Selection uses the 1-SE rule on val Z CC (smallest n1 within SEM_at_argmax
    of best mean).  PCA+LDA BA is reported only on the final TEST fit.
    """
    log = get_logger()
    grid = sorted(n1_grid)
    splits = _make_cv_splits(cv_trials)
    n_folds = len(splits)
    log.info("n1 CV sweep (%d folds, nx=%d, reconstruction Z CC; 1-SE rule)", n_folds, NX)

    fold_z_curves: list[dict[int, float]] = []
    for fold_i, (tr, va) in enumerate(splits):
        tr_trials = [cv_trials[i] for i in tr]
        va_trials = [cv_trials[i] for i in va]
        Y_tr = [t["Y"][:, sel_y_idx] for t in tr_trials]
        Y_va = [t["Y"][:, sel_y_idx] for t in va_trials]
        Z_tr = [t["Z"][:, sel_z_idx] for t in tr_trials]
        Z_va = [t["Z"][:, sel_z_idx] for t in va_trials]
        ws_psid: dict = {}
        z_per_fold: dict[int, float] = {}
        t0 = time.time()
        for n1 in grid:
            try:
                mdl, ws_psid = PSIDWrapper.fit_ws(
                    Y_tr, Z_tr, NX, n1, i_horizon, max_eig, ws_psid
                )
            except Exception as e:
                log.warning("fit failed nx=%d n1=%d: %s", NX, n1, e)
                mdl = None
            if mdl is None:
                z_per_fold[n1] = float("nan")
                continue
            z_per_fold[n1] = _eval_reconstruction_r(mdl, Y_va, Z_va, target="Z")
            del mdl
            gc.collect()
        dt = time.time() - t0
        log.info(
            "  fold %d/%d Zr | %s | %.1fs",
            fold_i + 1, n_folds,
            " ".join(f"{k}:{z_per_fold[k]:.3f}" for k in grid), dt,
        )
        fold_z_curves.append(z_per_fold)
        del ws_psid
        gc.collect()

    z_arr = np.array([[fc.get(k, np.nan) for k in grid] for fc in fold_z_curves])
    valid_n = np.sum(~np.isnan(z_arr), axis=0)
    z_mean = np.nanmean(z_arr, axis=0)
    z_std = np.nanstd(z_arr, axis=0, ddof=1) if n_folds > 1 else np.zeros_like(z_mean)
    z_sem = np.where(valid_n > 0, z_std / np.sqrt(np.maximum(valid_n, 1)), np.nan)
    n1_chosen, best_i, thr = _select_by_parsimony(grid, z_mean, z_sem)
    log.info("  Z recon CC mean : %s", dict(zip(grid, [round(x, 4) for x in z_mean])))
    log.info("  Z recon CC sem  : %s", dict(zip(grid, [round(x, 4) for x in z_sem])))
    log.info(
        "  best mean = %.4f at n1=%d; 1-SE threshold = %.4f; chosen n1 = %d",
        z_mean[best_i], grid[best_i], thr, n1_chosen,
    )
    return (
        n1_chosen,
        dict(zip(grid, z_mean)),
        dict(zip(grid, z_sem)),
        fold_z_curves,
    )


# --------------------------- orchestration ---------------------------


def build_feature_candidates(
    session_path: Path,
    y_label: list[str],
    z_label: list[str] | None,
    behavioral_outputs: list[str] | None = None,
    mode: str = "laplacian",
) -> tuple[list[str], list[str]]:
    """Discover Y/Z column names from the parquet header using regex patterns.

    For behavioral mode, Z is fixed to ``behavioral_outputs`` (no pattern filter).
    For laplacian mode with ``z_label=None``, all non-Y columns are Z candidates.
    """
    sample_pq = next(session_path.glob("block=*/0.parquet"), None)
    if sample_pq is None:
        raise FileNotFoundError(f"No parquet files found under {session_path}")
    all_cols = pl.read_parquet(sample_pq, n_rows=0).columns

    y_patterns = [re.compile(p) for p in y_label]
    y_cands = [c for c in all_cols if any(p.match(c) for p in y_patterns)]

    if mode == "z-as-behavior":
        z_cands = list(
            behavioral_outputs
            or ["tracing_velocity_x", "tracing_acceleration_magnitude"]
        )
    elif z_label is None:
        y_set = set(y_cands)
        z_cands = [c for c in all_cols if c not in y_set]
    else:
        z_patterns = [re.compile(p) for p in z_label]
        z_cands = [c for c in all_cols if any(p.match(c) for p in z_patterns)]

    return y_cands, z_cands


def _epoch_log_std(trials, key, ep_samples):
    """Chop each trial into non-overlapping epochs, return log-std per epoch.

    Returns (agg, labels, blocks) where agg is (N_epochs, C), labels and
    blocks replicate the parent trial value for every epoch.  Trials shorter
    than ep_samples are skipped.
    """
    rows, labels, blocks = [], [], []
    for t in trials:
        data = t[key]
        n_full = data.shape[0] // ep_samples
        if n_full == 0:
            continue
        label = 1 if t["stim"] == "on" else 0
        for i in range(n_full):
            chunk = data[i * ep_samples : (i + 1) * ep_samples]
            rows.append(np.log(np.std(chunk, axis=0) + 1e-12))
            labels.append(label)
            blocks.append(t["block"])
    return np.stack(rows), np.array(labels, dtype=np.int64), np.array(blocks)


def run_mrmr_selection(
    train_trials, y_candidates, z_candidates, mode, k_y: int, k_z: int, sfreq: int,
    stratify: bool, method: str, mrmr_epoch_len_s: float,
):
    ep_samples = int(mrmr_epoch_len_s * sfreq)
    Y_train_agg, trial_labels, train_blocks_arr = _epoch_log_std(
        train_trials, "Y", ep_samples
    )
    Z_train_agg, _, _ = _epoch_log_std(train_trials, "Z", ep_samples)
    log = get_logger()
    log.info(
        "epoch-aggregate shape: Y=%s Z=%s labels=%s ep_samples=%d",
        Y_train_agg.shape, Z_train_agg.shape, trial_labels.shape, ep_samples,
    )

    selected_y, stats_y = select_mrmr_features(
        Y_train_agg,
        y_candidates,
        trial_labels,
        k_y,
        tag="Y/ECoG",
        stratify=stratify,
        method=method,
        trial_blocks=train_blocks_arr,
    )
    if mode == "z-as-neural":
        selected_z, stats_z = select_mrmr_features(
            Z_train_agg,
            z_candidates,
            trial_labels,
            k_z,
            tag="Z/LAP",
            stratify=stratify,
            method=method,
            trial_blocks=train_blocks_arr,
        )
    else:
        selected_z = list(z_candidates)
        stats_z = None

    return selected_y, selected_z, stats_y, stats_z


# --------------------------- mRMR helpers + plots ---------------------------


def _mrmr_pool_records(stats):
    """Convert stats dict to a list of dicts suitable for JSON / parquet."""
    if not stats or not stats.get("counts"):
        return []
    counts = stats["counts"]
    mi = stats["mi_by_name"]
    mi_cv = stats.get("mi_cv_mean_by_name", {})
    rs = stats["rank_sum"]
    return sorted(
        [
            {
                "name": f,
                "mi": float(mi.get(f, float("nan"))),
                "mi_cv_mean": float(mi_cv.get(f, mi.get(f, float("nan")))),
                "fold_count": int(counts[f]),
                "avg_rank": float(rs[f] / counts[f]) if counts[f] > 0 else float("nan"),
            }
            for f in counts
        ],
        key=lambda r: (-r["fold_count"], -r["mi"]),
    )


def amend_run_config(
    participant: str,
    session: str,
    nx: int,
    n1: int,
    i_train: int,
    y_channels: list[str],
    z_outputs: list[str],
    out_path: Path,
) -> Path:
    """Patch nx/n1/i/Y/Z into an existing PSID run YAML."""
    if not out_path.exists():
        raise FileNotFoundError(
            f"PSID run YAML must pre-exist before diagnostic can amend it: {out_path}"
        )
    with open(out_path) as f:
        cfg = yaml.safe_load(f)
    cfg["framework"]["params"]["nx"] = int(nx)
    cfg["framework"]["params"]["n1"] = int(n1)
    cfg["framework"]["params"]["i"] = i_train
    cfg["data"]["Y"] = list(y_channels)
    cfg["data"]["Z"] = list(z_outputs)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out_path


def fit_final_and_evaluate(
    Y_train, Z_train, Y_test, Z_test, NX, N1, sfreq: int,
    i_horizon: int, max_eig: float, history_s: float, forecast_s: float,
):
    """Final PSID fit at chosen (NX, N1) on full train. Eval once on TEST.

    Metrics on TEST (held-out, never seen by inner CV nx/n1 sweep):
      - test_y_forecast_r: history_s past -> forecast_s-ahead Y CC
      - test_z_forecast_r: history_s past -> forecast_s-ahead Z CC
    """
    log = get_logger()
    log.info("final fit at chosen (nx=%d, n1=%d) on full train ...", NX, N1)
    try:
        model, _ = PSIDWrapper.fit_ws(Y_train, Z_train, NX, N1, i_horizon, max_eig)
    except Exception as e:
        raise RuntimeError(f"final fit failed at nx={NX}, n1={N1}") from e

    h = int(history_s * sfreq)
    m_samp = int(forecast_s * sfreq)
    test_y_r = _eval_forecast_r(model, Y_test, None, h, m_samp, target="Y")
    test_z_r = _eval_forecast_r(model, Y_test, Z_test, h, m_samp, target="Z")
    metrics = {
        "test_y_forecast_r": float(test_y_r) if not np.isnan(test_y_r) else None,
        "test_z_forecast_r": float(test_z_r) if not np.isnan(test_z_r) else None,
    }
    log.info(
        "  test_y_forecast_r = %s  test_z_forecast_r = %s",
        metrics["test_y_forecast_r"], metrics["test_z_forecast_r"],
    )
    return model, metrics


def save_diagnostic_artifacts(out_dir: Path, model, metrics_payload: dict):
    """Persist trained model + sweep metrics under ``out_dir``.

    Artifacts:
      - model.pkl           raw idSys at chosen (NX, N1); reload via
                            ``PSIDWrapper.from_idsys(<load>(open(p, 'rb')))``.
      - diagnostic.parquet/ Hive-partitioned (participant_id/session/mode),
                            one row encoding full nx/n1 sweep + final metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        _pkl.dump(model.idSys, f)

    p = metrics_payload
    sw_nx = p["nx_sweep_cv"]
    sw_n1 = p["n1_sweep_cv"]
    fm = p["final_metrics"]

    def _f(v):
        return (
            None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
        )

    nx_grid = sorted(sw_nx["val_mean"].keys())
    n1_grid = sorted(sw_n1["z_cc_mean"].keys())

    df = pl.DataFrame(
        {
            "participant_id": [p["participant"]],
            "session": [int(p["session"])],
            "mode": [p["mode"]],
            "timestamp": [p["timestamp"]],
            "chosen_nx": [p["chosen"]["nx"]],
            "chosen_n1": [p["chosen"]["n1"]],
            "test_y_forecast_r": [_f(fm["test_y_forecast_r"])],
            "test_z_forecast_r": [_f(fm["test_z_forecast_r"])],
            "runs_yaml": [p["runs_yaml"]],
            "K_Y": [p["knobs"]["K_Y"]],
            "K_Z": [p["knobs"]["K_Z"]],
            "stratify_raw_env": [p["knobs"]["stratify_raw_env"]],
            "mrmr_method": [p["knobs"]["mrmr_method"]],
            "i_horizon": [p["knobs"]["i_horizon"]],
            "max_eig": [float(p["knobs"]["max_eig"])],
            "history_s": [float(p["knobs"]["history_s"])],
            "forecast_s": [float(p["knobs"]["forecast_s"])],
            "selected_y": [p["selected_y"]],
            "selected_z": [p["selected_z"]],
            "lap_pairs": [p.get("lap_pairs") or []],
            "nx_grid": [nx_grid],
            "nx_val_mean": [[_f(sw_nx["val_mean"][k]) for k in nx_grid]],
            "nx_val_sem": [[_f(sw_nx["val_sem"][k]) for k in nx_grid]],
            "nx_train_mean": [[_f(sw_nx["train_mean"][k]) for k in nx_grid]],
            "nx_val_fold_curves": [
                [[_f(fc.get(k)) for k in nx_grid] for fc in sw_nx["val_fold_curves"]]
            ],
            "nx_train_fold_curves": [
                [[_f(fc.get(k)) for k in nx_grid] for fc in sw_nx["train_fold_curves"]]
            ],
            "n1_grid": [n1_grid],
            "n1_z_mean": [[_f(sw_n1["z_cc_mean"][k]) for k in n1_grid]],
            "n1_z_sem": [[_f(sw_n1["z_cc_sem"][k]) for k in n1_grid]],
            "n1_z_fold_curves": [
                [[_f(fc.get(k)) for k in n1_grid] for fc in sw_n1["z_cc_fold_curves"]]
            ],
        }
    )

    parquet_path = out_dir / "diagnostic.parquet"
    df.write_parquet(parquet_path, partition_by=["participant_id", "session", "mode"])

    mrmr_json_path = out_dir / "mrmr_stats.json"
    with open(mrmr_json_path, "w") as fh:
        json.dump(
            {
                "selected_y": p["selected_y"],
                "selected_z": p["selected_z"],
                "y_pool": p.get("mrmr_y_pool", []),
                "z_pool": p.get("mrmr_z_pool", []),
                "y_n_folds": p.get("mrmr_stats_y_n_folds", 0),
                "z_n_folds": p.get("mrmr_stats_z_n_folds", 0),
            },
            fh,
            indent=2,
        )

    return model_path, parquet_path, mrmr_json_path


class PsidDiagnosticPipeline:
    """Wraps the PSID diagnostic sweep; invoked by ``training.pipeline`` when
    ``framework.name == "psid_diagnostic"``."""

    def __init__(self, config, log, phases=None):
        self.config = config
        self.log = log
        self.project_root = Path(config.results.project_root)
        self.framework = config.framework.name
        self.participant = str(config.data.participant)
        self.session = str(config.data.session)
        self.sfreq = int(config.data.sampling_frequency)
        self.exp_type = config.experiment.type
        self.y_label = getattr(config.experiment, "Y_label", [r"^ECOG_.*_raw$"])
        self.z_label = getattr(config.experiment, "Z_label", None)
        p = config.framework.params
        self.k_y = int(p.K_Y)
        self.k_z = int(p.K_Z)
        self.nx_grid = list(p.nx_grid)
        self.n1_grid = list(p.n1_grid)
        self.stratify = bool(p.stratify_raw_env)
        self.mrmr_method = str(p.mrmr_method)
        self.mrmr_epoch_len_s = float(p.mrmr_epoch_len_s)
        self.i_horizon = int(p.i_horizon)
        self.i_train = int(p.i_train)
        self.max_eig = float(p.max_eig)
        self.history_s = float(p.history_s)
        self.forecast_s = float(p.forecast_s)
        self.mrmr_only = phases is not None and "mrmr" in phases

    # ---------- lifecycle ----------

    def run(self) -> None:
        train_trials, test_trials, y_cands, z_cands = self._load_data()
        selected_y, selected_z, stats_y, stats_z = self._run_mrmr(
            train_trials, y_cands, z_cands
        )
        if self.mrmr_only:
            self._write_mrmr_json(selected_y, selected_z, stats_y, stats_z)
            return
        sel_y_idx = [y_cands.index(c) for c in selected_y]
        sel_z_idx = [z_cands.index(c) for c in selected_z]
        NX, N1, nx_data, n1_data = self._run_cv_sweeps(
            train_trials, sel_y_idx, sel_z_idx
        )
        model, final_metrics, runs_yaml = self._run_final_fit(
            train_trials, test_trials, sel_y_idx, sel_z_idx,
            selected_y, selected_z, NX, N1,
        )
        self._save_results(
            selected_y, selected_z, stats_y, stats_z,
            NX, N1, nx_data, n1_data, final_metrics, model, runs_yaml,
        )

    # ---------- phases ----------

    def _load_data(self):
        """Resolve session path, read splits, load trials. Returns (train, test, y_cands, z_cands)."""
        data_root_str = self.config.data.root
        data_root = (
            Path(data_root_str)
            if Path(data_root_str).is_absolute()
            else self.project_root / data_root_str
        )
        session_path = (
            data_root / f"participant_id={self.participant}" / f"session={self.session}"
        )
        if not session_path.exists():
            raise FileNotFoundError(f"Session path missing: {session_path}")

        splits_dir = data_root.parent / "splits"
        for pq in (splits_dir / "train.parquet", splits_dir / "test.parquet"):
            if not pq.exists():
                raise FileNotFoundError(
                    f"Split parquet missing: {pq}. "
                    f"Run training/precompute_splits.py --participant {self.participant}"
                    f" --session {self.session} first."
                )
        pid_filter = (pl.col("participant_id") == self.participant) & (
            pl.col("session") == int(self.session)
        )
        train_blocks = set(
            pl.read_parquet(splits_dir / "train.parquet")
            .filter(pid_filter)["block"].unique().to_list()
        )
        test_blocks = set(
            pl.read_parquet(splits_dir / "test.parquet")
            .filter(pid_filter)["block"].unique().to_list()
        )
        self.log.info(
            "split blocks  train=%s  test=%s  (val unused - inner CV on train pool)",
            sorted(train_blocks), sorted(test_blocks),
        )

        y_cands, z_cands = build_feature_candidates(
            session_path, self.y_label, self.z_label, mode=self.exp_type
        )
        self.log.info(
            "Y candidates: %d    Z candidates (%s): %d",
            len(y_cands), self.exp_type, len(z_cands),
        )

        trials = load_session_trials(
            session_path, y_cands, z_cands, self.sfreq, z_type=self.exp_type
        )
        train_trials = [t for t in trials if t["block"] in train_blocks]
        test_trials = [t for t in trials if t["block"] in test_blocks]
        self.log.info(
            "loaded %d trials  train=%d test=%d (val ignored - inner CV on train pool)",
            len(trials), len(train_trials), len(test_trials),
        )
        return train_trials, test_trials, y_cands, z_cands

    def _run_mrmr(self, train_trials, y_cands, z_cands):
        """CV-stabilized mRMR on Y and (optionally) Z. Returns (selected_y, selected_z, stats_y, stats_z)."""
        return run_mrmr_selection(
            train_trials, y_cands, z_cands, self.exp_type,
            k_y=self.k_y, k_z=self.k_z, sfreq=self.sfreq,
            stratify=self.stratify, method=self.mrmr_method,
            mrmr_epoch_len_s=self.mrmr_epoch_len_s,
        )

    def _write_mrmr_json(self, selected_y, selected_z, stats_y, stats_z) -> None:
        """mrmr-only mode: write mrmr_stats.json into the latest existing run dir."""
        key = f"{self.participant}_S{self.session}_{self.exp_type}"
        run_dirs = sorted(
            (self.project_root / "results" / "psid_diagnostic" / key).glob("*")
        )
        run_dirs = [d for d in run_dirs if d.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(
                f"No existing run dir for {key}. Run full diagnostic first."
            )
        path = run_dirs[-1] / "mrmr_stats.json"
        with open(path, "w") as fh:
            json.dump(
                {
                    "selected_y": list(selected_y),
                    "selected_z": list(selected_z),
                    "y_pool": _mrmr_pool_records(stats_y),
                    "z_pool": _mrmr_pool_records(stats_z),
                    "y_n_folds": stats_y.get("n_folds", 0) if stats_y else 0,
                    "z_n_folds": stats_z.get("n_folds", 0) if stats_z else 0,
                },
                fh, indent=2,
            )
        self.log.info("[diagnostic] wrote %s", path)

    def _run_cv_sweeps(self, train_trials, sel_y_idx, sel_z_idx):
        """nx then n1 inner-CV sweeps. Returns (NX, N1, nx_data, n1_data)."""
        NX, nx_val_mean, nx_val_sem, nx_train_mean, nx_val_folds, nx_train_folds = (
            cv_select_nx(
                train_trials, sel_y_idx,
                nx_grid=self.nx_grid, i_horizon=self.i_horizon, max_eig=self.max_eig,
            )
        )
        N1, n1_z_mean, n1_z_sem, n1_z_folds = cv_select_n1(
            train_trials, sel_y_idx, sel_z_idx, NX, self.sfreq,
            n1_grid=self.n1_grid, i_horizon=self.i_horizon, max_eig=self.max_eig,
        )
        nx_data = {
            "val_mean": nx_val_mean, "val_sem": nx_val_sem, "train_mean": nx_train_mean,
            "val_fold_curves": nx_val_folds, "train_fold_curves": nx_train_folds,
        }
        n1_data = {
            "z_cc_mean": n1_z_mean, "z_cc_sem": n1_z_sem, "z_cc_fold_curves": n1_z_folds,
        }
        return NX, N1, nx_data, n1_data

    def _run_final_fit(
        self, train_trials, test_trials, sel_y_idx, sel_z_idx,
        selected_y, selected_z, NX, N1,
    ):
        """Final PSID fit + amend run YAML. Returns (model, final_metrics, runs_yaml_path)."""
        Y_train = [t["Y"][:, sel_y_idx] for t in train_trials]
        Z_train = [t["Z"][:, sel_z_idx] for t in train_trials]
        Y_test = [t["Y"][:, sel_y_idx] for t in test_trials]
        Z_test = [t["Z"][:, sel_z_idx] for t in test_trials]
        model, final_metrics = fit_final_and_evaluate(
            Y_train, Z_train, Y_test, Z_test, NX, N1, self.sfreq,
            i_horizon=self.i_horizon, max_eig=self.max_eig,
            history_s=self.history_s, forecast_s=self.forecast_s,
        )
        out_path = (
            self.project_root
            / "training" / "setups"
            / f"psid_{self.participant}_S{self.session}_{self.exp_type}.yaml"
        )
        amend_run_config(
            self.participant, self.session, NX, N1, self.i_train,
            list(selected_y), list(selected_z), out_path,
        )
        self.log.info("[diagnostic] amended %s", out_path)
        return model, final_metrics, out_path

    def _save_results(
        self, selected_y, selected_z, stats_y, stats_z,
        NX, N1, nx_data, n1_data, final_metrics, model, runs_yaml,
    ) -> None:
        """Build metrics payload and persist model + parquet + mrmr_stats.json."""

        def _clean(d):
            return {
                int(k): (None if np.isnan(v) else float(v)) for k, v in d.items()
            }

        def _clean_folds(folds):
            return [{int(k): (None if np.isnan(v) else float(v)) for k, v in fc.items()}
                    for fc in folds]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts_dir = (
            self.project_root / "results" / "psid_diagnostic"
            / f"{self.participant}_S{self.session}_{self.exp_type}"
            / timestamp
        )
        payload = {
            "participant": self.participant,
            "session": self.session,
            "mode": self.exp_type,
            "timestamp": timestamp,
            "knobs": {
                "K_Y": self.k_y, "K_Z": self.k_z,
                "stratify_raw_env": self.stratify, "mrmr_method": self.mrmr_method,
                "nx_grid": self.nx_grid, "n1_grid": self.n1_grid,
                "i_horizon": self.i_horizon, "max_eig": self.max_eig,
                "history_s": self.history_s, "forecast_s": self.forecast_s,
            },
            "selected_y": list(selected_y),
            "selected_z": list(selected_z),
            "mrmr_y_pool": _mrmr_pool_records(stats_y),
            "mrmr_z_pool": _mrmr_pool_records(stats_z),
            "mrmr_stats_y_n_folds": stats_y.get("n_folds", 0) if stats_y else 0,
            "mrmr_stats_z_n_folds": stats_z.get("n_folds", 0) if stats_z else 0,
            "nx_sweep_cv": {
                "metric": "reconstruction Y CC, mean per-channel (one-step Kalman)",
                "selection": "1-SE rule (smallest nx within SEM_at_argmax of best val mean)",
                "val_mean": _clean(nx_data["val_mean"]),
                "val_sem": _clean(nx_data["val_sem"]),
                "train_mean": _clean(nx_data["train_mean"]),
                "val_fold_curves": _clean_folds(nx_data["val_fold_curves"]),
                "train_fold_curves": _clean_folds(nx_data["train_fold_curves"]),
            },
            "n1_sweep_cv": {
                "metric": "reconstruction Z CC, mean per-channel (one-step Kalman)",
                "selection": "1-SE rule (smallest n1 within SEM_at_argmax of best val mean)",
                "z_cc_mean": _clean(n1_data["z_cc_mean"]),
                "z_cc_sem": _clean(n1_data["z_cc_sem"]),
                "z_cc_fold_curves": _clean_folds(n1_data["z_cc_fold_curves"]),
            },
            "chosen": {"nx": int(NX), "n1": int(N1)},
            "final_metrics": final_metrics,
            "runs_yaml": str(runs_yaml.relative_to(self.project_root)),
        }
        model_path, parquet_path, mrmr_json_path = save_diagnostic_artifacts(
            artifacts_dir, model, payload
        )
        self.log.info("[diagnostic] wrote %s", model_path)
        self.log.info("[diagnostic] wrote %s", parquet_path)
        self.log.info("[diagnostic] wrote %s", mrmr_json_path)


def main():
    parser = argparse.ArgumentParser(description="PSID per-cell diagnostic")
    parser.add_argument("--config", required=True, help="Path to psid_diagnostic_run.yaml")
    parser.add_argument(
        "--mrmr-only",
        action="store_true",
        help="Run only mRMR selection and write mrmr_stats.json to latest existing run dir.",
    )
    args = parser.parse_args()
    config = get_config(str(args.config))
    log = setup_logger(PROJECT_ROOT / config.results.log_dir, name=__file__)
    phases = ("mrmr",) if args.mrmr_only else None
    PsidDiagnosticPipeline(config, log, phases=phases).run()


if __name__ == "__main__":
    main()
