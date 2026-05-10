"""Predictions truncation sweep + forecast h-sweep classifier.

Predictions sweep (one classifier, many test points)
    Train 1 LDA per (sub-source, dbs-train) on full Xp epoched signal.
    Score each at t in ``t_cut_grid`` by trimming Xp to the first
    ``t * sfreq`` samples before epoching — no padding.

Forecast h-sweep (one classifier per h, many m_test points)
    For each h in ``h_grid`` train an LDA on Xf produced from h s of history.
    Score at each m_test by re-forecasting at m_test (deterministic for all
    frameworks with fixed weights).

Flipped variant (forecast only)
    For each h: run on- and off-model forecasts on the same Y window;
    label by model identity.

Permutation gate
    For any classifier with CV BA > ``perm_ba_gate``, run ``n_permutations``
    group-shuffled label permutations.

Output
    Flat parquet at ``<out_dir>/sweep_<ts>.parquet``: one row per
    (mode, sub_source, flipped, dbs_train, t_cut|h, m_test).

Entry point: ``training/pipeline.py --phases classification``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

from utils.classification import (
    ChronoGroupsSplit,
    create_pipeline,
    load_all_splits,
    prepare_epoched_data,
    run_cv,
    _load_framework_for_forecast,
)


# --- helpers ---


def _base_feature(sub_source: str) -> str:
    if sub_source.startswith("Xf"):
        return "Xf"
    return "Xp"


def _slice_subsource(
    X: np.ndarray, y: np.ndarray, sub_source: str, n1: int, nx: int
) -> np.ndarray:
    """Slice (n_epochs, n_time, n_channels) by sub-source suffix.

    _1 -> first n1 channels, _2 -> n1:nx channels,
    _with_dbs -> append DBS indicator channel, no suffix -> passthrough.
    """
    if sub_source.endswith("_1"):
        return X[:, :, :n1]
    if sub_source.endswith("_2"):
        return X[:, :, n1:nx]
    if sub_source.endswith("_with_dbs"):
        dbs = (y.astype(np.float64).reshape(-1, 1, 1)).repeat(X.shape[1], axis=1)
        return np.concatenate([X, dbs], axis=-1)
    return X


def _truncate_split_Xp(
    splits: Dict[str, Dict[str, Any]], t_cut_samples: int
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, split in splits.items():
        if split is None:
            continue
        new_split = dict(split)
        new_xp = []
        for xp in split.get("Xp") or []:
            if xp is None:
                new_xp.append(None)
                continue
            arr = np.asarray(xp)
            new_xp.append(arr[:t_cut_samples])
        new_split["Xp"] = new_xp
        out[name] = new_split
    return out


# --- permutation ---


def _permutation_pvalue(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_ba: float,
    n_permutations: int,
    fs: int,
    rng: np.random.Generator,
    allow_mixed_label_groups: bool = False,
) -> float:
    """Group-shuffled labels (chrono CV) -> null distribution; one-sided p."""
    chrono = ChronoGroupsSplit(
        allow_mixed_label_groups=allow_mixed_label_groups, warn_if_blocks_ignored=False
    )
    pipe = create_pipeline(fs=fs)
    pipe.set_params(classifier__solver="lsqr", classifier__shrinkage="auto")
    unique_groups = np.unique(groups)
    grp_label = {int(g): int(y[groups == g][0]) for g in unique_groups}
    null_scores: List[float] = []
    for _ in range(n_permutations):
        permuted = rng.permutation(unique_groups)
        new_label = dict(zip(unique_groups, [grp_label[int(g)] for g in permuted]))
        y_perm = np.array([new_label[int(g)] for g in groups], dtype=np.int64)
        chrono_splits = chrono.split(X, y_perm, groups)
        fold_bas: List[float] = []
        for tr, va in chrono_splits:
            fold_pipe = clone(pipe)
            fold_pipe.fit(X[tr], y_perm[tr])
            pred = fold_pipe.predict(X[va])
            fold_bas.append(balanced_accuracy_score(y_perm[va], pred))
        null_scores.append(float(np.mean(fold_bas)) if fold_bas else 0.5)
    null = np.asarray(null_scores)
    return float((np.sum(null >= cv_ba) + 1) / (n_permutations + 1))


# --- core: train-once / score-many ---


def _train_then_score(
    *,
    train_kwargs: Dict[str, Any],
    score_grid: List[Tuple[str, Dict[str, Any]]],
    pool_trials: List[Dict[str, Any]],
    feature_source: str,
    n1: int,
    nx: int,
    classifier_cfg: Any,
    sampling_freq: int,
    rng: np.random.Generator,
    log,
    allow_mixed_label_groups: bool = False,
) -> List[Dict[str, Any]]:
    """Fit one LDA on the full window, score at each entry of ``score_grid``."""
    base = _base_feature(feature_source)
    sfreq = int(sampling_freq)

    X_full, y_full, g_full, _ = prepare_epoched_data(
        pool_trials,
        feature_source=base,
        epoch_length_sec=classifier_cfg.epoch_length,
        overlap=classifier_cfg.epoch_overlap,
        fs=sfreq,
        n1=n1,
        nx=nx,
        **train_kwargs,
    )
    if X_full is None or len(X_full) == 0:
        raise RuntimeError(
            f"prepare_epoched_data produced no training epochs for "
            f"feature={base} kwargs={train_kwargs}"
        )
    X_full = _slice_subsource(X_full, y_full, feature_source, n1=n1, nx=nx)

    _, cv_ba, res = run_cv(
        X_full,
        y_full,
        g_full,
        n_splits=classifier_cfg.n_splits,
        fs=sfreq,
        param_grid=classifier_cfg.param_grid.LDA,
        feature_source=feature_source,
        allow_mixed_label_groups=allow_mixed_label_groups,
    )
    clf = res["best_pipeline"]

    rows: List[Dict[str, Any]] = []
    for label, kwargs in score_grid:
        pool_use = kwargs.pop("_pool_override", pool_trials)
        X_score, y_score, _, _ = prepare_epoched_data(
            pool_use,
            feature_source=base,
            epoch_length_sec=classifier_cfg.epoch_length,
            overlap=classifier_cfg.epoch_overlap,
            fs=sfreq,
            n1=n1,
            nx=nx,
            **kwargs,
        )
        if X_score is None or len(X_score) == 0:
            continue
        X_score = _slice_subsource(X_score, y_score, feature_source, n1=n1, nx=nx)
        pred = clf.predict(X_score)
        ba = balanced_accuracy_score(y_score, pred)
        rows.append(
            {"score_label": label, "n_score": int(len(y_score)), "ba": float(ba)}
        )

    p_value: float = float("nan")
    if cv_ba > classifier_cfg.perm_ba_gate and not allow_mixed_label_groups:
        log.info(
            f"    cv_ba={cv_ba:.3f} > gate; running {classifier_cfg.n_permutations} perms"
        )
        p_value = _permutation_pvalue(
            X_full,
            y_full,
            g_full,
            cv_ba,
            classifier_cfg.n_permutations,
            sfreq,
            rng,
            allow_mixed_label_groups=allow_mixed_label_groups,
        )
    for r in rows:
        r["cv_ba"] = float(cv_ba)
        r["p_value"] = p_value
        r["n_permutations"] = (
            0 if np.isnan(p_value) else int(classifier_cfg.n_permutations)
        )
    return rows


# --- sweeps ---


def run_predictions_sweep(
    *,
    pipeline: str,
    variants: Dict[str, str],
    timestamps: Dict[str, str],
    feature_sources: List[str],
    t_cut_grid: List[float],
    classifier_cfg: Any,
    sampling_freq: int,
    project_root: Path,
    config: Any,
    log,
) -> List[Dict[str, Any]]:
    """1 LDA per (sub-source, dbs-train); score at every t_cut."""
    sfreq = sampling_freq
    rng = np.random.default_rng(0)
    n1 = config.framework.params.n1
    nx = config.framework.params.nx
    out: List[Dict[str, Any]] = []

    for dbs_train in ("both", "on", "off"):
        variant = variants[dbs_train]
        run_ts = timestamps[dbs_train]
        variant_dir = project_root / "results" / pipeline / variant
        inf_dir = variant_dir / "inference"
        splits = load_all_splits(inf_dir, run_ts)
        if not splits:
            raise FileNotFoundError(
                f"no inference parquets under {inf_dir} for run_ts={run_ts}"
            )
        pool_split_dicts = {k: v for k, v in splits.items() if k in ("train", "val")}
        pool_full = list(pool_split_dicts.values())

        for sub in feature_sources:
            log.info(f"predictions  dbs={dbs_train} sub={sub}")
            score_grid: List[Tuple[str, Dict[str, Any]]] = []
            for t_cut in t_cut_grid:
                t_samples = int(round(t_cut * sfreq))
                trimmed = _truncate_split_Xp(pool_split_dicts, t_samples)
                score_grid.append(
                    (
                        f"t={t_cut:g}",
                        {
                            "mode": "prediction",
                            "_pool_override": list(trimmed.values()),
                        },
                    )
                )
            rows = _train_then_score(
                train_kwargs={"mode": "prediction"},
                score_grid=score_grid,
                pool_trials=pool_full,
                feature_source=sub,
                n1=n1,
                nx=nx,
                classifier_cfg=classifier_cfg,
                sampling_freq=sfreq,
                rng=rng,
                log=log,
            )
            for r in rows:
                t_val = float(r["score_label"].split("=")[1])
                out.append(
                    {
                        "mode": "predictions",
                        "pipeline": pipeline,
                        "variant": str(variant),
                        "run_ts": str(run_ts),
                        "dbs_train": dbs_train,
                        "data_dbs": dbs_train,
                        "sub_source": sub,
                        "flipped": False,
                        "t_cut_seconds": t_val,
                        "h_seconds": None,
                        "m_test_seconds": None,
                        "cv_ba": r["cv_ba"],
                        "ba_at_score": r["ba"],
                        "n_score": r["n_score"],
                        "n_permutations": r["n_permutations"],
                        "p_value": r["p_value"],
                    }
                )
    return out


def run_forecast_sweep(
    *,
    pipeline: str,
    variants: Dict[str, str],
    timestamps: Dict[str, str],
    feature_sources: List[str],
    h_grid: List[float],
    m_seconds: float,
    m_test_grid: List[float],
    classifier_cfg: Any,
    sampling_freq: int,
    project_root: Path,
    config: Any,
    log,
    flipped: bool = False,
) -> List[Dict[str, Any]]:
    """One LDA per (h, sub-source, dbs-train); score at every m_test."""
    sfreq = sampling_freq
    rng = np.random.default_rng(0)
    n1 = config.framework.params.n1
    nx = config.framework.params.nx
    out: List[Dict[str, Any]] = []

    for dbs_train in ("both", "on", "off"):
        variant = variants[dbs_train]
        run_ts = timestamps[dbs_train]
        variant_dir = project_root / "results" / pipeline / variant
        inf_dir = variant_dir / "inference"
        splits = load_all_splits(inf_dir, run_ts)
        if not splits:
            raise FileNotFoundError(f"no inference parquets under {inf_dir}")
        pool_full = [splits[k] for k in ("train", "val") if k in splits]

        framework = _load_framework_for_forecast(
            variant_dir, run_ts, project_root, config
        )
        flip_models: Dict[str, Any] = {}
        if flipped:
            for side in ("on", "off", "both"):
                v_dir = project_root / "results" / pipeline / variants[side]
                fw = _load_framework_for_forecast(
                    v_dir, timestamps[side], project_root, config
                )
                flip_models[side] = fw.model.idSys

        for sub in feature_sources:
            for h in h_grid:
                log.info(
                    f"forecast{' (flipped)' if flipped else ''}  dbs={dbs_train} sub={sub} h={h:.1f}"
                )
                train_kwargs: Dict[str, Any] = {
                    "mode": "forecast",
                    "history_horizon": float(h),
                    "forecast_horizon": float(m_seconds),
                    "framework": framework,
                }
                if flipped:
                    train_kwargs.update(
                        {
                            "model_on": flip_models["on"],
                            "model_off": flip_models["off"],
                            "model_both": flip_models["both"],
                            "target_future": True,
                            "framework": None,
                        }
                    )
                score_grid: List[Tuple[str, Dict[str, Any]]] = []
                for m_test in m_test_grid:
                    kw = dict(train_kwargs)
                    kw["forecast_horizon"] = float(m_test)
                    score_grid.append((f"m_test={m_test:g}", kw))

                rows = _train_then_score(
                    train_kwargs=train_kwargs,
                    score_grid=score_grid,
                    pool_trials=pool_full,
                    feature_source=sub,
                    n1=n1,
                    nx=nx,
                    classifier_cfg=classifier_cfg,
                    sampling_freq=sfreq,
                    rng=rng,
                    log=log,
                    allow_mixed_label_groups=flipped,
                )
                for r in rows:
                    m_val = float(r["score_label"].split("=")[1])
                    out.append(
                        {
                            "mode": "forecast",
                            "pipeline": pipeline,
                            "variant": str(variant),
                            "run_ts": str(run_ts),
                            "dbs_train": dbs_train,
                            "data_dbs": dbs_train,
                            "sub_source": sub,
                            "flipped": bool(flipped),
                            "t_cut_seconds": None,
                            "h_seconds": float(h),
                            "m_test_seconds": m_val,
                            "cv_ba": r["cv_ba"],
                            "ba_at_score": r["ba"],
                            "n_score": r["n_score"],
                            "n_permutations": r["n_permutations"],
                            "p_value": r["p_value"],
                        }
                    )
    return out


# --- orchestration ---


def run_sweep(
    *,
    pipeline: str,
    variants: Dict[str, str],
    timestamps: Dict[str, str],
    feature_sources_pred: List[str],
    t_cut_grid: List[float],
    feature_sources_forecast: List[str],
    h_grid: List[float],
    m_seconds: float,
    m_test_grid: List[float],
    classifier_cfg: Any,
    sampling_freq: int,
    project_root: Path,
    out_dir: Path,
    config: Any,
    log,
) -> Path:
    """Run all sweeps and write parquet to out_dir. Returns the written path."""
    common = dict(
        pipeline=pipeline,
        variants=variants,
        timestamps=timestamps,
        classifier_cfg=classifier_cfg,
        sampling_freq=sampling_freq,
        project_root=project_root,
        config=config,
        log=log,
    )
    forecast_kwargs = dict(
        feature_sources=feature_sources_forecast,
        h_grid=h_grid,
        m_seconds=m_seconds,
        m_test_grid=m_test_grid,
    )
    rows: List[Dict[str, Any]] = []
    log.info("predictions sweep ...")
    rows.extend(
        run_predictions_sweep(
            **common,
            feature_sources=feature_sources_pred,
            t_cut_grid=t_cut_grid,
        )
    )
    log.info("forecast sweep ...")
    rows.extend(run_forecast_sweep(**common, **forecast_kwargs, flipped=False))
    log.info("forecast sweep (flipped) ...")
    rows.extend(run_forecast_sweep(**common, **forecast_kwargs, flipped=True))

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sweep_{ts}.parquet"
    pl.DataFrame(rows).write_parquet(out_path)
    log.info("wrote %d rows -> %s", len(rows), out_path)
    return out_path
