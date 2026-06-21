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

import joblib
from joblib import Parallel, delayed
import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score

from utils.classification import (
    ChronoGroupsSplit,
    create_pipeline,
    load_all_splits,
    load_forecast_splits_precomputed,
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


def _one_permutation(
    perm_idx: int,
    X: np.ndarray,
    unique_groups: np.ndarray,
    grp_label: Dict[int, int],
    groups: np.ndarray,
    fs: int,
    base_seed: int,
) -> float:
    """Single permutation iteration — module-level for joblib pickling."""
    rng = np.random.default_rng(base_seed + perm_idx)
    permuted = rng.permutation(unique_groups)
    new_label = dict(zip(unique_groups.tolist(), [grp_label[int(g)] for g in permuted]))
    y_perm = np.array([new_label[int(g)] for g in groups], dtype=np.int64)
    chrono = ChronoGroupsSplit(allow_mixed_label_groups=False, warn_if_blocks_ignored=False)
    pipe = create_pipeline(fs=fs)
    pipe.set_params(classifier__solver="lsqr", classifier__shrinkage="auto")
    fold_bas: List[float] = []
    for tr, va in chrono.split(X, y_perm, groups):
        fold_pipe = clone(pipe)
        fold_pipe.fit(X[tr], y_perm[tr])
        pred = fold_pipe.predict(X[va])
        fold_bas.append(balanced_accuracy_score(y_perm[va], pred))
    return float(np.mean(fold_bas)) if fold_bas else 0.5


def _permutation_pvalue(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_ba: float,
    n_permutations: int,
    fs: int,
    rng: np.random.Generator,
    n_jobs: int = 4,
) -> Tuple[float, float, List[float]]:
    """Group-shuffled labels (chrono CV) -> null distribution; returns (p_value, null_mean_ba, null_scores)."""
    unique_groups = np.unique(groups)
    grp_label = {int(g): int(y[groups == g][0]) for g in unique_groups}
    base_seed = int(rng.integers(0, 2**31))
    null_scores: List[float] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_permutation)(
            i, X, unique_groups, grp_label, groups, fs, base_seed
        )
        for i in range(n_permutations)
    )
    null = np.asarray(null_scores)
    p = float((np.sum(null >= cv_ba) + 1) / (n_permutations + 1))
    return p, float(np.mean(null)), null_scores


def _one_permutation_flipped(
    perm_idx: int,
    X: np.ndarray,
    unique_groups: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    fs: int,
    base_seed: int,
) -> float:
    """Single flipped permutation — shuffle model-identity labels within each group."""
    rng = np.random.default_rng(base_seed + perm_idx)
    y_perm = y.copy()
    for g in unique_groups:
        mask = groups == g
        y_perm[mask] = rng.permutation(y[mask])
    chrono = ChronoGroupsSplit(allow_mixed_label_groups=True, warn_if_blocks_ignored=False)
    pipe = create_pipeline(fs=fs)
    pipe.set_params(classifier__solver="lsqr", classifier__shrinkage="auto")
    fold_bas: List[float] = []
    for tr, va in chrono.split(X, y_perm, groups):
        fold_pipe = clone(pipe)
        fold_pipe.fit(X[tr], y_perm[tr])
        pred = fold_pipe.predict(X[va])
        fold_bas.append(balanced_accuracy_score(y_perm[va], pred))
    return float(np.mean(fold_bas)) if fold_bas else 0.5


def _permutation_pvalue_flipped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_ba: float,
    n_permutations: int,
    fs: int,
    rng: np.random.Generator,
    n_jobs: int = 4,
) -> Tuple[float, float, List[float]]:
    """Within-group label shuffle -> null distribution for flipped classification."""
    unique_groups = np.unique(groups)
    base_seed = int(rng.integers(0, 2**31))
    null_scores: List[float] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_permutation_flipped)(
            i, X, unique_groups, groups, y, fs, base_seed
        )
        for i in range(n_permutations)
    )
    null = np.asarray(null_scores)
    p = float((np.sum(null >= cv_ba) + 1) / (n_permutations + 1))
    return p, float(np.mean(null)), null_scores


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
    precomputed_train: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    precomputed_scores: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[List[Dict[str, Any]], Any]:
    """Fit one LDA on the full window, score at each entry of ``score_grid``.

    precomputed_train: (X_base, y, groups) before sub-source slicing — skips
        prepare_epoched_data for the training step.
    precomputed_scores: label -> (X_base, y) before slicing — skips
        prepare_epoched_data for each matching score_grid entry.
    """
    base = _base_feature(feature_source)
    sfreq = int(sampling_freq)

    if precomputed_train is not None:
        X_full_base, y_full, g_full = precomputed_train
    else:
        X_full_base, y_full, g_full, _ = prepare_epoched_data(
            pool_trials,
            feature_source=base,
            epoch_length_sec=classifier_cfg.epoch_length,
            overlap=classifier_cfg.epoch_overlap,
            fs=sfreq,
            n1=n1,
            nx=nx,
            **train_kwargs,
        )
    if X_full_base is None or len(X_full_base) == 0:
        raise RuntimeError(
            f"prepare_epoched_data produced no training epochs for "
            f"feature={base} kwargs={train_kwargs}"
        )
    X_full = _slice_subsource(X_full_base, y_full, feature_source, n1=n1, nx=nx)

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
    cv_y_true: List[int] = res.get("cv_y_true", np.array([], dtype=np.int64)).tolist()
    cv_y_pred: List[int] = res.get("cv_y_pred", np.array([], dtype=np.int64)).tolist()
    cv_y_proba: List[float] = res.get("cv_y_proba", np.array([], dtype=np.float64)).tolist()
    y_proba: List[float] = res.get("y_proba", np.array([], dtype=np.float64)).tolist()
    cv_roc_auc: float = float(res.get("roc_auc", float("nan")))
    cv_fold_ba: List[float] = [
        float(f["balanced_accuracy"]) for f in res.get("fold_results", [])
    ]

    rows: List[Dict[str, Any]] = []
    for label, _ in score_grid:
        if precomputed_scores is None or label not in precomputed_scores:
            continue
        X_score_base, y_score = precomputed_scores[label]
        if X_score_base is None or len(X_score_base) == 0:
            continue
        X_score = _slice_subsource(X_score_base, y_score, feature_source, n1=n1, nx=nx)
        pred = clf.predict(X_score)
        ba = balanced_accuracy_score(y_score, pred)
        rows.append(
            {
                "score_label": label,
                "n_score": int(len(y_score)),
                "ba": float(ba),
                "y_true": y_score.tolist(),
                "y_pred": pred.tolist(),
            }
        )

    p_value: float = float("nan")
    perm_mean_ba: float = float("nan")
    perm_scores: List[float] = []
    if cv_ba > classifier_cfg.perm_ba_gate:
        log.info(
            f"    cv_ba={cv_ba:.3f} > gate; running {classifier_cfg.n_permutations} perms"
        )
        if allow_mixed_label_groups:
            p_value, perm_mean_ba, perm_scores = _permutation_pvalue_flipped(
                X_full,
                y_full,
                g_full,
                cv_ba,
                classifier_cfg.n_permutations,
                sfreq,
                rng,
            )
        else:
            p_value, perm_mean_ba, perm_scores = _permutation_pvalue(
                X_full,
                y_full,
                g_full,
                cv_ba,
                classifier_cfg.n_permutations,
                sfreq,
                rng,
            )
    for r in rows:
        r["cv_ba"] = float(cv_ba)
        r["cv_y_true"] = cv_y_true
        r["cv_y_pred"] = cv_y_pred
        r["cv_y_proba"] = cv_y_proba
        r["y_proba"] = y_proba
        r["cv_roc_auc"] = cv_roc_auc
        r["cv_fold_ba"] = cv_fold_ba
        r["p_value"] = p_value
        r["perm_mean_ba"] = perm_mean_ba
        r["perm_scores"] = perm_scores
        r["n_permutations"] = (
            0 if np.isnan(p_value) else int(classifier_cfg.n_permutations)
        )
    return rows, clf


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
    clf_dir: Optional[Path] = None,
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

        _common = dict(
            feature_source="Xp",
            epoch_length_sec=classifier_cfg.epoch_length,
            overlap=classifier_cfg.epoch_overlap,
            fs=sfreq,
            n1=n1,
            nx=nx,
            mode="prediction",
        )
        X_train_base, y_train, g_train, _ = prepare_epoched_data(pool_full, **_common)
        pre_scores_pred: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for t_cut in t_cut_grid:
            t_samples = int(round(t_cut * sfreq))
            trimmed = _truncate_split_Xp(pool_split_dicts, t_samples)
            X_s, y_s, _, _ = prepare_epoched_data(list(trimmed.values()), **_common)
            if X_s is not None and len(X_s) > 0:
                pre_scores_pred[f"t={t_cut:g}"] = (X_s, y_s)

        score_grid_pred: List[Tuple[str, Dict[str, Any]]] = [
            (f"t={t_cut:g}", {"mode": "prediction"}) for t_cut in t_cut_grid
        ]

        for sub in feature_sources:
            log.info(f"predictions  dbs={dbs_train} sub={sub}")
            rows, clf = _train_then_score(
                train_kwargs={"mode": "prediction"},
                score_grid=score_grid_pred,
                pool_trials=pool_full,
                feature_source=sub,
                n1=n1,
                nx=nx,
                classifier_cfg=classifier_cfg,
                sampling_freq=sfreq,
                rng=rng,
                log=log,
                precomputed_train=(
                    (X_train_base, y_train, g_train)
                    if X_train_base is not None
                    else None
                ),
                precomputed_scores=pre_scores_pred,
            )
            if clf_dir is not None:
                clf_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(clf, clf_dir / f"clf_pred_{dbs_train}_{sub}.joblib")
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
                        "cv_y_true": r["cv_y_true"],
                        "cv_y_pred": r["cv_y_pred"],
                        "cv_y_proba": r["cv_y_proba"],
                        "y_proba": r["y_proba"],
                        "cv_roc_auc": r["cv_roc_auc"],
                        "cv_fold_ba": r["cv_fold_ba"],
                        "ba_at_score": r["ba"],
                        "n_score": r["n_score"],
                        "n_permutations": r["n_permutations"],
                        "p_value": r["p_value"],
                        "perm_mean_ba": r["perm_mean_ba"],
                        "perm_scores": r["perm_scores"],
                        "y_true": r["y_true"],
                        "y_pred": r["y_pred"],
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
    clf_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """One LDA per (h, sub-source, dbs-train); score at every m_test."""
    sfreq = sampling_freq
    rng = np.random.default_rng(0)
    n1 = config.framework.params.n1
    nx = config.framework.params.nx
    out: List[Dict[str, Any]] = []

    # Flipped models are the same for every dbs_train iteration — load once.
    flip_models: Dict[str, Any] = {}
    if flipped:
        for side in ("on", "off", "both"):
            v_dir = project_root / "results" / pipeline / variants[side]
            fw = _load_framework_for_forecast(
                v_dir, timestamps[side], project_root, config
            )
            flip_models[side] = fw.model.idSys

    # DPAD imputes Xf from full Y_past, so only the dbs=both model is valid for
    # forecast classification — on/off models were trained on a single condition
    # and cannot meaningfully forecast from the opposing condition's observations.
    _dbs_states = ("both",) if pipeline.startswith("dpad") else ("both", "on", "off")
    for dbs_train in _dbs_states:
        variant = variants[dbs_train]
        run_ts = timestamps[dbs_train]
        variant_dir = project_root / "results" / pipeline / variant

        # Flipped needs inference splits (raw Y observations for re-forecasting).
        # Non-flipped uses pre-computed forecast parquets — no model loading needed.
        if flipped:
            inf_dir = variant_dir / "inference"
            splits = load_all_splits(inf_dir, run_ts)
            if not splits:
                raise FileNotFoundError(f"no inference parquets under {inf_dir}")
            pool_for_flipped = [splits[k] for k in ("train", "val") if k in splits]

        for h in h_grid:
            _common = dict(
                feature_source="Xf",
                epoch_length_sec=classifier_cfg.epoch_length,
                overlap=classifier_cfg.epoch_overlap,
                fs=sfreq,
                n1=n1,
                nx=nx,
            )

            if flipped:
                base_kwargs: Dict[str, Any] = {
                    "mode": "forecast",
                    "history_horizon": float(h),
                    "forecast_horizon": float(m_seconds),
                    "framework": None,
                    "model_on": flip_models["on"],
                    "model_off": flip_models["off"],
                    "model_both": flip_models["both"],
                    "target_future": True,
                }
                pool_h = pool_for_flipped
            else:
                # Load pre-computed X_future_pred — avoids re-running the model.
                # Missing h dirs (e.g. DPAD h3/h4) return None; skip gracefully.
                precomp = load_forecast_splits_precomputed(variant_dir, h)
                pool_h = [v for k, v in precomp.items() if k in ("train", "val") and v is not None]
                if not pool_h:
                    log.info(f"forecast  dbs={dbs_train} h={h:g}: no pre-computed parquets, skipping")
                    continue
                base_kwargs = {
                    "mode": "forecast",
                    "history_horizon": None,  # signals pre-computed path in prepare_epoched_data
                    "forecast_horizon": float(m_seconds),
                    "framework": None,
                }

            X_train_base, y_train, g_train, _ = prepare_epoched_data(
                pool_h, **_common, **base_kwargs
            )
            pre_scores: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for m_test in m_test_grid:
                kw_s = {**base_kwargs, "forecast_horizon": float(m_test)}
                X_s, y_s, _, _ = prepare_epoched_data(pool_h, **_common, **kw_s)
                if X_s is not None and len(X_s) > 0:
                    pre_scores[f"m_test={m_test:g}"] = (X_s, y_s)

            score_grid: List[Tuple[str, Dict[str, Any]]] = [
                (
                    f"m_test={m_test:g}",
                    {**base_kwargs, "forecast_horizon": float(m_test)},
                )
                for m_test in m_test_grid
            ]

            for sub in feature_sources:
                log.info(
                    f"forecast{' (flipped)' if flipped else ''}  dbs={dbs_train} sub={sub} h={h:.1f}"
                )
                rows, clf = _train_then_score(
                    train_kwargs=base_kwargs,
                    score_grid=score_grid,
                    pool_trials=pool_h,
                    feature_source=sub,
                    n1=n1,
                    nx=nx,
                    classifier_cfg=classifier_cfg,
                    sampling_freq=sfreq,
                    rng=rng,
                    log=log,
                    allow_mixed_label_groups=flipped,
                    precomputed_train=(
                        (X_train_base, y_train, g_train)
                        if X_train_base is not None
                        else None
                    ),
                    precomputed_scores=pre_scores,
                )
                if clf_dir is not None:
                    clf_dir.mkdir(parents=True, exist_ok=True)
                    tag = "flipped" if flipped else "forecast"
                    joblib.dump(
                        clf, clf_dir / f"clf_{tag}_{dbs_train}_{sub}_h{h:.1f}.joblib"
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
                            "cv_y_true": r["cv_y_true"],
                            "cv_y_pred": r["cv_y_pred"],
                            "cv_y_proba": r["cv_y_proba"],
                            "y_proba": r["y_proba"],
                            "cv_roc_auc": r["cv_roc_auc"],
                            "cv_fold_ba": r["cv_fold_ba"],
                            "ba_at_score": r["ba"],
                            "n_score": r["n_score"],
                            "n_permutations": r["n_permutations"],
                            "p_value": r["p_value"],
                            "perm_mean_ba": r["perm_mean_ba"],
                            "perm_scores": r["perm_scores"],
                            "y_true": r["y_true"],
                            "y_pred": r["y_pred"],
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
    cls_mode: str = "all",
) -> Path:
    """Run all sweeps and write parquet to out_dir. Returns the written path.

    cls_mode:
        "all"           — predictions + forecast + flipped (default)
        "forecast_only" — skip predictions sweep, run forecast + flipped only
    """
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
    clf_dir = out_dir / "classifiers"

    rows: List[Dict[str, Any]] = []
    if cls_mode != "forecast_only":
        log.info("predictions sweep ...")
        rows.extend(
            run_predictions_sweep(
                **common,
                feature_sources=feature_sources_pred,
                t_cut_grid=t_cut_grid,
                clf_dir=clf_dir,
            )
        )
    else:
        log.info("predictions sweep skipped (cls_mode=forecast_only)")
    log.info("forecast sweep ...")
    rows.extend(
        run_forecast_sweep(**common, **forecast_kwargs, flipped=False, clf_dir=clf_dir)
    )
    if not pipeline.startswith("dpad"):
        log.info("forecast sweep (flipped) ...")
        rows.extend(
            run_forecast_sweep(**common, **forecast_kwargs, flipped=True, clf_dir=clf_dir)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"sweep_{ts}.parquet"
    pl.DataFrame(rows, infer_schema_length=len(rows)).write_parquet(out_path)
    log.info("wrote %d rows -> %s", len(rows), out_path)
    return out_path
