"""sklearn pipeline + fixed-param CV loop.

LDA has no meaningful hyperparameters to tune in this pipeline — the config's
``param_grid`` always has a single combination (solver=lsqr, shrinkage=auto).
So instead of ``GridSearchCV`` we just fit the pipeline across
``ChronoGroupsSplit`` folds, collect balanced accuracy per fold, and re-fit on
the full training pool for the final ``best_pipeline``.
"""

from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .splits import ChronoGroupsSplit


def _reorder_dims_for_mne(X: np.ndarray) -> np.ndarray:
    """(n_epochs, n_samples, n_channels) -> (n_epochs, n_channels, n_samples) for MNE CSP."""
    return np.transpose(X, (0, 2, 1))


def create_pipeline(fs: float = 80, feature_source: str = "Xp") -> Pipeline:
    """LDA decoding pipeline: transpose -> CSP(4) -> scale -> LDA."""
    from mne.decoding import CSP

    steps = [
        ("transpose", FunctionTransformer(_reorder_dims_for_mne)),
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True)),
        ("scaler", StandardScaler()),
        ("classifier", LinearDiscriminantAnalysis()),
    ]
    return Pipeline(steps)


def _flatten_fixed_params(param_grid: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn a config param_grid (values wrapped in single-element lists) into
    plain kwargs for ``Pipeline.set_params``."""
    if not param_grid:
        return {}
    out: Dict[str, Any] = {}
    for key, values in param_grid.items():
        if isinstance(values, (list, tuple)):
            if len(values) != 1:
                raise ValueError(
                    f"param_grid[{key!r}] has {len(values)} values; expected exactly 1 "
                    "because we no longer run a hyperparameter search."
                )
            out[key] = values[0]
        else:
            out[key] = values
    return out


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    fs: float = 80,
    param_grid: Optional[Dict[str, Any]] = None,
    feature_source: str = "Xp",
    allow_mixed_label_groups: bool = False,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """Block-aware chronological CV with fixed LDA params.

    Runs ``ChronoGroupsSplit`` folds, records balanced accuracy per fold, then
    fits a final pipeline on all (X, y). Result dict mirrors the structure the
    rest of the codebase consumes (``best_pipeline``, ``fold_results``,
    ``best_cv_score``, aggregate metrics).
    """
    pipeline = create_pipeline(fs=fs, feature_source=feature_source)
    fixed_params = _flatten_fixed_params(param_grid)
    if fixed_params:
        pipeline.set_params(**fixed_params)

    chrono_cv = ChronoGroupsSplit(
        warn_if_blocks_ignored=True,
        allow_mixed_label_groups=allow_mixed_label_groups,
    )
    splits = chrono_cv.split(X, y, groups)
    effective_n_splits = len(splits)

    fold_scores = []
    fold_results = []
    cv_y_true_parts: list = []
    cv_y_pred_parts: list = []
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
        )
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_pipeline = clone(pipeline)
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            fold_pipeline.fit(X_train, y_train)
            y_pred_fold = fold_pipeline.predict(X_val)
            fold_score = balanced_accuracy_score(y_val, y_pred_fold)
            fold_scores.append(fold_score)
            cv_y_true_parts.append(y_val)
            cv_y_pred_parts.append(y_pred_fold)
            fold_results.append(
                {
                    "fold": fold_idx,
                    "train_indices": (int(train_idx.min()), int(train_idx.max())),
                    "val_indices": (int(val_idx.min()), int(val_idx.max())),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_on_train": int(np.sum(y_train == 1)),
                    "n_off_train": int(np.sum(y_train == 0)),
                    "n_on_val": int(np.sum(y_val == 1)),
                    "n_off_val": int(np.sum(y_val == 0)),
                    "accuracy": float(accuracy_score(y_val, y_pred_fold)),
                    "balanced_accuracy": float(fold_score),
                }
            )

        best_pipeline = clone(pipeline)
        best_pipeline.fit(X, y)

    best_cv_score = float(np.mean(fold_scores)) if fold_scores else float("nan")

    y_pred = best_pipeline.predict(X)
    y_proba = (
        best_pipeline.predict_proba(X)[:, 1]
        if hasattr(best_pipeline, "predict_proba")
        else y_pred
    )
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc_val = auc(fpr, tpr)

    cv_y_true = (
        np.concatenate(cv_y_true_parts)
        if cv_y_true_parts
        else np.array([], dtype=np.int64)
    )
    cv_y_pred = (
        np.concatenate(cv_y_pred_parts)
        if cv_y_pred_parts
        else np.array([], dtype=np.int64)
    )

    results = {
        "best_params": fixed_params,
        "best_cv_score": best_cv_score,
        "n_splits": effective_n_splits,
        "cv_method": "ChronoGroupsSplit",
        "fold_results": fold_results,
        "cv_y_true": cv_y_true,
        "cv_y_pred": cv_y_pred,
        "y_true": y,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=[0, 1]),
        "roc_auc": roc_auc_val,
        "fpr": fpr,
        "tpr": tpr,
        "best_pipeline": best_pipeline,
    }
    return fixed_params, best_cv_score, results
