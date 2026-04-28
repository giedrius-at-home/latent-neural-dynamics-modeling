"""Test-set evaluation, permutation test, and LDA orchestration helpers.

- ``evaluate_on_test_set``: full metric dict + class-balanced subset variant
  (min(n_on, n_off) per class, deterministic seed).
- ``run_permutation_test``: group-aware label shuffle over CV folds. Preserves
  class marginals since it permutes the per-group label multiset.
- ``run_lda_fit_and_test_only`` / ``apply_lda_permutation_to_results`` /
  ``run_classification_pipeline``: top-level orchestration called by
  ``classification/compute.py``.
"""
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.base import clone
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
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from .pipeline import run_cv
from .splits import ChronoGroupsSplit


def _balanced_test_subset(
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic min-class subsample so reported metrics reflect
    discrimination rather than majority-class prediction bias."""
    rng = np.random.RandomState(seed)
    y_arr = np.asarray(y_test)
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        return np.arange(len(y_arr)), y_arr
    n_min = int(counts.min())
    idx_keep = []
    for c in classes:
        idx_c = np.where(y_arr == c)[0]
        idx_keep.append(rng.choice(idx_c, size=n_min, replace=False))
    idx = np.sort(np.concatenate(idx_keep))
    return idx, y_arr[idx]


def evaluate_on_test_set(
    best_pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Full test metrics + class-balanced subset report when test is imbalanced."""
    y_pred = best_pipeline.predict(X_test)
    y_proba = (
        best_pipeline.predict_proba(X_test)[:, 1]
        if hasattr(best_pipeline, "predict_proba")
        else y_pred
    )

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)

    result = {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "roc_auc": roc_auc_val,
        "fpr": fpr,
        "tpr": tpr,
    }

    # Balanced subset — written only when test is actually imbalanced.
    idx_bal, y_bal = _balanced_test_subset(X_test, y_test, seed=0)
    if len(idx_bal) > 0 and len(idx_bal) < len(y_test):
        y_pred_bal = y_pred[idx_bal]
        y_proba_bal = y_proba[idx_bal]
        fpr_b, tpr_b, _ = roc_curve(y_bal, y_proba_bal)
        result["balanced_test_subset"] = {
            "n_per_class": int(len(idx_bal) // 2),
            "accuracy": accuracy_score(y_bal, y_pred_bal),
            "balanced_accuracy": balanced_accuracy_score(y_bal, y_pred_bal),
            "precision": precision_score(y_bal, y_pred_bal, zero_division=0),
            "recall": recall_score(y_bal, y_pred_bal, zero_division=0),
            "f1": f1_score(y_bal, y_pred_bal, zero_division=0),
            "confusion_matrix": confusion_matrix(y_bal, y_pred_bal, labels=[0, 1]),
            "roc_auc": auc(fpr_b, tpr_b),
        }
    return result


def run_permutation_test(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: Any,
    n_permutations: int = 100,
    scoring: str = "balanced_accuracy",
    logger: Optional[Any] = None,
) -> Tuple[float, np.ndarray, float]:
    """Group-level label shuffle null distribution.

    Labels are permuted across groups (one label per group), so the per-class
    marginal count is preserved within each permutation — this makes the null
    a "no class-discriminative signal" null, not a "no signal at all" null.
    """
    if logger:
        logger.info(
            f"Running group-aware permutation test ({n_permutations} permutations)..."
        )

    obs_scores = cross_val_score(
        pipeline, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=2
    )
    obs_mean_score = float(np.mean(obs_scores))

    unique_groups = np.unique(groups)
    group_to_label = {g: y[groups == g][0] for g in unique_groups}
    group_ids = list(group_to_label.keys())
    labels = np.array([group_to_label[g] for g in group_ids])

    def run_one_permutation(seed):
        rng = np.random.RandomState(seed)
        perm_labels = rng.permutation(labels)
        perm_map = dict(zip(group_ids, perm_labels))
        y_perm = np.array([perm_map[g] for g in groups])
        p_scores = cross_val_score(
            clone(pipeline), X, y_perm, cv=cv, groups=groups,
            scoring=scoring, n_jobs=1,
        )
        return float(np.mean(p_scores))

    permutation_scores = np.array(
        joblib.Parallel(n_jobs=8)(
            joblib.delayed(run_one_permutation)(i) for i in range(n_permutations)
        )
    )

    pvalue = (np.sum(permutation_scores >= obs_mean_score) + 1) / (n_permutations + 1)
    if logger:
        logger.info(
            f"Permutation Test - Obs: {obs_mean_score:.4f}, p-value: {pvalue:.4f}"
        )
    return obs_mean_score, permutation_scores, pvalue


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_lda_fit_and_test_only(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    config: Any,
    logger: Any,
    feature_source: str = "Xp",
) -> Dict[str, Any]:
    """Block-aware CV on train, then evaluate on test if provided."""
    n_splits = config.classification.n_splits
    sampling_freq = config.classification.sampling_freq
    param_grid = config.classification.get("param_grid", {}).get("LDA")
    flipped = config.classification.get("flipped", False)

    _, best_score, results = run_cv(
        X_train,
        y_train,
        groups_train,
        n_splits,
        sampling_freq,
        param_grid=param_grid,
        feature_source=feature_source,
        allow_mixed_label_groups=flipped,
    )
    logger.info(f"  LDA Balanced CV Score: {best_score:.4f}")

    if X_test is not None and y_test is not None:
        test_res = evaluate_on_test_set(results["best_pipeline"], X_test, y_test)
        results["test_results"] = test_res
        logger.info(
            f"  LDA Test Balanced Acc: {test_res['balanced_accuracy']:.4f}"
        )

    return results


def apply_lda_permutation_to_results(
    results: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    config: Any,
    logger: Any,
) -> None:
    results.pop("permutation_test", None)
    n_permutations = config.classification.get("n_permutations", 100)
    flipped = config.classification.get("flipped", False)
    chrono_cv = ChronoGroupsSplit(allow_mixed_label_groups=flipped)
    p_mean, p_scores, p_val = run_permutation_test(
        results["best_pipeline"],
        X_train,
        y_train,
        groups_train,
        chrono_cv,
        n_permutations=n_permutations,
        logger=logger,
    )
    results["permutation_test"] = {
        "score": p_mean,
        "pvalue": p_val,
        "n_permutations": n_permutations,
        "scores": np.asarray(p_scores, dtype=float).tolist(),
    }


def run_classification_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    config: Any,
    logger: Any,
    feature_source: str = "Xp",
) -> Dict[str, Any]:
    results = run_lda_fit_and_test_only(
        X_train, y_train, groups_train,
        X_test, y_test,
        config, logger,
        feature_source=feature_source,
    )
    if config.classification.get("permutation_test", False):
        apply_lda_permutation_to_results(
            results, X_train, y_train, groups_train, config, logger
        )
    return results
