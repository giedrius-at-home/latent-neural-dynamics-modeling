"""
Load test-set balanced accuracy from classification LDA pickles for Figure F1.

Each pickle is produced by `classification/compute.py` as `LDA_{feature_source}_prediction.pkl`
under `results/classification/{variant}/{run_ts}/`.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dashboard.thesis.specs import ClassificationF1PickleRef, FeatureGroupF1

GROUP_ORDER: Tuple[FeatureGroupF1, ...] = ("xp", "xp_1", "xp_2", "xp_with_dbs")
GROUP_DISPLAY = {
    "xp": "Xp<br>(full latent)",
    "xp_1": "Xp₁<br>(behav. relevant)",
    "xp_2": "Xp₂<br>(behav. irrelevant)",
    "xp_with_dbs": "Xp + DBS<br>(with state)",
}
GROUP_X = {"xp": 0.0, "xp_1": 1.0, "xp_2": 2.0, "xp_with_dbs": 3.0}


@dataclass(frozen=True)
class ClassificationF1Point:
    participant_label: str
    session_label: str
    group: FeatureGroupF1
    balanced_accuracy: float
    permutation_pvalue: Optional[float]
    model_label: str = "PSID"


def _extract_test_ba_and_perm(res: Dict[str, Any]) -> Tuple[float, Optional[float]]:
    tr = res.get("test_results")
    if isinstance(tr, dict) and tr.get("balanced_accuracy") is not None:
        ba = float(tr["balanced_accuracy"])
    else:
        ba = float(res.get("balanced_accuracy", float("nan")))
    pval = None
    pt = res.get("permutation_test")
    if isinstance(pt, dict) and pt.get("pvalue") is not None:
        try:
            pval = float(pt["pvalue"])
        except (TypeError, ValueError):
            pval = None
    return ba, pval


def load_pickle_metrics(pickle_path: Path) -> Tuple[float, Optional[float]]:
    if not pickle_path.is_file():
        raise FileNotFoundError(f"Classification pickle not found: {pickle_path}")
    with open(pickle_path, "rb") as f:
        res = pickle.load(f)
    if not isinstance(res, dict):
        raise ValueError(f"Expected dict in pickle, got {type(res)}")
    return _extract_test_ba_and_perm(res)


def collect_classification_f1_points(
    results_root: Path,
    refs: Sequence[ClassificationF1PickleRef],
    classification_parent: str = "classification",
) -> List[ClassificationF1Point]:
    """
    Resolve paths as `results_root / classification_parent / pickle_relative_path`.
    """
    base = results_root / classification_parent
    out: List[ClassificationF1Point] = []
    for ref in refs:
        p = base / ref.pickle_relative_path
        ba, pval = load_pickle_metrics(p)
        out.append(
            ClassificationF1Point(
                participant_label=ref.participant_label,
                session_label=ref.session_label,
                group=ref.group,
                balanced_accuracy=ba,
                permutation_pvalue=pval,
                model_label=getattr(ref, "model_label", "PSID"),
            )
        )
    return out


def group_star_flags(
    points: Sequence[ClassificationF1Point],
    alpha: float,
) -> dict[FeatureGroupF1, bool]:
    """True if any point in the group has permutation p-value below alpha."""
    flags = {g: False for g in GROUP_ORDER}
    for pt in points:
        if pt.permutation_pvalue is not None and pt.permutation_pvalue < alpha:
            flags[pt.group] = True
    return flags


# ---------------------------------------------------------------------------
# ROC curve data
# ---------------------------------------------------------------------------

import numpy as _np


@dataclass(frozen=True)
class ClassificationRocCurve:
    participant_label: str
    session_label: str
    group: FeatureGroupF1
    fpr: Any   # np.ndarray
    tpr: Any   # np.ndarray
    roc_auc: float


def collect_classification_roc_curves(
    results_root: Path,
    refs: Sequence[ClassificationF1PickleRef],
    classification_parent: str = "classification",
) -> List[ClassificationRocCurve]:
    """Load per-session ROC curve (fpr, tpr, roc_auc) from test_results in each pickle."""
    base = results_root / classification_parent
    out: List[ClassificationRocCurve] = []
    for ref in refs:
        p = base / ref.pickle_relative_path
        if not p.is_file():
            continue
        with open(p, "rb") as f:
            res = pickle.load(f)
        if not isinstance(res, dict):
            continue
        tr = res.get("test_results") or res
        fpr = tr.get("fpr")
        tpr = tr.get("tpr")
        auc = tr.get("roc_auc", res.get("roc_auc", float("nan")))
        if fpr is None or tpr is None:
            continue
        try:
            out.append(ClassificationRocCurve(
                participant_label=ref.participant_label,
                session_label=ref.session_label,
                group=ref.group,
                fpr=_np.asarray(fpr, dtype=float),
                tpr=_np.asarray(tpr, dtype=float),
                roc_auc=float(auc),
            ))
        except Exception:
            continue
    return out
