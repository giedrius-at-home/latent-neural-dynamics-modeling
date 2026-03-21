"""
Load fitted PSID models and compute behaviourally relevant Cy importance (4×29) per session.

Cy rows follow `input_channels` order (29 narrow-band features per ECoG contact, ECoG 1–4).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

from dashboard.thesis.loaders import load_split_results
from dashboard.thesis.neural_band_pearson import parse_parent_band

logger = logging.getLogger(__name__)

_N_FEATURES = 116
_N_CONTACTS = 4
_N_BAND_COLS = 29

_BAND_ORDER = ("Delta", "Theta", "Alpha", "Beta")


@dataclass(frozen=True)
class BandColumnLayout:
    """Column index ranges (inclusive) within 0..28 for one contact’s 29 features."""

    spans: Tuple[Tuple[str, int, int], ...]
    beta_col_start: float
    beta_col_end: float


def _first_contact_29(input_channels: Sequence[str]) -> List[str]:
    if len(input_channels) < _N_BAND_COLS:
        raise ValueError(
            f"Need at least {_N_BAND_COLS} input channels; got {len(input_channels)}"
        )
    return [str(input_channels[i]) for i in range(_N_BAND_COLS)]


def band_layout_from_channels(input_channels: Sequence[str]) -> BandColumnLayout:
    """
    Parse δ/θ/α/β column spans from the first 29 channel names (ECoG 1).
    Returns half-open [beta_col_start, beta_col_end] in data coordinates for a rectangle
    covering β columns (cell edges: -0.5 .. n-0.5).
    """
    first = _first_contact_29(input_channels)
    by_band: Dict[str, List[int]] = {b: [] for b in _BAND_ORDER}
    for j, ch in enumerate(first):
        lab = parse_parent_band(ch)
        if lab is None:
            continue
        if lab in by_band:
            by_band[lab].append(j)

    spans_list: List[Tuple[str, int, int]] = []
    for b in _BAND_ORDER:
        idx = by_band.get(b) or []
        if not idx:
            continue
        spans_list.append((b, min(idx), max(idx)))

    beta_idx = by_band.get("Beta") or []
    if not beta_idx:
        logger.warning("No Beta band columns parsed from first 29 channels; beta box disabled.")
        beta_col_start = 0.0
        beta_col_end = 0.0
    else:
        lo, hi = min(beta_idx), max(beta_idx)
        beta_col_start = float(lo) - 0.5
        beta_col_end = float(hi) + 0.5

    return BandColumnLayout(spans=tuple(spans_list), beta_col_start=beta_col_start, beta_col_end=beta_col_end)


def load_input_channels(
    results_root: Path,
    variant: str,
    run_ts: str,
    split: str,
) -> List[str]:
    res = load_split_results(results_root, variant, run_ts, split)
    if res is None:
        raise FileNotFoundError(
            f"No results for variant={variant!r} run_ts={run_ts!r} split={split!r}"
        )
    ch = res.get("input_channels")
    if not ch:
        raise ValueError(f"Missing input_channels in results for {variant}/{run_ts}")
    out = list(ch) if not isinstance(ch, list) else ch
    return [str(x) for x in out]


def load_psid_id_sys(model_path: Path) -> Any:
    if not model_path.is_file():
        raise FileNotFoundError(f"PSID model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_normalized_cy_importance(id_sys: Any) -> Tuple[np.ndarray, int]:
    """
    Returns (4, 29) importance matrix (per-panel max normalized) and n1.
    """
    if not hasattr(id_sys, "Cy") or id_sys.Cy is None:
        raise ValueError("idSys has no Cy matrix")
    cy = np.asarray(id_sys.Cy, dtype=float)
    if cy.ndim != 2:
        raise ValueError(f"Cy must be 2D; got shape {cy.shape}")
    if cy.shape[0] == _N_CONTACTS and cy.shape[1] == _N_FEATURES:
        cy = cy.T
    n1 = int(getattr(id_sys, "n1", cy.shape[1]))
    if n1 < 1:
        raise ValueError(f"Invalid n1={n1}")
    cy_rel = cy[:, :n1]
    if cy_rel.shape[0] != _N_FEATURES:
        raise ValueError(
            f"Expected Cy to have {_N_FEATURES} rows (4×29 narrow-band features); got {cy_rel.shape[0]}"
        )
    resh = cy_rel.reshape(_N_CONTACTS, _N_BAND_COLS, n1)
    imp = np.linalg.norm(resh, axis=2)
    m = float(np.max(imp)) if imp.size else 0.0
    if m <= 0:
        m = 1.0
    imp = imp / m
    return imp, n1


def resolve_model_path(results_root: Path, variant: str, run_ts: str) -> Path:
    return results_root / variant / f"model_{run_ts}.pkl"


def compute_panel(
    results_root: Path,
    variant: str,
    run_ts: str,
    split: str,
) -> Tuple[np.ndarray, int, List[str], BandColumnLayout]:
    """One participant×session: normalized importance, n1, channels, band layout."""
    model_path = resolve_model_path(results_root, variant, run_ts)
    channels = load_input_channels(results_root, variant, run_ts, split)
    if len(channels) != _N_FEATURES:
        raise ValueError(
            f"Expected {len(channels)}=={_N_FEATURES} input_channels for narrow-band layout; "
            f"check variant {variant}"
        )
    layout = band_layout_from_channels(channels)
    id_sys = load_psid_id_sys(model_path)
    imp, n1 = compute_normalized_cy_importance(id_sys)
    return imp, n1, channels, layout
