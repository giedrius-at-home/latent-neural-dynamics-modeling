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

_beta_layout_warned = False

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


def _first_single_contact_narrowband_29(
    input_channels: Sequence[str], contact: int = 1
) -> List[str]:
    """
    The 116-input YAML order interleaves ECoG 1–4 (delta block, then theta, …), so the
    first 29 list entries are **not** one contact’s δ/θ/α/β stack. Take channels whose
    names start with ``ECOG_<n>_`` or ``LFP_<n>_`` for one contact, in file order.
    """
    pref_ecog = f"ECOG_{contact}_"
    pref_lfp = f"LFP_{contact}_"
    out: List[str] = []
    for ch in input_channels:
        s = str(ch)
        su = s.upper()
        if su.startswith(pref_ecog.upper()) or su.startswith(pref_lfp.upper()):
            out.append(s)
        if len(out) >= _N_BAND_COLS:
            break
    if len(out) >= _N_BAND_COLS:
        return out[:_N_BAND_COLS]
    return _first_contact_29(input_channels)


def cy_heatmap_x_tick_labels(input_channels: Sequence[str]) -> List[str]:
    """29 x-axis labels (contact 1 narrow-band names), raw on-disk strings."""
    return list(_first_single_contact_narrowband_29(input_channels))


def band_layout_from_channels(input_channels: Sequence[str]) -> BandColumnLayout:
    """
    Parse δ/θ/α/β column spans from **one contact’s** 29 narrow-band names (ECoG 1 by default),
    not the first 29 rows of the interleaved 116-input list.

    Returns half-open [beta_col_start, beta_col_end] in data coordinates for a rectangle
    covering β columns (cell edges: -0.5 .. n-0.5).
    """
    global _beta_layout_warned
    first = _first_single_contact_narrowband_29(input_channels, contact=1)
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
        if not _beta_layout_warned:
            _beta_layout_warned = True
            logger.warning(
                "No Beta band columns parsed from first 29 channels; beta box disabled "
                "(further panels: no repeat)."
            )
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
    """Resolve input_channels from the variant's results, or from the matching DPAD variant
    (same participant/session) when the PSID parquet doesn't store them."""
    res = load_split_results(results_root, variant, run_ts, split)
    if res is None:
        raise FileNotFoundError(
            f"No results for variant={variant!r} run_ts={run_ts!r} split={split!r}"
        )
    ch = res.get("input_channels")
    if ch:
        return [str(x) for x in (list(ch) if not isinstance(ch, list) else ch)]

    import re
    from dashboard.subtabs.helpers import list_run_timestamps

    m = re.search(r"(PDI\d+)_(\d+)", variant)
    if m:
        pid, sess = m.group(1), m.group(2)
        pattern = f"dpad_{pid}_S{sess}_"
        for d in sorted(results_root.iterdir()):
            if d.is_dir() and d.name.startswith(pattern):
                ts_list = list_run_timestamps(d)
                if not ts_list:
                    continue
                dpad_res = load_split_results(results_root, d.name, ts_list[-1], split)
                if dpad_res and dpad_res.get("input_channels"):
                    return [str(x) for x in list(dpad_res["input_channels"])]

    raise ValueError(
        f"input_channels missing from {variant}/{run_ts} and no matching DPAD variant found"
    )


def load_psid_id_sys(model_path: Path) -> Any:
    if not model_path.is_file():
        raise FileNotFoundError(f"PSID model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_normalized_cy_importance(id_sys: Any) -> Tuple[np.ndarray, int]:
    """
    Returns (4, 29) importance matrix (per-panel max normalized) and n1.
    """
    cy_raw = getattr(id_sys, "Cy", None) or getattr(id_sys, "C", None)
    if cy_raw is None:
        raise ValueError("idSys has no Cy/C matrix")
    cy = np.asarray(cy_raw, dtype=float)
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


def compute_cy_signed_heatmap(
    results_root: Path,
    variant: str,
    run_ts: str,
    split: str,
) -> Tuple[np.ndarray, int, List[str], BandColumnLayout]:
    """
    Returns (4, 29) signed Cy matrix normalized per-ECoG-row to [-1, +1], n1,
    input channel names, and band layout.  Values are sum-across-latent-dims
    (signed) so negative weights show up in blue on a diverging colorscale.
    """
    model_path = resolve_model_path(results_root, variant, run_ts)
    channels = load_input_channels(results_root, variant, run_ts, split)
    if len(channels) != _N_FEATURES:
        raise ValueError(
            f"Expected {_N_FEATURES} input_channels; got {len(channels)} for {variant}"
        )
    layout = band_layout_from_channels(channels)
    id_sys = load_psid_id_sys(model_path)

    cy_raw = getattr(id_sys, "Cy", None) or getattr(id_sys, "C", None)
    if cy_raw is None:
        raise ValueError("idSys has no Cy/C matrix")
    cy = np.asarray(cy_raw, dtype=float)
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
            f"Expected Cy to have {_N_FEATURES} rows; got {cy_rel.shape[0]}"
        )

    resh = cy_rel.reshape(_N_CONTACTS, _N_BAND_COLS, n1)
    cy_signed = resh.sum(axis=2).copy()  # (4, 29), signed sum across latent dims

    for i in range(_N_CONTACTS):
        row_max = float(np.max(np.abs(cy_signed[i])))
        if row_max > 0:
            cy_signed[i] /= row_max

    return cy_signed, n1, channels, layout


def compute_cz_heatmap(id_sys: Any) -> Tuple[np.ndarray, int, List[str]]:
    """
    Returns (nz, n1) Cz matrix normalized to [-1, 1] by abs-max,
    the n1 value, and fallback y-axis labels ["z_0", "z_1", ...].

    Use output_channels from load_split_results() to replace the fallback labels.
    """
    cz = getattr(id_sys, "Cz", None)
    if cz is None:
        raise ValueError("idSys has no Cz matrix")
    cz = np.asarray(cz, dtype=float)
    if cz.ndim != 2:
        raise ValueError(f"Cz must be 2D; got shape {cz.shape}")
    n1 = int(getattr(id_sys, "n1", cz.shape[1]))
    if n1 < 1:
        raise ValueError(f"Invalid n1={n1}")
    cz_rel = cz[:, :n1]
    m = float(np.max(np.abs(cz_rel))) if cz_rel.size else 0.0
    if m <= 0:
        m = 1.0
    z_labels = [f"z_{i}" for i in range(cz_rel.shape[0])]
    return cz_rel / m, n1, z_labels


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
