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

_N_CONTACTS = 4
# Band cols derived dynamically: 29 for 80Hz (116/4), 15 for 200Hz (60/4)

_BAND_ORDER = ("Delta", "Theta", "Alpha", "Beta")


@dataclass(frozen=True)
class BandColumnLayout:
    """Column index ranges (inclusive) within 0..28 for one contact’s 29 features."""

    spans: Tuple[Tuple[str, int, int], ...]
    beta_col_start: float
    beta_col_end: float


def _n_band_cols(n_features: int) -> int:
    """Derive bands-per-contact from total feature count."""
    if n_features % _N_CONTACTS != 0:
        raise ValueError(f"Feature count {n_features} not divisible by {_N_CONTACTS} contacts")
    return n_features // _N_CONTACTS


def _first_contact_n(input_channels: Sequence[str], n_bands: int) -> List[str]:
    if len(input_channels) < n_bands:
        raise ValueError(
            f"Need at least {n_bands} input channels; got {len(input_channels)}"
        )
    return [str(input_channels[i]) for i in range(n_bands)]


def _first_single_contact_narrowband(
    input_channels: Sequence[str], contact: int = 1
) -> List[str]:
    """
    Extract one contact’s narrow-band channels from the interleaved input list.
    Works for both 116-input (29 per contact) and 60-input (15 per contact) layouts.
    """
    n_bands = _n_band_cols(len(input_channels))
    pref_ecog = f"ECOG_{contact}_"
    pref_lfp = f"LFP_{contact}_"
    out: List[str] = []
    for ch in input_channels:
        s = str(ch)
        su = s.upper()
        if su.startswith(pref_ecog.upper()) or su.startswith(pref_lfp.upper()):
            out.append(s)
        if len(out) >= n_bands:
            break
    if len(out) >= n_bands:
        return out[:n_bands]
    return _first_contact_n(input_channels, n_bands)


def cy_heatmap_x_tick_labels(input_channels: Sequence[str]) -> List[str]:
    """X-axis labels (contact 1 narrow-band names), raw on-disk strings."""
    return list(_first_single_contact_narrowband(input_channels))


def band_layout_from_channels(input_channels: Sequence[str]) -> BandColumnLayout:
    """
    Parse δ/θ/α/β column spans from **one contact’s** 29 narrow-band names (ECoG 1 by default),
    not the first 29 rows of the interleaved 116-input list.

    Returns half-open [beta_col_start, beta_col_end] in data coordinates for a rectangle
    covering β columns (cell edges: -0.5 .. n-0.5).
    """
    global _beta_layout_warned
    first = _first_single_contact_narrowband(input_channels, contact=1)
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


def _channels_from_training_yaml(variant: str) -> List[str]:
    """Load neural_input channels from the training YAML config for this variant."""
    import re, yaml
    m = re.search(r"(PDI\d+)_(\d+)", variant)
    if not m:
        raise ValueError(f"Cannot parse participant/session from variant: {variant}")
    pid, sess = m.group(1), m.group(2)
    yaml_dir = Path("training/setups")
    yaml_path = yaml_dir / f"psid_gs_{pid}_S{sess}_200Hz_narrow_band.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Training config not found: {yaml_path}")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]["channels"]["neural_input"]


def load_input_channels(
    results_root: Path,
    variant: str,
    run_ts: str,
    split: str,
) -> List[str]:
    """Resolve input_channels from results or training YAML."""
    res = load_split_results(results_root, variant, run_ts, split)
    if res is not None:
        ch = res.get("input_channels")
        if ch and len(ch) > 0:
            return [str(x) for x in (list(ch) if not isinstance(ch, list) else ch)]
    return _channels_from_training_yaml(variant)


def load_psid_id_sys(model_path: Path) -> Any:
    if not model_path.is_file():
        raise FileNotFoundError(f"PSID model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_normalized_cy_importance(id_sys: Any) -> Tuple[np.ndarray, int]:
    """
    Returns (4, n_bands) importance matrix (per-panel max normalized) and n1.
    Works for both 116 (4×29) and 60 (4×15) feature layouts.
    """
    cy_raw = getattr(id_sys, "Cy", None) or getattr(id_sys, "C", None)
    if cy_raw is None:
        raise ValueError("idSys has no Cy/C matrix")
    cy = np.asarray(cy_raw, dtype=float)
    if cy.ndim != 2:
        raise ValueError(f"Cy must be 2D; got shape {cy.shape}")
    n_features = cy.shape[0]
    if n_features == _N_CONTACTS and cy.shape[1] % _N_CONTACTS == 0:
        cy = cy.T
        n_features = cy.shape[0]
    if n_features % _N_CONTACTS != 0:
        raise ValueError(
            f"Cy rows ({n_features}) not divisible by {_N_CONTACTS} contacts"
        )
    n_band_cols = n_features // _N_CONTACTS
    n1 = int(getattr(id_sys, "n1", cy.shape[1]))
    if n1 < 1:
        raise ValueError(f"Invalid n1={n1}")
    cy_rel = cy[:, :n1]
    resh = cy_rel.reshape(_N_CONTACTS, n_band_cols, n1)
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
    id_sys = load_psid_id_sys(model_path)

    cy_raw = getattr(id_sys, "Cy", None) or getattr(id_sys, "C", None)
    if cy_raw is None:
        raise ValueError("idSys has no Cy/C matrix")
    cy = np.asarray(cy_raw, dtype=float)
    if cy.ndim != 2:
        raise ValueError(f"Cy must be 2D; got shape {cy.shape}")
    # Derive n_features from the Cy matrix itself (authoritative)
    n_features = cy.shape[0]
    if n_features == _N_CONTACTS and cy.shape[1] % _N_CONTACTS == 0 and cy.shape[1] > _N_CONTACTS:
        cy = cy.T
        n_features = cy.shape[0]
    if n_features % _N_CONTACTS != 0:
        raise ValueError(
            f"Cy rows ({n_features}) not divisible by {_N_CONTACTS} contacts for {variant}"
        )
    n_band_cols = n_features // _N_CONTACTS
    n1 = int(getattr(id_sys, "n1", cy.shape[1]))
    if n1 < 1:
        raise ValueError(f"Invalid n1={n1}")
    cy_rel = cy[:, :n1]

    channels = load_input_channels(results_root, variant, run_ts, split)
    if len(channels) != n_features:
        raise ValueError(
            f"Channel count {len(channels)} != Cy rows {n_features} for {variant}"
        )
    layout = band_layout_from_channels(channels)

    resh = cy_rel.reshape(_N_CONTACTS, n_band_cols, n1)
    # (n1, 4_contacts): norm of Cy over 29 bands for each (latent_dim, contact) pair
    # Transpose to (n1, 4): y=latent dims, x=ECoG contacts
    cy_per_dim_contact = np.linalg.norm(resh, axis=1).T  # (n1, 4_contacts)
    for j in range(cy_per_dim_contact.shape[0]):
        row_max = float(np.max(np.abs(cy_per_dim_contact[j])))
        if row_max > 0:
            cy_per_dim_contact[j] /= row_max

    return cy_per_dim_contact, n1, channels, layout


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
    if len(channels) % _N_CONTACTS != 0:
        raise ValueError(
            f"Channel count {len(channels)} not divisible by {_N_CONTACTS} contacts; "
            f"check variant {variant}"
        )
    layout = band_layout_from_channels(channels)
    id_sys = load_psid_id_sys(model_path)
    imp, n1 = compute_normalized_cy_importance(id_sys)
    return imp, n1, channels, layout
