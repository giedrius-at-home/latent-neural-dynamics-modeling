"""
Load test-split Xp and stim labels; build PSID (x₁,x₂) and DPAD (first two X dimensions) for thesis panels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from dashboard.thesis.loaders import load_split_results

logger = logging.getLogger(__name__)


def _ensure_time_first(xp: np.ndarray) -> np.ndarray:
    a = np.asarray(xp, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"Xp trial must be 2D; got shape {a.shape}")
    if a.shape[0] < a.shape[1] and a.shape[0] <= 16:
        a = a.T
    return a


def _trial_indices_by_stim(stim_list: Sequence[Any]) -> Tuple[List[int], List[int]]:
    off: List[int] = []
    on: List[int] = []
    for i, s in enumerate(stim_list):
        if s is None:
            continue
        sl = str(s).lower()
        if sl == "off":
            off.append(i)
        elif sl == "on":
            on.append(i)
    return off, on


@dataclass(frozen=True)
class PanelLatentData:
    """Per participant×session panel: 2D coords for KDE + trajectories."""

    session_label: str
    # PSID: columns 0,1 of x^(1)
    x_psid_off: np.ndarray
    y_psid_off: np.ndarray
    x_psid_on: np.ndarray
    y_psid_on: np.ndarray
    traj_psid_off: List[Tuple[np.ndarray, np.ndarray]]
    traj_psid_on: List[Tuple[np.ndarray, np.ndarray]]
    # DPAD: first two dimensions of X (same latent state as training; comparable to PSID x₁,x₂)
    x_dpad_off: np.ndarray
    y_dpad_off: np.ndarray
    x_dpad_on: np.ndarray
    y_dpad_on: np.ndarray
    traj_dpad_off: List[Tuple[np.ndarray, np.ndarray]]
    traj_dpad_on: List[Tuple[np.ndarray, np.ndarray]]
    pca_variance_ratio: Tuple[float, float]  # (0,0) when DPAD uses raw X dims (no PCA)
    x_range_psid: Tuple[float, float]
    y_range_psid: Tuple[float, float]
    x_range_dpad: Tuple[float, float]
    y_range_dpad: Tuple[float, float]


def _slice_x1(xp: np.ndarray, n1: int) -> np.ndarray:
    if xp.shape[1] < n1:
        raise ValueError(f"Xp has {xp.shape[1]} dims; need at least {n1}")
    return xp[:, :n1]


def _stack_condition_points(
    Xp_list: List[Any],
    trial_idx: List[int],
    n1: int,
) -> np.ndarray:
    """Return (N_points, n1) array of all timepoints from listed trials."""
    chunks: List[np.ndarray] = []
    for idx in trial_idx:
        if idx >= len(Xp_list) or Xp_list[idx] is None:
            continue
        x = _ensure_time_first(np.asarray(Xp_list[idx]))
        chunks.append(_slice_x1(x, n1))
    if not chunks:
        return np.zeros((0, n1), dtype=float)
    return np.vstack(chunks)


def _pick_trajectories(
    Xp_list: List[Any],
    trial_idx: List[int],
    n1: int,
    n_traj: int,
    rng: np.random.Generator,
    project: str,
    pca: Optional[Any] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not trial_idx:
        return []
    k = min(n_traj, len(trial_idx))
    chosen = rng.choice(trial_idx, size=k, replace=False)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in chosen:
        x = _ensure_time_first(np.asarray(Xp_list[idx]))
        sl = _slice_x1(x, n1)
        if project == "identity":
            out.append((sl[:, 0].copy(), sl[:, 1].copy()))
        else:
            if pca is None:
                raise ValueError("PCA model required for DPAD trajectories")
            xy = pca.transform(sl)
            out.append((xy[:, 0].copy(), xy[:, 1].copy()))
    return out


def load_panel_latent_data(
    results_root: Path,
    panel: Any,
    split: str,
    n_psid: int,
    n_dpad: int,
    n_traj: int,
    traj_seed: int,
) -> PanelLatentData:
    """
    panel: LatentPhasePanel with variants and run_ts.
    """
    if n_psid != 2:
        raise ValueError("Thesis latent PSID panel expects n_psid_latent=2 (x₁ vs x₂)")
    if n_dpad < 2:
        raise ValueError("n_dpad_latent must be >= 2")

    res_p = load_split_results(
        results_root, panel.psid_variant, panel.psid_run_ts, split
    )
    res_d = load_split_results(
        results_root, panel.dpad_variant, panel.dpad_run_ts, split
    )
    if res_p is None or res_d is None:
        raise FileNotFoundError(
            f"Missing split results for PSID or DPAD ({panel.session_label})"
        )

    Xp_p = res_p.get("Xp") or []
    stim_p = res_p.get("stim") or []
    Xp_d = res_d.get("Xp") or []
    stim_d = res_d.get("stim") or []

    if not Xp_p or not stim_p:
        raise ValueError("PSID split missing Xp or stim")
    if not Xp_d or not stim_d:
        raise ValueError("DPAD split missing Xp or stim")

    off_p, on_p = _trial_indices_by_stim(stim_p)
    off_d, on_d = _trial_indices_by_stim(stim_d)

    rng = np.random.default_rng(traj_seed)

    # --- PSID: direct x1, x2 ---
    def xy_psid(trials: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        xy = _stack_condition_points(Xp_p, trials, n_psid)
        if xy.shape[0] == 0:
            return np.array([]), np.array([])
        return xy[:, 0], xy[:, 1]

    x_po, y_po = xy_psid(off_p)
    x_pn, y_pn = xy_psid(on_p)

    # --- DPAD: first two columns of X (no PCA; aligns with PSID row for side-by-side comparison) ---
    X_off_d = _stack_condition_points(Xp_d, off_d, n_dpad)
    X_on_d = _stack_condition_points(Xp_d, on_d, n_dpad)

    def _xy_dpad(X_blk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if X_blk.shape[0] == 0:
            return np.array([]), np.array([])
        return X_blk[:, 0].copy(), X_blk[:, 1].copy()

    x_do, y_do = _xy_dpad(X_off_d)
    x_dn, y_dn = _xy_dpad(X_on_d)
    pca_var = (0.0, 0.0)

    traj_psid_off = _pick_trajectories(
        Xp_p, off_p, n_psid, n_traj, rng, "identity", None
    )
    traj_psid_on = _pick_trajectories(
        Xp_p, on_p, n_psid, n_traj, rng, "identity", None
    )
    rng2 = np.random.default_rng(traj_seed + 17)
    traj_dpad_off = _pick_trajectories(
        Xp_d, off_d, n_dpad, n_traj, rng2, "identity", None
    )
    traj_dpad_on = _pick_trajectories(
        Xp_d, on_d, n_dpad, n_traj, rng2, "identity", None
    )

    def _rng_xy(
        xo: np.ndarray,
        yo: np.ndarray,
        xn: np.ndarray,
        yn: np.ndarray,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        pts_x: List[np.ndarray] = []
        pts_y: List[np.ndarray] = []
        if len(xo) and len(yo) and len(xo) == len(yo):
            pts_x.append(np.asarray(xo, dtype=float))
            pts_y.append(np.asarray(yo, dtype=float))
        if len(xn) and len(yn) and len(xn) == len(yn):
            pts_x.append(np.asarray(xn, dtype=float))
            pts_y.append(np.asarray(yn, dtype=float))
        if not pts_x:
            return (0.0, 1.0), (0.0, 1.0)
        all_x = np.concatenate(pts_x)
        all_y = np.concatenate(pts_y)
        pad_x = 0.08 * (float(np.nanmax(all_x)) - float(np.nanmin(all_x)) + 1e-9)
        pad_y = 0.08 * (float(np.nanmax(all_y)) - float(np.nanmin(all_y)) + 1e-9)
        return (
            (float(np.nanmin(all_x)) - pad_x, float(np.nanmax(all_x)) + pad_x),
            (float(np.nanmin(all_y)) - pad_y, float(np.nanmax(all_y)) + pad_y),
        )

    xr_p, yr_p = _rng_xy(x_po, y_po, x_pn, y_pn)
    xr_d, yr_d = _rng_xy(x_do, y_do, x_dn, y_dn)

    return PanelLatentData(
        session_label=panel.session_label,
        x_psid_off=x_po,
        y_psid_off=y_po,
        x_psid_on=x_pn,
        y_psid_on=y_pn,
        traj_psid_off=traj_psid_off,
        traj_psid_on=traj_psid_on,
        x_dpad_off=x_do,
        y_dpad_off=y_do,
        x_dpad_on=x_dn,
        y_dpad_on=y_dn,
        traj_dpad_off=traj_dpad_off,
        traj_dpad_on=traj_dpad_on,
        pca_variance_ratio=pca_var,
        x_range_psid=xr_p,
        y_range_psid=yr_p,
        x_range_dpad=xr_d,
        y_range_dpad=yr_d,
    )


def kde_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    grid_n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns xi, yi, zi (density on grid) for scipy KDE."""
    from scipy.stats import gaussian_kde

    if len(x) < 2:
        raise ValueError("KDE needs at least 2 points")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        raise ValueError("KDE needs at least 2 finite points")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dx = xmax - xmin + 1e-9
    dy = ymax - ymin + 1e-9
    xmin -= 0.05 * dx
    xmax += 0.05 * dx
    ymin -= 0.05 * dy
    ymax += 0.05 * dy
    xi = np.linspace(xmin, xmax, grid_n)
    yi = np.linspace(ymin, ymax, grid_n)
    Xg, Yg = np.meshgrid(xi, yi, indexing="xy")
    try:
        kde = gaussian_kde(np.vstack([x, y]))
        # Narrower bandwidth → sharper, more separable within-panel DBS modes (thesis viz).
        kde.set_bandwidth(kde.factor * 0.72)
        zi = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    except Exception as e:
        logger.warning("KDE failed (%s); adding tiny jitter", e)
        rng = np.random.default_rng(0)
        jitter = rng.normal(0, 1e-6, size=(2, len(x)))
        kde = gaussian_kde(np.vstack([x, y]) + jitter)
        kde.set_bandwidth(kde.factor * 0.72)
        zi = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    return xi, yi, zi


def kde_on_fixed_grid(
    x: np.ndarray,
    y: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
) -> np.ndarray:
    """Evaluate a 2D KDE on an existing ``xi × yi`` mesh (same grid for stacked heatmaps)."""
    from scipy.stats import gaussian_kde

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return np.full((len(yi), len(xi)), np.nan)
    Xg, Yg = np.meshgrid(xi, yi, indexing="xy")
    try:
        kde = gaussian_kde(np.vstack([x, y]))
        kde.set_bandwidth(kde.factor * 0.72)
        return kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    except Exception as e:
        logger.warning("KDE on fixed grid failed (%s); jitter retry", e)
        rng = np.random.default_rng(0)
        jitter = rng.normal(0, 1e-6, size=(2, len(x)))
        kde = gaussian_kde(np.vstack([x, y]) + jitter)
        kde.set_bandwidth(kde.factor * 0.72)
        return kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
