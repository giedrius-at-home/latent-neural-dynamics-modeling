"""
Forecast RMSE vs horizon (z-scored tracing speed): per-step RMSE from Z_future_true / Z_future_pred.

For each trial and horizon index m, RMSE is |z_true(m) - z_pred(m)| with z-scoring along the
future segment using the true future (same rule as trial_rmse_z_for_model on instantaneous error).
Trials are aligned on (participant_id, session, block, trial); pooled across triplets and split
by stim (DBS-OFF vs ON). DPAD is not used (forecast implementation differs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from dashboard.thesis.aggregate_rmse import _key_index_map, _sem, normalize_stim
from dashboard.thesis.loaders import load_split_results
from dashboard.thesis.transforms import zscore_using_true_stats

if TYPE_CHECKING:
    from dashboard.thesis.specs import AlignedTriplet

logger = logging.getLogger(__name__)


def _time_first(z: np.ndarray) -> np.ndarray:
    """Ensure shape (n_steps, n_channels); horizon along rows."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("Z future must be 2D")
    # If stored as (n_ch, n_steps) with few channels, transpose
    if z.shape[0] <= 8 and z.shape[0] < z.shape[1]:
        return z.T
    return z


def _per_step_rmse_z_future(
    z_true_future: Any,
    z_pred_future: Any,
    channel_idx: int,
) -> Optional[np.ndarray]:
    """Return shape (m,) per-step RMSE in z-space, or None if invalid."""
    if z_true_future is None or z_pred_future is None:
        return None
    try:
        T = _time_first(np.asarray(z_true_future, dtype=float))
        P = _time_first(np.asarray(z_pred_future, dtype=float))
    except ValueError:
        return None
    if T.shape != P.shape:
        return None
    m = T.shape[0]
    n_ch = T.shape[1]
    if n_ch == 1:
        t_vec = T[:, 0]
        p_vec = P[:, 0]
    else:
        if channel_idx >= n_ch:
            return None
        t_vec = T[:, channel_idx]
        p_vec = P[:, channel_idx]
    zt = zscore_using_true_stats(t_vec, t_vec)
    zp = zscore_using_true_stats(t_vec, p_vec)
    return np.sqrt((zt - zp) ** 2)


@dataclass
class ForecastHorizonRmseData:
    """Sampled horizon (ms) and mean ± SEM for PSID and VARMA per DBS panel."""

    x_ms: np.ndarray
    mean_psid_off: np.ndarray
    sem_psid_off: np.ndarray
    mean_varma_off: np.ndarray
    sem_varma_off: np.ndarray
    mean_psid_on: np.ndarray
    sem_psid_on: np.ndarray
    mean_varma_on: np.ndarray
    sem_varma_on: np.ndarray
    naive_rmse: float
    crossover_ms_off: Optional[float]
    crossover_ms_on: Optional[float]
    n_triplets_used: int
    n_trials_off: int
    n_trials_on: int


def collect_forecast_horizon_rmse(
    results_root: Path,
    triplet_specs: List["AlignedTriplet"],
    channel_idx: int,
    split: str = "test",
    sampling_hz: float = 60.0,
    sample_every: int = 5,
    naive_rmse: float = 1.0,
) -> Optional[ForecastHorizonRmseData]:
    """
    Pool per-step forecast RMSE across triplets; split trials by stim from PSID split_res.
    Uses Z_future_true / Z_future_pred from parquet when present.
    """
    psid_off: List[np.ndarray] = []
    psid_on: List[np.ndarray] = []
    varma_off: List[np.ndarray] = []
    varma_on: List[np.ndarray] = []
    n_triplets_used = 0
    n_off = 0
    n_on = 0

    for tri in triplet_specs:
        res_p = load_split_results(results_root, tri.psid_variant, tri.psid_run_ts, split)
        res_v = load_split_results(results_root, tri.varma_variant, tri.varma_run_ts, split)
        if res_p is None or res_v is None:
            logger.warning(
                "Skipping triplet %s: missing PSID or VARMA results.",
                getattr(tri, "label", ""),
            )
            continue
        if not res_p.get("Z_future_true") or not res_p.get("Z_future_pred"):
            logger.warning(
                "Skipping triplet %s: PSID split missing Z_future_true/Z_future_pred.",
                getattr(tri, "label", ""),
            )
            continue
        if not res_v.get("Z_future_true") or not res_v.get("Z_future_pred"):
            logger.warning(
                "Skipping triplet %s: VARMA split missing Z_future_true/Z_future_pred.",
                getattr(tri, "label", ""),
            )
            continue

        mp = _key_index_map(res_p)
        mv = _key_index_map(res_v)
        common = set(mp.keys()) & set(mv.keys())
        if not common:
            logger.warning(
                "Skipping triplet %s: no overlapping trial keys (PSID vs VARMA).",
                getattr(tri, "label", ""),
            )
            continue

        n_triplets_used += 1
        for k in sorted(common, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))):
            i_p, i_v = mp[k], mv[k]
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue

            c_p = _per_step_rmse_z_future(
                res_p["Z_future_true"][i_p],
                res_p["Z_future_pred"][i_p],
                channel_idx,
            )
            c_v = _per_step_rmse_z_future(
                res_v["Z_future_true"][i_v],
                res_v["Z_future_pred"][i_v],
                channel_idx,
            )
            if c_p is None or c_v is None:
                continue
            mlen = min(c_p.size, c_v.size)
            if mlen == 0:
                continue
            c_p = c_p[:mlen].copy()
            c_v = c_v[:mlen].copy()

            if stim == "off":
                n_off += 1
                psid_off.append(c_p)
                varma_off.append(c_v)
            else:
                n_on += 1
                psid_on.append(c_p)
                varma_on.append(c_v)

    def _trim_stack(rows: List[np.ndarray]) -> List[np.ndarray]:
        if not rows:
            return []
        mmin = min(r.shape[0] for r in rows)
        return [r[:mmin] for r in rows]

    psid_off = _trim_stack(psid_off)
    psid_on = _trim_stack(psid_on)
    varma_off = _trim_stack(varma_off)
    varma_on = _trim_stack(varma_on)

    if not psid_off and not psid_on:
        return None

    m_max = 0
    for group in (psid_off, psid_on, varma_off, varma_on):
        for r in group:
            m_max = max(m_max, r.shape[0])
    if m_max == 0:
        return None

    idx = np.arange(0, m_max, sample_every, dtype=int)
    dt_ms = 1000.0 / sampling_hz
    x_ms = idx.astype(float) * dt_ms

    def _sample_mean_sem(rows: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not rows:
            return (
                np.full(len(idx), np.nan),
                np.full(len(idx), np.nan),
            )
        trimmed = [r[:m_max] for r in rows]
        arr = np.stack(trimmed, axis=0)
        sub = arr[:, idx]
        mean = np.nanmean(sub, axis=0)
        sem = np.array([_sem(sub[:, i]) for i in range(sub.shape[1])])
        return mean, sem

    mo, so = _sample_mean_sem(psid_off)
    mvo, svo = _sample_mean_sem(varma_off)
    mi, si = _sample_mean_sem(psid_on)
    mvi, svi = _sample_mean_sem(varma_on)

    def _crossover(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> Optional[float]:
        """First x where PSID mean (a) crosses above VARMA mean (b); linear interp on a - b."""
        if x.size < 2 or a.size != b.size:
            return None
        d = a - b
        if not (np.isfinite(d[0]) and d[0] < 0):
            return None
        for i in range(1, len(x)):
            if not (np.isfinite(d[i]) and np.isfinite(d[i - 1])):
                continue
            if d[i] >= 0.0 and d[i - 1] < 0:
                denom = d[i - 1] - d[i]
                if abs(denom) < 1e-15:
                    return float(x[i])
                t = d[i - 1] / denom
                t = float(np.clip(t, 0.0, 1.0))
                return float(x[i - 1] + t * (x[i] - x[i - 1]))
        return None

    xo = _crossover(x_ms, mo, mvo)
    xi = _crossover(x_ms, mi, mvi)

    return ForecastHorizonRmseData(
        x_ms=x_ms,
        mean_psid_off=mo,
        sem_psid_off=so,
        mean_varma_off=mvo,
        sem_varma_off=svo,
        mean_psid_on=mi,
        sem_psid_on=si,
        mean_varma_on=mvi,
        sem_varma_on=svi,
        naive_rmse=float(naive_rmse),
        crossover_ms_off=xo,
        crossover_ms_on=xi,
        n_triplets_used=n_triplets_used,
        n_trials_off=n_off,
        n_trials_on=n_on,
    )
