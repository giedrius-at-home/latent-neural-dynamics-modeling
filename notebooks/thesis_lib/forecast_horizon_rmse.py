"""
Forecast RMSE vs horizon: per-step RMSE from Z_future_* (behavioral) or Y_future_* (neural).

For each trial and horizon index m, error is |z_true(m) − z_pred(m)| after z-scoring **the whole
future window** with one pair (μ, σ) from the true future (same global standardization as
``trial_rmse_z_for_model`` over that segment). This is not “per-step z-scores”; late-horizon error
can shrink if predictions track the shape of the remaining trajectory under that scaling.
Trials are aligned on (participant_id, session, block, trial); pooled across triplets and split
by stim (DBS-OFF vs ON). DPAD is not used (forecast implementation differs).

Future columns are written by the trainer when test/validation runs include forecast evaluation; a test
parquet without them means forecast validation was skipped or the run omitted those tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from thesis_lib.aggregate_rmse import _key_index_map, _sem, _trial_key, normalize_stim
from thesis_lib.loaders import (
    ThesisDataError,
    load_split_results_required,
    resolve_neural_y_channel_idx,
)
from thesis_lib.transforms import reshape_future_z_time_first

if TYPE_CHECKING:
    from thesis_lib.specs import AlignedTriplet

logger = logging.getLogger(__name__)


def _per_step_abs_err_z_future(
    z_true_future: Any,
    z_pred_future: Any,
    channel_idx: int,
) -> Optional[np.ndarray]:
    """Return shape (m,) per-step absolute z-error for one Z channel, or None if invalid."""
    if z_true_future is None or z_pred_future is None:
        return None
    try:
        T = reshape_future_z_time_first(np.asarray(z_true_future, dtype=float))
        P = reshape_future_z_time_first(np.asarray(z_pred_future, dtype=float))
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
    msk = np.isfinite(t_vec) & np.isfinite(p_vec)
    if not np.any(msk):
        return None
    mu = float(np.mean(t_vec[msk]))
    sigma = float(np.std(t_vec[msk]))
    if sigma < 1e-12:
        sigma = 1.0
    zt = (t_vec - mu) / sigma
    zp = (p_vec - mu) / sigma
    err = np.abs(zt - zp)
    err[~np.isfinite(err)] = np.nan
    return err


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
    # Per-trial RMSE at global_horizon_ms (for bar+box global performance figure)
    trial_rmse_psid_off: List[float]
    trial_rmse_varma_off: List[float]
    trial_rmse_psid_on: List[float]
    trial_rmse_varma_on: List[float]
    global_horizon_ms: float


def collect_forecast_horizon_rmse(
    results_root: Path,
    triplet_specs: List["AlignedTriplet"],
    channel_idx: int,
    split: str = "test",
    sampling_hz: float = 80.0,
    sample_every: int = 5,
    naive_rmse: float = 1.0,
    *,
    forecast_target: str = "Z",
    neural_y_feature_name: str = "",
) -> ForecastHorizonRmseData:
    """
    Pool per-step forecast RMSE across triplets; split trials by stim from PSID split_res.
    Uses Z_future_* or Y_future_* from parquet when present.
    """
    if forecast_target not in ("Z", "Y"):
        raise ThesisDataError(f"collect_forecast_horizon_rmse: forecast_target must be Z or Y, got {forecast_target!r}.")
    if not triplet_specs:
        raise ThesisDataError("collect_forecast_horizon_rmse: triplet_specs is empty.")
    k_true = "Y_future_true" if forecast_target == "Y" else "Z_future_true"
    k_pred = "Y_future_pred" if forecast_target == "Y" else "Z_future_pred"

    psid_off: List[np.ndarray] = []
    psid_on: List[np.ndarray] = []
    varma_off: List[np.ndarray] = []
    varma_on: List[np.ndarray] = []
    n_triplets_used = 0
    n_off = 0
    n_on = 0

    for tri in triplet_specs:
        res_p = load_split_results_required(results_root, tri.psid_variant, tri.psid_run_ts, split)
        res_v = load_split_results_required(results_root, tri.varma_variant, tri.varma_run_ts, split)
        if not res_p.get(k_true) or not res_p.get(k_pred):
            raise ThesisDataError(
                f"Forecast horizon: PSID triplet {tri.label!r} missing {k_true} / {k_pred}."
            )
        if not res_v.get(k_true) or not res_v.get(k_pred):
            raise ThesisDataError(
                f"Forecast horizon: VARMA triplet {tri.label!r} missing {k_true} / {k_pred}."
            )

        ch_use = (
            resolve_neural_y_channel_idx(res_p, neural_y_feature_name, channel_idx)
            if forecast_target == "Y"
            else int(channel_idx)
        )

        n_triplets_used += 1
        n_psid_trials = len(res_p[k_true])

        mv = _key_index_map(res_v)

        for i_p in range(n_psid_trials):
            if res_p[k_true][i_p] is None or res_p[k_pred][i_p] is None:
                continue
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue

            c_p = _per_step_abs_err_z_future(
                res_p[k_true][i_p],
                res_p[k_pred][i_p],
                ch_use,
            )

            k_p = _trial_key(res_p, i_p)
            i_v = mv.get(k_p)
            if i_v is None:
                raise ThesisDataError(
                    f"Forecast horizon: VARMA missing trial key {k_p!r} for PSID triplet {tri.label!r}."
                )
            c_v = _per_step_abs_err_z_future(
                res_v[k_true][i_v],
                res_v[k_pred][i_v],
                ch_use,
            )

            if c_p is None:
                continue
            if c_v is None:
                raise ThesisDataError(
                    f"Forecast horizon: VARMA invalid future tensor at trial key {k_p!r} (triplet {tri.label!r})."
                )
            mlen = min(c_p.size, c_v.size)
            if mlen == 0:
                continue
            c_p = c_p[:mlen].copy()

            if stim == "off":
                n_off += 1
                psid_off.append(c_p)
                varma_off.append(c_v[:mlen].copy())
            else:
                n_on += 1
                psid_on.append(c_p)
                varma_on.append(c_v[:mlen].copy())

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
        raise ThesisDataError(
            f"Forecast horizon: no valid PSID forecast trials after filtering (check {k_true}/{k_pred} and stim)."
        )

    min_lens: List[int] = []
    for group in (psid_off, psid_on, varma_off, varma_on):
        if group:
            min_lens.append(min(r.shape[0] for r in group))
    if not min_lens:
        raise ThesisDataError("Forecast horizon: zero-length future windows after stacking.")
    m_plot = int(min(min_lens))
    if m_plot <= 0:
        raise ThesisDataError("Forecast horizon: zero-length future windows after stacking.")

    idx = np.arange(0, m_plot, sample_every, dtype=int)
    dt_ms = 1000.0 / sampling_hz
    x_ms = idx.astype(float) * dt_ms

    def _sample_mean_sem(rows: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if not rows:
            return (
                np.full(len(idx), np.nan),
                np.full(len(idx), np.nan),
            )
        trimmed = [r[:m_plot] for r in rows]
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

    # Per-trial RMSE at the global horizon (default 500 ms)
    _global_horizon_ms = 500.0
    _dt_ms = 1000.0 / sampling_hz
    _global_step = int(round(_global_horizon_ms / _dt_ms))
    _global_step = min(_global_step, m_plot - 1)

    def _per_trial_at_step(rows: List[np.ndarray]) -> List[float]:
        return [float(r[_global_step]) for r in rows if _global_step < len(r) and np.isfinite(r[_global_step])]

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
        trial_rmse_psid_off=_per_trial_at_step(psid_off),
        trial_rmse_varma_off=_per_trial_at_step(varma_off),
        trial_rmse_psid_on=_per_trial_at_step(psid_on),
        trial_rmse_varma_on=_per_trial_at_step(varma_on),
        global_horizon_ms=float(_global_step * _dt_ms),
    )
