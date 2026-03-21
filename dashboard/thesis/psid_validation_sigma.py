"""
σ for the PSID ±1σ shaded band only: pooled standard deviation of z-scored
validation residuals (z_true − z_psid) for one behavioral output channel.

This is not a full probabilistic forecast covariance — it is the empirical spread
of PSID errors on the validation split, reused as a constant band width on test plots.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from dashboard.subtabs.helpers import get_channel, get_trial_time_axis, transpose_if_needed
from dashboard.thesis.transforms import zscore_using_true_stats


def sigma_z_from_psid_val_residuals(
    val_res: Optional[Dict[str, Any]],
    channel_idx: int,
) -> Optional[float]:
    """
    Pool z-scored residuals (z_true - z_psid) across all validation trials for one
    output channel; return std of the pooled residuals as σ for the shaded band.
    """
    if val_res is None:
        return None
    Z_list = val_res.get("Z")
    Zp_list = val_res.get("Zp")
    if not Z_list or not Zp_list:
        return None

    pooled: list[np.ndarray] = []
    for trial_idx in range(len(Z_list)):
        if Z_list[trial_idx] is None or Zp_list[trial_idx] is None:
            continue
        z_arr = np.asarray(Z_list[trial_idx])
        zp_arr = np.asarray(Zp_list[trial_idx])
        n_samples = z_arr.shape[0] if z_arr.ndim >= 1 else 1
        if z_arr.ndim == 2 and z_arr.shape[0] != n_samples:
            n_samples = z_arr.shape[0]
        t_abs = get_trial_time_axis(val_res, trial_idx, n_samples)
        z_arr = transpose_if_needed(z_arr, len(t_abs))
        zp_arr = transpose_if_needed(zp_arr, len(t_abs))
        true_c = get_channel(z_arr, channel_idx, t_abs)
        pred_c = get_channel(zp_arr, channel_idx, t_abs)
        z_t = zscore_using_true_stats(true_c, true_c)
        z_p = zscore_using_true_stats(true_c, pred_c)
        pooled.append(z_t - z_p)

    if not pooled:
        return None
    cat = np.concatenate([p.ravel() for p in pooled])
    if cat.size == 0:
        return None
    return float(np.std(cat))
