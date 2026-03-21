from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np

from dashboard.subtabs.helpers import get_channel, get_trial_time_axis, transpose_if_needed
from dashboard.thesis.transforms import rmse_z, zscore_using_true_stats
from utils.classification import load_precomputed_results


InputMode = Literal["neural", "behavioral"]


@dataclass(frozen=True)
class TrialZSeries:
    """Z_true and model Zp predictions for one trial and one output channel."""

    t_abs: np.ndarray
    z_true_raw: np.ndarray
    z_psid: np.ndarray
    z_dpad: np.ndarray
    z_varma: np.ndarray


def load_split_results(
    results_root: Path,
    variant: str,
    run_timestamp: str,
    split: str,
) -> Optional[Dict[str, Any]]:
    variant_dir = results_root / variant
    return load_precomputed_results(variant_dir, run_timestamp, split)


def _prepare_z_array(z_trial: np.ndarray, split_res: Dict[str, Any], trial_idx: int) -> tuple[np.ndarray, np.ndarray]:
    z_arr = np.asarray(z_trial)
    n_samples = int(z_arr.shape[0]) if z_arr.ndim == 2 else len(z_arr)
    t_abs = get_trial_time_axis(split_res, trial_idx, n_samples)
    z_arr = transpose_if_needed(z_arr, len(t_abs))
    return z_arr, t_abs


def extract_trial_z_series(
    split_res_psid: Dict[str, Any],
    split_res_dpad: Dict[str, Any],
    split_res_varma: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
) -> TrialZSeries:
    """
    Extract raw Z and Zp from three result dicts. Requires the same trial ordering
    across runs (same split parquet / aligned training pipeline).
    """
    for label, res in (
        ("PSID", split_res_psid),
        ("DPAD", split_res_dpad),
        ("VARMA", split_res_varma),
    ):
        if res.get("Z") is None or trial_idx >= len(res["Z"]) or res["Z"][trial_idx] is None:
            raise ValueError(f"{label}: missing Z for trial_idx={trial_idx}")
        if res.get("Zp") is None or trial_idx >= len(res["Zp"]) or res["Zp"][trial_idx] is None:
            raise ValueError(f"{label}: missing Zp for trial_idx={trial_idx}")

    z_true_arr, t_abs = _prepare_z_array(split_res_psid["Z"][trial_idx], split_res_psid, trial_idx)
    true_c = get_channel(z_true_arr, channel_idx, t_abs)

    def zp_chan(res: Dict[str, Any]) -> np.ndarray:
        arr, t2 = _prepare_z_array(res["Zp"][trial_idx], res, trial_idx)
        if len(t2) != len(t_abs):
            raise ValueError("Time axis length mismatch between models for this trial.")
        return get_channel(arr, channel_idx, t_abs)

    return TrialZSeries(
        t_abs=t_abs,
        z_true_raw=true_c,
        z_psid=zp_chan(split_res_psid),
        z_dpad=zp_chan(split_res_dpad),
        z_varma=zp_chan(split_res_varma),
    )


def trial_rmse_z_for_model(
    split_res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
) -> float:
    """Single-model RMSE on z-scored output: sqrt(mean((z_true - z_pred)^2)) for one trial."""
    if split_res.get("Z") is None or trial_idx >= len(split_res["Z"]) or split_res["Z"][trial_idx] is None:
        raise ValueError(f"missing Z for trial_idx={trial_idx}")
    if split_res.get("Zp") is None or trial_idx >= len(split_res["Zp"]) or split_res["Zp"][trial_idx] is None:
        raise ValueError(f"missing Zp for trial_idx={trial_idx}")

    z_true_arr, t_abs = _prepare_z_array(split_res["Z"][trial_idx], split_res, trial_idx)
    true_c = get_channel(z_true_arr, channel_idx, t_abs)
    zp_arr, t2 = _prepare_z_array(split_res["Zp"][trial_idx], split_res, trial_idx)
    if len(t2) != len(t_abs):
        raise ValueError("Time axis mismatch between Z and Zp for this trial.")
    pred_c = get_channel(zp_arr, channel_idx, t_abs)
    zt = zscore_using_true_stats(true_c, true_c)
    zp = zscore_using_true_stats(true_c, pred_c)
    return rmse_z(zt, zp)
