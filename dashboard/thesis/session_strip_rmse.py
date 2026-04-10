"""
B2: Per-participant session-mean RMSE (z-scored output) for strip plots.
One dot = mean trial RMSE within one test session × model × DBS condition.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from dashboard.thesis.aggregate_rmse import _key_index_map, normalize_stim
from dashboard.thesis.loaders import load_split_results, trial_rmse_z_for_model
from dashboard.thesis.specs import AlignedTriplet

logger = logging.getLogger(__name__)

SessionKey = Tuple[str, str]


def _session_key_from_trial_key(trial_key: Tuple) -> SessionKey:
    pid, sess = trial_key[0], trial_key[1]
    return (str(pid), str(sess))


@dataclass
class StripPanelData:
    """Six cells: PSID/DPAD/VARMA × OFF/ON — session-mean and trial-level RMSE per cell."""

    panel_label: str
    triplet_label: str
    # Lists of session-mean RMSE (one float per session contributing to that cell)
    session_means: Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]
    mean_line_y: Tuple[float, float, float, float, float, float]
    # Individual trial RMSE values per cell (for box plots)
    trial_rmse: Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]


@dataclass
class StripFigureData:
    panels: List[StripPanelData]
    y_max: float


def _one_panel_session_data(
    results_root: Path,
    tri: AlignedTriplet,
    channel_idx: int,
    split: str,
) -> StripPanelData | None:
    res_p = load_split_results(results_root, tri.psid_variant, tri.psid_run_ts, split)
    res_v = load_split_results(results_root, tri.varma_variant, tri.varma_run_ts, split)
    if res_p is None:
        logger.warning("Strip panel %s: missing PSID results", tri.label)
        return None

    # Frameworks to iterate: PSID always, DPAD only if results exist, VARMA always.
    frameworks: Dict[str, Dict] = {"psid": res_p}
    if tri.dpad_run_ts:
        res_d = load_split_results(results_root, tri.dpad_variant, tri.dpad_run_ts, split)
        if res_d is not None:
            frameworks["dpad"] = res_d
    if res_v is not None:
        frameworks["varma"] = res_v

    mp = _key_index_map(res_p)
    key_maps = {name: _key_index_map(res) for name, res in frameworks.items()}
    common = set(mp.keys())

    bucket: Dict[Tuple[SessionKey, str, str], List[float]] = {}

    for k in common:
        stim = normalize_stim(res_p["stim"][mp[k]])
        if stim is None:
            continue
        sk = _session_key_from_trial_key(k)
        for model_name, res in frameworks.items():
            idx = key_maps[model_name].get(k)
            if idx is None:
                continue
            r = trial_rmse_z_for_model(res, idx, channel_idx)
            bucket.setdefault((sk, stim, model_name), []).append(r)

    # Session mean per bucket (mean of trial RMSEs in that session × stim × model)
    session_means: List[List[float]] = [[] for _ in range(6)]
    trial_rmse: List[List[float]] = [[] for _ in range(6)]

    def _cell_idx(stim: str, model: str) -> int:
        off = stim == "off"
        if model == "psid":
            return 0 if off else 1
        if model == "dpad":
            return 2 if off else 3
        return 4 if off else 5

    for (sk, stim, model), trial_rmses in bucket.items():
        idx = _cell_idx(stim, model)
        if trial_rmses:
            session_means[idx].append(float(np.mean(trial_rmses)))
            trial_rmse[idx].extend([float(v) for v in trial_rmses if np.isfinite(v)])

    mean_line_y = tuple(
        float(np.mean(session_means[i])) if session_means[i] else float("nan") for i in range(6)
    )

    return StripPanelData(
        panel_label="",
        triplet_label=tri.label,
        session_means=tuple(session_means[i] for i in range(6)),
        mean_line_y=mean_line_y,
        trial_rmse=tuple(trial_rmse[i] for i in range(6)),
    )


def collect_strip_figure_data(
    results_root: Path,
    panel_triplets: List[Tuple[str, AlignedTriplet]],
    channel_idx: int,
    split: str = "test",
    y_margin: float = 0.04,
) -> StripFigureData | None:
    """
    `panel_triplets`: list of (panel_label, AlignedTriplet), e.g. [("P01", tri1), ...].
    """
    panels: List[StripPanelData] = []
    all_vals: List[float] = []

    for panel_label, tri in panel_triplets:
        p = _one_panel_session_data(results_root, tri, channel_idx, split)
        if p is None:
            continue
        p = StripPanelData(
            panel_label=panel_label,
            triplet_label=p.triplet_label,
            session_means=p.session_means,
            mean_line_y=p.mean_line_y,
            trial_rmse=p.trial_rmse,
        )
        panels.append(p)
        for cell in p.trial_rmse:
            for v in cell:
                if np.isfinite(v):
                    all_vals.append(v)

    if not panels:
        return None

    ymax = max(all_vals) if all_vals else 0.85
    ymax = ymax * (1.0 + y_margin) if ymax > 0 else 0.85

    return StripFigureData(panels=panels, y_max=float(ymax))
