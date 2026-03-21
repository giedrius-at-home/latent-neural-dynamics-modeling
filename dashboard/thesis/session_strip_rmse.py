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
    """Six cells: PSID/DPAD/VARMA × OFF/ON — session-mean RMSE values per cell."""

    panel_label: str
    triplet_label: str
    # Lists of session-mean RMSE (one float per session contributing to that cell)
    session_means: Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]
    mean_line_y: Tuple[float, float, float, float, float, float]


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
    res_d = load_split_results(results_root, tri.dpad_variant, tri.dpad_run_ts, split)
    res_v = load_split_results(results_root, tri.varma_variant, tri.varma_run_ts, split)
    if res_p is None or res_d is None or res_v is None:
        logger.warning(
            "Strip panel %s: missing results (psid=%s, dpad=%s, varma=%s)",
            tri.label,
            res_p is not None,
            res_d is not None,
            res_v is not None,
        )
        return None

    mp = _key_index_map(res_p)
    md = _key_index_map(res_d)
    mv = _key_index_map(res_v)
    common = set(mp.keys()) & set(md.keys()) & set(mv.keys())
    if not common:
        logger.warning(
            "Strip panel %s: no overlapping trial keys across PSID/DPAD/VARMA.",
            tri.label,
        )
        return None

    # bucket[(session_key, stim, model)] -> list of trial-level RMSE
    bucket: Dict[Tuple[SessionKey, str, str], List[float]] = {}

    for k in common:
        i_p, i_d, i_v = mp[k], md[k], mv[k]
        stim = normalize_stim(res_p["stim"][i_p])
        if stim is None:
            continue
        sk = _session_key_from_trial_key(k)
        try:
            r_p = trial_rmse_z_for_model(res_p, i_p, channel_idx)
            r_d = trial_rmse_z_for_model(res_d, i_d, channel_idx)
            r_v = trial_rmse_z_for_model(res_v, i_v, channel_idx)
        except Exception as e:
            logger.debug("Strip skip trial %s: %s", k, e)
            continue

        for model, rv in (("psid", r_p), ("dpad", r_d), ("varma", r_v)):
            key = (sk, stim, model)
            bucket.setdefault(key, []).append(rv)

    # Session mean per bucket (mean of trial RMSEs in that session × stim × model)
    session_means: List[List[float]] = [[] for _ in range(6)]

    def append_cell(idx: int, vals: List[float]) -> None:
        if not vals:
            return
        session_means[idx].append(float(np.mean(vals)))

    for (sk, stim, model), trial_rmses in bucket.items():
        if stim == "off":
            if model == "psid":
                append_cell(0, trial_rmses)
            elif model == "dpad":
                append_cell(2, trial_rmses)
            else:
                append_cell(4, trial_rmses)
        else:
            if model == "psid":
                append_cell(1, trial_rmses)
            elif model == "dpad":
                append_cell(3, trial_rmses)
            else:
                append_cell(5, trial_rmses)

    mean_line_y = tuple(
        float(np.mean(session_means[i])) if session_means[i] else float("nan") for i in range(6)
    )

    return StripPanelData(
        panel_label="",
        triplet_label=tri.label,
        session_means=tuple(session_means[i] for i in range(6)),
        mean_line_y=mean_line_y,
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
        )
        panels.append(p)
        for cell in p.session_means:
            for v in cell:
                if np.isfinite(v):
                    all_vals.append(v)

    if not panels:
        return None

    ymax = max(all_vals) if all_vals else 0.85
    ymax = ymax * (1.0 + y_margin) if ymax > 0 else 0.85

    return StripFigureData(panels=panels, y_max=float(ymax))
