"""
Pool mean test-set Pearson r (neural self-prediction) by spectral band × model × DBS.

Bands are parent labels (Delta–Beta) parsed from `input_channels` (e.g. ECOG_1_delta_05_raw).
Per trial: Pearson r per output channel via `_pearson_list_2d`, then mean r within each band.
Band means are simple arithmetic means across channels; Fisher-z pooling can be added later.

VARMA is included alongside PSID and DPAD; all three must expose aligned `Y` / `Yp` for neural channels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from thesis_lib.aggregate_rmse import (
    TrialKey,
    _trial_key,
    normalize_stim,
)
from thesis_lib.loaders import load_split_results
from utils.stats import _pearson_list_2d

if TYPE_CHECKING:
    from thesis_lib.specs import AlignedTriplet

logger = logging.getLogger(__name__)

# Parent band token (lowercase) → display label (heatmap rows)
_BAND_DISPLAY = {
    "delta": "Delta",
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
}
_ALLOWED = frozenset(_BAND_DISPLAY.keys())

_MODEL_KEYS = ("psid", "dpad", "varma")  # column order


def _transpose_if_needed(data: np.ndarray, expected_len: int) -> np.ndarray:
    if (
        data.ndim == 2
        and data.shape[0] != expected_len
        and data.shape[1] == expected_len
    ):
        return data.T
    return data


def _n_time_samples(split_res: Dict[str, Any], trial_idx: int, arr: np.ndarray) -> int:
    tlist = split_res.get("time") or split_res.get("time_abs")
    if tlist and trial_idx < len(tlist) and tlist[trial_idx] is not None:
        t = np.asarray(tlist[trial_idx])
        return int(t.size) if t.size else int(arr.shape[0])
    return int(arr.shape[0])


def parse_parent_band(channel_name: str) -> Optional[str]:
    """
    Map a neural input channel name to a display band label (Delta–Beta only; gamma ignored).
    Expects `ECOG_<n>_<band>_...` or `LFP_<n>_<band>_...`; also scans other underscore tokens
    (e.g. unusual middle fields before ``beta_13_env``).
    """
    parts = str(channel_name).split("_")
    if len(parts) >= 3:
        if parts[0].upper() in ("ECOG", "LFP") and parts[1].isdigit():
            key = parts[2].lower()
            if key in _ALLOWED:
                return _BAND_DISPLAY[key]
    for p in parts:
        pl = p.lower()
        if pl in _ALLOWED:
            return _BAND_DISPLAY[pl]
    return None


def _band_indices_for_channels(input_channels: List[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, ch in enumerate(input_channels):
        lab = parse_parent_band(ch)
        if lab is None:
            continue
        out.setdefault(lab, []).append(i)
    return out


def _key_index_map_y(split_res: Dict[str, Any]) -> Dict[TrialKey, int]:
    yl = split_res.get("Y")
    if not yl:
        return {}
    out: Dict[TrialKey, int] = {}
    for i in range(len(yl)):
        if yl[i] is None:
            continue
        out[_trial_key(split_res, i)] = i
    return out


def _channels_match(a: List[str], b: List[str], c: List[str]) -> bool:
    return tuple(a) == tuple(b) == tuple(c)


def _prepare_y_trial(
    split_res: Dict[str, Any],
    trial_idx: int,
    y_trial: Any,
    yp_trial: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    if y_trial is None or yp_trial is None:
        raise ValueError("missing Y or Yp")
    yt = np.asarray(y_trial, dtype=float)
    yp = np.asarray(yp_trial, dtype=float)
    if yt.ndim != 2 or yp.ndim != 2:
        raise ValueError("Y/Yp must be 2D (time × channels)")
    n = _n_time_samples(split_res, trial_idx, yt)
    yt = _transpose_if_needed(yt, n)
    yp = _transpose_if_needed(yp, n)
    if yt.shape != yp.shape:
        raise ValueError("Y and Yp shape mismatch after transpose")
    return yt, yp


def _band_mean_r(
    r_per_ch: List[float],
    band_indices: Dict[str, List[int]],
    band_label: str,
) -> Optional[float]:
    idxs = band_indices.get(band_label)
    if not idxs:
        return None
    vals = []
    for j in idxs:
        if j >= len(r_per_ch):
            continue
        r = r_per_ch[j]
        if r is not None and np.isfinite(r):
            vals.append(float(r))
    if not vals:
        return None
    return float(np.mean(vals))


@dataclass
class NeuralBandHeatmapData:
    """Two panels (DBS-OFF / DBS-ON); columns PSID, DPAD, VARMA."""

    z_off: np.ndarray  # (n_bands, 3)
    z_on: np.ndarray
    band_labels: List[str]
    column_labels: Tuple[str, ...] = ("PSID", "DPAD", "VARMA")
    n_triplets_used: int = 0
    n_trials_off: int = 0
    n_trials_on: int = 0


def collect_neural_band_pearson(
    results_root: Path,
    triplet_specs: List["AlignedTriplet"],
    split: str = "test",
    band_row_order: Optional[Sequence[str]] = None,
) -> NeuralBandHeatmapData:
    """
    Align trials on `(participant_id, session, block, trial)` using `Y` lists
    (same as RMSE alignment, but keyed on Y instead of Z).
    """
    order = list(band_row_order) if band_row_order else ["Delta", "Theta", "Alpha", "Beta"]

    off_buckets: Dict[Tuple[str, str], List[float]] = {}
    on_buckets: Dict[Tuple[str, str], List[float]] = {}
    n_triplets_used = 0
    n_off = 0
    n_on = 0

    for tri in triplet_specs:
        res_p = load_split_results(results_root, tri.psid_variant, tri.psid_run_ts, split)
        res_d = load_split_results(results_root, tri.dpad_variant, tri.dpad_run_ts, split)
        res_v = load_split_results(results_root, tri.varma_variant, tri.varma_run_ts, split)
        if res_p is None or res_d is None:
            logger.warning(
                "Skipping triplet %s: missing PSID or DPAD results (psid=%s, dpad=%s)",
                getattr(tri, "label", ""), res_p is not None, res_d is not None,
            )
            continue

        ch_p = list(res_p.get("input_channels") or [])
        ch_d = list(res_d.get("input_channels") or [])
        ch_v = list(res_v.get("input_channels") or []) if res_v is not None else []
        channels = ch_p or ch_d or ch_v
        if not channels:
            raise ValueError(
                f"Triplet {getattr(tri, 'label', '')}: input_channels missing from all models "
                f"(PSID={bool(ch_p)}, DPAD={bool(ch_d)}, VARMA={bool(ch_v)})"
            )

        band_indices = _band_indices_for_channels(channels)
        if not band_indices:
            logger.warning(
                "Skipping triplet %s: no Delta–Beta channels in input_channels.",
                getattr(tri, "label", ""),
            )
            continue

        mp = _key_index_map_y(res_p)
        md = _key_index_map_y(res_d)
        mv = _key_index_map_y(res_v) if res_v is not None else {}
        common_pd = set(mp.keys()) & set(md.keys())
        if not common_pd:
            logger.warning(
                "Skipping triplet %s: no overlapping Y trial keys between PSID and DPAD.",
                getattr(tri, "label", ""),
            )
            continue

        n_triplets_used += 1
        for k in sorted(common_pd, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))):
            i_p, i_d = mp[k], md[k]
            i_v = mv.get(k)
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue
            target = off_buckets if stim == "off" else on_buckets
            if stim == "off":
                n_off += 1
            else:
                n_on += 1

            triple = [
                ("psid", res_p, i_p),
                ("dpad", res_d, i_d),
            ]
            if i_v is not None and res_v is not None:
                triple.append(("varma", res_v, i_v))
            for model_key, res, tidx in triple:
                try:
                    yt, yp = _prepare_y_trial(res, tidx, res["Y"][tidx], res["Yp"][tidx])
                    r_list = _pearson_list_2d(yt, yp)
                except Exception as e:
                    logger.debug("Skip trial %s model %s: %s", k, model_key, e)
                    continue
                for band_label in band_indices:
                    band_r = _band_mean_r(r_list, band_indices, band_label)
                    if band_r is None:
                        continue
                    key = (band_label, model_key)
                    target.setdefault(key, []).append(band_r)

    bands_present: set[str] = set()
    for d in (off_buckets, on_buckets):
        for (b, _) in d.keys():
            bands_present.add(b)
    row_labels = [b for b in order if b in bands_present]

    def _zmat(buckets: Dict[Tuple[str, str], List[float]]) -> np.ndarray:
        z = np.full((len(row_labels), 3), np.nan, dtype=float)
        for bi, band in enumerate(row_labels):
            for mi, mk in enumerate(_MODEL_KEYS):
                vals = buckets.get((band, mk), [])
                if vals:
                    z[bi, mi] = float(np.mean(np.asarray(vals, dtype=float)))
        return z

    return NeuralBandHeatmapData(
        z_off=_zmat(off_buckets),
        z_on=_zmat(on_buckets),
        band_labels=row_labels,
        n_triplets_used=n_triplets_used,
        n_trials_off=n_off,
        n_trials_on=n_on,
    )
