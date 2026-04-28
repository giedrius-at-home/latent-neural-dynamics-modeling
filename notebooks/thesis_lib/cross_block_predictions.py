"""
Stitched trial windows across DBS block boundaries: compare Ŷ from models trained
OFF-only, BOTH, and ON-only (plus cross-eval parquets) on the last OFF trial and
first ON trial of a session (and the reverse).

Rows: one per framework that loads (PSID, DPAD, VARMA in that order). Columns: OFF→ON stitch, ON→OFF stitch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from thesis_lib._helpers import get_channel
from thesis_lib.aggregate_rmse import _trial_key
from thesis_lib.exemplar_trials import (
    find_adjacent_off_then_on_trial_indices,
    find_block_boundary_off_then_on_trial_indices,
)
from thesis_lib.figure import _concat_gap
from thesis_lib.compose import _session_str
from thesis_lib.loaders import (
    _prepare_z_array,
    channels_as_str_list,
    load_split_results,
    load_split_results_required,
    neural_y_feature_label,
    resolve_output_channel_display,
    thesis_exemplar_tagline,
    trial_rmse_y_for_model,
    trial_rmse_z_for_model,
)
from thesis_lib.specs import (
    THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
    AlignedTriplet,
    ThesisCrossBlockSpec,
    _variant_cross_eval,
    _variant_off,
    _variant_on,
    triplet_branch_timestamp,
)
from thesis_lib.plot_scaffolds import (
    THESIS_TIME_AXIS_TITLE,
    apply_grid_xy_subplots,
    cross_block_annotation_font,
)
from thesis_lib.transforms import rmse_z, zscore_using_true_stats
from thesis_lib.constants import (
    WIDTH_TRUE,
    WIDTH_PSID,
    WIDTH_DPAD,
    WIDTH_VARMA,
    apply_thesis_style,
    paper_colors,
    true_line_color,
)

logger = logging.getLogger(__name__)

Framework = Literal["psid", "dpad", "varma"]


def _primary_seq(res: Dict[str, Any], y_mode: Literal["Z", "Y"]) -> Optional[List[Any]]:
    key = "Y" if y_mode == "Y" else "Z"
    return res.get(key)


def _trial_map_for_mode(res: Dict[str, Any], y_mode: Literal["Z", "Y"]) -> Dict[Tuple[Any, ...], int]:
    seq = _primary_seq(res, y_mode)
    if not seq:
        return {}
    out: Dict[Tuple[Any, ...], int] = {}
    for i in range(len(seq)):
        if seq[i] is None:
            continue
        out[_trial_key(res, i)] = i
    return out


def _aligned_trial_idx(
    res: Dict[str, Any],
    joint: Dict[str, Any],
    trial_key: Tuple[Any, ...],
    psid_row: int,
    y_mode: Literal["Z", "Y"],
) -> Optional[int]:
    """Map joint trial key / row to index in ``res``; fall back to same row if tables are length-aligned."""
    m = _trial_map_for_mode(res, y_mode)
    if trial_key in m:
        return m[trial_key]
    s = _primary_seq(res, y_mode)
    j = _primary_seq(joint, y_mode)
    if (
        s
        and j
        and len(s) == len(j)
        and 0 <= psid_row < len(s)
        and s[psid_row] is not None
    ):
        return psid_row
    return None

COLOR_OFF = "#185FA5"
COLOR_BOTH = "#2d6a4f"
COLOR_ON = "#993C1D"
_CONTEXT = "rgba(24, 95, 165, 0.09)"
_FWD = "rgba(153, 60, 29, 0.07)"
_BAND_OFF_FILL = "rgba(24, 95, 165, 0.18)"
_BAND_OFF_LINE = "rgba(24, 95, 165, 0.32)"
_BAND_BOTH_FILL = "rgba(45, 106, 79, 0.18)"
_BAND_BOTH_LINE = "rgba(45, 106, 79, 0.32)"
_BAND_ON_FILL = "rgba(153, 60, 29, 0.16)"
_BAND_ON_LINE = "rgba(153, 60, 29, 0.30)"


def _session_mean_rmse_checkpoint(
    res: Dict[str, Any],
    target_sess: str,
    ch: int,
    y_mode: Literal["Z", "Y"],
) -> Optional[float]:
    """Mean one-step RMSE (z-scored, same trial rule as C2) over test trials in ``target_sess`` for this checkpoint."""
    if not target_sess:
        return None
    key = "Y" if y_mode == "Y" else "Z"
    seq = res.get(key)
    if not seq:
        return None
    vals: list[float] = []
    for i in range(len(seq)):
        if seq[i] is None:
            continue
        if _session_str(res, i) != target_sess:
            continue
        try:
            if y_mode == "Y":
                vals.append(trial_rmse_y_for_model(res, i, ch))
            else:
                vals.append(trial_rmse_z_for_model(res, i, ch))
        except ValueError:
            pass
    if not vals:
        return None
    return float(np.mean(vals))


def _finite_run_slices(t: np.ndarray, y: np.ndarray) -> List[Tuple[int, int]]:
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(t) & np.isfinite(y)
    if t.size == 0 or y.size == 0 or t.size != y.size or not np.any(m):
        return []
    idx = np.flatnonzero(m)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = [int(idx[0])]
    ends: List[int] = []
    for b in breaks:
        ends.append(int(idx[b]) + 1)
        starts.append(int(idx[b + 1]))
    ends.append(int(idx[-1]) + 1)
    return list(zip(starts, ends))


def _add_checkpoint_rmse_bands(
    fig: go.Figure,
    row: int,
    col: int,
    t_plot: np.ndarray,
    z_off: np.ndarray,
    z_both: np.ndarray,
    z_on: np.ndarray,
    half_off: Optional[float],
    half_both: Optional[float],
    half_on: Optional[float],
    *,
    show_legend: bool,
) -> None:
    """Ŷ ± session-mean RMSE ribbons (C2-style), respecting NaN gaps in stitched time."""
    def one(
        y_hat: np.ndarray,
        half: Optional[float],
        fill_c: str,
        line_c: str,
        name: str,
        leg: str,
        show: bool,
    ) -> None:
        if half is None or not np.isfinite(half) or half <= 0:
            return
        yv = np.asarray(y_hat, dtype=float).ravel()
        tp = np.asarray(t_plot, dtype=float).ravel()
        if yv.size != tp.size:
            return
        first_poly = True
        for s, e in _finite_run_slices(tp, yv):
            sl = slice(s, e)
            tt = tp[sl]
            yh = yv[sl]
            if tt.size < 2:
                continue
            u = yh + half
            lo_b = yh - half
            x_band = np.concatenate([tt, tt[::-1]])
            y_band = np.concatenate([u, lo_b[::-1]])
            fig.add_trace(
                go.Scatter(
                    x=x_band,
                    y=y_band,
                    fill="toself",
                    fillcolor=fill_c,
                    line=dict(color=line_c, width=0.5),
                    name=name,
                    legendgroup=leg,
                    showlegend=show and first_poly,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            first_poly = False

    one(
        z_off,
        half_off,
        _BAND_OFF_FILL,
        _BAND_OFF_LINE,
        "Ŷ OFF-trained ± mean RMSE",
        "xb_off",
        show_legend,
    )
    one(
        z_both,
        half_both,
        _BAND_BOTH_FILL,
        _BAND_BOTH_LINE,
        "Ŷ BOTH-trained ± mean RMSE",
        "xb_both",
        show_legend,
    )
    one(
        z_on,
        half_on,
        _BAND_ON_FILL,
        _BAND_ON_LINE,
        "Ŷ ON-trained ± mean RMSE",
        "xb_on",
        show_legend,
    )


def _resolve_y_channel(res: Dict[str, Any], spec: ThesisCrossBlockSpec) -> int:
    if spec.forecast_target != "Y":
        return spec.channel_idx
    raw = (spec.neural_y_feature_name or "").strip()
    if not raw:
        return spec.channel_idx
    names = channels_as_str_list(res.get("input_channels"))
    for i, n in enumerate(names):
        if str(n).strip() == raw:
            return i
    key = raw.lower().replace(" ", "_")
    for i, n in enumerate(names):
        if str(n).lower().replace(" ", "_") == key:
            return i
    return spec.channel_idx


def _series(
    res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    y_mode: Literal["Z", "Y"],
    pred: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    kt, kp = ("Y", "Yp") if y_mode == "Y" else ("Z", "Zp")
    key = kp if pred else kt
    seq = res.get(key)
    if seq is None or trial_idx >= len(seq) or seq[trial_idx] is None:
        raise ValueError(f"missing {key}[{trial_idx}]")
    arr, t = _prepare_z_array(seq[trial_idx], res, trial_idx)
    y = get_channel(arr, channel_idx, t)
    return np.asarray(t, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()


def _tail(t: np.ndarray, *ys: np.ndarray, span_s: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    t = np.asarray(t, dtype=float).ravel()
    if t.size == 0 or span_s <= 0:
        return t, [np.asarray(y, dtype=float).ravel() for y in ys]
    hi = float(np.nanmax(t))
    m = t >= hi - float(span_s)
    return t[m], [np.asarray(y, dtype=float).ravel()[m] for y in ys]


def _head(t: np.ndarray, *ys: np.ndarray, span_s: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    t = np.asarray(t, dtype=float).ravel()
    if t.size == 0 or span_s <= 0:
        return t, [np.asarray(y, dtype=float).ravel() for y in ys]
    lo = float(np.nanmin(t))
    m = t <= lo + float(span_s)
    return t[m], [np.asarray(y, dtype=float).ravel()[m] for y in ys]


@dataclass(frozen=True)
class _CrossFwPack:
    res_j: Dict[str, Any]
    res_o: Dict[str, Any]
    res_n: Dict[str, Any]
    res_eo: Dict[str, Any]
    res_en: Dict[str, Any]
    variant_off: str
    ts_off: str
    variant_on: str
    ts_on: str


def _split_order(preferred: str) -> Tuple[str, ...]:
    """Try ``preferred`` first, then other splits (e.g. VARMA test may omit a boundary trial)."""
    rest = [s for s in ("test", "val", "train") if s != preferred]
    return (preferred, *rest)


def _results_with_trial_alignment(
    results_root: Path,
    variant: str,
    run_ts: str,
    joint_fw: Dict[str, Any],
    trial_key: Tuple[Any, ...],
    joint_row: int,
    y_mode: Literal["Z", "Y"],
    preferred_split: str,
) -> Optional[Dict[str, Any]]:
    for split in _split_order(preferred_split):
        r = load_split_results(results_root, variant, run_ts, split)
        if r is None:
            continue
        if _aligned_trial_idx(r, joint_fw, trial_key, joint_row, y_mode) is not None:
            return r
    return None


def _maybe_reload_off_on_for_boundary(
    results_root: Path,
    pack: _CrossFwPack,
    res_j: Dict[str, Any],
    k_off: Tuple[Any, ...],
    k_on: Tuple[Any, ...],
    i_off: int,
    i_on: int,
    y_mode: Literal["Z", "Y"],
    preferred_split: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """OFF- and ON-only checkpoints often store one split per file; test may miss the DBS-boundary trial."""
    res_o, res_n = pack.res_o, pack.res_n
    if _aligned_trial_idx(res_o, res_j, k_off, i_off, y_mode) is None:
        alt = _results_with_trial_alignment(
            results_root,
            pack.variant_off,
            pack.ts_off,
            res_j,
            k_off,
            i_off,
            y_mode,
            preferred_split,
        )
        if alt is not None:
            res_o = alt
    if _aligned_trial_idx(res_n, res_j, k_on, i_on, y_mode) is None:
        alt = _results_with_trial_alignment(
            results_root,
            pack.variant_on,
            pack.ts_on,
            res_j,
            k_on,
            i_on,
            y_mode,
            preferred_split,
        )
        if alt is not None:
            res_n = alt
    return res_o, res_n


def _load_fw(
    tri: AlignedTriplet,
    fw: Framework,
    results_root: Path,
    split: str,
) -> _CrossFwPack:
    if fw == "psid":
        vj, tj = tri.psid_variant, tri.psid_run_ts
        vo, to = _variant_off(vj), triplet_branch_timestamp(tri, "psid", "off")
        vn, tn = _variant_on(vj), triplet_branch_timestamp(tri, "psid", "on")
        veo, teo = _variant_cross_eval(_variant_on(vj), "off"), triplet_branch_timestamp(
            tri, "psid", "eval_off"
        )
        ven, ten = _variant_cross_eval(_variant_off(vj), "on"), triplet_branch_timestamp(
            tri, "psid", "eval_on"
        )
    elif fw == "dpad":
        vj, tj = tri.dpad_variant, tri.dpad_run_ts
        vo, to = _variant_off(vj), triplet_branch_timestamp(tri, "dpad", "off")
        vn, tn = _variant_on(vj), triplet_branch_timestamp(tri, "dpad", "on")
        veo, teo = _variant_cross_eval(_variant_on(vj), "off"), triplet_branch_timestamp(
            tri, "dpad", "eval_off"
        )
        ven, ten = _variant_cross_eval(_variant_off(vj), "on"), triplet_branch_timestamp(
            tri, "dpad", "eval_on"
        )
    else:
        vj, tj = tri.varma_variant, tri.varma_run_ts
        vo, to = _variant_off(vj), triplet_branch_timestamp(tri, "varma", "off")
        vn, tn = _variant_on(vj), triplet_branch_timestamp(tri, "varma", "on")
        veo, teo = _variant_cross_eval(_variant_on(vj), "off"), triplet_branch_timestamp(
            tri, "varma", "eval_off"
        )
        ven, ten = _variant_cross_eval(_variant_off(vj), "on"), triplet_branch_timestamp(
            tri, "varma", "eval_on"
        )
    return _CrossFwPack(
        res_j=load_split_results_required(results_root, vj, tj, split),
        res_o=load_split_results_required(results_root, vo, to, split),
        res_n=load_split_results_required(results_root, vn, tn, split),
        res_eo=load_split_results_required(results_root, veo, teo, split),
        res_en=load_split_results_required(results_root, ven, ten, split),
        variant_off=vo,
        ts_off=to,
        variant_on=vn,
        ts_on=tn,
    )


def _stitch_off_on(
    res_j: Dict[str, Any],
    res_o: Dict[str, Any],
    res_n: Dict[str, Any],
    res_eo: Dict[str, Any],
    res_en: Dict[str, Any],
    i_off: int,
    i_on: int,
    ch: int,
    y_mode: Literal["Z", "Y"],
    span_s: float,
    gap_s: float,
) -> Optional[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]
]:
    """OFF trial tail + ON trial head. Returns t_plot, z_true, z_off, z_both, z_on, rmse_mean, zref_raw."""
    k_off = _trial_key(res_j, i_off)
    k_on = _trial_key(res_j, i_on)
    io = _aligned_trial_idx(res_o, res_j, k_off, i_off, y_mode)
    ion = _aligned_trial_idx(res_n, res_j, k_on, i_on, y_mode)
    ieo = _aligned_trial_idx(res_eo, res_j, k_off, i_off, y_mode)
    ien = _aligned_trial_idx(res_en, res_j, k_on, i_on, y_mode)
    if io is None or ion is None or ieo is None or ien is None:
        return None

    t0, z0 = _series(res_j, i_off, ch, y_mode, False)
    p0o = _series(res_o, io, ch, y_mode, True)[1]
    p0j = _series(res_j, i_off, ch, y_mode, True)[1]
    p0n = _series(res_eo, ieo, ch, y_mode, True)[1]

    t1, z1 = _series(res_j, i_on, ch, y_mode, False)
    p1o = _series(res_en, ien, ch, y_mode, True)[1]
    p1j = _series(res_j, i_on, ch, y_mode, True)[1]
    p1n = _series(res_n, ion, ch, y_mode, True)[1]

    if min(len(t0), len(z0), len(p0o), len(p0j), len(p0n)) < 2:
        return None
    if min(len(t1), len(z1), len(p1o), len(p1j), len(p1n)) < 2:
        return None

    t0s, (z0s, p0os, p0js, p0ns) = _tail(t0, z0, p0o, p0j, p0n, span_s=span_s)
    t1s, (z1s, p1os, p1js, p1ns) = _head(t1, z1, p1o, p1j, p1n, span_s=span_s)
    if t0s.size < 2 or t1s.size < 2:
        return None

    zref = np.concatenate([z0s, z1s])
    z0t = zscore_using_true_stats(zref, z0s)
    z1t = zscore_using_true_stats(zref, z1s)
    q0o = zscore_using_true_stats(zref, p0os)
    q0b = zscore_using_true_stats(zref, p0js)
    q0n = zscore_using_true_stats(zref, p0ns)
    q1o = zscore_using_true_stats(zref, p1os)
    q1b = zscore_using_true_stats(zref, p1js)
    q1n = zscore_using_true_stats(zref, p1ns)

    t_plot, ys = _concat_gap(
        t0s,
        (z0t, q0o, q0b, q0n),
        t1s,
        (z1t, q1o, q1b, q1n),
        gap_s,
    )
    z_true, z_off, z_both, z_onm = ys[0], ys[1], ys[2], ys[3]

    zt = np.concatenate([z0t, z1t])
    pred_off = np.concatenate([q0o, q1o])
    pred_both = np.concatenate([q0b, q1b])
    pred_on = np.concatenate([q0n, q1n])
    rms = (
        float(rmse_z(zt, pred_off))
        + float(rmse_z(zt, pred_both))
        + float(rmse_z(zt, pred_on))
    ) / 3.0

    zref_raw = np.concatenate([np.asarray(z0s, dtype=float).ravel(), np.asarray(z1s, dtype=float).ravel()])
    return t_plot, z_true, z_off, z_both, z_onm, rms, zref_raw


def _stitch_on_off(
    res_j: Dict[str, Any],
    res_o: Dict[str, Any],
    res_n: Dict[str, Any],
    res_eo: Dict[str, Any],
    res_en: Dict[str, Any],
    i_off: int,
    i_on: int,
    ch: int,
    y_mode: Literal["Z", "Y"],
    span_s: float,
    gap_s: float,
) -> Optional[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]
]:
    """ON trial tail + OFF trial head; last return is zref_raw (unscaled true stitch segments)."""
    k_off = _trial_key(res_j, i_off)
    k_on = _trial_key(res_j, i_on)
    io = _aligned_trial_idx(res_o, res_j, k_off, i_off, y_mode)
    ion = _aligned_trial_idx(res_n, res_j, k_on, i_on, y_mode)
    ieo = _aligned_trial_idx(res_eo, res_j, k_off, i_off, y_mode)
    ien = _aligned_trial_idx(res_en, res_j, k_on, i_on, y_mode)
    if io is None or ion is None or ieo is None or ien is None:
        return None

    t0, z0 = _series(res_j, i_on, ch, y_mode, False)
    p0o = _series(res_en, ien, ch, y_mode, True)[1]
    p0j = _series(res_j, i_on, ch, y_mode, True)[1]
    p0n = _series(res_n, ion, ch, y_mode, True)[1]

    t1, z1 = _series(res_j, i_off, ch, y_mode, False)
    p1o = _series(res_o, io, ch, y_mode, True)[1]
    p1j = _series(res_j, i_off, ch, y_mode, True)[1]
    p1n = _series(res_eo, ieo, ch, y_mode, True)[1]

    t0s, (z0s, p0os, p0js, p0ns) = _tail(t0, z0, p0o, p0j, p0n, span_s=span_s)
    t1s, (z1s, p1os, p1js, p1ns) = _head(t1, z1, p1o, p1j, p1n, span_s=span_s)
    if t0s.size < 2 or t1s.size < 2:
        return None

    zref = np.concatenate([z0s, z1s])
    z0t = zscore_using_true_stats(zref, z0s)
    z1t = zscore_using_true_stats(zref, z1s)
    q0o = zscore_using_true_stats(zref, p0os)
    q0b = zscore_using_true_stats(zref, p0js)
    q0n = zscore_using_true_stats(zref, p0ns)
    q1o = zscore_using_true_stats(zref, p1os)
    q1b = zscore_using_true_stats(zref, p1js)
    q1n = zscore_using_true_stats(zref, p1ns)

    t_plot, ys = _concat_gap(
        t0s,
        (z0t, q0o, q0b, q0n),
        t1s,
        (z1t, q1o, q1b, q1n),
        gap_s,
    )
    z_true, z_off, z_both, z_onm = ys[0], ys[1], ys[2], ys[3]
    zt = np.concatenate([z0t, z1t])
    pred_off = np.concatenate([q0o, q1o])
    pred_both = np.concatenate([q0b, q1b])
    pred_on = np.concatenate([q0n, q1n])
    rms = (
        float(rmse_z(zt, pred_off))
        + float(rmse_z(zt, pred_both))
        + float(rmse_z(zt, pred_on))
    ) / 3.0
    zref_raw = np.concatenate([np.asarray(z0s, dtype=float).ravel(), np.asarray(z1s, dtype=float).ravel()])
    return t_plot, z_true, z_off, z_both, z_onm, rms, zref_raw


def _line_w(fw: Framework) -> float:
    if fw == "psid":
        return WIDTH_PSID
    if fw == "dpad":
        return WIDTH_DPAD
    return WIDTH_VARMA


# SegmentPack: (t, z_true, z_off_trained, z_both_trained, z_on_trained)
SegmentPack = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _extract_single_segment(
    res_j: Dict[str, Any],
    res_o: Dict[str, Any],
    res_n: Dict[str, Any],
    res_eo: Dict[str, Any],
    res_en: Dict[str, Any],
    trial_idx: int,
    is_off_trial: bool,
    ch: int,
    y_mode: Literal["Z", "Y"],
    span_s: float,
) -> Optional[SegmentPack]:
    """
    Extract one trial segment (tail for OFF trial, head for ON trial).
    Z-score each series using the trial's own true-signal statistics.
    Returns (t, z_true, z_off, z_both, z_on) or None on failure.
    """
    k = _trial_key(res_j, trial_idx)

    # For OFF trial: OFF-trained model on this trial + ON-trained cross-eval on this OFF trial
    # For ON trial: ON-trained model on this trial + OFF-trained cross-eval on this ON trial
    if is_off_trial:
        i_within = _aligned_trial_idx(res_o, res_j, k, trial_idx, y_mode)   # OFF-trained
        i_cross = _aligned_trial_idx(res_eo, res_j, k, trial_idx, y_mode)   # ON-trained cross-eval on OFF
        if i_within is None or i_cross is None:
            return None
    else:
        i_cross = _aligned_trial_idx(res_en, res_j, k, trial_idx, y_mode)   # OFF-trained cross-eval on ON
        i_within = _aligned_trial_idx(res_n, res_j, k, trial_idx, y_mode)   # ON-trained
        if i_within is None or i_cross is None:
            return None

    try:
        t, z_true = _series(res_j, trial_idx, ch, y_mode, False)
        p_both = _series(res_j, trial_idx, ch, y_mode, True)[1]
        if is_off_trial:
            p_off = _series(res_o, i_within, ch, y_mode, True)[1]   # OFF-trained
            p_on = _series(res_eo, i_cross, ch, y_mode, True)[1]    # ON-trained cross-eval
        else:
            p_off = _series(res_en, i_cross, ch, y_mode, True)[1]   # OFF-trained cross-eval
            p_on = _series(res_n, i_within, ch, y_mode, True)[1]    # ON-trained
    except Exception:
        return None

    if min(len(t), len(z_true), len(p_off), len(p_both), len(p_on)) < 2:
        return None

    # Extract tail (for OFF trial) or head (for ON trial)
    if is_off_trial:
        t_s, (zt_s, po_s, pb_s, pn_s) = _tail(t, z_true, p_off, p_both, p_on, span_s=span_s)
    else:
        t_s, (zt_s, po_s, pb_s, pn_s) = _head(t, z_true, p_off, p_both, p_on, span_s=span_s)

    if t_s.size < 2:
        return None

    # Z-score using this segment's own statistics
    zt_z = zscore_using_true_stats(zt_s, zt_s)
    po_z = zscore_using_true_stats(zt_s, po_s)
    pb_z = zscore_using_true_stats(zt_s, pb_s)
    pn_z = zscore_using_true_stats(zt_s, pn_s)

    # Shift t to start from 0
    t_rel = t_s - float(np.nanmin(t_s))
    return t_rel, zt_z, po_z, pb_z, pn_z


def build_cross_block_predictions_figure(
    spec: ThesisCrossBlockSpec,
    results_root: Path,
) -> Tuple[go.Figure, str]:
    res_joint = load_split_results_required(
        results_root,
        spec.joint_triplet.psid_variant,
        spec.joint_triplet.psid_run_ts,
        spec.split,
    )
    pair = find_block_boundary_off_then_on_trial_indices(res_joint)
    if pair is None:
        stim = res_joint.get("stim")
        sl = list(stim) if stim is not None else []
        pair = find_adjacent_off_then_on_trial_indices(sl)
    if pair is None:
        raise ValueError("Cross-block figure: no OFF→ON trial pair in joint PSID results.")
    i_off, i_on = pair

    ch = _resolve_y_channel(res_joint, spec)
    y_mode: Literal["Z", "Y"] = "Y" if spec.forecast_target == "Y" else "Z"
    span = float(spec.trial_segment_s)

    paper_bg, plot_bg = paper_colors(spec.theme)
    fg = true_line_color(spec.theme)
    c_true = true_line_color(spec.theme)

    fw_list: List[Framework] = ["psid", "dpad", "varma"]
    skipped: List[str] = []
    built: List[Tuple[Framework, SegmentPack, SegmentPack]] = []

    for fw in fw_list:
        try:
            packs = _load_fw(spec.joint_triplet, fw, results_root, spec.split)
        except Exception as e:
            logger.warning("Cross-block %s: load failed (%s)", fw, e)
            skipped.append(f"{fw} (load: {str(e)[:80]})")
            continue
        res_j = packs.res_j
        k_off = _trial_key(res_j, i_off)
        k_on = _trial_key(res_j, i_on)
        res_o, res_n = _maybe_reload_off_on_for_boundary(
            results_root, packs, res_j, k_off, k_on, i_off, i_on, y_mode, spec.split,
        )
        seg_off = _extract_single_segment(
            res_j, res_o, res_n, packs.res_eo, packs.res_en,
            i_off, True, ch, y_mode, span,
        )
        seg_on = _extract_single_segment(
            res_j, res_o, res_n, packs.res_eo, packs.res_en,
            i_on, False, ch, y_mode, span,
        )
        if seg_off is None or seg_on is None:
            logger.warning("Cross-block %s: segment extraction failed.", fw)
            skipped.append(f"{fw} (segment)")
            continue
        built.append((fw, seg_off, seg_on))

    if not built:
        detail = "; ".join(skipped) if skipped else "no detail"
        raise ValueError(
            f"Cross-block figure: no framework produced data. Skipped: {detail}"
        )

    n_rows = len(built)

    # Build subplot titles: one per (row × col), col1=OFF, col2=ON
    subplot_titles: List[str] = []
    for fw, _, _ in built:
        subplot_titles.append(f"{fw.upper()}: last OFF trial")
        subplot_titles.append(f"{fw.upper()}: first ON trial")

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        shared_yaxes=True,
        shared_xaxes=False,
        vertical_spacing=min(0.16, 0.065 + 0.035 * n_rows),
        horizontal_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    for ri, (fw, seg_off, seg_on) in enumerate(built, start=1):
        lw = _line_w(fw)
        dash_on = "8 2" if fw == "varma" else None

        for ci, (seg, bg_color) in enumerate(
            [(seg_off, _CONTEXT), (seg_on, _FWD)], start=1
        ):
            t_p, zt, zo, zb, zn = seg
            show_leg = ri == 1 and ci == 1

            x0 = float(np.nanmin(t_p))
            x1 = float(np.nanmax(t_p))
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=bg_color,
                layer="below", line_width=0,
                row=ri, col=ci,
            )

            fig.add_trace(
                go.Scatter(
                    x=t_p, y=zt,
                    name="y_true", mode="lines",
                    line=dict(color=c_true, width=WIDTH_TRUE),
                    showlegend=show_leg, legendgroup="true",
                    connectgaps=False,
                ),
                row=ri, col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_p, y=zo,
                    name="Ŷ OFF-trained", mode="lines",
                    line=dict(color=COLOR_OFF, width=lw),
                    showlegend=show_leg, legendgroup="off",
                    connectgaps=False,
                ),
                row=ri, col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_p, y=zb,
                    name="Ŷ BOTH-trained", mode="lines",
                    line=dict(color=COLOR_BOTH, width=lw),
                    showlegend=show_leg, legendgroup="both",
                    connectgaps=False,
                ),
                row=ri, col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_p, y=zn,
                    name="Ŷ ON-trained", mode="lines",
                    line=dict(color=COLOR_ON, width=lw, dash=dash_on),
                    showlegend=show_leg, legendgroup="on",
                    connectgaps=False,
                ),
                row=ri, col=ci,
            )

    apply_grid_xy_subplots(
        fig, n_rows=n_rows, n_cols=2, theme=spec.theme, nticks=8, x_title=THESIS_TIME_AXIS_TITLE
    )

    apply_thesis_style(
        fig,
        spec.theme,
        height=int(min(960, max(380, 260 * n_rows + 180))),
        margin=dict(l=80, r=28, t=56, b=120),
        legend_y=-0.08,
    )
    fig.update_annotations(font=cross_block_annotation_font(theme=spec.theme))

    _ylab_override = (spec.y_axis_label or "").strip()
    if _ylab_override:
        ylab = _ylab_override
    elif y_mode == "Z":
        och, _ = resolve_output_channel_display(
            res_joint, ch, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS
        )
        from thesis_lib.constants import d_score_axis_label
        ylab = d_score_axis_label(och)
    else:
        feat_disp = neural_y_feature_label(
            res_joint, ch, neural_y_feature_name=spec.neural_y_feature_name,
        )
        ylab = feat_disp  # neural: raw feature name, no prefix

    y_title_row = max(1, (n_rows + 1) // 2)
    fig.update_yaxes(title_text=ylab, row=y_title_row, col=1)

    feat = och if y_mode == "Z" else feat_disp
    cap = thesis_exemplar_tagline(
        res_joint, i_off, i_on, feat, participant_label=spec.participant_label,
    )
    sc = (spec.caption or "").strip()
    if sc:
        cap = f"{sc} · {cap}"
    cap += f" Left: last OFF trial ({span:.1f} s); right: first ON trial ({span:.1f} s). Independent x-axes."
    if skipped:
        cap += f" · Skipped: {'; '.join(skipped)}."
    return fig, cap
