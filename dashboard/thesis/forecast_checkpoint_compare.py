"""
Multi-step forecast comparison on single trials: OFF-trained, BOTH-trained, and ON-trained
checkpoints (plus cross-eval parquets where the trial DBS label mismatches training).

Layout: 3×2 (PSID / DPAD / VARMA × DBS-OFF trial / DBS-ON trial), same boundary trials as cross-block.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.thesis.aggregate_rmse import _key_index_map, _trial_key
from dashboard.thesis.c2_forecast_timeseries import (
    _absolute_time_hist_then_forecast,
    _fc_channel,
    _history_trace,
    _history_trace_before_forecast,
    _infer_history_len,
    _insert_hist_forecast_gap,
)
from dashboard.thesis.cross_block_predictions import (
    COLOR_BOTH,
    COLOR_OFF,
    COLOR_ON,
    Framework,
    _aligned_trial_idx,
    _load_fw,
    _maybe_reload_off_on_for_boundary,
)
from dashboard.thesis.exemplar_trials import (
    find_adjacent_off_then_on_trial_indices,
    find_block_boundary_off_then_on_trial_indices,
)
from dashboard.thesis.loaders import (
    load_split_results_required,
    neural_y_feature_label,
    resolve_neural_y_channel_idx_from_candidates,
    resolve_output_channel_display,
    split_res_with_nonempty_input_channels,
    thesis_exemplar_tagline,
)
from dashboard.thesis.plot_scaffolds import (
    THESIS_TIME_AXIS_TITLE,
    apply_grid_xy_subplots,
    cross_block_annotation_font,
)
from dashboard.thesis.specs import (
    THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
    AlignedTriplet,
    ThesisForecastCheckpointSpec,
    infer_varma_off_on_run_ts,
)
from dashboard.thesis.transforms import zscore_using_true_stats
from dashboard.thesis.constants import (
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_TICK,
    WIDTH_DPAD,
    WIDTH_PSID,
    WIDTH_TRUE,
    WIDTH_VARMA,
    legend_bgcolor,
    paper_colors,
    true_line_color,
)

logger = logging.getLogger(__name__)

_CONTEXT = "rgba(24, 95, 165, 0.05)"
_FORECAST = "rgba(153, 60, 29, 0.05)"
_RULE = "rgba(68, 68, 65, 0.5)"


def _line_w(fw: Framework) -> float:
    if fw == "psid":
        return WIDTH_PSID
    if fw == "dpad":
        return WIDTH_DPAD
    return WIDTH_VARMA


def _varma_ref_and_trial(
    tri: AlignedTriplet,
    results_root: Path,
    split: str,
    res_psid: Dict[str, Any],
    res_j_varma: Dict[str, Any],
    psid_trial_idx: int,
    panel: Literal["off", "on"],
) -> Tuple[Dict[str, Any], int]:
    """VARMA dbs_both: prefer OFF/ON split parquets when the trial key maps there (C2-style)."""
    if "dbs_both" not in tri.varma_variant:
        return res_j_varma, psid_trial_idx
    v_off = tri.varma_variant.replace("dbs_both", "dbs_off")
    v_on = tri.varma_variant.replace("dbs_both", "dbs_on")
    ts_off = tri.varma_run_ts_off
    ts_on = tri.varma_run_ts_on
    if ts_off is None or ts_on is None:
        inf = infer_varma_off_on_run_ts(tri.varma_variant, tri.varma_run_ts)
        if inf is not None:
            ts_off, ts_on = inf
    if ts_off is None or ts_on is None:
        return res_j_varma, psid_trial_idx
    res_v_off = load_split_results_required(results_root, v_off, ts_off, split)
    res_v_on = load_split_results_required(results_root, v_on, ts_on, split)
    k = _trial_key(res_psid, psid_trial_idx)
    if panel == "off":
        m = _key_index_map(res_v_off)
        if k in m:
            return res_v_off, m[k]
    else:
        m = _key_index_map(res_v_on)
        if k in m:
            return res_v_on, m[k]
    return res_j_varma, psid_trial_idx


def _has_fc(
    res: Dict[str, Any], tidx: int, k_true: str, k_pred: str
) -> bool:
    zt = res.get(k_true)
    zp = res.get(k_pred)
    if zt is None or zp is None:
        return False
    if tidx >= len(zt) or tidx >= len(zp):
        return False
    return zt[tidx] is not None and zp[tidx] is not None


def _true_and_pred_fc(
    res: Dict[str, Any],
    tidx: int,
    ch: int,
    k_true: str,
    k_pred: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not _has_fc(res, tidx, k_true, k_pred):
        return None
    try:
        return _fc_channel(res[k_true][tidx], res[k_pred][tidx], ch)
    except Exception:
        return None


def _build_checkpoint_cell(
    *,
    res_ref: Dict[str, Any],
    trial_ref: int,
    tripreds: List[Tuple[str, Dict[str, Any], Optional[int]]],
    channel_idx: int,
    forecast_target: Literal["Z", "Y"],
    history_ms: float,
    forecast_ms: float,
    sampling_hz: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    tripreds: list of (label for logging, res_dict, trial_idx or None) for OFF-, BOTH-, ON-trained preds.
    """
    y_mode = forecast_target == "Y"
    k_true = "Y_future_true" if y_mode else "Z_future_true"
    k_pred = "Y_future_pred" if y_mode else "Z_future_pred"
    concat_key = "Y_concat_for_plot" if y_mode else "Z_concat_for_plot"
    hist_key = "Y" if y_mode else "Z"

    ref_fc = _true_and_pred_fc(res_ref, trial_ref, channel_idx, k_true, k_pred)
    if ref_fc is None:
        logger.warning(
            "Forecast checkpoint: reference trial %s missing %s/%s",
            trial_ref,
            k_true,
            k_pred,
        )
        return None
    zt_fc, _ = ref_fc

    pred_arrays: List[Optional[np.ndarray]] = []
    sizes: List[int] = [int(zt_fc.size)]
    for label, res, tidx in tripreds:
        if tidx is None or res is None:
            pred_arrays.append(None)
            continue
        tp = _true_and_pred_fc(res, tidx, channel_idx, k_true, k_pred)
        if tp is None:
            logger.debug(
                "Forecast checkpoint: %s missing forecast at trial map %s",
                label,
                tidx,
            )
            pred_arrays.append(None)
        else:
            _, zp = tp
            pred_arrays.append(zp)
            sizes.append(int(zp.size))

    m_full = min(sizes) if sizes else 0
    if m_full <= 0:
        return None

    n_fc_cap = int(round(forecast_ms / 1000.0 * sampling_hz))
    m_fc = min(m_full, n_fc_cap)
    zt_fc = zt_fc[:m_fc]

    def _clip_pred(a: Optional[np.ndarray]) -> np.ndarray:
        if a is None:
            return np.full(m_fc, np.nan, dtype=float)
        return np.asarray(a[:m_fc], dtype=float)

    zpreds = [_clip_pred(a) for a in pred_arrays]

    h_infer = _infer_history_len(res_ref, trial_ref, m_full, concat_key)
    n_hist_cap = int(round(history_ms / 1000.0 * sampling_hz))
    if h_infer is not None:
        n_hist_want = min(h_infer, n_hist_cap)
        th = _history_trace_before_forecast(
            res_ref, trial_ref, channel_idx, n_hist_want, h_infer, hist_key=hist_key
        )
        hist_end = int(h_infer)
    else:
        n_hist_want = n_hist_cap
        th = _history_trace(res_ref, trial_ref, channel_idx, n_hist_want, hist_key=hist_key)
        hist_end = len(th)
    n_hist = len(th)

    t_abs = _absolute_time_hist_then_forecast(
        res_ref,
        trial_ref,
        hist_key,
        hist_end,
        n_hist,
        m_fc,
        sampling_hz,
    )
    if t_abs.size != n_hist + m_fc:
        n_tot = n_hist + m_fc
        dt = 1.0 / max(sampling_hz, 1e-6)
        t_abs = np.arange(n_tot, dtype=float) * dt

    true_concat = np.concatenate([th, zt_fc])
    z_true_plot = zscore_using_true_stats(true_concat, true_concat)

    def _pred_line(pred_fc: np.ndarray) -> np.ndarray:
        if pred_fc is None or np.all(np.isnan(pred_fc)):
            out = np.full(n_hist + m_fc, np.nan, dtype=float)
            return out
        pred_full = np.concatenate([th, pred_fc])
        z = zscore_using_true_stats(true_concat, pred_full)
        out = z.copy()
        out[:n_hist] = np.nan
        return out

    z_off = _pred_line(zpreds[0])
    z_both = _pred_line(zpreds[1])
    z_on = _pred_line(zpreds[2])

    return t_abs, z_true_plot, z_off, z_both, z_on, n_hist


def build_forecast_checkpoint_compare_figure(
    spec: ThesisForecastCheckpointSpec,
    results_root: Path,
) -> Tuple[go.Figure, str]:
    res_psid = load_split_results_required(
        results_root,
        spec.joint_triplet.psid_variant,
        spec.joint_triplet.psid_run_ts,
        spec.split,
    )
    pair = find_block_boundary_off_then_on_trial_indices(res_psid)
    if pair is None:
        stim = res_psid.get("stim")
        sl = list(stim) if stim is not None else []
        pair = find_adjacent_off_then_on_trial_indices(sl)
    if pair is None:
        raise ValueError("Forecast checkpoint figure: no OFF→ON trial pair in joint PSID results.")
    i_off, i_on = pair

    tri = spec.joint_triplet
    res_d = load_split_results_required(
        results_root, tri.dpad_variant, tri.dpad_run_ts, spec.split
    )
    res_v = load_split_results_required(
        results_root, tri.varma_variant, tri.varma_run_ts, spec.split
    )
    if spec.forecast_target == "Y":
        ch = resolve_neural_y_channel_idx_from_candidates(
            spec.neural_y_feature_name,
            spec.channel_idx,
            res_psid,
            res_d,
            res_v,
        )
    else:
        ch = spec.channel_idx
    meta_res = split_res_with_nonempty_input_channels(res_psid, res_d, res_v) or res_psid
    y_mode: Literal["Z", "Y"] = "Y" if spec.forecast_target == "Y" else "Z"
    paper_bg, plot_bg = paper_colors(spec.theme)
    fg = true_line_color(spec.theme)
    c_true = true_line_color(spec.theme)

    fw_list: List[Framework] = ["psid", "dpad", "varma"]
    skipped: List[str] = []
    built_rows: List[Tuple[Framework, bool, bool]] = []

    subplot_titles: List[str] = []
    for fw in fw_list:
        left = f"{fw.upper()} · DBS-OFF trial (multi-step Ŷ)"
        right = f"{fw.upper()} · DBS-ON trial (multi-step Ŷ)"
        subplot_titles.extend([left, right])

    fig = make_subplots(
        rows=3,
        cols=2,
        shared_yaxes="rows",
        shared_xaxes=False,
        vertical_spacing=0.09,
        horizontal_spacing=0.06,
        subplot_titles=tuple(subplot_titles) if subplot_titles else None,
    )

    for ri, fw in enumerate(fw_list, start=1):
        try:
            pack = _load_fw(tri, fw, results_root, spec.split)
        except Exception as e:
            logger.warning("Forecast checkpoint %s: load failed (%s)", fw, e)
            skipped.append(f"{fw} (load)")
            continue

        res_j = pack.res_j
        k_off = _trial_key(res_j, i_off)
        k_on = _trial_key(res_j, i_on)
        res_o, res_n = _maybe_reload_off_on_for_boundary(
            results_root,
            pack,
            res_j,
            k_off,
            k_on,
            i_off,
            i_on,
            y_mode,
            spec.split,
        )

        io = _aligned_trial_idx(res_o, res_j, k_off, i_off, y_mode)
        ion = _aligned_trial_idx(res_n, res_j, k_on, i_on, y_mode)
        ieo = _aligned_trial_idx(pack.res_eo, res_j, k_off, i_off, y_mode)
        ien = _aligned_trial_idx(pack.res_en, res_j, k_on, i_on, y_mode)

        i_j_off = _aligned_trial_idx(res_j, res_j, k_off, i_off, y_mode)
        i_j_on = _aligned_trial_idx(res_j, res_j, k_on, i_on, y_mode)
        if i_j_off is None:
            i_j_off = i_off
        if i_j_on is None:
            i_j_on = i_on

        res_ref_off, tr_off = _varma_ref_and_trial(
            tri, results_root, spec.split, res_psid, res_j, i_off, "off"
        )
        res_ref_on, tr_on = _varma_ref_and_trial(
            tri, results_root, spec.split, res_psid, res_j, i_on, "on"
        )

        cell_off = _build_checkpoint_cell(
            res_ref=res_ref_off,
            trial_ref=tr_off,
            tripreds=[
                ("OFF-trained", res_o, io),
                ("BOTH-trained", res_j, i_j_off),
                ("ON-trained (cross)", pack.res_eo, ieo),
            ],
            channel_idx=ch,
            forecast_target=spec.forecast_target,
            history_ms=spec.history_ms,
            forecast_ms=spec.forecast_ms,
            sampling_hz=spec.sampling_hz,
        )
        cell_on = _build_checkpoint_cell(
            res_ref=res_ref_on,
            trial_ref=tr_on,
            tripreds=[
                ("OFF-trained (cross)", pack.res_en, ien),
                ("BOTH-trained", res_j, i_j_on),
                ("ON-trained", res_n, ion),
            ],
            channel_idx=ch,
            forecast_target=spec.forecast_target,
            history_ms=spec.history_ms,
            forecast_ms=spec.forecast_ms,
            sampling_hz=spec.sampling_hz,
        )

        if cell_off is None and cell_on is None:
            skipped.append(f"{fw} (panels)")
            continue
        built_rows.append((fw, cell_off is not None, cell_on is not None))

        lw = _line_w(fw)

        def _add_cell(
            ci: int,
            cell: Optional[Tuple[np.ndarray, ...]],
        ) -> None:
            if cell is None:
                return
            t_abs, z_true, z_off, z_both, z_on, n_hist = cell
            t_plot = _insert_hist_forecast_gap(np.asarray(t_abs, dtype=float).ravel(), n_hist)
            z_true = _insert_hist_forecast_gap(np.asarray(z_true, dtype=float).ravel(), n_hist)
            z_off = _insert_hist_forecast_gap(np.asarray(z_off, dtype=float).ravel(), n_hist)
            z_both = _insert_hist_forecast_gap(np.asarray(z_both, dtype=float).ravel(), n_hist)
            z_on = _insert_hist_forecast_gap(np.asarray(z_on, dtype=float).ravel(), n_hist)

            t_abs = np.asarray(t_abs, dtype=float).ravel()
            if t_abs.size and n_hist > 0:
                fig.add_vrect(
                    x0=float(t_abs[0]),
                    x1=float(t_abs[n_hist - 1]),
                    fillcolor=_CONTEXT,
                    layer="below",
                    line_width=0,
                    row=ri,
                    col=ci,
                )
            if t_abs.size and n_hist < len(t_abs):
                fig.add_vrect(
                    x0=float(t_abs[n_hist]),
                    x1=float(t_abs[-1]),
                    fillcolor=_FORECAST,
                    layer="below",
                    line_width=0,
                    row=ri,
                    col=ci,
                )
                fig.add_vline(
                    x=float(t_abs[n_hist]),
                    line=dict(color=_RULE, width=0.5, dash="dash"),
                    row=ri,
                    col=ci,
                )

            show_leg = ri == 1 and ci == 1
            fig.add_trace(
                go.Scatter(
                    x=t_plot,
                    y=z_true,
                    mode="lines",
                    name="True (z)",
                    legendgroup="true",
                    line=dict(color=c_true, width=WIDTH_TRUE),
                    showlegend=show_leg,
                    connectgaps=False,
                ),
                row=ri,
                col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_plot,
                    y=z_off,
                    mode="lines",
                    name="Ŷ OFF-trained",
                    legendgroup="off",
                    line=dict(color=COLOR_OFF, width=lw),
                    showlegend=show_leg,
                    connectgaps=False,
                ),
                row=ri,
                col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_plot,
                    y=z_both,
                    mode="lines",
                    name="Ŷ BOTH-trained",
                    legendgroup="both",
                    line=dict(color=COLOR_BOTH, width=lw),
                    showlegend=show_leg,
                    connectgaps=False,
                ),
                row=ri,
                col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_plot,
                    y=z_on,
                    mode="lines",
                    name="Ŷ ON-trained",
                    legendgroup="on",
                    line=dict(color=COLOR_ON, width=lw),
                    showlegend=show_leg,
                    connectgaps=False,
                ),
                row=ri,
                col=ci,
            )

        _add_cell(1, cell_off)
        _add_cell(2, cell_on)

    if not built_rows:
        detail = "; ".join(skipped) if skipped else "no detail"
        raise ValueError(
            "Forecast checkpoint figure: no framework produced a panel. " f"Skipped: {detail}"
        )

    apply_grid_xy_subplots(
        fig, n_rows=3, n_cols=2, theme=spec.theme, nticks=12, x_title=THESIS_TIME_AXIS_TITLE
    )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=int(min(1100, max(520, 320 * 3 + 140))),
        margin=dict(l=72, r=24, t=100, b=128),
        title=dict(
            text=(
                f"{spec.section_title} · trials {i_off} (OFF) / {i_on} (ON) · "
                f"history {spec.history_ms:.0f} ms · horizon {spec.forecast_ms:.0f} ms"
            ),
            font=dict(size=FONT_SIZE_TICK + 1, family=FONT_FAMILY, color=fg),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor=legend_bgcolor(),
            font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
        ),
    )
    fig.update_annotations(font=cross_block_annotation_font(theme=spec.theme))

    _ylab_override = (spec.y_axis_label or "").strip()
    if _ylab_override:
        ylab = _ylab_override
    elif y_mode == "Z":
        och, _ = resolve_output_channel_display(
            res_psid, ch, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS
        )
        ylab = f"z-scored {och}"
    else:
        feat_disp = neural_y_feature_label(
            meta_res,
            ch,
            neural_y_feature_name=spec.neural_y_feature_name,
        )
        ylab = f"z-scored {feat_disp.replace('_', ' ')}"

    fig.update_yaxes(title_text=ylab, row=2, col=1)

    feat = ylab.removeprefix("z-scored ").strip() or ylab
    cap = thesis_exemplar_tagline(
        res_psid,
        i_off,
        i_on,
        feat,
        participant_label=spec.participant_label,
    )
    sc = (spec.caption or "").strip()
    if sc:
        cap = f"{sc} · {cap}"
    if skipped:
        cap += f" · Skipped: {'; '.join(skipped)}."
    return fig, cap
