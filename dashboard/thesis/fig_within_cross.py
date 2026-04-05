"""
Part A: 4-panel trial-level time series (true vs joint/within/cross).
Part B: 2-panel RMSE box plot (within vs cross per model).
Both use Plotly for thesis HTML embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.aggregate_rmse import _key_index_map, _trial_key, normalize_stim
from dashboard.thesis.loaders import (
    load_split_results,
    load_split_results_required,
    trial_rmse_z_for_model,
    _prepare_z_array,
    resolve_output_channel_display,
)
from dashboard.subtabs.helpers import get_channel
from dashboard.thesis.aggregate_rmse import WithinCrossRmseData
from dashboard.thesis.plot_scaffolds import THESIS_TIME_AXIS_TITLE
from dashboard.thesis.constants import (
    COLOR_PSID,
    COLOR_DPAD,
    COLOR_VARMA,
    DOT_SIZE,
    FONT_FAMILY,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    WIDTH_TRUE,
    WIDTH_PSID,
    apply_thesis_style,
    rmse_axis_label,
)
from dashboard.thesis.specs import (
    THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
    ThesisWithinCrossSpec,
    _variant_off,
    _variant_on,
    _variant_cross_eval,
    triplet_branch_timestamp,
)
from dashboard.thesis.transforms import rmse_z, zscore_using_true_stats

COLOR_TRUE = "#2C2C2A"
COLOR_JOINT = "#185FA5"
COLOR_WITHIN = "#993C1D"
COLOR_CROSS = "#854F0B"
_WC_CONTEXT = "rgba(24, 95, 165, 0.08)"
_WC_LATE = "rgba(153, 60, 29, 0.07)"
_WC_RULE = "rgba(68, 68, 65, 0.55)"


def _select_near_median_trials(
    split_res: Dict[str, Any],
    channel_idx: int,
    participant_id: str,
    n_per_condition: int = 2,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    """
    Return (off_trial_indices, on_trial_indices) where each is list of (trial_idx, stim).
    Selects n_per_condition trials nearest to median RMSE in each condition.
    """
    off_with_rmse: List[Tuple[int, float]] = []
    on_with_rmse: List[Tuple[int, float]] = []

    for i in range(len(split_res.get("Z", []))):
        if split_res["Z"][i] is None or split_res["Zp"][i] is None:
            continue
        pid = split_res["participant_id"][i]
        if isinstance(pid, (list, np.ndarray)) and len(pid) > 0:
            pid = str(pid[0])
        else:
            pid = str(pid)
        if pid != participant_id:
            continue
        stim = normalize_stim(split_res["stim"][i])
        if stim is None:
            continue
        try:
            r = trial_rmse_z_for_model(split_res, i, channel_idx)
        except Exception:
            continue
        if stim == "off":
            off_with_rmse.append((i, r))
        else:
            on_with_rmse.append((i, r))

    def _pick_near_median(pairs: List[Tuple[int, float]], n: int) -> List[int]:
        if len(pairs) <= n:
            return [p[0] for p in pairs]
        sorted_by_rmse = sorted(pairs, key=lambda x: x[1])
        med_idx = len(sorted_by_rmse) // 2
        indices = []
        for j in range(n):
            idx = med_idx - (n // 2) + j
            if idx < 0:
                idx = 0
            elif idx >= len(sorted_by_rmse):
                idx = len(sorted_by_rmse) - 1
            indices.append(sorted_by_rmse[idx][0])
        return indices[:n]

    off_idx = _pick_near_median(off_with_rmse, n_per_condition)
    on_idx = _pick_near_median(on_with_rmse, n_per_condition)
    return [(i, "off") for i in off_idx], [(i, "on") for i in on_idx]


def _extract_z_channel(
    split_res: Dict[str, Any], trial_idx: int, channel_idx: int, use_zp: bool = False
) -> Optional[np.ndarray]:
    key = "Zp" if use_zp else "Z"
    arr = split_res.get(key)
    if arr is None or trial_idx >= len(arr) or arr[trial_idx] is None:
        return None
    z_arr, t_abs = _prepare_z_array(arr[trial_idx], split_res, trial_idx)
    return get_channel(z_arr, channel_idx, t_abs)


def _rmse_for_trace(
    split_res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    use_zp: bool,
) -> float:
    z_true = _extract_z_channel(split_res, trial_idx, channel_idx, use_zp=False)
    z_pred = _extract_z_channel(split_res, trial_idx, channel_idx, use_zp=use_zp)
    if z_true is None or z_pred is None or len(z_true) != len(z_pred):
        return float("nan")
    zt = zscore_using_true_stats(z_true, z_true)
    zp = zscore_using_true_stats(z_true, z_pred)
    return float(rmse_z(zt, zp))


def _slice_tail(
    t: np.ndarray, *series: np.ndarray, tail_s: float
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Keep samples in the last ``tail_s`` seconds (absolute trial clock)."""
    t = np.asarray(t, dtype=float).ravel()
    if t.size == 0 or tail_s <= 0:
        return t, [np.asarray(s, dtype=float).ravel() for s in series]
    t_hi = float(np.nanmax(t))
    t_lo = t_hi - float(tail_s)
    m = t >= t_lo
    out_y = [np.asarray(s, dtype=float).ravel()[m] for s in series]
    return t[m], out_y


def _mask_before_split(y: np.ndarray, t: np.ndarray, t_split: float) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel().copy()
    if y.size != t.size:
        return y
    y[t < t_split] = np.nan
    return y


def build_within_cross_timeseries_figure(
    spec: ThesisWithinCrossSpec,
    results_root: Path,
    theme: ThesisTheme = ThesisTheme.LIGHT,
    save_path: Optional[Path] = None,
) -> Tuple[Figure, str]:
    """Part A: 4 panels, true vs joint/within/cross; last ``trial_tail_window_s`` only; cross drawn in late half."""
    tri = spec.joint_triplet
    pid = spec.participant_label

    res_joint = load_split_results_required(results_root, tri.psid_variant, tri.psid_run_ts, spec.split)
    _ch, _ = resolve_output_channel_display(
        res_joint, spec.channel_idx, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS
    )
    y_ylab = f"{_ch} (z-scored)"

    off_trials, on_trials = _select_near_median_trials(
        res_joint, spec.channel_idx, pid, n_per_condition=2
    )
    if len(off_trials) < 2 or len(on_trials) < 2:
        off_trials = off_trials + [(0, "off")] * (2 - len(off_trials))
        on_trials = on_trials + [(0, "on")] * (2 - len(on_trials))

    res_off = load_split_results_required(
        results_root,
        _variant_off(tri.psid_variant),
        triplet_branch_timestamp(tri, "psid", "off"),
        spec.split,
    )
    res_on = load_split_results_required(
        results_root,
        _variant_on(tri.psid_variant),
        triplet_branch_timestamp(tri, "psid", "on"),
        spec.split,
    )
    res_cross_off = load_split_results_required(
        results_root,
        _variant_cross_eval(_variant_on(tri.psid_variant), "off"),
        triplet_branch_timestamp(tri, "psid", "eval_off"),
        spec.split,
    )
    res_cross_on = load_split_results_required(
        results_root,
        _variant_cross_eval(_variant_off(tri.psid_variant), "on"),
        triplet_branch_timestamp(tri, "psid", "eval_on"),
        spec.split,
    )

    map_off = _key_index_map(res_off)
    map_on = _key_index_map(res_on)
    map_cross_off = _key_index_map(res_cross_off)
    map_cross_on = _key_index_map(res_cross_on)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=["DBS-OFF", "DBS-ON"],
        horizontal_spacing=0.07,
    )
    caption_rmse_lines: List[str] = []

    def _get_z(res: Dict, idx: int, zp: bool) -> Optional[np.ndarray]:
        return _extract_z_channel(res, idx, spec.channel_idx, use_zp=zp)

    def _get_t(res: Dict, idx: int) -> np.ndarray:
        z = res["Z"][idx]
        arr, t = _prepare_z_array(z, res, idx)
        return t

    tail_s = float(spec.trial_tail_window_s)
    frac = float(np.clip(spec.cross_visible_tail_fraction, 0.05, 0.95))

    # col 1 = DBS-OFF trials, col 2 = DBS-ON trials
    cond_groups = [
        ("off", off_trials[:2], 1),
        ("on",  on_trials[:2],  2),
    ]

    # Add background shading once per subplot (normalized time 0→tail_s)
    t_split_rel = (1.0 - frac) * tail_s
    for _col in (1, 2):
        fig.add_vrect(x0=0, x1=t_split_rel, fillcolor=_WC_CONTEXT, layer="below", line_width=0, row=1, col=_col)
        fig.add_vrect(x0=t_split_rel, x1=tail_s, fillcolor=_WC_LATE, layer="below", line_width=0, row=1, col=_col)
        fig.add_vline(x=t_split_rel, line=dict(color=_WC_RULE, width=0.8, dash="dash"), row=1, col=_col)

    for cond_stim, trial_list, col in cond_groups:
        for trial_num, (trial_idx, stim) in enumerate(trial_list):
            alpha = 1.0 if trial_num == 0 else 0.55
            k = _trial_key(res_joint, trial_idx)

            z_true = _get_z(res_joint, trial_idx, False)
            z_joint = _get_z(res_joint, trial_idx, True)
            t_raw = _get_t(res_joint, trial_idx)

            if z_true is None or z_joint is None:
                continue

            idx_off = map_off.get(k)
            idx_on = map_on.get(k)
            idx_cross_off = map_cross_off.get(k)
            idx_cross_on = map_cross_on.get(k)

            z_within = None
            z_cross = None
            if stim == "off" and idx_off is not None and res_off:
                z_within = _get_z(res_off, idx_off, True)
            if stim == "on" and idx_on is not None and res_on:
                z_within = _get_z(res_on, idx_on, True)
            if stim == "off" and idx_cross_off is not None and res_cross_off:
                z_cross = _get_z(res_cross_off, idx_cross_off, True)
            if stim == "on" and idx_cross_on is not None and res_cross_on:
                z_cross = _get_z(res_cross_on, idx_cross_on, True)

            t_full = np.asarray(t_raw, dtype=float).ravel()
            zt_full = np.asarray(z_true, dtype=float).ravel()
            zj_full = np.asarray(z_joint, dtype=float).ravel()
            zu_full = (
                np.asarray(z_within, dtype=float).ravel()
                if z_within is not None and len(z_within) == len(t_full)
                else None
            )
            zc_full = (
                np.asarray(z_cross, dtype=float).ravel()
                if z_cross is not None and len(z_cross) == len(t_full)
                else None
            )

            series_in = [zt_full, zj_full]
            if zu_full is not None:
                series_in.append(zu_full)
            if zc_full is not None:
                series_in.append(zc_full)
            t_sliced, ys_out = _slice_tail(t_full, *series_in, tail_s=tail_s)
            zt, zj = ys_out[0], ys_out[1]
            _ki = 2
            zu = ys_out[_ki] if zu_full is not None else None
            _ki += 1 if zu_full is not None else 0
            zc = ys_out[_ki] if zc_full is not None else None

            if t_sliced.size == 0:
                continue

            # Normalize to relative time starting at 0
            t_rel = t_sliced - float(np.nanmin(t_sliced))
            t_abs_split = (1.0 - frac) * (float(np.nanmax(t_rel)) or tail_s)
            if zc is not None:
                zc = _mask_before_split(zc, t_rel, t_abs_split)

            show_leg = (trial_num == 0 and col == 1)

            fig.add_trace(
                go.Scatter(
                    x=t_rel, y=zt, name="true", mode="lines",
                    line=dict(color=COLOR_TRUE, width=WIDTH_TRUE),
                    opacity=alpha, legendgroup="true",
                    showlegend=show_leg, connectgaps=False,
                ),
                row=1, col=col,
            )
            if zu is not None:
                fig.add_trace(
                    go.Scatter(
                        x=t_rel, y=zu, name="within", mode="lines",
                        line=dict(color=COLOR_WITHIN, width=1.35, dash="dash"),
                        opacity=alpha, legendgroup="within",
                        showlegend=show_leg, connectgaps=False,
                    ),
                    row=1, col=col,
                )
            if zc is not None:
                fig.add_trace(
                    go.Scatter(
                        x=t_rel, y=zc, name="cross", mode="lines",
                        line=dict(color=COLOR_CROSS, width=1.45, dash="dot"),
                        opacity=alpha, legendgroup="cross",
                        showlegend=show_leg, connectgaps=False,
                    ),
                    row=1, col=col,
                )
            fig.add_trace(
                go.Scatter(
                    x=t_rel, y=zj, name="both", mode="lines",
                    line=dict(color=COLOR_JOINT, width=WIDTH_PSID),
                    opacity=alpha, legendgroup="both",
                    showlegend=show_leg, connectgaps=False,
                ),
                row=1, col=col,
            )

            rmse_both = _rmse_for_trace(res_joint, trial_idx, spec.channel_idx, use_zp=True)
            rmse_within = float("nan")
            rmse_cross = float("nan")
            if stim == "off" and idx_off is not None and res_off:
                rmse_within = _rmse_for_trace(res_off, idx_off, spec.channel_idx, use_zp=True)
            if stim == "on" and idx_on is not None and res_on:
                rmse_within = _rmse_for_trace(res_on, idx_on, spec.channel_idx, use_zp=True)
            if stim == "off" and idx_cross_off is not None and res_cross_off:
                rmse_cross = _rmse_for_trace(res_cross_off, idx_cross_off, spec.channel_idx, use_zp=True)
            if stim == "on" and idx_cross_on is not None and res_cross_on:
                rmse_cross = _rmse_for_trace(res_cross_on, idx_cross_on, spec.channel_idx, use_zp=True)

            cond_lbl = "OFF" if stim == "off" else "ON"
            caption_rmse_lines.append(
                f"{cond_lbl} trial {trial_num + 1} (row {trial_idx}): "
                f"both {rmse_both:.3f}, within {rmse_within:.3f}, cross {rmse_cross:.3f}."
            )

    for col in (1, 2):
        fig.update_xaxes(
            title_text=THESIS_TIME_AXIS_TITLE,
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            row=1, col=col,
        )
        fig.update_yaxes(
            title_text=y_ylab if col == 1 else "",
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            row=1, col=col,
        )

    apply_thesis_style(
        fig, theme, height=340,
        margin=dict(l=60, r=20, t=50, b=72),
        legend_y=-0.18,
    )

    cap = (
        f"Left: DBS-OFF trials; right: DBS-ON. Each panel shows the last {tail_s:.1f} s of each trial "
        f"(relative time, 0 = trial end minus {tail_s:.1f} s). "
        f"Blue-tinted: early segment; coral: late segment (last {frac*100:.0f}% of window). "
        f"Cross-condition trace (dotted) starts at the vertical rule. "
        f"Traces: true (black), dbs_both (blue), same-condition within (dashed brown), cross-eval (dotted brown). "
        + " ".join(caption_rmse_lines)
    )
    if spec.caption:
        cap = f"{spec.caption} {cap}"

    if save_path:
        fig.write_image(str(save_path))
    return fig, cap


# ---------------------------------------------------------------------------
# Part B: RMSE box plot (within vs cross per model)
# ---------------------------------------------------------------------------

BW = 0.22
GAP = 0.06
GROUP_GAP = 0.45


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_within_cross_boxplot_figure(
    data: WithinCrossRmseData,
    theme: ThesisTheme = ThesisTheme.LIGHT,
    save_path: Optional[Path] = None,
    jitter_seed: int = 42,
) -> Figure:
    """Part B: 2 panels (DBS-OFF | DBS-ON), 3 model groups, within + cross bars each. Returns Plotly Figure."""
    models = ["PSID", "DPAD", "VARMA"]
    model_colors = [COLOR_PSID, COLOR_DPAD, COLOR_VARMA]
    off_cells = [
        (data.psid_off_within, data.psid_off_cross),
        (data.dpad_off_within, data.dpad_off_cross),
        (data.varma_off_within, data.varma_off_cross),
    ]
    on_cells = [
        (data.psid_on_within, data.psid_on_cross),
        (data.dpad_on_within, data.dpad_on_cross),
        (data.varma_on_within, data.varma_on_cross),
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["DBS-OFF", "DBS-ON"],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    rng = np.random.default_rng(jitter_seed)
    y_max = 0.0

    for col_idx, cells in enumerate([off_cells, on_cells]):
        x_pos = 0.0
        tick_positions = []
        for mi, (within_vals, cross_vals) in enumerate(cells):
            x_within = x_pos
            x_cross = x_pos + BW + GAP

            for vals, x, dashed, alpha in [
                (within_vals, x_within, False, 0.55),
                (cross_vals, x_cross, True, 0.35),
            ]:
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) == 0:
                    continue
                y_max = max(y_max, float(np.nanmax(arr)))
                q1, med, q3 = np.percentile(arr, [25, 50, 75])
                iqr = q3 - q1
                w_lo = max(float(np.nanmin(arr)), q1 - 1.5 * iqr) if iqr > 0 else float(np.nanmin(arr))
                w_hi = min(float(np.nanmax(arr)), q3 + 1.5 * iqr) if iqr > 0 else float(np.nanmax(arr))

                col = model_colors[mi]
                fillcolor = _hex_to_rgba(col, 0.30) if not dashed else "rgba(0,0,0,0)"

                fig.add_trace(
                    go.Box(
                        y=arr, x=[x] * len(arr),
                        marker_color=col, line=dict(color=col, width=1.2),
                        fillcolor=fillcolor, boxpoints=False, quartilemethod="exclusive",
                        name=f"{models[mi]} {'cross' if dashed else 'within'}",
                        showlegend=(col_idx == 0),
                        legendgroup=f"{models[mi]}_{'cross' if dashed else 'within'}",
                    ),
                    row=1, col=col_idx + 1,
                )
                jitter = rng.uniform(-0.08, 0.08, size=len(arr))
                fig.add_trace(
                    go.Scatter(
                        x=np.array([x] * len(arr)) + jitter, y=arr,
                        mode="markers", marker=dict(size=DOT_SIZE, color=col, opacity=alpha, line=dict(width=0)),
                        showlegend=False,
                    ),
                    row=1, col=col_idx + 1,
                )

            tick_positions.append(x_within + (BW + GAP) / 2)
            x_pos += 2 * BW + GAP + GROUP_GAP

        fig.update_xaxes(
            tickvals=tick_positions, ticktext=models,
            row=1, col=col_idx + 1,
        )

    y_max = max(y_max * 1.08, 0.85)
    apply_thesis_style(
        fig, theme, height=400,
        margin=dict(l=60, r=20, t=50, b=100),
        legend_y=-0.2,
    )
    # Figure-specific overrides: y-axis ranges and x-axis grid off
    fig.update_layout(
        yaxis=dict(range=[0, y_max]),
        yaxis2=dict(range=[0, y_max]),
    )
    for col in (1, 2):
        fig.update_xaxes(showgrid=False, row=1, col=col)
    fig.update_yaxes(
        title_text=rmse_axis_label(),
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        row=1, col=1,
    )

    if save_path:
        fig.write_image(str(save_path))
    return fig


def plot_within_cross_figures(save: bool = True) -> None:
    """Generate Part A and Part B figures; save to dashboard/thesis/figures/."""
    import logging
    from dashboard.thesis.plot_config import RESULTS_ROOT
    from dashboard.thesis.specs import THESIS_WITHIN_CROSS

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    log = logging.getLogger(__name__)

    try:
        fig_a, _ = build_within_cross_timeseries_figure(
            THESIS_WITHIN_CROSS, RESULTS_ROOT,
        )
        if save:
            for ext in ("png", "pdf"):
                try:
                    fig_a.write_image(str(out_dir / f"within_cross_timeseries.{ext}"))
                except Exception:
                    pass
    except Exception as e:
        log.warning("Could not build within-cross timeseries: %s", e)

    try:
        from dashboard.thesis.aggregate_rmse import collect_within_cross_rmse
        data = collect_within_cross_rmse(
            RESULTS_ROOT,
            [THESIS_WITHIN_CROSS.joint_triplet],
            THESIS_WITHIN_CROSS.channel_idx,
            split=THESIS_WITHIN_CROSS.split,
        )
        fig_b = build_within_cross_boxplot_figure(data)
        if save:
            for ext in ("png", "pdf"):
                try:
                    fig_b.write_image(str(out_dir / f"within_cross_boxplot.{ext}"))
                except Exception:
                    pass
    except Exception as e:
        log.warning("Could not build within-cross boxplot: %s", e)
