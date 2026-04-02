"""
Thesis appendix figures: preprocessing schematic, PSD comparison, tracing speed,
PSID grid search heatmaps (Pearson r, RMSE, lag), plus extras.
Provides Plotly build_* functions for HTML embedding and matplotlib plot_* for file export.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from scipy.signal import welch

from dashboard.thesis.constants import (
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.loaders import channels_as_str_list, load_split_results
from dashboard.thesis.plot_config import (
    COLORS,
    GRID_SEARCH_PARQUET,
    N1_VALS,
    NX_VALS,
    RESULTS_ROOT,
    SELECTED_CONFIG,
    apply_style,
    get_triplets,
    load_results_for_triplet,
)

FS = 80
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

C_RAW = "#D3D1C7"
C_PROC = "#9FE1CB"
C_FEAT = "#AFA9EC"
C_MERGE = "#85B7EB"
BAND_SHADE_COLOR = "#E24B4A"


# ─────────────────────────────────────────────────────────────────────────────
# Plotly build_* functions (for HTML embedding)
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline_figure() -> Figure:
    """Plotly flowchart for HTML embedding."""
    fig = go.Figure()
    shapes, ann = [], []

    def _rect(x, y, w, h, c):
        shapes.append(dict(type="rect", x0=x - w / 2, y0=y - h / 2, x1=x + w / 2, y1=y + h / 2,
                          line=dict(color="#888780", width=0.6), fillcolor=c))

    def _arr(x1, y1, x2, y2):
        ann.append(dict(x=x2, y=y2, ax=x1 - x2, ay=y1 - y2, axref="x", ayref="y", xref="x", yref="y",
                        text="", showarrow=True, arrowhead=2, arrowside="end", arrowcolor="#444441", arrowwidth=1.1))

    _sub = dict(size=10, color="#5F5E5A")
    _main = dict(size=12)

    _rect(2.5, 9.2, 2.8, 0.7, C_RAW)
    ann.extend([dict(x=2.5, y=9.2, text="Raw iEEG", showarrow=False, font=_main), dict(x=2.5, y=8.85, text="22 kHz · LFP 16ch + ECoG 4ch", showarrow=False, font=_sub)])
    _arr(2.5, 8.85, 2.5, 8.1)
    _rect(2.5, 7.7, 2.8, 0.7, C_PROC)
    ann.extend([dict(x=2.5, y=7.7, text="Resample + bandpass", showarrow=False, font=_main), dict(x=2.5, y=7.35, text="↓ 60 Hz · 3–28 Hz BP (MNE)", showarrow=False, font=_sub)])
    _arr(2.5, 7.35, 2.5, 6.6)
    _rect(2.5, 6.2, 2.8, 0.7, C_PROC)
    ann.extend([dict(x=2.5, y=6.2, text="Common avg. re-reference", showarrow=False, font=_main), dict(x=2.5, y=5.85, text="Per modality, per trial", showarrow=False, font=_sub)])
    _arr(2.5, 5.85, 2.5, 5.1)
    _rect(2.5, 4.7, 2.8, 0.7, C_PROC)
    ann.extend([dict(x=2.5, y=4.7, text="Scale ×10⁶", showarrow=False, font=_main), dict(x=2.5, y=4.35, text="→ microvolts", showarrow=False, font=_sub)])
    _arr(2.5, 4.35, 2.5, 3.6)
    _rect(2.5, 3.2, 2.8, 0.9, C_FEAT)
    ann.extend([dict(x=2.5, y=3.55, text="Narrowband features", showarrow=False, font=_main), dict(x=2.5, y=2.9, text="δ/θ/α raw · β 13–29 Hz Hilbert env", showarrow=False, font=_sub)])
    _rect(7.5, 9.2, 2.8, 0.7, C_RAW)
    ann.extend([dict(x=7.5, y=9.2, text="Tablet coordinates", showarrow=False, font=_main), dict(x=7.5, y=8.85, text="x(t), y(t) at variable rate", showarrow=False, font=_sub)])
    _arr(7.5, 8.85, 7.5, 8.1)
    _rect(7.5, 7.7, 2.8, 0.7, C_PROC)
    ann.extend([dict(x=7.5, y=7.7, text="Kinematic derivation", showarrow=False, font=_main), dict(x=7.5, y=7.35, text="v, a, j in x/y/xy/mag", showarrow=False, font=_sub)])
    _arr(7.5, 7.35, 7.5, 6.6)
    _rect(7.5, 6.2, 2.8, 0.7, C_PROC)
    ann.extend([dict(x=7.5, y=6.2, text="Savitzky-Golay smooth", showarrow=False, font=_main), dict(x=7.5, y=5.85, text="~200 ms window · 3rd order", showarrow=False, font=_sub)])
    _arr(7.5, 5.85, 7.5, 5.1)
    _rect(7.5, 4.7, 2.8, 0.7, C_PROC)
    ann.extend([dict(x=7.5, y=4.7, text="Interpolate to 60 Hz", showarrow=False, font=_main), dict(x=7.5, y=4.35, text="Linear onto neural grid", showarrow=False, font=_sub)])
    _arr(7.5, 4.35, 7.5, 3.6)
    _rect(7.5, 3.2, 2.8, 0.7, C_FEAT)
    ann.extend([dict(x=7.5, y=3.2, text="Tracing speed", showarrow=False, font=_main), dict(x=7.5, y=2.85, text="velocity magnitude (Z)", showarrow=False, font=_sub)])
    shapes.append(dict(type="rect", x0=1.0, y0=2.35, x1=9.0, y1=2.9, fillcolor="#FAEEDA", line=dict(color="#854F0B", width=0.7)))
    ann.append(dict(x=5.0, y=2.625, text="Trial segmentation: 9 s trial · ±2 s margin buffer", showarrow=False, font=dict(size=11, color="#412402")))
    _arr(2.5, 2.75, 2.5, 2.35)
    _arr(7.5, 2.75, 7.5, 2.35)
    _arr(2.5, 2.35, 4.5, 1.75)
    _arr(7.5, 2.35, 5.5, 1.75)
    _rect(5.0, 1.45, 3.2, 0.7, C_MERGE)
    ann.extend([dict(x=5.0, y=1.45, text="Model input (Y, Z) at 60 Hz", showarrow=False, font=_main), dict(x=5.0, y=1.1, text="Train 60% · Val 20% · Test 20%", showarrow=False, font=_sub)])
    for i, (c, lbl) in enumerate([(C_RAW, "Raw input"), (C_PROC, "Processing step"), (C_FEAT, "Feature / output"), (C_MERGE, "Model input")]):
        px = 0.55 + i * 2.3
        shapes.append(dict(type="rect", x0=px, y0=0.15, x1=px + 0.43, y1=0.43, fillcolor=c, line=dict(color="#888780", width=0.5)))
        ann.append(dict(x=px + 0.53, y=0.29, text=lbl, showarrow=False, font=dict(size=10, color="#444441")))

    fig.add_trace(go.Scatter(x=[0, 10], y=[0, 10], mode="markers", marker=dict(size=0, opacity=0), showlegend=False))
    fig.update_layout(title="Data preprocessing pipeline", xaxis=dict(visible=False, range=[0, 10]), yaxis=dict(visible=False, range=[0, 10]),
                      shapes=shapes, annotations=ann, margin=dict(l=40, r=40, t=60, b=50), font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE),
                      paper_bgcolor="white", plot_bgcolor="white", height=620)
    return fig


def build_psd_dbs_comparison_figure(band_shade: Optional[Tuple[float, float]] = (13, 29)) -> Figure:
    triplets = get_triplets()
    n_panels, ncols = len(triplets), 2
    nrows = (n_panels + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
        subplot_titles=[t.label or t.psid_variant for t in triplets],
    )
    OFF_C, ON_C = COLORS["dbs_off"], COLORS["dbs_on"]

    for pi, tri in enumerate(triplets):
        r, c = divmod(pi, ncols)
        trials = load_results_for_triplet(tri)
        if not trials:
            fig.add_trace(go.Scatter(x=[15], y=[-20], text=["no data"], mode="text", textfont=dict(color="gray"), showlegend=False), row=r + 1, col=c + 1)
            continue
        psds_off, psds_on, freqs_ref = [], [], None
        for row in trials:
            y = row.get("Y")
            stim = row.get("stim", "off")
            if y is None:
                continue
            y_arr = np.asarray(y)
            ch = y_arr.mean(axis=1) if y_arr.ndim == 2 else np.asarray(y).ravel()
            freqs, psd = _compute_trial_psd(ch, FS)
            if freqs_ref is None:
                freqs_ref = freqs
            (psds_off if stim == "off" else psds_on).append(psd)
        if freqs_ref is None:
            continue
        for cond, col_c, label, show in [(psds_off, OFF_C, "DBS-OFF", pi == 0), (psds_on, ON_C, "DBS-ON", pi == 0)]:
            if not cond:
                continue
            mat = np.vstack(cond)
            mean, sem = mat.mean(axis=0), mat.std(axis=0) / np.sqrt(len(mat))
            xb = np.concatenate([freqs_ref, freqs_ref[::-1]])
            yb = np.concatenate([mean + sem, (mean - sem)[::-1]])
            fc = f"rgba({int(col_c[1:3],16)},{int(col_c[3:5],16)},{int(col_c[5:7],16)},0.15)"
            fig.add_trace(go.Scatter(x=xb, y=yb, fill="toself", fillcolor=fc, line=dict(width=0), showlegend=show, name=label), row=r + 1, col=c + 1)
            fig.add_trace(go.Scatter(x=freqs_ref, y=mean, mode="lines", line=dict(color=col_c, width=1.8), showlegend=show, name=label), row=r + 1, col=c + 1)
        if band_shade is not None:
            fig.add_vrect(x0=band_shade[0], x1=band_shade[1], line=dict(width=0), fillcolor=BAND_SHADE_COLOR, opacity=0.05, row=r + 1, col=c + 1)
            for xv in band_shade:
                fig.add_vline(x=xv, line_dash="dash", line_color=BAND_SHADE_COLOR, line_width=0.8, opacity=0.6, row=r + 1, col=c + 1)

    _psd_theme = ThesisTheme.LIGHT
    _psd_bg, _psd_pbg = paper_colors(_psd_theme)
    _psd_grid = grid_color(_psd_theme)
    _psd_fg = true_line_color(_psd_theme)

    fig.update_layout(
        title="ECoG PSD: DBS-ON vs DBS-OFF",
        legend=dict(
            orientation="h", y=-0.08, x=0.5, xanchor="center",
            font=dict(size=FONT_SIZE_TICK),
        ),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=_psd_fg),
        margin=dict(b=88, t=56),
        height=360 * nrows,
        paper_bgcolor=_psd_bg,
        plot_bgcolor=_psd_pbg,
    )
    fig.update_annotations(font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY))
    for i in range(n_panels):
        ri, ci = divmod(i, ncols)
        fig.update_xaxes(
            title_text="Frequency (Hz)" if ri >= nrows - 1 else "",
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            title_standoff=10,
            showline=True, linecolor=_psd_fg, linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
            showgrid=True, gridcolor=_psd_grid,
            row=ri + 1,
            col=ci + 1,
        )
        fig.update_yaxes(
            title_text="PSD (dB/Hz)" if ci == 0 else "",
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            title_standoff=8,
            showline=True, linecolor=_psd_fg, linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
            showgrid=True, gridcolor=_psd_grid,
            row=ri + 1,
            col=ci + 1,
        )
    return fig


def _z_column_indices_for_outputs(triplet) -> tuple[int, int]:
    """Match saved ``output_channels`` names (YAML ``output:`` order)."""
    res = load_split_results(
        RESULTS_ROOT, triplet.psid_variant, triplet.psid_run_ts, "test"
    )
    names = channels_as_str_list(res.get("output_channels")) if res else []

    def _ix(exact: str) -> int | None:
        for i, n in enumerate(names):
            if str(n) == exact:
                return i
        return None

    iv = _ix("tracing_velocity_x")
    ia = _ix("tracing_acceleration_magnitude")
    return (iv if iv is not None else 0, ia if ia is not None else 1)


def _trial_z_column(z_arr: Any, col_idx: int) -> np.ndarray:
    if z_arr is None:
        return np.full(FS * 9, np.nan)
    za = np.asarray(z_arr, dtype=float)
    if za.ndim == 2:
        if col_idx >= za.shape[1]:
            return np.full(FS * 9, np.nan)
        za = za[:, col_idx]
    else:
        za = za.ravel()
    if len(za) < FS * 9:
        return np.pad(za, (0, FS * 9 - len(za)), constant_values=np.nan)
    return za[: FS * 9]


def _zscore_traces(z_off: list, z_on: list) -> tuple[list, list]:
    """Z-score each DBS condition using only that condition’s pooled samples (per output feature)."""

    def _norm(seq: list) -> list:
        if not seq:
            return seq
        pool = np.concatenate([np.asarray(a, dtype=float).ravel() for a in seq])
        pool = pool[np.isfinite(pool)]
        if not pool.size:
            return seq
        mu, sig = float(np.mean(pool)), float(np.std(pool))
        if sig < 1e-9:
            sig = 1.0
        return [(np.asarray(a, dtype=float) - mu) / sig for a in seq]

    return _norm(z_off), _norm(z_on)


def build_tracing_speed_dbs_comparison_figure() -> Figure:
    triplets = get_triplets()
    n_panels, ncols = len(triplets), 2
    nrows_pair = (n_panels + ncols - 1) // ncols
    nrows_plot = 2 * nrows_pair
    t = np.arange(FS * 9) / FS
    subplot_titles: list[str] = []
    for rp in range(nrows_pair):
        for c in range(ncols):
            pi = rp * ncols + c
            lab = (triplets[pi].label or "") if pi < n_panels else ""
            subplot_titles.append(
                f"{lab} — velocity" if lab else ""
            )
        for c in range(ncols):
            pi = rp * ncols + c
            lab = (triplets[pi].label or "") if pi < n_panels else ""
            subplot_titles.append(
                f"{lab} — acceleration" if lab else ""
            )
    fig = make_subplots(
        rows=nrows_plot,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
        subplot_titles=subplot_titles,
    )
    OFF_C, ON_C = COLORS["dbs_off"], COLORS["dbs_on"]

    for pi, tri in enumerate(triplets):
        r_pair, c = divmod(pi, ncols)
        r_vel = 2 * r_pair + 1
        r_acc = r_vel + 1
        trials = load_results_for_triplet(tri)
        if not trials:
            continue
        iv, ia = _z_column_indices_for_outputs(tri)

        def _collect(col_idx: int) -> tuple[list, list]:
            off_l, on_l = [], []
            for row in trials:
                z = row.get("Z")
                if z is None:
                    continue
                za = _trial_z_column(z, col_idx)
                stim = row.get("stim", "off")
                (off_l if stim == "off" else on_l).append(za)
            return off_l, on_l

        for row_plot, col_idx, y_title, metric_tag in (
            (r_vel, iv, "Velocity (z)", "vel"),
            (r_acc, ia, "Acceleration (z)", "acc"),
        ):
            z_off, z_on = _collect(col_idx)
            z_off, z_on = _zscore_traces(z_off, z_on)
            for cond, col_c, label in [(z_off, OFF_C, "DBS-OFF"), (z_on, ON_C, "DBS-ON")]:
                if not cond:
                    continue
                mat = np.vstack(cond)
                mean = np.nanmean(mat, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=mean,
                        mode="lines",
                        line=dict(color=col_c, width=2.0),
                        showlegend=(pi == 0),
                        name=f"{label} · {metric_tag}",
                        legendgroup=f"{label}_{metric_tag}",
                    ),
                    row=row_plot,
                    col=c + 1,
                )
            if c == 0:
                fig.update_yaxes(title_text=y_title, row=row_plot, col=c + 1)

    theme = ThesisTheme.LIGHT
    paper_bg, plot_bg = paper_colors(theme)
    _grid = grid_color(theme)
    _fg = true_line_color(theme)

    fig.update_layout(
        title="Tracing speed: DBS-ON vs DBS-OFF (trial means, z-scored per metric × DBS)",
        legend=dict(
            orientation="h", y=-0.06, x=0.5, xanchor="center",
            font=dict(size=FONT_SIZE_TICK),
        ),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=_fg),
        margin=dict(b=120, t=64, l=88, r=40),
        height=max(260 * nrows_plot + 100, 480),
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
    )
    fig.update_annotations(font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY))
    fig.update_xaxes(range=[0, 9])
    for rp in range(nrows_pair):
        for c in range(ncols):
            for ri_off in (2 * rp + 1, 2 * rp + 2):
                fig.update_xaxes(
                    showline=True, linecolor=_fg, linewidth=1,
                    tickfont=dict(size=FONT_SIZE_TICK),
                    title_text="time (s)" if (rp >= nrows_pair - 1 and ri_off == 2 * rp + 2) else "",
                    title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                    row=ri_off,
                    col=c + 1,
                )
                fig.update_yaxes(
                    showgrid=True, gridcolor=_grid,
                    showline=True, linecolor=_fg, linewidth=1,
                    tickfont=dict(size=FONT_SIZE_TICK),
                    title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                    row=ri_off,
                    col=c + 1,
                )
    return fig


def build_grid_search_figure(metric_col: str, cmap: str, vmin: float, vmax: float, inverse: bool, title: str) -> Figure:
    df = _load_grid_df()
    triplets = get_triplets()
    n_panels, ncols = len(triplets), 2
    nrows = (n_panels + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, vertical_spacing=0.2, horizontal_spacing=0.08,
                        subplot_titles=[t.label or t.psid_variant for t in triplets])
    for pi, tri in enumerate(triplets):
        r, c = divmod(pi, ncols)
        key = tri.label or tri.psid_variant or ""
        _m = re.search(r'(PDI[14])_(\d+)', key)
        if _m:
            pid, sess = _m.group(1), _m.group(2)
        else:
            pid = "PDI1" if "PDI1" in (tri.psid_variant or "") else "PDI4"
            sess = "2"
        mat = _build_heatmap_matrix(df, pid, sess, metric_col) if df is not None and metric_col in df.columns else np.full((len(N1_VALS), len(NX_VALS)), np.nan)
        z_plot = -mat if inverse else mat
        zmin, zmax = (-vmax, -vmin) if inverse else (vmin, vmax)
        text_mat = np.empty(mat.shape, dtype=object)
        # Determine text precision: if all valid values differ by < 1, show more decimals
        valid_vals = mat[~np.isnan(mat)]
        val_range = float(np.ptp(valid_vals)) if len(valid_vals) > 1 else 0.0
        for n1i in range(len(N1_VALS)):
            for xi in range(len(NX_VALS)):
                if N1_VALS[n1i] > NX_VALS[xi]:
                    z_plot[n1i, xi] = np.nan
                    text_mat[n1i, xi] = ""
                elif not np.isnan(mat[n1i, xi]):
                    v = mat[n1i, xi]
                    if val_range < 1.0 and abs(v) >= 100:
                        text_mat[n1i, xi] = f"{v:.2f}"
                    elif abs(v) < 100:
                        text_mat[n1i, xi] = f"{v:.2f}"
                    else:
                        text_mat[n1i, xi] = f"{v:.0f}"
                else:
                    text_mat[n1i, xi] = ""
        # Use per-panel z-range when global bounds don't span the data
        panel_valid = z_plot[~np.isnan(z_plot)]
        if len(panel_valid) > 0:
            p_min, p_max = float(np.min(panel_valid)), float(np.max(panel_valid))
            if p_min < zmin or p_max > zmax:
                zmin, zmax = p_min - 0.01 * abs(p_min), p_max + 0.01 * abs(p_max)
        # Integer cell indices so the optional best-cell marker aligns with heatmap cells.
        x_idx = np.arange(len(NX_VALS), dtype=float)
        y_idx = np.arange(len(N1_VALS), dtype=float)
        fig.add_trace(
            go.Heatmap(
                z=z_plot,
                x=x_idx,
                y=y_idx,
                text=text_mat,
                texttemplate="%{text}",
                colorscale=cmap,
                zmin=zmin,
                zmax=zmax,
                showscale=(pi == 0),
            ),
            row=r + 1,
            col=c + 1,
        )
        fig.update_xaxes(
            title_text="nx (neural input dim.)",
            tickmode="array",
            tickvals=x_idx,
            ticktext=[str(v) for v in NX_VALS],
            row=r + 1,
            col=c + 1,
        )
        fig.update_yaxes(
            title_text="n₁ (latent dim.)" if c == 0 else "",
            tickmode="array",
            tickvals=y_idx,
            ticktext=[str(v) for v in N1_VALS],
            row=r + 1,
            col=c + 1,
        )
        sel_key = f"{pid}_S{sess}"
        picked = SELECTED_CONFIG.get(sel_key)
        if picked is not None:
            nx_pick, n1_pick = picked
            try:
                bx = NX_VALS.index(int(nx_pick))
                by = N1_VALS.index(int(n1_pick))
            except (ValueError, TypeError):
                bx = by = -1
            if bx >= 0 and by >= 0:
                fig.add_shape(
                    type="rect",
                    x0=float(bx) - 0.5,
                    x1=float(bx) + 0.5,
                    y0=float(by) - 0.5,
                    y1=float(by) + 0.5,
                    line=dict(color="#C0392B", width=2.5),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                    row=r + 1,
                    col=c + 1,
                )
    fig.update_layout(
        title=title,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=true_line_color(ThesisTheme.LIGHT)),
        height=350 * nrows,
        paper_bgcolor=paper_colors(ThesisTheme.LIGHT)[0],
        plot_bgcolor=paper_colors(ThesisTheme.LIGHT)[1],
    )
    fig.update_annotations(font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY))
    return fig


def build_grid_search_pearson_figure() -> Figure:
    return build_grid_search_figure("pearson_fisher", "Blues", 0.25, 0.85, False, "Grid search: validation Pearson r")


def build_grid_search_rmse_figure() -> Figure:
    return build_grid_search_figure("rmse_Z", "Reds_r", 0.3, 1.2, False, "Grid search: validation RMSE(z) — behavioral output")


def build_grid_search_neural_rmse_figure() -> Figure:
    return build_grid_search_figure("rmse_Y", "Reds_r", 0.3, 1.2, False, "Grid search: validation RMSE — neural reconstruction")


def build_grid_search_lag_figure() -> Figure:
    return build_grid_search_figure("xcorr_lag_mean_ms", "viridis", 0, 100, False, "Grid search: validation lag (ms)")


def build_trial_count_summary_figure() -> Figure:
    triplets = get_triplets()
    rows = [{"session": t.label or "", "DBS-OFF": sum(1 for r in load_results_for_triplet(t) if r.get("stim") == "off"),
             "DBS-ON": sum(1 for r in load_results_for_triplet(t) if r.get("stim") == "on")} for t in triplets]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r["session"] for r in rows], y=[r["DBS-OFF"] for r in rows], name="DBS-OFF", marker_color=COLORS["dbs_off"], width=0.35))
    fig.add_trace(go.Bar(x=[r["session"] for r in rows], y=[r["DBS-ON"] for r in rows], name="DBS-ON", marker_color=COLORS["dbs_on"], width=0.35))
    _tc_fg = true_line_color(ThesisTheme.LIGHT)
    _tc_grid = grid_color(ThesisTheme.LIGHT)
    fig.update_layout(
        barmode="group",
        title="Trial count per session × DBS condition",
        xaxis=dict(
            title_text="",
            showline=True, linecolor=_tc_fg, linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
            showgrid=False,
        ),
        yaxis=dict(
            title_text="Trial count",
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            showgrid=True, gridcolor=_tc_grid,
            showline=True, linecolor=_tc_fg, linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
        ),
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=_tc_fg),
        legend=dict(orientation="h", font=dict(size=FONT_SIZE_TICK)),
        height=320,
        paper_bgcolor=paper_colors(ThesisTheme.LIGHT)[0],
        plot_bgcolor=paper_colors(ThesisTheme.LIGHT)[1],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Ablation grid search, vanilla comparison, Laplacian LFP figures
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_PARQUET = RESULTS_ROOT / "psid_ablation_narrow_band" / "results.parquet"

_ABLATION_SESSIONS = [
    ("PDI1", "2", "PDI1_S2", 80, 12),
    ("PDI1", "4", "PDI1_S4", 80, 6),
    ("PDI4", "2", "PDI4_S2", 80, 10),
    ("PDI4", "3", "PDI4_S3", 65, 10),
]


def build_ablation_heatmap_figure(metric_col: str = "pearson_mean", title: str = "PSID ablation: Pearson r (Z)") -> Figure:
    """Heatmap of nx × n1 ablation results for each session."""
    try:
        df = pl.read_parquet(ABLATION_PARQUET)
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Ablation results not found", x=0.5, y=0.5, showarrow=False)
        return fig

    ncols = 2
    nrows = 2
    subplot_titles = [s[2] for s in _ABLATION_SESSIONS]
    fig = make_subplots(rows=nrows, cols=ncols, vertical_spacing=0.2, horizontal_spacing=0.1,
                        subplot_titles=subplot_titles)

    for pi, (pid, sess, label, final_nx, final_n1) in enumerate(_ABLATION_SESSIONS):
        r, c = divmod(pi, ncols)
        sdf = df.filter((pl.col("participant_id") == pid) & (pl.col("session") == sess))
        # Deduplicate: keep latest run per nx/n1
        sdf = sdf.sort("run_name", descending=True).unique(subset=["nx", "n1"], keep="first")

        nx_vals = sorted(sdf["nx"].unique().to_list())
        n1_vals = sorted(sdf["n1"].unique().to_list())

        mat = np.full((len(n1_vals), len(nx_vals)), np.nan)
        text_mat = np.empty_like(mat, dtype=object)
        for row_data in sdf.iter_rows(named=True):
            nx_i = nx_vals.index(row_data["nx"])
            n1_i = n1_vals.index(row_data["n1"])
            val = row_data.get(metric_col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                mat[n1_i, nx_i] = val
                text_mat[n1_i, nx_i] = f"{val:.3f}"
            else:
                text_mat[n1_i, nx_i] = ""

        # Skip invalid cells where n1 > nx
        for n1_i, n1_v in enumerate(n1_vals):
            for nx_i, nx_v in enumerate(nx_vals):
                if n1_v > nx_v:
                    mat[n1_i, nx_i] = np.nan
                    text_mat[n1_i, nx_i] = ""

        valid = mat[~np.isnan(mat)]
        zmin = float(np.min(valid)) - 0.01 if len(valid) > 0 else 0
        zmax = float(np.max(valid)) + 0.01 if len(valid) > 0 else 1

        fig.add_trace(
            go.Heatmap(
                z=mat, x=list(range(len(nx_vals))), y=list(range(len(n1_vals))),
                text=text_mat, texttemplate="%{text}",
                colorscale="Blues", zmin=zmin, zmax=zmax,
                showscale=(pi == 0),
            ),
            row=r + 1, col=c + 1,
        )
        fig.update_xaxes(
            title_text="nx", tickmode="array",
            tickvals=list(range(len(nx_vals))), ticktext=[str(v) for v in nx_vals],
            row=r + 1, col=c + 1,
        )
        fig.update_yaxes(
            title_text="n1" if c == 0 else "", tickmode="array",
            tickvals=list(range(len(n1_vals))), ticktext=[str(v) for v in n1_vals],
            row=r + 1, col=c + 1,
        )
        # Mark the final (selected) config
        try:
            bx = nx_vals.index(final_nx)
            by = n1_vals.index(final_n1)
            fig.add_shape(
                type="rect",
                x0=float(bx) - 0.5, x1=float(bx) + 0.5,
                y0=float(by) - 0.5, y1=float(by) + 0.5,
                line=dict(color="#C0392B", width=2.5),
                fillcolor="rgba(0,0,0,0)", layer="above",
                row=r + 1, col=c + 1,
            )
        except ValueError:
            pass

    fg = true_line_color(ThesisTheme.LIGHT)
    fig.update_layout(
        title=title,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=700, paper_bgcolor=paper_colors(ThesisTheme.LIGHT)[0],
        plot_bgcolor=paper_colors(ThesisTheme.LIGHT)[1],
    )
    fig.update_annotations(font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY))
    return fig


def build_ablation_pearson_figure() -> Figure:
    return build_ablation_heatmap_figure("pearson_mean", "PSID ablation: Pearson r (behavioral Z)")


def build_ablation_rmse_y_figure() -> Figure:
    return build_ablation_heatmap_figure("rmse_Y", "PSID ablation: RMSE (neural Y)")


def build_vanilla_comparison_figure() -> Figure:
    """Grouped bar chart comparing improved vs vanilla PSID across sessions."""
    import json as _json

    fg = true_line_color(ThesisTheme.LIGHT)
    grd = grid_color(ThesisTheme.LIGHT)

    sessions = [
        ("PDI1_S2", "psid_behavioral_PDI1_2_nx_80_n12_i40_dbs_both_narrow_band",
         "psid_behavioral_PDI1_2_nx_80_n12_i40_vanilla_dbs_both_narrow_band"),
        ("PDI1_S4", "psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
         "psid_behavioral_PDI1_4_nx_80_n6_i40_vanilla_dbs_both_narrow_band"),
        ("PDI4_S2", "psid_behavioral_PDI4_2_nx_80_n10_i40_dbs_both_narrow_band",
         "psid_behavioral_PDI4_2_nx_80_n10_i40_vanilla_dbs_both_narrow_band"),
        ("PDI4_S3", "psid_behavioral_PDI4_3_nx65_n10_i40_dbs_both_narrow_band",
         "psid_behavioral_PDI4_3_nx65_n10_i40_vanilla_dbs_both_narrow_band"),
    ]

    labels, r_improved, r_vanilla, rmse_improved, rmse_vanilla = [], [], [], [], []
    for label, improved_dir, vanilla_dir in sessions:
        for name, result_dir, r_list, rmse_list in [
            ("improved", improved_dir, r_improved, rmse_improved),
            ("vanilla", vanilla_dir, r_vanilla, rmse_vanilla),
        ]:
            res_path = RESULTS_ROOT / result_dir
            if not res_path.exists():
                r_list.append(np.nan)
                rmse_list.append(np.nan)
                continue
            # Find train test_results parquet
            train_dir = res_path / "train"
            parquets = list(train_dir.glob("test_results_*.parquet")) if train_dir.exists() else []
            if not parquets:
                r_list.append(np.nan)
                rmse_list.append(np.nan)
                continue
            try:
                tdf = pl.read_parquet(parquets[0])
                # Try both column naming conventions
                for col in ["metric_pearson_r_mean_Z", "pearson_mean_Z"]:
                    if col in tdf.columns:
                        vals = tdf[col].to_list()
                        valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                        r_list.append(float(np.mean(valid)) if valid else np.nan)
                        break
                else:
                    r_list.append(np.nan)
                # Compute RMSE from Z and Zp
                z_true = np.array(tdf["Z"][0].to_list())
                z_pred = np.array(tdf["Zp"][0].to_list())
                rmse_val = float(np.sqrt(np.mean((z_true - z_pred) ** 2)))
                rmse_list.append(rmse_val)
            except Exception:
                r_list.append(np.nan)
                rmse_list.append(np.nan)
        labels.append(label)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Pearson r (Z)", "RMSE (Z)"],
                        horizontal_spacing=0.12)

    fig.add_trace(go.Bar(x=labels, y=r_improved, name="Improved (BK + rescale)",
                          marker_color="#3b82f6", width=0.35), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=r_vanilla, name="Vanilla PSID",
                          marker_color="#f97316", width=0.35), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=rmse_improved, name="Improved (BK + rescale)",
                          marker_color="#3b82f6", width=0.35, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=rmse_vanilla, name="Vanilla PSID",
                          marker_color="#f97316", width=0.35, showlegend=False), row=1, col=2)

    fig.update_layout(
        barmode="group",
        title="Improved PSID vs Vanilla (no backward Kalman, no A rescaling)",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=400, paper_bgcolor=paper_colors(ThesisTheme.LIGHT)[0],
        plot_bgcolor=paper_colors(ThesisTheme.LIGHT)[1],
        legend=dict(orientation="h", font=dict(size=FONT_SIZE_TICK)),
    )
    fig.update_yaxes(title_text="Pearson r", showgrid=True, gridcolor=grd, row=1, col=1)
    fig.update_yaxes(title_text="RMSE", showgrid=True, gridcolor=grd, row=1, col=2)
    return fig


def build_laplacian_prediction_figure() -> Figure:
    """Summary bar chart of Laplacian LFP prediction quality across sessions."""
    fg = true_line_color(ThesisTheme.LIGHT)
    grd = grid_color(ThesisTheme.LIGHT)

    lapl_sessions = [
        ("PDI1_S2", "psid_laplacian_PDI1_2_nx_80_n12_i20_dbs_both_2hz_band"),
        ("PDI1_S4", "psid_laplacian_PDI1_4_nx_80_n6_i20_dbs_both_2hz_band"),
        ("PDI4_S2", "psid_laplacian_PDI4_2_nx_80_n10_i20_dbs_both_2hz_band"),
        ("PDI4_S3", "psid_laplacian_PDI4_3_nx_65_n10_i20_dbs_both_2hz_band"),
    ]

    labels, r_y_vals, r_z_vals = [], [], []
    for label, result_dir in lapl_sessions:
        res_path = RESULTS_ROOT / result_dir / "train"
        parquets = list(res_path.glob("test_results_*.parquet")) if res_path.exists() else []
        if not parquets:
            r_y_vals.append(np.nan)
            r_z_vals.append(np.nan)
            labels.append(label)
            continue
        try:
            tdf = pl.read_parquet(parquets[0])
            r_y_vals.append(float(tdf["metric_pearson_r_mean"][0]))
            r_z_vals.append(float(tdf["metric_pearson_r_mean_Z"][0]))
        except Exception:
            r_y_vals.append(np.nan)
            r_z_vals.append(np.nan)
        labels.append(label)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=r_y_vals, name="Y (ECoG recon.)", marker_color="#3b82f6", width=0.35))
    fig.add_trace(go.Bar(x=labels, y=r_z_vals, name="Z (Laplacian pred.)", marker_color="#ef4444", width=0.35))

    fig.update_layout(
        barmode="group",
        title="PSID Laplacian LFP prediction: ECoG reconstruction vs depth Laplacian prediction",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=400,
        paper_bgcolor=paper_colors(ThesisTheme.LIGHT)[0],
        plot_bgcolor=paper_colors(ThesisTheme.LIGHT)[1],
        legend=dict(orientation="h", font=dict(size=FONT_SIZE_TICK)),
        yaxis=dict(title_text="Pearson r", showgrid=True, gridcolor=grd,
                   range=[0, 1.05]),
    )
    return fig


def build_laplacian_timeseries_figure() -> Figure:
    """Time series plot of Laplacian prediction for one example trial per session."""
    fg = true_line_color(ThesisTheme.LIGHT)

    lapl_sessions = [
        ("PDI1_S2", "psid_laplacian_PDI1_2_nx_80_n12_i20_dbs_both_2hz_band"),
        ("PDI1_S4", "psid_laplacian_PDI1_4_nx_80_n6_i20_dbs_both_2hz_band"),
        ("PDI4_S2", "psid_laplacian_PDI4_2_nx_80_n10_i20_dbs_both_2hz_band"),
        ("PDI4_S3", "psid_laplacian_PDI4_3_nx_65_n10_i20_dbs_both_2hz_band"),
    ]

    nrows = len(lapl_sessions)
    fig = make_subplots(rows=nrows, cols=1, vertical_spacing=0.08,
                        subplot_titles=[s[0] for s in lapl_sessions])

    for ri, (label, result_dir) in enumerate(lapl_sessions, 1):
        res_path = RESULTS_ROOT / result_dir / "train"
        parquets = list(res_path.glob("test_results_*.parquet")) if res_path.exists() else []
        if not parquets:
            fig.add_annotation(text="No data", x=0.5, y=0.5, xref="x domain", yref="y domain",
                               showarrow=False, row=ri, col=1)
            continue
        try:
            tdf = pl.read_parquet(parquets[0])
            z_true = np.array(tdf["Z"][0].to_list())
            z_pred = np.array(tdf["Zp"][0].to_list())
            # Plot first Laplacian channel (delta band)
            t = np.arange(z_true.shape[0]) / 80.0
            fig.add_trace(go.Scatter(x=t, y=z_true[:, 0], name="True" if ri == 1 else None,
                                      line=dict(color="#2563eb", width=1),
                                      showlegend=(ri == 1)), row=ri, col=1)
            fig.add_trace(go.Scatter(x=t, y=z_pred[:, 0], name="Predicted" if ri == 1 else None,
                                      line=dict(color="#dc2626", width=1, dash="dot"),
                                      showlegend=(ri == 1)), row=ri, col=1)
        except Exception:
            fig.add_annotation(text="Error loading", x=0.5, y=0.5, xref="x domain", yref="y domain",
                               showarrow=False, row=ri, col=1)

    fig.update_layout(
        title="Laplacian LFP prediction: LAPLACIAN_13-15 delta band (example trial)",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=200 * nrows + 80,
        paper_bgcolor=paper_colors(ThesisTheme.LIGHT)[0],
        plot_bgcolor=paper_colors(ThesisTheme.LIGHT)[1],
    )
    fig.update_xaxes(title_text="Time (s)", row=nrows, col=1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib plot_* functions (for PDF/PNG file export)
# ─────────────────────────────────────────────────────────────────────────────

def _box(ax, cx: float, cy: float, w: float, h: float, label: str, sublabel: str, color: str, fontsize: float = 8.5) -> None:
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#888780", linewidth=0.6, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(cx, cy + (0.12 if sublabel else 0), label, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center", fontsize=7.0, color="#5F5E5A", zorder=4)


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444441", lw=1.1), zorder=2)


def plot_preprocessing_pipeline(save: bool = True) -> plt.Figure:
    """Two-branch flowchart: neural (left) and behavioural (right) → Model input."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _box(ax, 2.5, 9.2, 2.8, 0.7, "Raw iEEG", "22 kHz · LFP 16ch + ECoG 4ch", C_RAW)
    _arrow(ax, 2.5, 8.85, 2.5, 8.1)
    _box(ax, 2.5, 7.7, 2.8, 0.7, "Resample + bandpass", "↓ 60 Hz · 3–28 Hz BP (MNE)", C_PROC)
    _arrow(ax, 2.5, 7.35, 2.5, 6.6)
    _box(ax, 2.5, 6.2, 2.8, 0.7, "Common avg. re-reference", "Per modality, per trial", C_PROC)
    _arrow(ax, 2.5, 5.85, 2.5, 5.1)
    _box(ax, 2.5, 4.7, 2.8, 0.7, "Scale ×10⁶", "→ microvolts", C_PROC)
    _arrow(ax, 2.5, 4.35, 2.5, 3.6)
    _box(ax, 2.5, 3.2, 2.8, 0.9, "Narrowband features",
         "δ/θ/α raw · β 13–29 Hz Hilbert env\n4 ch × 29 bands = 116 inputs (Y)", C_FEAT, fontsize=8.0)

    _box(ax, 7.5, 9.2, 2.8, 0.7, "Tablet coordinates", "x(t), y(t) at variable rate", C_RAW)
    _arrow(ax, 7.5, 8.85, 7.5, 8.1)
    _box(ax, 7.5, 7.7, 2.8, 0.7, "Kinematic derivation", "v, a, j in x/y/xy/mag", C_PROC)
    _arrow(ax, 7.5, 7.35, 7.5, 6.6)
    _box(ax, 7.5, 6.2, 2.8, 0.7, "Savitzky-Golay smooth", "~200 ms window · 3rd order", C_PROC)
    _arrow(ax, 7.5, 5.85, 7.5, 5.1)
    _box(ax, 7.5, 4.7, 2.8, 0.7, "Interpolate to 60 Hz", "Linear onto neural grid", C_PROC)
    _arrow(ax, 7.5, 4.35, 7.5, 3.6)
    _box(ax, 7.5, 3.2, 2.8, 0.7, "Tracing speed", "velocity magnitude (Z)", C_FEAT)

    seg_rect = mpatches.FancyBboxPatch(
        (1.0, 2.35), 8.0, 0.55,
        boxstyle="round,pad=0.04",
        facecolor="#FAEEDA", edgecolor="#854F0B", linewidth=0.7, zorder=2,
    )
    ax.add_patch(seg_rect)
    ax.text(5.0, 2.625, "Trial segmentation: 9 s trial · ±2 s margin buffer",
            ha="center", va="center", fontsize=8.0, color="#412402", zorder=4)
    _arrow(ax, 2.5, 2.75, 2.5, 2.35)
    _arrow(ax, 7.5, 2.75, 7.5, 2.35)
    _arrow(ax, 2.5, 2.35, 4.5, 1.75)
    _arrow(ax, 7.5, 2.35, 5.5, 1.75)
    _box(ax, 5.0, 1.45, 3.2, 0.7, "Model input (Y, Z) at 60 Hz", "Train 60% · Val 20% · Test 20%", C_MERGE, fontsize=8.5)

    for i, (c, lbl) in enumerate([(C_RAW, "Raw input"), (C_PROC, "Processing step"), (C_FEAT, "Feature / output"), (C_MERGE, "Model input")]):
        px = 0.55 + i * 2.3
        r = mpatches.FancyBboxPatch((px, 0.15), 0.35, 0.28, boxstyle="round,pad=0.03", facecolor=c, edgecolor="#888780", lw=0.5)
        ax.add_patch(r)
        ax.text(px + 0.45, 0.29, lbl, va="center", fontsize=7.5, color="#444441")

    fig.suptitle("Data preprocessing pipeline", fontsize=12, y=0.97)
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"preprocessing_pipeline.{ext}", bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PSD comparison: DBS-ON vs DBS-OFF
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trial_psd(signal_1d: np.ndarray, fs: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(len(signal_1d), fs * 2)
    freqs, psd = welch(signal_1d, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, 10 * np.log10(psd + 1e-20)


def plot_psd_dbs_comparison(
    band_shade: Optional[Tuple[float, float]] = (13, 29),
    save: bool = True,
) -> plt.Figure:
    """Mean ± SEM PSD per condition, per participant×session."""
    apply_style()
    triplets = get_triplets()
    n_panels = len(triplets)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    figsize = (4.5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    axes_flat = np.atleast_2d(axes).ravel()
    OFF_C, ON_C = COLORS["dbs_off"], COLORS["dbs_on"]

    for pi, tri in enumerate(triplets):
        ax = axes_flat[pi]
        trials = load_results_for_triplet(tri)
        if not trials:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue

        psds_off, psds_on = [], []
        freqs_ref = None
        for row in trials:
            y = row.get("Y")
            stim = row.get("stim", "off")
            if y is None:
                continue
            y_arr = np.asarray(y)
            channel_mean = y_arr.mean(axis=1) if y_arr.ndim == 2 else np.asarray(y).ravel()
            freqs, psd_db = _compute_trial_psd(channel_mean, fs=FS)
            if freqs_ref is None:
                freqs_ref = freqs
            if stim == "off":
                psds_off.append(psd_db)
            else:
                psds_on.append(psd_db)

        if freqs_ref is None:
            continue
        for cond_psds, col, label in [(psds_off, OFF_C, "DBS-OFF"), (psds_on, ON_C, "DBS-ON")]:
            if not cond_psds:
                continue
            mat = np.vstack(cond_psds)
            mean = mat.mean(axis=0)
            sem = mat.std(axis=0) / np.sqrt(len(mat))
            ax.fill_between(freqs_ref, mean - sem, mean + sem, color=col, alpha=0.15, lw=0)
            ax.plot(freqs_ref, mean, color=col, lw=1.8, label=label)
        if band_shade is not None:
            ax.axvspan(band_shade[0], band_shade[1], color=BAND_SHADE_COLOR, alpha=0.05)
            ax.axvline(band_shade[0], color=BAND_SHADE_COLOR, lw=0.8, ls="--", alpha=0.6)
            ax.axvline(band_shade[1], color=BAND_SHADE_COLOR, lw=0.8, ls="--", alpha=0.6)
        ax.text(0.04, 0.95, tri.label or f"{tri.psid_variant}", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        if pi % ncols == 0:
            ax.set_ylabel("power/freq (dB/Hz)")
        if pi >= n_panels - ncols:
            ax.set_xlabel("frequency (Hz)")

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    handles = [plt.Line2D([], [], color=OFF_C, lw=1.8, label="DBS-OFF mean"), plt.Line2D([], [], color=ON_C, lw=1.8, label="DBS-ON mean")]
    if band_shade is not None:
        handles.append(plt.Line2D([], [], color=BAND_SHADE_COLOR, lw=0.8, ls="--", label=f"band {band_shade[0]}-{band_shade[1]} Hz"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("ECoG PSD: DBS-ON vs DBS-OFF", fontsize=12, y=1.01)
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"psd_dbs_comparison.{ext}", bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tracing speed: DBS-ON vs DBS-OFF
# ─────────────────────────────────────────────────────────────────────────────

def plot_tracing_speed_dbs_comparison(save: bool = True) -> plt.Figure:
    """Mean traces for ``tracing_velocity_x`` and ``tracing_acceleration_magnitude`` (YAML output names)."""
    apply_style()
    triplets = get_triplets()
    n_panels = len(triplets)
    ncols = 2
    nrows_pair = (n_panels + ncols - 1) // ncols
    nrows_plot = 2 * nrows_pair
    figsize = (4.5 * ncols, 2.65 * nrows_plot)
    t = np.arange(FS * 9) / FS
    fig, axes = plt.subplots(
        nrows_plot,
        ncols,
        figsize=figsize,
        sharex="col",
        sharey=False,
        gridspec_kw={"hspace": 0.35, "wspace": 0.12},
    )
    axes = np.atleast_2d(axes)
    OFF_C, ON_C = COLORS["dbs_off"], COLORS["dbs_on"]
    n_slot = nrows_pair * ncols

    for idx in range(n_slot):
        r_pair, c = divmod(idx, ncols)
        r0 = 2 * r_pair
        ax_v, ax_a = axes[r0, c], axes[r0 + 1, c]
        vis = idx < n_panels
        ax_v.set_visible(vis)
        ax_a.set_visible(vis)
        if not vis:
            continue
        tri = triplets[idx]
        trials = load_results_for_triplet(tri)
        iv, ia = _z_column_indices_for_outputs(tri)

        def _collect(col_idx: int) -> tuple[list, list]:
            off_l, on_l = [], []
            for row in trials:
                z = row.get("Z")
                if z is None:
                    continue
                za = _trial_z_column(z, col_idx)
                stim = row.get("stim", "off")
                (off_l if stim == "off" else on_l).append(za)
            return off_l, on_l

        ax_v.set_title(f"{tri.label or tri.psid_variant}\ntracing_velocity_x", fontsize=8.5)
        ax_a.set_title("tracing_acceleration_magnitude", fontsize=8.5)

        if not trials:
            for ax in (ax_v, ax_a):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue

        for ax, col_idx, ylbl in (
            (ax_v, iv, "tracing_velocity_x (z)"),
            (ax_a, ia, "tracing_acceleration_magnitude (z)"),
        ):
            z_off, z_on = _collect(col_idx)
            z_off, z_on = _zscore_traces(z_off, z_on)
            for cond_list, col, label in [(z_off, OFF_C, "DBS-OFF"), (z_on, ON_C, "DBS-ON")]:
                if not cond_list:
                    continue
                mat = np.vstack(cond_list)
                mean = np.nanmean(mat, axis=0)
                ax.plot(t, mean, color=col, lw=1.8, label=f"{label} (n={len(cond_list)})")
            ax.set_xlim(0, 9)
            if c == 0:
                ax.set_ylabel(ylbl, fontsize=8)

    for c in range(ncols):
        axes[nrows_plot - 1, c].set_xlabel("time (s)", fontsize=8)

    handles = [
        plt.Line2D([], [], color=OFF_C, lw=1.8, label="DBS-OFF"),
        plt.Line2D([], [], color=ON_C, lw=1.8, label="DBS-ON"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "tracing_velocity_x and tracing_acceleration_magnitude (trial means, z per metric)",
        fontsize=11,
        y=1.01,
    )
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"tracing_speed_dbs_comparison.{ext}", bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Grid search heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def _load_grid_df() -> Optional[pl.DataFrame]:
    if not GRID_SEARCH_PARQUET.exists():
        return None
    try:
        return pl.read_parquet(GRID_SEARCH_PARQUET)
    except Exception:
        return None


def _build_heatmap_matrix(df: pl.DataFrame, participant_id: str, session: str, metric_col: str) -> np.ndarray:
    sub = df.filter(pl.col("participant_id") == participant_id)
    if "session" in sub.columns:
        sub = sub.filter(pl.col("session").cast(pl.Utf8) == str(session))
    mat = np.full((len(N1_VALS), len(NX_VALS)), np.nan)
    for xi, nx in enumerate(NX_VALS):
        for n1i, n1 in enumerate(N1_VALS):
            if n1 > nx:
                continue
            match = sub.filter((pl.col("nx") == nx) & (pl.col("n1") == n1))
            if not match.is_empty() and metric_col in match.columns:
                # Single run per cell when possible (avoid averaging duplicate grid-search rows).
                val = match[metric_col][0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    mat[n1i, xi] = float(val)
    return mat


def _plot_grid_heatmap(ax, mat: np.ndarray, vmin: float, vmax: float, cmap: str, inverse: bool, selected: Optional[Tuple[int, int]]) -> None:
    z = -mat if inverse else mat
    zmin, zmax = (-vmax, -vmin) if inverse else (vmin, vmax)
    im = ax.imshow(z, aspect="auto", vmin=zmin, vmax=zmax, cmap=cmap, origin="upper")
    for n1i, n1 in enumerate(N1_VALS):
        for xi, nx in enumerate(NX_VALS):
            if n1 > nx:
                ax.add_patch(mpatches.Rectangle((xi - 0.5, n1i - 0.5), 1, 1, hatch="///", facecolor="#D3D1C7", edgecolor="white", lw=0.3, zorder=2))
            else:
                val = mat[n1i, xi]
                if not np.isnan(val):
                    tc = "white" if (val > (vmin + vmax) / 2 + 0.05) else "#2C2C2A"
                    ax.text(xi, n1i, f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}", ha="center", va="center", fontsize=7.0, color=tc)
    if selected and selected[0] in NX_VALS and selected[1] in N1_VALS:
        xi_sel = NX_VALS.index(selected[0])
        n1i_sel = N1_VALS.index(selected[1])
        ax.add_patch(mpatches.Rectangle((xi_sel - 0.5, n1i_sel - 0.5), 1, 1, fill=False, edgecolor=BAND_SHADE_COLOR, linewidth=1.8, zorder=5))
    return im


def _plot_grid_search_figure(
    df: Optional[pl.DataFrame], metric_col: str, cmap: str, vmin: float, vmax: float, inverse: bool, title: str, basename: str, save: bool,
) -> plt.Figure:
    apply_style()
    triplets = get_triplets()
    n_panels = len(triplets)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    figsize = (4.5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, gridspec_kw={"hspace": 0.20, "wspace": 0.08})
    axes_flat = np.atleast_2d(axes).ravel()
    im_ref = None

    for pi, tri in enumerate(triplets):
        ax = axes_flat[pi]
        key = tri.label or tri.psid_variant or ""
        sel = SELECTED_CONFIG.get(key)
        if df is not None and metric_col in df.columns:
            _m = re.search(r'(PDI[14])_(\d+)', key)
            if _m:
                pid, sess = _m.group(1), _m.group(2)
            else:
                pid = "PDI1" if "PDI1" in (tri.psid_variant or "") else "PDI4"
                sess = "2"
            mat = _build_heatmap_matrix(df, pid, sess, metric_col)
        else:
            mat = np.full((len(N1_VALS), len(NX_VALS)), np.nan)
        im = _plot_grid_heatmap(ax, mat, vmin, vmax, cmap, inverse, sel)
        if im_ref is None and not np.all(np.isnan(mat)):
            im_ref = im
        ax.set_xticks(range(len(NX_VALS)))
        ax.set_xticklabels([str(v) for v in NX_VALS], fontsize=7.5, rotation=45, ha="right")
        ax.set_yticks(range(len(N1_VALS)))
        ax.set_yticklabels([str(v) for v in N1_VALS], fontsize=7.5)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("nx")
        if pi % ncols == 0:
            ax.set_ylabel("n₁")

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes_flat[:n_panels].tolist(), orientation="vertical", fraction=0.015, pad=0.02)
    fig.suptitle(title, fontsize=12, y=1.01)
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"{basename}.{ext}", bbox_inches="tight")
    return fig


def plot_grid_search_pearson(save: bool = True) -> plt.Figure:
    df = _load_grid_df()
    return _plot_grid_search_figure(df, "pearson_fisher", "Blues", 0.25, 0.85, False, "Grid search: validation Pearson r (Fisher Z)", "grid_search_pearson", save)


def plot_grid_search_rmse(save: bool = True) -> plt.Figure:
    df = _load_grid_df()
    return _plot_grid_search_figure(df, "rmse_Z", "Reds_r", 0.3, 1.2, False, "Grid search: validation RMSE (Z)", "grid_search_rmse", save)


def plot_grid_search_lag(save: bool = True) -> plt.Figure:
    df = _load_grid_df()
    return _plot_grid_search_figure(df, "xcorr_lag_mean_ms", "viridis", 0, 100, False, "Grid search: validation lag (ms)", "grid_search_lag", save)


# ─────────────────────────────────────────────────────────────────────────────
# Trial count summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_trial_count_summary(save: bool = True) -> plt.Figure:
    """Bar chart: trial counts per participant×session×condition."""
    apply_style()
    triplets = get_triplets()
    rows = []
    for tri in triplets:
        trials = load_results_for_triplet(tri)
        off = sum(1 for t in trials if t.get("stim") == "off")
        on = sum(1 for t in trials if t.get("stim") == "on")
        rows.append({"session": tri.label or "", "DBS-OFF": off, "DBS-ON": on, "total": off + on})

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(rows))
    w = 0.35
    ax.bar(x - w / 2, [r["DBS-OFF"] for r in rows], w, label="DBS-OFF", color=COLORS["dbs_off"])
    ax.bar(x + w / 2, [r["DBS-ON"] for r in rows], w, label="DBS-ON", color=COLORS["dbs_on"])
    ax.set_xticks(x)
    ax.set_xticklabels([r["session"] for r in rows])
    ax.set_ylabel("trial count")
    ax.legend()
    fig.suptitle("Trial count per session × DBS condition", fontsize=12)
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"trial_count_summary.{ext}", bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Placeholders
# ─────────────────────────────────────────────────────────────────────────────

def plot_beta_burst_placeholder(save: bool = True) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, "Beta burst duration/amplitude: requires burst detection pipeline.",
            ha="center", va="center", transform=ax.transAxes, fontsize=10, wrap=True)
    ax.axis("off")
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"beta_burst_placeholder.{ext}", bbox_inches="tight")
    return fig


def plot_residual_diagnostics_placeholder(save: bool = True) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, "Residual diagnostics (Ljung-Box): requires residual extraction from model.",
            ha="center", va="center", transform=ax.transAxes, fontsize=10, wrap=True)
    ax.axis("off")
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"residual_diagnostics_placeholder.{ext}", bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis appendix figures (matplotlib)")
    parser.add_argument("--figures", nargs="*",
                        default=["preprocessing", "psd", "tracing", "grid_pearson", "grid_rmse", "grid_lag", "trial_count", "beta_burst", "residual"],
                        help="Which figures to generate")
    parser.add_argument("--no-save", action="store_true", help="Show only, do not save")
    parser.add_argument("--band-shade", nargs=2, type=float, default=[13, 29], metavar=("LO", "HI"))
    args = parser.parse_args()

    save = not args.no_save
    band_shade = (float(args.band_shade[0]), float(args.band_shade[1]))

    for f in args.figures:
        if f == "preprocessing":
            plot_preprocessing_pipeline(save=save)
        elif f == "psd":
            plot_psd_dbs_comparison(band_shade=band_shade, save=save)
        elif f == "tracing":
            plot_tracing_speed_dbs_comparison(save=save)
        elif f == "grid_pearson":
            plot_grid_search_pearson(save=save)
        elif f == "grid_rmse":
            plot_grid_search_rmse(save=save)
        elif f == "grid_lag":
            plot_grid_search_lag(save=save)
        elif f == "trial_count":
            plot_trial_count_summary(save=save)
        elif f == "beta_burst":
            plot_beta_burst_placeholder(save=save)
        elif f == "residual":
            plot_residual_diagnostics_placeholder(save=save)
        else:
            print(f"Unknown figure: {f}")

    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
