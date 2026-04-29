"""
Combined grouped results figure: four subplot columns in one row.

  col 1: Panel A left  — within vs cross RMSE (DBS-OFF), small jittered dots + box quartiles
  col 2: Panel A right — within vs cross RMSE (DBS-ON),  small jittered dots + box quartiles
  col 3: Panel B       — ROC curves per feature group (mean ± std across sessions)
  col 4: Panel C       — Forecast RMSE vs horizon (PSID vs VARMA, OFF/ON as line style)

No subplot titles — caller puts section labels in surrounding HTML/markdown.

All data must be pre-loaded by the caller via the existing loaders:
  * collect_within_cross_rmse          → WithinCrossRmseData
  * collect_classification_roc_curves  → List[ClassificationRocCurve]
  * collect_forecast_horizon_rmse      → ForecastHorizonRmseData
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.thesis.aggregate_rmse import WithinCrossRmseData
from dashboard.thesis.classification_f1_data import (
    GROUP_ORDER,
    ClassificationRocCurve,
)
from dashboard.thesis.constants import (
    COLOR_DBS_ON,
    COLOR_DPAD,
    COLOR_PSID,
    COLOR_VARMA,
    FIGURE_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    dbs_badge_style,
    rmse_axis_label,
    grid_color,
    legend_bgcolor,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.forecast_horizon_rmse import ForecastHorizonRmseData

logger = logging.getLogger(__name__)

# ── feature-group colours (match classification_f1_figure.py) ────────────────
_FEAT_COLORS: dict[str, str] = {
    "xp": "#185FA5",
    "xp_1": "#0F6E56",
    "xp_2": "#993C1D",
    "xp_with_dbs": "#854F0B",
}
_FEAT_SHORT: dict[str, str] = {
    "xp": "Xp",
    "xp_1": "Xp\u2081",
    "xp_2": "Xp\u2082",
    "xp_with_dbs": "Xp+DBS",
}

# ── Panel A layout constants ──────────────────────────────────────────────────
_BW = 0.20
_GAP = 0.05
_GROUP_GAP = 0.40


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _xref(col: int) -> str:
    return "x" if col == 1 else f"x{col}"


def _yref(col: int) -> str:
    return "y" if col == 1 else f"y{col}"


# ── Panel A: within/cross RMSE strip ─────────────────────────────────────────


def _add_within_cross_col(
    fig: go.Figure,
    cells: list,
    models: list[str],
    colors: list[str],
    col: int,
    rng: np.random.Generator,
    show_legend: bool,
) -> float:
    """Box quartiles + small Gaussian-jittered dots per within/cross group. Returns y_max."""
    y_max = 0.0
    x_pos = 0.0
    tick_positions: list[float] = []

    for mi, (within_vals, cross_vals) in enumerate(cells):
        x_within = x_pos
        x_cross = x_pos + _BW + _GAP

        for vals, x, is_cross, dot_alpha in [
            (within_vals, x_within, False, 0.40),
            (cross_vals, x_cross, True, 0.30),
        ]:
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            y_max = max(y_max, float(np.nanmax(arr)))
            c = colors[mi]
            fillcolor = _hex_to_rgba(c, 0.25) if not is_cross else "rgba(0,0,0,0)"
            lname = f"{models[mi]} {'cross' if is_cross else 'within'}"

            # Box (quartile summary, no outlier points)
            fig.add_trace(
                go.Box(
                    y=arr,
                    x=[x] * len(arr),
                    marker_color=c,
                    line=dict(color=c, width=1.0),
                    fillcolor=fillcolor,
                    boxpoints=False,
                    quartilemethod="exclusive",
                    width=_BW * 0.85,
                    name=lname,
                    showlegend=show_legend,
                    legendgroup=lname,
                ),
                row=1,
                col=col,
            )
            # Small Gaussian-jittered dots (Pearson-r plot style)
            jitter = rng.normal(0.0, 0.055, size=len(arr))
            fig.add_trace(
                go.Scatter(
                    x=np.clip(
                        np.array([x] * len(arr)) + jitter, x - _BW * 0.5, x + _BW * 0.5
                    ),
                    y=arr,
                    mode="markers",
                    marker=dict(size=4, color=c, opacity=dot_alpha, line=dict(width=0)),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col,
            )

        tick_positions.append(x_within + (_BW + _GAP) / 2)
        x_pos += 2 * _BW + _GAP + _GROUP_GAP

    fig.update_xaxes(
        tickvals=tick_positions,
        ticktext=models,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=col,
    )
    return y_max


# ── Panel B: ROC curves ───────────────────────────────────────────────────────

_FPR_GRID = np.linspace(0, 1, 200)


def _add_roc_col(
    fig: go.Figure,
    roc_curves: List[ClassificationRocCurve],
    col: int,
    fg: str,
) -> None:
    """
    Mean ± 1 std ROC curve per feature group (interpolated onto common FPR grid).
    Diagonal chance reference.
    """
    xr = _xref(col)
    yr = _yref(col)

    # Reference diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="rgba(160,160,160,0.6)", width=1.0, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=col,
    )

    for g in GROUP_ORDER:
        curves = [c for c in roc_curves if c.group == g]
        if not curves:
            continue
        color = _FEAT_COLORS[g]
        # Interpolate each session's TPR onto the shared FPR grid
        tpr_interp = []
        aucs = []
        for c in curves:
            try:
                tpr_i = np.interp(_FPR_GRID, c.fpr, c.tpr)
                tpr_interp.append(tpr_i)
                if np.isfinite(c.roc_auc):
                    aucs.append(c.roc_auc)
            except Exception:
                continue
        if not tpr_interp:
            continue

        mat = np.stack(tpr_interp, axis=0)  # (n_sessions, 200)
        mean_tpr = np.mean(mat, axis=0)
        std_tpr = np.std(mat, axis=0)
        mean_auc = float(np.mean(aucs)) if aucs else float("nan")

        # SEM band (±1 std)
        upper = np.clip(mean_tpr + std_tpr, 0, 1)
        lower = np.clip(mean_tpr - std_tpr, 0, 1)
        xb = np.concatenate([_FPR_GRID, _FPR_GRID[::-1]])
        yb = np.concatenate([upper, lower[::-1]])
        fig.add_trace(
            go.Scatter(
                x=xb,
                y=yb,
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.15),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=g,
            ),
            row=1,
            col=col,
        )

        auc_str = f"{mean_auc:.2f}" if np.isfinite(mean_auc) else "—"
        fig.add_trace(
            go.Scatter(
                x=_FPR_GRID,
                y=mean_tpr,
                mode="lines",
                name=f"{_FEAT_SHORT[g]}  (AUC {auc_str})",
                line=dict(color=color, width=1.8),
                showlegend=True,
                legendgroup=g,
                hovertemplate=f"{_FEAT_SHORT[g]}<br>FPR=%{{x:.2f}}<br>TPR=%{{y:.2f}}<extra></extra>",
            ),
            row=1,
            col=col,
        )

    fig.update_xaxes(
        range=[0, 1],
        title_text="False positive rate",
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        tickvals=[0, 0.5, 1],
        showgrid=True,
        showline=True,
        zeroline=False,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=col,
    )
    fig.update_yaxes(
        range=[0, 1.05],
        title_text="True positive rate",
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        tickvals=[0, 0.5, 1],
        showgrid=True,
        showline=True,
        zeroline=False,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=col,
    )


# ── Panel C: forecast horizon RMSE ───────────────────────────────────────────


def _add_forecast_col(
    fig: go.Figure,
    horizon: ForecastHorizonRmseData,
    col: int,
    grid: str,
    fg: str,
    x_max_ms: float = 1000.0,
) -> float:
    """PSID vs VARMA forecast error curves (OFF solid / ON dashed). Returns y_max."""
    x = horizon.x_ms
    if x.size == 0:
        return 1.1

    _COLOR_PSID_ON = COLOR_DBS_ON

    traces = [
        (horizon.mean_psid_off, horizon.sem_psid_off, COLOR_PSID, None, "PSID DBS-OFF"),
        (
            horizon.mean_psid_on,
            horizon.sem_psid_on,
            _COLOR_PSID_ON,
            "dash",
            "PSID DBS-ON",
        ),
        (
            horizon.mean_varma_off,
            horizon.sem_varma_off,
            COLOR_VARMA,
            None,
            "VARMA DBS-OFF",
        ),
        (
            horizon.mean_varma_on,
            horizon.sem_varma_on,
            COLOR_VARMA,
            "dash",
            "VARMA DBS-ON",
        ),
    ]

    y_max = 0.0
    for mean, sem, color, dash, name in traces:
        if mean.size == 0:
            continue
        upper = mean + sem
        lower = mean - sem
        xb = np.concatenate([x, x[::-1]])
        yb = np.concatenate([upper, lower[::-1]])
        fig.add_trace(
            go.Scatter(
                x=xb,
                y=yb,
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.12),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=name,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.8, dash=dash or "solid"),
                showlegend=True,
                legendgroup=name,
                hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        finite_u = upper[np.isfinite(upper)]
        if finite_u.size:
            y_max = max(y_max, float(np.nanmax(finite_u)))

    fig.update_xaxes(
        range=[0, x_max_ms],
        tickmode="array",
        tickvals=[0, 250, 500, 750, 1000],
        title_text="Forecast horizon (ms)",
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linecolor=fg,
        linewidth=1,
        zeroline=False,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=col,
    )
    return max(y_max, 0.5)


# ── Main figure builder ───────────────────────────────────────────────────────


def build_session_grouped_figure(
    within_cross: WithinCrossRmseData,
    roc_curves: List[ClassificationRocCurve],
    horizon_data: ForecastHorizonRmseData,
    theme: ThesisTheme = ThesisTheme.LIGHT,
    jitter_seed: int = 42,
) -> go.Figure:
    """
    Build the combined 4-panel grouped results figure (no subplot titles).

    Parameters
    ----------
    within_cross:
        Per-trial within/cross RMSE for all sessions pooled.
    roc_curves:
        Per-session ROC curves for each feature group.
    horizon_data:
        Per-horizon mean ± SEM forecast error for PSID and VARMA.
    theme:
        Light or dark.
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)
    rng = np.random.default_rng(jitter_seed)

    models = ["PSID", "DPAD", "VARMA"]
    colors = [COLOR_PSID, COLOR_DPAD, COLOR_VARMA]

    off_cells = [
        (within_cross.psid_off_within, within_cross.psid_off_cross),
        (within_cross.dpad_off_within, within_cross.dpad_off_cross),
        (within_cross.varma_off_within, within_cross.varma_off_cross),
    ]
    on_cells = [
        (within_cross.psid_on_within, within_cross.psid_on_cross),
        (within_cross.dpad_on_within, within_cross.dpad_on_cross),
        (within_cross.varma_on_within, within_cross.varma_on_cross),
    ]

    fig = make_subplots(
        rows=1,
        cols=4,
        column_widths=[0.22, 0.22, 0.24, 0.32],
        horizontal_spacing=0.08,
    )

    # ── Panel A ──────────────────────────────────────────────────────────────
    ymax_off = _add_within_cross_col(
        fig, off_cells, models, colors, col=1, rng=rng, show_legend=True
    )
    ymax_on = _add_within_cross_col(
        fig, on_cells, models, colors, col=2, rng=rng, show_legend=False
    )
    ymax_ab = max(ymax_off, ymax_on, 0.85) * 1.08

    # DBS condition badges
    for badge_col, badge_label in [(1, "DBS-OFF"), (2, "DBS-ON")]:
        bfg, bbg = dbs_badge_style(badge_label)
        fig.add_annotation(
            x=0.04,
            y=0.97,
            xref=f"{_xref(badge_col)} domain",
            yref=f"{_yref(badge_col)} domain",
            text=f"<b>{badge_label}</b>",
            showarrow=False,
            font=dict(size=FONT_SIZE_BASE, color=bfg, family=FONT_FAMILY),
            bgcolor=bbg,
            bordercolor=bfg,
            borderwidth=1,
            borderpad=3,
        )

    # ── Panel B ──────────────────────────────────────────────────────────────
    _add_roc_col(fig, roc_curves, col=3, fg=fg)

    # ── Panel C ──────────────────────────────────────────────────────────────
    ymax_fc = _add_forecast_col(fig, horizon_data, col=4, grid=grid, fg=fg)

    # ── Axis styling (RMSE panels) ────────────────────────────────────────────
    _ykw_rmse = dict(
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linecolor=fg,
        linewidth=1,
        zeroline=False,
        tickfont=dict(size=FONT_SIZE_TICK),
    )
    fig.update_yaxes(
        range=[0, ymax_ab],
        title_text=rmse_axis_label(),
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        **_ykw_rmse,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[0, ymax_ab],
        showticklabels=False,
        **_ykw_rmse,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        range=[0, ymax_fc * 1.05],
        title_text=rmse_axis_label(),
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linecolor=fg,
        linewidth=1,
        zeroline=False,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=4,
    )

    for c in (1, 2):
        fig.update_xaxes(
            showgrid=False,
            showline=True,
            linecolor=fg,
            linewidth=1,
            zeroline=False,
            row=1,
            col=c,
        )

    # ── Layout ───────────────────────────────────────────────────────────────
    apply_thesis_style(
        fig,
        theme,
        height=FIGURE_HEIGHT,
        margin=dict(l=70, r=30, t=20, b=110),
        hovermode="closest",
        legend_y=-0.28,
    )
    fig.update_layout(boxmode="overlay")

    return fig
