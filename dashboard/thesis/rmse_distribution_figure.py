"""Bar + SEM + jittered trial RMSE dots; optional Wilcoxon brackets (vs VARMA)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from dashboard.thesis.aggregate_rmse import AggregateRmseData, p_to_stars
from dashboard.thesis.constants import (
    COLOR_DPAD,
    COLOR_PSID,
    COLOR_VARMA,
    FONT_FAMILY,
    ThesisTheme,
    grid_color,
    paper_colors,
    true_line_color,
)

# Bar fill opacity: DBS-OFF vs DBS-ON (within same model hue)
ALPHA_OFF = 0.85
ALPHA_ON = 0.45
# Slightly lower for scatter so bars read first
DOT_ALPHA_OFF = 0.75
DOT_ALPHA_ON = 0.4

X_POS = np.arange(6, dtype=float)
CATEGORY_LABELS = [
    "PSID<br>DBS-OFF",
    "PSID<br>DBS-ON",
    "DPAD-RNN<br>DBS-OFF",
    "DPAD-RNN<br>DBS-ON",
    "VARMA<br>DBS-OFF",
    "VARMA<br>DBS-ON",
]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _model_color_for_index(i: int) -> str:
    if i < 2:
        return COLOR_PSID
    if i < 4:
        return COLOR_DPAD
    return COLOR_VARMA


def _is_on_cell(i: int) -> bool:
    return i % 2 == 1


def build_rmse_distribution_figure(
    data: AggregateRmseData,
    theme: ThesisTheme,
    rng: np.random.Generator,
    jitter: float = 0.12,
    show_brackets: bool = True,
) -> Figure:
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    means = np.array(data.means, dtype=float)
    sems = np.array(data.sems, dtype=float)

    bar_colors = [
        _hex_to_rgba(
            _model_color_for_index(i),
            ALPHA_ON if _is_on_cell(i) else ALPHA_OFF,
        )
        for i in range(6)
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=X_POS,
            y=means,
            error_y=dict(
                type="data",
                array=sems,
                visible=True,
                thickness=1.5,
                color=fg,
            ),
            marker=dict(color=bar_colors, line=dict(width=0)),
            width=0.52,
            name="Mean ± SEM",
            showlegend=True,
        )
    )

    for i in range(6):
        pts = data.trial_rmse[i]
        if not pts:
            continue
        jt = rng.uniform(-jitter, jitter, size=len(pts))
        alpha = DOT_ALPHA_ON if _is_on_cell(i) else DOT_ALPHA_OFF
        fig.add_trace(
            go.Scatter(
                x=X_POS[i] + jt,
                y=pts,
                mode="markers",
                marker=dict(
                    size=5,
                    color=_hex_to_rgba(_model_color_for_index(i), alpha * 0.95),
                    line=dict(width=0),
                ),
                name="Test trials" if i == 0 else None,
                showlegend=i == 0,
                legendgroup="dots",
            )
        )

    ymax_data = 0.0
    for i in range(6):
        if np.isfinite(means[i]):
            ymax_data = max(ymax_data, float(means[i] + (sems[i] if np.isfinite(sems[i]) else 0)))
        for v in data.trial_rmse[i]:
            if np.isfinite(v):
                ymax_data = max(ymax_data, float(v))

    y_span = max(ymax_data * 0.12, 0.04)
    bracket_y: list[tuple[float, float, float, str]] = []
    if show_brackets and data.wilcoxon:
        w = data.wilcoxon
        y0 = ymax_data + y_span * 0.25
        step = y_span * 1.2
        pairs = [
            (0, 4, w.psid_vs_varma_off_p, "PSID vs VARMA (DBS-OFF)"),
            (2, 4, w.dpad_vs_varma_off_p, "DPAD vs VARMA (DBS-OFF)"),
            (1, 5, w.psid_vs_varma_on_p, "PSID vs VARMA (DBS-ON)"),
            (3, 5, w.dpad_vs_varma_on_p, "DPAD vs VARMA (DBS-ON)"),
        ]
        for k, (xa, xb, p, _lab) in enumerate(pairs):
            stars = p_to_stars(p)
            if not stars:
                continue
            y_line = y0 + k * step
            bracket_y.append((float(xa), float(xb), y_line, stars))

    y_top_brackets = 0.0
    for _, _, y_line, _ in bracket_y:
        y_top_brackets = max(y_top_brackets, y_line + y_span * 0.35)
    y_max = max(ymax_data * 1.08, y_top_brackets, ymax_data + y_span * 0.5)
    if bracket_y:
        y_max = max(y_max, y_top_brackets + y_span * 0.2)
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 0.85

    shapes = []
    for xv in (1.5, 3.5):
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=xv,
                x1=xv,
                y0=0,
                y1=y_max,
                line=dict(color=fg, width=1, dash="dash"),
                opacity=0.45,
            )
        )

    for xa, xb, y_line, stars in bracket_y:
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=xa,
                x1=xb,
                y0=y_line,
                y1=y_line,
                line=dict(color=fg, width=1.2),
            )
        )

    annotations = []
    for xa, xb, y_line, stars in bracket_y:
        annotations.append(
            dict(
                x=(xa + xb) / 2,
                y=y_line + y_span * 0.15,
                xref="x",
                yref="y",
                text=f"<b>{stars}</b>",
                showarrow=False,
                font=dict(size=13, color=fg, family=FONT_FAMILY),
            )
        )

    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=11, color=fg),
        xaxis=dict(
            tickmode="array",
            tickvals=list(X_POS),
            ticktext=CATEGORY_LABELS,
            title=dict(
                text="model × DBS condition",
                font=dict(size=12, family=FONT_FAMILY),
            ),
            showgrid=False,
            zeroline=False,
            linecolor=fg,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=dict(
                text="RMSE (z-scored tracing speed)",
                font=dict(size=12, family=FONT_FAMILY),
            ),
            range=[0, y_max * 1.02],
            showgrid=True,
            gridcolor=grid,
            linecolor=fg,
            mirror=True,
            zeroline=True,
            zerolinecolor=grid,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        margin=dict(l=72, r=32, t=48, b=140),
        shapes=shapes,
        annotations=annotations,
        hovermode="closest",
    )

    return fig
