"""Dual-panel heatmaps (DBS-OFF / DBS-ON): bands × models, shared Pearson r colorscale."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_DPAD,
    COLOR_PSID,
    COLOR_VARMA,
    FIGURE_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_TICK,
    ThesisTheme,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.neural_band_pearson import NeuralBandHeatmapData

# Light grey → strong blue (sequential); matches thesis PSID hue at high end
_BLUE_SEQUENTIAL = [
    [0.0, "rgb(250, 250, 250)"],
    [0.35, "rgb(200, 215, 235)"],
    [0.65, "rgb(100, 150, 205)"],
    [1.0, "rgb(24, 95, 165)"],
]

_ZMIN = 0.45
_ZMAX = 1.0


def _fmt_text(z: np.ndarray) -> list[list[str]]:
    out: list[list[str]] = []
    for row in z:
        out.append([f"{v:.2f}" if np.isfinite(v) else "" for v in row])
    return out


def build_neural_band_heatmap_figure(
    data: NeuralBandHeatmapData,
    theme: ThesisTheme,
) -> Figure:
    """Two heatmaps sharing `coloraxis` (z clipped to [0.4, 1.0])."""
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    x_labels = list(data.column_labels)
    row_labels = list(data.band_labels)
    nrows = len(row_labels)
    ncols = len(x_labels)

    if nrows == 0 or ncols == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No band data (check aligned PSID/DPAD/VARMA neural results).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=fg, family=FONT_FAMILY, size=14),
        )
        fig.update_layout(
            paper_bgcolor=paper_bg,
            plot_bgcolor=plot_bg,
            font=dict(family=FONT_FAMILY, color=fg),
            margin=dict(l=60, r=40, t=80, b=100),
        )
        return fig

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.12,
    )

    z_off = np.asarray(data.z_off, dtype=float)
    z_on = np.asarray(data.z_on, dtype=float)

    for col, (z_mat, title_badge) in enumerate(
        [(z_off, "DBS-OFF"), (z_on, "DBS-ON")], start=1
    ):
        text_m = _fmt_text(z_mat)
        hm = go.Heatmap(
            z=z_mat,
            x=x_labels,
            y=row_labels,
            text=text_m,
            texttemplate="%{text}",
            textfont=dict(size=FONT_SIZE_BASE, family=FONT_FAMILY, color="white"),
            coloraxis="coloraxis",
            showscale=False,
            hovertemplate="%{y} · %{x}<br>r = %{z:.3f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
        fig.add_trace(hm, row=1, col=col)

        xref = "x" if col == 1 else "x2"
        yref = "y" if col == 1 else "y2"
        header_colors = (COLOR_PSID, COLOR_DPAD, COLOR_VARMA)
        for j, (lab, hc) in enumerate(zip(x_labels, header_colors)):
            fig.add_annotation(
                x=x_labels[j],
                y=1.02,
                xref=xref,
                yref=f"{yref} domain",
                text=f"<b>{lab}</b>",
                showarrow=False,
                font=dict(color=hc, size=FONT_SIZE_BASE, family=FONT_FAMILY),
                xanchor="center",
                yanchor="bottom",
            )

        fig.add_annotation(
            x=0.0,
            y=1.12,
            xref=xref,
            yref=f"{yref} domain",
            text=f"<b>{title_badge}</b>",
            showarrow=False,
            font=dict(
                color=COLOR_DBS_OFF if title_badge == "DBS-OFF" else COLOR_DBS_ON,
                size=FONT_SIZE_BASE,
                family=FONT_FAMILY,
            ),
            xanchor="left",
            yanchor="top",
        )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=FIGURE_HEIGHT - 50,
        margin=dict(l=80, r=40, t=50, b=100),
        coloraxis=dict(
            cmin=_ZMIN,
            cmax=_ZMAX,
            colorscale=_BLUE_SEQUENTIAL,
            colorbar=dict(
                title=dict(text="Pearson r", font=dict(size=12)),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=-0.18,
                yanchor="top",
                len=0.55,
                thickness=16,
                tickmode="array",
                tickvals=[0.4, 0.6, 0.8, 1.0],
                tickfont=dict(color=fg),
            ),
        ),
        showlegend=False,
    )

    for i in (1, 2):
        fig.update_xaxes(
            row=1,
            col=i,
            showticklabels=False,
            side="bottom",
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            row=1,
            col=i,
            autorange="reversed",
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showticklabels=(i == 1),
        )

    return fig
