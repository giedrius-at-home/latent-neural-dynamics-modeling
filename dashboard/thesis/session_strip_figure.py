"""B2: Multi-panel session-mean RMSE strip plots (Plotly)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
from dashboard.thesis.session_strip_rmse import StripFigureData

# Fixed x layout (per mockup)
X_CENTERS = (0.0, 0.5, 1.4, 1.9, 2.8, 3.3)
V_SEP = (1.15, 2.55)
X_RANGE = (-0.25, 3.55)

CELL_COLORS = (COLOR_PSID, COLOR_PSID, COLOR_DPAD, COLOR_DPAD, COLOR_VARMA, COLOR_VARMA)
# cells 0,2,4 = OFF (circle); 1,3,5 = ON (square)
BOTTOM_TICK_TEXT = (
    "PSID<br>OFF",
    "PSID<br>ON",
    "DPAD<br>OFF",
    "DPAD<br>ON",
    "VARMA<br>OFF",
    "VARMA<br>ON",
)


def _cell_symbol(cell_idx: int) -> str:
    return "circle" if cell_idx % 2 == 0 else "square"


def build_session_strip_figure(
    data: StripFigureData,
    ncols: int,
    theme: ThesisTheme,
    rng: np.random.Generator,
    jitter: float = 0.05,
) -> go.Figure:
    n_panels = len(data.panels)
    nrows = int(np.ceil(n_panels / ncols)) if ncols else 1
    ncols = max(1, ncols)

    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_y=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    for pi, panel in enumerate(data.panels):
        row = pi // ncols + 1
        col = pi % ncols + 1
        show_leg = pi == 0

        for cell_idx in range(6):
            xc = X_CENTERS[cell_idx]
            color = CELL_COLORS[cell_idx]
            my = panel.mean_line_y[cell_idx]
            vals = panel.session_means[cell_idx]

            if np.isfinite(my):
                fig.add_trace(
                    go.Scatter(
                        x=[xc - 0.3, xc + 0.3],
                        y=[my, my],
                        mode="lines",
                        line=dict(color=color, width=2.5),
                        name="Participant mean (session avg.)" if show_leg and cell_idx == 0 else None,
                        legendgroup="meanline",
                        showlegend=show_leg and cell_idx == 0,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            if not vals:
                continue
            jt = rng.uniform(-jitter, jitter, size=len(vals))
            fig.add_trace(
                go.Scatter(
                    x=xc + jt,
                    y=vals,
                    mode="markers",
                    marker=dict(
                        size=7,
                        color=color,
                        symbol=_cell_symbol(cell_idx),
                        line=dict(width=0),
                    ),
                    name="Session mean RMSE" if show_leg and cell_idx == 0 else None,
                    legendgroup="sessions",
                    showlegend=show_leg and cell_idx == 0,
                ),
                row=row,
                col=col,
            )

        fig.add_annotation(
            row=row,
            col=col,
            xref="x domain",
            yref="y domain",
            x=0.04,
            y=0.93,
            xanchor="left",
            yanchor="top",
            text=f"<b>{panel.panel_label}</b>",
            showarrow=False,
            font=dict(size=11, family=FONT_FAMILY, color=fg),
        )

        for xv in V_SEP:
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=xv,
                x1=xv,
                y0=0,
                y1=data.y_max,
                line=dict(color=fg, width=1, dash="dash"),
                opacity=0.4,
                row=row,
                col=col,
            )

    c_title = max(1, (ncols + 1) // 2)

    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            idx = (r - 1) * ncols + (c - 1)
            if idx >= n_panels:
                fig.update_xaxes(visible=False, row=r, col=c)
                fig.update_yaxes(visible=False, row=r, col=c)
                continue

            fig.update_yaxes(
                range=[0, data.y_max],
                showgrid=True,
                gridcolor=grid,
                linecolor=fg,
                mirror=True,
                row=r,
                col=c,
            )
            fig.update_xaxes(
                range=list(X_RANGE),
                showgrid=False,
                zeroline=False,
                linecolor=fg,
                mirror=True,
                row=r,
                col=c,
            )
            if c == 1:
                fig.update_yaxes(
                    title_text="RMSE (z-scored tracing speed)",
                    title_font=dict(size=11, family=FONT_FAMILY),
                    row=r,
                    col=c,
                )
            else:
                fig.update_yaxes(title_text="", row=r, col=c)

            if r < nrows:
                fig.update_xaxes(showticklabels=False, row=r, col=c)
            else:
                xt = "model × DBS condition" if c == c_title else ""
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=list(X_CENTERS),
                    ticktext=list(BOTTOM_TICK_TEXT),
                    tickangle=0,
                    tickfont=dict(size=8),
                    title_text=xt,
                    title_font=dict(size=10, family=FONT_FAMILY),
                    row=r,
                    col=c,
                )

    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=11, color=fg),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        margin=dict(l=56, r=24, t=40, b=100),
        hovermode="closest",
    )

    return fig
