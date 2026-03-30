"""B2: Multi-panel session-mean RMSE strip plots and per-participant box plots (Plotly)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_DPAD,
    COLOR_PSID,
    COLOR_SEPARATOR,
    COLOR_VARMA,
    DOT_SIZE,
    FIGURE_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    WIDTH_MEAN,
    ThesisTheme,
    grid_color,
    legend_bgcolor,
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
        shared_yaxes=True,
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
                        line=dict(color="black", width=WIDTH_MEAN),
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
                        size=5,
                        color=color,
                        symbol=_cell_symbol(cell_idx),
                        line=dict(width=1.2, color=color),
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
            font=dict(size=FONT_SIZE_BASE, family=FONT_FAMILY, color=fg),
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
                line=dict(color=COLOR_SEPARATOR, width=0.7, dash="dash"),
                opacity=0.5,
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
                showline=True,
                linecolor=fg,
                linewidth=1,
                mirror=False,
                tickfont=dict(size=FONT_SIZE_TICK),
                row=r,
                col=c,
            )
            fig.update_xaxes(
                range=list(X_RANGE),
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor=fg,
                linewidth=1,
                mirror=False,
                row=r,
                col=c,
            )
            if c == 1:
                fig.update_yaxes(
                    title_text="RMSE (z-scored tracing speed)",
                    title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
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
                    tickangle=-32,
                    tickfont=dict(size=FONT_SIZE_TICK - 1),
                    title_text=xt,
                    title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                    title_standoff=12,
                    automargin=True,
                    row=r,
                    col=c,
                )

    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=FIGURE_HEIGHT,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZE_TICK),
            bgcolor=legend_bgcolor(),
        ),
        margin=dict(l=56, r=24, t=36, b=120),
        hovermode="closest",
    )

    return fig


def build_session_strip_boxplot_figure(
    data: StripFigureData,
    ncols: int,
    theme: ThesisTheme,
    rng: np.random.Generator,
    jitter: float = 0.10,
) -> go.Figure:
    """Per-participant box plots (trial-level RMSE) by model × DBS condition."""
    n_panels = len(data.panels)
    nrows = int(np.ceil(n_panels / max(1, ncols)))
    ncols = max(1, ncols)

    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
    )

    models = ["PSID", "DPAD", "VARMA"]
    model_colors = [COLOR_PSID, COLOR_DPAD, COLOR_VARMA]
    # cell order: 0=PSID-OFF, 1=PSID-ON, 2=DPAD-OFF, 3=DPAD-ON, 4=VARMA-OFF, 5=VARMA-ON
    _OFF_CELLS = [0, 2, 4]
    _ON_CELLS = [1, 3, 5]

    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    for pi, panel in enumerate(data.panels):
        row = pi // ncols + 1
        col = pi % ncols + 1
        show_leg = pi == 0

        for mi, (model, mc, off_cell, on_cell) in enumerate(
            zip(models, model_colors, _OFF_CELLS, _ON_CELLS)
        ):
            for cond_label, cell_idx, alpha in (("OFF", off_cell, 0.80), ("ON", on_cell, 0.45)):
                vals = panel.trial_rmse[cell_idx]
                if not vals:
                    continue
                arr = [v for v in vals if np.isfinite(v)]
                if not arr:
                    continue
                xpos = mi * 2 + (0 if cond_label == "OFF" else 1)
                leg_name = f"{model} {cond_label}"
                fig.add_trace(
                    go.Box(
                        y=arr,
                        x=[xpos] * len(arr),
                        name=leg_name,
                        marker_color=mc,
                        line=dict(color=mc, width=1.2),
                        fillcolor=_hex_to_rgba(mc, alpha * 0.35),
                        boxpoints=False,
                        quartilemethod="exclusive",
                        showlegend=show_leg,
                        legendgroup=leg_name,
                        width=0.7,
                    ),
                    row=row,
                    col=col,
                )
                jt = rng.uniform(-jitter * 0.4, jitter * 0.4, size=len(arr))
                fig.add_trace(
                    go.Scatter(
                        x=[xpos + jt[i] for i in range(len(arr))],
                        y=arr,
                        mode="markers",
                        marker=dict(size=5, color=_hex_to_rgba(mc, alpha * 0.85), line=dict(width=0)),
                        showlegend=False,
                        legendgroup=leg_name,
                        hovertemplate=f"{leg_name}<br>%{{y:.3f}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

        fig.add_annotation(
            row=row, col=col,
            xref="x domain", yref="y domain",
            x=0.04, y=0.93,
            xanchor="left", yanchor="top",
            text=f"<b>{panel.panel_label}</b>",
            showarrow=False,
            font=dict(size=FONT_SIZE_BASE, family=FONT_FAMILY, color=fg),
        )

        for xv in (1.5, 3.5):
            fig.add_shape(
                type="line", xref="x", yref="y",
                x0=xv, x1=xv, y0=0, y1=data.y_max,
                line=dict(color=COLOR_SEPARATOR, width=0.7, dash="dash"),
                opacity=0.5, row=row, col=col,
            )

    x_tick_vals = [0, 1, 2, 3, 4, 5]
    x_tick_text = [
        "PSID\nOFF", "PSID\nON",
        "DPAD\nOFF", "DPAD\nON",
        "VARMA\nOFF", "VARMA\nON",
    ]
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
                showgrid=True, gridcolor=grid,
                showline=True, linecolor=fg, linewidth=1,
                tickfont=dict(size=FONT_SIZE_TICK),
                row=r, col=c,
            )
            if c == 1:
                fig.update_yaxes(
                    title_text="RMSE (z-scored tracing speed)",
                    title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                    row=r, col=c,
                )
            if r < nrows:
                fig.update_xaxes(showticklabels=False, row=r, col=c)
            else:
                xt = "model × DBS condition" if c == c_title else ""
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=x_tick_vals,
                    ticktext=x_tick_text,
                    tickangle=-32,
                    tickfont=dict(size=FONT_SIZE_TICK - 1),
                    title_text=xt,
                    title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                    automargin=True,
                    row=r, col=c,
                )

    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=FIGURE_HEIGHT,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.18,
            xanchor="center", x=0.5,
            font=dict(size=FONT_SIZE_TICK), bgcolor=legend_bgcolor(),
        ),
        margin=dict(l=56, r=24, t=36, b=120),
        hovermode="closest",
        boxmode="overlay",
    )

    return fig
