"""Small-multiples heatmaps: PSID Cy_rel importance (4×29) per participant × session."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_BETA_BORDER,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_TICK,
    ThesisTheme,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.psid_cy_importance import compute_cy_signed_heatmap, cy_heatmap_x_tick_labels
from dashboard.thesis.specs import ThesisPsidCyImportanceSpec

logger = logging.getLogger(__name__)

# Diverging blue–white–red (matches Cz figure)
_DIVERGING_BWR = [
    [0.0, "rgb(24, 95, 165)"],
    [0.5, "rgb(250, 250, 250)"],
    [1.0, "rgb(220, 50, 32)"],
]

_Y_LABELS = ["ECoG 1", "ECoG 2", "ECoG 3", "ECoG 4"]
_X_IDX = list(range(29))


def _empty_placeholder(fig: Figure, row: int, col: int) -> None:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, row=row, col=col)


def build_psid_cy_importance_figure(
    spec: ThesisPsidCyImportanceSpec,
    results_root,
) -> Tuple[Figure, str]:
    """
    Build small-multiples figure. Returns (figure, caption).

    Ragged grid: columns = max sessions across rows; unused cells are blank.
    """
    rows = list(spec.rows)
    if not rows:
        raise ValueError("ThesisPsidCyImportanceSpec.rows is empty")

    max_cols = max(len(r.panels) for r in rows)
    nrows = len(rows)
    theme = spec.theme
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    subplot_titles: List[str | None] = []
    for r in rows:
        for c in range(max_cols):
            if c < len(r.panels):
                subplot_titles.append(r.panels[c].session_label)
            else:
                subplot_titles.append("")

    fig = make_subplots(
        rows=nrows,
        cols=max_cols,
        subplot_titles=subplot_titles or None,
        horizontal_spacing=0.06,
        vertical_spacing=0.09,
        row_titles=[rr.participant_label for rr in rows],
    )

    showscale_done = False

    for ri, row_spec in enumerate(rows, start=1):
        for ci in range(1, max_cols + 1):
            if ci > len(row_spec.panels):
                _empty_placeholder(fig, ri, ci)
                continue

            panel = row_spec.panels[ci - 1]
            try:
                z, _n1, _ch, layout = compute_cy_signed_heatmap(
                    results_root,
                    panel.psid_variant,
                    panel.psid_run_ts,
                    spec.split,
                )
            except Exception as e:
                logger.warning("Panel failed: %s: %s", panel, e)
                fig.add_annotation(
                    text=f"Error: {e}",
                    x=0.5,
                    y=0.5,
                    xref="x domain",
                    yref="y domain",
                    showarrow=False,
                    font=dict(size=10, color=fg, family=FONT_FAMILY),
                    row=ri,
                    col=ci,
                )
                continue

            z = np.asarray(z, dtype=float)
            x_lab = cy_heatmap_x_tick_labels(_ch)
            if len(x_lab) < z.shape[1]:
                x_lab = [f"col_{j}" for j in range(z.shape[1])]
            else:
                x_lab = x_lab[: z.shape[1]]
            hm_kw = dict(
                z=z,
                x=_X_IDX,
                y=_Y_LABELS,
                colorscale=_DIVERGING_BWR,
                zmin=-1.0,
                zmax=1.0,
                showscale=not showscale_done,
                hovertemplate="%{y}<br>col %{x}<br>Cy (signed) = %{z:.3f}<extra></extra>",
                xgap=1,
                ygap=1,
            )
            if not showscale_done:
                hm_kw["colorbar"] = dict(
                    title=dict(text="Norm. Cy (signed)", side="right", font=dict(size=11, family=FONT_FAMILY)),
                    len=0.55,
                    thickness=14,
                    tickfont=dict(size=10, family=FONT_FAMILY),
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                )
            hm = go.Heatmap(**hm_kw)
            fig.add_trace(hm, row=ri, col=ci)
            showscale_done = True

            if spec.show_beta_box and layout.beta_col_end > layout.beta_col_start:
                fig.add_shape(
                    type="rect",
                    x0=layout.beta_col_start,
                    x1=layout.beta_col_end,
                    y0=-0.5,
                    y1=3.5,
                    line=dict(color=COLOR_BETA_BORDER, width=1.5, dash="dash"),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                    row=ri,
                    col=ci,
                )

            if ci > 1:
                fig.update_yaxes(showticklabels=False, row=ri, col=ci)

            if ri < nrows:
                fig.update_xaxes(showticklabels=False, row=ri, col=ci)

            if ri == nrows:
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=_X_IDX,
                    ticktext=x_lab,
                    tickangle=-65,
                    tickfont=dict(size=7, family=FONT_FAMILY, color=fg),
                    automargin=True,
                    row=ri,
                    col=ci,
                )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg, size=FONT_SIZE_BASE),
        margin=dict(l=100, r=110, t=50, b=96),
    )

    for ri in range(1, nrows + 1):
        for ci in range(1, max_cols + 1):
            fig.update_xaxes(
                gridcolor=grid,
                zeroline=False,
                showgrid=False,
                row=ri,
                col=ci,
            )
            fig.update_yaxes(
                gridcolor=grid,
                zeroline=False,
                showgrid=False,
                row=ri,
                col=ci,
            )

    cap = spec.caption or ""
    return fig, cap
