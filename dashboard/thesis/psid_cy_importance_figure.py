"""Small-multiples heatmaps: PSID Cy_rel importance (4×29) per participant × session."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    FONT_FAMILY,
    ThesisTheme,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.psid_cy_importance import compute_panel
from dashboard.thesis.specs import ThesisPsidCyImportanceSpec

logger = logging.getLogger(__name__)

# Sequential blue (matches neural band figure)
_BLUE_SEQUENTIAL = [
    [0.0, "rgb(250, 250, 250)"],
    [0.35, "rgb(200, 215, 235)"],
    [0.65, "rgb(100, 150, 205)"],
    [1.0, "rgb(24, 95, 165)"],
]

_Y_LABELS = ["ECoG 1", "ECoG 2", "ECoG 3", "ECoG 4"]
_X = list(range(29))


def _band_symbol(band: str) -> str:
    return {"Delta": "δ", "Theta": "θ", "Alpha": "α", "Beta": "β"}.get(band, band)


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
                imp, _n1, _ch, layout = compute_panel(
                    results_root,
                    panel.psid_variant,
                    panel.psid_run_ts,
                    spec.split,
                )
            except Exception as e:
                logger.exception("Panel failed: %s", panel)
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

            z = np.asarray(imp, dtype=float)
            hm = go.Heatmap(
                z=z,
                x=_X,
                y=_Y_LABELS,
                colorscale=_BLUE_SEQUENTIAL,
                zmin=0.0,
                zmax=1.0,
                showscale=not showscale_done,
                colorbar=dict(
                    title=dict(text="Norm. ‖Cy_rel‖₂", side="right"),
                    len=0.55,
                    thickness=14,
                    tickfont=dict(size=10, family=FONT_FAMILY),
                    titlefont=dict(size=11, family=FONT_FAMILY),
                )
                if not showscale_done
                else False,
                hovertemplate="%{y} · col %{x}<br>importance = %{z:.3f}<extra></extra>",
                xgap=1,
                ygap=1,
            )
            fig.add_trace(hm, row=ri, col=ci)
            showscale_done = True

            if spec.show_beta_box and layout.beta_col_end > layout.beta_col_start:
                fig.add_shape(
                    type="rect",
                    x0=layout.beta_col_start,
                    x1=layout.beta_col_end,
                    y0=-0.5,
                    y1=3.5,
                    line=dict(color="rgba(200, 60, 60, 0.95)", width=2, dash="dash"),
                    fillcolor="rgba(0,0,0,0)",
                    layer="above",
                    row=ri,
                    col=ci,
                )

            if ci > 1:
                fig.update_yaxes(showticklabels=False, row=ri, col=ci)

            if ri < nrows:
                fig.update_xaxes(showticklabels=False, row=ri, col=ci)

            if ri == nrows and layout.spans:
                tickvals = [0.5 * (lo + hi) for (_b, lo, hi) in layout.spans]
                ticktext = [_band_symbol(b) for (b, _lo, _hi) in layout.spans]
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickfont=dict(size=12, family=FONT_FAMILY, color=fg),
                    row=ri,
                    col=ci,
                )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg, size=11),
        margin=dict(l=100, r=110, t=70, b=70),
        title=dict(text=spec.section_title, font=dict(size=15)),
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
