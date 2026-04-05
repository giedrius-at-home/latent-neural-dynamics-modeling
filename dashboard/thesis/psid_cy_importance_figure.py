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
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.psid_cy_importance import compute_cy_signed_heatmap
from dashboard.thesis.specs import ThesisPsidCyImportanceSpec

logger = logging.getLogger(__name__)

# Diverging blue–white–red (matches Cz figure)
_DIVERGING_BWR = [
    [0.0, "rgb(24, 95, 165)"],
    [0.5, "rgb(250, 250, 250)"],
    [1.0, "rgb(220, 50, 32)"],
]

_X_LABELS = ["ECoG_1", "ECoG_2", "ECoG_3", "ECoG_4"]
_X_IDX = list(range(4))


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
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
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
                    font=dict(size=FONT_SIZE_ANNOTATION, color=fg, family=FONT_FAMILY),
                    row=ri,
                    col=ci,
                )
                continue

            z = np.asarray(z, dtype=float)
            # z shape: (n1, 4) — rows=latent dims, cols=ECoG contacts
            # Limit to top dims by total norm to avoid unreadable 80-row heatmaps
            max_display_dims = 8
            if z.shape[0] > max_display_dims:
                row_norms = np.linalg.norm(z, axis=1)
                top_idx = np.argsort(row_norms)[::-1][:max_display_dims]
                top_idx = np.sort(top_idx)  # keep original order
                z = z[top_idx]
                y_labels = [f"dim {i}" for i in top_idx]
            else:
                y_labels = [f"dim {i}" for i in range(z.shape[0])]
            n1_dims = z.shape[0]
            hm_kw = dict(
                z=z,
                x=_X_IDX,
                y=y_labels,
                colorscale=_DIVERGING_BWR,
                zmin=-1.0,
                zmax=1.0,
                showscale=not showscale_done,
                hovertemplate="Latent %{y}<br>%{x}<br>|Cy| = %{z:.3f}<extra></extra>",
                xgap=2,
                ygap=2,
            )
            if not showscale_done:
                hm_kw["colorbar"] = dict(
                    title=dict(text="Norm. |Cy|", side="right", font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY)),
                    len=0.55,
                    thickness=14,
                    tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                )
            hm = go.Heatmap(**hm_kw)
            fig.add_trace(hm, row=ri, col=ci)
            showscale_done = True

            fig.update_xaxes(
                tickmode="array",
                tickvals=_X_IDX,
                ticktext=_X_LABELS,
                tickangle=-30,
                tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                row=ri,
                col=ci,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(n1_dims)),
                ticktext=y_labels,
                tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                title_text="Behav. rel. dim" if ci == 1 else "",
                title_font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
                row=ri,
                col=ci,
            )
            if ci > 1:
                fig.update_yaxes(showticklabels=False, row=ri, col=ci)

    apply_thesis_style(
        fig,
        theme,
        height=max(280 * nrows + 80, 500),
        margin=dict(l=100, r=110, t=56, b=96),
        show_legend=False,
    )

    for ri in range(1, nrows + 1):
        for ci in range(1, max_cols + 1):
            fig.update_xaxes(
                zeroline=False,
                showgrid=False,
                row=ri,
                col=ci,
            )
            fig.update_yaxes(
                zeroline=False,
                showgrid=False,
                row=ri,
                col=ci,
            )

    cap = spec.caption or ""
    return fig, cap
