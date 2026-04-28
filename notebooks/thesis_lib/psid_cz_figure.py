"""Small-multiples heatmaps: PSID Cz behavioural readout matrix (nz × n1) per participant × session."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    FONT_FAMILY,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_BASE,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.loaders import load_split_results
from dashboard.thesis.psid_cy_importance import (
    compute_cz_heatmap,
    load_psid_id_sys,
    resolve_model_path,
)
from dashboard.thesis.specs import ThesisPsidCzSpec

logger = logging.getLogger(__name__)

# Diverging blue–white–red (blue = negative, red = positive)
_DIVERGING_BWR = [
    [0.0, "rgb(24, 95, 165)"],
    [0.5, "rgb(250, 250, 250)"],
    [1.0, "rgb(220, 50, 32)"],
]


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


def _load_output_channel_names(
    results_root,
    variant: str,
    run_ts: str,
    split: str,
    n_fallback: int,
) -> List[str]:
    """Try to get behavioral output channel names from saved results; fall back to z_0, z_1, ..."""
    try:
        res = load_split_results(results_root, variant, run_ts, split)
        if res is not None:
            raw = res.get("output_channels")
            if raw is not None:
                names = [str(x).replace("_", " ") for x in (list(raw) if not isinstance(raw, list) else raw)]
                if names:
                    return names[:n_fallback] if len(names) >= n_fallback else names
    except Exception:
        pass
    return [f"z_{i}" for i in range(n_fallback)]


def build_psid_cz_figure(
    spec: ThesisPsidCzSpec,
    results_root,
) -> Tuple[Figure, str]:
    """
    Build small-multiples figure of Cz (nz × n1) per session.
    Returns (figure, caption).
    """
    rows = list(spec.rows)
    if not rows:
        raise ValueError("ThesisPsidCzSpec.rows is empty")

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
        vertical_spacing=0.16,
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
                model_path = resolve_model_path(results_root, panel.psid_variant, panel.psid_run_ts)
                id_sys = load_psid_id_sys(model_path)
                cz_norm, n1, _ = compute_cz_heatmap(id_sys)
            except Exception as e:
                logger.warning("Cz panel failed: %s: %s", panel, e)
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

            nz = cz_norm.shape[0]
            y_labels = _load_output_channel_names(
                results_root, panel.psid_variant, panel.psid_run_ts, spec.split, nz
            )
            # Limit displayed latent dims to top-N by column norm
            max_display_dims = 8
            if n1 > max_display_dims:
                col_norms = np.linalg.norm(cz_norm, axis=0)
                top_idx = np.argsort(col_norms)[::-1][:max_display_dims]
                top_idx = np.sort(top_idx)
                cz_norm = cz_norm[:, top_idx]
                x_labels = [f"dim {k}" for k in top_idx]
                n1 = max_display_dims
            else:
                x_labels = [f"dim {k}" for k in range(n1)]

            hm_kw = dict(
                z=cz_norm,
                x=list(range(n1)),
                y=y_labels,
                colorscale=_DIVERGING_BWR,
                zmin=-1.0,
                zmax=1.0,
                showscale=not showscale_done,
                hovertemplate="dim %{x}<br>%{y}<br>Cz = %{z:.3f}<extra></extra>",
                xgap=2,
                ygap=2,
            )
            if not showscale_done:
                hm_kw["colorbar"] = dict(
                    title=dict(text="Norm. Cz", side="right", font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY)),
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
                tickvals=list(range(n1)),
                ticktext=x_labels,
                tickangle=-45,
                tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                title_text="Behav. rel. dim" if ri == nrows else "",
                title_font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
                automargin=True,
                row=ri,
                col=ci,
            )
            fig.update_yaxes(
                tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                automargin=True,
                row=ri,
                col=ci,
            )
            if ci > 1:
                fig.update_yaxes(showticklabels=False, row=ri, col=ci)

    apply_thesis_style(
        fig,
        theme,
        height=max(300 * nrows + 80, 500),
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
