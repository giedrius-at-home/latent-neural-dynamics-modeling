"""Figure F1: test-set balanced accuracy dot plot (three feature sources, participant colours)."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from dashboard.thesis.classification_f1_data import (
    GROUP_DISPLAY,
    GROUP_ORDER,
    GROUP_X,
    ClassificationF1Point,
    collect_classification_f1_points,
    group_star_flags,
)
from dashboard.thesis.constants import (
    FONT_FAMILY,
    PARTICIPANT_COLORS,
    ThesisTheme,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.specs import ThesisClassificationF1Spec

logger = logging.getLogger(__name__)


def build_classification_f1_figure(
    spec: ThesisClassificationF1Spec,
    results_root,
) -> Tuple[Figure, str]:
    if not spec.points:
        fig = go.Figure()
        fig.add_annotation(
            text="No points in spec — add `ClassificationF1PickleRef` rows in `THESIS_CLASSIFICATION_F1`.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, family=FONT_FAMILY),
        )
        fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
        return fig, spec.caption or ""

    try:
        points = collect_classification_f1_points(
            results_root,
            spec.points,
            classification_parent=spec.classification_parent,
        )
    except Exception as e:
        logger.exception("F1 data load failed")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Failed to load classification pickles: {e}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12, family=FONT_FAMILY),
        )
        fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
        return fig, spec.caption or ""

    theme = spec.theme
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    rng = np.random.default_rng(spec.jitter_seed)
    hw = float(spec.jitter_half_width)

    # One trace per participant (clean legend)
    by_p: Dict[str, List[Tuple[float, float, str]]] = {}
    for pt in points:
        x0 = GROUP_X[pt.group]
        j = rng.uniform(-hw, hw)
        by_p.setdefault(pt.participant_label, []).append(
            (x0 + j, pt.balanced_accuracy, pt.session_label)
        )

    fig = go.Figure()

    # Median lines per group (behind points)
    for g in GROUP_ORDER:
        vals = [p.balanced_accuracy for p in points if p.group == g]
        if not vals:
            continue
        med = float(np.median(vals))
        x0 = GROUP_X[g]
        fig.add_trace(
            go.Scatter(
                x=[x0 - 0.32, x0 + 0.32],
                y=[med, med],
                mode="lines",
                line=dict(color="rgba(200,200,200,0.95)", width=4),
                name="median",
                legendgroup="median",
                showlegend=(g == "psid_xp"),
                hoverinfo="skip",
            )
        )

    def _p_order(s: str) -> int:
        if len(s) > 1 and s[0] == "P" and s[1:].isdigit():
            return int(s[1:])
        return 99

    for plab in sorted(by_p.keys(), key=_p_order):
        col = PARTICIPANT_COLORS.get(plab, "#888888")
        xs = [t[0] for t in by_p[plab]]
        ys = [t[1] for t in by_p[plab]]
        sess = [t[2] for t in by_p[plab]]
        cd = [[plab, s] for s in sess]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=11, color=col, line=dict(width=0)),
                name=plab,
                customdata=cd,
                hovertemplate="%{customdata[0]}<br>session %{customdata[1]}<br>BA=%{y:.3f}<extra></extra>",
            )
        )

    # Chance line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="rgba(220, 60, 60, 0.9)",
        line_width=1.5,
        layer="below",
    )
    fig.add_annotation(
        text="chance",
        xref="x",
        x=2.38,
        y=0.5,
        yref="y",
        xanchor="left",
        showarrow=False,
        font=dict(size=10, color="rgba(220, 60, 60, 0.95)", family=FONT_FAMILY),
    )

    # Vertical separators between groups
    for xv in (0.5, 1.5):
        fig.add_vline(
            x=xv,
            line_dash="dash",
            line_color=grid,
            line_width=1,
            layer="below",
        )

    # Permutation stars above medians
    flags = group_star_flags(points, spec.permutation_alpha)
    y_hi = float(spec.y_max)
    for g in GROUP_ORDER:
        if not flags.get(g):
            continue
        vals = [p.balanced_accuracy for p in points if p.group == g]
        if not vals:
            continue
        med = float(np.median(vals))
        x0 = GROUP_X[g]
        fig.add_annotation(
            x=x0,
            y=min(med + 0.06, y_hi - 0.02),
            text="*",
            showarrow=False,
            font=dict(size=18, color=fg, family=FONT_FAMILY),
            yref="y",
        )

    tickvals = [GROUP_X[g] for g in GROUP_ORDER]
    ticktext = [GROUP_DISPLAY[g] for g in GROUP_ORDER]

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg, size=12),
        title=dict(text=spec.section_title, font=dict(size=15)),
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=(-0.45, 2.45),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Balanced accuracy",
            range=[spec.y_min, spec.y_max],
            gridcolor=grid,
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=70, r=70, t=70, b=120),
        hovermode="closest",
    )

    cap = spec.caption or ""
    return fig, cap
