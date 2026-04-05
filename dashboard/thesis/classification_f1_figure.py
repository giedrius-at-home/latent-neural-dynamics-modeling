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
    COLOR_CHANCE,
    COLOR_SEPARATOR,
    FIGURE_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    PARTICIPANT_COLORS,
    WIDTH_MEAN,
    ThesisTheme,
    grid_color,
    legend_bgcolor,
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
            font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
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
            font=dict(size=FONT_SIZE_BASE, family=FONT_FAMILY),
        )
        fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
        return fig, spec.caption or ""

    theme = spec.theme
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    rng = np.random.default_rng(spec.jitter_seed)
    hw = max(float(spec.jitter_half_width), 0.15)

    # One trace per participant (clean legend)
    by_p: Dict[str, List[Tuple[float, float, str]]] = {}
    for pt in points:
        x0 = GROUP_X[pt.group]
        j = rng.uniform(-hw, hw)
        by_p.setdefault(pt.participant_label, []).append(
            (x0 + j, pt.balanced_accuracy, pt.session_label)
        )

    fig = go.Figure()

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
                marker=dict(size=5, color=col, opacity=0.7, line=dict(width=0)),
                name=plab,
                customdata=cd,
                hovertemplate="%{customdata[0]}<br>session %{customdata[1]}<br>BA=%{y:.3f}<extra></extra>",
            )
        )

    # Chance line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=COLOR_CHANCE,
        line_width=1.2,
        layer="below",
    )
    fig.add_annotation(
        text="chance",
        xref="x",
        x=3.38,
        y=0.5,
        yref="y",
        xanchor="left",
        showarrow=False,
        font=dict(size=FONT_SIZE_TICK, color=COLOR_CHANCE, family=FONT_FAMILY),
    )

    for xv in (0.5, 1.5, 2.5):
        fig.add_vline(
            x=xv,
            line_dash="dash",
            line_color=COLOR_SEPARATOR,
            line_width=0.7,
            opacity=0.5,
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
        font=dict(family=FONT_FAMILY, color=fg, size=FONT_SIZE_BASE),
        height=FIGURE_HEIGHT,
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            range=(-0.45, 3.45),
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=fg,
            linewidth=1,
            mirror=False,
            tickfont=dict(size=FONT_SIZE_TICK),
        ),
        yaxis=dict(
            title=dict(
                text="Balanced accuracy",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            ),
            range=[spec.y_min, spec.y_max],
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showline=True,
            linecolor=fg,
            linewidth=1,
            mirror=False,
            tickfont=dict(size=FONT_SIZE_TICK),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZE_TICK),
            bgcolor=legend_bgcolor(),
        ),
        margin=dict(l=70, r=70, t=36, b=80),
        hovermode="closest",
    )

    cap = spec.caption or ""
    return fig, cap


# ---------------------------------------------------------------------------
# Grouped bar chart: one cluster per session, bars = feature groups
# ---------------------------------------------------------------------------

# Feature-group colours (consistent, distinct from model colours)
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


def build_classification_grouped_bar_figure(
    points: List[ClassificationF1Point],
    theme: ThesisTheme = ThesisTheme.LIGHT,
    title: str | None = None,
    exclude_groups: set[str] | None = None,
) -> Figure:
    """
    Grouped bar chart: one cluster per participant-session, bars per feature group.
    Returns (figure, caption_rows) where caption_rows lists per-session stats.
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    active_groups = [g for g in GROUP_ORDER if not (exclude_groups and g in exclude_groups)]

    # Build session order from points
    seen: dict[str, None] = {}
    for pt in points:
        key = f"{pt.participant_label}_{pt.session_label}"
        seen.setdefault(key, None)
    session_labels = list(seen.keys())

    # Build lookup: (session_label, group) -> point
    lookup: dict[tuple[str, str], ClassificationF1Point] = {}
    for pt in points:
        key = f"{pt.participant_label}_{pt.session_label}"
        lookup[(key, pt.group)] = pt

    n_sessions = len(session_labels)
    n_groups = len(active_groups)
    bar_width = 0.18
    cluster_width = n_groups * bar_width + 0.1

    fig = go.Figure()

    for gi, grp in enumerate(active_groups):
        xs: list[float] = []
        ys: list[float] = []
        hover: list[str] = []
        for si, sess in enumerate(session_labels):
            x_center = si * (cluster_width + 0.3)
            x_bar = x_center + (gi - (n_groups - 1) / 2) * bar_width
            xs.append(x_bar)
            pt = lookup.get((sess, grp))
            ba = pt.balanced_accuracy if pt else float("nan")
            ys.append(ba)
            hover.append(f"{sess}<br>{_FEAT_SHORT[grp]}<br>BA={ba:.3f}")

        fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                width=bar_width * 0.9,
                name=_FEAT_SHORT[grp],
                marker=dict(color=_FEAT_COLORS.get(grp, "#888888")),
                hovertext=hover,
                hoverinfo="text",
                legendgroup=grp,
                showlegend=True,
            )
        )

    # Chance line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=COLOR_CHANCE,
        line_width=1.2,
        layer="below",
    )

    # X-axis tick positions and labels
    tick_xs = [si * (cluster_width + 0.3) for si in range(n_sessions)]

    # Vertical separators between clusters
    for si in range(1, n_sessions):
        xv = (tick_xs[si - 1] + tick_xs[si]) / 2
        fig.add_vline(
            x=xv,
            line_dash="dash",
            line_color=COLOR_SEPARATOR,
            line_width=0.7,
            opacity=0.4,
            layer="below",
        )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg, size=FONT_SIZE_BASE),
        height=FIGURE_HEIGHT,
        barmode="group",
        xaxis=dict(
            tickmode="array",
            tickvals=tick_xs,
            ticktext=session_labels,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=fg,
            linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
            title=dict(
                text="Participant × Session",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                standoff=10,
            ),
        ),
        yaxis=dict(
            title=dict(
                text="Balanced accuracy",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            ),
            range=[0.0, 1.08],
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showline=True,
            linecolor=fg,
            linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZE_TICK),
            bgcolor=legend_bgcolor(),
        ),
        margin=dict(l=70, r=40, t=48, b=100),
        hovermode="closest",
    )

    return fig


# ---------------------------------------------------------------------------
# Standard classification heatmap: sessions × feature groups
# ---------------------------------------------------------------------------


def build_standard_heatmap_figure(
    points: List[ClassificationF1Point],
    theme: ThesisTheme = ThesisTheme.LIGHT,
) -> Figure:
    """Sessions (rows) × feature groups (cols) balanced-accuracy heatmap."""
    from plotly.subplots import make_subplots
    from dashboard.thesis.constants import apply_thesis_style

    paper_bg, plot_bg = paper_colors(theme)
    fg = true_line_color(theme)

    # Build session × group grid
    seen: dict[str, None] = {}
    for pt in points:
        key = f"{pt.participant_label}_{pt.session_label}"
        seen.setdefault(key, None)
    session_labels = list(seen.keys())
    feat_keys = list(GROUP_ORDER)
    n_rows = len(session_labels)
    n_cols = len(feat_keys)

    grid = np.full((n_rows, n_cols), float("nan"))
    for pt in points:
        sess_key = f"{pt.participant_label}_{pt.session_label}"
        ri = session_labels.index(sess_key) if sess_key in session_labels else -1
        ci = feat_keys.index(pt.group) if pt.group in feat_keys else -1
        if ri >= 0 and ci >= 0:
            grid[ri, ci] = pt.balanced_accuracy

    text_vals = [
        [f"{v:.2f}" if np.isfinite(v) else "" for v in row] for row in grid
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=[_FEAT_SHORT.get(f, f) for f in feat_keys],
            y=session_labels,
            colorscale="RdYlGn",
            zmin=0.3,
            zmax=0.9,
            zmid=0.5,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=FONT_SIZE_ANNOTATION),
            showscale=True,
            colorbar=dict(
                title=dict(text="BA", side="right"),
                len=0.5,
                thickness=12,
                tickfont=dict(size=FONT_SIZE_TICK),
            ),
            hovertemplate="Session: %{y}<br>Feature: %{x}<br>BA=%{z:.3f}<extra></extra>",
        )
    )

    apply_thesis_style(fig, theme, height=360, margin=dict(l=100, r=80, t=36, b=60))
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Flipped classification heatmaps: h × m grid per session × feature
# ---------------------------------------------------------------------------

# Mapping from session label to flipped variant base + per-feature run timestamps
_FLIPPED_SESSIONS: dict[str, tuple[str, dict[str, str]]] = {
    "PDI1_S2": ("psid_behavioral_PDI1_2_nx_80_n12_i40_dbs_both_narrow_band_flipped", {
        "xp": "20260315_210934",
        "xp_1": "20260315_212227",
        "xp_2": "20260315_212605",
        "xp_with_dbs": "20260315_213629",
    }),
    "PDI1_S4": ("psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band_flipped", {
        "xp": "20260315_215824",
        "xp_1": "20260315_220623",
        "xp_2": "20260315_220842",
        "xp_with_dbs": "20260315_221545",
    }),
    "PDI4_S2": ("psid_behavioral_PDI4_2_nx_80_n10_i40_dbs_both_narrow_band_flipped", {
        "xp": "20260315_223343",
        "xp_1": "20260315_224234",
        "xp_2": "20260315_224519",
        "xp_with_dbs": "20260315_225241",
    }),
    "PDI4_S3": ("psid_behavioral_PDI4_3_nx65_n10_i40_dbs_both_narrow_band_flipped", {
        "xp": "20260315_202707",
        "xp_1": "20260315_203632",
        "xp_2": "20260315_203936",
        "xp_with_dbs": "20260315_204723",
    }),
}

_FLIPPED_FEAT_SUFFIX: dict[str, tuple[str, str]] = {
    "xp": ("", "Xp"),
    "xp_1": ("_xp_1", "Xp_1"),
    "xp_2": ("_xp_2", "Xp_2"),
    "xp_with_dbs": ("_xp_with_dbs", "Xp_with_dbs"),
}

_H_VALUES = [0.5, 1.5, 2.5, 3.5, 4.5]
_M_VALUES = [0.5, 1.0, 2.0]


def _load_flipped_heatmap_data(
    results_root,
    variant_base: str,
    run_ts: str,
    feat_suffix: str,
    feat_pkl_name: str,
) -> np.ndarray:
    """Return (len(h), len(m)) array of balanced accuracies."""
    import pickle
    from pathlib import Path

    cls_root = Path(results_root) / "classification"
    variant = variant_base + feat_suffix
    var_dir = cls_root / variant / run_ts

    grid = np.full((len(_H_VALUES), len(_M_VALUES)), float("nan"))
    for hi, h in enumerate(_H_VALUES):
        for mi, m in enumerate(_M_VALUES):
            hm_dir = var_dir / f"h{h}_m{m}"
            pkl = hm_dir / f"LDA_{feat_pkl_name}_flipped.pkl"
            if pkl.is_file():
                with open(pkl, "rb") as f:
                    res = pickle.load(f)
                tr = res.get("test_results", {})
                ba = tr.get("balanced_accuracy", res.get("balanced_accuracy", float("nan")))
                grid[hi, mi] = float(ba)
    return grid


def build_flipped_heatmap_figure(
    results_root,
    theme: ThesisTheme = ThesisTheme.LIGHT,
) -> Figure:
    """
    Single 4-row × 4-col figure: sessions (rows) × feature groups (cols).
    Each cell shows a h×m balanced-accuracy heatmap for PSID flipped predictions.
    No figure title — caller adds section labels in surrounding HTML/markdown.
    """
    from plotly.subplots import make_subplots

    paper_bg, plot_bg = paper_colors(theme)
    fg = true_line_color(theme)

    session_keys = list(_FLIPPED_SESSIONS.keys())
    feat_keys = list(GROUP_ORDER)
    n_rows = len(session_keys)
    n_cols = len(feat_keys)

    # Column headers in first row only; blank elsewhere
    all_subtitles = [_FEAT_SHORT[f] for f in feat_keys] + [""] * ((n_rows - 1) * n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=all_subtitles,
        horizontal_spacing=0.025,
        vertical_spacing=0.06,
    )

    h_labels = [f"{h:.1f}" for h in _H_VALUES]
    m_labels = [f"{m:.1f}" for m in _M_VALUES]

    for ri, sess_label in enumerate(session_keys):
        var_base, ts_map = _FLIPPED_SESSIONS[sess_label]
        for ci, feat in enumerate(feat_keys):
            suffix, pkl_name = _FLIPPED_FEAT_SUFFIX[feat]
            run_ts = ts_map.get(feat, ts_map.get("xp", ""))
            grid = _load_flipped_heatmap_data(
                results_root, var_base, run_ts, suffix, pkl_name
            )
            is_last = (ri == n_rows - 1) and (ci == n_cols - 1)
            text_vals = [
                [f"{v:.2f}" if np.isfinite(v) else "" for v in row]
                for row in grid
            ]
            fig.add_trace(
                go.Heatmap(
                    z=grid,
                    x=m_labels,
                    y=h_labels,
                    colorscale="RdYlGn",
                    zmin=0.3,
                    zmax=0.7,
                    zmid=0.5,
                    text=text_vals,
                    texttemplate="%{text}",
                    textfont=dict(size=FONT_SIZE_ANNOTATION - 2),
                    showscale=is_last,
                    colorbar=dict(
                        title=dict(text="BA", side="right"),
                        len=0.35,
                        x=1.01,
                        thickness=12,
                        tickfont=dict(size=FONT_SIZE_TICK),
                        tickvals=[0.3, 0.4, 0.5, 0.6, 0.7],
                        ticktext=["0.3", "0.4", "0.5", "0.6", "0.7"],
                    ) if is_last else None,
                    hovertemplate=(
                        f"{sess_label} — {_FEAT_SHORT[feat]}<br>"
                        "h=%{y} s,  m=%{x} s<br>BA=%{z:.3f}<extra></extra>"
                    ),
                ),
                row=ri + 1,
                col=ci + 1,
            )

    # Axis formatting per cell
    for ri in range(n_rows):
        for ci in range(n_cols):
            # Y-axis: show ticks + session label only in leftmost column
            if ci == 0:
                fig.update_yaxes(
                    title_text=session_keys[ri],
                    title_font=dict(size=FONT_SIZE_TICK - 1, family=FONT_FAMILY),
                    showticklabels=True,
                    autorange="reversed",
                    tickfont=dict(size=FONT_SIZE_TICK - 1),
                    row=ri + 1,
                    col=1,
                )
            else:
                fig.update_yaxes(
                    showticklabels=False,
                    autorange="reversed",
                    row=ri + 1,
                    col=ci + 1,
                )
            # X-axis: show ticks only in bottom row
            if ri == n_rows - 1:
                fig.update_xaxes(
                    showticklabels=True,
                    tickfont=dict(size=FONT_SIZE_TICK - 1),
                    row=ri + 1,
                    col=ci + 1,
                )
            else:
                fig.update_xaxes(
                    showticklabels=False,
                    row=ri + 1,
                    col=ci + 1,
                )

    # Shared axis labels as paper-space annotations
    fig.add_annotation(
        x=-0.02, y=0.5,
        xref="paper", yref="paper",
        text="h — history (s)",
        showarrow=False,
        textangle=-90,
        font=dict(size=FONT_SIZE_LABEL - 1, family=FONT_FAMILY, color=fg),
        xanchor="right",
        yanchor="middle",
    )
    fig.add_annotation(
        x=0.46, y=-0.05,
        xref="paper", yref="paper",
        text="m — forecast horizon (s)",
        showarrow=False,
        font=dict(size=FONT_SIZE_LABEL - 1, family=FONT_FAMILY, color=fg),
        xanchor="center",
        yanchor="top",
    )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg, size=FONT_SIZE_BASE),
        height=560,
        margin=dict(l=100, r=90, t=50, b=70),
    )

    return fig
