"""Bar + SEM + jittered trial RMSE dots; box-and-whisker variant; optional Wilcoxon brackets."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from thesis_lib.aggregate_rmse import AggregateRmseData, p_to_stars
from thesis_lib.constants import (
    COLOR_DPAD,
    COLOR_PSID,
    COLOR_SEPARATOR,
    COLOR_VARMA,
    PARTICIPANT_COLORS,
    FIGURE_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    grid_color,
    paper_colors,
    rmse_axis_label,
    true_line_color,
)

# Bar fill opacity: DBS-OFF vs DBS-ON (within same model hue)
ALPHA_OFF = 0.80
ALPHA_ON = 0.40
# Slightly lower for scatter so bars read first
DOT_ALPHA_OFF = 0.60
DOT_ALPHA_ON = 0.35

X_POS = np.arange(6, dtype=float)
CATEGORY_LABELS = [
    "PSID_OFF",
    "PSID_ON",
    "DPAD_OFF",
    "DPAD_ON",
    "VARMA_OFF",
    "VARMA_ON",
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
    y_axis_label: str | None = None,
    show_dots: bool = True,
    show_bars: bool = True,
) -> Figure:
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)
    y_label = y_axis_label or "RMSE(z)"

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

    if show_bars:
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
                name="Mean",
                showlegend=True,
            )
        )

    if show_dots:
        # Slightly more opaque dots when bars are absent so the strip reads on its own.
        dot_alpha_off = DOT_ALPHA_OFF if show_bars else 0.85
        dot_alpha_on = DOT_ALPHA_ON if show_bars else 0.55
        for i in range(6):
            pts = data.trial_rmse[i]
            if not pts:
                continue
            jt = rng.uniform(-jitter, jitter, size=len(pts))
            alpha = dot_alpha_on if _is_on_cell(i) else dot_alpha_off
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
                    name="One dot = one test trial RMSE" if i == 0 else None,
                    showlegend=i == 0,
                    legendgroup="dots",
                )
            )

    # ymax_data: bars contribute mean+SEM (always); dots contribute the 98th percentile
    # of pooled trial RMSEs so a single outlier trial does not blow up the y-axis range.
    ymax_data = 0.0
    for i in range(6):
        if np.isfinite(means[i]):
            ymax_data = max(
                ymax_data, float(means[i] + (sems[i] if np.isfinite(sems[i]) else 0))
            )
    if show_dots:
        all_pts = []
        for i in range(6):
            for v in data.trial_rmse[i]:
                if np.isfinite(v):
                    all_pts.append(float(v))
        if all_pts:
            ymax_data = max(ymax_data, float(np.percentile(all_pts, 98)))

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
            (0, 2, w.psid_vs_dpad_off_p, "PSID vs DPAD (DBS-OFF)"),
            (1, 3, w.psid_vs_dpad_on_p, "PSID vs DPAD (DBS-ON)"),
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
                line=dict(color=COLOR_SEPARATOR, width=0.7, dash="dash"),
                opacity=0.5,
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
                font=dict(size=FONT_SIZE_LABEL, color=fg, family=FONT_FAMILY),
            )
        )

    apply_thesis_style(
        fig,
        theme,
        height=FIGURE_HEIGHT,
        margin=dict(l=72, r=32, t=50, b=140),
        hovermode="closest",
        legend_y=-0.12,
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(X_POS),
            ticktext=CATEGORY_LABELS,
            title=dict(
                text="Model × DBS condition",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                standoff=14,
            ),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            ),
            range=[0, y_max * 1.02],
            zeroline=True,
            zerolinecolor=grid,
        ),
        shapes=shapes,
        annotations=annotations,
    )

    return fig


# Cell index -> (panel, model): (0,0)=PSID off, (1,0)=PSID on, (0,1)=DPAD off, ...
_CELL_TO_MODEL = ["PSID", "PSID", "DPAD", "DPAD", "VARMA", "VARMA"]
_MODEL_COLORS = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}
# DBS-OFF: cells 0,2,4 → PSID, DPAD, VARMA. DBS-ON: cells 1,3,5
_OFF_CELLS = [0, 2, 4]
_ON_CELLS = [1, 3, 5]


def build_rmse_boxplot_figure(
    data: AggregateRmseData,
    theme: ThesisTheme,
    rng: np.random.Generator,
    jitter: float = 0.12,
    title: str | None = None,
) -> Figure:
    """
    Two-panel box + strip: DBS-OFF | DBS-ON.
    Each panel: box-and-whisker (IQR, median, 5th–95th) per model + trial dots coloured by participant.
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    models = ["PSID", "DPAD", "VARMA"]
    model_colors = [COLOR_PSID, COLOR_DPAD, COLOR_VARMA]
    x_positions = list(range(3))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["DBS-OFF", "DBS-ON"],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    # Use trial_rmse_with_participant when available, else trial_rmse with "?" as participant
    has_pid = (
        hasattr(data, "trial_rmse_with_participant")
        and len(data.trial_rmse_with_participant) == 6
        and any(len(data.trial_rmse_with_participant[i]) > 0 for i in range(6))
    )
    if has_pid:
        cells_with_pid = data.trial_rmse_with_participant
    else:
        cells_with_pid = tuple(
            [(float(v), "?") for v in data.trial_rmse[i] if np.isfinite(v)]
            for i in range(6)
        )

    def _get_vals(cell_idx: int, model_idx: int) -> tuple[list[float], list[str]]:
        rows = (
            cells_with_pid[cell_idx]
            if has_pid
            else [(v, "?") for v in data.trial_rmse[cell_idx]]
        )
        vals = [r[0] for r in rows if np.isfinite(r[0])]
        pids = [str(r[1]) for r in rows if np.isfinite(r[0])]
        return vals, pids

    # Clip y-axis to the 95th percentile of pooled values so IQR boxes stay legible
    # even when a handful of outliers inflate the range (e.g. PSID baseline trials).
    all_vals: list[float] = []
    for col_idx, (_, cell_ids) in enumerate(
        [("DBS-OFF", _OFF_CELLS), ("DBS-ON", _ON_CELLS)]
    ):
        for mi in range(3):
            cell_idx = cell_ids[mi]
            vals_tmp, _ = _get_vals(cell_idx, mi)
            all_vals.extend(vals_tmp)
    if all_vals:
        p95 = float(np.percentile(all_vals, 95))
        p99 = float(np.percentile(all_vals, 99))
        y_max = max(p95 * 1.12, p99 * 1.02)
    else:
        y_max = 0.0

    for col_idx, (cond_label, cell_ids) in enumerate(
        [("DBS-OFF", _OFF_CELLS), ("DBS-ON", _ON_CELLS)]
    ):
        for mi, (model, col) in enumerate(zip(models, model_colors)):
            cell_idx = cell_ids[mi]
            vals, pids = _get_vals(cell_idx, mi)
            if not vals:
                continue

            arr = np.array(vals, dtype=float)

            q1, med, q3 = np.percentile(arr, [25, 50, 75])
            iqr = q3 - q1
            w_lo = max(float(np.nanmin(arr)), q1 - 1.5 * iqr)
            w_hi = min(float(np.nanmax(arr)), q3 + 1.5 * iqr)

            # Box: IQR fill + median line (Plotly Box with boxpoints=False)
            fig.add_trace(
                go.Box(
                    y=vals,
                    x=[mi] * len(vals),
                    name=model,
                    marker_color=col,
                    line=dict(color=col, width=1.8),
                    fillcolor=_hex_to_rgba(col, 0.45),
                    boxpoints=False,
                    quartilemethod="exclusive",
                    showlegend=(col_idx == 0),
                    legendgroup=model,
                ),
                row=1,
                col=col_idx + 1,
            )

            # Scatter: jittered points coloured by participant
            jt = rng.uniform(-jitter, jitter, size=len(vals))
            x_jittered = [mi + jt[i] for i in range(len(vals))]
            colors = [PARTICIPANT_COLORS.get(p, "#888888") for p in pids]

            fig.add_trace(
                go.Scatter(
                    x=x_jittered,
                    y=vals,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=colors,
                        line=dict(width=0),
                        symbol="circle",
                    ),
                    name="trials" if (col_idx == 0 and mi == 0) else None,
                    showlegend=(col_idx == 0 and mi == 0),
                    legendgroup="trials",
                    hovertext=[f"{p}<br>{v:.3f}" for p, v in zip(pids, vals)],
                ),
                row=1,
                col=col_idx + 1,
            )

    # Keep y_max capped at the 95th-percentile band established above; guard a minimum floor.
    y_max = max(y_max, 0.1)

    # Explicit participant colours in legend (dots use these; green ≈ PDI4, blue ≈ PDI1).
    legend_pids: set[str] = set()
    for i in range(6):
        rows = cells_with_pid[i] if has_pid else [(v, "?") for v in data.trial_rmse[i]]
        for r in rows:
            if len(r) >= 2 and np.isfinite(r[0]):
                legend_pids.add(str(r[1]))
    for pid in sorted(legend_pids):
        col = PARTICIPANT_COLORS.get(pid, "#888888")
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=11, color=col, line=dict(width=0)),
                name=f"trial dot: {pid}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    apply_thesis_style(
        fig,
        theme,
        height=FIGURE_HEIGHT,
        margin=dict(l=72, r=32, t=28, b=140),
        hovermode="closest",
        legend_y=-0.2,
    )

    # X-axis: model names for each subplot
    fig.update_xaxes(
        ticktext=models,
        tickvals=list(range(3)),
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=fg,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        ticktext=models,
        tickvals=list(range(3)),
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=fg,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text=rmse_axis_label(),
        range=[0, y_max],
        showgrid=True,
        gridcolor=grid,
        zeroline=True,
        zerolinecolor=grid,
        showline=True,
        linecolor=fg,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[0, y_max],
        showgrid=True,
        gridcolor=grid,
        zeroline=True,
        zerolinecolor=grid,
        showline=True,
        linecolor=fg,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=2,
    )

    return fig
