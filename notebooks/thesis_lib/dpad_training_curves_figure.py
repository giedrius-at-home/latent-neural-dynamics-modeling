"""
DPAD training curves and training-time comparison figures.

Figure A: 2×2 grid — one panel per participant-session.
  Each panel shows cumulative-epoch loss/val_loss for all 4 DPAD stages.
  Stage boundaries marked by vertical dashed lines.

Figure B: Grouped bar chart comparing total DPAD vs PSID wall-clock fit time per session.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_DPAD,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    grid_color,
    legend_bgcolor,
    paper_colors,
    true_line_color,
)

logger = logging.getLogger(__name__)

# Stage display names and colors
_STAGE_LABELS = [
    "Stage 1\n(model1)",
    "Stage 2\n(model1_Cy)",
    "Stage 3\n(model2)",
    "Stage 4\n(model2_Cz)",
]
_STAGE_KEYS = ["model1", "model1_Cy", "model2", "model2_Cz"]
_STAGE_COLORS = ["#185FA5", "#0F6E56", "#993C1D", "#854F0B"]
_STAGE_RULE_COLOR = "rgba(100,100,100,0.3)"

# Session identifiers to scan for
_DPAD_SESSION_GLOBS = [
    "dpad_PDI1_S2_dbs_both_narrow_band",
    "dpad_PDI1_S4_dbs_both_narrow_band",
    "dpad_PDI4_S2_dbs_both_narrow_band",
    "dpad_PDI4_S3_dbs_both_narrow_band",
]
_SESSION_LABELS = ["PDI1 S2", "PDI1 S4", "PDI4 S2", "PDI4 S3"]

# PSID fit times (wall-clock seconds, from "Calling PSID.PSID" → "Saved model checkpoint")
# extracted from training logs
_PSID_FIT_TIMES: Dict[str, float] = {
    "PDI1 S2": 201.0,
    "PDI1 S4": 675.7,
    "PDI4 S2": 730.6,
    "PDI4 S3": 435.7,
}

_COLOR_PSID = "#185FA5"
_COLOR_DPAD = "#993C1D"


def _load_training_history(
    results_root: Path, dir_name: str
) -> Optional[Dict[str, Any]]:
    path = results_root / dir_name / "training_history.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def build_dpad_training_curves_figure(
    results_root: Path,
    theme: ThesisTheme = ThesisTheme.LIGHT,
) -> go.Figure:
    """2×2 grid: one panel per DPAD session, showing 4-stage training loss curves."""
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=_SESSION_LABELS,
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
        shared_yaxes=False,
    )

    for idx, (dir_name, sess_label) in enumerate(
        zip(_DPAD_SESSION_GLOBS, _SESSION_LABELS)
    ):
        row = idx // 2 + 1
        col = idx % 2 + 1
        show_leg = idx == 0

        hist = _load_training_history(Path(results_root), dir_name)
        if hist is None:
            fig.add_annotation(
                text=f"No training_history.json<br>in {dir_name}",
                x=0.5,
                y=0.5,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                row=row,
                col=col,
            )
            continue

        # Build cumulative epoch axis
        cumulative_offset = 0
        stage_boundaries: List[int] = []
        all_epochs: List[float] = []
        all_loss: List[float] = []
        all_val_loss: List[float] = []

        for stage_key in _STAGE_KEYS:
            stage = hist.get(stage_key)
            if stage is None:
                continue
            epochs = stage.get("epochs", [])
            loss = stage.get("loss", [])
            val_loss = stage.get("val_loss", [])
            if not epochs or not loss:
                continue
            n = min(len(epochs), len(loss))
            cum_eps = [cumulative_offset + e for e in epochs[:n]]
            all_epochs.extend(cum_eps)
            all_loss.extend(loss[:n])
            if val_loss:
                nv = min(len(val_loss), n)
                all_val_loss.extend(val_loss[:nv])
                # pad to n if shorter
                all_val_loss.extend([float("nan")] * (n - nv))
            else:
                all_val_loss.extend([float("nan")] * n)
            cumulative_offset += max(epochs[:n]) + 1
            stage_boundaries.append(cumulative_offset)

        if not all_epochs:
            continue

        # Vertical stage boundary lines
        for bi, bx in enumerate(stage_boundaries[:-1]):
            fig.add_vline(
                x=bx,
                line=dict(color=_STAGE_RULE_COLOR, width=0.8, dash="dash"),
                row=row,
                col=col,
            )
            # Stage label at top of boundary — descriptive names
            _stage_short_names = ["Init", "Cy", "State", "Cz"]
            stage_label = (
                _stage_short_names[bi + 1]
                if bi + 1 < len(_stage_short_names)
                else f"S{bi + 2}"
            )
            fig.add_annotation(
                x=bx,
                y=1.02,
                xref="x",
                yref="y domain",
                text=stage_label,
                showarrow=False,
                font=dict(size=FONT_SIZE_TICK - 1, family=FONT_FAMILY, color=fg),
                xanchor="center",
                row=row,
                col=col,
            )

        # Train loss
        fig.add_trace(
            go.Scatter(
                x=all_epochs,
                y=all_loss,
                mode="lines",
                name="Train loss",
                legendgroup="train",
                showlegend=show_leg,
                line=dict(color=_COLOR_DPAD, width=2.8),
                connectgaps=False,
            ),
            row=row,
            col=col,
        )
        # Val loss
        if any(np.isfinite(v) for v in all_val_loss):
            fig.add_trace(
                go.Scatter(
                    x=all_epochs,
                    y=all_val_loss,
                    mode="lines",
                    name="Val loss",
                    legendgroup="val",
                    showlegend=show_leg,
                    line=dict(color=_COLOR_PSID, width=2.4, dash="dot"),
                    connectgaps=False,
                ),
                row=row,
                col=col,
            )

    # Axis styling
    for r in range(1, 3):
        for c in range(1, 3):
            fig.update_xaxes(
                showgrid=True,
                gridcolor=grid,
                zeroline=False,
                showline=True,
                linecolor=fg,
                linewidth=1,
                tickfont=dict(size=FONT_SIZE_TICK),
                title_text="Epoch" if r == 2 else "",
                title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                row=r,
                col=c,
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor=grid,
                zeroline=False,
                showline=True,
                linecolor=fg,
                linewidth=1,
                tickfont=dict(size=FONT_SIZE_TICK),
                title_text="Loss" if c == 1 else "",
                title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                row=r,
                col=c,
            )

    # Stage legend annotation
    stage_labels_flat = [
        "Init (model1)",
        "Cy (model1_Cy)",
        "State (model2)",
        "Cz (model2_Cz)",
    ]
    annotation_text = "  |  ".join(stage_labels_flat)
    fig.add_annotation(
        x=0.5,
        y=-0.07,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        font=dict(size=FONT_SIZE_TICK - 1, family=FONT_FAMILY, color=fg),
        xanchor="center",
    )

    fig.update_annotations(font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK, color=fg))

    from dashboard.thesis.constants import apply_thesis_style

    apply_thesis_style(
        fig, theme, height=720, margin=dict(l=72, r=24, t=72, b=80), legend_y=-0.08
    )
    return fig


def build_dpad_training_time_figure(
    results_root: Path,
    theme: ThesisTheme = ThesisTheme.LIGHT,
) -> go.Figure:
    """
    Grouped bar chart: DPAD total fit time vs PSID fit time per session.
    DPAD = sum of fit_time_s across 4 stages; PSID from hard-coded log-extracted values.
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    session_labels = _SESSION_LABELS.copy()
    dpad_times: List[float] = []
    psid_times: List[float] = []

    for dir_name, sess_label in zip(_DPAD_SESSION_GLOBS, _SESSION_LABELS):
        hist = _load_training_history(Path(results_root), dir_name)
        if hist is not None:
            total = sum(
                float(hist[s].get("fit_time_s", 0.0)) for s in _STAGE_KEYS if s in hist
            )
        else:
            total = float("nan")
        dpad_times.append(total / 60.0)  # convert to minutes
        psid_times.append(_PSID_FIT_TIMES.get(sess_label, float("nan")) / 60.0)

    fig = go.Figure()

    bar_w = 0.32
    n = len(session_labels)
    xs = list(range(n))

    # PSID bars
    fig.add_trace(
        go.Bar(
            x=[x - bar_w / 2 for x in xs],
            y=psid_times,
            width=bar_w * 0.9,
            name="PSID",
            marker=dict(color=_COLOR_PSID, opacity=0.85),
        )
    )
    # DPAD bars
    fig.add_trace(
        go.Bar(
            x=[x + bar_w / 2 for x in xs],
            y=dpad_times,
            width=bar_w * 0.9,
            name="DPAD",
            marker=dict(color=_COLOR_DPAD, opacity=0.85),
        )
    )

    # Ratio annotations
    for i, (p, d) in enumerate(zip(psid_times, dpad_times)):
        if np.isfinite(p) and np.isfinite(d) and p > 0:
            ratio = d / p
            fig.add_annotation(
                x=i,
                y=d + max(dpad_times) * 0.04,
                text=f"×{ratio:.0f}",
                showarrow=False,
                font=dict(size=FONT_SIZE_TICK - 1, family=FONT_FAMILY, color=fg),
                xanchor="center",
            )

    from dashboard.thesis.constants import apply_thesis_style

    apply_thesis_style(
        fig, theme, height=380, margin=dict(l=72, r=24, t=36, b=80), legend_y=-0.20
    )
    fig.update_layout(
        barmode="group",
        xaxis=dict(
            tickmode="array",
            tickvals=xs,
            ticktext=session_labels,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="Training time (min)",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            ),
            zeroline=False,
        ),
    )
    return fig
