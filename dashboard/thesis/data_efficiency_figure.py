"""
Data efficiency (sample complexity) figure.

Shows PSID behavioral RMSE as a function of training set size (n_train_trials),
one curve per participant-session, to illustrate data hungriness.

Data source: results/data_hungriness/psid_behavioral_*/data_hungriness_summary.json
Each JSON is a list of dicts with keys:
  n_train_trials, rmse_neural_mean, rmse_behavioral_mean, rmse_neural_se_mean, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from dashboard.thesis.constants import (
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    PARTICIPANT_COLORS,
    grid_color,
    legend_bgcolor,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.constants import ThesisTheme

logger = logging.getLogger(__name__)

_HUNGRINESS_SUBDIR = "data_hungriness"


def _session_display_label(dir_name: str) -> Tuple[str, str]:
    """Return (participant_label, session_label) from directory name."""
    # e.g. psid_behavioral_PDI1_2_nx_40_n4_i80_dbs_both_log_power
    parts = dir_name.split("_")
    # find PDIx and session number
    for i, p in enumerate(parts):
        if p.startswith("PDI") and len(p) > 3:
            pid = p  # e.g. PDI1
            sess = parts[i + 1] if i + 1 < len(parts) else "?"
            return pid, f"{pid}_S{sess}"
    return "?", dir_name[:20]


def _load_hungriness_data(
    hungriness_root: Path,
) -> List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns list of (participant_label, session_label, n_trials, rmse_mean, rmse_se).
    rmse = rmse_behavioral_mean (z-scored RMSE is not available here; raw RMSE used,
    normalised per session to [0, 1] for comparability).
    """
    results = []
    if not hungriness_root.is_dir():
        return results

    for json_path in sorted(hungriness_root.glob("*/data_hungriness_summary.json")):
        dir_name = json_path.parent.name
        pid, sess_label = _session_display_label(dir_name)
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            logger.warning("Could not load %s", json_path)
            continue

        # Sort by n_train_trials
        data = sorted(data, key=lambda d: d["n_train_trials"])
        n_arr = np.array([d["n_train_trials"] for d in data], dtype=float)
        rmse_arr = np.array([d.get("rmse_neural_mean", float("nan")) for d in data], dtype=float)
        se_arr = np.array([d.get("rmse_neural_se_mean", float("nan")) for d in data], dtype=float)
        results.append((pid, sess_label, n_arr, rmse_arr, se_arr))

    return results


def build_data_efficiency_figure(
    results_root: Path,
    theme: ThesisTheme = ThesisTheme.LIGHT,
    show_se_bands: bool = True,
) -> go.Figure:
    """
    Line plot: PSID neural RMSE vs n_train_trials, one line per participant-session.
    Coloured by participant (PDI1 = blue family, PDI2/PDI3/PDI4 = green family).
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    hungriness_root = Path(results_root) / _HUNGRINESS_SUBDIR
    session_data = _load_hungriness_data(hungriness_root)

    if not session_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No data_hungriness results found.",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14, family=FONT_FAMILY),
        )
        fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
        return fig

    # Participant → color mapping (extend PARTICIPANT_COLORS with shades)
    _extra_colors: Dict[str, str] = {
        "PDI1": "#185FA5",
        "PDI2": "#4CAF82",
        "PDI3": "#0F6E56",
        "PDI4": "#2D9659",
    }

    # Track which participants have been shown in legend
    seen_pids: set[str] = set()

    fig = go.Figure()

    for pid, sess_label, n_arr, rmse_arr, se_arr in session_data:
        color = PARTICIPANT_COLORS.get(pid, _extra_colors.get(pid, "#888888"))
        show_leg = pid not in seen_pids
        seen_pids.add(pid)

        # SE band
        if show_se_bands and np.any(np.isfinite(se_arr)):
            upper = rmse_arr + se_arr
            lower = np.clip(rmse_arr - se_arr, 0, None)
            fill_color = color.lstrip("#")
            r, g, b = int(fill_color[0:2], 16), int(fill_color[2:4], 16), int(fill_color[4:6], 16)
            rgba_fill = f"rgba({r},{g},{b},0.12)"
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([n_arr, n_arr[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=rgba_fill,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=n_arr,
                y=rmse_arr,
                mode="lines+markers",
                name=pid,
                legendgroup=pid,
                showlegend=show_leg,
                line=dict(color=color, width=1.6),
                marker=dict(size=4, color=color, opacity=0.8),
                hovertemplate=f"{sess_label}<br>n=%{{x}}<br>RMSE=%{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=520,
        xaxis=dict(
            title=dict(
                text="Training trials (n)",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            ),
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showline=True,
            linecolor=fg,
            linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
        ),
        yaxis=dict(
            title=dict(
                text="RMSE(z) \u2014 neural",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            ),
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
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor=legend_bgcolor(),
            font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
        ),
        margin=dict(l=72, r=24, t=36, b=80),
        hovermode="closest",
    )
    return fig
