from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


class ThesisTheme(str, Enum):
    """Light theme is default for paper figures (white background)."""

    LIGHT = "light"
    DARK = "dark"


# Fixed series colors (non-negotiable across thesis figures)
COLOR_TRUE_LIGHT = "#2C2C2A"
COLOR_TRUE_DARK = "#D3D1C7"
COLOR_PSID = "#185FA5"
COLOR_DPAD = "#993C1D"
COLOR_VARMA = "#888780"

# DBS condition badge / accent colours
COLOR_DBS_OFF = "#534AB7"
COLOR_DBS_ON = "#0F6E56"

# Annotation / decoration colours
COLOR_SEPARATOR = "#888780"
COLOR_CHANCE = "#E24B4A"
COLOR_BETA_BORDER = "#E24B4A"

# PSID uncertainty band (light blue tint over white)
COLOR_PSID_BAND_FILL = "rgba(24, 95, 165, 0.14)"
COLOR_PSID_BAND_LINE = "rgba(24, 95, 165, 0.25)"
# Session-mean RMSE ribbons around each model's ŷ (lighter tints)
COLOR_DPAD_BAND_FILL = "rgba(153, 60, 29, 0.12)"
COLOR_DPAD_BAND_LINE = "rgba(153, 60, 29, 0.20)"
COLOR_VARMA_BAND_FILL = "rgba(136, 135, 128, 0.12)"
COLOR_VARMA_BAND_LINE = "rgba(136, 135, 128, 0.22)"

# Line widths (pt-style mapping via Plotly width)
WIDTH_TRUE = 2.5
WIDTH_PSID = 2.0
WIDTH_DPAD = 2.0
WIDTH_VARMA = 1.8

# Trace opacity for prediction lines (< 1.0 to reduce overlap clutter)
OPACITY_PSID = 0.85
OPACITY_DPAD = 0.75
OPACITY_VARMA = 0.70
WIDTH_MEAN = 2.2

FONT_FAMILY = "Arial, Helvetica, sans-serif"
FONT_SIZE_BASE = 12
FONT_SIZE_TICK = 11
FONT_SIZE_LABEL = 13
FONT_SIZE_ANNOTATION = 10  # small in-chart annotations, badges, etc.

DOT_SIZE = 10

FIGURE_HEIGHT = 500
FIGURE_HEIGHT_STACKED = 700
# C2 forecast: two panels + dual time axes (matches dashboard.forecasting layout breathing room)
FIGURE_HEIGHT_C2_FORECAST = 880

# Participant identity (F1 classification dot plot; consistent across thesis)
PARTICIPANT_P01 = "#185FA5"
PARTICIPANT_P02 = "#639922"
PARTICIPANT_P03 = "#993C1D"
PARTICIPANT_P04 = "#854F0B"

PARTICIPANT_COLORS: dict[str, str] = {
    "P01": PARTICIPANT_P01,
    "P02": PARTICIPANT_P02,
    "P03": PARTICIPANT_P03,
    "P04": PARTICIPANT_P04,
    "PDI1": PARTICIPANT_P01,
    "PDI4": PARTICIPANT_P02,
    "PDI2": PARTICIPANT_P03,
    "PDI3": PARTICIPANT_P04,
}


def true_line_color(theme: ThesisTheme) -> str:
    return COLOR_TRUE_LIGHT if theme == ThesisTheme.LIGHT else COLOR_TRUE_DARK


def paper_colors(theme: ThesisTheme) -> Tuple[str, str]:
    """paper_bgcolor, plot_bgcolor."""
    if theme == ThesisTheme.LIGHT:
        return "#FFFFFF", "#FFFFFF"
    return "#1A1A18", "#222220"


def grid_color(theme: ThesisTheme) -> str:
    return (
        "rgba(0,0,0,0.15)" if theme == ThesisTheme.LIGHT else "rgba(255,255,255,0.20)"
    )


def legend_bgcolor() -> str:
    return "rgba(0,0,0,0)"


def rmse_axis_label(variable: str = "") -> str:
    """Standard y-axis label for RMSE of a z-scored variable, e.g. 'RMSE(z) — tracing_velocity_x'."""
    if variable:
        return f"RMSE(z) \u2014 {variable}"
    return "RMSE(z)"


def zscore_axis_label(channel: str = "") -> str:
    """Standard y-axis label for a z-scored signal, e.g. 'z-score \u2014 ECOG_1_delta_05_raw'."""
    if channel:
        return f"z-score \u2014 {channel}"
    return "z-score"


def dbs_badge_style(dbs_label: str) -> Tuple[str, str]:
    """Return (foreground, background) for a DBS badge annotation."""
    if dbs_label == "DBS-OFF":
        return COLOR_DBS_OFF, "rgba(83,74,183,0.15)"
    if dbs_label == "DBS-ON":
        return COLOR_DBS_ON, "rgba(15,110,86,0.15)"
    return COLOR_TRUE_DARK, "rgba(128,128,128,0.15)"


def d_score_axis_label(signal: str, axis: str = "") -> str:
    """Standard y-axis label for z-scored behavioral axes, e.g. ``z-score — tracing_velocity_x``.

    Note: name is historical; the data plotted is z-scored, so the label uses
    ``z-score`` (matches :func:`zscore_axis_label`).
    """
    full = f"{signal}_{axis}" if axis else signal
    return f"z-score \u2014 {full}"


def apply_thesis_style(
    fig: Figure,
    theme: ThesisTheme = ThesisTheme.LIGHT,
    *,
    height: int = FIGURE_HEIGHT,
    margin: dict | None = None,
    hovermode: str = "x unified",
    legend_y: float = -0.16,
    legend_orientation: str = "h",
    show_legend: bool = True,
) -> None:
    """Apply the shared thesis figure skeleton to *fig* in-place.

    Call this at the end of every ``build_*`` function instead of writing
    inline ``update_layout`` / ``update_xaxes`` / ``update_yaxes`` blocks.
    Figure-specific overrides (``barmode``, ``coloraxis``, extra annotations,
    etc.) can be applied *after* this call.
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    if margin is None:
        margin = dict(l=72, r=40, t=36, b=100)

    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=height,
        margin=margin,
        hovermode=hovermode,
        showlegend=show_legend,
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=legend_y,
            xanchor="center",
            x=0.5,
            bgcolor=legend_bgcolor(),
            font=dict(size=FONT_SIZE_TICK),
            itemsizing="constant",
            itemwidth=40,
            tracegroupgap=8,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linewidth=1,
        linecolor=fg,
        mirror=False,
        tickfont=dict(size=FONT_SIZE_TICK),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linewidth=1,
        linecolor=fg,
        mirror=False,
        tickfont=dict(size=FONT_SIZE_TICK),
    )
