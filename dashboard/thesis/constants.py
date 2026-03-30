from __future__ import annotations

from enum import Enum
from typing import Tuple


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
COLOR_PSID_BAND_FILL = "rgba(24, 95, 165, 0.22)"
COLOR_PSID_BAND_LINE = "rgba(24, 95, 165, 0.35)"
# Session-mean RMSE ribbons around each model's ŷ (lighter tints)
COLOR_DPAD_BAND_FILL = "rgba(153, 60, 29, 0.18)"
COLOR_DPAD_BAND_LINE = "rgba(153, 60, 29, 0.28)"
COLOR_VARMA_BAND_FILL = "rgba(136, 135, 128, 0.20)"
COLOR_VARMA_BAND_LINE = "rgba(136, 135, 128, 0.32)"

# Line widths (pt-style mapping via Plotly width)
WIDTH_TRUE = 2.5
WIDTH_PSID = 1.8
WIDTH_DPAD = 1.8
WIDTH_VARMA = 1.5
WIDTH_MEAN = 2.2

FONT_FAMILY = "Arial, Helvetica, sans-serif"
FONT_SIZE_BASE = 11
FONT_SIZE_TICK = 10
FONT_SIZE_LABEL = 11

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
    return "rgba(0,0,0,0.15)" if theme == ThesisTheme.LIGHT else "rgba(255,255,255,0.20)"


def legend_bgcolor() -> str:
    return "rgba(0,0,0,0)"


def dbs_badge_style(dbs_label: str) -> Tuple[str, str]:
    """Return (foreground, background) for a DBS badge annotation."""
    if dbs_label == "DBS-OFF":
        return COLOR_DBS_OFF, "rgba(83,74,183,0.15)"
    if dbs_label == "DBS-ON":
        return COLOR_DBS_ON, "rgba(15,110,86,0.15)"
    return COLOR_TRUE_DARK, "rgba(128,128,128,0.15)"
