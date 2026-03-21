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

# PSID uncertainty band (light blue tint over white)
COLOR_PSID_BAND_FILL = "rgba(24, 95, 165, 0.22)"
COLOR_PSID_BAND_LINE = "rgba(24, 95, 165, 0.35)"

# Line widths (pt-style mapping via Plotly width)
WIDTH_TRUE = 2.5
WIDTH_PSID = 1.5
WIDTH_DPAD = 1.5
WIDTH_VARMA = 1.2

FONT_FAMILY = "Arial, Helvetica, sans-serif"

# Participant identity (F1 classification dot plot; consistent across thesis)
PARTICIPANT_P01 = "#185FA5"  # blue
PARTICIPANT_P02 = "#2E7D4A"  # green
PARTICIPANT_P03 = "#E07A5F"  # coral
PARTICIPANT_P04 = "#C9A227"  # amber

PARTICIPANT_COLORS: dict[str, str] = {
    "P01": PARTICIPANT_P01,
    "P02": PARTICIPANT_P02,
    "P03": PARTICIPANT_P03,
    "P04": PARTICIPANT_P04,
}


def true_line_color(theme: ThesisTheme) -> str:
    return COLOR_TRUE_LIGHT if theme == ThesisTheme.LIGHT else COLOR_TRUE_DARK


def paper_colors(theme: ThesisTheme) -> Tuple[str, str]:
    """paper_bgcolor, plot_bgcolor."""
    if theme == ThesisTheme.LIGHT:
        return "#FFFFFF", "#F4F4F2"
    return "#1A1A18", "#222220"


def grid_color(theme: ThesisTheme) -> str:
    return "rgba(0,0,0,0.08)" if theme == ThesisTheme.LIGHT else "rgba(255,255,255,0.12)"
