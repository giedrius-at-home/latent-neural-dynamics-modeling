"""
Paper-style Plotly helpers for thesis notebooks.

Isolated from dashboard/thesis/constants.py — this module exists so the
notebooks can evolve their own style without touching the dashboard code.

Design targets (from the reference paper):
  - White canvas, no figure-level frame, no enclosing panel boxes
  - Only bottom + left spines; top + right hidden
  - Outward tick marks, thin dark axis lines, no grid
  - Arial/Helvetica 9-11 pt; bold only for panel letter labels
  - DBS ON = warm red-coral; DBS OFF = medium blue
  - Circles for behavioural results, diamonds for neural
  - Panel labels in the margin as "A - Description", bold uppercase letter
"""

from __future__ import annotations

from typing import Iterable, Optional

import plotly.graph_objects as go

# ── Colors ───────────────────────────────────────────────────────────────────
COLOR_DBS_OFF = "#3B82BD"   # medium blue
COLOR_DBS_ON  = "#E34A33"   # warm red-coral
COLOR_NS      = "#9E9E9E"   # non-significant / chance / secondary
COLOR_CHANCE  = "#BDBDBD"   # chance-level indicator (slightly lighter gray)
COLOR_AXIS    = "#1A1A1A"   # near-black for axes and spines
COLOR_TEXT    = "#1A1A1A"
COLOR_TRUE    = "#1A1A1A"   # ground-truth line color where needed
COLOR_PSID    = "#185FA5"   # model series (kept from old palette)
COLOR_DPAD    = "#993C1D"
COLOR_VARMA   = "#888780"
COLOR_BETA_BG = "rgba(0, 0, 0, 0.05)"  # light gray rectangle for selected freq band

# ── Fonts ────────────────────────────────────────────────────────────────────
FONT_FAMILY         = "Arial, Helvetica, sans-serif"
FONT_SIZE_BASE      = 10   # default body text
FONT_SIZE_TICK      = 9    # tick labels
FONT_SIZE_LABEL     = 11   # axis titles
FONT_SIZE_PANEL     = 11   # panel letter + description
FONT_SIZE_ANNOT     = 9    # in-chart annotations

# ── Markers ──────────────────────────────────────────────────────────────────
MARKER_BEHAVIORAL = "circle"
MARKER_NEURAL     = "diamond"
MARKER_SIZE       = 7

# ── Line widths ──────────────────────────────────────────────────────────────
LINE_WIDTH_MEAN     = 1.8
LINE_WIDTH_SD       = 1.0
LINE_WIDTH_AXIS     = 1.2
LINE_WIDTH_REF      = 1.0   # dashed reference lines (x=y, chance, etc.)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def apply_paper_style(
    fig: go.Figure,
    *,
    height: int = 400,
    margin: Optional[dict] = None,
    show_legend: bool = True,
    legend_y: float = -0.16,
    legend_orientation: str = "h",
) -> None:
    """Apply paper-style layout to ``fig`` in place.

    Call at the end of every figure block, then override per-figure specifics
    (barmode, coloraxis, x-range, etc.) afterwards.
    """
    if margin is None:
        margin = dict(l=70, r=30, t=50, b=80)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=COLOR_TEXT),
        height=height,
        margin=margin,
        showlegend=show_legend,
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=legend_y,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK, color=COLOR_TEXT),
            itemsizing="constant",
            tracegroupgap=6,
        ),
    )

    axis_kwargs = dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=LINE_WIDTH_AXIS,
        linecolor=COLOR_AXIS,
        mirror=False,
        ticks="outside",
        ticklen=4,
        tickwidth=LINE_WIDTH_AXIS,
        tickcolor=COLOR_AXIS,
        tickfont=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK, color=COLOR_TEXT),
        title=dict(font=dict(family=FONT_FAMILY, size=FONT_SIZE_LABEL, color=COLOR_TEXT)),
    )
    fig.update_xaxes(**axis_kwargs)
    fig.update_yaxes(**axis_kwargs)


def panel_label(
    fig: go.Figure,
    letter: str,
    description: str,
    *,
    row: Optional[int] = None,
    col: Optional[int] = None,
    x_offset: float = 0.0,
    y_offset: float = 0.08,
) -> None:
    """Add a bold ``A - Description`` panel label above-left of a sub-panel.

    Places the annotation in the white margin above the axes, in paper-style
    format. For single-panel figures, omit ``row``/``col``.
    """
    text = f"<b>{letter}</b> - {description}"
    xref = "paper" if row is None else None
    yref = "paper" if row is None else None

    if row is None:
        fig.add_annotation(
            x=0.0 + x_offset, y=1.0 + y_offset,
            xref="paper", yref="paper",
            text=text, showarrow=False,
            xanchor="left", yanchor="bottom",
            font=dict(family=FONT_FAMILY, size=FONT_SIZE_PANEL, color=COLOR_TEXT),
        )
        return

    fig.add_annotation(
        x=0.0 + x_offset, y=1.0 + y_offset,
        xref=f"x{'' if (row == 1 and col == 1) else _axis_suffix(fig, row, col)} domain",
        yref=f"y{'' if (row == 1 and col == 1) else _axis_suffix(fig, row, col)} domain",
        text=text, showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_PANEL, color=COLOR_TEXT),
    )


def _axis_suffix(fig: go.Figure, row: int, col: int) -> str:
    """Compute the Plotly axis index suffix for (row, col) in a subplot grid.

    Relies on fig._grid_ref laid down by ``make_subplots``. Returns '' for (1,1)
    or a numeric string like '3' for later cells.
    """
    grid_ref = getattr(fig, "_grid_ref", None)
    if grid_ref is None:
        return ""
    try:
        cell = grid_ref[row - 1][col - 1][0]
    except (IndexError, TypeError):
        return ""
    xaxis_name = cell.layout_keys[0]  # e.g. 'xaxis' or 'xaxis3'
    suffix = xaxis_name.replace("xaxis", "")
    return suffix


def stats_annotation(
    fig: go.Figure,
    r: Optional[float] = None,
    p: Optional[float] = None,
    r2: Optional[float] = None,
    *,
    row: Optional[int] = None,
    col: Optional[int] = None,
    x: float = 0.03,
    y: float = 0.97,
) -> None:
    """Add an inline r / P / R² annotation in the top-left of a sub-panel."""
    parts = []
    if r is not None:
        parts.append(f"r = {r:.2f}")
    if p is not None:
        parts.append(f"P < {p:.3g}" if p < 0.001 else f"P = {p:.3f}")
    if r2 is not None:
        parts.append(f"R² = {r2:.2f}")
    text = ", ".join(parts)
    if not text:
        return

    if row is None:
        fig.add_annotation(
            x=x, y=y, xref="paper", yref="paper",
            text=text, showarrow=False,
            xanchor="left", yanchor="top",
            font=dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOT, color=COLOR_TEXT),
        )
        return

    suffix = _axis_suffix(fig, row, col)
    fig.add_annotation(
        x=x, y=y,
        xref=f"x{suffix} domain", yref=f"y{suffix} domain",
        text=text, showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOT, color=COLOR_TEXT),
    )


def add_freq_band_highlight(
    fig: go.Figure,
    band: tuple[float, float],
    *,
    row: Optional[int] = None,
    col: Optional[int] = None,
    fillcolor: str = COLOR_BETA_BG,
) -> None:
    """Draw a light-gray rectangle across ``band`` on the (row, col) panel.

    Used behind PSD curves to mark the selected frequency band.
    """
    kwargs = dict(
        type="rect",
        xref="x" if row is None else f"x{_axis_suffix(fig, row, col)}",
        yref="y domain" if row is None else f"y{_axis_suffix(fig, row, col)} domain",
        x0=band[0], x1=band[1],
        y0=0, y1=1,
        fillcolor=fillcolor,
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(**kwargs)


def add_sig_bar(
    fig: go.Figure,
    x0: float,
    x1: float,
    y: float,
    text: str = "P < 0.01",
    *,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> None:
    """Draw a thin horizontal significance bar with a label above it."""
    suffix = _axis_suffix(fig, row, col) if row is not None else ""
    xref = f"x{suffix}" if row is not None else "x"
    yref = f"y{suffix}" if row is not None else "y"
    fig.add_shape(
        type="line", xref=xref, yref=yref,
        x0=x0, x1=x1, y0=y, y1=y,
        line=dict(color=COLOR_AXIS, width=LINE_WIDTH_AXIS),
    )
    fig.add_annotation(
        x=(x0 + x1) / 2, y=y,
        xref=xref, yref=yref,
        text=text, showarrow=False,
        xanchor="center", yanchor="bottom",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOT, color=COLOR_TEXT),
        yshift=2,
    )


def dbs_color(condition: str) -> str:
    """Return the canonical DBS color for ``condition`` ∈ {'on', 'off'}."""
    c = condition.lower()
    if c in ("on", "dbs-on", "dbs_on"):
        return COLOR_DBS_ON
    if c in ("off", "dbs-off", "dbs_off"):
        return COLOR_DBS_OFF
    return COLOR_NS


# ─────────────────────────────────────────────────────────────────────────────
# Compat layer for legacy dashboard-style imports used by sec2+ notebooks.
# Lets a notebook swap its import block in one place without rewriting every
# figure body. All theme arguments are ignored — paper style is forced.
# ─────────────────────────────────────────────────────────────────────────────
from enum import Enum as _Enum


class ThesisTheme(str, _Enum):
    LIGHT = "light"
    DARK = "dark"


# Model band ribbons (RGBA tints matching COLOR_PSID/DPAD/VARMA)
COLOR_PSID_BAND_FILL  = "rgba(24, 95, 165, 0.14)"
COLOR_PSID_BAND_LINE  = "rgba(24, 95, 165, 0.25)"
COLOR_DPAD_BAND_FILL  = "rgba(153, 60, 29, 0.12)"
COLOR_DPAD_BAND_LINE  = "rgba(153, 60, 29, 0.20)"
COLOR_VARMA_BAND_FILL = "rgba(136, 135, 128, 0.12)"
COLOR_VARMA_BAND_LINE = "rgba(136, 135, 128, 0.22)"

# Faint separator line
COLOR_SEPARATOR = "#E5E5E5"

# Trace line widths / opacities (paper-style: slightly thinner than dashboard)
WIDTH_TRUE  = 1.8
WIDTH_PSID  = 1.6
WIDTH_DPAD  = 1.6
WIDTH_VARMA = 1.4

OPACITY_PSID  = 0.90
OPACITY_DPAD  = 0.85
OPACITY_VARMA = 0.80

# Size knobs used by dashboard builders
FIGURE_HEIGHT = 420
FONT_SIZE_ANNOTATION = FONT_SIZE_ANNOT  # legacy alias
DOT_SIZE = 7

# Beta band border (used as outline when shading the 12-30 Hz band)
COLOR_BETA_BORDER = "#B0B0B0"

# Participant palette (F1 dot plots, per-subject colors across figures)
PARTICIPANT_P01 = "#185FA5"
PARTICIPANT_P02 = "#639922"
PARTICIPANT_P03 = "#993C1D"
PARTICIPANT_P04 = "#854F0B"

PARTICIPANT_COLORS: dict[str, str] = {
    "P01": PARTICIPANT_P01, "P02": PARTICIPANT_P02,
    "P03": PARTICIPANT_P03, "P04": PARTICIPANT_P04,
    "PDI1": PARTICIPANT_P01, "PDI4": PARTICIPANT_P02,
    "PDI2": PARTICIPANT_P03, "PDI3": PARTICIPANT_P04,
}


def true_line_color(theme: "ThesisTheme | None" = None) -> str:
    return COLOR_AXIS


def paper_colors(theme: "ThesisTheme | None" = None) -> tuple[str, str]:
    return "#FFFFFF", "#FFFFFF"


def grid_color(theme: "ThesisTheme | None" = None) -> str:
    return "rgba(0,0,0,0)"  # paper style: no grid


def legend_bgcolor() -> str:
    return "rgba(0,0,0,0)"


def rmse_axis_label(variable: str = "") -> str:
    if variable:
        return f"RMSE [z] \u2014 {variable}"
    return "RMSE [z]"


def zscore_axis_label(channel: str = "") -> str:
    if channel:
        return f"z-score \u2014 {channel}"
    return "z-score"


def d_score_axis_label(signal: str, axis: str = "") -> str:
    full = f"{signal}_{axis}" if axis else signal
    return f"z-score \u2014 {full}"


def dbs_badge_style(dbs_label: str) -> tuple[str, str]:
    """(foreground, background) for a DBS badge annotation."""
    if dbs_label in ("DBS-OFF", "DBS_OFF", "OFF"):
        return COLOR_DBS_OFF, hex_to_rgba(COLOR_DBS_OFF, 0.12)
    if dbs_label in ("DBS-ON", "DBS_ON", "ON"):
        return COLOR_DBS_ON, hex_to_rgba(COLOR_DBS_ON, 0.12)
    return COLOR_NS, "rgba(128,128,128,0.12)"


def apply_thesis_style(
    fig: go.Figure,
    theme: "ThesisTheme | None" = None,
    *,
    height: int = FIGURE_HEIGHT,
    margin: Optional[dict] = None,
    hovermode: str = "x unified",
    legend_y: float = -0.16,
    legend_orientation: str = "h",
    show_legend: bool = True,
) -> None:
    """Drop-in replacement for dashboard.thesis.constants.apply_thesis_style.

    Ignores ``theme`` and ``hovermode``; always applies the paper style defined
    in :func:`apply_paper_style`. Extra kwargs are forwarded.
    """
    apply_paper_style(
        fig,
        height=height,
        margin=margin,
        show_legend=show_legend,
        legend_y=legend_y,
        legend_orientation=legend_orientation,
    )
    if hovermode:
        fig.update_layout(hovermode=hovermode)
