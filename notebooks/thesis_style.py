"""Paper-style matplotlib helpers for thesis notebooks.

Static PNG output via matplotlib + scienceplots. No Plotly, no kaleido.

Usage:
    from thesis_style import apply_thesis_style, panel_label, COLOR_DBS_OFF, ...
    apply_thesis_style()                                # once per notebook
    fig, ax = plt.subplots(...)
    # ... plotting ...
    panel_label(ax, "A", "Description")
    fig.savefig(path)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 — registers the 'science' mpl style
from matplotlib.colors import to_rgba

# ── Colors ───────────────────────────────────────────────────────────────────
COLOR_DBS_OFF = "#3B82BD"   # medium blue
COLOR_DBS_ON  = "#E34A33"   # warm red-coral
COLOR_NS      = "#9E9E9E"
COLOR_CHANCE  = "#BDBDBD"
COLOR_AXIS    = "#1A1A1A"
COLOR_TEXT    = "#1A1A1A"
COLOR_PSID    = "#185FA5"
COLOR_DPAD    = "#993C1D"
COLOR_VARMA   = "#5B8C5A"
COLOR_TRUE    = "#1A1A1A"

# Participant palette (F1 dot plots, per-subject colors across figures)
PARTICIPANT_COLORS: dict[str, str] = {
    "P01": "#185FA5", "P02": "#639922", "P03": "#993C1D", "P04": "#854F0B",
    "PDI1": "#185FA5", "PDI4": "#639922", "PDI2": "#993C1D", "PDI3": "#854F0B",
}


def apply_thesis_style() -> None:
    """Apply the scienceplots 'science' base style.

    Trusts the scienceplots + ieee defaults — no custom rcParams.
    The only override is constrained_layout, which prevents the per-cell
    "does the legend fit?" tuning Plotly needed.
    """
    plt.style.use(["science", "ieee"])
    plt.rcParams.update({
        # Disable LaTeX — no TeX install, panel_label uses mathtext instead.
        "text.usetex":                   False,

        # Auto-layout prevents per-cell legend/margin tuning.
        "figure.constrained_layout.use": True,

        # Times isn't installed; fall back to Nimbus Roman (ghostscript) and
        # DejaVu Serif so the IEEE-style serif actually renders.
        "font.serif":      ["Nimbus Roman", "DejaVu Serif", "Times"],

        # Axis labels a little bigger than the IEEE default (which shrinks
        # everything for column width); titles slightly smaller so the panel
        # subtitle never dominates the data.
        "axes.labelsize":   7,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "axes.titlesize":   9,
        "axes.titleweight": "regular",
        "axes.titlepad":    8,

        # Legend breathing room around its border and between items.
        "legend.fontsize":     8,
        "legend.borderpad":    0.8,
        "legend.labelspacing": 0.5,
    })


def panel_label(ax, letter: str, text: str = "") -> None:
    """Set the axes subtitle to a bold 'A — text' label.

    Uses ``ax.set_title(loc='left')`` so constrained_layout allocates the
    space above the axes automatically — the label never overlaps data
    and font sizing stays consistent across subplots.
    """
    full = rf"$\bf{{{letter}}}$" + (f" — {text}" if text else "")
    ax.set_title(full, loc="left")


def hex_to_rgba(hex_color: str, alpha: float) -> tuple[float, float, float, float]:
    """Matplotlib-friendly RGBA tuple for ``(hex, alpha)``."""
    r, g, b, _ = to_rgba(hex_color)
    return (r, g, b, alpha)


def stack_bar_label(ax, container, *, values=None, fmt: str = "{:g}",
                    min_value: float = 0.5, fontsize: int = 7) -> None:
    """Label stacked-bar segments in the center with auto-contrast text.

    ``container`` is the ``BarContainer`` returned by ``ax.bar(...)``. Segments
    whose *visual height* is below ``min_value`` are skipped. Labels flip
    between black and white based on segment luma so they stay readable on
    both light and dark fills.

    Parameters
    ----------
    values
        Optional per-bar values to render instead of the bar height — useful
        when the bar encodes a fraction but you want to show absolute counts.
        Must be the same length as ``container``.
    fmt
        Python format string for numeric values. Examples: ``"{:d}"`` for
        integer counts, ``"{:.0%}"`` for percentages, ``"{:.2f}"`` for 2-dp
        floats. Defaults to ``"{:g}"`` (matplotlib's compact default).
    min_value
        Skip segments whose visual bar height is below this. Prevents tiny
        segments from being labelled with text that overflows into neighbours.
    """
    if values is not None and len(values) != len(container):
        raise ValueError("len(values) must equal len(container)")

    labels: list[str] = []
    colors: list[str] = []
    for i, rect in enumerate(container):
        h = rect.get_height()
        if abs(h) < min_value:
            labels.append("")
            colors.append("black")
            continue
        v = values[i] if values is not None else h
        labels.append(fmt.format(v))
        r, g, b, _ = rect.get_facecolor()
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        colors.append("white" if luma < 0.55 else "black")
    for want_color in ("black", "white"):
        subset = [lab if c == want_color else "" for lab, c in zip(labels, colors)]
        ax.bar_label(container, labels=subset, label_type="center",
                     fontsize=fontsize, color=want_color)


def dbs_color(condition: str) -> str:
    """Canonical DBS color for ``condition`` ∈ {'on', 'off'}."""
    c = condition.lower()
    if c in ("on", "dbs-on", "dbs_on"):
        return COLOR_DBS_ON
    if c in ("off", "dbs-off", "dbs_off"):
        return COLOR_DBS_OFF
    return COLOR_NS