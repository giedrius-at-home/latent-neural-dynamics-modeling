"""
Shared Plotly layout helpers for thesis figures — one place for axis styling and repeated title patterns.
"""

from __future__ import annotations

from typing import Any, Tuple

import plotly.graph_objects as go

from dashboard.thesis.constants import (
    FONT_FAMILY,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    grid_color,
    true_line_color,
)

# Shared axis labels (forecast / time-series figures)
THESIS_TIME_AXIS_TITLE = "Time (s)"
THESIS_REL_TIME_FROM_FORECAST_ONSET_TITLE = "Relative time from forecast onset (s)"
FORECAST_HORIZON_AXIS_TITLE = "Forecast horizon (ms)"


def thesis_bottom_time_axis_title(*, theme: ThesisTheme) -> dict:
    """Plotly ``title=`` dict for bottom x-axes showing absolute trial time."""
    fg = true_line_color(theme)
    return dict(
        text=THESIS_TIME_AXIS_TITLE,
        font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY, color=fg),
    )


def thesis_relative_forecast_onset_axis_title(*, theme: ThesisTheme) -> dict:
    """Plotly ``title=`` dict for C2-style mirrored x-axis (history negative, forecast positive)."""
    fg = true_line_color(theme)
    return dict(
        text=THESIS_REL_TIME_FROM_FORECAST_ONSET_TITLE,
        font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY, color=fg),
    )


def cross_block_column_title_pair(framework: str) -> Tuple[str, str]:
    """Subplot titles for one framework row: left OFF→ON stitch, right ON→OFF stitch."""
    fn = framework.strip().upper()
    return (
        f"{fn}: last OFF trial → first ON trial",
        f"{fn}: last ON trial → first OFF trial",
    )


def cross_block_subplot_title_list(frameworks_in_row_order: list[str]) -> list[str]:
    """Flat list for ``make_subplots(..., subplot_titles=...)`: FW left, FW right for each row."""
    titles: list[str] = []
    for fw in frameworks_in_row_order:
        left, right = cross_block_column_title_pair(fw)
        titles.extend([left, right])
    return titles


def apply_grid_xy_subplots(
    fig: go.Figure,
    *,
    n_rows: int,
    n_cols: int,
    theme: ThesisTheme,
    nticks: int | None = 12,
    x_title: str | None = None,
) -> None:
    """Standard grid + typography on every subplot axis (used by cross-block and similar grids)."""
    grid = grid_color(theme)
    fg = true_line_color(theme)
    tick_kwargs: dict[str, Any] = {}
    if nticks is not None:
        tick_kwargs["nticks"] = int(nticks)
    x_extra: dict[str, Any] = dict(tick_kwargs)
    if x_title:
        x_extra["title_text"] = x_title
        x_extra["title_font"] = dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg)
    for c in range(1, n_cols + 1):
        for r in range(1, n_rows + 1):
            fig.update_xaxes(
                showgrid=True,
                gridcolor=grid,
                showline=True,
                linecolor=fg,
                tickfont=dict(size=FONT_SIZE_TICK, color=fg),
                row=r,
                col=c,
                **x_extra,
            )
            fig.update_yaxes(
                showgrid=True,
                gridcolor=grid,
                showline=True,
                linecolor=fg,
                tickfont=dict(size=FONT_SIZE_TICK, color=fg),
                row=r,
                col=c,
                **tick_kwargs,
            )


def cross_block_annotation_font(*, theme: ThesisTheme) -> dict:
    fg = true_line_color(theme)
    return dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg)
