"""Two-panel forecast RMSE vs horizon (DBS-OFF / DBS-ON): PSID, VARMA, naive baseline."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_PSID,
    COLOR_VARMA,
    FONT_FAMILY,
    ThesisTheme,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.forecast_horizon_rmse import ForecastHorizonRmseData

_NAIVE = "#444441"
_BADGE_OFF = "#7B68C4"
_BADGE_ON = "#5CB85C"
_REF_LINE = "rgba(180,180,180,0.55)"
_REF_TEXT = "rgba(200,200,200,0.9)"


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _add_sem_band(
    fig: Figure,
    x: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    fillcolor: str,
    row: int,
    col: int,
    name: str | None,
    showlegend: bool,
) -> None:
    """Closed polygon (toself) so bands do not chain via tonexty across models."""
    upper = mean + sem
    lower = mean - sem
    xb = np.concatenate([x, x[::-1]])
    yb = np.concatenate([upper, lower[::-1]])
    fig.add_trace(
        go.Scatter(
            x=xb,
            y=yb,
            fill="toself",
            fillcolor=fillcolor,
            mode="lines",
            line=dict(width=0),
            showlegend=showlegend,
            name=name,
            hoverinfo="skip",
            legendgroup=name or "sem",
        ),
        row=row,
        col=col,
    )


def _add_model_line(
    fig: Figure,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    name: str,
    dash: str | None,
    symbol: str,
    row: int,
    col: int,
    showlegend: bool,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2, dash=dash or "solid"),
            marker=dict(
                color=color,
                size=7,
                symbol=symbol,
                line=dict(width=0),
            ),
            showlegend=showlegend,
            legendgroup=name,
        ),
        row=row,
        col=col,
    )


def _add_vline(fig: Figure, x0: float, row: int, col: int) -> None:
    fig.add_shape(
        type="line",
        x0=x0,
        x1=x0,
        y0=0,
        y1=1,
        yref="y domain",
        line=dict(color=_REF_LINE, width=1, dash="dot"),
        row=row,
        col=col,
    )


def build_forecast_rmse_figure(
    data: ForecastHorizonRmseData,
    theme: ThesisTheme,
    one_step_ms: float = 1000.0 / 60.0,
    x_max_ms: float = 1000.0,
) -> Figure:
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_y=True,
        horizontal_spacing=0.1,
    )

    ymax = max(float(data.naive_rmse) * 1.12, 1.05)
    for arr in (
        data.mean_psid_off,
        data.mean_varma_off,
        data.mean_psid_on,
        data.mean_varma_on,
    ):
        if arr.size:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(finite)) * 1.08)

    # tighten ymax from mean+sem where present
    for m, s in (
        (data.mean_psid_off, data.sem_psid_off),
        (data.mean_varma_off, data.sem_varma_off),
        (data.mean_psid_on, data.sem_psid_on),
        (data.mean_varma_on, data.sem_varma_on),
    ):
        if m.size and s.size and m.shape == s.shape:
            u = m + s
            u = u[np.isfinite(u)]
            if u.size:
                ymax = max(ymax, float(np.nanmax(u)) * 1.05)

    if not np.isfinite(ymax):
        ymax = 1.1

    x = data.x_ms
    naive_y = np.full_like(x, data.naive_rmse, dtype=float)

    def _panel(
        col: int,
        mean_p: np.ndarray,
        sem_p: np.ndarray,
        mean_v: np.ndarray,
        sem_v: np.ndarray,
        badge: str,
        badge_color: str,
        crossover: float | None,
    ) -> None:
        row = 1
        xaxis = "x1" if col == 1 else "x2"
        if x.size == 0:
            return
        _add_sem_band(
            fig,
            x,
            mean_p,
            sem_p,
            _hex_rgba(COLOR_PSID, 0.15),
            row,
            col,
            "±1 SEM (PSID)",
            showlegend=(col == 1),
        )
        _add_sem_band(
            fig,
            x,
            mean_v,
            sem_v,
            _hex_rgba(COLOR_VARMA, 0.15),
            row,
            col,
            "±1 SEM (VARMA)",
            showlegend=(col == 1),
        )
        _add_model_line(
            fig, x, mean_p, COLOR_PSID, "PSID", None, "circle", row, col, showlegend=(col == 1)
        )
        _add_model_line(
            fig,
            x,
            mean_v,
            COLOR_VARMA,
            "VARMA",
            "dash",
            "square",
            row,
            col,
            showlegend=(col == 1),
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=naive_y,
                mode="lines",
                name="Naïve (mean prediction)",
                line=dict(color=_NAIVE, width=1.2, dash="dot"),
                showlegend=(col == 1),
                legendgroup="naive",
                hovertemplate="%{y:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

        for xv in (250.0, 500.0):
            if xv <= x_max_ms:
                _add_vline(fig, xv, row, col)
                fig.add_annotation(
                    x=xv,
                    y=1.0,
                    xref=xaxis,
                    yref="y1 domain",
                    text=f"{int(xv)} ms",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    font=dict(size=9, color=_REF_TEXT),
                )

        if crossover is not None and 0 < crossover < x_max_ms:
            fig.add_shape(
                type="line",
                x0=crossover,
                x1=crossover,
                y0=0,
                y1=1,
                yref="y1 domain",
                line=dict(color="rgba(255,200,120,0.7)", width=1, dash="dash"),
                row=row,
                col=col,
            )
            fig.add_annotation(
                x=crossover,
                y=0.15,
                xref=xaxis,
                yref="y1 domain",
                text=f"crossover (~{crossover:.0f} ms)",
                showarrow=False,
                font=dict(size=9, color=_REF_TEXT),
                xanchor="center",
            )

        fig.add_annotation(
            x=0.04,
            y=0.93,
            xref=f"{xaxis} domain",
            yref="y1 domain",
            text=f"<b>{badge}</b>",
            showarrow=False,
            font=dict(size=11, color=badge_color),
            align="left",
            bgcolor="rgba(0,0,0,0.35)",
            borderpad=4,
        )

        fig.add_annotation(
            x=one_step_ms,
            y=0.02,
            xref=xaxis,
            yref="y1 domain",
            text="← 1-step",
            showarrow=False,
            font=dict(size=9, color=fg),
            xanchor="left",
        )

    _panel(
        1,
        data.mean_psid_off,
        data.sem_psid_off,
        data.mean_varma_off,
        data.sem_varma_off,
        "DBS-OFF",
        _BADGE_OFF,
        data.crossover_ms_off,
    )
    _panel(
        2,
        data.mean_psid_on,
        data.sem_psid_on,
        data.mean_varma_on,
        data.sem_varma_on,
        "DBS-ON",
        _BADGE_ON,
        data.crossover_ms_on,
    )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg),
        margin=dict(l=70, r=100, t=40, b=100),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(
        title_text="forecast horizon (ms)",
        range=[0, x_max_ms],
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        tickmode="array",
        tickvals=[0, 250, 500, 750, 1000],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="forecast horizon (ms)",
        range=[0, x_max_ms],
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        tickmode="array",
        tickvals=[0, 250, 500, 750, 1000],
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="RMSE (z-scored tracing speed)",
        range=[0, ymax],
        showgrid=True,
        gridcolor=grid,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showticklabels=True,
        range=[0, ymax],
        showgrid=True,
        gridcolor=grid,
        row=1,
        col=2,
    )

    return fig


def build_forecast_rmse_figure_or_empty(
    data: ForecastHorizonRmseData | None,
    theme: ThesisTheme,
) -> Figure:
    if data is None or data.x_ms.size == 0:
        paper_bg, plot_bg = paper_colors(theme)
        fg = true_line_color(theme)
        fig = go.Figure()
        fig.add_annotation(
            text="No forecast data (need Z_future_true / Z_future_pred in test parquet for PSID and VARMA).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=fg, family=FONT_FAMILY),
        )
        fig.update_layout(paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
        return fig
    return build_forecast_rmse_figure(data, theme)
