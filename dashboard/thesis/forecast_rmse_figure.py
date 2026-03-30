"""Two-panel forecast RMSE vs horizon (DBS-OFF / DBS-ON): PSID, VARMA, naive baseline."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_PSID,
    COLOR_SEPARATOR,
    COLOR_VARMA,
    FIGURE_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    dbs_badge_style,
    grid_color,
    legend_bgcolor,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.forecast_horizon_rmse import ForecastHorizonRmseData
from dashboard.thesis.plot_scaffolds import FORECAST_HORIZON_AXIS_TITLE

_NAIVE = "#444441"
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
            line=dict(color=color, width=1.8, dash=dash or "solid"),
            marker=dict(
                color=color,
                size=6,
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
    y_axis_title: str | None = None,
) -> Figure:
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)
    y_title = y_axis_title or "RMSE (z-scored tracing speed)"

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
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
        _unused: object,
        crossover: float | None,
    ) -> None:
        row = 1
        xaxis = "x" if col == 1 else "x2"
        yaxis_d = "y domain"
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
                    yref=yaxis_d,
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
                yref=yaxis_d,
                line=dict(color="rgba(255,200,120,0.7)", width=1, dash="dash"),
                row=row,
                col=col,
            )
            fig.add_annotation(
                x=crossover,
                y=0.15,
                xref=xaxis,
                yref=yaxis_d,
                text=f"crossover (~{crossover:.0f} ms)",
                showarrow=False,
                font=dict(size=9, color=_REF_TEXT),
                xanchor="center",
            )

        badge_fg, badge_bg = dbs_badge_style(badge)
        fig.add_annotation(
            x=0.04,
            y=0.93,
            xref=f"{xaxis} domain",
            yref=yaxis_d,
            text=f"<b>{badge}</b>",
            showarrow=False,
            font=dict(size=FONT_SIZE_BASE, color=badge_fg, family=FONT_FAMILY),
            align="left",
            bgcolor=badge_bg,
            bordercolor=badge_fg,
            borderwidth=1,
            borderpad=4,
        )

        fig.add_annotation(
            x=one_step_ms,
            y=0.02,
            xref=xaxis,
            yref=yaxis_d,
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
        None,
        data.crossover_ms_off,
    )
    _panel(
        2,
        data.mean_psid_on,
        data.sem_psid_on,
        data.mean_varma_on,
        data.sem_varma_on,
        "DBS-ON",
        None,
        data.crossover_ms_on,
    )

    _xkw = dict(
        range=[0, x_max_ms],
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        tickmode="array",
        tickvals=[0, 250, 500, 750, 1000],
        showline=True,
        linecolor=fg,
        linewidth=1,
        mirror=False,
        tickfont=dict(size=FONT_SIZE_TICK),
        title_text=FORECAST_HORIZON_AXIS_TITLE,
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
    )
    fig.update_xaxes(**_xkw, row=1, col=1)
    fig.update_xaxes(**_xkw, row=1, col=2)

    _ykw = dict(
        range=[0, ymax],
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linecolor=fg,
        linewidth=1,
        mirror=False,
        tickfont=dict(size=FONT_SIZE_TICK),
    )
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        **_ykw,
        row=1, col=1,
    )
    fig.update_yaxes(showticklabels=True, **_ykw, row=1, col=2)

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=FIGURE_HEIGHT,
        margin=dict(l=70, r=40, t=36, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZE_TICK),
            bgcolor=legend_bgcolor(),
        ),
        hovermode="x unified",
    )

    return fig


def build_forecast_rmse_figure_or_empty(
    data: ForecastHorizonRmseData | None,
    theme: ThesisTheme,
    y_axis_title: str | None = None,
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
    return build_forecast_rmse_figure(data, theme, y_axis_title=y_axis_title)


def build_forecast_global_rmse_figure(
    data: ForecastHorizonRmseData,
    theme: ThesisTheme,
    rng: np.random.Generator,
    jitter: float = 0.12,
    y_axis_title: str | None = None,
) -> Figure:
    """Bar (mean ± SEM) + jittered trial dots for forecast RMSE at global_horizon_ms."""
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)
    y_title = y_axis_title or f"RMSE (z) at ~{data.global_horizon_ms:.0f} ms horizon"

    cells = [
        ("PSID\nOFF", data.trial_rmse_psid_off, COLOR_PSID, 0.85),
        ("PSID\nON", data.trial_rmse_psid_on, COLOR_PSID, 0.45),
        ("VARMA\nOFF", data.trial_rmse_varma_off, COLOR_VARMA, 0.85),
        ("VARMA\nON", data.trial_rmse_varma_on, COLOR_VARMA, 0.45),
    ]
    x_pos = np.arange(len(cells), dtype=float)

    means = np.array([np.mean(c[1]) if c[1] else np.nan for c in cells])
    sems = np.array([
        (np.std(c[1], ddof=1) / np.sqrt(len(c[1]))) if len(c[1]) > 1 else 0.0
        for c in cells
    ])

    def _rgba(hex_c: str, a: float) -> str:
        h = hex_c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    bar_colors = [_rgba(c[2], c[3]) for c in cells]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_pos,
            y=means,
            error_y=dict(type="data", array=sems, visible=True, thickness=1.5, color=fg),
            marker=dict(color=bar_colors, line=dict(width=0)),
            width=0.52,
            name="Mean ± SEM",
            showlegend=True,
        )
    )

    for xi, (label, vals, color, alpha) in enumerate(cells):
        if not vals:
            continue
        jt = rng.uniform(-jitter, jitter, size=len(vals))
        fig.add_trace(
            go.Scatter(
                x=x_pos[xi] + jt,
                y=vals,
                mode="markers",
                marker=dict(size=5, color=_rgba(color, alpha * 0.9), line=dict(width=0)),
                name="Test trials" if xi == 0 else None,
                showlegend=(xi == 0),
                legendgroup="dots",
            )
        )

    ymax_data = float(np.nanmax(means + sems)) if np.any(np.isfinite(means)) else 1.0
    for c in cells:
        for v in c[1]:
            if np.isfinite(v):
                ymax_data = max(ymax_data, float(v))
    y_max = max(ymax_data * 1.1, 0.5)

    fig.add_shape(
        type="line", xref="x", yref="y",
        x0=1.5, x1=1.5, y0=0, y1=y_max,
        line=dict(color=COLOR_SEPARATOR, width=0.7, dash="dash"), opacity=0.5,
    )

    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=FIGURE_HEIGHT,
        xaxis=dict(
            tickmode="array",
            tickvals=list(x_pos),
            ticktext=[c[0] for c in cells],
            title=dict(text="Model × DBS condition", font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY), standoff=14),
            showgrid=False, zeroline=False, showline=True, linecolor=fg,
            tickfont=dict(size=FONT_SIZE_TICK),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY)),
            range=[0, y_max * 1.02],
            showgrid=True, gridcolor=grid, showline=True, linecolor=fg,
            tickfont=dict(size=FONT_SIZE_TICK),
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.12,
            xanchor="center", x=0.5,
            font=dict(size=FONT_SIZE_TICK), bgcolor=legend_bgcolor(),
        ),
        margin=dict(l=72, r=32, t=36, b=140),
        hovermode="closest",
    )

    return fig
