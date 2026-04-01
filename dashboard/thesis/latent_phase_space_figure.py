"""Latent phase space: PSID x₁ vs x₂ and DPAD PC1 vs PC2 (small multiples, KDE + trajectories)."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_TICK,
    ThesisTheme,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.latent_phase_space_data import kde_on_fixed_grid, load_panel_latent_data
from dashboard.thesis.specs import ThesisLatentPhaseSpec

logger = logging.getLogger(__name__)

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

_OFF_RGB = _hex_to_rgb(COLOR_DBS_OFF)
_ON_RGB = _hex_to_rgb(COLOR_DBS_ON)


def _rgba(r: int, g: int, b: int, a: float) -> str:
    return f"rgba({r},{g},{b},{a})"


def _kde_colorscale(
    z: np.ndarray,
    t_lo: float,
    t_hi: float,
    rgb: Tuple[int, int, int],
) -> Tuple[List[List], float, float]:
    """Piecewise colorscale: transparent below t_lo, mid opacity between, higher above t_hi."""
    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    span = zmax - zmin + 1e-15
    r, g, b = rgb

    def norm(t: float) -> float:
        return float(np.clip((t - zmin) / span, 0.0, 1.0))

    a0, a1, a2 = 0.0, 0.38, 0.62
    n_lo = norm(t_lo)
    n_hi = norm(t_hi)
    return (
        [
            [0.0, _rgba(r, g, b, a0)],
            [n_lo, _rgba(r, g, b, a0)],
            [min(n_lo + 1e-6, 1.0), _rgba(r, g, b, a1)],
            [n_hi, _rgba(r, g, b, a1)],
            [min(n_hi + 1e-6, 1.0), _rgba(r, g, b, a2)],
            [1.0, _rgba(r, g, b, a2 + 0.08)],
        ],
        zmin,
        zmax,
    )


def _common_xy_mesh(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    grid_n: int,
    pad: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.concatenate([np.asarray(a, dtype=float).ravel() for a, _ in pairs])
    ys = np.concatenate([np.asarray(b, dtype=float).ravel() for _, b in pairs])
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if xs.size == 0:
        return np.linspace(0, 1, grid_n), np.linspace(0, 1, grid_n)
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    dx = xmax - xmin + 1e-9
    dy = ymax - ymin + 1e-9
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    return np.linspace(xmin, xmax, grid_n), np.linspace(ymin, ymax, grid_n)


def _add_kde_heatmap(
    fig: Figure,
    row: int,
    col: int,
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
    p_lo: float,
    p_hi: float,
    rgb: Tuple[int, int, int],
    showlegend: bool,
    name: str,
    opacity: float,
) -> None:
    flat = zi[np.isfinite(zi)].ravel()
    if flat.size == 0:
        return
    t_lo = float(np.percentile(flat, p_lo))
    t_hi = float(np.percentile(flat, p_hi))
    cs, zmin, zmax = _kde_colorscale(zi, t_lo, t_hi, rgb)
    fig.add_trace(
        go.Heatmap(
            x=xi,
            y=yi,
            z=zi,
            zmin=zmin,
            zmax=zmax,
            colorscale=cs,
            showscale=False,
            opacity=opacity,
            hoverinfo="skip",
            name=name,
            legendgroup=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def _add_dual_kde_heatmaps(
    fig: Figure,
    row: int,
    col: int,
    x_off: np.ndarray,
    y_off: np.ndarray,
    x_on: np.ndarray,
    y_on: np.ndarray,
    grid_n: int,
    p_lo: float,
    p_hi: float,
    showlegend_off: bool,
    showlegend_on: bool,
) -> None:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    if len(x_off) >= 2:
        pairs.append((x_off, y_off))
    if len(x_on) >= 2:
        pairs.append((x_on, y_on))
    if not pairs:
        return
    xi, yi = _common_xy_mesh(pairs, grid_n)
    if len(x_off) >= 2:
        zo = kde_on_fixed_grid(x_off, y_off, xi, yi)
        _add_kde_heatmap(
            fig, row, col, xi, yi, zo, p_lo, p_hi, _OFF_RGB,
            False, "DBS-OFF (density)", 0.52,
        )
    if len(x_on) >= 2:
        zn = kde_on_fixed_grid(x_on, y_on, xi, yi)
        _add_kde_heatmap(
            fig, row, col, xi, yi, zn, p_lo, p_hi, _ON_RGB,
            False, "DBS-ON (density)", 0.55,
        )


def _empty_cell(fig: Figure, row: int, col: int) -> None:
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(opacity=0), showlegend=False),
        row=row,
        col=col,
    )
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, row=row, col=col)


def _xy_range_from_points(
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    pad: float = 0.05,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Axis limits from scattered samples only (no trajectory polylines)."""
    xa_parts: List[np.ndarray] = []
    ya_parts: List[np.ndarray] = []
    for xa, ya in zip(xs, ys):
        xa = np.asarray(xa, dtype=float).ravel()
        ya = np.asarray(ya, dtype=float).ravel()
        m = np.isfinite(xa) & np.isfinite(ya)
        if np.any(m):
            xa_parts.append(xa[m])
            ya_parts.append(ya[m])
    if not xa_parts:
        return (0.0, 1.0), (0.0, 1.0)
    xa = np.concatenate(xa_parts)
    ya = np.concatenate(ya_parts)
    xmin, xmax = float(np.min(xa)), float(np.max(xa))
    ymin, ymax = float(np.min(ya)), float(np.max(ya))
    dx = xmax - xmin + 1e-9
    dy = ymax - ymin + 1e-9
    return (xmin - pad * dx, xmax + pad * dx), (ymin - pad * dy, ymax + pad * dy)


def build_latent_phase_space_figure(
    spec: ThesisLatentPhaseSpec,
    results_root,
) -> Tuple[Figure, str]:
    rows_spec = list(spec.rows)
    n_p = len(rows_spec)
    if n_p == 0:
        raise ValueError("ThesisLatentPhaseSpec.rows is empty")
    max_cols = max(len(r.panels) for r in rows_spec)
    n_rows = 2 * n_p
    theme = spec.theme
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    p_lo, p_hi = spec.density_percentiles

    subplot_titles: List[str | None] = []
    for _ in range(2):
        for r in rows_spec:
            for c in range(max_cols):
                if c < len(r.panels):
                    subplot_titles.append(r.panels[c].session_label)
                else:
                    subplot_titles.append("")

    row_titles = [f"{r.participant_label} · PSID" for r in rows_spec] + [
        f"{r.participant_label} · DPAD" for r in rows_spec
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=max_cols,
        subplot_titles=subplot_titles[: n_rows * max_cols] if subplot_titles else None,
        vertical_spacing=0.07,
        horizontal_spacing=0.09,
        row_titles=row_titles,
    )

    cache: dict[tuple[int, int], PanelLatentData] = {}
    for pi, rspec in enumerate(rows_spec):
        for ci, panel in enumerate(rspec.panels):
            cache[(pi, ci)] = load_panel_latent_data(
                results_root,
                panel,
                spec.split,
                spec.n_psid_latent,
                spec.n_dpad_latent,
                spec.n_trajectory_trials,
                spec.trajectory_seed,
            )

    for ri in range(1, n_rows + 1):
        pi = (ri - 1) if (ri - 1) < n_p else (ri - 1 - n_p)
        is_psid = (ri - 1) < n_p
        rspec = rows_spec[pi]

        for ci in range(1, max_cols + 1):
            if ci > len(rspec.panels):
                _empty_cell(fig, ri, ci)
                continue

            data = cache[(pi, ci - 1)]

            if is_psid:
                xr, yr = _xy_range_from_points(
                    [data.x_psid_off, data.x_psid_on],
                    [data.y_psid_off, data.y_psid_on],
                )
                _add_dual_kde_heatmaps(
                    fig,
                    ri,
                    ci,
                    data.x_psid_off,
                    data.y_psid_off,
                    data.x_psid_on,
                    data.y_psid_on,
                    spec.kde_grid,
                    p_lo,
                    p_hi,
                    False,
                    False,
                )
                fig.update_xaxes(range=list(xr), row=ri, col=ci)
                fig.update_yaxes(range=list(yr), row=ri, col=ci)
                if ci == 1:
                    fig.update_yaxes(title_text="x₂", row=ri, col=ci)
                if ri == n_p:
                    fig.update_xaxes(title_text="x₁", row=ri, col=ci)
            else:
                xr, yr = _xy_range_from_points(
                    [data.x_dpad_off, data.x_dpad_on],
                    [data.y_dpad_off, data.y_dpad_on],
                )
                _add_dual_kde_heatmaps(
                    fig,
                    ri,
                    ci,
                    data.x_dpad_off,
                    data.y_dpad_off,
                    data.x_dpad_on,
                    data.y_dpad_on,
                    spec.kde_grid,
                    p_lo,
                    p_hi,
                    False,
                    False,
                )
                v1, v2 = data.pca_variance_ratio
                if v1 > 1e-9 or v2 > 1e-9:
                    fig.add_annotation(
                        text=f"PC1: {v1:.0%} · PC2: {v2:.0%}",
                        xref="x domain",
                        yref="y domain",
                        x=0.98,
                        y=0.02,
                        xanchor="right",
                        yanchor="bottom",
                        showarrow=False,
                        font=dict(size=9, color=fg, family=FONT_FAMILY),
                        row=ri,
                        col=ci,
                    )
                fig.update_xaxes(range=list(xr), row=ri, col=ci)
                fig.update_yaxes(range=list(yr), row=ri, col=ci)
                if ci == 1:
                    fig.update_yaxes(title_text="x₂", row=ri, col=ci)
                if ri == n_rows:
                    fig.update_xaxes(title_text="x₁", row=ri, col=ci)

    for ri in range(1, n_rows + 1):
        for ci in range(1, max_cols + 1):
            fig.update_xaxes(
                showticklabels=(ri == n_p or ri == n_rows),
                gridcolor=grid,
                zeroline=False,
                showline=True, linecolor=fg, linewidth=1,
                tickfont=dict(size=FONT_SIZE_TICK),
                row=ri,
                col=ci,
            )
            fig.update_yaxes(
                showticklabels=(ci == 1),
                gridcolor=grid,
                zeroline=False,
                showline=True, linecolor=fg, linewidth=1,
                tickfont=dict(size=FONT_SIZE_TICK),
                row=ri,
                col=ci,
            )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg, size=FONT_SIZE_BASE),
        height=max(240 * n_rows + 80, 600),
        margin=dict(l=132, r=48, t=56, b=64),
        showlegend=False,
    )

    cap = spec.caption or ""
    return fig, cap
