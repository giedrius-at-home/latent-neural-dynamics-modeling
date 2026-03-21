"""Latent phase space: PSID x₁ vs x₂ and DPAD PC1 vs PC2 (small multiples, KDE + trajectories)."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import FONT_FAMILY, ThesisTheme, grid_color, paper_colors, true_line_color
from dashboard.thesis.latent_phase_space_data import kde_density_grid, load_panel_latent_data
from dashboard.thesis.specs import ThesisLatentPhaseSpec

logger = logging.getLogger(__name__)

_OFF_RGB = (83, 74, 183)
_ON_RGB = (15, 110, 86)


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

    a0, a1, a2 = 0.0, 0.18, 0.32
    n_lo = norm(t_lo)
    n_hi = norm(t_hi)
    return (
        [
            [0.0, _rgba(r, g, b, a0)],
            [n_lo, _rgba(r, g, b, a0)],
            [min(n_lo + 1e-6, 1.0), _rgba(r, g, b, a1)],
            [n_hi, _rgba(r, g, b, a1)],
            [min(n_hi + 1e-6, 1.0), _rgba(r, g, b, a2)],
            [1.0, _rgba(r, g, b, a2 + 0.05)],
        ],
        zmin,
        zmax,
    )


def _add_kde(
    fig: Figure,
    row: int,
    col: int,
    x: np.ndarray,
    y: np.ndarray,
    grid_n: int,
    p_lo: float,
    p_hi: float,
    rgb: Tuple[int, int, int],
    showlegend: bool,
    name: str,
) -> None:
    if len(x) < 2:
        return
    try:
        xi, yi, zi = kde_density_grid(x, y, grid_n)
    except Exception as e:
        logger.warning("KDE skipped: %s", e)
        return
    flat = zi[np.isfinite(zi)].ravel()
    if flat.size == 0:
        return
    t_lo = float(np.percentile(flat, p_lo))
    t_hi = float(np.percentile(flat, p_hi))
    cs, zmin, zmax = _kde_colorscale(zi, t_lo, t_hi, rgb)
    r, g, b = rgb
    fig.add_trace(
        go.Contour(
            x=xi,
            y=yi,
            z=zi,
            zmin=zmin,
            zmax=zmax,
            colorscale=cs,
            showscale=False,
            line=dict(width=0),
            hoverinfo="skip",
            name=name,
            legendgroup=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def _empty_cell(fig: Figure, row: int, col: int) -> None:
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(opacity=0), showlegend=False),
        row=row,
        col=col,
    )
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, row=row, col=col)


def _extend_range(
    xr: Tuple[float, float],
    yr: Tuple[float, float],
    trajs: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_lo, x_hi = xr
    y_lo, y_hi = yr
    for tx, ty in trajs:
        if len(tx) == 0:
            continue
        x_lo = min(x_lo, float(np.nanmin(tx)))
        x_hi = max(x_hi, float(np.nanmax(tx)))
        y_lo = min(y_lo, float(np.nanmin(ty)))
        y_hi = max(y_hi, float(np.nanmax(ty)))
    px = 0.05 * (x_hi - x_lo + 1e-9)
    py = 0.05 * (y_hi - y_lo + 1e-9)
    return (x_lo - px, x_hi + px), (y_lo - py, y_hi + py)


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

    row_titles: List[str] = []
    for r in rows_spec:
        row_titles.append(f"{r.participant_label} · PSID")
    for r in rows_spec:
        row_titles.append(f"{r.participant_label} · DPAD-RNN")

    fig = make_subplots(
        rows=n_rows,
        cols=max_cols,
        subplot_titles=subplot_titles[: n_rows * max_cols] if subplot_titles else None,
        vertical_spacing=0.06,
        horizontal_spacing=0.07,
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

    first_psid_panel = (1, 1)

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
                xr, yr = data.x_range_psid, data.y_range_psid
                xr, yr = _extend_range(
                    xr,
                    yr,
                    data.traj_psid_off + data.traj_psid_on,
                )
                if len(data.x_psid_off) >= 2:
                    _add_kde(
                        fig,
                        ri,
                        ci,
                        data.x_psid_off,
                        data.y_psid_off,
                        spec.kde_grid,
                        p_lo,
                        p_hi,
                        _OFF_RGB,
                        (ri, ci) == first_psid_panel,
                        "DBS-OFF (density)",
                    )
                if len(data.x_psid_on) >= 2:
                    _add_kde(
                        fig,
                        ri,
                        ci,
                        data.x_psid_on,
                        data.y_psid_on,
                        spec.kde_grid,
                        p_lo,
                        p_hi,
                        _ON_RGB,
                        (ri, ci) == first_psid_panel,
                        "DBS-ON (density)",
                    )
                for tx, ty in data.traj_psid_off:
                    fig.add_trace(
                        go.Scatter(
                            x=tx,
                            y=ty,
                            mode="lines",
                            line=dict(color=_rgba(*_OFF_RGB, 0.95), width=1.0),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=ri,
                        col=ci,
                    )
                for tx, ty in data.traj_psid_on:
                    fig.add_trace(
                        go.Scatter(
                            x=tx,
                            y=ty,
                            mode="lines",
                            line=dict(color=_rgba(*_ON_RGB, 0.95), width=1.0),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=ri,
                        col=ci,
                    )
                fig.update_xaxes(range=list(xr), row=ri, col=ci)
                fig.update_yaxes(range=list(yr), row=ri, col=ci)
                if ci == 1:
                    fig.update_yaxes(title_text="x₂", row=ri, col=ci)
                if ri == n_p:
                    fig.update_xaxes(title_text="x₁", row=ri, col=ci)
            else:
                xr, yr = data.x_range_dpad, data.y_range_dpad
                xr, yr = _extend_range(
                    xr,
                    yr,
                    data.traj_dpad_off + data.traj_dpad_on,
                )
                if len(data.x_dpad_off) >= 2:
                    _add_kde(
                        fig,
                        ri,
                        ci,
                        data.x_dpad_off,
                        data.y_dpad_off,
                        spec.kde_grid,
                        p_lo,
                        p_hi,
                        _OFF_RGB,
                        False,
                        "DBS-OFF (density)",
                    )
                if len(data.x_dpad_on) >= 2:
                    _add_kde(
                        fig,
                        ri,
                        ci,
                        data.x_dpad_on,
                        data.y_dpad_on,
                        spec.kde_grid,
                        p_lo,
                        p_hi,
                        _ON_RGB,
                        False,
                        "DBS-ON (density)",
                    )
                for tx, ty in data.traj_dpad_off:
                    fig.add_trace(
                        go.Scatter(
                            x=tx,
                            y=ty,
                            mode="lines",
                            line=dict(color=_rgba(*_OFF_RGB, 0.95), width=1.0),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=ri,
                        col=ci,
                    )
                for tx, ty in data.traj_dpad_on:
                    fig.add_trace(
                        go.Scatter(
                            x=tx,
                            y=ty,
                            mode="lines",
                            line=dict(color=_rgba(*_ON_RGB, 0.95), width=1.0),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=ri,
                        col=ci,
                    )
                v1, v2 = data.pca_variance_ratio
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
                    fig.update_yaxes(title_text="PC2 of x⁽¹⁾", row=ri, col=ci)
                if ri == n_rows:
                    fig.update_xaxes(title_text="PC1 of x⁽¹⁾", row=ri, col=ci)

    for ri in range(1, n_rows + 1):
        for ci in range(1, max_cols + 1):
            fig.update_xaxes(
                showticklabels=(ri == n_p or ri == n_rows),
                gridcolor=grid,
                zeroline=False,
                row=ri,
                col=ci,
            )
            fig.update_yaxes(
                showticklabels=(ci == 1),
                gridcolor=grid,
                zeroline=False,
                row=ri,
                col=ci,
            )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg, size=11),
        margin=dict(l=100, r=40, t=90, b=60),
        title=dict(text=spec.section_title, font=dict(size=14)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )

    cap = spec.caption or ""
    return fig, cap
