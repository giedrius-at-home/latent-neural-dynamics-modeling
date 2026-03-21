"""
Figure C2: two stacked panels (DBS-OFF / DBS-ON), history + forecast in ms relative to t₀.

True signal only in history; PSID / DPAD / VARMA forecasts only in forecast half.
PSID ±1σ band grows as σ√m (σ from validation residual std).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.subtabs.helpers import get_channel, transpose_if_needed
from dashboard.thesis.loaders import _prepare_z_array, load_split_results
from dashboard.thesis.psid_validation_sigma import sigma_z_from_psid_val_residuals
from dashboard.thesis.specs import ThesisC2ForecastSpec
from dashboard.thesis.transforms import rmse_z, zscore_using_true_stats
from dashboard.thesis.constants import (
    COLOR_DPAD,
    COLOR_PSID,
    COLOR_VARMA,
    FONT_FAMILY,
    ThesisTheme,
    grid_color,
    paper_colors,
    true_line_color,
)

logger = logging.getLogger(__name__)

_CONTEXT = "rgba(24, 95, 165, 0.08)"
_FORECAST = "rgba(153, 60, 29, 0.07)"
_RULE = "rgba(68, 68, 65, 0.6)"
_BAND_FILL = "rgba(24, 95, 165, 0.13)"
_WIDTH_TRUE = 2.5
_WIDTH_MODEL = 1.8


def _time_first(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("Expected 2D Z future array")
    if z.shape[0] <= 8 and z.shape[0] < z.shape[1]:
        return z.T
    return z


def _fc_channel(z_true: Any, z_pred: Any, channel_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    if z_true is None or z_pred is None:
        raise ValueError("Missing Z_future_true or Z_future_pred")
    T = _time_first(np.asarray(z_true, dtype=float))
    P = _time_first(np.asarray(z_pred, dtype=float))
    if T.shape != P.shape:
        raise ValueError("Z_future_true / Z_future_pred shape mismatch")
    m, nc = T.shape[0], T.shape[1]
    if nc == 1:
        return T[:, 0].copy(), P[:, 0].copy()
    if channel_idx >= nc:
        raise ValueError("channel_idx out of range for Z_future")
    return T[:, channel_idx].copy(), P[:, channel_idx].copy()


def _infer_history_len(res: Dict[str, Any], trial_idx: int, m_fc: int) -> Optional[int]:
    zc = res.get("Z_concat_for_plot")
    if not zc or trial_idx >= len(zc) or zc[trial_idx] is None:
        return None
    a = np.asarray(zc[trial_idx])
    if a.ndim != 2:
        return None
    total = a.shape[0]
    if total > m_fc:
        return int(total - m_fc)
    return None


def _history_trace(
    res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    n_hist: int,
) -> np.ndarray:
    zt = res["Z"][trial_idx]
    z_arr, t_abs = _prepare_z_array(zt, res, trial_idx)
    true_c = get_channel(z_arr, channel_idx, t_abs)
    arr = np.asarray(true_c, dtype=float).ravel()
    n = min(len(arr), n_hist)
    return arr[:n].copy()


def _build_one_panel(
    res_p: Dict[str, Any],
    res_d: Dict[str, Any],
    res_v: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    spec: ThesisC2ForecastSpec,
    sigma_z: Optional[float],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]]:
    """
    Returns:
      t_ms, z_true_plot,
      z_psid_plot, z_dpad_plot, z_varma_plot (nan in history),
      upper, lower (PSID band, nan outside forecast),
      rmse_psid, rmse_dpad, rmse_varma
    """
    if (
        res_p.get("Z_future_true") is None
        or res_p.get("Z_future_pred") is None
        or trial_idx >= len(res_p["Z_future_true"])
    ):
        logger.warning("PSID missing Z_future_* for trial %s", trial_idx)
        return None

    zt_fc, zp_fc = _fc_channel(
        res_p["Z_future_true"][trial_idx],
        res_p["Z_future_pred"][trial_idx],
        channel_idx,
    )
    zd_fc, zdp_fc = _fc_channel(
        res_d["Z_future_true"][trial_idx],
        res_d["Z_future_pred"][trial_idx],
        channel_idx,
    )
    zv_fc, zvp_fc = _fc_channel(
        res_v["Z_future_true"][trial_idx],
        res_v["Z_future_pred"][trial_idx],
        channel_idx,
    )

    m = min(zt_fc.size, zp_fc.size, zd_fc.size, zdp_fc.size, zv_fc.size, zvp_fc.size)
    if m == 0:
        return None

    n_fc_cap = int(round(spec.forecast_ms / 1000.0 * spec.sampling_hz))
    m = min(m, n_fc_cap)
    zt_fc, zp_fc = zt_fc[:m], zp_fc[:m]
    zd_fc, zdp_fc = zd_fc[:m], zdp_fc[:m]
    zv_fc, zvp_fc = zv_fc[:m], zvp_fc[:m]

    m_fc = m
    h_infer = _infer_history_len(res_p, trial_idx, m_fc)
    n_hist_cap = int(round(spec.history_ms / 1000.0 * spec.sampling_hz))
    if h_infer is not None:
        n_hist = min(h_infer, n_hist_cap)
    else:
        n_hist = n_hist_cap

    th = _history_trace(res_p, trial_idx, channel_idx, n_hist)
    n_hist = len(th)

    dt_ms = 1000.0 / spec.sampling_hz
    t_hist = np.linspace(-spec.history_ms, 0.0, n_hist, endpoint=False)

    t_fc = np.arange(m_fc, dtype=float) * dt_ms
    t_fc = np.minimum(t_fc, spec.forecast_ms - 1e-6)

    true_concat = np.concatenate([th, zt_fc])
    z_true_plot = zscore_using_true_stats(true_concat, true_concat)

    def _pred_line(pred_fc: np.ndarray) -> np.ndarray:
        pred_full = np.concatenate([th, pred_fc])
        z = zscore_using_true_stats(true_concat, pred_full)
        out = z.copy()
        out[:n_hist] = np.nan
        return out

    z_psid = _pred_line(zp_fc)
    z_dpad = _pred_line(zdp_fc)
    z_varma = _pred_line(zvp_fc)

    t_full = np.concatenate([t_hist, t_fc])

    # RMSE on forecast window only (z-scored against forecast true)
    zt = zscore_using_true_stats(zt_fc, zt_fc)
    rmse_psid = rmse_z(zt, zscore_using_true_stats(zt_fc, zp_fc))
    rmse_dpad = rmse_z(zt, zscore_using_true_stats(zt_fc, zdp_fc))
    rmse_varma = rmse_z(zt, zscore_using_true_stats(zt_fc, zvp_fc))

    upper = np.full_like(z_true_plot, np.nan)
    lower = np.full_like(z_true_plot, np.nan)
    if sigma_z is not None and sigma_z > 0.0:
        z_psid_fc_z = zscore_using_true_stats(true_concat, np.concatenate([th, zp_fc]))[n_hist:]
        m_idx = np.arange(1, m_fc + 1, dtype=float)
        w = sigma_z * np.sqrt(m_idx)
        upper[n_hist:] = z_psid_fc_z + w
        lower[n_hist:] = z_psid_fc_z - w

    return (
        t_full,
        z_true_plot,
        z_psid,
        z_dpad,
        z_varma,
        upper,
        lower,
        rmse_psid,
        rmse_dpad,
        rmse_varma,
    )


def build_c2_forecast_figure(
    spec: ThesisC2ForecastSpec,
    results_root: Path,
) -> Tuple[go.Figure, str]:
    val_psid = load_split_results(
        results_root, spec.psid_variant, spec.psid_run_ts, "val"
    )
    sigma = sigma_z_from_psid_val_residuals(val_psid, spec.channel_idx)

    res_p = load_split_results(
        results_root, spec.psid_variant, spec.psid_run_ts, spec.split
    )
    res_d = load_split_results(
        results_root, spec.dpad_variant, spec.dpad_run_ts, spec.split
    )
    res_v = load_split_results(
        results_root, spec.varma_variant, spec.varma_run_ts, spec.split
    )

    if res_p is None or res_d is None or res_v is None:
        raise FileNotFoundError("Missing PSID, DPAD, or VARMA results for C2.")

    off = _build_one_panel(res_p, res_d, res_v, spec.trial_idx_off, spec.channel_idx, spec, sigma)
    on = _build_one_panel(res_p, res_d, res_v, spec.trial_idx_on, spec.channel_idx, spec, sigma)

    if off is None or on is None:
        raise ValueError(
            "C2: missing Z_future_true/Z_future_pred for one or both trials. "
            "Run forecast validation and save test parquet with forecast columns."
        )

    paper_bg, plot_bg = paper_colors(spec.theme)
    grid = grid_color(spec.theme)
    fg = true_line_color(spec.theme)
    c_true = true_line_color(spec.theme)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.08,
    )

    y_all = [off[1], on[1]]
    for rowdata in (off, on):
        for j in (2, 3, 4):
            a = rowdata[j]
            y_all.append(a[np.isfinite(a)])
        u, lo = rowdata[5], rowdata[6]
        y_all.append(u[np.isfinite(u)])
        y_all.append(lo[np.isfinite(lo)])
    y_all = np.concatenate(y_all)
    y_all = y_all[np.isfinite(y_all)]
    ymax = float(np.nanmax(y_all)) if y_all.size else 1.0
    ymin = float(np.nanmin(y_all)) if y_all.size else -1.0
    pad = 0.05 * (ymax - ymin + 1e-6)
    y0 = ymin - pad
    y1 = ymax + pad

    def _add_panel(
        row: int,
        rowdata: Tuple,
        badge: str,
        badge_color: str,
        letter: str,
    ) -> None:
        yref_d = f"y{row} domain"
        (
            t_full,
            z_true,
            z_psid,
            z_dpad,
            z_varma,
            upper,
            lower,
            r_p,
            r_d,
            r_v,
        ) = rowdata

        # Shading: context [-horizon, 0], forecast [0, horizon]
        fig.add_vrect(
            x0=-spec.history_ms,
            x1=0.0,
            fillcolor=_CONTEXT,
            layer="below",
            line_width=0,
            row=row,
            col=1,
        )
        fig.add_vrect(
            x0=0.0,
            x1=spec.forecast_ms,
            fillcolor=_FORECAST,
            layer="below",
            line_width=0,
            row=row,
            col=1,
        )
        fig.add_vline(
            x=0.0,
            line=dict(color=RULE, width=1, dash="dash"),
            row=row,
            col=1,
        )

        xb = np.concatenate([t_full, t_full[::-1]])
        yb = np.concatenate([upper, lower[::-1]])
        yb = np.where(np.isfinite(yb), yb, np.nan)
        if np.any(np.isfinite(yb)):
            fig.add_trace(
                go.Scatter(
                    x=xb,
                    y=yb,
                    fill="toself",
                    fillcolor=_BAND_FILL,
                    mode="lines",
                    line=dict(width=0),
                    name="PSID ±1σ (∝√m)" if row == 1 else None,
                    legendgroup="band",
                    showlegend=(row == 1),
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=z_true,
                mode="lines",
                name="True (z)",
                legendgroup="true",
                line=dict(color=c_true, width=_WIDTH_TRUE),
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=z_psid,
                mode="lines",
                name="PSID",
                legendgroup="psid",
                line=dict(color=COLOR_PSID, width=_WIDTH_MODEL),
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=z_dpad,
                mode="lines",
                name="DPAD-RNN",
                legendgroup="dpad",
                line=dict(color=COLOR_DPAD, width=_WIDTH_MODEL),
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_full,
                y=z_varma,
                mode="lines",
                name="VARMA",
                legendgroup="varma",
                line=dict(color=COLOR_VARMA, width=_WIDTH_MODEL, dash="dash"),
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )

        fig.add_annotation(
            x=spec.forecast_ms * 0.5,
            y=1.05,
            xref="x1",
            yref=yref_d,
            text="forecast",
            showarrow=False,
            font=dict(size=9, color="rgba(180,180,180,0.95)"),
            xanchor="center",
        )

        fig.add_annotation(
            x=0.02,
            y=0.97,
            xref="x1 domain",
            yref=yref_d,
            text=f"<b>{letter}</b>",
            showarrow=False,
            font=dict(size=11, color=fg),
            xanchor="left",
            yanchor="top",
        )

        fig.add_annotation(
            x=0.02,
            y=0.88,
            xref="x1 domain",
            yref=yref_d,
            text=f"<b>{badge}</b> · <b>{int(spec.forecast_ms)} ms</b>",
            showarrow=False,
            font=dict(size=10, color=badge_color),
            xanchor="left",
            yanchor="top",
        )

        def _fmt_rmse(x: float) -> str:
            return f"{x:.2f}" if np.isfinite(x) else "—"

        y_ann = y0 + 0.88 * (y1 - y0)
        fig.add_annotation(
            x=spec.forecast_ms * 0.92,
            y=y_ann,
            xref="x1",
            yref=f"y{row}",
            text=(
                f"<span style='color:{COLOR_PSID}'>PSID {_fmt_rmse(r_p)}</span> · "
                f"<span style='color:{COLOR_DPAD}'>DPAD {_fmt_rmse(r_d)}</span> · "
                f"<span style='color:{COLOR_VARMA}'>VARMA {_fmt_rmse(r_v)}</span>"
            ),
            showarrow=False,
            font=dict(size=10),
            xanchor="right",
            yanchor="top",
        )

        fig.add_annotation(
            x=0.0,
            y=1.12,
            xref="x1",
            yref=yref_d,
            text="t₀ (forecast onset)",
            showarrow=False,
            font=dict(size=9, color=fg),
            xanchor="center",
        )

    _add_panel(
        1,
        off,
        "DBS-OFF",
        "#7B68C4",
        "a",
    )
    _add_panel(
        2,
        on,
        "DBS-ON",
        "#5CB85C",
        "b",
    )

    tickvals = [
        int(-spec.history_ms),
        int(-spec.history_ms / 2),
        0,
        int(spec.forecast_ms / 2),
        int(spec.forecast_ms),
    ]
    fig.update_xaxes(
        title_text="time relative to t₀ (ms)",
        tickmode="array",
        tickvals=tickvals,
        range=[-spec.history_ms - 50, spec.forecast_ms + 50],
        showgrid=True,
        gridcolor=grid,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="",
        tickmode="array",
        tickvals=tickvals,
        range=[-spec.history_ms - 50, spec.forecast_ms + 50],
        showgrid=True,
        gridcolor=grid,
        showticklabels=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=spec.y_axis_label,
        range=[y0, y1],
        showgrid=True,
        gridcolor=grid,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[y0, y1],
        showgrid=True,
        gridcolor=grid,
        row=2,
        col=1,
    )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg),
        margin=dict(l=70, r=40, t=50, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        hovermode="x unified",
    )

    caption = spec.caption or (
        f"Participant {spec.participant_label}. Same trial indices as Figure A1 (OFF={spec.trial_idx_off}, ON={spec.trial_idx_on}). "
        "±1σ PSID band uses validation residual σ; not a Bayesian posterior."
    )
    return fig, caption
