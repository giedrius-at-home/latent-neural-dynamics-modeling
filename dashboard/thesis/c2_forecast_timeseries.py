"""
Figure C2: two stacked panels (DBS-OFF / DBS-ON), aligned with the predictions/forecasting dashboard.

Bottom x-axis: absolute trial time (s). Top x-axis (overlay): time relative to forecast onset (s),
same layout as ``dashboard.backbone.add_relative_time_axis`` (0 at the first forecast sample). True signal only in history; PSID / DPAD / VARMA forecasts only in the
forecast segment. A NaN gap is inserted at the history/forecast boundary so the true trace is not drawn across the onset. Optional PSID ±1σ band when ``show_psid_sigma_band`` is true.
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
from dashboard.thesis.aggregate_rmse import _key_index_map, _trial_key, normalize_stim
from dashboard.thesis.compose import _session_mean_rmse_y_triplet, _session_mean_rmse_z_triplet
from dashboard.thesis.exemplar_trials import resolve_off_on_indices_from_spec
from dashboard.thesis.figure import ThesisPanelData, _slice_trial_center
from dashboard.thesis.loaders import (
    _prepare_z_array,
    channels_as_str_list,
    extract_trial_y_series,
    extract_trial_z_series,
    load_split_results_required,
    resolve_output_channel_display,
    neural_y_feature_label,
    resolve_neural_y_channel_idx,
    thesis_exemplar_tagline,
)
from dashboard.thesis.psid_validation_sigma import sigma_z_from_psid_val_residuals
from dashboard.thesis.specs import (
    THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
    ThesisC2ForecastSpec,
    infer_varma_off_on_run_ts,
)
from dashboard.thesis.transforms import (
    reshape_future_z_time_first,
    rmse_z,
    z_true_and_preds,
    zscore_using_true_stats,
)
from dashboard.thesis.plot_scaffolds import (
    thesis_bottom_time_axis_title,
    thesis_relative_forecast_onset_axis_title,
)
from dashboard.thesis.constants import (
    COLOR_DPAD,
    COLOR_DPAD_BAND_FILL,
    COLOR_DPAD_BAND_LINE,
    COLOR_PSID,
    COLOR_PSID_BAND_FILL,
    COLOR_PSID_BAND_LINE,
    COLOR_VARMA,
    COLOR_VARMA_BAND_FILL,
    COLOR_VARMA_BAND_LINE,
    FIGURE_HEIGHT_C2_FORECAST,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_TICK,
    ThesisTheme,
    WIDTH_DPAD,
    WIDTH_PSID,
    WIDTH_TRUE,
    WIDTH_VARMA,
    dbs_badge_style,
    grid_color,
    legend_bgcolor,
    paper_colors,
    true_line_color,
)

logger = logging.getLogger(__name__)


def _slice_c2_prediction_panel(
    panel: ThesisPanelData,
    *,
    segment_s: float,
    side_extend_s: float,
) -> ThesisPanelData:
    t, zt, zp, zd, zv = _slice_trial_center(
        panel.t_abs,
        segment_s,
        panel.z_true,
        panel.z_psid,
        panel.z_dpad,
        panel.z_varma,
        side_extend_s=side_extend_s,
    )
    if t.size == 0:
        return panel
    return ThesisPanelData(
        t_abs=t,
        z_true=zt,
        z_psid=zp,
        z_dpad=zd,
        z_varma=zv,
        psid_sigma=panel.psid_sigma,
        dbs_label=panel.dbs_label,
        band_rmse_psid=panel.band_rmse_psid,
        band_rmse_dpad=panel.band_rmse_dpad,
        band_rmse_varma=panel.band_rmse_varma,
    )


_CONTEXT = "rgba(24, 95, 165, 0.08)"
_FORECAST = "rgba(153, 60, 29, 0.07)"
_RULE = "rgba(68, 68, 65, 0.6)"


def _all_nan_series(arr: np.ndarray) -> bool:
    return np.all(np.isnan(np.asarray(arr, dtype=float)))


def _add_self_prediction_stack_row(
    fig: go.Figure,
    row: int,
    panel: ThesisPanelData,
    theme: ThesisTheme,
    *,
    show_legend: bool,
    badge_text: str,
    prediction_line_dash: Optional[str],
    true_series_name: str = "y_true",
) -> None:
    """Full-trial one-step-style row (matches A1 stacked exemplar traces)."""
    c_true = true_line_color(theme)
    t = np.asarray(panel.t_abs, dtype=float).ravel()
    zt = np.asarray(panel.z_true, dtype=float).ravel()
    zp = np.asarray(panel.z_psid, dtype=float).ravel()
    zd = np.asarray(panel.z_dpad, dtype=float).ravel()
    zv = np.asarray(panel.z_varma, dtype=float).ravel()
    _pd = prediction_line_dash

    def _band_row(
        y_hat: np.ndarray,
        half: Optional[float],
        fill_c: str,
        line_c: str,
        name: str,
        leg: str,
        show: bool,
    ) -> None:
        if half is None or not np.isfinite(half) or half <= 0 or _all_nan_series(y_hat) or len(t) != len(y_hat):
            return
        u = y_hat + half
        lo_b = y_hat - half
        xb = np.concatenate([t, t[::-1]])
        yb = np.concatenate([u, lo_b[::-1]])
        fig.add_trace(
            go.Scatter(
                x=xb,
                y=yb,
                fill="toself",
                fillcolor=fill_c,
                line=dict(color=line_c, width=0.5),
                name=name,
                legendgroup=leg,
                showlegend=show,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

    has_b = any(
        [
            panel.band_rmse_psid is not None
            and np.isfinite(panel.band_rmse_psid)
            and panel.band_rmse_psid > 0,
            panel.band_rmse_dpad is not None
            and np.isfinite(panel.band_rmse_dpad)
            and panel.band_rmse_dpad > 0,
            panel.band_rmse_varma is not None
            and np.isfinite(panel.band_rmse_varma)
            and panel.band_rmse_varma > 0,
        ]
    )
    if has_b:
        _band_row(
            zp,
            panel.band_rmse_psid,
            COLOR_PSID_BAND_FILL,
            COLOR_PSID_BAND_LINE,
            "PSID ± mean RMSE",
            "sp_psid",
            False,
        )
        _band_row(
            zd,
            panel.band_rmse_dpad,
            COLOR_DPAD_BAND_FILL,
            COLOR_DPAD_BAND_LINE,
            "DPAD ± mean RMSE",
            "sp_dpad",
            False,
        )
        _band_row(
            zv,
            panel.band_rmse_varma,
            COLOR_VARMA_BAND_FILL,
            COLOR_VARMA_BAND_LINE,
            "VARMA ± mean RMSE",
            "sp_varma",
            False,
        )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=zt,
            mode="lines",
            name=true_series_name,
            legendgroup="sp_true",
            line=dict(color=c_true, width=WIDTH_TRUE),
            showlegend=show_legend,
            connectgaps=False,
        ),
        row=row,
        col=1,
    )
    if not _all_nan_series(zp):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=zp,
                mode="lines",
                name="PSID Ŷ" if _pd else "PSID",
                legendgroup="sp_psid_l",
                line=dict(color=COLOR_PSID, width=WIDTH_PSID, dash=_pd if _pd else None),
                showlegend=show_legend,
                connectgaps=False,
            ),
            row=row,
            col=1,
        )
    if not _all_nan_series(zd):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=zd,
                mode="lines",
                name="DPAD Ŷ" if _pd else "DPAD",
                legendgroup="sp_dpad_l",
                line=dict(color=COLOR_DPAD, width=WIDTH_DPAD, dash=_pd if _pd else None),
                showlegend=show_legend,
                connectgaps=False,
            ),
            row=row,
            col=1,
        )
    if not _all_nan_series(zv):
        fig.add_trace(
            go.Scatter(
                x=t,
                y=zv,
                mode="lines",
                name="VARMA Ŷ" if _pd else "VARMA",
                legendgroup="sp_varma_l",
                line=dict(color=COLOR_VARMA, width=WIDTH_VARMA, dash="8 2" if _pd else "dash"),
                showlegend=show_legend,
                connectgaps=False,
            ),
            row=row,
            col=1,
        )

    xd = "x domain" if row == 1 else f"x{row} domain"
    yd = "y domain" if row == 1 else f"y{row} domain"
    badge_fg, badge_bg = dbs_badge_style(panel.dbs_label)
    fig.add_annotation(
        x=0.01,
        y=0.97,
        xref=xd,
        yref=yd,
        text=f"<b>{badge_text}</b>",
        showarrow=False,
        font=dict(size=FONT_SIZE_BASE, color=badge_fg, family=FONT_FAMILY),
        bgcolor=badge_bg,
        bordercolor=badge_fg,
        borderwidth=1,
        borderpad=4,
        xanchor="left",
        yanchor="top",
    )


def _abs_ticks_only(
    t_abs: np.ndarray, n_tick: int = 15
) -> Tuple[np.ndarray, list[str], Optional[list[float]]]:
    """Bottom-axis ticks: absolute time only (no relative overlay)."""
    t_abs = np.asarray(t_abs, dtype=float).ravel()
    if t_abs.size == 0:
        return np.array([]), [], None
    lo, hi = float(np.nanmin(t_abs)), float(np.nanmax(t_abs))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    tickvals = np.linspace(lo, hi, max(2, int(n_tick)))
    abs_txt = [f"{v:.1f}" for v in tickvals]
    rng = [float(tickvals[0]), float(tickvals[-1])]
    return tickvals, abs_txt, rng


def _abs_and_rel_ticks(
    t_abs: np.ndarray, n_hist: int, n_tick: int = 15
) -> Tuple[np.ndarray, list[str], list[str], float]:
    """Uniform tick positions: bottom labels absolute (s), top labels relative to onset (s), dashboard-style."""
    t_abs = np.asarray(t_abs, dtype=float).ravel()
    if t_abs.size == 0:
        return np.array([]), [], [], 0.0
    lo, hi = float(np.nanmin(t_abs)), float(np.nanmax(t_abs))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    t0 = float(t_abs[n_hist]) if 0 <= n_hist < len(t_abs) else float(t_abs[-1])
    tickvals = np.linspace(lo, hi, max(2, int(n_tick)))
    abs_txt = [f"{v:.1f}" for v in tickvals]
    # History uses absolute time < t0 ⇒ negative relative seconds at forecast onset.
    rel_txt = [f"{(v - t0):.1f}" for v in tickvals]
    return tickvals, abs_txt, rel_txt, t0


def _insert_hist_forecast_gap(a: np.ndarray, n_hist: int) -> np.ndarray:
    """Insert NaN between last history and first forecast sample (no line across the boundary)."""
    a = np.asarray(a, dtype=float).ravel()
    if n_hist <= 0 or n_hist >= len(a):
        return a
    return np.concatenate([a[:n_hist], [np.nan], a[n_hist:]])


def _neural_channel_label(res: Dict[str, Any], channel_idx: int) -> str:
    """Neural feature name from ``input_channels`` (saved YAML order) when available; raw string."""
    names = channels_as_str_list(res.get("input_channels"))
    if channel_idx < 0:
        return f"neural column {channel_idx}"
    if channel_idx < len(names):
        return str(names[channel_idx])
    return f"neural Y column {channel_idx}"


def _resolve_c2_neural_channel_idx(res: Dict[str, Any], spec: ThesisC2ForecastSpec) -> int:
    if spec.forecast_target != "Y":
        return spec.channel_idx
    return resolve_neural_y_channel_idx(res, spec.neural_y_feature_name, spec.channel_idx)


def _fc_channel(z_true: Any, z_pred: Any, channel_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    if z_true is None or z_pred is None:
        raise ValueError("Missing future_true or future_pred slice")
    T = reshape_future_z_time_first(np.asarray(z_true, dtype=float))
    P = reshape_future_z_time_first(np.asarray(z_pred, dtype=float))
    if T.shape != P.shape:
        raise ValueError("future_true / future_pred shape mismatch")
    m, nc = T.shape[0], T.shape[1]
    if nc == 1:
        return T[:, 0].copy(), P[:, 0].copy()
    if channel_idx >= nc:
        raise ValueError("channel_idx out of range for future horizon")
    return T[:, channel_idx].copy(), P[:, channel_idx].copy()


def _absolute_time_seconds_for_forecast_panel(
    res: Dict[str, Any],
    trial_idx: int,
    concat_key: str,
    hist_key: str,
    n_hist: int,
    m_fc: int,
    sampling_hz: float,
    history_ms: float,
    forecast_ms: float,
) -> np.ndarray:
    """
    Absolute trial time (s) for each sample in [history || forecast], aligned with the
    predictions/forecasting dashboard (``time_margined``, else tail of ``get_trial_time_axis``,
    else offset + uniform grid like ``render_forecasting_tab``).
    """
    n_tot = int(n_hist + m_fc)
    if n_tot <= 0:
        return np.array([], dtype=float)

    tm_list = res.get("time_margined")
    if tm_list and trial_idx < len(tm_list) and tm_list[trial_idx] is not None:
        tm = np.asarray(tm_list[trial_idx], dtype=float).ravel()
        if tm.size >= n_tot:
            return tm[:n_tot].astype(float)
        if tm.size >= 2 and tm.size < n_tot:
            new_x = np.linspace(0.0, 1.0, n_tot)
            old_x = np.linspace(0.0, 1.0, tm.size)
            return np.interp(new_x, old_x, tm).astype(float)

    z_list = res.get(hist_key)
    if z_list and trial_idx < len(z_list) and z_list[trial_idx] is not None:
        _, t_ax = _prepare_z_array(z_list[trial_idx], res, trial_idx)
        t_ax = np.asarray(t_ax, dtype=float).ravel()
        if t_ax.size >= n_tot:
            return t_ax[-n_tot:].copy()

    offsets = res.get("offset", [])
    t_off = 0.0
    if offsets and trial_idx < len(offsets) and offsets[trial_idx] is not None:
        o = np.asarray(offsets[trial_idx]).ravel()
        if o.size:
            t_off = float(o[0])
    hist_sec = n_hist / max(sampling_hz, 1e-6)
    fut_sec = min(m_fc / max(sampling_hz, 1e-6), forecast_ms / 1000.0)
    total_sec = hist_sec + fut_sec
    return (t_off + np.linspace(0.0, total_sec, n_tot)).astype(float)


def _infer_history_len(
    res: Dict[str, Any], trial_idx: int, m_future_full: int, concat_key: str
) -> Optional[int]:
    """
    History sample count H in ``Z_concat_for_plot`` / ``Y_concat_for_plot`` such that
    concat length == H + M_full (M_full = full saved future length, **before** thesis display clip).
    """
    zc = res.get(concat_key)
    if zc is None or trial_idx >= len(zc) or zc[trial_idx] is None:
        return None
    a = np.asarray(zc[trial_idx])
    if a.ndim != 2:
        return None
    total = a.shape[0]
    if total > m_future_full:
        return int(total - m_future_full)
    return None


def _history_trace(
    res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    n_hist: int,
    hist_key: str = "Z",
) -> np.ndarray:
    zt = res[hist_key][trial_idx]
    z_arr, t_abs = _prepare_z_array(zt, res, trial_idx)
    true_c = get_channel(z_arr, channel_idx, t_abs)
    arr = np.asarray(true_c, dtype=float).ravel()
    n = min(len(arr), n_hist)
    return arr[:n].copy()


def _history_trace_before_forecast(
    res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    n_hist: int,
    history_end: int,
    hist_key: str = "Z",
) -> np.ndarray:
    """Last ``n_hist`` samples of the history segment ending at ``history_end`` (forecast starts there)."""
    zt = res[hist_key][trial_idx]
    z_arr, t_abs = _prepare_z_array(zt, res, trial_idx)
    true_c = get_channel(z_arr, channel_idx, t_abs)
    arr = np.asarray(true_c, dtype=float).ravel()
    he = min(max(int(history_end), 0), len(arr))
    nh = min(int(n_hist), he)
    return arr[he - nh : he].copy()


def _absolute_time_hist_then_forecast(
    res: Dict[str, Any],
    trial_idx: int,
    hist_key: str,
    history_end: int,
    n_hist: int,
    m_fc: int,
    sampling_hz: float,
) -> np.ndarray:
    """
    Absolute trial clock (s): ``n_hist`` history samples immediately before forecast onset, then ``m_fc``
    forecast samples. Enforces ~1/sampling_hz step spacing in each segment so history/forecast durations
    match sample counts (50/50 when counts cap equally).
    """
    dt = 1.0 / max(sampling_hz, 1e-6)
    zt = res[hist_key][trial_idx]
    _, t_ax = _prepare_z_array(zt, res, trial_idx)
    t_full = np.asarray(t_ax, dtype=float).ravel()
    he = min(max(int(history_end), 0), len(t_full))
    i0 = max(0, he - int(n_hist))
    if he + int(m_fc) <= len(t_full) and i0 < he:
        t_hist = t_full[i0:he].astype(float)
        t_fc = t_full[he : he + int(m_fc)].astype(float)
        if t_hist.size == n_hist and t_fc.size == m_fc:
            return np.concatenate([t_hist, t_fc])
    last_h = (
        float(t_full[he - 1])
        if he > 0 and t_full.size
        else (float(t_full[0]) if t_full.size else 0.0)
    )
    t_hist = last_h + dt * (np.arange(int(n_hist), dtype=float) - (int(n_hist) - 1))
    t_fc = last_h + dt * (1.0 + np.arange(int(m_fc), dtype=float))
    return np.concatenate([t_hist, t_fc]).astype(float)


def _build_one_panel(
    res_p: Dict[str, Any],
    res_d: Dict[str, Any],
    res_v: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    spec: ThesisC2ForecastSpec,
    sigma_z: Optional[float],
    missing_fc_logged: Optional[set[str]] = None,
    *,
    varma_trial_idx: Optional[int] = None,
) -> Optional[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
        int,
    ]
]:
    """
    Returns:
      t_abs_s, z_true_plot,
      z_psid_plot, z_dpad_plot, z_varma_plot (nan in history),
      upper, lower (PSID band, nan outside forecast),
      rmse_psid, rmse_dpad, rmse_varma,
      n_hist (split index: forecast starts at t_abs_s[n_hist])
    """
    y_mode = spec.forecast_target == "Y"
    k_true = "Y_future_true" if y_mode else "Z_future_true"
    k_pred = "Y_future_pred" if y_mode else "Z_future_pred"
    concat_key = "Y_concat_for_plot" if y_mode else "Z_concat_for_plot"
    hist_key = "Y" if y_mode else "Z"
    vi = int(varma_trial_idx) if varma_trial_idx is not None else int(trial_idx)

    def _has_fc(res: Dict[str, Any], tidx: int) -> bool:
        zt = res.get(k_true)
        zp = res.get(k_pred)
        if zt is None or zp is None:
            return False
        if tidx >= len(zt) or tidx >= len(zp):
            return False
        return zt[tidx] is not None and zp[tidx] is not None

    if not _has_fc(res_p, trial_idx):
        logger.warning(
            "PSID missing %s / %s for trial %s — cannot build C2 panel",
            k_true,
            k_pred,
            trial_idx,
        )
        return None

    zt_fc, zp_fc = _fc_channel(
        res_p[k_true][trial_idx],
        res_p[k_pred][trial_idx],
        channel_idx,
    )

    m_psid_fc = min(zt_fc.size, zp_fc.size)
    if m_psid_fc == 0:
        return None

    def _try_fc(
        label: str, res: Dict[str, Any], tidx: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not _has_fc(res, tidx):
            if missing_fc_logged is not None and label not in missing_fc_logged:
                missing_fc_logged.add(label)
                logger.warning(
                    "C2 %s: forecast columns missing or trial index unavailable — trace set to NaN "
                    "(align test parquets with PSID).",
                    label,
                )
            return None, None
        try:
            a, b = _fc_channel(
                res[k_true][tidx],
                res[k_pred][tidx],
                channel_idx,
            )
            return a, b
        except Exception as e:
            logger.warning("C2 %s: _fc_channel failed (%s) — trace set to NaN", label, e)
            return None, None

    zd_fc_raw, zdp_fc_raw = _try_fc("DPAD", res_d, trial_idx)
    zv_fc_raw, zvp_fc_raw = _try_fc("VARMA", res_v, vi)

    sizes = [zt_fc.size, zp_fc.size]
    for a in (zd_fc_raw, zdp_fc_raw, zv_fc_raw, zvp_fc_raw):
        if a is not None:
            sizes.append(a.size)
    m_full = min(sizes)
    if m_full == 0:
        return None

    n_fc_cap = int(round(spec.forecast_ms / 1000.0 * spec.sampling_hz))
    m_fc = min(m_full, n_fc_cap)
    zt_fc, zp_fc = zt_fc[:m_fc], zp_fc[:m_fc]

    def _clip_or_nan(arr: Optional[np.ndarray]) -> np.ndarray:
        if arr is None:
            return np.full(m_fc, np.nan)
        return arr[:m_fc]

    zd_fc = _clip_or_nan(zd_fc_raw)
    zdp_fc = _clip_or_nan(zdp_fc_raw)
    zv_fc = _clip_or_nan(zv_fc_raw)
    zvp_fc = _clip_or_nan(zvp_fc_raw)

    h_infer = _infer_history_len(res_p, trial_idx, m_full, concat_key)
    n_hist_cap = int(round(spec.history_ms / 1000.0 * spec.sampling_hz))
    if h_infer is not None:
        n_hist_want = min(h_infer, n_hist_cap)
        th = _history_trace_before_forecast(
            res_p, trial_idx, channel_idx, n_hist_want, h_infer, hist_key=hist_key
        )
        hist_end = int(h_infer)
    else:
        n_hist_want = n_hist_cap
        th = _history_trace(res_p, trial_idx, channel_idx, n_hist_want, hist_key=hist_key)
        hist_end = len(th)
    n_hist = len(th)

    t_abs = _absolute_time_hist_then_forecast(
        res_p,
        trial_idx,
        hist_key,
        hist_end,
        n_hist,
        m_fc,
        spec.sampling_hz,
    )
    if t_abs.size != n_hist + m_fc:
        n_tot = n_hist + m_fc
        dt = 1.0 / max(spec.sampling_hz, 1e-6)
        t_abs = np.arange(n_tot, dtype=float) * dt

    true_concat = np.concatenate([th, zt_fc])
    z_true_plot = zscore_using_true_stats(true_concat, true_concat)

    def _pred_line(pred_fc: np.ndarray) -> np.ndarray:
        if np.all(np.isnan(pred_fc)):
            out = np.full(n_hist + m_fc, np.nan)
            return out
        pred_full = np.concatenate([th, pred_fc])
        z = zscore_using_true_stats(true_concat, pred_full)
        out = z.copy()
        out[:n_hist] = np.nan
        return out

    z_psid = _pred_line(zp_fc)
    z_dpad = _pred_line(zdp_fc)
    z_varma = _pred_line(zvp_fc)

    zt_z = zscore_using_true_stats(zt_fc, zt_fc)

    def _safe_rmse(pred_fc: np.ndarray) -> float:
        if np.all(np.isnan(pred_fc)):
            return float("nan")
        return rmse_z(zt_z, zscore_using_true_stats(zt_fc, pred_fc))

    rmse_psid = _safe_rmse(zp_fc)
    rmse_dpad = _safe_rmse(zdp_fc)
    rmse_varma = _safe_rmse(zvp_fc)

    upper = np.full_like(z_true_plot, np.nan)
    lower = np.full_like(z_true_plot, np.nan)
    if sigma_z is not None and sigma_z > 0.0 and not np.all(np.isnan(zp_fc)):
        z_psid_fc_z = zscore_using_true_stats(true_concat, np.concatenate([th, zp_fc]))[n_hist:]
        m_idx = np.arange(1, m_fc + 1, dtype=float)
        w = sigma_z * np.sqrt(m_idx)
        upper[n_hist:] = z_psid_fc_z + w
        lower[n_hist:] = z_psid_fc_z - w

    return (
        t_abs,
        z_true_plot,
        z_psid,
        z_dpad,
        z_varma,
        upper,
        lower,
        rmse_psid,
        rmse_dpad,
        rmse_varma,
        n_hist,
    )


def build_c2_forecast_figure(
    spec: ThesisC2ForecastSpec,
    results_root: Path,
) -> Tuple[go.Figure, str]:
    sigma: Optional[float] = None
    if spec.forecast_target == "Z" and spec.show_psid_sigma_band:
        val_psid = load_split_results_required(
            results_root, spec.psid_variant, spec.psid_run_ts, "val"
        )
        sigma = sigma_z_from_psid_val_residuals(val_psid, spec.channel_idx)

    res_p = load_split_results_required(
        results_root, spec.psid_variant, spec.psid_run_ts, spec.split
    )
    i_off, i_on = resolve_off_on_indices_from_spec(
        trial_idx_off=spec.trial_idx_off,
        trial_idx_on=spec.trial_idx_on,
        use_adjacent_off_on_trials=spec.use_adjacent_off_on_trials,
        split_res=res_p,
    )
    res_d = load_split_results_required(
        results_root, spec.dpad_variant, spec.dpad_run_ts, spec.split
    )
    res_v = load_split_results_required(
        results_root, spec.varma_variant, spec.varma_run_ts, spec.split
    )
    res_v_off: Optional[Dict[str, Any]] = None
    res_v_on: Optional[Dict[str, Any]] = None
    if "dbs_both" in spec.varma_variant:
        v_off = spec.varma_variant.replace("dbs_both", "dbs_off")
        v_on = spec.varma_variant.replace("dbs_both", "dbs_on")
        ts_off = spec.varma_run_ts_off
        ts_on = spec.varma_run_ts_on
        if ts_off is None or ts_on is None:
            inf = infer_varma_off_on_run_ts(spec.varma_variant, spec.varma_run_ts)
            if inf is not None:
                ts_off, ts_on = inf
        if ts_off is None or ts_on is None:
            raise ValueError(
                "C2 dbs_both VARMA: set varma_run_ts_off/on on the spec or use a triplet listed in "
                "specs._ALL_TRIPLETS."
            )
        res_v_off = load_split_results_required(results_root, v_off, ts_off, spec.split)
        res_v_on = load_split_results_required(results_root, v_on, ts_on, spec.split)

    def _varma_res_idx(panel: str, psid_trial_idx: int) -> Tuple[Dict[str, Any], int]:
        if panel == "off" and res_v_off is not None:
            mp = _key_index_map(res_v_off)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_off, mp[k]
        if panel == "on" and res_v_on is not None:
            mp = _key_index_map(res_v_on)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_on, mp[k]
        return res_v, psid_trial_idx

    rv_off, jv_off = _varma_res_idx("off", i_off)
    rv_on, jv_on = _varma_res_idx("on", i_on)
    map_v_off = _key_index_map(rv_off)
    map_v_on = _key_index_map(rv_on)

    ch_idx = (
        _resolve_c2_neural_channel_idx(res_p, spec)
        if spec.forecast_target == "Y"
        else spec.channel_idx
    )

    if spec.forecast_target == "Z":
        och = channels_as_str_list(res_p.get("output_channels"))
        if ch_idx < len(och):
            _ch_c2 = och[ch_idx]
        else:
            if 0 <= ch_idx < len(THESIS_DECLARED_BEHAVIORAL_OUTPUTS):
                _ch_c2 = THESIS_DECLARED_BEHAVIORAL_OUTPUTS[ch_idx]
            else:
                _ch_c2, _ = resolve_output_channel_display(
                    res_p, ch_idx, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS
                )
        y_axis_label = (
            spec.y_axis_label
            if spec.y_axis_label != "z-scored tracing speed"
            else f"z-scored {_ch_c2}"
        )
    else:
        _ch_c2 = neural_y_feature_label(
            res_p,
            ch_idx,
            neural_y_feature_name=spec.neural_y_feature_name,
        )
        raw_yl = spec.y_axis_label.strip()
        if raw_yl and raw_yl != "z-scored tracing speed":
            y_axis_label = raw_yl
        else:
            y_axis_label = f"z-scored {_ch_c2}"

    if spec.show_session_rmse_bands:
        if spec.forecast_target == "Z":
            bands_off = _session_mean_rmse_z_triplet(
                res_p, res_d, rv_off, map_v_off, i_off, ch_idx, "off"
            )
            bands_on = _session_mean_rmse_z_triplet(
                res_p, res_d, rv_on, map_v_on, i_on, ch_idx, "on"
            )
        else:
            bands_off = _session_mean_rmse_y_triplet(
                res_p, res_d, rv_off, map_v_off, i_off, ch_idx, "off"
            )
            bands_on = _session_mean_rmse_y_triplet(
                res_p, res_d, rv_on, map_v_on, i_on, ch_idx, "on"
            )
    else:
        bands_off = (None, None, None)
        bands_on = (None, None, None)

    pred_rows_active = False
    j_pred_off = -1
    j_pred_on = -1
    pred_panel_off: Optional[ThesisPanelData] = None
    pred_panel_on: Optional[ThesisPanelData] = None
    if spec.include_next_trial_prediction_rows:
        j_po, j_pn = i_off + 1, i_on + 1
        stim_l = res_p.get("stim")
        if stim_l is None:
            stim_seq: list[Any] = []
        elif isinstance(stim_l, np.ndarray):
            stim_seq = stim_l.tolist()
        else:
            stim_seq = list(stim_l)
        key_n = "Z" if spec.forecast_target == "Z" else "Y"
        n_trials = len(res_p.get(key_n) or [])
        stim_ok = j_po < len(stim_seq) and j_pn < len(stim_seq)
        if j_po < n_trials and j_pn < n_trials and stim_ok:
            try:
                stim_next_off = normalize_stim(stim_seq[j_po])
                stim_next_on = normalize_stim(stim_seq[j_pn])
                dbs_lbl_o = "DBS-OFF" if stim_next_off == "off" else "DBS-ON"
                dbs_lbl_n = "DBS-OFF" if stim_next_on == "off" else "DBS-ON"
                rv_po, jv_po = _varma_res_idx(
                    "off" if stim_next_off == "off" else "on", j_po
                )
                rv_pn, jv_pn = _varma_res_idx(
                    "off" if stim_next_on == "off" else "on", j_pn
                )
                mp_po = _key_index_map(rv_po)
                mp_pn = _key_index_map(rv_pn)
                if spec.forecast_target == "Z":
                    bpr_o = _session_mean_rmse_z_triplet(
                        res_p, res_d, rv_po, mp_po, j_po, ch_idx, stim_next_off
                    )
                    bpr_n = _session_mean_rmse_z_triplet(
                        res_p, res_d, rv_pn, mp_pn, j_pn, ch_idx, stim_next_on
                    )
                    exo = extract_trial_z_series(
                        res_p, res_d, rv_po, j_po, ch_idx, varma_trial_idx=jv_po
                    )
                    exn = extract_trial_z_series(
                        res_p, res_d, rv_pn, j_pn, ch_idx, varma_trial_idx=jv_pn
                    )
                else:
                    bpr_o = _session_mean_rmse_y_triplet(
                        res_p, res_d, rv_po, mp_po, j_po, ch_idx, stim_next_off
                    )
                    bpr_n = _session_mean_rmse_y_triplet(
                        res_p, res_d, rv_pn, mp_pn, j_pn, ch_idx, stim_next_on
                    )
                    exo = extract_trial_y_series(
                        res_p, res_d, rv_po, j_po, ch_idx, varma_trial_idx=jv_po
                    )
                    exn = extract_trial_y_series(
                        res_p, res_d, rv_pn, j_pn, ch_idx, varma_trial_idx=jv_pn
                    )
                zt_a, zp_a, zd_a, zv_a = z_true_and_preds(
                    exo.z_true_raw, exo.z_psid, exo.z_dpad, exo.z_varma
                )
                zt_b, zp_b, zd_b, zv_b = z_true_and_preds(
                    exn.z_true_raw, exn.z_psid, exn.z_dpad, exn.z_varma
                )
                pred_panel_off = ThesisPanelData(
                    t_abs=exo.t_abs,
                    z_true=zt_a,
                    z_psid=zp_a,
                    z_dpad=zd_a,
                    z_varma=zv_a,
                    psid_sigma=None,
                    dbs_label=dbs_lbl_o,
                    band_rmse_psid=bpr_o[0],
                    band_rmse_dpad=bpr_o[1],
                    band_rmse_varma=bpr_o[2],
                )
                pred_panel_on = ThesisPanelData(
                    t_abs=exn.t_abs,
                    z_true=zt_b,
                    z_psid=zp_b,
                    z_dpad=zd_b,
                    z_varma=zv_b,
                    psid_sigma=None,
                    dbs_label=dbs_lbl_n,
                    band_rmse_psid=bpr_n[0],
                    band_rmse_dpad=bpr_n[1],
                    band_rmse_varma=bpr_n[2],
                )
                pred_panel_off = _slice_c2_prediction_panel(
                    pred_panel_off,
                    segment_s=spec.prediction_row_center_segment_s,
                    side_extend_s=spec.prediction_row_side_extend_s,
                )
                pred_panel_on = _slice_c2_prediction_panel(
                    pred_panel_on,
                    segment_s=spec.prediction_row_center_segment_s,
                    side_extend_s=spec.prediction_row_side_extend_s,
                )
                j_pred_off, j_pred_on = j_po, j_pn
                pred_rows_active = True
            except Exception as e:
                logger.warning("C2: next-trial prediction rows skipped (%s)", e)
        else:
            logger.warning(
                "C2: next-trial prediction rows skipped (need trials after exemplars: rows %s/%s).",
                j_po,
                j_pn,
            )

    _miss_fc: set[str] = set()
    off = _build_one_panel(
        res_p,
        res_d,
        rv_off,
        i_off,
        ch_idx,
        spec,
        sigma,
        missing_fc_logged=_miss_fc,
        varma_trial_idx=jv_off,
    )
    on = _build_one_panel(
        res_p,
        res_d,
        rv_on,
        i_on,
        ch_idx,
        spec,
        sigma,
        missing_fc_logged=_miss_fc,
        varma_trial_idx=jv_on,
    )

    if off is None or on is None:
        fk = (
            "Y_future_true/Y_future_pred"
            if spec.forecast_target == "Y"
            else "Z_future_true/Z_future_pred"
        )
        raise ValueError(
            f"C2: missing {fk} for one or both trials. "
            "Run forecast validation and save test parquet with forecast columns."
        )

    paper_bg, plot_bg = paper_colors(spec.theme)
    grid = grid_color(spec.theme)
    fg = true_line_color(spec.theme)
    c_true = true_line_color(spec.theme)

    n_subplot_rows = 4 if pred_rows_active else 2
    vspace = 0.09 if pred_rows_active else 0.16
    fig = make_subplots(
        rows=n_subplot_rows,
        cols=1,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=vspace,
    )

    def _y_from_forecast_rows(*rows: Tuple) -> Tuple[float, float]:
        ys: list[np.ndarray] = []
        for rowdata in rows:
            ys.append(rowdata[1][np.isfinite(rowdata[1])])
            for j in (2, 3, 4):
                a = rowdata[j]
                ys.append(a[np.isfinite(a)])
            if spec.show_psid_sigma_band:
                u, lo = rowdata[5], rowdata[6]
                ys.append(u[np.isfinite(u)])
                ys.append(lo[np.isfinite(lo)])
        y_flat = np.concatenate(ys) if ys else np.array([])
        y_flat = y_flat[np.isfinite(y_flat)]
        if y_flat.size == 0:
            return -1.0, 1.0
        ymax = float(np.nanmax(y_flat))
        ymin = float(np.nanmin(y_flat))
        pad = 0.05 * (ymax - ymin + 1e-6)
        return ymin - pad, ymax + pad

    y0_fc, y1_fc = _y_from_forecast_rows(off, on)

    y0_pr, y1_pr = y0_fc, y1_fc
    if pred_rows_active and pred_panel_off is not None and pred_panel_on is not None:
        ys_p: list[np.ndarray] = []
        for pan in (pred_panel_off, pred_panel_on):
            for arr in (pan.z_true, pan.z_psid, pan.z_dpad, pan.z_varma):
                a = np.asarray(arr, dtype=float).ravel()
                ys_p.append(a[np.isfinite(a)])
        yp = np.concatenate(ys_p) if ys_p else np.array([])
        yp = yp[np.isfinite(yp)]
        if yp.size:
            ymax = float(np.nanmax(yp))
            ymin = float(np.nanmin(yp))
            pad = 0.05 * (ymax - ymin + 1e-6)
            y0_pr, y1_pr = ymin - pad, ymax + pad

    def _add_panel(
        row: int,
        rowdata: Tuple,
        badge: str,
        row_bands: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None,
    ) -> None:
        _ysuf = "" if row == 1 else str(row)
        yref_d = f"y{_ysuf} domain"
        xref_d = f"x{_ysuf} domain"
        (
            t_full,
            z_true,
            z_psid,
            z_dpad,
            z_varma,
            upper,
            lower,
            _r_p,
            _r_d,
            _r_v,
            n_hist,
        ) = rowdata

        t_full = np.asarray(t_full, dtype=float).ravel()
        n_hist = int(n_hist)
        # Break line traces at history|forecast boundary (true was continuous; preds already NaN in history).
        t_plot = _insert_hist_forecast_gap(t_full, n_hist)
        z_true_p = _insert_hist_forecast_gap(np.asarray(z_true, dtype=float).ravel(), n_hist)
        z_psid_p = _insert_hist_forecast_gap(np.asarray(z_psid, dtype=float).ravel(), n_hist)
        z_dpad_p = _insert_hist_forecast_gap(np.asarray(z_dpad, dtype=float).ravel(), n_hist)
        z_varma_p = _insert_hist_forecast_gap(np.asarray(z_varma, dtype=float).ravel(), n_hist)
        upper_p = np.asarray(upper, dtype=float).ravel()
        lower_p = np.asarray(lower, dtype=float).ravel()

        if t_full.size and n_hist > 0:
            fig.add_vrect(
                x0=float(t_full[0]),
                x1=float(t_full[n_hist - 1]),
                fillcolor=_CONTEXT,
                layer="below",
                line_width=0,
                row=row,
                col=1,
            )
        if t_full.size and n_hist < len(t_full):
            fig.add_vrect(
                x0=float(t_full[n_hist]),
                x1=float(t_full[-1]),
                fillcolor=_FORECAST,
                layer="below",
                line_width=0,
                row=row,
                col=1,
            )
            fig.add_vline(
                x=float(t_full[n_hist]),
                line=dict(color=_RULE, width=1, dash="dash"),
                row=row,
                col=1,
            )

        if spec.show_session_rmse_bands and row_bands is not None:
            bp, bd, bv = row_bands
            t_f = t_full[n_hist:]
            show_leg = row == 1

            def _sess_band(
                y_hat: np.ndarray,
                half: Optional[float],
                fill_c: str,
                line_c: str,
                name: str,
                leg: str,
                show: bool,
            ) -> None:
                if half is None or not np.isfinite(half) or half <= 0:
                    return
                yh = np.asarray(y_hat[n_hist:], dtype=float).ravel()
                if t_f.size != yh.size or np.all(np.isnan(yh)):
                    return
                u = yh + half
                lo_b = yh - half
                x_band = np.concatenate([t_f, t_f[::-1]])
                y_band = np.concatenate([u, lo_b[::-1]])
                fig.add_trace(
                    go.Scatter(
                        x=x_band,
                        y=y_band,
                        fill="toself",
                        fillcolor=fill_c,
                        line=dict(color=line_c, width=0.5),
                        name=name,
                        legendgroup=leg,
                        showlegend=show,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=1,
                )

            _sess_band(
                z_psid,
                bp,
                COLOR_PSID_BAND_FILL,
                COLOR_PSID_BAND_LINE,
                "PSID ± mean RMSE",
                "s_rmse_p",
                False,
            )
            _sess_band(
                z_dpad,
                bd,
                COLOR_DPAD_BAND_FILL,
                COLOR_DPAD_BAND_LINE,
                "DPAD ± mean RMSE",
                "s_rmse_d",
                False,
            )
            _sess_band(
                z_varma,
                bv,
                COLOR_VARMA_BAND_FILL,
                COLOR_VARMA_BAND_LINE,
                "VARMA ± mean RMSE",
                "s_rmse_v",
                False,
            )

        if spec.show_psid_sigma_band:
            xb = np.concatenate([t_full, t_full[::-1]])
            yb = np.concatenate([upper_p, lower_p[::-1]])
            yb = np.where(np.isfinite(yb), yb, np.nan)
            if np.any(np.isfinite(yb)):
                fig.add_trace(
                    go.Scatter(
                        x=xb,
                        y=yb,
                        fill="toself",
                        fillcolor=COLOR_PSID_BAND_FILL,
                        mode="lines",
                        line=dict(color=COLOR_PSID_BAND_LINE, width=0),
                        name="PSID ±1σ (∝√m)",
                        legendgroup="band",
                        showlegend=False,
                        hoverinfo="skip",
                        connectgaps=False,
                    ),
                    row=row,
                    col=1,
                )

        fig.add_trace(
            go.Scatter(
                x=t_plot,
                y=z_true_p,
                mode="lines",
                name="y_true",
                legendgroup="true",
                line=dict(color=c_true, width=WIDTH_TRUE),
                showlegend=(row == 1),
                connectgaps=False,
            ),
            row=row,
            col=1,
        )

        _show = row == 1

        if not np.all(np.isnan(z_psid)):
            fig.add_trace(
                go.Scatter(
                    x=t_plot,
                    y=z_psid_p,
                    mode="lines",
                    name="y_hat_PSID",
                    legendgroup="psid",
                    line=dict(color=COLOR_PSID, width=WIDTH_PSID),
                    showlegend=_show,
                    connectgaps=False,
                ),
                row=row,
                col=1,
            )
        if not np.all(np.isnan(z_dpad)):
            fig.add_trace(
                go.Scatter(
                    x=t_plot,
                    y=z_dpad_p,
                    mode="lines",
                    name="y_hat_DPAD",
                    legendgroup="dpad",
                    line=dict(color=COLOR_DPAD, width=WIDTH_DPAD),
                    showlegend=_show,
                    connectgaps=False,
                ),
                row=row,
                col=1,
            )
        if not np.all(np.isnan(z_varma)):
            fig.add_trace(
                go.Scatter(
                    x=t_plot,
                    y=z_varma_p,
                    mode="lines",
                    name="y_hat_VARMA",
                    legendgroup="varma",
                    line=dict(color=COLOR_VARMA, width=WIDTH_VARMA, dash="8 2"),
                    showlegend=_show,
                    connectgaps=False,
                ),
                row=row,
                col=1,
            )

        badge_fg, badge_bg = dbs_badge_style(badge)
        fig.add_annotation(
            x=0.01,
            y=0.97,
            xref=xref_d,
            yref=yref_d,
            text=f"<b>{badge}</b>",
            showarrow=False,
            font=dict(size=FONT_SIZE_BASE, color=badge_fg, family=FONT_FAMILY),
            bgcolor=badge_bg,
            bordercolor=badge_fg,
            borderwidth=1,
            borderpad=4,
            xanchor="left",
            yanchor="top",
        )
    _add_panel(1, off, "DBS-OFF", bands_off)
    _add_panel(2, on, "DBS-ON", bands_on)

    if pred_rows_active and pred_panel_off is not None and pred_panel_on is not None:
        _add_self_prediction_stack_row(
            fig,
            3,
            pred_panel_off,
            spec.theme,
            show_legend=False,
            badge_text=f"One-step · trial {j_pred_off}",
            prediction_line_dash=None,
            true_series_name="y_true",
        )
        _add_self_prediction_stack_row(
            fig,
            4,
            pred_panel_on,
            spec.theme,
            show_legend=False,
            badge_text=f"One-step · trial {j_pred_on}",
            prediction_line_dash=None,
            true_series_name="y_true",
        )

    _x_title_kw = thesis_bottom_time_axis_title(theme=spec.theme)

    def _xkw_base() -> dict:
        return dict(
            showgrid=True,
            gridcolor=grid,
            showline=True,
            linewidth=1,
            linecolor=fg,
            mirror=False,
            tickfont=dict(size=FONT_SIZE_TICK, color=fg),
        )


    tv_off, abs_off, rel_off, _ = _abs_and_rel_ticks(off[0], off[10])
    tv_on, abs_on, rel_on, _ = _abs_and_rel_ticks(on[0], on[10])
    rng_off = [float(tv_off[0]), float(tv_off[-1])] if tv_off.size else None
    rng_on = [float(tv_on[0]), float(tv_on[-1])] if tv_on.size else None

    fig.update_xaxes(
        tickmode="array",
        tickvals=tv_off,
        ticktext=abs_off,
        range=rng_off,
        title=_x_title_kw,
        showticklabels=True,
        **_xkw_base(),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tv_on,
        ticktext=abs_on,
        range=rng_on,
        title=_x_title_kw,
        showticklabels=True,
        **_xkw_base(),
        row=2,
        col=1,
    )

    if pred_rows_active and pred_panel_off is not None and pred_panel_on is not None:
        tpo, apo, rgo = _abs_ticks_only(pred_panel_off.t_abs)
        tpn, apn, rgn = _abs_ticks_only(pred_panel_on.t_abs)
        fig.update_xaxes(
            tickmode="array",
            tickvals=tpo,
            ticktext=apo,
            range=rgo,
            title=_x_title_kw,
            showticklabels=False,
            **_xkw_base(),
            row=3,
            col=1,
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=tpn,
            ticktext=apn,
            range=rgn,
            title=_x_title_kw,
            showticklabels=True,
            **_xkw_base(),
            row=4,
            col=1,
        )

    _top_title = thesis_relative_forecast_onset_axis_title(theme=spec.theme)

    _ykw_base = dict(
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linewidth=1,
        linecolor=fg,
        mirror=False,
        tickfont=dict(size=FONT_SIZE_TICK, color=fg),
    )
    fig.update_yaxes(title_text="", range=[y0_fc, y1_fc], **_ykw_base, row=1, col=1)
    fig.update_yaxes(title_text="", range=[y0_fc, y1_fc], **_ykw_base, row=2, col=1)
    if pred_rows_active:
        fig.update_yaxes(title_text="", range=[y0_pr, y1_pr], **_ykw_base, row=3, col=1)
        fig.update_yaxes(title_text="", range=[y0_pr, y1_pr], **_ykw_base, row=4, col=1)

    _ox1 = "xaxis5" if pred_rows_active else "xaxis3"
    _ox2 = "xaxis6" if pred_rows_active else "xaxis4"
    _layout_h = int(FIGURE_HEIGHT_C2_FORECAST * (1.88 if pred_rows_active else 1.0))
    _leg_y = -0.14 if pred_rows_active else -0.18
    _margin_b = 128 if pred_rows_active else 118

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=fg),
        height=_layout_h,
        margin=dict(l=88, r=28, t=56, b=_margin_b),
        **{
            _ox1: dict(
                overlaying="x",
                side="top",
                anchor="y",
                tickmode="array",
                tickvals=tv_off,
                ticktext=rel_off,
                range=rng_off,
                title=_top_title,
                showgrid=False,
                tickfont=dict(size=FONT_SIZE_TICK, color=fg),
            ),
            _ox2: dict(
                overlaying="x2",
                side="top",
                anchor="y2",
                tickmode="array",
                tickvals=tv_on,
                ticktext=rel_on,
                range=rng_on,
                title=_top_title,
                showgrid=False,
                tickfont=dict(size=FONT_SIZE_TICK, color=fg),
            ),
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=_leg_y,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZE_TICK),
            bgcolor=legend_bgcolor(),
        ),
        hovermode="x unified",
    )

    feat_label = _ch_c2
    cap = thesis_exemplar_tagline(
        res_p,
        i_off,
        i_on,
        feat_label,
        participant_label=spec.participant_label,
    )
    extra = (spec.caption or "").strip()
    caption = f"{cap} · {extra}" if extra else cap
    return fig, caption
