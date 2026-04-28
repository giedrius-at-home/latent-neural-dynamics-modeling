from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from thesis_lib.plot_scaffolds import THESIS_TIME_AXIS_TITLE
from thesis_lib.constants import (
    COLOR_DPAD,
    COLOR_DPAD_BAND_FILL,
    COLOR_DPAD_BAND_LINE,
    COLOR_PSID,
    COLOR_PSID_BAND_FILL,
    COLOR_PSID_BAND_LINE,
    COLOR_VARMA,
    COLOR_VARMA_BAND_FILL,
    COLOR_VARMA_BAND_LINE,
    FIGURE_HEIGHT_STACKED,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    OPACITY_DPAD,
    OPACITY_PSID,
    OPACITY_VARMA,
    ThesisTheme,
    dbs_badge_style,
    grid_color,
    legend_bgcolor,
    paper_colors,
    true_line_color,
    WIDTH_DPAD,
    WIDTH_PSID,
    WIDTH_TRUE,
    WIDTH_VARMA,
)


@dataclass
class ThesisPanelData:
    """One horizontal strip: time axis and z-scored series."""

    t_abs: np.ndarray
    z_true: np.ndarray
    z_psid: np.ndarray
    z_dpad: np.ndarray
    z_varma: np.ndarray
    psid_sigma: Optional[float]
    dbs_label: str
    # Mean per-trial RMSE (z) in this session × DBS, per model; ribbon = ŷ ± value
    band_rmse_psid: Optional[float] = None
    band_rmse_dpad: Optional[float] = None
    band_rmse_varma: Optional[float] = None


def _finite_ys_for_autoscale(
    zt: np.ndarray,
    zp: np.ndarray,
    zd: np.ndarray,
    zv: np.ndarray,
    band_psid: Optional[float],
    band_dpad: Optional[float],
    band_varma: Optional[float],
) -> list[float]:
    out: list[float] = []
    for arr in (zt, zp, zd, zv):
        a = np.asarray(arr, dtype=float).ravel()
        out.extend(float(x) for x in a if np.isfinite(x))
    for arr, half in ((zp, band_psid), (zd, band_dpad), (zv, band_varma)):
        if half is None or not np.isfinite(half) or half <= 0:
            continue
        a = np.asarray(arr, dtype=float).ravel()
        for x in a:
            if np.isfinite(x):
                out.append(float(x - half))
                out.append(float(x + half))
    return out


def _y_range_from_values(vals: list[float], pad_frac: float = 0.08) -> tuple[float, float]:
    if not vals:
        return -2.5, 2.5
    lo, hi = min(vals), max(vals)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -2.5, 2.5
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    pad = max(span * pad_frac, 0.15)
    return lo - pad, hi + pad


def build_dbs_stacked_figure(
    panel_off: ThesisPanelData,
    panel_on: ThesisPanelData,
    theme: ThesisTheme,
    y_axis_label: str,
    caption: str = "",
    *,
    show_psid_sigma_band: bool = False,
    true_series_name: str = "y_true",
    prediction_line_dash: Optional[str] = None,
    use_absolute_time: bool = True,
) -> go.Figure:
    """
    Two rows sharing x: top = DBS-OFF, bottom = DBS-ON. X-axis labels only on bottom.

    Session-mean RMSE bands: when ``band_rmse_*`` are set on a panel, draws ŷ ± mean RMSE
    for that model (test trials in the same session × DBS as the exemplar).

    ``use_absolute_time``: if True, x = stored trial time axis; if False, x − x[0].
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    c_true = true_line_color(theme)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
    )

    def _all_nan(arr: np.ndarray) -> bool:
        return np.all(np.isnan(arr))

    def _add_rmse_band(
        row: int,
        show_leg: bool,
        t: np.ndarray,
        y_hat: np.ndarray,
        half: Optional[float],
        fill_c: str,
        line_c: str,
        name: str,
        leg_group: str,
    ) -> None:
        if half is None or not np.isfinite(half) or half <= 0 or _all_nan(y_hat) or len(t) != len(y_hat):
            return
        upper = np.asarray(y_hat, dtype=float) + half
        lower = np.asarray(y_hat, dtype=float) - half
        x_band = np.concatenate([t, t[::-1]])
        y_band = np.concatenate([upper, lower[::-1]])
        fig.add_trace(
            go.Scatter(
                x=x_band,
                y=y_band,
                fill="toself",
                fillcolor=fill_c,
                line=dict(color=line_c, width=0.5),
                name=name,
                legendgroup=leg_group,
                showlegend=show_leg,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

    y_ranges: list[tuple[float, float]] = []

    for row, panel in enumerate((panel_off, panel_on), start=1):
        t_raw = np.asarray(panel.t_abs, dtype=float)
        t = t_raw if use_absolute_time else (t_raw - t_raw[0])
        zt = np.asarray(panel.z_true, dtype=float)
        zp = np.asarray(panel.z_psid, dtype=float)
        zd = np.asarray(panel.z_dpad, dtype=float)
        zv = np.asarray(panel.z_varma, dtype=float)
        sigma = panel.psid_sigma

        show_leg = row == 1

        has_session_bands = any(
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

        if has_session_bands:
            _add_rmse_band(
                row, False, t, zp, panel.band_rmse_psid,
                COLOR_PSID_BAND_FILL, COLOR_PSID_BAND_LINE,
                "PSID ±1 SEM", "psid_sess_rmse",
            )
            _add_rmse_band(
                row, False, t, zd, panel.band_rmse_dpad,
                COLOR_DPAD_BAND_FILL, COLOR_DPAD_BAND_LINE,
                "DPAD ±1 SEM", "dpad_sess_rmse",
            )
            _add_rmse_band(
                row, False, t, zv, panel.band_rmse_varma,
                COLOR_VARMA_BAND_FILL, COLOR_VARMA_BAND_LINE,
                "VARMA ±1 SEM", "varma_sess_rmse",
            )
        elif (
            show_psid_sigma_band
            and not _all_nan(zp)
            and sigma is not None
            and sigma > 0
            and len(t) == len(zp)
        ):
            upper = zp + sigma
            lower = zp - sigma
            x_band = np.concatenate([t, t[::-1]])
            y_band = np.concatenate([upper, lower[::-1]])
            fig.add_trace(
                go.Scatter(
                    x=x_band,
                    y=y_band,
                    fill="toself",
                    fillcolor=COLOR_PSID_BAND_FILL,
                    line=dict(color=COLOR_PSID_BAND_LINE, width=0.5),
                    name="PSID ±1σ",
                    legendgroup="psid_band",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=zt,
                mode="lines",
                name=true_series_name,
                legendgroup="true",
                line=dict(color=c_true, width=WIDTH_TRUE),
                showlegend=show_leg,
            ),
            row=row,
            col=1,
        )

        _pd = prediction_line_dash
        if not _all_nan(zp):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=zp,
                    mode="lines",
                    name="y_hat_PSID",
                    legendgroup="psid",
                    line=dict(
                        color=COLOR_PSID,
                        width=WIDTH_PSID,
                        dash=_pd if _pd else None,
                    ),
                    opacity=OPACITY_PSID,
                    showlegend=show_leg,
                ),
                row=row,
                col=1,
            )

        if not _all_nan(zd):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=zd,
                    mode="lines",
                    name="y_hat_DPAD",
                    legendgroup="dpad",
                    line=dict(
                        color=COLOR_DPAD,
                        width=WIDTH_DPAD,
                        dash=_pd if _pd else None,
                    ),
                    opacity=OPACITY_DPAD,
                    showlegend=show_leg,
                ),
                row=row,
                col=1,
            )

        if not _all_nan(zv):
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=zv,
                    mode="lines",
                    name="y_hat_VARMA",
                    legendgroup="varma",
                    line=dict(
                        color=COLOR_VARMA,
                        width=WIDTH_VARMA,
                        dash="8 2" if prediction_line_dash else "dash",
                    ),
                    opacity=OPACITY_VARMA,
                    showlegend=show_leg,
                ),
                row=row,
                col=1,
            )

        y_ranges.append(
            _y_range_from_values(
                _finite_ys_for_autoscale(
                    zt, zp, zd, zv,
                    panel.band_rmse_psid if has_session_bands else None,
                    panel.band_rmse_dpad if has_session_bands else None,
                    panel.band_rmse_varma if has_session_bands else None,
                )
            )
        )

        badge_fg, badge_bg = dbs_badge_style(panel.dbs_label)
        fig.add_annotation(
            row=row,
            col=1,
            xref="x domain",
            yref="y domain",
            x=0.01,
            y=0.97,
            xanchor="left",
            yanchor="top",
            text=f"<b>{panel.dbs_label}</b>",
            showarrow=False,
            font=dict(size=FONT_SIZE_BASE, family=FONT_FAMILY, color=badge_fg),
            bgcolor=badge_bg,
            borderpad=4,
            bordercolor=badge_fg,
            borderwidth=1,
        )

    y0_lo = min(y_ranges[0][0], y_ranges[1][0])
    y0_hi = max(y_ranges[0][1], y_ranges[1][1])

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linewidth=1,
        linecolor=c_true,
        mirror=False,
        title_text=THESIS_TIME_AXIS_TITLE,
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY, color=c_true),
        showticklabels=False,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linewidth=1,
        linecolor=c_true,
        mirror=False,
        title_text=THESIS_TIME_AXIS_TITLE,
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY, color=c_true),
        tickfont=dict(size=FONT_SIZE_TICK),
        row=2,
        col=1,
    )

    for row in (1, 2):
        fig.update_yaxes(
            title_text=y_axis_label,
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY, color=c_true),
            showgrid=True,
            gridcolor=grid,
            showline=True,
            linewidth=1,
            linecolor=c_true,
            mirror=False,
            tickfont=dict(size=FONT_SIZE_TICK),
            range=[y0_lo, y0_hi],
            row=row,
            col=1,
        )

    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=c_true),
        height=FIGURE_HEIGHT_STACKED,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            bgcolor=legend_bgcolor(),
            font=dict(size=FONT_SIZE_TICK),
            itemsizing="constant",
            itemwidth=40,
            tracegroupgap=8,
        ),
        margin=dict(l=88, r=24, t=36, b=140),
        hovermode="x unified",
    )

    return fig


def _slice_trial_center(
    t_abs: np.ndarray,
    seg_s: float,
    z_true: np.ndarray,
    z_psid: np.ndarray,
    z_dpad: np.ndarray,
    z_varma: np.ndarray,
    *,
    side_extend_s: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t_abs, dtype=float).ravel()
    if t.size == 0:
        e = np.array([], dtype=float)
        return e, e, e, e, e
    t0e, t1e = float(np.nanmin(t)), float(np.nanmax(t))
    span = max(t1e - t0e, 1e-9)
    mid = 0.5 * (t0e + t1e)
    ext = max(float(side_extend_s), 0.0)
    half = min(seg_s / 2.0 + ext, max(span / 2.0 - 1e-6, 1e-6))
    lo, hi = mid - half, mid + half
    m = (t >= lo) & (t <= hi)
    return (
        t[m],
        np.asarray(z_true, dtype=float).ravel()[m],
        np.asarray(z_psid, dtype=float).ravel()[m],
        np.asarray(z_dpad, dtype=float).ravel()[m],
        np.asarray(z_varma, dtype=float).ravel()[m],
    )


def _concat_gap(
    t_a: np.ndarray,
    ys_a: Tuple[np.ndarray, ...],
    t_b: np.ndarray,
    ys_b: Tuple[np.ndarray, ...],
    gap_s: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    nan_t = np.array([np.nan], dtype=float)
    nan_y = np.array([np.nan], dtype=float)
    shift = float(t_a[-1]) + gap_s - float(t_b[0])
    t_b2 = t_b + shift
    t_out = np.concatenate([t_a, nan_t, t_b2])
    out: List[np.ndarray] = []
    for ya, yb in zip(ys_a, ys_b):
        out.append(np.concatenate([ya, nan_y, yb]))
    return t_out, out


def build_combined_gap_exemplar_figure(
    panel_off: ThesisPanelData,
    panel_on: ThesisPanelData,
    theme: ThesisTheme,
    y_axis_label: str,
    *,
    segment_s: float = 1.0,
    gap_s: float = 0.1,
    side_extend_s: float = 0.0,
    show_psid_sigma_band: bool = False,
    true_series_name: str = "y_true",
    prediction_line_dash: Optional[str] = None,
    use_absolute_time: bool = True,
) -> go.Figure:
    """
    Single axes: DBS-OFF then a NaN gap then DBS-ON (middle ``segment_s`` of each trial, no line across gap).

    Bottom x = concatenated absolute trial clock (ON segment shifted right by ``gap_s``).
    Top x = seconds from the OFF segment start through the ON segment (continuous relative timeline).
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    c_true = true_line_color(theme)
    _pd = prediction_line_dash

    def _prep(p: ThesisPanelData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t_raw = np.asarray(p.t_abs, dtype=float)
        t = t_raw if use_absolute_time else (t_raw - t_raw[0])
        return _slice_trial_center(
            t,
            segment_s,
            p.z_true,
            p.z_psid,
            p.z_dpad,
            p.z_varma,
            side_extend_s=side_extend_s,
        )

    to, zto, zpo, zdo, zvo = _prep(panel_off)
    tn, ztn, zpn, zdn, zvn = _prep(panel_on)
    if to.size == 0 or tn.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient time samples for combined exemplar", showarrow=False)
        return fig

    t_cat, ys_cat = _concat_gap(
        to,
        (zto, zpo, zdo, zvo),
        tn,
        (ztn, zpn, zdn, zvn),
        gap_s,
    )
    zt_c, zp_c, zd_c, zv_c = ys_cat

    finite_x = t_cat[np.isfinite(t_cat)]
    lo, hi = float(np.min(finite_x)), float(np.max(finite_x))
    tickvals = np.linspace(lo, hi, 15)

    t_anchor = float(to[0])

    def _rel_at(xv: float) -> float:
        # Concatenated absolute x: seconds from start of OFF middle segment through gap and ON segment.
        if not np.isfinite(xv):
            return float("nan")
        return float(xv - t_anchor)

    rel_lbl = [f"{_rel_at(float(tv)):.1f}" for tv in tickvals]
    abs_lbl = [f"{tv:.1f}" for tv in tickvals]

    vals = _finite_ys_for_autoscale(
        zto, zpo, zdo, zvo, panel_off.band_rmse_psid, panel_off.band_rmse_dpad, panel_off.band_rmse_varma
    ) + _finite_ys_for_autoscale(
        ztn, zpn, zdn, zvn, panel_on.band_rmse_psid, panel_on.band_rmse_dpad, panel_on.band_rmse_varma
    )
    y0, y1 = _y_range_from_values(vals)

    fig = go.Figure()

    def _band_segment(
        x_seg: np.ndarray,
        y_hat: np.ndarray,
        half: Optional[float],
        fill_c: str,
        line_c: str,
        name: str,
        leg: str,
        show_leg: bool,
    ) -> None:
        if half is None or not np.isfinite(half) or half <= 0:
            return
        yh = np.asarray(y_hat, dtype=float)
        if len(x_seg) != len(yh) or np.all(np.isnan(yh)):
            return
        u = yh + half
        lo_b = yh - half
        xb = np.concatenate([x_seg, x_seg[::-1]])
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
                showlegend=show_leg,
                hoverinfo="skip",
            )
        )

    shift = float(to[-1]) + gap_s - float(tn[0])
    x_on_cat = tn + shift
    # Per-segment session-mean RMSE ribbons (half-width differs OFF vs ON).
    _band_segment(
        to,
        zpo,
        panel_off.band_rmse_psid,
        COLOR_PSID_BAND_FILL,
        COLOR_PSID_BAND_LINE,
        "PSID ± mean RMSE",
        "bps",
        False,
    )
    _band_segment(
        x_on_cat,
        zpn,
        panel_on.band_rmse_psid,
        COLOR_PSID_BAND_FILL,
        COLOR_PSID_BAND_LINE,
        "PSID ± mean RMSE",
        "bps",
        False,
    )
    _band_segment(
        to,
        zdo,
        panel_off.band_rmse_dpad,
        COLOR_DPAD_BAND_FILL,
        COLOR_DPAD_BAND_LINE,
        "DPAD ± mean RMSE",
        "bpd",
        False,
    )
    _band_segment(
        x_on_cat,
        zdn,
        panel_on.band_rmse_dpad,
        COLOR_DPAD_BAND_FILL,
        COLOR_DPAD_BAND_LINE,
        "DPAD ± mean RMSE",
        "bpd",
        False,
    )
    _band_segment(
        to,
        zvo,
        panel_off.band_rmse_varma,
        COLOR_VARMA_BAND_FILL,
        COLOR_VARMA_BAND_LINE,
        "VARMA ± mean RMSE",
        "bv",
        False,
    )
    _band_segment(
        x_on_cat,
        zvn,
        panel_on.band_rmse_varma,
        COLOR_VARMA_BAND_FILL,
        COLOR_VARMA_BAND_LINE,
        "VARMA ± mean RMSE",
        "bv",
        False,
    )

    fig.add_trace(
        go.Scatter(
            x=t_cat,
            y=zt_c,
            mode="lines",
            name=true_series_name,
            line=dict(color=c_true, width=WIDTH_TRUE),
            connectgaps=False,
        )
    )
    if not np.all(np.isnan(zp_c)):
        fig.add_trace(
            go.Scatter(
                x=t_cat,
                y=zp_c,
                mode="lines",
                name="y_hat_PSID",
                line=dict(color=COLOR_PSID, width=WIDTH_PSID, dash=_pd if _pd else None),
                opacity=OPACITY_PSID,
                connectgaps=False,
            )
        )
    if not np.all(np.isnan(zd_c)):
        fig.add_trace(
            go.Scatter(
                x=t_cat,
                y=zd_c,
                mode="lines",
                name="y_hat_DPAD",
                line=dict(color=COLOR_DPAD, width=WIDTH_DPAD, dash=_pd if _pd else None),
                opacity=OPACITY_DPAD,
                connectgaps=False,
            )
        )
    if not np.all(np.isnan(zv_c)):
        fig.add_trace(
            go.Scatter(
                x=t_cat,
                y=zv_c,
                mode="lines",
                name="y_hat_VARMA",
                line=dict(color=COLOR_VARMA, width=WIDTH_VARMA, dash="8 2" if _pd else "dash"),
                opacity=OPACITY_VARMA,
                connectgaps=False,
            )
        )

    sigma = panel_off.psid_sigma
    if show_psid_sigma_band and sigma is not None and sigma > 0:
        for x_seg, y_seg, leg in (
            (to, zpo, True),
            (x_on_cat, zpn, False),
        ):
            if np.all(np.isnan(y_seg)):
                continue
            u = y_seg + sigma
            lo_b = y_seg - sigma
            xb = np.concatenate([x_seg, x_seg[::-1]])
            yb = np.concatenate([u, lo_b[::-1]])
            fig.add_trace(
                go.Scatter(
                    x=xb,
                    y=yb,
                    fill="toself",
                    fillcolor=COLOR_PSID_BAND_FILL,
                    line=dict(color=COLOR_PSID_BAND_LINE, width=0.5),
                    name="PSID ±1σ",
                    legendgroup="psid_sig",
                    showlegend=leg,
                    hoverinfo="skip",
                )
            )

    fig.add_annotation(
        x=float(np.nanmean(to)),
        y=0.97,
        xref="x",
        yref="y domain",
        text="<b>DBS-OFF</b>",
        showarrow=False,
        yanchor="top",
        font=dict(size=FONT_SIZE_BASE, family=FONT_FAMILY, color=c_true),
    )
    tnm = tn + (to[-1] + gap_s - tn[0])
    fig.add_annotation(
        x=float(np.nanmean(tnm)),
        y=0.97,
        xref="x",
        yref="y domain",
        text="<b>DBS-ON</b>",
        showarrow=False,
        yanchor="top",
        font=dict(size=FONT_SIZE_BASE, family=FONT_FAMILY, color=c_true),
    )

    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=c_true),
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=abs_lbl,
            title=dict(text=THESIS_TIME_AXIS_TITLE, font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY, color=c_true)),
            range=[lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)],
            showgrid=True,
            gridcolor=grid,
        ),
        yaxis=dict(
            title=dict(text=y_axis_label, font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY, color=c_true)),
            range=[y0, y1],
            showgrid=True,
            gridcolor=grid,
        ),
        xaxis2=dict(
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=tickvals,
            ticktext=rel_lbl,
            title=dict(
                text="Relative time from OFF segment start (s)",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY, color=c_true),
            ),
            showgrid=False,
            tickfont=dict(size=FONT_SIZE_TICK),
        ),
        height=int(FIGURE_HEIGHT_STACKED * 0.62),
        margin=dict(l=88, r=24, t=72, b=88),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            bgcolor=legend_bgcolor(),
            font=dict(size=FONT_SIZE_TICK),
        ),
        hovermode="x unified",
    )
    return fig


def _slice_trial_tail(
    t_abs: np.ndarray,
    seg_s: float,
    z_true: np.ndarray,
    z_psid: np.ndarray,
    z_dpad: np.ndarray,
    z_varma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the last ``seg_s`` seconds of each array; t retains absolute values."""
    t = np.asarray(t_abs, dtype=float).ravel()
    if t.size == 0:
        e = np.array([], dtype=float)
        return e, e, e, e, e
    t_hi = float(np.nanmax(t))
    t_lo = t_hi - float(seg_s)
    m = t >= t_lo
    arrays = [np.asarray(a, dtype=float).ravel() for a in (z_true, z_psid, z_dpad, z_varma)]
    return (t[m],) + tuple(a[m] for a in arrays)  # type: ignore[return-value]


def build_side_by_side_exemplar_figure(
    panel_off: ThesisPanelData,
    panel_on: ThesisPanelData,
    theme: ThesisTheme,
    y_axis_label: str,
    *,
    segment_s: float = 1.0,
    true_series_name: str = "y_true",
    prediction_line_dash: Optional[str] = None,
) -> go.Figure:
    """
    Two side-by-side subplots: DBS-OFF (left) | DBS-ON (right).
    Each shows the last ``segment_s`` seconds of the trial.
    Bottom x-axis: absolute trial time; top x-axis: relative time within the window (0 → segment_s).
    RMSE ribbons (session-mean ± half-width) are drawn around each prediction.
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    c_true = true_line_color(theme)
    _pd = prediction_line_dash

    def _prep(p: ThesisPanelData):
        t_raw = np.asarray(p.t_abs, dtype=float)
        # Make trial-relative (0 = trial start), then slice last segment_s seconds
        t_trial = (t_raw - t_raw[0]) if t_raw.size > 0 else t_raw
        t_sl, *zs = _slice_trial_tail(t_trial, segment_s, p.z_true, p.z_psid, p.z_dpad, p.z_varma)
        # trial_offset = start of window in trial-relative coords (e.g. 8.0)
        trial_offset = float(np.nanmin(t_sl)) if t_sl.size > 0 else 0.0
        # window-relative time: 0 → ~segment_s  (used for traces)
        t_win = (t_sl - trial_offset) if t_sl.size > 0 else t_sl
        return t_win, trial_offset, *zs

    to, off_offset, zto, zpo, zdo, zvo = _prep(panel_off)
    tn, on_offset, ztn, zpn, zdn, zvn = _prep(panel_on)

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=["DBS-OFF", "DBS-ON"],
        horizontal_spacing=0.08,
    )

    if to.size == 0 and tn.size == 0:
        fig.add_annotation(text="Insufficient time samples", showarrow=False)
        return fig

    vals = (
        _finite_ys_for_autoscale(
            zto, zpo, zdo, zvo,
            panel_off.band_rmse_psid, panel_off.band_rmse_dpad, panel_off.band_rmse_varma,
        )
        + _finite_ys_for_autoscale(
            ztn, zpn, zdn, zvn,
            panel_on.band_rmse_psid, panel_on.band_rmse_dpad, panel_on.band_rmse_varma,
        )
    )
    y0, y1 = _y_range_from_values(vals)

    # Bottom axis: window time (0 → segment_s).
    # Top axis: trial-relative time (e.g. 8.0 → 9.0 s from trial start).
    n_ticks = 6

    def _dual_ticks(t_win: np.ndarray, trial_offset: float):
        if t_win.size == 0:
            return [], [], []
        w0, w1 = float(np.nanmin(t_win)), float(np.nanmax(t_win))
        win_vals = np.linspace(w0, w1, n_ticks)
        trial_vals = win_vals + trial_offset
        return (
            [float(v) for v in win_vals],                # tick positions (window coords)
            [f"{v:.2f}" for v in win_vals],               # bottom labels: "0.00" … "1.00"
            [f"{v:.1f}" for v in trial_vals],             # top labels: "8.0" … "9.0"
        )

    ticks_off, win_text_off, trial_text_off = _dual_ticks(to, off_offset)
    ticks_on,  win_text_on,  trial_text_on  = _dual_ticks(tn, on_offset)

    for col, t, zt, zp, zd, zv, panel in (
        (1, to, zto, zpo, zdo, zvo, panel_off),
        (2, tn, ztn, zpn, zdn, zvn, panel_on),
    ):
        if t.size == 0:
            continue
        show_leg = col == 1

        # RMSE ribbon bands (session-mean ± half-width)
        for y_hat, half, fc, lc, name, lg in (
            (zp, panel.band_rmse_psid,  COLOR_PSID_BAND_FILL,  COLOR_PSID_BAND_LINE,  "PSID ± mean RMSE",  "bps"),
            (zd, panel.band_rmse_dpad,  COLOR_DPAD_BAND_FILL,  COLOR_DPAD_BAND_LINE,  "DPAD ± mean RMSE",  "bpd"),
            (zv, panel.band_rmse_varma, COLOR_VARMA_BAND_FILL, COLOR_VARMA_BAND_LINE, "VARMA ± mean RMSE", "bpv"),
        ):
            if half is None or not np.isfinite(half) or half <= 0:
                continue
            yh = np.asarray(y_hat, dtype=float)
            if len(t) != len(yh) or np.all(np.isnan(yh)):
                continue
            xb = np.concatenate([t, t[::-1]])
            yb = np.concatenate([yh + half, (yh - half)[::-1]])
            fig.add_trace(
                go.Scatter(
                    x=xb, y=yb, fill="toself",
                    fillcolor=fc, line=dict(color=lc, width=0.5),
                    name=name, legendgroup=lg, showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )

        show_leg = col == 1
        fig.add_trace(
            go.Scatter(
                x=t, y=zt, name=true_series_name, mode="lines",
                line=dict(color=c_true, width=WIDTH_TRUE),
                legendgroup="true", showlegend=show_leg,
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=zp, name="y_hat_PSID", mode="lines",
                line=dict(color=COLOR_PSID, width=WIDTH_PSID, dash=_pd),
                opacity=OPACITY_PSID,
                legendgroup="psid", showlegend=show_leg,
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=zd, name="y_hat_DPAD", mode="lines",
                line=dict(color=COLOR_DPAD, width=WIDTH_DPAD, dash=_pd),
                opacity=OPACITY_DPAD,
                legendgroup="dpad", showlegend=show_leg,
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=zv, name="y_hat_VARMA", mode="lines",
                line=dict(color=COLOR_VARMA, width=WIDTH_VARMA, dash=_pd),
                opacity=OPACITY_VARMA,
                legendgroup="varma", showlegend=show_leg,
            ),
            row=1, col=col,
        )

    for col in (1, 2):
        fig.update_yaxes(
            title_text=y_axis_label if col == 1 else "",
            showgrid=True, gridcolor=grid,
            range=[y0, y1],
            row=1, col=col,
        )

    # Bottom axes: window-relative time (0 → segment_s)
    fig.update_xaxes(
        tickmode="array", tickvals=ticks_off, ticktext=win_text_off,
        title_text="Time (s)",
        showgrid=True, gridcolor=grid,
        row=1, col=1,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=ticks_on, ticktext=win_text_on,
        title_text="Time (s)",
        showgrid=True, gridcolor=grid,
        row=1, col=2,
    )

    # Top axes: trial-relative time (e.g. 8.0 → 9.0 s from trial start)
    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_BASE, color=c_true),
        height=380,
        margin=dict(l=72, r=24, t=80, b=96),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.30,
            xanchor="center",
            x=0.5,
            bgcolor=legend_bgcolor(),
            font=dict(size=FONT_SIZE_TICK),
        ),
        hovermode="x unified",
        xaxis3=dict(
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=ticks_off,
            ticktext=trial_text_off,
            title=dict(text="trial time (s)", font=dict(size=FONT_SIZE_TICK)),
            showgrid=False,
            zeroline=False,
        ),
        xaxis4=dict(
            overlaying="x2",
            side="top",
            tickmode="array",
            tickvals=ticks_on,
            ticktext=trial_text_on,
            title=dict(text="trial time (s)", font=dict(size=FONT_SIZE_TICK)),
            showgrid=False,
            zeroline=False,
        ),
    )
    return fig
