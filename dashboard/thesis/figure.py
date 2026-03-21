from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_DPAD,
    COLOR_PSID,
    COLOR_PSID_BAND_FILL,
    COLOR_PSID_BAND_LINE,
    COLOR_VARMA,
    FONT_FAMILY,
    ThesisTheme,
    grid_color,
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


def build_dbs_stacked_figure(
    panel_off: ThesisPanelData,
    panel_on: ThesisPanelData,
    theme: ThesisTheme,
    y_axis_label: str,
    caption: str,
) -> go.Figure:
    """
    Two rows sharing x: top = DBS-OFF, bottom = DBS-ON. X-axis labels only on bottom.
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    c_true = true_line_color(theme)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
    )

    for row, panel in enumerate((panel_off, panel_on), start=1):
        t = np.asarray(panel.t_abs, dtype=float)
        zt = np.asarray(panel.z_true, dtype=float)
        zp = np.asarray(panel.z_psid, dtype=float)
        zd = np.asarray(panel.z_dpad, dtype=float)
        zv = np.asarray(panel.z_varma, dtype=float)
        sigma = panel.psid_sigma

        show_leg = row == 1

        if sigma is not None and sigma > 0 and len(t) == len(zp):
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
                    name="PSID ±1σ (val. residual σ)",
                    legendgroup="psid_band",
                    showlegend=show_leg,
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
                name="true tracing speed",
                legendgroup="true",
                line=dict(color=c_true, width=WIDTH_TRUE),
                showlegend=show_leg,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=zp,
                mode="lines",
                name="PSID",
                legendgroup="psid",
                line=dict(color=COLOR_PSID, width=WIDTH_PSID),
                showlegend=show_leg,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=zd,
                mode="lines",
                name="DPAD-RNN",
                legendgroup="dpad",
                line=dict(color=COLOR_DPAD, width=WIDTH_DPAD),
                showlegend=show_leg,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=zv,
                mode="lines",
                name="VARMA",
                legendgroup="varma",
                line=dict(
                    color=COLOR_VARMA,
                    width=WIDTH_VARMA,
                    dash="dash",
                ),
                showlegend=show_leg,
            ),
            row=row,
            col=1,
        )

        fig.add_annotation(
            row=row,
            col=1,
            xref="x domain",
            yref="y domain",
            x=0.02,
            y=0.97,
            xanchor="left",
            yanchor="top",
            text=f"<b>{panel.dbs_label}</b>",
            showarrow=False,
            font=dict(size=12, family=FONT_FAMILY, color=c_true),
            bgcolor="rgba(128,128,128,0.15)"
            if theme == ThesisTheme.LIGHT
            else "rgba(255,255,255,0.08)",
            borderpad=4,
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linewidth=1,
        linecolor=c_true,
        mirror=True,
        title_text="",
        showticklabels=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        showline=True,
        linewidth=1,
        linecolor=c_true,
        mirror=True,
        title_text="time (s)",
        title_font=dict(size=12, family=FONT_FAMILY, color=c_true),
        row=2,
        col=1,
    )

    for row in (1, 2):
        fig.update_yaxes(
            title_text=y_axis_label,
            title_font=dict(size=12, family=FONT_FAMILY, color=c_true),
            showgrid=True,
            gridcolor=grid,
            showline=True,
            linewidth=1,
            linecolor=c_true,
            mirror=True,
            range=[-2.5, 2.5],
            row=row,
            col=1,
        )

    fig.update_layout(
        template="plotly_white" if theme == ThesisTheme.LIGHT else "plotly_dark",
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, size=11, color=c_true),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.7)" if theme == ThesisTheme.LIGHT else "rgba(0,0,0,0.4)",
        ),
        margin=dict(l=72, r=24, t=48, b=120),
        annotations=[
            dict(
                text=f"<i>{caption}</i>",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.34,
                xanchor="center",
                yanchor="top",
                showarrow=False,
                font=dict(size=11, family=FONT_FAMILY, color=c_true),
            )
        ],
        hovermode="x unified",
    )

    return fig
