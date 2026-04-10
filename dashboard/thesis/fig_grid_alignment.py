"""Grid alignment figure for the thesis Data Verification section.

Shows, for a single trial, how the irregularly sampled behavioral stream
(iPad tracing / accelerometer, event-driven hardware) is interpolated onto a
uniform 200 Hz master grid shared with the neural data.

Three panels (stacked vertically):

1. Raw behavioral sample times as a rug, overlaid against the uniform 200 Hz
   master-grid rug, showing that raw ticks are denser/sparser in places and
   do not coincide with grid ticks.
2. Raw (connect-the-dots) vs interpolated behavioral signal for a single
   channel (e.g. ``tracing_velocity_x``), on the same time axis.
3. Sampling-jitter histogram of :math:`\\Delta t` between consecutive raw
   samples (whole trial), with the nominal 1 / ``target_fs`` tick marked.

The module intentionally only reads already-on-disk parquet data produced by
the 200 Hz resampling preprocessing; the "raw" irregular samples live in the
same trial rows under ``motion_time`` / ``x`` / ``y``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_PSID,
    COLOR_SEPARATOR,
    COLOR_TRUE_LIGHT,
    FONT_FAMILY,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    true_line_color,
)

_DEFAULT_DATA_ROOT = Path(
    "resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band"
)

# Window (seconds) used for the rug + signal overlay panel. Kept short so the
# reader can actually see individual sample ticks. The jitter histogram uses
# the entire trial regardless of this.
_OVERLAY_WINDOW_S = 0.35


def _load_trial_row(
    *,
    participant: str,
    session: str,
    data_root: Path,
    trial_index: int,
) -> dict:
    """Load a single trial row (as a dict-of-arrays) from the resampled parquet.

    Partition layout is ``participant_id=<p>/session=<s>/block=<b>/<file>.parquet``.
    We select only the columns we need so we do not pull the full ~900-column
    frame (PSD, epochs, etc.) for every trial.
    """
    p_part = f"participant_id={participant}"
    s_part = f"session={session}"
    glob_path = str(data_root / p_part / s_part / "*" / "*.parquet")

    needed = [
        "participant_id",
        "session",
        "block",
        "trial",
        "time_original",
        "motion_time",
        "x",
        "y",
    ]
    # Also pull whichever behavioral channel the caller requested, if it is a
    # derived (*_velocity_*, *_acceleration_*, *_jerk_*) column that already
    # sits in the file.
    df = pl.read_parquet(glob_path, columns=None, rechunk=False)

    # Sort by (block, trial) and pick the requested zero-based index.
    df = df.sort(["block", "trial"])
    if df.height == 0:
        raise RuntimeError(
            f"No trials found under {glob_path!s}; check participant/session."
        )
    if trial_index >= df.height:
        raise IndexError(
            f"trial_index={trial_index} out of range (session has {df.height} trials)."
        )

    row = df.row(trial_index, named=True)
    missing = [k for k in needed if row.get(k) is None]
    if missing:
        raise RuntimeError(
            f"Trial row is missing required columns: {missing}. "
            f"participant={participant} session={session} idx={trial_index}."
        )
    return row


def _raw_and_interp_signal(
    row: dict,
    channel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_raw, y_raw, t_grid, y_interp) for the requested channel.

    For derived channels (``tracing_velocity_x`` etc.), the interpolated
    version is read directly from the parquet (those columns are stored on
    the 200 Hz grid) and the raw version is computed from finite differences
    of ``x`` / ``y`` on ``motion_time`` — matching how the pipeline derives
    them before resampling.
    """
    t_raw = np.asarray(row["motion_time"], dtype=float)
    t_grid = np.asarray(row["time_original"], dtype=float)
    x_raw = np.asarray(row["x"], dtype=float)
    y_raw = np.asarray(row["y"], dtype=float)

    # Guard: sort raw by time if not monotonic (rare, but possible).
    order = np.argsort(t_raw)
    t_raw = t_raw[order]
    x_raw = x_raw[order]
    y_raw = y_raw[order]

    ch = channel.lower()
    if ch.endswith("_x"):
        pos_raw = x_raw
    elif ch.endswith("_y"):
        pos_raw = y_raw
    else:
        # Fall back to x if the channel name does not carry an axis suffix.
        pos_raw = x_raw

    if "velocity" in ch:
        sig_raw = _finite_diff(pos_raw, t_raw, order=1)
    elif "acceleration" in ch:
        sig_raw = _finite_diff(pos_raw, t_raw, order=2)
    elif "jerk" in ch:
        sig_raw = _finite_diff(pos_raw, t_raw, order=3)
    else:
        sig_raw = pos_raw

    # Interpolated version: prefer the pre-computed column if it exists in
    # the parquet row; otherwise, interpolate the raw signal onto the grid.
    if channel in row and row[channel] is not None:
        sig_grid = np.asarray(row[channel], dtype=float)
        if sig_grid.shape[0] != t_grid.shape[0]:
            sig_grid = np.interp(t_grid, t_raw, sig_raw)
    else:
        sig_grid = np.interp(t_grid, t_raw, sig_raw)

    return t_raw, sig_raw, t_grid, sig_grid


def _finite_diff(x: np.ndarray, t: np.ndarray, *, order: int) -> np.ndarray:
    """Central finite differences of ``x(t)`` on irregular ``t``."""
    out = np.asarray(x, dtype=float).copy()
    for _ in range(order):
        dt = np.diff(t)
        dy = np.diff(out)
        # Avoid divide-by-zero from duplicated timestamps.
        dt = np.where(dt == 0, np.nan, dt)
        deriv = dy / dt
        # Pad back to length N (repeat last value to keep arrays aligned).
        out = np.concatenate([deriv, [deriv[-1] if deriv.size else 0.0]])
    return out


def _axis_label_for_channel(channel: str) -> str:
    """Return a publication-friendly y-axis label for a behavioral channel."""
    ch = channel.lower()
    if "velocity" in ch:
        return f"{channel} (pixels / s)"
    if "acceleration" in ch:
        return f"{channel} (pixels / s\u00b2)"
    if "jerk" in ch:
        return f"{channel} (pixels / s\u00b3)"
    return f"{channel} (pixels)"


def build_grid_alignment_figure(
    *,
    participant: str,
    session: str,
    data_root: Path = _DEFAULT_DATA_ROOT,
    behavioral_channel: str = "tracing_velocity_x",
    trial_index: int = 0,
    target_fs: float = 200.0,
    theme: ThesisTheme = ThesisTheme.LIGHT,
) -> Figure:
    """Build the grid alignment verification figure for one trial.

    See module docstring for panel layout.
    """
    row = _load_trial_row(
        participant=participant,
        session=session,
        data_root=data_root,
        trial_index=trial_index,
    )

    t_raw, sig_raw, t_grid, sig_grid = _raw_and_interp_signal(row, behavioral_channel)

    # ------------------------------------------------------------------ windows
    # Center the overlay window on the trial midpoint so that we are not
    # showing edge-padding artefacts.
    t_start_full = float(min(t_raw.min(), t_grid.min()))
    t_end_full = float(max(t_raw.max(), t_grid.max()))
    t_mid = 0.5 * (t_start_full + t_end_full)
    w_half = _OVERLAY_WINDOW_S / 2.0
    win_lo = max(t_start_full, t_mid - w_half)
    win_hi = min(t_end_full, t_mid + w_half)

    m_raw = (t_raw >= win_lo) & (t_raw <= win_hi)
    m_grid = (t_grid >= win_lo) & (t_grid <= win_hi)

    t_raw_win = t_raw[m_raw]
    sig_raw_win = sig_raw[m_raw]
    t_grid_win = t_grid[m_grid]
    sig_grid_win = sig_grid[m_grid]

    # Time axes rebased to 0 (ms) for reader-friendly ticks inside the window.
    t0 = win_lo
    t_raw_ms = (t_raw_win - t0) * 1000.0
    t_grid_ms = (t_grid_win - t0) * 1000.0

    # Jitter histogram uses the entire raw stream (trial-wide).
    dt_ms = np.diff(t_raw) * 1000.0
    nominal_dt_ms = 1000.0 / float(target_fs)

    # ------------------------------------------------------------------ figure
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.14,
        row_heights=[0.18, 0.52, 0.30],
        subplot_titles=(
            "Sampling times: raw behavioral vs 200 Hz master grid",
            f"Raw vs interpolated signal — {behavioral_channel}",
            "Inter-sample interval distribution (\u0394t)",
        ),
    )

    fg = true_line_color(theme)
    raw_color = COLOR_TRUE_LIGHT if theme == ThesisTheme.LIGHT else "#D3D1C7"
    grid_color_line = COLOR_PSID

    # ---- Row 1: rug marks ---------------------------------------------------
    rug_grid_y = np.full_like(t_grid_ms, 1.0, dtype=float)
    rug_raw_y = np.full_like(t_raw_ms, 0.0, dtype=float)

    fig.add_trace(
        go.Scatter(
            x=t_grid_ms,
            y=rug_grid_y,
            mode="markers",
            name="200 Hz grid ticks",
            marker=dict(
                symbol="line-ns-open",
                size=18,
                color=grid_color_line,
                line=dict(width=1.6, color=grid_color_line),
            ),
            hovertemplate="grid t = %{x:.2f} ms<extra></extra>",
            showlegend=True,
            legendgroup="row1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_raw_ms,
            y=rug_raw_y,
            mode="markers",
            name="Raw behavioral samples",
            marker=dict(
                symbol="line-ns-open",
                size=18,
                color=raw_color,
                line=dict(width=1.6, color=raw_color),
            ),
            hovertemplate="raw t = %{x:.2f} ms<extra></extra>",
            showlegend=True,
            legendgroup="row1",
        ),
        row=1,
        col=1,
    )

    # Row-1 y-axis: hide ticks, just keep the two rug rows visible.
    fig.update_yaxes(
        row=1,
        col=1,
        tickmode="array",
        tickvals=[0.0, 1.0],
        ticktext=["raw", "grid"],
        range=[-0.6, 1.6],
        showgrid=False,
        zeroline=False,
        tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
    )
    fig.update_xaxes(
        row=1,
        col=1,
        title_text="Time within window (ms)",
        showgrid=True,
    )

    # ---- Row 2: raw vs interpolated signal ----------------------------------
    fig.add_trace(
        go.Scatter(
            x=t_raw_ms,
            y=sig_raw_win,
            mode="lines+markers",
            name="Raw (irregular samples)",
            line=dict(color=raw_color, width=1.6, dash="dot"),
            marker=dict(color=raw_color, size=6, symbol="circle-open"),
            hovertemplate="t = %{x:.2f} ms<br>raw = %{y:.3f}<extra></extra>",
            legendgroup="row2",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_grid_ms,
            y=sig_grid_win,
            mode="lines+markers",
            name="Interpolated (200 Hz grid)",
            line=dict(color=grid_color_line, width=2.0),
            marker=dict(color=grid_color_line, size=5, symbol="square"),
            hovertemplate="t = %{x:.2f} ms<br>interp = %{y:.3f}<extra></extra>",
            legendgroup="row2",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(
        row=2,
        col=1,
        title_text="Time within window (ms)",
        showgrid=True,
    )
    fig.update_yaxes(
        row=2,
        col=1,
        title_text=_axis_label_for_channel(behavioral_channel),
        showgrid=True,
    )

    # ---- Row 3: jitter histogram --------------------------------------------
    if dt_ms.size:
        p99 = float(np.nanpercentile(dt_ms, 99.0))
        hi = max(p99 * 1.05, nominal_dt_ms * 2.0)
        fig.add_trace(
            go.Histogram(
                x=dt_ms,
                xbins=dict(start=0.0, end=hi, size=max(hi / 60.0, 0.1)),
                marker=dict(
                    color=COLOR_SEPARATOR,
                    line=dict(color=fg, width=0.6),
                ),
                name="\u0394t between raw samples",
                showlegend=False,
                hovertemplate="\u0394t = %{x:.2f} ms<br>count = %{y}<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.update_xaxes(row=3, col=1, range=[0.0, hi])

    fig.update_xaxes(
        row=3,
        col=1,
        title_text="\u0394t between consecutive raw samples (ms)",
        showgrid=True,
    )
    fig.update_yaxes(
        row=3,
        col=1,
        title_text="count",
        showgrid=True,
    )

    # Nominal 1/fs reference line on the jitter histogram.
    fig.add_vline(
        x=nominal_dt_ms,
        line=dict(color=grid_color_line, width=1.8, dash="dash"),
        row=3,
        col=1,
        annotation=dict(
            text=f"1/{int(target_fs)} Hz = {nominal_dt_ms:.2f} ms",
            font=dict(
                color=grid_color_line,
                size=FONT_SIZE_ANNOTATION,
                family=FONT_FAMILY,
            ),
            showarrow=False,
            yanchor="top",
            y=1.0,
            xanchor="left",
            bgcolor="rgba(255,255,255,0.0)",
        ),
    )

    # ---- Common styling -----------------------------------------------------
    apply_thesis_style(
        fig,
        theme,
        height=600,
        margin=dict(l=86, r=32, t=54, b=96),
        hovermode="closest",
        legend_y=-0.12,
        show_legend=True,
    )

    # Subplot titles get the small annotation treatment so they do not look
    # like real chart titles.
    for ann in fig.layout.annotations:
        if ann.text in {
            "Sampling times: raw behavioral vs 200 Hz master grid",
            f"Raw vs interpolated signal — {behavioral_channel}",
            "Inter-sample interval distribution (\u0394t)",
        }:
            ann.font = dict(
                family=FONT_FAMILY,
                size=FONT_SIZE_LABEL,
                color=fg,
            )

    # Trial identity annotation (small, top-left of figure).
    block_id = row.get("block")
    trial_id = row.get("trial")
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.06,
        text=(
            f"{participant} · session {session} · block {block_id} · trial {trial_id}"
        ),
        showarrow=False,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_ANNOTATION, color=fg),
        xanchor="left",
        yanchor="bottom",
    )

    return fig
