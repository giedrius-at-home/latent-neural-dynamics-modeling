import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from textwrap import wrap

FONT_FAMILY = "Montserrat"


def _create_base_figure(title: str, x_axis_title: str, y_axis_title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16, family=FONT_FAMILY)),
        xaxis_title=dict(text=x_axis_title, font=dict(size=14, family=FONT_FAMILY)),
        yaxis_title=dict(text=y_axis_title, font=dict(size=14, family=FONT_FAMILY)),
        template="plotly_white",
        font=dict(family=FONT_FAMILY, size=12, color="#0e131f"),
        legend=dict(font=dict(size=12, family=FONT_FAMILY)),
        showlegend=True,
        margin=dict(l=60, r=80, t=40, b=60),
    )
    return fig


def _wrap_title(title: str, width: int = 60) -> str:
    return "<br>".join(wrap(title, width=width))


def _setup_dual_time_axis(
    fig: go.Figure, time_abs: np.ndarray, onset_time: float, y_data: np.ndarray = None
) -> go.Figure:
    if time_abs is None or len(time_abs) == 0:
        return fig

    time_abs = np.asarray(time_abs, dtype=float)
    rel_offset = float(onset_time) if onset_time is not None else float(time_abs.min())

    n_ticks = 5
    tickvals = np.linspace(time_abs.min(), time_abs.max(), n_ticks)
    rel_ticktext = [f"{(tv - rel_offset):.1f}" for tv in tickvals]
    abs_ticktext = [f"{tv:.1f}" for tv in tickvals]

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=abs_ticktext,
        title_text="Time (s)",
        title_font=dict(size=14, family=FONT_FAMILY),
        tickfont=dict(size=12),
    )

    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=tickvals,
            ticktext=rel_ticktext,
            title_text="",
            tickfont=dict(size=12),
            range=[float(time_abs.min()), float(time_abs.max())],
            matches="x",
            showgrid=False,
        )
    )

    if y_data is not None and len(y_data) > 0:
        y_min, y_max = float(np.nanmin(y_data)), float(np.nanmax(y_data))
    else:
        y_range = (
            fig.layout.yaxis.range
            if hasattr(fig.layout, "yaxis") and fig.layout.yaxis.range
            else None
        )
        if y_range:
            y_min, y_max = y_range[0], y_range[1]
        else:
            all_y = []
            for trace in fig.data:
                if hasattr(trace, "y") and trace.y is not None:
                    all_y.extend(
                        [y for y in trace.y if y is not None and not np.isnan(y)]
                    )
            if all_y:
                y_min, y_max = min(all_y), max(all_y)
            else:
                y_min, y_max = 0, 1

    y_span = y_max - y_min

    fig.add_annotation(
        text="relative",
        xref="paper",
        yref="y",
        x=1.02,
        y=y_max - 0.15 * y_span,
        showarrow=False,
        font=dict(size=11, family=FONT_FAMILY, color="#59546c"),
        xanchor="left",
    )

    fig.add_annotation(
        text="absolute",
        xref="paper",
        yref="y",
        x=1.02,
        y=y_min + 0.15 * y_span,
        showarrow=False,
        font=dict(size=11, family=FONT_FAMILY, color="#59546c"),
        xanchor="left",
    )

    return fig


def plot_trial_channel(
    trial_df: pl.DataFrame,
    channel: str,
    use_absolute_time: bool = True,
    add_dual_axis: bool = True,
) -> go.Figure:
    if trial_df.is_empty():
        return go.Figure().update_layout(title_text=f"No data for {channel}")

    participant_id = trial_df["participant_id"][0]
    session = trial_df["session"][0] if "session" in trial_df.columns else "?"
    block = trial_df["block"][0] if "block" in trial_df.columns else "?"
    trial = trial_df["trial"][0]
    stim_col = "stim"
    dbs_state = trial_df[stim_col][0]

    title = _wrap_title(
        f"{channel.replace('_', ' ')} Signal (P{participant_id}, S{session}, B{block}, T{trial}, Stim {dbs_state})"
    )
    x_axis_title = "Time (s)"
    fig = _create_base_figure(title, x_axis_title, "Amplitude (µV)")

    time_data = trial_df["time"].to_numpy()
    channel_data = trial_df[channel].to_numpy()

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=channel_data,
            mode="lines",
            name=channel,
            line=dict(color="black", width=1),
        )
    )

    chunk_margin = trial_df["chunk_margin"][0]
    original_duration = trial_df["margined_duration"][0] - 2 * chunk_margin

    event_start = time_data.min() + chunk_margin
    event_end = event_start + original_duration

    fig.add_vrect(
        x0=event_start,
        x1=event_end,
        fillcolor="rgba(0, 100, 0, 0.1)",
        layer="below",
        line_width=0,
    )
    fig.add_vline(
        x=event_start,
        line_dash="dash",
        line_color="#38405f",
        annotation_text="Event Start",
    )
    fig.add_vline(
        x=event_end, line_dash="dash", line_color="#59546c", annotation_text="Event End"
    )

    fig.update_layout(showlegend=False)

    if add_dual_axis:
        onset_time = event_start
        fig = _setup_dual_time_axis(fig, time_data, onset_time, channel_data)

    return fig


def plot_trial_coordinates(
    trial_df: pl.DataFrame,
    time: str,
    plot_over_time: bool = False,
    add_dual_axis: bool = True,
) -> go.Figure:
    if trial_df.is_empty() or trial_df["x"].is_null().all():
        return go.Figure().update_layout(title_text="No coordinate data available")

    p_id = trial_df["participant_id"][0]
    session = trial_df["session"][0]
    trial = trial_df["trial"][0]
    block = trial_df["block"][0] if "block" in trial_df.columns else "?"

    if plot_over_time:
        title = _wrap_title(
            f"Coordinates vs. Time (P{p_id}, S{session}, B{block}, T{trial})"
        )
        fig = _create_base_figure(title, "Time (s)", "Position")

        time_data = trial_df[time].to_numpy()
        x_data = trial_df["x"].to_numpy()
        y_data = trial_df["y"].to_numpy()

        fig.add_trace(go.Scatter(x=time_data, y=x_data, mode="lines", name="X-coord"))
        fig.add_trace(go.Scatter(x=time_data, y=y_data, mode="lines", name="Y-coord"))

        if add_dual_axis and len(time_data) > 0:
            onset_time = time_data.min()
            combined_y = np.concatenate([x_data, y_data])
            fig = _setup_dual_time_axis(fig, time_data, onset_time, combined_y)
    else:
        title = _wrap_title(f"2D Trajectory (P{p_id}, S{session}, B{block}, T{trial})")
        fig = _create_base_figure(title, "X Coordinate", "Y Coordinate")
        fig.add_trace(
            go.Scatter(
                x=trial_df["x"], y=trial_df["y"], mode="lines+markers", name="Path"
            )
        )
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        fig.add_trace(
            go.Scatter(
                x=trial_df.head(1)["x"],
                y=trial_df.head(1)["y"],
                mode="markers",
                marker=dict(color="green", size=10),
                name="Start",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trial_df.tail(1)["x"],
                y=trial_df.tail(1)["y"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="End",
            )
        )

    return fig


def plot_tracing_velocity_magnitude(
    trial_df: pl.DataFrame, time: str, add_dual_axis: bool = True
) -> go.Figure:
    if trial_df.is_empty() or trial_df["tracing_velocity_magnitude"].is_null().all():
        return go.Figure().update_layout(title_text="No speed data available")

    p_id = trial_df["participant_id"][0]
    session = trial_df["session"][0]
    trial = trial_df["trial"][0]
    block = trial_df["block"][0] if "block" in trial_df.columns else "?"
    title = _wrap_title(f"Tracing Speed (P{p_id}, S{session}, B{block}, T{trial})")
    fig = _create_base_figure(title, "Time (s)", "Speed (pixels/s)")

    time_data = trial_df[time].to_numpy()
    speed_data = trial_df["tracing_velocity_magnitude"].to_numpy()

    fig.add_trace(go.Scatter(x=time_data, y=speed_data, mode="lines", name="Speed"))
    fig.update_layout(showlegend=False)

    if add_dual_axis and len(time_data) > 0:
        onset_time = time_data.min()
        fig = _setup_dual_time_axis(fig, time_data, onset_time, speed_data)

    return fig


def plot_psd_heatmap(
    freqs: np.ndarray,
    psds: np.ndarray,
    title: str,
    times_abs: np.ndarray | None = None,
    add_rel_axis: bool = False,
    rel_offset: float | None = None,
) -> go.Figure:
    if freqs.size == 0 or psds.size == 0:
        return go.Figure().update_layout(title_text="No PSD data for heatmap")

    x_vals = (
        np.asarray(times_abs, dtype=float)
        if times_abs is not None
        else np.arange(len(psds), dtype=float)
    )

    x_label = "Time (s)" if times_abs is not None else "Time Epoch"
    fig = _create_base_figure(_wrap_title(title), x_label, "Frequency (Hz)")

    psds_arr = np.asarray([np.array(psd, dtype=float) for psd in psds])
    log_psds = 10 * np.log10(psds_arr.T) + 120

    fig.add_trace(
        go.Heatmap(
            z=log_psds,
            x=x_vals,
            y=freqs,
            colorscale="Viridis",
            colorbar=dict(title="Power/Freq (dB/Hz)"),
        )
    )

    if add_rel_axis and times_abs is not None and len(x_vals) > 1:
        rel0 = float(rel_offset) if rel_offset is not None else float(x_vals.min())
        n_ticks = 5
        tickvals = np.linspace(x_vals.min(), x_vals.max(), n_ticks)

        rel_ticktext = [f"{(tv - rel0):.1f}" for tv in tickvals]

        abs_ticktext = [f"{tv:.1f}" for tv in tickvals]

        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=abs_ticktext,
            title_text="Time (s)",
            title_font=dict(size=16, family="sans-serif"),
            tickfont=dict(size=13),
        )

        fig.update_layout(
            xaxis2=dict(
                overlaying="x",
                side="top",
                tickmode="array",
                tickvals=tickvals,
                ticktext=rel_ticktext,
                title_text="",
                tickfont=dict(size=13),
                range=[float(x_vals.min()), float(x_vals.max())],
                matches=None,
                showgrid=False,
            )
        )

        y_range = [freqs.min(), freqs.max()]
        y_min, y_max = y_range[0], y_range[1]
        y_span = y_max - y_min

        fig.add_annotation(
            text="relative",
            xref="paper",
            yref="y",
            x=1.12,
            y=y_max - 0.15 * y_span,
            showarrow=False,
            font=dict(size=13, family="sans-serif", color="gray"),
            xanchor="left",
        )

        fig.add_annotation(
            text="absolute",
            xref="paper",
            yref="y",
            x=1.12,
            y=y_min + 0.15 * y_span,
            showarrow=False,
            font=dict(size=13, family="sans-serif", color="gray"),
            xanchor="left",
        )

    return fig


def plot_average_psd(
    freqs: np.ndarray,
    psd_data: dict,
    title: str,
) -> go.Figure:
    fig = _create_base_figure(
        _wrap_title(title), "Frequency (Hz)", "Power/Frequency (dB/Hz)"
    )
    fig.update_layout(legend_title_text="Channel")

    colors = px.colors.qualitative.Plotly
    color_idx = 0

    for channel, data in psd_data.items():
        color = colors[color_idx % len(colors)]

        all_psds = []
        if "on" in data and data["on"].size > 0:
            all_psds.append(data["on"])
        if "off" in data and data["off"].size > 0:
            all_psds.append(data["off"])

        if all_psds:
            combined = np.vstack(all_psds)
            mean_psd = 10 * np.log10(np.mean(combined, axis=0)) + 120
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=mean_psd,
                    mode="lines",
                    name=f"{channel}",
                    line=dict(color=color, width=1.2),
                )
            )

        color_idx += 1

    return fig


def plot_average_psd_dbs_comparison(
    freqs: np.ndarray,
    psd_data: dict,
    title: str,
) -> go.Figure:
    fig = _create_base_figure(
        _wrap_title(title), "Frequency (Hz)", "Power/Frequency (dB/Hz)"
    )
    fig.update_layout(legend_title_text="Channel • DBS State")

    colors = px.colors.qualitative.Plotly
    color_idx = 0

    for channel, data in psd_data.items():
        color = colors[color_idx % len(colors)]

        if "on" in data and data["on"].size > 0:
            mean_psd_on = 10 * np.log10(np.mean(data["on"], axis=0)) + 120
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=mean_psd_on,
                    mode="lines",
                    name=f"{channel} • ON",
                    line=dict(color="#ff0035", width=1.2),  # strawberry_red (stim_on)
                )
            )

        if "off" in data and data["off"].size > 0:
            mean_psd_off = 10 * np.log10(np.mean(data["off"], axis=0)) + 120
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=mean_psd_off,
                    mode="lines",
                    name=f"{channel} • OFF",
                    line=dict(
                        color="#59546c", width=1.2, dash="dot"
                    ),  # vintage_grape (stim_off)
                )
            )

        color_idx += 1

    return fig
