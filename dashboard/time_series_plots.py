import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dashboard.backbone import (
    create_base_time_series_figure,
    PLOT_STYLE,
    PLOT_COLOR,
    PALETTE,
)


def plot_channel_time_series(trial_df: pl.DataFrame, channel: str) -> go.Figure:
    if trial_df.is_empty():
        return go.Figure().update_layout(title_text=f"No data for {channel}")

    time_data = trial_df["time"].to_numpy()
    channel_data = trial_df[channel].to_numpy()
    chunk_margin = trial_df["chunk_margin"][0]

    stim_state = trial_df["stim"][0] if "stim" in trial_df.columns else "off"

    event_start = time_data.min() + chunk_margin
    onset_time = event_start

    original_duration = trial_df["margined_duration"][0] - 2 * chunk_margin
    event_end = event_start + original_duration

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Amplitude (µV)",
        title="",
    )

    if stim_state == "on":
        bg_color = f"rgba(255, 0, 53, 0.08)"
        badge_color = PALETTE.strawberry_red
        badge_text = "DBS ON"
    else:
        bg_color = f"rgba(89, 84, 108, 0.08)"
        badge_color = PALETTE.vintage_grape
        badge_text = "DBS OFF"

    fig.add_vrect(
        x0=time_data.min(),
        x1=time_data.max(),
        fillcolor=bg_color,
        layer="below",
        line_width=0,
    )

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=channel_data,
            mode="lines",
            name=channel,
            line=dict(color="#8B0000", width=1.2),
            showlegend=False,
        )
    )

    fig.add_vrect(
        x0=event_start,
        x1=event_end,
        fillcolor="rgba(0, 150, 0, 0.08)",
        layer="below",
        line_width=0,
    )

    fig.add_vline(
        x=event_start,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text="Event Start",
        annotation_font=dict(size=10, color="green"),
    )
    fig.add_vline(
        x=event_end,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="Event End",
        annotation_font=dict(size=10, color="red"),
    )

    fig.add_annotation(
        text=badge_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=12, family=PLOT_STYLE.font_family, color="white"),
        bgcolor=badge_color,
        bordercolor=badge_color,
        borderwidth=2,
        borderpad=8,
        xanchor="left",
        yanchor="top",
    )
    return fig


def plot_multi_channel_time_series(
    trial_df: pl.DataFrame, channels: list[str]
) -> go.Figure:
    if trial_df.is_empty() or not channels:
        return go.Figure().update_layout(title_text="No data to plot")

    time_data = trial_df["time"].to_numpy()
    chunk_margin = trial_df["chunk_margin"][0]

    stim_state = trial_df["stim"][0] if "stim" in trial_df.columns else "off"

    event_start = time_data.min() + chunk_margin
    onset_time = event_start

    original_duration = trial_df["margined_duration"][0] - 2 * chunk_margin
    event_end = event_start + original_duration

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Amplitude (µV)",
        title="",
    )

    if stim_state == "on":
        bg_color = f"rgba(255, 0, 53, 0.08)"
        badge_color = PALETTE.strawberry_red
        badge_text = "DBS ON"
    else:
        bg_color = f"rgba(89, 84, 108, 0.08)"
        badge_color = PALETTE.vintage_grape
        badge_text = "DBS OFF"

    fig.add_vrect(
        x0=time_data.min(),
        x1=time_data.max(),
        fillcolor=bg_color,
        layer="below",
        line_width=0,
    )

    fig.add_vrect(
        x0=event_start,
        x1=event_end,
        fillcolor="rgba(0, 150, 0, 0.08)",
        layer="below",
        line_width=0,
    )

    colors = px.colors.qualitative.Plotly
    all_y_values = []
    for idx, channel in enumerate(channels):
        channel_data = trial_df[channel].to_numpy()
        all_y_values.extend(channel_data)

        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=channel_data,
                mode="lines",
                name=channel,
                line=dict(color=color, width=1.5),
                showlegend=True,
            )
        )

    fig.add_vline(
        x=event_start,
        line_dash="dash",
        line_color="green",
        line_width=2,
        annotation_text="Event Start",
        annotation_font=dict(size=10, color="green"),
    )
    fig.add_vline(
        x=event_end,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="Event End",
        annotation_font=dict(size=10, color="red"),
    )

    fig.add_annotation(
        text=badge_text,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        font=dict(size=12, family=PLOT_STYLE.font_family, color="white"),
        bgcolor=badge_color,
        bordercolor=badge_color,
        borderwidth=2,
        borderpad=8,
        xanchor="left",
        yanchor="top",
    )
    fig.update_layout(showlegend=True)

    return fig


def plot_coordinates_time_series(
    trial_df: pl.DataFrame, time_col: str = "motion_time"
) -> go.Figure:
    if trial_df.is_empty() or trial_df["x"].is_null().all():
        return go.Figure().update_layout(title_text="No coordinate data available")

    time_data = trial_df[time_col].to_numpy()
    x_data = trial_df["x"].to_numpy()
    y_data = trial_df["y"].to_numpy()

    onset_time = time_data.min()

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Position",
        title="",
    )

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=x_data,
            mode="lines",
            name="X-coord",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=y_data,
            mode="lines",
            name="Y-coord",
            line=dict(width=2),
        )
    )
    return fig


def plot_speed_time_series(
    trial_df: pl.DataFrame, time_col: str = "motion_time"
) -> go.Figure:
    if trial_df.is_empty() or trial_df["tracing_speed"].is_null().all():
        return go.Figure().update_layout(title_text="No speed data available")

    time_data = trial_df[time_col].to_numpy()
    speed_data = trial_df["tracing_speed"].to_numpy()

    onset_time = time_data.min()

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Speed (pixels/s)",
        title="",
    )

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=speed_data,
            mode="lines",
            name="Speed",
            line=dict(width=2),
            showlegend=False,
        )
    )

    return fig


def plot_acceleration_time_series(
    trial_df: pl.DataFrame, time_col: str = "motion_time"
) -> go.Figure:
    if trial_df.is_empty() or trial_df["tracing_acceleration"].is_null().all():
        return go.Figure().update_layout(title_text="No acceleration data available")

    time_data = trial_df[time_col].to_numpy()
    accel_data = trial_df["tracing_acceleration"].to_numpy()
    
    if "tracing_acceleration_magnitude" in trial_df.columns:
        mag_data = trial_df["tracing_acceleration_magnitude"].to_numpy()
    else:
        mag_data = np.abs(accel_data)

    onset_time = time_data.min()

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Acceleration (pixels/$s^2$)",
        title="",
    )

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=accel_data,
            mode="markers+lines",
            name="Acceleration",
            marker=dict(
                color=mag_data,
                colorscale="Viridis",
                size=4,
                showscale=True,
                colorbar=dict(title="Magnitude"),
            ),
            line=dict(width=1.5, color="rgba(100,100,100,0.3)"),
            showlegend=False,
        )
    )

    return fig


def plot_jerk_time_series(
    trial_df: pl.DataFrame, time_col: str = "motion_time"
) -> go.Figure:
    if trial_df.is_empty() or trial_df["tracing_jerk"].is_null().all():
        return go.Figure().update_layout(title_text="No jerk data available")

    time_data = trial_df[time_col].to_numpy()
    jerk_data = trial_df["tracing_jerk"].to_numpy()
    
    if "tracing_jerk_magnitude" in trial_df.columns:
        mag_data = trial_df["tracing_jerk_magnitude"].to_numpy()
    else:
        mag_data = np.abs(jerk_data)

    onset_time = time_data.min()

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Jerk (pixels/$s^3$)",
        title="",
    )

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=jerk_data,
            mode="markers+lines",
            name="Jerk",
            marker=dict(
                color=mag_data,
                colorscale="Plasma",
                size=4,
                showscale=True,
                colorbar=dict(title="Magnitude"),
            ),
            line=dict(width=1.5, color="rgba(100,100,100,0.3)"),
            showlegend=False,
        )
    )

    return fig



def plot_2d_trajectory(trial_df: pl.DataFrame) -> go.Figure:
    if trial_df.is_empty() or trial_df["x"].is_null().all():
        return go.Figure().update_layout(title_text="No coordinate data available")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trial_df["x"],
            y=trial_df["y"],
            mode="lines+markers",
            name="Path",
            line=dict(width=2),
        )
    )

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

    fig.update_layout(
        title=dict(
            text="",
            x=0.5,
            xanchor="center",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        xaxis=dict(
            title=dict(
                text="X Coordinate",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title=dict(
                text="Y Coordinate",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
        ),
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, size=PLOT_STYLE.tick_label_size),
        margin=dict(l=60, r=30, t=100, b=60),
    )

    return fig


def plot_cross_trial_speed(
    block_data: pl.DataFrame,
    speed_type: str = "combined",
    time_col: str = "motion_time",
) -> go.Figure:

    if block_data.is_empty():
        return go.Figure().update_layout(title_text="No data available")

    trials = sorted(block_data["trial"].unique().to_list())

    if not trials:
        return go.Figure().update_layout(title_text="No trials found")

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    if speed_type == "x":
        speed_col = "tracing_speed_x"
        y_label = "X Speed (pixels/s)"
    elif speed_type == "y":
        speed_col = "tracing_speed_y"
        y_label = "Y Speed (pixels/s)"
    else:
        speed_col = "tracing_speed"
        y_label = "Speed (pixels/s)"

    legend_prefix = "Trial"

    for idx, trial in enumerate(trials):
        trial_data = block_data.filter(pl.col("trial") == trial)

        if trial_data.is_empty():
            continue

        stim_state = trial_data["stim"][0] if "stim" in trial_data.columns else "off"

        if time_col not in trial_data.columns or speed_col not in trial_data.columns:
            continue

        try:
            time_data_raw = trial_data[time_col][0]
            speed_data_raw = trial_data[speed_col][0]

            if time_data_raw is None or speed_data_raw is None:
                continue

            if isinstance(time_data_raw, (list, pl.Series)):
                time_data = np.array(time_data_raw)
            elif isinstance(time_data_raw, np.ndarray):
                time_data = time_data_raw
            else:
                time_data = np.array([time_data_raw])

            if isinstance(speed_data_raw, (list, pl.Series)):
                speed_data = np.array(speed_data_raw)
            elif isinstance(speed_data_raw, np.ndarray):
                speed_data = speed_data_raw
            else:
                speed_data = np.array([speed_data_raw])

            if len(time_data) == 0 or len(speed_data) == 0:
                continue

            min_len = min(len(time_data), len(speed_data))
            time_data = time_data[:min_len]
            speed_data = speed_data[:min_len]

            if np.all(np.isnan(speed_data)):
                continue

        except Exception:
            continue

        time_relative = time_data - time_data.min()

        color = colors[idx % len(colors)]

        stim_label = "ON" if stim_state == "on" else "OFF"
        legend_label = f"{legend_prefix} {trial} (DBS {stim_label})"

        fig.add_trace(
            go.Scatter(
                x=time_relative,
                y=speed_data,
                mode="lines",
                name=legend_label,
                line=dict(color=color, width=1.5),
                showlegend=True,
            )
        )

    fig.update_layout(
        title=dict(
            text="",
            x=0.5,
            xanchor="center",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        xaxis=dict(
            title=dict(
                text="Time (s)",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, size=PLOT_STYLE.tick_label_size),
        legend=dict(
            font=dict(size=10, family=PLOT_STYLE.font_family),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        showlegend=True,
        margin=dict(l=60, r=30, t=100, b=60),
        hovermode="x unified",
    )

    return fig


def plot_session_average_speed(
    speed_type: str = "combined",
    time_col: str = "motion_time",
) -> go.Figure:
    from utils.data_loader import load_participant_session_data

    participant_id = st.session_state.get("participant_id")
    session = st.session_state.get("session")

    if not participant_id or not session:
        return go.Figure().update_layout(title_text="Session information not available")

    session_data = load_participant_session_data(participant_id, session)

    if speed_type == "x":
        speed_col = "tracing_speed_x"
        y_label = "X Speed (pixels/s)"
    elif speed_type == "y":
        speed_col = "tracing_speed_y"
        y_label = "Y Speed (pixels/s)"
    else:
        speed_col = "tracing_speed"
        y_label = "Speed (pixels/s)"

    dbs_on_data = session_data.filter(pl.col("stim") == "on")
    dbs_off_data = session_data.filter(pl.col("stim") == "off")

    fig = go.Figure()

    speeds_on, times_on = _extract_speed_data(dbs_on_data, time_col, speed_col)
    if speeds_on:
        mean_speed_on, std_speed_on, time_axis = _compute_block_average(
            speeds_on, times_on
        )
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=mean_speed_on,
                mode="lines",
                name=f"DBS ON (mean, n={len(speeds_on)} trials)",
                line=dict(color="#2ca02c", width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_axis, time_axis[::-1]]),
                y=np.concatenate(
                    [mean_speed_on + std_speed_on, (mean_speed_on - std_speed_on)[::-1]]
                ),
                fill="toself",
                fillcolor="rgba(44, 160, 44, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=True,
                name="DBS ON ± 1σ",
            )
        )

    if not dbs_off_data.is_empty():
        speeds_off, times_off = _extract_speed_data(dbs_off_data, time_col, speed_col)
        if speeds_off:
            mean_speed_off, std_speed_off, time_axis = _compute_block_average(
                speeds_off, times_off
            )

            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=mean_speed_off,
                    mode="lines",
                    name=f"DBS OFF (mean, n={len(speeds_off)} trials)",
                    line=dict(color="#d62728", width=3),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([time_axis, time_axis[::-1]]),
                    y=np.concatenate(
                        [
                            mean_speed_off + std_speed_off,
                            (mean_speed_off - std_speed_off)[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor="rgba(214, 39, 40, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=True,
                    name="DBS OFF ± 1σ",
                )
            )

    fig.update_layout(
        title=dict(
            text="",
            x=0.5,
            xanchor="center",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        xaxis=dict(
            title=dict(
                text="Time (s)",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
        ),
        template="plotly_white",
        font=dict(family=PLOT_STYLE.font_family, size=PLOT_STYLE.tick_label_size),
        legend=dict(
            font=dict(size=10, family=PLOT_STYLE.font_family),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        showlegend=True,
        margin=dict(l=60, r=30, t=100, b=60),
        hovermode="x unified",
    )

    return fig


def _extract_speed_data(data: pl.DataFrame, time_col: str, speed_col: str):
    speeds = []
    times = []

    for row in data.iter_rows(named=True):
        try:
            time_data_raw = row[time_col]
            speed_data_raw = row[speed_col]

            if time_data_raw is None or speed_data_raw is None:
                continue

            # Convert to numpy arrays
            if isinstance(time_data_raw, (list, pl.Series)):
                time_data = np.array(time_data_raw)
            elif isinstance(time_data_raw, np.ndarray):
                time_data = time_data_raw
            else:
                time_data = np.array([time_data_raw])

            if isinstance(speed_data_raw, (list, pl.Series)):
                speed_data = np.array(speed_data_raw)
            elif isinstance(speed_data_raw, np.ndarray):
                speed_data = speed_data_raw
            else:
                speed_data = np.array([speed_data_raw])

            if len(time_data) == 0 or len(speed_data) == 0:
                continue

            min_len = min(len(time_data), len(speed_data))
            time_data = time_data[:min_len]
            speed_data = speed_data[:min_len]
            if np.all(np.isnan(speed_data)):
                continue

            time_relative = time_data - time_data.min()

            speeds.append(speed_data)
            times.append(time_relative)

        except Exception:
            continue

    return speeds, times


def _compute_block_average(speeds, times):
    if not speeds or not times:
        return np.array([]), np.array([]), np.array([])

    max_time = max(t.max() for t in times)
    min_time = 0.0

    n_points = 100  # TODO: why this is interpolated?
    time_axis = np.linspace(min_time, max_time, n_points)

    interpolated_speeds = []
    for speed, time in zip(speeds, times):
        valid_mask = ~np.isnan(speed)
        if np.sum(valid_mask) < 2:
            continue

        time_valid = time[valid_mask]
        speed_valid = speed[valid_mask]

        interp_speed = np.interp(
            time_axis, time_valid, speed_valid, left=np.nan, right=np.nan
        )
        interpolated_speeds.append(interp_speed)

    if not interpolated_speeds:
        return np.array([]), np.array([]), np.array([])

    speed_matrix = np.vstack(interpolated_speeds)

    mean_speed = np.nanmean(speed_matrix, axis=0)
    std_speed = np.nanstd(speed_matrix, axis=0)

    return mean_speed, std_speed, time_axis
