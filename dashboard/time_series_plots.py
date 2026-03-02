import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from utils.data_loader import load_participant_session_data
from dashboard.backbone import (
    create_base_time_series_figure,
    PLOT_STYLE,
    PLOT_COLOR,
    PALETTE,
    PARTICIPANT_COLORS,
    LINE_STYLE,
    add_margin_visualization,
)

SAMPLING_FREQ = 60


def plot_neural_time_series(
    data_on: np.ndarray | None,
    data_off: np.ndarray | None,
    time_axis: np.ndarray,
    channel: str,
    participant_id: str = None,
    description: str = "",
    show_sem: bool = True,
    sem_on: np.ndarray | None = None,
    sem_off: np.ndarray | None = None,
    chunk_margin: float = 0.0,
) -> go.Figure:
    if data_on is None and data_off is None:
        return go.Figure()

    onset_time = time_axis.min() if len(time_axis) > 0 else 0.0

    fig = create_base_time_series_figure(
        time_abs=time_axis,
        onset_time=onset_time,
        y_label="Amplitude (µV)",
        title="",
    )

    if chunk_margin > 0:
        add_margin_visualization(fig, time_axis, chunk_margin)

    if participant_id and participant_id in PARTICIPANT_COLORS:
        colors = PARTICIPANT_COLORS[participant_id]
        color_on = colors.dbs_on
        color_off = colors.dbs_off
    else:
        color_on = PLOT_COLOR.stim_on
        color_off = PLOT_COLOR.stim_off

    if data_on is not None:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=data_on,
                mode="lines",
                name="DBS ON",
                line=dict(color=color_on, width=PLOT_STYLE.line_width_normal),
            )
        )
        if show_sem and sem_on is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([time_axis, time_axis[::-1]]),
                    y=np.concatenate([data_on + sem_on, (data_on - sem_on)[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({int(color_on[1:3], 16)}, {int(color_on[3:5], 16)}, {int(color_on[5:7], 16)}, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    name="DBS ON ± SEM",
                )
            )

    if data_off is not None:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=data_off,
                mode="lines",
                name="DBS OFF",
                line=dict(
                    color=color_off,
                    width=PLOT_STYLE.line_width_normal,
                    dash=LINE_STYLE.secondary,
                ),
            )
        )
        if show_sem and sem_off is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([time_axis, time_axis[::-1]]),
                    y=np.concatenate([data_off + sem_off, (data_off - sem_off)[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba({int(color_off[1:3], 16)}, {int(color_off[3:5], 16)}, {int(color_off[5:7], 16)}, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    name="DBS OFF ± SEM",
                )
            )

    if description:
        fig.add_annotation(
            text=description,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            showarrow=False,
            font=dict(size=11, family=PLOT_STYLE.font_family),
            xanchor="center",
            yanchor="top",
        )
        fig.update_layout(margin=dict(b=100))

    return fig


def plot_population_time_series(
    population_data: dict,  # {p_id: {'on': data, 'off': data}}
    time_axis: np.ndarray,
    ylabel: str = "Amplitude (µV)",
) -> go.Figure:
    """
    Plots time series averaged for each participant on a single plot.
    """
    onset_time = time_axis.min() if len(time_axis) > 0 else 0.0

    fig = create_base_time_series_figure(
        time_abs=time_axis,
        onset_time=onset_time,
        y_label=ylabel,
        title="",
    )

    for p_id, data in population_data.items():
        if p_id in PARTICIPANT_COLORS:
            colors = PARTICIPANT_COLORS[p_id]
        elif f"PDI_{p_id}" in PARTICIPANT_COLORS:
            colors = PARTICIPANT_COLORS[f"PDI_{p_id}"]
        elif f"PDI{p_id}" in PARTICIPANT_COLORS:
            colors = PARTICIPANT_COLORS[f"PDI{p_id}"]
        else:
            # Default fallback
            c_on = PLOT_COLOR.stim_on
            c_off = PLOT_COLOR.stim_off
            colors = None

        if colors:
            c_on = colors.dbs_on
            c_off = colors.dbs_off

        # Plot ON
        if "on" in data and data["on"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=data["on"],
                    mode="lines",
                    name=f"{p_id} ON",
                    line=dict(color=c_on, width=PLOT_STYLE.line_width_normal),
                )
            )

        # Plot OFF
        if "off" in data and data["off"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=data["off"],
                    mode="lines",
                    name=f"{p_id} OFF",
                    line=dict(
                        color=c_off,
                        width=PLOT_STYLE.line_width_normal,
                        dash=LINE_STYLE.secondary,
                    ),
                )
            )

    return fig


def plot_session_comparison_time_series(
    session_data: dict,  # {session_id: {'on': data, 'off': data}}
    time_axis: np.ndarray,
    participant_id: str,
    ylabel: str = "Amplitude (µV)",
) -> go.Figure:
    """
    Plots time series for all sessions of a single participant.
    """
    onset_time = time_axis.min() if len(time_axis) > 0 else 0.0

    fig = create_base_time_series_figure(
        time_abs=time_axis,
        onset_time=onset_time,
        y_label=ylabel,
        title="",
    )

    session_colors = px.colors.qualitative.Plotly

    # Sort sessions naturally
    sorted_sessions = sorted(
        session_data.keys(), key=lambda x: int(x) if x.isdigit() else x
    )

    for idx, s_id in enumerate(sorted_sessions):
        data = session_data[s_id]
        session_color = session_colors[idx % len(session_colors)]

        if "on" in data and data["on"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=data["on"],
                    mode="lines",
                    name=f"S{s_id} ON",
                    line=dict(
                        color=session_color, width=PLOT_STYLE.line_width_normal * 1.5
                    ),
                    opacity=0.8,
                )
            )

        if "off" in data and data["off"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=data["off"],
                    mode="lines",
                    name=f"S{s_id} OFF",
                    line=dict(
                        color=session_color,
                        width=PLOT_STYLE.line_width_normal * 1.5,
                        dash=LINE_STYLE.secondary,
                    ),
                    opacity=0.8,
                )
            )

    return fig


def compute_block_average(block_data: pl.DataFrame, channel: str, n_points: int = 100):
    dbs_on = block_data.filter(pl.col("stim") == "on")
    dbs_off = block_data.filter(pl.col("stim") == "off")

    def _aggregate_trials(df):
        if df.is_empty():
            return None, None
        signals = []
        for row in df.iter_rows(named=True):
            if channel in row and row[channel] is not None:
                sig = np.array(row[channel])
                if len(sig) > 0:
                    signals.append(sig)
        if not signals:
            return None, None
        max_len = max(len(s) for s in signals)
        time_axis = np.linspace(0, max_len / SAMPLING_FREQ, n_points)
        interpolated = []
        for sig in signals:
            t_orig = np.linspace(0, len(sig) / SAMPLING_FREQ, len(sig))
            interp = np.interp(time_axis, t_orig, sig)
            interpolated.append(interp)
        stacked = np.vstack(interpolated)
        mean = np.nanmean(stacked, axis=0)
        sem = np.nanstd(stacked, axis=0) / np.sqrt(len(interpolated))
        return mean, sem, time_axis

    result_on = _aggregate_trials(dbs_on)
    result_off = _aggregate_trials(dbs_off)

    mean_on = result_on[0] if result_on[0] is not None else None
    sem_on = result_on[1] if result_on[0] is not None else None
    mean_off = result_off[0] if result_off[0] is not None else None
    sem_off = result_off[1] if result_off[0] is not None else None
    time_axis = (
        result_on[2]
        if result_on[0] is not None
        else (result_off[2] if result_off[0] is not None else np.array([]))
    )

    return mean_on, sem_on, mean_off, sem_off, time_axis


def compute_session_average(
    session_data: pl.DataFrame, channel: str, n_points: int = 100
):
    return compute_block_average(session_data, channel, n_points)


def compute_participant_average_per_session(
    participant_data: pl.DataFrame, channel: str, n_points: int = 100
):
    sessions = participant_data["session"].unique().to_list()
    results = {}
    for session in sessions:
        session_df = participant_data.filter(pl.col("session") == session)
        mean_on, sem_on, mean_off, sem_off, time_axis = compute_block_average(
            session_df, channel, n_points
        )
        results[session] = {
            "mean_on": mean_on,
            "sem_on": sem_on,
            "mean_off": mean_off,
            "sem_off": sem_off,
            "time_axis": time_axis,
        }
    return results


def plot_channel_time_series(trial_df: pl.DataFrame, channel: str) -> go.Figure:
    if trial_df.is_empty():
        return go.Figure().update_layout(title_text=f"No data for {channel}")

    time_data = trial_df["time"].to_numpy()
    channel_data = trial_df[channel].to_numpy()
    chunk_margin = trial_df["chunk_margin"][0]

    stim_state = trial_df["stim"][0] if "stim" in trial_df.columns else "off"

    event_start = time_data.min() + chunk_margin
    onset_time = event_start

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Amplitude (µV)",
        title="",
    )

    if chunk_margin > 0:
        add_margin_visualization(fig, time_data, chunk_margin)

    if stim_state == "on":
        line_color = PLOT_COLOR.stim_on
        line_name = "DBS ON"
    else:
        line_color = PLOT_COLOR.stim_off
        line_name = "DBS OFF"

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=channel_data,
            mode="lines",
            name=line_name,
            line=dict(color=line_color, width=PLOT_STYLE.line_width_normal),
        )
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

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Amplitude (µV)",
        title="",
    )

    if chunk_margin > 0:
        add_margin_visualization(fig, time_data, chunk_margin)

    colors = px.colors.qualitative.Plotly
    for idx, channel in enumerate(channels):
        channel_data = trial_df[channel].to_numpy()
        color = colors[idx % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=channel_data,
                mode="lines",
                name=channel,
                line=dict(color=color, width=PLOT_STYLE.line_width_normal),
                showlegend=True,
            )
        )

    fig.update_layout(showlegend=True)

    return fig


def plot_coordinates_time_series(
    trial_df: pl.DataFrame, time_col: str = "motion_time"
) -> go.Figure:
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
    trial_df: pl.DataFrame,
    col_name: str = "tracing_velocity_magnitude",
    time_col: str = "motion_time",
) -> go.Figure:
    time_data = trial_df[time_col].to_numpy()
    speed_data = trial_df[col_name].to_numpy()

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
            name=col_name,
            line=dict(width=2, color=PALETTE.twilight_indigo),
            showlegend=False,
        )
    )

    return fig


def plot_acceleration_time_series(
    trial_df: pl.DataFrame,
    col_name: str = "tracing_acceleration_magnitude",
    time_col: str = "motion_time",
) -> go.Figure:
    time_data = trial_df[time_col].to_numpy()
    accel_data = trial_df[col_name].to_numpy()

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
            mode="lines",
            name="Acceleration",
            line=dict(width=2, color=PALETTE.strawberry_red),
            showlegend=False,
        )
    )

    return fig


def plot_jerk_time_series(
    trial_df: pl.DataFrame,
    col_name: str = "tracing_jerk_magnitude",
    time_col: str = "motion_time",
) -> go.Figure:
    time_data = trial_df[time_col].to_numpy()
    jerk_data = trial_df[col_name].to_numpy()

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
            mode="lines",
            name="Jerk",
            line=dict(width=2, color=PALETTE.twilight_indigo),
            showlegend=False,
        )
    )

    return fig


def plot_2d_trajectory(
    trial_df: pl.DataFrame, x_col: str = "x", y_col: str = "y", title: str = ""
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trial_df[x_col],
            y=trial_df[y_col],
            mode="lines+markers",
            name="Path",
            line=dict(width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=trial_df.head(1)[x_col],
            y=trial_df.head(1)[y_col],
            mode="markers",
            marker=dict(color="green", size=10),
            name="Start",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trial_df.tail(1)[x_col],
            y=trial_df.tail(1)[y_col],
            mode="markers",
            marker=dict(color="red", size=10),
            name="End",
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        xaxis=dict(
            title=dict(
                text=f"{x_col} value",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title=dict(
                text=f"{y_col} value",
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


def plot_component_time_series(
    trial_df: pl.DataFrame,
    components: list[str],
    time_col: str = "motion_time",
    title: str = "",
    y_label: str = "Value",
) -> go.Figure:
    if trial_df.is_empty() or not components:
        return go.Figure().update_layout(title_text="No data to plot")

    time_data = trial_df[time_col].to_numpy()
    onset_time = time_data.min()

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label=y_label,
        title=title,
    )

    for comp in components:
        if comp not in trial_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=trial_df[comp].to_numpy(),
                mode="lines",
                name=comp,
                line=dict(width=2),
            )
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
        speed_col = "tracing_velocity_x"
        y_label = "X Speed (pixels/s)"
    elif speed_type == "y":
        speed_col = "tracing_velocity_y"
        y_label = "Y Speed (pixels/s)"
    else:
        speed_col = "tracing_velocity_magnitude"
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


def plot_session_average_behavior(
    behavioral_col: str,
    y_label: str,
    time_col: str = "motion_time",
) -> go.Figure:

    participant_id = st.session_state.get("participant_id")
    session = st.session_state.get("session")
    dataset = st.session_state.get("selected_dataset")

    if not participant_id or not session:
        return go.Figure().update_layout(title_text="Session information not available")

    session_data = load_participant_session_data(
        participant_id, session, dataset=dataset
    )

    dbs_on_data = session_data.filter(pl.col("stim") == "on")
    dbs_off_data = session_data.filter(pl.col("stim") == "off")

    fig = go.Figure()

    speeds_on, times_on = _extract_speed_data(dbs_on_data, time_col, behavioral_col)
    if speeds_on:
        mean_speed_on, std_speed_on, time_axis = _compute_block_average(
            speeds_on, times_on
        )
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=mean_speed_on,
                mode="lines",
                name=f"DBS ON (n={len(speeds_on)})",
                line=dict(color=PLOT_COLOR.stim_on, width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([time_axis, time_axis[::-1]]),
                y=np.concatenate(
                    [mean_speed_on + std_speed_on, (mean_speed_on - std_speed_on)[::-1]]
                ),
                fill="toself",
                fillcolor="rgba(255, 0, 53, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=True,
                name="DBS ON $\pm 1\sigma$",
            )
        )

    if not dbs_off_data.is_empty():
        speeds_off, times_off = _extract_speed_data(dbs_off_data, time_col, behavioral_col)
        if speeds_off:
            mean_speed_off, std_speed_off, time_axis = _compute_block_average(
                speeds_off, times_off
            )

            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=mean_speed_off,
                    mode="lines",
                    name=f"DBS OFF (n={len(speeds_off)})",
                    line=dict(color=PLOT_COLOR.stim_off, width=3, dash="dot"),
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
                    fillcolor="rgba(89, 84, 108, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=True,
                    name="DBS OFF $\pm 1\sigma$",
                )
            )

    fig.update_layout(
        title="",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=40, t=10, b=40),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    fig.update_xaxes(
        title_text="Time (s)",
        showgrid=True,
        gridcolor="rgba(150, 150, 150, 0.4)",
        showline=False,
    )
    
    fig.update_yaxes(
        title_text=y_label,
        showgrid=True,
        gridcolor="rgba(150, 150, 150, 0.4)",
        showline=False,
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


def compute_cross_correlation_lag(
    signal1: np.ndarray, signal2: np.ndarray, max_lag_ms: float = 500.0
) -> tuple:
    s1 = (signal1 - np.nanmean(signal1)) / (np.nanstd(signal1) + 1e-10)
    s2 = (signal2 - np.nanmean(signal2)) / (np.nanstd(signal2) + 1e-10)

    s1 = np.nan_to_num(s1, nan=0.0)
    s2 = np.nan_to_num(s2, nan=0.0)

    correlation = np.correlate(s1, s2, mode="full")
    correlation = correlation / len(s1)

    lags = np.arange(-len(s2) + 1, len(s1))

    max_lag_samples = int(max_lag_ms / 1000.0 * SAMPLING_FREQ)
    lag_mask = np.abs(lags) <= max_lag_samples

    lags_restricted = lags[lag_mask]
    correlation_restricted = correlation[lag_mask]

    max_idx = np.argmax(np.abs(correlation_restricted))
    lag_samples = lags_restricted[max_idx]
    max_corr = correlation_restricted[max_idx]

    return lag_samples, max_corr, lags_restricted, correlation_restricted


def plot_signal_alignment(
    time_neural: np.ndarray,
    neural_data: np.ndarray,
    time_beh: np.ndarray,
    beh_data: np.ndarray,
    neural_channel: str,
    behavioral_var: str,
    chunk_margin: float,
    lags: np.ndarray,
    correlation: np.ndarray,
    lag_seconds: float,
) -> go.Figure:

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Signal Overlay: {neural_channel} vs {behavioral_var}",
            "Cross-Correlation (Lag Detection)",
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4],
    )

    neural_norm = (neural_data - np.nanmean(neural_data)) / (
        np.nanstd(neural_data) + 1e-10
    )
    beh_norm = (beh_data - np.nanmean(beh_data)) / (np.nanstd(beh_data) + 1e-10)

    fig.add_trace(
        go.Scatter(
            x=time_neural,
            y=neural_norm,
            mode="lines",
            name=f"{neural_channel} (normalized)",
            line=dict(color=PALETTE.twilight_indigo, width=1.5),
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=time_beh,
            y=beh_norm,
            mode="lines",
            name=f"{behavioral_var} (normalized)",
            line=dict(color=PALETTE.strawberry_red, width=1.5),
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    if chunk_margin > 0:
        margin_start = time_neural[0]
        margin_end_left = margin_start + chunk_margin
        margin_start_right = time_neural[-1] - chunk_margin

        fig.add_vrect(
            x0=margin_start,
            x1=margin_end_left,
            fillcolor="rgba(89, 84, 108, 0.1)",
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_vline(
            x=margin_end_left,
            line_dash="dash",
            line_color="#59546c",
            line_width=2,
            annotation_text="Margin End",
            annotation_position="top right",
            row=1,
            col=1,
        )

        fig.add_vrect(
            x0=margin_start_right,
            x1=time_neural[-1],
            fillcolor="rgba(89, 84, 108, 0.1)",
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_vline(
            x=margin_start_right,
            line_dash="dash",
            line_color="#59546c",
            line_width=2,
            annotation_text="Margin Start",
            annotation_position="top left",
            row=1,
            col=1,
        )

    lag_seconds_arr = lags / SAMPLING_FREQ

    fig.add_trace(
        go.Scatter(
            x=lag_seconds_arr,
            y=correlation,
            mode="lines",
            name="Cross-correlation",
            line=dict(color=PALETTE.vintage_grape, width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_vline(
        x=lag_seconds,
        line_dash="solid",
        line_color=PALETTE.strawberry_red,
        line_width=3,
        row=2,
        col=1,
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        row=2,
        col=1,
    )

    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Lag (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)

    return fig


def plot_raw_alignment(
    time_master: np.ndarray,
    neural_data: np.ndarray | None,
    time_raw: np.ndarray,
    beh_raw: np.ndarray,
    time_interp: np.ndarray,
    beh_interp: np.ndarray,
    neural_channel: str,
    behavioral_var: str,
    chunk_margin: float,
) -> go.Figure:

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if neural_data is not None and neural_channel:
        neural_norm = (neural_data - np.nanmean(neural_data)) / (
            np.nanstd(neural_data) + 1e-10
        )
        fig.add_trace(
            go.Scatter(
                x=time_master,
                y=neural_norm,
                mode="lines",
                name=f"{neural_channel} (Normalized)",
                line=dict(color=PALETTE.twilight_indigo, width=1.5, dash="dot"),
                opacity=0.5,
            ),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=time_interp,
            y=beh_interp,
            mode="lines+markers",
            name=f"{behavioral_var} (Interpolated Grid)",
            line=dict(color=PALETTE.strawberry_red, width=2),
            marker=dict(size=6, symbol="x"),
            opacity=0.9,
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=time_raw,
            y=beh_raw,
            mode="markers",
            name=f"{behavioral_var} (Raw unaligned)",
            marker=dict(color=PALETTE.vintage_grape, size=8, symbol="circle"),
            opacity=0.7,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=60, t=10, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    fig.update_xaxes(title_text="Time (s)")
    if neural_data is not None and neural_channel:
        fig.update_yaxes(title_text=f"{neural_channel} Normalized", secondary_y=False)
    fig.update_yaxes(title_text=f"{behavioral_var} Value", secondary_y=True)

    if chunk_margin > 0:
        add_margin_visualization(fig, time_master, chunk_margin)

    return fig


def plot_residual_violin(residuals: list, trials: list) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, res in enumerate(residuals):
        fig.add_trace(
            go.Violin(
                y=res,
                name=f"Trial {trials[i]}",
                box_visible=True,
                line_color=colors[i % len(colors)],
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                opacity=0.6,
                showlegend=False,
            )
        )
    fig.update_layout(
        title="",
        yaxis_title="Interpolation Residual Value",
        xaxis_title="Trial",
        template="plotly_white",
        margin=dict(l=60, r=30, t=10, b=40),
    )
    return fig


def plot_raw_alignment(
    time_master: np.ndarray,
    neural_data: np.ndarray | None,
    time_raw: np.ndarray,
    beh_raw: np.ndarray,
    time_interp: np.ndarray,
    beh_interp: np.ndarray,
    neural_channel: str,
    behavioral_var: str,
    chunk_margin: float,
) -> go.Figure:

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if neural_data is not None and neural_channel:
        neural_norm = (neural_data - np.nanmean(neural_data)) / (
            np.nanstd(neural_data) + 1e-10
        )
        fig.add_trace(
            go.Scatter(
                x=time_master,
                y=neural_norm,
                mode="lines",
                name=f"{neural_channel} (Normalized)",
                line=dict(color=PALETTE.twilight_indigo, width=1.5, dash="dot"),
                opacity=0.5,
            ),
            secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(
            x=time_interp,
            y=beh_interp,
            mode="lines+markers",
            name=f"{behavioral_var} (Interpolated Grid)",
            line=dict(color=PALETTE.strawberry_red, width=2),
            marker=dict(size=6, symbol="x"),
            opacity=0.9,
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=time_raw,
            y=beh_raw,
            mode="markers",
            name=f"{behavioral_var} (Raw unaligned)",
            marker=dict(color=PALETTE.vintage_grape, size=8, symbol="circle"),
            opacity=0.7,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=60, t=10, b=10),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    fig.update_xaxes(title_text="Time (s)")
    if neural_data is not None and neural_channel:
        fig.update_yaxes(title_text=f"{neural_channel} Normalized", secondary_y=False)
    fig.update_yaxes(title_text=f"{behavioral_var} Value", secondary_y=True)

    if chunk_margin > 0:
        add_margin_visualization(fig, time_master, chunk_margin)

    return fig


def plot_residual_violin(residuals: list, trials: list) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, res in enumerate(residuals):
        fig.add_trace(
            go.Violin(
                y=res,
                name=f"Trial {trials[i]}",
                box_visible=True,
                line_color=colors[i % len(colors)],
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                opacity=0.6,
                showlegend=False,
            )
        )
    fig.update_layout(
        title="",
        yaxis_title="Interpolation Residual Value",
        xaxis_title="Trial",
        template="plotly_white",
        margin=dict(l=60, r=30, t=10, b=40),
    )
    return fig


def plot_micro_alignment(
    time_master: np.ndarray,
    neural_data: np.ndarray | None,
    time_raw: np.ndarray,
    beh_raw_dict: dict,
    time_interp: np.ndarray,
    beh_interp_dict: dict,
    neural_channel: str,
    window_start: float,
    window_end: float,
) -> go.Figure:

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    mask_master = (time_master >= window_start) & (time_master <= window_end)
    mask_raw = (time_raw >= window_start) & (time_raw <= window_end)
    mask_interp = (time_interp >= window_start) & (time_interp <= window_end)

    t_m = time_master[mask_master]
    t_i = time_interp[mask_interp]
    t_r = time_raw[mask_raw]

    if len(t_m) > 0:
        for t in t_m:
            fig.add_vline(
                x=t,
                y0=0.0,
                y1=0.7,
                line_dash="dot",
                line_color="rgba(56, 64, 95, 0.3)",
                line_width=1.5,
                layer="below",
            )
            
    if len(t_r) > 0:
        for t in t_r:
            fig.add_vline(
                x=t,
                y0=0.0,
                y1=0.7,
                line_dash="dot",
                line_color="rgba(255, 0, 53, 0.2)",
                line_width=1,
                layer="below",
            )

    if neural_data is not None and neural_channel:
        n_m = neural_data[mask_master]

        fig.add_trace(
            go.Scatter(
                x=t_m,
                y=n_m,
                mode="lines+markers",
                name=f"{neural_channel}",
                line=dict(color=PALETTE.twilight_indigo, width=1.5),
                marker=dict(size=4, symbol="circle"),
            ),
            secondary_y=False,
        )

    colors_raw = {"x": PALETTE.vintage_grape, "y": PALETTE.cool_steel}
    colors_interp = {"x": PALETTE.strawberry_red, "y": PALETTE.ink_black}

    for var in beh_raw_dict.keys():
        b_r = beh_raw_dict[var][mask_raw]
        b_i = beh_interp_dict[var][mask_interp]

        fig.add_trace(
            go.Scatter(
                x=t_i,
                y=b_i,
                mode="markers",
                name=f"Interp {var.upper()}",
                marker=dict(color=colors_interp.get(var, "red"), size=7, symbol="square"),
            ),
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(
                x=t_r,
                y=b_r,
                mode="markers+lines",
                name=f"Raw {var.upper()}",
                marker=dict(color=colors_raw.get(var, "blue"), size=6, symbol="circle"),
                line=dict(color=colors_raw.get(var, "blue"), width=1, dash="dot"),
            ),
            secondary_y=True,
        )

    if len(beh_raw_dict) > 0:
        min_vals = []
        max_vals = []
        for var in beh_raw_dict.keys():
            b_r = beh_raw_dict[var][mask_raw]
            if len(b_r) > 0:
                min_vals.append(np.nanmin(b_r))
                max_vals.append(np.nanmax(b_r))

        if min_vals:
            min_y = min(min_vals)
            rng_y = max(max_vals) - min_y
            if rng_y == 0:
                rng_y = 1

            rug_raw_y = np.full_like(t_r, min_y - rng_y * 0.75)
            rug_grid_y = np.full_like(t_m, min_y - rng_y * 0.95)

            fig.add_trace(
                go.Scatter(
                    x=t_r,
                    y=rug_raw_y,
                    mode="markers",
                    name="Rug: Raw Time",
                    marker=dict(
                        symbol="line-ns-open", size=12, color=PALETTE.strawberry_red, line=dict(width=2)
                    ),
                    showlegend=False,
                ),
                secondary_y=True,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_m,
                    y=rug_grid_y,
                    mode="markers",
                    name="Rug: Grid Time",
                    marker=dict(
                        symbol="line-ns-open", size=12, color=PALETTE.twilight_indigo, line=dict(width=2)
                    ),
                    showlegend=False,
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title="",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=60, t=10, b=40),
        legend=dict(
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            orientation="h",
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    if len(t_m) > 0:
        # Use exact 80Hz grid ticks (0.0125s / 12.5ms intervals)
        start_tick = np.floor(window_start / 0.0125) * 0.0125
        ticks = np.arange(start_tick, window_end + 0.0125, 0.0125)
        # Show labels on every second tick to prevent overlap text
        ticktexts = [f"{(t - t_m[0])*1000:.1f}ms<br>{t:.3f}s" if i % 2 == 0 else "" for i, t in enumerate(ticks)]
        fig.update_xaxes(
            title_text="Time (s)",
            tickvals=ticks,
            ticktext=ticktexts,
            showgrid=True,
            gridcolor="rgba(150, 150, 150, 0.4)",
        )
    else:
        fig.update_xaxes(
            title_text="Time (s)",
            showgrid=True, gridcolor="rgba(150, 150, 150, 0.4)"
        )

    if neural_data is not None and neural_channel:
        fig.update_yaxes(
            title_text=f"{neural_channel} (µV)", secondary_y=False,
            showgrid=True, gridcolor="rgba(150, 150, 150, 0.4)"
        )
    fig.update_yaxes(
        title_text="Behavioral Coordinates (pixels)", secondary_y=True,
        showgrid=True, gridcolor="rgba(150, 150, 150, 0.4)"
    )

    return fig
