import numpy as np
import polars as pl
import plotly.graph_objects as go
from dashboard.backbone import create_base_time_series_figure, PLOT_STYLE


def plot_acceleration_time_series(
    trial_df: pl.DataFrame, time_col: str = "motion_time"
) -> go.Figure:
    if trial_df.is_empty() or trial_df["tracing_acceleration"].is_null().all():
        return go.Figure().update_layout(title_text="No acceleration data available")

    time_data = trial_df[time_col].to_numpy()
    accel_data = trial_df["tracing_acceleration"].to_numpy()
    mag_data = trial_df["tracing_acceleration_magnitude"].to_numpy()

    onset_time = time_data.min()

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Acceleration (pixels/s$^2$)",
        title="",
    )

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=accel_data,
            mode="lines",
            name="Acceleration",
            line=dict(color=mag_data, colorscale="Viridis", width=2),
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
    mag_data = trial_df["tracing_jerk_magnitude"].to_numpy()

    onset_time = time_data.min()

    fig = create_base_time_series_figure(
        time_abs=time_data,
        onset_time=onset_time,
        y_label="Jerk (pixels/s³)",
        title="",
    )

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=jerk_data,
            mode="lines",
            name="Jerk",
            line=dict(color=mag_data, colorscale="Plasma", width=2),
            showlegend=False,
        )
    )

    return fig
