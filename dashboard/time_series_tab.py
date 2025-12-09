import streamlit as st
import polars as pl
import numpy as np

from dashboard.time_series_plots import (
    plot_channel_time_series,
    plot_multi_channel_time_series,
    plot_coordinates_time_series,
    plot_speed_time_series,
    plot_2d_trajectory,
    plot_cross_trial_speed,
    plot_session_average_speed,
)
from dashboard.utils import get_channel_lists, get_trial_metadata
from dashboard.backbone import (
    format_title,
    format_trial_metadata,
    update_fig_title,
    PLOT_STYLE,
)


def prepare_motion_data(trial_data):
    motion_cols = []
    if "motion_time" in trial_data.columns:
        motion_cols.append("motion_time")
    if "x" in trial_data.columns:
        motion_cols.append("x")
    if "y" in trial_data.columns:
        motion_cols.append("y")
    if "tracing_speed" in trial_data.columns:
        motion_cols.append("tracing_speed")

    if not motion_cols:
        return None, []

    try:
        coords_data = trial_data.select(
            *motion_cols,
            "participant_id",
            "session",
            "block",
            "trial",
        ).explode(*motion_cols)
        return coords_data, motion_cols
    except (pl.exceptions.ShapeError, Exception):
        # TODO: Check why columns cannot be exploded
        # st.warning(
        #     "Motion data columns have mismatched lengths. Aligning to shortest column."
        # )

        lengths = {}
        for col in motion_cols:
            col_data = trial_data[col][0]
            if col_data is not None:
                lengths[col] = (
                    len(col_data) if isinstance(col_data, (list, pl.Series)) else 1
                )

        if not lengths:
            return None, motion_cols

        min_length = min(lengths.values())
        truncated_data = trial_data.select(
            "participant_id", "session", "block", "trial"
        )

        for col in motion_cols:
            col_data = trial_data[col][0]
            if col_data is not None:
                truncated = col_data[:min_length]
                truncated_data = truncated_data.with_columns(
                    pl.Series([truncated]).alias(col)
                )

        coords_data = truncated_data.explode(*motion_cols)
        return coords_data, motion_cols


def get_trial_duration(trial_data):
    duration = None
    if "margined_duration" in trial_data.columns:
        duration = trial_data["margined_duration"][0]
        if "chunk_margin" in trial_data.columns:
            duration -= 2 * trial_data["chunk_margin"][0]
    return duration


def render_neural_channels(trial_data, channels, channel_type, metadata_str):
    if not channels:
        st.info(f"No {channel_type} channels available.")
        return

    selected_channels = st.multiselect(
        f"Select {channel_type} Channel(s)",
        channels,
        key=f"{channel_type.lower()}_ts_{trial_data['trial'][0]}",
    )

    if selected_channels:
        cols_to_select = [
            "time",
            "chunk_margin",
            "margined_duration",
            "stim",
            "participant_id",
            "session",
            "block",
            "trial",
        ] + selected_channels

        df_exploded = trial_data.select(cols_to_select).explode(
            "time", *selected_channels
        )

        if len(selected_channels) > 1:
            fig = plot_multi_channel_time_series(df_exploded, selected_channels)
            title_parts = [
                f"{channel_type} Channels: {', '.join(selected_channels)}",
                metadata_str,
            ]
        else:
            fig = plot_channel_time_series(df_exploded, selected_channels[0])
            title_parts = [f"{selected_channels[0]} Signal", metadata_str]

        fig = update_fig_title(fig, title_parts)

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


def render_coordinates_plots(coords_data, metadata_str):
    if "x" not in coords_data.columns or "y" not in coords_data.columns:
        return

    fig_coords_time = plot_coordinates_time_series(coords_data, time_col="motion_time")

    fig_coords_time = update_fig_title(
        fig_coords_time, ["Coordinates vs Time", metadata_str]
    )
    st.plotly_chart(
        fig_coords_time,
        use_container_width=True,
    )

    fig_traj = plot_2d_trajectory(coords_data)
    fig_traj = update_fig_title(fig_traj, ["2D Trajectory", metadata_str])
    st.plotly_chart(
        fig_traj,
        use_container_width=True,
    )


def render_speed_plot(coords_data, metadata_str):
    if "tracing_speed" not in coords_data.columns:
        return

    fig_speed = plot_speed_time_series(coords_data, time_col="motion_time")

    fig_speed = update_fig_title(fig_speed, ["Tracing Speed", metadata_str])
    st.plotly_chart(
        fig_speed,
        use_container_width=True,
    )


def render_cross_trial_speed(block_data):
    st.subheader("Cross-Trial Speed Analysis")

    st.markdown("### Trial-Level Speed (Individual Trials)")
    st.markdown("Each line represents a single trial from the current block")

    speed_types = {"Combined": "combined", "X": "x", "Y": "y"}

    for display_name, internal_name in speed_types.items():
        fig_cross_trial = plot_cross_trial_speed(block_data, speed_type=internal_name)

        if len(fig_cross_trial.data) == 0:
            st.info(
                f"No motion data found for {display_name} speed in any trials in this block."
            )
        else:
            fig_cross_trial = update_fig_title(
                fig_cross_trial,
                [
                    f"{display_name} Speed Across All Trials ({len(fig_cross_trial.data)} trials)",
                    f"Participant {st.session_state.get('participant_id')}, Session {st.session_state.get('session')}, Block {st.session_state.get('block')}",
                ],
            )

            st.plotly_chart(
                fig_cross_trial,
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown("### Session-Level Average Speed (DBS ON vs OFF)")
    st.markdown(
        "Mean speed across **all trials in the entire session** for each DBS condition with ± 1 standard deviation"
    )

    for display_name, internal_name in speed_types.items():
        fig_session_avg = plot_session_average_speed(speed_type=internal_name)

        if len(fig_session_avg.data) == 0:
            st.info(f"No motion data found for {display_name} speed session averages.")
        else:
            fig_session_avg = update_fig_title(
                fig_session_avg,
                [
                    f"Session-Level Average {display_name} Speed (DBS ON vs OFF)",
                    f"Participant {st.session_state.get('participant_id')}, Session {st.session_state.get('session')}",
                ],
            )

            st.plotly_chart(
                fig_session_avg,
                use_container_width=True,
            )


def render_neural_tab(trial_data, lfp_channels, ecog_channels, metadata_str):
    render_neural_channels(trial_data, lfp_channels, "LFP", metadata_str)
    render_neural_channels(trial_data, ecog_channels, "ECoG", metadata_str)


def render_behavioral_tab(trial_data, metadata_str):
    coords_data, motion_cols = prepare_motion_data(trial_data)

    if coords_data is not None:
        render_coordinates_plots(coords_data, metadata_str)
        render_speed_plot(coords_data, metadata_str)
    elif motion_cols:
        st.info("Motion data could not be loaded for this trial.")
    else:
        st.info("No motion data available for this trial.")


def render_cross_trial_tab(block_data):
    render_cross_trial_speed(block_data)


def render_neural_behavioral_correlation(trial_data, lfp_channels, ecog_channels):
    st.markdown("### Neural-Behavioral Correlation Analysis")
    behavioral_vars = []
    if "tracing_speed" in trial_data.columns:
        behavioral_vars.append("tracing_speed")
    if "tracing_speed_x" in trial_data.columns:
        behavioral_vars.append("tracing_speed_x")
    if "tracing_speed_y" in trial_data.columns:
        behavioral_vars.append("tracing_speed_y")

    if not behavioral_vars:
        st.info("No behavioral variables available for correlation analysis.")
        return

    selected_behavior = st.selectbox(
        "Select Behavioral Variable", behavioral_vars, key="neural_beh_corr_var"
    )

    all_neural_channels = []
    if lfp_channels:
        all_neural_channels.extend(lfp_channels)
    if ecog_channels:
        all_neural_channels.extend(ecog_channels)

    if not all_neural_channels:
        st.info("No neural channels available.")
        return

    chunk_margin = (
        trial_data["chunk_margin"][0] if "chunk_margin" in trial_data.columns else 0
    )

    beh_data_raw = trial_data[selected_behavior][0]
    if beh_data_raw is None:
        st.warning(f"No data for {selected_behavior}")
        return

    beh_data = np.array(beh_data_raw).astype(float)

    correlations = []
    for ch in all_neural_channels:
        if ch not in trial_data.columns:
            continue

        ch_data_raw = trial_data[ch][0]
        if ch_data_raw is None:
            continue

        ch_data = np.array(ch_data_raw).astype(float)

        if chunk_margin > 0:
            ch_data = ch_data[chunk_margin:-chunk_margin]

        min_len = min(len(ch_data), len(beh_data))
        ch_data = ch_data[:min_len]
        beh_data_aligned = beh_data[:min_len]

        valid_mask = np.isfinite(ch_data) & np.isfinite(beh_data_aligned)
        ch_clean = ch_data[valid_mask]
        beh_clean = beh_data_aligned[valid_mask]

        if len(ch_clean) > 10:
            r = np.corrcoef(ch_clean, beh_clean)[0, 1]
            correlations.append((ch, r))

    if not correlations:
        st.warning("Could not compute correlations.")
        return

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    import plotly.graph_objects as go

    channels = [c[0] for c in correlations]
    r_values = [c[1] for c in correlations]

    colors = ["red" if r < 0 else "blue" for r in r_values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=channels,
            y=r_values,
            marker_color=colors,
            text=[f"{r:+.3f}" for r in r_values],
            textposition="outside",
            hovertemplate="%{x}<br>r = %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Linear Correlation: Neural Channels vs {selected_behavior}",
        xaxis_title="Channel",
        yaxis_title="Pearson r",
        yaxis=dict(range=[-1, 1]),
        height=500,
        showlegend=False,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_hline(
        y=0.3, line_dash="dot", line_color="green", opacity=0.3, annotation_text="r=0.3"
    )
    fig.add_hline(y=-0.3, line_dash="dot", line_color="green", opacity=0.3)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top 10 Channels by Absolute Correlation")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Channel**")
        for i, (ch, r) in enumerate(correlations[:10], 1):
            st.text(f"{i:2d}. {ch}")

    with col2:
        st.markdown("**Pearson r**")
        for i, (ch, r) in enumerate(correlations[:10], 1):
            st.text(f"{r:+.4f}")


def time_series_tab(block_data):
    st.header("Time-Series Analysis")

    trials_in_block = sorted(block_data["trial"].unique().to_list())
    selected_trial = st.selectbox(
        "Select a Trial to Render",
        options=trials_in_block,
    )

    trial_data = block_data.filter(pl.col("trial") == selected_trial)

    if trial_data is None or trial_data.is_empty():
        st.info("Select a block and trial to view time-series data.")
        return

    meta = get_trial_metadata(trial_data)
    duration = get_trial_duration(trial_data)

    metadata_str = format_trial_metadata(
        meta.get("participant_id"),
        meta.get("session"),
        meta.get("block"),
        meta.get("trial"),
        stim=meta.get("stim"),
        duration=duration,
    )

    st.subheader(metadata_str)

    lfp_channels, ecog_channels = get_channel_lists(block_data)

    neural_tab, behavioral_tab, neural_beh_tab, cross_trial_tab = st.tabs(
        ["Neural", "Behavioral", "Neural vs Behavioral", "Cross-Trial"]
    )

    with neural_tab:
        render_neural_tab(trial_data, lfp_channels, ecog_channels, metadata_str)

    with behavioral_tab:
        render_behavioral_tab(trial_data, metadata_str)

    with neural_beh_tab:
        render_neural_behavioral_correlation(trial_data, lfp_channels, ecog_channels)

    with cross_trial_tab:
        render_cross_trial_tab(block_data)
