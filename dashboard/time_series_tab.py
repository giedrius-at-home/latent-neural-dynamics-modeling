import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yaml
from pathlib import Path

SAMPLING_FREQ = 60

from dashboard.time_series_plots import (
    plot_channel_time_series,
    plot_multi_channel_time_series,
    plot_coordinates_time_series,
    plot_speed_time_series,
    plot_acceleration_time_series,
    plot_jerk_time_series,
    plot_2d_trajectory,
    plot_component_time_series,
    plot_cross_trial_speed,
    plot_session_average_speed,
    plot_signal_alignment,
    compute_cross_correlation_lag,
    plot_population_time_series,
    plot_session_comparison_time_series,
    plot_raw_alignment,
    plot_residual_violin,
    plot_micro_alignment,
)
from utils.data_loader import (
    get_available_datasets,
    get_all_participants,
    load_participant_data,
    get_participant_sessions,
)
from dashboard.utils import get_channel_lists, get_trial_metadata
from dashboard.backbone import (
from utils.data_loader import (
    load_participant_session_data,
    DATA_PATH,
)
from utils.data_loader import load_participant_session_data
from utils.sync import interpolate_to_grid
import plotly.graph_objects as go
    format_title,
    format_trial_metadata,
    update_fig_title,
    PLOT_STYLE,
)


def prepare_motion_data(trial_data):
    time_col = "motion_time" if "motion_time" in trial_data.columns else "time"
    motion_cols = [
        c for c in trial_data.columns if c.startswith("tracing_") or c in ["x", "y"]
    ]
    if time_col not in motion_cols:
        motion_cols.append(time_col)

    # Ensure all columns have the same length before exploding
    lens = [
        trial_data.select(pl.col(c).list.len().first()).to_series()[0]
        for c in motion_cols
    ]
    min_len = min(lens)

    coords_data = trial_data.select(
        *[pl.col(c).list.slice(0, min_len).alias(c) for c in motion_cols],
        "participant_id",
        "session",
        "block",
        "trial",
    ).explode(*motion_cols)

    return coords_data, motion_cols, time_col


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
            channels_str = ", ".join(selected_channels[:3])
            if len(selected_channels) > 3:
                channels_str += f" + {len(selected_channels) - 3} more"
            caption = f"Multi-Channel Neural Time Series — {channels_str}"
        else:
            fig = plot_channel_time_series(df_exploded, selected_channels[0])
            caption = f"Neural Time Series — {selected_channels[0]}"

        st.plotly_chart(
            fig,
            use_container_width=True,
        )
        st.caption(caption)


def render_behavioral_tab(trial_data, metadata_str):
    coords_data, motion_cols, time_col = prepare_motion_data(trial_data)

    groups = {}
    for col in motion_cols:
        if col == time_col:
            continue

        if col in ["x", "y"]:
            base = "position"
            comp = col
        elif col.endswith("_x"):
            base = col[:-2]
            comp = "x"
        elif col.endswith("_y"):
            base = col[:-2]
            comp = "y"
        elif col.endswith("_magnitude"):
            base = col[:-10]
            comp = "magnitude"
        else:
            base = col
            comp = "value"

        if base not in groups:
            groups[base] = {}
        groups[base][comp] = col

    sorted_groups = sorted(groups.items(), key=lambda x: (x[0] != "position", x[0]))

    for base, comps in sorted_groups:
        display_base = base.replace("tracing_", "").replace("_", " ").title()
        st.markdown(f"### {display_base}")
        if "x" in comps and "y" in comps:
            fig_ts = plot_component_time_series(
                coords_data,
                [comps["x"], comps["y"]],
                time_col=time_col,
                title=f"{display_base} Over Time",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            if "position" in base.lower():
                fig_2d = plot_2d_trajectory(
                    coords_data,
                    x_col=comps["x"],
                    y_col=comps["y"],
                )
                st.plotly_chart(fig_2d, use_container_width=True)

        elif "value" in comps:
            fig_ts = plot_component_time_series(
                coords_data,
                [comps["value"]],
                time_col=time_col,
                title=f"{display_base} Over Time",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        if "magnitude" in comps:
            col_name = comps["magnitude"]
            if "velocity" in col_name.lower() or "speed" in col_name.lower():
                fig_mag = plot_speed_time_series(
                    coords_data, col_name=col_name, time_col=time_col
                )
            elif "acceleration" in col_name.lower():
                fig_mag = plot_acceleration_time_series(
                    coords_data, col_name=col_name, time_col=time_col
                )
            elif "jerk" in col_name.lower():
                fig_mag = plot_jerk_time_series(
                    coords_data, col_name=col_name, time_col=time_col
                )
            else:
                fig_mag = plot_component_time_series(
                    coords_data,
                    [col_name],
                    time_col=time_col,
                    title=f"{display_base} Magnitude",
                )
            st.plotly_chart(fig_mag, use_container_width=True)

        st.markdown("---")


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
            st.plotly_chart(
                fig_cross_trial,
                use_container_width=True,
            )
            st.caption(
                f"{display_name} speed — Individual trial traces from current block"
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
            st.plotly_chart(
                fig_session_avg,
                use_container_width=True,
            )
            st.caption(
                f"{display_name} speed — Session average with ±1 SD (DBS ON vs OFF)"
            )


def render_neural_tab(trial_data, lfp_channels, ecog_channels, metadata_str):
    render_neural_channels(trial_data, lfp_channels, "LFP", metadata_str)
    render_neural_channels(trial_data, ecog_channels, "ECoG", metadata_str)


def render_cross_trial_tab(block_data):
    render_cross_trial_speed(block_data)


def render_signal_alignment_analysis(
    trial_data, neural_channel: str, behavioral_var: str, chunk_margin: float
):
    margin_samples = int(chunk_margin * SAMPLING_FREQ)

    neural_raw = trial_data[neural_channel][0]
    if neural_raw is None:
        return
    neural_data = np.array(neural_raw).astype(float)

    beh_raw = trial_data[behavioral_var][0]
    if beh_raw is None:
        return
    beh_data = np.array(beh_raw).astype(float)

    time_raw = trial_data["time"][0]
    if time_raw is None:
        return
    time_neural = np.array(time_raw).astype(float)

    if "time_original" in trial_data.columns:
        time_orig_raw = trial_data["time_original"][0]
        if time_orig_raw is not None:
            time_beh = np.array(time_orig_raw).astype(float)
        else:
            time_beh = None
    else:
        time_beh = None

    if time_beh is None:
        if margin_samples > 0 and len(time_neural) > 2 * margin_samples:
            time_beh = time_neural[margin_samples:-margin_samples][: len(beh_data)]
        else:
            time_beh = time_neural[: len(beh_data)]

    time_beh = time_beh[: len(beh_data)]

    if margin_samples > 0 and len(neural_data) > 2 * margin_samples:
        neural_trimmed = neural_data[margin_samples:-margin_samples]
    else:
        neural_trimmed = neural_data

    min_len = min(len(neural_trimmed), len(beh_data))
    neural_for_corr = neural_trimmed[:min_len]
    beh_for_corr = beh_data[:min_len]

    lag_samples, max_corr, lags, correlation = compute_cross_correlation_lag(
        neural_for_corr, beh_for_corr
    )
    lag_seconds = lag_samples / SAMPLING_FREQ

    fig = plot_signal_alignment(
        time_neural,
        neural_data,
        time_beh,
        beh_data,
        neural_channel,
        behavioral_var,
        chunk_margin,
        lags,
        correlation,
        lag_seconds,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Optimal Lag", f"{lag_seconds:.3f} s", f"{lag_samples} samples")
    with col2:
        st.metric("Max Correlation", f"{max_corr:.4f}")


def render_neural_behavioral_correlation(trial_data, lfp_channels, ecog_channels):
    st.markdown("### Neural-Behavioral Signal Analysis")

    # Dynamically find all behavioral columns
    behavioral_vars = sorted(
        [c for c in trial_data.columns if c.startswith("tracing_")]
    )
    if "x" in trial_data.columns:
        behavioral_vars = ["x", "y"] + behavioral_vars

    if not behavioral_vars:
        st.info("No behavioral variables available for analysis.")
        return

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
    margin_samples = int(chunk_margin * SAMPLING_FREQ)

    alignment_tab, correlation_tab = st.tabs(
        ["🔗 Signal Alignment (Cross-Correlation)", "📊 Linear Correlation"]
    )

    with alignment_tab:

        col1, col2 = st.columns(2)
        with col1:
            selected_neural = st.selectbox(
                "Select Neural Channel",
                all_neural_channels,
                key="alignment_neural_channel",
            )
        with col2:
            selected_beh_align = st.selectbox(
                "Select Behavioral Variable", behavioral_vars, key="alignment_beh_var"
            )

        if selected_neural and selected_beh_align:
            render_signal_alignment_analysis(
                trial_data, selected_neural, selected_beh_align, chunk_margin
            )

    with correlation_tab:
        st.markdown(
            "Linear correlation (Pearson r) between neural channels and behavioral variables."
        )

        selected_behavior = st.selectbox(
            "Select Behavioral Variable", behavioral_vars, key="neural_beh_corr_var"
        )

        beh_data_raw = trial_data[selected_behavior][0]
        if beh_data_raw is None:
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

            if margin_samples > 0 and len(ch_data) > 2 * margin_samples:
                ch_data = ch_data[margin_samples:-margin_samples]

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
            return

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)


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

        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig.add_hline(
            y=0.3,
            line_dash="dot",
            line_color="#38405f",
            opacity=0.3,
            annotation_text="r=0.3",
        )
        fig.add_hline(y=-0.3, line_dash="dot", line_color="#38405f", opacity=0.3)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top 20 Channels by Absolute Correlation")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Channel**")
            for i, (ch, r) in enumerate(correlations[:20], 1):
                st.text(f"{i:2d}. {ch}")

        with col2:
            st.markdown("**Pearson r**")
            for i, (ch, r) in enumerate(correlations[:20], 1):
                st.text(f"{r:+.4f}")


def render_population_level_tab(neural_channels):
    if not neural_channels:
        st.info("No neural channels available.")
        return

    st.markdown("#### Population Level Analysis (All Participants)")
    target_channel = st.selectbox(
        "Select Channel", neural_channels, key="pop_chan_select"
    )

    selected_dataset = st.session_state.get("selected_dataset")
    participants = get_all_participants(selected_dataset)

    if not participants:
        st.warning("No participants found.")
        return

    population_data = {}

    # We use a standard time axis based on the first participant to align data roughly
    # In a real scenario, we'd align to event onset (0) and use a fixed window.
    # Assuming standard window [-0.5, 2.5] or similar.
    # We will try to load one to get the time axis length.

    common_time_axis = None

    with st.spinner("Loading population data..."):
        for p_id in participants:
            try:
                # Load only required columns for participant to aggregate
                # This drastically reduces memory usage
                cols_to_load = [target_channel, "stim"]
                df = load_participant_data(p_id, selected_dataset, columns=cols_to_load)
                if df.is_empty() or target_channel not in df.columns:
                    continue

                on_df = df.filter(pl.col("stim") == "on")
                off_df = df.filter(pl.col("stim") == "off")

                def compute_mean_trace_and_time(sub_df, chan):
                    if sub_df.is_empty():
                        return None, None
                    cols = sub_df[chan].to_list()
                    max_len = max([len(x) for x in cols]) if cols else 0
                    if max_len == 0:
                        return None, None

                    mat = np.full((len(cols), max_len), np.nan)
                    for i, c in enumerate(cols):
                        arr = np.array(c)
                        length = min(len(arr), max_len)
                        mat[i, :length] = arr[:length]

                    mean_trace = np.nanmean(mat, axis=0)

                    # Try to get time axis from first trial
                    time_vec = None
                    if "time_vector" in sub_df.columns:  # unlikely
                        pass
                    else:
                        # Construct
                        dur = max_len / SAMPLING_FREQ
                        # Assume onset at 0? Or aligned relative?
                        # Usually standard preprocessing aligns onset.
                        # Let's assume 0 start for simplicity of "averaged raw data".
                        # Or better, look for 'onset' column.
                        onset = 0
                        if "onset" in sub_df.columns:
                            onset = sub_df["onset"][0]  # Just use first trial's onset

                        # If we don't have time vector, we make one.
                        # Ideally we should use 'time_original' if aligned.
                        time_vec = np.linspace(0, dur, max_len)
                        # Shift by onset?
                        # Usually 'onset' is the time where event happens.
                        # If data is [-1, 2], onset is 1.0.
                        # We want to align 0 to Event.
                        # But let's just plot 'Relative Time' or 'Absolute Time'.
                        # If trials are not aligned, averaging is meaningless.
                        # WE ASSUME ALIGNED DATA.

                    return mean_trace, time_vec

                on_mean, t_on = compute_mean_trace_and_time(on_df, target_channel)
                off_mean, t_off = compute_mean_trace_and_time(off_df, target_channel)

                population_data[p_id] = {"on": on_mean, "off": off_mean}

                if common_time_axis is None:
                    if t_on is not None:
                        common_time_axis = t_on
                    elif t_off is not None:
                        common_time_axis = t_off

            except Exception:
                continue

    if population_data and common_time_axis is not None:
        # Fix lengths if needed (simple trunction/pad logic or just slice)
        # We just use common_time_axis length
        pass

        fig = plot_population_time_series(population_data, common_time_axis)
        st.plotly_chart(fig, use_container_width=True)
        participant_list = ", ".join(population_data.keys())
        st.caption(f"Population Level Comparison — Participants: {participant_list}")
    else:
        st.info("Could not compute population averages.")


def render_session_level_tab(neural_channels):
    if not neural_channels:
        return

    st.markdown("#### Session Level Analysis (One Participant)")
    participant_id = st.session_state.get("participant_id")
    if not participant_id:
        st.warning("Please select a participant in sidebar.")
        return

    target_channel = st.selectbox(
        "Select Channel", neural_channels, key="sess_chan_select"
    )
    selected_dataset = st.session_state.get("selected_dataset")

    sessions = get_participant_sessions(selected_dataset).get(participant_id, {})

    if not sessions:
        st.warning("No sessions found.")
        return

    st.info(
        f"Found {len(sessions)} session(s) for {participant_id}: {list(sessions.keys())}"
    )

    session_data_map = {}
    common_time_axis = None

    with st.spinner(f"Loading session data for {participant_id}..."):
        for s_id in sessions:
            try:

                cols_to_load = [target_channel, "stim"]
                df = load_participant_session_data(
                    participant_id, s_id, selected_dataset, columns=cols_to_load
                )

                if df.is_empty() or target_channel not in df.columns:
                    continue

                on_df = df.filter(pl.col("stim") == "on")
                off_df = df.filter(pl.col("stim") == "off")

                def compute_mean_trace(sub_df, chan):
                    if sub_df.is_empty():
                        return None
                    cols = sub_df[chan].to_list()
                    max_len = max([len(x) for x in cols]) if cols else 0
                    if max_len == 0:
                        return None
                    mat = np.full((len(cols), max_len), np.nan)
                    for i, c in enumerate(cols):
                        arr = np.array(c)
                        length = min(len(arr), max_len)
                        mat[i, :length] = arr[:length]
                    return np.nanmean(mat, axis=0)

                on_mean = compute_mean_trace(on_df, target_channel)
                off_mean = compute_mean_trace(off_df, target_channel)

                session_data_map[s_id] = {"on": on_mean, "off": off_mean}

                if common_time_axis is None:
                    L = (
                        len(on_mean)
                        if on_mean is not None
                        else (len(off_mean) if off_mean is not None else 0)
                    )
                    if L > 0:
                        common_time_axis = np.linspace(0, L / SAMPLING_FREQ, L)

            except Exception:
                continue

    if session_data_map and common_time_axis is not None:
        fig = plot_session_comparison_time_series(
            session_data_map, common_time_axis, participant_id
        )
        st.plotly_chart(fig, use_container_width=True)
        session_list = ", ".join(
            [
                f"S{s}"
                for s in sorted(
                    session_data_map.keys(),
                    key=lambda x: int(x) if str(x).isdigit() else x,
                )
            ]
        )
        st.caption(f"Session Comparison — {participant_id} — Sessions: {session_list}")

        st.markdown("---")
        st.markdown("### Neural-Behavioral Correlation Matrix (Across Session)")

        # --- Dynamic Feature Discovery from Configs ---
        config_dir = Path("training/setups/psid_grid_search")
        cfg_neural = set()
        # Track per-session behavioral outputs from configs
        session_behav_map = {}  # s_id -> set of behavioral outputs

        # Search for configs matching this participant and selected sessions
        for s_id in sessions:
            yaml_files = list(
                config_dir.glob(f"gs_{participant_id}_S{s_id}_log_power*.yaml")
            )
            for yf in yaml_files:
                try:
                    with open(yf, "r") as f:
                        cfg = yaml.safe_load(f)
                    gs = cfg.get("model", {}).get("grid_search", {})

                    # Extract Neural Bands
                    nb = gs.get("neural_bands", [])
                    if nb and isinstance(nb[0], list):
                        for sl in nb:
                            cfg_neural.update(sl)
                    elif nb:
                        cfg_neural.update(nb)

                    # Extract Behavioral Outputs per session
                    if s_id not in session_behav_map:
                        session_behav_map[s_id] = set()
                    bo = gs.get("behavioral_outputs", [])
                    if bo and isinstance(bo[0], list):
                        for sl in bo:
                            session_behav_map[s_id].update(sl)
                    elif bo:
                        session_behav_map[s_id].update(bo)
                except:
                    continue

        # Fallback behaviors per session if none found
        fallback_behav = {
            "tracing_velocity_y",
            "tracing_velocity_magnitude",
            "tracing_jerk_x",
            "tracing_jerk_magnitude",
        }

        if not cfg_neural:
            # Fallback neural features (4 ECOG + 16 LFP pattern)
            CONFIG_BANDS = ["alpha", "beta", "delta", "highbeta", "lowbeta", "theta"]
            B_CHANS = ["ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"] + [
                f"LFP_{i}" for i in range(9, 17)
            ]
            cfg_neural = {
                f"{ch}_{band}_log_power" for band in CONFIG_BANDS for ch in B_CHANS
            }

        target_neural = sorted(list(cfg_neural))

        all_z_matrices = []
        all_avail_neural = set()
        all_avail_behav = set()

        with st.spinner("Analyzing neural-behavioral synchrony..."):
            for s_id in sessions:
                try:

                    p_partition = f"participant_id={participant_id}"
                    s_partition = f"session={s_id}"
                    test_path = DATA_PATH / selected_dataset / p_partition / s_partition
                    first_file = next(test_path.rglob("*.parquet"), None)

                    if not first_file:
                        continue

                    schema = pl.read_parquet_schema(first_file)
                    # For this session, find what's in schema that matches our target
                    s_neural = [c for c in target_neural if c in schema]
                    # Use per-session behavioral outputs from config
                    sess_target_behav = sorted(
                        list(session_behav_map.get(s_id, fallback_behav))
                    )
                    s_behav = [c for c in sess_target_behav if c in schema]

                    if not s_neural or not s_behav:
                        continue

                    s_cols = s_neural + s_behav
                    load_cols = s_cols + (
                        ["chunk_margin"] if "chunk_margin" in schema else []
                    )
                    df_sess = load_participant_session_data(
                        participant_id, s_id, selected_dataset, columns=load_cols
                    )

                    if df_sess.is_empty():
                        continue

                    # Determine chunk margin for neural trimming
                    if "chunk_margin" in df_sess.columns:
                        chunk_margin = df_sess["chunk_margin"][0]
                    else:
                        chunk_margin = 2  # default 2 seconds
                    margin_samples = int(chunk_margin * 80)  # 80 Hz sampling

                    # Process trial by trial: trim neural margin, then align
                    all_neural = []
                    all_behav = []

                    for row_idx in range(df_sess.height):
                        # Get behavioral arrays and their length
                        behav_arrays = []
                        behav_len = None
                        for bc in s_behav:
                            arr = np.array(df_sess[bc][row_idx], dtype=float)
                            if behav_len is None:
                                behav_len = len(arr)
                            behav_arrays.append(arr)

                        if behav_len is None or behav_len < 10:
                            continue

                        # Trim neural features by chunk margin on both sides
                        neural_arrays = []
                        for nc in s_neural:
                            arr = np.array(df_sess[nc][row_idx], dtype=float)
                            if margin_samples > 0 and len(arr) > 2 * margin_samples:
                                arr = arr[margin_samples:-margin_samples]
                            arr = arr[:behav_len]
                            if len(arr) < behav_len:
                                arr = np.pad(
                                    arr,
                                    (0, behav_len - len(arr)),
                                    constant_values=np.nan,
                                )
                            neural_arrays.append(arr)

                        all_neural.append(np.column_stack(neural_arrays))
                        all_behav.append(np.column_stack(behav_arrays))

                    if not all_neural:
                        continue

                    data_n = np.vstack(all_neural)
                    data_b = np.vstack(all_behav)

                    mask = np.all(np.isfinite(data_n), axis=1) & np.all(
                        np.isfinite(data_b), axis=1
                    )
                    if np.sum(mask) < 100:
                        continue

                    data_n, data_b = data_n[mask], data_b[mask]

                    # Pearson Correlation
                    xn = data_n - np.mean(data_n, axis=0)
                    xb = data_b - np.mean(data_b, axis=0)
                    std_n = np.linalg.norm(xn, axis=0)
                    std_b = np.linalg.norm(xb, axis=0)

                    mask_n = std_n > 1e-10
                    mask_b = std_b > 1e-10

                    r_sess = np.zeros((len(s_neural), len(s_behav)))
                    if np.any(mask_n) and np.any(mask_b):
                        r_sess[np.ix_(mask_n, mask_b)] = (
                            xn[:, mask_n] / std_n[mask_n]
                        ).T @ (xb[:, mask_b] / std_b[mask_b])

                    # Fisher Z
                    z_sess = np.arctanh(np.clip(r_sess, -0.999, 0.999))

                    all_z_matrices.append(
                        {
                            "z": z_sess,
                            "neural": s_neural,
                            "behav": s_behav,
                            "session": s_id,
                        }
                    )
                    all_avail_neural.update(s_neural)
                    all_avail_behav.update(s_behav)

                except Exception:
                    continue

        if all_z_matrices:
            # Show per-session heatmaps (each session has its own behavioral targets)
            for entry in all_z_matrices:
                z_mat = entry["z"]
                s_neural_list = entry["neural"]
                s_behav_list = entry["behav"]
                s_id = entry["session"]

                # Rank by mean absolute Fisher z (both directions informative for PSID)
                mean_abs_z = np.mean(np.abs(z_mat), axis=1)
                top_20_idx = np.argsort(mean_abs_z)[::-1][:20]

                top_20_labels = [s_neural_list[i] for i in top_20_idx]
                top_20_z = z_mat[top_20_idx, :]

                fig_heat = px.imshow(
                    top_20_z,
                    labels=dict(
                        x="Behavioral Output", y="Neural Feature", color="Fisher z"
                    ),
                    x=[
                        b.replace("tracing_", "").replace("_", " ").title()
                        for b in s_behav_list
                    ],
                    y=top_20_labels,
                    color_continuous_scale="RdBu_r",
                    aspect="auto",
                    zmin=-0.2,
                    zmax=0.2,
                )
                fig_heat.update_layout(
                    title=f"Top 20 Neural Features (Fisher z) — {participant_id} Session {s_id}",
                    template="plotly_white",
                    height=800,
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption(
                    f"Session {s_id} — Behavioral outputs: {', '.join(b.replace('tracing_', '') for b in s_behav_list)}"
                )
        else:
            st.warning(
                "Insufficient data or missing columns in the selected sessions for correlation analysis."
            )
    else:
        st.info("No session data available.")


def render_raw_alignment_tab(trial_data, block_data, lfp_channels, ecog_channels):

    st.markdown("### Raw Data Alignment & Interpolation Analysis")

    behavioral_vars = ["x", "y"]

    all_neural_channels = []
    if lfp_channels:
        all_neural_channels.extend(lfp_channels)
    if ecog_channels:
        all_neural_channels.extend(ecog_channels)

    selected_neural = st.selectbox(
        "Select Neural Channel (Optional)",
        ["None"] + all_neural_channels,
        key="raw_neural_channel",
    )

    beh_raw_x_list = trial_data["x"][0]
    beh_raw_y_list = trial_data["y"][0]
    time_raw_list = trial_data["motion_time"][0]
    time_master_list = trial_data["time_original"][0]
    time_neural_list = trial_data["time"][0]

    if (
        beh_raw_x_list is None
        or beh_raw_y_list is None
        or time_raw_list is None
        or time_master_list is None
        or time_neural_list is None
    ):
        st.info("Missing essential time or behavior data.")
        return

    time_raw = np.array(time_raw_list).astype(float)
    time_master = np.array(time_master_list).astype(float)
    time_neural = np.array(time_neural_list).astype(float)

    chunk_margin = (
        trial_data["chunk_margin"][0] if "chunk_margin" in trial_data.columns else 0.0
    )

    neural_data = None
    if selected_neural != "None" and selected_neural in trial_data.columns:
        nd = trial_data[selected_neural][0]
        if nd is not None:
            neural_data = np.array(nd).astype(float)

    beh_raw_dict = {}
    beh_interp_dict = {}

    for var in ["x", "y"]:
        b_r = np.array(trial_data[var][0]).astype(float)
        beh_raw_dict[var] = b_r
        beh_interp_dict[var] = interpolate_to_grid(b_r, time_raw, time_master)

    st.markdown("### Alignment Analysis")

    min_t = float(time_master[0])
    max_t = float(time_master[-1])
    window_start = min_t
    window_end = max_t

    fig_micro = plot_micro_alignment(
        time_master=time_neural,
        neural_data=neural_data,
        time_raw=time_raw,
        beh_raw_dict=beh_raw_dict,
        time_interp=time_master,
        beh_interp_dict=beh_interp_dict,
        neural_channel=selected_neural if selected_neural != "None" else "",
        window_start=window_start,
        window_end=window_end,
    )
    st.plotly_chart(fig_micro, use_container_width=True)

    st.markdown("---")
    st.markdown("### Cross-Trial Interpolation Residuals")

    trials = sorted(block_data["trial"].unique().to_list())
    all_residuals = []
    valid_trials = []

    global_residuals = []

    selected_beh = "x"

    for tr in trials:
        tr_data = block_data.filter(pl.col("trial") == tr)
        if tr_data.is_empty():
            continue

        br_l = tr_data[selected_beh][0]
        tr_l = tr_data["motion_time"][0]
        tm_l = tr_data["time_original"][0]

        if br_l is None or tr_l is None or tm_l is None:
            continue

        br = np.array(br_l).astype(float)
        tr_ts = np.array(tr_l).astype(float)
        tm_ts = np.array(tm_l).astype(float)

        bi = interpolate_to_grid(br, tr_ts, tm_ts)

        tr_res = []
        for i, t_m in enumerate(tm_ts):
            if np.isnan(bi[i]):
                continue
            idx_nearest = np.argmin(np.abs(tr_ts - t_m))
            tr_res.append(bi[i] - br[idx_nearest])

        if tr_res:
            res_arr = np.array(tr_res)
            all_residuals.append(res_arr)
            valid_trials.append(tr)
            global_residuals.extend(tr_res)

    if all_residuals:
        g_res = np.array(global_residuals)
        g_rmse = np.sqrt(np.mean(g_res**2))
        g_max_err = np.max(np.abs(g_res))

        fig2 = plot_residual_violin(all_residuals, valid_trials)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            f"**Residual Distribution**: Cross-trial spread of interpolation errors. | **Block RMSE**: {g_rmse:.4f} | **Block Max Error**: {g_max_err:.4f}"
        )
    else:
        st.info("Not enough data to compute cross-trial residuals.")


def time_series_tab(block_data):
    st.header("Time-Series Analysis")

    lfp_channels, ecog_channels, motion_channels = get_channel_lists(block_data)

    # Outer tabs for Level of Analysis
    trial_level_tab, session_level_tab, pop_level_tab = st.tabs(
        ["Trial Level", "Session Level", "Participant Level"]
    )

    with trial_level_tab:
        trials_in_block = sorted(block_data["trial"].unique().to_list())
        # If no trials, warn
        if not trials_in_block:
            st.warning("No trials in block.")
        else:
            selected_trial = st.selectbox(
                "Select a Trial to Render",
                options=trials_in_block,
            )

            trial_data = block_data.filter(pl.col("trial") == selected_trial)

            if trial_data is None or trial_data.is_empty():
                st.info("Select a block and trial to view time-series data.")
            else:
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

                neural_tab, behavioral_tab, cross_trial_tab, raw_alignment_tab = (
                    st.tabs(["Neural", "Behavioral", "Cross-Trial", "Raw Alignment"])
                )

                with neural_tab:
                    render_neural_tab(
                        trial_data, lfp_channels, ecog_channels, metadata_str
                    )

                with behavioral_tab:
                    render_behavioral_tab(trial_data, metadata_str)

                with cross_trial_tab:
                    render_cross_trial_speed(block_data)

                with raw_alignment_tab:
                    render_raw_alignment_tab(
                        trial_data, block_data, lfp_channels, ecog_channels
                    )

    with session_level_tab:
        render_session_level_tab(lfp_channels + ecog_channels)

    with pop_level_tab:
        render_population_level_tab(lfp_channels + ecog_channels)
