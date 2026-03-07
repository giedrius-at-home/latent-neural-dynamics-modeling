import streamlit as st
import polars as pl
from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Support custom results path via environment variable
# Usage: RESULTS_PATH=/path/to/local/results streamlit run dashboard_time_series_participant.py
results_root = Path(os.environ.get("RESULTS_PATH", project_root / "results"))

from utils.data_loader import (
    get_participant_sessions,
    load_participant_block_data,
    natural_sort_key,
    get_available_datasets,
    set_participants_path,
)
from dashboard.time_series_tab import (
    render_population_level_tab,
    render_group_raw_analysis,
)
from dashboard.time_series_plots import plot_participant_average_behavior
from utils.logger import setup_logger

logger = setup_logger("dashboard_logs", name=__name__)
logger.info("Time-Series Participant Level Dashboard script started.")

st.set_page_config(layout="wide")

st.title("Time-Series Analysis - Participant Level")

st.sidebar.header("Dataset Selection")

available_datasets = get_available_datasets()
if not available_datasets:
    st.sidebar.error("No datasets found in resampled_recordings/")
    st.stop()

selected_dataset = st.sidebar.selectbox(
    "Dataset",
    options=available_datasets,
    index=0,
    help="Select which resampled recordings dataset to use",
)
set_participants_path(selected_dataset)

if (
    "selected_dataset" not in st.session_state
    or st.session_state.get("selected_dataset") != selected_dataset
):
    st.session_state["selected_dataset"] = selected_dataset
    if "block_data" in st.session_state:
        del st.session_state["block_data"]

st.sidebar.divider()

participant_sessions = get_participant_sessions(selected_dataset)

if not participant_sessions:
    st.warning("No participant data found. Please check the data directory.")
    logger.warning("No participant data found.")
else:
    selected_participant_id = st.sidebar.selectbox(
        "Participant", options=list(participant_sessions.keys()), key="sb_participant"
    )

    # Set participant_id in session_state
    st.session_state["participant_id"] = selected_participant_id

    sessions_dict = participant_sessions[selected_participant_id]
    selected_session = st.sidebar.selectbox(
        "Session", options=list(sessions_dict.keys()), key="sb_session"
    )
    st.session_state["session"] = selected_session

    blocks = sessions_dict[selected_session]
    selected_block = st.sidebar.selectbox("Block", options=blocks, key="sb_block")

    if st.sidebar.button("Load Block Data"):
        with st.spinner(f"Loading Block {selected_block}..."):
            data = load_participant_block_data(
                selected_participant_id,
                selected_session,
                selected_block,
                selected_dataset,
            )
            st.session_state["block_data"] = data
            st.session_state["block"] = selected_block
            st.sidebar.success(
                f"Loaded: P{selected_participant_id} S{selected_session} B{selected_block}"
            )

# Participant Level Analysis - discover channels from participant data
block_data = st.session_state.get("block_data", None)
if block_data is not None and not block_data.is_empty():
    from dashboard.utils import get_channel_lists

    lfp_channels, ecog_channels, motion_channels = get_channel_lists(block_data)
else:
    # Discover channels from participant data (even without block_data)
    from dashboard.utils import discover_channels_from_participant_data

    if selected_participant_id:
        lfp_channels, ecog_channels = discover_channels_from_participant_data(
            selected_participant_id, selected_dataset
        )
    else:
        lfp_channels, ecog_channels = [], []

render_population_level_tab(lfp_channels + ecog_channels)
st.divider()

# Session-Level Average Behavior (DBS ON vs OFF) - Participant Level
st.markdown("### Session-Level Average Behavior (DBS ON vs OFF)")

behavioral_vars = [
    {"col": "tracing_velocity_magnitude", "name": "Speed", "unit": "pixels/s"},
    {"col": "tracing_velocity_y", "name": "Velocity in Y", "unit": "pixels/s"},
    {"col": "tracing_velocity_x", "name": "Velocity in X", "unit": "pixels/s"},
    {
        "col": "tracing_acceleration_magnitude",
        "name": "Acceleration Magnitude",
        "unit": "pixels/s²",
    },
    {
        "col": "tracing_acceleration_x",
        "name": "Acceleration in X",
        "unit": "pixels/s²",
    },
    {
        "col": "tracing_acceleration_y",
        "name": "Acceleration in Y",
        "unit": "pixels/s²",
    },
    {
        "col": "tracing_jerk_magnitude",
        "name": "Jerk Magnitude",
        "unit": "pixels/s³",
    },
    {"col": "tracing_jerk_x", "name": "Jerk in X", "unit": "pixels/s³"},
    {"col": "tracing_jerk_y", "name": "Jerk in Y", "unit": "pixels/s³"},
]

# Check if we have block_data to verify which columns exist
if block_data is not None and not block_data.is_empty():
    available_cols = set(block_data.columns)
else:
    # Try to load a sample to check columns
    try:
        from utils.data_loader import load_participant_session_data

        sample_data = load_participant_session_data(
            selected_participant_id,
            (
                list(participant_sessions[selected_participant_id].keys())[0]
                if participant_sessions.get(selected_participant_id)
                else None
            ),
            selected_dataset,
            columns=None,
        )
        available_cols = (
            set(sample_data.columns) if not sample_data.is_empty() else set()
        )
    except:
        available_cols = set()

for var in behavioral_vars:
    if var["col"] not in available_cols:
        continue

    fig_participant_avg = plot_participant_average_behavior(
        behavioral_col=var["col"], y_label=f"{var['name']} ({var['unit']})"
    )

    st.plotly_chart(fig_participant_avg, use_container_width=True)
    st.caption(f"{var['name']} — Participant average across sessions (DBS ON vs OFF)")

st.divider()
render_group_raw_analysis(selected_dataset)
