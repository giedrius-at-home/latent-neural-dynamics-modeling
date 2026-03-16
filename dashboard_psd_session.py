import streamlit as st
import polars as pl
from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Support custom results path via environment variable
# Usage: RESULTS_PATH=/path/to/local/results streamlit run dashboard_psd_session.py
results_root = Path(os.environ.get("RESULTS_PATH", project_root / "results"))

from utils.data_loader import (
    get_participant_sessions,
    load_participant_block_data,
    natural_sort_key,
    get_available_datasets,
    set_participants_path,
)
from dashboard.psd_analysis_tab import render_average_psd_session_level
from utils.logger import setup_logger

logger = setup_logger("dashboard_logs", name=__name__)
logger.info("PSD Session Level Dashboard script started.")

st.set_page_config(layout="wide")

st.title("PSD Analysis - Session Level")

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

    # Set participant_id in session_state so Session Level tab can access it
    st.session_state["participant_id"] = selected_participant_id

    sessions_dict = participant_sessions[selected_participant_id]
    selected_session = st.sidebar.selectbox(
        "Session", options=list(sessions_dict.keys()), key="sb_session"
    )
    # Set session in session_state for Session Level tab
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

# Session Level PSD Analysis - doesn't require block_data, but we can get channels from it if available
block_data = st.session_state.get("block_data", None)
if block_data is not None and not block_data.is_empty():
    from dashboard.utils import get_channel_lists

    lfp_channels, ecog_channels, motion_channels = get_channel_lists(block_data)
else:
    # Try to discover channels from session data
    # For now, use common channel names - render_average_psd_session_level will filter based on available data
    lfp_channels = [f"LFP_{i}" for i in range(1, 17)]
    ecog_channels = [f"ECOG_{i}" for i in range(1, 5)]

st.markdown("### Session-level Average PSD")

col1_session, col2_session = st.columns(2)

with col1_session:
    st.markdown("#### LFP")
    selected_lfp_session = st.multiselect(
        "Select LFP Channels", lfp_channels, key="lfp_psd_avg_session"
    )
    if selected_lfp_session:
        render_average_psd_session_level(selected_lfp_session, "LFP")

with col2_session:
    st.markdown("#### ECoG")
    selected_ecog_session = st.multiselect(
        "Select ECoG Channels", ecog_channels, key="ecog_psd_avg_session"
    )
    if selected_ecog_session:
        render_average_psd_session_level(selected_ecog_session, "ECoG")
