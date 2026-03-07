import streamlit as st
import polars as pl
from pathlib import Path
import sys
import os

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Support custom results path via environment variable
# Usage: RESULTS_PATH=/path/to/local/results streamlit run dashboard_grid_search.py
results_root = Path(os.environ.get("RESULTS_PATH", project_root / "results"))

from utils.data_loader import (
    get_participant_sessions,
    load_participant_block_data,
    natural_sort_key,
    get_available_datasets,
    set_participants_path,
)
from dashboard.grid_search_tab import grid_search_tab
from utils.logger import setup_logger

logger = setup_logger("dashboard_logs", name=__name__)
logger.info("Grid Search Dashboard script started.")

st.set_page_config(layout="wide")

st.title("Grid Search Dashboard")

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

    sessions_dict = participant_sessions[selected_participant_id]
    selected_session = st.sidebar.selectbox(
        "Session", options=list(sessions_dict.keys()), key="sb_session"
    )
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
            st.session_state["participant_id"] = selected_participant_id
            st.session_state["session"] = selected_session
            st.session_state["block"] = selected_block
            st.sidebar.success(
                f"Loaded: P{selected_participant_id} S{selected_session} B{selected_block}"
            )

grid_search_tab(project_root, results_root)
