import streamlit as st
import polars as pl
from pathlib import Path
import sys
import os

project_root = Path(os.path.dirname(__file__)).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.data_loader import (
    get_participant_sessions,
    load_participant_block_data,
    natural_sort_key,
)
from dashboard.time_series_tab import time_series_tab
from dashboard.psd_analysis_tab import psd_analysis_tab
from dashboard.model_predictions_tab import model_predictions_tab
from dashboard.dbs_classification_tab import dbs_classification_tab
from dashboard.state_space_matrices_tab import state_space_matrices_tab
from utils.logger import setup_logger

logger = setup_logger("dashboard_logs", name=__name__)
logger.info("Dashboard script started.")

st.set_page_config(layout="wide")

st.title("iEEG & Motion Analysis Dashboard")

st.sidebar.header("Selection")
participant_sessions = get_participant_sessions()

if not participant_sessions:
    st.warning("No participant data found. Please check the data directory.")
    logger.warning("No participant data found.")
else:
    selected_participant_id = st.sidebar.selectbox(
        "Participant", options=list(participant_sessions.keys())
    )

    sessions_dict = participant_sessions[selected_participant_id]
    selected_session = st.sidebar.selectbox(
        "Session", options=list(sessions_dict.keys())
    )
    blocks = sessions_dict[selected_session]
    selected_block = st.sidebar.selectbox("Block", options=blocks)

    if st.sidebar.button("Load Data") and selected_session:
        data = load_participant_block_data(
            selected_participant_id, selected_session, selected_block
        )
        st.session_state["block_data"] = data
        st.session_state["participant_id"] = selected_participant_id
        st.session_state["session"] = selected_session
        st.session_state["block"] = selected_block
        st.sidebar.success(
            f"Loaded: P{selected_participant_id} | S{selected_session} | B{selected_block}"
        )
        logger.info(
            f"Data loaded for P{selected_participant_id}, S{selected_session}, B{selected_block}"
        )

if "block_data" not in st.session_state:
    st.info(
        "Select a participant and session from the sidebar and click 'Load Data' to begin."
    )
else:
    block_data = st.session_state["block_data"]

    lfp_channels = sorted(
        [
            col
            for col in block_data.columns
            if col.lower().startswith("lfp")
            and ("psd" not in col and "epochs" not in col)
        ],
        key=natural_sort_key,
    )
    ecog_channels = sorted(
        [
            col
            for col in block_data.columns
            if col.lower().startswith("ecog")
            and ("psd" not in col and "epochs" not in col)
        ],
        key=natural_sort_key,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Time-Series Analysis",
            "Frequency (PSD) Analysis",
            "Model Predictions",
            "DBS Classification",
        ]
    )

    with tab1:
        time_series_tab(block_data)

    with tab2:
        psd_analysis_tab(block_data, lfp_channels, ecog_channels)

    with tab3:
        model_predictions_tab(project_root)

    with tab4:
        dbs_classification_tab(project_root)

    # with tab5:
    #     state_space_matrices_tab(project_root)
