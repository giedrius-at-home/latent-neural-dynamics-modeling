import streamlit as st
import polars as pl
import re
from pathlib import Path
from collections import defaultdict

from utils.file_handling import get_child_subchilds_tuples
from utils.logger import setup_logger

logger = setup_logger("dashboard_logs", name=__name__)

DATA_PATH = Path("resampled_recordings")
DEFAULT_DATASET = "participants_at_60Hz_scaled_1e6_full_features"
PARTICIPANTS_PATH = DATA_PATH / DEFAULT_DATASET


def get_available_datasets() -> list[str]:
    if not DATA_PATH.exists():
        return []
    datasets = [
        d.name
        for d in DATA_PATH.iterdir()
        if d.is_dir() and d.name.startswith("participants")
    ]
    return sorted(datasets)


def set_participants_path(dataset_name: str) -> Path:
    global PARTICIPANTS_PATH
    PARTICIPANTS_PATH = DATA_PATH / dataset_name
    logger.info(f"PARTICIPANTS_PATH set to: {PARTICIPANTS_PATH}")
    return PARTICIPANTS_PATH


def get_participants_path() -> Path:
    return PARTICIPANTS_PATH


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


@st.cache_data
def get_participant_sessions(dataset: str = None):
    participants_path = DATA_PATH / dataset if dataset else PARTICIPANTS_PATH
    logger.info(f"PARTICIPANTS_PATH: {participants_path}")
    logger.info(f"PARTICIPANTS_PATH exists: {participants_path.exists()}")
    if not participants_path.exists():
        st.error(f"Data directory not found at: {participants_path}")
        return {}

    session_tuples = get_child_subchilds_tuples(participants_path)
    logger.info(f"session_tuples: {session_tuples}")
    participant_sessions = defaultdict(lambda: defaultdict(list))
    for _1, p, s, b in session_tuples:
        p_id = p.split("=")[1]
        s_id = s.split("=")[1]
        b_id = b.split("=")[1]
        if b_id not in participant_sessions[p_id][s_id]:
            participant_sessions[p_id][s_id].append(b_id)

    for p_id in participant_sessions:
        for s_id in participant_sessions[p_id]:
            participant_sessions[p_id][s_id].sort(key=natural_sort_key)
        participant_sessions[p_id] = dict(
            sorted(
                participant_sessions[p_id].items(),
                key=lambda kv: natural_sort_key(kv[0]),
            )
        )

    return dict(sorted(participant_sessions.items()))


@st.cache_data
def load_participant_block_data(
    participant_id: str, session: str, block: str, dataset: str = None
):
    participants_path = DATA_PATH / dataset if dataset else PARTICIPANTS_PATH
    block_msg = f", Block {block}"
    st.info(f"Loading data for P{participant_id}, Session {session}{block_msg}...")

    p_partition = f"participant_id={participant_id}"
    s_partition = f"session={session}"
    b_partition = f"block={block}"

    p_partition_path = participants_path / p_partition / s_partition / b_partition / "*"

    print(f"Loading data from: {p_partition_path}")
    df = pl.read_parquet(p_partition_path)
    return df


@st.cache_data
def load_participant_session_data(
    participant_id: str, session: str, dataset: str = None
):
    participants_path = DATA_PATH / dataset if dataset else PARTICIPANTS_PATH
    st.info(f"Loading session data for P{participant_id}, Session {session}...")

    p_partition = f"participant_id={participant_id}"
    s_partition = f"session={session}"

    # Load all blocks in the session
    session_path = participants_path / p_partition / s_partition / "*" / "*"

    print(f"Loading session data from: {session_path}")
    df = pl.read_parquet(session_path)
    return df
