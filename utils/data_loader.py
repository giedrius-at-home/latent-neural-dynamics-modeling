import streamlit as st
import polars as pl
import re
from pathlib import Path
from collections import defaultdict

from utils.file_handling import get_child_subchilds_tuples
from utils.logger import setup_logger

logger = setup_logger("dashboard_logs", name=__name__)

DATA_PATH = Path("resampled_recordings")
PARTICIPANTS_PATH = DATA_PATH / "participants_at_60Hz_scaled_1e6"


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


@st.cache_data
def get_participant_sessions():
    logger.info(f"PARTICIPANTS_PATH: {PARTICIPANTS_PATH}")
    logger.info(f"PARTICIPANTS_PATH exists: {PARTICIPANTS_PATH.exists()}")
    if not PARTICIPANTS_PATH.exists():
        st.error(f"Data directory not found at: {PARTICIPANTS_PATH}")
        return {}

    session_tuples = get_child_subchilds_tuples(PARTICIPANTS_PATH)
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
def load_participant_block_data(participant_id: str, session: str, block: str):
    block_msg = f", Block {block}"
    st.info(f"Loading data for P{participant_id}, Session {session}{block_msg}...")

    p_partition = f"participant_id={participant_id}"
    s_partition = f"session={session}"
    b_partition = f"block={block}"

    p_partition_path = PARTICIPANTS_PATH / p_partition / s_partition / b_partition / "*"

    print(f"Loading data from: {p_partition_path}")
    df = pl.read_parquet(p_partition_path)
    return df


@st.cache_data
def load_participant_session_data(participant_id: str, session: str):
    """Load all blocks for a given participant and session."""
    st.info(f"Loading session data for P{participant_id}, Session {session}...")

    p_partition = f"participant_id={participant_id}"
    s_partition = f"session={session}"

    # Load all blocks in the session
    session_path = PARTICIPANTS_PATH / p_partition / s_partition / "*" / "*"

    print(f"Loading session data from: {session_path}")
    df = pl.read_parquet(session_path)
    return df
