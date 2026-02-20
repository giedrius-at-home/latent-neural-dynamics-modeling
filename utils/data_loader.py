import streamlit as st
import polars as pl
import re
from pathlib import Path
from collections import defaultdict

from utils.file_handling import get_child_subchilds_tuples
from utils.logger import setup_logger

logger = setup_logger("dashboard_logs", name=__name__)

DATA_PATH = Path("resampled_recordings")
PARTICIPANTS_PATH = None


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
    if participants_path is None:
        logger.warning("PARTICIPANTS_PATH is None and no dataset provided.")
        return {}

    logger.info(f"PARTICIPANTS_PATH: {participants_path}")
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


def get_all_participants(dataset: str = None) -> list[str]:
    """Returns a sorted list of all participant IDs in the dataset."""
    participants_path = DATA_PATH / dataset if dataset else PARTICIPANTS_PATH
    if participants_path is None or not participants_path.exists():
        return []

    # Assuming directory structure is participant_id=PDI_X/session=X/block=X
    # We can just list the top level directories
    p_dirs = [
        d
        for d in participants_path.iterdir()
        if d.is_dir() and d.name.startswith("participant_id=")
    ]
    p_ids = [d.name.split("=")[1] for d in p_dirs]
    return sorted(p_ids, key=natural_sort_key)


@st.cache_data
def load_participant_block_data(
    participant_id: str,
    session: str,
    block: str,
    dataset: str = None,
    columns: list[str] = None,
):
    participants_path = DATA_PATH / dataset if dataset else PARTICIPANTS_PATH
    if participants_path is None:
        raise ValueError("PARTICIPANTS_PATH is None and no dataset provided.")

    block_msg = f", Block {block}"
    st.info(f"Loading data for P{participant_id}, Session {session}{block_msg}...")

    p_partition = f"participant_id={participant_id}"
    s_partition = f"session={session}"
    b_partition = f"block={block}"

    p_partition_path = participants_path / p_partition / s_partition / b_partition / "*"

    logger.info(f"Loading data from: {p_partition_path}")
    try:
        df = pl.read_parquet(p_partition_path, columns=columns)
    except Exception as e:
        logger.warning(f"Failed to load block data with pl.read_parquet: {e}")
        logger.info("Attempting to load files individually...")
        df = _load_multiple_parquet_files(str(p_partition_path), columns=columns)
    return df


@st.cache_data
def load_participant_session_data(
    participant_id: str, session: str, dataset: str = None, columns: list[str] = None
):
    participants_path = DATA_PATH / dataset if dataset else PARTICIPANTS_PATH
    if participants_path is None:
        raise ValueError("PARTICIPANTS_PATH is None and no dataset provided.")

    p_partition = f"participant_id={participant_id}"
    s_partition = f"session={session}"

    session_path = participants_path / p_partition / s_partition / "*" / "*"

    try:
        df = pl.read_parquet(session_path, columns=columns, rechunk=False)
    except Exception as e:
        logger.warning(f"Failed to load session data with rechunk=False: {e}")
        logger.info("Attempting to load files individually...")
        df = _load_multiple_parquet_files(str(session_path), columns=columns)

    return df


@st.cache_data
def load_participant_data(
    participant_id: str, dataset: str = None, columns: list[str] = None
):
    participants_path = DATA_PATH / dataset if dataset else PARTICIPANTS_PATH
    if participants_path is None:
        raise ValueError("PARTICIPANTS_PATH is None and no dataset provided.")

    p_partition = f"participant_id={participant_id}"

    participant_path = participants_path / p_partition / "*" / "*" / "*"

    try:
        df = pl.read_parquet(participant_path, columns=columns, rechunk=False)
    except Exception as e:
        logger.warning(f"Failed to load participant data with rechunk=False: {e}")
        logger.info("Attempting to load files individually...")
        df = _load_multiple_parquet_files(str(participant_path), columns=columns)

    return df


def _load_multiple_parquet_files(
    path_glob: str, columns: list[str] = None
) -> pl.DataFrame:
    from glob import glob

    parquet_files = glob(path_glob)

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at {path_glob}")

    lazy_dfs = []
    for file in parquet_files:
        try:
            # Using scan_parquet is much more memory efficient
            # We select columns immediately to reduce memory usage during scanning
            lf = pl.scan_parquet(file)
            if columns:
                # Filter columns to only those present in the file
                available_cols = [c for c in columns if c in lf.columns]
                # Even if available_cols is empty, we select them to ensure we don't load other columns
                lf = lf.select(available_cols)
            lazy_dfs.append(lf)
        except Exception as file_error:
            logger.warning(f"Skipping file {file}: {file_error}")
            continue

    if not lazy_dfs:
        raise ValueError("No valid parquet files could be loaded")

    # Diagonal concat handles schema mismatches and type unification (like Null vs Float32)
    return pl.concat(lazy_dfs, how="diagonal").collect()
