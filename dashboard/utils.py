import polars as pl
import numpy as np
from utils.data_loader import natural_sort_key


def get_channel_lists(
    block_data: pl.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
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
    motion_channels = sorted(
        [
            col
            for col in block_data.columns
            if col.lower().startswith("tracing_")
            or col in ["x", "y", "motion_time"]
            or "velocity" in col.lower()
            or "acceleration" in col.lower()
            or "jerk" in col.lower()
        ],
        key=natural_sort_key,
    )
    return lfp_channels, ecog_channels, motion_channels


def discover_channels_from_data(
    participant_id: str, dataset: str, session: str = None
) -> tuple[list[str], list[str]]:
    """
    Discover LFP and ECoG channels from participant or session data by reading schema.
    If session is None, discovers from participant level, otherwise from session level.
    Returns (lfp_channels, ecog_channels).
    """
    from pathlib import Path
    from utils.data_loader import DATA_PATH

    try:
        # Build path based on whether session is provided
        p_partition = f"participant_id={participant_id}"
        if session is not None:
            s_partition = f"session={session}"
            data_path = DATA_PATH / dataset / p_partition / s_partition
        else:
            data_path = DATA_PATH / dataset / p_partition

        if not data_path.exists():
            return [], []

        # Find first parquet file to read schema
        first_file = next(data_path.rglob("*.parquet"), None)
        if not first_file:
            return [], []

        # Read schema only (much faster than loading data)
        schema = pl.read_parquet_schema(first_file)
        all_cols = set(schema.keys())

        lfp_channels = sorted(
            [
                col
                for col in all_cols
                if col.lower().startswith("lfp")
                and ("psd" not in col and "epochs" not in col)
            ],
            key=natural_sort_key,
        )
        ecog_channels = sorted(
            [
                col
                for col in all_cols
                if col.lower().startswith("ecog")
                and ("psd" not in col and "epochs" not in col)
            ],
            key=natural_sort_key,
        )

        return lfp_channels, ecog_channels
    except Exception:
        return [], []


# Convenience aliases for backward compatibility
def discover_channels_from_participant_data(
    participant_id: str, dataset: str
) -> tuple[list[str], list[str]]:
    """Discover channels from participant data (calls discover_channels_from_data with session=None)."""
    return discover_channels_from_data(participant_id, dataset, session=None)


def discover_channels_from_session_data(
    participant_id: str, session: str, dataset: str
) -> tuple[list[str], list[str]]:
    """Discover channels from session data (calls discover_channels_from_data with session)."""
    return discover_channels_from_data(participant_id, dataset, session=session)


def get_trial_metadata(trial_data: pl.DataFrame, trial_idx: int = 0) -> dict:
    if trial_data.is_empty():
        return {}

    meta = {}
    target_cols = ["participant_id", "session", "block", "trial", "stim"]

    for col in target_cols:
        if col in trial_data.columns:
            meta[col] = trial_data[col][trial_idx]

    return meta


def compute_per_trial_psd_means(df: pl.DataFrame, channel: str) -> list:
    means = []
    if df.is_empty():
        return means

    col_name = f"{channel}_psd_values"
    if col_name not in df.columns:
        return means

    for trial_psds in df[col_name].to_list():
        arr = np.array(trial_psds)
        if arr.ndim == 1:
            means.append(arr)
        elif arr.ndim >= 2 and arr.size > 0:
            means.append(arr.mean(axis=0))
    return means
