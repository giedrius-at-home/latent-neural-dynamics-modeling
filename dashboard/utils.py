import polars as pl
import numpy as np
from utils.data_loader import natural_sort_key


def get_channel_lists(block_data: pl.DataFrame) -> tuple[list[str], list[str]]:
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
    return lfp_channels, ecog_channels


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
