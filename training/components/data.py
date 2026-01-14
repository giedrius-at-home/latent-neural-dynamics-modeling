import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Union
from utils.config import Config
import torch
from torch.utils.data import Dataset
from utils.logger import get_logger


class TrialDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        data_params: Config,
        split: str = "train",
    ):
        self.parquet_path = Path(parquet_path)
        self.data_params = data_params
        self.split = split

        self.df = pl.read_parquet(self.parquet_path)

        self.neural_input = self.data_params.channels.neural_input
        self.behavioral_input = getattr(
            self.data_params.channels, "behavioral_input", None
        )
        self.output_channels = self.data_params.channels.output
        self.output_type = getattr(
            self.data_params.channels, "output_type", "behavioral"
        )

        logger = get_logger()
        logger.info(
            f"Loaded split='{self.split}' dataset from {self.parquet_path} with {len(self.df)} trials; "
            f"neural_input={self.neural_input}, behavioral_input={self.behavioral_input}, "
            f"output={self.output_channels}, output_type={self.output_type}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
        input_lag = self.data_params.channels.input_lag
        if input_lag is not None:
            input_channel_names = self.neural_input + [
                f"{ch}_lag{input_lag}" for ch in self.neural_input
            ]
        else:
            input_channel_names = self.neural_input

        row = self.df[idx]
        Y = self._extract_channels(row, input_channel_names)
        if input_lag is not None:
            Y = self._add_lagged_channels(Y)

        jitter = float(getattr(self.data_params, "jitter", None))
        if jitter is not None and self.split == "train":
            noise = np.random.normal(0, jitter, Y.shape)
            Y = Y + noise

        Z = self._extract_channels(row, self.output_channels)

        time_vec = row["time"][0]
        chunk_margin = row["chunk_margin"][0]
        chunk_margin_ts = int(
            np.round(chunk_margin * self.data_params.sampling_frequency)
        )
        margined_duration = row["margined_duration"][0]
        stim = row["stim"][0]

        offset = row["offset"][0]

        metadata = {
            "participant_id": row["participant_id"][0],
            "session": row["session"][0],
            "block": row["block"][0],
            "trial": row["trial"][0],
            "trial_length": Y.shape[0],
            "time": np.array(time_vec) if time_vec is not None else None,
            "offset": float(offset) if offset is not None else None,
            "chunk_margin": chunk_margin,
            "margined_duration": margined_duration,
            "stim": stim,
            "input_channels": input_channel_names,
            "original_input_channels": self.neural_input,
            "behavioral_input_channels": self.behavioral_input,
            "output_channels": self.output_channels,
            "sampling_frequency": self.data_params.sampling_frequency,
            "chunk_margin_ts": chunk_margin_ts,
            "input_lag": input_lag if input_lag else 0,
        }

        return Y, Z, metadata

    def _add_lagged_channels(self, Y: np.ndarray) -> np.ndarray:
        input_lag = getattr(self.data_params.channels, "input_lag", None)

        if input_lag is None or input_lag == 0:
            return Y

        n_samples, n_channels = Y.shape
        lagged_channels = []

        for ch_idx in range(n_channels):
            channel_data = Y[:, ch_idx : ch_idx + 1]
            lagged_data = np.roll(channel_data, shift=input_lag, axis=0)
            lagged_data[:input_lag] = 0
            lagged_channels.append(lagged_data)

        Y_with_lags = np.concatenate([Y] + lagged_channels, axis=1)

        return Y_with_lags

    def _extract_channels(
        self, row: pl.DataFrame, channel_list: List[str]
    ) -> np.ndarray:
        channel_data = []

        for channel_prefix in channel_list:
            matching_cols = [col for col in row.columns if col == f"{channel_prefix}"]

            if not matching_cols:
                raise ValueError(
                    f"No columns found for channel prefix: {channel_prefix}"
                )

            for col in matching_cols:
                trial = row[col][0]
                channel_data.append(np.array(trial))

        if len(channel_data) == 1:
            result = channel_data[0].reshape(-1, 1)
        else:
            result = np.column_stack(channel_data)

        if np.any(np.isnan(result)):
            df = pd.DataFrame(result)
            df = df.interpolate(method="linear", limit_direction="both", axis=0)
            df = df.ffill().bfill()
            result = df.values

        return result

    def get_all_data(self) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        Y_list = []
        Z_list = []
        meta_list = []
        for idx in range(len(self)):
            Y, Z, meta = self[idx]
            Y_list.append(Y)
            meta_list.append(meta)
            if Z is not None:
                Z_list.append(Z)
        return Y_list, Z_list, meta_list


class TrialDataLoader:

    def __init__(
        self,
        dataset: TrialDataset,
        batch_size: int = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataset))

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            Y_batch = [item[0] for item in batch]
            Z_batch = [item[1] for item in batch] if batch[0][1] is not None else None
            metadata_batch = [item[2] for item in batch]

            yield Y_batch, Z_batch, metadata_batch

    def get_full_dataset(self) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        return self.dataset.get_all_data()


def create_dataloaders(
    data_params: Config,
    results_config: Config,
) -> Tuple[TrialDataLoader, TrialDataLoader, TrialDataLoader]:

    split_dir = Path(results_config.save_dir) / "split"
    logger = get_logger()
    logger.info(f"Creating dataloaders from split directory: {split_dir}")

    train_dataset = TrialDataset(
        split_dir / "train.parquet", data_params, split="train"
    )

    val_dataset = TrialDataset(split_dir / "val.parquet", data_params, split="val")

    test_dataset = TrialDataset(split_dir / "test.parquet", data_params, split="test")

    logger.info(
        f"Loaded datasets: train={len(train_dataset)} trials, val={len(val_dataset)} trials, test={len(test_dataset)} trials"
    )

    train_loader = TrialDataLoader(
        train_dataset,
        batch_size=data_params.batch_size,
        shuffle=True,
    )

    val_loader = TrialDataLoader(
        val_dataset,
        batch_size=data_params.batch_size,
        shuffle=False,
    )

    test_loader = TrialDataLoader(
        test_dataset,
        batch_size=data_params.batch_size,
        shuffle=False,
    )

    logger.info(
        f"Constructed dataloaders with batch_size={data_params.batch_size}. "
        f"Train steps/epoch≈{len(train_loader)}, Val steps≈{len(val_loader)}, Test steps≈{len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
