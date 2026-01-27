import polars as pl

from utils.config import Config, get_config
from utils.frameworks import PSIDFramework
from training.components.data import create_dataloaders
from utils.logger import get_logger
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pickle
from utils.stats import pearson_r_per_channel
import h5py
import json

from utils.miscellaneous import length, flatten


class Tester:

    def __init__(self, config: Config, run_timestamp: Optional[str] = None):
        self.config = config
        self.model_params = config.model
        self.data_params = config.data
        self.results_config = config.results
        self.logger = get_logger()
        self.framework = None
        self.run_timestamp = run_timestamp

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    @classmethod
    def from_config_file(cls, config_path: str, run_timestamp: Optional[str] = None):
        cfg = get_config(config_path)
        return cls(cfg, run_timestamp=run_timestamp)

    def _init_framework(self):
        framework_type = str(self.model_params.name).split("_")[0]
        if framework_type == "psid":
            self.framework = PSIDFramework(self.config)
        elif framework_type == "dpad":
            from utils.frameworks import DPADFramework

            self.framework = DPADFramework(self.config)
        else:
            raise ValueError(
                f"Unknown or unsupported framework for testing: {framework_type}"
            )

    def _load_dataloaders(self):
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.data_params, self.results_config
        )

    def _load_model_for_run(self):
        results_dir = Path(self.results_config.save_dir)
        model_path = results_dir / f"model_{self.run_timestamp}.pkl"

        metadata_path = results_dir / f"model_{self.run_timestamp}_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.logger.info(f"Loading DPAD model with metadata: {metadata}")

        with open(model_path, "rb") as f:
            idSys = pickle.load(f)

        self._init_framework()
        self.framework.model = self.framework._initialize_model()
        self.framework.model.idSys = idSys

        if metadata_path.exists() and hasattr(idSys, "restoreModels"):
            self.logger.info("Restoring DPAD TensorFlow models from saved weights...")
            idSys.restoreModels()
            self.logger.info(f"Loaded DPAD model from {model_path}")
        else:
            self.logger.info(f"Loaded PSID model from {model_path}")

    @staticmethod
    def _get_metrics(
        Y_true: List[np.ndarray],
        Z_true: Optional[List[np.ndarray]],
        Yp: Optional[List[np.ndarray]],
        Zp: Optional[List[np.ndarray]],
        Xp: Optional[List[np.ndarray]],
        meta: Dict[str, List[Any]],
    ) -> Dict[str, Any]:

        # Ensure all predicted arrays match the length of Y_true (some models might return unsliced)
        if Yp is not None:
            Yp = [
                yp[: len(yt)] if yp is not None else None for yt, yp in zip(Y_true, Yp)
            ]
        if Zp is not None:
            Zp = [
                zp[: len(yt)] if zp is not None else None for yt, zp in zip(Y_true, Zp)
            ]
        if Xp is not None:
            Xp = [
                xp[: len(yt)] if xp is not None else None for yt, xp in zip(Y_true, Xp)
            ]

        pearson_per_trial, pearson_overall_mean = pearson_r_per_channel(Y_true, Yp)

        pearson_trial_means = []
        for r_list in pearson_per_trial:
            valid = [r for r in r_list if not (r is None or np.isnan(r))]
            pearson_trial_means.append(np.mean(valid) if len(valid) > 0 else np.nan)

        pearson_per_trial_Z = None
        pearson_overall_mean_Z = np.nan
        pearson_trial_means_Z = None

        if Z_true is not None and Zp is not None:
            Z_true_filtered = [z for z in Z_true if z is not None]
            Zp_filtered = [zp for zp in Zp if zp is not None]

            if len(Z_true_filtered) > 0 and len(Zp_filtered) > 0:
                pearson_per_trial_Z, pearson_overall_mean_Z = pearson_r_per_channel(
                    Z_true_filtered, Zp_filtered
                )

                pearson_trial_means_Z = []
                for r_list in pearson_per_trial_Z:
                    valid = [r for r in r_list if not (r is None or np.isnan(r))]
                    pearson_trial_means_Z.append(
                        np.mean(valid) if len(valid) > 0 else np.nan
                    )

        return {
            "Y": [y.tolist() for y in Y_true],
            "Z": (
                [z.tolist() if z is not None else None for z in Z_true]
                if Z_true is not None
                else None
            ),
            "Yp": [Yp_.tolist() for Yp_ in Yp] if Yp is not None else None,
            "Zp": (
                [Zp_.tolist() if Zp_ is not None else None for Zp_ in Zp]
                if Zp is not None
                else None
            ),
            "Xp": [Xp_.tolist() for Xp_ in Xp] if Xp is not None else None,
            "pearson_per_channel": pearson_per_trial,
            "pearson_mean": pearson_trial_means,
            "pearson_overall_mean": pearson_overall_mean,
            "pearson_per_channel_Z": pearson_per_trial_Z,
            "pearson_mean_Z": pearson_trial_means_Z,
            "pearson_overall_mean_Z": pearson_overall_mean_Z,
            "time": [t.tolist() for t in meta.get("time", [])],
            "offset": meta.get("offset", []),
            "chunk_margin": meta.get("chunk_margin", []),
            "margined_duration": meta.get("margined_duration", []),
            "stim": meta.get("stim", []),
            "participant_id": meta.get("participant_id", []),
            "session": meta.get("session", []),
            "block": meta.get("block", []),
            "trial": meta.get("trial", []),
            "input_channels": (
                meta.get("original_input_channels", [[]])[0]
                if meta.get("original_input_channels")
                else (
                    meta.get("input_channels", [[]])[0]
                    if meta.get("input_channels")
                    else []
                )
            ),
            "output_channels": (
                meta.get("output_channels", [[]])[0]
                if meta.get("output_channels")
                else []
            ),
        }

    def _slice_data(self, Y_list_margined, Z_list_margined, meta_list):
        sfreq = self.data_params.sampling_frequency
        neural_shift_ms = getattr(self.data_params, "neural_shift", None) or 0
        behavioral_shift_ms = getattr(self.data_params, "behavioral_shift", None) or 0

        neural_shift_samp = int(round(neural_shift_ms * sfreq / 1000))
        behavioral_shift_samp = int(round(behavioral_shift_ms * sfreq / 1000))

        _Y, _Z, _meta = [], [], []
        Z_list_margined = (
            [None] * len(Y_list_margined)
            if Z_list_margined is None
            else Z_list_margined
        )

        for Y, Z, meta in zip(Y_list_margined, Z_list_margined, meta_list):
            n_samples = Y.shape[0]
            chunk_margin_ts = meta.get("chunk_margin_ts", 0)

            base_start = chunk_margin_ts
            base_end = n_samples - chunk_margin_ts

            # Standard valid window logic
            y_start = base_start + max(0, neural_shift_samp)
            y_end = base_end + min(0, neural_shift_samp)
            z_start = base_start + max(0, behavioral_shift_samp)
            z_end = base_end + min(0, behavioral_shift_samp)

            valid_start = max(y_start, z_start)
            valid_end = min(y_end, z_end)

            if valid_end <= valid_start:
                continue

            Y_sliced = Y[
                valid_start - neural_shift_samp : valid_end - neural_shift_samp
            ]

            if Z is not None:
                if Z.shape[0] == n_samples:
                    Z_sliced = Z[
                        valid_start
                        - behavioral_shift_samp : valid_end
                        - behavioral_shift_samp
                    ]
                else:
                    z_len = valid_end - valid_start
                    z_offset = valid_start - chunk_margin_ts - behavioral_shift_samp
                    Z_sliced = Z[z_offset : z_offset + z_len]
            else:
                Z_sliced = None

            # Update meta
            new_meta = meta.copy()
            if "time" in new_meta and new_meta["time"] is not None:
                new_meta["time"] = new_meta["time"][valid_start:valid_end]

            new_meta["chunk_margin"] = 0.0
            new_meta["chunk_margin_ts"] = 0
            new_meta["margined_duration"] = float(len(Y_sliced)) / sfreq

            _Y.append(Y_sliced)
            _Z.append(Z_sliced)
            _meta.append(new_meta)

        _Z = None if all([_z is None for _z in _Z]) else _Z
        self.logger.info(
            f"Sliced data: Y={length(_Y)}, Z={length(_Z)}, meta={length(_meta)}"
        )
        return _Y, _Z, _meta

    def _remove_lagged_channels(self, Y_list, Yp_list, meta_list):
        input_lag = meta_list[0].get("input_lag", 0) if meta_list else 0

        self.logger.info(f"_remove_lagged_channels called: input_lag={input_lag}")

        if input_lag == 0:
            self.logger.info("No input lag, returning original data")
            return Y_list, Yp_list

        original_channels = meta_list[0].get("original_input_channels", [])
        n_original = len(original_channels)

        self.logger.info(f"Original channels: {original_channels} (n={n_original})")

        if n_original == 0:
            self.logger.warning(
                "No original_input_channels found in metadata, cannot remove lagged channels"
            )
            return Y_list, Yp_list

        if len(Y_list) > 0:
            self.logger.info(
                f"Before slicing - Y[0] shape: {Y_list[0].shape}, Yp[0] shape: {Yp_list[0].shape}"
            )

        Y_list_sliced = []
        Yp_list_sliced = []

        for i, (Y, Yp) in enumerate(zip(Y_list, Yp_list)):
            if Y.shape[1] < n_original:
                self.logger.error(
                    f"Trial {i}: Y has {Y.shape[1]} channels but trying to keep {n_original} original channels!"
                )
                Y_sliced = Y
                Yp_sliced = Yp
            else:
                Y_sliced = Y[:, :n_original]
                Yp_sliced = Yp[:, :n_original]

            Y_list_sliced.append(Y_sliced)
            Yp_list_sliced.append(Yp_sliced)

        if len(Y_list_sliced) > 0:
            self.logger.info(
                f"After slicing - Y[0] shape: {Y_list_sliced[0].shape}, Yp[0] shape: {Yp_list_sliced[0].shape}"
            )
            self.logger.info(
                f"Removed lagged channels: {Y_list[0].shape[1]} -> {Y_list_sliced[0].shape[1]} channels"
            )

        return Y_list_sliced, Yp_list_sliced

    def _remove_lagged_channels_from_forecast(self, f_res, meta_list):
        input_lag = meta_list[0].get("input_lag", 0) if meta_list else 0

        if input_lag == 0:
            self.logger.info(
                "No input lag in forecast, returning original forecast results"
            )
            return f_res

        original_channels = meta_list[0].get("original_input_channels", [])
        n_original = len(original_channels)

        if n_original == 0:
            self.logger.warning(
                "No original_input_channels found, cannot slice forecast results"
            )
            return f_res

        self.logger.info(
            f"Slicing forecast results to keep {n_original} original channels"
        )

        for key in ["Y_future_true", "Y_future_pred", "Y_concat_for_plot"]:
            if key in f_res and f_res[key]:
                sliced = []
                for arr in f_res[key]:
                    if arr is not None:
                        arr_np = np.array(arr)
                        if arr_np.ndim == 2 and arr_np.shape[1] > n_original:
                            arr_sliced = arr_np[:, :n_original]
                            sliced.append(arr_sliced.tolist())
                        else:
                            sliced.append(arr)
                    else:
                        sliced.append(None)
                f_res[key] = sliced

        return f_res

    def run_predictions(self):

        self._load_dataloaders()
        self._load_model_for_run()

        self.results = {}

        for split_name, loader in (
            ("train", self.train_loader),
            ("val", self.val_loader),
            ("test", self.test_loader),
        ):
            Y_list, _z, meta_list = loader.get_full_dataset()
            Y_list, Z_list, meta_list = self._slice_data(Y_list, _z, meta_list)

            Zp, Yp, Xp = self.framework._predict(Y_list)

            chunk_margin = meta_list[0].get("chunk_margin")
            f_res = self.framework.model.validate_forecast(
                Y_list, Z_list=Z_list, margin=chunk_margin
            )

            Y_list, Yp = self._remove_lagged_channels(Y_list, Yp, meta_list)

            f_res = self._remove_lagged_channels_from_forecast(f_res, meta_list)

            meta = {k: [d.get(k) for d in meta_list] for k in meta_list[0]}
            split_results = self._get_metrics(Y_list, Z_list, Yp, Zp, Xp, meta)

            split_results = split_results | f_res

            self.results[split_name] = split_results

    def run_predictions_selective(self, splits: List[str]):
        self._load_dataloaders()
        self._load_model_for_run()

        self.results = {}

        loader_map = {
            "train": self.train_loader,
            "val": self.val_loader,
            "test": self.test_loader,
        }

        for split_name in splits:
            if split_name not in loader_map:
                self.logger.warning(f"Unknown split '{split_name}', skipping...")
                continue

            loader = loader_map[split_name]
            Y_list, _z, meta_list = loader.get_full_dataset()
            Y_list, Z_list, meta_list = self._slice_data(Y_list, _z, meta_list)

            Zp, Yp, Xp = self.framework._predict(Y_list)

            chunk_margin = meta_list[0].get("chunk_margin")
            f_res = self.framework.model.validate_forecast(
                Y_list, Z_list=Z_list, margin=chunk_margin
            )

            Y_list, Yp = self._remove_lagged_channels(Y_list, Yp, meta_list)

            f_res = self._remove_lagged_channels_from_forecast(f_res, meta_list)

            meta = {k: [d.get(k) for d in meta_list] for k in meta_list[0]}
            split_results = self._get_metrics(Y_list, Z_list, Yp, Zp, Xp, meta)

            split_results = split_results | f_res

            self.results[split_name] = split_results

    def save_results(self):
        results_dir = Path(self.results_config.save_dir)
        for k in self.results:
            results_path = (
                results_dir / k / f"test_results_{self.run_timestamp}.parquet"
            )
            results_ = {}
            n_rows = len(self.results[k]["participant_id"])
            for col_name, col_value in self.results[k].items():
                if isinstance(col_value, list) and len(col_value) == n_rows:
                    if all(v is None for v in col_value):
                        results_[col_name] = pl.Series(
                            name=col_name, values=col_value, dtype=pl.Float32
                        )
                    else:
                        results_[col_name] = pl.Series(name=col_name, values=col_value)
                elif isinstance(col_value, (np.ndarray, dict)):
                    self.logger.warning(
                        f"Skipping column '{col_name}' in split '{k}': "
                        f"value is a {type(col_value)}, which is not supported for row-expansion."
                    )
                    continue
                else:
                    if col_value is None:
                        results_[col_name] = pl.Series(
                            name=col_name, values=[col_value] * n_rows, dtype=pl.Float32
                        )
                    elif isinstance(col_value, list):
                        if len(col_value) <= 1:
                            results_[col_name] = pl.Series(
                                name=col_name, values=[col_value[0]] * n_rows
                            )
                    else:
                        results_[col_name] = pl.Series(
                            name=col_name, values=[col_value] * n_rows
                        )
            results_df = pl.from_dict(results_)
            results_df.write_parquet(
                results_path,
                partition_by=["participant_id", "session", "block", "trial"],
            )

    def compute_and_save_stats(self):

        def create_dataset(group, name, data):
            if data is None:
                group.create_dataset(name, data=h5py.Empty("f"))
            else:
                group.create_dataset(name, data=data)

        results_dir = Path(self.results_config.save_dir)
        stats_path = results_dir / f"test_stats_{self.run_timestamp}.hdf5"
        with h5py.File(stats_path, "w") as f:
            f.attrs["model_name"] = self.model_params.name
            f.attrs["nx"] = self.model_params.nx
            f.attrs["n1"] = self.model_params.n1
            f.attrs["i"] = self.model_params.i
            f.attrs["run_timestamp"] = self.run_timestamp
            A_Eigs = np.linalg.eig(self.framework.model.idSys.A)[0]
            isStable = np.max(np.abs(A_Eigs)) < 1
            f.attrs["is_stable"] = isStable

            f.create_dataset("A", data=self.framework.model.idSys.A)
            f.create_dataset("Cy", data=self.framework.model.idSys.C)
            f.create_dataset("Cz", data=self.framework.model.idSys.Cz)
            f.create_dataset("Q", data=self.framework.model.idSys.Q)
            f.create_dataset("R", data=self.framework.model.idSys.R)
            f.create_dataset("S", data=self.framework.model.idSys.S)

            prep_group = f.create_group("preprocessing")

            create_dataset(
                prep_group, "Y_mean", self.framework.model.idSys.YPrepModel.mean
            )
            create_dataset(
                prep_group, "Y_std", self.framework.model.idSys.YPrepModel.std
            )
            create_dataset(
                prep_group, "Z_mean", self.framework.model.idSys.ZPrepModel.mean
            )
            create_dataset(
                prep_group, "Z_std", self.framework.model.idSys.ZPrepModel.std
            )

            stats_group = f.create_group("analysis_stats")

            A_11 = self.framework.model.idSys.A[
                0 : self.model_params.n1, 0 : self.model_params.n1
            ]
            stats_group.create_dataset("eigvals_relevant", data=np.linalg.eigvals(A_11))

            A_22 = self.framework.model.idSys.A[
                self.model_params.n1 : self.model_params.nx,
                self.model_params.n1 : self.model_params.nx,
            ]
            stats_group.create_dataset(
                "eigvals_irrelevant", data=np.linalg.eigvals(A_22)
            )

            stats_group.create_dataset(
                "z_readout_norm",
                data=np.linalg.norm(
                    self.framework.model.idSys.Cz[:, 0 : self.model_params.n1], axis=0
                ),
            )
            stats_group.create_dataset(
                "y_readout_norm",
                data=np.linalg.norm(self.framework.model.idSys.C, axis=0),
            )
