import json
import re
import pickle
import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path

from utils.config import Config
from utils.logger import get_logger
from utils.miscellaneous import length
from utils.split import create_splits
from utils.stats import pearson_r_per_channel
from training.components.data import create_dataloaders
from utils.frameworks import PSIDFramework, DPADFramework, VARMAOLSFramework


class Trainer:

    def __init__(self, config: Config, model_dbs: str):
        self.config = config
        self.model_params = config.framework.params
        self.data_params = config.data
        self.model_dbs = model_dbs
        self.logger = get_logger()

        self.variant_name = f"{config.experiment.name}_dbs_{model_dbs}"
        self.save_dir = Path(f"results/{config.framework.name}/{self.variant_name}")
        self.split_dir = self.save_dir / "split"
        self.framework_type = config.framework.name

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.run_timestamp = None
        self.framework = None

    def split_data(self, reuse_splits: bool = False):
        self.logger.info("Starting data split...")
        self.logger.info(
            f"Data params: participant={self.data_params.participant}, session={self.data_params.session}, "
            f"Y={self.data_params.Y}, Z={self.data_params.Z}"
        )

        existing_splits = (
            (self.split_dir / "train.parquet").exists()
            and (self.split_dir / "val.parquet").exists()
            and (self.split_dir / "test.parquet").exists()
        )

        if reuse_splits and existing_splits:
            self.logger.info(f"Reusing existing splits from {self.split_dir}")
            if self.model_dbs != "both":
                for split_name in ("train", "val", "test"):
                    split_path = self.split_dir / f"{split_name}.parquet"
                    df_split = pl.read_parquet(split_path)
                    if "stim" in df_split.columns:
                        before = len(df_split)
                        df_split = df_split.filter(pl.col("stim") == self.model_dbs)
                        after = len(df_split)
                        if after < before:
                            df_split.write_parquet(split_path)
                            self.logger.info(
                                f"Filtered {split_name} split to dbs={self.model_dbs}: "
                                f"{before} -> {after} trials"
                            )
        else:
            session_path = (
                Path(self.data_params.root)
                / f"participant_id={self.data_params.participant}"
                / f"session={self.data_params.session}"
            )
            base_cols = list(set(self.data_params.Y) | set(self.data_params.Z))

            probe_path = None
            for bf in sorted(session_path.glob("block=*/0.parquet")):
                probe_path = bf
                break
            if probe_path is not None:
                probe_all = pl.read_parquet(probe_path, n_rows=1)
                ecog_rx = re.compile(r"^ECOG_[1-4]_.*_raw$")
                lap_rx = re.compile(r"^LAPLACIAN_14-16_LFP_.*_raw$")
                beh_rx = re.compile(
                    r"^tracing_(velocity_[xy]|acceleration_magnitude|velocity_magnitude)$"
                )
                universal = [
                    c
                    for c in probe_all.columns
                    if ecog_rx.match(c) or lap_rx.match(c) or beh_rx.match(c)
                ]
                base_cols = list(set(base_cols) | set(universal))
            combined_cols = [pl.col(col) for col in base_cols]

            block_files = sorted(list(session_path.glob("block=*/0.parquet")))
            lazy_frames = []

            for bf in block_files:
                try:
                    block_num = int(bf.parent.name.split("=")[1])
                    if (
                        self.data_params.blocks != "all"
                        and block_num not in self.data_params.blocks
                    ):
                        continue

                    probe = pl.read_parquet(bf, columns=self.data_params.Y, n_rows=1)
                    if any(
                        probe[c].dtype == pl.Null
                        or (
                            isinstance(probe[c].dtype, pl.List)
                            and probe[c].list.get(0).dtype == pl.Null
                        )
                        for c in self.data_params.Y
                        if c in probe.columns
                    ):
                        self.logger.warning(f"Skipping empty block {block_num}")
                        continue

                    lf = pl.scan_parquet(bf).with_columns(
                        [
                            pl.lit(self.data_params.participant).alias(
                                "participant_id"
                            ),
                            pl.lit(str(self.data_params.session)).alias("session"),
                            pl.lit(block_num).alias("block"),
                        ]
                    )
                    lazy_frames.append(lf)
                except Exception as e:
                    self.logger.warning(f"Error loading block {bf}: {e}")

            if not lazy_frames:
                raise ValueError(
                    f"No valid data found for {self.data_params.participant}"
                )

            self.logger.info(f"Loading {len(lazy_frames)} blocks...")
            trial = (
                pl.concat(lazy_frames, how="diagonal")
                .select(
                    pl.col("participant_id"),
                    pl.col("session"),
                    pl.col("block"),
                    pl.col("trial"),
                    pl.when(pl.col("time").is_not_null())
                    .then(pl.col("time"))
                    .otherwise(None)
                    .alias("time"),
                    pl.col("chunk_margin"),
                    pl.col("margined_duration"),
                    pl.col("stim"),
                    *combined_cols,
                    pl.col("onset").alias("offset"),
                )
                .collect()
            )

            trial = trial.sort(
                [
                    pl.col("participant_id"),
                    pl.col("session"),
                    pl.col("block"),
                    pl.col("trial"),
                ],
                maintain_order=True,
            )

            if self.data_params.blocks != "all":
                trial = trial.filter(pl.col("block").is_in(self.data_params.blocks))

            if self.model_dbs != "both":
                trial = trial.filter(pl.col("stim") == self.model_dbs)
                self.logger.info(f"Filtered to {self.model_dbs} DBS condition")

            create_splits(trial, self.split_dir, self.data_params.root)

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.data_params, self.split_dir
        )

    def _slice_data(self, Y_list_margined, Z_list_margined, meta_list):
        sfreq = self.data_params.sampling_frequency

        _Y, _Z, _meta = [], [], []
        Z_list_margined = (
            [None] * len(Y_list_margined)
            if Z_list_margined is None
            else Z_list_margined
        )

        for Y, Z, meta in zip(Y_list_margined, Z_list_margined, meta_list):
            n_samples = Y.shape[0]
            chunk_margin_ts = meta.get("chunk_margin_ts", 0)

            valid_start = chunk_margin_ts
            valid_end = n_samples - chunk_margin_ts

            if valid_end <= valid_start:
                self.logger.warning(
                    f"Margin too large for trial: margin={chunk_margin_ts}"
                )
                continue

            Y_sliced = Y[valid_start:valid_end]

            if Z is not None:
                if Z.shape[0] == n_samples:
                    Z_sliced = Z[valid_start:valid_end]
                else:
                    z_len = valid_end - valid_start
                    z_offset = valid_start - chunk_margin_ts
                    Z_sliced = Z[z_offset : z_offset + z_len]
            else:
                Z_sliced = None

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

    def load_model(self, model_timestamp: str):

        model_path = self.save_dir / f"model_{model_timestamp}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.logger.info(f"Loading pre-trained model from {model_path}")

        if self.framework_type == "dpad":
            metadata_path = self.save_dir / f"model_{model_timestamp}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                if metadata.get("nx") != self.model_params.nx:
                    self.logger.warning(
                        f"Model nx={metadata.get('nx')} differs from config nx={self.model_params.nx}"
                    )
                if metadata.get("n1") != self.model_params.n1:
                    self.logger.warning(
                        f"Model n1={metadata.get('n1')} differs from config n1={self.model_params.n1}"
                    )

                self.logger.info(f"Loaded metadata: {metadata}")
            else:
                self.logger.warning(f"Metadata file not found: {metadata_path}")

        self.framework.model = self.framework._initialize_model()
        self.framework.model.load_from_file(str(model_path))
        self.logger.info("Model loaded successfully")

    def _init_framework(self):
        self.logger.info(f"Selected framework: {self.framework_type}")

        if self.framework_type == "psid":
            self.framework = PSIDFramework(self.config)
        elif self.framework_type == "dpad":
            self.framework = DPADFramework(self.config)
        elif self.framework_type == "varma":
            self.framework = VARMAOLSFramework(self.config)
        else:
            raise ValueError(f"Unknown framework type: {self.framework_type}")

    def _prepare_data(self):
        Y_train_m, Z_train_m, meta_train_m = self.train_loader.get_full_dataset()
        Y_val_m, Z_val_m, meta_val_m = self.val_loader.get_full_dataset()

        Y_train, Z_train, meta_train = self._slice_data(
            Y_train_m, Z_train_m, meta_train_m
        )
        Y_val, Z_val, meta_val = self._slice_data(Y_val_m, Z_val_m, meta_val_m)

        return Y_train, Z_train, meta_train, Y_val, Z_val, meta_val

    def _compute_z_correlation(self, Z, Zp):
        if Z is not None and Zp is not None:
            Z_filtered = [z for z in Z if z is not None]
            Zp_filtered = [zp for zp in Zp if zp is not None]
            if len(Z_filtered) > 0 and len(Zp_filtered) > 0:
                r_list, r_mean = pearson_r_per_channel(Z_filtered, Zp_filtered)
                return r_list, r_mean
        return None, None

    def _save_metadata(
        self,
        r_mean_val,
        r_mean_Z_val,
        r_per_channel_Y=None,
        r_per_channel_Z=None,
        Y_features=None,
        Z_features=None,
    ):
        ts = self.run_timestamp
        self.save_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "framework_type": self.framework_type,
            "nx": getattr(self.model_params, "nx", None),
            "n1": getattr(self.model_params, "n1", None),
            "i": getattr(self.model_params, "i", None),
            "r_mean_Y": float(r_mean_val) if r_mean_val is not None else None,
            "r_mean_Z": float(r_mean_Z_val) if r_mean_Z_val is not None else None,
        }
        if r_per_channel_Y is not None and Y_features is not None:
            if len(r_per_channel_Y) > 0 and isinstance(r_per_channel_Y[0], list):
                n_channels = len(Y_features)
                channel_stats = {}
                for ch_idx in range(n_channels):
                    ch_vals = [
                        trial[ch_idx]
                        for trial in r_per_channel_Y
                        if len(trial) > ch_idx
                        and trial[ch_idx] is not None
                        and not np.isnan(trial[ch_idx])
                    ]
                    if ch_vals:
                        channel_stats[Y_features[ch_idx]] = {
                            "mean": float(np.mean(ch_vals)),
                            "std": float(np.std(ch_vals)),
                            "min": float(np.min(ch_vals)),
                            "max": float(np.max(ch_vals)),
                            "n_trials": len(ch_vals),
                        }
                metadata["r_per_channel_Y"] = channel_stats
            else:
                metadata["r_per_channel_Y"] = {
                    ch: {"mean": float(r)}
                    for ch, r in zip(Y_features, r_per_channel_Y)
                    if r is not None and not np.isnan(r)
                }
            metadata["Y_features"] = Y_features

        if r_per_channel_Z is not None and Z_features is not None:
            if len(r_per_channel_Z) > 0 and isinstance(r_per_channel_Z[0], list):
                n_channels = len(Z_features)
                channel_stats = {}
                for ch_idx in range(n_channels):
                    ch_vals = [
                        trial[ch_idx]
                        for trial in r_per_channel_Z
                        if len(trial) > ch_idx
                        and trial[ch_idx] is not None
                        and not np.isnan(trial[ch_idx])
                    ]
                    if ch_vals:
                        channel_stats[Z_features[ch_idx]] = {
                            "mean": float(np.mean(ch_vals)),
                            "std": float(np.std(ch_vals)),
                            "min": float(np.min(ch_vals)),
                            "max": float(np.max(ch_vals)),
                            "n_trials": len(ch_vals),
                        }
                metadata["r_per_channel_Z"] = channel_stats
            else:
                metadata["r_per_channel_Z"] = {
                    ch: {"mean": float(r)}
                    for ch, r in zip(Z_features, r_per_channel_Z)
                    if r is not None and not np.isnan(r)
                }
            metadata["Z_features"] = Z_features

        metadata_path = self.save_dir / f"model_{ts}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        r_z_str = f"{r_mean_Z_val:.4f}" if r_mean_Z_val else "N/A"
        self.logger.info(f"Saved metadata to {metadata_path}")
        self.logger.info(f"Pearson R Y={r_mean_val:.4f}, Pearson R Z={r_z_str}")

    def _write_minimal_metadata(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.save_dir / f"model_{self.run_timestamp}_metadata.json"

        if self.framework_type == "dpad":
            metadata = {
                "framework_type": "dpad",
                "nx": self.model_params.nx,
                "n1": self.model_params.n1,
                "method_code": self.model_params.method_code,
                "epochs": self.model_params.epochs,
            }
        elif self.framework_type == "varma":
            metadata = {
                "framework_type": "varma",
                "p": self.model_params.p,
                "q": self.model_params.q,
                "long_ar_lags": self.model_params.long_ar_lags,
            }
        else:
            metadata = {
                "framework_type": "psid",
                "nx": self.model_params.nx,
                "n1": self.model_params.n1,
                "i": self.model_params.i,
                "A_eigen_constrain": self.model_params.A_eigen_constrain,
            }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Saved minimal metadata to {metadata_path}")

    def train(self, fast: bool = False):
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Training run timestamp: {self.run_timestamp}")

        self._init_framework()
        Y_train, Z_train, meta_train, Y_val, Z_val, meta_val = self._prepare_data()

        model_timestamp = getattr(self.model_params, "model_timestamp", None)
        if model_timestamp:
            self.logger.info(
                f"Loading pre-trained model with timestamp: {model_timestamp}"
            )
            self.load_model(model_timestamp)
        else:
            self.logger.info("Beginning training...")
            self.framework._train(Y_train, Z_train)

        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.save_dir / f"model_{self.run_timestamp}.pkl"

            if self.framework_type == "dpad":
                try:
                    self.framework.model.idSys.discardModels()
                    with open(model_path, "wb") as f:
                        pickle.dump(self.framework.model.idSys, f)
                finally:
                    self.framework.model.idSys.restoreModels()
            elif self.framework_type == "varma":
                with open(model_path, "wb") as f:
                    pickle.dump(self.framework.model, f)
            else:
                with open(model_path, "wb") as f:
                    pickle.dump(self.framework.model.idSys, f)

            self.logger.info(f"Saved model checkpoint to {model_path}")
        except Exception as e:
            self.logger.warning(f"Could not save model checkpoint: {e}")

        if fast:
            self.logger.info(
                "Fast mode enabled: Skipping all post-training predictions and forecasts."
            )
            self._write_minimal_metadata()
            return {}

        Zp_val, Yp_val, Xp_val = self.framework._predict(Y_val, Z_val)

        r_list_val, r_mean_val = pearson_r_per_channel(Y_val, Yp_val)
        r_list_Z_val, r_mean_Z_val = self._compute_z_correlation(Z_val, Zp_val)
        Y_features = meta_val[0].get("Y_features", []) if meta_val else []
        Z_features = meta_val[0].get("Z_features", []) if meta_val else []

        Zp_train, Yp_train, Xp_train = self.framework._predict(Y_train, Z_train)
        r_list_train, r_mean_train = pearson_r_per_channel(Y_train, Yp_train)

        fc = self.config.experiment.forecasts
        m_seconds = fc.default_m
        history_seconds = fc.h_grid[-1]

        train_results = {
            "Y": Y_train,
            "Z": Z_train,
            "Zp": Zp_train,
            "Yp": Yp_train,
            "Xp": Xp_train,
            "pearson_r_per_channel": r_list_train,
            "pearson_r_mean": r_mean_train,
            "Y_features": (meta_train[0].get("Y_features", []) if meta_train else []),
            "Z_features": (meta_train[0].get("Z_features", []) if meta_train else []),
        }

        r_list_Z_train, r_mean_Z_train = self._compute_z_correlation(Z_train, Zp_train)
        if r_list_Z_train is not None:
            train_results["pearson_r_per_channel_Z"] = r_list_Z_train
            train_results["pearson_r_mean_Z"] = r_mean_Z_train

        chunk_margin_train = meta_train[0].get("chunk_margin") if meta_train else 0
        train_forecast = self.framework._evaluate_forecast(
            Y_train,
            Z_list=Z_train,
            margin=chunk_margin_train,
            m_seconds=m_seconds,
            history_seconds=history_seconds,
        )
        train_results.update(train_forecast)

        self.save_results(
            train_results,
            self.train_loader.dataset.df,
            type="train",
            meta_list=meta_train,
        )

        val_results = {
            "Y": Y_val,
            "Z": Z_val,
            "Zp": Zp_val,
            "Yp": Yp_val,
            "Xp": Xp_val,
            "pearson_r_per_channel": r_list_val,
            "pearson_r_mean": r_mean_val,
            "Y_features": Y_features,
            "Z_features": Z_features,
        }
        if r_list_Z_val is not None:
            val_results["pearson_r_per_channel_Z"] = r_list_Z_val
            val_results["pearson_r_mean_Z"] = r_mean_Z_val

        chunk_margin_val = meta_val[0].get("chunk_margin") if meta_val else 0
        val_forecast = self.framework._evaluate_forecast(
            Y_val,
            Z_list=Z_val,
            margin=chunk_margin_val,
            m_seconds=m_seconds,
            history_seconds=history_seconds,
        )
        val_results.update(val_forecast)

        self.save_results(
            val_results, self.val_loader.dataset.df, type="val", meta_list=meta_val
        )

        return val_results

    def save_results(
        self,
        results: dict,
        input_df: pl.DataFrame,
        type: str,
        meta_list: list[dict] = None,
    ):

        def safe_tolist(arr):
            if arr is None:
                return None
            if hasattr(arr, "tolist"):
                return arr.tolist()
            return arr

        new_cols = []

        if "pearson_r_per_channel" in results:
            new_cols.append(
                pl.Series(
                    name="pearsonr_per_channel", values=results["pearson_r_per_channel"]
                )
            )

        if "pearson_r_per_channel_Z" in results:
            new_cols.append(
                pl.Series(
                    name="pearsonr_per_channel_Z",
                    values=results["pearson_r_per_channel_Z"],
                )
            )

        for key in ["Y", "Z", "Yp", "Zp", "Xp"]:
            if key in results and results[key] is not None:
                new_cols.append(
                    pl.Series(name=key, values=[safe_tolist(x) for x in results[key]])
                )

        if meta_list is not None:
            new_cols.append(
                pl.Series(
                    name="time", values=[safe_tolist(m.get("time")) for m in meta_list]
                )
            )
            new_cols.append(
                pl.Series(
                    name="chunk_margin",
                    values=[m.get("chunk_margin") for m in meta_list],
                )
            )
            new_cols.append(
                pl.Series(
                    name="margined_duration",
                    values=[m.get("margined_duration") for m in meta_list],
                )
            )

        if "pearson_r_mean" in results:
            new_cols.append(pl.lit(results["pearson_r_mean"]).alias("pearsonr_mean"))

        if "pearson_r_mean_Z" in results:
            new_cols.append(
                pl.lit(results["pearson_r_mean_Z"]).alias("pearsonr_mean_Z")
            )

        forecast_keys = [
            "Y_future_true",
            "Y_future_pred",
            "Y_concat_for_plot",
            "Z_future_true",
            "Z_future_pred",
            "Z_concat_for_plot",
            "X_future_pred",
            "pearson_per_channel",
            "pearson_per_channel_Z",
        ]

        for key in forecast_keys:
            if key in results and results[key] is not None:
                new_cols.append(
                    pl.Series(name=key, values=[safe_tolist(x) for x in results[key]])
                )

        if "Y_features" in results:
            n_rows = len(input_df)
            new_cols.append(
                pl.Series(name="Y_features", values=[results["Y_features"]] * n_rows)
            )

        if "Z_features" in results:
            n_rows = len(input_df)
            new_cols.append(
                pl.Series(name="Z_features", values=[results["Z_features"]] * n_rows)
            )

        metrics_df = input_df.with_columns(new_cols)
        if isinstance(results, dict):
            for k, v in results.items():
                if not isinstance(v, (list, dict)):
                    try:
                        metrics_df = metrics_df.with_columns(
                            pl.lit(v).alias(f"metric_{k}")
                        )
                    except Exception:
                        pass

        ts = self.run_timestamp
        out_path = self.save_dir / type / f"test_results_{ts}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.write_parquet(
            out_path, partition_by=["participant_id", "session", "block", "trial"]
        )
        self.logger.info(f"Detailed {type} results saved to {out_path}")

        try:
            model_path = self.save_dir / f"model_{ts}"

            if self.framework_type == "dpad":
                try:
                    self.framework.model.idSys.discardModels()
                    model_path_pkl = f"{model_path}.pkl"
                    with open(model_path_pkl, "wb") as f:
                        pickle.dump(self.framework.model.idSys, f)
                    self.logger.info(f"Saved DPAD model to {model_path_pkl}")
                finally:
                    self.framework.model.idSys.restoreModels()

                metadata = {
                    "framework_type": "dpad",
                    "nx": self.model_params.nx,
                    "n1": self.model_params.n1,
                    "method_code": self.model_params.method_code,
                    "epochs": self.model_params.epochs,
                }
                with open(self.save_dir / f"model_{ts}_metadata.json", "w") as f:
                    json.dump(metadata, f)

            elif self.framework_type == "varma":
                with open(f"{model_path}.pkl", "wb") as f:
                    pickle.dump(self.framework.model, f)
                self.logger.info(f"Saved VARMA-OLS model to {model_path}.pkl")

                metadata = {
                    "framework_type": "varma",
                    "p": self.model_params.p,
                    "q": self.model_params.q,
                    "long_ar_lags": self.model_params.long_ar_lags,
                    "K": self.framework.model.K,
                    "n_channels_Y": self.framework.model.n_channels_Y,
                    "n_channels_Z": self.framework.model.n_channels_Z,
                    "beta_shape": (
                        list(self.framework.model.beta.shape)
                        if getattr(self.framework.model, "beta", None) is not None
                        else None
                    ),
                }
                with open(self.save_dir / f"model_{ts}_metadata.json", "w") as f:
                    json.dump(metadata, f)

            else:
                with open(f"{model_path}.pkl", "wb") as f:
                    pickle.dump(self.framework.model.idSys, f)
                self.logger.info(f"Saved PSID model to {model_path}.pkl")

                metadata = {
                    "framework_type": "psid",
                    "nx": self.model_params.nx,
                    "n1": self.model_params.n1,
                    "i": getattr(self.model_params, "i", None),
                    "A_eigen_constrain": self.model_params.A_eigen_constrain,
                }
                with open(self.save_dir / f"model_{ts}_metadata.json", "w") as f:
                    json.dump(metadata, f)

        except Exception as e:
            self.logger.warning(f"Could not save model/trainer artifacts: {e}")
