import pickle
import json
from datetime import datetime
from utils.config import Config
import polars as pl
from pathlib import Path
from utils.split import create_splits
from training.components.data import create_dataloaders
from utils.logger import get_logger
from utils.miscellaneous import length


class Trainer:

    def __init__(self, config: Config):
        self.config = config
        self.model_params = config.model
        self.data_params = config.data
        self.results_config = config.results
        self.logger = get_logger()

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.run_timestamp = None

    def split_data(self, reuse_splits: bool = False):
        self.logger.info("Starting data split...")
        self.logger.info(
            f"Data params: participant={self.data_params.participant}, session={self.data_params.session}, "
            f"input_channels={self.data_params.channels.input}, output_channels={self.data_params.channels.output}, "
            f"is_behavioral_neural={self.data_params.channels.is_behavioral_neural}"
        )
        
        split_dir = Path(self.results_config.save_dir) / "split"
        existing_splits = (
            (split_dir / "train.parquet").exists() and
            (split_dir / "val.parquet").exists() and
            (split_dir / "test.parquet").exists()
        )
        
        if reuse_splits and existing_splits:
            self.logger.info(f"Reusing existing splits from {split_dir}")
        else:
            session_path = (
                Path(self.data_params.root)
                / f"participant_id={self.data_params.participant}"
                / f"session={self.data_params.session}"
            )
            base_cols = list(
                set(self.data_params.channels.input) | set(self.data_params.channels.output)
            )
            combined_cols = []
            for col in base_cols:
                combined_cols.append(pl.col(col))
                is_neural = col.startswith("LFP") or col.startswith("ECOG")
                is_input = col in self.data_params.channels.input
                if is_neural and (
                    is_input or self.data_params.channels.is_behavioral_neural
                ):
                    combined_cols.append(pl.col(f"{col}_epochs"))

            epoch_samp = f"{self.data_params.channels.input[0]}_epochs"
            trial = (
                pl.read_parquet(session_path)
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
                .with_columns(pl.col(epoch_samp).list.len().alias("n_epochs"))
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

            dbs_condition = self.data_params.dbs_condition
            if dbs_condition != "both":
                trial = trial.filter(pl.col("stim") == dbs_condition)
                self.logger.info(f"Filtered to {dbs_condition} DBS condition")

            create_splits(trial, self.data_params.split, self.results_config)

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.data_params, self.results_config
        )

    def _slice_data(self, Y_list_margined, Z_list_margined, meta_list):
        (
            _Y,
            _Z,
        ) = (
            [],
            [],
        )
        Z_list_margined = (
            [None] * len(Y_list_margined)
            if Z_list_margined is None
            else Z_list_margined
        )
        for Y, Z, meta in zip(Y_list_margined, Z_list_margined, meta_list):
            chunk_margin = meta["chunk_margin_ts"]

            Y_sliced = Y[chunk_margin:-chunk_margin]
            if self.data_params.channels.is_behavioral_neural:
                Z_sliced = Z[chunk_margin:-chunk_margin]
            else:
                Z_sliced = Z
            _Y.append(Y_sliced)
            _Z.append(Z_sliced)

        _Z = None if all([_z is None for _z in _Z]) else _Z
        self.logger.info(
            f"Sliced data: Y={length(_Y)}, Z={length(_Z)}, meta={length(meta_list)}"
        )
        return _Y, _Z

    def load_model(self, model_timestamp: str):

        out_dir = Path(self.results_config.save_dir)
        model_path = out_dir / f"model_{model_timestamp}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.logger.info(f"Loading pre-trained model from {model_path}")

        if self.framework_type == "dpad":
            metadata_path = out_dir / f"model_{model_timestamp}_metadata.json"
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
        self.framework_type = self.model_params.name.split("_")[0]
        self.logger.info(f"Selected framework: {self.framework_type}")

        if self.framework_type == "psid":
            from utils.frameworks import PSIDFramework
            self.framework = PSIDFramework(self.config)
        elif self.framework_type == "dpad":
            from utils.frameworks import DPADFramework
            self.framework = DPADFramework(self.config)
        else:
            raise ValueError(f"Unknown framework type: {self.framework_type}")

    def _prepare_data(self):
        Y_train, Z_train, meta_train = self.train_loader.get_full_dataset()
        Y_val, Z_val, meta_val = self.val_loader.get_full_dataset()

        Y_train, Z_train = self._slice_data(Y_train, Z_train, meta_train)
        Y_val, Z_val = self._slice_data(Y_val, Z_val, meta_val)

        return Y_train, Z_train, meta_train, Y_val, Z_val, meta_val

    def _compute_z_correlation(self, Z, Zp):
        from utils.stats import pearson_r_per_channel
        
        if Z is not None and Zp is not None:
            Z_filtered = [z for z in Z if z is not None]
            Zp_filtered = [zp for zp in Zp if zp is not None]
            if len(Z_filtered) > 0 and len(Zp_filtered) > 0:
                r_list, r_mean = pearson_r_per_channel(Z_filtered, Zp_filtered)
                return r_list, r_mean
        return None, None

    def _save_metadata(self, r_mean_val, r_mean_Z_val):
        ts = self.run_timestamp
        out_dir = Path(self.results_config.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "framework_type": self.framework_type,
            "nx": self.model_params.nx,
            "n1": self.model_params.n1,
            "i": getattr(self.model_params, "i", None),
            "r_mean_Y": float(r_mean_val) if r_mean_val is not None else None,
            "r_mean_Z": float(r_mean_Z_val) if r_mean_Z_val is not None else None,
        }
        
        metadata_path = out_dir / f"model_{ts}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        r_z_str = f"{r_mean_Z_val:.4f}" if r_mean_Z_val else "N/A"
        self.logger.info(f"Saved metadata to {metadata_path}")
        self.logger.info(f"R2 Y={r_mean_val:.4f}, R2 Z={r_z_str}")

    def train(self, fast: bool = False):
        if self.train_loader is None:
            raise ValueError("Data loaders not initialized. Call split_data() first.")

        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = "[FAST MODE] " if fast else ""
        self.logger.info(f"{mode_str}Training run timestamp: {self.run_timestamp}")

        self._init_framework()
        Y_train, Z_train, meta_train, Y_val, Z_val, meta_val = self._prepare_data()

        model_timestamp = getattr(self.model_params, "model_timestamp", None)
        if model_timestamp:
            self.logger.info(f"Loading pre-trained model with timestamp: {model_timestamp}")
            self.load_model(model_timestamp)
        else:
            self.logger.info("Beginning training...")
            self.framework._train(Y_train, Z_train)

        Zp_val, Yp_val, Xp_val = self.framework._predict(Y_val)

        from utils.stats import pearson_r_per_channel
        r_list_val, r_mean_val = pearson_r_per_channel(Y_val, Yp_val)
        _, r_mean_Z_val = self._compute_z_correlation(Z_val, Zp_val)

        if fast:
            self._save_metadata(r_mean_val, r_mean_Z_val)
            return {"pearson_r_mean": r_mean_val, "pearson_r_mean_Z": r_mean_Z_val}

        Zp_train, Yp_train, Xp_train = self.framework._predict(Y_train)
        r_list_train, r_mean_train = pearson_r_per_channel(Y_train, Yp_train)

        train_results = {
            "Y": Y_train, "Z": Z_train, "Zp": Zp_train, "Yp": Yp_train, "Xp": Xp_train,
            "pearson_r_per_channel": r_list_train, "pearson_r_mean": r_mean_train,
            "input_channels": meta_train[0].get("input_channels", []) if meta_train else [],
            "output_channels": meta_train[0].get("output_channels", []) if meta_train else [],
        }

        r_list_Z_train, r_mean_Z_train = self._compute_z_correlation(Z_train, Zp_train)
        if r_list_Z_train is not None:
            train_results["pearson_r_per_channel_Z"] = r_list_Z_train
            train_results["pearson_r_mean_Z"] = r_mean_Z_train

        chunk_margin_train = meta_train[0].get("chunk_margin") if meta_train else 0
        train_forecast = self.framework._validate_forecast(Y_train, Z_list=Z_train, margin=chunk_margin_train)
        train_results.update(train_forecast)

        self.save_results(train_results, self.train_loader.dataset.df, type="train")

        r_list_Z_val, _ = self._compute_z_correlation(Z_val, Zp_val)
        val_results = {
            "Y": Y_val, "Z": Z_val, "Zp": Zp_val, "Yp": Yp_val, "Xp": Xp_val,
            "pearson_r_per_channel": r_list_val, "pearson_r_mean": r_mean_val,
            "input_channels": meta_val[0].get("input_channels", []) if meta_val else [],
            "output_channels": meta_val[0].get("output_channels", []) if meta_val else [],
        }
        if r_list_Z_val is not None:
            val_results["pearson_r_per_channel_Z"] = r_list_Z_val
            val_results["pearson_r_mean_Z"] = r_mean_Z_val

        chunk_margin_val = meta_val[0].get("chunk_margin") if meta_val else 0
        val_forecast = self.framework._validate_forecast(Y_val, Z_list=Z_val, margin=chunk_margin_val)
        val_results.update(val_forecast)

        self.save_results(val_results, self.val_loader.dataset.df, type="val")

        return val_results

    def save_results(self, results: dict, input_df: pl.DataFrame, type: str):

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

        if "input_channels" in results:
            n_rows = len(input_df)
            new_cols.append(
                pl.Series(
                    name="input_channels", values=[results["input_channels"]] * n_rows
                )
            )

        if "output_channels" in results:
            n_rows = len(input_df)
            new_cols.append(
                pl.Series(
                    name="output_channels", values=[results["output_channels"]] * n_rows
                )
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
        out_dir = Path(self.results_config.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{type}_results_{ts}"
        metrics_df.write_parquet(
            out_path, partition_by=["participant_id", "session", "block", "trial"]
        )
        self.logger.info(f"Detailed {type} results saved to {out_path}")

        try:
            model_path = out_dir / f"model_{ts}"

            if self.framework_type == "dpad":
                self.framework.model.idSys.discardModels()

                model_path_pkl = f"{model_path}.pkl"
                with open(model_path_pkl, "wb") as f:
                    pickle.dump(self.framework.model.idSys, f)

                self.logger.info(f"Saved DPAD model to {model_path_pkl}")

                self.framework.model.idSys.restoreModels()

                metadata = {
                    "framework_type": "dpad",
                    "nx": self.model_params.nx,
                    "n1": self.model_params.n1,
                    "method_code": self.model_params.method_code,
                    "epochs": self.model_params.epochs,
                }
                with open(out_dir / f"model_{ts}_metadata.json", "w") as f:
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
                }
                with open(out_dir / f"model_{ts}_metadata.json", "w") as f:
                    json.dump(metadata, f)

        except Exception as e:
            self.logger.warning(f"Could not save model/trainer artifacts: {e}")
