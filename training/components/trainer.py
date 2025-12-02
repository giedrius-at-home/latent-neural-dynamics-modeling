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

    def split_data(self):
        self.logger.info("Starting data split...")
        self.logger.info(
            f"Data params: participant={self.data_params.participant}, session={self.data_params.session}, "
            f"input_channels={self.data_params.channels.input}, output_channels={self.data_params.channels.output}, "
            f"is_behavioral_neural={self.data_params.channels.is_behavioral_neural}"
        )
        session_path = (
            Path(self.data_params.root)
            / f"participant_id={self.data_params.participant}"
            / f"session={self.data_params.session}"
        )
        combined_cols = list(
            set(self.data_params.channels.input) | set(self.data_params.channels.output)
        )
        combined_cols = [pl.col(f"^{col}.*$") for col in combined_cols]

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
            .sort(
                [
                    pl.col("participant_id"),
                    pl.col("session"),
                    pl.col("block"),
                    pl.col("trial"),
                ],
                maintain_order=True,
            )
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
        """Load a pre-trained model from a saved checkpoint.

        Args:
            model_timestamp: Timestamp of the saved model (e.g., "20251125_153301")
        """
        out_dir = Path(self.results_config.save_dir)
        model_path = out_dir / f"model_{model_timestamp}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.logger.info(f"Loading pre-trained model from {model_path}")

        # Load metadata for DPAD models to validate compatibility
        if self.framework_type == "dpad":
            metadata_path = out_dir / f"model_{model_timestamp}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Validate key parameters match
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

        # Initialize the model wrapper first, then load the saved model
        self.framework.model = self.framework._initialize_model()
        self.framework.model.load_from_file(str(model_path))
        self.logger.info("Model loaded successfully")

    def train(self):
        if self.train_loader is None:
            raise ValueError("Data loaders not initialized. Call split_data() first.")

        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Training run timestamp: {self.run_timestamp}")

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

        Y_train, Z_train, meta_train = self.train_loader.get_full_dataset()
        Y_val, Z_val, meta_val = self.val_loader.get_full_dataset()

        Y_train, Z_train = self._slice_data(Y_train, Z_train, meta_train)
        Y_val, Z_val = self._slice_data(Y_val, Z_val, meta_val)

        # Check if we should load a pre-trained model or train from scratch
        model_timestamp = getattr(self.model_params, "model_timestamp", None)

        if model_timestamp:
            self.logger.info(
                f"Loading pre-trained model with timestamp: {model_timestamp}"
            )
            self.load_model(model_timestamp)
        else:
            self.logger.info("Beginning training...")
            self.framework._train(Y_train, Z_train)

        self.logger.info("Computing training set predictions...")
        Zp_train, Yp_train, Xp_train = self.framework._predict(Y_train)

        # Compute metrics including Z correlations
        from utils.stats import pearson_r_per_channel

        r_list_train, r_mean_train = pearson_r_per_channel(Y_train, Yp_train)

        train_results = {
            "Y": Y_train,
            "Z": Z_train,
            "Zp": Zp_train,
            "Yp": Yp_train,
            "Xp": Xp_train,
            "pearson_r_per_channel": r_list_train,
            "pearson_r_mean": r_mean_train,
            "input_channels": (
                meta_train[0].get("input_channels", []) if meta_train else []
            ),
            "output_channels": (
                meta_train[0].get("output_channels", []) if meta_train else []
            ),
        }

        # Compute Z correlations if Z data is available
        if Z_train is not None and Zp_train is not None:
            Z_train_filtered = [z for z in Z_train if z is not None]
            Zp_train_filtered = [zp for zp in Zp_train if zp is not None]
            if len(Z_train_filtered) > 0 and len(Zp_train_filtered) > 0:
                r_list_Z_train, r_mean_Z_train = pearson_r_per_channel(
                    Z_train_filtered, Zp_train_filtered
                )
                train_results["pearson_r_per_channel_Z"] = r_list_Z_train
                train_results["pearson_r_mean_Z"] = r_mean_Z_train

        self.logger.info("Computing training set forecasts...")
        chunk_margin_train = meta_train[0].get("chunk_margin") if meta_train else 0
        train_forecast = self.framework._validate_forecast(
            Y_train, Z_list=Z_train, margin=chunk_margin_train
        )
        train_results.update(train_forecast)

        self.logger.info(
            f"Training predictions complete. Results: {train_results.keys()}"
        )
        self.save_results(train_results, self.train_loader.dataset.df, type="train")

        self.logger.info("Beginning validation...")
        Zp_val, Yp_val, Xp_val = self.framework._predict(Y_val)

        # Compute metrics including Z correlations
        r_list_val, r_mean_val = pearson_r_per_channel(Y_val, Yp_val)

        val_results = {
            "Y": Y_val,
            "Z": Z_val,
            "Zp": Zp_val,
            "Yp": Yp_val,
            "Xp": Xp_val,
            "pearson_r_per_channel": r_list_val,
            "pearson_r_mean": r_mean_val,
            "input_channels": meta_val[0].get("input_channels", []) if meta_val else [],
            "output_channels": (
                meta_val[0].get("output_channels", []) if meta_val else []
            ),
        }

        # Compute Z correlations if Z data is available
        if Z_val is not None and Zp_val is not None:
            Z_val_filtered = [z for z in Z_val if z is not None]
            Zp_val_filtered = [zp for zp in Zp_val if zp is not None]
            if len(Z_val_filtered) > 0 and len(Zp_val_filtered) > 0:
                r_list_Z_val, r_mean_Z_val = pearson_r_per_channel(
                    Z_val_filtered, Zp_val_filtered
                )
                val_results["pearson_r_per_channel_Z"] = r_list_Z_val
                val_results["pearson_r_mean_Z"] = r_mean_Z_val

        self.logger.info("Computing validation set forecasts...")
        chunk_margin_val = meta_val[0].get("chunk_margin") if meta_val else 0
        val_forecast = self.framework._validate_forecast(
            Y_val, Z_list=Z_val, margin=chunk_margin_val
        )
        val_results.update(val_forecast)

        self.logger.info(f"Validation complete. Results: {val_results.keys()}")
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

        # Add Z correlations if available
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

        # Add Z mean correlation if available
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

        # Add channel metadata as scalar columns (same value for all rows)
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

        except Exception as e:
            self.logger.warning(f"Could not save model/trainer artifacts: {e}")
