from typing import Any, Dict, List, Optional, Tuple
from numpy.typing import NDArray
from utils.config import Config
from utils.logger import get_logger
from utils.miscellaneous import state_shape
from utils.stats import pearson_r_per_channel
import numpy as np
import sys
from pathlib import Path
import pandas as pd

psid_path = Path(__file__).parent.parent / "PSID"
if str(psid_path.parent) not in sys.path:
    sys.path.insert(0, str(psid_path.parent))

dpad_path = Path(__file__).parent.parent / "DPAD-main" / "source"
if str(dpad_path) not in sys.path:
    sys.path.insert(0, str(dpad_path))

Array2D = NDArray[np.float64]
TrialList = List[Array2D]


class BaseFramework:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.logger = get_logger()

    def _initialize_model(self):
        raise NotImplementedError

    def _train(self, Y: TrialList, Z: Optional[TrialList] = None):
        self.logger.info("Initializing model and starting training.")
        self.model = self._initialize_model()
        self.logger.info(f"Model initialized: {self.model}")
        return self.model.train(Y, Z)

    def _validate(self, Y: TrialList) -> Dict[str, Any]:
        self.logger.info("Starting validation...")
        return self.model.validate(Y)

    def _test(self, Y: TrialList) -> Dict[str, Any]:
        self.logger.info("Starting test...")
        return self.model.test(Y)

    def _predict(self, Y: TrialList):
        self.logger.info("Running prediction on provided data...")
        return self.model.predict(Y)

    def _forecast(self, m: int, Y_past: Array2D):
        self.logger.info(f"Running {m}-step ahead forecast...")
        return self.model.forecast(m, Y_past)

    def _validate_forecast(
        self,
        Y_list: TrialList,
        Z_list: Optional[TrialList] = None,
        margin: Optional[float] = None,
        Yp_val: Optional[TrialList] = None,
        Zp_val: Optional[TrialList] = None,
    ) -> Dict[str, Any]:
        self.logger.info("Starting forecast validation...")
        return self.model.validate_forecast(
            Y_list, Z_list=Z_list, margin=margin, Yp_val=Yp_val, Zp_val=Zp_val
        )


class PSIDWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.idSys = None

    def load_from_file(self, model_path: str):
        import pickle

        self.logger.info(f"Loading PSID model from {model_path}")
        with open(model_path, "rb") as f:
            self.idSys = pickle.load(f)
        self.logger.info("PSID model loaded successfully")
        return self.idSys

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        from PSID.PSID import PSID as PSIDClass

        nx: int = self.config.model.nx
        n1: int = self.config.model.n1
        i: int = self.config.model.i
        time_first: bool = self.config.model.time_first
        alpha_Q: float = float(getattr(self.config.model, "alpha_Q", 0.0))
        alpha_R: float = float(getattr(self.config.model, "alpha_R", 0.0))
        backward_kalman: bool = getattr(self.config.model, "backward_kalman", False)
        rescale_states: bool = getattr(self.config.model, "rescale_states", True)
        max_eigenvalue = getattr(self.config.model, "max_eigenvalue", 0.995)
        # Handle YAML null which becomes None
        if max_eigenvalue is None:
            max_eigenvalue = 1.0
        else:
            max_eigenvalue = float(max_eigenvalue)
        self.logger.info(
            f"Calling PSID.PSID with nx={nx}, n1={n1}, i={i}, time_first={time_first}, "
            f"alpha_Q={alpha_Q}, alpha_R={alpha_R}, backward_kalman={backward_kalman}, "
            f"rescale_states={rescale_states}, max_eigenvalue={max_eigenvalue}"
        )

        self.idSys = PSIDClass(
            Y,
            Z,
            nx,
            n1,
            i,
            zscore_Y=True,
            zscore_Z=True,
            remove_mean_Y=True,
            remove_mean_Z=True,
            time_first=time_first,
            alpha_Q=alpha_Q,
            alpha_R=alpha_R,
            backward_kalman=backward_kalman,
            rescale_states=rescale_states,
            max_eigenvalue=max_eigenvalue,
        )
        return self.idSys

    def predict(self, Y: TrialList):
        use_smoothing = getattr(self.idSys, "backward_kalman", False)
        return self.idSys.predict(Y, useSmoothing=use_smoothing)

    def validate(self, Y: TrialList) -> Dict[str, Any]:
        Zp, Yp, Xp = self.idSys.predict(Y)
        r_list, r_mean = pearson_r_per_channel(Y, Yp if Yp is not None else Y)
        result = {
            "Y": Y,
            "Zp": Zp,
            "Yp": Yp,
            "Xp": Xp,
            "Yp_shape": state_shape(Yp),
            "Zp_shape": (None if Zp is None else state_shape(Zp)),
            "Xp_shape": (None if Xp is None else state_shape(Xp)),
            "pearson_r_per_channel": r_list,
            "pearson_r_mean": r_mean,
        }

        return result

    def test(self, Y: TrialList) -> Dict[str, Any]:
        return self.validate(Y)

    def forecast(self, m: int, Y_past: Array2D):
        return self.idSys.forecast(m, Y_past)

    def validate_forecast(
        self,
        Y_list: TrialList,
        Z_list: Optional[TrialList] = None,
        margin: Optional[float] = None,
        Yp_val: Optional[TrialList] = None,
        Zp_val: Optional[TrialList] = None,
    ) -> Dict[str, Any]:

        m_seconds = self.config.model.forecast.m
        history_seconds = self.config.model.forecast.history
        sampling_freq = self.config.data.sampling_frequency
        m = int(m_seconds * sampling_freq)
        history = int(history_seconds * sampling_freq)
        margin_sec = margin if margin is not None else 0.0

        if Yp_val is None:
            Zp_val_local, Yp_val_local, Xp_val_local = self.predict(Y_list)
            Yp_val = Yp_val_local
            Zp_val = Zp_val_local

        all_residuals = []
        for y_true, y_pred in zip(Y_list, Yp_val):
            if y_pred is not None:
                residuals = y_true - y_pred
                all_residuals.append(residuals)

        if all_residuals:
            all_residuals_concat = np.concatenate(all_residuals, axis=0)
            residual_mean = np.mean(all_residuals_concat, axis=0)
            residual_std = np.std(all_residuals_concat, axis=0)
        else:
            residual_mean = 0.0
            residual_std = 0.0

        # TODO: Check Gaussian assumption for behavioral (Z) residuals and apply probabilistic forecasting if valid

        results = {
            "m": m,
            "Y_future_true": [],
            "Y_future_pred": [],
            "Y_concat_for_plot": [],
            "Z_future_true": [],
            "Z_future_pred": [],
            "Z_concat_for_plot": [],
            "X_future_pred": [],
            "pearson_per_channel": [],
            "pearson_per_channel_Z": [],
            "residual_mean": (
                residual_mean.tolist()
                if isinstance(residual_mean, np.ndarray)
                else residual_mean
            ),
            "residual_std": (
                residual_std.tolist()
                if isinstance(residual_std, np.ndarray)
                else residual_std
            ),
        }

        for idx, Y in enumerate(Y_list):
            T = Y.shape[0]
            if history + m > T:
                raise ValueError(
                    f"history + m ({history} + {m} = {history + m}) must not exceed trial length T={T}"
                )

            start = 0
            history_end = history
            forecast_end = history + m

            Y_past = Y[start:history_end]
            Y_future_true = Y[history_end:forecast_end]

            Z = Z_list[idx] if Z_list is not None and idx < len(Z_list) else None
            Z_future_true = Z[history_end:forecast_end] if Z is not None else None
            Z_past = Z[start:history_end] if Z is not None else None

            Zf, Yf, Xf = self.forecast(m, Y_past)

            Y_concat = np.concatenate([Y_past, Yf], axis=0)
            Z_concat = (
                np.concatenate([Z_past, Zf], axis=0)
                if Z_past is not None and Zf is not None
                else None
            )

            r_list, _ = pearson_r_per_channel([Y_future_true], [Yf])
            r_list = (
                r_list[0] if isinstance(r_list, list) and len(r_list) > 0 else r_list
            )

            if Z_future_true is not None and Zf is not None:
                r_list_Z, _ = pearson_r_per_channel([Z_future_true], [Zf])
                r_list_Z = (
                    r_list_Z[0]
                    if isinstance(r_list_Z, list) and len(r_list_Z) > 0
                    else r_list_Z
                )
            else:
                r_list_Z = []

            results["Y_future_true"].append(Y_future_true.tolist())
            results["Y_future_pred"].append(Yf.tolist())
            results["Y_concat_for_plot"].append(Y_concat.tolist())
            results["Z_future_true"].append(
                Z_future_true.tolist() if Z_future_true is not None else None
            )
            results["Z_future_pred"].append(Zf.tolist() if Zf is not None else None)
            results["Z_concat_for_plot"].append(
                Z_concat.tolist() if Z_concat is not None else None
            )
            results["X_future_pred"].append(Xf.tolist() if Xf is not None else None)
            results["pearson_per_channel"].append(r_list)
            results["pearson_per_channel_Z"].append(r_list_Z)

        flat_r = []
        for r in results["pearson_per_channel"]:
            if r is None:
                continue
            for v in r:
                if v is not None and not np.isnan(v):
                    flat_r.append(float(v))
        results["pearson_overall_mean"] = (
            float(np.mean(flat_r)) if len(flat_r) > 0 else np.nan
        )

        flat_r_Z = []
        for r in results["pearson_per_channel_Z"]:
            if r is None or not r:
                continue
            for v in r:
                if v is not None and not np.isnan(v):
                    flat_r_Z.append(float(v))
        results["pearson_overall_mean_Z"] = (
            float(np.mean(flat_r_Z)) if len(flat_r_Z) > 0 else np.nan
        )

        return results


class PSIDFramework(BaseFramework):
    def _initialize_model(self):
        return PSIDWrapper(self.config)


class DPADWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.idSys = None

    def load_from_file(self, model_path: str):
        import pickle

        self.logger.info(f"Loading DPAD model from {model_path}")
        with open(model_path, "rb") as f:
            self.idSys = pickle.load(f)

        self.idSys.restoreModels()
        self.idSys.set_steps_ahead([1])
        self.idSys.set_multi_step_with_data_gen(False)

        self.logger.info("DPAD model loaded and restored successfully")
        return self.idSys

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        from DPAD import DPADModel

        nx: int = self.config.model.nx
        n1: int = self.config.model.n1
        method_code: str = self.config.model.method_code
        epochs: int = self.config.model.epochs

        def safe_cast(val, cast_type):
            return cast_type(val) if val is not None else None

        dropout = safe_cast(getattr(self.config.model, "dropout", None), float)
        weight_decay = safe_cast(
            getattr(self.config.model, "weight_decay", None), float
        )
        hidden_size = safe_cast(getattr(self.config.model, "hidden_size", None), int)
        layers = safe_cast(getattr(self.config.model, "layers", None), int)
        loss_name = getattr(self.config.model, "loss_name", None)

        behavior_loss_weight = safe_cast(
            getattr(self.config.model, "behavior_loss_weight", 1.0), float
        )
        recon_loss_weight = safe_cast(
            getattr(self.config.model, "recon_loss_weight", 1.0), float
        )
        use_correlation_loss = getattr(self.config.model, "use_correlation_loss", True)
        fast = getattr(self.config.model, "fast", False)

        self.logger.info(
            f"Training DPAD with nx={nx}, n1={n1}, method_code={method_code}, epochs={epochs}"
        )
        Y_dpad = [y.T for y in Y]
        Z_dpad = [z.T for z in Z] if Z is not None else None

        self.idSys = DPADModel(log_dir=self.config.results.log_dir)
        args = DPADModel.prepare_args(method_code)

        # Apply overrides from config
        def update_arg_dict(d, key, val):
            if isinstance(d, dict):
                d[key] = val

        # Handle top-level fit arguments
        if dropout is not None:
            for arg_name in [
                "A1_args",
                "K1_args",
                "Cy1_args",
                "Cz1_args",
                "A2_args",
                "K2_args",
                "Cy2_args",
                "Cz2_args",
                "A_args",
                "K_args",
                "Cy_args",
                "Cz_args",
            ]:
                if arg_name in args:
                    update_arg_dict(args[arg_name], "dropout_rate", dropout)

        if weight_decay is not None:
            for arg_name in [
                "A1_args",
                "K1_args",
                "Cy1_args",
                "Cz1_args",
                "A2_args",
                "K2_args",
                "Cy2_args",
                "Cz2_args",
                "A_args",
                "K_args",
                "Cy_args",
                "Cz_args",
            ]:
                if arg_name in args:
                    update_arg_dict(args[arg_name], "kernel_regularizer_name", "l2")
                    update_arg_dict(
                        args[arg_name], "kernel_regularizer_args", {"l": weight_decay}
                    )

        if hidden_size is not None and layers is not None:
            for arg_name in [
                "A1_args",
                "K1_args",
                "Cy1_args",
                "Cz1_args",
                "A2_args",
                "K2_args",
                "Cy2_args",
                "Cz2_args",
                "A_args",
                "K_args",
                "Cy_args",
                "Cz_args",
            ]:
                if arg_name in args:
                    update_arg_dict(args[arg_name], "units", [hidden_size] * layers)
                    update_arg_dict(args[arg_name], "activation", "relu")
                    update_arg_dict(args[arg_name], "use_bias", True)

        parsed_loss = args.pop("loss_name", None)
        final_loss = loss_name if loss_name is not None else parsed_loss
        if not use_correlation_loss:
            final_loss = None

        parsed_bw = args.pop("behavior_loss_weight", 1.0)
        final_bw = (
            behavior_loss_weight
            if getattr(self.config.model, "behavior_loss_weight", None) is not None
            else parsed_bw
        )

        parsed_rw = args.pop("recon_loss_weight", 1.0)
        final_rw = (
            recon_loss_weight
            if getattr(self.config.model, "recon_loss_weight", None) is not None
            else parsed_rw
        )

        final_esm = getattr(
            self.config.model,
            "early_stopping_measure",
            args.pop("early_stopping_measure", "val_loss"),
        )
        final_esp = getattr(
            self.config.model,
            "early_stopping_patience",
            args.pop("early_stopping_patience", 16),
        )
        final_esmin = getattr(
            self.config.model,
            "start_from_epoch_rnn",
            args.pop("start_from_epoch_rnn", 0),
        )

        final_use_cnn = getattr(
            self.config.model,
            "use_cnn_envelope",
            args.pop("use_cnn_envelope", False),
        )
        final_cnn_args = getattr(
            self.config.model,
            "cnn_args",
            args.pop("cnn_args", {}),
        )

        self.idSys.fit(
            Y_dpad,
            Z=Z_dpad,
            nx=nx,
            n1=n1,
            epochs=epochs,
            loss_name=final_loss,
            behavior_loss_weight=final_bw,
            recon_loss_weight=final_rw,
            early_stopping_measure=final_esm,
            early_stopping_patience=final_esp,
            start_from_epoch_rnn=final_esmin,
            skip_predictions=fast,
            use_cnn_envelope=final_use_cnn,
            cnn_args=final_cnn_args,
            **args,
        )
        return self.idSys

    def predict(self, Y: TrialList):
        all_Zp, all_Yp, all_Xp = [], [], []

        block_samples = self.idSys.block_samples

        for y_trial in Y:
            original_len = y_trial.shape[0]
            remainder = original_len % block_samples

            if remainder != 0:
                pad_len = block_samples - remainder
                padding = np.zeros((pad_len, y_trial.shape[1]))
                y_trial_padded = np.concatenate([y_trial, padding], axis=0)
            else:
                y_trial_padded = y_trial

            result = self.idSys.predict(y_trial_padded)

            if len(result) == 3:
                Zp, Yp, Xp = result
            else:
                self.logger.warning(
                    f"Unexpected predict result length: {len(result)}, extracting first step only"
                )
                num_steps = len(result) // 3
                Zp = result[0]
                Yp = result[num_steps]
                Xp = result[2 * num_steps]

            if remainder != 0:
                Zp = Zp[:original_len] if Zp is not None else None
                Yp = Yp[:original_len] if Yp is not None else None
                Xp = Xp[:original_len] if Xp is not None else None

            all_Zp.append(np.asarray(Zp) if Zp is not None else None)
            all_Yp.append(np.asarray(Yp) if Yp is not None else None)
            all_Xp.append(np.asarray(Xp) if Xp is not None else None)

        return all_Zp, all_Yp, all_Xp

    def validate(self, Y: TrialList) -> Dict[str, Any]:
        Zp, Yp, Xp = self.predict(Y)
        r_list, r_mean = pearson_r_per_channel(Y, Yp)
        result = {
            "Y": Y,
            "Zp": Zp,
            "Yp": Yp,
            "Xp": Xp,
            "Yp_shape": state_shape(Yp),
            "Zp_shape": (None if Zp is None else state_shape(Zp)),
            "Xp_shape": (None if Xp is None else state_shape(Xp)),
            "pearson_r_per_channel": r_list,
            "pearson_r_mean": r_mean,
        }
        return result

    def test(self, Y: TrialList) -> Dict[str, Any]:
        return self.validate(Y)

    def forecast(
        self,
        m: int,
        Y_past: Array2D,
        residual_mean: Optional[Array2D] = None,
        residual_std: Optional[Array2D] = None,
    ) -> Tuple[Optional[Array2D], Optional[Array2D], Optional[Array2D]]:
        block_samples = self.idSys.block_samples
        ny = Y_past.shape[1]

        def _pad_to_block(arr):
            remainder = arr.shape[0] % block_samples
            if remainder != 0:
                pad_len = block_samples - remainder
                return np.concatenate([arr, np.zeros((pad_len, ny))], axis=0)
            return arr

        def _stack_last(steps_list):
            out = [
                arr[-1:, :] if arr is not None and len(arr.shape) == 2 else None
                for arr in steps_list
            ]
            valid = [v for v in out if v is not None]
            return np.vstack(valid) if valid else None

        self.idSys.set_steps_ahead(list(range(1, m + 1)))
        self.idSys.set_multi_step_with_data_gen(True, noise_samples=0)
        preds = self.idSys.predict(_pad_to_block(Y_past))

        Zf = _stack_last(preds[:m])
        Yf = _stack_last(preds[m : 2 * m])
        Xf = _stack_last(preds[2 * m : 3 * m])

        return Zf, Yf, Xf

    def validate_forecast(
        self,
        Y_list: TrialList,
        Z_list: Optional[TrialList] = None,
        margin: int = 0,
        Yp_val: Optional[TrialList] = None,
        Zp_val: Optional[TrialList] = None,
    ) -> Dict[str, Any]:
        m_seconds = self.config.model.forecast.m
        history_seconds = self.config.model.forecast.history
        sampling_freq = self.config.data.sampling_frequency
        m = int(m_seconds * sampling_freq)
        history = int(history_seconds * sampling_freq)

        self.idSys.set_steps_ahead([1])
        self.idSys.set_multi_step_with_data_gen(False)

        if Yp_val is None:
            Zp_val_local, Yp_val_local, Xp_val_local = self.predict(Y_list)
            Yp_val = Yp_val_local
            Zp_val = Zp_val_local

        all_residuals = []
        for y_true, y_pred in zip(Y_list, Yp_val):
            if y_pred is not None:
                all_residuals.append(y_true - y_pred)

        if all_residuals:
            all_residuals_concat = np.concatenate(all_residuals, axis=0)
            residual_mean = np.mean(all_residuals_concat, axis=0)
            residual_std = np.std(all_residuals_concat, axis=0)
        else:
            residual_mean = None
            residual_std = None

        results = {
            "m": m,
            "Y_future_true": [],
            "Y_future_pred": [],
            "Y_concat_for_plot": [],
            "Z_future_true": [],
            "Z_future_pred": [],
            "Z_concat_for_plot": [],
            "X_future_pred": [],
            "pearson_per_channel": [],
            "pearson_per_channel_Z": [],
            "residual_mean": (
                residual_mean.tolist() if residual_mean is not None else 0.0
            ),
            "residual_std": residual_std.tolist() if residual_std is not None else 0.0,
        }

        for idx, Y in enumerate(Y_list):
            T = Y.shape[0]
            if history + m > T:
                raise ValueError(
                    f"history + m ({history} + {m} = {history + m}) must not exceed trial length T={T}"
                )

            Y_history = Y[:history]
            Y_future_true = Y[history : history + m]

            Z = Z_list[idx] if Z_list is not None and idx < len(Z_list) else None
            Z_future_true = Z[history : history + m] if Z is not None else None
            Z_history = Z[:history] if Z is not None else None

            Zf, Yf, Xf = self.forecast(m, Y_history, residual_mean, residual_std)

            if Yf is not None:
                Y_concat = np.concatenate([Y_history, Yf], axis=0)
                r_list, _ = pearson_r_per_channel([Y_future_true], [Yf])
                r_list = r_list[0] if isinstance(r_list, list) else r_list
            else:
                Y_concat = Y_history
                r_list = []

            Z_concat = (
                np.concatenate([Z_history, Zf], axis=0)
                if Z_history is not None and Zf is not None
                else None
            )

            if Z_future_true is not None and Zf is not None:
                r_list_Z, _ = pearson_r_per_channel([Z_future_true], [Zf])
                r_list_Z = r_list_Z[0] if isinstance(r_list_Z, list) else r_list_Z
            else:
                r_list_Z = []

            results["Y_future_true"].append(Y_future_true.tolist())
            results["Y_future_pred"].append(Yf.tolist() if Yf is not None else None)
            results["Y_concat_for_plot"].append(Y_concat.tolist())
            results["Z_future_true"].append(
                Z_future_true.tolist() if Z_future_true is not None else None
            )
            results["Z_future_pred"].append(Zf.tolist() if Zf is not None else None)
            results["Z_concat_for_plot"].append(
                Z_concat.tolist() if Z_concat is not None else None
            )
            results["X_future_pred"].append(Xf.tolist() if Xf is not None else None)
            results["pearson_per_channel"].append(r_list)
            results["pearson_per_channel_Z"].append(r_list_Z)

        flat_r = [
            v
            for r in results["pearson_per_channel"]
            if r
            for v in r
            if v is not None and not np.isnan(v)
        ]
        results["pearson_overall_mean"] = float(np.mean(flat_r)) if flat_r else np.nan

        flat_r_Z = [
            v
            for r in results["pearson_per_channel_Z"]
            if r
            for v in r
            if v is not None and not np.isnan(v)
        ]
        results["pearson_overall_mean_Z"] = (
            float(np.mean(flat_r_Z)) if flat_r_Z else np.nan
        )

        return results


class DPADFramework(BaseFramework):
    def _initialize_model(self):
        return DPADWrapper(self.config)


class AutoARIMAWrapper:

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.models_Y = []
        self.models_Z = []
        self.Y_train = None
        self.Z_train = None
        self.Y_mean = None
        self.Y_std = None
        self.Z_mean = None
        self.Z_std = None
        
        # Store necessary params from config so they survive pickling
        self.forecast_m = getattr(config.model.forecast, 'm', 2.0)
        self.forecast_history = getattr(config.model.forecast, 'history', 5.0)
        self.sampling_freq = getattr(config.data, 'sampling_frequency', 60)
        self.n_fit_trials = getattr(config.model, 'n_fit_trials', 1)

    def __getstate__(self):
        """Exclude unpicklable logger and large training data from serialization."""
        state = self.__dict__.copy()
        state.pop("logger", None)
        state.pop("config", None)
        state.pop("Y_train", None)
        state.pop("Z_train", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = get_logger()
        # self.config remains None after unpickling, but we have stored params
        self.Y_train = None
        self.Z_train = None

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        from sktime.forecasting.arima import AutoARIMA

        self.logger.info("Training AutoARIMA models...")

        Y_concat = np.concatenate(Y, axis=0)
        n_channels_Y = Y_concat.shape[1] if Y_concat.ndim == 2 else 1

        self.Y_mean = np.mean(Y_concat, axis=0, keepdims=True)
        self.Y_std = np.std(Y_concat, axis=0, keepdims=True)
        self.Y_std[self.Y_std < 1e-10] = 1.0

        self.Y_train = [(y - self.Y_mean) / self.Y_std for y in Y]

        sp = getattr(self.config.model, 'sp', 1)
        max_p = getattr(self.config.model, 'max_p', 5)
        max_q = getattr(self.config.model, 'max_q', 5)
        max_d = getattr(self.config.model, 'max_d', 2)
        n_fit_trials = self.n_fit_trials

        n_available = len(self.Y_train)
        n_to_fit = min(n_fit_trials, n_available)
        fit_indices = np.linspace(
            0, n_available - 1, n_to_fit, dtype=int
        ).tolist()

        self.logger.info(
            f"AutoARIMA params: sp={sp}, max_p={max_p}, max_q={max_q}, max_d={max_d}, "
            f"n_channels_Y={n_channels_Y}, fitting per-trial on {n_to_fit} "
            f"representative trials (indices: {fit_indices})"
        )

        # Fit Y channels — per-trial windowed approach (NOT concatenated)
        self.models_Y = []
        for ch in range(n_channels_Y):
            best_model = None
            for i, trial_idx in enumerate(fit_indices):
                y_series = pd.Series(self.Y_train[trial_idx][:, ch])

                model = AutoARIMA(
                    sp=sp,
                    suppress_warnings=True,
                    max_p=max_p,
                    max_q=max_q,
                    max_d=max_d,
                    stepwise=True,
                    n_jobs=1,
                )
                model.fit(y_series)
                if best_model is None:
                    best_model = model
                self.logger.info(
                    f"Fitted AutoARIMA for Y ch {ch}, trial {trial_idx} "
                    f"({i + 1}/{n_to_fit})"
                )

            self.models_Y.append(best_model)
            self.logger.info(f"Selected model for Y channel {ch}")

        if Z is not None:
            Z_concat = np.concatenate(Z, axis=0)
            n_channels_Z = Z_concat.shape[1] if Z_concat.ndim == 2 else 1

            self.Z_mean = np.mean(Z_concat, axis=0, keepdims=True)
            self.Z_std = np.std(Z_concat, axis=0, keepdims=True)
            self.Z_std[self.Z_std < 1e-10] = 1.0

            self.Z_train = [(z - self.Z_mean) / self.Z_std for z in Z]

            self.models_Z = []
            for ch in range(n_channels_Z):
                best_model = None
                for i, trial_idx in enumerate(fit_indices):
                    z_series = pd.Series(self.Z_train[trial_idx][:, ch])
                    y_exog = pd.DataFrame(self.Y_train[trial_idx])

                    model = AutoARIMA(
                        sp=sp,
                        suppress_warnings=True,
                        max_p=max_p,
                        max_q=max_q,
                        max_d=max_d,
                        stepwise=True,
                        n_jobs=1,
                    )
                    model.fit(z_series, X=y_exog)
                    if best_model is None:
                        best_model = model
                    self.logger.info(
                        f"Fitted ARIMAX for Z ch {ch}, trial {trial_idx} "
                        f"({i + 1}/{n_to_fit})"
                    )

                self.models_Z.append(best_model)
                self.logger.info(f"Selected model for Z channel {ch}")

        self.logger.info("AutoARIMA training complete")
        return self

    def predict(self, Y: TrialList) -> Tuple[Optional[TrialList], TrialList, None]:
        
        all_Yp = []
        
        for y_trial in Y:
            y_zscored = (y_trial - self.Y_mean) / self.Y_std
            n_samples = y_trial.shape[0]
            n_channels = y_trial.shape[1] if y_trial.ndim == 2 else 1
            
            Yp_trial = np.zeros((n_samples, n_channels))
            for ch, model in enumerate(self.models_Y):
                try:
                    # Correct path for sktime AutoARIMA -> pmdarima -> statsmodels
                    # sktime.AutoARIMA has _forecaster (pmdarima.AutoARIMA)
                    # pmdarima.AutoARIMA has model_ (pmdarima.ARIMA) after fit
                    # pmdarima.ARIMA has arima_res_ (statsmodels result)
                    pm_auto = getattr(model, '_forecaster', None)
                    pm_model = getattr(pm_auto, 'model_', None) if pm_auto else None
                    res = getattr(pm_model, 'arima_res_', None) if pm_model else None
                    
                    if res is not None:
                        # Apply trained parameters to new data
                        new_res = res.apply(y_zscored[:, ch])
                        
                        # One-step-ahead predictions for the whole trial (in-sample for new_res)
                        preds = new_res.predict()
                        
                        # Handle potential shape mismatch if predict() returns different length
                        if len(preds) == n_samples:
                            Yp_trial[:, ch] = preds
                        else:
                            self.logger.warning(
                                f"SARIMAX.apply().predict() length mismatch for Y channel {ch}. "
                                f"Expected {n_samples}, got {len(preds)}. Falling back to rolling prediction."
                            )
                            import copy
                            model_clone = copy.deepcopy(model)
                            for t in range(n_samples):
                                pred = model_clone.predict(fh=[1])
                                Yp_trial[t, ch] = pred.values[0]
                                y_obs = pd.Series([y_zscored[t, ch]], index=[t])
                                model_clone.update(y_obs)
                    else:
                        self.logger.warning(f"Statsmodels results not found for Y channel {ch}. Falling back to rolling.")
                        import copy
                        model_clone = copy.deepcopy(model)
                        for t in range(n_samples):
                            pred = model_clone.predict(fh=[1])
                            Yp_trial[t, ch] = pred.values[0]
                            y_obs = pd.Series([y_zscored[t, ch]], index=[t])
                            model_clone.update(y_obs)
                        
                except Exception as e:
                    self.logger.warning(
                        f"Fast prediction failed for Y channel {ch}: {e}. "
                        f"Falling back to last known value."
                    )
                    Yp_trial[:, ch] = y_zscored[-1, ch] if n_samples > 0 else 0.0
            
            Yp_trial = Yp_trial * self.Y_std + self.Y_mean
            all_Yp.append(Yp_trial)
        
        all_Zp = None
        if self.models_Z and self.Z_mean is not None:
            all_Zp = []
            for trial_idx, y_trial in enumerate(Y):
                y_zscored = (y_trial - self.Y_mean) / self.Y_std
                n_samples = y_trial.shape[0]
                n_channels_Z = len(self.models_Z)
                Zp_trial = np.zeros((n_samples, n_channels_Z))
                
                y_exog = pd.DataFrame(y_zscored)
                
                for ch, model in enumerate(self.models_Z):
                    try:
                        # Bypass sktime — go through pmdarima/statsmodels directly
                        pm_auto = getattr(model, '_forecaster', None)
                        pm_model = getattr(pm_auto, 'model_', None) if pm_auto else None
                        res = getattr(pm_model, 'arima_res_', None) if pm_model else None
                        
                        if res is not None:
                            # Use a dummy endogenous series (zeros) since we don't
                            # have the true Z for this trial — we're decoding it.
                            z_dummy = np.zeros(n_samples)
                            new_res = res.apply(z_dummy, exog=y_exog)
                            preds = new_res.predict()
                            
                            if len(preds) == n_samples:
                                Zp_trial[:, ch] = preds
                            else:
                                Zp_trial[:, ch] = 0.0
                        else:
                            self.logger.warning(
                                f"No statsmodels result for Z channel {ch}. Using zero."
                            )
                            Zp_trial[:, ch] = 0.0
                            
                    except Exception as e:
                        self.logger.warning(
                            f"AutoARIMA prediction failed for Z channel {ch}: {e}. "
                            f"Falling back to zero."
                        )
                        Zp_trial[:, ch] = 0.0
                
                all_Zp.append(Zp_trial * self.Z_std + self.Z_mean)
        
        return all_Zp, all_Yp, None

    def validate(self, Y: TrialList) -> Dict[str, Any]:
        Zp, Yp, Xp = self.predict(Y)
        r_list, r_mean = pearson_r_per_channel(Y, Yp)
        result = {
            "Y": Y,
            "Zp": Zp,
            "Yp": Yp,
            "Xp": Xp,
            "Yp_shape": state_shape(Yp),
            "Zp_shape": (None if Zp is None else state_shape(Zp)),
            "Xp_shape": None,
            "pearson_r_per_channel": r_list,
            "pearson_r_mean": r_mean,
        }
        return result

    def test(self, Y: TrialList) -> Dict[str, Any]:
        return self.validate(Y)

    def forecast(self, m: int, Y_past: Array2D) -> Tuple[Optional[Array2D], Array2D, None]:
        
        n_channels_Y = Y_past.shape[1] if Y_past.ndim == 2 else 1
        Y_past_zscored = (Y_past - self.Y_mean) / self.Y_std
        Yf = np.zeros((m, n_channels_Y))
        
        for ch, model in enumerate(self.models_Y):
            try:
                pm_auto = getattr(model, '_forecaster', None)
                pm_model = getattr(pm_auto, 'model_', None) if pm_auto else None
                res = getattr(pm_model, 'arima_res_', None) if pm_model else None
                
                if res is not None:
                    # Apply parameters to the past data
                    new_res = res.apply(Y_past_zscored[:, ch])
                    
                    # Forecast from the end of the past data
                    forecast = new_res.forecast(steps=m)
                    Yf[:, ch] = forecast.values if hasattr(forecast, 'values') else forecast
                else:
                    import copy
                    model_updated = copy.deepcopy(model)
                    model_updated.update(pd.Series(Y_past_zscored[:, ch]))
                    fh = list(range(1, m + 1))
                    forecast = model_updated.predict(fh=fh)
                    Yf[:, ch] = forecast.values
            except Exception as e:
                self.logger.warning(f"Fast forecast failed for channel {ch}: {e}")
                Yf[:, ch] = Y_past_zscored[-1, ch]
        
        Yf = Yf * self.Y_std + self.Y_mean
        
        Zf = None
        if self.models_Z and self.Z_mean is not None:
            n_channels_Z = len(self.models_Z)
            Zf = np.zeros((m, n_channels_Z))
            
            Yf_zscored = (Yf - self.Y_mean) / self.Y_std
            Y_past_zscored = (Y_past - self.Y_mean) / self.Y_std
            
            for ch, model in enumerate(self.models_Z):
                try:
                    # Bypass sktime, go through pmdarima/statsmodels directly
                    pm_auto = getattr(model, '_forecaster', None)
                    pm_model = getattr(pm_auto, 'model_', None) if pm_auto else None
                    res = getattr(pm_model, 'arima_res_', None) if pm_model else None
                    
                    if res is not None:
                        # We need a "dummy" endogenous series for the history period.
                        # Use zeros since we don't have ground-truth Z for new data.
                        z_dummy = np.zeros(Y_past_zscored.shape[0])
                        y_exog_past = pd.DataFrame(Y_past_zscored)
                        
                        # Apply the fitted ARIMAX params to the history period
                        new_res = res.apply(z_dummy, exog=y_exog_past)
                        
                        # Forecast m steps ahead with future neural predictions as exog
                        y_exog_future = pd.DataFrame(Yf_zscored)
                        fc = new_res.forecast(steps=m, exog=y_exog_future)
                        Zf[:, ch] = fc.values if hasattr(fc, 'values') else fc
                    else:
                        self.logger.warning(
                            f"No statsmodels result found for Z channel {ch}. Using zero."
                        )
                        Zf[:, ch] = 0.0
                except Exception as e:
                    self.logger.warning(f"AutoARIMA Z forecast failed for channel {ch}: {e}")
                    Zf[:, ch] = 0.0
            Zf = Zf * self.Z_std + self.Z_mean
        
        return Zf, Yf, None

    def validate_forecast(
        self,
        Y_list: TrialList,
        Z_list: Optional[TrialList] = None,
        margin: Optional[float] = None,
        Yp_val: Optional[TrialList] = None,
        Zp_val: Optional[TrialList] = None,
    ) -> Dict[str, Any]:
        m_seconds = self.forecast_m
        history_seconds = self.forecast_history
        sampling_freq = self.sampling_freq
        m = int(m_seconds * sampling_freq)
        history = int(history_seconds * sampling_freq)

        if Yp_val is None:
            self.logger.info("Yp_val not provided to validate_forecast, running prediction...")
            Zp_val, Yp_val, Xp_val = self.predict(Y_list)

        all_residuals = []
        for y_true, y_pred in zip(Y_list, Yp_val):
            if y_pred is not None:
                all_residuals.append(y_true - y_pred)

        if all_residuals:
            all_residuals_concat = np.concatenate(all_residuals, axis=0)
            residual_mean = np.mean(all_residuals_concat, axis=0)
            residual_std = np.std(all_residuals_concat, axis=0)
        else:
            residual_mean = 0.0
            residual_std = 0.0

        results = {
            "m": m,
            "Y_future_true": [],
            "Y_future_pred": [],
            "Y_concat_for_plot": [],
            "Z_future_true": [],
            "Z_future_pred": [],
            "Z_concat_for_plot": [],
            "X_future_pred": [],
            "pearson_per_channel": [],
            "pearson_per_channel_Z": [],
            "residual_mean": (
                residual_mean.tolist()
                if isinstance(residual_mean, np.ndarray)
                else residual_mean
            ),
            "residual_std": (
                residual_std.tolist()
                if isinstance(residual_std, np.ndarray)
                else residual_std
            ),
        }

        for idx, Y in enumerate(Y_list):
            T = Y.shape[0]
            if history + m > T:
                raise ValueError(
                    f"history + m ({history} + {m} = {history + m}) must not exceed trial length T={T}"
                )

            start = 0
            history_end = history
            forecast_end = history + m

            Y_past = Y[start:history_end]
            Y_future_true = Y[history_end:forecast_end]

            Z = Z_list[idx] if Z_list is not None and idx < len(Z_list) else None
            Z_future_true = Z[history_end:forecast_end] if Z is not None else None
            Z_past = Z[start:history_end] if Z is not None else None

            Zf, Yf, Xf = self.forecast(m, Y_past)

            Y_concat = np.concatenate([Y_past, Yf], axis=0)
            Z_concat = (
                np.concatenate([Z_past, Zf], axis=0)
                if Z_past is not None and Zf is not None
                else None
            )

            r_list, _ = pearson_r_per_channel([Y_future_true], [Yf])
            r_list = (
                r_list[0] if isinstance(r_list, list) and len(r_list) > 0 else r_list
            )

            if Z_future_true is not None and Zf is not None:
                r_list_Z, _ = pearson_r_per_channel([Z_future_true], [Zf])
                r_list_Z = (
                    r_list_Z[0]
                    if isinstance(r_list_Z, list) and len(r_list_Z) > 0
                    else r_list_Z
                )
            else:
                r_list_Z = []

            results["Y_future_true"].append(Y_future_true.tolist())
            results["Y_future_pred"].append(Yf.tolist())
            results["Y_concat_for_plot"].append(Y_concat.tolist())
            results["Z_future_true"].append(
                Z_future_true.tolist() if Z_future_true is not None else None
            )
            results["Z_future_pred"].append(Zf.tolist() if Zf is not None else None)
            results["Z_concat_for_plot"].append(
                Z_concat.tolist() if Z_concat is not None else None
            )
            results["X_future_pred"].append(None)  # No latent states in ARIMA
            results["pearson_per_channel"].append(r_list)
            results["pearson_per_channel_Z"].append(r_list_Z)

        flat_r = []
        for r in results["pearson_per_channel"]:
            if r is None:
                continue
            for v in r:
                if v is not None and not np.isnan(v):
                    flat_r.append(float(v))
        results["pearson_overall_mean"] = (
            float(np.mean(flat_r)) if len(flat_r) > 0 else np.nan
        )

        flat_r_Z = []
        for r in results["pearson_per_channel_Z"]:
            if r is None or not r:
                continue
            for v in r:
                if v is not None and not np.isnan(v):
                    flat_r_Z.append(float(v))
        results["pearson_overall_mean_Z"] = (
            float(np.mean(flat_r_Z)) if len(flat_r_Z) > 0 else np.nan
        )

        return results


class AutoARIMAFramework(BaseFramework):
    def _initialize_model(self):
        return AutoARIMAWrapper(self.config)
