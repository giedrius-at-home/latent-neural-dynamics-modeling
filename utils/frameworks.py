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

    def _validate(self, Y: TrialList, Z: Optional[TrialList] = None) -> Dict[str, Any]:
        self.logger.info("Starting validation...")
        return self.model.validate(Y, Z)

    def _test(self, Y: TrialList, Z: Optional[TrialList] = None) -> Dict[str, Any]:
        self.logger.info("Starting test...")
        return self.model.test(Y, Z)

    def _predict(self, Y: TrialList, Z: Optional[TrialList] = None):
        self.logger.info("Running prediction on provided data...")
        return self.model.predict(Y, Z)

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

    def predict(self, Y: TrialList, Z: Optional[TrialList] = None):
        use_smoothing = getattr(self.idSys, "backward_kalman", False)
        return self.idSys.predict(Y, U=Z, useSmoothing=use_smoothing)

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
        add_noise = getattr(self.config.model, "add_process_noise_in_forecast", False)
        return self.idSys.forecast(m, Y_past, add_process_noise=add_noise)

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


class VARMAOLSWrapper:
    """
    VARMA(p, q) estimation via OLS using a Long-VAR residual proxy.

    All channels (neural + behavioral) are modeled jointly in one multivariate
    system, capturing cross-channel dynamics bidirectionally.

    Algorithm:
        1. Fit a high-order VAR to approximate the process and recover residuals.
        2. Build a design matrix: [intercept, AR lags of data, MA lags of residuals].
        3. Solve via OLS (lstsq) — no iterations, always converges.
        4. Forecast recursively: feed predictions back, future errors = 0.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()

        # VARMA orders: p = AR lags, q = MA lags; long_ar_lags = order of Long-VAR used for residual proxy
        self.p = getattr(config.model, "p", 20)
        self.q = getattr(config.model, "q", 1)
        self.long_ar_lags = getattr(config.model, "long_ar_lags", 30)
        self.ridge_alpha = getattr(config.model, "ridge_alpha", 100.0)

        # Forecast horizon (seconds) and history length (samples) for validate_forecast
        self.forecast_m = getattr(config.model.forecast, "m", 2.0)
        self.forecast_history = getattr(config.model.forecast, "history", 5.0)
        self.sampling_freq = getattr(config.data, "sampling_frequency", 80)

        # Fitted state: beta_ols maps regressors -> K outputs; K = n_channels_Y + n_channels_Z
        self.beta_ols = None
        self.n_channels_Y = None
        self.n_channels_Z = None
        self.K = None
        self.Y_mean = None
        self.Y_std = None
        self.Z_mean = None
        self.Z_std = None
        self.mean_all = None
        self.std_all = None
        # Innovation covariance (K x K); used when sampling future MA errors in forecast
        self.sigma_e = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("logger", None)
        state.pop("config", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = get_logger()

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        import statsmodels.api as sm
        from statsmodels.tsa.api import VAR

        self.logger.info("Training VARMA-OLS model...")

        p = self.p
        q = self.q
        long_ar_lags = self.long_ar_lags

        # Normalization: compute mean/std over all trials for z-scoring (avoid div-by-zero later)
        Y_concat = np.concatenate(Y, axis=0)
        self.n_channels_Y = Y_concat.shape[1] if Y_concat.ndim == 2 else 1
        self.Y_mean = np.mean(Y_concat, axis=0, keepdims=True)
        self.Y_std = np.std(Y_concat, axis=0, keepdims=True)
        self.Y_std[self.Y_std < 1e-10] = 1.0

        Z_concat = np.concatenate(Z, axis=0)
        self.n_channels_Z = Z_concat.shape[1] if Z_concat.ndim == 2 else 1
        self.Z_mean = np.mean(Z_concat, axis=0, keepdims=True)
        self.Z_std = np.std(Z_concat, axis=0, keepdims=True)
        self.Z_std[self.Z_std < 1e-10] = 1.0
        self.K = self.n_channels_Y + self.n_channels_Z

        self.logger.info(
            f"VARMA-OLS params: p={p}, q={q}, long_ar_lags={long_ar_lags}, "
            f"n_channels_Y={self.n_channels_Y}, n_channels_Z={self.n_channels_Z}, "
            f"K={self.K}"
        )

        mean_concat = np.concatenate([self.Y_mean, self.Z_mean], axis=1)
        std_concat = np.concatenate([self.Y_std, self.Z_std], axis=1)
        data_concat = np.concatenate([Y_concat, Z_concat], axis=1)

        data_concat_zscored = (data_concat - mean_concat) / std_concat

        # Long-VAR: high-order VAR fit on pooled z-scored data; its residuals proxy MA innovations
        self.logger.info(
            f"Training single Long-VAR on all trials concatenated (T={data_concat.shape[0]})"
        )
        df_concat = pd.DataFrame(data_concat_zscored)
        long_model = VAR(df_concat)
        long_results = long_model.fit(maxlags=long_ar_lags)
        self.long_var_coefs = long_results.coefs
        self.long_var_intercept = long_results.intercept

        # Per-trial design matrices and targets; concatenated into X_full, Y_full for one global fit
        all_X = []
        all_target = []

        for trial_idx in range(len(Y)):
            y_trial = (Y[trial_idx] - self.Y_mean) / self.Y_std

            if Z is not None and trial_idx < len(Z):
                z_trial = (Z[trial_idx] - self.Z_mean) / self.Z_std
                data_trial = np.concatenate([y_trial, z_trial], axis=1)
            else:
                data_trial = y_trial

            T_trial = data_trial.shape[0]
            if T_trial <= long_ar_lags + max(p, q):
                self.logger.warning(
                    f"Trial {trial_idx} too short ({T_trial} samples) for "
                    f"long_ar_lags={long_ar_lags} + max(p,q)={max(p,q)}. Skipping."
                )
                continue

            # Residual proxy e(t) = data(t) - Long-VAR one-step prediction; effective_data = data after dropping Long-VAR burn-in
            resid_proxy = self._get_long_var_residuals(data_trial)
            effective_data = data_trial[long_ar_lags:]

            # First time index we can predict: need p lags of y and q lags of e
            offset = max(p, q)
            target = effective_data[offset:]

            # Build regressors: [y(t-1), ..., y(t-p), e(t-1), ..., e(t-q)]; each lag is (n_times x K)
            X_list = []
            for i in range(1, p + 1):
                lag_start = offset - i
                lag_end = len(effective_data) - i
                X_list.append(effective_data[lag_start:lag_end])
            for i in range(1, q + 1):
                lag_start = offset - i
                lag_end = len(resid_proxy) - i
                X_list.append(resid_proxy[lag_start:lag_end])

            X_matrix = np.concatenate(X_list, axis=1)
            X_matrix = sm.add_constant(X_matrix)

            all_X.append(X_matrix)
            all_target.append(target)

            self.logger.info(
                f"Trial {trial_idx}: T={T_trial}, effective samples={len(target)}"
            )

        if not all_X:
            raise ValueError("No trials were long enough for VARMA-OLS fitting.")

        # Single global fit: stack all trials; beta_ols shape (1 + p*K + q*K, K)
        X_full = np.concatenate(all_X, axis=0)
        Y_full = np.concatenate(all_target, axis=0)

        self.logger.info(
            f"Design matrix shape: {X_full.shape}, Target shape: {Y_full.shape}"
        )

        if self.ridge_alpha > 0.0:
            from sklearn.linear_model import Ridge

            self.logger.info(
                f"Solving via Ridge regression (alpha={self.ridge_alpha})..."
            )
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge.fit(X_full, Y_full)
            self.beta_ols = ridge.coef_.T
        else:
            self.beta_ols = np.linalg.lstsq(X_full, Y_full, rcond=None)[0]

        self.logger.info(
            f"VARMA-OLS training complete. "
            f"Beta shape: {self.beta_ols.shape}, "
            f"n_features={self.beta_ols.shape[0]}, K={self.beta_ols.shape[1]}"
        )

        # Innovation covariance from OLS residuals; regularize if near-singular (for sampling in forecast)
        pred_full = X_full @ self.beta_ols
        resid_full = Y_full - pred_full
        sigma_e = np.cov(resid_full.T)
        if sigma_e.ndim == 0:
            sigma_e = np.array([[sigma_e]])
        sigma_e = np.atleast_2d(sigma_e)
        min_eig = np.min(np.linalg.eigvalsh(sigma_e))
        if min_eig < 1e-8:
            sigma_e = sigma_e + (1e-8 - min_eig) * np.eye(self.K)
        self.sigma_e = sigma_e

        # Optional: scale AR coefficients so companion matrix eigenvalues have modulus <= max_root
        stabilize = getattr(self.config.model, "stabilize_roots", False)
        max_root = float(getattr(self.config.model, "max_root", 0.999))
        if stabilize and max_root < 1.0 and self.p > 0:
            self._stabilize_ar_roots(max_root)

        # Combined mean/std for full K-vector (used when un-z-scoring predictions/forecasts)
        if self.Z_mean is not None:
            self.mean_all = np.concatenate([self.Y_mean, self.Z_mean], axis=1)
            self.std_all = np.concatenate([self.Y_std, self.Z_std], axis=1)
        else:
            self.mean_all = self.Y_mean
            self.std_all = self.Y_std

        return self

    def _stabilize_ar_roots(self, max_root: float) -> None:
        """Scale AR coefficients so the companion matrix has all eigenvalues with modulus <= max_root."""
        p, K = self.p, self.K
        # Extract AR coefficient blocks from beta_ols (rows 1..p*K; col 0 is intercept)
        Phi_list = []
        for i in range(1, p + 1):
            block = self.beta_ols[1 + (i - 1) * K : 1 + i * K, :]
            Phi_list.append(block.T)
        # Companion matrix F: top block row = [Phi_1 .. Phi_p], below = identity blocks
        F = np.zeros((p * K, p * K))
        for i in range(p):
            F[:K, i * K : (i + 1) * K] = Phi_list[i]
        for i in range(p - 1):
            F[(i + 1) * K : (i + 2) * K, i * K : (i + 1) * K] = np.eye(K)
        evals = np.linalg.eigvals(F)
        rho = np.max(np.abs(evals))
        if rho <= max_root:
            return
        scale = max_root / rho
        for i in range(p):
            Phi_list[i] = Phi_list[i] * scale
        for i in range(1, p + 1):
            self.beta_ols[1 + (i - 1) * K : 1 + i * K, :] = Phi_list[i - 1].T
        self.logger.info(f"AR roots stabilized: max modulus {rho:.4f} -> {max_root}")

    def load_from_file(self, model_path: str):
        import pickle

        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        self.__dict__.update(obj.__dict__)
        self.logger.info(f"Model loaded successfully from {model_path}")
        return self

    def _get_long_var_residuals(self, data_zscored: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction errors from the fitted Long-VAR; shape (T - long_ar_lags, K)."""
        L = self.long_ar_lags
        T, K = data_zscored.shape
        if T <= L:
            return np.zeros((0, K))

        preds = np.tile(self.long_var_intercept, (T - L, 1))
        for i in range(1, L + 1):
            preds += data_zscored[L - i : T - i] @ self.long_var_coefs[i - 1].T

        residuals = data_zscored[L:] - preds
        return residuals

    def _predict_trial(self, data_zscored: np.ndarray) -> np.ndarray:
        """
        One-step-ahead prediction for a single trial using the fitted VARMA-OLS model.

        Parameters
        ----------
        data_zscored : (T, K) z-scored data (neural + behavioral stacked)

        Returns
        -------
        predictions : (T, K) one-step-ahead predictions (z-scored)
        """
        import statsmodels.api as sm

        p = self.p
        q = self.q
        long_ar_lags = self.long_ar_lags
        T = data_zscored.shape[0]
        K = data_zscored.shape[1]

        if T <= long_ar_lags + max(p, q):
            self.logger.warning(
                f"Trial too short ({T}) for prediction. Returning zeros."
            )
            return np.zeros_like(data_zscored)

        resid_proxy = self._get_long_var_residuals(data_zscored)
        effective_data = data_zscored[long_ar_lags:]

        offset = max(p, q)
        n_predict = len(effective_data) - offset

        X_list = []
        for i in range(1, p + 1):
            lag_start = offset - i
            lag_end = len(effective_data) - i
            X_list.append(effective_data[lag_start:lag_end])
        for i in range(1, q + 1):
            lag_start = offset - i
            lag_end = len(resid_proxy) - i
            X_list.append(resid_proxy[lag_start:lag_end])

        X_matrix = np.concatenate(X_list, axis=1)
        X_matrix = sm.add_constant(X_matrix)

        predictions_segment = X_matrix @ self.beta_ols

        predictions = np.zeros_like(data_zscored)
        # First index where we have full AR/MA history (after Long-VAR burn-in and offset)
        start_idx = long_ar_lags + offset

        # Warm-up: no full regressors before start_idx; carry previous observation
        if start_idx > 0 and start_idx <= len(data_zscored):
            predictions[0] = data_zscored[0]
            for i in range(1, start_idx):
                predictions[i] = data_zscored[i - 1]

        predictions[start_idx : start_idx + n_predict] = predictions_segment

        return predictions

    def predict(
        self, Y: TrialList, Z: Optional[TrialList] = None
    ) -> Tuple[Optional[TrialList], TrialList, None]:
        """
        One-step-ahead prediction for all trials.
        Returns (Zp, Yp, None) to match the interface.
        """
        all_Yp = []
        all_Zp = [] if self.n_channels_Z > 0 else None

        for trial_idx, y_trial in enumerate(Y):
            y_zscored = (y_trial - self.Y_mean) / self.Y_std

            if self.n_channels_Z > 0:
                # Use actual Z if provided, otherwise use zeros as placeholder
                if Z is not None and trial_idx < len(Z) and Z[trial_idx] is not None:
                    z_trial = Z[trial_idx]
                    if z_trial.shape[0] == y_trial.shape[0]:
                        z_zscored = (z_trial - self.Z_mean) / self.Z_std
                        data_zscored = np.concatenate([y_zscored, z_zscored], axis=1)
                    else:
                        # Length mismatch, use zeros
                        self.logger.warning(
                            f"Trial {trial_idx}: Z length ({z_trial.shape[0]}) doesn't match Y length ({y_trial.shape[0]}). Using zeros."
                        )
                        z_placeholder = np.zeros((y_trial.shape[0], self.n_channels_Z))
                        data_zscored = np.concatenate(
                            [y_zscored, z_placeholder], axis=1
                        )
                else:
                    # Z not provided or None, use zeros as placeholder
                    z_placeholder = np.zeros((y_trial.shape[0], self.n_channels_Z))
                    data_zscored = np.concatenate([y_zscored, z_placeholder], axis=1)
            else:
                data_zscored = y_zscored

            preds_zscored = self._predict_trial(data_zscored)

            Yp_zscored = preds_zscored[:, : self.n_channels_Y]
            Yp = Yp_zscored * self.Y_std + self.Y_mean
            all_Yp.append(Yp)

            if self.n_channels_Z > 0:
                Zp_zscored = preds_zscored[:, self.n_channels_Y :]
                Zp = Zp_zscored * self.Z_std + self.Z_mean
                all_Zp.append(Zp)

        return all_Zp, all_Yp, None

    def validate(self, Y: TrialList, Z: Optional[TrialList] = None) -> Dict[str, Any]:
        Zp, Yp, Xp = self.predict(Y, Z)
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

    def test(self, Y: TrialList, Z: Optional[TrialList] = None) -> Dict[str, Any]:
        return self.validate(Y, Z)

    def _forecast_from_history(
        self,
        history_data: np.ndarray,
        history_resid: np.ndarray,
        n_steps: int,
        sample_future_ma_errors: bool = False,
    ) -> np.ndarray:
        """Recursive m-step forecast. Regressor order: const, y(t-1)..y(t-p), e(t-1)..e(t-q)."""
        p = self.p
        q = self.q
        K = self.K

        # Rolling window of last p observations and q residuals (lists of length-K arrays)
        current_y = list(history_data[-p:])
        current_e = list(history_resid[-q:]) if q > 0 else []

        forecasts = []
        sigma_e = getattr(self, "sigma_e", None)
        use_sampling = (
            sample_future_ma_errors and sigma_e is not None and sigma_e.size > 0
        )

        for _ in range(n_steps):
            # Build row [1, y(t-1), ..., y(t-p), e(t-1), ..., e(t-q)] to match training design
            regressors = [1.0]
            for i in range(1, p + 1):
                regressors.extend(current_y[-i])
            for i in range(1, q + 1):
                if len(current_e) >= i:
                    regressors.extend(current_e[-i])
                else:
                    regressors.extend(np.zeros(K))

            X_t = np.array(regressors)
            y_next = X_t @ self.beta_ols

            forecasts.append(y_next)
            current_y.append(y_next)
            # Future innovation: either zero (deterministic) or N(0, sigma_e)
            if use_sampling:
                current_e.append(np.random.multivariate_normal(np.zeros(K), sigma_e))
            else:
                current_e.append(np.zeros(K))

        return np.array(forecasts)

    def forecast(
        self, m: int, Y_past: Array2D, Z_past: Optional[Array2D] = None
    ) -> Tuple[Optional[Array2D], Array2D, None]:
        """Forecast m steps ahead given past (Y_past, Z_past). Returns (Zf, Yf, None) in original scale."""
        Y_past_zscored = (Y_past - self.Y_mean) / self.Y_std

        if self.n_channels_Z > 0:
            if Z_past is not None:
                Z_past_zscored = (Z_past - self.Z_mean) / self.Z_std
                data_zscored = np.concatenate([Y_past_zscored, Z_past_zscored], axis=1)
            else:
                z_placeholder = np.zeros((Y_past.shape[0], self.n_channels_Z))
                data_zscored = np.concatenate([Y_past_zscored, z_placeholder], axis=1)
        else:
            data_zscored = Y_past_zscored

        long_ar_lags = self.long_ar_lags
        required_len = long_ar_lags + max(self.p, self.q)

        # Pad with repeated first sample if history shorter than needed (avoids flat forecast start from zeros)
        padded_data = data_zscored
        if data_zscored.shape[0] < required_len:
            pad_len = required_len - data_zscored.shape[0]
            first_row = data_zscored[0:1]
            pad_arr = np.tile(first_row, (pad_len, 1))
            padded_data = np.concatenate([pad_arr, data_zscored], axis=0)

        resid_proxy = self._get_long_var_residuals(padded_data)
        effective_data = padded_data[long_ar_lags:]

        # With sample_future_ma_errors=False, future shocks are zero so the forecast can flatten toward the mean; True adds variance.
        sample_future_ma = getattr(self.config.model, "sample_future_ma_errors", False)
        forecasts_zscored = self._forecast_from_history(
            effective_data, resid_proxy, m, sample_future_ma_errors=sample_future_ma
        )

        Yf_zscored = forecasts_zscored[:, : self.n_channels_Y]
        Yf = Yf_zscored * self.Y_std + self.Y_mean

        Zf = None
        if self.n_channels_Z > 0:
            Zf_zscored = forecasts_zscored[:, self.n_channels_Y :]
            Zf = Zf_zscored * self.Z_std + self.Z_mean

        return Zf, Yf, None

    def validate_forecast(
        self,
        Y_list: TrialList,
        Z_list: Optional[TrialList] = None,
        margin: Optional[float] = None,
        Yp_val: Optional[TrialList] = None,
        Zp_val: Optional[TrialList] = None,
    ) -> Dict[str, Any]:
        """Fixed-horizon forecast evaluation: for each trial, take first `history` samples as past, forecast `m` steps, compare to true future."""
        m_seconds = self.forecast_m
        history_seconds = self.forecast_history
        sampling_freq = self.sampling_freq
        m = int(m_seconds * sampling_freq)
        history = int(history_seconds * sampling_freq)

        if Yp_val is None:
            self.logger.info("Yp_val not provided, running prediction...")
            Zp_val, Yp_val, _ = self.predict(Y_list)

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

            if margin is not None and isinstance(margin, (int, float)) and margin > 0:
                history_end = int(margin * sampling_freq)
            else:
                history_end = history

            Y_past = Y[:history_end]
            Y_future_true = Y[history_end : history_end + m]

            Z = Z_list[idx] if Z_list is not None and idx < len(Z_list) else None
            Z_future_true = Z[history_end : history_end + m] if Z is not None else None
            Z_past = Z[:history_end] if Z is not None else None

            Zf, Yf, _ = self.forecast(m, Y_past, Z_past=Z_past)

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
            results["X_future_pred"].append(None)
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


class VARMAOLSFramework(BaseFramework):
    def _initialize_model(self):
        return VARMAOLSWrapper(self.config)
