from typing import Any, Dict, List, Optional, Tuple
from numpy.typing import NDArray
from utils.config import Config
from utils.logger import get_logger
from utils.miscellaneous import state_shape
from utils.stats import pearson_r_per_channel
import numpy as np
import sys
from pathlib import Path

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
    ) -> Dict[str, Any]:
        self.logger.info("Starting forecast validation...")
        return self.model.validate_forecast(Y_list, Z_list=Z_list, margin=margin)


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
    ) -> Dict[str, Any]:

        m_seconds = self.config.model.forecast.m
        history_seconds = self.config.model.forecast.history
        sampling_freq = self.config.data.sampling_frequency
        m = int(m_seconds * sampling_freq)
        history = int(history_seconds * sampling_freq)
        margin_sec = margin if margin is not None else 0.0

        Zp_val, Yp_val, Xp_val = self.predict(Y_list)

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

        consistency_loss_weight: float = getattr(
            self.config.model,
            "consistency_loss_weight",
            getattr(self.config.model, "alpha_behavior", 0.0),
        )

        self.logger.info(
            f"Training DPAD with nx={nx}, n1={n1}, method_code={method_code}, epochs={epochs}, "
            f"consistency_loss_weight={consistency_loss_weight}"
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

        parsed_alpha = args.pop("consistency_loss_weight", 0.0)
        final_alpha = (
            consistency_loss_weight
            if getattr(self.config.model, "alpha_behavior", None) is not None
            or getattr(self.config.model, "consistency_loss_weight", None) is not None
            else parsed_alpha
        )

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

        self.idSys.fit(
            Y_dpad,
            Z=Z_dpad,
            nx=nx,
            n1=n1,
            epochs=epochs,
            consistency_loss_weight=final_alpha,
            loss_name=final_loss,
            behavior_loss_weight=final_bw,
            recon_loss_weight=final_rw,
            early_stopping_measure=final_esm,
            early_stopping_patience=final_esp,
            start_from_epoch_rnn=final_esmin,
            skip_predictions=fast,
            tb_make_prediction_plots=getattr(
                self.config.model, "tb_make_prediction_plots", False
            ),
            tb_make_prediction_scatters=getattr(
                self.config.model, "tb_make_prediction_scatters", False
            ),
            tb_plot_epoch_mod=getattr(self.config.model, "tb_plot_epoch_mod", 20),
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
    ) -> Dict[str, Any]:
        m_seconds = self.config.model.forecast.m
        history_seconds = self.config.model.forecast.history
        sampling_freq = self.config.data.sampling_frequency
        m = int(m_seconds * sampling_freq)
        history = int(history_seconds * sampling_freq)

        self.idSys.set_steps_ahead([1])
        self.idSys.set_multi_step_with_data_gen(False)

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
