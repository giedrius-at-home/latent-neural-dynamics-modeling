from typing import Any, Dict, List, Optional, Tuple
from numpy.typing import NDArray
from utils.config import Config
from utils.logger import get_logger
from utils.miscellaneous import state_shape
from utils.stats import pearson_r_per_channel
import numpy as np
import math
import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

psid_path = Path(__file__).parent.parent / "PSID"
if str(psid_path.parent) not in sys.path:
    sys.path.insert(0, str(psid_path.parent))

dpad_path = Path(__file__).parent.parent / "DPAD-main" / "source"
if str(dpad_path) not in sys.path:
    sys.path.insert(0, str(dpad_path))

Array2D = NDArray[np.float64]
TrialList = List[Array2D]


def _apply_hamming_trial_edge_taper(data: np.ndarray, n_taper: int) -> np.ndarray:
    """
    Apply Hamming-shaped edge taper to the first and last n_taper samples of a trial.
    Reduces power at segment boundaries so concatenated trials have smooth transitions.

    Parameters
    ----------
    data : (T, K) array
        Per-trial data (e.g. z-scored).
    n_taper : int
        Number of samples to taper at each end (0 = no taper).

    Returns
    -------
    (T, K) array
        Data with first and last n_taper samples scaled by Hamming ramps (0->1, 1->0).
    """
    if n_taper <= 0:
        return data
    T, K = data.shape
    if T <= 2 * n_taper:
        n_taper = max(0, (T - 1) // 2)
        if n_taper <= 0:
            return data
    out = np.array(data, copy=True, dtype=np.float64)
    # Hamming-shaped ramp 0 -> 1 over first n_taper samples (first half of Hamming(2*n_taper) normalized)
    h = np.hamming(2 * n_taper)
    ramp_up = (h[:n_taper] - h[0]) / (h[n_taper - 1] - h[0])
    ramp_down = (h[n_taper:] - h[-1]) / (h[n_taper] - h[-1])
    for ch in range(K):
        out[:n_taper, ch] *= ramp_up
        out[-n_taper:, ch] *= ramp_down
    return out


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
        if (
            hasattr(self.idSys, "A")
            and self.idSys.A is not None
            and len(self.idSys.A_powers_cache) == 0
        ):
            A = np.array(self.idSys.A)
            # Precompute up to 200 (2 seconds * 80 Hz = 160, 200 is reasonable)
            max_m = 200

            self.idSys.A_powers_cache = [A.copy()]
            for i in range(2, max_m + 1):
                self.idSys.A_powers_cache.append(self.idSys.A_powers_cache[-1] @ A)
            self.idSys.max_m_precomputed = max_m
            self.logger.info(
                f"Precomputed A matrix powers up to m={max_m} after loading model"
            )
        return self.idSys

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        from PSID.PSID import PSID as PSIDClass

        nx: int = self.config.model.nx
        n1: int = self.config.model.n1
        i: int = self.config.model.i
        time_first: bool = self.config.model.time_first

        backward_kalman: bool = getattr(self.config.model, "backward_kalman", False)
        rescale_states: bool = getattr(self.config.model, "rescale_states", True)
        max_eigenvalue = getattr(self.config.model, "max_eigenvalue", 0.995)
        if max_eigenvalue is None:
            max_eigenvalue = 1.0
        else:
            max_eigenvalue = float(max_eigenvalue)

        self.logger.info(
            f"Calling PSID.PSID with nx={nx}, n1={n1}, i={i}, time_first={time_first}, "
            f"backward_kalman={backward_kalman}, "
            f"rescale_states={rescale_states}, max_eigenvalue={max_eigenvalue}"
        )

        psid_kwargs = {
            "zscore_Y": True,
            "zscore_Z": True,
            "remove_mean_Y": True,
            "remove_mean_Z": True,
            "time_first": time_first,
            "backward_kalman": backward_kalman,
            "rescale_states": rescale_states,
            "max_eigenvalue": max_eigenvalue,
        }

        self.idSys = PSIDClass(Y, Z, nx, n1, i, **psid_kwargs)
        if hasattr(self.idSys, "A") and self.idSys.A is not None:
            A = np.array(self.idSys.A)
            max_m = 200

            self.idSys.A_powers_cache = [A.copy()]
            for i in range(2, max_m + 1):
                self.idSys.A_powers_cache.append(self.idSys.A_powers_cache[-1] @ A)
            self.idSys.max_m_precomputed = max_m
            self.logger.info(f"Precomputed A matrix powers up to m={max_m}")
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

    def forecast(
        self,
        m: int,
        Y_past: Array2D,
        Z_past: Optional[Array2D] = None,
    ):
        """
        Efficient m-step ahead forecast using state-space model.

        Complexity: O(len(Y_past) + m) instead of O(m * (len(Y_past) + m))

        Algorithm:
        1. Run Kalman filter once on Y_past to get initial state estimate
        2. Iterate state equation forward m times: x_{t+1} = A @ x_t
        3. Compute outputs: y_t = C @ x_t, z_t = Cz @ x_t

        ``Z_past`` must be passed when the fitted model uses behavioral inputs (same layout
        as ``validate_forecast`` / ``idSys.predict(Y, U=Z)``). Omitting U made multi-step Z
        forecasts nearly flat while one-step Zp remained well scaled.
        """
        if self.idSys is None:
            raise ValueError(
                "Model not initialized. Call train() or load_from_file() first."
            )

        # Step 1: Get initial state estimate by running Kalman filter once on past data
        # Use backward Kalman smoothing if available for better initial state estimate
        use_smoothing = getattr(self.idSys, "backward_kalman", False)
        # Convert Y_past to list format expected by predict
        Y_past_list = [Y_past]
        U_list = [Z_past] if Z_past is not None else None
        Zp_past, Yp_past, Xp_past = self.idSys.predict(
            Y_past_list, U=U_list, useSmoothing=use_smoothing
        )

        # Extract the final state estimate (last time step)
        if Xp_past is None or len(Xp_past) == 0:
            raise ValueError("Could not extract state estimate from past data")

        Xp_past_array = np.asarray(Xp_past[0]) if isinstance(Xp_past, list) else Xp_past
        if Xp_past_array.shape[0] == 0:
            raise ValueError("State estimate is empty")

        x0 = Xp_past_array[-1, :]  # Final state estimate: shape (nx,)

        # Get state-space matrices
        A = np.array(self.idSys.A)  # State transition matrix: (nx, nx)
        C = np.array(self.idSys.C)  # Output matrix for Y: (ny, nx)
        Cz = (
            np.array(self.idSys.Cz)
            if hasattr(self.idSys, "Cz") and self.idSys.Cz is not None
            else None
        )  # Output matrix for Z: (nz, nx)

        nx = A.shape[0]
        ny = C.shape[0]
        nz = Cz.shape[0] if Cz is not None else 0

        # Step 2: Iterate state equation forward m times
        # Use precomputed A powers if available for efficiency
        use_cache = (
            hasattr(self.idSys, "A_powers_cache")
            and len(self.idSys.A_powers_cache) > 0
            and hasattr(self.idSys, "max_m_precomputed")
            and m <= self.idSys.max_m_precomputed
        )

        if use_cache:
            # Efficient: compute all states at once using A^m powers
            # A_powers_cache[0] = A^1, A_powers_cache[1] = A^2, ..., A_powers_cache[t-1] = A^t
            # For state at step t: x_t = A^t @ x0
            Xf = np.zeros((m, nx))
            for t in range(1, m + 1):
                if t <= len(self.idSys.A_powers_cache):
                    # A_powers_cache[t-1] is A^t
                    Xf[t - 1, :] = self.idSys.A_powers_cache[t - 1] @ x0
                else:
                    # Fallback to iterative if beyond cache (shouldn't happen if m <= max_m_precomputed)
                    if t == 1:
                        Xf[t - 1, :] = A @ x0
                    else:
                        Xf[t - 1, :] = A @ Xf[t - 2, :]
        else:
            # Iterative forward propagation
            Xf = np.zeros((m, nx))
            x_current = x0.copy()
            for t in range(m):
                x_current = A @ x_current
                Xf[t, :] = x_current

        # Step 3: Compute outputs from states
        Yf = Xf @ C.T  # (m, nx) @ (nx, ny) -> (m, ny)

        Zf = None
        if Cz is not None and nz > 0:
            Zf = Xf @ Cz.T  # (m, nx) @ (nx, nz) -> (m, nz)

        return Zf, Yf, Xf

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

            Zf, Yf, Xf = self.forecast(m, Y_past, Z_past)

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


def _make_dpad_epoch_callback(csv_path, checkpoint_dir, save_every):
    import tensorflow as tf

    class _Cb(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if not logs:
                return
            loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            header = not csv_path.exists()
            with open(csv_path, "a") as f:
                if header:
                    f.write("epoch,loss,val_loss,rmse,val_rmse\n")
                rmse = math.sqrt(loss) if loss else ""
                val_rmse = math.sqrt(val_loss) if val_loss else ""
                f.write(f"{epoch},{loss},{val_loss},{rmse},{val_rmse}\n")

            if save_every and (epoch + 1) % save_every == 0:
                np.savez(
                    checkpoint_dir / f"weights_epoch_{epoch + 1}.npz",
                    *self.model.get_weights(),
                )

    return _Cb()


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
        fast = getattr(self.config.model, "fast", False)
        checkpoint_every = getattr(self.config.model, "checkpoint_every", 100)

        self.logger.info(
            f"Training DPAD with nx={nx}, n1={n1}, method_code={method_code}, epochs={epochs}"
        )
        Y_dpad = [y.T for y in Y]
        Z_dpad = [z.T for z in Z] if Z is not None else None

        save_dir = Path(self.config.results.save_dir)
        checkpoint_dir = save_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / "training_metrics.csv"

        cb = _make_dpad_epoch_callback(csv_path, checkpoint_dir, checkpoint_every)

        self.idSys = DPADModel(log_dir=self.config.results.log_dir)
        args = DPADModel.prepare_args(method_code)

        self.idSys.fit(
            Y_dpad,
            Z=Z_dpad,
            nx=nx,
            n1=n1,
            epochs=epochs,
            skip_predictions=fast,
            callbacks=[cb],
            **args,
        )

        self._save_training_history(save_dir)
        return self.idSys

    def _save_training_history(self, save_dir: Path):
        logs = getattr(self.idSys, "logs", {})
        if not logs:
            return

        history = {}
        for stage_name, stage_log in logs.items():
            if not isinstance(stage_log, dict):
                continue
            keras_hist = stage_log.get("history", {})
            entry = {
                "epochs": stage_log.get("epoch", []),
                "fit_time_s": stage_log.get("fit_time"),
            }
            for metric_name in ("loss", "val_loss"):
                values = keras_hist.get(metric_name, [])
                entry[metric_name] = values
                entry[metric_name.replace("loss", "rmse")] = [
                    math.sqrt(v) if v is not None and not math.isnan(v) else None
                    for v in values
                ]
            history[stage_name] = entry

        history_path = save_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        self.logger.info(f"Saved training history to {history_path}")

    def predict(self, Y: TrialList, Z: Optional[TrialList] = None):
        all_Zp, all_Yp, all_Xp = [], [], []

        self.idSys.set_steps_ahead([1])
        self.idSys.set_multi_step_with_data_gen(False)
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

            Zp, Yp, Xp = self.idSys.predict(y_trial_padded)

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

        if Yp_val is None:
            Zp_val_local, Yp_val_local, Xp_val_local = self.predict(Y_list)
            Yp_val = Yp_val_local
            Zp_val = Zp_val_local

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

            Zf, Yf, Xf = self.forecast(m, Y_history)

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
    VARMA(p, q) estimation via Ridge regression using a Long-VAR residual proxy.

    All channels (neural + behavioral) are modeled jointly in one multivariate
    system, capturing cross-channel dynamics bidirectionally.

    Algorithm:
        1. Fit a high-order VAR to approximate the process and recover residuals.
        2. Build a design matrix: [intercept, AR lags of data, MA lags of residuals].
        3. Solve via Ridge regression (alpha=0.01) — regularized fit per output channel.
        4. Forecast recursively: feed predictions back, future errors = 0.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()

        # VARMA orders: p = AR lags, q = MA lags; long_ar_lags = order of Long-VAR used for residual proxy
        self.p = getattr(config.model, "p", 20)
        self.q = getattr(config.model, "q", 1)
        self.long_ar_lags = getattr(config.model, "long_ar_lags", 30)

        # Forecast horizon (seconds) and history length (samples) for validate_forecast
        self.forecast_m = getattr(config.model.forecast, "m", 2.0)
        self.forecast_history = getattr(config.model.forecast, "history", 5.0)
        self.sampling_freq = getattr(config.data, "sampling_frequency", 80)
        # Trial edge taper (seconds) at boundaries when concatenating; 0 = no taper
        self.trial_edge_taper_sec = getattr(config.model, "trial_edge_taper_sec", 0.0)

        # Fitted state: beta maps regressors -> K outputs; K = n_channels_Y + n_channels_Z
        self.beta = None
        self.n_channels_Y = None
        self.n_channels_Z = None
        self.K = None

        # Per-channel normalization: store as dicts for clarity
        # Keys are channel indices (0, 1, 2, ...), values are mean/std for that channel
        self.channel_means = {}  # {channel_idx: mean_value}
        self.channel_stds = {}  # {channel_idx: std_value}
        self.channel_idx_to_type = {}  # {channel_idx: 'Y' or 'Z'} for reference

        # Keep old attributes for backward compatibility, but compute from dicts
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

    def _normalize_trial(
        self, data: np.ndarray, channel_offset: int, n_channels: int
    ) -> np.ndarray:
        """
        Apply per-channel z-scoring to a trial's data.

        Parameters
        ----------
        data : (T, n_channels) array
        channel_offset : int
            Starting channel index in global channel numbering
        n_channels : int
            Number of channels to normalize

        Returns
        -------
        data_zscored : (T, n_channels) normalized array
        """
        data_zscored = np.zeros_like(data)
        for ch_idx in range(n_channels):
            global_ch_idx = channel_offset + ch_idx
            mean_ch = self.channel_means[global_ch_idx]
            std_ch = self.channel_stds[global_ch_idx]
            data_zscored[:, ch_idx] = (data[:, ch_idx] - mean_ch) / std_ch
        return data_zscored

    def _denormalize_trial(
        self, data_zscored: np.ndarray, channel_offset: int, n_channels: int
    ) -> np.ndarray:
        """
        Reverse per-channel z-scoring.

        Parameters
        ----------
        data_zscored : (T, n_channels) normalized array
        channel_offset : int
            Starting channel index in global channel numbering
        n_channels : int
            Number of channels to denormalize

        Returns
        -------
        data : (T, n_channels) denormalized array
        """
        data = np.zeros_like(data_zscored)
        for ch_idx in range(n_channels):
            global_ch_idx = channel_offset + ch_idx
            mean_ch = self.channel_means[global_ch_idx]
            std_ch = self.channel_stds[global_ch_idx]
            data[:, ch_idx] = data_zscored[:, ch_idx] * std_ch + mean_ch
        return data

    def _build_design_matrix(
        self,
        effective_data: np.ndarray,
        resid_proxy: np.ndarray,
        p: int,
        q: int,
        offset: int,
    ) -> np.ndarray:
        """
        Build VARMA design matrix: [const, AR lags, MA lags].

        Parameters
        ----------
        effective_data : (T, K) data after Long-VAR burn-in
        resid_proxy : (T, K) residual proxy
        p : int
            AR order
        q : int
            MA order
        offset : int
            Offset for building lags

        Returns
        -------
        X_matrix : (T-offset, 1+p*K+q*K) design matrix with constant
        """
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
        return X_matrix

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        self.logger.info("Training VARMA-Ridge model (per-channel z-scoring)...")

        p = self.p
        q = self.q
        long_ar_lags = self.long_ar_lags

        # Normalization: compute mean/std per channel over all trials
        Y_concat = np.concatenate(Y, axis=0)
        self.n_channels_Y = Y_concat.shape[1] if Y_concat.ndim == 2 else 1

        # Compute per-channel stats for Y
        Y_mean_per_ch = np.mean(Y_concat, axis=0)
        Y_std_per_ch = np.std(Y_concat, axis=0)
        Y_std_per_ch[Y_std_per_ch < 1e-10] = 1.0

        # Store in dict: Y channels are 0 to n_channels_Y-1
        for ch_idx in range(self.n_channels_Y):
            self.channel_means[ch_idx] = float(Y_mean_per_ch[ch_idx])
            self.channel_stds[ch_idx] = float(Y_std_per_ch[ch_idx])
            self.channel_idx_to_type[ch_idx] = "Y"

        # Normalize Z (behavioral/output channels)
        Z_concat = np.concatenate(Z, axis=0)
        self.n_channels_Z = Z_concat.shape[1] if Z_concat.ndim == 2 else 1

        # Compute per-channel stats for Z
        Z_mean_per_ch = np.mean(Z_concat, axis=0)
        Z_std_per_ch = np.std(Z_concat, axis=0)
        Z_std_per_ch[Z_std_per_ch < 1e-10] = 1.0

        # Store in dict: Z channels start at n_channels_Y
        for ch_idx in range(self.n_channels_Z):
            global_ch_idx = self.n_channels_Y + ch_idx
            self.channel_means[global_ch_idx] = float(Z_mean_per_ch[ch_idx])
            self.channel_stds[global_ch_idx] = float(Z_std_per_ch[ch_idx])
            self.channel_idx_to_type[global_ch_idx] = "Z"

        self.K = self.n_channels_Y + self.n_channels_Z

        # Create arrays for backward compatibility (keepdims=True for broadcasting)
        self.Y_mean = Y_mean_per_ch.reshape(1, -1)
        self.Y_std = Y_std_per_ch.reshape(1, -1)
        self.Z_mean = Z_mean_per_ch.reshape(1, -1)
        self.Z_std = Z_std_per_ch.reshape(1, -1)

        # Log normalization statistics per channel
        self.logger.info(
            f"VARMA-Ridge params: p={p}, q={q}, long_ar_lags={long_ar_lags}, "
            f"n_channels_Y={self.n_channels_Y}, n_channels_Z={self.n_channels_Z}, "
            f"K={self.K}"
        )
        self.logger.info(
            f"Y normalization: mean range [{Y_mean_per_ch.min():.4f}, {Y_mean_per_ch.max():.4f}], "
            f"std range [{Y_std_per_ch.min():.4f}, {Y_std_per_ch.max():.4f}]"
        )
        self.logger.info(
            f"Z normalization: mean range [{Z_mean_per_ch.min():.4f}, {Z_mean_per_ch.max():.4f}], "
            f"std range [{Z_std_per_ch.min():.4f}, {Z_std_per_ch.max():.4f}]"
        )

        # Option A: z-score each trial with raw means/stds, then apply Hamming edge taper (reduces power at boundaries)
        n_taper = 0
        if self.trial_edge_taper_sec > 0:
            n_taper = max(1, int(round(self.sampling_freq * self.trial_edge_taper_sec)))
            self.logger.info(
                f"Trial edge taper: Hamming, {self.trial_edge_taper_sec}s -> n_taper={n_taper} samples"
            )

        tapered_trials: List[Array2D] = []
        for trial_idx in range(len(Y)):
            y_trial = Y[trial_idx]
            y_trial_zscored = self._normalize_trial(
                y_trial, channel_offset=0, n_channels=self.n_channels_Y
            )

            if Z is not None and trial_idx < len(Z):
                z_trial = Z[trial_idx]
                z_trial_zscored = self._normalize_trial(
                    z_trial,
                    channel_offset=self.n_channels_Y,
                    n_channels=self.n_channels_Z,
                )
                data_trial = np.concatenate([y_trial_zscored, z_trial_zscored], axis=1)
            else:
                data_trial = y_trial_zscored
            data_trial = _apply_hamming_trial_edge_taper(data_trial, n_taper)
            tapered_trials.append(data_trial)

        # Long-VAR on concatenated tapered z-scored data
        data_concat_zscored = np.concatenate(tapered_trials, axis=0)
        self.logger.info(
            f"Training single Long-VAR on all trials concatenated (T={data_concat_zscored.shape[0]})"
        )
        df_concat = pd.DataFrame(data_concat_zscored)
        long_model = VAR(df_concat)
        long_results = long_model.fit(maxlags=long_ar_lags)
        self.long_var_coefs = long_results.coefs
        self.long_var_intercept = long_results.intercept

        # Per-trial design matrices and targets; use same tapered z-scored trials
        all_X = []
        all_target = []

        for trial_idx in range(len(tapered_trials)):
            data_trial = tapered_trials[trial_idx]
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

            # Build design matrix
            X_matrix = self._build_design_matrix(
                effective_data, resid_proxy, p, q, offset
            )

            all_X.append(X_matrix)
            all_target.append(target)

            self.logger.info(
                f"Trial {trial_idx}: T={T_trial}, effective samples={len(target)}"
            )

        if not all_X:
            raise ValueError("No trials were long enough for VARMA-Ridge fitting.")

        # Single global fit: stack all trials; beta shape (1 + p*K + q*K, K)
        X_full = np.concatenate(all_X, axis=0)
        Y_full = np.concatenate(all_target, axis=0)

        self.logger.info(
            f"Design matrix shape: {X_full.shape}, Target shape: {Y_full.shape}"
        )

        # Fit Ridge for each output channel separately
        self.beta = np.zeros((X_full.shape[1], self.K))
        for ch_idx in range(self.K):
            ridge = Ridge(alpha=0.01, fit_intercept=False)
            ridge.fit(X_full, Y_full[:, ch_idx])
            self.beta[:, ch_idx] = ridge.coef_

        self.logger.info(
            f"VARMA-Ridge training complete. "
            f"Beta shape: {self.beta.shape}, "
            f"n_features={self.beta.shape[0]}, K={self.beta.shape[1]}"
        )

        pred_full = X_full @ self.beta
        resid_full = Y_full - pred_full
        sigma_e = np.cov(resid_full.T)
        if sigma_e.ndim == 0:
            sigma_e = np.array([[sigma_e]])
        sigma_e = np.atleast_2d(sigma_e)
        min_eig = np.min(np.linalg.eigvalsh(sigma_e))
        if min_eig < 1e-8:
            sigma_e = sigma_e + (1e-8 - min_eig) * np.eye(self.K)
        self.sigma_e = sigma_e

        max_root = float(getattr(self.config.model, "max_root", 0.999))
        if self.p > 0:
            self._stabilize_ar_roots(max_root)

        if self.Z_mean is not None:
            self.mean_all = np.concatenate([self.Y_mean, self.Z_mean], axis=1)
            self.std_all = np.concatenate([self.Y_std, self.Z_std], axis=1)
        else:
            self.mean_all = self.Y_mean
            self.std_all = self.Y_std

        return self

    def _build_companion_matrix(self, beta, p, K) -> np.ndarray:
        """Build companion matrix from beta AR coefficients."""
        companion = np.zeros((p * K, p * K))
        for i in range(p):
            A_i_T = beta[1 + i * K : 1 + (i + 1) * K, :]
            A_i = A_i_T.T  # AR matrix is the transpose of the coefficient block
            companion[0:K, i * K : (i + 1) * K] = A_i
        if p > 1:
            companion[K:, :-K] = np.eye((p - 1) * K)
        return companion

    def _stabilize_ar_roots(self, max_root: float) -> None:
        """
        Scale AR coefficients so the companion matrix has all eigenvalues with modulus <= max_root.
        Uses lag-dependent scaling (gamma^lag) to preserve relative importance of different lags,
        """
        p, K = self.p, self.K
        if p == 0:
            return

        # Build companion matrix to check current eigenvalues
        companion = self._build_companion_matrix(self.beta, p, K)
        eigenvalues = np.linalg.eigvals(companion)
        max_pole = np.max(np.abs(eigenvalues))

        if max_pole > 0:
            gamma = max_root / max_pole

            # If gamma != 1, we multiply each AR(lag) matrix by gamma^lag
            # This exactly shifts the largest pole to target_pole_radius
            for i in range(p):
                lag = i + 1
                self.beta[1 + i * K : 1 + (i + 1) * K, :] *= gamma**lag

            # Verify the stabilization worked
            companion_reg = self._build_companion_matrix(self.beta, p, K)
            eigenvalues_reg = np.linalg.eigvals(companion_reg)
            max_pole_reg = np.max(np.abs(eigenvalues_reg))
            self.logger.info(
                f"AR roots stabilized: max modulus {max_pole:.4f} -> {max_pole_reg:.4f} "
                f"(target: {max_root:.4f}, gamma: {gamma:.6f})"
            )

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
        One-step-ahead prediction for a single trial using the fitted VARMA-Ridge model.

        Parameters
        ----------
        data_zscored : (T, K) z-scored data (neural + behavioral stacked)

        Returns
        -------
        predictions : (T, K) one-step-ahead predictions (z-scored)
        """
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

        X_matrix = self._build_design_matrix(effective_data, resid_proxy, p, q, offset)

        predictions_segment = X_matrix @ self.beta

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
        Per-channel z-scoring applied.
        Returns (Zp, Yp, None) to match the interface.
        """
        all_Yp = []
        all_Zp = [] if self.n_channels_Z > 0 else None

        for trial_idx, y_trial in enumerate(Y):
            # Apply per-channel z-scoring to Y
            y_trial_zscored = self._normalize_trial(
                y_trial, channel_offset=0, n_channels=self.n_channels_Y
            )

            if self.n_channels_Z > 0:
                # Use actual Z if provided, otherwise use zeros as placeholder
                if Z is not None and trial_idx < len(Z) and Z[trial_idx] is not None:
                    z_trial = Z[trial_idx]
                    if z_trial.shape[0] == y_trial.shape[0]:
                        # Apply per-channel z-scoring to Z
                        z_trial_zscored = self._normalize_trial(
                            z_trial,
                            channel_offset=self.n_channels_Y,
                            n_channels=self.n_channels_Z,
                        )
                        data_zscored = np.concatenate(
                            [y_trial_zscored, z_trial_zscored], axis=1
                        )
                    else:
                        # Length mismatch, use zeros
                        self.logger.warning(
                            f"Trial {trial_idx}: Z length ({z_trial.shape[0]}) doesn't match Y length ({y_trial.shape[0]}). Using zeros."
                        )
                        z_placeholder = np.zeros((y_trial.shape[0], self.n_channels_Z))
                        data_zscored = np.concatenate(
                            [y_trial_zscored, z_placeholder], axis=1
                        )
                else:
                    # Z not provided or None, use zeros as placeholder
                    z_placeholder = np.zeros((y_trial.shape[0], self.n_channels_Z))
                    data_zscored = np.concatenate(
                        [y_trial_zscored, z_placeholder], axis=1
                    )
            else:
                data_zscored = y_trial_zscored

            preds_zscored = self._predict_trial(data_zscored)

            # Un-scale per-channel
            Yp_zscored = preds_zscored[:, : self.n_channels_Y]
            Yp = self._denormalize_trial(
                Yp_zscored, channel_offset=0, n_channels=self.n_channels_Y
            )
            all_Yp.append(Yp)

            if self.n_channels_Z > 0:
                Zp_zscored = preds_zscored[:, self.n_channels_Y :]
                Zp = self._denormalize_trial(
                    Zp_zscored,
                    channel_offset=self.n_channels_Y,
                    n_channels=self.n_channels_Z,
                )
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
    ) -> np.ndarray:
        """Recursive m-step forecast. Regressor order: const, y(t-1)..y(t-p), e(t-1)..e(t-q)."""
        p = self.p
        q = self.q
        K = self.K

        # Rolling window of last p observations and q residuals (lists of length-K arrays)
        current_y = list(history_data[-p:])
        current_e = list(history_resid[-q:]) if q > 0 else []

        forecasts = []

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
            y_next = X_t @ self.beta

            forecasts.append(y_next)
            current_y.append(y_next)
            # Future innovation: always zero (deterministic forecast)
            current_e.append(np.zeros(K))

        return np.array(forecasts)

    def forecast(
        self, m: int, Y_past: Array2D, Z_past: Optional[Array2D] = None
    ) -> Tuple[Optional[Array2D], Array2D, None]:
        """
        Forecast m steps ahead given past (Y_past, Z_past). Returns (Zf, Yf, None) in original scale.
        Per-channel z-scoring applied.
        """
        # Apply per-channel z-scoring to Y_past
        Y_past_zscored = self._normalize_trial(
            Y_past, channel_offset=0, n_channels=self.n_channels_Y
        )

        if self.n_channels_Z > 0:
            if Z_past is not None:
                # Apply per-channel z-scoring to Z_past
                Z_past_zscored = self._normalize_trial(
                    Z_past,
                    channel_offset=self.n_channels_Y,
                    n_channels=self.n_channels_Z,
                )
                data_zscored = np.concatenate([Y_past_zscored, Z_past_zscored], axis=1)
            else:
                z_placeholder = np.zeros((Y_past.shape[0], self.n_channels_Z))
                data_zscored = np.concatenate([Y_past_zscored, z_placeholder], axis=1)
        else:
            data_zscored = Y_past_zscored

        p = self.p
        q = self.q
        long_ar_lags = self.long_ar_lags

        # No padding - just use what we have
        if data_zscored.shape[0] < p:
            raise ValueError(
                f"History too short: need at least {p} samples for AR order, got {data_zscored.shape[0]}"
            )

        # Get the actual last p samples
        history_data = data_zscored[-p:]

        # Compute residuals from the actual data (not padded)
        # We need at least long_ar_lags samples to compute residuals
        if data_zscored.shape[0] >= long_ar_lags:
            resid_proxy = self._get_long_var_residuals(data_zscored)
            # Get the last q residuals (like supervisor: history_resid[-q:])
            if resid_proxy.shape[0] >= q:
                history_resid = resid_proxy[-q:]
            else:
                # If not enough residuals, use zeros (like supervisor does for future errors)
                history_resid = np.zeros((q, self.K))
        else:
            # If not enough data for long-VAR, use zeros for residuals
            history_resid = np.zeros((q, self.K))

        # Forecast using actual history (no padding, no effective_data slicing)
        forecasts_zscored = self._forecast_from_history(
            history_data, history_resid, m
        )

        # Un-scale per-channel
        Yf_zscored = forecasts_zscored[:, : self.n_channels_Y]
        Yf = self._denormalize_trial(
            Yf_zscored, channel_offset=0, n_channels=self.n_channels_Y
        )

        Zf = None
        if self.n_channels_Z > 0:
            Zf_zscored = forecasts_zscored[:, self.n_channels_Y :]
            Zf = self._denormalize_trial(
                Zf_zscored,
                channel_offset=self.n_channels_Y,
                n_channels=self.n_channels_Z,
            )

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
