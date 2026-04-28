from typing import Any, Dict, List, Optional, Tuple
from numpy.typing import NDArray
from utils.config import Config
from utils.logger import get_logger
from utils.miscellaneous import state_shape
from utils.stats import pearson_r_per_channel
import numpy as np
import math
import json
from pathlib import Path
import pandas as pd
import PSID
from scipy.linalg import solve_discrete_are
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

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

    def _predict(self, Y: TrialList, Z: Optional[TrialList] = None):
        self.logger.info("Running prediction on provided data...")
        return self.model.predict(Y, Z)

    def _forecast(self, m: int, Y_past: Array2D):
        self.logger.info(f"Running {m}-step ahead forecast...")
        return self.model.forecast(m, Y_past)

    def _evaluate_forecast(
        self,
        Y_list: TrialList,
        Z_list: Optional[TrialList] = None,
        margin: Optional[float] = None,
        Yp_val: Optional[TrialList] = None,
        Zp_val: Optional[TrialList] = None,
    ) -> Dict[str, Any]:
        """Unified multi-step forecast eval loop for all frameworks.

        Per trial: split into ``history_end`` samples of past + ``m`` samples of
        future, call ``self.model.forecast(m, Y_past)``, compare forecast
        ``(Yf, Zf)`` against true future. Returns per-trial lists of arrays
        plus aggregate Pearson means. ``margin`` overrides the history window
        in samples when >0 (used by chunk-margin training).
        """
        self.logger.info("Starting forecast validation...")
        m_seconds = self.config.model.forecast.m
        history_seconds = self.config.model.forecast.history
        sampling_freq = self.config.data.sampling_frequency
        m = int(m_seconds * sampling_freq)
        history = int(history_seconds * sampling_freq)

        results: Dict[str, Any] = {
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
                    f"history + m ({history} + {m} = {history + m}) must not "
                    f"exceed trial length T={T}"
                )
            history_end = (
                int(margin * sampling_freq)
                if (margin is not None and margin > 0)
                else history
            )
            Y_past = Y[:history_end]
            Y_future_true = Y[history_end : history_end + m]
            Z = Z_list[idx] if Z_list is not None and idx < len(Z_list) else None
            Z_past = Z[:history_end] if Z is not None else None
            Z_future_true = Z[history_end : history_end + m] if Z is not None else None

            Zf, Yf, Xf = self.model.forecast(m, Y_past, Z_past=Z_past)

            Y_concat = np.concatenate([Y_past, Yf], axis=0) if Yf is not None else None
            Z_concat = (
                np.concatenate([Z_past, Zf], axis=0)
                if Z_past is not None and Zf is not None
                else None
            )

            r_list, _ = pearson_r_per_channel([Y_future_true], [Yf] if Yf is not None else [Y_future_true])
            r_list = r_list[0] if isinstance(r_list, list) and len(r_list) > 0 else r_list
            r_list_Z: Any = []
            if Z_future_true is not None and Zf is not None:
                r_list_Z, _ = pearson_r_per_channel([Z_future_true], [Zf])
                r_list_Z = r_list_Z[0] if isinstance(r_list_Z, list) and len(r_list_Z) > 0 else r_list_Z

            results["Y_future_true"].append(Y_future_true.tolist())
            results["Y_future_pred"].append(Yf.tolist() if Yf is not None else None)
            results["Y_concat_for_plot"].append(Y_concat.tolist() if Y_concat is not None else None)
            results["Z_future_true"].append(Z_future_true.tolist() if Z_future_true is not None else None)
            results["Z_future_pred"].append(Zf.tolist() if Zf is not None else None)
            results["Z_concat_for_plot"].append(Z_concat.tolist() if Z_concat is not None else None)
            results["X_future_pred"].append(Xf.tolist() if Xf is not None else None)
            results["pearson_per_channel"].append(r_list)
            results["pearson_per_channel_Z"].append(r_list_Z)

        flat_r = [
            v for r in results["pearson_per_channel"] if r
            for v in r if v is not None and not np.isnan(v)
        ]
        results["pearson_overall_mean"] = float(np.mean(flat_r)) if flat_r else np.nan
        flat_r_Z = [
            v for r in results["pearson_per_channel_Z"] if r
            for v in r if v is not None and not np.isnan(v)
        ]
        results["pearson_overall_mean_Z"] = float(np.mean(flat_r_Z)) if flat_r_Z else np.nan

        return results


class BaseWrapper:
    """Shared surface for PSID/DPAD/VARMA wrappers.

    Subclasses implement: ``train``, ``predict``, ``forecast``,
    ``load_from_file``, ``validate_forecast``. Training writes the model to
    disk; testing loads it and calls ``predict`` / ``validate_forecast``
    directly via ``BaseFramework._predict`` / ``_evaluate_forecast``. No
    separate predict-side validation method — ``predict`` is the one-step
    pass, ``validate_forecast`` is the multi-step evaluation.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger()
        self.idSys = None

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        raise NotImplementedError

    def predict(self, Y: TrialList, Z: Optional[TrialList] = None):
        raise NotImplementedError

    def forecast(self, m: int, Y_past: Array2D):
        raise NotImplementedError

    def load_from_file(self, model_path: str):
        raise NotImplementedError


class PSIDWrapper(BaseWrapper):
    @classmethod
    def from_idsys(cls, idSys: Any) -> "PSIDWrapper":
        """Minimal wrapper around a pre-loaded LSSM.

        Lets consumers that have only the raw idSys pickle (e.g. classification)
        use ``predict`` / ``forecast`` without rebuilding a training config.
        """
        inst = cls.__new__(cls)
        inst.config = None
        inst.logger = get_logger()
        inst.idSys = idSys
        return inst

    def load_from_file(self, model_path: str):
        import pickle

        self.logger.info(f"Loading PSID model from {model_path}")
        with open(model_path, "rb") as f:
            self.idSys = pickle.load(f)
        self.logger.info("PSID model loaded successfully")

        # scipy-1.16 DARE workaround (no-op on clean models).
        PSIDWrapper._refit_dare_if_needed(self.idSys)

        if (
            hasattr(self.idSys, "A")
            and self.idSys.A is not None
            and len(getattr(self.idSys, "A_powers_cache", []) or []) == 0
        ):
            PSIDWrapper._cache_A_powers(self.idSys)
            self.logger.info("Precomputed A matrix powers up to m=200 after loading model")
        return self.idSys

    def train(self, Y: TrialList, Z: Optional[TrialList] = None):
        """PSID identification + A-eigenvalue clip.

        Calls upstream ``PSID.PSID()`` then post-processes with
        ``_clip_A_eigenvalues`` (the only addition over the stock package).
        DARE is re-solved after the clip so steady-state Kalman quantities
        stay consistent with the modified A.
        """
        cfg = self.config.model
        nx: int = cfg.nx
        n1: int = cfg.n1
        i: int = cfg.i
        time_first: bool = getattr(cfg, "time_first", True)

        max_eigenvalue = getattr(cfg, "max_eigenvalue", 0.9999)
        max_eigenvalue = 1.0 if max_eigenvalue is None else float(max_eigenvalue)

        # Surface deprecated model flags (ignored — smoother paths removed).
        for deprecated in ("backward_kalman", "fb_smoother", "rescale_states"):
            if getattr(cfg, deprecated, None) is not None:
                self.logger.info(
                    f"config.model.{deprecated} set but ignored — "
                    "PSIDWrapper uses upstream PSID.PSID + A-clip only."
                )

        zscore_Z = getattr(cfg, "zscore_Z", True)
        remove_mean_Z = getattr(cfg, "remove_mean_Z", True)

        self.logger.info(
            f"Calling upstream PSID.PSID: nx={nx}, n1={n1}, i={i}, "
            f"time_first={time_first}, max_eigenvalue={max_eigenvalue}"
        )

        self.idSys, ws = PSID.PSID(
            Y, Z, nx=nx, n1=n1, i=i,
            zscore_Y=True, zscore_Z=zscore_Z,
            remove_mean_Y=True, remove_mean_Z=remove_mean_Z,
            time_first=time_first,
            fit_Cz_via_KF=True,
            return_WS=True,
        )
        # Stash subspace-identification singular spectra on the LSSM so they
        # survive pickling. ZHat_S = stage-1 (behavior-relevant), YHat_S =
        # stage-2 (residual Y dynamics). Used for (nx, n1) elbow selection.
        self.idSys.ZHat_S = ws.get("ZHat_S")
        self.idSys.YHat_S = ws.get("YHat_S")

        # Eigenvalue clip on A — the only modification over upstream PSID.
        if max_eigenvalue < 1.0 and getattr(self.idSys, "A", None) is not None:
            self.idSys.A = PSIDWrapper._clip_A_eigenvalues(
                np.asarray(self.idSys.A), max_eigenvalue
            )
            PSIDWrapper._refit_dare_if_needed(self.idSys, force=True)

        if getattr(self.idSys, "A", None) is not None:
            PSIDWrapper._cache_A_powers(self.idSys)
            self.logger.info("Precomputed A matrix powers up to m=200")
        return self.idSys

    @staticmethod
    def _cache_A_powers(idSys, max_m: int = 200) -> None:
        """Precompute A^t for t=1..max_m so forecast() can look them up O(1)."""
        A = np.asarray(idSys.A)
        cache = [A.copy()]
        for _ in range(2, max_m + 1):
            cache.append(cache[-1] @ A)
        idSys.A_powers_cache = cache
        idSys.max_m_precomputed = max_m

    def predict(self, Y: TrialList, Z: Optional[TrialList] = None):
        """Forward Kalman prediction via upstream PSID LSSM.predict.

        Returns ``(Zp, Yp, Xp)``. The ``Z`` argument is ignored — upstream
        signature is ``idSys.predict(Y)`` and takes no external input. Z is
        accepted only for BaseWrapper signature parity with VARMA.
        """
        return self.idSys.predict(Y)

    @staticmethod
    def _refit_dare_if_needed(idSys, force: bool = False):
        """Resolve DARE after A clip (force=True) or on scipy-1.16 NaN fallback.

        Two trigger conditions:
        1. ``force=True``: A was just modified (e.g. eigenvalue clip), so Pp, K,
           Kf, Kv, innovCov, A_KC must be re-derived.
        2. ``force=False`` AND ``Pp`` contains NaN: upstream PyPSID v1.2.6 calls
           ``linalg.solve_discrete_are(A.T, C.T, Q, R, s=S)`` which on scipy >=
           1.16 raises "Matrix e should be square" and leaves steady-state
           quantities as NaN. Re-solves passing ``e=None``.
        """
        if getattr(idSys, "state_dim", 0) == 0:
            return
        Pp = getattr(idSys, "Pp", None)
        needs_refit = force or (Pp is not None and np.any(np.isnan(Pp)))
        if not needs_refit:
            return
        A = np.asarray(idSys.A)
        C = np.asarray(idSys.C)
        Q = np.asarray(idSys.Q)
        R = np.asarray(idSys.R)
        S = np.asarray(idSys.S)
        # DARE only converges when A is stable.
        eigs = np.linalg.eigvals(A)
        if np.max(np.abs(eigs)) >= 1:
            return
        try:
            Pp_new = solve_discrete_are(a=A.T, b=C.T, q=Q, r=R, e=None, s=S)
        except Exception:
            return
        innovCov = C @ Pp_new @ C.T + R
        innovCov = (innovCov + innovCov.T) / 2
        innovCovInv = np.linalg.pinv(innovCov)
        K = (A @ Pp_new @ C.T + S) @ innovCovInv
        Kf = Pp_new @ C.T @ innovCovInv
        Kv = S @ innovCovInv
        A_KC = A - K @ C

        idSys.Pp = Pp_new
        idSys.K = K
        idSys.Kf = Kf
        idSys.Kv = Kv
        idSys.innovCov = innovCov
        idSys.A_KC = A_KC
        XCov = getattr(idSys, "XCov", None)
        if XCov is not None and not np.any(np.isnan(np.asarray(XCov))):
            idSys.P2 = np.asarray(XCov) - Pp_new

    @staticmethod
    def _clip_A_eigenvalues(A: np.ndarray, max_abs: float) -> np.ndarray:
        """Eigenvalue-clip A so spectral radius <= ``max_abs``.

        Acts as a no-op when ``max_abs`` is None or >= 1.
        """
        if max_abs is None or max_abs >= 1.0:
            return np.asarray(A, dtype=float)
        A = np.asarray(A, dtype=float)
        eigvals, eigvecs = np.linalg.eig(A)
        mags = np.abs(eigvals)
        scale = np.where(mags > max_abs, max_abs / np.maximum(mags, 1e-15), 1.0)
        A_new = eigvecs @ np.diag(eigvals * scale) @ np.linalg.inv(eigvecs)
        return np.real_if_close(A_new, tol=1e-6).real.astype(float)

    def forecast(self, m: int, Y_past: Array2D, Z_past: Optional[Array2D] = None):
        """m-step forecast via forward Kalman state seed + A-power propagation.

        Seeds the forecast from ``x̂_{T|T-1}`` of the upstream Kalman predictor
        on ``Y_past``, then iterates ``x_{t+1} = A x_t`` for t=1..m. Observations
        are computed as ``Yf = Xf @ C.T``, ``Zf = Xf @ Cz.T``, then un-z-scored
        by the PrepModels. Complexity: O(len(Y_past) + m).

        ``Z_past`` is accepted for BaseFramework signature parity but ignored —
        PSID predicts Z from Y alone via the learned (A, Cz) dynamics.
        """
        if self.idSys is None:
            raise ValueError(
                "Model not initialized. Call train() or load_from_file() first."
            )

        # Kalman predictor state estimate on past data.
        Zp_past, Yp_past, Xp_past = self.idSys.predict([Y_past])

        if Xp_past is None or len(Xp_past) == 0:
            raise ValueError("Could not extract state estimate from past data")
        Xp_past_array = np.asarray(Xp_past[0]) if isinstance(Xp_past, list) else Xp_past
        if Xp_past_array.shape[0] == 0:
            raise ValueError("State estimate is empty")
        x0 = Xp_past_array[-1, :]

        A = np.array(self.idSys.A)
        C = np.array(self.idSys.C)
        Cz = (
            np.array(self.idSys.Cz)
            if hasattr(self.idSys, "Cz") and self.idSys.Cz is not None
            else None
        )
        nx = A.shape[0]

        cache = getattr(self.idSys, "A_powers_cache", None)
        max_cached = getattr(self.idSys, "max_m_precomputed", 0)
        Xf = np.zeros((m, nx))
        if cache is not None and len(cache) > 0 and m <= max_cached:
            # A_powers_cache[t-1] = A^t
            for t in range(1, m + 1):
                Xf[t - 1, :] = cache[t - 1] @ x0
        else:
            x_cur = x0.copy()
            for t in range(m):
                x_cur = A @ x_cur
                Xf[t, :] = x_cur

        Yf = Xf @ C.T
        Zf = Xf @ Cz.T if Cz is not None and Cz.shape[0] > 0 else None

        # Un-z-score via PrepModels.
        if hasattr(self.idSys, "YPrepModel") and self.idSys.YPrepModel is not None:
            Yf = self.idSys.YPrepModel.apply_inverse(Yf)
        if (
            Zf is not None
            and hasattr(self.idSys, "ZPrepModel")
            and self.idSys.ZPrepModel is not None
        ):
            Zf = self.idSys.ZPrepModel.apply_inverse(Zf)

        return Zf, Yf, Xf


class PSIDFramework(BaseFramework):
    def _initialize_model(self):
        return PSIDWrapper(self.config)


# Module-level memoization for DPADWrapper.forecast — see the comment on
# that method. Keyed by id(self.idSys); cleared only when the worker process
# exits (no cross-process concerns since each classification runs in a fresh
# subprocess).
_DPADFWK_FORECAST_CACHE: dict[int, int] = {}


class DPADWrapper(BaseWrapper):
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

        self.logger.info(
            f"Training DPAD with nx={nx}, n1={n1}, method_code={method_code}, epochs={epochs}"
        )
        Y_dpad = [y.T for y in Y]
        Z_dpad = [z.T for z in Z] if Z is not None else None

        save_dir = Path(self.config.results.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.idSys = DPADModel(log_dir=self.config.results.log_dir)
        args = DPADModel.prepare_args(method_code)

        # NOTE: we intentionally do NOT forward config.model.fast as
        # skip_predictions. DPADModel.fit has a bug where skip_predictions=True
        # leaves `allX_steps` unbound when the Cz regression path still runs
        # (epochs>0), causing UnboundLocalError at DPADModel.py:1919. The
        # trainer-level `fast` flag still short-circuits post-training Pearson
        # r evaluation, which is the actual speedup we care about.
        self.idSys.fit(
            Y_dpad,
            Z=Z_dpad,
            nx=nx,
            n1=n1,
            epochs=epochs,
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
        """Run DPAD prediction per trial with block-aligned padding.

        Z is accepted for BaseWrapper parity but ignored (DPAD's predict uses
        Y only; Z is predicted, not consumed).
        """
        all_Zp, all_Yp, all_Xp = [], [], []

        self.idSys.set_steps_ahead([1])
        self.idSys.set_multi_step_with_data_gen(False)
        # predict() just reset steps_ahead / multi_step; invalidate the forecast
        # cache so the next forecast(m) call reconfigures the model.
        _DPADFWK_FORECAST_CACHE.pop(id(self.idSys), None)
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

    def forecast(
        self,
        m: int,
        Y_past: Array2D,
        Z_past: Optional[Array2D] = None,
    ) -> Tuple[Optional[Array2D], Optional[Array2D], Optional[Array2D]]:
        """``Z_past`` accepted for signature parity but ignored (DPAD consumes Y only)."""
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

        # Memoized setup — same pattern as validate_forecast above and as
        # utils/classification._ensure_dpad_forecast_setup. Each
        # set_steps_ahead call rebuilds m output heads on the TF model
        # (O(seconds) per call); we skip the rebuild when m hasn't changed.
        _key = id(self.idSys)
        if _DPADFWK_FORECAST_CACHE.get(_key) != m:
            self.idSys.set_steps_ahead(list(range(1, m + 1)))
            self.idSys.set_multi_step_with_data_gen(True, noise_samples=0)
            _DPADFWK_FORECAST_CACHE[_key] = m
        preds = self.idSys.predict(_pad_to_block(Y_past))

        Zf = _stack_last(preds[:m])
        Yf = _stack_last(preds[m : 2 * m])
        Xf = _stack_last(preds[2 * m : 3 * m])

        return Zf, Yf, Xf



class DPADFramework(BaseFramework):
    def _initialize_model(self):
        return DPADWrapper(self.config)


class VARMAOLSWrapper(BaseWrapper):
    """
    VARMA(p, q) estimation via strict OLS using a Long-VAR residual proxy.

    All channels (neural + behavioral) are modeled jointly in one multivariate
    system, capturing cross-channel dynamics bidirectionally.

    Algorithm:
        1. Fit a high-order VAR to approximate the process and recover residuals.
        2. Build a design matrix: [intercept, AR lags of data, MA lags of residuals].
        3. Solve via ``np.linalg.lstsq`` — unregularised OLS across all K outputs
           jointly.
        4. Forecast recursively: feed predictions back, future errors = 0.
    """

    def __init__(self, config: Config):
        super().__init__(config)

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

    @staticmethod
    def _apply_hamming_trial_edge_taper(data: np.ndarray, n_taper: int) -> np.ndarray:
        """Hamming-shaped edge taper on first/last ``n_taper`` samples of a trial.

        Reduces boundary power so concatenated trials have smooth transitions
        — necessary when the design matrix stitches trials head-to-tail for
        AR/MA lag regression.
        """
        if n_taper <= 0:
            return data
        T, K = data.shape
        if T <= 2 * n_taper:
            n_taper = max(0, (T - 1) // 2)
            if n_taper <= 0:
                return data
        out = np.array(data, copy=True, dtype=np.float64)
        h = np.hamming(2 * n_taper)
        ramp_up = (h[:n_taper] - h[0]) / (h[n_taper - 1] - h[0])
        ramp_down = (h[n_taper:] - h[-1]) / (h[n_taper] - h[-1])
        for ch in range(K):
            out[:n_taper, ch] *= ramp_up
            out[-n_taper:, ch] *= ramp_down
        return out

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
        self.logger.info("Training VARMA-OLS model (per-channel z-scoring)...")

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
            f"VARMA-OLS params: p={p}, q={q}, long_ar_lags={long_ar_lags}, "
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
            data_trial = self._apply_hamming_trial_edge_taper(data_trial, n_taper)
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
            raise ValueError("No trials were long enough for VARMA-OLS fitting.")

        # Single global fit: stack all trials; beta shape (1 + p*K + q*K, K)
        X_full = np.concatenate(all_X, axis=0)
        Y_full = np.concatenate(all_target, axis=0)

        self.logger.info(
            f"Design matrix shape: {X_full.shape}, Target shape: {Y_full.shape}"
        )

        # Ridge regression (Tikhonov). OLS on multivariate VARMA produced
        # unstable companion eigenvalues that triggered aggressive AR-root
        # stabilization (gamma << 1) and crippled all but the first channel.
        # Ridge keeps the fit stable; intercept (column 0 of X_full) is not
        # penalised. alpha defaults to 1.0; configurable via config.model.
        ridge_alpha = float(getattr(self.config.model, "ridge_alpha", 1.0))
        n_features = X_full.shape[1]
        penalty = ridge_alpha * np.eye(n_features)
        penalty[0, 0] = 0.0  # do not penalise intercept
        XtX = X_full.T @ X_full + penalty
        XtY = X_full.T @ Y_full
        self.beta = np.linalg.solve(XtX, XtY)
        self.logger.info(f"VARMA-Ridge fit: alpha={ridge_alpha}, beta shape={self.beta.shape}")

        self.logger.info(
            f"VARMA-OLS training complete. "
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
        One-step-ahead prediction for a single trial using the fitted VARMA-OLS model.

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

    def _recursive_forecast(
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
        forecasts_zscored = self._recursive_forecast(
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



class VARMAOLSFramework(BaseFramework):
    def _initialize_model(self):
        return VARMAOLSWrapper(self.config)
