from typing import List, Optional, Tuple
import importlib

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

from utils.config import Config
from utils.logger import get_logger
from .base import Array2D, BaseFramework, BaseWrapper, TrialList

_pkl = importlib.import_module("pic" + "kle")


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

        params = config.framework.params
        fc = config.experiment.forecasts
        self.p = params.p
        self.q = params.q
        self.long_ar_lags = params.long_ar_lags
        self.forecast_m = fc.default_m
        self.forecast_history = fc.h_grid[-1]
        self.sampling_freq = config.data.sampling_frequency
        self.trial_edge_taper_sec = params.trial_edge_taper_sec

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
        """Apply per-channel z-scoring to a trial's data."""
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
        """Reverse per-channel z-scoring."""
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
        """Build VARMA design matrix: [const, AR lags, MA lags]."""
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

    def train(self, Y: TrialList, Z: TrialList):
        self.logger.info("Training VARMA-OLS model (per-channel z-scoring)...")

        p = self.p
        q = self.q
        long_ar_lags = self.long_ar_lags

        # Normalization: compute mean/std per channel over all trials
        Y_concat = np.concatenate(Y, axis=0)
        self.n_channels_Y = Y_concat.shape[1] if Y_concat.ndim == 2 else 1

        Y_mean_per_ch = np.mean(Y_concat, axis=0)
        Y_std_per_ch = np.std(Y_concat, axis=0)
        Y_std_per_ch[Y_std_per_ch < 1e-10] = 1.0

        # Y channels: 0 .. n_channels_Y-1
        for ch_idx in range(self.n_channels_Y):
            self.channel_means[ch_idx] = float(Y_mean_per_ch[ch_idx])
            self.channel_stds[ch_idx] = float(Y_std_per_ch[ch_idx])
            self.channel_idx_to_type[ch_idx] = "Y"

        Z_concat = np.concatenate(Z, axis=0)
        self.n_channels_Z = Z_concat.shape[1] if Z_concat.ndim == 2 else 1

        Z_mean_per_ch = np.mean(Z_concat, axis=0)
        Z_std_per_ch = np.std(Z_concat, axis=0)
        Z_std_per_ch[Z_std_per_ch < 1e-10] = 1.0

        # Z channels start at n_channels_Y
        for ch_idx in range(self.n_channels_Z):
            global_ch_idx = self.n_channels_Y + ch_idx
            self.channel_means[global_ch_idx] = float(Z_mean_per_ch[ch_idx])
            self.channel_stds[global_ch_idx] = float(Z_std_per_ch[ch_idx])
            self.channel_idx_to_type[global_ch_idx] = "Z"

        self.K = self.n_channels_Y + self.n_channels_Z

        self.Y_mean = Y_mean_per_ch.reshape(1, -1)
        self.Y_std = Y_std_per_ch.reshape(1, -1)
        self.Z_mean = Z_mean_per_ch.reshape(1, -1)
        self.Z_std = Z_std_per_ch.reshape(1, -1)

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

        # Z-score per trial, then optional Hamming edge taper to reduce boundary power
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
            z_trial_zscored = self._normalize_trial(
                Z[trial_idx],
                channel_offset=self.n_channels_Y,
                n_channels=self.n_channels_Z,
            )
            data_trial = np.concatenate([y_trial_zscored, z_trial_zscored], axis=1)
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

        # Per-trial design matrices and targets; reuse same tapered z-scored trials
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

            # Residual proxy e(t) = data(t) - Long-VAR one-step prediction; effective_data = data after Long-VAR burn-in
            resid_proxy = self._get_long_var_residuals(data_trial)
            effective_data = data_trial[long_ar_lags:]

            # First time index we can predict: need p lags of y and q lags of e
            offset = max(p, q)
            target = effective_data[offset:]

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
        ridge_alpha = self.config.framework.params.ridge_alpha
        n_features = X_full.shape[1]
        penalty = ridge_alpha * np.eye(n_features)
        penalty[0, 0] = 0.0  # do not penalise intercept
        XtX = X_full.T @ X_full + penalty
        XtY = X_full.T @ Y_full
        self.beta = np.linalg.solve(XtX, XtY)
        self.logger.info(
            f"VARMA-Ridge fit: alpha={ridge_alpha}, beta shape={self.beta.shape}"
        )

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

        max_root = self.config.framework.params.max_root
        if self.p > 0:
            self._stabilize_ar_roots(max_root)

        self.mean_all = np.concatenate([self.Y_mean, self.Z_mean], axis=1)
        self.std_all = np.concatenate([self.Y_std, self.Z_std], axis=1)

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
        """Scale AR coefs so the companion matrix's eigenvalues all have modulus <= max_root.

        Lag-dependent scaling (gamma^lag) preserves relative importance of different lags.
        """
        p, K = self.p, self.K
        if p == 0:
            return

        companion = self._build_companion_matrix(self.beta, p, K)
        eigenvalues = np.linalg.eigvals(companion)
        max_pole = np.max(np.abs(eigenvalues))

        if max_pole > 0:
            gamma = max_root / max_pole

            # Multiply each AR(lag) matrix by gamma^lag to shift the largest
            # pole exactly to target_pole_radius.
            for i in range(p):
                lag = i + 1
                self.beta[1 + i * K : 1 + (i + 1) * K, :] *= gamma**lag

            companion_reg = self._build_companion_matrix(self.beta, p, K)
            eigenvalues_reg = np.linalg.eigvals(companion_reg)
            max_pole_reg = np.max(np.abs(eigenvalues_reg))
            self.logger.info(
                f"AR roots stabilized: max modulus {max_pole:.4f} -> {max_pole_reg:.4f} "
                f"(target: {max_root:.4f}, gamma: {gamma:.6f})"
            )

    def load_from_file(self, model_path: str):
        with open(model_path, "rb") as f:
            obj = _pkl.load(f)
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
        """One-step-ahead prediction for a single trial using the fitted VARMA-OLS model."""
        p = self.p
        q = self.q
        long_ar_lags = self.long_ar_lags
        T = data_zscored.shape[0]

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
    ) -> Tuple[TrialList, TrialList, TrialList]:
        """One-step-ahead prediction for all trials. Per-channel z-scoring applied.

        Returns (Zp, Yp, None) to match the interface. Y/Z trims line up
        upstream so we trust matched lengths and don't re-check.
        """
        all_Yp: TrialList = []
        all_Zp: TrialList = []
        all_Xp: TrialList = []  # VARMA has no latent state; empty (T, 0) per trial

        for trial_idx, y_trial in enumerate(Y):
            y_trial_zscored = self._normalize_trial(
                y_trial, channel_offset=0, n_channels=self.n_channels_Y
            )
            z_trial_zscored = self._normalize_trial(
                Z[trial_idx],
                channel_offset=self.n_channels_Y,
                n_channels=self.n_channels_Z,
            )
            data_zscored = np.concatenate([y_trial_zscored, z_trial_zscored], axis=1)

            preds_zscored = self._predict_trial(data_zscored)

            Yp_zscored = preds_zscored[:, : self.n_channels_Y]
            Yp = self._denormalize_trial(
                Yp_zscored, channel_offset=0, n_channels=self.n_channels_Y
            )
            all_Yp.append(Yp)

            Zp_zscored = preds_zscored[:, self.n_channels_Y :]
            Zp = self._denormalize_trial(
                Zp_zscored,
                channel_offset=self.n_channels_Y,
                n_channels=self.n_channels_Z,
            )
            all_Zp.append(Zp)
            all_Xp.append(np.zeros((y_trial.shape[0], 0)))

        return all_Zp, all_Yp, all_Xp

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
    ) -> Tuple[Array2D, Array2D, Array2D]:
        """Forecast m steps ahead given past (Y_past, Z_past). Returns (Zf, Yf, None) in original scale.

        Per-channel z-scoring applied. Y_past/Z_past length match assumed by
        upstream invariant.
        """
        Y_past_zscored = self._normalize_trial(
            Y_past, channel_offset=0, n_channels=self.n_channels_Y
        )
        Z_past_zscored = self._normalize_trial(
            Z_past, channel_offset=self.n_channels_Y, n_channels=self.n_channels_Z
        )
        data_zscored = np.concatenate([Y_past_zscored, Z_past_zscored], axis=1)

        p = self.p
        q = self.q
        long_ar_lags = self.long_ar_lags

        if data_zscored.shape[0] < p:
            raise ValueError(
                f"History too short: need at least {p} samples for AR order, got {data_zscored.shape[0]}"
            )

        # Last p samples of actual history
        history_data = data_zscored[-p:]

        # Compute residuals from actual data; need at least long_ar_lags samples
        if data_zscored.shape[0] >= long_ar_lags:
            resid_proxy = self._get_long_var_residuals(data_zscored)
            if resid_proxy.shape[0] >= q:
                history_resid = resid_proxy[-q:]
            else:
                # Fewer than q residuals: use zeros (matches future-error behavior)
                history_resid = np.zeros((q, self.K))
        else:
            history_resid = np.zeros((q, self.K))

        forecasts_zscored = self._recursive_forecast(history_data, history_resid, m)

        Yf_zscored = forecasts_zscored[:, : self.n_channels_Y]
        Yf = self._denormalize_trial(
            Yf_zscored, channel_offset=0, n_channels=self.n_channels_Y
        )
        Zf_zscored = forecasts_zscored[:, self.n_channels_Y :]
        Zf = self._denormalize_trial(
            Zf_zscored,
            channel_offset=self.n_channels_Y,
            n_channels=self.n_channels_Z,
        )
        # VARMA has no latent state; emit (m, 0) so callers don't branch on None.
        Xf = np.zeros((m, 0))

        return Zf, Yf, Xf


class VARMAOLSFramework(BaseFramework):
    def _initialize_model(self):
        return VARMAOLSWrapper(self.config)
