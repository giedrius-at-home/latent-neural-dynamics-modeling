from typing import Any, Optional
import importlib
import numpy as np
import PSID
from scipy.linalg import solve_discrete_are

from utils.logger import get_logger
from .base import Array2D, BaseFramework, BaseWrapper, TrialList

_pkl = importlib.import_module("pic" + "kle")


class PSIDWrapper(BaseWrapper):
    @classmethod
    def from_idsys(cls, idSys: Any) -> "PSIDWrapper":
        """Minimal wrapper around a pre-loaded LSSM.

        Lets consumers that have only the raw idSys ckpt (e.g. classification)
        use ``predict`` / ``forecast`` without rebuilding a training config.
        """
        inst = cls.__new__(cls)
        inst.config = None
        inst.logger = get_logger()
        inst.idSys = idSys
        return inst

    def load_from_file(self, model_path: str):
        self.logger.info(f"Loading PSID model from {model_path}")
        with open(model_path, "rb") as f:
            self.idSys = _pkl.load(f)
        self.logger.info("PSID model loaded successfully")

        # scipy-1.16 DARE workaround (no-op on clean models).
        PSIDWrapper._refit_dare_if_needed(self.idSys)

        if (
            hasattr(self.idSys, "A")
            and self.idSys.A is not None
            and len(getattr(self.idSys, "A_powers_cache", []) or []) == 0
        ):
            PSIDWrapper._cache_A_powers(self.idSys)
            self.logger.info(
                "Precomputed A matrix powers up to m=200 after loading model"
            )
        return self.idSys

    def train(self, Y: TrialList, Z: TrialList):
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

        zscore_Z = getattr(cfg, "zscore_Z", True)
        remove_mean_Z = getattr(cfg, "remove_mean_Z", True)

        self.logger.info(
            f"Calling upstream PSID.PSID: nx={nx}, n1={n1}, i={i}, "
            f"time_first={time_first}, max_eigenvalue={max_eigenvalue}"
        )

        self.idSys, ws = PSID.PSID(
            Y,
            Z,
            nx=nx,
            n1=n1,
            i=i,
            zscore_Y=True,
            zscore_Z=zscore_Z,
            remove_mean_Y=True,
            remove_mean_Z=remove_mean_Z,
            time_first=time_first,
            fit_Cz_via_KF=True,
            return_WS=True,
        )
        # Stash subspace-identification singular spectra on the LSSM so they
        # survive serialization. ZHat_S = stage-1 (behavior-relevant), YHat_S =
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
        accepted as optional only for BaseWrapper signature parity with VARMA.
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
        Cz = np.array(self.idSys.Cz)
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
        Zf = Xf @ Cz.T

        # Un-z-score via PrepModels.
        Yf = self.idSys.YPrepModel.apply_inverse(Yf)
        Zf = self.idSys.ZPrepModel.apply_inverse(Zf)

        return Zf, Yf, Xf


class PSIDFramework(BaseFramework):
    def _initialize_model(self):
        return PSIDWrapper(self.config)
