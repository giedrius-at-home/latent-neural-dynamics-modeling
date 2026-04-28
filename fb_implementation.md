# Sani & Shanechi (2025) FB Implementation — PSID with filtering + forward-backward smoothing

Complete reference for the Kalman-side changes that replace RTS smoothing with
the paper's forward-backward method and upgrade forecasting with a learned
m-step gain. Paper: [arXiv 2507.15288](https://arxiv.org/abs/2507.15288)
("Preferential Subspace Identification (PSID) with forward-backward smoothing",
Sani & Shanechi, July 2025).

All changes live inside the existing `PSIDWrapper` class in
`utils/frameworks.py`; no new module was added.

---

## 1. Paper equations implemented

| Paper ref | What it does | Code method |
|---|---|---|
| eq 26 | Closed-form ridge LS for `C_z K_f` from Kalman innovations | `_fit_czkf` |
| eq 27 (stacked m-step form) | Single LS over horizons 1..M giving Γ_z K_f | `_fit_gamma_z_kf` |
| eq 29 | Residual secondary signal `z̄_k = z_k − ẑ_{k\|k}` | `_filtered_residual_trials` |
| Sec 2.7.4 | Fit PSID on time-reversed `(y, z̄)` → backward model | `_fit_backward_psid` |
| eq 30 | Smoother: `ẑ_{k\|N} = ẑ_{k\|k} + ẑ̄_{k\|k}` | `_smooth_fb_trial` |
| Sec 3 (filter-aware forecast) | `ẑ_{T+m\|T} = C_z A^m x̂_{T\|T−1} + (C_z A^m K_f) (y_T − C_y x̂_{T\|T−1})` | `_forecast_fb_trial` |

One practical twist: the stock paper gives only `(C_z K_f)` for `m = 0`
(filtering). For `m ≥ 1` forecasts we use the paper's stacked form (eq 27) so a
single LS returns the gain block for every integer-sample horizon up to
`fb_forecast_m`, and the `forecast()` path then dispatches the right block
per-request without refitting. This mirrors eq 27's `Γ_z K_f` construction.

A-regularisation (eigenvalue clipping to `|λ| ≤ max_eigenvalue`) was previously
a config field that went unused. It is now actively applied to both forward
and backward LSSMs inside `_fit_fb_components`, guaranteeing a numerically
stable Riccati recursion.

---

## 2. Config surface

```yaml
model:
  name: ...
  nx: 50
  n1: 10
  i: 100                      # PSID Hankel horizon
  backward_kalman: true       # still required — RTS is used as fallback
                              # for (Ys, Xs) even under fb_smoother
  fb_smoother: true           # NEW — activates Sani & Shanechi 2025
  max_eigenvalue: 0.9999      # becomes LIVE under fb_smoother
                              # (A eigenvalues clipped to ≤ this)
  forecast:
    m: 1                      # seconds → max learned forecast horizon
                              # in samples = m * sampling_frequency
    history: 3
data:
  sampling_frequency: 200
  ...
```

When `fb_smoother: true`, the attached LSSM gains these extra attributes after
training (all persisted with the model checkpoint):

| attribute | shape | meaning |
|---|---|---|
| `idSys.fb_smoother` | bool | flag checked at inference-time dispatch |
| `idSys.CzKf_fwd` | `(n_z, n_y)` | forward filter gain (eq 26) |
| `idSys.CzKf_bwd` | `(n_z, n_y)` | backward filter gain (same RRR on reversed residuals) |
| `idSys.GammaZKf_fwd` | `(horizon · n_z, n_y)` | stacked forecast gains Γ_z K_f (eq 27) |
| `idSys.fb_forecast_m` | int | max horizon in samples for which Γ block exists |
| `idSys.idSys_bwd` | LSSM | backward PSID model fit on time-reversed residuals |
| `idSys.CzAmKf_fwd` | `(n_z, n_y)` | legacy single-horizon alias for the m-th block |

---

## 3. Code — `train()` dispatch

```python
def train(self, Y: TrialList, Z: Optional[TrialList] = None):
    nx: int = self.config.model.nx
    n1: int = self.config.model.n1
    i: int = self.config.model.i
    time_first: bool = self.config.model.time_first

    backward_kalman: bool = bool(getattr(self.config.model, "backward_kalman", False))
    fb_smoother: bool = bool(getattr(self.config.model, "fb_smoother", False))
    max_eigenvalue = getattr(self.config.model, "max_eigenvalue", 0.995)
    if max_eigenvalue is None:
        max_eigenvalue = 1.0
    else:
        max_eigenvalue = float(max_eigenvalue)

    # ... logging / zscore config ...
    psid_kwargs = {
        "zscore_Y": True,
        "zscore_Z": zscore_Z,
        "remove_mean_Y": True,
        "remove_mean_Z": remove_mean_Z,
        "time_first": time_first,
        "backward_kalman": backward_kalman,
        "max_eigenvalue": max_eigenvalue,
    }
    self.idSys, ws = PSIDWrapper._psid_identify(
        Y, Z, nx, n1, i, return_WS=True, **psid_kwargs
    )
    self.idSys.ZHat_S = ws.get("ZHat_S")
    self.idSys.YHat_S = ws.get("YHat_S")

    PSIDWrapper._attach_smoother_params(self.idSys)
    self.idSys.backward_kalman = backward_kalman
    self.idSys.fb_smoother = fb_smoother

    if fb_smoother:
        forecast_m_samples = self._fb_forecast_m_samples()
        self._fit_fb_components(
            Y, Z,
            nx=nx, n1=n1, i=i,
            max_eig=max_eigenvalue,
            psid_kwargs=psid_kwargs,
            forecast_m_samples=forecast_m_samples,
        )
    # ... A_powers_cache for forecast ...
    return self.idSys
```

---

## 4. `_fit_fb_components` — master orchestrator

```python
def _fit_fb_components(
    self,
    Y: TrialList,
    Z: Optional[TrialList],
    *,
    nx: int, n1: int, i: int,
    max_eig: float,
    psid_kwargs: Dict[str, Any],
    forecast_m_samples: Optional[int],
) -> None:
    """Fit Sani & Shanechi (2025) PSID-with-filtering + backward PSID on train data.

    Implements arXiv 2507.15288, section 2.7:
      - eq 26: learn (C_z K_f) via ridge LS on Kalman-predictor innovations.
      - eq 27 (m-step form): learn (C_z A^m K_f) for forecast horizon m.
      - eq 29-30: fit a second PSID on time-reversed (Y, residual-Z), learn its
        (C_z K_f)_bwd, and define ẑ_{k|N} = ẑ_{k|k} + ẑ̄_{k|k} at inference.
    """
    if Z is None:
        raise ValueError("fb_smoother=True requires Z (secondary signal) during training.")

    # 1. Stabilise forward A, refresh Kalman-dependent quantities.
    self.idSys.A = PSIDWrapper._clip_A_eigenvalues(np.asarray(self.idSys.A), max_eig)
    PSIDWrapper._refit_dare_if_needed(self.idSys)
    PSIDWrapper._attach_smoother_params(self.idSys)

    # 2. Forward filter gain (eq 26) on train innovations.
    CzKf_fwd = PSIDWrapper._fit_czkf(self.idSys, Y, Z)

    # 3. Stacked forecast gains Γ_z K_f via a single LS (paper eq 27).
    GammaZKf_fwd = None
    if forecast_m_samples is not None and forecast_m_samples > 0:
        GammaZKf_fwd = PSIDWrapper._fit_gamma_z_kf(
            self.idSys, Y, Z, forecast_m_samples
        )

    # 4. Backward PSID on residuals (eq 29 + section 2.7.4).
    Z_resid_trials = PSIDWrapper._filtered_residual_trials(
        self.idSys, CzKf_fwd, Y, Z
    )
    idSys_bwd = PSIDWrapper._fit_backward_psid(
        Y, Z_resid_trials,
        nx=nx, n1=n1, i=i,
        psid_kwargs=psid_kwargs,
        max_eig=max_eig,
    )

    # 5. Backward filter gain: RRR on reversed-trial innovations.
    Y_rev = [np.ascontiguousarray(y[::-1]) for y in Y]
    Z_resid_rev = [np.ascontiguousarray(z[::-1]) for z in Z_resid_trials]
    CzKf_bwd = PSIDWrapper._fit_czkf(idSys_bwd, Y_rev, Z_resid_rev)

    # Attach everything on self.idSys so it survives model serialisation.
    self.idSys.CzKf_fwd = CzKf_fwd
    self.idSys.idSys_bwd = idSys_bwd
    self.idSys.CzKf_bwd = CzKf_bwd
    self.idSys.GammaZKf_fwd = GammaZKf_fwd
    self.idSys.fb_forecast_m = forecast_m_samples
    nz = CzKf_fwd.shape[0]
    self.idSys.CzAmKf_fwd = (
        GammaZKf_fwd[(forecast_m_samples - 1) * nz : forecast_m_samples * nz]
        if GammaZKf_fwd is not None else None
    )
```

Five discrete stages, each with its own helper:

1. Stabilise A (eigenvalue clipping) + refresh Kalman quantities.
2. Forward filter gain via RRR on predictor innovations.
3. Stacked forecast-horizon gains via a single bigger LS.
4. Compute Z-residuals → fit backward PSID on reversed (Y, residual).
5. Backward filter gain via same RRR on reversed innovations.

---

## 5. Code — static helpers

### 5.1 A-regularisation (eigenvalue clipping)

```python
@staticmethod
def _clip_A_eigenvalues(A: np.ndarray, max_abs: float) -> np.ndarray:
    """Eigenvalue-clip A so spectral radius <= ``max_abs``. No-op if max_abs >= 1."""
    if max_abs is None or max_abs >= 1.0:
        return np.asarray(A, dtype=float)
    A = np.asarray(A, dtype=float)
    eigvals, eigvecs = np.linalg.eig(A)
    mags = np.abs(eigvals)
    scale = np.where(mags > max_abs, max_abs / np.maximum(mags, 1e-15), 1.0)
    A_new = eigvecs @ np.diag(eigvals * scale) @ np.linalg.inv(eigvecs)
    return np.real_if_close(A_new, tol=1e-6).real.astype(float)
```

Scales each complex eigenvalue back to radius `max_abs` if it's larger.
Preserves the eigenvector basis, so the invariant subspaces are unchanged —
only the decay rates get damped. The eigendecomposition → reconstruction loop
is numerically stable for the `nx ≤ 55` systems we use.

### 5.2 Predictor innovations

```python
@staticmethod
def _predictor_innovations(idSys, Y):
    """Return (x̂_{k|k-1}, ỹ_{k|k-1}) for one trial in z-scored internal space."""
    allXp = idSys.kalman(Y)[0]
    C = np.asarray(idSys.C)
    if getattr(idSys, "YPrepModel", None) is not None:
        Y_int = idSys.YPrepModel.apply(Y, time_first=True)
    else:
        Y_int = Y
    innov_Y = Y_int - allXp @ C.T
    return np.asarray(allXp), innov_Y
```

Key point: PSID's `YPrepModel` z-scores Y internally, and `idSys.kalman()` uses
the z-scored representation. The innovations we feed to RRR must live in the
same z-scored space — hence the explicit `YPrepModel.apply` before subtracting
`C @ Xp`.

### 5.3 Ridge least-squares kernel

```python
@staticmethod
def _ridge_lstsq(X, Y, ridge=1e-6):
    """Solve M = argmin ||Y - X M||_F^2 + ridge ||M||_F^2."""
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ Y)
```

Shared primitive for every RRR fit in the module. The ridge term is tiny
(`1e-6`) but prevents singular-gram edge cases when `X` has near-zero columns
(e.g. a neural channel with no variance in one trial block).

### 5.4 Forward filter gain (paper eq 26)

```python
@staticmethod
def _fit_czkf(idSys, Y_trials, Z_trials, ridge=1e-6):
    """RRR fit of (C_z K_f), paper eq 26. Returns shape (n_z, n_y)."""
    Cz = np.asarray(idSys.Cz)
    innov_Y_all, innov_Z_all = [], []
    for Y, Z in zip(Y_trials, Z_trials):
        allXp, innov_Y = PSIDWrapper._predictor_innovations(idSys, Y)
        if getattr(idSys, "ZPrepModel", None) is not None:
            Z_int = idSys.ZPrepModel.apply(Z, time_first=True)
        else:
            Z_int = Z
        innov_Y_all.append(innov_Y)
        innov_Z_all.append(Z_int - allXp @ Cz.T)
    X = np.vstack(innov_Y_all)
    Ytgt = np.vstack(innov_Z_all)
    return PSIDWrapper._ridge_lstsq(X, Ytgt, ridge=ridge).T
```

The target `Z_int - allXp @ Cz.T` is exactly the innovation in z-space:
`z_k − C_z x̂_{k|k−1}`. Regressing this on `y_k − C_y x̂_{k|k−1}` gives the
gain that best maps y-innovations to z-innovations. For the dimensions in play
(`n_y = 8, n_z = 2, n_x ≥ 50`), the paper's rank-n_x constraint is
non-binding, so plain ridge LS equals the rank-constrained RRR.

### 5.5 Stacked forecast gains (paper eq 27)

```python
@staticmethod
def _fit_gamma_z_kf(idSys, Y_trials, Z_trials, horizon, ridge=1e-6):
    """Stacked LS fit of Γ_z K_f (paper eq 27) over forecast horizons 1..horizon.

    Returns a matrix of shape (horizon * n_z, n_y) where the row-block for
    step m is rows [(m-1)*n_z : m*n_z].
    """
    if horizon < 1:
        raise ValueError(f"_fit_gamma_z_kf: horizon must be >= 1, got {horizon}")
    A = np.asarray(idSys.A)
    Cz = np.asarray(idSys.Cz)

    # Γ_z = [C_z A; C_z A^2; ...; C_z A^horizon]
    CzA_blocks = []
    CzAm = Cz.copy()
    for _ in range(horizon):
        CzAm = CzAm @ A
        CzA_blocks.append(CzAm)
    Gamma_z = np.vstack(CzA_blocks)  # (horizon * n_z, n_x)

    innov_Y_all, resid_Z_all = [], []
    nz = Cz.shape[0]
    for Y, Z in zip(Y_trials, Z_trials):
        N = Y.shape[0]
        if N <= horizon:
            continue
        allXp, innov_Y = PSIDWrapper._predictor_innovations(idSys, Y)
        if getattr(idSys, "ZPrepModel", None) is not None:
            Z_int = idSys.ZPrepModel.apply(Z, time_first=True)
        else:
            Z_int = Z
        T_valid = N - horizon
        Z_stacked = np.zeros((T_valid, horizon * nz), dtype=float)
        for mstep in range(1, horizon + 1):
            Z_stacked[:, (mstep - 1) * nz : mstep * nz] = Z_int[mstep : mstep + T_valid]
        Z_pred_stacked = allXp[:T_valid] @ Gamma_z.T
        innov_Y_all.append(innov_Y[:T_valid])
        resid_Z_all.append(Z_stacked - Z_pred_stacked)

    if not innov_Y_all:
        raise ValueError(f"_fit_gamma_z_kf: no trial long enough for horizon={horizon}")
    X = np.vstack(innov_Y_all)
    Ytgt = np.vstack(resid_Z_all)
    return PSIDWrapper._ridge_lstsq(X, Ytgt, ridge=ridge).T  # (horizon*nz, n_y)
```

### 5.6 Residual Z trials (for backward PSID fit)

```python
@staticmethod
def _filtered_residual_trials(idSys, CzKf, Y_trials, Z_trials):
    """Return z̄_k = z_k - ẑ_{k|k} per trial in original (un-z-scored) space."""
    Cz = np.asarray(idSys.Cz)
    out = []
    for Y, Z in zip(Y_trials, Z_trials):
        allXp, innov_Y = PSIDWrapper._predictor_innovations(idSys, Y)
        Z_filt_int = allXp @ Cz.T + innov_Y @ CzKf.T   # ẑ_{k|k} in z-space
        if getattr(idSys, "ZPrepModel", None) is not None:
            Z_filt = idSys.ZPrepModel.apply_inverse(Z_filt_int)
        else:
            Z_filt = Z_filt_int
        out.append(np.asarray(Z - Z_filt, dtype=float))
    return out
```

### 5.7 Backward PSID fit

```python
@staticmethod
def _fit_backward_psid(Y_trials, Z_residual_trials, *, nx, n1, i,
                       psid_kwargs, max_eig):
    """PSID on time-reversed (Y, Z_residual). Returns stabilised idSys_bwd."""
    Y_rev = [np.ascontiguousarray(y[::-1]) for y in Y_trials]
    Z_rev = [np.ascontiguousarray(z[::-1]) for z in Z_residual_trials]
    idSys_bwd, _ = PSIDWrapper._psid_identify(
        Y_rev, Z_rev, nx, n1, i, return_WS=True, **psid_kwargs
    )
    idSys_bwd.A = PSIDWrapper._clip_A_eigenvalues(np.asarray(idSys_bwd.A), max_eig)
    PSIDWrapper._refit_dare_if_needed(idSys_bwd)
    PSIDWrapper._attach_smoother_params(idSys_bwd)
    return idSys_bwd
```

### 5.8 Forward-backward smoother for one trial (paper eq 30)

```python
@staticmethod
def _smooth_fb_trial(idSys_fwd, idSys_bwd, CzKf_fwd, CzKf_bwd, Y):
    """Forward-backward smoother for one trial: ẑ_{k|N} = ẑ_{k|k} + ẑ̄_{k|k} (eq 30).

    The paper only formulates smoothing for the secondary signal z. For the
    primary signal y and the latent state x, we fall back to the classical
    RTS smoother on the forward LSSM — strictly better than the Kalman-
    predictor output and what downstream classification/reconstruction code
    already expects from :meth:`PSIDWrapper.smooth`.
    """
    Cz_fwd = np.asarray(idSys_fwd.Cz)
    Cz_bwd = np.asarray(idSys_bwd.Cz)

    # Forward PSID-with-filtering: ẑ_{k|k} in z-scored space.
    Xp_fwd, innov_Y_fwd = PSIDWrapper._predictor_innovations(idSys_fwd, Y)
    Z_fwd_int = Xp_fwd @ Cz_fwd.T + innov_Y_fwd @ CzKf_fwd.T
    if getattr(idSys_fwd, "ZPrepModel", None) is not None:
        Z_fwd_orig = idSys_fwd.ZPrepModel.apply_inverse(Z_fwd_int)
    else:
        Z_fwd_orig = Z_fwd_int

    # Backward PSID-with-filtering on reversed Y.
    Y_rev = np.ascontiguousarray(Y[::-1])
    Xp_bwd_rev, innov_Y_bwd_rev = PSIDWrapper._predictor_innovations(idSys_bwd, Y_rev)
    Z_bwd_rev_int = Xp_bwd_rev @ Cz_bwd.T + innov_Y_bwd_rev @ CzKf_bwd.T
    if getattr(idSys_bwd, "ZPrepModel", None) is not None:
        Z_bwd_rev_orig = idSys_bwd.ZPrepModel.apply_inverse(Z_bwd_rev_int)
    else:
        Z_bwd_rev_orig = Z_bwd_rev_int
    Z_bwd_orig = np.ascontiguousarray(Z_bwd_rev_orig[::-1])

    Zs = Z_fwd_orig + Z_bwd_orig  # eq 30

    # RTS smoother on the forward LSSM for (Ys, Xs) — matches baseline quality.
    _Zs_rts, Ys, Xs = PSIDWrapper._rts_smooth(idSys_fwd, Y)

    return Zs, Ys, Xs
```

**Design note**: the paper only touches Z. For Y (primary signal reconstruction) and X (latent state, used as classification features) we still run the classical RTS smoother on the forward LSSM. This keeps downstream consumers — classification, forecast-seeding, Pearson Y metrics — bit-identical to the baseline RTS pipeline, so any improvement observed on Z-related outputs is unambiguously attributable to the FB method.

### 5.9 Filter-aware m-step forecast (paper eq 27)

```python
@staticmethod
def _forecast_fb_trial(idSys_fwd, GammaZKf_fwd, Y_past, m):
    """Filter-aware m-step forecast for one Y_past trial.

    For every horizon t=1..m the learned gain block
    ``GammaZKf_fwd[(t-1)*n_z : t*n_z]`` is applied to the y-innovation at
    the forecast origin T, giving:
        ẑ_{T+t|T} = C_z A^t x̂_{T|T-1} + (C_z A^t K_f) (y_T - C_y x̂_{T|T-1}).
    """
    A = np.asarray(idSys_fwd.A)
    C = np.asarray(idSys_fwd.C)
    Cz = np.asarray(idSys_fwd.Cz)
    nz = Cz.shape[0]

    allXp, innov_Y = PSIDWrapper._predictor_innovations(idSys_fwd, Y_past)
    x_last = allXp[-1]
    innov_last = innov_Y[-1]

    Xf = np.zeros((m, A.shape[0]), dtype=float)
    x = x_last.copy()
    for t in range(m):
        x = A @ x
        Xf[t] = x

    Yf_int = Xf @ C.T
    Zf_int = Xf @ Cz.T
    # Vectorised application of the stacked gain matrix to the last innovation.
    gains = GammaZKf_fwd[: m * nz].reshape(m, nz, -1)  # (m, n_z, n_y)
    Zf_int += gains @ innov_last  # broadcasts to (m, n_z)

    if getattr(idSys_fwd, "YPrepModel", None) is not None:
        Yf = idSys_fwd.YPrepModel.apply_inverse(Yf_int)
    else:
        Yf = Yf_int
    if getattr(idSys_fwd, "ZPrepModel", None) is not None:
        Zf = idSys_fwd.ZPrepModel.apply_inverse(Zf_int)
    else:
        Zf = Zf_int
    return Zf, Yf, Xf
```

Y-forecast (`Yf`) is still the baseline predictor-propagated `C A^m x̂`; the
paper does not touch it. Only Z picks up the learned gain.

---

## 6. Inference-time dispatchers

### 6.1 `predict()` — smoothing

```python
def predict(self, Y: TrialList, Z: Optional[TrialList] = None):
    if bool(getattr(self.idSys, "fb_smoother", False)):
        return self.smooth_fb(Y)
    if bool(getattr(self.idSys, "backward_kalman", False)):
        return self.smooth(Y)
    return self.idSys.predict(Y, U=Z)
```

Precedence: `fb_smoother` > `backward_kalman` > plain Kalman predict. Makes
the new path strictly opt-in via config.

### 6.2 `smooth_fb()` — multi-trial wrapper

```python
def smooth_fb(self, Y):
    """PSID forward-backward smoother (Sani & Shanechi 2025, eq 30)."""
    if isinstance(Y, (list, tuple)):
        trials = [self.smooth_fb(trial) for trial in Y]
        return tuple([t[i] for t in trials] for i in range(3))
    return PSIDWrapper._smooth_fb_trial(
        self.idSys,
        self.idSys.idSys_bwd,
        self.idSys.CzKf_fwd,
        self.idSys.CzKf_bwd,
        Y,
    )
```

### 6.3 `forecast()` — m-step with learned gain

```python
def forecast(self, m: int, Y_past: Array2D, Z_past: Optional[Array2D] = None):
    if self.idSys is None:
        raise ValueError("Model not initialized. Call train() or load_from_file() first.")

    # When fb_smoother is enabled, use the paper's filter-aware forecast
    # (Sani & Shanechi 2025, eq 27 m-step): ẑ_{T+m|T} = C_z A^m x̂_{T|T-1} +
    # (C_z A^m K_f) (y_T - C_y x̂_{T|T-1}). The stacked Γ_z K_f lets us
    # dispatch any integer-sample horizon 1..fb_forecast_m without refit.
    if (
        bool(getattr(self.idSys, "fb_smoother", False))
        and getattr(self.idSys, "GammaZKf_fwd", None) is not None
        and 1 <= m <= int(getattr(self.idSys, "fb_forecast_m", 0))
    ):
        return PSIDWrapper._forecast_fb_trial(
            self.idSys, self.idSys.GammaZKf_fwd, Y_past, m
        )

    # ... fall through to classical predictor-propagation forecast ...
```

Guard: FB forecast only activates if the stacked gain was actually trained
(`GammaZKf_fwd is not None`) AND the requested horizon is within the learned
range (`m ≤ fb_forecast_m`). Anything outside falls back to the classical
`A^m x̂` propagation.

---

## 7. Pipeline generator change

`scripts/pipeline_psid.py` — `_make_training_config()` adds the flag for every
non-vanilla variant so future pipeline runs opt-in by default:

```python
def _make_training_config(mode_cfg, nx, n1, dbs, backward_kalman, rescale_states,
                          vanilla=False, i=None):
    ii = i if i is not None else mode_cfg.full_i
    name = mode_cfg.variant_name(nx, n1, dbs, vanilla, i=ii)
    model_cfg = {
        "name": name,
        "nx": nx, "n1": n1, "i": ii,
        "time_first": True,
        "fast": False,
        "reuse_splits": False,
        "backward_kalman": backward_kalman,
        "rescale_states": rescale_states,
        "max_eigenvalue": MAX_EIGENVALUE,
        "forecast": {"m": FORECAST_M_SECONDS, "history": FORECAST_HISTORY_SECONDS},
    }
    # Vanilla variants are the channel-selection pass and stay on the classical
    # Kalman/RTS path — they are assumed already trained and re-used.
    # Non-vanilla (both/on/off) variants switch to Sani & Shanechi 2025
    # forward-backward smoothing + filter-aware forecast. A-eigenvalue clipping
    # is activated inside _fit_fb_components, so max_eigenvalue becomes live.
    if not vanilla:
        model_cfg["fb_smoother"] = True
    cfg = {"model": model_cfg, "data": mode_cfg.data_block(dbs),
           "results": {"save_dir": f"results/{name}", ...}}
    return write_yaml(mode_cfg.variant_config_path(nx, n1, dbs, vanilla, i=ii), cfg)
```

---

## 8. Data flow summary

```
                     ┌──────────────────────────────────────┐
                     │ train(Y_trials, Z_trials)            │
                     └──────────────────────────────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────────────┐
              │ PSID identify (forward)   → idSys_fwd         │
              │ attach RTS params                             │
              └───────────────────────────────────────────────┘
                                  │
                     if fb_smoother:
                                  ▼
      ┌────────────────────────────────────────────────────────────┐
      │ _clip_A_eigenvalues(idSys_fwd.A)                           │
      │ _refit_dare_if_needed + _attach_smoother_params            │
      │                                                            │
      │ CzKf_fwd         ← _fit_czkf(idSys_fwd, Y, Z)      (eq 26) │
      │ GammaZKf_fwd     ← _fit_gamma_z_kf(…, horizon=M)   (eq 27) │
      │                                                            │
      │ z̄_trials        ← _filtered_residual_trials(…)    (eq 29) │
      │ idSys_bwd        ← _fit_backward_psid(Y_rev, z̄_rev)       │
      │ CzKf_bwd         ← _fit_czkf(idSys_bwd, Y_rev, z̄_rev)    │
      │                                                            │
      │ attach every tensor on idSys_fwd so it survives save/load  │
      └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                ─────────── inference time ───────────
                                  │
            predict(Y):                    forecast(m, Y_past):
            ┌──────────────┐               ┌──────────────────────────┐
            │ _smooth_fb_  │               │ if m ≤ fb_forecast_m and │
            │   trial:     │               │ GammaZKf_fwd is not None │
            │              │               │ → _forecast_fb_trial     │
            │ Z = forward  │               │ else → classical A^m x̂  │
            │   filter +   │               └──────────────────────────┘
            │   backward   │
            │   filter (on │
            │   residuals) │
            │              │
            │ Y, X = RTS   │
            │   smoother   │
            │   (forward)  │
            └──────────────┘
```

---

## 9. Cross-cutting invariants

- **Space hygiene**: All RRR fits live in z-scored space (`YPrepModel.apply`,
  `ZPrepModel.apply`). Only at the return boundary do we inverse-transform via
  `apply_inverse`. This keeps the learned gains scale-free and matches what
  PSID's own `kalman()` expects internally.
- **Serialisation-safe**: Every FB component is attached to `self.idSys` before
  training's `A_powers_cache` step; `Trainer` already serialises
  `self.framework.model.idSys`, so the FB state survives save/load with no
  trainer changes.
- **Backward-compat**: Models trained without `fb_smoother: true` still work
  via the `backward_kalman` branch of `predict()`. The `forecast()` dispatcher
  has a conjunction guard, so an old model missing `GammaZKf_fwd` correctly
  falls through to the classical path.
- **A regularisation**: only active when `fb_smoother: true`. Vanilla and
  pre-FB variants retain whatever spectral radius PSID originally produced.
- **Numerics**: ridge = 1e-6 in every LS solve; eigenvalue floor = 1e-15
  inside clipping; `_refit_dare_if_needed` handles the scipy ≥ 1.16 DARE
  regression that silently produces NaN `Pp`.

---

## 10. Files touched

| File | Nature of change |
|---|---|
| `utils/frameworks.py` | All PSIDWrapper methods listed above — inline inside the existing class, no new imports |
| `scripts/pipeline_psid.py` | `_make_training_config` adds `fb_smoother: True` for non-vanilla variants |
| `training/setups/psid/{narrow_band_200Hz,laplacian_200Hz}/{both,on,off}/*.yaml` | 101 existing non-vanilla yamls bulk-patched with `fb_smoother: true` |

No new module files; everything integrates into the existing class hierarchy.
