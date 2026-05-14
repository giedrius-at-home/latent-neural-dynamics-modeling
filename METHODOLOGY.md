# Methodology: Latent Neural Dynamics Modeling Pipeline

## Overview

Three latent-state models — PSID, VARMA, and DPAD — are trained on intracranial ECoG recordings from participants undergoing deep brain stimulation (DBS). The central question: does the latent state inferred from neural signals carry information about DBS condition (on/off), and does this information emerge from behavior-relevant or behavior-irrelevant dynamics?

Each framework fits a separate model per DBS condition (DBS-on only, DBS-off only, both conditions pooled), yielding three variants per session. The pipeline runs: preprocessing -> splits -> PSID diagnostic (feature + hyperparameter selection) -> training (PSID/VARMA/DPAD) -> predictions -> forecasts -> classification.

---

## 1. Participants and Sessions

| Participant | Sessions used |
|-------------|---------------|
| PDI1        | S2, S4        |
| PDI4        | S2, S3        |

Each session contains alternating DBS-on and DBS-off blocks. Within each block, participants perform a continuous motor tracing task. Trials are the individual task epochs within a block.

**Experiment types:** Two variants are run per session:
- `z-as-behavior`: Z (the prioritized output) = behavioral signals (hand velocity, acceleration)
- `z-as-neural`: Z = LFP/Laplacian channels -- tests whether the latent state encodes subcortical dynamics

This yields **8 config pairs** (4 sessions x 2 experiment types) per framework, for **24 trained models** per framework (8 configs x 3 DBS variants each).

---

## 2. Data Preprocessing

**Script:** `preprocessing/package_recordings.py`  
**Config:** `preprocessing/participants_at_200Hz_scaled_1e6_raw_envelope.yaml`

### Preprocessing steps

1. **Resample** to 200 Hz.
2. **Scale** all signals by 1e6 (converts V to uV for numerical stability).
3. **Common Average Reference (CAR):** subtract mean across all ECoG channels at each timepoint.
4. **Notch filter** at 50, 100, 150, 200 Hz to suppress power line interference and harmonics.
5. **Narrowband decomposition:** bandpass into 17 frequency bands covering 4-93 Hz (gap at 47-53 Hz to avoid notch artefact):
   - Bands (Hz): 4-8, 8-12, 12-17, 17-22, 18-23, 23-28, 28-32, 32-37, 37-42, 42-47, 53-58, 58-63, 63-68, 68-73, 73-78, 78-83, 83-88, 88-93
   - Actual config has 17 `raw_bands` and 17 `envelope_bands`; band lists overlap by design to allow both raw and envelope views of same frequency.
6. **Hilbert envelope:** for each band, compute the analytic signal magnitude to produce the amplitude envelope. This yields 34 features per ECoG channel: 17 raw narrowband + 17 envelope signals.
7. **Output format:** Hive-partitioned Parquet stored as `resampled_recordings/participants_at_200Hz_scaled_1e6_raw_envelope/{participant_id=...}/{session=...}/{block=...}/0.parquet`.

### Chunk margin

During training, 400 samples (2 seconds at 200 Hz) are stripped from both ends of every trial via `_slice_data()` in `training/components/trainer.py`. This removes filter edge artefacts introduced during bandpass filtering. The `chunk_margin: 2` parameter in the preprocessing config specifies this 2-second margin.

---

## 3. Train / Val / Test Splits

**Script:** `training/precompute_splits.py`

Splits are precomputed once and stored alongside the data to ensure all frameworks use identical train/val/test assignments.

### Algorithm

```
min_n = min(number_of_on_blocks, number_of_off_blocks)
n_train = round(0.50 * min_n)   # default: 50%
n_test  = round(0.40 * min_n)   # default: 40%
n_val   = min_n - n_train - n_test   # remainder (~10%)
```

Blocks are assigned chronologically: first `n_train` pairs go to train, then `n_val` pairs to val, then `n_test` pairs to test. The balanced-pair design guarantees that each split contains at least one on-block and one off-block, which is required for the ChronoGroupsSplit classifier CV to function.

The 50/10/40 split was chosen specifically to ensure >= 2 blocks in val (the original 60/10/30 produced single-class val from certain sessions). The `min_n` floor prevents imbalanced splits when on/off block counts differ.

**Output files** (written to `{data_root}/splits/`):
- `train.parquet`, `val.parquet`, `test.parquet`
- Each row: `{participant_id, session, block, split, stim}` where `stim` is `on` or `off`.

**Runtime loading** (`training/components/trainer.py::split_data()`):
- Reads the precomputed split files.
- For `model_dbs_state = "on"`: filters to stim==on trials only.
- For `model_dbs_state = "off"`: filters to stim==off trials only.
- For `model_dbs_state = "both"`: uses all trials.
- Predictions and classification always run on `dbs_both` split assignments so all three model variants are evaluated on the same common trial set.

---

## 4. PSID Diagnostic

### ChronoGroupsSplit (shared splitter)

`utils/classification/splits.py::ChronoGroupsSplit` is the single CV splitter used across all cross-validation steps in the pipeline: mRMR fold voting, nx/n1 diagnostic CV, classification CV, and flipped classifier CV.

Algorithm:
- `min_n = min(n_on_blocks, n_off_blocks)`.
- Zip on-blocks and off-blocks chronologically into `min_n` pairs.
- Leave-one-pair-out: fold `i` holds out pair `i` (one on-block + one off-block).
- All training folds contain only the preceding pairs — no future data leaks.

Guarantees balanced classes in every fold. Number of folds = `min_n`.

---

**Script:** `training/pipelines/psid_diagnostic.py`  
**Purpose:** Select which Y channels and which hyperparameters (nx, n1) to use for PSID, VARMA, and DPAD.

The diagnostic runs four sequential stages.

### Stage 1: mRMR Feature Selection

**Feature extraction:**
- For each ECoG channel, compute per-trial log-standard-deviation: `log(std(channel_signal) + 1e-12)`.
- Epoch each trial into 2-second windows (400 samples), compute std per epoch. Average across all epochs to get one value per channel per trial.
- This produces a matrix `[n_trials x n_channels]` used as the mRMR input.

**mRMR algorithm (MIQ method):**
- Mutual information quotient: at each step, select the channel `j` that maximizes `MI(j; Z) / mean(MI(j; already_selected))`.
- MI estimated via continuous k-nearest-neighbors estimator (scikit-learn `mutual_info_regression`, k=3).
- Z target: first principal component of the Z matrix (behavioral or LFP channels).

**Cross-validation stabilization (`ChronoGroupsSplit`, see Section 4):**
- mRMR runs separately on each fold's training data.
- Final channel ranking: vote-aggregate across folds (channel ranked #k in fold i earns score `n_channels - k`; sum across folds; sort descending).
- Top 12 channels by aggregate score are selected.

**For z-as-neural configs**, an equivalent mRMR step selects the top-8 LFP/Laplacian channels using `mrmr_top_k_lfp_from_diagnostic()`.

### Stage 2: Cross-validate nx (latent state dimension)

- Candidate grid: `nx_grid = [20, 40, 60, 80, 100]`.
- For each candidate nx, run `cv_select_nx()`:
  - ChronoGroupsSplit CV on training data.
  - Fit PSID with `n1 = min(nz, nx)` (Z-dimension saturated).
  - Evaluate Z-reconstruction correlation coefficient (CC) on hold-out fold.
  - **Hankel workspace reuse:** the Hankel SVD (most expensive step) is computed once with `PSID.PSID(..., WS=None, return_WS=True)` and reused across subsequent calls with `PSID.PSID(..., WS=ws)`. This reduces cost by ~10x for large nx.
- Apply 1-SE rule: choose smallest nx whose mean CV score is within 1 standard error of the best mean score. This favors parsimonious models.
- In practice all sessions converge to **nx = 64** after the diagnostic.

### Stage 3: Cross-validate n1 (behavior-prioritized dimensions)

- Fixed nx from Stage 2. Candidate grid: `n1_grid = [1, 2, 4, 8, nx//2]` (capped at nx).
- For each candidate n1, same ChronoGroupsSplit CV on training data.
- Evaluate Z-reconstruction CC on hold-out.
- 1-SE rule to select smallest n1 that performs within 1 SE of best.

**Final n1 values per session:**

| Session        | Experiment type | n1 |
|----------------|-----------------|-----|
| PDI1_S2        | z-as-behavior   | 1   |
| PDI1_S2        | z-as-neural     | 1   |
| PDI1_S4        | z-as-behavior   | 8   |
| PDI1_S4        | z-as-neural     | 1   |
| PDI4_S2        | z-as-behavior   | 1   |
| PDI4_S2        | z-as-neural     | 2   |
| PDI4_S3        | z-as-behavior   | 1   |
| PDI4_S3        | z-as-neural     | 1   |

### Stage 4: Final fit and YAML patching

- Fit PSID on the full training set with the chosen nx and n1.
- Evaluate on train, val, test splits and log CC metrics.
- Call `amend_run_config()` to write the selected nx, n1, and channel lists back into the training YAML configs for PSID, VARMA, and DPAD. This is what generates the `training/setups/psid_PDI*.yaml`, `training/setups/varma_PDI*.yaml`, and `training/setups/dpad_modal/*.yaml` files.

---

## 5. Y Channel Selection Per Session

All three frameworks (PSID, VARMA, DPAD) use the same 12 ECoG Y channels per session, selected by the diagnostic mRMR. The channels are specified in the YAML config under `data.Y`.

### PDI1 Session 2

```
ECOG_3_gamma_88_93_raw     ECOG_3_beta_28_32_raw
ECOG_1_beta_18_23_raw      ECOG_4_gamma_73_78_raw
ECOG_2_gamma_58_63_raw     ECOG_1_gamma_88_93_raw
ECOG_1_gamma_37_42_env     ECOG_4_gamma_73_78_env
ECOG_1_beta_23_28_env      ECOG_4_gamma_53_58_env
ECOG_2_gamma_42_47_env     ECOG_1_alpha_8_12_env
```
High-gamma raw dominant (88-93 Hz), mixed with beta raw and gamma/alpha envelopes.

### PDI1 Session 4

```
ECOG_2_theta_4_8_raw       ECOG_3_gamma_68_73_raw
ECOG_1_gamma_73_78_raw     ECOG_3_gamma_63_68_raw
ECOG_4_gamma_53_58_raw     ECOG_1_alpha_8_12_raw
ECOG_1_gamma_37_42_env     ECOG_3_theta_4_8_env
ECOG_1_beta_13_18_env      ECOG_3_gamma_68_73_env
ECOG_4_gamma_42_47_env     ECOG_1_gamma_78_83_env
```
Theta raw appears (4-8 Hz), plus high-gamma raw; envelope channels span broader range.

### PDI4 Session 2

```
ECOG_4_gamma_83_88_raw     ECOG_1_beta_28_32_raw
ECOG_3_theta_4_8_raw       ECOG_1_gamma_83_88_raw
ECOG_3_beta_28_32_raw      ECOG_3_gamma_88_93_raw
ECOG_4_gamma_83_88_env     ECOG_3_gamma_88_93_env
ECOG_4_theta_4_8_env       ECOG_1_alpha_8_12_env
ECOG_4_beta_23_28_env      ECOG_3_gamma_42_47_env
```
High-gamma (83-93 Hz) + theta raw mix; similar envelope diversity.

### PDI4 Session 3

```
ECOG_4_beta_23_28_raw      ECOG_4_beta_28_32_raw
ECOG_3_beta_23_28_raw      ECOG_3_beta_28_32_raw
ECOG_2_beta_23_28_raw      ECOG_2_beta_28_32_raw
ECOG_4_beta_23_28_env      ECOG_4_beta_28_32_env
ECOG_3_beta_23_28_env      ECOG_3_beta_28_32_env
ECOG_2_beta_23_28_env      ECOG_2_beta_28_32_env
```
Strongly beta-dominated (23-32 Hz); all raw+envelope of same bands across 3 electrodes.

### Z channels (z-as-neural, PSID/DPAD)

For z-as-neural, Z consists of top-8 LFP Laplacian channels selected by mRMR on the LFP space, e.g. for PDI1_S2:
```
LAPLACIAN_13-15_LFP_gamma_88_93_raw    LAPLACIAN_14-16_LFP_gamma_83_88_raw
LAPLACIAN_14-16_LFP_gamma_63_68_raw    LAPLACIAN_13-15_LFP_gamma_58_63_raw
LAPLACIAN_12-14_LFP_gamma_88_93_raw    LAPLACIAN_11-13_LFP_gamma_88_93_raw
LAPLACIAN_9-11_LFP_beta_18_23_env      LAPLACIAN_14-16_LFP_gamma_63_68_env
```
For z-as-behavior, Z = `[tracing_velocity_x, tracing_acceleration_magnitude]` (2 channels, same across all sessions).

---

## 6. PSID: Model and Computation

**Library:** PSID 1.2.6 (`utils/frameworks/psid.py`)

### State-space model

PSID identifies a linear state-space system:

```
x(t+1) = A x(t) + w(t)         (state dynamics)
y(t)   = Cy x(t) + v(t)        (neural observation)
z(t)   = Cz x(t) + e(t)        (prioritized output)
```

where `x` is the `nx`-dimensional latent state. The first `n1` dimensions are prioritized to reconstruct `z` (behavioral or LFP targets). The remaining `nx - n1` dimensions capture residual `y` variance.

### Training

`PSIDWrapper.fit_ws()` calls `PSID.PSID(Y_concat, Z_concat, nx, n1, i=10, WS=ws, return_WS=True)`:
- `i=10`: Hankel lag parameter (block rows in the Hankel matrix).
- Trials concatenated along the time axis before fitting; PSID operates on the full session.
- Returns `idSys` object containing matrices: A (nx x nx), Cy (ny x nx), Cz (nz x nx), Q, R, S (noise covariance matrices).

### Eigenvalue clipping

After PSID fit, A matrix poles are checked for stability:

```python
eigvals, V = np.linalg.eig(A)
mags = np.abs(eigvals)
max_abs = 0.9999
unstable = mags > max_abs
scale = max_abs / mags[unstable]
eigvals[unstable] *= scale
A_clipped = V @ np.diag(eigvals) @ np.linalg.inv(V)
```

This clips eigenvalue modulus to <= 0.9999, ensuring the discrete-time system is stable (all poles inside the unit circle). Done in `_clip_A_eigenvalues()`.

### DARE refit

After eigenvalue clipping, the Kalman filter covariances (Q, R, S) are refit by re-solving the Discrete Algebraic Riccati Equation (DARE):

```python
from scipy.linalg import solve_discrete_are
P = solve_discrete_are(A.T, C.T, Q, R, e=None, s=S)
```

The `e=None` argument is required for scipy >= 1.16 compatibility (a scipy API change removed the default `e` parameter). Done in `_refit_dare_if_needed()`.

### Cz regression refit

After A clipping, the Z readout matrix Cz is re-estimated from the clipped system:

1. Run the Kalman filter (using clipped A and refitted DARE covariances) on training trials to produce latent state trajectories `X_hat`.
2. Regress Z against `X_hat` via ordinary least squares: `Cz = (Z.T @ X_hat) @ np.linalg.inv(X_hat.T @ X_hat)`.

This corrects the Cz that PSID estimated from the original (unclipped) A.

### A-powers cache

For efficient forecasting, precompute `A^1, A^2, ..., A^200` and store on `idSys`:

```python
idSys.A_powers = [np.eye(nx)]  # A^0
for k in range(1, 201):
    idSys.A_powers.append(idSys.A_powers[-1] @ A)
```

This enables O(1) per-step lookup during forecast. Done in `_cache_A_powers()`.

### Prediction (state inference)

`PSIDWrapper.predict()` runs the Kalman filter trial-by-trial:
- Input: Y (ny x T).
- Output: `Xp` (nx x T), `Zp = Cz @ Xp` (nz x T).
- Uses the steady-state Kalman gain `K = P @ C.T @ inv(C @ P @ C.T + R)` computed from DARE solution.

### Forecasting

`PSIDWrapper.forecast(m, Y_past)`:
1. Run Kalman filter on `Y_past` to get `x0 = Xp[-1]` (final state).
2. For each step `t` in 1..m: `Xf[t] = A_powers[t] @ x0` (zero-input propagation, noise = 0).
3. `Zf = Cz @ Xf`.

Horizon `m` is specified in seconds; converted to samples as `m_samples = round(m * sampling_freq)`. The h-grid `[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]` seconds is evaluated during the forecasts phase.

### Model storage

Trained PSID model stored as a binary-format object file:  
`results/psid/{experiment_name}_dbs_{on|off|both}/model_{timestamp}.pkl`

The timestamp (format: `YYYYMMDD_HHMMSS`) is used to identify and load the latest model across pipeline phases.

---

## 7. VARMA: Model and Computation

**File:** `utils/frameworks/varma.py`

### Model

VARMA(p, q) models the joint `[Y; Z]` process:

```
[Y(t); Z(t)] = sum_{k=1}^{p} AR_k [Y(t-k); Z(t-k)] + MA_1 eps(t-1) + eps(t)
```

with `p = q = 30` (autoregressive and moving-average lag both 30 samples at 200 Hz = 0.15 seconds).

### Training

Five steps in `VARMAWrapper.train()`:

**Step 1 - Per-channel z-score normalization:**
```python
mu, sigma = np.mean(Y_concat, axis=0), np.std(Y_concat, axis=0)
Y_norm = (Y_concat - mu) / (sigma + 1e-8)
```
Applied to the concatenated training trials.

**Step 2 - Hamming edge taper:**
- Create a 20-sample (0.1s at 200 Hz) Hamming window ramp.
- Apply the rising ramp to the first 20 samples and the falling ramp to the last 20 samples of each trial before concatenation.
- Purpose: smooth the discontinuities at trial boundaries caused by concatenation; prevents spectral leakage in the autoregressive fit.

**Step 3 - Long-VAR residual proxy for MA term:**
- Fit a VAR(30) (Vector AutoRegression of order 30) on the concatenated training data via OLS.
- Compute residuals `eps_hat = [Y; Z] - VAR30_fit`.
- These residuals serve as a proxy for the MA(1) innovations `eps(t-1)` in the VARMA model.

**Step 4 - Build VARMA design matrix:**
For each timepoint `t` (from 30 onwards), stack:
```
phi(t) = [Y(t-1); ... Y(t-30); Z(t-1); ...; Z(t-30); eps_hat(t-1)]
```
This is a regression design matrix of shape `[T-30, (ny+nz)*30 + (ny+nz)]`.

**Step 5 - Ridge OLS solve:**
```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(Phi, [Y; Z])
```
Coefficients yield the AR and MA matrices.

### Eigenvalue stabilization

After OLS, the AR companion matrix is checked for stability. The companion matrix stacks the AR lag matrices into a block form whose eigenvalues equal the VAR poles.

```python
gamma = max_root / max_pole  # scale factor if max_pole > max_root
for lag in range(1, p+1):
    AR_coeff[lag] *= gamma ** lag  # lag-dependent geometric shrinkage
```

The lag-dependent scaling `gamma^lag` ensures that higher-lag coefficients are shrunk more aggressively, respecting the geometric decay structure of stable AR processes.

### Prediction

VARMA has no latent state. `VARMAWrapper.predict()` returns `Xp = zeros((T, 0))` -- an empty latent matrix. VARMA does not participate in the latent-state classification step.

### Forecasting

`VARMAWrapper.forecast(m, Y_past)`:
1. Z-score `Y_past` using training statistics.
2. Run the Long-VAR(30) on `Y_past` to recover residual proxy `eps_hat`.
3. Recursive VARMA: for each step `t` in 1..m:
   - Shift history, append predicted value.
   - `eps_hat_future = 0` (no future innovations observed).
   - `[Y_pred(t); Z_pred(t)] = sum AR_k * history + MA_1 * eps_hat[-1]`
4. Return `Zf = Z_pred[1..m]`.

### Model storage

Binary-format object file at:  
`results/varma/{experiment_name}_dbs_{on|off|both}/model_{timestamp}.pkl`

Note: VARMA configs do not include a `classification` block because VARMA produces no latent state Xp. Only forecast-based classification (Xf) applies.

---

## 8. DPAD: Model and Computation

**Library:** DPAD 0.0.9 (TensorFlow/Keras backend)  
**File:** `utils/frameworks/dpad.py`  
**Method code:** `DPAD_uAKCzCy2HL32U`

### Model

DPAD extends PSID to nonlinear Y readouts using neural network layers, while keeping a linear A matrix for the state dynamics:

```
x(t+1) = A x(t) + w(t)         (linear dynamics)
y(t)   = f_Cy(x(t)) + v(t)    (nonlinear neural observation, f_Cy is a neural net)
z(t)   = Cz x(t) + e(t)        (linear Z readout)
```

### Training

`DPADWrapper.train()`:
- Transposes data to channels-first format: `[y.T for y in Y_trials]` -> list of `(ny, T)` arrays.
- Calls `DPADModel.fit(Y, Z, epochs=1000, checkpoint_every=100, fast=True, reuse_splits=False, steps_ahead=[1], steps_ahead_loss_weights=[1.0])`.
- `fast=True`: enables faster approximate fitting mode.
- `reuse_splits=False`: always recompute internal train/val split.
- `steps_ahead=[1]`: optimize 1-step-ahead prediction loss only.

**Hyperparameters (fixed across all sessions):**
- `nx = 64` (same as PSID)
- `n1 = 4` (fixed, not swept; DPAD n1 is less sensitive due to nonlinear readout)
- `epochs = 1000`
- `checkpoint_every = 100`

### Serialization

DPAD models contain TensorFlow computation graphs that cannot be directly stored as binary-format objects. The save/load protocol is:
1. `idSys.discardModels()`: removes the TF graph (Keras model), leaving only numpy matrices.
2. Save the stripped `idSys` as binary-format object file.
3. On load: `idSys.restoreModels()`: reconstructs the TF graph from the saved matrices.

This two-step protocol is required because TF graphs are not natively serializable as binary objects.

### Prediction

`DPADWrapper.predict(Y_trials)`:
- DPAD requires input length to be a multiple of the block size. Each trial is zero-padded to the next multiple: `pad = (-T) % block_samples`.
- After padding, calls `idSys.predict()` which runs the DPAD recurrent inference.
- Trims output back to original length.
- Returns `Xp` (nx x T) latent states and `Zp = Cz @ Xp` reconstructions.

### Forecasting

`DPADWrapper.forecast(m, Y_past)`:
- DPAD supports multi-step-ahead prediction via `set_steps_ahead([1, 2, ..., m_samples])`.
- This call is expensive (reconfigures Keras model). It is memoized: `_DPADFWK_FORECAST_CACHE` maps `(idSys_id, m_samples)` -> configured model.
- After `set_steps_ahead`, runs `idSys.predict(Y_past_padded)` which produces predictions at steps 1..m.
- Extracts the m-step-ahead predictions: `Xf = vstack([preds[2*m+i][-1:] for i in range(m)])`.

### DPAD-specific install

DPAD 0.0.9 pins PSID==1.2.5 as a transitive dependency. The Modal image build works around this:
1. Install DPAD (and all other packages) via requirements file -- this installs PSID 1.2.5.
2. Overwrite with: `pip install --no-deps PSID==1.2.6` -- installs our version without pulling in conflicting deps.

---

## 9. Predictions Phase

**Script:** `training/components/tester.py::run_predictions_incremental()`  
**Phase:** `predictions`

For each trained model (dbs_both/on/off), run Kalman/DPAD inference on every trial in the specified splits (train, val, test).

### Incremental execution

- Checks if the output Parquet partition already exists before processing each trial.
- If the partition exists: skip (allows resume after crash).
- If not: run `model.predict(Y_trial)`, compute `Zp = Cz @ Xp`, write partition.

### Output format (Hive-partitioned Parquet)

```
results/{framework}/{experiment_name}_dbs_{variant}/predictions/
  split={train|val|test}/
    participant_id={PDI1|PDI4}/
      session={2|4}/
        block={block_id}/
          trial={trial_id}/
            0.parquet
```

Each partition contains columns: `[Xp_0, ..., Xp_{nx-1}, Zp_0, ..., Zp_{nz-1}, Z_0, ..., Z_{nz-1}, Y_0, ..., Y_{ny-1}]`.

### System matrix export

After prediction, `compute_and_save_stats()` writes system matrices to HDF5:
```
results/{framework}/{experiment_name}_dbs_{variant}/system_matrices.h5
```
Contains: A, Cy, Cz, Q, R, S, eigenvalues of A (relevant and irrelevant subspaces), readout norms.

---

## 10. Forecasts Phase

**Phase:** `forecasts`  
**h-grid:** `[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]` seconds  
**m (context window):** 2.0 seconds (default)

For each horizon `h` in the h-grid, for each model variant (both/on/off), call `model.forecast(h, Y_past)` on each test trial:

```
Y_past = trial[:m_samples]      # first m seconds of Y
Y_future = trial[m_samples:]    # remainder
Zf = model.forecast(h, Y_past)  # predicted Z at h seconds ahead
Z_true = Z_trial[m_samples + h_samples]  # ground truth at h
```

Performance metric: **Pearson r** between `Zf` and `Z_true` across trials (per-channel, reported as mean CC).

Outputs stored to:
```
results/{framework}/{experiment_name}_dbs_{variant}/forecasts/h={h_value}/
```
Same Hive-partitioned Parquet structure as predictions.

### m_test_grid

During the classification sweep (not the forecasts phase), additional context window lengths are tested:
`m_test_grid = [0.5, 1.0, 1.5, 2.0]` seconds. This sweeps over how much past Y is provided at test time.

---

## 11. Classification Phase

**Script:** `training/sweep.py`  
**Purpose:** Decode DBS condition (on vs off) from latent state features.

### Feature sources

Four prediction-based feature sources:
- `Xp`: full nx-dimensional latent state (from model trained on both conditions).
- `Xp_1`: first n1 dimensions (Z-predictive subspace).
- `Xp_2`: remaining nx-n1 dimensions (Y-only subspace).
- `Xp_with_dbs`: Xp concatenated with a DBS condition indicator.

Four forecast-based feature sources:
- `Xf`, `Xf_1`, `Xf_2`, `Xf_with_dbs` (same decomposition, using forecast latent states).

### Classifier pipeline

`create_pipeline()` in `utils/classification/pipeline.py`:

```
FunctionTransformer(reorder_dims_for_mne)   # (n_trials, nx, T) -> MNE format
-> CSP(n_components=4, reg='ledoit_wolf', log=True)   # MNE CSP spatial filter
-> StandardScaler()
-> LDA(solver='lsqr', shrinkage='auto')
```

- **CSP (Common Spatial Patterns):** learns 4 spatial filters that maximize variance ratio between DBS-on and DBS-off classes. Ledoit-Wolf regularization is used for robust covariance estimation. Log-variance of filtered signals are the features.
- **LDA:** linear discriminant analysis with automatic shrinkage (Ledoit-Wolf); `lsqr` solver for numerical stability.

### ChronoGroupsSplit CV

Same `ChronoGroupsSplit` splitter defined in Section 4. Applied here to the epoched trial pool; folds hold out one on-block + one off-block pair chronologically.

### t_cut sweep

Classification is evaluated at multiple time cutoffs within each trial:
`t_cut_grid = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]` seconds.

For each `t_cut`:
1. Truncate each trial's Xp to the first `t_cut * sampling_freq` samples.
2. Re-epoch (see below), re-fit CSP+LDA, score balanced accuracy (BA).

This produces a "temporal learning curve" showing how quickly the latent state becomes discriminable.

### Epoching

`epoch_trial()` in `utils/classification/data.py`:
- `epoch_length = 0.5` seconds (100 samples at 200 Hz).
- `epoch_overlap = 0.25` seconds (50 samples) -> 50% overlap.
- Each trial is cut into overlapping 0.5s windows.
- CSP+LDA trains on these windows (n_epochs x nx x 100 tensor).

### Flipped (counterfactual) classifier

`_generate_flipped_latents()`:
- Takes the DBS-on model and DBS-off model.
- Runs both on the same Y window (regardless of true DBS condition).
- Labels the on-model's output as "on" and the off-model's output as "off".
- Trains CSP+LDA on these flipped labels.

This tests whether the dynamical structure of the two models is itself discriminable, independent of the actual DBS condition of the recording. A high BA here means the models learned different dynamics.

### Permutation test

After CV, if `cv_ba > perm_ba_gate (0.5)`:
- Draw 100 permuted label vectors (group-shuffle: shuffle entire block labels to preserve within-block structure).
- Refit CSP+LDA on each permutation, score BA.
- One-sided p-value: fraction of permutations with BA >= observed BA.

`n_splits = 5` (CV folds for final score aggregation); `n_permutations = 100`.

### Sub-source decomposition

Classification runs separately for each combination of:
- Feature source: {Xp, Xp_1, Xp_2, Xp_with_dbs}
- Model variant: {both, on, off}
- t_cut value from t_cut_grid

For forecasts, additionally swept over:
- h values from h_grid
- m_test values from m_test_grid

All results written to:
```
results/{framework}/{experiment_name}_dbs_both/classification/
  sweep_results.parquet
```

---

## 12. Modal Execution Infrastructure

**Script:** `training/pipelines/dpad_modal.py`  
**App:** `dpad-pipeline` on Modal

DPAD training is computationally intensive (1000 epochs, nx=64 nonlinear model). Training runs on Modal cloud GPU infrastructure.

### Image build

```python
modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu")
    .apt_install("git", "build-essential")
    .pip_install_from_requirements("modal_requirements.txt")
    .run_commands(
        "pip install --no-deps PSID==1.2.6",
        "pip install torch==2.9.1 scipy==1.16.0",
    )
    .env({"TF_CPP_MIN_LOG_LEVEL": "2"})
    .add_local_dir(PROJECT_ROOT, "/app", ignore=[...])
```

- Base: TF 2.15.0 with GPU support (CUDA).
- Two-step PSID install: first via requirements (gets 1.2.5 from DPAD dep), then overwrite with 1.2.6 using `--no-deps`.
- `torch==2.9.1` and `scipy==1.16.0` added explicitly (excluded from conda export).

### Volumes

| Volume name           | Mounted at                       | Purpose                  |
|-----------------------|----------------------------------|--------------------------|
| `dpad-data`           | `/app/resampled_recordings`      | Input ECoG data          |
| `dpad-results`        | `/app/results`                   | Model outputs            |
| `dpad-training-setups`| `/app/training/setups`           | Config YAML files        |

Volumes persist across runs. After each training or post-train container, `_results_vol.commit()` and `_training_vol.commit()` are called to flush writes.

### Fan-out execution

`_run_sweep()` performs a two-stage fan-out:

**Stage 1 - Train (24 containers):**
```python
train_args = [(cfg, side) for (cfg, sides) in entries for side in sides]
# entries: 8 configs x 3 sides (both/on/off) = 24 containers
for r in train_one.starmap(train_args):
    print(r)
```
- Each container: A10G GPU, 86400s (24h) timeout.
- All 24 train containers run in parallel; `starmap` blocks until all complete.

**Stage 2 - Post-train (8 containers):**
```python
post_args = [cfg for (cfg, _) in entries]
# 8 configs, one post-train container each
for r in run_post_train.map(post_args):
    print(r)
```
- Each container: A10G GPU, 86400s (24h) timeout.
- Runs predictions + forecasts + classification for one config.
- Stage 2 starts only after all Stage 1 containers have completed.

### Config discovery

`_load_sweep_entries(configs_dir, mode_filter)`:
- Globs `*.yaml` in the directory.
- Reads `experiment.type` from each config.
- If `mode_filter` is set, keeps only configs where `experiment.type == mode_filter`.
- Returns `[(config_path, sides)]` list.

### Sweep invocation

```bash
# All configs, all modes
modal run training/pipelines/dpad_modal.py::sweep \
    --configs-dir training/setups/dpad_modal

# Filter to one experiment type
modal run training/pipelines/dpad_modal.py::sweep \
    --configs-dir training/setups/dpad_modal --mode z-as-neural

# Only run training phase (skip post-train)
modal run training/pipelines/dpad_modal.py::sweep \
    --configs-dir training/setups/dpad_modal --phases train
```

---

## 13. Config Schema Reference

### Full annotated YAML structure

```yaml
framework:
  name: psid          # psid | varma | dpad | dpad_modal
  params:
    nx: 64            # latent state dimension (all sessions post-diagnostic)
    n1: 1             # Z-prioritized dims (session-specific, see table above)
    # PSID-only: i=10 (Hankel lag), internal to PSIDWrapper
    # VARMA-only:
    p: 30             # AR order (samples)
    q: 30             # MA order (samples; VARMA uses 1 MA lag in design matrix but p=q=30 for Long-VAR)
    # DPAD-only:
    epochs: 1000
    checkpoint_every: 100
    method_code: DPAD_uAKCzCy2HL32U
    fast: true
    reuse_splits: false
    steps_ahead: [1]
    steps_ahead_loss_weights: [1.0]

data:
  root: resampled_recordings/participants_at_200Hz_scaled_1e6_raw_envelope
  participant: PDI1
  session: 2
  Y:                  # 12 mRMR-selected ECoG channels (session-specific)
    - ECOG_3_gamma_88_93_raw
    - ...
  Z:                  # either 2 behavioral or 8 LFP channels
    - tracing_velocity_x
    - tracing_acceleration_magnitude
  sampling_frequency: 200

experiment:
  name: '{framework.name}_{experiment.type}_{data.participant}_S{data.session}_nx_{framework.params.nx}_n1_{framework.params.n1}_e{framework.params.epochs}'
  type: z-as-behavior   # z-as-behavior | z-as-neural

  train:
    model_dbs_state:
      - both            # train on all trials
      - 'on'            # train on DBS-on trials only
      - 'off'           # train on DBS-off trials only

  predictions:
    splits: [train, val, test]

  forecasts:
    default_h: 5.0      # seconds (not used by sweep; sweep uses h_grid)
    default_m: 2.0      # context window in seconds
    h_grid: [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    m_test_grid: [0.5, 1.0, 1.5, 2.0]

  classification:
    epoch_length: 0.5        # seconds per epoch
    epoch_overlap: 0.25      # overlap between epochs in seconds
    n_splits: 5              # CV folds for final score
    t_cut_grid: [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # seconds
    perm_ba_gate: 0.5        # min BA to trigger permutation test
    n_permutations: 100
    prediction_significance_window: 0.5    # seconds
    forecast_insignificance_window: 0.1    # seconds
    param_grid:
      LDA:
        classifier__solver: [lsqr]
        classifier__shrinkage: [auto]
    feature_sources_pred: [Xp, Xp_1, Xp_2, Xp_with_dbs]
    feature_sources_forecast: [Xf, Xf_1, Xf_2, Xf_with_dbs]

results:
  project_root: .
  save_dir: results/{framework.name}/{experiment.name}
  model_dir: results/{framework.name}/{experiment.name}
  log_dir: logs/{framework.name}
  checkpoint_dir: checkpoints/{framework.name}
  setups_dir: training/setups/{framework.name}
```

### Template interpolation

The `{framework.name}`, `{data.participant}`, etc. tokens in string values are resolved at runtime by the `DotDict` config loader (`training/config.py`). The resolved experiment name for PSID PDI1_S2 z-as-behavior with nx=64, n1=1, epochs=0 (PSID has no epoch count) would be:
```
psid_z-as-behavior_PDI1_S2_nx_64_n1_1
```

---

## 14. Full Pipeline Flowchart

```
Raw recordings
     |
     v
preprocessing/package_recordings.py
  - Resample to 200 Hz
  - Scale x 1e6
  - Notch filter 50/100/150/200 Hz
  - CAR
  - Narrowband decompose (17 raw + 17 envelope bands)
  - Write Hive-partitioned Parquet
     |
     v
training/precompute_splits.py
  - Balanced block-chronological split (50/10/40)
  - Write splits/*.parquet
     |
     v
training/pipelines/psid_diagnostic.py
  - Stage 1: mRMR (log-std features, ChronoCV fold voting, top-12 ECoG)
  - Stage 2: CV nx (Hankel reuse, 1-SE rule) -> nx=64
  - Stage 3: CV n1 (fixed nx, Z CC, 1-SE rule) -> n1 session-specific
  - Stage 4: Final fit + amend_run_config() -> patches all training YAMLs
     |
     +----------+----------+
     |          |          |
     v          v          v
   PSID       VARMA      DPAD (Modal)
  train()    train()    Modal fan-out
  fit_ws()   LongVAR    train_one x24
  clip_A     Ridge OLS  A10G GPU
  DARE       stabilize  86400s/container
  Cz refit   AR roots
  A^k cache
     |          |          |
     v          v          v
  model_*.pkl  model_*.pkl  model_*.pkl
     |
     v
predictions phase
  - run_predictions_incremental()
  - Trial-by-trial Kalman filter (PSID/DPAD) or recursive (VARMA)
  - Resume on crash (check partition exists)
  - Write Hive Parquet + HDF5 system matrices
     |
     v
forecasts phase
  - h in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0] s
  - PSID: x0=Kalman[-1], Xf[t]=A^t @ x0
  - VARMA: recursive prediction with zero future innovations
  - DPAD: set_steps_ahead (memoized), extract m-step heads
  - Metric: Pearson r
     |
     v
classification phase
  - Epoch trials (0.5s, 50% overlap)
  - Feature sources: Xp, Xp_1, Xp_2, Xp_with_dbs
  - Pipeline: CSP(4, ledoit_wolf) -> StandardScaler -> LDA(lsqr, auto)
  - ChronoGroupsSplit CV (leave-one-block-pair-out)
  - t_cut sweep [0.5..9.0 s]
  - Flipped (counterfactual) classifier
  - Permutation test (100 group-shuffles, if BA > 0.5)
  - Write sweep_results.parquet
```

---

## 15. Key Parameters Summary

| Parameter              | Value             | Where set          |
|------------------------|-------------------|--------------------|
| Sampling frequency     | 200 Hz            | Preprocessing config |
| Scale factor           | 1e6               | Preprocessing config |
| Chunk margin (strip)   | 400 samples (2s)  | trainer.py         |
| Notch freqs            | 50/100/150/200 Hz | Preprocessing config |
| Frequency bands        | 17 raw + 17 env   | Preprocessing config |
| Band range             | 4-93 Hz (gap 47-53)| Preprocessing config |
| Split ratio            | 50/10/40          | precompute_splits.py |
| mRMR top-k ECoG        | 12 channels       | psid_diagnostic.py |
| mRMR top-k LFP         | 8 channels        | psid_diagnostic.py |
| Hankel lag i           | 10                | psid.py            |
| nx (all sessions)      | 64                | YAML (post-diag)   |
| n1 (session-specific)  | 1-8               | YAML (post-diag)   |
| Eigenvalue clip        | 0.9999            | psid.py            |
| A-powers cached        | 1..200 steps      | psid.py            |
| VARMA AR order p       | 30 samples        | varma.py           |
| VARMA MA order q       | 1 (design matrix) | varma.py           |
| Long-VAR order         | 30                | varma.py           |
| Hamming edge taper     | 20 samples (0.1s) | varma.py           |
| Ridge alpha            | 1.0               | varma.py           |
| DPAD epochs            | 1000              | YAML               |
| DPAD n1                | 4 (fixed)         | YAML               |
| DPAD steps_ahead       | [1]               | YAML               |
| Modal GPU              | A10G              | dpad_modal.py      |
| Train timeout          | 86400s            | dpad_modal.py      |
| Post-train timeout     | 86400s            | dpad_modal.py      |
| h-grid (forecast)      | [0.5..5.0] s      | YAML               |
| default_m (forecast)   | 2.0 s             | YAML               |
| m_test_grid            | [0.5..2.0] s      | YAML               |
| epoch_length (classify)| 0.5 s             | YAML               |
| epoch_overlap          | 0.25 s (50%)      | YAML               |
| t_cut_grid             | [0.5..9.0] s      | YAML               |
| CSP n_components       | 4                 | classification/pipeline.py |
| CSP reg                | ledoit_wolf       | classification/pipeline.py |
| LDA solver             | lsqr              | classification/pipeline.py |
| LDA shrinkage          | auto              | classification/pipeline.py |
| n_permutations         | 100               | YAML               |
| perm_ba_gate           | 0.5               | YAML               |
| n_splits (CV)          | 5                 | YAML               |
