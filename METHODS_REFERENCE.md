# Methods — Sole Reference

Canonical source for the thesis Methods chapter. Covers preprocessing, channel
selection, all three modeling pipelines (PSID / VARMA / DPAD), classification,
and standing assumptions. Last update 2026-04-20 to reflect mRMR / i=100.

---

## 1. Participants and sessions

Four sessions from the Dold (2026) protocol were selected as the strongest
candidates for DBS-on/off classification under pilot VARMA:

| Session | Participant | Session # | Blocks recorded | Usable blocks |
|---|---|---|---|---|
| PDI1_S2 | PDI1 | 2 | 12 | 12 |
| PDI1_S4 | PDI1 | 4 | 10 | 10 (blocks 11, 12 missing: no iEEG files) |
| PDI4_S2 | PDI4 | 2 | 9  | 9 (blocks 1–3 missing: no iEEG files) |
| PDI4_S3 | PDI4 | 3 | 10 | 10 (blocks 7, 9 fragmented and dropped) |

All sessions follow the same 12-block × 12-trial nominal structure. Each block
is purely DBS-ON or DBS-OFF; alternation pattern is fixed per session.

## 2. Data acquisition

Simultaneous recordings of:
- **Cortical ECoG** (intracranial): 4 contacts, labelled `ECOG_1`–`ECOG_4`.
- **Deep-brain LFP**: 9 contacts on the DBS lead (Boston Scientific Vercise
  Cartesia directional lead), used to derive 7 laplacian bipolar contrasts
  `LAPLACIAN_{8-10, 9-11, 10-12, 11-13, 12-14, 13-15, 14-16}`.
- **Behavioral traces**: computer-screen cursor (x, y) during a visuomotor
  tracing task, sampled asynchronously; derived metrics:
  - `tracing_velocity_x` — horizontal velocity
  - `tracing_acceleration_magnitude` — acceleration magnitude

Trial protocol: participant traces a shape shown on-screen. `trial_type` codes
`{10, 21}` mark the active trial windows that feed downstream preprocessing.

## 3. Preprocessing pipeline

Configuration file: `preprocessing/participants_at_200Hz_scaled_1e6_narrow_band.yaml`
(executed by `preprocessing/package_recordings.py`).

| Stage | Parameter | Value |
|---|---|---|
| Resample | output rate | **200 Hz** |
| Notch | frequencies | 50 Hz, 100 Hz |
| Common average reference | `apply_car` | true (per modality) |
| Signal scaling | `scale_factor` | 1e6 (volts → microvolts) |
| Chunk margin | `chunk_margin` | 2 s (pre/post-trial padding for ECoG) |
| LFP handling | `drop_lfp` | false (retained for Q1 cross-modal decoding) |
| Pause filter | `max_pause_seconds` | 2.0 s |

**Narrow-band decomposition.** 15 bands per raw channel, labelled
`{name}_{lo}_{hi}_raw`:

| Family | Bands (Hz) |
|---|---|
| θ  | 4–8 |
| α  | 8–12 |
| β  | 12–17, 17–22, 22–27, 27–30 |
| γ  | 30–35, 35–40, 40–45, 45–50, 50–55, 55–60, 60–65, 70–75, 75–80 |

**Rationale.** Follows Sani et al. (2021). Narrow bands preserve spectral
resolution needed for Stage-2 SVD ranks in PSID; the 65–70 Hz range around
the DBS sub-harmonic is dropped and 130 Hz stimulation is notched out of the
upper edge of the analysis band.

Output layout (Hive-partitioned parquet):
```
resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band/
  participant_id={PID}/session={S}/block={B}/0.parquet
```
One row per trial, with list-of-float columns per channel × band.

**Alignment note**: within each trial, ECoG arrays have length
`margined_duration × fs` (with pre/post margin, typically 2600 samples at
200 Hz for a 9 s trial + 2×2 s margin). Behavioral tracing arrays are trial-
content only (no margin; 1800 samples for a 9 s trial). Downstream tools align
using the `time`, `chunk_margin`, and `margined_duration` fields on each row.

## 4. Quality control — trial cleaning

Applied inside the split creation step (`utils/split.py`):

1. **Fragmented blocks**: drop entire block if `is_fragmented == True` in the
   intermediate `data/participants_2/` table. (Typical cause: recording file
   is corrupt or truncated.)
2. **Protocol violations**: drop trials flagged `remove_due_to_protocol`
   (jumps > threshold in tracing, or trial duration < 9 s protocol minimum).
3. **Stale traces**: drop trials with any stationary cursor window ≥ 2.0 s
   (`max_pause_seconds` rule).

Reporting: per-session counts of removed trials are shown in
`thesis_sec1_data_verification.ipynb` (trial_count_barplot figure).

## 5. Feature set

For every session, each trial yields:

| Role | Channels | Total |
|---|---|---|
| Neural input (Y) | ECoG contacts 1–4 × 15 bands | **60** |
| Behavioral target (Z, behavioral mode) | `tracing_velocity_x`, `tracing_acceleration_magnitude` | **2** |
| LFP target (Z, laplacian mode) | `LAPLACIAN_14-16_LFP_{band}_raw` × 15 bands | **15** |
| Additional laplacian derivations (sanity) | d8-10, d9-11, d10-12, d11-13, d12-14, d13-15 × 15 bands each | 90 |

The laplacian d14-16 contrast was chosen as the canonical Q1 target because it
is spatially common across participants (most ventral pair of the DBS lead) and
is highly correlated with the other laplacian pairs on the same session (see
`report/thesis/figures/laplacian_channel_relatedness.png`).

## 6. Train / validation / test split

**Scheme**: within-session chronological, trial-level partitioning.

**Ratios** (`utils/split.py:create_splits`):

| Split | Fraction |
|---|---|
| Train | 50.0 % |
| Val | 12.5 % |
| Test | 37.5 % |

Determined by cumulative `n_epochs` (sample count per trial). Splits are
deterministic across modes and models — PSID behavioral and PSID laplacian
runs on the same session reuse the exact same time segments. VARMA and DPAD
copy the split parquets from the PSID variant to guarantee bit-exact alignment.

**Rationale for 50/12.5/37.5**: The original 60/10/30 produced single-class
validation windows on sessions with long consecutive DBS blocks. Widening val
to 12.5 % and using a 50/12.5/37.5 split ensures val spans at least two
DBS-condition transitions on every session. See
`report/thesis/figures/split_block_diagram.png` for the block alternation
pattern and cut placements per session.

## 7. Channel selection — K=8 mRMR

For each session × target side, 8 ECoG channels are selected from the 60
candidates using **minimum redundancy, maximum relevance** (mRMR):

- Relevance per channel: `mean_j |Pearson(Y_i, Z_j)|` averaged over the
  target channels.
- Redundancy: `|Pearson(Y_i, Y_k)|` across the remaining pool.
- Greedy selection: pick the channel with highest `relevance - mean(redundancy with already-selected)`.

Implementation: `scripts/overnight_all_sessions.py:mrmr_select`. Deterministic
for a given session + target pair (seeded numpy RNG).

**Why K=8**: compromise between dimensionality (lower K = faster / more
stable PSID fit at larger nx) and retained information. K not swept to plateau
due to time constraints; the chosen value was validated empirically on PDI4
cells (r_Yp ≈ 0.98–1.00 at nx=50).

The selected channel list is written into a training YAML under
`training/setups/psid/narrow_band_200Hz/overnight/` and carried through all
three pipelines via `--channels-from <yaml>`.

## 8. PSID pipeline

Entry: `scripts/pipeline_psid.py`. Phases 3–6 (grid search at Phase 1-2 is
skipped when channel set is pre-selected via `--channels-from`).

### 8.1 Model

PSID (Sani et al., 2021) with the PyPSID library (`from PSID import PSID`).
Two-stage subspace identification:

- **Stage 1**: projection of Z onto past Y → SVD selects `n1` behavior-relevant
  latent dimensions.
- **Stage 2**: SVD of the Y residual → selects `n2 = nx - n1` neural-residual
  dimensions.

Output: linear state-space model `idSys` with matrices A (nx × nx), Cy
(nY × nx), Cz (nZ × nx), and noise covariances Q, R, S, plus Kalman gain K.

### 8.2 Canonical hyperparameters

| Session | nx | n1 | i | Source |
|---|---|---|---|---|
| PDI1_S2 | 55 | 15 | 100 | `configs/diagnostic/elbow_choices.yaml` |
| PDI1_S4 | 50 | 10 | 100 | same |
| PDI4_S2 | 50 | 10 | 100 | same |
| PDI4_S3 | 50 | 10 | 100 | same |

- `i` = 100 → 0.5 s block-Hankel horizon at 200 Hz. Covers ≥ 2 cycles of θ
  (4 Hz → 250 ms) and well above Nyquist for all other bands. 130 Hz DBS
  stimulation is notched out of the analysis range.
- `nx` = latent state dimension (n1 behavior-relevant + n2 neural-residual).
- Target channels: behavioral mode = 2 tracing cols; laplacian mode = 15 LFP
  bands on d14-16.

**Stability modifications** (enabled for all runs):

- `backward_kalman: true` — RTS smoother on the forward Kalman posterior
  (provides access to future observations during training).
- `rescale_states: true` + `max_eigenvalue: 0.9999` — post-fit eigenvalue
  rescaling to ensure A has spectral radius ≤ 1 − ε, stabilising long-horizon
  forecasts.

**Retry-lower fallback** on SVD/DARE failures:
`RETRY_I_SEQUENCE = [75, 50, 45, 40, 35, 30]` descending. None fired on the
canonical i=100 run (conditioning OK with K=8 channels).

### 8.3 Pipeline phases

1. (skipped when `--channels-from` present) Grid search over (nx, n1).
2. (skipped) Best-CV-BA selection.
3. **Full training** × 4 variants per cell: `both`, `on`, `off`, `vanilla`.
   `vanilla` uses both conditions but the DBS label is hidden from the model
   during training (sanity comparison).
4. **Cross-condition evaluation**: the on-trained model is tested on off
   trials and vice versa; same for the off-trained model.
5. **Classification** (see §12): LDA on latent trajectories with the h × m
   grid for forecast-based inputs.
6. **Thesis HTML / specs update** (deprecated; skipped via `--skip-phase-6`).

### 8.4 Output layout

```
results/psid_{mode}_{PID}_{S}_nx_{nx}_n{n1}_i{i}{_vanilla}_dbs_{cond}_200Hz_narrow_band/
    model_{ts}.pkl               # serialised idSys (PyPSID LSSM object)
    model_{ts}_metadata.json     # nx, n1, i, channels, timestamps
    split/{train,val,test}.parquet
    test/test_results_{ts}.parquet/
      participant_id=…/session=…/block=…/trial=…/0.parquet
    cross_eval/                  # on-vs-off cross-evals
    classification/              # Phase 5 outputs
    config.yaml                  # training config used
```

Cross-eval dirs are named with `_eval_on` / `_eval_off` suffixes.

## 9. VARMA pipeline

Entry: `scripts/pipeline_varma.py`. Channel set inherited from PSID via
`--channels-from <yaml>` (same K=8 mRMR channels).

### 9.1 Model

Vector AutoRegressive Moving Average VARMA(p, q) fit per DBS condition via
`statsmodels.tsa.statespace.varmax` on the concatenated training-split time
series.

### 9.2 Canonical hyperparameters

| Param | Value |
|---|---|
| `P` (AR lags) | 30 |
| `Q` (MA lags) | 1 |
| `LONG_AR_LAGS` | 30 |
| `TRIAL_EDGE_TAPER_SEC` | 0.1 |
| Channels | K=8 mRMR from PSID yaml |

Splits copied from the PSID variant of the same cell / mode (bit-exact
alignment).

### 9.3 Pipeline phases (mRMR-era)

1. **Channel selection** — skipped; inherits from PSID yaml.
2. **Train** — 3 VARMA fits per cell (`both`, `on`, `off`).
3. **Cross-condition eval** — same on/off pattern as PSID.
4. **Classification** — SKIPPED per user instruction (Giedrius 2026-04-19).
   VARMA is not used as a classification feature source; `--end-phase 3`
   stops pipeline before Phase 4 fires.
5. **Thesis HTML** — deprecated, skipped.

### 9.4 Status caveat (2026-04-20)

Bobby + jacque VARMA models currently on disk (timestamps Apr 18 12:16–14:33)
were trained against the **pre-i=100** PSID splits. Their prediction/forecast
parquets use old alignment. Need a retrain against i=100 PSID variants. Jacque
VARMA laplacian is missing entirely (launcher's autodetect_psid_best didn't
scan i=100; see DATA_LOCATIONS.md §2).

## 10. DPAD pipeline

Entry: `scripts/pipeline_dpad.py`. **Behavioral mode only** (laplacian mode
is a pending refactor).

### 10.1 Model

DPAD (dynamical preferential-pursuit autoencoder; Sani et al., 2024) —
non-linear state-space generalisation of PSID with RNN encoder/decoder.
TensorFlow 2 backend via the upstream DPAD library.

### 10.2 Canonical hyperparameters

| Param | Value |
|---|---|
| Method code | `DPAD_uAKCzCy2HL32U` |
| Epochs | 3000 |
| Checkpoint every | 100 epochs |
| Channels | K=8 mRMR (inherited via `--channels-from`) |
| nx / n1 | inherited from PSID per cell (PDI1_S2: 55/15; others: 50/10) |
| psid_i | 100 (to find PSID splits) |

Splits copied from PSID via `pipeline_dpad._copy_splits_from_psid`.

### 10.3 Pipeline phases

1. **Train** × 3 conditions (both/on/off). Skip-if-exists: if `model_*.pkl`
   already in variant dir, skip training.
2. **Test (inference)** — `training.test --incremental --splits train val test`
   writes per-trial parquet partitions; no wall-clock timeout.
3. **Cross-condition evaluation** — same on/off logic as PSID.
4. **Classification** (see §12) — reads PSID's CV-best (h, m) as a fixed point
   (no DPAD-side grid search).
5. Deprecated / skipped.

### 10.4 Classification handoff (DPAD reuses PSID choices)

DPAD's Phase 4 calls `_find_psid_best_hm()` to read the (h, m) that maximised
CV-BA in PSID's own Phase 5. This is methodologically cleaner than running a
separate grid on DPAD: PSID selected the window, DPAD's permutation test is
conditionally valid given PSID's choice, avoiding selective-inference
contamination.

### 10.5 Output layout

```
results/dpad_behavioral_{PID}_{S}_nx_{nx}_n{n1}_e{epochs}_mrmr8_dbs_{cond}_200Hz_narrow_band/
    model_{ts}.pkl               (trained network weights + metadata)
    training_history.json
    {train,val,test}/             (Phase 2 parquet partitions)
```

## 11. Cross-condition evaluation

For each cell, two cross-eval runs are produced:

| Trained on | Evaluated on | Output dir tag |
|---|---|---|
| `dbs_on` | `dbs_off` trials | `_eval_off` |
| `dbs_off` | `dbs_on` trials | `_eval_on` |

Purpose: quantify how much of the predictive gain is DBS-state-specific vs.
shared dynamics. Used for:

- Within- vs cross-condition RMSE boxplots (§9.4 of RESULTS_PLAN).
- Flipped classifier (§12.2) — cross-model classification head.

## 12. Classification — DBS state inference

Entry: `classification/compute.py`, configured per-cell by yamls written by
each pipeline's Phase 4/5.

### 12.1 Feature sources

4 per cell, each tested independently:

| Label | Content | Dimension |
|---|---|---|
| `Xp` | full latent trajectory | nx |
| `Xp_1` | behavior-relevant subspace | n1 |
| `Xp_2` | neural-residual subspace | nx − n1 |
| `Xp_with_dbs` | `Xp` concatenated with the DBS label | nx + 1 (sanity ceiling; BA → 1.0 by label leakage) |

Features are extracted from **forecast-based** trajectories: at each sample,
the model is rolled forward `m` seconds using only the `h`-second preceding
history. Per trial, the resulting trajectory is averaged over an
epoch-length window (0.5 s, 0.25 s overlap) to produce the feature vector.

### 12.2 Feature-selection window — h × m grid

For **PSID**, the Phase 5 grid tests each (h, m):

| Axis | Values (s) |
|---|---|
| `h` (history) | 0.5, 1.0, 1.5, 2.5, 4.5 |
| `m` (forecast horizon) | 0.5, 1.0, 2.0 |

PSID picks the `(h, m)` that maximises CV balanced accuracy per feature
source, then runs a permutation test at that point only.

For **DPAD**, no grid: `_find_psid_best_hm()` reads PSID's best-CV (h, m) and
runs classification + perm test at that fixed point. See §10.4.

### 12.3 Classifier pipeline

Full sklearn `Pipeline` from `utils/classification.py:create_pipeline`:


```python
Pipeline([
    ("transpose", FunctionTransformer(reorder_dims_for_mne)),   # (trials, channels, time) → MNE layout
    ("csp",       CSP(n_components=4, reg="ledoit_wolf", log=True)),
    ("scaler",    StandardScaler()),
    ("classifier", LinearDiscriminantAnalysis()),
])
```

**CSP** (Common Spatial Patterns, `mne.decoding.CSP`): supervised spatial
filter that finds linear combinations of input channels whose variance ratio
between the two classes (DBS-ON vs DBS-OFF) is maximised. Parameters:

- `n_components=4` — keep the 4 most discriminative spatial filters (2 per
  class-extreme).
- `reg="ledoit_wolf"` — Ledoit–Wolf shrinkage on class-conditional covariance
  matrices (handles small-sample ill-conditioning when ny ≈ ntrials).
- `log=True` — take log of per-trial per-filter variance, standard for CSP
  features before linear classification.

**StandardScaler**: zero-mean, unit-variance per CSP feature across the
training pool.

**LDA** (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`):
discriminant classifier on the 4-dim CSP-log-variance feature vector.
Hyperparameters are set through the `GridSearchCV` `param_grid` rather than
inline on the `Pipeline` constructor (single-value grid, so in effect a
constant assignment — see `pipeline_psid.py:443–445` and
`pipeline_dpad.py:496–498`):

```python
"LDA": {
    "classifier__solver":    ["lsqr"],
    "classifier__shrinkage": ["auto"],
}
```

- `solver='lsqr'` — closed-form least-squares solution. Efficient for
  low-dim inputs (post-CSP we have 4 features).
- `shrinkage='auto'` — Ledoit–Wolf automatic shrinkage of the within-class
  covariance toward the identity. Regularises the Σ_W⁻¹ step that LDA needs
  when feature dimension is close to sample size or the estimated Σ_W is
  ill-conditioned. Required because the CSP output can have near-colinear
  components on small-sample cells.

**CV**: wrapped in a 5-fold chronological `ChronoGroupsSplit` that respects
trial ordering and block boundaries — no leakage of nearby trials into train
+ test of the same fold.

**Note on feature source**: the "feature source" column (Xp, Xp_1, Xp_2,
Xp_with_dbs; §12.1) selects which latent block is fed in as the input
channels to the CSP step. CSP then learns spatial filters on those latents;
LDA classifies on the CSP features. All four feature sources go through the
same pipeline.

### 12.4 Flipped classifier

Cross-model classification: models for `dbs_on`, `dbs_off`, `dbs_both` are
loaded simultaneously; each test trial is scored under **all three** model
heads. Asks whether the DBS signal is generic trajectory structure or
condition-specific. Uses the `*_flipped/<ts>/` classification dir per cell.

### 12.5 Permutation test

`apply_lda_permutation_to_results` in `utils/classification.py`:

- Shuffle labels within train fold, re-fit, measure CV BA.
- `n_permutations` default 100; BH-adjustable across cells × feature sources
  at the analysis stage.
- `p = (n_better + 1) / (n_perm + 1)` (reported in `results["permutation_test"]`).
- Minimum resolvable p ≈ 0.01.

### 12.6 Timeouts

All classification phases have `timeout=None` (per user instruction
2026-04-19) — run until done. Flipped mode formerly had an 8 h cap; now
uncapped.

## 13. Software stack & environments

| Component | Version |
|---|---|
| Python | 3.11 (miniconda `neuro` env on both bobby and jacque) |
| PyPSID | custom fork in `utils/frameworks.py` with RTS smoother hook |
| DPAD | upstream 2024; hotfix to `allX_steps` unbound var in `utils/frameworks.py` |
| VARMA | `statsmodels.tsa.statespace.varmax` |
| Data frames | `polars` (not pandas) |
| Plotting | `plotly` with paper-style helpers in `notebooks/thesis_style.py` |

Two compute hosts:

- **bobby** — RTX 3050 (4 GB VRAM), handles PDI4 cells + figure rendering.
- **jacque** — MX150 (2 GB VRAM), handles PDI1 cells. TF GPU memory growth
  forced via `TF_FORCE_GPU_ALLOW_GROWTH=true`. Per `reference_brain_server`
  memory, jacque has 8 CPUs + 25 GB swap.

## 14. Assumptions and caveats

1. **PSID linearity** — PSID assumes linear Gaussian dynamics. Non-linear
   effects (e.g., non-stationary gain changes under DBS) are approximated in
   the latent state transitions but may not be fully captured.

2. **Behavioral subspace orthogonality** — Stage-1 SVD enforces that `Xp_1`
   (behavior-relevant latents) is orthogonal in data space to `Xp_2` (neural
   residual). Downstream classification uses this split; a failure of the
   orthogonality assumption would leak information across the two feature
   sources.

3. **Cross-modal PSID at chance on LFP** — PSID's linear, behavior-
   supervised objective does not transfer cleanly to LFP reconstruction
   (Q1). Significant Xp classification (§12) on the same cells is still
   meaningful because classification operates on latent trajectory geometry,
   not Z-prediction fidelity.

4. **VARMA AR artefact** — VARMA's short-horizon prediction advantage on
   neural signals is dominated by autoregressive self-prediction, not
   cross-modal decoding. Any VARMA gain on laplacian Z is an AR artefact;
   flag in captions.

5. **Session asymmetry for Q3** — PDI1 sessions have no spectral DBS-on/off
   separation (see `dbs_significance_heatmap.png`); they serve as floor
   controls. The DBS-state-inference hypothesis is testable primarily on PDI4
   sessions. PDI1_S4's surprisingly high classification BA (~0.76) is an
   outlier to this expectation — worth re-examining the spectral heatmap
   for that cell specifically.

6. **Within-session chronology** — The 50/12.5/37.5 split is strictly
   chronological. No leakage of future trials into past-training, but later
   trials in test may reflect fatigue or learning effects. Accepted as the
   standard neural-decoding practice.

7. **DBS label leakage ceiling** — `Xp_with_dbs` always classifies at 1.0
   with high significance by construction. Report only as a sanity check,
   not a scientific result.

8. **Permutation p-value resolution** — With `n_perm = 100`, the minimum
   resolvable p-value is ~0.01. Cells at p = 0.05 are at the discrete
   boundary; Benjamini–Hochberg correction across feature sources within a
   cell may push some borderlines above 0.05.

9. **DPAD laplacian absent** — `pipeline_dpad.py` is behavioral-only
   (variant prefix, output channels, training subdir hardcoded). Adding
   laplacian DPAD requires a pipeline refactor and ~48 h retrain (see
   pipeline_runs.md TODO).

10. **Forecast rollout computational cost** — DPAD's forecast step is a
    per-trial RNN rollout (O(trials × m × t_samples) sequentially on GPU).
    Classification at Phase 4 is dominated by this cost, not by the LDA fit.

---

## 15. Pipeline command reference

| Operation | Command |
|---|---|
| Full mRMR + PSID + VARMA + DPAD, one host subset | `python scripts/overnight_all_sessions.py --subset {pdi4,pdi1} --algo {psid,varma,dpad} [--skip-mrmr-if-yaml-exists]` |
| Single-cell PSID recovery (e.g. Phase 5 only) | `python scripts/pipeline_psid.py --participant PDI4 --session 2 --mode behavioral --start-phase 5 --end-phase 5 --best-nx 50 --best-n1 10 --skip-phase-6` |
| Single-cell VARMA (Phase 3+, reuse mRMR yaml) | `python scripts/pipeline_varma.py --participant PDI4 --session 2 --mode behavioral --channels-from <yaml> --end-phase 3` |
| Single-cell DPAD recovery (after Phase 1 done) | `python scripts/pipeline_dpad.py --participant PDI4 --session 2 --nx 50 --n1 10 --psid-i 100 --channels-from <yaml> --start-phase 2 --skip-phase-5` |
| PSID SVD diagnostic (cached spectra) | `scripts/pipeline_psid_diagnostic.py` writes `spectra.parquet` per family under `results/diagnostic/{SESSION}_psid_spectra/psid/{ecog,laplacian}/`; reuse via `scripts/replot_psid_spectra.py` |

## 16. Related docs

- `RESULTS_PLAN.md` — chapter-level plan threading RQ1/RQ2/RQ3 through the
  figures.
- `DATA_LOCATIONS.md` — per-artefact inventory across bobby + jacque,
  including staleness flags.
- `pipeline_runs.md` — historical record of earlier runs (pre-mRMR top-5 era).
- `report/thesis/figures/technical_descriptions/_section_4_*_plan.md` — per-
  section figure plans with per-figure status and sources.
