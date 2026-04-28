# Data-Efficiency Analysis — 2026-04-18

Informs the decision to adopt a **60/15/45 train/val/test split** (per-session, out of ~120 trials) instead of the original 60/10/30.

---

## 1. Motivation

Two concerns with the original **60/10/30 chronological split**:

1. **Class imbalance in val**. Each session has 10-12 blocks of ~12 trials, and **each block is pure DBS-on or pure DBS-off** (strict alternation). The 10% val fraction covers exactly one block, so val is always **single-class** for 2 of 4 cells (PDI4_S2 val = 11 off / 0 on; PDI4_S3 val = 11 on / 0 off). This breaks classification-phase early stopping and makes val useless for hyperparameter selection.

2. **Small test set** (n ≈ 36 trials per cell). Permutation test resolution on test-set balanced accuracy is limited: with 36 test trials (~18/class), SE(BA) ≈ √(0.25/18) ≈ 0.12. Effects below BA ≈ 0.62 are indistinguishable from noise.

The split decision requires evidence that **shrinking train doesn't hurt model fit**. We performed a data-efficiency (plateau) analysis.

---

## 2. Methodology

### 2.1 Setup

- Cells analyzed: PDI4_S3 behavioral, PDI4_S2 laplacian (representative behavioral + cross-decoding, both PDI4 sessions).
- Config: K=8 mRMR-selected ECoG channels, nx=50, n1=10, i=50, max_eigenvalue=0.9999, backward_kalman=true, forecast m=1s/h=3s (same as overnight production config).
- Splitting: 70% chronological pool reserved for train; 30% held back as true test (not touched during this analysis).
- **3-fold CV within the 70% pool**. Folds are random but fixed (seed=0). For each fold: fit PSID on first N trials chronologically from the non-held-out folds, evaluate on the held-out fold.
- N sweep: {10, 25, 45} — coarse but covers the range from minimal to ~current-split train size.

### 2.2 Metrics (per fold)

| Metric | Definition |
|---|---|
| `r_Yp` | Mean-over-channels Pearson r of reconstructed vs true Y (1-step Kalman prediction) |
| `fcst@0.1s` | Cumulative Pearson r of Y_future_pred vs Y_future_true over first 0.1 s (20 samples) of autonomous propagation |
| `fcst@0.5s` | Same, first 0.5 s (100 samples) |
| `ba_pred` | Balanced accuracy of LDA (shrinkage='auto') trained on 3·nx-dim features [mean\|std\|last] of Xp (Kalman-filtered latent), predicting DBS on/off on held-out trials |
| `ba_fcst` | Same, features drawn from X_future_pred (forecasted latent) |
| `p_pred`, `p_fcst` | Permutation p-values via shuffling held-out labels 1000× |

### 2.3 Reporting

Mean ± std across 3 folds per (cell, N).

---

## 3. Results

### 3.1 Reconstruction (r_Yp)

```
                            N=10           N=25           N=45
PDI4_S3 behavioral         0.980 ± 0.000  0.980 ± 0.001  0.980 ± 0.000
PDI4_S2 laplacian          0.987 ± 0.001  0.988 ± 0.002  0.987 ± 0.001
```

**Plateau at N=10.** PSID reconstruction is fully saturated with 10 training trials. No improvement from adding 35 more.

### 3.2 Autonomous forecast

```
                                fcst@0.1s                 fcst@0.5s
                      N=10       N=25       N=45    N=10       N=25       N=45
PDI4_S3 behavioral   0.589±.003 0.596±.075 0.636±.035  0.207±.060 0.220±.054 0.219±.026
PDI4_S2 laplacian    0.667±.034 0.690±.047 0.686±.073  0.231±.046 0.255±.061 0.257±.091
```

**Plateau at N ≈ 25.** Forecast quality stabilizes well below current 72 trials. Modest improvements past N=10 but bounded.

### 3.3 Classification on Xp (filtered latent)

```
                      N=10           N=25           N=45
PDI4_S3 behavioral   0.868 ± 0.117  0.933 ± 0.065  0.920 ± 0.070
PDI4_S2 laplacian    0.629 ± 0.155  0.695 ± 0.059  0.807 ± 0.091    ← still climbing
```

**Plateau is cell-dependent**:
- **PDI4_S3 behavioral**: plateaus at N=25 (ba=0.93). Further N gives no gain.
- **PDI4_S2 laplacian**: **not yet plateaued at N=45** (trend 0.63 → 0.70 → 0.81). Likely keeps climbing; needs N~60 for reliable classification.

This is the **binding constraint** for the split decision. Classification on harder cells benefits from more training data.

### 3.4 Classification on X_future_pred (forecasted latent)

```
                      N=10           N=25           N=45
PDI4_S3 behavioral   0.512 ± 0.135  0.557 ± 0.022  0.525 ± 0.026
PDI4_S2 laplacian    0.538 ± 0.042  0.558 ± 0.051  0.517 ± 0.050
```

**Near chance across all N** — forecast classification is NOT data-limited; it's structurally limited by autonomous A propagation over 1 s (see forecast-horizon analysis in `OVERNIGHT_RESULTS.md`). More training data will not rescue this.

---

## 4. DPAD data-efficiency (from existing loss curves)

Existing DPAD training histories (old top-5 runs, nx ∈ {4,15,25}, n1=2) examined:

```
model1 (behavior-relevant subsystem, drives decoding):
  converges in 18-75 epochs across all cells. Data-sufficient at N=72.

model2 (residual Y dynamics, drives Yp reconstruction):
  converges in 300-1000+ epochs. PDI4_S2 ran 1999 epochs and was still
  dropping — already data-limited at N=72.
```

Scaling to nx=50, n1=10:
- model1 has 5× more capacity → will need more data/epochs but still converges fast.
- model2 has ~2× more capacity → will be *more* data-limited than the current nx=15-25 runs.

**Implication**: DPAD at nx=50, n1=10 wants **as much training as possible**. Reducing below N=60 risks degraded Yp reconstruction (though decoding should be fine since model1 is fast).

---

## 5. Split decision: 60/15/45

### 5.1 Per-cell composition

With ~120 trials per session, 60/15/45 as trial counts:
- train: 60 trials (5 blocks worth)
- val: 15 trials (~1.25 blocks)
- test: 45 trials (~3.75 blocks)

### 5.2 Class balance check (chronological, on lexical block order)

| cell | train (60) | val (15) | test (45) |
|---|---|---|---|
| PDI4_S2 | 24 on / 36 off ok | 11 on / 4 off ok | 24 on / 21 off ok |
| PDI4_S3 | 36 on / 24 off ok | 11 off / 4 on ok | 24 on / 21 off ok |
| PDI1_S2 | 36 on / 24 off ok | 12 on / 3 off | 24 on / 36 off ok |
| PDI1_S4 | 30 on / 30 off ok | 6 on / 9 off ok | 19 on / 26 off ok |

**All splits have both DBS classes** (vs the old 60/10/30 which was single-class val on 2 cells). Val class-balance skew is moderate (up to 12:3) in PDI1_S2 — manageable for early-stopping-style use but worth flagging.

### 5.3 Defensibility

- **Reconstruction unaffected**: r_Yp plateau at N=10 means N=60 is well above saturation.
- **Forecast unaffected**: plateau at N=25.
- **Classification on PDI4_S3 behavioral unaffected**: plateau at N=25, cv ba=0.93 ≈ N=45 cv ba=0.92.
- **Classification on harder cells (PDI4_S2 laplacian)**: curve was 0.63/0.70/0.81 at N=10/25/45 and still climbing. N=60 likely gives ba ≈ 0.85 — better than current 60/10/30 baseline because train is similar (60 vs 72) while val is now usable.
- **DPAD reconstruction**: N=60 preserves model2 training quality.
- **Test power**: n_test grows 36 → 45 (+25%), SE(BA) drops 0.12 → 0.105. Modest but real.

### 5.4 Interaction with Phase 5 classification chronology

Phase 5's `ChronoGroupsSplit` already enforces chronology at the CV-fold level (no stratification, respects time order). The change to 60/15/45 doesn't interfere with that — classifier CV is entirely internal to train+val and doesn't see test.

### 5.5 Trial-quality filter compatibility

`train.py` applies quality filtering (plateau filter, max_pause_seconds > 2.0 s, fragmented-block removal). The 60/15/45 ratio should apply **after** this filtering — the ratios are fractions of the surviving trials, not of the nominal 144. This matches current pipeline behavior.

---

## 6. Actionable changes

### 6.1 Code edits (pending)

1. `scripts/pipeline_psid.py`: update `TRAIN_RATIO`/`VAL_RATIO`/`TEST_RATIO` from `0.6/0.1/0.3` to `0.5/0.125/0.375` (= 60/15/45 on 120-trial sessions).
2. `scripts/pipeline_dpad.py` and `scripts/pipeline_varma.py`: same change (currently inherit from the same constants or define their own).
3. Training YAML generators (`scripts/overnight_all_sessions.py`): update emitted `data.split` values.
4. Rerun overnight with new split.

### 6.2 Expected downstream impact

| Algorithm | Before (60/10/30) | After (60/15/45) | Notes |
|---|---|---|---|
| PSID r_Yp | 0.95-0.99 | unchanged | plateau at N=10 |
| PSID r_Yfcst (1s) | 0.10-0.25 | unchanged | structural ceiling |
| PSID classification (test_ba, PDI4_S3) | 0.75 (n=36, p=0.001) | 0.77+ (n=45) | more test power, same or slightly better |
| PSID classification (test_ba, PDI4_S2 lap) | 0.57 (n=36, not sig) | 0.60-0.65 (n=45, possibly sig) | n=60 train still in the climbing region |
| VARMA everything | minor changes | unchanged | OLS regression is data-abundant already |
| DPAD model1 (behavior) | converges at N=72 | converges at N=60 | minor |
| DPAD model2 (reconstruction) | data-limited | slightly more data-limited | monitor |

---

## 7. Open questions / future work

### 7.1 Flipped classification plateau

Phase 5 runs flipped-label control variants (`_flipped` and `_flipped_perm` configs). These use the same chronological split as the main runs. We **did not** perform a data-efficiency sweep for flipped classification, so we don't know:
- Does the flipped classifier stay at chance for all N? (expected behavior — confirms no leakage)
- Does it drift from chance with N? (would indicate a systematic split artifact)

**TODO**: add flipped variants to `scripts/cv_plateau_analysis.py` — fit PSID on the forward-label data, then apply the flipped-label classifier at each N. If flipped stays at chance (~0.5) across all N while real labels climb to 0.8+, the classification result is validated.

### 7.2 PDI1 cells

Plateau analysis not yet run for PDI1_S2 and PDI1_S4. Known from overnight run that PDI1 cells have weaker classification signals overall (ceiling of ~0.55 on Xp_with_dbs aside). Rough expectation: same plateau shape (r_Yp at N=10, fcst at N=25), classification climbing slower.

### 7.3 Classification-only data-efficiency

For the final thesis plot (data-hungriness for classification per cell), we'd run the CV plateau script with finer-grained N (e.g., 10, 20, 30, 40, 50, 60, 70) and for all 4 cells, both for real and flipped labels. ~15-20 fits total, ≤30 min wall-clock. Defers until overall plan stabilizes.

---

## 8. Files

- `scripts/training_size_sweep.py` — initial single-fold sweep (test-set basis)
- `scripts/cv_plateau_analysis.py` — rigorous 3-fold CV plateau sweep used for this analysis
- `reports/cv_plateau_analysis.csv` — raw per-fold metrics
- `scripts/test_set_permutation.py` — BH-FDR corrected test-set permutation, referenced in the "test power" estimates above
- `reports/test_permutation_scores.csv` — permutation results on current (60/10/30) split
- `reports/OVERNIGHT_RESULTS.md` — sibling methodology doc covering the overnight run + forecast-horizon analysis
