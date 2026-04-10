# Thesis Results HTML — Visual Inspection Report

**Date**: 2026-04-08
**File**: `thesis_results.html` (4.6 MB, 650 lines, 46 321 px rendered height)
**Stats**: 60 figures, 60 captions, 12 RMSE tables, 8 sections, 3 errors, 0 warnings

---

## Critical Issues

### 1. Three forecast RMSE horizon figures failed (RED ERROR boxes)

| Location | Error |
|----------|-------|
| Line 278 (Neural Reconstruction & Forecast) | `Failed to build neural forecast RMSE figure` |
| Line 555 (Behavioral Decoding & Forecast) | `Failed to build forecast RMSE figure` (velocity) |
| Line 556 (Behavioral Decoding & Forecast) | `Failed to build forecast RMSE figure` (acceleration) |

All three share the **same root cause**:

> `VARMA missing trial key ('PDI4', '3', '8', '11') for PSID triplet 'PDI4_S3'`

**Why it happens**: PDI4_S3 PSID was retrained at **200 Hz** (`psid_behavioral_PDI4_3_nx_25_n6_i30_dbs_both_200Hz_narrow_band`) while VARMA remains at **80 Hz** (`varma_PDI4_S3_dbs_both_narrow_band`). The different sampling rates produce different trial splits, so trial key `('PDI4', '3', '8', '11')` exists in the PSID test set but not in VARMA's. The error is raised at `forecast_horizon_rmse.py:174`.

**Fix**: Retrain VARMA for PDI4_S3 at 200 Hz, or add trial-key alignment tolerance in the forecast horizon code.

**Note**: Lines 555-556 show the same error text twice — the message doesn't distinguish which behavioral channel (velocity vs acceleration) failed.

### 2. "thesis default output label" leaks into 8 captions

All 8 behavioral decoding captions (lines 324, 352, 380, 408, 436, 464, 492, 520) end with the internal debug text `· thesis default output label`.

**Source**: `dashboard/thesis/compose.py:305` — `caption = f"{caption} · thesis default output label"` is appended whenever the declared output list is used instead of parquet metadata. This should be removed or replaced with the actual output channel name.

---

## Design & Consistency Issues

### 3. Trial count triple-counting — CAPTION BUG (FIXED)

| Figure | OFF trials | ON trials | Total | Source |
|--------|-----------|-----------|-------|--------|
| Pooled RMSE caption (line 121) | 126 | 189 | 315 | `html_report.py:417-418` |
| Neural band heatmap caption (line 609) | 42 | 63 | 105 | `neural_band_pearson.py` |

**Root cause**: `html_report.py:417-418` summed trial counts across all 3 model cells (`[0,2,4]` for OFF and `[1,3,5]` for ON), triple-counting each unique trial. The actual unique test trials per session (dbs_both split):
- PDI1_S2: 12 OFF / 18 ON = 30
- PDI1_S4: 7 OFF / 15 ON = 22
- PDI4_S2: 12 OFF / 12 ON = 24
- PDI4_S3: 11 OFF / 18 ON = 29 (DPAD) / 25 ON (PSID 200Hz), intersection = 29

Total unique: **42 OFF / 63 ON = 105** — the neural heatmap was correct all along.

**Fix applied**: Changed to `len(agg.trial_rmse[0])` / `len(agg.trial_rmse[1])` (single model). Same fix for per-session captions.

### 4. No page title (`<h1>`)

The HTML has no `<h1>` element — it begins directly with `<h2>Data and Preprocessing</h2>`. A top-level heading (e.g., "Thesis Results" or the thesis title) should be added for structure and accessibility.

### 5. Cross-eval / generalisation figures limited to PDI1 S2 only

The following specs all reference **only PDI1 S2**:
- `THESIS_WITHIN_CROSS` — within vs cross-condition RMSE
- `THESIS_CROSS_BLOCK` — cross-block boundary decoding (neural Y and behavioral Z)
- `THESIS_FORECAST_CHECKPOINT` — forecast from OFF / BOTH / ON checkpoints (neural and behavioral)

This means the entire generalisation narrative (a key thesis section) is built on a single participant-session. Adding at least one more session (e.g., PDI4 S3 where classification was significant) would strengthen the claims.

### 6. PDI4_S3 uses a fundamentally different PSID configuration

| Session | nx | n1 | iterations | Sampling |
|---------|----|----|------------|----------|
| PDI1_S2 | 80 | 12 | 20 | 80 Hz |
| PDI1_S4 | 75 | 6 | 20 | 80 Hz |
| PDI4_S2 | 80 | 12 | 20 | 80 Hz |
| **PDI4_S3** | **25** | **6** | **30** | **200 Hz** |

PDI4_S3 has 3× fewer latent dimensions, 50% more iterations, and 2.5× the sampling rate. No caption mentions this difference. This affects comparability in any pooled analysis (aggregate RMSE bars, neural band heatmaps, strip plots).

### 7. PDI4_S3 VARMA acceleration RMSE is anomalous

Caption at line 158:
> VARMA OFF: 2.083 ± 0.284. VARMA ON: 5.434 ± 0.248.

Typical VARMA acceleration RMSE for other sessions is 0.4–0.7. A value of **5.4** is 10× higher. Visually confirmed in chunk 11 — the bar towers over everything else. This could be a legitimate data effect (PDI4_S3 has different dynamics) or a symptom of the 200 Hz / 80 Hz mismatch between PSID and VARMA.

---

## Minor Issues

### 8. No figure numbering

Figures are identified by bold caption titles but not numbered (e.g., "Figure 1", "Figure 2"). For a thesis appendix this is fine, but if these feed into the main document, numbering aids cross-referencing.

### 9. Per-session bar charts lack individual trial dots

Pooled RMSE bar charts (chunks 5-6) overlay jittered dots for every test trial. Per-session bar charts (chunks 7-11) show only mean ± SEM bars without individual dots. This is a deliberate design choice (per-session has fewer trials) but creates a visual inconsistency within the Model Comparison section.

### 10. Flipped classification omits PDI1 S4 and PDI4 S2

The flipped classification heatmap caption (line 566) explains:
> "Only sessions with statistically significant non-flipped forecast results are included (PDI1 S2, PDI4 S3). PDI1 S4 and PDI4 S2 did not yield significant forecast classification and are omitted."

This is documented, but the standard classification heatmap (line 571) does include all 4 sessions. Having different session sets across the two heatmaps may confuse readers expecting a 1:1 comparison.

### 11. Dense captions

Several captions are extremely long single paragraphs (latent phase space = 7 sentences; Cy importance = 8 sentences). These read well as technical documentation but may be overwhelming in a thesis figure caption. Consider moving methodology details to the main text and keeping captions to 2–3 sentences.

---

## Section-by-Section Visual Inspection Summary

### Data and Preprocessing (6 figures)
- **Trial count bar chart**: Clear, correct labeling (PDI1_S2: 12/18, PDI1_S4: 7/15, PDI4_S2: 12/12, PDI4_S3: 11/25)
- **PSD plots (4 sessions)**: Consistent style, beta band dashed lines visible, SEM ribbons present
- **Tracing speed DBS comparison**: Multi-panel layout, DBS-ON vs DBS-OFF clearly distinguished
- **Verdict**: OK

### Model Comparison (14 figures)
- **Pooled RMSE bars (2)**: Jittered dots, error bars, Wilcoxon brackets present. VARMA dramatically lower RMSE than PSID/DPAD
- **Per-session RMSE bars (8)**: Consistent layout, proper axis labels. PDI4_S3 VARMA acceleration anomaly visually obvious
- **Strip plots (1)**: Correct symbol encoding (circle=OFF, square=ON), participant means visible
- **Session-mean RMSE caption**: Says "RMSE on z-scored tracing speed" — consistent with velocity channel
- **Verdict**: OK structurally; PDI4_S3 anomaly is a data concern

### Neural Reconstruction and Forecast (9 figures + 1 error)
- **Neural exemplar time series (4)**: Black=observed, colored=model predictions, RMSE ribbons visible. All 4 sessions render correctly
- **RMSE tables (4)**: Present above each exemplar with per-model, per-DBS values
- **Neural forecast exemplars (4)**: History/forecast split clear, vertical dashed line separates windows
- **Neural forecast RMSE horizon**: MISSING due to VARMA key error
- **Verdict**: Missing forecast RMSE figure is the one error here

### Behavioral Decoding and Forecast (19 figures + 2 errors)
- **Behavioral decoding time series (8)**: Velocity (4 sessions) + acceleration (4 sessions), RMSE tables above each
- **Behavioral forecast exemplars (8)**: History + forecast panels, consistent layout
- **Behavioral forecast RMSE horizon (2 expected, 0 rendered)**: Both failed (velocity + acceleration)
- **"thesis default output label"** text visible in all 8 decoding captions
- **Verdict**: Two missing forecast RMSE figures + caption text leak

### DBS Classification (4 figures)
- **Classification bar chart**: Xp, Xp1, Xp2, Xp+DBS groups clearly distinguished. Xp+DBS = 1.000 for all sessions (expected — DBS channel is trivially separable). Chance line at 0.5
- **Flipped classification heatmap**: 2 sessions × 4 feature groups. Color scale appropriate
- **Standard classification heatmap**: 4 sessions × 4 feature groups. Consistent with bar chart values
- **ROC curves**: Standard and flipped side by side, chance diagonal present
- **Verdict**: OK

### Generalisation and Latent Analysis (12 figures)
- **Within vs cross RMSE box plots**: Shows PSID within < cross for both DBS conditions. Box plot formatting correct, label encoding clear
- **Cross-block decoding neural Y**: 3 models × 2 conditions. OFF/BOTH/ON checkpoint colors distinguishable. Cross-condition predictions diverge as expected at boundary
- **Cross-block decoding behavioral Z**: Same layout. VARMA cross-condition trace shows high variability on mismatched trials — expected
- **Forecast checkpoints neural Y_future**: History + forecast layout. PSID shows clear checkpoint divergence. VARMA BOTH-trained forecast oscillates dramatically on OFF trial — worth noting in text
- **Forecast checkpoints behavioral Z_future**: PSID/DPAD show reasonable forecasts. VARMA forecasts are nearly flat (mean-converging)
- **Latent phase space**: KDE contours with trial trajectories. PDI4_S3 DPAD shows very elongated/linear distribution. DBS condition separation varies by session/model
- **Neural self-prediction heatmap**: Pearson r values by spectral band. Delta/Theta bands show highest r. Trial count discrepancy noted (issue #3)
- **Improved PSID vs Vanilla**: Clear improvement shown
- **Grid search heatmap**: 12 panels, red borders on selected configs, values annotated
- **Cy importance heatmap**: Beta band outline visible, per-panel normalization correct
- **Cz readout matrix**: Blue-red diverging scale, appropriate for signed weights
- **Verdict**: Structurally OK but limited to PDI1 S2 for cross-eval (issue #5)

### Laplacian LFP Prediction (2 figures)
- **Summary bar chart**: ECoG r (blue) >> LFP prediction r (red). Best r = 0.195 for PDI1 S4
- **Time series examples**: True signal (blue) vs PSID prediction (red dashed). Model captures low-frequency trends but misses oscillations — consistent with caption
- **Verdict**: OK

### Appendix (2 figures)
- **DPAD training curves**: 4-stage loss with vertical dashed boundaries. Train (solid) / val (dotted) separation clear
- **Data efficiency**: RMSE vs training set size. Shaded SEM bands. Improvement saturates at ~20–30 trials
- **Verdict**: OK

---

## Action Items (prioritized)

1. **Fix VARMA PDI4_S3 200 Hz alignment** — retrain VARMA at 200 Hz or implement trial-key fallback → unblocks 3 forecast RMSE figures
2. ~~Remove "thesis default output label"~~ **FIXED** in compose.py:305
3. ~~Investigate neural trial count~~ **RESOLVED**: caption bug (triple-counting). **FIXED** in html_report.py:417-418 and 464-465
4. **Add `<h1>` title** to the HTML
5. **Expand cross-eval to more sessions** (at least PDI4 S3 where classification is significant)
6. **Note PDI4_S3 config differences** in a footnote or supplementary table
7. **Consider de-duplicating error text** on lines 555-556 to show which channel failed
8. **PDI4_S3 behavioral PSID performance** — Pearson r ≈ 0 for Z, RMSE ≈ 1.0 (baseline). Only neural reconstruction is meaningful (r=0.65). This may require using different n1 or nx
