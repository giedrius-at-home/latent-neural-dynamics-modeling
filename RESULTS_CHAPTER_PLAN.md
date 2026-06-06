# Results Chapter — Figure Plan

Three research questions. Reconstruction (1-step) and forecast (m-step) unified per Q —
identical figure structure, different horizon.
Design principle: subplots where layout calls for it, compose in LaTeX otherwise.

**Per-session layout rule:** never combine three models side-by-side (scales differ).
Use 3 separate subplots (one per model), each with 4-session rows.

**Statistical testing (applies to all metric figures):**
- Between-model: paired t-test on per-session metric vectors.
- Against zero: 1-sample t-test, Bonferroni-corrected across sessions.
- Independence assumption: trial-level samples assumed approximately independent
  (autocorrelation decays within-trial for neural data); add caveat if using step-level metrics.

---

## 3.0  Data Characterisation & Diagnostics

### 3.0.1  Raw data — sec1

**fig_005_dbs_significance_heatmap.png — Channel x session DBS significance heatmap**
- Heatmap: sessions x channels, colour = -log10(p) of DBS-on vs DBS-off PSD difference per band
- Justifies mRMR channel picks; baseline for spectral separability
- Source: sec1

*Appendix: fig_002/003 PSD curves, fig_beh behavioral traces, fig_trials/split counts*

---

### 3.0.2  Model diagnostics — sec2a

**fig_044_mrmr_selection.png — mRMR channel selection heatmap**
- Top-6 ECoG + top-6 LFP = 12 channels per session; session-stable vs session-specific picks
- Cross-check against fig_005
- Source: sec2a

*Appendix: fig_045 corrmat, fig_046 relevance bars, fig_039 DPAD training curves*

---

## Q1 — How well do PSID/DPAD latents predict subcortical LFP and cortical ECoG?

Y target = 12 ECoG features (cortical). Z target = top-8 LFP (subcortical).
Both Y and Z equally important. VARMA included as AR baseline.

### Q1.1  1-step-ahead (reconstruction)

**fig_q1_recon_raincloud.png — Pooled Y + Z reconstruction quality overview**
- Raincloud: Yp vs Y and Zp vs Z across sessions, PSID/DPAD/VARMA
- Brief overview; not focal result

**fig_q1_recon_decomp.png — Signal decomposition on Yp + Zp (amplitude / inst. freq / phase)**
- 3 panels per model:
  - Amplitude: Hilbert envelope correlation, Yp vs Y and Zp vs Z
  - Instantaneous frequency: correlation of d(phase)/dt, Yp vs Y and Zp vs Z
  - Phase: circular correlation or PLV, Yp vs Y and Zp vs Z
- What aspect of the neural signal is captured at 1-step

**fig_latent_X1X2.png — X1 vs X2 subspace information content (PSID)**
- Reconstruct Y + Z using X1-only vs X2-only readout; compare RMSE/r
- Which subspace drives Y and Z reconstruction

**fig_061_cy_importance.png — PSID Cy readout heatmap**
- |Cy|: which latent dims drive each Y channel

**fig_062_cz_readout.png — PSID Cz readout matrix**
- |Cz|: which latent dims drive Z; compare Cy vs Cz loading

*Appendix: per-channel boxplots (Zp vs Z per LFP channel), per-session strip plots*

---

### Q1.2  m-step-ahead (forecast)

**fig_q1_forecast_decay.png — Y + Z forecast accuracy vs horizon [PRIMARY]**
- RMSE + Pearson r vs h [0 ... 2 s], one line per model
- Shaded band = quantile 0.10-0.95 across sessions
- Annotate: rate-of-change flattens after ~0.5 s; verify 0.95 stable from 500 ms

**fig_q1_forecast_decomp.png — Signal decomposition on Yf + Zf at 0.5 s (amplitude / inst. freq / phase)**
- Same 3-panel structure as fig_q1_recon_decomp, applied to Yf + Zf
- Which signal aspects survive m-step forecast horizon

**fig_A_dynamics.png — A-matrix: A_both vs A_on vs A_off (PSID)**
- Eigenvalue spectra overlay or per-eigenvalue Cohen's d across DBS conditions
- Does DBS reshape latent dynamics, not just latent state value?

*Appendix: per-session 4-panel — (ECoG->LFP) and (ECoG->ECoG recon), one model per subplot*

---

## Q2 — How well do PSID/DPAD latents predict tracing kinematics and cortical ECoG?

Y target = 12 ECoG features. Z target = tracing velocity_x and acceleration_magnitude.
Both Y and Z equally important. VARMA included as AR baseline.

### Q2.1  1-step-ahead (reconstruction)

**fig_q2_recon_raincloud.png — Pooled Y + Z reconstruction quality overview**
- Raincloud: Yp vs Y and Zp vs Z across sessions, PSID/DPAD/VARMA
- Brief overview; not focal result

**fig_q2_recon_decomp.png — Signal decomposition on Yp + Zp (amplitude / inst. freq / phase)**
- Same 3-panel structure as Q1, applied to behavioral Y + Z targets
- What aspect of ECoG and kinematics is captured at 1-step

*Appendix: Figs 9-16 per-session strip plots; per-feature boxplots*

---

### Q2.2  m-step-ahead (forecast)

**fig_q2_forecast_decay.png — Y + Z forecast accuracy vs horizon [PRIMARY]**
- RMSE + Pearson r vs h [0 ... 2 s], one line per model
- Shaded band = quantile 0.10-0.95; annotate plateau after ~0.5 s

**fig_q2_forecast_decomp.png — Signal decomposition on Yf + Zf at 0.5 s (amplitude / inst. freq / phase)**
- Same 3-panel structure as Q1 forecast decomp, behavioral Y + Z targets

*Appendix: per-session 4-panel — (ECoG->LFP forecast) and (ECoG->kinematics forecast)*

---

## Q3 — Do learned latents separate stimulation-on from stimulation-off without the DBS label?

Source: sec5, sec6. PSID + VARMA (DPAD parquets pending).

**Feature source inventory (columns in classification heatmaps):**
- Prediction-based: Xp, Xp1, Xp2, Xp+dbs
- Forecast-based: Xf, Xf1, Xf2, Xf+dbs
- Model-conditioned: Xf from model trained DBS-on vs DBS-off (cross-condition)
- Cross-label (flipped): LDA trained on one condition, scored on other — adversarial null

**fig_050_classification_heatmap.png — cv_ba heatmap, sessions x feature sources**
- All feature sources above as columns; 0.5 anchored midpoint; permutation stars

**fig_051_flipped_heatmap.png — Flipped-label BA (adversarial null)**
- Near chance = model forecasts DBS-agnostic

**fig_052b_gap_heatmap.png — Generalisation gap (within - flipped BA)**
- Positive = DBS-state-specific structure in latent

**fig_053_tcut_Xp.png — Xp BA vs time window (PSID only)**
- 0.5 s windows; mark onset of significance (first window > chance, Bonferroni-corrected)
- How quickly does latent state become DBS-informative?

**fig_054_tcut_Xf.png — Xf BA vs time window (PSID only)**
- 0.1 s windows; mark onset of insignificance
- Links to forecast decay plateau in Q1.2/Q2.2

**fig_059_latent_phase.png — Latent phase space KDE**
- 2D KDE of Xp dims by DBS condition; visible separation vs low BA = classifier bottleneck

**fig_060_classification_dimensionality.png — Classification vs dimensionality grid**
- cv_ba heatmap over nx x n1; is model dimension-saturated?

**fig_063_data_efficiency.png — Data efficiency curve**
- BA vs training data fraction (k-fold CV inside training pool only)
- Saturated at 30% = signal quality is limit, not data quantity

*Note: DPAD classification parquets not yet available. Add DPAD row when parquets arrive.*

---

## Appendix

**sec1:** fig_002/003 PSD curves, fig_beh, fig_trials/split counts

**sec2a:** fig_045 corrmat, fig_046 relevance bars, fig_039 DPAD training curves

**sec2b/2c:** per-session strip plots; per-channel/per-feature boxplots

**sec2e:** Figs 18-21, 29-36, 50-55 — exemplar single-trial overlays (qualitative)

**sec7:** A-matrix eigenvalue spectrum + Cohen's d per latent dim between DBS conditions

**sec8:** Per-trial RMSE/r heatmaps over session timeline — non-stationarity audit
