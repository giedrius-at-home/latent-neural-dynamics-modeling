# Results Chapter — Figure Plan

Structure: four research questions. Each question answered by both experiment types
(z-as-behavior, z-as-neural) with all relevant metrics together.
Band-limited analysis (fig_A/B/C) is part of prediction/forecast evidence, not standalone.
Design principle: one plot per figure, compose multi-panel layouts in LaTeX.

---

## 3.0  Data Characterisation & Diagnostics

### 3.0.1  Raw data — sec1

**fig_005_dbs_significance_heatmap.png — Channel x session DBS significance heatmap**
- Heatmap: sessions x channels, colour = -log10(p) of DBS-on vs DBS-off PSD difference per band
- Which channels carry DBS-state information; justifies mRMR picks (fig_044)
- Source: sec1

**[NEW] fig_spectral_separability — Frequency-band information content per session**
- Per-channel, per-band: how much signal is DBS-modulated; which bands can be discarded
- Frames which bands models should reconstruct well (links to Q1 band-limited metrics)
- Source: sec1 / sec2a

*Appendix: fig_002/003 PSD curves, fig_beh behavioral traces, fig_trials/split counts*

---

### 3.0.2  Model diagnostics — sec2a

**fig_044*_mrmr_selection_*.png — mRMR channel selection heatmap**
- Top-6 ECoG + top-6 LFP = 12 channels per session; session-stable vs session-specific picks
- Cross-check against fig_005: mRMR should agree with significance map
- Source: sec2a

*Appendix: fig_045 corrmat, fig_046 relevance bars, fig_039 DPAD training curves*

---

## Q1 — Can models reconstruct neural/behavioral dynamics from ECoG?

Both experiment types answer the same question with different Z targets.
Band-limited metrics (fig_A/B) are part of the reconstruction evidence here.

### Q1.1  z-as-behavior — sec2b

**Figs 7-8 — Pooled behavioral reconstruction, 24-box layout**
- RMSE / Pearson r: 3 models x 2 DBS conditions x 2 output channels, all sessions pooled
- Source: sec2b

**Fig 17 — Session-mean RMSE strip/box**
- Session on x-axis, mean RMSE per model; highlights outlier sessions
- Counts for Q1.1 and Q1.2

**Figs 46-49 — Per-session LFP reconstruction time series**
- True vs reconstructed laplacian LFP, one trial per session; qualitative envelope check
- Counts for Q1.1 and Q1.2

**[NEW] fig_A — Band-limited Pearson r heatmap (Zp vs Z, prediction)**
- Bandpass Zp + Z into delta/theta/alpha/beta/low-gamma/high-gamma
- r per band; heatmap rows=sessions, cols=bands, one panel per model
- Which frequency components reconstructed well; align with fig_005 significance bands
- Counts for Q1.1 and Q1.2

**[NEW] fig_B — Amplitude envelope correlation (Zp vs Z)**
- Hilbert envelope on Zp and Z; correlate per session per model
- High envelope r + low waveform r = model captures power modulation, not phase
- Counts for Q1.1 and Q1.2

*Appendix: Figs 9-16 per-session strip plots*

---

### Q1.2  z-as-neural — sec2c

**fig_070_lap_per_cell_yz_box.png — Laplacian per cell, Z reconstruction**
- Box plots: Zp vs Z (top-8 LFP target), PSID/DPAD/VARMA; Y self-recon as sanity check
- Source: sec2c

**fig_072_lap_pool_raincloud_yz.png — Laplacian Y vs Z per model (pooled)**
- Raincloud: Y recon vs Z recon quality; separation shows whether Z systematically worse
- Source: sec2c

**fig_073_beh_per_cell_yz_box.png — Behavioral per cell Y + Z (in sec2c)**
- z-as-behavior variant inside sec2c for direct z-type comparison
- Counts for Q1.1 and Q1.2

**fig_A, fig_B** — shared with Q1.1 (separate panels per experiment type)

---

## Q2 — Can models forecast future dynamics?

Forecast = model propagates forward h seconds without observing Y.
Band-limited forecast (fig_C) + windowed decay are part of forecast evidence here.

### Q2.1  z-as-behavior — sec2d

**fig_083_beh_per_cell_yz_box_forecast.png — Behavioral forecast per cell**
- 24-box layout for Zf; gap vs Figs 7-8 = reliance on live Y
- Source: sec2d

**fig_084_beh_pool_raincloud_y_forecast.png — Pooled forecast raincloud, Yf**
- RMSE of Yf by model; VARMA = AR baseline; PSID/DPAD gap = latent-state benefit
- Source: sec2d

**fig_086_pool_horizon_rmse_pearson.png — Forecast quality vs horizon h**
- RMSE + Pearson r vs h [0.5 ... 5 s], one line per model
- Counts for Q2.1 and Q2.2

**[NEW] fig_forecast_decay — Windowed performance decay (~0.1 s windows)**
- RMSE/r in sliding windows across horizon; shape: fast drop vs plateau vs gradual
- Counts for Q2.1 and Q2.2; alongside fig_086

**[NEW] fig_C — Band-limited forecast quality (Zf vs Z)**
- Repeat fig_A + fig_B logic on forecast outputs
- High-frequency bands degrade faster; links to t_cut plateau in Q3
- Counts for Q2.1 and Q2.2

---

### Q2.2  z-as-neural — sec2d

**fig_080_lap_per_cell_yz_box_forecast.png — Laplacian forecast per cell**
- 24-box Zf vs Z, laplacian; counterpart to fig_083

**fig_081_lap_pool_raincloud_y_forecast.png — Laplacian pooled forecast raincloud**
- RMSE of Yf in laplacian experiment; does z-as-neural forecast own modality better?

**fig_086, fig_forecast_decay, fig_C** — shared with Q2.1 (separate panels per experiment type)

---

## Q3 — Does the latent state encode DBS state?

Both experiment types share these figures (separate panels per experiment type).
Source: sec5

**fig_050_classification_heatmap.png — cv_ba heatmap, sessions x feature sources**
- 0.5 anchored midpoint; permutation stars; Xp_1 vs Xp_2 = key diagnostic
- Preferred for paper; fig_049 bar version -> supplementary

**fig_051_flipped_heatmap.png — Flipped-label BA (adversarial null)**
- LDA trained on one DBS condition's Xf, scored on other
- Near chance = model forecasts DBS-agnostic

**fig_052b_gap_heatmap.png — Generalisation gap (within - flipped BA)**
- Positive = DBS-state-specific structure; near-zero = no cross-condition generalisation

**fig_053_tcut_analysis.png — BA vs time window (t_cut)**
- x = t_cut [0.5 ... 9 s], y = ba_at_score; one line per session; dotted cv_ba reference
- Temporal dynamics: how quickly does latent state encode DBS state?
- Separate panels per experiment type; link to fig_C (band degradation at same timescale)

*Note: DPAD classification parquets not yet available — fig_050/051/052b/053 currently PSID + VARMA only. Add DPAD row when parquets arrive.*

---

## Q4 — Does supervised target choice (behavior vs neural) change encoding quality?

Source: sec6

**fig_059_latent_phase.png — Latent phase space KDE**
- 2D KDE of Xp dims by DBS condition; visible separation with low BA = classifier bottleneck

**fig_060_classification_dimensionality.png — Classification vs dimensionality grid**
- cv_ba heatmap over nx x n1; is model dimension-saturated?

**fig_061_cy_importance.png — PSID Cy readout heatmap**
- |Cy|: which latent dims drive each Y channel; high n1-subspace weight = DBS in beh subspace

**fig_062_cz_readout.png — PSID Cz readout matrix**
- Same as fig_061 for Z; similar Cy/Cz loading = shared latent representation

**fig_063_data_efficiency.png — Data efficiency curve**
- BA vs training data fraction (k-fold CV inside training pool only)
- Saturated at 30% = signal quality is limit, not data quantity

---

## Appendix

**sec1:** fig_002/003 PSD curves, fig_beh, fig_trials/split counts — data processing overview

**sec2a:** fig_045 corrmat, fig_046 relevance bars, fig_039 DPAD training curves

**sec2b:** Figs 9-16 per-session behavioral strip plots

**sec2e:** Figs 18-21, 29-36, 50-55 — exemplar single-trial overlays (qualitative)

**sec7:** A-matrix eigenvalue spectrum + Cohen's d per latent dim between DBS conditions

**sec8:** Per-trial RMSE/r heatmaps over session timeline — non-stationarity audit

---

## Decisions log

1. sec5 split into separate panels for z-as-behavior vs z-as-neural.
2. DPAD classification parquets not yet available — plan for DPAD row when ready.
3. Raw Zp/Zf timeseries confirmed in parquets — band-limited analysis unblocked.
4. fig_053 t_cut: separate panels per experiment type.
5. fig_amp_phase_freq merged into fig_A/B (not standalone).
6. Windowed forecast decay alongside fig_086.
7. 3.3 band-limited dissolved into Q1 (fig_A/B) and Q2 (fig_C).
8. Structure reorganised around 4 research questions, not experiment types.
