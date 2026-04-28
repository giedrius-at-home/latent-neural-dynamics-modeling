# mRMR time-lagged relevance — per-session plan

**Selection restricted to training split** (chronological first 50% of cumulative epochs) — prevents test-trial leakage into feature choice.

Relevance: per trial, |corr(feature, behavior at t+lag)| at 21 lags across ±250 ms; mean of the top-3 |r| values per trial; sample-weighted average across trials.
Relevance is then averaged across the two behavior dims.

Redundancy: per-trial feature-feature Pearson (sample-weighted across trials, then absolute value). **Instantaneous** — does not account for lagged co-linearity between features. A pair that looks non-redundant here may still be lagged-duplicate; documented limitation of this script.

Stability: **bootstrap × 5** at 80% trial-level subsample. `boot=k/N` = number of folds the feature landed in top-K. **core** = features picked in every fold.

Compare against the standard-Pearson plan in `reduced_training_plan.md`.

## Per-session / per-family selection

### PDI1_S2

- **ecog** (top-3 lag mean, K=8)  — 72 trials, 129478 samples
   1. `ECOG_2_theta_4_8_raw`  (rel=0.0633, score=0.0633, boot=5/5)
   2. `ECOG_1_beta_12_17_raw`  (rel=0.0167, score=0.0147, boot=2/5)
   3. `ECOG_2_beta_27_30_raw`  (rel=0.0058, score=0.0046, boot=3/5)
   4. `ECOG_3_gamma_45_50_raw`  (rel=0.0035, score=0.0022, boot=5/5)
   5. `ECOG_3_gamma_75_80_raw`  (rel=0.0025, score=-0.0008, boot=5/5)
   6. `ECOG_1_theta_4_8_raw`  (rel=0.0610, score=-0.0126, boot=5/5)
   7. `ECOG_4_gamma_60_65_raw`  (rel=0.0031, score=-0.0259, boot=5/5)
   8. `ECOG_3_beta_27_30_raw`  (rel=0.0057, score=-0.0278, boot=4/5)
  - **core** (selected in all 5 bootstrap folds): `ECOG_2_theta_4_8_raw`, `ECOG_3_gamma_45_50_raw`, `ECOG_3_gamma_75_80_raw`, `ECOG_1_theta_4_8_raw`, `ECOG_4_gamma_60_65_raw`

- **laplacian** (top-3 lag mean, K=8)  — 72 trials, 129478 samples
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0744, score=0.0744, boot=5/5)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0228, score=0.0092, boot=5/5)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0090, score=-0.0147, boot=5/5)
   4. `LAPLACIAN_14-16_LFP_gamma_70_75_raw`  (rel=0.0086, score=-0.0255, boot=5/5)
   5. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0113, score=-0.0256, boot=5/5)
   6. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0409, score=-0.0768, boot=5/5)
   7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0115, score=-0.0889, boot=5/5)
   8. `LAPLACIAN_14-16_LFP_gamma_35_40_raw`  (rel=0.0101, score=-0.1161, boot=5/5)
  - **core** (selected in all 5 bootstrap folds): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_70_75_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_gamma_35_40_raw`

### PDI1_S4

- **ecog** (top-3 lag mean, K=8)  — 53 trials, 95314 samples
   1. `ECOG_1_theta_4_8_raw`  (rel=0.0606, score=0.0606, boot=4/5)
   2. `ECOG_4_beta_12_17_raw`  (rel=0.0135, score=0.0111, boot=4/5)
   3. `ECOG_3_beta_27_30_raw`  (rel=0.0039, score=0.0027, boot=4/5)
   4. `ECOG_2_gamma_55_60_raw`  (rel=0.0027, score=0.0012, boot=1/5)
   5. `ECOG_3_gamma_75_80_raw`  (rel=0.0015, score=-0.0271, boot=2/5)
   6. `ECOG_1_gamma_35_40_raw`  (rel=0.0026, score=-0.0279, boot=3/5)
   7. `ECOG_3_theta_4_8_raw`  (rel=0.0578, score=-0.0144, boot=3/5)
   8. `ECOG_1_beta_22_27_raw`  (rel=0.0048, score=-0.0495, boot=4/5)
  - **core** (selected in all 5 bootstrap folds): _none_

- **laplacian** (top-3 lag mean, K=8)  — 53 trials, 95314 samples
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0703, score=0.0703, boot=5/5)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0163, score=0.0042, boot=5/5)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0088, score=-0.0062, boot=5/5)
   4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0090, score=-0.0162, boot=5/5)
   5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.0074, score=-0.0161, boot=5/5)
   6. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0326, score=-0.0935, boot=5/5)
   7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0097, score=-0.0980, boot=5/5)
   8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  (rel=0.0089, score=-0.1045, boot=5/5)
  - **core** (selected in all 5 bootstrap folds): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`

### PDI4_S2

- **ecog** (top-3 lag mean, K=8)  — 59 trials, 106008 samples
   1. `ECOG_3_theta_4_8_raw`  (rel=0.0378, score=0.0378, boot=5/5)
   2. `ECOG_3_beta_12_17_raw`  (rel=0.0061, score=0.0037, boot=4/5)
   3. `ECOG_2_beta_27_30_raw`  (rel=0.0034, score=0.0022, boot=1/5)
   4. `ECOG_2_gamma_45_50_raw`  (rel=0.0020, score=0.0008, boot=4/5)
   5. `ECOG_2_gamma_75_80_raw`  (rel=0.0015, score=-0.0025, boot=1/5)
   6. `ECOG_3_gamma_30_35_raw`  (rel=0.0032, score=-0.0282, boot=1/5)
   7. `ECOG_1_gamma_60_65_raw`  (rel=0.0017, score=-0.0241, boot=1/5)
   8. `ECOG_2_theta_4_8_raw`  (rel=0.0377, score=-0.0305, boot=5/5)
  - **core** (selected in all 5 bootstrap folds): `ECOG_3_theta_4_8_raw`, `ECOG_2_theta_4_8_raw`

- **laplacian** (top-3 lag mean, K=8)  — 59 trials, 106008 samples
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0541, score=0.0541, boot=5/5)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0058, score=-0.0012, boot=5/5)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0044, score=-0.0040, boot=5/5)
   4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0043, score=-0.0089, boot=5/5)
   5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.0042, score=-0.0118, boot=5/5)
   6. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0041, score=-0.1230, boot=5/5)
   7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0102, score=-0.1021, boot=5/5)
   8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  (rel=0.0044, score=-0.1054, boot=5/5)
  - **core** (selected in all 5 bootstrap folds): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`

### PDI4_S3

- **ecog** (top-3 lag mean, K=8)  — 58 trials, 104259 samples
   1. `ECOG_2_theta_4_8_raw`  (rel=0.0451, score=0.0451, boot=5/5)
   2. `ECOG_3_beta_12_17_raw`  (rel=0.0076, score=0.0053, boot=5/5)
   3. `ECOG_3_gamma_35_40_raw`  (rel=0.0027, score=0.0012, boot=4/5)
   4. `ECOG_3_gamma_60_65_raw`  (rel=0.0016, score=-0.0018, boot=4/5)
   5. `ECOG_2_beta_22_27_raw`  (rel=0.0034, score=-0.0062, boot=5/5)
   6. `ECOG_2_gamma_75_80_raw`  (rel=0.0017, score=-0.0233, boot=4/5)
   7. `ECOG_2_gamma_40_45_raw`  (rel=0.0026, score=-0.0329, boot=5/5)
   8. `ECOG_3_theta_4_8_raw`  (rel=0.0421, score=-0.0383, boot=4/5)
  - **core** (selected in all 5 bootstrap folds): `ECOG_2_theta_4_8_raw`, `ECOG_3_beta_12_17_raw`, `ECOG_2_beta_22_27_raw`, `ECOG_2_gamma_40_45_raw`

- **laplacian** (top-3 lag mean, K=8)  — 58 trials, 104259 samples
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0544, score=0.0544, boot=5/5)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0075, score=-0.0011, boot=5/5)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0032, score=-0.0022, boot=5/5)
   4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0032, score=-0.0063, boot=5/5)
   5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.0021, score=-0.0058, boot=5/5)
   6. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  (rel=0.0026, score=-0.1048, boot=5/5)
   7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0140, score=-0.0902, boot=5/5)
   8. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0046, score=-0.0839, boot=5/5)
  - **core** (selected in all 5 bootstrap folds): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`
