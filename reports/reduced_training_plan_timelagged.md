# mRMR time-lagged relevance — per-session plan

Relevance: per trial, |corr(feature, behavior at t+lag)| at 21 lags across ±250 ms; mean of the top-3 |r| values per trial; sample-weighted average across trials.
Relevance is then averaged across the two behavior dims.

Redundancy: per-trial feature-feature Pearson (sample-weighted across trials, then absolute value). **Instantaneous** — does not account for lagged co-linearity between features. A pair that looks non-redundant here may still be lagged-duplicate; documented limitation of this script.

Stability: **bootstrap × 5** at 80% trial-level subsample. `boot=k/N` = number of folds the feature landed in top-K. **core** = features picked in every fold.

Compare against the standard-Pearson plan in `reduced_training_plan.md`.

## Per-session / per-family selection

### PDI1_S2

- **ecog** (top-3 lag mean, K=8)  — 144 trials, 258958 samples
   1. `ECOG_2_theta_4_8_raw`  (rel=0.0650, score=0.0650, boot=5/5)
   2. `ECOG_2_beta_12_17_raw`  (rel=0.0161, score=0.0140, boot=3/5)
   3. `ECOG_3_beta_27_30_raw`  (rel=0.0052, score=0.0041, boot=5/5)
   4. `ECOG_3_gamma_45_50_raw`  (rel=0.0033, score=0.0019, boot=4/5)
   5. `ECOG_2_gamma_75_80_raw`  (rel=0.0024, score=-0.0017, boot=2/5)
   6. `ECOG_1_theta_4_8_raw`  (rel=0.0621, score=-0.0106, boot=5/5)
   7. `ECOG_1_gamma_75_80_raw`  (rel=0.0023, score=-0.0239, boot=3/5)
   8. `ECOG_2_gamma_35_40_raw`  (rel=0.0038, score=-0.0271, boot=2/5)
  - **core** (selected in all 5 bootstrap folds): `ECOG_2_theta_4_8_raw`, `ECOG_3_beta_27_30_raw`, `ECOG_1_theta_4_8_raw`

- **laplacian** (top-3 lag mean, K=8)  — 144 trials, 258958 samples
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0760, score=0.0760, boot=5/5)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0213, score=0.0072, boot=5/5)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0095, score=-0.0150, boot=5/5)
   4. `LAPLACIAN_14-16_LFP_gamma_70_75_raw`  (rel=0.0091, score=-0.0265, boot=5/5)
   5. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0116, score=-0.0255, boot=5/5)
   6. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0398, score=-0.0759, boot=5/5)
   7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0112, score=-0.0912, boot=5/5)
   8. `LAPLACIAN_14-16_LFP_gamma_35_40_raw`  (rel=0.0107, score=-0.1167, boot=5/5)
  - **core** (selected in all 5 bootstrap folds): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_70_75_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_gamma_35_40_raw`

### PDI1_S4

- **ecog** (top-3 lag mean, K=8)  — 106 trials, 190621 samples
   1. `ECOG_2_theta_4_8_raw`  (rel=0.0609, score=0.0609, boot=4/5)
   2. `ECOG_1_beta_12_17_raw`  (rel=0.0128, score=0.0107, boot=5/5)
   3. `ECOG_3_beta_27_30_raw`  (rel=0.0036, score=0.0025, boot=5/5)
   4. `ECOG_3_gamma_45_50_raw`  (rel=0.0023, score=0.0010, boot=4/5)
   5. `ECOG_4_gamma_75_80_raw`  (rel=0.0014, score=-0.0003, boot=4/5)
   6. `ECOG_3_alpha_8_12_raw`  (rel=0.0273, score=-0.0501, boot=4/5)
   7. `ECOG_1_gamma_30_35_raw`  (rel=0.0033, score=-0.0514, boot=4/5)
   8. `ECOG_3_beta_17_22_raw`  (rel=0.0063, score=-0.0468, boot=4/5)
  - **core** (selected in all 5 bootstrap folds): `ECOG_1_beta_12_17_raw`, `ECOG_3_beta_27_30_raw`

- **laplacian** (top-3 lag mean, K=8)  — 106 trials, 190621 samples
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0713, score=0.0713, boot=5/5)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0156, score=0.0054, boot=5/5)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0078, score=-0.0057, boot=5/5)
   4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0084, score=-0.0145, boot=5/5)
   5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.0079, score=-0.0171, boot=5/5)
   6. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0316, score=-0.0981, boot=5/5)
   7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0086, score=-0.0986, boot=5/5)
   8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  (rel=0.0089, score=-0.1062, boot=5/5)
  - **core** (selected in all 5 bootstrap folds): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`

### PDI4_S2

- **ecog** (top-3 lag mean, K=8)  — 119 trials, 213814 samples
   1. `ECOG_3_theta_4_8_raw`  (rel=0.0365, score=0.0365, boot=5/5)
   2. `ECOG_1_beta_12_17_raw`  (rel=0.0062, score=0.0039, boot=3/5)
   3. `ECOG_2_beta_27_30_raw`  (rel=0.0033, score=0.0018, boot=4/5)
   4. `ECOG_2_gamma_45_50_raw`  (rel=0.0018, score=-0.0000, boot=5/5)
   5. `ECOG_3_gamma_75_80_raw`  (rel=0.0013, score=-0.0029, boot=4/5)
   6. `ECOG_3_gamma_30_35_raw`  (rel=0.0032, score=-0.0280, boot=4/5)
   7. `ECOG_2_gamma_75_80_raw`  (rel=0.0014, score=-0.0317, boot=5/5)
   8. `ECOG_1_gamma_50_55_raw`  (rel=0.0017, score=-0.0343, boot=2/5)
  - **core** (selected in all 5 bootstrap folds): `ECOG_3_theta_4_8_raw`, `ECOG_2_gamma_45_50_raw`, `ECOG_2_gamma_75_80_raw`

- **laplacian** (top-3 lag mean, K=8)  — 119 trials, 213814 samples
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0531, score=0.0531, boot=5/5)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0056, score=-0.0009, boot=5/5)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0042, score=-0.0036, boot=5/5)
   4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0039, score=-0.0082, boot=5/5)
   5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.0040, score=-0.0117, boot=5/5)
   6. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0038, score=-0.1243, boot=5/5)
   7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0103, score=-0.1035, boot=5/5)
   8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  (rel=0.0041, score=-0.1065, boot=5/5)
  - **core** (selected in all 5 bootstrap folds): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`

### PDI4_S3

- **ecog** (top-3 lag mean, K=8)  — 117 trials, 210309 samples
   1. `ECOG_2_theta_4_8_raw`  (rel=0.0468, score=0.0468, boot=5/5)
   2. `ECOG_3_beta_12_17_raw`  (rel=0.0071, score=0.0050, boot=3/5)
   3. `ECOG_3_gamma_35_40_raw`  (rel=0.0029, score=0.0014, boot=5/5)
   4. `ECOG_3_gamma_60_65_raw`  (rel=0.0017, score=-0.0019, boot=5/5)
   5. `ECOG_2_beta_22_27_raw`  (rel=0.0036, score=-0.0053, boot=5/5)
   6. `ECOG_2_gamma_75_80_raw`  (rel=0.0017, score=-0.0243, boot=5/5)
   7. `ECOG_2_gamma_40_45_raw`  (rel=0.0028, score=-0.0342, boot=5/5)
   8. `ECOG_3_theta_4_8_raw`  (rel=0.0412, score=-0.0382, boot=3/5)
  - **core** (selected in all 5 bootstrap folds): `ECOG_2_theta_4_8_raw`, `ECOG_3_gamma_35_40_raw`, `ECOG_3_gamma_60_65_raw`, `ECOG_2_beta_22_27_raw`, `ECOG_2_gamma_75_80_raw`, `ECOG_2_gamma_40_45_raw`

- **laplacian** (top-3 lag mean, K=8)  — 117 trials, 210309 samples
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0554, score=0.0554, boot=5/5)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0071, score=-0.0014, boot=5/5)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0033, score=-0.0014, boot=5/5)
   4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0030, score=-0.0042, boot=5/5)
   5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.0019, score=-0.0041, boot=5/5)
   6. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0137, score=-0.1127, boot=5/5)
   7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0047, score=-0.0974, boot=5/5)
   8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  (rel=0.0026, score=-0.0882, boot=5/5)
  - **core** (selected in all 5 bootstrap folds): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`
