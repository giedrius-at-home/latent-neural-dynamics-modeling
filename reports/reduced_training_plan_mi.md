# mRMR mutual-information relevance — per-session plan

Relevance: `mean_j MI(feature_i, behavior_j)` using sklearn's k-NN MI estimator (n_neighbors=3).
Redundancy: instantaneous |Pearson r| between features.

Compare against `reduced_training_plan.md` (Pearson) and `reduced_training_plan_timelagged.md` (lagged Pearson).

## Per-session / per-family selection

### PDI1_S2

- **ecog** (MI relevance, K=8)
   1. `ECOG_4_gamma_35_40_raw`  (rel=0.0094, score=0.0094)
   2. `ECOG_1_beta_12_17_raw`  (rel=0.0068, score=0.0059)
   3. `ECOG_2_theta_4_8_raw`  (rel=0.0047, score=0.0032)
   4. `ECOG_2_gamma_75_80_raw`  (rel=0.0049, score=-0.0002)
   5. `ECOG_3_gamma_50_55_raw`  (rel=0.0006, score=-0.0151)
   6. `ECOG_3_beta_22_27_raw`  (rel=0.0028, score=-0.0185)
   7. `ECOG_1_gamma_75_80_raw`  (rel=0.0010, score=-0.0275)
   8. `ECOG_2_beta_22_27_raw`  (rel=0.0028, score=-0.0443)

- **laplacian** (MI relevance, K=8)
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0084, score=0.0084)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0025, score=-0.0007)
   3. `LAPLACIAN_14-16_LFP_beta_22_27_raw`  (rel=0.0025, score=-0.0221)
   4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0026, score=-0.0395)
   5. `LAPLACIAN_14-16_LFP_gamma_70_75_raw`  (rel=0.0016, score=-0.0503)
   6. `LAPLACIAN_14-16_LFP_gamma_30_35_raw`  (rel=0.0069, score=-0.1125)
   7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0019, score=-0.1293)
   8. `LAPLACIAN_14-16_LFP_gamma_55_60_raw`  (rel=0.0018, score=-0.1480)

### PDI1_S4

- **ecog** (MI relevance, K=8)
   1. `ECOG_1_gamma_30_35_raw`  (rel=0.0116, score=0.0116)
   2. `ECOG_2_beta_17_22_raw`  (rel=0.0109, score=0.0088)
   3. `ECOG_3_theta_4_8_raw`  (rel=0.0078, score=0.0034)
   4. `ECOG_4_gamma_75_80_raw`  (rel=0.0109, score=0.0046)
   5. `ECOG_3_gamma_55_60_raw`  (rel=0.0069, score=-0.0083)
   6. `ECOG_1_alpha_8_12_raw`  (rel=0.0046, score=-0.0702)
   7. `ECOG_3_gamma_40_45_raw`  (rel=0.0087, score=-0.0681)
   8. `ECOG_3_beta_27_30_raw`  (rel=0.0008, score=-0.0645)

- **laplacian** (MI relevance, K=8)
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0137, score=0.0137)
   2. `LAPLACIAN_14-16_LFP_gamma_35_40_raw`  (rel=0.0042, score=0.0025)
   3. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0040, score=-0.0171)
   4. `LAPLACIAN_14-16_LFP_gamma_55_60_raw`  (rel=0.0032, score=-0.0574)
   5. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0048, score=-0.0891)
   6. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.0025, score=-0.1009)
   7. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0023, score=-0.1238)
   8. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0044, score=-0.1282)

### PDI4_S2

- **ecog** (MI relevance, K=8)
   1. `ECOG_3_beta_22_27_raw`  (rel=0.0116, score=0.0116)
   2. `ECOG_1_alpha_8_12_raw`  (rel=0.0091, score=0.0040)
   3. `ECOG_2_gamma_60_65_raw`  (rel=0.0032, score=-0.0003)
   4. `ECOG_2_gamma_35_40_raw`  (rel=0.0000, score=-0.0068)
   5. `ECOG_1_gamma_70_75_raw`  (rel=0.0000, score=-0.0137)
   6. `ECOG_2_beta_17_22_raw`  (rel=0.0008, score=-0.0240)
   7. `ECOG_1_gamma_45_50_raw`  (rel=0.0053, score=-0.0304)
   8. `ECOG_2_theta_4_8_raw`  (rel=0.0088, score=-0.0335)

- **laplacian** (MI relevance, K=8)
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0191, score=0.0191)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0059, score=0.0011)
   3. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.0034, score=-0.0042)
   4. `LAPLACIAN_14-16_LFP_beta_22_27_raw`  (rel=0.0002, score=-0.0030)
   5. `LAPLACIAN_14-16_LFP_gamma_70_75_raw`  (rel=0.0063, score=-0.0235)
   6. `LAPLACIAN_14-16_LFP_gamma_30_35_raw`  (rel=0.0050, score=-0.1237)
   7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0102, score=-0.1183)
   8. `LAPLACIAN_14-16_LFP_gamma_50_55_raw`  (rel=0.0041, score=-0.1419)

### PDI4_S3

- **ecog** (MI relevance, K=8)
   1. `ECOG_2_theta_4_8_raw`  (rel=0.0099, score=0.0099)
   2. `ECOG_3_gamma_30_35_raw`  (rel=0.0054, score=0.0033)
   3. `ECOG_2_beta_12_17_raw`  (rel=0.0048, score=0.0016)
   4. `ECOG_2_gamma_55_60_raw`  (rel=0.0033, score=0.0016)
   5. `ECOG_3_gamma_75_80_raw`  (rel=0.0059, score=-0.0153)
   6. `ECOG_2_beta_22_27_raw`  (rel=0.0032, score=-0.0265)
   7. `ECOG_1_gamma_45_50_raw`  (rel=0.0067, score=-0.0375)
   8. `ECOG_3_alpha_8_12_raw`  (rel=0.0068, score=-0.0413)

- **laplacian** (MI relevance, K=8)
   1. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.0083, score=0.0083)
   2. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.0063, score=0.0049)
   3. `LAPLACIAN_14-16_LFP_gamma_50_55_raw`  (rel=0.0034, score=-0.0089)
   4. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.0005, score=-0.0263)
   5. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.0000, score=-0.0689)
   6. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.0071, score=-0.0641)
   7. `LAPLACIAN_14-16_LFP_gamma_40_45_raw`  (rel=0.0034, score=-0.0898)
   8. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.0050, score=-0.1232)
