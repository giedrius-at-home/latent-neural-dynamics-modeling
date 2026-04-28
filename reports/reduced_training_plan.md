# Reduced PSID Training Plan

_Generated from `configs/diagnostic/elbow_choices.yaml` on 2026-04-23 12:52_

## Per-session / per-family feature selection

### PDI1_S2

- **ecog** (n1=15, nx=165, K=8)
   1. `ECOG_3_theta_4_8_raw`  (rel=0.007, score=0.007)
   2. `ECOG_3_beta_12_17_raw`  (rel=0.001, score=0.001)
   3. `ECOG_2_beta_27_30_raw`  (rel=0.000, score=-0.000)
   4. `ECOG_2_gamma_45_50_raw`  (rel=0.000, score=-0.001)
   5. `ECOG_2_gamma_75_80_raw`  (rel=0.000, score=-0.004)
   6. `ECOG_1_gamma_75_80_raw`  (rel=0.000, score=-0.031)
   7. `ECOG_3_beta_27_30_raw`  (rel=0.000, score=-0.036)
   8. `ECOG_3_gamma_50_55_raw`  (rel=0.000, score=-0.043)

- **laplacian** (n1=10, nx=80, K=8)
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.015, score=0.015)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.001, score=-0.013)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.001, score=-0.024)
   4. `LAPLACIAN_14-16_LFP_gamma_70_75_raw`  (rel=0.002, score=-0.035)
   5. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.002, score=-0.039)
   6. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.001, score=-0.115)
   7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.000, score=-0.097)
   8. `LAPLACIAN_14-16_LFP_gamma_35_40_raw`  (rel=0.001, score=-0.125)

### PDI1_S4

- **ecog** (n1=10, nx=80, K=8)
   1. `ECOG_1_alpha_8_12_raw`  (rel=0.002, score=0.002)
   2. `ECOG_3_gamma_45_50_raw`  (rel=0.000, score=0.000)
   3. `ECOG_3_beta_22_27_raw`  (rel=0.001, score=0.000)
   4. `ECOG_1_gamma_75_80_raw`  (rel=0.000, score=-0.001)
   5. `ECOG_1_beta_17_22_raw`  (rel=0.000, score=-0.027)
   6. `ECOG_1_gamma_30_35_raw`  (rel=0.000, score=-0.048)
   7. `ECOG_2_gamma_60_65_raw`  (rel=0.000, score=-0.046)
   8. `ECOG_3_theta_4_8_raw`  (rel=0.001, score=-0.052)

- **laplacian** (n1=10, nx=70, K=8)
   1. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.002, score=0.002)
   2. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.000, score=-0.000)
   3. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.000, score=-0.005)
   4. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.000, score=-0.010)
   5. `LAPLACIAN_14-16_LFP_gamma_30_35_raw`  (rel=0.000, score=-0.038)
   6. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.001, score=-0.079)
   7. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  (rel=0.000, score=-0.126)
   8. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.001, score=-0.126)

### PDI4_S2

- **ecog** (n1=10, nx=160, K=8)
   1. `ECOG_1_theta_4_8_raw`  (rel=0.002, score=0.002)
   2. `ECOG_3_beta_12_17_raw`  (rel=0.000, score=-0.000)
   3. `ECOG_2_gamma_30_35_raw`  (rel=0.000, score=-0.001)
   4. `ECOG_3_gamma_50_55_raw`  (rel=0.000, score=-0.003)
   5. `ECOG_2_gamma_75_80_raw`  (rel=0.000, score=-0.007)
   6. `ECOG_1_gamma_75_80_raw`  (rel=0.000, score=-0.023)
   7. `ECOG_3_beta_22_27_raw`  (rel=0.000, score=-0.021)
   8. `ECOG_2_beta_17_22_raw`  (rel=0.000, score=-0.043)

- **laplacian** (n1=10, nx=65, K=8)
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.001, score=0.001)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.000, score=-0.005)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.000, score=-0.006)
   4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.000, score=-0.010)
   5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.000, score=-0.013)
   6. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.000, score=-0.133)
   7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.000, score=-0.113)
   8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  (rel=0.000, score=-0.107)

### PDI4_S3

- **ecog** (n1=10, nx=160, K=8)
   1. `ECOG_4_theta_4_8_raw`  (rel=0.008, score=0.008)
   2. `ECOG_3_beta_12_17_raw`  (rel=0.000, score=-0.001)
   3. `ECOG_3_gamma_35_40_raw`  (rel=0.000, score=-0.001)
   4. `ECOG_3_gamma_60_65_raw`  (rel=0.000, score=-0.004)
   5. `ECOG_2_beta_22_27_raw`  (rel=0.000, score=-0.009)
   6. `ECOG_2_gamma_75_80_raw`  (rel=0.000, score=-0.024)
   7. `ECOG_2_gamma_40_45_raw`  (rel=0.000, score=-0.036)
   8. `ECOG_3_beta_22_27_raw`  (rel=0.000, score=-0.044)

- **laplacian** (n1=10, nx=65, K=8)
   1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  (rel=0.006, score=0.006)
   2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  (rel=0.000, score=-0.006)
   3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  (rel=0.000, score=-0.006)
   4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  (rel=0.000, score=-0.010)
   5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  (rel=0.000, score=-0.008)
   6. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  (rel=0.000, score=-0.109)
   7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  (rel=0.000, score=-0.102)
   8. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  (rel=0.000, score=-0.094)

## Generated training configs

- `training/setups/psid/narrow_band_200Hz/both/psid_ecog_PDI1_S2_nx_165_n15_i35_mrmr8_dbs_both_200Hz_narrow_band.yaml`
- `training/setups/psid/narrow_band_200Hz/both/psid_laplacian_PDI1_S2_nx_80_n10_i35_mrmr8_dbs_both_200Hz_narrow_band.yaml`
- `training/setups/psid/narrow_band_200Hz/both/psid_ecog_PDI1_S4_nx_80_n10_i35_mrmr8_dbs_both_200Hz_narrow_band.yaml`
- `training/setups/psid/narrow_band_200Hz/both/psid_laplacian_PDI1_S4_nx_70_n10_i35_mrmr8_dbs_both_200Hz_narrow_band.yaml`
- `training/setups/psid/narrow_band_200Hz/both/psid_ecog_PDI4_S2_nx_160_n10_i35_mrmr8_dbs_both_200Hz_narrow_band.yaml`
- `training/setups/psid/narrow_band_200Hz/both/psid_laplacian_PDI4_S2_nx_65_n10_i35_mrmr8_dbs_both_200Hz_narrow_band.yaml`
- `training/setups/psid/narrow_band_200Hz/both/psid_ecog_PDI4_S3_nx_160_n10_i35_mrmr8_dbs_both_200Hz_narrow_band.yaml`
- `training/setups/psid/narrow_band_200Hz/both/psid_laplacian_PDI4_S3_nx_65_n10_i35_mrmr8_dbs_both_200Hz_narrow_band.yaml`
