# Per-cell mRMR top-8 channels

Extracted from each canonical dbs_both training YAML.


## PDI1_S2 / behavioral  (nx=55, n1=15)

Neural inputs (top-8 mRMR):

  1. `ECOG_2_gamma_35_40_raw`
  2. `ECOG_3_gamma_75_80_raw`
  3. `ECOG_4_theta_4_8_raw`
  4. `ECOG_2_beta_12_17_raw`
  5. `ECOG_3_beta_22_27_raw`
  6. `ECOG_3_gamma_50_55_raw`
  7. `ECOG_4_gamma_75_80_raw`
  8. `ECOG_3_beta_12_17_raw`

Outputs (Z targets): `tracing_velocity_x`, `tracing_acceleration_magnitude`


## PDI1_S4 / behavioral  (nx=50, n1=10)

Neural inputs (top-8 mRMR):

  1. `ECOG_3_beta_22_27_raw`
  2. `ECOG_1_gamma_45_50_raw`
  3. `ECOG_3_gamma_70_75_raw`
  4. `ECOG_1_alpha_8_12_raw`
  5. `ECOG_1_beta_17_22_raw`
  6. `ECOG_1_gamma_30_35_raw`
  7. `ECOG_3_theta_4_8_raw`
  8. `ECOG_3_beta_12_17_raw`

Outputs (Z targets): `tracing_velocity_x`, `tracing_acceleration_magnitude`


## PDI4_S2 / behavioral  (nx=50, n1=10)

Neural inputs (top-8 mRMR):

  1. `ECOG_3_alpha_8_12_raw`
  2. `ECOG_1_beta_17_22_raw`
  3. `ECOG_2_gamma_55_60_raw`
  4. `ECOG_2_gamma_30_35_raw`
  5. `ECOG_1_gamma_75_80_raw`
  6. `ECOG_3_gamma_40_45_raw`
  7. `ECOG_2_theta_4_8_raw`
  8. `ECOG_2_gamma_75_80_raw`

Outputs (Z targets): `tracing_velocity_x`, `tracing_acceleration_magnitude`


## PDI4_S3 / behavioral  (nx=50, n1=10)

Neural inputs (top-8 mRMR):

  1. `ECOG_4_gamma_50_55_raw`
  2. `ECOG_3_beta_17_22_raw`
  3. `ECOG_1_alpha_8_12_raw`
  4. `ECOG_3_gamma_75_80_raw`
  5. `ECOG_1_gamma_30_35_raw`
  6. `ECOG_2_theta_4_8_raw`
  7. `ECOG_2_beta_22_27_raw`
  8. `ECOG_2_gamma_35_40_raw`

Outputs (Z targets): `tracing_velocity_x`, `tracing_acceleration_magnitude`


## PDI1_S2 / laplacian  (nx=55, n1=15)

Neural inputs (top-8 mRMR):

  1. `ECOG_2_gamma_60_65_raw`
  2. `ECOG_1_beta_17_22_raw`
  3. `ECOG_3_gamma_35_40_raw`
  4. `ECOG_2_alpha_8_12_raw`
  5. `ECOG_1_gamma_75_80_raw`
  6. `ECOG_2_beta_27_30_raw`
  7. `ECOG_3_theta_4_8_raw`
  8. `ECOG_4_gamma_40_45_raw`

Outputs (Z targets): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_beta_22_27_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_30_35_raw`, `LAPLACIAN_14-16_LFP_gamma_35_40_raw`, `LAPLACIAN_14-16_LFP_gamma_40_45_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_50_55_raw`, `LAPLACIAN_14-16_LFP_gamma_55_60_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`, `LAPLACIAN_14-16_LFP_gamma_70_75_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`


## PDI1_S4 / laplacian  (nx=50, n1=10)

Neural inputs (top-8 mRMR):

  1. `ECOG_4_beta_17_22_raw`
  2. `ECOG_4_gamma_40_45_raw`
  3. `ECOG_1_theta_4_8_raw`
  4. `ECOG_2_gamma_70_75_raw`
  5. `ECOG_1_beta_27_30_raw`
  6. `ECOG_3_alpha_8_12_raw`
  7. `ECOG_1_gamma_55_60_raw`
  8. `ECOG_3_beta_27_30_raw`

Outputs (Z targets): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_beta_22_27_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_30_35_raw`, `LAPLACIAN_14-16_LFP_gamma_35_40_raw`, `LAPLACIAN_14-16_LFP_gamma_40_45_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_50_55_raw`, `LAPLACIAN_14-16_LFP_gamma_55_60_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`, `LAPLACIAN_14-16_LFP_gamma_70_75_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`


## PDI4_S2 / laplacian  (nx=50, n1=10)

Neural inputs (top-8 mRMR):

  1. `ECOG_4_gamma_45_50_raw`
  2. `ECOG_2_gamma_75_80_raw`
  3. `ECOG_3_theta_4_8_raw`
  4. `ECOG_3_beta_12_17_raw`
  5. `ECOG_2_beta_27_30_raw`
  6. `ECOG_1_gamma_70_75_raw`
  7. `ECOG_3_beta_22_27_raw`
  8. `ECOG_2_beta_12_17_raw`

Outputs (Z targets): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_beta_22_27_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_30_35_raw`, `LAPLACIAN_14-16_LFP_gamma_35_40_raw`, `LAPLACIAN_14-16_LFP_gamma_40_45_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_50_55_raw`, `LAPLACIAN_14-16_LFP_gamma_55_60_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`, `LAPLACIAN_14-16_LFP_gamma_70_75_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`


## PDI4_S3 / laplacian  (nx=50, n1=10)

Neural inputs (top-8 mRMR):

  1. `ECOG_3_gamma_55_60_raw`
  2. `ECOG_4_beta_27_30_raw`
  3. `ECOG_2_gamma_75_80_raw`
  4. `ECOG_3_alpha_8_12_raw`
  5. `ECOG_2_gamma_40_45_raw`
  6. `ECOG_3_beta_17_22_raw`
  7. `ECOG_2_theta_4_8_raw`
  8. `ECOG_2_gamma_55_60_raw`

Outputs (Z targets): `LAPLACIAN_14-16_LFP_theta_4_8_raw`, `LAPLACIAN_14-16_LFP_alpha_8_12_raw`, `LAPLACIAN_14-16_LFP_beta_12_17_raw`, `LAPLACIAN_14-16_LFP_beta_17_22_raw`, `LAPLACIAN_14-16_LFP_beta_22_27_raw`, `LAPLACIAN_14-16_LFP_beta_27_30_raw`, `LAPLACIAN_14-16_LFP_gamma_30_35_raw`, `LAPLACIAN_14-16_LFP_gamma_35_40_raw`, `LAPLACIAN_14-16_LFP_gamma_40_45_raw`, `LAPLACIAN_14-16_LFP_gamma_45_50_raw`, `LAPLACIAN_14-16_LFP_gamma_50_55_raw`, `LAPLACIAN_14-16_LFP_gamma_55_60_raw`, `LAPLACIAN_14-16_LFP_gamma_60_65_raw`, `LAPLACIAN_14-16_LFP_gamma_70_75_raw`, `LAPLACIAN_14-16_LFP_gamma_75_80_raw`
