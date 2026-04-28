# mRMR library cross-check — Mazzanti `mrmr-selection` vs. in-house Pearson

`K=8`, `seed=0`, per-trial align+finite-mask, per-trial Tukey(α=0.1), train-split first 50% of trials. Behavior targets: `tracing_velocity_x`, `tracing_acceleration_magnitude`. Matches the preprocessing path of `pipeline_psid_diagnostic.py`.

Three selections per cell:

- **shipped** — parsed from `reports/reduced_training_plan.md` (the actual picks
  the pipeline was trained on; relevance computed from diagnostic correlation matrices).
- **inline** — in-house algorithm (`|Pearson|` relevance mean across behavior targets,
  `|Pearson|` redundancy, difference scheme `score = rel − red`) recomputed here on
  session-concatenated trial data. `J(shipped, inline)` isolates data-path drift.
- **mazzanti** — `mrmr_regression` with multi-target mean F-statistic relevance,
  Pearson redundancy, quotient scheme `score = rel / mean(red)`.

Tags after each pick: `[S]` = in shipped, `[I]` = in inline, `[M]` = in mazzanti.

## Per-session / per-family comparison

### PDI1_S2

- **ecog** — K=8  J(shipped, inline)=0.60  J(inline, mazzanti)=0.07  J(shipped, mazzanti)=0.07  ρ(inline, mazzanti on intersection)=n/a
  - **shipped**:
      1. `ECOG_1_theta_4_8_raw`  [M]
      2. `ECOG_2_beta_12_17_raw`  [I]
      3. `ECOG_3_beta_27_30_raw`  [I]
      4. `ECOG_2_gamma_45_50_raw`  [I]
      5. `ECOG_2_gamma_75_80_raw`  [I]
      6. `ECOG_1_gamma_75_80_raw`  [I]
      7. `ECOG_2_beta_27_30_raw`  [I]
      8. `ECOG_3_gamma_50_55_raw`
  - **inline**:
      1. `ECOG_3_theta_4_8_raw`  [M]
      2. `ECOG_2_beta_12_17_raw`  [S]
      3. `ECOG_3_beta_27_30_raw`  [S]
      4. `ECOG_2_gamma_45_50_raw`  [S]
      5. `ECOG_2_gamma_75_80_raw`  [S]
      6. `ECOG_1_gamma_75_80_raw`  [S]
      7. `ECOG_2_beta_27_30_raw`  [S]
      8. `ECOG_3_gamma_45_50_raw`
  - **mazzanti**:
      1. `ECOG_2_alpha_8_12_raw`
      2. `ECOG_3_beta_17_22_raw`
      3. `ECOG_3_theta_4_8_raw`  [I]
      4. `ECOG_4_gamma_45_50_raw`
      5. `ECOG_2_theta_4_8_raw`
      6. `ECOG_4_alpha_8_12_raw`
      7. `ECOG_2_beta_17_22_raw`
      8. `ECOG_1_theta_4_8_raw`  [S]

- **laplacian** — K=8  J(shipped, inline)=1.00  J(inline, mazzanti)=0.33  J(shipped, mazzanti)=0.33  ρ(inline, mazzanti on intersection)=+1.00
  - **shipped**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [IM]
      2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [IM]
      3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  [IM]
      4. `LAPLACIAN_14-16_LFP_gamma_70_75_raw`  [I]
      5. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  [I]
      6. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [IM]
      7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  [I]
      8. `LAPLACIAN_14-16_LFP_gamma_35_40_raw`  [I]
  - **inline**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [SM]
      2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [SM]
      3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  [SM]
      4. `LAPLACIAN_14-16_LFP_gamma_70_75_raw`  [S]
      5. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  [S]
      6. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [SM]
      7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  [S]
      8. `LAPLACIAN_14-16_LFP_gamma_35_40_raw`  [S]
  - **mazzanti**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [SI]
      2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [SI]
      3. `LAPLACIAN_14-16_LFP_gamma_50_55_raw`
      4. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  [SI]
      5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`
      6. `LAPLACIAN_14-16_LFP_gamma_40_45_raw`
      7. `LAPLACIAN_14-16_LFP_gamma_55_60_raw`
      8. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [SI]

### PDI1_S4

- **ecog** — K=8  J(shipped, inline)=0.45  J(inline, mazzanti)=0.14  J(shipped, mazzanti)=0.14  ρ(inline, mazzanti on intersection)=+1.00
  - **shipped**:
      1. `ECOG_3_theta_4_8_raw`  [I]
      2. `ECOG_1_beta_12_17_raw`  [M]
      3. `ECOG_3_gamma_45_50_raw`  [I]
      4. `ECOG_3_beta_22_27_raw`  [IM]
      5. `ECOG_1_gamma_75_80_raw`  [I]
      6. `ECOG_1_beta_27_30_raw`
      7. `ECOG_2_gamma_60_65_raw`  [I]
      8. `ECOG_3_beta_12_17_raw`
  - **inline**:
      1. `ECOG_1_alpha_8_12_raw`  [M]
      2. `ECOG_3_gamma_45_50_raw`  [S]
      3. `ECOG_3_beta_22_27_raw`  [SM]
      4. `ECOG_1_gamma_75_80_raw`  [S]
      5. `ECOG_1_beta_17_22_raw`
      6. `ECOG_1_gamma_30_35_raw`
      7. `ECOG_2_gamma_60_65_raw`  [S]
      8. `ECOG_3_theta_4_8_raw`  [S]
  - **mazzanti**:
      1. `ECOG_1_alpha_8_12_raw`  [I]
      2. `ECOG_3_beta_22_27_raw`  [SI]
      3. `ECOG_1_gamma_50_55_raw`
      4. `ECOG_2_beta_12_17_raw`
      5. `ECOG_2_theta_4_8_raw`
      6. `ECOG_1_beta_12_17_raw`  [S]
      7. `ECOG_2_alpha_8_12_raw`
      8. `ECOG_4_beta_12_17_raw`

- **laplacian** — K=8  J(shipped, inline)=0.78  J(inline, mazzanti)=0.60  J(shipped, mazzanti)=0.60  ρ(inline, mazzanti on intersection)=+0.20
  - **shipped**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [IM]
      2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [IM]
      3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  [M]
      4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  [I]
      5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  [IM]
      6. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  [I]
      7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [IM]
      8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  [IM]
  - **inline**:
      1. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [SM]
      2. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  [S]
      3. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  [S]
      4. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  [SM]
      5. `LAPLACIAN_14-16_LFP_gamma_30_35_raw`  [M]
      6. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [SM]
      7. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  [SM]
      8. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [SM]
  - **mazzanti**:
      1. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [SI]
      2. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  [S]
      3. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [SI]
      4. `LAPLACIAN_14-16_LFP_gamma_70_75_raw`
      5. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [SI]
      6. `LAPLACIAN_14-16_LFP_gamma_30_35_raw`  [I]
      7. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  [SI]
      8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  [SI]

### PDI4_S2

- **ecog** — K=8  J(shipped, inline)=0.07  J(inline, mazzanti)=0.14  J(shipped, mazzanti)=0.07  ρ(inline, mazzanti on intersection)=+1.00
  - **shipped**:
      1. `ECOG_2_theta_4_8_raw`  [M]
      2. `ECOG_3_beta_12_17_raw`
      3. `ECOG_3_gamma_35_40_raw`
      4. `ECOG_3_gamma_60_65_raw`
      5. `ECOG_2_beta_22_27_raw`
      6. `ECOG_2_gamma_75_80_raw`  [I]
      7. `ECOG_2_gamma_45_50_raw`
      8. `ECOG_3_beta_22_27_raw`
  - **inline**:
      1. `ECOG_3_theta_4_8_raw`  [M]
      2. `ECOG_2_beta_12_17_raw`
      3. `ECOG_3_beta_27_30_raw`
      4. `ECOG_3_gamma_45_50_raw`
      5. `ECOG_2_gamma_75_80_raw`  [S]
      6. `ECOG_1_gamma_75_80_raw`
      7. `ECOG_2_gamma_30_35_raw`  [M]
      8. `ECOG_2_gamma_50_55_raw`
  - **mazzanti**:
      1. `ECOG_3_theta_4_8_raw`  [I]
      2. `ECOG_1_beta_12_17_raw`
      3. `ECOG_2_gamma_30_35_raw`  [I]
      4. `ECOG_1_theta_4_8_raw`
      5. `ECOG_2_theta_4_8_raw`  [S]
      6. `ECOG_4_alpha_8_12_raw`
      7. `ECOG_2_alpha_8_12_raw`
      8. `ECOG_1_alpha_8_12_raw`

- **laplacian** — K=8  J(shipped, inline)=0.78  J(inline, mazzanti)=0.45  J(shipped, mazzanti)=0.33  ρ(inline, mazzanti on intersection)=+0.50
  - **shipped**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [IM]
      2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [I]
      3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  [I]
      4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  [I]
      5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  [IM]
      6. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  [IM]
      7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [IM]
      8. `LAPLACIAN_14-16_LFP_gamma_55_60_raw`
  - **inline**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [SM]
      2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [S]
      3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  [S]
      4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  [S]
      5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  [SM]
      6. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  [SM]
      7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [SM]
      8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  [M]
  - **mazzanti**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [SI]
      2. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  [SI]
      3. `LAPLACIAN_14-16_LFP_gamma_35_40_raw`
      4. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  [I]
      5. `LAPLACIAN_14-16_LFP_gamma_30_35_raw`
      6. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  [SI]
      7. `LAPLACIAN_14-16_LFP_gamma_50_55_raw`
      8. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [SI]

### PDI4_S3

- **ecog** — K=8  J(shipped, inline)=0.23  J(inline, mazzanti)=0.14  J(shipped, mazzanti)=0.14  ρ(inline, mazzanti on intersection)=+1.00
  - **shipped**:
      1. `ECOG_3_theta_4_8_raw`  [M]
      2. `ECOG_2_beta_12_17_raw`  [M]
      3. `ECOG_3_gamma_35_40_raw`  [I]
      4. `ECOG_3_gamma_60_65_raw`  [I]
      5. `ECOG_2_beta_27_30_raw`
      6. `ECOG_2_gamma_75_80_raw`  [I]
      7. `ECOG_3_beta_17_22_raw`
      8. `ECOG_2_gamma_45_50_raw`
  - **inline**:
      1. `ECOG_4_theta_4_8_raw`  [M]
      2. `ECOG_3_beta_12_17_raw`
      3. `ECOG_3_gamma_35_40_raw`  [S]
      4. `ECOG_3_gamma_60_65_raw`  [S]
      5. `ECOG_2_beta_22_27_raw`
      6. `ECOG_2_gamma_75_80_raw`  [S]
      7. `ECOG_2_gamma_40_45_raw`  [M]
      8. `ECOG_3_beta_22_27_raw`
  - **mazzanti**:
      1. `ECOG_4_theta_4_8_raw`  [I]
      2. `ECOG_2_beta_12_17_raw`  [S]
      3. `ECOG_2_theta_4_8_raw`
      4. `ECOG_1_theta_4_8_raw`
      5. `ECOG_3_alpha_8_12_raw`
      6. `ECOG_1_beta_22_27_raw`
      7. `ECOG_3_theta_4_8_raw`  [S]
      8. `ECOG_2_gamma_40_45_raw`  [I]

- **laplacian** — K=8  J(shipped, inline)=1.00  J(inline, mazzanti)=0.45  J(shipped, mazzanti)=0.45  ρ(inline, mazzanti on intersection)=+0.90
  - **shipped**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [IM]
      2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [IM]
      3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  [I]
      4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  [IM]
      5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  [I]
      6. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  [IM]
      7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [IM]
      8. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  [I]
  - **inline**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [SM]
      2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [SM]
      3. `LAPLACIAN_14-16_LFP_beta_27_30_raw`  [S]
      4. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  [SM]
      5. `LAPLACIAN_14-16_LFP_gamma_75_80_raw`  [S]
      6. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  [SM]
      7. `LAPLACIAN_14-16_LFP_beta_17_22_raw`  [S]
      8. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [SM]
  - **mazzanti**:
      1. `LAPLACIAN_14-16_LFP_theta_4_8_raw`  [SI]
      2. `LAPLACIAN_14-16_LFP_beta_12_17_raw`  [SI]
      3. `LAPLACIAN_14-16_LFP_gamma_40_45_raw`
      4. `LAPLACIAN_14-16_LFP_gamma_70_75_raw`
      5. `LAPLACIAN_14-16_LFP_beta_22_27_raw`
      6. `LAPLACIAN_14-16_LFP_gamma_45_50_raw`  [SI]
      7. `LAPLACIAN_14-16_LFP_alpha_8_12_raw`  [SI]
      8. `LAPLACIAN_14-16_LFP_gamma_60_65_raw`  [SI]

## Summary

| session | family | J(ship,inline) | J(inline,maz) | J(ship,maz) | ρ(inline,maz) | n_samples |
| --- | --- | --- | --- | --- | --- | --- |
| PDI1_S2 | ecog | 0.60 | 0.07 | 0.07 | n/a | 129478 |
| PDI1_S2 | laplacian | 1.00 | 0.33 | 0.33 | +1.00 | 129478 |
| PDI1_S4 | ecog | 0.45 | 0.14 | 0.14 | +1.00 | 95314 |
| PDI1_S4 | laplacian | 0.78 | 0.60 | 0.60 | +0.20 | 95314 |
| PDI4_S2 | ecog | 0.07 | 0.14 | 0.07 | +1.00 | 106008 |
| PDI4_S2 | laplacian | 0.78 | 0.45 | 0.33 | +0.50 | 106008 |
| PDI4_S3 | ecog | 0.23 | 0.14 | 0.14 | +1.00 | 104259 |
| PDI4_S3 | laplacian | 1.00 | 0.45 | 0.45 | +0.90 | 104259 |
