# Sani & Shanechi (2025) FB Pipeline — PSID Re-run Plan

**Date:** 2026-04-21
**Goal:** Re-run the PSID pipeline with the forward-backward smoother + filter-aware
forecast (arXiv 2507.15288) replacing the classical RTS smoother. Vanilla variants
are already trained (used for channel selection) and are **skipped**.

## Method changes vs. previous runs

| Aspect | Previous | This re-run |
|---|---|---|
| Smoother on inference | RTS (classical backward sweep on filtered states) | PSID-with-filtering + forward-backward (eq 26, 29, 30) |
| Forecast | Baseline `A^m` state propagation | `A^m x̂ + (C_z A^m K_f)·ỹ` — learned m-step gain (eq 27, stacked) |
| `A` regularisation | Flag present but inactive | **Active**: eigenvalue-clip to ρ ≤ 0.9999 on both forward and backward LSSMs |
| Config flag | `backward_kalman: true` | `backward_kalman: true` **+** `fb_smoother: true` |

Code: `PSIDWrapper._fit_fb_components` / `_smooth_fb_trial` / `_forecast_fb_trial` in
`utils/frameworks.py`. Config generator: `scripts/pipeline_psid.py` adds
`fb_smoother: True` to all non-vanilla configs automatically.

## 24 non-vanilla training configs (8 cell-modes × 3 DBS conditions)

All at `i = 100`, `max_eigenvalue = 0.9999`, `fb_smoother: true`, 50/12.5/37.5
train/val/test split, mRMR-selected 8 neural channels.

### Behavioral (12 runs, decode tracing kinematics from ECoG)

| Cell | nx | n1 | Config files |
|---|---|---|---|
| PDI1_S2 | 55 | 15 | `{both,on,off}/psid_behavioral_PDI1_2_nx_55_n15_i100_dbs_{both,on,off}_200Hz_narrow_band.yaml` |
| PDI1_S4 | 50 | 10 | `{both,on,off}/psid_behavioral_PDI1_4_nx_50_n10_i100_dbs_{both,on,off}_200Hz_narrow_band.yaml` |
| PDI4_S2 | 50 | 10 | `{both,on,off}/psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_{both,on,off}_200Hz_narrow_band.yaml` |
| PDI4_S3 | 50 | 10 | `{both,on,off}/psid_behavioral_PDI4_3_nx_50_n10_i100_dbs_{both,on,off}_200Hz_narrow_band.yaml` |

### Laplacian (12 runs, decode LFP-laplacian 14-16 from ECoG)

| Cell | nx | n1 | Config files |
|---|---|---|---|
| PDI1_S2 | 55 | 15 | `laplacian_200Hz/{both,on,off}/psid_laplacian_PDI1_2_nx_55_n15_i100_dbs_{both,on,off}_200Hz_narrow_band.yaml` |
| PDI1_S4 | 50 | 10 | `laplacian_200Hz/{both,on,off}/psid_laplacian_PDI1_4_nx_50_n10_i100_dbs_{both,on,off}_200Hz_narrow_band.yaml` |
| PDI4_S2 | 50 | 10 | `laplacian_200Hz/{both,on,off}/psid_laplacian_PDI4_2_nx_50_n10_i100_dbs_{both,on,off}_200Hz_narrow_band.yaml` |
| PDI4_S3 | 50 | 10 | `laplacian_200Hz/{both,on,off}/psid_laplacian_PDI4_3_nx_50_n10_i100_dbs_{both,on,off}_200Hz_narrow_band.yaml` |

Config root: `training/setups/psid/narrow_band_200Hz/` for behavioral,
`training/setups/psid/laplacian_200Hz/` for laplacian.

## Vanilla variants — SKIPPED (already trained, reused for channel selection)

- `psid_{behavioral|laplacian}_{PDI1_2|PDI1_4|PDI4_2|PDI4_3}_nx_{...}_n{...}_i{...}_vanilla_dbs_both_*.yaml` (8 models, untouched)

## Per-variant orchestration (train → test → classify on the go)

For each of the 24 variants, the chain script runs sequentially:

1. **Train** — `python -m training.train --config <cfg.yaml>`
   - Fits forward PSID → eigenvalue-clip A → learn (C_z K_f) and Γ_z K_f via
     RRR → fit backward PSID on z-residuals → learn backward (C_z K_f). ~4-5
     min per variant at nx=50 n1=10 i=100.
2. **Test** — `python -m training.test --config <cfg.yaml>`
   - Runs `predict` (forward-backward smoother) and `validate_forecast`
     (filter-aware forecast) over train/val/test. Writes per-trial predictions
     parquets and HDF5 stats. ~1-2 min.
3. **Classify** (only for `dbs_both` variants) — `python -m classification.compute
   --config classification/setups/<clf.yaml>`
   - LDA on Xp features with 1×5 splits, prediction + forecast modes at
     h ∈ {1, 2, 3} s × m ∈ {0.5, 1.0} s. Writes balanced-accuracy pkls. ~1-2
     min. (on/off variants do not have standalone classification configs;
     they contribute via cross-condition evaluation only.)

**Classification scope**: 8 standalone classifications (one per cell-mode, on the
dbs_both model). Cross-condition classification (on vs off model, flipped labels)
is an additional dashboard-level analysis not covered here.

## Runtime estimate

- Per variant: ~8 min (train + test + classify where applicable)
- 24 variants: ~3 hours sequential
- Parallelizable to ~1-1.5 hours with 2 concurrent trainings (CPU budget permitting)

## Outputs

- `results/<variant>/model_<ts>.pkl` — trained LSSM with FB components
  (CzKf_fwd, CzKf_bwd, GammaZKf_fwd, idSys_bwd) attached.
- `results/<variant>/test/test_results_<ts>.parquet/…` — per-trial prediction
  + forecast arrays (partitioned by participant/session/block/trial).
- `results/<variant>/test_stats_<ts>.hdf5` — learned state-space parameters +
  preprocessing stats.
- `results/classification/<variant>/<ts>/LDA_Xp_{prediction,forecast}_*.pkl` —
  classification BAs (dbs_both only).
