# Results Layout

## Naming convention (2026-04-13)

```
{model}_{type}_{PID}_{SESS}_nx_{NX}_n{N1}[_i{I}][_e{EPOCHS}][_top5]_dbs_{COND}_200Hz_narrow_band
```

- `model`: psid, dpad, varma
- `type`: behavioral (always, for now)
- `PID`: PDI1 or PDI4
- `SESS`: session number (2, 3, or 4)
- `NX`: latent state dimension
- `N1`: behaviorally-relevant latent count (n1 ≤ nx)
- `I`: PSID iteration count (grid search uses gs_i=30, full uses full_i=50)
- `EPOCHS`: DPAD training epochs (3000)
- `top5`: VARMA only — uses top-5 PSID channels; DPAD dirs also have `top5` in name (historical artifact) but actually get all 60 channels
- `COND`: dbs_both, dbs_off, dbs_on

## Grid search results

**Location:** `results/psid_gs_{PID}_S{SESS}_200Hz_narrow_band/`

Each contains:
- `results.parquet` — summary with columns: participant_id, session, timestamp, nx, n1, run_name, pearson_mean, etc.
- `psid_gs_..._run{NNN}_{TS}/` — individual run directories
  - `model_{TS}.pkl` — trained PSID model
  - `model_{TS}_metadata.json` — has `nx`, `n1`, `i`, etc.
  - `split/` — train/val/test split indices

Run numbering: run000=(nx=4,n1=2), run001=(nx=4,n1=4), run002=(nx=8,n1=2), run003=(nx=8,n1=4), run004=(nx=15,n1=2), run005=(nx=15,n1=4), run006=(nx=25,n1=2), run007=(nx=25,n1=4)

**Sessions with data:**
- PDI1_S2: 8 runs, ts=20260410_224415
- PDI1_S4: 8 runs, ts=20260411_031957
- PDI4_S2: 8 runs, ts=20260411_004416
- PDI4_S3: 7 runs (run003 missing = nx=8,n1=4), ts=20260411_063215

## Grid search classification

**Location:** `results/classification/gs_200Hz/{run_dir_name}/{cls_timestamp}/`

Each contains:
- `LDA_Xp_prediction.pkl` — dict with keys: accuracy, balanced_accuracy, best_cv_score, best_params, best_pipeline, confusion_matrix, cv_method, f1, fold_results, fpr, grid_search_results, n_combinations_tested, n_splits, precision, recall, roc_auc, tpr, y_pred, y_proba, y_true
- `h4.5_m2.0/LDA_Xp_forecast.pkl` — forecast-based classification

**Important:** `best_cv_score` = mean CV balanced accuracy (used by pipeline for model selection). `balanced_accuracy` = test-set BA. These differ slightly.

Also has laplacian variants: `psid_gs_lap_{PID}_S{SESS}_200Hz_narrow_band_...`

## Full training results

**Location:** `results/{variant_name}/`

Each contains:
- `model_{TS}.pkl` — trained model
- `model_{TS}_metadata.json` — config metadata
- `split/` — split indices
- `test/test_results_{TS}.parquet/` — Hive-partitioned parquet (participant_id/session/block/trial)
  - Columns: Y, Z, Yp, Zp, Xp, pearson_per_channel, pearson_mean, pearson_overall_mean, pearson_per_channel_Z, pearson_mean_Z, pearson_overall_mean_Z, time, offset, chunk_margin, margined_duration, stim, participant_id, session, block, trial
  - **GOTCHA: no `input_channels` column** — the loader falls back to scanning for ECOG_/LFP_ prefixed columns, which don't exist in the new 200Hz parquets. This breaks `resolve_neural_y_channel_idx` in loaders.py.

## Selected configs (from pipeline grid search CV BA)

| Session  | nx  | n1 | PSID i | CV BA  |
|----------|-----|----|--------|--------|
| PDI1_S2  | 4   | 2  | 35/50  | 0.5245 |
| PDI1_S4  | 25  | 2  | 50     | 0.7449 |
| PDI4_S2  | 15  | 2  | 35/50  | 0.5268 |
| PDI4_S3  | 25  | 2  | 35/50  | 0.5631 |

## DPAD status (2026-04-13)

All 12 DPAD directories (4 sessions x 3 conditions) have:
- `model_*.pkl` + `training_history.json` (training complete)
- **NO** test/train/val subdirectories (not tested yet)
- Need `pipeline_dpad.py --start-phase 2` to generate test parquets

## Local vs jacque

- PDI1 (S2, S4) — trained and classified **locally**
- PDI4 (S2, S3) — trained on **jacque** (10.0.0.2), results synced back via rsync
  - Grid search model dirs synced ✓
  - Classification results were NOT synced (re-run locally 2026-04-13)
  - Full training results synced ✓
