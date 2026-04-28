# Data Formats

## Test results parquet (2026-04-13)

**Location:** `results/{variant}/test/test_results_{TS}.parquet/`
**Format:** Hive-partitioned parquet (participant_id/session/block/trial)

| Column | Type | Description |
|--------|------|-------------|
| Y | list[float] | True neural output (per trial, all channels) |
| Z | list[float] | True behavioral output |
| Yp | list[float] | Predicted neural output |
| Zp | list[float] | Predicted behavioral output |
| Xp | list[float] | Latent states |
| pearson_per_channel | list[float] | Per-channel Pearson r |
| pearson_mean | float | Mean Pearson r across channels |
| pearson_overall_mean | float | Overall Pearson r |
| pearson_per_channel_Z | list[float] | Per-channel Pearson r for Z |
| pearson_mean_Z | float | Mean Pearson r for Z |
| pearson_overall_mean_Z | float | Overall Pearson r for Z |
| time | list[float] | Time array (seconds, relative) |
| offset | float | Offset into recording |
| chunk_margin | float | Margin around chunk |
| margined_duration | float | Duration with margins |
| stim | int | DBS stimulation state (0=off, 1=on) |
| participant_id | str | e.g., "PDI1" |
| session | int | e.g., 2 |
| block | int | Block number |
| trial | int | Trial number within block |

**Missing:** `input_channels`, `output_channels` — not written by the current tester.

## Grid search results.parquet (2026-04-13)

**Location:** `results/psid_gs_{PID}_S{SESS}_200Hz_narrow_band/results.parquet`

Key columns: participant_id, session, timestamp, nx, n1, run_name, success, pearson_mean, pearson_median, pearson_fisher, r_squared, n_trials, xcorr_mean, rmse_Z, rmse_Y, neural_input, behavioral_output

## Classification result dict keys (2026-04-13)

**Source:** `LDA_Xp_prediction.pkl`

accuracy, balanced_accuracy, best_cv_score, best_params, best_pipeline, confusion_matrix, cv_method, f1, fold_results, fpr, grid_search_results, n_combinations_tested, n_splits, precision, recall, roc_auc, tpr, y_pred, y_proba, y_true

- `best_cv_score`: mean CV balanced accuracy (pipeline uses this for model selection)
- `balanced_accuracy`: BA on aggregated held-out fold predictions
- `fold_results`: list of per-fold dicts with their own `balanced_accuracy`

## Model metadata JSON keys (2026-04-13)

**Source:** `model_{TS}_metadata.json`

PSID: framework_type, nx, n1, i, backward_kalman, rescale_states, max_eigenvalue

## DPAD training history (2026-04-13)

**Source:** `training_history.json` inside each DPAD result dir

JSON with epoch-level loss values. Used for DPAD training curves figure.
