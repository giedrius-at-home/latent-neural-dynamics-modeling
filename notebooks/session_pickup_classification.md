# Session Pickup — Classification Results + New Forecasts

Last updated: 2026-06-06 — ALL RUNS COMPLETE

---

## Status: DONE (2026-06-06)

All 8 DPAD + 8 PSID classification configs have parquets in `results/{dpad,psid}/*/classification/sweep_*.parquet`.
Latest timestamps: DPAD 2026-06-03, PSID 2026-06-02/04.

### Completed configs

| Framework | Session | z-type | Has parquet |
|-----------|---------|--------|-------------|
| DPAD | PDI1_S2 | z-as-behavior | yes (rerun 2026-06-03) |
| DPAD | PDI1_S2 | z-as-neural | yes |
| DPAD | PDI1_S4 | z-as-behavior | yes |
| DPAD | PDI1_S4 | z-as-neural | yes (rerun after crash) |
| DPAD | PDI4_S2 | z-as-behavior | yes |
| DPAD | PDI4_S2 | z-as-neural | yes |
| DPAD | PDI4_S3 | z-as-behavior | yes |
| DPAD | PDI4_S3 | z-as-neural | yes |
| PSID | all 8 | both z-types | yes (jacque rsync done) |

### Remaining TODOs
- [ ] Flip `SHOW_DPAD = True` in `thesis_sec5_classification.ipynb` cell 5
- [ ] Verify new parquets have `perm_mean_ba`, `y_true`, `y_pred` columns
- [ ] Build confusion matrices from y_true/y_pred

---

## Historical: run status (2026-06-03)

### Local — DPAD classification chain (`scripts/wait_then_classify.sh`)

| Config | Status | Notes |
|--------|--------|-------|
| dpad_modal_PDI1_S2_z-as-behavior | DONE (02:33 Jun 2) | parquet missing new columns — needs rerun |
| dpad_modal_PDI1_S2_z-as-neural | DONE (07:51 Jun 2) | parquet OK |
| dpad_modal_PDI1_S4_z-as-behavior | DONE (11:28 Jun 2) | parquet OK |
| dpad_modal_PDI1_S4_z-as-neural | CRASHED (13:27 Jun 2) | NaN in Xf for dbs=on — fixed in sweep.py; needs rerun |
| dpad_modal_PDI4_S2_z-as-behavior | NOT STARTED | chain died after crash |
| dpad_modal_PDI4_S2_z-as-neural | NOT STARTED | |
| dpad_modal_PDI4_S3_z-as-behavior | NOT STARTED | |
| dpad_modal_PDI4_S3_z-as-neural | NOT STARTED | |

Estimated ~5-8h per config (z-as-neural longer than z-as-behavior due to 8 LFP targets vs 2).
Chain is fully automated — no manual intervention needed until all 8 are done.

Monitor: `tail -f /tmp/classify_logs/wait_then_classify.log`
Per-config: `tail -f /tmp/classify_logs/dpad_modal_PDI1_S2_z-as-neural.log`

### Jacque (10.0.0.2) — PSID classification

PSID classification started 08:39 UTC. Currently running LEDOIT_WOLF covariance estimation
(visible in log). All 8 PSID configs run sequentially.

Monitor: `ssh jacque "tail -f /tmp/classify_logs_psid/main.log"`

Jacque has no pre-existing classification parquets — will produce fresh output for all 8 sessions.

---

## What changed in the code (sweep.py)

### New columns in classification parquets

All new runs (DPAD configs 2-8, all PSID on jacque) will have:

| Column | Type | Note |
|--------|------|------|
| `perm_mean_ba` | float (NaN if gate not crossed) | Mean of null distribution from permutation test |
| `y_true` | list[int] | Per-trial true labels for confusion matrix reconstruction |
| `y_pred` | list[int] | Per-trial predicted labels |

`perm_ba_gate = 0.5` — permutation test only runs when `cv_ba > 0.5`.
`perm_mean_ba` is always saved (NaN when gate not crossed), so the column is always present.

### Config 1 needs rerun

`dpad_modal_PDI1_S2_z-as-behavior` ran before the column fix.
Its parquet `sweep_20260602_023301.parquet` has `predictions` + `forecast` rows (144 total)
but NO `perm_mean_ba`, `y_true`, `y_pred`.

**After all other configs finish:** rerun config 1 manually:
```bash
TF_FORCE_GPU_ALLOW_GROWTH=true python -m training.pipeline \
  --config training/setups/dpad_modal/dpad_modal_PDI1_S2_z-as-behavior.yaml \
  --phases classification \
  2>&1 | tee /tmp/classify_logs/dpad_PDI1_S2_zb_rerun.log
```

### h_grid trimmed

All 8 `dpad_modal_PDI*.yaml` now have `h_grid: [5.0]` only (was 7 values).
This means classification sweeps only run `h=5.0, m=2.0` — consistent with the
trained forecast horizon. No multi-h grid search anymore.

### Classifiers saved

DPAD classification now saves LDA classifiers as `.joblib` files in:
`results/dpad/<variant>/classification/classifiers/clf_*.joblib`

This enables PSID to load classifiers instead of retraining if needed later.

---

## Parquet schema comparison

### PSID (old, local — pre-fix)
```
cols: mode, pipeline, variant, run_ts, dbs_train, data_dbs, sub_source,
      flipped, t_cut_seconds, h_seconds, m_test_seconds,
      cv_ba, ba_at_score, n_score, n_permutations, p_value
rows: 792 per session
```
No `perm_mean_ba`, no `y_true`, no `y_pred`.
Old PSID parquets (from 2026-05-13/14) are **stale** — jacque run replaces them.

### DPAD config 1 (overnight, missing fix)
```
Same cols as PSID above. modes: predictions + forecast. rows: 144.
```

### DPAD configs 2-8 + new PSID (with fix)
```
+ perm_mean_ba: float
+ y_true: list[int]
+ y_pred: list[int]
```

---

## Forecast parquets (new from overnight jobs, Jun 1)

All 24 DPAD forecast variants have h5 forecast parquets:
`results/dpad/<variant>/forecast/h5/{train,val,test}/test_results_20260513_*.parquet`

All 4 sessions x 2 z-types x 3 dbs_train conditions = 24 variants. Complete.
These were computed by the overnight forecast jobs (PIDs 2015794-2015801) before classification.

---

## After runs complete — checklist

- [ ] Rsync PSID results from jacque back to local:
  ```bash
  rsync -avz --progress jacque:repos/latent-neural-dynamics-modeling/results/psid/ \
    results/psid/ --exclude "split/" --exclude "train/" --exclude "val/"
  ```
  (only sync classification/ subdirs to avoid re-copying the 42GB of inference parquets)

- [ ] Verify new PSID parquets have `perm_mean_ba`, `y_true`, `y_pred`

- [ ] Rerun DPAD config 1 (see above)

- [ ] Flip SHOW_DPAD = True in `notebooks/thesis_sec5_classification.ipynb` cell 5
  once at least one DPAD config is complete and verified

- [ ] Verify confusion matrix reconstruction works with new `y_true`/`y_pred` columns

---

## sec5 classification notebook — what to update

File: `notebooks/thesis_sec5_classification.ipynb`

Key flag: `SHOW_DPAD = False` in cell 5 — flip to `True` to enable DPAD panels.

Data loaders called:
- `build_cls_points_from_parquet(mode="predictions", flipped=False, dbs_filter="both")` — predictions sweep
- `build_cls_points_from_parquet(mode="forecast", flipped=True, dbs_filter="both")` — flipped forecast
- `build_forecast_bests_from_parquet()` — best forecast horizon per session

DPAD data slots (`dpad_ba`, `dpad_pv`, `dpad_flip_ba`, `dpad_flip_pv`) are initialized as NaN
arrays when SHOW_DPAD=False. They need to be populated from the new DPAD parquets.

Confusion matrices: cell 7 calls `_draw_confusion_cell` with `cm=None` if no data.
With new `y_true`/`y_pred` columns, can reconstruct `cm` via `sklearn.metrics.confusion_matrix`.

---

## Known issues / caveats

- **PDI1 spectral floor**: PDI1_S2 and PDI1_S4 have near-zero DBS-on/off spectral separation
  (confirmed in sec1 fig_spectral_separability). Expect cv_ba ~0.5 for those sessions — real
  signal only testable on PDI4_S2 and PDI4_S3.

- **VARMA classification excluded**: VARMA produces empty Xp. Classification is PSID + DPAD only.

- **PSID laplacian at chance**: PSID cross-decoding (ECoG->LFP) fails (r~0). LFP
  classification results for PSID should be interpreted as near-chance.

- **Old PSID parquets**: Keep old local files until jacque run verified complete and rsynced.
  Do not delete until new parquets confirmed correct.
