# Gotchas

## Test parquets missing input_channels (2026-04-13)

**Problem:** New 200Hz pipeline test parquets (`test/test_results_{TS}.parquet`) don't have an `input_channels` column. The loader in `utils/classification.py:962-976` falls back to scanning for `ECOG_*` / `LFP_*` column prefixes, but those don't exist in the parquet columns either (only Y, Z, Yp, Zp, Xp, etc.).

**Impact:** `dashboard/thesis/loaders.py:resolve_neural_y_channel_idx` raises `ValueError: input_channels is empty; cannot resolve 'ECOG_1_theta_4_8_raw'`. This breaks the neural time series figures in thesis_sec2.

**Workaround:** None yet. Need to either:
1. Add `input_channels` to the tester output, or
2. Read channels from the training config/split metadata, or
3. Patch the loader to read from model metadata

## PDI4 grid search classification was on jacque (2026-04-13)

**Problem:** The chain pipeline ran PDI4 on jacque. Grid search model dirs were synced back, but `results/classification/gs_200Hz/psid_gs_PDI4_*` entries were NOT synced. `pipeline_runs.md` showed complete results (from logs) but local classification data was missing.

**Fix:** Re-ran classification locally with `pipeline_psid.py --start-phase 2 --end-phase 2`. Values match within stochastic CV variation.

## best_cv_score vs balanced_accuracy (2026-04-13)

**Problem:** Classification result dicts have both `best_cv_score` (mean CV balanced accuracy from GridSearchCV) and `balanced_accuracy` (computed on aggregated held-out fold predictions). They differ slightly.

**Rule:** Pipeline selects by `best_cv_score`. The notebook grid search heatmap now also uses `best_cv_score` for consistency.

## DPAD top5 naming (2026-04-13)

**Problem:** DPAD result directories contain `top5` in their names (e.g., `dpad_behavioral_PDI1_2_nx_4_n2_e3000_top5_dbs_both_200Hz_narrow_band`). This was from `--use-psid-channels --top-n 5` being passed during training. But conceptually DPAD gets all 60 channels like PSID — only VARMA uses top-5 selection.

**Decision:** Keep results as-is on disk. The `top5` in the name is a historical artifact from the training script, not a meaningful filter for DPAD.

## _variant_off() breaks with different i values (2026-04-13)

**Problem:** `specs.py` has `_variant_off()` that does `replace("dbs_both", "dbs_off")`. But PSID `both` variant may use a different `i` value than `off`/`on`. For example, PDI1_S2 both uses `i35` but off/on might use different values.

**Impact:** Cross-eval figures need explicit off/on timestamps, not string replacement from the both variant.
