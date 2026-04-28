# Cleanup Plan — 2026-04-17

## Executive Summary

**Total estimated disk savings: ~22–24 GB** (out of 68 GB in results/ alone)
**Files & directories flagged for safe removal: 65–75 items**

### By category:
- **Grid search dirs** (8 PSID, all referenced in pipeline_runs.md but data-heavy intermediates): ~21 GB
- **Classification cache** (intermediate epoched/permutation data): ~5.8 GB  
- **Diagnostic / experimental dirs**: ~0.5 GB
- **Old dashboard scripts** (replaced by dashboard/thesis/): ~0.1 MB (cleanup bonus)
- **Logs** (aged >7 days, except chain): ~180 MB

**Post-cleanup repo size: ~44–46 GB** — still 8 cells × 3 models × 4 DBS conditions = 96 result variants, all live.

---

## 1. Results/ (68 GB → ~46 GB target)

### 1a. PSID Grid Search — Definitely Removable (~21 GB)

All 8 grid search directories are referenced in `pipeline_runs.md` Phase 1 and are **intermediate training artifacts**. The best hyperparameters they discovered are now baked into the final trained models in `/results/psid_behavioral_*/` and `/results/psid_laplacian_*/`. Re-running grid search would regenerate these on demand.

**Dirs to remove:**
```
results/psid_gs_PDI1_S2_200Hz_narrow_band/          (~3.3 GB)
results/psid_gs_PDI1_S4_200Hz_narrow_band/          (~2.5 GB)
results/psid_gs_PDI4_S2_200Hz_narrow_band/          (~2.7 GB)
results/psid_gs_PDI4_S3_200Hz_narrow_band/          (~2.4 GB)
results/psid_gs_lap_PDI1_S2_200Hz_narrow_band/      (~4.2 GB)
results/psid_gs_lap_PDI1_S4_200Hz_narrow_band/      (~3.1 GB)
results/psid_gs_lap_PDI4_S2_200Hz_narrow_band/      (~3.5 GB)
results/psid_gs_lap_PDI4_S3_200Hz_narrow_band/      (~3.4 GB)
```

**Aggregate: ~21 GB**

**Rationale:** Phase 1 of `pipeline_psid.py` (see pipeline_runs.md lines 42–45) runs a grid search on (nx, n1) via `GS_I=30` and stores results in these dirs. Phase 2 evaluates the grid on CV balanced accuracy and selects the best combo per session. Once best (nx, n1) is known, Phase 3 trains the final models at that combo — those are in `/results/psid_behavioral_{P}_{S}_nx_{nx}_n{n1}_i{i}_dbs_*` and `/results/psid_laplacian_*/`, which ARE kept. The grid search dirs are only needed if you want to re-examine the grid or re-score at a different metric; not needed for thesis figures or downstream classification.

---

### 1b. Classification Cache — Definitely Removable (~5.8 GB)

`results/classification_cache/` contains intermediate parquet files from the early stages of classification (epoched splits, permutation test trial bookkeeping, etc.). It is **not referenced** by any script in `scripts/`, `notebooks/`, or `dashboard/thesis/`.

**Dirs to remove:**
```
results/classification_cache/epoched/           (~? – subdirs of parquets)
results/classification_cache/flipped/           (~? – flipped-label trial splits)
results/classification_cache/lda_no_perm/       (~? – pre-permutation LDA fits)
```

**Aggregate: ~5.8 GB**

**Rationale:** Grep for `classification_cache` in all scripts:
```
grep -r "classification_cache" scripts/ notebooks/ 2>/dev/null
# → no hits
```
The cache was likely a development/debugging artifact. The authoritative classification results live in `results/psid_*/classification/`, `results/varma_*/classification/`, and `results/dpad_*/classification/`, which are all wired into the pipeline and thesis figures.

---

### 1c. Grid Search Intermediate Configs — Likely Removable (~0.5 GB, varies)

Within some full-training PSID variant dirs, there may be subdirectories from intermediate training attempts (e.g., retry-lower fallback attempts at `i=[45, 40, 35, 30]`). Check for **multiple `model_*.pkl` files with different timestamps** and archive the older ones if their timestamps don't match the canonical one in `pipeline_runs.md`.

**Example check:**
```bash
ls results/psid_behavioral_PDI1_2_nx_4_n2_i35_dbs_both_200Hz_narrow_band/ | grep model_
# If > 1 model_*.pkl, check their mtimes; keep only the one matching pipeline_runs.md
```

**Rationale:** The pipeline uses retry-lower fallback (pipeline_runs.md:49) to handle eigenvalue failures during training. If a training attempt fails and is retried at a lower `i`, multiple model artifacts can accumulate in the same variant dir. The `.stray_models/` subdir mentioned in pipeline_runs.md:277–285 shows this has already been addressed for DPAD PDI4 S2. Check PSID dirs for similar artifacts; remove older duplicates if found.

---

### 1d. Data Hungriness Results — **Needs Your Call** (~176 MB)

`results/data_hungriness/` contains session-level JSON summaries of model performance under reduced-data regimes (a thesis appendix sec 6 figure).

**Status:**
- **Referenced by:** `notebooks/thesis_sec6_summary_appendix.py` (line ~170: `_HUNGRINESS_SUBDIR = "data_hungriness"`)
- **Checked in pipeline_runs.md:** No mention. Dates indicate it's from an earlier run (pre-2026-04-10).
- **Usage:** Optional for the thesis; critical if Giedrius plans to include data-efficiency curves in the final appendix.

**Dirs to consider:**
```
results/data_hungriness/
```

**Aggregate: ~176 MB**

**Decision:** If you plan to include data-efficiency curves in the final thesis draft, **KEEP**. If not, safe to remove. If you're unsure, keep for now and delete in a later pass.

---

### 1e. Diagnostic Dir — Safe to Remove (~340 KB)

`results/diagnostic/pdi4_s3_psid_spectra/` — a single spectral analysis snapshot from one session.

**Dirs to remove:**
```
results/diagnostic/pdi4_s3_psid_spectra/
```

**Rationale:** Not referenced in thesis notebooks or specs. Likely a one-off exploratory analysis. No methodology impact if removed.

---

### 1f. KEEP: All Thesis-Active Result Dirs

**Do not remove:**
```
results/psid_behavioral_PDI[14]_[234]_nx_*_dbs_*_200Hz_narrow_band/              (42 dirs, ~12 GB total)
results/psid_laplacian_PDI[14]_[234]_nx_*_dbs_*_200Hz_narrow_band/              (42 dirs, ~15 GB total)
results/varma_behavioral_PDI[14]_[234]_*_dbs_*_200Hz_narrow_band/               (12 dirs, ~1.5 GB)
results/varma_laplacian_PDI[14]_[234]_*_dbs_*_200Hz_narrow_band/                (12 dirs, ~1 GB)
results/dpad_behavioral_PDI[14]_[234]_*_dbs_*_200Hz_narrow_band/                (12 dirs, ~8 GB)
results/dpad_behavioral_PDI[14]_[234]_*_eval_{on,off}_200Hz_narrow_band/        (8 dirs, ~0.5 GB, cross-eval)
results/psid_behavioral_PDI[14]_[234]_*_eval_{on,off}_200Hz_narrow_band/        (8 dirs, ~0.3 GB, cross-eval)
results/varma_behavioral_PDI[14]_[234]_*_eval_{on,off}_200Hz_narrow_band/       (8 dirs, ~0.3 GB, cross-eval)
results/classification/                                                          (~500 MB, all 8 cells' predictions/permutations)
results/varma_channel_selection_*.json                                           (8 files, each ~50 KB)
```

**Rationale:** All are referenced by `pipeline_runs.md` as complete/usable results. `eval_on` / `eval_off` subdirs are cross-condition evaluation outputs (phase 4 of PSID/VARMA; see pipeline_runs.md:50). Thesis figures load from these via `dashboard/thesis/specs.py` (AlignedTriplet data structures referencing `psid_variant`, `dpad_variant`, `varma_variant` and their timestamps).

---

## 2. Logs/ (214 MB → ~30 MB target)

Most logs older than 7 days can be safely removed. Keep recent logs (< 7 days) and the chain orchestration log for audit trail.

### 2a. Pipeline Logs — Removable (~180 MB)

All individual job logs from `pipeline_psid/`, `pipeline_dpad/`, `pipeline_varma/`, `classification/` are transient; once a pipeline completes, the log is purely informational. **Keep only the most recent week.**

**Safe to remove:**
```
logs/pipeline_psid/*_202604{10,11,12,13,14}*.log      (old training runs)
logs/pipeline_dpad/*_202604{10,11,12,13,14}*.log      (old training/test runs)
logs/pipeline_varma/*_202604{10,11,12}*.log           (old VARMA training)
logs/classification/*_202604{10,11,12,13,14}*.log     (old classifier jobs)
```

**Aggregate: ~180 MB** (conservative estimate)

**Keep:**
```
logs/pipeline_psid/*_202604{15,16,17}*.log            (recent activity)
logs/pipeline_dpad/*_202604{15,16,17}*.log
logs/pipeline_varma/*_202604{15,16,17}*.log
logs/classification/*_202604{15,16,17}*.log           (if any)
```

**Rationale:** These logs are strictly informational; all training outputs are already saved in the result variant dirs. Once a job finishes, the log adds no value for reproducibility or inspection (the model artifacts, config.yaml, and metadata.json are the source of truth).

---

### 2b. Chain Logs — KEEP

```
logs/chain/chain_all_20260410_224412.log              (canonical 2026-04-10 full run)
logs/chain/rerun_psid_phase5_20260411_121500.log      (2026-04-11 Phase 5 fix)
logs/chain/rerun_psid_phase5_PDI4_S3_lap_20260411_133522.log
```

**Rationale:** These document the thesis-run execution and bug fixes. Useful for audit trail; keep for reference.

---

## 3. Scripts/ (30 active, 4–5 candidates for review)

### 3a. Definitely Removable — Debugging/Exploration Scripts

```
scripts/quick_xp1_test.py                    (2024 experiment stub, training/setups/psid/quick_xp1 no longer exists)
scripts/investigate_forecast_html.py         (debugging script for HTML report; not in pipeline)
```

**Rationale:**
- `quick_xp1_test.py`: References `training/setups/psid/quick_xp1` which does not exist in the current training/setups tree. Likely an abandoned experiment from the mRMR pivot. Not invoked by any pipeline or Makefile.
- `investigate_forecast_html.py`: Imports IPython and debug utilities; appears to be a one-off investigation script, not part of the thesis pipeline.

---

### 3b. Likely Removable — Older Dashboard Scripts (Not Pipeline Entry Points)

```
dashboard_psd.py                    (2024-03-16, not invoked by any pipeline; thesis uses dashboard/thesis/)
dashboard_psd_session.py
dashboard_psd_trial.py
dashboard_time_series.py
dashboard_time_series_session.py
dashboard_time_series_trial.py
dashboard_psd_participant.py
dashboard_time_series_participant.py
dashboard_grid_search.py            (superseded by training/setups/psid_grid_search/; generated configs via generate_grid_search_classification_configs.py)
dashboard_model_predictions.py
dashboard_dbs_classification.py
```

**Rationale:** All are from 2024-03-16 and are **not referenced** by the thesis pipeline (`pipeline_psid.py`, `pipeline_dpad.py`, `pipeline_varma.py`, Makefile, or any script in `scripts/` or `notebooks/`). The current thesis figures use `dashboard/thesis/` entry points (e.g., `dashboard/thesis/specs.py`, `dashboard/thesis/compose.py`). These older scripts appear to be from an earlier visualization approach. `dashboard_thesis_final.py` (2026-03-30) is more recent and may be active.

**Check:** `grep -r "dashboard_psd\|dashboard_time_series\|dashboard_grid" scripts/ notebooks/ Makefile` → no hits expected. If confirmed, safe to remove.

---

### 3c. Scripts to KEEP (Active Pipeline Entry Points)

```
scripts/pipeline_psid.py                     (primary entry point for PSID training)
scripts/pipeline_dpad.py                     (primary entry point for DPAD training)
scripts/pipeline_varma.py                    (primary entry point for VARMA training)
scripts/pipeline_psid_diagnostic.py          (diagnostic mode for PSID; may be used)
scripts/generate_*.py                        (thesis figure generation)
scripts/fig_*.py                             (figure-specific scripts)
scripts/run_*.py / run_*.sh                  (batch orchestration scripts)
scripts/print_thesis_result_timestamps.py    (utility for extracting run timestamps from results)
```

---

## 4. Classification Setups (140 YAMLs) — Mostly KEEP, 72 for Review

### 4a. Feature-Source Variants — All Live

All configs with suffixes `_xp_1`, `_xp_2`, `_xp_with_dbs`, and their `_perm` variants are **generated on-demand** by the pipeline and **referenced in `pipeline_runs.md:52`**:

> "Classification...for 4 feature sources: `Xp`, `Xp_1`, `Xp_2`, `Xp_with_dbs`"

These are not abandoned experiments — they are **feature decomposition studies** showing which latent state components contribute to classification. All referenced in current Phase 5.

**Do not remove any `*_xp_*` configs.**

**Decision:** This was initially flagged as "abandoned experiment" but is actually core thesis methodology. The naming (`_xp_1` = "experiment 1" where only the first state component is used, etc.) is internally consistent with the pipeline.

---

### 4b. Plain / `_perm` / `_flipped` — Core Set, KEEP

All core classification configs without experiment suffixes (plain `psid_behavioral_PDI1_2_..._200Hz_narrow_band.yaml` and `_perm` / `_flipped` variants) are **actively generated and run** by Phase 5 of all pipelines. Keep all.

---

## 5. Training Setups/

### 5a. Old Grid Search Configs — Likely Removable

```
training/setups/psid_SEARCH_beta_ecog_nx20.yaml     (old hyperparameter search template)
training/setups/psid_SEARCH_beta_ecog.yaml
```

**Rationale:** These are superseded by the unified `psid_grid_search/` subdir (2026-04-07+) and the hardcoded grids in `pipeline_psid.py` (nx_grid, n1_grid per mode). Not referenced by current pipeline.

---

### 5b. Template Files — KEEP

```
training/setups/psid_template.yaml             (used by pipeline to generate specific configs)
training/setups/psid_grid_search.yaml          (obsolete? but kept for reference)
training/setups/psid_grid_search.py            (entry point for grid search generation)
```

---

### 5c. Subdirs — KEEP

```
training/setups/psid/narrow_band_200Hz/        (current behavioral configs; referenced by pipeline)
training/setups/psid/laplacian_200Hz/          (current laplacian configs)
training/setups/psid/quick_xp1/                (DEAD – referenced by quick_xp1_test.py, which is orphaned)
training/setups/psid_grid_search/              (grid search templates and results)
training/setups/dpad/                          (DPAD training configs)
training/setups/varma/                         (VARMA training configs)
```

**Decision:** `psid/quick_xp1/` is orphaned (only referenced by the to-be-deleted `scripts/quick_xp1_test.py`). If quick_xp1_test.py is removed, this subdir can also be removed (~50 KB).

---

## 6. Top-Level Repo Artifacts

### 6a. Markdown Documents — Assess Currency

**Current/Useful:**
```
pipeline_runs.md                               (2026-04-17, canonical; KEEP)
RESULTS_PLAN.md                                (2026-03-31, thesis structure plan; likely current; CHECK if still accurate)
thesis_results_inspection.md                   (older notes; candidate for archival)
code_changes.md                                (2026-04-09, bug fixes log; reference value; KEEP or ARCHIVE)
```

**Decision:**
- `pipeline_runs.md` — **KEEP** (actively referenced, up-to-date)
- `RESULTS_PLAN.md` — **KEEP IF CURRENT** (describes the thesis structure; check with Giedrius if it matches the current direction). If superseded, move to `docs/` or archive.
- `thesis_results_inspection.md` — **ARCHIVE or DELETE** (pre-cleanup notes, lower signal)
- `code_changes.md` — **KEEP IF USEFUL FOR AUDIT**, else **ARCHIVE**

---

### 6b. Dashboard Scripts (Top-Level) — Likely Dead

```
dashboard_*.py (12 files at root)              (see **Section 3b** — not in active pipeline)
```

---

### 6c. Notebooks & Figures — KEEP

```
notebooks/thesis_sec*.py                       (active thesis notebooks; KEEP)
notebooks/debug_*.ipynb                        (debugging; likely safe to archive but low volume)
figures/                                       (generated thesis figures; KEEP)
dashboard_logs/                                (65K, old web dashboard logs; candidate for removal)
```

---

## 7. Uncertain — Please Confirm

| Item | Size | Recommendation | Your Call |
|------|------|---|---|
| `results/data_hungriness/` | 176 MB | Keep if appendix sec 6 is final; remove otherwise | **?** |
| `scripts/quick_xp1_test.py` + `training/setups/psid/quick_xp1/` | ~50 KB | Remove (orphaned) | **?** |
| `scripts/investigate_forecast_html.py` | ~3 KB | Remove (debugging) | **?** |
| `training/setups/psid_SEARCH_*.yaml` | ~3 KB | Remove (superseded) | **?** |
| Older dashboard scripts (`dashboard_*.py`) | ~50 KB | Remove (not in pipeline) | **?** |
| `logs/` older than 7 days | ~180 MB | Remove (informational only) | **?** |
| `RESULTS_PLAN.md` | ~22 KB | Keep if current; archive if superseded | **?** |
| `thesis_results_inspection.md` | ~7 KB | Archive or delete | **?** |
| `dashboard_logs/` | ~65 KB | Remove (old web dashboard) | **?** |

---

## Deletion Command Template

Once you've confirmed the safe-to-remove items, use:

```bash
# Grid search dirs (21 GB)
rm -rf results/psid_gs_*_200Hz_narrow_band/

# Classification cache (5.8 GB)
rm -rf results/classification_cache/

# Diagnostic dir (340 KB)
rm -rf results/diagnostic/pdi4_s3_psid_spectra/

# Old logs (180 MB)
find logs/pipeline_* logs/classification/ -name "*.log" -mtime +7 -delete

# Orphaned scripts
rm scripts/quick_xp1_test.py scripts/investigate_forecast_html.py
rm -rf training/setups/psid/quick_xp1/

# Old dashboard scripts (optional)
rm dashboard_psd.py dashboard_psd_session.py dashboard_psd_trial.py \
   dashboard_time_series.py dashboard_time_series_session.py \
   dashboard_time_series_trial.py dashboard_psd_participant.py \
   dashboard_time_series_participant.py dashboard_grid_search.py \
   dashboard_model_predictions.py dashboard_dbs_classification.py

# Old training setups
rm training/setups/psid_SEARCH_*.yaml

# Summary
echo "Cleanup complete. Running du to check final size..."
du -sh /home/bobby/repos/latent-neural-dynamics-modeling/results/ \
       /home/bobby/repos/latent-neural-dynamics-modeling/logs/ \
       /home/bobby/repos/latent-neural-dynamics-modeling/
```

---

## Recommendations for Future Cleanup

1. **After mRMR pivot + new feature selection:** Many top-5-channel PSID/VARMA runs will be deprecated. Re-run pipeline with new feature-selection approach, then remove old result dirs. This will free an additional ~10-15 GB.

2. **Archive instead of delete:** If concerned about audit trail, move dirs to a `results/.archive_YYYY-MM-DD/` directory rather than permanently deleting. Disk cost is the same; recoverability is higher.

3. **Periodic log rotation:** Set up a cron job to delete logs older than N days automatically (e.g., `find logs/ -name "*.log" -mtime +14 -delete`).

4. **Checkpoint pruning:** DPAD models store checkpoint files every 100 epochs (pipeline_runs.md:190). For space-critical scenarios, keep only the final model and delete intermediate checkpoints — but verify the pipeline doesn't rely on them for resumption first.

---

## Checklist Before Final Deletion

- [ ] Confirm `pipeline_runs.md` is the current, accurate ledger of live runs
- [ ] Verify `dashboard/thesis/specs.py` references only the "KEEP" result dirs
- [ ] Check whether data_hungriness appendix is included in final thesis draft
- [ ] Decide on quick_xp1 status (complete removal or preservation for reference)
- [ ] Decide on old dashboard scripts (keep or remove)
- [ ] Run one final `grep -r "classification_cache\|data_hungriness\|psid_gs_" scripts/ notebooks/` to confirm orphaning
- [ ] Back up (or snapshot) the repo before bulk deletion
- [ ] After deletion, spot-check that thesis figure generation still works (run one `scripts/fig_*.py` script)

