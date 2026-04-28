# Pipeline Runs — 200 Hz Narrow-Band (thesis rerun 2026-04-10/11)

Three modeling pipelines (PSID → VARMA → DPAD) were re-run after several
bug fixes (see **Bugs fixed** at the bottom). This document captures the
configuration and the per-cell results as of 2026-04-11.

- **Cells** = 4 sessions × 2 feature modes = **8 cells**
  - Sessions: `PDI1 S2`, `PDI1 S4`, `PDI4 S2`, `PDI4 S3`
  - Modes: `behavioral` (decode tracing kinematics from ECoG),
    `laplacian` (decode LFP laplacian-14-16 from ECoG)
- **DBS conditions** trained per cell: `both`, `on`, `off`, `vanilla` (both, no DBS label as input)

## Data & preprocessing (shared)

| Field | Value |
|---|---|
| Config | `preprocessing/participants_at_200Hz_scaled_1e6_narrow_band.yaml` |
| Data root | `resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band` |
| Sampling frequency | 200 Hz |
| Scale factor | 1e6 |
| Notch | 50 Hz + 100 Hz |
| CAR | enabled (`apply_car: true`) |
| Raw bands | 15 narrow bands from θ to γ 75–80 Hz (see yaml) |
| Max pause | 2.0 s |

**ECoG input**: 60 channels = 4 electrodes × 15 bands, `ECOG_{1..4}_{band}_raw`.

**Split**: within-session chronological 60/10/30 train/val/test, `min_train_epochs=50`.

Splits are deterministic across modes — so behavioral and laplacian runs on
the same session reuse the exact same time segments, and VARMA/DPAD copy
their splits from the PSID variant to guarantee alignment.

---

## PSID pipeline

Unified entry point: **`scripts/pipeline_psid.py`** (replaces all old
`pipeline_psid_*.py` / `pipeline_psid_laplacian_*.py` per-session scripts).

### Phases
1. **Grid search** — `GS_I=30`, `(nx, n1)` pruned grid (see Mode config below),
   1 worker (RAM-limited), output `results/psid_gs*` / `results/psid_gs_lap*`.
2. **Best (nx, n1)** — scored on CV balanced accuracy from
   forecast-based classification at `(h=4.5, m=2.0)` (history 4.5 s, forecast 2.0 s).
   Not Pearson r — see `feedback_use_pipeline_script.md`.
3. **Full training** — 4 variants per best combo: `both`, `on`, `off`, `vanilla`.
   `FULL_I` per mode (see below). 2 training jobs in parallel.
   Retry-lower fallback on eigenvalue failures: `i ∈ [full_i, 45, 40, 35, 30]`.
4. **Cross-evaluation** — eval each fit on its own test split, plus on-vs-off cross evals.
5. **Classification** — coarse `h × m` grid (no permutations) → permutation test
   on CV-best, for 4 feature sources: `Xp`, `Xp_1`, `Xp_2`, `Xp_with_dbs`,
   and the flipped-label classifier.
6. **Thesis specs/HTML update** — behavioral only (laplacian triplet not wired up).

### Mode config

| | Behavioral | Laplacian |
|---|---|---|
| `full_i` | 50 | 45 |
| `nx_grid` | `[4, 8, 15, 25]` | `[15, 25, 40, 55]` |
| `n1_grid` | `[2, 4]` | `[4, 8]` |
| Valid combos | 7 `(n1<nx)` | 8 `(n1<nx)` |
| Output type | `behavioral` | `neural` |
| Outputs | `tracing_velocity_x`, `tracing_acceleration_magnitude` | `LAPLACIAN_14-16_LFP_{band}` × 15 bands |
| Variant prefix | `psid_behavioral_{P}_{S}` | `psid_laplacian_{P}_{S}` |
| Training yaml subdir | `psid/narrow_band_200Hz` | `psid/laplacian_200Hz` |
| GS base | `psid_gs_{P}_S{S}_200Hz_narrow_band` | `psid_gs_lap_{P}_S{S}_200Hz_narrow_band` |

### Per-cell results

Chain log: `logs/chain/chain_all_20260410_224412.log`.
Phase 5 was re-run on 2026-04-11 after the `block_samples` classification fix
(log: `logs/chain/rerun_psid_phase5_20260411_121500.log` + single-cell rerun
`logs/chain/rerun_psid_phase5_PDI4_S3_lap_20260411_133522.log`).

| Cell | Best nx | Best n1 | CV BA | `i` actually used (both / on / off / vanilla) |
|---|---|---|---|---|
| PDI1 S2 behavioral | 4  | 2 | 0.5000 | 35 / 50 / 50 / 45 |
| PDI1 S2 laplacian  | 15 | 8 | 0.5208 | 30 / 45 / 45 / 40 |
| PDI1 S4 behavioral | 25 | 2 | 0.7500 | 50 / 50 / 50 / 50 |
| PDI1 S4 laplacian  | 25 | 8 | 0.7556 | 45 / 45 / 45 / 45 |
| PDI4 S2 behavioral | 15 | 2 | 0.5268 | 35 / 50 / 50 / 30 |
| PDI4 S2 laplacian  | 40 | 8 | 0.5003 | 30 / 45 / 45 / 35 |
| PDI4 S3 behavioral | 25 | 2 | 0.5631 | 35 / 50 / 50 / 35 |
| PDI4 S3 laplacian  | 40 | 8 | 0.6124 | 30 / 45 / 45 / 30 |

*CV BA = phase-2 CV balanced accuracy at (h=4.5, m=2.0).* When `both` or
`vanilla` fell below `full_i` it is because the retry-lower path fired on an
eigenvalue blowup; the `on`/`off` sub-fits train on less data and generally
stayed at `full_i`.

### Results directory naming

```
results/psid_{mode}_{P}_{S}_nx_{nx}_n{n1}_i{i}{_vanilla}_dbs_{dbs}_200Hz_narrow_band/
```

Each variant dir contains:
- `model_{ts}.pkl`, `model_{ts}_metadata.json`
- `split/{train,val,test}.pkl` (copied from phase-3 training)
- `test/`, `cross_eval/` (phase 4)
- `classification/` (phase 5)
- `config.yaml` — exact training config used

---

## VARMA pipeline

Unified entry point: **`scripts/pipeline_varma.py`**.

VARMA uses only a **subset of channels** selected from the PSID channel
importance scores — the script autodetects the latest PSID full-training
variant and reads the top-N channels by observability/controllability union.

### Phases
1. **Channel selection** — reads PSID `model_*.pkl` for the cell, computes
   top-`TOP_N` (default 5) channels per criterion, takes the union.
   Writes `results/varma_channel_selection_{mode}_{P}_{S}.json`.
   This JSON is also what DPAD consumes when `--use-psid-channels` is set.
2. **Train** — VARMA(p=30, q=1) per DBS condition (`both`, `on`, `off`),
   splits copied from the PSID variant for bit-exact alignment.
3. **Cross-evaluation** — same on/off cross evals as PSID.
4. **Classification** — same feature sources, same (h, m) grid.
5. **Thesis specs/HTML update** — behavioral only (laplacian not wired up).

### Key constants

| Field | Value |
|---|---|
| `P` (AR lags) | 30 |
| `Q` (MA lags) | 1 |
| `LONG_AR_LAGS` | 30 |
| `TRIAL_EDGE_TAPER_SEC` | 0.1 |
| `TOP_N` channels | 5 |
| `DEFAULT_PSID_I_BEHAV` | 50 |
| `DEFAULT_PSID_I_LAP` | 45 |

`autodetect_psid_best` scans `i ∈ [psid_i, 45, 40, 35, 30]` descending and
uses the highest `i` where a real `model_*.pkl` exists — same pattern as
the Phase 5 fallback in PSID.

### Per-cell results

Initial chain run produced the PDI1 S4 cells; the other 6 were redone via
`scripts/rerun_varma_failed.sh` after the autodetect fix.
Log: `logs/pipeline_varma/rerun_varma_failed_20260411_112101.log`.

| Cell | # channels | PSID source | Timestamp (both / on / off) |
|---|---|---|---|
| PDI1 S2 behavioral | 6  | nx=4  n1=2 | 112419 / 112449 / 112506 |
| PDI1 S2 laplacian  | 9  | nx=15 n1=8 | 112557 / 112724 / 112814 |
| PDI1 S4 behavioral | 10 | nx=25 n1=2 | 104843 / 104912 / 104931 |
| PDI1 S4 laplacian  | 10 | nx=25 n1=8 | 105016 / 105123 / 105201 |
| PDI4 S2 behavioral | 8  | nx=15 n1=2 | 112104 / 112133 / 112149 |
| PDI4 S2 laplacian  | 8  | nx=40 n1=8 | 112945 / 113102 / 113140 |
| PDI4 S3 behavioral | 8  | nx=25 n1=2 | 112239 / 112309 / 112328 |
| PDI4 S3 laplacian  | 8  | nx=40 n1=8 | 113254 / 113357 / 113440 |

All timestamps are from 2026-04-11. Variant dirs live under
`results/varma_{mode}_{P}_{S}_p30_q1_top5_dbs_{dbs}_200Hz_narrow_band/`.

> **Note**: the 11:35:45 thesis-HTML regeneration inside Phase 5 failed with
> `ImportError: cannot import name 'LAPLACIAN_SESSIONS'` — the laplacian
> triplet is not wired into `dashboard/thesis/specs.py`. VARMA models
> themselves are fine; only the optional HTML report step trips.

---

## DPAD pipeline

Unified entry point: **`scripts/pipeline_dpad.py`**. **Behavioral only.**

> **TODO (2026-04-17)**: add DPAD laplacian mode. Requires: (a) `--mode
> behavioral|laplacian` flag mirroring `pipeline_psid.py`; (b)
> `DPADConfig.mode_config()` helper for laplacian output channels (LFP
> bands), training subdir `dpad/laplacian_200Hz`, variant prefix
> `dpad_laplacian_*`; (c) retrain Phase 1-3 for 4 sessions x laplacian
> (~48 h); (d) re-run Phase 4 classification (~4 h/cell). Downstream:
> `classification_grouped_bar_laplacian_dpad.png` + ROC laplacian panel B
> will auto-fill on next `generate_classification_figures.py` run once
> `results/classification/dpad_laplacian_*` exists.

### Key constants

| Field | Value |
|---|---|
| Epochs | 3000 |
| Method code | `DPAD_uAKCzCy2HL32U` |
| Checkpoint every | 100 epochs |
| Channels | top-5 union from PSID (via `--use-psid-channels`) |
| `nx / n1` | autodetected from PSID best for the cell |
| `psid_i` | autodetected (same retry-lower fallback scan as VARMA) |
| Train timeout | 24 h per variant |

Variant name: `dpad_behavioral_{P}_{S}_nx_{nx}_n{n1}_e3000_top5_dbs_{dbs}_200Hz_narrow_band`.

### Status (as of 2026-04-15)

Rerun launched 2026-04-11 11:37:22 UTC in screen `dpad_rerun` (bobby) after
the `allX_steps` library crash was worked around in `utils/frameworks.py`
(stop passing `skip_predictions=fast` to `DPADModel.fit`).
Initial train log: `logs/pipeline_dpad/local_dpad_20260411_113722.log`.

**Split across two hosts (2026-04-11 17:59 UTC onwards):**

| Host | Screen | Script | Sessions |
|---|---|---|---|
| bobby (local) | `dpad_rerun` | `scripts/run_dpad_local.sh` | PDI1 S2, PDI1 S4 |
| jacque (10.0.0.2) | `dpad_jacque` | `scripts/run_dpad_jacque_PDI4.sh` | PDI4 S2, PDI4 S3 |

Collision prevention: `pipeline_dpad.py:303` has skip-if-exists (checks
for an existing model artifact in the target variant dir before training).
Jacque's PDI4 dirs were synced back to bobby on 2026-04-15 via
`rsync -avz -e 'ssh -i ~/.ssh/id_ed25519_nopass' jacque@10.0.0.2:~/.../results/`.

### Phase 1+2 per-cell (all 12 variants complete on bobby)

Canonical `(model_ts / test_parquet_ts)` are identical for every variant:

| Cell | nx | n1 | dbs_both | dbs_on | dbs_off |
|---|---|---|---|---|---|
| PDI1 S2 | 4  | 2 | 20260411_113724 | 20260411_142313 | 20260411_164725 |
| PDI1 S4 | 25 | 2 | 20260411_230413 | 20260412_000112 | 20260412_001727 |
| PDI4 S2 | 15 | 2 | 20260411_175909 | 20260411_223341 | 20260412_064157 |
| PDI4 S3 | 25 | 2 | 20260412_102335 | 20260412_134603 | 20260412_165957 |

Phase 2 on jacque originally hit `TEST_TIMEOUT=7200` (2h) on PDI4 S3
(`logs/pipeline_dpad/jacque_dpad_20260411_175906.log` ends with
`subprocess.TimeoutExpired`). After patching `training/test.py` to accept
`--incremental --splits test` (2026-04-14, see
`project_dpad_test_gotchas.md` in Claude memory), test was re-run manually
for the 6 PDI4 variants; current `test/test_results_*.parquet` dirs all
carry mtimes from 2026-04-14.

### Phase 3 complete; Phase 4 redesigned (2026-04-15)

**Phase 3 (cross-condition eval)** completed 2026-04-15 for all 4 cells.
Each cell produced `results/dpad_*_eval_off/` + `results/dpad_*_eval_on/`.

**Phase 4 (classification)** — original design (grid-search h × m, permute on
CV-best) crashed at the `CLS_TIMEOUT=3600s` wall on every cell. Root cause:
DPAD forecast is an RNN rollout per trial per CV fold, not a closed-form
Kalman recursion; even a single (h, m) config takes >1 h at h=0.5, m=0.5.

**Redesign**: `scripts/pipeline_dpad.py` now imports PSID's CV-best (h, m)
per feature source via `_find_psid_best_hm()` and runs DPAD classification
at that **fixed** cell — no grid search on DPAD. Methodologically stronger
because DPAD's permutation test is conditionally valid (PSID chose the
window, DPAD didn't); removes the selective-inference confound that
applies to PSID's own p-values.

`CLS_TIMEOUT` bumped 3600 → 14400 (4 h) as a safety net.

Log: `logs/pipeline_dpad/dpad_phase4_aligned_{ts}.log`.

PDI4 PSID Phase 5 was missing until 2026-04-15; re-ran it via
`scripts/pipeline_psid.py --start-phase 5 --end-phase 5` for PDI4 S2
(`--best-nx 15 --best-n1 2`) and PDI4 S3 (`--best-nx 25 --best-n1 2`).
Completed cleanly at 16:08 UTC; fed the PSID (h, m) picks into DPAD.

PSID CV-best (h, m) per cell (from 2026-04-15 Phase 5):

| Cell | Xp | Xp_1 | Xp_2 | Xp_with_dbs |
|---|---|---|---|---|
| PDI1_S2 | (0.5, 0.5) 0.500 | (0.5, 0.5) 0.500 | (0.5, 0.5) 0.594 | (0.5, 0.5) 1.000 |
| PDI1_S4 | (0.5, 0.5) 0.744 | (1.5, 0.5) 0.606 | (0.5, 0.5) 0.744 | (0.5, 0.5) 1.000 |
| PDI4_S2 | (0.5, 0.5) 0.616 | (1.5, 0.5) 0.514 | (0.5, 0.5) 0.616 | (0.5, 0.5) 1.000 |
| PDI4_S3 | (4.5, 0.5) 0.534 | (0.5, 0.5) 0.515 | (1.5, 2.0) 0.549 | (0.5, 0.5) 1.000 |

`Xp_with_dbs` = 1.0 across the board is expected: the feature concatenates
the DBS label with `Xp`, so label leakage trivially achieves perfect
classification. Report as a sanity-check ceiling, not a model score.

### Stray artifacts (cleaned up)

- `results/dpad_behavioral_PDI4_2_nx_15_n2_e3000_top5_dbs_on_200Hz_narrow_band/`
  had a second model tagged `20260412_023310` from a retrain on jacque;
  the 2026-04-14 test re-run targeted the older `20260411_223341` model,
  so the stray artifact was moved into `.stray_models/` inside the variant
  dir (preserved, not deleted) so `get_latest_model_ts` picks the
  canonical one.
- PDI4 S3 variants all have two model artifacts. The lexicographically
  later one matches the test parquet in every case, so
  `get_latest_model_ts` picks correctly without intervention.

---

## Classification status (2026-04-16)

> **Always check this section first before building thesis figures.**
> Permutation p-values and flipped-classifier results are incomplete until
> all screens below finish. Standard (non-flipped) classification is usable now.

### Architecture recap

- **PSID**: runs full `H_GRID × M_GRID` = 15 combinations, picks CV-best (h, m)
  per feature source, then runs permutation test only at that best point.
- **DPAD**: no grid search — reads PSID's CV-best (h, m) and classifies at that
  fixed point. Methodologically cleaner: PSID chose the window, DPAD did not.
- **Flipped classifier**: loads on/off/both models simultaneously; uses h/m from
  the **flipped PSID** grid (`*_flipped` classification dir) — must exist
  independently, never falls back to the standard h/m.

### Bugs found and fixed 2026-04-16

1. **Sentinel initialization** (`pipeline_psid.py:750`, `pipeline_dpad.py:438`).
   `best_h, best_m, best_ba = -1.0, H_GRID[0], M_GRID[0]` — `best_ba` was
   initialised to `M_GRID[0]=0.5`. Any session where all flipped BAs ≤ 0.5
   (at-chance, e.g. PDI1 S2 behavioral) passed the `if best_ba < 0` guard
   with `best_h=-1.0` intact. This wrote a ghost `h-1.0_m0.5/` dir and passed
   `h=-1.0` (nonsensical) to `compute.py`.
   **Fix**: `best_h, best_m, best_ba = H_GRID[0], M_GRID[0], -1.0`.

2. **`fallback_to_standard` removed from `pipeline_dpad.py`**.
   When PSID flipped classification was absent, DPAD silently used the
   standard h/m for the flipped classifier (wrong feature space).
   **Fix**: removed entirely — missing flipped PSID now raises `FileNotFoundError`.

3. **Flipped classification timeout** (`pipeline_psid.py`, `pipeline_dpad.py`).
   Both pipelines used `timeout=3600` (1 h) for the flipped step. Flipped loads
   3 models simultaneously; for nx≥15 behavioral or nx≥25 laplacian it timed
   out before saving any results. Because `TimeoutExpired` propagated uncaught,
   **the entire Step 2 (standard permutation tests) never ran** for any session.
   **Fix**: `CLS_TIMEOUT_FLIPPED = 28800` (8 h) for all flipped calls in both
   pipelines; flipped moved outside `configs_step1` loop so Step 2 runs even
   if flipped times out in the future.

4. **TF GPU OOM on jacque** (MX150, 2 GB VRAM).
   TF pre-allocates full VRAM at startup → `cudaSetDevice() failed: out of memory`
   on every DPAD classification job on jacque.
   **Fix**: `TF_FORCE_GPU_ALLOW_GROWTH=true` injected into every classification
   subprocess via `run_cmd(extra_env={"TF_FORCE_GPU_ALLOW_GROWTH": "true"})`.
   All DPAD Phase 4 classification moved to bobby (RTX 3050, 4 GB) for safety.

### Ghost dirs cleaned up

| Dir | Problem |
|---|---|
| `psid_behavioral_PDI4_2_*_flipped/20260415_153747` | `h-1.0_m0.5` from sentinel bug |
| `psid_behavioral_PDI1_2_*_flipped/20260416_101806` | `h-1.0_m0.5` from sentinel bug (today's bad re-run) |
| `psid_behavioral_PDI1_4_*_flipped/20260411_122917` | only `h0.5_m0.5` — partial grid, Step 2 aborted |
| `psid_behavioral_PDI4_3_*_flipped/20260415_160135` | only `h0.5_m0.5` — partial grid, Step 2 aborted |
| `psid_laplacian_PDI4_3_*_flipped/20260411_140159` | `h-1.0_m0.5` from sentinel bug |
| `psid_laplacian_PDI1_4_*_flipped/20260411_125010` | partial perm test, re-running |

### Current classification status (as of 2026-04-16 19:30)

#### PSID behavioral — `screen -r psid_cls` (CPU, runs concurrently with DPAD)

Queue: PDI4 S2 → PDI4 S3 → PDI1 S2 → PDI1 S4.
Log: `logs/pipeline_dpad/psid_phase5_rerun_20260416_*.log`

| Cell | Standard grid | Flipped grid | Perms |
|---|---|---|---|
| PDI1 S2 | ✓ 4 preds, 24 pkls, 6 h/m | running (never completed) | queued |
| PDI1 S4 | ✓ 4 preds, 24 pkls, 6 h/m | running (was partial h0.5 only) | queued |
| PDI4 S2 | ✓ 4 preds, 24 pkls, 6 h/m | ✓ 6 h/m complete | running |
| PDI4 S3 | ✓ 4 preds, 24 pkls, 6 h/m | running (was partial h0.5 only) | queued |

#### PSID laplacian — `screen -r psid_lap_cls` (CPU)

Queue: PDI1 S2 → PDI1 S4 → PDI4 S3. PDI4 S2 has no laplacian session.
Log: `logs/pipeline_dpad/psid_lap_phase5_rerun_20260416_*.log`

| Cell | Standard grid | Flipped grid | Perms |
|---|---|---|---|
| PDI1 S2 | ✓ 4 preds, 24 pkls, 6 h/m | running (never existed) | queued |
| PDI1 S4 | ✓ 4 preds, 24 pkls, 6 h/m | ✓ 6 h/m complete (2026-04-11) | running |
| PDI4 S2 | N/A | — | — |
| PDI4 S3 | ✓ 4 preds, 24 pkls, 6 h/m | ✓ 6 h/m complete (2026-04-11) | running |

#### DPAD behavioral — `screen -r dpad_cls` (RTX 3050, GPU, bobby)

Uses PSID-best h/m (fixed point, no grid). Queue: PDI4 S2 flipped → perms →
PDI4 S3 → PDI1 S4 → PDI1 S2.
Log: `logs/pipeline_dpad/all_cls_20260416_100929.log`

| Cell | Preds | Forecast pkls | Flipped | Perms |
|---|---|---|---|---|
| PDI1 S2 | ✓ 4/4 | ✓ 4 (Xp/Xp_1/Xp_2/Xp_with_dbs) | queued | queued |
| PDI1 S4 | ⚠ 3/4 (xp_with_dbs missing) | ⚠ 2/4 | queued | queued |
| PDI4 S2 | ✓ 4/4 | ✓ 4 | **running** (started 16:27) | queued |
| PDI4 S3 | nothing yet | nothing yet | queued | queued |

### What is usable right now

Standard (non-flipped) DBS classification, predictions, and reconstruction
metrics are ready for all 8 PSID cells (behavioral + laplacian) and for
DPAD PDI1 S2 + PDI4 S2. Permutation p-values and flipped-classifier results
are pending but not needed for the core classification accuracy figures.

---

## Chain orchestration

**`scripts/run_chain_all.sh`** — master chain:
- **Phase A (PSID)**: local `PDI1` + jacque (`10.0.0.2`) `PDI4` in parallel.
  Local runner: `scripts/run_psid_local_PDI1.sh`. Jacque runner:
  `scripts/run_psid_jacque_PDI4.sh`.
- **Phase B (rsync)**: pull PDI4 PSID results back from jacque.
- **Phase C (VARMA)**: local, all 4 sessions × (behavioral, laplacian).
- **Phase D (DPAD)**: local, all 4 sessions (behavioral only).

`pipeline_varma.py` Phases C/D were designed to be local-only — DPAD & VARMA
do not fan out to jacque. If you need to move VARMA to jacque later, the
`autodetect_psid_best` scan already handles the i-fallback case.

### Re-run helper scripts

| Script | Purpose |
|---|---|
| `scripts/rerun_varma_failed.sh` | Re-run the 6 VARMA cells that failed with the old autodetect bug (all cells now succeed — kept for reference) |
| `scripts/rerun_psid_classification.sh` | Re-run Phase 5 only across all 8 cells, reading best `(nx, n1)` from an existing chain log. Takes 1 optional arg: chain-log path (default `logs/chain/chain_all_20260410_224412.log`) |

To re-run a single cell's Phase 5:
```bash
python scripts/pipeline_psid.py --participant PDI4 --session 3 --mode laplacian \
    --start-phase 5 --end-phase 5 --best-nx 40 --best-n1 8 --skip-phase-6
```

---

## Bugs fixed during this run

All fixes are on branch `varma-pipeline`.

1. **`utils/classification.py:165` — `block_samples` AttributeError.**
   Hit only the flipped-label classification path inside PSID Phase 5. The
   other three phase-5 sources were unaffected (so Phase 5 was re-run in full
   rather than regenerating the whole chain). Fix: routes `block_samples`
   through the correct keyword arg in `_forecast_latent_trajectory`.

2. **`scripts/pipeline_varma.py:208` — `autodetect_psid_best` hard-coded `i`.**
   Previously returned only `(nx, n1)` using `cfg.psid_i` as a fixed path
   component, even though PSID training routinely falls back to lower `i`
   via the retry-lower path. Fix: scans `[psid_i, 45, 40, 35, 30]`
   descending, returns `(nx, n1, i_found)`, and the caller updates
   `cfg.psid_i` to the found value.

3. **`scripts/pipeline_dpad.py:193` — same autodetect bug.**
   Mirror fix: scans descending, caller updates local `psid_i`.

4. **`scripts/pipeline_psid.py:978-997` — Phase 5 entry hard-coded `full_i`
   when `--start-phase > 3`.** Same retry-lower blindness, but on the
   resume-from-Phase-5 path. Fix: scans `[full_i, 45, 40, 35, 30]`
   descending for each DBS variant independently (since `on`/`off` fits
   often stay at `full_i` while `both`/`vanilla` drop), and populates
   `i_used` with the per-key real values so Phase 5 classification reads
   the correct variant dirs.

5. **`utils/frameworks.py` — DPAD `allX_steps` UnboundLocalError.**
   The library `DPADModel.py:1919` accesses `allX_steps` unconditionally in
   the Cz regression path (`epochs > 0`), but only defines it inside
   `if not skip_predictions:`. Our trainer was passing
   `skip_predictions=config.model.fast`, which triggered the crash on the
   fast path whenever `epochs > 0`. Fix: stopped forwarding
   `skip_predictions` to `DPADModel.fit`. The trainer-level `fast` flag
   still short-circuits post-training Pearson r evaluation (the actual
   speedup we wanted), so there is no perf regression.

## Known latent bugs (not fixed — not blocking)

- **`scripts/pipeline_dpad.py` — single-`psid_i` autodetect.**
  `autodetect_psid_best` picks one `i` (the highest where `dbs_both` has a
  model) and reuses it for all DBS conditions when copying splits via
  `_copy_splits_from_psid`. When `both` fell back via retry-lower but
  `on`/`off`/`vanilla` stayed at `full_i`, DPAD warns
  `PSID split dir not found: ..._i{fallback}_dbs_{on,off,vanilla}/split` and
  regenerates splits from scratch. Since splits are deterministic (fixed
  ratios + `within_session_split=True`) the regenerated splits are
  bit-identical to PSID's, so the warning is cosmetic. Next rerun should
  scan per-DBS-variant `i` like the PSID Phase 5 entry fix.

## 2026-04-23 → 2026-04-26 — mrmr8 family + multi-host DPAD + lap Y→Y fix

Everything above was the 2026-04-10/11 rerun. Since 2026-04-23 the canonical
family for figures is `_mrmr8` — Y input = top-8 ECoG via mRMR-vs-behaviour;
Z output (laplacian mode) = top-8 LFP via mRMR-vs-behaviour; nx/n1 from
`configs/diagnostic/elbow_choices.yaml`. VARMA stays `p=30, q=1, mrmr8`.

### Canonical (nx, n1) per cell

From `configs/diagnostic/elbow_choices.yaml`:

| Cell | behavioral (nx, n1) | laplacian (nx, n1) |
|---|---|---|
| PDI1_S2 | 150, 10 | 75, 10 |
| PDI1_S4 | 100, 10 | 80, 10 |
| PDI4_S2 | 150, 7  | 60, 7  |
| PDI4_S3 | 150, 15 | 55, 10 |

### Bug fixed 2026-04-25 — PSID/DPAD/VARMA laplacian Z output

Old laplacian variants used wrong Z:
- PSID lap fell back to `neural_input` as Z → **Y→Y self-prediction** (`r_Z ≈ 0.99` was decorative noise)
- DPAD/VARMA lap used all 15 LFP bands instead of top-8

Fix: new helper `scripts/_pipeline_common.py:mrmr_top_k_lfp_from_diagnostic`
reads top-K LFP picks from
`results/diagnostic/{P}_{S}_psid_spectra/mrmr_timelagged/laplacian.parquet`.
PSID/DPAD/VARMA now produce identical Z channel sets per cell. Old broken
dirs archived to `results/_archive_lap_pre_topk_lfp_20260425/`.

### Multi-host allocation (2026-04-24 onward)

| Host | Hardware | Owns |
|---|---|---|
| bobby (local) | RTX 3050 Ti 4 GB | PSID + VARMA all cells; DPAD nothing (was killed Apr 26 to avoid duplication with rafael) |
| rafael (Lambda Cloud) | A10 24 GB, ~$1.20/hr | All DPAD training (12 behavioral + 12 laplacian) + cross_cond eval |
| bobby puller | `~/pull_rafael_dpad.sh` | rsync `dpad_*_200Hz_narrow_band` rafael→bobby every 2 min |

DPAD lap order on rafael (`scripts/run_dpad_parallel_pt2.sh`): PDI4 S2/S3
behavioral → PDI1 S2/S4 lap → PDI4 S2/S3 lap.

### 2026-04-26 lap retrain (bobby)

PSID + VARMA lap retrained against the new Z channels. Chain script:
`scripts/run_lap_retrain_20260425.sh`. PSID nx/n1 from elbow yaml; per-cell
logs at `logs/lap_retrain_20260425/`.

### Status (as of 2026-04-26)

| Cell × mode | PSID | VARMA | DPAD |
|---|---|---|---|
| PDI1_S2 behavioral | done (Apr 23) | done (Apr 24) | done (Apr 24) |
| PDI1_S2 laplacian  | done (Apr 25) | done (Apr 25) | done (Apr 25, rafael) |
| PDI1_S4 behavioral | done (Apr 24) | done (Apr 24) | done (Apr 24) |
| PDI1_S4 laplacian  | done (Apr 25) | done (Apr 25) | done (Apr 25, rafael) |
| PDI4_S2 behavioral | done (Apr 24) | done (Apr 24) | done (Apr 24, rafael) |
| PDI4_S2 laplacian  | done (Apr 25) | done (Apr 25) | done (Apr 26, rafael) |
| PDI4_S3 behavioral | done (Apr 24) | done (Apr 24) | done (Apr 25, rafael) |
| PDI4_S3 laplacian  | done (Apr 25) | done (Apr 25) | **training in flight on rafael** (started 08:47 UTC) |

Cross_cond eval dirs (`*_eval_on`, `*_eval_off`) on rafael, syncing.
Inference (`training.test`) all rc=0 so far via `auto_infer_watcher.sh`
(CPU mode v3, MAX_PAR=3) — kept on CPU since it's keeping pace with training.

### Bookkeeping

- `notebooks/thesis_triplets.csv` — canonical lookup. Rows dated **2026-04-26**
  are current; older rows kept as history. Auto-refreshed by
  `scripts/refresh_thesis_triplets.py` (scans `results/`, picks highest-i
  PSID dir per cell, reads `model_*.pkl` mtimes for run timestamps,
  replaces same-(cell, side) rows within 2 days).
- Three `side` values per cell:
  - `behavioral` — Y = 8 ECoG mRMR-top, Z = behavior
  - `laplacian` — Y = 8 ECoG mRMR-top, Z = 8 LFP mRMR-top
  - `y2y` — **appendix-only PSID baseline.** Y = Z = 8 ECoG mRMR-top
    (preserved from the pre-fix lap variants where Z was incorrectly set to
    Y). No LFP/laplacian channels involved at all — ECoG self-prediction.
    DPAD/VARMA never had a Y→Y variant, so those columns are blank for y2y
    rows. Backed by symlinks `results/psid_y2y_*` →
    `results/_archive_lap_pre_topk_lfp_20260425/psid_laplacian_*`.
    Useful for showing that PSID forecast is weak even when Z=Y (i.e. the
    `Cz` collapse is structural, not LFP-specific).
- `cross_cond_summary_20260424.csv` — also stale once DPAD lap finishes;
  needs regenerating against the post-fix lap dirs.

### Notification watchers (bobby, all detached, PPID=1)

| Script | Fires when | Action |
|---|---|---|
| `scripts/watch_psid_varma_done.sh` | bobby PSID+VARMA chain supervisor PID exits (PID read from `/home/bobby/.notify_logs/lap_retrain_supervisor.pid`) | email + drop flag + launch `run_lap_dpad_bobby.sh` (currently moot — DPAD on rafael now) |
| `scripts/watch_dpad_behavioral_done.sh` | rafael has all 12 behavioral mrmr8 train dirs with `model_*.pkl` AND no behavioral procs alive for 3 polls | email + flag |
| `scripts/watch_rafael_dpad_train_done.sh` | rafael's `run_dpad_parallel_pt2.sh` PID exits AND no `run_dpad_phase --phases train` procs | sleep 6 min for puller, run `refresh_thesis_triplets.py`, email with diff |
| `scripts/watch_lap_ready.sh` | both PSID+VARMA + DPAD-behavioral flags present | email "lap launch GO" (informational) |

Email plumbing: `scripts/notify_email.py` (Gmail SMTP via app password in
`~/.email_notify_creds`, chmod 600). Logs at `~/.notify_logs/`.

### Bugs fixed 2026-04-25/26

5. **`scripts/pipeline_psid.py`** — laplacian Z silently used `neural_input`
   (Y→Y self-prediction). Fixed to read top-K LFP via new
   `mrmr_top_k_lfp_from_diagnostic` helper.
6. **`scripts/pipeline_dpad.py`** — laplacian Z used all 15 LFP bands. Same
   fix pattern.
7. **`scripts/pipeline_varma.py`** — same as DPAD.
8. **`utils/frameworks.py:DPADWrapper.predict`** — module-level
   `_DPADFWK_FORECAST_CACHE` not invalidated when `set_steps_ahead([1])` is
   reset, causing `vstack` shape mismatch on subsequent `forecast(m=N)` calls.
   Fix: pop cache entry inside `predict()`.
9. **`scripts/run_cross_condition_eval.py:27`** — imported
   `dashboard.subtabs.helpers.save_split_results` but `dashboard/` was moved
   to `archives/dashboard_streamlit/`. Fix: inlined the 5-line helper.
10. **`scripts/run_cross_condition_eval.py:99`** — called
    `tester.framework.model.validate_forecast(...)` (renamed during refactor
    to `BaseFramework._evaluate_forecast`). Fix: updated call site.

## Historical note

The previous version of this document only covered PDI4 S3 under the old
per-session `pipeline_psid_PDI4_S3.py` script and an older grid
`nx=[2,4,8,15,25,30] × n1=[2,4,6]`, max `i=30`. Those numbers are no longer
comparable to the current unified pipeline — the grid, `full_i`, forecast
scoring metric, and channel selection method all changed. See
`feedback_use_pipeline_script.md` in the Claude memory for why we now score
by balanced accuracy at (h=4.5, m=2.0) rather than Pearson r.
