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

### Status

Rerun launched 2026-04-11 11:37:22 UTC in screen `dpad_rerun` (bobby) after
the `allX_steps` library crash was worked around in `utils/frameworks.py`
(stop passing `skip_predictions=fast` to `DPADModel.fit`).
Log: `logs/pipeline_dpad/local_dpad_20260411_113722.log`.

**Split across two hosts (2026-04-11 17:59 UTC onwards):**

| Host | Screen | Script | Sessions |
|---|---|---|---|
| bobby (local) | `dpad_rerun` | `scripts/run_dpad_local.sh` | PDI1 S2, PDI1 S4, PDI4 S2, PDI4 S3 |
| jacque (10.0.0.2) | `dpad_jacque` | `scripts/run_dpad_jacque_PDI4.sh` | PDI4 S2, PDI4 S3 |

Collision prevention: `pipeline_dpad.py:303` has skip-if-exists (checks
for `model_*.pkl` in the target variant dir before training). A sync
watcher on bobby — screen `dpad_sync`, script
`scripts/dpad_jacque_sync_watcher.sh` — rsyncs jacque's
`results/dpad_behavioral_PDI4_*` dirs back to local every 15 min and
exits once jacque logs `ALL JACQUE DPAD RUNS COMPLETE`. When the local
runner eventually reaches PDI4 S2/S3, jacque's models are already visible
and local skips.

PDI1 S2 `dbs_both` took ~2h 46m end-to-end (11:37 → 14:23, stages:
RNN1 → Cy1 → RNN2 → Cz2 → persistence). The full rerun is expected to
run for ~1–2 days; monitor via TensorBoard event files under each
variant's `logs/` subdir rather than the main script log (keras progress
bars use `\r` and don't flush to `tee`).

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

## Historical note

The previous version of this document only covered PDI4 S3 under the old
per-session `pipeline_psid_PDI4_S3.py` script and an older grid
`nx=[2,4,8,15,25,30] × n1=[2,4,6]`, max `i=30`. Those numbers are no longer
comparable to the current unified pipeline — the grid, `full_i`, forecast
scoring metric, and channel selection method all changed. See
`feedback_use_pipeline_script.md` in the Claude memory for why we now score
by balanced accuracy at (h=4.5, m=2.0) rather than Pearson r.
