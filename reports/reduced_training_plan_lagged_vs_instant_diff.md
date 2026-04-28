# Channel selection: instantaneous vs time-lagged mRMR — what changes?

**Source:** `scripts/compare_mrmr_timelagged.py` (run 2026-04-22)
**Compared against:** `reports/reduced_training_plan.md` (production, instantaneous Pearson)
**See also:** `notebooks/thesis_sec2a_diagnostics.py` — Fig 44 (instant) vs Fig 47 (lagged + bootstrap).

## Why we care

The production selection (`scripts/generate_reduced_training_configs.py`) picks
channels by instantaneous Pearson. Cross-correlation between ECoG and behaviour
typically peaks at ~100-200 ms lag, not at lag 0, so a lag-sensitive metric
should recover theta/beta channels that track movement with a physiologically
plausible delay — and instantaneous Pearson systematically under-weights them.

The time-lagged variant uses per-trial mean-of-top-3 |r| across ±250 ms lags
(21 lag values), sample-weight-averaged across trials, with 5-fold bootstrap
stability at 80% trial-level subsample. See notebook Fig 47 methods text for
the full definition; bugs fixed in the upgraded script are documented inline
in `compare_mrmr_timelagged.py`.

## Laplacian selection is unchanged

| Session | Instant top-8 ≡ Lagged top-8 | Bootstrap core |
|---|---|---|
| PDI1_S2 | ✓ identical | 8/8 |
| PDI1_S4 | ✓ identical | 8/8 |
| PDI4_S2 | ✓ identical | 8/8 |
| PDI4_S3 | ✓ identical | 8/8 |

LFP has 15 candidate features × 1 channel. mRMR picks are dominated by the
relevance ranking; redundancy matters less. The metric choice is academic here,
and bootstrap is 8/8 on every session — selection is locked in regardless.

## ECoG selection changes substantially

| Session | Picks shared | New in lagged | Dropped from instant | ECoG bootstrap core |
|---|---|---|---|---|
| PDI1_S2 | 5 / 8 | 3 | 3 | 3/8 |
| PDI1_S4 | **2 / 8** | 6 | 6 | 2/8 |
| PDI4_S2 | **2 / 8** | 6 | 6 | 3/8 |
| PDI4_S3 | 4 / 8 | 4 | 4 | 6/8 |
| **Average** | **3.2 / 8 (41%)** | **4.8** | **4.8** | 3.5/8 |

On average, **4.8 of 8 ECoG channels per session change** when moving from
instantaneous to time-lagged selection. Only 41% of picks are shared.

### What's entering: low-frequency (theta) ECoG picks

| Session | Notable channels introduced by lagged metric |
|---|---|
| PDI1_S2 | `E2_theta_4_8`, `E3_gamma_45_50`, `E2_gamma_35_40` |
| PDI1_S4 | `E2_theta_4_8`, `E3_beta_27_30`, `E1_gamma_30_35`, `E3_alpha_8_12`, `E3_beta_17_22`, `E4_gamma_75_80` |
| PDI4_S2 | `E3_theta_4_8`, `E2_beta_27_30`, `E3_gamma_30_35`, `E3_gamma_75_80`, `E1_beta_12_17`, `E1_gamma_50_55` |
| PDI4_S3 | `E2_theta_4_8`, `E2_beta_22_27`, `E3_beta_12_17`, `E2_gamma_40_45` |

A **theta channel** (`E{2,3}_theta_4_8`) appears as new in **all 4 sessions**.
Theta-band cortical activity couples to slow kinematic tracing with a
~100-200 ms lag — precisely the kind of signal the instantaneous metric
misses and the lagged metric catches. This is the most defensible biological
signal the metric change recovers.

## Bootstrap stability caveat

Only 2-6 of 8 ECoG picks per session (avg 3.5/8) are "core" — selected in
every bootstrap fold. So a chunk of the 59% difference between metrics is
drawn from the unstable tail, where rank ordering flips under resampling
regardless of metric. The top 2-3 channels tend to be stable under both
metrics; the variability is in ranks 4-8.

## Retraining implications (Scope B)

Switching production to the lagged selection requires:

1. Patch `scripts/generate_reduced_training_configs.py` to read
   `mrmr_timelagged/{family}.parquet` instead of running `mrmr_select` on
   `correlation/{family}.parquet`. Low-risk edit.
2. Regenerate **training YAMLs** for 4 sessions × 2 families ×
   3 DBS conditions × 3 model families = 72 YAMLs. Changes the
   `data.channels.neural_input` list; other fields unchanged.
3. Retrain **PSID × 24 runs** (4 sessions × 2 families × 3 DBS conditions).
   Budget: full pipeline phases for each cell. Multi-hour but parallelizable.
4. Retrain **DPAD × 12 runs** (ECoG only; 4 × 3). Slower than PSID.
5. Retrain **VARMA × 12 runs**. Fast.
6. Rerun stages 4-6 (evaluation, classification, figure rebuilds) from
   `reports/pipeline_results/`.

Total realistic cost: **overnight run** assuming jobs parallelize across
PSID / DPAD (jacque) / VARMA concurrently. Existing `pipeline_psid.py` and
related scripts already support `--start-phase`/`--end-phase` so only
stage 3 onwards need rerunning.

## Recommendation for the thesis

Either choice is defensible if the methods section is honest. Two framings:

- **Stay with instantaneous** — write that "channel selection was robust
  across metrics for LFP, and ~3 of 8 ECoG picks per session were stable
  under bootstrap; the time-lagged variant introduces physiologically
  plausible theta channels but does not materially change downstream model
  performance [cite comparison]." Requires adding the comparison to the
  methods section but no retraining.
- **Switch to lagged** — the stricter choice. Gives you "channels were
  selected via mRMR with per-trial top-3 lag mean relevance and 5-fold
  bootstrap stability" which is the publication-grade sentence. Costs the
  overnight retrain.

Both are defensible; the second is more defensible.
