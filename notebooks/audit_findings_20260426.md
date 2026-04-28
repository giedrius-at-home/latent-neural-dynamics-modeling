# Notebook audit — 2026-04-26

Findings against canonical Apr 26 mrmr8 family.

## Refresh status

- `notebooks/thesis_triplets.csv` — refreshed; 8 fresh rows (2026-04-26) appended; old Apr 22 rows preserved as history.
- `cross_cond_summary_20260424.csv` — not consumed by any notebook (only `pipeline_runs.md` references it). Cosmetic; defer regen.

## Static scan — sec2c, sec2d, sec5

| Notebook | Stale refs? | Notes |
|---|---|---|
| `thesis_sec2c_neural_recon_group.py` | clean | Reads via `thesis_loaders` → triplets CSV. Should pick up Apr 26 automatically. |
| `thesis_sec2d_neural_forecast_group.py` | clean | Same. |
| `thesis_sec5_classification.py` | **32 hardcoded paths** | All point at Apr 22 family (`nx_55_n15` / `nx_50_n10`, ts `20260421_*` / `20260422_*`). Needs refactor to dynamic loader OR manual update. |

## Classification inventory (Apr 26 family)

| Framework × mode | Cells classified | Notes |
|---|---|---|
| PSID behavioral | 4/4 | ts `20260423_233019..20260424_051052`, 4 features each |
| PSID laplacian  | 4/4 | multiple ts per cell (post-fix re-runs); pick latest |
| DPAD behavioral | **0/4** | Apr 22 family classified (old `nx_55/50` mrmr8). Apr 26 nx_150/100/150/150 never classified. |
| DPAD laplacian  | **0/4** | PDI4_S3 lap still training; rest done but Phase 4 never run. |
| VARMA behavioral| **0/4** | VARMA classification never run for any family. |
| VARMA laplacian | **0/4** | Same. |

## Strategic blocker for sec5

`sec5` currently expects 4-feature (xp/xp_1/xp_2/xp_with_dbs) classification
results from PSID + DPAD + VARMA. Apr 26 has only PSID. To salvage sec5
faithfully, three options:

1. **PSID-only sec5** — drop DPAD/VARMA panels. Honest about scope.
2. **Mixed-family sec5** — PSID Apr 26 + DPAD Apr 22 (still mrmr8 but with the
   pre-elbow nx). Note in caption. Skip VARMA.
3. **Run Phase 5 only** — no retraining; just classification on existing
   DPAD-Apr26 + VARMA-Apr26 models. Cost: hours, not days. PSID's CV-best
   `(h, m)` is reused per `pipeline_dpad.py` redesign so it's deterministic.

Strict boundary: Giedrius said "no more experimenting". Phase 5 = bookkeeping
(no model fitting), so debatable. Decide before sec5 work begins.

## VARMA test-parquet skew (PDI4_S2/S3 behavioral)

`scripts/refresh_thesis_triplets.py:latest_model_ts` was patched to require
that the picked model_TS.pkl also has a corresponding
`test/test_results_TS.parquet`. Without the patch, refresh selected
2026-04-24 215302 VARMA models for PDI4_S2/S3 behavioral whose test phase
never ran — sec2c crashed when loaders tried the missing parquet.

Tradeoff: triplets now point at the next-newer VARMA model (Apr 24 ~110000)
that *was* tested, ~10h older. Acceptable bookkeeping. Re-running
`training.test` for the 6 stale VARMA models would replace the parquets if a
future pipeline pass requires the latest weights.

## Other sections

Not yet audited statically — sec1, sec2a, sec2b, sec2e, sec2_model_validation,
sec6, sec7, sec8. Deferring per Q3-G scope.
