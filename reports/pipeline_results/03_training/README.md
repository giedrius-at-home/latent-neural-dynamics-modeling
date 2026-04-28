# Stage 3 — Full training (FB-enabled PSID + DPAD + VARMA)

## Job

Train the production models per (cell, side, model_family, dbs_condition):

- **PSID** — subspace identification at the (nx, n1) elbow on the mRMR-8 channel set, with Sani & Shanechi 2025 forward-backward smoother + filter-aware forecast and A-eigenvalue clipping (`max_eigenvalue = 0.9999`). Primary model.
- **DPAD** — neural-network alternative on the same 8 channels (behavioral mode only; no laplacian).
- **VARMA** — classical multivariate AR reference on the same 8 channels.

## Producer

Unified pipeline:

```bash
python scripts/pipeline_psid.py \
    --participant PDI4 --session 2 --mode behavioral \
    --best-nx 50 --best-n1 10
```

Phases inside `pipeline_psid.py`:
- **Phase 3** — train 4 variants in parallel: `both`, `on`, `off`, `vanilla_both`
- **Phase 4** — cross-condition eval (`on → off`, `off → on`)
- **Phase 5** — classification grid + permutation (stage 5 below)
- **Phase 6** — `specs.py` update for the thesis triplet registry

Phases 1 and 2 (old grid-search over nx × n1) were removed — elbow choice from
stage 1 is now the source of truth.

DPAD + VARMA are driven by their own pipelines (`pipeline_dpad.py`, VARMA via
overnight scripts). All three families reuse the same PSID `split/` to
guarantee identical train/val/test trials per cell.

## Registry — the (variant, run_ts) triplets

The authoritative triplet registry (consumed by every thesis notebook) is
[`notebooks/thesis_sec2_common.py`](../../../notebooks/thesis_sec2_common.py).

### Behavioral (ECoG) triplets

| Cell | PSID variant, run_ts | DPAD run_ts | VARMA run_ts |
|---|---|---|---|
| PDI1_S2 | `psid_behavioral_PDI1_2_nx_55_n15_i100_dbs_both_200Hz_narrow_band` @ 20260421_222439 | 20260419_010200 | 20260420_133800 |
| PDI1_S4 | `psid_behavioral_PDI1_4_nx_50_n10_i100_dbs_both_200Hz_narrow_band` @ 20260422_000702 | 20260419_093953 | 20260420_134630 |
| PDI4_S2 | `psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band` @ 20260421_202056 | 20260418_225805 | 20260420_113757 |
| PDI4_S3 | `psid_behavioral_PDI4_3_nx_50_n10_i100_dbs_both_200Hz_narrow_band` @ 20260421_202721 | 20260419_074635 | 20260420_114212 |

### Laplacian (LFP) triplets (no DPAD)

| Cell | PSID variant, run_ts | VARMA run_ts |
|---|---|---|
| PDI1_S2 | `psid_laplacian_PDI1_2_nx_55_n15_i100_dbs_both_200Hz_narrow_band` @ 20260422_001503 | 20260420_134013 |
| PDI1_S4 | `psid_laplacian_PDI1_4_nx_50_n10_i100_dbs_both_200Hz_narrow_band` @ 20260422_003140 | 20260420_134815 |
| PDI4_S2 | `psid_laplacian_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band` @ 20260421_203357 | 20260420_113921 |
| PDI4_S3 | `psid_laplacian_PDI4_3_nx_50_n10_i100_dbs_both_200Hz_narrow_band` @ 20260421_204455 | 20260420_114334 |

Single-condition (on / off) timestamps are also in `thesis_sec2_common.py` on
the same `AlignedTriplet` entries (`psid_run_ts_off`, `psid_run_ts_on`, etc.).

## Model-type matrix — what's on disk

| Family × mode × condition | Models trained | Notes |
|---|---|---|
| PSID behavioral ×4 cells × {both, on, off} | ✓ 12/12 | FB smoother + A-clip, i=100 |
| PSID laplacian ×4 cells × {both, on, off} | ✓ 12/12 | " |
| PSID vanilla (both only) ×4 behavioral + ×4 laplacian | ✓ 8/8 | no FB, no A-clip — channel-selection baseline |
| DPAD behavioral ×4 cells × {both, on, off} | ✓ 12/12 | PDI1 trained on jacque |
| DPAD laplacian | — | intentional: no DPAD laplacian |
| VARMA behavioral ×4 × {both, on, off} | ✓ 12/12 | includes eval_on / eval_off variants |
| VARMA laplacian ×4 × {both, on, off} | ✓ 12/12 | " |

Each `results/<variant>/` dir contains:
```
model_<ts>.pkl              # the serialised model
model_<ts>_metadata.json    # model type, nx, n1, max_eigenvalue, fb_smoother
split/                      # train / val / test parquets
logs/                       # train/test console logs + markdown
test/                       # test-split forecast parquets (stage 4)
train/, val/                # reconstruction parquets
```

## Why the split is 60/15/45 not 60/10/30

See [`../../DATA_EFFICIENCY_ANALYSIS.md`](../../DATA_EFFICIENCY_ANALYSIS.md).
The original 60/10/30 gave single-class val on 2 of 4 cells. 60/15/45 with
`within_session_split=True` yields every cell having both DBS classes in
every split.

## Feeds into stage 4

Model pkls → test evaluation (`python -m training.test --config <yaml>`),
which writes `test/test_results_<ts>.parquet/`. See stage 4.

## Symlinks

- `results_root/` → `../../../results/` (all variant dirs live here)
- `training_configs/` → `../../../training/setups/`
- `sec2_common/` → `../../../notebooks/thesis_sec2_common.py` (the registry file itself)

## FB ablation (side-story)

A 4-condition comparison — RTS-no-clip vs FB-no-clip vs RTS-clip vs FB-clip —
is in [`../../../fb_ablation_results.md`](../../../fb_ablation_results.md).
Shows FB helps forecast classification by ≈ +0.02 mean BA across cells.
