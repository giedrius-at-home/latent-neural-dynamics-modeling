# DPAD multi-host run — session summary

## Goal
Train all 24 DPAD variants (4 sessions × 2 modes × 3 dbs conditions) and run inference, exploiting parallelism between a paid GPU (rafael) and a local GPU (bobby/neuro).

## Hosts and roles
| Host | Hardware | Role |
|---|---|---|
| **rafael** (Lambda Cloud) | NVIDIA A10 24 GB, 30 vCPU, 222 GB RAM, ~$1.20/hr | Heavy GPU training (large nx ECoG variants). |
| **bobby/neuro** (local) | NVIDIA RTX 3050 Ti Laptop 4 GB | Smaller laplacian variants; PSID/VARMA work. |
| **bobby puller** | bash on neuro | Polls rafael every 2 min, rsyncs every `dpad_*` variant dir back to neuro incrementally (model_*.pkl, training_history.json, splits, test parquets). |

## Setup actions taken on rafael
- Installed `nvidia-driver-580-open` (image was bare).
- Miniconda + recreated `neuro` env from bobby's full env export. Pip install with `--no-deps -r pip-reqs.txt --extra-index-url https://download.pytorch.org/whl/cpu` to handle `torch==2.9.1+cpu`.
- Wrote `~/miniconda3/envs/neuro/etc/conda/activate.d/tf_cuda_libs.sh` — adds the pip nvidia-cu12 packages to LD_LIBRARY_PATH so TF 2.15 finds CUDA. Verified TF sees A10 + DPAD/PSID import.
- SSH: bobby↔rafael via `id_rafael` keypair (no passphrase). Laptop direct: added laptop's pubkey to bobby's authorized_keys; `Host neuro` alias in `~/.ssh/config`.
- Synced from bobby: refactored `varma-pipeline` branch code (~25 MB), `resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band/` (4.2 GB), targeted PSID `split/*.parquet` per variant (~3.3 GB), LFP diagnostic parquets per session.

## Bugs found and fixed
1. **`utils/frameworks.py:DPADWrapper.predict`** — resets `set_steps_ahead([1])` and `set_multi_step_with_data_gen(False)` every call, but didn't invalidate the module-level forecast cache `_DPADFWK_FORECAST_CACHE`. Subsequent `forecast(m=N)` calls assumed N-step setup but model was actually 1-step → `predict()` returned 3 outputs instead of 3N → `_stack_last(preds[:m])` failed with `vstack` shape mismatch (size 2 vs 8).
   - **Fix**: pop cache entry inside `predict()` after the reset.
2. **`scripts/run_cross_condition_eval.py`** — line 27 imported `from dashboard.subtabs.helpers import save_split_results` but `dashboard/` was moved to `archives/dashboard_streamlit/`.
   - **Fix**: inlined the trivial 5-line `save_split_results` (pickle.dump). Dropped the dashboard import.
3. **`scripts/run_cross_condition_eval.py`** — line 99 called `tester.framework.model.validate_forecast(...)`. Method renamed/moved during refactor to `BaseFramework._evaluate_forecast` in `utils/frameworks.py:44`.
   - **Fix**: change call to `tester.framework._evaluate_forecast(...)`.
4. **DPAD laplacian outputs were 15 LFP bands instead of top-8 by mRMR.** User updated `pipeline_dpad.py:DPADConfig` + added `_pipeline_common.mrmr_top_k_lfp_from_diagnostic` reading `results/diagnostic/{P}_{S}_psid_spectra/mrmr_timelagged/laplacian.parquet`. Synced to rafael.

## Pipelines built on rafael
- **`scripts/run_dpad_phase.py`** — wrapper that monkey-patches `DPADPipeline.PHASES` so we can run individual phases (`--phases train` for GPU-only, `--phases test` for CPU-only, etc.).
- **`scripts/run_dpad_parallel_pt2.sh`** — supervisor that does serial GPU training for the rafael-allocated session×mode list.
- **`scripts/auto_infer_watcher.sh`** (v3, MAX_PAR=3) — polls every 60 s for finished `model_*.pkl` per variant, launches `python -m training.test` on CPU with `CUDA_VISIBLE_DEVICES=""` + 10 OMP threads. Capped at 3 concurrent. Tracks launched variants in `/tmp/auto_infer_launched.txt`.

## Allocation tracking
| Host | Variants assigned | Status |
|---|---|---|
| Rafael | All 12 ecog variants + all 6 PDI4 lapl + cross_cond on PDI1/PDI4_S2 | 10 ecog trained, PDI4_S3 ecog dbs_on training, dbs_off pending. PDI4 lapl pending. |
| Bobby (was running) | PDI1 lapl S2 + S4 (6 variants total) | First run (15 LFP outputs, before fix) killed. Second run (8-LFP fix verified, dbs_both started) killed by user to redo PSID+VARMA. |

## Cross_cond status (rafael)
6 of 8 cross_cond eval dirs done — all PDI1 (S2+S4) + PDI4_S2 (both directions). Pending PDI4_S3 (after its 3 trains finish) and all PDI4 lapl + PDI1 lapl.

## Classify phase
**Not started for any session.** `phase_classify` reads `results/classification/{psid_variant}/{timestamp}/` for PSID's best (h, m). That directory hasn't been synced from bobby. Will need to sync after PSID classification has been done on bobby (and PSID lapl re-done with corrected outputs).

## Pending decisions / actions
- User redoing PSID + VARMA on bobby with corrected laplacian outputs (8 LFP bands instead of Y→Y).
- After PSID lapl re-runs: re-sync the new PSID `split/*.parquet` for laplacian variants to rafael (current splits were generated under the old Y→Y target structure but the parquets themselves contain all 89 columns, so should still work — verify before assuming).
- Bobby's PDI1 lapl DPAD training to be relaunched once PSID is fixed.
- Eventual GPU-side inference for any remaining variants once rafael training queue empties (faster than CPU).

## Bobby puller
Still running on neuro (`~/pull_rafael_dpad.sh`, in `~/.sync_logs/rafael_dpad_pull.log`). Idempotent rsync every 2 min — safe to leave indefinitely. Catches both train-done and inference-done artifacts.

## Cost so far
~26 hours of rafael runtime ≈ $31 (out of estimated ~$48-60 for full 24-train).
