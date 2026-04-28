# DPAD training — run reference (post-Scope-A, train-only lagged-mRMR features)

Generated 2026-04-22. DPAD is deferred until after the PSID variant-selection pass
(PSID + VARMA launched first, DPAD later because each DPAD run is ~3 h).

## Configs (24 total — 12 behavioral + 12 laplacian)

4 sessions × 2 families (behavioral ECoG → kinematics; laplacian LFP → LFP
autoencoder, new — never trained before) × 3 DBS conditions.

The laplacian DPAD YAMLs are adapted from the behavioral template with
`output_type: neural` and `output` = same laplacian 8-feature list as
`neural_input` (matches PSID-laplacian convention).

Each YAML uses canonical `(nx, n1)` matching PSID: **PDI1_S2 = (55, 15)**, others = **(50, 10)**.
Features come from `results/diagnostic/{pid}_{sess}_psid_spectra/mrmr_timelagged/ecog.parquet` (train-only lagged-mRMR top-8).

### Behavioral (ECoG → kinematics)

| Session | DBS | YAML |
|---|---|---|
| PDI1_S2 | both | `training/setups/dpad/narrow_band_200Hz/both/dpad_behavioral_PDI1_2_nx_55_n15_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml` |
| PDI1_S2 | off  | `training/setups/dpad/narrow_band_200Hz/off/dpad_behavioral_PDI1_2_nx_55_n15_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml`  |
| PDI1_S2 | on   | `training/setups/dpad/narrow_band_200Hz/on/dpad_behavioral_PDI1_2_nx_55_n15_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml`   |
| PDI1_S4 | both | `training/setups/dpad/narrow_band_200Hz/both/dpad_behavioral_PDI1_4_nx_50_n10_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml` |
| PDI1_S4 | off  | `training/setups/dpad/narrow_band_200Hz/off/dpad_behavioral_PDI1_4_nx_50_n10_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml`  |
| PDI1_S4 | on   | `training/setups/dpad/narrow_band_200Hz/on/dpad_behavioral_PDI1_4_nx_50_n10_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml`   |
| PDI4_S2 | both | `training/setups/dpad/narrow_band_200Hz/both/dpad_behavioral_PDI4_2_nx_50_n10_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml` |
| PDI4_S2 | off  | `training/setups/dpad/narrow_band_200Hz/off/dpad_behavioral_PDI4_2_nx_50_n10_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml`  |
| PDI4_S2 | on   | `training/setups/dpad/narrow_band_200Hz/on/dpad_behavioral_PDI4_2_nx_50_n10_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml`   |
| PDI4_S3 | both | `training/setups/dpad/narrow_band_200Hz/both/dpad_behavioral_PDI4_3_nx_50_n10_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml` |
| PDI4_S3 | off  | `training/setups/dpad/narrow_band_200Hz/off/dpad_behavioral_PDI4_3_nx_50_n10_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml`  |
| PDI4_S3 | on   | `training/setups/dpad/narrow_band_200Hz/on/dpad_behavioral_PDI4_3_nx_50_n10_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml`   |

### Laplacian (LFP autoencoder — new, never trained before)

| Session | DBS | YAML |
|---|---|---|
| PDI1_S2 | both | `training/setups/dpad/laplacian_200Hz/both/dpad_laplacian_PDI1_2_nx_55_n15_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml` |
| PDI1_S2 | off  | `training/setups/dpad/laplacian_200Hz/off/dpad_laplacian_PDI1_2_nx_55_n15_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml`  |
| PDI1_S2 | on   | `training/setups/dpad/laplacian_200Hz/on/dpad_laplacian_PDI1_2_nx_55_n15_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml`   |
| PDI1_S4 | both | `training/setups/dpad/laplacian_200Hz/both/dpad_laplacian_PDI1_4_nx_50_n10_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml` |
| PDI1_S4 | off  | `training/setups/dpad/laplacian_200Hz/off/dpad_laplacian_PDI1_4_nx_50_n10_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml`  |
| PDI1_S4 | on   | `training/setups/dpad/laplacian_200Hz/on/dpad_laplacian_PDI1_4_nx_50_n10_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml`   |
| PDI4_S2 | both | `training/setups/dpad/laplacian_200Hz/both/dpad_laplacian_PDI4_2_nx_50_n10_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml` |
| PDI4_S2 | off  | `training/setups/dpad/laplacian_200Hz/off/dpad_laplacian_PDI4_2_nx_50_n10_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml`  |
| PDI4_S2 | on   | `training/setups/dpad/laplacian_200Hz/on/dpad_laplacian_PDI4_2_nx_50_n10_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml`   |
| PDI4_S3 | both | `training/setups/dpad/laplacian_200Hz/both/dpad_laplacian_PDI4_3_nx_50_n10_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml` |
| PDI4_S3 | off  | `training/setups/dpad/laplacian_200Hz/off/dpad_laplacian_PDI4_3_nx_50_n10_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml`  |
| PDI4_S3 | on   | `training/setups/dpad/laplacian_200Hz/on/dpad_laplacian_PDI4_3_nx_50_n10_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml`   |

All 12 YAMLs are emitted by `scripts/gen_variants_from_lagged_mrmr.py`. Re-run that
script to regenerate if features change.

## Single-run command

```bash
cd /home/bobby/repos/latent-neural-dynamics-modeling
python -m training.train --config training/setups/dpad/.../<config>.yaml
```

Runs write to `results/<config-stem>/`:
- `model_<ts>.pkl` — trained DPAD
- `split/{train,val,test}.parquet` — chronological 50/12.5/37.5 split
- `test/test_results_<ts>.parquet` — per-trial predictions (Z = behavior, Y = ECoG)
- `logs/train_<ts>.md` — structured log with per-epoch metrics

## Runtime

Per-run on 8-core machine (reference: PDI4_S3 dbs_on on bobby took 3h15m):
- 3000 epochs × ~4 s/epoch ≈ **3–3.5 h per run**
- 24 runs × 3 h = **72 CPU-hours serial**

## Parallelization plan (overnight)

Bobby (8 cores) + jacque (8 cores). Each runs 2 DPAD concurrent (4 cores each BLAS):
- Each machine: 12 DPAD @ 2 parallel = 6 waves × ~4 h = **~24 h wall-clock**
- Both machines concurrent splitting 12 each = **~24 h** (one full day)

If only one machine available: ~2 days. For the thesis timeline, split behavioral vs
laplacian DPAD across separate nights if needed.

## Dispatcher snippet

```bash
# On bobby, 2 concurrent slots
cat > /tmp/dpad_bobby.list << 'EOF'
training/setups/dpad/narrow_band_200Hz/both/dpad_behavioral_PDI1_2_nx_55_n15_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/off/dpad_behavioral_PDI1_2_nx_55_n15_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/on/dpad_behavioral_PDI1_2_nx_55_n15_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/both/dpad_behavioral_PDI1_4_nx_50_n10_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/off/dpad_behavioral_PDI1_4_nx_50_n10_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/on/dpad_behavioral_PDI1_4_nx_50_n10_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml
EOF

# On jacque, same structure with PDI4_* configs:
# (scp the jacque list to jacque)
cat > /tmp/dpad_jacque.list << 'EOF'
training/setups/dpad/narrow_band_200Hz/both/dpad_behavioral_PDI4_2_nx_50_n10_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/off/dpad_behavioral_PDI4_2_nx_50_n10_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/on/dpad_behavioral_PDI4_2_nx_50_n10_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/both/dpad_behavioral_PDI4_3_nx_50_n10_e3000_mrmr8_dbs_both_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/off/dpad_behavioral_PDI4_3_nx_50_n10_e3000_mrmr8_dbs_off_200Hz_narrow_band.yaml
training/setups/dpad/narrow_band_200Hz/on/dpad_behavioral_PDI4_3_nx_50_n10_e3000_mrmr8_dbs_on_200Hz_narrow_band.yaml
EOF

# Launcher (same script on both machines):
cd /home/bobby/repos/latent-neural-dynamics-modeling  # or /home/jacque/... on jacque
mkdir -p logs/dpad_launch
cat /tmp/dpad_bobby.list | xargs -n1 -P2 -I{} bash -c '
  name=$(basename {} .yaml)
  echo "[$(date +%H:%M:%S)] START $name"
  python -m training.train --config {} \
    > "logs/dpad_launch/${name}.log" 2>&1
  echo "[$(date +%H:%M:%S)] END   $name rc=$?"
'
```

## Progress / success check

For each of the 12, verify:

```bash
# Each run is successful if:
# 1. Model file exists
ls results/<config-stem>/model_*.pkl
# 2. Test results parquet exists
ls results/<config-stem>/test/test_results_*.parquet
# 3. Last log line says "Training completed successfully!"
tail -1 results/<config-stem>/logs/train_*.md
```

## When to run

- **Not now** — we're running PSID + VARMA first to pick the best PSID variant. DPAD compares to the winner PSID.
- **After PSID variant selection** (pick winner from Fig 37 retrained). Then run DPAD across all 12 configs.
- DPAD results feed the 3-model sec2b/c/d figures (`mpl_rmse_boxplot`, etc).

## Full-pipeline wrapper (pipeline_dpad.py)

If later you want classification + permutation tests on the DPAD models:

```bash
python scripts/pipeline_dpad.py \
  --session PDI4_S3 --family ecog --dbs both \
  --best-nx 50 --best-n1 10 \
  --start-phase 3 --end-phase 5
```

Phase 5 = classification grid + 100-permutation test. Adds ~30-60 min per run on top
of the 3 h training time.
