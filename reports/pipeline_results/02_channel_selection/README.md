# Stage 2 — Channel selection (mRMR top-K per cell × side)

## Job

With the stage-1 n1 elbow in hand, pick the **top-K = 8** most informative
channels (via mRMR — max relevance, min redundancy) from the full 60-band ECoG
set (or 15-band laplacian set) for each cell. The reduced channel set is what
the FULL PSID / DPAD / VARMA runs train on (stage 3).

## Producer

```bash
python scripts/generate_reduced_training_configs.py \
    --elbow-choices configs/diagnostic/elbow_choices.yaml
```

Reads `elbow_choices.yaml` (stage 1 output) and emits one training YAML per
(cell, family, dbs_condition, model_type). Each YAML's
`data.channels.neural_input` list is the mRMR-selected channel set.

## Where the selections live (the decision this stage outputs)

The top-8 channel list is **not** stored in a separate `.json` any more — it
lives inline in each training YAML's `data.channels.neural_input` field.

Canonical dbs_both PSID YAMLs (each one carries the top-8 for that cell × side):

| Cell × Side | Training YAML |
|---|---|
| PDI1_S2 ecog | `training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI1_2_nx_55_n15_i100_dbs_both_200Hz_narrow_band.yaml` |
| PDI1_S4 ecog | `training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI1_4_nx_50_n10_i100_dbs_both_200Hz_narrow_band.yaml` |
| PDI4_S2 ecog | `training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band.yaml` |
| PDI4_S3 ecog | `training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI4_3_nx_50_n10_i100_dbs_both_200Hz_narrow_band.yaml` |
| PDI1_S2 laplacian | `training/setups/psid/laplacian_200Hz/both/psid_laplacian_PDI1_2_nx_55_n15_i100_dbs_both_200Hz_narrow_band.yaml` |
| PDI1_S4 laplacian | `training/setups/psid/laplacian_200Hz/both/psid_laplacian_PDI1_4_nx_50_n10_i100_dbs_both_200Hz_narrow_band.yaml` |
| PDI4_S2 laplacian | `training/setups/psid/laplacian_200Hz/both/psid_laplacian_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band.yaml` |
| PDI4_S3 laplacian | `training/setups/psid/laplacian_200Hz/both/psid_laplacian_PDI4_3_nx_50_n10_i100_dbs_both_200Hz_narrow_band.yaml` |

## The selected channels (snapshot)

ECoG PSID top-8 per cell (from the `neural_input` lists in the YAMLs above):

| Cell | Selected ECoG channels |
|---|---|
| **PDI1_S2** | see `channels_per_cell.md` (auto-extracted) |
| **PDI1_S4** | " |
| **PDI4_S2** | " |
| **PDI4_S3** | " |

Run `python scripts/print_neural_inputs.py` (if it exists) or just
`grep -A 10 neural_input <yaml>` to dump them.

## DPAD / VARMA channel sets

DPAD and VARMA are trained on the **same top-8 ECoG set** per cell (for
apples-to-apples comparison). Their YAMLs live under
`training/setups/{dpad,varma}/narrow_band_200Hz/both/`.

For cells where the DPAD or VARMA YAMLs haven't been rsynced back from jacque
(PDI1 cells, often), `notebooks/thesis_loaders.py` falls back to:
1. Parquet-column `input_channels` (DPAD stores it; VARMA doesn't),
2. Training-log `logs/train_<ts>.md` YAML block parse (universal).

## Relationship between top-K and n1

`target_K = 8` for every cell even though n1 varies (10 or 15). This is
because the stage-1 elbow is the *upper bound* on behavior-relevant latent
dimensions, and n1 ≤ min(target_K, true_elbow). An 8-channel input with 8
latent dims is well-conditioned; going lower than 8 on input would artificially
cap the behavior-relevant subspace.

## Feeds into stage 3

Each generated YAML is consumed directly by:

```bash
python -m training.train --config training/setups/psid/.../<variant>.yaml
```

or by `scripts/pipeline_psid.py` which calls `training.train` internally.

## Symlinks

- `training_setups/` → symlink to `training/setups/` (all family YAMLs)
