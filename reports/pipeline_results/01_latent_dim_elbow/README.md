# Stage 1 — Latent-dim elbow (nx, n1)

## Job

For each cell × feature-side, fit **vanilla PSID** (no smoother, no A-clip) on
the full 60-channel ECoG set (or full 15-band laplacian set), extract the
stage-1 and stage-2 singular spectra, and read the elbows:

- **Stage-1 singular values (`ZHat_S`)** → the *behavior-relevant* latent dim cutoff → **n1**
- **Stage-2 singular values (`YHat_S`)** → the *neural-additional* dims cutoff → **n2**; **nx = n1 + n2** (upper bound)
- **Target-K** for downstream channel selection is matched to the n1 elbow (upper bound on behavior-relevant dimensions)

The chosen values are recorded in [`configs/diagnostic/elbow_choices.yaml`](../../../configs/diagnostic/elbow_choices.yaml) (symlinked below).

## Producer

```bash
python scripts/pipeline_psid_diagnostic.py --config configs/diagnostic/pdi4_s3_psid_spectra.yaml
```

(One config per cell under `configs/diagnostic/`; each runs independently.)

## Artifacts on disk

Per cell: `results/diagnostic/{cell}_psid_spectra/`

```
correlation/<family>.parquet       # full correlation matrix (pre-mRMR redundancy)
correlation/<family>.png           # cluster-ordered heatmap + dendrogram
psid/<family>/spectra.parquet      # ZHat_S, YHat_S singular spectra
psid/<family>/matrices.npz         # A, Cy, Cz, Q, R, S + the two spectra
spectra_summary.png                # multi-panel spectra comparison (the elbow figure)
run_manifest.json                  # config snapshot + auto-detected elbows
```

Where `<family>` ∈ {`ecog`, `laplacian`}.

## Chosen values (the decision this stage outputs)

From `elbow_choices.yaml` (reviewed against each cell's `spectra_summary.png`):

| Cell | ecog (n1, nx, K) | laplacian (n1, nx, K) |
|---|---|---|
| PDI1_S2 | (15, 165, 8) | (10, 80, 8) |
| PDI1_S4 | (10, 80, 8) | (10, 70, 8) |
| PDI4_S2 | (10, 160, 8) | (10, 65, 8) |
| PDI4_S3 | (10, 160, 8) | (10, 65, 8) |

Notes:
- `nx` in this YAML is the *diagnostic* upper bound (n1 + stage-2 elbow). Actual training uses a **smaller nx** — typically (50, 10) or (55, 15) — see stage 3. The diagnostic nx is only for the vanilla-60-channel spectra figure.
- `target_K = 8` for all cells because the stage-1 elbow sits at 10-15, and mRMR top-8 gives enough signal headroom while keeping the channel count manageable for the reduced training runs.

## Vanilla full-spectrum matrices (step-1/step-2 outputs)

Each `psid/{ecog,laplacian}/matrices.npz` is a standalone snapshot of the
subspace-ID outputs on the full 60-channel ECoG feature set:

| Key | Shape | What |
|---|---|---|
| `A` | (nx, nx) | state transition |
| `Cy` | (60, nx) | neural projection |
| `Cz` | (2, nx) | behavior projection |
| `Q`, `R`, `S` | noise / cross covariances | |
| `ZHat_S` | (n1·i,) | **stage-1 singular values** |
| `YHat_S` | (n2·i,) | **stage-2 singular values** |

## Feeds into stage 2

The (nx, n1, target_K) values chosen here feed:

- `scripts/generate_reduced_training_configs.py` — emits reduced-feature training YAMLs with the chosen (n1, nx) per (cell, family)
- `scripts/pipeline_psid.py --best-nx <nx> --best-n1 <n1>` — uses them in Phase 3 training

## Symlinks to raw artifacts

See sibling directories:
- `configs_diagnostic/` → symlink to `configs/diagnostic/`
- `diagnostic_results/` → symlink to `results/diagnostic/`
