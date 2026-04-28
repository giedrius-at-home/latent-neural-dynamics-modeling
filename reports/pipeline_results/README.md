# Results pipeline — narrative index

This directory is a **curated, navigable index** of the results produced by the
PSID + DPAD + VARMA pipeline for the thesis. Raw artifacts live under
`results/` / `training/setups/` / `thesis_figures/` (see repo root); this index
tells the *story* and points (via symlinks or explicit paths) to what each
stage produced and why it feeds the next.

The story, in six stages:

```
┌─────────────────────────┐
│ 1. Latent-dim elbow     │   diagnostic singular spectra → pick (nx, n1)
│    (per cell × side)    │   per cell × feature-side (ecog / laplacian).
└─────────────┬───────────┘
              │
┌─────────────▼───────────┐
│ 2. Channel selection    │   mRMR top-8 per cell × side (behavioral /
│    (per cell × side)    │   neural / laplacian). Feeds the reduced
└─────────────┬───────────┘   training YAMLs.
              │
┌─────────────▼───────────┐
│ 3. Full training        │   4 variants per (cell, mode):
│    (FB smoother on)     │     both / on / off / vanilla_both
└─────────────┬───────────┘   at the chosen (nx, n1), i=100,
              │               max_eigenvalue=0.9999.
┌─────────────▼───────────┐
│ 4. Evaluation           │   test-split reconstruction + forecast;
│    (+ cross-condition)  │   on→off and off→on cross-eval.
└─────────────┬───────────┘
              │
┌─────────────▼───────────┐
│ 5. Classification       │   LDA on latent states:
│    (h × m grid + perm)  │   prediction mode + 6 forecast (h, m) cells,
└─────────────┬───────────┘   4 feature sources (Xp / Xp_1 / Xp_2 / Xp+DBS),
              │               flipped (negative control) + permutation test.
┌─────────────▼───────────┐
│ 6. Thesis figures       │   sec1 … sec7: plots consumed from stages 1-5.
│    (sec1 … sec8)        │
└─────────────────────────┘
```

Each stage folder has its own **README.md** that:
1. Describes the stage's job,
2. Lists which configs / scripts produce its artifacts,
3. Points at the exact files / directories (relative to repo root),
4. Identifies the decision that feeds into the next stage.

## Index

| Stage | Folder | Decision it produces |
|---|---|---|
| 1 | [`01_latent_dim_elbow/`](01_latent_dim_elbow/README.md) | (nx, n1) per cell × side, recorded in `configs/diagnostic/elbow_choices.yaml` |
| 2 | [`02_channel_selection/`](02_channel_selection/README.md) | top-8 channels per cell × side, baked into each training YAML's `data.channels.neural_input` |
| 3 | [`03_training/`](03_training/README.md) | trained model timestamps per (variant, mode, dbs_condition) — recorded in `notebooks/thesis_sec2_common.py` |
| 4 | [`04_evaluation/`](04_evaluation/README.md) | test-split parquets and cross-condition eval parquets per variant |
| 5 | [`05_classification/`](05_classification/README.md) | LDA pkls: per-feature × h×m × perm/flipped per variant |
| 6 | [`06_figures/`](06_figures/README.md) | figure numbers → source notebook → source stage |

## Related docs (pre-existing, still authoritative)

- **`reports/NX_CHOICE_ANALYSIS.md`** — rationale for nx selection
- **`reports/DATA_EFFICIENCY_ANALYSIS.md`** — split design (60/15/45 fix)
- **`reports/OVERNIGHT_RESULTS.md`** — earlier run results summary
- **`reports/reduced_training_plan.md`** — channel-selection plan
- **`pipeline_runs.md`** (repo root) — live log of completed / in-flight runs
- **`fb_implementation.md`** (repo root) — Sani & Shanechi 2025 FB smoother implementation
- **`fb_ablation_results.md`** (repo root) — 4-condition ablation table

## Cells

Everything below operates on the 4 canonical (participant, session) cells:

| Cell | Participant | Session | (nx, n1) ecog | (nx, n1) laplacian |
|---|---|---|---|---|
| **PDI1_S2** | PDI1 | 2 | (55, 15) | (55, 15) |
| **PDI1_S4** | PDI1 | 4 | (50, 10) | (50, 10) |
| **PDI4_S2** | PDI4 | 2 | (50, 10) | (50, 10) |
| **PDI4_S3** | PDI4 | 3 | (50, 10) | (50, 10) |

And three model families: PSID (primary), DPAD (behavioral only), VARMA (reference).
