# Thesis story — end-to-end narrative

This is the single document that walks through the full thesis pipeline from
raw data to every figure, section by section. It ties stages 1-6 (see
`reports/pipeline_results/`) to the concrete `notebooks/thesis_sec*.py` files
that consume them.

> **Companion index**: [`reports/pipeline_results/README.md`](pipeline_results/README.md)
> gives the folder-level map with symlinks to raw artifacts.
> **This document** is the prose arc + figure catalogue.

---

## Table of contents

1. [Data + preprocessing](#data)
2. [Stage 1 — Latent-dim elbow (nx, n1)](#stage-1)
3. [Stage 2 — Channel selection (mRMR top-8)](#stage-2)
4. [Stage 3 — Full training (PSID + DPAD + VARMA)](#stage-3)
5. [Stage 4 — Evaluation (test + cross-condition)](#stage-4)
6. [Stage 5 — Classification (DBS decoding)](#stage-5)
7. [Stage 6 — Thesis figures](#stage-6)
8. [Side story — FB smoother ablation](#fb-ablation)
9. [Full notebook-by-notebook catalogue](#notebook-catalogue)

---

<a name="data"></a>
## 0. Data + preprocessing

**Dataset**: invasive ECoG + behavioral tracing task (DBS on/off) in Parkinsonian
patients. 4 canonical (participant, session) cells drive the thesis:

| Cell | Participant | Session |
|---|---|---|
| PDI1_S2 | PDI1 | 2 |
| PDI1_S4 | PDI1 | 4 |
| PDI4_S2 | PDI4 | 2 |
| PDI4_S3 | PDI4 | 3 |

**Preprocessing** (already done; not covered here in detail):

- ECoG sampled @ 200 Hz, narrow-band filtered into 15 bands across 4 electrodes → 60 features total
- Laplacian 14-16 LFP pair available as alternative feature set (15 narrow bands from one electrode pair)
- Behavioral output = tracing kinematics: `tracing_velocity_x`, `tracing_acceleration_magnitude`
- Trial inventory: 12 blocks × 12 trials nominally per session; fragmented-block + plateau filters drop noise
- Within-session chronological 60/15/45 split (not 60/10/30 — see `reports/DATA_EFFICIENCY_ANALYSIS.md` for why)
- Each block is **pure DBS on or off** (no mid-block transitions), so block boundaries are the DBS-state transitions

**Consumed in**: `thesis_sec1_data_verification` (data integrity figs 1-7) and by every downstream notebook via the split parquets at `results/<variant>/split/{train,val,test}.parquet`.

---

<a name="stage-1"></a>
## 1. Stage 1 — Latent-dim elbow (nx, n1)

**The question**: how many latent dimensions does the neural data actually
support for (a) behavior-relevant and (b) residual neural dynamics?

**Method** — Vanilla PSID on the FULL 60-channel ECoG (or 15-band laplacian)
with no A-clip and no smoother. Extract stage-1 singular values `ZHat_S` and
stage-2 singular values `YHat_S`. Elbow of `ZHat_S` = **n1** (behavior-relevant
dims); elbow of `YHat_S` = **n2** (neural-residual dims); **nx = n1 + n2**.

**Producer**:

```bash
python scripts/pipeline_psid_diagnostic.py --config configs/diagnostic/<cell>.yaml
```

**Outputs**:

- `results/diagnostic/<cell>_psid_spectra/psid/{ecog,laplacian}/matrices.npz`  — A, Cy, Cz, Q, R, S, ZHat_S, YHat_S
- `results/diagnostic/<cell>_psid_spectra/spectra_summary.png`  — the elbow figure you inspect
- `results/diagnostic/<cell>_psid_spectra/run_manifest.json`  — auto-detected elbow values

**Manual pick** — You inspect each `spectra_summary.png` and record the chosen
(n1, nx, target_K) into `configs/diagnostic/elbow_choices.yaml`. This is the
**canonical source of truth** for what (nx, n1) the rest of the pipeline uses.

**Chosen values**:

| Cell | ecog (n1, nx, K) | laplacian (n1, nx, K) |
|---|---|---|
| PDI1_S2 | (15, 165, 8) | (10, 80, 8) |
| PDI1_S4 | (10, 80, 8) | (10, 70, 8) |
| PDI4_S2 | (10, 160, 8) | (10, 65, 8) |
| PDI4_S3 | (10, 160, 8) | (10, 65, 8) |

Notes:
- `nx` in the diagnostic YAML is a high upper bound (for full-60-channel runs)
- **Actual training nx** is smaller: (50, 10) everywhere except PDI1_S2 which is (55, 15)
- `target_K = 8` uniform across cells (stage-1 elbow sits at 10-15, 8 channels is a safe upper bound on behavior-relevant signal headroom)

**Consumed in**:
- `thesis_sec2a_diagnostics` — fig 38 (PSID grid-search BA heatmap used originally to justify (nx, n1); the elbow story is now the primary justification)
- `thesis_sec6_summary_appendix` — fig 60 (classification vs dimensionality heatmap as supplementary)

---

<a name="stage-2"></a>
## 2. Stage 2 — Channel selection (mRMR top-8)

**The question**: of the 60 ECoG features (or 15 LFP bands), which 8 carry the
most behavior-relevant + non-redundant signal per cell?

**Method** — mRMR (max relevance, min redundancy) against the behavioral
outputs. Applied separately per cell × side. Runs against the train+val
portion of the chronological split (not test, to avoid leakage).

**Producer**:

```bash
python scripts/generate_reduced_training_configs.py \
    --elbow-choices configs/diagnostic/elbow_choices.yaml
```

Emits one training YAML per (cell, family, dbs_condition, model_type). Each
YAML's `data.channels.neural_input` is the mRMR-selected 8-channel list.

**Where the selections live**: inline in each training YAML at
`training/setups/{psid,dpad,varma}/{narrow_band_200Hz,laplacian_200Hz}/{both,on,off}/<variant>.yaml`.

All 8 per-cell × per-side picks are enumerated in
[`pipeline_results/02_channel_selection/channels_per_cell.md`](pipeline_results/02_channel_selection/channels_per_cell.md).

**Why the same 8 channels work for DPAD + VARMA**: to keep the comparison
apples-to-apples, DPAD and VARMA use the same 8 that PSID mRMR picked for the
same cell × side. This eliminates "PSID looks better because it has better
features" as a confounder.

**Consumed in**:
- Every stage-3 training run picks these 8 channels from the YAML
- `thesis_sec7_subspace_dynamics` fig 6 (channel importance scatter — uses all 60 ECoG to show how the 8 rank)

---

<a name="stage-3"></a>
## 3. Stage 3 — Full training (PSID + DPAD + VARMA)

**Three model families, same 8-channel input, same split**:

### PSID (primary model)

Subspace ID with Sani & Shanechi 2025 forward-backward smoother + filter-aware
forecast + A-eigenvalue clipping (`max_eigenvalue = 0.9999`). Core innovation
of the thesis is switching on the FB smoother while keeping the same A, Cy, Cz.

4 variants per (cell, mode):
1. `dbs_both` — trained on all trials (canonical variant for reconstruction + classification)
2. `dbs_on` — trained on DBS-on trials only
3. `dbs_off` — trained on DBS-off trials only
4. `vanilla_both` — same data as (1) but **no FB**, **no A-clip** (channel-selection baseline / ablation reference)

Plus `eval_on` / `eval_off` cross-evaluation variants (use off-model on on-data and vice versa).

### DPAD (neural-network alternative)

Behavioral mode only (no laplacian variant). Same 8 channels, same split.
Emphasis on convergence (sec 2a fig 39) and direct PSID-vs-DPAD comparison.

### VARMA (classical reference)

AR(p=30), q=1, trained for `both`, `on`, `off`. Reference for "how much does a
classical model do with the same features?"

### Producer

Unified:

```bash
python scripts/pipeline_psid.py \
    --participant PDI4 --session 2 --mode behavioral \
    --best-nx 50 --best-n1 10
```

Phases:
- **3**: train 4 variants (both / on / off / vanilla_both), parallel, with retry ladder on eigenvalue failures
- **4**: cross-condition eval
- **5**: classification grid + permutation
- **6**: `specs.py` triplet registry update

Phases 1 + 2 (old nx/n1 grid search) are **removed**; stage 1 elbow picks feed
Phase 3 directly as CLI args.

### Registry — the (variant, run_ts) triplets

Authoritative file: [`notebooks/thesis_sec2_common.py`](../notebooks/thesis_sec2_common.py).
Every thesis notebook imports triplets from this module.

**Behavioral**:

| Cell | PSID variant | PSID ts | DPAD ts | VARMA ts |
|---|---|---|---|---|
| PDI1_S2 | `psid_behavioral_PDI1_2_nx_55_n15_i100_dbs_both_200Hz_narrow_band` | 20260421_222439 | 20260419_010200 | 20260420_133800 |
| PDI1_S4 | `psid_behavioral_PDI1_4_nx_50_n10_i100_dbs_both_200Hz_narrow_band` | 20260422_000702 | 20260419_093953 | 20260420_134630 |
| PDI4_S2 | `psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band` | 20260421_202056 | 20260418_225805 | 20260420_113757 |
| PDI4_S3 | `psid_behavioral_PDI4_3_nx_50_n10_i100_dbs_both_200Hz_narrow_band` | 20260421_202721 | 20260419_074635 | 20260420_114212 |

**Laplacian** (no DPAD; PSID + VARMA only):

| Cell | PSID variant | PSID ts | VARMA ts |
|---|---|---|---|
| PDI1_S2 | `psid_laplacian_PDI1_2_nx_55_n15_i100_dbs_both_200Hz_narrow_band` | 20260422_001503 | 20260420_134013 |
| PDI1_S4 | `psid_laplacian_PDI1_4_nx_50_n10_i100_dbs_both_200Hz_narrow_band` | 20260422_003140 | 20260420_134815 |
| PDI4_S2 | `psid_laplacian_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band` | 20260421_203357 | 20260420_113921 |
| PDI4_S3 | `psid_laplacian_PDI4_3_nx_50_n10_i100_dbs_both_200Hz_narrow_band` | 20260421_204455 | 20260420_114334 |

### Consumed in

All of sec2 (b, c, d, e, model_validation), sec5, sec6, sec7 — basically any
notebook that reads model output parquets pulls (variant, run_ts) from this
registry.

---

<a name="stage-4"></a>
## 4. Stage 4 — Evaluation (test + cross-condition)

**Job**: run every trained model on its `test/` split, emit reconstruction +
multi-step forecast parquets, and run cross-condition eval.

### Target conventions

| Target | behavioral mode | laplacian mode |
|---|---|---|
| **Y** | 8-channel ECoG (self-reconstruction) | 8-channel ECoG (self-reconstruction) |
| **Z** | 2-channel behavior | 15-band LFP |

"Reconstruction" = one-step-ahead (also called prediction in some code).
"Forecast" = multi-step ahead, `m` seconds into the future.

### Cross-condition eval

For each `dbs_on` model there's a `<variant>_eval_off` sibling dir that runs
the same model on DBS-off trials. Analogously for `dbs_off` → `_eval_on`. This
tests: does a DBS-on-trained model generalise to DBS-off data? (sec5 figs 53-54.)

### FB dispatch at inference

The FB smoother is **not part of the stored matrices**; it's a `bool` flag on
the model instance. `predict()` / `forecast()` dispatch on it. This lets us:
- Save ONE FB-enabled model and read back RTS outputs by flipping the flag at inference (`mode="forward"` / `mode="fb"`)
- Build the 4-condition ablation (FB/RTS × clip/no-clip) without retraining (see `scripts/run_option_a_rts_clip.sh`)

### Consumed in

- sec2b (fig 7-16 behavioral RMSE)
- sec2c (fig 22-23 per-band heatmaps, fig 39-49 reconstruction)
- sec2d (fig 23-36, 56-59 forecast)
- sec2e (fig 18-21, 29-36 single-trial exemplars; fig 44-45 best-PSID)
- sec8 (all figs — per-trial × per-channel heatmaps on test split)

---

<a name="stage-5"></a>
## 5. Stage 5 — Classification (DBS-OFF vs DBS-ON decoding)

**The headline thesis result**: can you decode DBS state from latent neural
activity? If yes, the latent subspace is capturing DBS-relevant dynamics.

### Protocol

**Prediction mode** (epochs of current latent `Xp`):
- 4 feature sources: `Xp` (full latent), `Xp_1` (behavior subspace), `Xp_2` (residual), `Xp+DBS` (with DBS covariate)
- 5-fold CV grid-search LDA hyperparams
- Final test BA on held-out epochs

**Forecast mode** (epochs of `m`-second-ahead latent forecasts):
- h × m grid: h ∈ {1.0, 2.0, 3.0} s, m ∈ {0.5, 1.0} s → 6 cells
- 4 feature sources × 6 cells = 24 forecast classifiers per variant
- Pipeline CV-picks best (h, m), runs permutation on that

**Negative controls**:
- **Flipped** — labels shuffled within fold, BA ≈ 0.5 expected
- **Permutation test** — block-permuted labels, n=1000, p-value on observed BA

### Why on/off single-condition models aren't classified

A `dbs_on`-trained model has only DBS-on trials in its test set → only one
class → classification is undefined. Only `dbs_both` variants are classified.

### Outputs

`results/classification/<variant>/<run_ts>/{LDA_Xp_prediction, LDA_Xp_1_prediction, ...}.<ext>` plus forecast subdirs `h{h}_m{m}/LDA_Xp_forecast.<ext>` etc. Flipped + permutation land under sibling dirs (`<variant>_flipped/`, `<variant>_perm/`).

### Coverage (2026-04-22)

- PSID behavioral 4/4, laplacian 4/4 — full stats
- DPAD behavioral 4/4 (PDI1 rsynced from jacque, flipped in flight)
- DPAD permutation: configs exist, haven't been launched

### Consumed in

- sec5 (fig 49 grouped BA bar chart, fig 50 standard heatmap, fig 51 flipped heatmap, fig 52 ROC, fig 53-54 within vs cross-condition)
- sec6 fig 60 (classification vs dimensionality supplementary)

---

<a name="stage-6"></a>
## 6. Stage 6 — Thesis figures

All figures land under `thesis_figures/sec{N}/fig_{NNN}_{desc}.png` at
`scale=2`. Styling lives in `notebooks/thesis_style.py` (central) with
per-figure helpers in `notebooks/thesis_lib/`.

**Rules** (from memory):
- Import from `notebooks/thesis_style.py`, never `dashboard/thesis/constants.py` (dashboard is being retired)
- Use `panel_label()` + `apply_paper_style()`
- Compact height, plain `-` panel labels, metric-only y-axis, max 2 colors
- Raw session names (`PDI1_S2`, not `PDI1 S2`)
- IQR-zoomed y-ranges for RMSE boxplots (Tukey window)

**Producer**: jupytext-paired `thesis_sec*.py` / `.ipynb` notebooks. Rerun
via `jupyter nbconvert --execute --inplace`.

---

<a name="fb-ablation"></a>
## 7. Side story — FB smoother ablation

To isolate what the FB smoother buys you, we built a 4-condition comparison:

| Condition | A-clip | FB smoother |
|---|---|---|
| RTS-no-clip | ✗ | ✗ |
| FB-no-clip | ✗ | ✓ |
| RTS-clip | ✓ | ✗ (derived via flag-flip at inference) |
| FB-clip | ✓ | ✓ (the canonical training) |

Producer:
- `scripts/run_fb_chain.sh` trains the no-clip variants
- `scripts/run_option_a_rts_clip.sh` derives RTS-clip by flag-flipping FB-clip models
- `scripts/aggregate_fb_ablation.py` builds the comparison table

Output: [`fb_ablation_results.md`](../fb_ablation_results.md) + `.csv` (project root).
Main finding: FB helps forecast classification by ≈ +0.02 mean BA across cells;
reconstruction is near-identical between FB and RTS because A, Cy, Cz are the
same — only the smoother applies different math.

Not yet wired into any thesis figure; would go in sec2 or an appendix.

---

<a name="notebook-catalogue"></a>
## 8. Notebook-by-notebook catalogue (what's in each thesis_sec*.py)

Each notebook is paired `.ipynb` + `.py` (jupytext). The `.py` is the source
of truth; edits propagate via `jupytext --sync`.

### `thesis_sec1_data_verification.py` — 7 figs (sec1/)

Data integrity checks. Produces:

| Fig | Description |
|---|---|
| 1 | Trial inventory — original vs preprocessed trial counts, fragmented blocks, split distribution |
| 2 | Trial count summary — DBS-OFF / DBS-ON bar chart per session |
| 3-6 | PSD DBS comparison — power spectral density per ECoG channel, 4 sessions |
| 7 | Tracing speed DBS comparison — mean velocity + acceleration traces |

**Purpose**: verify the data is clean before any modelling claim rests on it.
Reads `data/participants_2/` hive parquet + split files.

### `thesis_sec2a_diagnostics.py` — 3 figs (sec2/)

Model selection + training diagnostics.

| Fig | Description |
|---|---|
| 37 | Vanilla PSID vs RTS + A-clip — grouped bar chart across 4 sessions; behavioral + neural metrics in RMSE / Pearson / VAF |
| 38 | PSID grid-search BA heatmap — per-session (nx, n1) grid with chosen cell outlined; supplementary elbow justification |
| 39 | DPAD training curves — training + val MSE across 4 sessions × 3 DBS conds; validates DPAD convergence |

**Purpose**: justifies why you chose RTS+A-clip over vanilla, and why the
specific (nx, n1). Supports stages 1 and 3.

### `thesis_sec2b_behavioral.py` — 10+ figs (sec2/)

Behavioral decoding group metrics — "how well does each model predict
tracing velocity / acceleration from neural?"

| Fig | Description |
|---|---|
| 7-8 | Pooled per-trial behavioral metric (velocity, acceleration) across all 4 sessions × 3 PNGs each (RMSE / Pearson / VAF) |
| 9-16 | Per-session per-trial metric (2 behavioral channels × 4 sessions × 3 metrics) |
| 17 | Session-mean RMSE box-plus-strip summary (legacy layout, RMSE only) |

**3-PNG pattern** is consistent throughout sec2: emit RMSE / Pearson / VAF
variants and pick the best-reading metric at compose time.

### `thesis_sec2c_neural_recon_group.py` — many figs (sec2/)

Neural reconstruction (one-step).

| Fig | Description |
|---|---|
| 22 | ECoG per-band metric heatmap (3 PNGs) — parent-band × model, DBS-OFF / DBS-ON panels |
| 23 | Laplacian LFP per-band metric heatmap (3 PNGs) — companion to fig 22 for laplacian mode |
| 39 | Pooled per-trial neural prediction metric (3 PNGs; PSID-best channel per session) |
| 40-43 | Per-session per-trial neural prediction (4 sessions × 3 metrics) |
| 46-49 | Per-session LFP reconstruction (4 sessions × 3 metrics; PSID + VARMA 2-model layout) |

**Note**: fig 23 is used by TWO notebooks (sec2c + sec2d) for different
content — naming clash to fix.

### `thesis_sec2d_neural_forecast_group.py` — many figs (sec2/)

Neural forecast (multi-step). PSID + VARMA only — DPAD forecast head not
implemented.

| Fig | Description |
|---|---|
| 23 | Forecast RMSE vs horizon (0-1000 ms) — SEM bands + crossover annotation vs naïve baseline |
| 24 | Pooled per-trial forecast metric (3 PNGs) |
| 25-28 | Per-session forecast metric distributions (4 sessions × 3 metrics) |
| 29-36 | Neural forecast exemplars — best trial per (session × DBS-cond) selected by minimising max(rmse_psid, rmse_varma) |
| 56-59 | Per-session LFP forecast (4 sessions × 3 metrics; PSID + VARMA, per-session PSID-best band) |

### `thesis_sec2e_exemplars.py` — single-trial panels (sec2/)

Per-trial, per-session time-series demonstrations.

| Fig | Description |
|---|---|
| 18-21 | Neural reconstruction time series — side-by-side DBS-OFF / DBS-ON per session; true vs PSID / DPAD / VARMA one-step |
| 29-36 | Neural forecast exemplars — history + forecast window with true + PSID + VARMA traces (same fig numbers as sec2d — they're the same content, rendered once) |
| 44 | Best PSID one-step prediction by RMSE across 6 shared features |
| 45 | Best PSID forecast by VAF across 6 shared features |
| 51-55 | 12-panel exemplar figures — 4 participants × 3 feature types × 2 target modes × 3 metrics = 6 figures |

### `thesis_sec2_model_validation.py` — index notebook

Meta-notebook. Enumerates what's in sec2a-e with builder function references.
Exists as an audit trail more than a figure producer.

| Fig | Builder | Count |
|---|---|---|
| 7-8 | `build_rmse_boxplot_figure()` | 2 |
| 9-16 | `build_rmse_boxplot_figure()` | 8 |
| 17 | inline | 1 |
| 18-21 | `compose_thesis_neural_figure()` | 4 |
| 22 | inline | 1 |
| 23 | `build_forecast_rmse_figure_or_empty()` | 1 |
| 24 | `build_rmse_boxplot_figure()` | 1 |
| 25-28 | `build_rmse_boxplot_figure()` | 4 |
| 29-36 | inline | 8 |
| 37 | inline | 1 |
| 38 | inline | 1 |
| 39 | `build_rmse_boxplot_figure()` | 1 |
| 40-43 | `build_rmse_boxplot_figure()` | 4 |

### `thesis_sec5_classification.py` — RQ3 (sec5/)

**DBS Classification & Cross-Condition** — 10 figures.

| Fig | Builder |
|---|---|
| 49 | `build_classification_grouped_bar_figure()` — LDA BA by model × feature source × DBS cond |
| 50 | `build_standard_heatmap_figure()` — classification heatmap |
| 51 | `build_flipped_heatmap_figure()` — negative-control heatmap |
| 52 | `build_roc_curve_figure()` — ROC curves for the best classifiers |
| 53-54 | `build_within_cross_boxplot_figure()` — within vs cross-condition RMSE |

**Reads**: `results/classification/<variant>/<ts>/LDA_*_prediction.*` and forecast variants.

### `thesis_sec6_summary_appendix.py` — 6 figs (sec6/)

Summary + appendix.

| Fig | Description |
|---|---|
| 59 | Latent phase space trajectories — `build_latent_phase_space_figure()` |
| 60 | Classification vs dimensionality heatmap (grid-search ablation) — `build_classification_heatmap_figure()` |
| 61 | PSID Cy importance heatmap — `build_psid_cy_importance_figure()` |
| 62 | PSID Cz readout matrix — `build_psid_cz_figure()` |
| 63 | DPAD training curves — `build_dpad_training_curves_figure()` |
| 64 | Data efficiency — `build_data_efficiency_figure()` |

### `thesis_sec7_subspace_dynamics.py` — subspace analysis (sec7/)

Latent subspace properties across DBS on/off.

| Section | Output |
|---|---|
| 1 | Behavioral DBS effect summary table |
| 2 | Latent state trial-level statistics per session |
| 3 | PSD of latent states per session |
| 4 | A matrix structure + eigenvalues per session |
| 5 | C / Cz matrix loadings per session |
| 6 | Channel importance scatter per session |
| 7 | Classifier comparison (mean vs cov) per session |
| 8 | Cross-run summary table |

Reads: PSID both / on / off matrices; compares latent dynamics between DBS states.

### `thesis_sec8_per_trial_heatmaps.py` — test-set deep dive (sec8/)

Per-trial × per-channel test-set heatmaps for identifying exemplar trials and
diagnosing per-channel behavior.

| Fig | Description |
|---|---|
| 1 | Neural prediction Pearson r — trials × 60 neural channels |
| 2 | Neural prediction RMSE — trials × 60 |
| 3 | Neural forecast Pearson r — trials × 60 (Y_future) |
| 4 | Neural forecast RMSE — trials × 60 (Y_future) |
| 5 | Behavioral prediction Pearson r + RMSE — trials × 2 |
| 6 | Behavioral forecast Pearson r + RMSE — trials × 2 (Z_future) |
| 7 | Mean neural Pearson r summary — electrodes × bands, all sessions |

Used to **pick exemplar trials** for sec2e figures 29-36 (forecast exemplars).

---

## 9. Current known gaps

From my latest audit, these are the remaining figure / result gaps:

1. **Pooled LFP prediction** (fig 44 / 45 slot) — missing (laplacian equivalent of fig 39)
2. **Pooled LFP forecast** — missing (laplacian equivalent of fig 24)
3. **Behavior forecast** (pooled + per-session) — completely absent; would use `forecast_target="Z"` on behavioral triplets
4. **DPAD permutation tests** — configs exist, not yet launched
5. **DPAD PDI1_S4 flipped** — in flight on jacque (PID 1973032, ~2h elapsed)
6. **Fig 23 naming clash** — `fig_023_neural_forecast_rmse.png` (sec2d) and `fig_023_lfp_band_*.png` (sec2c) both claim fig 23

---

## 10. How to rebuild from scratch

In the order stages feed each other:

```bash
# Stage 1 — elbow discovery
python scripts/pipeline_psid_diagnostic.py --config configs/diagnostic/<cell>.yaml
# inspect results/diagnostic/<cell>_psid_spectra/spectra_summary.png,
# edit configs/diagnostic/elbow_choices.yaml to record (n1, nx, target_K) picks

# Stage 2 — reduced-feature training YAMLs (mRMR picks baked in)
python scripts/generate_reduced_training_configs.py \
    --elbow-choices configs/diagnostic/elbow_choices.yaml

# Stage 3-5 — PSID training + eval + classification (one participant/session)
python scripts/pipeline_psid.py \
    --participant PDI4 --session 2 --mode behavioral \
    --best-nx 50 --best-n1 10

# DPAD (separate pipeline, PDI1 on jacque)
python scripts/pipeline_dpad.py --participant PDI1 --session 2 --nx 55 --n1 15

# VARMA via overnight scripts
python scripts/overnight_all_sessions.py --algo varma --subset all

# Stage 6 — figures (after triplet registry in thesis_sec2_common.py is current)
for nb in notebooks/thesis_sec*.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 "$nb" &
done
wait
```

---

## 11. File/directory quick reference

| What | Where |
|---|---|
| **This narrative** | `reports/THESIS_STORY.md` |
| **Folder index w/ symlinks** | `reports/pipeline_results/README.md` |
| **Elbow picks** | `configs/diagnostic/elbow_choices.yaml` |
| **Diagnostic results** | `results/diagnostic/<cell>_psid_spectra/` |
| **Training YAMLs** | `training/setups/{psid,dpad,varma}/...` |
| **Model outputs** | `results/<variant>/` |
| **Classification outputs** | `results/classification/<variant>/<ts>/` |
| **Classification configs** | `classification/setups/*.yaml` |
| **Triplet registry** | `notebooks/thesis_sec2_common.py` |
| **FB implementation** | `utils/frameworks.py` (PSIDWrapper class) |
| **Figure styling** | `notebooks/thesis_style.py` |
| **Figure helpers** | `notebooks/thesis_lib/` |
| **Thesis notebooks** | `notebooks/thesis_sec*.{py,ipynb}` (jupytext-paired) |
| **Rendered figures** | `thesis_figures/sec{N}/*.png` |
| **FB ablation** | `fb_ablation_results.md` + `.csv` (project root) |
| **FB implementation doc** | `fb_implementation.md` (project root) |
| **Live pipeline log** | `pipeline_runs.md` (project root) |
| **Data efficiency analysis** | `reports/DATA_EFFICIENCY_ANALYSIS.md` |
| **nx choice analysis** | `reports/NX_CHOICE_ANALYSIS.md` |
