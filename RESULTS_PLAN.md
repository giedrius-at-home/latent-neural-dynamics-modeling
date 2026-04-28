# Results Section: End-to-End Execution Plan

A scaffold to work through section by section. Each block tells you: what the
section does, what figures/tables go in, what to delete from the current draft,
and what to write fresh.

This is the working document; `RESULTS_PLAN.pdf` is the earlier snapshot.
Differences from the PDF are narrative (laplacian parallels, forecast
confidence bands, permutation scores, DPAD flexibility, style conventions)
woven in where they belong rather than as an addendum.

---

## File restructure

```
sections/04_results/
├── 04_results.tex                    # update \input list
├── 04_00_data.tex                    # KEEP, light edits
├── 04_01_model_selection.tex         # KEEP, expand
├── 04_02_one_step_prediction.tex     # NEW (merges old 02 + 03 prediction parts)
├── 04_03_forecasting.tex             # NEW (merges old 02 + 03 forecasting parts)
├── 04_04_classification.tex          # KEEP, trim (interpretability content out)
├── 04_05_interpretability.tex        # NEW
├── 04_06_cross_condition.tex         # KEEP, renumber from old 05
└── 04_07_summary.tex                 # REWRITE, renumber from old 06
```

Delete after migration: `04_02_rq1_neural.tex`, `04_03_rq2_behavioral.tex`,
`04_04_rq3_classification.tex`, `04_05_cross_condition.tex`, `04_06_summary.tex`.

## Figure folder convention

Copy each PNG you keep from `thesis_figures/` into `report/thesis/figures/`
under a clean snake_case name. Don't reference the source paths directly —
they have spaces and em-dashes that LaTeX hates.

---

## Thesis-wide conventions

These apply to every figure in the Results chapter; documented once so they're
not repeated in each section.

**Two data streams, always both.** Every cell in the study was trained in two
modes: `behavioral` (neural → tracing kinematics) and `laplacian` (ECoG →
LAPLACIAN-14-16 LFP, 15 bands). Figures that summarize model performance show
both streams where the metric makes sense — either as parallel figures
(behavioral version + laplacian version, not side-by-side in one panel, so
they compose cleanly into a collage for the final report) or as a combined
subsection (e.g. 4.3.3 Cross-modal is the laplacian counterpart to 4.3.1).
DPAD has no laplacian variant, so the laplacian versions are 2-model (PSID +
VARMA); all multi-model code uses `has_dpad_data()` / `show_dpad=False` to
gracefully drop DPAD when absent.

**Model colors.** PSID blue, DPAD red, VARMA gray. DBS-OFF blue badge, DBS-ON
red badge. Participant dot colors from `PARTICIPANT_COLORS`.

**Font hierarchy (strict).** `FONT_SIZE_TICK` for all body text (axis tick
labels, data annotations, cell values in heatmaps). `FONT_SIZE_LABEL` for axis
titles. `FONT_SIZE_BASE` for headings / figure titles. No other sizes used
anywhere.

**Line widths and separators.** Use Plotly defaults. Don't hardcode
`line.width` or set custom separator thickness — whatever is in
`thesis_style.apply_thesis_style()` is canonical. Cell gaps in heatmaps stay
at the current `xgap=2, ygap=2` since they're part of the visual style, not
line-weight overrides.

**Shaded confidence on forecast traces.** Every multi-step forecast exemplar
figure (4.4 forecast exemplars, 4.5 forecast confidence figure) carries a
±1 SE shaded band around each model's predicted trace, computed empirically
from pooled per-step residuals across test trials. This is Q5 option (C) —
chosen because it's the only method that gives PSID, DPAD, and VARMA bands
with the same meaning so the three models compare apples-to-apples. Model-
internal Kalman covariance was considered but rejected because it only works
for PSID, creating an asymmetric figure.

**Permutation p-values.** 4.5 classification figures display the permutation
p-value alongside each BA cell (annotated `**` for p<0.01, `*` for p<0.05, `ns`
otherwise). Comes from DPAD/PSID Phase 4 Step 2 output. When Step 2 hasn't run
yet for a cell, the figure shows the BA alone with no asterisk; the caption
notes "permutation pending". `pipeline_runs.md` is the source of truth for
which cells have Step 2 done.

**Sec1 is the style reference.** `thesis_sec1_data_verification.ipynb` is the
cleanest existing notebook — panel labels, captions, subplot structure,
margin choices. Match its patterns when building or rebuilding figures
elsewhere (4.2-4.8). Discrepancies with the current sec2 notebooks get
resolved in sec1's favor.

**Per-figure technical description.** For every figure listed in 4.1-4.7,
produce two artefacts alongside the PNG:
1. A **figure-specific TODO list** — granular build steps written *before*
   coding the figure, not after. Goes in
   `report/thesis/figures/technical_descriptions/<fig_name>.todo.md`.
2. A **technical description** — what the figure shows, how it's computed
   (loaders, aggregators, builders), the style choices (colorscale ranges,
   heights, annotations), and any caveats. Written *after* the figure is
   finalized. Same directory, `<fig_name>.md`.
Both live under `report/thesis/figures/technical_descriptions/` so they
ship with the thesis supplement and anyone re-running the code has the
recipe.

**Thorough over efficient.** One figure at a time. Don't batch-regenerate
multiple figures in a single command to save tokens — each figure gets its
own inspection pass, its own caption draft, its own tech-MD. Debugging a
broken batch is slower than walking through them one at a time.

---

## 4.1 Dataset and preprocessing verification

**Status:** KEEP `04_00_data.tex`, light edits.

**Purpose.** Show the dataset is balanced enough to model and that OFF/ON
conditions produce a real difference at the signal level — in both the ECoG
surface signal (behavioral-mode input) and the Laplacian LFP (laplacian-mode
output).

**Narrative beats.**
1. Trial counts after exclusion (one sentence + barplot, appendix with the
   42 removed trials broken down by reason).
2. Beta-band PSD differs OFF vs ON on ECoG (one figure, 4-session grid).
3. Laplacian LFP PSD differs OFF vs ON (companion figure, same layout) — the
   laplacian signal is what the cross-modal models predict, so verifying a
   condition effect here mirrors the ECoG check.
4. Tracing speed differs OFF vs ON (one figure).
5. Refer to appendix about time step grid alignment (behavioral signal must
   be time aligned with neural, sampling rates differ).
6. Closing sentence: these are the signatures the latent models will be asked
   to recover.

**Figures — every figure emitted by `thesis_sec1_data_verification.ipynb` has
a home in the report (main text or appendix).**

| label | source | new name | location |
|---|---|---|---|
| `fig:trial_counts_bar` | `thesis_figures/sec1/fig_001_trial_count.png` | `trial_count_barplot.png` | main text |
| `fig:trial_distributions` | `thesis_figures/sec1/fig_trial_distributions.png` | `trial_distributions.png` | appendix (pools removal counts per reason across sessions) |
| `fig:psd_ecog_PDI1_S2` | `thesis_figures/sec1/fig_002_psd_PDI1_S2.png` (PDI1_S2, 2×2 of 4 contacts) | `psd_ecog_PDI1_S2.png` | main text subfigure |
| `fig:psd_ecog_PDI1_S4` | `thesis_figures/sec1/fig_003_psd_PDI1_S4.png` | `psd_ecog_PDI1_S4.png` | main text subfigure |
| `fig:psd_ecog_PDI4_S2` | `thesis_figures/sec1/fig_004_psd_PDI4_S2.png` | `psd_ecog_PDI4_S2.png` | main text subfigure |
| `fig:psd_ecog_PDI4_S3` | `thesis_figures/sec1/fig_005_psd_PDI4_S3.png` | `psd_ecog_PDI4_S3.png` | main text subfigure |
| `fig:psd_ecog1_summary` | `thesis_figures/sec1/fig_002_psd_ecog1.png` (ECOG_1 only, 2×2 session view) | `psd_ecog1_summary.png` | appendix (one-contact overview) |
| `fig:psd_comparison_lfp` | `thesis_figures/sec1/fig_003_psd_laplacian_d14.png` | `psd_laplacian_d14.png` | main text |
| `fig:dbs_significance_heatmap` | `thesis_figures/sec1/fig_005_dbs_significance_heatmap.png` | `dbs_significance_heatmap.png` | main text |
| `fig:behav_velocity` | `thesis_figures/sec1/fig_beh_tracing_velocity_x.png` | `behavioral_tracing_velocity.png` | main text |
| `fig:behav_acceleration` | `thesis_figures/sec1/fig_beh_tracing_acceleration_magnitude.png` | `behavioral_tracing_acceleration.png` | main text |
| `fig:grid_alignment_A` | `thesis_figures/sec1/fig_grid_alignment_A.png` | `grid_alignment_A_timeline.png` | appendix |
| `fig:grid_alignment_B` | `thesis_figures/sec1/fig_grid_alignment_B.png` | `grid_alignment_B_density.png` | appendix |
| `fig:grid_alignment_C` | `thesis_figures/sec1/fig_grid_alignment_C.png` | `grid_alignment_C_trajectory.png` | appendix |

The DBS significance heatmap is load-bearing for the narrative: it shows
the dataset has the right frequency content to be separated in the first
place (permutation t-test of DBS-OFF vs DBS-ON PSD, per frequency bin, per
ECoG contact and Laplacian D14). Without this figure the reader can't tell
whether a later null classification result (e.g., on PDI1_S2/S4) reflects a
model limitation or a floor in the signal itself.

**Tables.** `tab:trial_counts` (existing — fill in train+val totals marked TODO).

**Removed trials breakdown (for appendix).** 42 trials total across 4 sessions,
by reason:

| Participant | Session | DBS | Reason | Count |
|---|---|---|---|---|
| PDI1 | 4 | OFF | plateau (>2.0s) | 7 |
| PDI1 | 4 | OFF | protocol/events | 2 |
| PDI1 | 4 | ON | plateau (>2.0s) | 3 |
| PDI1 | 4 | ON | protocol/events | 2 |
| PDI4 | 2 | ON | protocol/events | 1 |
| PDI4 | 3 | OFF | fragmented | 24 |
| PDI4 | 3 | OFF | plateau (>2.0s) | 2 |
| PDI4 | 3 | ON | plateau (>2.0s) | 1 |

Missing blocks per session (absent-from-iEEG reasoning):

| Session | Missing blocks | Reason |
|---|---|---|
| PDI1_S2 | none | all 12 blocks present |
| PDI1_S4 | 11, 12 | no iEEG recording files on disk |
| PDI4_S2 | 1, 2, 3 | no iEEG files; labels exist but no neural recording |
| PDI4_S3 | 7, 9 | present in `participants_2` but `is_fragmented=True`, dropped during split creation |

**Action items.**
- [ ] Fill in train+val totals in trial count table.
- [ ] Copy `fig_001_trial_count.png` into `figures/` as `trial_count_barplot.png`.
- [ ] Copy ECoG per-session PSD panels and assemble as 4-panel grid
      `psd_ecog_4_sessions.png` (or copy each session independently; decide
      during implementation).
- [ ] Copy `fig_003_psd_laplacian_d14.png` into `figures/` as
      `psd_laplacian_d14.png`.
- [ ] Copy `fig_006_tracing_speed.png` into `figures/` as `tracing_speed.png`.
- [ ] Copy `fig_005_dbs_significance_heatmap.png` (or split a/b variants)
      into `figures/` as `dbs_significance_heatmap.png`.
- [ ] Pick representative grid-alignment figure for main text and move
      `fig_008/009/010_grid_alignment_*` to appendix.
- [ ] Add appendix content for the 42 removed trials.
- [ ] Replace closing sentence about RQ3 with a forward reference to latent
      dynamics recovery.
- [ ] Per-figure: write `.todo.md` before coding, write `.md` technical
      description after. Goes under `report/thesis/figures/technical_descriptions/`.

All source figures for 4.1 already exist in `thesis_figures/sec1/` — no sec1
notebook rerun required for this section. Only copying + renaming + caption
writing.

---

## 4.2 Model selection and configuration

**Status:** KEEP `04_01_model_selection.tex`, expand.

**Purpose.** Justify model choices before reporting any model performance.
Reader leaves with: "I trust why he picked these dimensions and these
variants" — for both behavioral and laplacian PSID variants.

**Subsections.**
- **4.2.1 Latent dimensionality (grid search)** — one grid per mode
  (behavioral: `nx ∈ {4,8,15,25}`, `n1 ∈ {2,4}`; laplacian: `nx ∈ {15,25,40,55}`,
  `n1 ∈ {4,8}`), four sessions each, selected via CV balanced accuracy on
  forecast-based classification.
- **4.2.2 Improved-PSID validation** — RTS smoother + eigenvalue rescaling
  beats vanilla on both modes. Small but consistent; justifies what's used
  downstream.
- **4.2.3 DPAD training behaviour** — training curves and convergence for
  behavioral mode. No laplacian subsection in the current run (DPAD has no
  laplacian variant trained yet); thesis will eventually include a laplacian
  DPAD subsection once that training is done, so leave room in the narrative.

**Narrative beats.**
1. Grid search landscape: selected `(nx, n1)` per session per mode, with
   landscape heatmaps showing the chosen cell.
2. Vanilla vs improved PSID — reported for both modes so the improvement
   isn't conditional on the data stream.
3. DPAD training curves: shows the checkpoint used is at a stable point,
   not picked arbitrarily.

**Figures.**

| label | source | new name |
|---|---|---|
| `fig:grid_search_behavioral` | sec2a grid-search heatmap (behavioral) | `grid_search_behavioral.png` |
| `fig:grid_search_laplacian` | sec2a grid-search heatmap (laplacian) | `grid_search_laplacian.png` |
| `fig:psid_vanilla` | `thesis_figures/sec2/fig_037_*` (vanilla vs improved, per mode) | `psid_vanilla_comparison.png` |
| `fig:dpad_training` | `thesis_figures/sec2/fig_020a_*` (DPAD training curves) | `dpad_training_curves.png` |

**Tables.** `tab:selected_dims` — existing, extend to cover both modes.

**Action items.**
- [ ] Confirm both-mode grid-search figures exist in sec2a; if not, add.
- [ ] Copy vanilla comparison figure to `figures/`.
- [ ] Copy DPAD training curves to `figures/`.
- [ ] Pull DPAD data-efficiency analysis to appendix; keep training curves
      in main text.
- [ ] Write short paragraph framing DPAD convergence and checkpoint selection.

---

## 4.3 One-step prediction

**Status:** NEW file `04_02_one_step_prediction.tex` (merges old prediction
parts of `04_02_rq1_neural.tex` and `04_03_rq2_behavioral.tex`).

**Purpose.** Same models, same trials, different output channels. Compare
PSID, DPAD, VARMA on what they reconstruct from one step of context, across
both neural self-prediction (ECoG → ECoG) and cross-modal (ECoG → LFP).

**Subsections.**
- **4.3.1 Neural self-prediction** (behavioral mode: ECoG Y/Yp, all 3 models).
- **4.3.2 Behavioural decoding** (behavioral mode: tracing kinematics Z/Zp,
  all 3 models, both velocity and acceleration).
- **4.3.3 Cross-modal prediction** (laplacian mode: ECoG → LFP Z/Zp, 2 models:
  PSID + VARMA, no DPAD).

**Narrative beats.**
1. *Neural (4.3.1).* RMSE table → model ranking → exemplar trace → spectral-
   band breakdown showing where the prediction quality lives (Fig 22 heatmap).
2. *Behavioural (4.3.2).* Velocity table → acceleration table → exemplar trace
   → pooled comparison. Note where VARMA wins on 1-step (AR ceiling) and
   where it blows up (high-frequency behavioural traces).
3. *Cross-modal (4.3.3).* Per-band LFP heatmap (Fig 23) shows PSID fails at
   ECoG→LFP decoding (r ≈ 0 across all bands) while VARMA hits ~1.0 by 1-step
   AR on its own output. Keep short — a negative result you want visible but
   not central.

**Figures.**

| label | source | new name |
|---|---|---|
| `fig:neural_exemplars` | sec2e Figs 18-21 (4 files → 1 multi-panel) | `neural_exemplars_grid.png` |
| `fig:neural_band_heatmap` | `thesis_figures/sec2/fig_022_neural_band_pearson.png` (+ `_rmse` / `_vaf` variants in appendix) | `neural_band_pearson.png` |
| `fig:behav_exemplar_vel` | sec2e behavioural velocity exemplar (pick representative) | `behavioral_exemplar_velocity.png` |
| `fig:behav_exemplar_acc` | sec2e behavioural acceleration exemplar | `behavioral_exemplar_acceleration.png` |
| `fig:pooled_rmse_vel` | sec2b Figs 7-8 velocity (24-box session-grouped, Pearson or RMSE) | `pooled_rmse_velocity.png` |
| `fig:pooled_rmse_acc` | sec2b Figs 7-8 acceleration | `pooled_rmse_acceleration.png` |
| `fig:laplacian_band` | `thesis_figures/sec2/fig_023_lfp_band_pearson.png` (+ `_rmse` / `_vaf` appendix) | `laplacian_band_pearson.png` |
| `fig:laplacian_exemplar` | sec2e LFP exemplar | `laplacian_exemplar.png` |

**Tables.** `tab:neural_rmse`, `tab:behavioral_rmse_vel`, `tab:behavioral_rmse_acc`
— move into the new file from old `04_02_rq1_neural.tex` / `04_03_rq2_behavioral.tex`.

**Move from Results to Discussion.**
- "VARMA operates directly on observed signals without latent decomposition;
  for one-step-ahead prediction on the same signal it reconstructs, this is
  expected." (current `04_02_rq1_neural.tex:16`) → Discussion.
- "Surface ECoG carries limited information about depth LFP Laplacian signals
  in this dataset" — keep the fact in Results, move the mechanism explanation
  to Discussion.

**Appendix overflow.** Per-session exemplars (3 of 4 sessions for both neural
and behavioural). RMSE + VAF variants of the band heatmaps (only Pearson in
main text).

**Action items.**
- [ ] Create `04_02_one_step_prediction.tex`.
- [ ] Copy figures to `figures/` with clean names.
- [ ] Combine 4 neural exemplars into a single multi-panel figure.
- [ ] Confirm 4.3.3 LFP exemplar available; if not, add to sec2e.
- [ ] Move RMSE tables from old files.
- [ ] Move interpretive sentences to Discussion.
- [ ] Assign remaining exemplars to appendix.

---

## 4.4 Multi-step forecasting

**Status:** NEW file `04_03_forecasting.tex` (merges forecasting parts from
old `04_02_rq1_neural.tex` + `04_03_rq2_behavioral.tex`).

**Purpose.** Same models, same data, but now extrapolating instead of
reconstructing. Headline question: does the one-step ranking survive at
longer horizons, and on both data streams?

**Subsections.**
- **4.4.1 Neural forecasting** (behavioral mode: ECoG future prediction).
- **4.4.2 Behavioural forecasting** (behavioral mode: tracing future).
- **4.4.3 LFP forecasting** (laplacian mode: ECoG → LFP future, 2-model).
- **4.4.4 Forecast checkpoints (degradation curves)** — how initial-condition
  source affects forecast quality.

**Narrative beats.**
1. RMSE vs horizon for neural — show where each model crosses the naive
   baseline. This is where VARMA's 1-step advantage decays, giving PSID/DPAD
   room to win at longer horizons.
2. RMSE vs horizon for behavioural — separately for velocity and acceleration.
3. RMSE vs horizon for LFP — 2-model. PSID catches up to VARMA at long
   horizons once the AR ceiling starts decaying; this is where the laplacian
   story becomes positive.
4. One representative forecast trace per channel, **with ±1 SE shaded band
   around each model's predicted trace** (empirical bootstrap from per-step
   residuals across test trials). Bands give apples-to-apples uncertainty for
   PSID, DPAD, VARMA.
5. Forecast checkpoints: initial-condition source matters, not just the model.

**Figures.**

| label | source | new name |
|---|---|---|
| `fig:neural_forecast_horizon` | sec2d neural forecast horizon | `neural_forecast_horizon.png` |
| `fig:behav_forecast_horizon` | sec2d behavioural horizon (velocity + acceleration combined, 2-panel) | `behavioral_forecast_horizon.png` |
| `fig:laplacian_forecast_horizon` | sec2d LFP horizon (Figs 56-59 combined) | `laplacian_forecast_horizon.png` |
| `fig:forecast_exemplar_neural` | sec2e Figs 29-36 / 50-55 representative neural, **with SE band** | `forecast_exemplar_neural.png` |
| `fig:forecast_exemplar_behav` | sec2e representative behavioural, **with SE band** | `forecast_exemplar_behavioral.png` |
| `fig:forecast_exemplar_lfp` | sec2e representative LFP forecast, **with SE band** | `forecast_exemplar_lfp.png` |
| `fig:forecast_checkpoints` | sec2d checkpoints (neural + behavioural, 2-panel) | `forecast_checkpoints.png` |

**Tables.** None — figures carry the message.

**Appendix overflow.** Remaining 3 neural + 7 behavioural + 3 LFP forecast
exemplars.

**Action items.**
- [ ] Create `04_03_forecasting.tex`.
- [ ] Copy figures to `figures/` with clean names.
- [ ] Combine velocity/acceleration forecast horizon plots into one 2-panel.
- [ ] Combine neural/behavioural forecast checkpoints into one 2-panel.
- [ ] Pick one representative exemplar per channel (neural, behavioural, LFP).
- [ ] **Add shaded SE band to all forecast exemplar figures** (sec2d/sec2e
      code change; empirical pooled-trial residuals).
- [ ] Assign remaining exemplars to appendix.

---

## 4.5 Clinical state classification

**Status:** KEEP old `04_04_rq3_classification.tex`, trim and renumber to
`04_04_classification.tex`.

**Purpose.** Can the latent states discriminate DBS-ON vs DBS-OFF without
ever having seen the label? Pipeline-ceiling check first, then the real test.
Covers both behavioral-mode and laplacian-mode classification (each model
independently produces predictions that can be fed into the LDA).

**Subsections.**
- **4.5.1 Standard classification (prediction-based and forecast-based)** —
  full latent, `X^(1)`, `X^(2)`, `Xp_with_dbs` baseline. Both data streams.
- **4.5.2 Cross-label-based classification** (flipped mode: classify
  forecasted traces on DBS_ON model or DBS_OFF model from the history of the
  DBS_BOTH model — i.e., "which model does this trajectory look like").

**Narrative beats.**
1. Pipeline ceiling: `Xp_with_dbs` → BA = 1.000 across sessions. The pipeline
   works.
2. Real test: full latent / `X^(1)` / `X^(2)` — report BA ± permutation p
   across all sessions and both data streams. Covers prediction-based and
   forecast-based classification. Flag where `X^(1)` ≈ `X^(2)` (subspace
   separation doesn't help).
3. Cross-label classification (4.5.2) — comparable BA, briefly stated. Works
   both for behavioral and laplacian (same machinery).
4. Forecast confidence: m-step forecast traces with ±1 SE RMSE shading. Shows
   prediction certainty aligned with classification — wide bands = uncertain
   trace, narrow = confident. Ties into 4.4's shaded-band figures but framed
   for the classification narrative (how uncertain is the forecast we're
   classifying from?).

**Figures.**

| label | source | new name |
|---|---|---|
| `fig:classification_bar_behavioral` | sec5 `14a_classification_grouped_bar_0.png` (behavioral) | `classification_grouped_bar_behavioral.png` |
| `fig:classification_bar_laplacian` | sec5 grouped bar for laplacian | `classification_grouped_bar_laplacian.png` |
| `fig:classification_flipped` | sec5 `14b_classification_flipped_heatmap.png` | `classification_flipped_heatmap.png` |
| `fig:forecast_confidence` | new — reuse 4.4 forecast exemplars, annotate for classification context | `forecast_confidence.png` |

**Tables.** `tab:classification` — existing; extend with columns for BA +
permutation p per cell per feature source, split into behavioral and laplacian
rows.

**Permutation display.** Each BA cell in both the bar figure and the table is
decorated with `**` (p<0.01), `*` (p<0.05), or `ns` otherwise, from the
Step 2 permutation test output. Where Step 2 has not yet completed for a
cell, the cell shows BA alone and the caption notes "permutation pending".
`pipeline_runs.md` is the source of truth.

**Move OUT of this section.**
- C_y heatmap → 4.6 interpretability.
- C_z heatmap → 4.6 interpretability.
- Latent phase space → 4.6 interpretability.
- ROC curves and standard heatmap → Appendix.

**Move from Results to Discussion.**
- "The behaviourally relevant subspace does not consistently outperform the
  behaviourally irrelevant subspace" — keep finding, move the "this means..."
  interpretation to Discussion.

**Action items.**
- [ ] Rename/copy to `04_04_classification.tex`.
- [ ] Merge forecast-based standard classification results into 4.5.1.
- [ ] Remove C_y / C_z / latent phase space (relocate to 4.6).
- [ ] Generate laplacian-mode grouped bar figure in sec5.
- [ ] Copy both bar figures into `figures/`.
- [ ] Create forecast confidence figure (reuse of 4.4 exemplar with
      classification caption, or new).
- [ ] **Wire permutation p-values into bar figure annotations and table
      columns.** Pull from DPAD Phase 4 Step 2 / PSID Phase 5 output.
- [ ] Move ROC + standard heatmap to appendix.

---

## 4.6 What the models learned (interpretability)

**Status:** NEW file `04_05_interpretability.tex`. This is the section that
does not currently exist as a coherent unit.

**Purpose.** Open up the box. Give the reader a mechanistic view of PSID
(both modes) and DPAD before the discussion has to refer to anything
abstract.

**Subsections.**
- **4.6.1 PSID neural readout (C_y): which channels and bands matter** —
  behavioral mode (ECoG output) and laplacian mode (LFP output) both get
  their own C_y figures.
- **4.6.2 PSID behavioural readout (C_z)** — how latents map to kinematics
  (behavioral mode only; laplacian mode's Z is LFP, covered indirectly via
  C_y).
- **4.6.3 PSID transition matrix (A): time constants and oscillations** —
  eigenvalue scatter on unit disc, time constants and oscillation frequencies
  per session. Both modes — comparable physiological time constants would be
  a coherence check; divergent would be a finding.
- **4.6.4 Latent phase-space geometry** — PSID `x_1`-`x_2` plane and DPAD
  PC1-PC2, with OFF/ON KDE contours. Visual separation motivates 4.7.
- **4.6.5 DPAD: what is and isn't inspectable** — linearisation around
  trajectory mean is the closest analogue; nonlinearity precludes closed-form
  A/C_y/C_z inspection. State the trade-off plainly.

**Narrative beats.**
1. **C_y (ECoG).** `||C_y[:, :n_1]||` across contacts × bands. Beta-band
   weight is the headline.
2. **C_y (LFP).** Same machinery, laplacian mode. Tests whether beta
   dominance transfers to the LFP readout or the subcortical signal lives in
   a different band.
3. **C_z.** `n_1` behaviourally relevant dimensions projected onto `n_z`
   kinematic outputs. One column = one latent dim, signed weights.
4. **A matrix (both modes).** Eigenvalue scatter on the unit disc →
   `τ = -1/log|λ|` for slow modes, `f = arg(λ)/(2π·dt)` for complex pairs.
   Compare sessions within mode; compare modes within session.
5. **Phase space.** PSID and DPAD behavioral latent planes, with OFF/ON KDE
   contours.
6. **DPAD inspectability.** One paragraph.

**Figures.**

| label | source | new name |
|---|---|---|
| `fig:psid_cy_behavioral` | sec5 `21_psid_cy_importance_0.png` (behavioral) | `psid_cy_importance_behavioral.png` |
| `fig:psid_cy_laplacian` | sec5 C_y heatmap for laplacian PSID | `psid_cy_importance_laplacian.png` |
| `fig:psid_cz` | sec5 `22_psid_cz_0.png` | `psid_cz_heatmap.png` |
| `fig:psid_eigs_behavioral` | **new**, behavioral-mode A eigenvalues | `psid_a_eigenvalues_behavioral.png` |
| `fig:psid_eigs_laplacian` | **new**, laplacian-mode A eigenvalues | `psid_a_eigenvalues_laplacian.png` |
| `fig:latent_phase` | sec6/sec7 `13_latent_phase_space_0.png` | `latent_phase_space.png` |

**Tables.** `tab:psid_time_constants` — new, with columns: mode, session,
`n_1`, dominant time constant (ms), oscillation frequency (Hz), spectral
radius. Rows cover 2 modes × 4 sessions = 8 rows.

**Write fresh.** Almost the whole section. Only inherited material: C_y
paragraph and phase-space paragraph from old `04_04`.

**Move from Results to Discussion.**
- "Beta-band features (13-29 Hz) carry the largest weights … matching the
  documented role of beta oscillations in motor-related cortical activity" —
  keep first half, move second half to Discussion.

**Action items.**
- [ ] Create `04_05_interpretability.tex`.
- [ ] Copy C_y / C_z / phase-space figures to `figures/`.
- [ ] **Generate new figures:** PSID A-matrix eigenvalue spectrum for both
      modes. Single script loading the saved PSID model files under
      `results/psid_behavioral_*/model_*` and `results/psid_laplacian_*/model_*`,
      computing eigs, producing two figures.
- [ ] **Generate new table:** time constants + oscillation frequencies, both
      modes × four sessions.
- [ ] Generate laplacian C_y figure in sec5.
- [ ] Move C_y paragraph from old `04_04`.
- [ ] Move latent phase-space paragraph from old `04_04`.
- [ ] Write C_z subsection fresh.
- [ ] Write A-matrix subsection fresh (include both modes).
- [ ] Write DPAD paragraph (short, acknowledging limits of inspection).

---

## 4.7 Cross-condition generalisation

**Status:** KEEP old `04_05_cross_condition.tex`, renumber to
`04_06_cross_condition.tex`, expand.

**Purpose.** Are the structures from 4.6 condition-specific? If yes, that's
the strongest evidence the models have learned something physiologically
meaningful — not dataset artefacts. Evaluated on both data streams for
consistency.

**Narrative beats.**
1. Three models per session per mode (OFF, ON, BOTH). Within-condition vs
   cross-condition RMSE comparison. Both behavioral and laplacian.
2. Cross-block decoding around block boundaries — visual evidence that
   condition-specific models predict their own condition better.
3. A-matrix similarity. Cosine similarity between vectorised A matrices for
   OFF/ON/BOTH, both modes. Low similarity = different learned dynamics.

**Figures.**

| label | source | new name |
|---|---|---|
| `fig:within_cross_boxplot_behavioral` | sec? within-cross boxplot, behavioral | `within_cross_boxplot_behavioral.png` |
| `fig:within_cross_boxplot_laplacian` | within-cross boxplot, laplacian | `within_cross_boxplot_laplacian.png` |
| `fig:cross_block_neural` | `24_cross_block_0.png` | `cross_block_neural.png` |
| `fig:cross_block_behav` | `24_cross_block_1.png` | `cross_block_behavioral.png` |
| `fig:a_similarity_behavioral` | **new**, behavioral-mode A cosine similarity | `psid_a_cosine_similarity_behavioral.png` |
| `fig:a_similarity_laplacian` | **new**, laplacian-mode A cosine similarity | `psid_a_cosine_similarity_laplacian.png` |

**Tables.** `tab:a_cosine_similarity` — new, rows: mode × session. Columns:
OFF-vs-ON cosine similarity, BOTH-vs-OFF, BOTH-vs-ON.

**Action items.**
- [ ] Rename/copy to `04_06_cross_condition.tex`.
- [ ] Verify within-cross boxplots exist for both modes; add if missing.
- [ ] Copy cross-block figures to `figures/`.
- [ ] **Generate new figures:** A-matrix cosine similarity for both modes.
      Shares the A-loading script from 4.6; one additional pass to compute
      cosine on vectorised A matrices.
- [ ] **Generate new table:** cosine similarity values, both modes × four
      sessions.
- [ ] Write A-matrix similarity paragraph.
- [ ] Reframe section intro to connect back to 4.6 interpretability.

---

## 4.8 Summary of findings

**Status:** REWRITE old `04_06_summary.tex`, renumber to `04_07_summary.tex`.

**Purpose.** Track the new section order (4.1-4.7), not the RQ order. RQs
become inline annotations.

**Structure (one paragraph per bullet):**
- **Data and selection.** Dataset balanced; OFF/ON differences in both ECoG
  and LFP; selected `nx ~ 4-25` (behavioral) / `15-40` (laplacian); improved
  PSID > vanilla in both modes.
- **One-step prediction.** Ranking on neural, ranking on behaviour, where
  VARMA wins (1-step AR ceiling). Cross-modal (4.3.3) is a negative result:
  ECoG does not carry enough information to decode Laplacian LFP through a
  latent bottleneck. (RQ1, RQ2)
- **Forecasting.** Headline horizon at which each model crosses the naïve
  baseline. Bands widen with horizon — shows where uncertainty overtakes
  signal. (RQ1, RQ2)
- **Classification.** Pipeline ceiling 1.000; latent-only BA 0.53-0.67 with
  permutation p-values; `X^(1)` and `X^(2)` comparable; both data streams
  show the same pattern. (RQ3)
- **Interpretability.** `C_y` beta-dominated (ECoG); `C_y` LFP pattern
  [narrative depends on result]; `A` time constants in physiologically
  plausible range for both modes. (cross-cuts all RQs)
- **Cross-condition.** Condition-specific dynamics; `A` matrices differ
  measurably between OFF and ON in both modes.

**Action items.**
- [ ] Rewrite to follow section order 4.1-4.7.
- [ ] Add RQ annotations inline rather than as headings.

---

## Cross-cutting LaTeX hygiene

- [ ] Replace every `\textcolor{red}{[UPDATE: ...]}` left over from Methods.
- [ ] Resolve every `% TODO: export ...` comment before tagging the section
      as done.
- [ ] Cross-reference all the new labels: `\ref{fig:psid_eigs_behavioral}`,
      `\ref{fig:psid_eigs_laplacian}`, `\ref{tab:psid_time_constants}`,
      `\ref{fig:a_similarity_behavioral}`, etc.

---

## Updated `04_results.tex` input list

```latex
\subsection{Dataset and Preprocessing Verification}
\label{subsec:data_results}
\input{sections/04_results/04_00_data}

\subsection{Model Selection and Configuration}
\label{subsec:model_selection_results}
\input{sections/04_results/04_01_model_selection}

\subsection{One-Step Prediction}
\label{subsec:one_step_results}
\input{sections/04_results/04_02_one_step_prediction}

\subsection{Multi-Step Forecasting}
\label{subsec:forecasting_results}
\input{sections/04_results/04_03_forecasting}

\subsection{Clinical State Classification}
\label{subsec:classification_results}
\input{sections/04_results/04_04_classification}

\subsection{Model Interpretability}
\label{subsec:interpretability_results}
\input{sections/04_results/04_05_interpretability}

\subsection{Cross-Condition Generalisation}
\label{subsec:cross_condition_results}
\input{sections/04_results/04_06_cross_condition}

\subsection{Summary of Findings}
\label{subsec:results_summary}
\input{sections/04_results/04_07_summary}
```

---

## New figures to generate

All the figures below do not exist in the current `thesis_figures/` tree.
Grouped by shared upstream script.

**Saved PSID model files (behavioral + laplacian, per-condition variants).**
Shared loader for items 1-4. One pass reads every PSID model artefact under
`results/psid_*behavioral*/model_*` and `results/psid_*laplacian*/model_*`,
extracts `A`, `Cy`, `Cz`.

1. `psid_a_eigenvalues_behavioral.png` (4.6.3) — eigenvalue scatter + time
   constants, 4-panel grid per session, behavioral mode.
2. `psid_a_eigenvalues_laplacian.png` (4.6.3) — same, laplacian mode.
3. `psid_a_cosine_similarity_behavioral.png` (4.7) — OFF/ON/BOTH cosine
   similarity, behavioral mode.
4. `psid_a_cosine_similarity_laplacian.png` (4.7) — same, laplacian mode.

**Data / preprocessing.**

5. `time_grid_alignment.png` (4.1) — behavioural signal resampling to neural
   grid (one-off preprocessing visualization).
6. `psd_laplacian_4_sessions.png` (4.1) — Laplacian-14-16 LFP PSD comparison
   per session, DBS-OFF vs DBS-ON, matching the ECoG PSD layout.

**Classification / forecast narrative.**

7. `forecast_confidence.png` (4.5) — forecast exemplar with ±1 SE band and
   classification annotation. Reuse 4.4 exemplar machinery. not like one but we want all the forecast plots to show the confidence shade.

**Sec2 shaded-band update.**

8. 4.4 forecast exemplars (neural + behavioural + LFP) **regenerated with
   ±1 SE bands** around each model's predicted trace. Existing figure
   infrastructure; new code in sec2d/sec2e to compute and add the band.

---

## Notebook dependencies

How the `thesis_sec*.ipynb` notebooks feed the report sections:

| Notebook | Feeds report section(s) | Needs rerun? |
|---|---|---|
| `thesis_sec1_data_verification` | 4.1 | +laplacian PSD panels, time-grid figure |
| `thesis_sec2a_diagnostics` | 4.2 | verify both modes present |
| `thesis_sec2b_behavioral` | 4.3.2 | OK |
| `thesis_sec2c_neural_recon_group` | 4.3.1 + 4.3.3 (laplacian heatmap) | done |
| `thesis_sec2d_neural_forecast_group` | 4.4 | add shaded SE bands |
| `thesis_sec2e_exemplars` | 4.3 + 4.4 exemplars | add shaded SE bands |
| `thesis_sec5_classification` | 4.5 | add laplacian bar, wire perm p-values |
| `thesis_sec6_summary_appendix` | 4.6 + appendix | verify laplacian coverage |
| `thesis_sec7_subspace_dynamics` | 4.6 interpretability | A-matrix figures+tables |
| `thesis_sec8_per_trial_heatmaps` | Appendix only | — |

All sec2 notebooks have the cwd-robust import patch applied.

`pipeline_runs.md` is the source of truth for which pipeline outputs are
complete; consult it before any figure regeneration.

---

## Pipeline dependencies (external to notebooks)

- **DPAD Phase 4 classification** still running (as of session start). Feeds
  4.5 figures. Expected completion: Sunday. Sec5 figures use `show_dpad=False`
  fallback for cells not yet complete.
- **PSID Phase 5 classification** complete for all 4 cells.
- **Permutation tests (Step 2)** arrive after each cell's Step 1 completes;
  triggers perm-p annotation in 4.5 figures.

---

## Order of attack (integrated)

1. **Thesis-wide conventions** — apply font hierarchy + default line widths
   across `thesis_style.py` / sec2 notebooks. Quick win; no new figures,
   style pass only. Lands in sec2a-e reruns automatically.
2. **4.1 + 4.2** — mostly existing, fastest wins. Promote appendix figures,
   clean filenames, add laplacian PSD + time-grid alignment.
3. **4.3 + 4.4 pair** — biggest content moves. Share prediction-vs-forecast
   framing. 4.4 requires sec2d/sec2e shaded-band code change first, then
   regenerate forecast exemplars.
4. **4.5** — trim C_y/C_z out, add laplacian bar, wire permutation p-values
   (as perm tests land from Step 2).
5. **4.6** — new A-matrix eigenvalue script (both modes). Tackle the
   figure-generation script first, then write the section around it.
6. **4.7** — A-matrix cosine similarity. Same A-loader script as 4.6, one
   additional output pass.
7. **4.8** — last, once everything above is stable.

---

## Open questions / pending decisions

- **Q5 (forecast confidence method) — RESOLVED** to option (C): empirical
  ±1 SE from pooled per-step residuals across test trials. Applied identically
  to PSID, DPAD, VARMA so the three models compare apples-to-apples.
- **Q6 (A-matrix script scope) — RESOLVED** to option (b): cover both
  behavioral and laplacian PSID models for 4.6/4.7 "both sides" story.
- **Open:** which representative exemplars are picked for 4.3/4.4 per-channel
  main-text figures (the rest go to appendix). To decide during
  implementation.
- **Open:** whether the forecast confidence figure in 4.5 is a reuse of 4.4's
  exemplar or a dedicated figure. To decide when implementing.
- **Open:** `time_grid_alignment.png` source — new preprocessing visualization
  or source from existing pipeline diagnostics.