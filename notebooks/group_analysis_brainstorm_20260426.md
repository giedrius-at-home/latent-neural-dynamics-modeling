# Group analysis brainstorm — sec2c/2d/sec8 — 2026-04-26

Scope: prediction (sec2c) + forecast (sec2d) + per-trial heatmaps (sec8).
Constraint: weak-effect regime — favor honest variability over significance.
Three frameworks (PSID/DPAD/VARMA), 4 cells × 2 modes = 8 cells.

## What sec2c/2d already have (after Apr 25 expansion)

- `mpl_rmse_boxplot` — per-cell box (8 boxes per metric)
- `mpl_session_grouped_boxplot` — same with grouping
- `mpl_raincloud` — distribution detail
- `mpl_ecdf` — cumulative shape
- `mpl_block_running` — temporal running stat
- `mpl_cascade_train_val_test` — split comparison

All are **per-cell distribution displays**. Show "what each cell looks like".
None answer cross-cell or cross-metric questions.

## Gap

Group analysis = aggregate or relate **across cells**. Currently absent.
Existing figures show 8 boxes side by side; the reader is asked to integrate
visually. Problems:

1. No quantitative summary that survives weak effects (box overlaps =
   "effect ambiguous" but no number for *how* ambiguous).
2. No way to see if a cell that ranks high on metric A also ranks high on
   metric B (cross-metric structure).
3. No way to see if a model that wins on cell X wins on cell Y (cross-cell
   structure).
4. Within-vs-cross condition gap is reported by box but not by per-trial
   linkage.

---

## Approach A — "Forest + paired scatter" (minimal, recommended)

Add 3 figure types per section. Each is one or two panels of matplotlib.

### A1. Forest plot per cell × model

- y-axis: 24 rows = 8 cells × 3 models (PSID/DPAD/VARMA)
- x-axis: metric (Pearson r or RMSE)
- Each row: point estimate (cell × model mean over trials) + bootstrap 95% CI
- Bottom row: pooled estimate (random-effect mean) with its CI
- Separate panels for OFF vs ON DBS

**Why:** survives weak effects — CI width tells you "is this number reliable",
overlapping CIs are visually obvious. Forest is the standard meta-analysis
display; treats cells as studies. Pooled row gives single number for the
abstract.

### A2. Within vs cross-condition scatter

- One panel per model. x = within-condition test RMSE, y = cross-condition
  RMSE (off model on on data, etc.). Identity line.
- One point per cell. Color = cell. Shape = direction (off→on vs on→off).
- Read: points above identity = generalization gap.

**Why:** answers "how much worse is the model when DBS state flips?" with a
single visual. Cross-eval parquets already exist on disk.

### A3. DBS-OFF vs DBS-ON paired trial scatter

- One panel per cell × model. x = mean Pearson r in OFF block, y = mean in ON
  block. Identity line.
- Each point = one trial pair (block-aligned).
- Read: deviation from identity = within-cell DBS sensitivity.

**Why:** answers "does individual-trial performance flip with DBS?" Tests
whether group-level DBS effect is from systematic shift or trial reshuffling.

---

## Approach B — A + cross-metric / cross-model scatter (moderate)

Add to A:

### B1. Cross-metric scatter (per trial)

- x = trial-level Pearson r, y = trial-level RMSE, color by cell, shape by
  model. One scatter per (recon vs forecast).
- Spearman ρ in title.

**Why:** if ρ near -1, the metrics are redundant (just report one). If far
from -1, the trial-level rankings disagree → which trials are "good" depends
on metric.

### B2. Cross-model agreement scatter

- x = PSID per-trial Pearson r, y = DPAD per-trial Pearson r. Identity line.
- One panel per cell. Off-diagonal points = model disagreement.
- Repeat for PSID vs VARMA.

**Why:** if all models agree per-trial, framework choice is moot in the
weak-effect regime. If models disagree, ensemble or select per-trial may
help. Likely answer: high ρ → frameworks roughly equivalent.

### B3. Reconstruction vs forecast scatter (per cell)

- x = mean reconstruction r (sec2c), y = mean forecast r at h=1s (sec2d).
  One point per cell × model.
- Identity line (forecast ≤ reconstruction is the expectation; deviations
  are notable).

**Why:** quantifies "how much information is lost going from one-step to
multi-step". Reveals whether forecast collapses to mean (low r) on cells
where recon was OK.

---

## Approach C — A + B + hierarchical effect estimate (heaviest)

Add to B:

### C1. Bayesian random-effect model

For each (metric × dbs × split), fit:
  `metric ~ model + (1 | cell)`

with `pymc` or `statsmodels.MixedLM`. Report fixed-effect difference
PSID − DPAD with 95% CI. Plot the posterior.

**Why:** gives the single number Giedrius needs in the abstract: "across 8
cells, PSID outperforms DPAD by X (95% CI [a, b])". Honest under weak
effects: CI width tells the truth about pooled signal.

**Cost:** one new helper file, ~150 lines. Adds `pymc` or relies on existing
`statsmodels`. Approach B already gives this visually via the forest plot;
C makes it a number.

---

## Sec8 group-analysis additions

Sec8 is per-trial heatmaps for ONE PSID model per session. Currently
PSID-only (no DPAD/VARMA overlay). Group-analysis ideas:

### S1. Cross-session heatmap collapse

- Aggregate the per-(electrode, band) Pearson r across all 4 sessions
  (mean, SEM). Plot 4×15 heatmap (electrodes × bands). Mark cells where
  CI excludes 0.
- Existing summary fig at `# ## Summary: Mean Pearson R by electrode and band`
  may already partially do this — verify after sec8 re-run.

### S2. Per-trial × per-cell stripe

- Concatenate all trials across all 4 sessions into one long axis. y =
  60 channels. Color = Pearson r. Vertical lines mark session boundaries.
- Reveals whether failure modes are cell-specific or systematic.

### S3. PSID vs DPAD per-trial scatter

- For each session, x = PSID per-trial mean r, y = DPAD per-trial mean r.
  Identity line.
- Sec8 currently PSID-only; this requires loading DPAD predictions too.

---

## Concrete scatter-plot menu (Giedrius asked)

In rough order of expected signal:

| # | Scatter | Asks | Effort |
|---|---|---|---|
| 1 | Within-cond RMSE vs cross-cond RMSE | DBS generalization gap | low |
| 2 | OFF-block vs ON-block per-trial r | DBS within-cell sensitivity | low |
| 3 | Recon r vs forecast r at h=1s | info loss going to multi-step | low |
| 4 | PSID r vs DPAD r per trial | framework agreement | mid |
| 5 | Pearson r vs RMSE per trial | metric redundancy | low |
| 6 | nx vs cell-mean r | does latent dim help? | low |
| 7 | Mean per-channel r vs mean per-trial r | which axis owns variance? | mid |
| 8 | DPAD epoch budget vs r | training-curve plateau | mid |

---

## Recommendation

**Approach A** if Giedrius wants minimum new code with maximum honesty
gain. **Approach B** if scatter-heavy display is desired ("scatter plots
too" was explicit). **Approach C** only if abstract needs a single
pooled-effect sentence.

For the thesis, **B** seems best balance: 6-7 new figures total, all are
~50 lines each in `thesis_sec2_common.py`, builds on existing helpers,
gives honest weak-effect summary AND new visual material.
