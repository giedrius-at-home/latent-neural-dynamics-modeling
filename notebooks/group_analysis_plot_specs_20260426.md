# Group analysis plot specs — A + B — 2026-04-26

## Data-flow correction (applies everywhere)

Models output **Z only**. No Y→Y self-reconstruction in the thesis.
- behavioral: Z = `tracing_velocity_x`, `tracing_acceleration_magnitude` (2 vars)
- laplacian: Z = top-8 LFP laplacian 14-16 across narrow bands

All A1-B3 plots, all forest/scatter axes, all heatmaps below operate on
**Z reconstruction or Z forecast**. Where original sec2c spec said "Y/Yp"
or "ECOG self-reconstruction", read as Z metric instead.

Per-cell channel-of-interest convention (figs 39-43 family + new plots):
**per-cell best Z channel = Z column where PSID achieves highest test r**.
Recompute per cell; do not fix across cells.

Sec2e exemplars: depict best AND worst Z channel reconstruction
(per-cell each). New panel layout below.

## Stale figures to drop

- Fig 22 (ECoG neural per-band Y/Yp band heatmap, behavioral mode):
  drop — Y self-reconstruction is not a thesis result.
- Fig 23 (laplacian LFP per-band Z reconstruction): keep — already on Z.


Six new plot types, each shipped to sec2c (reconstruction) and sec2d
(forecast), plus three additions to sec8. All matplotlib + thesis_style.
Weak-effect framing throughout (CIs, no significance stars).

Color rules (existing):
- `COLOR_PSID`, `COLOR_DPAD`, `COLOR_VARMA`
- `COLOR_DBS_OFF`, `COLOR_DBS_ON`
- `COLOR_NS` for null/missing markers

Cell ordering: PDI1_S2, PDI1_S4, PDI4_S2, PDI4_S3 (canonical).

---

## A1. Forest plot per cell × model

### What it shows

Per-cell mean metric for each model, with 95% bootstrap CI bars. Pooled
random-effect estimate at bottom. Survives weak effects: CI width =
"trust this number".

### Layout

- Figure: 2 panels side-by-side (DBS-OFF, DBS-ON). Same y-axis.
- Each panel: 25 rows.
  - 24 rows = (8 cells × 3 models). Block-grouped by cell with thin
    horizontal separator lines between cells.
  - Row 25 (offset visually) = pooled across cells, one per model.
- y-axis: cell labels grouped on left (e.g. `PDI1_S2  ─ PSID / DPAD / VARMA`).
- x-axis: metric (Pearson r 0–1, or RMSE z-units).
- Marker: filled circle (PSID), open circle (DPAD), open square (VARMA),
  matched colors. Size = 6.
- CI bars: horizontal segment at the same y, full color, lw=1.5.
- Pooled rows: larger marker (size=10), bold colour, dashed CI bar to
  visually separate from per-cell rows.
- Vertical reference: chance line if Pearson (`r=0`) drawn dotted gray.

### Data flow

Per cell × model × DBS × split = test:
- Load all per-trial metric values across the cell's trials.
- Bootstrap 1000x trial-resample → mean ± 2.5/97.5 percentile.
- Pooled = inverse-variance weighted mean over cells (DerSimonian-Laird
  or simple `1/SE²` weight); CI from random-effect SE.

### Title text

`Per-cell {metric} forest, {DBS}. Bootstrap 95% CI; pooled (RE) at bottom.`

---

## A2. Within vs cross-condition scatter

### What it shows

Per cell per model: how much worse the model gets when DBS state flips at
test time. Generalization gap.

### Layout

- 1×3 subplot row (one per model: PSID, DPAD, VARMA).
- Each panel: x = within-condition test RMSE, y = cross-condition RMSE.
- 16 points = 8 cells × 2 directions (off-trained→on-test, on-trained→off-test).
- Color: cell (categorical 8-color map; or stick with cell-name annotations).
- Shape: triangle-up = off→on, triangle-down = on→off.
- Identity line `y = x` thin gray, dashed.
- Padding: shared axis limits across panels (data range × 1.1).
- Legend (loc="best"): direction shapes only.

### Annotation

Per panel: top-right corner `mean Δ = +X.XX (±SEM)` where Δ = cross − within.

### Data flow

Per cell × model × direction:
- Within = mean trial RMSE in `_dbs_{cond}` test parquet (matching trained cond).
- Cross = mean trial RMSE in `_dbs_{cond}_eval_{other}` parquet.
- One point per (cell, direction).

### Missing-data rule

If cross-eval parquet missing for a (cell × model) → render `×` marker at
y=axis-max with annotation `n/a`. Don't drop the point silently.

---

## A3. DBS-OFF vs DBS-ON paired-trial scatter

### What it shows

Per-trial agreement between DBS conditions when the same model predicts
within its training pool. Tests whether DBS effect is per-trial reshuffling
or systematic shift.

### Layout

- 4×3 subplot grid: rows = cells (PDI1_S2 … PDI4_S3), cols = models.
- Each panel: x = mean Pearson r per OFF trial, y = mean Pearson r per ON
  trial. Per-trial here = mean over output channels.
- Reference: identity line `y = x` thin gray.
- Marker: circle, alpha 0.6, size 18; color = COLOR_DBS_OFF for points
  below identity, COLOR_DBS_ON for points above (so you can read the
  asymmetry).
- Shared x/y limits per row (so cells with low r don't get stretched
  comparison).

### Annotation

Per panel top-left: `n_off=A  n_on=B  Δ=mean(on)-mean(off)=+x.xx`.

### Note on pairing

Trials don't have a 1:1 correspondence across DBS blocks. Two options:
1. Plot OFF trials and ON trials as a *cloud*, no actual pairs (cleaner;
   recommended).
2. Bin by trial-position-in-block, pair by bin — risks artifacts.

Use option 1 with the title "OFF vs ON trial r distribution" so readers
don't assume pairing.

---

## B1. Per-trial Pearson r vs RMSE

### What it shows

Trial-level metric agreement: do per-trial r and RMSE rank trials the same?
If yes (high |ρ|), the metrics are redundant and one is enough.

### Layout

- 2×3 subplot grid: rows = (DBS-OFF, DBS-ON), cols = models.
- Each panel: x = trial Pearson r, y = trial RMSE (z).
- One point per trial across all 8 cells. Color = cell (8 categorical
  colors).
- Translucent (alpha 0.4); marker size 12.
- Inset top-right: Spearman ρ between r and RMSE for that panel.

### Title

`Trial-level metric agreement, {DBS}. ρ=… per model.`

### Why useful

If ρ ≈ −1 → reporting both r and RMSE doubles the figure space without
new signal. Frees up real estate elsewhere.

---

## B2. PSID vs DPAD per-trial r scatter

### What it shows

Do PSID and DPAD predict the same trials well? If yes, framework choice
doesn't matter for trial-level performance.

### Layout

- 2×4 subplot grid: rows = (DBS-OFF, DBS-ON), cols = cells.
- Each panel: x = PSID per-trial mean r, y = DPAD per-trial mean r.
- Identity line.
- Marker: circle, alpha 0.7, size 14, COLOR_DBS_{condition}.
- Title per panel: cell + Pearson ρ between PSID-r and DPAD-r over trials.

### Why both PSID and DPAD only (skip VARMA)

Adds 8 extra panels otherwise; VARMA-vs-PSID adds little signal
(memory: VARMA AR-baseline near-perfect on its own outputs, not
informative for cross-model agreement). If desired, mirror panel for
PSID vs VARMA in appendix.

---

## B3. Reconstruction r vs forecast r at h=1s

### What it shows

How much information persists from one-step recon to one-second forecast.
Cells where forecast collapses to mean (low r) despite OK recon →
multi-step prediction is lossy.

### Layout

- 1 panel, square.
- x = mean recon r per cell × model (across trials, DBS=both).
- y = mean forecast r at horizon=1.0 s per cell × model.
- 24 points = 8 cells × 3 models.
- Shape: cell symbol (`o,s,^,D,p,h,v,X` for the 8 cells).
- Color: model.
- Identity line (forecast ≤ recon expected; deviations notable).
- Annotation per quadrant: `recon high & forecast low → multi-step lossy`,
  `both low → cell uninformative`, etc.
- Range x∈[0, 1], y∈[0, 1].

### Title

`One-step vs 1s-ahead Pearson r per cell × model`.

---

## Sec2e exemplars — best + worst Z channel

Replace Figs 50-55 (or extend) with per-cell exemplar showing both:
- top row: trial with **highest** PSID Z r (best-case reconstruction)
- bottom row: trial with **lowest** PSID Z r (worst-case reconstruction)

Layout: 4×4 grid. 4 rows = cells. Cols: best-OFF, best-ON, worst-OFF,
worst-ON. Each panel: trial time series with Z true (gray), Zp PSID
(blue), Zp DPAD (orange), Zp VARMA (red). Title: cell + DBS + r value.

Channel selection: per-cell best Z channel (rule above). Worst channel
panel uses same best-channel pick (so we're showing the variability of
the same channel, not switching channels), unless Giedrius prefers the
genuine worst-channel per cell — flag this decision before coding sec2e.

## Sec8 additions (S1, S2, S3)

### S1. Cross-session electrode×band Pearson r heatmap

- 4×15 heatmap (electrodes × bands).
- Each cell = mean trial r across all 4 sessions.
- Stippling overlay (`/`) on cells where bootstrap 95% CI excludes 0.
- 2-panel: DBS-OFF / DBS-ON.
- Already-existing fig may partially do this (check after sec8 re-run).

### S2. Per-trial × per-channel stripe (PSID-only, 4 sessions concatenated)

- Long axis: trials concatenated across sessions (vertical lines mark
  session boundaries).
- y-axis: 60 channels.
- Color: trial-channel Pearson r.
- Stripes give immediate read of "are bad trials cell-specific or
  systematic" without per-session decomposition.

### S3. PSID vs DPAD per-trial r scatter

- 2×2 subplot per session.
- Same structure as B2 but PSID-only sec8 currently — adds DPAD load.
- Lower priority; defer if sec8 time-budget tight.

---

## Existing helpers to reuse

In `thesis_sec2_common.py`:
- `_boxplot_colored`, `mpl_rmse_boxplot` — keep for original boxes
- `mpl_raincloud`, `mpl_ecdf` — keep for distribution detail
- `mpl_block_running` — keep for temporal trace

Add to `thesis_sec2_common.py`:
- `mpl_forest_per_cell_model(...)` — A1
- `mpl_within_vs_cross(...)` — A2
- `mpl_off_vs_on_pair(...)` — A3
- `mpl_metric_agreement_scatter(...)` — B1
- `mpl_model_agreement_scatter(...)` — B2
- `mpl_recon_vs_forecast(...)` — B3 (lives in 2c+2d shared utility)

Each ~50–80 lines. Together ~400 lines of new helper code.

---

## Figure numbers (proposed)

Reusing the unallocated 60s + 70s slots:
- sec2c: 68 forest, 69 within-vs-cross, 70 off-vs-on, 71 r-vs-RMSE,
  72 PSID-vs-DPAD, 73 recon-vs-forecast (also referenced from sec2d).
- sec2d: 74 forest, 75 within-vs-cross, 76 off-vs-on, 77 r-vs-RMSE,
  78 PSID-vs-DPAD.
- sec8: 79 cross-session heatmap, 80 stripe, 81 PSID-vs-DPAD per-trial.

Final numbering subject to where Giedrius wants them in chapter order.

---

## Missing-data policy (model-aware None)

Per Giedrius: "make the plots as if they exist there but we can add None
if they don't exist". Concrete rules:

1. Forest plot (A1): if a (cell × model) has no trials, draw a hollow
   marker at the model's expected x position with no error bar; annotate
   row label with `(n=0)`. Pooled estimate omits that cell.
2. Scatter plots (A2/A3/B1/B2/B3): missing point → render `×` glyph at
   axes corner with text `n/a` in COLOR_NS gray; do not skip the slot.
3. Heatmaps (S1): missing cell → COLOR_NS gray fill, label `n/a`.

This way figure layout is invariant under data arrival; once DPAD lap
PDI4_S3 lands tonight, the placeholder × becomes a real point.

---

## Order of build

1. Implement A1 first (forest); reuse for both sec2c + sec2d.
2. Then A2 (within-vs-cross). Loads cross-eval parquets; biggest data wiring.
3. Then A3 (off-vs-on cloud).
4. Then B-series in any order.
5. Sec8 S1 last (smallest delta over existing summary).

Each gets `todo.md` before code + `tech.md` after, per per-figure workflow
memory.
