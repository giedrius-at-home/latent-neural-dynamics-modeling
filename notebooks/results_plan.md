# Results Chapter — Production Plan (Deadline: Friday 10am)

## Time windows
- Thursday morning: before ~9am (~2-3h) — covers sec1 + sec2c start
- Thursday work breaks: 4 × 15 min slots
- Thursday evening: 18:00–23:00 (~5h)
- Friday morning: 7:00–10:00 (3h) — writing only, no new figures

## Agent instructions

When picking up this plan:
- One task at a time. Show the specific file + cell to touch, not a paragraph of context.
- After each fix: show the diff or the exact line changed, then stop and wait.
- If something looks wrong in the data or figure, flag it immediately with `DISCUSS:` in the session log — don't silently work around it.
- Prefer asking one short question over making an assumption.
- Fast, clear, checkable. Each response = one completed checkbox.

---

## Priority order
1. **sec2c** — laplacian reconstruction (Q1.2)
2. **sec2d** — forecast (Q2)
3. **sec5** — DBS classification (Q3) + fig_053 t_cut
4. **sec6** — target choice (Q4)
5. **sec2e** — exemplars (nice to have)
6. **sec1** — data overview (tonight)
7. sec2a / sec2b — **discard / low priority**

## Running log

All changes, fixes, and data discoveries go in one shared file:
**`notebooks/results_session_log.md`**

Format per entry:
```
## [date] [notebook]
- Fixed: [figure] — [what was wrong] → [what changed]
- Found: [observation about data or code]
- DISCUSS: [anything to go through together before finalising]
```

Keep one section per work session. After each session, we review `DISCUSS:` items together and go through the plots.

---

## Context

Most figures exist in `thesis_figures/` and `report/thesis/figures/`.
Main work = display polish (remove titles, fix axis labels, fix legends, spacing) + bullet points.
No new model runs. DPAD partial (PDI1_S2 behavioral only) — `has_dpad_data` fallback already wired; blank panels auto-fill when jacque results arrive.

[NEW] figures are **stretch goals** — add on the go if time allows:
- `fig_spectral_separability` — new cell in sec1/sec2a
- `fig_B` — amplitude envelope correlation, new cell in sec2c
- `fig_C` — band-limited forecast, new cell in sec2d
- `fig_forecast_decay` — windowed decay, new cell in sec2d

---

## TONIGHT (Wed) — Sec1 data overview

**Safe stop after any numbered item.**

- [ ] 1. Open `notebooks/thesis_sec1_data_verification.ipynb`
- [ ] 2. PSD plots (fig_002–005): remove figure-level title; y = "PSD (dB/Hz)", x = "Frequency (Hz)"
- [ ] 3. Behavioral traces (fig_beh_*): y = "z-score"; no internal column names
- [ ] 4. DBS significance heatmap: colorbar = "-log₁₀(p)"; session/channel labels readable
- [ ] 5. Trial / split counts: integer bar labels, no axis clutter
- [ ] 6. Grid alignment: axis titles informative
- [ ] 7. Rerun all sec1 cells → verify PNGs in `thesis_figures/sec1/`
- [ ] 8. Write 3–5 bullet points: trial counts, PSD separation, DBS modulation strength

**Stretch:**
- [ ] `[NEW] fig_spectral_separability` — per-channel per-band DBS modulation, new cell

---

## THURSDAY MORNING — Before 8am (~2h)

Focus: sec2c + start sec2d display pass.

### sec2c — neural reconstruction (Q1.2)
- [ ] Open `notebooks/thesis_sec2c_neural_recon_group.ipynb`
- [ ] fig_070 per-cell box: y = "Pearson r" / "RMSE (z-score)"; no internal channel name
- [ ] fig_072 raincloud: legend = clean model names; colors = PSID blue / VARMA gray / DPAD red
- [ ] fig_071–075: consistent labels throughout
- [ ] Rerun changed cells
- [ ] 2–3 bullet points for Q1.2

**Stretch:**
- [ ] `[NEW] fig_B` — Hilbert envelope correlation on Zp + Z, new sec2c cell

### sec2d — forecast start (Q2)
- [ ] Open `notebooks/thesis_sec2d_neural_forecast_group.ipynb`
- [ ] fig_086 horizon: x = "Forecast horizon (s)", y = "RMSE (z-score)" / "Pearson r"
- [ ] fig_083/084 forecast box/raincloud: y labels match sec2c equivalents
- [ ] Rerun → save

---

## THURSDAY BREAKS — 4 × 15 min

**Break 1 — sec2d deeper**
- [ ] LFP per-session forecast figs 056–059: subplot titles = "PDI1_S2 - theta 4–8 Hz" style
- [ ] fig_064–067 raincloud/ECDF/cascade: axis labels, legend entries
- [ ] fig_080–082 laplacian forecast: labels match sec2c

**Break 2 — sec5 classification first pass**
- [ ] Open `notebooks/thesis_sec5_classification.ipynb`
- [ ] fig_050 heatmap: colorbar = "Balanced accuracy"; feature names readable (Xp, Xp_1, Xp_2, Xp_DBS)
- [ ] fig_051 flipped: subtitle = "flipped labels"; fig_052 ROC: x = "FPR", y = "TPR"
- [ ] Rerun → save

**Break 3 — sec5 cross-block / checkpoints**
- [ ] fig_055–058 cross-block / forecast checkpoints: y = "Balanced accuracy"; informative subplot titles
- [ ] Check all sec5 for any remaining title/label issues

**Break 4 — sec6 target choice**
- [ ] Open `notebooks/thesis_sec6_summary_appendix.ipynb`
- [ ] fig_059 phase KDE: axes = "Latent dim 1/2"; legend = DBS ON / DBS OFF
- [ ] fig_060: colorbar = "Balanced accuracy"; fig_063: x = "Training data fraction"
- [ ] fig_061–062 Cy/Cz: colorbar label present
- [ ] Rerun → save

---

## THURSDAY EVENING — 18:00–23:00 (~5h)

Main production block. Safe to stop between tasks.

### Task 1 — sec2d complete + [NEW] figures (45 min)
- [ ] Any remaining sec2d display fixes
- [ ] 2–3 bullet points for Q2: degradation rate, VARMA baseline vs PSID/DPAD gap
- **Stretch:**
- [ ] `[NEW] fig_C` — band-limited forecast quality (Zf vs Z), new sec2d cell
- [ ] `[NEW] fig_forecast_decay` — RMSE/r in sliding windows, new sec2d cell

### Task 2 — sec5 fig_053 t_cut + bullet points (1h)
- [ ] `fig_053_tcut_analysis.png` — BA vs t_cut (s), one line per session, dotted cv_ba reference
  - Source: `results/psid/.../classification/sweep_20260513_053451.parquet`
  - New cell in sec5; x = t_cut [0.5–9 s], y = balanced accuracy
- [ ] 3–4 bullet points: PDI4 above chance, PDI1 at floor, Xp_1 vs Xp_2, honest weak-effect framing

### Task 3 — sec6 deeper + bullet points (30 min)
- [ ] Confirm sec6 figures clean from break 4
- [ ] 2–3 bullet points for Q4: target choice effects, Cy/Cz loading, data efficiency plateau

### Task 4 — sec2e exemplars (30 min)
- [ ] Open `notebooks/thesis_sec2e_exemplars.ipynb`
- [ ] Neural exemplars 018–021: y = "z-score [neural]"; height compact; legend = True / PSID / DPAD / VARMA
- [ ] Behavioral: y = "z-score [velocity]" / "[acceleration]"
- [ ] Laplacian: y = "z-score [LFP]"
- [ ] Rerun → save

### Remaining time — bullet points
- [ ] Complete any missing RQ bullet points
- [ ] Each bullet: finding → interpretation → caveat

---

## FRIDAY MORNING — 7:00–10:00 (3h, writing only)

- [ ] Q1 reconstruction: r/RMSE interpretation; PDI1 spectral floor; VARMA AR ceiling
- [ ] Q2 forecast: degradation rate; VARMA baseline; PSID/DPAD latent-state benefit
- [ ] Q3 classification: PDI4 above chance, PDI1 at floor; Xp_1 vs Xp_2; honest weak-effect framing
- [ ] Q4 target choice: behavioral vs neural supervision effects; Cy/Cz; data efficiency
- [ ] Cross-cutting: spectral separability (sec1 fig_005) explains PDI1 ceiling throughout all RQs

---

## Skip list (firm)
- sec2a diagnostics — discard
- sec2b behavioral reconstruction — discard
- `[NEW] fig_A` — `fig_022_neural_band_pearson.png` already covers this, reuse

---

## Figure locations quick ref

| Section | Figures | Notebook |
|---------|---------|----------|
| Sec1 data | `thesis_figures/sec1/` | thesis_sec1_data_verification.ipynb |
| Q1.2 neural | `thesis_figures/sec2/fig_070–075_*` | thesis_sec2c_neural_recon_group.ipynb |
| Q2 forecast | `thesis_figures/sec2/fig_080–086_*` | thesis_sec2d_neural_forecast_group.ipynb |
| Q3 classify | `thesis_figures/sec5/` | thesis_sec5_classification.ipynb |
| Q4 target | `thesis_figures/sec6/` | thesis_sec6_summary_appendix.ipynb |
| Exemplars | `thesis_figures/sec2/fig_018–021_*` + `report/thesis/figures/*_exemplar_*` | thesis_sec2e_exemplars.ipynb |
| Appendix | `thesis_figures/sec7/`, `thesis_figures/sec8/` | thesis_sec7_*, thesis_sec8_* |
