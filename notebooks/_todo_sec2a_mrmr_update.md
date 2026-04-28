# Sec2a mRMR figure update — plan

## Why

- Fig 44 already loads Mazzanti MIQ picks from `configs/diagnostic/mrmr_picks.yaml`
  and shows MI relevance heatmap — but markdown around Fig 44 still calls it
  "MI-mRMR (MIQ variant)" while Fig 47 markdown opens with **"Fig 44 picked
  channels with instantaneous Pearson r"** — contradiction.
- Production is **Mazzanti MIQ** (MI relevance + Pearson redundancy, FCQ quotient,
  vote-rank multi-target). Lagged Pearson exists as artifact but is **not**
  what trained models use.
- Fig 48 was "shipped Pearson vs Mazzanti" comparison — both columns are now
  Mazzanti, figure is meaningless.

## Per-figure workflow

One figure at a time. Stop after each, let Giedrius inspect.

### Step 1 — Fig 44 (Mazzanti MIQ heatmap)
- **Code**: already correct. Confirm `_mrmr_select` returns yaml picks, heatmap is MI.
- **Markdown**: rewrite header to make the production claim explicit
  ("Mazzanti `mrmr_regression`, MI relevance + Pearson redundancy, FCQ quotient,
   per-target vote-rank aggregation. Cached at configs/diagnostic/mrmr_picks.yaml.").
- **Style**: tighten figsize; sec1 uses (5.4, 4.2)-(7.5, 4.8). Currently sec2a Fig 44
  is (5.4, 4.2) — keep.
- **Panel labels**: A/B with informative titles. Already conformant.

### Step 2 — Fig 45 (60×60 corr matrix), Fig 46 (relevance column)
- Restyle to match sec1 — compact figsize, panel_label, no forced kwargs in legend.
- Currently single-panel; OK as-is. Light pass only.

### Step 3 — Fig 47 (lagged + bootstrap)
- Reframe markdown: **"alternative selection metric (not used in production)".**
  Show that LFP picks are identical (8/8 bootstrap) and ECoG picks differ in 41%
  of cases, but Mazzanti MIQ was kept because it captures MI nonlinearity that
  lagged Pearson cannot.
- Keep figure as informative; maybe demote to optional/appendix.
- Code stays.

### Step 4 — Fig 48 (delete)
- Remove cell. Mazzanti is now production → "shipped vs Mazzanti" is identity.
- Move panel_letter sequence: Fig 49+ stay numbered as-is.

### Step 5 — Fig 36 (PSID scree)
- Already conformant per inspection. No-op unless we identify a style break.

## Out of scope

- DPAD/VARMA mRMR comparisons (handled in sec2c/d).
- LFP-Z `mrmr_top_k_lfp_from_diagnostic` artifact (different code path).
- Retraining (no model changes — picks identical).

## Acceptance

- All sec2a figs pass thesis-figures quality checklist.
- No "instantaneous Pearson" narrative left in markdown (consistent with code).
- Fig 48 deleted; sec2a header table updated.
- `.md` tech description written per figure when complete.
