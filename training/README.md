# training

Everything between the preprocessed trial table and the results tree the figures
are drawn from: splitting the data, choosing input features and latent
dimensions, fitting the models, running the two forecasts, and classifying
DBS state from the latents.

One entry point does all of it. The framework is declared inside the config, not
on the command line:

```bash
python -m training.pipeline --config training/setups/psid_PDI1_S2_z-as-behavior.yaml
```

`framework.name` ∈ `psid | dpad | varma | dpad_modal | psid_diagnostic`.

---

## Step 1 — splits

```bash
python training/precompute_splits.py                                # all sessions
python training/precompute_splits.py --participant PDI4 --session 3
python training/precompute_splits.py --train 0.5 --test 0.4         # default fractions otherwise
```

Splits are decided **once**, at the level of whole blocks, and shared by every
framework and every variant — so PSID, DPAD and VARMA are always compared on
identical trials.

How blocks are assigned: group blocks by DBS condition, sort each group
chronologically, take `min_n = min(#off_blocks, #on_blocks)`, then give the first
`train_frac · min_n` blocks of each condition to train, the last
`test_frac · min_n` to test, and everything in between (plus any surplus blocks of
the longer condition) to val. Chronological rather than random, so a model never
trains on a block recorded after the one it is tested on.

Written to `resampled_recordings/splits/{train,val,test}.parquet` with columns
`participant_id, session, block, trial, dbs_state`. Re-running one session
replaces just that session's rows.

---

## Step 2 — pick input features and latent dimensions (`psid_diagnostic`)

The preprocessed table has hundreds of candidate signal columns. This stage picks
the 12 that go into the models, and the latent dimensions to fit, then writes them
straight into the run YAML.

```bash
python -m training.pipeline --config training/setups/psid_diagnostic_PDI1_S2_z-as-behavior.yaml
```

What it does:

1. **Feature selection** — cross-validated mRMR (minimum redundancy, maximum
   relevance) against the DBS label, on log-power epochs of each candidate
   feature. A feature is one channel × one band × raw or envelope, so the four
   ECoG channels alone offer 136 of them. With `stratify_raw_env: true` the
   budget is split evenly, so
   `K_Y: 12` yields **6 raw band signals + 6 Hilbert envelopes**. The same is done
   for the Laplacian candidates in `z-as-neural` mode (`K_Z: 12`);
   `z-as-behavior` mode pins Z to `tracing_velocity_x` and
   `tracing_acceleration_magnitude` instead.
2. **nx sweep** — fit PSID with `n1=0` over `nx_grid` in an inner CV on the train
   trials, score **one-step-ahead** Y reconstruction correlation on the fold's
   validation trials, and take the smallest `nx` within one standard error of
   the best (1-SE rule).
3. **n1 sweep** — same procedure over `n1_grid` at the chosen `nx`, scored on
   one-step-ahead Z reconstruction correlation.
4. **Final fit** at the chosen `(nx, n1)` on the full train set, evaluated once on
   test — held-out from the sweeps. This is the only place the diagnostic scores
   a multi-step forecast: `history_s` seconds of history, `forecast_s` seconds
   ahead, for both Y and Z.
5. **Amend the run YAML**: `training/setups/psid_<PID>_S<SESS>_<type>.yaml` gets
   its `nx`, `n1`, `i`, `Y` and `Z` fields patched in. The file must already
   exist; the diagnostic only fills it in. Copy the resulting `Y`/`Z` into the
   matching VARMA and DPAD configs so all three frameworks see identical inputs.

Also writes, under `results/psid_diagnostic/<PID>_S<SESS>_<type>/<TS>/`:
`diagnostic.parquet` (the full sweep curves, feeding the selection figures in
section 1), `mrmr_stats.json` (which features were picked, and how often each
survived across folds) and `model.pkl` (the final diagnostic fit).

**Every model in the thesis uses these 12 input features — 6 raw + 6 envelope.**
No framework runs on a different feature count.

---

## Step 3 — fit, forecast, classify

```bash
python -m training.pipeline --config <cfg>                                 # all phases
python -m training.pipeline --config <cfg> --phases train --dbs off        # one phase, one condition
python -m training.pipeline --config <cfg> --phases predictions,forecasts
python -m training.pipeline --config <cfg> --phases classification --cls-mode forecast_only
```

| Phase | What happens | Written to |
|---|---|---|
| `train` | one model per DBS condition in `experiment.train.model_dbs_state` (`both`, `on`, `off`) | `model_<TS>.pkl`, `model_<TS>_metadata.json`, `split/` |
| `predictions` | one-step-ahead forecast over every trial of each split in `experiment.predictions.splits` | `inference/<split>/test_results_<TS>.parquet` |
| `forecasts` | for each `h` in `forecasts.h_grid`: take `h` seconds of history, forecast `default_m` seconds ahead | `forecast/h<H>/<split>/test_results_<TS>.parquet` |
| `classification` | LDA sweep over the latent states + permutation test | `sweep_<TS>.parquet`, `classifiers/*.joblib` |

A phase runs only if its block exists in the YAML; `--phases` overrides that.
`train` is skipped when a `model_*.pkl` already sits in the variant directory —
delete it, or change `experiment.name`, to refit.

Two forecasts are what a model is asked for. Both are forecasts — they differ in
how far ahead, and in whether the model keeps seeing input:

- **one-step-ahead forecast** — run through the whole trial, at each sample
  forecasting the next one while still seeing the neural input. Latents from this
  pass are `Xp`.
- **m-step-ahead forecast** — see `h` seconds of history, then run free for `m`
  seconds with no further input. Latents from this pass are `Xf`.

The code calls the first one a **prediction**: the phase is `predictions`, the
method is `BaseWrapper.predict`, the latents are `Xp` (p for prediction), and the
output lands in `inference/`. Nothing was renamed — read "prediction" in the code
as "one-step-ahead forecast".

Single steps are callable on their own if you want just one piece:
`python training/train.py --config <cfg> --dbs both`, or
`training.test.test(config, model_dbs, run_timestamp, h=, m=)`.

### DPAD

DPAD trains on GPU. Locally, with `framework.name: dpad`:

```bash
python -m training.pipeline --config training/setups/examples/dpad_run.yaml
```

Training is the slow part (3000 epochs in the thesis configs); both forecasts use
a compiled TensorFlow fast path. To use cloud GPUs instead, set
`framework.name: dpad_modal` and run the same command, or drive Modal directly for
multi-session sweeps:

```bash
modal run training/pipelines/dpad_modal.py::check                                          # GPU smoke test
modal run training/pipelines/dpad_modal.py::sweep --configs-dir training/setups/dpad_modal
modal run training/pipelines/dpad_modal.py::sweep --configs-dir training/setups/dpad_modal --mode z-as-neural
```

Two-stage fan-out, one A10G container per (config, DBS condition): stage 1 runs
`--phases train`, stage 2 runs `--phases predictions,forecasts,classification`
once stage 1 has fully drained. Volumes must be populated first —
`dpad-data` → `/app/resampled_recordings`, `dpad-results` → `/app/results`,
`dpad-training-setups` → `/app/training/setups`. Container deps are pinned in
`modal_requirements.txt`.

### The classification sweep

Three sweeps run per session, all with the same LDA and the same chronological,
block-grouped cross-validation:

- **predictions** (`Xp`, the one-step-ahead latents) — one classifier per
  (feature source, training condition), trained on the full `Xp`, then scored at
  each `t_cut` in `t_cut_grid` by truncating the latents to the first `t_cut`
  seconds. Answers: how much of a trial do you need to see to tell DBS on from
  off?
- **forecast** (`Xf`, the m-step-ahead latents) — one classifier per history
  window `h`, trained on `Xf`, scored at each `m_test`. Answers: does the DBS
  signature survive into the part the model imagined rather than observed?
- **flipped** — the on-model and the off-model both forecast the same window, and
  the label is which model produced it. Not run for DPAD.

Feature-source suffixes: no suffix = all latents, `_1` = the first `n1`
(behaviourally relevant) latents, `_2` = latents `n1:nx`, `_with_dbs` = latents
plus a DBS indicator, which is a ceiling/sanity control and not a real
result.

Any classifier scoring above `perm_ba_gate` gets `n_permutations` block-shuffled
label permutations for a p-value. **The statistic being tested is `cv_ba`** — never
quote a different accuracy next to one of these p-values.

---

## Run configs (`training/setups/`)

Named `{framework}_{PID}_S{SESS}_{experiment.type}.yaml`, e.g.
`varma_PDI4_S3_z-as-neural.yaml`. DPAD-on-Modal configs live in `dpad_modal/`,
and `examples/` holds one minimal config per framework.

```yaml
framework:
  name: psid                 # psid | dpad | varma | dpad_modal | psid_diagnostic
  params:                    # PSID:  nx, n1, i, A_eigen_constrain, time_first
    nx: 64                   # DPAD:  nx, n1, epochs, checkpoint_every, method_code, steps_ahead
    n1: 1                    # VARMA: p, q, long_ar_lags, ridge_alpha, max_root, trial_edge_taper_sec
    i: 100
    fast: false
    reuse_splits: false      # true = reuse results/<fw>/<variant>/split/ instead of rebuilding
data:
  root: resampled_recordings/participants_at_200Hz_scaled_1e6_raw_envelope
  participant: PDI1
  session: 2
  Y: [ECOG_3_gamma_88_93_raw, ..., ECOG_1_alpha_8_12_env]     # the 12 selected features
  Z: [tracing_velocity_x, tracing_acceleration_magnitude]     # or 12 Laplacian features
  sampling_frequency: 200
experiment:
  name: '{framework.name}_{experiment.type}_{data.participant}_S{data.session}_nx_{framework.params.nx}_n1_{framework.params.n1}'
  type: z-as-behavior
  train:       {model_dbs_state: [both, 'on', 'off']}
  predictions: {splits: [train, val, test]}
  forecasts:   {default_h: 5.0, default_m: 2.0, h_grid: [...], m_test_grid: [...]}
  classification:
    epoch_length: 0.5
    epoch_overlap: 0.25
    n_splits: 5
    t_cut_grid: [0.5, 1, ..., 9]
    perm_ba_gate: 0.5        # permutation test only above this CV balanced accuracy
    n_permutations: 500
    param_grid: {LDA: {classifier__solver: [lsqr], classifier__shrinkage: [auto]}}
    feature_sources_pred:     [Xp, Xp_1, Xp_2, Xp_with_dbs]
    feature_sources_forecast: [Xf, Xf_1, Xf_2, Xf_with_dbs]
results:
  project_root: .
  save_dir: results/{framework.name}/{experiment.name}
  log_dir:  logs/{framework.name}
```

`{a.b.c}` placeholders are resolved against the whole config when it loads
(`utils/config.py`), which is why `experiment.name` can build a directory name out
of the parameters.

Quote `'on'` and `'off'` — unquoted, YAML parses them as booleans. The pipeline
maps `True`/`False` back defensively, but the configs should not rely on it.

---

## What lands in `results/`

```
results/<framework>/<experiment.name>_dbs_<both|on|off>/
├── model_<TS>.pkl
├── model_<TS>_metadata.json          framework_type, nx, n1, i, max_eigenvalue, ...
├── test_stats_<TS>.hdf5              aggregate stats written after testing
├── split/{train,val,test}.parquet    the trials this variant actually used
├── inference/{train,val,test}/test_results_<TS>.parquet/   hive: participant_id/session/block/trial
├── forecast/h<H>/{train,val,test}/test_results_<TS>.parquet/
└── classification/
    ├── sweep_<TS>.parquet
    └── classifiers/clf_{pred,forecast,flipped}_<dbs>_<sub_source>[_h<H>].joblib
```

A real one: `results/psid/psid_z-as-neural_PDI4_S3_nx_64_n1_1_dbs_both/`. Forecast
horizons appear as `h0.5`, `h1`, `h1.5`, `h2`, `h3`, `h4`, `h5` — trailing zeros
are trimmed. `<TS>` is the training timestamp (`YYYYmmdd_HHMMSS`) and ties a model
to everything produced from it.

Older variant directories carry a different naming scheme
(`psid_behavioral_PDI1_2_nx_64_n1_i60_dbs_both`) and a `train/`, `val/`, `test/`
set of parquets at the top level, from the pre-`inference/` layout. Ignore those;
the notebooks pick the newest run per variant.

Forecast parquet (both kinds share the schema), one row per trial:

| Column | Meaning |
|---|---|
| `Y`, `Z` | true neural input and target |
| `Yp`, `Zp` | forecast neural and target |
| `Xp` | latent states of the one-step-ahead pass |
| `pearson_per_channel`, `pearson_mean`, `pearson_overall_mean` | Y-side correlations |
| `pearson_per_channel_Z`, `pearson_mean_Z`, `pearson_overall_mean_Z` | Z-side correlations |
| `time`, `offset`, `chunk_margin`, `margined_duration` | timing |
| `stim` | the trial's DBS state |
| `Y_features`, `Z_features` | the feature names behind `Y` and `Z`, in column order |
| `participant_id`, `session`, `block`, `trial` | partition keys |

`Y_features` / `Z_features` are how you recover which features a run used —
there is no `input_channels` column. A `z-as-neural` run shows 12 entries in each.

The `forecast/h<H>/` parquets add the free-running part on top of those columns:
`m` (horizon in samples), `Y_future_true`, `Y_future_pred`, `Z_future_true`,
`Z_future_pred`, `X_future_pred` (the m-step-ahead latents — `Xf` is the sweep's
name for them, not a column name), and `Y_concat_for_plot` / `Z_concat_for_plot`,
which glue history and forecast together for figures.

Sweep parquet, one row per (mode, feature source, flipped, training condition,
`t_cut` or `h`, `m_test`) — 26 columns, ~800 rows for a session:

| Group | Columns |
|---|---|
| what was run | `mode`, `pipeline`, `variant`, `run_ts`, `dbs_train`, `data_dbs`, `sub_source`, `flipped` |
| where it was scored | `t_cut_seconds`, `h_seconds`, `m_test_seconds` (only the relevant one is set; the others are null) |
| cross-validated score | `cv_ba`, `cv_roc_auc`, `cv_fold_ba`, `cv_y_true`, `cv_y_pred`, `cv_y_proba` |
| score at that grid point | `ba_at_score`, `n_score`, `y_true`, `y_pred`, `y_proba` |
| permutation test | `n_permutations`, `p_value`, `perm_mean_ba`, `perm_scores` |

`cv_ba` and `ba_at_score` are different numbers: `cv_ba` is the cross-validated
balanced accuracy of the classifier, `ba_at_score` is the balanced accuracy at
that particular truncation or forecast point. **The permutation `p_value` tests
`cv_ba`** — quoting `ba_at_score` next to it is a mismatch.

---

## Files

| File | Role |
|---|---|
| `pipeline.py` | the entry point; picks a pipeline class from `framework.name` |
| `pipelines/_base.py` | phase machine shared by PSID / DPAD / VARMA |
| `pipelines/dpad_modal.py` | Modal image, volumes, two-stage GPU fan-out |
| `pipelines/psid_diagnostic.py` | mRMR feature selection + nx/n1 sweeps + YAML emission |
| `precompute_splits.py` | shared block-chronological splits |
| `train.py`, `test.py` | single-phase entry points |
| `sweep.py` | classification sweeps and permutation test |
| `components/trainer.py` | data loading, split application, fitting, model saving |
| `components/tester.py` | the forecast loops and parquet writing |
| `components/data.py` | `TrialDataset` — one trial to `(Y, Z, metadata)` |
| `../utils/frameworks/` | the PSID, DPAD and VARMA wrappers behind a shared interface |
| `../utils/classification/` | epoching, chronological grouped CV, permutation machinery |
