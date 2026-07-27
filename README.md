# latent-neural-dynamics-modeling

Latent state-space modelling of intracranial recordings (ECoG + STN LFP) during a
copy-draw tracing task, with DBS on/off blocks. Three model families — **PSID**,
**DPAD**, **VARMA** — are fitted per session, used to predict and forecast neural
and kinematic signals, and their latent states are fed to an LDA classifier that
decodes DBS state. Thesis figures come out of the notebooks in `notebooks/`.

Two experiment types run through the same code path:

| `experiment.type` | Z (target) | shown as |
|---|---|---|
| `z-as-behavior` | tracing kinematics | kinematics-target |
| `z-as-neural`   | LFP / Laplacian channels | LFP-target |

Sessions: `PDI1_S2`, `PDI1_S4`, `PDI4_S2`, `PDI4_S3`.

---

## Repo structure

```
preprocessing/          raw recordings -> 200 Hz band-limited trial table
  package_recordings.py     entry point
  components/               participants.py (iEEG + CAR + bands + chunking), motion.py, events.py
  participants_at_200Hz_scaled_1e6_{narrow_band,raw_envelope}.yaml

training/               fit models, run inference, classify latents
  pipeline.py               unified runner — the main entry point
  precompute_splits.py      shared block-chronological train/val/test splits
  train.py / test.py        single-step entry points
  sweep.py                  LDA sweep + permutation test over latent features
  components/               trainer.py, tester.py, data.py (TrialDataset)
  pipelines/                _base.py (phase machine), dpad_modal.py (GPU), psid_diagnostic.py
  setups/                   run YAMLs: {framework}_{PID}_S{SESS}_{type}.yaml (+ examples/, dpad_modal/)

utils/                  config.py ({} interpolation), ieeg.py (bands, Welch/Morlet),
                        motion.py, sync.py, split.py, stats.py, plots.py, polars.py,
                        frameworks/{psid,dpad,varma}.py, classification/ (epoching, chrono CV, permutation)

notebooks/              thesis figure notebooks + modules/ (loaders.py, style.py, sec*_common.py)
scripts/                simulation / calibration / notebook-generation helpers
environment/, environment.yml, modal_requirements.txt

data/  resampled_recordings/  results/  logs/  thesis_figures/  report/    <- gitignored
```

**The data and results trees are not in the repo — they live on the compute host:**

```bash
ssh bobby@neuro
cd ~/repos/latent-neural-dynamics-modeling
~/miniconda3/envs/neuro/bin/python -m training.pipeline --config ...
```

---

## Install

```bash
bash environment/create_env.sh        # conda env create -f environment/environment.yaml -> env "neuro"
conda activate neuro
pip install PSID==1.2.6 DPAD==0.0.9 mne statsmodels pyyaml torch
pip install modal                     # only for the Modal/GPU DPAD path
```

`environment/environment.yaml` is the minimal host env (python 3.11, tensorflow
2.15, polars, sklearn, jupyterlab); the modelling libraries above are pip-only.
A fuller local env (plotly, streamlit, mne, ray, CPU torch) is `environment.yml`
(env name `neuro_local`). Modal container deps are pinned in
`modal_requirements.txt` — base image `tensorflow/tensorflow:2.15.0-gpu`, with
PSID 1.2.6 force-installed `--no-deps` over DPAD's 1.2.5 pin.

Run everything **from the repo root**; configs use repo-relative paths.

---

## How to run

### 1. Preprocess: raw → 200 Hz band table

```bash
python -m preprocessing.package_recordings \
  --config preprocessing/participants_at_200Hz_scaled_1e6_raw_envelope.yaml \
  --participant PDI1 --session 2          # both optional; omit to do everything
```

Load iEEG parquet → drop LFP_1–8 → CAR across ECoG (per block, so implicitly per
DBS state) → notch → band-pass into `raw_bands` (+ Hilbert `envelope_bands`) →
resample 1000 → 200 Hz → scale ×1e6 → join motion → smooth and differentiate
kinematics → chunk trials with `chunk_margin` seconds of margin, splitting on
pauses longer than `max_pause_seconds` → write partitioned parquet.

Knobs live under `ieeg_process:`. **`root_directory:` is absolute in the shipped
YAMLs — edit it for your machine.**

### 2. Splits

```bash
python training/precompute_splits.py                              # all sessions
python training/precompute_splits.py --participant PDI4 --session 3
python training/precompute_splits.py --train 0.5 --test 0.4
```

Block-level, chronological, balanced across DBS conditions, written once to
`resampled_recordings/splits/{train,val,test}.parquet` and shared by every
framework and variant. Re-running one session upserts and leaves the others alone.

### 3. Model pipeline

One entry point for all frameworks; the framework comes from `framework.name` in
the YAML (`psid | dpad | varma | dpad_modal | psid_diagnostic`).

```bash
python -m training.pipeline --config training/setups/psid_PDI1_S2_z-as-behavior.yaml
python -m training.pipeline --config <cfg> --phases train --dbs off
python -m training.pipeline --config <cfg> --phases predictions,forecasts
python -m training.pipeline --config <cfg> --phases classification --cls-mode forecast_only
```

| Phase | Does | Writes |
|---|---|---|
| `train` | fit one model per `experiment.train.model_dbs_state` (`both`/`on`/`off`) | `model_<TS>.pkl`, `model_<TS>_metadata.json`, `split/` |
| `predictions` | trial-by-trial inference on `experiment.predictions.splits` | `inference/<split>/test_results_<TS>.parquet` |
| `forecasts` | for each `h` in `forecasts.h_grid`, forecast `default_m` s ahead | `forecast/h<H>/<split>/test_results_<TS>.parquet` |
| `classification` | LDA sweep over latent features + permutation test | `sweep_<TS>.parquet`, `classifiers/*.joblib` |

A phase runs only if its block exists in the YAML; `--phases` overrides that.
Single steps are also callable directly: `python training/train.py --config <cfg> --dbs both`,
and `training.test.test(config, model_dbs, run_timestamp, h=, m=)`.

### 4. DPAD on Modal (GPU)

```bash
modal run training/pipelines/dpad_modal.py::check
modal run training/pipelines/dpad_modal.py::sweep --configs-dir training/setups/dpad_modal
modal run training/pipelines/dpad_modal.py::sweep --configs-dir training/setups/dpad_modal --mode z-as-neural
```

Two-stage fan-out, one A10G container per (config, dbs): stage 1 `--phases train`,
stage 2 `--phases predictions,forecasts,classification` once stage 1 drains.
Volumes: `dpad-data` → `/app/resampled_recordings`, `dpad-results` → `/app/results`,
`dpad-training-setups` → `/app/training/setups` — upload recordings and configs
there first.

### 5. Figures

```bash
jupyter lab
jupyter nbconvert --to notebook --execute --inplace notebooks/thesis_sec5_classification.ipynb
python notebooks/_runner.py <script.py>       # headless, plotly show() disabled
```

| Notebook | Content |
|---|---|
| `thesis_sec1_data_verification` | trial inventory, PSDs, nx/n1 selection, DPAD training curves |
| `thesis_sec2c/2d/2e` | neural reconstruction, forecast, exemplar trials |
| `thesis_sec5_classification` | DBS decoding |
| `thesis_sec5b_group_permutation` | generated by `scripts/build_group_perm_notebook.py` |
| `thesis_sec7_subspace_dynamics`, `thesis_sec7b_matrix_dynamics` | subspace + matrix dynamics |
| `thesis_sec_appendix` | appendix figures |

The notebooks `chdir` to `/home/bobby/repos/latent-neural-dynamics-modeling` in
their first cell and read `results/` directly — run them on the compute host, or
edit that cell. They auto-discover the newest run timestamp per variant
(`modules/loaders.discover_session_run`, `utils/thesis_result_timestamps.py`), so
there is no timestamp bookkeeping. Output goes to `thesis_figures/`.

### End-to-end, one session

```bash
conda activate neuro
python -m preprocessing.package_recordings \
  --config preprocessing/participants_at_200Hz_scaled_1e6_raw_envelope.yaml \
  --participant PDI1 --session 2
python training/precompute_splits.py --participant PDI1 --session 2
python -m training.pipeline --config training/setups/psid_PDI1_S2_z-as-behavior.yaml
python -m training.pipeline --config training/setups/varma_PDI1_S2_z-as-behavior.yaml
modal run training/pipelines/dpad_modal.py::sweep --configs-dir training/setups/dpad_modal
jupyter nbconvert --to notebook --execute --inplace notebooks/thesis_sec5_classification.ipynb
```

---

## Run config (`training/setups/*.yaml`)

Named `{framework}_{PID}_S{SESS}_{experiment.type}.yaml`; one example per
framework in `training/setups/examples/`.

```yaml
framework:
  name: psid                 # psid | dpad | varma | dpad_modal | psid_diagnostic
  params:                    # PSID: nx, n1, i, A_eigen_constrain, time_first
    nx: 64                   # DPAD: nx, n1, epochs, checkpoint_every, method_code, steps_ahead
    n1: 1                    # VARMA: p, q, long_ar_lags, ridge_alpha, max_root, trial_edge_taper_sec
    i: 100
    fast: false
    reuse_splits: false      # true = reuse results/<fw>/<variant>/split/
data:
  root: resampled_recordings/participants_at_200Hz_scaled_1e6_raw_envelope
  participant: PDI1
  session: 2
  Y: [ECOG_3_gamma_88_93_raw, ECOG_1_alpha_8_12_env, ...]     # neural inputs (parquet column names)
  Z: [tracing_velocity_x, tracing_acceleration_magnitude]     # target
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
    perm_ba_gate: 0.5        # permutation test runs only if CV BA exceeds this
    n_permutations: 500
    param_grid: {LDA: {classifier__solver: [lsqr], classifier__shrinkage: [auto]}}
    feature_sources_pred:     [Xp, Xp_1, Xp_2, Xp_with_dbs]
    feature_sources_forecast: [Xf, Xf_1, Xf_2, Xf_with_dbs]
results:
  project_root: .
  save_dir: results/{framework.name}/{experiment.name}
  log_dir:  logs/{framework.name}
```

`{a.b.c}` placeholders resolve against the whole config at load time
(`utils/config.py`). Feature-source suffixes: `_1` = first `n1` latents
(behaviourally relevant), `_2` = latents `n1:nx`, `_with_dbs` = features plus a
DBS indicator channel (a ceiling/sanity control).

---

## Data shapes

### Raw input (`data/`)

```
data/
├── participants.tsv
├── participants_2/participant_id=PDI1/session=2/block=1/*.parquet
└── PDI1/ses-2/motion/*_task-copydraw_run-<R>_chunk-<C>_motion.tsv   (columns x, y)
                     *_task-copydraw_run-<R>_motion.json             (key dbs_stim)
```

Columns required in each `participants_2` partition: `participant_id` (str),
`session` (int), `block` (int, one DBS state per block), `trials` (list[int],
exploded to one row per trial), `onsets` (list[float], indexed by `trial-1`),
`ieeg_parquet` (str path), `session_path` (str, locates the motion dir), `stim`
(str; `on`/`true`/`1` → on, `off`/`false`/`0` → off), optional `is_fragmented`
(bool; `True` rows dropped).

iEEG parquet fields, each a list of float32 at **1000 Hz**: `LFP_1`…`LFP_16`
(1–8 dropped early; 9–16 feed the Laplacians), `ECOG_1`…`ECOG_4`,
`EOG_1`…`EOG_4`, `sfreq`. Events come from BIDS `*_events.tsv`
(`onset, duration, trial_type, value, sample`), keeping `trial_type` in [10, 21].

### Preprocessed recordings (`resampled_recordings/`)

`participants_at_200Hz_scaled_1e6_raw_envelope/participant_id=PDI1/session=2/block=3/0.parquet`,
one row per **trial**:

| Column | Type | Meaning |
|---|---|---|
| `participant_id`, `session`, `block`, `trial` | str/int | partition keys |
| `onset`, `margined_onset`, `margined_duration` | float | trial timing (s) |
| `time`, `time_original`, `motion_time` | list[float] | time vectors |
| `original_length_ts`, `start_ts`, `chunk_margin` | int/float | chunking metadata |
| `stim` | str | `on` / `off` |
| `x`, `y`, `x_smooth`, `y_smooth` | list | pen coordinates, raw and smoothed |
| `tracing_velocity_x/y`, `tracing_velocity_magnitude`, `tracing_acceleration_magnitude`, `tracing_jerk_*` | list[float] | Savitzky-Golay derivatives |
| `<CHANNEL>_<band>_raw` / `_env` | list[float] | band-limited signal / Hilbert envelope at 200 Hz |

Channel prefixes: `ECOG_1..4` and `LAPLACIAN_14-16_LFP_*`
(`D_k = LFP_k − 2·LFP_{k+1} + LFP_{k+2}`). Band names come straight from the
preprocessing YAML — e.g. `ECOG_3_gamma_88_93_raw`, `ECOG_1_alpha_8_12_env`. The
`raw_envelope` config gives 17 raw + 17 envelope bands per channel (gap 47–53 Hz
for mains, stop at 93 Hz); `narrow_band` gives raw bands only. Per trial the
model layer stacks these into `Y ∈ (T, n_Y)` and `Z ∈ (T, n_Z)`, with
`T = margined_duration × 200`.

Splits: `resampled_recordings/splits/{train,val,test}.parquet`, columns
`participant_id, session, block, trial, dbs_state`.

### Results (`results/`)

```
results/<framework>/<experiment.name>_dbs_<both|on|off>/
├── model_<TS>.pkl
├── model_<TS>_metadata.json          # framework_type, nx, n1, i, max_eigenvalue, ...
├── split/{train,val,test}.parquet
├── inference/<split>/test_results_<TS>.parquet/     # hive: participant_id/session/block/trial
├── forecast/h<H>/<split>/test_results_<TS>.parquet/
├── sweep_<TS>.parquet
└── classifiers/clf_{pred,forecast,flipped}_<dbs>_<sub_source>[_h<H>].joblib
```

Test/inference parquet, one row per trial: `Y`, `Z` (true), `Yp`, `Zp`
(predicted/forecast), `Xp` (latents; `Xf` in forecast dirs),
`pearson_per_channel`, `pearson_mean`, `pearson_overall_mean` and their `_Z`
counterparts, `time`, `offset`, `chunk_margin`, `margined_duration`, `stim`, and
the four partition keys.

Sweep parquet, one row per (mode, sub_source, flipped, dbs_train, t_cut | h,
m_test): `mode`, `pipeline`, `variant`, `run_ts`, `dbs_train`, `data_dbs`,
`sub_source`, `flipped`, `t_cut_seconds`, `h_seconds`, `m_test_seconds`,
`cv_ba`, `cv_roc_auc`, `cv_fold_ba`, `cv_y_true`, `cv_y_pred`, `cv_y_proba`,
`y_proba`, plus permutation columns when `perm_ba_gate` fires. **`cv_ba` is the
statistic the permutation p-value tests** — do not quote a different accuracy
next to a p-value.

Sweep modes: **predictions** (one LDA per sub-source × dbs-train on the full
`Xp`, scored at each `t_cut` by truncating to the first `t_cut·fs` samples),
**forecast** (one LDA per `h` on `Xf`, scored at each `m_test`), and **flipped**
(on- and off-models forecast the same `Y` window, label = model identity; not run
for DPAD).

---

## Gotchas

- Preprocessing YAMLs carry absolute paths (`root_directory: /home/bobby/...`).
- Test parquets have **no `input_channels` column** — take channel names from the
  run config or model metadata instead.
- `--phases train` is a no-op if a `model_*.pkl` already exists in the variant
  dir. Delete it, or use a new `experiment.name`, to retrain.
- Quote `'on'` / `'off'` in YAML — bare `on`/`off` parse as booleans (the pipeline
  maps `True`/`False` back defensively).
- Result dirs containing `top5` are a naming artifact: DPAD there still uses the
  full channel set; only VARMA was ever top-5.
