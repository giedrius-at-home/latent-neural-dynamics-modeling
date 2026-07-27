# latent-neural-dynamics-modeling

Code for a master's thesis on latent state-space modelling of intracranial
recordings under deep brain stimulation.

---

## Install

```bash
git clone https://github.com/giedrius-at-home/latent-neural-dynamics-modeling.git
cd latent-neural-dynamics-modeling

bash environment/create_env.sh        # conda env create -f environment/environment.yaml -> env "neuro"
conda activate neuro
pip install torch                     # CPU build, used by the data loader
```

`environment/environment.yaml` now carries everything the pipeline imports:
python 3.11, tensorflow 2.15, `polars=1.34.*`, scikit-learn, PSID, DPAD, mne,
statsmodels, plus `modal`, `feature-engine` and `mrmr-selection` — the last two
because the standard workflow starts with the `psid_diagnostic` stage, and
`training/pipeline.py` imports it (and `dpad_modal`) at module scope.

Run everything **from the repo root** — configs use repo-relative paths.

## Get the data

The recordings and the results are not in the repo — they are large, and
`data/`, `resampled_recordings/`, `results/`, `logs/`, `thesis_figures/` and
`report/` are all gitignored for that reason. Expect to work on whichever
machine holds them: clone the repo there, create the env there, and run from
that checkout.

The code assumes only that the data sits under the repo root, at the paths in
the section below. If your copy lives elsewhere, point `data.root` in a run YAML
and `data_directory` in a preprocessing YAML at it. Have no data at all? See
[Run it without the real data](#run-it-without-the-real-data).

## Run it

Four steps, in order. One session (`PDI1_S2`, kinematics as the target) end to end:

```bash
# 1. raw recordings -> 200 Hz band-limited trial table
python -m preprocessing.package_recordings \
  --config preprocessing/participants_at_200Hz_scaled_1e6_raw_envelope.yaml \
  --participant PDI1 --session 2

# 2. block-chronological train/val/test splits, shared by every model
python training/precompute_splits.py --participant PDI1 --session 2

# 3. pick the 12 input channels and the latent dimensions;
#    this writes them into the run YAML used by step 4
python -m training.pipeline --config training/setups/psid_diagnostic_PDI1_S2_z-as-behavior.yaml

# 4. fit each model family, run its forecasts, classify DBS state from the latents
python -m training.pipeline --config training/setups/psid_PDI1_S2_z-as-behavior.yaml
python -m training.pipeline --config training/setups/varma_PDI1_S2_z-as-behavior.yaml
python -m training.pipeline --config training/setups/dpad_modal/dpad_modal_PDI1_S2_z-as-behavior.yaml
```

Step 4 writes models, per-trial forecast parquets and a classification sweep
parquet under `results/<framework>/<variant>/`. The notebooks turn that tree into
the thesis figures.

Repeat steps 2–4 for the other sessions (`PDI1_S4`, `PDI4_S2`, `PDI4_S3`) and for
the other target type (`z-as-neural`).

## Run it without the real data

`generate_data.py` writes a fake participant at the very start of the chain: the
input `package_recordings` consumes. Preprocess it as usual, and the resulting
200 Hz table feeds the training pipeline like any other session. Useful for
trying the pipeline on a laptop, or checking a change end to end before running
it on the real recordings.

```bash
python generate_data.py
python -m preprocessing.package_recordings --config <your local preprocessing YAML>
python training/precompute_splits.py --data-root fake_data/participants_fake_200Hz \
    --participant FAKE1 --session 1
python -m training.pipeline --config <run YAML pointed at that data root>

python generate_data.py --clean
```

The preprocessing YAML is yours to write — copy
`preprocessing/participants_at_200Hz_scaled_1e6_raw_envelope.yaml` and change
four keys:

```yaml
root_directory: <repo root>
data_directory: "{root_directory}/fake_data/raw"       # holds participants_2/
save_directory: "{root_directory}/fake_data"           # output lands here
output_participants_table_name: "participants_fake_200Hz"
```

Trimming `raw_bands` / `envelope_bands` from 17 bands to 3 or 4 keeps the run
quick; everything else — `resampled_freq`, `chunk_margin`, `notch_freqs`,
`apply_car` — can stay as shipped. See `preprocessing/README.md` for what each
key does.

The fake participant is `FAKE1`, session `1`, so it cannot collide with real
data. Nothing about the results is meaningful — the point is that every stage
runs. What gets written:

```
fake_data/
└── raw/
    ├── participants_2/participant_id=FAKE1/session=1/block=<b>/0.parquet
    ├── resampled/sub-FAKE1_ses-1_task-copydraw_run-<b>_ieeg.parquet
    └── sub-FAKE1/ses-1/motion/
        └── sub-FAKE1_ses-1_task-copydraw_run-<b>_chunk-<t>_tracksys-wacom_motion.tsv
```

Dimensions default to a real session (PDI1_S2): 12 blocks x 12 trials, 9 s per
trial, onsets 18 s apart, ~277 s of 1000 Hz recording per block, ~1069 pen
samples per trial, and the same 25 channels (`LFP_1..16`, `ECOG_1..4`,
`EOG_1..4`, `sfreq`). The intermediate table carries the real column names and
dtypes. Preprocessing turns that into 13 s trials of 2600 samples at 200 Hz,
exactly as it does for real recordings; the only deliberate difference is 4
bands instead of 17, so the output is 105 columns rather than 365.

The channel samples are plain Gaussian noise — the point is to exercise the
stages, not to produce meaningful results. Blocks, trials per block, trial
length, onset spacing and seed are flags; `--status` shows what exists and
`--clean` removes it. At the defaults the raw tree is ~290 MB and preprocessing
takes well under a minute with four bands.

Nothing extra to install — the conda env already carries everything a run
imports.

## Where things are

```
preprocessing/     raw recordings -> 200 Hz band-limited trial table      -> preprocessing/README.md
training/          feature/order selection, model fitting, forecasting,   -> training/README.md
                   DBS classification sweep
notebooks/         thesis figures built from the results tree
utils/             shared library: config loading, band processing, splits,
                   model wrappers (frameworks/), classification helpers
environment/       conda env definition
```

Each stage's README explains what it does, what it drops, and what it writes.
Read them in that order.

---

## What the project actually does

Parkinson's patients with implanted electrodes (ECoG over cortex, LFP from the
STN) trace shapes on a tablet while DBS is switched on and off between blocks.
The question is whether stimulation leaves a signature in the *dynamics* of the
neural activity — not just its average power.

Three latent state-space model families are fitted per session — **PSID**,
**DPAD**, **VARMA** — each learning a low-dimensional latent state that drives
both the neural signals and a target `Z`. Each model is then asked to forecast:

- **one-step-ahead forecast** — run through the whole trial, at each sample
  forecasting the next one while still seeing the neural input. Latents from this
  pass are `Xp`.
- **m-step-ahead forecast** — see `h` seconds of history, then run free for `m`
  seconds with no further input. Latents from this pass are `Xf`.

In the code the first one is called **prediction**, not forecast: the pipeline
phase is `predictions`, the wrapper method is `predict`, and the output goes to
`inference/`. The docs say one-step-ahead forecast; the code says prediction.
They are the same thing.

An LDA classifier then tries to decode DBS on vs off from those latents, and a
block-shuffled permutation test says whether it beat chance.

Sessions: `PDI1_S2`, `PDI1_S4`, `PDI4_S2`, `PDI4_S3`. Each is fitted in two
flavours, set by `experiment.type` in the run config:

| `experiment.type` | Z (the target the model tracks) |
|---|---|
| `z-as-behavior` | tracing kinematics (velocity, acceleration) |
| `z-as-neural`   | LFP Laplacian channels |

Every model in the thesis is fitted on the same **12 neural input channels** — 6
band-limited raw signals and 6 Hilbert envelopes, chosen per session by
cross-validated mRMR in step 3. That holds for PSID, DPAD and VARMA alike.
