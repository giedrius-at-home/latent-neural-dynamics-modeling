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
pip install PSID==1.2.6 DPAD==0.0.9 mne statsmodels pyyaml torch
```

`environment/environment.yaml` covers python 3.11, tensorflow 2.15, polars,
scikit-learn and jupyterlab; the modelling libraries above are pip-only. Add
`pip install modal` only if you intend to train DPAD on cloud GPUs.

Run everything **from the repo root** — configs use repo-relative paths.

## Get the data

The recordings and the results are not in the repo: they are large, and they live
on the compute host. Work there.

```bash
ssh bobby@neuro
cd ~/repos/latent-neural-dynamics-modeling
~/miniconda3/envs/neuro/bin/python -m training.pipeline --config ...
```

`data/`, `resampled_recordings/`, `results/`, `logs/`, `thesis_figures/` and
`report/` are all gitignored for that reason.

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

`generate_data.py` writes a fake participant in the same shape as the real
recordings, at the very start of the chain — the intermediate table, the 1000 Hz
iEEG parquets and the motion sidecars that preprocessing consumes. Useful for
trying the pipeline on a laptop, or checking a change end to end before it goes
near the compute host.

```bash
python generate_data.py --blocks 10 --trials-per-block 3
python -m preprocessing.package_recordings --config fake_data/preprocessing_fake.yaml
python training/precompute_splits.py --data-root fake_data/participants_fake_200Hz \
    --participant FAKE1 --session 1
python -m training.pipeline --config <run YAML pointed at that data root>

python generate_data.py --clean
```

The fake participant is `FAKE1`, session `1`, so it cannot collide with real
data. Nothing about the results is meaningful — the point is that every stage
runs.

## Where things are

```
preprocessing/     raw recordings -> 200 Hz band-limited trial table      -> preprocessing/README.md
training/          feature/order selection, model fitting, forecasting,   -> training/README.md
                   DBS classification sweep
notebooks/         thesis figures from the results tree                   -> notebooks/README.md
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
