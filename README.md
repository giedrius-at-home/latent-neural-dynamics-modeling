# latent-neural-dynamics-modeling

Code for a master's thesis. Intracranial recordings (ECoG + STN LFP) from
Parkinson's patients doing a copy-draw tracing task, with DBS switched on and off
between blocks, are fitted with three latent state-space model families — **PSID**,
**DPAD**, **VARMA**. Each model does two things with a trial: a **one-step-ahead
prediction** pass over the whole trial, and an **m-step-ahead forecast** from a
window of past history. The models' latent states are then handed to an LDA
classifier that tries to decode whether DBS was on or off. The thesis figures are
built from those outputs.

Sessions: `PDI1_S2`, `PDI1_S4`, `PDI4_S2`, `PDI4_S3`. Each session is fitted in
two flavours, set by `experiment.type` in the run config:

| `experiment.type` | Z (the target the model tracks) |
|---|---|
| `z-as-behavior` | tracing kinematics (velocity, acceleration) |
| `z-as-neural`   | LFP Laplacian channels |

---

## Where things are

```
preprocessing/     raw recordings -> 200 Hz band-limited trial table      -> preprocessing/README.md
training/          feature/order selection, model fitting, inference,     -> training/README.md
                   DBS classification sweep
notebooks/         thesis figures from the results tree                   -> notebooks/README.md
utils/             shared library: config loading, band processing, splits,
                   model wrappers (frameworks/), classification helpers
environment/       conda env definition
```

Each stage has its own README explaining what it does, what it drops, and what it
writes. Read them in the order above.

Not in git, because they are large and machine-local: `data/`,
`resampled_recordings/`, `results/`, `logs/`, `thesis_figures/`, `report/`.

**The data and the results live on the compute host, not in this repo:**

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
```

`environment/environment.yaml` covers python 3.11, tensorflow 2.15, polars,
scikit-learn and jupyterlab; the modelling libraries above are pip-only. Add
`pip install modal` only if you intend to train DPAD on cloud GPUs.

Run everything **from the repo root** — configs use repo-relative paths.

---

## Running the whole thing

Four steps, in order. Details and options are in each stage's README.

```bash
# 1. raw recordings -> 200 Hz band-limited trial table         [preprocessing/README.md]
python -m preprocessing.package_recordings \
  --config preprocessing/participants_at_200Hz_scaled_1e6_raw_envelope.yaml \
  --participant PDI1 --session 2

# 2. shared block-chronological train/val/test splits          [training/README.md]
python training/precompute_splits.py --participant PDI1 --session 2

# 3. pick the 12 input channels and the latent dimensions,
#    which writes a ready-to-train run YAML                    [training/README.md]
python -m training.pipeline --config training/setups/psid_diagnostic_PDI1_S2_z-as-behavior.yaml

# 4. fit + predict + forecast + classify, per framework        [training/README.md]
python -m training.pipeline --config training/setups/psid_PDI1_S2_z-as-behavior.yaml
python -m training.pipeline --config training/setups/varma_PDI1_S2_z-as-behavior.yaml
python -m training.pipeline --config training/setups/dpad_modal/dpad_modal_PDI1_S2_z-as-behavior.yaml
```

Step 4 writes models, per-trial prediction/forecast parquets and a classification
sweep parquet under `results/<framework>/<variant>/`. The notebooks read that tree
and produce the figures — see `notebooks/README.md`.

Every model in the thesis is fitted on **12 neural input channels**: 6
band-limited raw signals and 6 Hilbert envelopes, chosen per session by
cross-validated mRMR in step 3. That holds for PSID, DPAD and VARMA alike.
