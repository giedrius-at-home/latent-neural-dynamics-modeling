# Stage 4 — Evaluation (test + cross-condition)

## Job

For every trained model, generate test-split reconstruction and multi-step
forecast outputs. Also run cross-condition eval: apply an OFF-trained model to
ON data (`eval_on`) and vice versa.

## Producer

```bash
python -m training.test --config training/setups/.../<variant>.yaml
```

Called automatically by Phase 4 of `pipeline_psid.py` and its DPAD/VARMA
counterparts.

For cross-condition eval (`on→off`, `off→on`), the pipeline generates variant
directories like `<variant>_eval_on` / `<variant>_eval_off` alongside the main
directory — these reuse the off-trained / on-trained model but run the test
step against the opposite DBS condition's trials.

## Artifacts on disk

Per variant (`results/<variant>/`):

```
test/
  test_results_<ts>.parquet/
    participant_id=<p>/session=<s>/block=<b>/trial=<t>/0.parquet
      columns: Y, Z, Yp, Zp, Xp,
               Y_future_true, Y_future_pred, Z_future_true, Z_future_pred,
               X_future_pred,
               pearson_per_channel, pearson_mean, ...
               time, offset, chunk_margin, margined_duration, stim,
               input_channels, output_channels, m
  test_stats_<ts>.hdf5       # trial-level summary stats
train/, val/                 # same structure for train/val splits
```

`Y_future_*` and `Z_future_*` are the `m`-second-ahead forecast arrays —
populated when the model supports forecast (all PSID variants + VARMA; DPAD
has `Y_future_pred = None` because its forecast head isn't implemented).

## Cross-condition variant dirs

For each (cell, mode), there are 8 single-condition variant dirs:

```
psid_<mode>_<cell>_nx_<nx>_n<n1>_i100_dbs_on_200Hz_narrow_band
psid_<mode>_<cell>_nx_<nx>_n<n1>_i100_dbs_on_200Hz_narrow_band_eval_off
psid_<mode>_<cell>_nx_<nx>_n<n1>_i100_dbs_off_200Hz_narrow_band
psid_<mode>_<cell>_nx_<nx>_n<n1>_i100_dbs_off_200Hz_narrow_band_eval_on
```

Same pattern for VARMA (`p30_q1_mrmr8` variant family).

`_eval_on` / `_eval_off` dirs share the saved-model file of their parent (via
config) but have their own `test/` directory with cross-condition parquets.

## What each model predicts

| target | behavioral-mode Z is… | laplacian-mode Z is… |
|---|---|---|
| **Y** (reconstruction + forecast) | 8-channel ECoG (top-8 mRMR) | 8-channel ECoG (top-8 mRMR) |
| **Z** (reconstruction + forecast) | 2-channel behavior (`tracing_velocity_x`, `tracing_acceleration_magnitude`) | 15-band LFP (laplacian-pair narrow bands) |

Forecast target is controlled by `forecast_target="Y"` vs `"Z"` in the
downstream notebooks. For behavioral mode, forecasting Y = "predict the future
neural signal from past neural"; forecasting Z = "predict future behavior from
past neural". For laplacian mode, Z is the 15-band LFP — so Z forecast =
predict future LFP from past ECoG.

## FB smoother and what it changes at test time

The FB smoother **lives inside `predict()` / `forecast()` dispatch** on the
saved model. That means:
- Same training = same `A, Cy, Cz, Q, R, S` matrices.
- FB (bool attribute `fb_smoother` on the model) toggles whether reconstruction
  uses forward-only Kalman (RTS) or forward-backward (FB).
- `mode="forward"` / `mode="fb"` parameter on `predict()` / `forecast()`
  overrides the saved flag — this is what `scripts/run_option_a_rts_clip.sh`
  exploits to derive RTS + CLIP results from FB + CLIP models without
  retraining.

## Feeds into stage 5

Test parquets → classification feature extraction (Xp, Xp_1, Xp_2, Xp+DBS) →
LDA classifier. See stage 5.

## Symlinks

- `results_root/` → `../../../results/` (all test parquets per variant)
