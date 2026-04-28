# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: neuro
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sec 2d — Neural forecast (multi-step) group metrics
#
# Group-level forecast performance for predicting future neural activity (z-scored).
# PSID + VARMA only (DPAD forecast head not implemented).
#
# Each pooled / per-session scope emits **three PNGs** (RMSE, Pearson r, VAF).
#
# * Fig 23     — forecast RMSE vs horizon (0-1000 ms), DBS-OFF/ON panels
# * Figs 24-28 — pooled + per-session per-trial forecast metric distributions
# * Figs 56-59 — per-session LFP (laplacian) forecast distributions

# %%
import sys, os
os.chdir('/home/bobby/repos/latent-neural-dynamics-modeling')
sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import numpy as np
import matplotlib.pyplot as plt

from thesis_style import (
    COLOR_PSID, COLOR_DPAD, COLOR_VARMA,
    COLOR_DBS_OFF, COLOR_DBS_ON, COLOR_NS,
    apply_thesis_style, panel_label,
)
from thesis_sec2_common import *

from thesis_loaders import (
    load_split_results_required, channels_as_str_list,
    resolve_neural_y_channel_idx, neural_y_feature_label,
)
from thesis_utils import (
    AggregateRmseData, _key_index_map, _sem, normalize_stim,
    trial_metric_forecast_for_model, pick_best_feature_for_psid,
    metric_axis_label,
)

from thesis_lib.forecast_horizon_rmse import collect_forecast_horizon_rmse

apply_thesis_style()

# %% [markdown]
# ## Figs 64-67: Forecast distribution alternatives + drift visualisations (pooled Y, RMSE)
#
# Companion to sec2c Figs 60-63 but on the **forecast** (not reconstruction)
# axis. Same per-trial RMSE on the pooled neural Y channel; four lenses to
# read DBS-OFF vs DBS-ON and chronological drift.
#
# * Fig 64 — raincloud
# * Fig 65 — ECDF
# * Fig 66 — block-running median curve over 12 chronological bins
# * Fig 67 — train→val→test cascade with paired-median arrows

# %%
from thesis_sec2_common import (
    mpl_raincloud as _mpl_raincloud,
    mpl_ecdf as _mpl_ecdf,
    mpl_block_running as _mpl_block_running,
    mpl_cascade_train_val_test as _mpl_cascade,
)


def _collect_forecast_trial_metric(triplet_specs, channel_idx, metric, *,
                                   forecast_target="Y",
                                   neural_y_feature_name="ECOG_1_beta_27_30_raw",
                                   split="test"):
    """Pooled per-trial forecast metric across triplets. PSID + VARMA only;
    DPAD index left empty for compatibility with `mpl_raincloud` 6-cell layout
    (slots 0-1 PSID, 2-3 DPAD empty, 4-5 VARMA, OFF/ON paired)."""
    cells = [[] for _ in range(6)]
    cells_with_pid = [[] for _ in range(6)]
    for tri in triplet_specs:
        res_p = load_split_results_required(results_root, tri.psid_variant, tri.psid_run_ts, split)
        res_v = load_split_results_required(results_root, tri.varma_variant, tri.varma_run_ts, split)
        if forecast_target == "Y":
            try:
                ch_use = resolve_neural_y_channel_idx(res_p, neural_y_feature_name, channel_idx)
            except (ValueError, KeyError):
                ch_use = int(channel_idx)
        else:
            ch_use = int(channel_idx)
        mp = _key_index_map(res_p)
        mv = _key_index_map(res_v)
        common = set(mp.keys()) & set(mv.keys())
        for k in sorted(common, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))):
            i_p, i_v = mp[k], mv[k]
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue
            r_p = trial_metric_forecast_for_model(res_p, i_p, ch_use, metric, forecast_target=forecast_target)
            r_v = trial_metric_forecast_for_model(res_v, i_v, ch_use, metric, forecast_target=forecast_target)
            if not (np.isfinite(r_p) and np.isfinite(r_v)):
                continue
            pid = str(k[0])
            if stim == "off":
                cells[0].append(r_p); cells_with_pid[0].append((r_p, pid))
                cells[4].append(r_v); cells_with_pid[4].append((r_v, pid))
            else:
                cells[1].append(r_p); cells_with_pid[1].append((r_p, pid))
                cells[5].append(r_v); cells_with_pid[5].append((r_v, pid))
    means = tuple(float(np.mean(c)) if len(c) else float("nan") for c in cells)
    sems = tuple(_sem(np.array(c)) if len(c) > 1 else float("nan") for c in cells)
    return AggregateRmseData(
        means=means, sems=sems,
        trial_rmse=tuple(cells),
        trial_rmse_with_participant=tuple(cells_with_pid),
        n_triplets_used=len(triplet_specs),
    )


def _forecast_at(split):
    return _collect_forecast_trial_metric(
        ALL_TRIPLETS, 0, "rmse", forecast_target="Y",
        neural_y_feature_name="ECOG_1_beta_27_30_raw", split=split,
    )

_fc_test = _forecast_at("test")

fig = _mpl_raincloud(_fc_test, metric="rmse", target_label="pooled forecast")
fig.savefig(str(OUT / "fig_064_pooled_forecast_raincloud.png"))
plt.show()
print("Fig 64: raincloud — pooled per-trial Y forecast RMSE; per-model panels, DBS-OFF/ON columns.")

fig = _mpl_ecdf(_fc_test, metric="rmse", target_label="pooled forecast")
fig.savefig(str(OUT / "fig_065_pooled_forecast_ecdf.png"))
plt.show()
print("Fig 65: ECDF — pooled per-trial Y forecast RMSE per (model × DBS).")

fig = _mpl_block_running(_fc_test, metric="rmse", target_label="pooled forecast")
fig.savefig(str(OUT / "fig_066_pooled_forecast_block_running.png"))
plt.show()
print("Fig 66: block-running — chronological 12-block bins; median curve = drift signature.")

_fc_train = _forecast_at("train")
_fc_val = _forecast_at("val")
fig = _mpl_cascade(_fc_train, _fc_val, _fc_test, metric="rmse", target_label="pooled forecast")
fig.savefig(str(OUT / "fig_067_pooled_forecast_cascade.png"))
plt.show()
print("Fig 67: cascade — train→val→test per-split strip + median; arrow = chronological-drift gap.")


# %% [markdown]
# ## Figs 80-85 — Forecast group analyses (mirror of sec2c 70-75)
#
# Per-cell box (Y + Z forecast metrics) + pooled rainclouds (Y self-forecast,
# Y vs Z paired). Two modes: laplacian (figs 80-82) and behavioral (83-85).
# All three frameworks (PSID, DPAD, VARMA) — DPAD forecast head IS produced
# (Y_future_pred / Z_future_pred present in DPAD parquets).

# %%
def _collect_per_cell_forecast(triplets, *, target="Z", metric="pearson",
                                split="test"):
    """dict[(cell_label, model, dbs)] -> 1-D np.ndarray trial means across chans.
    Reads `*_future_pred / *_future_true` and reduces horizon window to one
    scalar per (trial, channel) via `trial_metric_forecast_for_model`."""
    out: dict = {}
    for tri in triplets:
        for model_name, variant, run_ts in (
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ):
            if not variant or not run_ts:
                continue
            try:
                res = load_split_results_required(results_root, variant, run_ts, split)
            except Exception:
                continue
            k_true = "Y_future_true" if target == "Y" else "Z_future_true"
            arr = res.get(k_true, [])
            if not arr:
                continue
            ref = np.asarray(arr[0]) if arr[0] is not None else None
            if ref is None or ref.ndim < 2:
                continue
            n_chans = ref.shape[-1]
            off_vals, on_vals = [], []
            for i in range(len(arr)):
                stim = normalize_stim(res["stim"][i])
                if stim is None:
                    continue
                ch_vals = []
                for ch in range(n_chans):
                    try:
                        v = trial_metric_forecast_for_model(
                            res, i, ch, metric, forecast_target=target)
                    except (ValueError, IndexError, KeyError):
                        v = float("nan")
                    if np.isfinite(v):
                        ch_vals.append(v)
                if not ch_vals:
                    continue
                (off_vals if stim == "off" else on_vals).append(float(np.mean(ch_vals)))
            out[(tri.label, model_name, "off")] = np.array(off_vals, dtype=float)
            out[(tri.label, model_name, "on")] = np.array(on_vals, dtype=float)
    return out


# %% [markdown]
# ### Laplacian forecast (figs 80-82)

# %%
_lap_cells_fc = [t.label for t in ALL_TRIPLETS_LAP]
_lap_y_r_fc = _collect_per_cell_forecast(ALL_TRIPLETS_LAP, target="Y", metric="pearson")
_lap_y_n_fc = _collect_per_cell_forecast(ALL_TRIPLETS_LAP, target="Y", metric="rmse")
_lap_z_r_fc = _collect_per_cell_forecast(ALL_TRIPLETS_LAP, target="Z", metric="pearson")
_lap_z_n_fc = _collect_per_cell_forecast(ALL_TRIPLETS_LAP, target="Z", metric="rmse")

fig = mpl_per_cell_yz_box(
    _lap_y_r_fc, _lap_y_n_fc, _lap_z_r_fc, _lap_z_n_fc, _lap_cells_fc)
fig.savefig(str(OUT / "fig_080_lap_per_cell_yz_box_forecast.png"), bbox_inches="tight")
plt.show()
print("Fig 80 (lap forecast): 4 rows × 4 cells. Y self-forecast + Z forecast.")

# %%
_lap_y_r_fc_pool = pool_dbs_cells(_lap_y_r_fc, _lap_cells_fc)
_lap_y_n_fc_pool = pool_dbs_cells(_lap_y_n_fc, _lap_cells_fc)
fig = mpl_raincloud_dbs_pair_vert(_lap_y_r_fc_pool, _lap_y_n_fc_pool)
fig.savefig(str(OUT / "fig_081_lap_pool_raincloud_y_forecast.png"), bbox_inches="tight")
plt.show()
print("Fig 81 (Y self-forecast, lap, pooled): vertical raincloud per framework. DBS-OFF/ON.")

# %%
_lap_yz_r_fc_pool = pool_yz_dbs_cells(_lap_y_r_fc, _lap_z_r_fc, _lap_cells_fc)
_lap_yz_n_fc_pool = pool_yz_dbs_cells(_lap_y_n_fc, _lap_z_n_fc, _lap_cells_fc)
fig = mpl_raincloud_yz_pair_vert(_lap_yz_r_fc_pool, _lap_yz_n_fc_pool)
fig.savefig(str(OUT / "fig_082_lap_pool_raincloud_yz_forecast.png"), bbox_inches="tight")
plt.show()
print("Fig 82 (Y vs Z forecast, lap, pooled): vertical raincloud, Y left + Z right per framework.")


# %% [markdown]
# ### Behavioral forecast (figs 83-85)

# %%
_beh_cells_fc = [t.label for t in ALL_TRIPLETS]
_beh_y_r_fc = _collect_per_cell_forecast(ALL_TRIPLETS, target="Y", metric="pearson")
_beh_y_n_fc = _collect_per_cell_forecast(ALL_TRIPLETS, target="Y", metric="rmse")
_beh_z_r_fc = _collect_per_cell_forecast(ALL_TRIPLETS, target="Z", metric="pearson")
_beh_z_n_fc = _collect_per_cell_forecast(ALL_TRIPLETS, target="Z", metric="rmse")

fig = mpl_per_cell_yz_box(
    _beh_y_r_fc, _beh_y_n_fc, _beh_z_r_fc, _beh_z_n_fc, _beh_cells_fc)
fig.savefig(str(OUT / "fig_083_beh_per_cell_yz_box_forecast.png"), bbox_inches="tight")
plt.show()
print("Fig 83 (beh forecast): 4 rows × 4 cells. Y self-forecast + Z (behavior) forecast.")

# %%
_beh_y_r_fc_pool = pool_dbs_cells(_beh_y_r_fc, _beh_cells_fc)
_beh_y_n_fc_pool = pool_dbs_cells(_beh_y_n_fc, _beh_cells_fc)
fig = mpl_raincloud_dbs_pair_vert(_beh_y_r_fc_pool, _beh_y_n_fc_pool)
fig.savefig(str(OUT / "fig_084_beh_pool_raincloud_y_forecast.png"), bbox_inches="tight")
plt.show()
print("Fig 84 (Y self-forecast, beh, pooled): vertical raincloud per framework. DBS-OFF/ON.")

# %%
_beh_yz_r_fc_pool = pool_yz_dbs_cells(_beh_y_r_fc, _beh_z_r_fc, _beh_cells_fc)
_beh_yz_n_fc_pool = pool_yz_dbs_cells(_beh_y_n_fc, _beh_z_n_fc, _beh_cells_fc)
fig = mpl_raincloud_yz_pair_vert(_beh_yz_r_fc_pool, _beh_yz_n_fc_pool)
fig.savefig(str(OUT / "fig_085_beh_pool_raincloud_yz_forecast.png"), bbox_inches="tight")
plt.show()
print("Fig 85 (Y vs Z forecast, beh, pooled): vertical raincloud, Y left + Z right per framework.")


# %% [markdown]
# ## Fig 86 — Pooled forecast vs horizon (RMSE + Pearson r)
#
# How forecast quality decays with horizon. Top: per-step RMSE (z-scored).
# Bottom: per-step Pearson r across trials. Pooled across all behavioral
# cells. Lines: framework hue (PSID/DPAD/VARMA). Style: solid = DBS-OFF,
# dashed = DBS-ON. Shaded band = SEM across trials.

# %%
def _collect_horizon_per_step(triplets, *, target="Y", channel_idx=0,
                                neural_y_feature_name="ECOG_1_beta_27_30_raw",
                                split="test"):
    """Per (model, dbs), pool across triplets/trials a (n_trials, n_steps) matrix
    of (true, pred). Trims to common horizon length across triplets."""
    from thesis_lib.transforms import reshape_future_z_time_first
    k_true = "Y_future_true" if target == "Y" else "Z_future_true"
    k_pred = "Y_future_pred" if target == "Y" else "Z_future_pred"
    out: dict = {(m, d): {"true": [], "pred": []}
                  for m in ("PSID", "DPAD", "VARMA")
                  for d in ("off", "on")}
    for tri in triplets:
        for model_name, variant, run_ts in (
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ):
            if not variant or not run_ts:
                continue
            try:
                res = load_split_results_required(results_root, variant, run_ts, split)
            except Exception:
                continue
            arr_t = res.get(k_true, [])
            arr_p = res.get(k_pred, [])
            if not arr_t or not arr_p:
                continue
            if target == "Y":
                try:
                    ch_use = resolve_neural_y_channel_idx(
                        res, neural_y_feature_name, channel_idx)
                except (ValueError, KeyError):
                    ch_use = int(channel_idx)
            else:
                ch_use = int(channel_idx)
            for i in range(len(arr_t)):
                if arr_t[i] is None or arr_p[i] is None:
                    continue
                stim = normalize_stim(res["stim"][i])
                if stim is None:
                    continue
                try:
                    T = reshape_future_z_time_first(np.asarray(arr_t[i], dtype=float))
                    P = reshape_future_z_time_first(np.asarray(arr_p[i], dtype=float))
                except (ValueError, KeyError):
                    continue
                if T.shape != P.shape:
                    continue
                if ch_use >= T.shape[1]:
                    continue
                t_vec = T[:, ch_use]
                p_vec = P[:, ch_use]
                msk = np.isfinite(t_vec) & np.isfinite(p_vec)
                if not msk.all():
                    continue
                mu, sigma = float(np.mean(t_vec)), float(np.std(t_vec))
                if sigma < 1e-12:
                    sigma = 1.0
                out[(model_name, stim)]["true"].append((t_vec - mu) / sigma)
                out[(model_name, stim)]["pred"].append((p_vec - mu) / sigma)
    res_arrays: dict = {}
    for k, d in out.items():
        if not d["true"]:
            continue
        m_min = min(t.size for t in d["true"])
        if m_min == 0:
            continue
        T = np.stack([t[:m_min] for t in d["true"]], axis=0)
        P = np.stack([p[:m_min] for p in d["pred"]], axis=0)
        res_arrays[k] = (T, P)
    return res_arrays


def _per_step_rmse_and_pearson(T, P):
    """Returns (x indices, rmse_per_step, sem_rmse, pearson_per_step)."""
    n_trials, n_steps = T.shape
    err = T - P
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    sem = np.std(err ** 2, axis=0, ddof=1) / np.sqrt(n_trials) / (2 * rmse + 1e-12)
    rs = np.full(n_steps, np.nan)
    for s in range(n_steps):
        t = T[:, s]; p = P[:, s]
        if np.std(t) > 1e-12 and np.std(p) > 1e-12 and n_trials > 1:
            rs[s] = float(np.corrcoef(t, p)[0, 1])
    return rmse, sem, rs


# %%
SAMPLING_HZ = 200.0
SAMPLE_EVERY = 5
DT_MS = 1000.0 / SAMPLING_HZ
_horizon_arrays = _collect_horizon_per_step(
    ALL_TRIPLETS, target="Y", channel_idx=5,
    neural_y_feature_name="ECOG_1_beta_27_30_raw")

fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
ax_rmse, ax_r = axes
panel_label(ax_rmse, "A", "Forecast NRMSE vs horizon")
panel_label(ax_r,    "B", "Pearson r vs horizon")
ax_rmse.axhline(1.0, color="black", linestyle=":", linewidth=0.6,
                alpha=0.30, zorder=0)
ax_r.axhline(0.0,    color="black", linestyle=":", linewidth=0.6,
             alpha=0.30, zorder=0)
model_color = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}
dbs_style = {"off": "-", "on": "--"}
for (m, d), (T, P) in _horizon_arrays.items():
    idx = np.arange(0, T.shape[1], SAMPLE_EVERY)
    x_ms = idx.astype(float) * DT_MS
    rmse, sem, rs = _per_step_rmse_and_pearson(T[:, idx], P[:, idx])
    color = model_color[m]
    style = dbs_style[d]
    label = f"{m} {d.upper()}"
    ax_rmse.plot(x_ms, rmse, color=color, linestyle=style, linewidth=1.4,
                 label=label)
    ax_rmse.fill_between(x_ms, rmse - sem, rmse + sem, color=color,
                          alpha=0.12, linewidth=0)
    ax_r.plot(x_ms, rs, color=color, linestyle=style, linewidth=1.4)
ax_rmse.set_ylabel("NRMSE")
ax_rmse.set_yscale("log")
ax_r.set_ylabel("Pearson r")
ax_r.set_xlabel("Horizon (ms)")
ax_rmse.legend(loc="best", frameon=False, fontsize=7, ncol=2)
fig.savefig(str(OUT / "fig_086_pool_horizon_rmse_pearson.png"),
             bbox_inches="tight")
plt.show()
print(
    "Fig 86: pooled forecast vs horizon. RMSE (top, log y) + Pearson r (bottom). "
    "Solid = DBS-OFF, dashed = DBS-ON. Hue = framework. "
    "Confirms VARMA AR collapses to mean (r drops fast); PSID/DPAD horizon shape."
)
