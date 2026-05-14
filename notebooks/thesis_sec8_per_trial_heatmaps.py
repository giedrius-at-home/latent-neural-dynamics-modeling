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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Section 8: Per-Trial × Per-Channel Test-Set Heatmaps
#
# Detailed breakdown of PSID model performance on the **test set** for each session.
# Also used to identify representative trials for the sec2 forecast exemplars.
#
# | # | Figure | Description |
# |---|--------|-------------|
# | 1 | Neural prediction Pearson R | trials × 60 neural channels |
# | 2 | Neural prediction RMSE | trials × 60 neural channels |
# | 3 | Neural forecast Pearson R | trials × 60 neural channels (Y_future) |
# | 4 | Neural forecast RMSE | trials × 60 neural channels (Y_future) |
# | 5 | Behavioral prediction Pearson R + RMSE | trials × 2 behavioral channels |
# | 6 | Behavioral forecast Pearson R + RMSE | trials × 2 behavioral channels (Z_future) |
# | 7 | Mean neural Pearson R summary | electrodes × bands, all sessions |
#
# Each figure type is generated for all 4 sessions (PSID model, 200Hz narrow band).
# Raw feature names (ECOG_*_raw, tracing_*) are kept verbatim.

# %%
import sys, os

os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")
sys.path.insert(0, ".")
sys.path.insert(0, "notebooks")

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from thesis_style import apply_thesis_style, panel_label

apply_thesis_style()

OUT = Path("thesis_figures/sec8")
OUT.mkdir(parents=True, exist_ok=True)
results_root = Path("results").resolve()

# %% [markdown]
# ## Configuration
#
# 4 sessions with PSID model timestamps (test split). Pulled from the canonical
# `thesis_triplets.csv` via `thesis_sec2_common.ALL_TRIPLETS` so this notebook
# automatically tracks the current PSID runs without manual edits.

# %%
from thesis_sec2_common import ALL_TRIPLETS

SESSIONS = [
    {
        "label": tri.label.replace("_", " "),
        "variant": tri.psid_variant,
        "run_ts": tri.psid_run_ts,
    }
    for tri in ALL_TRIPLETS
]

BEHAV_CHANNELS = ["tracing_velocity_x", "tracing_acceleration_magnitude"]

# %% [markdown]
# ## Data loading

# %%
from thesis_lib.loaders import load_split_results_required
from thesis_loaders import resolve_input_channels


def load_test_results(session):
    """Load test-set results for one session. Returns the results dict."""
    return load_split_results_required(
        results_root, session["variant"], session["run_ts"], "test"
    )


def channels_for(session, res):
    """Input (Y) channel names with YAML fallback — current PSID parquets have
    empty ``input_channels`` and need the training YAML to name the 60 ECoG bands."""
    return resolve_input_channels(res, session["variant"])


def get_channel_names(res, variant):
    """Get neural channel names (via YAML fallback), sorted by electrode then band."""
    channels = resolve_input_channels(res, variant)

    # Sort: electrode number, then band frequency
    def sort_key(ch):
        parts = ch.replace("ECOG_", "").replace("_raw", "").split("_", 1)
        electrode = int(parts[0])
        band = parts[1] if len(parts) > 1 else ""
        # Extract start frequency for ordering
        freq = 0
        for token in band.split("_"):
            try:
                freq = int(token)
                break
            except ValueError:
                continue
        return (electrode, freq, band)

    return sorted(channels, key=sort_key)


def trial_label(res, i):
    """B11.T3 ON etc."""
    stim = res["stim"][i].upper() if res["stim"][i] else "?"
    return f'B{res["block"][i]}.T{res["trial"][i]} {stim}'


# Sort trial indices: by block, then trial, grouping by stim
def sorted_trial_indices(res):
    """Return trial indices sorted by stim (OFF first), then block, then trial."""
    n = len(res["stim"])
    indices = list(range(n))

    def key(i):
        stim_order = 0 if res["stim"][i] == "off" else 1
        return (stim_order, res["block"][i], res["trial"][i])

    return sorted(indices, key=key)


# %% [markdown]
# ## Helper: compute per-trial per-channel RMSE


# %%
def compute_per_trial_rmse_neural(res, channel_order, raw_channels):
    """Compute RMSE for each trial × neural channel from Y vs Yp.

    Returns (n_trials, n_channels) array in the order of channel_order.
    """
    channels = raw_channels
    # Build channel index mapping: channel_order[j] -> position in Y columns
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    col_indices = [ch_to_idx[ch] for ch in channel_order]

    n_trials = len(res["Y"])
    n_ch = len(channel_order)
    rmse_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        y_true = np.array(res["Y"][i])  # (T, n_channels)
        y_pred = np.array(res["Yp"][i])  # (T, n_channels)
        for j, ci in enumerate(col_indices):
            diff = y_true[:, ci] - y_pred[:, ci]
            rmse_matrix[i, j] = np.sqrt(np.mean(diff**2))

    return rmse_matrix


def compute_per_trial_pearson_neural(res, channel_order, raw_channels):
    """Extract per-trial Pearson R for neural channels from precomputed results.

    Returns (n_trials, n_channels) array in the order of channel_order.
    """
    channels = raw_channels
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    col_indices = [ch_to_idx[ch] for ch in channel_order]

    n_trials = len(res["pearson_per_channel"])
    n_ch = len(channel_order)
    r_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        ppc = res["pearson_per_channel"][i]
        if ppc is not None:
            ppc = np.array(ppc)
            for j, ci in enumerate(col_indices):
                if ci < len(ppc):
                    r_matrix[i, j] = ppc[ci]

    return r_matrix


def compute_per_trial_metrics_behavioral(res):
    """Compute per-trial Pearson R and RMSE for behavioral channels (Z vs Zp).

    Returns (pearson_matrix, rmse_matrix) each of shape (n_trials, 2).
    """
    n_trials = len(res["Z"])
    n_ch = 2  # velocity_x, accel_mag
    r_matrix = np.full((n_trials, n_ch), np.nan)
    rmse_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        z_true = np.array(res["Z"][i])  # (T, 2)
        z_pred = np.array(res["Zp"][i])  # (T, 2)
        for j in range(min(n_ch, z_true.shape[1])):
            zt = z_true[:, j]
            zp = z_pred[:, j]
            # RMSE
            rmse_matrix[i, j] = np.sqrt(np.mean((zt - zp) ** 2))
            # Pearson R
            mask = np.isfinite(zt) & np.isfinite(zp)
            if mask.sum() > 2:
                r_matrix[i, j] = np.corrcoef(zt[mask], zp[mask])[0, 1]

    return r_matrix, rmse_matrix


def compute_per_trial_metrics_neural_forecast(res, channel_order, raw_channels):
    """Per-trial × per-channel Pearson R and RMSE for the neural forecast.

    Uses Y_future_true / Y_future_pred (shape (T_forecast, n_neural)). Channels are
    reordered to match channel_order. Returns (r_matrix, rmse_matrix), both
    (n_trials, len(channel_order)), or (None, None) if forecast data absent.
    """
    if "Y_future_true" not in res or res.get("Y_future_true") is None:
        return None, None

    from thesis_lib.transforms import reshape_future_z_time_first

    channels = raw_channels
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    col_indices = [ch_to_idx[ch] for ch in channel_order]

    yft = res["Y_future_true"]
    yfp = res["Y_future_pred"]
    n_trials = len(yft)
    n_ch = len(channel_order)
    r_matrix = np.full((n_trials, n_ch), np.nan)
    rmse_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        if yft[i] is None or yfp[i] is None:
            continue
        try:
            T = reshape_future_z_time_first(np.asarray(yft[i], dtype=float))
            P = reshape_future_z_time_first(np.asarray(yfp[i], dtype=float))
        except ValueError:
            continue
        if T.shape != P.shape:
            continue
        for j, ci in enumerate(col_indices):
            if ci >= T.shape[1]:
                continue
            t_vec = T[:, ci]
            p_vec = P[:, ci]
            mask = np.isfinite(t_vec) & np.isfinite(p_vec)
            if mask.sum() < 3:
                continue
            mu = float(np.mean(t_vec[mask]))
            sigma = float(np.std(t_vec[mask]))
            if sigma < 1e-12:
                sigma = 1.0
            zt = (t_vec - mu) / sigma
            zp = (p_vec - mu) / sigma
            rmse_matrix[i, j] = float(np.sqrt(np.nanmean((zt - zp) ** 2)))
            r_matrix[i, j] = float(np.corrcoef(t_vec[mask], p_vec[mask])[0, 1])

    return r_matrix, rmse_matrix


def compute_per_trial_metrics_forecast(res):
    """Compute per-trial Pearson R and RMSE for behavioral forecasts.

    Uses Z_future_true and Z_future_pred from the results.
    Returns (pearson_matrix, rmse_matrix) each of shape (n_trials, 2), or None if no forecast data.
    """
    if "Z_future_true" not in res or res.get("Z_future_true") is None:
        return None, None

    zft = res["Z_future_true"]
    zfp = res["Z_future_pred"]
    n_trials = len(zft)
    n_ch = 2
    r_matrix = np.full((n_trials, n_ch), np.nan)
    rmse_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        if zft[i] is None or zfp[i] is None:
            continue
        zt = np.array(zft[i])  # (T_forecast, 2)
        zp = np.array(zfp[i])
        if zt.ndim < 2 or zp.ndim < 2:
            continue
        for j in range(min(n_ch, zt.shape[1])):
            z_t = zt[:, j]
            z_p = zp[:, j]
            rmse_matrix[i, j] = np.sqrt(np.mean((z_t - z_p) ** 2))
            mask = np.isfinite(z_t) & np.isfinite(z_p)
            if mask.sum() > 2:
                r_matrix[i, j] = np.corrcoef(z_t[mask], z_p[mask])[0, 1]

    return r_matrix, rmse_matrix


# %% [markdown]
# ## Neural Prediction — Pearson R Heatmap (test set)
#
# Rows = test trials (sorted OFF first, then ON), columns = 60 neural channels
# (grouped by electrode, then frequency band).

# %%
# Collect per-session data once so we can reuse it for RMSE heatmaps, forecast heatmaps,
# and representative-trial selection at the end without reloading results 4 times.
SESSION_DATA = {}

for sess in SESSIONS:
    res = load_test_results(sess)
    # Raw order (from parquet OR YAML fallback) and the sort-for-plotting order.
    raw_channels = resolve_input_channels(res, sess["variant"])
    ch_order = get_channel_names(res, sess["variant"])
    trial_order = sorted_trial_indices(res)
    t_labels = [trial_label(res, i) for i in trial_order]

    r_mat_pred = compute_per_trial_pearson_neural(res, ch_order, raw_channels)
    rmse_mat_pred = compute_per_trial_rmse_neural(res, ch_order, raw_channels)
    r_mat_behav, rmse_mat_behav = compute_per_trial_metrics_behavioral(res)
    r_mat_zfore, rmse_mat_zfore = compute_per_trial_metrics_forecast(res)

    SESSION_DATA[sess["label"]] = dict(
        res=res,
        ch_order=ch_order,
        raw_channels=raw_channels,
        trial_order=trial_order,
        t_labels=t_labels,
        r_pred=r_mat_pred,
        rmse_pred=rmse_mat_pred,
        r_behav=r_mat_behav,
        rmse_behav=rmse_mat_behav,
        r_zfore=r_mat_zfore,
        rmse_zfore=rmse_mat_zfore,
    )

    r_sorted = r_mat_pred[trial_order, :]
    n_trials = len(t_labels)
    n_ch = len(ch_order)

    # Width scales with channel count so long raw names fit rotated; height scales with
    # trial count so labels remain readable without overlap.
    fig_w = max(9.0, n_ch * 0.14 + 2.0)
    fig_h = max(3.5, n_trials * 0.13 + 1.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        r_sorted,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(n_ch))
    ax.set_xticklabels(ch_order, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n_trials))
    ax.set_yticklabels(t_labels, fontsize=6)
    ax.set_xlabel("neural channel (raw name)")
    ax.set_ylabel("trial (block.trial stim)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    panel_label(ax, "A", f"Neural prediction Pearson r — {sess['label']}")
    fig.savefig(OUT / f'neural_pearson_{sess["label"].replace(" ", "_")}.png')
    plt.show()
print(
    f"Neural prediction Pearson r per trial × channel ({len(SESSIONS)} sessions, PSID test split). "
    f"Rows = test trials sorted DBS-OFF first then DBS-ON (block.trial stim labels). "
    f"Columns = all {len(SESSION_DATA[SESSIONS[0]['label']]['ch_order'])} narrow-band ECoG channels "
    f"(raw feature names retained, sorted by electrode then band centre frequency). "
    f"Red = positive r, blue = negative. Use this to spot which channels/trials the "
    f"reconstruction fits well vs poorly."
)

# %% [markdown]
# ## Neural Prediction — RMSE Heatmap (test set)

# %%
for sess in SESSIONS:
    sd = SESSION_DATA[sess["label"]]
    rmse_sorted = sd["rmse_pred"][sd["trial_order"], :]
    n_trials = len(sd["t_labels"])
    n_ch = len(sd["ch_order"])

    fig_w = max(9.0, n_ch * 0.14 + 2.0)
    fig_h = max(3.5, n_trials * 0.13 + 1.2)

    rmse_vmax = float(np.nanmax(rmse_sorted)) if np.isfinite(rmse_sorted).any() else 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        rmse_sorted,
        cmap="Blues",
        vmin=0,
        vmax=rmse_vmax,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(n_ch))
    ax.set_xticklabels(sd["ch_order"], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n_trials))
    ax.set_yticklabels(sd["t_labels"], fontsize=6)
    ax.set_xlabel("neural channel (raw name)")
    ax.set_ylabel("trial (block.trial stim)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="RMSE")
    panel_label(ax, "A", f"Neural prediction RMSE — {sess['label']}")
    fig.savefig(OUT / f'neural_rmse_{sess["label"].replace(" ", "_")}.png')
    plt.show()
print(
    "Neural prediction RMSE per trial × channel. Same layout as Pearson r heatmap but "
    "sequential colorscale (darker blue = higher RMSE). Complements the r view: "
    "a channel can look 'red' in r (correctly tracks shape) yet still have a biased offset "
    "inflating RMSE."
)

# %% [markdown]
# ## Neural Forecast — Pearson R & RMSE Heatmaps (test set)
#
# Same layout as the neural prediction heatmaps but using ``Y_future_true`` /
# ``Y_future_pred`` — tells us which channels the PSID autoregressive forecast handles
# well and which it can't extrapolate.

# %%
for sess in SESSIONS:
    sd = SESSION_DATA[sess["label"]]
    r_neu_f, rmse_neu_f = compute_per_trial_metrics_neural_forecast(
        sd["res"], sd["ch_order"], sd["raw_channels"]
    )
    if r_neu_f is None:
        print(f"  {sess['label']}: no Y_future data, skipping neural forecast heatmap.")
        continue
    sd["r_yfore"] = r_neu_f
    sd["rmse_yfore"] = rmse_neu_f

    trial_order = sd["trial_order"]
    r_sorted = r_neu_f[trial_order, :]
    rmse_sorted = rmse_neu_f[trial_order, :]
    n_trials = len(sd["t_labels"])
    n_ch = len(sd["ch_order"])

    fig_w = max(9.0, n_ch * 0.14 + 2.0)
    fig_h = max(3.5, n_trials * 0.13 + 1.2)

    # Pearson r panel
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        r_sorted,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(n_ch))
    ax.set_xticklabels(sd["ch_order"], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n_trials))
    ax.set_yticklabels(sd["t_labels"], fontsize=6)
    ax.set_xlabel("neural channel (raw name)")
    ax.set_ylabel("trial (block.trial stim)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    panel_label(ax, "A", f"Neural forecast Pearson r — {sess['label']}")
    fig.savefig(OUT / f'neural_forecast_pearson_{sess["label"].replace(" ", "_")}.png')
    plt.show()

    # RMSE panel
    rmse_vmax = float(np.nanmax(rmse_sorted)) if np.isfinite(rmse_sorted).any() else 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        rmse_sorted,
        cmap="Blues",
        vmin=0,
        vmax=rmse_vmax,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(n_ch))
    ax.set_xticklabels(sd["ch_order"], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n_trials))
    ax.set_yticklabels(sd["t_labels"], fontsize=6)
    ax.set_xlabel("neural channel (raw name)")
    ax.set_ylabel("trial (block.trial stim)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="RMSE(z)")
    panel_label(ax, "A", f"Neural forecast RMSE(z) — {sess['label']}")
    fig.savefig(OUT / f'neural_forecast_rmse_{sess["label"].replace(" ", "_")}.png')
    plt.show()
print(
    "Neural forecast Pearson r and RMSE per trial × channel. Metrics are computed on "
    "Y_future_true vs Y_future_pred over the full forecast horizon; RMSE is on the "
    "per-trial z-scored true signal so values are directly comparable to the RMSE(z) axis "
    "used in sec2. Use these heatmaps to pick the representative trials shown in sec2's "
    "neural forecast exemplars (highest mean r across channels = best to plot)."
)

# %% [markdown]
# ## Behavioral Prediction — Pearson R & RMSE (test set)
#
# Two panels side by side for the two behavioural outputs, using raw feature names.


# %%
def _behavioral_pair_heatmap(
    sess_label, t_labels, r_sorted, rmse_sorted, png_name, *, r_title, rmse_title
):
    n_trials = len(t_labels)
    fig_h = max(3.5, n_trials * 0.15 + 1.2)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, fig_h), sharey=True)

    # Panel A: Pearson r
    ax = axes[0]
    im_r = ax.imshow(
        r_sorted,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(BEHAV_CHANNELS)))
    ax.set_xticklabels(BEHAV_CHANNELS, rotation=20, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n_trials))
    ax.set_yticklabels(t_labels, fontsize=6)
    ax.set_ylabel("trial (block.trial stim)")
    fig.colorbar(im_r, ax=ax, shrink=0.7, label="Pearson r")
    panel_label(ax, "A", r_title)

    # Panel B: RMSE
    ax = axes[1]
    rmse_vmax = float(np.nanmax(rmse_sorted)) if np.isfinite(rmse_sorted).any() else 1.0
    im_e = ax.imshow(
        rmse_sorted,
        cmap="Blues",
        vmin=0,
        vmax=rmse_vmax,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(BEHAV_CHANNELS)))
    ax.set_xticklabels(BEHAV_CHANNELS, rotation=20, ha="right", fontsize=7)
    fig.colorbar(im_e, ax=ax, shrink=0.7, label="RMSE")
    panel_label(ax, "B", rmse_title)

    fig.savefig(OUT / png_name)
    plt.show()


for sess in SESSIONS:
    sd = SESSION_DATA[sess["label"]]
    r_sorted = sd["r_behav"][sd["trial_order"], :]
    rmse_sorted = sd["rmse_behav"][sd["trial_order"], :]
    _behavioral_pair_heatmap(
        sess["label"],
        sd["t_labels"],
        r_sorted,
        rmse_sorted,
        png_name=f'behav_pred_{sess["label"].replace(" ", "_")}.png',
        r_title=f"Behav prediction Pearson r — {sess['label']}",
        rmse_title=f"Behav prediction RMSE — {sess['label']}",
    )
print(
    "Behavioural prediction Pearson r (left) and RMSE (right) per trial for both "
    "outputs (tracing_velocity_x, tracing_acceleration_magnitude). RMSE is computed "
    "directly on the supplied Zp / Z columns (thesis z-scored units)."
)

# %% [markdown]
# ## Behavioral Forecast — Pearson R & RMSE (test set)
#
# Same layout as the behavioural prediction pair but using ``Z_future_true`` vs
# ``Z_future_pred`` (forecast horizon).

# %%
for sess in SESSIONS:
    sd = SESSION_DATA[sess["label"]]
    if sd["r_zfore"] is None:
        print(f"  {sess['label']}: no Z_future data, skipping.")
        continue
    r_sorted = sd["r_zfore"][sd["trial_order"], :]
    rmse_sorted = sd["rmse_zfore"][sd["trial_order"], :]
    _behavioral_pair_heatmap(
        sess["label"],
        sd["t_labels"],
        r_sorted,
        rmse_sorted,
        png_name=f'behav_forecast_{sess["label"].replace(" ", "_")}.png',
        r_title=f"Behav forecast Pearson r — {sess['label']}",
        rmse_title=f"Behav forecast RMSE — {sess['label']}",
    )
print(
    "Behavioural forecast Pearson r and RMSE per trial. Same layout as the prediction "
    "version; metrics computed on Z_future_true vs Z_future_pred over the forecast horizon."
)

# %% [markdown]
# ## Summary: Mean Pearson R by electrode and band (test set)
#
# Aggregated heatmap: electrodes × frequency bands, mean Pearson R across test trials.
# Uses raw band-key suffixes from the channel names (no reformatting).

# %%
import re

# Raw band-key suffixes exactly as they appear inside ECOG_<e>_<band_key>_raw.
BAND_KEYS = [
    "theta_4_8",
    "alpha_8_12",
    "beta_12_17",
    "beta_17_22",
    "beta_22_27",
    "beta_27_30",
    "gamma_30_35",
    "gamma_35_40",
    "gamma_40_45",
    "gamma_45_50",
    "gamma_50_55",
    "gamma_55_60",
    "gamma_60_65",
    "gamma_70_75",
    "gamma_75_80",
]
ELECTRODE_LABELS = ["ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"]

n_sessions = len(SESSIONS)
fig, axes = plt.subplots(1, n_sessions, figsize=(3.4 * n_sessions, 2.6), sharey=True)
panel_letters = ["A", "B", "C", "D", "E", "F"]

im = None
for si, sess in enumerate(SESSIONS):
    ax = axes[si]
    sd = SESSION_DATA[sess["label"]]
    ch_order = sd["ch_order"]
    mean_r = np.nanmean(sd["r_pred"], axis=0)  # (n_ch,)

    grid = np.full((len(ELECTRODE_LABELS), len(BAND_KEYS)), np.nan)
    for ci, ch in enumerate(ch_order):
        m = re.match(r"ECOG_(\d+)_(.+)_raw", ch)
        if not m:
            continue
        electrode = int(m.group(1)) - 1
        band_key = m.group(2)
        for bi, bk in enumerate(BAND_KEYS):
            if band_key == bk or band_key.startswith(bk):
                grid[electrode, bi] = mean_r[ci]
                break

    im = ax.imshow(
        grid,
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(BAND_KEYS)))
    ax.set_xticklabels(BAND_KEYS, rotation=45, ha="right", fontsize=6)
    if si == 0:
        ax.set_yticks(np.arange(len(ELECTRODE_LABELS)))
        ax.set_yticklabels(ELECTRODE_LABELS, fontsize=7)
    panel_label(ax, panel_letters[si], sess["label"])

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="r")
fig.savefig(OUT / "neural_pearson_summary.png")
plt.show()
print(
    "Mean Pearson r aggregated over test trials, reorganised into a (4 electrodes × "
    f"{len(BAND_KEYS)} bands) grid for each session. Raw band keys are kept verbatim. "
    "Highlights which electrode × band entries the PSID model reconstructs most reliably."
)

# %% [markdown]
# ## Representative-trial selection for sec2 forecast exemplars
#
# For each session and DBS condition, rank test trials by mean forecast quality and
# list the top few so they can be dropped straight into sec2's forecast exemplar figures.
# "Mean neural r" = average Pearson r across all neural channels in ``Y_future``.
# "Mean behav r" = average across (velocity_x, accel_mag) in ``Z_future``.


# %%
def _top_trials_for_sec2(sess_label: str, sd: dict, k: int = 3) -> None:
    res = sd["res"]
    n_trials = len(res["stim"])
    stim = [res["stim"][i] for i in range(n_trials)]
    blocks = [res["block"][i] for i in range(n_trials)]
    trials = [res["trial"][i] for i in range(n_trials)]

    has_y_f = sd.get("r_yfore") is not None
    has_z_f = sd.get("r_zfore") is not None
    mean_r_y = (
        np.nanmean(sd["r_yfore"], axis=1) if has_y_f else np.full(n_trials, np.nan)
    )
    mean_r_z = (
        np.nanmean(sd["r_zfore"], axis=1) if has_z_f else np.full(n_trials, np.nan)
    )
    mean_rmse_y = (
        np.nanmean(sd["rmse_yfore"], axis=1) if has_y_f else np.full(n_trials, np.nan)
    )
    mean_rmse_z = (
        np.nanmean(sd["rmse_zfore"], axis=1) if has_z_f else np.full(n_trials, np.nan)
    )

    print(f"  -- {sess_label} --")
    for cond in ("off", "on"):
        idxs = [i for i in range(n_trials) if stim[i] == cond]
        if not idxs:
            continue
        # Rank by (mean neural forecast r + mean behavioural forecast r) — picks trials
        # where both modalities look good in the forecast.
        scored = []
        for i in idxs:
            score = 0.0
            if has_y_f and np.isfinite(mean_r_y[i]):
                score += mean_r_y[i]
            if has_z_f and np.isfinite(mean_r_z[i]):
                score += mean_r_z[i]
            scored.append((score, i))
        scored.sort(key=lambda x: -x[0])
        top = scored[:k]
        for rank, (_, i) in enumerate(top, 1):
            print(
                f"    DBS-{cond.upper()} #{rank}: B{blocks[i]}.T{trials[i]} "
                f"(neural r={mean_r_y[i]:.3f}, behav r={mean_r_z[i]:.3f}, "
                f"neural RMSE={mean_rmse_y[i]:.3f}, behav RMSE={mean_rmse_z[i]:.3f})"
            )


print("Representative trials (top 3 by mean forecast Pearson r across modalities):")
for sess in SESSIONS:
    _top_trials_for_sec2(sess["label"], SESSION_DATA[sess["label"]], k=3)
print(
    "\nUse the top DBS-OFF and DBS-ON entries as the (block, trial) arguments in sec2's "
    "neural/behavioural forecast exemplar figures."
)

# %%
n = len(list(OUT.glob("*.png")))
print(f"Section 8 total: {n} figures saved")
