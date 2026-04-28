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
os.chdir('/home/bobby/repos/latent-neural-dynamics-modeling')
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Canonical thesis palette + style helper (consistent with sec1–sec6 figures)
from notebooks.thesis_style import (
    FONT_FAMILY, FONT_SIZE_BASE, FONT_SIZE_LABEL,
    apply_thesis_style, ThesisTheme,
)

OUT = Path('thesis_figures/sec8'); OUT.mkdir(parents=True, exist_ok=True)
results_root = Path('results').resolve()

# Thesis-aligned colorscales (mirrors neural_band_heatmap_figure._BLUE_SEQUENTIAL
# and psid_cy_importance_figure._DIVERGING_BWR — same hues used in sec2/sec6).
COLORSCALE_DIVERGING = [
    [0.0, "rgb(24, 95, 165)"],   # PSID blue (negative)
    [0.5, "rgb(250, 250, 250)"],
    [1.0, "rgb(220, 50, 32)"],   # red (positive)
]
COLORSCALE_SEQUENTIAL = [
    [0.0, "rgb(250, 250, 250)"],
    [0.35, "rgb(200, 215, 235)"],
    [0.65, "rgb(100, 150, 205)"],
    [1.0, "rgb(24, 95, 165)"],   # PSID blue
]

# %% [markdown]
# ## Configuration
#
# 4 sessions with PSID model timestamps (test split).

# %%
SESSIONS = [
    {
        "label": "PDI1 S2",
        "variant": "psid_behavioral_PDI1_2_nx_25_n2_i50_dbs_both_200Hz_narrow_band",
        "run_ts": "20260408_222003",
    },
    {
        "label": "PDI1 S4",
        "variant": "psid_behavioral_PDI1_4_nx_15_n2_i50_dbs_both_200Hz_narrow_band",
        "run_ts": "20260408_194919",
    },
    {
        "label": "PDI4 S2",
        "variant": "psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band",
        "run_ts": "20260408_162132",
    },
    {
        "label": "PDI4 S3",
        "variant": "psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band",
        "run_ts": "20260408_185522",
    },
]

BEHAV_CHANNELS = ["tracing_velocity_x", "tracing_acceleration_magnitude"]

# %% [markdown]
# ## Data loading

# %%
from dashboard.thesis.loaders import load_split_results_required


def load_test_results(session):
    """Load test-set results for one session. Returns the results dict."""
    return load_split_results_required(
        results_root, session["variant"], session["run_ts"], "test"
    )


def get_channel_names(res):
    """Get neural channel names, sorted by electrode then band."""
    channels = res["input_channels"]
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
def compute_per_trial_rmse_neural(res, channel_order):
    """Compute RMSE for each trial × neural channel from Y vs Yp.

    Returns (n_trials, n_channels) array in the order of channel_order.
    """
    channels = res["input_channels"]
    # Build channel index mapping: channel_order[j] -> position in Y columns
    ch_to_idx = {ch: i for i, ch in enumerate(channels)}
    col_indices = [ch_to_idx[ch] for ch in channel_order]

    n_trials = len(res["Y"])
    n_ch = len(channel_order)
    rmse_matrix = np.full((n_trials, n_ch), np.nan)

    for i in range(n_trials):
        y_true = np.array(res["Y"][i])   # (T, n_channels)
        y_pred = np.array(res["Yp"][i])  # (T, n_channels)
        for j, ci in enumerate(col_indices):
            diff = y_true[:, ci] - y_pred[:, ci]
            rmse_matrix[i, j] = np.sqrt(np.mean(diff ** 2))

    return rmse_matrix


def compute_per_trial_pearson_neural(res, channel_order):
    """Extract per-trial Pearson R for neural channels from precomputed results.

    Returns (n_trials, n_channels) array in the order of channel_order.
    """
    channels = res["input_channels"]
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
        z_true = np.array(res["Z"][i])   # (T, 2)
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


def compute_per_trial_metrics_neural_forecast(res, channel_order):
    """Per-trial × per-channel Pearson R and RMSE for the neural forecast.

    Uses Y_future_true / Y_future_pred (shape (T_forecast, n_neural)). Channels are
    reordered to match channel_order. Returns (r_matrix, rmse_matrix), both
    (n_trials, len(channel_order)), or (None, None) if forecast data absent.
    """
    if "Y_future_true" not in res or res.get("Y_future_true") is None:
        return None, None

    from dashboard.thesis.transforms import reshape_future_z_time_first

    channels = res["input_channels"]
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
    ch_order = get_channel_names(res)  # raw channel names (e.g. ECOG_1_theta_4_8_raw)
    trial_order = sorted_trial_indices(res)
    t_labels = [trial_label(res, i) for i in trial_order]

    r_mat_pred = compute_per_trial_pearson_neural(res, ch_order)
    rmse_mat_pred = compute_per_trial_rmse_neural(res, ch_order)
    r_mat_behav, rmse_mat_behav = compute_per_trial_metrics_behavioral(res)
    r_mat_zfore, rmse_mat_zfore = compute_per_trial_metrics_forecast(res)

    SESSION_DATA[sess["label"]] = dict(
        res=res, ch_order=ch_order, trial_order=trial_order, t_labels=t_labels,
        r_pred=r_mat_pred, rmse_pred=rmse_mat_pred,
        r_behav=r_mat_behav, rmse_behav=rmse_mat_behav,
        r_zfore=r_mat_zfore, rmse_zfore=rmse_mat_zfore,
    )

    r_sorted = r_mat_pred[trial_order, :]

    # Wider figure + fewer pixels per row so long raw channel names fit; heatmap width
    # scales with channel count (60 channels × ~22 px each + margins).
    n_ch = len(ch_order)
    w = max(1700, n_ch * 28 + 320)
    h = max(420, 16 * len(t_labels) + 160)

    fig = go.Figure(go.Heatmap(
        z=r_sorted, x=ch_order, y=t_labels,
        colorscale=COLORSCALE_DIVERGING, zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="Pearson r", font=dict(size=FONT_SIZE_BASE)),
                      tickfont=dict(size=FONT_SIZE_BASE - 1), len=0.9),
    ))
    fig.update_xaxes(title_text="neural channel (raw name)",
                     tickangle=90, tickfont=dict(size=9, family=FONT_FAMILY))
    fig.update_yaxes(title_text="trial (block.trial stim)", autorange="reversed")
    apply_thesis_style(fig, ThesisTheme.LIGHT, height=h,
                       margin=dict(l=140, r=60, t=24, b=180), show_legend=False)
    fig.update_layout(width=w)
    fig.write_image(str(OUT / f'neural_pearson_{sess["label"].replace(" ", "_")}.png'),
                    width=w, height=h, scale=2)
    fig.show()
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

    n_ch = len(sd["ch_order"])
    w = max(1700, n_ch * 28 + 320)
    h = max(420, 16 * len(sd["t_labels"]) + 160)

    fig = go.Figure(go.Heatmap(
        z=rmse_sorted, x=sd["ch_order"], y=sd["t_labels"],
        colorscale=COLORSCALE_SEQUENTIAL, reversescale=True,
        colorbar=dict(title=dict(text="RMSE", font=dict(size=FONT_SIZE_BASE)),
                      tickfont=dict(size=FONT_SIZE_BASE - 1), len=0.9),
    ))
    fig.update_xaxes(title_text="neural channel (raw name)",
                     tickangle=90, tickfont=dict(size=9, family=FONT_FAMILY))
    fig.update_yaxes(title_text="trial (block.trial stim)", autorange="reversed")
    apply_thesis_style(fig, ThesisTheme.LIGHT, height=h,
                       margin=dict(l=140, r=60, t=24, b=180), show_legend=False)
    fig.update_layout(width=w)
    fig.write_image(str(OUT / f'neural_rmse_{sess["label"].replace(" ", "_")}.png'),
                    width=w, height=h, scale=2)
    fig.show()
print(
    "Neural prediction RMSE per trial × channel. Same layout as Pearson r heatmap but "
    "sequential colorscale (darker blue = lower RMSE = better). Complements the r view: "
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
    r_neu_f, rmse_neu_f = compute_per_trial_metrics_neural_forecast(sd["res"], sd["ch_order"])
    if r_neu_f is None:
        print(f"  {sess['label']}: no Y_future data, skipping neural forecast heatmap.")
        continue
    sd["r_yfore"] = r_neu_f
    sd["rmse_yfore"] = rmse_neu_f

    trial_order = sd["trial_order"]
    r_sorted = r_neu_f[trial_order, :]
    rmse_sorted = rmse_neu_f[trial_order, :]

    n_ch = len(sd["ch_order"])
    w = max(1700, n_ch * 28 + 320)
    h = max(420, 16 * len(sd["t_labels"]) + 160)

    fig = go.Figure(go.Heatmap(
        z=r_sorted, x=sd["ch_order"], y=sd["t_labels"],
        colorscale=COLORSCALE_DIVERGING, zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="Pearson r", font=dict(size=FONT_SIZE_BASE)),
                      tickfont=dict(size=FONT_SIZE_BASE - 1), len=0.9),
    ))
    fig.update_xaxes(title_text="neural channel (raw name)",
                     tickangle=90, tickfont=dict(size=9, family=FONT_FAMILY))
    fig.update_yaxes(title_text="trial (block.trial stim)", autorange="reversed")
    apply_thesis_style(fig, ThesisTheme.LIGHT, height=h,
                       margin=dict(l=140, r=60, t=24, b=180), show_legend=False)
    fig.update_layout(width=w)
    fig.write_image(str(OUT / f'neural_forecast_pearson_{sess["label"].replace(" ", "_")}.png'),
                    width=w, height=h, scale=2)
    fig.show()

    fig = go.Figure(go.Heatmap(
        z=rmse_sorted, x=sd["ch_order"], y=sd["t_labels"],
        colorscale=COLORSCALE_SEQUENTIAL, reversescale=True,
        colorbar=dict(title=dict(text="RMSE(z)", font=dict(size=FONT_SIZE_BASE)),
                      tickfont=dict(size=FONT_SIZE_BASE - 1), len=0.9),
    ))
    fig.update_xaxes(title_text="neural channel (raw name)",
                     tickangle=90, tickfont=dict(size=9, family=FONT_FAMILY))
    fig.update_yaxes(title_text="trial (block.trial stim)", autorange="reversed")
    apply_thesis_style(fig, ThesisTheme.LIGHT, height=h,
                       margin=dict(l=140, r=60, t=24, b=180), show_legend=False)
    fig.update_layout(width=w)
    fig.write_image(str(OUT / f'neural_forecast_rmse_{sess["label"].replace(" ", "_")}.png'),
                    width=w, height=h, scale=2)
    fig.show()
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
# Each panel gets its own colorbar tucked against its right edge so trial labels
# never collide with the scale.

# %%
def _behavioral_pair_heatmap(
    sess_label: str, t_labels: list, r_sorted, rmse_sorted, png_name: str,
) -> None:
    h = max(420, 18 * len(t_labels) + 160)
    w = 900  # wider than before so raw feature names (tracing_*) and colorbars fit.
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.26,
    )
    fig.add_trace(go.Heatmap(
        z=r_sorted, x=BEHAV_CHANNELS, y=t_labels,
        colorscale=COLORSCALE_DIVERGING, zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title=dict(text="Pearson r", font=dict(size=FONT_SIZE_BASE)),
                      tickfont=dict(size=FONT_SIZE_BASE - 1),
                      x=0.435, xanchor="left", len=0.82, thickness=12),
    ), row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=rmse_sorted, x=BEHAV_CHANNELS, y=t_labels,
        colorscale=COLORSCALE_SEQUENTIAL, reversescale=True,
        colorbar=dict(title=dict(text="RMSE (raw units)", font=dict(size=FONT_SIZE_BASE)),
                      tickfont=dict(size=FONT_SIZE_BASE - 1),
                      x=1.02, xanchor="left", len=0.82, thickness=12),
    ), row=1, col=2)

    fig.update_xaxes(tickangle=-20, tickfont=dict(size=FONT_SIZE_BASE - 1))
    fig.update_yaxes(autorange="reversed", row=1, col=1,
                     title_text="trial (block.trial stim)")

    # In-panel labels replace the subplot titles (matches sec2 / sec7 convention).
    fig.add_annotation(x=0.5, y=1.03, xref="x domain", yref="y domain",
                       text="<b>Pearson r</b>", showarrow=False,
                       font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY))
    fig.add_annotation(x=0.5, y=1.03, xref="x2 domain", yref="y2 domain",
                       text="<b>RMSE (raw units)</b>", showarrow=False,
                       font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY))

    apply_thesis_style(fig, ThesisTheme.LIGHT, height=h,
                       margin=dict(l=160, r=140, t=56, b=120), show_legend=False)
    fig.update_layout(width=w)
    fig.write_image(str(OUT / png_name), width=w, height=h, scale=2)
    fig.show()


for sess in SESSIONS:
    sd = SESSION_DATA[sess["label"]]
    r_sorted = sd["r_behav"][sd["trial_order"], :]
    rmse_sorted = sd["rmse_behav"][sd["trial_order"], :]
    _behavioral_pair_heatmap(
        sess["label"], sd["t_labels"], r_sorted, rmse_sorted,
        png_name=f'behav_pred_{sess["label"].replace(" ", "_")}.png',
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
        sess["label"], sd["t_labels"], r_sorted, rmse_sorted,
        png_name=f'behav_forecast_{sess["label"].replace(" ", "_")}.png',
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
BAND_KEYS = ["theta_4_8", "alpha_8_12", "beta_12_17", "beta_17_22", "beta_22_27",
             "beta_27_30", "gamma_30_35", "gamma_35_40", "gamma_40_45",
             "gamma_45_50", "gamma_50_55", "gamma_55_60", "gamma_60_65",
             "gamma_70_75", "gamma_75_80"]
ELECTRODE_LABELS = ["ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"]

n_sessions = len(SESSIONS)
fig = make_subplots(rows=1, cols=n_sessions,
                    subplot_titles=[s["label"] for s in SESSIONS],
                    horizontal_spacing=0.06)

for si, sess in enumerate(SESSIONS, 1):
    sd = SESSION_DATA[sess["label"]]
    ch_order = sd["ch_order"]
    mean_r = np.nanmean(sd["r_pred"], axis=0)  # (60,)

    grid = np.full((len(ELECTRODE_LABELS), len(BAND_KEYS)), np.nan)
    for ci, ch in enumerate(ch_order):
        m = re.match(r'ECOG_(\d+)_(.+)_raw', ch)
        if not m:
            continue
        electrode = int(m.group(1)) - 1
        band_key = m.group(2)
        for bi, bk in enumerate(BAND_KEYS):
            if band_key == bk or band_key.startswith(bk):
                grid[electrode, bi] = mean_r[ci]
                break

    fig.add_trace(go.Heatmap(
        z=grid, x=BAND_KEYS, y=ELECTRODE_LABELS,
        colorscale=COLORSCALE_DIVERGING, zmid=0, zmin=-0.5, zmax=0.5,
        colorbar=dict(title=dict(text="r", font=dict(size=FONT_SIZE_BASE)),
                      tickfont=dict(size=FONT_SIZE_BASE - 1), len=0.9)
                 if si == n_sessions else dict(len=0),
        showscale=(si == n_sessions),
    ), row=1, col=si)

fig.update_xaxes(tickangle=-45, tickfont=dict(size=FONT_SIZE_BASE - 2))
apply_thesis_style(fig, ThesisTheme.LIGHT, height=360,
                   margin=dict(l=100, r=100, t=48, b=140), show_legend=False)
fig.update_layout(width=380 * n_sessions)
fig.write_image(str(OUT / 'neural_pearson_summary.png'),
                width=380 * n_sessions, height=360, scale=2)
fig.show()
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
    mean_r_y = np.nanmean(sd["r_yfore"], axis=1) if has_y_f else np.full(n_trials, np.nan)
    mean_r_z = np.nanmean(sd["r_zfore"], axis=1) if has_z_f else np.full(n_trials, np.nan)
    mean_rmse_y = np.nanmean(sd["rmse_yfore"], axis=1) if has_y_f else np.full(n_trials, np.nan)
    mean_rmse_z = np.nanmean(sd["rmse_zfore"], axis=1) if has_z_f else np.full(n_trials, np.nan)

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
n = len(list(OUT.glob('*.png')))
print(f'Section 8 total: {n} figures saved')
