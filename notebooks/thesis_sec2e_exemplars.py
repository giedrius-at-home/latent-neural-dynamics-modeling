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
# # Sec 2e — Single-trial exemplars
#
# Per-trial, per-session demonstrations of true signal vs. PSID/DPAD/VARMA predictions.
#
# * Figs 18-21 — neural reconstruction time series (side-by-side DBS-OFF / DBS-ON, 4 sessions)
# * Figs 29-36 — neural forecast exemplars (best trial per condition x 4 sessions)
# * Figs 50-55 — 4x3 exemplar grids (recon / forecast x RMSE / Pearson / VAF)

# %%
import sys, os
os.chdir('/home/bobby/repos/latent-neural-dynamics-modeling')
sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from thesis_style import (
    COLOR_PSID, COLOR_DPAD, COLOR_VARMA, COLOR_TRUE,
    COLOR_DBS_OFF, COLOR_DBS_ON,
    apply_thesis_style, panel_label, hex_to_rgba,
)
from thesis_sec2_common import *

from thesis_lib.aggregate_rmse import _key_index_map, _trial_key, normalize_stim
from thesis_lib.loaders import (
    load_split_results, load_split_results_required, channels_as_str_list,
    resolve_neural_y_channel_idx, neural_y_feature_label,
)

apply_thesis_style()

# %% [markdown]
# ## Figs 18-21: Neural reconstruction time series
#
# Side-by-side panels (DBS-OFF | DBS-ON) per session: true `ECOG_1_beta_27_30_raw`
# vs. PSID/DPAD/VARMA one-step prediction. Inlined matplotlib builder; data
# loaders remain imports from `thesis_lib`.

# %%
from thesis_lib.compose import _session_mean_rmse_y_triplet
from thesis_lib.exemplar_trials import (
    find_best_trial_indices_per_condition,
    find_best_channel_and_trial,
    resolve_off_on_indices_from_spec,
)
from thesis_lib.loaders import extract_trial_y_series, thesis_exemplar_tagline, ThesisDataError
from thesis_lib.specs import infer_varma_off_on_run_ts
from thesis_lib.transforms import rmse_z, z_true_and_preds

EXEMPLAR_METRIC = "rmse"
AUTO_BEST_CHANNEL = False


def _slice_trial_tail(t_abs, seg_s, z_true, z_psid, z_dpad, z_varma):
    t = np.asarray(t_abs, dtype=float).ravel()
    if t.size == 0:
        e = np.array([], dtype=float)
        return e, e, e, e, e
    t_hi = float(np.nanmax(t))
    t_lo = t_hi - float(seg_s)
    m = t >= t_lo
    arrays = [np.asarray(a, dtype=float).ravel() for a in (z_true, z_psid, z_dpad, z_varma)]
    return (t[m],) + tuple(a[m] for a in arrays)


def _mpl_side_by_side_exemplar(panel_off, panel_on, y_axis_label, *, segment_s=1.0):
    """Matplotlib OFF | ON exemplar builder.

    ``panel_off`` / ``panel_on`` are simple dicts with keys:
    ``t_abs, z_true, z_psid, z_dpad, z_varma,
     band_rmse_psid, band_rmse_dpad, band_rmse_varma``.
    """
    def _prep(p):
        t_raw = np.asarray(p["t_abs"], dtype=float)
        t_trial = (t_raw - t_raw[0]) if t_raw.size > 0 else t_raw
        t_sl, zt, zp, zd, zv = _slice_trial_tail(
            t_trial, segment_s, p["z_true"], p["z_psid"], p["z_dpad"], p["z_varma"],
        )
        trial_offset = float(np.nanmin(t_sl)) if t_sl.size > 0 else 0.0
        t_win = (t_sl - trial_offset) if t_sl.size > 0 else t_sl
        return t_win, zt, zp, zd, zv

    to, zto, zpo, zdo, zvo = _prep(panel_off)
    tn, ztn, zpn, zdn, zvn = _prep(panel_on)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)

    for ax, t, zt, zp, zd, zv, panel, letter, title in (
        (axes[0], to, zto, zpo, zdo, zvo, panel_off, "A", "DBS-OFF"),
        (axes[1], tn, ztn, zpn, zdn, zvn, panel_on, "B", "DBS-ON"),
    ):
        if t.size == 0:
            panel_label(ax, letter, f"{title} (no data)")
            continue

        # Mean-RMSE bands around each model prediction.
        for y_hat, half, col in (
            (zp, panel["band_rmse_psid"], COLOR_PSID),
            (zd, panel["band_rmse_dpad"], COLOR_DPAD),
            (zv, panel["band_rmse_varma"], COLOR_VARMA),
        ):
            if half is None or not np.isfinite(half) or half <= 0:
                continue
            yh = np.asarray(y_hat, dtype=float)
            if len(t) != len(yh) or np.all(np.isnan(yh)):
                continue
            ax.fill_between(t, yh - half, yh + half, color=col, alpha=0.15, linewidth=0)

        ax.plot(t, zt, color=COLOR_TRUE, linewidth=1.4, label="y_true")
        if not np.all(np.isnan(zp)):
            ax.plot(t, zp, color=COLOR_PSID, linewidth=1.2, label="PSID")
        if not np.all(np.isnan(zd)):
            ax.plot(t, zd, color=COLOR_DPAD, linewidth=1.2, linestyle="--", label="DPAD")
        if not np.all(np.isnan(zv)):
            ax.plot(t, zv, color=COLOR_VARMA, linewidth=1.2, linestyle=":", label="VARMA")

        ax.set_xlabel("Time (s)")
        panel_label(ax, letter, title)

    axes[0].set_ylabel(y_axis_label)
    axes[0].legend()
    return fig


def compose_thesis_neural_figure(spec, results_root):
    res_p = load_split_results_required(results_root, spec.psid_variant, spec.psid_run_ts, spec.split)
    res_d = load_split_results(results_root, spec.dpad_variant, spec.dpad_run_ts, spec.split)
    res_v = load_split_results_required(results_root, spec.varma_variant, spec.varma_run_ts, spec.split)

    def _n_y_channels(res):
        if res is None or not res.get("Y"):
            return None
        arr = np.asarray(res["Y"][0])
        return int(arr.shape[1]) if arr.ndim == 2 else 1
    n_caps = [c for c in (_n_y_channels(res_p), _n_y_channels(res_d), _n_y_channels(res_v)) if c]
    n_cap = min(n_caps) if n_caps else None

    ch_idx = spec.neural_y_channel_idx
    if AUTO_BEST_CHANNEL:
        ch_trial = find_best_channel_and_trial(res_p, input_mode="neural", n_channels=n_cap)
        if ch_trial is not None:
            ch_off, i_off, ch_on, i_on = ch_trial
            ch_idx = ch_off
        else:
            i_off, i_on = resolve_off_on_indices_from_spec(
                trial_idx_off=spec.trial_idx_off, trial_idx_on=spec.trial_idx_on,
                use_adjacent_off_on_trials=spec.use_adjacent_off_on_trials, split_res=res_p)
    else:
        best_pair = find_best_trial_indices_per_condition(
            res_p, channel_idx=ch_idx, input_mode="neural")
        if best_pair is not None:
            i_off, i_on = best_pair
        else:
            i_off, i_on = resolve_off_on_indices_from_spec(
                trial_idx_off=spec.trial_idx_off, trial_idx_on=spec.trial_idx_on,
                use_adjacent_off_on_trials=spec.use_adjacent_off_on_trials, split_res=res_p)

    res_v_off = res_v_on = None
    if "dbs_both" in spec.varma_variant:
        v_off = spec.varma_variant.replace("dbs_both", "dbs_off")
        v_on = spec.varma_variant.replace("dbs_both", "dbs_on")
        ts_off, ts_on = spec.varma_run_ts_off, spec.varma_run_ts_on
        if ts_off is None or ts_on is None:
            inf = infer_varma_off_on_run_ts(spec.varma_variant, spec.varma_run_ts)
            if inf is not None:
                ts_off, ts_on = inf
        if ts_off is None or ts_on is None:
            raise ThesisDataError(
                f"ThesisNeuralTimeseriesSpec {spec.section_title!r}: varma_run_ts_off/on required "
                f"for dbs_both VARMA (or match _ALL_TRIPLETS).")
        res_v_off = load_split_results_required(results_root, v_off, ts_off, spec.split)
        res_v_on = load_split_results_required(results_root, v_on, ts_on, spec.split)

    def _varma_res_and_idx(panel, psid_trial_idx):
        if panel == "off" and res_v_off is not None:
            mp = _key_index_map(res_v_off)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_off, mp[k]
        elif panel == "on" and res_v_on is not None:
            mp = _key_index_map(res_v_on)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_on, mp[k]
        if res_v is not None:
            return res_v, psid_trial_idx
        return {"Y": [], "Yp": []}, psid_trial_idx

    res_v_off_use, idx_v_off = _varma_res_and_idx("off", i_off)
    res_v_on_use, idx_v_on = _varma_res_and_idx("on", i_on)
    map_v_off = _key_index_map(res_v_off_use)
    map_v_on = _key_index_map(res_v_on_use)

    band_p_off, band_d_off, band_v_off = _session_mean_rmse_y_triplet(
        res_p, res_d, res_v_off_use, map_v_off, i_off, ch_idx, "off")
    band_p_on, band_d_on, band_v_on = _session_mean_rmse_y_triplet(
        res_p, res_d, res_v_on_use, map_v_on, i_on, ch_idx, "on")

    off = extract_trial_y_series(res_p, res_d, res_v_off_use, i_off, ch_idx,
                                 varma_trial_idx=idx_v_off)
    on = extract_trial_y_series(res_p, res_d, res_v_on_use, i_on, ch_idx,
                                varma_trial_idx=idx_v_on)

    zt_o, zp_o, zd_o, zv_o = z_true_and_preds(off.z_true_raw, off.z_psid, off.z_dpad, off.z_varma)
    zt_n, zp_n, zd_n, zv_n = z_true_and_preds(on.z_true_raw, on.z_psid, on.z_dpad, on.z_varma)

    panel_off = dict(
        t_abs=off.t_abs, z_true=zt_o, z_psid=zp_o, z_dpad=zd_o, z_varma=zv_o,
        band_rmse_psid=band_p_off, band_rmse_dpad=band_d_off, band_rmse_varma=band_v_off,
    )
    panel_on = dict(
        t_abs=on.t_abs, z_true=zt_n, z_psid=zp_n, z_dpad=zd_n, z_varma=zv_n,
        band_rmse_psid=band_p_on, band_rmse_dpad=band_d_on, band_rmse_varma=band_v_on,
    )

    y_meta = neural_y_feature_label(res_p, ch_idx, neural_y_feature_name=spec.neural_y_feature_name)
    caption = thesis_exemplar_tagline(res_p, i_off, i_on, y_meta,
                                      participant_label=spec.participant_label)
    ce = (spec.caption_extra or "").strip()
    if ce:
        caption = f"{caption} · {ce}"

    fig = _mpl_side_by_side_exemplar(
        panel_off, panel_on, y_axis_label=f"z-score — {y_meta}",
        segment_s=spec.exemplar_mid_segment_s,
    )
    return fig, caption


fig_num = 18
for n_spec in THESIS_NEURAL_TIMESERIES:
    fig, cap = compose_thesis_neural_figure(n_spec, results_root)
    fname = f'fig_{fig_num:03d}_neural_ts_{n_spec.section_title.replace(" ","_")}.png'
    fig.savefig(str(OUT / fname))
    plt.show()
    print(
        f"Fig {fig_num}: Neural reconstruction time series for {n_spec.section_title} "
        f"(channel '{n_spec.neural_y_feature_name}'). True signal vs. PSID/DPAD/VARMA one-step "
        f"predictions; DBS-OFF and DBS-ON exemplar trials shown side by side."
    )
    print(f"        {cap}")
    fig_num += 1

# %% [markdown]
# ## Figs 29-36: Neural forecast exemplars (per-session x OFF/ON)
#
# Single-panel forecast exemplars — history (light blue) + forecast window (light red)
# with true + PSID + VARMA traces. Best trial per (session, condition) selected by
# minimising `max(rmse_psid, rmse_varma)` over the entire forecast window.

# %%
from thesis_lib.forecast_horizon_rmse import _per_step_abs_err_z_future

_FORECAST_CTX_ALPHA = 0.08
_FORECAST_FUT_ALPHA = 0.07


def _trial_forecast_rmse(res, k_true, k_pred, trial_idx, channel_idx):
    if res is None:
        return float("nan")
    zt = res.get(k_true); zp = res.get(k_pred)
    if zt is None or zp is None or trial_idx >= len(zt) or trial_idx >= len(zp):
        return float("nan")
    err = _per_step_abs_err_z_future(zt[trial_idx], zp[trial_idx], channel_idx)
    if err is None or err.size == 0:
        return float("nan")
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite ** 2)))


def _select_best_forecast_trial(res_p, res_v, res_v_split, ch_use, condition_stim):
    if res_p is None:
        return None, None, float("nan"), float("nan")
    stim_seq = res_p.get("stim") or []
    n = len(res_p.get("Y_future_true") or [])
    map_v = _key_index_map(res_v_split if res_v_split is not None else res_v)
    best = (None, None, float("inf"), float("nan"), float("nan"))
    for i_p in range(n):
        if i_p >= len(stim_seq):
            break
        if normalize_stim(stim_seq[i_p]) != condition_stim:
            continue
        k = _trial_key(res_p, i_p)
        jv = map_v.get(k)
        if jv is None:
            continue
        r_p = _trial_forecast_rmse(res_p, "Y_future_true", "Y_future_pred", i_p, ch_use)
        r_v = _trial_forecast_rmse(
            res_v_split if res_v_split is not None else res_v,
            "Y_future_true", "Y_future_pred", jv, ch_use,
        )
        if not (np.isfinite(r_p) and np.isfinite(r_v)):
            continue
        score = max(r_p, r_v)
        if score < best[2]:
            best = (i_p, jv, score, r_p, r_v)
    if best[0] is None:
        return None, None, float("nan"), float("nan")
    return best[0], best[1], best[3], best[4]


def _resolve_varma_off_on(spec):
    res_p = load_split_results_required(results_root, spec.psid_variant, spec.psid_run_ts, "test")
    res_v = load_split_results_required(results_root, spec.varma_variant, spec.varma_run_ts, "test")
    res_v_off = res_v_on = None
    if "dbs_both" in spec.varma_variant:
        v_off_var = spec.varma_variant.replace("dbs_both", "dbs_off")
        v_on_var = spec.varma_variant.replace("dbs_both", "dbs_on")
        if spec.varma_run_ts_off:
            res_v_off = load_split_results_required(results_root, v_off_var, spec.varma_run_ts_off, "test")
        if spec.varma_run_ts_on:
            res_v_on = load_split_results_required(results_root, v_on_var, spec.varma_run_ts_on, "test")
    ch_use = resolve_neural_y_channel_idx(res_p, spec.neural_y_feature_name, spec.channel_idx)
    return res_p, res_v, res_v_off, res_v_on, ch_use


# Inline matplotlib forecast-panel builder replacing thesis_lib.c2_forecast_timeseries._build_one_panel's
# plotting half. We still rely on the data-preparation helpers; to keep alignment exact we re-use
# `_build_one_panel` (which is pure data-shaping despite its name) and simply render with matplotlib.
from thesis_lib.c2_forecast_timeseries import _build_one_panel as _forecast_rowdata


def _mpl_forecast_panel(rowdata, neu_lbl, condition_label):
    """Render one forecast exemplar panel (history + forecast)."""
    (t_full, z_true, z_psid, _z_dpad, z_varma, _u, _l, _rp, _rd, _rv, n_hist) = rowdata
    t_full = np.asarray(t_full, dtype=float).ravel()
    n_hist = int(n_hist)

    def _gap(a):
        a = np.asarray(a, dtype=float).ravel()
        if 0 < n_hist < len(a):
            return np.concatenate([a[:n_hist], [np.nan], a[n_hist:]])
        return a

    t_plot = _gap(t_full)

    fig, ax = plt.subplots(figsize=(8.5, 3.2))

    if t_full.size and n_hist > 0:
        ax.axvspan(float(t_full[0]), float(t_full[n_hist - 1]),
                   color=COLOR_DBS_OFF, alpha=_FORECAST_CTX_ALPHA, linewidth=0)
    if t_full.size and n_hist < len(t_full):
        ax.axvspan(float(t_full[n_hist]), float(t_full[-1]),
                   color=COLOR_DBS_ON, alpha=_FORECAST_FUT_ALPHA, linewidth=0)
        ax.axvline(float(t_full[n_hist]), color="#444441", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.plot(t_plot, _gap(z_true), color=COLOR_TRUE, linewidth=1.4, label="y_true")
    if not np.all(np.isnan(z_psid)):
        ax.plot(t_plot, _gap(z_psid), color=COLOR_PSID, linewidth=1.2, label="PSID")
    if not np.all(np.isnan(z_varma)):
        ax.plot(t_plot, _gap(z_varma), color=COLOR_VARMA, linewidth=1.2,
                linestyle=(0, (4, 1)), label="VARMA")

    ax.set_xlabel("trial time (s)")
    ax.set_ylabel(f"z-score — {neu_lbl}")
    panel_label(ax, "A", f"DBS-{condition_label}")
    ax.legend()
    return fig


fig_num = 29
for c2_spec in THESIS_C2_FORECASTS:
    res_p, res_v, res_v_off, res_v_on, ch_use = _resolve_varma_off_on(c2_spec)
    inn = channels_as_str_list(res_p.get("input_channels"))
    neu_lbl = inn[ch_use] if ch_use < len(inn) else c2_spec.neural_y_feature_name

    for cond_lbl, cond_stim, rv_split in (("OFF", "off", res_v_off), ("ON", "on", res_v_on)):
        i_best, jv_best, rmse_p, rmse_v = _select_best_forecast_trial(
            res_p, res_v, rv_split, ch_use, cond_stim,
        )
        if i_best is None:
            print(f"Fig {fig_num}: SKIP - no aligned forecast trial for "
                  f"{c2_spec.section_title} {cond_lbl}.")
            fig_num += 1
            continue
        rv_use = rv_split if rv_split is not None else res_v
        rowdata = _forecast_rowdata(
            res_p, None, rv_use, i_best, ch_use, c2_spec,
            sigma_z=None, varma_trial_idx=jv_best,
        )
        if rowdata is None:
            print(f"Fig {fig_num}: SKIP - forecast row data unavailable for "
                  f"{c2_spec.section_title} {cond_lbl}.")
            fig_num += 1
            continue
        fig = _mpl_forecast_panel(rowdata, neu_lbl, cond_lbl)
        fig.savefig(str(OUT / f'fig_{fig_num:03d}_neural_forecast_{c2_spec.section_title}_{cond_lbl}.png'))
        plt.show()
        keys_p = (
            res_p.get("participant_id"), res_p.get("session"),
            res_p.get("block"), res_p.get("trial"),
        )
        def _g(seq, idx, default="?"):
            try:
                return seq[idx]
            except Exception:
                return default
        pid_v  = _g(keys_p[0], i_best) if keys_p[0] is not None else "?"
        sess_v = _g(keys_p[1], i_best) if keys_p[1] is not None else "?"
        blk_v  = _g(keys_p[2], i_best) if keys_p[2] is not None else "?"
        tri_v  = _g(keys_p[3], i_best) if keys_p[3] is not None else "?"
        print(
            f"Fig {fig_num}: Neural forecast exemplar - {c2_spec.section_title} DBS-{cond_lbl} "
            f"(channel '{neu_lbl}'). Best trial selected by minimising max(rmse_psid, rmse_varma) "
            f"over the forecast window. Trial = participant {pid_v}, session {sess_v}, "
            f"block {blk_v}, trial {tri_v} (PSID row {i_best}). "
            f"Forecast RMSE: PSID={rmse_p:.3f}, VARMA={rmse_v:.3f}."
        )
        fig_num += 1

# %% [markdown]
# ## Figs 50-55: 12-panel exemplar figures (4 participants x 3 feature types)
#
# Two target modes (reconstruction and forecast) x three metrics (RMSE,
# Pearson r, VAF) = **6 figures**. Each figure shows a 4x3 grid (rows =
# participants, cols = feature type: behavioural kinematic / neural ECoG /
# laplacian LFP). For every panel, PSID picks the best feature of its type on
# the test split under that figure's metric; VARMA and DPAD are overlaid on
# the same PSID-selected feature so the comparison is apples-to-apples. LFP
# panels are 2-model (no DPAD laplacian).

# %%
import yaml as _yaml
from functools import lru_cache as _lru_cache
from thesis_utils import metric_display_name, score_channel

# ---------------------------------------------------------------------------
# Feature universes per type. Shared between PSID / DPAD / VARMA: the 5 top
# channels VARMA uses are the universe (VARMA bottlenecks on top-5, PSID and
# DPAD see all 60 but we compare them on the shared set so the pick is
# meaningful across all three).
# ---------------------------------------------------------------------------
BEHAVIORAL_FEATS = list(THESIS_DECLARED_BEHAVIORAL_OUTPUTS)  # 2 kinematic channels


@_lru_cache(maxsize=16)
def _varma_top5_channels(varma_variant: str) -> tuple[str, ...]:
    """Read the top-5 channel list VARMA was trained on (from its YAML)."""
    matches = list(Path('training/setups/varma').glob(f'**/{varma_variant}.yaml'))
    if not matches:
        return tuple()
    with open(matches[0]) as f:
        y = _yaml.safe_load(f)
    try:
        return tuple(str(c) for c in y['data']['channels']['neural_input'])
    except Exception:
        return tuple()


def _neural_feats_for(tri) -> list[str]:
    return list(_varma_top5_channels(tri.varma_variant))


LFP_FEATS = list(LAPLACIAN_BAND_NAMES)


@_lru_cache(maxsize=64)
def _psid_channel_list(psid_variant: str) -> tuple[str, ...]:
    matches = list(Path('training/setups/psid').glob(f'**/{psid_variant}.yaml'))
    if not matches:
        return tuple()
    with open(matches[0]) as f:
        y = _yaml.safe_load(f)
    try:
        return tuple(str(c) for c in y['data']['channels']['neural_input'])
    except Exception:
        return tuple()


@_lru_cache(maxsize=128)
def _yaml_channels(model: str, variant: str, key: str) -> tuple[str, ...]:
    """Read `data.channels[key]` from training setup YAML for given model/variant.

    `model` is one of 'psid', 'dpad', 'varma'. `key` typically 'neural_input'
    (Y inputs) or 'output' (Z outputs). Returns empty tuple if YAML missing
    or key not present — callers fall back to parquet `output_channels` /
    column count.
    """
    matches = list(Path(f'training/setups/{model}').glob(f'**/{variant}.yaml'))
    if not matches:
        return tuple()
    with open(matches[0]) as f:
        y = _yaml.safe_load(f)
    try:
        return tuple(str(c) for c in y['data']['channels'][key])
    except Exception:
        return tuple()


def _feat_index_psid(tri, feat_name: str) -> int:
    names = _psid_channel_list(tri.psid_variant)
    for i, n in enumerate(names):
        if n == feat_name:
            return i
    return -1


def _score_all_trials(res, ch_idx, target_true, target_pred, metric):
    vals = []
    for i in range(len(res.get(target_true, []))):
        tt = res[target_true][i]; pp = res[target_pred][i]
        if tt is None or pp is None:
            continue
        arr_t = np.asarray(tt, dtype=float); arr_p = np.asarray(pp, dtype=float)
        if arr_t.ndim != 2 or arr_p.ndim != 2 or arr_t.shape != arr_p.shape:
            continue
        if ch_idx >= arr_t.shape[1]:
            continue
        try:
            v = score_channel(arr_t[:, ch_idx], arr_p[:, ch_idx], metric)
            if np.isfinite(v):
                vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return float('nan')
    return float(np.mean(vals))


def _pick_best_feat(tri, feat_type: str, metric: str, target: str) -> tuple[str, int, float]:
    """Return (feature_name, column_index_in_Z_or_Y, psid_score)."""
    tri_use = tri
    if feat_type == 'lfp':
        lap_map = {t.label: t for t in ALL_TRIPLETS_LAP}
        tri_use = lap_map.get(tri.label, tri)
    res = load_split_results_required(
        results_root, tri_use.psid_variant, tri_use.psid_run_ts, 'test'
    )
    if feat_type == 'behavioral':
        candidate_feats = BEHAVIORAL_FEATS
        candidate_idxs = list(range(len(BEHAVIORAL_FEATS)))
        key_t, key_p = ('Z', 'Zp') if target == 'recon' else ('Z_future_true', 'Z_future_pred')
    elif feat_type == 'neural':
        candidate_feats = _neural_feats_for(tri_use)
        candidate_idxs = [_feat_index_psid(tri_use, f) for f in candidate_feats]
        key_t, key_p = ('Y', 'Yp') if target == 'recon' else ('Y_future_true', 'Y_future_pred')
    elif feat_type == 'lfp':
        # PSID lap top-K LFP names live in the training YAML `data.channels.output`
        # (mrmr-picked subset, NOT the global 15-band order). Each pipeline
        # picks its own top-K independently — DPAD/VARMA may pick different bands.
        names = _yaml_channels('psid', tri_use.psid_variant, 'output')
        if not names:
            # YAML missing — fall back to parquet output_channels, then to anonymous Z_i.
            names = tuple(str(c) for c in (res.get('output_channels') or []))
        if not names:
            Z0 = res['Z'][0] if res.get('Z') else None
            n_bands = np.asarray(Z0).shape[1] if (Z0 is not None and np.asarray(Z0).ndim == 2) else 0
            names = tuple(f'Z_{i}' for i in range(n_bands))
        candidate_feats = list(names)
        candidate_idxs = list(range(len(names)))
        key_t, key_p = ('Z', 'Zp') if target == 'recon' else ('Z_future_true', 'Z_future_pred')
    else:
        raise ValueError(feat_type)
    best_name, best_idx, best_val = candidate_feats[0], candidate_idxs[0], float('nan')
    for name, ci in zip(candidate_feats, candidate_idxs):
        if ci < 0:
            continue
        v = _score_all_trials(res, ci, key_t, key_p, metric)
        if not np.isfinite(v):
            continue
        if not np.isfinite(best_val):
            best_name, best_idx, best_val = name, ci, v
            continue
        better = (v < best_val) if metric == 'rmse' else (v > best_val)
        if better:
            best_name, best_idx, best_val = name, ci, v
    return best_name, best_idx, best_val


def _pick_best_trial(res, ch_idx, key_t, key_p, metric):
    best_i, best_v = 0, float('nan')
    for i in range(len(res.get(key_t, []))):
        tt = res[key_t][i]; pp = res[key_p][i]
        if tt is None or pp is None:
            continue
        arr_t = np.asarray(tt, dtype=float); arr_p = np.asarray(pp, dtype=float)
        if arr_t.ndim != 2 or arr_p.ndim != 2 or arr_t.shape != arr_p.shape:
            continue
        if ch_idx >= arr_t.shape[1]:
            continue
        try:
            v = score_channel(arr_t[:, ch_idx], arr_p[:, ch_idx], metric)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        if not np.isfinite(best_v):
            best_i, best_v = i, v
            continue
        better = (v < best_v) if metric == 'rmse' else (v > best_v)
        if better:
            best_i, best_v = i, v
    return best_i, best_v


def _extract_trace_pair(res, ch_idx, trial_idx, key_t, key_p):
    tt = res[key_t][trial_idx]; pp = res[key_p][trial_idx]
    arr_t = np.asarray(tt, dtype=float); arr_p = np.asarray(pp, dtype=float)
    t_vec = np.arange(arr_t.shape[0]) / 200.0  # 200 Hz
    return t_vec, arr_t[:, ch_idx], arr_p[:, ch_idx]


def _collect_residuals_for_channel(res, ch_idx, key_t, key_p):
    rows = []
    for i in range(len(res.get(key_t, []) or [])):
        tt = res[key_t][i]
        pp = res[key_p][i]
        if tt is None or pp is None:
            continue
        at = np.asarray(tt, dtype=float)
        ap = np.asarray(pp, dtype=float)
        if at.shape != ap.shape or at.ndim != 2:
            continue
        if ch_idx >= at.shape[1]:
            continue
        r = ap[:, ch_idx] - at[:, ch_idx]
        if np.any(np.isfinite(r)):
            rows.append(r)
    return rows


def _ci_band_offsets(residual_rows, alpha=0.05):
    if not residual_rows:
        return None, None
    m_min = min(r.size for r in residual_rows)
    arr = np.stack([r[:m_min] for r in residual_rows], axis=0)
    if arr.shape[0] < 2:
        return None, None
    q_low = np.nanquantile(arr, alpha / 2, axis=0)
    q_high = np.nanquantile(arr, 1 - alpha / 2, axis=0)
    return q_low, q_high


def _mpl_add_model_with_ci(ax, t_vec, y_pred, color, label, linestyle, res_model,
                           ch_idx, key_t, key_p, alpha=0.05):
    if res_model is not None:
        q_low, q_high = _ci_band_offsets(
            _collect_residuals_for_channel(res_model, ch_idx, key_t, key_p),
            alpha=alpha,
        )
        if q_low is not None:
            m = min(len(y_pred), q_low.size)
            lo = y_pred[:m] + q_low[:m]
            hi = y_pred[:m] + q_high[:m]
            ax.fill_between(t_vec[:m], lo, hi,
                            color=color, alpha=0.15, linewidth=0)
    ax.plot(t_vec, y_pred, color=color, linewidth=1.1, linestyle=linestyle, label=label)


def build_exemplar_grid(target: str, metric: str, fig_num: int):
    feat_types = ['behavioral', 'neural', 'lfp']
    col_titles = {'behavioral': 'Behavioural kinematics',
                  'neural': 'Neural (ECoG top-5)',
                  'lfp': 'Laplacian LFP'}
    key_map = {('recon', 'behavioral'): ('Z', 'Zp'),
               ('recon', 'neural'):     ('Y', 'Yp'),
               ('recon', 'lfp'):        ('Z', 'Zp'),
               ('forecast', 'behavioral'): ('Z_future_true', 'Z_future_pred'),
               ('forecast', 'neural'):     ('Y_future_true', 'Y_future_pred'),
               ('forecast', 'lfp'):        ('Z_future_true', 'Z_future_pred')}

    lap_by_label = {t.label: t for t in ALL_TRIPLETS_LAP}
    fig, axes = plt.subplots(4, 3, figsize=(10.0, 9.5))
    feature_picks = []
    legend_added = False

    for ri, tri in enumerate(ALL_TRIPLETS):
        for ci, feat_type in enumerate(feat_types):
            ax = axes[ri, ci]
            tri_use = lap_by_label[tri.label] if feat_type == 'lfp' else tri
            key_t, key_p = key_map[(target, feat_type)]
            panel_letter = chr(ord('A') + ri * 3 + ci)
            try:
                feat_name, ch_idx, psid_score = _pick_best_feat(tri, feat_type, metric, target)
                if ch_idx < 0:
                    raise ValueError(f'no matching PSID channel for {feat_name}')
                res_p = load_split_results_required(
                    results_root, tri_use.psid_variant, tri_use.psid_run_ts, 'test')
                trial_i, _ = _pick_best_trial(res_p, ch_idx, key_t, key_p, metric)
                t_vec, y_true, y_psid = _extract_trace_pair(res_p, ch_idx, trial_i, key_t, key_p)
                if len(y_true) == 0 or not np.any(np.isfinite(y_true)):
                    raise ValueError('empty trace')

                res_v = load_split_results(results_root, tri_use.varma_variant, tri_use.varma_run_ts, 'test')
                res_d = load_split_results(results_root, tri_use.dpad_variant, tri_use.dpad_run_ts, 'test') \
                    if tri_use.dpad_run_ts else None

                def _find_col(res_o, feat_name, fallback_idx, target_kind='input'):
                    """Resolve column index for `feat_name` in `res_o`.

                    `target_kind`='input' searches `input_channels` (Y / neural);
                    'output' searches `output_channels` (Z / behavioral or LFP).
                    Returns -1 if name not found AND fallback_idx is out of range.
                    """
                    if res_o is None:
                        return -1
                    chans_field = 'output_channels' if target_kind == 'output' else 'input_channels'
                    names = [str(c) for c in (res_o.get(chans_field) or [])]
                    for i, n in enumerate(names):
                        if n == feat_name:
                            return i
                    try:
                        first = res_o.get(key_t, [None])[0]
                        if first is None:
                            return -1
                        arr_first = np.asarray(first)
                        ncol = arr_first.shape[1] if arr_first.ndim == 2 else 0
                        return fallback_idx if fallback_idx < ncol else -1
                    except Exception:
                        return -1

                ax.plot(t_vec, y_true, color=COLOR_TRUE, linewidth=1.4,
                        label=("true" if not legend_added else None))
                _mpl_add_model_with_ci(
                    ax, t_vec, y_psid, COLOR_PSID,
                    label=("PSID" if not legend_added else None),
                    linestyle="-", res_model=res_p, ch_idx=ch_idx,
                    key_t=key_t, key_p=key_p,
                )

                if res_v is not None:
                    if feat_type == 'neural':
                        ch_v = -1
                        varma_ch_names = _varma_top5_channels(tri_use.varma_variant)
                        for i, n in enumerate(varma_ch_names):
                            if n == feat_name:
                                ch_v = i; break
                    elif feat_type == 'lfp':
                        # VARMA's top-K LFP picks may differ from PSID's; look up by name.
                        ch_v = _find_col(res_v, feat_name, ch_idx, target_kind='output')
                    else:  # behavioral — 2 fixed kinematics, same order across pipelines
                        ch_v = ch_idx
                    key = _trial_key(res_p, trial_i)
                    mv = _key_index_map(res_v) if res_v else {}
                    i_v = mv.get(key, trial_i)
                    if ch_v >= 0 and key_t in res_v and i_v < len(res_v[key_t]) \
                            and res_v[key_t][i_v] is not None:
                        try:
                            tv, _, y_v = _extract_trace_pair(res_v, ch_v, i_v, key_t, key_p)
                            _mpl_add_model_with_ci(
                                ax, tv, y_v, COLOR_VARMA,
                                label=("VARMA" if not legend_added else None),
                                linestyle=":", res_model=res_v, ch_idx=ch_v,
                                key_t=key_t, key_p=key_p,
                            )
                        except Exception as e:
                            print(f"  VARMA overlay failed {tri.label}/{feat_type}: {e.__class__.__name__}: {e}")

                if res_d is not None:
                    if feat_type == 'behavioral':
                        ch_d = ch_idx  # 2 fixed kinematics, same order across pipelines
                    elif feat_type == 'lfp':
                        # DPAD's top-K LFP picks differ from PSID's (independent mRMR);
                        # look up feat_name in DPAD output_channels. -1 if DPAD didn't pick it.
                        ch_d = _find_col(res_d, feat_name, ch_idx, target_kind='output')
                    else:  # neural — both pipelines share mrmr8 ECoG selection
                        ch_d = _find_col(res_d, feat_name, ch_idx, target_kind='input')
                    md = _key_index_map(res_d)
                    i_d = md.get(_trial_key(res_p, trial_i), -1)
                    if ch_d >= 0 and i_d >= 0 and key_t in res_d \
                            and i_d < len(res_d[key_t]) and res_d[key_t][i_d] is not None:
                        try:
                            td, _, y_d = _extract_trace_pair(res_d, ch_d, i_d, key_t, key_p)
                            _mpl_add_model_with_ci(
                                ax, td, y_d, COLOR_DPAD,
                                label=("DPAD" if not legend_added else None),
                                linestyle="--", res_model=res_d, ch_idx=ch_d,
                                key_t=key_t, key_p=key_p,
                            )
                        except Exception as e:
                            print(f"  DPAD overlay failed {tri.label}/{feat_type}: {e.__class__.__name__}: {e}")

                feat_short = feat_name.replace('_raw', '').replace('ECOG_', 'ECoG-')
                panel_label(
                    ax, panel_letter,
                    f'{tri.label} - {feat_short} ({metric_display_name(metric)}={psid_score:.2f})',
                )
                feature_picks.append((tri.label, feat_type, feat_name, float(psid_score)))
                legend_added = True
            except Exception as e:
                panel_label(ax, panel_letter,
                            f'{tri.label} - (no data: {e.__class__.__name__})')

            if ri == 3:
                ax.set_xlabel('time (s)')
            if ci == 0:
                ax.set_ylabel('raw signal')
            if ri == 0:
                # Inject column header above row 1 via the panel subtitle.
                existing = ax.get_title(loc="left")
                ax.set_title(col_titles[feat_type], loc="center")
                ax.set_title(existing, loc="left")

    fig.legend()
    return fig, feature_picks


_EXEMPLAR_PLAN = [
    (50, 'recon',    'rmse'),
    (51, 'recon',    'pearson'),
    (52, 'recon',    'vaf'),
    (53, 'forecast', 'rmse'),
    (54, 'forecast', 'pearson'),
    (55, 'forecast', 'vaf'),
]

for fig_num, target, metric in _EXEMPLAR_PLAN:
    fig, picks = build_exemplar_grid(target, metric, fig_num)
    stem = f'exemplars_{target}_{metric}'
    fig.savefig(str(OUT / f'fig_{fig_num:03d}_{stem}.png'))
    plt.show()
    print(f'Fig {fig_num}: {target} exemplars by {metric_display_name(metric)} — picks:')
    for lbl, ft, feat, score in picks:
        print(f'  {lbl:>8}  {ft:<11}  {feat:<32}  PSID {metric_display_name(metric)}={score:.3f}')

# %% [markdown]
# So for each participant the PSID best feature per metric is the canonical
# pick for each exemplar panel. VARMA and DPAD are overlaid on the same
# PSID-selected feature so model comparison is apples-to-apples.
#
# LFP panels are 2-model (no DPAD laplacian). For behavioural and neural
# (ECoG top-5), all three models are plotted when DPAD parquets are available.
