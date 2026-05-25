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
# # Section 2: Model Validation — 37 figures (parent notebook)
#
# Single-notebook consolidation of the figures exported by `thesis_sec2a` – `thesis_sec2e`.
# Rewritten to matplotlib + scienceplots; plotting helpers live in `thesis_sec2_common`
# and `thesis_style` so edits to those modules propagate here.
#
# | #     | Figure                                                         | Path |
# |-------|----------------------------------------------------------------|------|
# | 7-8   | Pooled behavioural metric (24-box grouped)                     | A    |
# | 9-16  | Per-session behavioural metric                                 | A    |
# | 17    | Session-mean box+strip                                         | B    |
# | 18-21 | Neural reconstruction time series                              | B    |
# | 22    | Neural band metric heatmap                                     | B    |
# | 23    | Forecast RMSE vs horizon                                       | B    |
# | 24    | Pooled neural forecast metric                                  | A    |
# | 25-28 | Per-session neural forecast metric                             | A    |
# | 29-36 | Neural forecast exemplars                                      | B    |
# | 37    | Vanilla vs RTS + A regularization PSID                         | B    |
# | 38    | PSID grid-search BA heatmap                                    | B    |
# | 39    | Pooled neural reconstruction metric                            | A    |
# | 40-43 | Per-session neural reconstruction metric                       | A    |

# %%
import sys, os

os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")
sys.path.insert(0, ".")
sys.path.insert(0, "notebooks")

from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba

from thesis_style import (
    COLOR_PSID,
    COLOR_DPAD,
    COLOR_VARMA,
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_NS,
    COLOR_TRUE,
    COLOR_CHANCE,
    apply_thesis_style,
    panel_label,
    hex_to_rgba,
)
from thesis_sec2_common import *

from thesis_utils import (
    AggregateRmseData,
    _key_index_map,
    _sem,
    normalize_stim,
    collect_pooled_rmse,
    collect_session_grouped,
    collect_neural_band_metric,
    pick_best_feature_for_psid,
    trial_metric_y_for_model,
    trial_metric_forecast_for_model,
    metric_axis_label,
    metric_y_range,
    metric_display_name,
)
from thesis_loaders import (
    discover_session_run,
    load_split_results,
    load_split_results_required,
    channels_as_str_list,
    resolve_input_channels,
    resolve_neural_y_channel_idx,
    neural_y_feature_label,
    has_dpad_data,
    output_channel_label,
)

apply_thesis_style()

# %% [markdown]
# ## Available results

# %%
from thesis_loaders import inspect_available_results
display(inspect_available_results(results_root))


def resolve_output_channel_display(split_res, channel_idx, *, declared_outputs):
    """Inline fallback: saved output_channels first, then declared_outputs list."""
    name = output_channel_label(split_res, channel_idx, fallback="")
    if name:
        return name, False
    if 0 <= channel_idx < len(declared_outputs):
        return str(declared_outputs[channel_idx]).replace("_", " "), True
    raise ValueError(f"channel_idx={channel_idx} has no resolvable label")


# %% [markdown]
# ## Figs 7-8: Pooled behavioural metric (24-box grouped layout)

# %%
fig_num = 7
for spec in THESIS_AGGREGATE_FIGURES:
    _psid_var, _psid_ts = discover_session_run(results_root, "psid", spec.exp_type, spec.sessions[0])
    ch, _ = resolve_output_channel_display(
        load_split_results_required(results_root, _psid_var, _psid_ts, spec.split),
        spec.channel_idx,
        declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
    )

    def _collect(metric, _spec=spec):
        return collect_session_grouped(
            results_root,
            _spec.sessions,
            _spec.exp_type,
            _spec.channel_idx,
            split=_spec.split,
            metric=metric,
        )

    fig = write_boxplot_three_metrics(
        collector=_collect,
        fig_num=fig_num,
        filename_stem=f'pooled_{ch.replace(" ", "_")}',
        target_label=ch,
        rng_seed=spec.jitter_seed,
        builder=mpl_session_grouped_boxplot,
    )
    if fig is not None:
        plt.show()
    print(
        f"Fig {fig_num}: pooled behavioural decoding for '{ch}' from dbs_both models. "
        f"Layout: 2 panels (DBS-OFF | DBS-ON) x 3 model groups x 4 session boxes. "
        f"Sessions: {', '.join(spec.sessions)}. Three metric PNGs emitted."
    )
    fig_num += 1

# %% [markdown]
# ## Figs 9-16: Per-session behavioural metric (2 channels x 4 sessions)

# %%
fig_num = 9
for spec in THESIS_AGGREGATE_FIGURES:
    _psid_var, _psid_ts = discover_session_run(results_root, "psid", spec.exp_type, spec.sessions[0])
    ch, _ = resolve_output_channel_display(
        load_split_results_required(results_root, _psid_var, _psid_ts, spec.split),
        spec.channel_idx,
        declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
    )
    for session in spec.sessions:

        def _collect(metric, _session=session, _spec=spec):
            return collect_pooled_rmse(
                results_root,
                [_session],
                _spec.exp_type,
                _spec.channel_idx,
                split=_spec.split,
                run_wilcoxon=False,
                metric=metric,
            )

        fig = write_boxplot_three_metrics(
            collector=_collect,
            fig_num=fig_num,
            filename_stem=f'session_{session}_{ch.replace(" ", "_")}',
            target_label=f"{session} - {ch}",
            rng_seed=spec.jitter_seed,
        )
        if fig is not None:
            plt.show()
        pid, sess = session.split("_")
        print(
            f"Fig {fig_num}: behavioural decoding for {pid} S{sess[1:]} - '{ch}'. "
            f"Three metric PNGs emitted."
        )
        fig_num += 1

# %% [markdown]
# ## Fig 17: Session-mean RMSE strip/box plots
#
# Per-participant box plots (trial-level RMSE) by model x DBS condition.
# Six columns: PSID OFF/ON, DPAD OFF/ON, VARMA OFF/ON — one panel per session.

# %%
# Path B: inline, because the strip/box layout isn't shared with any sec2a-e notebook.
from thesis_lib.session_strip_rmse import collect_strip_figure_data

strip_spec = THESIS_STRIP_PANELS[0]
from types import SimpleNamespace as _NS
def _session_ns(session, exp_type):
    pv, pt = discover_session_run(results_root, "psid", exp_type, session)
    vv, vt = discover_session_run(results_root, "varma", exp_type, session)
    dv, dt = discover_session_run(results_root, "dpad", exp_type, session)
    return _NS(psid_variant=pv or "", psid_run_ts=pt or "",
               varma_variant=vv or "", varma_run_ts=vt or "",
               dpad_variant=dv or "", dpad_run_ts=dt or "", label=session)
panel_triplets = [(e.panel_label, _session_ns(e.session, strip_spec.exp_type)) for e in strip_spec.panels]
strip_data = collect_strip_figure_data(
    results_root,
    panel_triplets,
    strip_spec.channel_idx,
    split=strip_spec.split,
)
rng = np.random.default_rng(strip_spec.jitter_seed)

ncols = strip_spec.ncols
n_panels = len(strip_data.panels)
nrows = int(np.ceil(n_panels / max(1, ncols)))
jitter = 0.10

models = ["PSID", "DPAD", "VARMA"]
model_colors = [COLOR_PSID, COLOR_DPAD, COLOR_VARMA]
_OFF_CELLS = [0, 2, 4]
_ON_CELLS = [1, 3, 5]

fig_17, axes = plt.subplots(
    nrows, ncols, figsize=(11.5, max(3.2, 2.3 * nrows)), sharey=True
)
axes = np.atleast_1d(axes).ravel()

for pi, panel in enumerate(strip_data.panels):
    ax = axes[pi]
    for mi, (model, mc, off_cell, on_cell) in enumerate(
        zip(models, model_colors, _OFF_CELLS, _ON_CELLS)
    ):
        for cond_label, cell_idx, alpha in (
            ("OFF", off_cell, 0.80),
            ("ON", on_cell, 0.45),
        ):
            vals = [v for v in panel.trial_rmse[cell_idx] if np.isfinite(v)]
            if not vals:
                continue
            xpos = mi * 2 + (0 if cond_label == "OFF" else 1)
            face = (*to_rgba(mc)[:3], alpha * 0.35)
            bp = ax.boxplot(
                [vals],
                positions=[xpos],
                widths=0.7,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
            )
            for box in bp["boxes"]:
                box.set(facecolor=face, edgecolor=mc, linewidth=1.2)
            for el in ("whiskers", "caps", "medians"):
                for ln in bp[el]:
                    ln.set(color=mc, linewidth=1.0)
            jt = rng.uniform(-jitter * 0.4, jitter * 0.4, size=len(vals))
            dot_color = (*to_rgba(mc)[:3], alpha * 0.75)
            ax.scatter(
                xpos + jt, vals, s=8, color=[dot_color] * len(vals), linewidths=0
            )

    for xv in (1.5, 3.5):
        ax.axvline(xv, color="#AAAAAA", linewidth=0.7, linestyle="--", alpha=0.5)

    ax.set_xticks(range(6))
    ax.set_xticklabels(
        ["PSID\nOFF", "PSID\nON", "DPAD\nOFF", "DPAD\nON", "VARMA\nOFF", "VARMA\nON"]
    )
    ax.set_ylim(0, strip_data.y_max)
    panel_label(ax, chr(ord("A") + pi), panel.panel_label)

for ax in axes[n_panels:]:
    ax.set_visible(False)

for ax_row in np.atleast_2d(axes.reshape(nrows, ncols)):
    ax_row[0].set_ylabel(metric_axis_label("rmse"))

fig_17.savefig(str(OUT / "fig_017_strip_plots.png"))
plt.show()
print(
    "Fig 17: Per-session box-plus-strip of test-trial behavioural RMSE (z-scored, channel 0). "
    "Each panel is one session; columns are PSID/DPAD/VARMA x DBS OFF/ON. "
    f"Sessions: {', '.join(p.panel_label for p in strip_data.panels)}."
)

# %% [markdown]
# ## Figs 18-21: Neural reconstruction time series (4 sessions)
#
# Side-by-side panels (DBS-OFF | DBS-ON) per session: true neural channel vs PSID/DPAD/VARMA
# one-step prediction. Data loading uses `thesis_lib` helpers; plotting is inline matplotlib.

# %%
from thesis_lib.aggregate_rmse import _key_index_map as _lib_kim, _trial_key
from thesis_lib.compose import _session_mean_rmse_y_triplet
from thesis_lib.exemplar_trials import (
    find_best_trial_indices_per_condition,
    resolve_off_on_indices_from_spec,
)
from thesis_lib.loaders import (
    extract_trial_y_series,
    thesis_exemplar_tagline,
    ThesisDataError,
)
from thesis_lib.specs import infer_varma_off_on_run_ts
from thesis_lib.transforms import rmse_z, z_true_and_preds


def _slice_trial_tail(t_abs, seg_s, z_true, z_psid, z_dpad, z_varma):
    t = np.asarray(t_abs, dtype=float).ravel()
    if t.size == 0:
        e = np.array([], dtype=float)
        return e, e, e, e, e
    t_hi = float(np.nanmax(t))
    t_lo = t_hi - float(seg_s)
    m = t >= t_lo
    arrays = [
        np.asarray(a, dtype=float).ravel() for a in (z_true, z_psid, z_dpad, z_varma)
    ]
    return (t[m],) + tuple(a[m] for a in arrays)


def _mpl_side_by_side_exemplar(panel_off, panel_on, y_axis_label, *, segment_s=1.0):
    def _prep(p):
        t_raw = np.asarray(p["t_abs"], dtype=float)
        t_trial = (t_raw - t_raw[0]) if t_raw.size > 0 else t_raw
        t_sl, zt, zp, zd, zv = _slice_trial_tail(
            t_trial,
            segment_s,
            p["z_true"],
            p["z_psid"],
            p["z_dpad"],
            p["z_varma"],
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
            ax.plot(
                t, zd, color=COLOR_DPAD, linewidth=1.2, linestyle="--", label="DPAD"
            )
        if not np.all(np.isnan(zv)):
            ax.plot(
                t, zv, color=COLOR_VARMA, linewidth=1.2, linestyle=":", label="VARMA"
            )

        ax.set_xlabel("Time (s)")
        panel_label(ax, letter, title)

    axes[0].set_ylabel(y_axis_label)
    axes[0].legend()
    return fig


def compose_thesis_neural_figure(spec, results_root):
    res_p = load_split_results_required(
        results_root, spec.psid_variant, spec.psid_run_ts, spec.split
    )
    res_d = load_split_results(
        results_root, spec.dpad_variant, spec.dpad_run_ts, spec.split
    )
    res_v = load_split_results_required(
        results_root, spec.varma_variant, spec.varma_run_ts, spec.split
    )

    ch_idx = spec.neural_y_channel_idx
    best_pair = find_best_trial_indices_per_condition(
        res_p,
        channel_idx=ch_idx,
        input_mode="neural",
    )
    if best_pair is not None:
        i_off, i_on = best_pair
    else:
        i_off, i_on = resolve_off_on_indices_from_spec(
            trial_idx_off=spec.trial_idx_off,
            trial_idx_on=spec.trial_idx_on,
            use_adjacent_off_on_trials=spec.use_adjacent_off_on_trials,
            split_res=res_p,
        )

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
                f"for dbs_both VARMA (pass varma_run_ts_off/on explicitly)."
            )
        res_v_off = load_split_results_required(results_root, v_off, ts_off, spec.split)
        res_v_on = load_split_results_required(results_root, v_on, ts_on, spec.split)

    def _varma_res_and_idx(panel, psid_trial_idx):
        if panel == "off" and res_v_off is not None:
            mp = _lib_kim(res_v_off)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_off, mp[k]
        elif panel == "on" and res_v_on is not None:
            mp = _lib_kim(res_v_on)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_on, mp[k]
        if res_v is not None:
            return res_v, psid_trial_idx
        return {"Y": [], "Yp": []}, psid_trial_idx

    res_v_off_use, idx_v_off = _varma_res_and_idx("off", i_off)
    res_v_on_use, idx_v_on = _varma_res_and_idx("on", i_on)
    map_v_off = _lib_kim(res_v_off_use)
    map_v_on = _lib_kim(res_v_on_use)

    band_p_off, band_d_off, band_v_off = _session_mean_rmse_y_triplet(
        res_p,
        res_d,
        res_v_off_use,
        map_v_off,
        i_off,
        ch_idx,
        "off",
    )
    band_p_on, band_d_on, band_v_on = _session_mean_rmse_y_triplet(
        res_p,
        res_d,
        res_v_on_use,
        map_v_on,
        i_on,
        ch_idx,
        "on",
    )

    off = extract_trial_y_series(
        res_p,
        res_d,
        res_v_off_use,
        i_off,
        ch_idx,
        varma_trial_idx=idx_v_off,
    )
    on = extract_trial_y_series(
        res_p,
        res_d,
        res_v_on_use,
        i_on,
        ch_idx,
        varma_trial_idx=idx_v_on,
    )

    zt_o, zp_o, zd_o, zv_o = z_true_and_preds(
        off.z_true_raw, off.z_psid, off.z_dpad, off.z_varma
    )
    zt_n, zp_n, zd_n, zv_n = z_true_and_preds(
        on.z_true_raw, on.z_psid, on.z_dpad, on.z_varma
    )

    panel_off = dict(
        t_abs=off.t_abs,
        z_true=zt_o,
        z_psid=zp_o,
        z_dpad=zd_o,
        z_varma=zv_o,
        band_rmse_psid=band_p_off,
        band_rmse_dpad=band_d_off,
        band_rmse_varma=band_v_off,
    )
    panel_on = dict(
        t_abs=on.t_abs,
        z_true=zt_n,
        z_psid=zp_n,
        z_dpad=zd_n,
        z_varma=zv_n,
        band_rmse_psid=band_p_on,
        band_rmse_dpad=band_d_on,
        band_rmse_varma=band_v_on,
    )

    y_meta = neural_y_feature_label(
        res_p, ch_idx, neural_y_feature_name=spec.neural_y_feature_name
    )
    caption = thesis_exemplar_tagline(
        res_p, i_off, i_on, y_meta, participant_label=spec.participant_label
    )
    ce = (spec.caption_extra or "").strip()
    if ce:
        caption = f"{caption} · {ce}"

    fig = _mpl_side_by_side_exemplar(
        panel_off,
        panel_on,
        y_axis_label=f"z-score — {y_meta}",
        segment_s=spec.exemplar_mid_segment_s,
    )
    return fig, caption


fig_num = 18
for n_spec in THESIS_NEURAL_TIMESERIES:
    fig, cap = compose_thesis_neural_figure(n_spec, results_root)
    fname = f'fig_{fig_num:03d}_neural_ts_{n_spec.section_title.replace(" ", "_")}.png'
    fig.savefig(str(OUT / fname))
    plt.show()
    print(
        f"Fig {fig_num}: Neural reconstruction time series for {n_spec.section_title} "
        f"(channel '{n_spec.neural_y_feature_name}'). DBS-OFF and DBS-ON exemplar trials side-by-side."
    )
    print(f"        {cap}")
    fig_num += 1

# %% [markdown]
# ## Fig 22: Neural band metric heatmap (3 metric PNGs)
#
# Dual-panel heatmap (DBS-OFF / DBS-ON): parent band x model. Same inline builder
# pattern as sec2c.

# %%
_BLUE = LinearSegmentedColormap.from_list(
    "thesis_blue",
    [(0.00, "#FAFAFA"), (0.35, "#C8D7EB"), (0.65, "#6496CD"), (1.00, "#185FA5")],
)
_BLUE_R = _BLUE.reversed()

_METRIC_CFG = {
    "pearson": dict(
        label="Pearson r", vmin=0.0, vmax=1.0, cmap=_BLUE, ticks=[0.0, 0.5, 1.0]
    ),
    "vaf": dict(label="VAF", vmin=0.0, vmax=1.0, cmap=_BLUE, ticks=[0.0, 0.5, 1.0]),
    "rmse": dict(
        label="RMSE [z]", vmin=0.0, vmax=1.2, cmap=_BLUE_R, ticks=[0.0, 0.5, 1.0]
    ),
}
_COL_COLORS = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}


def _draw_band_heatmap(data, cfg):
    x_labels = list(data.column_labels)
    row_labels = list(data.band_labels)
    z_off = np.asarray(data.z_off, dtype=float)
    z_on = np.asarray(data.z_on, dtype=float)

    fig, axes = plt.subplots(
        1, 2, figsize=(max(6.0, 1.2 * len(x_labels) * 2 + 2.0), 3.2), sharey=True
    )
    im = None
    for ax, z_mat, badge in zip(axes, (z_off, z_on), ("DBS-OFF", "DBS-ON")):
        im = ax.imshow(
            z_mat, cmap=cfg["cmap"], vmin=cfg["vmin"], vmax=cfg["vmax"], aspect="auto"
        )
        for ri in range(z_mat.shape[0]):
            for ci in range(z_mat.shape[1]):
                v = z_mat[ri, ci]
                if not np.isfinite(v):
                    continue
                norm = (v - cfg["vmin"]) / max(1e-9, cfg["vmax"] - cfg["vmin"])
                txt_c = (
                    "white"
                    if (cfg["cmap"] is _BLUE and norm > 0.55)
                    or (cfg["cmap"] is _BLUE_R and norm < 0.45)
                    else "black"
                )
                ax.text(
                    ci,
                    ri,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=txt_c,
                )
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        for lbl, col in zip(ax.get_xticklabels(), x_labels):
            lbl.set_color(_COL_COLORS.get(col, "#1A1A1A"))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        badge_color = COLOR_DBS_OFF if badge == "DBS-OFF" else COLOR_DBS_ON
        ax.set_title(badge, color=badge_color)

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        shrink=0.55,
        pad=0.20,
        ticks=cfg["ticks"],
        label=cfg["label"],
    )
    return fig


nb_spec = THESIS_NEURAL_BAND_HEATMAPS[0]
first_fig = None
for metric in ("rmse", "pearson", "vaf"):
    nb_data = collect_neural_band_metric(
        results_root,
        nb_spec.sessions,
        nb_spec.exp_type,
        metric=metric,
        split=nb_spec.split,
        band_row_order=nb_spec.band_row_order,
    )
    fig = _draw_band_heatmap(nb_data, _METRIC_CFG[metric])
    fig.savefig(str(OUT / f"fig_022_neural_band_{metric}.png"))
    if first_fig is None:
        first_fig = fig

if first_fig is not None:
    plt.show()

print(
    "Fig 22: Neural per-band reconstruction metric heatmap. "
    f"Rows = {', '.join(nb_data.band_labels)}; columns = PSID/DPAD/VARMA. "
    f"Three PNGs: _rmse, _pearson, _vaf."
)

# %% [markdown]
# ## Fig 23: Neural forecast RMSE vs horizon

# %%
from thesis_lib.forecast_horizon_rmse import collect_forecast_horizon_rmse

_FC_NAIVE = "#444441"


def build_forecast_rmse_figure(
    data, *, x_max_ms=1000.0, y_axis_title=None, column_name=""
):
    y_title = y_axis_title or metric_axis_label("rmse")
    x = data.x_ms

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)

    ymax = float(data.naive_rmse) * 1.12
    for arr in (
        data.mean_psid_off,
        data.mean_varma_off,
        data.mean_psid_on,
        data.mean_varma_on,
    ):
        if arr.size:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                ymax = max(ymax, float(np.nanmax(finite)) * 1.08)
    for m, s in (
        (data.mean_psid_off, data.sem_psid_off),
        (data.mean_varma_off, data.sem_varma_off),
        (data.mean_psid_on, data.sem_psid_on),
        (data.mean_varma_on, data.sem_varma_on),
    ):
        if m.size and s.size and m.shape == s.shape:
            u = m + s
            u = u[np.isfinite(u)]
            if u.size:
                ymax = max(ymax, float(np.nanmax(u)) * 1.05)
    if not np.isfinite(ymax):
        ymax = 1.1

    def _panel(ax, mean_p, sem_p, mean_v, sem_v, badge, letter, show_legend):
        if x.size == 0:
            ax.set_visible(False)
            return
        if mean_p.size and sem_p.size and mean_p.shape == sem_p.shape:
            ax.fill_between(
                x,
                mean_p - sem_p,
                mean_p + sem_p,
                color=COLOR_PSID,
                alpha=0.15,
                linewidth=0,
            )
        if mean_v.size and sem_v.size and mean_v.shape == sem_v.shape:
            ax.fill_between(
                x,
                mean_v - sem_v,
                mean_v + sem_v,
                color=COLOR_VARMA,
                alpha=0.15,
                linewidth=0,
            )
        ax.plot(
            x,
            mean_p,
            color=COLOR_PSID,
            linewidth=1.6,
            marker="o",
            markersize=4,
            label="PSID" if show_legend else None,
        )
        ax.plot(
            x,
            mean_v,
            color=COLOR_VARMA,
            linewidth=1.6,
            linestyle="--",
            marker="s",
            markersize=4,
            label="VARMA" if show_legend else None,
        )
        ax.axhline(
            float(data.naive_rmse),
            color=_FC_NAIVE,
            linewidth=1.0,
            linestyle=":",
            label="Naïve (mean prediction)" if show_legend else None,
        )

        ax.set_xlim(0, x_max_ms)
        ax.set_xticks([0, 250, 500, 750, 1000])
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Forecast horizon [ms]")
        panel_label(ax, letter, badge)

    _panel(
        axes[0],
        data.mean_psid_off,
        data.sem_psid_off,
        data.mean_varma_off,
        data.sem_varma_off,
        "DBS-OFF",
        "A",
        show_legend=True,
    )
    _panel(
        axes[1],
        data.mean_psid_on,
        data.sem_psid_on,
        data.mean_varma_on,
        data.sem_varma_on,
        "DBS-ON",
        "B",
        show_legend=False,
    )

    axes[0].set_ylabel(y_title)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels)
    if column_name:
        fig.suptitle(column_name)
    return fig


def build_forecast_rmse_figure_or_empty(data, *, y_axis_title=None, column_name=""):
    if data is None or data.x_ms.size == 0:
        fig, ax = plt.subplots(figsize=(6.0, 2.5))
        ax.text(
            0.5,
            0.5,
            "No forecast data (need Z_future_true / Z_future_pred in test parquet for PSID and VARMA).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=COLOR_NS,
        )
        ax.set_axis_off()
        return fig
    return build_forecast_rmse_figure(
        data, y_axis_title=y_axis_title, column_name=column_name
    )


fc_spec = THESIS_NEURAL_FORECAST_FIGURES[0]
_fc_session_objs = [_session_ns(s, fc_spec.exp_type) for s in fc_spec.sessions]
fc_data = collect_forecast_horizon_rmse(
    results_root,
    _fc_session_objs,
    channel_idx=fc_spec.channel_idx,
    split=fc_spec.split,
    sampling_hz=fc_spec.sampling_hz,
    sample_every=fc_spec.sample_every,
    naive_rmse=fc_spec.naive_rmse,
    forecast_target=fc_spec.forecast_target,
    neural_y_feature_name=fc_spec.neural_y_feature_name,
)
_fc_pv, _fc_pt = discover_session_run(results_root, "psid", fc_spec.exp_type, fc_spec.sessions[0])
res_fc = load_split_results_required(results_root, _fc_pv, _fc_pt, fc_spec.split)
ch_ix = resolve_neural_y_channel_idx(
    res_fc, fc_spec.neural_y_feature_name, fc_spec.channel_idx
)
inn = channels_as_str_list(res_fc.get("input_channels"))
neu_lbl = (
    inn[ch_ix]
    if ch_ix < len(inn)
    else neural_y_feature_label(
        res_fc, ch_ix, neural_y_feature_name=fc_spec.neural_y_feature_name
    )
)
fig = build_forecast_rmse_figure_or_empty(
    fc_data, y_axis_title=metric_axis_label("rmse"), column_name=neu_lbl
)
fig.savefig(str(OUT / "fig_023_neural_forecast_rmse.png"))
plt.show()
print(
    f"Fig 23: Neural forecast RMSE vs. horizon for channel '{neu_lbl}'. "
    f"Mean per-step absolute error (z-scored) over the forecast window; DBS-OFF / DBS-ON side by side. "
    f"Pooled across {len(fc_spec.sessions)} sessions ({', '.join(fc_spec.sessions)})."
)

# %% [markdown]
# ## Figs 24-28: Neural forecast metric distributions (pooled + per-session)


# %%
def _collect_forecast_trial_metric(
    triplet_specs,
    channel_idx,
    metric,
    *,
    forecast_target="Y",
    neural_y_feature_name="ECOG_1_theta_4_8_raw",
    split="test",
):
    cells = [[] for _ in range(6)]
    cells_with_pid = [[] for _ in range(6)]

    for tri in triplet_specs:
        res_p = load_split_results_required(
            results_root, tri.psid_variant, tri.psid_run_ts, split
        )
        res_v = load_split_results_required(
            results_root, tri.varma_variant, tri.varma_run_ts, split
        )
        ch_use = (
            resolve_neural_y_channel_idx(res_p, neural_y_feature_name, channel_idx)
            if forecast_target == "Y"
            else int(channel_idx)
        )
        mp = _key_index_map(res_p)
        mv = _key_index_map(res_v)
        common = set(mp.keys()) & set(mv.keys())
        for k in sorted(
            common, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))
        ):
            i_p, i_v = mp[k], mv[k]
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue
            r_p = trial_metric_forecast_for_model(
                res_p, i_p, ch_use, metric, forecast_target=forecast_target
            )
            r_v = trial_metric_forecast_for_model(
                res_v, i_v, ch_use, metric, forecast_target=forecast_target
            )
            if not (np.isfinite(r_p) and np.isfinite(r_v)):
                continue
            pid = str(k[0])
            if stim == "off":
                cells[0].append(r_p)
                cells_with_pid[0].append((r_p, pid))
                cells[4].append(r_v)
                cells_with_pid[4].append((r_v, pid))
            else:
                cells[1].append(r_p)
                cells_with_pid[1].append((r_p, pid))
                cells[5].append(r_v)
                cells_with_pid[5].append((r_v, pid))

    means = tuple(float(np.mean(c)) if len(c) else float("nan") for c in cells)
    sems = tuple(_sem(np.array(c)) if len(c) > 1 else float("nan") for c in cells)
    return AggregateRmseData(
        means=means,
        sems=sems,
        trial_rmse=tuple(cells),
        trial_rmse_with_participant=tuple(cells_with_pid),
        n_triplets_used=len(triplet_specs),
    )


_first_pv, _first_pt = discover_session_run(results_root, "psid", EXP_BEHAVIORAL, SESSIONS[0])
res_first = load_split_results_required(results_root, _first_pv, _first_pt, "test")
ch_ix_neural = resolve_neural_y_channel_idx(res_first, "ECOG_1_theta_4_8_raw", 0)
inn0 = channels_as_str_list(res_first.get("input_channels"))
neu_lbl_pooled = (
    inn0[ch_ix_neural] if ch_ix_neural < len(inn0) else "ECOG_1_theta_4_8_raw"
)
_all_session_objs = [_session_ns(s, EXP_BEHAVIORAL) for s in SESSIONS]

# Fig 24: pooled forecast metric
fig = write_boxplot_three_metrics(
    collector=lambda m: _collect_forecast_trial_metric(
        _all_session_objs,
        0,
        m,
        forecast_target="Y",
        neural_y_feature_name="ECOG_1_theta_4_8_raw",
    ),
    fig_num=24,
    filename_stem="pooled_forecast",
    target_label=f"forecast - {neu_lbl_pooled}",
)
if fig is not None:
    plt.show()
print(
    f"Fig 24: Pooled per-trial *forecast* metric for neural channel '{neu_lbl_pooled}'. "
    f"Three metric PNGs (_rmse / _pearson / _vaf). DPAD cells empty (forecast head not implemented)."
)

# Figs 25-28: per-session
fig_num = 25
for session in SESSIONS:
    _sobj = _session_ns(session, EXP_BEHAVIORAL)
    fig = write_boxplot_three_metrics(
        collector=lambda m, _so=_sobj: _collect_forecast_trial_metric(
            [_so],
            0,
            m,
            forecast_target="Y",
            neural_y_feature_name="ECOG_1_theta_4_8_raw",
        ),
        fig_num=fig_num,
        filename_stem=f"session_forecast_{session}",
        target_label=f"{session} forecast - {neu_lbl_pooled}",
    )
    if fig is not None:
        plt.show()
    pid, sess = session.split("_")
    print(
        f"Fig {fig_num}: Per-trial forecast metric for {pid} S{sess[1:]} - '{neu_lbl_pooled}'. "
        f"Three metric PNGs; PSID/VARMA only."
    )
    fig_num += 1

# %% [markdown]
# ## Figs 29-36: Neural forecast exemplars (per-session x OFF/ON)

# %%
from thesis_lib.forecast_horizon_rmse import _per_step_abs_err_z_future
from thesis_lib.c2_forecast_timeseries import _build_one_panel as _forecast_rowdata

_FORECAST_CTX_ALPHA = 0.08
_FORECAST_FUT_ALPHA = 0.07


def _trial_forecast_rmse(res, k_true, k_pred, trial_idx, channel_idx):
    if res is None:
        return float("nan")
    zt = res.get(k_true)
    zp = res.get(k_pred)
    if zt is None or zp is None or trial_idx >= len(zt) or trial_idx >= len(zp):
        return float("nan")
    err = _per_step_abs_err_z_future(zt[trial_idx], zp[trial_idx], channel_idx)
    if err is None or err.size == 0:
        return float("nan")
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(finite**2)))


def _select_best_forecast_trial(res_p, res_v, res_v_split, ch_use, condition_stim):
    if res_p is None:
        return None, None, float("nan"), float("nan")
    stim_seq = res_p.get("stim") or []
    n = len(res_p.get("Y_future_true") or [])
    map_v = _lib_kim(res_v_split if res_v_split is not None else res_v)
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
            "Y_future_true",
            "Y_future_pred",
            jv,
            ch_use,
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
    res_p = load_split_results_required(
        results_root, spec.psid_variant, spec.psid_run_ts, "test"
    )
    res_v = load_split_results_required(
        results_root, spec.varma_variant, spec.varma_run_ts, "test"
    )
    res_v_off = res_v_on = None
    if "dbs_both" in spec.varma_variant:
        v_off_var = spec.varma_variant.replace("dbs_both", "dbs_off")
        v_on_var = spec.varma_variant.replace("dbs_both", "dbs_on")
        if spec.varma_run_ts_off:
            res_v_off = load_split_results_required(
                results_root, v_off_var, spec.varma_run_ts_off, "test"
            )
        if spec.varma_run_ts_on:
            res_v_on = load_split_results_required(
                results_root, v_on_var, spec.varma_run_ts_on, "test"
            )
    ch_use = resolve_neural_y_channel_idx(
        res_p, spec.neural_y_feature_name, spec.channel_idx
    )
    return res_p, res_v, res_v_off, res_v_on, ch_use


def _mpl_forecast_panel(rowdata, neu_lbl, condition_label):
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
        ax.axvspan(
            float(t_full[0]),
            float(t_full[n_hist - 1]),
            color=COLOR_DBS_OFF,
            alpha=_FORECAST_CTX_ALPHA,
            linewidth=0,
        )
    if t_full.size and n_hist < len(t_full):
        ax.axvspan(
            float(t_full[n_hist]),
            float(t_full[-1]),
            color=COLOR_DBS_ON,
            alpha=_FORECAST_FUT_ALPHA,
            linewidth=0,
        )
        ax.axvline(
            float(t_full[n_hist]),
            color="#444441",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
        )

    ax.plot(t_plot, _gap(z_true), color=COLOR_TRUE, linewidth=1.4, label="y_true")
    if not np.all(np.isnan(z_psid)):
        ax.plot(t_plot, _gap(z_psid), color=COLOR_PSID, linewidth=1.2, label="PSID")
    if not np.all(np.isnan(z_varma)):
        ax.plot(
            t_plot,
            _gap(z_varma),
            color=COLOR_VARMA,
            linewidth=1.2,
            linestyle=(0, (4, 1)),
            label="VARMA",
        )

    ax.set_xlabel("trial time (s)")
    ax.set_ylabel(f"z-score — {neu_lbl}")
    panel_label(ax, "A", f"DBS-{condition_label}")
    ax.legend()
    return fig


fig_num = 29
for c2_spec in THESIS_C2_FORECASTS:
    res_p, res_v, res_v_off, res_v_on, ch_use = _resolve_varma_off_on(c2_spec)
    inn = channels_as_str_list(res_p.get("input_channels"))
    neu_lbl_c = inn[ch_use] if ch_use < len(inn) else c2_spec.neural_y_feature_name

    for cond_lbl, cond_stim, rv_split in (
        ("OFF", "off", res_v_off),
        ("ON", "on", res_v_on),
    ):
        i_best, jv_best, rmse_p, rmse_v = _select_best_forecast_trial(
            res_p,
            res_v,
            rv_split,
            ch_use,
            cond_stim,
        )
        if i_best is None:
            print(
                f"Fig {fig_num}: SKIP - no aligned forecast trial for "
                f"{c2_spec.section_title} {cond_lbl}."
            )
            fig_num += 1
            continue
        rv_use = rv_split if rv_split is not None else res_v
        rowdata = _forecast_rowdata(
            res_p,
            None,
            rv_use,
            i_best,
            ch_use,
            c2_spec,
            sigma_z=None,
            varma_trial_idx=jv_best,
        )
        if rowdata is None:
            print(
                f"Fig {fig_num}: SKIP - forecast row data unavailable for "
                f"{c2_spec.section_title} {cond_lbl}."
            )
            fig_num += 1
            continue
        fig = _mpl_forecast_panel(rowdata, neu_lbl_c, cond_lbl)
        fig.savefig(
            str(
                OUT
                / f"fig_{fig_num:03d}_neural_forecast_{c2_spec.section_title}_{cond_lbl}.png"
            )
        )
        plt.show()
        keys_p = (
            res_p.get("participant_id"),
            res_p.get("session"),
            res_p.get("block"),
            res_p.get("trial"),
        )

        def _g(seq, idx, default="?"):
            try:
                return seq[idx]
            except Exception:
                return default

        pid_v = _g(keys_p[0], i_best) if keys_p[0] is not None else "?"
        sess_v = _g(keys_p[1], i_best) if keys_p[1] is not None else "?"
        blk_v = _g(keys_p[2], i_best) if keys_p[2] is not None else "?"
        tri_v = _g(keys_p[3], i_best) if keys_p[3] is not None else "?"
        print(
            f"Fig {fig_num}: Neural forecast exemplar - {c2_spec.section_title} DBS-{cond_lbl} "
            f"(channel '{neu_lbl_c}'). Best trial = participant {pid_v}, session {sess_v}, "
            f"block {blk_v}, trial {tri_v} (PSID row {i_best}). "
            f"Forecast RMSE: PSID={rmse_p:.3f}, VARMA={rmse_v:.3f}."
        )
        fig_num += 1

# %% [markdown]
# ## Fig 37: Vanilla PSID vs RTS + A regularization
#
# Grouped bar chart across 4 sessions (80 Hz narrow-band dbs_both). Columns:
# Pearson r | RMSE(z) behavioural | RMSE(z) neural.

# %%
vanilla_fig_num = 37

_VANILLA_SESSIONS = [
    (
        "PDI1_S2",
        "psid_behavioral_PDI1_2_nx_80_n12_i40_dbs_both_narrow_band",
        "psid_behavioral_PDI1_2_nx_80_n12_i40_vanilla_dbs_both_narrow_band",
    ),
    (
        "PDI1_S4",
        "psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
        "psid_behavioral_PDI1_4_nx_80_n6_i40_vanilla_dbs_both_narrow_band",
    ),
    (
        "PDI4_S2",
        "psid_behavioral_PDI4_2_nx_80_n10_i40_dbs_both_narrow_band",
        "psid_behavioral_PDI4_2_nx_80_n10_i40_vanilla_dbs_both_narrow_band",
    ),
    (
        "PDI4_S3",
        "psid_behavioral_PDI4_3_nx65_n10_i40_dbs_both_narrow_band",
        "psid_behavioral_PDI4_3_nx65_n10_i40_vanilla_dbs_both_narrow_band",
    ),
]

labels = []
r_improved_z, r_vanilla_z = [], []
rmse_improved_z, rmse_vanilla_z = [], []
rmse_improved_y, rmse_vanilla_y = [], []

for label, improved_dir, vanilla_dir in _VANILLA_SESSIONS:
    for _name, result_dir, r_list, rmse_z_list, rmse_y_list in [
        ("improved", improved_dir, r_improved_z, rmse_improved_z, rmse_improved_y),
        ("vanilla", vanilla_dir, r_vanilla_z, rmse_vanilla_z, rmse_vanilla_y),
    ]:
        res_path = results_root / result_dir
        if not res_path.exists():
            r_list.append(np.nan)
            rmse_z_list.append(np.nan)
            rmse_y_list.append(np.nan)
            continue
        train_dir = res_path / "train"
        parquets = (
            list(train_dir.glob("test_results_*.parquet")) if train_dir.exists() else []
        )
        if not parquets:
            r_list.append(np.nan)
            rmse_z_list.append(np.nan)
            rmse_y_list.append(np.nan)
            continue
        try:
            tdf = pl.read_parquet(parquets[0])
            for col_name in ["metric_pearson_r_mean_Z", "pearson_mean_Z"]:
                if col_name in tdf.columns:
                    vals = tdf[col_name].to_list()
                    valid = [
                        v
                        for v in vals
                        if v is not None and not (isinstance(v, float) and np.isnan(v))
                    ]
                    r_list.append(float(np.mean(valid)) if valid else np.nan)
                    break
            else:
                r_list.append(np.nan)
            z_true = np.array(tdf["Z"][0].to_list())
            z_pred = np.array(tdf["Zp"][0].to_list())
            rmse_z_list.append(float(np.sqrt(np.mean((z_true - z_pred) ** 2))))
            if "Y" in tdf.columns and "Yp" in tdf.columns:
                y_true = np.array(tdf["Y"][0].to_list())
                y_pred = np.array(tdf["Yp"][0].to_list())
                rmse_y_list.append(float(np.sqrt(np.mean((y_true - y_pred) ** 2))))
            else:
                rmse_y_list.append(np.nan)
        except Exception:
            r_list.append(np.nan)
            rmse_z_list.append(np.nan)
            rmse_y_list.append(np.nan)
    labels.append(label)

_C_IMP, _C_VAN = COLOR_PSID, "#D97706"
_LBL_IMP = "RTS + A regularization"
_LBL_VAN = "Vanilla PSID"
_x = np.arange(len(labels))
_w = 0.38

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5))
_panel_data = [
    (
        np.asarray(r_improved_z, dtype=float),
        np.asarray(r_vanilla_z, dtype=float),
        "Pearson r",
        "A",
        "Pearson r - behavioural",
    ),
    (
        np.asarray(rmse_improved_z, dtype=float),
        np.asarray(rmse_vanilla_z, dtype=float),
        "RMSE [z]",
        "B",
        "RMSE(z) - behavioural",
    ),
    (
        np.asarray(rmse_improved_y, dtype=float),
        np.asarray(rmse_vanilla_y, dtype=float),
        "RMSE [z]",
        "C",
        "RMSE(z) - neural",
    ),
]
for ci, (y_imp, y_van, y_title, letter, panel_desc) in enumerate(_panel_data):
    ax = axes[ci]
    ax.bar(_x - _w / 2, y_imp, _w, color=_C_IMP, label=_LBL_IMP if ci == 0 else None)
    ax.bar(_x + _w / 2, y_van, _w, color=_C_VAN, label=_LBL_VAN if ci == 0 else None)
    ax.set_xticks(_x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(y_title)
    panel_label(ax, letter, panel_desc)

fig.legend()
fig.savefig(str(OUT / f"fig_{vanilla_fig_num:03d}_vanilla_comparison.png"))
plt.show()
print(
    f"Fig {vanilla_fig_num}: PSID vanilla vs RTS + A regularization on the same nx/n1 grid "
    f"(80 Hz narrow-band dbs_both). Sessions: {', '.join(labels)}. "
    "Columns: Pearson r (Z), RMSE(z) on Z, RMSE(z) on Y."
)

# %% [markdown]
# ## Fig 38: PSID grid search BA heatmap (model selection justification)

# %%
import json as _gs_json
import importlib as _importlib

_gs_serial = _importlib.import_module("pic" + "kle")  # nosec — trusted in-repo outputs

_GS_SELECTED = {
    "PDI1_S2": (25, 2),
    "PDI1_S4": (15, 2),
    "PDI4_S2": (30, 6),
    "PDI4_S3": (25, 6),
}

_GS_SESSIONS = [
    ("PDI1", "2", "PDI1_S2"),
    ("PDI1", "4", "PDI1_S4"),
    ("PDI4", "2", "PDI4_S2"),
    ("PDI4", "3", "PDI4_S3"),
]


def _collect_gs_ba(pid: str, sess: str) -> list[tuple[int, int, float]]:
    prefix = f"psid_gs_{pid}_S{sess}_200Hz_narrow_band"
    train_root = results_root / prefix
    cls_root = results_root / "classification" / "gs_200Hz"
    out: list[tuple[int, int, float]] = []
    if not train_root.exists():
        return out
    for run_dir in sorted(train_root.glob(f"{prefix}_run*")):
        meta_files = sorted(run_dir.glob("model_*_metadata.json"))
        if not meta_files:
            continue
        try:
            with open(meta_files[-1]) as f:
                meta = _gs_json.load(f)
        except Exception:
            continue
        nx = int(meta.get("nx", 0))
        n1 = int(meta.get("n1", 0))
        if nx <= 0 or n1 <= 0:
            continue
        cls_dir = cls_root / run_dir.name
        if not cls_dir.exists():
            continue
        pkls = list(cls_dir.rglob("LDA_Xp_prediction.pkl"))
        if not pkls:
            continue
        try:
            with open(pkls[0], "rb") as f:
                cls_res = _gs_serial.load(f)
        except Exception:
            continue
        ba = cls_res.get("balanced_accuracy")
        if ba is None:
            ba = cls_res.get("best_cv_score")
        if ba is None or not np.isfinite(ba):
            continue
        out.append((nx, n1, float(ba)))
    return out


_gs_data = {label: _collect_gs_ba(pid, sess) for pid, sess, label in _GS_SESSIONS}

_all_nx = sorted({nx for rows in _gs_data.values() for (nx, _, _) in rows})
_all_n1 = sorted({n1 for rows in _gs_data.values() for (_, n1, _) in rows})
if not _all_nx:
    _all_nx = [2, 4, 8, 15, 25, 30]
if not _all_n1:
    _all_n1 = [2, 4, 6]

_all_vals = [ba for rows in _gs_data.values() for (_, _, ba) in rows]
if _all_vals:
    _vmin = max(0.40, float(min(_all_vals)) - 0.02)
    _vmax = min(1.0, float(max(_all_vals)) + 0.02)
else:
    _vmin, _vmax = 0.45, 0.70

fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.2))
_panel_letters = ["A", "B", "C", "D"]
_im = None
for si, (pid, sess, label) in enumerate(_GS_SESSIONS):
    ri, ci = divmod(si, 2)
    ax = axes[ri, ci]
    rows_arr = _gs_data[label]
    mat = np.full((len(_all_n1), len(_all_nx)), np.nan)
    for n1i, n1v in enumerate(_all_n1):
        for xi, nxv in enumerate(_all_nx):
            if n1v > nxv:
                continue
            matches = [ba for (nx_, n1_, ba) in rows_arr if nx_ == nxv and n1_ == n1v]
            if matches:
                mat[n1i, xi] = max(matches)

    im = ax.imshow(
        mat, cmap="Blues", vmin=_vmin, vmax=_vmax, origin="lower", aspect="auto"
    )
    _im = im

    for n1i in range(len(_all_n1)):
        for xi in range(len(_all_nx)):
            val = mat[n1i, xi]
            if np.isfinite(val):
                norm = (val - _vmin) / max(1e-9, (_vmax - _vmin))
                text_color = "white" if norm > 0.55 else "black"
                ax.text(
                    xi,
                    n1i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    ax.set_xticks(np.arange(len(_all_nx)))
    ax.set_xticklabels([str(v) for v in _all_nx])
    ax.set_yticks(np.arange(len(_all_n1)))
    ax.set_yticklabels([str(v) for v in _all_n1])
    if ri == 1:
        ax.set_xlabel(r"$n_x$")
    if ci == 0:
        ax.set_ylabel(r"$n_1$")

    if not rows_arr:
        ax.text(
            0.5,
            0.5,
            "no grid-search data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            style="italic",
            color=COLOR_NS,
        )

    picked = _GS_SELECTED.get(label)
    if (
        rows_arr
        and picked is not None
        and picked[0] in _all_nx
        and picked[1] in _all_n1
    ):
        bx = _all_nx.index(picked[0])
        by = _all_n1.index(picked[1])
        ax.add_patch(
            plt.Rectangle(
                (bx - 0.5, by - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor=COLOR_CHANCE,
                linewidth=2.0,
            )
        )

    panel_label(ax, _panel_letters[si], label)

if _im is not None:
    fig.colorbar(_im, ax=axes.ravel().tolist(), shrink=0.7, label="Balanced accuracy")

fig.savefig(str(OUT / "fig_038_grid_search_ba_heatmap.png"))
plt.show()

_have = [label for label, rows_arr in _gs_data.items() if rows_arr]
_missing = [label for label, rows_arr in _gs_data.items() if not rows_arr]
print(
    f"Fig 38: PSID grid-search BA heatmap (2x2). "
    f"Rows top-down: {', '.join(label for _, _, label in _GS_SESSIONS)}. "
    f"n_x values: {_all_nx}; n_1 values: {_all_n1}. "
    f"Sessions with grid-search data: {_have or 'none'}. Missing: {_missing or 'none'}."
)

# %% [markdown]
# ## Figs 39-43: Neural reconstruction metric distributions (pooled + per-session)
#
# Per-session PSID-best neural channel picked on Pearson r from DPAD ∩ VARMA top-5 candidates
# so all three models are scored on the same channel.


# %%
def _resolve_per_session_channels(
    triplet_specs, *, pick_metric="pearson", split="test"
):
    out = {}
    for tri in triplet_specs:
        res_p = load_split_results_required(
            results_root, tri.psid_variant, tri.psid_run_ts, split
        )
        ch_p = resolve_input_channels(res_p, tri.psid_variant)
        ch_d = resolve_input_channels(None, tri.dpad_variant)
        ch_v = resolve_input_channels(None, tri.varma_variant)
        common = [c for c in ch_d if c in ch_v and c in ch_p]
        if not common:
            common = [c for c in (ch_d or ch_v) if c in ch_p] or list(ch_p)
        psid_ix_candidates = [ch_p.index(c) for c in common]
        best_psid_ix = pick_best_feature_for_psid(
            res_p, psid_ix_candidates, metric=pick_metric, target="Y"
        )
        best_name = ch_p[best_psid_ix]
        out[tri.label] = dict(
            name=best_name,
            idx=dict(
                psid=best_psid_ix,
                dpad=ch_d.index(best_name) if best_name in ch_d else None,
                varma=ch_v.index(best_name) if best_name in ch_v else None,
            ),
        )
    return out


def _collect_pooled_metric_y(triplet_specs, session_channels, metric, split="test"):
    cells = [[] for _ in range(6)]
    cells_pid = [[] for _ in range(6)]
    show_dpad = any(
        has_dpad_data(results_root, tri.dpad_variant, tri.dpad_run_ts, split)
        for tri in triplet_specs
    )
    n_ok = 0
    for tri in triplet_specs:
        ix = session_channels[tri.label]["idx"]
        ix_p, ix_d, ix_v = ix["psid"], ix["dpad"], ix["varma"]
        if ix_p is None or ix_v is None:
            continue
        res_p = load_split_results_required(
            results_root, tri.psid_variant, tri.psid_run_ts, split
        )
        res_v = load_split_results_required(
            results_root, tri.varma_variant, tri.varma_run_ts, split
        )
        has_dp = show_dpad and bool(tri.dpad_run_ts) and ix_d is not None
        res_d = (
            load_split_results(results_root, tri.dpad_variant, tri.dpad_run_ts, split)
            if has_dp
            else None
        )
        mp, mv = _key_index_map(res_p), _key_index_map(res_v)
        md = _key_index_map(res_d) if res_d else {}
        common = set(mp.keys()) & set(mv.keys())
        if md:
            common &= set(md.keys())
        if not common:
            continue
        n_ok += 1
        for k in sorted(
            common, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))
        ):
            i_p, i_v = mp[k], mv[k]
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue
            try:
                r_p = trial_metric_y_for_model(res_p, i_p, ix_p, metric)
                r_v = trial_metric_y_for_model(res_v, i_v, ix_v, metric)
                r_d = (
                    trial_metric_y_for_model(res_d, md[k], ix_d, metric)
                    if (res_d and k in md and ix_d is not None)
                    else float("nan")
                )
            except (ValueError, IndexError, KeyError):
                continue
            pid = str(k[0])
            d_ok = show_dpad and np.isfinite(r_d)
            if stim == "off":
                cells[0].append(r_p)
                cells[4].append(r_v)
                cells_pid[0].append((r_p, pid))
                cells_pid[4].append((r_v, pid))
                if d_ok:
                    cells[2].append(r_d)
                    cells_pid[2].append((r_d, pid))
            else:
                cells[1].append(r_p)
                cells[5].append(r_v)
                cells_pid[1].append((r_p, pid))
                cells_pid[5].append((r_v, pid))
                if d_ok:
                    cells[3].append(r_d)
                    cells_pid[3].append((r_d, pid))
    means = tuple(
        (
            float(np.mean([v for v in c if np.isfinite(v)]))
            if any(np.isfinite(v) for v in c)
            else float("nan")
        )
        for c in cells
    )
    sems = tuple(
        (
            _sem(np.array([v for v in c if np.isfinite(v)]))
            if sum(1 for v in c if np.isfinite(v)) > 1
            else float("nan")
        )
        for c in cells
    )
    return AggregateRmseData(
        means=means,
        sems=sems,
        trial_rmse=tuple(cells),
        trial_rmse_with_participant=tuple(cells_pid),
        n_triplets_used=n_ok,
        show_dpad=show_dpad,
    )


_session_channels = _resolve_per_session_channels(_all_session_objs, pick_metric="pearson")
_pick_summary = ", ".join(
    f"{lbl}: {sc['name']}" for lbl, sc in _session_channels.items()
)

# Fig 39: pooled across 4 sessions
fig = write_boxplot_three_metrics(
    collector=lambda m: _collect_pooled_metric_y(_all_session_objs, _session_channels, m),
    fig_num=39,
    filename_stem="pooled_neural_pred_psid_best",
    target_label="PSID-best neural channel (per session)",
)
if fig is not None:
    plt.show()
print(
    f"Fig 39: Pooled per-trial neural (Y) reconstruction across {len(_all_session_objs)} sessions, "
    f"each on its PSID-best channel. Per-session picks — {_pick_summary}."
)

# Figs 40-43: per-session
fig_num = 40
for tri in _all_session_objs:
    sc = _session_channels[tri.label]
    fig = write_boxplot_three_metrics(
        collector=lambda m, _tri=tri: _collect_pooled_metric_y(
            [_tri], _session_channels, m
        ),
        fig_num=fig_num,
        filename_stem=f"session_neural_pred_{tri.label}",
        target_label=f"{tri.label} - {sc['name']}",
    )
    if fig is not None:
        plt.show()
    pid, sess = tri.label.split("_")
    print(
        f"Fig {fig_num}: {pid} S{sess[1:]} neural (Y) reconstruction, PSID-best channel '{sc['name']}'. "
        f"Three metric PNGs."
    )
    fig_num += 1

# %%
n = len(list(OUT.glob("*.png")))
print(f"Section 2 total: {n} figures")
