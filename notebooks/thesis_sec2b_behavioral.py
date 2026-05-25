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
# # Sec 2b — Behavioural decoding group metrics
#
# Group-level per-trial metric for predicting behaviour (tracing velocity_x,
# acceleration_magnitude) from neural activity with PSID / DPAD / VARMA.
#
# Each boxplot scope emits **three PNGs** (RMSE, Pearson r, VAF on raw signals)
# so the best metric for the thesis narrative can be picked on inspection.
#
# * Figs 7-8   — pooled per-trial metric across all 4 sessions (each: 3 PNGs)
# * Figs 9-16  — per-session per-trial metric (2 channels x 4 sessions; each: 3 PNGs)
# * Fig 17     — per-participant box-plus-strip summary (RMSE only, legacy layout)
# * Figs 46-49 — per-session LFP (laplacian) reconstruction

# %%
import sys, os

os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")
sys.path.insert(0, ".")
sys.path.insert(0, "notebooks")

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from thesis_style import (
    COLOR_DPAD,
    COLOR_PSID,
    COLOR_VARMA,
    PARTICIPANT_COLORS,
    apply_thesis_style,
    panel_label,
)
from thesis_sec2_common import *

from thesis_utils import (
    collect_pooled_rmse,
    collect_session_grouped,
    pick_best_feature_for_psid,
    metric_axis_label,
    metric_y_range,
)
from thesis_loaders import (
    discover_session_run,
    load_split_results_required,
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
# ## Figs 7-8: Pooled behavioural metric (velocity, acceleration) — 24-box grouped layout

# %%
# Figs 7-8 use the GROUPED layout: 2 DBS panels x 3 model groups x 4 session boxes.
# Predictions come from the dbs_both models; trials are split by stim into the two panels.
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
    panel_letter = chr(ord("A") + (fig_num - 7))
    if fig is not None:
        plt.show()
    print(
        f"Fig {fig_num}{panel_letter}: behavioural decoding for '{ch}' from dbs_both models.\n"
        f"  - layout: 2 panels (DBS-OFF | DBS-ON) x 3 model groups x 4 session boxes.\n"
        f"  - sessions: {', '.join(spec.sessions)}.\n"
        f"  - split: {spec.split} (three metric PNGs emitted: _rmse / _pearson / _vaf)."
    )
    fig_num += 1

# %% [markdown]
# ## Figs 9-16: Per-session metric strip plots (2 channels x 4 sessions)

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
            f"Fig {fig_num}: Per-trial behavioural prediction for participant {pid}, "
            f"session {sess[1:]}, output channel '{ch}'. Three metrics emitted as "
            f"separate PNGs; grouped box + per-trial dots; DBS-OFF / DBS-ON panels."
        )
        fig_num += 1

# %% [markdown]
# ## Figs 46-49: Per-session LFP (laplacian) reconstruction
#
# Mirror of Figs 9-16, but with LFP-band outputs instead of behavioural kinematic channels.
# The LFP band shown is the one PSID reconstructs best under the figure's metric.
# DPAD has no laplacian model — these are 2-model boxplots (PSID + VARMA).

# %%
fig_num = 46
for session in SESSIONS:
    _psid_var, _psid_ts = discover_session_run(results_root, "psid", EXP_NEURAL, session)
    if not _psid_ts:
        continue
    res_p = load_split_results_required(results_root, _psid_var, _psid_ts, "test")
    Z0 = res_p["Z"][0] if res_p.get("Z") else None
    n_bands = min(np.asarray(Z0).shape) if Z0 is not None else len(LAPLACIAN_BAND_NAMES)
    band_indices = list(range(min(n_bands, len(LAPLACIAN_BAND_NAMES))))

    def _collect(metric, _session=session, _res_p=res_p, _idxs=band_indices):
        best_idx = pick_best_feature_for_psid(_res_p, _idxs, metric, target="Z")
        return collect_pooled_rmse(
            results_root,
            [_session],
            EXP_NEURAL,
            best_idx,
            split="test",
            run_wilcoxon=False,
            metric=metric,
            include_dpad=False,
        )

    best_name_idx = pick_best_feature_for_psid(res_p, band_indices, "pearson", target="Z")
    band_stem = laplacian_band_label(best_name_idx).replace(".", "").replace("/", "_")

    fig = write_boxplot_three_metrics(
        collector=_collect,
        fig_num=fig_num,
        filename_stem=f"session_{session}_lfp_{band_stem}",
        target_label=f"{session} LFP",
        rng_seed=42,
    )
    if fig is not None:
        plt.show()
    picks = {
        m: laplacian_band_label(
            pick_best_feature_for_psid(res_p, band_indices, m, target="Z")
        )
        for m in ("rmse", "pearson", "vaf")
    }
    print(
        f"Fig {fig_num}: {session} LFP reconstruction -- per-metric best bands picked from PSID test split.\n"
        f"  RMSE-best: {picks['rmse']}  |  Pearson-best: {picks['pearson']}  |  VAF-best: {picks['vaf']}.\n"
        f"  2-model layout (PSID + VARMA; no DPAD laplacian). Three metric PNGs emitted."
    )
    fig_num += 1

# %% [markdown]
# ## Fig 17: Session-mean RMSE strip/box plots
#
# Per-participant box plots (trial-level RMSE) by model x DBS condition.
# Six columns: PSID OFF/ON, DPAD OFF/ON, VARMA OFF/ON — one panel per session.

# %%
# Inline builder: uses StripFigureData collector (pure-numpy) from thesis_lib.
# No reuse elsewhere after migration — kept local per the "inline if not
# reused" convention.
from thesis_lib.session_strip_rmse import collect_strip_figure_data

strip_spec = THESIS_STRIP_PANELS[0]
panel_triplets = [(e.panel_label, e.triplet) for e in strip_spec.panels]
strip_data = collect_strip_figure_data(
    results_root, panel_triplets, strip_spec.channel_idx, split=strip_spec.split
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

    # Vertical separators between model groups.
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
    ax_row[0].set_ylabel("RMSE [z]")

fig_17.savefig(str(OUT / "fig_017_strip_plots.png"))
plt.show()
print(
    "Fig 17: Per-session box-plus-strip of test-trial behavioural RMSE (z-scored, channel 0). "
    "Each panel is one session; columns are PSID/DPAD/VARMA x DBS OFF/ON. "
    f"Sessions: {', '.join(p.panel_label for p in strip_data.panels)}."
)
