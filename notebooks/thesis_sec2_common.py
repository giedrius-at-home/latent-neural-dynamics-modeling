"""Shared setup for the split sec2 notebooks (2a-2e).

Each sec2{a..e}.ipynb does `from thesis_sec2_common import *` as its first cell.
All constants / specs / triplets live here so edits propagate to every notebook.
"""

from __future__ import annotations

import sys
import os

os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")
sys.path.insert(0, ".")

from pathlib import Path
import numpy as np

OUT = Path("thesis_figures/sec2")
OUT.mkdir(parents=True, exist_ok=True)
results_root = Path("results").resolve()

from thesis_specs import (
    StripPanelEntry,
    ThesisTheme,
    ThesisAggregateRmseSpec,
    ThesisStripPanelsSpec,
    ThesisNeuralTimeseriesSpec,
    ThesisNeuralBandHeatmapSpec,
    ThesisForecastRmseSpec,
    ThesisC2ForecastSpec,
)
from thesis_loaders import (
    discover_session_run,
    EXP_BEHAVIORAL,
    EXP_NEURAL,
    SESSIONS,
    inspect_available_results,
)

# Laplacian LFP output-band order (column index in Z). Test parquets don't store
# output_channels, so we hardcode the 15 narrow bands in the order the trainer emits.
# Source: preprocessing/participants_at_200Hz_scaled_1e6_narrow_band.yaml + laplacian mode spec.
LAPLACIAN_BAND_NAMES = (
    "theta_4-8",
    "alpha_8-12",
    "beta_12-17",
    "beta_17-22",
    "beta_22-27",
    "beta_27-30",
    "gamma_30-35",
    "gamma_35-40",
    "gamma_40-45",
    "gamma_45-50",
    "gamma_50-55",
    "gamma_55-60",
    "gamma_60-65",
    "gamma_70-75",
    "gamma_75-80",
)


def laplacian_band_label(idx: int) -> str:
    """Readable LFP band name by Z-column index (fallback when output_channels is empty)."""
    if 0 <= idx < len(LAPLACIAN_BAND_NAMES):
        return LAPLACIAN_BAND_NAMES[idx]
    return f"band_{idx}"


# Spec lists
THESIS_AGGREGATE_FIGURES = [
    ThesisAggregateRmseSpec(
        section_title="Pooled RMSE ch0", channel_idx=0,
        sessions=list(SESSIONS), exp_type=EXP_BEHAVIORAL,
    ),
    ThesisAggregateRmseSpec(
        section_title="Pooled RMSE ch1", channel_idx=1,
        sessions=list(SESSIONS), exp_type=EXP_BEHAVIORAL,
    ),
]
THESIS_STRIP_PANELS = [
    ThesisStripPanelsSpec(
        section_title="Session-mean RMSE strip plots",
        channel_idx=0,
        exp_type=EXP_BEHAVIORAL,
        panels=[
            StripPanelEntry(panel_label="PDI1 S2", session="PDI1_S2"),
            StripPanelEntry(panel_label="PDI1 S4", session="PDI1_S4"),
            StripPanelEntry(panel_label="PDI4 S2", session="PDI4_S2"),
            StripPanelEntry(panel_label="PDI4 S3", session="PDI4_S3"),
        ],
    ),
]
_TRIAL_INDICES = {
    "PDI1_S2": (11, 27),
    "PDI1_S4": (17, 5),
    "PDI4_S2": (5, 18),
    "PDI4_S3": (9, 25),
}
THESIS_NEURAL_TIMESERIES = []
THESIS_C2_FORECASTS = []
for _session, (_off, _on) in _TRIAL_INDICES.items():
    _psid_var, _psid_ts = discover_session_run(results_root, "psid", EXP_BEHAVIORAL, _session)
    _dpad_var, _dpad_ts = discover_session_run(results_root, "dpad", EXP_BEHAVIORAL, _session)
    _varma_var, _varma_ts = discover_session_run(results_root, "varma", EXP_BEHAVIORAL, _session)
    _, _varma_ts_off = discover_session_run(results_root, "varma", EXP_BEHAVIORAL, _session, "dbs_off")
    _, _varma_ts_on = discover_session_run(results_root, "varma", EXP_BEHAVIORAL, _session, "dbs_on")
    if not _psid_ts or not _varma_ts:
        continue
    THESIS_NEURAL_TIMESERIES.append(ThesisNeuralTimeseriesSpec(
        section_title=_session.replace("_", " "),
        participant_label=_session.split("_")[0],
        psid_variant=_psid_var,
        dpad_variant=_dpad_var,
        varma_variant=_varma_var,
        psid_run_ts=_psid_ts,
        dpad_run_ts=_dpad_ts,
        varma_run_ts=_varma_ts,
        split="test",
        trial_idx_off=_off,
        trial_idx_on=_on,
        neural_y_channel_idx=5,
        neural_y_feature_name="ECOG_1_beta_27_30_raw",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        varma_run_ts_off=_varma_ts_off or None,
        varma_run_ts_on=_varma_ts_on or None,
    ))
    THESIS_C2_FORECASTS.append(ThesisC2ForecastSpec(
        section_title=_session,
        participant_label=_session.split("_")[0],
        psid_variant=_psid_var,
        dpad_variant=_dpad_var,
        varma_variant=_varma_var,
        psid_run_ts=_psid_ts,
        dpad_run_ts=_dpad_ts,
        varma_run_ts=_varma_ts,
        split="test",
        trial_idx_off=_off,
        trial_idx_on=_on,
        channel_idx=5,
        forecast_target="Y",
        neural_y_feature_name="ECOG_1_beta_27_30_raw",
        varma_run_ts_off=_varma_ts_off or None,
        varma_run_ts_on=_varma_ts_on or None,
    ))
THESIS_NEURAL_BAND_HEATMAPS = [
    ThesisNeuralBandHeatmapSpec(
        section_title="Neural band Pearson r",
        sessions=list(SESSIONS), exp_type=EXP_BEHAVIORAL,
        band_row_order=("Theta", "Alpha", "Beta"),
    ),
]
THESIS_NEURAL_FORECAST_FIGURES = [
    ThesisForecastRmseSpec(
        section_title="Neural forecast RMSE",
        channel_idx=5,
        sessions=list(SESSIONS), exp_type=EXP_BEHAVIORAL,
        forecast_target="Y",
        neural_y_feature_name="ECOG_1_beta_27_30_raw",
    ),
]

THESIS_DECLARED_BEHAVIORAL_OUTPUTS = (
    "tracing_velocity_x",
    "tracing_acceleration_magnitude",
)

# ---------------------------------------------------------------------------
# Matplotlib boxplot builders — shared across sec2b / sec2c / sec2d / model_validation.
#
# Consume the ``AggregateRmseData`` / ``SessionGroupedData`` dataclasses produced
# by ``thesis_utils`` collectors (which are pure-numpy and import-safe).
# ---------------------------------------------------------------------------
_METRIC_FILENAME_SUFFIX = {"rmse": "rmse", "pearson": "pearson", "vaf": "vaf"}


def _boxplot_colored(ax, vals, x_pos, color, *, fill_alpha=0.45, width=0.7):
    """One matplotlib box at ``x_pos`` with a filled body + coloured whiskers."""
    from matplotlib.colors import to_rgba

    face = (*to_rgba(color)[:3], fill_alpha)
    bp = ax.boxplot(
        [vals],
        positions=[x_pos],
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for box in bp["boxes"]:
        box.set(facecolor=face, edgecolor=color, linewidth=1.2)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set(color=color, linewidth=1.2)
    return bp


def mpl_rmse_boxplot(
    data,
    metric="rmse",
    target_label: str = "",
    *,
    figsize=(8.0, 3.8),
    jitter: float = 0.12,
    rng_seed: int = 42,
):
    """Two-panel box + strip: DBS-OFF | DBS-ON, one box per model, dots per participant.

    Matplotlib replacement for the legacy ``build_rmse_boxplot_figure``.
    Returns a single matplotlib Figure; caller handles ``savefig`` + ``plt.show``.
    """
    import matplotlib.pyplot as plt
    from thesis_style import COLOR_PSID, COLOR_DPAD, COLOR_VARMA, PARTICIPANT_COLORS
    from thesis_utils import metric_axis_label, metric_y_range

    rng = np.random.default_rng(rng_seed)
    show_dpad = getattr(data, "show_dpad", True)
    if show_dpad:
        models = ["PSID", "DPAD", "VARMA"]
        colors = [COLOR_PSID, COLOR_DPAD, COLOR_VARMA]
        off_cells, on_cells = [0, 2, 4], [1, 3, 5]
    else:
        models = ["PSID", "VARMA"]
        colors = [COLOR_PSID, COLOR_VARMA]
        off_cells, on_cells = [0, 4], [1, 5]

    cells_with_pid = getattr(data, "trial_rmse_with_participant", None)
    if cells_with_pid is None or not any(len(c) for c in cells_with_pid):
        cells_with_pid = tuple(
            [(float(v), "?") for v in data.trial_rmse[i] if np.isfinite(v)]
            for i in range(6)
        )

    def _vp(idx):
        rows = cells_with_pid[idx]
        vals = [r[0] for r in rows if np.isfinite(r[0])]
        pids = [str(r[1]) for r in rows if np.isfinite(r[0])]
        return vals, pids

    all_vals = []
    for idx in off_cells + on_cells:
        v, _ = _vp(idx)
        all_vals.extend(v)
    y_min, y_max = metric_y_range(metric, np.asarray(all_vals, dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    legend_pids = set()

    for ax, cond_cells, cond_title in zip(
        axes, (off_cells, on_cells), ("DBS-OFF", "DBS-ON")
    ):
        for mi, (model, col, cell_idx) in enumerate(zip(models, colors, cond_cells)):
            vals, pids = _vp(cell_idx)
            if not vals:
                continue
            _boxplot_colored(ax, vals, mi, col)
            jt = rng.uniform(-jitter, jitter, size=len(vals))
            dot_cols = [PARTICIPANT_COLORS.get(p, "#888888") for p in pids]
            ax.scatter(mi + jt, vals, s=10, c=dot_cols, linewidths=0)
            legend_pids.update(pids)
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models)
        ax.set_title(cond_title)
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel(metric_axis_label(metric))

    # Participant legend (single shared, placed by matplotlib).
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=PARTICIPANT_COLORS.get(p, "#888888"),
            markeredgecolor="none",
            label=p,
        )
        for p in sorted(legend_pids)
    ]
    if handles:
        axes[0].legend(handles=handles, title="participant")
    return fig


def mpl_session_grouped_boxplot(
    data,
    metric="rmse",
    target_label: str = "",
    *,
    figsize=(10.0, 4.6),
    jitter: float = 0.12,
    rng_seed: int = 42,
):
    """Two-panel 3-model × 4-session grouped boxplot (24 boxes). Matplotlib replacement
    for the plotly ``build_session_grouped_boxplot``. Dots coloured by model.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from thesis_style import (
        COLOR_PSID,
        COLOR_DPAD,
        COLOR_VARMA,
        panel_label as _panel_label,
    )
    from thesis_utils import metric_axis_label, metric_y_range

    rng = np.random.default_rng(rng_seed)
    models = list(data.models)
    color_map = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}
    model_colors = [color_map[m] for m in models]
    sessions = list(data.session_labels)
    n_sess = len(sessions)
    group_gap = 1
    xpos = {
        (m, s): mi * (n_sess + group_gap) + si
        for mi, m in enumerate(models)
        for si, s in enumerate(sessions)
    }

    all_vals = [
        v
        for cond in ("off", "on")
        for m in models
        for s in sessions
        for v in data.scores[cond][m][s]
        if np.isfinite(v)
    ]
    y_min, y_max = metric_y_range(metric, np.asarray(all_vals, dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, cond, letter, title in zip(
        axes, ("off", "on"), ("A", "B"), ("DBS-OFF", "DBS-ON")
    ):
        for m, col in zip(models, model_colors):
            for s in sessions:
                vals = [v for v in data.scores[cond][m][s] if np.isfinite(v)]
                if not vals:
                    continue
                xp = xpos[(m, s)]
                _boxplot_colored(ax, vals, xp, col, width=0.7)
                jt = rng.uniform(-jitter, jitter, size=len(vals))
                dot_col = (*to_rgba(col)[:3], 0.55)
                ax.scatter(
                    xp + jt, vals, s=6, color=[dot_col] * len(vals), linewidths=0
                )
        ax.set_xticks([xpos[(m, s)] for m in models for s in sessions])
        ax.set_xticklabels(
            [s for m in models for s in sessions], rotation=45, ha="right"
        )
        ax.set_ylim(y_min, y_max)
        _panel_label(ax, letter, title)
    axes[0].set_ylabel(metric_axis_label(metric))
    return fig


def write_boxplot_three_metrics(
    *,
    collector,
    fig_num: int,
    filename_stem: str,
    target_label: str,
    theme=ThesisTheme.LIGHT,
    rng_seed: int = 42,
    caption_header: str = "",
    metrics=None,
    builder=None,
):
    """Emit one PNG per metric for a boxplot scope.

    ``collector`` is ``Callable[[MetricKind], data]`` — caller closes over
    triplets / channel / split and re-runs the aggregation for each metric.

    ``builder`` is ``Callable[[data, metric, ...], matplotlib.Figure]``.
    Defaults to the pooled ``mpl_rmse_boxplot``; pass
    ``mpl_session_grouped_boxplot`` for the 24-box layout.

    Filenames: ``fig_{num:03d}_{stem}_{metric}.png``. Returns the first
    (``rmse``) figure so the caller can ``fig.savefig`` / ``plt.show`` once.
    """
    from thesis_utils import METRIC_KINDS, metric_display_name

    if metrics is None:
        metrics = METRIC_KINDS
    if builder is None:
        builder = mpl_rmse_boxplot

    first_fig = None
    for metric in metrics:
        try:
            agg = collector(metric)
        except Exception as e:
            print(
                f"  ! {metric_display_name(metric)} ({target_label}): collection failed: {e}"
            )
            continue
        fig = builder(agg, metric=metric, target_label=target_label, rng_seed=rng_seed)
        suffix = _METRIC_FILENAME_SUFFIX[metric]
        fig.savefig(str(OUT / f"fig_{fig_num:03d}_{filename_stem}_{suffix}.png"))
        if first_fig is None:
            first_fig = fig
    if caption_header:
        print(caption_header)
    return first_fig


# ---------------------------------------------------------------------------
# Group-plot expansion (2026-04-25): four matplotlib helpers that consume the
# same ``AggregateRmseData`` shape as ``mpl_rmse_boxplot``. They emphasise the
# DBS-OFF vs DBS-ON contrast (DBS = primary color, model = x-position) and
# expose chronological drift that pooled boxes hide.
#
# Cell-index convention (matches AggregateRmseData):
#   0=PSID off, 1=PSID on, 2=DPAD off, 3=DPAD on, 4=VARMA off, 5=VARMA on
# ---------------------------------------------------------------------------


def _model_layout(data):
    """Resolve (models, off_cells, on_cells) honouring ``data.show_dpad``.

    If show_dpad is set but DPAD cells are empty (e.g. forecast collector
    that doesn't populate DPAD), fall back to the 2-model layout so the
    figure doesn't render an empty DPAD panel."""
    show = getattr(data, "show_dpad", True)
    if show:
        rows = (
            data.trial_rmse_with_participant
            if (getattr(data, "trial_rmse_with_participant", None))
            else data.trial_rmse
        )
        dpad_filled = any(len(rows[i]) for i in (2, 3))
        if not dpad_filled:
            show = False
    if show:
        return ["PSID", "DPAD", "VARMA"], [0, 2, 4], [1, 3, 5]
    return ["PSID", "VARMA"], [0, 4], [1, 5]


def _vals_at(data, idx):
    """Finite values for one of the 6 cells, falling back to ``trial_rmse``."""
    rows = (
        data.trial_rmse_with_participant[idx]
        if (
            getattr(data, "trial_rmse_with_participant", None)
            and len(data.trial_rmse_with_participant[idx])
        )
        else [(float(v), "?") for v in data.trial_rmse[idx]]
    )
    return [r[0] for r in rows if np.isfinite(r[0])]


def mpl_raincloud(
    data,
    metric="rmse",
    target_label: str = "",
    *,
    figsize=(8.0, 3.8),
    jitter: float = 0.10,
    rng_seed: int = 42,
):
    """Half-violin + strip + median bar; one panel per model, two columns
    (DBS-OFF | DBS-ON). Honest distribution shape on small N."""
    import matplotlib.pyplot as plt
    from thesis_style import COLOR_DBS_OFF, COLOR_DBS_ON
    from thesis_utils import metric_axis_label, metric_y_range

    rng = np.random.default_rng(rng_seed)
    models, off_cells, on_cells = _model_layout(data)
    all_vals = []
    for cell_idx in off_cells + on_cells:
        all_vals.extend(_vals_at(data, cell_idx))
    y_min, y_max = metric_y_range(metric, np.asarray(all_vals, dtype=float))

    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model, off_i, on_i in zip(axes, models, off_cells, on_cells):
        for x_pos, vals, color in (
            (0, _vals_at(data, off_i), COLOR_DBS_OFF),
            (1, _vals_at(data, on_i), COLOR_DBS_ON),
        ):
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            # Half-violin (right side only of x_pos - 0.18).
            v = ax.violinplot(
                [arr],
                positions=[x_pos - 0.18],
                widths=0.55,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in v["bodies"]:
                paths = body.get_paths()[0]
                # Clip to the right half (>= x_pos - 0.18) to make it half-violin.
                paths.vertices[:, 0] = np.clip(
                    paths.vertices[:, 0], x_pos - 0.18, np.inf
                )
                body.set(facecolor=color, edgecolor=color, alpha=0.30, linewidth=0.8)
            # Strip dots offset to the right.
            jt = rng.uniform(-jitter, jitter, size=len(arr))
            ax.scatter(
                np.full_like(arr, x_pos + 0.10) + jt,
                arr,
                s=10,
                c=color,
                alpha=0.55,
                linewidths=0,
            )
            # Median bar.
            med = float(np.median(arr))
            ax.hlines(med, x_pos - 0.05, x_pos + 0.30, colors=color, linewidth=2.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["OFF", "ON"])
        ax.set_title(model)
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel(metric_axis_label(metric))
    return fig


def mpl_ecdf(
    data,
    metric="rmse",
    target_label: str = "",
    *,
    figsize=(8.0, 3.8),
    rng_seed: int = 42,
):
    """ECDF per (model × DBS) — tail comparison without box distortion.
    One panel per DBS condition; one line per model; OFF=blue, ON=red is
    encoded by panel; per-model color discriminates lines within each panel."""
    import matplotlib.pyplot as plt
    from thesis_style import COLOR_PSID, COLOR_DPAD, COLOR_VARMA
    from thesis_utils import metric_axis_label, metric_y_range

    models, off_cells, on_cells = _model_layout(data)
    color_map = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}
    all_vals = []
    for cell_idx in off_cells + on_cells:
        all_vals.extend(_vals_at(data, cell_idx))
    x_min, x_max = metric_y_range(metric, np.asarray(all_vals, dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for ax, cells, title in zip(axes, (off_cells, on_cells), ("DBS-OFF", "DBS-ON")):
        for model, cell_idx in zip(models, cells):
            vals = _vals_at(data, cell_idx)
            if not vals:
                continue
            arr = np.sort(np.asarray(vals, dtype=float))
            y = np.arange(1, len(arr) + 1) / len(arr)
            ax.step(
                arr,
                y,
                where="post",
                color=color_map[model],
                linewidth=1.5,
                label=f"{model} (n={len(arr)})",
            )
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel(metric_axis_label(metric))
    axes[0].set_ylabel("cumulative fraction")
    axes[0].legend()
    return fig


def mpl_block_running(
    data,
    metric="rmse",
    target_label: str = "",
    *,
    figsize=(9.0, 4.0),
    n_blocks: int = 12,
    rng_seed: int = 42,
):
    """Trial-order block running: per-trial scores binned into ``n_blocks``
    equal-count chronological bins; strip dots per bin + median curve. Reveals
    drift at block resolution. Two panels (OFF, ON), one curve per model."""
    import matplotlib.pyplot as plt
    from thesis_style import COLOR_PSID, COLOR_DPAD, COLOR_VARMA
    from thesis_utils import metric_axis_label, metric_y_range

    rng = np.random.default_rng(rng_seed)
    models, off_cells, on_cells = _model_layout(data)
    color_map = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}
    all_vals = []
    for cell_idx in off_cells + on_cells:
        all_vals.extend(_vals_at(data, cell_idx))
    y_min, y_max = metric_y_range(metric, np.asarray(all_vals, dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, cells, title in zip(axes, (off_cells, on_cells), ("DBS-OFF", "DBS-ON")):
        for model, cell_idx in zip(models, cells):
            vals = _vals_at(data, cell_idx)
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            n = len(arr)
            edges = np.linspace(0, n, n_blocks + 1).astype(int)
            centers, medians = [], []
            color = color_map[model]
            for b in range(n_blocks):
                lo, hi = edges[b], edges[b + 1]
                if hi <= lo:
                    continue
                bin_vals = arr[lo:hi]
                center = 0.5 * (lo + hi) / n
                centers.append(center)
                medians.append(float(np.median(bin_vals)))
                jt = rng.uniform(-0.012, 0.012, size=len(bin_vals))
                ax.scatter(
                    np.full_like(bin_vals, center) + jt,
                    bin_vals,
                    s=8,
                    color=color,
                    alpha=0.30,
                    linewidths=0,
                )
            if centers:
                ax.plot(
                    centers,
                    medians,
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=4,
                    label=model,
                )
        ax.set_xlim(0, 1)
        ax.set_xlabel("trial position (chronological, normalised)")
        ax.set_title(title)
        ax.set_ylim(y_min, y_max)
    axes[0].set_ylabel(metric_axis_label(metric))
    axes[0].legend()
    return fig


def mpl_cascade_train_val_test(
    data_train,
    data_val,
    data_test,
    metric="rmse",
    target_label: str = "",
    *,
    figsize=(9.0, 4.0),
    jitter: float = 0.06,
    rng_seed: int = 42,
):
    """Train→val→test cascade with paired median arrows. Two panels
    (DBS-OFF, DBS-ON); per panel x=[train, val, test] × models. Strip dots
    + median bar per cell; arrows connect medians across splits per model.
    The train→test gap IS the chronological-drift signature."""
    import matplotlib.pyplot as plt
    from thesis_style import COLOR_PSID, COLOR_DPAD, COLOR_VARMA
    from thesis_utils import metric_axis_label, metric_y_range

    rng = np.random.default_rng(rng_seed)
    splits = [("train", data_train), ("val", data_val), ("test", data_test)]
    # Use the test data layout as canonical (same triplets used for all splits).
    models, off_cells, on_cells = _model_layout(data_test)
    color_map = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}

    all_vals = []
    for _, d in splits:
        if d is None:
            continue
        for cell_idx in off_cells + on_cells:
            all_vals.extend(_vals_at(d, cell_idx))
    y_min, y_max = metric_y_range(metric, np.asarray(all_vals, dtype=float))

    n_models = len(models)
    n_splits = len(splits)
    group_gap = 1
    # x position: model_index * (n_splits + group_gap) + split_index
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, cells, title in zip(axes, (off_cells, on_cells), ("DBS-OFF", "DBS-ON")):
        xticks, xlabels = [], []
        for mi, (model, cell_idx) in enumerate(zip(models, cells)):
            color = color_map[model]
            split_medians = []
            for si, (split_name, d) in enumerate(splits):
                xp = mi * (n_splits + group_gap) + si
                xticks.append(xp)
                xlabels.append(split_name)
                if d is None:
                    continue
                vals = _vals_at(d, cell_idx)
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=float)
                jt = rng.uniform(-jitter, jitter, size=len(arr))
                ax.scatter(
                    np.full_like(arr, xp) + jt,
                    arr,
                    s=8,
                    color=color,
                    alpha=0.30,
                    linewidths=0,
                )
                med = float(np.median(arr))
                ax.hlines(med, xp - 0.30, xp + 0.30, colors=color, linewidth=2.0)
                split_medians.append((xp, med))
            # Paired-median arrow: train→val→test for this model.
            for (x0, y0), (x1, y1) in zip(split_medians[:-1], split_medians[1:]):
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.8),
                )
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylim(y_min, y_max)
        # Model band labels above the model groups.
        for mi, model in enumerate(models):
            x_center = mi * (n_splits + group_gap) + (n_splits - 1) / 2
            ax.text(
                x_center,
                1.02,
                model,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
                color=color_map[model],
            )
    axes[0].set_ylabel(metric_axis_label(metric))
    return fig


def mpl_forest_per_cell_model(
    per_cell_data: dict,
    cells: list[str],
    models: list[str] = ("PSID", "DPAD", "VARMA"),
    metric: str = "pearson",
    *,
    figsize=(8.5, 5.0),
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
):
    """A1 forest plot — per-cell × model with bootstrap 95% CI; pooled rows on top.

    ``per_cell_data``: dict[(cell, model, dbs)] -> 1-D np.ndarray of trial-level
    metric values. Missing entries (key absent OR empty array) render as a gray
    `x` at axis center (no CI bar) so the row stays in the layout.

    Two side-by-side panels (DBS-OFF, DBS-ON). Pooled = inverse-variance
    weighted mean across cells per model with CI from `1/sqrt(Σ wᵢ)`.
    """
    import matplotlib.pyplot as plt
    from thesis_style import COLOR_PSID, COLOR_DPAD, COLOR_VARMA

    color_map = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}
    marker_map = {"PSID": "o", "DPAD": "s", "VARMA": "^"}
    rng = np.random.default_rng(rng_seed)

    def _ci(vals: np.ndarray) -> tuple[float, float, float]:
        boot = rng.choice(vals, size=(n_bootstrap, len(vals)), replace=True).mean(
            axis=1
        )
        return (
            float(vals.mean()),
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, dbs, title in zip(axes, ("off", "on"), ("DBS-OFF", "DBS-ON")):
        ytick_pos, ytick_lab = [], []
        y = 0.0
        # Pooled rows first (top of plot once y is inverted).
        for model in models:
            cell_means, cell_ses = [], []
            for c in cells:
                v = per_cell_data.get((c, model, dbs))
                if v is None or len(v) == 0:
                    continue
                v = np.asarray(v, float)
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                cell_means.append(v.mean())
                cell_ses.append(
                    v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
                )
            if cell_means:
                w = np.array(
                    [1.0 / s**2 if np.isfinite(s) and s > 0 else 1.0 for s in cell_ses]
                )
                pooled = float(np.average(cell_means, weights=w))
                pooled_se = float(1.0 / np.sqrt(w.sum()))
                ax.errorbar(
                    pooled,
                    y,
                    xerr=1.96 * pooled_se,
                    fmt=marker_map[model],
                    color=color_map[model],
                    markersize=10,
                    capsize=3,
                    lw=1.8,
                    markeredgecolor="black",
                    markeredgewidth=0.6,
                    linestyle="--",
                )
            ytick_pos.append(y)
            ytick_lab.append(f"POOLED {model}")
            y += 1.0
        y += 0.6  # gap to per-cell rows
        # Per-cell × model rows.
        for c in cells:
            for model in models:
                v = per_cell_data.get((c, model, dbs))
                if v is None or len(v) == 0:
                    ax.plot(
                        0.5 if metric == "pearson" else 0.0,
                        y,
                        "x",
                        color="#888888",
                        markersize=8,
                    )
                else:
                    v = np.asarray(v, float)
                    v = v[np.isfinite(v)]
                    if len(v) == 0:
                        ax.plot(
                            0.5 if metric == "pearson" else 0.0,
                            y,
                            "x",
                            color="#888888",
                            markersize=8,
                        )
                    else:
                        m, lo, hi = _ci(v)
                        ax.errorbar(
                            m,
                            y,
                            xerr=[[m - lo], [hi - m]],
                            fmt=marker_map[model],
                            color=color_map[model],
                            markersize=6,
                            capsize=2,
                            lw=1.2,
                        )
                ytick_pos.append(y)
                ytick_lab.append(f"{c} {model}")
                y += 1.0
            y += 0.4  # gap between cells

        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_lab, fontsize=7)
        ax.set_title(title)
        ax.invert_yaxis()
        if metric == "pearson":
            ax.axvline(0.0, color="#888888", lw=0.6, linestyle=":")
            ax.set_xlim(-0.1, 1.0)
            ax.set_xlabel("Pearson r")
        else:
            ax.set_xlabel(metric)

    return fig


# ───────────────────────────────────────────────────────────────────────
# Vertical paired raincloud helpers (used by sec2c, sec2d).
#
# Layout per group slot (one framework / one cell): the two compared groups
# (Y vs Z, or DBS-OFF vs DBS-ON) are mirrored about the slot center. From
# slot center outward each side stacks: strip dots → boxplot → half-violin.
# Components do not overlap. Vertical orientation: metric on y-axis.
# ───────────────────────────────────────────────────────────────────────


def kde_half_violin_vert(
    ax,
    vals,
    *,
    x_anchor,
    color,
    alpha,
    hatch=None,
    logy=False,
    bw=0.55,
    width=0.18,
    pad=1.0,
    npts=400,
    side="right",
):
    """Vertical half-violin grown from `x_anchor` outward by up to `width`."""
    import matplotlib.pyplot as plt  # noqa: F401
    from scipy.stats import gaussian_kde

    y_in = np.log10(vals) if logy else np.asarray(vals, dtype=float)
    if y_in.size < 2 or np.allclose(y_in.std(), 0.0):
        return
    kde = gaussian_kde(y_in, bw_method=bw)
    s = y_in.std()
    ys = np.linspace(y_in.min() - pad * s, y_in.max() + pad * s, npts)
    dens = kde(ys)
    if dens.max() <= 0:
        return
    dens = dens / dens.max() * width
    ys_plot = np.power(10.0, ys) if logy else ys
    if side == "right":
        xs_l = np.full_like(dens, x_anchor)
        xs_r = x_anchor + dens
    else:
        xs_l = x_anchor - dens
        xs_r = np.full_like(dens, x_anchor)
    ax.fill_betweenx(
        ys_plot,
        xs_l,
        xs_r,
        facecolor=color,
        alpha=alpha,
        edgecolor=(color if hatch else "none"),
        linewidth=(0.6 if hatch else 0.0),
        hatch=hatch,
        zorder=2,
    )


def mpl_per_cell_yz_box(
    data_y_r,
    data_y_nrmse,
    data_z_r,
    data_z_nrmse,
    cells,
    *,
    models=("PSID", "DPAD", "VARMA"),
    mode_label="",
):
    """4 metric rows × N cells. Box hue = framework; DBS-OFF lighter, DBS-ON
    hatched darker. Mirrors pooled-raincloud Y/Z encoding convention."""
    import matplotlib.pyplot as plt
    from thesis_style import COLOR_PSID, COLOR_DPAD, COLOR_VARMA, panel_label

    model_color = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}
    dbs_alpha = {"off": 0.30, "on": 0.85}
    dbs_hatch = {"off": None, "on": "////"}
    n_cells = len(cells)
    fig, axes = plt.subplots(4, n_cells, figsize=(2.6 * n_cells, 8.4), sharex=True)
    if n_cells == 1:
        axes = axes.reshape(4, 1)
    rows = [
        (data_y_r, "Y  Pearson r"),
        (data_y_nrmse, "Y  NRMSE"),
        (data_z_r, "Z  Pearson r"),
        (data_z_nrmse, "Z  NRMSE"),
    ]
    panel_letters = "ABCD"
    for col, cell in enumerate(cells):
        for row, (data, ylabel) in enumerate(rows):
            ax = axes[row, col]
            for mi, m in enumerate(models):
                color = model_color[m]
                for di, d in enumerate(("off", "on")):
                    vals = data.get((cell, m, d))
                    if vals is None or len(vals) == 0:
                        continue
                    a = dbs_alpha[d]
                    hatch = dbs_hatch[d]
                    xp = mi + (di - 0.5) * 0.36
                    box_kwargs = dict(
                        facecolor=color, alpha=a, edgecolor=color, linewidth=0.8
                    )
                    if hatch is not None:
                        box_kwargs["hatch"] = hatch
                    ax.boxplot(
                        [vals],
                        positions=[xp],
                        widths=0.32,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=box_kwargs,
                        medianprops=dict(color=color, linewidth=1.2),
                        whiskerprops=dict(color=color, linewidth=0.8),
                        capprops=dict(color=color, linewidth=0.8),
                    )
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models if row == 3 else [])
            ax.set_xlim(-0.6, len(models) - 0.4)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                panel_label(ax, panel_letters[col], cell)
    if mode_label:
        fig.suptitle(mode_label, fontsize=10, y=0.995)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#888888", alpha=0.30, edgecolor="none"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#888888",
            alpha=0.85,
            edgecolor="#555555",
            linewidth=0.6,
            hatch="////",
        ),
    ]
    fig.legend(
        handles,
        ["DBS-OFF", "DBS-ON"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.subplots_adjust(bottom=0.12)
    return fig


def _raincloud_paired_layout(
    ax, mi, vals, *, color, alpha, hatch, side, logy, rng, strip_cap, kde_width=0.18
):
    """Render one (slot mi, side) raincloud component. Lays out from slot
    center outward: strip → box → violin. No overlap."""
    strip_x = mi - 0.08 if side == "left" else mi + 0.08
    box_x = mi - 0.20 if side == "left" else mi + 0.20
    anchor = mi - 0.30 if side == "left" else mi + 0.30
    kde_half_violin_vert(
        ax,
        vals,
        x_anchor=anchor,
        color=color,
        alpha=alpha,
        hatch=hatch,
        logy=logy,
        bw=0.55,
        width=kde_width,
        side=side,
    )
    box_kwargs = dict(facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.0)
    if hatch is not None:
        box_kwargs["hatch"] = hatch
    ax.boxplot(
        [vals],
        positions=[box_x],
        widths=0.08,
        vert=True,
        patch_artist=True,
        showfliers=False,
        boxprops=box_kwargs,
        medianprops=dict(color=color, linewidth=1.4),
        whiskerprops=dict(color=color, linewidth=1.0),
        capprops=dict(color=color, linewidth=1.0),
    )
    if len(vals) > strip_cap:
        idx = rng.choice(len(vals), size=strip_cap, replace=False)
        sv = vals[idx]
    else:
        sv = vals
    jx = rng.uniform(-0.04, 0.04, size=len(sv))
    ax.scatter(strip_x + jx, sv, c=color, alpha=min(alpha, 0.35), s=7, linewidths=0)


def mpl_raincloud_dbs_pair_vert(
    pool_r,
    pool_n,
    *,
    models=("PSID", "DPAD", "VARMA"),
    rng_seed=42,
    strip_cap=80,
    figsize=(7.0, 9.0),
):
    """Vertical raincloud, 2 stacked panels (top=r, bottom=NRMSE log).
    Per framework slot: DBS-OFF mirrored left, DBS-ON mirrored right
    (DBS palette colors, both solid)."""
    import matplotlib.pyplot as plt
    from thesis_style import COLOR_DBS_OFF, COLOR_DBS_ON, panel_label

    rng = np.random.default_rng(rng_seed)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    panels = [
        (axes[0], pool_r, "Pearson r", False, None, "A", "Pearson r"),
        (axes[1], pool_n, "NRMSE (log)", True, 1.0, "B", "NRMSE"),
    ]
    dbs_color = {"off": COLOR_DBS_OFF, "on": COLOR_DBS_ON}
    dbs_side = {"off": "left", "on": "right"}
    for ax, data, ylabel, logy, chance_y, plabel, ptitle in panels:
        panel_label(ax, plabel, ptitle)
        if logy:
            ax.set_yscale("log")
        if chance_y is not None:
            ax.axhline(
                chance_y,
                color="black",
                linestyle=":",
                linewidth=0.6,
                alpha=0.30,
                zorder=0,
            )
        for mi, m in enumerate(models):
            for d in ("off", "on"):
                vals = data.get((m, d))
                if vals is None or len(vals) == 0:
                    continue
                if logy:
                    vals = vals[vals > 0]
                    if len(vals) == 0:
                        continue
                _raincloud_paired_layout(
                    ax,
                    mi,
                    vals,
                    color=dbs_color[d],
                    alpha=0.45,
                    hatch=None,
                    side=dbs_side[d],
                    logy=logy,
                    rng=rng,
                    strip_cap=strip_cap,
                )
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_xlim(-0.55, len(models) - 0.45)
        ax.set_ylabel(ylabel)
    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLOR_DBS_OFF,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.6,
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLOR_DBS_ON,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.6,
        ),
    ]
    fig.legend(
        handles,
        ["DBS-OFF", "DBS-ON"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.subplots_adjust(bottom=0.10)
    return fig


def mpl_raincloud_yz_pair_vert(
    pool_r,
    pool_n,
    *,
    models=("PSID", "DPAD", "VARMA"),
    rng_seed=42,
    strip_cap=80,
    figsize=(7.0, 9.0),
):
    """Vertical raincloud, 2 stacked panels (top=r, bottom=NRMSE log).
    Per framework slot: Y mirrored left (solid lighter), Z mirrored right
    (hatched darker). Slot color = framework hue."""
    import matplotlib.pyplot as plt
    from thesis_style import COLOR_PSID, COLOR_DPAD, COLOR_VARMA, panel_label

    model_color = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}
    target_alpha = {"Y": 0.30, "Z": 0.85}
    target_hatch = {"Y": None, "Z": "////"}
    target_side = {"Y": "left", "Z": "right"}
    rng = np.random.default_rng(rng_seed)
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    panels = [
        (axes[0], pool_r, "Pearson r", False, None, "A", "Pearson r"),
        (axes[1], pool_n, "NRMSE (log)", True, 1.0, "B", "NRMSE"),
    ]
    for ax, data, ylabel, logy, chance_y, plabel, ptitle in panels:
        panel_label(ax, plabel, ptitle)
        if logy:
            ax.set_yscale("log")
        if chance_y is not None:
            ax.axhline(
                chance_y,
                color="black",
                linestyle=":",
                linewidth=0.6,
                alpha=0.30,
                zorder=0,
            )
        for mi, m in enumerate(models):
            color = model_color[m]
            for t in ("Y", "Z"):
                vals = data.get((m, t))
                if vals is None or len(vals) == 0:
                    continue
                if logy:
                    vals = vals[vals > 0]
                    if len(vals) == 0:
                        continue
                _raincloud_paired_layout(
                    ax,
                    mi,
                    vals,
                    color=color,
                    alpha=target_alpha[t],
                    hatch=target_hatch[t],
                    side=target_side[t],
                    logy=logy,
                    rng=rng,
                    strip_cap=strip_cap,
                )
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_xlim(-0.55, len(models) - 0.45)
        ax.set_ylabel(ylabel)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#888888", alpha=0.30, edgecolor="none"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#888888",
            alpha=0.85,
            edgecolor="#555555",
            linewidth=0.6,
            hatch="////",
        ),
    ]
    fig.legend(
        handles,
        ["Y (neural self-recon)", "Z (target decoding)"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.subplots_adjust(bottom=0.10)
    return fig


def pool_dbs_cells(per_cell_data, cells, *, models=("PSID", "DPAD", "VARMA")):
    """dict[(cell, model, dbs)] -> dict[(model, dbs)] pooled across cells."""
    out: dict = {}
    for m in models:
        for d in ("off", "on"):
            buf: list[float] = []
            for cell in cells:
                arr = per_cell_data.get((cell, m, d))
                if arr is None:
                    continue
                buf.extend([float(v) for v in arr if np.isfinite(v)])
            out[(m, d)] = np.array(buf, dtype=float)
    return out


def pool_yz_dbs_cells(
    per_cell_y, per_cell_z, cells, *, models=("PSID", "DPAD", "VARMA")
):
    """Pool Y and Z dicts across cells AND DBS into dict[(model, target)]."""
    out: dict = {}
    for m in models:
        for target_name, source in (("Y", per_cell_y), ("Z", per_cell_z)):
            buf: list[float] = []
            for c in cells:
                for d in ("off", "on"):
                    arr = source.get((c, m, d))
                    if arr is None:
                        continue
                    buf.extend([float(v) for v in arr if np.isfinite(v)])
            out[(m, target_name)] = np.array(buf, dtype=float)
    return out
