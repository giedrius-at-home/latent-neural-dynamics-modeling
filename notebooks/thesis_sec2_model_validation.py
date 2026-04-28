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
# # Section 2: Model Validation — 37 figures
#
# | # | Figure | Builder | Count |
# |---|--------|--------|-------|
# | 7-8 | Pooled behavioral prediction RMSE (velocity, acceleration) | `build_rmse_boxplot_figure()` | 2 |
# | 9-16 | Per-session behavioral prediction RMSE | `build_rmse_boxplot_figure()` | 8 |
# | 17 | Session-mean RMSE box+strip | inline | 1 |
# | 18-21 | Neural reconstruction time series | `compose_thesis_neural_figure()` | 4 |
# | 22 | Neural band Pearson r heatmap | inline | 1 |
# | 23 | Neural forecast RMSE vs horizon | `build_forecast_rmse_figure_or_empty()` | 1 |
# | 24 | Pooled neural forecast RMSE | `build_rmse_boxplot_figure()` | 1 |
# | 25-28 | Per-session neural forecast RMSE | `build_rmse_boxplot_figure()` | 4 |
# | 29-36 | Neural forecast exemplars (best trial per condition, split panels) | inline | 8 |
# | 37 | Vanilla vs RTS + A regularization PSID | inline | 1 |
# | 38 | PSID grid search BA heatmap (model selection) | inline | 1 |
# | 39 | Pooled neural reconstruction RMSE | `build_rmse_boxplot_figure()` | 1 |
# | 40-43 | Per-session neural reconstruction RMSE | `build_rmse_boxplot_figure()` | 4 |

# %%
import sys, os
os.chdir('/home/bobby/repos/latent-neural-dynamics-modeling')
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = Path('thesis_figures/sec2'); OUT.mkdir(parents=True, exist_ok=True)
results_root = Path('results').resolve()

from dashboard.thesis.specs import (
    AlignedTriplet, StripPanelEntry,
    ThesisAggregateRmseSpec, ThesisStripPanelsSpec,
    ThesisNeuralTimeseriesSpec, ThesisNeuralBandHeatmapSpec,
    ThesisForecastRmseSpec, ThesisC2ForecastSpec,
)
from notebooks.thesis_style import (
    COLOR_DBS_OFF, COLOR_DBS_ON, COLOR_PSID, COLOR_DPAD, COLOR_VARMA, COLOR_SEPARATOR,
    COLOR_PSID_BAND_FILL, COLOR_PSID_BAND_LINE,
    COLOR_DPAD_BAND_FILL, COLOR_DPAD_BAND_LINE,
    COLOR_VARMA_BAND_FILL, COLOR_VARMA_BAND_LINE,
    WIDTH_TRUE, WIDTH_PSID, WIDTH_DPAD, WIDTH_VARMA,
    OPACITY_PSID, OPACITY_DPAD, OPACITY_VARMA,
    FIGURE_HEIGHT, FONT_FAMILY, FONT_SIZE_ANNOTATION,
    FONT_SIZE_BASE, FONT_SIZE_LABEL, FONT_SIZE_TICK,
    ThesisTheme, apply_thesis_style, grid_color, paper_colors, true_line_color, rmse_axis_label,
    legend_bgcolor, dbs_badge_style,
    apply_paper_style, panel_label, add_freq_band_highlight, hex_to_rgba,
    COLOR_AXIS, LINE_WIDTH_MEAN, LINE_WIDTH_SD, LINE_WIDTH_REF,
)

# %%
# ---------------------------------------------------------------------------
# 4 session triplets (inline config)
# ---------------------------------------------------------------------------
TRIPLET_PDI1_S2 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI1_2_nx_25_n2_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_222003",
    dpad_variant="dpad_behavioral_PDI1_2_nx_25_n2_e3000_dbs_both_200Hz_narrow_band",
    dpad_run_ts="",
    varma_variant="varma_behavioral_PDI1_2_p30_q1_top20_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260409_105705",
    label="PDI1_S2",
    psid_run_ts_off="20260408_224606", psid_run_ts_on="20260408_223912",
    dpad_run_ts_off=None, dpad_run_ts_on=None,
    varma_run_ts_off="20260409_113230", varma_run_ts_on="20260409_112938",
    varma_run_ts_eval_off="20260409_110048", varma_run_ts_eval_on="20260409_110433",
)
TRIPLET_PDI1_S4 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI1_4_nx_15_n2_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_194919",
    dpad_variant="dpad_behavioral_PDI1_4_nx_15_n2_e3000_dbs_both_200Hz_narrow_band",
    dpad_run_ts="",
    varma_variant="varma_behavioral_PDI1_4_p30_q1_top20_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260409_104823",
    label="PDI1_S4",
    psid_run_ts_off="20260408_200652", psid_run_ts_on="20260408_200052",
    dpad_run_ts_off=None, dpad_run_ts_on=None,
    varma_run_ts_off="20260409_112734", varma_run_ts_on="20260409_112612",
    varma_run_ts_eval_off="20260409_105059", varma_run_ts_eval_on="20260409_105339",
)
TRIPLET_PDI4_S2 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_162132",
    dpad_variant="dpad_behavioral_PDI4_2_nx_30_n6_e3000_dbs_both_200Hz_narrow_band",
    dpad_run_ts="",
    varma_variant="varma_behavioral_PDI4_2_p30_q1_top20_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260409_111451",
    label="PDI4_S2",
    psid_run_ts_off="20260408_164031", psid_run_ts_on="20260408_163407",
    dpad_run_ts_off=None, dpad_run_ts_on=None,
    varma_run_ts_off="20260409_111913", varma_run_ts_on="20260409_111754",
    varma_run_ts_eval_off="20260409_111754", varma_run_ts_eval_on="20260409_111913",
)
TRIPLET_PDI4_S3 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_185522",
    dpad_variant="dpad_behavioral_PDI4_3_nx_25_n6_e3000_dbs_both_200Hz_narrow_band",
    dpad_run_ts="",
    varma_variant="varma_behavioral_PDI4_3_p30_q1_top20_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260409_110921",
    label="PDI4_S3",
    psid_run_ts_off="20260408_191423", psid_run_ts_on="20260408_190749",
    dpad_run_ts_off=None, dpad_run_ts_on=None,
    varma_run_ts_off="20260409_111318", varma_run_ts_on="20260409_111147",
    varma_run_ts_eval_off="20260409_111147", varma_run_ts_eval_on="20260409_111318",
)
ALL_TRIPLETS = [TRIPLET_PDI1_S2, TRIPLET_PDI1_S4, TRIPLET_PDI4_S2, TRIPLET_PDI4_S3]

# Spec lists
THESIS_AGGREGATE_FIGURES = [
    ThesisAggregateRmseSpec(section_title="Pooled RMSE ch0", channel_idx=0, triplets=ALL_TRIPLETS),
    ThesisAggregateRmseSpec(section_title="Pooled RMSE ch1", channel_idx=1, triplets=ALL_TRIPLETS),
]
THESIS_STRIP_PANELS = [
    ThesisStripPanelsSpec(
        section_title="Session-mean RMSE strip plots", channel_idx=0,
        panels=[
            StripPanelEntry(panel_label="PDI1 S2", triplet=TRIPLET_PDI1_S2),
            StripPanelEntry(panel_label="PDI1 S4", triplet=TRIPLET_PDI1_S4),
            StripPanelEntry(panel_label="PDI4 S2", triplet=TRIPLET_PDI4_S2),
            StripPanelEntry(panel_label="PDI4 S3", triplet=TRIPLET_PDI4_S3),
        ],
    ),
]
THESIS_NEURAL_TIMESERIES = [
    ThesisNeuralTimeseriesSpec(
        section_title=tri.label.replace("_", " "), participant_label=tri.label.split("_")[0],
        psid_variant=tri.psid_variant, dpad_variant=tri.dpad_variant, varma_variant=tri.varma_variant,
        psid_run_ts=tri.psid_run_ts, dpad_run_ts=tri.dpad_run_ts, varma_run_ts=tri.varma_run_ts,
        split="test", trial_idx_off=off, trial_idx_on=on, neural_y_channel_idx=0,
        neural_y_feature_name="ECOG_1_theta_4_8_raw", use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        varma_run_ts_off=tri.varma_run_ts_off, varma_run_ts_on=tri.varma_run_ts_on,
    )
    for tri, (off, on) in zip(ALL_TRIPLETS, [(11,27), (17,5), (5,18), (9,25)])
]
THESIS_NEURAL_BAND_HEATMAPS = [
    ThesisNeuralBandHeatmapSpec(
        section_title="Neural band Pearson r", triplets=ALL_TRIPLETS,
        band_row_order=("Theta", "Alpha", "Beta"),  # no Delta at 200 Hz
    ),
]
THESIS_NEURAL_FORECAST_FIGURES = [
    ThesisForecastRmseSpec(
        section_title="Neural forecast RMSE", channel_idx=0, triplets=ALL_TRIPLETS,
        forecast_target="Y", neural_y_feature_name="ECOG_1_theta_4_8_raw",
    ),
]
_TRIAL_INDICES = {"PDI1_S2": (11,27), "PDI1_S4": (17,5), "PDI4_S2": (5,18), "PDI4_S3": (9,25)}
THESIS_C2_FORECASTS = [
    ThesisC2ForecastSpec(
        section_title=tri.label, participant_label=tri.label.split("_")[0],
        psid_variant=tri.psid_variant, dpad_variant=tri.dpad_variant, varma_variant=tri.varma_variant,
        psid_run_ts=tri.psid_run_ts, dpad_run_ts=tri.dpad_run_ts, varma_run_ts=tri.varma_run_ts,
        split="test", trial_idx_off=_TRIAL_INDICES[tri.label][0], trial_idx_on=_TRIAL_INDICES[tri.label][1],
        channel_idx=0, forecast_target="Y", neural_y_feature_name="ECOG_1_theta_4_8_raw",
        varma_run_ts_off=tri.varma_run_ts_off, varma_run_ts_on=tri.varma_run_ts_on,
    )
    for tri in ALL_TRIPLETS
]

THESIS_DECLARED_BEHAVIORAL_OUTPUTS = ("tracing_velocity_x", "tracing_acceleration_magnitude")

# %% [markdown]
# ## Figs 7-8: Pooled behavioral RMSE (velocity, acceleration)

# %%
from dashboard.thesis.aggregate_rmse import collect_pooled_rmse, p_to_stars
from dashboard.thesis.loaders import (
    load_split_results_required, resolve_output_channel_display,
    channels_as_str_list, neural_y_feature_label, resolve_neural_y_channel_idx,
)
from dashboard.thesis.rmse_distribution_figure import (
    build_rmse_boxplot_figure as _dashboard_build_rmse_boxplot_figure,
)


def build_rmse_boxplot_figure(*args, **kwargs):
    """Paper-style wrapper around the dashboard box+strip builder."""
    fig = _dashboard_build_rmse_boxplot_figure(*args, **kwargs)
    apply_paper_style(
        fig,
        height=420,
        margin=dict(l=72, r=32, t=28, b=140),
        legend_y=-0.2,
    )
    return fig

# --- Inlined: build_rmse_distribution_figure (lifted from dashboard/thesis/rmse_distribution_figure.py) ---
_ALPHA_OFF = 0.80
_ALPHA_ON = 0.40
_DOT_ALPHA_OFF = 0.60
_DOT_ALPHA_ON = 0.35
_X_POS = np.arange(6, dtype=float)
_CATEGORY_LABELS = [
    "PSID_OFF", "PSID_ON", "DPAD_OFF", "DPAD_ON", "VARMA_OFF", "VARMA_ON",
]


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _model_color_for_index(i):
    if i < 2:
        return COLOR_PSID
    if i < 4:
        return COLOR_DPAD
    return COLOR_VARMA


def _is_on_cell(i):
    return i % 2 == 1


def build_rmse_distribution_figure(
    data, theme, rng, jitter=0.12, show_brackets=True,
    y_axis_label=None, show_dots=True, show_bars=True,
):
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)
    y_label = y_axis_label or "RMSE(z)"

    means = np.array(data.means, dtype=float)
    sems = np.array(data.sems, dtype=float)

    bar_colors = [
        _hex_to_rgba(
            _model_color_for_index(i),
            _ALPHA_ON if _is_on_cell(i) else _ALPHA_OFF,
        )
        for i in range(6)
    ]

    fig = go.Figure()

    if show_bars:
        fig.add_trace(
            go.Bar(
                x=_X_POS,
                y=means,
                error_y=dict(
                    type="data", array=sems, visible=True,
                    thickness=1.5, color=fg,
                ),
                marker=dict(color=bar_colors, line=dict(width=0)),
                width=0.52, name="Mean", showlegend=True,
            )
        )

    if show_dots:
        dot_alpha_off = _DOT_ALPHA_OFF if show_bars else 0.85
        dot_alpha_on = _DOT_ALPHA_ON if show_bars else 0.55
        for i in range(6):
            pts = data.trial_rmse[i]
            if not pts:
                continue
            jt = rng.uniform(-jitter, jitter, size=len(pts))
            alpha = dot_alpha_on if _is_on_cell(i) else dot_alpha_off
            fig.add_trace(
                go.Scatter(
                    x=_X_POS[i] + jt,
                    y=pts,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=_hex_to_rgba(_model_color_for_index(i), alpha * 0.95),
                        line=dict(width=0),
                    ),
                    name="One dot = one test trial RMSE" if i == 0 else None,
                    showlegend=i == 0,
                    legendgroup="dots",
                )
            )

    ymax_data = 0.0
    for i in range(6):
        if np.isfinite(means[i]):
            ymax_data = max(ymax_data, float(means[i] + (sems[i] if np.isfinite(sems[i]) else 0)))
    if show_dots:
        all_pts = []
        for i in range(6):
            for v in data.trial_rmse[i]:
                if np.isfinite(v):
                    all_pts.append(float(v))
        if all_pts:
            ymax_data = max(ymax_data, float(np.percentile(all_pts, 98)))

    y_span = max(ymax_data * 0.12, 0.04)
    bracket_y = []
    if show_brackets and data.wilcoxon:
        w = data.wilcoxon
        y0 = ymax_data + y_span * 0.25
        step = y_span * 1.2
        pairs = [
            (0, 4, w.psid_vs_varma_off_p, "PSID vs VARMA (DBS-OFF)"),
            (2, 4, w.dpad_vs_varma_off_p, "DPAD vs VARMA (DBS-OFF)"),
            (1, 5, w.psid_vs_varma_on_p, "PSID vs VARMA (DBS-ON)"),
            (3, 5, w.dpad_vs_varma_on_p, "DPAD vs VARMA (DBS-ON)"),
            (0, 2, w.psid_vs_dpad_off_p, "PSID vs DPAD (DBS-OFF)"),
            (1, 3, w.psid_vs_dpad_on_p, "PSID vs DPAD (DBS-ON)"),
        ]
        for k, (xa, xb, p, _lab) in enumerate(pairs):
            stars = p_to_stars(p)
            if not stars:
                continue
            y_line = y0 + k * step
            bracket_y.append((float(xa), float(xb), y_line, stars))

    y_top_brackets = 0.0
    for _, _, y_line, _ in bracket_y:
        y_top_brackets = max(y_top_brackets, y_line + y_span * 0.35)
    y_max = max(ymax_data * 1.08, y_top_brackets, ymax_data + y_span * 0.5)
    if bracket_y:
        y_max = max(y_max, y_top_brackets + y_span * 0.2)
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 0.85

    shapes = []
    for xv in (1.5, 3.5):
        shapes.append(
            dict(
                type="line", xref="x", yref="y",
                x0=xv, x1=xv, y0=0, y1=y_max,
                line=dict(color=COLOR_SEPARATOR, width=0.7, dash="dash"),
                opacity=0.5,
            )
        )

    for xa, xb, y_line, stars in bracket_y:
        shapes.append(
            dict(
                type="line", xref="x", yref="y",
                x0=xa, x1=xb, y0=y_line, y1=y_line,
                line=dict(color=fg, width=1.2),
            )
        )

    annotations = []
    for xa, xb, y_line, stars in bracket_y:
        annotations.append(
            dict(
                x=(xa + xb) / 2, y=y_line + y_span * 0.15,
                xref="x", yref="y",
                text=f"<b>{stars}</b>", showarrow=False,
                font=dict(size=FONT_SIZE_LABEL, color=fg, family=FONT_FAMILY),
            )
        )

    apply_thesis_style(
        fig, theme, height=FIGURE_HEIGHT,
        margin=dict(l=72, r=32, t=50, b=140),
        hovermode="closest", legend_y=-0.12,
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(_X_POS),
            ticktext=_CATEGORY_LABELS,
            title=dict(
                text="Model \u00d7 DBS condition",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                standoff=14,
            ),
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            ),
            range=[0, y_max * 1.02],
            zeroline=True, zerolinecolor=grid,
        ),
        shapes=shapes,
        annotations=annotations,
    )
    return fig

fig_num = 7
for spec in THESIS_AGGREGATE_FIGURES:
    ch, _ = resolve_output_channel_display(
        load_split_results_required(results_root, spec.triplets[0].psid_variant,
                                     spec.triplets[0].psid_run_ts, spec.split),
        spec.channel_idx, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
    )
    agg = collect_pooled_rmse(results_root, spec.triplets, spec.channel_idx,
                              split=spec.split, run_wilcoxon=spec.run_wilcoxon)
    rng = np.random.default_rng(spec.jitter_seed)
    # Pooled view: grouped box-and-whisker (PSID/DPAD/VARMA x DBS-OFF/DBS-ON) with dots by participant.
    fig = build_rmse_boxplot_figure(agg, spec.theme, rng)
    fig.write_image(str(OUT / f'fig_{fig_num:03d}_pooled_rmse_{ch.replace(" ","_")}.png'),
                    width=900, height=500, scale=2)
    fig.show()
    print(
        f"Fig {fig_num}: Pooled per-trial behavioural prediction RMSE for output channel '{ch}'. "
        f"Grouped box-and-whisker (quartiles + median) with jittered per-trial dots coloured by "
        f"participant; two panels for DBS-OFF / DBS-ON, three models per panel (PSID/DPAD/VARMA). "
        f"Trials pooled across all 4 sessions ({', '.join(t.label for t in spec.triplets)})."
    )
    fig_num += 1

# %% [markdown]
# ## Figs 9-16: Per-session RMSE strip plots (2 channels x 4 sessions)

# %%
fig_num = 9
for spec in THESIS_AGGREGATE_FIGURES:
    ch, _ = resolve_output_channel_display(
        load_split_results_required(results_root, spec.triplets[0].psid_variant,
                                     spec.triplets[0].psid_run_ts, spec.split),
        spec.channel_idx, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
    )
    for tri in spec.triplets:
        agg = collect_pooled_rmse(results_root, [tri], spec.channel_idx,
                                   split=spec.split, run_wilcoxon=False)
        rng = np.random.default_rng(spec.jitter_seed)
        # Per-session: grouped box-and-whisker consistent with pooled Figs 7-8.
        fig = build_rmse_boxplot_figure(agg, spec.theme, rng)
        fig.write_image(str(OUT / f'fig_{fig_num:03d}_session_rmse_{tri.label}_{ch.replace(" ","_")}.png'),
                        width=900, height=500, scale=2)
        fig.show()
        pid, sess = tri.label.split("_")
        print(
            f"Fig {fig_num}: Per-trial behavioural prediction RMSE for participant {pid}, "
            f"session {sess[1:]}, output channel '{ch}'. Grouped box-and-whisker plus "
            f"per-trial dots; two panels for DBS-OFF / DBS-ON, three models per panel "
            f"(PSID/DPAD/VARMA)."
        )
        fig_num += 1

# %% [markdown]
# ## Fig 17: Session-mean RMSE strip/box plots
#
# Per-participant box plots (trial-level RMSE) by model x DBS condition.
# Six columns: PSID OFF/ON, DPAD OFF/ON, VARMA OFF/ON — one panel per session.

# %%
from dashboard.thesis.session_strip_rmse import collect_strip_figure_data, StripFigureData

strip_spec = THESIS_STRIP_PANELS[0]
panel_triplets = [(e.panel_label, e.triplet) for e in strip_spec.panels]
strip_data = collect_strip_figure_data(results_root, panel_triplets,
                                        strip_spec.channel_idx, split=strip_spec.split)
rng = np.random.default_rng(strip_spec.jitter_seed)

# --- Build strip/box figure (inlined) ---
data = strip_data
ncols = strip_spec.ncols
theme = strip_spec.theme
jitter = 0.10

n_panels = len(data.panels)
nrows = int(np.ceil(n_panels / max(1, ncols)))
paper_bg, plot_bg = paper_colors(theme)
grid = grid_color(theme)
fg = true_line_color(theme)

fig = make_subplots(rows=nrows, cols=ncols, shared_yaxes=True,
                    horizontal_spacing=0.10, vertical_spacing=0.14)

models = ["PSID", "DPAD", "VARMA"]
model_colors = [COLOR_PSID, COLOR_DPAD, COLOR_VARMA]
# Cell order: 0=PSID-OFF, 1=PSID-ON, 2=DPAD-OFF, 3=DPAD-ON, 4=VARMA-OFF, 5=VARMA-ON
_OFF_CELLS = [0, 2, 4]
_ON_CELLS  = [1, 3, 5]

def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

for pi, panel in enumerate(data.panels):
    row = pi // ncols + 1
    col = pi % ncols + 1
    show_leg = pi == 0

    for mi, (model, mc, off_cell, on_cell) in enumerate(
        zip(models, model_colors, _OFF_CELLS, _ON_CELLS)
    ):
        for cond_label, cell_idx, alpha in (("OFF", off_cell, 0.80), ("ON", on_cell, 0.45)):
            vals = panel.trial_rmse[cell_idx]
            if not vals:
                continue
            arr = [v for v in vals if np.isfinite(v)]
            if not arr:
                continue
            xpos = mi * 2 + (0 if cond_label == "OFF" else 1)
            leg_name = f"{model} {cond_label}"
            # Box plot (quartiles)
            fig.add_trace(go.Box(
                y=arr, x=[xpos] * len(arr), name=leg_name,
                marker_color=mc, line=dict(color=mc, width=1.2),
                fillcolor=_hex_to_rgba(mc, alpha * 0.35),
                boxpoints=False, quartilemethod="exclusive",
                showlegend=show_leg, legendgroup=leg_name, width=0.7,
            ), row=row, col=col)
            # Jittered dots overlay
            jt = rng.uniform(-jitter * 0.4, jitter * 0.4, size=len(arr))
            fig.add_trace(go.Scatter(
                x=[xpos + jt[i] for i in range(len(arr))], y=arr, mode="markers",
                marker=dict(size=5, color=_hex_to_rgba(mc, alpha * 0.85), line=dict(width=0)),
                showlegend=False, legendgroup=leg_name,
                hovertemplate=f"{leg_name}<br>%{{y:.3f}}<extra></extra>",
            ), row=row, col=col)

    # Panel label annotation (top-left)
    fig.add_annotation(
        row=row, col=col, xref="x domain", yref="y domain",
        x=0.04, y=0.93, xanchor="left", yanchor="top",
        text=f"<b>{panel.panel_label}</b>", showarrow=False,
        font=dict(size=FONT_SIZE_BASE, family=FONT_FAMILY, color=fg),
    )
    # Vertical separators between model groups
    for xv in (1.5, 3.5):
        fig.add_shape(
            type="line", xref="x", yref="y", x0=xv, x1=xv, y0=0, y1=data.y_max,
            line=dict(color=COLOR_SEPARATOR, width=0.7, dash="dash"),
            opacity=0.5, row=row, col=col,
        )

# X-axis tick labels: model x DBS condition
x_tick_vals = [0, 1, 2, 3, 4, 5]
x_tick_text = ["PSID\nOFF", "PSID\nON", "DPAD\nOFF", "DPAD\nON", "VARMA\nOFF", "VARMA\nON"]
c_title = max(1, (ncols + 1) // 2)

for r in range(1, nrows + 1):
    for c in range(1, ncols + 1):
        idx = (r - 1) * ncols + (c - 1)
        if idx >= n_panels:
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, row=r, col=c)
            continue
        fig.update_yaxes(
            range=[0, data.y_max], showgrid=True, gridcolor=grid,
            showline=True, linecolor=fg, linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK), row=r, col=c,
        )
        if c == 1:
            fig.update_yaxes(title_text=rmse_axis_label(),
                             title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                             row=r, col=c)
        if r < nrows:
            fig.update_xaxes(showticklabels=False, row=r, col=c)
        else:
            xt = "model x DBS condition" if c == c_title else ""
            fig.update_xaxes(
                tickmode="array", tickvals=x_tick_vals, ticktext=x_tick_text,
                tickangle=-32, tickfont=dict(size=FONT_SIZE_TICK - 1),
                title_text=xt, title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                automargin=True, row=r, col=c,
            )

apply_thesis_style(fig, theme, height=max(FIGURE_HEIGHT, 180 * nrows + 120),
                   margin=dict(l=72, r=24, t=36, b=140), hovermode="closest", legend_y=-0.22)
fig.update_layout(boxmode="overlay")
fig.write_image(str(OUT / 'fig_017_strip_plots.png'), width=1200, height=600, scale=2)
fig.show()
print(
    "Fig 17: Per-session box-plus-strip of test-trial behavioural RMSE (z-scored, channel 0). "
    "Each panel is one session; columns are PSID/DPAD/VARMA x DBS OFF/ON. "
    f"Sessions: {', '.join(p.panel_label for p in strip_data.panels)}."
)

# %% [markdown]
# ## Figs 18-21: Neural reconstruction time series (4 sessions)

# %%
# --- Inlined: compose_thesis_neural_figure + build_side_by_side_exemplar_figure ---
# Lifted from dashboard/thesis/compose.py and dashboard/thesis/figure.py.
# Data-loading helpers remain imports; only the Figure-building pieces were inlined.
from dashboard.thesis.aggregate_rmse import _key_index_map, _trial_key
from dashboard.thesis.compose import _session_mean_rmse_y_triplet
from dashboard.thesis.exemplar_trials import (
    find_best_trial_indices_per_condition,
    resolve_off_on_indices_from_spec,
)
from dashboard.thesis.figure import ThesisPanelData
from dashboard.thesis.loaders import (
    load_split_results, extract_trial_y_series, thesis_exemplar_tagline,
)
from dashboard.thesis.loaders import ThesisDataError
from dashboard.thesis.rmse_callout import RmseRow, rmse_callout_dataframe
from dashboard.thesis.specs import infer_varma_off_on_run_ts
from dashboard.thesis.transforms import rmse_z, z_true_and_preds


def _slice_trial_tail(t_abs, seg_s, z_true, z_psid, z_dpad, z_varma):
    """Return the last ``seg_s`` seconds of each array; t retains absolute values."""
    t = np.asarray(t_abs, dtype=float).ravel()
    if t.size == 0:
        e = np.array([], dtype=float)
        return e, e, e, e, e
    t_hi = float(np.nanmax(t))
    t_lo = t_hi - float(seg_s)
    m = t >= t_lo
    arrays = [np.asarray(a, dtype=float).ravel() for a in (z_true, z_psid, z_dpad, z_varma)]
    return (t[m],) + tuple(a[m] for a in arrays)


def _finite_ys_for_autoscale(zt, zp, zd, zv, band_psid, band_dpad, band_varma):
    out = []
    for arr in (zt, zp, zd, zv):
        a = np.asarray(arr, dtype=float).ravel()
        out.extend(float(x) for x in a if np.isfinite(x))
    for arr, half in ((zp, band_psid), (zd, band_dpad), (zv, band_varma)):
        if half is None or not np.isfinite(half) or half <= 0:
            continue
        a = np.asarray(arr, dtype=float).ravel()
        for x in a:
            if np.isfinite(x):
                out.append(float(x - half))
                out.append(float(x + half))
    return out


def _y_range_from_values(vals, pad_frac=0.08):
    if not vals:
        return -2.5, 2.5
    lo, hi = min(vals), max(vals)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -2.5, 2.5
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    pad = max(span * pad_frac, 0.15)
    return lo - pad, hi + pad


def build_side_by_side_exemplar_figure(
    panel_off, panel_on, theme, y_axis_label,
    *, segment_s=1.0, true_series_name="y_true", prediction_line_dash=None,
):
    """Two side-by-side subplots: DBS-OFF (left) | DBS-ON (right)."""
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    c_true = true_line_color(theme)
    _pd = prediction_line_dash

    def _prep(p):
        t_raw = np.asarray(p.t_abs, dtype=float)
        t_trial = (t_raw - t_raw[0]) if t_raw.size > 0 else t_raw
        t_sl, *zs = _slice_trial_tail(t_trial, segment_s, p.z_true, p.z_psid, p.z_dpad, p.z_varma)
        trial_offset = float(np.nanmin(t_sl)) if t_sl.size > 0 else 0.0
        t_win = (t_sl - trial_offset) if t_sl.size > 0 else t_sl
        return t_win, trial_offset, *zs

    to, off_offset, zto, zpo, zdo, zvo = _prep(panel_off)
    tn, on_offset, ztn, zpn, zdn, zvn = _prep(panel_on)

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=["DBS-OFF", "DBS-ON"],
        horizontal_spacing=0.08,
    )

    if to.size == 0 and tn.size == 0:
        fig.add_annotation(text="Insufficient time samples", showarrow=False)
        return fig

    vals = (
        _finite_ys_for_autoscale(
            zto, zpo, zdo, zvo,
            panel_off.band_rmse_psid, panel_off.band_rmse_dpad, panel_off.band_rmse_varma,
        )
        + _finite_ys_for_autoscale(
            ztn, zpn, zdn, zvn,
            panel_on.band_rmse_psid, panel_on.band_rmse_dpad, panel_on.band_rmse_varma,
        )
    )
    y0, y1 = _y_range_from_values(vals)

    n_ticks = 6

    def _dual_ticks(t_win, trial_offset):
        if t_win.size == 0:
            return [], [], []
        w0, w1 = float(np.nanmin(t_win)), float(np.nanmax(t_win))
        win_vals = np.linspace(w0, w1, n_ticks)
        trial_vals = win_vals + trial_offset
        return (
            [float(v) for v in win_vals],
            [f"{v:.2f}" for v in win_vals],
            [f"{v:.1f}" for v in trial_vals],
        )

    ticks_off, win_text_off, trial_text_off = _dual_ticks(to, off_offset)
    ticks_on, win_text_on, trial_text_on = _dual_ticks(tn, on_offset)

    for col, t, zt, zp, zd, zv, panel in (
        (1, to, zto, zpo, zdo, zvo, panel_off),
        (2, tn, ztn, zpn, zdn, zvn, panel_on),
    ):
        if t.size == 0:
            continue
        show_leg = col == 1

        for y_hat, half, fc, lc, name, lg in (
            (zp, panel.band_rmse_psid, COLOR_PSID_BAND_FILL, COLOR_PSID_BAND_LINE, "PSID \u00b1 mean RMSE", "bps"),
            (zd, panel.band_rmse_dpad, COLOR_DPAD_BAND_FILL, COLOR_DPAD_BAND_LINE, "DPAD \u00b1 mean RMSE", "bpd"),
            (zv, panel.band_rmse_varma, COLOR_VARMA_BAND_FILL, COLOR_VARMA_BAND_LINE, "VARMA \u00b1 mean RMSE", "bpv"),
        ):
            if half is None or not np.isfinite(half) or half <= 0:
                continue
            yh = np.asarray(y_hat, dtype=float)
            if len(t) != len(yh) or np.all(np.isnan(yh)):
                continue
            xb = np.concatenate([t, t[::-1]])
            yb = np.concatenate([yh + half, (yh - half)[::-1]])
            fig.add_trace(
                go.Scatter(
                    x=xb, y=yb, fill="toself",
                    fillcolor=fc, line=dict(color=lc, width=0.5),
                    name=name, legendgroup=lg, showlegend=False,
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )

        show_leg = col == 1
        fig.add_trace(
            go.Scatter(
                x=t, y=zt, name=true_series_name, mode="lines",
                line=dict(color=c_true, width=WIDTH_TRUE),
                legendgroup="true", showlegend=show_leg,
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=zp, name="y_hat_PSID", mode="lines",
                line=dict(color=COLOR_PSID, width=WIDTH_PSID, dash=_pd),
                opacity=OPACITY_PSID,
                legendgroup="psid", showlegend=show_leg,
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=zd, name="y_hat_DPAD", mode="lines",
                line=dict(color=COLOR_DPAD, width=WIDTH_DPAD, dash=_pd),
                opacity=OPACITY_DPAD,
                legendgroup="dpad", showlegend=show_leg,
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=t, y=zv, name="y_hat_VARMA", mode="lines",
                line=dict(color=COLOR_VARMA, width=WIDTH_VARMA, dash=_pd),
                opacity=OPACITY_VARMA,
                legendgroup="varma", showlegend=show_leg,
            ),
            row=1, col=col,
        )

    for col in (1, 2):
        fig.update_yaxes(
            title_text=y_axis_label if col == 1 else "",
            showgrid=True, gridcolor=grid,
            range=[y0, y1],
            row=1, col=col,
        )

    fig.update_xaxes(
        tickmode="array", tickvals=ticks_off, ticktext=win_text_off,
        title_text="Time [s]",
        showgrid=True, gridcolor=grid,
        row=1, col=1,
    )
    fig.update_xaxes(
        tickmode="array", tickvals=ticks_on, ticktext=win_text_on,
        title_text="Time [s]",
        showgrid=True, gridcolor=grid,
        row=1, col=2,
    )

    apply_paper_style(
        fig,
        height=380,
        margin=dict(l=72, r=24, t=80, b=96),
        legend_y=-0.30,
    )
    fig.update_layout(
        hovermode="x unified",
        xaxis3=dict(
            overlaying="x", side="top",
            tickmode="array", tickvals=ticks_off, ticktext=trial_text_off,
            title=dict(text="trial time [s]", font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY)),
            showgrid=False, zeroline=False, showline=False, ticks="outside", ticklen=3,
        ),
        xaxis4=dict(
            overlaying="x2", side="top",
            tickmode="array", tickvals=ticks_on, ticktext=trial_text_on,
            title=dict(text="trial time [s]", font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY)),
            showgrid=False, zeroline=False, showline=False, ticks="outside", ticklen=3,
        ),
    )
    return fig


def compose_thesis_neural_figure(spec, results_root):
    """Neural Y vs Yhat (z-scored per trial), two DBS rows. Returns (figure, rmse_df, caption)."""
    res_p = load_split_results_required(
        results_root, spec.psid_variant, spec.psid_run_ts, spec.split
    )
    # Prefer best-quality (lowest PSID RMSE) trial per DBS condition so exemplars are representative,
    # not arbitrary. Fall back to the spec's adjacent-block-boundary logic if quality ranking fails.
    best_pair = find_best_trial_indices_per_condition(
        res_p, channel_idx=spec.neural_y_channel_idx, input_mode="neural",
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
    res_d = load_split_results(
        results_root, spec.dpad_variant, spec.dpad_run_ts, spec.split
    )
    res_v = load_split_results_required(
        results_root, spec.varma_variant, spec.varma_run_ts, spec.split
    )
    res_v_off = None
    res_v_on = None
    if "dbs_both" in spec.varma_variant:
        v_off = spec.varma_variant.replace("dbs_both", "dbs_off")
        v_on = spec.varma_variant.replace("dbs_both", "dbs_on")
        ts_off = spec.varma_run_ts_off
        ts_on = spec.varma_run_ts_on
        if ts_off is None or ts_on is None:
            inf = infer_varma_off_on_run_ts(spec.varma_variant, spec.varma_run_ts)
            if inf is not None:
                ts_off, ts_on = inf
        if ts_off is None or ts_on is None:
            raise ThesisDataError(
                f"ThesisNeuralTimeseriesSpec {spec.section_title!r}: varma_run_ts_off/on required "
                f"for dbs_both VARMA (or match _ALL_TRIPLETS)."
            )
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
        res_p, res_d, res_v_off_use, map_v_off, i_off,
        spec.neural_y_channel_idx, "off",
    )
    band_p_on, band_d_on, band_v_on = _session_mean_rmse_y_triplet(
        res_p, res_d, res_v_on_use, map_v_on, i_on,
        spec.neural_y_channel_idx, "on",
    )

    off = extract_trial_y_series(
        res_p, res_d, res_v_off_use, i_off,
        spec.neural_y_channel_idx, varma_trial_idx=idx_v_off,
    )
    on = extract_trial_y_series(
        res_p, res_d, res_v_on_use, i_on,
        spec.neural_y_channel_idx, varma_trial_idx=idx_v_on,
    )

    zt_o, zp_o, zd_o, zv_o = z_true_and_preds(
        off.z_true_raw, off.z_psid, off.z_dpad, off.z_varma
    )
    zt_n, zp_n, zd_n, zv_n = z_true_and_preds(
        on.z_true_raw, on.z_psid, on.z_dpad, on.z_varma
    )

    rmse_off = RmseRow(
        "OFF \u2014 RMSE",
        rmse_z(zt_o, zp_o),
        rmse_z(zt_o, zd_o),
        rmse_z(zt_o, zv_o),
    )
    rmse_on = RmseRow(
        "ON \u2014 RMSE",
        rmse_z(zt_n, zp_n),
        rmse_z(zt_n, zd_n),
        rmse_z(zt_n, zv_n),
    )

    panel_off = ThesisPanelData(
        t_abs=off.t_abs,
        z_true=zt_o, z_psid=zp_o, z_dpad=zd_o, z_varma=zv_o,
        psid_sigma=None, dbs_label="DBS-OFF",
        band_rmse_psid=band_p_off, band_rmse_dpad=band_d_off, band_rmse_varma=band_v_off,
    )
    panel_on = ThesisPanelData(
        t_abs=on.t_abs,
        z_true=zt_n, z_psid=zp_n, z_dpad=zd_n, z_varma=zv_n,
        psid_sigma=None, dbs_label="DBS-ON",
        band_rmse_psid=band_p_on, band_rmse_dpad=band_d_on, band_rmse_varma=band_v_on,
    )

    y_meta = neural_y_feature_label(
        res_p, spec.neural_y_channel_idx,
        neural_y_feature_name=spec.neural_y_feature_name,
    )
    y_axis_label = y_meta

    caption = thesis_exemplar_tagline(
        res_p, i_off, i_on, y_meta, participant_label=spec.participant_label
    )
    ce = (spec.caption_extra or "").strip()
    if ce:
        caption = f"{caption} \u00b7 {ce}"

    # This notebook always uses exemplar_layout="side_by_side" (see THESIS_NEURAL_TIMESERIES).
    fig = build_side_by_side_exemplar_figure(
        panel_off, panel_on, spec.theme,
        y_axis_label=y_axis_label,
        segment_s=spec.exemplar_mid_segment_s,
        true_series_name="y_true",
        prediction_line_dash="4 3",
    )
    rmse_df = rmse_callout_dataframe(rmse_off, rmse_on)
    return fig, rmse_df, caption


fig_num = 18
for n_spec in THESIS_NEURAL_TIMESERIES:
    fig, rmse_df, cap = compose_thesis_neural_figure(n_spec, results_root)
    fig.write_image(str(OUT / f'fig_{fig_num:03d}_neural_ts_{n_spec.section_title.replace(" ","_")}.png'),
                    width=1100, height=700, scale=2)
    fig.show()
    print(
        f"Fig {fig_num}: Neural reconstruction time series for {n_spec.section_title} "
        f"(channel '{n_spec.neural_y_feature_name}'). True signal vs. PSID/DPAD/VARMA one-step "
        f"predictions; DBS-OFF and DBS-ON exemplar trials shown side by side."
    )
    print(f"        {cap}")
    fig_num += 1

# %% [markdown]
# ## Fig 22: Neural band Pearson r heatmap
#
# Dual-panel heatmap (DBS-OFF / DBS-ON): frequency bands x models,
# shared Pearson r colorscale.

# %%
from dashboard.thesis.neural_band_pearson import collect_neural_band_pearson, NeuralBandHeatmapData

nb_spec = THESIS_NEURAL_BAND_HEATMAPS[0]
nb_data = collect_neural_band_pearson(results_root, nb_spec.triplets, split=nb_spec.split,
                                      band_row_order=nb_spec.band_row_order)

# --- Build heatmap figure (inlined) ---
# Sequential blue colorscale: light grey -> strong blue
_BLUE_SEQUENTIAL = [
    [0.0, "rgb(250, 250, 250)"],
    [0.35, "rgb(200, 215, 235)"],
    [0.65, "rgb(100, 150, 205)"],
    [1.0, "rgb(24, 95, 165)"],
]
_ZMIN, _ZMAX = 0.45, 1.0

def _fmt_text(z):
    return [[f"{v:.2f}" if np.isfinite(v) else "" for v in row] for row in z]

x_labels = list(nb_data.column_labels)
row_labels = list(nb_data.band_labels)

fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.12)

z_off = np.asarray(nb_data.z_off, dtype=float)
z_on = np.asarray(nb_data.z_on, dtype=float)

for col_i, (z_mat, title_badge) in enumerate([(z_off, "DBS-OFF"), (z_on, "DBS-ON")], start=1):
    text_m = _fmt_text(z_mat)
    # Heatmap trace with cell annotations
    fig.add_trace(go.Heatmap(
        z=z_mat, x=x_labels, y=row_labels,
        text=text_m, texttemplate="%{text}",
        textfont=dict(size=FONT_SIZE_BASE, family=FONT_FAMILY, color="white"),
        coloraxis="coloraxis", showscale=False,
        hovertemplate="%{y} . %{x}<br>r = %{z:.3f}<extra></extra>",
        xgap=2, ygap=2,
    ), row=1, col=col_i)

    # Color-coded column headers (PSID blue, DPAD brown, VARMA grey)
    xref = "x" if col_i == 1 else "x2"
    yref = "y" if col_i == 1 else "y2"
    for j, (lab, hc) in enumerate(zip(x_labels, [COLOR_PSID, COLOR_DPAD, COLOR_VARMA])):
        fig.add_annotation(
            x=x_labels[j], y=1.02, xref=xref, yref=f"{yref} domain",
            text=f"<b>{lab}</b>", showarrow=False,
            font=dict(color=hc, size=FONT_SIZE_BASE, family=FONT_FAMILY),
            xanchor="center", yanchor="bottom",
        )
    # Panel title badge (DBS-OFF / DBS-ON)
    badge_color = COLOR_DBS_OFF if title_badge == "DBS-OFF" else COLOR_DBS_ON
    fig.add_annotation(
        x=0.0, y=1.12, xref=xref, yref=f"{yref} domain",
        text=f"<b>{title_badge}</b>", showarrow=False,
        font=dict(color=badge_color, size=FONT_SIZE_BASE, family=FONT_FAMILY),
        xanchor="left", yanchor="top",
    )

apply_thesis_style(fig, nb_spec.theme, height=FIGURE_HEIGHT + 50,
                   margin=dict(l=100, r=40, t=70, b=120), show_legend=False)

fg = true_line_color(nb_spec.theme)
fig.update_layout(coloraxis=dict(
    cmin=_ZMIN, cmax=_ZMAX, colorscale=_BLUE_SEQUENTIAL,
    colorbar=dict(
        title=dict(text="Pearson r", font=dict(size=FONT_SIZE_BASE)),
        orientation="h", x=0.5, xanchor="center", y=-0.18, yanchor="top",
        len=0.55, thickness=16,
        tickmode="array", tickvals=[0.4, 0.6, 0.8, 1.0], tickfont=dict(color=fg),
    ),
))
for i in (1, 2):
    fig.update_xaxes(row=1, col=i, showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(row=1, col=i, autorange="reversed", showgrid=True,
                     gridcolor=grid_color(nb_spec.theme), zeroline=False,
                     showticklabels=(i == 1),
                     tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY))

fig.write_image(str(OUT / 'fig_022_neural_band_heatmap.png'), width=1100, height=600, scale=2)
fig.show()
print(
    "Fig 22: Neural band Pearson r heatmap. Rows are frequency bands "
    f"({', '.join(nb_data.band_labels)}); columns are PSID/DPAD/VARMA. "
    "Left panel: DBS-OFF, right panel: DBS-ON. Cell value = mean Pearson r between true and "
    f"predicted neural channel, averaged across trials and the {len(nb_spec.triplets)} sessions "
    f"({', '.join(t.label for t in nb_spec.triplets)})."
)

# %% [markdown]
# ## Fig 23: Neural forecast RMSE vs horizon

# %%
from dashboard.thesis.forecast_horizon_rmse import collect_forecast_horizon_rmse

# --- Inlined: build_forecast_rmse_figure_or_empty + build_forecast_rmse_figure ---
# Lifted from dashboard/thesis/forecast_rmse_figure.py (data loader remains imported).
_FC_NAIVE = "#444441"
_FC_REF_LINE = "rgba(180,180,180,0.55)"
_FC_REF_TEXT = "rgba(200,200,200,0.9)"


def _fc_hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _fc_add_sem_band(fig, x, mean, sem, fillcolor, row, col, name, showlegend):
    upper = mean + sem
    lower = mean - sem
    xb = np.concatenate([x, x[::-1]])
    yb = np.concatenate([upper, lower[::-1]])
    fig.add_trace(
        go.Scatter(
            x=xb, y=yb, fill="toself",
            fillcolor=fillcolor, mode="lines", line=dict(width=0),
            showlegend=showlegend, name=name, hoverinfo="skip",
            legendgroup=name or "sem",
        ),
        row=row, col=col,
    )


def _fc_add_model_line(fig, x, y, color, name, dash, symbol, row, col, showlegend):
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines+markers", name=name,
            line=dict(color=color, width=1.8, dash=dash or "solid"),
            marker=dict(color=color, size=6, symbol=symbol, line=dict(width=0)),
            showlegend=showlegend, legendgroup=name,
        ),
        row=row, col=col,
    )


def build_forecast_rmse_figure(
    data, theme, one_step_ms=1000.0 / 60.0, x_max_ms=1000.0,
    y_axis_title=None, column_name="",
):
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)
    y_title = y_axis_title or rmse_axis_label()

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.1)

    ymax = float(data.naive_rmse) * 1.12
    for arr in (data.mean_psid_off, data.mean_varma_off, data.mean_psid_on, data.mean_varma_on):
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

    x = data.x_ms
    naive_y = np.full_like(x, data.naive_rmse, dtype=float)

    def _panel(col, mean_p, sem_p, mean_v, sem_v, badge, _unused, crossover):
        row = 1
        xaxis = "x" if col == 1 else "x2"
        yaxis_d = "y domain"
        if x.size == 0:
            return
        _fc_add_sem_band(fig, x, mean_p, sem_p, _fc_hex_rgba(COLOR_PSID, 0.15),
                         row, col, None, showlegend=False)
        _fc_add_sem_band(fig, x, mean_v, sem_v, _fc_hex_rgba(COLOR_VARMA, 0.15),
                         row, col, None, showlegend=False)
        _fc_add_model_line(fig, x, mean_p, COLOR_PSID, "y_hat_PSID", None, "circle",
                           row, col, showlegend=(col == 1))
        _fc_add_model_line(fig, x, mean_v, COLOR_VARMA, "y_hat_VARMA", "dash", "square",
                           row, col, showlegend=(col == 1))
        fig.add_trace(
            go.Scatter(
                x=x, y=naive_y, mode="lines",
                name="Na\u00efve (mean prediction)",
                line=dict(color=_FC_NAIVE, width=1.2, dash="dot"),
                showlegend=(col == 1), legendgroup="naive",
                hovertemplate="%{y:.3f}<extra></extra>",
            ),
            row=row, col=col,
        )

        if crossover is not None and 0 < crossover < x_max_ms:
            fig.add_shape(
                type="line", x0=crossover, x1=crossover, y0=0, y1=1,
                yref=yaxis_d,
                line=dict(color="rgba(255,200,120,0.7)", width=1, dash="dash"),
                row=row, col=col,
            )
            fig.add_annotation(
                x=crossover, y=0.15, xref=xaxis, yref=yaxis_d,
                text=f"crossover (~{crossover:.0f} ms)", showarrow=False,
                font=dict(size=FONT_SIZE_ANNOTATION - 1, color=_FC_REF_TEXT),
                xanchor="center",
            )

        badge_fg, badge_bg = dbs_badge_style(badge)
        fig.add_annotation(
            x=0.04, y=0.93, xref=f"{xaxis} domain", yref=yaxis_d,
            text=f"<b>{badge}</b>", showarrow=False,
            font=dict(size=FONT_SIZE_BASE, color=badge_fg, family=FONT_FAMILY),
            align="left", bgcolor=badge_bg, bordercolor=badge_fg,
            borderwidth=1, borderpad=4,
        )
        fig.add_annotation(
            x=one_step_ms, y=0.02, xref=xaxis, yref=yaxis_d,
            text="\u2190 1-step", showarrow=False,
            font=dict(size=FONT_SIZE_ANNOTATION - 1, color=fg),
            xanchor="left",
        )

    _panel(1, data.mean_psid_off, data.sem_psid_off, data.mean_varma_off, data.sem_varma_off,
           "DBS-OFF", None, data.crossover_ms_off)
    _panel(2, data.mean_psid_on, data.sem_psid_on, data.mean_varma_on, data.sem_varma_on,
           "DBS-ON", None, data.crossover_ms_on)

    _xkw = dict(
        range=[0, x_max_ms], showgrid=True, gridcolor=grid, zeroline=False,
        tickmode="array", tickvals=[0, 250, 500, 750, 1000],
        showline=True, linecolor=fg, linewidth=1, mirror=False,
        tickfont=dict(size=FONT_SIZE_TICK),
        title_text="Forecast horizon [ms]",
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
    )
    fig.update_xaxes(**_xkw, row=1, col=1)
    fig.update_xaxes(**_xkw, row=1, col=2)

    _ykw = dict(
        range=[0, ymax], showgrid=True, gridcolor=grid,
        showline=True, linecolor=fg, linewidth=1, mirror=False,
        tickfont=dict(size=FONT_SIZE_TICK),
    )
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        **_ykw, row=1, col=1,
    )
    fig.update_yaxes(showticklabels=True, **_ykw, row=1, col=2)

    apply_thesis_style(
        fig, theme, height=FIGURE_HEIGHT,
        margin=dict(l=70, r=40, t=56 if column_name else 36, b=80),
        legend_y=-0.16,
    )
    if column_name:
        fig.update_layout(
            title=column_name,
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        )

    return fig


def build_forecast_rmse_figure_or_empty(data, theme, y_axis_title=None, column_name=""):
    if data is None or data.x_ms.size == 0:
        paper_bg, plot_bg = paper_colors(theme)
        fg = true_line_color(theme)
        fig = go.Figure()
        fig.add_annotation(
            text="No forecast data (need Z_future_true / Z_future_pred in test parquet for PSID and VARMA).",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=fg, family=FONT_FAMILY),
        )
        fig.update_layout(paper_bgcolor=paper_bg, plot_bgcolor=plot_bg)
        return fig
    return build_forecast_rmse_figure(data, theme, y_axis_title=y_axis_title, column_name=column_name)


fc_spec = THESIS_NEURAL_FORECAST_FIGURES[0]
fc_data = collect_forecast_horizon_rmse(
    results_root, fc_spec.triplets, channel_idx=fc_spec.channel_idx, split=fc_spec.split,
    sampling_hz=fc_spec.sampling_hz, sample_every=fc_spec.sample_every,
    naive_rmse=fc_spec.naive_rmse, forecast_target=fc_spec.forecast_target,
    neural_y_feature_name=fc_spec.neural_y_feature_name,
)
res_fc = load_split_results_required(results_root, fc_spec.triplets[0].psid_variant,
                                      fc_spec.triplets[0].psid_run_ts, fc_spec.split)
ch_ix = resolve_neural_y_channel_idx(res_fc, fc_spec.neural_y_feature_name, fc_spec.channel_idx)
inn = channels_as_str_list(res_fc.get('input_channels'))
neu_lbl = inn[ch_ix] if ch_ix < len(inn) else neural_y_feature_label(
    res_fc, ch_ix, neural_y_feature_name=fc_spec.neural_y_feature_name)
fig = build_forecast_rmse_figure_or_empty(fc_data, fc_spec.theme,
                                           y_axis_title=rmse_axis_label(neu_lbl), column_name=neu_lbl)
fig.write_image(str(OUT / 'fig_023_neural_forecast_rmse.png'), width=1000, height=500, scale=2)
fig.show()
print(
    f"Fig 23: Neural forecast RMSE vs. horizon for channel '{neu_lbl}'. "
    f"Mean per-step absolute error (z-scored) over the forecast window, separated by DBS condition. "
    f"Pooled across {len(fc_spec.triplets)} sessions ({', '.join(t.label for t in fc_spec.triplets)})."
)

# %% [markdown]
# ## Figs 24-28: Neural forecast RMSE distributions (pooled + per-session)
#
# Mirror of the prediction RMSE distribution plots (Figs 7-16) but using forecast trial RMSE,
# computed as sqrt(mean(err^2)) where err is the per-step absolute z-error over the entire
# forecast window (from `_per_step_abs_err_z_future`). PSID + VARMA only — DPAD cells empty
# because the forecast head is not yet implemented for DPAD.

# %%
from dashboard.thesis.aggregate_rmse import (
    AggregateRmseData, _key_index_map, _trial_key, _sem, normalize_stim,
)
from dashboard.thesis.forecast_horizon_rmse import _per_step_abs_err_z_future

def _trial_forecast_rmse(res, k_true, k_pred, trial_idx, channel_idx):
    """sqrt(mean(err^2)) where err is the per-step abs z-error over the entire future window."""
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
    return float(np.sqrt(np.mean(finite ** 2)))


def collect_forecast_trial_rmse(triplet_specs, channel_idx, *,
                                 forecast_target="Y",
                                 neural_y_feature_name="ECOG_1_theta_4_8_raw",
                                 split="test"):
    """Build an AggregateRmseData with PSID off/on and VARMA off/on cells filled from
    per-trial forecast RMSE. DPAD cells remain empty (forecast head missing)."""
    k_true = "Y_future_true" if forecast_target == "Y" else "Z_future_true"
    k_pred = "Y_future_pred" if forecast_target == "Y" else "Z_future_pred"
    cells = [[] for _ in range(6)]
    cells_with_pid = [[] for _ in range(6)]

    for tri in triplet_specs:
        res_p = load_split_results_required(results_root, tri.psid_variant, tri.psid_run_ts, split)
        res_v = load_split_results_required(results_root, tri.varma_variant, tri.varma_run_ts, split)
        ch_use = (
            resolve_neural_y_channel_idx(res_p, neural_y_feature_name, channel_idx)
            if forecast_target == "Y"
            else int(channel_idx)
        )

        mp = _key_index_map(res_p)
        mv = _key_index_map(res_v)
        common = set(mp.keys()) & set(mv.keys())
        for k in sorted(common, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))):
            i_p, i_v = mp[k], mv[k]
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue
            r_p = _trial_forecast_rmse(res_p, k_true, k_pred, i_p, ch_use)
            r_v = _trial_forecast_rmse(res_v, k_true, k_pred, i_v, ch_use)
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

# Pooled forecast RMSE (1 figure: neural Y, channel 0)
res_first = load_split_results_required(results_root, ALL_TRIPLETS[0].psid_variant,
                                         ALL_TRIPLETS[0].psid_run_ts, "test")
ch_ix_neural = resolve_neural_y_channel_idx(res_first, "ECOG_1_theta_4_8_raw", 0)
inn0 = channels_as_str_list(res_first.get("input_channels"))
neu_lbl_pooled = inn0[ch_ix_neural] if ch_ix_neural < len(inn0) else "neural Y"

agg_pooled = collect_forecast_trial_rmse(ALL_TRIPLETS, channel_idx=0,
                                          forecast_target="Y",
                                          neural_y_feature_name="ECOG_1_theta_4_8_raw")
rng = np.random.default_rng(42)
fig = build_rmse_boxplot_figure(agg_pooled, ThesisTheme.LIGHT, rng)
fig.write_image(str(OUT / 'fig_024_pooled_forecast_rmse.png'), width=900, height=500, scale=2)
fig.show()
print(
    f"Fig 24: Pooled per-trial *forecast* RMSE for neural channel '{neu_lbl_pooled}'. "
    f"Grouped box-and-whisker (PSID/DPAD/VARMA x DBS-OFF/DBS-ON) with dots coloured by "
    f"participant; trial-level RMSE = sqrt(mean(err^2)) over the entire forecast window "
    f"(z-scored future). DPAD cells empty (forecast head not implemented). Pooled across "
    f"{len(ALL_TRIPLETS)} sessions ({', '.join(t.label for t in ALL_TRIPLETS)})."
)

# Per-session forecast RMSE (4 figures)
fig_num = 25
for tri in ALL_TRIPLETS:
    agg_s = collect_forecast_trial_rmse([tri], channel_idx=0,
                                         forecast_target="Y",
                                         neural_y_feature_name="ECOG_1_theta_4_8_raw")
    rng = np.random.default_rng(42)
    fig = build_rmse_boxplot_figure(agg_s, ThesisTheme.LIGHT, rng)
    fig.write_image(str(OUT / f'fig_{fig_num:03d}_session_forecast_rmse_{tri.label}.png'),
                    width=900, height=500, scale=2)
    fig.show()
    pid, sess = tri.label.split("_")
    print(
        f"Fig {fig_num}: Per-trial *forecast* RMSE for participant {pid}, session {sess[1:]}, "
        f"neural channel '{neu_lbl_pooled}'. Grouped box-and-whisker plus per-trial dots; "
        f"DBS-OFF / DBS-ON panels (PSID/VARMA). DPAD cells empty (forecast head not implemented)."
    )
    fig_num += 1

# %% [markdown]
# ## Figs 29-36: Neural forecast exemplars — best trial per condition (4 sessions x 2 conditions)
#
# One panel per (session, DBS condition). Best trial selected by minimising the joint RMSE
# (max(rmse_psid, rmse_varma)) over all forecast trials in that condition. Single panels are
# rendered separately so they can be composed as subplots in the LaTeX report.

# %%
from dashboard.thesis.c2_forecast_timeseries import _build_one_panel
# COLOR_PSID/DPAD/VARMA, FIGURE_HEIGHT, WIDTH_* already imported from notebooks.thesis_style above

# Render constants matching the dashboard forecasting tab
_FORECAST_CTX_FILL = "rgba(24, 95, 165, 0.08)"   # history shading
_FORECAST_FUT_FILL = "rgba(153, 60, 29, 0.07)"   # forecast shading
_FORECAST_RULE     = "rgba(68, 68, 65, 0.6)"

def _select_best_forecast_trial(res_p, res_v, res_v_split, ch_use, condition_stim):
    """For PSID trials whose stim matches `condition_stim`, find the trial that minimises
    max(rmse_psid, rmse_varma) on the entire forecast window. Returns (i_p, jv, rmse_p, rmse_v)
    or (None, None, nan, nan) if no candidate."""
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
    """Returns (res_v, res_v_off, res_v_on, res_p, ch_use)."""
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


def _build_split_forecast_figure(spec, condition_label, rowdata, neu_lbl):
    """Render a single-panel forecast exemplar (no DBS badge) from `_build_one_panel` output."""
    (t_full, z_true, z_psid, _z_dpad, z_varma, _u, _l, _rp, _rd, _rv, n_hist) = rowdata
    t_full = np.asarray(t_full, dtype=float).ravel()
    n_hist = int(n_hist)

    def _gap(a):
        a = np.asarray(a, dtype=float).ravel()
        if 0 < n_hist < len(a):
            return np.concatenate([a[:n_hist], [np.nan], a[n_hist:]])
        return a

    t_plot = _gap(t_full)
    fig = go.Figure()
    fg = true_line_color(ThesisTheme.LIGHT)

    if t_full.size and n_hist > 0:
        fig.add_vrect(x0=float(t_full[0]), x1=float(t_full[n_hist - 1]),
                       fillcolor=_FORECAST_CTX_FILL, layer="below", line_width=0)
    if t_full.size and n_hist < len(t_full):
        fig.add_vrect(x0=float(t_full[n_hist]), x1=float(t_full[-1]),
                       fillcolor=_FORECAST_FUT_FILL, layer="below", line_width=0)
        fig.add_vline(x=float(t_full[n_hist]), line=dict(color=_FORECAST_RULE, width=1, dash="dash"))

    fig.add_trace(go.Scatter(x=t_plot, y=_gap(z_true), mode="lines", name="y_true",
                              line=dict(color=fg, width=WIDTH_TRUE), connectgaps=False))
    if not np.all(np.isnan(z_psid)):
        fig.add_trace(go.Scatter(x=t_plot, y=_gap(z_psid), mode="lines", name="y_hat_PSID",
                                  line=dict(color=COLOR_PSID, width=WIDTH_PSID), connectgaps=False))
    if not np.all(np.isnan(z_varma)):
        fig.add_trace(go.Scatter(x=t_plot, y=_gap(z_varma), mode="lines", name="y_hat_VARMA",
                                  line=dict(color=COLOR_VARMA, width=WIDTH_VARMA, dash="8 2"),
                                  connectgaps=False))

    apply_thesis_style(fig, ThesisTheme.LIGHT, height=FIGURE_HEIGHT,
                        margin=dict(l=72, r=32, t=24, b=80), legend_y=-0.22)
    fig.update_yaxes(title_text=f"z-score \u2014 {neu_lbl}",
                     tickfont=dict(size=FONT_SIZE_TICK))
    fig.update_xaxes(title_text="Trial time [s]", tickfont=dict(size=FONT_SIZE_TICK))
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
            print(f"Fig {fig_num}: SKIP — no aligned forecast trial for {c2_spec.section_title} {cond_lbl}.")
            fig_num += 1
            continue
        rv_use = rv_split if rv_split is not None else res_v
        rowdata = _build_one_panel(
            res_p, None, rv_use, i_best, ch_use, c2_spec, sigma_z=None,
            varma_trial_idx=jv_best,
        )
        if rowdata is None:
            print(f"Fig {fig_num}: SKIP — _build_one_panel returned None for {c2_spec.section_title} {cond_lbl}.")
            fig_num += 1
            continue
        fig = _build_split_forecast_figure(c2_spec, cond_lbl, rowdata, neu_lbl)
        fig.write_image(
            str(OUT / f'fig_{fig_num:03d}_neural_forecast_{c2_spec.section_title}_{cond_lbl}.png'),
            width=1100, height=500, scale=2,
        )
        fig.show()
        # Trial metadata for caption
        keys_p = (
            res_p.get("participant_id"), res_p.get("session"),
            res_p.get("block"), res_p.get("trial"),
        )
        def _g(seq, idx, default="?"):
            try:
                return seq[idx]
            except Exception:
                return default
        pid_v  = _g(keys_p[0], i_best, "?") if keys_p[0] is not None else "?"
        sess_v = _g(keys_p[1], i_best, "?") if keys_p[1] is not None else "?"
        blk_v  = _g(keys_p[2], i_best, "?") if keys_p[2] is not None else "?"
        tri_v  = _g(keys_p[3], i_best, "?") if keys_p[3] is not None else "?"
        print(
            f"Fig {fig_num}: Neural forecast exemplar — {c2_spec.section_title} DBS-{cond_lbl} "
            f"(channel '{neu_lbl}'). Best trial selected by minimising max(rmse_psid, rmse_varma) "
            f"over the forecast window. Trial = participant {pid_v}, session {sess_v}, "
            f"block {blk_v}, trial {tri_v} (PSID row {i_best}). "
            f"Forecast RMSE: PSID={rmse_p:.3f}, VARMA={rmse_v:.3f}."
        )
        fig_num += 1

# %% [markdown]
# ## Fig 28: Vanilla vs Improved PSID comparison
#
# Grouped bar chart: improved PSID (with BK + rescale) vs vanilla PSID.
# Three metrics: Pearson r, RMSE(z) behavioral, RMSE(z) neural.

# %%
from dashboard.thesis.plot_config import RESULTS_ROOT

vanilla_fig_num = 37  # follows the split forecast exemplars (29-36)

# Session variants for improved vs vanilla comparison (80 Hz narrowband runs)
_VANILLA_SESSIONS = [
    ("PDI1_S2", "psid_behavioral_PDI1_2_nx_80_n12_i40_dbs_both_narrow_band",
     "psid_behavioral_PDI1_2_nx_80_n12_i40_vanilla_dbs_both_narrow_band"),
    ("PDI1_S4", "psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
     "psid_behavioral_PDI1_4_nx_80_n6_i40_vanilla_dbs_both_narrow_band"),
    ("PDI4_S2", "psid_behavioral_PDI4_2_nx_80_n10_i40_dbs_both_narrow_band",
     "psid_behavioral_PDI4_2_nx_80_n10_i40_vanilla_dbs_both_narrow_band"),
    ("PDI4_S3", "psid_behavioral_PDI4_3_nx65_n10_i40_dbs_both_narrow_band",
     "psid_behavioral_PDI4_3_nx65_n10_i40_vanilla_dbs_both_narrow_band"),
]

labels = []
r_improved_z, r_vanilla_z = [], []
rmse_improved_z, rmse_vanilla_z = [], []
rmse_improved_y, rmse_vanilla_y = [], []

for label, improved_dir, vanilla_dir in _VANILLA_SESSIONS:
    for name, result_dir, r_list, rmse_z_list, rmse_y_list in [
        ("improved", improved_dir, r_improved_z, rmse_improved_z, rmse_improved_y),
        ("vanilla", vanilla_dir, r_vanilla_z, rmse_vanilla_z, rmse_vanilla_y),
    ]:
        res_path = RESULTS_ROOT / result_dir
        if not res_path.exists():
            r_list.append(np.nan); rmse_z_list.append(np.nan); rmse_y_list.append(np.nan)
            continue
        train_dir = res_path / "train"
        parquets = list(train_dir.glob("test_results_*.parquet")) if train_dir.exists() else []
        if not parquets:
            r_list.append(np.nan); rmse_z_list.append(np.nan); rmse_y_list.append(np.nan)
            continue
        try:
            tdf = pl.read_parquet(parquets[0])
            # Pearson r (Z-scored)
            for col_name in ["metric_pearson_r_mean_Z", "pearson_mean_Z"]:
                if col_name in tdf.columns:
                    vals = tdf[col_name].to_list()
                    valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    r_list.append(float(np.mean(valid)) if valid else np.nan)
                    break
            else:
                r_list.append(np.nan)
            # RMSE behavioral (Z)
            z_true = np.array(tdf["Z"][0].to_list())
            z_pred = np.array(tdf["Zp"][0].to_list())
            rmse_z_list.append(float(np.sqrt(np.mean((z_true - z_pred) ** 2))))
            # RMSE neural (Y)
            if "Y" in tdf.columns and "Yp" in tdf.columns:
                y_true = np.array(tdf["Y"][0].to_list())
                y_pred = np.array(tdf["Yp"][0].to_list())
                rmse_y_list.append(float(np.sqrt(np.mean((y_true - y_pred) ** 2))))
            else:
                rmse_y_list.append(np.nan)
        except Exception:
            r_list.append(np.nan); rmse_z_list.append(np.nan); rmse_y_list.append(np.nan)
    labels.append(label)

# 3-panel figure: Pearson r | RMSE(z) behavioral | RMSE(z) neural
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.10)
_c_imp, _c_van = COLOR_PSID, "#D97706"  # blue = RTS + A regularization, amber = vanilla
_LBL_IMP = "RTS + A regularization"
_LBL_VAN = "Vanilla PSID"
fig.add_trace(go.Bar(x=labels, y=r_improved_z, name=_LBL_IMP,
                      marker_color=_c_imp, width=0.35), row=1, col=1)
fig.add_trace(go.Bar(x=labels, y=r_vanilla_z, name=_LBL_VAN,
                      marker_color=_c_van, width=0.35), row=1, col=1)
fig.add_trace(go.Bar(x=labels, y=rmse_improved_z, marker_color=_c_imp, width=0.35,
                      showlegend=False, name=_LBL_IMP), row=1, col=2)
fig.add_trace(go.Bar(x=labels, y=rmse_vanilla_z, marker_color=_c_van, width=0.35,
                      showlegend=False, name=_LBL_VAN), row=1, col=2)
fig.add_trace(go.Bar(x=labels, y=rmse_improved_y, marker_color=_c_imp, width=0.35,
                      showlegend=False, name=_LBL_IMP), row=1, col=3)
fig.add_trace(go.Bar(x=labels, y=rmse_vanilla_y, marker_color=_c_van, width=0.35,
                      showlegend=False, name=_LBL_VAN), row=1, col=3)

grd = grid_color(ThesisTheme.LIGHT)
apply_thesis_style(fig, ThesisTheme.LIGHT, height=420, margin=dict(l=60, r=40, t=36, b=80))
fig.update_layout(barmode="group")
fig.update_yaxes(title_text="Pearson r", showgrid=True, gridcolor=grd, row=1, col=1)
fig.update_yaxes(title_text="RMSE(z) \u2014 behavioral", showgrid=True, gridcolor=grd, row=1, col=2)
fig.update_yaxes(title_text="RMSE(z) \u2014 neural", showgrid=True, gridcolor=grd, row=1, col=3)

fig.write_image(str(OUT / f'fig_{vanilla_fig_num:03d}_vanilla_comparison.png'), width=1100, height=500, scale=2)
fig.show()
print(
    f"Fig {vanilla_fig_num}: PSID vanilla vs. RTS-smoothed + A-regularised variant on the same nx/n1 grid "
    f"(80 Hz narrow-band runs, dbs_both). Pooled across 4 sessions ({', '.join(labels)}). "
    f"Left: Pearson r on Z (behavioural). Middle: RMSE(z) on Z. Right: RMSE(z) on Y (neural reconstruction)."
)

# %% [markdown]
# ## Fig 38: PSID grid search BA heatmap (model selection justification)
#
# Per-session heatmaps of LDA test balanced accuracy across the (nx, n1) grid.
# Each cell = balanced accuracy of a DBS-state classifier trained on Xp latents
# from the PSID model with that (nx, n1). The red-outlined cell = the config
# chosen for the rest of section 2.

# %%
import json as _gs_json
import pickle as _gs_pkl

from notebooks.thesis_style import COLOR_CHANCE

# Selected configs used throughout sec2 (matching ALL_TRIPLETS above)
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
    """Return list of (nx, n1, balanced_accuracy) for each grid-search run of this session."""
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
                cls_res = _gs_pkl.load(f)
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

# Build the union of (nx, n1) values actually observed (fall back to a sensible grid)
_all_nx = sorted({nx for rows in _gs_data.values() for (nx, _, _) in rows})
_all_n1 = sorted({n1 for rows in _gs_data.values() for (_, n1, _) in rows})
if not _all_nx:
    _all_nx = [2, 4, 8, 15, 25, 30]
if not _all_n1:
    _all_n1 = [2, 4, 6]

# Color scale range across all available cells
_all_vals = [ba for rows in _gs_data.values() for (_, _, ba) in rows]
if _all_vals:
    _vmin = max(0.40, float(min(_all_vals)) - 0.02)
    _vmax = min(1.0, float(max(_all_vals)) + 0.02)
else:
    _vmin, _vmax = 0.45, 0.70

_gs_nrows, _gs_ncols = len(_GS_SESSIONS), 1
_gs_titles = [f"{label}" for _, _, label in _GS_SESSIONS]

fig = make_subplots(
    rows=_gs_nrows, cols=_gs_ncols,
    vertical_spacing=0.08, horizontal_spacing=0.06,
    subplot_titles=_gs_titles,
)

for ri, (pid, sess, label) in enumerate(_GS_SESSIONS):
    rows_arr = _gs_data[label]
    mat = np.full((len(_all_n1), len(_all_nx)), np.nan)
    text_mat = np.empty_like(mat, dtype=object)
    for n1i, n1v in enumerate(_all_n1):
        for xi, nxv in enumerate(_all_nx):
            if n1v > nxv:
                text_mat[n1i, xi] = ""
                continue
            # Take the best BA if multiple runs exist for the same (nx, n1)
            matches = [ba for (nx_, n1_, ba) in rows_arr if nx_ == nxv and n1_ == n1v]
            if matches:
                val = max(matches)
                mat[n1i, xi] = val
                text_mat[n1i, xi] = f"{val:.2f}"
            else:
                text_mat[n1i, xi] = ""

    x_idx = np.arange(len(_all_nx), dtype=float)
    y_idx = np.arange(len(_all_n1), dtype=float)
    fig.add_trace(
        go.Heatmap(
            z=mat, x=x_idx, y=y_idx,
            text=text_mat, texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale="Blues",
            zmin=_vmin, zmax=_vmax,
            showscale=(ri == 0),
            colorbar=dict(title="Bal. acc.", len=0.35, y=0.82) if ri == 0 else None,
        ),
        row=ri + 1, col=1,
    )
    fig.update_xaxes(
        title_text="n<sub>x</sub>" if ri == _gs_nrows - 1 else "",
        tickmode="array", tickvals=x_idx,
        ticktext=[str(v) for v in _all_nx],
        row=ri + 1, col=1,
    )
    fig.update_yaxes(
        title_text="n<sub>1</sub>",
        tickmode="array", tickvals=y_idx,
        ticktext=[str(v) for v in _all_n1],
        row=ri + 1, col=1,
    )
    if not rows_arr:
        fig.add_annotation(
            text="<i>no grid-search data</i>",
            x=0.5, y=0.5, xref="x domain", yref="y domain",
            showarrow=False,
            font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color="#888780"),
            row=ri + 1, col=1,
        )
    picked = _GS_SELECTED.get(label)
    if rows_arr and picked is not None and picked[0] in _all_nx and picked[1] in _all_n1:
        bx = _all_nx.index(picked[0])
        by = _all_n1.index(picked[1])
        fig.add_shape(
            type="rect",
            x0=float(bx) - 0.5, x1=float(bx) + 0.5,
            y0=float(by) - 0.5, y1=float(by) + 0.5,
            line=dict(color=COLOR_CHANCE, width=2.5),
            fillcolor="rgba(0,0,0,0)", layer="above",
            row=ri + 1, col=1,
        )

apply_thesis_style(
    fig, ThesisTheme.LIGHT,
    height=220 * _gs_nrows + 40,
    margin=dict(l=60, r=80, t=44, b=60),
    show_legend=False,
)
fig.update_annotations(font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY))

fig.write_image(str(OUT / 'fig_038_grid_search_ba_heatmap.png'), width=900, height=220 * _gs_nrows + 120, scale=2)
fig.show()

_missing = [label for label, rows_arr in _gs_data.items() if not rows_arr]
_have = [label for label, rows_arr in _gs_data.items() if rows_arr]
print(
    f"Fig 38 \u2014 PSID grid-search BA heatmap used to pick (n_x, n_1). "
    f"Each cell: LDA test balanced accuracy for DBS-state classification of Xp latents "
    f"from a PSID model trained at that (n_x, n_1) on dbs_both (200 Hz narrowband). "
    f"Red-outlined cell = selected config used in all other sec2 figures. "
    f"Rows (top\u2192bottom): {', '.join(label for _, _, label in _GS_SESSIONS)}. "
    f"n_x values: {_all_nx}; n_1 values: {_all_n1}. "
    f"Data source: results/psid_gs_{{session}}_200Hz_narrow_band/*/model_*_metadata.json "
    f"+ results/classification/gs_200Hz/{{run}}/LDA_Xp_prediction.pkl. "
    f"Sessions with grid-search data: {_have or 'none'}. "
    f"Sessions missing grid-search data: {_missing or 'none'}."
)

# %% [markdown]
# ## Figs 39-43: Neural reconstruction RMSE distributions (pooled + per-session)
#
# Grouped box-and-whisker of per-trial neural Y (one-step) prediction RMSE by
# PSID/DPAD/VARMA x DBS-OFF/DBS-ON. Parallel to the behavioural distribution
# figures (Figs 7-16) but scored on the neural channel used in the reconstruction
# time series panels (Figs 18-21).

# %%
from dashboard.thesis.aggregate_rmse import (
    AggregateRmseData as _AggRmseY,
    _key_index_map as _kim_y,
    _sem as _sem_y,
    normalize_stim as _norm_stim_y,
)
from dashboard.thesis.loaders import trial_rmse_y_for_model


def _collect_pooled_rmse_y(triplet_specs, y_channel_idx, split="test"):
    """Neural-Y analogue of collect_pooled_rmse: scores Y/Yp per trial via trial_rmse_y_for_model."""
    cells = [[] for _ in range(6)]
    cells_pid = [[] for _ in range(6)]
    n_ok = 0
    for tri in triplet_specs:
        res_p = load_split_results_required(results_root, tri.psid_variant, tri.psid_run_ts, split)
        res_v = load_split_results_required(results_root, tri.varma_variant, tri.varma_run_ts, split)
        has_dpad = bool(tri.dpad_run_ts)
        res_d = load_split_results(results_root, tri.dpad_variant, tri.dpad_run_ts, split) if has_dpad else None
        mp, mv = _kim_y(res_p), _kim_y(res_v)
        md = _kim_y(res_d) if res_d else {}
        common = set(mp.keys()) & set(mv.keys())
        if md:
            common &= set(md.keys())
        if not common:
            continue
        n_ok += 1
        for k in sorted(common, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))):
            i_p, i_v = mp[k], mv[k]
            stim = _norm_stim_y(res_p["stim"][i_p])
            if stim is None:
                continue
            try:
                r_p = trial_rmse_y_for_model(res_p, i_p, y_channel_idx)
                r_v = trial_rmse_y_for_model(res_v, i_v, y_channel_idx)
                r_d = (
                    trial_rmse_y_for_model(res_d, md[k], y_channel_idx)
                    if (res_d and k in md) else float("nan")
                )
            except (ValueError, IndexError, KeyError):
                continue
            pid = str(k[0])
            if stim == "off":
                cells[0].append(r_p); cells[2].append(r_d); cells[4].append(r_v)
                cells_pid[0].append((r_p, pid))
                cells_pid[2].append((r_d, pid))
                cells_pid[4].append((r_v, pid))
            else:
                cells[1].append(r_p); cells[3].append(r_d); cells[5].append(r_v)
                cells_pid[1].append((r_p, pid))
                cells_pid[3].append((r_d, pid))
                cells_pid[5].append((r_v, pid))
    means = tuple(float(np.mean([v for v in c if np.isfinite(v)])) if any(np.isfinite(v) for v in c) else float("nan") for c in cells)
    sems = tuple(_sem_y(np.array([v for v in c if np.isfinite(v)])) if sum(1 for v in c if np.isfinite(v)) > 1 else float("nan") for c in cells)
    return _AggRmseY(
        means=means, sems=sems,
        trial_rmse=tuple(cells),
        trial_rmse_with_participant=tuple(cells_pid),
        n_triplets_used=n_ok,
    )


# Resolve neural Y channel once (same channel as neural reconstruction time series).
_NEURAL_Y_FEATURE = "ECOG_1_theta_4_8_raw"
_res_y0 = load_split_results_required(
    results_root, ALL_TRIPLETS[0].psid_variant, ALL_TRIPLETS[0].psid_run_ts, "test",
)
_y_ch_ix = resolve_neural_y_channel_idx(_res_y0, _NEURAL_Y_FEATURE, 0)
_inn_y = channels_as_str_list(_res_y0.get("input_channels"))
_neu_lbl_y = _inn_y[_y_ch_ix] if _y_ch_ix < len(_inn_y) else _NEURAL_Y_FEATURE

# Fig 39: pooled across 4 sessions
agg_y_pooled = _collect_pooled_rmse_y(ALL_TRIPLETS, y_channel_idx=_y_ch_ix)
rng = np.random.default_rng(42)
fig = build_rmse_boxplot_figure(agg_y_pooled, ThesisTheme.LIGHT, rng)
fig.write_image(
    str(OUT / f'fig_039_pooled_neural_pred_rmse_{_neu_lbl_y.replace(" ","_")}.png'),
    width=900, height=500, scale=2,
)
fig.show()
print(
    f"Fig 39: Pooled per-trial neural (Y) reconstruction RMSE for channel '{_neu_lbl_y}'. "
    f"Grouped box-and-whisker (PSID/DPAD/VARMA x DBS-OFF/DBS-ON) with per-trial dots coloured "
    f"by participant. Trials pooled across {len(ALL_TRIPLETS)} sessions "
    f"({', '.join(t.label for t in ALL_TRIPLETS)})."
)

# Figs 40-43: per-session
fig_num = 40
for tri in ALL_TRIPLETS:
    agg_y_s = _collect_pooled_rmse_y([tri], y_channel_idx=_y_ch_ix)
    rng = np.random.default_rng(42)
    fig = build_rmse_boxplot_figure(agg_y_s, ThesisTheme.LIGHT, rng)
    fig.write_image(
        str(OUT / f'fig_{fig_num:03d}_session_neural_pred_rmse_{tri.label}.png'),
        width=900, height=500, scale=2,
    )
    fig.show()
    pid, sess = tri.label.split("_")
    print(
        f"Fig {fig_num}: Per-trial neural (Y) reconstruction RMSE for participant {pid}, "
        f"session {sess[1:]}, channel '{_neu_lbl_y}'. Grouped box-and-whisker plus per-trial "
        f"dots; DBS-OFF / DBS-ON panels (PSID/DPAD/VARMA)."
    )
    fig_num += 1

# %%
n = len(list(OUT.glob('*.png')))
print(f'Section 2 total: {n} figures')
