"""
Build a single static HTML file with the same thesis sections as `dashboard_thesis_final.py`.

Run: ``python scripts/generate_thesis_html.py`` (or ``--output`` / ``RESULTS_PATH``; use ``-v`` for thesis WARNING logs).
Optional: ``THESIS_USE_SPEC_TIMESTAMPS=1`` pins run IDs to ``specs.py`` instead of latest on disk.
Serve locally: ``python scripts/serve_thesis_html.py`` → open ``http://127.0.0.1:8765/thesis_results.html``.
Plotly.js is loaded once from CDN; figures are embedded as interactive HTML fragments (not PNG images).
"""

from __future__ import annotations

import html
import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from plotly.io import to_html

from dashboard.thesis.aggregate_rmse import collect_pooled_rmse
from dashboard.thesis.c2_forecast_timeseries import build_c2_forecast_figure
from dashboard.thesis.cross_block_predictions import (
    build_cross_block_predictions_figure,
)
from dashboard.thesis.forecast_checkpoint_compare import (
    build_forecast_checkpoint_compare_figure,
)
from dashboard.thesis.classification_f1_data import (
    GROUP_ORDER,
    collect_classification_f1_points,
)
from dashboard.thesis.classification_f1_figure import (
    build_classification_f1_figure,
    build_classification_grouped_bar_figure,
    _FEAT_SHORT,
)
from dashboard.thesis.compose import compose_thesis_figure, compose_thesis_neural_figure
from dashboard.thesis.forecast_horizon_rmse import collect_forecast_horizon_rmse
from dashboard.thesis.forecast_rmse_figure import (
    build_forecast_global_rmse_figure,
    build_forecast_rmse_figure_or_empty,
)
from dashboard.thesis.loaders import (
    channels_as_str_list,
    load_split_results_required,
    neural_y_feature_label,
    resolve_neural_y_channel_idx,
    resolve_output_channel_display,
)
from dashboard.thesis.latent_phase_space_figure import build_latent_phase_space_figure
from dashboard.thesis.neural_band_heatmap_figure import build_neural_band_heatmap_figure
from dashboard.thesis.neural_band_pearson import collect_neural_band_pearson
from dashboard.thesis.psid_cy_importance_figure import build_psid_cy_importance_figure
from dashboard.thesis.psid_cz_figure import build_psid_cz_figure
from dashboard.thesis.constants import rmse_axis_label
from dashboard.thesis.rmse_distribution_figure import (
    build_rmse_distribution_figure,
    build_rmse_boxplot_figure,
)
from dashboard.thesis.session_strip_figure import build_session_strip_boxplot_figure
from dashboard.thesis.session_strip_rmse import collect_strip_figure_data
from dashboard.thesis.fig_within_cross import (
    build_within_cross_timeseries_figure,
    build_within_cross_boxplot_figure,
)
from dashboard.thesis.fig_appendix import (
    build_psd_dbs_comparison_figures,
    build_tracing_speed_dbs_comparison_figure,
    build_classification_heatmap_figure,
    build_trial_count_summary_figure,
    build_vanilla_comparison_figure,
    build_laplacian_prediction_figure,
    build_laplacian_timeseries_figure,
)
from dashboard.thesis.fig_grid_alignment import build_grid_alignment_figure
from dashboard.thesis.fig_raw_vs_laplacian import build_raw_vs_laplacian_figure
from dashboard.thesis.dpad_training_curves_figure import (
    build_dpad_training_curves_figure,
    build_dpad_training_time_figure,
)
from dashboard.thesis.data_efficiency_figure import build_data_efficiency_figure
from dashboard.thesis.aggregate_rmse import collect_within_cross_rmse
import dashboard.thesis.specs as thesis_specs
from dashboard.thesis.specs import (
    DEFAULT_AGGREGATE_CAPTION,
    DEFAULT_C2_FORECAST_CAPTION,
    DEFAULT_CLASSIFICATION_F1_CAPTION,
    DEFAULT_FORECAST_CAPTION,
    DEFAULT_LATENT_PHASE_CAPTION,
    DEFAULT_NEURAL_FORECAST_CAPTION,
    DEFAULT_NEURAL_BAND_CAPTION,
    DEFAULT_NEURAL_TS_CAPTION,
    DEFAULT_PSID_CY_IMPORTANCE_CAPTION,
    DEFAULT_PSID_CZ_CAPTION,
    DEFAULT_STRIP_CAPTION,
    DEFAULT_FORECAST_CHECKPOINT_CAPTION,
    THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
)

_CSS = """
:root {
  --bg: #ffffff;
  --fg: #2c2c2a;
  --muted: #555550;
  --rule: #e0e0dc;
  --err-bg: #fce8e8;
  --err-fg: #b91c1c;
  --warn-bg: #fef3cd;
  --warn-fg: #856404;
  --cap-bg: #f8f8f6;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.55;
}
main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 2rem 1.5rem 3rem;
}
h1 { font-size: 1.6rem; font-weight: 650; margin: 0 0 0.75rem; }
h2 {
  font-size: 1.15rem;
  font-weight: 600;
  margin: 2rem 0 0.75rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--rule);
}
h2:first-of-type { border-top: none; padding-top: 0; }
h3 { font-size: 1rem; font-weight: 600; margin: 1.25rem 0 0.5rem; }
p.caption, .caption {
  font-size: 0.84rem;
  color: var(--muted);
  margin: 0.25rem 0 1.2rem;
  padding: 0.5rem 0.75rem;
  background: var(--cap-bg);
  border-left: 3px solid var(--rule);
  line-height: 1.55;
}
p.error {
  background: var(--err-bg);
  color: var(--err-fg);
  padding: 0.6rem 1rem;
  border-radius: 5px;
  border: 1px solid #f5c6cb;
  margin: 0.5rem 0 1rem;
  font-size: 0.88rem;
}
p.warning {
  background: var(--warn-bg);
  color: var(--warn-fg);
  padding: 0.6rem 1rem;
  border-radius: 5px;
  border: 1px solid #f0d77e;
  margin: 0.5rem 0 1rem;
  font-size: 0.88rem;
}
.figure-wrap { margin: 0.5rem 0 0.25rem; width: 100%; overflow-x: auto; }
.rmse-label { font-weight: 600; margin: 0.75rem 0 0.35rem; font-size: 0.9rem; }
table.rmse-table {
  border-collapse: collapse;
  font-size: 0.82rem;
  margin: 0 0 0.75rem;
}
table.rmse-table th, table.rmse-table td {
  border: 1px solid var(--rule);
  padding: 0.35rem 0.6rem;
  text-align: left;
}
table.rmse-table th { background: #f6f6f4; }
hr.section { border: none; border-top: 1px solid var(--rule); margin: 2rem 0; }
p.report-toc code { font-size: 0.82em; color: var(--muted); }
"""


def _escape(s: object) -> str:
    return html.escape(str(s), quote=True)


def _p_error(msg: object) -> str:
    return f'<p class="error">{_escape(msg)}</p>'


def _p_warning(msg: object) -> str:
    return f'<p class="warning">{_escape(msg)}</p>'


def _p_caption(text: str, *, raw: bool = False) -> str:
    """Caption paragraph. Use raw=True when text already contains safe HTML."""
    body = text if raw else _escape(text)
    return f'<p class="caption">{body}</p>'


def _df_to_html(df: pd.DataFrame) -> str:
    return df.to_html(
        classes="rmse-table",
        index=False,
        border=0,
        escape=True,
        na_rep="N/A",
    )


class _PlotlyEmbed:
    """Emit Plotly fragments; include plotly.js CDN only on the first figure."""

    def __init__(self) -> None:
        self._first = True

    def append_figure(self, parts: list[str], fig: Figure) -> None:
        inc: bool | str
        if self._first:
            inc = "cdn"
            self._first = False
        else:
            inc = False
        parts.append(
            to_html(
                fig,
                include_plotlyjs=inc,
                full_html=False,
                config={"responsive": True},
            )
        )


@contextmanager
def _thesis_report_log_level(*, verbose: bool):
    """
    During HTML export, suppress expected WARNING/INFO noise from thesis collectors
    (skipped triplets, strip panels) unless verbose is True.
    """
    log = logging.getLogger("dashboard.thesis")
    prev = log.level
    try:
        if not verbose:
            log.setLevel(logging.ERROR)
        yield
    finally:
        log.setLevel(prev)


def build_thesis_html_document(
    results_root: Path,
    *,
    verbose_logging: bool = False,
    project_root: Path | None = None,
    use_latest_result_timestamps: bool | None = None,
) -> str:
    """Return full HTML document string (self-contained except Plotly CDN)."""
    if use_latest_result_timestamps is None:
        use_latest_result_timestamps = (
            os.environ.get("THESIS_USE_SPEC_TIMESTAMPS", "").strip() != "1"
        )
    with _thesis_report_log_level(verbose=verbose_logging):
        return _build_thesis_html_document_body(
            results_root,
            project_root=project_root,
            use_latest_result_timestamps=use_latest_result_timestamps,
        )


def _transition(text: str) -> str:
    """Render italic muted transition text between sections."""
    return f'<p style="color: var(--muted); font-style: italic; margin: 1.5rem 0 0.5rem; font-size: 0.92rem;">{text}</p>'


def _build_thesis_html_document_body(
    results_root: Path,
    project_root: Path | None = None,
    *,
    use_latest_result_timestamps: bool,
) -> str:
    split = "test"
    if use_latest_result_timestamps:
        from dashboard.thesis.timestamp_discovery import build_thesis_dashboard_specs

        d = build_thesis_dashboard_specs(results_root, split=split)
        THESIS_FIGURES = d.figures
        THESIS_NEURAL_TIMESERIES = d.neural_timeseries
        THESIS_C2_FORECASTS = d.c2_forecasts
        THESIS_CROSS_BLOCK = d.cross_block
        THESIS_FORECAST_CHECKPOINT = d.forecast_checkpoint
        THESIS_NEURAL_FORECAST_FIGURES = d.neural_forecast_figures
        THESIS_CLASSIFICATION_F1 = d.classification_f1
        THESIS_AGGREGATE_FIGURES = d.aggregate_figures
        THESIS_STRIP_PANELS = d.strip_panels
        THESIS_FORECAST_FIGURES = d.forecast_figures
        THESIS_NEURAL_BAND_HEATMAPS = d.neural_band_heatmaps
        THESIS_LATENT_PHASE = d.latent_phase
        THESIS_PSID_CY_IMPORTANCE = d.psid_cy_importance
        THESIS_PSID_CZ_HEATMAP = d.psid_cz_heatmap
        THESIS_WITHIN_CROSS = d.within_cross
    else:
        THESIS_FIGURES = thesis_specs.THESIS_FIGURES
        THESIS_NEURAL_TIMESERIES = thesis_specs.THESIS_NEURAL_TIMESERIES
        THESIS_C2_FORECASTS = thesis_specs.THESIS_C2_FORECASTS
        THESIS_CROSS_BLOCK = thesis_specs.THESIS_CROSS_BLOCK
        THESIS_FORECAST_CHECKPOINT = thesis_specs.THESIS_FORECAST_CHECKPOINT
        THESIS_NEURAL_FORECAST_FIGURES = thesis_specs.THESIS_NEURAL_FORECAST_FIGURES
        THESIS_CLASSIFICATION_F1 = thesis_specs.THESIS_CLASSIFICATION_F1
        THESIS_AGGREGATE_FIGURES = thesis_specs.THESIS_AGGREGATE_FIGURES
        THESIS_STRIP_PANELS = thesis_specs.THESIS_STRIP_PANELS
        THESIS_FORECAST_FIGURES = thesis_specs.THESIS_FORECAST_FIGURES
        THESIS_NEURAL_BAND_HEATMAPS = thesis_specs.THESIS_NEURAL_BAND_HEATMAPS
        THESIS_LATENT_PHASE = thesis_specs.THESIS_LATENT_PHASE
        THESIS_PSID_CY_IMPORTANCE = thesis_specs.THESIS_PSID_CY_IMPORTANCE
        THESIS_PSID_CZ_HEATMAP = thesis_specs.THESIS_PSID_CZ_HEATMAP
        THESIS_WITHIN_CROSS = thesis_specs.THESIS_WITHIN_CROSS

    parts: list[str] = []
    plotly_e = _PlotlyEmbed()

    def add_fig(fig: Figure) -> None:
        parts.append('<div class="figure-wrap">')
        plotly_e.append_figure(parts, fig)
        parts.append("</div>")

    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append("<title>Thesis Results</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<main>")

    # Title and introduction
    parts.append(
        "<h1>Latent Neural Dynamics Modelling for Deep Brain Stimulation: Results</h1>"
    )
    parts.append(
        "<p>This report evaluates three modelling frameworks for learning latent representations "
        "from neural recordings during deep brain stimulation (DBS). "
        "<b>PSID</b> (Preferential Subspace Identification) decomposes neural dynamics into "
        "behaviourally relevant and irrelevant latent subspaces using a linear state-space model. "
        "<b>DPAD</b> extends this decomposition with deep recurrent networks. "
        "<b>VARMA</b> serves a dual role: as a non-latent reconstruction baseline "
        "(testing whether subspace decomposition improves prediction) and as a classification "
        "upper bound (its predictions retain all information in the original input features). "
        "All models are trained at 200 Hz on narrow-band ECoG features.</p>"
    )

    c2_neural_specs = [s for s in THESIS_C2_FORECASTS if s.forecast_target == "Y"]
    c2_z_specs = [s for s in THESIS_C2_FORECASTS if s.forecast_target != "Y"]

    # ===================================================================
    # Section 1: Data Verification
    # ===================================================================

    parts.append("<h2>1. Data Verification</h2>")

    # Trial count per session and DBS condition
    try:
        fig_tc = build_trial_count_summary_figure()
        add_fig(fig_tc)
        parts.append(
            _p_caption(
                "<b>Trial count per session and DBS condition.</b> "
                "Number of trials per participant-session and DBS condition after preprocessing.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build trial count figure: {e}"))

    # ECoG PSD: DBS-ON vs DBS-OFF (one figure per participant-session)
    parts.append('<hr class="section">')
    try:
        psd_figures = build_psd_dbs_comparison_figures()
        for psd_label, fig_psd in psd_figures:
            add_fig(fig_psd)
            parts.append(
                _p_caption(
                    f"<b>ECoG PSD \u2014 {_escape(psd_label)}.</b> "
                    "Per-channel mean PSD \u00b11 SEM across trials. "
                    "Vertical dashed lines = 13\u201329 Hz beta band.",
                    raw=True,
                )
            )
    except Exception as e:
        parts.append(_p_error(f"Failed to build PSD figures: {e}"))

    # Tracing speed: DBS-ON vs DBS-OFF
    parts.append('<hr class="section">')
    try:
        fig_ts = build_tracing_speed_dbs_comparison_figure()
        add_fig(fig_ts)
        parts.append(
            _p_caption(
                "<b>Tracing speed: DBS-ON vs DBS-OFF.</b> "
                "Mean z-scored tracing speed per DBS condition, averaged across trials.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build tracing speed figure: {e}"))

    # Grid alignment: raw behavioural samples vs 200 Hz master grid
    parts.append('<hr class="section">')
    try:
        fig_ga = build_grid_alignment_figure(participant="PDI1", session="2")
        add_fig(fig_ga)
        parts.append(
            _p_caption(
                "<b>Grid alignment: raw behavioural samples vs. 200 Hz master grid.</b> "
                "Top: raw sample times (rug marks) overlaid on the master 200 Hz grid. "
                "Middle: raw behavioural signal (circles) vs. interpolated values on the grid (line). "
                "Bottom: distribution of inter-sample intervals showing sampling jitter.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build grid alignment figure: {e}"))

    # Raw ECoG vs Laplacian LFP
    parts.append('<hr class="section">')
    try:
        fig_rl = build_raw_vs_laplacian_figure(participant="PDI1", session="2")
        add_fig(fig_rl)
        parts.append(
            _p_caption(
                "<b>Raw ECoG vs. Laplacian LFP reference.</b> "
                "Left column: monopolar ECoG traces. Right column: Laplacian-referenced STN LFP "
                "used as the behavioural target for the cross-modal PSID variant. "
                "Spatial referencing suppresses common-mode noise and isolates local contributions.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build raw-vs-laplacian figure: {e}"))

    parts.append(
        _transition(
            "With the data verified, we assess whether the models reconstruct neural and behavioural signals."
        )
    )

    # ===================================================================
    # Section 2: Model Validation
    # ===================================================================

    parts.append("<h2>2. Model Validation</h2>")

    # Pooled behavioral RMSE bars
    for spec in THESIS_AGGREGATE_FIGURES:
        ch_agg, _da = resolve_output_channel_display(
            load_split_results_required(
                results_root,
                spec.triplets[0].psid_variant,
                spec.triplets[0].psid_run_ts,
                spec.split,
            ),
            spec.channel_idx,
            declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
        )
        try:
            agg = collect_pooled_rmse(
                results_root,
                spec.triplets,
                spec.channel_idx,
                split=spec.split,
                run_wilcoxon=spec.run_wilcoxon,
            )
            rng = np.random.default_rng(spec.jitter_seed)
            fig_b = build_rmse_distribution_figure(
                agg,
                spec.theme,
                rng,
                show_brackets=False,
                y_axis_label=rmse_axis_label(ch_agg),
            )
            add_fig(fig_b)
            n_off = len(agg.trial_rmse[0])
            n_on = len(agg.trial_rmse[1])
            parts.append(
                _p_caption(
                    f"<b>Pooled behavioral RMSE ({_escape(ch_agg)}).</b> "
                    f"Bar height = mean RMSE(z) across {agg.n_triplets_used} participant-sessions "
                    f"({n_off} OFF / {n_on} ON trials), error bars = \u00b11 SEM. "
                    f"PSID, DPAD, and VARMA evaluated on identical held-out trials.",
                    raw=True,
                )
            )
            rng_box = np.random.default_rng(spec.jitter_seed)
            fig_box = build_rmse_boxplot_figure(
                agg,
                spec.theme,
                rng_box,
            )
            add_fig(fig_box)
            parts.append(
                _p_caption(
                    f"<b>Pooled behavioral RMSE distribution ({_escape(ch_agg)}).</b> "
                    f"Grouped boxplots (PSID / DPAD / VARMA × DBS-OFF / DBS-ON) over all pooled "
                    f"trial-level RMSEs ({n_off} OFF / {n_on} ON trials, "
                    f"{agg.n_triplets_used} participant-sessions).",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build aggregate figure: {e}"))

    # Per-session RMSE bars
    parts.append('<hr class="section">')
    for spec in THESIS_AGGREGATE_FIGURES:
        ch_ps, _dps = resolve_output_channel_display(
            load_split_results_required(
                results_root,
                spec.triplets[0].psid_variant,
                spec.triplets[0].psid_run_ts,
                spec.split,
            ),
            spec.channel_idx,
            declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
        )
        for tri in spec.triplets:
            session_label = tri.label or "unknown"
            try:
                agg_single = collect_pooled_rmse(
                    results_root,
                    [tri],
                    spec.channel_idx,
                    split=spec.split,
                    run_wilcoxon=False,
                )
                rng_s = np.random.default_rng(spec.jitter_seed)
                fig_session = build_rmse_distribution_figure(
                    agg_single,
                    spec.theme,
                    rng_s,
                    show_brackets=False,
                    show_dots=False,
                    y_axis_label=rmse_axis_label(ch_ps),
                )
                add_fig(fig_session)
                n_off_s = len(agg_single.trial_rmse[0])
                n_on_s = len(agg_single.trial_rmse[1])
                stat_parts = []
                model_labels = ["PSID", "PSID", "DPAD", "DPAD", "VARMA", "VARMA"]
                cond_labels = ["OFF", "ON", "OFF", "ON", "OFF", "ON"]
                for i in range(6):
                    m_v = (
                        agg_single.means[i]
                        if i < len(agg_single.means)
                        else float("nan")
                    )
                    s_v = (
                        agg_single.sems[i] if i < len(agg_single.sems) else float("nan")
                    )
                    if math.isfinite(m_v):
                        stat_parts.append(
                            f"{model_labels[i]} {cond_labels[i]}: {m_v:.3f} \u00b1 {s_v:.3f}"
                        )
                stat_str = ". ".join(stat_parts) + "." if stat_parts else ""
                parts.append(
                    _p_caption(
                        f"<b>{_escape(session_label)}. Trial-level RMSE(z) \u2014 {_escape(ch_ps)}.</b> "
                        f"{n_off_s} OFF / {n_on_s} ON held-out trials. "
                        f"{_escape(stat_str)}",
                        raw=True,
                    )
                )
            except Exception as e:
                parts.append(
                    _p_error(
                        f"Failed to build per-session RMSE figure for {session_label}: {e}"
                    )
                )

    # Session-mean RMSE strip plots
    parts.append('<hr class="section">')
    for strip_spec in THESIS_STRIP_PANELS:
        try:
            panel_triplets = [(e.panel_label, e.triplet) for e in strip_spec.panels]
            strip_data = collect_strip_figure_data(
                results_root,
                panel_triplets,
                strip_spec.channel_idx,
                split=strip_spec.split,
            )
            if strip_data is None or not strip_data.panels:
                raise ValueError("No panels produced.")
            rng = np.random.default_rng(strip_spec.jitter_seed)
            fig_s = build_session_strip_boxplot_figure(
                strip_data,
                ncols=strip_spec.ncols,
                theme=strip_spec.theme,
                rng=rng,
            )
            add_fig(fig_s)
            parts.append(
                _p_caption(
                    "<b>Session-mean RMSE strip plots.</b> "
                    "Per-participant breakdown of model RMSE across sessions. "
                    f"{_escape(strip_spec.caption or '')}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build strip plot: {e}"))

    # Neural reconstruction time series
    parts.append('<hr class="section">')
    for n_spec in THESIS_NEURAL_TIMESERIES:
        try:
            fig_n, rmse_n, cap_n = compose_thesis_neural_figure(n_spec, results_root)
            add_fig(fig_n)
            parts.append(_df_to_html(rmse_n))
            parts.append(
                _p_caption(
                    f"<b>Neural reconstruction ({_escape(n_spec.section_title)}).</b> "
                    f"Observed ECoG (black) vs. model predictions. "
                    f"Shaded ribbons = session-mean RMSE around each prediction. "
                    f"{_escape(cap_n)}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build neural exemplar figure: {e}"))

    # Neural band heatmaps (moved from Generalisation)
    parts.append('<hr class="section">')
    for nb_spec in THESIS_NEURAL_BAND_HEATMAPS:
        try:
            nb_data = collect_neural_band_pearson(
                results_root,
                nb_spec.triplets,
                split=nb_spec.split,
                band_row_order=nb_spec.band_row_order,
            )
            if nb_data.n_triplets_used == 0:
                raise ValueError("Zero triplets passed filters.")
            fig_nb = build_neural_band_heatmap_figure(nb_data, nb_spec.theme)
            add_fig(fig_nb)
            parts.append(
                _p_caption(
                    "<b>Neural self-prediction (Pearson r by spectral band).</b> "
                    f"Pooled across {nb_data.n_triplets_used} session(s) "
                    f"({nb_data.n_trials_off} OFF / {nb_data.n_trials_on} ON trials). "
                    f"{_escape(nb_spec.caption or '')}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build neural band heatmap: {e}"))

    # Neural forecast RMSE vs horizon
    parts.append('<hr class="section">')
    for fc_spec in THESIS_NEURAL_FORECAST_FIGURES:
        try:
            fc_data = collect_forecast_horizon_rmse(
                results_root,
                fc_spec.triplets,
                channel_idx=fc_spec.channel_idx,
                split=fc_spec.split,
                sampling_hz=fc_spec.sampling_hz,
                sample_every=fc_spec.sample_every,
                naive_rmse=fc_spec.naive_rmse,
                forecast_target=fc_spec.forecast_target,
                neural_y_feature_name=fc_spec.neural_y_feature_name,
            )
            res_fc = load_split_results_required(
                results_root,
                fc_spec.triplets[0].psid_variant,
                fc_spec.triplets[0].psid_run_ts,
                fc_spec.split,
            )
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
            y_fc_title = rmse_axis_label(neu_lbl)
            fig_fc = build_forecast_rmse_figure_or_empty(
                fc_data,
                fc_spec.theme,
                y_axis_title=y_fc_title,
                column_name=neu_lbl,
            )
            add_fig(fig_fc)
            parts.append(
                _p_caption(
                    f"<b>Neural forecast RMSE vs. horizon ({_escape(neu_lbl)}).</b> "
                    f"PSID and VARMA multi-step prediction error, "
                    f"pooled across {fc_data.n_triplets_used} session(s) "
                    f"({fc_data.n_trials_off} OFF / {fc_data.n_trials_on} ON trials). "
                    f"Dashed line = na\u00efve (mean) baseline. Shaded bands = \u00b11 SEM.",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build neural forecast RMSE figure: {e}"))

    # Neural forecast exemplars
    parts.append('<hr class="section">')
    for c2_spec in c2_neural_specs:
        try:
            fig_c2, cap_c2 = build_c2_forecast_figure(c2_spec, results_root)
            add_fig(fig_c2)
            parts.append(
                _p_caption(
                    f"<b>Neural forecast exemplar ({_escape(c2_spec.section_title)}).</b> "
                    f"Left = history window, right = multi-step forecast. "
                    f"Shaded bands = session-mean RMSE. "
                    f"{_escape(cap_c2 or '')}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build neural forecast figure: {e}"))

    # Vanilla vs Improved PSID comparison (moved from Generalisation)
    parts.append('<hr class="section">')
    try:
        fig_vanilla = build_vanilla_comparison_figure()
        add_fig(fig_vanilla)
        parts.append(
            _p_caption(
                "<b>Improved PSID vs Vanilla.</b> "
                "Comparison of PSID with backward Kalman filter and A-matrix eigenvalue rescaling (improved) "
                "vs standard PSID without these enhancements (vanilla). "
                "Left: Pearson r for behavioral output. Right: RMSE for behavioral output.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build vanilla comparison figure: {e}"))

    parts.append(
        _transition(
            "Having validated neural reconstruction, we test whether surface ECoG can predict depth LFP signals."
        )
    )

    # ===================================================================
    # Section 3: RQ1 -- Cross-Modal Prediction (ECoG to LFP)
    # ===================================================================

    parts.append("<h2>3. RQ1 \u2014 Cross-Modal Prediction (ECoG to LFP)</h2>")
    parts.append(
        "<p>Can surface ECoG features predict depth LFP signals recorded at the DBS electrode? "
        "We train PSID with Laplacian-referenced LFP as the behavioural target, "
        "testing whether subdural recordings carry sufficient information about subcortical activity.</p>"
    )

    try:
        fig_lapl_bar = build_laplacian_prediction_figure()
        add_fig(fig_lapl_bar)
        parts.append(
            _p_caption(
                "<b>Laplacian LFP prediction summary.</b> "
                "Blue = ECoG self-reconstruction Pearson r (Y). "
                "Red = depth Laplacian LFP prediction Pearson r (Z). "
                "Surface ECoG contains limited information about depth LFP Laplacian. "
                "The best cross-modal prediction remains modest, suggesting that "
                "the subdural grid does not provide a reliable proxy for deep target activity.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build Laplacian prediction figure: {e}"))

    parts.append('<hr class="section">')
    try:
        fig_lapl_ts = build_laplacian_timeseries_figure()
        add_fig(fig_lapl_ts)
        parts.append(
            _p_caption(
                "<b>Laplacian LFP prediction time series (example trial).</b> "
                "True Laplacian signal vs. PSID prediction. "
                "The model captures some low-frequency trends but misses fine oscillatory structure.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build Laplacian time series figure: {e}"))

    parts.append(
        _transition(
            "Cross-modal prediction is limited; we now examine whether the models decode behavioural output from neural activity."
        )
    )

    # ===================================================================
    # Section 4: RQ2 -- Behavioral Decoding
    # ===================================================================

    parts.append("<h2>4. RQ2 \u2014 Behavioural Decoding</h2>")
    parts.append(
        "<p>Can latent neural dynamics predict concurrent behavioural output (tracing speed)? "
        "PSID and DPAD decode behaviour through learned latent states, while VARMA "
        "predicts directly from neural features without subspace decomposition.</p>"
    )

    # Behavioral decoding time series
    for spec in THESIS_FIGURES:
        try:
            fig, rmse_df, caption = compose_thesis_figure(spec, results_root)
        except Exception as e:
            parts.append(_p_error(f"Failed to build figure: {e}"))
            continue
        add_fig(fig)
        parts.append(_df_to_html(rmse_df))
        parts.append(
            _p_caption(
                f"<b>Behavioural decoding ({_escape(spec.section_title)}).</b> "
                f"Observed output (black) vs. model predictions. "
                f"Ribbons = session-mean RMSE. RMSE table above. "
                f"{_escape(caption)}",
                raw=True,
            )
        )

    # Behavioral forecast exemplars
    parts.append('<hr class="section">')
    for c2_spec in c2_z_specs:
        try:
            fig_c2, cap_c2 = build_c2_forecast_figure(c2_spec, results_root)
            add_fig(fig_c2)
            parts.append(
                _p_caption(
                    f"<b>Behavioural forecast ({_escape(c2_spec.section_title)}).</b> "
                    f"History window (left) and multi-step forecast (right). "
                    f"{_escape(cap_c2 or '')}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build behavioral forecast figure: {e}"))

    # Behavioral forecast RMSE vs horizon
    parts.append('<hr class="section">')
    for fc_spec in THESIS_FORECAST_FIGURES:
        try:
            fc_data = collect_forecast_horizon_rmse(
                results_root,
                fc_spec.triplets,
                channel_idx=fc_spec.channel_idx,
                split=fc_spec.split,
                sampling_hz=fc_spec.sampling_hz,
                sample_every=fc_spec.sample_every,
                naive_rmse=fc_spec.naive_rmse,
                forecast_target=fc_spec.forecast_target,
                neural_y_feature_name=fc_spec.neural_y_feature_name,
            )
            res_fc = load_split_results_required(
                results_root,
                fc_spec.triplets[0].psid_variant,
                fc_spec.triplets[0].psid_run_ts,
                fc_spec.split,
            )
            ch_fc, _dfc = resolve_output_channel_display(
                res_fc,
                fc_spec.channel_idx,
                declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
            )
            y_fc_title = rmse_axis_label(ch_fc)
            fig_fc = build_forecast_rmse_figure_or_empty(
                fc_data,
                fc_spec.theme,
                y_axis_title=y_fc_title,
                column_name=ch_fc,
            )
            add_fig(fig_fc)
            parts.append(
                _p_caption(
                    f"<b>Behavioural forecast RMSE vs. horizon ({_escape(ch_fc)}).</b> "
                    f"Pooled across {fc_data.n_triplets_used} session(s) "
                    f"({fc_data.n_trials_off} OFF / {fc_data.n_trials_on} ON trials). "
                    f"Shaded bands = \u00b11 SEM. Dashed = na\u00efve baseline. "
                    f"RMSE near 1.0 (z-scored baseline) indicates that decoding is modest; "
                    f"VARMA\u2019s competitive performance suggests that the latent decomposition "
                    f"trades accuracy for interpretability rather than improving prediction.",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build forecast RMSE figure: {e}"))

    parts.append(
        _transition(
            "Although behavioural decoding is modest, the latent representations may still encode DBS state, "
            "which we test next through classification and cross-condition generalisation."
        )
    )

    # ===================================================================
    # Section 5: RQ3 -- DBS Classification and Cross-Condition Generalisation
    # ===================================================================

    parts.append(
        "<h2>5. RQ3 \u2014 DBS Classification and Cross-Condition Generalisation</h2>"
    )
    parts.append(
        "<p>Do the learned latent states encode DBS stimulation state? "
        "We apply CSP + LDA classification to latent subspaces (X<sub>p</sub>, X<sub>p,1</sub>, X<sub>p,2</sub>) "
        "and test whether models trained on one DBS condition generalise to the other.</p>"
    )

    # Classification grouped bar chart
    for f1_spec in THESIS_CLASSIFICATION_F1:
        model_labels = sorted(
            {getattr(ref, "model_label", "PSID") for ref in f1_spec.points}
        )
        model_str = " / ".join(model_labels)
        try:
            cls_points = collect_classification_f1_points(
                results_root,
                f1_spec.points,
                classification_parent=f1_spec.classification_parent,
            )
            fig_cls_bar = build_classification_grouped_bar_figure(
                cls_points,
                theme=f1_spec.theme,
                exclude_groups={"xp_with_dbs"},
            )
            add_fig(fig_cls_bar)
            cap_lines: list[str] = []
            for pt in cls_points:
                pval_str = (
                    f"p = {pt.permutation_pvalue:.4f}"
                    if pt.permutation_pvalue is not None
                    else "\u2014"
                )
                cap_lines.append(
                    f"{pt.participant_label}_{pt.session_label} "
                    f"{_FEAT_SHORT.get(pt.group, pt.group)}: "
                    f"BA = {pt.balanced_accuracy:.3f} ({pval_str})"
                )
            parts.append(
                _p_caption(
                    f"<b>DBS classification \u2014 balanced accuracy ({_escape(model_str)} latent states).</b> "
                    f"CSP + LDA on held-out test set. Dashed red line = chance (0.5). "
                    f"{' | '.join(_escape(c) for c in cap_lines)}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(
                _p_error(f"Failed to build classification grouped bar chart: {e}")
            )

    # Standard classification heatmap
    parts.append('<hr class="section">')
    for f1_spec in THESIS_CLASSIFICATION_F1:
        try:
            from dashboard.thesis.classification_f1_figure import (
                build_standard_heatmap_figure,
            )

            cls_pts = collect_classification_f1_points(
                results_root,
                f1_spec.points,
                classification_parent=f1_spec.classification_parent,
            )
            fig_std_hm = build_standard_heatmap_figure(cls_pts, theme=f1_spec.theme)
            add_fig(fig_std_hm)
            parts.append(
                _p_caption(
                    "<b>Standard classification \u2014 balanced accuracy heatmap.</b> "
                    "Rows = participant-sessions. Columns = feature groups. "
                    "Chance level = 0.5.",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(
                _p_error(f"Failed to build standard classification heatmap: {e}")
            )

    # Flipped classification heatmaps
    parts.append('<hr class="section">')
    try:
        from dashboard.thesis.classification_f1_figure import (
            build_flipped_heatmap_figure,
        )

        flipped_fig = build_flipped_heatmap_figure(results_root)
        add_fig(flipped_fig)
        parts.append(
            _p_caption(
                "<b>Flipped classification \u2014 balanced accuracy (h \u00d7 m).</b> "
                "Rows = sessions, columns = feature groups. "
                "Latent states from condition-specific models (DBS-OFF, DBS-ON) "
                "are swapped to create a synthetic DBS mismatch. "
                "Chance level \u2248 0.5.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build flipped classification heatmaps: {e}"))

    # ROC curves (standard + flipped)
    parts.append('<hr class="section">')
    try:
        from dashboard.thesis.roc_curve_figure import build_roc_curve_figure

        fig_roc = build_roc_curve_figure(results_root)
        add_fig(fig_roc)
        parts.append(
            _p_caption(
                "<b>ROC curves \u2014 standard and flipped classification.</b> "
                "Mean ROC across sessions for each feature group. "
                "Dashed red = chance diagonal.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build ROC curve figure: {e}"))

    # Within vs cross-condition RMSE
    parts.append('<hr class="section">')
    for wc_spec in THESIS_WITHIN_CROSS:
        try:
            wc_data = collect_within_cross_rmse(
                results_root,
                [wc_spec.joint_triplet],
                wc_spec.channel_idx,
                split=wc_spec.split,
            )
            fig_b = build_within_cross_boxplot_figure(wc_data, theme=wc_spec.theme)
            add_fig(fig_b)
            parts.append(
                _p_caption(
                    f"<b>{_escape(wc_spec.section_title)}.</b> "
                    "1-step RMSE distribution for models trained on the same (within) vs. opposite (cross) "
                    "DBS condition, evaluated on held-out test trials.",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(
                _p_error(
                    f"Failed to build within-cross boxplot ({wc_spec.section_title}): {e}"
                )
            )

    # Cross-block decoding
    parts.append('<hr class="section">')
    for xb_spec in THESIS_CROSS_BLOCK:
        try:
            fig_xb, cap_xb = build_cross_block_predictions_figure(xb_spec, results_root)
            add_fig(fig_xb)
            parts.append(
                _p_caption(
                    "<b>Cross-block decoding.</b> "
                    "1 s segments around the OFF/ON block boundary. "
                    "Each row = one model; left = last OFF trial, right = first ON trial. "
                    "Coloured lines = predictions from OFF-trained, BOTH-trained, and ON-trained checkpoints. "
                    f"{_escape(cap_xb)}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build cross-block figure: {e}"))

    # Multi-step forecast checkpoints
    parts.append('<hr class="section">')
    for fc_ck_spec in THESIS_FORECAST_CHECKPOINT:
        try:
            fig_fc_ck, cap_fc_ck = build_forecast_checkpoint_compare_figure(
                fc_ck_spec, results_root
            )
            add_fig(fig_fc_ck)
            parts.append(
                _p_caption(
                    f"<b>Multi-step forecast \u2014 OFF / BOTH / ON checkpoints ({_escape(fc_ck_spec.section_title)}).</b> "
                    "Each row = one model; left = DBS-OFF trial, right = DBS-ON trial. "
                    "Lines = multi-step predictions from three training checkpoints. "
                    f"{_escape(cap_fc_ck or '')}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build forecast checkpoint figure: {e}"))

    parts.append(
        _transition(
            "We now synthesize findings across all three research questions and present supplementary analyses."
        )
    )

    # ===================================================================
    # Section 6: Summary, Supplementary Analysis, and Appendix
    # ===================================================================

    parts.append("<h2>6. Summary, Supplementary Analysis, and Appendix</h2>")
    parts.append(
        "<p><b>Synthesis.</b> "
        "Cross-modal prediction (RQ1) confirms that surface ECoG carries limited depth LFP information. "
        "Behavioural decoding (RQ2) shows RMSE near the z-scored baseline across models, with VARMA "
        "remaining competitive \u2014 the latent decomposition provides interpretability (condition-specific "
        "subspaces, latent phase separation) rather than accuracy gains. "
        "DBS classification (RQ3) yields balanced accuracies of 0.53\u20130.67 from latent states; "
        "the X<sub>p</sub>+DBS=1.0 result confirms that DBS artifacts dominate raw classification. "
        "The value of the subspace approach lies in its capacity to separate "
        "condition-specific neural dynamics, as shown by latent phase trajectories and "
        "cross-condition generalisation patterns, even when prediction accuracy is modest.</p>"
    )

    # Latent phase space
    parts.append('<hr class="section">')
    for lp_spec in THESIS_LATENT_PHASE:
        try:
            fig_lp, cap_lp = build_latent_phase_space_figure(lp_spec, results_root)
            add_fig(fig_lp)
            parts.append(
                _p_caption(
                    "<b>Latent phase space trajectories (PSID vs. DPAD).</b> "
                    "KDE density contours with sample trial trajectories per DBS condition. "
                    "PSID: x\u2081 vs x\u2082 (behaviourally relevant subspace). "
                    "DPAD: PC1 vs PC2. "
                    f"{_escape(cap_lp or '')}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build latent phase space figure: {e}"))

    # PSID ablation: nx/n1 sensitivity
    parts.append('<hr class="section">')
    try:
        fig_cls = build_classification_heatmap_figure()
        add_fig(fig_cls)
        parts.append(
            _p_caption(
                "<b>DBS classification accuracy vs latent dimensionality (n<sub>x</sub>, n<sub>1</sub>).</b> "
                "Balanced accuracy (LDA, 5-fold chronological CV) on X<sub>p</sub>, X<sub>p,1</sub>, X<sub>p,2</sub> "
                "across grid search (n<sub>x</sub> = 60\u201380) and ablation (n<sub>x</sub> = 1\u201380) runs. "
                "Red border = selected configuration.",
                raw=True,
            )
        )
    except Exception as e:
        parts.append(_p_error(f"Failed to build classification heatmap figure: {e}"))

    # PSID Cy importance
    parts.append('<hr class="section">')
    for cy_spec in THESIS_PSID_CY_IMPORTANCE:
        try:
            fig_cy, cap_cy = build_psid_cy_importance_figure(cy_spec, results_root)
            add_fig(fig_cy)
            parts.append(
                _p_caption(
                    "<b>PSID behaviourally relevant Cy importance.</b> "
                    "Heatmap of \u2016Cy[:, :n\u2081]\u2016 across ECoG contacts and spectral bands. "
                    f"{_escape(cap_cy or '')}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build PSID Cy importance figure: {e}"))

    # PSID Cz readout
    parts.append('<hr class="section">')
    for cz_spec in THESIS_PSID_CZ_HEATMAP:
        try:
            fig_cz, cap_cz = build_psid_cz_figure(cz_spec, results_root)
            add_fig(fig_cz)
            parts.append(
                _p_caption(
                    "<b>PSID Cz behavioural readout matrix.</b> "
                    f"{_escape(cap_cz or '')}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build PSID Cz heatmap figure: {e}"))

    # Appendix items
    parts.append('<hr class="section">')
    parts.append("<h3>Appendix</h3>")
    appendix_specs = [
        (
            "DPAD training curves (4-stage loss)",
            lambda: build_dpad_training_curves_figure(results_root),
            "Train (solid) and validation (dotted) loss per epoch across all 4 DPAD training stages "
            "(model1 \u2192 model1_Cy \u2192 model2 \u2192 model2_Cz). Vertical dashed lines mark stage boundaries. "
            "One panel per participant-session.",
        ),
        (
            "Data efficiency: PSID neural RMSE vs training set size",
            lambda: build_data_efficiency_figure(results_root),
            "PSID neural reconstruction RMSE as a function of the number of training trials, "
            "across 7 participant-sessions. Shaded bands = \u00b11 SEM. "
            "Rapid improvement up to ~20\u201330 trials suggests moderate data hungriness.",
        ),
    ]
    for title, builder, cap_extra in appendix_specs:
        try:
            fig = builder()
            add_fig(fig)
            parts.append(
                _p_caption(
                    f"<b>{_escape(title)}.</b> {_escape(cap_extra)}",
                    raw=True,
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build {title}: {e}"))

    parts.append("</main>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


def default_results_root(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    return Path(os.environ.get("RESULTS_PATH", root / "results"))


def write_thesis_html_report(
    output_path: Path,
    results_root: Path | None = None,
    *,
    project_root: Path | None = None,
    verbose_logging: bool = False,
    use_latest_result_timestamps: bool | None = None,
) -> Path:
    """Write the static HTML report to ``output_path``. Returns the path written."""
    rr = (
        results_root if results_root is not None else default_results_root(project_root)
    )
    proj = project_root or Path(__file__).resolve().parents[2]
    if use_latest_result_timestamps is None:
        use_latest_result_timestamps = (
            os.environ.get("THESIS_USE_SPEC_TIMESTAMPS", "").strip() != "1"
        )
    html_doc = build_thesis_html_document(
        rr,
        verbose_logging=verbose_logging,
        project_root=proj,
        use_latest_result_timestamps=use_latest_result_timestamps,
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path
