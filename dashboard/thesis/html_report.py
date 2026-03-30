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
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure
from plotly.io import to_html

from dashboard.thesis.aggregate_rmse import collect_pooled_rmse
from dashboard.thesis.c2_forecast_timeseries import build_c2_forecast_figure
from dashboard.thesis.cross_block_predictions import build_cross_block_predictions_figure
from dashboard.thesis.forecast_checkpoint_compare import build_forecast_checkpoint_compare_figure
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
    resolve_neural_y_channel_idx,
    resolve_output_channel_display,
)
from dashboard.thesis.latent_phase_space_figure import build_latent_phase_space_figure
from dashboard.thesis.neural_band_heatmap_figure import build_neural_band_heatmap_figure
from dashboard.thesis.neural_band_pearson import collect_neural_band_pearson
from dashboard.thesis.psid_cy_importance_figure import build_psid_cy_importance_figure
from dashboard.thesis.psid_cz_figure import build_psid_cz_figure
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
    build_psd_dbs_comparison_figure,
    build_tracing_speed_dbs_comparison_figure,
    build_grid_search_pearson_figure,
    build_grid_search_rmse_figure,
    build_grid_search_lag_figure,
    build_trial_count_summary_figure,
)
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
  --muted: #666660;
  --rule: #e0e0dc;
  --err-bg: #fce8e8;
  --err-fg: #b91c1c;
  --warn-bg: #fef3cd;
  --warn-fg: #856404;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.5;
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
p.caption, .caption { font-size: 0.84rem; color: var(--muted); margin: 0.4rem 0 0.75rem; }
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
.figure-wrap { margin: 0.5rem 0 1rem; width: 100%; overflow-x: auto; }
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


def _p_caption(text: str) -> str:
    return f'<p class="caption">{_escape(text)}</p>'


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
    parts.append("<title>Results</title>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<main>")

    c2_neural_specs = [s for s in THESIS_C2_FORECASTS if s.forecast_target == "Y"]
    c2_z_specs = [s for s in THESIS_C2_FORECASTS if s.forecast_target != "Y"]

    # ===================================================================
    # GROUP RESULTS — Per-session RMSE box plots (first section)
    # ===================================================================
    parts.append('<hr class="section">')
    parts.append("<h2>Per-session RMSE box plots (model × DBS)</h2>")
    parts.append(
        _p_caption(
            "One box plot per participant-session. Each figure uses only the trials from "
            "that single session. Box: IQR. Whiskers: 1.5×IQR. Dots: individual trials."
        )
    )
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
            parts.append(f"<h3>{_escape(session_label)} — {_escape(ch_ps)}</h3>")
            try:
                agg_single = collect_pooled_rmse(
                    results_root,
                    [tri],
                    spec.channel_idx,
                    split=spec.split,
                    run_wilcoxon=False,
                )
                rng_s = np.random.default_rng(spec.jitter_seed)
                fig_session = build_rmse_boxplot_figure(
                    agg_single,
                    spec.theme,
                    rng_s,
                    title=f"{session_label} — Trial RMSE by model × DBS ({ch_ps})",
                )
                add_fig(fig_session)
                n_trials = sum(len(c) for c in agg_single.trial_rmse)
                parts.append(
                    _p_caption(
                        f"Session {session_label}: {n_trials} total trial RMSE values "
                        f"(channel: {ch_ps})."
                    )
                )
            except Exception as e:
                parts.append(
                    _p_error(f"Failed to build per-session box plot for {session_label}: {e}")
                )

    # ===================================================================
    # GROUP RESULTS — Classification grouped bar chart (second section)
    # ===================================================================
    parts.append('<hr class="section">')
    parts.append("<h2>DBS classification — balanced accuracy by session</h2>")
    for f1_spec in THESIS_CLASSIFICATION_F1:
        # Determine model label(s) from the spec points
        model_labels = sorted({getattr(ref, "model_label", "PSID") for ref in f1_spec.points})
        model_str = " / ".join(model_labels)
        parts.append(f"<h3>{_escape(f1_spec.section_title)} ({_escape(model_str)} latent states)</h3>")
        try:
            cls_points = collect_classification_f1_points(
                results_root,
                f1_spec.points,
                classification_parent=f1_spec.classification_parent,
            )
            fig_cls_bar = build_classification_grouped_bar_figure(
                cls_points,
                theme=f1_spec.theme,
                title=f"DBS classification — balanced accuracy by session ({model_str})",
            )
            add_fig(fig_cls_bar)
            # Build stats caption from the data
            cap_lines: list[str] = []
            for pt in cls_points:
                pval_str = f"p = {pt.permutation_pvalue:.4f}" if pt.permutation_pvalue is not None else "permutation test not run"
                cap_lines.append(
                    f"{pt.participant_label}_{pt.session_label} "
                    f"{_FEAT_SHORT.get(pt.group, pt.group)}: "
                    f"BA = {pt.balanced_accuracy:.3f} ({pval_str})"
                )
            parts.append(
                _p_caption(
                    f"CSP + LDA on {model_str} latent states. "
                    "Balanced accuracy on held-out test set. "
                    "Chance level = 0.5 (dashed red line). "
                    + " | ".join(cap_lines)
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build classification grouped bar chart: {e}"))

    # ===================================================================
    # GROUP RESULTS — Flipped classification heatmaps (h × m grid)
    # ===================================================================
    parts.append('<hr class="section">')
    parts.append("<h2>Flipped classification — balanced accuracy (h × m)</h2>")
    parts.append(
        _p_caption(
            "Heatmap of balanced accuracy across history (h) and forecast (m) windows. "
            "Flipped protocol: latent states from condition-specific models (DBS-OFF, DBS-ON) "
            "are swapped to create a synthetic DBS mismatch. Chance level ≈ 0.5."
        )
    )
    try:
        from dashboard.thesis.classification_f1_figure import build_flipped_heatmap_figure
        flipped_fig = build_flipped_heatmap_figure(results_root)
        add_fig(flipped_fig)
    except Exception as e:
        parts.append(_p_error(f"Failed to build flipped classification heatmaps: {e}"))

    # ===================================================================
    # Individual figure sections follow
    # ===================================================================
    parts.append('<hr class="section">')
    parts.append("<h2>Neural exemplars (Y vs Ŷ)</h2>")
    if (DEFAULT_NEURAL_TS_CAPTION or "").strip():
        parts.append(_p_caption(DEFAULT_NEURAL_TS_CAPTION))
    for n_spec in THESIS_NEURAL_TIMESERIES:
        parts.append(f"<h3>{_escape(n_spec.section_title)}</h3>")
        try:
            fig_n, rmse_n, cap_n = compose_thesis_neural_figure(n_spec, results_root)
            add_fig(fig_n)
            parts.append('<p class="rmse-label">Per-trial RMSE on z-scored neural Y</p>')
            parts.append(_df_to_html(rmse_n))
            parts.append(_p_caption(cap_n))
        except Exception as e:
            parts.append(_p_error(f"Failed to build neural exemplar figure: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Figure C2 — Neural forecast (Y_future)</h2>")
    for c2_spec in c2_neural_specs:
        parts.append(f"<h3>{_escape(c2_spec.section_title)}</h3>")
        try:
            fig_c2, cap_c2 = build_c2_forecast_figure(c2_spec, results_root)
            add_fig(fig_c2)
            parts.append(_p_caption(cap_c2 or DEFAULT_C2_FORECAST_CAPTION))
        except Exception as e:
            parts.append(_p_error(f"Failed to build Figure C2: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Forecast RMSE vs horizon — neural Y</h2>")
    for fc_spec in THESIS_NEURAL_FORECAST_FIGURES:
        parts.append(f"<h3>{_escape(fc_spec.section_title)}</h3>")
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
            parts.append(
                _p_caption(
                    f"Triplets: {fc_data.n_triplets_used} · trials OFF/ON: {fc_data.n_trials_off} / {fc_data.n_trials_on}."
                )
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
            neu_lbl = inn[ch_ix] if ch_ix < len(inn) else f"Y column {ch_ix}"
            y_fc_title = f"RMSE (z-scored {neu_lbl})"
            fig_fc = build_forecast_rmse_figure_or_empty(
                fc_data, fc_spec.theme, y_axis_title=y_fc_title
            )
            add_fig(fig_fc)
            parts.append(_p_caption(fc_spec.caption or DEFAULT_NEURAL_FORECAST_CAPTION))
            parts.append(f"<h4>Global forecast RMSE at {fc_data.global_horizon_ms:.0f} ms</h4>")
            try:
                fig_fg = build_forecast_global_rmse_figure(
                    fc_data,
                    fc_spec.theme,
                    rng=np.random.default_rng(42),
                    y_axis_title=y_fc_title,
                )
                add_fig(fig_fg)
            except Exception as _eg:
                parts.append(_p_error(f"Failed to build global neural forecast RMSE figure: {_eg}"))
        except Exception as e:
            parts.append(_p_error(f"Failed to build neural forecast RMSE figure: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Time-series exemplars (behavioral Z)</h2>")
    for spec in THESIS_FIGURES:
        parts.append(f"<h3>{_escape(spec.section_title)}</h3>")
        try:
            fig, rmse_df, caption = compose_thesis_figure(spec, results_root)
        except Exception as e:
            parts.append(_p_error(f"Failed to build figure: {e}"))
            continue
        add_fig(fig)
        parts.append('<p class="rmse-label">Per-trial RMSE (z-scored)</p>')
        parts.append(_df_to_html(rmse_df))
        parts.append(_p_caption(caption))

    parts.append('<hr class="section">')
    parts.append("<h2>Figure C2 — Behavioral forecast (Z_future)</h2>")
    for c2_spec in c2_z_specs:
        parts.append(f"<h3>{_escape(c2_spec.section_title)}</h3>")
        try:
            fig_c2, cap_c2 = build_c2_forecast_figure(c2_spec, results_root)
            add_fig(fig_c2)
            parts.append(_p_caption(cap_c2 or DEFAULT_C2_FORECAST_CAPTION))
        except Exception as e:
            parts.append(_p_error(f"Failed to build Figure C2: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Cross-block decoding (OFF↔ON trial stitches)</h2>")
    parts.append(_p_caption("1 s segments around the OFF/ON block boundary; see each figure title for layout."))
    for xb_spec in THESIS_CROSS_BLOCK:
        parts.append(f"<h3>{_escape(xb_spec.section_title)}</h3>")
        try:
            fig_xb, cap_xb = build_cross_block_predictions_figure(xb_spec, results_root)
            add_fig(fig_xb)
            parts.append(_p_caption(cap_xb))
        except Exception as e:
            parts.append(_p_error(f"Failed to build cross-block figure: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Multi-step forecast — OFF / BOTH / ON checkpoints</h2>")
    parts.append(
        _p_caption(
            "One column per DBS trial (same indices as cross-block); within each cell: three trained checkpoints."
        )
    )
    for fc_ck_spec in THESIS_FORECAST_CHECKPOINT:
        parts.append(f"<h3>{_escape(fc_ck_spec.section_title)}</h3>")
        try:
            fig_fc_ck, cap_fc_ck = build_forecast_checkpoint_compare_figure(fc_ck_spec, results_root)
            add_fig(fig_fc_ck)
            parts.append(_p_caption(cap_fc_ck or DEFAULT_FORECAST_CHECKPOINT_CAPTION))
        except Exception as e:
            parts.append(_p_error(f"Failed to build forecast checkpoint figure: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>PSID — behaviourally relevant Cy importance</h2>")
    for cy_spec in THESIS_PSID_CY_IMPORTANCE:
        parts.append(f"<h3>{_escape(cy_spec.section_title)}</h3>")
        n_cells = sum(len(r.panels) for r in cy_spec.rows)
        parts.append(
            _p_caption(
                f"Panels in spec: {n_cells} (participant × session). "
                "Edit PsidCyRow / PsidCyPanel in dashboard/thesis/specs.py."
            )
        )
        try:
            fig_cy, cap_cy = build_psid_cy_importance_figure(cy_spec, results_root)
            add_fig(fig_cy)
            parts.append(_p_caption(cap_cy or DEFAULT_PSID_CY_IMPORTANCE_CAPTION))
        except Exception as e:
            parts.append(_p_error(f"Failed to build PSID Cy importance figure: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>PSID — Cz behavioural readout</h2>")
    for cz_spec in THESIS_PSID_CZ_HEATMAP:
        parts.append(f"<h3>{_escape(cz_spec.section_title)}</h3>")
        n_cells = sum(len(r.panels) for r in cz_spec.rows)
        parts.append(
            _p_caption(
                f"Panels in spec: {n_cells} (participant × session). "
                "Edit ThesisPsidCzSpec in dashboard/thesis/specs.py."
            )
        )
        try:
            fig_cz, cap_cz = build_psid_cz_figure(cz_spec, results_root)
            add_fig(fig_cz)
            parts.append(_p_caption(cap_cz or DEFAULT_PSID_CZ_CAPTION))
        except Exception as e:
            parts.append(_p_error(f"Failed to build PSID Cz heatmap figure: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Latent phase space (PSID vs DPAD)</h2>")
    for lp_spec in THESIS_LATENT_PHASE:
        parts.append(f"<h3>{_escape(lp_spec.section_title)}</h3>")
        n_cells = sum(len(r.panels) for r in lp_spec.rows)
        parts.append(
            _p_caption(
                f"Panels: {n_cells} (participant × session). "
                "Edit LatentPhaseRow / LatentPhasePanel in dashboard/thesis/specs.py."
            )
        )
        try:
            fig_lp, cap_lp = build_latent_phase_space_figure(lp_spec, results_root)
            add_fig(fig_lp)
            parts.append(_p_caption(cap_lp or DEFAULT_LATENT_PHASE_CAPTION))
        except Exception as e:
            parts.append(_p_error(f"Failed to build latent phase space figure: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>DBS classification — Figure F1 (balanced accuracy)</h2>")
    for f1_spec in THESIS_CLASSIFICATION_F1:
        parts.append(f"<h3>{_escape(f1_spec.section_title)}</h3>")
        n_pts = len(f1_spec.points)
        parts.append(
            _p_caption(
                f"Pickle refs in spec: {n_pts}. "
                "Edit ClassificationF1PickleRef in THESIS_CLASSIFICATION_F1 in dashboard/thesis/specs.py."
            )
        )
        try:
            fig_f1, cap_f1 = build_classification_f1_figure(f1_spec, results_root)
            add_fig(fig_f1)
            parts.append(_p_caption(cap_f1 or DEFAULT_CLASSIFICATION_F1_CAPTION))
        except Exception as e:
            parts.append(_p_error(f"Failed to build Figure F1: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Pooled test-set RMSE (model × DBS)</h2>")
    for spec in THESIS_AGGREGATE_FIGURES:
        parts.append(f"<h3>{_escape(spec.section_title)}</h3>")
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
        parts.append(
            _p_caption(
                f"Aligned triplets: {len(spec.triplets)}. "
                f"RMSE uses output channel index {spec.channel_idx} ({ch_agg}), "
                "z-scored per trial using true-trace statistics. "
                "PSID, DPAD, and VARMA must share identical trial keys on the test split."
            )
        )
        try:
            agg = collect_pooled_rmse(
                results_root,
                spec.triplets,
                spec.channel_idx,
                split=spec.split,
                run_wilcoxon=spec.run_wilcoxon,
            )
            parts.append(
                _p_caption(
                    f"Triplets loaded successfully: {agg.n_triplets_used} "
                    "(zero means no overlapping trial keys across PSID/DPAD/VARMA)."
                )
            )
            cap = spec.caption or DEFAULT_AGGREGATE_CAPTION
            rng = np.random.default_rng(spec.jitter_seed)
            fig_b = build_rmse_distribution_figure(
                agg,
                spec.theme,
                rng,
                show_brackets=spec.show_brackets,
            )
            add_fig(fig_b)
            parts.append(_p_caption(cap))
            fig_box = build_rmse_boxplot_figure(agg, spec.theme, rng)
            add_fig(fig_box)
            parts.append(
                _p_caption(
                    "Box: IQR. Whiskers: 1.5×IQR. "
                    "Dots: individual trials; colour = participant (see legend on the box plot: "
                    "green ≈ PDI4, blue ≈ PDI1 in the default colour map)."
                )
            )
        except Exception as e:
            parts.append(_p_error(f"Failed to build aggregate figure: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Session-mean RMSE strip plots (per participant)</h2>")
    for strip_spec in THESIS_STRIP_PANELS:
        parts.append(f"<h3>{_escape(strip_spec.section_title)}</h3>")
        parts.append(
            _p_caption(
                f"Panels in spec: {len(strip_spec.panels)}, grid: {strip_spec.ncols} columns. "
                "Replace StripPanelEntry rows in dashboard/thesis/specs.py with real triplets per participant."
            )
        )
        try:
            panel_triplets = [(e.panel_label, e.triplet) for e in strip_spec.panels]
            strip_data = collect_strip_figure_data(
                results_root,
                panel_triplets,
                strip_spec.channel_idx,
                split=strip_spec.split,
            )
            if strip_data is None or not strip_data.panels:
                raise ValueError(
                    "Strip plot: no panels (missing results or no overlapping trial keys across models)."
                )
            rng = np.random.default_rng(strip_spec.jitter_seed)
            fig_s = build_session_strip_boxplot_figure(
                strip_data,
                ncols=strip_spec.ncols,
                theme=strip_spec.theme,
                rng=rng,
            )
            add_fig(fig_s)
            parts.append(_p_caption(strip_spec.caption or DEFAULT_STRIP_CAPTION))
        except Exception as e:
            parts.append(_p_error(f"Failed to build strip plot: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Forecast RMSE vs horizon (PSID vs VARMA)</h2>")
    for fc_spec in THESIS_FORECAST_FIGURES:
        parts.append(f"<h3>{_escape(fc_spec.section_title)}</h3>")
        parts.append(_p_caption(f"Aligned triplets: {len(fc_spec.triplets)}."))
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
            parts.append(
                _p_caption(
                    f"Triplets loaded: {fc_data.n_triplets_used} · "
                    f"trials OFF / ON: {fc_data.n_trials_off} / {fc_data.n_trials_on}."
                )
            )
            res_fc = load_split_results_required(
                results_root,
                fc_spec.triplets[0].psid_variant,
                fc_spec.triplets[0].psid_run_ts,
                fc_spec.split,
            )
            ch_fc, _dfc = resolve_output_channel_display(
                res_fc, fc_spec.channel_idx, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS
            )
            y_fc_title = f"RMSE (z-scored {ch_fc})"
            fig_fc = build_forecast_rmse_figure_or_empty(
                fc_data, fc_spec.theme, y_axis_title=y_fc_title
            )
            add_fig(fig_fc)
            parts.append(_p_caption(fc_spec.caption or DEFAULT_FORECAST_CAPTION))
            parts.append(f"<h4>Global forecast RMSE at {fc_data.global_horizon_ms:.0f} ms</h4>")
            try:
                fig_fg = build_forecast_global_rmse_figure(
                    fc_data,
                    fc_spec.theme,
                    rng=np.random.default_rng(42),
                    y_axis_title=y_fc_title,
                )
                add_fig(fig_fg)
            except Exception as _eg:
                parts.append(_p_error(f"Failed to build global forecast RMSE figure: {_eg}"))
        except Exception as e:
            parts.append(_p_error(f"Failed to build forecast RMSE figure: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Neural self-prediction (band × model × DBS)</h2>")
    for nb_spec in THESIS_NEURAL_BAND_HEATMAPS:
        parts.append(f"<h3>{_escape(nb_spec.section_title)}</h3>")
        parts.append(
            _p_caption(
                f"Aligned triplets: {len(nb_spec.triplets)}. "
                "Requires Y and Ŷ and input_channels with ECoG/LFP narrow-band names in saved test parquets."
            )
        )
        try:
            nb_data = collect_neural_band_pearson(
                results_root,
                nb_spec.triplets,
                split=nb_spec.split,
                band_row_order=nb_spec.band_row_order,
            )
            parts.append(
                _p_caption(
                    f"Triplets with data: {nb_data.n_triplets_used} · "
                    f"trials pooled (OFF / ON): {nb_data.n_trials_off} / {nb_data.n_trials_on}."
                )
            )
            if nb_data.n_triplets_used == 0:
                raise ValueError(
                    "Neural band heatmap: zero triplets passed filters (check Y/Ŷ overlap and input_channels)."
                )
            fig_nb = build_neural_band_heatmap_figure(nb_data, nb_spec.theme)
            add_fig(fig_nb)
            parts.append(_p_caption(nb_spec.caption or DEFAULT_NEURAL_BAND_CAPTION))
        except Exception as e:
            parts.append(_p_error(f"Failed to build neural band heatmap: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Within vs cross-condition decoding</h2>")
    parts.append(
        _p_caption(
            "Neural-focused section order: band heatmaps above; here, RMSE summary first, then trial-level "
            "behavioral decoding. Part B: RMSE box plot per model (within vs cross). "
            "Part A: four panels (see caption under the figure for layout, time window, and RMSE). "
            "Strict timestamps: each AlignedTriplet must set psid/dpad/varma *_run_ts_off and *_run_ts_on "
            "plus joint timestamps for eval_* folders."
        )
    )
    parts.append("<h3>Part B — 1-step RMSE distribution</h3>")
    try:
        wc_data = collect_within_cross_rmse(
            results_root,
            [THESIS_WITHIN_CROSS.joint_triplet],
            THESIS_WITHIN_CROSS.channel_idx,
            split=THESIS_WITHIN_CROSS.split,
        )
        fig_b = build_within_cross_boxplot_figure(wc_data, theme=THESIS_WITHIN_CROSS.theme)
        add_fig(fig_b)
    except Exception as e:
        parts.append(_p_error(f"Failed to build within-cross boxplot: {e}"))
    parts.append("<h3>Part A — Trial-level decoding</h3>")
    try:
        fig_a, cap_within_cross_ts = build_within_cross_timeseries_figure(
            THESIS_WITHIN_CROSS, results_root, theme=THESIS_WITHIN_CROSS.theme
        )
        add_fig(fig_a)
        if cap_within_cross_ts:
            parts.append(_p_caption(cap_within_cross_ts))
    except Exception as e:
        parts.append(_p_error(f"Failed to build within-cross timeseries: {e}"))

    parts.append('<hr class="section">')
    parts.append("<h2>Appendix — Data characterisation and model selection</h2>")
    appendix_specs = [
        (
            "ECoG PSD: DBS-ON vs DBS-OFF",
            lambda: build_psd_dbs_comparison_figure(),
            "Filled ribbons: ±1 SEM across trials in each DBS condition. Lines: mean PSD. "
            "Vertical dashed lines: 13–29 Hz beta band edges.",
        ),
        (
            "Tracing speed: DBS-ON vs DBS-OFF",
            lambda: build_tracing_speed_dbs_comparison_figure(),
            "Solid lines only: mean z-scored tracing speed per DBS condition, averaged across trials in each panel "
            "(inter-trial SEM is omitted here for clarity).",
        ),
        ("PSID grid search: validation Pearson r", lambda: build_grid_search_pearson_figure(), None),
        ("PSID grid search: validation RMSE", lambda: build_grid_search_rmse_figure(), None),
        ("PSID grid search: validation lag (ms)", lambda: build_grid_search_lag_figure(), None),
        ("Trial count per session × DBS condition", lambda: build_trial_count_summary_figure(), None),
    ]
    for title, builder, cap_extra in appendix_specs:
        try:
            fig = builder()
            parts.append(f"<h3>{_escape(title)}</h3>")
            add_fig(fig)
            if cap_extra:
                parts.append(_p_caption(cap_extra))
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
    rr = results_root if results_root is not None else default_results_root(project_root)
    proj = project_root or Path(__file__).resolve().parents[2]
    if use_latest_result_timestamps is None:
        use_latest_result_timestamps = os.environ.get("THESIS_USE_SPEC_TIMESTAMPS", "").strip() != "1"
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
