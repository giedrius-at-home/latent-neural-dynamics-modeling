"""
Thesis final-results dashboard: specs in `dashboard/thesis/specs.py`, with run timestamps resolved
to the latest on-disk results per variant (set env `THESIS_USE_SPEC_TIMESTAMPS=1` to disable).

View-transition CSS (`::view-transition-*`) applies to full-document navigation in browsers;
Streamlit does not use that API, so those rules have no effect here. Static HTML with the same
sections: `python scripts/generate_thesis_html.py` → `thesis_results.html`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

results_root = Path(os.environ.get("RESULTS_PATH", project_root / "results"))

import numpy as np
import streamlit as st

from dashboard.thesis.aggregate_rmse import collect_pooled_rmse
from dashboard.thesis.c2_forecast_timeseries import build_c2_forecast_figure
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
from dashboard.thesis.neural_band_heatmap_figure import build_neural_band_heatmap_figure
from dashboard.thesis.neural_band_pearson import collect_neural_band_pearson
from dashboard.thesis.classification_f1_figure import (
    build_classification_f1_figure,
    build_flipped_heatmap_figure,
)
from dashboard.thesis.latent_phase_space_figure import build_latent_phase_space_figure
from dashboard.thesis.psid_cy_importance_figure import build_psid_cy_importance_figure
from dashboard.thesis.psid_cz_figure import build_psid_cz_figure
from dashboard.thesis.rmse_distribution_figure import (
    build_rmse_boxplot_figure,
    build_rmse_distribution_figure,
)
from dashboard.thesis.session_strip_figure import build_session_strip_boxplot_figure
from dashboard.thesis.session_strip_rmse import collect_strip_figure_data
from dashboard.thesis.aggregate_rmse import collect_within_cross_rmse
from dashboard.thesis.classification_f1_data import (
    collect_classification_f1_points,
    collect_classification_roc_curves,
)
from dashboard.thesis.session_grouped_figure import build_session_grouped_figure
from dashboard.thesis.specs import (
    DEFAULT_AGGREGATE_CAPTION,
    DEFAULT_C2_FORECAST_CAPTION,
    DEFAULT_FORECAST_CAPTION,
    DEFAULT_NEURAL_BAND_CAPTION,
    DEFAULT_NEURAL_FORECAST_CAPTION,
    DEFAULT_NEURAL_TS_CAPTION,
    DEFAULT_CLASSIFICATION_F1_CAPTION,
    DEFAULT_LATENT_PHASE_CAPTION,
    DEFAULT_PSID_CY_IMPORTANCE_CAPTION,
    DEFAULT_PSID_CZ_CAPTION,
    DEFAULT_STRIP_CAPTION,
    THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
)
import dashboard.thesis.specs as thesis_specs

if os.environ.get("THESIS_USE_SPEC_TIMESTAMPS", "").strip() == "1":
    THESIS_FIGURES = thesis_specs.THESIS_FIGURES
    THESIS_NEURAL_TIMESERIES = thesis_specs.THESIS_NEURAL_TIMESERIES
    THESIS_C2_FORECASTS = thesis_specs.THESIS_C2_FORECASTS
    THESIS_NEURAL_FORECAST_FIGURES = thesis_specs.THESIS_NEURAL_FORECAST_FIGURES
    THESIS_CLASSIFICATION_F1 = thesis_specs.THESIS_CLASSIFICATION_F1
    THESIS_AGGREGATE_FIGURES = thesis_specs.THESIS_AGGREGATE_FIGURES
    THESIS_STRIP_PANELS = thesis_specs.THESIS_STRIP_PANELS
    THESIS_FORECAST_FIGURES = thesis_specs.THESIS_FORECAST_FIGURES
    THESIS_NEURAL_BAND_HEATMAPS = thesis_specs.THESIS_NEURAL_BAND_HEATMAPS
    THESIS_LATENT_PHASE = thesis_specs.THESIS_LATENT_PHASE
    THESIS_PSID_CY_IMPORTANCE = thesis_specs.THESIS_PSID_CY_IMPORTANCE
    THESIS_PSID_CZ_HEATMAP = thesis_specs.THESIS_PSID_CZ_HEATMAP
else:
    from dashboard.thesis.timestamp_discovery import build_thesis_dashboard_specs

    _td = build_thesis_dashboard_specs(results_root, split="test")
    THESIS_FIGURES = _td.figures
    THESIS_NEURAL_TIMESERIES = _td.neural_timeseries
    THESIS_C2_FORECASTS = _td.c2_forecasts
    THESIS_NEURAL_FORECAST_FIGURES = _td.neural_forecast_figures
    THESIS_CLASSIFICATION_F1 = _td.classification_f1
    THESIS_AGGREGATE_FIGURES = _td.aggregate_figures
    THESIS_STRIP_PANELS = _td.strip_panels
    THESIS_FORECAST_FIGURES = _td.forecast_figures
    THESIS_NEURAL_BAND_HEATMAPS = _td.neural_band_heatmaps
    THESIS_LATENT_PHASE = _td.latent_phase
    THESIS_PSID_CY_IMPORTANCE = _td.psid_cy_importance
    THESIS_PSID_CZ_HEATMAP = _td.psid_cz_heatmap

st.set_page_config(layout="wide", page_title="Thesis results")

c2_neural_specs = [s for s in THESIS_C2_FORECASTS if s.forecast_target == "Y"]
c2_z_specs = [s for s in THESIS_C2_FORECASTS if s.forecast_target != "Y"]

st.markdown("## Neural exemplars (Y vs Ŷ)")
if (DEFAULT_NEURAL_TS_CAPTION or "").strip():
    st.caption(DEFAULT_NEURAL_TS_CAPTION)
for n_spec in THESIS_NEURAL_TIMESERIES:
    st.markdown(f"### {n_spec.section_title}")
    try:
        fig_n, rmse_n, cap_n = compose_thesis_neural_figure(n_spec, results_root)
        st.plotly_chart(fig_n, use_container_width=True)
        st.markdown("**Per-trial RMSE on z-scored neural Y**")
        st.dataframe(rmse_n, hide_index=True, use_container_width=False)
        st.caption(cap_n)
    except Exception as e:
        st.error(f"Failed to build neural exemplar figure: {e}")

st.markdown("---")
st.markdown("## Figure C2 — Neural forecast (Y_future)")

for c2_spec in c2_neural_specs:
    st.markdown(f"### {c2_spec.section_title}")
    try:
        fig_c2, cap_c2 = build_c2_forecast_figure(c2_spec, results_root)
        st.plotly_chart(fig_c2, use_container_width=True)
        st.caption(cap_c2 or DEFAULT_C2_FORECAST_CAPTION)
    except Exception as e:
        st.error(f"Failed to build Figure C2: {e}")

st.markdown("---")
st.markdown("## Forecast RMSE vs horizon — neural Y")

for fc_spec in THESIS_NEURAL_FORECAST_FIGURES:
    st.markdown(f"### {fc_spec.section_title}")
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
        st.caption(
            f"Triplets loaded: **{fc_data.n_triplets_used}** · "
            f"trials OFF / ON: **{fc_data.n_trials_off}** / **{fc_data.n_trials_on}**."
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
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption(fc_spec.caption or DEFAULT_NEURAL_FORECAST_CAPTION)
        st.markdown(f"#### Global forecast RMSE at {fc_data.global_horizon_ms:.0f} ms")
        try:
            fig_fg = build_forecast_global_rmse_figure(
                fc_data,
                fc_spec.theme,
                rng=np.random.default_rng(42),
                y_axis_title=y_fc_title,
            )
            st.plotly_chart(fig_fg, use_container_width=True)
        except Exception as _eg:
            st.error(f"Failed to build global neural forecast RMSE figure: {_eg}")
    except Exception as e:
        st.error(f"Failed to build neural forecast RMSE figure: {e}")

st.markdown("---")
st.markdown("## Time-series exemplars (behavioral Z)")
for spec in THESIS_FIGURES:
    st.markdown(f"### {spec.section_title}")
    try:
        fig, rmse_df, caption = compose_thesis_figure(spec, results_root)
    except Exception as e:
        st.error(f"Failed to build figure: {e}")
        continue

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Per-trial RMSE (z-scored)**")
    st.dataframe(rmse_df, hide_index=True, use_container_width=False)
    st.caption(caption)

st.markdown("---")
st.markdown("## Figure C2 — Behavioral forecast (Z_future)")

for c2_spec in c2_z_specs:
    st.markdown(f"### {c2_spec.section_title}")
    try:
        fig_c2, cap_c2 = build_c2_forecast_figure(c2_spec, results_root)
        st.plotly_chart(fig_c2, use_container_width=True)
        st.caption(cap_c2 or DEFAULT_C2_FORECAST_CAPTION)
    except Exception as e:
        st.error(f"Failed to build Figure C2: {e}")

st.markdown("---")
st.markdown("## PSID — behaviourally relevant Cy importance")

for cy_spec in THESIS_PSID_CY_IMPORTANCE:
    st.markdown(f"### {cy_spec.section_title}")
    n_cells = sum(len(r.panels) for r in cy_spec.rows)
    st.caption(
        f"Panels in spec: **{n_cells}** (participant × session). "
        "Edit `PsidCyRow` / `PsidCyPanel` in `dashboard/thesis/specs.py`."
    )
    try:
        fig_cy, cap_cy = build_psid_cy_importance_figure(cy_spec, results_root)
        st.plotly_chart(fig_cy, use_container_width=True)
        st.caption(cap_cy or DEFAULT_PSID_CY_IMPORTANCE_CAPTION)
    except Exception as e:
        st.error(f"Failed to build PSID Cy importance figure: {e}")

st.markdown("---")
st.markdown("## PSID — Cz behavioural readout")

for cz_spec in THESIS_PSID_CZ_HEATMAP:
    st.markdown(f"### {cz_spec.section_title}")
    n_cells = sum(len(r.panels) for r in cz_spec.rows)
    st.caption(
        f"Panels in spec: **{n_cells}** (participant × session). "
        "Edit `ThesisPsidCzSpec` in `dashboard/thesis/specs.py`."
    )
    try:
        fig_cz, cap_cz = build_psid_cz_figure(cz_spec, results_root)
        st.plotly_chart(fig_cz, use_container_width=True)
        st.caption(cap_cz or DEFAULT_PSID_CZ_CAPTION)
    except Exception as e:
        st.error(f"Failed to build PSID Cz heatmap figure: {e}")

st.markdown("---")
st.markdown("## Latent phase space (PSID vs DPAD)")

for lp_spec in THESIS_LATENT_PHASE:
    st.markdown(f"### {lp_spec.section_title}")
    n_cells = sum(len(r.panels) for r in lp_spec.rows)
    st.caption(
        f"Panels: **{n_cells}** (participant × session). "
        "Edit `LatentPhaseRow` / `LatentPhasePanel` in `dashboard/thesis/specs.py`."
    )
    try:
        fig_lp, cap_lp = build_latent_phase_space_figure(lp_spec, results_root)
        st.plotly_chart(fig_lp, use_container_width=True)
        st.caption(cap_lp or DEFAULT_LATENT_PHASE_CAPTION)
    except Exception as e:
        st.error(f"Failed to build latent phase space figure: {e}")

st.markdown("---")
st.markdown("## DBS classification — Figure F1 (balanced accuracy)")

for f1_spec in THESIS_CLASSIFICATION_F1:
    st.markdown(f"### {f1_spec.section_title}")
    n_pts = len(f1_spec.points)
    st.caption(
        f"Pickle refs in spec: **{n_pts}**. "
        "Edit `ClassificationF1PickleRef` in `THESIS_CLASSIFICATION_F1` in `dashboard/thesis/specs.py`."
    )
    try:
        fig_f1, cap_f1 = build_classification_f1_figure(f1_spec, results_root)
        st.plotly_chart(fig_f1, use_container_width=True)
        st.caption(cap_f1 or DEFAULT_CLASSIFICATION_F1_CAPTION)
    except Exception as e:
        st.error(f"Failed to build Figure F1: {e}")

st.markdown("---")
st.markdown("## Pooled test-set RMSE (model × DBS)")

for spec in THESIS_AGGREGATE_FIGURES:
    st.markdown(f"### {spec.section_title}")
    st.caption(
        f"Aligned triplets in spec: **{len(spec.triplets)}**. "
        "Add `AlignedTriplet` rows in `dashboard/thesis/specs.py` to pool more participants/sessions."
    )
    try:
        agg = collect_pooled_rmse(
            results_root,
            spec.triplets,
            spec.channel_idx,
            split=spec.split,
            run_wilcoxon=spec.run_wilcoxon,
        )
        st.caption(
            f"Triplets loaded successfully: **{agg.n_triplets_used}** "
            "(zero means no overlapping trial keys across PSID/DPAD/VARMA)."
        )
        cap = spec.caption or DEFAULT_AGGREGATE_CAPTION
        rng = np.random.default_rng(spec.jitter_seed)
        fig_b = build_rmse_distribution_figure(
            agg,
            spec.theme,
            rng,
            show_brackets=spec.show_brackets,
        )
        st.plotly_chart(fig_b, use_container_width=True)
        st.caption(cap)
        fig_box = build_rmse_boxplot_figure(agg, spec.theme, rng)
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption(
            "Box: IQR. Whiskers: 1.5×IQR. "
            "Dots: trials coloured by participant (legend lists PDI1 / PDI4 colours; green ≈ PDI4, blue ≈ PDI1)."
        )
    except Exception as e:
        st.error(f"Failed to build aggregate figure: {e}")

st.markdown("---")
st.markdown("## Session-mean RMSE strip plots (per participant)")

for strip_spec in THESIS_STRIP_PANELS:
    st.markdown(f"### {strip_spec.section_title}")
    st.caption(
        f"Panels in spec: **{len(strip_spec.panels)}**, grid: **{strip_spec.ncols}** columns. "
        "Replace `StripPanelEntry` rows in `dashboard/thesis/specs.py` with real triplets per participant."
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
                "Strip plot: no panels (missing results or no overlapping trial keys)."
            )
        rng = np.random.default_rng(strip_spec.jitter_seed)
        fig_s = build_session_strip_boxplot_figure(
            strip_data,
            ncols=strip_spec.ncols,
            theme=strip_spec.theme,
            rng=rng,
        )
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption(strip_spec.caption or DEFAULT_STRIP_CAPTION)
    except Exception as e:
        st.error(f"Failed to build strip plot: {e}")

st.markdown("---")
st.markdown("## Forecast RMSE vs horizon (PSID vs VARMA)")

for fc_spec in THESIS_FORECAST_FIGURES:
    st.markdown(f"### {fc_spec.section_title}")
    st.caption(
        f"Aligned triplets: **{len(fc_spec.triplets)}** (PSID + VARMA only; uses `Z_future_*` in test parquet)."
    )
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
        st.caption(
            f"Triplets loaded: **{fc_data.n_triplets_used}** · "
            f"trials OFF / ON: **{fc_data.n_trials_off}** / **{fc_data.n_trials_on}**."
        )
        res_fc = load_split_results_required(
            results_root,
            fc_spec.triplets[0].psid_variant,
            fc_spec.triplets[0].psid_run_ts,
            fc_spec.split,
        )
        ch_fc, _ = resolve_output_channel_display(
            res_fc, fc_spec.channel_idx, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS
        )
        y_fc_title = f"RMSE (z-scored {ch_fc})"
        fig_fc = build_forecast_rmse_figure_or_empty(
            fc_data, fc_spec.theme, y_axis_title=y_fc_title
        )
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption(fc_spec.caption or DEFAULT_FORECAST_CAPTION)
        st.markdown(f"#### Global forecast RMSE at {fc_data.global_horizon_ms:.0f} ms")
        try:
            fig_fg = build_forecast_global_rmse_figure(
                fc_data,
                fc_spec.theme,
                rng=np.random.default_rng(42),
                y_axis_title=y_fc_title,
            )
            st.plotly_chart(fig_fg, use_container_width=True)
        except Exception as _eg:
            st.error(f"Failed to build global forecast RMSE figure: {_eg}")
    except Exception as e:
        st.error(f"Failed to build forecast RMSE figure: {e}")

st.markdown("---")
st.markdown("## Neural self-prediction (band × model × DBS)")

for nb_spec in THESIS_NEURAL_BAND_HEATMAPS:
    st.markdown(f"### {nb_spec.section_title}")
    st.caption(
        f"Aligned triplets in spec: **{len(nb_spec.triplets)}**. "
    )
    try:
        nb_data = collect_neural_band_pearson(
            results_root,
            nb_spec.triplets,
            split=nb_spec.split,
            band_row_order=nb_spec.band_row_order,
        )
        st.caption(
            f"Triplets with data: **{nb_data.n_triplets_used}** · "
            f"trials pooled (OFF / ON): **{nb_data.n_trials_off}** / **{nb_data.n_trials_on}**."
        )
        if nb_data.n_triplets_used == 0:
            raise ValueError(
                "Neural band heatmap: zero triplets (check Y/Ŷ overlap and input_channels)."
            )
        fig_nb = build_neural_band_heatmap_figure(nb_data, nb_spec.theme)
        st.plotly_chart(fig_nb, use_container_width=True)
        st.caption(nb_spec.caption or DEFAULT_NEURAL_BAND_CAPTION)
    except Exception as e:
        st.error(f"Failed to build neural band heatmap: {e}")

st.markdown("---")
st.markdown("## Grouped session-level comparison (RMSE · ROC · Forecast)")
st.caption(
    "Panels A/B: within- vs cross-condition RMSE for PSID, DPAD, and VARMA (DBS-OFF / DBS-ON). "
    "Panel C: mean ROC curve per latent feature group (mean ± 1 std across sessions). "
    "Panel D: forecast RMSE vs horizon — PSID vs VARMA, DBS-OFF (solid) / DBS-ON (dashed)."
)

try:
    _all_triplets = thesis_specs._ALL_TRIPLETS
    _f1_spec = THESIS_CLASSIFICATION_F1[0] if THESIS_CLASSIFICATION_F1 else None
    _fc_spec = THESIS_FORECAST_FIGURES[0] if THESIS_FORECAST_FIGURES else None

    if not _all_triplets:
        st.warning("No aligned triplets found for grouped comparison.")
    elif _f1_spec is None:
        st.warning("No classification spec found for grouped comparison.")
    elif _fc_spec is None:
        st.warning("No forecast spec found for grouped comparison.")
    else:
        _wc_data = collect_within_cross_rmse(
            results_root, _all_triplets, channel_idx=0, split="test"
        )
        _roc_curves = collect_classification_roc_curves(
            results_root, _f1_spec.points,
            classification_parent=_f1_spec.classification_parent,
        )
        _fc_data = collect_forecast_horizon_rmse(
            results_root,
            _fc_spec.triplets,
            channel_idx=_fc_spec.channel_idx,
            split=_fc_spec.split,
            sampling_hz=_fc_spec.sampling_hz,
            sample_every=_fc_spec.sample_every,
            naive_rmse=_fc_spec.naive_rmse,
            forecast_target=_fc_spec.forecast_target,
        )
        st.caption(
            f"Within/cross triplets: **{_wc_data.n_triplets_used}** · "
            f"ROC curves loaded: **{len(_roc_curves)}** · "
            f"Forecast triplets: **{_fc_data.n_triplets_used}** "
            f"(OFF trials: {_fc_data.n_trials_off}, ON trials: {_fc_data.n_trials_on})"
        )
        _theme_grouped = (
            THESIS_CLASSIFICATION_F1[0].theme
            if THESIS_CLASSIFICATION_F1
            else thesis_specs.ThesisTheme.LIGHT
        )
        fig_grouped = build_session_grouped_figure(
            _wc_data, _roc_curves, _fc_data, theme=_theme_grouped,
        )
        st.plotly_chart(fig_grouped, use_container_width=True)
except Exception as e:
    st.error(f"Failed to build grouped session comparison figure: {e}")

st.markdown("---")
st.markdown("## Flipped classification — balanced accuracy (PSID)")
st.caption(
    "Each cell: LDA balanced accuracy for DBS-state classification using latent features "
    "from the cross-condition (flipped) PSID model. "
    "Rows: session. Columns: feature group. "
    "Axes: h = history window (s), m = forecast horizon (s). "
    "Green > 0.5 means the flipped model's latent state retains DBS information; "
    "near 0.5 means it does not."
)

try:
    _theme_flipped = (
        THESIS_CLASSIFICATION_F1[0].theme
        if THESIS_CLASSIFICATION_F1
        else thesis_specs.ThesisTheme.LIGHT
    )
    fig_flipped = build_flipped_heatmap_figure(results_root, theme=_theme_flipped)
    st.plotly_chart(fig_flipped, use_container_width=True)
except Exception as e:
    st.error(f"Failed to build flipped heatmap figure: {e}")
