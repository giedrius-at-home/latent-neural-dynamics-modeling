"""
Thesis final-results dashboard: hardcoded figure specs in `dashboard/thesis/specs.py`.

View-transition CSS (`::view-transition-*`) applies to full-document navigation in browsers;
Streamlit does not use that API, so those rules have no effect here. Use a static HTML export
if you need that animation elsewhere.
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
from dashboard.thesis.compose import compose_thesis_figure
from dashboard.thesis.forecast_horizon_rmse import collect_forecast_horizon_rmse
from dashboard.thesis.forecast_rmse_figure import build_forecast_rmse_figure_or_empty
from dashboard.thesis.neural_band_heatmap_figure import build_neural_band_heatmap_figure
from dashboard.thesis.neural_band_pearson import collect_neural_band_pearson
from dashboard.thesis.classification_f1_figure import build_classification_f1_figure
from dashboard.thesis.latent_phase_space_figure import build_latent_phase_space_figure
from dashboard.thesis.psid_cy_importance_figure import build_psid_cy_importance_figure
from dashboard.thesis.rmse_distribution_figure import build_rmse_distribution_figure
from dashboard.thesis.session_strip_figure import build_session_strip_figure
from dashboard.thesis.session_strip_rmse import collect_strip_figure_data
from dashboard.thesis.specs import (
    DEFAULT_AGGREGATE_CAPTION,
    DEFAULT_C2_FORECAST_CAPTION,
    DEFAULT_FORECAST_CAPTION,
    DEFAULT_NEURAL_BAND_CAPTION,
    DEFAULT_CLASSIFICATION_F1_CAPTION,
    DEFAULT_LATENT_PHASE_CAPTION,
    DEFAULT_PSID_CY_IMPORTANCE_CAPTION,
    DEFAULT_STRIP_CAPTION,
    THESIS_AGGREGATE_FIGURES,
    THESIS_C2_FORECASTS,
    THESIS_CLASSIFICATION_F1,
    THESIS_FIGURES,
    THESIS_FORECAST_FIGURES,
    THESIS_NEURAL_BAND_HEATMAPS,
    THESIS_LATENT_PHASE,
    THESIS_PSID_CY_IMPORTANCE,
    THESIS_STRIP_PANELS,
)

st.set_page_config(layout="wide", page_title="Thesis results")
st.title("Thesis final results")

st.markdown("## Time-series exemplars")
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
st.markdown("## Figure C2 — Forecast time-series (history + forecast)")

for c2_spec in THESIS_C2_FORECASTS:
    st.markdown(f"### {c2_spec.section_title}")
    st.caption(
        f"Same trial indices as time-series exemplars (A1): OFF row = trial **{c2_spec.trial_idx_off}**, "
        f"ON row = **{c2_spec.trial_idx_on}**."
    )
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
st.markdown("## Latent phase space (PSID vs DPAD-RNN)")

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
            st.warning("No strip-plot data (missing results or no overlapping trial keys).")
            continue
        rng = np.random.default_rng(strip_spec.jitter_seed)
        fig_s = build_session_strip_figure(
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
        )
        if fc_data is not None:
            st.caption(
                f"Triplets loaded: **{fc_data.n_triplets_used}** · "
                f"trials OFF / ON: **{fc_data.n_trials_off}** / **{fc_data.n_trials_on}**."
            )
        fig_fc = build_forecast_rmse_figure_or_empty(fc_data, fc_spec.theme)
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption(fc_spec.caption or DEFAULT_FORECAST_CAPTION)
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
        fig_nb = build_neural_band_heatmap_figure(nb_data, nb_spec.theme)
        st.plotly_chart(fig_nb, use_container_width=True)
        st.caption(nb_spec.caption or DEFAULT_NEURAL_BAND_CAPTION)
    except Exception as e:
        st.error(f"Failed to build neural band heatmap: {e}")
