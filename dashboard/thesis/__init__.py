"""Thesis figure helpers (stacked DBS OFF/ON, z-scored traces, RMSE callout)."""

from dashboard.thesis.aggregate_rmse import collect_pooled_rmse
from dashboard.thesis.c2_forecast_timeseries import build_c2_forecast_figure
from dashboard.thesis.classification_f1_figure import build_classification_f1_figure
from dashboard.thesis.latent_phase_space_figure import build_latent_phase_space_figure
from dashboard.thesis.psid_cy_importance_figure import build_psid_cy_importance_figure
from dashboard.thesis.compose import compose_thesis_figure
from dashboard.thesis.forecast_horizon_rmse import collect_forecast_horizon_rmse
from dashboard.thesis.forecast_rmse_figure import build_forecast_rmse_figure_or_empty
from dashboard.thesis.neural_band_heatmap_figure import build_neural_band_heatmap_figure
from dashboard.thesis.neural_band_pearson import collect_neural_band_pearson
from dashboard.thesis.rmse_distribution_figure import build_rmse_distribution_figure
from dashboard.thesis.session_strip_figure import build_session_strip_figure
from dashboard.thesis.session_strip_rmse import collect_strip_figure_data
from dashboard.thesis.specs import (
    AlignedTriplet,
    StripPanelEntry,
    THESIS_AGGREGATE_FIGURES,
    THESIS_C2_FORECASTS,
    THESIS_FIGURES,
    THESIS_FORECAST_FIGURES,
    THESIS_LATENT_PHASE,
    THESIS_NEURAL_BAND_HEATMAPS,
    THESIS_PSID_CY_IMPORTANCE,
    THESIS_STRIP_PANELS,
    ThesisAggregateRmseSpec,
    ThesisC2ForecastSpec,
    ThesisClassificationF1Spec,
    ThesisFigureSpec,
    ThesisForecastRmseSpec,
    ThesisNeuralBandHeatmapSpec,
    ThesisLatentPhaseSpec,
    ThesisPsidCyImportanceSpec,
    ThesisStripPanelsSpec,
    LatentPhasePanel,
    LatentPhaseRow,
    PsidCyPanel,
    PsidCyRow,
    ClassificationF1PickleRef,
)

__all__ = [
    "AlignedTriplet",
    "build_c2_forecast_figure",
    "build_classification_f1_figure",
    "build_latent_phase_space_figure",
    "build_psid_cy_importance_figure",
    "build_forecast_rmse_figure_or_empty",
    "build_neural_band_heatmap_figure",
    "build_rmse_distribution_figure",
    "build_session_strip_figure",
    "collect_forecast_horizon_rmse",
    "collect_neural_band_pearson",
    "collect_pooled_rmse",
    "collect_strip_figure_data",
    "compose_thesis_figure",
    "StripPanelEntry",
    "ThesisAggregateRmseSpec",
    "ThesisC2ForecastSpec",
    "ThesisClassificationF1Spec",
    "ThesisFigureSpec",
    "ThesisForecastRmseSpec",
    "ThesisNeuralBandHeatmapSpec",
    "ThesisLatentPhaseSpec",
    "ThesisPsidCyImportanceSpec",
    "ThesisStripPanelsSpec",
    "LatentPhasePanel",
    "LatentPhaseRow",
    "PsidCyPanel",
    "PsidCyRow",
    "ClassificationF1PickleRef",
    "THESIS_AGGREGATE_FIGURES",
    "THESIS_C2_FORECASTS",
    "THESIS_CLASSIFICATION_F1",
    "THESIS_FIGURES",
    "THESIS_FORECAST_FIGURES",
    "THESIS_LATENT_PHASE",
    "THESIS_NEURAL_BAND_HEATMAPS",
    "THESIS_PSID_CY_IMPORTANCE",
    "THESIS_STRIP_PANELS",
]
