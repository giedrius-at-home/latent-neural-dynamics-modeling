"""Thesis figure spec dataclasses and variant helpers — no dashboard dependencies.

Import these instead of thesis_lib.specs in any thesis notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional


class ThesisTheme(str, Enum):
    """Theme enum kept for sec2+ spec dataclass defaults.

    The matplotlib-based thesis_style is theme-agnostic (paper style only);
    this enum survives as a typed field value until sec2 specs drop it.
    """

    LIGHT = "light"
    DARK = "dark"


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
InputMode = Literal["neural", "behavioral"]
ExemplarLayout = Literal["stacked", "combined_gap", "side_by_side"]

THESIS_DECLARED_BEHAVIORAL_OUTPUTS: tuple[str, ...] = (
    "tracing_velocity_x",
    "tracing_acceleration_magnitude",
)


class ThesisDataError(FileNotFoundError):
    """Missing or incomplete thesis results."""


# ---------------------------------------------------------------------------
# Core triplet dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedTriplet:
    """One participant/session slice: matching PSID, DPAD, and VARMA result folders + timestamps."""

    psid_variant: str
    psid_run_ts: str
    dpad_variant: str
    dpad_run_ts: str
    varma_variant: str
    varma_run_ts: str
    label: str = ""
    psid_run_ts_off: str | None = None
    psid_run_ts_on: str | None = None
    dpad_run_ts_off: str | None = None
    dpad_run_ts_on: str | None = None
    varma_run_ts_off: str | None = None
    varma_run_ts_on: str | None = None
    varma_run_ts_eval_off: str | None = None
    varma_run_ts_eval_on: str | None = None


@dataclass(frozen=True)
class StripPanelEntry:
    """One subplot: display label + aligned triplet."""

    panel_label: str
    triplet: AlignedTriplet


# ---------------------------------------------------------------------------
# Figure spec dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThesisStripPanelsSpec:
    """Session-mean RMSE strip plots (one panel per participant slice)."""

    section_title: str
    channel_idx: int
    panels: list[StripPanelEntry]
    split: str = "test"
    ncols: int = 4
    theme: ThesisTheme = ThesisTheme.LIGHT
    jitter_seed: int = 42
    caption: str = ""


@dataclass(frozen=True)
class ThesisAggregateRmseSpec:
    """Pooled test-trial RMSE boxplot (3 models x 2 DBS), mean +/- SEM."""

    section_title: str
    channel_idx: int
    triplets: list[AlignedTriplet]
    split: str = "test"
    theme: ThesisTheme = ThesisTheme.LIGHT
    run_wilcoxon: bool = True
    jitter_seed: int = 42
    caption: str = ""
    show_brackets: bool = True


@dataclass(frozen=True)
class ThesisForecastRmseSpec:
    """Forecast horizon RMSE: PSID vs VARMA vs naive baseline, two DBS panels."""

    section_title: str
    channel_idx: int
    triplets: list[AlignedTriplet]
    split: str = "test"
    sampling_hz: float = 80.0
    sample_every: int = 5
    naive_rmse: float = 1.0
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption: str = ""
    forecast_target: Literal["Z", "Y"] = "Z"
    neural_y_feature_name: str = ""


@dataclass(frozen=True)
class ThesisNeuralBandHeatmapSpec:
    """Pooled neural self-prediction: mean Pearson r by spectral band x model x DBS."""

    section_title: str
    triplets: list[AlignedTriplet]
    split: str = "test"
    theme: ThesisTheme = ThesisTheme.LIGHT
    band_row_order: list[str] | None = None
    caption: str = ""


@dataclass(frozen=True)
class ThesisNeuralTimeseriesSpec:
    """Neural self-prediction exemplar: Y vs Yhat for one saved neural output channel."""

    section_title: str
    participant_label: str
    psid_variant: str
    dpad_variant: str
    varma_variant: str
    psid_run_ts: str
    dpad_run_ts: str
    varma_run_ts: str
    split: str
    trial_idx_off: int
    trial_idx_on: int
    neural_y_channel_idx: int
    neural_y_feature_name: str = ""
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption_extra: str = ""
    varma_run_ts_off: str | None = None
    varma_run_ts_on: str | None = None
    use_adjacent_off_on_trials: bool = False
    exemplar_layout: ExemplarLayout = "stacked"
    exemplar_mid_segment_s: float = 1.0
    exemplar_side_extend_s: float = 0.0
    exemplar_abs_gap_s: float = 0.1


@dataclass(frozen=True)
class ThesisC2ForecastSpec:
    """Figure C2: forecast time-series (DBS-OFF / DBS-ON panels)."""

    section_title: str
    participant_label: str
    psid_variant: str
    dpad_variant: str
    varma_variant: str
    psid_run_ts: str
    dpad_run_ts: str
    varma_run_ts: str
    split: str
    trial_idx_off: int
    trial_idx_on: int
    channel_idx: int
    history_ms: float = 1000.0
    forecast_ms: float = 1000.0
    sampling_hz: float = 80.0
    theme: ThesisTheme = ThesisTheme.LIGHT
    y_axis_label: str = "z-scored tracing speed"
    caption: str = ""
    forecast_target: Literal["Z", "Y"] = "Z"
    neural_y_feature_name: str = ""
    use_adjacent_off_on_trials: bool = False
    varma_run_ts_off: str | None = None
    varma_run_ts_on: str | None = None
    show_session_rmse_bands: bool = True
    show_psid_sigma_band: bool = False
    include_next_trial_prediction_rows: bool = False
    prediction_row_center_segment_s: float = 0.0
    prediction_row_side_extend_s: float = 1.0


# ---------------------------------------------------------------------------
# Variant string helpers
# ---------------------------------------------------------------------------


def _variant_off(variant: str) -> str:
    return variant.replace("dbs_on", "dbs_off").replace("dbs_both", "dbs_off")


def _variant_on(variant: str) -> str:
    return variant.replace("dbs_off", "dbs_on").replace("dbs_both", "dbs_on")


def _variant_cross_eval(source_variant: str, target: str) -> str:
    return f"{source_variant}_eval_{target}"


TripletBranch = Literal["off", "on", "eval_off", "eval_on"]


def _need_ts(value: Optional[str], field: str, tri: AlignedTriplet) -> str:
    if value is None or not str(value).strip():
        raise ThesisDataError(
            f"AlignedTriplet {tri.label!r}: {field} must be set for strict loading."
        )
    return str(value).strip()


def triplet_branch_timestamp(
    tri: AlignedTriplet,
    framework: Literal["psid", "dpad", "varma"],
    branch: TripletBranch,
) -> str:
    """Run timestamp for this framework/branch combination."""
    if framework == "psid":
        if branch == "off":
            return _need_ts(tri.psid_run_ts_off, "psid_run_ts_off", tri)
        if branch == "on":
            return _need_ts(tri.psid_run_ts_on, "psid_run_ts_on", tri)
        if branch == "eval_off":
            return _need_ts(tri.psid_run_ts_on, "psid_run_ts_on", tri)
        if branch == "eval_on":
            return _need_ts(tri.psid_run_ts_off, "psid_run_ts_off", tri)
    elif framework == "dpad":
        if branch == "off":
            return _need_ts(tri.dpad_run_ts_off, "dpad_run_ts_off", tri)
        if branch == "on":
            return _need_ts(tri.dpad_run_ts_on, "dpad_run_ts_on", tri)
        if branch == "eval_off":
            return _need_ts(tri.dpad_run_ts_on, "dpad_run_ts_on", tri)
        if branch == "eval_on":
            return _need_ts(tri.dpad_run_ts_off, "dpad_run_ts_off", tri)
    elif framework == "varma":
        if branch == "off":
            return _need_ts(tri.varma_run_ts_off, "varma_run_ts_off", tri)
        if branch == "on":
            return _need_ts(tri.varma_run_ts_on, "varma_run_ts_on", tri)
        if branch == "eval_off":
            ev = getattr(tri, "varma_run_ts_eval_off", None)
            if ev and str(ev).strip():
                return str(ev).strip()
            return _need_ts(tri.varma_run_ts_on, "varma_run_ts_on", tri)
        if branch == "eval_on":
            ev = getattr(tri, "varma_run_ts_eval_on", None)
            if ev and str(ev).strip():
                return str(ev).strip()
            return _need_ts(tri.varma_run_ts_off, "varma_run_ts_off", tri)
    raise ValueError(f"Unknown framework {framework!r}")
