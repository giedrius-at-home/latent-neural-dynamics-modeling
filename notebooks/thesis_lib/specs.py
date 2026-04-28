"""Hardcoded thesis figure specs — edit lists here (no UI selection)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from thesis_lib.constants import ThesisTheme
from thesis_lib.errors import ThesisDataError

InputMode = Literal["neural", "behavioral"]
ExemplarLayout = Literal["stacked", "combined_gap", "side_by_side"]

# Authoritative behavioral output names (YAML `output:`); used in captions when parquet lacks ``output_channels``.
THESIS_DECLARED_BEHAVIORAL_OUTPUTS: tuple[str, ...] = (
    "tracing_velocity_x",
    "tracing_acceleration_magnitude",
)

FeatureGroupF1 = Literal["xp", "xp_1", "xp_2", "xp_with_dbs"]


@dataclass(frozen=True)
class AlignedTriplet:
    """One participant/session slice: matching PSID, DPAD, and VARMA result folders + run timestamps."""

    psid_variant: str
    psid_run_ts: str
    dpad_variant: str
    dpad_run_ts: str
    varma_variant: str
    varma_run_ts: str
    label: str = ""
    # Required for strict within/cross and condition-specific loads (no silent timestamp fallback).
    psid_run_ts_off: str | None = None
    psid_run_ts_on: str | None = None
    dpad_run_ts_off: str | None = None
    dpad_run_ts_on: str | None = None
    varma_run_ts_off: str | None = None
    varma_run_ts_on: str | None = None
    # Cross-condition VARMA test parquets often use a different timestamp than the ON/OFF training runs.
    varma_run_ts_eval_off: str | None = None
    varma_run_ts_eval_on: str | None = None


@dataclass(frozen=True)
class StripPanelEntry:
    """One subplot: display label + aligned triplet for that participant/session slice."""

    panel_label: str
    triplet: AlignedTriplet


@dataclass(frozen=True)
class ThesisStripPanelsSpec:
    """B2: Session-mean RMSE strip plots (one panel per participant slice)."""

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
    """Pooled test-trial RMSE bar chart (3 models × 2 DBS), mean ± SEM, jittered dots."""

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
    """
    Forecast horizon RMSE: PSID vs VARMA vs naïve baseline, two DBS panels.

    ``forecast_target=\"Z\"`` uses ``Z_future_*`` (behavioral outputs). ``forecast_target=\"Y\"`` uses
    ``Y_future_*`` (neural); set ``neural_y_feature_name`` to the training YAML neural input string when
    resolving the column index.

    DPAD is omitted when forecast outputs are not comparable.
    """

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
    """
    Pooled neural self-prediction: mean Pearson r between Y and Ŷ by spectral band (rows),
    model (PSID / DPAD / VARMA), and DBS (OFF vs ON panels).

    Uses `AlignedTriplet` (same as RMSE pooling): all three runs must expose aligned `Y` / `Yp`
    and matching `input_channels` for narrow-band neural columns.
    """

    section_title: str
    triplets: list[AlignedTriplet]
    split: str = "test"
    band_row_order: tuple[str, ...] = ("Delta", "Theta", "Alpha", "Beta")
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption: str = ""


@dataclass(frozen=True)
class PsidCyPanel:
    """One session column: PSID result folder + run timestamp for `model_<run_ts>.pkl`."""

    session_label: str
    psid_variant: str
    psid_run_ts: str


@dataclass(frozen=True)
class PsidCyRow:
    """One participant row: ordered session columns (left to right)."""

    participant_label: str
    panels: tuple[PsidCyPanel, ...]


@dataclass(frozen=True)
class ThesisPsidCyImportanceSpec:
    """
    PSID behaviourally relevant subspace: per-panel–normalized ‖Cy[:, :n₁]‖₂ row norms
    reshaped to 4×29 (ECoG × narrowband). One row per participant; one column per session.
    """

    section_title: str
    rows: tuple[PsidCyRow, ...]
    split: str = "test"
    theme: ThesisTheme = ThesisTheme.LIGHT
    show_beta_box: bool = True
    importance_gamma: float = 0.5  # power transform applied to [0,1] importance; <1 expands dim values (0.5 = sqrt)
    caption: str = ""


@dataclass(frozen=True)
class ThesisPsidCzSpec:
    """
    PSID behavioural readout matrix Cz (nz × n1), abs-max normalized to [-1, 1].
    One row per participant; one column per session.
    """

    section_title: str
    rows: tuple[PsidCyRow, ...]  # reuses PsidCyPanel / PsidCyRow (same sessions, same model pkl)
    split: str = "test"
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption: str = ""


@dataclass(frozen=True)
class LatentPhasePanel:
    """One session: aligned PSID and DPAD test runs (same participant/session slice)."""

    session_label: str
    psid_variant: str
    psid_run_ts: str
    dpad_variant: str
    dpad_run_ts: str


@dataclass(frozen=True)
class LatentPhaseRow:
    """One participant row: ordered session columns (left to right)."""

    participant_label: str
    panels: tuple[LatentPhasePanel, ...]


@dataclass(frozen=True)
class ThesisLatentPhaseSpec:
    """
    Latent phase space: PSID plots x₁ vs x₂ directly (behaviourally relevant subspace);
    DPAD plots PC1 vs PC2 of x⁽¹⁾ (PCA fit on all test trials, both DBS conditions).
    KDE contours + representative trial trajectories per condition; axes independent per panel.
    """

    section_title: str
    rows: tuple[LatentPhaseRow, ...]
    split: str = "test"
    n_psid_latent: int = 2
    n_dpad_latent: int = 4
    n_trajectory_trials: int = 3
    trajectory_seed: int = 42
    kde_grid: int = 80
    density_percentiles: tuple[float, float] = (30.0, 58.0)
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption: str = ""


@dataclass(frozen=True)
class ThesisC2ForecastSpec:
    """
    Figure C2: forecast time-series (2×1, DBS-OFF / DBS-ON). Same trial indices as A1 (`ThesisFigureSpec`).

    Each panel: history (true only) + forecast (true + PSID / DPAD / VARMA). Time in ms relative to t₀=0;
    history and forecast spans default to 1000 ms each (configurable).

    ``forecast_target=\"Z\"`` uses ``Z_future_*`` (behavioral). ``forecast_target=\"Y\"`` uses ``Y_future_*``
    (neural). For Y, set ``neural_y_feature_name`` to the training YAML neural input string; the column index is
    resolved against saved ``input_channels``. If that name is missing from the parquet, ``channel_idx`` is used.
    """

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
    # 50/50 display window: both must match for equal history vs forecast span on the axis.
    history_ms: float = 1000.0
    forecast_ms: float = 1000.0
    sampling_hz: float = 80.0
    theme: ThesisTheme = ThesisTheme.LIGHT
    y_axis_label: str = "z-scored tracing speed"
    caption: str = ""
    forecast_target: Literal["Z", "Y"] = "Z"
    neural_y_feature_name: str = ""
    use_adjacent_off_on_trials: bool = False
    # When ``dbs_both`` VARMA: OFF/ON parquets; timestamps inferred via ``infer_varma_off_on_run_ts`` if unset.
    varma_run_ts_off: str | None = None
    varma_run_ts_on: str | None = None
    show_session_rmse_bands: bool = True
    # PSID ±1σ (∝√m) fill in forecast panels (validation residual σ). Off by default so forecast matches one-step rows.
    show_psid_sigma_band: bool = False
    # One-step rows below forecast panels (disabled by default; use cross-block figure for stitched trials).
    include_next_trial_prediction_rows: bool = False
    # If enabled: window ≈ center_segment + 2×side_extend around trial mid (see ``_slice_trial_center``).
    prediction_row_center_segment_s: float = 0.0
    prediction_row_side_extend_s: float = 1.0


@dataclass(frozen=True)
class ThesisWithinCrossSpec:
    """
    Part A (4-panel trial-level time series) + Part B (2-panel RMSE box within vs cross).

    Variants derived from joint_triplet: off/on replace dbs_both; cross eval = {on}_eval_off, {off}_eval_on.
    """

    section_title: str
    participant_label: str
    joint_triplet: AlignedTriplet
    channel_idx: int
    split: str = "test"
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption: str = ""
    # Part A: show only the last ``trial_tail_window_s`` seconds per trial; cross-condition trace only in the last fraction.
    trial_tail_window_s: float = 1.0
    cross_visible_tail_fraction: float = 0.5


@dataclass(frozen=True)
class ThesisCrossBlockSpec:
    """
    Three rows (PSID / DPAD / VARMA) × two columns (OFF→ON stitch, ON→OFF stitch).
    Each cell: last ``trial_segment_s`` of the first trial + gap + first ``trial_segment_s`` of the second,
    with Ŷ from OFF-trained, BOTH-trained, and ON-trained checkpoints (cross-eval where needed).
    """

    section_title: str
    participant_label: str
    joint_triplet: AlignedTriplet
    channel_idx: int
    split: str = "test"
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption: str = ""
    forecast_target: Literal["Z", "Y"] = "Z"
    neural_y_feature_name: str = ""
    trial_segment_s: float = 1.0
    concat_gap_s: float = 0.12
    y_axis_label: str = ""


@dataclass(frozen=True)
class ThesisForecastCheckpointSpec:
    """
    3×2 grid (PSID / DPAD / VARMA × DBS-OFF trial / DBS-ON trial): multi-step Ŷ from OFF-, BOTH-, and
    ON-trained checkpoints (cross-eval parquets for mismatched DBS). Same OFF/ON boundary trials as cross-block.
    """

    section_title: str
    participant_label: str
    joint_triplet: AlignedTriplet
    channel_idx: int
    split: str = "test"
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption: str = ""
    forecast_target: Literal["Z", "Y"] = "Z"
    neural_y_feature_name: str = ""
    history_ms: float = 1000.0
    forecast_ms: float = 1000.0
    sampling_hz: float = 80.0
    y_axis_label: str = ""


@dataclass(frozen=True)
class ThesisFigureSpec:
    """
    One publication figure: PSID / DPAD / VARMA must share aligned trial order for the chosen split.

    Set `trial_idx_off` and `trial_idx_on` to the row indices of DBS-OFF and DBS-ON trials
    (e.g. from a `dbs_both` training run).
    """

    section_title: str
    input_mode: InputMode
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
    y_axis_label: str = "speed (z)"
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption_extra: str = ""
    varma_run_ts_off: str | None = None
    varma_run_ts_on: str | None = None
    use_adjacent_off_on_trials: bool = False
    exemplar_layout: ExemplarLayout = "stacked"
    # ``combined_gap``: total time shown per trial ≈ ``exemplar_mid_segment_s + 2*exemplar_side_extend_s``,
    # centered on trial mid (capped by trial span). Default 1 s per OFF / ON segment before the concat gap.
    exemplar_mid_segment_s: float = 1.0
    exemplar_side_extend_s: float = 0.0
    exemplar_abs_gap_s: float = 0.1


@dataclass(frozen=True)
class ThesisNeuralTimeseriesSpec:
    """
    Neural self-prediction exemplar: Y vs Ŷ for one saved neural output channel.

    ``neural_y_channel_idx`` indexes the column in per-trial Y (same trial rows as behavioral A1).
    Captions use saved ``input_channels`` (and optional ``neural_y_feature_name`` = YAML ``neural_input``).
    """

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
    # Same string as YAML ``neural_input`` when set; improves captions if ``input_channels`` is missing.
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
class ClassificationF1PickleRef:
    """
    One participant × session × feature source: path to `LDA_*_prediction.pkl` under
    `RESULTS_PATH/classification/` (see `classification/compute.py`).
    """

    participant_label: str
    session_label: str
    group: FeatureGroupF1
    pickle_relative_path: str
    model_label: str = "PSID"


@dataclass(frozen=True)
class ThesisClassificationF1Spec:
    """
    Figure F1: test-set balanced accuracy (CSP + LDA), dot plot per session, three feature groups.
    Add a second list entry with a different `section_title` for flipped / DBS-channel / Xp_n variants.
    """

    section_title: str
    points: tuple[ClassificationF1PickleRef, ...]
    classification_parent: str = "classification"
    jitter_seed: int = 42
    jitter_half_width: float = 0.2
    y_min: float = 0.4
    y_max: float = 1.0
    permutation_alpha: float = 0.05
    theme: ThesisTheme = ThesisTheme.LIGHT
    caption: str = ""


# --- Edit this list for your paper figures ---
DEFAULT_C2_FORECAST_CAPTION = ""

DEFAULT_PSID_CY_IMPORTANCE_CAPTION = (
    "Each cell: importance = ‖Cy[:, :n₁]‖₂ after reshaping rows of Cy to 4 contacts × 29 narrow-band "
    "features (row order matches `input_channels` in the saved results). "
    "Values are divided by the maximum in that panel (per-panel normalization); absolute magnitudes "
    "are not comparable across sessions (latent rotational ambiguity and session-wise signal scaling). "
    "Horizontal axis groups use plain band names (Delta, Theta, Alpha, Beta); the dashed outline marks the beta band columns from channel names. "
    "In PSID notation, x_p1 is the behaviourally relevant latent block (dimension n₁); x_p2 denotes the complementary / residual block "
    "(Cy[:, n₁:] is omitted here). A square-root transform is applied to the normalized importance before display to improve contrast."
)

DEFAULT_PSID_CZ_CAPTION = (
    "Cz is the behavioural readout matrix (nz × n₁): each row corresponds to one behavioral output, "
    "each column to a behaviourally relevant latent dimension. "
    "Values are normalized by the element-wise absolute maximum within each panel. "
    "Blue = negative weight, red = positive weight, white = near zero. "
    "Absolute magnitudes are not comparable across sessions."
)

DEFAULT_NEURAL_TS_CAPTION = ""

DEFAULT_LATENT_PHASE_CAPTION = (
    "PSID: x₁ vs x₂ in the behaviourally relevant subspace (no PCA). "
    "DPAD: PC1 vs PC2 of x⁽¹⁾; PCA was fit on all test-set timepoints (DBS-OFF and DBS-ON combined) "
    "within each panel so the projection is not biased toward separating conditions. "
    "Explained variance fractions are panel-specific (caption annotations). "
    "Filled contours: 2D Gaussian KDE (slightly sharpened bandwidth for clearer within-model DBS modes) "
    "at two density levels (percentiles of the KDE grid); "
    "thin lines: three randomly chosen trials per condition. "
    "Axes are not shared across panels (each model instance has its own latent geometry). "
    "DBS-OFF: purple; DBS-ON: teal (condition colours, distinct from model identity colours)."
)

# ---------------------------------------------------------------------------
# Aligned triplets for all available participant × session combos
# ---------------------------------------------------------------------------

_TRIPLET_PDI1_S2 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI1_2_nx_4_n2_i35_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260411_000914",
    dpad_variant="dpad_PDI1_S2_dbs_both_narrow_band",
    dpad_run_ts="20260317_190752",
    varma_variant="varma_behavioral_PDI1_2_p30_q1_top5_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260411_112419",
    label="PDI1_S2",
    psid_run_ts_off="20260410_234659",
    psid_run_ts_on="20260410_233414",
    dpad_run_ts_off="20260317_221506",
    dpad_run_ts_on="20260317_210429",
    varma_run_ts_off="20260323_222014",
    varma_run_ts_on="20260323_220659",
    varma_run_ts_eval_off="20260324_123937",
    varma_run_ts_eval_on="20260324_120832",
)
_TRIPLET_PDI1_S4 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI1_4_nx_25_n2_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260411_034923",
    dpad_variant="dpad_PDI1_S4_dbs_both_narrow_band",
    dpad_run_ts="20260317_231003",
    varma_variant="varma_PDI1_S4_dbs_both_narrow_band",
    varma_run_ts="20260323_210143",
    label="PDI1_S4",
    psid_run_ts_off="20260411_035933",
    psid_run_ts_on="20260411_034923",
    dpad_run_ts_off="20260318_002329",
    dpad_run_ts_on="20260317_235513",
    varma_run_ts_off="20260323_225810",
    varma_run_ts_on="20260323_224723",
    varma_run_ts_eval_off="20260324_124824",
    varma_run_ts_eval_on="20260324_121627",
)
_TRIPLET_PDI4_S2 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_162132",
    dpad_variant="dpad_PDI4_S2_dbs_both_narrow_band",
    dpad_run_ts="20260318_003944",
    varma_variant="varma_behavioral_PDI4_2_p30_q1_top5_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260411_112104",
    label="PDI4_S2",
    psid_run_ts_off="20260408_164031",
    psid_run_ts_on="20260408_163407",
    dpad_run_ts_off="20260318_033013",
    dpad_run_ts_on="20260318_023809",
    varma_run_ts_off="20260323_233524",
    varma_run_ts_on="20260323_232355",
    varma_run_ts_eval_off="20260324_125505",
    varma_run_ts_eval_on="20260324_122240",
)
_TRIPLET_PDI4_S3 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_185522",
    dpad_variant="dpad_PDI4_S3_dbs_both_narrow_band",
    dpad_run_ts="20260318_045702",
    varma_variant="varma_behavioral_PDI4_3_p30_q1_top5_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260411_112239",
    label="PDI4_S3",
    psid_run_ts_off="20260408_191423",
    psid_run_ts_on="20260408_190749",
    dpad_run_ts_off="20260318_072456",
    dpad_run_ts_on="20260318_061958",
    varma_run_ts_off="20260324_001741",
    varma_run_ts_on="20260324_000445",
    varma_run_ts_eval_off="20260324_130210",
    varma_run_ts_eval_on="20260324_122958",
)

_ALL_TRIPLETS = [_TRIPLET_PDI1_S2, _TRIPLET_PDI1_S4, _TRIPLET_PDI4_S2, _TRIPLET_PDI4_S3]


def infer_varma_off_on_run_ts(varma_variant: str, varma_run_ts: str) -> tuple[str, str] | None:
    """
    For a ``dbs_both`` VARMA variant + joint run timestamp, return matching OFF/ON VARMA run timestamps
    when they appear in ``_ALL_TRIPLETS``; otherwise None.
    """
    for tri in _ALL_TRIPLETS:
        if tri.varma_variant == varma_variant and tri.varma_run_ts == varma_run_ts:
            return tri.varma_run_ts_off, tri.varma_run_ts_on
    return None


# Neural band heatmaps (Pearson r on Y vs Ŷ): same sessions as RMSE triplets by default.
# Edit this list independently if your PSID/DPAD/VARMA checkpoints for band pooling differ
# from behavioral-output PSID folders (still need matching input_channels with ECOG_/LFP_ bands).
_NEURAL_BAND_TRIPLET_PDI1_S2 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI1_2_nx_4_n2_i35_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260411_000914",
    dpad_variant="dpad_PDI1_S2_dbs_both_narrow_band",
    dpad_run_ts="20260317_190752",
    varma_variant="varma_behavioral_PDI1_2_p30_q1_top5_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260411_112419",
    label="PDI1_S2",
    psid_run_ts_off="20260410_234659",
    psid_run_ts_on="20260410_233414",
    dpad_run_ts_off="20260317_221506",
    dpad_run_ts_on="20260317_210429",
    varma_run_ts_off="20260323_222014",
    varma_run_ts_on="20260323_220659",
    varma_run_ts_eval_off="20260324_123937",
    varma_run_ts_eval_on="20260324_120832",
)
_NEURAL_BAND_TRIPLET_PDI1_S4 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI1_4_nx_25_n2_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260411_034923",
    dpad_variant="dpad_PDI1_S4_dbs_both_narrow_band",
    dpad_run_ts="20260317_231003",
    varma_variant="varma_PDI1_S4_dbs_both_narrow_band",
    varma_run_ts="20260323_210143",
    label="PDI1_S4",
    psid_run_ts_off="20260411_035933",
    psid_run_ts_on="20260411_034923",
    dpad_run_ts_off="20260318_002329",
    dpad_run_ts_on="20260317_235513",
    varma_run_ts_off="20260323_225810",
    varma_run_ts_on="20260323_224723",
    varma_run_ts_eval_off="20260324_124824",
    varma_run_ts_eval_on="20260324_121627",
)
_NEURAL_BAND_TRIPLET_PDI4_S2 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_162132",
    dpad_variant="dpad_PDI4_S2_dbs_both_narrow_band",
    dpad_run_ts="20260318_003944",
    varma_variant="varma_behavioral_PDI4_2_p30_q1_top5_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260411_112104",
    label="PDI4_S2",
    psid_run_ts_off="20260408_164031",
    psid_run_ts_on="20260408_163407",
    dpad_run_ts_off="20260318_033013",
    dpad_run_ts_on="20260318_023809",
    varma_run_ts_off="20260323_233524",
    varma_run_ts_on="20260323_232355",
    varma_run_ts_eval_off="20260324_125505",
    varma_run_ts_eval_on="20260324_122240",
)
_NEURAL_BAND_TRIPLET_PDI4_S3 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_185522",
    dpad_variant="dpad_PDI4_S3_dbs_both_narrow_band",
    dpad_run_ts="20260318_045702",
    varma_variant="varma_behavioral_PDI4_3_p30_q1_top5_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260411_112239",
    label="PDI4_S3",
    psid_run_ts_off="20260408_191423",
    psid_run_ts_on="20260408_190749",
    dpad_run_ts_off="20260318_072456",
    dpad_run_ts_on="20260318_061958",
    varma_run_ts_off="20260324_001741",
    varma_run_ts_on="20260324_000445",
    varma_run_ts_eval_off="20260324_130210",
    varma_run_ts_eval_on="20260324_122958",
)
_NEURAL_BAND_TRIPLETS = [
    _NEURAL_BAND_TRIPLET_PDI1_S2,
    _NEURAL_BAND_TRIPLET_PDI1_S4,
    _NEURAL_BAND_TRIPLET_PDI4_S2,
    _NEURAL_BAND_TRIPLET_PDI4_S3,
]


def _variant_off(variant: str) -> str:
    """Replace dbs_both/dbs_on with dbs_off."""
    return variant.replace("dbs_on", "dbs_off").replace("dbs_both", "dbs_off")


def _variant_on(variant: str) -> str:
    """Replace dbs_off/dbs_both with dbs_on."""
    return variant.replace("dbs_off", "dbs_on").replace("dbs_both", "dbs_on")


def _variant_cross_eval(source_variant: str, target: str) -> str:
    """Cross-condition eval: source_variant trained on A, evaluated on B. target='off'|'on'."""
    return f"{source_variant}_eval_{target}"


def _need_ts(value: str | None, field: str, tri: AlignedTriplet) -> str:
    if value is None or not str(value).strip():
        raise ThesisDataError(
            f"AlignedTriplet {tri.label!r}: {field} must be set for strict loading (condition-specific / cross-eval)."
        )
    return str(value).strip()


TripletBranch = Literal["off", "on", "eval_off", "eval_on"]


def triplet_branch_timestamp(
    tri: AlignedTriplet,
    framework: Literal["psid", "dpad", "varma"],
    branch: TripletBranch,
) -> str:
    """
    Run timestamp for ``test``/``val`` results for this branch.

    ``eval_off`` uses the ON-trained model evaluated on OFF trials → timestamp of the ON run.
    ``eval_on`` uses the OFF-trained model on ON trials → timestamp of the OFF run.
    """
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
            if ev is not None and str(ev).strip():
                return str(ev).strip()
            return _need_ts(tri.varma_run_ts_on, "varma_run_ts_on", tri)
        if branch == "eval_on":
            ev = getattr(tri, "varma_run_ts_eval_on", None)
            if ev is not None and str(ev).strip():
                return str(ev).strip()
            return _need_ts(tri.varma_run_ts_off, "varma_run_ts_off", tri)
    raise ValueError(f"Unknown framework {framework!r}")


# ---------------------------------------------------------------------------
# Within-Cross: trial-level time series (Part A) + RMSE box (Part B)
# ---------------------------------------------------------------------------

THESIS_WITHIN_CROSS = ThesisWithinCrossSpec(
    section_title="Within vs cross-condition decoding",
    participant_label="PDI1",
    joint_triplet=_TRIPLET_PDI1_S2,
    channel_idx=0,
)


# ---------------------------------------------------------------------------
# A1 — Time-series exemplars (one per participant/session, median-RMSE trial)
# ---------------------------------------------------------------------------

THESIS_FIGURES: list[ThesisFigureSpec] = [
    ThesisFigureSpec(
        section_title="PDI1 Session 2",
        input_mode="behavioral",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=11,
        trial_idx_on=27,
        channel_idx=0,
        varma_run_ts_off=_TRIPLET_PDI1_S2.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI1_S2.varma_run_ts_on,
    ),
    ThesisFigureSpec(
        section_title="PDI1 Session 4",
        input_mode="behavioral",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S4.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S4.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S4.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S4.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S4.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S4.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=17,
        trial_idx_on=5,
        channel_idx=0,
        varma_run_ts_off=_TRIPLET_PDI1_S4.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI1_S4.varma_run_ts_on,
    ),
    ThesisFigureSpec(
        section_title="PDI4 Session 2",
        input_mode="behavioral",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=5,
        trial_idx_on=18,
        channel_idx=0,
        varma_run_ts_off=_TRIPLET_PDI4_S2.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI4_S2.varma_run_ts_on,
    ),
    ThesisFigureSpec(
        section_title="PDI4 Session 3",
        input_mode="behavioral",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S3.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S3.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S3.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S3.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S3.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S3.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=9,
        trial_idx_on=25,
        channel_idx=0,
        varma_run_ts_off=_TRIPLET_PDI4_S3.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI4_S3.varma_run_ts_on,
    ),
    ThesisFigureSpec(
        section_title="PDI1 Session 2 — tracing acceleration magnitude",
        input_mode="behavioral",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=11,
        trial_idx_on=27,
        channel_idx=1,
        varma_run_ts_off=_TRIPLET_PDI1_S2.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI1_S2.varma_run_ts_on,
    ),
    ThesisFigureSpec(
        section_title="PDI1 Session 4 — tracing acceleration magnitude",
        input_mode="behavioral",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S4.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S4.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S4.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S4.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S4.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S4.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=17,
        trial_idx_on=5,
        channel_idx=1,
        varma_run_ts_off=_TRIPLET_PDI1_S4.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI1_S4.varma_run_ts_on,
    ),
    ThesisFigureSpec(
        section_title="PDI4 Session 2 — tracing acceleration magnitude",
        input_mode="behavioral",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=5,
        trial_idx_on=18,
        channel_idx=1,
        varma_run_ts_off=_TRIPLET_PDI4_S2.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI4_S2.varma_run_ts_on,
    ),
    ThesisFigureSpec(
        section_title="PDI4 Session 3 — tracing acceleration magnitude",
        input_mode="behavioral",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S3.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S3.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S3.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S3.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S3.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S3.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=9,
        trial_idx_on=25,
        channel_idx=1,
        varma_run_ts_off=_TRIPLET_PDI4_S3.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI4_S3.varma_run_ts_on,
    ),
]

# Canonical neural feature string (YAML ``neural_input``); matches ``input_channels`` in saved results.
THESIS_C2_NEURAL_Y_FEATURE_NAME = "ECOG_1_delta_05_raw"

# One neural exemplar per session (optional list — add entries like PDI1 S4 by copying the block).
THESIS_NEURAL_TIMESERIES: list[ThesisNeuralTimeseriesSpec] = [
    ThesisNeuralTimeseriesSpec(
        section_title="Neural exemplar — PDI1 Session 2",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=11,
        trial_idx_on=27,
        neural_y_channel_idx=0,
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
        varma_run_ts_off=_TRIPLET_PDI1_S2.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI1_S2.varma_run_ts_on,
    ),
    ThesisNeuralTimeseriesSpec(
        section_title="Neural exemplar — PDI1 Session 4",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S4.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S4.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S4.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S4.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S4.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S4.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=17,
        trial_idx_on=5,
        neural_y_channel_idx=0,
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
        varma_run_ts_off=_TRIPLET_PDI1_S4.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI1_S4.varma_run_ts_on,
    ),
    ThesisNeuralTimeseriesSpec(
        section_title="Neural exemplar — PDI4 Session 2",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=5,
        trial_idx_on=18,
        neural_y_channel_idx=0,
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
        varma_run_ts_off=_TRIPLET_PDI4_S2.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI4_S2.varma_run_ts_on,
    ),
    ThesisNeuralTimeseriesSpec(
        section_title="Neural exemplar — PDI4 Session 3",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S3.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S3.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S3.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S3.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S3.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S3.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        exemplar_layout="side_by_side",
        trial_idx_off=9,
        trial_idx_on=25,
        neural_y_channel_idx=0,
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
        varma_run_ts_off=_TRIPLET_PDI4_S3.varma_run_ts_off,
        varma_run_ts_on=_TRIPLET_PDI4_S3.varma_run_ts_on,
    ),
]


# ---------------------------------------------------------------------------
# C2 — Forecast time-series (one per participant/session)
# ---------------------------------------------------------------------------

# Neural Y_future: use ``THESIS_C2_NEURAL_Y_FEATURE_NAME`` / ``neural_y_feature_name`` on specs (defined above).
THESIS_C2_FORECASTS: list[ThesisC2ForecastSpec] = [
    ThesisC2ForecastSpec(
        section_title="Figure C2 — PDI1 S2 (1000 ms horizon)",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=11,
        trial_idx_on=27,
        channel_idx=0,
        caption=DEFAULT_C2_FORECAST_CAPTION,
    ),
    ThesisC2ForecastSpec(
        section_title="Figure C2 — PDI1 S4 (1000 ms horizon)",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S4.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S4.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S4.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S4.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S4.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S4.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=17,
        trial_idx_on=5,
        channel_idx=0,
        caption=DEFAULT_C2_FORECAST_CAPTION,
    ),
    ThesisC2ForecastSpec(
        section_title="Figure C2 — PDI4 S2 (1000 ms horizon)",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=5,
        trial_idx_on=18,
        channel_idx=0,
        caption=DEFAULT_C2_FORECAST_CAPTION,
    ),
    ThesisC2ForecastSpec(
        section_title="Figure C2 — PDI4 S3 (1000 ms horizon)",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S3.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S3.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S3.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S3.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S3.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S3.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=9,
        trial_idx_on=25,
        channel_idx=0,
        caption=DEFAULT_C2_FORECAST_CAPTION,
    ),
    ThesisC2ForecastSpec(
        section_title="Figure C2 — PDI1 S2 (1000 ms horizon, acceleration)",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=11,
        trial_idx_on=27,
        channel_idx=1,
        caption=DEFAULT_C2_FORECAST_CAPTION,
    ),
    ThesisC2ForecastSpec(
        section_title="Figure C2 — PDI1 S4 (1000 ms horizon, acceleration)",
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S4.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S4.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S4.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S4.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S4.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S4.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=17,
        trial_idx_on=5,
        channel_idx=1,
        caption=DEFAULT_C2_FORECAST_CAPTION,
    ),
    ThesisC2ForecastSpec(
        section_title="Figure C2 — PDI4 S2 (1000 ms horizon, acceleration)",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=5,
        trial_idx_on=18,
        channel_idx=1,
        caption=DEFAULT_C2_FORECAST_CAPTION,
    ),
    ThesisC2ForecastSpec(
        section_title="Figure C2 — PDI4 S3 (1000 ms horizon, acceleration)",
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S3.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S3.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S3.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S3.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S3.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S3.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=9,
        trial_idx_on=25,
        channel_idx=1,
        caption=DEFAULT_C2_FORECAST_CAPTION,
    ),
    ThesisC2ForecastSpec(
        section_title=(
            "Neural horizon — PDI1 S2 (Y_future, "
            + THESIS_C2_NEURAL_Y_FEATURE_NAME
            + ")"
        ),
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=11,
        trial_idx_on=27,
        channel_idx=0,
        y_axis_label="",
        caption=DEFAULT_C2_FORECAST_CAPTION,
        forecast_target="Y",
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
    ),
    ThesisC2ForecastSpec(
        section_title=(
            "Neural horizon — PDI1 S4 (Y_future, "
            + THESIS_C2_NEURAL_Y_FEATURE_NAME
            + ")"
        ),
        participant_label="PDI1",
        psid_variant=_TRIPLET_PDI1_S4.psid_variant,
        dpad_variant=_TRIPLET_PDI1_S4.dpad_variant,
        varma_variant=_TRIPLET_PDI1_S4.varma_variant,
        psid_run_ts=_TRIPLET_PDI1_S4.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI1_S4.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI1_S4.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=17,
        trial_idx_on=5,
        channel_idx=0,
        y_axis_label="",
        caption=DEFAULT_C2_FORECAST_CAPTION,
        forecast_target="Y",
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
    ),
    ThesisC2ForecastSpec(
        section_title=(
            "Neural horizon — PDI4 S2 (Y_future, "
            + THESIS_C2_NEURAL_Y_FEATURE_NAME
            + ")"
        ),
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S2.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S2.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S2.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S2.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S2.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S2.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=5,
        trial_idx_on=18,
        channel_idx=0,
        y_axis_label="",
        caption=DEFAULT_C2_FORECAST_CAPTION,
        forecast_target="Y",
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
    ),
    ThesisC2ForecastSpec(
        section_title=(
            "Neural horizon — PDI4 S3 (Y_future, "
            + THESIS_C2_NEURAL_Y_FEATURE_NAME
            + ")"
        ),
        participant_label="PDI4",
        psid_variant=_TRIPLET_PDI4_S3.psid_variant,
        dpad_variant=_TRIPLET_PDI4_S3.dpad_variant,
        varma_variant=_TRIPLET_PDI4_S3.varma_variant,
        psid_run_ts=_TRIPLET_PDI4_S3.psid_run_ts,
        dpad_run_ts=_TRIPLET_PDI4_S3.dpad_run_ts,
        varma_run_ts=_TRIPLET_PDI4_S3.varma_run_ts,
        split="test",
        use_adjacent_off_on_trials=True,
        trial_idx_off=9,
        trial_idx_on=25,
        channel_idx=0,
        y_axis_label="",
        caption=DEFAULT_C2_FORECAST_CAPTION,
        forecast_target="Y",
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
    ),
]

THESIS_CROSS_BLOCK: list[ThesisCrossBlockSpec] = [
    ThesisCrossBlockSpec(
        section_title="Cross-block decoding (neural Y)",
        participant_label="PDI1",
        joint_triplet=_TRIPLET_PDI1_S2,
        channel_idx=0,
        forecast_target="Y",
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
    ),
    ThesisCrossBlockSpec(
        section_title="Cross-block decoding (behavioral Z)",
        participant_label="PDI1",
        joint_triplet=_TRIPLET_PDI1_S2,
        channel_idx=0,
    ),
]

DEFAULT_FORECAST_CHECKPOINT_CAPTION = (
    "Per trial: true z uses history+forecast from the joint run; Ŷ from OFF / BOTH / ON checkpoints "
    "(ON on OFF trials and OFF on ON trials via cross-eval parquets)."
)

THESIS_FORECAST_CHECKPOINT: list[ThesisForecastCheckpointSpec] = [
    ThesisForecastCheckpointSpec(
        section_title="Forecast checkpoints (neural Y_future)",
        participant_label="PDI1",
        joint_triplet=_TRIPLET_PDI1_S2,
        channel_idx=0,
        forecast_target="Y",
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
        caption=DEFAULT_FORECAST_CHECKPOINT_CAPTION,
    ),
    ThesisForecastCheckpointSpec(
        section_title="Forecast checkpoints (behavioral Z_future)",
        participant_label="PDI1",
        joint_triplet=_TRIPLET_PDI1_S2,
        channel_idx=0,
        caption=DEFAULT_FORECAST_CHECKPOINT_CAPTION,
    ),
]


# ---------------------------------------------------------------------------
# D1 — PSID Cy importance (small multiples: participants × sessions)
# ---------------------------------------------------------------------------

THESIS_PSID_CY_IMPORTANCE: list[ThesisPsidCyImportanceSpec] = [
    ThesisPsidCyImportanceSpec(
        section_title="PSID — behaviourally relevant Cy importance (small multiples)",
        rows=(
            PsidCyRow(
                participant_label="PDI1",
                panels=(
                    PsidCyPanel("S2", _TRIPLET_PDI1_S2.psid_variant, _TRIPLET_PDI1_S2.psid_run_ts),
                    PsidCyPanel("S4", _TRIPLET_PDI1_S4.psid_variant, _TRIPLET_PDI1_S4.psid_run_ts),
                ),
            ),
            PsidCyRow(
                participant_label="PDI4",
                panels=(
                    PsidCyPanel("S2", _TRIPLET_PDI4_S2.psid_variant, _TRIPLET_PDI4_S2.psid_run_ts),
                    PsidCyPanel("S3", _TRIPLET_PDI4_S3.psid_variant, _TRIPLET_PDI4_S3.psid_run_ts),
                ),
            ),
        ),
        split="test",
        show_beta_box=True,
        caption=DEFAULT_PSID_CY_IMPORTANCE_CAPTION,
    ),
]


# ---------------------------------------------------------------------------
# D2 — PSID Cz behavioural readout (nz × n1 per session)
# ---------------------------------------------------------------------------

THESIS_PSID_CZ_HEATMAP: list[ThesisPsidCzSpec] = [
    ThesisPsidCzSpec(
        section_title="PSID — Cz behavioural readout matrix (nz × n₁)",
        rows=(
            PsidCyRow(
                participant_label="PDI1",
                panels=(
                    PsidCyPanel("S2", _TRIPLET_PDI1_S2.psid_variant, _TRIPLET_PDI1_S2.psid_run_ts),
                    PsidCyPanel("S4", _TRIPLET_PDI1_S4.psid_variant, _TRIPLET_PDI1_S4.psid_run_ts),
                ),
            ),
            PsidCyRow(
                participant_label="PDI4",
                panels=(
                    PsidCyPanel("S2", _TRIPLET_PDI4_S2.psid_variant, _TRIPLET_PDI4_S2.psid_run_ts),
                    PsidCyPanel("S3", _TRIPLET_PDI4_S3.psid_variant, _TRIPLET_PDI4_S3.psid_run_ts),
                ),
            ),
        ),
        split="test",
        caption=DEFAULT_PSID_CZ_CAPTION,
    ),
]


# ---------------------------------------------------------------------------
# E1 — Latent phase space (participants × sessions)
# ---------------------------------------------------------------------------

THESIS_LATENT_PHASE: list[ThesisLatentPhaseSpec] = [
    ThesisLatentPhaseSpec(
        section_title="Latent phase space — PSID (x₁ vs x₂) and DPAD (PCA of x⁽¹⁾)",
        rows=(
            LatentPhaseRow(
                participant_label="PDI1",
                panels=(
                    LatentPhasePanel(
                        session_label="S2",
                        psid_variant=_TRIPLET_PDI1_S2.psid_variant,
                        psid_run_ts=_TRIPLET_PDI1_S2.psid_run_ts,
                        dpad_variant=_TRIPLET_PDI1_S2.dpad_variant,
                        dpad_run_ts=_TRIPLET_PDI1_S2.dpad_run_ts,
                    ),
                    LatentPhasePanel(
                        session_label="S4",
                        psid_variant=_TRIPLET_PDI1_S4.psid_variant,
                        psid_run_ts=_TRIPLET_PDI1_S4.psid_run_ts,
                        dpad_variant=_TRIPLET_PDI1_S4.dpad_variant,
                        dpad_run_ts=_TRIPLET_PDI1_S4.dpad_run_ts,
                    ),
                ),
            ),
            LatentPhaseRow(
                participant_label="PDI4",
                panels=(
                    LatentPhasePanel(
                        session_label="S2",
                        psid_variant=_TRIPLET_PDI4_S2.psid_variant,
                        psid_run_ts=_TRIPLET_PDI4_S2.psid_run_ts,
                        dpad_variant=_TRIPLET_PDI4_S2.dpad_variant,
                        dpad_run_ts=_TRIPLET_PDI4_S2.dpad_run_ts,
                    ),
                    LatentPhasePanel(
                        session_label="S3",
                        psid_variant=_TRIPLET_PDI4_S3.psid_variant,
                        psid_run_ts=_TRIPLET_PDI4_S3.psid_run_ts,
                        dpad_variant=_TRIPLET_PDI4_S3.dpad_variant,
                        dpad_run_ts=_TRIPLET_PDI4_S3.dpad_run_ts,
                    ),
                ),
            ),
        ),
        split="test",
        caption=DEFAULT_LATENT_PHASE_CAPTION,
    ),
]


# ---------------------------------------------------------------------------
# F1 — Classification (empty until classification pickles are added)
# ---------------------------------------------------------------------------

DEFAULT_CLASSIFICATION_F1_CAPTION = (
    "Test-set balanced accuracy (CSP + LDA), one point per participant × session × feature source. "
    "Hyperparameters selected with chronological block-level CV; evaluation on the held-out test split. "
    "Grey horizontal segment: median per group (no SEM — small-N design). "
    "Red dashed line: chance (0.5). "
    "Asterisk: permutation test p < α from the saved pickle when present."
)

THESIS_CLASSIFICATION_F1: list[ThesisClassificationF1Spec] = [
    ThesisClassificationF1Spec(
        section_title="Figure F1 — DBS classification (balanced accuracy)",
        points=(
            # PDI1 S2
            ClassificationF1PickleRef("PDI1", "S2", "xp", "psid_behavioral_PDI1_2_nx_4_n2_i35_dbs_both_200Hz_narrow_band/20260411_000914/LDA_Xp_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S2", "xp_1", "psid_behavioral_PDI1_2_nx_4_n2_i35_dbs_both_200Hz_narrow_band/20260411_000914/LDA_Xp_1_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S2", "xp_2", "psid_behavioral_PDI1_2_nx_4_n2_i35_dbs_both_200Hz_narrow_band/20260411_000914/LDA_Xp_2_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S2", "xp_with_dbs", "psid_behavioral_PDI1_2_nx_4_n2_i35_dbs_both_200Hz_narrow_band/20260411_000914/LDA_Xp_with_dbs_prediction.pkl"),
            # PDI1 S4
            ClassificationF1PickleRef("PDI1", "S4", "xp", "psid_behavioral_PDI1_4_nx_25_n2_i50_dbs_both_200Hz_narrow_band/20260411_034923/LDA_Xp_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S4", "xp_1", "psid_behavioral_PDI1_4_nx_25_n2_i50_dbs_both_200Hz_narrow_band/20260411_034923/LDA_Xp_1_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S4", "xp_2", "psid_behavioral_PDI1_4_nx_25_n2_i50_dbs_both_200Hz_narrow_band/20260411_034923/LDA_Xp_2_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S4", "xp_with_dbs", "psid_behavioral_PDI1_4_nx_25_n2_i50_dbs_both_200Hz_narrow_band/20260411_034923/LDA_Xp_with_dbs_prediction.pkl"),
            # PDI4 S2
            ClassificationF1PickleRef("PDI4", "S2", "xp", "psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band/20260408_162132/LDA_Xp_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S2", "xp_1", "psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band/20260408_162132/LDA_Xp_1_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S2", "xp_2", "psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band/20260408_162132/LDA_Xp_2_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S2", "xp_with_dbs", "psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band/20260408_162132/LDA_Xp_with_dbs_prediction.pkl"),
            # PDI4 S3
            ClassificationF1PickleRef("PDI4", "S3", "xp", "psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band/20260408_185522/LDA_Xp_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S3", "xp_1", "psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band/20260408_185522/LDA_Xp_1_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S3", "xp_2", "psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band/20260408_185522/LDA_Xp_2_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S3", "xp_with_dbs", "psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band/20260408_185522/LDA_Xp_with_dbs_prediction.pkl"),
        ),
        caption=DEFAULT_CLASSIFICATION_F1_CAPTION,
    ),
]


# ---------------------------------------------------------------------------
# B1 — Pooled RMSE distribution (all triplets)
# ---------------------------------------------------------------------------

DEFAULT_AGGREGATE_CAPTION = (
    "Mean ± SEM across pooled test trials. Bars: lighter fill = DBS-ON, darker = DBS-OFF. "
    "Jittered dots = single-trial RMSE (z-scored to each trial’s true trace). "
    "Dot colour = participant (see legend on the box plot below: blue ≈ PDI1, green ≈ PDI4). "
    "Horizontal bracket lines with stars: paired Wilcoxon across trials (PSID vs VARMA or DPAD vs VARMA), "
    "separately for DBS-OFF and DBS-ON. "
    "Higher RMSE under DBS-ON can reflect kinematic differences, not only model error."
)

THESIS_AGGREGATE_FIGURES: list[ThesisAggregateRmseSpec] = [
    ThesisAggregateRmseSpec(
        section_title="Test-set RMSE by model × DBS (pooled) — tracing velocity",
        channel_idx=0,
        triplets=_ALL_TRIPLETS,
        split="test",
        run_wilcoxon=True,
        jitter_seed=42,
        caption=DEFAULT_AGGREGATE_CAPTION,
        show_brackets=True,
    ),
    ThesisAggregateRmseSpec(
        section_title="Test-set RMSE by model × DBS (pooled) — tracing acceleration magnitude",
        channel_idx=1,
        triplets=_ALL_TRIPLETS,
        split="test",
        run_wilcoxon=True,
        jitter_seed=43,
        caption=DEFAULT_AGGREGATE_CAPTION,
        show_brackets=True,
    ),
]


# ---------------------------------------------------------------------------
# B2 — Strip plots (one panel per participant/session)
# ---------------------------------------------------------------------------

DEFAULT_STRIP_CAPTION = (
    "Each dot = mean RMSE of test-set trials within one session. "
    "Horizontal segment = participant mean across sessions in that model × DBS cell. "
    "Circle = DBS-OFF, square = DBS-ON. RMSE on z-scored tracing speed. "
    "OFF vs ON dots differ by design (different trials and DBS states)."
)

THESIS_STRIP_PANELS: list[ThesisStripPanelsSpec] = [
    ThesisStripPanelsSpec(
        section_title="Session-mean RMSE by participant (test set)",
        channel_idx=0,
        panels=[
            StripPanelEntry("PDI1 S2", _TRIPLET_PDI1_S2),
            StripPanelEntry("PDI1 S4", _TRIPLET_PDI1_S4),
            StripPanelEntry("PDI4 S2", _TRIPLET_PDI4_S2),
            StripPanelEntry("PDI4 S3", _TRIPLET_PDI4_S3),
        ],
        split="test",
        ncols=2,
        jitter_seed=42,
        caption=DEFAULT_STRIP_CAPTION,
    ),
]


# ---------------------------------------------------------------------------
# C1 — Forecast RMSE vs horizon (all triplets)
# ---------------------------------------------------------------------------

DEFAULT_FORECAST_CAPTION = (
    "Mean absolute z-error ± SEM over test-set trials (pooled across aligned triplets), at each "
    "forecast step (sampled every 5 steps ≈ 62.5 ms at 80 Hz). "
    "Each trial uses one (μ, σ) from the true *future* window to z-score both true and pred, "
    "so the curve need not rise monotonically. Naïve baseline = predicting the mean (≈1.0 in z-units). "
    "DPAD excluded when forecast trajectories are not available for a comparable pipeline."
)

DEFAULT_NEURAL_FORECAST_CAPTION = (
    "Same pooling and z-scoring as behavioral horizon panels, applied to neural Y_future_* columns."
)

THESIS_NEURAL_FORECAST_FIGURES: list[ThesisForecastRmseSpec] = [
    ThesisForecastRmseSpec(
        section_title="Forecast RMSE vs horizon (PSID vs VARMA) — neural Y",
        channel_idx=0,
        triplets=_ALL_TRIPLETS,
        split="test",
        forecast_target="Y",
        neural_y_feature_name=THESIS_C2_NEURAL_Y_FEATURE_NAME,
        caption=DEFAULT_NEURAL_FORECAST_CAPTION,
    ),
]

THESIS_FORECAST_FIGURES: list[ThesisForecastRmseSpec] = [
    ThesisForecastRmseSpec(
        section_title="Forecast RMSE vs horizon (PSID vs VARMA) — tracing velocity",
        channel_idx=0,
        triplets=_ALL_TRIPLETS,
        split="test",
        caption=DEFAULT_FORECAST_CAPTION,
    ),
    ThesisForecastRmseSpec(
        section_title="Forecast RMSE vs horizon (PSID vs VARMA) — tracing acceleration magnitude",
        channel_idx=1,
        triplets=_ALL_TRIPLETS,
        split="test",
        caption=DEFAULT_FORECAST_CAPTION,
    ),
]


# ---------------------------------------------------------------------------
# B3 — Neural band heatmaps (all triplets; requires neural Y/Yp)
# ---------------------------------------------------------------------------

DEFAULT_NEURAL_BAND_CAPTION = (
    "Cells = mean Pearson r across test-set trials pooled over aligned triplets; "
    "within each trial, band values average r across all narrow-band channels mapped to that row "
    "(Delta–Beta, low→high). "
    "PSID and DPAD are trained for neural self-prediction; VARMA is shown for comparison "
    "when neural outputs are present in the saved split."
)

THESIS_NEURAL_BAND_HEATMAPS: list[ThesisNeuralBandHeatmapSpec] = [
    ThesisNeuralBandHeatmapSpec(
        section_title="Neural self-prediction (Pearson r) by band × model × DBS",
        triplets=_NEURAL_BAND_TRIPLETS,
        split="test",
        band_row_order=("Delta", "Theta", "Alpha", "Beta"),
        caption=DEFAULT_NEURAL_BAND_CAPTION,
    ),
]
