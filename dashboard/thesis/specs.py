"""Hardcoded thesis figure specs — edit lists here (no UI selection)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dashboard.thesis.constants import ThesisTheme

InputMode = Literal["neural", "behavioral"]

FeatureGroupF1 = Literal["psid_xp", "dpad_xp", "raw_ecog"]


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
    theme: ThesisTheme = ThesisTheme.DARK
    jitter_seed: int = 42
    caption: str = ""


@dataclass(frozen=True)
class ThesisAggregateRmseSpec:
    """Pooled test-trial RMSE bar chart (3 models × 2 DBS), mean ± SEM, jittered dots."""

    section_title: str
    channel_idx: int
    triplets: list[AlignedTriplet]
    split: str = "test"
    theme: ThesisTheme = ThesisTheme.DARK
    run_wilcoxon: bool = True
    jitter_seed: int = 42
    caption: str = ""
    show_brackets: bool = True


@dataclass(frozen=True)
class ThesisForecastRmseSpec:
    """
    Forecast horizon RMSE (z-scored tracing speed): PSID vs VARMA vs naïve baseline, two DBS panels.

    Requires `Z_future_true` and `Z_future_pred` in test parquet for both PSID and VARMA runs.
    DPAD-RNN is omitted when forecast outputs are not comparable.
    """

    section_title: str
    channel_idx: int
    triplets: list[AlignedTriplet]
    split: str = "test"
    sampling_hz: float = 60.0
    sample_every: int = 5
    naive_rmse: float = 1.0
    theme: ThesisTheme = ThesisTheme.DARK
    caption: str = ""


@dataclass(frozen=True)
class ThesisNeuralBandHeatmapSpec:
    """
    Pooled neural self-prediction: mean Pearson r between Y and Ŷ by spectral band (rows),
    model (PSID / DPAD-RNN / VARMA), and DBS (OFF vs ON panels).

    Uses `AlignedTriplet` (same as RMSE pooling): all three runs must expose aligned `Y` / `Yp`
    and matching `input_channels` for narrow-band neural columns.
    """

    section_title: str
    triplets: list[AlignedTriplet]
    split: str = "test"
    band_row_order: tuple[str, ...] = ("Delta", "Theta", "Alpha", "Beta")
    theme: ThesisTheme = ThesisTheme.DARK
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
    theme: ThesisTheme = ThesisTheme.DARK
    show_beta_box: bool = True
    caption: str = ""


@dataclass(frozen=True)
class LatentPhasePanel:
    """One session: aligned PSID and DPAD-RNN test runs (same participant/session slice)."""

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
    DPAD-RNN plots PC1 vs PC2 of x⁽¹⁾ (PCA fit on all test trials, both DBS conditions).
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
    density_percentiles: tuple[float, float] = (25.0, 55.0)
    theme: ThesisTheme = ThesisTheme.DARK
    caption: str = ""


@dataclass(frozen=True)
class ThesisC2ForecastSpec:
    """
    Figure C2: forecast time-series (2×1, DBS-OFF / DBS-ON). Same trial indices as A1 (`ThesisFigureSpec`).

    Each panel: history (true only) + forecast (true + PSID / DPAD / VARMA). Time in ms relative to t₀=0;
    history and forecast spans default to 2000 ms each (configurable).
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
    history_ms: float = 2000.0
    forecast_ms: float = 2000.0
    sampling_hz: float = 60.0
    theme: ThesisTheme = ThesisTheme.DARK
    y_axis_label: str = "z-scored tracing speed"
    caption: str = ""


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
    theme: ThesisTheme = ThesisTheme.DARK
    caption: str = ""


# --- Edit this list for your paper figures ---
DEFAULT_C2_FORECAST_CAPTION = (
    "Forecast time-series at 2000 ms horizon. Rows: DBS-OFF / DBS-ON. "
    "Left half (blue shading): observed history — true tracing speed only (model saw neural activity, not behaviour). "
    "Right half (coral shading): forecast from latent state at t₀; true continues in black. "
    "PSID: blue; DPAD-RNN: coral; VARMA: grey dashed. "
    "Shaded blue band: ±1σ PSID uncertainty, width ∝ σ√m (σ from validation residual std). "
    "RMSE values are for the forecast window only (z-scored). "
    "Same representative trial as Figure A1 (median-RMSE test trial)."
)

DEFAULT_PSID_CY_IMPORTANCE_CAPTION = (
    "Each cell: importance = ‖Cy[:, :n₁]‖₂ after reshaping rows of Cy to 4 contacts × 29 narrow-band "
    "features (row order matches `input_channels` in the saved results). "
    "Values are divided by the maximum in that panel (per-panel normalization); absolute magnitudes "
    "are not comparable across sessions (latent rotational ambiguity and session-wise signal scaling). "
    "The dashed outline marks the β band columns parsed from channel names. "
    "Residual subspace Cy[:, n₁:] is not shown (supplementary material)."
)

DEFAULT_LATENT_PHASE_CAPTION = (
    "PSID: x₁ vs x₂ in the behaviourally relevant subspace (no PCA). "
    "DPAD-RNN: PC1 vs PC2 of x⁽¹⁾; PCA was fit on all test-set timepoints (DBS-OFF and DBS-ON combined) "
    "within each panel so the projection is not biased toward separating conditions. "
    "Explained variance fractions are panel-specific (caption annotations). "
    "Filled contours: 2D Gaussian KDE at two density levels (25th and 55th percentiles of the KDE grid); "
    "thin lines: three randomly chosen trials per condition. "
    "Axes are not shared across panels (each model instance has its own latent geometry). "
    "DBS-OFF: purple; DBS-ON: teal (condition colours, distinct from model identity colours)."
)

THESIS_LATENT_PHASE: list[ThesisLatentPhaseSpec] = [
    ThesisLatentPhaseSpec(
        section_title="Latent phase space — PSID (x₁ vs x₂) and DPAD-RNN (PCA of x⁽¹⁾)",
        rows=(
            LatentPhaseRow(
                participant_label="P01",
                panels=(
                    LatentPhasePanel(
                        session_label="S4",
                        psid_variant="psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
                        psid_run_ts="20260315_142838",
                        dpad_variant="dpad_PDI1_S4_dbs_both_narrow_band",
                        dpad_run_ts="20260317_231003",
                    ),
                ),
            ),
        ),
        split="test",
        theme=ThesisTheme.DARK,
        caption=DEFAULT_LATENT_PHASE_CAPTION,
    ),
]

DEFAULT_CLASSIFICATION_F1_CAPTION = (
    "Test-set balanced accuracy (CSP + LDA), one point per participant × session × feature source. "
    "Hyperparameters selected with chronological block-level CV; evaluation on the held-out test split. "
    "Grey horizontal segment: median per group (no SEM — small-N design). "
    "Red dashed line: chance (0.5). "
    "Populate `ClassificationF1PickleRef` with paths relative to `RESULTS_PATH/classification/` "
    "(files `LDA_<feature>_prediction.pkl` from `classification/compute.py`). "
    "Asterisk: permutation test p < α from the saved pickle when present."
)

THESIS_CLASSIFICATION_F1: list[ThesisClassificationF1Spec] = [
    ThesisClassificationF1Spec(
        section_title="Figure F1 — DBS classification (balanced accuracy)",
        points=(),
        theme=ThesisTheme.DARK,
        caption=DEFAULT_CLASSIFICATION_F1_CAPTION,
    ),
    ThesisClassificationF1Spec(
        section_title="Figure F1b — DBS classification (flipped / alternate feature source)",
        points=(),
        theme=ThesisTheme.DARK,
        caption=(
            "Same layout as F1 for runs with `flipped: true`, different `prediction_feature_source` "
            "(e.g. Xp_1, Xp_2), or DBS-channel inputs. Add `ClassificationF1PickleRef` rows when "
            "classification pickles exist."
        ),
    ),
]

THESIS_PSID_CY_IMPORTANCE: list[ThesisPsidCyImportanceSpec] = [
    ThesisPsidCyImportanceSpec(
        section_title="PSID — behaviourally relevant Cy importance (small multiples)",
        rows=(
            PsidCyRow(
                participant_label="P01",
                panels=(
                    PsidCyPanel(
                        session_label="S4",
                        psid_variant="psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
                        psid_run_ts="20260315_142838",
                    ),
                ),
            ),
        ),
        split="test",
        theme=ThesisTheme.DARK,
        show_beta_box=True,
        caption=DEFAULT_PSID_CY_IMPORTANCE_CAPTION,
    ),
]

THESIS_C2_FORECASTS: list[ThesisC2ForecastSpec] = [
    ThesisC2ForecastSpec(
        section_title="Figure C2 — Forecast time-series (2000 ms horizon)",
        participant_label="P01",
        psid_variant="psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
        dpad_variant="dpad_PDI1_S4_dbs_both_narrow_band",
        varma_variant="varma_PDI1_S4_dbs_both_narrow_band",
        psid_run_ts="20260315_142838",
        dpad_run_ts="20260317_231003",
        varma_run_ts="20260316_134639",
        split="test",
        trial_idx_off=0,
        trial_idx_on=1,
        channel_idx=0,
        theme=ThesisTheme.DARK,
        caption=DEFAULT_C2_FORECAST_CAPTION,
    ),
]

THESIS_FIGURES: list[ThesisFigureSpec] = [
    ThesisFigureSpec(
        section_title="Example — PDI1 Session 4 (behavioral input, dbs_both)",
        input_mode="behavioral",
        participant_label="P01",
        psid_variant="psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
        dpad_variant="dpad_PDI1_S4_dbs_both_narrow_band",
        varma_variant="varma_PDI1_S4_dbs_both_narrow_band",
        psid_run_ts="20260315_142838",
        dpad_run_ts="20260317_231003",
        varma_run_ts="20260316_134639",
        split="test",
        trial_idx_off=0,
        trial_idx_on=1,
        channel_idx=0,
        y_axis_label="speed (z)",
        theme=ThesisTheme.LIGHT,
        caption_extra="Median-RMSE trial from test set (placeholder indices — replace after inspecting stim labels).",
    ),
]

DEFAULT_AGGREGATE_CAPTION = (
    "Mean ± SEM across test-set trials (N pooled across aligned triplets). "
    "DBS encoded by bar opacity — solid = DBS-OFF, lighter = DBS-ON. "
    "Dots = individual trial RMSE (z-scored tracing speed), jittered horizontally. "
    "Higher RMSE under DBS-ON vs OFF can reflect kinematic/signal-structure differences, not only model error."
)

# --- Pooled RMSE distribution figures (edit triplets to add participants/sessions) ---
THESIS_AGGREGATE_FIGURES: list[ThesisAggregateRmseSpec] = [
    ThesisAggregateRmseSpec(
        section_title="Test-set RMSE by model × DBS (pooled)",
        channel_idx=0,
        triplets=[
            AlignedTriplet(
                psid_variant="psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
                psid_run_ts="20260315_142838",
                dpad_variant="dpad_PDI1_S4_dbs_both_narrow_band",
                dpad_run_ts="20260317_231003",
                varma_variant="varma_PDI1_S4_dbs_both_narrow_band",
                varma_run_ts="20260316_134639",
                label="PDI1_S4",
            ),
        ],
        split="test",
        theme=ThesisTheme.DARK,
        run_wilcoxon=True,
        jitter_seed=42,
        caption=DEFAULT_AGGREGATE_CAPTION,
        show_brackets=True,
    ),
]

DEFAULT_STRIP_CAPTION = (
    "Each dot = mean RMSE of test-set trials within one session. "
    "Horizontal segment = participant mean across sessions in that model × DBS cell. "
    "Circle = DBS-OFF, square = DBS-ON. RMSE on z-scored tracing speed. "
    "Absolute RMSE can differ across participants; compare model ordering within each panel."
)

# --- B2 strip plots: one entry per panel (add/replace triplets for each participant/session) ---
_EXAMPLE_STRIP_TRIPLET = AlignedTriplet(
    psid_variant="psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
    psid_run_ts="20260315_142838",
    dpad_variant="dpad_PDI1_S4_dbs_both_narrow_band",
    dpad_run_ts="20260317_231003",
    varma_variant="varma_PDI1_S4_dbs_both_narrow_band",
    varma_run_ts="20260316_134639",
    label="PDI1_S4",
)

THESIS_STRIP_PANELS: list[ThesisStripPanelsSpec] = [
    ThesisStripPanelsSpec(
        section_title="Session-mean RMSE by participant (test set)",
        channel_idx=0,
        panels=[
            StripPanelEntry(f"P0{i}", _EXAMPLE_STRIP_TRIPLET) for i in range(1, 9)
        ],
        split="test",
        ncols=4,
        theme=ThesisTheme.DARK,
        jitter_seed=42,
        caption=DEFAULT_STRIP_CAPTION,
    ),
]

DEFAULT_NEURAL_BAND_CAPTION = (
    "Cells = mean Pearson r (Y vs Ŷ) across test-set trials pooled over aligned triplets; "
    "within each trial, band values average r across all narrow-band channels mapped to that row "
    "(Delta–Beta, low→high). "
    "PSID and DPAD-RNN are trained for neural self-prediction; VARMA is shown for comparison "
    "when neural outputs are present in the saved split. "
    "Replace `psid_variant` / `psid_run_ts` in `THESIS_NEURAL_BAND_HEATMAPS` with a neural-input "
    "PSID run that matches DPAD/VARMA channels and trial keys."
)

# Neural band heatmaps — requires PSID/DPAD/VARMA each with multi-channel Y and matching `input_channels`.
DEFAULT_FORECAST_CAPTION = (
    "Mean RMSE ± SEM over test-set trials (pooled across aligned triplets), evaluated at each "
    "forecast step (sampled every 5 steps ≈ 83 ms at 60 Hz). "
    "RMSE on z-scored tracing speed. Naïve baseline = predicting the mean (≈1.0 in z-units). "
    "DPAD-RNN excluded when forecast trajectories are not available for a comparable pipeline."
)

THESIS_FORECAST_FIGURES: list[ThesisForecastRmseSpec] = [
    ThesisForecastRmseSpec(
        section_title="Forecast RMSE vs horizon (PSID vs VARMA)",
        channel_idx=0,
        triplets=[
            AlignedTriplet(
                psid_variant="psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
                psid_run_ts="20260315_142838",
                dpad_variant="dpad_PDI1_S4_dbs_both_narrow_band",
                dpad_run_ts="20260317_231003",
                varma_variant="varma_PDI1_S4_dbs_both_narrow_band",
                varma_run_ts="20260316_134639",
                label="PDI1_S4",
            ),
        ],
        split="test",
        theme=ThesisTheme.DARK,
        caption=DEFAULT_FORECAST_CAPTION,
    ),
]

THESIS_NEURAL_BAND_HEATMAPS: list[ThesisNeuralBandHeatmapSpec] = [
    ThesisNeuralBandHeatmapSpec(
        section_title="Neural self-prediction (Pearson r) by band × model × DBS",
        triplets=[
            AlignedTriplet(
                psid_variant="psid_neural_PDI1_S2_dbs_both_narrow_band",
                psid_run_ts="REPLACE_WITH_RUN_TS",
                dpad_variant="dpad_PDI1_S2_dbs_both_narrow_band",
                dpad_run_ts="20260317_190752",
                varma_variant="varma_PDI1_S2_dbs_both_narrow_band",
                varma_run_ts="20260317_190752",
                label="PDI1_S2",
            ),
        ],
        split="test",
        band_row_order=("Delta", "Theta", "Alpha", "Beta"),
        theme=ThesisTheme.DARK,
        caption=DEFAULT_NEURAL_BAND_CAPTION,
    ),
]
