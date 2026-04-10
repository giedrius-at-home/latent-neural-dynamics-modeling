# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # Section 5: RQ3 — DBS Classification & Cross-Condition — 10 figures
#
# | # | Figure | Builder | Count |
# |---|--------|--------|-------|
# | 49 | Classification grouped bar chart | `build_classification_grouped_bar_figure()` | 1 |
# | 50 | Standard classification heatmap | `build_standard_heatmap_figure()` | 1 |
# | 51 | Flipped classification heatmap | `build_flipped_heatmap_figure()` | 1 |
# | 52 | ROC curves | `build_roc_curve_figure()` | 1 |
# | 53-54 | Within vs cross-condition RMSE | `build_within_cross_boxplot_figure()` | 2 |
# | 55-56 | Cross-block predictions | `build_cross_block_predictions_figure()` | 2 |
# | 57-58 | Forecast checkpoint comparison | `build_forecast_checkpoint_compare_figure()` | 2 |

# %%
import sys, os
os.chdir('/home/bobby/repos/latent-neural-dynamics-modeling')
sys.path.insert(0, '.')

from pathlib import Path
OUT = Path('thesis_figures/sec5'); OUT.mkdir(parents=True, exist_ok=True)

from notebooks.thesis_style import ThesisTheme
from dashboard.thesis.specs import (
    AlignedTriplet,
    ClassificationF1PickleRef,
    ThesisClassificationF1Spec,
    ThesisCrossBlockSpec,
    ThesisForecastCheckpointSpec,
    ThesisWithinCrossSpec,
)

results_root = Path('results').resolve()

# ---------------------------------------------------------------------------
# Session triplets (inline)
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

# ---------------------------------------------------------------------------
# Classification F1 spec (1 spec, 16 points)
# ---------------------------------------------------------------------------

THESIS_CLASSIFICATION_F1 = [
    ThesisClassificationF1Spec(
        section_title="Figure F1 — DBS classification (balanced accuracy)",
        points=(
            # PDI1 S2
            ClassificationF1PickleRef("PDI1", "S2", "xp", "psid_behavioral_PDI1_2_nx_25_n2_i50_dbs_both_200Hz_narrow_band/20260408_222003/LDA_Xp_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S2", "xp_1", "psid_behavioral_PDI1_2_nx_25_n2_i50_dbs_both_200Hz_narrow_band/20260408_222003/LDA_Xp_1_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S2", "xp_2", "psid_behavioral_PDI1_2_nx_25_n2_i50_dbs_both_200Hz_narrow_band/20260408_222003/LDA_Xp_2_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S2", "xp_with_dbs", "psid_behavioral_PDI1_2_nx_25_n2_i50_dbs_both_200Hz_narrow_band/20260408_222003/LDA_Xp_with_dbs_prediction.pkl"),
            # PDI1 S4 (old 80Hz narrowband classification — kept as-is since that's what exists on disk)
            ClassificationF1PickleRef("PDI1", "S4", "xp", "psid_behavioral_PDI1_4_nx_75_n6_i20_dbs_both_narrow_band/20260403_120113/LDA_Xp_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S4", "xp_1", "psid_behavioral_PDI1_4_nx_75_n6_i20_dbs_both_narrow_band/20260403_120113/LDA_Xp_1_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S4", "xp_2", "psid_behavioral_PDI1_4_nx_75_n6_i20_dbs_both_narrow_band/20260403_120113/LDA_Xp_2_prediction.pkl"),
            ClassificationF1PickleRef("PDI1", "S4", "xp_with_dbs", "psid_behavioral_PDI1_4_nx_75_n6_i20_dbs_both_narrow_band/20260403_120113/LDA_Xp_with_dbs_prediction.pkl"),
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
            # PDI4 S2 — LAPLACIAN 200Hz (most recent training, 2026-04-09)
            ClassificationF1PickleRef("PDI4", "S2 (lap)", "xp", "psid_laplacian_PDI4_2_nx_55_n8_i50_dbs_both_200Hz_narrow_band/20260409_105205/LDA_Xp_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S2 (lap)", "xp_1", "psid_laplacian_PDI4_2_nx_55_n8_i50_dbs_both_200Hz_narrow_band/20260409_105205/LDA_Xp_1_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S2 (lap)", "xp_2", "psid_laplacian_PDI4_2_nx_55_n8_i50_dbs_both_200Hz_narrow_band/20260409_105205/LDA_Xp_2_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S2 (lap)", "xp_with_dbs", "psid_laplacian_PDI4_2_nx_55_n8_i50_dbs_both_200Hz_narrow_band/20260409_105205/LDA_Xp_with_dbs_prediction.pkl"),
            # PDI4 S3 — LAPLACIAN 200Hz (most recent training, 2026-04-09; highest BAs in the set)
            ClassificationF1PickleRef("PDI4", "S3 (lap)", "xp", "psid_laplacian_PDI4_3_nx_80_n12_i50_dbs_both_200Hz_narrow_band/20260409_064142/LDA_Xp_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S3 (lap)", "xp_1", "psid_laplacian_PDI4_3_nx_80_n12_i50_dbs_both_200Hz_narrow_band/20260409_064142/LDA_Xp_1_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S3 (lap)", "xp_2", "psid_laplacian_PDI4_3_nx_80_n12_i50_dbs_both_200Hz_narrow_band/20260409_064142/LDA_Xp_2_prediction.pkl"),
            ClassificationF1PickleRef("PDI4", "S3 (lap)", "xp_with_dbs", "psid_laplacian_PDI4_3_nx_80_n12_i50_dbs_both_200Hz_narrow_band/20260409_064142/LDA_Xp_with_dbs_prediction.pkl"),
        ),
    ),
]

# ---------------------------------------------------------------------------
# Within vs cross-condition (2 specs)
# ---------------------------------------------------------------------------

THESIS_WITHIN_CROSS = [
    ThesisWithinCrossSpec(
        section_title="Within vs cross-condition decoding — PDI1 S2",
        participant_label="PDI1",
        joint_triplet=TRIPLET_PDI1_S2,
        channel_idx=0,
    ),
    ThesisWithinCrossSpec(
        section_title="Within vs cross-condition decoding — PDI4 S3",
        participant_label="PDI4",
        joint_triplet=TRIPLET_PDI4_S3,
        channel_idx=0,
    ),
]

# ---------------------------------------------------------------------------
# Cross-block predictions (2 specs)
# ---------------------------------------------------------------------------

THESIS_CROSS_BLOCK = [
    ThesisCrossBlockSpec(
        section_title="Cross-block decoding (neural Y)",
        participant_label="PDI1",
        joint_triplet=TRIPLET_PDI1_S2,
        channel_idx=0,
        forecast_target="Y",
        neural_y_feature_name="ECOG_1_theta_4_8_raw",
    ),
    ThesisCrossBlockSpec(
        section_title="Cross-block decoding (behavioral Z)",
        participant_label="PDI1",
        joint_triplet=TRIPLET_PDI1_S2,
        channel_idx=0,
    ),
]

# ---------------------------------------------------------------------------
# Forecast checkpoint comparison (2 specs)
# ---------------------------------------------------------------------------

THESIS_FORECAST_CHECKPOINT = [
    ThesisForecastCheckpointSpec(
        section_title="Forecast checkpoints (neural Y_future)",
        participant_label="PDI1",
        joint_triplet=TRIPLET_PDI1_S2,
        channel_idx=0,
        forecast_target="Y",
        neural_y_feature_name="ECOG_1_theta_4_8_raw",
        caption=(
            "Per trial: true z uses history+forecast from the joint run; "
            "\u0176 from OFF / BOTH / ON checkpoints "
            "(ON on OFF trials and OFF on ON trials via cross-eval parquets)."
        ),
    ),
    ThesisForecastCheckpointSpec(
        section_title="Forecast checkpoints (behavioral Z_future)",
        participant_label="PDI1",
        joint_triplet=TRIPLET_PDI1_S2,
        channel_idx=0,
        caption=(
            "Per trial: true z uses history+forecast from the joint run; "
            "\u0176 from OFF / BOTH / ON checkpoints "
            "(ON on OFF trials and OFF on ON trials via cross-eval parquets)."
        ),
    ),
]

# %% [markdown]
# ## Fig 49: Classification — prediction vs forecast (side-by-side)

# %%
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, Optional, Sequence

from notebooks.thesis_style import (
    COLOR_CHANCE,
    COLOR_SEPARATOR,
    FIGURE_HEIGHT,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    grid_color,
    legend_bgcolor,
    paper_colors,
    true_line_color,
)

# ---------------------------------------------------------------------------
# Inline classification-F1 data model + loaders (was dashboard.thesis.classification_f1_data)
# ---------------------------------------------------------------------------

GROUP_ORDER: Tuple[str, ...] = ("xp", "xp_1", "xp_2", "xp_with_dbs")
GROUP_DISPLAY = {
    "xp": "Xp<br>(full latent)",
    "xp_1": "Xp\u2081<br>(behav. relevant)",
    "xp_2": "Xp\u2082<br>(behav. irrelevant)",
    "xp_with_dbs": "Xp + DBS<br>(with state)",
}
GROUP_X = {"xp": 0.0, "xp_1": 1.0, "xp_2": 2.0, "xp_with_dbs": 3.0}


@dataclass(frozen=True)
class ClassificationF1Point:
    participant_label: str
    session_label: str
    group: str
    balanced_accuracy: float
    permutation_pvalue: Optional[float]
    model_label: str = "PSID"


def _extract_test_ba_and_perm(res: Dict[str, Any]) -> Tuple[float, Optional[float]]:
    tr = res.get("test_results")
    if isinstance(tr, dict) and tr.get("balanced_accuracy") is not None:
        ba = float(tr["balanced_accuracy"])
    else:
        ba = float(res.get("balanced_accuracy", float("nan")))
    pval = None
    pt = res.get("permutation_test")
    if isinstance(pt, dict) and pt.get("pvalue") is not None:
        try:
            pval = float(pt["pvalue"])
        except (TypeError, ValueError):
            pval = None
    return ba, pval


def collect_classification_f1_points(
    results_root: Path,
    refs: Sequence[ClassificationF1PickleRef],
    classification_parent: str = "classification",
) -> List[ClassificationF1Point]:
    base = results_root / classification_parent
    out: List[ClassificationF1Point] = []
    for ref in refs:
        p = base / ref.pickle_relative_path
        if not p.is_file():
            raise FileNotFoundError(f"Classification pickle not found: {p}")
        with open(p, "rb") as f:
            res = pickle.load(f)
        if not isinstance(res, dict):
            raise ValueError(f"Expected dict in pickle, got {type(res)}")
        ba, pval = _extract_test_ba_and_perm(res)
        out.append(
            ClassificationF1Point(
                participant_label=ref.participant_label,
                session_label=ref.session_label,
                group=ref.group,
                balanced_accuracy=ba,
                permutation_pvalue=pval,
                model_label=getattr(ref, "model_label", "PSID"),
            )
        )
    return out


@dataclass(frozen=True)
class ClassificationRocCurve:
    participant_label: str
    session_label: str
    group: str
    fpr: Any
    tpr: Any
    roc_auc: float


def collect_classification_roc_curves(
    results_root: Path,
    refs: Sequence[ClassificationF1PickleRef],
    classification_parent: str = "classification",
) -> List[ClassificationRocCurve]:
    base = results_root / classification_parent
    out: List[ClassificationRocCurve] = []
    for ref in refs:
        p = base / ref.pickle_relative_path
        if not p.is_file():
            continue
        with open(p, "rb") as f:
            res = pickle.load(f)
        if not isinstance(res, dict):
            continue
        tr = res.get("test_results") or res
        fpr = tr.get("fpr")
        tpr = tr.get("tpr")
        auc = tr.get("roc_auc", res.get("roc_auc", float("nan")))
        if fpr is None or tpr is None:
            continue
        try:
            out.append(ClassificationRocCurve(
                participant_label=ref.participant_label,
                session_label=ref.session_label,
                group=ref.group,
                fpr=np.asarray(fpr, dtype=float),
                tpr=np.asarray(tpr, dtype=float),
                roc_auc=float(auc),
            ))
        except Exception:
            continue
    return out


# Feature-group colours
_FEAT_COLORS: dict[str, str] = {
    "xp": "#185FA5",
    "xp_1": "#0F6E56",
    "xp_2": "#993C1D",
    "xp_with_dbs": "#854F0B",
}
_FEAT_SHORT: dict[str, str] = {
    "xp": "Xp",
    "xp_1": "Xp\u2081",
    "xp_2": "Xp\u2082",
    "xp_with_dbs": "Xp+DBS",
}


# ---------------------------------------------------------------------------
# Forecast classification BA loader — picks best (h, m) per (session, feature)
# by held-out test balanced accuracy.
# ---------------------------------------------------------------------------


def _best_forecast_test_ba(
    run_dir: Path, feat: str,
) -> Tuple[float, Optional[str]]:
    """Return (best_test_BA, best_hm_label) across h*_m* subdirs of run_dir."""
    if not run_dir.is_dir():
        return float("nan"), None
    best_ba = -1.0
    best_hm: Optional[str] = None
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("h"):
            continue
        p = d / f"LDA_{feat}_forecast.pkl"
        if not p.is_file():
            continue
        try:
            with open(p, "rb") as f:
                r = pickle.load(f)
        except Exception:
            continue
        if not isinstance(r, dict):
            continue
        tr = r.get("test_results") or {}
        tba = tr.get("balanced_accuracy", None)
        if tba is None:
            continue
        try:
            tba = float(tba)
        except (TypeError, ValueError):
            continue
        if tba > best_ba:
            best_ba = tba
            best_hm = d.name
    return (best_ba if best_ba > 0 else float("nan")), best_hm


def collect_forecast_bests(
    refs: List[ClassificationF1PickleRef],
    results_root: Path,
    classification_parent: str = "classification",
) -> Dict[Tuple[str, str], Tuple[float, Optional[str]]]:
    """For each (session, feat) in refs, return best forecast test BA + h/m label."""
    base = results_root / classification_parent
    out: Dict[Tuple[str, str], Tuple[float, Optional[str]]] = {}
    for ref in refs:
        # Derive run dir from prediction pickle path: <variant>/<ts>/LDA_*.pkl
        rel = Path(ref.pickle_relative_path)
        run_dir = base / rel.parent
        # feat name from "LDA_<Feat>_prediction.pkl"
        stem = rel.stem  # LDA_Xp_prediction
        feat = stem.replace("LDA_", "").replace("_prediction", "")
        sess_key = f"{ref.participant_label}_{ref.session_label}"
        out[(sess_key, ref.group)] = _best_forecast_test_ba(run_dir, feat)
    return out


# ---------------------------------------------------------------------------
# 2-panel builder: prediction (left) vs forecast (right)
# ---------------------------------------------------------------------------


def build_classification_pred_forecast_figure(
    points: List[ClassificationF1Point],
    forecast_bests: Dict[Tuple[str, str], Tuple[float, Optional[str]]],
    theme: ThesisTheme = ThesisTheme.LIGHT,
    exclude_groups: set[str] | None = None,
) -> Figure:
    """Two-panel grouped bar chart: left=prediction test-BA, right=best forecast test-BA."""
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    active_groups = [g for g in GROUP_ORDER if not (exclude_groups and g in exclude_groups)]

    seen: dict[str, None] = {}
    for pt in points:
        seen.setdefault(f"{pt.participant_label}_{pt.session_label}", None)
    session_labels = list(seen.keys())

    pred_lookup: dict[tuple[str, str], ClassificationF1Point] = {}
    for pt in points:
        pred_lookup[(f"{pt.participant_label}_{pt.session_label}", pt.group)] = pt

    n_sessions = len(session_labels)
    n_groups = len(active_groups)
    bar_width = 0.18
    cluster_width = n_groups * bar_width + 0.1
    tick_xs = [si * (cluster_width + 0.3) for si in range(n_sessions)]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=("Prediction (x̂ₚ at t)", "Forecast (best h, m)"),
    )

    for panel_col, (source, lookup_fn) in enumerate(
        [
            ("pred", lambda s, g: (
                pred_lookup[(s, g)].balanced_accuracy if (s, g) in pred_lookup else float("nan"),
                None,
            )),
            ("forecast", lambda s, g: forecast_bests.get((s, g), (float("nan"), None))),
        ],
        start=1,
    ):
        for gi, grp in enumerate(active_groups):
            xs: list[float] = []
            ys: list[float] = []
            hover: list[str] = []
            for si, sess in enumerate(session_labels):
                x_center = si * (cluster_width + 0.3)
                x_bar = x_center + (gi - (n_groups - 1) / 2) * bar_width
                xs.append(x_bar)
                ba, hm = lookup_fn(sess, grp)
                ys.append(ba)
                hover_txt = f"{sess}<br>{_FEAT_SHORT[grp]}<br>BA={ba:.3f}"
                if source == "forecast" and hm:
                    hover_txt += f"<br>{hm}"
                hover.append(hover_txt)

            fig.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    width=bar_width * 0.9,
                    name=_FEAT_SHORT[grp],
                    marker=dict(color=_FEAT_COLORS.get(grp, "#888888")),
                    hovertext=hover,
                    hoverinfo="text",
                    legendgroup=grp,
                    showlegend=(panel_col == 1),
                ),
                row=1, col=panel_col,
            )

        # chance line
        fig.add_hline(
            y=0.5, line_dash="dash", line_color=COLOR_CHANCE,
            line_width=1.2, layer="below",
            row=1, col=panel_col,
        )
        # session separators
        for si in range(1, n_sessions):
            xv = (tick_xs[si - 1] + tick_xs[si]) / 2
            fig.add_vline(
                x=xv, line_dash="dash", line_color=COLOR_SEPARATOR,
                line_width=0.7, opacity=0.4, layer="below",
                row=1, col=panel_col,
            )

    # BA text annotations + perm-star (prediction panel) or h/m label (forecast panel)
    for si, sess in enumerate(session_labels):
        x_center = si * (cluster_width + 0.3)
        for gi, grp in enumerate(active_groups):
            x_bar = x_center + (gi - (n_groups - 1) / 2) * bar_width

            # Prediction panel: BA + perm-star
            pt = pred_lookup.get((sess, grp))
            if pt is not None and np.isfinite(pt.balanced_accuracy):
                star = "*" if (pt.permutation_pvalue is not None and pt.permutation_pvalue < 0.05) else ""
                fig.add_annotation(
                    x=x_bar, y=pt.balanced_accuracy + 0.015,
                    text=f"{pt.balanced_accuracy:.2f}{star}",
                    showarrow=False,
                    font=dict(size=FONT_SIZE_TICK - 3, family=FONT_FAMILY),
                    xanchor="center", yanchor="bottom",
                    row=1, col=1,
                )

            # Forecast panel: BA + h/m label stacked
            ba, hm = forecast_bests.get((sess, grp), (float("nan"), None))
            if np.isfinite(ba):
                label = f"{ba:.2f}"
                if hm:
                    label += f"<br><span style='font-size:8px'>{hm}</span>"
                fig.add_annotation(
                    x=x_bar, y=ba + 0.015,
                    text=label,
                    showarrow=False,
                    font=dict(size=FONT_SIZE_TICK - 3, family=FONT_FAMILY),
                    xanchor="center", yanchor="bottom",
                    row=1, col=2,
                )

    # layout + axes
    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg, size=FONT_SIZE_BASE),
        height=FIGURE_HEIGHT + 80,
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZE_TICK),
            bgcolor=legend_bgcolor(),
        ),
        margin=dict(l=70, r=30, t=56, b=110),
        hovermode="closest",
    )

    for col in (1, 2):
        fig.update_xaxes(
            tickmode="array",
            tickvals=tick_xs,
            ticktext=session_labels,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor=fg,
            linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
            title=dict(
                text="Participant \u00d7 Session",
                font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                standoff=10,
            ),
            row=1, col=col,
        )

    fig.update_yaxes(
        title=dict(
            text="Balanced accuracy",
            font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        ),
        range=[0.0, 1.15],
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        showline=True,
        linecolor=fg,
        linewidth=1,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1, col=1,
    )
    fig.update_yaxes(
        range=[0.0, 1.15],
        showgrid=True,
        gridcolor=grid,
        zeroline=False,
        showline=True,
        linecolor=fg,
        linewidth=1,
        tickfont=dict(size=FONT_SIZE_TICK),
        row=1, col=2,
    )

    return fig


for f1_spec in THESIS_CLASSIFICATION_F1:
    cls_points = collect_classification_f1_points(
        results_root, f1_spec.points, classification_parent=f1_spec.classification_parent,
    )
    forecast_bests = collect_forecast_bests(
        list(f1_spec.points), results_root,
        classification_parent=f1_spec.classification_parent,
    )
    fig = build_classification_pred_forecast_figure(
        cls_points, forecast_bests,
        theme=f1_spec.theme, exclude_groups={'xp_with_dbs'},
    )

    fig.write_image(str(OUT / 'fig_049_classification_bar.png'), width=1400, height=680, scale=2)
    fig.show()

    session_keys: list[str] = []
    for pt in cls_points:
        k = f"{pt.participant_label}_{pt.session_label}"
        if k not in session_keys:
            session_keys.append(k)

    print(
        "Fig 49 — DBS state classification (held-out test balanced accuracy). "
        "Left panel: prediction from PSID latents x\u0302\u209a(t). "
        "Right panel: classification from the best multi-step forecast (h,m annotated), "
        "selected per-session by highest test BA. "
        "Features: Xp (full latent), Xp\u2081 (behaviour-relevant), Xp\u2082 (behaviour-irrelevant). "
        "Xp+DBS excluded (trivial leak). Chance line at 0.5. Asterisks (*) in the left panel "
        "mark permutation-test p<0.05. "
        f"Sessions plotted: {', '.join(session_keys)}. "
        "The '(lap)' rows use PSID runs with ECoG\u2192laplacian-LFP output "
        "(training 2026-04-09) and show the highest BAs of the set — PDI4_S3 (lap) reaches "
        "Xp=0.74, Xp\u2082=0.80 on prediction and Xp=0.81 on forecast (h4.5, m0.5). "
        "Other sessions use 200 Hz narrow-band PSID runs with ECoG\u2192ECoG output "
        "(2026-04-08); PDI1_S4 uses the 2026-04-03 80 Hz run pending 200 Hz retraining."
    )
    print()
    print(f"{'session':<14} {'feat':<6} {'pred_BA':<8} {'pred_p':<7} {'fcst_BA':<8} {'fcst_hm':<12}")
    for pt in cls_points:
        if pt.group == "xp_with_dbs":
            continue
        sess = f"{pt.participant_label}_{pt.session_label}"
        pp = f"{pt.permutation_pvalue:.3f}" if pt.permutation_pvalue is not None else "  n/a"
        fba, fhm = forecast_bests.get((sess, pt.group), (float("nan"), None))
        fba_s = f"{fba:.3f}" if np.isfinite(fba) else "   -"
        fhm_s = fhm if fhm else "   -"
        print(f"{sess:<14} {_FEAT_SHORT[pt.group]:<6} {pt.balanced_accuracy:.3f}    {pp}  {fba_s}   {fhm_s}")

# %% [markdown]
# ## Fig 50: Standard classification heatmap

# %%
from plotly.subplots import make_subplots
from notebooks.thesis_style import (
    FONT_SIZE_ANNOTATION,
    apply_thesis_style,
)


def build_standard_heatmap_figure(
    points: List[ClassificationF1Point],
    theme: ThesisTheme = ThesisTheme.LIGHT,
) -> Figure:
    """Sessions (rows) x feature groups (cols) balanced-accuracy heatmap."""
    paper_bg, plot_bg = paper_colors(theme)
    fg = true_line_color(theme)

    seen: dict[str, None] = {}
    for pt in points:
        key = f"{pt.participant_label}_{pt.session_label}"
        seen.setdefault(key, None)
    session_labels = list(seen.keys())
    feat_keys = list(GROUP_ORDER)
    n_rows = len(session_labels)
    n_cols = len(feat_keys)

    grid = np.full((n_rows, n_cols), float("nan"))
    for pt in points:
        sess_key = f"{pt.participant_label}_{pt.session_label}"
        ri = session_labels.index(sess_key) if sess_key in session_labels else -1
        ci = feat_keys.index(pt.group) if pt.group in feat_keys else -1
        if ri >= 0 and ci >= 0:
            grid[ri, ci] = pt.balanced_accuracy

    text_vals = [
        [f"{v:.2f}" if np.isfinite(v) else "" for v in row] for row in grid
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=[_FEAT_SHORT.get(f, f) for f in feat_keys],
            y=session_labels,
            colorscale="RdYlGn",
            zmin=0.3,
            zmax=0.9,
            zmid=0.5,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=FONT_SIZE_ANNOTATION),
            showscale=True,
            colorbar=dict(
                title=dict(text="BA", side="right"),
                len=0.5,
                thickness=12,
                tickfont=dict(size=FONT_SIZE_TICK),
            ),
            hovertemplate="Session: %{y}<br>Feature: %{x}<br>BA=%{z:.3f}<extra></extra>",
        )
    )

    apply_thesis_style(fig, theme, height=360, margin=dict(l=100, r=80, t=36, b=60))
    fig.update_yaxes(autorange="reversed")
    return fig


for f1_spec in THESIS_CLASSIFICATION_F1:
    cls_points = collect_classification_f1_points(
        results_root, f1_spec.points, classification_parent=f1_spec.classification_parent,
    )
    fig = build_standard_heatmap_figure(cls_points, theme=f1_spec.theme)
    fig.write_image(str(OUT / 'fig_050_classification_heatmap.png'), width=1100, height=500, scale=2)
    fig.show()

# %% [markdown]
# ## Fig 51: Flipped classification heatmap

# %%
import pickle


# Mapping from session label to flipped variant base + per-feature run timestamps
_FLIPPED_SESSIONS: dict[str, tuple[str, dict[str, str]]] = {
    "PDI1_S2": ("psid_behavioral_PDI1_2_nx_80_n12_i40_dbs_both_narrow_band_flipped", {
        "xp": "20260315_210934",
        "xp_1": "20260315_212227",
        "xp_2": "20260315_212605",
        "xp_with_dbs": "20260315_213629",
    }),
    "PDI1_S4": ("psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band_flipped", {
        "xp": "20260315_215824",
        "xp_1": "20260315_220623",
        "xp_2": "20260315_220842",
        "xp_with_dbs": "20260315_221545",
    }),
    "PDI4_S2": ("psid_behavioral_PDI4_2_nx_80_n10_i40_dbs_both_narrow_band_flipped", {
        "xp": "20260315_223343",
        "xp_1": "20260315_224234",
        "xp_2": "20260315_224519",
        "xp_with_dbs": "20260315_225241",
    }),
    "PDI4_S3": ("psid_behavioral_PDI4_3_nx65_n10_i40_dbs_both_narrow_band_flipped", {
        "xp": "20260315_202707",
        "xp_1": "20260315_203632",
        "xp_2": "20260315_203936",
        "xp_with_dbs": "20260315_204723",
    }),
}

_FLIPPED_FEAT_SUFFIX: dict[str, tuple[str, str]] = {
    "xp": ("", "Xp"),
    "xp_1": ("_xp_1", "Xp_1"),
    "xp_2": ("_xp_2", "Xp_2"),
    "xp_with_dbs": ("_xp_with_dbs", "Xp_with_dbs"),
}

_H_VALUES = [0.5, 1.5, 2.5, 3.5, 4.5]
_M_VALUES = [0.5, 1.0, 2.0]


def _load_flipped_heatmap_data(
    results_root,
    variant_base: str,
    run_ts: str,
    feat_suffix: str,
    feat_pkl_name: str,
) -> np.ndarray:
    """Return (len(h), len(m)) array of balanced accuracies."""
    cls_root = Path(results_root) / "classification"
    variant = variant_base + feat_suffix
    var_dir = cls_root / variant / run_ts

    grid = np.full((len(_H_VALUES), len(_M_VALUES)), float("nan"))
    for hi, h in enumerate(_H_VALUES):
        for mi, m in enumerate(_M_VALUES):
            hm_dir = var_dir / f"h{h}_m{m}"
            pkl = hm_dir / f"LDA_{feat_pkl_name}_flipped.pkl"
            if pkl.is_file():
                with open(pkl, "rb") as f:
                    res = pickle.load(f)
                tr = res.get("test_results", {})
                ba = tr.get("balanced_accuracy", res.get("balanced_accuracy", float("nan")))
                grid[hi, mi] = float(ba)
    return grid


def build_flipped_heatmap_figure(
    results_root,
    theme: ThesisTheme = ThesisTheme.LIGHT,
) -> Figure:
    """
    Single 4-row x 4-col figure: sessions (rows) x feature groups (cols).
    Each cell shows a h x m balanced-accuracy heatmap for PSID flipped predictions.
    """
    paper_bg, plot_bg = paper_colors(theme)
    fg = true_line_color(theme)

    session_keys = list(_FLIPPED_SESSIONS.keys())
    feat_keys = list(GROUP_ORDER)
    n_rows = len(session_keys)
    n_cols = len(feat_keys)

    all_subtitles = [_FEAT_SHORT[f] for f in feat_keys] + [""] * ((n_rows - 1) * n_cols)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=all_subtitles,
        horizontal_spacing=0.025,
        vertical_spacing=0.06,
    )

    h_labels = [f"{h:.1f}" for h in _H_VALUES]
    m_labels = [f"{m:.1f}" for m in _M_VALUES]

    for ri, sess_label in enumerate(session_keys):
        var_base, ts_map = _FLIPPED_SESSIONS[sess_label]
        for ci, feat in enumerate(feat_keys):
            suffix, pkl_name = _FLIPPED_FEAT_SUFFIX[feat]
            run_ts = ts_map.get(feat, ts_map.get("xp", ""))
            grid = _load_flipped_heatmap_data(
                results_root, var_base, run_ts, suffix, pkl_name
            )
            is_last = (ri == n_rows - 1) and (ci == n_cols - 1)
            text_vals = [
                [f"{v:.2f}" if np.isfinite(v) else "" for v in row]
                for row in grid
            ]
            fig.add_trace(
                go.Heatmap(
                    z=grid,
                    x=m_labels,
                    y=h_labels,
                    colorscale="RdYlGn",
                    zmin=0.3,
                    zmax=0.7,
                    zmid=0.5,
                    text=text_vals,
                    texttemplate="%{text}",
                    textfont=dict(size=FONT_SIZE_ANNOTATION - 2),
                    showscale=is_last,
                    colorbar=dict(
                        title=dict(text="BA", side="right"),
                        len=0.35,
                        x=1.01,
                        thickness=12,
                        tickfont=dict(size=FONT_SIZE_TICK),
                        tickvals=[0.3, 0.4, 0.5, 0.6, 0.7],
                        ticktext=["0.3", "0.4", "0.5", "0.6", "0.7"],
                    ) if is_last else None,
                    hovertemplate=(
                        f"{sess_label} \u2014 {_FEAT_SHORT[feat]}<br>"
                        "h=%{y} s,  m=%{x} s<br>BA=%{z:.3f}<extra></extra>"
                    ),
                ),
                row=ri + 1,
                col=ci + 1,
            )

    for ri in range(n_rows):
        for ci in range(n_cols):
            if ci == 0:
                fig.update_yaxes(
                    title_text=session_keys[ri],
                    title_font=dict(size=FONT_SIZE_TICK - 1, family=FONT_FAMILY),
                    showticklabels=True,
                    autorange="reversed",
                    tickfont=dict(size=FONT_SIZE_TICK - 1),
                    row=ri + 1,
                    col=1,
                )
            else:
                fig.update_yaxes(
                    showticklabels=False,
                    autorange="reversed",
                    row=ri + 1,
                    col=ci + 1,
                )
            if ri == n_rows - 1:
                fig.update_xaxes(
                    showticklabels=True,
                    tickfont=dict(size=FONT_SIZE_TICK - 1),
                    row=ri + 1,
                    col=ci + 1,
                )
            else:
                fig.update_xaxes(
                    showticklabels=False,
                    row=ri + 1,
                    col=ci + 1,
                )

    fig.add_annotation(
        x=-0.02, y=0.5,
        xref="paper", yref="paper",
        text="h \u2014 history (s)",
        showarrow=False,
        textangle=-90,
        font=dict(size=FONT_SIZE_LABEL - 1, family=FONT_FAMILY, color=fg),
        xanchor="right",
        yanchor="middle",
    )
    fig.add_annotation(
        x=0.46, y=-0.05,
        xref="paper", yref="paper",
        text="m \u2014 forecast horizon (s)",
        showarrow=False,
        font=dict(size=FONT_SIZE_LABEL - 1, family=FONT_FAMILY, color=fg),
        xanchor="center",
        yanchor="top",
    )

    fig.update_layout(
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=fg, size=FONT_SIZE_BASE),
        height=560,
        margin=dict(l=100, r=90, t=50, b=70),
    )

    return fig


fig = build_flipped_heatmap_figure(results_root)
fig.write_image(str(OUT / 'fig_051_flipped_heatmap.png'), width=1100, height=500, scale=2)
fig.show()

# %% [markdown]
# ## Fig 52: ROC curves — per session, one panel per (participant × session)

# %%
from notebooks.thesis_style import COLOR_CHANCE as _COLOR_CHANCE


def _load_roc_from_pickle(pkl_path: Path) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return (fpr, tpr, roc_auc) from a classification LDA pickle's test_results."""
    if not pkl_path.is_file():
        return None
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    tr = res.get("test_results") or res
    fpr = tr.get("fpr")
    tpr = tr.get("tpr")
    auc = tr.get("roc_auc", res.get("roc_auc", float("nan")))
    if fpr is None or tpr is None:
        return None
    try:
        return np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float), float(auc)
    except Exception:
        return None


def build_per_session_roc_figure(
    refs: List[ClassificationF1PickleRef],
    results_root: Path,
    theme: ThesisTheme = ThesisTheme.LIGHT,
    classification_parent: str = "classification",
    exclude_groups: set[str] | None = None,
) -> Figure:
    """One ROC panel per (participant × session); within a panel, one curve per feature group."""
    fg = true_line_color(theme)
    base = Path(results_root) / classification_parent

    # Collect unique session keys in order
    session_keys: list[str] = []
    per_session: dict[str, dict[str, ClassificationF1PickleRef]] = {}
    for ref in refs:
        sk = f"{ref.participant_label}_{ref.session_label}"
        if sk not in per_session:
            per_session[sk] = {}
            session_keys.append(sk)
        per_session[sk][ref.group] = ref

    active_groups = [g for g in GROUP_ORDER if not (exclude_groups and g in exclude_groups)]

    # Layout: 2 rows × ceil(n/2) cols (falls back to 1 row for few sessions)
    n = len(session_keys)
    ncols = 3 if n >= 5 else (2 if n >= 3 else n)
    nrows = (n + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=list(session_keys),
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
        shared_xaxes=False,
        shared_yaxes=False,
    )

    first_panel = True
    for idx, sk in enumerate(session_keys):
        row = idx // ncols + 1
        col = idx % ncols + 1

        for feat in active_groups:
            ref = per_session[sk].get(feat)
            if ref is None:
                continue
            pkl_path = base / ref.pickle_relative_path
            roc = _load_roc_from_pickle(pkl_path)
            if roc is None:
                continue
            fpr, tpr, auc = roc
            color = _FEAT_COLORS.get(feat, "#888888")
            label = f"{_FEAT_SHORT.get(feat, feat)} (AUC={auc:.2f})"
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode="lines",
                    name=_FEAT_SHORT.get(feat, feat),
                    line=dict(color=color, width=2.0),
                    hovertemplate=f"{sk}<br>{label}<br>FPR=%{{x:.2f}}  TPR=%{{y:.2f}}<extra></extra>",
                    legendgroup=feat,
                    showlegend=first_panel,
                ),
                row=row, col=col,
            )

        # Chance diagonal
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color=_COLOR_CHANCE, width=1.0, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row, col=col,
        )
        first_panel = False

    apply_thesis_style(
        fig, theme,
        height=260 * nrows + 120,
        margin=dict(l=70, r=30, t=60, b=90),
    )

    for idx in range(n):
        row = idx // ncols + 1
        col = idx % ncols + 1
        fig.update_xaxes(
            title_text="FPR" if row == nrows else "",
            title_font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
            range=[0, 1],
            row=row, col=col,
        )
        fig.update_yaxes(
            title_text="TPR" if col == 1 else "",
            title_font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
            range=[0, 1.02],
            row=row, col=col,
        )

    return fig


# Use same refs as Fig 49 (prediction pickles from the 6 sessions)
_roc_refs = list(THESIS_CLASSIFICATION_F1[0].points)
fig = build_per_session_roc_figure(_roc_refs, results_root, exclude_groups={'xp_with_dbs'})
fig.write_image(str(OUT / 'fig_052_roc_curves.png'), width=1300, height=760, scale=2)
fig.show()

_session_labels_roc = []
for r in _roc_refs:
    sk = f"{r.participant_label}_{r.session_label}"
    if sk not in _session_labels_roc:
        _session_labels_roc.append(sk)
print(
    "Fig 52 — Per-session ROC curves (held-out test split). "
    "One panel per participant\u00d7session; within each panel one curve per feature group "
    "(Xp, Xp\u2081, Xp\u2082). AUC values in the legend entries of each panel. "
    "Chance diagonal in red. Xp+DBS excluded (trivial leak). "
    f"Sessions plotted: {', '.join(_session_labels_roc)}. "
    "Data source: same prediction pickles used in Fig 49 — PDI4 sessions (lap) use the "
    "2026-04-09 laplacian LFP runs; other sessions use the 2026-04-08 200 Hz narrow-band "
    "PSID runs; PDI1_S4 uses the 2026-04-03 80 Hz run pending 200 Hz retraining."
)

# %% [markdown]
# ## Figs 53-54: Within vs cross-condition RMSE boxplot

# %%
from dashboard.thesis.aggregate_rmse import collect_within_cross_rmse, WithinCrossRmseData
from notebooks.thesis_style import (
    COLOR_PSID, COLOR_DPAD, COLOR_VARMA, DOT_SIZE,
    rmse_axis_label,
)

# Boxplot geometry (lifted from dashboard.thesis.fig_within_cross)
_WC_BW = 0.22
_WC_GAP = 0.06
_WC_GROUP_GAP = 0.45


def _wc_hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_within_cross_boxplot_figure(
    data: WithinCrossRmseData,
    theme: ThesisTheme = ThesisTheme.LIGHT,
    jitter_seed: int = 42,
) -> Figure:
    """2 panels (DBS-OFF | DBS-ON), 3 model groups, within + cross bars each."""
    models = ["PSID", "DPAD", "VARMA"]
    model_colors = [COLOR_PSID, COLOR_DPAD, COLOR_VARMA]
    off_cells = [
        (data.psid_off_within, data.psid_off_cross),
        (data.dpad_off_within, data.dpad_off_cross),
        (data.varma_off_within, data.varma_off_cross),
    ]
    on_cells = [
        (data.psid_on_within, data.psid_on_cross),
        (data.dpad_on_within, data.dpad_on_cross),
        (data.varma_on_within, data.varma_on_cross),
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["DBS-OFF", "DBS-ON"],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    rng = np.random.default_rng(jitter_seed)
    y_max = 0.0

    for col_idx, cells in enumerate([off_cells, on_cells]):
        x_pos = 0.0
        tick_positions = []
        for mi, (within_vals, cross_vals) in enumerate(cells):
            x_within = x_pos
            x_cross = x_pos + _WC_BW + _WC_GAP

            for vals, x, dashed, alpha in [
                (within_vals, x_within, False, 0.55),
                (cross_vals, x_cross, True, 0.35),
            ]:
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) == 0:
                    continue
                y_max = max(y_max, float(np.nanmax(arr)))

                col = model_colors[mi]
                fillcolor = _wc_hex_to_rgba(col, 0.30) if not dashed else "rgba(0,0,0,0)"

                fig.add_trace(
                    go.Box(
                        y=arr, x=[x] * len(arr),
                        marker_color=col, line=dict(color=col, width=1.2),
                        fillcolor=fillcolor, boxpoints=False, quartilemethod="exclusive",
                        name=f"{models[mi]} {'cross' if dashed else 'within'}",
                        showlegend=(col_idx == 0),
                        legendgroup=f"{models[mi]}_{'cross' if dashed else 'within'}",
                    ),
                    row=1, col=col_idx + 1,
                )
                jitter = rng.uniform(-0.08, 0.08, size=len(arr))
                fig.add_trace(
                    go.Scatter(
                        x=np.array([x] * len(arr)) + jitter, y=arr,
                        mode="markers",
                        marker=dict(size=DOT_SIZE, color=col, opacity=alpha, line=dict(width=0)),
                        showlegend=False,
                    ),
                    row=1, col=col_idx + 1,
                )

            tick_positions.append(x_within + (_WC_BW + _WC_GAP) / 2)
            x_pos += 2 * _WC_BW + _WC_GAP + _WC_GROUP_GAP

        fig.update_xaxes(
            tickvals=tick_positions, ticktext=models,
            row=1, col=col_idx + 1,
        )

    y_max = max(y_max * 1.08, 0.85)
    apply_thesis_style(
        fig, theme, height=400,
        margin=dict(l=60, r=20, t=50, b=100),
        legend_y=-0.2,
    )
    fig.update_layout(
        yaxis=dict(range=[0, y_max]),
        yaxis2=dict(range=[0, y_max]),
    )
    for col in (1, 2):
        fig.update_xaxes(showgrid=False, row=1, col=col)
    fig.update_yaxes(
        title_text=rmse_axis_label(),
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        row=1, col=1,
    )
    return fig


fig_num = 53
for wc_spec in THESIS_WITHIN_CROSS:
    wc_data = collect_within_cross_rmse(
        results_root, [wc_spec.joint_triplet], wc_spec.channel_idx, split=wc_spec.split,
    )
    fig = build_within_cross_boxplot_figure(wc_data, theme=wc_spec.theme)
    fname = f'fig_{fig_num:03d}_within_cross_{wc_spec.participant_label}.png'
    fig.write_image(str(OUT / fname), width=1000, height=600, scale=2)
    fig.show()
    trip = wc_spec.joint_triplet
    print(
        f"Fig {fig_num} \u2014 Within vs cross-condition decoding RMSE(z) boxplots for "
        f"{wc_spec.participant_label} ({trip.label}). "
        "Each panel = one DBS condition (OFF | ON). Bars: PSID, DPAD, VARMA; "
        "solid = within-condition (train and test on same DBS state), "
        "dashed/open = cross-condition (train on one state, test on the other). "
        f"Channel index {wc_spec.channel_idx}. "
        f"Data source: joint run {trip.psid_variant}/{trip.psid_run_ts} (PSID), "
        f"{trip.varma_variant}/{trip.varma_run_ts} (VARMA); DPAD shown if "
        f"dpad_run_ts is populated for this triplet."
    )
    fig_num += 1

# %% [markdown]
# ## Figs 55-56: Cross-block predictions

# %%
import logging
# Data-fetching helpers stay imported (match lift-and-shift rule: builders inline, loaders imported)
from dashboard.thesis.cross_block_predictions import (
    _resolve_y_channel, _load_fw, _maybe_reload_off_on_for_boundary,
    _extract_single_segment, _line_w,
    COLOR_OFF as _XB_COLOR_OFF,
    COLOR_BOTH as _XB_COLOR_BOTH,
    COLOR_ON as _XB_COLOR_ON,
    _CONTEXT as _XB_CONTEXT,
    _FWD as _XB_FWD,
)
from dashboard.thesis.exemplar_trials import (
    find_adjacent_off_then_on_trial_indices,
    find_block_boundary_off_then_on_trial_indices,
)
from dashboard.thesis.loaders import (
    load_split_results_required,
    neural_y_feature_label,
    resolve_output_channel_display,
    thesis_exemplar_tagline,
)
from dashboard.thesis.aggregate_rmse import _trial_key
from dashboard.thesis.plot_scaffolds import (
    THESIS_TIME_AXIS_TITLE,
    apply_grid_xy_subplots,
    cross_block_annotation_font,
)
from dashboard.thesis.specs import THESIS_DECLARED_BEHAVIORAL_OUTPUTS
from notebooks.thesis_style import WIDTH_TRUE, d_score_axis_label

_xb_logger = logging.getLogger("thesis_sec5_cross_block")


def build_cross_block_predictions_figure(
    spec,
    results_root: Path,
):
    """Stitched OFF→ON and ON→OFF trial windows, one row per framework (PSID/DPAD/VARMA)."""
    res_joint = load_split_results_required(
        results_root,
        spec.joint_triplet.psid_variant,
        spec.joint_triplet.psid_run_ts,
        spec.split,
    )
    pair = find_block_boundary_off_then_on_trial_indices(res_joint)
    if pair is None:
        stim = res_joint.get("stim")
        sl = list(stim) if stim is not None else []
        pair = find_adjacent_off_then_on_trial_indices(sl)
    if pair is None:
        raise ValueError("Cross-block figure: no OFF\u2192ON trial pair in joint PSID results.")
    i_off, i_on = pair

    ch = _resolve_y_channel(res_joint, spec)
    y_mode = "Y" if spec.forecast_target == "Y" else "Z"
    span = float(spec.trial_segment_s)

    c_true = true_line_color(spec.theme)

    fw_list = ["psid", "dpad", "varma"]
    skipped: list[str] = []
    built: list = []

    for fw in fw_list:
        try:
            packs = _load_fw(spec.joint_triplet, fw, results_root, spec.split)
        except Exception as e:
            _xb_logger.warning("Cross-block %s: load failed (%s)", fw, e)
            skipped.append(f"{fw} (load: {str(e)[:80]})")
            continue
        res_j = packs.res_j
        k_off = _trial_key(res_j, i_off)
        k_on = _trial_key(res_j, i_on)
        res_o, res_n = _maybe_reload_off_on_for_boundary(
            results_root, packs, res_j, k_off, k_on, i_off, i_on, y_mode, spec.split,
        )
        seg_off = _extract_single_segment(
            res_j, res_o, res_n, packs.res_eo, packs.res_en,
            i_off, True, ch, y_mode, span,
        )
        seg_on = _extract_single_segment(
            res_j, res_o, res_n, packs.res_eo, packs.res_en,
            i_on, False, ch, y_mode, span,
        )
        if seg_off is None or seg_on is None:
            _xb_logger.warning("Cross-block %s: segment extraction failed.", fw)
            skipped.append(f"{fw} (segment)")
            continue
        built.append((fw, seg_off, seg_on))

    if not built:
        detail = "; ".join(skipped) if skipped else "no detail"
        raise ValueError(f"Cross-block figure: no framework produced data. Skipped: {detail}")

    n_rows = len(built)
    subplot_titles: list[str] = []
    for fw, _, _ in built:
        subplot_titles.append(f"{fw.upper()}: last OFF trial")
        subplot_titles.append(f"{fw.upper()}: first ON trial")

    fig = make_subplots(
        rows=n_rows, cols=2,
        shared_yaxes=True,
        shared_xaxes=False,
        vertical_spacing=min(0.16, 0.065 + 0.035 * n_rows),
        horizontal_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    for ri, (fw, seg_off, seg_on) in enumerate(built, start=1):
        lw = _line_w(fw)
        dash_on = "8 2" if fw == "varma" else None

        for ci, (seg, bg_color) in enumerate(
            [(seg_off, _XB_CONTEXT), (seg_on, _XB_FWD)], start=1
        ):
            t_p, zt, zo, zb, zn = seg
            show_leg = ri == 1 and ci == 1

            x0 = float(np.nanmin(t_p))
            x1 = float(np.nanmax(t_p))
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor=bg_color,
                layer="below", line_width=0, row=ri, col=ci,
            )

            fig.add_trace(
                go.Scatter(
                    x=t_p, y=zt, name="y_true", mode="lines",
                    line=dict(color=c_true, width=WIDTH_TRUE),
                    showlegend=show_leg, legendgroup="true", connectgaps=False,
                ),
                row=ri, col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_p, y=zo, name="\u0176 OFF-trained", mode="lines",
                    line=dict(color=_XB_COLOR_OFF, width=lw),
                    showlegend=show_leg, legendgroup="off", connectgaps=False,
                ),
                row=ri, col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_p, y=zb, name="\u0176 BOTH-trained", mode="lines",
                    line=dict(color=_XB_COLOR_BOTH, width=lw),
                    showlegend=show_leg, legendgroup="both", connectgaps=False,
                ),
                row=ri, col=ci,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_p, y=zn, name="\u0176 ON-trained", mode="lines",
                    line=dict(color=_XB_COLOR_ON, width=lw, dash=dash_on),
                    showlegend=show_leg, legendgroup="on", connectgaps=False,
                ),
                row=ri, col=ci,
            )

    apply_grid_xy_subplots(
        fig, n_rows=n_rows, n_cols=2, theme=spec.theme,
        nticks=8, x_title=THESIS_TIME_AXIS_TITLE,
    )
    apply_thesis_style(
        fig, spec.theme,
        height=int(min(960, max(380, 260 * n_rows + 180))),
        margin=dict(l=80, r=28, t=56, b=120),
        legend_y=-0.08,
    )
    fig.update_annotations(font=cross_block_annotation_font(theme=spec.theme))

    _ylab_override = (spec.y_axis_label or "").strip()
    feat_disp = ""
    och = ""
    if _ylab_override:
        ylab = _ylab_override
    elif y_mode == "Z":
        och, _ = resolve_output_channel_display(
            res_joint, ch, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
        )
        ylab = d_score_axis_label(och)
    else:
        feat_disp = neural_y_feature_label(
            res_joint, ch, neural_y_feature_name=spec.neural_y_feature_name,
        )
        ylab = feat_disp

    y_title_row = max(1, (n_rows + 1) // 2)
    fig.update_yaxes(title_text=ylab, row=y_title_row, col=1)

    feat = och if y_mode == "Z" else feat_disp
    cap = thesis_exemplar_tagline(
        res_joint, i_off, i_on, feat, participant_label=spec.participant_label,
    )
    sc = (spec.caption or "").strip()
    if sc:
        cap = f"{sc} \u00b7 {cap}"
    cap += f" Left: last OFF trial ({span:.1f} s); right: first ON trial ({span:.1f} s). Independent x-axes."
    if skipped:
        cap += f" \u00b7 Skipped: {'; '.join(skipped)}."
    return fig, cap


fig_num = 55
for xb_spec in THESIS_CROSS_BLOCK:
    try:
        fig, cap = build_cross_block_predictions_figure(xb_spec, results_root)
    except Exception as e:
        print(f"Fig {fig_num} \u2014 SKIPPED: {e}")
        fig_num += 1
        continue
    fname = f'fig_{fig_num:03d}_cross_block_{xb_spec.section_title.replace(" ", "_")[:30]}.png'
    fig.write_image(str(OUT / fname), width=1100, height=800, scale=2)
    fig.show()
    trip = xb_spec.joint_triplet
    tgt = "neural Y" if xb_spec.forecast_target == "Y" else "behavioral Z"
    print(
        f"Fig {fig_num} \u2014 Cross-block decoding ({tgt}) for {xb_spec.participant_label} "
        f"({trip.label}). Rows: PSID / DPAD / VARMA. Columns: last OFF trial (left) | "
        f"first ON trial (right). Traces: y_true (black), OFF-trained \u0176 (blue), "
        f"BOTH-trained \u0176 (green), ON-trained \u0176 (red-dashed for VARMA). "
        f"Data source: joint run {trip.psid_variant}/{trip.psid_run_ts}; condition-specific "
        f"OFF/ON checkpoints from the same triplet. Caption: {cap}"
    )
    fig_num += 1

# %% [markdown]
# ## Figs 57-58: Forecast checkpoint comparison

# %%
# Data-fetching helpers stay imported; builder is inlined
from dashboard.thesis.forecast_checkpoint_compare import (
    _varma_ref_and_trial, _build_checkpoint_cell,
    _CONTEXT as _FC_CONTEXT,
    _FORECAST as _FC_FORECAST,
    _RULE as _FC_RULE,
    _DIVERGENCE_THRESHOLD as _FC_DIVERGENCE_THRESHOLD,
)
from dashboard.thesis.cross_block_predictions import _aligned_trial_idx
from dashboard.thesis.loaders import (
    load_split_results,
    resolve_neural_y_channel_idx_from_candidates,
    split_res_with_nonempty_input_channels,
)


def build_forecast_checkpoint_compare_figure(spec, results_root: Path):
    """3 rows (PSID/DPAD/VARMA) x 2 cols (OFF trial | ON trial) multi-step forecast comparison
    across OFF / BOTH / ON training checkpoints."""
    res_psid = load_split_results_required(
        results_root,
        spec.joint_triplet.psid_variant,
        spec.joint_triplet.psid_run_ts,
        spec.split,
    )
    pair = find_block_boundary_off_then_on_trial_indices(res_psid)
    if pair is None:
        stim = res_psid.get("stim")
        sl = list(stim) if stim is not None else []
        pair = find_adjacent_off_then_on_trial_indices(sl)
    if pair is None:
        raise ValueError("Forecast checkpoint figure: no OFF\u2192ON trial pair in joint PSID results.")
    i_off, i_on = pair

    tri = spec.joint_triplet
    res_d = load_split_results(
        results_root, tri.dpad_variant, tri.dpad_run_ts, spec.split,
    ) if tri.dpad_run_ts else None
    res_v = load_split_results_required(
        results_root, tri.varma_variant, tri.varma_run_ts, spec.split,
    )
    if spec.forecast_target == "Y":
        ch = resolve_neural_y_channel_idx_from_candidates(
            spec.neural_y_feature_name, spec.channel_idx, res_psid, res_d, res_v,
        )
    else:
        ch = spec.channel_idx
    meta_res = split_res_with_nonempty_input_channels(res_psid, res_d, res_v) or res_psid
    y_mode = "Y" if spec.forecast_target == "Y" else "Z"
    fg = true_line_color(spec.theme)
    c_true = true_line_color(spec.theme)

    fw_list = ["psid", "dpad", "varma"]
    skipped: list[str] = []
    built_rows: list = []

    subplot_titles: list[str] = []
    for fw in fw_list:
        subplot_titles.extend([
            f"{fw.upper()} \u00b7 DBS-OFF trial (multi-step \u0176)",
            f"{fw.upper()} \u00b7 DBS-ON trial (multi-step \u0176)",
        ])

    fig = make_subplots(
        rows=3, cols=2,
        shared_yaxes="rows",
        shared_xaxes=False,
        vertical_spacing=0.09,
        horizontal_spacing=0.06,
        subplot_titles=tuple(subplot_titles),
    )

    def _insert_hist_forecast_gap(arr, n_hist):
        # Matches dashboard.thesis.figure._insert_hist_forecast_gap minus gap visual
        if n_hist <= 0 or n_hist >= len(arr):
            return arr
        out = np.concatenate([arr[:n_hist], [np.nan], arr[n_hist:]])
        return out

    for ri, fw in enumerate(fw_list, start=1):
        try:
            pack = _load_fw(tri, fw, results_root, spec.split)
        except Exception as e:
            _xb_logger.warning("Forecast checkpoint %s: load failed (%s)", fw, e)
            skipped.append(f"{fw} (load)")
            continue

        res_j = pack.res_j
        k_off = _trial_key(res_j, i_off)
        k_on = _trial_key(res_j, i_on)
        res_o, res_n = _maybe_reload_off_on_for_boundary(
            results_root, pack, res_j, k_off, k_on, i_off, i_on, y_mode, spec.split,
        )

        io = _aligned_trial_idx(res_o, res_j, k_off, i_off, y_mode)
        ion = _aligned_trial_idx(res_n, res_j, k_on, i_on, y_mode)
        ieo = _aligned_trial_idx(pack.res_eo, res_j, k_off, i_off, y_mode)
        ien = _aligned_trial_idx(pack.res_en, res_j, k_on, i_on, y_mode)

        i_j_off = _aligned_trial_idx(res_j, res_j, k_off, i_off, y_mode) or i_off
        i_j_on = _aligned_trial_idx(res_j, res_j, k_on, i_on, y_mode) or i_on

        res_ref_off, tr_off = _varma_ref_and_trial(
            tri, results_root, spec.split, res_psid, res_j, i_off, "off",
        )
        res_ref_on, tr_on = _varma_ref_and_trial(
            tri, results_root, spec.split, res_psid, res_j, i_on, "on",
        )

        cell_off = _build_checkpoint_cell(
            res_ref=res_ref_off, trial_ref=tr_off,
            tripreds=[
                ("OFF-trained", res_o, io),
                ("BOTH-trained", res_j, i_j_off),
                ("ON-trained (cross)", pack.res_eo, ieo),
            ],
            channel_idx=ch, forecast_target=spec.forecast_target,
            history_ms=spec.history_ms, forecast_ms=spec.forecast_ms,
            sampling_hz=spec.sampling_hz,
        )
        cell_on = _build_checkpoint_cell(
            res_ref=res_ref_on, trial_ref=tr_on,
            tripreds=[
                ("OFF-trained (cross)", pack.res_en, ien),
                ("BOTH-trained", res_j, i_j_on),
                ("ON-trained", res_n, ion),
            ],
            channel_idx=ch, forecast_target=spec.forecast_target,
            history_ms=spec.history_ms, forecast_ms=spec.forecast_ms,
            sampling_hz=spec.sampling_hz,
        )

        if cell_off is None and cell_on is None:
            skipped.append(f"{fw} (panels)")
            continue
        built_rows.append((fw, cell_off is not None, cell_on is not None))

        lw = _line_w(fw)

        def _add_cell(ci, cell):
            if cell is None:
                return
            t_abs, z_true, z_off, z_both, z_on, n_hist = cell

            if fw == "dpad":
                fc_arrays = [z_off[n_hist:], z_both[n_hist:], z_on[n_hist:]]
                max_abs = max(
                    (float(np.nanmax(np.abs(a))) for a in fc_arrays if np.any(np.isfinite(a))),
                    default=0.0,
                )
                if max_abs > _FC_DIVERGENCE_THRESHOLD:
                    fig.add_annotation(
                        text="<i>DPAD forecast diverged</i>",
                        x=0.5, y=0.5, xref="x domain", yref="y domain",
                        showarrow=False,
                        font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                        row=ri, col=ci,
                    )
                    return

            t_plot = _insert_hist_forecast_gap(np.asarray(t_abs, dtype=float).ravel(), n_hist)
            z_true_p = _insert_hist_forecast_gap(np.asarray(z_true, dtype=float).ravel(), n_hist)
            z_off_p = _insert_hist_forecast_gap(np.asarray(z_off, dtype=float).ravel(), n_hist)
            z_both_p = _insert_hist_forecast_gap(np.asarray(z_both, dtype=float).ravel(), n_hist)
            z_on_p = _insert_hist_forecast_gap(np.asarray(z_on, dtype=float).ravel(), n_hist)

            t_abs_arr = np.asarray(t_abs, dtype=float).ravel()
            if t_abs_arr.size and n_hist > 0:
                fig.add_vrect(
                    x0=float(t_abs_arr[0]), x1=float(t_abs_arr[n_hist - 1]),
                    fillcolor=_FC_CONTEXT, layer="below", line_width=0,
                    row=ri, col=ci,
                )
            if t_abs_arr.size and n_hist < len(t_abs_arr):
                fig.add_vrect(
                    x0=float(t_abs_arr[n_hist]), x1=float(t_abs_arr[-1]),
                    fillcolor=_FC_FORECAST, layer="below", line_width=0,
                    row=ri, col=ci,
                )
                fig.add_vline(
                    x=float(t_abs_arr[n_hist]),
                    line=dict(color=_FC_RULE, width=0.5, dash="dash"),
                    row=ri, col=ci,
                )

            show_leg = ri == 1 and ci == 1
            fig.add_trace(go.Scatter(
                x=t_plot, y=z_true_p, mode="lines", name="y_true",
                legendgroup="true",
                line=dict(color=c_true, width=WIDTH_TRUE),
                showlegend=show_leg, connectgaps=False,
            ), row=ri, col=ci)
            fig.add_trace(go.Scatter(
                x=t_plot, y=z_off_p, mode="lines", name="\u0176 OFF-trained",
                legendgroup="off",
                line=dict(color=_XB_COLOR_OFF, width=lw),
                showlegend=show_leg, connectgaps=False,
            ), row=ri, col=ci)
            fig.add_trace(go.Scatter(
                x=t_plot, y=z_both_p, mode="lines", name="\u0176 BOTH-trained",
                legendgroup="both",
                line=dict(color=_XB_COLOR_BOTH, width=lw),
                showlegend=show_leg, connectgaps=False,
            ), row=ri, col=ci)
            fig.add_trace(go.Scatter(
                x=t_plot, y=z_on_p, mode="lines", name="\u0176 ON-trained",
                legendgroup="on",
                line=dict(color=_XB_COLOR_ON, width=lw),
                showlegend=show_leg, connectgaps=False,
            ), row=ri, col=ci)

        _add_cell(1, cell_off)
        _add_cell(2, cell_on)

    if not built_rows:
        detail = "; ".join(skipped) if skipped else "no detail"
        raise ValueError(
            f"Forecast checkpoint figure: no framework produced a panel. Skipped: {detail}"
        )

    apply_grid_xy_subplots(
        fig, n_rows=3, n_cols=2, theme=spec.theme,
        nticks=12, x_title=THESIS_TIME_AXIS_TITLE,
    )
    apply_thesis_style(
        fig, spec.theme,
        height=int(min(1100, max(520, 320 * 3 + 140))),
        margin=dict(l=72, r=24, t=56, b=128),
        legend_y=-0.08,
    )
    fig.update_annotations(font=cross_block_annotation_font(theme=spec.theme))

    _ylab_override = (spec.y_axis_label or "").strip()
    feat_disp = ""
    och = ""
    if _ylab_override:
        ylab = _ylab_override
        feat = ""
    elif y_mode == "Z":
        och, _ = resolve_output_channel_display(
            res_psid, ch, declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
        )
        ylab = d_score_axis_label(och)
        feat = och
    else:
        feat_disp = neural_y_feature_label(
            meta_res, ch, neural_y_feature_name=spec.neural_y_feature_name,
        )
        ylab = feat_disp
        feat = feat_disp

    fig.update_yaxes(title_text=ylab, row=2, col=1)

    cap = thesis_exemplar_tagline(
        res_psid, i_off, i_on, feat, participant_label=spec.participant_label,
    )
    sc = (spec.caption or "").strip()
    if sc:
        cap = f"{sc} \u00b7 {cap}"
    if skipped:
        cap += f" \u00b7 Skipped: {'; '.join(skipped)}."
    return fig, cap


fig_num = 57
for fc_ck_spec in THESIS_FORECAST_CHECKPOINT:
    try:
        fig, cap = build_forecast_checkpoint_compare_figure(fc_ck_spec, results_root)
    except Exception as e:
        print(f"Fig {fig_num} \u2014 SKIPPED: {e}")
        fig_num += 1
        continue
    fname = f'fig_{fig_num:03d}_forecast_ckpt_{fc_ck_spec.section_title.replace(" ", "_")[:30]}.png'
    fig.write_image(str(OUT / fname), width=1100, height=800, scale=2)
    fig.show()
    trip = fc_ck_spec.joint_triplet
    tgt = "neural Y_future" if fc_ck_spec.forecast_target == "Y" else "behavioral Z_future"
    print(
        f"Fig {fig_num} \u2014 Forecast checkpoint comparison ({tgt}) for "
        f"{fc_ck_spec.participant_label} ({trip.label}). "
        "Rows: PSID / DPAD / VARMA. Columns: DBS-OFF trial (left) | DBS-ON trial (right). "
        f"Each row shows history (shaded blue) + multi-step forecast (shaded red) with "
        f"history={fc_ck_spec.history_ms} ms, forecast={fc_ck_spec.forecast_ms} ms at "
        f"{fc_ck_spec.sampling_hz} Hz. Traces: y_true (black), \u0176 OFF-trained (blue), "
        f"\u0176 BOTH-trained (green), \u0176 ON-trained (red) \u2014 cross-eval checkpoints "
        "apply ON-trained model on OFF trials and vice versa. Caption: " + cap
    )
    fig_num += 1

# %%
n = len(list(OUT.glob('*.png')))
print(f'Section 5 total: {n} figures (expected 10)')
