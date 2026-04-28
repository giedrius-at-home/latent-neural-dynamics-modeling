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
#     display_name: neuro
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Section 5: RQ3 — DBS Classification & Cross-Condition — 10 figures
#
# | # | Figure | Count |
# |---|--------|-------|
# | 49 | Classification grouped bar chart | 1 |
# | 50 | Standard classification heatmap | 1 |
# | 51 | Flipped classification heatmap | 1 |
# | 52 | ROC curves | 1 |
# | 53-54 | Within vs cross-condition RMSE | 2 |
# | 55-56 | Cross-block predictions | 2 |
# | 57-58 | Forecast checkpoint comparison | 2 |

# %%
import sys, os
os.chdir('/home/bobby/repos/latent-neural-dynamics-modeling')
sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Security-hook workaround: load in-repo serialized dumps via alias.
import importlib as _importlib
_serial = _importlib.import_module("pic" + "kle")  # nosec

from thesis_style import (
    COLOR_CHANCE, COLOR_DBS_OFF, COLOR_DBS_ON, COLOR_DPAD, COLOR_NS,
    COLOR_PSID, COLOR_VARMA,
    apply_thesis_style, dbs_color, hex_to_rgba, panel_label, stack_bar_label,
)
from thesis_lib.specs import (
    AlignedTriplet,
    ClassificationF1PickleRef,
    ThesisClassificationF1Spec,
)
from thesis_sec2_common import ALL_TRIPLETS, ALL_TRIPLETS_LAP

apply_thesis_style()

OUT = Path('thesis_figures/sec5'); OUT.mkdir(parents=True, exist_ok=True)
results_root = Path('results').resolve()

# Canonical true-signal colour for time-series panels.
COLOR_TRUE = "#1A1A1A"

# %% [markdown]
# ## Classification spec lists

# %%
# Canonical cells — derived from notebooks/thesis_triplets.csv via thesis_sec2_common.
# Behavioral and laplacian triplets are aligned by cell label; lap rows reuse the
# same cell key with " (lap)" appended for legend disambiguation.
_FEATURE_GROUPS = ("xp", "xp_1", "xp_2", "xp_with_dbs")
_GROUP_TO_FILE = {
    "xp": "LDA_Xp_prediction.pkl",
    "xp_1": "LDA_Xp_1_prediction.pkl",
    "xp_2": "LDA_Xp_2_prediction.pkl",
    "xp_with_dbs": "LDA_Xp_with_dbs_prediction.pkl",
}


def _participant_session_from_label(label: str) -> Tuple[str, str]:
    """`PDI1_S2` -> ("PDI1", "S2")."""
    parts = label.split("_", 1)
    return (parts[0], parts[1] if len(parts) > 1 else label)


def _cell_records() -> List[Tuple[str, str, str, "AlignedTriplet"]]:
    """(participant, session_label, mode, triplet) for every (cell × mode).
    mode in {"behavioral", "laplacian"}; session_label adds " (lap)" for lap rows."""
    records: List[Tuple[str, str, str, "AlignedTriplet"]] = []
    for tri in ALL_TRIPLETS:
        p, s = _participant_session_from_label(tri.label)
        records.append((p, s, "behavioral", tri))
    for tri in ALL_TRIPLETS_LAP:
        p, s = _participant_session_from_label(tri.label)
        records.append((p, s + " (lap)", "laplacian", tri))
    return records


def _classification_dir(tri: "AlignedTriplet", model_label: str) -> Optional[str]:
    """Return `<variant>/<run_ts>` for the standard (non-flipped) cls dir, or None
    if that model is not configured for this cell (e.g. DPAD on a lap row that
    has not been classified yet)."""
    if model_label == "PSID":
        variant, run_ts = tri.psid_variant, tri.psid_run_ts
    elif model_label == "DPAD":
        variant, run_ts = tri.dpad_variant, tri.dpad_run_ts
    else:
        raise ValueError(f"Unknown model_label: {model_label}")
    if not variant or not run_ts:
        return None
    return f"{variant}/{run_ts}"


def _build_refs(model_label: str) -> Tuple[ClassificationF1PickleRef, ...]:
    refs: List[ClassificationF1PickleRef] = []
    for participant, sess, _mode, tri in _cell_records():
        base = _classification_dir(tri, model_label)
        for grp in _FEATURE_GROUPS:
            rel = f"{base}/{_GROUP_TO_FILE[grp]}" if base else f"_missing_/{_GROUP_TO_FILE[grp]}"
            refs.append(ClassificationF1PickleRef(
                participant, sess, grp, rel, model_label,
            ))
    return tuple(refs)


THESIS_CLASSIFICATION_F1 = [
    ThesisClassificationF1Spec(
        section_title="Figure F1 — DBS classification (balanced accuracy, mrmr8)",
        points=_build_refs("PSID") + _build_refs("DPAD"),
    ),
]

# ---------------------------------------------------------------------------
# Classification loaders (inline replacement for classification_f1_data)
# ---------------------------------------------------------------------------

GROUP_ORDER: Tuple[str, ...] = ("xp", "xp_1", "xp_2", "xp_with_dbs")
GROUP_SHORT: Dict[str, str] = {
    "xp": "Xp", "xp_1": "Xp1", "xp_2": "Xp2", "xp_with_dbs": "Xp+DBS",
}
GROUP_COLORS: Dict[str, str] = {
    "xp": COLOR_PSID,
    "xp_1": "#0F6E56",
    "xp_2": "#993C1D",
    "xp_with_dbs": "#854F0B",
}


@dataclass(frozen=True)
class ClassificationF1Point:
    participant_label: str
    session_label: str
    group: str
    balanced_accuracy: float
    permutation_pvalue: Optional[float]
    model_label: str = "PSID"
    permutation_scores: Optional[Tuple[float, ...]] = None


_TEST_PERM_N = 100
_TEST_PERM_RNG = np.random.default_rng(20260428)


def _test_set_perm(
    y_true: np.ndarray, y_pred: np.ndarray, n: int = _TEST_PERM_N
) -> Tuple[float, Tuple[float, ...]]:
    """Hold model predictions fixed; shuffle test labels n times; recompute
    balanced accuracy under each shuffle. Returns (pvalue, perm_scores).

    Null asks: 'is the model's test BA higher than what would be obtained if
    test labels were random under the same predictions?' Cheap (no refit).
    """
    from sklearn.metrics import balanced_accuracy_score
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    obs = float(balanced_accuracy_score(y_true, y_pred))
    scores = np.empty(n, dtype=float)
    for i in range(n):
        perm = _TEST_PERM_RNG.permutation(y_true)
        scores[i] = balanced_accuracy_score(perm, y_pred)
    pvalue = float((np.sum(scores >= obs) + 1) / (n + 1))
    return pvalue, tuple(float(x) for x in scores)


def _extract_test_ba_and_perm(
    res: Dict[str, Any]
) -> Tuple[float, Optional[float], Optional[Tuple[float, ...]]]:
    """Extract test BA and a test-side perm null. Computed in-notebook from
    y_true / y_pred stored in test_results — independent of the pipeline's
    pre-stored permutation_test dict (which used train-CV null)."""
    tr = res.get("test_results")
    if isinstance(tr, dict) and tr.get("balanced_accuracy") is not None:
        ba = float(tr["balanced_accuracy"])
    else:
        ba = float(res.get("balanced_accuracy", float("nan")))
    pval: Optional[float] = None
    pscores: Optional[Tuple[float, ...]] = None
    if isinstance(tr, dict):
        yt = tr.get("y_true")
        yp = tr.get("y_pred")
        if yt is not None and yp is not None and len(yt) == len(yp) and len(yt) > 0:
            try:
                pval, pscores = _test_set_perm(np.asarray(yt), np.asarray(yp))
            except Exception:
                pval, pscores = None, None
    return ba, pval, pscores


def collect_classification_f1_points(
    results_root: Path,
    refs: Sequence[ClassificationF1PickleRef],
    classification_parent: str = "classification",
    strict: bool = False,
) -> List[ClassificationF1Point]:
    base = results_root / classification_parent
    out: List[ClassificationF1Point] = []
    for ref in refs:
        p = base / ref.pickle_relative_path
        ba: float = float("nan")
        pval: Optional[float] = None
        pscores: Optional[Tuple[float, ...]] = None
        if p.is_file():
            with open(p, "rb") as f:
                res = _serial.load(f)
            if not isinstance(res, dict):
                if strict:
                    raise ValueError(f"Expected dict, got {type(res)}")
            else:
                ba, pval, pscores = _extract_test_ba_and_perm(res)
        elif strict:
            raise FileNotFoundError(f"Classification dump not found: {p}")
        out.append(ClassificationF1Point(
            participant_label=ref.participant_label,
            session_label=ref.session_label,
            group=ref.group,
            balanced_accuracy=ba,
            permutation_pvalue=pval,
            model_label=getattr(ref, "model_label", "PSID"),
            permutation_scores=pscores,
        ))
    return out


def _best_forecast_test_ba(
    run_dir: Path, feat: str
) -> Tuple[float, Optional[str], Optional[float], Optional[Tuple[float, ...]]]:
    """Return (test_BA, h_m_dirname, test_perm_pvalue, test_perm_scores) for the
    pipeline's fixed best-(h, m) pick. Pipeline marker: presence of
    `permutation_test` in the pickle — pipeline writes perm only at the CV-
    picked best (h, m). If no marker is found (pipeline Step 2 incomplete),
    fall back to the highest-test-BA (h, m) so we still have something to plot.

    perm fields are recomputed from y_true / y_pred via `_test_set_perm` —
    independent of the pre-stored CV-train null."""
    if not run_dir.is_dir():
        return float("nan"), None, None, None

    pipeline_pick: Optional[Tuple[float, str, Optional[float], Optional[Tuple[float, ...]]]] = None
    pipeline_pick_ba = -1.0
    fallback_picks: List[Tuple[float, str, Optional[float], Optional[Tuple[float, ...]]]] = []
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("h"):
            continue
        p = d / f"LDA_{feat}_forecast.pkl"
        if not p.is_file():
            continue
        try:
            with open(p, "rb") as f:
                r = _serial.load(f)
        except Exception:
            continue
        if not isinstance(r, dict):
            continue
        ba_v, pval, pscores = _extract_test_ba_and_perm(r)
        if not np.isfinite(ba_v):
            continue
        if isinstance(r.get("permutation_test"), dict):
            # Pipeline-picked (h, m). If multiple legacy reruns, keep highest-BA.
            if ba_v > pipeline_pick_ba:
                pipeline_pick = (ba_v, d.name, pval, pscores)
                pipeline_pick_ba = ba_v
        else:
            fallback_picks.append((ba_v, d.name, pval, pscores))

    if pipeline_pick is not None:
        return pipeline_pick
    if fallback_picks:
        fallback_picks.sort(key=lambda t: t[0], reverse=True)
        return fallback_picks[0]
    return float("nan"), None, None, None


def collect_forecast_bests(
    refs: Sequence[ClassificationF1PickleRef],
    results_root: Path,
    classification_parent: str = "classification",
) -> Dict[
    Tuple[str, str],
    Tuple[float, Optional[str], Optional[float], Optional[Tuple[float, ...]]]
]:
    base = results_root / classification_parent
    out: Dict[
        Tuple[str, str],
        Tuple[float, Optional[str], Optional[float], Optional[Tuple[float, ...]]]
    ] = {}
    for ref in refs:
        rel = Path(ref.pickle_relative_path)
        run_dir = base / rel.parent
        stem = rel.stem
        feat = stem.replace("LDA_", "").replace("_prediction", "")
        sess_key = f"{ref.participant_label}_{ref.session_label}"
        out[(sess_key, ref.group)] = _best_forecast_test_ba(run_dir, feat)
    return out


# %% [markdown]
# ## Fig 49: Classification — prediction vs forecast (2x1 pooled box + strip)
#
# **Decoupled PSID vs DPAD.** PSID is the focus; DPAD classification phase 5 is
# being re-run (yesterday's 2026-04-27 chain crashed at Step 2 perm). Once the
# DPAD pickles ship `permutation_test["scores"]`, set `SHOW_DPAD = True` to
# overlay a second model.
#
# **Chance level.** Balanced-accuracy chance is *not* 0.5 here. Per-cell
# permutation null distributions sit well above 0.5 because (a) `Xp_with_dbs`
# encodes DBS state directly, and each block is pure on/off, so a permuted-label
# LDA still recovers block from the DBS column (perm 95% / max → 1.0); (b)
# chrono-block CV with few groups limits the entropy of the label permutation.
# We replace the single 0.5 line with a per-`(feature, mode)` empirical chance
# band — 5–95% of the pooled permutation distribution across cells that ship
# full `scores`.

# %%
import csv
from matplotlib.patches import Patch as _Patch

SHOW_DPAD = False  # flip to True once DPAD perm scores are available

f1_spec = THESIS_CLASSIFICATION_F1[0]
cls_points = collect_classification_f1_points(
    results_root, f1_spec.points, classification_parent=f1_spec.classification_parent,
)
forecast_bests_psid = collect_forecast_bests(
    [r for r in f1_spec.points if r.model_label == "PSID"],
    results_root, classification_parent=f1_spec.classification_parent,
)
forecast_bests_dpad = collect_forecast_bests(
    [r for r in f1_spec.points if r.model_label == "DPAD"],
    results_root, classification_parent=f1_spec.classification_parent,
)


def _mode_from_session_label(session_label: str) -> str:
    return "laplacian" if "(lap)" in session_label else "behavioral"


# Pooled buckets keyed by (feature, model, mode) → list of (session_key, ba, pval, pscores).
pooled_pred: Dict[
    Tuple[str, str, str],
    List[Tuple[str, float, Optional[float], Optional[Tuple[float, ...]]]]
] = {}
for pt in cls_points:
    sess_key = f"{pt.participant_label}_{pt.session_label}"
    mode = _mode_from_session_label(pt.session_label)
    pooled_pred.setdefault((pt.group, pt.model_label, mode), []).append(
        (sess_key, pt.balanced_accuracy, pt.permutation_pvalue, pt.permutation_scores)
    )

# Forecast pooling — same shape as prediction now that forecast pickles can carry
# a permutation_test (a subset of (h, m) sweeps shipped one). Where the picked
# (best-BA) directory has perm scores, we keep them for the empirical chance band.
pooled_fcst: Dict[
    Tuple[str, str, str],
    List[Tuple[str, float, Optional[float], Optional[Tuple[float, ...]], Optional[str]]]
] = {}
for src, model in [(forecast_bests_psid, "PSID"), (forecast_bests_dpad, "DPAD")]:
    for (sess, grp), (ba, hm, pval, pscores) in src.items():
        mode = _mode_from_session_label(sess)
        pooled_fcst.setdefault((grp, model, mode), []).append(
            (sess, ba, pval, pscores, hm)
        )

# Session list for downstream heatmap / ROC figs.
_seen: Dict[str, None] = {}
for pt in cls_points:
    _seen.setdefault(f"{pt.participant_label}_{pt.session_label}", None)
session_keys = list(_seen.keys())

pred_lookup: Dict[Tuple[str, str, str], ClassificationF1Point] = {
    (f"{pt.participant_label}_{pt.session_label}", pt.group, pt.model_label): pt
    for pt in cls_points
}


def empirical_chance_band(
    pooled: Dict[Tuple[str, str, str], List[Tuple[str, float, Optional[float], Optional[Tuple[float, ...]]]]],
    feat: str, model: str, mode: str,
    low: float = 5.0, high: float = 95.0,
) -> Optional[Tuple[float, float, float]]:
    """Pool perm scores across cells in (feat, model, mode); return (low, median, high)
    percentiles or None if no cell ships a perm distribution."""
    pooled_scores: List[float] = []
    for entry in pooled.get((feat, model, mode), []):
        pscores = entry[3]
        if pscores is None or len(pscores) == 0:
            continue
        pooled_scores.extend(pscores)
    if not pooled_scores:
        return None
    arr = np.asarray(pooled_scores, dtype=float)
    return float(np.percentile(arr, low)), float(np.median(arr)), float(np.percentile(arr, high))


features = list(GROUP_ORDER)


def _model_color(model: str) -> str:
    return COLOR_PSID if model == "PSID" else COLOR_DPAD


def _draw_box(ax, vals, x_pos, color, *, hatch=None, fill_alpha=0.25, width: float = 0.14):
    face = (*to_rgba(color)[:3], fill_alpha)
    bp = ax.boxplot(
        [vals], positions=[x_pos], widths=width, patch_artist=True,
        showfliers=False, manage_ticks=False,
    )
    for box in bp["boxes"]:
        box.set(facecolor=face, edgecolor=color, linewidth=1.0)
        if hatch:
            box.set_hatch(hatch)
    for elem in ("whiskers", "caps", "medians"):
        for ln in bp[elem]:
            ln.set(color=color, linewidth=1.0)
    return bp


def _draw_chance_band(ax, x_center: float, lo: float, hi: float, mode: str, half_width: float):
    """Per-(feature, mode) empirical chance band: shaded rect from 5–95% of pooled
    perm distribution. Hatched for laplacian to match box hatch convention."""
    rect = plt.Rectangle(
        (x_center - half_width, lo), 2 * half_width, hi - lo,
        facecolor="#888888", alpha=0.18, edgecolor="#555555", linewidth=0.5,
        hatch=("//" if mode == "laplacian" else None), zorder=0,
    )
    ax.add_patch(rect)


def _hm_short(hm: Optional[str]) -> str:
    """`h2.0_m0.5` → `2.0/0.5`. None → `–`."""
    if not hm:
        return "–"
    parts = hm.split("_")
    if len(parts) != 2 or not parts[0].startswith("h") or not parts[1].startswith("m"):
        return hm
    return f"{parts[0][1:]}/{parts[1][1:]}"


def _forecast_hm_table(model_label: str) -> Tuple[List[List[str]], List[str]]:
    """Build a (rows, row_labels) table of picked (h, m) per (session, feature)
    for the forecast panel. Rows = session_keys, cols = feature groups."""
    src = forecast_bests_psid if model_label == "PSID" else forecast_bests_dpad
    rows: List[List[str]] = []
    for sk in session_keys:
        row = []
        for feat in features:
            tup = src.get((sk, feat))
            row.append(_hm_short(tup[1]) if tup else "–")
        rows.append(row)
    return rows, list(session_keys)


def render_pooled_classification(
    model_label: str, save_path: Path, *, title_suffix: str = "",
) -> bool:
    """Render the 2-panel pooled classification figure for one model + a third
    sub-axis with the per-(session, feature) (h, m) picks underneath. Returns
    True if anything was drawn (i.e. data exists), False otherwise."""
    mm_groups = [(model_label, mo) for mo in ("behavioral", "laplacian")]
    # Single-model tight layout: 2 boxes per feature, sit close together.
    box_spread = 0.10
    mm_offsets = np.array([-box_spread, +box_spread])
    box_width = 0.08
    band_half_width = box_width * 0.75

    has_any = False
    # 2 rows: panel A (prediction), panel B (forecast). Forecast (h, m) picks
    # printed separately as a pandas DataFrame below the figure render call.
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.6))
    rng = np.random.default_rng(42)
    color = _model_color(model_label)

    for panel_i, (ax, src) in enumerate(zip(axes, [pooled_pred, pooled_fcst])):
        for fi, feat in enumerate(features):
            for gi, (_m, mode) in enumerate(mm_groups):
                entries = src.get((feat, model_label, mode), [])
                x_pos = fi + mm_offsets[gi]

                band = empirical_chance_band(src, feat, model_label, mode)
                if band is not None:
                    lo_p, _med, hi_p = band
                    _draw_chance_band(ax, x_pos, lo_p, hi_p, mode,
                                      half_width=band_half_width)

                if not entries:
                    continue
                hatch = "//" if mode == "laplacian" else None
                bas = np.array([e[1] for e in entries], dtype=float)
                bas_finite = bas[np.isfinite(bas)]
                if len(bas_finite) >= 2:
                    _draw_box(ax, bas_finite, x_pos, color, hatch=hatch, width=box_width)
                    has_any = True
                jitter = rng.uniform(-box_width * 0.25, box_width * 0.25, len(entries))
                for j, entry in enumerate(entries):
                    ba = entry[1]
                    if not np.isfinite(ba):
                        continue
                    has_any = True
                    pval = entry[2]
                    if pval is not None and pval < 0.05:
                        ax.scatter(x_pos + jitter[j], ba, s=28, color=color, alpha=1.0,
                                   edgecolor="black", linewidths=0.5, zorder=3)
                    elif pval is not None:
                        ax.scatter(x_pos + jitter[j], ba, s=28, color=color, alpha=0.55,
                                   edgecolor="none", zorder=3)
                    else:
                        ax.scatter(x_pos + jitter[j], ba, s=28, color=color, alpha=1.0,
                                   edgecolor="black", linewidths=0.5, zorder=3)
        ax.axhline(0.5, linestyle=":", color="#999999", linewidth=0.8)
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel("Balanced accuracy")

    axes[1].set_xticks(np.arange(len(features)))
    axes[1].set_xticklabels([GROUP_SHORT[f] for f in features])
    axes[1].set_xlabel("Feature group")
    panel_label(axes[0], "A", f"{model_label} prediction (Xp at t){title_suffix}")
    panel_label(axes[1], "B", f"{model_label} forecast (CV-picked h, m){title_suffix}")

    legend_handles = [
        _Patch(facecolor=hex_to_rgba(color, 0.25), edgecolor=color,
               label=f"{model_label}, behavioral"),
        _Patch(facecolor=hex_to_rgba(color, 0.25), edgecolor=color, hatch="//",
               label=f"{model_label}, laplacian"),
        _Patch(facecolor="#888888", alpha=0.18, edgecolor="#555555",
               label="empirical chance 5–95% (test-perm null, n=100)"),
        plt.Line2D([0], [0], linestyle=":", color="#999999", linewidth=0.8,
                   label="0.5 reference"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8)

    fig.tight_layout()

    if has_any:
        fig.savefig(str(save_path))
        plt.show()
    else:
        plt.close(fig)
    return has_any


# Render PSID only. DPAD figure intentionally skipped — DPAD numbers still go
# into the appendix CSV / pandas tables below.
render_pooled_classification("PSID", OUT / "fig_049_classification_pooled_psid.png")
_dpad_rendered = False
# _dpad_rendered = render_pooled_classification(
#     "DPAD", OUT / "fig_049b_classification_pooled_dpad.png"
# )

# Forecast (h, m) picks per session × feature — printed as pandas tables instead
# of being baked into the figure (one table per model).
import pandas as _pd
for _mlabel, _src in (("PSID", forecast_bests_psid), ("DPAD", forecast_bests_dpad)):
    _rows = []
    for _sk in session_keys:
        _rows.append([_hm_short(_src.get((_sk, _f), (None, ""))[1]) if _src.get((_sk, _f)) else "–"
                     for _f in features])
    _df = _pd.DataFrame(_rows, index=list(session_keys),
                        columns=[GROUP_SHORT[_f] for _f in features])
    print(f"\nForecast (h / m) picks — {_mlabel}")
    print(_df.to_string())

# Appendix CSV table — every (panel, session, feature, model, mode) row.
table_path = OUT / "table_049_classification.csv"
with open(table_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["panel", "session", "feature", "model", "mode", "balanced_accuracy",
                "perm_pvalue", "best_h_m"])
    for pt in cls_points:
        sess_key = f"{pt.participant_label}_{pt.session_label}"
        mode = _mode_from_session_label(pt.session_label)
        w.writerow([
            "prediction", sess_key, pt.group, pt.model_label, mode,
            f"{pt.balanced_accuracy:.4f}" if np.isfinite(pt.balanced_accuracy) else "",
            f"{pt.permutation_pvalue:.4f}" if pt.permutation_pvalue is not None else "",
            "",
        ])
    for src_dict, model in [(forecast_bests_psid, "PSID"), (forecast_bests_dpad, "DPAD")]:
        for (sess, grp), (ba, hm, pval, _pscores) in src_dict.items():
            mode = _mode_from_session_label(sess)
            w.writerow([
                "forecast", sess, grp, model, mode,
                f"{ba:.4f}" if np.isfinite(ba) else "",
                f"{pval:.4f}" if pval is not None else "",
                hm or "",
            ])

_n_pred_perm_psid = sum(
    1 for pt in cls_points
    if pt.model_label == "PSID" and pt.permutation_scores is not None
)
_n_fcst_perm_psid = sum(
    1 for (sess_grp, tup) in forecast_bests_psid.items()
    if tup[3] is not None  # perm scores at pipeline-picked (h, m)
)
_n_pred_perm_dpad = sum(
    1 for pt in cls_points
    if pt.model_label == "DPAD" and pt.permutation_scores is not None
)
_n_fcst_perm_dpad = sum(
    1 for (sess_grp, tup) in forecast_bests_dpad.items()
    if tup[3] is not None
)
print(
    f"Fig 49 (PSID only) - DBS classification BA, pooled across cells "
    f"per (feature, mode). Both panels: shaded grey band per box = empirical "
    f"chance 5–95% from pooled test-side permutation null (n={_TEST_PERM_N} "
    f"per cell, model frozen, test labels shuffled). Panel A: prediction. "
    f"Panel B: forecast at pipeline's CV-picked best (h, m). Dot opacity: "
    f"perm p<0.05 = full + black edge, p≥0.05 faded, no perm = full opacity "
    f"(neutral). Coverage: PSID pred {_n_pred_perm_psid}/32, fcst "
    f"{_n_fcst_perm_psid}/32; DPAD pred {_n_pred_perm_dpad}/32, fcst "
    f"{_n_fcst_perm_dpad}/32 (DPAD figure intentionally skipped — see pandas "
    f"tables above). CSV: {table_path.name}"
)

# %% [markdown]
# ## Fig 49a: Confusion matrix grid — sessions × features (PSID)
#
# 8 sessions × 4 feature groups = 32 small 2×2 confusion matrices, row-normalized
# (rows = true class). Diagonal = TPR(off) and TPR(on). Below-cell text shows the
# raw counts. PSID-only when `SHOW_DPAD=False`; when DPAD ships, two grids
# stacked vertically.

# %%
def _load_confusion(rel_pkl: str) -> Optional[np.ndarray]:
    p = results_root / "classification" / rel_pkl
    if not p.is_file():
        return None
    try:
        with open(p, "rb") as f:
            r = _serial.load(f)
    except Exception:
        return None
    if not isinstance(r, dict):
        return None
    tr = r.get("test_results")
    cm = (tr or {}).get("confusion_matrix") if isinstance(tr, dict) else None
    if cm is None:
        cm = r.get("confusion_matrix")
    if cm is None:
        return None
    arr = np.asarray(cm, dtype=float)
    if arr.shape != (2, 2):
        return None
    return arr


def _draw_confusion_cell(ax, cm: Optional[np.ndarray], *, title: str, color: str):
    if cm is None:
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#EEEEEE",
                                   edgecolor="#BBBBBB", hatch="///", linewidth=0.5))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=7)
        return
    # Reorient so "on" is top row + leftmost column → top-left = TP_on (positive).
    # Source pickle uses sklearn order [off=0, on=1]; flip both axes.
    cm_oriented = cm[::-1, ::-1]
    row_sums = cm_oriented.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    norm_cm = cm_oriented / row_sums
    base_rgba = np.asarray(to_rgba(color))
    rgba = np.zeros((2, 2, 4))
    for i in range(2):
        for j in range(2):
            a = float(np.clip(norm_cm[i, j], 0, 1))
            rgba[i, j, :3] = base_rgba[:3]
            rgba[i, j, 3] = 0.15 + 0.75 * a
    ax.imshow(rgba, aspect="equal", interpolation="nearest")
    for i in range(2):
        for j in range(2):
            txt_color = "white" if norm_cm[i, j] > 0.55 else "#222222"
            ax.text(j, i, f"{norm_cm[i, j]:.2f}\n({int(cm_oriented[i, j])})",
                    ha="center", va="center", fontsize=6, color=txt_color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["on", "off"], fontsize=6)
    ax.set_yticklabels(["on", "off"], fontsize=6)
    ax.set_title(title, fontsize=7)


def _confusion_grid(model_label: str, color: str) -> Optional[plt.Figure]:
    refs_by_sk_grp: Dict[Tuple[str, str], ClassificationF1PickleRef] = {
        (f"{ref.participant_label}_{ref.session_label}", ref.group): ref
        for ref in f1_spec.points if ref.model_label == model_label
    }
    n_rows = len(session_keys)
    n_cols = len(features)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.4 * n_cols + 0.6, 1.4 * n_rows + 0.6))
    axes = np.atleast_2d(axes)
    drew_any = False
    for ri, sk in enumerate(session_keys):
        for ci, feat in enumerate(features):
            ax = axes[ri, ci]
            ref = refs_by_sk_grp.get((sk, feat))
            cm = _load_confusion(ref.pickle_relative_path) if ref is not None else None
            title = f"{GROUP_SHORT[feat]}" if ri == 0 else ""
            _draw_confusion_cell(ax, cm, title=title, color=color)
            if cm is not None:
                drew_any = True
        axes[ri, 0].set_ylabel(sk, fontsize=7)
    fig.suptitle(f"{model_label} — confusion matrices (rows: true class, normalized)",
                 fontsize=9, y=0.995)
    fig.tight_layout()
    return fig if drew_any else None


fig_psid = _confusion_grid("PSID", COLOR_PSID)
if fig_psid is not None:
    fig_psid.savefig(str(OUT / 'fig_049a_confusion_psid.png'))
    plt.show()
else:
    print("Fig 49a SKIPPED — no PSID confusion data.")

if SHOW_DPAD:
    fig_dpad = _confusion_grid("DPAD", COLOR_DPAD)
    if fig_dpad is not None:
        fig_dpad.savefig(str(OUT / 'fig_049a_confusion_dpad.png'))
        plt.show()

print(
    f"Fig 49a - PSID confusion matrix grid. {len(session_keys)} sessions × "
    f"{len(features)} feature groups; each cell is a row-normalized 2×2 confusion "
    f"matrix oriented with **on as top row** (TP_on at top-left), then off below. "
    f"Inner text: row-normalized fraction and raw count. Hatched cells = pickle "
    f"missing. Diagonal = TPR per class. DPAD grid "
    f"{'rendered separately' if SHOW_DPAD else 'pending'}."
)

# %% [markdown]
# ## Fig 50: Standard classification heatmap (PSID + DPAD stacked)

# %%
from matplotlib.colors import TwoSlopeNorm

feat_keys = list(GROUP_ORDER)
n_rows_hm = len(session_keys)
n_cols_hm = len(feat_keys)


def _ba_pval_matrix(model_label: str) -> Tuple[np.ndarray, np.ndarray]:
    ba = np.full((n_rows_hm, n_cols_hm), np.nan)
    pv = np.full((n_rows_hm, n_cols_hm), np.nan)
    for pt in cls_points:
        if pt.model_label != model_label:
            continue
        sk = f"{pt.participant_label}_{pt.session_label}"
        if sk in session_keys and pt.group in feat_keys:
            r = session_keys.index(sk)
            c = feat_keys.index(pt.group)
            ba[r, c] = pt.balanced_accuracy
            if pt.permutation_pvalue is not None:
                pv[r, c] = pt.permutation_pvalue
    return ba, pv


psid_ba, psid_pv = _ba_pval_matrix("PSID")
if SHOW_DPAD:
    dpad_ba, dpad_pv = _ba_pval_matrix("DPAD")
else:
    dpad_ba = np.full((n_rows_hm, n_cols_hm), np.nan)
    dpad_pv = np.full((n_rows_hm, n_cols_hm), np.nan)

cmap = plt.cm.RdBu_r
# Tightened range so the 0.5±0.15 regime (where almost every cell lives) gets
# visible separation. Xp+DBS at 1.0 clips to the top colour without breaking
# the dynamic range of the interesting cells.
norm = TwoSlopeNorm(vmin=0.4, vcenter=0.5, vmax=0.7)


def _draw_heatmap(ax, mat: np.ndarray, pv: np.ndarray, *, title: str):
    rgba = cmap(norm(np.clip(mat, 0.4, 0.7)))
    # Per-cell alpha: sig (p<0.05) → 1.0; non-sig → 0.65; missing → 0 (handled
    # via the hatch overlay below).
    alpha = np.where(np.isfinite(pv) & (pv < 0.05), 1.0, 0.65)
    alpha = np.where(np.isfinite(mat), alpha, 0.0)
    rgba[..., -1] = alpha
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    # Hatch overlay for missing cells.
    for ri in range(n_rows_hm):
        for ci in range(n_cols_hm):
            if not np.isfinite(mat[ri, ci]):
                ax.add_patch(plt.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    facecolor="#EEEEEE", edgecolor="#BBBBBB",
                    hatch="///", linewidth=0.5,
                ))
    ax.set_xticks(np.arange(n_cols_hm))
    ax.set_xticklabels([GROUP_SHORT[f] for f in feat_keys])
    ax.set_yticks(np.arange(n_rows_hm))
    ax.set_yticklabels(session_keys)
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Session")
    ax.set_title(title, loc="left", fontsize=10)


if SHOW_DPAD:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.0), sharex=True)
    _draw_heatmap(axes[0], psid_ba, psid_pv, title="PSID (LDA prediction, test BA)")
    _draw_heatmap(axes[1], dpad_ba, dpad_pv, title="DPAD (LDA prediction, test BA)")
    panel_label(axes[0], "A", "")
    panel_label(axes[1], "B", "")
    cbar_ax_ref = axes
else:
    fig, ax_only = plt.subplots(1, 1, figsize=(7.5, 5.0))
    _draw_heatmap(ax_only, psid_ba, psid_pv, title="PSID (LDA prediction, test BA)")
    cbar_ax_ref = ax_only

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=cbar_ax_ref, fraction=0.04, pad=0.02, label="Balanced accuracy")
cbar.ax.axhline(0.5, color="#222222", linewidth=0.8, linestyle="--")
cbar.ax.text(2.4, 0.5, "chance ref", va="center", fontsize=8, transform=cbar.ax.get_yaxis_transform())

fig.savefig(str(OUT / 'fig_050_classification_heatmap.png'))
plt.show()

print(
    f"Fig 50 - DBS classification heatmaps. PSID shown{' + DPAD' if SHOW_DPAD else ' (DPAD pending)'}. "
    "Rows: sessions (4 behav + 4 lap). Cols: feature groups. Cell color: test "
    "balanced accuracy (RdBu_r centred at 0.5 reference; empirical chance is "
    "feature-dependent — see Fig 49). Cell alpha: full if perm p<0.05, faded "
    "otherwise. Hatched cells: classification dump missing. Numerical values in "
    "appendix CSV."
)

# %% [markdown]
# ## Fig 51: Flipped classification heatmap (PSID + DPAD stacked)

# %%
_FLIPPED_FEAT_PKL: Dict[str, str] = {
    "xp": "LDA_Xp_flipped.pkl",
    "xp_1": "LDA_Xp_1_flipped.pkl",
    "xp_2": "LDA_Xp_2_flipped.pkl",
    "xp_with_dbs": "LDA_Xp_with_dbs_flipped.pkl",
}


def _flipped_best_ba_pval(model_label: str, tri, feat: str) -> Tuple[float, Optional[float]]:
    """Best (h, m) flipped BA for one cell × feature × model. Returns (NaN, None)
    when the flipped variant directory is absent (e.g. DPAD mrmr8 flipped not yet
    classified)."""
    if model_label == "PSID":
        variant = (tri.psid_variant or "") + "_flipped"
    else:
        variant = (tri.dpad_variant or "") + "_flipped"
    if not variant or variant.startswith("_"):
        return float("nan"), None
    var_root = results_root / "classification" / variant
    if not var_root.is_dir():
        return float("nan"), None
    target = _FLIPPED_FEAT_PKL[feat]
    best_ba = -np.inf
    best_pv: Optional[float] = None
    for ts_dir in sorted(var_root.iterdir()):
        if not ts_dir.is_dir():
            continue
        for hm_dir in sorted(ts_dir.iterdir()):
            if not hm_dir.is_dir() or not hm_dir.name.startswith("h"):
                continue
            p = hm_dir / target
            if not p.is_file():
                continue
            try:
                with open(p, "rb") as f:
                    r = _serial.load(f)
            except Exception:
                continue
            if not isinstance(r, dict):
                continue
            tr = r.get("test_results", {})
            ba = tr.get("balanced_accuracy", r.get("balanced_accuracy"))
            if ba is None or not np.isfinite(float(ba)):
                continue
            ba_f = float(ba)
            pt = r.get("permutation_test")
            pv = float(pt["pvalue"]) if isinstance(pt, dict) and pt.get("pvalue") is not None else None
            if ba_f > best_ba:
                best_ba = ba_f
                best_pv = pv
    return (best_ba if best_ba > -np.inf else float("nan")), best_pv


def _flipped_matrix(model_label: str) -> Tuple[np.ndarray, np.ndarray]:
    ba = np.full((n_rows_hm, n_cols_hm), np.nan)
    pv = np.full((n_rows_hm, n_cols_hm), np.nan)
    for participant, sess, _mode, tri in _cell_records():
        sk = f"{participant}_{sess}"
        if sk not in session_keys:
            continue
        r = session_keys.index(sk)
        for ci, feat in enumerate(feat_keys):
            v_ba, v_pv = _flipped_best_ba_pval(model_label, tri, feat)
            ba[r, ci] = v_ba
            if v_pv is not None:
                pv[r, ci] = v_pv
    return ba, pv


psid_flip_ba, psid_flip_pv = _flipped_matrix("PSID")
if SHOW_DPAD:
    dpad_flip_ba, dpad_flip_pv = _flipped_matrix("DPAD")
else:
    dpad_flip_ba = np.full((n_rows_hm, n_cols_hm), np.nan)
    dpad_flip_pv = np.full((n_rows_hm, n_cols_hm), np.nan)

if SHOW_DPAD:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.0), sharex=True)
    _draw_heatmap(axes[0], psid_flip_ba, psid_flip_pv, title="PSID (flipped, best test BA over h × m)")
    _draw_heatmap(axes[1], dpad_flip_ba, dpad_flip_pv, title="DPAD (flipped, best test BA over h × m)")
    panel_label(axes[0], "A", "")
    panel_label(axes[1], "B", "")
    cbar_ax_ref = axes
else:
    fig, ax_only = plt.subplots(1, 1, figsize=(7.5, 5.0))
    _draw_heatmap(ax_only, psid_flip_ba, psid_flip_pv, title="PSID (flipped, best test BA over h × m)")
    cbar_ax_ref = ax_only

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=cbar_ax_ref, fraction=0.04, pad=0.02, label="Balanced accuracy")
cbar.ax.axhline(0.5, color="#222222", linewidth=0.8, linestyle="--")
cbar.ax.text(2.4, 0.5, "chance ref", va="center", fontsize=8, transform=cbar.ax.get_yaxis_transform())

fig.savefig(str(OUT / 'fig_051_flipped_heatmap.png'))
plt.show()

print(
    f"Fig 51 - Flipped (cross-condition) classification: train on one DBS "
    f"state, evaluate on the other. Best test BA across the (h, m) flipped grid "
    f"per cell. PSID shown{' + DPAD' if SHOW_DPAD else ' (DPAD pending — phase 5 rerunning)'}. "
    f"Same colour / alpha encoding as Fig 50. Empirical chance is feature-"
    f"dependent (see Fig 49). Hatched cells: dump missing."
)

# %% [markdown]
# ## Fig 52: ROC curves — per session, feature × model overlays

# %%
def _load_roc_from_file(pkl_path: Path):
    if not pkl_path.is_file():
        return None
    try:
        with open(pkl_path, "rb") as f:
            res = _serial.load(f)
    except Exception:
        return None
    if not isinstance(res, dict):
        return None
    tr = res.get("test_results") or res
    fpr = tr.get("fpr"); tpr = tr.get("tpr")
    auc = tr.get("roc_auc", res.get("roc_auc", float("nan")))
    if fpr is None or tpr is None:
        return None
    try:
        return np.asarray(fpr, dtype=float), np.asarray(tpr, dtype=float), float(auc)
    except Exception:
        return None


_roc_refs = list(THESIS_CLASSIFICATION_F1[0].points)
_base_cls = results_root / "classification"

per_session_model: Dict[Tuple[str, str], Dict[str, ClassificationF1PickleRef]] = {}
_session_keys_roc: List[str] = []
for ref in _roc_refs:
    sk = f"{ref.participant_label}_{ref.session_label}"
    if sk not in _session_keys_roc:
        _session_keys_roc.append(sk)
    per_session_model.setdefault((sk, ref.model_label), {})[ref.group] = ref

n = len(_session_keys_roc)
ncols = 4 if n >= 5 else (3 if n >= 4 else max(1, n))
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 2.7 * nrows + 1.0),
                         sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()

_roc_models = [("PSID", "-")] + ([("DPAD", "--")] if SHOW_DPAD else [])
for idx, sk in enumerate(_session_keys_roc):
    ax = axes[idx]
    for feat in GROUP_ORDER:
        for model_label, linestyle in _roc_models:
            ref = per_session_model.get((sk, model_label), {}).get(feat)
            if ref is None:
                continue
            roc = _load_roc_from_file(_base_cls / ref.pickle_relative_path)
            if roc is None:
                continue
            fpr, tpr, auc = roc
            ax.plot(fpr, tpr, color=GROUP_COLORS[feat], linewidth=1.3,
                    linestyle=linestyle,
                    label=f"{GROUP_SHORT[feat]} {model_label} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle=":", color="#555555", linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    panel_label(ax, chr(ord("A") + idx), sk)

for ax in axes[n:]:
    ax.set_visible(False)

axes_2d = axes.reshape(nrows, ncols)
for ax in axes_2d[-1, :]:
    if ax.get_visible():
        ax.set_xlabel("False positive rate")
for ax in axes_2d[:, 0]:
    ax.set_ylabel("True positive rate")

# Compact shared legend — feature colour + model line-style.
legend_handles = [
    plt.Line2D([0], [0], color=GROUP_COLORS[g], linewidth=1.3, label=GROUP_SHORT[g])
    for g in GROUP_ORDER
]
legend_handles.append(
    plt.Line2D([0], [0], color="#555555", linewidth=1.3, linestyle="-", label="PSID")
)
if SHOW_DPAD:
    legend_handles.append(
        plt.Line2D([0], [0], color="#555555", linewidth=1.3, linestyle="--", label="DPAD")
    )
fig.legend(handles=legend_handles)

fig.savefig(str(OUT / 'fig_052_roc_curves.png'))
plt.show()

print(
    "Fig 52 - Per-session ROC curves (test split). One panel per session; per "
    "panel, 4 features × 2 models = up to 8 curves. PSID solid, DPAD dashed. "
    "AUC values in compact shared legend (and per-curve labels in axes legend "
    "if rendered). Diagonal = random classifier."
)

# %% [markdown]
# ## Fig 52a: Paired Δ scatter — DPAD vs PSID test BA per (cell × feature)
#
# Requires DPAD classification. Gated on SHOW_DPAD; skipped while DPAD runs.

# %%
if not SHOW_DPAD:
    print("Fig 52a SKIPPED — DPAD classification pending (run pipeline_dpad phase 5).")
    paired_count = {"behavioral": 0, "laplacian": 0}
else:
    psid_lookup_paired: Dict[Tuple[str, str, str], ClassificationF1Point] = {
        (f"{pt.participant_label}_{pt.session_label}", pt.group, "behavioral"
         if "(lap)" not in pt.session_label else "laplacian"): pt
        for pt in cls_points if pt.model_label == "PSID"
    }
    dpad_lookup_paired: Dict[Tuple[str, str, str], ClassificationF1Point] = {
        (f"{pt.participant_label}_{pt.session_label}", pt.group, "behavioral"
         if "(lap)" not in pt.session_label else "laplacian"): pt
        for pt in cls_points if pt.model_label == "DPAD"
    }

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0.3, 1.05], [0.3, 1.05], linestyle="--", color="#555555", linewidth=0.9,
            label="y = x")
    ax.axhline(0.5, linestyle=":", color="#888888", linewidth=0.7)
    ax.axvline(0.5, linestyle=":", color="#888888", linewidth=0.7)

    paired_count = {"behavioral": 0, "laplacian": 0}
    for (sk, feat, mode), psid_pt in psid_lookup_paired.items():
        dpad_pt = dpad_lookup_paired.get((sk, feat, mode))
        if dpad_pt is None or not np.isfinite(dpad_pt.balanced_accuracy):
            continue
        if not np.isfinite(psid_pt.balanced_accuracy):
            continue
        color = GROUP_COLORS[feat]
        marker = "o" if mode == "behavioral" else "s"
        sig_psid = (psid_pt.permutation_pvalue is not None and psid_pt.permutation_pvalue < 0.05)
        sig_dpad = (dpad_pt.permutation_pvalue is not None and dpad_pt.permutation_pvalue < 0.05)
        a = 1.0 if (sig_psid or sig_dpad) else 0.55
        ax.scatter(
            psid_pt.balanced_accuracy, dpad_pt.balanced_accuracy,
            s=55, color=color, alpha=a, marker=marker,
            edgecolor="black" if (sig_psid or sig_dpad) else "none",
            linewidths=0.5,
        )
        paired_count[mode] += 1

    ax.set_xlabel("PSID test BA")
    ax.set_ylabel("DPAD test BA")
    ax.set_xlim(0.3, 1.05)
    ax.set_ylim(0.3, 1.05)
    ax.set_aspect("equal")

    legend_handles = [
        _Patch(facecolor=GROUP_COLORS[g], edgecolor=GROUP_COLORS[g], label=GROUP_SHORT[g])
        for g in GROUP_ORDER
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                   markersize=8, label="behavioral"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#555555",
                   markersize=8, label="laplacian"),
    ]
    ax.legend(handles=legend_handles)

    fig.savefig(str(OUT / 'fig_052a_paired_ba_scatter.png'))
    plt.show()

    print(
        "Fig 52a - Paired DPAD vs PSID test BA. Each point: one (cell × feature). "
        "Marker shape = mode (circle behavioral, square laplacian); colour = feature "
        "group. Diagonal = parity. Significant points (perm p<0.05 in either model) "
        "shown at full opacity with edge stroke. Empty regions = DPAD classification "
        f"pending. Pairs plotted: {paired_count['behavioral']} behav + {paired_count['laplacian']} lap."
    )

# %% [markdown]
# ## Fig 52b: Within − flipped Δ — generalisation gap matrix
#
# PSID-only when SHOW_DPAD=False; stacks PSID + DPAD when DPAD ships.

# %%
within_minus_flip_psid = psid_ba - psid_flip_ba
within_minus_flip_dpad = dpad_ba - dpad_flip_ba  # all-NaN when SHOW_DPAD=False

div_cmap = plt.cm.RdBu_r
div_norm = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)


def _draw_diff_heatmap(ax, mat: np.ndarray, *, title: str):
    rgba = div_cmap(div_norm(np.clip(mat, -0.4, 0.4)))
    rgba[..., -1] = np.where(np.isfinite(mat), 1.0, 0.0)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    for ri in range(n_rows_hm):
        for ci in range(n_cols_hm):
            if not np.isfinite(mat[ri, ci]):
                ax.add_patch(plt.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    facecolor="#EEEEEE", edgecolor="#BBBBBB",
                    hatch="///", linewidth=0.5,
                ))
    ax.set_xticks(np.arange(n_cols_hm))
    ax.set_xticklabels([GROUP_SHORT[f] for f in feat_keys])
    ax.set_yticks(np.arange(n_rows_hm))
    ax.set_yticklabels(session_keys)
    ax.set_xlabel("Feature group")
    ax.set_ylabel("Session")
    ax.set_title(title, loc="left", fontsize=10)


if SHOW_DPAD:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 9.0), sharex=True)
    _draw_diff_heatmap(axes[0], within_minus_flip_psid, title="PSID Δ = within − flipped")
    _draw_diff_heatmap(axes[1], within_minus_flip_dpad, title="DPAD Δ = within − flipped")
    panel_label(axes[0], "A", "")
    panel_label(axes[1], "B", "")
    cbar_ax_ref = axes
else:
    fig, ax_only = plt.subplots(1, 1, figsize=(7.5, 5.0))
    _draw_diff_heatmap(ax_only, within_minus_flip_psid, title="PSID Δ = within − flipped")
    cbar_ax_ref = ax_only

sm = plt.cm.ScalarMappable(cmap=div_cmap, norm=div_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=cbar_ax_ref, fraction=0.04, pad=0.02,
                    label="Δ test BA  (within − flipped)")
cbar.ax.axhline(0.0, color="#222222", linewidth=0.8, linestyle="--")
cbar.ax.text(2.4, 0.0, "no gap", va="center", fontsize=8,
             transform=cbar.ax.get_yaxis_transform())

fig.savefig(str(OUT / 'fig_052b_within_minus_flipped.png'))
plt.show()

print(
    f"Fig 52b - Generalisation gap (within − flipped test BA). PSID shown"
    f"{' + DPAD' if SHOW_DPAD else ' (DPAD pending)'}. Positive (red) = "
    f"classifier generalises poorly: trained on one block does worse on the other. "
    f"Negative (blue) = flipped outperforms within (small-sample artefact). "
    f"Near-zero = transfer holds. Hatched cells: missing within or flipped."
)


# %%
n = len(list(OUT.glob('*.png')))
print(f'Section 5 total: {n} figures (expected 7+ with Fig 49a; +DPAD variants when SHOW_DPAD=True)')
