"""ROC curve figures for standard and flipped DBS classification."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.classification_f1_data import GROUP_ORDER
from dashboard.thesis.classification_f1_figure import (
    _FEAT_COLORS,
    _FEAT_SHORT,
    _FLIPPED_FEAT_SUFFIX,
    _FLIPPED_SESSIONS,
    _H_VALUES,
    _M_VALUES,
)
from dashboard.thesis.constants import (
    COLOR_CHANCE,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    true_line_color,
)

logger = logging.getLogger(__name__)

# Standard classification sessions (same as specs.py THESIS_CLASSIFICATION_F1 entries)
_STANDARD_SESSIONS: dict[str, tuple[str, str]] = {
    "PDI1_S2": ("psid_behavioral_PDI1_2_nx_80_n12_i40_dbs_both_narrow_band", "20260315_200324"),
    "PDI1_S4": ("psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band", "20260315_142838"),
    "PDI4_S2": ("psid_behavioral_PDI4_2_nx_80_n10_i40_dbs_both_narrow_band", "20260315_144054"),
    "PDI4_S3": ("psid_behavioral_PDI4_3_nx65_n10_i40_dbs_both_narrow_band", "20260315_200805"),
}

_STANDARD_PKL_NAMES: dict[str, str] = {
    "xp": "LDA_Xp_prediction.pkl",
    "xp_1": "LDA_Xp_1_prediction.pkl",
    "xp_2": "LDA_Xp_2_prediction.pkl",
    "xp_with_dbs": "LDA_Xp_with_dbs_prediction.pkl",
}


def _load_roc(pkl_path: Path) -> Tuple[np.ndarray, np.ndarray, float] | None:
    if not pkl_path.is_file():
        return None
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    tr = res.get("test_results", res)
    fpr = tr.get("fpr")
    tpr = tr.get("tpr")
    auc = tr.get("roc_auc", float("nan"))
    if fpr is None or tpr is None:
        return None
    return np.asarray(fpr), np.asarray(tpr), float(auc)


def build_roc_curve_figure(
    results_root,
    theme: ThesisTheme = ThesisTheme.LIGHT,
) -> Figure:
    """Two-column figure: standard (left) and flipped (right) ROC curves.

    Each panel overlays ROC curves for all four feature conditions,
    averaged across sessions.
    """
    fg = true_line_color(theme)
    cls_root = Path(results_root) / "classification"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Standard classification", "Flipped classification"],
        horizontal_spacing=0.12,
    )

    # --- Standard ROC (left panel) ---
    for feat in GROUP_ORDER:
        all_fpr, all_tpr = [], []
        for sess_label, (variant, run_ts) in _STANDARD_SESSIONS.items():
            pkl_name = _STANDARD_PKL_NAMES.get(feat)
            if pkl_name is None:
                continue
            pkl_path = cls_root / variant / run_ts / pkl_name
            roc = _load_roc(pkl_path)
            if roc is not None:
                fpr, tpr, _ = roc
                all_fpr.append(fpr)
                all_tpr.append(tpr)

        if all_fpr:
            # Interpolate to common FPR grid and average
            common_fpr = np.linspace(0, 1, 200)
            interp_tpr = np.array([np.interp(common_fpr, f, t) for f, t in zip(all_fpr, all_tpr)])
            mean_tpr = interp_tpr.mean(axis=0)
            color = _FEAT_COLORS.get(feat, "#888888")
            fig.add_trace(
                go.Scatter(
                    x=common_fpr, y=mean_tpr, mode="lines",
                    name=_FEAT_SHORT.get(feat, feat),
                    line=dict(color=color, width=2.0),
                    legendgroup=feat, showlegend=True,
                ),
                row=1, col=1,
            )

    # --- Flipped ROC (right panel) ---
    # Use best h/m combination (h=0.5, m=0.5 — first available)
    for feat in GROUP_ORDER:
        all_fpr, all_tpr = [], []
        suffix, pkl_name = _FLIPPED_FEAT_SUFFIX[feat]
        for sess_label, (var_base, ts_map) in _FLIPPED_SESSIONS.items():
            variant = var_base + suffix
            run_ts = ts_map.get(feat, ts_map.get("xp", ""))
            # Try multiple h/m combinations, pick the first available
            found = False
            for h in _H_VALUES:
                for m in _M_VALUES:
                    hm_dir = cls_root / variant / run_ts / f"h{h}_m{m}"
                    pkl_path = hm_dir / f"LDA_{pkl_name}_flipped.pkl"
                    roc = _load_roc(pkl_path)
                    if roc is not None:
                        fpr, tpr, _ = roc
                        all_fpr.append(fpr)
                        all_tpr.append(tpr)
                        found = True
                        break  # use first h for this session
                if found:
                    break

        if all_fpr:
            common_fpr = np.linspace(0, 1, 200)
            interp_tpr = np.array([np.interp(common_fpr, f, t) for f, t in zip(all_fpr, all_tpr)])
            mean_tpr = interp_tpr.mean(axis=0)
            color = _FEAT_COLORS.get(feat, "#888888")
            fig.add_trace(
                go.Scatter(
                    x=common_fpr, y=mean_tpr, mode="lines",
                    name=_FEAT_SHORT.get(feat, feat),
                    line=dict(color=color, width=2.0),
                    legendgroup=feat, showlegend=False,
                ),
                row=1, col=2,
            )

    # Diagonal chance line on both panels
    for col in (1, 2):
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color=COLOR_CHANCE, width=1.0, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=col,
        )

    apply_thesis_style(fig, theme, height=480, margin=dict(l=60, r=40, t=56, b=80))

    for col in (1, 2):
        fig.update_xaxes(
            title_text="False Positive Rate",
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            range=[0, 1], row=1, col=col,
        )
        fig.update_yaxes(
            title_text="True Positive Rate" if col == 1 else "",
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            range=[0, 1.05], row=1, col=col,
        )

    return fig
