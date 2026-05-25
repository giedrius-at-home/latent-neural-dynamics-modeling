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
# # Section 5: DBS classification
#
# DBS state classification from PSID latent states (Xp = prediction features).
#
# | Fig | Content |
# |-----|---------|
# | 49  | cv_ba per feature source (predictions + forecasts, behavioral + neural) |
# | 50  | cv_ba heatmap (cells x feature sources) |
# | 51  | Forecast flipped-label BA (adversarial null check) |
# | 52b | Generalisation gap: within − flipped BA (forecast mode) |
# | 53  | t_cut analysis — ba_at_score vs time window |
#
# Note: ROC curves (Fig 52) and confusion matrices (Fig 49a) require per-trial
# y_true/y_pred which are not stored in the sweep parquet format. Those figures
# are skipped. DPAD classification has not been run (no sweep parquets exist).

# %%
import sys, os

os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")
sys.path.insert(0, ".")
sys.path.insert(0, "notebooks")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, TwoSlopeNorm

from thesis_style import (
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_PSID,
    COLOR_VARMA,
    apply_thesis_style,
    panel_label,
)
from thesis_loaders import (
    load_latest_classification,
    inspect_available_results,
    EXP_BEHAVIORAL,
    EXP_NEURAL,
    SESSIONS,
)

apply_thesis_style()

OUT = Path("thesis_figures/sec5")
OUT.mkdir(parents=True, exist_ok=True)
results_root = Path("results").resolve()

# %% [markdown]
# ## Available results

# %%
display(inspect_available_results(results_root))

# %% [markdown]
# ## Data loading

# %%
# Load latest PSID classification sweep parquets for all 4 sessions x 2 exp types.
cls_beh: dict[str, pd.DataFrame] = {}
cls_neu: dict[str, pd.DataFrame] = {}
for session in SESSIONS:
    df_b = load_latest_classification(results_root, "psid", EXP_BEHAVIORAL, session)
    if df_b is not None:
        cls_beh[session] = df_b
    df_n = load_latest_classification(results_root, "psid", EXP_NEURAL, session)
    if df_n is not None:
        cls_neu[session] = df_n

print(f"Behavioral: {list(cls_beh.keys())}")
print(f"Neural:     {list(cls_neu.keys())}")

# %%
# Summary table — cv_ba for Xp (predictions, dbs_train=both, not flipped)
rows = []
for session in SESSIONS:
    row = {"session": session}
    for label, data in [("beh Xp", cls_beh), ("beh Xp_with_dbs", cls_beh),
                         ("neu Xp", cls_neu), ("neu Xp_with_dbs", cls_neu)]:
        src = label.split()[-1]
        exp_data = data.get(session)
        if exp_data is not None:
            pred = exp_data[
                (exp_data["mode"] == "predictions")
                & (~exp_data["flipped"])
                & (exp_data["dbs_train"] == "both")
            ]
            sub = pred[pred["sub_source"] == src]
            row[label] = f"{sub['cv_ba'].iloc[0]:.3f}" if len(sub) else "-"
        else:
            row[label] = "-"
    rows.append(row)
display(pd.DataFrame(rows).set_index("session"))

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PRED_SOURCES = ["Xp", "Xp_1", "Xp_2", "Xp_with_dbs"]
FCST_SOURCES = ["Xf", "Xf_1", "Xf_2", "Xf_with_dbs"]
SOURCE_SHORT = {
    "Xp": "Xp", "Xp_1": "Xp1", "Xp_2": "Xp2", "Xp_with_dbs": "Xp+DBS",
    "Xf": "Xf", "Xf_1": "Xf1", "Xf_2": "Xf2", "Xf_with_dbs": "Xf+DBS",
}
SOURCE_COLORS = {
    "Xp": COLOR_PSID, "Xp_1": "#0F6E56", "Xp_2": "#993C1D", "Xp_with_dbs": "#854F0B",
    "Xf": COLOR_PSID, "Xf_1": "#0F6E56", "Xf_2": "#993C1D", "Xf_with_dbs": "#854F0B",
}


def _cv_ba(df: pd.DataFrame, mode: str, src: str, *, flipped: bool = False) -> float:
    """Extract cv_ba for (mode, sub_source) from a classification parquet."""
    if df is None:
        return float("nan")
    sub = df[
        (df["mode"] == mode)
        & (df["flipped"] == flipped)
        & (df["dbs_train"] == "both")
        & (df["sub_source"] == src)
    ]
    return float(sub["cv_ba"].iloc[0]) if len(sub) else float("nan")


def _best_ba_at_score(df: pd.DataFrame, mode: str, src: str) -> tuple[float, float]:
    """Return (best ba_at_score, t_cut_seconds) across the t_cut grid."""
    if df is None:
        return float("nan"), float("nan")
    sub = df[
        (df["mode"] == mode)
        & (~df["flipped"])
        & (df["dbs_train"] == "both")
        & (df["sub_source"] == src)
    ]
    if not len(sub):
        return float("nan"), float("nan")
    idx = sub["ba_at_score"].idxmax()
    return float(sub.loc[idx, "ba_at_score"]), float(sub.loc[idx, "t_cut_seconds"])

# %% [markdown]
# ## Fig 49: cv_ba by feature source — predictions and forecasts

# %%
# 2x2 grid: rows = behavioral/neural, cols = predictions/forecasts.
# One dot per cell; x = feature source; y = cv_ba.
rng = np.random.default_rng(42)

fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), sharey=False)
mode_pairs = [("predictions", PRED_SOURCES), ("forecast", FCST_SOURCES)]
exp_pairs = [("behavioral", cls_beh), ("neural", cls_neu)]

for row_i, (exp_label, cls_data) in enumerate(exp_pairs):
    for col_i, (mode, sources) in enumerate(mode_pairs):
        ax = axes[row_i, col_i]
        ax.axhline(0.5, color="#888888", lw=0.7, linestyle="--", alpha=0.6, zorder=0)
        for xi, src in enumerate(sources):
            color = SOURCE_COLORS[src]
            vals = [_cv_ba(cls_data.get(session), mode, src) for session in SESSIONS]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            jt = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(xi + jt, vals, s=30, c=color, alpha=0.85, linewidths=0, zorder=3)
            ax.hlines(
                np.median(vals), xi - 0.22, xi + 0.22, colors=color, lw=1.8, zorder=4
            )
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([SOURCE_SHORT[s] for s in sources], rotation=30, ha="right")
        ax.set_ylim(0.35, 1.05)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        panel_label(ax, chr(ord("A") + row_i * 2 + col_i), f"{exp_label} - {mode}")
        if col_i == 0:
            ax.set_ylabel("cv balanced accuracy")

fig.tight_layout()
fig.savefig(str(OUT / "fig_049_classification_cv_ba.png"))
plt.show()
print(
    "Fig 49: cv_ba per feature source. Rows: behavioral/neural decoding mode. "
    "Cols: prediction features (Xp*) vs forecast features (Xf*). "
    "One dot per session; median bar. Dashed line at chance (0.5). "
    "Xp_with_dbs / Xf_with_dbs at 1.0 (trivially classifiable — includes DBS channel)."
)

# %% [markdown]
# ## Fig 50: cv_ba heatmap — cells x feature sources

# %%
def _cv_ba_matrix(cls_data: dict, sources: list, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (cv_ba matrix, p_value matrix) with rows=SESSIONS, cols=sources."""
    ba = np.full((len(SESSIONS), len(sources)), np.nan)
    pv = np.full((len(SESSIONS), len(sources)), np.nan)
    for ri, session in enumerate(SESSIONS):
        df = cls_data.get(session)
        if df is None:
            continue
        for ci, src in enumerate(sources):
            sub = df[
                (df["mode"] == mode)
                & (~df["flipped"])
                & (df["dbs_train"] == "both")
                & (df["sub_source"] == src)
            ]
            if len(sub):
                ba[ri, ci] = float(sub["cv_ba"].iloc[0])
                if not sub["p_value"].isna().all():
                    pv[ri, ci] = float(sub["p_value"].dropna().iloc[0])
    return ba, pv


cmap_hm = plt.cm.RdBu_r
norm_hm = TwoSlopeNorm(vmin=0.4, vcenter=0.5, vmax=0.8)


def _draw_heatmap(ax, ba: np.ndarray, pv: np.ndarray, *, title: str, sources: list, sessions: list):
    rgba = cmap_hm(norm_hm(np.clip(ba, 0.4, 0.8)))
    alpha = np.where(np.isfinite(pv) & (pv < 0.05), 1.0, 0.65)
    alpha = np.where(np.isfinite(ba), alpha, 0.0)
    rgba[..., -1] = alpha
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    for ri in range(len(sessions)):
        for ci in range(len(sources)):
            if not np.isfinite(ba[ri, ci]):
                ax.add_patch(
                    plt.Rectangle(
                        (ci - 0.5, ri - 0.5), 1, 1,
                        facecolor="#EEEEEE", edgecolor="#BBBBBB", hatch="///", lw=0.5
                    )
                )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([SOURCE_SHORT[s] for s in sources], rotation=30, ha="right")
    ax.set_yticks(range(len(sessions)))
    ax.set_yticklabels(sessions)
    ax.set_title(title, loc="left", fontsize=9)


beh_pred_ba, beh_pred_pv = _cv_ba_matrix(cls_beh, PRED_SOURCES, "predictions")
neu_pred_ba, neu_pred_pv = _cv_ba_matrix(cls_neu, PRED_SOURCES, "predictions")

fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
_draw_heatmap(axes[0], beh_pred_ba, beh_pred_pv, title="Behavioral (prediction Xp*)", sources=PRED_SOURCES, sessions=list(SESSIONS))
_draw_heatmap(axes[1], neu_pred_ba, neu_pred_pv, title="Neural (prediction Xp*)", sources=PRED_SOURCES, sessions=list(SESSIONS))
panel_label(axes[0], "A", "")
panel_label(axes[1], "B", "")

sm = plt.cm.ScalarMappable(cmap=cmap_hm, norm=norm_hm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.04, pad=0.02, label="cv balanced accuracy")
cbar.ax.axhline(0.5, color="#222222", lw=0.8, linestyle="--")

fig.savefig(str(OUT / "fig_050_classification_heatmap.png"))
plt.show()
print(
    "Fig 50: cv_ba heatmap. Rows: cells (PDI1_S2 to PDI4_S3). "
    "Cols: feature sources. Color: cv balanced accuracy (RdBu_r centred at 0.5). "
    "Cell alpha: full if permutation p<0.05 (only Xp_with_dbs has p_value). "
    "Top: behavioral decoding (Xp from behavior-trained model). "
    "Bottom: neural decoding (Xp from LFP-trained model)."
)

# %% [markdown]
# ## Fig 51: Forecast flipped-label BA (null check)
#
# `flipped=True` rows use the same model but with DBS labels permuted.
# If `flipped cv_ba` is close to `non-flipped cv_ba`, the classifier
# is exploiting a non-DBS structure (e.g. temporal autocorrelation).
# This is a within-condition adversarial null, NOT cross-condition transfer.

# %%
beh_fcst_ba, _ = _cv_ba_matrix(cls_beh, FCST_SOURCES, "forecast")
neu_fcst_ba, _ = _cv_ba_matrix(cls_neu, FCST_SOURCES, "forecast")

def _flipped_ba_matrix(cls_data, sources, mode):
    ba = np.full((len(SESSIONS), len(sources)), np.nan)
    for ri, session in enumerate(SESSIONS):
        df = cls_data.get(session)
        if df is None:
            continue
        for ci, src in enumerate(sources):
            sub = df[
                (df["mode"] == mode)
                & (df["flipped"] == True)
                & (df["dbs_train"] == "both")
                & (df["sub_source"] == src)
            ]
            if len(sub):
                ba[ri, ci] = float(sub["cv_ba"].iloc[0])
    return ba

beh_fcst_flip_ba = _flipped_ba_matrix(cls_beh, FCST_SOURCES, "forecast")
neu_fcst_flip_ba = _flipped_ba_matrix(cls_neu, FCST_SOURCES, "forecast")

fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
_draw_heatmap(axes[0], beh_fcst_flip_ba, np.full_like(beh_fcst_flip_ba, np.nan),
              title="Behavioral forecast (flipped labels)", sources=FCST_SOURCES, sessions=list(SESSIONS))
_draw_heatmap(axes[1], neu_fcst_flip_ba, np.full_like(neu_fcst_flip_ba, np.nan),
              title="Neural forecast (flipped labels)", sources=FCST_SOURCES, sessions=list(SESSIONS))
panel_label(axes[0], "A", "")
panel_label(axes[1], "B", "")

sm2 = plt.cm.ScalarMappable(cmap=cmap_hm, norm=norm_hm)
sm2.set_array([])
fig.colorbar(sm2, ax=axes, fraction=0.04, pad=0.02, label="cv balanced accuracy (flipped)")

fig.savefig(str(OUT / "fig_051_flipped_heatmap.png"))
plt.show()
print(
    "Fig 51: Forecast classification with permuted DBS labels (null check). "
    "High cv_ba with flipped labels means classifier exploits non-DBS structure. "
    "Compare with Fig 52b (within - flipped delta) to gauge genuine DBS signal."
)

# %% [markdown]
# ## Fig 52b: Generalisation gap — within minus flipped (forecast)

# %%
gap_beh = beh_fcst_ba - beh_fcst_flip_ba
gap_neu = neu_fcst_ba - neu_fcst_flip_ba

div_cmap = plt.cm.RdBu_r
div_norm = TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4)


def _draw_diff_heatmap(ax, mat, *, title, sources, sessions):
    rgba = div_cmap(div_norm(np.clip(mat, -0.4, 0.4)))
    rgba[..., -1] = np.where(np.isfinite(mat), 1.0, 0.0)
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    for ri in range(len(sessions)):
        for ci in range(len(sources)):
            if not np.isfinite(mat[ri, ci]):
                ax.add_patch(
                    plt.Rectangle(
                        (ci - 0.5, ri - 0.5), 1, 1,
                        facecolor="#EEEEEE", edgecolor="#BBBBBB", hatch="///", lw=0.5
                    )
                )
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([SOURCE_SHORT[s] for s in sources], rotation=30, ha="right")
    ax.set_yticks(range(len(sessions)))
    ax.set_yticklabels(sessions)
    ax.set_title(title, loc="left", fontsize=9)


fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
_draw_diff_heatmap(axes[0], gap_beh, title="Behavioral forecast gap (within - flipped)", sources=FCST_SOURCES, sessions=list(SESSIONS))
_draw_diff_heatmap(axes[1], gap_neu, title="Neural forecast gap (within - flipped)", sources=FCST_SOURCES, sessions=list(SESSIONS))
panel_label(axes[0], "A", "")
panel_label(axes[1], "B", "")

sm3 = plt.cm.ScalarMappable(cmap=div_cmap, norm=div_norm)
sm3.set_array([])
cbar3 = fig.colorbar(sm3, ax=axes, fraction=0.04, pad=0.02, label="delta cv_ba (within - flipped)")
cbar3.ax.axhline(0.0, color="#222222", lw=0.8, linestyle="--")

fig.savefig(str(OUT / "fig_052b_gap_heatmap.png"))
plt.show()
print(
    "Fig 52b: Generalisation gap = within_cv_ba - flipped_cv_ba (forecast mode). "
    "Red (positive) = model classifies real DBS labels better than shuffled labels. "
    "Near-zero = performance indistinguishable from permuted null. "
    "Note: flipped=True rows in the parquet use permuted DBS labels within the same "
    "session data — this is NOT cross-condition transfer (train on OFF, score on ON)."
)

# %% [markdown]
# ## Fig 53: Balanced accuracy vs time window (t_cut analysis)
#
# How much signal time is needed to reach peak classification accuracy?
# ba_at_score varies with t_cut_seconds; cv_ba is constant (training-pool CV).

# %%
pred_sources_show = ["Xp", "Xp_1", "Xp_2"]  # skip Xp_with_dbs (trivially 1.0)

fig, axes = plt.subplots(2, len(SESSIONS), figsize=(11.0, 6.5), sharey="row", sharex=True)

exp_pairs_plot = [("behavioral", cls_beh), ("neural", cls_neu)]

for row_i, (exp_label, cls_data) in enumerate(exp_pairs_plot):
    for col_i, session in enumerate(SESSIONS):
        ax = axes[row_i, col_i]
        df = cls_data.get(session)
        ax.axhline(0.5, color="#888888", lw=0.6, linestyle="--", alpha=0.5, zorder=0)
        if df is not None:
            pred_rows = df[
                (df["mode"] == "predictions")
                & (~df["flipped"])
                & (df["dbs_train"] == "both")
            ]
            for src in pred_sources_show:
                sub = pred_rows[pred_rows["sub_source"] == src].sort_values("t_cut_seconds")
                if not len(sub):
                    continue
                color = SOURCE_COLORS[src]
                ax.plot(sub["t_cut_seconds"], sub["ba_at_score"], color=color,
                        lw=1.5, label=SOURCE_SHORT[src])
                ax.axhline(sub["cv_ba"].iloc[0], color=color, lw=0.8,
                           linestyle=":", alpha=0.6)
        ax.set_ylim(0.35, 1.05)
        if row_i == 0:
            ax.set_title(session.replace("_", " "), fontsize=9)
        if col_i == 0:
            ax.set_ylabel(f"{exp_label}\nba at score")
        if row_i == 1:
            ax.set_xlabel("t cut (s)")
        panel_label(ax, chr(ord("A") + row_i * len(SESSIONS) + col_i), "")

# Single legend from last ax
handles, labels = axes[0, -1].get_legend_handles_labels()
if handles:
    axes[0, -1].legend(handles=handles, labels=labels, fontsize=7, loc="lower right")

fig.tight_layout()
fig.savefig(str(OUT / "fig_053_tcut_analysis.png"))
plt.show()
print(
    "Fig 53: ba_at_score vs t_cut_seconds (solid lines) with cv_ba horizontal reference (dotted). "
    "Rows: behavioral / neural. Cols: cells. Xp_with_dbs excluded (trivially 1.0). "
    "Solid lines show how test accuracy grows with more signal time; "
    "dotted reference is the training-pool CV estimate."
)

# %%
# Skipped figures (require per-trial y_true/y_pred not available in sweep parquet):
# - Fig 49a: Confusion matrix grid
# - Fig 52: ROC curves
# - Fig 52a: DPAD vs PSID test BA (no DPAD classification data)
print(
    "Skipped: Fig 49a (confusion matrix), Fig 52 (ROC curves), Fig 52a (DPAD). "
    "These require per-trial y_true / y_pred which the sweep parquet does not store."
)
