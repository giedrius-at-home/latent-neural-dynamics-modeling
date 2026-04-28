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
# # Sec 2a — Model selection & training diagnostics
#
# * Fig 36 — PSID scree / elbow analysis (justifies n1, nx)
# * Fig 37 — Improved vs vanilla PSID
# * Fig 39 — DPAD training curves

# %%
import sys, os
os.chdir('/home/bobby/repos/latent-neural-dynamics-modeling')
sys.path.insert(0, '.')
sys.path.insert(0, 'notebooks')

import re
import json
import numpy as np
import pandas as pd
import polars as pl
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from thesis_style import (
    COLOR_DBS_OFF, COLOR_DBS_ON, COLOR_DPAD, COLOR_NS, COLOR_PSID,
    apply_thesis_style, panel_label,
)
from thesis_sec2_common import *

apply_thesis_style()

# %% [markdown]
# ## Fig 36: PSID scree / elbow analysis — justification for n1, nx
#
# Two-stage SVD spectra from the PSID identification stage.
# Stage 1 (`ZHat_S`) is the behavior-relevant cross-covariance; its rank
# determines `n1` (behavior-decodable latent dims). Stage 2 (`YHat_S`) is the
# full past-future covariance; its rank determines `nx` (total latent dims).
# Dashed = `n1` chosen. Solid = `nx` chosen. Values come from the canonical
# `psid_variant` strings in `thesis_triplets.csv`.

# %%
_VARIANT_NXN1 = re.compile(r'nx_(\d+)_n(\d+)')


def _parse_nx_n1(variant: str) -> tuple[int, int]:
    m = _VARIANT_NXN1.search(variant)
    if not m:
        raise ValueError(f"no nx_N_nM token in variant {variant!r}")
    return int(m.group(1)), int(m.group(2))


def _session_spectra_dir(label: str) -> str:
    """PDI1_S2 -> results/diagnostic/PDI1_2_psid_spectra."""
    pid, sess = label.split("_S")
    return f"results/diagnostic/{pid}_{sess}_psid_spectra"


def _load_spectra(label: str, mode: str):
    """Return (index, ZHat_S, YHat_S) arrays for one session × mode."""
    p = f"{_session_spectra_dir(label)}/psid/{mode}/spectra.parquet"
    df = pl.read_parquet(p)
    return (df["index"].to_numpy(),
            df["ZHat_S"].to_numpy(),
            df["YHat_S"].to_numpy())


# Elbows are recomputed from the fresh spectra at plot time — no stale YAML
# cache. YAML holds only the thresholds (n1_relative_to_peak, nx_cumulative_energy)
# and target_K per family. See reports/ELBOW_SELECTION.md.
_ELBOW_CFG = yaml.safe_load(open("configs/diagnostic/elbow_choices.yaml"))
_THR_N1 = float(_ELBOW_CFG.get("thresholds", {}).get("n1_relative_to_peak", 0.10))
_THR_NX = float(_ELBOW_CFG.get("thresholds", {}).get("nx_cumulative_energy", 0.90))
_TARGET_K = {
    label: {fam: int(_ELBOW_CFG["sessions"][label][fam].get("target_K", 8))
            for fam in ("ecog", "laplacian")}
    for label in _ELBOW_CFG.get("sessions", {})
}


def _resolve_elbows(label: str, fam: str, z=None, y=None):
    """Manual elbows from configs/diagnostic/elbow_choices.yaml.
    Required keys: sessions[label][fam].{n1, nx}. No auto-compute fallback —
    inspect the spectra_summary PNGs and fill the yaml by eye.
    """
    sess = _ELBOW_CFG.get("sessions", {}).get(label, {}).get(fam, {})
    if "n1" not in sess or "nx" not in sess:
        raise KeyError(
            f"elbow_choices.yaml is missing n1/nx for sessions[{label!r}][{fam!r}]. "
            f"Open results/diagnostic/{label.replace('_S', '_')}_psid_spectra/spectra_summary.png, "
            f"pick elbows by eye, add n1 and nx to the yaml."
        )
    return int(sess["n1"]), int(sess["nx"])

from thesis_loaders import resolve_input_channels

# Per-session label → laplacian triplet lookup so we can pair ecog/laplacian
# rows inside one session's figure.
_LAP_BY_LABEL = {t.label: t for t in ALL_TRIPLETS_LAP}


def _plot_session_scree(label: str, fig_letter: str):
    """Single-session 2x2: rows = (ecog, laplacian), cols = (stage 1, stage 2).
    Dashed = n1; solid = n2 = nx - n1. n1/nx computed from the fresh
    spectra at render time using the thresholds in elbow_choices.yaml.
    """
    idx_e, z_e, y_e = _load_spectra(label, "ecog")
    idx_l, z_l, y_l = _load_spectra(label, "laplacian")

    ecog_n1, ecog_nx = _resolve_elbows(label, "ecog")
    lap_n1, lap_nx = _resolve_elbows(label, "laplacian")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2))

    rows = [
        ("ECOG & Tracing Kinematics",      axes[0], idx_e, z_e, y_e, ecog_n1, ecog_nx),
        ("ECOG & Laplacian LFP",           axes[1], idx_l, z_l, y_l, lap_n1, lap_nx),
    ]
    for row_i, (mode, (ax1, ax2), idx, z_s, y_s, n1, nx) in enumerate(rows):
        n2 = nx - n1

        ax1.plot(idx, z_s, color=COLOR_PSID, marker="o", markersize=2.0, linewidth=0.8)
        ax1.axvline(n1, color=COLOR_PSID, linestyle="--", linewidth=1.2,
                    label=f"$n_1$ = {n1}")
        ax1.set_ylabel(r"singular value $\sigma_i$")
        ax1.legend()
        panel_label(ax1, chr(ord("A") + row_i * 2),
                    f"{mode}, behaviour-relevant stage")

        ax2.plot(idx, y_s, color=COLOR_PSID, marker="o", markersize=1.4, linewidth=0.7)
        ax2.axvline(n2, color=COLOR_DBS_OFF, linestyle="-", linewidth=1.2,
                    label=f"$n_2$ = {n2}")
        ax2.legend()
        panel_label(ax2, chr(ord("A") + row_i * 2 + 1),
                    f"{mode}, residual-dynamics stage")

        if row_i == 1:
            ax1.set_xlabel(r"index $i$")
            ax2.set_xlabel(r"index $i$")
    fig.savefig(str(OUT / f"fig_036{fig_letter}_psid_scree_{label}.png"))
    plt.show()


for letter, tri in zip("abcd", ALL_TRIPLETS):
    _plot_session_scree(tri.label, letter)
    e_n1, e_nx = _resolve_elbows(tri.label, "ecog")
    l_n1, l_nx = _resolve_elbows(tri.label, "laplacian")
    e_K = _TARGET_K.get(tri.label, {}).get("ecog", 8)
    l_K = _TARGET_K.get(tri.label, {}).get("laplacian", 8)
    print(
        f"Fig 36{letter}: {tri.label} PSID scree — "
        f"ecog (n1={e_n1}, nx={e_nx}, K={e_K})  "
        f"laplacian (n1={l_n1}, nx={l_nx}, K={l_K}). "
        f"Manual elbows from configs/diagnostic/elbow_choices.yaml."
    )

# %% [markdown]
# ## Fig 44: mRMR channel selection — which top-K features go into PSID/DPAD/VARMA
#
# After the scree analysis fixes `(n1, n2)`, mRMR (minimum-Redundancy
# Maximum-Relevance) picks the top-K neural channels that jointly maximise
# relevance to the behaviour targets while minimising redundancy among the
# selected set.
#
# **Production metric: Mazzanti MIQ.** Implementation uses
# `mrmr-selection` ([smazzanti/mrmr](https://github.com/smazzanti/mrmr)),
# called once per behaviour target (`tracing_velocity_x`,
# `tracing_acceleration_magnitude`) and aggregated by per-target vote-rank:
#
# 1. **Relevance** — mean mutual information `MI(feature_i, target_j)` from
#    sklearn's k-NN MI estimator (n_neighbors=3, capped at 30k samples,
#    train-blocks only).
# 2. **Redundancy** — mean absolute Pearson correlation `|r(feat_i, sel_j)|`
#    among already-selected features.
# 3. **Score** — quotient form (FCQ): `relevance / (mean(redundancy) + ε)`.
#    Mazzanti's `denominator="mean"`.
# 4. **Multi-target aggregation** — run `mrmr_regression` per target with a
#    pool of 3K, then aggregate by vote-rank: rank-r feature contributes
#    `pool - r` votes; final top-K = highest-vote features. Captures targets
#    where one channel dominates without averaging away signal.
#
# Three panels under **symmetric Mazzanti MIQ** selection — same algorithm
# (per-target MI relevance, Pearson redundancy, FCQ quotient, vote-rank
# across targets, pool=3K), only the (features, targets) pair changes per
# panel. Full method docs in `reports/feature_selection_methods.md`.
#
# * **Panel A — ECoG-Y for behavioural mode.** 60 ECoG narrow-bands ranked
#   against 2 behaviour dims (tracing velocity_x, accel_magnitude). Cache:
#   `configs/diagnostic/mrmr_picks.yaml` family `ecog`.
# * **Panel B — ECoG-Y for laplacian mode.** 60 ECoG narrow-bands ranked
#   against 15 LFP_14-16 bands (matches the `ECoG → LFP` training task in
#   laplacian mode). Cache: `mrmr_picks.yaml` family `laplacian`.
# * **Panel C — LFP-Z for laplacian mode.** 15 LFP_14-16 bands ranked
#   against the 60 ECoG bands (predictability — picks LFP bands the
#   cortical input set can predict). Cache: `mrmr_picks.yaml` family
#   `lfp_z_predictability`.
#
# Heatmap gradients = precomputed mean MI with kinematics
# (`scripts/precompute_mi_relevance.py` →
# `mi_relevance/{ecog,laplacian,lfp_z}.parquet`). Outlined cells = top-K
# picks ranked 1..K. K = 8 per family.
#
# Heatmap gradient is **mean MI** with the target set, precomputed once by
# `scripts/precompute_mi_relevance.py` and cached at
# `results/diagnostic/{P}_{S}_psid_spectra/mi_relevance/{family}.parquet`
# (families: `ecog`, `lfp_z`). Outlined cells = top-K picks ranked 1..K.
# All production models use **K = 8** per family.

# %%



_MRMR_PICKS = yaml.safe_load(
    open("configs/diagnostic/mrmr_picks.yaml")
)["sessions"]


def _mrmr_select(corr_df, candidate_features, behavior_names, K, *, label=None, family=None):
    """Pull the cached MI-mRMR top-K picks written by
    ``scripts/_pipeline_common.py:mrmr_top_k_from_diagnostic`` (MIQ variant:
    MI relevance averaged across targets, Pearson redundancy).

    ``corr_df``/``candidate_features``/``behavior_names`` are kept for API
    compatibility with the heatmap grids but are no longer used for picking.
    Falls back to greedy Pearson mRMR on ``corr_df`` when the yaml entry is
    missing — so sec2a renders even before the MI cache is written.
    """
    if label and family:
        rec = _MRMR_PICKS.get(label, {}).get(family)
        if rec and "selected" in rec:
            return list(rec["selected"][:K])

    abs_corr = corr_df.abs()
    rel = abs_corr.loc[candidate_features, behavior_names].mean(axis=1)
    selected: list[str] = []
    remaining = set(candidate_features)
    for _ in range(min(K, len(candidate_features))):
        if not selected:
            score = rel.copy()
        else:
            red = abs_corr.loc[list(remaining), selected].mean(axis=1)
            score = rel.loc[list(remaining)] - red
        best = score.idxmax()
        selected.append(best)
        remaining.discard(best)
    return selected


_ECOG_RX = re.compile(r"^ECOG_(\d)_(\w+)_raw$")
_LAP_RX = re.compile(r"^LAPLACIAN_14-16_LFP_(\w+)_raw$")
_BAND_ORDER = [
    "theta_4_8", "alpha_8_12",
    "beta_12_17", "beta_17_22", "beta_22_27", "beta_27_30",
    "gamma_30_35", "gamma_35_40", "gamma_40_45", "gamma_45_50",
    "gamma_50_55", "gamma_55_60", "gamma_60_65", "gamma_70_75", "gamma_75_80",
]
_BAND_SHORT = {
    "theta_4_8": "4-8",   "alpha_8_12": "8-12",
    "beta_12_17": "12-17", "beta_17_22": "17-22",
    "beta_22_27": "22-27", "beta_27_30": "27-30",
    "gamma_30_35": "30-35", "gamma_35_40": "35-40",
    "gamma_40_45": "40-45", "gamma_45_50": "45-50",
    "gamma_50_55": "50-55", "gamma_55_60": "55-60",
    "gamma_60_65": "60-65", "gamma_70_75": "70-75", "gamma_75_80": "75-80",
}
_BEHAVIOR = ["tracing_velocity_x", "tracing_acceleration_magnitude"]
_K = 8


def _ecog_grid(corr_df, selected):
    """4x15 |corr| matrix + {(row,col): rank} for the selected set."""
    mat = np.full((4, len(_BAND_ORDER)), np.nan)
    rank = {}
    rel = corr_df.abs().loc[:, _BEHAVIOR].mean(axis=1)
    for feat, val in rel.items():
        m = _ECOG_RX.match(feat)
        if not m:
            continue
        electrode = int(m.group(1)) - 1
        band = m.group(2)
        if band in _BAND_ORDER:
            mat[electrode, _BAND_ORDER.index(band)] = float(val)
    for i, feat in enumerate(selected, start=1):
        m = _ECOG_RX.match(feat)
        if m:
            rank[(int(m.group(1)) - 1, _BAND_ORDER.index(m.group(2)))] = i
    return mat, rank


def _lap_grid(corr_df, selected):
    """1x15 |corr| row + {col: rank} for the selected set."""
    mat = np.full((1, len(_BAND_ORDER)), np.nan)
    rank = {}
    rel = corr_df.abs().loc[:, _BEHAVIOR].mean(axis=1)
    for feat, val in rel.items():
        m = _LAP_RX.match(feat)
        if not m:
            continue
        band = m.group(1)
        if band in _BAND_ORDER:
            mat[0, _BAND_ORDER.index(band)] = float(val)
    for i, feat in enumerate(selected, start=1):
        m = _LAP_RX.match(feat)
        if m:
            rank[(0, _BAND_ORDER.index(m.group(1)))] = i
    return mat, rank


def _plot_cells(ax, mat, rank, ylabels, cmap="Blues"):
    vmax = float(np.nanmax(mat))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(_BAND_ORDER)))
    ax.set_xticklabels([_BAND_SHORT[b] for b in _BAND_ORDER], rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    for (ri, ci), n in rank.items():
        ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1.0, 1.0,
                                    fill=False, edgecolor="black", linewidth=1.5))
        # Selection order number, auto-contrast text color
        v = mat[ri, ci]
        norm = v / max(vmax, 1e-9)
        ax.text(ci, ri, str(n), ha="center", va="center",
                color="white" if norm > 0.55 else "black",
                fontsize=8, fontweight="bold")
    return im


_MI_REL_CACHE: dict = {}


def _mi_relevance_from_parquet(label: str, family: str) -> dict:
    """Read precomputed mean-MI per feature from
    ``results/diagnostic/{P}_{S}_psid_spectra/mi_relevance/{family}.parquet``
    (produced by ``scripts/precompute_mi_relevance.py``). Returns
    ``{feat: mean_MI_in_nats}``. Empty dict if missing — heatmap will then
    fall back to all-NaN (white cells), and the user should run the precompute
    script."""
    if (label, family) in _MI_REL_CACHE:
        return _MI_REL_CACHE[(label, family)]
    pid, sess = label.split("_S")
    p = Path(f"results/diagnostic/{pid}_{sess}_psid_spectra/mi_relevance/{family}.parquet")
    if not p.exists():
        _MI_REL_CACHE[(label, family)] = {}
        return {}
    df = pd.read_parquet(p)
    out = dict(zip(df["feature"].tolist(), df["relevance"].astype(float).tolist()))
    _MI_REL_CACHE[(label, family)] = out
    return out


def _mi_relevance_from_yaml(label: str, family: str) -> dict:
    """Pull cached ``relevance`` dict {feat: mean_MI} for the given cell from
    mrmr_picks.yaml. Returns empty dict if missing (notebook then falls back
    to Pearson |r| gradient)."""
    rec = _MRMR_PICKS.get(label, {}).get(family, {})
    return dict(rec.get("relevance", {}))


def _compute_ecog_lap_cross(participant: str, session: str, max_trials: int = 30):
    """Compute mean |Pearson| between each ECoG feature (60) and the 15
    LAPLACIAN_14-16 LFP bands, averaged over train-block trials. Returns a
    dict {ecog_feat: mean_|r|}. Cached in memory via lru-ish dict.
    """
    import polars as pl
    from pathlib import Path
    import yaml as _yaml

    split_cfg = _yaml.safe_load(
        open(f"configs/splits/{participant}_S{session}.yaml")
    )
    train_blocks = set(split_cfg["train_blocks"])
    sess_dir = Path(
        f"resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band/"
        f"participant_id={participant}/session={session}"
    )
    per_feat = {}
    n_trials_seen = 0
    for b_dir in sorted(sess_dir.glob("block=*")):
        bid = int(b_dir.name.split("=")[1])
        if bid not in train_blocks:
            continue
        pq = list(b_dir.glob("*.parquet"))
        if not pq:
            continue
        df = pl.read_parquet(pq[0])
        ecog_cols = [c for c in df.columns if c.startswith("ECOG_") and c.endswith("_raw")]
        lap_cols = [c for c in df.columns
                    if c.startswith("LAPLACIAN_14-16_LFP") and c.endswith("_raw")]
        for row in df.iter_rows(named=True):
            if n_trials_seen >= max_trials:
                break
            min_len = min(len(row[c]) for c in ecog_cols + lap_cols)
            E = np.column_stack([np.asarray(row[c][:min_len], float) for c in ecog_cols])
            L = np.column_stack([np.asarray(row[c][:min_len], float) for c in lap_cols])
            mask = np.all(np.isfinite(E), axis=1) & np.all(np.isfinite(L), axis=1)
            E, L = E[mask], L[mask]
            if len(E) < 50:
                continue
            for i, feat in enumerate(ecog_cols):
                x = E[:, i]
                if x.std() == 0:
                    continue
                rs = []
                for j in range(L.shape[1]):
                    y = L[:, j]
                    if y.std() == 0:
                        continue
                    rs.append(abs(np.corrcoef(x, y)[0, 1]))
                if rs:
                    per_feat.setdefault(feat, []).append(np.mean(rs))
            n_trials_seen += 1
        if n_trials_seen >= max_trials:
            break
    return {k: float(np.mean(v)) for k, v in per_feat.items()}


_ECOG_LAP_CACHE: dict = {}


def _plot_session_mrmr(tri, fig_letter: str):
    """Three panels showing the top-K Mazzanti MIQ picks used in the
    laplacian/behavioural pipelines. All three use the same algorithm
    (per-target MIQ + vote-rank); only the (features, targets) pair changes:
      A — behavioural mode ECoG-Y (60 ECoG → 2 kinematics)
      B — laplacian   mode ECoG-Y (60 ECoG → 15 LFP bands)
      C — laplacian   mode LFP-Z (15 LFP → 60 ECoG, predictability)
    Picks loaded from ``configs/diagnostic/mrmr_picks.yaml`` families
    ``ecog``, ``laplacian``, ``lfp_z_predictability``. Heatmap gradients are
    precomputed mean MI with kinematics
    (``mi_relevance/{ecog,laplacian,lfp_z}.parquet``).
    See reports/feature_selection_methods.md for the full method docs.
    """
    base = _session_spectra_dir(tri.label)
    corr_e = pd.read_parquet(f"{base}/correlation/ecog.parquet").set_index("index")
    corr_l = pd.read_parquet(f"{base}/correlation/laplacian.parquet").set_index("index")
    ecog_feats = [c for c in corr_e.columns if _ECOG_RX.match(c)]
    lap_targets = [c for c in corr_l.columns if re.match(r"^LAPLACIAN_14-16_LFP_.*_raw$", c)]

    # Panel A — ECoG-Y picks for behavioural mode.
    sel_behav_y = _mrmr_select(corr_e, ecog_feats, _BEHAVIOR, _K,
                               label=tri.label, family="ecog")
    # Panel B — ECoG-Y picks for laplacian mode (target = LFP bands).
    sel_lap_y = _mrmr_select(corr_l, ecog_feats, lap_targets, _K,
                             label=tri.label, family="laplacian")
    # Panel C — LFP-Z picks via Mazzanti MIQ predictability (LFP_15 → ECoG_60
    # vote-rank). Cache: configs/diagnostic/mrmr_picks.yaml family
    # `lfp_z_predictability`, written by
    # scripts/precompute_lfp_z_predictability_picks.py.
    sel_lap_z = list(
        _MRMR_PICKS.get(tri.label, {}).get("lfp_z_predictability", {}).get("selected", [])
    )[:_K]

    # Heatmap gradients from precomputed MI parquets.
    def _ecog_mi_mat(family: str):
        rel = _mi_relevance_from_parquet(tri.label, family)
        mat = np.full((4, len(_BAND_ORDER)), np.nan)
        for feat, val in rel.items():
            m = _ECOG_RX.match(feat)
            if not m:
                continue
            mat[int(m.group(1)) - 1, _BAND_ORDER.index(m.group(2))] = float(val)
        return mat

    def _lfp_mi_row():
        rel = _mi_relevance_from_parquet(tri.label, "lfp_z")
        mat = np.full((1, len(_BAND_ORDER)), np.nan)
        for feat, val in rel.items():
            m = _LAP_RX.match(feat)
            if not m:
                continue
            band = m.group(1)
            if band in _BAND_ORDER:
                mat[0, _BAND_ORDER.index(band)] = float(val)
        return mat

    mat_behav = _ecog_mi_mat("ecog")
    mat_lap_y = _ecog_mi_mat("laplacian")
    mat_lap_z = _lfp_mi_row()

    def _ecog_rank_map(selected):
        out = {}
        for i, feat in enumerate(selected, start=1):
            m = _ECOG_RX.match(feat)
            if m:
                out[(int(m.group(1)) - 1, _BAND_ORDER.index(m.group(2)))] = i
        return out

    def _lfp_rank_map(selected):
        out = {}
        for i, feat in enumerate(selected, start=1):
            m = _LAP_RX.match(feat)
            if m and m.group(1) in _BAND_ORDER:
                out[(0, _BAND_ORDER.index(m.group(1)))] = i
        return out

    rank_behav = _ecog_rank_map(sel_behav_y)
    rank_lap_y = _ecog_rank_map(sel_lap_y)
    rank_lap_z = _lfp_rank_map(sel_lap_z)

    fig, axes = plt.subplots(
        3, 1, figsize=(5.4, 5.4),
        gridspec_kw=dict(height_ratios=[4, 4, 1.5]),
    )

    im_a = _plot_cells(axes[0], mat_behav, rank_behav,
                       ylabels=[f"ECoG {i}" for i in range(1, 5)])
    panel_label(axes[0], "A", "Behavioural — ECoG (Y) → kinematics (Z)")

    im_b = _plot_cells(axes[1], mat_lap_y, rank_lap_y,
                       ylabels=[f"ECoG {i}" for i in range(1, 5)])
    panel_label(axes[1], "B", "Laplacian — ECoG (Y) picks vs LFP targets")

    im_c = _plot_cells(axes[2], mat_lap_z, rank_lap_z,
                       ylabels=["LAP 14-16"])
    panel_label(axes[2], "C", "Laplacian — LFP (Z) picks vs ECoG predictability")
    axes[2].set_xlabel("narrow band (Hz)")

    fig.colorbar(im_a, ax=axes[0], shrink=0.85, label="mean MI with kinematics (nats)")
    fig.colorbar(im_b, ax=axes[1], shrink=0.85, label="mean MI with LFP bands (nats)")
    fig.colorbar(im_c, ax=axes[2], shrink=0.85, label="mean MI with kinematics (nats)")
    fig.savefig(str(OUT / f"fig_044{fig_letter}_mrmr_selection_{tri.label}.png"))
    plt.show()
    return sel_behav_y, sel_lap_y, sel_lap_z


for letter, tri in zip("abcd", ALL_TRIPLETS):
    sel_behav_y, sel_lap_y, sel_lap_z = _plot_session_mrmr(tri, letter)
    print(
        f"Fig 44{letter}: {tri.label} top-{_K} picks (Mazzanti MIQ symmetric).\n"
        f"  behav  Y input (ECoG → kinematics):       {sel_behav_y}\n"
        f"  lap    Y input (ECoG → LFP):               {sel_lap_y}\n"
        f"  lap    Z output (LFP → ECoG predictability): {sel_lap_z}"
    )

# %% [markdown]
# ## Fig 45: feature correlation structure (60×60)
#
# Full feature-feature absolute correlation matrix. Grid lines separate the 4
# electrodes. The dim diagonal blocks reveal within-electrode band-to-band
# correlations (weak — narrow-band filtering makes different frequencies nearly
# orthogonal); the bright anti-diagonal stripes in the off-diagonal blocks are
# volume-conduction redundancy (same-band signals on different electrodes pick
# up nearly identical cortical sources).

# %%
def _plot_correlation_matrix(tri, fig_letter: str):
    base = _session_spectra_dir(tri.label)
    corr = pd.read_parquet(f"{base}/correlation/ecog.parquet").set_index("index")
    ecog_feats = [c for c in corr.columns if _ECOG_RX.match(c)]
    ff = corr.abs().loc[ecog_feats, ecog_feats].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.3))
    im = ax.imshow(ff, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    for k in (15, 30, 45):
        ax.axvline(k - 0.5, color="black", linewidth=0.6)
        ax.axhline(k - 0.5, color="black", linewidth=0.6)
    ax.set_xticks([7.5, 22.5, 37.5, 52.5])
    ax.set_xticklabels([f"ECoG {i}" for i in range(1, 5)])
    ax.set_yticks([7.5, 22.5, 37.5, 52.5])
    ax.set_yticklabels([f"ECoG {i}" for i in range(1, 5)])
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$|r|$")
    fig.savefig(str(OUT / f"fig_045{fig_letter}_corrmat_{tri.label}.png"))
    plt.show()


for letter, tri in zip("abcd", ALL_TRIPLETS):
    _plot_correlation_matrix(tri, letter)
    print(f"Fig 45{letter}: {tri.label} ECoG feature-feature |r| matrix (60×60).")

# %% [markdown]
# ## Fig 46: relevance column — feature-behaviour correlation
#
# `|r|` between each of the 60 ECoG features and the two behaviour targets
# (tracing velocity_x, acceleration magnitude). This is the **relevance**
# mRMR uses to pick feature 1 (pure argmax of the mean across columns). All
# subsequent picks combine this relevance with the redundancy penalty from
# Fig 45. Relevance values are small (~0.001-0.008) because ECoG-to-behaviour
# Pearson coupling is weak — PSID leverages dynamics, not static correlation.

# %%
def _plot_relevance(tri, fig_letter: str):
    base = _session_spectra_dir(tri.label)
    corr = pd.read_parquet(f"{base}/correlation/ecog.parquet").set_index("index")
    ecog_feats = [c for c in corr.columns if _ECOG_RX.match(c)]
    fb = corr.abs().loc[ecog_feats, _BEHAVIOR].to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(2.2, 3.8))
    im = ax.imshow(fb, cmap="Blues", vmin=0.0, vmax=float(fb.max() * 1.1),
                   aspect="auto")
    for k in (15, 30, 45):
        ax.axhline(k - 0.5, color="black", linewidth=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["velocity_x", "|acceleration|"], rotation=0, fontsize=8)
    ax.set_yticks([7.5, 22.5, 37.5, 52.5])
    ax.set_yticklabels([f"ECoG {i}" for i in range(1, 5)])
    fig.colorbar(im, ax=ax, shrink=0.85, label=r"$|r|$ with behaviour")
    fig.savefig(str(OUT / f"fig_046{fig_letter}_relevance_{tri.label}.png"))
    plt.show()


for letter, tri in zip("abcd", ALL_TRIPLETS):
    _plot_relevance(tri, letter)
    print(f"Fig 46{letter}: {tri.label} feature-behaviour relevance |r|.")

# %% [markdown]
# ## Fig 39: DPAD training curves
#
# Training loss (MSE) and validation loss for all DPAD models across 4 sessions × 3 DBS conditions.
# Shows convergence behavior of the DPAD neural network models (e3000 max epochs, early stopping).

# %%


_DPAD_SESSIONS = [
    ("PDI1_S2", "dpad_behavioral_PDI1_2_nx_4_n2_e3000_top5_dbs_{cond}_200Hz_narrow_band"),
    ("PDI1_S4", "dpad_behavioral_PDI1_4_nx_25_n2_e3000_top5_dbs_{cond}_200Hz_narrow_band"),
    ("PDI4_S2", "dpad_behavioral_PDI4_2_nx_15_n2_e3000_top5_dbs_{cond}_200Hz_narrow_band"),
    ("PDI4_S3", "dpad_behavioral_PDI4_3_nx_25_n2_e3000_top5_dbs_{cond}_200Hz_narrow_band"),
]
_DPAD_CONDS = ["both", "off", "on"]
_COND_COLORS = {"both": COLOR_DPAD, "off": COLOR_DBS_OFF, "on": COLOR_DBS_ON}
_DPAD_PLOT_STAGES = ["model1_Cy", "model2"]
# Theoretical names from the DPAD paper (Sani et al. 2024).
_STAGE_TITLES = {
    "model1_Cy": "Neural encoder (model1_Cy)",
    "model2": "Total dynamics model (model2)",
}

fig, axes = plt.subplots(4, 2, figsize=(8.5, 9.0), sharex=True)
_panel_letters = ['A', 'B', 'C', 'D']

for ri, (label, pattern) in enumerate(_DPAD_SESSIONS):
    for ci, stage in enumerate(_DPAD_PLOT_STAGES):
        ax = axes[ri, ci]
        for cond in _DPAD_CONDS:
            variant = pattern.format(cond=cond)
            hist_path = results_root / variant / 'training_history.json'
            if not hist_path.exists():
                continue
            with open(hist_path) as f:
                hist = json.load(f)
            sdata = hist.get(stage, {})
            epochs = sdata.get('epochs', [])
            train_loss = sdata.get('loss', [])
            val_loss = sdata.get('val_loss', [])
            if not epochs:
                continue
            col = _COND_COLORS[cond]
            ax.plot(epochs, train_loss, color=col, linewidth=1.2,
                    label=f"{cond} train" if (ri == 0 and ci == 0) else None)
            ax.plot(epochs, val_loss, color=col, linewidth=1.0, linestyle=':',
                    label=f"{cond} val" if (ri == 0 and ci == 0) else None)
        if ri == 0:
            ax.set_title(_STAGE_TITLES[stage])
        if ci == 0:
            ax.set_ylabel("Loss (MSE)")
            panel_label(ax, _panel_letters[ri], label)
        if ri == len(_DPAD_SESSIONS) - 1:
            ax.set_xlabel("epoch")

fig.legend()
fig.savefig(str(OUT / 'fig_039_dpad_training_curves.png'))
plt.show()
print(
    'Fig 39: DPAD training curves over 4 sessions x 3 DBS conditions.\n'
    "  Left column (model1_Cy): the 'neural encoder' RNN that maps Y -> behaviorally-relevant latent X1.\n"
    "  Right column (model2): the 'total dynamics' model that adds non-behaviorally-relevant latents (X2).\n"
    '  Solid = train loss, dotted = val loss. model1 and model2_Cz stages omitted (near-flat, <1% change).'
)
