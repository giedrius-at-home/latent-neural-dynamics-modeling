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
# # Sec 2c — Z reconstruction (one-step) group metrics
#
# * Fig 70 — per-cell × model × DBS two-metric box (Pearson r top, RMSE bottom). Laplacian first; trio extends to neural Y + behavior.

# %%
import sys, os

os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")
sys.path.insert(0, ".")
sys.path.insert(0, "notebooks")

import numpy as np
import matplotlib.pyplot as plt

from thesis_style import (
    apply_thesis_style,
    panel_label,
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_PSID,
    COLOR_DPAD,
    COLOR_VARMA,
)
from thesis_sec2_common import (
    ALL_TRIPLETS,
    ALL_TRIPLETS_LAP,
    OUT,
    results_root,
    mpl_per_cell_yz_box,
    mpl_raincloud_dbs_pair_vert,
    mpl_raincloud_yz_pair_vert,
    pool_dbs_cells,
    pool_yz_dbs_cells,
)

from thesis_loaders import load_split_results_required
from thesis_utils import (
    normalize_stim,
    trial_metric_y_for_model,
    trial_metric_z_for_model,
)

apply_thesis_style()

# %% [markdown]
# ## Fig 70 — Per-cell Y + Z two-metric box (laplacian mode)
#
# Per-cell × per-model × DBS, both reconstructions:
# * Rows 1-2: Y self-recon (top-8 ECOG mRMR) — Pearson r, NRMSE
# * Rows 3-4: Z decoding (top-8 LFP laplacian) — Pearson r, NRMSE
#
# 4 rows × 4 cols (cells: PDI1_S2, PDI1_S4, PDI4_S2, PDI4_S3). Each subplot:
# x = PSID/DPAD/VARMA, hue = DBS-OFF (blue) / DBS-ON (red). Per-trial values
# averaged across channels. NRMSE = RMSE in z-score units of true target
# (scale-free across cells/models).
#
# Mode label: ECOG → laplacian LFP. No pooling — sec1/sec2a per-session style.


# %%
def _collect_per_cell_metric(triplets, target="Z", metric="pearson", split="test"):
    """dict[(cell_label, model, dbs)] -> 1-D np.ndarray trial means across chans."""
    score_fn = trial_metric_z_for_model if target == "Z" else trial_metric_y_for_model
    out: dict = {}
    for tri in triplets:
        for model_name, variant, run_ts in (
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ):
            if not variant or not run_ts:
                continue
            try:
                res = load_split_results_required(results_root, variant, run_ts, split)
            except Exception:
                continue
            arr = res.get(target, [])
            if len(arr) == 0:
                continue
            ref = np.asarray(arr[0])
            n_chans = ref.shape[-1] if ref.ndim >= 1 else 1
            off_vals, on_vals = [], []
            for i in range(len(arr)):
                stim = normalize_stim(res["stim"][i])
                if stim is None:
                    continue
                ch_vals = []
                for ch in range(n_chans):
                    try:
                        v = score_fn(res, i, ch, metric)
                    except (ValueError, IndexError, KeyError):
                        v = float("nan")
                    if np.isfinite(v):
                        ch_vals.append(v)
                if not ch_vals:
                    continue
                (off_vals if stim == "off" else on_vals).append(float(np.mean(ch_vals)))
            out[(tri.label, model_name, "off")] = np.array(off_vals, dtype=float)
            out[(tri.label, model_name, "on")] = np.array(on_vals, dtype=float)
    return out


_lap_cells = [t.label for t in ALL_TRIPLETS_LAP]
_lap_y_r = _collect_per_cell_metric(ALL_TRIPLETS_LAP, target="Y", metric="pearson")
_lap_y_n = _collect_per_cell_metric(ALL_TRIPLETS_LAP, target="Y", metric="rmse")
_lap_z_r = _collect_per_cell_metric(ALL_TRIPLETS_LAP, target="Z", metric="pearson")
_lap_z_n = _collect_per_cell_metric(ALL_TRIPLETS_LAP, target="Z", metric="rmse")

fig = mpl_per_cell_yz_box(
    _lap_y_r,
    _lap_y_n,
    _lap_z_r,
    _lap_z_n,
    _lap_cells,
    mode_label="",
)
fig.savefig(str(OUT / "fig_070_lap_per_cell_yz_box.png"), bbox_inches="tight")
plt.show()
print(
    "Fig 70 (laplacian): 4 rows × 4 cells. Y self-recon (top-8 ECOG) + "
    "Z decoding (top-8 LFP). Pearson r + NRMSE per target."
)


# %% [markdown]
# ## Fig 71 — Zoom-out: pooled raincloud, Y self-recon (laplacian mode)
#
# Pool trials across 4 cells per (model × DBS) group. Half-violin (distribution
# shape) + strip dots (per-trial values) + box (median + IQR). 1 row × 2 cols:
# Pearson r left, NRMSE right (log). 6 rainclouds per panel (3 models × DBS).
#
# Pooling justified: Pearson r and NRMSE are already cell-scale-comparable
# (NRMSE = z-score units of true target). VARMA kept as baseline.

# %%
_lap_y_r_pool = pool_dbs_cells(_lap_y_r, _lap_cells)
_lap_y_n_pool = pool_dbs_cells(_lap_y_n, _lap_cells)

fig = mpl_raincloud_dbs_pair_vert(_lap_y_r_pool, _lap_y_n_pool)
fig.savefig(str(OUT / "fig_071_lap_pool_raincloud_y.png"), bbox_inches="tight")
plt.show()
print(
    "Fig 71 (Y self-recon, pooled across 4 cells): horizontal raincloud per model. "
    "DBS-OFF/ON overlap. Dashed reference: r=0 (left), NRMSE=1 (right)."
)


# %% [markdown]
# ## Fig 72 — Y vs Z reconstruction per model (laplacian, pooled DBS + cells)
#
# Same horizontal raincloud as Fig 71 but the within-row dimension is
# **target** (Y self-recon vs Z decoding) instead of DBS. Each row colored
# by framework (PSID blue / DPAD rust / VARMA green). Y = alpha 0.35
# (translucent), Z = alpha 0.75 (full). DBS-OFF/ON pooled into single
# distribution per (model, target).
#
# Question this answers: per model, how does latent → Y regression compare
# to latent → Z regression? Both heads share the same latent state; this
# isolates the regression-quality difference.

# %%
_lap_yz_r_pool = pool_yz_dbs_cells(_lap_y_r, _lap_z_r, _lap_cells)
_lap_yz_n_pool = pool_yz_dbs_cells(_lap_y_n, _lap_z_n, _lap_cells)

fig = mpl_raincloud_yz_pair_vert(_lap_yz_r_pool, _lap_yz_n_pool)
fig.savefig(str(OUT / "fig_072_lap_pool_raincloud_yz.png"), bbox_inches="tight")
plt.show()
print(
    "Fig 72 (Y vs Z, pooled across 4 cells × 2 DBS states): horizontal "
    "raincloud per model. Row color = framework; Y = lighter, Z = fuller."
)


# %% [markdown]
# ## Behavioral mode — group figures (mirror of 70/71/72)
#
# Same templates, swap `ALL_TRIPLETS_LAP` → `ALL_TRIPLETS`. Y = top-8 ECoG
# self-recon, Z = 2 behavioral targets (tracing_velocity_x, tracing_acc_mag).

# %%
_beh_cells = [t.label for t in ALL_TRIPLETS]
_beh_y_r = _collect_per_cell_metric(ALL_TRIPLETS, target="Y", metric="pearson")
_beh_y_n = _collect_per_cell_metric(ALL_TRIPLETS, target="Y", metric="rmse")
_beh_z_r = _collect_per_cell_metric(ALL_TRIPLETS, target="Z", metric="pearson")
_beh_z_n = _collect_per_cell_metric(ALL_TRIPLETS, target="Z", metric="rmse")


# %% [markdown]
# ## Fig 73 — Behavioral per-cell box (Y self-recon + Z behavior decoding)

# %%
fig = mpl_per_cell_yz_box(
    _beh_y_r,
    _beh_y_n,
    _beh_z_r,
    _beh_z_n,
    _beh_cells,
    mode_label="",
)
fig.savefig(str(OUT / "fig_073_beh_per_cell_yz_box.png"), bbox_inches="tight")
plt.show()
print(
    "Fig 73 (behavioral): 4 rows × 4 cells. Y self-recon (top-8 ECoG) + "
    "Z decoding (behavioral vars). Pearson r + NRMSE per target."
)


# %% [markdown]
# ## Fig 74 — Behavioral pooled raincloud, Y self-recon (DBS-OFF/ON)

# %%
_beh_y_r_pool = pool_dbs_cells(_beh_y_r, _beh_cells)
_beh_y_n_pool = pool_dbs_cells(_beh_y_n, _beh_cells)

fig = mpl_raincloud_dbs_pair_vert(_beh_y_r_pool, _beh_y_n_pool)
fig.savefig(str(OUT / "fig_074_beh_pool_raincloud_y.png"), bbox_inches="tight")
plt.show()
print(
    "Fig 74 (Y self-recon, behavioral, pooled across 4 cells): horizontal "
    "raincloud per model. DBS-OFF/ON overlap. Refs: r=0, NRMSE=1."
)


# %% [markdown]
# ## Fig 75 — Behavioral Y vs Z reconstruction per model (pooled DBS + cells)

# %%
_beh_yz_r_pool = pool_yz_dbs_cells(_beh_y_r, _beh_z_r, _beh_cells)
_beh_yz_n_pool = pool_yz_dbs_cells(_beh_y_n, _beh_z_n, _beh_cells)

fig = mpl_raincloud_yz_pair_vert(_beh_yz_r_pool, _beh_yz_n_pool)
fig.savefig(str(OUT / "fig_075_beh_pool_raincloud_yz.png"), bbox_inches="tight")
plt.show()
print(
    "Fig 75 (Y vs Z, behavioral, pooled across 4 cells × 2 DBS states): "
    "horizontal raincloud per model. Y = top-8 ECoG; Z = behavior vars."
)
