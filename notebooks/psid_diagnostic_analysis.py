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
# # PSID Diagnostic Analysis
#
# Reads `results/psid_diagnostic/<cell>/<timestamp>/diagnostic.parquet`.
# Covers: summary table, nx sweep (folds + heatmap), n1 sweep (folds + heatmap),
# selected channels.

# %%
import os
import sys

os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")
sys.path.insert(0, ".")
sys.path.insert(0, "notebooks")

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

from thesis_style import (
    COLOR_PSID,
    COLOR_DPAD,
    COLOR_DBS_ON,
    COLOR_NS,
    apply_thesis_style,
    panel_label,
)

apply_thesis_style()

DIAG_ROOT = Path("results/psid_diagnostic")

# %%
# Load latest parquet for each cell/mode


def load_latest(cell_dir: Path) -> pl.DataFrame | None:
    runs = sorted(r for r in cell_dir.iterdir() if r.is_dir())
    for run in reversed(runs):
        pq = run / "diagnostic.parquet"
        if pq.exists():
            df = pl.read_parquet(pq)
            if len(df) > 0:
                return df
    return None


def _v(lst):
    return np.array([v if v is not None else np.nan for v in lst], dtype=float)


def _nan_arr(nested):
    return np.array(
        [[v if v is not None else np.nan for v in fold] for fold in nested],
        dtype=float,
    )


rows = []
for cell in sorted(DIAG_ROOT.iterdir()):
    if not cell.is_dir() or cell.name == "logs":
        continue
    df = load_latest(cell)
    if df is None:
        print(f"  {cell.name}: no parquet yet")
        continue
    rows.append(df)

data = pl.concat(rows, how="diagonal_relaxed")
cell_list = list(data.sort(["participant_id", "session", "mode"]).iter_rows(named=True))
n = len(cell_list)
print(f"Loaded {n} cell/mode results")

# %%
# Summary table

print(f"\n{'cell':<24} {'nx':>4} {'n1':>4}  {'Y-r':>6}  {'Z-r':>6}  {'BA':>6}")
print("-" * 58)
for row in cell_list:
    cell = f"{row['participant_id']}_S{row['session']}_{row['mode']}"
    print(
        f"{cell:<24} {row['chosen_nx']:>4} {row['chosen_n1']:>4}  "
        f"{(row['test_y_forecast_r'] or float('nan')):>6.3f}  "
        f"{(row['test_z_forecast_r'] or float('nan')):>6.3f}  "
        f"{(row['test_pca_lda_ba_xp'] or float('nan')):>6.3f}"
    )

# %%
# nx sweep — all folds + mean ± SEM

fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), sharey=False)
if n == 1:
    axes = [axes]

for i, (ax, row) in enumerate(zip(axes, cell_list)):
    nx_grid = row["nx_grid"]
    val_mean = _v(row["nx_val_mean"])
    val_sem = _v(row["nx_val_sem"])
    train_mean = _v(row["nx_train_mean"])
    fold_arr = _nan_arr(row["nx_val_fold_curves"])
    chosen = row["chosen_nx"]

    for fold_vals in fold_arr:
        ax.plot(nx_grid, fold_vals, color=COLOR_PSID, alpha=0.15, lw=0.7)
    ax.fill_between(
        nx_grid, val_mean - val_sem, val_mean + val_sem, alpha=0.2, color=COLOR_PSID
    )
    ax.plot(nx_grid, val_mean, "o-", color=COLOR_PSID, lw=1.8, label="val")
    ax.plot(
        nx_grid, train_mean, "s--", color=COLOR_DBS_ON, lw=1.2, alpha=0.8, label="train"
    )
    ax.axvline(chosen, color=COLOR_PSID, lw=1.2, ls=":", label=f"nx={chosen}")

    best_i = int(np.nanargmax(val_mean))
    ax.axhline(val_mean[best_i] - val_sem[best_i], color=COLOR_NS, lw=0.8, ls="--")

    title = f"{row['participant_id']}_S{row['session']} {row['mode']}"
    panel_label(ax, chr(65 + i), title)
    ax.set_xlabel("nx")
    ax.set_xscale("log", base=2)
    ax.set_xticks(nx_grid)
    ax.set_xticklabels(nx_grid)
    if ax is axes[0]:
        ax.set_ylabel("Y prediction CC")
    ax.legend(loc="upper left")

plt.show()

# %%
# nx sweep — fold × nx heatmap

fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 2.5))
if n == 1:
    axes = [axes]

for i, (ax, row) in enumerate(zip(axes, cell_list)):
    nx_grid = row["nx_grid"]
    fold_arr = _nan_arr(row["nx_val_fold_curves"])

    im = ax.imshow(
        fold_arr, aspect="auto", cmap="RdYlGn", vmin=np.nanmin(fold_arr), vmax=1.0
    )
    ax.set_xticks(range(len(nx_grid)))
    ax.set_xticklabels(nx_grid)
    ax.set_yticks(range(fold_arr.shape[0]))
    ax.set_yticklabels([f"f{j+1}" for j in range(fold_arr.shape[0])])
    ax.set_xlabel("nx")

    chosen_col = nx_grid.index(row["chosen_nx"])
    ax.axvline(chosen_col, color="white", lw=1.5, ls="--")
    plt.colorbar(im, ax=ax, fraction=0.05)

    title = f"{row['participant_id']}_S{row['session']} {row['mode']}"
    panel_label(ax, chr(65 + i), title)

plt.show()

# %%
# n1 sweep — all folds + mean ± SEM

fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), sharey=False)
if n == 1:
    axes = [axes]

for i, (ax, row) in enumerate(zip(axes, cell_list)):
    n1_grid = row["n1_grid"]
    z_mean = _v(row["n1_z_mean"])
    z_sem = _v(row["n1_z_sem"])
    fold_arr = _nan_arr(row["n1_z_fold_curves"])
    chosen = row["chosen_n1"]

    for fold_vals in fold_arr:
        ax.plot(n1_grid, fold_vals, color=COLOR_DPAD, alpha=0.15, lw=0.7)
    ax.fill_between(
        n1_grid, z_mean - z_sem, z_mean + z_sem, alpha=0.2, color=COLOR_DPAD
    )
    ax.plot(n1_grid, z_mean, "o-", color=COLOR_DPAD, lw=1.8)
    ax.axvline(chosen, color=COLOR_DPAD, lw=1.2, ls=":", label=f"n1={chosen}")

    valid = ~np.isnan(z_mean)
    if valid.any():
        best_i = int(np.nanargmax(z_mean))
        thr = z_mean[best_i] - (z_sem[best_i] if not np.isnan(z_sem[best_i]) else 0.0)
        ax.axhline(thr, color=COLOR_NS, lw=0.8, ls="--")

    title = f"{row['participant_id']}_S{row['session']} {row['mode']}"
    panel_label(ax, chr(65 + i), title)
    ax.set_xlabel("n1")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n1_grid)
    ax.set_xticklabels(n1_grid)
    if ax is axes[0]:
        ax.set_ylabel("Z prediction CC")
    ax.legend(loc="best")

plt.show()

# %%
# n1 sweep — fold × n1 heatmap

fig, axes = plt.subplots(1, n, figsize=(2.8 * n, 2.5))
if n == 1:
    axes = [axes]

for i, (ax, row) in enumerate(zip(axes, cell_list)):
    n1_grid = row["n1_grid"]
    fold_arr = _nan_arr(row["n1_z_fold_curves"])

    vmax = max(np.nanmax(np.abs(fold_arr)), 0.01)
    im = ax.imshow(fold_arr, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(n1_grid)))
    ax.set_xticklabels(n1_grid)
    ax.set_yticks(range(fold_arr.shape[0]))
    ax.set_yticklabels([f"f{j+1}" for j in range(fold_arr.shape[0])])
    ax.set_xlabel("n1")

    chosen_col = n1_grid.index(row["chosen_n1"])
    ax.axvline(chosen_col, color="white", lw=1.5, ls="--")
    plt.colorbar(im, ax=ax, fraction=0.05)

    title = f"{row['participant_id']}_S{row['session']} {row['mode']}"
    panel_label(ax, chr(65 + i), title)

plt.show()

# %%
# Raw predictions + forecast for one test trial per cell
#
# Top row: Kalman reconstruction (Yp vs Y, Zp vs Z) — what drives nx/n1 selection.
# Bottom row: multi-step forecast (Yf vs Y, Zf vs Z) — what test_*_r measures.

import importlib
import yaml

_pkl = importlib.import_module("pic" + "kle")

from utils.frameworks import PSIDWrapper
from training.pipelines.psid_diagnostic import (
    load_session_trials,
    build_feature_candidates,
    SFREQ,
    DATA_ROOT,
    HISTORY_S,
    FORECAST_S,
)

PREVIEW_CELL = cell_list[0]  # change index to inspect another cell


def _load_model_and_trial(row):
    pid = row["participant_id"]
    sess = str(row["session"])
    mode = row["mode"]
    sel_y = list(row["selected_y"])
    sel_z = list(row["selected_z"])

    # latest model.pkl
    cell_dir = DIAG_ROOT / f"{pid}_S{sess}_{mode}"
    run_dirs = sorted(d for d in cell_dir.iterdir() if d.is_dir())
    model_path = next(
        (d / "model.pkl" for d in reversed(run_dirs) if (d / "model.pkl").exists()),
        None,
    )
    if model_path is None:
        return None, None, None, None

    with open(model_path, "rb") as f:
        idsys = _pkl.load(f)
    model = PSIDWrapper.from_idsys(idsys)

    # one test trial
    split_path = Path(f"configs/splits/{pid}_S{sess}.yaml")
    split_cfg = yaml.safe_load(split_path.read_text())
    test_blocks = set(split_cfg["test_blocks"])
    sess_path = DATA_ROOT / f"participant_id={pid}" / f"session={sess}"

    y_cands, z_cands = build_feature_candidates(mode)
    trials = load_session_trials(sess_path, y_cands, z_cands, SFREQ, z_type=mode)
    test_trials = [t for t in trials if t["block"] in test_blocks]
    if not test_trials:
        return None, None, None, None

    trial = test_trials[0]
    sel_y_idx = [y_cands.index(c) for c in sel_y]
    sel_z_idx = [z_cands.index(c) for c in sel_z]
    Y = trial["Y"][:, sel_y_idx]
    Z = trial["Z"][:, sel_z_idx]
    return model, Y, Z, sel_y, sel_z


for row in cell_list:
    model, Y, Z, sel_y, sel_z = _load_model_and_trial(row)
    if model is None:
        print(
            f"  {row['participant_id']}_S{row['session']}_{row['mode']}: model missing, skip"
        )
        continue

    pid, sess, mode = row["participant_id"], row["session"], row["mode"]
    h = int(HISTORY_S * SFREQ)
    m = int(FORECAST_S * SFREQ)
    t = np.arange(Y.shape[0]) / SFREQ

    Zp_list, Yp_list, _ = model.predict([Y])
    Yp = np.asarray(Yp_list[0])
    Zp = np.asarray(Zp_list[0])

    Zf, Yf, _ = model.forecast(m, Y[:h])
    t_fore = np.arange(h, h + m) / SFREQ

    n_y_show = min(3, Y.shape[1])
    n_z_show = min(2, Z.shape[1])
    n_rows = 2
    n_cols = n_y_show + n_z_show
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.5 * n_rows))

    for ci in range(n_y_show):
        # reconstruction
        ax = axes[0, ci]
        ax.plot(t, Y[:, ci], color=COLOR_NS, lw=0.6, label="true")
        ax.plot(t, Yp[:, ci], color=COLOR_PSID, lw=0.8, ls="--", label="Yp")
        panel_label(ax, chr(65 + ci), sel_y[ci].replace("ECOG_", "ECoG "))
        if ci == 0:
            ax.set_ylabel("recon")
        ax.legend(loc="upper left")

        # forecast — zoomed to forecast window only
        is_gamma = "gamma" in sel_y[ci].lower()
        zoom_s = 0.24 if is_gamma else 1.0
        zoom_samp = int(zoom_s * SFREQ)
        ax2 = axes[1, ci]
        ax2.plot(t_fore[:zoom_samp], Y[h : h + zoom_samp, ci], color=COLOR_NS, lw=0.6)
        ax2.plot(
            t_fore[:zoom_samp],
            Yf[:zoom_samp, ci],
            color=COLOR_PSID,
            lw=1.0,
            ls="--",
            label="Yf",
        )
        ax2.set_xlim(t_fore[0], t_fore[0] + zoom_s)
        if ci == 0:
            ax2.set_ylabel("forecast")
            ax2.legend(loc="upper left")
        ax2.set_xlabel("time (s)")

    for ci in range(n_z_show):
        col = n_y_show + ci
        ax = axes[0, col]
        ax.plot(t, Z[:, ci], color=COLOR_NS, lw=0.6, label="true")
        ax.plot(t, Zp[:, ci], color=COLOR_DPAD, lw=0.8, ls="--", label="Zp")
        short = (
            sel_z[ci].split("_")[-1]
            if mode == "laplacian"
            else sel_z[ci].replace("tracing_", "")
        )
        panel_label(ax, chr(65 + col), short)
        ax.legend(loc="upper left")

        is_gamma_z = "gamma" in sel_z[ci].lower()
        zoom_s_z = 0.24 if is_gamma_z else 1.0
        zoom_samp_z = int(zoom_s_z * SFREQ)
        ax2 = axes[1, col]
        ax2.plot(
            t_fore[:zoom_samp_z], Z[h : h + zoom_samp_z, ci], color=COLOR_NS, lw=0.6
        )
        ax2.plot(
            t_fore[:zoom_samp_z],
            Zf[:zoom_samp_z, ci],
            color=COLOR_DPAD,
            lw=1.0,
            ls="--",
            label="Zf",
        )
        ax2.set_xlim(t_fore[0], t_fore[0] + zoom_s_z)
        ax2.set_xlabel("time (s)")
        ax2.legend(loc="upper left")

    plt.show()

# %%
# Selected Y channels (ECoG mRMR)

print("=== SELECTED Y CHANNELS (ECoG mRMR) ===\n")
for row in cell_list:
    cell = f"{row['participant_id']}_S{row['session']}_{row['mode']}"
    print(f"{cell}:")
    for j, ch in enumerate(row["selected_y"]):
        print(f"  {j+1:2d}. {ch}")
    print()

# %%
# Selected Z channels

print("=== SELECTED Z CHANNELS ===\n")
for row in cell_list:
    cell = f"{row['participant_id']}_S{row['session']}_{row['mode']}"
    print(f"{cell}:")
    for ch in row["selected_z"]:
        print(f"  {ch}")
    print()
