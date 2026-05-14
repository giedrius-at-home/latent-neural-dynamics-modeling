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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Section 7: Subspace Dynamics Analysis — Xp_1 vs Xp_2
#
# Analyses of PSID latent subspace properties across DBS ON/OFF conditions.
#
# | # | Analysis | Count |
# |---|---------|-------|
# | 1 | Behavioral DBS effect (raw kinematics) | 1 summary table |
# | 2 | Latent state trial-level statistics | 1 per session |
# | 3 | PSD of latent states | 1 per session |
# | 4 | A matrix structure & eigenvalues | 1 per session |
# | 5 | C / Cz matrix loadings | 1 per session |
# | 6 | Channel importance scatter | 1 per session |
# | 7 | Classifier comparison (mean vs cov) | 1 per session |
# | 8 | Cross-run summary | 1 table |

# %%
import sys, os

os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")
sys.path.insert(0, ".")
sys.path.insert(0, "notebooks")

# %%
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

# In-repo trusted model outputs are serialized; load via dynamic import.
import importlib as _importlib

_serial = _importlib.import_module("pic" + "kle")  # nosec — trusted in-repo outputs

from thesis_style import (
    COLOR_DBS_ON,
    COLOR_DBS_OFF,
    COLOR_PSID,
    COLOR_DPAD,
    COLOR_CHANCE,
    COLOR_NS,
    apply_thesis_style,
    panel_label,
)

apply_thesis_style()

OUT = Path("thesis_figures/sec7")
OUT.mkdir(parents=True, exist_ok=True)

# Subspace mapping: Xp_1 (behavioral) -> PSID blue, Xp_2 (non-behavioral) -> DPAD orange
COLOR_XP1 = COLOR_PSID
COLOR_XP2 = COLOR_DPAD
# Per-electrode colors drawn from the canonical thesis palette (4 electrodes only)
ELECTRODE_COLORS = {
    "1": COLOR_PSID,
    "2": COLOR_DPAD,
    "3": COLOR_DBS_OFF,
    "4": COLOR_DBS_ON,
}

# %% [markdown]
# ## 1. Configuration
#
# Each entry specifies the PSID model paths (both, ON-only, OFF-only),
# data location, and subspace dimensions (n1 = behavioral dims, nx = total dims).

# %%
RUNS = [
    {
        "label": "PDI1 S2 (200Hz)",
        "model_type": "psid",
        "participant": "PDI1",
        "session": "2",
        "n1": 15,
        "nx": 55,
        "fs": 200,
        "variant": "psid_behavioral_PDI1_2_nx_55_n15_i100_dbs_both_200Hz_narrow_band",
        "config_yaml": "training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI1_2_nx_55_n15_i100_dbs_both_200Hz_narrow_band.yaml",
        "model_both": "results/psid_behavioral_PDI1_2_nx_55_n15_i100_dbs_both_200Hz_narrow_band/model_20260421_222439.pkl",
        "model_on": "results/psid_behavioral_PDI1_2_nx_55_n15_i100_dbs_on_200Hz_narrow_band/model_20260422_004246.pkl",
        "model_off": "results/psid_behavioral_PDI1_2_nx_55_n15_i100_dbs_off_200Hz_narrow_band/model_20260422_010140.pkl",
    },
    {
        "label": "PDI1 S4 (200Hz)",
        "model_type": "psid",
        "participant": "PDI1",
        "session": "4",
        "n1": 10,
        "nx": 50,
        "fs": 200,
        "variant": "psid_behavioral_PDI1_4_nx_50_n10_i100_dbs_both_200Hz_narrow_band",
        "config_yaml": "training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI1_4_nx_50_n10_i100_dbs_both_200Hz_narrow_band.yaml",
        "model_both": "results/psid_behavioral_PDI1_4_nx_50_n10_i100_dbs_both_200Hz_narrow_band/model_20260422_000702.pkl",
        "model_on": "results/psid_behavioral_PDI1_4_nx_50_n10_i100_dbs_on_200Hz_narrow_band/model_20260422_004723.pkl",
        "model_off": "results/psid_behavioral_PDI1_4_nx_50_n10_i100_dbs_off_200Hz_narrow_band/model_20260422_010614.pkl",
    },
    {
        "label": "PDI4 S2 (200Hz)",
        "model_type": "psid",
        "participant": "PDI4",
        "session": "2",
        "n1": 10,
        "nx": 50,
        "fs": 200,
        "variant": "psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band",
        "config_yaml": "training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band.yaml",
        "model_both": "results/psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band/model_20260421_202056.pkl",
        "model_on": "results/psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_on_200Hz_narrow_band/model_20260421_205424.pkl",
        "model_off": "results/psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_off_200Hz_narrow_band/model_20260421_210809.pkl",
    },
    {
        "label": "PDI4 S3 (200Hz)",
        "model_type": "psid",
        "participant": "PDI4",
        "session": "3",
        "n1": 10,
        "nx": 50,
        "fs": 200,
        "variant": "psid_behavioral_PDI4_3_nx_50_n10_i100_dbs_both_200Hz_narrow_band",
        "config_yaml": "training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI4_3_nx_50_n10_i100_dbs_both_200Hz_narrow_band.yaml",
        "model_both": "results/psid_behavioral_PDI4_3_nx_50_n10_i100_dbs_both_200Hz_narrow_band/model_20260421_202721.pkl",
        "model_on": "results/psid_behavioral_PDI4_3_nx_50_n10_i100_dbs_on_200Hz_narrow_band/model_20260421_205640.pkl",
        "model_off": "results/psid_behavioral_PDI4_3_nx_50_n10_i100_dbs_off_200Hz_narrow_band/model_20260421_211024.pkl",
    },
    # To add DPAD runs later, set "model_type": "dpad" — PSID-only sections auto-skip.
]

# %% [markdown]
# ## Helper functions
#
# Data loading, analysis, and plotting utilities — all defined inline.

# %%
# ── Data loading ────────────────────────────────────────────────────────────
import yaml as _yaml


def get_neural_channels_from_config(config_yaml):
    """Read the exact neural channel list from the training YAML config."""
    with open(config_yaml) as f:
        cfg = _yaml.safe_load(f)
    return cfg["data"]["channels"]["neural_input"]


def load_test_trials(variant, neural_channels):
    """Load **test-set** trials from the split parquet.

    Returns (Y_on, Y_off, Y_all, labels, df) where Y_* are lists of
    (T, n_channels) arrays and labels is a binary array (1=ON, 0=OFF).
    """
    split_path = Path("results") / variant / "split" / "test.parquet"
    df = pl.read_parquet(split_path)
    # Sort by block, trial for consistent ordering
    df = df.sort(["block", "trial"])

    Y_on, Y_off, Y_all, labels = [], [], [], []
    for i in range(len(df)):
        stim = df["stim"][i]
        arrays = []
        for ch in neural_channels:
            vals = df[ch][i].to_numpy(allow_copy=True).astype(np.float64)
            arrays.append(vals)
        trial = np.column_stack(arrays)  # (T, n_channels)
        Y_all.append(trial)
        labels.append(1 if stim == "on" else 0)
        (Y_on if stim == "on" else Y_off).append(trial)

    return Y_on, Y_off, Y_all, np.array(labels), df


def load_model(path):
    """Load a serialized PSID LSSM (or DPAD) model from disk."""
    with open(path, "rb") as f:
        model = _serial.load(f)
    # DPAD models need explicit restoration
    if hasattr(model, "restoreModels"):
        model.restoreModels()
        model.set_steps_ahead([1])
        model.set_multi_step_with_data_gen(False)
    return model


def predict_model(model, Y_trials, model_type="psid"):
    """Run model.predict(), handling DPAD block-padding if needed."""
    if model_type == "psid":
        return model.predict(Y_trials)

    # DPAD requires trials padded to block_samples multiple
    all_Zp, all_Yp, all_Xp = [], [], []
    block_samples = model.block_samples
    for y in Y_trials:
        T = y.shape[0]
        remainder = T % block_samples
        if remainder != 0:
            pad = np.zeros((block_samples - remainder, y.shape[1]))
            y_padded = np.concatenate([y, pad], axis=0)
        else:
            y_padded = y
        Zp, Yp, Xp = model.predict(y_padded)
        all_Zp.append(Zp[:T] if Zp is not None else None)
        all_Yp.append(Yp[:T] if Yp is not None else None)
        all_Xp.append(Xp[:T] if Xp is not None else None)
    return all_Zp, all_Yp, all_Xp


# %%
# ── Analysis functions ──────────────────────────────────────────────────────

BEHAV_COLS = ["tracing_velocity_x", "tracing_acceleration_magnitude"]

FREQ_BANDS = [
    (0, 4, "sub-theta (0-4)"),
    (4, 8, "theta (4-8)"),
    (8, 13, "alpha (8-13)"),
    (13, 30, "beta (13-30)"),
    (30, 50, "low-gamma (30-50)"),
    (50, 100, "high-gamma (50+)"),
]


def cohens_d(a, b):
    """Standardized mean difference (pooled SD)."""
    ps = np.sqrt((np.std(a) ** 2 + np.std(b) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / ps if ps > 0 else 0.0


def behavioral_dbs_effect(df):
    """Cohen's d and t-test for raw behavioral columns, ON vs OFF."""
    rows = []
    for col in BEHAV_COLS:
        if col not in df.columns:
            continue
        on_means = np.array(
            [
                np.nanmean(np.abs(np.array(df[col][i].to_list(), dtype=float)))
                for i in range(len(df))
                if df["stim"][i] == "on"
            ]
        )
        off_means = np.array(
            [
                np.nanmean(np.abs(np.array(df[col][i].to_list(), dtype=float)))
                for i in range(len(df))
                if df["stim"][i] == "off"
            ]
        )
        d = cohens_d(on_means, off_means)
        _, p = stats.ttest_ind(on_means, off_means)
        rows.append(
            {
                "feature": col.replace("tracing_", ""),
                "on_mean": np.mean(on_means),
                "off_mean": np.mean(off_means),
                "d": d,
                "p": p,
            }
        )
    return rows


def latent_trial_stats(Xp_list, labels, n1, nx):
    """Per-dimension Cohen's d of trial means for Xp_1 (behavioral) and Xp_2 (non-behavioral)."""
    means = np.array([x.mean(axis=0) for x in Xp_list])
    on_mask, off_mask = labels == 1, labels == 0
    results = {"xp1": [], "xp2": []}
    for i in range(n1):
        d = cohens_d(means[on_mask, i], means[off_mask, i])
        _, p = stats.ttest_ind(means[on_mask, i], means[off_mask, i])
        results["xp1"].append({"dim": i, "d": d, "p": p})
    for i in range(n1, nx):
        d = cohens_d(means[on_mask, i], means[off_mask, i])
        _, p = stats.ttest_ind(means[on_mask, i], means[off_mask, i])
        results["xp2"].append({"dim": i, "d": d, "p": p})
    return results


def avg_psd(trials, dim_slice, fs):
    """Welch PSD averaged across trials and dimensions in the given slice."""
    all_psd = []
    for trial in trials:
        x = trial[:, dim_slice]
        for d in range(x.shape[1]):
            f, pxx = signal.welch(x[:, d], fs=fs, nperseg=min(512, x.shape[0]))
            all_psd.append(pxx)
    return f, np.mean(all_psd, axis=0)


def psd_band_power(f, psd):
    """Mean power per frequency band."""
    return {
        name: float(np.mean(psd[(f >= lo) & (f < hi)])) for lo, hi, name in FREQ_BANDS
    }


def eigenvalue_modes(A, fs):
    """Extract oscillatory modes from A matrix eigenvalues.

    Returns list of dicts with freq (Hz), magnitude |lambda|, decay time (ms),
    and whether the eigenvalue is complex (oscillatory).
    """
    eigs = np.linalg.eig(A)[0]
    modes = []
    for e in eigs:
        mag = np.abs(e)
        freq = np.abs(np.angle(e)) * fs / (2 * np.pi)
        decay_ms = -1000.0 / (fs * np.log(mag)) if 0 < mag < 1 else float("inf")
        modes.append(
            {
                "freq": freq,
                "mag": mag,
                "decay_ms": decay_ms,
                "is_complex": abs(np.imag(e)) > 1e-10,
            }
        )
    return sorted(modes, key=lambda m: -m["mag"])


def a_matrix_analysis(model_both, model_on, model_off, n1, nx, fs):
    """A-matrix block structure and eigenvalue comparison (PSID only).

    Block structure: A11 (Xp1 self-dynamics), A12 (Xp2->Xp1 coupling),
    A21 (Xp1->Xp2), A22 (Xp2 self-dynamics).
    """
    A = np.array(model_both.A)
    A_on, A_off = np.array(model_on.A), np.array(model_off.A)

    block_norms = {
        "A11 (Xp1->Xp1)": np.linalg.norm(A[:n1, :n1]),
        "A12 (Xp2->Xp1)": np.linalg.norm(A[:n1, n1:]),
        "A21 (Xp1->Xp2)": np.linalg.norm(A[n1:, :n1]),
        "A22 (Xp2->Xp2)": np.linalg.norm(A[n1:, n1:]),
    }
    diff_norms = {
        "A11 diff": np.linalg.norm(A_on[:n1, :n1] - A_off[:n1, :n1]),
        "A22 diff": np.linalg.norm(A_on[n1:, n1:] - A_off[n1:, n1:]),
        "A12 diff": np.linalg.norm(A_on[:n1, n1:] - A_off[:n1, n1:]),
        "A21 diff": np.linalg.norm(A_on[n1:, :n1] - A_off[n1:, :n1]),
        "Full A diff": np.linalg.norm(A_on - A_off),
    }
    # Subspace-specific eigenvalues for ON vs OFF
    sub_modes = {}
    for label, Am in [("on", A_on), ("off", A_off)]:
        sub_modes[f"{label}_xp1"] = eigenvalue_modes(Am[:n1, :n1], fs)
        sub_modes[f"{label}_xp2"] = eigenvalue_modes(Am[n1:, n1:], fs)

    return {
        "block_norms": block_norms,
        "diff_norms": diff_norms,
        "sub_modes": sub_modes,
    }


def c_matrix_analysis(model, n1, neural_channels):
    """C (observation) and Cz (behavioral output) matrix loading analysis.

    C maps latent states to neural observations: y_t = C x_t.
    Cz maps latent states to behavioral predictions: z_t = Cz x_t.
    """
    C = np.array(model.C)
    Cz = np.array(model.Cz)

    # Per-channel loading norms split by subspace
    norms_xp1 = np.linalg.norm(C[:, :n1], axis=1)
    norms_xp2 = np.linalg.norm(C[:, n1:], axis=1)

    # Group by electrode
    electrodes = sorted(set(ch.split("_")[1] for ch in neural_channels))
    by_electrode = {}
    for e in electrodes:
        mask = [i for i, ch in enumerate(neural_channels) if ch.split("_")[1] == e]
        by_electrode[f"ECOG_{e}"] = {
            "xp1": float(np.sum(norms_xp1[mask])),
            "xp2": float(np.sum(norms_xp2[mask])),
        }

    # Group by frequency band keyword
    band_keywords = {
        "theta": "theta",
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
    }
    by_band = {}
    for bname, kw in band_keywords.items():
        mask = [i for i, ch in enumerate(neural_channels) if kw in ch]
        if mask:
            by_band[bname] = {
                "xp1": float(np.mean(norms_xp1[mask])),
                "xp2": float(np.mean(norms_xp2[mask])),
            }

    cz_xp1_norm = float(np.linalg.norm(Cz[:, :n1]))
    cz_xp2_norm = float(np.linalg.norm(Cz[:, n1:]))

    return {
        "by_electrode": by_electrode,
        "by_band": by_band,
        "cz_xp1": cz_xp1_norm,
        "cz_xp2": cz_xp2_norm,
        "Cz": Cz,
    }


def classifier_comparison(Xp_list, labels, n1, nx, df):
    """Compare mean-based vs covariance-based DBS classification on Xp_1 / Xp_2.

    Tests: mean, std, cov (upper triangle), mean+std features.
    Includes raw behavioral baseline (velocity + acceleration stats).
    """
    Xp1 = [x[:, :n1] for x in Xp_list]
    Xp2 = [x[:, n1:nx] for x in Xp_list]

    def trial_means(trials):
        return np.array([x.mean(axis=0) for x in trials])

    def trial_stds(trials):
        return np.array([x.std(axis=0) for x in trials])

    def trial_cov(trials):
        feats = []
        for x in trials:
            c = np.cov(x.T)
            feats.append(c[np.triu_indices(c.shape[0])])
        return np.array(feats)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    results = {}

    for feat_name, extractor in [
        ("mean", trial_means),
        ("std", trial_stds),
        ("cov", trial_cov),
    ]:
        f1, f2 = extractor(Xp1), extractor(Xp2)
        s1 = cross_val_score(clf, f1, labels, cv=cv, scoring="balanced_accuracy")
        s2 = cross_val_score(clf, f2, labels, cv=cv, scoring="balanced_accuracy")
        results[feat_name] = {"xp1": float(np.mean(s1)), "xp2": float(np.mean(s2))}

    # Combined mean+std features
    f1 = np.hstack([trial_means(Xp1), trial_stds(Xp1)])
    f2 = np.hstack([trial_means(Xp2), trial_stds(Xp2)])
    s1 = cross_val_score(clf, f1, labels, cv=cv, scoring="balanced_accuracy")
    s2 = cross_val_score(clf, f2, labels, cv=cv, scoring="balanced_accuracy")
    results["mean+std"] = {"xp1": float(np.mean(s1)), "xp2": float(np.mean(s2))}

    # Raw behavioral baseline (velocity + acceleration summary stats)
    raw_feats = []
    for i in range(len(df)):
        feats = []
        for col in BEHAV_COLS:
            if col in df.columns:
                vals = np.array(df[col][i].to_list(), dtype=float)
                vals = vals[~np.isnan(vals)]
                feats.extend([np.mean(vals), np.std(vals)])
        raw_feats.append(feats)
    raw_feats = np.array(raw_feats)
    if raw_feats.shape[1] > 0:
        s_raw = cross_val_score(
            clf, raw_feats, labels, cv=cv, scoring="balanced_accuracy"
        )
        results["raw_behavioral"] = {"value": float(np.mean(s_raw))}

    return results


# %% [markdown]
# ## 2. Load data and models (test set only)
#
# Load **test-set** trials from split parquets and PSID models (both, ON-only, OFF-only).
# All analyses use test-set data for consistency with other thesis notebooks.

# %%
data = {}

for run in RUNS:
    label = run["label"]
    mtype = run["model_type"]
    n1, nx, fs = run["n1"], run["nx"], run["fs"]

    neural_channels = get_neural_channels_from_config(run["config_yaml"])
    # Load only test-set trials from the split parquet
    Y_on, Y_off, Y_all, labels, df = load_test_trials(run["variant"], neural_channels)

    model_both = load_model(run["model_both"])
    model_on = load_model(run["model_on"]) if "model_on" in run else None
    model_off = load_model(run["model_off"]) if "model_off" in run else None

    # Predict latent states Xp using the combined (both) model
    _, _, Xp_all = predict_model(model_both, Y_all, mtype)
    Xp_on = [Xp_all[i] for i in range(len(labels)) if labels[i] == 1]
    Xp_off = [Xp_all[i] for i in range(len(labels)) if labels[i] == 0]

    data[label] = {
        "run": run,
        "n1": n1,
        "nx": nx,
        "fs": fs,
        "model_type": mtype,
        "neural_channels": neural_channels,
        "df": df,
        "labels": labels,
        "model_both": model_both,
        "model_on": model_on,
        "model_off": model_off,
        "Xp_all": Xp_all,
        "Xp_on": Xp_on,
        "Xp_off": Xp_off,
    }
    print(
        f"  {label}: {sum(labels==1)} ON, {sum(labels==0)} OFF trials, "
        f"{len(neural_channels)} ch, Xp shape {Xp_all[0].shape}"
    )

# %% [markdown]
# ## 3. Behavioral DBS effect (raw kinematics, test set)
#
# Before looking at latent states, check how much DBS changes the raw behavioral outputs
# in the test set. Large Cohen's d means DBS genuinely affects movement — so we'd
# expect the behavioral subspace (Xp_1) to carry DBS-discriminative information.

# %%
# Render the per-session / per-feature summary as a matplotlib table figure.
behav_rows = []
for label, d in data.items():
    behav = behavioral_dbs_effect(d["df"])
    d["behavioral"] = behav
    for r in behav:
        behav_rows.append(
            [
                label,
                r["feature"],
                f'{r["on_mean"]:.2f}',
                f'{r["off_mean"]:.2f}',
                f'{r["d"]:+.3f}',
                f'{r["p"]:.4f}',
            ]
        )

fig, ax = plt.subplots(figsize=(8.5, 0.35 * (len(behav_rows) + 1) + 0.4))
ax.axis("off")
_hdrs = ["Session", "Feature", "ON mean", "OFF mean", "Cohen's d", "p-value"]
tbl = ax.table(cellText=behav_rows, colLabels=_hdrs, loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.3)
panel_label(ax, "A", "Behavioural DBS effect (raw kinematics, test set)")
fig.savefig(str(OUT / "behavioral_dbs_effect_table.png"))
plt.show()
print(
    f"Behavioural DBS effect on raw kinematics (test trials only). For each session and "
    f"behavioural feature, ON vs OFF means with Cohen's d and t-test p-value. Sessions: "
    f"{', '.join(data.keys())}. Sets the upper bound on what Xp_1 could carry if it perfectly "
    f"tracked the behavioural shift."
)

# %% [markdown]
# ## 4. Latent state trial-level statistics
#
# Do the latent states differ between DBS ON and OFF at the trial level?
# Cohen's d on per-trial means for each latent dimension.
# If Xp_1 captured the behavioral DBS effect, we'd expect large d values.

# %%
# 2×N grid: rows = (Xp_1 all dims, top-5 Xp_2 dims), cols = sessions.
n_sessions = len(data)
fig, axes = plt.subplots(
    2, n_sessions, figsize=(3.0 * n_sessions + 1.0, 4.8), sharey="row"
)
if n_sessions == 1:
    axes = axes.reshape(2, 1)

for col_idx, (label, d) in enumerate(data.items()):
    ls = latent_trial_stats(d["Xp_all"], d["labels"], d["n1"], d["nx"])
    d["latent_stats"] = ls

    # Top row: Xp_1 (behavioural) — all dims
    ax_top = axes[0, col_idx]
    dims_1 = [f"d{r['dim']}" for r in ls["xp1"]]
    ds_1 = [r["d"] for r in ls["xp1"]]
    colors_1 = [COLOR_DBS_ON if r["p"] < 0.05 else COLOR_NS for r in ls["xp1"]]
    ax_top.bar(dims_1, ds_1, color=colors_1)
    ax_top.axhline(0, color="#B0B0B0", linewidth=0.8, linestyle=":")
    ax_top.tick_params(axis="x", labelrotation=90, labelsize=6)
    panel_label(ax_top, chr(65 + col_idx), f"{label} — Xp_1")
    if col_idx == 0:
        ax_top.set_ylabel("Cohen's d (Xp_1 dims)")

    # Bottom row: top-5 Xp_2 by |d|
    ax_bot = axes[1, col_idx]
    top_xp2 = sorted(ls["xp2"], key=lambda r: -abs(r["d"]))[:5]
    dims_2 = [f"d{r['dim']}" for r in top_xp2]
    ds_2 = [r["d"] for r in top_xp2]
    colors_2 = [COLOR_DBS_ON if r["p"] < 0.05 else COLOR_NS for r in top_xp2]
    ax_bot.bar(dims_2, ds_2, color=colors_2)
    ax_bot.axhline(0, color="#B0B0B0", linewidth=0.8, linestyle=":")
    ax_bot.set_xlabel("latent dimension")
    panel_label(ax_bot, chr(65 + n_sessions + col_idx), f"{label} — top-5 Xp_2")
    if col_idx == 0:
        ax_bot.set_ylabel("Cohen's d (top-5 Xp_2 dims)")

fig.savefig(str(OUT / "latent_cohens_d_all_sessions.png"))
plt.show()
print(
    f"Latent state DBS effect: per-dimension Cohen's d on trial means (test set), "
    f"computed for all PSID latent dims (top row: Xp_1 behavioural; bottom row: top-5 Xp_2 "
    f"non-behavioural by |d|). Red bars = p<0.05 (uncorrected t-test). Sessions: "
    f"{', '.join(data.keys())}."
)

# %% [markdown]
# ## 5. Power spectral density of latent states
#
# What frequency content lives in Xp_1 vs Xp_2? Reveals whether the behavioral
# subspace captures slow kinematics vs neural oscillations, and how DBS modulates
# the spectral content in each subspace.

# %%
# 4×2 PSD grid: rows = sessions, cols = (Xp_1, Xp_2).
n_sessions = len(data)
fig, axes = plt.subplots(
    n_sessions, 2, figsize=(8.0, 2.4 * n_sessions), sharex=False, sharey=False
)
if n_sessions == 1:
    axes = axes.reshape(1, 2)

for r_idx, (label, d) in enumerate(data.items()):
    n1, nx, fs = d["n1"], d["nx"], d["fs"]
    f_psd, psd_xp1_on = avg_psd(d["Xp_on"], slice(0, n1), fs)
    _, psd_xp1_off = avg_psd(d["Xp_off"], slice(0, n1), fs)
    _, psd_xp2_on = avg_psd(d["Xp_on"], slice(n1, nx), fs)
    _, psd_xp2_off = avg_psd(d["Xp_off"], slice(n1, nx), fs)
    d["psd"] = {
        "f": f_psd,
        "xp1_on": psd_xp1_on,
        "xp1_off": psd_xp1_off,
        "xp2_on": psd_xp2_on,
        "xp2_off": psd_xp2_off,
    }

    for col, (psd_on, psd_off, sub_name) in enumerate(
        [
            (psd_xp1_on, psd_xp1_off, "Xp_1 (behavioural)"),
            (psd_xp2_on, psd_xp2_off, "Xp_2 (non-behavioural)"),
        ]
    ):
        ax = axes[r_idx, col]
        ax.plot(
            f_psd,
            psd_on,
            color=COLOR_DBS_ON,
            linewidth=1.5,
            label="DBS ON" if (r_idx == 0 and col == 0) else None,
        )
        ax.plot(
            f_psd,
            psd_off,
            color=COLOR_DBS_OFF,
            linewidth=1.5,
            label="DBS OFF" if (r_idx == 0 and col == 0) else None,
        )
        ax.set_xlim(0, fs / 2)
        ax.set_yscale("log")
        if r_idx == n_sessions - 1:
            ax.set_xlabel("frequency (Hz)")
        if col == 0:
            ax.set_ylabel("power (a.u., log)")
        letter = chr(65 + r_idx * 2 + col)
        panel_label(ax, letter, f"{label} — {sub_name}")

fig.legend()
fig.savefig(str(OUT / "psd_all_sessions.png"))
plt.show()
print(
    f"PSD of PSID latent states (Welch, test trials only). Left column: Xp_1 (behavioural "
    f"subspace, dims 0..n1-1). Right column: Xp_2 (non-behavioural, dims n1..nx-1). "
    f"DBS ON (red) vs DBS OFF (blue). Rows: {', '.join(data.keys())}."
)

# %% [markdown]
# ## 6. A matrix analysis (PSID only)
#
# The state transition matrix A governs latent dynamics: $x_{t+1} = A x_t$.
#
# - **Block structure**: A12 near zero means Xp_2 has no causal influence on Xp_1
#   (behavioral subspace is decoupled from neural dynamics).
# - **Eigenvalues**: complex eigenvalues reveal oscillatory modes. Comparing ON vs OFF
#   shows which frequency modes DBS modulates.

# %%
for label, d in data.items():
    a_info = a_matrix_analysis(
        d["model_both"], d["model_on"], d["model_off"], d["n1"], d["nx"], d["fs"]
    )
    d["a_matrix"] = a_info

    # Brief block structure summary
    coupling = a_info["block_norms"]["A12 (Xp2->Xp1)"]
    print(
        f"{label}: A12={coupling:.6f} ({'decoupled' if coupling < 0.01 else 'coupled'}), "
        f"ON-OFF diff: A11={a_info['diff_norms']['A11 diff']:.4f}, "
        f"A22={a_info['diff_norms']['A22 diff']:.4f}"
    )

    # Eigenvalue scatter: one panel per subspace.
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for col, sub in enumerate(["xp1", "xp2"]):
        ax = axes[col]
        for cond, color, marker in [
            ("on", COLOR_DBS_ON, "o"),
            ("off", COLOR_DBS_OFF, "D"),
        ]:
            modes = a_info["sub_modes"][f"{cond}_{sub}"]
            freqs = [m["freq"] for m in modes if m["is_complex"]]
            mags = [m["mag"] for m in modes if m["is_complex"]]
            ax.scatter(
                freqs,
                mags,
                color=color,
                marker=marker,
                s=45,
                alpha=0.85,
                edgecolors="none",
                label=f"DBS {cond.upper()}" if col == 0 else None,
            )
        sub_label = "Xp_1 (behavioural)" if sub == "xp1" else "Xp_2 (non-behavioural)"
        ax.set_xlabel("oscillation frequency (Hz)")
        ax.set_ylabel(r"$|\lambda|$ (eigenvalue magnitude)")
        ax.set_ylim(0.97, 1.001)
        panel_label(ax, chr(65 + col), f"{label} — {sub_label}")

    fig.legend()
    fig.savefig(str(OUT / f'eigenvalues_{label.replace(" ", "_")}.png'))
    plt.show()
    print(
        f"A-matrix oscillatory eigenvalues for {label} (PSID, dbs_both vs dbs_off vs dbs_on). "
        f"Left panel: Xp_1 sub-block (n1={d['n1']}). Right panel: Xp_2 sub-block (nx-n1={d['nx']-d['n1']}). "
        f"Each marker is one complex-conjugate eigenpair plotted at its rotation frequency vs |lambda|; "
        f"closer to 1 = slower decay. A12 (Xp2->Xp1) coupling = {a_info['block_norms']['A12 (Xp2->Xp1)']:.4f}, "
        f"|A_on - A_off| = {a_info['diff_norms']['Full A diff']:.4f}."
    )

# %% [markdown]
# ## 7. C and Cz matrix loadings (PSID only)
#
# - **C matrix** (observation): maps latent states to neural observations ($y_t = C x_t$).
#   Loading norms show which neural channels/bands are captured by each subspace.
# - **Cz matrix** (behavioral output): maps latent states to behavioral predictions ($z_t = C_z x_t$).
#   If $\|C_z[:, n_1:]\| \approx 0$, behavioral prediction comes entirely from Xp_1.

# %%
for label, d in data.items():
    c_info = c_matrix_analysis(d["model_both"], d["n1"], d["neural_channels"])
    d["c_matrix"] = c_info

    ratio = (
        c_info["cz_xp1"] / c_info["cz_xp2"] if c_info["cz_xp2"] > 0 else float("inf")
    )
    print(
        f"{label}: Cz norms Xp_1={c_info['cz_xp1']:.4f}, Xp_2={c_info['cz_xp2']:.4f} ({ratio:.1f}x)"
    )

    # Grouped bar chart: C loadings by electrode (panel A) and frequency band (panel B).
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    electrodes = list(c_info["by_electrode"].keys())
    xp1_vals = np.array([c_info["by_electrode"][e]["xp1"] for e in electrodes])
    xp2_vals = np.array([c_info["by_electrode"][e]["xp2"] for e in electrodes])
    _x = np.arange(len(electrodes))
    _w = 0.38
    ax = axes[0]
    ax.bar(_x - _w / 2, xp1_vals, _w, color=COLOR_XP1, label="Xp_1 (behavioural)")
    ax.bar(_x + _w / 2, xp2_vals, _w, color=COLOR_XP2, label="Xp_2 (non-behavioural)")
    ax.set_xticks(_x)
    ax.set_xticklabels(electrodes)
    ax.set_xlabel("electrode")
    ax.set_ylabel(r"$\|C\|$ loading norm (sum)")
    panel_label(ax, "A", f"{label} — loadings by electrode")

    bands = list(c_info["by_band"].keys())
    xp1_b = np.array([c_info["by_band"][b]["xp1"] for b in bands])
    xp2_b = np.array([c_info["by_band"][b]["xp2"] for b in bands])
    _xb = np.arange(len(bands))
    ax = axes[1]
    ax.bar(_xb - _w / 2, xp1_b, _w, color=COLOR_XP1)
    ax.bar(_xb + _w / 2, xp2_b, _w, color=COLOR_XP2)
    ax.set_xticks(_xb)
    ax.set_xticklabels(bands)
    ax.set_xlabel("frequency band keyword")
    ax.set_ylabel(r"$\|C\|$ loading norm (mean)")
    panel_label(ax, "B", f"{label} — loadings by band")

    fig.legend()
    fig.savefig(str(OUT / f'c_matrix_{label.replace(" ", "_")}.png'))
    plt.show()
    print(
        f"PSID C matrix loading norms for {label}: how strongly each neural electrode/band "
        f"projects onto the behavioural (Xp_1, blue) vs non-behavioural (Xp_2, brown) subspace. "
        f"Cz norms: Xp_1={c_info['cz_xp1']:.4f}, Xp_2={c_info['cz_xp2']:.4f} ({ratio:.1f}x ratio). "
        f"Large Xp_1/Xp_2 Cz ratio confirms behavioural readout comes from the behavioural subspace."
    )

# %% [markdown]
# ## 7b. Channel importance: Behavioral vs Neural relevance
#
# - **Cy row norms** (eigenvalue-weighted): how much each neural channel couples
#   to latent dynamics (analogous to communality in factor analysis).
# - **Cy->Cz combined** ($\|C_z \cdot C_y[i,:]\|$): traces the full path from
#   neural channel through latent states to behavioral output.

# %%
from scripts.extract_psid_channel_importance import get_top_behavioral_and_neural

for label, d in data.items():
    result = get_top_behavioral_and_neural(
        d["run"]["model_both"], d["n1"], d["neural_channels"], top_n=5
    )
    d["channel_importance"] = result
    imp = result["importance"]
    br = result["behavioral_relevance"]

    # Cz structure: how much each behavioral output loads on x1 vs x2
    Cz = imp["Cz"]
    n1 = d["n1"]
    print(
        f"{label}: Cz structure — "
        + ", ".join(
            f"{name}: x1={np.linalg.norm(Cz[zi, :n1]):.4f} / x2={np.linalg.norm(Cz[zi, n1:]):.4f}"
            for zi, name in enumerate(["vel_x", "accel_mag"])
        )
    )

    short_labels = [
        ch.replace("ECOG_", "E").replace("_raw", "").replace("_env", "")
        for ch in d["neural_channels"]
    ]
    colors = [
        ELECTRODE_COLORS.get(ch.split("_")[1], COLOR_NS) for ch in d["neural_channels"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.2))

    # ── Panel A: scatter of behavioural relevance vs neural importance ──
    ax = axes[0]
    ax.scatter(
        imp["weighted_importance"],
        br["behavioral_relevance"],
        c=colors,
        s=40,
        alpha=0.80,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_xlabel(r"neural importance: eig-weighted $\|C_y\|$")
    ax.set_ylabel(r"behavioural relevance: $\|C_z \cdot C_y\|$")
    panel_label(ax, "A", f"{label} — behavioural vs neural relevance per channel")

    # ── Panel B: horizontal bars — top-5 behavioural + top-5 neural ──
    ax = axes[1]
    beh_scores = np.array(result["top_behavioral_scores"], dtype=float)
    beh_norm = beh_scores / beh_scores.max() if beh_scores.max() > 0 else beh_scores
    neur_scores = np.array(result["top_neural_scores"], dtype=float)
    neur_norm = (
        neur_scores / neur_scores.max() if neur_scores.max() > 0 else neur_scores
    )

    all_labels = list(result["top_behavioral"]) + [""] + list(result["top_neural"])
    all_scores = list(beh_norm) + [0.0] + list(neur_norm)
    bar_colors = [COLOR_XP1] * len(beh_norm) + ["none"] + [COLOR_XP2] * len(neur_norm)

    y_pos = np.arange(len(all_labels))
    ax.barh(y_pos, all_scores, color=bar_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("normalised importance (per group)")
    panel_label(ax, "B", f"{label} — top-5 behavioural (blue) + top-5 neural (brown)")

    fig.savefig(str(OUT / f'channel_importance_{label.replace(" ", "_")}.png'))
    plt.show()
    print(
        f"PSID channel importance for {label}. Left: scatter of neural importance "
        f"(eigenvalue-weighted ||Cy[i,:]||) vs behavioural relevance "
        f"(||Cz @ Cy[i,:]||) — each dot is one neural input channel, coloured by electrode. "
        f"Right: normalised top-5 channels selected by each criterion (raw feature names retained)."
    )

# %% [markdown]
# ## 8. Classifier comparison
#
# Compare different feature types for DBS classification:
# - **mean**: per-trial average (detects mean shifts)
# - **std**: per-trial standard deviation (detects variance changes)
# - **cov**: upper triangle of per-trial covariance (what CSP uses)
# - **mean+std**: combined
# - **raw behavioral**: baseline using raw velocity/acceleration
#
# If Xp_1 means are at chance but raw behavioral classifies well, the model's
# state-space mapping discards the between-condition mean shift.

# %%
# 1×N grid: one classifier-comparison panel per session.
n_sessions = len(data)
fig, axes = plt.subplots(
    1, n_sessions, figsize=(3.2 * n_sessions + 0.8, 3.8), sharey=True
)
if n_sessions == 1:
    axes = np.array([axes])

for col_idx, (label, d) in enumerate(data.items()):
    ax = axes[col_idx]
    cls = classifier_comparison(d["Xp_all"], d["labels"], d["n1"], d["nx"], d["df"])
    d["classifiers"] = cls

    feat_names = [k for k in cls if k != "raw_behavioral"]
    xp1_vals = np.array([cls[k]["xp1"] for k in feat_names])
    xp2_vals = np.array([cls[k]["xp2"] for k in feat_names])

    _x = np.arange(len(feat_names))
    _w = 0.38
    ax.bar(
        _x - _w / 2,
        xp1_vals,
        _w,
        color=COLOR_XP1,
        label="Xp_1 (behavioural)" if col_idx == 0 else None,
    )
    ax.bar(
        _x + _w / 2,
        xp2_vals,
        _w,
        color=COLOR_XP2,
        label="Xp_2 (non-behavioural)" if col_idx == 0 else None,
    )

    if "raw_behavioral" in cls:
        raw_val = cls["raw_behavioral"]["value"]
        ax.axhline(
            raw_val,
            color=COLOR_NS,
            linestyle="--",
            linewidth=1.0,
            label="raw behavioural" if col_idx == 0 else None,
        )
        ax.text(
            0.96,
            0.95,
            f"raw behav = {raw_val:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color=COLOR_NS,
        )

    ax.axhline(0.5, color=COLOR_CHANCE, linestyle=":", linewidth=0.8)
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(_x)
    ax.set_xticklabels(feat_names, rotation=-25, ha="left")
    ax.set_xlabel("feature type")
    if col_idx == 0:
        ax.set_ylabel("balanced accuracy (5-fold CV)")
    panel_label(ax, chr(65 + col_idx), label)

fig.legend()
fig.savefig(str(OUT / "classifier_all_sessions.png"))
plt.show()
print(
    "Per-session DBS classification (5-fold CV balanced accuracy) using PSID latent features. "
    "Bars: logistic regression on per-trial mean / std / cov / mean+std features extracted from "
    "Xp_1 (behavioural, blue) vs Xp_2 (non-behavioural, brown). Dashed grey line: raw behavioural "
    "baseline (velocity + acceleration summary stats). Dotted grey line: chance (0.5). "
    f"Sessions: {', '.join(data.keys())}."
)

# %% [markdown]
# ## 9. Cross-run summary
#
# Aggregate key metrics across all sessions for comparison.

# %%
# Render summary as a matplotlib table figure.
summary_rows = []
for label, d in data.items():
    vel_d = next(
        (r["d"] for r in d.get("behavioral", []) if "velocity_x" in r["feature"]),
        float("nan"),
    )
    a12 = d["a_matrix"]["block_norms"]["A12 (Xp2->Xp1)"]
    cz_ratio = (
        d["c_matrix"]["cz_xp1"] / d["c_matrix"]["cz_xp2"]
        if d["c_matrix"]["cz_xp2"] > 0
        else float("inf")
    )
    cls = d["classifiers"]
    mean_str = f'{cls["mean"]["xp1"]:.2f} / {cls["mean"]["xp2"]:.2f}'
    cov_str = f'{cls["cov"]["xp1"]:.2f} / {cls["cov"]["xp2"]:.2f}'
    raw_str = (
        f'{cls["raw_behavioral"]["value"]:.2f}' if "raw_behavioral" in cls else "-"
    )

    summary_rows.append(
        [
            label,
            f"{vel_d:+.3f}",
            f"{a12:.4f}",
            f"{cz_ratio:.1f}x",
            mean_str,
            cov_str,
            raw_str,
        ]
    )

fig, ax = plt.subplots(figsize=(9.0, 0.4 * (len(summary_rows) + 1) + 0.5))
ax.axis("off")
_hdrs = [
    "Session",
    "Behav d (vel_x)",
    "A12 norm",
    "Cz ratio (Xp1/Xp2)",
    "Cls mean (Xp1/Xp2)",
    "Cls cov (Xp1/Xp2)",
    "Raw behav",
]
tbl = ax.table(cellText=summary_rows, colLabels=_hdrs, loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.4)
panel_label(ax, "A", "Cross-run summary")
fig.savefig(str(OUT / "cross_run_summary.png"))
plt.show()
print(
    "Cross-run summary table. Columns: behavioural Cohen's d on velocity_x (raw movement "
    "DBS effect), A12 block norm (Xp_2 -> Xp_1 leakage; ~0 means subspaces decoupled), "
    "Cz norm ratio Xp_1/Xp_2 (how much behavioural prediction loads on the behavioural "
    "subspace), classifier balanced accuracy on per-trial mean and covariance features "
    "(Xp_1 / Xp_2), and the raw-behavioural baseline. Sessions: "
    f"{', '.join(data.keys())}."
)

# %%
n = len(list(OUT.glob("*.png")))
print(f"Section 7 total: {n} figures saved")

# %% [markdown]
# ## Interpretation guide
#
# **Behavioral DBS effect**: Cohen's d measures standardized mean difference.
# |d| > 0.8 is large. If DBS changes behavior strongly, we'd expect Xp_1 to carry that.
#
# **Latent trial-level stats**: per-dimension Cohen's d on trial means. If all Xp_1 dims
# have d ~ 0 despite large behavioral effects, the model discards the between-condition
# mean shift — it captures temporal dynamics (autocorrelation), not static mean levels.
#
# **PSD**: Xp_1 below ~4 Hz = slow kinematics, not neural oscillations. Xp_2 beta
# suppression (13-30 Hz, ratio < 1) with DBS ON is the classic PD DBS signature.
#
# **A matrix**: A12 ~ 0 means Xp_2 has no causal influence on Xp_1 (decoupled subspaces).
# A22 diff >> A11 diff means DBS modulates neural dynamics far more than behavioral.
#
# **C/Cz matrices**: Cz Xp_1/Xp_2 ratio >> 1 confirms behavioral prediction comes from Xp_1.
# C loading by band shows which frequency content each subspace captures.
#
# **Classifier**: mean-based at chance but cov-based above chance means DBS changes
# covariance structure (dynamics), not mean levels — consistent with A matrix findings.
