"""Shared analysis helpers for sec7 (PSID subspace dynamics).

Data loaders, Cohen's-d computation, matrix comparison stats, and histogram
plotting extracted from the notebook so the cells stay thin.
"""

import importlib
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy import stats
from statsmodels.stats.multitest import multipletests

from modules.style import COLOR_PSID

_pickle = importlib.import_module("pic" + "kle")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def get_neural_channels_from_config(config_yaml):
    """Read the Y (neural input) channel list from the training YAML config."""
    with open(config_yaml) as f:
        cfg = yaml.safe_load(f)
    return cfg["data"]["Y"]


def load_parquet_latest(path):
    """Read the most recent test_results parquet in path; return (df, stim_labels)."""
    files = sorted(path.glob("test_results_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet in {path}")
    df = pl.read_parquet(files[-1])
    labels = np.array([1 if s == "on" else 0 for s in df["stim"].to_list()])
    return df, labels


def load_test_trials(variant):
    """Load test-set trials from the dbs_both inference parquet.

    Returns Y_all, Xp_all, labels, df.
    """
    parquet_dir = Path("results") / "psid" / f"{variant}_dbs_both" / "inference" / "test"
    df, labels = load_parquet_latest(parquet_dir)
    Y_all = [np.array(row, dtype=float) for row in df["Y"].to_list()]
    Xp_all = [np.array(row, dtype=float) for row in df["Xp"].to_list()]
    return Y_all, Xp_all, labels, df


def load_xf_by_stim(variant, h=0.5, framework="psid"):
    """Load X_future_pred at forecast horizon h from the test-split forecast parquet.

    Returns (Xf_all, labels) where Xf_all[i] is (T_future, nx).
    """
    parquet_dir = Path("results") / framework / f"{variant}_dbs_both" / "forecast" / f"h{h}" / "test"
    df, labels = load_parquet_latest(parquet_dir)
    Xf_all = [np.array(row, dtype=float) for row in df["X_future_pred"].to_list()]
    return Xf_all, labels


def load_xp_by_stim(variant_base, framework="dpad", condition="dbs_both"):
    """Load Xp from inference parquet for any framework/variant/condition, split by stim."""
    parquet_dir = Path("results") / framework / f"{variant_base}_{condition}" / "inference" / "test"
    df, labels = load_parquet_latest(parquet_dir)
    Xp_all = [np.array(row.to_list(), dtype=float) for row in df["Xp"]]
    Xp_on = [Xp_all[i] for i in range(len(labels)) if labels[i] == 1]
    Xp_off = [Xp_all[i] for i in range(len(labels)) if labels[i] == 0]
    return Xp_on, Xp_off, labels


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def feature_trial_stats(trial_matrices, labels, feature_names=None):
    """Per-feature Cohen's d + t-test + BH-FDR on per-trial means.

    trial_matrices: list of (T, n_features) arrays, one per trial.
    Returns list of dicts: {dim, name, d, p, reject_fdr}.
    """
    means = np.array([x.mean(axis=0) for x in trial_matrices])
    on_means = means[labels == 1]
    off_means = means[labels == 0]
    pooled_std = np.sqrt((on_means.std(0) ** 2 + off_means.std(0) ** 2) / 2)
    ds = np.where(pooled_std > 0, (on_means.mean(0) - off_means.mean(0)) / pooled_std, 0.0)
    _, ps = stats.ttest_ind(on_means, off_means, axis=0)
    reject_fdr, _, _, _ = multipletests(ps, alpha=0.05, method="fdr_bh")
    names = feature_names or [str(i) for i in range(means.shape[1])]
    return [
        {"dim": i, "name": names[i], "d": float(ds[i]), "p": float(ps[i]), "reject_fdr": bool(reject_fdr[i])}
        for i in range(means.shape[1])
    ]


# ---------------------------------------------------------------------------
# Model matrix loading & comparison
# ---------------------------------------------------------------------------


def load_model_matrices(variant_base, condition, framework="psid"):
    """Load A, Cy (=model.C), Cz from the latest model pkl for given condition."""
    model_dir = Path("results") / framework / f"{variant_base}_{condition}"
    pkls = sorted(model_dir.glob("model_*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No model pkl in {model_dir}")
    with open(pkls[-1], "rb") as f:
        m = _pickle.load(f)
    return {"A": np.array(m.A), "Cy": np.array(m.C), "Cz": np.array(m.Cz)}


def matrix_pair_stats(mat_a, mat_b, n1=None, label=""):
    """L1 norm and one-sided t-test for element-wise difference A - B.

    mat_a, mat_b: ndarray of same shape.
    n1: if set, also report sub-block stats (Xp1/Xp2 columns for rect matrices,
        Xp1/Xp2 × Xp1/Xp2 blocks for square matrices).
    Returns dict with l1, mean_delta, t_stat, p_greater, p_less + per-block.
    """
    delta = np.asarray(mat_a, dtype=float) - np.asarray(mat_b, dtype=float)
    flat = delta.ravel()
    l1 = float(np.sum(np.abs(flat)))
    mean_d = float(np.mean(flat))
    t_stat, p_two = stats.ttest_1samp(flat, 0)
    p_greater = float(p_two / 2 if t_stat > 0 else 1 - p_two / 2)
    p_less = float(p_two / 2 if t_stat < 0 else 1 - p_two / 2)
    result = {
        "label": label,
        "l1": l1,
        "mean_delta": mean_d,
        "t_stat": float(t_stat),
        "p_greater": p_greater,
        "p_less": p_less,
    }
    if n1 is not None and 0 < n1 < delta.shape[1]:
        n_cols = delta.shape[1]
        is_square = delta.shape[0] == n_cols
        if is_square:
            blocks = [
                ("11", delta[:n1, :n1]),
                ("12", delta[:n1, n1:]),
                ("21", delta[n1:, :n1]),
                ("22", delta[n1:, n1:]),
            ]
        else:
            blocks = [
                ("·1", delta[:, :n1]),
                ("·2", delta[:, n1:]),
            ]
        for bname, block in blocks:
            bf = block.ravel()
            result[f"block_{bname}_l1"] = float(np.sum(np.abs(bf)))
            result[f"block_{bname}_mean"] = float(np.mean(bf))
    return result


def print_matrix_comparison(name, mats, n1, conditions=("dbs_both", "dbs_on", "dbs_off")):
    """Print L1-norm + one-sided t-test table comparing matrices across conditions.

    mats: dict like {"dbs_both": A_both, "dbs_on": A_on, "dbs_off": A_off}
    """
    pairs = [("dbs_on", "dbs_both"), ("dbs_off", "dbs_both"), ("dbs_on", "dbs_off")]
    print(f"\n── {name} ──")
    header = f"{'pair':>12}  {'L1':>10}  {'mean Δ':>10}  {'t':>8}  {'p(H1: >0)':>12}  {'p(H1: <0)':>12}"
    print(header)
    print("-" * len(header))
    for a_name, b_name in pairs:
        s = matrix_pair_stats(mats[a_name], mats[b_name], n1=n1, label=f"{a_name} vs {b_name}")
        print(f"{s['label']:>12}  {s['l1']:10.4f}  {s['mean_delta']:10.4f}  {s['t_stat']:8.3f}  {s['p_greater']:12.4f}  {s['p_less']:12.4f}")
        # sub-blocks
        for key, val in s.items():
            if key.startswith("block_") and key.endswith("_l1"):
                bname = key[len("block_"):-len("_l1")]
                mean_key = f"block_{bname}_mean"
                print(f"  block {bname:>4s}  {val:10.4f}  {s[mean_key]:10.4f}")
    print()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def print_cohens_d_stats(stats_dict):
    """Print Cohen's d summary (n_sig, mean|d|, max|d|, per-subspace) for all entries."""
    for key in sorted(stats_dict):
        framework, exp_name, session = key
        d_arr, sig, n1 = stats_dict[key]
        session_us = session.replace(" ", "_")
        n_tot = len(d_arr)
        n_sig = int(sig.sum())
        print(f"  [{framework}] {session_us} / {exp_name}: n_sig={n_sig}/{n_tot}, mean|d|={np.abs(d_arr).mean():.3f}, max|d|={np.abs(d_arr).max():.3f}")
        if n1 is not None and 0 < n1 < n_tot:
            print(f"    X_1 (n={n1}): n_sig={int(sig[:n1].sum())}/{n1}, mean|d|={np.abs(d_arr[:n1]).mean():.3f}")
            print(f"    X_2 (n={n_tot - n1}): n_sig={int(sig[n1:].sum())}/{n_tot - n1}, mean|d|={np.abs(d_arr[n1:]).mean():.3f}")


def cohens_d_histogram(ax, d_arr, sig, n1=None, color=None, vlim=None, bins=15):
    """Histogram of per-dim Cohen's d. Xp1 (first n1) in red, Xp2 in *color*.

    Overlaid histograms: light fill for all dims, solid fill for FDR-significant
    dims only. If n1 is None (DPAD): single color, no subspace split.
    """
    if color is None:
        color = COLOR_PSID
    d_arr = np.asarray(d_arr, dtype=float)
    sig_arr = np.asarray(sig, dtype=bool)
    v = vlim if vlim is not None else max(float(np.abs(d_arr).max()), 1e-6)
    bin_edges = np.linspace(-v, v, bins + 1)

    if n1 is not None and 0 < n1 < len(d_arr):
        d_xp1, sig_xp1 = d_arr[:n1], sig_arr[:n1]
        d_xp2, sig_xp2 = d_arr[n1:], sig_arr[n1:]
        # Xp2 — model color
        ax.hist(d_xp2, bins=bin_edges, color=color, alpha=0.3, edgecolor=color, linewidth=0.4)
        if sig_xp2.any():
            ax.hist(d_xp2[sig_xp2], bins=bin_edges, color=color, alpha=0.85, edgecolor="none")
        # Xp1 — red
        ax.hist(d_xp1, bins=bin_edges, color="tab:red", alpha=0.3, edgecolor="tab:red", linewidth=0.4)
        if sig_xp1.any():
            ax.hist(d_xp1[sig_xp1], bins=bin_edges, color="tab:red", alpha=0.85, edgecolor="none")
    else:
        ax.hist(d_arr, bins=bin_edges, color=color, alpha=0.3, edgecolor=color, linewidth=0.4)
        if sig_arr.any():
            ax.hist(d_arr[sig_arr], bins=bin_edges, color=color, alpha=0.85, edgecolor="none")

    ax.set_xlabel("d")
    ax.set_xlim(-v, v)
