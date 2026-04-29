# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # Simplified PSID pipeline (single session, DBS=both)
#
# Goal: end-to-end exploration in one notebook —
# 1. Load one session's trials (ECoG candidates + behavioural targets + DBS labels).
# 2. mRMR feature selection on ECoG, target = DBS state (`feature_engine.MRMR`).
# 3. Train PSID on selected channels.
# 4. One-step prediction + multi-step forecast on val/test.
# 5. DBS on/off classification on PSID latents (LDA).
#
# Compared to `scripts/pipeline_psid.py`:
# - No subprocess phases, no YAML configs, no on/off split models.
# - Single mRMR pass instead of cached `mrmr_picks.yaml`.
# - In-process LDA only — no permutations, no h×m grid, no `Xp_with_dbs` flavours.

# %%
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/bobby/repos/latent-neural-dynamics-modeling")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import polars as pl
import yaml
import matplotlib.pyplot as plt
from feature_engine.selection import MRMR
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import PSID
from PSID.PSID import fitCzViaKFRegression
from utils.frameworks import PSIDWrapper
from utils.stats import pearson_r_per_channel

# %% [markdown]
# ## Knobs

# %%
PARTICIPANT = "PDI4"
SESSION = "2"
SFREQ = 200

DATA_ROOT = (
    PROJECT_ROOT
    / "resampled_recordings"
    / "participants_at_200Hz_scaled_1e6_narrow_band"
)
SESSION_PATH = DATA_ROOT / f"participant_id={PARTICIPANT}" / f"session={SESSION}"

BANDS = [
    "theta_4_8_raw",
    "alpha_8_12_raw",
    "beta_12_17_raw",
    "beta_17_22_raw",
    "beta_22_27_raw",
    "beta_27_30_raw",
    "gamma_30_35_raw",
    "gamma_35_40_raw",
    "gamma_40_45_raw",
    "gamma_45_50_raw",
    "gamma_50_55_raw",
    "gamma_55_60_raw",
    "gamma_60_65_raw",
    "gamma_70_75_raw",
    "gamma_75_80_raw",
]
Y_CANDIDATES = [f"ECOG_{e}_{b}" for e in range(1, 5) for b in BANDS]

# Laplacian skip-1 pairs entirely within LFP channels 9..16 — Z output pool
LAP_PAIRS_9_16 = ["9-11", "10-12", "11-13", "12-14", "13-15", "14-16"]
Z_CANDIDATES = [f"LAPLACIAN_{p}_LFP_{b}" for p in LAP_PAIRS_9_16 for b in BANDS]
print(f"Y candidates (ECoG): {len(Y_CANDIDATES)}")
print(f"Z candidates (laplacian LFP 9..16): {len(Z_CANDIDATES)}")

K_Y = 25  # final Y/ECoG features (after both MRMR stages)
K_Z = 8  # final Z/LFP features (after both MRMR stages)
K_Y_INT = 40  # intermediate pool after stage-1 MRMR for Y
K_Z_INT = 30  # intermediate pool after stage-1 MRMR for Z


# nx/n1 picked by val-fold cross-validated CC (Sani et al protocol, single-fold
# variant — see "nx/n1 selection" cell below). Set USE_CV_SELECTION=False to
# fall back on configs/diagnostic/elbow_choices.yaml.
USE_CV_SELECTION = True
# nx capped at user's magnitude elbow (~100-120). n1 capped at behavior
# subspace elbow (~50-60). PSID hard limits (nz·i, ny·i) are far above these
# and don't bind here.
NX_GRID = [120]  # smoke-test: single point
N1_GRID = [50]  # smoke-test: single point

# Early stop on plateau (Sani parsimony rule, single-fold variant): walk grid
# in increasing order; stop when next CC falls within tolerance of the running
# best — pick smallest nx/n1 achieving near-max CC. Saves time, prefers
# parsimony.
PLATEAU_TOL = 0.005
I_HORIZON = 100
SKIP_CZ_KALMAN_REFIT = True  # use LS-on-Xk Cz only; skip post-DARE Kalman refit
MAX_EIG = 0.99  # clip A eigenvalues so DARE refit succeeds (scipy 1.16 needs e=None)

# Fallback elbow values (used if USE_CV_SELECTION=False or CV fails).
elbow_path = PROJECT_ROOT / "configs" / "diagnostic" / "elbow_choices.yaml"
elbow = yaml.safe_load(elbow_path.read_text())
elbow_rec = elbow["sessions"][f"{PARTICIPANT}_S{SESSION}"]["laplacian"]
NX_FALLBACK = int(elbow_rec["nx"])
N1_FALLBACK = int(elbow_rec["n1"])
HISTORY_S = 2.0
FORECAST_S = 1.0
EPOCH_LEN_S = 0.5
EPOCH_OVERLAP = 0.25

split_path = PROJECT_ROOT / "configs" / "splits" / f"{PARTICIPANT}_S{SESSION}.yaml"
split_cfg = yaml.safe_load(split_path.read_text())
TRAIN_BLOCKS = set(split_cfg["train_blocks"])
VAL_BLOCKS = set(split_cfg["val_blocks"])
TEST_BLOCKS = set(split_cfg["test_blocks"])
print(
    f"split blocks  train={sorted(TRAIN_BLOCKS)}  val={sorted(VAL_BLOCKS)}  test={sorted(TEST_BLOCKS)}"
)

# %% [markdown]
# ## Load session → list of trial dicts
#
# Each trial row in parquet stores per-channel signals as `List[Float]`. Both
# ECoG and laplacian LFP columns carry `chunk_margin` seconds of pre/post
# padding (length n). After symmetric trim they are identical length, no
# alignment needed.


# %%
def load_session_trials(session_path, y_cols, z_cols, sfreq):
    rows = []
    for bf in sorted(session_path.glob("block=*/0.parquet")):
        block_num = int(bf.parent.name.split("=")[1])
        df = pl.read_parquet(bf)
        for i in range(len(df)):
            r = df[i]
            cm_ts = int(round(float(r["chunk_margin"][0]) * sfreq))
            Y_full = np.column_stack(
                [np.asarray(r[c][0], dtype=np.float64) for c in y_cols]
            )
            Z_full = np.column_stack(
                [np.asarray(r[c][0], dtype=np.float64) for c in z_cols]
            )
            n = Y_full.shape[0]
            rows.append(
                {
                    "block": block_num,
                    "trial": int(r["trial"][0]),
                    "stim": r["stim"][0],
                    "Y": Y_full[cm_ts : n - cm_ts],
                    "Z": Z_full[cm_ts : n - cm_ts],
                }
            )
    return rows


trials = load_session_trials(SESSION_PATH, Y_CANDIDATES, Z_CANDIDATES, SFREQ)
n_on = sum(t["stim"] == "on" for t in trials)
n_off = sum(t["stim"] == "off" for t in trials)
print(f"loaded {len(trials)} trials  (on={n_on}, off={n_off})")
print(
    f"first trial: Y.shape={trials[0]['Y'].shape}, Z.shape={trials[0]['Z'].shape}, stim={trials[0]['stim']}"
)

# %% [markdown]
# ## Block-level split (canonical, from `configs/splits/{P}_S{S}.yaml`)

# %%
train_trials = [t for t in trials if t["block"] in TRAIN_BLOCKS]
val_trials = [t for t in trials if t["block"] in VAL_BLOCKS]
test_trials = [t for t in trials if t["block"] in TEST_BLOCKS]
for name, ts in (("train", train_trials), ("val", val_trials), ("test", test_trials)):
    on = sum(t["stim"] == "on" for t in ts)
    off = sum(t["stim"] == "off" for t in ts)
    print(f"{name}: n={len(ts)}  on={on}  off={off}")

# %% [markdown]
# ## Feature selection — pairwise cross-corr → FCQ vs DBS
#
# Per-trial log(std) per channel = log-RMS = canonical band-power proxy for
# narrow-band zero-mean signals. Shape: 48 rows (trials) × 60 (Y) / 90 (Z).
#
# **Stage 1 — cross-modality via pairwise correlation:**
#   For each Y candidate i, score = `max_j |corr(Y_i_logpower, Z_j_logpower)|`
#   over the 48 trials. Picks Y features that have at least one Z partner with
#   strong linear coupling — same-band pairs (shared oscillatory generators)
#   surface naturally. PSID stage-1 only finds linear coupling, so this matches
#   what PSID will exploit. Symmetric for Z (max over Y partners).
#
# **Stage 2 — DBS separability:**
#   `MRMR(method="FCQ", regression=False)` on stage-1 survivors. F-statistic
#   (ANOVA F) for class separation, Pearson redundancy, quotient scheme.
#   F-stat is the linear measure of how well a feature splits on/off groups —
#   matches PSID's linearity assumption better than RF importance.

# %%
Y_train_agg = np.stack([np.log(np.std(t["Y"], axis=0) + 1e-12) for t in train_trials])
Z_train_agg = np.stack([np.log(np.std(t["Z"], axis=0) + 1e-12) for t in train_trials])
trial_labels = np.array(
    [1 if t["stim"] == "on" else 0 for t in train_trials], dtype=np.int64
)
print(
    f"trial-aggregate shape: Y={Y_train_agg.shape}  Z={Z_train_agg.shape}  labels={trial_labels.shape}"
)

K_Y_INT_eff = min(K_Y_INT, len(Y_CANDIDATES))
K_Z_INT_eff = min(K_Z_INT, len(Z_CANDIDATES))


def block_demean(arr_agg, blocks):
    """Subtract per-block mean → removes block-level drift confound from cross-corr."""
    out = arr_agg.astype(np.float64, copy=True)
    for b in np.unique(blocks):
        m = blocks == b
        out[m] -= out[m].mean(axis=0)
    return out


def feature_band(name, bands):
    return next(b for b in bands if name.endswith(b))


def pairwise_cross_corr_same_band(X_agg, X_names, Y_agg, Y_names, bands):
    """Same-band partner search.
    For each X feature at band b, score = max |corr| over Y features at band b.
    Returns (score, partner_idx_in_Y) per X feature.
    """
    Xz = (X_agg - X_agg.mean(0)) / (X_agg.std(0) + 1e-12)
    Yz = (Y_agg - Y_agg.mean(0)) / (Y_agg.std(0) + 1e-12)
    R = (Xz.T @ Yz) / X_agg.shape[0]
    score = np.zeros(len(X_names))
    partner = np.full(len(X_names), -1, dtype=np.int64)
    for i, xn in enumerate(X_names):
        b = feature_band(xn, bands)
        same = [j for j, yn in enumerate(Y_names) if yn.endswith(b)]
        if not same:
            continue
        absR = np.abs(R[i, same])
        best = absR.argmax()
        score[i] = absR[best]
        partner[i] = same[best]
    return score, partner


def cross_corr_select(src_agg, src_names, partner_agg, partner_names, blocks, k, tag):
    src_dm = block_demean(src_agg, blocks)
    par_dm = block_demean(partner_agg, blocks)
    score, partner_idx = pairwise_cross_corr_same_band(
        src_dm, src_names, par_dm, partner_names, BANDS
    )
    order = np.argsort(score)[::-1]
    chosen = order[:k]
    print(f"[{tag}] stage-1 (block-demean + same-band cross-corr) → top-{k}")
    for i in chosen:
        pj = partner_idx[i]
        partner_str = partner_names[pj] if pj >= 0 else "(none)"
        print(
            f"    {src_names[i]:<40s}  partner={partner_str:<40s}  |r|={score[i]:.3f}"
        )
    inter_names = [src_names[i] for i in chosen]
    return inter_names, score, partner_idx


def fcq_select(src_agg, src_names, dbs_target, k, tag):
    df = pd.DataFrame(src_agg, columns=src_names)
    print(f"[{tag}] stage-2 FCQ vs DBS → top-{k}")
    sel = MRMR(method="FCQ", max_features=k, regression=False, random_state=42)
    sel.fit(df, dbs_target)
    final = list(sel.transform(df).columns)
    for c in final:
        print(f"    {c}")
    return final


train_blocks_arr = np.array([t["block"] for t in train_trials])

# --- Y selection ---
y_inter_cols, y_cross_score, y_partner_idx = cross_corr_select(
    Y_train_agg,
    Y_CANDIDATES,
    Z_train_agg,
    Z_CANDIDATES,
    train_blocks_arr,
    K_Y_INT_eff,
    tag="Y/ECoG",
)
y_inter_idx = [Y_CANDIDATES.index(c) for c in y_inter_cols]
selected_y = fcq_select(
    Y_train_agg[:, y_inter_idx], y_inter_cols, trial_labels, K_Y, tag="Y/ECoG"
)

# --- Z selection ---
z_inter_cols, z_cross_score, z_partner_idx = cross_corr_select(
    Z_train_agg,
    Z_CANDIDATES,
    Y_train_agg,
    Y_CANDIDATES,
    train_blocks_arr,
    K_Z_INT_eff,
    tag="Z/LFP",
)
z_inter_idx = [Z_CANDIDATES.index(c) for c in z_inter_cols]
selected_z = fcq_select(
    Z_train_agg[:, z_inter_idx], z_inter_cols, trial_labels, K_Z, tag="Z/LFP"
)


# %% [markdown]
# ### Raw-signal correlation cleanup
#
# RFCQ ranked features in log-power space; redundancy was scored as Pearson on
# log-RMS. Two channels can be linearly de-correlated in log-power but highly
# correlated in *raw* narrow-band waveform (volume conduction). PSID would see
# redundant raw inputs. Drop features with |r| > `RAW_CORR_THRESH` on raw
# concatenated train signals, keeping MRMR-higher-ranked feature in each pair.

# %%
RAW_CORR_THRESH = 0.95


def drop_correlated_in_order(arr, names, threshold):
    keep_idx = []
    kept_names = []
    for i in range(arr.shape[1]):
        ok = True
        for j in keep_idx:
            r = np.corrcoef(arr[:, i], arr[:, j])[0, 1]
            if abs(r) > threshold:
                ok = False
                break
        if ok:
            keep_idx.append(i)
            kept_names.append(names[i])
    return kept_names


y_raw = np.concatenate(
    [t["Y"][:, [Y_CANDIDATES.index(c) for c in selected_y]] for t in train_trials],
    axis=0,
)
y_pre = list(selected_y)
selected_y = drop_correlated_in_order(y_raw, selected_y, RAW_CORR_THRESH)
print(
    f"Y/ECoG raw-corr cleanup: {len(y_pre)} → {len(selected_y)}  (|r|>{RAW_CORR_THRESH})"
)
for c in y_pre:
    if c not in selected_y:
        print(f"    dropped: {c}")
sel_y_idx = [Y_CANDIDATES.index(c) for c in selected_y]

z_raw = np.concatenate(
    [t["Z"][:, [Z_CANDIDATES.index(c) for c in selected_z]] for t in train_trials],
    axis=0,
)
z_pre = list(selected_z)
selected_z = drop_correlated_in_order(z_raw, selected_z, RAW_CORR_THRESH)
print(
    f"Z/LFP raw-corr cleanup: {len(z_pre)} → {len(selected_z)}  (|r|>{RAW_CORR_THRESH})"
)
for c in z_pre:
    if c not in selected_z:
        print(f"    dropped: {c}")
sel_z_idx = [Z_CANDIDATES.index(c) for c in selected_z]

# %% [markdown]
# ### Selection visual

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, max(4, max(K_Y, K_Z) * 0.22)))
axes[0].barh(
    range(len(selected_y))[::-1], range(len(selected_y), 0, -1), color="#3a7bd5"
)
axes[0].set_yticks(range(len(selected_y))[::-1])
axes[0].set_yticklabels(selected_y, fontsize=7)
axes[0].set_xlabel("MRMR rank (DBS stage)")
axes[0].set_title(f"Y/ECoG top-{K_Y} (2-stage MRMR)")

axes[1].barh(range(len(selected_z))[::-1], range(len(selected_z), 0, -1), color="#d33")
axes[1].set_yticks(range(len(selected_z))[::-1])
axes[1].set_yticklabels(selected_z, fontsize=7)
axes[1].set_xlabel("MRMR rank (DBS stage)")
axes[1].set_title(f"Z/LFP top-{K_Z} (2-stage MRMR)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## PSID training
#
# PSID accepts list-of-trials directly — no concat needed. Each trial is one
# epoch, block-aware via the precomputed split. Z-scoring/mean removal handled
# internally by upstream `PSID.PSID` (defaults `zscore_Y=True, zscore_Z=True`),
# so we don't pre-zscore.
#
# `fit_Cz_via_KF` choices (upstream PSID kwarg):
#   - `True`  → fit Cz by regressing Z onto Kalman-predicted state Xp_KF.
#               Closer to deployment behaviour (Z predictions use same Xp).
#   - `False` → fit Cz by regressing Z onto smoothed/non-KF state estimate.
#               Tighter fit on training set but can mis-generalize at inference.
# `True` is the pipeline production default.

# %%
Y_train = [t["Y"][:, sel_y_idx] for t in train_trials]
Z_train = [t["Z"][:, sel_z_idx] for t in train_trials]
Y_val = [t["Y"][:, sel_y_idx] for t in val_trials]
Z_val = [t["Z"][:, sel_z_idx] for t in val_trials]
Y_test = [t["Y"][:, sel_y_idx] for t in test_trials]
Z_test = [t["Z"][:, sel_z_idx] for t in test_trials]


# %% [markdown]
# ### nx / n1 selection — Sani et al. cross-validated CC (single-fold, val-based)
#
# Sani et al. (2021) protocol distinguishes two estimates:
#
# **nx** (total neural dim) — fit SID (= PSID with `n1=0`) on train, predict
# Y on val, compute Pearson r per Y channel averaged across channels (= CC).
# Pick max-CC nx. Sani uses k-fold CV + 1-sem parsimony rule; we use
# single val fold + max-CC for speed (caveat noted).
#
# **n1** (behaviour-relevant dim) — given chosen nx, fit PSID(nx, n1) on
# train, predict Z on val, compute per-Z-channel Pearson r averaged. Pick
# max-CC n1.
#
# Both rules use Z (not Y) decoding only at the n1 stage — n1 is supervised.
# Y self-prediction is a generative-fit criterion at the nx stage.


# %%
def _avg_per_channel_cc(true_list, pred_list):
    """Pearson per channel (mean across trials), then mean across channels.
    Matches Sani's "CC averaged across behavior dimensions" exactly.
    """
    per_trial, _ = pearson_r_per_channel(true_list, pred_list)
    arr = np.array([list(r) for r in per_trial], dtype=float)
    ch_means = np.nanmean(arr, axis=0)
    return float(np.nanmean(ch_means))


def fit_psid_one(Y_tr, Z_tr, nx, n1, i, ws=None):
    """Fit + clip + DARE refit + Cz refit. Returns (wrapped_model, ws).

    Pass `ws` from previous call on same data to reuse upstream PSID's cached
    block-Hankel matrices and SVDs (Yp, ZHat_U/S, YHat_U/S). Upstream
    invalidates stage-2 cache when n1 changes (PSID.py line 363).

    `fit_Cz_via_KF=False` upstream — gives us the GOOD LS-on-Xk Cz unaffected
    by the broken DARE inside upstream's LSSM constructor. After our own DARE
    refit (with valid K), we re-run `fitCzViaKFRegression` so Cz is fit on the
    Kalman-predicted state we'll use at inference (matching the original
    `fit_Cz_via_KF=True` semantics, but with valid K).
    """
    try:
        idSys, ws_out = PSID.PSID(
            Y_tr,
            Z_tr if n1 > 0 else None,
            nx=nx,
            n1=n1,
            i=i,
            time_first=True,
            fit_Cz_via_KF=False,  # skip upstream's broken Cz refit
            WS=ws if ws is not None else dict(),
            return_WS=True,
        )
        idSys.A = PSIDWrapper._clip_A_eigenvalues(np.asarray(idSys.A), MAX_EIG)
        PSIDWrapper._refit_dare_if_needed(idSys, force=True)
        # Optional: re-fit Cz on Kalman-predicted state (matches upstream
        # fit_Cz_via_KF=True intent, but on valid post-clip K).
        if (
            (not SKIP_CZ_KALMAN_REFIT)
            and Z_tr is not None
            and n1 > 0
            and not np.any(np.isnan(idSys.K))
        ):
            try:
                idSys.Cz = fitCzViaKFRegression(idSys, Y_tr, Z_tr, time_first=True)
            except Exception as e:
                print(f"    Cz refit skipped nx={nx} n1={n1}: {e}")
        PSIDWrapper._cache_A_powers(idSys)
        return PSIDWrapper.from_idsys(idSys), ws_out
    except Exception as e:
        print(f"    fit failed nx={nx} n1={n1}: {e}")
        return None, ws


def _pick_parsimony(results, tol, fallback):
    """Smallest grid key whose CC is within `tol` of running max.
    Returns `fallback` if all results are NaN.
    """
    valid = [(k, v) for k, v in sorted(results.items()) if not np.isnan(v)]
    if not valid:
        print(f"  all CCs NaN → fallback to {fallback}")
        return fallback
    best = max(v for _, v in valid)
    for k, v in valid:
        if v >= best - tol:
            return k
    return valid[0][0]


def sweep_with_early_stop(grid, fit_predict_cc, label, tol):
    results = {}
    running_max = -np.inf
    for k in sorted(grid):
        cc = fit_predict_cc(k)
        results[k] = cc
        marker = ""
        if not np.isnan(cc):
            if cc > running_max:
                running_max = cc
            elif cc < running_max - tol:
                results[k] = cc
                marker = "  (plateau — stop)"
                print(f"  {label}={k:3d}  CC = {cc:.4f}{marker}")
                break
        print(f"  {label}={k:3d}  CC = {cc:.4f}{marker}")
    return results


if USE_CV_SELECTION:
    # --- nx sweep via SID self-prediction CC ---
    # ws_sid caches Yp + YHat_U/S — recomputed once at first call, reused across
    # all subsequent nx values (n1=0 throughout, only the truncation changes).
    print(f"nx sweep (n1=0, SID, Y self-pred CC on val, plateau_tol={PLATEAU_TOL}):")
    ws_sid = dict()

    def _nx_cc(nx):
        global ws_sid
        m, ws_sid = fit_psid_one(Y_train, None, nx, 0, I_HORIZON, ws_sid)
        if m is None:
            return float("nan")
        _, Yp_cv, _ = m.predict(Y_val)
        return _avg_per_channel_cc(Y_val, Yp_cv)

    nx_results = sweep_with_early_stop(NX_GRID, _nx_cc, "nx", PLATEAU_TOL)
    NX = _pick_parsimony(nx_results, PLATEAU_TOL, NX_FALLBACK)
    cc_str = (
        f"{nx_results[NX]:.4f}"
        if NX in nx_results and not np.isnan(nx_results.get(NX, np.nan))
        else "fallback"
    )
    print(f"chosen nx = {NX}  (val Y CC = {cc_str})")

    # --- n1 sweep via PSID Z decoding CC ---
    # Fresh ws_psid: stage-1 SVD (ZHat) is reused across all n1 calls (data
    # unchanged); stage-2 SVD is invalidated each n1 (residual depends on n1).
    n1_grid = [n for n in N1_GRID if 0 < n <= NX]
    print(f"n1 sweep (nx={NX}, Z decoding CC on val, plateau_tol={PLATEAU_TOL}):")
    ws_psid = dict()

    def _n1_cc(n1):
        global ws_psid
        m, ws_psid = fit_psid_one(Y_train, Z_train, NX, n1, I_HORIZON, ws_psid)
        if m is None:
            return float("nan")
        Zp_cv, _, _ = m.predict(Y_val)
        return _avg_per_channel_cc(Z_val, Zp_cv)

    n1_results = sweep_with_early_stop(n1_grid, _n1_cc, "n1", PLATEAU_TOL)
    N1 = _pick_parsimony(n1_results, PLATEAU_TOL, min(N1_FALLBACK, NX))
    cc_str = (
        f"{n1_results[N1]:.4f}"
        if N1 in n1_results and not np.isnan(n1_results.get(N1, np.nan))
        else "fallback"
    )
    print(f"chosen n1 = {N1}  (val Z CC = {cc_str})")
else:
    NX, N1 = NX_FALLBACK, N1_FALLBACK
    print(f"USE_CV_SELECTION=False → fallback elbow yaml: nx={NX}  n1={N1}")


# %%
# Final fit at chosen (nx, n1) on full train.
idSys, ws = PSID.PSID(
    Y_train,
    Z_train,
    nx=NX,
    n1=N1,
    i=I_HORIZON,
    time_first=True,
    fit_Cz_via_KF=False,  # bypass upstream's broken Cz-via-Kalman path
    return_WS=True,
)
idSys.ZHat_S = ws.get("ZHat_S")
idSys.YHat_S = ws.get("YHat_S")

# Eigenvalue clip is optional; clipping triggers DARE refit which may fail on
# scipy-1.16 → NaN K → constant Zp. Skip when MAX_EIG is None.
if MAX_EIG is not None:
    idSys.A = PSIDWrapper._clip_A_eigenvalues(np.asarray(idSys.A), MAX_EIG)
    PSIDWrapper._refit_dare_if_needed(idSys, force=True)
# Optionally re-fit Cz on Kalman-predicted state with valid K.
if (not SKIP_CZ_KALMAN_REFIT) and N1 > 0 and not np.any(np.isnan(idSys.K)):
    idSys.Cz = fitCzViaKFRegression(idSys, Y_train, Z_train, time_first=True)
PSIDWrapper._cache_A_powers(idSys)
if np.any(np.isnan(getattr(idSys, "K", np.asarray([])))):
    print("WARNING: idSys.K contains NaN — Zp will be constant (DARE failed).")

A_eigs = np.linalg.eigvals(idSys.A)
print(
    f"trained PSID  nx={NX} n1={N1} i={I_HORIZON}  "
    f"max|eig(A)|={np.max(np.abs(A_eigs)):.4f}"
)

psid_model = PSIDWrapper.from_idsys(idSys)

# %% [markdown]
# ### Singular spectra (informs nx/n1 elbow choice)

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for ax, S, ttl in (
    (axes[0], idSys.ZHat_S, "Stage-1 ZHat (behaviour-relevant)"),
    (axes[1], idSys.YHat_S, "Stage-2 YHat (residual)"),
):
    if S is None:
        ax.set_visible(False)
        continue
    ax.plot(np.arange(1, len(S) + 1), S, marker="o", ms=3, color="#3a7bd5")
    ax.set_xlabel("singular index")
    ax.set_ylabel("singular value")
    ax.set_title(ttl, fontsize=10)
    ax.set_yscale("log")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## One-step prediction (val + test)

# %%
Zp_tr, Yp_tr, Xp_tr = psid_model.predict(Y_train)
Zp_val, Yp_val, Xp_val = psid_model.predict(Y_val)
Zp_te, Yp_te, Xp_te = psid_model.predict(Y_test)

_, r_y_val = pearson_r_per_channel(Y_val, Yp_val)
_, r_y_te = pearson_r_per_channel(Y_test, Yp_te)
_, r_z_val = pearson_r_per_channel(Z_val, Zp_val)
_, r_z_te = pearson_r_per_channel(Z_test, Zp_te)

# Diagnose: when Zp std ~ 0, ECoG→LFP cross-decoding collapsed (PSID stage-1
# found Z unobservable from Y). Per-channel std in original units.
zp_std_te = np.mean([np.std(z, axis=0) for z in Zp_te], axis=0)
zt_std_te = np.mean([np.std(z, axis=0) for z in Z_test], axis=0)
print(f"val   Y-pred Pearson r = {r_y_val:.4f}   Z-pred Pearson r = {r_z_val:.4f}")
print(f"test  Y-pred Pearson r = {r_y_te:.4f}   Z-pred Pearson r = {r_z_te:.4f}")
print(f"test Zp std/Z std ratio: {(zp_std_te / np.maximum(zt_std_te, 1e-12)).round(3)}")

# %% [markdown]
# ### Per-channel Pearson r (test split)

# %%
r_y_per_ch_te, _ = pearson_r_per_channel(Y_test, Yp_te)
r_z_per_ch_te, _ = pearson_r_per_channel(Z_test, Zp_te)
r_y_means = np.nanmean(np.array([list(r) for r in r_y_per_ch_te], dtype=float), axis=0)
r_z_means = np.nanmean(np.array([list(r) for r in r_z_per_ch_te], dtype=float), axis=0)

fig, axes = plt.subplots(
    1, 2, figsize=(12, max(3, max(len(selected_y), len(selected_z)) * 0.22))
)
axes[0].barh(range(len(selected_y))[::-1], r_y_means, color="#3a7bd5")
axes[0].axvline(0, color="k", lw=0.5)
axes[0].set_yticks(range(len(selected_y))[::-1])
axes[0].set_yticklabels(selected_y, fontsize=7)
axes[0].set_xlabel("mean Pearson r (test)")
axes[0].set_title("Y reconstruction (ECoG)")

axes[1].barh(range(len(selected_z))[::-1], r_z_means, color="#d33")
axes[1].axvline(0, color="k", lw=0.5)
axes[1].set_yticks(range(len(selected_z))[::-1])
axes[1].set_yticklabels(selected_z, fontsize=7)
axes[1].set_xlabel("mean Pearson r (test)")
axes[1].set_title("Z cross-decoding (LFP)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Example trial — true vs predicted Z (LFP, top-3 channels)

# %%
n_show = min(3, len(selected_z))
fig, axes = plt.subplots(n_show, 1, figsize=(8, 2.0 * n_show), sharex=True)
if n_show == 1:
    axes = [axes]
trial0 = test_trials[0]
Z_true_tr = Z_test[0]
Z_pred_tr = Zp_te[0]
t_axis = np.arange(Z_true_tr.shape[0]) / SFREQ
for ax, ch_idx in zip(axes, range(n_show)):
    ax.plot(t_axis, Z_true_tr[:, ch_idx], color="k", lw=1, label="true")
    ax.plot(t_axis, Z_pred_tr[:, ch_idx], color="#d33", lw=1, label="pred")
    ax.set_ylabel(selected_z[ch_idx], fontsize=7)
    ax.legend(loc="best", fontsize=8)
axes[-1].set_xlabel("time (s)")
fig.suptitle(
    f"test trial b{trial0['block']} t{trial0['trial']} (stim={trial0['stim']})",
    fontsize=10,
)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Multi-step forecast
#
# Per trial: feed `HISTORY_S` of past Y, forecast `FORECAST_S` ahead. Compare
# `Yf` to true future on the same window. Z forecast comes for free via the
# learned `Cz`.

# %%
m = int(FORECAST_S * SFREQ)
h = int(HISTORY_S * SFREQ)


def forecast_split(Y_split, Z_split, h, m, psid_model):
    out = {"Yf": [], "Zf": [], "Xf": [], "Y_true": [], "Z_true": []}
    for Y, Z in zip(Y_split, Z_split):
        if Y.shape[0] < h + m:
            continue
        Zf, Yf, Xf = psid_model.forecast(m, Y[:h])
        out["Yf"].append(Yf)
        out["Zf"].append(Zf)
        out["Xf"].append(Xf)
        out["Y_true"].append(Y[h : h + m])
        out["Z_true"].append(Z[h : h + m])
    return out


fc_val = forecast_split(Y_val, Z_val, h, m, psid_model)
fc_te = forecast_split(Y_test, Z_test, h, m, psid_model)

_, r_yf_val = pearson_r_per_channel(fc_val["Y_true"], fc_val["Yf"])
_, r_yf_te = pearson_r_per_channel(fc_te["Y_true"], fc_te["Yf"])
_, r_zf_val = pearson_r_per_channel(fc_val["Z_true"], fc_val["Zf"])
_, r_zf_te = pearson_r_per_channel(fc_te["Z_true"], fc_te["Zf"])

print(f"forecast {FORECAST_S:.1f}s  history {HISTORY_S:.1f}s")
print(f"val   Y-fcst r = {r_yf_val:.4f}   Z-fcst r = {r_zf_val:.4f}")
print(f"test  Y-fcst r = {r_yf_te:.4f}   Z-fcst r = {r_zf_te:.4f}")

# %% [markdown]
# ### Forecast example (test trial 0, top-3 LFP)

# %%
n_show = min(3, len(selected_z))
fig, axes = plt.subplots(n_show, 1, figsize=(8, 2.0 * n_show), sharex=True)
if n_show == 1:
    axes = [axes]
t_past = np.arange(h) / SFREQ
t_fut = np.arange(h, h + m) / SFREQ
trial0 = test_trials[0]
for ax, ch_idx in zip(axes, range(n_show)):
    Z_full = Z_test[0]
    ax.plot(t_past, Z_full[:h, ch_idx], color="k", lw=1, label="past (true)")
    ax.plot(
        t_fut,
        fc_te["Z_true"][0][:, ch_idx],
        color="k",
        lw=1,
        ls="--",
        label="future (true)",
    )
    ax.plot(t_fut, fc_te["Zf"][0][:, ch_idx], color="#d33", lw=1, label="forecast")
    ax.set_ylabel(selected_z[ch_idx], fontsize=7)
    ax.legend(loc="best", fontsize=8)
axes[-1].set_xlabel("time (s)")
fig.suptitle(
    f"forecast — b{trial0['block']} t{trial0['trial']} (stim={trial0['stim']})",
    fontsize=10,
)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## DBS classification on PSID latents (Xp)
#
# Epoch each trial's `Xp` with `EPOCH_LEN_S` windows + `EPOCH_OVERLAP`
# overlap, mean-pool per epoch → fixed-length feature vector. Train LDA on
# train+val, score on test.

# %%
ep_len = int(EPOCH_LEN_S * SFREQ)
step = max(1, int(ep_len * (1 - EPOCH_OVERLAP)))


def epochs_from_trials(Xp_list, trial_dicts, ep_len, step):
    X_eps, y_eps, groups = [], [], []
    for Xp, td in zip(Xp_list, trial_dicts):
        if Xp is None:
            continue
        Xp = np.asarray(Xp)
        T = Xp.shape[0]
        if T < ep_len:
            continue
        label = 1 if td["stim"] == "on" else 0
        for s in range(0, T - ep_len + 1, step):
            X_eps.append(Xp[s : s + ep_len].mean(axis=0))
            y_eps.append(label)
            groups.append((td["block"], td["trial"]))
    return np.asarray(X_eps), np.asarray(y_eps), groups


X_fit, y_fit, _ = epochs_from_trials(
    Xp_tr + Xp_val, train_trials + val_trials, ep_len, step
)
X_te, y_te, _ = epochs_from_trials(Xp_te, test_trials, ep_len, step)
print(f"epochs: fit={X_fit.shape}, test={X_te.shape}")

scaler = StandardScaler().fit(X_fit)
clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
clf.fit(scaler.transform(X_fit), y_fit)

y_pred = clf.predict(scaler.transform(X_te))
ba = balanced_accuracy_score(y_te, y_pred)
cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
print(f"test balanced accuracy: {ba:.4f}")
print(f"confusion matrix [rows=true off/on, cols=pred off/on]:\n{cm}")

# %% [markdown]
# ### Confusion matrix

# %%
fig, ax = plt.subplots(figsize=(3.5, 3.2))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
        )
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["off", "on"])
ax.set_yticklabels(["off", "on"])
ax.set_xlabel("pred")
ax.set_ylabel("true")
ax.set_title(f"DBS classification — BA={ba:.3f}")
fig.colorbar(im, ax=ax, fraction=0.045)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Latent separability (first 2 dims, test set)

# %%
if Xp_te[0] is not None and Xp_te[0].shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, color, name in ((0, "#3a7bd5", "off"), (1, "#d33", "on")):
        mask = y_te == label
        if mask.any():
            ax.scatter(
                X_te[mask, 0], X_te[mask, 1], s=8, alpha=0.5, color=color, label=name
            )
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("Xp[0] (mean over epoch)")
    ax.set_ylabel("Xp[1] (mean over epoch)")
    ax.set_title("test epochs in first 2 latent dims")
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Summary
#
# | metric                      | value |
# |-----------------------------|-------|
# | mRMR-selected channels      | see list above |
# | PSID nx / n1                | configured at top |
# | val Y-pred Pearson r        | computed above |
# | test Y-pred Pearson r       | computed above |
# | val Z-pred Pearson r        | computed above |
# | test Z-pred Pearson r       | computed above |
# | val  forecast Y r           | computed above |
# | test forecast Y r           | computed above |
# | DBS LDA balanced accuracy   | computed above |
