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
# Settled config (200 Hz raw+envelope dataset):
# 1. Load one session's trials (ECoG Y candidates, laplacian LFP Z candidates).
# 2. CV-stabilized single-stage mRMR-MIQ vs DBS, raw/env stratified, K=12.
#    Splits: ChronoGroupsSplit (block-aware, label-balanced).
# 3. Grid-search nx (val Y forecast r) + n1 (val LDA BA on Xp) + MAX_EIG.
# 4. Final PSID fit on chosen (nx, n1, MAX_EIG).
# 5. One-step prediction + multi-step forecast on val/test.
# 6. DBS classification on Xp and Xf via CSP→LDA and PCA→LDA.

# %%
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/bobby/repos/latent-neural-dynamics-modeling")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import time
import numpy as np
import pandas as pd
import polars as pl
import yaml
import matplotlib.pyplot as plt
from feature_engine.selection import MRMR
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
import PSID
from PSID.PSID import fitCzViaKFRegression
from utils.frameworks import PSIDWrapper
from utils.classification.splits import ChronoGroupsSplit
from utils.stats import pearson_r_per_channel
from utils.classification.pipeline import create_pipeline

# %% [markdown]
# ## Knobs

# %%
PARTICIPANT = "PDI1"
SESSION = "4"
SFREQ = 200

DATASET_TAG = "participants_at_200Hz_scaled_1e6_raw_envelope"
DATA_ROOT = PROJECT_ROOT / "resampled_recordings" / DATASET_TAG
SESSION_PATH = DATA_ROOT / f"participant_id={PARTICIPANT}" / f"session={SESSION}"

# 200 Hz output, 17 narrow-band freq ranges × {raw, env} = 34 feature names.
# Gap [47, 53] around 50 Hz mains. Bands stop at 93 (Nyquist margin + skip
# 100 Hz harmonic).
_BAND_FREQS = [
    ("theta_4_8", [4, 8]),
    ("alpha_8_12", [8, 12]),
    ("beta_13_18", [13, 18]),
    ("beta_18_23", [18, 23]),
    ("beta_23_28", [23, 28]),
    ("beta_28_32", [28, 32]),
    ("gamma_32_37", [32, 37]),
    ("gamma_37_42", [37, 42]),
    ("gamma_42_47", [42, 47]),
    ("gamma_53_58", [53, 58]),
    ("gamma_58_63", [58, 63]),
    ("gamma_63_68", [63, 68]),
    ("gamma_68_73", [68, 73]),
    ("gamma_73_78", [73, 78]),
    ("gamma_78_83", [78, 83]),
    ("gamma_83_88", [83, 88]),
    ("gamma_88_93", [88, 93]),
]
BANDS = [f"{name}_{suf}" for name, _ in _BAND_FREQS for suf in ("raw", "env")]
BEHAVIOR_COLS = ["tracing_velocity_x", "tracing_acceleration_magnitude"]
LAP_PAIRS_9_16 = ["9-11", "10-12", "11-13", "12-14", "13-15", "14-16"]
ECOG_COLS = [f"ECOG_{e}_{b}" for e in range(1, 5) for b in BANDS]
LAP_COLS = [f"LAPLACIAN_{p}_LFP_{b}" for p in LAP_PAIRS_9_16 for b in BANDS]

# Y = ECoG bands (Sani et al. setup, neural input). Z = laplacian LFP for
# cross-modal decoding.
Y_SOURCE = "ecog"
Z_SOURCE = "laplacian"
Y_CANDIDATES = list(ECOG_COLS)
Z_CANDIDATES = list(LAP_COLS)
print(f"Y candidates (ECoG): {len(Y_CANDIDATES)}")
print(f"Z candidates ({Z_SOURCE}): {len(Z_CANDIDATES)}")

# K=12, raw/env stratified ⇒ 6 raw + 6 env per modality.
K_Y = 12
K_Z = 12
STRATIFY_RAW_ENV = True

# Grid search: nx by val Y forecast r (plateau-stop), n1 by val LDA BA on
# Xp (full grid argmax — non-monotone), MAX_EIG by val Y forecast r.
NX_GRID = [10, 20, 30, 50, 80]
N1_GRID = [2, 4, 8, 12, 20, 30, 50]
PLATEAU_TOL = 0.005
I_HORIZON = 100
SKIP_CZ_KALMAN_REFIT = (
    False  # refit Cz on Kalman-predicted Xp after A clip + DARE refit
)
MAX_EIG = 0.996  # clip A eigenvalues so DARE refit succeeds (scipy 1.16 needs e=None)

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
            n_y = Y_full.shape[0]
            n_z = Z_full.shape[0]
            # ECoG/LAP cols carry chunk_margin pre/post pad → trim. Behavior
            # cols (tracing_*) are interpolated to original_length_ts only
            # (no margin) → leave as-is. Detect via length match against Y.
            y_trim = Y_full[cm_ts : n_y - cm_ts]
            z_trim = Z_full[cm_ts : n_z - cm_ts] if n_z == n_y else Z_full
            # Harmonize: rounding in time_original vs band sample count can
            # introduce 1-sample drift between behavior and ECoG. Truncate to
            # common length so PSID sees aligned (Y_t, Z_t).
            T = min(y_trim.shape[0], z_trim.shape[0])
            y_trim = y_trim[:T]
            z_trim = z_trim[:T]
            rows.append(
                {
                    "block": block_num,
                    "trial": int(r["trial"][0]),
                    "stim": r["stim"][0],
                    "Y": y_trim,
                    "Z": z_trim,
                }
            )
    return rows


trials = load_session_trials(SESSION_PATH, Y_CANDIDATES, Z_CANDIDATES, SFREQ)

# (Block-level z-score was tried and removed: each block is pure DBS-on or
# pure DBS-off, so per-block mean removal *zeros out* the inter-block DBS
# signature → classification crashed to chance and forecast first-0.1s r
# halved. PSID's internal zscore_Y=True over the whole train pool keeps
# the block differences and is the right default.)

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
# ## Feature selection — single-stage CV mRMR-MIQ vs DBS
#
# Per-trial log(std) per channel = log-RMS = canonical band-power proxy for
# narrow-band zero-mean signals. Shape: (n_trials, n_features).
# `cv_fcq_select` runs mRMR-MIQ (mutual info / corr) over ChronoGroupsSplit
# folds. MIQ over FCQ → captures nonlinear DBS effects (phase-amplitude
# coupling). Chrono splits → block-aware, label-balanced folds.

# %%
Y_train_agg = np.stack([np.log(np.std(t["Y"], axis=0) + 1e-12) for t in train_trials])
Z_train_agg = np.stack([np.log(np.std(t["Z"], axis=0) + 1e-12) for t in train_trials])
trial_labels = np.array(
    [1 if t["stim"] == "on" else 0 for t in train_trials], dtype=np.int64
)
print(
    f"trial-aggregate shape: Y={Y_train_agg.shape}  Z={Z_train_agg.shape}  labels={trial_labels.shape}"
)


def _cv_mrmr_pool(src_agg, src_names, dbs_target, k, tag, method, trial_blocks):
    """One CV-stabilized mRMR pass on a single feature pool.

    Per fold (ChronoGroupsSplit), fit mRMR on fold-train rows, take ``2*k``
    ranked picks, record (count, rank-sum). Aggregate by count desc, full-train
    MI desc, mean_rank asc.
    """
    from sklearn.feature_selection import mutual_info_classif

    if src_agg.shape[1] == 0:
        return []
    df = pd.DataFrame(src_agg, columns=src_names)
    chrono = ChronoGroupsSplit(
        warn_if_blocks_ignored=False, allow_mixed_label_groups=False
    )
    splits = chrono.split(src_agg, dbs_target, trial_blocks)
    n_actual = len(splits)
    print(f"[{tag}] ChronoGroupsSplit {n_actual} folds | pool={len(src_names)} | k={k}")
    counts: dict[str, int] = {}
    rank_sum: dict[str, int] = {}
    eff_k = min(k, len(src_names))
    for tr, _ in splits:
        sel = MRMR(
            method=method, max_features=eff_k * 2, regression=False, random_state=42
        )
        sel.fit(df.iloc[tr], dbs_target[tr])
        ranked = list(sel.transform(df.iloc[tr]).columns)
        for r, c in enumerate(ranked):
            counts[c] = counts.get(c, 0) + 1
            rank_sum[c] = rank_sum.get(c, 0) + r

    mi_full = mutual_info_classif(src_agg, dbs_target, random_state=42)
    mi_by_name = dict(zip(src_names, mi_full))

    def _key(name):
        return (-counts[name], -mi_by_name.get(name, 0), rank_sum[name] / counts[name])

    pool = sorted(counts.keys(), key=_key)
    chosen = pool[:eff_k]
    print(f"  {'feature':<48s} {'folds':>6s} {'rank':>6s} {'MI':>8s}")
    for c in chosen:
        print(
            f"  {c:<48s} {counts[c]:>3d}/{n_actual:<2d} "
            f"{rank_sum[c]/counts[c]:>6.1f} {mi_by_name.get(c, float('nan')):>8.4f}"
        )
    print(f"  [debug] {len(counts)}/{len(src_names)} candidates picked in ≥1 fold")
    return chosen


def cv_fcq_select(
    src_agg,
    src_names,
    dbs_target,
    k,
    tag,
    stratify=False,
    method="MIQ",
    trial_blocks=None,
):
    """CV mRMR vs DBS label.

    stratify=True ⇒ run mRMR independently on raw-only and env-only sub-pools,
    K/2 each. Raw and env carry different information (raw → phase, env →
    amplitude); separate ranking lets each compete fairly within its type
    instead of penalizing same-band raw↔env pairs as redundant.

    stratify=False ⇒ single pool ranking.
    """
    if not stratify:
        return _cv_mrmr_pool(
            src_agg, src_names, dbs_target, k, tag, method, trial_blocks
        )

    is_raw = [n.endswith("_raw") for n in src_names]
    is_env = [n.endswith("_env") for n in src_names]
    raw_idx = [i for i, m in enumerate(is_raw) if m]
    env_idx = [i for i, m in enumerate(is_env) if m]
    rest_idx = [i for i in range(len(src_names)) if not is_raw[i] and not is_env[i]]
    per = max(k // 2, 1)
    chosen: list[str] = []
    if raw_idx:
        chosen += _cv_mrmr_pool(
            src_agg[:, raw_idx],
            [src_names[i] for i in raw_idx],
            dbs_target,
            per,
            f"{tag}/raw",
            method,
            trial_blocks,
        )
    if env_idx:
        chosen += _cv_mrmr_pool(
            src_agg[:, env_idx],
            [src_names[i] for i in env_idx],
            dbs_target,
            per,
            f"{tag}/env",
            method,
            trial_blocks,
        )
    if rest_idx and len(chosen) < k:
        chosen += _cv_mrmr_pool(
            src_agg[:, rest_idx],
            [src_names[i] for i in rest_idx],
            dbs_target,
            k - len(chosen),
            f"{tag}/rest",
            method,
            trial_blocks,
        )
    return chosen[:k]


train_blocks_arr = np.array([t["block"] for t in train_trials])

selected_y = cv_fcq_select(
    Y_train_agg,
    Y_CANDIDATES,
    trial_labels,
    K_Y,
    tag="Y/ECoG",
    stratify=STRATIFY_RAW_ENV,
    trial_blocks=train_blocks_arr,
)
selected_z = cv_fcq_select(
    Z_train_agg,
    Z_CANDIDATES,
    trial_labels,
    K_Z,
    tag="Z",
    stratify=STRATIFY_RAW_ENV,
    trial_blocks=train_blocks_arr,
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
sel_y_idx = [Y_CANDIDATES.index(c) for c in selected_y]
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
axes[0].set_title(f"Y/ECoG top-{K_Y} (CV mRMR-MIQ)")

axes[1].barh(range(len(selected_z))[::-1], range(len(selected_z), 0, -1), color="#d33")
axes[1].set_yticks(range(len(selected_z))[::-1])
axes[1].set_yticklabels(selected_z, fontsize=7)
axes[1].set_xlabel("MRMR rank (DBS stage)")
axes[1].set_title(f"Z/LFP top-{K_Z} (CV mRMR-MIQ)")
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
# ### Why nx → forecast, n1 → classification
#
# PSID's state has block structure: `x_t = [x1; x2]` with `x1 ∈ R^{n1}` the
# behavior-relevant subspace and `x2 ∈ R^{nx-n1}` the residual Y subspace.
# The system matrices factor as:
#
# ```
#   A = [[A11, A12],
#        [ 0,  A22]]            x1 evolves coupled with x2; x2 is autonomous
#   z_t = Cz · x_t = [Cz1, 0] · x_t   ⇒  Z depends only on x1
#   y_t = C  · x_t = [C1,  C2]· x_t   ⇒  Y uses both
# ```
#
# By construction, stage-1 PSID picks `x1` to maximize Z-predictability from
# the Y subspace. Stage-2 fills `x2` with whatever Y dynamics remain after
# the Z-projection. So:
#
# - **n1 → DBS classification**. When Z correlates with DBS state (Z=lap or
#   Z=behavior tracks DBS), `x1` ends up containing the DBS-modulated
#   directions of Y. LDA on Xp picks them up. Larger n1 = richer Z-aligned
#   subspace = better BA. Tiny n1 (1–4) under-resolves the DBS signature.
# - **nx − n1 → Y forecast**. `x2` captures Y's stationary autoregressive
#   structure (oscillations, envelopes). Larger n2 = more Y modes available
#   for `A^t` to extrapolate.
# - Yes, the **classifier sees the full Xp (all nx dims)**, but PSID's
#   construction guarantees `x2 ⊥ Z`. If DBS information lives only in Z's
#   span (which is what the data suggests), then `x2` carries little extra
#   class info; n1 is the load-bearing dimension for BA.
#
# Two-objective sweep:
#   1. **nx**: SID (n1=0) on train, val Y forecast r per-trial-feat → argmax.
#   2. **n1** (given nx): PSID(nx, n1) on train, val LDA BA on Xp, full-grid
#      argmax — BA can dip near 0.5 at small n1 then peak at n1≈10–30 so
#      plateau-stop fires too early.
#   3. **MAX_EIG** (given nx, n1): post-hoc clip sweep maximizing val Y
#      forecast r (no upstream re-fit; just A clip + DARE refit).


# %%
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


def _val_forecast_avg_r(model, Y_split, Z_split, h_samples, m_samples, target):
    """Mean per-trial-feat Pearson r over val forecasts (target='Y' or 'Z')."""
    true_list, pred_list = [], []
    for Y, Z in zip(Y_split, Z_split):
        if Y.shape[0] < h_samples + m_samples:
            continue
        Zf, Yf, _ = model.forecast(m_samples, Y[:h_samples])
        if target == "Y":
            true_list.append(Y[h_samples : h_samples + m_samples])
            pred_list.append(Yf)
        else:
            true_list.append(Z[h_samples : h_samples + m_samples])
            pred_list.append(Zf)
    if not true_list:
        return float("nan")
    _, r = pearson_r_per_channel(true_list, pred_list)
    return r


h_samp = int(HISTORY_S * SFREQ)
m_samp = int(FORECAST_S * SFREQ)

# --- nx sweep: val Y forecast r, plateau-stop ---
print(f"nx sweep (n1=0, SID, val Y forecast r, plateau_tol={PLATEAU_TOL}):")
ws_sid = dict()


def _nx_cc(nx):
    global ws_sid
    m, ws_sid = fit_psid_one(Y_train, None, nx, 0, I_HORIZON, ws_sid)
    if m is None:
        return float("nan")
    return _val_forecast_avg_r(m, Y_val, Z_val, h_samp, m_samp, target="Y")


nx_results = sweep_with_early_stop(NX_GRID, _nx_cc, "nx", PLATEAU_TOL)
# Parsimony: smallest nx with r within PLATEAU_TOL of running max.
valid_nx = [(k, v) for k, v in sorted(nx_results.items()) if not np.isnan(v)]
if valid_nx:
    best_r = max(v for _, v in valid_nx)
    NX = next(k for k, v in valid_nx if v >= best_r - PLATEAU_TOL)
else:
    NX = NX_GRID[0]
print(f"chosen nx = {NX}  (val Y forecast r = {nx_results[NX]:.4f})")

# Release SID Hankel cache before PSID stage-2 builds its own (Y+Z) Hankel.
# Keeping both alive doubles peak RAM at large nx and triggers OOM on long
# sessions (PDI1_S2 ~30k samples × i=100 × ny=12 → ~290 MB Hankel each).
import gc

ws_sid = None
gc.collect()

# --- n1 sweep: val LDA BA on Xp, full-grid argmax (BA non-monotone) ---
n1_grid = [n for n in N1_GRID if 0 < n <= NX]
print(f"n1 sweep (nx={NX}, val LDA BA on Xp):")
ws_psid = dict()


def _ba_on_xp_val(model):
    Xp_tr_list = model.predict(Y_train)[2]
    Xp_va_list = model.predict(Y_val)[2]
    ep_len_v = int(EPOCH_LEN_S * SFREQ)
    step_v = max(1, int(ep_len_v * (1 - EPOCH_OVERLAP)))

    def _epochs(Xp_list, dicts):
        X, y = [], []
        for Xp, td in zip(Xp_list, dicts):
            if Xp is None:
                continue
            Xp = np.asarray(Xp)
            T = Xp.shape[0]
            if T < ep_len_v:
                continue
            lab = 1 if td["stim"] == "on" else 0
            for s in range(0, T - ep_len_v + 1, step_v):
                X.append(Xp[s : s + ep_len_v].mean(axis=0))
                y.append(lab)
        return np.asarray(X), np.asarray(y)

    X_tr, y_tr = _epochs(Xp_tr_list, train_trials)
    X_va, y_va = _epochs(Xp_va_list, val_trials)
    if len(X_tr) == 0 or len(X_va) == 0 or len(set(y_va)) < 2:
        return float("nan")
    sc = StandardScaler().fit(X_tr)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(sc.transform(X_tr), y_tr)
    y_pred = clf.predict(sc.transform(X_va))
    return float(balanced_accuracy_score(y_va, y_pred))


def _n1_cc(n1):
    global ws_psid
    m, ws_psid = fit_psid_one(Y_train, Z_train, NX, n1, I_HORIZON, ws_psid)
    if m is None:
        return float("nan")
    return _ba_on_xp_val(m)


n1_results: dict[int, float] = {}
for n1 in sorted(n1_grid):
    t0 = time.time()
    cc = _n1_cc(n1)
    dt = time.time() - t0
    n1_results[n1] = cc
    print(f"  n1={n1:3d}  BA = {cc:.4f}  ({dt:.1f}s)")
valid_n1 = [(k, v) for k, v in n1_results.items() if not np.isnan(v)]
N1 = max(valid_n1, key=lambda kv: kv[1])[0] if valid_n1 else min(n1_grid)
print(f"chosen n1 = {N1}  (val LDA BA on Xp = {n1_results[N1]:.4f})")


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

# MAX_EIG grid search — pick clip that maximizes val Y forecast r.
# Each candidate clones idSys, applies clip + DARE refit on a copy of the
# upstream A, evaluates val Y forecast r. Cheap because no full refit.
MAX_EIG_GRID = [0.99, 0.996, 0.999, 0.9999, 0.99999, 0.999999]
A_orig = np.asarray(idSys.A).copy()
h_samp_eig = int(HISTORY_S * SFREQ)
m_samp_eig = int(FORECAST_S * SFREQ)

print(f"\nMAX_EIG grid sweep (val Y fcst r):")
best_max_eig, best_r = None, -np.inf
for me in MAX_EIG_GRID:
    cand = copy.deepcopy(idSys)
    cand.A = PSIDWrapper._clip_A_eigenvalues(A_orig, me)
    PSIDWrapper._refit_dare_if_needed(cand, force=True)
    PSIDWrapper._cache_A_powers(cand)
    if np.any(np.isnan(getattr(cand, "K", np.asarray([])))):
        print(f"  MAX_EIG={me:.6f}  K=NaN (DARE failed) — skip")
        continue
    wrap = PSIDWrapper.from_idsys(cand)
    val_r = _val_forecast_avg_r(wrap, Y_val, Z_val, h_samp_eig, m_samp_eig, "Y")
    print(f"  MAX_EIG={me:.6f}  val Y fcst r = {val_r:.4f}")
    if not np.isnan(val_r) and val_r > best_r:
        best_r, best_max_eig = val_r, me

if best_max_eig is None:
    best_max_eig = MAX_EIG  # fallback to default
print(f"chosen MAX_EIG = {best_max_eig}  (val Y fcst r = {best_r:.4f})")
MAX_EIG = best_max_eig

# Apply the chosen clip + DARE refit to the live model.
idSys.A = PSIDWrapper._clip_A_eigenvalues(A_orig, MAX_EIG)
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
fig.tight_layout()
plt.show()

# %% [markdown]
# ## One-step prediction (val + test)


# %%
def sani_cc(true_list, pred_list):
    """Sani et al. CC, per-feature flavor: concat all trials' samples per
    feature (mRMR-selected band×electrode column) → 1 Pearson r per feature
    → mean across features. We feed PSID a *pool* of K_Y=K_Z=8 selected
    features, not raw channels — so "per feature" is the relevant axis.

    Differs from ``pearson_r_per_channel`` which means over (trial, feature)
    pairs (each trial contributes its own r). CC pools temporal samples
    across trials *before* correlating → captures feature-level fit, not
    trial-to-trial variability.
    """
    if not true_list:
        return float("nan"), []
    true_cat = np.concatenate(true_list, axis=0)
    pred_cat = np.concatenate(pred_list, axis=0)
    n_feat = true_cat.shape[1]
    rs = []
    for c in range(n_feat):
        t = true_cat[:, c]
        p = pred_cat[:, c]
        if np.std(t) < 1e-12 or np.std(p) < 1e-12:
            continue
        r = np.corrcoef(t, p)[0, 1]
        if np.isfinite(r):
            rs.append(r)
    return (float(np.mean(rs)) if rs else float("nan")), rs


Zp_tr, Yp_tr, Xp_tr = psid_model.predict(Y_train)
Zp_val, Yp_val, Xp_val = psid_model.predict(Y_val)
Zp_te, Yp_te, Xp_te = psid_model.predict(Y_test)

# Two-flavor metric:
#   r_*    : pearson_r_per_channel — mean over (trial, channel) pairs.
#   cc_*   : Sani et al. CC — concat trials per channel, 1 r per channel,
#            mean across channels. CC pools temporal samples before
#            correlating, so it represents channel-level fit not trial noise.
_, r_y_val = pearson_r_per_channel(Y_val, Yp_val)
_, r_y_te = pearson_r_per_channel(Y_test, Yp_te)
_, r_z_val = pearson_r_per_channel(Z_val, Zp_val)
_, r_z_te = pearson_r_per_channel(Z_test, Zp_te)
cc_y_val, _ = sani_cc(Y_val, Yp_val)
cc_y_te, _ = sani_cc(Y_test, Yp_te)
cc_z_val, _ = sani_cc(Z_val, Zp_val)
cc_z_te, _ = sani_cc(Z_test, Zp_te)

# Diagnose: when Zp std ~ 0, ECoG→LFP cross-decoding collapsed (PSID stage-1
# found Z unobservable from Y). Per-channel std in original units.
zp_std_te = np.mean([np.std(z, axis=0) for z in Zp_te], axis=0)
zt_std_te = np.mean([np.std(z, axis=0) for z in Z_test], axis=0)
print(
    f"val   Y-pred  r = {r_y_val:.4f}  CC = {cc_y_val:.4f}    Z-pred  r = {r_z_val:.4f}  CC = {cc_z_val:.4f}"
)
print(
    f"test  Y-pred  r = {r_y_te:.4f}  CC = {cc_y_te:.4f}    Z-pred  r = {r_z_te:.4f}  CC = {cc_z_te:.4f}"
)
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
# ### Example trial — true vs predicted (test trial 0)
#
# Top row(s): Y/ECoG channels — what PSID self-predicts via `(A, C)`.
# Bottom row(s): Z/LFP (or Z=Y) channels — predicted via `Cz`.


# %%
def plot_pred_panel(true_arr, pred_arr, names, sfreq, color, title, n_show=None):
    if n_show is None:
        n_show = true_arr.shape[1]
    fig, axes = plt.subplots(n_show, 1, figsize=(8, 1.4 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    t_axis = np.arange(true_arr.shape[0]) / sfreq
    for ax, ch_idx in zip(axes, range(n_show)):
        ax.plot(t_axis, true_arr[:, ch_idx], color="k", lw=1, label="true")
        ax.plot(t_axis, pred_arr[:, ch_idx], color=color, lw=1, label="pred")
        ax.set_ylabel(names[ch_idx], fontsize=7)
        if ch_idx == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()


trial0 = test_trials[0]
trial_tag = f"test trial b{trial0['block']} t{trial0['trial']} (stim={trial0['stim']})"
plot_pred_panel(
    Y_test[0],
    Yp_te[0],
    selected_y,
    SFREQ,
    "#3a7bd5",
    f"Y/ECoG self-prediction — {trial_tag}",
)
plot_pred_panel(
    Z_test[0], Zp_te[0], selected_z, SFREQ, "#d33", f"Z reconstruction — {trial_tag}"
)

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
cc_yf_val, _ = sani_cc(fc_val["Y_true"], fc_val["Yf"])
cc_yf_te, _ = sani_cc(fc_te["Y_true"], fc_te["Yf"])
cc_zf_val, _ = sani_cc(fc_val["Z_true"], fc_val["Zf"])
cc_zf_te, _ = sani_cc(fc_te["Z_true"], fc_te["Zf"])

print(f"forecast {FORECAST_S:.1f}s  history {HISTORY_S:.1f}s")
print(
    f"val   Y-fcst  r = {r_yf_val:.4f}  CC = {cc_yf_val:.4f}    Z-fcst  r = {r_zf_val:.4f}  CC = {cc_zf_val:.4f}"
)
print(
    f"test  Y-fcst  r = {r_yf_te:.4f}  CC = {cc_yf_te:.4f}    Z-fcst  r = {r_zf_te:.4f}  CC = {cc_zf_te:.4f}"
)


def per_horizon_pearson(true_list, pred_list):
    """Per-step Pearson r averaged across channels and trials.
    Each entry in lists has shape (m, n_ch). Returns array of length m.
    """
    m = true_list[0].shape[0]
    n_ch = true_list[0].shape[1]
    rs = np.full(m, np.nan)
    for k in range(m):
        true_k = np.array([t[k, :] for t in true_list])  # (n_trials, n_ch)
        pred_k = np.array([p[k, :] for p in pred_list])
        ch_rs = []
        for c in range(n_ch):
            tk = true_k[:, c]
            pk = pred_k[:, c]
            if np.std(tk) < 1e-12 or np.std(pk) < 1e-12:
                continue
            r = np.corrcoef(tk, pk)[0, 1]
            if np.isfinite(r):
                ch_rs.append(r)
        if ch_rs:
            rs[k] = float(np.mean(ch_rs))
    return rs


def per_horizon_rmse(true_list, pred_list):
    """Per-step RMSE averaged across channels and trials. Z-scores per
    channel (using true std on test) so RMSE is dimensionless / comparable
    across bands. Returns array of length m.
    """
    m = true_list[0].shape[0]
    n_ch = true_list[0].shape[1]
    # per-channel std across all (trial, time) for normalization
    all_true = np.concatenate(true_list, axis=0)
    ch_std = all_true.std(axis=0)
    rmses = np.full(m, np.nan)
    for k in range(m):
        true_k = np.array([t[k, :] for t in true_list])
        pred_k = np.array([p[k, :] for p in pred_list])
        sq = (true_k - pred_k) ** 2  # (n_trials, n_ch)
        per_ch_rmse = np.sqrt(sq.mean(axis=0))
        norm = per_ch_rmse / np.maximum(ch_std, 1e-12)
        rmses[k] = float(np.nanmean(norm))
    return rmses


def per_horizon_r2(true_list, pred_list):
    """Per-step R² per feature, mean across features.

    R²_k,c = 1 − Σ_trials (y_{trial,k,c} − ŷ_{trial,k,c})²
                / Σ_trials (y_{trial,k,c} − ȳ_{k,c})²
    where ȳ_{k,c} = mean across trials at horizon k for feature c.

    Penalizes scale + bias drift; Pearson r misses both. Negative R² →
    forecast worse than horizon-mean baseline. Returns array of length m.
    """
    m = true_list[0].shape[0]
    r2s = np.full(m, np.nan)
    for k in range(m):
        true_k = np.array([t[k, :] for t in true_list])  # (n_trials, n_ch)
        pred_k = np.array([p[k, :] for p in pred_list])
        ch_mean_k = true_k.mean(axis=0)  # (n_ch,)
        ss_res = ((true_k - pred_k) ** 2).sum(axis=0)  # (n_ch,)
        ss_tot = ((true_k - ch_mean_k) ** 2).sum(axis=0)  # (n_ch,)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1.0 - ss_res / np.where(ss_tot > 1e-12, ss_tot, np.nan)
        r2s[k] = float(np.nanmean(r2))
    return r2s


def state_norm_trajectory(Xf_list):
    """||x_t||₂ / ||x_0||₂ averaged across forecasts. Stable A → decay; if
    growing or oscillating, A modes near unit circle are problematic.
    Each Xf is (m, nx). Returns array of length m.
    """
    if not Xf_list:
        return np.array([])
    m = Xf_list[0].shape[0]
    norms = np.zeros((len(Xf_list), m))
    for i, Xf in enumerate(Xf_list):
        Xf = np.asarray(Xf)
        x0_norm = np.linalg.norm(Xf[0]) + 1e-12
        for k in range(m):
            norms[i, k] = np.linalg.norm(Xf[k]) / x0_norm
    return norms.mean(axis=0)


# Targets: forecast r ≥ 0.5 in first ~0.1 s, ≥ 0.1–0.2 overall.
y_pearson_te = per_horizon_pearson(fc_te["Y_true"], fc_te["Yf"])
z_pearson_te = per_horizon_pearson(fc_te["Z_true"], fc_te["Zf"])
y_rmse_te = per_horizon_rmse(fc_te["Y_true"], fc_te["Yf"])
z_rmse_te = per_horizon_rmse(fc_te["Z_true"], fc_te["Zf"])
y_r2_te = per_horizon_r2(fc_te["Y_true"], fc_te["Yf"])
z_r2_te = per_horizon_r2(fc_te["Z_true"], fc_te["Zf"])
xf_norm_te = state_norm_trajectory(fc_te["Xf"])

# A-eigenvalue stability snapshot
A_eigs_final = np.linalg.eigvals(idSys.A)
print(
    f"\nstability:  max|eig(A)| = {np.max(np.abs(A_eigs_final)):.4f}  "
    f"# complex pairs = {int(np.sum(np.abs(A_eigs_final.imag) > 1e-9) / 2)}  "
    f"# near-unit (|e|>0.99) = {int(np.sum(np.abs(A_eigs_final) > 0.99))}"
)
print(f"\nper-horizon test forecast (m={m} steps = {FORECAST_S:.1f}s):")
print(
    "  step | t (s) | Y r    | Z r    | Y R²    | Z R²    | Y rmse | Z rmse | ||Xf||/||X0||"
)
for k in range(m):
    print(
        f"  {k+1:4d} | {(k+1)/SFREQ:5.3f} | "
        f"{y_pearson_te[k]:+.3f} | {z_pearson_te[k]:+.3f} | "
        f"{y_r2_te[k]:+.3f} | {z_r2_te[k]:+.3f} | "
        f"{y_rmse_te[k]:.3f}  | {z_rmse_te[k]:.3f}  | "
        f"{xf_norm_te[k]:.3f}"
    )
n_first = max(1, int(0.1 * SFREQ))
print(f"\n  first 0.1 s mean Y r = {np.nanmean(y_pearson_te[:n_first]):.3f}")
print(f"  first 0.1 s mean Z r = {np.nanmean(z_pearson_te[:n_first]):.3f}")
print(f"  full {FORECAST_S:.1f}s mean Y r = {np.nanmean(y_pearson_te):.3f}")
print(f"  full {FORECAST_S:.1f}s mean Z r = {np.nanmean(z_pearson_te):.3f}")
print(f"  full {FORECAST_S:.1f}s mean Y rmse = {np.nanmean(y_rmse_te):.3f}")
print(f"  full {FORECAST_S:.1f}s mean Z rmse = {np.nanmean(z_rmse_te):.3f}")

# %% [markdown]
# ### Forecast Pearson r + RMSE over horizon (test)

# %%
fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
t_axis = (np.arange(m) + 1) / SFREQ
ax_r.plot(t_axis, y_pearson_te, color="#3a7bd5", marker="o", ms=3, label="Y")
ax_r.plot(t_axis, z_pearson_te, color="#d33", marker="o", ms=3, label="Z")
ax_r.axhline(0.5, color="0.5", ls=":", lw=0.8)
ax_r.axhline(0.0, color="k", lw=0.5)
ax_r.set_xlabel("forecast horizon (s)")
ax_r.set_ylabel("Pearson r")
ax_r.set_title("test forecast r over horizon")
ax_r.legend(loc="best", fontsize=9)

ax_e.plot(t_axis, y_rmse_te, color="#3a7bd5", marker="o", ms=3, label="Y")
ax_e.plot(t_axis, z_rmse_te, color="#d33", marker="o", ms=3, label="Z")
ax_e.axhline(1.0, color="0.5", ls=":", lw=0.8)
ax_e.set_xlabel("forecast horizon (s)")
ax_e.set_ylabel("normalized RMSE (per-channel)")
ax_e.set_title("test forecast RMSE over horizon")
ax_e.legend(loc="best", fontsize=9)
fig.tight_layout()
plt.show()

# Save figure + per-horizon metrics for headless / chained runs.
fig_dir = PROJECT_ROOT / "results" / "simplified_pipeline"
fig_dir.mkdir(parents=True, exist_ok=True)
tag = f"{PARTICIPANT}_S{SESSION}_Z={Z_SOURCE}_Y={Y_SOURCE}"
fig.savefig(fig_dir / f"forecast_horizon_{tag}.png", dpi=120)
np.savez(
    fig_dir / f"forecast_horizon_{tag}.npz",
    t_axis=t_axis,
    y_pearson=y_pearson_te,
    z_pearson=z_pearson_te,
    y_rmse=y_rmse_te,
    z_rmse=z_rmse_te,
)
print(f"saved {fig_dir / f'forecast_horizon_{tag}.png'}")

# %% [markdown]
# ### Forecast example (test trial 0)
#
# Past (`HISTORY_S` s): true. Future (`FORECAST_S` s): true (dashed) vs
# forecast (solid). One panel-set for Y, one for Z.


# %%
def plot_forecast_panel(
    full_arr, true_future, pred_future, names, h, m, sfreq, color, title, n_show=None
):
    if n_show is None:
        n_show = full_arr.shape[1]
    fig, axes = plt.subplots(n_show, 1, figsize=(8, 1.4 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    t_past = np.arange(h) / sfreq
    t_fut = np.arange(h, h + m) / sfreq
    for ax, ch_idx in zip(axes, range(n_show)):
        ax.plot(t_past, full_arr[:h, ch_idx], color="k", lw=1, label="past (true)")
        ax.plot(
            t_fut,
            true_future[:, ch_idx],
            color="k",
            lw=1,
            ls="--",
            label="future (true)",
        )
        ax.plot(t_fut, pred_future[:, ch_idx], color=color, lw=1, label="forecast")
        ax.set_ylabel(names[ch_idx], fontsize=7)
        if ch_idx == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.show()


trial0 = test_trials[0]
fcst_tag = f"forecast — b{trial0['block']} t{trial0['trial']} (stim={trial0['stim']})"
plot_forecast_panel(
    Y_test[0],
    fc_te["Y_true"][0],
    fc_te["Yf"][0],
    selected_y,
    h,
    m,
    SFREQ,
    "#3a7bd5",
    f"Y/ECoG forecast — {fcst_tag}",
)
plot_forecast_panel(
    Z_test[0],
    fc_te["Z_true"][0],
    fc_te["Zf"][0],
    selected_z,
    h,
    m,
    SFREQ,
    "#d33",
    f"Z forecast — {fcst_tag}",
)

# %% [markdown]
# ## DBS classification on PSID latents (Xp) — CSP → LDA
#
# Epoch each trial's `Xp` with `EPOCH_LEN_S` windows + `EPOCH_OVERLAP` overlap.
# Each epoch is a `(ep_len, nx)` slice of the latent trajectory — kept as a
# multivariate time-series, NOT mean-pooled.
#
# Pipeline (`utils/classification/pipeline.create_pipeline`):
#   1. transpose → MNE shape `(n_epochs, n_channels, n_samples)`
#   2. CSP `n_components=4`, `reg="ledoit_wolf"`, `log=True`
#      — finds 4 spatial filters in latent space that maximize the variance
#      ratio between DBS classes; outputs log-variance per filter (4 features)
#   3. StandardScaler
#   4. LDA (default solver, no manual shrinkage)
#
# Compared to the prior mean-pool LDA: latent dim is reduced by CSP (4 << nx)
# *before* LDA, so we no longer fit LDA in the full latent space. CSP also uses
# *time-series* structure within each epoch (variance of the projected signal),
# which mean-pool throws away.

# %%
ep_len = int(EPOCH_LEN_S * SFREQ)
step = max(1, int(ep_len * (1 - EPOCH_OVERLAP)))


def csp_epochs_from_trials(Xp_list, trial_dicts, ep_len, step):
    """Slice each trial's Xp into overlapping (ep_len, nx) chunks.
    Returns X (n_epochs, ep_len, nx), y, groups (block_id, trial_id).
    """
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
            X_eps.append(Xp[s : s + ep_len])
            y_eps.append(label)
            groups.append((td["block"], td["trial"]))
    return np.asarray(X_eps), np.asarray(y_eps), groups


X_fit, y_fit, _ = csp_epochs_from_trials(
    Xp_tr + Xp_val, train_trials + val_trials, ep_len, step
)
X_te, y_te, _ = csp_epochs_from_trials(Xp_te, test_trials, ep_len, step)
print(f"epochs: fit={X_fit.shape}, test={X_te.shape}  (n_epochs, ep_len, nx)")

pipe = create_pipeline(fs=SFREQ)
pipe.fit(X_fit, y_fit)
y_pred = pipe.predict(X_te)
ba = balanced_accuracy_score(y_te, y_pred)
cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
print(f"test balanced accuracy (CSP→LDA): {ba:.4f}")
print(f"confusion matrix [rows=true off/on, cols=pred off/on]:\n{cm}")

# %% [markdown]
# ### Alt: PCA → LDA
#
# Classical EEG/BCI baseline. Mean-pool per epoch → ``(n_epochs, nx)`` →
# StandardScaler → PCA(K) → LDA. Unsupervised dim reduction (variance-based)
# guards LDA against high-D overfitting. Compare BA against CSP→LDA above.

# %%
PCA_COMPONENTS = 8

X_fit_mp = X_fit.mean(axis=1)  # (n_epochs, nx) — drops time axis
X_te_mp = X_te.mean(axis=1)

pca_pipe = SkPipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("clf", LinearDiscriminantAnalysis()),
    ]
)
pca_pipe.fit(X_fit_mp, y_fit)
y_pred_pca = pca_pipe.predict(X_te_mp)
ba_pca = balanced_accuracy_score(y_te, y_pred_pca)
cm_pca = confusion_matrix(y_te, y_pred_pca, labels=[0, 1])
print(f"test balanced accuracy (PCA({PCA_COMPONENTS})→LDA): {ba_pca:.4f}")
print(f"confusion matrix [rows=true off/on, cols=pred off/on]:\n{cm_pca}")

# %% [markdown]
# ## Classification on **forecast** latents (Xf)
#
# Goal: forecasts are the primary target — classification on Xf is the
# downstream evidence that forecast quality preserves DBS-relevant structure.
#
# Per trial, slide a `(h, m)` window across time at stride `m` (non-overlap).
# Each window: feed past `h` to PSID, get `m`-step forecast Xf shape `(m, nx)`.
# Stack across trials with class labels → CSP/PCA-LDA on **future** latents.

# %%
FORECAST_STRIDE = m  # non-overlapping forecasts within each trial


def collect_forecast_chunks(Y_split, trial_dicts, h, m, stride, psid_model):
    X_eps, y_eps = [], []
    for Y, td in zip(Y_split, trial_dicts):
        T = Y.shape[0]
        if T < h + m:
            continue
        label = 1 if td["stim"] == "on" else 0
        for s in range(0, T - h - m + 1, stride):
            _, _, Xf_chunk = psid_model.forecast(m, Y[s : s + h])
            X_eps.append(np.asarray(Xf_chunk))
            y_eps.append(label)
    return np.asarray(X_eps), np.asarray(y_eps)


X_fit_xf, y_fit_xf = collect_forecast_chunks(
    Y_train + Y_val, train_trials + val_trials, h, m, FORECAST_STRIDE, psid_model
)
X_te_xf, y_te_xf = collect_forecast_chunks(
    Y_test, test_trials, h, m, FORECAST_STRIDE, psid_model
)
print(f"Xf chunks: fit={X_fit_xf.shape}, test={X_te_xf.shape}  (n_chunks, m, nx)")

# CSP→LDA on Xf
pipe_xf = create_pipeline(fs=SFREQ)
pipe_xf.fit(X_fit_xf, y_fit_xf)
y_pred_xf = pipe_xf.predict(X_te_xf)
ba_xf = balanced_accuracy_score(y_te_xf, y_pred_xf)
cm_xf = confusion_matrix(y_te_xf, y_pred_xf, labels=[0, 1])
print(f"Xf  CSP→LDA  BA = {ba_xf:.4f}")
print(f"Xf  CSP→LDA  CM:\n{cm_xf}")

# PCA(K)→LDA on Xf (mean-pool over forecast horizon)
X_fit_xf_mp = X_fit_xf.mean(axis=1)
X_te_xf_mp = X_te_xf.mean(axis=1)
pca_pipe_xf = SkPipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS)),
        ("clf", LinearDiscriminantAnalysis()),
    ]
)
pca_pipe_xf.fit(X_fit_xf_mp, y_fit_xf)
y_pred_xf_pca = pca_pipe_xf.predict(X_te_xf_mp)
ba_xf_pca = balanced_accuracy_score(y_te_xf, y_pred_xf_pca)
cm_xf_pca = confusion_matrix(y_te_xf, y_pred_xf_pca, labels=[0, 1])
print(f"Xf  PCA({PCA_COMPONENTS})→LDA  BA = {ba_xf_pca:.4f}")
print(f"Xf  PCA→LDA  CM:\n{cm_xf_pca}")

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
# ### CSP feature scatter (first 2 components, test set)
#
# Project each test epoch through CSP filters → log-variance features. Plot
# epochs in the (CSP-1, CSP-2) plane to see DBS class separation.

# %%
csp_step = pipe.named_steps["csp"]
X_te_mne = pipe.named_steps["transpose"].transform(X_te)
X_te_csp = csp_step.transform(X_te_mne)  # (n_epochs, n_components)
if X_te_csp.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, color, name in ((0, "#3a7bd5", "off"), (1, "#d33", "on")):
        mask = y_te == label
        if mask.any():
            ax.scatter(
                X_te_csp[mask, 0],
                X_te_csp[mask, 1],
                s=8,
                alpha=0.5,
                color=color,
                label=name,
            )
    ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("CSP-1 (log-var)")
    ax.set_ylabel("CSP-2 (log-var)")
    ax.set_title("test epochs in first 2 CSP components")
    fig.tight_layout()
    plt.show()
