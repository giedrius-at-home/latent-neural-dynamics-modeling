# Feature selection methods for PSID/DPAD/VARMA training

This document describes the **Mazzanti MIQ** mRMR-style feature selection
method used across all three selection steps in the latent-neural-dynamics
pipeline (behavioural ECoG-Y, laplacian ECoG-Y, laplacian LFP-Z). One
recipe; only the `(features, targets)` pair varies per step.

MIQ is **m**inimum-Redundancy Maximum-Relevance with mutual-information
relevance and Pearson-correlation redundancy combined via the FCQ quotient
score. Multi-target aggregation uses per-target vote-rank with pool=3K to
preserve target-specific information that a single mean-MI ranking would
average away.

---

## 1. Where the method is used

A single recipe — **Mazzanti MIQ** — is applied symmetrically across all
three selection steps in the pipeline. Only the `(features, targets)` pair
changes per step:

| Selection step | Features (X) | Targets (Y) | Candidates → picks |
|---|---|---|---|
| ECoG-Y (behavioural mode) | 60 ECoG bands | 2 behaviour dims | 60 → 8 |
| ECoG-Y (laplacian mode)   | 60 ECoG bands | 15 LFP_14-16 bands | 60 → 8 |
| LFP-Z (laplacian mode)    | 15 LFP bands  | 60 ECoG bands     | 15 → 8 |

Behavioural mode: ECoG-Y picks are selected against the 2 behaviour
targets. Z output is the 2-dimensional behaviour vector itself — no
selection needed.

Laplacian mode: ECoG-Y picks are selected against the 15 LFP bands
(matching the `ECoG → LFP` training task). LFP-Z picks are selected by
**ECoG-predictability** — i.e. which LFP bands the cortical input set
shares the most information with — vote-rank aggregated across the 60
ECoG targets. This pre-filters the prediction target to the LFP bands
that are actually predictable from cortex, avoiding noise/background
bands that would dilute the model's training signal.

All three steps share the same MIQ implementation (MI relevance via
sklearn k-NN, Pearson redundancy, FCQ quotient, per-target vote-rank with
pool=3K). K = 8 picks per step. Picks are cached and consumed by
`pipeline_psid.py`, `pipeline_dpad.py`, and `pipeline_varma.py` so all
three frameworks train on byte-identical Y/Z column sets.

---

## 2. Method A — Mazzanti MIQ (ECoG → behaviour / LFP)

### 2.1 What MIQ stands for

**M**utual-**I**nformation **Q**uotient. The mRMR family has four canonical
variants (Peng et al. 2005):

| Variant | Relevance | Redundancy | Score |
|---|---|---|---|
| MID | MI(feat, target) | mean MI(feat, sel) | rel − red |
| MIQ | MI(feat, target) | mean MI(feat, sel) | rel / red |
| FCD | F-statistic       | mean Pearson |r|  | rel − red |
| FCQ | F-statistic       | mean Pearson |r|  | rel / red |

Our use is a **hybrid**: MI relevance (like MID/MIQ) but Pearson redundancy
(like FCD/FCQ). The Mazzanti library calls this configuration when you pass
`relevance=<MI callable>, redundancy="c", denominator="mean"`. Common
shorthand in the library docs is "MIQ-style FCQ" — quotient form, MI
relevance, Pearson redundancy.

### 2.2 Algorithm (greedy, K iterations)

```
Initialize: selected = []

For k = 1..K:
    For each candidate i not in selected:
        rel_i = MI(feature_i, target)            # k-NN MI estimator
        red_i = mean_{j in selected} |r(feat_i, feat_j)|   # 0 if k=1
        score_i = rel_i / (red_i + epsilon)
    selected.append(argmax_i score_i)
```

The first pick is the maximum-relevance feature (no selected set yet).
Subsequent picks are penalized for redundancy with already-chosen features.
Quotient form (vs additive difference) means a feature with low redundancy
gets a multiplicative score advantage — favors orthogonal selections.

### 2.3 Mutual information estimator

Mazzanti's library accepts a custom relevance function. Our wrapper passes
sklearn's `mutual_info_regression`:

```python
from sklearn.feature_selection import mutual_info_regression

def rel_fn(X, y):
    return mutual_info_regression(X, y, n_neighbors=3, random_state=0)
```

The estimator uses Kraskov-Stögbauer-Grassberger (KSG) k-nearest-neighbor
mutual information. With `n_neighbors=3`, MI for one feature is:

```
MI(X, Y) = ψ(N) - <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(k)
```

where ψ is the digamma function, k=3 is the neighbor count, and (n_x, n_y)
are counts of points within the k-NN distance in the marginal projections.
The KSG estimator is unbiased asymptotically, but variance scales as 1/√n —
which matters for our **sampling cap**.

### 2.4 Multi-target aggregation: vote-rank

Behaviour has 2 targets (`tracing_velocity_x`, `tracing_acceleration_magnitude`).
Mazzanti's `mrmr_regression` is single-target. We extend with **per-target
vote-rank**:

1. Run mRMR per target independently with an oversampled pool size:
   `pool = min(3*K, n_features) = 24` (when K=8, n_features=60).
2. Each target produces an ordered list of `pool` features.
3. Each feature receives `(pool - rank)` votes per target; rank-1 contributes
   24 votes, rank-2 contributes 23, …, rank-24 contributes 0.
4. Sum votes across targets, take the top-K by total vote count.

This is implemented in `scripts/_pipeline_common.py:185-194`:

```python
n_workers = min(Y.shape[1], 8)
with ThreadPoolExecutor(max_workers=n_workers) as ex:
    all_picks = list(ex.map(_picks_for_target, range(Y.shape[1])))

vote_rank = pd.Series(0.0, index=feat_cols)
for picks in all_picks:
    for r, feat in enumerate(picks):
        vote_rank[feat] += max(pool - r, 0)
ranked = vote_rank.sort_values(ascending=False).index.tolist()
top_k = ranked[:K]
```

**Why pool=3K and not just K?** Oversampling preserves information about
features that miss top-K for one target but stay relevant in the extended
ranking. With pool=K, a feature ranked 9th for target_1 contributes 0 votes
even if it's rank-1 for target_0 — the algorithm cannot tell "great for one
target only" apart from "irrelevant to both". With pool=3K, the rank-9 entry
contributes ~16 votes (from a pool of 24), so the "great for one only" case
keeps a strong total while a "mid for both" case earns ~12+12=24 — a clean
ordering emerges.

### 2.5 Sampling considerations

Train data for one session is ~100k-150k samples (concatenated across train
blocks). MI estimation with k-NN is O(n²) in the worst case, so production
caps at **30,000 samples**, randomly subsampled with `seed=0`:

```python
rng = np.random.default_rng(0)
if len(X_df) > max_samples:  # max_samples = 30_000
    idx = rng.choice(len(X_df), max_samples, replace=False)
    X_df = X_df.iloc[idx].reset_index(drop=True)
    Y = Y[idx]
```

**Why random and not chronological?** Each train session contains alternating
DBS-on/DBS-off blocks. Chronological truncation (taking the first N samples)
would skew the sample toward the earlier DBS state. Random subsampling
preserves the on/off mix → MI estimates are not condition-confounded.

Empirical sweep on PDI1_S2 LFP MIQ shows that under-sampling shifts picks
substantially — the 12k-cap setup picks a different rank-4..8 tail than
the 30k-cap setup. Production uses 30k.

### 2.6 Code locations

- **Implementation**: `scripts/_pipeline_common.py:135-209`
  (`mrmr_top_k_from_diagnostic`)
- **Cache**: `configs/diagnostic/mrmr_picks.yaml` keyed by
  `sessions[<label>][<family>]` where family ∈ {`ecog`, `laplacian`, `lfp_z`}
- **Pipeline use**: `pipeline_psid.py:484` calls `load_mrmr_picks_from_yaml`
  → falls back to `mrmr_top_k_from_diagnostic` on cache miss

### 2.7 Recompute MI relevance for figures

The MI relevance per feature (the heatmap gradient in sec2a Fig 44) is
precomputed by `scripts/precompute_mi_relevance.py` and stored at
`results/diagnostic/{P}_{S}_psid_spectra/mi_relevance/{family}.parquet`.
Three families:

| Family    | Features            | Targets          | Used in Fig 44 |
|---|---|---|---|
| `ecog`      | 60 ECoG bands       | 2 kinematics     | Panel A gradient |
| `laplacian` | 60 ECoG bands       | 15 LFP bands     | Panel B gradient |
| `lfp_z`     | 15 LFP bands        | 2 kinematics     | Panel C gradient |


## 4. Limitations to acknowledge

1. **Asymmetric metric pair in MIQ**: MI relevance + Pearson redundancy is
   inconsistent. Pure MID (MI relevance + MI redundancy) would be cleaner;
   we use the hybrid because Pearson redundancy is fast and stable on
   30k samples, while MI on every (i, j) feature pair is O(F²·n) and noisy.
2. **No bootstrap stage on MIQ.** A single deterministic pass per
   `(features, targets)` pair gives the picks; we have not measured
   stability under trial-level resampling. The mrmr-selection library
   does not implement bootstrap; adding one would require wrapping the
   per-target call inside an outer loop over subsamples.
3. **LFP-Z selection target is the ECoG input set, not behaviour.** The
   laplacian pipeline trains `ECoG → LFP`; the LFP-Z bands are
   selected by predictability (which LFP bands the cortical input set
   carries information about), not by behavioural relevance. This
   matches the model's training task and avoids injecting an off-task
   signal into the prediction target.

---

## 5. Reproducibility — how to recompute everything

```bash
# Mazzanti MIQ picks for ECoG-Y (behavioural + laplacian families).
# Outputs configs/diagnostic/mrmr_picks.yaml.
python scripts/regenerate_mrmr_picks_yaml.py        # ECoG families

# Mazzanti MIQ picks for LFP-Z (laplacian mode predictability — features=
# LFP_15, targets=ECoG_60, vote-rank). Outputs configs/diagnostic/mrmr_picks.yaml
# under family `lfp_z_predictability`.
python scripts/precompute_lfp_z_predictability_picks.py

# MI relevance for Fig 44 heatmap gradients. Outputs
# results/diagnostic/{P}_{S}_psid_spectra/mi_relevance/{ecog,laplacian,lfp_z}.parquet
python scripts/precompute_mi_relevance.py
```

All scripts read precomputed splits from `configs/splits/{P}_S{S}.yaml` and
the resampled per-block parquets at
`resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band/`.

Typical wall-time per session × family:

| Computation | Time |
|---|---|
| Mazzanti MIQ (60 ECoG features, 2 or 15 targets, 30k samples) | ~10 s |
| Mazzanti MIQ (15 LFP features, 60 ECoG targets, 30k samples)  | ~3 min |
| MI relevance precompute (60 features, 30k samples)            | ~7 s |

---

## 6. Open questions

- **Bootstrap stability filter for MIQ?** The pipeline produces one
  deterministic pick set per `(features, targets)` pair; trial-level
  resampling could quantify which picks are stable across folds vs which
  are borderline. Wrapping the per-target call inside an outer 5-fold
  loop would surface a "core" subset of picks that are consistent across
  trial subsamples vs borderline picks that flip under resampling.
- **Lagged MIQ?** Combining the time-lag awareness of cross-correlation
  with MI relevance is an open avenue. Computationally heavy (k-NN MI
  at each of 21 lags × multiple targets) but conceptually appealing for
  signals with known physiological delays.
- **Per-session method choice?** Currently fixed: MIQ everywhere. A more
  flexible pipeline could pick the method per session based on which
  one gives higher cross-validated downstream classification accuracy.

---

## 7. References

- Peng, H., Long, F., Ding, C. (2005). *Feature Selection Based on Mutual
  Information: Criteria of Max-Dependency, Max-Relevance, and
  Min-Redundancy*. IEEE PAMI.
- Mazzanti, S. (2021). *mrmr-selection*: pure-Python MIQ/FCQ implementation,
  https://github.com/smazzanti/mrmr.
- Kraskov, A., Stögbauer, H., Grassberger, P. (2004). *Estimating mutual
  information*. Physical Review E. — k-NN MI estimator used by sklearn's
  `mutual_info_regression`.
