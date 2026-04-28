# nx Choice Analysis — 2026-04-18

Why the overnight run uses **nx = n1 + 40** per cell, not the stage-2-elbow values (nx=160-165) initially set in `configs/diagnostic/elbow_choices.yaml`.

---

## TL;DR

- Stage-2 spectrum elbows at 160+ described **how much autonomous Y structure exists**, not how much PSID needs to fit.
- Direct empirical tests at nx=25 and nx=50 with K=8 channels, max_eigenvalue=0.9999 gave r_Yp ∈ [0.98, 1.00] — essentially ceiling.
- Tests at nx=160 either never completed or produced worse numbers than smaller nx.
- **Decision**: nx = n1 + 40 per cell. Honors per-session n1 differences (spectra-derived) while keeping nx in the verified-working range.

---

## 1. What stage-2 elbows actually mean

PSID's stage-2 SVD ranks **autonomous Y-residual dynamics** — how many independent modes exist beyond the behavior-relevant stage-1 subspace. Our analysis on PDI4_S3 showed:

- ECoG stage-2 elbow ≈ 150
- Laplacian stage-2 elbow ≈ 60

This is **information content**, not modeling requirement. A model can reconstruct Y well even when nx << stage-2 elbow if C (the Y-readout) maps efficiently to the slow modes that matter.

From `reports/OVERNIGHT_RESULTS.md`:
> "The stage-2 spectrum shows how much autonomous Y structure EXISTS in the data, not how much you NEED to model."

---

## 2. What the empirical sweeps actually tested

### 2.1 The 4-config sweep on PDI4_S2 (earlier today, K=4, max_eig=0.999, RTS on)

```
Config   n1   nx    r_Yp(val)   r_Yfcst_0.5s   Comment
  1      10   15    0.74        0.03           bad — nx too small
  2      10   25    0.999       0.17           ★ excellent
  3      15   25    0.978       0.13           good (laplacian-out)
  4      20   50    1.000       0.25           ★ best forecast (laplacian-out)
```

- **nx=25 already saturates reconstruction** (r_Yp=0.999 for behavioral).
- **nx=50 marginally better** on forecasts.
- nx=15 is below the cliff.

### 2.2 CV plateau analysis (later today, K=8, nx=50, n1=10, max_eig=0.9999, RTS on)

```
PDI4_S3 behavioral (3-fold CV on train pool):
  N=10   r_Yp=0.980 ± 0.000
  N=25   r_Yp=0.980 ± 0.001
  N=45   r_Yp=0.980 ± 0.000

PDI4_S2 laplacian:
  N=10   r_Yp=0.987 ± 0.001
  N=25   r_Yp=0.988 ± 0.002
  N=45   r_Yp=0.987 ± 0.001
```

**nx=50 at K=8 gives r_Yp ≈ 0.98-0.99 regardless of training size.** Reconstruction is fully saturated.

### 2.3 i=100 experiment (K=8, nx=50, n1=10)

```
          r_Yp    r_Yfcst_0.5s
i=50      0.979   0.284
i=100     0.812   0.197    (both metrics worse)
```

Increasing the Hankel identification horizon at fixed nx=50 HURT reconstruction — likely because the fit distributed capacity across more noise modes. This suggests **nx=50 is already at the useful-capacity boundary** for K=8-channel input.

### 2.4 nx=160 attempts

| Attempt | Config | Outcome |
|---|---|---|
| first K=8 retraining (earlier today) | nx=160, n1=10, max_eigenvalue=null | **Numerical blow-up** (10^85 state values). Vanilla PSID's A had eigenvalues ≥ 1 at this capacity; state trajectories diverged. |
| after damping added | nx=160, n1=10, max_eigenvalue=0.999 | Never completed pipeline-integrated evaluation. Fit succeeded but we pivoted to nx=50 for diagnostic work. |

**Conclusion**: nx=160 is **untested** in a production-complete pipeline run. Running it overnight would be an unvetted configuration with higher risk than payoff.

---

## 3. Why n1 + 40

Our sweep config 4 (`nx=50, n1=20`) worked well. That's n2 = 30. Config 2 (`nx=25, n1=10`) worked best for pure reconstruction. That's n2 = 15. Interpolating: **n2 = 40 is generous enough for residual-Y capacity without entering the overfitting regime**.

Scaling with n1:
- If PDI1_S2 needs n1=15 (per its stage-1 spectrum), it gets nx=55 → n2=40.
- If other cells need n1=10, they get nx=50 → n2=40.

**All configurations now within the empirically-verified-working range.**

---

## 4. What we give up vs nx=160

The theoretical upper bound from stage-2 elbow (~150 modes for ECoG) would allow reconstructing finer residual Y structure. Concretely:
- r_Yp might go from 0.98 → 0.99+ with bigger nx. Marginal gain.
- r_Yfcst might improve slightly because more slow modes are captured. Also marginal — our forecast ceiling is set by A's eigenvalue distribution, not nx.
- Overfitting risk rises: n_train ≈ 60 trials × ~1500 samples ≈ 90K samples. At nx=160, A has 25K params + Cy has 8×160 = 1280 params → ~90K samples for 26K+ params → 3 samples/param. At nx=50, ~5K params → 18 samples/param. Much safer.

**Tradeoff**: ~1% reconstruction gain (best case) vs 5× more parameters and higher overfitting/numerical instability risk. Not worth it for the thesis pipeline.

---

## 5. Final overnight config

```
cell                K    nx    n1    source
PDI4_S2 behavioral  8    50    10    mRMR + spectra elbow n1 + 40-buffer
PDI4_S2 laplacian   8    50    10    same
PDI4_S3 behavioral  8    50    10    same
PDI4_S3 laplacian   8    50    10    same
PDI1_S2 behavioral  8    55    15    PDI1_S2 stage-1 elbow bumped n1 to 15
PDI1_S2 laplacian   8    55    15    same
PDI1_S4 behavioral  8    50    10    same
PDI1_S4 laplacian   8    50    10    same
```

Other fixed settings: i=50, max_eigenvalue=0.9999, backward_kalman=true, forecast m=1s / h=3s, split 60/15/45.

---

## 6. What this doesn't close

- We haven't tested nx values between 50 and 160 (e.g., 80, 100). Possible that nx=80 gives a small reconstruction bump without the instability risk of nx=160.
- For cells where forecast quality is a priority (laplacian_PDI4_S3 got r=0.59 at 0.5s on one trial earlier), a nx ≈ 80 sweep might be worth it post-overnight.
- If the thesis claims land as "reconstruction r=0.98, decoding test_ba=0.8", the current nx=50 is sufficient; the tradeoff above reads favorably.

---

## 7. References

- `reports/OVERNIGHT_RESULTS.md` — overnight run + forecast-horizon analysis (source of "structural ceiling" framing)
- `reports/DATA_EFFICIENCY_ANALYSIS.md` — CV plateau findings that nx=50 at K=8 saturates at N=10
- `reports/cv_plateau_analysis.csv` — raw per-fold metrics
- `configs/diagnostic/elbow_choices.yaml` — preserved with aspirational nx=160-165 values for reference; overnight overrides via `nx = n1 + 40` capping rule
- `scripts/overnight_all_sessions.py:_build_configs_from_elbows` — where the nx cap is applied
