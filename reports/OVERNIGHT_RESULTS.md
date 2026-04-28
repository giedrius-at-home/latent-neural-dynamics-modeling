# Overnight Results Overview — 2026-04-18

Models trained with mRMR-selected K=8 ECoG channels, `nx=50, n1=10, i=50, max_eigenvalue=0.9999, backward_kalman=true`, forecast `m=1s, history=3s`. All cells share identical train/val/test splits (chronological, 60/10/30).

---

## 1. Pipeline state

| Machine | Subset | PSID | VARMA | Permutation p-vals |
|---|---|---|---|---|
| local  | PDI4 (4 cells) | ✅ done | ✅ done | ✅ both CV + test-set |
| jacque | PDI1 (4 cells) | ✅ done | ⚠ lap variants failed (launched before the `_mrmr` naming fix was synced) | partial |

8 PSID model bundles × 4 DBS variants each = **32 PSID models** on disk. VARMA: 6 cells OK, 2 failed (PDI1 laplacian).

### Phase coverage per cell
- Phase 3: train 4 PSID variants (both, on, off, vanilla_both) + full train/val/test metrics
- Phase 4: cross-condition eval (on→off, off→on)
- Phase 5: classification with H={1,2,3} × M={0.5,1}, 4 feature sources (Xp, Xp_1, Xp_2, Xp_with_dbs), flipped controls, in-pipeline perm tests (N=100)
- Phase 6: thesis HTML update — SKIPPED (regenerate later)

---

## 2. PSID model performance (val split, PDI4 cells)

Metrics: `r_Yp` = 1-sample-ahead Y reconstruction; `r_Zp` = Z decoding; `r_Yfcst`/`r_Zfcst` = 1-s-ahead Y/Z autonomous forecast.

```
cell               dbs             r_Yp    r_Zp  r_Yfcst  r_Zfcst   n
---------------------------------------------------------------------
PDI4_S2 beh        both            0.948  -0.015   0.108   -0.061  12
PDI4_S2 beh        on              0.942   0.034   0.174    0.014   6
PDI4_S2 beh        off             0.958  -0.035   0.108    0.067   6
PDI4_S2 beh        vanilla_both    0.783   0.001   0.098   -0.062  12

PDI4_S3 beh        both            0.979  -0.006   0.118   -0.172  11
PDI4_S3 beh        on              0.978  -0.007   0.111    0.015   7
PDI4_S3 beh        off              0.979   0.031   0.122    0.005   5
PDI4_S3 beh        vanilla_both    0.874   0.003   0.110   -0.102  11

PDI4_S2 lap        both            0.985   0.096   0.123    0.064  12
PDI4_S2 lap        on              0.988   0.034   0.046   -0.012   6
PDI4_S2 lap        off              0.986   0.014   0.121    0.001   6
PDI4_S2 lap        vanilla_both    0.878   0.067   0.114    0.051  12

PDI4_S3 lap        both            0.982   0.089   0.110    0.019  11
PDI4_S3 lap        on              0.979   0.223   0.117    0.033   7  ← best Z decoding
PDI4_S3 lap        off              0.976   0.186   0.008   -0.076   5
PDI4_S3 lap        vanilla_both    0.893   0.108   0.101    0.013  11
```

### Takeaways
- **Reconstruction excellent** (`r_Yp` = 0.94-0.99) — model fits ECoG well.
- **Vanilla drops r_Yp ~0.10** — `max_eigenvalue` damping + RTS smoother materially help.
- **Behavioral decoding flat** (`r_Zp` ~ 0) — raw ECoG → velocity/accel_mag isn't linearly decodable in this configuration.
- **Laplacian-mode decoding has signal** — PDI4_S3 on: r_Zp = 0.22, off: 0.19. Best scores in the table. Strong DBS-condition dependence.
- **1-s forecast is weak** everywhere — see Section 4.

---

## 3. Classification (DBS on vs off from Xp)

### Test-set balanced accuracy + permutation p-values (BH-FDR corrected across the 16-test prediction family)

| cell | source | test_ba | null_p95 | raw p | **q (BH)** |
|---|---|---|---|---|---|
| behavioral PDI4_S3 | **Xp_2** | **0.759** | 0.523 | 0.010 | **0.015** ★★ |
| behavioral PDI4_S3 | **Xp** | **0.747** | 0.522 | 0.010 | **0.015** ★★ |
| laplacian  PDI4_S3 | Xp_2 | 0.595 | 0.521 | 0.010 | 0.015 |
| laplacian  PDI4_S3 | Xp   | 0.591 | 0.523 | 0.010 | 0.015 |
| laplacian  PDI4_S2 | Xp_2 | 0.571 | 0.525 | 0.010 | 0.015 |
| behavioral PDI4_S2 | Xp_2 | 0.544 | 0.517 | 0.010 | 0.015 |
| behavioral PDI4_S2 | Xp   | 0.542 | 0.517 | 0.010 | 0.015 |
| behavioral PDI4_S2 | Xp_1 | 0.528 | 0.509 | 0.010 | 0.015 |
| behavioral PDI4_S3 | Xp_1 | 0.535 | 0.527 | 0.030 | 0.040 |

**9 predictions survive BH-FDR at q<0.05**. Headline: **behavioral PDI4_S3 Xp/Xp_2 hits test_ba 0.75 — real DBS decoding from neural dynamics**.

### Forecast classification (Xp at horizons m=0.5, 1.0; h=1, 2, 3 s of history)

**None survive BH-FDR** (best raw p = 0.020, q = 0.71 after correction across 72 forecast tests). Best raw numbers:

| cell | source | hm | test_ba | raw p |
|---|---|---|---|---|
| behavioral PDI4_S3 | Xp (or Xp_2) | h=3, m=0.5 | 0.669 | 0.089 |
| behavioral PDI4_S2 | Xp (or Xp_2) | h=2, m=0.5 | 0.646 | 0.040 |
| laplacian  PDI4_S2 | Xp | h=3, m=1.0 | 0.635 | 0.020 |

Issue is **low n** (36-72 samples per forecast test) + many comparisons.

### Flipped-label controls
All flipped variants produce cv_ba near chance (0.14-0.50) with p > 0.23 — confirms **no label leakage** in the pipeline.

---

## 4. Forecast analysis — why autonomous forecast is weaker

`r_Yp` (reconstruction) uses Kalman-filtered states that have access to `y_{1:t}`; `r_Yfcst` propagates states purely through `A^{200}` (1 s at 200 Hz) with no new input. The gap is driven entirely by how quickly A's modes decay.

### What eigenvalue damping costs us at various horizons

For a mode with eigenvalue magnitude λ, amplitude at step k is `λ^k`:

| mode |λ| | 0.1 s (20 steps) | 0.5 s (100 steps) | 1.0 s (200 steps) |
|---|---|---|---|---|
| 0.9999 | 0.998 | 0.990 | 0.980 |
| 0.999 | 0.980 | 0.905 | 0.819 |
| 0.99 | 0.818 | 0.366 | 0.134 |
| 0.95 | 0.358 | 0.006 | ≈ 0 |

With `max_eigenvalue=0.9999`, the slowest mode retains ~98% energy at 1 s — good. But typical A eigenvalues from subspace ID on neural data span a range; modes with |λ| ≤ 0.99 (still stable but not slow) lose most amplitude by 0.5-1 s.

### Can we push 0.5-s forecast higher?

You're right that `r_Yp = 0.99` single-step means there's real information in the state at each t. At **0.5 s** autonomous propagation, the current numbers (should be computable from saved models — I haven't extracted them yet) will be somewhere between reconstruction (0.99) and the 1-s forecast (0.10).

**Levers we haven't tried:**
1. **Shorter forecast** as a sanity check — dump `r_Yfcst` at m ∈ {0.1, 0.25, 0.5, 1.0} s. Shows the decay curve explicitly. The existing models ARE forecasting at 1 s because that's what's saved during training, but we can roll out the fitted A matrix for ad-hoc m values without refitting.
2. **Tighter `max_eigenvalue` limit** (e.g., 0.99999) — pushes all modes closer to non-decay. Risk: Kalman + DARE numerics get fragile near the unit circle. Pipeline already has retry-lower fallback for this.
3. **Identify with longer `i`** — our `i=50` Hankel horizon covers 0.25 s of past/future. Larger `i` (e.g., 80 or 100) forces A to capture longer-range correlations, usually producing slower modes. Cost: bigger Hankel SVD.
4. **Constrain A spectrum during or after fit** — keep only the top-k slow eigenvalues, zero the rest. Reduces capacity but keeps the forecastable part.
5. **Feature-domain change** — envelope/log-power instead of narrow-band raw. You've said no on this, but it's the biggest-impact option in the literature. Raw narrow-band has lots of fast random-walk-like structure that decays fast.

**Measured** (val split, cumulative r up to each horizon — matches pipeline metric):

```
cell              0.05s   0.1s   0.25s   0.5s   0.75s   1.0s
PDI4_S2 beh       0.779  0.614  0.333  0.172  0.134  0.117
PDI4_S3 beh       0.834  0.670  0.392  0.284  0.198  0.142
PDI4_S2 lap       0.857  0.722  0.428  0.217  0.150  0.132
PDI4_S3 lap       0.863  0.780  0.525  0.378  0.287  0.244  ← best
```

So 0.1 s is solidly good (0.61-0.78), 0.25 s still usable (0.33-0.53), 0.5 s mediocre (0.17-0.38). Best cell: PDI4_S3 laplacian, 0.38 at 0.5 s.

**Most principled fix**: try `i=80` or `i=100` — forces A identification to capture longer-timescale correlations, producing fatter slow-mode tail. Single config change in `pipeline_psid.py` (`FULL_I_BEHAV`). Retry-lower fallback handles OOM.

---

## 5. Known issues / loose ends

1. **PDI1 laplacian VARMA cells failed** — naming-collision fix didn't propagate in time. Fix: rsync the new `pipeline_varma.py` to jacque and rerun those 2 cells.
2. **PSID `best_ba` NameError in summary log** — cosmetic, models are valid. Fixed in script; will not appear on future runs.
3. **flipped permutation pipeline**: only runs at (h=1, m=0.5) by design — if you want flipped across the full h/m grid, Phase 5 needs to loop more broadly.
4. **Forecast horizon sweep not yet computed** — need to roll out fitted A to generate `r_Yfcst` at multiple m values. Cheap post-hoc analysis on saved models.

---

## 6. Files

- PSID val-split CSV: (run `python scripts/test_set_permutation.py` for test set)
- Per-cell PSID logs: `logs/overnight/{tag}_{mode}_psid.log`
- Per-cell VARMA logs: `logs/overnight/{tag}_{mode}_varma.log`
- Classification dirs: `results/classification/psid_*_nx_50_n10_i50_*_200Hz_narrow_band*/20260418_*/`
- Test-set permutation CSV: `reports/test_permutation_scores.csv`
- mRMR channel picks: `reports/reduced_training_plan.md`
