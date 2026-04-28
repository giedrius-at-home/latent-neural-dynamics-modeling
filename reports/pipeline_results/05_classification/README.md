# Stage 5 — Classification (DBS-OFF vs DBS-ON decoding)

## Job

Train an LDA classifier to decode DBS state from the learned latent states.
Decodability of DBS from **neural latents** is the main thesis result: if the
latent subspace truly captures DBS-relevant dynamics, LDA balanced accuracy
(BA) should be well above chance (0.5) and well above the permutation null.

## Protocol

For every **dbs_both** variant (PSID, DPAD, VARMA — single-condition on/off
variants have a single DBS class in their test split, so they are **not**
classified):

1. **Prediction mode** (epochs of current latent state Xp):
   - 4 feature sources: `Xp`, `Xp_1`, `Xp_2`, `Xp+DBS` (latent with DBS covariate)
   - 5-fold cross-validation grid-search on LDA hyperparameters
   - Final test BA on held-out epochs

2. **Forecast mode** (epochs of forecasted latent states):
   - h × m grid: h ∈ {1.0, 2.0, 3.0} s history, m ∈ {0.5, 1.0} s forecast window → 6 cells
   - 4 feature sources × 6 cells = 24 forecast classifiers per variant
   - In practice, pipeline picks the CV-best (h, m) for permutation; stores results for the rest of the grid as available

3. **Negative controls**:
   - **Flipped** — labels shuffled within fold; BA should be ≈ 0.5
   - **Permutation test** — block-permuted labels, n=1000, reports p-value on observed BA

## Producer

```bash
python -m classification.compute --config classification/setups/<variant>.yaml
```

Driven by Phase 5 of `pipeline_psid.py` (for PSID) and equivalents in
`pipeline_dpad.py` / VARMA scripts.

## Config types per variant

For `<variant>` = `psid_behavioral_PDI4_2_nx_50_n10_i100_dbs_both_200Hz_narrow_band`
(example), the full config set is:

```
classification/setups/<variant>.yaml                 # base Xp prediction
classification/setups/<variant>_xp_1.yaml            # Xp_1 feature
classification/setups/<variant>_xp_2.yaml            # Xp_2 feature
classification/setups/<variant>_xp_with_dbs.yaml     # Xp + DBS covariate
classification/setups/<variant>_flipped.yaml         # flipped-label control
classification/setups/<variant>_perm.yaml            # permutation for Xp
classification/setups/<variant>_xp_1_perm.yaml       # permutation for Xp_1
classification/setups/<variant>_xp_2_perm.yaml       # " Xp_2
classification/setups/<variant>_xp_with_dbs_perm.yaml# " Xp+DBS
classification/setups/<variant>_flipped_perm.yaml    # flipped permutation
```

## Output layout

Per variant, under `results/classification/<variant>/<run_ts>/`:

```
LDA_Xp_prediction.pklz           # full classifier result dict (base)
LDA_Xp_1_prediction.pklz
LDA_Xp_2_prediction.pklz
LDA_Xp_with_dbs_prediction.pklz
h1.0_m0.5/
  LDA_Xp_forecast.pklz           # forecast-mode classifier (h=1.0, m=0.5)
  LDA_Xp_1_forecast.pklz
  LDA_Xp_2_forecast.pklz
  LDA_Xp_with_dbs_forecast.pklz
h1.0_m1.0/  ...
h2.0_m0.5/  ...
h2.0_m1.0/  ...
h3.0_m0.5/  ...
h3.0_m1.0/  ...
```

(The `.pklz` extension here is shorthand — actual files are serialised binary
under their real `.pkl` extension.)

Each result dict has keys:

```
['accuracy', 'balanced_accuracy', 'best_cv_score', 'best_params',
 'best_pipeline', 'confusion_matrix', 'cv_method', 'f1', 'fold_results',
 'fpr', 'grid_search_results', 'n_combinations_tested', 'n_splits',
 'precision', 'recall', 'roc_auc', 'test_results', 'tpr',
 'y_pred', 'y_proba', 'y_true']
```

Flipped + permutation results land under sibling variant dirs:

```
results/classification/<variant>_flipped/<ts>/LDA_Xp_flipped.<ext>
results/classification/<variant>_perm/<ts>/...  # with permutation_results key
```

## Coverage summary (as of 2026-04-22)

| Family | dbs_both base | dbs_both flipped | dbs_both perm |
|---|---|---|---|
| PSID behavioral ×4 | ✓ 4/4 | ✓ | ✓ |
| PSID laplacian ×4 | ✓ 4/4 | partial | partial |
| DPAD behavioral ×4 | ✓ 4/4 (jacque-rsynced PDI1 included) | PDI4 2/2, PDI1 in flight on jacque | TODO |
| VARMA | — | — | — (VARMA is reference; no classification) |

### Gaps that remain (as of now)

- DPAD PDI1_S4 flipped is **currently running** on jacque (PID 1973032, ~2h elapsed); once it lands the flipped coverage for PDI1 is complete.
- DPAD permutation tests for all 4 cells have configs but haven't been launched.
- No on/off DPAD classifications — **intentional** (single-class test split).

## Feeds into stage 6 (figures)

Classification outputs feed `thesis_sec5_classification` which emits the BA
bar plots + confusion matrices. `classification_f1_*` modules under
`notebooks/thesis_lib/` do the heavy lifting.

## Symlinks

- `classification_root/` → `../../../results/classification/`
- `classification_configs/` → `../../../classification/setups/`
