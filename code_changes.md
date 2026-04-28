# Code Changes — Training & Frameworks Bug Fixes

Date: 2026-04-09

---

## 1. DBS Condition Filtering with `reuse_splits` (trainer.py)

**File:** `training/components/trainer.py`
**Problem:** When `reuse_splits: true`, the trainer loaded existing train/val/test parquet splits
without filtering by DBS condition (`on`/`off`). This meant all VARMA on/off/both models were
trained on identical data (the `both` split), producing byte-for-byte identical models.

**Root cause:** The `split_data()` method only applied DBS filtering when creating new splits,
but the `reuse_splits` code path skipped directly to loading existing splits with no filtering.

**Fix:** After loading reused splits, filter each split parquet by `stim` column when
`dbs_condition != "both"`.

```python
# training/components/trainer.py — inside split_data(), after line 44

if reuse_splits and existing_splits:
    self.logger.info(f"Reusing existing splits from {split_dir}")
    dbs_condition = self.data_params.dbs_condition
    if dbs_condition != "both":
        for split_name in ("train", "val", "test"):
            split_path = split_dir / f"{split_name}.parquet"
            df_split = pl.read_parquet(split_path)
            if "stim" in df_split.columns:
                before = len(df_split)
                df_split = df_split.filter(pl.col("stim") == dbs_condition)
                after = len(df_split)
                if after < before:
                    df_split.write_parquet(split_path)
                    self.logger.info(
                        f"Filtered {split_name} split to dbs_condition={dbs_condition}: "
                        f"{before} → {after} trials"
                    )
```

---

## 2. Empty Split Crashes (tester.py)

**File:** `training/components/tester.py`
**Problem:** After applying the DBS condition filter above, some splits (especially `val` for
on/off conditions) can end up with 0 trials. The tester then crashed in two places:

### 2a. `run_predictions()` and `run_predictions_selective()` — IndexError on empty meta_list

**Root cause:** `meta_list[0].get("chunk_margin")` crashes when `meta_list` is empty after
`_slice_data()` returns no trials.

**Fix:** Guard against empty meta_list in both methods:

```python
# training/components/tester.py — in run_predictions() and run_predictions_selective()
# Added after the _slice_data() call

if not meta_list:
    self.logger.warning(f"Skipping {split_name}: no trials after slicing")
    self.results[split_name] = {}
    continue
```

### 2b. `save_results()` — KeyError on empty split results

**Root cause:** `save_results()` tried to access `self.results[k]["participant_id"]` for empty
split dicts (set to `{}` by the guard above), causing a KeyError.

**Fix:** Skip saving for empty splits:

```python
# training/components/tester.py — in save_results()
# Added at the start of the results loop

if not self.results[k]:
    self.logger.warning(f"Skipping save for empty split '{k}'")
    continue
```

---

## 3. PSID Forecast Amplitude Collapse (frameworks.py)

**File:** `utils/frameworks.py`
**Problem:** PSID forecasts had correct shape/correlation but only ~5% of the true signal
amplitude. Forecast RMSE was artificially low and visually the forecasts looked flat.

**Root cause:** The `PSIDWrapper.forecast()` method computes `Yf = Xf @ C.T` (line 303), which
produces output in **z-scored space** (zero mean, unit variance). However, the true Y values that
forecasts are compared against (in `validate_forecast()`) are in **original space**.

The `predict()` method doesn't have this problem because it delegates to `self.idSys.predict()`,
which internally calls `YPrepModel.apply_inverse()` to un-z-score the output.

**Evidence:**
- `C @ Xp` gives std=0.76, while `idSys.predict()` output has std=3.92
- `apply_inverse(C @ Xp)` matches `predict()` output exactly (diff = 0.000000)
- `YPrepModel` stores per-channel mean (range [-0.005, 0.001]) and std (range [1.5, 18.0])

**Fix:** Apply `YPrepModel.apply_inverse()` and `ZPrepModel.apply_inverse()` to the forecast
output to bring it back to original space:

```python
# utils/frameworks.py — in PSIDWrapper.forecast(), after computing Yf and Zf

# Step 4: Un-z-score to original space (predict() does this internally via
# idSys.predict, but manual C @ x multiplication stays in z-scored space)
if hasattr(self.idSys, "YPrepModel") and self.idSys.YPrepModel is not None:
    Yf = self.idSys.YPrepModel.apply_inverse(Yf)
if Zf is not None and hasattr(self.idSys, "ZPrepModel") and self.idSys.ZPrepModel is not None:
    Zf = self.idSys.ZPrepModel.apply_inverse(Zf)
```

**Result after fix:**
- Amplitude ratio improved from ~5% to ~85% of true signal
- RMSE now reflects actual prediction error in original units
- Forecast shape (Pearson r) unchanged — the fix is a linear transform
