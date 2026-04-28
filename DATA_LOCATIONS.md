# Data Locations — mRMR / i=100 canonical run

Snapshot as of 2026-04-19. **Bobby** (`/home/bobby/...`) holds PDI4 cells.
**Jacque** (`jacque@10.0.0.2:/home/jacque/...`) holds PDI1 cells. **Nothing
has been synced between hosts** — each artefact lives only on the host that
produced it.

Two participant subsets, two modes (behavioral / laplacian), three DBS
conditions (both / on / off) plus a vanilla variant for PSID. K=8 mRMR
channels per side, nx/n1 from `configs/diagnostic/elbow_choices.yaml`,
i=100 (default after 2026-04-18 fix to `FULL_I_BEHAV/LAP=100` and
`RETRY_I_SEQUENCE=[75, 50, 45, 40, 35, 30]`).

---

## 1. PSID — both hosts

Status: **complete on both hosts (8/8 cells)**.

### Bobby (PDI4 subset, 4 cells)
```
results/psid_{behavioral,laplacian}_PDI4_{2,3}_nx_50_n10_i100_dbs_{both,off,on}_200Hz_narrow_band/
results/psid_{behavioral,laplacian}_PDI4_{2,3}_nx_50_n10_i100_vanilla_dbs_both_200Hz_narrow_band/
```
- Per cell: 4 variant dirs (both, on, off, vanilla), each with `model_*.pkl`,
  `model_*_metadata.json`, `split/{train,val,test}.parquet`,
  `test/test_results_*.parquet/`, `cross_eval/` (Phase 4 PSID), `classification/`.
- Cross-condition eval dirs: `..._dbs_off_..._eval_on/` and `..._dbs_on_..._eval_off/`.

### Jacque (PDI1 subset, 4 cells)
```
~/repos/latent-neural-dynamics-modeling/results/psid_{behavioral,laplacian}_PDI1_{2,4}_nx_{55,50}_n{15,10}_i100_dbs_{both,off,on}_200Hz_narrow_band/
```
- PDI1_S2 uses nx=55, n1=15. PDI1_S4 uses nx=50, n1=10. Per `elbow_choices.yaml`.
- Same per-variant directory structure as bobby.

### Classification (per cell × mode × `_flipped`)
- Bobby: `results/classification/psid_{behavioral,laplacian}_PDI4_{2,3}_nx_50_n10_i100_dbs_both_200Hz_narrow_band[_flipped]/`
- Jacque: `results/classification/psid_{behavioral,laplacian}_PDI1_{2,4}_nx_{55,50}_n{15,10}_i100_dbs_both_200Hz_narrow_band[_flipped]/`
- Each contains a `<run_ts>/` dir with the h × m grid (`h*_m*/`) holding
  `LDA_{Xp,Xp_1,Xp_2,Xp_with_dbs}_forecast.pkl` + the prediction-step pkls
  at the run_ts root (`LDA_*_prediction.pkl`).
- Permutation-test results are written into the same forecast pkls under
  `["permutation_test"] = {"score", "pvalue", "n_permutations"}`.

---

## 2. VARMA — both hosts (⚠ STALE — see flag below)

```
Bobby:  results/varma_{behavioral,laplacian}_PDI4_{2,3}_p30_q1_mrmr8_dbs_{both,off,on}_200Hz_narrow_band/
Jacque: ~/.../results/varma_behavioral_PDI1_{2,4}_p30_q1_mrmr8_dbs_{both,off,on}_200Hz_narrow_band/
```

### ⚠ Issue 1 — bobby + jacque VARMA models are PRE-i=100
- Bobby: `model_20260418_121*_*.pkl` (timestamps **12:16-12:18** on Apr 18)
- Jacque: `model_20260418_143*_*.pkl` (**14:32-14:33** jacque-time)
- These are from the original 14:36 PSID launch's chained VARMA waiter,
  trained against the **i=50** PSID variants.
- The 22:36 VARMA "rerun" with `--skip-mrmr-if-yaml-exists` hit pipeline_varma's
  skip-if-exists logic and reused the existing models. Phase 2 (test) and
  Phase 3 (cross-eval) parquets are also from 12:16-14:33 — pre-i=100.

### ⚠ Issue 2 — VARMA laplacian missing on jacque
- 6 jacque VARMA dirs exist (behavioral × 3 conditions × 2 sessions). No
  laplacian. Logs show `FileNotFoundError: No PSID full-training variant
  found for PDI1 S2 (laplacian) scanning i in [45, 40, 35, 30]`.
- Root cause: `pipeline_varma.py:autodetect_psid_best` scans
  `{cfg.psid_i} | {45, 40, 35, 30}` where `cfg.psid_i` defaults to
  `DEFAULT_PSID_I_LAP=45`. Never includes 100.
- Fix needed: bump `DEFAULT_PSID_I_BEHAV=DEFAULT_PSID_I_LAP=100` in
  `scripts/pipeline_varma.py`, OR pass `--psid-i 100` from the launcher.
- Bobby VARMA laplacian succeeded only because the 12:18 timestamp shows it
  was created against pre-i=100 models that did exist at i=45/50.

### Cross-eval and classification (VARMA)
- Bobby: `..._eval_on/` and `..._eval_off/` exist for each variant (Phase 3
  ran). Phase 4 classification sub-dirs exist but logged crashes (cls_perm_Xp
  AttributeError from data shape mismatch with stale models). **VARMA Phase 4
  is excluded from Q3 figures per user instruction** so this is harmless.

### Recommendation
1. Delete bobby's stale `varma_*_PDI4_*_p30_q1_mrmr8_*` dirs (or move to
   `.stray_models/`).
2. Patch `pipeline_varma.py` to scan `{100, 75, 50, 45, 40, 35, 30}` for
   PSID i.
3. Re-launch VARMA with `--end-phase 3` so Phase 4 isn't attempted.
4. Sync new VARMA dirs back to bobby (see §5).

---

## 3. DPAD — split bobby/jacque, IN PROGRESS

```
Bobby:  results/dpad_behavioral_PDI4_{2,3}_nx_50_n10_e3000_mrmr8_dbs_{both,off,on}_200Hz_narrow_band/
Jacque: ~/.../results/dpad_behavioral_PDI1_{2,4}_nx_{55,50}_n{15,10}_e3000_mrmr8_dbs_{both,off,on}_200Hz_narrow_band/
```

- Behavioral only — `pipeline_dpad.py` is not laplacian-aware (separate
  refactor per `pipeline_runs.md` TODO).
- 2 cells per host × 3 DBS conditions = 6 DPAD models per host.

### Status (2026-04-19 16:30)
- **Bobby PDI4_S2**: Phase 1 ✓ (3 model_*.pkl on disk). Phase 2 recovery
  in progress (started 15:30, --incremental --splits train val test, no
  timeout) — child PID 1028528.
- **Bobby PDI4_S3**: Phase 1 ✓ (3 model_*.pkl). Phase 2 will fire after
  PDI4_S2 finishes Phase 2-4.
- **Jacque PDI1_S2**: Phase 1 ✓. Phase 2 timed out yesterday (stale code);
  needs recovery `--start-phase 2` after jacque queue clears.
- **Jacque PDI1_S4**: still in Phase 1 (`off` variant training, ~4h17 CPU
  since 17:53 jacque-time).

### Classification (DPAD)
- Pending — fires in Phase 4 of the DPAD pipeline once Phase 2 + Phase 3
  finish for each cell.
- Will live at `results/classification/dpad_behavioral_PDI{1,4}_*_e3000_mrmr8_*/`.
- Uses PSID's CV-best (h, m) as a fixed point per `_find_psid_best_hm()`
  (no DPAD-side h/m grid search).

---

## 4. mRMR YAMLs (channel selection inputs)

Generated by `scripts/overnight_all_sessions.py:emit_training_yaml` and
consumed by all three pipelines via `--channels-from`.

```
Bobby:  training/setups/psid/narrow_band_200Hz/overnight/
        - psid_behavioral_PDI4_S{2,3}_nx_50_n10_i100_mrmr8_dbs_both_200Hz_narrow_band_overnight.yaml
        - psid_laplacian_PDI4_S{2,3}_nx_50_n10_i100_mrmr8_dbs_both_200Hz_narrow_band_overnight.yaml
Jacque: ~/.../training/setups/psid/narrow_band_200Hz/overnight/
        - psid_behavioral_PDI1_S{2,4}_nx_{55,50}_n{15,10}_i100_mrmr8_dbs_both_200Hz_narrow_band_overnight.yaml
        - psid_laplacian_PDI1_S{2,4}_nx_{55,50}_n{15,10}_i100_mrmr8_dbs_both_200Hz_narrow_band_overnight.yaml
```

mRMR is deterministic on a given session's data, so the channel selections
are reproducible. Stale `_i50_` yamls also exist in those dirs from the
original launch but are unused (filename tag changed to `_i100_` after the
2026-04-18 launcher fix).

---

## 5. Sync status — bobby ↔ jacque

**Nothing has been synced.** Confirmed by:
- `ls /home/bobby/.../results/*PDI1*nx_*i100*` → empty
- `ls /home/jacque/.../results/*PDI4*nx_*i100*` → empty

### What needs to land on bobby (the analysis machine)
For figures to read all 8 cells in one place, pull jacque's PDI1 results to
bobby. Suggested rsync (run from bobby):

```bash
rsync -av -e 'ssh -i ~/.ssh/id_ed25519_nopass' \
  jacque@10.0.0.2:'~/repos/latent-neural-dynamics-modeling/results/psid_*PDI1_*_i100_*_200Hz_narrow_band' \
  /home/bobby/repos/latent-neural-dynamics-modeling/results/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519_nopass' \
  jacque@10.0.0.2:'~/repos/latent-neural-dynamics-modeling/results/dpad_behavioral_PDI1_*_e3000_mrmr8_*' \
  /home/bobby/repos/latent-neural-dynamics-modeling/results/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519_nopass' \
  jacque@10.0.0.2:'~/repos/latent-neural-dynamics-modeling/results/classification/psid_*PDI1_*_i100_*' \
  /home/bobby/repos/latent-neural-dynamics-modeling/results/classification/

rsync -av -e 'ssh -i ~/.ssh/id_ed25519_nopass' \
  jacque@10.0.0.2:'~/repos/latent-neural-dynamics-modeling/results/classification/dpad_behavioral_PDI1_*_e3000_mrmr8_*' \
  /home/bobby/repos/latent-neural-dynamics-modeling/results/classification/
```

(Defer the DPAD sync until jacque DPAD Phase 2-4 recovery completes — would
be wasted bandwidth on incomplete dirs otherwise.)

### Decision needed
- **Run figure code on bobby with synced data** (recommended — one machine,
  one source of truth) — sync after jacque DPAD finishes.
- Or **separate per-host figure runs** then composite — more brittle.

---

## 6. Logs

```
Bobby:  logs/overnight/
        - PDI4_S{2,3}_{behavioral,laplacian}_psid.log
        - PDI4_S{2,3}_{behavioral,laplacian}_varma.log     [stale; pre-i=100 VARMA]
        - PDI4_S{2,3}_behavioral_dpad.log                  [PDI4_S2 = recovery in progress]
        - batch_local_dpad_recovery_20260419_153012.log    [current bobby DPAD launcher log]

Jacque: ~/.../logs/overnight/
        - PDI1_S{2,4}_{behavioral,laplacian}_psid.log
        - PDI1_S{2,4}_behavioral_varma.log                 [stale; pre-i=100 VARMA]
        - PDI1_S{2,4}_laplacian_varma.log                  [empty; failed at autodetect]
        - PDI1_S{2,4}_behavioral_dpad.log                  [PDI1_S4 = still in Phase 1]
```

---

## 7. Quick reconciliation checklist

After all jobs finish:
- [ ] Bobby DPAD recovery (PDI4_S2 then PDI4_S3) → Phase 2-4 clean
- [ ] Jacque DPAD recovery (PDI1_S2 + PDI1_S4) → Phase 2-4 clean
- [ ] Decide on VARMA mRMR retrain (Issue 1 + 2 above)
- [ ] rsync jacque PDI1 PSID + DPAD + classification → bobby
- [ ] Confirm all 8 cells × 3 model families × per-DBS-condition variant dirs land in `/home/bobby/.../results/`
- [ ] Verify figure-builder loaders (sec1/sec2/sec5/sec6/sec7) point to mRMR/i=100 variant names (currently many notebook configs reference April-8 / April-11 stale paths per `_section_4_*` plan files)
