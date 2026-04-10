# PSID Pipeline Runs — 200Hz Narrow-Band

## PDI4 Session 3

| Field | Value |
|-------|-------|
| **Participant** | PDI4 |
| **Session** | 3 |
| **Data** | `preprocessing/participants_at_200Hz_scaled_1e6_narrow_band.yaml` |
| **Data root** | `resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band` |
| **Sampling freq** | 200 Hz |
| **Channels** | 60 (15 bands x 4 ECoG electrodes, all raw) |
| **Behavioral outputs** | tracing_velocity_x, tracing_acceleration_magnitude |
| **Split** | 60/10/30 (train/val/test) |
| **GS iterations** | 30 |
| **Full iterations** | 30 (limited by 14GB RAM at nx=25) |
| **GS grid** | nx=[2,4,8,15,25,30] x n1=[2,4,6] |
| **GS succeeded** | 12/15 runs |
| **Best nx/n1** | nx=25, n1=6 (balanced_accuracy=0.7234) |
| **Pipeline script** | `scripts/pipeline_psid_PDI4_S3.py` |

### Timestamps

| Variant | Timestamp | Config |
|---------|-----------|--------|
| dbs_both (non-vanilla) | 20260408_104900 | `training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI4_3_nx_25_n6_i30_dbs_both_200Hz_narrow_band.yaml` |
| dbs_on (non-vanilla) | 20260408_133019 | `training/setups/psid/narrow_band_200Hz/on/psid_behavioral_PDI4_3_nx_25_n6_i30_dbs_on_200Hz_narrow_band.yaml` |
| dbs_off (non-vanilla) | 20260408_133326 | `training/setups/psid/narrow_band_200Hz/off/psid_behavioral_PDI4_3_nx_25_n6_i30_dbs_off_200Hz_narrow_band.yaml` |
| vanilla dbs_both | 20260408_111728 | `training/setups/psid/narrow_band_200Hz/both/psid_behavioral_PDI4_3_nx_25_n6_i30_vanilla_dbs_both_200Hz_narrow_band.yaml` |

### Grid Search Results

| nx | n1 | pearson_mean | GS balanced_acc |
|----|----|--------------| --------------- |
| 8 | 6 | 0.0185 | — |
| 25 | 6 | 0.0107 | 0.7234 (best) |
| 30 | 6 | 0.0101 | 0.6833 |
| 15 | 2 | 0.0060 | 0.5505 |
| 25 | 4 | 0.0046 | — |

### Classification (Phase 5, 100 permutations)

| Feature source | balanced_accuracy |
|----------------|-------------------|
| Xp (all 25 states) | 0.7289 |
| Xp_1 (6 behav-relevant) | 0.5520 |
| Xp_2 (19 remaining) | 0.7279 |
| Xp_with_dbs | 1.0000 |
| flipped Xp (h=1.5,2.5,4.5) | 0.7289 |

### Notes
- 14GB RAM limits PSID iterations: i=30 is max for nx=25 with 60 channels at 200Hz
- i=30 at 200Hz = 150ms horizon
- DBS state is in the non-behaviorally-relevant subspace (Xp_2=0.73) not Xp_1 (0.55) — PSID successfully separates behavioral from DBS dynamics
- Xp_with_dbs=1.0 is expected (DBS label concatenated as feature — sanity check)
- Pipeline completed 2026-04-08 14:38

---

## PDI4 Session 2

*Pending*

## PDI1 Session 2

*Pending*

## PDI1 Session 4

*Pending*
