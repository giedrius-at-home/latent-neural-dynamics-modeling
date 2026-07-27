# preprocessing

Turns raw 1000 Hz intracranial recordings plus the tablet motion files into one
tidy parquet table: **one row per trial**, holding band-limited neural signals and
smoothed kinematics at 200 Hz. Everything downstream reads that table and nothing
else.

```bash
python -m preprocessing.package_recordings \
  --config preprocessing/participants_at_200Hz_scaled_1e6_raw_envelope.yaml \
  --participant PDI1 --session 2      # both flags optional; omit to process everything
```

Two configs ship with the repo:

| Config | Produces |
|---|---|
| `participants_at_200Hz_scaled_1e6_raw_envelope.yaml` | 17 band-limited raw signals **and** 17 Hilbert envelopes per channel — the one used for the thesis |
| `participants_at_200Hz_scaled_1e6_narrow_band.yaml` | 15 raw bands only, no envelopes (earlier iteration) |

**`root_directory:` in both configs is an absolute path on the compute host.**
Change it before running anywhere else.

---

## What it reads

The recordings follow a BIDS-like tree. BIDS (Brain Imaging Data Structure) is the
standard naming convention for neuroimaging datasets — directories and filenames
carry the metadata, e.g.
`sub-PDI1/ses-2/motion/sub-PDI1_ses-2_task-copydraw_run-3_chunk-1_motion.tsv`
means participant PDI1, session 2, copy-draw task, block 3, trial 1.

Preprocessing does **not** parse the original `.mat`/`.edf` files. It reads an
intermediate parquet table (`data/participants_2/`) that a prior conversion step
produced, plus the motion sidecar files:

```
data/
├── participants.tsv                                  participant/session table
├── participants_2/participant_id=PDI1/session=2/block=1/*.parquet
└── <session dir>/motion/
    ├── ..._task-copydraw_run-<block>_chunk-<trial>_motion.tsv    pen coordinates (columns: x, y)
    └── ..._task-copydraw_run-<block>_motion.json                 DBS state (key: dbs_stim)
```

Columns each `participants_2` partition must have:

| Column | Type | Meaning |
|---|---|---|
| `participant_id` | str | `PDI1`, `PDI4` |
| `session` | int | 2, 3, 4 |
| `block` | int | one block = one DBS state; stim never switches mid-block |
| `trials` | list[int] | trial numbers in the block; exploded into one row per trial |
| `onsets` | list[float] | trial onset times (s), indexed by `trial - 1` |
| `ieeg_parquet` | str | path to the block's iEEG parquet |
| `session_path` | str | used to find the motion directory |
| `stim` | str | DBS state (`on`/`true`/`1` vs `off`/`false`/`0`) |
| `is_fragmented` | bool | optional; `True` rows are dropped |

The iEEG parquet holds one list of float32 samples per channel at **1000 Hz**:
`LFP_1`…`LFP_16`, `ECOG_1`…`ECOG_4`, `EOG_1`…`EOG_4`, `sfreq`.

Trial onsets come from the BIDS `*_events.tsv` files
(`onset, duration, trial_type, value, sample`); only `trial_type` values 10–21 —
the copy-draw trial markers — are kept.

---

## What it does, in order

1. **Load** the block's iEEG parquet and unnest it into one column per channel.
2. **Drop LFP_1…LFP_8** immediately. Only the deep contacts 9–16 are used, and
   dropping the rest early keeps the band processing from blowing up memory.
   (`drop_lfp: true` drops all LFP channels instead.)
3. **Common average reference** across `ECOG_1..4`: subtract the per-timestep mean
   of the four ECoG channels from each. CAR runs per row, i.e. per block, and a
   block is a single DBS state — so the ON-block mean is built only from ON
   samples and there is no cross-state leakage.
4. **Notch filter** the mains frequency and its harmonics (`notch_freqs`).
5. **Band-pass** into every band named in `raw_bands`, and for each band in
   `envelope_bands` also take the Hilbert amplitude envelope. The `raw_envelope`
   config uses 17 bands from 4 to 93 Hz, with a gap at 47–53 Hz for the 50 Hz
   mains and an upper stop at 93 Hz to stay clear of Nyquist and the 100 Hz
   harmonic. Raw bands keep the oscillation's phase (which a linear model can
   extrapolate); envelopes keep the slow amplitude modulation (which carries the
   DBS on/off difference).
6. **Build LFP Laplacians** from the deep contacts:
   `D_k = LFP_k − 2·LFP_{k+1} + LFP_{k+2}`, giving `LAPLACIAN_9-11_LFP_*` through
   `LAPLACIAN_14-16_LFP_*`. These are the `z-as-neural` targets.
7. **Resample** 1000 → 200 Hz and **scale** by `scale_factor` (1e6, i.e. volts to
   microvolts).
8. **Join the motion data**: pen `x`/`y` per trial from the tsv files, DBS state
   from the json, resampled onto the neural time grid.
9. **Smooth and differentiate** the trajectory with a Savitzky-Golay filter to get
   `tracing_velocity_*`, `tracing_acceleration_*`, `tracing_jerk_*`.
10. **Cut into trials** around each onset, keeping `chunk_margin` seconds of
    context on both sides. A trial is split further wherever the tablet stopped
    reporting for more than `max_pause_seconds`.
11. **Write** the result partitioned by participant / session / block.

Things that get discarded along the way: `is_fragmented` blocks, LFP_1–8, EOG
channels, blocks whose neural columns are all null, and every intermediate path
column.

---

## What it writes

```
resampled_recordings/participants_at_200Hz_scaled_1e6_raw_envelope/
    participant_id=PDI1/session=2/block=3/0.parquet
```

One row per **trial**:

| Column | Type | Meaning |
|---|---|---|
| `participant_id`, `session`, `block`, `trial` | str/int | partition keys |
| `onset`, `margined_onset`, `margined_duration` | float | trial timing in seconds |
| `time`, `time_original`, `motion_time` | list[float] | time vectors |
| `original_length_ts`, `start_ts`, `chunk_margin` | int/float | chunking metadata |
| `stim` | str | `on` / `off` |
| `x`, `y`, `x_smooth`, `y_smooth` | list | pen coordinates, raw and smoothed |
| `tracing_velocity_x/y`, `tracing_velocity_magnitude`, `tracing_acceleration_magnitude`, `tracing_jerk_*` | list[float] | Savitzky-Golay derivatives |
| `<channel>_<band>_raw` | list[float] | band-limited signal at 200 Hz |
| `<channel>_<band>_env` | list[float] | Hilbert envelope of that band |

Channel names are built from the prefix and the band key in the config, so they
read e.g. `ECOG_3_gamma_88_93_raw`, `ECOG_1_alpha_8_12_env`,
`LAPLACIAN_14-16_LFP_beta_18_23_env`.

Each signal column is a list of `margined_duration × 200` samples. The model layer
stacks the columns named in a run config into `Y ∈ (T, n_Y)` and `Z ∈ (T, n_Z)`.

---

## Files

| File | Role |
|---|---|
| `package_recordings.py` | CLI entry point (`--config`, `--participant`, `--session`) |
| `components/participants.py` | the main pipeline: iEEG load, CAR, bands, Laplacians, chunking, write |
| `components/motion.py` | pen coordinates and DBS state from the motion tsv/json |
| `components/events.py` | trial onsets from the BIDS events tsv |
| `../utils/ieeg.py` | band-pass, Hilbert, Welch/Morlet power, resampling |
| `../utils/motion.py`, `../utils/sync.py` | derivatives, interpolation onto the neural grid |
