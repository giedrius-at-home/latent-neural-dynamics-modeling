# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Section 1: Data Verification
#
# | # | Figure | Description |
# |---|--------|-------------|
# | 1 | Trial inventory | Original vs preprocessed trial counts, missing blocks, split distribution |
# | 2 | Trial count summary | Bar chart of DBS-OFF / DBS-ON trial counts per session |
# | 3-6 | PSD DBS comparison | Power spectral density per ECoG channel, 4 sessions |
# | 7 | Tracing speed DBS comparison | Mean velocity & acceleration traces by DBS condition |

# %%
import sys, os

os.chdir("/home/bobby/repos/latent-neural-dynamics-modeling")
sys.path.insert(0, ".")
sys.path.insert(0, "notebooks")

# %%
from collections import namedtuple
from pathlib import Path
import numpy as np
import polars as pl
import yaml
import matplotlib.pyplot as plt
from scipy.signal import welch

from thesis_style import (
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_DPAD,
    COLOR_PSID,
    apply_thesis_style,
    hex_to_rgba,
    panel_label,
    stack_bar_label,
)

apply_thesis_style()

# %%
OUT = Path("thesis_figures/sec1")
OUT.mkdir(parents=True, exist_ok=True)
results_root = Path("results").resolve()
# PSD cells read the raw pre-split parquets — current per-session PSID variants
# only keep a selected subset of neural bands, so the split parquets no longer
# contain all 4 ECoG channels or the Laplacian contacts used by the PSD cells.
raw_data_root = Path("resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band")
ECOG_CHANNELS = ["ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"]

# %%
# Session registry — derived from the canonical triplets in
# notebooks/thesis_triplets.csv (loaded via thesis_sec2_common). Single source
# of truth: append a newer-dated row in the CSV to swap variants/timestamps.
# The split parquets are identical across variant choices for a given
# (participant, session), so pointing sec1 at the current modelling variants
# guarantees sec1 reads the same splits the rest of the thesis uses.
from thesis_sec2_common import ALL_TRIPLETS

SessionSpec = namedtuple(
    "SessionSpec", ["label", "participant", "session", "psid_variant", "psid_run_ts"]
)

def _session_spec_from_triplet(tri) -> SessionSpec:
    pid, sess = tri.label.split("_S")
    return SessionSpec(
        label=tri.label, participant=pid, session=int(sess),
        psid_variant=tri.psid_variant, psid_run_ts=tri.psid_run_ts,
    )

SESSIONS = [_session_spec_from_triplet(t) for t in ALL_TRIPLETS]

# %% [markdown]
# ## Trial inventory
#
# Each session nominally contains **12 blocks x 12 trials** = 144 trials.
# The preprocessing pipeline loads `data/participants_2/` (intermediate hive
# parquet built from EEG event markers with `trial_type` in `[10, 21]`).
# This table carries an `is_fragmented` flag per block.
#
# Two filters are applied when creating the final train/val/test splits:
#
# 1. **Fragmented blocks** (`is_fragmented = True`) are dropped entirely.
# 2. **Plateau filter** (`max_pause_seconds > 2.0 s`) removes individual
#    trials where the (x, y) cursor stays stationary too long.
#
# Note: `data/participants.parquet` exists but is **not used** by the
# modelling pipeline — it is an older metadata file enriched with
# `PDI_labels_trials.csv` yscores and has different trial counts.
# The split files are the ground truth for all downstream modelling.
#
# ### Missing blocks (expected 12 per session)
#
# | Session | Missing blocks | Reason |
# |---------|---------------|--------|
# | PDI1_S2 | none | All 12 blocks present |
# | PDI1_S4 | 11, 12 | No iEEG recording files on disk. Session ended early or recordings lost. |
# | PDI4_S2 | 1, 2, 3 | No iEEG files. Block 1 absent from all sources. Blocks 2-3 have entries in `PDI_labels.csv` but no neural recording exists. |
# | PDI4_S3 | 7, 9 | Present in `participants_2` with 12 trials each but `is_fragmented = True`. Dropped during split creation. |
#

# %%
from collections import defaultdict

p2_root = Path("data/participants_2")


def _p2_block_info(pid, ses):
    """Read participants_2: {block: {'trials': [...], 'frag': bool, 'stim': str}}."""
    ses_path = p2_root / f"participant_id={pid}" / f"session={ses}"
    info = {}
    for bd in ses_path.glob("block=*"):
        bnum = int(bd.name.split("=")[1])
        pf = list(bd.glob("*.parquet"))
        if not pf:
            continue
        df = pl.read_parquet(pf[0])
        trials_val = df["trials"][0]
        trials_list = trials_val.to_list() if hasattr(trials_val, "to_list") else list(trials_val)
        frag = bool(df["is_fragmented"][0]) if "is_fragmented" in df.columns else False
        stim = str(df["stim"][0]) if "stim" in df.columns else "?"
        info[bnum] = {"trials": trials_list, "frag": frag, "stim": stim}
    return info


def _split_block_trials(variant):
    """Read split parquets: {block: {trial: stim}}."""
    base = results_root / variant / "split"
    bt = defaultdict(dict)
    for split in ("train", "val", "test"):
        df = pl.read_parquet(base / f"{split}.parquet")
        for r in df.select("block", "trial", "stim").iter_rows(named=True):
            bt[int(r["block"])][int(r["trial"])] = str(r["stim"])
    return bt


summary_rows = []
all_removal_details = []
dbs_dist_rows = []
frag_total, plateau_total, protocol_total = 0, 0, 0

for s in SESSIONS:
    p2 = _p2_block_info(s.participant, s.session)
    splits = _split_block_trials(s.psid_variant)
    all_expected = set(range(1, 13))
    present_blocks = set(p2.keys())
    missing_no_data = sorted(all_expected - present_blocks)

    n_frag, n_plateau, n_protocol = 0, 0, 0
    frag_blocks = []
    p2_off, p2_on, split_off, split_on = 0, 0, 0, 0

    print(f"\n{'='*60}")
    print(f"  {s.label}  ({s.participant} session {s.session})")
    print(f"{'='*60}")

    if missing_no_data:
        print(f"  Blocks with NO recording: {missing_no_data}")

    for b in sorted(present_blocks):
        info = p2[b]
        p2_trials = set(info["trials"])
        s_trial_dict = splits.get(b, {})
        s_trials = set(s_trial_dict.keys())
        dbs = "DBS-" + info["stim"].upper()

        if info["stim"].lower() == "off":
            p2_off += len(p2_trials)
        else:
            p2_on += len(p2_trials)
        for stim_val in s_trial_dict.values():
            if stim_val in ("off", "0"):
                split_off += 1
            else:
                split_on += 1

        if info["frag"]:
            frag_blocks.append(b)
            n_frag += len(p2_trials)
            print(f"  Block {b:2d} ({dbs}): FRAGMENTED — all {len(p2_trials)} trials dropped")
            for t in sorted(p2_trials):
                all_removal_details.append({
                    "participant": s.participant, "session": s.session,
                    "block": b, "trial": t, "dbs": dbs, "reason": "fragmented",
                })
            continue

        expected_full = set(range(1, 13))
        protocol_removed = sorted(expected_full - p2_trials)
        if protocol_removed:
            n_protocol += len(protocol_removed)
            for t in protocol_removed:
                all_removal_details.append({
                    "participant": s.participant, "session": s.session,
                    "block": b, "trial": t, "dbs": dbs, "reason": "protocol/events",
                })

        plateau_removed = sorted(p2_trials - s_trials)
        if plateau_removed:
            n_plateau += len(plateau_removed)
            for t in plateau_removed:
                all_removal_details.append({
                    "participant": s.participant, "session": s.session,
                    "block": b, "trial": t, "dbs": dbs, "reason": "plateau (>2.0s)",
                })

        parts = []
        if protocol_removed:
            parts.append(f"protocol={protocol_removed}")
        if plateau_removed:
            parts.append(f"plateau={plateau_removed}")
        status = "; ".join(parts) if parts else "ok"
        print(f"  Block {b:2d} ({dbs}): p2={len(p2_trials)}, splits={len(s_trials)}  [{status}]")

    p2_total_all = sum(len(p2[b]["trials"]) for b in p2)
    p2_total_nonfrag = sum(len(p2[b]["trials"]) for b in p2 if not p2[b]["frag"])
    split_total = sum(len(splits[b]) for b in splits)
    summary_rows.append({
        "Session": s.label,
        "Blocks (p2)": len(p2),
        "Fragmented": len(frag_blocks),
        "Blocks (splits)": len(splits),
        "p2 trials": p2_total_all,
        "Split trials": split_total,
        "Protocol removed": n_protocol,
        "Plateau removed": n_plateau,
        "Fragmented removed": n_frag,
    })
    dbs_dist_rows.append({
        "Session": s.label,
        "p2 OFF": p2_off,
        "p2 ON": p2_on,
        "p2 total": p2_off + p2_on,
        "split OFF": split_off,
        "split ON": split_on,
        "split total": split_total,
        "removed OFF": p2_off - split_off,
        "removed ON": p2_on - split_on,
    })
    frag_total += n_frag
    plateau_total += n_plateau
    protocol_total += n_protocol

# --- 1. Removal summary ---
print(f"\n{'='*60}")
print("  REMOVAL SUMMARY")
print(f"{'='*60}")
print(pl.DataFrame(summary_rows))
print(
    f"\nTotals: {protocol_total} protocol, "
    f"{plateau_total} plateau (max_pause > 2.0 s), "
    f"{frag_total} fragmented"
)

# --- 2. DBS condition distribution per session ---
print(f"\n{'='*60}")
print("  DBS CONDITION DISTRIBUTION (per session)")
print(f"{'='*60}")
print(pl.DataFrame(dbs_dist_rows))

# --- 2b. Split manifest CSV (one row per session) — bookkeeping for
#         balanced block-chronological splits loaded from configs/splits/. ---
splits_cfg_dir = Path("configs/splits")
manifest_rows = []
for s in SESSIONS:
    cfg_path = splits_cfg_dir / f"{s.participant}_S{s.session}.yaml"
    if not cfg_path.exists():
        print(f"  [warn] missing {cfg_path}")
        continue
    with open(cfg_path) as _f:
        alloc = yaml.safe_load(_f)

    def _n_trials(blocks, stim):
        total = 0
        for b in blocks:
            match = split_dbs_df.filter(
                (pl.col("session") == s.label) & (pl.col("split") != "")
            )
            # fall back to re-reading block_info
            p2 = _p2_block_info(s.participant, s.session).get(b, {})
            if p2 and str(p2["stim"]).lower() == stim:
                total += len(p2["trials"])
        return total

    manifest_rows.append({
        "session": s.label,
        "strategy": alloc.get("strategy", "balanced_block_chronological"),
        "train_blocks": "|".join(str(b) for b in alloc["train_blocks"]),
        "val_blocks": "|".join(str(b) for b in alloc["val_blocks"]),
        "test_blocks": "|".join(str(b) for b in alloc["test_blocks"]),
        "train_off": alloc["summary"]["train"]["n_trials_off"],
        "train_on": alloc["summary"]["train"]["n_trials_on"],
        "val_off": alloc["summary"]["val"]["n_trials_off"],
        "val_on": alloc["summary"]["val"]["n_trials_on"],
        "test_off": alloc["summary"]["test"]["n_trials_off"],
        "test_on": alloc["summary"]["test"]["n_trials_on"],
    })
if manifest_rows:
    manifest_df = pl.DataFrame(manifest_rows)
    manifest_csv = OUT / "split_manifest.csv"
    manifest_df.write_csv(str(manifest_csv))
    print(f"\n  Wrote split manifest → {manifest_csv}")
    print(manifest_df)

# --- 3. Removals by DBS condition and reason ---
if all_removal_details:
    removal_df = pl.DataFrame(all_removal_details)

    print(f"\n{'='*60}")
    print("  REMOVALS BY REASON AND DBS CONDITION")
    print(f"{'='*60}")
    grand = (
        removal_df.group_by("reason", "dbs").len()
        .sort("reason", "dbs").rename({"len": "count"})
    )
    print(grand)

    per_session = (
        removal_df.group_by("participant", "session", "dbs", "reason").len()
        .sort("participant", "session", "dbs", "reason").rename({"len": "count"})
    )
    print(f"\n  Per session:")
    print(per_session)

    # --- 4. Full list of every removed trial ---
    print(f"\n{'='*60}")
    print(f"  ALL {len(removal_df)} REMOVED TRIALS")
    print(f"{'='*60}")
    with pl.Config(tbl_rows=100):
        print(removal_df.sort("participant", "session", "block", "trial"))


# %%
# Enrich removal details with session label
_sess_lookup = {(s.participant, s.session): s.label for s in SESSIONS}
for d in all_removal_details:
    d["label"] = _sess_lookup.get((d["participant"], d["session"]), "")

# --- Collect per-split DBS counts ---
# Source of truth: configs/splits/{P}_S{s}.yaml — block-level fold assignments
# from training/precompute_splits.py. Per-block trial dict comes from existing
# results split parquets (which carry the post-plateau-filter trial set). This
# avoids stale results/*/split/*.parquet fold assignments from old pipeline
# runs and displays the NEW balanced block-chronological splits immediately.
splits_cfg_dir = Path("configs/splits")
split_dbs_rows = []
for s in SESSIONS:
    cfg_path = splits_cfg_dir / f"{s.participant}_S{s.session}.yaml"
    with open(cfg_path) as _f:
        alloc = yaml.safe_load(_f)
    per_block_trials = _split_block_trials(s.psid_variant)  # {block: {trial: stim}}
    for split_name in ("train", "val", "test"):
        blocks = alloc[f"{split_name}_blocks"]
        n_off = n_on = 0
        for b in blocks:
            for stim in per_block_trials.get(b, {}).values():
                if stim in ("off", "0"):
                    n_off += 1
                else:
                    n_on += 1
        split_dbs_rows.append({
            "session": s.label, "split": split_name,
            "OFF": n_off, "ON": n_on, "total": n_off + n_on,
        })
split_dbs_df = pl.DataFrame(split_dbs_rows)
sessions = [s.label for s in SESSIONS]

session_totals = {
    sl: int(split_dbs_df.filter(pl.col("session") == sl)["total"].sum())
    for sl in sessions
}

split_order = ["train", "val", "test"]
# Split-coloured bars (red/blue/gray) with DBS as alpha: OFF lighter, ON full.
SPLIT_COLOR = {"train": "#D32F2F", "val": "#1976D2", "test": "#616161"}
DBS_ALPHA = {"OFF": 0.55, "ON": 1.00}
BAR_W = 0.5

import numpy as np

reason_meta = [
    ("fragmented",      "#E24B4A"),
    ("protocol/events", "#888780"),
    ("plateau (>2.0s)", COLOR_DBS_OFF),
]


def _split_dbs_values(sp, dbs, as_fraction):
    out = []
    for sl in sessions:
        r = split_dbs_df.filter(
            (pl.col("session") == sl) & (pl.col("split") == sp))
        c = r[dbs][0] if r.height else 0
        if as_fraction:
            out.append(c / session_totals[sl] if session_totals[sl] else 0)
        else:
            out.append(c)
    return np.array(out, dtype=float)


# ── Figure 1: Trials removed by reason (Panel A, standalone) ─────────────
fig_a, ax_a = plt.subplots(figsize=(6.5, 3.2))
bottoms = np.zeros(len(sessions))
for reason, color in reason_meta:
    counts = np.array([sum(1 for d in all_removal_details
                           if d["label"] == sl and d["reason"] == reason)
                       for sl in sessions], dtype=float)
    bars = ax_a.bar(sessions, counts, bottom=bottoms, color=color,
                    width=0.55, label=reason)
    stack_bar_label(ax_a, bars, fmt="{:.0f}", min_value=1)
    bottoms += counts
ax_a.set_ylabel("Trials")
ax_a.set_ylim(bottom=0)
ax_a.legend(title="Removal reason", loc="upper left", borderaxespad=0.0)
panel_label(ax_a, "A", "Trials removed by reason")
fig_a.savefig(str(OUT / "fig_trials_removed.png"))
plt.show()

# ── Figure 2: Split ratios with DBS condition (Panel B) ──────────────────
# Stack order: split outer (train, val, test), DBS inner (OFF, ON).
# Colour = split (red/blue/gray); alpha = DBS (OFF lighter, ON full).
DBS_SERIES = [("OFF", None), ("ON", None)]

fig_b, ax_b = plt.subplots(figsize=(6.5, 3.2))
bottoms = np.zeros(len(sessions))
for sp in split_order:
    for dbs, _ in DBS_SERIES:
        vals_frac = _split_dbs_values(sp, dbs, as_fraction=True)
        vals_abs = _split_dbs_values(sp, dbs, as_fraction=False)
        bars = ax_b.bar(sessions, vals_frac, bottom=bottoms,
                        color=hex_to_rgba(SPLIT_COLOR[sp], DBS_ALPHA[dbs]),
                        width=0.55, label=f"{sp} DBS-{dbs}")
        stack_bar_label(ax_b, bars, values=vals_abs, fmt="{:.0f}",
                        min_value=0.02)
        bottoms += vals_frac
ax_b.set_ylabel("Split ratio")
ax_b.set_ylim(0, 1.05)
ax_b.legend(title="DBS / split", loc="upper left", borderaxespad=0.0)
panel_label(ax_b, "B", "Split ratio with DBS condition")
fig_b.savefig(str(OUT / "fig_split_ratio.png"))
plt.show()

# ── Figure 3: Trial counts per split (Panel C) ──────────────────────────
fig_c, ax_c = plt.subplots(figsize=(6.5, 3.2))
bottoms = np.zeros(len(sessions))
for sp in split_order:
    for dbs, _ in DBS_SERIES:
        vals = _split_dbs_values(sp, dbs, as_fraction=False)
        bars = ax_c.bar(sessions, vals, bottom=bottoms,
                        color=hex_to_rgba(SPLIT_COLOR[sp], DBS_ALPHA[dbs]),
                        width=0.55, label=f"{sp} DBS-{dbs}")
        stack_bar_label(ax_c, bars, fmt="{:.0f}", min_value=1)
        bottoms += vals
ax_c.set_ylabel("Trials")
ax_c.legend(title="DBS / split", loc="upper left", borderaxespad=0.0)
panel_label(ax_c, "C", "Trial counts per split")
fig_c.savefig(str(OUT / "fig_split_counts.png"))
plt.show()


# %% [markdown]
# ## PSD DBS comparison (ECOG_1, all sessions)
#

# %%
from scipy.signal import welch
from scipy.stats import mannwhitneyu

FS_SPLIT = 200
MARGIN_S = 2
MARGIN_SAMP = MARGIN_S * FS_SPLIT
FREQ_CUTOFF = 85

def _session_raw_files(participant, session):
    return sorted(
        (raw_data_root / f"participant_id={participant}" / f"session={session}").glob("*/*.parquet")
    )


_sample_paths = _session_raw_files(SESSIONS[0].participant, SESSIONS[0].session)
_sample_df = pl.read_parquet(_sample_paths[0], n_rows=1)
ECOG1_COLS = sorted([c for c in _sample_df.columns if c.startswith("ECOG_1") and c.endswith("_raw")])
print(f"ECOG_1 bands ({len(ECOG1_COLS)}): "
      f"{[c.replace('ECOG_1_','').replace('_raw','') for c in ECOG1_COLS]}")

session_psds_ecog1 = {}
freqs_ref = None

for s in SESSIONS:
    psds = {"off": [], "on": []}
    for fp in _session_raw_files(s.participant, s.session):
        df = pl.read_parquet(fp, columns=["stim"] + ECOG1_COLS)
        for row in df.iter_rows(named=True):
            cond = "off" if row["stim"] in ("off", "0") else "on"
            sig = np.zeros(len(row[ECOG1_COLS[0]]))
            for col in ECOG1_COLS:
                sig += np.array(row[col], dtype=float)
            sig = sig[MARGIN_SAMP:-MARGIN_SAMP]
            sig = sig[np.isfinite(sig)]
            if len(sig) < 400:
                continue
            nperseg = min(len(sig), FS_SPLIT * 2)
            freqs, psd = welch(sig, fs=FS_SPLIT, nperseg=nperseg, noverlap=nperseg // 2)
            psd_db = 10 * np.log10(psd + 1e-20)
            if freqs_ref is None:
                freqs_ref = freqs
            psds[cond].append(psd_db)
    session_psds_ecog1[s.label] = psds
    print(f"  {s.label}: OFF={len(psds['off'])} trials, ON={len(psds['on'])} trials")

freq_mask = freqs_ref <= FREQ_CUTOFF
freqs_plot = freqs_ref[freq_mask]
for sl in session_psds_ecog1:
    for ck in ("off", "on"):
        session_psds_ecog1[sl][ck] = [p[freq_mask] for p in session_psds_ecog1[sl][ck]]

# --- Build 2x2 stitched figure ---
channel = "ECOG_1"


def _psd_mean_sem(cond_psds):
    mat = np.vstack(cond_psds)
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0) / np.sqrt(len(mat))
    return mean, sem


def _draw_psd_cell(ax, freqs, psds_dict, conds, alpha=0.18):
    for cond_key, col_c, cond_label in conds:
        if not psds_dict[cond_key]:
            continue
        mean, sem = _psd_mean_sem(psds_dict[cond_key])
        ax.fill_between(freqs, mean - sem, mean + sem,
                        color=col_c, alpha=alpha, linewidth=0)
        ax.plot(freqs, mean, color=col_c, label=cond_label)


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.5, 4.8))
conds = [("off", COLOR_DBS_OFF, "DBS-OFF"), ("on", COLOR_DBS_ON, "DBS-ON")]
for s_idx, s in enumerate(SESSIONS):
    ax = axes.flat[s_idx]
    _draw_psd_cell(ax, freqs_plot, session_psds_ecog1[s.label], conds)
    ax.set_xlim(0, FREQ_CUTOFF)
    ax.set_xticks(range(0, FREQ_CUTOFF + 1, 10))
    panel_label(ax, chr(65 + s_idx), s.label)

for ax in axes[-1, :]:
    ax.set_xlabel("Frequency (Hz)")
for ax in axes[:, 0]:
    ax.set_ylabel(f"PSD (dB/Hz) \u2014 {channel}")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels,frameon=False)
fig.savefig(str(OUT / "fig_002_psd_ecog1.png"))
plt.show()

# %% [markdown]
# ## PSD DBS comparison — one 2x2 figure per session (4 contacts each)
#
# Same PSD pipeline as the ECoG_1 cell above, but computes all four ECoG
# contacts and produces a separate 2x2 figure per session — four files
# (`fig_002_psd_PDI1_S2.png`, ..., `fig_005_psd_PDI4_S3.png`) that compose
# cleanly as subfigures in the report without a single massive grid.

# %%
# Per-session ECoG PSDs (2x2 of 4 contacts).

ECOG_CHANNELS = [1, 2, 3, 4]

# Collect narrow-band column sets per channel.
_psd_cols_by_ecog = {
    ch: sorted([c for c in _sample_df.columns if c.startswith(f"ECOG_{ch}_") and c.endswith("_raw")])
    for ch in ECOG_CHANNELS
}

# Compute per-channel session PSDs.
psds_by_channel = {}
for ch in ECOG_CHANNELS:
    cols = _psd_cols_by_ecog[ch]
    session_psds = {}
    for s in SESSIONS:
        psds = {"off": [], "on": []}
        for fp in _session_raw_files(s.participant, s.session):
            df = pl.read_parquet(fp, columns=["stim"] + cols)
            for row in df.iter_rows(named=True):
                cond = "off" if row["stim"] in ("off", "0") else "on"
                sig = np.zeros(len(row[cols[0]]))
                for col in cols:
                    sig += np.array(row[col], dtype=float)
                sig = sig[MARGIN_SAMP:-MARGIN_SAMP]
                sig = sig[np.isfinite(sig)]
                if len(sig) < 400:
                    continue
                nperseg = min(len(sig), FS_SPLIT * 2)
                _, psd = welch(sig, fs=FS_SPLIT, nperseg=nperseg, noverlap=nperseg // 2)
                psds[cond].append(10 * np.log10(psd + 1e-20))
        session_psds[s.label] = psds
    psds_by_channel[ch] = session_psds
    print(f"  ECOG_{ch} computed")

# Trim to the shared plotting frequency axis.
for ch in ECOG_CHANNELS:
    for sl in psds_by_channel[ch]:
        for ck in ("off", "on"):
            psds_by_channel[ch][sl][ck] = [p[freq_mask] for p in psds_by_channel[ch][sl][ck]]

# One 2x2 figure per session — four output files.
_SESSION_FIG_IDX = {"PDI1_S2": 2, "PDI1_S4": 3, "PDI4_S2": 4, "PDI4_S3": 5}
conds = [("off", COLOR_DBS_OFF, "DBS-OFF"), ("on", COLOR_DBS_ON, "DBS-ON")]

for s in SESSIONS:
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.5, 4.8))
    for c_idx, ch in enumerate(ECOG_CHANNELS):
        ax = axes.flat[c_idx]
        _draw_psd_cell(ax, freqs_plot, psds_by_channel[ch][s.label], conds)
        ax.set_xlim(0, FREQ_CUTOFF)
        ax.set_xticks(range(0, FREQ_CUTOFF + 1, 10))
        panel_label(ax, chr(65 + c_idx), f"ECOG_{ch}")

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PSD (dB/Hz) \u2014 {s.label}")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    
    fig_num = _SESSION_FIG_IDX[s.label]
    out_name = f"fig_{fig_num:03d}_psd_{s.label}.png"
    fig.savefig(str(OUT / out_name))
    plt.show()
    print(f"{s.label} PSD saved \u2014 {out_name}")

# %% [markdown]
# ## PSD — Laplacian D14 (LFP 14-16), all sessions
#

# %%
# Laplacian D14 PSD — no significance shading

_lap_sample = pl.read_parquet(_sample_paths[0], n_rows=1)
LAP_COLS = sorted([c for c in _lap_sample.columns if c.startswith("LAPLACIAN_14-16") and c.endswith("_raw")])
print(f"Laplacian D14 bands ({len(LAP_COLS)})")

lap_channel = "D14 (LFP 14\u201316)"
session_psds_lap = {}
freqs_ref_lap = None

for s in SESSIONS:
    psds = {"off": [], "on": []}
    for fp in _session_raw_files(s.participant, s.session):
        df = pl.read_parquet(fp, columns=["stim"] + LAP_COLS)
        for row in df.iter_rows(named=True):
            cond = "off" if row["stim"] in ("off", "0") else "on"
            sig = np.zeros(len(row[LAP_COLS[0]]))
            for col in LAP_COLS:
                sig += np.array(row[col], dtype=float)
            sig = sig[MARGIN_SAMP:-MARGIN_SAMP]
            sig = sig[np.isfinite(sig)]
            if len(sig) < 400:
                continue
            nperseg = min(len(sig), FS_SPLIT * 2)
            freqs, psd = welch(sig, fs=FS_SPLIT, nperseg=nperseg, noverlap=nperseg // 2)
            psd_db = 10 * np.log10(psd + 1e-20)
            if freqs_ref_lap is None:
                freqs_ref_lap = freqs
            psds[cond].append(psd_db)
    session_psds_lap[s.label] = psds
    print(f"  {s.label}: OFF={len(psds['off'])} trials, ON={len(psds['on'])} trials")

freq_mask_lap = freqs_ref_lap <= FREQ_CUTOFF
freqs_plot_lap = freqs_ref_lap[freq_mask_lap]
for sl in session_psds_lap:
    for ck in ("off", "on"):
        session_psds_lap[sl][ck] = [p[freq_mask_lap] for p in session_psds_lap[sl][ck]]

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.5, 4.8))
for s_idx, s in enumerate(SESSIONS):
    ax = axes.flat[s_idx]
    psds = session_psds_lap.get(s.label, {"off": [], "on": []})
    _draw_psd_cell(ax, freqs_plot_lap, psds, conds)
    ax.set_xlim(0, FREQ_CUTOFF)
    ax.set_xticks(range(0, FREQ_CUTOFF + 1, 10))
    panel_label(ax, chr(65 + s_idx), s.label)

for ax in axes[-1, :]:
    ax.set_xlabel("Frequency (Hz)")
for ax in axes[:, 0]:
    ax.set_ylabel(f"PSD (dB/Hz) \u2014 {lap_channel}")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center",
           bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)

fig.savefig(str(OUT / "fig_003_psd_laplacian_d14.png"))
plt.show()

# %% [markdown]
# ## Behavioral signals — raw trial traces (velocity, acceleration)
#

# %%
# Behavioral raw data — z-scored per trial, mean ± SEM bands (smoothed)

FS_BEH = 200
TRIAL_SEC = 9
N_SAMP = TRIAL_SEC * FS_BEH
SMOOTH_WIN = 41
beh_vars = ["tracing_velocity_x", "tracing_acceleration_magnitude"]

def _smooth(arr, win):
    kernel = np.ones(win) / win
    padded = np.pad(arr, win // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]

for var_name in beh_vars:
    traces_by_session = {}
    for s in SESSIONS:
        base = results_root / s.psid_variant / "split"
        traces = {"off": [], "on": []}
        for split in ("train", "val", "test"):
            fp = base / f"{split}.parquet"
            if not fp.exists():
                continue
            df = pl.read_parquet(fp, columns=["stim", var_name])
            for row in df.iter_rows(named=True):
                cond = "off" if row["stim"] in ("off", "0") else "on"
                sig = np.array(row[var_name], dtype=float)[:N_SAMP]
                if len(sig) < N_SAMP:
                    continue
                mu, sd = np.nanmean(sig), np.nanstd(sig)
                if sd < 1e-8:
                    continue
                sig = (sig - mu) / sd
                sig = np.nan_to_num(sig, nan=0.0)
                traces[cond].append(sig)
        traces_by_session[s.label] = traces

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.5, 4.8))
    t_axis = np.arange(N_SAMP) / FS_BEH
    for s_idx, s in enumerate(SESSIONS):
        ax = axes.flat[s_idx]
        traces = traces_by_session[s.label]
        for cond_key, col_c, cond_label in [("off", COLOR_DBS_OFF, "DBS-OFF"),
                                             ("on", COLOR_DBS_ON, "DBS-ON")]:
            cond_traces = traces[cond_key]
            if not cond_traces:
                continue
            mat = np.vstack(cond_traces)
            mean = _smooth(mat.mean(axis=0), SMOOTH_WIN)
            sem = _smooth(mat.std(axis=0) / np.sqrt(len(mat)), SMOOTH_WIN)
            ax.fill_between(t_axis, mean - sem, mean + sem,
                            color=col_c, alpha=0.20, linewidth=0)
            ax.plot(t_axis, mean, color=col_c, label=cond_label)
        ax.set_xlim(0, TRIAL_SEC)
        ax.set_xticks(range(0, TRIAL_SEC + 1))
        panel_label(ax, chr(65 + s_idx), s.label)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"z-score \u2014 {var_name}")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    
    safe_name = var_name.replace(" ", "_")
    fig.savefig(str(OUT / f"fig_beh_{safe_name}.png"))
    plt.show()
    print(f"{var_name} figure saved")

# %% [markdown]
# ## DBS significance heatmap (sessions × frequency)

# %%
# DBS significance heatmap — single figure
# 3 rows x 2 cols: ECOG 1-4 in top 2 rows, Laplacian spanning bottom row
# Shared colour scale, one colorbar

from scipy.stats import mannwhitneyu

BAND_ORDER = [
    ("theta_4_8",     4,  8),
    ("alpha_8_12",    8, 12),
    ("beta_12_17",   12, 17),
    ("beta_17_22",   17, 22),
    ("beta_22_27",   22, 27),
    ("beta_27_30",   27, 30),
    ("gamma_30_35",  30, 35),
    ("gamma_35_40",  35, 40),
    ("gamma_40_45",  40, 45),
    ("gamma_45_50",  45, 50),
    ("gamma_50_55",  50, 55),
    ("gamma_55_60",  55, 60),
    ("gamma_60_65",  60, 65),
    ("gamma_70_75",  70, 75),
    ("gamma_75_80",  75, 80),
]

BAND_GROUPS = [("theta", 4, 8), ("alpha", 8, 12), ("beta", 12, 30), ("gamma", 30, 80)]

def _compute_band_pvals(channel_prefix):
    n_bands = len(BAND_ORDER)
    pmat = np.full((len(SESSIONS), n_bands), np.nan)
    cols_needed = ["stim"] + [f"{channel_prefix}{bn}_raw" for bn, _, _ in BAND_ORDER]
    for si, s in enumerate(SESSIONS):
        off_powers, on_powers = [], []
        for fp in _session_raw_files(s.participant, s.session):
            df = pl.read_parquet(fp, columns=cols_needed)
            for row in df.iter_rows(named=True):
                cond = "off" if row["stim"] in ("off", "0") else "on"
                powers = []
                for bn, _, _ in BAND_ORDER:
                    arr = np.array(row[f"{channel_prefix}{bn}_raw"], dtype=float)
                    arr = arr[MARGIN_SAMP:-MARGIN_SAMP]
                    powers.append(np.nanmean(arr ** 2))
                (off_powers if cond == "off" else on_powers).append(powers)
        if len(off_powers) < 2 or len(on_powers) < 2:
            continue
        off_a, on_a = np.array(off_powers), np.array(on_powers)
        for bi in range(n_bands):
            try:
                _, p = mannwhitneyu(off_a[:, bi], on_a[:, bi], alternative="two-sided")
                pmat[si, bi] = -np.log10(max(p, 1e-20))
            except ValueError:
                pass
    return pmat

channels_all = [
    ("ECOG_1", "ECOG_1_"),
    ("ECOG_2", "ECOG_2_"),
    ("ECOG_3", "ECOG_3_"),
    ("ECOG_4", "ECOG_4_"),
    ("Laplacian D14", "LAPLACIAN_14-16_LFP_"),
]
pmats = {}
for ch_name, ch_prefix in channels_all:
    pmats[ch_name] = _compute_band_pvals(ch_prefix)
    print(f"  {ch_name} computed")

sess_labels = [s.label for s in SESSIONS]
global_zmax = max(np.nanmax(pm) for pm in pmats.values() if np.isfinite(pm).any())
x_mids = np.array([(lo + hi) / 2 for _, lo, hi in BAND_ORDER])
band_centers = [f"{int((lo + hi) / 2)}" for _, lo, hi in BAND_ORDER]

panel_map = [
    ("ECOG_1", 0, 0), ("ECOG_2", 0, 1),
    ("ECOG_3", 1, 0), ("ECOG_4", 1, 1),
    ("Laplacian D14", 2, 0),
]
letters = ["A", "B", "C", "D", "E"]

fig, axes = plt.subplots(3, 2, figsize=(9, 6.5))
axes[2, 1].axis("off")

im = None
for idx, (ch_name, r, c) in enumerate(panel_map):
    ax = axes[r, c]
    pm = pmats[ch_name]
    pm_display = np.where(np.isfinite(pm), pm, 0)

    im = ax.imshow(pm_display, aspect="auto", cmap="Greens",
                   vmin=0, vmax=global_zmax, interpolation="nearest")
    ax.set_xticks(range(len(BAND_ORDER)))
    ax.set_xticklabels(band_centers)
    ax.set_yticks(range(len(sess_labels)))
    ax.set_yticklabels(sess_labels)

    # Significance stars — auto-contrast on dark vs light cells
    for si in range(len(SESSIONS)):
        for bi in range(len(BAND_ORDER)):
            val = pm[si, bi]
            if not np.isfinite(val):
                continue
            if val > 3:   txt, clr = "***", "white"
            elif val > 2: txt, clr = "**",  "white"
            elif val > -np.log10(0.05): txt, clr = "*", "#333"
            else: continue
            ax.text(bi, si, txt, ha="center", va="center", color=clr)

    is_bottom = (r == 1 and c == 1) or (r == 2)
    if is_bottom:
        ax.set_xlabel("Frequency (Hz, band center)")
    panel_label(ax, letters[idx], ch_name)

# constrained_layout places a shared colorbar alongside the empty bottom-right
fig.colorbar(im, ax=axes[2, 1], label=r"$-\log_{10}(p)$", fraction=0.8)

fig.savefig(str(OUT / "fig_005_dbs_significance_heatmap.png"))
plt.show()
print("DBS significance heatmap saved")

# Print summary
band_labels = [f"{lo}-{hi}" for _, lo, hi in BAND_ORDER]
print(f"\n{'='*80}")
print(f"DBS significance (-log10 p, Mann\u2013Whitney U)")
print(f"{'='*80}")
for ch_name in [c[0] for c in channels_all]:
    pm = pmats[ch_name]
    print(f"\n  {ch_name}:")
    for si, sl in enumerate(sess_labels):
        vals = []
        for bi in range(len(BAND_ORDER)):
            v = pm[si, bi]
            if not np.isfinite(v): vals.append("n/a")
            elif v > 3: vals.append("***")
            elif v > 2: vals.append("**")
            elif v > -np.log10(0.05): vals.append("*")
            else: vals.append("ns")
        print(f"    {sl}: {' '.join(f'{v:>4}' for v in vals)}")

# %% [markdown]
# ## Grid alignment — behavioral resampling to 200 Hz neural grid
#
# **A** Interpolation on the common time grid (100 ms window).  
# **B** x position density: raw vs aligned (KDE, pooled across sessions).  
# **C** Sample tracing trajectory (single trial).

# %%
# Grid alignment — three standalone figures

from scipy.stats import gaussian_kde
from thesis_lib.fig_grid_alignment import _load_trial_row, _raw_and_interp_signal

GRID_DATA_ROOT = Path("resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band")
MARGIN_SAMP = 400
# Raw behavioural = DPAD-warm, aligned = PSID-blue. Neural dot uses a brighter red
# to distinguish it from the DBS-ON red elsewhere.
RAW_COLOR, ALIGNED_COLOR, NEURAL_COLOR = COLOR_DPAD, COLOR_PSID, "#C43A31"

EX_P, EX_S, EX_T = "PDI1", "2", 30
row = _load_trial_row(participant=EX_P, session=EX_S,
                      data_root=GRID_DATA_ROOT, trial_index=EX_T)
print(f"Example: {EX_P} session {EX_S}, block {row.get('block')}, trial {row.get('trial')} (idx {EX_T})")

t_raw_beh, vel_raw, t_grid_full, vel_grid = _raw_and_interp_signal(row, "tracing_velocity_x")
valid_grid = ~np.isnan(t_grid_full)
t_grid = t_grid_full[valid_grid]

x_raw = np.asarray(row["x"], dtype=float)
y_raw = np.asarray(row["y"], dtype=float)
t_motion = np.asarray(row["motion_time"], dtype=float)
order = np.argsort(t_motion)
x_raw, y_raw, t_motion = x_raw[order], y_raw[order], t_motion[order]

print(f"  ECOG_1_alpha: {len(t_grid)} pts (200 Hz)")
print(f"  tracing_velocity_x (raw): {len(t_raw_beh)} pts (~{1.0/np.median(np.diff(np.sort(t_raw_beh))):.0f} Hz)")
print(f"  tracing_velocity_x (aligned): {len(t_grid)} pts (200 Hz)")
print(f"  Tracing path: x [{np.nanmin(x_raw):.0f}, {np.nanmax(x_raw):.0f}], y [{np.nanmin(y_raw):.0f}, {np.nanmax(y_raw):.0f}]")

mid = 0.5 * (t_grid.min() + t_grid.max())
WIN_S = 0.100
win_lo, win_hi = mid - WIN_S / 2, mid + WIN_S / 2
mg = (t_grid >= win_lo) & (t_grid <= win_hi)
tg_w = (t_grid[mg] - win_lo) * 1000
mr = (t_raw_beh >= win_lo) & (t_raw_beh <= win_hi)
tr_w = (t_raw_beh[mr] - win_lo) * 1000
print(f"  Window: {WIN_S*1000:.0f} ms \u2014 {len(tg_w)} grid, {len(tr_w)} raw")

# Pooled distributions
all_x_raw, all_x_interp = [], []
for s in SESSIONS:
    gp = str(GRID_DATA_ROOT / f"participant_id={s.participant}" / f"session={s.session}" / "*" / "*.parquet")
    df = pl.read_parquet(gp, columns=["motion_time", "x", "time_original"])
    for idx in range(df.height):
        r = df.row(idx, named=True)
        t_r = np.asarray(r["motion_time"], dtype=float)
        x_r = np.asarray(r["x"], dtype=float)
        t_g = np.asarray(r["time_original"], dtype=float)
        o = np.argsort(t_r); t_r, x_r = t_r[o], x_r[o]
        m = ~np.isnan(x_r) & ~np.isnan(t_r)
        if m.sum() < 10: continue
        t_r, x_r = t_r[m], x_r[m]
        vg = ~np.isnan(t_g); t_g = t_g[vg]
        all_x_raw.append(x_r)
        all_x_interp.append(np.interp(t_g, t_r, x_r))
all_x_raw = np.concatenate(all_x_raw)
all_x_interp = np.concatenate(all_x_interp)
print(f"Pooled: {len(all_x_raw)} raw, {len(all_x_interp)} interp ({', '.join(s.label for s in SESSIONS)})")

# ── A: Interpolation on the common time grid ──
fig_a, ax_a = plt.subplots(figsize=(9, 3.2))
for t_tick in tg_w:
    ax_a.axvline(t_tick, color=(0.627, 0.627, 0.608), alpha=0.30,
                 linewidth=0.9, zorder=1)
ax_a.scatter(tg_w, np.full(len(tg_w), 2.0), color=NEURAL_COLOR, s=40,
             label="ECOG_1_alpha (200 Hz)", zorder=3)
ax_a.scatter(tr_w, np.full(len(tr_w), 1.0), color=RAW_COLOR, s=40,
             label="tracing_velocity_x raw (~119 Hz)", zorder=3)
ax_a.scatter(tg_w, np.full(len(tg_w), 0.0), color=ALIGNED_COLOR, s=40,
             label="tracing_velocity_x aligned (200 Hz)", zorder=3)
ax_a.set_yticks([0, 1, 2])
ax_a.set_yticklabels(["tracing_velocity_x\n(aligned)",
                      "tracing_velocity_x\n(raw)",
                      "ECOG_1_alpha"])
ax_a.set_ylim(-0.5, 2.5)
ax_a.set_xlim(tg_w.min() - 1, tg_w.max() + 1)
ax_a.set_xlabel("Time (ms)")
panel_label(ax_a, "A", "Interpolation on the common time grid")
fig_a.savefig(str(OUT / "fig_grid_alignment_A.png"))
plt.show()
print("Saved: fig_grid_alignment_A.png")

# ── B: x position density: raw vs aligned ──
fig_b, ax_b = plt.subplots(figsize=(9, 3.3))
lo_clip = np.percentile(np.concatenate([all_x_raw, all_x_interp]), 0.5)
hi_clip = np.percentile(np.concatenate([all_x_raw, all_x_interp]), 99.5)
kde_x_pts = np.linspace(lo_clip, hi_clip, 400)
raw_clip = all_x_raw[(all_x_raw >= lo_clip) & (all_x_raw <= hi_clip)]
interp_clip = all_x_interp[(all_x_interp >= lo_clip) & (all_x_interp <= hi_clip)]
kde_raw = gaussian_kde(raw_clip, bw_method=0.06)
kde_int = gaussian_kde(interp_clip, bw_method=0.06)
raw_y = kde_raw(kde_x_pts)
int_y = kde_int(kde_x_pts)
ax_b.fill_between(kde_x_pts, 0, raw_y, color=RAW_COLOR, alpha=0.12)
ax_b.plot(kde_x_pts, raw_y, color=RAW_COLOR, linewidth=2.0,
          label="x position (raw)")
ax_b.plot(kde_x_pts, int_y, color=ALIGNED_COLOR, linewidth=2.0,
          linestyle="--", label="x position (interpolated)")
ax_b.set_xlabel("x position (px)")
ax_b.set_ylabel("Density")
ax_b.legend(frameon=False)
panel_label(ax_b, "B", "x position density: raw vs aligned")
fig_b.savefig(str(OUT / "fig_grid_alignment_B.png"))
plt.show()
print("Saved: fig_grid_alignment_B.png")

# ── C: Sample tracing trajectory ──
fig_c, ax_c = plt.subplots(figsize=(3.3, 3.3))
ax_c.plot(x_raw, y_raw, color=RAW_COLOR, linewidth=1.5)
ax_c.set_xlabel("x (px)")
ax_c.set_ylabel("y (px)")
ax_c.set_aspect("equal", adjustable="datalim")
panel_label(ax_c, "C", "Sample tracing trajectory")
fig_c.savefig(str(OUT / "fig_grid_alignment_C.png"))
plt.show()
print("Saved: fig_grid_alignment_C.png")
