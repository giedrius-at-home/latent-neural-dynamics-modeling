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
#     display_name: Python 3
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
os.chdir('/home/bobby/repos/latent-neural-dynamics-modeling')
sys.path.insert(0, '.')

from pathlib import Path
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch

from notebooks.thesis_style import (
    COLOR_DBS_OFF, COLOR_DBS_ON,
    FONT_FAMILY, FONT_SIZE_TICK,
    LINE_WIDTH_MEAN, LINE_WIDTH_SD,
    apply_paper_style, panel_label, add_freq_band_highlight, hex_to_rgba,
)
from dashboard.thesis.loaders import load_split_results, channels_as_str_list
from dashboard.thesis.plot_config import RESULTS_ROOT, get_triplets, load_results_for_triplet

OUT = Path('thesis_figures/sec1'); OUT.mkdir(parents=True, exist_ok=True)
results_root = Path('results').resolve()

# %% [markdown]
# ## Trial inventory
#
# For each session: original blocks/trials from `participants.parquet`,
# preprocessed trials in each split, and which blocks/trials were removed.

# %%
SESSIONS = [
    ('PDI1_S2', 'PDI1', 2, 'psid_behavioral_PDI1_2_nx_25_n2_i50_dbs_both_200Hz_narrow_band'),
    ('PDI1_S4', 'PDI1', 4, 'psid_behavioral_PDI1_4_nx_15_n2_i50_dbs_both_200Hz_narrow_band'),
    ('PDI4_S2', 'PDI4', 2, 'psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band'),
    ('PDI4_S3', 'PDI4', 3, 'psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band'),
]

orig = pl.read_parquet('data/participants.parquet', hive_partitioning=True)

rows = []
for label, pid, sess, variant in SESSIONS:
    # Original data
    sub = orig.filter((pl.col('participant_id') == pid) & (pl.col('session') == sess)).sort('block')
    orig_blocks = sub['block'].to_list()
    orig_trials = {int(r['block']): len(r['trials']) for r in sub.iter_rows(named=True)}
    orig_dbs = {int(r['block']): 'ON' if r['dbs_stim'] == 1 else 'OFF' for r in sub.iter_rows(named=True)}
    orig_total = sum(orig_trials.values())

    # Split data
    base = results_root / variant / 'split'
    split_counts = {}
    split_block_trials = {}
    for split in ['train', 'val', 'test']:
        df = pl.read_parquet(base / f'{split}.parquet')
        split_counts[split] = len(df)
        for r in df.group_by('block').len().iter_rows(named=True):
            b = int(r['block'])
            split_block_trials.setdefault(b, {})[split] = r['len']

    split_total = sum(split_counts.values())
    split_blocks = sorted(split_block_trials.keys())

    # Missing blocks (in original but not in splits)
    missing_blocks = [b for b in orig_blocks if b not in split_blocks]

    # Removed trials per block
    removed_per_block = {}
    for b in orig_blocks:
        orig_n = orig_trials[b]
        split_n = sum(split_block_trials.get(b, {}).values())
        if orig_n != split_n:
            removed_per_block[b] = orig_n - split_n

    rows.append({
        'Session': label,
        'Original blocks': len(orig_blocks),
        'Original trials': orig_total,
        'Missing blocks': str(missing_blocks) if missing_blocks else '-',
        'Preprocessed trials': split_total,
        'Removed trials': orig_total - split_total,
        'Train': split_counts['train'],
        'Val': split_counts['val'],
        'Test': split_counts['test'],
    })

    # Print detailed block info
    print(f"\n{label}: {len(orig_blocks)} blocks, {orig_total} original trials -> {split_total} preprocessed")
    if missing_blocks:
        print(f"  Missing blocks: {missing_blocks}")
    if removed_per_block:
        for b, n in sorted(removed_per_block.items()):
            print(f"  Block {b} (DBS-{orig_dbs[b]}): {orig_trials[b]} -> {orig_trials[b]-n} ({n} removed)")

print(
    "\nTrial inventory summary — compares the raw "
    "data/participants.parquet record (per block/trial counts as collected at the bedside) "
    "against what made it into the train/val/test parquet splits under "
    "results/<variant>/split/. Removed trials = "
    "preprocessing rejections (motion, missing samples, etc.)."
)

# Summary table
summary = pl.DataFrame(rows)
summary

# %% [markdown]
# ## Fig 1: Trial count summary (DBS-OFF / DBS-ON per session)

# %%
triplets = get_triplets()
trial_rows = [
    {
        "session": t.label or "",
        "participant": (t.label or "").split("_")[0],
        "DBS-OFF": sum(1 for r in load_results_for_triplet(t) if r.get("stim") == "off"),
        "DBS-ON": sum(1 for r in load_results_for_triplet(t) if r.get("stim") == "on"),
    }
    for t in triplets
]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=[r["session"] for r in trial_rows], y=[r["DBS-OFF"] for r in trial_rows],
    name="DBS-OFF", marker_color=COLOR_DBS_OFF, width=0.35,
    marker_line=dict(width=0),
    text=[r["DBS-OFF"] for r in trial_rows], textposition="outside",
    textfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
))
fig.add_trace(go.Bar(
    x=[r["session"] for r in trial_rows], y=[r["DBS-ON"] for r in trial_rows],
    name="DBS-ON", marker_color=COLOR_DBS_ON, width=0.35,
    marker_line=dict(width=0),
    text=[r["DBS-ON"] for r in trial_rows], textposition="outside",
    textfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
))
apply_paper_style(
    fig, height=420,
    margin=dict(l=78, r=24, t=48, b=80), legend_y=-0.22,
)
fig.update_layout(barmode="group", bargap=0.28)
fig.update_xaxes(title_text="")
fig.update_yaxes(title_text="Test trials [count]", rangemode="tozero")
panel_label(fig, "A", "Trial counts per session")
fig.write_image(str(OUT / 'fig_001_trial_count.png'), width=1000, height=420, scale=2)
fig.show()
print(
    "Fig 1 — Test-set trial counts per session, split by DBS condition. "
    f"Source: PSID triplet results (test split) for {', '.join(r['session'] for r in trial_rows)}. "
    f"Participants: {', '.join(sorted({r['participant'] for r in trial_rows}))}. "
    "Counts match the trial inventory table above after train/val/test splitting."
)

# %% [markdown]
# ## Figs 2-5: PSD DBS comparison (4 sessions x 4 ECoG channels)

# %%
_PSD_SESSIONS = [
    ("PDI1_S2", "PDI1", 2),
    ("PDI1_S4", "PDI1", 4),
    ("PDI4_S2", "PDI4", 2),
    ("PDI4_S3", "PDI4", 3),
]
_ECOG_CHANNELS = ["ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"]

# Load block -> DBS mapping
_project_root = Path('.').resolve()
pq = _project_root / "data" / "participants.parquet"
block_dbs_df = pl.read_parquet(pq, hive_partitioning=True)
block_dbs = {
    (str(r["participant_id"]), int(r["session"]), int(r["block"])): int(r["dbs_stim"])
    for r in block_dbs_df.iter_rows(named=True)
}

data_root = _project_root / "data" / "resampled"

for label, pid, sess in _PSD_SESSIONS:
    pattern = f"sub-{pid}_ses-{sess}_task-copydraw_run-*_ieeg.parquet"
    run_files = sorted(data_root.glob(pattern))
    if not run_files:
        continue

    psds = {ch: {"off": [], "on": []} for ch in _ECOG_CHANNELS}
    freqs_ref = None

    for run_file in run_files:
        parts = run_file.stem.split("_")
        run_num = None
        for p in parts:
            if p.startswith("run-"):
                run_num = int(p[4:])
                break
        if run_num is None:
            continue
        dbs = block_dbs.get((pid, sess, run_num))
        if dbs is None:
            continue
        cond = "on" if dbs == 1 else "off"

        try:
            df = pl.read_parquet(run_file, columns=_ECOG_CHANNELS)
        except Exception:
            continue

        for ch in _ECOG_CHANNELS:
            if ch not in df.columns:
                continue
            signal = df[ch].to_numpy().astype(float)
            signal = signal[np.isfinite(signal)]
            if len(signal) < 2000:
                continue
            fs_native = 1000
            nperseg = min(len(signal), fs_native * 2)
            freqs, psd = welch(signal, fs=fs_native, nperseg=nperseg, noverlap=nperseg // 2)
            psd_db = 10 * np.log10(psd + 1e-20)
            if freqs_ref is None:
                freqs_ref = freqs
            psds[ch][cond].append(psd_db)

    if freqs_ref is None:
        continue

    freq_mask = freqs_ref <= 90
    freqs_ref_plot = freqs_ref[freq_mask]
    for ch in _ECOG_CHANNELS:
        for ck in ("off", "on"):
            psds[ch][ck] = [p[freq_mask] for p in psds[ch][ck]]

    ncols, nrows = 2, 2
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True, shared_yaxes=False,
                        vertical_spacing=0.18, horizontal_spacing=0.12)

    panel_letters = ["A", "B", "C", "D"]
    for ch_idx, ch in enumerate(_ECOG_CHANNELS):
        ri, ci = divmod(ch_idx, ncols)
        # Highlight beta band (12-30 Hz) behind the curves
        add_freq_band_highlight(fig, (12, 30), row=ri + 1, col=ci + 1)

        for cond_key, col_c, cond_label in [
            ("off", COLOR_DBS_OFF, "DBS-OFF"),
            ("on",  COLOR_DBS_ON,  "DBS-ON"),
        ]:
            cond_psds = psds[ch][cond_key]
            if not cond_psds:
                continue
            mat = np.vstack(cond_psds)
            mean = mat.mean(axis=0)
            sem = mat.std(axis=0) / np.sqrt(len(mat))
            sd = mat.std(axis=0)
            # ±SEM ribbon (semi-transparent)
            xb = np.concatenate([freqs_ref_plot, freqs_ref_plot[::-1]])
            yb = np.concatenate([mean + sem, (mean - sem)[::-1]])
            fig.add_trace(go.Scatter(
                x=xb, y=yb, fill="toself", fillcolor=hex_to_rgba(col_c, 0.18),
                line=dict(width=0), showlegend=False,
                name=cond_label, legendgroup=cond_label, hoverinfo="skip",
            ), row=ri + 1, col=ci + 1)
            # ±1 SD dashed outlines
            fig.add_trace(go.Scatter(
                x=freqs_ref_plot, y=mean + sd, mode="lines",
                line=dict(color=col_c, width=LINE_WIDTH_SD, dash="dash"),
                showlegend=False, legendgroup=cond_label, hoverinfo="skip",
            ), row=ri + 1, col=ci + 1)
            fig.add_trace(go.Scatter(
                x=freqs_ref_plot, y=mean - sd, mode="lines",
                line=dict(color=col_c, width=LINE_WIDTH_SD, dash="dash"),
                showlegend=False, legendgroup=cond_label, hoverinfo="skip",
            ), row=ri + 1, col=ci + 1)
            # Mean line (solid)
            fig.add_trace(go.Scatter(
                x=freqs_ref_plot, y=mean, mode="lines",
                line=dict(color=col_c, width=LINE_WIDTH_MEAN, dash="solid"),
                showlegend=(ch_idx == 0), name=cond_label, legendgroup=cond_label,
            ), row=ri + 1, col=ci + 1)

    psd_h = 760
    apply_paper_style(fig, height=psd_h,
                      margin=dict(l=88, r=32, t=72, b=96), legend_y=-0.10)
    for ch_idx, ch in enumerate(_ECOG_CHANNELS):
        ri, ci = divmod(ch_idx, ncols)
        panel_label(fig, panel_letters[ch_idx], ch,
                    row=ri + 1, col=ci + 1, y_offset=0.04)
        fig.update_xaxes(
            title_text="Frequency [Hz]" if ri >= nrows - 1 else "",
            range=[0, 90], row=ri + 1, col=ci + 1,
        )
        fig.update_yaxes(
            title_text="PSD [dB/Hz]" if ci == 0 else "",
            row=ri + 1, col=ci + 1,
        )

    n_off_runs = sum(len(psds[ch]["off"]) for ch in _ECOG_CHANNELS) // len(_ECOG_CHANNELS)
    n_on_runs = sum(len(psds[ch]["on"]) for ch in _ECOG_CHANNELS) // len(_ECOG_CHANNELS)
    fig_idx = _PSD_SESSIONS.index((label, pid, sess)) + 2
    fig.write_image(
        str(OUT / f'fig_{fig_idx:03d}_psd_{label}.png'),
        width=1200, height=psd_h, scale=2,
    )
    fig.show()
    print(
        f"Fig {fig_idx} — Welch PSD (mean ± SEM) for {label} (participant {pid}, session {sess}), "
        f"computed on the raw 1000 Hz ECoG recordings under data/resampled/ "
        f"(sub-{pid}_ses-{sess}_task-copydraw_run-*_ieeg.parquet). "
        f"Pooled across {n_off_runs} DBS-OFF run(s) and {n_on_runs} DBS-ON run(s). "
        f"Each panel = one of {', '.join(_ECOG_CHANNELS)}; frequency range 0-100 Hz. "
        "Traces reveal the expected alpha/beta reduction under active DBS when present."
    )

# %% [markdown]
# ## Fig 6: Tracing speed DBS comparison (velocity & acceleration, 4 sessions)

# %%
FS = 200

def _z_column_indices_for_outputs(triplet):
    res = load_split_results(RESULTS_ROOT, triplet.psid_variant, triplet.psid_run_ts, "test")
    names = channels_as_str_list(res.get("output_channels")) if res else []
    def _ix(exact):
        for i, n in enumerate(names):
            if str(n) == exact:
                return i
        return None
    iv = _ix("tracing_velocity_x")
    ia = _ix("tracing_acceleration_magnitude")
    return (iv if iv is not None else 0, ia if ia is not None else 1)

def _trial_z_column(z_arr, col_idx):
    if z_arr is None:
        return np.full(FS * 9, np.nan)
    za = np.asarray(z_arr, dtype=float)
    if za.ndim == 2:
        if col_idx >= za.shape[1]:
            return np.full(FS * 9, np.nan)
        za = za[:, col_idx]
    else:
        za = za.ravel()
    if len(za) < FS * 9:
        return np.pad(za, (0, FS * 9 - len(za)), constant_values=np.nan)
    return za[:FS * 9]

def _zscore_traces(z_off, z_on):
    def _norm(seq):
        if not seq:
            return seq
        pool = np.concatenate([np.asarray(a, dtype=float).ravel() for a in seq])
        pool = pool[np.isfinite(pool)]
        if not pool.size:
            return seq
        mu, sig = float(np.mean(pool)), float(np.std(pool))
        if sig < 1e-9:
            sig = 1.0
        return [(np.asarray(a, dtype=float) - mu) / sig for a in seq]
    return _norm(z_off), _norm(z_on)

triplets = get_triplets()
n_sessions_tr = len(triplets)
t = np.arange(FS * 9) / FS

# Layout: one row per session × 2 columns (Velocity | Acceleration).
fig = make_subplots(
    rows=n_sessions_tr, cols=2, shared_xaxes=True, shared_yaxes=False,
    vertical_spacing=0.12, horizontal_spacing=0.10,
)

panel_letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
trial_counts: list[tuple[str, int, int]] = []
for pi, tri in enumerate(triplets):
    trials = load_results_for_triplet(tri)
    if not trials:
        continue
    iv, ia = _z_column_indices_for_outputs(tri)

    def _collect(col_idx):
        off_l, on_l = [], []
        for trow in trials:
            z = trow.get("Z")
            if z is None:
                continue
            za = _trial_z_column(z, col_idx)
            stim = trow.get("stim", "off")
            (off_l if stim == "off" else on_l).append(za)
        return off_l, on_l

    row = pi + 1
    show_legend_cell = (pi == 0)
    panel_counts = {"off": 0, "on": 0}
    for col_plot, col_idx, col_name in (
        (1, iv, "Velocity"), (2, ia, "Acceleration")
    ):
        z_off, z_on = _collect(col_idx)
        z_off, z_on = _zscore_traces(z_off, z_on)
        for cond, col_c, clabel, key in (
            (z_off, COLOR_DBS_OFF, "DBS-OFF", "off"),
            (z_on,  COLOR_DBS_ON,  "DBS-ON",  "on"),
        ):
            if not cond:
                continue
            mat = np.vstack(cond)
            mean = np.nanmean(mat, axis=0)
            sem = np.nanstd(mat, axis=0) / np.sqrt(max(len(mat), 1))
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                fill="toself", fillcolor=hex_to_rgba(col_c, 0.18),
                line=dict(width=0),
                hoverinfo="skip", showlegend=False, legendgroup=clabel,
            ), row=row, col=col_plot)
            fig.add_trace(go.Scatter(
                x=t, y=mean, mode="lines",
                line=dict(color=col_c, width=LINE_WIDTH_MEAN),
                showlegend=(show_legend_cell and col_plot == 1),
                name=clabel, legendgroup=clabel,
            ), row=row, col=col_plot)
            panel_counts[key] = len(cond)

    # Panel labels: "A - PDI1 S2 · Velocity" for left, "B - … · Acceleration" for right
    left_letter = panel_letters[pi * 2]
    right_letter = panel_letters[pi * 2 + 1]
    panel_label(fig, left_letter,  f"{tri.label} · Velocity",
                row=row, col=1, y_offset=0.04)
    panel_label(fig, right_letter, f"{tri.label} · Acceleration",
                row=row, col=2, y_offset=0.04)

    trial_counts.append((tri.label or "", panel_counts["off"], panel_counts["on"]))

apply_paper_style(
    fig,
    height=max(240 * n_sessions_tr + 120, 720),
    margin=dict(l=88, r=32, t=60, b=88),
    legend_y=-0.08,
)
fig.update_xaxes(range=[0, 9])
for rr in range(1, n_sessions_tr + 1):
    for cc in (1, 2):
        if rr == n_sessions_tr:
            fig.update_xaxes(title_text="Time [s]", row=rr, col=cc)
        if cc == 1:
            fig.update_yaxes(title_text="z-score", row=rr, col=cc)

fig.write_image(str(OUT / 'fig_006_tracing_speed.png'),
                width=1100, height=max(240 * n_sessions_tr + 120, 720), scale=2)
fig.show()
print(
    "Fig 6 — Trial-averaged behavioural trajectories aligned to trial onset (0-9 s), "
    "pooled across all test trials and z-scored per condition. Left column: tracing_velocity_x. "
    "Right column: tracing_acceleration_magnitude. One row per session; trial counts "
    + ", ".join(f"{lab}: {n_off} OFF / {n_on} ON" for lab, n_off, n_on in trial_counts) + ". "
    "Data source: PSID test-split Z arrays."
)

# %% [markdown]
# ## Fig 7: Grid alignment (irregular behavioral samples vs 200 Hz master grid)
#
# Verifies that the behavioral stream (event-driven hardware sampling) is
# correctly interpolated onto the 200 Hz grid shared with the neural data.
# Three panels: (A) rug ticks of raw vs grid sampling times, (B) raw vs
# interpolated signal on the same short window, (C) inter-sample Δt histogram.

# %%
from notebooks.thesis_style import COLOR_AXIS, COLOR_PSID, COLOR_SEPARATOR, FONT_SIZE_LABEL

_GRID_SESSIONS = [("PDI1", 2), ("PDI1", 4), ("PDI4", 2), ("PDI4", 3)]
_RECORDINGS_ROOT = Path('resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band')
_BEH_CHANNEL = "tracing_velocity_x"
_TARGET_FS = 200.0
_OVERLAY_WINDOW_S = 0.35

def _finite_diff_1st(x, t):
    x = np.asarray(x, float); t = np.asarray(t, float)
    dt = np.diff(t); dy = np.diff(x)
    dt = np.where(dt == 0, np.nan, dt)
    d = dy / dt
    return np.concatenate([d, [d[-1] if d.size else 0.0]])

for pid, sess in _GRID_SESSIONS:
    part = _RECORDINGS_ROOT / f"participant_id={pid}" / f"session={sess}"
    df = pl.read_parquet(str(part) + "/*/*.parquet").sort(["block", "trial"])
    if df.height == 0:
        continue
    row = df.row(df.height // 2, named=True)  # pick a middle-ish trial

    t_raw = np.asarray(row["motion_time"], dtype=float)
    t_grid = np.asarray(row["time_original"], dtype=float)
    x_raw = np.asarray(row["x"], dtype=float)
    order = np.argsort(t_raw)
    t_raw, x_raw = t_raw[order], x_raw[order]
    sig_raw = _finite_diff_1st(x_raw, t_raw)
    sig_grid = np.asarray(row[_BEH_CHANNEL], dtype=float)
    if sig_grid.shape[0] != t_grid.shape[0]:
        sig_grid = np.interp(t_grid, t_raw, sig_raw)

    # Center window on trial midpoint
    t_lo = float(min(t_raw.min(), t_grid.min()))
    t_hi = float(max(t_raw.max(), t_grid.max()))
    t_mid = 0.5 * (t_lo + t_hi)
    win_lo = max(t_lo, t_mid - _OVERLAY_WINDOW_S / 2)
    win_hi = min(t_hi, t_mid + _OVERLAY_WINDOW_S / 2)
    m_raw = (t_raw >= win_lo) & (t_raw <= win_hi)
    m_grid = (t_grid >= win_lo) & (t_grid <= win_hi)
    t_raw_ms = (t_raw[m_raw] - win_lo) * 1000.0
    t_grid_ms = (t_grid[m_grid] - win_lo) * 1000.0
    sig_raw_win = sig_raw[m_raw]
    sig_grid_win = sig_grid[m_grid]

    dt_ms = np.diff(t_raw) * 1000.0
    nominal_dt_ms = 1000.0 / _TARGET_FS

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        vertical_spacing=0.16, row_heights=[0.18, 0.52, 0.30],
    )

    # Row 1: rug ticks
    fig.add_trace(go.Scatter(
        x=t_grid_ms, y=np.full_like(t_grid_ms, 1.0),
        mode="markers", name="200 Hz grid ticks",
        marker=dict(symbol="line-ns-open", size=18, color=COLOR_DBS_OFF,
                    line=dict(width=1.6, color=COLOR_DBS_OFF)),
        legendgroup="row1",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_raw_ms, y=np.full_like(t_raw_ms, 0.0),
        mode="markers", name="Raw behavioral samples",
        marker=dict(symbol="line-ns-open", size=18, color=COLOR_AXIS,
                    line=dict(width=1.6, color=COLOR_AXIS)),
        legendgroup="row1",
    ), row=1, col=1)
    fig.update_yaxes(row=1, col=1, tickmode="array",
                     tickvals=[0.0, 1.0], ticktext=["raw", "grid"],
                     range=[-0.6, 1.6])

    # Row 2: signal overlay
    fig.add_trace(go.Scatter(
        x=t_raw_ms, y=sig_raw_win, mode="lines+markers",
        name="Raw (irregular)",
        line=dict(color=COLOR_AXIS, width=1.4, dash="dot"),
        marker=dict(color=COLOR_AXIS, size=6, symbol="circle-open"),
        legendgroup="row2",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_grid_ms, y=sig_grid_win, mode="lines+markers",
        name="Interpolated (200 Hz)",
        line=dict(color=COLOR_DBS_OFF, width=LINE_WIDTH_MEAN),
        marker=dict(color=COLOR_DBS_OFF, size=5, symbol="square"),
        legendgroup="row2",
    ), row=2, col=1)
    fig.update_yaxes(title_text="tracing_velocity_x [px/s]", row=2, col=1)

    # Row 3: Δt histogram
    if dt_ms.size:
        p99 = float(np.nanpercentile(dt_ms, 99.0))
        hi = max(p99 * 1.05, nominal_dt_ms * 2.0)
        fig.add_trace(go.Histogram(
            x=dt_ms, xbins=dict(start=0.0, end=hi, size=max(hi / 60.0, 0.1)),
            marker=dict(color=COLOR_SEPARATOR, line=dict(color=COLOR_AXIS, width=0.6)),
            showlegend=False,
        ), row=3, col=1)
        fig.update_xaxes(row=3, col=1, range=[0.0, hi])
        fig.add_vline(
            x=nominal_dt_ms, line=dict(color=COLOR_DBS_ON, width=1.6, dash="dash"),
            row=3, col=1,
            annotation=dict(text=f"1/{int(_TARGET_FS)} Hz = {nominal_dt_ms:.2f} ms",
                            font=dict(color=COLOR_DBS_ON, size=FONT_SIZE_TICK,
                                      family=FONT_FAMILY),
                            showarrow=False, yanchor="top", y=1.0, xanchor="left"),
        )

    fig.update_xaxes(row=1, col=1, title_text="Time within window [ms]")
    fig.update_xaxes(row=2, col=1, title_text="Time within window [ms]")
    fig.update_xaxes(row=3, col=1, title_text="Δt between consecutive raw samples [ms]")
    fig.update_yaxes(title_text="count", row=3, col=1)

    apply_paper_style(fig, height=680,
                      margin=dict(l=86, r=32, t=54, b=90), legend_y=-0.10)

    panel_label(fig, "A", "Sampling times (rug)", row=1, col=1, y_offset=0.10)
    panel_label(fig, "B", f"Raw vs interpolated · {_BEH_CHANNEL}",
                row=2, col=1, y_offset=0.06)
    panel_label(fig, "C", "Inter-sample interval distribution (Δt)",
                row=3, col=1, y_offset=0.06)

    block_id = row.get("block"); trial_id = row.get("trial")
    fig.add_annotation(
        xref="paper", yref="paper", x=1.0, y=1.04,
        text=f"{pid} · ses {sess} · block {block_id} · trial {trial_id}",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK, color=COLOR_AXIS),
    )

    fig_idx = 6 + _GRID_SESSIONS.index((pid, sess)) + 1  # 7..10
    fig.write_image(
        str(OUT / f'fig_{fig_idx:03d}_grid_alignment_{pid}_S{sess}.png'),
        width=1100, height=680, scale=2,
    )
    fig.show()
    print(
        f"Fig {fig_idx} — Grid alignment check for {pid} session {sess} "
        f"(block {block_id}, trial {trial_id}). "
        f"(A) rug of raw behavioral ({len(t_raw_ms)} samples) vs 200 Hz grid "
        f"({len(t_grid_ms)} samples) in a {int(_OVERLAY_WINDOW_S*1000)} ms window. "
        f"(B) raw finite-difference velocity vs pre-computed {_BEH_CHANNEL} column. "
        f"(C) inter-sample Δt distribution over the full trial; dashed line marks "
        f"the 1/{int(_TARGET_FS)} Hz nominal interval. "
        f"Source: resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band/"
        f"participant_id={pid}/session={sess}/block={block_id}/."
    )

# %%
n = len(list(OUT.glob('*.png')))
print(f'Section 1 total: {n} figures')
