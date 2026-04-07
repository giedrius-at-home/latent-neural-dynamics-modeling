"""
Thesis appendix figures: preprocessing schematic, PSD comparison, tracing speed,
PSID grid search heatmaps (Pearson r, RMSE, lag), plus extras.
Provides Plotly build_* functions for HTML embedding and matplotlib plot_* for file export.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
from scipy.signal import welch

from dashboard.thesis.constants import (
    COLOR_CHANCE,
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_PSID,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    d_score_axis_label,
    grid_color,
    rmse_axis_label,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.loaders import channels_as_str_list, load_split_results
from dashboard.thesis.plot_config import (
    COLORS,
    RESULTS_ROOT,
    SELECTED_CONFIG,
    apply_style,
    get_triplets,
    load_results_for_triplet,
)

FS = 80
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

C_RAW = "#D3D1C7"
C_PROC = "#9FE1CB"
C_FEAT = "#AFA9EC"
C_MERGE = "#85B7EB"
BAND_SHADE_COLOR = "#E24B4A"


# ─────────────────────────────────────────────────────────────────────────────
# Plotly build_* functions (for HTML embedding)
# ─────────────────────────────────────────────────────────────────────────────


def _load_raw_ecog_block_to_dbs() -> Dict[Tuple[str, int, int], int]:
    """Return {(participant_id, session, block): dbs_stim} from participants.parquet."""
    _project_root = Path(__file__).resolve().parents[2]
    pq = _project_root / "data" / "participants.parquet"
    if not pq.exists():
        return {}
    try:
        df = pl.read_parquet(pq, hive_partitioning=True)
    except Exception:
        return {}
    out: Dict[Tuple[str, int, int], int] = {}
    for row in df.iter_rows(named=True):
        pid = str(row["participant_id"])
        sess = int(row["session"])
        block = int(row["block"])
        dbs = int(row["dbs_stim"])
        out[(pid, sess, block)] = dbs
    return out


# Sessions to show PSD for (matches thesis model sessions)
_PSD_SESSIONS = [
    ("PDI1_S2", "PDI1", 2),
    ("PDI1_S4", "PDI1", 4),
    ("PDI4_S2", "PDI4", 2),
    ("PDI4_S3", "PDI4", 3),
]
_ECOG_CHANNELS = ["ECOG_1", "ECOG_2", "ECOG_3", "ECOG_4"]


def build_psd_dbs_comparison_figures() -> List[Tuple[str, Figure]]:
    """Return one (label, Figure) per participant-session.

    Each figure has one subplot per ECoG electrode (4 total), with
    DBS-OFF and DBS-ON mean PSD traces and ±1 SEM bands.
    Uses raw ECoG data from data/resampled/ parquets.
    """
    from dashboard.thesis.constants import apply_thesis_style

    _project_root = Path(__file__).resolve().parents[2]
    data_root = _project_root / "data" / "resampled"
    block_dbs = _load_raw_ecog_block_to_dbs()

    OFF_C, ON_C = COLOR_DBS_OFF, COLOR_DBS_ON
    figures: List[Tuple[str, Figure]] = []

    for label, pid, sess in _PSD_SESSIONS:
        # Find all run parquets for this participant-session
        pattern = f"sub-{pid}_ses-{sess}_task-copydraw_run-*_ieeg.parquet"
        run_files = sorted(data_root.glob(pattern))
        if not run_files:
            continue

        # Collect per-channel PSDs split by DBS condition
        psds: Dict[str, Dict[str, List[np.ndarray]]] = {
            ch: {"off": [], "on": []} for ch in _ECOG_CHANNELS
        }
        freqs_ref: Optional[np.ndarray] = None

        for run_file in run_files:
            # Extract run number from filename
            stem = run_file.stem  # sub-PDI1_ses-2_task-copydraw_run-5_ieeg
            parts = stem.split("_")
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
                # Native sampling rate from the parquet (1000 Hz)
                fs_native = 1000
                nperseg = min(len(signal), fs_native * 2)
                freqs, psd = welch(signal, fs=fs_native, nperseg=nperseg, noverlap=nperseg // 2)
                psd_db = 10 * np.log10(psd + 1e-20)
                if freqs_ref is None:
                    freqs_ref = freqs
                psds[ch][cond].append(psd_db)

        if freqs_ref is None:
            continue

        # Trim to 0–100 Hz for display
        freq_mask = freqs_ref <= 100
        freqs_ref = freqs_ref[freq_mask]
        for ch in _ECOG_CHANNELS:
            for cond_key in ("off", "on"):
                psds[ch][cond_key] = [p[freq_mask] for p in psds[ch][cond_key]]

        n_channels = len(_ECOG_CHANNELS)
        ncols = 2
        nrows = (n_channels + ncols - 1) // ncols
        subplot_titles = list(_ECOG_CHANNELS) + ([""] * (nrows * ncols - n_channels))

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.10,
            horizontal_spacing=0.10,
            subplot_titles=subplot_titles,
        )

        for ch_idx, ch in enumerate(_ECOG_CHANNELS):
            ri, ci = divmod(ch_idx, ncols)

            for cond_key, col_c, cond_label, line_dash in [
                ("off", OFF_C, "DBS-OFF", "dash"),
                ("on", ON_C, "DBS-ON", "solid"),
            ]:
                cond_psds = psds[ch][cond_key]
                if not cond_psds:
                    continue
                mat = np.vstack(cond_psds)
                mean = mat.mean(axis=0)
                sem = mat.std(axis=0) / np.sqrt(len(mat))
                # SEM band
                xb = np.concatenate([freqs_ref, freqs_ref[::-1]])
                yb = np.concatenate([mean + sem, (mean - sem)[::-1]])
                r_c, g_c, b_c = int(col_c[1:3], 16), int(col_c[3:5], 16), int(col_c[5:7], 16)
                fc = f"rgba({r_c},{g_c},{b_c},0.18)"
                fig.add_trace(
                    go.Scatter(
                        x=xb, y=yb, fill="toself", fillcolor=fc,
                        line=dict(width=0), showlegend=False,
                        name=cond_label, legendgroup=cond_label, hoverinfo="skip",
                    ),
                    row=ri + 1, col=ci + 1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=freqs_ref, y=mean, mode="lines",
                        line=dict(color=col_c, width=2.2, dash=line_dash),
                        showlegend=(ch_idx == 0), name=cond_label,
                        legendgroup=cond_label,
                    ),
                    row=ri + 1, col=ci + 1,
                )

        apply_thesis_style(
            fig, ThesisTheme.LIGHT,
            height=max(280 * nrows + 80, 480),
            margin=dict(l=72, r=40, t=56, b=100),
            legend_y=-0.08,
        )
        fig.update_annotations(font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY))

        # Axis labels for edge subplots
        for ch_idx in range(n_channels):
            ri, ci = divmod(ch_idx, ncols)
            fig.update_xaxes(
                title_text="Frequency (Hz)" if ri >= nrows - 1 else "",
                title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                row=ri + 1, col=ci + 1,
            )
            fig.update_yaxes(
                title_text="PSD (dB/Hz)" if ci == 0 else "",
                title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                row=ri + 1, col=ci + 1,
            )

        figures.append((label, fig))

    return figures


def _z_column_indices_for_outputs(triplet) -> tuple[int, int]:
    """Match saved ``output_channels`` names (YAML ``output:`` order)."""
    res = load_split_results(
        RESULTS_ROOT, triplet.psid_variant, triplet.psid_run_ts, "test"
    )
    names = channels_as_str_list(res.get("output_channels")) if res else []

    def _ix(exact: str) -> int | None:
        for i, n in enumerate(names):
            if str(n) == exact:
                return i
        return None

    iv = _ix("tracing_velocity_x")
    ia = _ix("tracing_acceleration_magnitude")
    return (iv if iv is not None else 0, ia if ia is not None else 1)


def _trial_z_column(z_arr: Any, col_idx: int) -> np.ndarray:
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
    return za[: FS * 9]


def _zscore_traces(z_off: list, z_on: list) -> tuple[list, list]:
    """Z-score each DBS condition using only that condition’s pooled samples (per output feature)."""

    def _norm(seq: list) -> list:
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


def build_tracing_speed_dbs_comparison_figure() -> Figure:
    triplets = get_triplets()
    n_panels, ncols = len(triplets), 2
    nrows_pair = (n_panels + ncols - 1) // ncols
    nrows_plot = 2 * nrows_pair
    t = np.arange(FS * 9) / FS
    subplot_titles: list[str] = []
    for rp in range(nrows_pair):
        for c in range(ncols):
            pi = rp * ncols + c
            lab = (triplets[pi].label or "") if pi < n_panels else ""
            subplot_titles.append(
                f"{lab} — velocity" if lab else ""
            )
        for c in range(ncols):
            pi = rp * ncols + c
            lab = (triplets[pi].label or "") if pi < n_panels else ""
            subplot_titles.append(
                f"{lab} — acceleration" if lab else ""
            )
    fig = make_subplots(
        rows=nrows_plot,
        cols=ncols,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
        subplot_titles=subplot_titles,
    )
    OFF_C, ON_C = COLOR_DBS_OFF, COLOR_DBS_ON

    for pi, tri in enumerate(triplets):
        r_pair, c = divmod(pi, ncols)
        r_vel = 2 * r_pair + 1
        r_acc = r_vel + 1
        trials = load_results_for_triplet(tri)
        if not trials:
            continue
        iv, ia = _z_column_indices_for_outputs(tri)

        def _collect(col_idx: int) -> tuple[list, list]:
            off_l, on_l = [], []
            for row in trials:
                z = row.get("Z")
                if z is None:
                    continue
                za = _trial_z_column(z, col_idx)
                stim = row.get("stim", "off")
                (off_l if stim == "off" else on_l).append(za)
            return off_l, on_l

        for row_plot, col_idx, y_title, metric_tag in (
            (r_vel, iv, d_score_axis_label("tracing_velocity", "magnitude"), "vel"),
            (r_acc, ia, d_score_axis_label("tracing_acceleration", "magnitude"), "acc"),
        ):
            z_off, z_on = _collect(col_idx)
            z_off, z_on = _zscore_traces(z_off, z_on)
            for cond, col_c, label in [(z_off, OFF_C, "DBS-OFF"), (z_on, ON_C, "DBS-ON")]:
                if not cond:
                    continue
                mat = np.vstack(cond)
                mean = np.nanmean(mat, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=mean,
                        mode="lines",
                        line=dict(color=col_c, width=2.0),
                        showlegend=(pi == 0),
                        name=f"{label} · {metric_tag}",
                        legendgroup=f"{label}_{metric_tag}",
                    ),
                    row=row_plot,
                    col=c + 1,
                )
            if c == 0:
                fig.update_yaxes(title_text=y_title, row=row_plot, col=c + 1)

    theme = ThesisTheme.LIGHT
    paper_bg, plot_bg = paper_colors(theme)
    _grid = grid_color(theme)
    _fg = true_line_color(theme)

    apply_thesis_style(
        fig, theme,
        height=max(260 * nrows_plot + 100, 480),
        margin=dict(b=120, t=64, l=88, r=40),
        legend_y=-0.06,
    )
    fig.update_annotations(font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY))
    fig.update_xaxes(range=[0, 9])
    for rp in range(nrows_pair):
        for c in range(ncols):
            for ri_off in (2 * rp + 1, 2 * rp + 2):
                fig.update_xaxes(
                    showline=True, linecolor=_fg, linewidth=1,
                    tickfont=dict(size=FONT_SIZE_TICK),
                    title_text="time (s)" if (rp >= nrows_pair - 1 and ri_off == 2 * rp + 2) else "",
                    title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                    row=ri_off,
                    col=c + 1,
                )
                fig.update_yaxes(
                    showgrid=True, gridcolor=_grid,
                    showline=True, linecolor=_fg, linewidth=1,
                    tickfont=dict(size=FONT_SIZE_TICK),
                    title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
                    row=ri_off,
                    col=c + 1,
                )
    return fig


def _load_classification_scores() -> pl.DataFrame:
    """Load balanced CV accuracy from all grid search + ablation classification pickles."""
    import pickle as _pkl

    rows = []
    cls_root = RESULTS_ROOT / "classification"
    for source_dir in ["psid_grid_search_narrow_band", "psid_ablation_narrow_band"]:
        base = cls_root / source_dir
        if not base.exists():
            continue
        for pkl_path in base.rglob("LDA_*_prediction.pkl"):
            feat = pkl_path.stem.replace("LDA_", "").replace("_prediction", "")
            run_dir = pkl_path.parent.parent  # up past the run_ts dir
            config_path = Path(__file__).resolve().parents[2] / "classification" / "setups" / f"{run_dir.name}.yaml"
            if not config_path.exists():
                continue
            text = config_path.read_text()
            m_nx = re.search(r"nx:\s*(\d+)", text)
            m_n1 = re.search(r"n1:\s*(\d+)", text)
            if not m_nx or not m_n1:
                continue
            nx, n1 = int(m_nx.group(1)), int(m_n1.group(1))
            # Extract participant/session
            m_ps = re.search(r"(PDI\d+)_S(\d+)", run_dir.name)
            if not m_ps:
                continue
            pid, sess = m_ps.group(1), m_ps.group(2)
            try:
                with open(pkl_path, "rb") as f:
                    res = _pkl.load(f)
                score = res.get("best_cv_score", float("nan"))
            except Exception:
                continue
            rows.append({"participant_id": pid, "session": sess, "nx": nx, "n1": n1, "feature": feat, "score": score})
    return pl.DataFrame(rows) if rows else pl.DataFrame({"participant_id": [], "session": [], "nx": [], "n1": [], "feature": [], "score": []})


_CLASSIFICATION_SESSIONS = [
    ("PDI1", "2", "PDI1_S2"),
    ("PDI1", "4", "PDI1_S4"),
    ("PDI4", "2", "PDI4_S2"),
    ("PDI4", "3", "PDI4_S3"),
]


def build_classification_heatmap_figure() -> Figure:
    """Unified heatmap of DBS classification balanced accuracy (grid search + ablation).

    3 columns (Xp, Xp_1, Xp_2) x 4 rows (sessions). nx on x-axis, n1 on y-axis.
    """
    df = _load_classification_scores()
    features = ["Xp", "Xp_1", "Xp_2"]
    sessions = _CLASSIFICATION_SESSIONS
    nrows, ncols = len(sessions), len(features)

    subplot_titles = [
        f"{label}  —  {feat}" for label in [s[2] for s in sessions] for feat in features
    ]
    fig = make_subplots(
        rows=nrows, cols=ncols,
        vertical_spacing=0.10, horizontal_spacing=0.06,
        subplot_titles=subplot_titles,
    )

    # Discover combined axis values from the data
    all_nx = sorted(df["nx"].unique().to_list()) if not df.is_empty() else [1, 2, 5, 60, 65, 70, 75, 80]
    all_n1 = sorted(df["n1"].unique().to_list()) if not df.is_empty() else [1, 2, 5, 6, 8, 10, 12]

    # Global color range across all panels
    global_min, global_max = 0.45, 0.70
    if not df.is_empty():
        vals = df["score"].drop_nulls().drop_nans()
        if len(vals) > 0:
            global_min = max(0.40, float(vals.min()) - 0.02)
            global_max = min(1.0, float(vals.max()) + 0.02)

    for ri, (pid, sess, label) in enumerate(sessions):
        for ci, feat in enumerate(features):
            sub = df.filter(
                (pl.col("participant_id") == pid)
                & (pl.col("session") == sess)
                & (pl.col("feature") == feat)
            )
            # Deduplicate: keep best score per (nx, n1)
            if not sub.is_empty():
                sub = sub.sort("score", descending=True).unique(subset=["nx", "n1"], keep="first")

            mat = np.full((len(all_n1), len(all_nx)), np.nan)
            text_mat = np.empty_like(mat, dtype=object)
            for n1i, n1v in enumerate(all_n1):
                for xi, nxv in enumerate(all_nx):
                    if n1v > nxv:
                        text_mat[n1i, xi] = ""
                        continue
                    match = sub.filter((pl.col("nx") == nxv) & (pl.col("n1") == n1v))
                    if not match.is_empty():
                        val = float(match["score"][0])
                        mat[n1i, xi] = val
                        text_mat[n1i, xi] = f"{val:.2f}"
                    else:
                        text_mat[n1i, xi] = ""

            x_idx = np.arange(len(all_nx), dtype=float)
            y_idx = np.arange(len(all_n1), dtype=float)
            fig.add_trace(
                go.Heatmap(
                    z=mat, x=x_idx, y=y_idx,
                    text=text_mat, texttemplate="%{text}",
                    textfont=dict(size=9),
                    colorscale="Blues",
                    zmin=global_min, zmax=global_max,
                    showscale=(ri == 0 and ci == ncols - 1),
                    colorbar=dict(title="Bal. acc.", len=0.3, y=0.85) if (ri == 0 and ci == ncols - 1) else None,
                ),
                row=ri + 1, col=ci + 1,
            )
            fig.update_xaxes(
                title_text="n<sub>x</sub>" if ri == nrows - 1 else "",
                tickmode="array", tickvals=x_idx,
                ticktext=[str(v) for v in all_nx],
                row=ri + 1, col=ci + 1,
            )
            fig.update_yaxes(
                title_text="n<sub>1</sub>" if ci == 0 else "",
                tickmode="array", tickvals=y_idx,
                ticktext=[str(v) for v in all_n1],
                row=ri + 1, col=ci + 1,
            )
            # Mark the selected config
            sel_key = f"{pid}_S{sess}"
            picked = SELECTED_CONFIG.get(sel_key)
            if picked is not None:
                nx_pick, n1_pick = picked
                if nx_pick in all_nx and n1_pick in all_n1:
                    bx = all_nx.index(nx_pick)
                    by = all_n1.index(n1_pick)
                    fig.add_shape(
                        type="rect",
                        x0=float(bx) - 0.5, x1=float(bx) + 0.5,
                        y0=float(by) - 0.5, y1=float(by) + 0.5,
                        line=dict(color=COLOR_CHANCE, width=2.5),
                        fillcolor="rgba(0,0,0,0)", layer="above",
                        row=ri + 1, col=ci + 1,
                    )

    apply_thesis_style(
        fig, ThesisTheme.LIGHT,
        height=250 * nrows,
        margin=dict(l=60, r=60, t=40, b=60),
        show_legend=False,
    )
    fig.update_annotations(font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY))
    return fig


def build_trial_count_summary_figure() -> Figure:
    triplets = get_triplets()
    rows = [{"session": t.label or "", "DBS-OFF": sum(1 for r in load_results_for_triplet(t) if r.get("stim") == "off"),
             "DBS-ON": sum(1 for r in load_results_for_triplet(t) if r.get("stim") == "on")} for t in triplets]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[r["session"] for r in rows], y=[r["DBS-OFF"] for r in rows],
        name="DBS-OFF", marker_color=COLOR_DBS_OFF, width=0.35,
        text=[r["DBS-OFF"] for r in rows], textposition="auto",
    ))
    fig.add_trace(go.Bar(
        x=[r["session"] for r in rows], y=[r["DBS-ON"] for r in rows],
        name="DBS-ON", marker_color=COLOR_DBS_ON, width=0.35,
        text=[r["DBS-ON"] for r in rows], textposition="auto",
    ))
    theme = ThesisTheme.LIGHT
    apply_thesis_style(fig, theme, height=320, margin=dict(l=72, r=24, t=36, b=60), legend_y=-0.20)
    fig.update_layout(
        barmode="group",
        xaxis=dict(title_text="", showgrid=False),
        yaxis=dict(
            title_text="Trial count",
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Ablation grid search, vanilla comparison, Laplacian LFP figures
# ─────────────────────────────────────────────────────────────────────────────



def build_vanilla_comparison_figure() -> Figure:
    """Grouped bar chart comparing improved vs vanilla PSID across sessions."""
    import json as _json

    fg = true_line_color(ThesisTheme.LIGHT)
    grd = grid_color(ThesisTheme.LIGHT)

    sessions = [
        ("PDI1_S2", "psid_behavioral_PDI1_2_nx_80_n12_i40_dbs_both_narrow_band",
         "psid_behavioral_PDI1_2_nx_80_n12_i40_vanilla_dbs_both_narrow_band"),
        ("PDI1_S4", "psid_behavioral_PDI1_4_nx_80_n6_i40_dbs_both_narrow_band",
         "psid_behavioral_PDI1_4_nx_80_n6_i40_vanilla_dbs_both_narrow_band"),
        ("PDI4_S2", "psid_behavioral_PDI4_2_nx_80_n10_i40_dbs_both_narrow_band",
         "psid_behavioral_PDI4_2_nx_80_n10_i40_vanilla_dbs_both_narrow_band"),
        ("PDI4_S3", "psid_behavioral_PDI4_3_nx65_n10_i40_dbs_both_narrow_band",
         "psid_behavioral_PDI4_3_nx65_n10_i40_vanilla_dbs_both_narrow_band"),
    ]

    labels: List[str] = []
    r_improved_z, r_vanilla_z = [], []
    rmse_improved_z, rmse_vanilla_z = [], []
    rmse_improved_y, rmse_vanilla_y = [], []

    for label, improved_dir, vanilla_dir in sessions:
        for name, result_dir, r_list, rmse_z_list, rmse_y_list in [
            ("improved", improved_dir, r_improved_z, rmse_improved_z, rmse_improved_y),
            ("vanilla", vanilla_dir, r_vanilla_z, rmse_vanilla_z, rmse_vanilla_y),
        ]:
            res_path = RESULTS_ROOT / result_dir
            if not res_path.exists():
                r_list.append(np.nan)
                rmse_z_list.append(np.nan)
                rmse_y_list.append(np.nan)
                continue
            train_dir = res_path / "train"
            parquets = list(train_dir.glob("test_results_*.parquet")) if train_dir.exists() else []
            if not parquets:
                r_list.append(np.nan)
                rmse_z_list.append(np.nan)
                rmse_y_list.append(np.nan)
                continue
            try:
                tdf = pl.read_parquet(parquets[0])
                # Pearson r (Z)
                for col in ["metric_pearson_r_mean_Z", "pearson_mean_Z"]:
                    if col in tdf.columns:
                        vals = tdf[col].to_list()
                        valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                        r_list.append(float(np.mean(valid)) if valid else np.nan)
                        break
                else:
                    r_list.append(np.nan)
                # RMSE (Z)
                z_true = np.array(tdf["Z"][0].to_list())
                z_pred = np.array(tdf["Zp"][0].to_list())
                rmse_z_list.append(float(np.sqrt(np.mean((z_true - z_pred) ** 2))))
                # RMSE (Y) — neural reconstruction
                if "Y" in tdf.columns and "Yp" in tdf.columns:
                    y_true = np.array(tdf["Y"][0].to_list())
                    y_pred = np.array(tdf["Yp"][0].to_list())
                    rmse_y_list.append(float(np.sqrt(np.mean((y_true - y_pred) ** 2))))
                else:
                    rmse_y_list.append(np.nan)
            except Exception:
                r_list.append(np.nan)
                rmse_z_list.append(np.nan)
                rmse_y_list.append(np.nan)
        labels.append(label)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Pearson r", "RMSE(z)", "RMSE(z) — neural"],
        horizontal_spacing=0.10,
    )

    _c_imp, _c_van = COLOR_PSID, "#D97706"
    fig.add_trace(go.Bar(x=labels, y=r_improved_z, name="Improved (BK + rescale)",
                          marker_color=_c_imp, width=0.35), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=r_vanilla_z, name="Vanilla PSID",
                          marker_color=_c_van, width=0.35), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=rmse_improved_z, name="Improved (BK + rescale)",
                          marker_color=_c_imp, width=0.35, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=rmse_vanilla_z, name="Vanilla PSID",
                          marker_color=_c_van, width=0.35, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=labels, y=rmse_improved_y, name="Improved (BK + rescale)",
                          marker_color=_c_imp, width=0.35, showlegend=False), row=1, col=3)
    fig.add_trace(go.Bar(x=labels, y=rmse_vanilla_y, name="Vanilla PSID",
                          marker_color=_c_van, width=0.35, showlegend=False), row=1, col=3)

    from dashboard.thesis.constants import apply_thesis_style
    apply_thesis_style(fig, ThesisTheme.LIGHT, height=420, margin=dict(l=60, r=40, t=56, b=80))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Pearson r", showgrid=True, gridcolor=grd, row=1, col=1)
    fig.update_yaxes(title_text=rmse_axis_label(), showgrid=True, gridcolor=grd, row=1, col=2)
    fig.update_yaxes(title_text="RMSE(z) — neural", showgrid=True, gridcolor=grd, row=1, col=3)
    return fig


def build_laplacian_prediction_figure() -> Figure:
    """Summary bar chart of Laplacian LFP prediction quality across sessions."""
    fg = true_line_color(ThesisTheme.LIGHT)
    grd = grid_color(ThesisTheme.LIGHT)

    lapl_sessions = [
        ("PDI1_S2", "psid_laplacian_PDI1_2_nx_80_n12_i20_dbs_both_2hz_band"),
        ("PDI1_S4", "psid_laplacian_PDI1_4_nx_80_n6_i20_dbs_both_2hz_band"),
        ("PDI4_S2", "psid_laplacian_PDI4_2_nx_80_n10_i20_dbs_both_2hz_band"),
        ("PDI4_S3", "psid_laplacian_PDI4_3_nx_65_n10_i20_dbs_both_2hz_band"),
    ]

    labels, r_y_vals, r_z_vals = [], [], []
    for label, result_dir in lapl_sessions:
        res_path = RESULTS_ROOT / result_dir / "train"
        parquets = list(res_path.glob("test_results_*.parquet")) if res_path.exists() else []
        if not parquets:
            r_y_vals.append(np.nan)
            r_z_vals.append(np.nan)
            labels.append(label)
            continue
        try:
            tdf = pl.read_parquet(parquets[0])
            r_y_vals.append(float(tdf["metric_pearson_r_mean"][0]))
            r_z_vals.append(float(tdf["metric_pearson_r_mean_Z"][0]))
        except Exception:
            r_y_vals.append(np.nan)
            r_z_vals.append(np.nan)
        labels.append(label)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=r_y_vals, name="Y (ECoG recon.)", marker_color=COLOR_PSID, width=0.35))
    fig.add_trace(go.Bar(x=labels, y=r_z_vals, name="Z (Laplacian pred.)", marker_color=COLOR_CHANCE, width=0.35))

    apply_thesis_style(fig, ThesisTheme.LIGHT, height=400, margin=dict(l=72, r=24, t=36, b=60), legend_y=-0.20)
    fig.update_layout(
        barmode="group",
        yaxis=dict(title_text="Pearson r", range=[0, 1.05]),
    )
    return fig


def build_laplacian_timeseries_figure() -> Figure:
    """Time series plot of Laplacian prediction for one example trial per session."""
    fg = true_line_color(ThesisTheme.LIGHT)

    lapl_sessions = [
        ("PDI1_S2", "psid_laplacian_PDI1_2_nx_80_n12_i20_dbs_both_2hz_band"),
        ("PDI1_S4", "psid_laplacian_PDI1_4_nx_80_n6_i20_dbs_both_2hz_band"),
        ("PDI4_S2", "psid_laplacian_PDI4_2_nx_80_n10_i20_dbs_both_2hz_band"),
        ("PDI4_S3", "psid_laplacian_PDI4_3_nx_65_n10_i20_dbs_both_2hz_band"),
    ]

    nrows = len(lapl_sessions)
    _c_true = true_line_color(ThesisTheme.LIGHT)
    # Set colorway explicitly so it starts with our palette — prevents template override
    fig = make_subplots(rows=nrows, cols=1, vertical_spacing=0.08,
                        subplot_titles=[s[0] for s in lapl_sessions])
    fig.update_layout(colorway=[_c_true, COLOR_PSID])

    for ri, (label, result_dir) in enumerate(lapl_sessions, 1):
        res_path = RESULTS_ROOT / result_dir / "train"
        parquets = list(res_path.glob("test_results_*.parquet")) if res_path.exists() else []
        if not parquets:
            fig.add_annotation(text="No data", x=0.5, y=0.5, xref="x domain", yref="y domain",
                               showarrow=False, row=ri, col=1)
            continue
        try:
            tdf = pl.read_parquet(parquets[0])
            z_true = np.array(tdf["Z"][0].to_list())
            z_pred = np.array(tdf["Zp"][0].to_list())
            # Plot first Laplacian channel (delta band)
            t = np.arange(z_true.shape[0]) / 80.0
            _c_true = true_line_color(ThesisTheme.LIGHT)
            fig.add_trace(go.Scatter(x=t, y=z_true[:, 0], name="y_true",
                                      legendgroup="y_true",
                                      marker=dict(color=_c_true),
                                      line=dict(color=_c_true, width=1.8),
                                      showlegend=(ri == 1)), row=ri, col=1)
            fig.add_trace(go.Scatter(x=t, y=z_pred[:, 0], name="y_hat_PSID",
                                      legendgroup="y_hat_PSID",
                                      marker=dict(color=COLOR_PSID),
                                      line=dict(color=COLOR_PSID, width=2.0, dash="dash"),
                                      showlegend=(ri == 1)), row=ri, col=1)
        except Exception:
            fig.add_annotation(text="Error loading", x=0.5, y=0.5, xref="x domain", yref="y domain",
                               showarrow=False, row=ri, col=1)

    from dashboard.thesis.constants import apply_thesis_style
    apply_thesis_style(
        fig, ThesisTheme.LIGHT,
        height=200 * nrows + 80,
        margin=dict(l=72, r=40, t=56, b=80),
        legend_y=-0.08,
    )
    # Force trace colors after apply_thesis_style (colorway can override explicit colors)
    _c_true = true_line_color(ThesisTheme.LIGHT)
    for trace in fig.data:
        if trace.legendgroup == "y_true":
            trace.line.color = _c_true
        elif trace.legendgroup == "y_hat_PSID":
            trace.line.color = COLOR_PSID
    fig.update_xaxes(title_text="Time (s)", row=nrows, col=1)
    fig.update_yaxes(title_text="z-score", title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY), col=1, row=nrows // 2 + 1)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib plot_* functions (for PDF/PNG file export)
# ─────────────────────────────────────────────────────────────────────────────

def _box(ax, cx: float, cy: float, w: float, h: float, label: str, sublabel: str, color: str, fontsize: float = 8.5) -> None:
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#888780", linewidth=0.6, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(cx, cy + (0.12 if sublabel else 0), label, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center", fontsize=7.0, color="#5F5E5A", zorder=4)


def _arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444441", lw=1.1), zorder=2)


def plot_preprocessing_pipeline(save: bool = True) -> plt.Figure:
    """Two-branch flowchart: neural (left) and behavioural (right) → Model input."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _box(ax, 2.5, 9.2, 2.8, 0.7, "Raw iEEG", "22 kHz · LFP 16ch + ECoG 4ch", C_RAW)
    _arrow(ax, 2.5, 8.85, 2.5, 8.1)
    _box(ax, 2.5, 7.7, 2.8, 0.7, "Resample + bandpass", "↓ 60 Hz · 3–28 Hz BP (MNE)", C_PROC)
    _arrow(ax, 2.5, 7.35, 2.5, 6.6)
    _box(ax, 2.5, 6.2, 2.8, 0.7, "Common avg. re-reference", "Per modality, per trial", C_PROC)
    _arrow(ax, 2.5, 5.85, 2.5, 5.1)
    _box(ax, 2.5, 4.7, 2.8, 0.7, "Scale ×10⁶", "→ microvolts", C_PROC)
    _arrow(ax, 2.5, 4.35, 2.5, 3.6)
    _box(ax, 2.5, 3.2, 2.8, 0.9, "Narrowband features",
         "δ/θ/α raw · β 13–29 Hz Hilbert env\n4 ch × 29 bands = 116 inputs (Y)", C_FEAT, fontsize=8.0)

    _box(ax, 7.5, 9.2, 2.8, 0.7, "Tablet coordinates", "x(t), y(t) at variable rate", C_RAW)
    _arrow(ax, 7.5, 8.85, 7.5, 8.1)
    _box(ax, 7.5, 7.7, 2.8, 0.7, "Kinematic derivation", "v, a, j in x/y/xy/mag", C_PROC)
    _arrow(ax, 7.5, 7.35, 7.5, 6.6)
    _box(ax, 7.5, 6.2, 2.8, 0.7, "Savitzky-Golay smooth", "~200 ms window · 3rd order", C_PROC)
    _arrow(ax, 7.5, 5.85, 7.5, 5.1)
    _box(ax, 7.5, 4.7, 2.8, 0.7, "Interpolate to 60 Hz", "Linear onto neural grid", C_PROC)
    _arrow(ax, 7.5, 4.35, 7.5, 3.6)
    _box(ax, 7.5, 3.2, 2.8, 0.7, "Tracing speed", "velocity magnitude (Z)", C_FEAT)

    seg_rect = mpatches.FancyBboxPatch(
        (1.0, 2.35), 8.0, 0.55,
        boxstyle="round,pad=0.04",
        facecolor="#FAEEDA", edgecolor="#854F0B", linewidth=0.7, zorder=2,
    )
    ax.add_patch(seg_rect)
    ax.text(5.0, 2.625, "Trial segmentation: 9 s trial · ±2 s margin buffer",
            ha="center", va="center", fontsize=8.0, color="#412402", zorder=4)
    _arrow(ax, 2.5, 2.75, 2.5, 2.35)
    _arrow(ax, 7.5, 2.75, 7.5, 2.35)
    _arrow(ax, 2.5, 2.35, 4.5, 1.75)
    _arrow(ax, 7.5, 2.35, 5.5, 1.75)
    _box(ax, 5.0, 1.45, 3.2, 0.7, "Model input (Y, Z) at 60 Hz", "Train 60% · Val 20% · Test 20%", C_MERGE, fontsize=8.5)

    for i, (c, lbl) in enumerate([(C_RAW, "Raw input"), (C_PROC, "Processing step"), (C_FEAT, "Feature / output"), (C_MERGE, "Model input")]):
        px = 0.55 + i * 2.3
        r = mpatches.FancyBboxPatch((px, 0.15), 0.35, 0.28, boxstyle="round,pad=0.03", facecolor=c, edgecolor="#888780", lw=0.5)
        ax.add_patch(r)
        ax.text(px + 0.45, 0.29, lbl, va="center", fontsize=7.5, color="#444441")

    fig.suptitle("Data preprocessing pipeline", fontsize=12, y=0.97)
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"preprocessing_pipeline.{ext}", bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PSD comparison: DBS-ON vs DBS-OFF
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trial_psd(signal_1d: np.ndarray, fs: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(len(signal_1d), fs * 2)
    freqs, psd = welch(signal_1d, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return freqs, 10 * np.log10(psd + 1e-20)


def plot_psd_dbs_comparison(
    band_shade: Optional[Tuple[float, float]] = (13, 29),
    save: bool = True,
) -> plt.Figure:
    """Mean ± SEM PSD per condition, per participant×session."""
    apply_style()
    triplets = get_triplets()
    n_panels = len(triplets)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    figsize = (4.5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True, gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    axes_flat = np.atleast_2d(axes).ravel()
    OFF_C, ON_C = COLOR_DBS_OFF, COLOR_DBS_ON

    for pi, tri in enumerate(triplets):
        ax = axes_flat[pi]
        trials = load_results_for_triplet(tri)
        if not trials:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue

        psds_off, psds_on = [], []
        freqs_ref = None
        for row in trials:
            y = row.get("Y")
            stim = row.get("stim", "off")
            if y is None:
                continue
            y_arr = np.asarray(y)
            channel_mean = y_arr.mean(axis=1) if y_arr.ndim == 2 else np.asarray(y).ravel()
            freqs, psd_db = _compute_trial_psd(channel_mean, fs=FS)
            if freqs_ref is None:
                freqs_ref = freqs
            if stim == "off":
                psds_off.append(psd_db)
            else:
                psds_on.append(psd_db)

        if freqs_ref is None:
            continue
        for cond_psds, col, label in [(psds_off, OFF_C, "DBS-OFF"), (psds_on, ON_C, "DBS-ON")]:
            if not cond_psds:
                continue
            mat = np.vstack(cond_psds)
            mean = mat.mean(axis=0)
            sem = mat.std(axis=0) / np.sqrt(len(mat))
            ax.fill_between(freqs_ref, mean - sem, mean + sem, color=col, alpha=0.15, lw=0)
            ax.plot(freqs_ref, mean, color=col, lw=1.8, label=label)
        if band_shade is not None:
            ax.axvspan(band_shade[0], band_shade[1], color=BAND_SHADE_COLOR, alpha=0.05)
            ax.axvline(band_shade[0], color=BAND_SHADE_COLOR, lw=0.8, ls="--", alpha=0.6)
            ax.axvline(band_shade[1], color=BAND_SHADE_COLOR, lw=0.8, ls="--", alpha=0.6)
        ax.text(0.04, 0.95, tri.label or f"{tri.psid_variant}", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        if pi % ncols == 0:
            ax.set_ylabel("power/freq (dB/Hz)")
        if pi >= n_panels - ncols:
            ax.set_xlabel("frequency (Hz)")

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    handles = [plt.Line2D([], [], color=OFF_C, lw=1.8, label="DBS-OFF mean"), plt.Line2D([], [], color=ON_C, lw=1.8, label="DBS-ON mean")]
    if band_shade is not None:
        handles.append(plt.Line2D([], [], color=BAND_SHADE_COLOR, lw=0.8, ls="--", label=f"band {band_shade[0]}-{band_shade[1]} Hz"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("ECoG PSD: DBS-ON vs DBS-OFF", fontsize=12, y=1.01)
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"psd_dbs_comparison.{ext}", bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tracing speed: DBS-ON vs DBS-OFF
# ─────────────────────────────────────────────────────────────────────────────

def plot_tracing_speed_dbs_comparison(save: bool = True) -> plt.Figure:
    """Mean traces for ``tracing_velocity_x`` and ``tracing_acceleration_magnitude`` (YAML output names)."""
    apply_style()
    triplets = get_triplets()
    n_panels = len(triplets)
    ncols = 2
    nrows_pair = (n_panels + ncols - 1) // ncols
    nrows_plot = 2 * nrows_pair
    figsize = (4.5 * ncols, 2.65 * nrows_plot)
    t = np.arange(FS * 9) / FS
    fig, axes = plt.subplots(
        nrows_plot,
        ncols,
        figsize=figsize,
        sharex="col",
        sharey=False,
        gridspec_kw={"hspace": 0.35, "wspace": 0.12},
    )
    axes = np.atleast_2d(axes)
    OFF_C, ON_C = COLOR_DBS_OFF, COLOR_DBS_ON
    n_slot = nrows_pair * ncols

    for idx in range(n_slot):
        r_pair, c = divmod(idx, ncols)
        r0 = 2 * r_pair
        ax_v, ax_a = axes[r0, c], axes[r0 + 1, c]
        vis = idx < n_panels
        ax_v.set_visible(vis)
        ax_a.set_visible(vis)
        if not vis:
            continue
        tri = triplets[idx]
        trials = load_results_for_triplet(tri)
        iv, ia = _z_column_indices_for_outputs(tri)

        def _collect(col_idx: int) -> tuple[list, list]:
            off_l, on_l = [], []
            for row in trials:
                z = row.get("Z")
                if z is None:
                    continue
                za = _trial_z_column(z, col_idx)
                stim = row.get("stim", "off")
                (off_l if stim == "off" else on_l).append(za)
            return off_l, on_l

        ax_v.set_title(f"{tri.label or tri.psid_variant}\ntracing_velocity_x", fontsize=8.5)
        ax_a.set_title("tracing_acceleration_magnitude", fontsize=8.5)

        if not trials:
            for ax in (ax_v, ax_a):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
            continue

        for ax, col_idx, ylbl in (
            (ax_v, iv, d_score_axis_label("tracing_velocity", "x")),
            (ax_a, ia, d_score_axis_label("tracing_acceleration", "magnitude")),
        ):
            z_off, z_on = _collect(col_idx)
            z_off, z_on = _zscore_traces(z_off, z_on)
            for cond_list, col, label in [(z_off, OFF_C, "DBS-OFF"), (z_on, ON_C, "DBS-ON")]:
                if not cond_list:
                    continue
                mat = np.vstack(cond_list)
                mean = np.nanmean(mat, axis=0)
                ax.plot(t, mean, color=col, lw=1.8, label=f"{label} (n={len(cond_list)})")
            ax.set_xlim(0, 9)
            if c == 0:
                ax.set_ylabel(ylbl, fontsize=8)

    for c in range(ncols):
        axes[nrows_plot - 1, c].set_xlabel("time (s)", fontsize=8)

    handles = [
        plt.Line2D([], [], color=OFF_C, lw=1.8, label="DBS-OFF"),
        plt.Line2D([], [], color=ON_C, lw=1.8, label="DBS-ON"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "tracing_velocity_x and tracing_acceleration_magnitude (trial means, z per metric)",
        fontsize=11,
        y=1.01,
    )
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"tracing_speed_dbs_comparison.{ext}", bbox_inches="tight")
    return fig




# ─────────────────────────────────────────────────────────────────────────────
# Trial count summary
# ─────────────────────────────────────────────────────────────────────────────

def plot_trial_count_summary(save: bool = True) -> plt.Figure:
    """Bar chart: trial counts per participant×session×condition."""
    apply_style()
    triplets = get_triplets()
    rows = []
    for tri in triplets:
        trials = load_results_for_triplet(tri)
        off = sum(1 for t in trials if t.get("stim") == "off")
        on = sum(1 for t in trials if t.get("stim") == "on")
        rows.append({"session": tri.label or "", "DBS-OFF": off, "DBS-ON": on, "total": off + on})

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(rows))
    w = 0.35
    ax.bar(x - w / 2, [r["DBS-OFF"] for r in rows], w, label="DBS-OFF", color=COLORS["dbs_off"])
    ax.bar(x + w / 2, [r["DBS-ON"] for r in rows], w, label="DBS-ON", color=COLORS["dbs_on"])
    ax.set_xticks(x)
    ax.set_xticklabels([r["session"] for r in rows])
    ax.set_ylabel("trial count")
    ax.legend()
    fig.suptitle("Trial count per session × DBS condition", fontsize=12)
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"trial_count_summary.{ext}", bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Placeholders
# ─────────────────────────────────────────────────────────────────────────────

def plot_beta_burst_placeholder(save: bool = True) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, "Beta burst duration/amplitude: requires burst detection pipeline.",
            ha="center", va="center", transform=ax.transAxes, fontsize=10, wrap=True)
    ax.axis("off")
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"beta_burst_placeholder.{ext}", bbox_inches="tight")
    return fig


def plot_residual_diagnostics_placeholder(save: bool = True) -> plt.Figure:
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, "Residual diagnostics (Ljung-Box): requires residual extraction from model.",
            ha="center", va="center", transform=ax.transAxes, fontsize=10, wrap=True)
    ax.axis("off")
    if save:
        for ext in ("pdf", "png"):
            fig.savefig(OUT_DIR / f"residual_diagnostics_placeholder.{ext}", bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis appendix figures (matplotlib)")
    parser.add_argument("--figures", nargs="*",
                        default=["preprocessing", "psd", "tracing", "grid_pearson", "grid_rmse", "grid_lag", "trial_count", "beta_burst", "residual"],
                        help="Which figures to generate")
    parser.add_argument("--no-save", action="store_true", help="Show only, do not save")
    parser.add_argument("--band-shade", nargs=2, type=float, default=[13, 29], metavar=("LO", "HI"))
    args = parser.parse_args()

    save = not args.no_save
    band_shade = (float(args.band_shade[0]), float(args.band_shade[1]))

    for f in args.figures:
        if f == "preprocessing":
            plot_preprocessing_pipeline(save=save)
        elif f == "psd":
            plot_psd_dbs_comparison(band_shade=band_shade, save=save)
        elif f == "tracing":
            plot_tracing_speed_dbs_comparison(save=save)
        elif f == "grid_pearson":
            plot_grid_search_pearson(save=save)
        elif f == "grid_rmse":
            plot_grid_search_rmse(save=save)
        elif f == "grid_lag":
            plot_grid_search_lag(save=save)
        elif f == "trial_count":
            plot_trial_count_summary(save=save)
        elif f == "beta_burst":
            plot_beta_burst_placeholder(save=save)
        elif f == "residual":
            plot_residual_diagnostics_placeholder(save=save)
        else:
            print(f"Unknown figure: {f}")

    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
