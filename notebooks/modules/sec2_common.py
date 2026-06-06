"""Shared helpers for sec2c (reconstruction) and sec2d (forecast) notebooks.

Each notebook provides its own `load_fn(variant, run_ts, split) -> dict` and
`score_fn(res, trial_i, ch_i, metric) -> float` so the collectors work for
both reconstruction parquets and forecast parquets without code duplication.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from dataclasses import dataclass
from pathlib import Path
from scipy.signal import hilbert
from scipy.stats import pearsonr, gaussian_kde

from modules.style import (
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_PSID,
    COLOR_DPAD,
    COLOR_VARMA,
    panel_label,
)
from modules.loaders import (
    discover_session_run,
    load_split_results_required,
    load_precomputed_results,
)
from modules.utils import normalize_stim
from modules.lib.transforms import reshape_future_z_time_first


@dataclass
class SessionObj:
    label: str
    psid_variant: str = ""
    psid_run_ts: str = ""
    dpad_variant: str = ""
    dpad_run_ts: str = ""
    varma_variant: str = ""
    varma_run_ts: str = ""


def make_session_obj(results_root, session, exp_type):
    pv, pt = discover_session_run(results_root, "psid", exp_type, session)
    dv, dt = discover_session_run(results_root, "dpad", exp_type, session)
    vv, vt = discover_session_run(results_root, "varma", exp_type, session)
    return SessionObj(
        label=session,
        psid_variant=pv or "",
        psid_run_ts=pt or "",
        dpad_variant=dv or "",
        dpad_run_ts=dt or "",
        varma_variant=vv or "",
        varma_run_ts=vt or "",
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = ["PSID", "DPAD", "VARMA"]
MODEL_COLS = {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}
BAND_ORDER = ["theta", "alpha", "beta", "gamma"]
CH_TYPES = [("raw", "oscillatory\n(raw)"), ("env", "amplitude\n(env)")]
SAMPLING_HZ = 200.0

# ---------------------------------------------------------------------------
# Hilbert metric functions
# ---------------------------------------------------------------------------


def amp_r(t_sig, p_sig):
    ht = hilbert(t_sig)
    hp = hilbert(p_sig)
    return float(pearsonr(np.abs(ht), np.abs(hp))[0])


def phase_plv(t_sig, p_sig):
    ht = hilbert(t_sig)
    hp = hilbert(p_sig)
    return float(np.abs(np.mean(np.exp(1j * (np.angle(ht) - np.angle(hp))))))


def inst_freq_r(t_sig, p_sig):
    ht = hilbert(t_sig)
    hp = hilbert(p_sig)
    return float(
        pearsonr(np.diff(np.unwrap(np.angle(ht))), np.diff(np.unwrap(np.angle(hp))))[0]
    )


DECOMP_METRICS = [
    ("amplitude_r", "r (amplitude)", 0.0, amp_r),
    ("inst_freq_r", "r (inst. freq)", 0.0, inst_freq_r),
    ("phase_plv", "PLV", None, phase_plv),
]

# ---------------------------------------------------------------------------
# Channel name helpers
# ---------------------------------------------------------------------------


def load_channel_names(results_root, session_label, exp_type, feat_col):
    """Read Z_features / Y_features list from PSID train parquet."""
    var, ts = discover_session_run(results_root, "psid", exp_type, session_label)
    if not var or not ts:
        return []
    pq = Path(results_root) / "psid" / var / "train" / f"test_results_{ts}.parquet"
    if not pq.exists():
        return []
    df = pl.read_parquet(str(pq))
    if feat_col not in df.columns:
        return []
    return df[0][feat_col][0].to_list()


def parse_band(ch_name):
    n = ch_name.lower()
    for b in BAND_ORDER:
        if b in n:
            return b
    return "unknown"


# ---------------------------------------------------------------------------
# Trial iterator — core loop shared by all collectors
# ---------------------------------------------------------------------------


def _iter_trial_channels(
    session_objs,
    exp_type,
    target,
    split,
    load_fn,
    ch_names_fn,
    *,
    raw_only=False,
    all_ch=False,
):
    """Yield (session_label, model_name, stim, ch_idx, ch_name, true_vec, pred_vec).

    load_fn(variant, run_ts, split) -> results dict with keys:
        target key (truth) and target+'p' key (pred), plus 'stim'.
    ch_names_fn(session_label) -> list[str]
    raw_only: only channels whose name ends in '_raw'
    all_ch:   use all channels regardless of suffix (overrides raw_only)
    """
    true_key = target
    pred_key = target + "p"
    for tri in session_objs:
        ch_names = ch_names_fn(tri.label)
        if not ch_names:
            continue
        if all_ch:
            use_idx = list(range(len(ch_names)))
        elif raw_only:
            use_idx = [i for i, n in enumerate(ch_names) if n.endswith("_raw")]
        else:
            use_idx = list(range(len(ch_names)))
        if not use_idx:
            continue
        for model_name, variant, run_ts in [
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ]:
            if not variant or not run_ts:
                continue
            try:
                res = load_fn(variant, run_ts, split)
            except Exception:
                continue
            true_list = res.get(true_key, [])
            pred_list = res.get(pred_key, [])
            if not true_list:
                continue
            for i in range(len(true_list)):
                stim = normalize_stim(res["stim"][i])
                if stim is None:
                    continue
                z = np.array(true_list[i], dtype=float)
                zp = np.array(pred_list[i] if pred_list else [], dtype=float)
                if z.ndim == 1:
                    z = z[:, None]
                if zp.ndim == 1:
                    zp = zp[:, None] if zp.size else np.full_like(z, np.nan)
                for ch in use_idx:
                    if ch >= z.shape[1] or ch >= zp.shape[1]:
                        continue
                    t, p = z[:, ch], zp[:, ch]
                    mask = np.isfinite(t) & np.isfinite(p)
                    if mask.sum() < 20:
                        continue
                    yield tri.label, model_name, stim, ch, ch_names[ch], t[mask], p[
                        mask
                    ]


# ---------------------------------------------------------------------------
# Metric collectors (use _iter_trial_channels)
# ---------------------------------------------------------------------------


def collect_scalar_metric(
    session_objs, exp_type, target, metric, split, load_fn, score_fn, ch_names_fn
):
    """dict[(session, model, stim)] -> 1-D array of per-trial-channel values."""
    out = {}
    for tri in session_objs:
        ch_names = ch_names_fn(tri.label)
        if not ch_names:
            continue
        for model_name, variant, run_ts in [
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ]:
            if not variant or not run_ts:
                continue
            try:
                res = load_fn(variant, run_ts, split)
            except Exception:
                continue
            n_trials = len(res.get(target, []))
            for i in range(n_trials):
                stim = normalize_stim(res["stim"][i])
                if stim is None:
                    continue
                ch_vals = []
                for ch in range(len(ch_names)):
                    try:
                        v = score_fn(res, i, ch, metric)
                    except Exception:
                        continue
                    if np.isfinite(v):
                        ch_vals.append(float(v))
                if ch_vals:
                    key = (tri.label, model_name, stim)
                    out.setdefault(key, []).append(float(np.mean(ch_vals)))
    return {k: np.array(v) for k, v in out.items()}


def collect_rawenv_metrics(
    session_objs, exp_type, target, metric, split, load_fn, score_fn, ch_names_fn
):
    """Pooled across sessions: dict[(model, 'raw'|'env')] -> array."""
    out = {}
    for tri in session_objs:
        ch_names = ch_names_fn(tri.label)
        if not ch_names:
            continue
        ch_types = ["env" if n.endswith("_env") else "raw" for n in ch_names]
        for model_name, variant, run_ts in [
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ]:
            if not variant or not run_ts:
                continue
            try:
                res = load_fn(variant, run_ts, split)
            except Exception:
                continue
            for i in range(len(res.get(target, []))):
                if normalize_stim(res["stim"][i]) is None:
                    continue
                for ch, ctype in enumerate(ch_types):
                    try:
                        v = score_fn(res, i, ch, metric)
                    except Exception:
                        continue
                    if np.isfinite(v):
                        out.setdefault((model_name, ctype), []).append(float(v))
    return {k: np.array(v) for k, v in out.items()}


def collect_rawenv_metrics_per_session(
    session_objs, exp_type, target, metric, split, load_fn, score_fn, ch_names_fn
):
    """Per-session: dict[(session, model, 'raw'|'env')] -> array."""
    out = {}
    for tri in session_objs:
        ch_names = ch_names_fn(tri.label)
        if not ch_names:
            continue
        ch_types = ["env" if n.endswith("_env") else "raw" for n in ch_names]
        for model_name, variant, run_ts in [
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ]:
            if not variant or not run_ts:
                continue
            try:
                res = load_fn(variant, run_ts, split)
            except Exception:
                continue
            for i in range(len(res.get(target, []))):
                if normalize_stim(res["stim"][i]) is None:
                    continue
                for ch, ctype in enumerate(ch_types):
                    try:
                        v = score_fn(res, i, ch, metric)
                    except Exception:
                        continue
                    if np.isfinite(v):
                        out.setdefault((tri.label, model_name, ctype), []).append(
                            float(v)
                        )
    return {k: np.array(v) for k, v in out.items()}


def collect_hilbert_metrics_per_session(
    session_objs, exp_type, target, split, load_fn, ch_names_fn
):
    """Per-session Hilbert amplitude r + PLV (raw channels only).
    dict[(session, model, 'amplitude_r'|'phase_plv')] -> array."""
    out = {}
    for sess, model, stim, ch, ch_name, t, p in _iter_trial_channels(
        session_objs, exp_type, target, split, load_fn, ch_names_fn, raw_only=True
    ):
        if len(t) < 50:
            continue
        ht = hilbert(t)
        hp = hilbert(p)
        try:
            ar = float(pearsonr(np.abs(ht), np.abs(hp))[0])
        except Exception:
            ar = np.nan
        plv = float(np.abs(np.mean(np.exp(1j * (np.angle(ht) - np.angle(hp))))))
        if np.isfinite(ar):
            out.setdefault((sess, model, "amplitude_r"), []).append(ar)
        if np.isfinite(plv):
            out.setdefault((sess, model, "phase_plv"), []).append(plv)
    return {k: np.array(v) for k, v in out.items()}


def collect_decomp_metrics(
    session_objs, exp_type, target, split, load_fn, ch_names_fn, raw_only=True
):
    """Pooled amplitude_r, inst_freq_r, phase_plv.
    dict[(model, metric_name)] -> array."""
    out = {}
    for sess, model, stim, ch, ch_name, t, p in _iter_trial_channels(
        session_objs,
        exp_type,
        target,
        split,
        load_fn,
        ch_names_fn,
        raw_only=raw_only,
        all_ch=(not raw_only),
    ):
        if len(t) < 50:
            continue
        for metric_name, _, _, fn in DECOMP_METRICS:
            try:
                v = fn(t, p)
                if np.isfinite(v):
                    out.setdefault((model, metric_name), []).append(v)
            except Exception:
                pass
    return {k: np.array(v) for k, v in out.items()}


def collect_band_grouped_metrics(
    session_objs, exp_type, target, split, load_fn, ch_names_fn
):
    """Single pass: returns (corr_raw, corr_env, plv) keyed by (model, band)."""
    corr_raw, corr_env, plv_dict = {}, {}, {}
    for sess, model, stim, ch, ch_name, t, p in _iter_trial_channels(
        session_objs, exp_type, target, split, load_fn, ch_names_fn
    ):
        band = parse_band(ch_name)
        ctype = "env" if ch_name.endswith("_env") else "raw"
        try:
            r = float(pearsonr(t, p)[0])
        except Exception:
            r = np.nan
        if np.isfinite(r):
            (corr_raw if ctype == "raw" else corr_env).setdefault(
                (model, band), []
            ).append(r)
        if ctype == "raw" and len(t) >= 50:
            ht = hilbert(t)
            hp = hilbert(p)
            plv = float(np.abs(np.mean(np.exp(1j * (np.angle(ht) - np.angle(hp))))))
            if np.isfinite(plv):
                plv_dict.setdefault((model, band), []).append(plv)
    return (
        {k: np.array(v) for k, v in corr_raw.items()},
        {k: np.array(v) for k, v in corr_env.items()},
        {k: np.array(v) for k, v in plv_dict.items()},
    )


def collect_per_feature_means(
    session_objs, exp_type, target, metric, split, load_fn, score_fn, ch_names_fn
):
    """Mean metric per feature per session per model.
    Returns ({(session, model): array[n_feat]}, {session: [feat_names]})."""
    raw_vals = {}
    feat_map = {}
    for tri in session_objs:
        feat_names = ch_names_fn(tri.label)
        if not feat_names:
            continue
        feat_map[tri.label] = feat_names
        for model_name, variant, run_ts in [
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ]:
            if not variant or not run_ts:
                continue
            try:
                res = load_fn(variant, run_ts, split)
            except Exception:
                continue
            for i in range(len(res.get(target, []))):
                if normalize_stim(res["stim"][i]) is None:
                    continue
                for fi in range(len(feat_names)):
                    try:
                        v = score_fn(res, i, fi, metric)
                    except Exception:
                        continue
                    if np.isfinite(v):
                        raw_vals.setdefault((tri.label, model_name, fi), []).append(
                            float(v)
                        )
    result = {
        (sess, model): np.array(
            [
                np.mean(raw_vals.get((sess, model, fi), [np.nan]))
                for fi in range(len(feat_names))
            ]
        )
        for sess, feat_names in feat_map.items()
        for model in MODELS
    }
    return result, feat_map


# ---------------------------------------------------------------------------
# DBS pooling
# ---------------------------------------------------------------------------


def pool_dbs_cells(per_cell_data, cells, *, models=None):
    """dict[(cell, model, dbs)] -> dict[(model, dbs)] pooled across cells."""
    if models is None:
        models = MODELS
    out = {}
    for m in models:
        for d in ("off", "on"):
            buf = [
                float(v)
                for cell in cells
                for v in (per_cell_data.get((cell, m, d)) or [])
                if np.isfinite(v)
            ]
            out[(m, d)] = np.array(buf, dtype=float)
    return out


# ---------------------------------------------------------------------------
# Raincloud primitives
# ---------------------------------------------------------------------------


def kde_half_violin_vert(
    ax,
    vals,
    *,
    x_anchor,
    color,
    alpha,
    hatch=None,
    logy=False,
    bw=0.55,
    width=0.18,
    pad=1.0,
    npts=400,
    side="right",
):
    """Vertical half-violin grown from x_anchor outward."""
    y_in = np.log10(vals) if logy else np.asarray(vals, dtype=float)
    if y_in.size < 2 or np.allclose(y_in.std(), 0.0):
        return
    kde = gaussian_kde(y_in, bw_method=bw)
    s = y_in.std()
    ys = np.linspace(y_in.min() - pad * s, y_in.max() + pad * s, npts)
    dens = kde(ys)
    if dens.max() <= 0:
        return
    dens = dens / dens.max() * width
    ys_plot = np.power(10.0, ys) if logy else ys
    xs_l = np.full_like(dens, x_anchor) if side == "right" else x_anchor - dens
    xs_r = x_anchor + dens if side == "right" else np.full_like(dens, x_anchor)
    ax.fill_betweenx(
        ys_plot,
        xs_l,
        xs_r,
        facecolor=color,
        alpha=alpha,
        edgecolor=(color if hatch else "none"),
        linewidth=(0.6 if hatch else 0.0),
        hatch=hatch,
        zorder=2,
    )


def raincloud_slot(
    ax,
    pos,
    vals,
    *,
    color,
    alpha=0.65,
    side="right",
    logy=False,
    rng,
    strip_cap=80,
    kde_width=0.18,
):
    """Half-violin + box + strip at position pos."""
    strip_x = pos - 0.08 if side == "left" else pos + 0.08
    box_x = pos - 0.20 if side == "left" else pos + 0.20
    anchor = pos - 0.30 if side == "left" else pos + 0.30
    kde_half_violin_vert(
        ax,
        vals,
        x_anchor=anchor,
        color=color,
        alpha=alpha,
        logy=logy,
        bw=0.55,
        width=kde_width,
        side=side,
    )
    ax.boxplot(
        [vals],
        positions=[box_x],
        widths=0.08,
        vert=True,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.0),
        medianprops=dict(color=color, linewidth=1.4),
        whiskerprops=dict(color=color, linewidth=1.0),
        capprops=dict(color=color, linewidth=1.0),
    )
    sv = (
        vals[rng.choice(len(vals), strip_cap, replace=False)]
        if len(vals) > strip_cap
        else vals
    )
    ax.scatter(
        strip_x + rng.uniform(-0.04, 0.04, len(sv)),
        sv,
        c=color,
        alpha=min(alpha, 0.35),
        s=7,
        linewidths=0,
    )


# ---------------------------------------------------------------------------
# Standard figure builders
# ---------------------------------------------------------------------------


def dbs_raincloud_fig(pool_r, pool_n, *, models=None, model_cols=None, figsize=None):
    """Per-model subplots: 2 rows (r, RMSE) x N cols (models). Each subplot has DBS-OFF left, DBS-ON right."""
    if models is None:
        models = MODELS
    if model_cols is None:
        model_cols = MODEL_COLS
    n = len(models)
    if figsize is None:
        figsize = (2.8 * n, 7)
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, n, figsize=figsize, layout="constrained")
    axes = np.atleast_2d(axes)
    for col, model in enumerate(models):
        panel_label(axes[0, col], "ABC"[col], model)
        for row, (pool, ylabel, logy) in enumerate(
            [
                (pool_r, "Pearson r", False),
                (pool_n, "NRMSE (log)", True),
            ]
        ):
            ax = axes[row, col]
            if logy:
                ax.set_yscale("log")
            for d, color, side in [
                ("off", COLOR_DBS_OFF, "left"),
                ("on", COLOR_DBS_ON, "right"),
            ]:
                vals = pool.get((model, d), np.array([]))
                if len(vals) < 2:
                    continue
                if logy:
                    vals = vals[vals > 0]
                if len(vals) < 2:
                    continue
                raincloud_slot(
                    ax, 0, vals, color=color, alpha=0.45, side=side, logy=logy, rng=rng
                )
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(ylabel)
    from matplotlib.patches import Patch

    fig.legend(
        [
            Patch(facecolor=COLOR_DBS_OFF, alpha=0.7, edgecolor="black", linewidth=0.6),
            Patch(facecolor=COLOR_DBS_ON, alpha=0.7, edgecolor="black", linewidth=0.6),
        ],
        ["DBS-OFF", "DBS-ON"],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.subplots_adjust(bottom=0.10)
    return fig


def decomp_fig(data, *, models=None, model_cols=None):
    """3x3 grid: rows=metrics (amp/inst-freq/PLV), cols=models."""
    if models is None:
        models = MODELS
    if model_cols is None:
        model_cols = MODEL_COLS
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(3, 3, figsize=(9, 7), layout="constrained", sharey="row")
    for col, model in enumerate(models):
        for row, (metric, ylabel, ref, _) in enumerate(DECOMP_METRICS):
            ax = axes[row, col]
            vals = data.get((model, metric), np.array([]))
            if len(vals) > 1:
                raincloud_slot(
                    ax,
                    0,
                    vals,
                    color=model_cols[model],
                    alpha=0.65,
                    side="right",
                    rng=rng,
                    strip_cap=80,
                )
            if ref is not None:
                ax.axhline(ref, color="#bbb", lw=0.8, ls=":")
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.6)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                panel_label(ax, "ABC"[col], model)
    return fig


def band_raincloud_fig(
    ax,
    data,
    ylabel,
    *,
    ref_line=0.0,
    legend=False,
    models=None,
    model_cols=None,
    rng_seed=42,
):
    """Band-grouped raincloud on a single axes."""
    if models is None:
        models = MODELS
    if model_cols is None:
        model_cols = MODEL_COLS
    rng = np.random.default_rng(rng_seed)
    tick_pos, tick_lab, pos = [], [], 0
    for band in BAND_ORDER:
        g0 = pos
        for model in models:
            vals = data.get((model, band), np.array([]))
            if len(vals) > 1:
                raincloud_slot(
                    ax,
                    pos,
                    vals,
                    color=model_cols[model],
                    alpha=0.65,
                    side="right",
                    rng=rng,
                    strip_cap=60,
                )
            pos += 1
        tick_pos.append((g0 + pos - 1) / 2)
        tick_lab.append(band)
        pos += 1
    if ref_line is not None:
        ax.axhline(ref_line, color="#bbb", lw=0.8, ls=":")
    ax.set_ylabel(ylabel)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_xlim(-0.8, pos - 1.2)
    if legend:
        from matplotlib.patches import Patch

        ax.legend([Patch(fc=model_cols[m], alpha=0.72) for m in models], models)


def hilbert_raincloud_sessions_fig(
    data, session_labels, *, models=None, model_cols=None, figsize=None
):
    """Per-model subplots: 2 rows (amplitude r, PLV) x N cols (models). Sessions pooled."""
    if models is None:
        models = MODELS
    if model_cols is None:
        model_cols = MODEL_COLS
    rng = np.random.default_rng(42)
    from matplotlib.patches import Patch

    n_models = len(models)
    if figsize is None:
        figsize = (2.8 * n_models, 7)
    metrics = [("amplitude_r", "r (amplitude)", 0.0), ("phase_plv", "PLV", None)]
    fig, axes = plt.subplots(2, n_models, figsize=figsize, layout="constrained")
    axes = np.atleast_2d(axes)
    for col, model in enumerate(models):
        panel_label(axes[0, col], "ABC"[col], model)
        for row, (metric, ylabel, ref) in enumerate(metrics):
            ax = axes[row, col]
            # pool all sessions for this (model, metric)
            pooled = []
            for sess in session_labels:
                v = data.get((sess, model, metric), np.array([]))
                if len(v) > 0:
                    pooled.append(v)
            vals = np.concatenate(pooled) if pooled else np.array([])
            if len(vals) > 1:
                raincloud_slot(
                    ax,
                    0,
                    vals,
                    color=model_cols[model],
                    alpha=0.65,
                    side="right",
                    rng=rng,
                    strip_cap=60,
                )
            if ref is not None:
                ax.axhline(ref, color="#bbb", lw=0.8, ls=":")
            ax.set_xticks([])
            ax.set_ylabel(ylabel)
    return fig


def rawenv_raincloud_sessions_fig(
    data_r, data_n, session_labels, *, models=None, model_cols=None, figsize=None
):
    """Per-model subplots: 2 rows (r, NRMSE) x N cols (models). Sessions pooled, raw vs env per subplot."""
    if models is None:
        models = MODELS
    if model_cols is None:
        model_cols = MODEL_COLS
    rng = np.random.default_rng(42)
    from matplotlib.patches import Patch

    n_models = len(models)
    if figsize is None:
        figsize = (2.8 * n_models, 7)
    fig, axes = plt.subplots(2, n_models, figsize=figsize, layout="constrained")
    axes = np.atleast_2d(axes)
    for col, model in enumerate(models):
        panel_label(axes[0, col], "ABC"[col], model)
        for row, (pool, ylabel) in enumerate(
            [
                (data_r, "r"),
                (data_n, "NRMSE"),
            ]
        ):
            ax = axes[row, col]
            for pos, (ttype, tlabel) in enumerate(CH_TYPES):
                pooled = []
                for sess in session_labels:
                    v = pool.get((sess, model, ttype), np.array([]))
                    if len(v) > 0:
                        pooled.append(v)
                vals = np.concatenate(pooled) if pooled else np.array([])
                if len(vals) > 1:
                    raincloud_slot(
                        ax,
                        pos,
                        vals,
                        color=model_cols[model],
                        alpha=0.65,
                        side="right",
                        rng=rng,
                        strip_cap=60,
                    )
            ax.set_xticks(range(len(CH_TYPES)))
            ax.set_xticklabels([t[1] for t in CH_TYPES])
            ax.set_xlim(-0.6, len(CH_TYPES) - 0.4)
            ax.set_ylabel(ylabel)
    return fig


# ---------------------------------------------------------------------------
# Horizon / step-level metric functions
# ---------------------------------------------------------------------------


def per_step_rmse_and_pearson(T, P):
    """(rmse, sem, pearson_r) per horizon step. T, P: (n_trials, n_steps)."""
    n_trials, n_steps = T.shape
    err = T - P
    rmse = np.sqrt(np.mean(err**2, axis=0))
    sem = np.std(err**2, axis=0, ddof=1) / np.sqrt(n_trials) / (2 * rmse + 1e-12)
    rs = np.full(n_steps, np.nan)
    for s in range(n_steps):
        t, p = T[:, s], P[:, s]
        if np.std(t) > 1e-12 and np.std(p) > 1e-12 and n_trials > 1:
            rs[s] = float(np.corrcoef(t, p)[0, 1])
    return rmse, sem, rs


def horizon_quantile_bands(T, P, window_size=20, sampling_hz=200.0):
    """Per-row windowed metrics -> dict with median and Q10/Q90 bands.
    T, P: (n_rows, n_steps). Returns None if fewer than 1 window."""
    n_rows, n_steps = T.shape
    n_windows = n_steps // window_size
    if n_windows == 0:
        return None
    x_ms = np.array(
        [(w + 0.5) * window_size / sampling_hz * 1000 for w in range(n_windows)]
    )
    rows_r = np.full((n_rows, n_windows), np.nan)
    rows_rmse = np.full((n_rows, n_windows), np.nan)
    for w in range(n_windows):
        sl = slice(w * window_size, (w + 1) * window_size)
        t_w, p_w = T[:, sl], P[:, sl]
        rows_rmse[:, w] = np.sqrt(np.mean((t_w - p_w) ** 2, axis=1))
        for i in range(n_rows):
            if np.std(t_w[i]) < 1e-12:
                continue
            try:
                rows_r[i, w] = float(pearsonr(t_w[i], p_w[i])[0])
            except Exception:
                pass
    return dict(
        x_ms=x_ms,
        med_r=np.nanmedian(rows_r, axis=0),
        q10_r=np.nanpercentile(rows_r, 10, axis=0),
        q90_r=np.nanpercentile(rows_r, 90, axis=0),
        med_rmse=np.nanmedian(rows_rmse, axis=0),
        q10_rmse=np.nanpercentile(rows_rmse, 10, axis=0),
        q90_rmse=np.nanpercentile(rows_rmse, 90, axis=0),
    )


def horizon_decay_fig(
    horizon_data,
    panel_r="Pearson r",
    panel_n="NRMSE",
    *,
    models=None,
    model_cols=None,
    figsize=None,
):
    """Per-model subplots: 2 rows (r, NRMSE) x N cols (models). Solid=OFF, dashed=ON."""
    if models is None:
        models = MODELS
    if model_cols is None:
        model_cols = MODEL_COLS
    dbs_style = {"off": "-", "on": "--"}
    n_models = len(models)
    if figsize is None:
        figsize = (3.2 * n_models, 7)
    fig, axes = plt.subplots(
        2, n_models, figsize=figsize, sharex=True, layout="constrained"
    )
    axes = np.atleast_2d(axes)
    for col, model in enumerate(models):
        c = model_cols[model]
        panel_label(axes[0, col], "ABC"[col], model)
        ax_r, ax_n = axes[0, col], axes[1, col]
        for (m, dbs), (T, P) in sorted(horizon_data.items()):
            if m != model:
                continue
            bands = horizon_quantile_bands(T, P)
            if bands is None:
                continue
            ls = dbs_style[dbs]
            x = bands["x_ms"]
            label = f"{dbs.upper()}"
            ax_r.plot(x, bands["med_r"], color=c, ls=ls, lw=1.4, label=label)
            ax_r.fill_between(
                x, bands["q10_r"], bands["q90_r"], color=c, alpha=0.12, lw=0
            )
            ax_n.plot(x, bands["med_rmse"], color=c, ls=ls, lw=1.4)
            ax_n.fill_between(
                x, bands["q10_rmse"], bands["q90_rmse"], color=c, alpha=0.12, lw=0
            )
        ax_r.axhline(0, color="#bbb", lw=0.8, ls=":")
        ax_r.set_ylabel(panel_r)
        ax_n.axhline(1, color="#bbb", lw=0.8, ls=":")
        ax_n.set_ylabel(panel_n)
        ax_n.set_xlabel("Horizon (ms)")
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=7)
    return fig


def collect_decomp_at_horizon(horizon_data, horizon_ms=500.0, sampling_hz=200.0):
    """Extract first horizon_ms of each (trial, channel) row; compute decomp metrics.
    Returns dict[(model, metric_name)] -> array."""
    h_steps = int(horizon_ms * sampling_hz / 1000)
    out = {}
    for (model, dbs), (T, P) in horizon_data.items():
        if T.shape[1] < h_steps:
            continue
        for i in range(T.shape[0]):
            t, p = T[i, :h_steps], P[i, :h_steps]
            if not (np.isfinite(t).all() and np.isfinite(p).all()) or len(t) < 20:
                continue
            for metric_name, _, _, fn in DECOMP_METRICS:
                try:
                    v = fn(t, p)
                    if np.isfinite(v):
                        out.setdefault((model, metric_name), []).append(v)
                except Exception:
                    pass
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Figure helpers — per-feature bars
# ---------------------------------------------------------------------------


def collect_per_cell_metric(
    session_objs, load_fn, score_fn, target="Z", metric="pearson", split="test"
):
    """dict[(session, model, dbs)] -> array of per-trial means across channels."""
    out = {}
    for tri in session_objs:
        for model_name, variant, run_ts in [
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ]:
            if not variant or not run_ts:
                continue
            res = load_fn(variant, run_ts, split)
            arr = res.get(target, [])
            if not arr:
                continue
            n_chans = np.asarray(arr[0]).shape[-1] if np.asarray(arr[0]).ndim else 1
            off_vals, on_vals = [], []
            for i in range(len(arr)):
                stim = normalize_stim(res["stim"][i])
                if stim is None:
                    continue
                ch_vals = [float(score_fn(res, i, ch, metric)) for ch in range(n_chans)]
                ch_vals = [v for v in ch_vals if np.isfinite(v)]
                if ch_vals:
                    (off_vals if stim == "off" else on_vals).append(
                        float(np.mean(ch_vals))
                    )
            out[(tri.label, model_name, "off")] = np.array(off_vals)
            out[(tri.label, model_name, "on")] = np.array(on_vals)
    return out


# ---------------------------------------------------------------------------
# PSID subspace helpers
# ---------------------------------------------------------------------------


def load_psid_cz(results_root, variant, run_ts):
    path = Path(results_root) / "psid" / variant / f"test_stats_{run_ts}.hdf5"
    with h5py.File(str(path), "r") as f:
        return f["Cz"][:], int(f.attrs["n1"])


def subspace_r_trial(xp, z_true, cz, n1):
    """Pearson r per subspace (full/X1/X2), averaged over channels."""
    zps = {
        "full": xp @ cz.T,
        "X1": xp[:, :n1] @ cz[:, :n1].T,
        "X2": xp[:, n1:] @ cz[:, n1:].T,
    }
    out = {}
    for name, zp in zps.items():
        rs = [
            pearsonr(z_true[:, ch], zp[:, ch])[0]
            for ch in range(z_true.shape[1])
            if np.isfinite(z_true[:, ch]).all()
            and np.isfinite(zp[:, ch]).all()
            and z_true.shape[0] > 1
        ]
        out[name] = float(np.mean(rs)) if rs else np.nan
    return out


def collect_x1x2(session_objs, results_root, exp_type, split="test"):
    """Per-session PSID subspace Pearson r. Returns dict[session] -> {full/X1/X2: array}."""
    out = {}
    for tri in session_objs:
        cz, n1 = load_psid_cz(results_root, tri.psid_variant, tri.psid_run_ts)
        res = load_split_results_required(
            results_root, tri.psid_variant, tri.psid_run_ts, split
        )
        vals = {"full": [], "X1": [], "X2": []}
        for i in range(len(res.get("Z", []))):
            if normalize_stim(res["stim"][i]) is None:
                continue
            z = np.array(res["Z"][i], dtype=float)
            xp = np.array(res["Xp"][i], dtype=float)
            if z.ndim == 1:
                z = z[:, None]
            if xp.ndim == 1:
                xp = xp[:, None]
            for name, r in subspace_r_trial(xp, z, cz, n1).items():
                if np.isfinite(r):
                    vals[name].append(r)
        out[tri.label] = {k: np.array(v) for k, v in vals.items()}
    return out


def x1x2_fig(data, sessions):
    COLORS = {"full": COLOR_PSID, "X1": "#8bbad4", "X2": "#bbbbbb"}
    LABELS = {"full": "Full", "X1": "X1 only", "X2": "X2 only"}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained", sharey=True)
    for ax, sess in zip(axes, sessions):
        for xi, label in enumerate(("full", "X1", "X2")):
            vals = data.get(sess, {}).get(label, np.array([]))
            if not len(vals):
                continue
            med = np.median(vals)
            q25, q75 = np.percentile(vals, [25, 75])
            ax.bar(
                xi,
                med,
                color=COLORS[label],
                alpha=0.85,
                label=LABELS[label] if sess == sessions[0] else None,
            )
            ax.errorbar(
                xi,
                med,
                yerr=[[med - q25], [q75 - med]],
                fmt="none",
                color="black",
                capsize=3,
                linewidth=1.0,
            )
        ax.axhline(0, color="#bbb", lw=0.8, ls=":")
        ax.set_xticks(range(3))
        ax.set_xticklabels(list(LABELS.values()), fontsize=8)
        panel_label(ax, "ABCD"[sessions.index(sess)], sess)
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(fontsize=8)
    return fig


def per_feature_bar_fig(
    means_data,
    feat_names_map,
    session_labels,
    *,
    ylabel="r (mean)",
    models=None,
    model_cols=None,
):
    """2x2 bar chart: per-feature mean r, one panel per session."""
    if models is None:
        models = MODELS
    if model_cols is None:
        model_cols = MODEL_COLS
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), layout="constrained")
    bar_w = 0.25
    for idx, (sess, ax) in enumerate(zip(session_labels, axes.flatten())):
        feat_names = feat_names_map.get(sess, [])
        if not feat_names:
            panel_label(ax, "ABCD"[idx], sess)
            ax.text(
                0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center"
            )
            continue
        x = np.arange(len(feat_names))
        for mi, model in enumerate(models):
            means = means_data.get((sess, model), np.full(len(feat_names), np.nan))
            ax.bar(
                x + (mi - 1) * bar_w,
                means,
                width=bar_w,
                color=model_cols[model],
                alpha=0.75,
                label=model if idx == 0 else None,
            )
        ax.axhline(0, color="#bbb", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel(ylabel)
        panel_label(ax, "ABCD"[idx], sess)
    axes.flatten()[0].legend()
    return fig


# ---------------------------------------------------------------------------
# Forecast-specific loaders and collectors
# ---------------------------------------------------------------------------


def load_forecast(results_root, variant, split, horizon="h5"):
    """Load forecast-split results from parquet. Returns {} if not found."""
    framework = variant.split("_", 1)[0]
    fdir = Path(results_root) / framework / variant / "forecast" / horizon
    files = sorted((fdir / split).glob("test_results_*.parquet"))
    if not files:
        return {}
    ts = files[0].name[len("test_results_") : -len(".parquet")]
    return load_precomputed_results(fdir, ts, split)


def pool_yz_dbs_cells(
    per_cell_y, per_cell_z, cells, *, models=None, y_label="Y", z_label="Z"
):
    """Pool Y and Z metric dicts across cells and DBS into dict[(model, label)]."""
    if models is None:
        models = MODELS
    out = {}
    for m in models:
        for label, source in ((y_label, per_cell_y), (z_label, per_cell_z)):
            buf = []
            for c in cells:
                for d in ("off", "on"):
                    arr = source.get((c, m, d))
                    if arr is not None:
                        buf.extend([float(v) for v in arr if np.isfinite(v)])
            out[(m, label)] = np.array(buf, dtype=float)
    return out


def yz_raincloud_fig(pool_r, pool_n, *, models=None, model_cols=None, figsize=None):
    """Per-model subplots: 2 rows (r, RMSE) x N cols (models). Within each subplot,
    Y/Z sides distinguished by style (light=Y, hatched=Z)."""
    if models is None:
        models = MODELS
    if model_cols is None:
        model_cols = MODEL_COLS
    targets = list(dict.fromkeys(t for _, t in pool_r))
    n_targets = len(targets)
    span = 0.80
    offsets = [(i - (n_targets - 1) / 2) * span / n_targets for i in range(n_targets)]
    rng = np.random.default_rng(42)

    def _style(label):
        is_z = label.upper().startswith("Z")
        return {"alpha": 0.80 if is_z else 0.32, "hatch": "////" if is_z else None}

    kde_w = min(0.14, span / n_targets * 0.50)
    box_w = min(0.06, span / n_targets * 0.22)
    n_models = len(models)
    if figsize is None:
        figsize = (2.8 * n_models, 7)

    fig, axes = plt.subplots(2, n_models, figsize=figsize, layout="constrained")
    axes = np.atleast_2d(axes)
    for col, model in enumerate(models):
        color = model_cols[model]
        for row, (pool, ylabel, logy, plabel) in enumerate(
            [
                (pool_r, "Pearson r", False, "A"),
                (pool_n, "NRMSE (log)", True, "B"),
            ]
        ):
            ax = axes[row, col]
            if logy:
                ax.set_yscale("log")
            ax.axhline(
                1.0 if logy else 0.0,
                color="black",
                ls=":",
                lw=0.6,
                alpha=0.30,
                zorder=0,
            )
            for ti, t in enumerate(targets):
                vals = pool.get((model, t))
                if vals is None or len(vals) < 2:
                    continue
                if logy:
                    vals = vals[vals > 0]
                if len(vals) < 2:
                    continue
                st = _style(t)
                off = offsets[ti]
                side = "left" if off <= 0 else "right"
                pos = off  # centre at 0
                anchor = pos + (-kde_w if side == "left" else kde_w)
                box_x = pos + (-box_w * 1.5 if side == "left" else box_w * 1.5)
                strip_x = pos + (-box_w * 0.4 if side == "left" else box_w * 0.4)
                kde_half_violin_vert(
                    ax,
                    vals,
                    x_anchor=anchor,
                    color=color,
                    alpha=st["alpha"],
                    hatch=st["hatch"],
                    logy=logy,
                    bw=0.55,
                    width=kde_w,
                    side=side,
                )
                bp = dict(
                    facecolor=color, alpha=st["alpha"], edgecolor=color, linewidth=1.0
                )
                if st["hatch"]:
                    bp["hatch"] = st["hatch"]
                ax.boxplot(
                    [vals],
                    positions=[box_x],
                    widths=box_w,
                    vert=True,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=bp,
                    medianprops=dict(color=color, linewidth=1.4),
                    whiskerprops=dict(color=color, linewidth=1.0),
                    capprops=dict(color=color, linewidth=1.0),
                )
                sv = (
                    vals[rng.choice(len(vals), 60, replace=False)]
                    if len(vals) > 60
                    else vals
                )
                ax.scatter(
                    strip_x + rng.uniform(-0.02, 0.02, len(sv)),
                    sv,
                    c=color,
                    alpha=min(st["alpha"], 0.28),
                    s=5,
                    linewidths=0,
                )
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(ylabel)
            panel_label(ax, f"{plabel}{'ABCDEFGH'[col]}", model)
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor="#888888",
            alpha=_style(t)["alpha"],
            edgecolor="#555555" if _style(t)["hatch"] else "none",
            linewidth=0.6,
            hatch=_style(t)["hatch"] or "",
            label=t,
        )
        for t in targets
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(targets),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.subplots_adjust(bottom=0.12)
    return fig


def collect_horizon_all_channels(session_objs, load_fn, target="Z", split="test"):
    """Pool forecast horizon across ALL channels; z-score per trial-channel.
    Returns dict[(model, dbs)] -> (T_mat, P_mat) each (n_rows, n_steps)."""
    k_true = "Z_future_true" if target == "Z" else "Y_future_true"
    k_pred = "Z_future_pred" if target == "Z" else "Y_future_pred"
    out = {(m, d): {"true": [], "pred": []} for m in MODELS for d in ("off", "on")}
    for tri in session_objs:
        for model_name, variant, run_ts in [
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ]:
            if not variant or not run_ts:
                continue
            res = load_fn(variant, run_ts, split)
            arr_t = res.get(k_true, [])
            arr_p = res.get(k_pred, [])
            if not arr_t or not arr_p:
                continue
            for i in range(len(arr_t)):
                if arr_t[i] is None or arr_p[i] is None:
                    continue
                stim = normalize_stim(res["stim"][i])
                if stim is None:
                    continue
                try:
                    T = reshape_future_z_time_first(np.asarray(arr_t[i], dtype=float))
                    P = reshape_future_z_time_first(np.asarray(arr_p[i], dtype=float))
                except (ValueError, Exception):
                    continue
                if T.shape != P.shape:
                    continue
                for ch in range(T.shape[1]):
                    t_ch, p_ch = T[:, ch], P[:, ch]
                    if not (np.isfinite(t_ch).all() and np.isfinite(p_ch).all()):
                        continue
                    mu = float(np.mean(t_ch))
                    sigma = float(np.std(t_ch)) or 1.0
                    out[(model_name, stim)]["true"].append((t_ch - mu) / sigma)
                    out[(model_name, stim)]["pred"].append((p_ch - mu) / sigma)
    result = {}
    for k, d in out.items():
        if not d["true"]:
            continue
        m_min = min(t.size for t in d["true"])
        if m_min == 0:
            continue
        T_mat = np.stack([t[:m_min] for t in d["true"]], axis=0)
        P_mat = np.stack([p[:m_min] for p in d["pred"]], axis=0)
        result[k] = (T_mat, P_mat)
    return result


def hilbert_horizon_collect(session_objs, load_fn, target="Z", split="test"):
    """Stack per-trial Hilbert analytic signal into (n_rows, n_steps) matrices.
    Returns dict[(model, dbs)] -> (AT, AP, PT, PP) each (n_rows, n_steps)."""
    k_true = "Z_future_true" if target == "Z" else "Y_future_true"
    k_pred = "Z_future_pred" if target == "Z" else "Y_future_pred"
    out = {
        (m, d): {"amp_t": [], "amp_p": [], "phase_t": [], "phase_p": []}
        for m in MODELS
        for d in ("off", "on")
    }
    for tri in session_objs:
        for model_name, variant, run_ts in [
            ("PSID", tri.psid_variant, tri.psid_run_ts),
            ("DPAD", tri.dpad_variant, tri.dpad_run_ts),
            ("VARMA", tri.varma_variant, tri.varma_run_ts),
        ]:
            if not variant or not run_ts:
                continue
            res = load_fn(variant, run_ts, split)
            arr_t = res.get(k_true, [])
            arr_p = res.get(k_pred, [])
            if not arr_t or not arr_p:
                continue
            for i in range(len(arr_t)):
                if arr_t[i] is None or arr_p[i] is None:
                    continue
                stim = normalize_stim(res["stim"][i])
                if stim is None:
                    continue
                try:
                    T = reshape_future_z_time_first(np.asarray(arr_t[i], dtype=float))
                    P = reshape_future_z_time_first(np.asarray(arr_p[i], dtype=float))
                except (ValueError, Exception):
                    continue
                if T.shape != P.shape:
                    continue
                for ch in range(T.shape[1]):
                    t_ch, p_ch = T[:, ch], P[:, ch]
                    if not (np.isfinite(t_ch).all() and np.isfinite(p_ch).all()):
                        continue
                    if len(t_ch) < 10:
                        continue
                    ht = hilbert(t_ch)
                    hp = hilbert(p_ch)
                    out[(model_name, stim)]["amp_t"].append(np.abs(ht))
                    out[(model_name, stim)]["amp_p"].append(np.abs(hp))
                    out[(model_name, stim)]["phase_t"].append(np.angle(ht))
                    out[(model_name, stim)]["phase_p"].append(np.angle(hp))
    result = {}
    for k, d in out.items():
        if not d["amp_t"]:
            continue
        m_min = min(a.size for a in d["amp_t"])
        if m_min < 2:
            continue
        AT = np.stack([a[:m_min] for a in d["amp_t"]])
        AP = np.stack([a[:m_min] for a in d["amp_p"]])
        PT = np.stack([a[:m_min] for a in d["phase_t"]])
        PP = np.stack([a[:m_min] for a in d["phase_p"]])
        result[k] = (AT, AP, PT, PP)
    return result


def hilbert_per_step(AT, AP, PT, PP):
    """Per-step amplitude r, PLV, mean instantaneous-frequency error."""
    n_steps = AT.shape[1]
    amp_r_arr = np.full(n_steps, np.nan)
    for h in range(n_steps):
        at, ap = AT[:, h], AP[:, h]
        if np.std(at) > 1e-10 and np.std(ap) > 1e-10:
            amp_r_arr[h] = float(np.corrcoef(at, ap)[0, 1])
    plv = np.abs(np.mean(np.exp(1j * (PT - PP)), axis=0))
    FT = np.diff(np.unwrap(PT, axis=1), axis=1) * (SAMPLING_HZ / (2 * np.pi))
    FP = np.diff(np.unwrap(PP, axis=1), axis=1) * (SAMPLING_HZ / (2 * np.pi))
    freq_err = np.append(np.mean(np.abs(FT - FP), axis=0), np.nan)
    return amp_r_arr, plv, freq_err


def window_decay(T, P, window_size=20):
    """Windowed Pearson r and RMSE vs forecast horizon."""
    n_steps = T.shape[1]
    n_windows = n_steps // window_size
    centers_ms = np.array(
        [(w + 0.5) * window_size / SAMPLING_HZ * 1000 for w in range(n_windows)]
    )
    rs, rmses = [], []
    for w in range(n_windows):
        sl = slice(w * window_size, (w + 1) * window_size)
        t_flat = T[:, sl].flatten()
        p_flat = P[:, sl].flatten()
        if np.std(t_flat) > 1e-12 and np.std(p_flat) > 1e-12:
            rs.append(float(np.corrcoef(t_flat, p_flat)[0, 1]))
        else:
            rs.append(float("nan"))
        rmses.append(float(np.sqrt(np.mean((t_flat - p_flat) ** 2))))
    return centers_ms, np.array(rs), np.array(rmses)
