"""Shared helpers for sec2c (reconstruction) and sec2d (forecast) notebooks.

Each notebook provides its own `load_fn(variant, run_ts, split) -> dict` and
`score_fn(res, trial_i, ch_i, metric) -> float` so the collectors work for
both reconstruction parquets and forecast parquets without code duplication.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import polars as pl
from dataclasses import dataclass
from pathlib import Path

from scipy.signal import hilbert, coherence as scipy_coherence
from scipy.stats import pearsonr, gaussian_kde

from modules.style import (
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    COLOR_RAW,
    COLOR_ENV,
    COLOR_PSID,
    COLOR_DPAD,
    COLOR_VARMA,
    PARTICIPANT_COLORS,
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
SESSION_COLORS = {
    "PDI1_S2": "#377EB8",  # blue
    "PDI1_S4": "#FF7F00",  # orange
    "PDI4_S2": "#4DAF4A",  # green
    "PDI4_S3": "#984EA3",  # purple
}
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


def spectral_coherence(t_sig, p_sig):
    nperseg = max(16, min(128, len(t_sig) // 4))
    _, Cxy = scipy_coherence(t_sig, p_sig, nperseg=nperseg)
    return float(np.mean(Cxy))


DECOMP_METRICS = [
    ("amplitude_r", "amplitude correlation coefficient", 0.0, amp_r),
    ("spectral_coherence", "mean spectral coherence", None, spectral_coherence),
    ("phase_plv", "phase-locking value", None, phase_plv),
]

# Distinct from DBS colors (blue / coral-red): Okabe-Ito pink, amber, teal
_DECOMP_MODEL_COLS = {
    "PSID": "#CC79A7",
    "DPAD": "#E69F00",
    "VARMA": "#009E73",
}

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


_BAND_SYM = {
    "delta": r"\delta",
    "theta": r"\theta",
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
}


def _feat_sort_key(name):
    """Sort order: ECoG before LFP, raw before env, low->high freq, electrode/pair asc."""
    _BO = {"delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4}
    is_env = name.endswith("_env")
    if name.startswith("ECOG_"):
        p = name.split("_")
        return (0, int(is_env), _BO.get(p[2], 9), int(p[3]), int(p[1]))
    if name.startswith("LAPLACIAN_"):
        p = name.split("_")
        return (1, int(is_env), _BO.get(p[3], 9), int(p[4]), int(p[1].split("-")[0]))
    return (2, int(is_env), 9, 0, 0)


def fmt_feature_short(name: str) -> str:
    """LaTeX mathtext x-tick label for per-feature heatmaps.

    ECOG_3_gamma_88_93_raw              -> $\\mathrm{ECoG}_{3}$ $\\gamma_{88}$ raw
    LAPLACIAN_13-15_LFP_gamma_88_93_raw -> $\\mathrm{LFP}_{13-15}$ $\\gamma_{88}$ raw
    """
    if name.startswith("ECOG_"):
        p = name.split("_")
        elec, band, f_lo, typ = p[1], p[2], p[3], p[5]
        sym = _BAND_SYM.get(band, band)
        return f"$\\mathrm{{ECoG}}_{{{elec}}}$ ${sym}_{{{f_lo}}}$ {typ}"
    if name.startswith("LAPLACIAN_"):
        p = name.split("_")
        pair, band, f_lo, typ = p[1], p[3], p[4], p[6]
        sym = _BAND_SYM.get(band, band)
        return f"$\\mathrm{{LFP}}_{{{pair}}}$ ${sym}_{{{f_lo}}}$ {typ}"
    return name


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




def collect_decomp_metrics(
    session_objs, exp_type, target, split, load_fn, ch_names_fn
):
    """Pooled amplitude_r, spectral_coherence, phase_plv split by raw/env/kin.
    dict[(model, metric_name, ch_type)] -> array, ch_type in {'raw', 'env', 'kin'}."""
    out = {}
    for sess, model, stim, ch, ch_name, t, p in _iter_trial_channels(
        session_objs, exp_type, target, split, load_fn, ch_names_fn, all_ch=True
    ):
        if len(t) < 50:
            continue
        if ch_name.endswith("_raw"):
            ch_type = "raw"
        elif ch_name.endswith("_env"):
            ch_type = "env"
        else:
            ch_type = "kin"
        for metric_name, _, _, fn in DECOMP_METRICS:
            try:
                v = fn(t, p)
                if np.isfinite(v):
                    out.setdefault((model, metric_name, ch_type), []).append(v)
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


def collect_band_by_session(session_objs, exp_type, target, split, load_fn, ch_names_fn):
    """Like collect_band_grouped_metrics but keys (model, band, session) -> np.array.

    Returns (corr_raw, corr_env) for per-session dot coloring.
    """
    raw_dict, env_dict = {}, {}
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
            d = raw_dict if ctype == "raw" else env_dict
            d.setdefault((model, band, sess), []).append(r)
    return (
        {k: np.array(v) for k, v in raw_dict.items()},
        {k: np.array(v) for k, v in env_dict.items()},
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
            buf = []
            for cell in cells:
                arr = per_cell_data.get((cell, m, d))
                if arr is None:
                    continue
                for v in arr:
                    fv = float(v)
                    if np.isfinite(fv):
                        buf.append(fv)
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
    scatter=True,
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
    if scatter:
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




def yz_model_raincloud_fig(
    pool_y,
    pool_z,
    *,
    metric_name="Pearson's r",
    ylabel=None,
    step_label="one-step-ahead",
    y_target_name="ECoG",
    z_target_name="Laplacian LFP",
    models=None,
    model_cols=None,
    figsize=None,
):
    """2-row (Y top, Z bottom) x N-col (models) raincloud. DBS-off left, DBS-on right.

    pool_y / pool_z: dict[(model, dbs_state), np.array]
    """
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
        for row, (pool, target_name) in enumerate(
            [(pool_y, y_target_name), (pool_z, z_target_name)]
        ):
            ax = axes[row, col]
            letter = "ABCDEF"[row * n + col]
            panel_label(ax, letter, f"{target_name} {step_label}\nof {model} framework")

            for d, color, side in [
                ("off", COLOR_DBS_OFF, "left"),
                ("on", COLOR_DBS_ON, "right"),
            ]:
                vals = pool.get((model, d), np.array([]))
                vals = vals[np.isfinite(vals)]
                if len(vals) < 2:
                    continue
                raincloud_slot(
                    ax, 0, vals, color=color, alpha=0.45, side=side, logy=False, rng=rng
                )
            ax.set_xticks([])
            if col == 0:
                ax.set_ylabel(metric_name if ylabel is None else ylabel)

    axes[-1, -1].legend(
        [
            Patch(facecolor=COLOR_DBS_OFF, alpha=0.7, edgecolor="black", linewidth=0.6),
            Patch(facecolor=COLOR_DBS_ON, alpha=0.7, edgecolor="black", linewidth=0.6),
        ],
        ["DBS-OFF", "DBS-ON"],
        loc="lower right",
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    return fig


def decomp_fig(
    data_y,
    data_z,
    *,
    models=None,
    y_target_name="ECoG",
    z_target_name="Laplacian LFP",
):
    """3x3 grid: rows=metrics (amplitude/coherence/PLV), cols=models.

    data_y / data_z: dict[(model, metric, ch_type), array], ch_type in {'raw', 'env', 'kin'}.
    Y target sits left, Z target right. Each shows the raw signal vs its amplitude
    envelope as a half-violin + slim box (no per-trial strip, kept uncrowded).
    Column = model (header on top row), row = signal component (ylabel on left).
    """
    if models is None:
        models = MODELS
    rng = np.random.default_rng(42)
    alpha = 0.75

    # Y group at x=-0.5, Z group at x=+0.5. Within each group raw faces left,
    # env faces right, so the two targets never overlap and stay comparable.
    has_kin = any(k[2] == "kin" for k in data_z)
    z_env_type = "kin" if has_kin else "env"
    groups = [
        (data_y, "raw", -0.5, "left",  COLOR_RAW),
        (data_y, "env", -0.5, "right", COLOR_ENV),
        (data_z, "raw",  0.5, "left",  COLOR_RAW),
        (data_z, z_env_type, 0.5, "right", COLOR_ENV),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(10, 8.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.20, w_pad=0.12)
    for col, model in enumerate(models):
        for row, (metric, ylabel, ref, _) in enumerate(DECOMP_METRICS):
            ax = axes[row, col]
            letter = "ABCDEFGHI"[row * len(models) + col]
            # Model name only heads the top row; other rows carry just the letter.
            panel_label(ax, letter, model if row == 0 else "")
            for data, ch_type, pos, side, color in groups:
                vals = data.get((model, metric, ch_type), np.array([]))
                vals = vals[np.isfinite(vals)]
                if len(vals) > 1:
                    raincloud_slot(
                        ax, pos, vals, color=color, alpha=alpha,
                        side=side, rng=rng, scatter=False, kde_width=0.24,
                    )
            ax.set_xticks([-0.5, 0.5])
            ax.set_xlim(-1.05, 1.05)
            if row == len(DECOMP_METRICS) - 1:
                ax.set_xticklabels([y_target_name, z_target_name], fontsize=9)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(ylabel)

    axes[0, -1].legend(
        [
            Patch(facecolor=COLOR_RAW, alpha=alpha, edgecolor="black", linewidth=0.6),
            Patch(facecolor=COLOR_ENV, alpha=alpha, edgecolor="black", linewidth=0.6),
        ],
        ["raw signal", "envelope"],
        loc="upper right",
        frameon=False,
        fontsize=8,
        ncol=1,
    )
    return fig


def band_raincloud_fig(
    ax,
    data,
    ylabel,
    *,
    ref_line=None,
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
        ax.legend([Patch(fc=model_cols[m], alpha=0.72) for m in models], models)


def band_butterfly_raincloud_fig(
    ax,
    data_raw,
    data_env,
    ylabel,
    model,
    *,
    session_labels,
    session_colors,
    model_col,
    show_legend=False,
    rng_seed=42,
):
    """Per-band butterfly: raw faces left, env faces right.

    KDE + box drawn in model color (pooled). Scatter dots colored by session.
    data_raw / data_env: {(model, band, session): np.array}
    """
    rng = np.random.default_rng(rng_seed)
    alpha_bg = 0.40
    strip_cap = 18
    # Fixed per-session x-sub-offsets so each session's dots form a distinct column.
    n_sess = len(session_labels)
    sess_sub = {s: (i - (n_sess - 1) / 2) * 0.022 for i, s in enumerate(session_labels)}

    band_positions = [bi * 1.4 for bi in range(len(BAND_ORDER))]
    for bi, band in enumerate(BAND_ORDER):
        pos = band_positions[bi]
        raw_pooled = np.concatenate(
            [data_raw.get((model, band, s), np.array([])) for s in session_labels]
        )
        env_pooled = np.concatenate(
            [data_env.get((model, band, s), np.array([])) for s in session_labels]
        )
        raw_pooled = raw_pooled[np.isfinite(raw_pooled)]
        env_pooled = env_pooled[np.isfinite(env_pooled)]

        if len(raw_pooled) > 1:
            raincloud_slot(ax, pos, raw_pooled, color=model_col, alpha=alpha_bg,
                           side="left", rng=rng, scatter=False)
        if len(env_pooled) > 1:
            raincloud_slot(ax, pos, env_pooled, color=model_col, alpha=alpha_bg,
                           side="right", rng=rng, scatter=False)

        for sess in session_labels:
            scol = session_colors[sess]
            xsub = sess_sub[sess]
            for d, strip_base, marker in [
                (data_raw, pos - 0.08, "o"),
                (data_env, pos + 0.08, "^"),
            ]:
                sv = d.get((model, band, sess), np.array([]))
                sv = sv[np.isfinite(sv)]
                if len(sv) < 1:
                    continue
                sv_sub = sv[rng.choice(len(sv), min(strip_cap, len(sv)), replace=False)]
                ax.scatter(
                    np.full(len(sv_sub), strip_base + xsub),
                    sv_sub,
                    c=scol, alpha=0.75, s=14, linewidths=0, marker=marker,
                )

    if show_legend:
        session_handles = [Patch(color=session_colors[s], label=s) for s in session_labels]
        type_handles = [
            Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=5, label="raw"),
            Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=5, label="env"),
        ]
        handles = session_handles + type_handles
        ax.legend(
            handles=handles,
            loc="upper right",
            frameon=False,
            fontsize=8,
            ncol=int(np.ceil(len(handles) / 2)),
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(band_positions)
    ax.set_xticklabels(BAND_ORDER)
    ax.set_xlim(band_positions[0] - 0.65, band_positions[-1] + 0.65)






# ---------------------------------------------------------------------------
# Horizon / step-level metric functions
# ---------------------------------------------------------------------------




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
    panel_r="Pearson's r",
    panel_n="Normalized RMSE",
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
        ax_n.axhline(1, color="#bbb", lw=0.8, ls=":")
        if col == 0:
            ax_r.set_ylabel(panel_r)
            ax_n.set_ylabel(panel_n)
        ax_n.set_xlabel("Horizon (ms)")
    handles = [
        Line2D([0], [0], color="#555555", ls="-", lw=1.4, label="DBS-OFF"),
        Line2D([0], [0], color="#555555", ls="--", lw=1.4, label="DBS-ON"),
    ]
    axes[-1, -1].legend(
        handles=handles, loc="lower right", frameon=False, fontsize=9, ncol=2
    )
    return fig




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










def per_feature_bar_fig(
    means_data,
    feat_names_map,
    session_labels,
    *,
    ylabel="Pearson's r (mean)",
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


def per_feature_heatmap_fig(
    means_data,
    feat_names_map,
    session_labels,
    *,
    vmin=-0.3,
    vmax=1.0,
    models=None,
    model_cols=None,
):
    """3 vertically stacked heatmaps (PSID / DPAD / VARMA), each panel sessions x features.

    means_data    : {(session, model): array[n_feat]}  from collect_per_feature_means
    feat_names_map: {session: [feat_names]}
    session_labels: SESSIONS tuple (4 strings)

    X-axis = union of all features across sessions, sorted by _feat_sort_key.
    NaN where feature not in that session's mRMR set — rendered as gray.
    """
    if models is None:
        models = MODELS
    if model_cols is None:
        model_cols = MODEL_COLS

    # Build sorted union of all features
    all_feats = sorted(
        {f for feats in feat_names_map.values() for f in feats},
        key=_feat_sort_key,
    )
    feat_to_col = {f: i for i, f in enumerate(all_feats)}
    n_feats = len(all_feats)
    n_sess = len(session_labels)
    x_labels = [fmt_feature_short(f) for f in all_feats]

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#cccccc")

    fig, axes = plt.subplots(len(models), 1, figsize=(14, 9), layout="constrained")
    if len(models) == 1:
        axes = [axes]

    for i, (model, ax) in enumerate(zip(models, axes)):
        mat = np.full((n_sess, n_feats), np.nan)
        for si, sess in enumerate(session_labels):
            sess_feats = feat_names_map.get(sess, [])
            vals = means_data.get((sess, model), np.full(len(sess_feats), np.nan))
            for fi, fname in enumerate(sess_feats):
                col = feat_to_col.get(fname)
                if col is not None and fi < len(vals):
                    mat[si, col] = vals[fi]

        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(
            masked,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(n_feats))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n_sess))
        ax.set_yticklabels(session_labels, fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson's r")
        panel_label(ax, "ABC"[i], model)

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






