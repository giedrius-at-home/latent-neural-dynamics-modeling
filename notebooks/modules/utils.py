"""Trial metrics + pooled-RMSE aggregation for thesis sec2 notebooks.

Data collectors (pure-numpy) are always importable; the legacy Plotly figure
builders below are dead code kept only for grep-able reference. Sec2 notebooks
now draw their own matplotlib boxplots via helpers in ``thesis_sec2_common``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

# Legacy Plotly surface — guarded so the module imports on matplotlib-only
# setups. Callers of the builders below should have migrated to the
# matplotlib versions in thesis_sec2_common.
try:
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover — plotly optional after migration
    go = None
    Figure = Any  # type: ignore[misc, assignment]
    make_subplots = None

from modules.loaders import (
    ThesisDataError,
    _prepare_z_array,
    discover_session_run,
    get_channel,
    has_dpad_data,
    load_split_results,
    load_split_results_required,
    reshape_future_z_time_first,
    resolve_input_channels,
    resolve_output_channels,
    trial_rmse_z_for_model,
    zscore_using_true_stats,
)
from modules.specs import ThesisTheme
from modules.style import (
    COLOR_DPAD,
    COLOR_PSID,
    COLOR_VARMA,
    PARTICIPANT_COLORS,
)

# Legacy Plotly symbols — stubs so the builders below don't NameError at import.
# The builders are dead code post-matplotlib migration; we keep the body for
# reference but none of these constants are used by the data collectors.
FONT_FAMILY = "serif"
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 9


def apply_thesis_style(*a, **kw):
    return None


def grid_color(*a, **kw):
    return "#E0E0E0"


def panel_label(*a, **kw):
    return None


def paper_colors(*a, **kw):
    return ("#FFFFFF", "#FFFFFF")


def true_line_color(*a, **kw):
    return "#1A1A1A"


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trial metrics
# ---------------------------------------------------------------------------

MetricKind = Literal["rmse", "pearson", "vaf"]
METRIC_KINDS: tuple[MetricKind, ...] = ("rmse", "pearson", "vaf")


def metric_display_name(metric: MetricKind) -> str:
    return {"rmse": "RMSE", "pearson": "Pearson r", "vaf": "VAF"}[metric]


def metric_axis_label(metric: MetricKind, target_label: str = "") -> str:
    core = {"rmse": "RMSE [z]", "pearson": "Pearson r", "vaf": "VAF"}[metric]
    return f"{core} - {target_label}" if target_label else core


def metric_y_range(metric: MetricKind, values: np.ndarray) -> tuple[float, float]:
    """Metric-aware y-range that focuses on the IQR so boxes are readable.

    Pearson/VAF are bounded so the previous logic still applies. For RMSE we
    switched from a 95th-percentile ceiling (which left the boxes squashed
    against the bottom when outliers dragged the upper limit) to a Tukey-style
    IQR window: y in [max(0, Q1 - 1.5·IQR), Q3 + 1.5·IQR]. Outlier dots
    beyond the whiskers are clipped, which is fine because we already pass
    ``boxpoints=False`` and overlay strip dots ourselves.
    """
    finite = values[np.isfinite(values)] if values.size else np.array([], dtype=float)
    if metric == "pearson":
        lo = -0.2 if finite.size == 0 else min(-0.2, float(np.min(finite)) - 0.05)
        return (lo, 1.05)
    if metric == "vaf":
        lo = (
            -0.5
            if finite.size == 0
            else min(-0.5, float(np.percentile(finite, 1)) - 0.05)
        )
        return (max(lo, -2.0), 1.05)
    # rmse: IQR-focused window — boxes open up, outliers handled by strip dots
    if finite.size == 0:
        return (0.0, 1.0)
    q1 = float(np.percentile(finite, 25))
    q3 = float(np.percentile(finite, 75))
    iqr = q3 - q1
    if iqr <= 0:  # degenerate distribution; fall back to median-based window
        med = float(np.median(finite))
        return (0.0, max(med * 1.5, 0.1))
    y_lo = max(0.0, q1 - 1.5 * iqr)
    y_hi = q3 + 1.5 * iqr
    return (y_lo, y_hi * 1.02)


def _score_pearson(true_c: np.ndarray, pred_c: np.ndarray) -> float:
    if true_c.std() == 0 or pred_c.std() == 0:
        return float("nan")
    r = float(np.corrcoef(true_c, pred_c)[0, 1])
    return r if np.isfinite(r) else float("nan")


def _score_vaf(true_c: np.ndarray, pred_c: np.ndarray) -> float:
    """VAF on raw time series: 1 - var(resid) / var(true). Scale- and bias-sensitive."""
    if true_c.size == 0 or pred_c.size == 0:
        return float("nan")
    var_t = float(np.var(true_c))
    if var_t < 1e-12:
        return float("nan")
    return 1.0 - float(np.var(true_c - pred_c)) / var_t


def score_channel(true_c: np.ndarray, pred_c: np.ndarray, metric: MetricKind) -> float:
    """Score one trial on one output channel."""
    if true_c.size == 0 or pred_c.size == 0:
        return float("nan")
    if metric == "rmse":
        zt = zscore_using_true_stats(true_c, true_c)
        zp = zscore_using_true_stats(true_c, pred_c)
        return float(np.sqrt(np.mean((zt - zp) ** 2)))
    if metric == "pearson":
        return _score_pearson(true_c, pred_c)
    if metric == "vaf":
        return _score_vaf(true_c, pred_c)
    raise ValueError(f"Unknown metric: {metric!r}")


def _trial_metric_from_keys(
    split_res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    true_key: str,
    pred_key: str,
    metric: MetricKind,
) -> float:
    t = split_res.get(true_key)
    p = split_res.get(pred_key)
    if t is None or p is None or trial_idx >= len(t) or trial_idx >= len(p):
        raise ValueError(f"missing {true_key}/{pred_key} for trial_idx={trial_idx}")
    if t[trial_idx] is None or p[trial_idx] is None:
        raise ValueError(f"null {true_key}/{pred_key} for trial_idx={trial_idx}")
    true_arr, t_abs = _prepare_z_array(t[trial_idx], split_res, trial_idx)
    pred_arr, t2 = _prepare_z_array(p[trial_idx], split_res, trial_idx)
    if len(t_abs) != len(t2):
        raise ValueError("Time axis mismatch between true and pred for this trial.")
    true_c = np.asarray(get_channel(true_arr, channel_idx, t_abs), dtype=float)
    pred_c = np.asarray(get_channel(pred_arr, channel_idx, t_abs), dtype=float)
    return score_channel(true_c, pred_c, metric)


def trial_metric_z_for_model(
    split_res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    metric: MetricKind,
) -> float:
    """Behavioral channel (Z/Zp)."""
    return _trial_metric_from_keys(split_res, trial_idx, channel_idx, "Z", "Zp", metric)


def trial_metric_y_for_model(
    split_res: Dict[str, Any],
    trial_idx: int,
    y_channel_idx: int,
    metric: MetricKind,
) -> float:
    """Neural channel (Y/Yp)."""
    return _trial_metric_from_keys(
        split_res, trial_idx, y_channel_idx, "Y", "Yp", metric
    )


def trial_metric_forecast_for_model(
    split_res: Dict[str, Any],
    trial_idx: int,
    channel_idx: int,
    metric: MetricKind,
    *,
    forecast_target: str = "Y",
) -> float:
    """Multi-step forecast: reduces (m, ch) horizon window to one scalar."""
    k_true = "Y_future_true" if forecast_target == "Y" else "Z_future_true"
    k_pred = "Y_future_pred" if forecast_target == "Y" else "Z_future_pred"
    zt = split_res.get(k_true)
    zp = split_res.get(k_pred)
    if zt is None or zp is None or trial_idx >= len(zt) or trial_idx >= len(zp):
        return float("nan")
    try:
        T = reshape_future_z_time_first(np.asarray(zt[trial_idx], dtype=float))
        P = reshape_future_z_time_first(np.asarray(zp[trial_idx], dtype=float))
    except ValueError:
        return float("nan")
    if T.shape != P.shape:
        return float("nan")
    n_ch = T.shape[1]
    ci = 0 if n_ch == 1 else channel_idx
    if ci >= n_ch:
        return float("nan")
    t_vec = T[:, ci]
    p_vec = P[:, ci]
    msk = np.isfinite(t_vec) & np.isfinite(p_vec)
    if not np.any(msk):
        return float("nan")
    t_vec, p_vec = t_vec[msk], p_vec[msk]
    if metric == "rmse":
        mu = float(np.mean(t_vec))
        sigma = float(np.std(t_vec)) or 1.0
        err = np.abs((t_vec - mu) / sigma - (p_vec - mu) / sigma)
        return float(np.sqrt(np.mean(err**2)))
    if metric == "pearson":
        return _score_pearson(t_vec, p_vec)
    if metric == "vaf":
        return _score_vaf(t_vec, p_vec)
    raise ValueError(f"Unknown metric: {metric!r}")


# ---------------------------------------------------------------------------
# Aggregate collection helpers
# ---------------------------------------------------------------------------

TrialKey = Tuple[Any, Any, Any, Any]


def normalize_stim(val: Any) -> Optional[str]:
    """Return 'on' or 'off', or None if unknown."""
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        val = val[0]
    s = str(val).strip().lower()
    if s in ("on", "1", "true"):
        return "on"
    if s in ("off", "0", "false"):
        return "off"
    return None


def _trial_key(split_res: Dict[str, Any], trial_idx: int) -> TrialKey:
    def _one(x: Any) -> str:
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
            x = x[0]
        return str(x)

    pid = split_res["participant_id"][trial_idx]
    sess = split_res["session"][trial_idx]
    blk = split_res["block"][trial_idx]
    tri = split_res["trial"][trial_idx]
    return (_one(pid), _one(sess), _one(blk), _one(tri))


def _key_index_map(split_res: Dict[str, Any]) -> Dict[TrialKey, int]:
    zl = split_res.get("Z")
    if not zl:
        return {}
    out: Dict[TrialKey, int] = {}
    for i in range(len(zl)):
        if zl[i] is None:
            continue
        k = _trial_key(split_res, i)
        out[k] = i
    return out


def _sem(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        return float("nan")
    return float(np.std(a, ddof=1) / np.sqrt(n))


def _wilcoxon_paired(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Paired Wilcoxon signed-rank; returns p-value or None if test cannot run."""
    from scipy import stats

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = min(x.size, y.size)
    if m < 6:
        return None
    x, y = x[:m], y[:m]
    d = x - y
    if np.allclose(d, 0):
        return None
    try:
        try:
            res = stats.wilcoxon(x, y, alternative="two-sided", method="auto")
        except TypeError:
            res = stats.wilcoxon(x, y, alternative="two-sided")
        p = float(res.pvalue)
        return p if np.isfinite(p) else None
    except Exception as e:
        logger.debug("Wilcoxon skipped: %s", e)
        return None


def p_to_stars(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


@dataclass
class WilcoxonResults:
    psid_vs_varma_off_p: Optional[float] = None
    psid_vs_varma_on_p: Optional[float] = None
    dpad_vs_varma_off_p: Optional[float] = None
    dpad_vs_varma_on_p: Optional[float] = None
    psid_vs_dpad_off_p: Optional[float] = None
    psid_vs_dpad_on_p: Optional[float] = None


@dataclass
class AggregateRmseData:
    """Six cells in order: PSID off, PSID on, DPAD off, DPAD on, VARMA off, VARMA on."""

    labels: Tuple[str, ...] = (
        "PSID DBS-OFF",
        "PSID DBS-ON",
        "DPAD DBS-OFF",
        "DPAD DBS-ON",
        "VARMA DBS-OFF",
        "VARMA DBS-ON",
    )
    means: Tuple[float, ...] = (0.0,) * 6
    sems: Tuple[float, ...] = (0.0,) * 6
    trial_rmse: Tuple[List[float], ...] = field(
        default_factory=lambda: tuple([] for _ in range(6))
    )
    trial_rmse_with_participant: Tuple[List[Tuple[float, str]], ...] = field(
        default_factory=lambda: tuple([] for _ in range(6))
    )
    wilcoxon: WilcoxonResults = field(default_factory=WilcoxonResults)
    n_triplets_used: int = 0
    show_dpad: bool = True


def _discover_all(
    results_root: Path,
    sessions: List[str],
    exp_type: str,
) -> Dict[str, Dict[str, Tuple[str, str]]]:
    """Return {session: {fw: (variant, run_ts)}} for psid/varma/dpad, skipping sessions
    where psid or varma is missing."""
    out: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for session in sessions:
        psid = discover_session_run(results_root, "psid", exp_type, session)
        varma = discover_session_run(results_root, "varma", exp_type, session)
        if not psid[0] or not varma[0]:
            continue
        dpad = discover_session_run(results_root, "dpad", exp_type, session)
        out[session] = {"psid": psid, "varma": varma, "dpad": dpad}
    return out


def collect_pooled_rmse(
    results_root: Path,
    sessions: List[str],
    exp_type: str,
    channel_idx: int,
    split: str = "test",
    run_wilcoxon: bool = True,
    include_dpad: Optional[bool] = None,
    metric: MetricKind = "rmse",
) -> AggregateRmseData:
    """Pool per-trial scores across sessions for PSID/DPAD/VARMA.

    ``include_dpad``: None (default) auto-detects per session via has_dpad_data.
    ``metric``: 'rmse' (default), 'pearson', or 'vaf'.
    """
    score_fn = (
        trial_rmse_z_for_model
        if metric == "rmse"
        else (lambda res, i, ch: trial_metric_z_for_model(res, i, ch, metric))
    )

    session_runs = _discover_all(results_root, sessions, exp_type)
    if not session_runs:
        raise ThesisDataError(
            "collect_pooled_rmse: no sessions have valid PSID+VARMA results."
        )

    if include_dpad is None:
        show_dpad = any(
            has_dpad_data(results_root, runs["dpad"][0], runs["dpad"][1], split)
            for runs in session_runs.values()
            if runs["dpad"][0]
        )
    else:
        show_dpad = bool(include_dpad)

    score_buckets: List[List[float]] = [[] for _ in range(6)]
    buckets_with_pid: List[List[Tuple[float, str]]] = [[] for _ in range(6)]
    paired_off_psid_varma: List[Tuple[float, float]] = []
    paired_on_psid_varma: List[Tuple[float, float]] = []
    paired_off_dpad_varma: List[Tuple[float, float]] = []
    paired_on_dpad_varma: List[Tuple[float, float]] = []
    paired_off_psid_dpad: List[Tuple[float, float]] = []
    paired_on_psid_dpad: List[Tuple[float, float]] = []

    n_ok = 0
    for session, runs in session_runs.items():
        psid_var, psid_ts = runs["psid"]
        varma_var, varma_ts = runs["varma"]
        dpad_var, dpad_ts = runs["dpad"]

        res_p = load_split_results_required(results_root, psid_var, psid_ts, split)
        res_v = load_split_results_required(results_root, varma_var, varma_ts, split)
        has_dpad = show_dpad and bool(dpad_ts)
        res_d = (
            load_split_results(results_root, dpad_var, dpad_ts, split)
            if has_dpad
            else None
        )

        mp = _key_index_map(res_p)
        mv = _key_index_map(res_v)
        md = _key_index_map(res_d) if res_d else {}
        common_pd = set(mp.keys()) & set(mv.keys())
        if md:
            common_pd &= set(md.keys())
        if not common_pd:
            raise ThesisDataError(
                f"Pooled RMSE session {session!r}: no trial keys common to PSID and VARMA."
            )

        n_ok += 1
        for k in sorted(
            common_pd, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))
        ):
            i_p, i_v = mp[k], mv[k]
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue
            try:
                r_p = score_fn(res_p, i_p, channel_idx)
                r_d = (
                    score_fn(res_d, md[k], channel_idx)
                    if res_d and k in md
                    else float("nan")
                )
                r_v = score_fn(res_v, i_v, channel_idx)
            except Exception as e:
                logger.debug("Skip trial %s: %s", k, e)
                continue

            pid = str(k[0]) if k else "?"
            d_ok = show_dpad and np.isfinite(r_d)
            if stim == "off":
                score_buckets[0].append(r_p)
                buckets_with_pid[0].append((r_p, pid))
                score_buckets[4].append(r_v)
                buckets_with_pid[4].append((r_v, pid))
                paired_off_psid_varma.append((r_p, r_v))
                if d_ok:
                    score_buckets[2].append(r_d)
                    buckets_with_pid[2].append((r_d, pid))
                    paired_off_dpad_varma.append((r_d, r_v))
                    paired_off_psid_dpad.append((r_p, r_d))
            else:
                score_buckets[1].append(r_p)
                buckets_with_pid[1].append((r_p, pid))
                score_buckets[5].append(r_v)
                buckets_with_pid[5].append((r_v, pid))
                paired_on_psid_varma.append((r_p, r_v))
                if d_ok:
                    score_buckets[3].append(r_d)
                    buckets_with_pid[3].append((r_d, pid))
                    paired_on_dpad_varma.append((r_d, r_v))
                    paired_on_psid_dpad.append((r_p, r_d))

    if n_ok == 0:
        raise ThesisDataError(
            "collect_pooled_rmse: no sessions produced aligned PSID/DPAD/VARMA trial keys."
        )
    if sum(len(b) for b in score_buckets) == 0:
        raise ThesisDataError(
            "collect_pooled_rmse: no trial score values after alignment (check stim and Z/Zp)."
        )

    means = tuple(float(np.mean(b)) if len(b) else float("nan") for b in score_buckets)
    sems = tuple(
        _sem(np.array(b)) if len(b) > 1 else float("nan") for b in score_buckets
    )

    w = WilcoxonResults()
    if run_wilcoxon:
        for pairs, attr in [
            (paired_off_psid_varma, "psid_vs_varma_off_p"),
            (paired_on_psid_varma, "psid_vs_varma_on_p"),
            (paired_off_dpad_varma, "dpad_vs_varma_off_p"),
            (paired_on_dpad_varma, "dpad_vs_varma_on_p"),
            (paired_off_psid_dpad, "psid_vs_dpad_off_p"),
            (paired_on_psid_dpad, "psid_vs_dpad_on_p"),
        ]:
            if pairs:
                x_arr, y_arr = zip(*pairs)
                setattr(w, attr, _wilcoxon_paired(np.array(x_arr), np.array(y_arr)))

    return AggregateRmseData(
        means=means,
        sems=sems,
        trial_rmse=tuple(score_buckets),
        trial_rmse_with_participant=tuple(buckets_with_pid),
        wilcoxon=w,
        n_triplets_used=n_ok,
        show_dpad=show_dpad,
    )


# ---------------------------------------------------------------------------
# Boxplot figure builder
# ---------------------------------------------------------------------------


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_rmse_boxplot_figure(
    data: AggregateRmseData,
    theme: ThesisTheme,
    rng: np.random.Generator,
    jitter: float = 0.12,
    metric: MetricKind = "rmse",
    target_label: str = "",
) -> Figure:
    """Two-panel box + strip: DBS-OFF | DBS-ON. One box per model, dots coloured by participant.

    ``metric``       — selects y-axis label + range (rmse/pearson/vaf).
    ``target_label`` — NOT shown on y-axis per convention; pass for forward-compat.
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    show_dpad = getattr(data, "show_dpad", True)
    if show_dpad:
        models = ["PSID", "DPAD", "VARMA"]
        model_colors = [COLOR_PSID, COLOR_DPAD, COLOR_VARMA]
        off_cells = [0, 2, 4]
        on_cells = [1, 3, 5]
    else:
        models = ["PSID", "VARMA"]
        model_colors = [COLOR_PSID, COLOR_VARMA]
        off_cells = [0, 4]
        on_cells = [1, 5]
    n_models = len(models)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["DBS-OFF", "DBS-ON"],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    has_pid = (
        hasattr(data, "trial_rmse_with_participant")
        and len(data.trial_rmse_with_participant) == 6
        and any(len(data.trial_rmse_with_participant[i]) > 0 for i in range(6))
    )
    cells_with_pid = (
        data.trial_rmse_with_participant
        if has_pid
        else tuple(
            [(float(v), "?") for v in data.trial_rmse[i] if np.isfinite(v)]
            for i in range(6)
        )
    )

    def _get_vals(cell_idx: int) -> tuple[list[float], list[str]]:
        rows = cells_with_pid[cell_idx]
        vals = [r[0] for r in rows if np.isfinite(r[0])]
        pids = [str(r[1]) for r in rows if np.isfinite(r[0])]
        return vals, pids

    all_vals: list[float] = []
    for _, cell_ids in [("DBS-OFF", off_cells), ("DBS-ON", on_cells)]:
        for ci in cell_ids:
            v, _ = _get_vals(ci)
            all_vals.extend(v)
    y_min, y_max = metric_y_range(metric, np.asarray(all_vals, dtype=float))

    for col_idx, (_, cell_ids) in enumerate(
        [("DBS-OFF", off_cells), ("DBS-ON", on_cells)]
    ):
        for mi, (model, col) in enumerate(zip(models, model_colors)):
            cell_idx = cell_ids[mi]
            vals, pids = _get_vals(cell_idx)
            if not vals:
                continue

            fig.add_trace(
                go.Box(
                    y=vals,
                    x=[mi] * len(vals),
                    name=model,
                    marker_color=col,
                    line=dict(color=col, width=1.8),
                    fillcolor=_hex_to_rgba(col, 0.45),
                    boxpoints=False,
                    quartilemethod="exclusive",
                    showlegend=(col_idx == 0),
                    legendgroup=model,
                ),
                row=1,
                col=col_idx + 1,
            )

            jt = rng.uniform(-jitter, jitter, size=len(vals))
            colors = [PARTICIPANT_COLORS.get(p, "#888888") for p in pids]
            fig.add_trace(
                go.Scatter(
                    x=[mi + jt[i] for i in range(len(vals))],
                    y=vals,
                    mode="markers",
                    marker=dict(
                        size=5, color=colors, line=dict(width=0), symbol="circle"
                    ),
                    showlegend=False,
                    legendgroup="trials",
                    hovertext=[f"{p}<br>{v:.3f}" for p, v in zip(pids, vals)],
                ),
                row=1,
                col=col_idx + 1,
            )

    # Participant colour legend entries
    legend_pids: set[str] = set()
    for i in range(6):
        for r in cells_with_pid[i]:
            if len(r) >= 2 and np.isfinite(r[0]):
                legend_pids.add(str(r[1]))
    for pid in sorted(legend_pids):
        col = PARTICIPANT_COLORS.get(pid, "#888888")
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=11, color=col, line=dict(width=0)),
                name=pid,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    apply_thesis_style(
        fig,
        theme,
        height=420,
        margin=dict(l=72, r=28, t=36, b=90),
        hovermode="closest",
        legend_y=-0.18,
    )
    fig.update_layout(title=None)

    x_kw = dict(
        ticktext=models,
        tickvals=list(range(n_models)),
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=fg,
        tickfont=dict(size=FONT_SIZE_TICK),
    )
    fig.update_xaxes(**x_kw, row=1, col=1)
    fig.update_xaxes(**x_kw, row=1, col=2)

    y_kw = dict(
        range=[y_min, y_max],
        showgrid=True,
        gridcolor=grid,
        zeroline=True,
        zerolinecolor=grid,
        showline=True,
        linecolor=fg,
        tickfont=dict(size=FONT_SIZE_TICK),
    )
    fig.update_yaxes(
        title_text=metric_axis_label(metric),
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        **y_kw,
        row=1,
        col=1,
    )
    fig.update_yaxes(**y_kw, row=1, col=2)

    return fig


# ---------------------------------------------------------------------------
# Neural-band metric heatmap aggregation (sec2c Fig 22)
# ---------------------------------------------------------------------------
# Replaces thesis_lib.neural_band_pearson.collect_neural_band_pearson with
# (a) metric parameter (rmse / pearson / vaf) and (b) per-model channel-name
# resolution. The dashboard version reused one model's band index list on every
# model's r-list, producing uninterpretable cells when PSID has 60 channels and
# DPAD/VARMA only 6.

_BAND_DISPLAY = {
    "delta": "Delta",
    "theta": "Theta",
    "alpha": "Alpha",
    "beta": "Beta",
    "gamma": "Gamma",
}
_ALLOWED_BANDS = frozenset(_BAND_DISPLAY.keys())


def parse_parent_band(channel_name: str) -> Optional[str]:
    """Map an ECoG / LFP / LAPLACIAN_xx-yy_LFP channel name to its display band label.
    Returns None when no band token is found."""
    parts = str(channel_name).split("_")
    if len(parts) >= 3 and parts[0].upper() in ("ECOG", "LFP") and parts[1].isdigit():
        key = parts[2].lower()
        if key in _ALLOWED_BANDS:
            return _BAND_DISPLAY[key]
    for p in parts:
        pl = p.lower()
        if pl in _ALLOWED_BANDS:
            return _BAND_DISPLAY[pl]
    return None


@dataclass
class NeuralBandMetricData:
    """Per-band mean metric × model × DBS condition.

    Rows = parent band labels present in the data (subset of Delta/Theta/Alpha/Beta/Gamma).
    Columns = model display names (default: PSID, DPAD, VARMA).
    Each cell: per trial, take the metric for each of that model's own channels
    in the band; mean within the trial; then mean across trials and sessions.
    """

    z_off: np.ndarray
    z_on: np.ndarray
    band_labels: List[str]
    column_labels: Tuple[str, ...]
    metric: str
    target: Literal["Y", "Z"] = "Y"
    n_triplets_used: int = 0
    n_trials_off: int = 0
    n_trials_on: int = 0


def _band_indices(channels: list[str], order: list[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {b: [] for b in order}
    for i, name in enumerate(channels):
        band = parse_parent_band(name)
        if band in out:
            out[band].append(i)
    return out


def collect_neural_band_metric(
    results_root: Path,
    sessions: List[str],
    exp_type: str,
    metric: MetricKind = "pearson",
    split: str = "test",
    band_row_order: Optional[Tuple[str, ...]] = None,
    *,
    target: Literal["Y", "Z"] = "Y",
    models: Tuple[str, ...] = ("psid", "dpad", "varma"),
) -> NeuralBandMetricData:
    """Per-band mean metric x model x DBS condition, across trials x sessions.

    ``target='Y'``: score neural self-reconstruction (Y/Yp), channels resolved
    via ``resolve_input_channels``. Default; used by sec2c Fig 22 (behavioral-mode
    ECoG -> ECoG).

    ``target='Z'``: score output-signal prediction (Z/Zp), channels resolved
    via ``resolve_output_channels``. Used by sec2c Fig 23 (laplacian-mode
    ECoG -> LFP). Since laplacian test parquets have empty output_channels,
    the YAML fallback is essential.

    ``models``: tuple of {'psid', 'dpad', 'varma'} to include as columns.
    Laplacian callers pass ``('psid', 'varma')`` since DPAD has no laplacian variant.
    """
    order = (
        list(band_row_order)
        if band_row_order
        else ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    )
    model_keys = tuple(models)
    display_map = {"psid": "PSID", "dpad": "DPAD", "varma": "VARMA"}
    column_labels = tuple(display_map[m] for m in model_keys)
    resolve_fn = resolve_input_channels if target == "Y" else resolve_output_channels
    score_fn = trial_metric_y_for_model if target == "Y" else trial_metric_z_for_model

    session_runs = _discover_all(results_root, sessions, exp_type)

    band_buckets: Dict[str, Dict[Tuple[str, str], List[float]]] = {
        cond: {(b, m): [] for b in order for m in model_keys} for cond in ("off", "on")
    }
    n_sessions_used = 0
    n_trials = {"off": 0, "on": 0}

    for session, runs in session_runs.items():
        res = {
            m: load_split_results(results_root, runs[m][0], runs[m][1], split)
            for m in model_keys
            if runs.get(m, ("", ""))[0]
        }
        channels = {m: resolve_fn(res[m], runs[m][0]) for m in model_keys if m in res}
        band_idx = {
            m: _band_indices(channels[m], order) for m in model_keys if m in channels
        }

        if "psid" not in model_keys or res.get("psid") is None:
            ref = next((m for m in model_keys if res.get(m) is not None), None)
            if ref is None:
                continue
        else:
            ref = "psid"
        idx_maps = {m: _key_index_map(res[m]) for m in model_keys if res.get(m)}
        common = set(idx_maps[ref].keys())
        for m in model_keys:
            if m in idx_maps and m != ref:
                common &= set(idx_maps[m].keys())
        if not common:
            continue
        n_sessions_used += 1

        for k in sorted(common, key=lambda x: tuple(str(c) for c in x)):
            stim = normalize_stim(res[ref]["stim"][idx_maps[ref][k]])
            if stim is None:
                continue
            n_trials[stim] += 1
            for m in model_keys:
                if m not in idx_maps or not channels.get(m):
                    continue
                tidx = idx_maps[m][k]
                n_ch = len(channels[m])
                ch_vals: list[float] = []
                for ch in range(n_ch):
                    try:
                        v = score_fn(res[m], tidx, ch, metric)
                    except Exception:
                        v = float("nan")
                    ch_vals.append(v)
                for band in order:
                    ch_idxs = band_idx[m].get(band, [])
                    vals = [
                        ch_vals[i]
                        for i in ch_idxs
                        if i < len(ch_vals) and np.isfinite(ch_vals[i])
                    ]
                    if vals:
                        band_buckets[stim][(band, m)].append(float(np.mean(vals)))

    bands_present = {
        b for cond in band_buckets for (b, _m), lst in band_buckets[cond].items() if lst
    }
    row_labels = [b for b in order if b in bands_present]

    def _zmat(cond: str) -> np.ndarray:
        z = np.full((len(row_labels), len(model_keys)), np.nan, dtype=float)
        for bi, band in enumerate(row_labels):
            for mi, m in enumerate(model_keys):
                vals = band_buckets[cond][(band, m)]
                if vals:
                    z[bi, mi] = float(np.mean(vals))
        return z

    return NeuralBandMetricData(
        z_off=_zmat("off"),
        z_on=_zmat("on"),
        band_labels=row_labels,
        column_labels=column_labels,
        metric=metric,
        target=target,
        n_triplets_used=n_sessions_used,
        n_trials_off=n_trials["off"],
        n_trials_on=n_trials["on"],
    )


# ---------------------------------------------------------------------------
# Best-feature picker (used by sec2e exemplars + LFP side blocks)
# ---------------------------------------------------------------------------


def pick_best_feature_for_psid(
    split_res: Dict[str, Any],
    candidate_indices: List[int],
    metric: MetricKind,
    *,
    target: Literal["Z", "Y"] = "Z",
) -> int:
    """Return the channel index in ``candidate_indices`` whose PSID predictions
    score best on ``metric`` across all trials in ``split_res``.

    ``target="Z"`` scores Z/Zp (behavioral / LFP bands, depending on mode);
    ``target="Y"`` scores Y/Yp (neural self-prediction).

    Best = highest for pearson/vaf, lowest for rmse. Nan-only channels are
    skipped; if every candidate is nan-only, falls back to ``candidate_indices[0]``.
    """
    if not candidate_indices:
        raise ValueError("pick_best_feature_for_psid: candidate_indices is empty")
    score_fn = trial_metric_z_for_model if target == "Z" else trial_metric_y_for_model
    n_trials = len(split_res.get(target, []) or [])
    best_idx = candidate_indices[0]
    best_val: Optional[float] = None
    for ch in candidate_indices:
        vals: list[float] = []
        for t in range(n_trials):
            try:
                v = score_fn(split_res, t, ch, metric)
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(float(v))
        if not vals:
            continue
        m = float(np.mean(vals))
        if best_val is None:
            best_val, best_idx = m, ch
            continue
        better = (m < best_val) if metric == "rmse" else (m > best_val)
        if better:
            best_val, best_idx = m, ch
    return best_idx


# ---------------------------------------------------------------------------
# Session-grouped aggregation (used by sec2b Figs 7-8 — 24-box layout)
# ---------------------------------------------------------------------------


@dataclass
class SessionGroupedData:
    """4 sessions x 3 models x 2 DBS conditions = 24 cells.

    ``scores[condition][model][session_label]`` is a list of per-trial scores.
    """

    session_labels: Tuple[str, ...]
    models: Tuple[str, ...]
    scores: Dict[str, Dict[str, Dict[str, List[float]]]]
    participants: Dict[str, str]  # session_label -> participant_id (for dot colors)
    show_dpad: bool


def collect_session_grouped(
    results_root: Path,
    sessions: List[str],
    exp_type: str,
    channel_idx: int,
    split: str = "test",
    metric: MetricKind = "rmse",
    include_dpad: Optional[bool] = None,
) -> SessionGroupedData:
    """Per-session + per-model + per-DBS trial scores from ``dbs_both`` models.

    The ``dbs_both`` model is loaded for each session; its trials are split by
    ``stim`` into DBS-OFF / DBS-ON groups.
    """
    session_runs = _discover_all(results_root, sessions, exp_type)
    if not session_runs:
        raise ThesisDataError(
            "collect_session_grouped: no sessions have valid PSID+VARMA results."
        )

    if include_dpad is None:
        show_dpad = any(
            has_dpad_data(results_root, runs["dpad"][0], runs["dpad"][1], split)
            for runs in session_runs.values()
            if runs["dpad"][0]
        )
    else:
        show_dpad = bool(include_dpad)

    models = ("PSID", "DPAD", "VARMA") if show_dpad else ("PSID", "VARMA")
    session_labels = tuple(session_runs.keys())
    scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        cond: {m: {lbl: [] for lbl in session_labels} for m in models}
        for cond in ("off", "on")
    }
    participants: Dict[str, str] = {}

    score_fn = (
        trial_rmse_z_for_model
        if metric == "rmse"
        else (lambda res, i, ch: trial_metric_z_for_model(res, i, ch, metric))
    )

    for session, runs in session_runs.items():
        psid_var, psid_ts = runs["psid"]
        varma_var, varma_ts = runs["varma"]
        dpad_var, dpad_ts = runs["dpad"]

        res_p = load_split_results_required(results_root, psid_var, psid_ts, split)
        res_v = load_split_results_required(results_root, varma_var, varma_ts, split)
        has_dpad = show_dpad and bool(dpad_ts)
        res_d = (
            load_split_results(results_root, dpad_var, dpad_ts, split)
            if has_dpad
            else None
        )

        mp = _key_index_map(res_p)
        mv = _key_index_map(res_v)
        md = _key_index_map(res_d) if res_d else {}
        common = set(mp.keys()) & set(mv.keys())
        if md:
            common &= set(md.keys())
        if not common:
            raise ThesisDataError(
                f"collect_session_grouped: session {session!r} has no common trial keys."
            )

        for k in sorted(
            common, key=lambda x: (str(x[0]), str(x[1]), str(x[2]), str(x[3]))
        ):
            i_p, i_v = mp[k], mv[k]
            stim = normalize_stim(res_p["stim"][i_p])
            if stim is None:
                continue
            try:
                r_p = score_fn(res_p, i_p, channel_idx)
                r_v = score_fn(res_v, i_v, channel_idx)
                r_d = (
                    score_fn(res_d, md[k], channel_idx)
                    if res_d and k in md
                    else float("nan")
                )
            except Exception as e:
                logger.debug("Skip trial %s: %s", k, e)
                continue

            scores[stim]["PSID"][session].append(r_p)
            scores[stim]["VARMA"][session].append(r_v)
            if show_dpad and np.isfinite(r_d):
                scores[stim]["DPAD"][session].append(r_d)

            participants.setdefault(session, str(k[0]) if k else "?")

    return SessionGroupedData(
        session_labels=session_labels,
        models=models,
        scores=scores,
        participants=participants,
        show_dpad=show_dpad,
    )


def build_session_grouped_boxplot(
    data: SessionGroupedData,
    theme: ThesisTheme,
    rng: np.random.Generator,
    jitter: float = 0.12,
    metric: MetricKind = "rmse",
    target_label: str = "",
) -> Figure:
    """Two-panel grouped boxplot: DBS-OFF | DBS-ON.

    Each panel shows ``len(models)`` model groups, each group containing
    ``len(session_labels)`` session boxes. Boxes coloured by model; strip
    dots coloured by participant (same convention as pooled builder).
    """
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    models = list(data.models)
    model_colors = [
        {"PSID": COLOR_PSID, "DPAD": COLOR_DPAD, "VARMA": COLOR_VARMA}[m]
        for m in models
    ]
    sessions = list(data.session_labels)
    n_sess = len(sessions)
    group_gap = 1  # empty slot between model groups
    # x position: model i group starts at i * (n_sess + group_gap)
    xpos = {
        (m, s): mi * (n_sess + group_gap) + si
        for mi, m in enumerate(models)
        for si, s in enumerate(sessions)
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.07,
    )

    all_vals: list[float] = []
    for cond in ("off", "on"):
        for m in models:
            for s in sessions:
                all_vals.extend(v for v in data.scores[cond][m][s] if np.isfinite(v))
    y_min, y_max = metric_y_range(metric, np.asarray(all_vals, dtype=float))

    for col_idx, cond in enumerate(("off", "on")):
        for mi, (m, colour) in enumerate(zip(models, model_colors)):
            for si, s in enumerate(sessions):
                vals = [v for v in data.scores[cond][m][s] if np.isfinite(v)]
                if not vals:
                    continue
                xp = xpos[(m, s)]
                fig.add_trace(
                    go.Box(
                        y=vals,
                        x=[xp] * len(vals),
                        name=m if (col_idx == 0 and si == 0) else None,
                        marker_color=colour,
                        line=dict(color=colour, width=1.4),
                        fillcolor=_hex_to_rgba(colour, 0.45),
                        boxpoints=False,
                        quartilemethod="exclusive",
                        showlegend=(col_idx == 0 and si == 0),
                        legendgroup=m,
                        width=0.7,
                    ),
                    row=1,
                    col=col_idx + 1,
                )
                jt = rng.uniform(-jitter, jitter, size=len(vals))
                pid = data.participants.get(s, "?")
                # Per-trial dots use the model color (with opacity) so the
                # cluster is visually tied to its box.
                dot_col = _hex_to_rgba(colour, 0.6)
                fig.add_trace(
                    go.Scatter(
                        x=[xp + j for j in jt],
                        y=vals,
                        mode="markers",
                        marker=dict(size=4, color=dot_col, line=dict(width=0)),
                        showlegend=False,
                        legendgroup="trials",
                        hovertext=[f"{s} ({pid})<br>{v:.3f}" for v in vals],
                    ),
                    row=1,
                    col=col_idx + 1,
                )

    # Build shared x tick layout: one tick per session per model group.
    # Keep underscored session names (PDI1_S2 etc.) per style rule #7.
    tickvals = [xpos[(m, s)] for m in models for s in sessions]
    ticktext = [s for m in models for s in sessions]

    apply_thesis_style(
        fig,
        theme,
        height=460,
        margin=dict(l=76, r=28, t=62, b=104),
        hovermode="closest",
        legend_y=-0.24,
    )
    fig.update_layout(title=None)

    # Panel labels for the two DBS conditions (G2 convention).
    panel_label(fig, "A", "DBS-OFF", row=1, col=1, y_offset=0.10)
    panel_label(fig, "B", "DBS-ON", row=1, col=2, y_offset=0.10)

    # Legend entry for the per-trial dots — gray placeholder, note in caption
    # explains dots take the model colour of the box they sit in.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=5, color="rgba(100,100,100,0.6)", line=dict(width=0)),
            name="per-trial (model colour)",
            showlegend=True,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    x_kw = dict(
        tickvals=tickvals,
        ticktext=ticktext,
        showgrid=False,
        zeroline=False,
        showline=True,
        linecolor=fg,
        tickangle=-45,
        tickfont=dict(size=FONT_SIZE_TICK - 1),
    )
    fig.update_xaxes(**x_kw, row=1, col=1)
    fig.update_xaxes(**x_kw, row=1, col=2)

    y_kw = dict(
        range=[y_min, y_max],
        showgrid=True,
        gridcolor=grid,
        zeroline=True,
        zerolinecolor=grid,
        showline=True,
        linecolor=fg,
        tickfont=dict(size=FONT_SIZE_TICK),
    )
    fig.update_yaxes(
        title_text=metric_axis_label(metric),
        title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        **y_kw,
        row=1,
        col=1,
    )
    fig.update_yaxes(**y_kw, row=1, col=2)

    return fig


# ---------------------------------------------------------------------------
# Forecast SE band helper (4.4 exemplars + 4.5 forecast confidence)
# ---------------------------------------------------------------------------


def compute_per_step_se_band(
    true_futures: list,
    pred_futures: list,
    channel_idx: int = 0,
) -> np.ndarray:
    """Return per-step SE of residuals, pooled across test trials.

    For a forecast exemplar figure, shade ``pred_trace[m] +/- SE_band[m]``
    around the chosen trial's prediction. SE is computed as the standard
    deviation of per-step signed residuals across trials divided by
    sqrt(n_trials). Same recipe for PSID, DPAD, VARMA (apples-to-apples).

    Parameters
    ----------
    true_futures, pred_futures : list
        Per-trial forecast arrays. Each element shape ``(m,)`` or
        ``(m, n_ch)``; the channel index is extracted for multi-channel.
    channel_idx : int
        Channel to extract when arrays are 2-D.

    Returns
    -------
    se : ndarray, shape (m_min,)
        SE per forecast step. NaN where fewer than 2 trials contribute.
    """
    rows = []
    for zt, zp in zip(true_futures, pred_futures):
        if zt is None or zp is None:
            continue
        T = np.asarray(zt, dtype=float)
        P = np.asarray(zp, dtype=float)
        if T.shape != P.shape:
            continue
        if T.ndim == 1:
            t_vec, p_vec = T, P
        else:
            if T.shape[-1] <= channel_idx:
                continue
            t_vec, p_vec = T[..., channel_idx], P[..., channel_idx]
        resid = p_vec - t_vec
        if np.any(np.isfinite(resid)):
            rows.append(resid)

    if not rows:
        return np.array([])

    m_min = min(r.size for r in rows)
    arr = np.stack([r[:m_min] for r in rows], axis=0)
    n = arr.shape[0]
    if n < 2:
        return np.full(m_min, np.nan)
    sd = np.nanstd(arr, axis=0, ddof=1)
    return sd / np.sqrt(n)


# ---------------------------------------------------------------------------
# Forecast 95% CI band helper (empirical quantile of pooled residuals)
# ---------------------------------------------------------------------------


def compute_per_step_ci_band(
    true_futures: list,
    pred_futures: list,
    channel_idx: int = 0,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step 95% CI bounds from pooled residual quantiles.

    Residual[m, trial] = pred_future[m, trial] - true_future[m, trial].
    For each step m, the ``alpha/2`` and ``1 - alpha/2`` quantiles of the
    trial-axis residual distribution give the envelope offsets. Apply to an
    exemplar prediction trace as::

        lower_band[m] = pred_trace[m] + q_low[m]
        upper_band[m] = pred_trace[m] + q_high[m]

    Same recipe for PSID / DPAD / VARMA so bands are comparable.

    Returns
    -------
    q_low, q_high : ndarray, shape (m_min,)
        Quantile offsets at alpha/2 and 1-alpha/2. NaN where fewer than 2
        trials contribute.
    """
    rows = []
    for zt, zp in zip(true_futures, pred_futures):
        if zt is None or zp is None:
            continue
        T = np.asarray(zt, dtype=float)
        P = np.asarray(zp, dtype=float)
        if T.shape != P.shape:
            continue
        if T.ndim == 1:
            t_vec, p_vec = T, P
        else:
            if T.shape[-1] <= channel_idx:
                continue
            t_vec, p_vec = T[..., channel_idx], P[..., channel_idx]
        resid = p_vec - t_vec
        if np.any(np.isfinite(resid)):
            rows.append(resid)

    if not rows:
        return np.array([]), np.array([])

    m_min = min(r.size for r in rows)
    arr = np.stack([r[:m_min] for r in rows], axis=0)
    if arr.shape[0] < 2:
        return np.full(m_min, np.nan), np.full(m_min, np.nan)
    q_low = np.nanquantile(arr, alpha / 2, axis=0)
    q_high = np.nanquantile(arr, 1 - alpha / 2, axis=0)
    return q_low, q_high


def pick_best_feature_per_metric(
    psid_split_res: dict,
    candidate_indices: list,
    *,
    target: str = "Z",
    metrics: tuple = ("rmse", "pearson", "vaf"),
) -> dict:
    """Return {metric: best_feature_idx} picked from PSID test scores.

    Wrapper around ``pick_best_feature_for_psid`` that returns the per-metric
    picks in one pass. ``candidate_indices`` should already be the PSID ∩ DPAD
    ∩ VARMA channel intersection so the picked feature is plottable for all
    three models.
    """
    return {
        m: pick_best_feature_for_psid(
            psid_split_res, candidate_indices, m, target=target
        )
        for m in metrics
    }
