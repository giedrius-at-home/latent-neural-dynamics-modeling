"""Shared helpers for sec2e single-trial exemplar figures.

Pure helpers + figure builders extracted from thesis_sec2e_exemplars.ipynb so the
notebook is markdown + spec builders + render calls (matching sec6/7/8_common).
Data loaders remain in modules.lib.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl

from modules.style import (
    COLOR_PSID,
    COLOR_DPAD,
    COLOR_VARMA,
    COLOR_TRUE,
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    panel_label,
)
from modules.loaders import (
    discover_session_run,
    load_split_results,
    load_split_results_required,
)
from modules.lib.aggregate_rmse import _key_index_map, _trial_key, normalize_stim
from modules.lib.loaders import (
    load_precomputed_results,
    resolve_neural_y_channel_idx,
    extract_trial_y_series,
    extract_trial_z_series,
    thesis_exemplar_tagline,
    ThesisDataError,
)
from modules.lib.c2_forecast_timeseries import _build_one_panel as _forecast_rowdata
from modules.lib.compose import _session_mean_rmse_y_triplet
from modules.lib.transforms import z_true_and_preds
from modules.lib.forecast_horizon_rmse import _per_step_abs_err_z_future
from modules.lib.plot_config import _ALL_TRIPLETS

_FORECAST_CTX_ALPHA = 0.08
_FORECAST_FUT_ALPHA = 0.07


def _infer_varma_off_on_run_ts(varma_variant, varma_run_ts, _all_triplets):
    """Return matching OFF/ON VARMA timestamps from triplets list, or None."""
    for tri in _all_triplets:
        if tri.varma_variant == varma_variant and tri.varma_run_ts == varma_run_ts:
            return tri.varma_run_ts_off, tri.varma_run_ts_on
    return None


def _consec_pair_y(res_p, ch_idx):
    """Best adjacent OFF/ON pair scored by sum of Y Pearson r."""
    Y, Yp = res_p.get("Y", []), res_p.get("Yp", [])
    stims = res_p.get("stim", [])
    best, best_score = (None, None), -np.inf
    for i in range(len(stims) - 1):
        si = normalize_stim(stims[i])
        sj = normalize_stim(stims[i + 1])
        if si is None or sj is None or si == sj:
            continue
        off_i = i if si == "off" else i + 1
        on_i = i if si == "on" else i + 1
        score = 0.0
        for ti in (off_i, on_i):
            if ti >= len(Y) or ti >= len(Yp):
                continue
            y = np.asarray(Y[ti], dtype=float)
            yp = np.asarray(Yp[ti], dtype=float)
            ci = min(ch_idx, y.shape[1] - 1) if y.ndim == 2 else 0
            yc = y[:, ci] if y.ndim == 2 else y.ravel()
            ypc = yp[:, ci] if yp.ndim == 2 else yp.ravel()
            n = min(len(yc), len(ypc))
            if n < 10 or yc[:n].std() < 1e-9 or ypc[:n].std() < 1e-9:
                continue
            r = float(np.corrcoef(yc[:n], ypc[:n])[0, 1])
            if np.isfinite(r):
                score += r
        if score > best_score:
            best_score, best = score, (off_i, on_i)
    return best  # (off_idx, on_idx) or (None, None)


def _consec_pair_z(res_p, ch_idx):
    """Best adjacent OFF/ON pair scored by sum of Z Pearson r."""
    Z, Zp = res_p.get("Z", []), res_p.get("Zp", [])
    stims = res_p.get("stim", [])
    best, best_score = (None, None), -np.inf
    for i in range(len(stims) - 1):
        si = normalize_stim(stims[i])
        sj = normalize_stim(stims[i + 1])
        if si is None or sj is None or si == sj:
            continue
        off_i = i if si == "off" else i + 1
        on_i = i if si == "on" else i + 1
        score = 0.0
        for ti in (off_i, on_i):
            if ti >= len(Z) or ti >= len(Zp):
                continue
            z = np.asarray(Z[ti], dtype=float)
            zp = np.asarray(Zp[ti], dtype=float)
            ci = min(ch_idx, z.shape[1] - 1) if z.ndim == 2 else 0
            zc = z[:, ci] if z.ndim == 2 else z.ravel()
            zpc = zp[:, ci] if zp.ndim == 2 else zp.ravel()
            n = min(len(zc), len(zpc))
            if n < 10 or zc[:n].std() < 1e-9 or zpc[:n].std() < 1e-9:
                continue
            r = float(np.corrcoef(zc[:n], zpc[:n])[0, 1])
            if np.isfinite(r):
                score += r
        if score > best_score:
            best_score, best = score, (off_i, on_i)
    return best


def _y_feat_name(psid_variant, ch_idx, results_root):
    """Resolve actual Y (ECoG) feature name at ch_idx from PSID train parquet."""
    try:
        fdir = results_root / "psid" / psid_variant / "train"
        pqs = list(fdir.glob("test_results_*.parquet"))
        if not pqs:
            return f"Y ch{ch_idx}"
        df = pl.read_parquet(str(pqs[0]))
        if "Y_features" not in df.columns:
            return f"Y ch{ch_idx}"
        names = df[0]["Y_features"][0].to_list()
        return names[ch_idx] if ch_idx < len(names) else f"Y ch{ch_idx}"
    except Exception:
        return f"Y ch{ch_idx}"


def _slice_trial_tail(t_abs, seg_s, z_true, z_psid, z_dpad, z_varma):
    t = np.asarray(t_abs, dtype=float).ravel()
    if t.size == 0:
        e = np.array([], dtype=float)
        return e, e, e, e, e
    t_hi = float(np.nanmax(t))
    t_lo = t_hi - float(seg_s)
    m = t >= t_lo
    arrays = [
        np.asarray(a, dtype=float).ravel() for a in (z_true, z_psid, z_dpad, z_varma)
    ]
    return (t[m],) + tuple(a[m] for a in arrays)


def _mpl_side_by_side_exemplar(
    panel_off,
    panel_on,
    *,
    channel_label="",
    session_label="",
    segment_s=1.0,
):
    def _prep(p):
        t_raw = np.asarray(p["t_abs"], dtype=float)
        t_sl, zt, zp, zd, zv = _slice_trial_tail(
            t_raw, segment_s, p["z_true"], p["z_psid"], p["z_dpad"], p["z_varma"]
        )
        return t_sl, zt, zp, zd, zv

    to, zto, zpo, zdo, zvo = _prep(panel_off)
    tn, ztn, zpn, zdn, zvn = _prep(panel_on)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    for ax, t, zt, zp, zd, zv, panel, letter, title in (
        (axes[0], to, zto, zpo, zdo, zvo, panel_off, "A", "DBS-OFF"),
        (axes[1], tn, ztn, zpn, zdn, zvn, panel_on, "B", "DBS-ON"),
    ):
        if t.size == 0:
            panel_label(
                ax,
                letter,
                (
                    f"{session_label}, {title}, {channel_label}"
                    if channel_label
                    else f"{session_label}, {title}"
                ),
            )
            continue
        for y_hat, half, col in (
            (zp, panel["band_rmse_psid"], COLOR_PSID),
            (zd, panel["band_rmse_dpad"], COLOR_DPAD),
            (zv, panel["band_rmse_varma"], COLOR_VARMA),
        ):
            if half is None or not np.isfinite(half) or half <= 0:
                continue
            yh = np.asarray(y_hat, dtype=float)
            if len(t) != len(yh) or np.all(np.isnan(yh)):
                continue
            ax.fill_between(t, yh - half, yh + half, color=col, alpha=0.15, linewidth=0)
        ax.plot(t, zt, color=COLOR_TRUE, linewidth=1.4, label="true")
        if not np.all(np.isnan(zp)):
            ax.plot(t, zp, color=COLOR_PSID, linewidth=1.2, label="PSID")
        if not np.all(np.isnan(zd)):
            ax.plot(
                t, zd, color=COLOR_DPAD, linewidth=1.2, linestyle="--", label="DPAD"
            )
        if not np.all(np.isnan(zv)):
            ax.plot(
                t, zv, color=COLOR_VARMA, linewidth=1.2, linestyle=":", label="VARMA"
            )
        ax.set_xlabel("trial time (s)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        panel_label(
            ax,
            letter,
            (
                f"{session_label}, {title}, {channel_label}"
                if channel_label
                else f"{session_label}, {title}"
            ),
        )
    axes[0].legend()
    return fig


def compose_thesis_neural_figure(
    runs,
    results_root,
    i_off,
    i_on,
    channel_idx,
    *,
    section_title,
    participant_label,
    split="test",
    segment_s=1.0,
):
    res_p = load_split_results_required(
        results_root, runs.psid_variant, runs.psid_run_ts, split
    )
    res_d = load_split_results(results_root, runs.dpad_variant, runs.dpad_run_ts, split)
    res_v = load_split_results_required(
        results_root, runs.varma_variant, runs.varma_run_ts, split
    )

    ch_idx = channel_idx

    res_v_off = res_v_on = None
    if "dbs_both" in runs.varma_variant:
        v_off = runs.varma_variant.replace("dbs_both", "dbs_off")
        v_on = runs.varma_variant.replace("dbs_both", "dbs_on")
        ts_off, ts_on = runs.varma_run_ts_off, runs.varma_run_ts_on
        if ts_off is None or ts_on is None:
            inf = _infer_varma_off_on_run_ts(runs.varma_variant, runs.varma_run_ts, _ALL_TRIPLETS)
            if inf is not None:
                ts_off, ts_on = inf
        if ts_off is None or ts_on is None:
            raise ThesisDataError(f"varma_run_ts_off/on required for {section_title!r}")
        res_v_off = load_split_results_required(results_root, v_off, ts_off, split)
        res_v_on = load_split_results_required(results_root, v_on, ts_on, split)

    def _varma_res_and_idx(panel, psid_trial_idx):
        rv = res_v_off if panel == "off" else res_v_on
        if rv is not None:
            k = _trial_key(res_p, psid_trial_idx)
            mp = _key_index_map(rv)
            if k in mp:
                return rv, mp[k]
        return (
            (res_v, psid_trial_idx)
            if res_v is not None
            else ({"Y": [], "Yp": []}, psid_trial_idx)
        )

    res_v_off_use, idx_v_off = _varma_res_and_idx("off", i_off)
    res_v_on_use, idx_v_on = _varma_res_and_idx("on", i_on)
    map_v_off = _key_index_map(res_v_off_use)
    map_v_on = _key_index_map(res_v_on_use)

    band_p_off, band_d_off, band_v_off = _session_mean_rmse_y_triplet(
        res_p, res_d, res_v_off_use, map_v_off, i_off, ch_idx, "off"
    )
    band_p_on, band_d_on, band_v_on = _session_mean_rmse_y_triplet(
        res_p, res_d, res_v_on_use, map_v_on, i_on, ch_idx, "on"
    )

    off = extract_trial_y_series(
        res_p, res_d, res_v_off_use, i_off, ch_idx, varma_trial_idx=idx_v_off
    )
    on = extract_trial_y_series(
        res_p, res_d, res_v_on_use, i_on, ch_idx, varma_trial_idx=idx_v_on
    )

    zt_o, zp_o, zd_o, zv_o = z_true_and_preds(
        off.z_true_raw, off.z_psid, off.z_dpad, off.z_varma
    )
    zt_n, zp_n, zd_n, zv_n = z_true_and_preds(
        on.z_true_raw, on.z_psid, on.z_dpad, on.z_varma
    )

    panel_off = dict(
        t_abs=off.t_abs,
        z_true=zt_o,
        z_psid=zp_o,
        z_dpad=zd_o,
        z_varma=zv_o,
        band_rmse_psid=band_p_off,
        band_rmse_dpad=band_d_off,
        band_rmse_varma=band_v_off,
    )
    panel_on = dict(
        t_abs=on.t_abs,
        z_true=zt_n,
        z_psid=zp_n,
        z_dpad=zd_n,
        z_varma=zv_n,
        band_rmse_psid=band_p_on,
        band_rmse_dpad=band_d_on,
        band_rmse_varma=band_v_on,
    )

    y_meta = _y_feat_name(runs.psid_variant, ch_idx, results_root)
    caption = thesis_exemplar_tagline(
        res_p, i_off, i_on, y_meta, participant_label=participant_label
    )

    fig = _mpl_side_by_side_exemplar(
        panel_off,
        panel_on,
        channel_label=y_meta,
        session_label=section_title,
        segment_s=segment_s,
    )
    return fig, caption


def _trial_forecast_rmse(res, k_true, k_pred, trial_idx, channel_idx):
    if res is None:
        return float("nan")
    zt = res.get(k_true)
    zp = res.get(k_pred)
    if zt is None or zp is None or trial_idx >= len(zt) or trial_idx >= len(zp):
        return float("nan")
    err = _per_step_abs_err_z_future(zt[trial_idx], zp[trial_idx], channel_idx)
    if err is None or err.size == 0:
        return float("nan")
    finite = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else float("nan")


def _consec_pair_fc(res_p, k_true, k_pred, ch_idx):
    """Best adjacent OFF/ON pair by min max(rmse_off, rmse_on) in forecast results."""
    stims = res_p.get("stim", [])
    n = len(res_p.get(k_true) or [])
    best, best_score = (None, None), float("inf")
    for i in range(min(len(stims) - 1, n - 1)):
        si = normalize_stim(stims[i])
        sj = normalize_stim(stims[i + 1])
        if si is None or sj is None or si == sj:
            continue
        off_i = i if si == "off" else i + 1
        on_i = i if si == "on" else i + 1
        r_off = _trial_forecast_rmse(res_p, k_true, k_pred, off_i, ch_idx)
        r_on = _trial_forecast_rmse(res_p, k_true, k_pred, on_i, ch_idx)
        score = max(r_off, r_on)
        if np.isfinite(score) and score < best_score:
            best_score, best = score, (off_i, on_i)
    return best  # (off_idx, on_idx) or (None, None)


def _load_h5_forecast(results_root, variant, split="test", horizon="h5"):
    if not variant:
        return None
    fw = variant.split("_", 1)[0]
    fdir = results_root / fw / variant / "forecast" / horizon
    files = sorted((fdir / split).glob("test_results_*.parquet"))
    if not files:
        return None
    ts = files[0].name[len("test_results_") : -len(".parquet")]
    try:
        return load_precomputed_results(fdir, ts, split)
    except Exception:
        return None


def _resolve_varma_off_on(runs, channel_idx, neural_y_feature_name, results_root):
    res_p = load_split_results_required(
        results_root, runs.psid_variant, runs.psid_run_ts, "test"
    )
    res_v = load_split_results_required(
        results_root, runs.varma_variant, runs.varma_run_ts, "test"
    )
    res_v_off = res_v_on = None
    if "dbs_both" in runs.varma_variant:
        v_off_var = runs.varma_variant.replace("dbs_both", "dbs_off")
        v_on_var = runs.varma_variant.replace("dbs_both", "dbs_on")
        if runs.varma_run_ts_off:
            res_v_off = load_split_results_required(
                results_root, v_off_var, runs.varma_run_ts_off, "test"
            )
        if runs.varma_run_ts_on:
            res_v_on = load_split_results_required(
                results_root, v_on_var, runs.varma_run_ts_on, "test"
            )
    ch_use = resolve_neural_y_channel_idx(res_p, neural_y_feature_name, channel_idx)
    return res_p, res_v, res_v_off, res_v_on, ch_use


def _mpl_forecast_panel(
    rowdata, neu_lbl, condition_label, *, channel_label="", session_label=""
):
    (t_full, z_true, z_psid, z_dpad, z_varma, _u, _l, _rp, _rd, _rv, n_hist) = rowdata
    t_full = np.asarray(t_full, dtype=float).ravel()
    n_hist = int(n_hist)

    def _gap(a):
        a = np.asarray(a, dtype=float).ravel()
        if 0 < n_hist < len(a):
            return np.concatenate([a[:n_hist], [np.nan], a[n_hist:]])
        return a

    t_plot = _gap(t_full)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))

    if t_full.size and n_hist > 0:
        ax.axvspan(
            float(t_full[0]),
            float(t_full[n_hist - 1]),
            color=COLOR_DBS_OFF,
            alpha=_FORECAST_CTX_ALPHA,
            linewidth=0,
        )
    if t_full.size and n_hist < len(t_full):
        ax.axvspan(
            float(t_full[n_hist]),
            float(t_full[-1]),
            color=COLOR_DBS_ON,
            alpha=_FORECAST_FUT_ALPHA,
            linewidth=0,
        )
        ax.axvline(
            float(t_full[n_hist]),
            color="#444441",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
        )

    ax.plot(t_plot, _gap(z_true), color=COLOR_TRUE, linewidth=1.4, label="true")
    if not np.all(np.isnan(z_psid)):
        ax.plot(t_plot, _gap(z_psid), color=COLOR_PSID, linewidth=1.2, label="PSID")
    if not np.all(np.isnan(z_dpad)):
        ax.plot(
            t_plot,
            _gap(z_dpad),
            color=COLOR_DPAD,
            linewidth=1.2,
            linestyle="--",
            label="DPAD",
        )
    if not np.all(np.isnan(z_varma)):
        ax.plot(
            t_plot,
            _gap(z_varma),
            color=COLOR_VARMA,
            linewidth=1.2,
            linestyle=(0, (4, 1)),
            label="VARMA",
        )

    ax.set_xlabel("trial time (s)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    panel_label(
        ax,
        "A",
        (
            f"{session_label}, DBS-{condition_label}, {channel_label}"
            if channel_label
            else f"{session_label}, DBS-{condition_label}"
        ),
    )
    ax.legend()
    return fig


def _z_feat_name_from_fc(fc_p, sess_label, results_root):
    """Get actual Z feature name for ch0 from PSID train parquet."""
    try:
        # fc_p variant is like psid_z-as-neural_PDI1_S2_...
        # find matching train parquet
        psid_dirs = sorted(
            (results_root / "psid").glob(
                f"*z-as-neural*{sess_label.replace(' ','_')}*/train"
            )
        )
        if not psid_dirs:
            return "Z ch0"
        pqs = list(psid_dirs[0].glob("test_results_*.parquet"))
        if not pqs:
            return "Z ch0"
        df = pl.read_parquet(str(pqs[0]))
        if "Z_features" not in df.columns:
            return "Z ch0"
        names = df[0]["Z_features"][0].to_list()
        return names[0] if names else "Z ch0"
    except Exception:
        return "Z ch0"


def _load_cond_forecast(results_root, model, exp_type, session, condition, horizon="h5"):
    """Load h5 forecast results for model/condition/session. Returns None if missing."""
    pv, pt = discover_session_run(
        results_root, model.lower(), exp_type, session, condition
    )
    if not pt:
        return None
    fw = pv.split("_", 1)[0]
    fdir = results_root / fw / pv / "forecast" / horizon
    files = sorted((fdir / "test").glob("test_results_*.parquet"))
    if not files:
        return None
    ts = files[0].name[len("test_results_") : -len(".parquet")]
    try:
        return load_precomputed_results(fdir, ts, "test")
    except Exception:
        return None


def _pick_median_rmse_trial(fc_res, target_stim, ch_idx=0, *, none_if_empty=False):
    """Trial index closest to median NRMSE for target_stim."""
    if fc_res is None:
        return None if none_if_empty else 0
    stims = fc_res.get("stim", [])
    Z_true = fc_res.get("Z_future_true", [])
    Z_pred = fc_res.get("Z_future_pred", [])
    pairs = []
    for i, stim in enumerate(stims):
        if normalize_stim(stim) != target_stim or i >= len(Z_true) or i >= len(Z_pred):
            continue
        zt = np.asarray(Z_true[i], dtype=float)
        zp = np.asarray(Z_pred[i], dtype=float)
        zt_ch = zt[:, ch_idx] if zt.ndim == 2 else zt.ravel()
        zp_ch = zp[:, ch_idx] if zp.ndim == 2 else zp.ravel()
        nrmse = float(np.sqrt(np.nanmean((zt_ch - zp_ch) ** 2)))
        if np.isfinite(nrmse):
            pairs.append((nrmse, i))
    if not pairs:
        return None if none_if_empty else 0
    pairs.sort()
    return pairs[len(pairs) // 2][1]


def _draw_cond_row(
    ax, fc_models, ref_fc, ref_trial_idx, ch_idx, row_title, letter, fs=200.0
):
    """Draw one row: context + multi-model Z forecasts + true future.

    ref_fc / ref_trial_idx: the reference parquet (dbs_both PSID) used to look up
    the matching trial in each model's parquet via key matching.
    """
    _STYLES = {
        "PSID": ("-", COLOR_PSID),
        "DPAD": ("--", COLOR_DPAD),
        "VARMA": ((0, (4, 1)), COLOR_VARMA),
    }

    # Key-match ref trial into each model's parquet
    ref_key = _trial_key(ref_fc, ref_trial_idx)
    model_tidx = {}
    for mname, fc in fc_models.items():
        if fc is None:
            model_tidx[mname] = None
            continue
        mp = _key_index_map(fc)
        idx = mp.get(ref_key)
        model_tidx[mname] = idx  # None if trial not present

    # Use first available model as axis reference
    ref_model = next(
        (m for m in ("PSID", "DPAD", "VARMA") if model_tidx.get(m) is not None),
        None,
    )
    if ref_model is None:
        panel_label(ax, letter, row_title)
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    ref_tidx = model_tidx[ref_model]
    ref_data = fc_models[ref_model]
    concat = np.asarray(ref_data["Z_concat_for_plot"][ref_tidx], dtype=float)
    ft = np.asarray(ref_data["Z_future_true"][ref_tidx], dtype=float)
    n_total, n_future = concat.shape[0], ft.shape[0]
    n_hist = n_total - n_future

    z_hist = concat[:n_hist, ch_idx] if concat.ndim == 2 else concat[:n_hist]
    z_true_fut = ft[:, ch_idx] if ft.ndim == 2 else ft.ravel()
    t_all = np.arange(n_total) / fs
    t_fut = t_all[n_hist:]

    ax.axvspan(
        t_all[0], t_all[n_hist - 1], color=COLOR_DBS_OFF, alpha=0.06, linewidth=0
    )
    ax.axvspan(t_all[n_hist], t_all[-1], color=COLOR_DBS_ON, alpha=0.06, linewidth=0)
    ax.axvline(t_all[n_hist], color="#555", lw=0.7, ls="--", alpha=0.5)

    ax.plot(t_all[:n_hist], z_hist, color="#aaa", lw=0.8, alpha=0.7, label="history")
    ax.plot(t_fut, z_true_fut, color=COLOR_TRUE, lw=1.6, label="true")

    for mname, fc in fc_models.items():
        tidx = model_tidx[mname]
        if tidx is None or fc is None or tidx >= len(fc.get("Z_future_pred", [])):
            continue
        fp = np.asarray(fc["Z_future_pred"][tidx], dtype=float)
        z_pred = fp[:, ch_idx] if fp.ndim == 2 else fp.ravel()
        ls, col = _STYLES[mname]
        ax.plot(t_fut, z_pred, color=col, lw=1.2, ls=ls, label=mname)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    panel_label(ax, letter, row_title)


_COND_MODEL_COLORS = {
    "PSID": COLOR_PSID,
    "DPAD": COLOR_DPAD,
    "VARMA": COLOR_VARMA,
}
_COND_LINE_STYLES = {
    "dbs_both": ("-", "both"),
    "dbs_off": ("--", "off"),
    "dbs_on": ((0, (4, 1)), "on"),
}


def _draw_cond_model_panel(
    ax, fc_conds, ref_fc, ref_trial_idx, ch_idx, model_name, panel_title, letter, fs=200.0
):
    """Single panel: all DBS condition variants for one framework.

    fc_conds: {"dbs_both": fc, "dbs_off": fc, "dbs_on": fc}
    Overlays history (gray) + true future (COLOR_TRUE) + one line per condition variant
    using the framework color with different line styles.
    """
    model_color = _COND_MODEL_COLORS.get(model_name, COLOR_PSID)

    ref_key = _trial_key(ref_fc, ref_trial_idx)
    cond_tidx = {}
    for cname, fc in fc_conds.items():
        if fc is None:
            cond_tidx[cname] = None
            continue
        cond_tidx[cname] = _key_index_map(fc).get(ref_key)

    ref_cond = next(
        (c for c in ("dbs_both", "dbs_off", "dbs_on") if cond_tidx.get(c) is not None),
        None,
    )
    if ref_cond is None:
        panel_label(ax, letter, panel_title)
        return

    ref_tidx = cond_tidx[ref_cond]
    ref_data = fc_conds[ref_cond]
    concat = np.asarray(ref_data["Z_concat_for_plot"][ref_tidx], dtype=float)
    ft = np.asarray(ref_data["Z_future_true"][ref_tidx], dtype=float)
    n_total = concat.shape[0]
    n_future = ft.shape[0]
    n_hist = n_total - n_future

    z_hist = concat[:n_hist, ch_idx] if concat.ndim == 2 else concat[:n_hist]
    z_true_fut = ft[:, ch_idx] if ft.ndim == 2 else ft.ravel()
    t_all = np.arange(n_total) / fs
    t_fut = t_all[n_hist:]

    ax.axvspan(t_all[0], t_all[n_hist - 1], color=COLOR_DBS_OFF, alpha=_CTX_ALPHA, linewidth=0)
    ax.axvspan(t_all[n_hist], t_all[-1], color=COLOR_DBS_ON, alpha=_FUT_ALPHA, linewidth=0)
    ax.axvline(t_all[n_hist], color="#444441", lw=1.0, ls="--", alpha=0.6)

    ax.plot(t_all[:n_hist], z_hist, label="history", **_HIST_KW)
    ax.plot(t_fut, z_true_fut, label="observed", **_TRUE_KW)

    for cname, fc in fc_conds.items():
        tidx = cond_tidx[cname]
        ls, lbl = _COND_LINE_STYLES[cname]
        if tidx is None or fc is None or tidx >= len(fc.get("Z_future_pred", [])):
            continue
        fp = np.asarray(fc["Z_future_pred"][tidx], dtype=float)
        z_pred = fp[:, ch_idx] if fp.ndim == 2 else fp.ravel()
        ax.plot(t_fut, z_pred, color=model_color, lw=1.3, ls=ls, label=lbl)

    _clip_ylim_to_true(ax, np.concatenate([np.asarray(z_hist).ravel(), np.asarray(z_true_fut).ravel()]))
    ax.set_xlabel("trial time (s)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    panel_label(ax, letter, panel_title)


# =============================================================================
# Unified single-trial exemplar design (2x2: rows Y/Z, cols 1-step / m-step).
# One shared style table drives every recon and forecast panel so the whole
# appendix looks identical across figures.
# =============================================================================
_MODEL_STYLE = {
    "PSID": ("-", COLOR_PSID),
    "DPAD": ((0, (5, 2)), COLOR_DPAD),
    "VARMA": ((0, (1, 1.2)), COLOR_VARMA),
}
_TRUE_KW = dict(color=COLOR_TRUE, linewidth=1.6)
_HIST_KW = dict(color="#9a9a9a", linewidth=0.9, alpha=0.8)
_CTX_ALPHA = 0.07
_FUT_ALPHA = 0.06


def _clip_ylim_to_true(ax, true_trace, pad=1.5):
    """Frame the y-axis on the observed trace so an unstable model readout
    (PSID's linear kinematics readout can overshoot by >10x) leaves the frame
    instead of flattening everything else. No-op if the true trace is empty."""
    t = np.asarray(true_trace, dtype=float).ravel()
    t = t[np.isfinite(t)]
    if t.size == 0:
        return
    lo, hi = float(np.min(t)), float(np.max(t))
    c = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo), 1e-6) * pad
    ax.set_ylim(c - half, c + half)


def _draw_recon_ax(ax, series, *, letter, title, segment_s=None, legend=False):
    """One-step reconstruction on a given axis: observed + PSID/DPAD/VARMA.

    ``series`` is a TrialZSeries (t_abs, z_true_raw, z_psid, z_dpad, z_varma).
    Traces are z-scored on the true trial statistics for a common scale.
    """
    t = np.asarray(series.t_abs, dtype=float).ravel()
    zt, zp, zd, zv = z_true_and_preds(
        series.z_true_raw, series.z_psid, series.z_dpad, series.z_varma
    )
    if segment_s is not None and t.size:
        t, zt, zp, zd, zv = _slice_trial_tail(t, segment_s, zt, zp, zd, zv)
    if t.size == 0:
        panel_label(ax, letter, title)
        return
    ax.plot(t, zt, label="observed", **_TRUE_KW)
    for arr, name in ((zp, "PSID"), (zd, "DPAD"), (zv, "VARMA")):
        a = np.asarray(arr, dtype=float).ravel()
        if a.size == t.size and not np.all(np.isnan(a)):
            ls, col = _MODEL_STYLE[name]
            ax.plot(t, a, color=col, linewidth=1.3, linestyle=ls, label=name)
    _clip_ylim_to_true(ax, zt)
    ax.set_xlabel("trial time (s)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    if legend:
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    panel_label(ax, letter, title)


def _draw_forecast_ax(ax, rowdata, *, letter, title, legend=False):
    """Multi-step forecast on a given axis, matching the recon design.

    Grey history region + vertical forecast-onset divider, observed trace with a
    gap at the split, PSID/DPAD/VARMA forecasts in the shared styles.
    """
    (t_full, z_true, z_psid, z_dpad, z_varma, _u, _l, _rp, _rd, _rv, n_hist) = rowdata
    t_full = np.asarray(t_full, dtype=float).ravel()
    n_hist = int(n_hist)

    def _gap(a):
        a = np.asarray(a, dtype=float).ravel()
        if 0 < n_hist < len(a):
            return np.concatenate([a[:n_hist], [np.nan], a[n_hist:]])
        return a

    t_plot = _gap(t_full)
    if t_full.size and n_hist > 0:
        ax.axvspan(
            float(t_full[0]), float(t_full[n_hist - 1]),
            color=COLOR_DBS_OFF, alpha=_CTX_ALPHA, linewidth=0,
        )
    if t_full.size and n_hist < len(t_full):
        ax.axvspan(
            float(t_full[n_hist]), float(t_full[-1]),
            color=COLOR_DBS_ON, alpha=_FUT_ALPHA, linewidth=0,
        )
        ax.axvline(
            float(t_full[n_hist]), color="#444441",
            linewidth=1.0, linestyle="--", alpha=0.6,
        )
    ax.plot(t_plot, _gap(z_true), label="observed", **_TRUE_KW)
    for arr, name in ((z_psid, "PSID"), (z_dpad, "DPAD"), (z_varma, "VARMA")):
        a = np.asarray(arr, dtype=float).ravel()
        if not np.all(np.isnan(a)):
            ls, col = _MODEL_STYLE[name]
            ax.plot(t_plot, _gap(a), color=col, linewidth=1.3, linestyle=ls, label=name)
    _clip_ylim_to_true(ax, z_true)
    ax.set_xlabel("trial time (s)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    if legend:
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    panel_label(ax, letter, title)


def _fc_trial_idx(fc, ref_fc, ref_idx):
    """Map a reference trial index (in ref_fc) into fc's own index space."""
    if fc is None:
        return None
    return _key_index_map(fc).get(_trial_key(ref_fc, ref_idx))


def compose_exemplar_2x2(
    runs,
    results_root,
    exp_type,
    *,
    session_label,
    y_channel_idx,
    z_channel_idx,
    neural_y_feature,
    z_feat_label,
    condition,
    horizon="h5",
    segment_s=1.0,
):
    """Build one 2x2 single-trial exemplar for a (session, exp_type, condition).

    Rows: Y (cortical ECoG) then Z (prioritised signal).
    Cols: one-step reconstruction then multi-step forecast.
    A single trial (the requested DBS condition) is shown in all four panels.
    Returns (fig, info) or (None, reason).
    """
    stim = "off" if condition.lower() == "off" else "on"

    # One-step results (test split)
    res_p = load_split_results_required(
        results_root, runs.psid_variant, runs.psid_run_ts, "test"
    )
    res_d = load_split_results(results_root, runs.dpad_variant, runs.dpad_run_ts, "test")
    res_v = load_split_results_required(
        results_root, runs.varma_variant, runs.varma_run_ts, "test"
    )
    ch_y = resolve_neural_y_channel_idx(res_p, neural_y_feature, y_channel_idx)

    # Forecast results (h5)
    fc_p = _load_h5_forecast(results_root, runs.psid_variant, horizon=horizon)
    fc_d = _load_h5_forecast(results_root, runs.dpad_variant, horizon=horizon)
    fc_v_both = _load_h5_forecast(results_root, runs.varma_variant, horizon=horizon)
    _voff = runs.varma_variant.replace("dbs_both", "dbs_off") if "dbs_both" in runs.varma_variant else None
    _von = runs.varma_variant.replace("dbs_both", "dbs_on") if "dbs_both" in runs.varma_variant else None
    fc_v_off = _load_h5_forecast(results_root, _voff, horizon=horizon)
    fc_v_on = _load_h5_forecast(results_root, _von, horizon=horizon)
    if fc_p is None:
        return None, "no PSID forecast parquet"

    # Pick a trial of the requested condition from the PSID forecast parquet
    tidx_fc = _pick_median_rmse_trial(fc_p, stim, z_channel_idx, none_if_empty=True)
    if tidx_fc is None:
        return None, f"no {stim} forecast trial"

    # Map into the one-step PSID index space via trial key
    tidx_1s = _key_index_map(res_p).get(_trial_key(fc_p, tidx_fc))
    if tidx_1s is None:
        return None, "forecast trial not in one-step parquet"

    # VARMA condition-matched forecast source + index
    fc_v = (fc_v_off if stim == "off" else fc_v_on) or fc_v_both
    jv_fc = _fc_trial_idx(fc_v, fc_p, tidx_fc)
    v_1s = _key_index_map(res_v).get(_trial_key(res_p, tidx_1s))

    y_1s = extract_trial_y_series(res_p, res_d, res_v, tidx_1s, ch_y, varma_trial_idx=v_1s)
    z_1s = extract_trial_z_series(res_p, res_d, res_v, tidx_1s, z_channel_idx, varma_trial_idx=v_1s)

    y_fc = _forecast_rowdata(
        fc_p, fc_d, fc_v, tidx_fc, ch_y, forecast_target="Y", varma_trial_idx=jv_fc
    )
    z_fc = _forecast_rowdata(
        fc_p, fc_d, fc_v, tidx_fc, z_channel_idx, forecast_target="Z", varma_trial_idx=jv_fc
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4), layout="constrained")
    y_lbl = neural_y_feature
    _draw_recon_ax(
        axes[0, 0], y_1s, letter="A",
        title=f"Y (ECoG {y_lbl}), one-step", segment_s=segment_s, legend=True,
    )
    if y_fc is not None:
        _draw_forecast_ax(axes[0, 1], y_fc, letter="B", title="Y (ECoG), forecast")
    else:
        panel_label(axes[0, 1], "B", "Y (ECoG), forecast")
    _draw_recon_ax(
        axes[1, 0], z_1s, letter="C",
        title=f"Z ({z_feat_label}), one-step", segment_s=segment_s,
    )
    if z_fc is not None:
        _draw_forecast_ax(axes[1, 1], z_fc, letter="D", title=f"Z ({z_feat_label}), forecast")
    else:
        panel_label(axes[1, 1], "D", f"Z ({z_feat_label}), forecast")

    fig.suptitle(
        f"{session_label}, {exp_type}, DBS-{condition.upper()}",
        fontsize=11, fontweight="bold",
    )
    info = dict(tidx_fc=tidx_fc, tidx_1s=tidx_1s, ch_y=ch_y, stim=stim)
    return fig, info
