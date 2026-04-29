"""Glue: load results, z-score, RMSE, build Plotly figure + RMSE rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from thesis_lib.aggregate_rmse import _key_index_map, _trial_key, normalize_stim
from thesis_lib.constants import d_score_axis_label
from thesis_lib.exemplar_trials import resolve_off_on_indices_from_spec
from thesis_lib.figure import (
    ThesisPanelData,
    build_combined_gap_exemplar_figure,
    build_dbs_stacked_figure,
    build_side_by_side_exemplar_figure,
)
from thesis_lib.loaders import (
    extract_trial_y_series,
    extract_trial_z_series,
    channels_as_str_list,
    load_split_results,
    load_split_results_required,
    neural_y_feature_label,
    resolve_output_channel_display,
    thesis_exemplar_tagline,
    trial_rmse_y_for_model,
    trial_rmse_z_for_model,
)
from thesis_lib.rmse_callout import RmseRow, rmse_callout_dataframe
from thesis_lib.specs import (
    THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
    ThesisFigureSpec,
    ThesisNeuralTimeseriesSpec,
    infer_varma_off_on_run_ts,
)
from thesis_lib.loaders import ThesisDataError
from thesis_lib.transforms import rmse_z, z_true_and_preds


def _session_str(split_res: Dict[str, Any], trial_idx: int) -> str:
    seq = split_res.get("session")
    if seq is None or trial_idx >= len(seq):
        return ""
    x = seq[trial_idx]
    if isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0:
        x = x[0]
    return str(x).strip()


def _mean_finite(vals: list[float]) -> Optional[float]:
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return float(np.mean(a))


def _session_mean_rmse_z_triplet(
    res_p: Dict[str, Any],
    res_d: Optional[Dict[str, Any]],
    res_v: Dict[str, Any],
    varma_key_map: Dict[Tuple[Any, ...], int],
    exemplar_trial_idx: int,
    channel_idx: int,
    stim_want: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Mean per-trial RMSE (z) per model over PSID test rows in same session × DBS as exemplar."""
    target_sess = _session_str(res_p, exemplar_trial_idx)
    if not target_sess:
        return None, None, None
    z_list = res_p.get("Z")
    if not z_list:
        return None, None, None
    stim_list = res_p.get("stim")
    rp, rd, rv = [], [], []
    for i in range(len(z_list)):
        if z_list[i] is None:
            continue
        if _session_str(res_p, i) != target_sess:
            continue
        st = (
            normalize_stim(stim_list[i])
            if stim_list is not None and i < len(stim_list)
            else None
        )
        if st != stim_want:
            continue
        try:
            rp.append(trial_rmse_z_for_model(res_p, i, channel_idx))
        except ValueError:
            pass
        if res_d is not None:
            try:
                rd.append(trial_rmse_z_for_model(res_d, i, channel_idx))
            except ValueError:
                pass
        j = varma_key_map.get(_trial_key(res_p, i))
        if j is not None:
            try:
                rv.append(trial_rmse_z_for_model(res_v, j, channel_idx))
            except ValueError:
                pass
    return _mean_finite(rp), _mean_finite(rd), _mean_finite(rv)


def _session_mean_rmse_y_triplet(
    res_p: Dict[str, Any],
    res_d: Optional[Dict[str, Any]],
    res_v: Dict[str, Any],
    varma_key_map: Dict[Tuple[Any, ...], int],
    exemplar_trial_idx: int,
    y_channel_idx: int,
    stim_want: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    target_sess = _session_str(res_p, exemplar_trial_idx)
    if not target_sess:
        return None, None, None
    y_list = res_p.get("Y")
    if not y_list:
        return None, None, None
    stim_list = res_p.get("stim")
    rp, rd, rv = [], [], []
    for i in range(len(y_list)):
        if y_list[i] is None:
            continue
        if _session_str(res_p, i) != target_sess:
            continue
        st = (
            normalize_stim(stim_list[i])
            if stim_list is not None and i < len(stim_list)
            else None
        )
        if st != stim_want:
            continue
        try:
            rp.append(trial_rmse_y_for_model(res_p, i, y_channel_idx))
        except ValueError:
            pass
        if res_d is not None:
            try:
                rd.append(trial_rmse_y_for_model(res_d, i, y_channel_idx))
            except ValueError:
                pass
        j = varma_key_map.get(_trial_key(res_p, i))
        if j is not None:
            try:
                rv.append(trial_rmse_y_for_model(res_v, j, y_channel_idx))
            except ValueError:
                pass
    return _mean_finite(rp), _mean_finite(rd), _mean_finite(rv)


def compose_thesis_figure(spec: ThesisFigureSpec, results_root: Path):
    """
    Returns (plotly.Figure, rmse_df: pandas.DataFrame, caption: str).
    """
    res_p = load_split_results_required(
        results_root, spec.psid_variant, spec.psid_run_ts, spec.split
    )
    i_off, i_on = resolve_off_on_indices_from_spec(
        trial_idx_off=spec.trial_idx_off,
        trial_idx_on=spec.trial_idx_on,
        use_adjacent_off_on_trials=spec.use_adjacent_off_on_trials,
        split_res=res_p,
    )
    res_d = load_split_results(
        results_root, spec.dpad_variant, spec.dpad_run_ts, spec.split
    )
    res_v = load_split_results_required(
        results_root, spec.varma_variant, spec.varma_run_ts, spec.split
    )
    res_v_off: Optional[dict] = None
    res_v_on: Optional[dict] = None
    if "dbs_both" in spec.varma_variant:
        v_off = spec.varma_variant.replace("dbs_both", "dbs_off")
        v_on = spec.varma_variant.replace("dbs_both", "dbs_on")
        ts_off = spec.varma_run_ts_off
        ts_on = spec.varma_run_ts_on
        if ts_off is None or ts_on is None:
            inf = infer_varma_off_on_run_ts(spec.varma_variant, spec.varma_run_ts)
            if inf is not None:
                ts_off, ts_on = inf
        if ts_off is None or ts_on is None:
            raise ThesisDataError(
                f"ThesisFigureSpec {spec.section_title!r}: varma_run_ts_off and varma_run_ts_on "
                f"must be set (or match _ALL_TRIPLETS) when using dbs_both VARMA."
            )
        res_v_off = load_split_results_required(results_root, v_off, ts_off, spec.split)
        res_v_on = load_split_results_required(results_root, v_on, ts_on, spec.split)

    _empty_varma = {"Z": [], "Zp": []}

    def _varma_res_and_idx(panel: str, psid_trial_idx: int):
        if panel == "off" and res_v_off is not None:
            mp = _key_index_map(res_v_off)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_off, mp[k]
        elif panel == "on" and res_v_on is not None:
            mp = _key_index_map(res_v_on)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_on, mp[k]
        if res_v is not None:
            return res_v, psid_trial_idx
        return _empty_varma, psid_trial_idx

    res_v_off_use, idx_v_off = _varma_res_and_idx("off", i_off)
    res_v_on_use, idx_v_on = _varma_res_and_idx("on", i_on)
    map_v_off = _key_index_map(res_v_off_use)
    map_v_on = _key_index_map(res_v_on_use)

    band_p_off, band_d_off, band_v_off = _session_mean_rmse_z_triplet(
        res_p,
        res_d,
        res_v_off_use,
        map_v_off,
        i_off,
        spec.channel_idx,
        "off",
    )
    band_p_on, band_d_on, band_v_on = _session_mean_rmse_z_triplet(
        res_p,
        res_d,
        res_v_on_use,
        map_v_on,
        i_on,
        spec.channel_idx,
        "on",
    )

    off = extract_trial_z_series(
        res_p,
        res_d,
        res_v_off_use,
        i_off,
        spec.channel_idx,
        varma_trial_idx=idx_v_off,
    )
    on = extract_trial_z_series(
        res_p,
        res_d,
        res_v_on_use,
        i_on,
        spec.channel_idx,
        varma_trial_idx=idx_v_on,
    )

    zt_o, zp_o, zd_o, zv_o = z_true_and_preds(
        off.z_true_raw, off.z_psid, off.z_dpad, off.z_varma
    )
    zt_n, zp_n, zd_n, zv_n = z_true_and_preds(
        on.z_true_raw, on.z_psid, on.z_dpad, on.z_varma
    )

    rmse_off = RmseRow(
        "OFF — RMSE",
        rmse_z(zt_o, zp_o),
        rmse_z(zt_o, zd_o),
        rmse_z(zt_o, zv_o),
    )
    rmse_on = RmseRow(
        "ON — RMSE",
        rmse_z(zt_n, zp_n),
        rmse_z(zt_n, zd_n),
        rmse_z(zt_n, zv_n),
    )

    panel_off = ThesisPanelData(
        t_abs=off.t_abs,
        z_true=zt_o,
        z_psid=zp_o,
        z_dpad=zd_o,
        z_varma=zv_o,
        psid_sigma=None,
        dbs_label="DBS-OFF",
        band_rmse_psid=band_p_off,
        band_rmse_dpad=band_d_off,
        band_rmse_varma=band_v_off,
    )
    panel_on = ThesisPanelData(
        t_abs=on.t_abs,
        z_true=zt_n,
        z_psid=zp_n,
        z_dpad=zd_n,
        z_varma=zv_n,
        psid_sigma=None,
        dbs_label="DBS-ON",
        band_rmse_psid=band_p_on,
        band_rmse_dpad=band_d_on,
        band_rmse_varma=band_v_on,
    )

    meta_ch = channels_as_str_list(res_p.get("output_channels"))
    if spec.channel_idx < len(meta_ch):
        ch_name = meta_ch[spec.channel_idx]
        used_decl = False
    else:
        if 0 <= spec.channel_idx < len(THESIS_DECLARED_BEHAVIORAL_OUTPUTS):
            ch_name = THESIS_DECLARED_BEHAVIORAL_OUTPUTS[spec.channel_idx]
            used_decl = True
        else:
            ch_name, used_decl = resolve_output_channel_display(
                res_p,
                spec.channel_idx,
                declared_outputs=THESIS_DECLARED_BEHAVIORAL_OUTPUTS,
            )
    y_axis_label = d_score_axis_label(ch_name)
    caption = thesis_exemplar_tagline(
        res_p, i_off, i_on, ch_name, participant_label=spec.participant_label
    )
    ce = (spec.caption_extra or "").strip()
    if ce:
        caption = f"{caption} · {ce}"
    if used_decl:
        pass  # declared output names already included in caption via channel name

    if spec.exemplar_layout == "side_by_side":
        fig = build_side_by_side_exemplar_figure(
            panel_off,
            panel_on,
            spec.theme,
            y_axis_label=y_axis_label,
            segment_s=spec.exemplar_mid_segment_s,
        )
    elif spec.exemplar_layout == "combined_gap":
        fig = build_combined_gap_exemplar_figure(
            panel_off,
            panel_on,
            spec.theme,
            y_axis_label=y_axis_label,
            segment_s=spec.exemplar_mid_segment_s,
            gap_s=spec.exemplar_abs_gap_s,
            side_extend_s=spec.exemplar_side_extend_s,
            show_psid_sigma_band=False,
            use_absolute_time=True,
        )
    else:
        fig = build_dbs_stacked_figure(
            panel_off,
            panel_on,
            spec.theme,
            y_axis_label=y_axis_label,
            caption=caption,
            show_psid_sigma_band=False,
            use_absolute_time=True,
        )

    rmse_df = rmse_callout_dataframe(rmse_off, rmse_on)
    return fig, rmse_df, caption


def compose_thesis_neural_figure(spec: ThesisNeuralTimeseriesSpec, results_root: Path):
    """
    Neural Y vs Ŷ (z-scored per trial), two DBS rows. No PSID σ band (defined for behavioral Z on val).
    Returns (figure, rmse_df, caption).
    """
    res_p = load_split_results_required(
        results_root, spec.psid_variant, spec.psid_run_ts, spec.split
    )
    i_off, i_on = resolve_off_on_indices_from_spec(
        trial_idx_off=spec.trial_idx_off,
        trial_idx_on=spec.trial_idx_on,
        use_adjacent_off_on_trials=spec.use_adjacent_off_on_trials,
        split_res=res_p,
    )
    res_d = load_split_results(
        results_root, spec.dpad_variant, spec.dpad_run_ts, spec.split
    )
    res_v = load_split_results_required(
        results_root, spec.varma_variant, spec.varma_run_ts, spec.split
    )
    res_v_off: Optional[dict] = None
    res_v_on: Optional[dict] = None
    if "dbs_both" in spec.varma_variant:
        v_off = spec.varma_variant.replace("dbs_both", "dbs_off")
        v_on = spec.varma_variant.replace("dbs_both", "dbs_on")
        ts_off = spec.varma_run_ts_off
        ts_on = spec.varma_run_ts_on
        if ts_off is None or ts_on is None:
            inf = infer_varma_off_on_run_ts(spec.varma_variant, spec.varma_run_ts)
            if inf is not None:
                ts_off, ts_on = inf
        if ts_off is None or ts_on is None:
            raise ThesisDataError(
                f"ThesisNeuralTimeseriesSpec {spec.section_title!r}: varma_run_ts_off/on required "
                f"for dbs_both VARMA (or match _ALL_TRIPLETS)."
            )
        res_v_off = load_split_results_required(results_root, v_off, ts_off, spec.split)
        res_v_on = load_split_results_required(results_root, v_on, ts_on, spec.split)

    def _varma_res_and_idx(panel: str, psid_trial_idx: int):
        if panel == "off" and res_v_off is not None:
            mp = _key_index_map(res_v_off)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_off, mp[k]
        elif panel == "on" and res_v_on is not None:
            mp = _key_index_map(res_v_on)
            k = _trial_key(res_p, psid_trial_idx)
            if k in mp:
                return res_v_on, mp[k]
        if res_v is not None:
            return res_v, psid_trial_idx
        return {"Y": [], "Yp": []}, psid_trial_idx

    res_v_off_use, idx_v_off = _varma_res_and_idx("off", i_off)
    res_v_on_use, idx_v_on = _varma_res_and_idx("on", i_on)
    map_v_off = _key_index_map(res_v_off_use)
    map_v_on = _key_index_map(res_v_on_use)

    band_p_off, band_d_off, band_v_off = _session_mean_rmse_y_triplet(
        res_p,
        res_d,
        res_v_off_use,
        map_v_off,
        i_off,
        spec.neural_y_channel_idx,
        "off",
    )
    band_p_on, band_d_on, band_v_on = _session_mean_rmse_y_triplet(
        res_p,
        res_d,
        res_v_on_use,
        map_v_on,
        i_on,
        spec.neural_y_channel_idx,
        "on",
    )

    off = extract_trial_y_series(
        res_p,
        res_d,
        res_v_off_use,
        i_off,
        spec.neural_y_channel_idx,
        varma_trial_idx=idx_v_off,
    )
    on = extract_trial_y_series(
        res_p,
        res_d,
        res_v_on_use,
        i_on,
        spec.neural_y_channel_idx,
        varma_trial_idx=idx_v_on,
    )

    zt_o, zp_o, zd_o, zv_o = z_true_and_preds(
        off.z_true_raw, off.z_psid, off.z_dpad, off.z_varma
    )
    zt_n, zp_n, zd_n, zv_n = z_true_and_preds(
        on.z_true_raw, on.z_psid, on.z_dpad, on.z_varma
    )

    rmse_off = RmseRow(
        "OFF — RMSE",
        rmse_z(zt_o, zp_o),
        rmse_z(zt_o, zd_o),
        rmse_z(zt_o, zv_o),
    )
    rmse_on = RmseRow(
        "ON — RMSE",
        rmse_z(zt_n, zp_n),
        rmse_z(zt_n, zd_n),
        rmse_z(zt_n, zv_n),
    )

    panel_off = ThesisPanelData(
        t_abs=off.t_abs,
        z_true=zt_o,
        z_psid=zp_o,
        z_dpad=zd_o,
        z_varma=zv_o,
        psid_sigma=None,
        dbs_label="DBS-OFF",
        band_rmse_psid=band_p_off,
        band_rmse_dpad=band_d_off,
        band_rmse_varma=band_v_off,
    )
    panel_on = ThesisPanelData(
        t_abs=on.t_abs,
        z_true=zt_n,
        z_psid=zp_n,
        z_dpad=zd_n,
        z_varma=zv_n,
        psid_sigma=None,
        dbs_label="DBS-ON",
        band_rmse_psid=band_p_on,
        band_rmse_dpad=band_d_on,
        band_rmse_varma=band_v_on,
    )

    y_meta = neural_y_feature_label(
        res_p,
        spec.neural_y_channel_idx,
        neural_y_feature_name=spec.neural_y_feature_name,
    )
    y_axis_label = y_meta  # neural channel: use raw feature name as-is

    caption = thesis_exemplar_tagline(
        res_p, i_off, i_on, y_meta, participant_label=spec.participant_label
    )
    ce = (spec.caption_extra or "").strip()
    if ce:
        caption = f"{caption} · {ce}"

    if spec.exemplar_layout == "side_by_side":
        fig = build_side_by_side_exemplar_figure(
            panel_off,
            panel_on,
            spec.theme,
            y_axis_label=y_axis_label,
            segment_s=spec.exemplar_mid_segment_s,
            true_series_name="y_true",
            prediction_line_dash="4 3",
        )
    elif spec.exemplar_layout == "combined_gap":
        fig = build_combined_gap_exemplar_figure(
            panel_off,
            panel_on,
            spec.theme,
            y_axis_label=y_axis_label,
            segment_s=spec.exemplar_mid_segment_s,
            gap_s=spec.exemplar_abs_gap_s,
            side_extend_s=spec.exemplar_side_extend_s,
            show_psid_sigma_band=False,
            true_series_name="y_true",
            prediction_line_dash="4 3",
            use_absolute_time=True,
        )
    else:
        fig = build_dbs_stacked_figure(
            panel_off,
            panel_on,
            spec.theme,
            y_axis_label=y_axis_label,
            caption=caption,
            show_psid_sigma_band=False,
            true_series_name="y_true",
            prediction_line_dash="4 3",
            use_absolute_time=True,
        )
    rmse_df = rmse_callout_dataframe(rmse_off, rmse_on)
    return fig, rmse_df, caption
