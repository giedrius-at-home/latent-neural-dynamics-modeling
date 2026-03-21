"""Glue: load results, z-score, RMSE, build Plotly figure + RMSE rows."""

from __future__ import annotations

from pathlib import Path

from dashboard.thesis.figure import ThesisPanelData, build_dbs_stacked_figure
from dashboard.thesis.loaders import extract_trial_z_series, load_split_results
from dashboard.thesis.psid_validation_sigma import sigma_z_from_psid_val_residuals
from dashboard.thesis.rmse_callout import RmseRow, rmse_callout_dataframe
from dashboard.thesis.specs import ThesisFigureSpec
from dashboard.thesis.transforms import rmse_z, z_true_and_preds


def compose_thesis_figure(spec: ThesisFigureSpec, results_root: Path):
    """
    Returns (plotly.Figure, rmse_df: pandas.DataFrame, caption: str).
    """
    val_psid = load_split_results(
        results_root, spec.psid_variant, spec.psid_run_ts, "val"
    )
    sigma = sigma_z_from_psid_val_residuals(val_psid, spec.channel_idx)

    res_p = load_split_results(
        results_root, spec.psid_variant, spec.psid_run_ts, spec.split
    )
    res_d = load_split_results(
        results_root, spec.dpad_variant, spec.dpad_run_ts, spec.split
    )
    res_v = load_split_results(
        results_root, spec.varma_variant, spec.varma_run_ts, spec.split
    )

    for name, r in ("PSID", res_p), ("DPAD", res_d), ("VARMA", res_v):
        if r is None:
            raise FileNotFoundError(
                f"No precomputed results for {name} variant — run Tester / check paths."
            )

    off = extract_trial_z_series(
        res_p, res_d, res_v, spec.trial_idx_off, spec.channel_idx
    )
    on = extract_trial_z_series(
        res_p, res_d, res_v, spec.trial_idx_on, spec.channel_idx
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
        psid_sigma=sigma,
        dbs_label="DBS-OFF",
    )
    panel_on = ThesisPanelData(
        t_abs=on.t_abs,
        z_true=zt_n,
        z_psid=zp_n,
        z_dpad=zd_n,
        z_varma=zv_n,
        psid_sigma=sigma,
        dbs_label="DBS-ON",
    )

    cap_parts = [
        f"Participant {spec.participant_label} — {spec.caption_extra}".strip(),
        f"Y: {spec.y_axis_label.replace('(z)', '').strip()} (z-scored).",
        "Shaded region: ±1 σ band from PSID validation residual std (pooled z-residuals).",
    ]
    caption = " ".join(p for p in cap_parts if p)

    fig = build_dbs_stacked_figure(
        panel_off,
        panel_on,
        spec.theme,
        y_axis_label=spec.y_axis_label,
        caption=caption,
    )

    rmse_df = rmse_callout_dataframe(rmse_off, rmse_on)
    return fig, rmse_df, caption
