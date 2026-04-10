# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Section 6: Summary, Supplementary Analysis, and Appendix — 6 figures
#
# | # | Figure | Builder |
# |---|--------|--------|
# | 59 | Latent phase space trajectories | `build_latent_phase_space_figure()` |
# | 60 | Classification vs dimensionality heatmap | `build_classification_heatmap_figure()` |
# | 61 | PSID Cy importance heatmap | `build_psid_cy_importance_figure()` |
# | 62 | PSID Cz readout matrix | `build_psid_cz_figure()` |
# | 63 | DPAD training curves | `build_dpad_training_curves_figure()` |
# | 64 | Data efficiency | `build_data_efficiency_figure()` |

# %%
import sys, os
os.chdir('/home/bobby/repos/latent-neural-dynamics-modeling')
sys.path.insert(0, '.')

from pathlib import Path
OUT = Path('thesis_figures/sec6'); OUT.mkdir(parents=True, exist_ok=True)

results_root = Path('results').resolve()

# ---------------------------------------------------------------------------
# Inline session config — dataclass types only from specs.py
# ---------------------------------------------------------------------------
from dashboard.thesis.specs import (
    AlignedTriplet,
    LatentPhasePanel, LatentPhaseRow, ThesisLatentPhaseSpec,
    PsidCyPanel, PsidCyRow, ThesisPsidCyImportanceSpec, ThesisPsidCzSpec,
)
from notebooks.thesis_style import ThesisTheme

# -- Session triplets -------------------------------------------------------

TRIPLET_PDI1_S2 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI1_2_nx_25_n2_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_222003",
    dpad_variant="dpad_behavioral_PDI1_2_nx_25_n2_e3000_dbs_both_200Hz_narrow_band",
    dpad_run_ts="",
    varma_variant="varma_behavioral_PDI1_2_p30_q1_top20_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260409_105705",
    label="PDI1_S2",
    psid_run_ts_off="20260408_224606", psid_run_ts_on="20260408_223912",
    dpad_run_ts_off=None, dpad_run_ts_on=None,
    varma_run_ts_off="20260409_113230", varma_run_ts_on="20260409_112938",
    varma_run_ts_eval_off="20260409_110048", varma_run_ts_eval_on="20260409_110433",
)
TRIPLET_PDI1_S4 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI1_4_nx_15_n2_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_194919",
    dpad_variant="dpad_behavioral_PDI1_4_nx_15_n2_e3000_dbs_both_200Hz_narrow_band",
    dpad_run_ts="",
    varma_variant="varma_behavioral_PDI1_4_p30_q1_top20_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260409_104823",
    label="PDI1_S4",
    psid_run_ts_off="20260408_200652", psid_run_ts_on="20260408_200052",
    dpad_run_ts_off=None, dpad_run_ts_on=None,
    varma_run_ts_off="20260409_112734", varma_run_ts_on="20260409_112612",
    varma_run_ts_eval_off="20260409_105059", varma_run_ts_eval_on="20260409_105339",
)
TRIPLET_PDI4_S2 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI4_2_nx_30_n6_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_162132",
    dpad_variant="dpad_behavioral_PDI4_2_nx_30_n6_e3000_dbs_both_200Hz_narrow_band",
    dpad_run_ts="",
    varma_variant="varma_behavioral_PDI4_2_p30_q1_top20_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260409_111451",
    label="PDI4_S2",
    psid_run_ts_off="20260408_164031", psid_run_ts_on="20260408_163407",
    dpad_run_ts_off=None, dpad_run_ts_on=None,
    varma_run_ts_off="20260409_111913", varma_run_ts_on="20260409_111754",
    varma_run_ts_eval_off="20260409_111754", varma_run_ts_eval_on="20260409_111913",
)
TRIPLET_PDI4_S3 = AlignedTriplet(
    psid_variant="psid_behavioral_PDI4_3_nx_25_n6_i50_dbs_both_200Hz_narrow_band",
    psid_run_ts="20260408_185522",
    dpad_variant="dpad_behavioral_PDI4_3_nx_25_n6_e3000_dbs_both_200Hz_narrow_band",
    dpad_run_ts="",
    varma_variant="varma_behavioral_PDI4_3_p30_q1_top20_dbs_both_200Hz_narrow_band",
    varma_run_ts="20260409_110921",
    label="PDI4_S3",
    psid_run_ts_off="20260408_191423", psid_run_ts_on="20260408_190749",
    dpad_run_ts_off=None, dpad_run_ts_on=None,
    varma_run_ts_off="20260409_111318", varma_run_ts_on="20260409_111147",
    varma_run_ts_eval_off="20260409_111147", varma_run_ts_eval_on="20260409_111318",
)

# -- Spec objects -----------------------------------------------------------

THESIS_LATENT_PHASE = [
    ThesisLatentPhaseSpec(
        section_title="Latent phase-space trajectories",
        rows=(
            LatentPhaseRow("PDI1", (
                LatentPhasePanel("S2", TRIPLET_PDI1_S2.psid_variant, TRIPLET_PDI1_S2.psid_run_ts,
                                 TRIPLET_PDI1_S2.dpad_variant, TRIPLET_PDI1_S2.dpad_run_ts),
                LatentPhasePanel("S4", TRIPLET_PDI1_S4.psid_variant, TRIPLET_PDI1_S4.psid_run_ts,
                                 TRIPLET_PDI1_S4.dpad_variant, TRIPLET_PDI1_S4.dpad_run_ts),
            )),
            LatentPhaseRow("PDI4", (
                LatentPhasePanel("S2", TRIPLET_PDI4_S2.psid_variant, TRIPLET_PDI4_S2.psid_run_ts,
                                 TRIPLET_PDI4_S2.dpad_variant, TRIPLET_PDI4_S2.dpad_run_ts),
                LatentPhasePanel("S3", TRIPLET_PDI4_S3.psid_variant, TRIPLET_PDI4_S3.psid_run_ts,
                                 TRIPLET_PDI4_S3.dpad_variant, TRIPLET_PDI4_S3.dpad_run_ts),
            )),
        ),
    ),
]

THESIS_PSID_CY_IMPORTANCE = [
    ThesisPsidCyImportanceSpec(
        section_title="PSID Cy importance",
        rows=(
            PsidCyRow("PDI1", (
                PsidCyPanel("S2", TRIPLET_PDI1_S2.psid_variant, TRIPLET_PDI1_S2.psid_run_ts),
                PsidCyPanel("S4", TRIPLET_PDI1_S4.psid_variant, TRIPLET_PDI1_S4.psid_run_ts),
            )),
            PsidCyRow("PDI4", (
                PsidCyPanel("S2", TRIPLET_PDI4_S2.psid_variant, TRIPLET_PDI4_S2.psid_run_ts),
                PsidCyPanel("S3", TRIPLET_PDI4_S3.psid_variant, TRIPLET_PDI4_S3.psid_run_ts),
            )),
        ),
        split="test",
        show_beta_box=True,
    ),
]

THESIS_PSID_CZ_HEATMAP = [
    ThesisPsidCzSpec(
        section_title="PSID Cz readout matrix",
        rows=(
            PsidCyRow("PDI1", (
                PsidCyPanel("S2", TRIPLET_PDI1_S2.psid_variant, TRIPLET_PDI1_S2.psid_run_ts),
                PsidCyPanel("S4", TRIPLET_PDI1_S4.psid_variant, TRIPLET_PDI1_S4.psid_run_ts),
            )),
            PsidCyRow("PDI4", (
                PsidCyPanel("S2", TRIPLET_PDI4_S2.psid_variant, TRIPLET_PDI4_S2.psid_run_ts),
                PsidCyPanel("S3", TRIPLET_PDI4_S3.psid_variant, TRIPLET_PDI4_S3.psid_run_ts),
            )),
        ),
        split="test",
    ),
]

# %% [markdown]
# ## Fig 59: Latent phase space trajectories

# %%
import logging
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from notebooks.thesis_style import (
    COLOR_DBS_OFF,
    COLOR_DBS_ON,
    FONT_FAMILY,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_BASE,
    FONT_SIZE_TICK,
    apply_thesis_style,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.latent_phase_space_data import kde_on_fixed_grid, load_panel_latent_data, PanelLatentData

logger = logging.getLogger(__name__)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


_OFF_RGB = _hex_to_rgb(COLOR_DBS_OFF)
_ON_RGB = _hex_to_rgb(COLOR_DBS_ON)


def _rgba(r: int, g: int, b: int, a: float) -> str:
    return f"rgba({r},{g},{b},{a})"


def _kde_colorscale(
    z: np.ndarray,
    t_lo: float,
    t_hi: float,
    rgb: Tuple[int, int, int],
) -> Tuple[List[list], float, float]:
    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    span = zmax - zmin + 1e-15
    r, g, b = rgb

    def norm(t: float) -> float:
        return float(np.clip((t - zmin) / span, 0.0, 1.0))

    a0, a1, a2 = 0.0, 0.38, 0.62
    n_lo = norm(t_lo)
    n_hi = norm(t_hi)
    return (
        [
            [0.0, _rgba(r, g, b, a0)],
            [n_lo, _rgba(r, g, b, a0)],
            [min(n_lo + 1e-6, 1.0), _rgba(r, g, b, a1)],
            [n_hi, _rgba(r, g, b, a1)],
            [min(n_hi + 1e-6, 1.0), _rgba(r, g, b, a2)],
            [1.0, _rgba(r, g, b, a2 + 0.08)],
        ],
        zmin,
        zmax,
    )


def _common_xy_mesh(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    grid_n: int,
    pad: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.concatenate([np.asarray(a, dtype=float).ravel() for a, _ in pairs])
    ys = np.concatenate([np.asarray(b, dtype=float).ravel() for _, b in pairs])
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if xs.size == 0:
        return np.linspace(0, 1, grid_n), np.linspace(0, 1, grid_n)
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    dx = xmax - xmin + 1e-9
    dy = ymax - ymin + 1e-9
    xmin -= pad * dx
    xmax += pad * dx
    ymin -= pad * dy
    ymax += pad * dy
    return np.linspace(xmin, xmax, grid_n), np.linspace(ymin, ymax, grid_n)


def _add_kde_heatmap(
    fig: Figure,
    row: int,
    col: int,
    xi: np.ndarray,
    yi: np.ndarray,
    zi: np.ndarray,
    p_lo: float,
    p_hi: float,
    rgb: Tuple[int, int, int],
    showlegend: bool,
    name: str,
    opacity: float,
) -> None:
    flat = zi[np.isfinite(zi)].ravel()
    if flat.size == 0:
        return
    t_lo = float(np.percentile(flat, p_lo))
    t_hi = float(np.percentile(flat, p_hi))
    cs, zmin, zmax = _kde_colorscale(zi, t_lo, t_hi, rgb)
    fig.add_trace(
        go.Heatmap(
            x=xi,
            y=yi,
            z=zi,
            zmin=zmin,
            zmax=zmax,
            colorscale=cs,
            showscale=False,
            opacity=opacity,
            hoverinfo="skip",
            name=name,
            legendgroup=name,
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def _add_dual_kde_heatmaps(
    fig: Figure,
    row: int,
    col: int,
    x_off: np.ndarray,
    y_off: np.ndarray,
    x_on: np.ndarray,
    y_on: np.ndarray,
    grid_n: int,
    p_lo: float,
    p_hi: float,
    showlegend_off: bool,
    showlegend_on: bool,
) -> None:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    if len(x_off) >= 2:
        pairs.append((x_off, y_off))
    if len(x_on) >= 2:
        pairs.append((x_on, y_on))
    if not pairs:
        return
    xi, yi = _common_xy_mesh(pairs, grid_n)
    if len(x_off) >= 2:
        zo = kde_on_fixed_grid(x_off, y_off, xi, yi)
        _add_kde_heatmap(
            fig, row, col, xi, yi, zo, p_lo, p_hi, _OFF_RGB,
            False, "DBS-OFF (density)", 0.40,
        )
    if len(x_on) >= 2:
        zn = kde_on_fixed_grid(x_on, y_on, xi, yi)
        _add_kde_heatmap(
            fig, row, col, xi, yi, zn, p_lo, p_hi, _ON_RGB,
            False, "DBS-ON (density)", 0.45,
        )


def _empty_cell(fig: Figure, row: int, col: int) -> None:
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(opacity=0), showlegend=False),
        row=row,
        col=col,
    )
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, row=row, col=col)


def _xy_range_from_points(
    xs: List[np.ndarray],
    ys: List[np.ndarray],
    pad: float = 0.05,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xa_parts: List[np.ndarray] = []
    ya_parts: List[np.ndarray] = []
    for xa, ya in zip(xs, ys):
        xa = np.asarray(xa, dtype=float).ravel()
        ya = np.asarray(ya, dtype=float).ravel()
        m = np.isfinite(xa) & np.isfinite(ya)
        if np.any(m):
            xa_parts.append(xa[m])
            ya_parts.append(ya[m])
    if not xa_parts:
        return (0.0, 1.0), (0.0, 1.0)
    xa = np.concatenate(xa_parts)
    ya = np.concatenate(ya_parts)
    xmin, xmax = float(np.min(xa)), float(np.max(xa))
    ymin, ymax = float(np.min(ya)), float(np.max(ya))
    dx = xmax - xmin + 1e-9
    dy = ymax - ymin + 1e-9
    return (xmin - pad * dx, xmax + pad * dx), (ymin - pad * dy, ymax + pad * dy)


for lp_spec in THESIS_LATENT_PHASE:
    rows_spec = list(lp_spec.rows)
    n_p = len(rows_spec)
    if n_p == 0:
        raise ValueError("ThesisLatentPhaseSpec.rows is empty")
    max_cols = max(len(r.panels) for r in rows_spec)
    n_rows = 2 * n_p
    theme = lp_spec.theme
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    p_lo, p_hi = lp_spec.density_percentiles

    subplot_titles: List[str | None] = []
    for _ in range(2):
        for r in rows_spec:
            for c in range(max_cols):
                if c < len(r.panels):
                    subplot_titles.append(r.panels[c].session_label)
                else:
                    subplot_titles.append("")

    row_titles = [f"{r.participant_label} · PSID" for r in rows_spec] + [
        f"{r.participant_label} · DPAD" for r in rows_spec
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=max_cols,
        subplot_titles=subplot_titles[: n_rows * max_cols] if subplot_titles else None,
        vertical_spacing=0.07,
        horizontal_spacing=0.09,
        row_titles=row_titles,
    )

    cache: dict[tuple[int, int], PanelLatentData] = {}
    for pi, rspec in enumerate(rows_spec):
        for ci, panel in enumerate(rspec.panels):
            cache[(pi, ci)] = load_panel_latent_data(
                results_root,
                panel,
                lp_spec.split,
                lp_spec.n_psid_latent,
                lp_spec.n_dpad_latent,
                lp_spec.n_trajectory_trials,
                lp_spec.trajectory_seed,
            )

    for ri in range(1, n_rows + 1):
        pi = (ri - 1) if (ri - 1) < n_p else (ri - 1 - n_p)
        is_psid = (ri - 1) < n_p
        rspec = rows_spec[pi]

        for ci in range(1, max_cols + 1):
            if ci > len(rspec.panels):
                _empty_cell(fig, ri, ci)
                continue

            data = cache[(pi, ci - 1)]

            if is_psid:
                xr, yr = _xy_range_from_points(
                    [data.x_psid_off, data.x_psid_on],
                    [data.y_psid_off, data.y_psid_on],
                )
                _add_dual_kde_heatmaps(
                    fig,
                    ri,
                    ci,
                    data.x_psid_off,
                    data.y_psid_off,
                    data.x_psid_on,
                    data.y_psid_on,
                    lp_spec.kde_grid,
                    p_lo,
                    p_hi,
                    False,
                    False,
                )
                fig.update_xaxes(range=list(xr), row=ri, col=ci)
                fig.update_yaxes(range=list(yr), row=ri, col=ci)
                if ci == 1:
                    fig.update_yaxes(title_text="x\u2082", row=ri, col=ci)
                if ri == n_p:
                    fig.update_xaxes(title_text="x\u2081", row=ri, col=ci)
            else:
                has_dpad = (
                    len(np.asarray(data.x_dpad_off).ravel()) >= 2
                    or len(np.asarray(data.x_dpad_on).ravel()) >= 2
                )
                if not has_dpad:
                    fig.add_annotation(
                        text="DPAD retraining at 200 Hz<br>(data unavailable)",
                        xref="x domain",
                        yref="y domain",
                        x=0.5,
                        y=0.5,
                        xanchor="center",
                        yanchor="middle",
                        showarrow=False,
                        font=dict(size=FONT_SIZE_ANNOTATION, color=fg, family=FONT_FAMILY),
                        row=ri,
                        col=ci,
                    )
                    _empty_cell(fig, ri, ci)
                    continue
                xr, yr = _xy_range_from_points(
                    [data.x_dpad_off, data.x_dpad_on],
                    [data.y_dpad_off, data.y_dpad_on],
                )
                _add_dual_kde_heatmaps(
                    fig,
                    ri,
                    ci,
                    data.x_dpad_off,
                    data.y_dpad_off,
                    data.x_dpad_on,
                    data.y_dpad_on,
                    lp_spec.kde_grid,
                    p_lo,
                    p_hi,
                    False,
                    False,
                )
                v1, v2 = data.pca_variance_ratio
                if v1 > 1e-9 or v2 > 1e-9:
                    fig.add_annotation(
                        text=f"PC1: {v1:.0%} \u00b7 PC2: {v2:.0%}",
                        xref="x domain",
                        yref="y domain",
                        x=0.98,
                        y=0.02,
                        xanchor="right",
                        yanchor="bottom",
                        showarrow=False,
                        font=dict(size=FONT_SIZE_ANNOTATION - 1, color=fg, family=FONT_FAMILY),
                        row=ri,
                        col=ci,
                    )
                fig.update_xaxes(range=list(xr), row=ri, col=ci)
                fig.update_yaxes(range=list(yr), row=ri, col=ci)
                if ci == 1:
                    fig.update_yaxes(title_text="x\u2082", row=ri, col=ci)
                if ri == n_rows:
                    fig.update_xaxes(title_text="x\u2081", row=ri, col=ci)

    for ri in range(1, n_rows + 1):
        for ci in range(1, max_cols + 1):
            fig.update_xaxes(
                showticklabels=(ri == n_p or ri == n_rows),
                gridcolor=grid,
                zeroline=False,
                showline=True, linecolor=fg, linewidth=1,
                tickfont=dict(size=FONT_SIZE_TICK),
                row=ri,
                col=ci,
            )
            fig.update_yaxes(
                showticklabels=(ci == 1),
                gridcolor=grid,
                zeroline=False,
                showline=True, linecolor=fg, linewidth=1,
                tickfont=dict(size=FONT_SIZE_TICK),
                row=ri,
                col=ci,
            )

    apply_thesis_style(
        fig, theme,
        height=max(240 * n_rows + 80, 600),
        margin=dict(l=132, r=48, t=56, b=64),
        show_legend=False,
        hovermode="closest",
    )

    fig.write_image(str(OUT / 'fig_059_latent_phase.png'), width=1200, height=800, scale=2)
    fig.show()
    print(
        "Fig 59 — Latent phase-space KDE (x1,x2) for PSID (top 2 rows) and DPAD (bottom 2 rows). "
        "Participants: PDI1 (S2, S4), PDI4 (S2, S3). "
        "Blue = DBS-OFF density, green = DBS-ON density, computed on test-split latent trajectories. "
        f"PSID variant: {TRIPLET_PDI1_S2.psid_variant.rsplit('_',5)[0]}... (2026-04-08 200 Hz narrow-band). "
        "DPAD: retraining at 200 Hz — rows show placeholder until runs land."
    )

# %% [markdown]
# ## Fig 60: Classification vs dimensionality heatmap (grid search ablation)

# %%
import re
import pickle as _pkl

import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from notebooks.thesis_style import (
    COLOR_CHANCE,
    FONT_FAMILY,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
)
from dashboard.thesis.plot_config import SELECTED_CONFIG, RESULTS_ROOT


def _load_classification_scores() -> pl.DataFrame:
    rows = []
    cls_root = RESULTS_ROOT / "classification"
    for source_dir in ["psid_grid_search_narrow_band", "psid_ablation_narrow_band"]:
        base = cls_root / source_dir
        if not base.exists():
            continue
        for pkl_path in base.rglob("LDA_*_prediction.pkl"):
            feat = pkl_path.stem.replace("LDA_", "").replace("_prediction", "")
            run_dir = pkl_path.parent.parent
            config_path = Path("classification") / "setups" / f"{run_dir.name}.yaml"
            if not config_path.exists():
                continue
            text = config_path.read_text()
            m_nx = re.search(r"nx:\s*(\d+)", text)
            m_n1 = re.search(r"n1:\s*(\d+)", text)
            if not m_nx or not m_n1:
                continue
            nx, n1 = int(m_nx.group(1)), int(m_n1.group(1))
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

df = _load_classification_scores()
features = ["Xp", "Xp_1", "Xp_2"]
sessions = _CLASSIFICATION_SESSIONS
nrows, ncols = len(sessions), len(features)

subplot_titles = [
    f"{label}  \u2014  {feat}" for label in [s[2] for s in sessions] for feat in features
]
fig = make_subplots(
    rows=nrows, cols=ncols,
    vertical_spacing=0.10, horizontal_spacing=0.06,
    subplot_titles=subplot_titles,
)

all_nx = sorted(df["nx"].unique().to_list()) if not df.is_empty() else [1, 2, 5, 60, 65, 70, 75, 80]
all_n1 = sorted(df["n1"].unique().to_list()) if not df.is_empty() else [1, 2, 5, 6, 8, 10, 12]

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

fig.write_image(str(OUT / 'fig_060_classification_dimensionality.png'), width=1200, height=600, scale=2)
fig.show()
print(
    "Fig 60 — PSID classification vs dimensionality ablation heatmap. "
    "Cell color = best LDA balanced-accuracy CV score across (nx, n1) grid; "
    "red outlined cell = selected config from SELECTED_CONFIG. "
    "Rows: PDI1_S2, PDI1_S4, PDI4_S2, PDI4_S3 (test splits). "
    "Cols: Xp (full state), Xp_1 (behav-rel), Xp_2 (behav-irrel). "
    "Data: results/classification/{psid_grid_search_narrow_band, psid_ablation_narrow_band}."
)

# %% [markdown]
# ## Fig 61: PSID Cy importance heatmap

# %%
import logging
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from notebooks.thesis_style import (
    COLOR_BETA_BORDER,
    FONT_FAMILY,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.psid_cy_importance import compute_cy_signed_heatmap

logger = logging.getLogger(__name__)

_DIVERGING_BWR_CY = [
    [0.0, "rgb(24, 95, 165)"],
    [0.5, "rgb(250, 250, 250)"],
    [1.0, "rgb(220, 50, 32)"],
]
_X_LABELS = ["ECoG_1", "ECoG_2", "ECoG_3", "ECoG_4"]
_X_IDX = list(range(4))


def _empty_placeholder_cy(fig: Figure, row: int, col: int) -> None:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(opacity=0), showlegend=False, hoverinfo="skip",
        ),
        row=row, col=col,
    )
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, row=row, col=col)


for cy_spec in THESIS_PSID_CY_IMPORTANCE:
    rows = list(cy_spec.rows)
    if not rows:
        raise ValueError("ThesisPsidCyImportanceSpec.rows is empty")

    max_cols = max(len(r.panels) for r in rows)
    nrows = len(rows)
    theme = cy_spec.theme
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    subplot_titles_cy: List[str | None] = []
    for r in rows:
        for c in range(max_cols):
            if c < len(r.panels):
                subplot_titles_cy.append(r.panels[c].session_label)
            else:
                subplot_titles_cy.append("")

    fig = make_subplots(
        rows=nrows,
        cols=max_cols,
        subplot_titles=subplot_titles_cy or None,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
        row_titles=[rr.participant_label for rr in rows],
    )

    showscale_done = False

    for ri, row_spec in enumerate(rows, start=1):
        for ci in range(1, max_cols + 1):
            if ci > len(row_spec.panels):
                _empty_placeholder_cy(fig, ri, ci)
                continue

            panel = row_spec.panels[ci - 1]
            try:
                z, _n1, _ch, layout = compute_cy_signed_heatmap(
                    results_root,
                    panel.psid_variant,
                    panel.psid_run_ts,
                    cy_spec.split,
                )
            except Exception as e:
                logger.warning("Panel failed: %s: %s", panel, e)
                fig.add_annotation(
                    text=f"Error: {e}",
                    x=0.5, y=0.5,
                    xref="x domain", yref="y domain",
                    showarrow=False,
                    font=dict(size=FONT_SIZE_ANNOTATION, color=fg, family=FONT_FAMILY),
                    row=ri, col=ci,
                )
                continue

            z = np.asarray(z, dtype=float)
            max_display_dims = 8
            if z.shape[0] > max_display_dims:
                row_norms = np.linalg.norm(z, axis=1)
                top_idx = np.argsort(row_norms)[::-1][:max_display_dims]
                top_idx = np.sort(top_idx)
                z = z[top_idx]
                y_labels = [f"dim {i}" for i in top_idx]
            else:
                y_labels = [f"dim {i}" for i in range(z.shape[0])]
            n1_dims = z.shape[0]
            hm_kw = dict(
                z=z,
                x=_X_IDX,
                y=y_labels,
                colorscale=_DIVERGING_BWR_CY,
                zmin=-1.0,
                zmax=1.0,
                showscale=not showscale_done,
                hovertemplate="Latent %{y}<br>%{x}<br>|Cy| = %{z:.3f}<extra></extra>",
                xgap=2,
                ygap=2,
            )
            if not showscale_done:
                hm_kw["colorbar"] = dict(
                    title=dict(text="Norm. |Cy|", side="right", font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY)),
                    len=0.55,
                    thickness=14,
                    tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                )
            hm = go.Heatmap(**hm_kw)
            fig.add_trace(hm, row=ri, col=ci)
            showscale_done = True

            fig.update_xaxes(
                tickmode="array",
                tickvals=_X_IDX,
                ticktext=_X_LABELS,
                tickangle=-30,
                tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                row=ri, col=ci,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(n1_dims)),
                ticktext=y_labels,
                tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                title_text="Behav. rel. dim" if ci == 1 else "",
                title_font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
                row=ri, col=ci,
            )
            if ci > 1:
                fig.update_yaxes(showticklabels=False, row=ri, col=ci)

    apply_thesis_style(
        fig,
        theme,
        height=max(280 * nrows + 80, 500),
        margin=dict(l=100, r=110, t=56, b=96),
        show_legend=False,
    )

    for ri in range(1, nrows + 1):
        for ci in range(1, max_cols + 1):
            fig.update_xaxes(zeroline=False, showgrid=False, row=ri, col=ci)
            fig.update_yaxes(zeroline=False, showgrid=False, row=ri, col=ci)

    fig.write_image(str(OUT / 'fig_061_cy_importance.png'), width=1200, height=600, scale=2)
    fig.show()
    print(
        "Fig 61 — PSID Cy channel-importance heatmap. Rows = top-8 behav.-rel. latent dims, "
        "cols = ECoG channels (ECoG_1..ECoG_4); cell = signed normalized |Cy| (red +, blue -). "
        "Participants: PDI1 (S2, S4), PDI4 (S2, S3). Split: test. "
        "PSID runs: 2026-04-08 200 Hz narrow-band (psid_behavioral_*_dbs_both_200Hz_narrow_band)."
    )

# %% [markdown]
# ## Fig 62: PSID Cz readout matrix

# %%
import logging
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from notebooks.thesis_style import (
    FONT_FAMILY,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_BASE,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    grid_color,
    paper_colors,
    true_line_color,
)
from dashboard.thesis.loaders import load_split_results
from dashboard.thesis.psid_cy_importance import (
    compute_cz_heatmap,
    load_psid_id_sys,
    resolve_model_path,
)

logger = logging.getLogger(__name__)

_DIVERGING_BWR_CZ = [
    [0.0, "rgb(24, 95, 165)"],
    [0.5, "rgb(250, 250, 250)"],
    [1.0, "rgb(220, 50, 32)"],
]


def _empty_placeholder_cz(fig: Figure, row: int, col: int) -> None:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(opacity=0), showlegend=False, hoverinfo="skip",
        ),
        row=row, col=col,
    )
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, row=row, col=col)


def _load_output_channel_names(
    results_root_path,
    variant: str,
    run_ts: str,
    split: str,
    n_fallback: int,
) -> List[str]:
    try:
        res = load_split_results(results_root_path, variant, run_ts, split)
        if res is not None:
            raw = res.get("output_channels")
            if raw is not None:
                names = [str(x).replace("_", " ") for x in (list(raw) if not isinstance(raw, list) else raw)]
                if names:
                    return names[:n_fallback] if len(names) >= n_fallback else names
    except Exception:
        pass
    return [f"z_{i}" for i in range(n_fallback)]


for cz_spec in THESIS_PSID_CZ_HEATMAP:
    rows = list(cz_spec.rows)
    if not rows:
        raise ValueError("ThesisPsidCzSpec.rows is empty")

    max_cols = max(len(r.panels) for r in rows)
    nrows = len(rows)
    theme = cz_spec.theme
    paper_bg, plot_bg = paper_colors(theme)
    grid = grid_color(theme)
    fg = true_line_color(theme)

    subplot_titles_cz: List[str | None] = []
    for r in rows:
        for c in range(max_cols):
            if c < len(r.panels):
                subplot_titles_cz.append(r.panels[c].session_label)
            else:
                subplot_titles_cz.append("")

    fig = make_subplots(
        rows=nrows,
        cols=max_cols,
        subplot_titles=subplot_titles_cz or None,
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
        row_titles=[rr.participant_label for rr in rows],
    )

    showscale_done = False

    for ri, row_spec in enumerate(rows, start=1):
        for ci in range(1, max_cols + 1):
            if ci > len(row_spec.panels):
                _empty_placeholder_cz(fig, ri, ci)
                continue

            panel = row_spec.panels[ci - 1]
            try:
                model_path = resolve_model_path(results_root, panel.psid_variant, panel.psid_run_ts)
                id_sys = load_psid_id_sys(model_path)
                cz_norm, n1, _ = compute_cz_heatmap(id_sys)
            except Exception as e:
                logger.warning("Cz panel failed: %s: %s", panel, e)
                fig.add_annotation(
                    text=f"Error: {e}",
                    x=0.5, y=0.5,
                    xref="x domain", yref="y domain",
                    showarrow=False,
                    font=dict(size=FONT_SIZE_ANNOTATION, color=fg, family=FONT_FAMILY),
                    row=ri, col=ci,
                )
                continue

            nz = cz_norm.shape[0]
            y_labels = _load_output_channel_names(
                results_root, panel.psid_variant, panel.psid_run_ts, cz_spec.split, nz
            )
            max_display_dims = 8
            if n1 > max_display_dims:
                col_norms = np.linalg.norm(cz_norm, axis=0)
                top_idx = np.argsort(col_norms)[::-1][:max_display_dims]
                top_idx = np.sort(top_idx)
                cz_norm = cz_norm[:, top_idx]
                x_labels = [f"dim {k}" for k in top_idx]
                n1 = max_display_dims
            else:
                x_labels = [f"dim {k}" for k in range(n1)]

            hm_kw = dict(
                z=cz_norm,
                x=list(range(n1)),
                y=y_labels,
                colorscale=_DIVERGING_BWR_CZ,
                zmin=-1.0,
                zmax=1.0,
                showscale=not showscale_done,
                hovertemplate="dim %{x}<br>%{y}<br>Cz = %{z:.3f}<extra></extra>",
                xgap=2,
                ygap=2,
            )
            if not showscale_done:
                hm_kw["colorbar"] = dict(
                    title=dict(text="Norm. Cz", side="right", font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY)),
                    len=0.55,
                    thickness=14,
                    tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                )
            hm = go.Heatmap(**hm_kw)
            fig.add_trace(hm, row=ri, col=ci)
            showscale_done = True

            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(n1)),
                ticktext=x_labels,
                tickangle=-45,
                tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                title_text="Behav. rel. dim" if ri == nrows else "",
                title_font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY),
                automargin=True,
                row=ri, col=ci,
            )
            fig.update_yaxes(
                tickfont=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
                automargin=True,
                row=ri, col=ci,
            )
            if ci > 1:
                fig.update_yaxes(showticklabels=False, row=ri, col=ci)

    apply_thesis_style(
        fig,
        theme,
        height=max(300 * nrows + 80, 500),
        margin=dict(l=100, r=110, t=56, b=96),
        show_legend=False,
    )

    for ri in range(1, nrows + 1):
        for ci in range(1, max_cols + 1):
            fig.update_xaxes(zeroline=False, showgrid=False, row=ri, col=ci)
            fig.update_yaxes(zeroline=False, showgrid=False, row=ri, col=ci)

    fig.write_image(str(OUT / 'fig_062_cz_readout.png'), width=1200, height=600, scale=2)
    fig.show()
    print(
        "Fig 62 — PSID Cz readout matrix: how behav.-rel. latent dims project onto behavior channels "
        "(z_0 = DBS-OFF, z_1 = DBS-ON). Cols = top-8 latent dims (by norm), rows = behavior outputs; "
        "cell = signed normalized Cz (red +, blue -). "
        "Participants: PDI1 (S2, S4), PDI4 (S2, S3). "
        "PSID runs: 2026-04-08 200 Hz narrow-band."
    )

# %% [markdown]
# ## Fig 63: Data efficiency

# %%
import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go

from notebooks.thesis_style import (
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    PARTICIPANT_COLORS,
    ThesisTheme,
    apply_thesis_style,
    grid_color,
    legend_bgcolor,
    paper_colors,
    true_line_color,
)

logger = logging.getLogger(__name__)

_HUNGRINESS_SUBDIR = "data_hungriness"


def _session_display_label(dir_name: str) -> Tuple[str, str]:
    parts = dir_name.split("_")
    for i, p in enumerate(parts):
        if p.startswith("PDI") and len(p) > 3:
            pid = p
            sess = parts[i + 1] if i + 1 < len(parts) else "?"
            return pid, f"{pid}_S{sess}"
    return "?", dir_name[:20]


def _load_hungriness_data(
    hungriness_root: Path,
) -> List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    results = []
    if not hungriness_root.is_dir():
        return results
    for json_path in sorted(hungriness_root.glob("*/data_hungriness_summary.json")):
        dir_name = json_path.parent.name
        pid, sess_label = _session_display_label(dir_name)
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            logger.warning("Could not load %s", json_path)
            continue
        data = sorted(data, key=lambda d: d["n_train_trials"])
        n_arr = np.array([d["n_train_trials"] for d in data], dtype=float)
        rmse_arr = np.array([d.get("rmse_neural_mean", float("nan")) for d in data], dtype=float)
        se_arr = np.array([d.get("rmse_neural_se_mean", float("nan")) for d in data], dtype=float)
        results.append((pid, sess_label, n_arr, rmse_arr, se_arr))
    return results


theme = ThesisTheme.LIGHT
show_se_bands = True
paper_bg, plot_bg = paper_colors(theme)
grid = grid_color(theme)
fg = true_line_color(theme)

hungriness_root = Path(results_root) / _HUNGRINESS_SUBDIR
session_data = _load_hungriness_data(hungriness_root)

if not session_data:
    fig = go.Figure()
    fig.add_annotation(
        text="No data_hungriness results found.",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
else:
    _extra_colors: Dict[str, str] = {
        "PDI1": "#185FA5",
        "PDI2": "#4CAF82",
        "PDI3": "#0F6E56",
        "PDI4": "#2D9659",
    }

    seen_pids: set[str] = set()
    fig = go.Figure()

    for pid, sess_label, n_arr, rmse_arr, se_arr in session_data:
        color = PARTICIPANT_COLORS.get(pid, _extra_colors.get(pid, "#888888"))
        show_leg = pid not in seen_pids
        seen_pids.add(pid)

        if show_se_bands and np.any(np.isfinite(se_arr)):
            upper = rmse_arr + se_arr
            lower = np.clip(rmse_arr - se_arr, 0, None)
            fill_color = color.lstrip("#")
            r, g, b = int(fill_color[0:2], 16), int(fill_color[2:4], 16), int(fill_color[4:6], 16)
            rgba_fill = f"rgba({r},{g},{b},0.12)"
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([n_arr, n_arr[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=rgba_fill,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=n_arr,
                y=rmse_arr,
                mode="lines+markers",
                name=pid,
                legendgroup=pid,
                showlegend=show_leg,
                line=dict(color=color, width=1.6),
                marker=dict(size=4, color=color, opacity=0.8),
                hovertemplate=f"{sess_label}<br>n=%{{x}}<br>RMSE=%{{y:.3f}}<extra></extra>",
            )
        )

    apply_thesis_style(
        fig,
        theme,
        height=520,
        margin=dict(l=72, r=24, t=36, b=80),
        hovermode="closest",
        legend_y=-0.22,
    )
    fig.update_xaxes(
        title=dict(
            text="Training trials (n)",
            font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        ),
        zeroline=False,
    )
    fig.update_yaxes(
        title=dict(
            text="RMSE(z) \u2014 neural",
            font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
        ),
        zeroline=False,
    )

fig.write_image(str(OUT / 'fig_063_data_efficiency.png'), width=1000, height=500, scale=2)
fig.show()
print(
    "Fig 63 — Data efficiency: PSID neural-RMSE vs training trials (n). "
    "One trace per session, colored by participant (PDI1-PDI4); shaded band = +/-1 SE across seeds. "
    "All sessions available under results/data_hungriness/*/data_hungriness_summary.json. "
    "X = n_train_trials, Y = rmse_neural_mean."
)

# %% [markdown]
# ## Fig 64: DPAD training curves

# %%
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from notebooks.thesis_style import (
    COLOR_DPAD,
    FONT_FAMILY,
    FONT_SIZE_BASE,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    grid_color,
    legend_bgcolor,
    paper_colors,
    true_line_color,
)

logger = logging.getLogger(__name__)

_STAGE_LABELS = ["Stage 1\n(model1)", "Stage 2\n(model1_Cy)", "Stage 3\n(model2)", "Stage 4\n(model2_Cz)"]
_STAGE_KEYS = ["model1", "model1_Cy", "model2", "model2_Cz"]
_STAGE_COLORS = ["#185FA5", "#0F6E56", "#993C1D", "#854F0B"]
_STAGE_RULE_COLOR = "rgba(100,100,100,0.3)"

_DPAD_SESSION_GLOBS = [
    "dpad_PDI1_S2_dbs_both_narrow_band",
    "dpad_PDI1_S4_dbs_both_narrow_band",
    "dpad_PDI4_S2_dbs_both_narrow_band",
    "dpad_PDI4_S3_dbs_both_narrow_band",
]
_SESSION_LABELS = ["PDI1 S2", "PDI1 S4", "PDI4 S2", "PDI4 S3"]

_COLOR_PSID = "#185FA5"
_COLOR_DPAD = "#993C1D"


def _load_training_history(results_root_path: Path, dir_name: str) -> Optional[Dict[str, Any]]:
    path = results_root_path / dir_name / "training_history.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


theme = ThesisTheme.LIGHT
paper_bg, plot_bg = paper_colors(theme)
grid = grid_color(theme)
fg = true_line_color(theme)

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=_SESSION_LABELS,
    horizontal_spacing=0.10,
    vertical_spacing=0.16,
    shared_yaxes=False,
)

for idx, (dir_name, sess_label) in enumerate(zip(_DPAD_SESSION_GLOBS, _SESSION_LABELS)):
    row = idx // 2 + 1
    col = idx % 2 + 1
    show_leg = idx == 0

    hist = _load_training_history(results_root, dir_name)
    if hist is None:
        fig.add_annotation(
            text=f"No training_history.json<br>in {dir_name}",
            x=0.5, y=0.5,
            xref="x domain", yref="y domain",
            showarrow=False,
            font=dict(size=FONT_SIZE_TICK, family=FONT_FAMILY, color=fg),
            row=row, col=col,
        )
        continue

    cumulative_offset = 0
    stage_boundaries: List[int] = []
    all_epochs: List[float] = []
    all_loss: List[float] = []
    all_val_loss: List[float] = []

    for stage_key in _STAGE_KEYS:
        stage = hist.get(stage_key)
        if stage is None:
            continue
        epochs = stage.get("epochs", [])
        loss = stage.get("loss", [])
        val_loss = stage.get("val_loss", [])
        if not epochs or not loss:
            continue
        n = min(len(epochs), len(loss))
        cum_eps = [cumulative_offset + e for e in epochs[:n]]
        all_epochs.extend(cum_eps)
        all_loss.extend(loss[:n])
        if val_loss:
            nv = min(len(val_loss), n)
            all_val_loss.extend(val_loss[:nv])
            all_val_loss.extend([float("nan")] * (n - nv))
        else:
            all_val_loss.extend([float("nan")] * n)
        cumulative_offset += max(epochs[:n]) + 1
        stage_boundaries.append(cumulative_offset)

    if not all_epochs:
        continue

    for bi, bx in enumerate(stage_boundaries[:-1]):
        fig.add_vline(
            x=bx,
            line=dict(color=_STAGE_RULE_COLOR, width=0.8, dash="dash"),
            row=row, col=col,
        )
        _stage_short_names = ["Init", "Cy", "State", "Cz"]
        stage_label = _stage_short_names[bi + 1] if bi + 1 < len(_stage_short_names) else f"S{bi + 2}"
        fig.add_annotation(
            x=bx,
            y=1.02,
            xref="x",
            yref="y domain",
            text=stage_label,
            showarrow=False,
            font=dict(size=FONT_SIZE_TICK - 1, family=FONT_FAMILY, color=fg),
            xanchor="center",
            row=row, col=col,
        )

    fig.add_trace(
        go.Scatter(
            x=all_epochs,
            y=all_loss,
            mode="lines",
            name="Train loss",
            legendgroup="train",
            showlegend=show_leg,
            line=dict(color=_COLOR_DPAD, width=2.8),
            connectgaps=False,
        ),
        row=row, col=col,
    )
    if any(np.isfinite(v) for v in all_val_loss):
        fig.add_trace(
            go.Scatter(
                x=all_epochs,
                y=all_val_loss,
                mode="lines",
                name="Val loss",
                legendgroup="val",
                showlegend=show_leg,
                line=dict(color=_COLOR_PSID, width=2.4, dash="dot"),
                connectgaps=False,
            ),
            row=row, col=col,
        )

for r in range(1, 3):
    for c in range(1, 3):
        fig.update_xaxes(
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showline=True,
            linecolor=fg,
            linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
            title_text="Epoch" if r == 2 else "",
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            row=r, col=c,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=grid,
            zeroline=False,
            showline=True,
            linecolor=fg,
            linewidth=1,
            tickfont=dict(size=FONT_SIZE_TICK),
            title_text="Loss" if c == 1 else "",
            title_font=dict(size=FONT_SIZE_LABEL, family=FONT_FAMILY),
            row=r, col=c,
        )

stage_labels_flat = ["Init (model1)", "Cy (model1_Cy)", "State (model2)", "Cz (model2_Cz)"]
annotation_text = "  |  ".join(stage_labels_flat)
fig.add_annotation(
    x=0.5, y=-0.22,
    xref="paper", yref="paper",
    text=annotation_text,
    showarrow=False,
    font=dict(size=FONT_SIZE_TICK - 1, family=FONT_FAMILY, color=fg),
    xanchor="center",
)

fig.update_annotations(font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK, color=fg))

apply_thesis_style(fig, theme, height=720, margin=dict(l=72, r=24, t=72, b=140), legend_y=-0.14)

fig.write_image(str(OUT / 'fig_064_dpad_training.png'), width=1200, height=600, scale=2)
fig.show()
print(
    "Fig 64 — DPAD 4-stage training curves (train loss = red, val loss = blue dotted) "
    "concatenated across stages: Init (model1), Cy (model1_Cy), State (model2), Cz (model2_Cz). "
    "Panels: PDI1 S2, PDI1 S4, PDI4 S2, PDI4 S3. "
    "Data: results/dpad_{PDI*_S*}_dbs_both_narrow_band/training_history.json. "
    "Note: sessions may show 'No training_history.json' if DPAD retraining at 200 Hz is incomplete."
)

# %%
n = len(list(OUT.glob('*.png')))
print(f'Section 6 total: {n} figures')
