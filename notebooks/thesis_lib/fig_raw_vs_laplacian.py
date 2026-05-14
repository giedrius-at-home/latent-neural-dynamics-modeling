"""
Figure: raw monopolar ECoG vs Laplacian-referenced LFP for one representative
trial.

The thesis argues that Laplacian (spatial) re-referencing of STN LFP channels
removes shared common-mode noise / slow drift and emphasises local field
activity. This helper builds a publication-quality Plotly figure showing,
side-by-side for a single trial in a single session, a few raw monopolar ECoG
cortical channels (one per row in the left column) next to the Laplacian STN
LFP channels that the corresponding PSID model actually predicts (right
column). All channels are displayed in the same narrow frequency band so the
comparison is about referencing, not about spectral content.

In the current narrow-band recordings dataset there are no un-referenced
monopolar LFP columns in the parquet files (only ``LAPLACIAN_*_LFP_*``), so we
cannot build a within-contact raw-vs-Laplacian comparison. Instead we contrast
raw cortical ECoG (``ECOG_N_<band>_raw``) against the Laplacian STN LFP
targets that the laplacian PSID variant uses as output
(``LAPLACIAN_N-M_LFP_<band>_raw``). This matches the user's fallback choice
documented in ``training/setups/psid/laplacian_200Hz/both``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from dashboard.thesis.constants import (
    COLOR_PSID,
    COLOR_TRUE_LIGHT,
    FONT_FAMILY,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_LABEL,
    FONT_SIZE_TICK,
    ThesisTheme,
    apply_thesis_style,
    true_line_color,
)

_DEFAULT_DATA_ROOT = Path(
    "resampled_recordings/participants_at_200Hz_scaled_1e6_narrow_band"
)
_SAMPLING_RATE_HZ = 200.0
_LAPLACIAN_LFP_PREFIXES: Tuple[str, ...] = (
    "LAPLACIAN_14-16_LFP",
    "LAPLACIAN_13-15_LFP",
    "LAPLACIAN_12-14_LFP",
    "LAPLACIAN_11-13_LFP",
    "LAPLACIAN_10-12_LFP",
    "LAPLACIAN_9-11_LFP",
    "LAPLACIAN_8-10_LFP",
)


def _session_dir(data_root: Path, participant: str, session: str) -> Path:
    return data_root / f"participant_id={participant}" / f"session={session}"


def _first_block_parquet(session_dir: Path) -> Path:
    if not session_dir.is_dir():
        raise FileNotFoundError(
            f"No session directory at {session_dir}. "
            "Check participant/session and resampled_recordings root."
        )
    block_dirs = sorted(
        (
            d
            for d in session_dir.iterdir()
            if d.is_dir() and d.name.startswith("block=")
        ),
        key=lambda p: int(p.name.split("=", 1)[1]),
    )
    for block_dir in block_dirs:
        pqs = sorted(block_dir.glob("*.parquet"))
        if pqs:
            return pqs[0]
    raise FileNotFoundError(f"No parquet files under any block in {session_dir}.")


def _laplacian_pair_for(
    raw_prefix: str,
    band: str,
    available_columns: Sequence[str],
) -> str | None:
    """Pick a Laplacian LFP channel for this raw ECoG row.

    The mapping between cortical ECoG contacts and STN Laplacian bipolar pairs
    is not anatomical — we simply display one Laplacian channel per row so the
    reader can visually compare drift / common-mode content across a few
    examples. We cycle through the available ``LAPLACIAN_*_LFP`` prefixes in
    the order declared in the PSID laplacian training yamls (14-16 is the
    primary target, then falling back to neighbouring bipolar pairs).
    """
    col_set = set(available_columns)
    try:
        ecog_idx = int(raw_prefix.split("_", 2)[1]) - 1
    except (IndexError, ValueError):
        ecog_idx = 0
    n = len(_LAPLACIAN_LFP_PREFIXES)
    for offset in range(n):
        lap_prefix = _LAPLACIAN_LFP_PREFIXES[(ecog_idx + offset) % n]
        candidate = f"{lap_prefix}_{band}"
        if candidate in col_set:
            return candidate
    return None


def _extract_trial_series(
    df: pd.DataFrame,
    column: str,
    trial_index: int,
) -> np.ndarray:
    val = df[column].iloc[trial_index]
    arr = np.asarray(val, dtype=float)
    if arr.ndim != 1:
        arr = arr.squeeze()
    return arr


def _window_slice(
    t: np.ndarray,
    window_seconds: float,
) -> slice:
    n = len(t)
    if n <= 1 or window_seconds <= 0:
        return slice(0, n)
    dt = float(np.median(np.diff(t))) if n > 1 else 1.0 / _SAMPLING_RATE_HZ
    if dt <= 0:
        dt = 1.0 / _SAMPLING_RATE_HZ
    max_samples = int(round(window_seconds / dt))
    return slice(0, min(n, max_samples))


def build_raw_vs_laplacian_figure(
    *,
    participant: str,
    session: str,
    data_root: Path = _DEFAULT_DATA_ROOT,
    band: str = "beta_17_22_raw",
    raw_channel_prefixes: Tuple[str, ...] = (
        "ECOG_1",
        "ECOG_2",
        "ECOG_3",
        "ECOG_4",
    ),
    trial_index: int = 0,
    window_seconds: float = 15.0,
    theme: ThesisTheme = ThesisTheme.LIGHT,
) -> Figure:
    """Return a side-by-side raw ECoG vs Laplacian LFP time-series figure.

    Left column: raw monopolar cortical ECoG (``ECOG_N_<band>_raw``).
    Right column: Laplacian-referenced STN LFP targets used by the PSID
    laplacian variant (``LAPLACIAN_N-M_LFP_<band>_raw``). One row per cortical
    ECoG contact in ``raw_channel_prefixes``, all for the same trial and band.
    """
    raw_channel_prefixes = tuple(raw_channel_prefixes)
    if not raw_channel_prefixes:
        raise ValueError("raw_channel_prefixes must contain at least one entry.")

    session_dir = _session_dir(data_root, participant, session)
    parquet_path = _first_block_parquet(session_dir)
    df = pd.read_parquet(parquet_path)

    if trial_index < 0 or trial_index >= len(df):
        raise IndexError(
            f"trial_index={trial_index} out of range for {parquet_path.name} "
            f"(n_trials={len(df)})."
        )

    columns = list(df.columns)
    raw_columns: List[str] = []
    laplacian_columns: List[str | None] = []
    missing_raw: List[str] = []
    for prefix in raw_channel_prefixes:
        raw_name = f"{prefix}_{band}"
        if raw_name not in columns:
            missing_raw.append(raw_name)
            continue
        raw_columns.append(raw_name)
        laplacian_columns.append(_laplacian_pair_for(prefix, band, columns))

    if not raw_columns:
        raise ValueError(
            f"None of the requested raw channels exist in parquet {parquet_path}. "
            f"Missing: {missing_raw}"
        )

    n_rows = len(raw_columns)
    block_val = parquet_path.parent.name.split("=", 1)[1]
    trial_label = (
        int(df["trial"].iloc[trial_index]) if "trial" in df.columns else trial_index
    )

    t_full = _extract_trial_series(df, "time", trial_index)
    t_full = t_full - float(t_full[0]) if len(t_full) else t_full
    sl = _window_slice(t_full, window_seconds)
    t = t_full[sl]

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        shared_xaxes=True,
        horizontal_spacing=0.07,
        vertical_spacing=0.06,
        column_titles=(
            "Raw monopolar ECoG",
            "Laplacian-referenced STN LFP",
        ),
    )

    raw_color = COLOR_TRUE_LIGHT if theme == ThesisTheme.LIGHT else "#D3D1C7"
    lap_color = COLOR_PSID

    for row_i, (raw_col, lap_col) in enumerate(
        zip(raw_columns, laplacian_columns), start=1
    ):
        show_legend = row_i == 1

        raw_y = _extract_trial_series(df, raw_col, trial_index)[sl]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=raw_y,
                mode="lines",
                name="Raw ECoG",
                line=dict(color=raw_color, width=1.4),
                legendgroup="raw",
                showlegend=show_legend,
                hovertemplate="%{y:.2f} at %{x:.2f}s<extra>Raw ECoG</extra>",
            ),
            row=row_i,
            col=1,
        )

        if lap_col is not None:
            lap_y = _extract_trial_series(df, lap_col, trial_index)[sl]
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=lap_y,
                    mode="lines",
                    name="Laplacian LFP",
                    line=dict(color=lap_color, width=1.6),
                    legendgroup="laplacian",
                    showlegend=show_legend,
                    hovertemplate="%{y:.2f} at %{x:.2f}s<extra>Laplacian LFP</extra>",
                ),
                row=row_i,
                col=2,
            )
        else:
            fig.add_annotation(
                text="no matching Laplacian channel",
                xref=f"x{2 * row_i}" if row_i > 1 else "x2",
                yref=f"y{2 * row_i}" if row_i > 1 else "y2",
                x=float(t[len(t) // 2]) if len(t) else 0.0,
                y=0.0,
                showarrow=False,
                font=dict(
                    family=FONT_FAMILY,
                    size=FONT_SIZE_ANNOTATION,
                    color=true_line_color(theme),
                ),
            )

        fig.update_yaxes(
            title_text=raw_col.replace(f"_{band}", ""),
            title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
            row=row_i,
            col=1,
        )
        fig.update_yaxes(
            title_text=(lap_col.replace(f"_{band}", "") if lap_col else ""),
            title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_TICK),
            row=row_i,
            col=2,
        )

    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_LABEL),
        row=n_rows,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(family=FONT_FAMILY, size=FONT_SIZE_LABEL),
        row=n_rows,
        col=2,
    )

    figure_height = max(260, 170 * n_rows + 160)
    apply_thesis_style(
        fig,
        theme,
        height=figure_height,
        margin=dict(l=88, r=40, t=78, b=90),
        hovermode="x unified",
        legend_y=-0.12,
    )

    band_label = band.replace("_raw", "").replace("_", "–") + " Hz"
    title_text = (
        f"{participant} session {session}, block {block_val}, trial {trial_label} "
        f"({band_label})"
    )
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            font=dict(family=FONT_FAMILY, size=FONT_SIZE_LABEL + 1),
        ),
    )
    for ann in fig.layout.annotations:
        if ann.text in ("Raw monopolar ECoG", "Laplacian-referenced STN LFP"):
            ann.font = dict(family=FONT_FAMILY, size=FONT_SIZE_LABEL)

    return fig
