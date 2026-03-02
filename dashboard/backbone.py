from utils.classes import DotDict
import plotly.graph_objects as go
import numpy as np
from textwrap import wrap
import streamlit as st

PALETTE = DotDict(
    {
        "ink_black": "#0e131f",
        "twilight_indigo": "#38405f",
        "vintage_grape": "#59546c",
        "cool_steel": "#8b939c",
        "strawberry_red": "#ff0035",
    }
)

PLOT_COLOR = DotDict(
    {"stim_on": PALETTE.strawberry_red, "stim_off": PALETTE.vintage_grape}
)

PLOT_STYLE = DotDict(
    {
        "font_family": "Arial, sans-serif",
        "title_size": 14,
        "axis_label_size": 12,
        "tick_label_size": 11,
        "line_width_normal": 1.0,
        "line_width_thick": 2.5,
    }
)

LINE_STYLE = DotDict(
    {
        "primary": "solid",
        "secondary": "dot",
    }
)

PARTICIPANT_COLORS = DotDict(
    {
        "PDI1": DotDict(
            {
                "base": "#1f77b4",
                "dbs_on": "#5da5da",
                "dbs_off": "#0e4d7a",
            }
        ),
        "PDI2": DotDict(
            {
                "base": "#ff7f0e",
                "dbs_on": "#ffb366",
                "dbs_off": "#cc5500",
            }
        ),
        "PDI3": DotDict(
            {
                "base": "#2ca02c",
                "dbs_on": "#5fd35f",
                "dbs_off": "#1a6b1a",
            }
        ),
        "PDI4": DotDict(
            {
                "base": "#9467bd",
                "dbs_on": "#b999d4",
                "dbs_off": "#5c3d7a",
            }
        ),
    }
)


def format_title(parts: list[str], max_line_length: int = 60) -> str:

    title = " ".join(parts)
    return "<br>".join(wrap(title, width=max_line_length))


def format_trial_metadata(
    participant_id, session, block, trial, stim=None, duration=None, fs=None
) -> str:

    parts = [
        f"Participant {participant_id}",
        f"Session {session}",
        f"Block {block}",
        f"Trial {trial}",
    ]

    if stim is not None:
        stim_label = "ON" if stim == "on" else "OFF"
        parts.append(f"DBS {stim_label}")
    if duration is not None:
        parts.append(f"{duration:.2f}s")
    if fs is not None:
        parts.append(f"{fs}Hz")

    return " • ".join(parts)


def update_fig_title(fig, title_parts: list[str], max_line_length: int = 60):
    text = format_title(title_parts, max_line_length)
    fig.update_layout(
        title=dict(
            text=text,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        margin=dict(t=80 + (text.count("<br>") * 20)),
    )
    return fig


def add_relative_time_axis(fig, time_abs, onset_time):
    if time_abs is None or len(time_abs) == 0:
        return fig

    time_abs = np.asarray(time_abs, dtype=float)
    rel_offset = float(onset_time) if onset_time is not None else float(time_abs.min())

    n_ticks = 10
    tickvals = np.linspace(time_abs.min(), time_abs.max(), n_ticks)

    rel_ticktext = [f"{(tv - rel_offset):.1f}" for tv in tickvals]

    abs_ticktext = [f"{tv:.1f}" for tv in tickvals]

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=abs_ticktext,
        title_text="Time (s)",
        title_font=dict(size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family),
        tickfont=dict(size=PLOT_STYLE.tick_label_size),
        showgrid=True,
        gridcolor="rgba(200, 200, 200, 0.4)",
    )

    fig.update_layout(
        xaxis2=dict(
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=tickvals,
            ticktext=rel_ticktext,
            title_text="",
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            range=[float(time_abs.min()), float(time_abs.max())],
            matches=None,
            showgrid=False,
        )
    )

    y_range = fig.layout.yaxis.range if fig.layout.yaxis.range else [0, 1]
    if y_range is None or len(y_range) == 0:
        if fig.data and len(fig.data) > 0:
            all_y = []
            for trace in fig.data:
                if hasattr(trace, "y") and trace.y is not None:
                    all_y.extend([y for y in trace.y if y is not None])
            if all_y:
                y_min, y_max = min(all_y), max(all_y)
                y_range = [y_min, y_max]
            else:
                y_range = [0, 1]
        else:
            y_range = [0, 1]

    return fig


def create_base_time_series_figure(
    time_abs,
    onset_time,
    y_label: str,
    title: str = "",
):

    time_abs = np.asarray(time_abs, dtype=float)
    rel_offset = float(onset_time) if onset_time is not None else float(time_abs.min())

    n_ticks = 10
    tickvals = np.linspace(time_abs.min(), time_abs.max(), n_ticks)

    rel_ticktext = [f"{(tv - rel_offset):.1f}" for tv in tickvals]
    abs_ticktext = [f"{tv:.1f}" for tv in tickvals]

    fig = go.Figure()

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=PLOT_STYLE.title_size, family=PLOT_STYLE.font_family),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=tickvals,
            ticktext=abs_ticktext,
            title=dict(
                text="Time (s)",
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            range=[float(time_abs.min()), float(time_abs.max())],
            showgrid=True,
            gridcolor="#F0F0F0",
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        xaxis2=dict(
            overlaying="x",
            side="top",
            tickmode="array",
            tickvals=tickvals,
            ticktext=rel_ticktext,
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            range=[float(time_abs.min()), float(time_abs.max())],
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            showgrid=True,
            gridcolor="#F0F0F0",
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        template="plotly_white",
        plot_bgcolor="white",
        font=dict(
            family=PLOT_STYLE.font_family,
            size=PLOT_STYLE.tick_label_size,
            color="black",
        ),
        legend=dict(
            font=dict(size=10, family=PLOT_STYLE.font_family),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E5E5E5",
            borderwidth=1,
        ),
        showlegend=True,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    return fig


def add_margin_visualization(
    fig,
    time_abs,
    chunk_margin: float,
    margin_color: str = "rgba(89, 84, 108, 0.1)",
    line_color: str = "#59546c",
):
    if chunk_margin <= 0 or len(time_abs) == 0:
        return fig

    time_min = (
        float(time_abs[0])
        if hasattr(time_abs, "__getitem__")
        else float(time_abs.min())
    )
    time_max = (
        float(time_abs[-1])
        if hasattr(time_abs, "__getitem__")
        else float(time_abs.max())
    )

    margin_end_left = time_min + chunk_margin
    margin_start_right = time_max - chunk_margin

    fig.add_vrect(
        x0=time_min,
        x1=margin_end_left,
        fillcolor=margin_color,
        layer="below",
        line_width=0,
    )
    fig.add_vline(
        x=margin_end_left,
        line_dash="dash",
        line_color=line_color,
        line_width=2,
        annotation_text="Margin End",
        annotation_position="top right",
        annotation_font=dict(size=10, color=line_color),
    )

    fig.add_vrect(
        x0=margin_start_right,
        x1=time_max,
        fillcolor=margin_color,
        layer="below",
        line_width=0,
    )
    fig.add_vline(
        x=margin_start_right,
        line_dash="dash",
        line_color=line_color,
        line_width=2,
        annotation_text="Margin Start",
        annotation_position="top left",
        annotation_font=dict(size=10, color=line_color),
    )

    return fig


def add_caption_below(fig, caption_text: str):
    fig.add_annotation(
        text=caption_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.15,
        showarrow=False,
        font=dict(
            size=11, family=PLOT_STYLE.font_family, color=PALETTE.twilight_indigo
        ),
        xanchor="center",
        yanchor="top",
    )
    current_margin = fig.layout.margin
    fig.update_layout(
        margin=dict(
            l=current_margin.l if current_margin.l else 60,
            r=current_margin.r if current_margin.r else 80,
            t=current_margin.t if current_margin.t else 60,
            b=100,
        )
    )
    return fig


def create_base_psd_heatmap_figure(
    x_label: str = "Time (s)", y_label: str = "Frequency (Hz)"
):
    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text=x_label,
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.4)",
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.4)",
        ),
        template="plotly_white",
        font=dict(
            family=PLOT_STYLE.font_family,
            size=PLOT_STYLE.tick_label_size,
            color="#0e131f",
        ),
        margin=dict(l=60, r=80, t=40, b=60),
    )

    return fig


def create_base_psd_line_figure(
    x_label: str = "Frequency (Hz)", y_label: str = "Power/Frequency (dB/Hz)"
):
    fig = go.Figure()

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text=x_label,
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.4)",
        ),
        yaxis=dict(
            title=dict(
                text=y_label,
                font=dict(
                    size=PLOT_STYLE.axis_label_size, family=PLOT_STYLE.font_family
                ),
            ),
            tickfont=dict(size=PLOT_STYLE.tick_label_size),
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.4)",
        ),
        template="plotly_white",
        font=dict(
            family=PLOT_STYLE.font_family,
            size=PLOT_STYLE.tick_label_size,
            color="#0e131f",
        ),
        legend=dict(
            font=dict(size=PLOT_STYLE.tick_label_size, family=PLOT_STYLE.font_family)
        ),
        showlegend=True,
        margin=dict(l=60, r=80, t=40, b=60),
    )

    return fig


def render_styled_table(df, key: str = None) -> None:
    """Render a DataFrame as a beautifully styled Plotly table matching the dashboard scheme."""

    headerColor = PALETTE.twilight_indigo
    rowEvenColor = "white"
    rowOddColor = "rgba(240, 240, 245, 0.5)"

    # Repeat colors enough times for all rows
    row_colors = []
    for i in range(len(df)):
        row_colors.append(rowEvenColor if i % 2 == 0 else rowOddColor)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df.columns),
                    line_color="rgba(200, 200, 200, 0.5)",
                    fill_color=headerColor,
                    align=["left"] + ["center"] * (len(df.columns) - 1),
                    font=dict(color="white", size=14, family=PLOT_STYLE.font_family),
                    height=35,
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    line_color="rgba(200, 200, 200, 0.5)",
                    fill_color=[row_colors * len(df.columns)],
                    align=["left"] + ["center"] * (len(df.columns) - 1),
                    font=dict(
                        color=PALETTE.ink_black, size=13, family=PLOT_STYLE.font_family
                    ),
                    height=30,
                ),
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=min(600, max(150, len(df) * 30 + 50))
    )

    st.plotly_chart(fig, use_container_width=True, key=key)
