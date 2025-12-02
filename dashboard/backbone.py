from utils.classes import DotDict
import plotly.graph_objects as go
import numpy as np

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
        "font_family": "sans-serif",
        "title_size": 16,
        "axis_label_size": 14,
        "tick_label_size": 12,
        "line_width_normal": 2,
        "line_width_thick": 3,
    }
)


def format_title(parts: list[str], max_line_length: int = 60) -> str:
    from textwrap import wrap

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

    n_ticks = 5
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

    n_ticks = 5
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
        margin=dict(l=60, r=80, t=100, b=60),
    )

    return fig
