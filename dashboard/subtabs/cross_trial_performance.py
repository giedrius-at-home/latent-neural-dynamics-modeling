import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any


def render_cross_trial_performance_tab(split_res: Dict[str, Any]):
    st.subheader("Performance Overview Across All Trials")

    pearson_y = split_res.get("pearson_per_channel", [])
    pearson_z = split_res.get("pearson_per_channel_Z", [])

    y_chan_names = split_res.get("input_channels", [])
    z_chan_names = split_res.get("output_channels", [])

    if not pearson_y and not pearson_z:
        st.info("No Pearson correlation results available for this split.")
        return

    neural_available = False
    if pearson_y and len(pearson_y) > 0:
        try:
            r_y_arr = np.array(pearson_y)
            if r_y_arr.ndim == 2:
                n_trials, n_chans = r_y_arr.shape
                if not y_chan_names or len(y_chan_names) != n_chans:
                    y_chan_names = [f"Ch{i}" for i in range(n_chans)]

                df_y = pd.DataFrame(r_y_arr, columns=y_chan_names)
                df_y["Trial"] = np.arange(n_trials)
                df_y["Mean_R"] = df_y[y_chan_names].mean(axis=1)
                neural_available = True
        except Exception as e:
            st.error(f"Error processing neural Pearson data: {e}")

    behavioral_available = False
    if pearson_z and len(pearson_z) > 0:
        try:
            r_z_arr = np.array(pearson_z)
            if r_z_arr.ndim == 2:
                n_trials, n_chans_z = r_z_arr.shape
                if not z_chan_names or len(z_chan_names) != n_chans_z:
                    z_chan_names = [f"Z_Ch{i}" for i in range(n_chans_z)]

                df_z = pd.DataFrame(r_z_arr, columns=z_chan_names)
                df_z["Trial"] = np.arange(n_trials)
                df_z["Mean_R"] = df_z[z_chan_names].mean(axis=1)
                behavioral_available = True
        except Exception as e:
            st.error(f"Error processing behavioral Pearson data: {e}")

    if not neural_available and not behavioral_available:
        st.info("Could not format Pearson data for heatmaps.")
        return

    sort_option = st.radio(
        "Sort trials by:",
        options=[
            "Original Order (Trial Index)",
            "Neural Mean Performance",
            "Behavioral Mean Performance",
        ],
        index=0,
        horizontal=True,
    )

    if sort_option == "Neural Mean Performance" and neural_available:
        df_y = df_y.sort_values("Mean_R", ascending=False)
        sort_order = df_y["Trial"].values
        if behavioral_available:
            df_z = df_z.set_index("Trial").loc[sort_order].reset_index()
    elif sort_option == "Behavioral Mean Performance" and behavioral_available:
        df_z = df_z.sort_values("Mean_R", ascending=False)
        sort_order = df_z["Trial"].values
        if neural_available:
            df_y = df_y.set_index("Trial").loc[sort_order].reset_index()
    else:
        pass

    def create_heatmap(df, channels, title, colorscale="Viridis"):
        y_labels = [f"Trial {i}" for i in df["Trial"]]

        fig = go.Figure(
            data=go.Heatmap(
                z=df[channels].values,
                x=channels,
                y=y_labels,
                colorscale=colorscale,
                colorbar=dict(title="Pearson r"),
                zmin=0,
                zmax=1,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Channels",
            yaxis_title="Trials",
            height=max(400, 300 + (len(df) * 20)),
            template="plotly_white",
        )
        return fig

    if neural_available:
        st.markdown("### Neural Prediction Performance (Pearson r)")
        fig_y = create_heatmap(
            df_y, y_chan_names, "Neural Performance Heatmap", "Plasma"
        )
        st.plotly_chart(fig_y, use_container_width=True)

    if behavioral_available:
        st.markdown("### Behavioral Prediction Performance (Pearson r)")
        fig_z = create_heatmap(
            df_z, z_chan_names, "Behavioral Performance Heatmap", "Viridis"
        )
        st.plotly_chart(fig_z, use_container_width=True)
