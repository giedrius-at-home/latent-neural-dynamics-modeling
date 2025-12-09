"""RSA visualization subtab for Model Predictions."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict

from utils.rsa import compute_rdm_from_trials, rsa_multiple_comparisons


def render_rsa_subtab(split_res: Dict, results_dir):
    st.subheader("Representational Similarity Analysis (RSA)")

    Xp = split_res.get("Xp")
    Yp = split_res.get("Yp")
    Zp = split_res.get("Zp")

    if Xp is None or len(Xp) == 0:
        st.warning("No latent state predictions (Xp) found")
        return

    st.markdown("---")

    with st.spinner("Computing RDMs..."):
        rdms = {}

        rdms["Latent (Xp)"] = compute_rdm_from_trials(Xp)

        if Yp is not None and len(Yp) > 0:
            rdms["Neural (Yp)"] = compute_rdm_from_trials(Yp)

        if Zp is not None and len(Zp) > 0:
            rdms["Behavior (Zp)"] = compute_rdm_from_trials(Zp)

    if len(rdms) < 2:
        st.warning("Need at least 2 representational spaces to perform RSA")
        return

    st.markdown("### Representational Dissimilarity Matrices")

    n_rdms = len(rdms)
    cols = st.columns(n_rdms)

    for idx, (name, rdm) in enumerate(rdms.items()):
        with cols[idx]:
            fig = go.Figure(
                data=go.Heatmap(
                    z=rdm,
                    colorscale="Viridis",
                    colorbar=dict(title="Dissimilarity"),
                )
            )
            fig.update_layout(
                title=f"{name} RDM",
                xaxis_title="Trial",
                yaxis_title="Trial",
                height=400,
                width=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### RSA Correlations (Pearson)")

    with st.spinner("Computing RSA correlations..."):
        results = rsa_multiple_comparisons(rdms)

    df_results = pd.DataFrame(results)
    df_results = df_results[["comparison", "correlation", "p_original", "p_corrected"]]
    df_results.columns = ["Comparison", "Pearson r", "p-value", "p-value (Bonferroni)"]

    df_results["Pearson r"] = df_results["Pearson r"].apply(lambda x: f"{x:.4f}")
    df_results["p-value"] = df_results["p-value"].apply(lambda x: f"{x:.4e}")
    df_results["p-value (Bonferroni)"] = df_results["p-value (Bonferroni)"].apply(
        lambda x: f"{x:.4e}"
    )

    st.dataframe(df_results, use_container_width=True, hide_index=True)
