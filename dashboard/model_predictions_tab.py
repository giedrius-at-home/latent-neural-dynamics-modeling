import streamlit as st
from pathlib import Path

from utils.config import get_config
from dashboard.subtabs import (
    list_variants,
    list_run_timestamps,
    config_for_variant,
    check_precomputed_results,
    load_precomputed_results,
    compute_predictions_selective,
    render_predictions_tab,
    render_forecasting_tab,
    render_latent_states_tab,
    render_cross_correlation_analysis_tab,
)
from dashboard.subtabs.rsa_analysis import render_rsa_subtab


def model_predictions_tab(project_root):
    st.header("Model Predictions")

    RESULTS_ROOT = project_root / "results"

    variants = list_variants(RESULTS_ROOT)
    if len(variants) == 0:
        st.info("No result variants found under results/.")
        return

    variant = st.selectbox("Model variant", options=variants, key="pred_variant")
    variant_dir = RESULTS_ROOT / variant
    runs = list_run_timestamps(variant_dir)

    if len(runs) == 0:
        st.info("No runs found for this variant yet. Train a model first.")
        return

    run_ts = st.selectbox("Run timestamp", options=runs, key="pred_run")
    cfg_path = config_for_variant(project_root, variant)

    if cfg_path is None:
        st.error(
            f"Config not found for variant '{variant}'. Expected at training/setups/{variant}.yaml"
        )
        return

    available_results = check_precomputed_results(variant_dir, run_ts)

    st.markdown("### Select Splits to Visualize")
    selected_splits = st.multiselect(
        "Choose splits",
        options=["train", "val", "test"],
        default=["val"] if available_results.get("val") else [],
        key="selected_splits",
    )

    button_label = "Load/Compute Results" if selected_splits else "Select splits first"
    if st.button(button_label, key="btn_load_results", disabled=not selected_splits):
        st.session_state["predictions_key"] = (
            str(cfg_path),
            run_ts,
            tuple(selected_splits),
        )

    @st.cache_resource(show_spinner=True)
    def _cached_predictions_selective(
        config_path: str, run_timestamp: str, splits_to_compute: tuple
    ):
        return compute_predictions_selective(
            config_path, run_timestamp, splits_to_compute
        )

    pred_key = st.session_state.get("predictions_key")
    if pred_key and pred_key[0] == str(cfg_path) and pred_key[1] == run_ts:
        requested_splits = pred_key[2] if len(pred_key) > 2 else selected_splits

        pred_results = {}

        with st.spinner("Loading/computing results..."):
            for split in requested_splits:
                if available_results.get(split):
                    with st.spinner(f"Loading pre-computed {split} results..."):
                        loaded = load_precomputed_results(variant_dir, run_ts, split)
                        if loaded:
                            pred_results[split] = loaded
                        else:
                            st.error(f"Failed to load {split} results")

            cfg = get_config(str(cfg_path))
            neural_input = (
                getattr(cfg.data.channels, "neural_input", None)
                or getattr(cfg.data.channels, "input", None)
                or []
            )
            behavioral_input = (
                getattr(cfg.data.channels, "behavioral_input", None) or []
            )
            input_chans = (
                list(neural_input) + list(behavioral_input)
                if behavioral_input
                else list(neural_input)
            )
            output_chans = cfg.data.channels.output

            for split, res in pred_results.items():
                if not res.get("input_channels") and input_chans:
                    res["input_channels"] = input_chans
                if not res.get("output_channels") and output_chans:
                    res["output_channels"] = output_chans

            splits_to_compute = tuple(
                [s for s in requested_splits if not available_results.get(s)]
            )
            if splits_to_compute:
                with st.spinner(f"Computing {', '.join(splits_to_compute)} results..."):
                    try:
                        computed = _cached_predictions_selective(
                            str(cfg_path), run_ts, splits_to_compute
                        )
                        pred_results.update(computed)
                    except Exception as e:
                        st.error(f"Computation failed: {e}")

        if pred_results:
            split = st.selectbox(
                "Split to visualize",
                options=list(pred_results.keys()),
                key="pred_split",
            )
            split_res = pred_results.get(split)

            if not split_res:
                st.info("No results for selected split.")
                return

            Y_true = split_res["Y"]
            n_trials = len(Y_true)
            trial_indices = list(range(n_trials))
            trial_idx = st.selectbox("Trial", options=trial_indices, key="pred_trial")

            pid_list = split_res.get("participant_id", [])
            ses_list = split_res.get("session", [])
            blk_list = split_res.get("block", [])
            tri_list = split_res.get("trial", [])

            hdr_pid = (
                pid_list[trial_idx]
                if pid_list
                else st.session_state.get("participant_id")
            )
            hdr_ses = (
                ses_list[trial_idx] if ses_list else st.session_state.get("session")
            )
            hdr_blk = blk_list[trial_idx] if blk_list else st.session_state.get("block")
            hdr_tri = tri_list[trial_idx] if tri_list else trial_idx

            st.subheader(
                f"Participant {hdr_pid} | Session {hdr_ses} | Block {hdr_blk} | Trial {hdr_tri}"
            )

            (
                performance_tab,
                lag_analysis_tab,
                predictions_subtab,
                forecasting_subtab,
                latent_states_subtab,
                rsa_subtab,
            ) = st.tabs(
                [
                    "Global Performance",
                    "Lag Analysis",
                    "Predictions",
                    "Forecasting",
                    "Latent States",
                    "RSA Analysis",
                ]
            )

            st.session_state["config_path"] = cfg_path
            st.session_state["run_timestamp"] = run_ts

            with performance_tab:
                from dashboard.subtabs import render_cross_trial_performance_tab
                render_cross_trial_performance_tab(split_res)

            with lag_analysis_tab:
                cfg = get_config(str(cfg_path))
                fs = getattr(cfg.data, "sampling_frequency", 60.0)
                render_cross_correlation_analysis_tab(split_res, sampling_freq=fs)

            with predictions_subtab:
                render_predictions_tab(split_res, trial_idx, cfg_path)

            with forecasting_subtab:
                render_forecasting_tab(
                    split_res, trial_idx, cfg_path, run_ts, Y_true, split_res["Yp"]
                )

            with latent_states_subtab:
                render_latent_states_tab(split_res, trial_idx)

            with rsa_subtab:
                render_rsa_subtab(split_res, variant_dir)
