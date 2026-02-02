import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import sys
import polars as pl
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.classification import (
    CLASSIFIERS,
    prepare_epoched_data,
    run_full_classification_analysis,
)
from utils.logger import setup_logger, get_logger
from utils.config import get_config


def _get_first(val):
    if isinstance(val, list):
        return val[0]
    return val


def _get_model_path(project_root, variant_cfg):
    v = _get_first(variant_cfg)
    return project_root / "results" / v.variant / f"model_{v.run_ts}.pkl"


def load_all_splits(
    variant_dir: Path, run_ts: str
) -> Dict[str, Optional[Dict[str, Any]]]:
    from dashboard.subtabs import load_precomputed_results

    splits = {}
    for split_name in ["train", "val", "test"]:
        splits[split_name] = load_precomputed_results(variant_dir, run_ts, split_name)
    return splits


def compute_classification_for_config(
    logger,
    config,
    mode: str,
    feature_source: str,
    variant_dir=None,
    run_ts=None,
    A_on=None,
    A_off=None,
    A_both=None,
    trials=None,
):
    flipped = config.classification.get("flipped", False)

    if flipped:
        X_train, y_train, groups_train, _ = prepare_epoched_data(
            [trials],
            feature_source=feature_source,
            epoch_length_sec=config.classification.epoch_length,
            overlap=config.classification.epoch_overlap,
            fs=config.classification.sampling_freq,
            A_on=A_on,
            A_off=A_off,
            A_both=A_both,
            h=config.classification.get("h"),
            m=config.classification.get("m"),
        )
        X_test, y_test, _, _ = prepare_epoched_data(
            [trials],
            feature_source=feature_source,
            epoch_length_sec=config.classification.epoch_length,
            overlap=config.classification.epoch_overlap,
            fs=config.classification.sampling_freq,
            target_future=True,
            A_on=A_on,
            A_off=A_off,
            A_both=A_both,
            h=config.classification.get("h"),
            m=config.classification.get("m"),
        )
    else:
        splits = load_all_splits(variant_dir, run_ts)
        trainval_list = [
            r for r in [splits.get("train"), splits.get("val")] if r is not None
        ]
        test_list = [splits.get("test")] if splits.get("test") is not None else []

        X_train, y_train, groups_train, _ = prepare_epoched_data(
            trainval_list,
            feature_source=feature_source,
            epoch_length_sec=config.classification.epoch_length,
            overlap=config.classification.epoch_overlap,
            fs=config.classification.sampling_freq,
            mode=mode,
            forecast_horizon_sec=config.classification.get("forecast_horizon"),
        )
        X_test, y_test, _, _ = (
            prepare_epoched_data(
                test_list,
                feature_source=feature_source,
                epoch_length_sec=config.classification.epoch_length,
                overlap=config.classification.epoch_overlap,
                fs=config.classification.sampling_freq,
                mode=mode,
                forecast_horizon_sec=config.classification.get("forecast_horizon"),
            )
            if test_list
            else (None, None, None, None)
        )

    if X_train is None:
        logger.warning(f"No samples for {mode}/{feature_source}. Skipping.")
        return

    for clf_name in CLASSIFIERS:
        results = run_full_classification_analysis(
            clf_name,
            X_train,
            y_train,
            groups_train,
            X_test,
            y_test,
            config,
            logger,
            feature_source=feature_source,
        )

        results_dir = (
            Path(config.results.project_root)
            / "results"
            / (_get_first(config.run.dbs_on).variant if flipped else config.run.variant)
        )
        if flipped:
            h_val = config.classification.get("h", 1.0)
            m_val = config.classification.get("m", 1.0)
            save_dir = results_dir / f"flipped_h{h_val}_m{m_val}"
        else:
            save_dir = results_dir / f"{run_ts}/classification"
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / f"{clf_name}_{feature_source}_{mode}.pkl", "wb") as f:
            pickle.dump(results, f)


def compute(config):
    logger = get_logger()
    project_root = Path(config.results.project_root)
    flipped = config.classification.get("flipped", False)

    if flipped:
        A_on = get_dynamics(project_root, config.run.dbs_on)
        A_off = get_dynamics(project_root, config.run.dbs_off)
        A_both = (
            get_dynamics(project_root, config.run.dbs_both)
            if hasattr(config.run, "dbs_both")
            else None
        )

        variants_to_load = []
        if hasattr(config.run, "dbs_both"):
            variants_to_load.append(config.run.dbs_both)
        else:
            if hasattr(config.run, "dbs_on"):
                variants_to_load.append(config.run.dbs_on)
            if hasattr(config.run, "dbs_off"):
                variants_to_load.append(config.run.dbs_off)

        trials = {
            "Xp": [],
            "stim": [],
            "session": [],
            "block": [],
            "participant_id": [],
            "trial": [],
        }
        trial_idx = 0
        for variant_cfg in variants_to_load:
            v_cfg = _get_first(variant_cfg)
            results_root = project_root / "results" / v_cfg.variant
            
            train_dir = results_root / f"train_results_{v_cfg.run_ts}"
            val_dir = results_root / f"val_results_{v_cfg.run_ts}"

            logger.info(f"Loading Source Data from: {v_cfg.variant}")
            result_files = sorted(list(train_dir.rglob("0.parquet")) + list(val_dir.rglob("0.parquet")))
            for f in tqdm(result_files, desc="Trials", leave=False):
                df = pl.read_parquet(f)
                xp_arr = np.array(df["Xp"].to_list()).squeeze() 
                if xp_arr.ndim == 2 and xp_arr.shape[0] < xp_arr.shape[1]:
                    xp_arr = xp_arr.T 
                trials["Xp"].append(xp_arr)
                trials["stim"].append(df["stim"][0])
                trials["session"].append(0)
                trials["block"].append(trial_idx)
                trials["participant_id"].append("p0")
                trials["trial"].append(trial_idx)
                trial_idx += 1

        h_values = config.classification.get("h", [1.0])
        m_values = config.classification.get("m", [1.0])
        
        if not isinstance(h_values, list):
            h_values = [h_values]
        if not isinstance(m_values, list):
            m_values = [m_values]
        
        all_results = []
        for h_val in h_values:
            for m_val in m_values:
                logger.info(f"Running flipped classification with h={h_val}s, m={m_val}s")
                
                config.classification.h = h_val
                config.classification.m = m_val
                
                compute_classification_for_config(
                    logger,
                    config,
                    "flipped",
                    config.classification.get("prediction_feature_source", "Xp"),
                    A_on=A_on,
                    A_off=A_off,
                    A_both=A_both,
                    trials=trials,
                )
    else:
        verify_test_results(logger, project_root, config)
        for mode in ["prediction", "forecast"]:
            compute_classification_for_config(
                logger,
                config,
                mode,
                config.classification.get(f"{mode}_feature_source", "Xp"),
                variant_dir=project_root / "results" / config.run.variant,
                run_ts=config.run.run_ts,
            )


def get_dynamics(project_root, variant_cfg):
    with open(_get_model_path(project_root, variant_cfg), "rb") as f:
        return pickle.load(f).A


def verify_test_results(logger, project_root, config):
    test_dir = (
        project_root
        / "results"
        / config.run.variant
        / f"test_results_{config.run.run_ts}"
    )
    if not test_dir.exists():
        from training.components.tester import Tester

        tester = Tester(
            get_config(
                str(project_root / "training" / "setups" / f"{config.run.variant}.yaml")
            ),
            run_timestamp=config.run.run_ts,
        )
        tester.run_predictions()
        tester.save_results()


def main():
    parser = argparse.ArgumentParser(description="Unified DBS Classification")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = get_config(args.config)
    setup_logger(config.results.log_dir, name=__file__)
    compute(config)


if __name__ == "__main__":
    main()
