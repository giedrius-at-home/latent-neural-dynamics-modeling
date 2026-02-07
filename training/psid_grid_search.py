#!/usr/bin/env python3
"""
PSID Grid Search Runner

Reads a config with grid_search section and runs all parameter combinations
through fast training, then aggregates metrics into a single CSV.

Usage:
    python -m training.psid_grid_search --config training/setups/psid_grid_search.yaml
"""

import argparse
import itertools
import json
import subprocess
import sys
import tempfile
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import yaml


def fisher_z(r: float) -> float:
    """Fisher z-transform for correlation averaging."""
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    """Inverse Fisher z-transform."""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def max_cross_correlation(true: np.ndarray, pred: np.ndarray, max_lag: int = 30) -> float:
    """Compute max cross-correlation within a lag window."""
    if len(true) < 10 or len(pred) < 10:
        return np.nan

    true = (true - np.mean(true)) / (np.std(true) + 1e-8)
    pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)

    max_r = -1
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            t, p = true[-lag:], pred[:lag]
        elif lag > 0:
            t, p = true[:-lag], pred[lag:]
        else:
            t, p = true, pred

        if len(t) > 10:
            r = np.corrcoef(t, p)[0, 1]
            if not np.isnan(r) and r > max_r:
                max_r = r

    return max_r if max_r > -1 else np.nan


def compute_session_metrics(val_dir: Path) -> dict:
    """Compute aggregated session-level metrics from validation results."""
    pearson_trials = []
    xcorr_trials = []

    for parquet_file in val_dir.rglob("*.parquet"):
        try:
            df = pl.read_parquet(parquet_file)

            # Get behavioral Pearson r
            if "metric_pearson_r_mean_Z" in df.columns:
                val = df["metric_pearson_r_mean_Z"].item()
                if val is not None and not np.isnan(val):
                    pearson_trials.append(val)

            # Compute cross-correlation if data available
            if "Z_future_true" in df.columns and "Z_future_pred" in df.columns:
                z_true = df["Z_future_true"].item()
                z_pred = df["Z_future_pred"].item()
                if z_true is not None and z_pred is not None:
                    z_true = np.array(z_true)
                    z_pred = np.array(z_pred)
                    if z_true.ndim >= 1 and len(z_true) > 0:
                        try:
                            z_true_arr = np.vstack(z_true)
                            z_pred_arr = np.vstack(z_pred)
                            ch_xcorrs = []
                            for ch in range(z_true_arr.shape[1]):
                                xc = max_cross_correlation(z_true_arr[:, ch], z_pred_arr[:, ch])
                                if not np.isnan(xc):
                                    ch_xcorrs.append(xc)
                            if ch_xcorrs:
                                xcorr_trials.append(np.mean(ch_xcorrs))
                        except Exception:
                            pass

        except Exception:
            continue

    metrics = {}

    if pearson_trials:
        valid = [r for r in pearson_trials if not np.isnan(r)]
        if valid:
            metrics["pearson_mean"] = np.mean(valid)
            metrics["pearson_median"] = np.median(valid)
            # Trimmed mean (10%)
            n_trim = max(1, int(0.1 * len(valid)))
            sorted_vals = sorted(valid)
            metrics["pearson_trimmed"] = np.mean(sorted_vals[n_trim:-n_trim]) if len(sorted_vals) > 2 * n_trim else np.mean(valid)
            # Fisher z mean
            z_vals = [fisher_z(r) for r in valid]
            metrics["pearson_fisher"] = inverse_fisher_z(np.mean(z_vals))
            # R squared (approximate)
            metrics["r_squared"] = np.mean(valid) ** 2
            # Stability
            metrics["cv"] = np.std(valid) / (np.abs(np.mean(valid)) + 1e-8)
            # Above threshold
            metrics["pct_above_zero"] = 100 * np.mean([r > 0 for r in valid])
            metrics["pct_above_03"] = 100 * np.mean([r > 0.3 for r in valid])
            metrics["n_trials"] = len(valid)

    if xcorr_trials:
        valid_xc = [x for x in xcorr_trials if not np.isnan(x)]
        if valid_xc:
            metrics["xcorr_mean"] = np.mean(valid_xc)
            metrics["xcorr_median"] = np.median(valid_xc)

    return metrics


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_combinations(grid_search: dict) -> list[dict]:
    """Generate all parameter combinations from grid_search section."""
    # Separate grid params from special list params
    param_names = []
    param_values = []

    for key, val in grid_search.items():
        if isinstance(val, list):
            param_names.append(key)
            param_values.append(val)
        else:
            # Single value, still include
            param_names.append(key)
            param_values.append([val])

    combinations = []
    for combo in itertools.product(*param_values):
        combo_dict = dict(zip(param_names, combo))

        if "nx" in combo_dict and "n1" in combo_dict:
            if combo_dict["n1"] > combo_dict["nx"]:
                continue

        combinations.append(combo_dict)

    return combinations


def create_run_config(base_config: dict, combo: dict, run_name: str) -> dict:
    """Create a config for a specific run by merging base config with grid params."""
    config = yaml.safe_load(yaml.dump(base_config))  # deep copy

    for key, val in combo.items():
        if key == "neural_bands":
            config["data"]["channels"]["neural_input"] = val
        elif key == "behavioral_outputs":
            config["data"]["channels"]["output"] = val
        else:
            config["model"][key] = val

    config["model"]["name"] = run_name
    config["model"]["fast"] = False 
    config["model"]["reuse_splits"] = True

    if "grid_search" in config["model"]:
        del config["model"]["grid_search"]

    config["results"] = {
        "save_dir": f"results/{run_name}",
        "log_dir": f"results/{run_name}/logs",
        "checkpoint_dir": "checkpoints",
    }

    return config


def run_training_subprocess(config: dict, combo: dict, run_name: str, participant: str, session: str, timestamp: str) -> dict:
    """Worker function to run training in a subprocess."""
    temp_fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix=f"grid_run_{run_name}_")
    try:
        with os.fdopen(temp_fd, "w") as f:
            yaml.dump(config, f)

        print(f"  Starting Run: {run_name} | Params: {combo}")
        result = subprocess.run(
            [sys.executable, "-m", "training.train", "--config", temp_config_path],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        success = result.returncode == 0
        metrics = {}
        if success:
            run_results_dir = Path(f"results/{run_name}")
            val_dir = find_latest_val_results(run_results_dir)
            if val_dir:
                metrics = compute_session_metrics(val_dir)
            print(f"  [SUCCESS] {run_name}")
        else:
            print(f"  [FAILED] {run_name}")

        return {
            "participant_id": participant,
            "session": session,
            "timestamp": timestamp,
            **combo,
            **metrics,
            "run_name": run_name,
            "success": success,
        }
    except Exception as e:
        print(f"  [ERROR] {run_name}: {e}")
        return {
            "participant_id": participant,
            "session": session,
            "timestamp": timestamp,
            **combo,
            "run_name": run_name,
            "success": False,
        }
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def find_latest_val_results(results_dir: Path) -> Path | None:
    """Find the most recent val_results directory."""
    val_dirs = list(results_dir.glob("val_results_*"))
    if not val_dirs:
        return None
    return max(val_dirs, key=lambda p: p.stat().st_mtime)

import os

def main():
    parser = argparse.ArgumentParser(description="PSID Grid Search Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to grid search config")
    parser.add_argument("--participant", type=str, help="Override participant ID")
    parser.add_argument("--session", type=str, help="Override session ID")
    parser.add_argument("--dry-run", action="store_true", help="Print combinations without running")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.participant:
        config["data"]["participant"] = args.participant
    if args.session:
        config["data"]["session"] = args.session

    if "grid_search" not in config.get("model", {}):
        print("Error: Config must have model.grid_search section")
        return 1

    grid_search = config["model"]["grid_search"]
    combinations = generate_combinations(grid_search)

    print(f"Generated {len(combinations)} parameter combinations")

    if args.limit:
        combinations = combinations[: args.limit]
        print(f"Limited to {len(combinations)} runs")

    if args.dry_run:
        print("\nDry run - combinations to try:")
        for i, combo in enumerate(combinations):
            print(f"  {i + 1}. {combo}")
        return 0

    participant = config["data"].get("participant", "unknown")
    session = str(config["data"].get("session", "unknown"))

    base_name = config["model"].get("name", "psid_grid_search")
    base_name = f"{base_name}_{participant}_s{session}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/{base_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []


    
    print(f"\nStarting grid search with {args.workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i, combo in enumerate(combinations):
            run_name = f"{base_name}_run{i:03d}_{timestamp}"
            run_config = create_run_config(config, combo, run_name)
            
            futures.append(
                executor.submit(
                    run_training_subprocess, 
                    run_config, 
                    combo, 
                    run_name, 
                    participant, 
                    session, 
                    timestamp
                )
            )
            
        for future in concurrent.futures.as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Unhandled exception in worker: {e}")

    if results:
        df = pl.DataFrame(results)
        parquet_path = output_dir / "results.parquet"
        
        if parquet_path.exists():
            existing_df = pl.read_parquet(parquet_path)
            combined_df = pl.concat([existing_df, df], how="diagonal")
            combined_df.write_parquet(
                parquet_path, 
                use_pyarrow=True,
                partition_by=["participant_id", "session"]
            )
        else:
            df.write_parquet(
                parquet_path,
                use_pyarrow=True,
                partition_by=["participant_id", "session"]
            )
            
        print(f"\nResults saved to partitioned parquet: {parquet_path}")

    return 0


if __name__ == "__main__":
    exit(main())
