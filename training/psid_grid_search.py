#!/usr/bin/env python3
"""
PSID Grid Search Runner with Integrated Aggregation & Visualization.

Reads a config with grid_search section, runs all parameter combinations
through training, aggregates metrics into a single partitioned Parquet,
and generates interactive HTML visualizations.

Usage:
    python -m training.psid_grid_search --config training/setups/gs_PDI4_S3_part1.yaml
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml
import numpy as np
from scipy import signal


# ---------------------------------------------------------------------------
# Helper / metrics functions (shared with aggregate_grid_search_metrics.py)
# ---------------------------------------------------------------------------


def fisher_z(r: float) -> float:
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def max_cross_correlation(
    true: np.ndarray, pred: np.ndarray, max_lag: int = 30
) -> tuple[float, float]:
    if len(true) < 10 or len(pred) < 10:
        return np.nan, np.nan
    true = (true - np.mean(true)) / (np.std(true) + 1e-8)
    pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)
    max_r, best_lag = -1, 0
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
                best_lag = lag
    return (max_r, float(best_lag)) if max_r > -1 else (np.nan, np.nan)


def compute_mean_coherence(
    true: np.ndarray, pred: np.ndarray, fs: float = 60.0
) -> float:
    if len(true) < 32 or len(pred) < 32:
        return np.nan
    if true.ndim > 1:
        cohs = []
        for i in range(true.shape[1]):
            _, Cxy = signal.coherence(
                true[:, i], pred[:, i], fs=fs, nperseg=min(len(true), 128)
            )
            cohs.append(np.mean(Cxy))
        return np.mean(cohs)
    else:
        _, Cxy = signal.coherence(true, pred, fs=fs, nperseg=min(len(true), 128))
        return np.mean(Cxy)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_combinations(grid_search: dict) -> list[dict]:
    """Generate all parameter combinations from grid_search section."""
    param_names = []
    param_values = []

    for key, val in grid_search.items():
        if isinstance(val, list):
            param_names.append(key)
            param_values.append(val)
        else:
            param_names.append(key)
            param_values.append([val])

    combinations = []
    for combo in itertools.product(*param_values):
        combo_dict = dict(zip(param_names, combo))
        # Skip invalid combos where n1 > nx
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

    base_save_dir = base_config.get("results", {}).get("save_dir", "results")
    config["results"] = {
        "save_dir": f"{base_save_dir}/{run_name}",
        "log_dir": f"{base_save_dir}/{run_name}/logs",
        "checkpoint_dir": "checkpoints",
    }

    return config


# ---------------------------------------------------------------------------
# Aggregation: compute session-level metrics from validation parquets
# ---------------------------------------------------------------------------


def compute_session_metrics(val_dir: Path) -> dict:
    """Compute aggregated session-level metrics from validation results."""
    pearson_trials = []
    xcorr_trials = []
    xcorr_lag_trials = []
    z_true_all, z_pred_all = [], []
    y_true_all, y_pred_all = [], []

    for parquet_file in val_dir.rglob("*.parquet"):
        try:
            df = pl.read_parquet(parquet_file)

            if "pearsonr_per_channel_Z" in df.columns:
                val = df["pearsonr_per_channel_Z"].item()
                if val is not None:
                    valid_ch = [r for r in val if r is not None and not np.isnan(r)]
                    if valid_ch:
                        pearson_trials.append(float(np.mean(valid_ch)))

            # Keep raw signal collection for xcorr and coherence (behavioral Z only as requested)
            if "Z_future_true" in df.columns and "Z_future_pred" in df.columns:
                z_t = df["Z_future_true"].item()
                z_p = df["Z_future_pred"].item()
                if z_t is not None and z_p is not None:
                    z_t_arr = np.array(z_t)
                    z_p_arr = np.array(z_p)
                    if z_t_arr.ndim >= 1 and len(z_t_arr) > 0:
                        z_t_flat = (
                            np.vstack(z_t_arr) if z_t_arr.dtype == object else z_t_arr
                        )
                        z_p_flat = (
                            np.vstack(z_p_arr) if z_p_arr.dtype == object else z_p_arr
                        )
                        z_true_all.append(z_t_flat)
                        z_pred_all.append(z_p_flat)

                        ch_xcorrs, ch_lags = [], []
                        for ch in range(z_t_flat.shape[1]):
                            xc, lag = max_cross_correlation(
                                z_t_flat[:, ch], z_p_flat[:, ch]
                            )
                            if not np.isnan(xc):
                                ch_xcorrs.append(xc)
                                ch_lags.append(lag)
                        if ch_xcorrs:
                            xcorr_trials.append(np.mean(ch_xcorrs))
                            xcorr_lag_trials.append(np.mean(ch_lags))
        except Exception:
            continue

    metrics: dict[str, Any] = {}
    if pearson_trials:
        valid = [r for r in pearson_trials if not np.isnan(r)]
        if valid:
            metrics["pearson_mean"] = float(np.mean(valid))
            metrics["pearson_median"] = float(np.median(valid))
            z_vals = [fisher_z(r) for r in valid]
            metrics["pearson_fisher"] = float(inverse_fisher_z(np.mean(z_vals)))
            metrics["r_squared"] = float(np.mean([r**2 for r in valid]))
            metrics["n_trials"] = len(valid)

    if xcorr_trials:
        valid_xc = [x for x in xcorr_trials if not np.isnan(x)]
        if valid_xc:
            metrics["xcorr_mean"] = float(np.mean(valid_xc))
            metrics["xcorr_median"] = float(np.median(valid_xc))

    if xcorr_lag_trials:
        valid_lags = [lag for lag in xcorr_lag_trials if not np.isnan(lag)]
        if valid_lags:
            fs = 60.0
            metrics["xcorr_lag_mean_ms"] = float((np.mean(valid_lags) / fs) * 1000.0)
            metrics["xcorr_lag_median_ms"] = float(
                (np.median(valid_lags) / fs) * 1000.0
            )

    # Coherence on concatenated Z
    if z_true_all:
        z_true_cat = np.concatenate(z_true_all, axis=0)
        z_pred_cat = np.concatenate(z_pred_all, axis=0)
        coh_z = compute_mean_coherence(z_true_cat, z_pred_cat)
        if not np.isnan(coh_z):
            metrics["coherence_Z_mean"] = float(coh_z)

    # Coherence on concatenated Y
    if y_true_all:
        y_true_cat = np.concatenate(y_true_all, axis=0)
        y_pred_cat = np.concatenate(y_pred_all, axis=0)
        coh_y = compute_mean_coherence(y_true_cat, y_pred_cat)
        if not np.isnan(coh_y):
            metrics["coherence_Y_mean"] = float(coh_y)

    return metrics


def find_latest_val_results(results_dir: Path) -> Path | None:
    """Find the most recent val_results directory."""
    val_dirs = list(results_dir.glob("val_results_*"))
    if not val_dirs:
        return None
    return max(val_dirs, key=lambda p: p.stat().st_mtime)


def extract_config_from_metadata(run_dir: Path) -> dict:
    """Extract model params and channel info from metadata JSON."""
    meta_files = list(run_dir.glob("model_*_metadata.json"))
    if not meta_files:
        return {}
    meta_file = max(meta_files, key=lambda p: p.stat().st_mtime)
    try:
        with open(meta_file, "r") as f:
            meta = json.load(f)
        params = {}
        for key in [
            "nx",
            "n1",
            "alpha_Q",
            "alpha_R",
            "backward_kalman",
            "rescale_states",
            "max_eigenvalue",
            "i",
            "participant_id",
            "session",
            "input_channels",
            "output_channels",
            "r_mean_Y",
            "r_mean_Z",
        ]:
            if key in meta:
                params[key] = meta[key]
        return params
    except Exception:
        return {}


def extract_channels_from_val_results(val_dir: Path) -> dict:
    """Read input_channels and output_channels from the validation parquet files."""
    result: dict[str, Any] = {}
    if val_dir is None:
        return result
    for pq in val_dir.rglob("*.parquet"):
        try:
            df = pl.read_parquet(pq)
            if "input_channels" in df.columns and "input_channels" not in result:
                val = df["input_channels"].head(1).item()
                if val is not None:
                    result["input_channels"] = (
                        val if isinstance(val, list) else list(val)
                    )
            if "output_channels" in df.columns and "output_channels" not in result:
                val = df["output_channels"].head(1).item()
                if val is not None:
                    result["output_channels"] = (
                        val if isinstance(val, list) else list(val)
                    )
            if "input_channels" in result and "output_channels" in result:
                break
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------


def run_training_subprocess(
    config: dict,
    combo: dict,
    run_name: str,
    participant: str,
    session: str,
    timestamp: str,
) -> dict:
    """Worker: run training in a subprocess and return result row with metrics."""
    temp_fd, temp_config_path = tempfile.mkstemp(
        suffix=".yaml", prefix=f"grid_run_{run_name}_"
    )
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
        if success:
            print(f"  [SUCCESS] {run_name}")
        else:
            print(f"  [FAILED] {run_name}")
            if result.stderr:
                err_lines = result.stderr.strip().split("\n")
                print(f"    Error: {err_lines[-1]}")

        # ------------------------------------------------------------------
        # Aggregate metrics from the run's val_results directory
        # ------------------------------------------------------------------
        run_metrics: dict[str, Any] = {}
        run_save_dir = Path(config["results"]["save_dir"])
        if success and run_save_dir.exists():
            val_dir = find_latest_val_results(run_save_dir)
            if val_dir:
                run_metrics = compute_session_metrics(val_dir)

            # Extract channels from metadata, fallback to validation parquets
            meta_info = extract_config_from_metadata(run_save_dir)
            if "input_channels" in meta_info:
                run_metrics["neural_input"] = json.dumps(meta_info["input_channels"])
            if "output_channels" in meta_info:
                run_metrics["behavioral_output"] = json.dumps(
                    meta_info["output_channels"]
                )

            # Fallback: read channels from the val parquets directly
            if (
                "neural_input" not in run_metrics
                or "behavioral_output" not in run_metrics
            ):
                ch_info = extract_channels_from_val_results(val_dir)
                if "input_channels" in ch_info and "neural_input" not in run_metrics:
                    run_metrics["neural_input"] = json.dumps(ch_info["input_channels"])
                if (
                    "output_channels" in ch_info
                    and "behavioral_output" not in run_metrics
                ):
                    run_metrics["behavioral_output"] = json.dumps(
                        ch_info["output_channels"]
                    )

        # Serialise list-valued combo entries (e.g. neural_bands) to strings
        serialised_combo = {}
        for k, v in combo.items():
            if isinstance(v, (list, tuple)):
                serialised_combo[k] = json.dumps(v)
            else:
                serialised_combo[k] = v

        return {
            "participant_id": participant,
            "session": session,
            "timestamp": timestamp,
            **serialised_combo,
            "run_name": run_name,
            "success": success,
            **run_metrics,
        }
    except Exception as e:
        print(f"  [ERROR] {run_name}: {e}")
        serialised_combo = {}
        for k, v in combo.items():
            serialised_combo[k] = json.dumps(v) if isinstance(v, (list, tuple)) else v
        return {
            "participant_id": participant,
            "session": session,
            "timestamp": timestamp,
            **serialised_combo,
            "run_name": run_name,
            "success": False,
        }
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def save_partitioned_parquet(df: pl.DataFrame, output_dir: Path) -> None:
    """
    Save grid search results as a single parquet file.
    participant_id and session are kept as regular columns.
    Appends to existing data if present.
    """
    parquet_path = output_dir / "results.parquet"

    if parquet_path.exists():
        try:
            existing_df = pl.read_parquet(parquet_path)
            df = pl.concat([existing_df, df], how="diagonal")
        except Exception:
            pass  # no readable existing data

    output_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path)

    print(f"\n  Results saved to: {parquet_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PSID Grid Search Runner")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to grid search config"
    )
    parser.add_argument("--participant", type=str, help="Override participant ID")
    parser.add_argument("--session", type=str, help="Override session ID")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print combinations without running"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs")
    parser.add_argument(
        "--workers", "-w", type=int, default=1, help="Number of parallel workers"
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training, only re-aggregate existing results and visualise",
    )
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
        print("\nDry run – combinations to try:")
        for i, combo in enumerate(combinations):
            print(f"  {i + 1}. {combo}")
        return 0

    participant = config["data"].get("participant", "unknown")
    session = str(config["data"].get("session", "unknown"))

    base_name = config["model"].get("name", "psid_grid_search")
    if participant not in base_name or f"s{session}" not in base_name.lower():
        base_name = f"{base_name}_{participant}_s{session}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.get("results", {}).get("save_dir", f"results/{base_name}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Aggregate-only mode: re-scan existing run directories
    # ------------------------------------------------------------------
    if args.aggregate_only:
        print("\n=== Aggregate-only mode ===")
        results = []
        base_save_dir = config.get("results", {}).get("save_dir", "results")
        run_dirs = sorted(Path(base_save_dir).glob(f"{base_name}_run*"))
        print(f"Found {len(run_dirs)} existing run directories.")

        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            val_dir = find_latest_val_results(run_dir)
            if not val_dir:
                continue
            try:
                metrics = compute_session_metrics(val_dir)
                meta_info = extract_config_from_metadata(run_dir)

                entry: dict[str, Any] = {
                    "participant_id": meta_info.get("participant_id", participant),
                    "session": str(meta_info.get("session", session)),
                    "run_name": run_dir.name,
                    "success": True,
                }
                # Add model params from metadata
                for k in [
                    "nx",
                    "n1",
                    "alpha_Q",
                    "alpha_R",
                    "backward_kalman",
                    "rescale_states",
                    "max_eigenvalue",
                    "i",
                ]:
                    if k in meta_info:
                        entry[k] = meta_info[k]

                # Channels: prefer metadata, fallback to val parquets
                if "input_channels" in meta_info:
                    entry["neural_input"] = json.dumps(meta_info["input_channels"])
                if "output_channels" in meta_info:
                    entry["behavioral_output"] = json.dumps(
                        meta_info["output_channels"]
                    )
                if "neural_input" not in entry or "behavioral_output" not in entry:
                    ch_info = extract_channels_from_val_results(val_dir)
                    if "input_channels" in ch_info and "neural_input" not in entry:
                        entry["neural_input"] = json.dumps(ch_info["input_channels"])
                    if (
                        "output_channels" in ch_info
                        and "behavioral_output" not in entry
                    ):
                        entry["behavioral_output"] = json.dumps(
                            ch_info["output_channels"]
                        )

                entry.update(metrics)
                results.append(entry)
            except Exception as e:
                print(f"  Error processing {run_dir.name}: {e}")

        if not results:
            print("No results found to aggregate.")
            return 1

        df = pl.DataFrame(results)
        save_partitioned_parquet(df, output_dir)
        return 0

    # ------------------------------------------------------------------
    # Normal mode: run grid search then aggregate
    # ------------------------------------------------------------------
    results = []

    print(f"\nStarting grid search with {args.workers} worker(s)...")

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
                    timestamp,
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

        # Save partitioned parquet
        save_partitioned_parquet(df, output_dir)

        # Print quick summary
        n_success = df.filter(pl.col("success") == True).height
        n_total = len(results)
        print(f"\n{'=' * 60}")
        print(f"Grid Search Complete: {n_success}/{n_total} runs succeeded")
        if "pearson_mean" in df.columns:
            successful = df.filter(pl.col("success") == True)
            if not successful.is_empty():
                best = successful.sort("pearson_mean", descending=True).head(1)
                print(
                    f"Best Pearson r: {best['pearson_mean'].item():.4f}  (run: {best['run_name'].item()})"
                )
        print(f"Results: {output_dir}")
        print(f"{'=' * 60}")
    else:
        print("No results collected.")

    return 0


if __name__ == "__main__":
    exit(main())
