#!/usr/bin/env python3
"""
Grid Search Results Analyzer for PSID

This script automatically aggregates results from a grid search run and identifies
the best parameter combinations based on validation metrics (Pearson correlation).

Usage:
    python training/analyze_grid_search.py [--results-dir RESULTS_DIR] [--top N]

Output:
    - Summary CSV with all runs ranked by performance
    - Top N best parameter combinations printed to console
    - Interactive HTML report with visualizations
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime


def find_all_runs(results_dir: Path) -> List[Dict[str, Any]]:
    """Find all model runs in the results directory."""
    runs = []

    # Find all metadata files
    for metadata_file in results_dir.glob("model_*_metadata.json"):
        timestamp = metadata_file.stem.replace("model_", "").replace("_metadata", "")

        # Load metadata
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Find corresponding validation results
        val_dir = results_dir / f"val_results_{timestamp}"
        if not val_dir.exists():
            continue

        # Load validation metrics from parquet files
        metrics = load_validation_metrics(val_dir)

        run_info = {"timestamp": timestamp, **metadata, **metrics}
        runs.append(run_info)

    return runs


def fisher_z(r: float) -> float:
    """Fisher z-transform for correlation averaging."""
    # Clamp to avoid infinity
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z: float) -> float:
    """Inverse Fisher z-transform."""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def max_cross_correlation(
    true: np.ndarray, pred: np.ndarray, max_lag: int = 30
) -> float:
    """Compute max cross-correlation within a lag window (shape similarity).

    This finds the best alignment between signals and returns the max correlation,
    accounting for potential delay between prediction and ground truth.
    """
    if len(true) < 10 or len(pred) < 10:
        return np.nan

    # Normalize signals
    true = (true - np.mean(true)) / (np.std(true) + 1e-8)
    pred = (pred - np.mean(pred)) / (np.std(pred) + 1e-8)

    # Compute cross-correlation
    n = len(true)
    max_r = -1

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            t = true[-lag:]
            p = pred[:lag]
        elif lag > 0:
            t = true[:-lag]
            p = pred[lag:]
        else:
            t = true
            p = pred

        if len(t) > 10:
            r = np.corrcoef(t, p)[0, 1]
            if not np.isnan(r) and r > max_r:
                max_r = r

    return max_r if max_r > -1 else np.nan


def load_validation_metrics(val_dir: Path) -> Dict[str, float]:
    """Load validation metrics from parquet files and compute aggregated stats.

    Uses the pre-computed metrics stored in the parquet files:
    - metric_pearson_r_mean_Z: per-trial behavioral Pearson r
    - metric_pearson_overall_mean_Z: overall behavioral Pearson
    - metric_pearson_r_mean: per-trial neural Pearson r

    Also computes cross-correlation for shape similarity with lag tolerance.
    """
    pearson_behavioral_trials = []
    pearson_neural_trials = []
    overall_behavioral = []
    xcorr_behavioral = []

    # Read all trial parquet files
    parquet_files = list(val_dir.rglob("*.parquet"))

    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)

            # Get the behavioral Pearson r for this trial
            if "metric_pearson_r_mean_Z" in df.columns:
                val = df["metric_pearson_r_mean_Z"].iloc[0]
                if pd.notna(val):
                    pearson_behavioral_trials.append(val)

            # Get the overall behavioral Pearson
            if "metric_pearson_overall_mean_Z" in df.columns:
                val = df["metric_pearson_overall_mean_Z"].iloc[0]
                if pd.notna(val):
                    overall_behavioral.append(val)

            # Get the neural Pearson r
            if "metric_pearson_r_mean" in df.columns:
                val = df["metric_pearson_r_mean"].iloc[0]
                if pd.notna(val):
                    pearson_neural_trials.append(val)

            # Compute cross-correlation for shape matching
            if "Z_future_true" in df.columns and "Z_future_pred" in df.columns:
                z_true_arr = df["Z_future_true"].iloc[0]
                z_pred_arr = df["Z_future_pred"].iloc[0]

                if z_true_arr is not None and z_pred_arr is not None:
                    z_true = np.array(z_true_arr)
                    z_pred = np.array(z_pred_arr)

                    # For each channel, compute cross-correlation
                    if z_true.ndim == 1 and len(z_true) > 0:
                        # Each element is a multi-channel array
                        # Stack to get (time, channels)
                        try:
                            z_true_stacked = np.vstack(z_true)
                            z_pred_stacked = np.vstack(z_pred)

                            # Compute xcorr for each channel and average
                            n_channels = z_true_stacked.shape[1]
                            channel_xcorrs = []
                            for ch in range(n_channels):
                                xcorr = max_cross_correlation(
                                    z_true_stacked[:, ch],
                                    z_pred_stacked[:, ch],
                                    max_lag=30,
                                )
                                if not np.isnan(xcorr):
                                    channel_xcorrs.append(xcorr)

                            if channel_xcorrs:
                                xcorr_behavioral.append(np.mean(channel_xcorrs))
                        except:
                            pass

        except Exception as e:
            continue

    # Compute robust metrics for behavioral
    metrics = {}

    if pearson_behavioral_trials:
        valid = [r for r in pearson_behavioral_trials if not np.isnan(r)]

        if valid:
            z_vals = [fisher_z(r) for r in valid]
            fisher_mean = inverse_fisher_z(np.mean(z_vals))
            median_r = np.median(valid)
            pct_good = 100 * np.mean([r > 0.3 for r in valid])
            best_r = np.max(valid)
            concat_r = (
                np.mean(overall_behavioral) if overall_behavioral else fisher_mean
            )

            metrics.update(
                {
                    "fisher_mean_behavioral": fisher_mean,
                    "median_behavioral": median_r,
                    "best_behavioral": best_r,
                    "pct_good_behavioral": pct_good,
                    "concat_r_behavioral": concat_r,
                    "n_trials_behavioral": len(valid),
                }
            )
        else:
            metrics.update(
                {
                    "fisher_mean_behavioral": np.nan,
                    "median_behavioral": np.nan,
                    "best_behavioral": np.nan,
                    "pct_good_behavioral": np.nan,
                    "concat_r_behavioral": np.nan,
                    "n_trials_behavioral": 0,
                }
            )
    else:
        metrics.update(
            {
                "fisher_mean_behavioral": np.nan,
                "median_behavioral": np.nan,
                "best_behavioral": np.nan,
                "pct_good_behavioral": np.nan,
                "concat_r_behavioral": np.nan,
                "n_trials_behavioral": 0,
            }
        )

    # Cross-correlation metrics (shape similarity)
    if xcorr_behavioral:
        valid_xcorr = [x for x in xcorr_behavioral if not np.isnan(x)]
        if valid_xcorr:
            metrics.update(
                {
                    "xcorr_mean_behavioral": np.mean(valid_xcorr),
                    "xcorr_median_behavioral": np.median(valid_xcorr),
                    "xcorr_best_behavioral": np.max(valid_xcorr),
                }
            )
        else:
            metrics.update(
                {
                    "xcorr_mean_behavioral": np.nan,
                    "xcorr_median_behavioral": np.nan,
                    "xcorr_best_behavioral": np.nan,
                }
            )
    else:
        metrics.update(
            {
                "xcorr_mean_behavioral": np.nan,
                "xcorr_median_behavioral": np.nan,
                "xcorr_best_behavioral": np.nan,
            }
        )

    # Neural metrics
    if pearson_neural_trials:
        valid = [r for r in pearson_neural_trials if not np.isnan(r)]
        if valid:
            z_vals = [fisher_z(r) for r in valid]
            metrics.update(
                {
                    "fisher_mean_neural": inverse_fisher_z(np.mean(z_vals)),
                    "median_neural": np.median(valid),
                    "best_neural": np.max(valid),
                    "n_trials_neural": len(valid),
                }
            )
        else:
            metrics.update(
                {
                    "fisher_mean_neural": np.nan,
                    "median_neural": np.nan,
                    "best_neural": np.nan,
                    "n_trials_neural": 0,
                }
            )
    else:
        metrics.update(
            {
                "fisher_mean_neural": np.nan,
                "median_neural": np.nan,
                "best_neural": np.nan,
                "n_trials_neural": 0,
            }
        )

    return metrics


def create_summary_dataframe(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary DataFrame from all runs."""
    df = pd.DataFrame(runs)

    # Sort by cross-correlation (shape similarity) first, then fall back to other metrics
    sort_col = None
    for col in [
        "xcorr_mean_behavioral",
        "concat_r_behavioral",
        "fisher_mean_behavioral",
        "median_behavioral",
    ]:
        if col in df.columns and df[col].notna().any():
            sort_col = col
            break

    if sort_col:
        df = df.sort_values(sort_col, ascending=False)

    df = df.reset_index(drop=True)
    df.index = df.index + 1  # 1-indexed ranking
    df.index.name = "rank"

    return df


def print_best_runs(df: pd.DataFrame, top_n: int = 10):
    """Print the top N best performing runs."""
    print(f"\n{'='*80}")
    print(f"TOP {top_n} BEST PARAMETER COMBINATIONS")
    print(f"{'='*80}\n")

    param_cols = [
        "nx",
        "n1",
        "alpha_Q",
        "alpha_R",
        "backward_kalman",
        "rescale_states",
        "max_eigenvalue",
    ]
    metric_cols = [
        "xcorr_mean_behavioral",
        "xcorr_best_behavioral",
        "fisher_mean_behavioral",
        "median_behavioral",
        "best_behavioral",
        "n_trials_behavioral",
    ]

    # Filter to existing columns
    param_cols = [c for c in param_cols if c in df.columns]
    metric_cols = [c for c in metric_cols if c in df.columns]

    display_cols = param_cols + metric_cols

    if len(df) == 0:
        print("No results found!")
        return

    top_df = df.head(top_n)[display_cols]

    # Format for display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(top_df.to_string())

    # Print best configuration details
    if len(df) > 0:
        best = df.iloc[0]
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION:")
        print(f"{'='*80}")
        for col in param_cols:
            if col in best:
                print(f"  {col}: {best[col]}")
        print()
        print("Metrics (behavioral):")
        for col in metric_cols:
            if col in best and not pd.isna(best[col]):
                if "pct" in col or "n_trials" in col:
                    print(f"  {col}: {best[col]:.1f}")
                else:
                    print(f"  {col}: {best[col]:.4f}")


def create_html_report(df: pd.DataFrame, output_path: Path):
    """Create an interactive HTML report with visualizations."""

    total_runs = len(df)

    if (
        "concat_r_behavioral" in df.columns
        and len(df) > 0
        and df["concat_r_behavioral"].notna().any()
    ):
        best_concat_r = f"{df['concat_r_behavioral'].max():.4f}"
    else:
        best_concat_r = "N/A"

    best_nx = df["nx"].iloc[0] if len(df) > 0 and "nx" in df.columns else "N/A"
    best_n1 = df["n1"].iloc[0] if len(df) > 0 and "n1" in df.columns else "N/A"

    results_table = df.head(50).to_html(classes="results-table", index=True)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PSID Grid Search Results</title>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #e0e0e0;
                margin: 0;
                padding: 20px;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            h1 {{
                text-align: center;
                background: linear-gradient(90deg, #00d4ff, #7c3aed);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.5em;
                margin-bottom: 30px;
            }}
            h2 {{ color: #00d4ff; }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #00d4ff;
            }}
            .stat-label {{ color: #888; font-size: 0.9em; }}
            .plot-container {{
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                overflow: hidden;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid rgba(255,255,255,0.1);
            }}
            th {{
                background: rgba(124, 58, 237, 0.3);
                font-weight: 600;
            }}
            tr:hover {{ background: rgba(255,255,255,0.05); }}
            .best {{ color: #00ff88; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PSID Grid Search Results</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_runs}</div>
                    <div class="stat-label">Total Runs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{best_concat_r}</div>
                    <div class="stat-label">Best Concat r (behavioral)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{best_nx}</div>
                    <div class="stat-label">Best nx</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{best_n1}</div>
                    <div class="stat-label">Best n1</div>
                </div>
            </div>
            
            <h2>All Results (Ranked by Performance)</h2>
            {results_table}
        </div>
    </body>
    </html>
    """

    output_path.write_text(html_content)
    print(f"\nHTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze PSID grid search results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/psid_SEARCH_beta_ecog",
        help="Path to results directory",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Number of top results to display"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output CSV path for full results"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print(f"Scanning results in: {results_dir}")

    # Find and load all runs
    runs = find_all_runs(results_dir)

    if len(runs) == 0:
        print("No completed runs found!")
        return 1

    print(f"Found {len(runs)} completed runs")

    # Create summary DataFrame
    df = create_summary_dataframe(runs)

    # Save to CSV
    output_csv = (
        Path(args.output) if args.output else results_dir / "grid_search_summary.csv"
    )
    df.to_csv(output_csv)
    print(f"Full results saved to: {output_csv}")

    # Print top results
    print_best_runs(df, args.top)

    # Create HTML report
    html_path = results_dir / "grid_search_report.html"
    create_html_report(df, html_path)

    return 0


if __name__ == "__main__":
    exit(main())
