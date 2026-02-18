#!/usr/bin/env python3
"""
Estimate total export size for Neptune projects.
Usage: uv run python estimate_size.py <project-id>
Example: uv run python estimate_size.py si-inc/molmo
"""

import os
import sys
import logging
import neptune_query as nq
from neptune_query import filters, runs as nq_runs
from neptune_query.filters import Attribute, AttributeFilter

# Silence Neptune logging
logging.getLogger("neptune").setLevel(logging.ERROR)


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def estimate_project_size(project_id: str):
    """Estimate the total export size for a project."""

    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if not api_token:
        print("Error: NEPTUNE_API_TOKEN not set")
        sys.exit(1)

    nq.set_api_token(api_token)

    print(f"Estimating export size for: {project_id}\n")

    # List all runs
    print("1. Counting runs...")
    runs_filter = filters.Filter.matches(
        filters.Attribute("sys/custom_run_id", type="string"), ".+"
    )
    run_ids = nq_runs.list_runs(project=project_id, runs=runs_filter)
    num_runs = len(run_ids)
    print(f"   Found {num_runs} runs")

    if num_runs == 0:
        print("\nNo runs found in project.")
        return

    # Fetch all attributes to count data points
    print("\n2. Analyzing attributes...")

    try:
        # Get all attributes (this might take a while for large projects)
        # We'll fetch a sample of runs to estimate
        sample_size = min(10, num_runs)
        sample_runs = run_ids[:sample_size]

        print(f"   Sampling {sample_size} runs to estimate...")

        # Fetch parameters (single values)
        params_df = nq_runs.fetch_runs_table(
            project=project_id,
            runs=sample_runs,
            attributes=AttributeFilter(
                type=["float", "int", "string", "bool", "datetime", "string_set"]
            ),
        )

        # Estimate parameters: count non-null values
        param_values = params_df.notna().sum().sum() if not params_df.empty else 0
        avg_param_values_per_run = param_values / sample_size if sample_size > 0 else 0
        total_param_values = int(avg_param_values_per_run * num_runs)

        # Estimate parquet size for parameters (~100 bytes per value)
        param_size = total_param_values * 100

        print(f"   Parameters: ~{total_param_values:,} values")
        print(f"   Estimated size: {format_size(param_size)}")

    except Exception as e:
        print(f"   Warning: Could not analyze attributes: {e}")
        param_size = 0
        total_param_values = 0

    # Estimate series/metrics size
    print("\n3. Estimating series/metrics size...")
    try:
        # Fetch series attributes (these have multiple points)
        series_df = nq_runs.fetch_runs_table(
            project=project_id,
            runs=sample_runs,
            attributes=AttributeFilter(
                type=["float_series", "string_series", "histogram_series"]
            ),
        )

        num_series = series_df.shape[1] if not series_df.empty else 0
        avg_series_per_run = num_series / sample_size if sample_size > 0 else 0
        total_series_count = int(avg_series_per_run * num_runs)

        # Rough estimate: 1000 points per series, 50 bytes per point
        estimated_points = total_series_count * 1000
        series_size = estimated_points * 50

        print(f"   Series attributes: ~{total_series_count:,}")
        print(f"   Estimated points: ~{estimated_points:,}")
        print(f"   Estimated size: {format_size(series_size)}")

    except Exception as e:
        print(f"   Warning: Could not analyze series: {e}")
        series_size = 0

    # Estimate file sizes
    print("\n4. Estimating file sizes...")
    print("   Note: File sizes may not be available via API")
    print("   Using rough estimate based on typical ML projects...")

    # Very rough estimate: average 100MB per run in files
    avg_files_per_run = 100 * 1024 * 1024  # 100 MB
    files_size = avg_files_per_run * num_runs

    print(f"   Estimated files: {format_size(files_size)}")

    # Total estimate
    total_size = param_size + series_size + files_size

    print(f"\n{'='*60}")
    print(f"ESTIMATED EXPORT SIZE")
    print(f"{'='*60}")
    print(f"  Parameters/config: {format_size(param_size):>15s}")
    print(f"  Metrics/series:    {format_size(series_size):>15s}")
    print(f"  Files (estimate):  {format_size(files_size):>15s}")
    print(f"{'='*60}")
    print(f"  Total (estimate):  {format_size(total_size):>15s}")
    print(f"{'='*60}\n")

    print("Note: This is a rough estimate based on sampling.")
    print("Actual size may vary significantly, especially for files.")
    print("Run a small export first to calibrate the estimate.\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python estimate_size.py <project-id>")
        print("Example: uv run python estimate_size.py si-inc/molmo")
        sys.exit(1)

    project_id = sys.argv[1]
    estimate_project_size(project_id)
