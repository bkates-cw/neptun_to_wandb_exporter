#!/usr/bin/env python3
"""
List all run IDs for a Neptune project.
Usage: uv run python list_run_ids.py <project-id>
Example: uv run python list_run_ids.py si-inc/Inverse-Inverse-Dynamics
"""

import os
import sys
import logging
import neptune_query as nq
from neptune_query import filters, runs as nq_runs

# Silence Neptune logging
logging.getLogger("neptune").setLevel(logging.ERROR)


def list_run_ids(project_id: str):
    """List all run IDs for a project."""

    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if not api_token:
        print("Error: NEPTUNE_API_TOKEN not set")
        sys.exit(1)

    nq.set_api_token(api_token)

    print(f"Fetching run IDs for: {project_id}\n")

    # List all runs
    runs_filter = filters.Filter.matches(
        filters.Attribute("sys/custom_run_id", type="string"), ".+"
    )
    run_ids = nq_runs.list_runs(project=project_id, runs=runs_filter)

    print(f"Found {len(run_ids)} runs\n")
    print("="*60)
    print("Run IDs:")
    print("="*60)

    for run_id in sorted(run_ids):
        print(f"  {run_id}")

    print("="*60)
    print(f"Total: {len(run_ids)} runs")
    print("="*60)

    # Analyze patterns
    print("\nPattern Analysis:")
    print("-"*60)

    # Check for common prefixes
    prefixes = {}
    for run_id in run_ids:
        # Get first word/token before hyphen or underscore
        parts = run_id.split('-')
        if len(parts) > 0:
            prefix = parts[0]
            prefixes[prefix] = prefixes.get(prefix, 0) + 1

    if prefixes:
        print("\nRun ID prefixes (first word before '-'):")
        for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
            print(f"  {prefix}: {count} runs")

    # Check for date patterns
    has_dates = sum(1 for run_id in run_ids if any(char.isdigit() for char in run_id))
    print(f"\nRuns with numbers/dates: {has_dates}/{len(run_ids)}")

    print("\nSuggested parallel split:")
    print("-"*60)
    if len(prefixes) >= 2:
        # Sort prefixes alphabetically and suggest splits
        sorted_prefixes = sorted(prefixes.keys())
        mid = len(sorted_prefixes) // 2
        group1 = sorted_prefixes[:mid]
        group2 = sorted_prefixes[mid:]

        count1 = sum(prefixes[p] for p in group1)
        count2 = sum(prefixes[p] for p in group2)

        print(f"Group 1 ({count1} runs): {', '.join(group1)}")
        print(f'  -r "^({"|".join(group1)}).*"')
        print()
        print(f"Group 2 ({count2} runs): {', '.join(group2)}")
        print(f'  -r "^({"|".join(group2)}).*"')
    else:
        print("Not enough pattern diversity for prefix-based splitting.")
        print("Consider alphabetical split:")
        print(f'  Group 1: -r "^[a-m].*"')
        print(f'  Group 2: -r "^[n-z].*"')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python list_run_ids.py <project-id>")
        print("Example: uv run python list_run_ids.py si-inc/Inverse-Inverse-Dynamics")
        sys.exit(1)

    project_id = sys.argv[1]
    list_run_ids(project_id)
