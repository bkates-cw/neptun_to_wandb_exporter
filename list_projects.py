#!/usr/bin/env python3
"""
List all projects in a Neptune workspace with run counts.
Usage: uv run python list_projects.py <workspace> [--neptune3]
Example: uv run python list_projects.py si-inc (uses Neptune 2 by default)
Example: uv run python list_projects.py si-inc --neptune3 (use Neptune 3)
"""

import os
import sys
import logging
from typing import Dict, List, Tuple

# Silence Neptune logging
logging.getLogger("neptune").setLevel(logging.ERROR)

try:
    # Try Neptune 2.x API first
    from neptune import management

    def list_projects_neptune2(workspace: str = None):
        """List projects using Neptune 2.x management API."""
        projects = management.get_project_list()

        if workspace:
            # Filter by workspace
            projects = [p for p in projects if p.startswith(f"{workspace}/")]

        return projects

    USE_NEPTUNE2 = True
except ImportError:
    USE_NEPTUNE2 = False


def list_projects_rest_api(workspace: str):
    """List projects using Neptune REST API (Neptune 2)."""
    import requests

    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if not api_token:
        print("Error: NEPTUNE_API_TOKEN not set")
        sys.exit(1)

    # Neptune 2 API endpoint
    base_url = "https://app.neptune.ai"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # Get workspace projects
    response = requests.get(
        f"{base_url}/api/leaderboard/v1/workspaces/{workspace}/projects",
        headers=headers
    )

    if response.status_code != 200:
        print(f"Error: Failed to fetch projects (status {response.status_code})")
        print(f"Response: {response.text}")
        sys.exit(1)

    data = response.json()
    projects = [f"{workspace}/{p['name']}" for p in data.get('entries', [])]
    return projects


def get_run_count_neptune3(project_id: str) -> int:
    """Get run count using Neptune 3 (neptune-query)."""
    import neptune_query as nq
    from neptune_query import filters, runs as nq_runs

    try:
        # List all runs (including archived)
        runs_filter = filters.Filter.matches(
            filters.Attribute("sys/custom_run_id", type="string"), ".+"
        )
        run_ids = nq_runs.list_runs(project=project_id, runs=runs_filter)
        return len(run_ids)
    except Exception as e:
        return -1  # Error


def get_run_count_neptune2(project_id: str) -> int:
    """Get run count using Neptune 2."""
    import neptune

    try:
        with neptune.init_project(project=project_id, mode="read-only") as project:
            runs_table = project.fetch_runs_table(
                columns=["sys/id"],
                trashed=None,  # Include all runs
                progress_bar=False,
            ).to_pandas()
            return len(runs_table)
    except Exception as e:
        return -1  # Error


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: uv run python list_projects.py <workspace> [--neptune3]")
        print("Example: uv run python list_projects.py si-inc")
        print("Example: uv run python list_projects.py si-inc --neptune3")
        sys.exit(1)

    workspace = sys.argv[1]
    use_neptune3 = len(sys.argv) == 3 and sys.argv[2] == "--neptune3"

    api_token = os.getenv("NEPTUNE_API_TOKEN")
    if not api_token:
        print("Error: NEPTUNE_API_TOKEN not set")
        sys.exit(1)

    print(f"Fetching projects for workspace: {workspace}")

    if USE_NEPTUNE2:
        print("Using Neptune 2.x management API...")
        projects = list_projects_neptune2(workspace)
    else:
        print("Using Neptune REST API...")
        projects = list_projects_rest_api(workspace)

    if not projects:
        print(f"No projects found for workspace: {workspace}")
        sys.exit(0)

    print(f"\nFound {len(projects)} project(s). Fetching run counts...\n")

    # Get run counts for each project (default to Neptune 2)
    if use_neptune3:
        import neptune_query as nq
        nq.set_api_token(api_token)
        get_count_func = get_run_count_neptune3
        print("Using Neptune 3 API for run counts\n")
    else:
        get_count_func = get_run_count_neptune2
        print("Using Neptune 2 API for run counts\n")

    projects_with_counts: List[Tuple[str, int]] = []
    for i, project in enumerate(sorted(projects), 1):
        print(f"[{i}/{len(projects)}] Checking {project}...", end=" ", flush=True)
        count = get_count_func(project)
        projects_with_counts.append((project, count))
        if count >= 0:
            print(f"{count} runs")
        else:
            print("ERROR")

    # Print summary sorted by run count (least to most)
    print(f"\n{'='*60}")
    print(f"Projects in {workspace} (sorted by run count)")
    print(f"{'='*60}")
    total_runs = 0

    # Sort by run count (errors at the end)
    projects_with_counts.sort(key=lambda x: (x[1] < 0, x[1]))

    for project, count in projects_with_counts:
        if count >= 0:
            print(f"  {project:40s} {count:6d} runs")
            total_runs += count
        else:
            print(f"  {project:40s} {'ERROR':>6s}")

    print(f"{'='*60}")
    print(f"  {'Total':40s} {total_runs:6d} runs")
    print(f"{'='*60}\n")
