#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <project>"
    echo "Example: $0 byyoung3/testing2"
    exit 1
fi

# Toggle to exclude monitoring/hardware metrics
EXCLUDE_MONITORING=true

# Toggle to disable ALL data filtering (no --runs-query, no --attributes)
DISABLE_FILTERING=false

PROJECT="$1"
DATA_DIR="./exports/data"
FILES_DIR="./exports/files"
PARALLEL_JOBS=4
START_DATE="2025-01-01"
END_DATE="2026-12-31"
INTERVAL_DAYS=365

# Use gdate on macOS for GNU date functions
if command -v gdate >/dev/null 2>&1; then
    DATE_CMD=gdate
else
    DATE_CMD=date
fi

# Convert start and end dates to seconds
start_ts=$($DATE_CMD -d "$START_DATE" +%s)
end_ts=$($DATE_CMD -d "$END_DATE" +%s)
interval_sec=$((INTERVAL_DAYS*24*60*60))

mkdir -p "$DATA_DIR"
mkdir -p "$FILES_DIR"

pids=()
current_ts=$start_ts

run_export() {
    local from_ts=$1
    local to_ts=$2

    from_date=$($DATE_CMD -u -d @$from_ts +%Y-%m-%dT00:00:00Z)
    to_date=$($DATE_CMD -u -d @$to_ts +%Y-%m-%dT23:59:59Z)

    echo "Exporting $from_date -> $to_date"

    # Build the command as an array for proper argument handling
    CMD=(uv run neptune-exporter export
        --exporter neptune2
        -p "$PROJECT"
        -d "$DATA_DIR"
        -f "$FILES_DIR"
        --verbose
    )

    # Add date filter unless filtering is disabled
    if [ "$DISABLE_FILTERING" != true ]; then
        CMD+=(--runs-query "(\`sys/creation_time\`:datetime >= \"$from_date\") AND (\`sys/creation_time\`:datetime <= \"$to_date\")")
    fi

    # Add monitoring exclusion unless filtering is disabled
    if [ "$EXCLUDE_MONITORING" = true ] && [ "$DISABLE_FILTERING" != true ]; then
        CMD+=(--attributes "^(?!monitoring).*$")
    fi

    "${CMD[@]}" &

    pids+=($!)
    if [ "${#pids[@]}" -ge "$PARALLEL_JOBS" ]; then
        wait "${pids[@]}"
        pids=()
    fi
}

if [ "$DISABLE_FILTERING" = true ]; then
    # Single unfiltered export — no date splitting, no parallelism needed
    echo "Filtering disabled — running single unfiltered export"
    uv run neptune-exporter export \
        --exporter neptune2 \
        -p "$PROJECT" \
        -d "$DATA_DIR" \
        -f "$FILES_DIR" \
        --verbose
else
    while [ "$current_ts" -le "$end_ts" ]; do
        next_ts=$((current_ts + interval_sec - 1))
        if [ "$next_ts" -gt "$end_ts" ]; then
            next_ts=$end_ts
        fi

        run_export "$current_ts" "$next_ts"
        current_ts=$((next_ts + 1))
    done

    # Wait for any remaining background jobs
    if [ "${#pids[@]}" -gt 0 ]; then
        wait "${pids[@]}"
    fi
fi

echo "All exports finished."
