#!/bin/bash
#
# Parallel Neptune export using run ID pattern splitting
# Usage: ./parallel-export-by-runid.sh <project> <num-jobs>
# Example: ./parallel-export-by-runid.sh si-inc/Inverse-Inverse-Dynamics 4

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <project> <num-jobs>"
    echo "Example: $0 si-inc/Inverse-Inverse-Dynamics 4"
    exit 1
fi

PROJECT="$1"
NUM_JOBS="$2"
BASE_DATA_DIR="${DATA_DIR:-$HOME/si-inc-exports/data}"
BASE_FILES_DIR="${FILES_DIR:-$HOME/si-inc-exports/files}"
EXPORTER="${EXPORTER:-neptune3}"
# Note: EXCLUDE_MONITORING disabled because Neptune doesn't support negative lookahead regex (?!)
EXCLUDE_MONITORING="${EXCLUDE_MONITORING:-false}"

# Check NEPTUNE_API_TOKEN
if [ -z "$NEPTUNE_API_TOKEN" ]; then
    echo "Error: NEPTUNE_API_TOKEN not set"
    exit 1
fi

echo "=========================================="
echo "Parallel Neptune Export by Run ID Pattern"
echo "=========================================="
echo "Project: $PROJECT"
echo "Parallel jobs: $NUM_JOBS"
echo "Exporter: $EXPORTER"
echo "Base data dir: $BASE_DATA_DIR"
echo "Base files dir: $BASE_FILES_DIR"
echo "=========================================="
echo

# Define run ID pattern splits based on number of jobs
case $NUM_JOBS in
    2)
        PATTERNS=("^[a-m].*" "^[n-z].*")
        LABELS=("a-m" "n-z")
        ;;
    4)
        PATTERNS=("^[a-f].*" "^[g-n].*" "^[o-s].*" "^[t-z].*")
        LABELS=("a-f" "g-n" "o-s" "t-z")
        ;;
    8)
        PATTERNS=("^[a-c].*" "^[d-f].*" "^[g-i].*" "^[j-l].*" "^[m-o].*" "^[p-r].*" "^[s-u].*" "^[v-z].*")
        LABELS=("a-c" "d-f" "g-i" "j-l" "m-o" "p-r" "s-u" "v-z")
        ;;
    *)
        echo "Error: NUM_JOBS must be 2, 4, or 8"
        exit 1
        ;;
esac

# Create directories
mkdir -p "$BASE_DATA_DIR"
mkdir -p "$BASE_FILES_DIR"

pids=()

# Launch parallel exports (all to same directory)
for i in "${!PATTERNS[@]}"; do
    pattern="${PATTERNS[$i]}"
    label="${LABELS[$i]}"

    log_file="${BASE_DATA_DIR}/export-${label}.log"

    echo "Job $((i+1))/${NUM_JOBS}: Exporting runs matching pattern '$pattern'"

    # Build command
    CMD=(uv run neptune-exporter export
        --exporter "$EXPORTER"
        -p "$PROJECT"
        -r "$pattern"
        -d "$BASE_DATA_DIR"
        -f "$BASE_FILES_DIR"
        --verbose
    )

    # Add monitoring exclusion if enabled
    if [ "$EXCLUDE_MONITORING" = true ]; then
        CMD+=(--attributes "^(?!monitoring).*$")
    fi

    # Run in background
    "${CMD[@]}" > "$log_file" 2>&1 &
    job_pid=$!
    pids+=($job_pid)

    echo "  Started (PID: ${job_pid}, log: ${log_file})"
    echo
done

echo "=========================================="
echo "All $NUM_JOBS jobs launched. Waiting for completion..."
echo "=========================================="
echo

# Wait for all jobs to complete
for pid in "${pids[@]}"; do
    wait "$pid"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Warning: Job with PID $pid exited with code $exit_code"
    fi
done

echo
echo "=========================================="
echo "All exports finished!"
echo "=========================================="
echo
echo "Data exported to: ${BASE_DATA_DIR}"
echo "Files exported to: ${BASE_FILES_DIR}"
echo
echo "Log files:"
for label in "${LABELS[@]}"; do
    echo "  ${BASE_DATA_DIR}/export-${label}.log"
done
echo
