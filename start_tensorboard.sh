#!/bin/bash
# start_tensorboard.sh — Launch TensorBoard for SciLibMath_v2 experiments
# Usage:
#   bash start_tensorboard.sh              # all runs
#   bash start_tensorboard.sh e3           # only runs matching "e3"
#   bash start_tensorboard.sh e3,e5,e7     # multiple filters (comma-separated)
#
# TensorBoard UI: http://<host>:14714
# Auto-refreshes every 30s, picks up new data from ongoing experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

PORT=14714
RUNS_DIR="$SCRIPT_DIR/runs"
PIDFILE="$SCRIPT_DIR/.tensorboard.pid"

# Stop existing instance if running
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing TensorBoard (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PIDFILE"
fi

# Build logdir spec
FILTER="${1:-}"
if [ -z "$FILTER" ]; then
    # All runs — TensorBoard will show each as a separate run
    LOGDIR="$RUNS_DIR"
    echo "Serving ALL runs from $RUNS_DIR"
else
    # Filtered: build --logdir_spec name:path,name:path,...
    LOGDIR_SPEC=""
    IFS=',' read -ra PATTERNS <<< "$FILTER"
    for pattern in "${PATTERNS[@]}"; do
        pattern=$(echo "$pattern" | xargs)  # trim whitespace
        for dir in "$RUNS_DIR"/${pattern}*; do
            if [ -d "$dir" ]; then
                name=$(basename "$dir")
                if [ -n "$LOGDIR_SPEC" ]; then
                    LOGDIR_SPEC="${LOGDIR_SPEC},"
                fi
                LOGDIR_SPEC="${LOGDIR_SPEC}${name}:${dir}"
            fi
        done
    done

    if [ -z "$LOGDIR_SPEC" ]; then
        echo "No runs found matching filter: $FILTER"
        echo "Available runs:"
        ls "$RUNS_DIR"
        exit 1
    fi
fi

echo "Starting TensorBoard on port $PORT..."
echo "URL: http://$(hostname -I | awk '{print $1}'):$PORT"

if [ -z "$FILTER" ]; then
    tensorboard \
        --logdir "$LOGDIR" \
        --port "$PORT" \
        --bind_all \
        --reload_interval 30 \
        --reload_multifile true \
        &
else
    tensorboard \
        --logdir_spec "$LOGDIR_SPEC" \
        --port "$PORT" \
        --bind_all \
        --reload_interval 30 \
        --reload_multifile true \
        &
fi

TB_PID=$!
echo "$TB_PID" > "$PIDFILE"
echo "TensorBoard started (PID $TB_PID)"
echo "Stop with: bash stop_tensorboard.sh"
