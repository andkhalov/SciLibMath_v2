#!/bin/bash
# stop_tensorboard.sh — Stop TensorBoard instance
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$SCRIPT_DIR/.tensorboard.pid"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping TensorBoard (PID $PID)..."
        kill "$PID"
        echo "Stopped."
    else
        echo "TensorBoard (PID $PID) not running."
    fi
    rm -f "$PIDFILE"
else
    echo "No TensorBoard PID file found."
    # Try to find and kill anyway
    PIDS=$(pgrep -f "tensorboard.*--port 14714" || true)
    if [ -n "$PIDS" ]; then
        echo "Found TensorBoard process(es): $PIDS"
        kill $PIDS 2>/dev/null || true
        echo "Killed."
    fi
fi
