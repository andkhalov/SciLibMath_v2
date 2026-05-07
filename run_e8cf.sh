#!/bin/bash
# run_e8cf.sh — EXP-013: E8cf differentiable controller sweep
# 10% data, 5 epochs (same as sweep13/baselines settings)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

OVERRIDES="data.dataset_fraction=0.1 data.batch_size=64 data.num_workers=8 training.epochs=5 eval.eval_every_steps=50 checkpoint.enabled=false"

echo "=== E8cf: Differentiable T-S Controller ==="
echo "Date: $(date -Iseconds)"
echo "Overrides: $OVERRIDES"

mkdir -p logs
LOG_FILE="logs/exp013_e8cf_differentiable.log"

echo "Running e8cf_differentiable ..."
if python code/train.py --config configs/e8cf_differentiable.yaml $OVERRIDES > "$LOG_FILE" 2>&1; then
    echo "  OK"
else
    echo "  FAILED — see $LOG_FILE"
    LAST_RUN=$(ls -td runs/e8cf_differentiable_* 2>/dev/null | head -1)
    if [ -n "$LAST_RUN" ]; then
        echo "  Cleaning up: $LAST_RUN"
        rm -rf "$LAST_RUN"
    fi
    exit 1
fi

echo "Done: $(date -Iseconds)"
echo "TensorBoard: tensorboard --logdir runs/"
