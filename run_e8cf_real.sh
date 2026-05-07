#!/bin/bash
# run_e8cf_real.sh — E8cf v5: residual MLP consequents
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate

OVERRIDES="data.dataset_fraction=0.1 data.batch_size=64 data.num_workers=8 training.epochs=5 eval.eval_every_steps=50 checkpoint.enabled=false"

echo "=== E8cf_real: Residual MLP Consequents ==="
echo "Date: $(date -Iseconds)"

mkdir -p logs
LOG_FILE="logs/exp013_e8cf_real.log"

echo "Running e8cf_real ..."
if python code/train.py --config configs/e8cf_real.yaml $OVERRIDES > "$LOG_FILE" 2>&1; then
    echo "  OK"
else
    echo "  FAILED — see $LOG_FILE"
    LAST_RUN=$(ls -td runs/e8cf_differentiable_* 2>/dev/null | head -1)
    if [ -n "$LAST_RUN" ]; then rm -rf "$LAST_RUN"; fi
    exit 1
fi
echo "Done: $(date -Iseconds)"
