#!/bin/bash
# run_sweep_baselines.sh — EXP-013: Multi-task balancing baselines
# 10% data, 5 epochs (same as sweep13 settings)
# Ref: exp_reports/exp013_baselines_plan.md

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source venv/bin/activate

OVERRIDES="data.dataset_fraction=0.1 data.batch_size=64 data.num_workers=8 training.epochs=5 eval.eval_every_steps=50 checkpoint.enabled=false"

CONFIGS=(
    bl3_uncertainty
    bl1_gradnorm
    bl2_pcgrad
)

echo "=== EXP-013: Multi-Task Balancing Baselines ==="
echo "Date: $(date -Iseconds)"
echo "Overrides: $OVERRIDES"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

TOTAL=${#CONFIGS[@]}
CURRENT=0
FAILED=()

mkdir -p logs

for cfg in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    LOG_FILE="logs/exp013_${cfg}.log"
    echo "[$CURRENT/$TOTAL] Running $cfg ..."
    if python code/train.py --config "configs/${cfg}.yaml" $OVERRIDES > "$LOG_FILE" 2>&1; then
        echo "  OK ($cfg)"
    else
        echo "  FAILED ($cfg) — see $LOG_FILE"
        FAILED+=("$cfg")
        # Clean up failed run's TensorBoard logs
        LAST_RUN=$(ls -td runs/${cfg}_* 2>/dev/null | head -1)
        if [ -n "$LAST_RUN" ]; then
            echo "  Cleaning up failed run: $LAST_RUN"
            rm -rf "$LAST_RUN"
        fi
    fi
done

echo ""
echo "=== EXP-013 Sweep Complete ==="
echo "Date: $(date -Iseconds)"
echo "Total: $TOTAL experiments"
echo "Failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed experiments: ${FAILED[*]}"
fi
echo "TensorBoard: tensorboard --logdir runs/"
