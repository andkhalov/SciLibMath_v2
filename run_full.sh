#!/bin/bash
# run_full.sh — Full training run (100% data, 15 epochs, checkpoints enabled)
# Usage: bash run_full.sh [SWEEP_NAME]
#   SWEEP_NAME defaults to "full" if not provided

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SWEEP_NAME="${1:-full}"

source "$SCRIPT_DIR/venv/bin/activate"

# Common overrides for full training
# - 100% data, 15 epochs
# - test_fraction=0.02 (~20k held-out), val_fraction=0.03 (~30k for eval during training)
# - checkpoints enabled, keep_best=3
# - eval every 1000 steps (not 200 — too frequent for 14k steps/epoch)
# - run_tag=full for TensorBoard naming
COMMON_OVERRIDES="data.dataset_fraction=1.0 data.test_fraction=0.02 data.val_fraction=0.03 data.num_workers=8 training.epochs=15 eval.eval_every_steps=1000 checkpoint.enabled=true checkpoint.keep_best=3 run_tag=full"

# batch_size=64 for all (same as sweep — proven stable on RTX 3090)
OVERRIDES_RN="${COMMON_OVERRIDES} data.batch_size=64"
OVERRIDES_CX="${COMMON_OVERRIDES} data.batch_size=64"

# Experiments with their backbone type
declare -A EXPERIMENTS
EXPERIMENTS=(
    [e1_pairwise_cnxt]="cnxt"
    [e1_pairwise]="rn"
    [e1b_pairwise]="rn"
    [e8c_low_va]="rn"
    [e8c_low_va_cnxt]="cnxt"
    [e9_potential]="rn"
)

# Order matters for sequential execution
EXPERIMENT_ORDER=(
    e1_pairwise_cnxt
    e1_pairwise
    e1b_pairwise
    e8c_low_va
    e8c_low_va_cnxt
    e9_potential
)

mkdir -p logs

bash "$SCRIPT_DIR/start_tensorboard.sh" 2>/dev/null || true
echo ""

echo "=== SciLibMath_v2 Full Training ($SWEEP_NAME) ==="
echo "Experiments: ${EXPERIMENT_ORDER[*]}"
echo "Total: ${#EXPERIMENT_ORDER[@]} experiments"
echo "ResNet18: batch=256, ConvNeXt: batch=96"
echo "Data: 100%, 15 epochs, val=3%, test=2% (held-out)"
echo "Start: $(date -Iseconds)"
echo ""

MAIN_LOG="logs/${SWEEP_NAME}_main.log"
echo "=== Sweep $SWEEP_NAME started $(date -Iseconds) ===" > "$MAIN_LOG"

FAILED=()
SUCCEEDED=()

for exp in "${EXPERIMENT_ORDER[@]}"; do
    backbone="${EXPERIMENTS[$exp]}"
    if [ "$backbone" = "cnxt" ]; then
        OVERRIDES="$OVERRIDES_CX"
    else
        OVERRIDES="$OVERRIDES_RN"
    fi

    echo "========================================" | tee -a "$MAIN_LOG"
    echo "  Running: $exp (backbone=$backbone)" | tee -a "$MAIN_LOG"
    echo "  Config: configs/${exp}.yaml" | tee -a "$MAIN_LOG"
    echo "  Overrides: $OVERRIDES" | tee -a "$MAIN_LOG"
    echo "  Time: $(date -Iseconds)" | tee -a "$MAIN_LOG"
    echo "========================================" | tee -a "$MAIN_LOG"

    LOG_FILE="logs/${SWEEP_NAME}_${exp}.log"

    if python code/train.py --config "configs/${exp}.yaml" $OVERRIDES 2>&1 | tee "$LOG_FILE"; then
        SUCCEEDED+=("$exp")
        echo "  $exp DONE at $(date -Iseconds)" | tee -a "$MAIN_LOG"
    else
        FAILED+=("$exp")
        echo "  $exp FAILED at $(date -Iseconds)" | tee -a "$MAIN_LOG"
    fi

    echo "" | tee -a "$MAIN_LOG"
done

echo "=== Sweep $SWEEP_NAME complete $(date -Iseconds) ===" | tee -a "$MAIN_LOG"
echo "Succeeded: ${#SUCCEEDED[@]}/${#EXPERIMENT_ORDER[@]} (${SUCCEEDED[*]:-none})" | tee -a "$MAIN_LOG"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}" | tee -a "$MAIN_LOG"
fi
echo "" | tee -a "$MAIN_LOG"
echo "TensorBoard running on port 14714"
echo "Logs: logs/${SWEEP_NAME}_*.log"
