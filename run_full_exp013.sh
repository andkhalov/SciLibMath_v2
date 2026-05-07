#!/bin/bash
# run_full_exp013.sh — Full training for EXP-013 baselines + e8cf
# 4 experiments: BL1 (GradNorm), BL3 (Uncertainty), e8cf_unc (v3), e8cf_real (v5)
# 100% data, 15 epochs, checkpoints enabled
# Expected: ~20h per experiment, ~80h total (sequential)

set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate

COMMON="data.dataset_fraction=1.0 data.test_fraction=0.02 data.val_fraction=0.03 data.batch_size=64 data.num_workers=8 training.epochs=15 eval.eval_every_steps=1000 checkpoint.enabled=true checkpoint.keep_best=3 run_tag=full"

# Experiment order: fastest/simplest first
declare -a CONFIGS=(
    "bl3_uncertainty"
    "bl1_gradnorm"
    "e8cf_differentiable"    # uses e8cf_unc config (restore elastic_gamma=0.002)
    "e8cf_real"              # v5 residual MLP
)

echo "=== EXP-013 Full Training ==="
echo "Date: $(date -Iseconds)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Experiments: ${CONFIGS[*]}"
echo ""

mkdir -p logs
MAIN_LOG="logs/exp013_full_main.log"
echo "=== EXP-013 Full Training $(date -Iseconds) ===" > "$MAIN_LOG"

TOTAL=${#CONFIGS[@]}
CURRENT=0
FAILED=()

for cfg in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    LOG_FILE="logs/exp013_full_${cfg}.log"
    echo "[$CURRENT/$TOTAL] Running $cfg (full) ..." | tee -a "$MAIN_LOG"
    echo "  Start: $(date -Iseconds)" | tee -a "$MAIN_LOG"

    if python code/train.py --config "configs/${cfg}.yaml" $COMMON > "$LOG_FILE" 2>&1; then
        echo "  OK ($cfg) at $(date -Iseconds)" | tee -a "$MAIN_LOG"
    else
        echo "  FAILED ($cfg) — see $LOG_FILE" | tee -a "$MAIN_LOG"
        FAILED+=("$cfg")
    fi
    echo "" | tee -a "$MAIN_LOG"
done

echo "=== EXP-013 Full Training Complete ===" | tee -a "$MAIN_LOG"
echo "Date: $(date -Iseconds)" | tee -a "$MAIN_LOG"
echo "Total: $TOTAL, Failed: ${#FAILED[@]}" | tee -a "$MAIN_LOG"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed: ${FAILED[*]}" | tee -a "$MAIN_LOG"
fi
