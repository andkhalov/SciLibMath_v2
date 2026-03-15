#!/bin/bash
# run_ablation.sh — Sequential ablation sweep
# Usage: bash run_ablation.sh [SWEEP_NAME]
#   SWEEP_NAME defaults to "sweep" if not provided
#
# Ref: MATH.md M.3, TZ.md Sec 6
# Results go to runs/ (TensorBoard), checkpoints/ (best models).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sweep name for log prefix (e.g., sweep11)
SWEEP_NAME="${1:-sweep}"

# Activate venv
source "$SCRIPT_DIR/venv/bin/activate"

# Ablation overrides (EXP-008: 5% data, 5 epochs)
OVERRIDES="data.dataset_fraction=0.05 data.batch_size=64 data.num_workers=8 training.epochs=5 eval.eval_every_steps=200"

# Family A experiments (10)
EXPERIMENTS_A=(
    e1_pairwise
    e3_centroid_reg
    e4_composite_static
    e5_composite_learnable
    e6_fuzzy
    e7_lyapunov
    e8c_active
    e9_potential
    e10_potential_fuzzy
    e8c_low_va
)

# Family B experiments (3)
EXPERIMENTS_B=(
    e1b_pairwise
    e3b_centroid_reg
    e4b_composite_static
)

# ConvNeXt-Tiny backbone (10) — EXP-008
EXPERIMENTS_CNXT=(
    e1_pairwise_cnxt
    e3_centroid_reg_cnxt
    e4_composite_static_cnxt
    e5_composite_learnable_cnxt
    e6_fuzzy_cnxt
    e7_lyapunov_cnxt
    e8c_active_cnxt
    e9_potential_cnxt
    e10_potential_fuzzy_cnxt
    e8c_low_va_cnxt
)

# EXP-008 (sweep11): Family A + Family B + ConvNeXt, 23 experiments
EXPERIMENTS_F=(
    "${EXPERIMENTS_A[@]}"
    "${EXPERIMENTS_B[@]}"
    "${EXPERIMENTS_CNXT[@]}"
)

ALL_EXPERIMENTS=("${EXPERIMENTS_F[@]}")

# Create logs dir
mkdir -p logs

# Start TensorBoard for live monitoring
bash "$SCRIPT_DIR/start_tensorboard.sh" 2>/dev/null || true
echo ""

echo "=== SciLibMath_v2 Ablation Sweep ($SWEEP_NAME) ==="
echo "Experiments: ${ALL_EXPERIMENTS[*]}"
echo "Total: ${#ALL_EXPERIMENTS[@]} experiments"
echo "Overrides: $OVERRIDES"
echo "Log prefix: ${SWEEP_NAME}_"
echo "Start: $(date -Iseconds)"
echo ""

# Main log
MAIN_LOG="logs/${SWEEP_NAME}_main.log"
echo "=== Sweep $SWEEP_NAME started $(date -Iseconds) ===" > "$MAIN_LOG"

FAILED=()
SUCCEEDED=()

for exp in "${ALL_EXPERIMENTS[@]}"; do
    echo "========================================" | tee -a "$MAIN_LOG"
    echo "  Running: $exp" | tee -a "$MAIN_LOG"
    echo "  Config: configs/${exp}.yaml" | tee -a "$MAIN_LOG"
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
echo "Succeeded: ${#SUCCEEDED[@]}/${#ALL_EXPERIMENTS[@]} (${SUCCEEDED[*]:-none})" | tee -a "$MAIN_LOG"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}" | tee -a "$MAIN_LOG"
fi
echo "" | tee -a "$MAIN_LOG"
echo "TensorBoard still running on port 14714 — stop with: bash stop_tensorboard.sh"
echo "Logs: logs/${SWEEP_NAME}_*.log"
