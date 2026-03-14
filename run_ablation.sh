#!/bin/bash
# run_ablation.sh — Sequential ablation sweep E1-E7 + E1b-E4b on 10% data
# Usage: bash run_ablation.sh [SWEEP_NAME]
#   SWEEP_NAME defaults to "sweep" if not provided
#
# Ref: MATH.md M.3, TZ.md Sec 6
# Each experiment runs 3 epochs on 10% of data with batch_size=64.
# Results go to runs/ (TensorBoard), checkpoints/ (best models).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sweep name for log prefix (e.g., sweep5)
SWEEP_NAME="${1:-sweep}"

# Activate venv
source "$SCRIPT_DIR/venv/bin/activate"

# Ablation overrides
# bs=64: safe for RTX 3090 24GB with 5×SciRus-tiny + patch-based visual pipeline.
# bs=128 OOMs on worst-case batches (W=1200, K=37 patches × 128 = 4736 ResNet forwards + 5 transformer backwards).
OVERRIDES="data.dataset_fraction=0.2 data.batch_size=64 data.num_workers=8 training.epochs=5 eval.eval_every_steps=200"

# Family A experiments
EXPERIMENTS_A=(
    e1_pairwise
    e2_centroid
    e3_centroid_reg
    e4_composite_static
    e5_composite_learnable
    e6_fuzzy
    e7_lyapunov
    e8_nonlinear
    e9_potential
    e10_potential_fuzzy
)

# Family B experiments
EXPERIMENTS_B=(
    e1b_pairwise
    e2b_centroid
    e3b_centroid_reg
    e4b_composite_static
)

# Family C experiments (calibration variants, EXP-005)
EXPERIMENTS_C=(
    e3c_low_va
    e6c_low_va
    e10c_low_va
    e8c_active
)

# EXP-006: rho=0.3, combos, contrast boost
EXPERIMENTS_D=(
    e6_rho03
    e8c_rho03
    e8c_low_va
    e8c_rho03_low_va
    e8c_boost
)

ALL_EXPERIMENTS=("${EXPERIMENTS_A[@]}" "${EXPERIMENTS_B[@]}" "${EXPERIMENTS_C[@]}" "${EXPERIMENTS_D[@]}")

# Create logs dir
mkdir -p logs

# Start TensorBoard for live monitoring
bash "$SCRIPT_DIR/start_tensorboard.sh" 2>/dev/null || true
echo ""

echo "=== SciLibMath_v2 Ablation Sweep ($SWEEP_NAME, 10% data, 3 epochs) ==="
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
