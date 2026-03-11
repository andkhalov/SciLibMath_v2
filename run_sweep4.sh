#!/bin/bash
# run_sweep4.sh — Honest sweep: 5% data, 3 epochs, E1-E7 (Family A) + E1b-E4b (Family B)
# Ref: writing/HYPOTHESIS.md, writing/TZ.md
# Expected: ~4 hours total on RTX 3090

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source venv/bin/activate

OVERRIDES="data.dataset_fraction=0.05 data.test_fraction=0.167 data.batch_size=64 data.num_workers=8 training.epochs=3 eval.eval_every_steps=200"

# Family A experiments
CONFIGS_A=(
    e1_pairwise
    e2_centroid
    e3_centroid_reg
    e4_composite_static
    e5_composite_learnable
    e6_fuzzy
    e7_lyapunov
)

# Family B experiments
CONFIGS_B=(
    e1b_pairwise
    e2b_centroid
    e3b_centroid_reg
    e4b_composite_static
)

echo "=== SciLibMath_v2 Sweep 4: 5% data, 3 epochs ==="
echo "Date: $(date -Iseconds)"
echo "Overrides: $OVERRIDES"
echo ""

TOTAL=$((${#CONFIGS_A[@]} + ${#CONFIGS_B[@]}))
CURRENT=0
FAILED=()

run_experiment() {
    local config=$1
    local log_file="logs/sweep4_${config}.log"
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TOTAL] Running $config ..."
    if python code/train.py --config "configs/${config}.yaml" $OVERRIDES > "$log_file" 2>&1; then
        echo "  OK ($config)"
    else
        echo "  FAILED ($config) — see $log_file"
        FAILED+=("$config")
    fi
}

mkdir -p logs

# Family A
echo "--- Family A (7 experiments) ---"
for cfg in "${CONFIGS_A[@]}"; do
    run_experiment "$cfg"
done

# Family B
echo ""
echo "--- Family B (4 experiments) ---"
for cfg in "${CONFIGS_B[@]}"; do
    run_experiment "$cfg"
done

# Summary
echo ""
echo "=== Sweep 4 Complete ==="
echo "Date: $(date -Iseconds)"
echo "Total: $TOTAL experiments"
echo "Failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed experiments: ${FAILED[*]}"
fi
echo "TensorBoard: tensorboard --logdir runs/"
