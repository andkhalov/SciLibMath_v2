#!/bin/bash
# run_weep3.sh — Sequential sweep E1-E7: weep-3
# 50k train objects, 10k eval objects, 1 epoch, full fine-tuning
#
# Bug fixes applied:
#   B1: LOO centroid R@k (no self-retrieval)
#   B2: "ru" in weight_norm logging
#   B3: Fuzzy controller z-score normalization + recalibrated MFs + alpha=0.05
#   B4: full fine-tuning (112M trainable params)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source "$SCRIPT_DIR/venv/bin/activate"

# Overrides: 60k objects (50k train + 10k eval), 1 epoch, bs=64
OVERRIDES="data.dataset_fraction=0.0617 data.test_fraction=0.167 data.batch_size=64 data.num_workers=8 training.epochs=1 eval.eval_every_steps=100"

EXPERIMENTS=(
    e1_pairwise
    e2_centroid
    e3_centroid_reg
    e4_composite_static
    e5_composite_learnable
    e6_fuzzy
    e7_lyapunov
)

# Ensure directories exist
mkdir -p logs checkpoints

# Start TensorBoard for live monitoring
bash "$SCRIPT_DIR/start_tensorboard.sh" 2>/dev/null || true
echo ""

echo "=== SciLibMath_v2 WEEP-3 Sweep ==="
echo "  50k train / 10k eval / 1 epoch / all weights unfrozen (112M params)"
echo "  Experiments: ${EXPERIMENTS[*]}"
echo "  Overrides: $OVERRIDES"
echo "  Start: $(date -Iseconds)"
echo ""

FAILED=()

for exp in "${EXPERIMENTS[@]}"; do
    echo "========================================"
    echo "  Running: $exp"
    echo "  Config: configs/${exp}.yaml"
    echo "  Time: $(date -Iseconds)"
    echo "========================================"

    if python code/train.py --config "configs/${exp}.yaml" $OVERRIDES 2>&1 | tee "logs/weep3_${exp}.log"; then
        echo ""
        echo "  $exp DONE at $(date -Iseconds)"
    else
        echo ""
        echo "  $exp FAILED at $(date -Iseconds)"
        FAILED+=("$exp")
    fi
    echo ""
done

echo "=== WEEP-3 Sweep Complete ==="
echo "  Finish: $(date -Iseconds)"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED experiments: ${FAILED[*]}"
else
    echo "  All 7 experiments passed"
fi
echo "  TensorBoard: http://localhost:14714"
echo "  Logs: logs/weep3_*.log"
