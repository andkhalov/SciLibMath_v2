#!/bin/bash
# run_weep3b.sh — Sequential sweep E1-E4 Family B: weep-3b
# 50k train objects, 10k eval objects, 1 epoch
# Family B: 1 shared text encoder + 1 visual encoder (~28M params)
# E5-E7 excluded (fuzzy controller requires separate modality weights)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source "$SCRIPT_DIR/venv/bin/activate"

# Overrides: 60k objects (50k train + 10k eval), 1 epoch, bs=64
OVERRIDES="data.dataset_fraction=0.0617 data.test_fraction=0.167 data.batch_size=64 data.num_workers=8 training.epochs=1 eval.eval_every_steps=100"

EXPERIMENTS=(
    e1b_pairwise
    e2b_centroid
    e3b_centroid_reg
    e4b_composite_static
)

# Ensure directories exist
mkdir -p logs checkpoints

# Start TensorBoard for live monitoring
bash "$SCRIPT_DIR/start_tensorboard.sh" 2>/dev/null || true
echo ""

echo "=== SciLibMath_v2 WEEP-3B Sweep (Family B) ==="
echo "  50k train / 10k eval / 1 epoch / shared encoder (~28M params)"
echo "  Experiments: ${EXPERIMENTS[*]}"
echo "  Overrides: $OVERRIDES"
echo "  Start: $(date -Iseconds)"
echo ""

FAILED=()

for exp in "${EXPERIMENTS[@]}"; do
    echo "========================================"
    echo "  Running: $exp (Family B)"
    echo "  Config: configs/${exp}.yaml"
    echo "  Time: $(date -Iseconds)"
    echo "========================================"

    if python code/train.py --config "configs/${exp}.yaml" $OVERRIDES 2>&1 | tee "logs/weep3b_${exp}.log"; then
        echo ""
        echo "  $exp DONE at $(date -Iseconds)"
    else
        echo ""
        echo "  $exp FAILED at $(date -Iseconds)"
        FAILED+=("$exp")
    fi
    echo ""
done

echo "=== WEEP-3B Sweep Complete ==="
echo "  Finish: $(date -Iseconds)"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED experiments: ${FAILED[*]}"
else
    echo "  All 4 Family B experiments passed"
fi
echo "  TensorBoard: http://localhost:14714"
echo "  Logs: logs/weep3b_*.log"
