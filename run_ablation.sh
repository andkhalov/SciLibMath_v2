#!/bin/bash
# run_ablation.sh — Sequential ablation sweep E1-E7 on 10% data
# Usage: bash run_ablation.sh
#
# Ref: MATH.md M.3, TZ.md Sec 6
# Each experiment runs 3 epochs on 10% of data with batch_size=64.
# Results go to runs/ (TensorBoard), checkpoints/ (best models).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source "$SCRIPT_DIR/venv/bin/activate"

# Ablation overrides
# bs=64: safe for RTX 3090 24GB with 5×SciRus-tiny + patch-based visual pipeline.
# bs=128 OOMs on worst-case batches (W=1200, K=37 patches × 128 = 4736 ResNet forwards + 5 transformer backwards).
OVERRIDES="data.dataset_fraction=0.1 data.batch_size=64 data.num_workers=8 training.epochs=3 eval.eval_every_steps=200"

EXPERIMENTS=(
    e1_pairwise
    e2_centroid
    e3_centroid_reg
    e4_composite_static
    e5_composite_learnable
    e6_fuzzy
    e7_lyapunov
)

# Start TensorBoard for live monitoring
bash "$SCRIPT_DIR/start_tensorboard.sh"
echo ""

echo "=== SciLibMath_v2 Ablation Sweep (10% data, 3 epochs) ==="
echo "Experiments: ${EXPERIMENTS[*]}"
echo "Overrides: $OVERRIDES"
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    echo "========================================"
    echo "  Running: $exp"
    echo "  Config: configs/${exp}.yaml"
    echo "  Time: $(date -Iseconds)"
    echo "========================================"

    python code/train.py --config "configs/${exp}.yaml" $OVERRIDES 2>&1 | tee "logs/${exp}_ablation.log"

    echo ""
    echo "  $exp DONE at $(date -Iseconds)"
    echo ""
done

echo "=== All experiments complete ==="
echo "TensorBoard still running on port 14714 — stop with: bash stop_tensorboard.sh"
echo "Run: python code/analyze_ablation.py to generate summary table"
