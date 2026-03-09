# SciLibMath_v2

Multimodal contrastive learning for mathematical objects across 5 modalities (en, ru, lean, latex, img) with centroid geometry, fuzzy T-S controller, and Lyapunov stability constraints.

## Quick Start

```bash
# Setup environment
bash init.sh

# Activate
source venv/bin/activate

# Run experiment (e.g. E4: composite loss, static weights)
python code/train.py --config configs/e4_composite_static.yaml
```

## Experiments

| ID | Name | Description |
|----|------|-------------|
| E1 | Pairwise InfoNCE | CLIP-style baseline, all C(5,2)=10 pairs |
| E2 | Centroid InfoNCE | Centroid as anchor |
| E3 | Centroid + Reg | + alignment + radial regularization |
| E4 | Composite Static | Full loss, static weights |
| E5 | LossMixer | MLP-learned weights |
| E6 | Fuzzy T-S | Fuzzy controller for weights |
| E7 | + Lyapunov | E6 + Lyapunov stability constraint |

## Formalization

Mathematical specification: `writing/MATH.md` v2.6.2 (frozen reference).

## Dataset

SciLibRuModal v2: 972,711 multimodal mathematical objects.
Auto-downloaded by `init.sh` from S3.

## Hardware

- GPU: NVIDIA RTX 3090 (24GB)
- CUDA 12.8, PyTorch 2.10
