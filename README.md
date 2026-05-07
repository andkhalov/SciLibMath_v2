# SciLibMath v2

Multimodal contrastive learning for mathematical objects.

Five modalities (English, Russian, Lean 4, LaTeX, formula images) aligned in a shared embedding space with centroid geometry and a Takagi-Sugeno fuzzy controller for adaptive hyperparameter management.

## Key Results

| Method | cm@1 | cR@1 | D_intra |
|--------|------|------|---------|
| **E8c T-S fuzzy controller** | **0.707** | **0.999** | 0.0006 |
| BL3 Uncertainty Weighting | 0.697 | 0.999 | 0.0005 |
| E1 Pairwise InfoNCE | 0.697 | 0.996 | 2.624 |
| BL1 GradNorm | 0.695 | 1.000 | 0.0006 |
| E9 Potential loss | 0.689 | 0.999 | 0.024 |

Full training: 972K objects, 15 epochs, RTX 3090. Best model checkpoint: [S3](https://s3.scilibai.ru/scilibmath-v2-checkpoints/e8c_low_va_cnxt/best_model.pt)

**Metrics:**
- **cm@1** (mean cross-modal R@1): average retrieval accuracy across all 20 directed modality pairs
- **cR@1** (centroid R@1): leave-one-out centroid retrieval (4 modalities reconstruct centroid, retrieve correct object)
- **D_intra**: mean squared distance from modality embeddings to centroid (lower = tighter clusters)

---

## Dataset

**SciLibRuModal v2** — 972,711 multimodal mathematical objects from Mathlib (Lean 4).

| Modality | Description | Example |
|----------|-------------|---------|
| `en` | English description | "Every natural number greater than 1 has a prime divisor" |
| `ru` | Russian description | "Каждое натуральное число больше 1 имеет простой делитель" |
| `lean` | Lean 4 formal statement | `theorem Nat.exists_prime_and_dvd ...` |
| `latex` | LaTeX notation | `$\forall n > 1, \exists p \text{ prime}, p \mid n$` |
| `img` | Rendered formula image | PNG render of LaTeX |

Auto-download via `init.sh`.

---

## Architecture

### Family A — 5 Separate Encoders

Each modality has a dedicated encoder projecting into a shared 256-dimensional space:

| Modality | Backbone | Params |
|----------|----------|--------|
| `en`, `ru` | SciRus-tiny 3.5 (ModernBERT, 312d) + linear projection | ~16M each |
| `lean` | SciRus-tiny + custom BPE tokenizer (16K vocab) | ~16M |
| `latex` | SciRus-tiny + custom BPE tokenizer (16K vocab) | ~16M |
| `img` | ConvNeXt-Pico (patch-based) + AlignNet | ~9M |

Total: ~59M trainable parameters.

### Centroid Geometry

Each object is represented as a **region** in embedding space, not a point:

```
c_i = (1/M) Σ_m e_i^m
```

The centroid serves as the invariant anchor for contrastive learning across M modalities.

### T-S Fuzzy Controller (E8c)

Observes training state s_t ∈ R^18 (loss values, trends, per-modality EMAs, collapse indicators) and adjusts 11 hyperparameters via 7 interpretable rules:

| Rule | Condition | Action |
|------|-----------|--------|
| R0 Healthy | Loss low, trend decreasing | No correction |
| R1 Imbalance | High modality loss variance | Rebalance weights |
| R2 Collapse | High collapse score | Increase τ, increase λ_reg |
| R3 Stagnation | High loss, stable trend | Increase λ_align, decrease τ |
| R4 Overshoot | Loss increasing | Soften (increase τ) |
| R5 Conflict | High gradient conflict proxy | Increase λ_reg |
| R6 Visual | Image loss high and stable | Increase λ_va |

Residual nonlinear MLP consequents (per-rule) provide adaptive correction in early training phase.

---

## Training Modes

### Without controller

| ID | Loss | Description |
|----|------|-------------|
| E1 | Pairwise InfoNCE | CLIP-style, all C(5,2)=10 modality pairs |
| E2 | Centroid InfoNCE | Centroid as anchor |
| E3 | Centroid + Reg | E2 + alignment + radial regularization |
| E4 | Composite Static | Full loss, static weights |

### With T-S controller

| ID | Loss | Description |
|----|------|-------------|
| E6 | Fuzzy T-S | Linear consequents, stochastic exploration |
| E7 | + Lyapunov | E6 + Lyapunov stability constraint |
| **E8c** | **Nonlinear T-S** | **MLP consequents, best overall (cm@1=0.707)** |
| E9 | Potential | Attraction/repulsion potential loss |

### Baselines (EXP-013)

| ID | Method | Reference |
|----|--------|-----------|
| BL1 | GradNorm | Chen et al., ICML 2018 |
| BL3 | Uncertainty Weighting | Kendall et al., CVPR 2018 |

---

## Full Training Results

100% data, 15 epochs, seed=42, RTX 3090.

### Cross-Modal Retrieval

| Method | cm@1 | cm@3 | cm@10 |
|--------|------|------|-------|
| E8c_low_va_cnxt | **0.707** | 0.821 | 0.875 |
| BL3 uncertainty | 0.697 | 0.814 | 0.871 |
| E1_pairwise_cnxt | 0.697 | 0.829 | **0.888** |
| BL1 gradnorm | 0.695 | 0.812 | 0.869 |
| E9 potential | 0.689 | 0.806 | 0.864 |

### Per-Modality → Centroid R@1

| Method | en→c | ru→c | lean→c | latex→c | img→c |
|--------|------|------|--------|---------|-------|
| E8c | 0.949 | 0.970 | 0.918 | 0.923 | 0.911 |
| BL3 | **0.964** | **0.986** | **0.921** | 0.897 | 0.907 |
| BL1 | 0.955 | 0.977 | 0.916 | **0.923** | 0.917 |
| E1 | 0.889 | 0.916 | 0.841 | 0.905 | **0.923** |

---

## Quick Start

```bash
# Initialize (venv + download dataset from S3)
bash init.sh

# Activate environment
source venv/bin/activate

# Train best model (E8c with ConvNeXt)
python code/train.py --config configs/e8c_low_va_cnxt.yaml

# Train pairwise baseline
python code/train.py --config configs/e1_pairwise_cnxt.yaml

# Train with Uncertainty Weighting baseline
python code/train.py --config configs/bl3_uncertainty.yaml

# 10% data sweep (fast evaluation)
python code/train.py --config configs/e8c_low_va_cnxt.yaml \
  data.dataset_fraction=0.1 training.epochs=5 eval.eval_every_steps=50
```

## Pre-trained Checkpoint

Best model (E8c_low_va_cnxt, cm@1=0.707):

```bash
# Download from S3
rclone copy scilib-store:scilibmath-v2-checkpoints/e8c_low_va_cnxt/best_model.pt checkpoints/e8c_low_va_cnxt/

# Or direct URL
wget https://s3.scilibai.ru/scilibmath-v2-checkpoints/e8c_low_va_cnxt/best_model.pt -P checkpoints/e8c_low_va_cnxt/
```

## Project Structure

```
v_2/
├── code/
│   ├── train.py              # Main training loop
│   ├── evaluate.py           # Standalone evaluation
│   ├── models/               # Family A/B architectures
│   ├── losses/               # InfoNCE, composite, potential losses
│   ├── controller/           # T-S fuzzy controller, Lyapunov
│   ├── baselines/            # GradNorm, UW, PCGrad
│   ├── metrics/              # Retrieval, geometry metrics
│   └── experiment_logging/   # TensorBoard + S3 backup
├── configs/                  # YAML configs (E1–E10, BL1–BL3, E8cf variants)
├── data/                     # SciLibRuModal v2 dataset loader
├── exp_reports/              # 13 experiment reports
├── writing/                  # Documentation (MATH.md, LIT.md, etc.)
├── requirements.txt
└── init.sh                   # Setup script
```

## Requirements

- Python 3.11+
- PyTorch 2.0+ with CUDA
- GPU: NVIDIA RTX 3090 (24GB) or equivalent
- Full list: `requirements.txt`

## Citation

```bibtex
@misc{scilibmath2026,
  title={SciLibMath v2: Multimodal Contrastive Learning for Mathematical Objects with Fuzzy Adaptive Control},
  author={Khalov, A. P.},
  year={2026},
  url={https://github.com/andkhalov/SciLibMath_v2}
}
```

## License

MIT
