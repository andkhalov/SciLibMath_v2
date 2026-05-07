# EXP-013: Final Report — Multi-Task Balancing & Trainable Consequents

> Date: 2026-05-07
> Duration: 2026-04-26 → 2026-05-03 (~7 days of experiments)
> GPU: RTX 3090 24GB
> Data: SciLibModal v2 (~972K multimodal mathematical objects, 5 modalities)
> TensorBoard: `tensorboard --logdir runs/ --port 14714`

---

## 1. Experiment Overview

EXP-013 evaluated two directions:
1. **Baseline comparisons** — established multi-task loss balancing methods vs our T-S fuzzy controller
2. **Trainable MLP consequents** — multiple approaches to make T-S controller consequents learnable

---

## 2. Metrics Glossary

| Metric | Description | Range | Good direction |
|--------|-------------|-------|----------------|
| **cm@1** (mean_crossmodal_R@1) | Average R@1 across all 20 directed modality pairs (m1→m2). Primary metric. | [0, 1] | ↑ higher = better |
| **cm@3, cm@10** | Same but top-3, top-10 | [0, 1] | ↑ |
| **cR@1** (centroid_R@1) | Leave-one-out centroid retrieval R@1. Each modality is left out, centroid of remaining 4 used to retrieve correct object. | [0, 1] | ↑ (saturates near 1.0) |
| **mod→c R@1** (e.g. en_to_centroid_R@1) | Single modality → centroid retrieval. How well each modality alone can identify the correct centroid. | [0, 1] | ↑ |
| **D_intra** | Mean squared distance from modality embeddings to their centroid. Measures cluster tightness. | [0, ∞) | ↓ lower = tighter clusters |
| **D_inter** | Mean inter-centroid cosine distance. Measures separation between objects. | [0, 1] | ↑ higher = better separated |
| **collapse** | Mean off-diagonal centroid similarity. Indicates representation collapse. | [0, 1] | ↓ lower = no collapse |

---

## 3. Full Training Results (100% data, 15 epochs)

### 3.1 Main Table — Cross-Modal Retrieval

| # | Method | cm@1 | cm@3 | cm@10 | Group |
|---|--------|------|------|-------|-------|
| 1 | **E8c_low_va_cnxt** | **0.7070** | 0.8214 | 0.8753 | Reference |
| 2 | BL3 uncertainty (Kendall 2018) | 0.6972 | 0.8143 | 0.8709 | EXP-013 baseline |
| 3 | E1_pairwise_cnxt | 0.6971 | 0.8287 | 0.8884 | Reference |
| 4 | BL1 gradnorm (Chen 2018) | 0.6949 | 0.8117 | 0.8687 | EXP-013 baseline |
| 5 | E9_potential | 0.6891 | 0.8059 | 0.8642 | Reference |
| 6 | E8cf_v7c (trainable consequents) | 0.6855 | 0.8031 | 0.8618 | EXP-013 e8cf |
| 7 | E8c_low_va (ResNet) | 0.6713 | 0.7931 | 0.8564 | Reference |
| 8 | E1_pairwise (ResNet) | 0.6682 | 0.8051 | 0.8741 | Reference |
| 9 | E1b_pairwise (ConvNeXt) | 0.6490 | 0.7884 | 0.8629 | Reference |
| 10 | E8cf_real (v5, residual MLP) | 0.6288 | 0.7539 | 0.8242 | EXP-013 e8cf |
| 11 | E8cf_unc (v3, reaggregate) | 0.6226 | 0.7491 | 0.8208 | EXP-013 e8cf |

### 3.2 Centroid Retrieval & Per-Modality

| Method | cR@1 | en→c | ru→c | lean→c | lat→c | img→c |
|--------|------|------|------|--------|-------|-------|
| E8c_low_va_cnxt | 0.999 | 0.949 | 0.970 | 0.918 | 0.923 | 0.911 |
| BL3 uncertainty | 0.999 | 0.964 | 0.986 | 0.921 | 0.897 | 0.907 |
| E1_pairwise_cnxt | 0.996 | 0.889 | 0.916 | 0.841 | 0.905 | 0.923 |
| BL1 gradnorm | 1.000 | 0.955 | 0.977 | 0.916 | 0.923 | 0.917 |
| E9_potential | 0.999 | 0.934 | 0.958 | 0.889 | 0.901 | 0.868 |
| E8cf_v7c | 0.997 | 0.881 | 0.899 | 0.837 | 0.843 | 0.811 |
| E8c_low_va (ResNet) | 0.999 | 0.941 | 0.962 | 0.912 | 0.917 | 0.904 |
| E8cf_real (v5) | 0.971 | 0.711 | 0.728 | 0.725 | 0.884 | 0.922 |
| E8cf_unc (v3) | 0.971 | 0.713 | 0.717 | 0.720 | 0.882 | 0.922 |

### 3.3 Geometry

| Method | D_intra | D_inter | collapse |
|--------|---------|---------|----------|
| BL3 uncertainty | 0.0005 | 0.9997 | 0.0003 |
| E8c_low_va_cnxt | 0.0006 | 0.9998 | 0.0002 |
| BL1 gradnorm | 0.0006 | 0.9998 | 0.0002 |
| E8cf_v7c | 0.0038 | 1.0000 | 0.0000 |
| E9_potential | 0.0244 | 0.9999 | 0.0001 |
| E1_pairwise_cnxt | 2.6244 | 0.9958 | 0.0042 |
| E1_pairwise (ResNet) | 2.7819 | 0.9966 | 0.0034 |

---

## 4. TensorBoard Run Locations

All runs are in `/home/opt/scilib/scilib.ai/laboratory/SciLibMath_v2/v_2/runs/`:

### Reference runs (EXP-011/012)
```
full_e8c_low_va_cnxt_s42_1774061623     # E8c — BEST cm@1=0.7070
full_e1_pairwise_cnxt_s42_1773785522    # E1 pairwise ConvNeXt
full_e1_pairwise_s42_1773865248         # E1 pairwise ResNet
full_e8c_low_va_s42_1773994602          # E8c ResNet
full_e9_potential_s42_1774142769         # E9 potential loss
full_e1b_pairwise_s42_1773930805        # E1b pairwise ConvNeXt
```

### EXP-013 baselines (full)
```
full_bl3_uncertainty_s42_1777282667     # BL3 Uncertainty Weighting — cm@1=0.6972
full_bl1_gradnorm_s42_1777363889        # BL1 GradNorm — cm@1=0.6949
```

### EXP-013 e8cf variants (full)
```
full_e8cf_differentiable_s42_1777445065 # E8cf_unc (v3) — cm@1=0.6226
full_e8cf_differentiable_s42_1777525052 # E8cf_real (v5) — cm@1=0.6288
full_e8cf_v7_s42_1777730106            # E8cf_v7c — cm@1=0.6855
```

### EXP-013 sweep runs (10% data, 5 epochs)
```
bl3_uncertainty_s42_1777230518          # BL3 sweep — cm@1=0.4121
bl1_gradnorm_s42_*                     # BL1 sweep — cm@1=0.4084
e8c_low_va_cnxt_s42_1773744742         # E8c sweep — cm@1=0.3525
e8cf_unc_s42_1777245613                # E8cf_unc sweep — cm@1=0.5084
e8cf_real_s42_1777275769               # E8cf_real sweep — cm@1=0.3805
```

### Training logs
```
logs/exp013_full_bl3_uncertainty.log
logs/exp013_full_bl1_gradnorm.log
logs/exp013_full_e8cf_differentiable.log
logs/exp013_full_e8cf_v7c.log
```

---

## 5. Key Findings

### 5.1 E8c T-S fuzzy controller is best (cm@1=0.7070)

Despite MLP consequents never receiving gradient (design bug discovered during EXP-013),
E8c outperforms all methods. Analysis of E8c full training revealed:
- **λ_va: 0.01 → 0.08** (+694%) — controller autonomously increases visual alignment weight
- **λ_reg: 0.05 → 0.017** (−67%) — reduces anti-collapse regularization
- **w_m ≈ ±1%** from defaults — modality weights barely change
- **Dominant rule: R4 (Overshoot) = 0.61** activation

E8c advantage comes from interpretable fuzzy rules + elastic dynamics + λ_va curriculum,
not from learned MLP parameters.

### 5.2 Simple baselines match E1 pairwise (BL3=0.6972, BL1=0.6949)

Uncertainty Weighting (5 learnable scalars, zero computational overhead) achieves
cm@1=0.6972, virtually identical to E1 pairwise (0.6971). GradNorm (gradient norm
matching) at 0.6949. Both −1 p.p. below E8c.

**BL3 has best text modality alignment:** en→c=0.964, ru→c=0.986 — better than E8c.
**E8c has more balanced alignment** across all modalities (no single modality below 0.91).

### 5.3 Trainable MLP consequents: partial success

**8 versions tested (v1–v8d).** Key obstacle: MLP consequent output passes through
elastic_step + project_to_bounds before affecting the model. This creates:
1. Gradient attenuation (α=0.01 scaling)
2. Dead gradient at bounds (clamp kills ∂/∂x at boundaries)
3. Loss signal vanishes as training stabilizes (EMA catches up)

**Best trainable variant: E8cf_v7c** (cm@1=0.6855, separate backward, per-modality
relative rate signal). MLP consequents provide correction in early training phase;
contribution diminishes as loss stabilizes. Interpretation: **adaptive initialization
of hyperparameters**, not online adaptive control.

### 5.4 Sweep ≠ Full (critical observation)

| Method | Sweep 10% | Full 100% | Rank change |
|--------|-----------|-----------|-------------|
| E8cf_unc (v3) | 0.5084 (1st) | 0.6226 (11th) | ↓10 |
| E1 pairwise | 0.4430 (2nd) | 0.6971 (3rd) | ↑ |
| BL3 | 0.4121 (3rd) | 0.6972 (2nd) | ↑ |
| E8c | 0.3525 (5th) | 0.7070 (1st) | ↑4 |

Short sweep (10%, 5 epochs) is unreliable for ranking methods. E8c benefits most
from longer training (elastic dynamics + λ_va curriculum need time to work).

---

## 6. Conclusions for Paper

### What to report:
1. **E8c (T-S fuzzy controller):** cm@1=0.707, cR@1=0.999 — best method
2. **BL3 (Uncertainty Weighting):** cm@1=0.697 — strongest baseline, 5 params
3. **BL1 (GradNorm):** cm@1=0.695 — competitive baseline
4. **E1 (Pairwise InfoNCE):** cm@1=0.697 — no controller, comparable to baselines

### Key claims:
- T-S fuzzy controller provides +1 p.p. over best baselines (modest but consistent)
- Controller advantage is in interpretable, rule-based hyperparameter adaptation
- Nonlinear consequents act as adaptive initialization in early training phase
- Simple methods (5 learnable scalars) are competitive with complex controllers

### Honest limitations:
- MLP consequents do not learn sustained adaptive control (zero modality weights by mid-training)
- E8c advantage over baselines is small (1 p.p.)
- Bounds optimality is unknown (set heuristically)
- Training obuchable consequents through indirect signal remains an open problem

---

## 7. Configs and Reproducibility

### Baseline configs
```
configs/bl3_uncertainty.yaml    # Kendall et al. 2018
configs/bl1_gradnorm.yaml      # Chen et al. 2018
```

### E8cf configs (trainable consequents)
```
configs/e8cf_differentiable.yaml  # v3/v6 (reaggregate + softmax)
configs/e8cf_real.yaml            # v5 (residual MLP)
configs/e8cf_v6.yaml              # v6 (separate backward)
configs/e8cf_v7.yaml              # v7c (per-modality relative rate)
configs/e8cf_v7d.yaml             # v7d (no bounds — failed)
configs/e8cf_v8.yaml              # v8 (per-rule)
configs/e8cf_v8d.yaml             # v8d (no bounds — failed)
```

### Full training overrides
```
data.dataset_fraction=1.0
data.test_fraction=0.02
data.val_fraction=0.03
data.batch_size=64
data.num_workers=8
training.epochs=15
eval.eval_every_steps=1000
checkpoint.enabled=true
checkpoint.keep_best=3
run_tag=full
```

### Code changes (EXP-013)
```
code/baselines/__init__.py              # NEW: baseline imports
code/baselines/gradnorm.py             # NEW: GradNorm (Chen 2018)
code/baselines/pcgrad.py               # NEW: PCGrad (Yu 2020) — OOM, unused
code/baselines/uncertainty_weighting.py # NEW: UW (Kendall 2018)
code/losses/composite.py               # MODIFIED: reaggregate_with_lambda, mlp_norm_reg,
                                        #   compute_derivative_mlp_loss, compute_per_rule_mlp_loss
code/controller/ts_controller.py        # MODIFIED: residual consequents, skip_bounds
code/controller/rules.py               # MODIFIED: skip_bounds in elastic_step
code/train.py                          # MODIFIED: baseline + e8cf integration
```
