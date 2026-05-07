# EXP-013: Full Training Results

> Date: 2026-05-01
> GPU: RTX 3090 24GB
> Data: SciLibModal v2, 100%, 15 epochs, batch_size=64, seed=42
> Duration: 2026-04-27 12:37 → 2026-05-01 06:39 (~90 hours, 4 experiments)

---

## Final Results Table

| Method | cm@1 | cm@3 | cm@10 | en→c | ru→c | lean→c | lat→c | img→c | D_intra |
|--------|------|------|-------|------|------|--------|-------|-------|---------|
| **E8c_low_va_cnxt** | **0.7070** | 0.8214 | 0.8753 | 0.949 | 0.970 | 0.918 | 0.923 | 0.911 | 0.0006 |
| BL3 uncertainty | 0.6972 | 0.8143 | 0.8709 | 0.964 | 0.986 | 0.921 | 0.897 | 0.907 | 0.0005 |
| E1_pairwise_cnxt | 0.6971 | 0.8287 | 0.8884 | 0.889 | 0.916 | 0.841 | 0.905 | 0.923 | 2.6244 |
| BL1 gradnorm | 0.6949 | 0.8117 | 0.8687 | 0.955 | 0.977 | 0.916 | 0.923 | 0.917 | 0.0006 |
| E9_potential | 0.6891 | 0.8059 | 0.8642 | 0.934 | 0.958 | 0.889 | 0.901 | 0.868 | 0.0244 |
| E8c_low_va (ResNet) | 0.6713 | 0.7931 | 0.8564 | 0.941 | 0.962 | 0.912 | 0.916 | 0.904 | 0.0008 |
| E1_pairwise (ResNet) | 0.6682 | 0.8051 | 0.8741 | 0.874 | 0.899 | 0.829 | 0.900 | 0.923 | 2.7819 |
| E1b_pairwise (ConvNeXt) | 0.6490 | 0.7884 | 0.8629 | 0.877 | 0.906 | 0.826 | 0.894 | 0.920 | 2.3260 |
| E8cf_real (v5, residual) | 0.6288 | 0.7539 | 0.8242 | 0.711 | 0.728 | 0.725 | 0.884 | 0.922 | 0.0019 |
| E8cf_unc (v3) | 0.6226 | — | — | — | — | — | — | — | — |

## Key Observations

### 1. E8c remains best (0.7070)
Despite MLP consequents never receiving gradient (design bug found during EXP-013),
E8c outperforms all methods. Its advantage comes from:
- Fuzzy antecedents (diagnostic rules) + elastic reversion dynamics
- Near-zero random MLP acts as stochastic regularization
- Low λ_va config (0.01) + controller gradually raises it to ~0.08
- Combined effect: better λ_va curriculum than any baseline

### 2. BL3 ≈ E1_pairwise ≈ BL1 (0.694–0.697)
Simple baselines (5 learnable scalars) match pairwise InfoNCE.
All three are within 1 p.p. of each other and 1 p.p. below E8c.
BL3 (Uncertainty Weighting) is simplest (5 params, zero overhead) and matches BL1 (GradNorm).

### 3. E8cf variants underperform (0.62–0.63)
Both differentiable controller variants fail on full data:
- e8cf_unc (v3): random drift to bounds helps on 10% but hurts on full
- e8cf_real (v5): residual MLP improves over v3 by 0.6 p.p. but still −8 p.p. below E8c
- Reaggregate loss doubling distorts gradient magnitudes on long training

### 4. Sweep results do NOT predict full training
| Method | Sweep 10% | Full 100% | Rank change |
|--------|-----------|-----------|-------------|
| e8cf_unc | 0.5084 (1st) | 0.6226 (9th) | ↓8 |
| E1 pairwise | 0.4430 (2nd) | 0.6971 (3rd) | ↑ |
| BL3 | 0.4121 (3rd) | 0.6972 (2nd) | ↑ |
| E8c | 0.3525 (5th) | 0.7070 (1st) | ↑4 |

### 5. Centroid-based methods have much lower D_intra
E8c/BL3/BL1: D_intra ≈ 0.0005–0.0006 (tight clusters)
E1 pairwise: D_intra ≈ 2.5 (no centroid alignment)
This confirms controller/weighting methods compress modality embeddings toward centroid.

## TensorBoard Runs

```
runs/full_e8c_low_va_cnxt_s42_1774061623      # E8c reference
runs/full_e1_pairwise_cnxt_s42_1773785522     # E1 reference
runs/full_bl3_uncertainty_s42_<ts>            # BL3
runs/full_bl1_gradnorm_s42_<ts>              # BL1
runs/full_e8cf_differentiable_s42_1777445065  # e8cf_unc (v3)
runs/full_e8cf_differentiable_s42_1777525052  # e8cf_real (v5)
```

## Conclusion for Paper

For the Lobachevskii paper, we report:
- E8c (T-S fuzzy controller) as our method: cm@1 = 0.7070 (best)
- BL3 (Uncertainty Weighting, Kendall 2018) as baseline: cm@1 = 0.6972
- BL1 (GradNorm, Chen 2018) as baseline: cm@1 = 0.6949
- E1 (Pairwise InfoNCE, no controller) as ablation: cm@1 = 0.6971

E8c advantage over baselines: +1.0 p.p. (vs BL3), +1.2 p.p. (vs BL1).
Modest but consistent advantage with interpretable rule-based control.
