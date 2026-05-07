# EXP-013: Multi-Task Balancing Baselines (GradNorm, PCGrad, Uncertainty Weighting)

> Date: 2026-04-26
> Status: PLANNED
> GPU: RTX 3090 24GB
> Branch: dev (SciLibMath_v2/v_2 local repo)

---

## Goal

Implement and evaluate three established multi-task loss balancing methods as baselines
for comparison with our T-S fuzzy controller (E8c). Needed for Lobachevskii Journal paper
to provide direct experimental comparison (not just literature references).

## Baselines

| ID | Method | Reference | Mechanism |
|----|--------|-----------|-----------|
| BL1 | GradNorm | Chen et al. ICML 2018 | Gradient norm matching → adaptive w_m |
| BL2 | PCGrad | Yu et al. NeurIPS 2020 | Gradient surgery (projection of conflicting gradients) |
| BL3 | Uncertainty Weighting | Kendall et al. CVPR 2018 | Learnable log-variance → adaptive w_m |

## Experimental Setup

### Shared Configuration (same as E8c sweep13)

- **Loss:** Composite (same structure as E8c: global contrast + align + rad + reg + per-modality personal losses + visual alignment)
- **Static hyperparameters (from e8c_low_va_cnxt.yaml):**
  - tau=0.07, lambda_align=0.3, lambda_rad=0.1, lambda_reg=0.05, lambda_va=0.01
  - w_g=1.0, C_clip=10.0, rho=0.1
  - alpha_tau=0.5, tau_target=0.0
  - visual_backbone=convnext_pico.d1_in1k
- **Data:** SciLibModal v2, same splits as sweep13
- **Seed:** 42

### Sweep (10% data)
- dataset_fraction=0.1, epochs=5, batch_size=64
- Naming: `bl1_gradnorm_s42_<ts>`, `bl2_pcgrad_s42_<ts>`, `bl3_uncertainty_s42_<ts>`
- Checkpoints: DISABLED (speed)

### Full Run (100% data)
- dataset_fraction=1.0, epochs=10, batch_size=64
- Naming: `full_bl1_gradnorm_s42_<ts>`, `full_bl2_pcgrad_s42_<ts>`, `full_bl3_uncertainty_s42_<ts>`
- Checkpoints: ENABLED (keep_best=3)

## What Each Baseline Replaces

In E8c, the T-S fuzzy controller updates λ ∈ R^11:
```
[τ, λ_align, λ_rad, λ_reg, λ_va, w_en, w_ru, w_lean, w_latex, w_img, w_g]
```

All three baselines replace ONLY the modality weight adaptation mechanism (w_en..w_img):

| | τ | λ_align | λ_rad | λ_reg | λ_va | w_m (5) | w_g |
|---|---|---|---|---|---|---|---|
| E8c (our) | adaptive (controller) | adaptive | adaptive | adaptive | adaptive | adaptive | adaptive |
| BL1 GradNorm | static | static | static | static | static | **GradNorm** | static |
| BL2 PCGrad | static | static | static | static | static | **static** (gradient surgery) | static |
| BL3 UW | static | static | static | static | static | **learned σ_m** | static |

## BL1: GradNorm — Adaptation Details

**"Tasks" = 5 per-modality composite losses:**
```
L_m = w_m × (personal_align_m + personal_contrast_m + λ_reg × personal_reg_m)
```

**Algorithm per step:**
1. Compute total loss L = w_g × L_global + Σ w_m × L_m + λ_va × L_va
2. For each modality m: compute G_m = ||∇_W_shared (w_m × L_m)||_2
3. Compute G_bar = mean(G_m)
4. Compute r_m = (L_m(t)/L_m(0)) / mean(L_j(t)/L_j(0))
5. L_grad = Σ_m |G_m - G_bar × r_m^α|
6. Update w_m via gradient of L_grad (target is stop-gradient)
7. Renormalize: w_m ← w_m × 5 / Σ w_j

**W_shared:** All model parameters (no single shared layer in Family A architecture).
**Hyperparameters:** α=1.5, lr_w=0.025 (paper defaults).

**Implementation:** Modify train.py to add GradNorm logic after composite loss forward.
Log: `controller/lambda/w_en..w_img` (same TB keys as E8c for comparison).

## BL2: PCGrad — Adaptation Details

**"Tasks" = 5 per-modality losses (same as BL1).**

**Algorithm per step:**
1. Forward pass → compute 5 per-modality losses + global loss
2. For each L_m: backward separately → store gradient g_m
3. For each g_m, project against all other g_j where cos(g_m, g_j) < 0
4. Aggregate: g_total = Σ g_m_projected + grad(L_global + L_va)
5. Optimizer step with g_total

**Overhead:** 5 separate backward passes (vs 1 in standard training). ~3-5× slower.
**No additional hyperparameters.**

**Implementation:** Custom PCGrad optimizer wrapper or modify backward logic in train.py.
Weights w_m are STATIC (from config, same as E4_composite_static).
Log: `pcgrad/conflict_count`, `pcgrad/mean_cosine` for diagnostics.

## BL3: Uncertainty Weighting — Adaptation Details

**5 learnable parameters:** s_m = log(σ²_m), init=0, added to optimizer.

**Modified loss:**
```
L_total = w_g × L_global + Σ_m [exp(-s_m) × L_m + s_m] + λ_va × L_va
```

**Effective weight:** w_m_eff = exp(-s_m). High uncertainty → low weight.
**Regularizer:** s_m prevents σ → ∞ (degenerate zero-weight solution).

**No additional hyperparameters.** s_m trained with model lr via AdamW.

**Implementation:** Add nn.Parameter for 5 log-variances. Modify composite loss forward.
Log: `controller/lambda/w_en..w_img` as exp(-s_m) values, `uncertainty/s_en..s_img` raw values.

## Metrics (same as all other experiments)

- **Primary:** mean_crossmodal_R@1 (cm@1)
- **Per-modality:** {mod}_to_centroid_R@1, {mod}_to_{mod}_R@1
- **Geometry:** D_intra, D_inter, collapse_score, modality_balance
- **Loss components:** loss/total, loss/* (all components)

## Comparison Plan

Compare sweep results of BL1/BL2/BL3 against:
- **E8c_low_va_cnxt** (best centroid, fuzzy controller) — primary comparison
- **E4_composite_static_cnxt** (static weights, no controller) — ablation baseline
- **E1_pairwise_cnxt** (best overall, no modality balancing) — upper bound

Key questions:
1. Does any baseline match E8c cm@1?
2. Does GradNorm/UW produce similar weight trajectories to E8c controller?
3. Does PCGrad gradient surgery help more than weight adaptation?
4. Cost-benefit: simpler method (UW) vs complex method (E8c) — worth the complexity?

## File Changes

New files:
- `code/baselines/gradnorm.py` — GradNorm implementation
- `code/baselines/pcgrad.py` — PCGrad implementation
- `code/baselines/uncertainty_weighting.py` — Uncertainty Weighting implementation
- `code/baselines/__init__.py`
- `configs/bl1_gradnorm.yaml`
- `configs/bl2_pcgrad.yaml`
- `configs/bl3_uncertainty.yaml`
- `run_sweep_baselines.sh`

Modified files:
- `code/train.py` — add baseline integration (new experiment names in build_loss_fn, baseline logic in training loop)

NOT modified:
- Existing loss functions (losses/*.py)
- Existing configs (e1-e10, h51-h55)
- Existing runs/ and checkpoints/ data
- Model architecture (models/*.py)

## Safety

- All existing TensorBoard logs in runs/ MUST be preserved
- If a baseline run crashes, delete ONLY its log directory (bl*_*)
- Never delete e1-e10, h51-h55, full_* directories
- Checkpoints from previous experiments untouched

## Rollback

```bash
# Remove baseline code and configs only
rm -rf code/baselines/
rm configs/bl{1,2,3}_*.yaml
rm run_sweep_baselines.sh
# Revert train.py changes
git checkout -- code/train.py
```
