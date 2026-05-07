# EXP-013 E8cf: Differentiable T-S Controller — Lab Journal

> Date: 2026-04-27
> GPU: RTX 3090 24GB
> Data: SciLibModal v2, 10% (sweep), seed=42

---

## Timeline

### v1 — Supervisory MSE (failed)
- MLP target: u_target from heuristic loss imbalance signals
- L_mlp = w · MSE(u_t, u_target)
- **Result:** MLP collapsed to zero output. Target too far from MLP range.
- **Discarded.** Run deleted.

### v2 — Reaggregate + softmax + L2 weight decay (MLP collapsed)
- Reaggregate: detached loss components × differentiable λ
- Softmax on modality weights, quadratic regularizer μ=1.0
- L2 weight decay γ=0.001 on MLP weights
- **Result:** L2 killed MLP (norm 30→0.18). cm@1=0.3224.
- **Discarded.** Run deleted.

### v2b — Reaggregate + softmax(clamped λ) + anti-collapse
- Anti-collapse reg (min_norm=5.0) instead of L2 decay, μ=10.0
- **Result:** cm@1=0.5001. MLP norm 30→44.
- **Issue:** modality weights saturate at bounds, clamp kills gradient.
- **Discarded.** Run deleted.

### v3 (e8cf_unc) — Best metric, MLP decorative ← KEPT for reference
- Softmax from unclamped u_t[5:10], elastic_gamma=0.002
- **Result: cm@1 = 0.5084** (best sweep result across all methods)
- **Run preserved:** `runs/e8cf_unc_s42_1777245613`
- **Config:** elastic_gamma=0.002, init_scale=0.1, mlp_lr_multiplier=10.0,
  reagg_mu=10.0, mlp_min_norm=5.0, mlp_norm_gamma=1.0,
  softmax from raw u_t (unclamped)
- **Post-mortem analysis:** MLP consequents get near-zero gradient
  (exp_avg_norm ≈ 0.0003). Softmax weights = uniform (1.0) throughout.
  MLP is a passenger — cm@1 comes from random drift to bounds, not learning.
  Lambda trajectory: all w_m → bounds (0.3 or 3.0) by step 2000 via
  noise+micro-correction drift with absorbing boundaries (clamp).

### v4 — No elastic (elastic_gamma=0)
- **Result:** cm@1=0.4728 — worse. λ saturates to bounds even faster.
- **Discarded.** Run deleted.

---

## Root Cause Analysis: Why MLP Consequents Don't Learn (v1–v4)

### Problem 1: MLP replaces linear, doesn't augment
```python
# ts_controller.py:178-183
if self.nl_consequents is not None:
    consequent = self.nl_consequents[r](s_t)  # MLP REPLACES linear
else:
    consequent = A_r @ s_t + b_r              # linear rules
```
MLP (init_scale=0.1) outputs ~0.01 vs linear ~1-10. MLP is 100-1000× weaker.

### Problem 2: Gradient path too indirect
L_reagg → softmax(u_t[5:10]) → u_t → h_bar[r] × MLP_r(s_t) → W_MLP.
With uniform softmax (because MLP ≈ 0), gradient ∂L/∂z_k ≈ 0.

### Problem 3: Bounds as absorbing barriers
Random drift + noise push λ toward bounds. Clamp holds them there.
Elastic reversion (γ=0.002) too weak to pull back.
Result: λ trajectory = random walk with absorbing boundaries, not control.

---

## v5 (e8cf_real) — Residual MLP Architecture (NEXT)

### Fix: MLP augments linear, doesn't replace
```python
linear = A_r @ s_t + b_r           # hardcoded base strategy
nonlinear = self.nl_consequents[r](s_t)  # learnable correction
consequent = linear + nonlinear     # residual connection
```

### Why this should work:
1. At init (MLP ≈ 0): consequent ≈ linear → behavior = E6 (proven baseline)
2. MLP learns CORRECTION to linear strategy, not full strategy from scratch
3. Gradient through `linear + nonlinear` is same magnitude as through `linear`
   → MLP gets strong gradient signal from the start
4. MLP can learn nonlinear refinements the linear rules can't express

### Expected behavior:
- Early training: e8cf_real ≈ E6 (linear rules dominate)
- Mid training: MLP correction grows, adapts to training dynamics
- Late training: MLP provides situation-specific nonlinear adjustments

### v5 full — Residual MLP, full data (completed)
- Config: `e8cf_real.yaml` (experiment=e8cf_differentiable)
- **Result: cm@1 = 0.6288** — worse than E8c (0.7070) by 7.8 p.p.
- Root cause: `loss + L_reagg` doubles loss magnitude → distorts model gradient
- Text modalities destroyed (en/ru/lean: 0.71 vs 0.95 in E8c)
- Run: `runs/full_e8cf_differentiable_s42_1777525052`

### v6 (e8cf_v6) — Separate backward ← CURRENT (full run)

**Проблема v5:** `loss_dict["loss"] + L_reagg` удваивает magnitude loss → GradScaler,
OneCycleLR, gradient clipping работают на искажённом масштабе → модель деградирует
(en/ru/lean: 0.71 vs 0.95). Reaggregate полезен для MLP, но разрушителен для модели.

**Решение v6:** два раздельных backward pass в одном training step:

```
Step t:
  1. Forward: model(batch) → embeddings → CompositeLoss → loss_dict
  2. Controller: s_t → fuzzy rules → h_bar → consequent = linear + MLP → u_t → elastic → new_λ
  3. MLP backward: L_reagg = reaggregate(loss_dict, new_λ, u_t)
     L_reagg.backward(retain_graph=True)
     → grad flows: L_reagg → softmax(u_t[5:10]) → u_t → MLP_r(s_t) → W_MLP
     → model params get ZERO grad (loss components detached)
  4. Model backward: loss_dict["loss"].backward()
     → grad flows: L → model params (standard composite loss)
     → MLP params get ZERO grad (not in this computation graph)
  5. optimizer.step() → updates both model AND MLP (each from own gradient)
```

**Семантика разделённого градиента:**
- Модель учится минимизировать composite loss при ТЕКУЩИХ λ (как E8c)
- MLP учится предсказывать λ-коррекцию, которая минимизирует ПЕРЕСОБРАННЫЙ loss
  (с softmax-нормализованными весами модальностей)
- Два обучающихся агента, не мешающих друг другу:
  - Модель: «при данных весах, выучи лучшие эмбеддинги»
  - Контроллер: «при данных лоссах, выучи лучшее распределение весов»

**Почему это должно работать:**
- Model loss = точно как E8c → ожидаем cm@1 ≈ 0.707 (E8c level)
- MLP gradient = чистый сигнал от reaggregate → MLP учится без distortion
- Если MLP выучит полезное → cm@1 > E8c (бонус от адаптивной коррекции)
- Если MLP не выучит → cm@1 = E8c (не хуже, т.к. модель не затронута)

**Архитектура контроллера (v6):**
```
s_t ∈ R^18 (detached: L_t, ΔL, EMA, Var_m, collapse, L_mod, EMA_mod)
    ↓
7 fuzzy antecedents (Gaussian MFs, z-normalized)
    ↓ h_bar ∈ R^7
for each rule r:
    consequent_r = A_r @ s_t + b_r + MLP_r(s_t)   ← residual
    u_t += h_bar[r] * consequent_r
    ↓
u_t ∈ R^11
    ↓ elastic_step (α=0.01, γ=0.002)
new_λ ∈ R^11 (clamped to bounds)
    ↓
set_lambda_vector(new_λ.detach())  → scalars for next forward pass
softmax(u_t[5:10]) × M            → differentiable modality weights for L_reagg
```

**MLP consequent:** 18 → 32 → ReLU → 11 (init_scale=0.01, ~0.6k params × 7 rules = ~4.2k total)
**Regularizers:** anti-collapse (min_norm=5.0), quadratic on hyperparams (μ=10.0)
**LR:** MLP gets 10× model lr (mlp_lr_multiplier=10.0)

**Config:** `configs/e8cf_v6.yaml`
**Run:** `runs/full_e8cf_v6_s42_1777665069`
**Status:** TERMINATED at epoch 6/15 (2026-05-02). cm@1=0.5957, en→c=0.68, ru→c=0.69.
Same text modality degradation as v5 — separate backward did NOT fix the root cause.
Root cause: MLP via softmax(u_t) still biases w_img ↑ and w_text ↓ through set_lambda_vector.
Run and checkpoints deleted.

### v7/v8 — Per-modality / per-rule signal (NEXT)

**Diagnosis:** All 7 MLP consequents receive ONE gradient signal (∂L_reagg/∂u_t).
They cannot distinguish how their correction affects each modality separately.
The softmax finds "boost img, suppress text" as global optimum — hurts text modalities.

**v7 idea — per-modality signal:**
Each modality gets its own L_reagg_m. MLP learns separately how its correction
affects en, ru, lean, latex, img. No single softmax across all — prevents
"rob from text to pay img" strategy.

**v8 idea — per-rule specialized signal:**
Each MLP_r gets a surrogate loss matching its rule's semantics.

### v7b/v8b — deficit-based (ReLU(L_m - EMA_m)) — FAILED
- Sign fixed: (-u_t) · deficit → correct direction (increase weight for lagging)
- Problem: deficit → 0 as EMA catches up (within ~100 steps at β=0.99)
- Loss falls monotonically → L_m ≈ EMA_m most of the time → no signal
- v7b sweep: cm@1=0.4565 but u_t → 0 by mid-training

### v7c/v8c — relative training rate (CURRENT, 2026-05-02)
- Fix: replace deficit with r_m = L_m / EMA_m (ratio, never zero while loss > 0)
- r_m > 1: modality worse than its trend → MLP increases its weight
- r_m < 1: modality better than trend → MLP decreases (but signal still exists)
- L_mlp = Σ_m (-u_t[5+m]) · r_m — perpetual gradient, proportional to relative rate
- Expected: MLP outputs never collapse because signal never vanishes

### v7c/v7d/v8d sweep results (2026-05-02)

| Sweep | Signal | Bounds | cm@1 | u_t[5:10] | Notes |
|-------|--------|--------|------|-----------|-------|
| v7c | per-modality relative rate | С bounds | **0.4737** | →0 | MLP norm 29→226, best result |
| v7d | per-modality relative rate | Без bounds | 0.0108 | →0 | λ diverges, model collapse |
| v8d | per-rule relative rate | Без bounds | 0.0044 | →0 | λ diverges, model collapse |

**Conclusion:** Bounds are ESSENTIAL for stability. Without them λ diverges.
v7c best among trainable consequent variants (0.4737 > E8c 0.3525 on sweep).
MLP modality weights [5:10] collapse to zero in all cases — gradient signal
vanishes as L_m/EMA_m → 1 (loss stabilizes). MLP norm grows on OTHER components
(τ, λ_align, etc.) but those are bounded by clamp.

**v7c launched on full data** (2026-05-02) to compare with E8c full (0.7070).
Config: `e8cf_v7.yaml`, run_tag=full.
Key question: does v7c sustain its sweep advantage on full training?

### On bounds optimality

Current bounds (from MATH.md M.3.6):
- τ ∈ [0.01, 0.2], λ_align ∈ [0.01, 1.0], λ_rad ∈ [0.001, 0.5]
- λ_reg ∈ [0.001, 0.5], λ_va ∈ [0.01, 0.5], w_m ∈ [0.3, 3.0]

These were set heuristically. To find optimal bounds:
1. Observe λ trajectories from best runs (E8c full: τ→0.075, λ_va→0.08, w_m≈1±0.01)
2. Bounds mostly not hit in E8c full (except possibly at early transient)
3. For v7c: bounds immediately saturated (w_m→0.3 or 3.0 by step 2000)
4. Optimal bounds = wide enough for MLP to explore, narrow enough to prevent divergence
5. This is a hyperparameter search problem — could use Bayesian optimization on bounds

### Rollback to v3 (best sweep metric):
```bash
# v3 run: runs/e8cf_unc_s42_1777245613
# To restore v3 code: revert compute_correction to MLP-only (no residual)
# Config: elastic_gamma=0.002, all other params same
```
