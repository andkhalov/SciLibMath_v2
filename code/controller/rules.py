"""Fuzzy rules R0-R6 and consequent matrices A_r.
Ref: MATH.md M.6.3, M.6.3*

7 rules covering 6 training regimes:
  R0: Healthy (no action needed)
  R1: Modality Imbalance (rebalance w_m)
  R2: Collapse Risk (increase τ, λ_reg)
  R3: Stagnation (increase λ_align, decrease τ)
  R4: Overshoot (decrease all λ)
  R5: Gradient Conflict (decrease λ_align, increase λ_reg)
  R6: Visual Misalignment (increase λ_va)

Each rule r: u_t^r = A_r · s_t + b_r
  A_r ∈ R^{11×18}, b_r ∈ R^{11}
"""

import torch

# State vector indices (MATH.md M.6.3*)
# Group 1 (aggregate): L_t(0), ΔL(1), EMA_L(2), Var_m(3), collapse(4), ΔL_align(5), ΔL_reg(6), grad_conflict(7)
# Group 2 (per-mod loss): L_en(8), L_ru(9), L_lean(10), L_latex(11), L_img(12)
# Group 3 (per-mod EMA): EMA_en(13), EMA_ru(14), EMA_lean(15), EMA_latex(16), EMA_img(17)

# Output vector indices (λ_t ∈ R^11, MATH.md M.3.6):
# τ(0), λ_align(1), λ_rad(2), λ_reg(3), λ_va(4), w_en(5), w_ru(6), w_lean(7), w_latex(8), w_img(9), w_g(10)

S_DIM = 18
U_DIM = 11


def build_rule_matrices(alpha: float = 0.01) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build (A_r, b_r) for each of the 7 rules.
    Ref: MATH.md M.6.3*

    Returns:
        list of (A_r, b_r) tuples, A_r ∈ R^{11×18}, b_r ∈ R^{11}
    """
    rules = []

    # --- R0: Healthy (no correction needed) ---
    A0 = torch.zeros(U_DIM, S_DIM)
    b0 = torch.zeros(U_DIM)
    rules.append((A0, b0))

    # --- R1: Modality Imbalance ---
    # When Var_m is HI: increase w_m for high-loss modalities, decrease for low
    # A_{R1}: Var(3) → +λ_align(1); L_m(8-12) → -w_m(5-9); EMA_m(13-17) → -w_m(5-9); Var(3) → +w_g(10)
    A1 = torch.zeros(U_DIM, S_DIM)
    A1[1, 3] = alpha * 2       # Var → +λ_align
    for i, s_idx in enumerate([8, 9, 10, 11, 12]):  # L_m → -w_m
        A1[5 + i, s_idx] = -alpha
    for i, s_idx in enumerate([13, 14, 15, 16, 17]):  # EMA_m → -w_m
        A1[5 + i, s_idx] = -alpha * 0.5
    A1[10, 3] = alpha          # Var → +w_g
    b1 = torch.zeros(U_DIM)
    rules.append((A1, b1))

    # --- R2: Collapse Risk ---
    # When collapse is HI: increase τ (soften), increase λ_reg, decrease λ_align
    A2 = torch.zeros(U_DIM, S_DIM)
    A2[0, 4] = alpha * 3      # collapse → +τ
    A2[3, 4] = alpha * 2      # collapse → +λ_reg
    A2[1, 4] = -alpha          # collapse → -λ_align
    b2 = torch.zeros(U_DIM)
    rules.append((A2, b2))

    # --- R3: Stagnation ---
    # When EMA_L is STB and L is HI: increase λ_align, decrease τ
    A3 = torch.zeros(U_DIM, S_DIM)
    A3[1, 0] = alpha * 2      # L_t → +λ_align (when stagnating at high loss)
    A3[0, 2] = -alpha          # EMA_L → -τ (sharpen contrast)
    b3 = torch.zeros(U_DIM)
    rules.append((A3, b3))

    # --- R4: Overshoot ---
    # When ΔL is INC (loss increasing): decrease all λ, increase τ
    A4 = torch.zeros(U_DIM, S_DIM)
    A4[0, 1] = alpha * 2      # ΔL → +τ (soften)
    A4[1, 1] = -alpha          # ΔL → -λ_align
    A4[2, 1] = -alpha * 0.5   # ΔL → -λ_rad
    A4[3, 1] = -alpha * 0.5   # ΔL → -λ_reg
    b4 = torch.zeros(U_DIM)
    rules.append((A4, b4))

    # --- R5: Gradient Conflict ---
    # When grad_conflict is HI: decrease λ_align, increase λ_reg
    A5 = torch.zeros(U_DIM, S_DIM)
    A5[1, 7] = -alpha          # grad_conflict → -λ_align
    A5[3, 7] = alpha * 2      # grad_conflict → +λ_reg
    b5 = torch.zeros(U_DIM)
    rules.append((A5, b5))

    # --- R6: Visual Misalignment ---
    # When L_img is HI and EMA_img is STB/INC: increase λ_va
    A6 = torch.zeros(U_DIM, S_DIM)
    A6[4, 12] = alpha * 2     # L_img → +λ_va
    A6[4, 17] = alpha          # EMA_img → +λ_va
    b6 = torch.zeros(U_DIM)
    rules.append((A6, b6))

    return rules


# Box constraints for λ_t (MATH.md M.3.6, M.6.3b — narrowed after EXP-001 corner-crashing)
LAMBDA_BOUNDS = torch.tensor([
    [0.01, 0.2],   # τ
    [0.01, 1.0],   # λ_align
    [0.001, 0.5],  # λ_rad
    [0.001, 0.5],  # λ_reg
    [0.01, 0.5],   # λ_va
    [0.3, 3.0],    # w_en   (narrowed from [0.1, 5.0])
    [0.3, 3.0],    # w_ru   (narrowed from [0.1, 5.0])
    [0.3, 3.0],    # w_lean (narrowed from [0.1, 5.0])
    [0.3, 3.0],    # w_latex(narrowed from [0.1, 5.0])
    [0.3, 3.0],    # w_img  (narrowed from [0.1, 5.0])
    [0.3, 3.0],    # w_g    (narrowed from [0.1, 5.0])
])

# Default λ_0 for elastic mean-reversion (MATH.md M.6.3b)
LAMBDA_DEFAULT = torch.tensor([
    0.07,   # τ
    0.3,    # λ_align
    0.1,    # λ_rad
    0.05,   # λ_reg
    0.1,    # λ_va
    1.0,    # w_en
    1.0,    # w_ru
    1.5,    # w_lean
    1.5,    # w_latex
    1.0,    # w_img
    1.0,    # w_g
])


def project_to_bounds(lam: torch.Tensor, bounds: torch.Tensor = None) -> torch.Tensor:
    """Project λ_t onto feasible box Λ (MATH.md M.3.6).
    Π_Λ(λ) = clamp(λ, lower, upper)
    """
    if bounds is None:
        bounds = LAMBDA_BOUNDS.to(lam.device)
    return torch.clamp(lam, min=bounds[:, 0], max=bounds[:, 1])


def elastic_step(
    lambda_t: torch.Tensor,
    u_t: torch.Tensor,
    alpha: float = 0.001,
    gamma: float = 0.01,
    lambda_default: torch.Tensor = None,
    bounds: torch.Tensor = None,
) -> torch.Tensor:
    """Stochastic T-S update with elastic mean-reversion (MATH.md M.6.3b).

    λ_{t+1} = Π_Λ(λ_t + α · u_t + γ · (λ_0 - λ_t))

    Args:
        lambda_t: [11] current hyperparameters
        u_t: [11] controller correction (raw, before alpha scaling)
        alpha: controller step size
        gamma: elastic reversion coefficient
        lambda_default: [11] default values for reversion
        bounds: [11, 2] box constraints
    """
    if lambda_default is None:
        lambda_default = LAMBDA_DEFAULT.to(lambda_t.device)
    else:
        lambda_default = lambda_default.to(lambda_t.device)

    reversion = gamma * (lambda_default - lambda_t)
    lambda_new = lambda_t + alpha * u_t + reversion
    return project_to_bounds(lambda_new, bounds)
