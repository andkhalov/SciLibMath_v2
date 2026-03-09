"""Takagi-Sugeno Fuzzy Controller for E6.
Ref: MATH.md M.6

u_t = Σ_r h̄_r(s_t) · (A_r · s_t + b_r)
λ_{t+1} = Π_Λ(λ_t + u_t)
"""

import torch

from .membership import (
    loss_variable, trend_variable, variance_variable, collapse_variable,
    FuzzyVariable,
)
from .rules import build_rule_matrices, project_to_bounds, U_DIM


class TSFuzzyController:
    """Takagi-Sugeno Fuzzy Controller (E6).
    Ref: MATH.md M.6

    Takes s_t ∈ R^18, outputs u_t ∈ R^11 (correction to λ_t).
    7 rules R0-R6 with Gaussian antecedent MFs.
    """

    def __init__(self, alpha: float = 0.01, device: torch.device = None):
        self.device = device or torch.device("cpu")
        self.rules = build_rule_matrices(alpha)

        # Move to device
        self.rules = [(A.to(self.device), b.to(self.device)) for A, b in self.rules]

        # Linguistic variables for antecedent evaluation
        self.lv_loss = loss_variable("L")
        self.lv_trend = trend_variable("EMA")
        self.lv_var = variance_variable("Var")
        self.lv_collapse = collapse_variable("collapse")

    def _evaluate_antecedents(self, s_t: torch.Tensor) -> torch.Tensor:
        """Compute firing strength h_r(s_t) for each rule.
        Ref: MATH.md M.6.3

        Returns:
            h: [7] normalized firing strengths
        """
        L_t = s_t[0]
        delta_L = s_t[1]
        ema_L = s_t[2]
        var_m = s_t[3]
        collapse = s_t[4]
        grad_conflict = s_t[7]
        L_img = s_t[12]
        ema_img = s_t[17]

        # Fuzzify key variables
        loss_mf = self.lv_loss.fuzzify(L_t)
        trend_mf = self.lv_trend.fuzzify(ema_L)
        var_mf = self.lv_var.fuzzify(var_m)
        collapse_mf = self.lv_collapse.fuzzify(collapse)
        img_loss_mf = self.lv_loss.fuzzify(L_img)
        img_trend_mf = self.lv_trend.fuzzify(ema_img)
        delta_mf = self.lv_trend.fuzzify(delta_L)

        eps = 1e-8

        # Rule firing strengths (product T-norm for AND)
        h = torch.zeros(7, device=s_t.device)

        # R0: Healthy — L is LO AND EMA is DEC
        h[0] = loss_mf["LO"] * trend_mf["DEC"]

        # R1: Imbalance — Var is HI
        h[1] = var_mf["HI"]

        # R2: Collapse — collapse is HI
        h[2] = collapse_mf["HI"]

        # R3: Stagnation — L is HI AND EMA is STB
        h[3] = loss_mf["HI"] * trend_mf["STB"]

        # R4: Overshoot — ΔL is INC (loss increasing)
        h[4] = delta_mf["INC"]

        # R5: Gradient Conflict — grad_conflict is HI (using var proxy)
        conflict_mf = self.lv_var.fuzzify(grad_conflict)
        h[5] = conflict_mf["HI"]

        # R6: Visual Misalignment — L_img is HI AND (EMA_img is STB OR INC)
        h[6] = img_loss_mf["HI"] * torch.max(img_trend_mf["STB"], img_trend_mf["INC"])

        # Normalize (MATH.md M.6.3: h̄_r = h_r / Σ_k h_k)
        h_sum = h.sum() + eps
        h_bar = h / h_sum

        return h_bar

    def compute_correction(self, s_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute control output u_t ∈ R^11.
        Ref: MATH.md M.6.3: u_t = Σ_r h̄_r(s_t) · (A_r · s_t + b_r)

        Args:
            s_t: [18] state vector
        Returns:
            u_t: [11] correction vector
            h_bar: [7] normalized rule activations
        """
        h_bar = self._evaluate_antecedents(s_t)
        u_t = torch.zeros(U_DIM, device=s_t.device)

        for r, (A_r, b_r) in enumerate(self.rules):
            consequent = A_r @ s_t + b_r  # [11]
            u_t = u_t + h_bar[r] * consequent

        return u_t, h_bar

    def step(
        self, s_t: torch.Tensor, lambda_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full controller step: s_t, λ_t → λ_{t+1}.
        Ref: MATH.md M.3.6: λ_{t+1} = Π_Λ(λ_t + u_t)

        Args:
            s_t: [18] state vector
            lambda_t: [11] current hyperparameters
        Returns:
            lambda_new: [11] updated hyperparameters (projected)
            u_t: [11] correction
            h_bar: [7] rule activations
        """
        u_t, h_bar = self.compute_correction(s_t)
        lambda_new = project_to_bounds(lambda_t + u_t)
        return lambda_new, u_t, h_bar
