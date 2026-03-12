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
from .rules import (
    build_rule_matrices, build_nonlinear_rules, project_to_bounds, elastic_step,
    U_DIM, LAMBDA_DEFAULT,
)


class TSFuzzyController:
    """Takagi-Sugeno Fuzzy Controller (E6).
    Ref: MATH.md M.6

    Takes s_t ∈ R^18, outputs u_t ∈ R^11 (correction to λ_t).
    7 rules R0-R6 with Gaussian antecedent MFs.
    """

    def __init__(self, alpha: float = 0.001, device: torch.device = None,
                 warmup_steps: int = 200, norm_momentum: float = 0.99,
                 step_frequency: int = 10, noise_sigma: float = 0.01,
                 noise_anneal: bool = True, elastic_gamma: float = 0.01,
                 total_steps: int = 10000,
                 nonlinear_consequents: bool = False,
                 consequent_hidden: int = 32):
        self.device = device or torch.device("cpu")
        self.alpha = alpha
        self.nonlinear_consequents = nonlinear_consequents

        # Build rules with alpha=1.0 (scaling handled in elastic_step)
        self.rules = build_rule_matrices(alpha=1.0)

        # Move to device
        self.rules = [(A.to(self.device), b.to(self.device)) for A, b in self.rules]

        # E8: Nonlinear MLP consequents (MATH.md M.3.8)
        self.nl_consequents = None
        if nonlinear_consequents:
            self.nl_consequents = build_nonlinear_rules(
                n_rules=len(self.rules), hidden=consequent_hidden,
            ).to(self.device)

        # Linguistic variables for antecedent evaluation
        self.lv_loss = loss_variable("L")
        self.lv_trend = trend_variable("EMA")
        self.lv_var = variance_variable("Var")
        self.lv_collapse = collapse_variable("collapse")

        # Running z-score normalization (Bug 3a fix)
        self.running_mean = torch.zeros(18, device=self.device)
        self.running_var = torch.ones(18, device=self.device)
        self.norm_momentum = norm_momentum
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Stochastic exploration (MATH.md M.6.3b)
        self.step_frequency = step_frequency
        self.noise_sigma = noise_sigma
        self.noise_anneal = noise_anneal
        self.elastic_gamma = elastic_gamma
        self.total_steps = total_steps
        self._last_lambda = None  # cached output for non-update steps

    def _normalize_state(self, s_t: torch.Tensor) -> torch.Tensor:
        """Z-score normalize s_t using running statistics.
        During warmup returns raw values (MFs won't be meaningful anyway).
        After warmup, maps s_t to ~N(0,1) per dimension so MFs calibrated
        for z-scored space activate properly.
        """
        self.step_count += 1
        beta = self.norm_momentum
        self.running_mean = beta * self.running_mean + (1 - beta) * s_t.detach()
        self.running_var = beta * self.running_var + (1 - beta) * (s_t.detach() - self.running_mean) ** 2

        if self.step_count < self.warmup_steps:
            return s_t  # raw during warmup

        std = (self.running_var + 1e-8).sqrt()
        return (s_t - self.running_mean) / std

    def _evaluate_antecedents(self, s_t: torch.Tensor) -> torch.Tensor:
        """Compute firing strength h_r(s_t) for each rule.
        Ref: MATH.md M.6.3

        Returns:
            h: [7] normalized firing strengths
        """
        # Normalize to z-score space for calibrated MFs
        s_t = self._normalize_state(s_t)

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
            if self.nl_consequents is not None:
                consequent = self.nl_consequents[r](s_t)  # [11] MLP (MATH.md M.3.8)
            else:
                consequent = A_r @ s_t + b_r  # [11] linear
            u_t = u_t + h_bar[r] * consequent

        return u_t, h_bar

    def step(
        self, s_t: torch.Tensor, lambda_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full controller step: s_t, λ_t → λ_{t+1} (MATH.md M.6.3b).

        Applies controller every step_frequency steps. Between updates,
        returns cached λ. Adds stochastic noise with optional annealing
        and elastic mean-reversion.

        Args:
            s_t: [18] state vector
            lambda_t: [11] current hyperparameters
        Returns:
            lambda_new: [11] updated hyperparameters (projected)
            u_t: [11] correction (zero if skipped)
            h_bar: [7] rule activations (zero if skipped)
        """
        # Skip update if not on step_frequency boundary (after warmup)
        if self.step_count > self.warmup_steps and self.step_count % self.step_frequency != 0:
            u_t = torch.zeros(U_DIM, device=s_t.device)
            h_bar = torch.zeros(7, device=s_t.device)
            return lambda_t, u_t, h_bar

        u_t, h_bar = self.compute_correction(s_t)

        # Add stochastic exploration noise (MATH.md M.6.3b)
        if self.noise_sigma > 0:
            if self.noise_anneal:
                sigma_t = self.noise_sigma * max(0, 1.0 - self.step_count / self.total_steps)
            else:
                sigma_t = self.noise_sigma
            noise = torch.randn_like(u_t) * sigma_t
            u_t = u_t + noise

        # Elastic step with mean-reversion (MATH.md M.6.3b)
        lambda_new = elastic_step(
            lambda_t, u_t,
            alpha=self.alpha,
            gamma=self.elastic_gamma,
        )

        return lambda_new, u_t, h_bar
