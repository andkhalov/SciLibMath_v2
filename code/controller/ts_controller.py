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
                 consequent_hidden: int = 32,
                 init_scale: float = 0.01,
                 w_min: float = None):
        self.device = device or torch.device("cpu")
        self.alpha = alpha
        self.nonlinear_consequents = nonlinear_consequents
        self.w_min = w_min  # H51: minimum modality weight floor
        self.skip_bounds = False  # v7d/v8d: disable project_to_bounds

        # Build rules with alpha=1.0 (scaling handled in elastic_step)
        self.rules = build_rule_matrices(alpha=1.0)

        # Move to device
        self.rules = [(A.to(self.device), b.to(self.device)) for A, b in self.rules]

        # E8: Nonlinear MLP consequents (MATH.md M.3.8)
        self.nl_consequents = None
        if nonlinear_consequents:
            self.nl_consequents = build_nonlinear_rules(
                n_rules=len(self.rules), hidden=consequent_hidden,
                init_scale=init_scale,
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
        self.step_count = 0          # counts normalization calls (legacy)
        self._training_step = 0      # counts ALL step() calls (for step_frequency)
        self._cached_h_bar = torch.zeros(7, device=self.device)  # last firing's h_bar (for logging)

        # Stochastic exploration (MATH.md M.6.3b)
        self.step_frequency = step_frequency
        self.noise_sigma = noise_sigma
        self.noise_anneal = noise_anneal
        self.elastic_gamma = elastic_gamma
        self.total_steps = total_steps
        self._last_lambda = None  # cached output for non-update steps

    def _update_running_stats(self, s_t: torch.Tensor):
        """Update running mean/var from every training step (even skipped ones).
        Called from step() unconditionally so stats stay fresh.
        """
        beta = self.norm_momentum
        s = s_t.detach()
        self.running_mean = beta * self.running_mean + (1 - beta) * s
        self.running_var = beta * self.running_var + (1 - beta) * (s - self.running_mean) ** 2

    def _normalize_state(self, s_t: torch.Tensor) -> torch.Tensor:
        """Z-score normalize s_t using running statistics.
        During warmup returns raw values (MFs won't be meaningful anyway).
        After warmup, maps s_t to ~N(0,1) per dimension so MFs calibrated
        for z-scored space activate properly.

        Note: running stats are updated in _update_running_stats() called from step().
        """
        self.step_count += 1  # legacy counter (tracks normalization calls)

        if self._training_step < self.warmup_steps:
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
            linear = A_r @ s_t + b_r  # [11] hardcoded base strategy
            if self.nl_consequents is not None:
                # Residual: MLP learns correction to linear rule (v5/e8cf_real)
                nonlinear = self.nl_consequents[r](s_t)  # [11] learnable correction
                consequent = linear + nonlinear
            else:
                consequent = linear
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
        # Always increment training step and update running stats
        # (FIX: step_count was only incremented inside _normalize_state,
        #  which was skipped on non-update steps → step_count stuck forever)
        self._training_step += 1
        self._update_running_stats(s_t)

        # Skip update if not on step_frequency boundary (after warmup)
        if self._training_step > self.warmup_steps and self._training_step % self.step_frequency != 0:
            u_t = torch.zeros(U_DIM, device=s_t.device)
            return lambda_t, u_t, self._cached_h_bar  # return last firing's h_bar for logging

        u_t, h_bar = self.compute_correction(s_t)
        self._cached_h_bar = h_bar.detach().clone()  # cache for skipped steps

        # Add stochastic exploration noise (MATH.md M.6.3b)
        if self.noise_sigma > 0:
            if self.noise_anneal:
                sigma_t = self.noise_sigma * max(0, 1.0 - self._training_step / self.total_steps)
            else:
                sigma_t = self.noise_sigma
            noise = torch.randn_like(u_t) * sigma_t
            u_t = u_t + noise

        # Elastic step with mean-reversion (MATH.md M.6.3b)
        lambda_new = elastic_step(
            lambda_t, u_t,
            alpha=self.alpha,
            gamma=self.elastic_gamma,
            w_min=self.w_min,
            skip_bounds=self.skip_bounds,
        )

        return lambda_new, u_t, h_bar
