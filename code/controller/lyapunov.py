"""Lyapunov stability regularization for E7.
Ref: MATH.md M.7

V_t = α · L̃_t + β · ||Δλ_t||² + γ · Var_m(w_{m,t})

E7 adds V_t as soft constraint: penalize if V_{t+1} > V_t.
"""

import torch


class LyapunovRegularizer:
    """Lyapunov function V_t and stability penalty (E7).
    Ref: MATH.md M.7

    V_t measures system "energy". Training should decrease V_t on average.
    If V_t increases, add penalty to loss.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.5,
        penalty_weight: float = 0.1,
        xi: float = 0.01,
        device: torch.device = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.penalty_weight = penalty_weight
        self.xi = xi  # Threshold for allowable V growth (MATH.md M.7.2b)
        self.device = device or torch.device("cpu")
        self.V_prev = None

    def compute_V(
        self, loss_normalized: float, delta_lambda: torch.Tensor, modality_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute V_t.
        Ref: MATH.md M.7: V_t = α·L̃_t + β·||Δλ_t||² + γ·Var_m(w_m)

        Args:
            loss_normalized: L̃_t (normalized aggregate loss)
            delta_lambda: Δλ_t = λ_t - λ_{t-1}
            modality_weights: [5] current w_m values
        Returns:
            V_t: scalar
        """
        L_term = self.alpha * loss_normalized
        lambda_term = self.beta * (delta_lambda ** 2).sum()
        var_term = self.gamma * modality_weights.var()

        V_t = L_term + lambda_term + var_term
        return V_t

    def get_penalty(
        self, loss_normalized: float, delta_lambda: torch.Tensor, modality_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute Lyapunov soft constraint for E7 (MATH.md M.7.2b).

        L_lyapunov = max(0, ΔV_t - ξ) · penalty_weight

        Penalizes only GROWTH of V_t beyond threshold ξ, not absolute value.
        This allows bounded fluctuations while preventing runaway instability.

        Returns:
            penalty: scalar (0 if ΔV ≤ ξ)
            info: dict with V_t, delta_V for logging
        """
        V_t = self.compute_V(loss_normalized, delta_lambda, modality_weights)

        info = {"V_t": V_t.item()}

        if self.V_prev is not None:
            delta_V = V_t - self.V_prev
            info["delta_V"] = delta_V.item()
            # Soft constraint: penalize only growth beyond ξ threshold
            penalty = self.penalty_weight * torch.relu(delta_V - self.xi)
        else:
            penalty = torch.tensor(0.0, device=self.device)
            info["delta_V"] = 0.0

        self.V_prev = V_t.detach()
        return penalty, info
