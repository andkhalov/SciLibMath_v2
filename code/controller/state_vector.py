"""State vector s_t ∈ R^18 computation.
Ref: MATH.md M.5

s_t components (3 groups):
  Group 1 — Aggregate (8): L_t, ΔL_t, EMA_L, Var_m(L_m), collapse_t,
                            ΔL_align, ΔL_reg, gradient_conflict
  Group 2 — Per-modality loss (5): L_{en,t}, L_{ru,t}, L_{lean,t}, L_{latex,t}, L_{img,t}
  Group 3 — Per-modality EMA (5): EMA_{en,t}, ..., EMA_{img,t}
"""

import torch
from models.constants import MODALITIES


class StateTracker:
    """Tracks training state and computes s_t ∈ R^18.
    Ref: MATH.md M.5
    """

    def __init__(self, beta: float = 0.99, device: torch.device = None):
        self.beta = beta
        self.device = device or torch.device("cpu")

        # Buffers
        self.prev_loss = None
        self.ema_delta_loss = torch.tensor(0.0, device=self.device)
        self.per_modality_ema = {m: torch.tensor(0.0, device=self.device) for m in MODALITIES}
        self.step = 0

    def update(self, loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Update state from current loss dict, return s_t ∈ R^18.

        Args:
            loss_dict: output from CompositeLoss.forward()
        Returns:
            s_t: [18] state vector
        """
        L_t = loss_dict["loss"].detach()

        # ΔL_t
        if self.prev_loss is not None:
            delta_L = L_t - self.prev_loss
        else:
            delta_L = torch.tensor(0.0, device=self.device)

        # EMA of ΔL
        self.ema_delta_loss = self.beta * self.ema_delta_loss + (1 - self.beta) * delta_L

        # Per-modality losses
        mod_losses = {}
        for mod in MODALITIES:
            key = f"loss_{mod}_contrast"
            if key in loss_dict:
                mod_losses[mod] = loss_dict[key].detach()
            else:
                mod_losses[mod] = torch.tensor(0.0, device=self.device)

        # Variance of modality losses
        mod_loss_vec = torch.stack([mod_losses[m] for m in MODALITIES])
        var_mod = mod_loss_vec.var()

        # Collapse indicator: use collapse_score from geometry metrics if available,
        # otherwise fall back to loss_reg_global as proxy
        collapse_t = loss_dict.get("collapse_score",
                     loss_dict.get("loss_reg_global", torch.tensor(0.0, device=self.device))).detach()

        # Delta align / delta reg
        delta_align = loss_dict.get("loss_align_global", torch.tensor(0.0, device=self.device)).detach()
        delta_reg = loss_dict.get("loss_reg_global", torch.tensor(0.0, device=self.device)).detach()

        # Loss variance proxy for gradient conflict (MATH.md M.5 Implementation Note:
        # direct gradient cosine requires O(M²) backward passes — impractical)
        loss_variance = var_mod  # Proxy: high Var_m(L_m) correlates with gradient conflict

        # Per-modality EMA (smoothed loss value, not delta)
        for mod in MODALITIES:
            self.per_modality_ema[mod] = self.beta * self.per_modality_ema[mod] + (1 - self.beta) * mod_losses[mod]

        # Assemble s_t ∈ R^18
        # Group 1: aggregate (8)
        group1 = torch.stack([
            L_t, delta_L, self.ema_delta_loss, var_mod,
            collapse_t, delta_align, delta_reg, loss_variance,
        ])

        # Group 2: per-modality loss (5)
        group2 = torch.stack([mod_losses[m] for m in MODALITIES])

        # Group 3: per-modality EMA (5)
        group3 = torch.stack([self.per_modality_ema[m] for m in MODALITIES])

        s_t = torch.cat([group1, group2, group3])  # [18]

        self.prev_loss = L_t
        self.step += 1

        return s_t
