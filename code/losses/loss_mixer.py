"""LossMixer for E5: learnable loss weighting via MLP.
Ref: MATH.md M.3.5

LossMixer takes current loss component values as input and outputs
adaptive weights. This is the data-driven alternative to static weights (E4)
and fuzzy controller (E6).

W = MLP(L_components) ∈ R^{M×K} — flat parameterization without λ/w hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.constants import MODALITIES


class LossMixer(nn.Module):
    """MLP-based adaptive loss weighting (E5).
    Ref: MATH.md M.3.5

    Input: vector of current loss component values.
    Output: weight vector for each component.
    """

    def __init__(
        self,
        n_components: int = 20,  # MATH.md M.3.4*: 20 atomic loss components
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_components = n_components

        self.mlp = nn.Sequential(
            nn.Linear(n_components, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components),
            nn.Softplus(),  # Ensure positive weights
        )

        # Initialize close to uniform
        with torch.no_grad():
            self.mlp[-2].bias.fill_(0.0)  # Softplus(0) = ln(2) ≈ 0.7

    def forward(self, loss_components: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss_components: [n_components] detached loss values
        Returns:
            weights: [n_components] adaptive weights (positive)
        """
        # Detach inputs — MLP learns from loss values, not through them
        x = loss_components.detach()
        weights = self.mlp(x)
        return weights


class LossMixerComposite(nn.Module):
    """E5: Composite loss with LossMixer-learned weights.
    Ref: MATH.md M.3.5

    Replaces static λ/w hierarchy with learned MLP weights.
    """

    def __init__(self, tau: float = 0.07, lambda_va: float = 0.1, hidden_dim: int = 64):
        super().__init__()
        self.tau = tau
        self.lambda_va = lambda_va
        # 20 atomic components per MATH.md M.3.4*
        self.mixer = LossMixer(n_components=20, hidden_dim=hidden_dim)

    def forward(
        self,
        loss_dict: dict[str, torch.Tensor],
        visual_align_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            loss_dict: dict of pre-computed loss components (from CompositeLoss internals)
            visual_align_loss: scalar
        Returns:
            dict with 'loss' (weighted total) and component details
        """
        # Collect all atomic components into a vector
        component_names = [
            "loss_contrast_global", "loss_align_global", "loss_rad", "loss_reg_global",
        ]
        for mod in MODALITIES:
            component_names.extend([
                f"loss_{mod}_align", f"loss_{mod}_contrast", f"loss_{mod}_reg"
            ])
        component_names.append("loss_va")

        # Build component vector
        components = []
        for name in component_names:
            if name == "loss_va":
                components.append(visual_align_loss)
            elif name in loss_dict:
                components.append(loss_dict[name])
            else:
                components.append(torch.tensor(0.0, device=visual_align_loss.device))

        component_vec = torch.stack(components)  # [20]
        weights = self.mixer(component_vec)  # [20]

        # Weighted sum
        L_total = (weights * component_vec).sum()

        return {
            "loss": L_total,
            "mixer_weights": weights.detach(),
            "components": component_vec.detach(),
            **loss_dict,
        }
