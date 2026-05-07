"""BL3: Uncertainty Weighting (Kendall et al. CVPR 2018).

Multi-task loss balancing via learned homoscedastic uncertainty parameters.

For classification-like losses (InfoNCE ≈ cross-entropy):
    L_total = Σ_m [exp(-s_m) * L_m + s_m]

where s_m = log(σ²_m) are learnable parameters.
"""

import torch
import torch.nn as nn
from models.constants import MODALITIES


class UncertaintyWeighting(nn.Module):
    """Learnable per-modality uncertainty parameters for loss balancing.

    Replaces static modality weights w_m with exp(-s_m) where s_m is learned.
    """

    def __init__(self, modalities: list[str] = None, init_val: float = 0.0):
        super().__init__()
        self.modalities = modalities or list(MODALITIES)
        # s_m = log(σ²_m), initialized to 0 → σ²=1 → weight=1
        self.log_vars = nn.ParameterDict({
            mod: nn.Parameter(torch.tensor(init_val))
            for mod in self.modalities
        })

    def get_weights(self) -> dict[str, float]:
        """Return effective weights exp(-s_m) for each modality."""
        return {mod: torch.exp(-self.log_vars[mod]).item() for mod in self.modalities}

    def get_log_vars(self) -> dict[str, float]:
        """Return raw s_m values for logging."""
        return {mod: self.log_vars[mod].item() for mod in self.modalities}

    def reweight_loss(self, modality_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply uncertainty weighting to per-modality losses.

        Args:
            modality_losses: {mod: L_m} per-modality scalar losses

        Returns:
            Σ_m [exp(-s_m) * L_m + s_m]
        """
        total = torch.tensor(0.0, device=next(iter(modality_losses.values())).device)
        for mod in self.modalities:
            if mod in modality_losses:
                s_m = self.log_vars[mod]
                precision = torch.exp(-s_m)
                total = total + precision * modality_losses[mod] + s_m
        return total
