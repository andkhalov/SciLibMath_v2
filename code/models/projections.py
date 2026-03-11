"""Projection heads: R^{d'} → R^d (WITHOUT L2 normalization).
Ref: MATH.md M.1.1 — projection g_m = Linear(d'_m, d).
MATH.md M.0.4 — normalization applied separately at similarity computation time.

[Key design]: ProjectionHead does NOT normalize. Centroid is computed from
unnormalized embeddings and normalized separately. This matters because
mean(normalize(x)) ≠ normalize(mean(x)).
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection: d_in → d_out. NO L2-normalization.

    Ref: MATH.md M.1.1, M.0.4.
    Normalization happens downstream (in loss functions, metrics, centroid).
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_in] backbone features
        Returns:
            z: [B, d_out] projected embeddings (NOT normalized)
        """
        return self.proj(x)
