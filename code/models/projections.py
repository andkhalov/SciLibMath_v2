"""Projection heads: R^{d'} → R^d with L2 normalization.
Ref: MATH.md M.1.1 — projection g_m maps backbone output to shared embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """2-layer MLP projection: d_in → hidden → d_out, then L2-normalize.

    Ref: MATH.md M.1.1. After projection, all embeddings lie on S^{d-1}.
    [Assumption A.1]: embeddings are L2-normalized.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_in] backbone features
        Returns:
            z: [B, d_out] L2-normalized embeddings on S^{d-1}
        """
        z = self.net(x)
        z = F.normalize(z, dim=-1)  # L2 normalize → unit hypersphere
        return z
