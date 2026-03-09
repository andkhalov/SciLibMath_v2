"""Alignment and radial regularization losses.
Ref: MATH.md M.3.3 (E3)

L_align: pull modality embeddings toward their centroid.
L_rad: ensure embeddings stay near unit hypersphere (radial consistency).
L_anti_collapse: prevent representation collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """Alignment loss: minimize distance from each embedding to its centroid.
    Ref: MATH.md M.3.3, L_align

    L_align = (1/NM) * sum_i sum_m ||e_m^i - c_i||^2
    """

    def forward(
        self, embeddings: dict[str, torch.Tensor], centroid: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: {modality: [B, d]}
            centroid: [B, d]
        Returns:
            loss: scalar
        """
        total = 0.0
        for emb in embeddings.values():
            total = total + ((emb - centroid) ** 2).sum(dim=-1).mean()
        return total / len(embeddings)


class RadialLoss(nn.Module):
    """Radial regularization: keep centroid norms close to 1.
    Ref: MATH.md M.3.3, L_rad

    L_rad = (1/N) * sum_i (||c_i|| - 1)^2
    """

    def forward(self, centroid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            centroid: [B, d] (may not be exactly unit norm)
        Returns:
            loss: scalar
        """
        norms = centroid.norm(dim=-1)  # [B]
        return ((norms - 1.0) ** 2).mean()


class AntiCollapseLoss(nn.Module):
    """Anti-collapse regularization via centroid similarity monitoring.
    Ref: MATH.md M.3.3

    L_reg = ReLU(s̄_neg) + ReLU(1 - s̄_pos)
    where S = centroid similarity matrix, s̄_neg = mean off-diagonal,
    s̄_pos = mean diagonal.

    [Мотивация]: VICReg-inspired. Prevents all centroids collapsing
    to same point (which would trivially minimize contrastive loss).
    """

    def forward(self, centroid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            centroid: [B, d] normalized centroids
        Returns:
            loss: scalar
        """
        # Similarity matrix S[i,j] = <c_i, c_j>
        centroid_norm = F.normalize(centroid, dim=-1)
        S = torch.mm(centroid_norm, centroid_norm.t())  # [B, B]

        B = S.size(0)
        if B < 2:
            return torch.tensor(0.0, device=centroid.device)

        # Diagonal (self-similarity, should be ~1)
        diag = S.diag()  # [B]
        s_pos = diag.mean()

        # Off-diagonal (should be low / negative)
        mask = ~torch.eye(B, dtype=torch.bool, device=S.device)
        s_neg = S[mask].mean()

        # MATH.md M.3.3: ReLU(s̄_neg) + ReLU(1 - s̄_pos)
        loss = F.relu(s_neg) + F.relu(1.0 - s_pos)
        return loss
