"""Alignment and radial regularization losses.
Ref: MATH.md M.3.3 (E3)

L_align: pull modality embeddings toward their centroid (adaptive weights + clipping).
L_rad: per-modality distance to centroid targets radius ρ.
L_anti_collapse: prevent representation collapse.

[Key design per MATH.md M.3.3]: L_align and L_rad operate in UNNORMALIZED space
(e_i^m, c_i). Normalization is applied only for cosine similarity in L_contrast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """Alignment loss with adaptive weights and clipping.
    Ref: MATH.md M.3.3, L_align

    L_align = (1/NM) Σ_i Σ_m γ_m · w_i^m · min(||e_i^m - c_i||², C_clip)

    where w_i^m = ||e_i^m - c_i|| / (Σ_{m'} ||e_i^{m'} - c_i|| + ε)
    focuses learning on modalities that are furthest from centroid.
    """

    def __init__(self, C_clip: float = 10.0):
        super().__init__()
        self.C_clip = C_clip

    def forward(
        self, embeddings: dict[str, torch.Tensor], centroid: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: {modality: [B, d]} UNNORMALIZED
            centroid: [B, d] UNNORMALIZED
        Returns:
            loss: scalar
        """
        mods = list(embeddings.keys())
        M = len(mods)

        # Compute per-modality distances for adaptive weights
        dists = {}  # mod -> [B]
        for mod in mods:
            dists[mod] = (embeddings[mod] - centroid).norm(dim=-1)  # [B]

        # Sum of distances for normalization: [B]
        dist_sum = sum(dists.values()) + 1e-9

        total = 0.0
        for mod in mods:
            w_i_m = dists[mod] / dist_sum  # [B] adaptive weight (MATH.md M.3.3)
            sq_dist = ((embeddings[mod] - centroid) ** 2).sum(dim=-1)  # [B]
            clipped = torch.clamp(sq_dist, max=self.C_clip)  # [B]
            total = total + (w_i_m * clipped).mean()

        return total / M


class RadialLoss(nn.Module):
    """Radial regularization: per-modality distance to centroid targets radius ρ.
    Ref: MATH.md M.3.3, L_rad

    L_rad = (1/NM) Σ_i Σ_m (||e_i^m - c_i|| - ρ)²

    ρ > 0 prevents modality collapse (all e_i^m → c_i) while keeping
    modalities near the hypersphere surface around centroid.
    """

    def __init__(self, rho: float = 0.1):
        super().__init__()
        self.rho = rho

    def forward(
        self, embeddings: dict[str, torch.Tensor], centroid: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: {modality: [B, d]} UNNORMALIZED
            centroid: [B, d] UNNORMALIZED
        Returns:
            loss: scalar
        """
        total = 0.0
        M = 0
        for emb in embeddings.values():
            dist = (emb - centroid).norm(dim=-1)  # [B] ||e_m^i - c_i||
            total = total + ((dist - self.rho) ** 2).mean()
            M += 1
        return total / max(M, 1)


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
