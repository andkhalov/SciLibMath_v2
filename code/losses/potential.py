"""Potential function loss for E9/E10: harmonic attraction + log-barrier repulsion.
Ref: MATH.md M.3.9

Replaces L_align + L_rad with a force-based formulation:
  L = k_a * attract + k_r * repel
  attract = (1/NM) Σ_i Σ_m ||e_m^i - c_i||²
  repel = -(1/C(N,2)) Σ_{i<j} log(||c_i - c_j|| + ε)

Properties:
  - No fixed target radius ρ (avoids over-compression problem)
  - Attraction force ∝ distance (stronger for far modalities)
  - Repulsion force ∝ 1/distance (stronger for close centroids)
"""

import torch
import torch.nn as nn


class PotentialLoss(nn.Module):
    """Potential function loss: harmonic attraction + log-barrier repulsion.
    Ref: MATH.md M.3.9
    """

    def __init__(self, k_a: float = 1.0, k_r: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.k_a = k_a
        self.k_r = k_r
        self.eps = eps

    def forward(
        self, embeddings: dict[str, torch.Tensor], centroid: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            embeddings: {modality: [B, d]} UNNORMALIZED
            centroid: [B, d] UNNORMALIZED
        Returns:
            loss: scalar (U_attract + U_repel)
            info: dict with components for logging
        """
        B = centroid.size(0)

        # --- U_attract: harmonic attraction to centroid ---
        # (1/NM) Σ_i Σ_m ||e_m^i - c_i||²
        attract = torch.tensor(0.0, device=centroid.device)
        M = 0
        for emb in embeddings.values():
            attract = attract + ((emb - centroid) ** 2).sum(dim=-1).mean()
            M += 1
        attract = attract / max(M, 1)
        U_attract = self.k_a * attract

        # --- U_repel: log-barrier repulsion between centroids ---
        # -(1/C(N,2)) Σ_{i<j} log(||c_i - c_j|| + ε)
        if B < 2:
            U_repel = torch.tensor(0.0, device=centroid.device)
        else:
            # Pairwise distances between centroids
            # [B, B] distance matrix
            diff = centroid.unsqueeze(0) - centroid.unsqueeze(1)  # [B, B, d]
            dists = diff.norm(dim=-1)  # [B, B]

            # Upper triangular (i < j)
            mask = torch.triu(torch.ones(B, B, device=centroid.device, dtype=torch.bool), diagonal=1)
            pairwise_dists = dists[mask]  # [C(N,2)]

            n_pairs = pairwise_dists.numel()
            log_dists = torch.log(pairwise_dists + self.eps)
            U_repel = -self.k_r * log_dists.sum() / max(n_pairs, 1)

        loss = U_attract + U_repel

        info = {
            "U_attract": U_attract.detach(),
            "U_repel": U_repel.detach() if isinstance(U_repel, torch.Tensor) else torch.tensor(U_repel),
        }

        return loss, info
