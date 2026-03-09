"""InfoNCE loss variants: Pairwise (E1) and Centroid (E2).
Ref: MATH.md M.3.1 (E1), M.3.2 (E2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


def _infonce_loss(anchors: torch.Tensor, positives: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric InfoNCE loss for one pair of modalities.

    MATH.md M.3.1: L_pair = (L_NCE^{m,m'} + L_NCE^{m',m}) / 2

    Args:
        anchors: [B, d] normalized embeddings (modality m)
        positives: [B, d] normalized embeddings (modality m')
        tau: temperature scalar
    Returns:
        loss: scalar
    """
    # Similarity matrix [B, B]: sim[i,j] = <anchor_i, positive_j> / tau
    sim = torch.mm(anchors, positives.t()) / tau  # [B, B]

    B = sim.size(0)
    labels = torch.arange(B, device=sim.device)

    # Forward: anchor → positive
    loss_fwd = F.cross_entropy(sim, labels)
    # Backward: positive → anchor
    loss_bwd = F.cross_entropy(sim.t(), labels)

    return (loss_fwd + loss_bwd) / 2


class PairwiseInfoNCE(nn.Module):
    """E1: Pairwise InfoNCE over all C(M,2) modality pairs.
    Ref: MATH.md M.3.1

    L_E1 = (1/C(M,2)) * sum_{m<m'} L_pair(m, m')
    For M=5: C(5,2)=10 pairs, equal weights.
    """

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(self, embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            embeddings: {modality: [B, d]} L2-normalized embeddings
        Returns:
            dict with 'loss' (scalar) and 'pair_losses' (dict of per-pair losses)
        """
        modalities = list(embeddings.keys())
        pairs = list(combinations(modalities, 2))
        pair_losses = {}
        total = 0.0

        for m1, m2 in pairs:
            pair_loss = _infonce_loss(embeddings[m1], embeddings[m2], self.tau)
            pair_losses[f"{m1}_{m2}"] = pair_loss
            total = total + pair_loss

        total = total / len(pairs)
        return {"loss": total, "pair_losses": pair_losses}


class CentroidInfoNCE(nn.Module):
    """E2: Centroid InfoNCE — each modality vs centroid.
    Ref: MATH.md M.3.2

    L_E2 = (1/M) * sum_m L_NCE(e_m, c)
    where c_i = (1/M) * sum_m e_m^i (centroid of object i).
    """

    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(
        self, embeddings: dict[str, torch.Tensor], centroid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            embeddings: {modality: [B, d]}
            centroid: [B, d] object centroids
        Returns:
            dict with 'loss' and 'modality_losses'
        """
        modality_losses = {}
        total = 0.0

        for mod, emb in embeddings.items():
            mod_loss = _infonce_loss(emb, centroid, self.tau)
            modality_losses[mod] = mod_loss
            total = total + mod_loss

        total = total / len(embeddings)
        return {"loss": total, "modality_losses": modality_losses}
