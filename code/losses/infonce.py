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
    MATH.md M.0.4: normalization applied at similarity computation time.

    Args:
        anchors: [B, d] embeddings (normalized internally)
        positives: [B, d] embeddings (normalized internally)
        tau: temperature scalar
    Returns:
        loss: scalar
    """
    # Normalize for cosine similarity (MATH.md M.0.4, Assumption A.2)
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)

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
    """E2: Centroid InfoNCE with modality dropout.
    Ref: MATH.md M.3.2

    Step 1: Full centroid c_i = (1/M) sum_m e_i^m, normalized.
    Step 2: Perturbed centroid c'_i via Bernoulli modality dropout.
    Step 3: L_E2 = InfoNCE(ĉ_i, ĉ'_i)

    At eval time: no dropout, loss = InfoNCE(ĉ, ĉ) (sanity baseline).
    """

    def __init__(self, tau: float = 0.07, p_drop: float = 0.3):
        super().__init__()
        self.tau = tau
        self.p_drop = p_drop

    def _perturbed_centroid(
        self, embeddings: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute perturbed centroid via modality dropout.

        MATH.md M.3.2:
            δ_i^m ~ Bernoulli(1 - p_drop)
            If all δ=0 for object i, set one random δ=1.
            c'_i = Σ_m δ_i^m · e_i^m / (Σ_m δ_i^m + ε)

        Args:
            embeddings: {modality: [B, d]} unnormalized embeddings
        Returns:
            perturbed_centroid: [B, d] normalized
        """
        mods = list(embeddings.keys())
        M = len(mods)
        B = embeddings[mods[0]].size(0)
        device = embeddings[mods[0]].device

        # Bernoulli mask: [B, M], 1 = keep, 0 = drop
        keep_prob = 1.0 - self.p_drop
        mask = torch.bernoulli(torch.full((B, M), keep_prob, device=device))  # [B, M]

        # Guarantee at least one modality per object
        all_zero = mask.sum(dim=1) == 0  # [B]
        if all_zero.any():
            rand_idx = torch.randint(0, M, (all_zero.sum().item(),), device=device)
            mask[all_zero, rand_idx] = 1.0

        # Weighted sum
        emb_stack = torch.stack([embeddings[m] for m in mods], dim=1)  # [B, M, d]
        mask_exp = mask.unsqueeze(-1)  # [B, M, 1]
        summed = (emb_stack * mask_exp).sum(dim=1)  # [B, d]
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)  # [B, 1]
        c_prime = summed / counts  # [B, d]
        c_prime = F.normalize(c_prime, dim=-1)

        return c_prime

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        centroid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            embeddings: {modality: [B, d]} — unnormalized from projection heads
            centroid: [B, d] — full centroid (normalized)
        Returns:
            dict with 'loss' and 'modality_losses'
        """
        # MATH.md M.3.2: L_E2 = InfoNCE(ĉ, ĉ')
        if self.training:
            c_prime = self._perturbed_centroid(embeddings)
        else:
            c_prime = centroid  # eval: no dropout

        loss = _infonce_loss(centroid, c_prime, self.tau)

        # Per-modality losses (diagnostic, not in MATH.md loss but useful for logging)
        modality_losses = {}
        for mod, emb in embeddings.items():
            emb_norm = F.normalize(emb, dim=-1)
            modality_losses[mod] = _infonce_loss(emb_norm, centroid, self.tau)

        return {"loss": loss, "modality_losses": modality_losses}
