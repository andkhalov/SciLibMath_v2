"""Contrastive visual alignment loss.
Ref: MATH.md M.1.2* — L_visual_align between AlignNet output and LaTeX token embeddings.

L_va(i)     = ||Pool(v_tilde)_i - Pool(Embed_latex)_i||^2            (positive)
L_va_neg(j) = max(0, δ - ||Pool(v_tilde)_i - Pool(Embed_latex)_j||) (negative, in-batch)

Total: L_va = (1/N) Σ_i [ L_va_pos(i) + (1/(N-1)) Σ_{j≠i} L_va_neg(i,j) ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualAlignLoss(nn.Module):
    """Contrastive visual alignment loss.

    Positive: MSE between visual_pooled and latex_pooled for same object.
    Negative: hinge loss pushing apart different objects.

    Args:
        margin: δ_va for hinge loss on negative pairs (default 1.0)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, visual_pooled: torch.Tensor, latex_pooled: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_pooled: [B, d_text] mean-pooled AlignNet output
            latex_pooled: [B, d_text] mean-pooled SciRus-tiny embedding layer output for latex
        Returns:
            loss: scalar (no lambda — lambda applied externally)
        """
        B = visual_pooled.size(0)

        # Positive: MSE per sample, then mean
        # ||v_i - t_i||^2
        pos_dist = ((visual_pooled - latex_pooled) ** 2).sum(dim=-1)  # [B]
        loss_pos = pos_dist.mean()

        if B < 2:
            return loss_pos

        # Negative: in-batch hinge
        # Pairwise L2 distances: [B, B]
        # dist[i,j] = ||v_i - t_j||
        diff = visual_pooled.unsqueeze(1) - latex_pooled.unsqueeze(0)  # [B, B, d]
        pair_dist = diff.norm(dim=-1)  # [B, B]

        # Hinge: max(0, margin - dist) for off-diagonal
        hinge = F.relu(self.margin - pair_dist)  # [B, B]

        # Zero out diagonal (positive pairs)
        diag_mask = torch.eye(B, dtype=torch.bool, device=visual_pooled.device)
        hinge = hinge.masked_fill(diag_mask, 0.0)

        loss_neg = hinge.sum() / (B * (B - 1))

        return loss_pos + loss_neg
