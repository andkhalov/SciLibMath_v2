"""AlignNet: auxiliary visual-textual alignment module.
Ref: MATH.md M.1.2* — L_visual_align between centroid of text embeddings
and visual embedding, mediated by learned projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignNet(nn.Module):
    """Alignment network for visual modality.

    Computes alignment loss between text centroid and visual embedding
    via a learned projection. See MATH.md M.1.2*.

    L_visual_align = ||g_align(c_text) - e_img||^2
    where c_text = mean(e_en, e_ru, e_lean, e_latex).
    """

    def __init__(self, d: int, d_hidden: int = None):
        """
        Args:
            d: embedding dimension
            d_hidden: hidden dim of alignment projection (default: d)
        """
        super().__init__()
        d_hidden = d_hidden or d
        self.projection = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d),
        )

    def forward(
        self, text_embeddings: dict[str, torch.Tensor], img_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Compute visual alignment loss.

        Args:
            text_embeddings: {modality: [B, d]} for text modalities
            img_embedding: [B, d] visual embedding
        Returns:
            loss: scalar, mean squared distance
        """
        # Centroid of text embeddings
        text_stack = torch.stack(list(text_embeddings.values()), dim=0)  # [M_text, B, d]
        c_text = text_stack.mean(dim=0)  # [B, d]

        # Project and normalize
        c_proj = self.projection(c_text)
        c_proj = F.normalize(c_proj, dim=-1)
        img_norm = F.normalize(img_embedding, dim=-1)

        # MSE loss
        loss = ((c_proj - img_norm) ** 2).sum(dim=-1).mean()
        return loss
