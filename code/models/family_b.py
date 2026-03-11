"""Family B: 1 shared symbolic encoder + 1 visual encoder.
Ref: MATH.md M.1.3

Family B uses a single shared TextEncoder for all 4 text modalities
(en, ru, lean, latex) and a separate VisualEncoder for images.

  Shared encoder → proj_sym → e_i^m ∈ R^d   (m ∈ {en, ru, lean, latex})
  Visual encoder → proj_img → e_i^img ∈ R^d

For retrieval, a one-hot modality tag is appended: ẽ_i^m = [e_i^m; onehot(m)].
For loss/centroid computation, raw e_i^m ∈ R^d is used (no tag).

E1-E4 only — fuzzy controller (E5-E7) requires separate modality weights.
~28M params (vs ~112M for Family A).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TextEncoder, VisualEncoder
from .projections import ProjectionHead
from .constants import MODALITIES, TEXT_MODALITIES
from losses.visual_align import VisualAlignLoss


class FamilyB(nn.Module):
    """2-encoder multimodal model (Family B).

    1 shared text encoder + 1 visual encoder, 2 projection heads.
    Interface identical to FamilyA: forward(), encode(), get_param_groups().
    """

    def __init__(
        self,
        text_backbone: str = "mlsa-iai-msu-lab/sci-rus-tiny3.5-zh",
        visual_backbone: str = "resnet18",
        visual_pretrained: bool = True,
        embedding_dim: int = 256,
        visual_patch_size: int = 64,
        visual_patch_stride: int = 32,
        visual_align_margin: float = 1.0,
        shared_vocab_size: int | None = None,
    ):
        super().__init__()

        # 1 shared text encoder for all 4 text modalities (MATH.md M.1.3)
        self.text_encoder = TextEncoder(
            model_name=text_backbone,
            vocab_size=shared_vocab_size,
        )

        # Visual encoder (identical to Family A, MATH.md M.1.2)
        self.visual_encoder = VisualEncoder(
            text_backbone_name=text_backbone,
            resnet_name=visual_backbone,
            pretrained=visual_pretrained,
            patch_size=visual_patch_size,
            patch_stride=visual_patch_stride,
        )

        # 2 projection heads: symbolic and visual
        text_d = self.text_encoder.output_dim
        vis_d = self.visual_encoder.output_dim

        self.proj_sym = ProjectionHead(text_d, embedding_dim)
        self.proj_img = ProjectionHead(vis_d, embedding_dim)

        # Visual alignment loss (MATH.md M.1.2*)
        self.visual_align_loss_fn = VisualAlignLoss(margin=visual_align_margin)

        self.embedding_dim = embedding_dim
        self._text_backbone_name = text_backbone

    def encode(self, batch: dict) -> dict[str, torch.Tensor]:
        """Encode all modalities.

        Args:
            batch: collated batch from MultimodalCollator
        Returns:
            embeddings: {modality: [B, d]} UNNORMALIZED (per MATH.md M.0.4)
        """
        embeddings = {}

        # Text modalities — shared encoder, shared projection
        for mod in TEXT_MODALITIES:
            ids = batch[f"{mod}_input_ids"]
            mask = batch[f"{mod}_attention_mask"]
            features = self.text_encoder(ids, mask)  # [B, d']
            embeddings[mod] = self.proj_sym(features)  # [B, d]

        # Visual modality
        vis_features = self.visual_encoder(batch["img"], batch["img_widths"])
        embeddings["img"] = self.proj_img(vis_features)

        return embeddings

    def forward(self, batch: dict) -> dict:
        """Full forward: encode → centroid → alignment loss.

        Returns dict with same keys as FamilyA:
            embeddings: {modality: [B, d]} UNNORMALIZED
            centroid: [B, d] UNNORMALIZED
            centroid_norm: [B, d] L2-normalized
            visual_align_loss: scalar
        """
        embeddings = {}

        # Text modalities — shared encoder + shared projection
        for mod in TEXT_MODALITIES:
            ids = batch[f"{mod}_input_ids"]
            mask = batch[f"{mod}_attention_mask"]
            features = self.text_encoder(ids, mask)
            embeddings[mod] = self.proj_sym(features)

        # Visual — single pass (Fix #6 pattern)
        vis_features, visual_pooled = self.visual_encoder.forward_with_aligned(
            batch["img"], batch["img_widths"]
        )
        embeddings["img"] = self.proj_img(vis_features)

        # Centroid: MATH.md M.0.4 — mean of UNNORMALIZED, then normalize separately
        emb_stack = torch.stack([embeddings[m] for m in MODALITIES], dim=0)  # [M, B, d]
        centroid = emb_stack.mean(dim=0)  # [B, d]
        centroid_norm = F.normalize(centroid, dim=-1)

        # Visual alignment loss (MATH.md M.1.2*)
        latex_ids = batch["latex_input_ids"]
        latex_mask = batch["latex_attention_mask"]
        latex_embed_layer = self.text_encoder.get_embedding_layer()
        if latex_embed_layer is not None:
            with torch.no_grad():
                latex_token_embs = latex_embed_layer(latex_ids)  # [B, L, d_text]
            mask_exp = latex_mask.unsqueeze(-1).float()
            latex_pooled = (latex_token_embs * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        else:
            with torch.no_grad():
                latex_pooled = self.text_encoder(latex_ids, latex_mask)

        va_loss = self.visual_align_loss_fn(visual_pooled, latex_pooled.detach())

        return {
            "embeddings": embeddings,
            "centroid": centroid,
            "centroid_norm": centroid_norm,
            "visual_align_loss": va_loss,
        }

    def get_param_groups(self, lr: float, lr_embed_ratio: float = 0.1):
        """Parameter groups with discriminative learning rates (MATH.md M.2.4)."""
        # Backbone params (lower lr)
        backbone_params = list(self.text_encoder.backbone.parameters())
        backbone_params.extend(self.visual_encoder.parameters())
        backbone_ids = {id(p) for p in backbone_params}

        # Projection head params (full lr)
        head_params = list(self.proj_sym.parameters()) + list(self.proj_img.parameters())
        head_params = [p for p in head_params if id(p) not in backbone_ids]

        return [
            {"params": backbone_params, "lr": lr * lr_embed_ratio},
            {"params": head_params, "lr": lr},
        ]
