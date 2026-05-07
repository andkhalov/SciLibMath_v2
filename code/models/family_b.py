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
        visual_align_targets: list[str] | None = None,
        align_hidden_dim: int = 512,
        align_dropout: float = 0.1,
        freeze_resnet_layers: int = 2,
        shared_vocab_size: int | None = None,
    ):
        super().__init__()
        self.visual_align_targets = visual_align_targets or ["latex"]

        # 1 shared text encoder for all 4 text modalities (MATH.md M.1.3)
        self.text_encoder = TextEncoder(
            model_name=text_backbone,
            vocab_size=shared_vocab_size,
        )

        # Visual encoder (identical to Family A, MATH.md M.1.2) with EXP-003 fixes
        self.visual_encoder = VisualEncoder(
            text_backbone_name=text_backbone,
            resnet_name=visual_backbone,
            pretrained=visual_pretrained,
            patch_size=visual_patch_size,
            patch_stride=visual_patch_stride,
            align_hidden_dim=align_hidden_dim,
            align_dropout=align_dropout,
            freeze_resnet_layers=freeze_resnet_layers,
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

        # Visual alignment loss (MATH.md M.1.2*) — multi-target (EXP-003 Fix P4)
        va_loss = torch.tensor(0.0, device=visual_pooled.device)
        for target_mod in self.visual_align_targets:
            t_ids = batch[f"{target_mod}_input_ids"]
            t_mask = batch[f"{target_mod}_attention_mask"]
            t_embed_layer = self.text_encoder.get_embedding_layer()
            if t_embed_layer is not None:
                with torch.no_grad():
                    t_token_embs = t_embed_layer(t_ids)  # [B, L, d_text]
                mask_exp = t_mask.unsqueeze(-1).float()
                t_pooled = (t_token_embs * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
            else:
                with torch.no_grad():
                    t_pooled = self.text_encoder(t_ids, t_mask)
            va_loss = va_loss + self.visual_align_loss_fn(visual_pooled, t_pooled.detach())
        va_loss = va_loss / len(self.visual_align_targets)

        return {
            "embeddings": embeddings,
            "centroid": centroid,
            "centroid_norm": centroid_norm,
            "visual_align_loss": va_loss,
        }

    def get_param_groups(self, lr: float, lr_embed_ratio: float = 0.1,
                         lr_visual_ratio: float = 0.5):
        """Parameter groups with discriminative learning rates (MATH.md M.2.4).

        Three groups (EXP-003 Fix P3):
          0: text backbone — lr × lr_embed_ratio (0.1)
          1: visual pipeline (unfrozen ResNet + AlignNet + transformer) — lr × lr_visual_ratio (0.5)
          2: projection heads — lr (full)
        """
        # Group 0: text encoder backbone
        text_backbone_params = list(self.text_encoder.backbone.parameters())
        text_backbone_ids = {id(p) for p in text_backbone_params}

        # Group 1: visual encoder (only trainable params)
        visual_params = [p for p in self.visual_encoder.parameters() if p.requires_grad]
        visual_ids = {id(p) for p in visual_params}

        # Group 2: projection heads
        head_params = list(self.proj_sym.parameters()) + list(self.proj_img.parameters())
        head_ids = {id(p) for p in head_params}

        # Remove overlaps
        text_backbone_params = [p for p in text_backbone_params
                                if id(p) not in head_ids and id(p) not in visual_ids]
        visual_params = [p for p in visual_params if id(p) not in head_ids]

        return [
            {"params": text_backbone_params, "lr": lr * lr_embed_ratio},
            {"params": visual_params, "lr": lr * lr_visual_ratio},
            {"params": head_params, "lr": lr},
        ]
