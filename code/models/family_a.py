"""Family A: 5 independent encoders + projection heads.
Ref: MATH.md M.1, M.1.3

Family A has separate encoder per modality:
  en, ru, lean, latex → TextEncoder + ProjectionHead
  img → VisualEncoder (patches→ResNet→AlignNet→SciRus-tiny) + ProjectionHead

Output: dict of UNNORMALIZED embeddings {modality: [B, d]} and centroid [B, d].
Per MATH.md M.0.4: normalization applied downstream at similarity computation time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TextEncoder, VisualEncoder
from .projections import ProjectionHead
from .constants import MODALITIES, TEXT_MODALITIES
from losses.visual_align import VisualAlignLoss


class FamilyA(nn.Module):
    """5-encoder multimodal model (Family A).

    Each modality has its own encoder and projection head.
    MATH.md M.0.4: centroid c_i = mean(unnormalized), then normalize separately.
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
        lean_vocab_size: int | None = None,
        latex_vocab_size: int | None = None,
    ):
        super().__init__()

        # Text encoders (MATH.md M.1), with optional vocab extension (M.2.3)
        _vocab_sizes = {"lean": lean_vocab_size, "latex": latex_vocab_size}
        self.text_encoders = nn.ModuleDict()
        for mod in TEXT_MODALITIES:
            self.text_encoders[mod] = TextEncoder(
                model_name=text_backbone,
                vocab_size=_vocab_sizes.get(mod),
            )

        # Visual encoder (MATH.md M.1.2)
        self.visual_encoder = VisualEncoder(
            text_backbone_name=text_backbone,
            resnet_name=visual_backbone,
            pretrained=visual_pretrained,
            patch_size=visual_patch_size,
            patch_stride=visual_patch_stride,
        )

        # Projection heads: Linear(d', d) per MATH.md M.1.1
        text_d = self.text_encoders["en"].output_dim
        vis_d = self.visual_encoder.output_dim  # d_text (312)

        self.projections = nn.ModuleDict()
        for mod in TEXT_MODALITIES:
            self.projections[mod] = ProjectionHead(text_d, embedding_dim)
        self.projections["img"] = ProjectionHead(vis_d, embedding_dim)

        # Visual alignment loss (MATH.md M.1.2*)
        self.visual_align_loss_fn = VisualAlignLoss(margin=visual_align_margin)

        self.embedding_dim = embedding_dim
        self._text_backbone_name = text_backbone

    def encode(self, batch: dict) -> dict[str, torch.Tensor]:
        """Encode all modalities in a batch.

        Args:
            batch: collated batch from MultimodalCollator
        Returns:
            embeddings: {modality: [B, d]} UNNORMALIZED (per MATH.md M.0.4)
        """
        embeddings = {}

        # Text modalities
        for mod in TEXT_MODALITIES:
            ids = batch[f"{mod}_input_ids"]
            mask = batch[f"{mod}_attention_mask"]
            features = self.text_encoders[mod](ids, mask)  # [B, d']
            embeddings[mod] = self.projections[mod](features)  # [B, d], NOT normalized

        # Visual modality
        img = batch["img"]  # [B, 1, H, W]
        widths = batch["img_widths"]  # [B]
        vis_features = self.visual_encoder(img, widths)  # [B, d_text]
        embeddings["img"] = self.projections["img"](vis_features)  # [B, d], NOT normalized

        return embeddings

    def forward(self, batch: dict) -> dict:
        """Full forward: encode → compute centroid → compute alignment loss.

        MATH.md M.0.4: centroid from UNNORMALIZED embeddings, then normalize.
        Uses forward_with_aligned() to avoid double ResNet computation (Fix #6).

        Returns dict with:
            embeddings: {modality: [B, d]} UNNORMALIZED
            centroid: [B, d] UNNORMALIZED
            centroid_norm: [B, d] L2-normalized
            visual_align_loss: scalar
        """
        embeddings = {}

        # Text modalities
        for mod in TEXT_MODALITIES:
            ids = batch[f"{mod}_input_ids"]
            mask = batch[f"{mod}_attention_mask"]
            features = self.text_encoders[mod](ids, mask)  # [B, d']
            embeddings[mod] = self.projections[mod](features)  # [B, d]

        # Visual modality — single pass through ResNet (Fix #6)
        img = batch["img"]
        widths = batch["img_widths"]
        vis_features, visual_pooled = self.visual_encoder.forward_with_aligned(img, widths)
        embeddings["img"] = self.projections["img"](vis_features)  # [B, d]

        # Centroid: MATH.md M.0.4 — mean of UNNORMALIZED, then normalize separately
        emb_stack = torch.stack([embeddings[m] for m in MODALITIES], dim=0)  # [M, B, d]
        centroid = emb_stack.mean(dim=0)  # [B, d] UNNORMALIZED
        centroid_norm = F.normalize(centroid, dim=-1)  # [B, d] normalized

        # Visual alignment loss (MATH.md M.1.2*)
        # visual_pooled: already computed above (mean-pooled AlignNet output)
        # latex_pooled: mean-pooled token embeddings from LaTeX encoder
        latex_ids = batch["latex_input_ids"]
        latex_mask = batch["latex_attention_mask"]
        latex_embed_layer = self.text_encoders["latex"].get_embedding_layer()
        if latex_embed_layer is not None:
            with torch.no_grad():
                latex_token_embs = latex_embed_layer(latex_ids)  # [B, L, d_text]
            # Masked mean pool
            mask_exp = latex_mask.unsqueeze(-1).float()
            latex_pooled = (latex_token_embs * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
        else:
            # Fallback: use full encoder output (detached)
            with torch.no_grad():
                latex_pooled = self.text_encoders["latex"](latex_ids, latex_mask)

        va_loss = self.visual_align_loss_fn(visual_pooled, latex_pooled.detach())

        return {
            "embeddings": embeddings,
            "centroid": centroid,          # UNNORMALIZED (for alignment/radial)
            "centroid_norm": centroid_norm, # normalized (for retrieval/logging)
            "visual_align_loss": va_loss,
        }

    def get_param_groups(self, lr: float, lr_embed_ratio: float = 0.1):
        """Parameter groups with discriminative learning rates (MATH.md M.2.4).

        Returns list of param groups for optimizer.
        """
        # Backbone params (lower lr) — text encoder backbones + visual encoder backbones
        backbone_params = []
        for mod in TEXT_MODALITIES:
            backbone_params.extend(self.text_encoders[mod].backbone.parameters())
        # Visual encoder: ResNet + SciRus-tiny transformer
        backbone_params.extend(self.visual_encoder.parameters())

        # Exclude AlignNet and projection params that are already in backbone_params
        backbone_ids = {id(p) for p in backbone_params}

        # Projection head params (full lr)
        head_params = []
        for proj in self.projections.values():
            head_params.extend(proj.parameters())

        # Filter out any overlap
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in backbone_params if id(p) not in head_ids]

        return [
            {"params": backbone_params, "lr": lr * lr_embed_ratio},
            {"params": head_params, "lr": lr},
        ]
