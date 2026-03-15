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
        visual_align_targets: list[str] | None = None,
        align_hidden_dim: int = 512,
        align_dropout: float = 0.1,
        freeze_resnet_layers: int = 2,
        lean_vocab_size: int | None = None,
        latex_vocab_size: int | None = None,
    ):
        super().__init__()
        self.visual_align_targets = visual_align_targets or ["latex"]

        # Text encoders (MATH.md M.1), with optional vocab extension (M.2.3)
        _vocab_sizes = {"lean": lean_vocab_size, "latex": latex_vocab_size}
        self.text_encoders = nn.ModuleDict()
        for mod in TEXT_MODALITIES:
            self.text_encoders[mod] = TextEncoder(
                model_name=text_backbone,
                vocab_size=_vocab_sizes.get(mod),
            )

        # Visual encoder: patches → CNN → AlignNet → SciRus-tiny (MATH.md M.1.2)
        # visual_backbone selects CNN: resnet18, resnet50, convnext_pico.d1_in1k, etc.
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

        # Visual alignment loss (MATH.md M.1.2*) — multi-target (EXP-003 Fix P4)
        # visual_pooled: already computed above (mean-pooled AlignNet output)
        va_loss = torch.tensor(0.0, device=visual_pooled.device)
        for target_mod in self.visual_align_targets:
            t_ids = batch[f"{target_mod}_input_ids"]
            t_mask = batch[f"{target_mod}_attention_mask"]
            t_embed_layer = self.text_encoders[target_mod].get_embedding_layer()
            if t_embed_layer is not None:
                with torch.no_grad():
                    t_token_embs = t_embed_layer(t_ids)  # [B, L, d_text]
                mask_exp = t_mask.unsqueeze(-1).float()
                t_pooled = (t_token_embs * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
            else:
                with torch.no_grad():
                    t_pooled = self.text_encoders[target_mod](t_ids, t_mask)
            va_loss = va_loss + self.visual_align_loss_fn(visual_pooled, t_pooled.detach())
        va_loss = va_loss / len(self.visual_align_targets)

        return {
            "embeddings": embeddings,
            "centroid": centroid,          # UNNORMALIZED (for alignment/radial)
            "centroid_norm": centroid_norm, # normalized (for retrieval/logging)
            "visual_align_loss": va_loss,
        }

    def get_param_groups(self, lr: float, lr_embed_ratio: float = 0.1,
                         lr_visual_ratio: float = 0.5):
        """Parameter groups with discriminative learning rates (MATH.md M.2.4).

        Three groups (EXP-003 Fix P3):
          0: text backbones — lr × lr_embed_ratio (0.1)
          1: visual pipeline (unfrozen ResNet + AlignNet + transformer) — lr × lr_visual_ratio (0.5)
          2: projection heads — lr (full)
        """
        # Group 0: text encoder backbones
        text_backbone_params = []
        for mod in TEXT_MODALITIES:
            text_backbone_params.extend(self.text_encoders[mod].backbone.parameters())
        text_backbone_ids = {id(p) for p in text_backbone_params}

        # Group 1: visual encoder (only trainable params — frozen layers excluded)
        visual_params = [p for p in self.visual_encoder.parameters() if p.requires_grad]
        visual_ids = {id(p) for p in visual_params}

        # Group 2: projection heads
        head_params = []
        for proj in self.projections.values():
            head_params.extend(proj.parameters())
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
