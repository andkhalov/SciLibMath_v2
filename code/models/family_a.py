"""Family A: 5 independent encoders + projection heads.
Ref: MATH.md M.1, M.1.3

Family A has separate encoder per modality:
  en, ru, lean, latex → TextEncoder + ProjectionHead
  img → VisualEncoder + ProjectionHead

Output: dict of normalized embeddings {modality: [B, d]} and centroid [B, d].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import TextEncoder, VisualEncoder
from .projections import ProjectionHead
from .align_net import AlignNet

# MATH.md M.0: M=5 modalities
MODALITIES = ["en", "ru", "lean", "latex", "img"]
TEXT_MODALITIES = ["en", "ru", "lean", "latex"]


class FamilyA(nn.Module):
    """5-encoder multimodal model (Family A).

    Each modality has its own encoder and projection head.
    Centroid c_i = (1/M) * sum_m e_m^i (MATH.md M.0).
    """

    def __init__(
        self,
        text_backbone: str = "sentence-transformers/all-MiniLM-L6-v2",
        visual_backbone: str = "resnet50",
        visual_pretrained: bool = True,
        embedding_dim: int = 256,
        projection_hidden_dim: int = 512,
        text_unfreeze_ratio: float = 1.0,
    ):
        super().__init__()

        # Text encoders (MATH.md M.1)
        self.text_encoders = nn.ModuleDict()
        for mod in TEXT_MODALITIES:
            self.text_encoders[mod] = TextEncoder(
                model_name=text_backbone,
                unfreeze_ratio=text_unfreeze_ratio,
            )

        # Visual encoder (MATH.md M.1.1)
        self.visual_encoder = VisualEncoder(
            backbone_name=visual_backbone,
            pretrained=visual_pretrained,
        )

        # Projection heads (one per modality)
        text_d = self.text_encoders["en"].output_dim
        vis_d = self.visual_encoder.output_dim

        self.projections = nn.ModuleDict()
        for mod in TEXT_MODALITIES:
            self.projections[mod] = ProjectionHead(text_d, projection_hidden_dim, embedding_dim)
        self.projections["img"] = ProjectionHead(vis_d, projection_hidden_dim, embedding_dim)

        # AlignNet (MATH.md M.1.2*)
        self.align_net = AlignNet(embedding_dim)

        self.embedding_dim = embedding_dim

    def encode(self, batch: dict) -> dict[str, torch.Tensor]:
        """Encode all modalities in a batch.

        Args:
            batch: collated batch from MultimodalCollator
        Returns:
            embeddings: {modality: [B, d]} L2-normalized embeddings
        """
        embeddings = {}

        # Text modalities
        for mod in TEXT_MODALITIES:
            ids = batch[f"{mod}_input_ids"]
            mask = batch[f"{mod}_attention_mask"]
            features = self.text_encoders[mod](ids, mask)  # [B, d']
            embeddings[mod] = self.projections[mod](features)  # [B, d], normalized

        # Visual modality
        img = batch["img"]  # [B, 1, H, W]
        vis_features = self.visual_encoder(img)  # [B, d']
        embeddings["img"] = self.projections["img"](vis_features)  # [B, d], normalized

        return embeddings

    def forward(self, batch: dict) -> dict:
        """Full forward: encode → compute centroid → compute alignment loss.

        Returns dict with:
            embeddings: {modality: [B, d]}
            centroid: [B, d]
            visual_align_loss: scalar
        """
        embeddings = self.encode(batch)

        # Centroid: c_i = (1/M) * sum_m e_m^i (MATH.md M.0)
        emb_stack = torch.stack([embeddings[m] for m in MODALITIES], dim=0)  # [M, B, d]
        centroid = emb_stack.mean(dim=0)  # [B, d]
        centroid = F.normalize(centroid, dim=-1)  # re-normalize

        # Visual alignment loss (MATH.md M.1.2*)
        text_embs = {m: embeddings[m] for m in TEXT_MODALITIES}
        va_loss = self.align_net(text_embs, embeddings["img"])

        return {
            "embeddings": embeddings,
            "centroid": centroid,
            "visual_align_loss": va_loss,
        }

    def get_param_groups(self, lr: float, lr_embed_ratio: float = 0.1):
        """Parameter groups with discriminative learning rates (MATH.md M.2.4).

        Returns list of param groups for optimizer.
        """
        # Backbone params (lower lr)
        backbone_params = []
        for mod in TEXT_MODALITIES:
            backbone_params.extend(self.text_encoders[mod].backbone.parameters())
        backbone_params.extend(self.visual_encoder.parameters())

        # Projection + AlignNet params (full lr)
        head_params = []
        for proj in self.projections.values():
            head_params.extend(proj.parameters())
        head_params.extend(self.align_net.parameters())

        return [
            {"params": backbone_params, "lr": lr * lr_embed_ratio},
            {"params": head_params, "lr": lr},
        ]
