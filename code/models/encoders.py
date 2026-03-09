"""Modality-specific encoders.
Ref: MATH.md M.1 — Architecture of 5 encoders f_m: X^m → R^d

Text encoders: transformer backbone → mean pooling → R^{d'}
Visual encoder: ResNet → adaptive avg pool → R^{d'}
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchvision import models as tv_models


class TextEncoder(nn.Module):
    """Transformer-based text encoder for one modality.

    Wraps a HuggingFace model. Supports partial unfreezing (M.2.4).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        unfreeze_ratio: float = 1.0,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.output_dim = self.backbone.config.hidden_size

        # Apply unfreezing strategy (MATH.md M.2.4)
        if unfreeze_ratio < 1.0:
            self._freeze_layers(unfreeze_ratio)

    def _freeze_layers(self, unfreeze_ratio: float):
        """Freeze bottom (1-ratio) of transformer layers."""
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            layers = self.backbone.encoder.layer
        elif hasattr(self.backbone, "layers"):
            layers = self.backbone.layers
        else:
            return  # Can't determine layers, skip

        n_layers = len(layers)
        n_freeze = int(n_layers * (1.0 - unfreeze_ratio))

        # Freeze embeddings
        if hasattr(self.backbone, "embeddings"):
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False

        # Freeze bottom layers
        for i, layer in enumerate(layers):
            if i < n_freeze:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
        Returns:
            pooled: [B, d'] mean-pooled hidden states
        """
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, d']

        # Mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (hidden * mask_expanded).sum(dim=1)  # [B, d']
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        pooled = summed / counts  # [B, d']

        return pooled


class VisualEncoder(nn.Module):
    """ResNet-based visual encoder.
    Ref: MATH.md M.1.1 — ResNet feature extraction + adaptive avg pool.

    Input: [B, 1, H, W] grayscale images (variable width, H=64).
    Output: [B, d'] feature vector.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
    ):
        super().__init__()

        # Load ResNet with best available pretrained weights
        weights = "DEFAULT" if pretrained else None
        if backbone_name == "resnet50":
            resnet = tv_models.resnet50(weights=weights)
        elif backbone_name == "resnet18":
            resnet = tv_models.resnet18(weights=weights)
        else:
            resnet = tv_models.resnet50(weights=weights)

        # Replace first conv to accept 1 channel (grayscale)
        # Original: Conv2d(3, 64, 7, 2, 3)
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Initialize from pretrained: average RGB channels
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Keep everything except conv1 and fc
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.output_dim = 512 if backbone_name == "resnet18" else 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] grayscale images
        Returns:
            features: [B, d']
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # [B, d', 1, 1]
        x = x.flatten(1)  # [B, d']
        return x
