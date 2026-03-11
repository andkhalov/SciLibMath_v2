"""Modality-specific encoders.
Ref: MATH.md M.1 — Architecture of 5 encoders f_m: X^m → R^d

Text encoders: transformer backbone → mean pooling → R^{d'}
Visual encoder: patches → ResNet18 → AlignNet → SciRus-tiny → mean pool → R^{d'}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torchvision import models as tv_models


class TextEncoder(nn.Module):
    """Transformer-based text encoder for one modality.

    Wraps a HuggingFace model. All layers trainable (full fine-tuning).
    """

    def __init__(
        self,
        model_name: str = "mlsa-iai-msu-lab/sci-rus-tiny3.5-zh",
        vocab_size: int | None = None,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        # Resize embeddings for extended vocabulary (MATH.md M.2.3)
        if vocab_size is not None and vocab_size != self.backbone.config.vocab_size:
            self.backbone.resize_token_embeddings(vocab_size)

        self.output_dim = self.backbone.config.hidden_size

    def get_embedding_layer(self):
        """Return the token embedding layer for inputs_embeds extraction."""
        if hasattr(self.backbone, "embeddings"):
            return self.backbone.embeddings
        return None

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
    """Patch-based visual encoder: patches → ResNet18 → AlignNet → SciRus-tiny → pool.
    Ref: MATH.md M.1.2

    Pipeline:
      Step 1: Extract patches [B, 1, H, W] → [B, K, 1, P, P]
      Step 2: ResNet18 feature extraction per patch → [B, K, 512]
      Step 3: AlignNet: Linear(512, d_text) + LayerNorm → [B, K, d_text]
      Step 4: SciRus-tiny(inputs_embeds) + masked mean pool → [B, d_text]

    Input: [B, 1, H=64, W_padded] grayscale images + widths [B].
    Output: [B, d_text] feature vector (pre-projection).
    """

    def __init__(
        self,
        text_backbone_name: str = "mlsa-iai-msu-lab/sci-rus-tiny3.5-zh",
        resnet_name: str = "resnet18",
        pretrained: bool = True,
        patch_size: int = 64,
        patch_stride: int = 32,
        align_hidden_dim: int = 512,
        align_dropout: float = 0.1,
        freeze_resnet_layers: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.freeze_resnet_layers = freeze_resnet_layers

        # --- Step 2: ResNet18 feature extractor (no FC, no avgpool) ---
        weights = "DEFAULT" if pretrained else None
        if resnet_name == "resnet18":
            resnet = tv_models.resnet18(weights=weights)
            resnet_out_dim = 512
        else:
            resnet = tv_models.resnet50(weights=weights)
            resnet_out_dim = 2048

        # Replace first conv: 3 channels → 1 channel (grayscale)
        old_conv = resnet.conv1
        self.resnet_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.resnet_conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4
        self.resnet_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze early ResNet layers (EXP-003 Fix P3)
        if freeze_resnet_layers >= 1:
            for m in [self.resnet_conv1, self.resnet_bn1, self.resnet_layer1]:
                for p in m.parameters():
                    p.requires_grad = False
        if freeze_resnet_layers >= 2:
            for p in self.resnet_layer2.parameters():
                p.requires_grad = False

        # --- Step 3: AlignNet — MATH.md M.1.2 Step 3 ---
        # Deeper MLP with GELU nonlinearity (EXP-003 Fix P2)
        text_backbone = AutoModel.from_pretrained(text_backbone_name)
        d_text = text_backbone.config.hidden_size  # 312 for sci-rus-tiny

        self.align_net = nn.Sequential(
            nn.Linear(resnet_out_dim, align_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(align_hidden_dim),
            nn.Dropout(align_dropout),
            nn.Linear(align_hidden_dim, d_text),
            nn.LayerNorm(d_text),
        )

        # --- Step 4: SciRus-tiny for visual token processing ---
        # Separate instance — processes visual tokens via self-attention
        self.transformer = text_backbone  # reuse loaded model
        self.d_text = d_text
        self.output_dim = d_text

    def extract_patches(
        self, img: torch.Tensor, widths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract patches from variable-width images.

        Args:
            img: [B, 1, H, W_padded] padded grayscale images
            widths: [B] original widths before padding
        Returns:
            patches: [B, K_max, 1, P, P] extracted patches
            mask: [B, K_max] bool — True for valid patches
        """
        B, C, H, W = img.shape
        P = self.patch_size
        S = self.patch_stride

        # Max number of patches across batch
        K_max = max(1, (W - P) // S + 1)

        patches = img.new_zeros(B, K_max, C, P, P)
        mask = torch.zeros(B, K_max, dtype=torch.bool, device=img.device)

        for b in range(B):
            w_b = widths[b].item()
            k_b = max(1, (w_b - P) // S + 1)
            for k in range(min(k_b, K_max)):
                start = k * S
                end = start + P
                if end <= W:
                    patches[b, k] = img[b, :, :H, start:end]
                    mask[b, k] = True

        return patches, mask

    def _resnet_forward(self, x: torch.Tensor) -> torch.Tensor:
        """ResNet feature extraction (no FC).

        Args:
            x: [N, 1, P, P] patches
        Returns:
            features: [N, 512] (resnet18) or [N, 2048] (resnet50)
        """
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)
        x = self.resnet_pool(x)  # [N, C, 1, 1]
        x = x.flatten(1)  # [N, C]
        return x

    def forward(
        self, img: torch.Tensor, widths: torch.Tensor
    ) -> torch.Tensor:
        """Full visual encoding pipeline.

        Args:
            img: [B, 1, H, W_padded]
            widths: [B] original widths
        Returns:
            features: [B, d_text] — ready for projection head
        """
        pooled, _ = self.forward_with_aligned(img, widths)
        return pooled

    def forward_with_aligned(
        self, img: torch.Tensor, widths: torch.Tensor,
        resnet_chunk_size: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full pipeline returning BOTH transformer output AND AlignNet output.
        Avoids double ResNet computation (Fix #6).

        Args:
            img: [B, 1, H, W_padded]
            widths: [B] original widths
            resnet_chunk_size: max patches per ResNet forward (prevents OOM)
        Returns:
            pooled: [B, d_text] — transformer output (for projection head)
            aligned_pooled: [B, d_text] — AlignNet output (for L_visual_align)
        """
        B = img.size(0)

        # Step 1: Extract patches
        patches, mask = self.extract_patches(img, widths)  # [B, K, 1, P, P], [B, K]
        K = patches.size(1)

        # Step 2: ResNet per patch — chunked to prevent OOM with large B*K
        flat_patches = patches.view(B * K, 1, self.patch_size, self.patch_size)
        N_flat = flat_patches.size(0)

        if N_flat <= resnet_chunk_size:
            flat_features = self._resnet_forward(flat_patches)  # [N_flat, 512]
        else:
            chunks = []
            for start in range(0, N_flat, resnet_chunk_size):
                end = min(start + resnet_chunk_size, N_flat)
                chunks.append(self._resnet_forward(flat_patches[start:end]))
            flat_features = torch.cat(chunks, dim=0)

        patch_features = flat_features.view(B, K, -1)  # [B, K, 512]

        # Step 3: AlignNet → [B, K, d_text]
        aligned = self.align_net(patch_features)  # [B, K, d_text]

        # AlignNet pooled output (pre-transformer, for L_visual_align)
        mask_expanded = mask.unsqueeze(-1).float()  # [B, K, 1]
        aligned_summed = (aligned * mask_expanded).sum(dim=1)
        aligned_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        aligned_pooled = aligned_summed / aligned_counts  # [B, d_text]

        # Step 4: SciRus-tiny with inputs_embeds
        attn_mask = mask.long()  # [B, K]
        attn_mask[:, 0] = 1  # Ensure at least one token

        transformer_out = self.transformer(
            inputs_embeds=aligned,
            attention_mask=attn_mask,
        )
        hidden = transformer_out.last_hidden_state  # [B, K, d_text]

        # Masked mean pool (transformer output)
        t_mask_expanded = attn_mask.unsqueeze(-1).float()
        summed = (hidden * t_mask_expanded).sum(dim=1)  # [B, d_text]
        counts = t_mask_expanded.sum(dim=1).clamp(min=1e-9)  # [B, 1]
        pooled = summed / counts  # [B, d_text]

        return pooled, aligned_pooled

    def get_aligned_tokens(
        self, img: torch.Tensor, widths: torch.Tensor
    ) -> torch.Tensor:
        """Return AlignNet output BEFORE SciRus-tiny — for L_visual_align.
        Note: prefer forward_with_aligned() to avoid double computation.

        Args:
            img: [B, 1, H, W_padded]
            widths: [B] original widths
        Returns:
            aligned_pooled: [B, d_text] mean-pooled AlignNet features
        """
        _, aligned_pooled = self.forward_with_aligned(img, widths)
        return aligned_pooled
