"""Composite loss for E3-E7 experiments.
Ref: MATH.md M.3.3 (E3), M.3.4 (E4), M.3.4* (loss anatomy)

Full loss hierarchy (MATH.md M.3.4*):
  L = w_g * [L_contrast + λ_align * L_align + λ_rad * L_rad + λ_reg * L_reg]
    + Σ_m w_m * [L_align^m + L_contrast^m + λ_reg * L_reg^m]
    + λ_va * L_visual_align

Two levels of weights:
  Level 1 (type): λ = {τ, λ_align, λ_rad, λ_reg, λ_va}
  Level 2 (modality): w = {w_en, w_ru, w_lean, w_latex, w_img, w_g}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .infonce import CentroidInfoNCE, _infonce_loss
from .alignment import AlignmentLoss, RadialLoss, AntiCollapseLoss
from models.family_a import MODALITIES, TEXT_MODALITIES


class CompositeLoss(nn.Module):
    """Full composite loss (E3/E4 with static weights).
    Ref: MATH.md M.3.4*

    Combines global contrastive + alignment + regularization
    with per-modality personal losses.
    """

    def __init__(
        self,
        tau: float = 0.07,
        lambda_align: float = 0.3,
        lambda_rad: float = 0.1,
        lambda_reg: float = 0.05,
        lambda_va: float = 0.1,
        modality_weights: dict[str, float] | None = None,
        w_g: float = 1.0,
    ):
        super().__init__()

        # Store hyperparams (MATH.md M.3.6: λ_t ∈ R^11)
        self.tau = tau
        self.lambda_align = lambda_align
        self.lambda_rad = lambda_rad
        self.lambda_reg = lambda_reg
        self.lambda_va = lambda_va
        self.w_g = w_g

        # Per-modality weights (default uniform)
        default_w = {m: 1.0 for m in MODALITIES}
        self.modality_weights = modality_weights or default_w

        # Loss components
        self.centroid_nce = CentroidInfoNCE(tau=tau)
        self.alignment = AlignmentLoss()
        self.radial = RadialLoss()
        self.anti_collapse = AntiCollapseLoss()

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        centroid: torch.Tensor,
        visual_align_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            embeddings: {modality: [B, d]} normalized
            centroid: [B, d] normalized
            visual_align_loss: scalar from AlignNet
        Returns:
            dict with 'loss' (total), all components, per-modality losses
        """
        result = {}

        # --- Global losses ---
        # Centroid InfoNCE (MATH.md M.3.2)
        nce_out = self.centroid_nce(embeddings, centroid)
        L_contrast = nce_out["loss"]
        result["loss_contrast_global"] = L_contrast

        # Alignment (MATH.md M.3.3)
        L_align = self.alignment(embeddings, centroid)
        result["loss_align_global"] = L_align

        # Radial (MATH.md M.3.3)
        L_rad = self.radial(centroid)
        result["loss_rad"] = L_rad

        # Anti-collapse (MATH.md M.3.3)
        L_reg = self.anti_collapse(centroid)
        result["loss_reg_global"] = L_reg

        # Global composite
        L_global = L_contrast + self.lambda_align * L_align + self.lambda_rad * L_rad + self.lambda_reg * L_reg

        # --- Per-modality personal losses (MATH.md M.3.4) ---
        L_personal = torch.tensor(0.0, device=centroid.device)
        for mod in MODALITIES:
            w_m = self.modality_weights.get(mod, 1.0)
            emb = embeddings[mod]

            # Personal alignment: ||e_m - c||^2
            personal_align = ((emb - centroid) ** 2).sum(dim=-1).mean()

            # Personal contrast: InfoNCE(e_m, c)
            personal_contrast = _infonce_loss(emb, centroid, self.tau)

            # Personal anti-collapse on per-modality embeddings
            emb_norm = F.normalize(emb, dim=-1)
            S_m = torch.mm(emb_norm, emb_norm.t())
            B = S_m.size(0)
            if B > 1:
                mask = ~torch.eye(B, dtype=torch.bool, device=S_m.device)
                personal_reg = F.relu(S_m[mask].mean())
            else:
                personal_reg = torch.tensor(0.0, device=centroid.device)

            L_m = w_m * (personal_align + personal_contrast + self.lambda_reg * personal_reg)
            L_personal = L_personal + L_m

            result[f"loss_{mod}_align"] = personal_align
            result[f"loss_{mod}_contrast"] = personal_contrast
            result[f"loss_{mod}_reg"] = personal_reg

        # --- Total ---
        # MATH.md M.3.4*: L = w_g * L_global + L_personal + λ_va * L_va
        L_total = self.w_g * L_global + L_personal + self.lambda_va * visual_align_loss

        result["loss"] = L_total
        result["loss_global"] = L_global
        result["loss_personal"] = L_personal
        result["loss_va"] = visual_align_loss
        result["modality_losses"] = nce_out.get("modality_losses", {})

        return result

    def get_lambda_vector(self) -> torch.Tensor:
        """Return current λ_t ∈ R^11 (MATH.md M.3.6).

        [τ, λ_align, λ_rad, λ_reg, λ_va, w_en, w_ru, w_lean, w_latex, w_img, w_g]
        """
        return torch.tensor([
            self.tau,
            self.lambda_align,
            self.lambda_rad,
            self.lambda_reg,
            self.lambda_va,
            self.modality_weights.get("en", 1.0),
            self.modality_weights.get("ru", 1.0),
            self.modality_weights.get("lean", 1.0),
            self.modality_weights.get("latex", 1.0),
            self.modality_weights.get("img", 1.0),
            self.w_g,
        ])

    def set_lambda_vector(self, lam: torch.Tensor):
        """Update hyperparameters from λ_t ∈ R^11 (for fuzzy controller E6-E7)."""
        self.tau = lam[0].item()
        self.lambda_align = lam[1].item()
        self.lambda_rad = lam[2].item()
        self.lambda_reg = lam[3].item()
        self.lambda_va = lam[4].item()
        self.modality_weights["en"] = lam[5].item()
        self.modality_weights["ru"] = lam[6].item()
        self.modality_weights["lean"] = lam[7].item()
        self.modality_weights["latex"] = lam[8].item()
        self.modality_weights["img"] = lam[9].item()
        self.w_g = lam[10].item()
