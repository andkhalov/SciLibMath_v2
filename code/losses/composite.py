"""Composite loss for E3-E7 experiments.
Ref: MATH.md M.3.3 (E3), M.3.4 (E4), M.3.4* (loss anatomy)

Full loss hierarchy (MATH.md M.3.4*):
  L = w_g * [L_contrast + λ_align * L_align + λ_rad * L_rad + λ_reg * L_reg]
    + Σ_m w_m * [L_align^m + L_contrast^m + λ_reg * L_reg^m]
    + λ_va * L_visual_align

Two levels of weights:
  Level 1 (type): λ = {τ, λ_align, λ_rad, λ_reg, λ_va}
  Level 2 (modality): w = {w_en, w_ru, w_lean, w_latex, w_img, w_g}

[Key design per MATH.md M.0.4, M.3.3]:
  - embeddings and centroid arrive UNNORMALIZED from model
  - L_align and L_rad operate in unnormalized space
  - L_contrast normalizes internally for cosine similarity
  - L_reg normalizes centroids internally
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .infonce import CentroidInfoNCE, _infonce_loss
from .alignment import AlignmentLoss, RadialLoss, AntiCollapseLoss
from .potential import PotentialLoss
from models.constants import MODALITIES, TEXT_MODALITIES


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
        p_drop: float = 0.3,
        C_clip: float = 10.0,
        rho: float = 0.1,
        # Adaptive temperature (MATH.md M.3.3)
        alpha_tau: float = 0.5,
        tau_target: float = 0.0,
        tau_min: float = 0.01,
        tau_max: float = 0.5,
        # Potential loss (MATH.md M.3.9, E9/E10)
        use_potential: bool = False,
        k_a: float = 1.0,
        k_r: float = 0.1,
        # Contrast weight (EXP-006: boost contrast priority)
        contrast_weight: float = 1.0,
    ):
        super().__init__()

        # Store hyperparams (MATH.md M.3.6: λ_t ∈ R^11)
        self.tau = tau
        self.lambda_align = lambda_align
        self.lambda_rad = lambda_rad
        self.lambda_reg = lambda_reg
        self.lambda_va = lambda_va
        self.w_g = w_g
        self.use_potential = use_potential
        self.contrast_weight = contrast_weight

        # Adaptive temperature params (MATH.md M.3.3)
        self.alpha_tau = alpha_tau
        self.tau_target = tau_target
        self.tau_min = tau_min
        self.tau_max = tau_max

        # L_va curriculum scaling (MATH.md M.2.4, EXP-008)
        self._va_scale = 1.0

        # Per-modality weights (default uniform)
        default_w = {m: 1.0 for m in MODALITIES}
        self.modality_weights = modality_weights or default_w

        # Loss components
        self.centroid_nce = CentroidInfoNCE(tau=tau, p_drop=p_drop)
        self.alignment = AlignmentLoss(C_clip=C_clip)
        self.radial = RadialLoss(rho=rho)
        self.anti_collapse = AntiCollapseLoss()

        # Potential loss (E9/E10 — replaces alignment+radial)
        self.potential = PotentialLoss(k_a=k_a, k_r=k_r) if use_potential else None

    def _compute_adaptive_tau(self, centroid: torch.Tensor) -> float:
        """Compute adaptive temperature τ_eff from collapse score.
        Ref: MATH.md M.3.3

        τ_eff = clamp(τ · (1 - α_τ · (s̄_neg - τ_target)), τ_min, τ_max)
        """
        B = centroid.size(0)
        if B < 2:
            return self.tau

        centroid_norm = F.normalize(centroid, dim=-1)
        S = torch.mm(centroid_norm, centroid_norm.t())
        mask = ~torch.eye(B, dtype=torch.bool, device=S.device)
        s_neg = S[mask].mean().item()

        tau_eff = self.tau * (1.0 - self.alpha_tau * (s_neg - self.tau_target))
        tau_eff = max(self.tau_min, min(self.tau_max, tau_eff))
        return tau_eff

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        centroid: torch.Tensor,
        visual_align_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            embeddings: {modality: [B, d]} UNNORMALIZED (per MATH.md M.0.4)
            centroid: [B, d] UNNORMALIZED (per MATH.md M.0.4)
            visual_align_loss: scalar from AlignNet
        Returns:
            dict with 'loss' (total), all components, per-modality losses
        """
        result = {}

        # Adaptive temperature (MATH.md M.3.3)
        tau_eff = self._compute_adaptive_tau(centroid)
        self.centroid_nce.tau = tau_eff
        result["tau_eff"] = torch.tensor(tau_eff, device=centroid.device)

        # --- Global losses ---
        # Centroid InfoNCE (MATH.md M.3.2) — normalizes internally
        nce_out = self.centroid_nce(embeddings, centroid)
        L_contrast = nce_out["loss"]
        result["loss_contrast_global"] = L_contrast

        # Anti-collapse on normalized centroids (MATH.md M.3.3)
        L_reg = self.anti_collapse(centroid)
        result["loss_reg_global"] = L_reg

        # Branch: potential loss (E9/E10) vs alignment+radial (E3-E8)
        if self.potential is not None:
            # MATH.md M.3.9: L_potential = U_attract + U_repel
            L_potential, pot_info = self.potential(embeddings, centroid)
            result["loss_potential"] = L_potential
            result["U_attract"] = pot_info["U_attract"]
            result["U_repel"] = pot_info["U_repel"]
            # For compatibility: report zero align/rad
            result["loss_align_global"] = torch.tensor(0.0, device=centroid.device)
            result["loss_rad"] = torch.tensor(0.0, device=centroid.device)
            L_global = self.contrast_weight * L_contrast + L_potential + self.lambda_reg * L_reg
        else:
            # Alignment in unnormalized space (MATH.md M.3.3)
            L_align = self.alignment(embeddings, centroid)
            result["loss_align_global"] = L_align

            # Radial in unnormalized space (MATH.md M.3.3)
            L_rad = self.radial(embeddings, centroid)
            result["loss_rad"] = L_rad

            # Global composite
            L_global = self.contrast_weight * L_contrast + self.lambda_align * L_align + self.lambda_rad * L_rad + self.lambda_reg * L_reg

        # --- Per-modality personal losses (MATH.md M.3.4) ---
        L_personal = torch.tensor(0.0, device=centroid.device)

        # Pre-compute adaptive weights for personal alignment (same as global)
        dists = {}
        mods = list(embeddings.keys())
        for mod in mods:
            dists[mod] = (embeddings[mod] - centroid).norm(dim=-1)  # [B]
        dist_sum = sum(dists.values()) + 1e-9  # [B]

        for mod in MODALITIES:
            w_m = self.modality_weights.get(mod, 1.0)
            emb = embeddings[mod]

            # Personal alignment: w_i^m · min(||e_m - c||², C_clip) (MATH.md M.3.4)
            w_i_m = dists[mod] / dist_sum  # [B]
            sq_dist = ((emb - centroid) ** 2).sum(dim=-1)  # [B]
            personal_align = (w_i_m * torch.clamp(sq_dist, max=self.alignment.C_clip)).mean()

            # Personal contrast: InfoNCE(e_m, c) — normalizes internally
            personal_contrast = _infonce_loss(emb, centroid, tau_eff)

            # Personal anti-collapse on per-modality embeddings (MATH.md M.3.4)
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
        # MATH.md M.3.4*: L = w_g * L_global + L_personal + λ_va * va_scale * L_va
        L_total = self.w_g * L_global + L_personal + self.lambda_va * self._va_scale * visual_align_loss

        result["loss"] = L_total
        result["loss_global"] = L_global
        result["loss_personal"] = L_personal
        result["loss_va"] = visual_align_loss
        result["modality_losses"] = nce_out.get("modality_losses", {})

        return result

    def get_lambda_vector(self) -> torch.Tensor:
        """Return current λ_t ∈ R^11 (MATH.md M.3.6).

        [τ, λ_align/k_a, λ_rad/k_r, λ_reg, λ_va, w_en, w_ru, w_lean, w_latex, w_img, w_g]

        For E9/E10 (use_potential=True): positions 1,2 hold k_a, k_r instead of λ_align, λ_rad.
        """
        if self.potential is not None:
            align_or_ka = self.potential.k_a
            rad_or_kr = self.potential.k_r
        else:
            align_or_ka = self.lambda_align
            rad_or_kr = self.lambda_rad
        return torch.tensor([
            self.tau,
            align_or_ka,
            rad_or_kr,
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
        """Update hyperparameters from λ_t ∈ R^11 (for fuzzy controller E6-E8, E10)."""
        self.tau = lam[0].item()
        if self.potential is not None:
            self.potential.k_a = lam[1].item()
            self.potential.k_r = lam[2].item()
        else:
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

    def set_va_scale(self, scale: float):
        """Scale λ_va for curriculum scheduling (MATH.md M.2.4). scale ∈ [0, 1]."""
        self._va_scale = scale
