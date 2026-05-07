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

from .infonce import CentroidInfoNCE, PairwiseInfoNCE, _infonce_loss
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
        # EXP-009: pairwise vs centroid global contrastive
        contrast_mode: str = "centroid",
        # H55: alignment warmup (suppress alignment during early steps)
        align_warmup_steps: int = 0,
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
        self.align_warmup_steps = align_warmup_steps
        self._current_step = 0  # updated via set_step()

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

        # Contrast mode: "centroid" (default) or "pairwise" (EXP-009)
        self.contrast_mode = contrast_mode
        if contrast_mode == "pairwise":
            self.pairwise_nce = PairwiseInfoNCE(tau=tau)

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
        result["tau_eff"] = torch.tensor(tau_eff, device=centroid.device)

        # H55: alignment warmup — suppress alignment loss during early steps
        align_scale = 0.0 if (self.align_warmup_steps > 0 and self._current_step < self.align_warmup_steps) else 1.0

        # --- Global losses ---
        # Global contrastive: centroid (default) or pairwise (EXP-009)
        if self.contrast_mode == "pairwise":
            self.pairwise_nce.tau = tau_eff
            nce_out = self.pairwise_nce(embeddings)
        else:
            self.centroid_nce.tau = tau_eff
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

            # Global composite (H55: align_scale=0 during warmup)
            L_global = self.contrast_weight * L_contrast + align_scale * self.lambda_align * L_align + self.lambda_rad * L_rad + self.lambda_reg * L_reg

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

    def reaggregate_with_lambda(
        self,
        loss_dict: dict[str, torch.Tensor],
        lam: torch.Tensor,
        u_t: torch.Tensor = None,
        mu: float = 1.0,
        model_loss: bool = True,
    ) -> torch.Tensor:
        """Re-aggregate loss with differentiable λ for MLP consequent training (E8cf).

        Loss components are detached from model graph and re-combined with λ as
        a differentiable tensor. Gradient flows: L_reagg → λ → u_t → MLP params.
        Model params get gradient only from loss_dict["loss"] (original loss).

        Modality weights use raw u_t[5:10] (unclamped) → softmax × M: gradient
        always flows, never killed by project_to_bounds clamp.

        Hyperparameters lam[1:5], lam[10] (clamped) + quadratic regularizer.

        λ layout: [τ, λ_align, λ_rad, λ_reg, λ_va, w_en, w_ru, w_lean, w_latex, w_img, w_g]
        """
        M = len(MODALITIES)  # 5
        dev = loss_dict["loss_contrast_global"].device

        # --- Modality weights: softmax on RAW u_t (unclamped) ---
        # u_t comes directly from MLP, no clamp → gradient always flows.
        # If u_t not provided, fall back to lam (backward compat).
        if u_t is not None:
            w_logits = u_t[5:10]
        else:
            w_logits = lam[5:10]
        w_softmax = torch.softmax(w_logits, dim=0) * M  # sum = M
        w_m = {mod: w_softmax[i] for i, mod in enumerate(MODALITIES)}

        # --- Hyperparameters ---
        lam_align = lam[1]
        lam_rad = lam[2]
        lam_reg = lam[3]
        lam_va = lam[4]
        w_g = lam[10]

        # Quadratic regularizer toward defaults (differentiable)
        lam_defaults = torch.tensor([0.3, 0.1, 0.05, 0.01, 1.0], device=dev)
        lam_current = torch.stack([lam_align, lam_rad, lam_reg, lam_va, w_g])
        L_reg_hyper = mu * ((lam_current - lam_defaults) ** 2).sum()

        # --- Detach loss components: grad only through λ, not model ---
        L_contrast = loss_dict["loss_contrast_global"].detach()
        L_align = loss_dict.get("loss_align_global", torch.tensor(0.0, device=dev)).detach()
        L_rad = loss_dict.get("loss_rad", torch.tensor(0.0, device=dev)).detach()
        L_reg_gl = loss_dict["loss_reg_global"].detach()

        L_global = self.contrast_weight * L_contrast + lam_align * L_align + lam_rad * L_rad + lam_reg * L_reg_gl

        # Per-modality personal losses with softmax weights
        L_personal = torch.tensor(0.0, device=dev)
        for mod in MODALITIES:
            pa = loss_dict.get(f"loss_{mod}_align", torch.tensor(0.0, device=dev)).detach()
            pc = loss_dict.get(f"loss_{mod}_contrast", torch.tensor(0.0, device=dev)).detach()
            pr = loss_dict.get(f"loss_{mod}_reg", torch.tensor(0.0, device=dev)).detach()
            L_personal = L_personal + w_m[mod] * (pa + pc + lam_reg * pr)

        L_va = loss_dict.get("loss_va", torch.tensor(0.0, device=dev)).detach()

        L_reagg = w_g * L_global + L_personal + lam_va * self._va_scale * L_va + L_reg_hyper

        if model_loss:
            # Combined: loss_dict["loss"] → model grad, L_reagg → MLP grad
            return loss_dict["loss"] + L_reagg
        else:
            # L_reagg only (for separate backward in v6)
            return L_reagg

    def compute_derivative_mlp_loss(
        self,
        loss_dict: dict[str, torch.Tensor],
        u_t: torch.Tensor,
        s_t: torch.Tensor,
    ) -> torch.Tensor:
        """Per-modality relative training rate MLP signal (v7c).

        Signal: relative inverse training rate r_m = L_m / L_m(0).
        Never zero (while loss > 0). Higher r_m = modality trains slower.

        L_mlp = Σ_m (-u_t[5+m]) · r_m

        Gradient: ∂L/∂u_t[5+m] = -r_m < 0 → optimizer pushes u_t[5+m] UP
        proportionally to how slowly modality trains. MLP increases weight
        for lagging modalities, perpetually.
        """
        dev = u_t.device
        L_mlp = torch.tensor(0.0, device=dev)

        # s_t[8:13] = per-modality current losses
        # Use EMA as proxy for "initial" (EMA at beta=0.99 ≈ long-term average)
        # Relative rate: L_m / EMA_m — if > 1: worse than average, if < 1: better
        for i, mod in enumerate(MODALITIES):
            L_m = s_t[8 + i].detach()
            EMA_m = s_t[13 + i].detach()
            r_m = L_m / (EMA_m + 1e-8)  # relative training rate, always > 0
            L_mlp = L_mlp + (-u_t[5 + i]) * r_m

        return L_mlp

    def compute_derivative_per_rule_mlp_loss(
        self,
        loss_dict: dict[str, torch.Tensor],
        u_t: torch.Tensor,
        h_bar: torch.Tensor,
        controller,
        s_t: torch.Tensor,
    ) -> torch.Tensor:
        """Per-rule derivative-based MLP signal (v8b).

        Same principle as v7b but applied per-rule:
        Each MLP_r is penalized when its rule's "responsibility metric" worsens.

        R0 (Healthy): penalty = ||mlp_out|| when healthy (should output zero)
        R1 (Imbalance): penalty = mlp_w · ReLU(L_m - EMA_m) for highest-loss modality
        R2 (Collapse): penalty = mlp_out[3] · ReLU(collapse - EMA_collapse)
        R3 (Stagnation): penalty = mlp_out[1] · ReLU(L - EMA_L)
        R4 (Overshoot): penalty = mlp_out[0] · ReLU(ΔL)  (penalize when loss increases)
        R5 (Conflict): same as R1
        R6 (Visual): penalty = mlp_out[9] · ReLU(L_img - EMA_img)
        """
        dev = u_t.device
        rule_losses = torch.zeros(7, device=dev)

        # Extract signals from s_t
        L_t = s_t[0].detach()
        delta_L = s_t[1].detach()
        ema_L = s_t[2].detach()
        collapse = s_t[4].detach()

        # Per-modality: relative training rates r_m = L_m / EMA_m
        L_mods = s_t[8:13].detach()
        EMA_mods = s_t[13:18].detach()
        rel_rates = L_mods / (EMA_mods + 1e-8)  # [5], always > 0

        for r in range(7):
            if controller.nl_consequents is None:
                continue
            mlp_out = controller.nl_consequents[r](s_t)  # [11]

            if r == 0:  # Healthy: should not correct → penalize any output
                rule_losses[r] = mlp_out.norm() * 0.1
            elif r == 1:  # Imbalance: increase weight for slower modalities
                rule_losses[r] = (-mlp_out[5:10] * rel_rates).sum()
            elif r == 2:  # Collapse: should increase λ_reg
                rule_losses[r] = -mlp_out[3] * torch.relu(collapse - 0.01)
            elif r == 3:  # Stagnation: should increase λ_align
                r_total = L_t / (ema_L + 1e-8)
                rule_losses[r] = -mlp_out[1] * r_total
            elif r == 4:  # Overshoot: should increase τ when loss rising
                rule_losses[r] = -mlp_out[0] * torch.relu(delta_L)
            elif r == 5:  # Conflict: same as imbalance
                rule_losses[r] = (-mlp_out[5:10] * rel_rates).sum()
            elif r == 6:  # Visual: increase img weight proportional to its rel rate
                rule_losses[r] = -mlp_out[9] * rel_rates[4]

        # Weight by rule activation
        L_mlp = (h_bar.detach() * rule_losses).sum()
        return L_mlp

    @staticmethod
    def mlp_norm_reg(controller, min_norm: float = 5.0, gamma: float = 1.0) -> torch.Tensor:
        """Anti-collapse regularizer for MLP consequent weights.

        Penalizes only when total weight norm drops below min_norm.
        MLP is free to grow, but cannot collapse to zero output.

        L_reg = γ · max(0, min_norm - ||W||_total)²
        """
        total_norm_sq = torch.tensor(0.0, device=next(controller.nl_consequents.parameters()).device)
        for mlp in controller.nl_consequents:
            for p in mlp.parameters():
                total_norm_sq = total_norm_sq + p.norm() ** 2
        total_norm = total_norm_sq.sqrt()
        deficit = (min_norm - total_norm).clamp(min=0)
        return gamma * deficit ** 2

    def set_va_scale(self, scale: float):
        """Scale λ_va for curriculum scheduling (MATH.md M.2.4). scale ∈ [0, 1]."""
        self._va_scale = scale

    def set_step(self, step: int):
        """Set current global step for alignment warmup (H55)."""
        self._current_step = step
