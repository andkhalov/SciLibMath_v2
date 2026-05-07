"""BL1: GradNorm (Chen et al. ICML 2018).

Gradient Normalization for Adaptive Loss Balancing.

Algorithm:
1. Maintain learnable weights w_m per modality
2. Compute gradient norms G_m = ||∇_W(w_m * L_m)||_2 on shared params
3. Compute loss ratio r_m = (L_m(t)/L_m(0)) / mean(L_j(t)/L_j(0))
4. L_grad = Σ |G_m - G_bar * r_m^α|
5. Update w_m via gradient of L_grad, renormalize to sum=T
"""

import torch
import torch.nn as nn
from models.constants import MODALITIES


class GradNormBalancer(nn.Module):
    """GradNorm adaptive loss balancing.

    Adjusts per-modality weights based on gradient norm matching
    and relative inverse training rates.
    """

    def __init__(
        self,
        modalities: list[str] = None,
        alpha: float = 1.5,
        lr_w: float = 0.025,
    ):
        super().__init__()
        self.modalities = modalities or list(MODALITIES)
        self.alpha = alpha
        self.lr_w = lr_w
        self.T = len(self.modalities)

        # Learnable task weights, init=1.0
        self.task_weights = nn.ParameterDict({
            mod: nn.Parameter(torch.tensor(1.0))
            for mod in self.modalities
        })

        # Initial losses L_m(0) — set on first step
        self.initial_losses: dict[str, float] = {}
        self._initialized = False

    def get_weights(self) -> dict[str, float]:
        """Return current w_m values."""
        return {mod: self.task_weights[mod].item() for mod in self.modalities}

    def step(
        self,
        modality_losses: dict[str, torch.Tensor],
        model: nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Perform one GradNorm update step.

        Args:
            modality_losses: {mod: L_m} per-modality losses (detached scalars or requires_grad)
            model: the model whose parameters we compute gradient norms on

        Returns:
            dict with weighted losses and GradNorm diagnostics
        """
        device = next(iter(modality_losses.values())).device

        # Record initial losses on first call
        if not self._initialized:
            self.initial_losses = {mod: modality_losses[mod].detach().item()
                                   for mod in self.modalities if mod in modality_losses}
            self._initialized = True

        # Step 1: Compute weighted total loss
        weighted_losses = {}
        for mod in self.modalities:
            if mod in modality_losses:
                weighted_losses[mod] = self.task_weights[mod] * modality_losses[mod]

        # Step 2: Compute gradient norms G_m = ||∇_W(w_m * L_m)||_2
        # We compute on ALL model parameters (Family A has no single shared layer)
        grad_norms = {}
        shared_params = [p for p in model.parameters() if p.requires_grad]

        for mod in self.modalities:
            if mod not in weighted_losses:
                continue
            # Compute gradients of w_m * L_m w.r.t. model params
            grads = torch.autograd.grad(
                weighted_losses[mod],
                shared_params,
                retain_graph=True,
                create_graph=True,  # need grad through G_m for L_grad
                allow_unused=True,
            )
            # L2 norm of concatenated gradients
            gnorm = torch.tensor(0.0, device=device)
            for g in grads:
                if g is not None:
                    gnorm = gnorm + g.norm() ** 2
            grad_norms[mod] = gnorm.sqrt()

        if not grad_norms:
            return {"weighted_total": torch.tensor(0.0, device=device)}

        # Step 3: Average gradient norm
        G_bar = torch.stack(list(grad_norms.values())).mean()

        # Step 4: Loss ratios (inverse training rate)
        loss_ratios = {}
        for mod in self.modalities:
            if mod in modality_losses and mod in self.initial_losses:
                L_init = self.initial_losses[mod]
                if L_init > 1e-8:
                    loss_ratios[mod] = modality_losses[mod].detach().item() / L_init
                else:
                    loss_ratios[mod] = 1.0

        if loss_ratios:
            mean_ratio = sum(loss_ratios.values()) / len(loss_ratios)
            if mean_ratio > 1e-8:
                r = {mod: loss_ratios[mod] / mean_ratio for mod in loss_ratios}
            else:
                r = {mod: 1.0 for mod in loss_ratios}
        else:
            r = {mod: 1.0 for mod in self.modalities}

        # Step 5: GradNorm loss
        # L_grad = Σ |G_m - G_bar * r_m^α|
        # Target is treated as constant (stop gradient on G_bar * r^α)
        L_grad = torch.tensor(0.0, device=device)
        for mod in self.modalities:
            if mod in grad_norms and mod in r:
                target = (G_bar * (r[mod] ** self.alpha)).detach()
                L_grad = L_grad + torch.abs(grad_norms[mod] - target)

        # Step 6: Update w_m via gradient of L_grad
        # Only update task_weights, not model params
        if L_grad.requires_grad:
            w_grads = torch.autograd.grad(
                L_grad,
                list(self.task_weights.values()),
                retain_graph=True,
                allow_unused=True,
            )
            with torch.no_grad():
                for (mod, param), grad in zip(self.task_weights.items(), w_grads):
                    if grad is not None:
                        param.data -= self.lr_w * grad

        # Step 7: Renormalize to sum = T
        with torch.no_grad():
            w_sum = sum(p.data.item() for p in self.task_weights.values())
            if w_sum > 1e-8:
                for p in self.task_weights.values():
                    p.data *= self.T / w_sum
            # Clamp to positive
            for p in self.task_weights.values():
                p.data.clamp_(min=0.01)

        return {
            "L_grad": L_grad.detach(),
            "G_bar": G_bar.detach(),
            "grad_norms": {mod: gn.detach().item() for mod, gn in grad_norms.items()},
            "loss_ratios": r,
        }
