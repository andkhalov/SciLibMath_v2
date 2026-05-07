"""BL2: PCGrad (Yu et al. NeurIPS 2020).

Gradient Surgery for Multi-Task Learning.

Algorithm:
1. Compute per-task gradients g_m
2. For each g_m, project against conflicting g_j (cos < 0)
3. Aggregate projected gradients
4. Apply standard optimizer step with aggregated gradient

No additional hyperparameters.
"""

import torch
import torch.nn as nn
import random
from models.constants import MODALITIES


class PCGradOptimizer:
    """PCGrad gradient surgery wrapper.

    Wraps around the standard optimizer to apply gradient projection
    before each parameter update step.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, modalities: list[str] = None):
        self.optimizer = optimizer
        self.modalities = modalities or list(MODALITIES)

    def step(
        self,
        modality_losses: dict[str, torch.Tensor],
        shared_params: list[nn.Parameter],
        global_loss: torch.Tensor = None,
        scaler=None,
    ) -> dict[str, float]:
        """Perform PCGrad update step.

        Args:
            modality_losses: {mod: L_m} per-modality losses (with grad graph)
            shared_params: list of model parameters to compute gradients on
            global_loss: additional loss (L_global + L_va) to add without surgery
            scaler: GradScaler for AMP (or None)

        Returns:
            dict with conflict statistics for logging
        """
        device = next(iter(modality_losses.values())).device

        # Step 1: Compute per-task gradients
        task_grads = {}
        active_mods = [mod for mod in self.modalities if mod in modality_losses]

        for mod in active_mods:
            loss_m = modality_losses[mod]
            if scaler:
                loss_m = scaler.scale(loss_m)
            grads = torch.autograd.grad(
                loss_m,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            # Flatten all gradients into single vector
            flat = []
            for g in grads:
                if g is not None:
                    if scaler:
                        flat.append(g.data.flatten() / scaler.get_scale())
                    else:
                        flat.append(g.data.flatten())
                else:
                    # Parameter not used by this modality - zero gradient
                    pass
            if flat:
                task_grads[mod] = torch.cat(flat)
            else:
                task_grads[mod] = torch.zeros(1, device=device)

        # Step 2-3: PCGrad projection
        conflict_count = 0
        total_checks = 0
        cosine_sum = 0.0

        projected_grads = {mod: task_grads[mod].clone() for mod in active_mods}

        for mod_i in active_mods:
            # Random ordering of other tasks (important per paper)
            others = [m for m in active_mods if m != mod_i]
            random.shuffle(others)

            for mod_j in others:
                total_checks += 1
                g_i = projected_grads[mod_i]
                g_j = task_grads[mod_j]  # use original (not projected) for reference

                dot = torch.dot(g_i, g_j)
                g_j_norm_sq = torch.dot(g_j, g_j)

                cos_ij = dot / (g_i.norm() * g_j.norm() + 1e-8)
                cosine_sum += cos_ij.item()

                if dot < 0:
                    # Conflict: project g_i onto normal plane of g_j
                    conflict_count += 1
                    projected_grads[mod_i] = g_i - (dot / (g_j_norm_sq + 1e-8)) * g_j

        # Aggregate projected gradients
        agg_grad = torch.zeros_like(list(projected_grads.values())[0])
        for g in projected_grads.values():
            agg_grad = agg_grad + g

        # Step 4: Add global loss gradient (without surgery)
        if global_loss is not None:
            if scaler:
                global_loss_scaled = scaler.scale(global_loss)
            else:
                global_loss_scaled = global_loss
            global_grads = torch.autograd.grad(
                global_loss_scaled,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            global_flat = []
            for g in global_grads:
                if g is not None:
                    if scaler:
                        global_flat.append(g.data.flatten() / scaler.get_scale())
                    else:
                        global_flat.append(g.data.flatten())
            if global_flat:
                agg_grad = agg_grad + torch.cat(global_flat)

        # Step 5: Apply aggregated gradient to model parameters
        self.optimizer.zero_grad()
        idx = 0
        for param in shared_params:
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            numel = param.numel()
            if idx + numel <= agg_grad.shape[0]:
                param.grad.data = agg_grad[idx:idx + numel].reshape(param.shape)
                idx += numel

        if scaler:
            # Re-scale gradients for the scaler
            for param in shared_params:
                if param.grad is not None:
                    param.grad.data *= scaler.get_scale()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()

        return {
            "conflict_count": conflict_count,
            "total_checks": total_checks,
            "conflict_ratio": conflict_count / max(total_checks, 1),
            "mean_cosine": cosine_sum / max(total_checks, 1),
        }
