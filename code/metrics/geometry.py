"""Geometric metrics: D_intra, D_inter, collapse score, modality balance.
Ref: MATH.md M.3.3, M.8
"""

import torch
import torch.nn.functional as F

from models.family_a import MODALITIES


@torch.no_grad()
def compute_geometry_metrics(
    embeddings: dict[str, torch.Tensor],
    centroids: torch.Tensor,
) -> dict[str, float]:
    """Compute geometric quality metrics.

    Ref: MATH.md M.3.3, CLAUDE.md Sec 3.3

    Returns:
        D_intra: mean intra-object radius (lower = tighter clusters)
        D_inter: mean inter-centroid distance (higher = better separation)
        collapse_score: s̄_neg from MATH.md M.3.3 (lower = no collapse)
        modality_balance: Var_m(||e_m||) across modalities (lower = balanced)
    """
    results = {}
    B = centroids.size(0)

    # --- D_intra: (1/NM) Σ_i Σ_m ||e_m^i - c_i||^2 ---
    d_intra = 0.0
    n_mod = 0
    for mod, emb in embeddings.items():
        d_intra += ((emb - centroids) ** 2).sum(dim=-1).mean().item()
        n_mod += 1
    results["D_intra"] = d_intra / max(n_mod, 1)

    # --- D_inter: mean pairwise centroid distance ---
    centroid_norm = F.normalize(centroids, dim=-1)
    sim_matrix = torch.mm(centroid_norm, centroid_norm.t())  # [B, B]

    if B > 1:
        mask = ~torch.eye(B, dtype=torch.bool, device=centroids.device)
        # Distance = 1 - cosine_similarity
        distances = 1.0 - sim_matrix[mask]
        results["D_inter"] = distances.mean().item()

        # --- Collapse score: s̄_neg ---
        results["collapse_score"] = sim_matrix[mask].mean().item()
    else:
        results["D_inter"] = 0.0
        results["collapse_score"] = 0.0

    # --- Modality balance: Var_m(mean_norms) ---
    mod_norms = []
    for mod, emb in embeddings.items():
        mod_norms.append(emb.norm(dim=-1).mean().item())
    if mod_norms:
        t = torch.tensor(mod_norms)
        results["modality_balance"] = t.var().item()
    else:
        results["modality_balance"] = 0.0

    # --- Per-modality D_intra ---
    for mod, emb in embeddings.items():
        results[f"D_intra_{mod}"] = ((emb - centroids) ** 2).sum(dim=-1).mean().item()

    return results
