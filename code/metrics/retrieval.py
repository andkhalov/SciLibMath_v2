"""Retrieval metrics: R@k for centroid and cross-modal retrieval.
Ref: MATH.md M.8, TZ.md Sec 9
"""

import torch
import torch.nn.functional as F


def recall_at_k(
    queries: torch.Tensor,
    targets: torch.Tensor,
    k: int = 1,
) -> float:
    """Compute Recall@k: fraction of queries where correct target is in top-k.

    Args:
        queries: [N, d] query embeddings (normalized)
        targets: [N, d] target embeddings (normalized)
        k: top-k threshold
    Returns:
        R@k as float in [0, 1]
    """
    # Similarity matrix [N, N]
    sim = torch.mm(queries, targets.t())  # [N, N]
    N = sim.size(0)

    # For each query, find top-k indices
    _, topk_idx = sim.topk(k, dim=1)  # [N, k]

    # Check if correct index (diagonal) is in top-k
    correct = torch.arange(N, device=sim.device).unsqueeze(1)  # [N, 1]
    hits = (topk_idx == correct).any(dim=1).float()  # [N]

    return hits.mean().item()


@torch.no_grad()
def compute_retrieval_metrics(
    embeddings: dict[str, torch.Tensor],
    centroids: torch.Tensor,
    ks: list[int] = [1, 3, 10],
) -> dict[str, float]:
    """Compute full retrieval metrics suite.

    Ref: MATH.md M.8, evaluation protocol from CLAUDE.md Sec 7.

    Metrics:
      - centroid_R@k: query = centroid, target = centroid (self-retrieval baseline)
      - {mod}_to_centroid_R@k: query = modality embedding, target = centroid
      - {m1}_to_{m2}_R@k: cross-modal retrieval for selected pairs
    """
    results = {}

    # Centroid self-retrieval (sanity check — should be high)
    for k_val in ks:
        results[f"centroid_R@{k_val}"] = recall_at_k(centroids, centroids, k_val)

    # Modality → centroid retrieval
    for mod, emb in embeddings.items():
        for k_val in ks:
            results[f"{mod}_to_centroid_R@{k_val}"] = recall_at_k(emb, centroids, k_val)

    # Cross-modal retrieval (key pairs)
    cross_pairs = [("en", "ru"), ("en", "lean"), ("en", "latex"), ("lean", "latex")]
    for m1, m2 in cross_pairs:
        if m1 in embeddings and m2 in embeddings:
            for k_val in ks:
                results[f"{m1}_to_{m2}_R@{k_val}"] = recall_at_k(
                    embeddings[m1], embeddings[m2], k_val
                )

    return results
