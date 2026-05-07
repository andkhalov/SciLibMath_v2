"""Standalone evaluation script.
Loads checkpoint and runs full evaluation suite.

Usage:
    python code/evaluate.py --checkpoint checkpoints/e4_composite_static/best_model.pt
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils import get_device, set_seed
from data import create_dataloaders
from models import FamilyA, MODALITIES
from metrics import compute_retrieval_metrics, compute_geometry_metrics


@torch.no_grad()
def evaluate_full(model, test_loader, device, ks=[1, 3, 10]):
    """Full evaluation with detailed output."""
    model.eval()
    all_embeddings = {m: [] for m in MODALITIES}
    all_centroids = []

    for batch in test_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = model(batch)
        for m in MODALITIES:
            all_embeddings[m].append(out["embeddings"][m].cpu())
        all_centroids.append(out["centroid"].cpu())

    embeddings = {m: torch.cat(all_embeddings[m]) for m in MODALITIES}
    centroids = torch.cat(all_centroids)

    print(f"Evaluated {centroids.size(0)} samples")
    print(f"Embedding dim: {centroids.size(1)}")

    # Retrieval
    retrieval = compute_retrieval_metrics(embeddings, centroids, ks)
    print("\n=== Retrieval Metrics ===")
    for name, val in sorted(retrieval.items()):
        print(f"  {name}: {val:.4f}")

    # Geometry
    geometry = compute_geometry_metrics(embeddings, centroids)
    print("\n=== Geometry Metrics ===")
    for name, val in sorted(geometry.items()):
        print(f"  {name}: {val:.4f}")

    return {**retrieval, **geometry}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    device = get_device(cfg.device)
    set_seed(cfg.seed)

    # Data
    _, test_loader, _ = create_dataloaders(
        data_dir=cfg.data.dataset_path,
        image_root=cfg.data.get("image_root"),
        batch_size=args.batch_size,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.seed,
        tokenizer_name=cfg.model.get("text_backbone", "sentence-transformers/all-MiniLM-L6-v2"),
    )

    # Model
    model = FamilyA(
        text_backbone=cfg.model.text_backbone,
        visual_backbone=cfg.model.visual_backbone,
        visual_pretrained=False,  # weights from checkpoint
        embedding_dim=cfg.model.embedding_dim,
        projection_hidden_dim=cfg.model.projection_hidden_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    print(f"Loaded: {args.checkpoint}")
    print(f"Experiment: {cfg.experiment}, Epoch: {ckpt.get('epoch', '?')}")
    print(f"Best metric: {ckpt.get('best_metric', '?'):.4f}")

    evaluate_full(model, test_loader, device, cfg.eval.retrieval_k)


if __name__ == "__main__":
    main()
