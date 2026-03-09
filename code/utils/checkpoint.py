"""Checkpoint save/load with best-model tracking."""

import json
from pathlib import Path
from dataclasses import dataclass, field

import torch
from omegaconf import OmegaConf, DictConfig


@dataclass
class CheckpointState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_epoch: int = 0


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    state: CheckpointState,
    cfg: DictConfig,
    scaler=None,
) -> None:
    """Save training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler=None,
    scaler=None,
    device: torch.device = None,
) -> CheckpointState:
    """Load checkpoint and restore state."""
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return CheckpointState(
        epoch=ckpt.get("epoch", 0),
        global_step=ckpt.get("global_step", 0),
        best_metric=ckpt.get("best_metric", 0.0),
        best_epoch=ckpt.get("best_epoch", 0),
    )


def manage_checkpoints(ckpt_dir: str | Path, keep_best: int = 3) -> None:
    """Remove old checkpoints, keep only the best N by filename sort."""
    ckpt_dir = Path(ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    # Keep best_model.pt always; prune epoch checkpoints
    while len(ckpts) > keep_best:
        oldest = ckpts.pop(0)
        oldest.unlink()
