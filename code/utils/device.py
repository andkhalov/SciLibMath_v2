"""Device setup and mixed precision context."""

import torch


def get_device(cfg_device: str = "cuda") -> torch.device:
    """Get torch device, fallback to CPU if CUDA unavailable."""
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_amp_context(enabled: bool, device: torch.device):
    """Get autocast context manager for mixed precision.

    Returns (autocast_ctx, GradScaler_or_None).
    """
    if enabled and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        return torch.amp.autocast("cuda", dtype=torch.float16), scaler
    # Dummy context that does nothing
    return torch.amp.autocast(device.type, enabled=False), None
