"""Reproducibility: seed fixing for all RNGs."""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: random seed value
        deterministic: if True, force deterministic CUDA ops (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
