"""TensorBoard logger for experiment tracking.
Ref: CLAUDE.md Sec 5 — what to log (scalars, histograms).
"""

import time
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf


class TBLogger:
    """Wrapper around TensorBoard SummaryWriter with experiment conventions."""

    def __init__(self, cfg: DictConfig):
        exp_name = cfg.get("experiment", "unknown")
        seed = cfg.get("seed", 0)
        run_name = f"{exp_name}_s{seed}_{int(time.time())}"

        log_dir = Path(cfg.logging.tensorboard_dir) / run_name
        self.writer = SummaryWriter(str(log_dir))
        self.log_dir = log_dir

        # Save config as text
        self.writer.add_text("config", f"```yaml\n{OmegaConf.to_yaml(cfg)}\n```", 0)

    def log_scalars(self, tag_prefix: str, scalars: dict, step: int):
        """Log multiple scalars under a prefix."""
        for name, value in scalars.items():
            if hasattr(value, "item"):
                value = value.item()
            self.writer.add_scalar(f"{tag_prefix}/{name}", value, step)

    def log_scalar(self, tag: str, value, step: int):
        if hasattr(value, "item"):
            value = value.item()
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int):
        self.writer.add_histogram(tag, values, step)

    def log_loss_components(self, loss_dict: dict, step: int):
        """Log all loss components from composite loss."""
        for name, value in loss_dict.items():
            if name == "loss":
                self.log_scalar("loss/total", value, step)
            elif name.startswith("loss_"):
                self.log_scalar(f"loss/{name}", value, step)
            elif name == "modality_losses":
                for mod, mloss in value.items():
                    self.log_scalar(f"loss/modality/{mod}", mloss, step)

    def log_training_stats(self, stats: dict, step: int):
        """Log training stats (lr, grad_norm, etc.)."""
        self.log_scalars("training", stats, step)

    def log_eval_metrics(self, metrics: dict, step: int):
        """Log evaluation metrics (retrieval, geometry)."""
        self.log_scalars("metrics", metrics, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
