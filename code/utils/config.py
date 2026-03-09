"""Config loading and merging via OmegaConf."""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load experiment config, merge with base.yaml, apply CLI overrides.

    Args:
        config_path: path to experiment YAML (e.g. configs/e4_composite_static.yaml)
        overrides: list of "key=value" strings from CLI
    Returns:
        Merged DictConfig
    """
    config_path = Path(config_path)
    base_path = config_path.parent / "base.yaml"

    # Load base config
    if base_path.exists():
        base_cfg = OmegaConf.load(base_path)
    else:
        base_cfg = OmegaConf.create({})

    # Load experiment config
    exp_cfg = OmegaConf.load(config_path)

    # Merge: experiment overrides base
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    # Apply CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save config to YAML file (for reproducibility)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
