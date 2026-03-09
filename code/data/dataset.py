"""Multimodal dataset wrapper for SciLibRuModal v2.
Ref: MATH.md M.0 (object definition o_i = {x_i^1, ..., x_i^M})

Each sample is a multimodal object with 5 modalities:
  en (English text), ru (Russian text), lean (formal statement),
  latex (formula), img (rendered image).
"""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from scilibrumodal_v2_data import load_scilibrumodal_v2, ImageLoadingConfig

from .transforms import get_image_transform


# Map source int to filename prefix
_SOURCE_PREFIX = {0: "herald_statements", 1: "atlas_statements", 2: "mathlib_statements"}


class SciLibModalDataset(Dataset):
    """PyTorch Dataset wrapping scilibrumodal-v2.

    Returns dict with keys: en, ru, lean, latex, img, source, row_id.
    Text fields are raw strings (tokenized in collator).
    Image is PIL or tensor depending on transform.
    """

    def __init__(
        self,
        data_dir: str | Path,
        image_root: str | Path | None = None,
        split: str = "train",
        indices: list[int] | None = None,
        image_transform=None,
    ):
        data_dir = Path(data_dir)
        image_root = Path(image_root) if image_root else data_dir / "img"

        # Load HuggingFace dataset
        img_cfg = ImageLoadingConfig(
            image_root=image_root,
            mode="pil",
            convert="L",
        ) if image_root.exists() else None

        dsd = load_scilibrumodal_v2(str(data_dir), images=img_cfg, normalize=True)
        self._ds = dsd[split]

        # Apply index subset (for train/test split)
        if indices is not None:
            self._ds = self._ds.select(indices)

        self._image_root = image_root
        self._image_transform = image_transform or get_image_transform()
        self._has_images = img_cfg is not None

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._ds[idx]
        sample = {
            "en": row["informal_statement_en"],
            "ru": row["informal_statement_ru"],
            "lean": row["formal_statement"],
            "latex": row["formula_md"],
            "source": row["source"],
            "row_id": row["row_id"],
        }

        # Image: apply transform
        if self._has_images and "image" in row and row["image"] is not None:
            img = row["image"]
            if self._image_transform:
                img = self._image_transform(img)
            sample["img"] = img
        else:
            # Fallback: try loading from disk
            source = row["source"]
            row_id = row["row_id"]
            prefix = _SOURCE_PREFIX.get(source, f"source{source}")
            img_path = self._image_root / f"{prefix}_row{row_id}.png"
            if img_path.exists():
                img = Image.open(img_path).convert("L")
                if self._image_transform:
                    img = self._image_transform(img)
                sample["img"] = img
            else:
                # Create dummy image tensor [1, 64, 64]
                sample["img"] = torch.zeros(1, 64, 64)

        return sample
