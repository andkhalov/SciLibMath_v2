"""Multimodal DataLoader with collation for 5 modalities.
Ref: MATH.md M.0 — batch of N objects, each with M=5 modalities.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .dataset import SciLibModalDataset
from .transforms import pad_image_batch


class MultimodalCollator:
    """Collates multimodal samples into batched tensors.

    Text modalities → tokenized (input_ids, attention_mask).
    Image modality → padded to max width.
    """

    def __init__(
        self,
        tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self._text_keys = ["en", "ru", "lean", "latex"]

    def __call__(self, samples: list[dict]) -> dict:
        batch = {}

        # Tokenize each text modality
        for key in self._text_keys:
            texts = [s[key] for s in samples]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch[f"{key}_input_ids"] = encoded["input_ids"]
            batch[f"{key}_attention_mask"] = encoded["attention_mask"]

        # Pad images to same width
        images = [s["img"] for s in samples]
        batch["img"], batch["img_widths"] = pad_image_batch(images)

        # Metadata
        batch["source"] = torch.tensor([s["source"] for s in samples])
        batch["row_id"] = torch.tensor([s["row_id"] for s in samples])

        return batch


def create_dataloaders(
    data_dir: str | Path,
    image_root: str | Path | None = None,
    batch_size: int = 64,
    test_fraction: float = 0.05,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_length: int = 256,
) -> tuple[DataLoader, DataLoader, int]:
    """Create train and test DataLoaders with fixed random split.

    Returns:
        (train_loader, test_loader, dataset_size)
    """
    # Full dataset (loaded once, then split by indices)
    full_ds = SciLibModalDataset(data_dir=data_dir, image_root=image_root)
    n = len(full_ds)

    # Fixed random split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_test = int(n * test_fraction)
    test_idx = indices[:n_test].tolist()
    train_idx = indices[n_test:].tolist()

    # Use Subset to avoid reloading Arrow data
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    test_ds = torch.utils.data.Subset(full_ds, test_idx)

    collator = MultimodalCollator(
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=False,
    )

    return train_loader, test_loader, n
