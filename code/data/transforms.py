"""Image transforms for visual modality.
Ref: MATH.md M.1.1 (visual encoder input)
"""

import torch
from torchvision import transforms
from PIL import Image


def get_image_transform(target_height: int = 64, max_width: int = 512):
    """Standard transform: grayscale → tensor → normalize → pad to fixed width.

    Images are already 64px height. We normalize and pad width.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # [1, H, W], values in [0, 1]
        transforms.Normalize(mean=[0.449], std=[0.226]),  # grayscale ImageNet stats
    ])


def pad_image_batch(images: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad list of [1, H, W] tensors to same width. Returns (batch, widths).

    Args:
        images: list of tensors with shape [1, H, W_i] (variable width)
    Returns:
        batch: [B, 1, H, max_W] padded tensor
        widths: [B] original widths
    """
    widths = torch.tensor([img.shape[-1] for img in images])
    max_w = widths.max().item()
    h = images[0].shape[-2]

    batch = torch.zeros(len(images), 1, h, max_w)
    for i, img in enumerate(images):
        w = img.shape[-1]
        batch[i, :, :, :w] = img

    return batch, widths
