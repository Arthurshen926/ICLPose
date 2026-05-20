"""Small visualization helpers for POFD-FS diagnostics."""

from __future__ import annotations

import torch
from PIL import Image


def utility_to_uint8(utility: torch.Tensor) -> Image.Image:
    """Convert a utility map to a grayscale PIL image."""
    tensor = utility.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        tensor = tensor[0]
    tensor = tensor - tensor.min()
    denom = tensor.max().clamp_min(1.0e-6)
    array = (tensor / denom * 255.0).clamp(0, 255).byte().numpy()
    return Image.fromarray(array, mode="L")
