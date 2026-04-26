from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch


def safe_torch_load(path: str | Path, *, map_location: Any = "cpu") -> Any:
    """Load trusted local checkpoints across PyTorch serialization defaults."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=map_location, weights_only=False)
