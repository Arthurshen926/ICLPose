"""Raw VFM token-bank records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Tuple


@dataclass(frozen=True)
class TokenLayerSpec:
    name: str
    model: str
    layer: str
    channels: int
    stride: int


@dataclass(frozen=True)
class TokenBankRecord:
    image_id: str
    token_path: Path
    layers: Tuple[TokenLayerSpec, ...]
    split: str
    scene: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def require_raw_tokens(self) -> None:
        for layer in self.layers:
            if layer.channels <= 0 or layer.stride <= 0:
                raise ValueError(f"invalid token layer spec: {layer.name}")
