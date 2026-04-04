"""Shared SigLIP2 projection utilities for grounding-aware supervision."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class SigLIP2FeatureProjection(nn.Module):
    """Project RADIO 1280d features into SigLIP2 visual embedding space."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(1280, num_heads=16, init_values=1e-5)
            for _ in range(2)
        ])
        self.mlp_fc1 = nn.Linear(1280, 1520)
        self.mlp_final = nn.Sequential(
            nn.LayerNorm(1520),
            nn.GELU(),
            nn.Linear(1520, 1536),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 1280] -> [B, N, 1536]."""
        x = self.blocks(x)
        x = self.mlp_fc1(x)
        x = self.mlp_final(x)
        return x

    @classmethod
    def from_radio_checkpoint(cls, ckpt_path: str) -> "SigLIP2FeatureProjection":
        chk = torch.load(ckpt_path, map_location="cpu")
        sd = chk["state_dict"]
        proj = cls()
        proj_sd = {}
        prefix = "_feature_projections.siglip2-g."
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            new_k = k[len(prefix):]
            if new_k.startswith("mlp.fc1"):
                new_k = new_k.replace("mlp.fc1", "mlp_fc1")
            elif new_k.startswith("mlp.final"):
                new_k = new_k.replace("mlp.final", "mlp_final")
            proj_sd[new_k] = v.float()
        proj.load_state_dict(proj_sd, strict=True)
        return proj
