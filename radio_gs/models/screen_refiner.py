"""Screen-space feature refiner for correcting alpha-blending artifacts.

Alpha-blending in 3DGS produces over-smoothed features because each pixel is
a weighted average of overlapping Gaussians. This module learns to "undo" the
averaging in screen space using a lightweight residual CNN.

Architecture:
    rendered_64d → [Conv-BN-ReLU] × N blocks (with residual) → refined_64d
    Each block: Conv3×3 → BN → ReLU → Conv3×3 → BN → residual add → ReLU
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two 3×3 convolutions."""

    def __init__(self, channels: int, expansion: int = 1):
        super().__init__()
        mid = channels * expansion
        self.conv1 = nn.Conv2d(channels, mid, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ScreenSpaceRefiner(nn.Module):
    """Lightweight CNN that refines rendered latent features in screen space.

    The refiner takes alpha-blended feature maps (optionally with RGB guide)
    and produces corrected feature maps that, when decoded, better match the
    original RADIO features. Uses a residual architecture so it can start
    from identity.

    Args:
        latent_dim: Input/output feature dimension (default 64).
        hidden_dim: Hidden channel width (default 128).
        num_blocks: Number of residual blocks (default 4).
        dropout: Dropout rate for regularization (default 0.1).
        extra_channels: Additional input channels (e.g. 3 for RGB guide).
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        dropout: float = 0.1,
        extra_channels: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.extra_channels = extra_channels
        in_channels = latent_dim + extra_channels

        # Project to hidden dim
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        ]

        # Residual blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim))

        # Dropout for regularization
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Project back to latent dim (residual connection added in forward)
        layers.append(nn.Conv2d(hidden_dim, latent_dim, 1, bias=True))

        self.net = nn.Sequential(*layers)

        # Initialize last conv to near-zero so refiner starts as identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        tag = f"+{extra_channels}ch guide" if extra_channels > 0 else ""
        print(f"[ScreenSpaceRefiner] {latent_dim}d, {num_blocks} blocks, "
              f"hidden={hidden_dim}, {n_params:.2f}M params {tag}")

    def forward(self, x: torch.Tensor, guide: torch.Tensor | None = None) -> torch.Tensor:
        """Refine rendered features with residual correction.

        Args:
            x: Rendered latent feature map [B, D, H, W].
            guide: Optional guide signal [B, extra_channels, H, W] (e.g. RGB).

        Returns:
            Refined feature map [B, D, H, W].
        """
        if guide is not None:
            inp = torch.cat([x, guide], dim=1)
        else:
            inp = x
        delta = self.net(inp)
        return x + delta


class ScreenSpaceRefinerLight(nn.Module):
    """Ultra-lightweight refiner with just 3 conv layers (~50K params).

    For when the full refiner overfits. Uses depthwise separable convolutions.
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(latent_dim, latent_dim, 3, padding=1, groups=latent_dim, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            # Pointwise conv
            nn.Conv2d(latent_dim, latent_dim * 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim * 2),
            nn.ReLU(inplace=True),
            # Back to latent dim
            nn.Conv2d(latent_dim * 2, latent_dim, 1, bias=True),
        )
        # Zero-init last layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        n_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[ScreenSpaceRefinerLight] {latent_dim}d, {n_params:.3f}M params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
