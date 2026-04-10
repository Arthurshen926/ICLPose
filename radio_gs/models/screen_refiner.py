"""Screen-space feature refiner for correcting alpha-blending artifacts.

Alpha-blending in 3DGS produces over-smoothed features because each pixel is
a weighted average of overlapping Gaussians. This module learns to "undo" the
averaging in screen space using a lightweight residual CNN.

Architecture:
    rendered_64d → [Conv-BN-ReLU] × N blocks (with residual) → refined_64d
    Each block: Conv3×3 → BN → ReLU → Conv3×3 → BN → residual add → ReLU
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_refiner_extra_channels(
    *,
    rgb_guide: bool = False,
    depth_guide: bool = False,
    depth_grad: bool = False,
    alpha_guide: bool = False,
    boundary_guide: bool = False,
) -> int:
    """Return the number of auxiliary guide channels expected by the refiner."""
    extra_ch = 3 if rgb_guide else 0
    if depth_guide:
        extra_ch += 3 if depth_grad else 1
    if alpha_guide:
        extra_ch += 1
    if boundary_guide:
        extra_ch += 1
    return extra_ch


def _ensure_map_4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    return x.float()


def _normalize_per_image(x: torch.Tensor) -> torch.Tensor:
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    return (x - lo) / (hi - lo + 1e-6)


def _gradient_channels(x: torch.Tensor, grad_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0)) * grad_scale
    dy = F.pad(dy, (0, 0, 0, 1)) * grad_scale
    return dx, dy


def build_depth_guide(
    depth_map: torch.Tensor,
    *,
    depth_grad: bool = False,
    grad_scale: float = 10.0,
) -> torch.Tensor:
    """Build a normalized depth guide for the screen refiner."""
    depth = _normalize_per_image(_ensure_map_4d(depth_map))
    if depth_grad:
        dx, dy = _gradient_channels(depth, grad_scale=grad_scale)
        return torch.cat([depth, dx, dy], dim=1)
    return depth


def build_boundary_guide(
    depth_map: torch.Tensor,
    alpha_map: Optional[torch.Tensor] = None,
    *,
    grad_scale: float = 10.0,
) -> torch.Tensor:
    """Build a single-channel geometry boundary cue from depth/alpha edges."""
    depth = _normalize_per_image(_ensure_map_4d(depth_map))
    dx, dy = _gradient_channels(depth, grad_scale=1.0)
    boundary = torch.sqrt(dx.square() + dy.square() + 1e-8)
    if alpha_map is not None:
        alpha = _ensure_map_4d(alpha_map).clamp(0.0, 1.0)
        adx, ady = _gradient_channels(alpha, grad_scale=1.0)
        alpha_boundary = torch.sqrt(adx.square() + ady.square() + 1e-8)
        boundary = torch.maximum(boundary, alpha_boundary)
    boundary = _normalize_per_image(boundary)
    return boundary * grad_scale


def build_refiner_guide(
    render_result: dict,
    *,
    rgb_guide: Optional[torch.Tensor] = None,
    use_depth_guide: bool = False,
    use_depth_grad: bool = False,
    depth_grad_scale: float = 10.0,
    use_alpha_guide: bool = False,
    use_boundary_guide: bool = False,
) -> Optional[torch.Tensor]:
    """Assemble the optional guide tensor used by the screen-space refiner."""
    parts: list[torch.Tensor] = []
    if rgb_guide is not None:
        parts.append(rgb_guide)
    if use_depth_guide:
        parts.append(
            build_depth_guide(
                render_result["depth_map"],
                depth_grad=use_depth_grad,
                grad_scale=depth_grad_scale,
            )
        )
    alpha_map = render_result.get("alpha_map")
    if use_alpha_guide and alpha_map is not None:
        parts.append(_ensure_map_4d(alpha_map).clamp(0.0, 1.0))
    if use_boundary_guide:
        parts.append(
            build_boundary_guide(
                render_result["depth_map"],
                alpha_map=alpha_map,
                grad_scale=depth_grad_scale,
            )
        )
    if not parts:
        return None
    return torch.cat(parts, dim=1)


def _make_norm(channels: int, norm_type: str = "gn") -> nn.Module:
    """Create normalization layer. GroupNorm is more stable with mixed precision."""
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    num_groups = min(32, channels)
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels)


class ResidualBlock(nn.Module):
    """Simple residual block with two 3×3 convolutions."""

    def __init__(self, channels: int, expansion: int = 1, norm_type: str = "gn"):
        super().__init__()
        mid = channels * expansion
        self.conv1 = nn.Conv2d(channels, mid, 3, padding=1, bias=False)
        self.bn1 = _make_norm(mid, norm_type)
        self.conv2 = nn.Conv2d(mid, channels, 3, padding=1, bias=False)
        self.bn2 = _make_norm(channels, norm_type)
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
        norm_type: str = "gn",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.extra_channels = extra_channels
        in_channels = latent_dim + extra_channels

        # Project to hidden dim
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            _make_norm(hidden_dim, norm_type),
            nn.ReLU(inplace=True),
        ]

        # Residual blocks
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, norm_type=norm_type))

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
        elif self.extra_channels > 0:
            # No guide provided but network expects extra channels — zero-pad
            pad = torch.zeros(
                x.shape[0], self.extra_channels, x.shape[2], x.shape[3],
                device=x.device, dtype=x.dtype,
            )
            inp = torch.cat([x, pad], dim=1)
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
            _make_norm(latent_dim),
            nn.ReLU(inplace=True),
            # Pointwise conv
            nn.Conv2d(latent_dim, latent_dim * 2, 1, bias=False),
            _make_norm(latent_dim * 2),
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
