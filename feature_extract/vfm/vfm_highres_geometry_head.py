"""High-resolution depth and normal decoders on frozen RADIO/VFM tokens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GeometryHeadOutput:
    depth: torch.Tensor
    normal: torch.Tensor
    confidence: torch.Tensor


class RadioHighResGeometryHead(nn.Module):
    """Decode low-resolution RADIO tokens into high-resolution geometry maps."""

    def __init__(
        self,
        in_channels: int = 1280,
        hidden_channels: int = 128,
        min_depth: float = 1e-3,
        architecture: str = "shared",
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        if int(hidden_channels) <= 0:
            raise ValueError("hidden_channels must be positive")
        if float(min_depth) <= 0.0:
            raise ValueError("min_depth must be positive")
        if str(architecture) not in {"shared", "separate_decoders"}:
            raise ValueError("architecture must be 'shared' or 'separate_decoders'")
        self.min_depth = float(min_depth)
        self.architecture = str(architecture)
        mid = max(int(hidden_channels) // 2, 32)
        self.mid_channels = int(mid)
        self.stem = nn.Sequential(
            nn.GroupNorm(1, int(in_channels)),
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine = self._make_decoder(int(hidden_channels), int(mid))
        if self.architecture == "separate_decoders":
            self.depth_refine = self._make_decoder(int(hidden_channels), int(mid))
            self.normal_refine = self._make_decoder(int(hidden_channels), int(mid))
        else:
            self.depth_refine = self.refine
            self.normal_refine = self.refine
        self.depth_head = nn.Conv2d(mid, 1, kernel_size=3, padding=1)
        self.normal_head = nn.Conv2d(mid, 3, kernel_size=3, padding=1)
        self.confidence_head = nn.Conv2d(mid, 1, kernel_size=3, padding=1)

    @staticmethod
    def _make_decoder(hidden_channels: int, mid_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(mid_channels), kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor, output_size: tuple[int, int]) -> GeometryHeadOutput:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape (B, C, H, W)")
        out_h, out_w = int(output_size[0]), int(output_size[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError("output_size must be positive")
        x = self.stem(tokens)
        x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
        depth_features = self.depth_refine(x)
        normal_features = self.normal_refine(x)
        log_depth = self.depth_head(depth_features).squeeze(1)
        depth = torch.exp(torch.clamp(log_depth, min=float(np.log(self.min_depth)), max=float(np.log(1e4))))
        normal = F.normalize(self.normal_head(normal_features), dim=1, eps=1e-6)
        confidence = torch.sigmoid(self.confidence_head(depth_features)).squeeze(1)
        return GeometryHeadOutput(depth=depth, normal=normal, confidence=confidence)


def masked_log_depth_l1(pred_depth: torch.Tensor, target_depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool()
    if pred_depth.shape != target_depth.shape or pred_depth.shape != valid.shape:
        raise ValueError("pred_depth, target_depth, and valid_mask must have matching shapes")
    if not bool(torch.any(valid)):
        return pred_depth.sum() * 0.0
    pred = torch.clamp(pred_depth[valid], min=1e-6)
    target = torch.clamp(target_depth[valid], min=1e-6)
    return torch.mean(torch.abs(torch.log(pred) - torch.log(target)))


def masked_normal_cosine_loss(pred_normal: torch.Tensor, target_normal: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.bool()
    if pred_normal.shape != target_normal.shape:
        raise ValueError("pred_normal and target_normal must have matching shapes")
    if pred_normal.ndim != 4 or pred_normal.shape[1] != 3:
        raise ValueError("normal tensors must have shape (B, 3, H, W)")
    if valid.shape != pred_normal.shape[0:1] + pred_normal.shape[2:4]:
        raise ValueError("valid_mask must have shape (B, H, W)")
    if not bool(torch.any(valid)):
        return pred_normal.sum() * 0.0
    pred = F.normalize(pred_normal.permute(0, 2, 3, 1)[valid], dim=-1, eps=1e-6)
    target = F.normalize(target_normal.permute(0, 2, 3, 1)[valid], dim=-1, eps=1e-6)
    return torch.mean(1.0 - torch.sum(pred * target, dim=-1))


def masked_geometry_loss(
    pred_depth: torch.Tensor,
    pred_normal: torch.Tensor,
    target_depth: torch.Tensor,
    target_normal: torch.Tensor,
    valid_mask: torch.Tensor,
    normal_weight: float = 0.25,
) -> torch.Tensor:
    return masked_log_depth_l1(pred_depth, target_depth, valid_mask) + float(normal_weight) * masked_normal_cosine_loss(
        pred_normal,
        target_normal,
        valid_mask,
    )
