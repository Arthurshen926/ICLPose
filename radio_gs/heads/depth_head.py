"""Depth prediction head for RADIO-GS decoded features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional


class DepthHead(nn.Module):
    """Predicts metric depth from 1280-d RADIO features.

    Args:
        feature_dim: Input feature dimension (default 1280).
        hidden_dim: Hidden layer width for MLP / DPT variants.
        num_layers: Number of hidden layers for the MLP variant.
        head_type: One of 'linear', 'mlp', 'dpt'.
        output_activation: 'softplus' or 'sigmoid'.
        min_depth: Minimum predicted depth.
        max_depth: Maximum predicted depth.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        hidden_dim: int = 256,
        num_layers: int = 3,
        head_type: str = "mlp",
        output_activation: str = "softplus",
        min_depth: float = 0.01,
        max_depth: float = 10.0,
    ) -> None:
        super().__init__()
        self.head_type = head_type
        self.output_activation = output_activation
        self.min_depth = min_depth
        self.max_depth = max_depth

        if head_type == "linear":
            self.head = nn.Conv2d(feature_dim, 1, 1)
        elif head_type == "mlp":
            self.head = self._build_mlp(feature_dim, hidden_dim, num_layers)
        elif head_type == "dpt":
            self.head = _LightweightDPT(feature_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown head_type '{head_type}'")

    def _build_mlp(self, in_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Conv2d(in_dim, out_dim, 1))
            if i < num_layers - 1:
                layers.append(nn.GroupNorm(min(32, out_dim), out_dim))
                layers.append(nn.GELU())
            in_dim = out_dim
        return nn.Sequential(*layers)

    def _apply_activation(self, x: Tensor) -> Tensor:
        if self.output_activation == "softplus":
            return F.softplus(x) + self.min_depth
        elif self.output_activation == "sigmoid":
            return torch.sigmoid(x) * (self.max_depth - self.min_depth) + self.min_depth
        raise ValueError(f"Unknown output_activation '{self.output_activation}'")

    def forward(self, features: Tensor) -> Tensor:
        """Predict depth from RADIO features.

        Args:
            features: [B, 1280, H, W] decoded RADIO features.

        Returns:
            [B, 1, H, W] predicted metric depth.
        """
        raw = self.head(features)
        return self._apply_activation(raw)


class _LightweightDPT(nn.Module):
    """Simplified DPT-style decoder with multi-scale processing and skip connections."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.refine1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.out_conv = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        skip = self.reduce(x)
        x_up = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
        x_up = self.refine1(x_up)
        x_up = F.interpolate(x_up, scale_factor=2, mode="bilinear", align_corners=False)
        skip_up = F.interpolate(skip, size=x_up.shape[2:], mode="bilinear", align_corners=False)
        x_up = self.refine2(x_up + skip_up)
        x_up = self.out_conv(x_up)
        return F.interpolate(x_up, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)


class DepthLoss(nn.Module):
    """Depth estimation loss with masking for invalid pixels.

    Args:
        loss_type: One of 'l1', 'scale_invariant', 'berhu'.
        weight: Scalar multiplier applied to the loss.
    """

    def __init__(self, loss_type: str = "scale_invariant", weight: float = 1.0) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.weight = weight

    def forward(self, pred: Tensor, gt: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Compute depth loss.

        Args:
            pred: [B, 1, H, W] predicted depth.
            gt: [B, 1, H, W] ground-truth depth.
            mask: Optional [B, 1, H, W] boolean mask. Defaults to gt > 0.

        Returns:
            Scalar loss tensor.
        """
        if mask is None:
            mask = gt > 0
        mask = mask.bool()

        if mask.sum() == 0:
            return pred.sum() * 0.0

        pred_m = pred[mask]
        gt_m = gt[mask]

        if self.loss_type == "l1":
            loss = F.l1_loss(pred_m, gt_m)
        elif self.loss_type == "scale_invariant":
            loss = self._scale_invariant(pred_m, gt_m)
        elif self.loss_type == "berhu":
            loss = self._berhu(pred_m, gt_m)
        else:
            raise ValueError(f"Unknown loss_type '{self.loss_type}'")

        return loss * self.weight

    @staticmethod
    def _scale_invariant(pred: Tensor, gt: Tensor, lam: float = 0.5) -> Tensor:
        d = torch.log(pred.clamp(min=1e-8)) - torch.log(gt.clamp(min=1e-8))
        return torch.mean(d ** 2) - lam * (torch.mean(d) ** 2)

    @staticmethod
    def _berhu(pred: Tensor, gt: Tensor) -> Tensor:
        diff = (pred - gt).abs()
        c = 0.2 * diff.max().detach()
        l1_mask = diff <= c
        loss = torch.where(l1_mask, diff, (diff ** 2 + c ** 2) / (2 * c + 1e-8))
        return loss.mean()
