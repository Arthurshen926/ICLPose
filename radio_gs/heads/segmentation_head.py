"""Semantic segmentation head for RADIO-GS decoded features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional


class SegmentationHead(nn.Module):
    """Predicts semantic segmentation logits from 1280-d RADIO features.

    Args:
        feature_dim: Input feature dimension (default 1280).
        num_classes: Number of semantic classes.
        hidden_dim: Hidden layer width for the MLP variant.
        num_layers: Number of hidden layers for the MLP variant.
        head_type: One of 'linear', 'mlp'.
    """

    def __init__(
        self,
        feature_dim: int = 1280,
        num_classes: int = 40,
        hidden_dim: int = 256,
        num_layers: int = 2,
        head_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head_type = head_type

        if head_type == "linear":
            self.head = nn.Conv2d(feature_dim, num_classes, 1)
        elif head_type == "mlp":
            self.head = self._build_mlp(feature_dim, hidden_dim, num_layers, num_classes)
        else:
            raise ValueError(f"Unknown head_type '{head_type}'")

    @staticmethod
    def _build_mlp(
        in_dim: int, hidden_dim: int, num_layers: int, out_dim: int,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for i in range(num_layers):
            is_last = i == num_layers - 1
            cur_out = out_dim if is_last else hidden_dim
            layers.append(nn.Conv2d(in_dim, cur_out, 1))
            if not is_last:
                layers.append(nn.GroupNorm(min(32, cur_out), cur_out))
                layers.append(nn.GELU())
            in_dim = cur_out
        return nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        """Predict per-pixel class logits.

        Args:
            features: [B, 1280, H, W] decoded RADIO features.

        Returns:
            [B, num_classes, H, W] pre-softmax logits.
        """
        return self.head(features)


class SegmentationLoss(nn.Module):
    """Segmentation loss supporting cross-entropy and focal variants.

    Args:
        loss_type: One of 'ce' (cross-entropy), 'focal'.
        ignore_index: Label index to ignore in loss computation.
        class_weights: Optional per-class weight tensor of shape [C].
        label_smoothing: Label smoothing factor for cross-entropy.
    """

    def __init__(
        self,
        loss_type: str = "ce",
        ignore_index: int = 255,
        class_weights: Optional[Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )

    def forward(
        self, pred_logits: Tensor, gt_labels: Tensor, mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute segmentation loss.

        Args:
            pred_logits: [B, C, H, W] predicted class logits.
            gt_labels: [B, H, W] ground-truth class indices (long).
            mask: Optional [B, H, W] boolean mask for valid pixels.

        Returns:
            Scalar loss tensor.
        """
        if mask is not None:
            gt_labels = gt_labels.clone()
            gt_labels[~mask.bool()] = self.ignore_index

        if self.loss_type == "ce":
            return F.cross_entropy(
                pred_logits,
                gt_labels,
                weight=self.class_weights,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            )
        elif self.loss_type == "focal":
            return self._focal_loss(pred_logits, gt_labels)
        raise ValueError(f"Unknown loss_type '{self.loss_type}'")

    def _focal_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> Tensor:
        ce = F.cross_entropy(
            logits, targets, weight=self.class_weights,
            ignore_index=self.ignore_index, reduction="none",
        )
        pt = torch.exp(-ce)
        loss = alpha * (1.0 - pt) ** gamma * ce
        return loss.mean()


def compute_miou(
    pred: Tensor, gt: Tensor, num_classes: int, ignore_index: int = 255,
) -> float:
    """Compute mean Intersection-over-Union.

    Args:
        pred: [B, H, W] predicted class indices.
        gt: [B, H, W] ground-truth class indices.
        num_classes: Total number of classes.
        ignore_index: Label index to exclude.

    Returns:
        Mean IoU as a Python float.
    """
    valid = gt != ignore_index
    pred_v = pred[valid]
    gt_v = gt[valid]

    ious: List[float] = []
    for c in range(num_classes):
        pred_c = pred_v == c
        gt_c = gt_v == c
        intersection = (pred_c & gt_c).sum().item()
        union = (pred_c | gt_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / max(len(ious), 1)


def compute_pixel_accuracy(
    pred: Tensor, gt: Tensor, ignore_index: int = 255,
) -> float:
    """Compute pixel-wise classification accuracy.

    Args:
        pred: [B, H, W] predicted class indices.
        gt: [B, H, W] ground-truth class indices.
        ignore_index: Label index to exclude.

    Returns:
        Accuracy as a Python float.
    """
    valid = gt != ignore_index
    if valid.sum() == 0:
        return 0.0
    return (pred[valid] == gt[valid]).float().mean().item()
