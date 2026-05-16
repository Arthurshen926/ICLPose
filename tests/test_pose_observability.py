import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from feature_extract.pose_observability import feature_pose_fisher_stats


def test_feature_pose_fisher_stats_prefers_spatially_varying_features():
    h, w = 8, 10
    depth = torch.ones(1, 1, h, w)
    intrinsics = torch.tensor([[20.0, 20.0, (w - 1) / 2.0, (h - 1) / 2.0]])
    constant = torch.ones(1, 4, h, w)
    xs = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(1, 2, h, w)
    ys = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(1, 2, h, w)
    varying = torch.cat([xs, ys], dim=1)

    const_stats = feature_pose_fisher_stats(constant, depth, intrinsics)
    varying_stats = feature_pose_fisher_stats(varying, depth, intrinsics)

    assert varying_stats["logdet"].item() > const_stats["logdet"].item()
    assert varying_stats["trace"].item() > const_stats["trace"].item()
    assert varying_stats["valid_frac"].item() == 1.0


def test_feature_pose_fisher_stats_handles_empty_mask():
    feature = torch.randn(1, 3, 4, 5)
    depth = torch.ones(1, 1, 4, 5)
    intrinsics = torch.tensor([[10.0, 10.0, 2.0, 2.0]])
    mask = torch.zeros(1, 1, 4, 5)

    stats = feature_pose_fisher_stats(feature, depth, intrinsics, mask=mask)

    assert torch.isfinite(stats["logdet"])
    assert stats["valid_frac"].item() == 0.0


def test_feature_pose_fisher_stats_can_normalize_feature_scale():
    h, w = 8, 10
    depth = torch.ones(1, 1, h, w)
    intrinsics = torch.tensor([[20.0, 20.0, (w - 1) / 2.0, (h - 1) / 2.0]])
    xs = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(1, 2, h, w)
    ys = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(1, 2, h, w)
    feature = torch.cat([xs, ys], dim=1)

    raw = feature_pose_fisher_stats(feature, depth, intrinsics)
    scaled_raw = feature_pose_fisher_stats(feature * 10.0, depth, intrinsics)
    unit = feature_pose_fisher_stats(feature, depth, intrinsics, normalize_channels=True)
    scaled_unit = feature_pose_fisher_stats(feature * 10.0, depth, intrinsics, normalize_channels=True)

    assert scaled_raw["trace"].item() > raw["trace"].item() * 50.0
    assert torch.allclose(unit["logdet"], scaled_unit["logdet"], atol=1e-3)
