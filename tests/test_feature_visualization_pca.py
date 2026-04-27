import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from feature_field.visualize_feature_comparison import target_basis_pca_colorize


def test_target_basis_pca_colorize_uses_same_colors_for_matching_features():
    target = torch.randn(5, 4, 6)
    pred_vis, target_vis = target_basis_pca_colorize([target.clone()], [target.clone()])

    assert torch.allclose(pred_vis[0], target_vis[0], atol=1e-6)


def test_target_basis_pca_colorize_preserves_spatial_variation_without_position_mean():
    target = torch.zeros(5, 4, 6)
    target[0, :, 3:] = 5.0
    pred = target.clone()

    pred_vis, _ = target_basis_pca_colorize([pred], [target])
    left = pred_vis[0][:, :, :3].mean()
    right = pred_vis[0][:, :, 3:].mean()

    assert (left - right).abs().item() > 0.1


if __name__ == "__main__":
    test_target_basis_pca_colorize_uses_same_colors_for_matching_features()
    test_target_basis_pca_colorize_preserves_spatial_variation_without_position_mean()
    print("feature visualization PCA tests passed")
