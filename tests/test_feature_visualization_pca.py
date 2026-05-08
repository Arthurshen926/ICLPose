import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import tempfile
from feature_field.visualize_feature_comparison import (
    CachedQueryFeatureProvider,
    target_basis_pca_colorize,
)


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


def test_cached_query_feature_provider_loads_asymmetric_features_by_fid():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fine_dir = root / "fine_geo"
        coarse_dir = root / "coarse_sem"
        fine_dir.mkdir()
        coarse_dir.mkdir()
        torch.save(torch.randn(96, 8, 10), fine_dir / "rgb_7_fine_geo_96x8x10.pt")
        torch.save(torch.randn(32, 4, 5), coarse_dir / "rgb_7_coarse_sem_32x4x5.pt")

        provider = CachedQueryFeatureProvider(root, device="cpu", expected_fine_dim=96, expected_coarse_dim=32)
        fine, coarse = provider.get_by_fid(7)

        assert fine.shape == (1, 96, 8, 10)
        assert coarse.shape == (1, 32, 4, 5)

if __name__ == "__main__":
    test_target_basis_pca_colorize_uses_same_colors_for_matching_features()
    test_target_basis_pca_colorize_preserves_spatial_variation_without_position_mean()
    test_cached_query_feature_provider_loads_asymmetric_features_by_fid()
    print("feature visualization PCA tests passed")
