import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import tempfile
from types import SimpleNamespace
from PIL import Image
import numpy as np

from feature_extract.students.radio_query_student import RadioQueryStudent
from feature_field.visualize_feature_comparison import (
    CachedQueryFeatureProvider,
    OnlineQueryStudentProvider,
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


def test_online_query_student_provider_restores_fine_loc_and_uses_export_key():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image_path = root / "query.png"
        cfg_path = root / "query.yaml"
        ckpt_path = root / "query.pth"

        arr = np.zeros((64, 80, 3), dtype=np.uint8)
        arr[:, :, 0] = np.arange(80, dtype=np.uint8)[None, :]
        arr[:, :, 1] = np.arange(64, dtype=np.uint8)[:, None]
        Image.fromarray(arr).save(image_path)

        cfg_path.write_text(
            """
dataset:
  input_hw: [64, 80]
  feature_hw: [16, 20]
  coarse_feature_hw: [8, 10]
model:
  feature_dim: 16
  fine_feature_dim: 16
  coarse_feature_dim: 8
  base_channels: 8
  stage_dims: [8, 12, 16, 20]
  l2_normalize: true
  fine_loc_head: true
  fine_loc_zero_init: false
  fine_loc_highres_source: stage2
  fine_loc_highres_zero_init: false
  export_fine_key: fine_loc
export:
  fine_key: fine_loc
""",
            encoding="utf-8",
        )
        model = RadioQueryStudent(
            feature_dim=16,
            fine_feature_dim=16,
            coarse_feature_dim=8,
            base_channels=8,
            stage_dims=(8, 12, 16, 20),
            output_hw=(16, 20),
            coarse_output_hw=(8, 10),
            input_hw=(64, 80),
            fine_loc_head=True,
            fine_loc_zero_init=False,
            fine_loc_highres_source="stage2",
            fine_loc_highres_zero_init=False,
        )
        model.eval()
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

        provider = OnlineQueryStudentProvider(
            cfg_path,
            ckpt_path,
            device="cpu",
            expected_fine_dim=16,
            expected_coarse_dim=8,
        )
        fine, coarse = provider.get_for_camera(SimpleNamespace(image=str(image_path)))

        img = torch.from_numpy(arr.astype("float32") / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            direct = model(img)

        assert provider.fine_key == "fine_loc"
        assert fine.shape == (1, 16, 16, 20)
        assert coarse.shape == (1, 8, 8, 10)
        assert torch.allclose(fine, direct["fine_loc"], atol=1e-6)
        assert not torch.allclose(fine, direct["fine"], atol=1e-5)


if __name__ == "__main__":
    test_target_basis_pca_colorize_uses_same_colors_for_matching_features()
    test_target_basis_pca_colorize_preserves_spatial_variation_without_position_mean()
    test_cached_query_feature_provider_loads_asymmetric_features_by_fid()
    test_online_query_student_provider_restores_fine_loc_and_uses_export_key()
    print("feature visualization PCA tests passed")
