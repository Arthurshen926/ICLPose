from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries
from feature_extract.scene_coord_init_export import (
    build_scene_coord_init_entries,
    scene_coord_pnp_from_prediction,
    unnormalize_scene_coord,
)


def _synthetic_scene_coord_prediction(height=12, width=16):
    fx, fy = 80.0, 82.0
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    z = 3.0 + 0.03 * xx.float() + 0.05 * yy.float()
    x = (xx.float() - cx) * z / fx
    y = (yy.float() - cy) * z / fy
    world = torch.stack([x, y, z], dim=0)
    intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return world, intr


def test_scene_coord_pnp_from_prediction_recovers_identity_pose():
    world, intr = _synthetic_scene_coord_prediction()
    center = torch.tensor([1.0, -2.0, 0.5])
    scale = 8.0
    pred = (world - center.view(3, 1, 1)) / scale

    pose, info = scene_coord_pnp_from_prediction(
        pred,
        center=center,
        scale=scale,
        intrinsics=intr,
        reproj_threshold=1.0,
        n_iters=2000,
        min_inliers=20,
        max_points=200,
    )

    assert pose is not None
    assert info["success"] is True
    assert info["num_inliers"] >= 20
    assert np.allclose(pose, np.eye(4), atol=1e-3)


def test_unnormalize_scene_coord_accepts_batched_tensor():
    world, _intr = _synthetic_scene_coord_prediction(height=4, width=5)
    center = torch.tensor([0.2, 0.3, 0.4])
    scale = 2.5
    pred = ((world - center.view(3, 1, 1)) / scale).unsqueeze(0)

    restored = unnormalize_scene_coord(pred, center=center, scale=scale)

    assert restored.shape == (1, 3, 4, 5)
    assert torch.allclose(restored[0], world, atol=1e-6)


def test_build_scene_coord_init_entries_use_retrieval_init_schema(tmp_path):
    sample = {
        "img_id": 11,
        "image_name": "seq1/frame00011.png",
        "image_stem": "seq1_frame00011",
        "pose_w2c": np.eye(4, dtype=np.float32),
    }
    pose = np.eye(4, dtype=np.float32)

    entries, stats = build_scene_coord_init_entries(
        query_samples=[sample],
        pose_predictions=[pose],
        scores=[37.0],
        inlier_counts=[37],
        source_name="scene_coord_pnp_test",
        save_path=str(tmp_path / "scene_coord_init.npz"),
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(tmp_path / "scene_coord_init.npz"))

    assert stats["method_used"] == "scene_coord_pnp_test"
    assert stats["num_success"] == 1
    assert entries[0]["retrieval_score"] == 37.0
    assert loaded_entries[0]["query_img_id"] == 11
    assert loaded_entries[0]["pose_init_candidates"].shape == (1, 4, 4)
    assert loaded_entries[0]["candidate_valid_mask"].tolist() == [True]
    assert loaded_stats["counts_by_source"]["scene_coord_pnp_test"] == 1
