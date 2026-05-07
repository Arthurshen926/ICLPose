from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries
from feature_retrieval.pairwise_pnp_init_export import (
    build_pairwise_pnp_entries,
    map_ref_keypoints_to_world_points,
)


def test_map_ref_keypoints_to_world_points_uses_nearest_valid_colmap_observation():
    query_kpts = np.array(
        [
            [100.0, 50.0],
            [200.0, 60.0],
            [300.0, 70.0],
            [400.0, 80.0],
        ],
        dtype=np.float32,
    )
    ref_kpts = np.array(
        [
            [10.2, 10.1],   # valid point id 11
            [20.0, 20.0],   # nearest observation has invalid point id
            [55.0, 55.0],   # too far from any observation
            [30.3, 30.2],   # valid point id 33
        ],
        dtype=np.float32,
    )
    colmap_xys = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]], dtype=np.float32)
    colmap_point_ids = np.array([11, -1, 33], dtype=np.int64)
    point_xyz_by_id = {
        11: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        33: np.array([0.0, 3.0, 0.0], dtype=np.float32),
    }

    pts3d, pts2d, distances = map_ref_keypoints_to_world_points(
        query_kpts,
        ref_kpts,
        colmap_xys,
        colmap_point_ids,
        point_xyz_by_id,
        max_distance_px=1.0,
    )

    assert pts3d.shape == (2, 3)
    assert pts2d.tolist() == [[100.0, 50.0], [400.0, 80.0]]
    assert np.allclose(pts3d, np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32))
    assert np.all(distances < 0.5)


def test_build_pairwise_pnp_entries_orders_successful_candidates_and_writes_schema(tmp_path):
    query_samples = [
        {"img_id": 9, "image_name": "seq/query.png", "image_stem": "seq_query", "pose_w2c": np.eye(4, dtype=np.float32)}
    ]
    pose_low = np.eye(4, dtype=np.float32)
    pose_low[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    pose_high = np.eye(4, dtype=np.float32)
    pose_high[:3, 3] = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    fallback_pose = np.eye(4, dtype=np.float32) * 2.0
    candidates = [
        [
            {
                "pose_w2c": pose_low,
                "retrieval_frame_id": 1,
                "retrieval_image_name": "seq/train_low.png",
                "retrieval_score": 0.4,
                "pnp_success": True,
                "num_inliers": 12,
            },
            {
                "pose_w2c": fallback_pose,
                "retrieval_frame_id": 2,
                "retrieval_image_name": "seq/train_fail.png",
                "retrieval_score": 0.9,
                "pnp_success": False,
                "num_inliers": 0,
            },
            {
                "pose_w2c": pose_high,
                "retrieval_frame_id": 3,
                "retrieval_image_name": "seq/train_high.png",
                "retrieval_score": 0.3,
                "pnp_success": True,
                "num_inliers": 25,
            },
        ]
    ]

    entries, stats = build_pairwise_pnp_entries(
        query_samples=query_samples,
        candidate_results_by_query=candidates,
        source_name="pairwise_sift_pnp_test",
        save_path=str(tmp_path / "pairwise_init.npz"),
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(tmp_path / "pairwise_init.npz"))

    assert entries[0]["retrieval_frame_id"] == 3
    assert entries[0]["retrieval_image_name"] == "seq/train_high.png"
    assert np.allclose(entries[0]["pose_init"], pose_high)
    assert entries[0]["candidate_valid_mask"].tolist() == [True, True, True]
    assert stats["num_pnp_success_candidates"] == 2
    assert stats["num_queries_with_pnp_success"] == 1
    assert loaded_entries[0]["pose_init_candidates"].shape == (3, 4, 4)
    assert loaded_entries[0]["retrieval_frame_ids_candidates"].tolist() == [3, 1, 2]
    assert loaded_stats["method_used"] == "pairwise_sift_pnp_test"


def test_build_pairwise_pnp_entries_uses_selection_score_for_candidate_scores():
    query_samples = [
        {"img_id": 9, "image_name": "seq/query.png", "image_stem": "seq_query", "pose_w2c": np.eye(4, dtype=np.float32)}
    ]
    pose_low = np.eye(4, dtype=np.float32)
    pose_high = np.eye(4, dtype=np.float32)
    candidates = [
        [
            {
                "pose_w2c": pose_low,
                "retrieval_frame_id": 1,
                "retrieval_image_name": "seq/train_low.png",
                "retrieval_score": 0.4,
                "pnp_success": True,
                "num_inliers": 12,
                "selection_score": 3.5,
            },
            {
                "pose_w2c": pose_high,
                "retrieval_frame_id": 3,
                "retrieval_image_name": "seq/train_high.png",
                "retrieval_score": 0.3,
                "pnp_success": True,
                "num_inliers": 25,
                "selection_score": 1.25,
            },
        ]
    ]

    entries, _ = build_pairwise_pnp_entries(
        query_samples=query_samples,
        candidate_results_by_query=candidates,
        source_name="pairwise_sift_pnp_test",
    )

    assert entries[0]["retrieval_frame_ids_candidates"].tolist() == [3, 1]
    assert entries[0]["retrieval_scores_candidates"].tolist() == [1.25, 3.5]


def test_build_pairwise_pnp_entries_preserves_candidate_quality_fields(tmp_path):
    query_samples = [
        {"img_id": 9, "image_name": "seq/query.png", "image_stem": "seq_query", "pose_w2c": np.eye(4, dtype=np.float32)}
    ]
    candidates = [
        [
            {
                "pose_w2c": np.eye(4, dtype=np.float32),
                "retrieval_frame_id": 1,
                "retrieval_image_name": "seq/train_a.png",
                "retrieval_score": 0.4,
                "pnp_success": True,
                "num_inliers": 12,
                "num_matches": 20,
                "pnp_reproj_rmse": 2.5,
                "pnp_reproj_median": 1.5,
                "pnp_inlier_ratio": 0.6,
                "pnp_inlier_conf_mean": 0.8,
                "selection_score": 2.0,
            },
            {
                "pose_w2c": np.eye(4, dtype=np.float32),
                "retrieval_frame_id": 3,
                "retrieval_image_name": "seq/train_b.png",
                "retrieval_score": 0.3,
                "pnp_success": True,
                "num_inliers": 25,
                "num_matches": 50,
                "pnp_reproj_rmse": 4.5,
                "pnp_reproj_median": 3.0,
                "pnp_inlier_ratio": 0.5,
                "pnp_inlier_conf_mean": 0.7,
                "selection_score": 1.0,
            },
        ]
    ]

    entries, _ = build_pairwise_pnp_entries(
        query_samples=query_samples,
        candidate_results_by_query=candidates,
        source_name="pairwise_sift_pnp_test",
        save_path=str(tmp_path / "pairwise_init.npz"),
    )
    loaded_entries, _ = load_retrieval_init_entries(str(tmp_path / "pairwise_init.npz"))

    assert np.allclose(entries[0]["retrieval_original_scores_candidates"], [0.3, 0.4])
    assert entries[0]["retrieval_pnp_num_inliers_candidates"].tolist() == [25.0, 12.0]
    assert entries[0]["retrieval_pnp_num_matches_candidates"].tolist() == [50.0, 20.0]
    assert np.allclose(entries[0]["retrieval_pnp_reproj_rmse_candidates"], [4.5, 2.5])
    assert np.allclose(entries[0]["retrieval_pnp_inlier_ratio_candidates"], [0.5, 0.6])
    assert np.allclose(loaded_entries[0]["retrieval_pnp_inlier_conf_mean_candidates"], [0.7, 0.8])


def test_build_pairwise_pnp_entries_keeps_retrieval_fallback_when_all_pnp_fail():
    query_samples = [
        {"img_id": 10, "image_name": "seq/query2.png", "image_stem": "seq_query2", "pose_w2c": np.eye(4, dtype=np.float32)}
    ]
    fallback_pose = np.eye(4, dtype=np.float32)
    fallback_pose[:3, 3] = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    candidates = [
        [
            {
                "pose_w2c": fallback_pose,
                "retrieval_frame_id": 7,
                "retrieval_image_name": "seq/train7.png",
                "retrieval_score": 0.8,
                "pnp_success": False,
                "num_inliers": 3,
            }
        ]
    ]

    entries, stats = build_pairwise_pnp_entries(
        query_samples=query_samples,
        candidate_results_by_query=candidates,
        source_name="pairwise_sift_pnp_test",
    )

    assert entries[0]["init_source"] == "pairwise_sift_pnp_test_retrieval_fallback"
    assert np.allclose(entries[0]["pose_init"], fallback_pose)
    assert entries[0]["candidate_valid_mask"].tolist() == [True]
    assert stats["num_queries_with_pnp_success"] == 0
