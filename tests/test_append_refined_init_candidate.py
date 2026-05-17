from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
try:
    import faiss  # noqa: F401
except ImportError:
    sys.modules.setdefault("faiss", types.ModuleType("faiss"))

from data.radio_loc_retrieval_dataset import load_retrieval_init_entries, save_retrieval_init_entries
from feature_retrieval.tools.append_refined_init_candidate import append_refined_init_candidate


def _pose_at(value: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = float(value)
    return pose


def _base_entry(idx: int) -> dict:
    candidates = np.stack([_pose_at(1.0), _pose_at(2.0)], axis=0).astype(np.float32)
    return {
        "query_img_id": idx,
        "query_image_name": f"seq/frame{idx:05d}.png",
        "query_image_stem": f"seq_frame{idx:05d}",
        "pose_init": candidates[0].copy(),
        "init_source": "base_init",
        "retrieval_frame_id": 10,
        "retrieval_image_name": "ref/base.png",
        "retrieval_score": 0.7,
        "pose_init_candidates": candidates,
        "candidate_valid_mask": np.array([True, False], dtype=bool),
        "retrieval_frame_ids_candidates": np.array([10, 20], dtype=np.int64),
        "retrieval_image_names_candidates": np.array(["ref/a.png", "ref/b.png"]),
        "retrieval_scores_candidates": np.array([0.7, 0.6], dtype=np.float32),
        "retrieval_pnp_success_candidates": np.array([1.0, 0.0], dtype=np.float32),
        "retrieval_pnp_num_inliers_candidates": np.array([11.0, 0.0], dtype=np.float32),
        "retrieval_pnp_num_matches_candidates": np.array([20.0, 0.0], dtype=np.float32),
        "retrieval_pnp_reproj_rmse_candidates": np.array([2.0, np.inf], dtype=np.float32),
        "retrieval_pnp_reproj_median_candidates": np.array([1.0, np.inf], dtype=np.float32),
        "retrieval_pnp_inlier_ratio_candidates": np.array([0.55, 0.0], dtype=np.float32),
        "retrieval_pnp_inlier_conf_mean_candidates": np.array([0.4, 0.0], dtype=np.float32),
    }


def test_append_refined_init_candidate_preserves_active_pose_and_appends_quality_fields(tmp_path):
    base_path = tmp_path / "base.npz"
    save_retrieval_init_entries([_base_entry(1)], {"method_used": "base"}, str(base_path))
    refined_path = tmp_path / "refined.npz"
    np.savez(
        refined_path,
        query_image_names=np.array(["seq/frame00001.png"]),
        query_image_stems=np.array(["seq_frame00001"]),
        pose_inits=np.stack([_pose_at(101.0)]).astype(np.float32),
        init_sources=np.array(["render_loftr_refine"]),
        refine_success=np.array([True], dtype=bool),
        refine_num_inliers=np.array([30], dtype=np.int32),
        refine_num_raw_matches=np.array([50], dtype=np.int32),
    )
    save_path = tmp_path / "appended.npz"

    exported_entries, stats = append_refined_init_candidate(
        str(base_path),
        str(refined_path),
        str(save_path),
        candidate_name="render_init_loftr_refined",
    )
    loaded_entries, loaded_stats = load_retrieval_init_entries(str(save_path))
    entry = loaded_entries[0]

    assert stats == loaded_stats
    assert stats["num_entries"] == 1
    assert stats["num_appended_valid"] == 1
    assert len(exported_entries) == 1
    assert np.allclose(entry["pose_init"], _pose_at(1.0))
    assert entry["init_source"] == "base_init"
    assert entry["pose_init_candidates"].shape == (3, 4, 4)
    assert np.allclose(entry["pose_init_candidates"][2], _pose_at(101.0))
    assert entry["candidate_valid_mask"].tolist() == [True, False, True]
    assert entry["retrieval_image_names_candidates"][2] == "render_init_loftr_refined"
    assert entry["retrieval_pnp_success_candidates"].tolist() == [1.0, 0.0, 1.0]
    assert entry["retrieval_pnp_num_inliers_candidates"].tolist() == [11.0, 0.0, 30.0]
    assert entry["retrieval_pnp_num_matches_candidates"].tolist() == [20.0, 0.0, 50.0]
    assert entry["retrieval_pnp_inlier_ratio_candidates"][2] == np.float32(0.6)
    assert np.isinf(entry["retrieval_pnp_reproj_rmse_candidates"][2])


def test_append_refined_init_candidate_can_keep_compact_base_prefix(tmp_path):
    base_path = tmp_path / "base.npz"
    save_retrieval_init_entries([_base_entry(1)], {"method_used": "base"}, str(base_path))
    refined_path = tmp_path / "refined.npz"
    np.savez(
        refined_path,
        query_image_names=np.array(["seq/frame00001.png"]),
        query_image_stems=np.array(["seq_frame00001"]),
        pose_inits=np.stack([_pose_at(101.0)]).astype(np.float32),
        init_sources=np.array(["render_loftr_refine"]),
        refine_success=np.array([True], dtype=bool),
        refine_num_inliers=np.array([30], dtype=np.int32),
        refine_num_raw_matches=np.array([50], dtype=np.int32),
    )

    append_refined_init_candidate(
        str(base_path),
        str(refined_path),
        str(tmp_path / "compact.npz"),
        candidate_name="render_init_loftr_refined",
        max_base_candidates=1,
    )
    loaded_entries, _stats = load_retrieval_init_entries(str(tmp_path / "compact.npz"))

    entry = loaded_entries[0]
    assert entry["pose_init_candidates"].shape == (2, 4, 4)
    assert np.allclose(entry["pose_init_candidates"][0], _pose_at(1.0))
    assert np.allclose(entry["pose_init_candidates"][1], _pose_at(101.0))
    assert entry["candidate_valid_mask"].tolist() == [True, True]
    assert entry["retrieval_image_names_candidates"] == ["ref/a.png", "render_init_loftr_refined"]
    assert entry["retrieval_pnp_num_inliers_candidates"].tolist() == [11.0, 30.0]
