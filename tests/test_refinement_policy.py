from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.localizability.refinement_policy import (  # noqa: E402
    build_guarded_refinement_entries,
    pose_delta_trans_rot,
)


def _pose_at(center_x_m: float, *, yaw_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(float(yaw_deg))
    c = np.cos(yaw)
    s = np.sin(yaw)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pose[0, 3] = -float(center_x_m)
    return pose


def _entry(name: str, center_x_m: float, *, source: str) -> dict:
    pose = _pose_at(center_x_m)
    return {
        "query_img_id": 0,
        "query_image_name": name,
        "query_image_stem": Path(name).with_suffix("").as_posix().replace("/", "_"),
        "pose_init": pose,
        "init_source": source,
        "retrieval_frame_id": -1,
        "retrieval_image_name": "",
        "retrieval_score": 0.0,
        "pose_init_candidates": pose[None],
        "candidate_valid_mask": np.array([True]),
        "retrieval_frame_ids_candidates": np.array([-1], dtype=np.int64),
        "retrieval_image_names_candidates": [""],
        "retrieval_scores_candidates": np.array([0.0], dtype=np.float32),
    }


def test_pose_delta_reports_camera_center_and_rotation_difference():
    trans_m, rot_deg = pose_delta_trans_rot(_pose_at(0.0), _pose_at(0.25, yaw_deg=3.0))

    assert trans_m == pytest.approx(0.25, abs=1.0e-6)
    assert rot_deg == pytest.approx(3.0, abs=1.0e-3)


def test_guarded_refinement_accepts_only_successful_small_delta_updates():
    identity_entries = [
        _entry("seq/q_accept.png", 0.00, source="pofd_top1"),
        _entry("seq/q_large_delta.png", 0.00, source="pofd_top1"),
        _entry("seq/q_low_inliers.png", 0.00, source="pofd_top1"),
        _entry("seq/q_failed.png", 0.00, source="pofd_top1"),
    ]
    refined_entries = [
        _entry("seq/q_accept.png", 0.04, source="render_loftr"),
        _entry("seq/q_large_delta.png", 0.90, source="render_loftr"),
        _entry("seq/q_low_inliers.png", 0.03, source="render_loftr"),
        _entry("seq/q_failed.png", 0.02, source="render_loftr"),
    ]
    refinement_metadata = {
        "seq/q_accept.png": {"refine_success": True, "refine_num_inliers": 1800},
        "seq/q_large_delta.png": {"refine_success": True, "refine_num_inliers": 1900},
        "seq/q_low_inliers.png": {"refine_success": True, "refine_num_inliers": 20},
        "seq/q_failed.png": {"refine_success": False, "refine_num_inliers": 2000},
    }

    guarded_entries, stats, diagnostics = build_guarded_refinement_entries(
        identity_entries,
        refined_entries,
        refinement_metadata=refinement_metadata,
        min_inliers=1000,
        max_delta_trans_m=0.35,
        max_delta_rot_deg=5.0,
    )

    assert [entry["query_image_name"] for entry in guarded_entries] == [
        "seq/q_accept.png",
        "seq/q_large_delta.png",
        "seq/q_low_inliers.png",
        "seq/q_failed.png",
    ]
    assert guarded_entries[0]["pose_init"][0, 3] == pytest.approx(-0.04)
    assert guarded_entries[0]["init_source"] == "guarded_render_loftr_refine"
    assert guarded_entries[1]["pose_init"][0, 3] == pytest.approx(0.0)
    assert guarded_entries[1]["init_source"] == "guarded_identity_fallback"
    assert guarded_entries[2]["init_source"] == "guarded_identity_fallback"
    assert guarded_entries[3]["init_source"] == "guarded_identity_fallback"
    assert stats["accepted_refined"] == 1
    assert stats["fallback_identity"] == 3
    assert stats["reject_large_delta"] == 1
    assert stats["reject_low_inliers"] == 1
    assert stats["reject_solver_failed"] == 1
    assert diagnostics["accepted_mask"].tolist() == [True, False, False, False]
