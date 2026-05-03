from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from feature_retrieval.render_loftr_pnp_init_export import (
    candidate_ref_entries_from_retrieval_entry,
    loftr_result_to_candidate,
)


class _Result:
    def __init__(self, *, success, pose_w2c=None, num_inliers=0, failure_reason=""):
        self.success = success
        self.pose_w2c = pose_w2c
        self.num_inliers = num_inliers
        self.failure_reason = failure_reason


def test_candidate_ref_entries_from_retrieval_entry_respects_topk_and_valid_mask():
    entry = {
        "pose_init_candidates": np.stack([np.eye(4), np.eye(4) * 2, np.eye(4) * 3]).astype(np.float32),
        "candidate_valid_mask": np.array([True, False, True]),
        "retrieval_frame_ids_candidates": np.array([10, 20, 30], dtype=np.int64),
        "retrieval_image_names_candidates": ["a.png", "b.png", "c.png"],
        "retrieval_scores_candidates": np.array([0.9, 0.8, 0.7], dtype=np.float32),
    }

    refs = candidate_ref_entries_from_retrieval_entry(entry, topk=3)

    assert [r["retrieval_frame_id"] for r in refs] == [10, 30]
    assert [r["retrieval_image_name"] for r in refs] == ["a.png", "c.png"]
    assert np.allclose(refs[1]["fallback_pose_w2c"], np.eye(4, dtype=np.float32) * 3)


def test_loftr_result_to_candidate_uses_estimated_pose_on_success():
    fallback = np.eye(4, dtype=np.float32)
    estimated = np.eye(4, dtype=np.float32)
    estimated[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    candidate = loftr_result_to_candidate(
        _Result(success=True, pose_w2c=estimated, num_inliers=42),
        fallback_pose_w2c=fallback,
        retrieval_frame_id=7,
        retrieval_image_name="ref.png",
        retrieval_score=0.5,
    )

    assert candidate["pnp_success"] is True
    assert candidate["num_inliers"] == 42
    assert np.allclose(candidate["pose_w2c"], estimated)


def test_loftr_result_to_candidate_falls_back_to_retrieval_pose_on_failure():
    fallback = np.eye(4, dtype=np.float32) * 4.0

    candidate = loftr_result_to_candidate(
        _Result(success=False, num_inliers=3, failure_reason="too_few_matches"),
        fallback_pose_w2c=fallback,
        retrieval_frame_id=8,
        retrieval_image_name="ref_fail.png",
        retrieval_score=0.25,
    )

    assert candidate["pnp_success"] is False
    assert candidate["num_inliers"] == 3
    assert candidate["failure_reason"] == "too_few_matches"
    assert np.allclose(candidate["pose_w2c"], fallback)
