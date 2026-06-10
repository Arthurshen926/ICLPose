from __future__ import annotations

import numpy as np

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.render_pose_protocol import (
    RenderPoseSelection,
    group_topk_reference_poses,
    group_top_reference_poses,
    sample_se3_perturbation,
    parse_world_offset,
    select_render_pose,
    translate_pose_world,
)


def test_parse_world_offset_requires_three_values() -> None:
    assert np.allclose(parse_world_offset("0.25,0,-1"), [0.25, 0.0, -1.0])


def test_translate_pose_world_moves_camera_center_without_changing_rotation() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [0.0, 0.0, -2.0]

    shifted = translate_pose_world(pose, np.asarray([0.5, 0.0, 0.0], dtype=np.float64))

    assert np.allclose(shifted[:3, :3], np.eye(3))
    assert np.allclose(shifted[:3, 3], [-0.5, 0.0, -2.0])


def test_sample_se3_perturbation_is_deterministic_and_bounded() -> None:
    pose = np.eye(4, dtype=np.float64)

    first = sample_se3_perturbation(
        pose,
        max_translation_m=0.25,
        max_rotation_deg=5.0,
        seed=7,
        key="q.png:B",
    )
    second = sample_se3_perturbation(
        pose,
        max_translation_m=0.25,
        max_rotation_deg=5.0,
        seed=7,
        key="q.png:B",
    )

    assert np.allclose(first.pose_w2c, second.pose_w2c)
    assert np.linalg.norm(first.translation_world) <= 0.25 + 1e-9
    assert np.linalg.norm(first.rotation_deg_xyz) <= np.sqrt(3.0) * 5.0 + 1e-9
    assert first.translation_error_m <= 0.25 + 1e-9
    assert first.rotation_error_deg <= np.sqrt(3.0) * 5.0 + 1e-6


def test_group_top_reference_poses_selects_best_prior_per_query() -> None:
    candidates = [
        CandidateHypothesis(
            candidate_id="q:1",
            candidate_type="reference_pose",
            query_id="q.png",
            pose=np.eye(4).tolist(),
            pose_error=PoseCost(translation_m=1.0, rotation_deg=0.0),
            prior_score=-1.0,
        ),
        CandidateHypothesis(
            candidate_id="q:0",
            candidate_type="reference_pose",
            query_id="q.png",
            pose=(np.eye(4) * 2.0).tolist(),
            pose_error=PoseCost(translation_m=0.5, rotation_deg=0.0),
            prior_score=0.0,
        ),
    ]

    grouped = group_top_reference_poses(candidates)

    assert grouped["q.png"].candidate_id == "q:0"


def test_group_topk_reference_poses_keeps_ranked_candidates_per_query() -> None:
    candidates = [
        CandidateHypothesis(
            candidate_id=f"q:{idx}",
            candidate_type="reference_pose",
            query_id="q.png",
            pose=np.eye(4).tolist(),
            prior_score=float(10 - idx),
            metadata={"retrieval_rank": idx},
        )
        for idx in range(6)
    ]

    grouped = group_topk_reference_poses(candidates, top_k=5)

    assert [candidate.candidate_id for candidate in grouped["q.png"]] == ["q:0", "q:1", "q:2", "q:3", "q:4"]


def test_select_render_pose_supports_gt_offset_and_reference() -> None:
    gt_pose = np.eye(4, dtype=np.float64)
    reference_pose = np.eye(4, dtype=np.float64)
    reference_pose[:3, 3] = [0.0, 0.0, -3.0]
    reference = CandidateHypothesis(
        candidate_id="ref",
        candidate_type="reference_pose",
        query_id="q.png",
        pose=reference_pose.tolist(),
        pose_error=PoseCost(translation_m=3.0, rotation_deg=0.0),
        prior_score=0.0,
    )

    offset = select_render_pose(
        "q.png",
        gt_pose,
        mode="gt_offset",
        world_offset=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    )
    ref = select_render_pose("q.png", gt_pose, mode="reference_top1", reference_top1={"q.png": reference})

    assert isinstance(offset, RenderPoseSelection)
    assert offset.label == "gt_offset:1.000,0.000,0.000"
    assert np.allclose(offset.pose_w2c[:3, 3], [-1.0, 0.0, 0.0])
    assert ref.label == "reference_top1"
    assert np.allclose(ref.pose_w2c, reference_pose)
