from __future__ import annotations

import numpy as np
import json
import torch

from feature_extract.tools.vfm.evaluate_goal_maplet_visibility_pose_acquisition import (
    _pose_errors,
)
from feature_extract.vfm.localization_goal_maplet.local_pose_supervision import (
    LOCAL_POSE_SUPERVISION_SEMANTICS,
    build_local_pose_supervision_candidates,
    complete_quadratic_candidate_indices,
    deterministic_global_joint_coordinates,
)
from feature_extract.vfm.localization_goal_maplet.se3_local_quadratic import (
    complete_quadratic_probe_coordinates,
    left_retract_pose_w2c,
)
from feature_extract.tools.vfm.train_evaluate_goal_maplet_fulltoken_pose_ranker import (
    _load_local_supervision_dataset,
    _tiered_energy_landscape_metrics,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_fulltoken_pose_ranker import (
    _domain_retention_metrics,
    _trajectory_monotonic_metrics,
)
from feature_extract.tools.vfm.merge_goal_maplet_local_and_natural_pose_supervision import (
    _validate_anchor_evidence,
)
from feature_extract.tools.vfm.build_goal_maplet_pose_energy_trajectory_supervision import (
    interpolate_pose_product,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256
from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    multiscale_pose_ranking_loss,
    natural_hard_negative_pose_ranking_loss,
    trajectory_monotonic_pose_ranking_loss,
)


def test_local_pose_supervision_is_deterministic_multiscale_and_error_exact():
    target = np.eye(4)
    target[:3, 3] = [1.0, -2.0, 3.0]
    first = build_local_pose_supervision_candidates(target)
    second = build_local_pose_supervision_candidates(target)
    assert LOCAL_POSE_SUPERVISION_SEMANTICS.endswith("_v3")
    assert first[0].shape == (315, 4, 4)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    translation, rotation = _pose_errors(first[0], target)
    np.testing.assert_allclose(translation, first[1], atol=1.0e-7)
    np.testing.assert_allclose(rotation, first[2], atol=1.0e-5)
    assert np.max(first[1][:67]) == 8.0
    assert np.max(first[1]) <= np.sqrt(3.0) * 8.0 + 1.0e-5
    assert np.max(first[2][:67]) == 40.0
    assert np.max(first[2]) <= 45.0 + 1.0e-5
    # The final 120 rows are all signed two-axis combinations at 0.5x/1x.
    assert np.all((first[1][67:187] > 0.0) | (first[2][67:187] > 0.0))
    assert np.any((first[1][67:187] > 0.0) & (first[2][67:187] > 0.0))
    index = complete_quadratic_candidate_indices()
    coordinates = complete_quadratic_probe_coordinates()
    assert set(index) == set(coordinates)
    for name, coordinate in coordinates.items():
        expected = left_retract_pose_w2c(
            target, coordinate, translation_step_m=1.0, rotation_step_degrees=10.0,
        )
        np.testing.assert_allclose(first[0][index[name]], expected, atol=0.0, rtol=0.0)


def test_global_joint_design_is_symmetric_bounded_and_coupled():
    coordinates = deterministic_global_joint_coordinates()
    assert coordinates.shape == (128, 6)
    np.testing.assert_array_equal(coordinates[1::2], -coordinates[0::2])
    assert np.all(np.max(np.abs(coordinates[:, :3]), axis=1) <= 1.0)
    assert np.all(np.linalg.norm(coordinates[:, 3:], axis=1) <= 1.0 + 1e-12)
    assert np.all(np.linalg.norm(coordinates[:, :3], axis=1) > 0.0)
    assert np.all(np.linalg.norm(coordinates[:, 3:], axis=1) > 0.0)


def test_local_supervision_loader_recomputes_content(tmp_path):
    poses, translation, rotation = build_local_pose_supervision_candidates(np.eye(4))
    arrays = {
        "image_ids": np.asarray(["seq13/frame00001.png"]),
        "source_query_rows": np.asarray([0], dtype=np.int64),
        "candidate_poses_w2c": poses[None],
        "translation_m": translation[None],
        "rotation_deg": rotation[None],
        "candidate_valid": np.ones((1, poses.shape[0]), dtype=bool),
    }
    metadata = {
        "artifact_type": "goal_maplet_local_pose_supervision_dataset_v1",
        "content_sha256": arrays_sha256(arrays),
    }
    path = tmp_path / "local.npz"
    np.savez_compressed(
        path, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    loaded, _ = _load_local_supervision_dataset(path)
    np.testing.assert_array_equal(loaded["translation_m"], translation[None])
    with np.load(path, allow_pickle=False) as data:
        values = {name: np.asarray(data[name]) for name in data.files}
    values["translation_m"] = values["translation_m"].copy()
    values["translation_m"][0, 1] += 1.0
    np.savez_compressed(path, **values)
    with np.testing.assert_raises_regex(ValueError, "content differs"):
        _load_local_supervision_dataset(path)


def test_tiered_energy_landscape_separates_local_and_wide_failures():
    translation = np.asarray([[0.0, 0.25, 0.75, 1.5, 6.0]], dtype=np.float32)
    rotation = np.asarray([[0.0, 2.0, 8.0, 15.0, 35.0]], dtype=np.float32)
    valid = np.ones_like(translation, dtype=bool)
    # Correct through the medium tier, but a distant distractor outranks GT.
    score = np.asarray([[1.0, 0.8, 0.6, 0.4, 1.2]], dtype=np.float32)
    metrics = _tiered_energy_landscape_metrics(score, translation, rotation, valid)
    assert metrics["strict_0_5m_5deg"]["gt_anchor_top1_rate"] == 1.0
    assert metrics["local_1m_10deg"]["gt_anchor_top1_rate"] == 1.0
    assert metrics["medium_2m_20deg"]["gt_anchor_top1_rate"] == 1.0
    assert metrics["wide_8m_45deg"]["gt_anchor_top1_rate"] == 0.0
    assert metrics["wide_8m_45deg"]["mean_gt_anchor_margin"] < 0.0


def test_multiscale_pose_ranking_loss_rewards_anchor_and_local_order():
    translation = np.asarray([[0.0, 0.25, 0.5, 1.0, 2.0, 8.0]], dtype=np.float32)
    rotation = np.asarray([[0.0, 2.5, 5.0, 10.0, 20.0, 45.0]], dtype=np.float32)
    valid = np.ones_like(translation, dtype=bool)
    correct = np.asarray([[2.0, 1.5, 1.0, 0.5, 0.0, -1.0]], dtype=np.float32)
    reversed_score = correct[:, ::-1].copy()
    good = multiscale_pose_ranking_loss(
        torch.as_tensor(correct), torch.as_tensor(translation),
        torch.as_tensor(rotation), torch.as_tensor(valid),
    )
    bad = multiscale_pose_ranking_loss(
        torch.as_tensor(reversed_score), torch.as_tensor(translation),
        torch.as_tensor(rotation), torch.as_tensor(valid),
    )
    assert float(good) < float(bad)


def test_multiscale_loss_skips_only_unavailable_query_scale():
    score = torch.tensor([[2.0, 1.0], [2.0, 0.0]])
    # Query one has strict supervision; query two has only medium/wide support.
    translation = torch.tensor([[0.0, 0.25], [0.0, 1.5]])
    rotation = torch.tensor([[0.0, 2.0], [0.0, 15.0]])
    valid = torch.ones_like(score, dtype=torch.bool)
    loss = multiscale_pose_ranking_loss(score, translation, rotation, valid)
    assert torch.isfinite(loss)


def test_domain_retention_distinguishes_basin_ranking_from_exact_pose():
    score = np.asarray([[9.0, 0.0, 2.0, 1.0]], dtype=np.float32)
    translation = np.asarray([[0.0, 5.0, 1.5, 0.25]], dtype=np.float32)
    rotation = np.asarray([[0.0, 10.0, 15.0, 40.0]], dtype=np.float32)
    valid = np.ones_like(score, dtype=bool)
    metrics = _domain_retention_metrics(score, translation, rotation, valid)
    # Candidate two is the highest-scored medium basin even though candidate
    # three is spatially close and no candidate is an exact strict solution.
    assert metrics["medium_2m_20deg"]["ranked_domain_recall_at_1"] == 1.0
    assert metrics["declared_2m_45deg"]["raw_candidate_domain_recall"] == 1.0


def test_anchor_evidence_merge_accepts_one_float16_ulp_only():
    anchor = np.zeros((9, 1, 1), dtype=np.float16)
    anchor[0, 0, 0] = np.float16(0.25)
    adjacent = anchor.copy()
    adjacent[0, 0, 0] = np.nextafter(
        adjacent[0, 0, 0], np.float16(np.inf), dtype=np.float16,
    )
    # Near-zero signed phase reductions may span several ULPs while remaining
    # far below the common half-epsilon absolute replay bound.
    adjacent[3, 0, 0] = np.float16(7.0e-6)
    audit = _validate_anchor_evidence(anchor, adjacent)
    assert audit["maximum_stable_channel_float16_ulp_distance"] == 1
    assert audit["changed_value_count"] == 2

    two_steps = adjacent.copy()
    two_steps[0, 0, 0] = np.nextafter(
        two_steps[0, 0, 0], np.float16(np.inf), dtype=np.float16,
    )
    with np.testing.assert_raises_regex(ValueError, "stable.*over one ULP"):
        _validate_anchor_evidence(anchor, two_steps)


def test_natural_hard_negative_loss_is_not_diluted_by_local_rows():
    # Rows 1:5 are local probes; rows 5: are frozen natural candidates.
    translation = torch.tensor([[0.0, 0.1, 0.5, 1.0, 2.0, 1.5, 6.0]])
    rotation = torch.tensor([[0.0, 1.0, 5.0, 10.0, 20.0, 15.0, 35.0]])
    valid = torch.ones_like(translation, dtype=torch.bool)
    good = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]])
    bad = good.clone()
    bad[0, 5:] = torch.tensor([4.0, 2.0])
    good_loss = natural_hard_negative_pose_ranking_loss(
        good, translation, rotation, valid, natural_candidate_start_row=5,
    )
    bad_loss = natural_hard_negative_pose_ranking_loss(
        bad, translation, rotation, valid, natural_candidate_start_row=5,
    )
    assert float(good_loss) < float(bad_loss)

    # Changing only the numerous local-probe logits cannot change this term.
    changed_local = good.clone()
    changed_local[0, 1:5] = 100.0
    changed_loss = natural_hard_negative_pose_ranking_loss(
        changed_local, translation, rotation, valid, natural_candidate_start_row=5,
    )
    torch.testing.assert_close(good_loss, changed_loss)


def test_pose_energy_trajectory_uses_center_line_and_shortest_rotation_arc():
    seed = np.eye(4, dtype=np.float64)
    target = np.eye(4, dtype=np.float64)
    angle = np.deg2rad(40.0)
    target[:3, :3] = np.asarray([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    target_center = np.asarray([2.0, -4.0, 6.0])
    target[:3, 3] = -target[:3, :3] @ target_center
    np.testing.assert_array_equal(interpolate_pose_product(seed, target, 0.0), seed)
    np.testing.assert_array_equal(interpolate_pose_product(seed, target, 1.0), target)
    halfway = interpolate_pose_product(seed, target, 0.5)
    center = -halfway[:3, :3].T @ halfway[:3, 3]
    np.testing.assert_allclose(center, target_center / 2.0, atol=1.0e-12)
    _, rotation_to_seed = _pose_errors(halfway[None], seed)
    _, rotation_to_target = _pose_errors(halfway[None], target)
    np.testing.assert_allclose(rotation_to_seed, [20.0], atol=1.0e-8)
    np.testing.assert_allclose(rotation_to_target, [20.0], atol=1.0e-8)


def test_trajectory_monotonic_loss_prefers_rising_paths_and_equalizes_seeds():
    # Anchor plus two three-step paths.  Local ordering inside each seed is
    # averaged before seeds are averaged, so one path cannot dominate by size.
    seed = torch.tensor([[-1, 1, 1, 1, 2, 2, 2]])
    alpha = torch.tensor([[1.0, 0.0, 0.4, 0.8, 0.0, 0.4, 0.8]])
    valid = torch.ones_like(seed, dtype=torch.bool)
    rising = torch.tensor([[3.0, 0.0, 1.0, 2.0, -1.0, 0.0, 1.0]])
    falling = torch.tensor([[3.0, 2.0, 1.0, 0.0, 1.0, 0.0, -1.0]])
    good = trajectory_monotonic_pose_ranking_loss(rising, seed, alpha, valid)
    bad = trajectory_monotonic_pose_ranking_loss(falling, seed, alpha, valid)
    assert float(good) < float(bad)
    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        trajectory_monotonic_pose_ranking_loss(
            rising, seed, alpha.scatter(1, torch.tensor([[2]]), 0.0), valid,
        )


def test_trajectory_metrics_separate_local_steps_from_complete_paths():
    seeds = np.asarray([[-1, 1, 1, 1, 2, 2, 2]], dtype=np.int64)
    alpha = np.asarray([[1.0, 0.0, 0.5, 0.9, 0.0, 0.5, 0.9]])
    score = np.asarray([[4.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0]])
    metrics = _trajectory_monotonic_metrics(
        score, seeds, alpha, np.ones_like(seeds, dtype=bool),
    )
    assert metrics["consecutive_nondecreasing_rate"] == 0.75
    assert metrics["complete_path_nondecreasing_rate"] == 0.5
    assert metrics["anchor_above_last_path_sample_rate"] == 1.0
