from __future__ import annotations

import pytest
import numpy as np
import torch

from feature_extract.vfm.localization_goal_maplet.fulltoken_pose_ranking import (
    FACTORIZED_FULLTOKEN_POSE_RANKER_SEMANTICS,
    FULLTOKEN_POSE_RANKING_CHANNELS,
    TYPED_FULLTOKEN_POSE_RANKING_CHANNELS,
    FactorizedFullTokenCandidatePoseRanker,
    FullTokenCandidatePoseRanker,
    compact_fulltoken_pose_ranking_features,
    compact_typed_fulltoken_pose_ranking_features,
    continuous_seed_domain_recall_metrics,
    distinct_pose_basin_recall_metrics,
    greedy_distinct_pose_basin_order,
    listwise_pose_ranking_loss,
    load_factorized_fulltoken_candidate_pose_ranker,
    load_fulltoken_candidate_pose_ranker,
    round_robin_union_score_rows,
)
from feature_extract.vfm.localization_goal_maplet.lineage import arrays_sha256


def test_compact_fulltoken_features_preserve_token_and_two_phase_axes():
    height, width, dim = 3, 4, 5
    query = torch.randn(height * width, dim, generator=torch.Generator().manual_seed(7))
    target = query[:, None].clone()
    mass = torch.full((height * width, 1), 0.75)
    valid = torch.ones_like(mass, dtype=torch.bool)
    features = compact_fulltoken_pose_ranking_features(
        query, target, mass, valid, height=height, width=width,
    )
    assert features.shape == (len(FULLTOKEN_POSE_RANKING_CHANNELS), height, width)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[0], torch.ones(height, width))
    torch.testing.assert_close(features[1], torch.full((height, width), 0.75))
    torch.testing.assert_close(features[2], torch.full((height, width), 0.75))
    torch.testing.assert_close(features[3, :, :-1], torch.ones(height, width - 1))
    torch.testing.assert_close(features[6, :-1], torch.ones(height - 1, width))
    assert torch.count_nonzero(features[3, :, -1]) == 0
    assert torch.count_nonzero(features[6, -1]) == 0


def test_missing_target_zeroes_support_and_above_floor_evidence():
    query = torch.randn(12, 4, generator=torch.Generator().manual_seed(9))
    target = query[:, None].clone()
    mass = torch.ones(12, 1)
    valid = torch.ones(12, 1, dtype=torch.bool)
    valid[5] = False
    features = compact_fulltoken_pose_ranking_features(
        query, target, mass, valid, height=3, width=4,
    )
    row, column = divmod(5, 4)
    assert features[1, row, column] == 0
    assert features[2, row, column] == 0
    # Both incident horizontal edges and both incident vertical edges vanish.
    assert features[4, row, column - 1] == 0
    assert features[4, row, column] == 0
    assert features[7, row - 1, column] == 0
    assert features[7, row, column] == 0


def test_compact_features_reject_multi_slot_target_to_avoid_hidden_pooling():
    with pytest.raises(ValueError, match="exactly one"):
        compact_fulltoken_pose_ranking_features(
            torch.ones(4, 3), torch.ones(4, 2, 3),
            torch.ones(4, 2), torch.ones(4, 2, dtype=torch.bool),
            height=2, width=2,
        )


def test_spatial_ranker_only_consumes_candidate_conditioned_feature_grid():
    model = FullTokenCandidatePoseRanker()
    features = torch.randn(4, len(FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64)
    score = model(features)
    assert score.shape == (4,)
    assert torch.isfinite(score).all()


def test_factorized_ranker_shares_encoder_but_preserves_three_independent_heads():
    reference = FullTokenCandidatePoseRanker()
    model = FactorizedFullTokenCandidatePoseRanker()
    model.initialize_from_single_ranker(reference)
    features = torch.randn(3, len(FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64)
    initial = model(features)
    assert set(initial) == {"location", "orientation", "joint"}
    torch.testing.assert_close(initial["location"], initial["orientation"])
    torch.testing.assert_close(initial["location"], initial["joint"])
    with torch.no_grad():
        model.location_head[-1].bias.add_(1.0)
    changed = model(features)
    torch.testing.assert_close(changed["orientation"], changed["joint"])
    torch.testing.assert_close(changed["location"], changed["joint"] + 1.0)


def test_typed_geometry_appends_sign_invariant_layout_and_warm_start_preserves_score():
    query = torch.randn(4, 3, generator=torch.Generator().manual_seed(17))
    target = query[:, None].clone()
    mass = torch.ones(4, 1)
    valid = torch.ones(4, 1, dtype=torch.bool)
    axis = torch.zeros(2, 2, 6)
    axis[..., 0] = 1.0
    typed = compact_typed_fulltoken_pose_ranking_features(
        query, target, mass, valid, axis,
        torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 2),
        height=2, width=2,
    )
    assert typed.shape == (len(TYPED_FULLTOKEN_POSE_RANKING_CHANNELS), 2, 2)
    reference = FactorizedFullTokenCandidatePoseRanker()
    expanded = FactorizedFullTokenCandidatePoseRanker(
        input_channels=len(TYPED_FULLTOKEN_POSE_RANKING_CHANNELS)
    )
    expanded.initialize_from_factorized_ranker(reference)
    old = torch.randn(2, len(FULLTOKEN_POSE_RANKING_CHANNELS), 36, 64)
    extra = torch.zeros(2, len(TYPED_FULLTOKEN_POSE_RANKING_CHANNELS) - old.shape[1], 36, 64)
    expected, actual = reference(old), expanded(torch.cat((old, extra), dim=1))
    for name in reference.HEAD_NAMES:
        torch.testing.assert_close(actual[name], expected[name])


def test_factorized_ranker_loader_recomputes_state_content_hash(tmp_path):
    model = FactorizedFullTokenCandidatePoseRanker()
    state = model.state_dict()
    arrays = {name: value.detach().numpy() for name, value in state.items()}
    path = tmp_path / "factorized.pt"
    torch.save({
        "artifact_type": "goal_maplet_factorized_fulltoken_candidate_pose_ranker_v1",
        "model_semantics": FACTORIZED_FULLTOKEN_POSE_RANKER_SEMANTICS,
        "model_content_sha256": arrays_sha256(arrays),
        "state_dict": state,
    }, path)
    loaded, metadata = load_factorized_fulltoken_candidate_pose_ranker(path)
    assert isinstance(loaded, FactorizedFullTokenCandidatePoseRanker)
    assert metadata["model_semantics"] == FACTORIZED_FULLTOKEN_POSE_RANKER_SEMANTICS
    payload = torch.load(path)
    payload["state_dict"]["joint_head.2.bias"] += 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="content differs"):
        load_factorized_fulltoken_candidate_pose_ranker(path)


def test_listwise_pose_ranking_loss_rewards_correct_order():
    translation = torch.tensor([[0.0, 0.5, 2.0]])
    rotation = torch.tensor([[0.0, 5.0, 20.0]])
    valid = torch.ones_like(translation, dtype=torch.bool)
    correct = listwise_pose_ranking_loss(
        torch.tensor([[3.0, 1.0, -2.0]]), translation, rotation, valid,
    )
    reversed_order = listwise_pose_ranking_loss(
        torch.tensor([[-2.0, 1.0, 3.0]]), translation, rotation, valid,
    )
    assert correct < reversed_order


def test_greedy_pose_basin_order_removes_duplicate_without_duplicate_fill():
    poses = torch.eye(4).repeat(5, 1, 1).numpy()
    poses[2, 0, 3] = -0.2
    poses[3, 0, 3] = -1.0
    angle = torch.deg2rad(torch.tensor(10.0)).item()
    poses[4, :3, :3] = torch.tensor([
        [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0.0],
        [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle)), 0.0],
        [0.0, 0.0, 1.0],
    ]).numpy()
    order = greedy_distinct_pose_basin_order(
        np.asarray([100.0, 4.0, 3.0, 2.0, 1.0]), poses, np.ones(5, dtype=bool),
    )
    # Candidate zero is the diagnostic anchor; candidate two is within the
    # same 0.5m/5deg basin as the higher-scored candidate one.
    np.testing.assert_array_equal(order, [1, 3, 4])


def test_vectorized_greedy_pose_nms_matches_scalar_reference():
    generator = np.random.default_rng(71)
    count = 96
    poses = np.tile(np.eye(4), (count, 1, 1))
    centers = generator.normal(size=(count, 3))
    axis = generator.normal(size=(count, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = generator.uniform(-0.4, 0.4, size=count)
    for row in range(count):
        x, y, z = axis[row]
        skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        rotation = (
            np.eye(3) + np.sin(angle[row]) * skew
            + (1.0 - np.cos(angle[row])) * (skew @ skew)
        )
        poses[row, :3, :3] = rotation
        poses[row, :3, 3] = -rotation @ centers[row]
    score = generator.normal(size=count)
    valid = generator.random(count) > 0.1

    indices = np.flatnonzero(valid)
    indices = indices[indices != 0]
    ordered = indices[np.lexsort((indices, -score[indices]))]
    scalar: list[int] = []
    for candidate in ordered.tolist():
        duplicate = False
        for previous in scalar:
            translation = np.linalg.norm(centers[candidate] - centers[previous])
            relative = poses[candidate, :3, :3] @ poses[previous, :3, :3].T
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            rotation = np.degrees(np.arccos(cosine))
            if translation <= 0.5 + 1.0e-12 and rotation <= 5.0 + 1.0e-10:
                duplicate = True
                break
        if not duplicate:
            scalar.append(candidate)

    vectorized = greedy_distinct_pose_basin_order(score, poses, valid)
    np.testing.assert_array_equal(vectorized, np.asarray(scalar, dtype=np.int64))


def test_distinct_basin_recall_reports_raw_support_and_ranked_topk_separately():
    poses = np.tile(np.eye(4), (1, 4, 1, 1))
    poses[0, 1, 0, 3] = -3.0
    poses[0, 2, 0, 3] = -1.0
    poses[0, 3, 0, 3] = -0.2
    result = distinct_pose_basin_recall_metrics(
        np.asarray([[100.0, 3.0, 2.0, 1.0]]),
        poses,
        np.asarray([[0.0, 3.0, 1.0, 0.2]]),
        np.asarray([[0.0, 0.0, 8.0, 2.0]]),
        np.ones((1, 4), dtype=bool),
        topk=(1, 2),
    )
    assert result["raw_candidate_strict_0_5m_5deg"] == 1.0
    assert result["strict_recall_at_1"] == 0.0
    assert result["strict_recall_at_2"] == 0.0
    assert result["loose_recall_at_2"] == 1.0


def test_continuous_seed_domain_recall_is_not_coarse_pose_point_recall():
    poses = np.tile(np.eye(4), (1, 3, 1, 1))
    # Candidate one is 7m from the target: not a loose pose-point hit, but its
    # +/-8m continuous translation cube contains the target exactly.
    poses[0, 1, 0, 3] = -7.0
    poses[0, 2, 0, 3] = -20.0
    result = continuous_seed_domain_recall_metrics(
        np.asarray([[100.0, 2.0, 1.0]]), poses, np.ones((1, 3), dtype=bool),
    )
    assert result["raw_strict_domain_acquisition"] == 1.0
    assert result["strict_domain_acquisition_at_1"] == 1.0
    assert result["acquisition_is_not_search_or_localization_success"] is True


def test_ranker_loader_recomputes_state_content_hash(tmp_path):
    model = FullTokenCandidatePoseRanker()
    state = model.state_dict()
    arrays = {name: value.detach().numpy() for name, value in state.items()}
    path = tmp_path / "ranker.pt"
    torch.save({
        "artifact_type": "goal_maplet_fulltoken_candidate_pose_ranker_v1",
        "model_semantics": (
            "shared_candidate_conditioned_spatial_cnn_listwise_ranker_v1"
        ),
        "model_content_sha256": arrays_sha256(arrays),
        "retrieval_prior_fusion_weight": 0.5,
        "state_dict": state,
    }, path)
    loaded, metadata = load_fulltoken_candidate_pose_ranker(path)
    assert isinstance(loaded, FullTokenCandidatePoseRanker)
    assert metadata["retrieval_prior_fusion_weight"] == 0.5
    payload = torch.load(path)
    payload["state_dict"]["head.2.bias"] = payload["state_dict"]["head.2.bias"] + 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="content differs"):
        load_fulltoken_candidate_pose_ranker(path)


def test_round_robin_union_preserves_complementary_expert_heads_without_weight():
    poses = np.tile(np.eye(4), (1, 5, 1, 1))
    for candidate in range(1, 5):
        poses[0, candidate, 0, 3] = -float(candidate)
    experts = np.asarray([[
        [100.0, 4.0, 3.0, 2.0, 1.0],
        [100.0, 1.0, 2.0, 3.0, 4.0],
    ]])
    union = round_robin_union_score_rows(
        experts, poses, np.ones((1, 5), dtype=bool),
    )
    order = greedy_distinct_pose_basin_order(
        union[0], poses[0], np.ones(5, dtype=bool),
    )
    np.testing.assert_array_equal(order[:4], [1, 4, 2, 3])
