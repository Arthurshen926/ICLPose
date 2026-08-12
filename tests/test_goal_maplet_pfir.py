import numpy as np

from feature_extract.vfm.localization_goal_maplet.pfir import (
    ContributorLabels,
    QuerySupportPosterior,
    contributor_maplet_distribution,
    contributor_multiscale_in_map_probability,
    contributor_multiscale_maplet_distribution,
    contributor_multiscale_child_distribution,
    evaluate_pfir,
)

from test_goal_maplet_physical_map import _inputs
from feature_extract.vfm.localization_goal_maplet import build_goal_maplet_physical_map


def test_pfir_uses_multi_positive_contributor_mass():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4)
    labels = ContributorLabels(
        topk_primitive_ids=np.asarray([[[0, -1], [1, -1]], [[8, -1], [-1, -1]]]),
        topk_weights=np.asarray([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]),
        pose_w2c=np.eye(4),
    )
    truth, null = contributor_maplet_distribution(labels, physical, [[0.5, 0.5]], [[0.5, 0.5]])
    np.testing.assert_allclose(truth[0, 0], 2.0 / 3.0)
    np.testing.assert_allclose(null[0], 1.0 / 3.0)
    posterior = QuerySupportPosterior(
        xy=np.asarray([[0.5, 0.5]]),
        extent=np.asarray([[0.5, 0.5]]),
        candidate_maplet_ids=np.asarray([[7, -1]]),
        candidate_probabilities=np.asarray([[0.7, 0.0]]),
        null_probabilities=np.asarray([0.3]),
    )
    report = evaluate_pfir(posterior, truth, null, physical, np.eye(4))
    assert report["weighted_recall_at_1"] == 1.0
    assert report["multi_positive_ap"] > 0.0


def test_multiscale_pfir_uses_weighted_masks_not_average_box():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4)
    ids = np.full((4, 4, 1), -1, dtype=np.int64)
    weights = np.zeros((4, 4, 1), dtype=np.float32)
    ids[:2, :2, 0], weights[:2, :2, 0] = 0, 1.0
    # Primitive 8 is a declared clean scene primitive but is not owned by the
    # only maplet, so it contributes exact null mass.
    ids[2:, 2:, 0], weights[2:, 2:, 0] = 8, 1.0
    labels = ContributorLabels(ids, weights, np.eye(4))
    truth, null = contributor_multiscale_maplet_distribution(
        labels,
        physical,
        np.asarray([[0, 0], [1, 1]]),
        token_height=2,
        token_width=2,
        pool_sizes=(1,),
        pool_weights=(1.0,),
    )
    np.testing.assert_allclose([truth[0, 0], null[0]], [1.0, 0.0])
    np.testing.assert_allclose([truth[1, 0], null[1]], [0.0, 1.0])


def test_grouped_multiscale_pfir_preserves_irregular_union():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4)
    ids = np.full((4, 4, 1), 8, dtype=np.int64)
    weights = np.ones((4, 4, 1), dtype=np.float32)
    ids[:2, :2, 0] = 0
    ids[2:, 2:, 0] = 1
    labels = ContributorLabels(ids, weights, np.eye(4))
    truth, null = contributor_multiscale_maplet_distribution(
        labels,
        physical,
        np.asarray([[0, 0], [1, 1]]),
        token_height=2,
        token_width=2,
        pool_sizes=(1,),
        pool_weights=(1.0,),
        group_member_offsets=np.asarray([0, 2]),
        group_member_token_indices=np.asarray([0, 1]),
    )
    np.testing.assert_allclose([truth[0, 0], null[0]], [1.0, 0.0])


def test_integral_validity_target_is_exactly_equivalent_to_full_distribution():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(
        maplets, region, geometry, poses,
        minimum_child_count=2, maximum_child_count=4,
    )
    rng = np.random.default_rng(7)
    ids = rng.choice(np.asarray([-1, 0, 1, 2, 8]), size=(7, 9, 3))
    weights = rng.random((7, 9, 3), dtype=np.float32)
    weights[ids < 0] = 0.0
    labels = ContributorLabels(ids, weights, np.eye(4))
    token_xy = np.stack(np.meshgrid(np.arange(3), np.arange(2)), axis=-1).reshape(-1, 2)
    truth, expected_null = contributor_multiscale_maplet_distribution(
        labels, physical, token_xy,
        token_height=2, token_width=3,
        pool_sizes=(1, 3, 5), pool_weights=(0.5, 0.3, 0.2),
    )
    in_map, actual_null = contributor_multiscale_in_map_probability(
        labels, physical, token_xy,
        token_height=2, token_width=3,
        pool_sizes=(1, 3, 5), pool_weights=(0.5, 0.3, 0.2),
    )
    np.testing.assert_allclose(actual_null, expected_null, atol=1.0e-6)
    np.testing.assert_allclose(in_map, np.sum(truth, axis=1), atol=1.0e-6)


def test_integral_validity_target_matches_grouped_irregular_union():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(
        maplets, region, geometry, poses,
        minimum_child_count=2, maximum_child_count=4,
    )
    ids = np.asarray([[[0], [8]], [[1], [8]]], dtype=np.int64)
    labels = ContributorLabels(ids, np.ones_like(ids, dtype=np.float32), np.eye(4))
    token_xy = np.asarray([[0, 0], [1, 1]])
    offsets = np.asarray([0, 2])
    members = np.asarray([0, 1])
    truth, expected_null = contributor_multiscale_maplet_distribution(
        labels, physical, token_xy,
        token_height=2, token_width=2,
        pool_sizes=(1,), pool_weights=(1.0,),
        group_member_offsets=offsets, group_member_token_indices=members,
    )
    in_map, actual_null = contributor_multiscale_in_map_probability(
        labels, physical, token_xy,
        token_height=2, token_width=2,
        pool_sizes=(1,), pool_weights=(1.0,),
        group_member_offsets=offsets, group_member_token_indices=members,
    )
    np.testing.assert_allclose(actual_null, expected_null, atol=1.0e-6)
    np.testing.assert_allclose(in_map, np.sum(truth, axis=1), atol=1.0e-6)


def test_child_pfir_preserves_parent_partition_mass():
    geometry, maplets, region, poses = _inputs()
    physical = build_goal_maplet_physical_map(maplets, region, geometry, poses, minimum_child_count=2, maximum_child_count=4)
    labels = ContributorLabels(
        np.asarray([[[0], [1]], [[2], [8]]]),
        np.ones((2, 2, 1), dtype=np.float32),
        np.eye(4),
    )
    truth, null = contributor_multiscale_child_distribution(
        labels,
        physical,
        np.asarray([[0, 0], [1, 1]]),
        token_height=2,
        token_width=2,
    )
    np.testing.assert_allclose(np.sum(truth[0]), 1.0)
    np.testing.assert_allclose(null[0], 0.0)
    np.testing.assert_allclose(np.sum(truth[1]), 0.0)
    np.testing.assert_allclose(null[1], 1.0)
