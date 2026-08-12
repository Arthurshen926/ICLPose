from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.sparse_vfm_pose_likelihood import (
    _render_primitive_sample_owner_batch,
    _stratified_primitive_samples,
    score_pose_conditioned_sparse_primitives,
    score_pose_conditioned_sparse_vfm,
)
from feature_extract.vfm.localization_goal_maplet.view_conditioned_field import (
    SCHEMA as VIEW_FIELD_SCHEMA,
    ViewConditionedPrimitiveField,
)


def _physical():
    return SimpleNamespace(
        maplet_ids=np.asarray([10, 20], dtype=np.int64),
        child_centers=np.asarray([[-1.0, 0.0, 4.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        child_normals=np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float64),
        child_extents=np.asarray([[0.25, 0.25, 0.05], [0.25, 0.25, 0.05]], dtype=np.float64),
        child_parent_rows=np.asarray([0, 1], dtype=np.int64),
        maplet_sidedness=np.asarray([2, 2], dtype=np.uint8),
    )


def _primitive_physical():
    value = _physical()
    value.content_sha256 = "synthetic_sparse_primitive_map"
    value.primitive_ids = np.asarray([100, 200], dtype=np.int64)
    value.primitive_centers = value.child_centers.copy()
    value.primitive_normals = value.child_normals.copy()
    value.primitive_tangent1 = np.asarray([[1.0, 0.0, 0.0]] * 2, dtype=np.float64)
    value.primitive_tangent2 = np.asarray([[0.0, 1.0, 0.0]] * 2, dtype=np.float64)
    value.primitive_scale1 = np.asarray([0.40, 0.40], dtype=np.float64)
    value.primitive_scale2 = np.asarray([0.40, 0.40], dtype=np.float64)
    value.primitive_opacity = np.ones((2,), dtype=np.float64)
    value.primitive_sidedness = np.asarray([2, 2], dtype=np.uint8)
    value.child_frames = np.tile(np.eye(3, dtype=np.float64)[None], (2, 1, 1))
    value.child_member_offsets = np.asarray([0, 1, 2], dtype=np.int64)
    value.child_member_primitive_rows = np.asarray([0, 1], dtype=np.int64)
    value.child_member_weights = np.ones((2,), dtype=np.float32)
    value.membership_offsets = np.asarray([0, 1, 2], dtype=np.int64)
    value.membership_primitive_rows = np.asarray([0, 1], dtype=np.int64)
    value.membership_weights = np.ones((2,), dtype=np.float32)
    value.member_slice = lambda row: slice(
        int(value.membership_offsets[row]), int(value.membership_offsets[row + 1])
    )
    return value


def test_sparse_primitive_score_uses_low_rank_view_conditioned_code_inside_chart():
    physical = _primitive_physical()
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    canonical = np.eye(2, dtype=np.float32)
    conditioned_first = np.asarray([1.0, 0.8], dtype=np.float32)
    conditioned_first /= np.linalg.norm(conditioned_first)
    query = np.tile(np.asarray([[1.0, 1.0]], dtype=np.float32), (32, 1))
    query[2 * 8 + 2] = conditioned_first
    query[2 * 8 + 6] = canonical[1]
    view_field = ViewConditionedPrimitiveField(
        primitive_rows=np.asarray([0, 1], dtype=np.int64),
        residual_basis=np.asarray([[0.0, 1.0]], dtype=np.float32),
        coefficients=np.asarray([
            [[0.8], [0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0], [0.0], [0.0]],
        ], dtype=np.float16),
        observation_count=np.asarray([4, 4], dtype=np.int32),
        mean_local_direction=np.asarray([[0.0, 0.0, 1.0]] * 2, dtype=np.float32),
        direction_concentration=np.zeros((2,), dtype=np.float32),
        minimum_direction_cosine=np.full((2,), -1.0, dtype=np.float32),
        mean_log_projected_scale=np.zeros((2,), dtype=np.float32),
        minimum_log_projected_scale=np.full((2,), -10.0, dtype=np.float32),
        maximum_log_projected_scale=np.full((2,), 10.0, dtype=np.float32),
        physical_map_sha256=physical.content_sha256,
        canonical_field_sha256="canonical",
        metadata={"artifact_type": VIEW_FIELD_SCHEMA, "minimum_views": 3},
    )
    baseline = score_pose_conditioned_sparse_primitives(
        np.eye(4)[None], query, np.asarray([0, 1]), canonical, np.ones((2,)),
        physical, camera, token_height=4, token_width=8,
        primitives_per_child=1, maximum_splat_radius_tokens=1,
        score_semantics="fixed_grid", device="cpu",
    )
    conditioned = score_pose_conditioned_sparse_primitives(
        np.eye(4)[None], query, np.asarray([0, 1]), canonical, np.ones((2,)),
        physical, camera, token_height=4, token_width=8,
        primitives_per_child=1, maximum_splat_radius_tokens=1,
        score_semantics="fixed_grid", view_conditioned_field=view_field, device="cpu",
    )
    assert conditioned.scores[0] > baseline.scores[0]
    assert conditioned.conditioned_feature_coverage[0] > 0.0


def test_sparse_vfm_score_uses_pose_to_align_map_features_to_query_tokens():
    physical = _physical()
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    # Under the identity pose the two children project near token centers
    # (2,2) and (6,2).  Swapping the camera x-axis swaps their image order.
    query = np.tile(np.asarray([[1.0, 1.0]], dtype=np.float32), (32, 1))
    query[2 * 8 + 2] = np.asarray([1.0, 0.0])
    query[2 * 8 + 6] = np.asarray([0.0, 1.0])
    map_descriptor = np.eye(2, dtype=np.float32)
    identity = np.eye(4, dtype=np.float64)
    flipped = np.eye(4, dtype=np.float64)
    flipped[0, 0] = -1.0
    flipped[2, 2] = -1.0
    # Keep the scene in front while rotating 180 degrees about camera y.
    flipped[2, 3] = 8.0
    result = score_pose_conditioned_sparse_vfm(
        np.stack([identity, flipped]),
        query, query, map_descriptor, map_descriptor,
        np.ones((2,), dtype=bool), np.ones((2,), dtype=bool),
        physical, camera, token_height=4, token_width=8,
        maximum_splat_radius_tokens=1, batch_size=2, device="cpu",
    )
    assert result.scores.shape == (2,)
    assert result.raw_cosine_scores.shape == (2,)
    assert result.marginal_centered_scores.shape == (2,)
    assert result.log_partition_llr_scores.shape == (2,)
    assert result.scores[0] > result.scores[1]
    assert result.feature_coverage[0] > 0.0


def test_sparse_vfm_score_has_fixed_full_grid_denominator_and_typed_missing():
    physical = _physical()
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    query = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (32, 1))
    descriptor = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = score_pose_conditioned_sparse_vfm(
        np.eye(4)[None], query, query, descriptor, descriptor,
        np.ones((2,), dtype=bool), np.asarray([True, False]),
        physical, camera, token_height=4, token_width=8,
        maximum_splat_radius_tokens=1, device="cpu",
    )
    # The child without a code still occludes, but is explicit missing feature
    # evidence rather than being removed from the rendered denominator.
    assert result.rendered_coverage[0] > result.feature_coverage[0] > 0.0
    assert np.isfinite(result.scores[0])


def test_sparse_vfm_can_score_disconnected_candidate_maplets_with_full_occlusion():
    physical = _physical()
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    query = np.tile(np.asarray([[1.0, 1.0]], dtype=np.float32), (32, 1))
    query[2 * 8 + 2] = np.asarray([1.0, 0.0])
    descriptor = np.eye(2, dtype=np.float32)
    result = score_pose_conditioned_sparse_vfm(
        np.stack([np.eye(4), np.eye(4)]),
        query, query, descriptor, descriptor,
        np.ones((2,), dtype=bool), np.ones((2,), dtype=bool),
        physical, camera, token_height=4, token_width=8,
        allowed_parent_rows=np.asarray([[0, -1], [1, -1]]),
        score_semantics="raw_cosine", maximum_splat_radius_tokens=1, device="cpu",
    )
    # Both children still participate in the z-buffer, but only the identity
    # hypothesis listed for this pose may contribute feature evidence.
    assert np.isclose(result.rendered_coverage[0], result.rendered_coverage[1])
    assert result.feature_coverage[0] > 0.0
    assert result.scores[0] > result.scores[1]


def test_sparse_real_primitive_codes_preserve_pose_dependent_surface_phase():
    physical = _primitive_physical()
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    query = np.tile(np.asarray([[1.0, 1.0]], dtype=np.float32), (32, 1))
    query[2 * 8 + 2] = np.asarray([1.0, 0.0])
    query[2 * 8 + 6] = np.asarray([0.0, 1.0])
    identity = np.eye(4, dtype=np.float64)
    flipped = np.eye(4, dtype=np.float64)
    flipped[0, 0], flipped[2, 2], flipped[2, 3] = -1.0, -1.0, 8.0
    result = score_pose_conditioned_sparse_primitives(
        np.stack([identity, flipped]), query,
        np.asarray([0, 1]), np.eye(2, dtype=np.float32), np.ones((2,)),
        physical, camera, token_height=4, token_width=8,
        primitives_per_child=1, batch_size=2,
        maximum_splat_radius_tokens=1, device="cpu",
    )
    assert result.sample_count == 2
    assert result.fixed_grid_scores.shape == (2,)
    assert result.visible_sample_mean_scores.shape == (2,)
    assert result.scores[0] > result.scores[1]
    assert result.rendered_coverage[0] > 0.0
    assert result.feature_coverage[0] > 0.0


def test_sparse_primitive_z_tie_uses_stable_id_not_input_order():
    physical = _primitive_physical()
    physical.primitive_ids = np.asarray([200, 100], dtype=np.int64)
    physical.primitive_centers[:] = np.asarray([0.0, 0.0, 4.0], dtype=np.float64)
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))

    stable_winners = []
    for rows in (np.asarray([0, 1]), np.asarray([1, 0])):
        owner, rendered = _render_primitive_sample_owner_batch(
            physical, rows, np.eye(4, dtype=np.float64)[None], camera,
            token_height=4, token_width=8, device="cpu",
            maximum_splat_radius_tokens=1,
        )
        owner = owner.cpu().numpy()[0]
        rendered = rendered.cpu().numpy()[0]
        winners = np.full(owner.shape, -1, dtype=np.int64)
        valid = owner >= 0
        winners[valid] = physical.primitive_ids[rows[owner[valid]]]
        assert np.all(winners[valid] == 100)
        assert np.any(rendered)
        stable_winners.append(winners)
    np.testing.assert_array_equal(stable_winners[0], stable_winners[1])


def test_sparse_primitive_score_is_invariant_to_pose_batch_size():
    physical = _primitive_physical()
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    query = np.tile(np.asarray([[1.0, 1.0]], dtype=np.float32), (32, 1))
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[1, 0, 3] = 0.05
    poses[2, 1, 3] = -0.05
    arguments = (
        poses, query, np.asarray([0, 1]), np.eye(2, dtype=np.float32),
        np.ones((2,)), physical, camera,
    )
    first = score_pose_conditioned_sparse_primitives(
        *arguments, token_height=4, token_width=8, primitives_per_child=1,
        batch_size=1, maximum_splat_radius_tokens=1, device="cpu",
    )
    second = score_pose_conditioned_sparse_primitives(
        *arguments, token_height=4, token_width=8, primitives_per_child=1,
        batch_size=3, maximum_splat_radius_tokens=1, device="cpu",
    )
    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.rendered_coverage, second.rendered_coverage)
    np.testing.assert_array_equal(first.feature_coverage, second.feature_coverage)


def test_sparse_primitives_respect_candidate_identity_and_query_region_mask():
    physical = _primitive_physical()
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    query = np.tile(np.asarray([[1.0, 1.0]], dtype=np.float32), (32, 1))
    query[2 * 8 + 2] = np.asarray([1.0, 0.0])
    planned = np.zeros((2, 32), dtype=bool)
    planned[:, 2 * 8 + 2] = True
    result = score_pose_conditioned_sparse_primitives(
        np.stack([np.eye(4), np.eye(4)]), query,
        np.asarray([0, 1]), np.eye(2, dtype=np.float32), np.ones((2,)),
        physical, camera, token_height=4, token_width=8,
        primitives_per_child=1, batch_size=2,
        maximum_splat_radius_tokens=1, score_semantics="fixed_grid",
        allowed_parent_rows=np.asarray([[0, -1], [1, -1]]),
        query_token_mask=planned, device="cpu",
    )
    assert result.scores[0] > result.scores[1]
    assert result.rendered_coverage[0] == result.rendered_coverage[1]


def _foreground_occluder_physical():
    value = _physical()
    value.content_sha256 = "foreground_occluder_map"
    value.maplet_ids = np.asarray([10], dtype=np.int64)
    value.child_centers = np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64)
    value.child_normals = np.asarray([[0.0, 0.0, -1.0]], dtype=np.float64)
    value.child_extents = np.asarray([[0.5, 0.5, 0.05]], dtype=np.float64)
    value.child_parent_rows = np.asarray([0], dtype=np.int64)
    value.maplet_sidedness = np.asarray([2], dtype=np.uint8)
    value.primitive_ids = np.asarray([100, 200], dtype=np.int64)
    # Primitive 0 has no feature/owner and occludes feature-bearing primitive 1.
    value.primitive_centers = np.asarray([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], dtype=np.float64)
    value.primitive_normals = np.asarray([[0.0, 0.0, -1.0]] * 2, dtype=np.float64)
    value.primitive_scale1 = np.asarray([0.5, 0.5], dtype=np.float64)
    value.primitive_scale2 = np.asarray([0.5, 0.5], dtype=np.float64)
    value.primitive_opacity = np.ones((2,), dtype=np.float64)
    value.primitive_sidedness = np.asarray([2, 2], dtype=np.uint8)
    value.child_frames = np.eye(3, dtype=np.float64)[None]
    value.child_member_offsets = np.asarray([0, 1], dtype=np.int64)
    value.child_member_primitive_rows = np.asarray([1], dtype=np.int64)
    value.child_member_weights = np.ones((1,), dtype=np.float32)
    value.membership_offsets = np.asarray([0, 1], dtype=np.int64)
    value.membership_primitive_rows = np.asarray([1], dtype=np.int64)
    value.membership_weights = np.ones((1,), dtype=np.float32)
    value.member_slice = lambda row: slice(
        int(value.membership_offsets[row]), int(value.membership_offsets[row + 1])
    )
    return value


def test_sparse_feature_sampling_keeps_featureless_foreground_occluders():
    physical = _foreground_occluder_physical()
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    query = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (32, 1))
    result = score_pose_conditioned_sparse_primitives(
        np.eye(4)[None], query,
        np.asarray([1]), np.asarray([[1.0, 0.0]], dtype=np.float32), np.ones((1,)),
        physical, camera, token_height=4, token_width=8,
        primitives_per_child=1, maximum_splat_radius_tokens=1,
        score_semantics="fixed_grid", device="cpu",
    )
    assert result.rendered_coverage[0] > 0.0
    assert result.feature_coverage[0] == 0.0
    assert result.scores[0] == 0.0


def test_sparse_parent_padding_never_matches_unowned_primitive():
    physical = _foreground_occluder_physical()
    camera = ColmapCamera(0, 0, 8, 4, (8.0, 4.0, 2.0))
    query = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (32, 1))
    result = score_pose_conditioned_sparse_primitives(
        np.eye(4)[None], query,
        np.asarray([0]), np.asarray([[1.0, 0.0]], dtype=np.float32), np.ones((1,)),
        physical, camera, token_height=4, token_width=8,
        primitives_per_child=0, maximum_splat_radius_tokens=1,
        score_semantics="fixed_grid", allowed_parent_rows=np.asarray([[-1]]), device="cpu",
    )
    assert result.rendered_coverage[0] > 0.0
    assert result.feature_coverage[0] == 0.0
    assert result.scores[0] == 0.0


def test_stratified_primitive_sampling_is_rotation_equivariant_for_row_frames():
    points = np.asarray([
        [-1.4, -0.2, 0.0], [-0.9, 0.7, 0.0], [-0.1, -0.8, 0.0],
        [0.3, 0.4, 0.0], [1.0, -0.5, 0.0], [1.5, 0.9, 0.0],
    ], dtype=np.float64)

    def fixture(content_sha, centers, frame):
        count = centers.shape[0]
        return SimpleNamespace(
            content_sha256=content_sha,
            primitive_ids=np.arange(count, dtype=np.int64),
            primitive_centers=centers,
            primitive_opacity=np.linspace(0.7, 1.0, count),
            primitive_scale1=np.ones((count,), dtype=np.float64),
            primitive_scale2=np.ones((count,), dtype=np.float64),
            child_centers=np.zeros((1, 3), dtype=np.float64),
            child_frames=frame[None],
            child_extents=np.asarray([[2.0, 1.0, 0.1]], dtype=np.float64),
            child_parent_rows=np.asarray([0], dtype=np.int64),
            child_member_offsets=np.asarray([0, count], dtype=np.int64),
            child_member_primitive_rows=np.arange(count, dtype=np.int64),
            child_member_weights=np.ones((count,), dtype=np.float32),
        )

    angle = np.deg2rad(37.0)
    rotation = np.asarray([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    base_frame = np.eye(3, dtype=np.float64)
    base = fixture("sampling_base", points, base_frame)
    rotated = fixture("sampling_rotated", points @ rotation.T, base_frame @ rotation.T)
    confidence = np.linspace(1.0, 0.8, points.shape[0])
    first = _stratified_primitive_samples(
        base, np.arange(points.shape[0]), confidence, primitives_per_child=3,
    )
    second = _stratified_primitive_samples(
        rotated, np.arange(points.shape[0]), confidence, primitives_per_child=3,
    )
    np.testing.assert_array_equal(first, second)
