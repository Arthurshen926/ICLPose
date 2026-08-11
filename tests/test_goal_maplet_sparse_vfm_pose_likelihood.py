from types import SimpleNamespace

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization_goal_maplet.sparse_vfm_pose_likelihood import (
    score_pose_conditioned_sparse_primitives,
    score_pose_conditioned_sparse_vfm,
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
