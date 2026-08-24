from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from feature_extract.vfm.localization_goal_maplet.parent_support_layout_guide import (
    ParentLayoutCamera,
    SCORE_SEMANTICS,
    TOKEN_FOOTPRINT_PHASE,
    adapt_ranked_factor_pairs_for_exact_scorer,
    projected_image_bounds_to_radio_token_footprint,
    score_parent_support_layout_guide,
)
from feature_extract.vfm.localization_goal_maplet.physical_map import (
    DOUBLE_SIDED,
    SINGLE_SIDED,
)
from feature_extract.vfm.localization_goal_maplet.pure_retrieval import (
    all_radio_token_coordinates,
)


def _physical(*, normal=(0.0, 0.0, -1.0), sidedness=SINGLE_SIDED):
    return SimpleNamespace(
        maplet_ids=np.asarray([10], dtype=np.int64),
        maplet_centers=np.asarray([[0.0, 0.0, 4.0]], dtype=np.float64),
        maplet_frames=np.asarray([np.eye(3)], dtype=np.float64),
        maplet_extents=np.asarray([[0.45, 0.45, 0.05]], dtype=np.float64),
        maplet_normals=np.asarray([normal], dtype=np.float64),
        maplet_sidedness=np.asarray([sidedness], dtype=np.uint8),
        content_sha256="physical",
    )


def _retrieval():
    xy = all_radio_token_coordinates(4, 4)
    ids = np.full((16, 1), -1, dtype=np.int64)
    probability = np.zeros((16, 1), dtype=np.float32)
    central = (
        (xy[:, 0] >= 1) & (xy[:, 0] < 3)
        & (xy[:, 1] >= 1) & (xy[:, 1] < 3)
    )
    ids[central, 0] = 10
    probability[central, 0] = 1.0
    return SimpleNamespace(
        token_xy=xy,
        token_parent_ids=ids,
        token_parent_probabilities=probability,
        metadata={"token_height": 4, "token_width": 4},
        physical_map_sha256="physical",
    )


def _factors():
    positions = np.asarray([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    identity = np.eye(3, dtype=np.float64)
    behind = np.diag([-1.0, 1.0, -1.0])
    return (
        positions,
        np.stack([identity, behind]),
        np.asarray([True, True]),
        np.asarray([1, 2], dtype=np.int16),
    )


def test_parent_layout_guide_prefers_matching_footprint_and_is_batch_stable():
    camera = ParentLayoutCamera(1, 4, 4, (4.0, 4.0, 2.0, 2.0))
    factors = _factors()
    small = score_parent_support_layout_guide(
        *factors, _retrieval(), _physical(), camera,
        maximum_query_parents=1, topk=4, candidate_batch_size=1,
    )
    large = score_parent_support_layout_guide(
        *factors, _retrieval(), _physical(), camera,
        maximum_query_parents=1, topk=4, candidate_batch_size=32,
    )
    assert small.score_semantics == SCORE_SEMANTICS
    assert small.total_factor_pair_count == 4
    assert small.top_position_factor_indices[0] == 0
    assert small.top_orientation_factor_indices[0] == 0
    assert small.top_scores[0] == pytest.approx(1.0)
    assert small.top_visible_parent_counts[0] == 1
    assert small.top_projected_token_footprint_mass[0] == 4
    for name in small.__dataclass_fields__:
        left, right = getattr(small, name), getattr(large, name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    pairs = np.stack([
        small.top_position_factor_indices,
        small.top_orientation_factor_indices,
    ], axis=1)
    assert np.unique(pairs, axis=0).shape[0] == 4


def test_parent_layout_guide_explicitly_gates_depth_and_unsigned_incidence():
    camera = ParentLayoutCamera(2, 4, 4, (4.0, 2.0, 2.0, 0.0))
    factors = _factors()
    reversed_normal = score_parent_support_layout_guide(
        *factors, _retrieval(), _physical(normal=(0.0, 0.0, 1.0)), camera,
        maximum_query_parents=1, topk=4,
    )
    # Reversing an unsigned normal cannot change a held-route score.
    double = score_parent_support_layout_guide(
        *factors, _retrieval(),
        _physical(normal=(0.0, 0.0, 1.0), sidedness=DOUBLE_SIDED), camera,
        maximum_query_parents=1, topk=4,
    )
    np.testing.assert_array_equal(reversed_normal.top_scores, double.top_scores)
    assert double.top_scores[0] == pytest.approx(1.0)
    # The 180-degree orientation has the parent behind the camera and cannot
    # acquire any footprint even though a parent identity is present.
    behind_rows = double.top_orientation_factor_indices == 1
    assert np.all(double.top_positive_depth_parent_counts[behind_rows] == 0)
    assert np.all(double.top_visible_parent_counts[behind_rows] == 0)

    grazing = score_parent_support_layout_guide(
        *factors, _retrieval(), _physical(normal=(1.0, 0.0, 0.0)), camera,
        maximum_query_parents=1, topk=4,
    )
    assert np.all(grazing.top_scores == 0.0)
    origin_rows = grazing.top_position_factor_indices == 0
    assert np.all(grazing.top_front_facing_parent_counts[origin_rows] == 0)


def test_projected_footprint_uses_existing_pixel_edge_token_phase_without_half_shift():
    assert "edge_aligned" in TOKEN_FOOTPRINT_PHASE
    low = np.asarray([[0.0, 0.0], [16.0, 8.0], [63.9, 31.9]])
    high = np.asarray([[16.0, 8.0], [32.0, 16.0], [64.0, 32.0]])
    x0, y0, x1, y1 = projected_image_bounds_to_radio_token_footprint(
        low, high, image_width=64, image_height=32,
        token_width=4, token_height=4,
    )
    np.testing.assert_array_equal(x0, [0, 1, 3])
    np.testing.assert_array_equal(y0, [0, 1, 3])
    np.testing.assert_array_equal(x1, [1, 2, 4])
    np.testing.assert_array_equal(y1, [1, 2, 4])


def test_parent_layout_guide_query_denominator_keeps_unselected_mass():
    retrieval = _retrieval()
    ids = np.concatenate([
        retrieval.token_parent_ids,
        np.full((16, 1), 20, dtype=np.int64),
    ], axis=1)
    probability = np.concatenate([
        retrieval.token_parent_probabilities,
        np.full((16, 1), 0.25, dtype=np.float32),
    ], axis=1)
    physical = _physical()
    physical.maplet_ids = np.asarray([10, 20], dtype=np.int64)
    physical.maplet_centers = np.asarray([[0.0, 0.0, 4.0], [100.0, 0.0, 4.0]])
    physical.maplet_frames = np.asarray([np.eye(3), np.eye(3)])
    physical.maplet_extents = np.asarray([[0.45, 0.45, 0.05]] * 2)
    physical.maplet_normals = np.asarray([[0.0, 0.0, -1.0]] * 2)
    physical.maplet_sidedness = np.asarray([SINGLE_SIDED] * 2, dtype=np.uint8)
    retrieval.token_parent_ids = ids
    retrieval.token_parent_probabilities = probability
    result = score_parent_support_layout_guide(
        *_factors(), retrieval, physical,
        ParentLayoutCamera(0, 4, 4, (4.0, 2.0, 2.0)),
        maximum_query_parents=1, topk=1,
    )
    assert result.selected_query_parent_probability_mass_total == pytest.approx(4.0)
    assert result.complete_query_parent_probability_mass == pytest.approx(8.0)
    # Matching selected mass is not renormalized back to one after dropping
    # the query-only tail: 4 / sqrt(8*4) = 1/sqrt(2).
    assert result.top_scores[0] == pytest.approx(1.0 / np.sqrt(2.0))


def test_parent_layout_guide_rejects_unknown_or_duplicate_parent_identity():
    camera = ParentLayoutCamera(0, 4, 4, (4.0, 2.0, 2.0))
    unknown = _retrieval()
    unknown.token_parent_ids[5, 0] = 999
    with pytest.raises(ValueError, match="unknown"):
        score_parent_support_layout_guide(
            *_factors(), unknown, _physical(), camera,
        )
    duplicate = _retrieval()
    duplicate.token_parent_ids = np.repeat(duplicate.token_parent_ids, 2, axis=1)
    duplicate.token_parent_probabilities = np.repeat(
        duplicate.token_parent_probabilities * 0.5, 2, axis=1,
    )
    with pytest.raises(ValueError, match="repeats"):
        score_parent_support_layout_guide(
            *_factors(), duplicate, _physical(), camera,
        )


def test_exact_scorer_adapter_materializes_only_ranked_factor_pairs_with_provenance():
    position, rotation, valid, source = _factors()
    adapted = adapt_ranked_factor_pairs_for_exact_scorer(
        position, rotation, valid, source,
        np.asarray([1, 0, 1]), np.asarray([0, 1, 1]),
        np.asarray([1, 2, 2]), np.asarray([0.9, 0.8, 0.7]),
        maximum_pairs=2,
    )
    assert adapted["candidate_poses_w2c"].shape == (2, 4, 4)
    assert adapted["position_factor_indices"].tolist() == [1, 0]
    assert adapted["orientation_factor_indices"].tolist() == [0, 1]
    assert adapted["position_seed_indices"].tolist() == [0, 0]
    assert adapted["position_offset_indices"].tolist() == [1, 0]
    np.testing.assert_array_equal(
        adapted["parent_layout_guide_scores"], [0.9, 0.8],
    )
    # Pose 0 has identity R and center [2,0,0], hence t=-C.
    np.testing.assert_allclose(
        adapted["candidate_poses_w2c"][0, :3, 3], [-2.0, 0.0, 0.0],
    )
    centers = -np.swapaxes(
        adapted["candidate_poses_w2c"][:, :3, :3], 1, 2,
    ) @ adapted["candidate_poses_w2c"][:, :3, 3, None]
    np.testing.assert_allclose(
        centers[..., 0], position.reshape(-1, 3)[[1, 0]], atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="provenance"):
        adapt_ranked_factor_pairs_for_exact_scorer(
            position, rotation, valid, source,
            np.asarray([1]), np.asarray([0]), np.asarray([2]), np.asarray([0.9]),
            maximum_pairs=1,
        )
