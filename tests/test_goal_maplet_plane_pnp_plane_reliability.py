from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_direct_plane_pnp_plane_reliability import (
    _damped_fraction,
    _mapping_depth_residuals,
    _plane_reliability_weights,
    _query_relative_depth_dispersion,
    _token_depth_dispersion,
)


def test_plane_reliability_is_bounded_and_decreases_with_residual() -> None:
    residual = np.asarray([0.0, 0.05, 0.10, 1.0], np.float64)
    weight = _plane_reliability_weights(residual)
    assert np.all(np.diff(weight) <= 0.0)
    assert float(np.min(weight)) == 0.5
    assert float(np.max(weight)) <= 1.5
    assert 0.8 <= float(np.median(weight)) <= 1.1


def test_plane_reliability_is_invariant_to_query_inventory_length_one() -> None:
    np.testing.assert_allclose(_plane_reliability_weights(np.asarray([0.03])), [1.0])


def test_tiered_damping_is_monotonic_with_primary_support() -> None:
    kwargs = dict(
        maximum_primary_ratio=0.4,
        refinement_fraction=0.25,
        very_low_primary_ratio=0.35,
        very_low_refinement_fraction=0.5,
    )
    assert _damped_fraction(0.30, **kwargs) == 0.5
    assert _damped_fraction(0.37, **kwargs) == 0.25
    assert _damped_fraction(0.40, **kwargs) == 0.0


def test_token_depth_dispersion_detects_a_mixed_surface_token() -> None:
    depth = np.asarray([
        [5.0, 5.0, 8.0, 12.0],
        [5.0, 5.0, 8.0, 12.0],
        [6.0, 6.0, 9.0, 9.0],
        [6.0, 6.0, 9.0, 9.0],
    ])
    median, mad = _token_depth_dispersion(depth, (2, 2))
    np.testing.assert_allclose(median, [5.0, 10.0, 6.0, 9.0])
    np.testing.assert_allclose(mad, [0.0, 2.0, 0.0, 0.0])


def test_mapping_depth_residual_combines_local_mad_and_depth_disagreement() -> None:
    K = np.asarray([[10.0, 0.0, 0.5], [0.0, 10.0, 0.5], [0.0, 0.0, 1.0]])
    world = np.asarray([[0.0, 0.0, 5.0], [0.5, 0.0, 5.0]])
    median = np.asarray([5.0, 7.0, 5.0, 5.0])
    mad = np.asarray([0.1, 0.2, 0.0, 0.0])
    residual = _mapping_depth_residuals(
        world, np.eye(4), K, 0.0, median, mad, (2, 2), (4, 4),
    )
    assert np.isclose(residual[0], 0.1)
    assert residual[1] > 2.0


def test_query_relative_depth_dispersion_is_scale_invariant_and_detects_edge() -> None:
    depth = np.asarray([[5.0, 5.0, 8.0, 12.0], [5.0, 5.0, 8.0, 12.0]])
    valid = np.ones_like(depth, bool)
    first = _query_relative_depth_dispersion(depth, valid, (1, 2))
    second = _query_relative_depth_dispersion(depth * 7.0, valid, (1, 2))
    np.testing.assert_allclose(first, second)
    assert np.isclose(first[0], 0.0)
    assert first[1] > 0.0
