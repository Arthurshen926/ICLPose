import numpy as np
import pytest

from feature_extract.tools.vfm.blend_goal_maplet_cross_coordinate_correspondences import (
    _blend,
    _moment_matched_scalar_variance,
    _moment_matched_world_covariance,
    _positive_part_surface_gain,
)


def test_scalar_moment_match_preserves_endpoints():
    first = np.asarray([[0.0, 0.0], [1.0, 2.0]])
    second = np.asarray([[2.0, 0.0], [3.0, 4.0]])
    a = np.asarray([1.0, 2.0])
    b = np.asarray([3.0, 4.0])
    np.testing.assert_allclose(_moment_matched_scalar_variance(first, second, a, b, 0.0), a)
    np.testing.assert_allclose(_moment_matched_scalar_variance(first, second, a, b, 1.0), b)


def test_scalar_moment_match_retains_coordinate_model_disagreement():
    value = _moment_matched_scalar_variance(
        np.asarray([[0.0, 0.0]]), np.asarray([[2.0, 0.0]]),
        np.asarray([1.0]), np.asarray([1.0]), 0.25,
    )
    # Within-component variance 1 plus alpha(1-alpha)*trace(delta deltaT)/2.
    np.testing.assert_allclose(value, [1.375])


def test_world_moment_match_is_psd_and_includes_between_model_axis():
    output = _moment_matched_world_covariance(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[2.0, 0.0, 0.0]]),
        np.asarray([np.diag([0.0, 1.0, 0.0])]),
        0.25,
    )
    np.testing.assert_allclose(
        output[0], np.diag([0.75, 0.25, 1e-8]), atol=1e-12,
    )
    assert np.min(np.linalg.eigvalsh(output[0])) >= 0.0


def test_moment_match_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="fraction"):
        _moment_matched_scalar_variance(
            np.zeros((1, 2)), np.zeros((1, 2)), np.ones(1), np.ones(1), 1.1,
        )


def test_blend_rejects_unregistered_scope_before_accessing_arrays():
    with pytest.raises(ValueError, match="scope"):
        _blend({}, {}, {}, fraction=0.25, coordinate_blend_scope="query_only")


def test_surface_gain_removes_one_two_dimensional_rms_noise_radius():
    gain = _positive_part_surface_gain(
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.asarray([[2.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        np.asarray([np.diag([0.5, 0.5, 0.0]), np.diag([0.5, 0.5, 0.0])]),
    )
    np.testing.assert_allclose(gain, [0.5, 0.0])
