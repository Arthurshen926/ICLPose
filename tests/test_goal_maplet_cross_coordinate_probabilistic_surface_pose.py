import numpy as np

from feature_extract.tools.vfm.refine_goal_maplet_cross_coordinate_probabilistic_surface_pose import (
    _candidate_pose,
    _mixture_statistics,
    _paired_query_hypotheses,
)


def _hypotheses(world, pixel, token, existence, variance, covariance=None):
    count = len(token)
    return {
        "world": np.asarray(world, np.float64).reshape(count, 3),
        "pixel": np.asarray(pixel, np.float64).reshape(count, 2),
        "token": np.asarray(token, np.int64),
        "plane": np.zeros(count, np.int64),
        "existence_probability": np.asarray(existence, np.float64),
        "query_variance_px2": np.asarray(variance, np.float64),
        "centroid_covariance_world_m2": (
            np.zeros((count, 3, 3), np.float64)
            if covariance is None else np.asarray(covariance, np.float64)
        ),
    }


def test_mixture_responsibilities_include_explicit_null_and_normalize():
    K = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
    hypothesis = _hypotheses(
        [[0, 0, 2], [0.2, 0, 2]], [[0, 0], [1, 0]], [3, 3], [0.8, 0.4], [1, 1],
    )
    _, responsibility, null, _, _ = _mixture_statistics(np.eye(4), hypothesis, K, 0.0)
    np.testing.assert_allclose(responsibility.sum() + null[0], 1.0, atol=1e-12)
    assert np.all(responsibility >= 0.0)
    assert 0.0 <= null[0] <= 1.0


def test_duplicate_identical_coordinate_arm_has_no_multiplicity_bonus():
    K = np.asarray([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 1.0]])
    one = _hypotheses([[0, 0, 2]], [[0, 0]], [3], [0.8], [1])
    two = _hypotheses(
        [[0, 0, 2], [0, 0, 2]], [[0, 0], [0, 0]], [3, 3], [0.8, 0.8], [1, 1],
    )
    one_likelihood, one_resp, one_null, _, _ = _mixture_statistics(np.eye(4), one, K, 0.0)
    two_likelihood, two_resp, two_null, _, _ = _mixture_statistics(np.eye(4), two, K, 0.0)
    np.testing.assert_allclose(one_likelihood, two_likelihood, atol=1e-12)
    np.testing.assert_allclose(one_resp.sum(), two_resp.sum(), atol=1e-12)
    np.testing.assert_allclose(one_null, two_null, atol=1e-12)


def test_centroid_covariance_not_footprint_scatter_controls_surface_variance():
    K = np.asarray([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
    base = _hypotheses([[0, 0, 2]], [[0, 0]], [0], [1.0], [1.0])
    _, _, _, _, sigma_base = _mixture_statistics(np.eye(4), base, K, 0.0)
    uncertain = dict(base)
    uncertain["centroid_covariance_world_m2"] = np.eye(3)[None] * 0.01
    _, _, _, _, sigma_uncertain = _mixture_statistics(np.eye(4), uncertain, K, 0.0)
    assert sigma_uncertain[0] > sigma_base[0]


def test_paired_query_hypotheses_keeps_each_match_once_per_coordinate_arm():
    base = {
        "correspondence_offsets": np.asarray([0, 2]),
        "world_points": np.asarray([[0, 0, 2], [1, 0, 2]], np.float64),
        "query_measurements_xy": np.asarray([[1, 2], [3, 4]], np.float64),
        "query_tokens": np.asarray([4, 5]),
        "provenance_region_plane_atlas_row": np.asarray([[0, 7, 0], [0, 8, 1]]),
        "correspondence_match_probability": np.asarray([0.8, 0.7]),
        "prototype_plane_pixel_purity": np.asarray([0.5, 1.0]),
        "query_measurement_variance_px2": np.asarray([1.0, 2.0]),
    }
    surface = {key: np.asarray(value).copy() for key, value in base.items()}
    surface["world_points"] += 0.1
    surface["prototype_centroid_covariance_world_m2"] = np.repeat(
        (np.eye(3) * 0.01)[None], 2, axis=0,
    )
    merged = _paired_query_hypotheses(base, surface, 0)
    assert merged["coordinate_arm"].tolist() == [0, 0, 1, 1]
    assert merged["token"].tolist() == [4, 5, 4, 5]
    np.testing.assert_allclose(merged["existence_probability"], [0.4, 0.7, 0.4, 0.7])
    np.testing.assert_allclose(merged["centroid_covariance_world_m2"][:2], 0.0)


def test_candidate_pose_uses_bounded_left_se3_parameterization():
    initial = np.eye(4); initial[:3, 3] = [1.0, 0.0, 0.0]
    output = _candidate_pose(initial, [0.0, 0.0, np.pi / 2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(output[:3, 3], [0.0, 1.0, 0.0], atol=1e-12)
