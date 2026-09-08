from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_plane_uv_radio_correspondences import (
    _atlas_local_cell_feature_lookup,
    _clip_chart_coordinate_to_original_cell,
    _core_seeded_metric_homography_filter,
    _metric_homography_filter,
    _metric_homography_filter_with_projection,
    _region_token_measurements,
    _runtime_local_radio_context,
    _top_distinct_hypotheses,
)
from feature_extract.tools.vfm.train_goal_maplet_mapping_subtoken_head import MappingSubtokenHead


def test_metric_homography_keeps_an_exact_planar_mapping() -> None:
    token = np.asarray([0, 10, 64 * 10, 64 * 10 + 10, 64 * 20 + 20])
    xy = np.c_[token % 64, token // 64]
    uv = np.c_[0.5 * xy[:, 0] + 2.0, -0.25 * xy[:, 1] + 3.0]
    assert _metric_homography_filter(token, uv, threshold_m=0.1).all()


def test_metric_homography_returns_the_same_continuous_uv_projection() -> None:
    token = np.asarray([0, 10, 64 * 10, 64 * 10 + 10, 64 * 20 + 20])
    xy = np.c_[token % 64, token // 64]
    uv = np.c_[0.5 * xy[:, 0] + 2.0, -0.25 * xy[:, 1] + 3.0]
    keep, projected, valid = _metric_homography_filter_with_projection(
        token, uv, threshold_m=0.1,
    )
    assert keep.all()
    assert valid.all()
    np.testing.assert_allclose(projected, uv, atol=1e-12)


def test_core_seeded_homography_does_not_let_boundary_outliers_set_warp() -> None:
    token = np.asarray([0, 10, 640, 650, 20, 30, 660, 670])
    xy = np.c_[token % 64, token // 64].astype(np.float64)
    uv = xy.copy()
    uv[4:] += np.asarray([20.0, -15.0])
    fraction = np.asarray([1.0] * 4 + [0.5] * 4)
    keep = _core_seeded_metric_homography_filter(
        token, uv, fraction, threshold_m=0.1,
    )
    np.testing.assert_array_equal(keep, [True, True, True, True, False, False, False, False])


def test_top_hypotheses_are_stable_and_distinct() -> None:
    token = np.asarray([4, 4, 4, 4, 9])
    score = np.asarray([0.8, 0.9, 0.7, 0.95, 1.0])
    plane = np.asarray([1, 1, 2, 3, 1])
    texel = np.asarray([10, 10, 20, 30, 40])
    chosen = _top_distinct_hypotheses(token, score, plane, texel, maximum_per_token=3)
    np.testing.assert_array_equal(chosen, [3, 1, 2, 4])


def test_region_measurement_uses_only_observed_same_plane_pixels() -> None:
    labels = np.full((144, 256), -1, np.int32)
    labels[:4, :2] = 7
    labels[:4, 4:8] = 7
    token, visible, measurement = _region_token_measurements(labels, 7)
    np.testing.assert_array_equal(token, [0, 1])
    np.testing.assert_allclose(visible, [0.5, 1.0])
    np.testing.assert_allclose(measurement, [[0.5, 1.5], [5.5, 1.5]])


def test_chart_offset_is_clipped_to_original_half_open_metric_cell() -> None:
    prototype = np.asarray([[0.49, -0.01], [-0.51, 1.01]])
    offset = np.asarray([[0.20, 0.20], [-1.00, -0.20]])
    result = _clip_chart_coordinate_to_original_cell(prototype, offset, 0.5)
    lower = np.floor(prototype / 0.5) * 0.5
    assert np.all(result >= lower)
    assert np.all(result < lower + 0.5)
    assert result[0, 0] == np.nextafter(0.5, 0.0)
    assert result[0, 1] == np.nextafter(0.0, -0.5)
    np.testing.assert_allclose(result[1], [-1.0, 1.0])


def test_mapping_pair_head_measurement_is_hypothesis_specific() -> None:
    import torch
    torch.manual_seed(4)
    head = MappingSubtokenHead(64, 16)
    query = torch.nn.functional.normalize(torch.randn(2, 64), dim=1)
    mapping = torch.nn.functional.normalize(torch.randn(2, 64), dim=1)
    mean, variance, probability = head(query, mapping, torch.asarray([10, 10]))
    assert not torch.equal(mean[0], mean[1])
    assert torch.all(variance > 0)
    assert torch.all((torch.sigmoid(probability) >= 0) & (torch.sigmoid(probability) <= 1))


def test_runtime_local_correlation_uses_anonymous_cell_mean_and_region_mask() -> None:
    feature = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], np.float32,
    )
    lookup = _atlas_local_cell_feature_lookup(
        np.asarray([0, 3]),
        np.asarray([[0.10, 0.10], [0.20, 0.20], [0.60, 0.10]]),
        np.asarray([0, 0, 1]), feature, 0.5,
    )
    query = np.zeros((36 * 64, 2), np.float32)
    query[65] = [1.0, 0.0]; query[66] = [0.0, 1.0]
    context = _runtime_local_radio_context(
        query, np.asarray([65]), np.asarray([7]), {7: np.asarray([65, 66])},
        np.asarray([[1.0, 0.0]], np.float32), np.asarray([[0.10, 0.10]]),
        np.asarray([0]), lookup, 0.5,
    )
    assert context.shape == (1, 99)
    assert context[0, 4 * 9 + 4] == 1.0
    assert context[0, 5 * 9 + 5] == 1.0
    # A geometrically adjacent query token outside the selected plane region is masked.
    assert context[0, 3 * 9 + 4] == 0.0
