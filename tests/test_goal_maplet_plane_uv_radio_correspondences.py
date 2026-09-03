from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_plane_uv_radio_correspondences import (
    _core_seeded_metric_homography_filter,
    _metric_homography_filter,
    _top_distinct_hypotheses,
)


def test_metric_homography_keeps_an_exact_planar_mapping() -> None:
    token = np.asarray([0, 10, 64 * 10, 64 * 10 + 10, 64 * 20 + 20])
    xy = np.c_[token % 64, token // 64]
    uv = np.c_[0.5 * xy[:, 0] + 2.0, -0.25 * xy[:, 1] + 3.0]
    assert _metric_homography_filter(token, uv, threshold_m=0.1).all()


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
