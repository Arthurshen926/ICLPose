from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization_goal_maplet.observed_child_modes import (
    update_online_child_modes,
)


def test_online_child_modes_keep_bounded_distinct_views_and_merge_duplicates() -> None:
    centers = np.zeros((2, 2, 2), dtype=np.float32)
    mass = np.zeros((2, 2), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int32)
    update_online_child_modes(
        centers, mass, counts,
        np.asarray([0, 0, 0, 1]),
        np.asarray([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32),
        np.ones(4), minimum_angular_residual=0.1,
    )
    assert counts.tolist() == [2, 1]
    np.testing.assert_allclose(centers[0, 0], [1, 0])
    np.testing.assert_allclose(centers[0, 1], [0, 1])
    np.testing.assert_allclose(mass[0], [2, 1])


def test_online_child_modes_are_deterministic_for_fixed_mapping_order() -> None:
    def run():
        centers = np.zeros((1, 3, 3), dtype=np.float32)
        mass = np.zeros((1, 3), dtype=np.float64)
        counts = np.zeros(1, dtype=np.int32)
        update_online_child_modes(
            centers, mass, counts, np.zeros(5, dtype=np.int64),
            np.asarray([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 0]]),
            np.asarray([1, 2, 1, 1, 3]), minimum_angular_residual=0.05,
        )
        return centers, mass, counts
    first, second = run(), run()
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
