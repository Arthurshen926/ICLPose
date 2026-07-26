from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.surface_feature_refinement import (
    _pose_distance,
    _view_log_weights,
)


def test_refinement_pose_distance_uses_camera_center() -> None:
    first = np.eye(4, dtype=np.float64)
    second = np.eye(4, dtype=np.float64)
    second[:3, 3] = np.asarray([-0.25, 0.0, 0.0])
    translation, rotation = _pose_distance(first, second)
    assert np.isclose(translation, 0.25)
    assert np.isclose(rotation, 0.0)


def test_refinement_view_mixture_excludes_query_and_normalizes() -> None:
    rows, log_weights = _view_log_weights(
        ("query", "front", "side"),
        support_view_directions=np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        support_quality=np.ones((3,), dtype=np.float32),
        query_view_direction=np.asarray([1.0, 0.0, 0.0]),
        excluded_image_id="query",
        view_mode_scores={},
        temperature=0.1,
    )
    np.testing.assert_array_equal(rows, np.asarray([1, 2]))
    assert np.isclose(np.exp(log_weights).sum(), 1.0)
    assert log_weights[0] > log_weights[1]
