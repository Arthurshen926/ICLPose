from __future__ import annotations

import numpy as np

from feature_extract.vfm.matcha_keypoint_distillation import NON_KEYPOINT_LABEL, build_keypoint_label_map


def test_build_keypoint_label_map_keeps_highest_scored_alike_keypoint_on_collision() -> None:
    labels, stats = build_keypoint_label_map(
        np.asarray([[1.0, 1.0], [6.0, 6.0]], dtype=np.float32),
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
        scores=np.asarray([0.2, 0.9], dtype=np.float32),
    )

    assert labels[0, 0] == 54
    assert labels[0, 1] == NON_KEYPOINT_LABEL
    assert stats["valid_keypoint_count"] == 2
    assert stats["positive_count"] == 1
    assert stats["collision_count"] == 1


def test_build_keypoint_label_map_filters_non_finite_and_out_of_frame_keypoints() -> None:
    labels, stats = build_keypoint_label_map(
        np.asarray(
            [
                [4.0, 4.0],
                [np.nan, 1.0],
                [-1.0, 3.0],
                [16.0, 2.0],
            ],
            dtype=np.float32,
        ),
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
    )

    assert labels.tolist() == [[36, NON_KEYPOINT_LABEL], [NON_KEYPOINT_LABEL, NON_KEYPOINT_LABEL]]
    assert stats["keypoint_count"] == 4
    assert stats["valid_keypoint_count"] == 1
    assert stats["positive_count"] == 1


def test_build_keypoint_label_map_reports_valid_count_for_empty_alike_output() -> None:
    labels, stats = build_keypoint_label_map(
        np.zeros((0, 2), dtype=np.float32),
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
    )

    assert labels.tolist() == [[NON_KEYPOINT_LABEL, NON_KEYPOINT_LABEL], [NON_KEYPOINT_LABEL, NON_KEYPOINT_LABEL]]
    assert stats["keypoint_count"] == 0
    assert stats["valid_keypoint_count"] == 0
    assert stats["positive_count"] == 0
