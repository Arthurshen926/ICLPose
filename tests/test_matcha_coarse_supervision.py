from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.matcha_coarse_supervision import (
    MatchaCoarseSupervisionConfig,
    build_matcha_coarse_supervision,
    cell_offset_labels,
    cell_offset_soft_labels,
)


def _camera(width: int = 16, height: int = 16) -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=width, height=height, params=(10.0, 10.0, width / 2, height / 2))


def _wide_camera(width: int = 256, height: int = 256) -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=width, height=height, params=(100.0, 100.0, width / 2, height / 2))


def test_cell_offset_labels_quantize_within_eight_by_eight_cell() -> None:
    labels, valid = cell_offset_labels(
        np.asarray([[4.0, 4.0], [15.9, 0.1], [-1.0, 3.0]], dtype=np.float64),
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
        offset_bins=8,
    )

    assert valid.tolist() == [True, True, False]
    assert labels.tolist()[:2] == [36, 7]


def test_cell_offset_soft_labels_keep_probability_distribution_near_patch_boundary() -> None:
    soft, valid = cell_offset_soft_labels(
        np.asarray([[7.9, 4.0], [4.0, 4.0]], dtype=np.float64),
        image_width=16,
        image_height=16,
        grid_width=2,
        grid_height=2,
        offset_bins=8,
        sigma_bins=0.75,
    )

    assert valid.tolist() == [True, True]
    assert soft.shape == (2, 65)
    assert np.allclose(soft.sum(axis=1), 1.0)
    assert np.all(soft[:, 64] == 0.0)
    assert np.count_nonzero(soft[0, :64] > 0.01) > 1
    assert int(np.argmax(soft[1, :64])) == 36


def test_matcha_coarse_supervision_identity_pose_roundtrip_and_dedup() -> None:
    camera = _camera()
    depth = np.full((16, 16), 4.0, dtype=np.float32)
    supervision = build_matcha_coarse_supervision(
        render_depth=depth,
        query_depth=depth,
        render_camera=camera,
        query_camera=camera,
        render_pose_w2c=np.eye(4, dtype=np.float64),
        query_pose_w2c=np.eye(4, dtype=np.float64),
        render_grid_hw=(2, 2),
        query_grid_hw=(2, 2),
        config=MatchaCoarseSupervisionConfig(roundtrip_threshold_px=0.5),
    )

    assert supervision.count == 4
    assert supervision.query_indices.tolist() == [0, 1, 2, 3]
    assert supervision.render_indices.tolist() == [0, 1, 2, 3]
    assert np.allclose(supervision.query_xy, supervision.render_xy)
    assert supervision.query_offset_labels.tolist() == [36, 36, 36, 36]
    assert supervision.render_offset_labels.tolist() == [36, 36, 36, 36]
    assert supervision.query_offset_soft_labels.shape == (4, 65)
    assert np.allclose(supervision.query_offset_soft_labels.sum(axis=1), 1.0)
    assert np.all(supervision.confidence_targets > 0.0)


def test_matcha_coarse_supervision_uses_render_seed_xy_for_non_center_offsets() -> None:
    camera = _camera()
    depth = np.full((16, 16), 4.0, dtype=np.float32)
    render_seed_xy = np.asarray([[3.0, 3.0], [13.0, 13.0]], dtype=np.float64)

    supervision = build_matcha_coarse_supervision(
        render_depth=depth,
        query_depth=depth,
        render_camera=camera,
        query_camera=camera,
        render_pose_w2c=np.eye(4, dtype=np.float64),
        query_pose_w2c=np.eye(4, dtype=np.float64),
        render_grid_hw=(2, 2),
        query_grid_hw=(2, 2),
        render_seed_xy=render_seed_xy,
        config=MatchaCoarseSupervisionConfig(roundtrip_threshold_px=0.5),
    )

    assert supervision.count == 2
    assert supervision.query_indices.tolist() == [0, 3]
    assert supervision.render_indices.tolist() == [0, 3]
    assert np.allclose(supervision.query_xy, render_seed_xy)
    assert np.allclose(supervision.render_xy, render_seed_xy)
    assert supervision.query_offset_labels.tolist() == [27, 45]
    assert supervision.render_offset_labels.tolist() == [27, 45]


def test_matcha_coarse_supervision_rejects_invalid_query_depth_roundtrip() -> None:
    camera = _camera()
    render_depth = np.full((16, 16), 4.0, dtype=np.float32)
    query_depth = np.zeros((16, 16), dtype=np.float32)

    supervision = build_matcha_coarse_supervision(
        render_depth=render_depth,
        query_depth=query_depth,
        render_camera=camera,
        query_camera=camera,
        render_pose_w2c=np.eye(4, dtype=np.float64),
        query_pose_w2c=np.eye(4, dtype=np.float64),
        render_grid_hw=(2, 2),
        query_grid_hw=(2, 2),
    )

    assert supervision.count == 0


def test_matcha_coarse_supervision_collects_visibility_failures_as_no_match() -> None:
    camera = _camera()
    depth = np.full((16, 16), 4.0, dtype=np.float32)
    render_alpha = np.ones((16, 16), dtype=np.float32)
    render_alpha[:8, :8] = 0.0

    supervision = build_matcha_coarse_supervision(
        render_depth=depth,
        query_depth=depth,
        render_alpha=render_alpha,
        query_alpha=np.ones((16, 16), dtype=np.float32),
        render_camera=camera,
        query_camera=camera,
        render_pose_w2c=np.eye(4, dtype=np.float64),
        query_pose_w2c=np.eye(4, dtype=np.float64),
        render_grid_hw=(2, 2),
        query_grid_hw=(2, 2),
        config=MatchaCoarseSupervisionConfig(
            roundtrip_threshold_px=0.5,
            alpha_threshold=0.5,
            collect_no_match=True,
        ),
    )

    assert supervision.count == 3
    assert supervision.no_match_count == 1
    assert supervision.no_match_query_indices.tolist() == [0]
    assert supervision.no_match_render_indices.tolist() == [0]
    assert supervision.no_match_reason_ids.tolist() == [1]


def test_matcha_coarse_supervision_pose_confidence_labels_positive_ignore_negative() -> None:
    camera = _wide_camera()
    render_depth = np.full((256, 256), 4.0, dtype=np.float32)
    query_depth = np.full((256, 256), 4.0, dtype=np.float32)
    render_seed_xy = np.asarray([[32.0, 128.0], [96.0, 128.0], [160.0, 128.0]], dtype=np.float64)
    query_sample_xy = np.asarray([[52, 128], [116, 128], [180, 128]], dtype=np.int64)
    query_depth[query_sample_xy[0, 1] - 1 : query_sample_xy[0, 1] + 2, query_sample_xy[0, 0] - 1 : query_sample_xy[0, 0] + 2] = 4.0
    query_depth[query_sample_xy[1, 1] - 1 : query_sample_xy[1, 1] + 2, query_sample_xy[1, 0] - 1 : query_sample_xy[1, 0] + 2] = 2.0
    query_depth[query_sample_xy[2, 1] - 1 : query_sample_xy[2, 1] + 2, query_sample_xy[2, 0] - 1 : query_sample_xy[2, 0] + 2] = 1.5
    query_pose = np.eye(4, dtype=np.float64)
    query_pose[0, 3] = 0.8

    supervision = build_matcha_coarse_supervision(
        render_depth=render_depth,
        query_depth=query_depth,
        render_camera=camera,
        query_camera=camera,
        render_pose_w2c=np.eye(4, dtype=np.float64),
        query_pose_w2c=query_pose,
        render_grid_hw=(1, 4),
        query_grid_hw=(1, 4),
        render_seed_xy=render_seed_xy,
        config=MatchaCoarseSupervisionConfig(
            roundtrip_threshold_px=40.0,
            pose_confidence_labels=True,
            pose_confidence_positive_threshold_px=8.0,
            pose_confidence_negative_threshold_px=24.0,
        ),
    )

    assert supervision.count == 3
    assert supervision.confidence_targets.tolist() == [1.0, 0.0, 0.0]
    assert supervision.confidence_ignore_mask.tolist() == [False, True, False]


def test_matcha_coarse_supervision_pose_usable_no_match_keeps_positive_confidence_target() -> None:
    camera = _wide_camera()
    render_depth = np.full((256, 256), 4.0, dtype=np.float32)
    query_depth = np.full((256, 256), 2.5, dtype=np.float32)
    query_pose = np.eye(4, dtype=np.float64)
    query_pose[0, 3] = 0.15

    supervision = build_matcha_coarse_supervision(
        render_depth=render_depth,
        query_depth=query_depth,
        render_camera=camera,
        query_camera=camera,
        render_pose_w2c=np.eye(4, dtype=np.float64),
        query_pose_w2c=query_pose,
        render_grid_hw=(1, 4),
        query_grid_hw=(1, 4),
        render_seed_xy=np.asarray([[96.0, 128.0]], dtype=np.float64),
        config=MatchaCoarseSupervisionConfig(
            roundtrip_threshold_px=1.5,
            collect_no_match=True,
            pose_confidence_labels=True,
            pose_confidence_positive_threshold_px=8.0,
            pose_confidence_negative_threshold_px=24.0,
        ),
    )

    assert supervision.count == 0
    assert supervision.no_match_count == 1
    assert np.allclose(supervision.no_match_roundtrip_errors_px, [2.25])
    assert supervision.no_match_confidence_targets.tolist() == [1.0]
    assert supervision.no_match_confidence_ignore_mask.tolist() == [False]
