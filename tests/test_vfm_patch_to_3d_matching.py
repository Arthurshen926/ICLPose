import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.map_lifting import SelectedTrackFeatureBank, TrackFeature, save_selected_track_bank_npz
from feature_extract.vfm.patch_to_3d_matching import (
    PatchTo3DMatchingConfig,
    build_patch_positive_sets,
    evaluate_patch_matches,
    match_query_patches_to_landmarks,
    token_patch_boxes,
)
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, token_grid_xy


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))


def _xyz_from_xy(xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = (xy[:, 0] - 50.0) / 80.0 * z
    y = (xy[:, 1] - 50.0) / 80.0 * z
    return np.stack([x, y, z], axis=1).astype(np.float64)


def test_patch_positive_sets_count_any_visible_landmark_inside_token_patch() -> None:
    token_xy = token_grid_xy(3, 3, image_width=100, image_height=100)
    center_token = 4
    projected_xy = token_xy[center_token] + np.asarray([12.0, 0.0], dtype=np.float64)
    xyz = _xyz_from_xy(projected_xy[None, :], np.asarray([5.0], dtype=np.float64))
    index = LandmarkMapIndex(
        track_ids=np.asarray([10], dtype=np.int64),
        xyz=xyz,
        features=np.ones((1, 4), dtype=np.float32),
        mean_variances=np.zeros((1,), dtype=np.float32),
        observation_counts=np.ones((1,), dtype=np.int64),
        observation_image_ids=(("ref.png",),),
    )

    positives = build_patch_positive_sets(
        index,
        pose_w2c=np.eye(4, dtype=np.float64),
        camera=_camera(),
        token_width=3,
        token_height=3,
    )

    assert positives.by_token[center_token].track_ids == {10}
    assert positives.by_token[center_token].count == 1


def test_patch_evaluator_counts_patch_correct_when_pixel_gt5_fails() -> None:
    query_map = np.zeros((2, 3, 3), dtype=np.float32)
    query_map[:, 1, 1] = np.asarray([1.0, 0.0], dtype=np.float32)
    center_xy = token_grid_xy(3, 3, image_width=100, image_height=100)[4]
    projected_xy = center_xy + np.asarray([12.0, 0.0], dtype=np.float64)
    index = LandmarkMapIndex(
        track_ids=np.asarray([10], dtype=np.int64),
        xyz=_xyz_from_xy(projected_xy[None, :], np.asarray([5.0], dtype=np.float64)),
        features=np.asarray([[1.0, 0.0]], dtype=np.float32),
        mean_variances=np.zeros((1,), dtype=np.float32),
        observation_counts=np.ones((1,), dtype=np.int64),
        observation_image_ids=(("ref.png",),),
    )
    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(top_k=1, min_similarity=0.5, match_mode="nn"),
        image_width=100,
        image_height=100,
    )
    positives = build_patch_positive_sets(index, np.eye(4, dtype=np.float64), _camera(), 3, 3)

    stats = evaluate_patch_matches(matches, positives, np.eye(4, dtype=np.float64), _camera(), stride_px=49.5)

    assert stats["patch_at_1"] == 1.0
    assert stats["gt_precision_5px"] == 0.0
    assert stats["gt_precision_stride"] == 1.0


def test_soft_mutual_topk_keeps_patch_level_many_to_one_candidates() -> None:
    query_map = np.zeros((2, 1, 2), dtype=np.float32)
    query_map[:, 0, 0] = np.asarray([1.0, 0.0], dtype=np.float32)
    query_map[:, 0, 1] = np.asarray([0.99, 0.01], dtype=np.float32)
    index = LandmarkMapIndex(
        track_ids=np.asarray([1, 2], dtype=np.int64),
        xyz=np.zeros((2, 3), dtype=np.float64),
        features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        mean_variances=np.zeros((2,), dtype=np.float32),
        observation_counts=np.ones((2,), dtype=np.int64),
        observation_image_ids=(("a",), ("a",)),
    )

    matches = match_query_patches_to_landmarks(
        query_map,
        index,
        PatchTo3DMatchingConfig(top_k=1, mutual_top_k=2, min_similarity=0.5, match_mode="soft_mutual"),
        image_width=100,
        image_height=10,
    )

    assert [(match.token_index, match.track_id) for match in matches] == [(0, 1), (1, 1)]


def test_token_patch_boxes_cover_stride_sized_regions() -> None:
    boxes = token_patch_boxes(token_width=3, token_height=3, image_width=100, image_height=100)

    center = boxes[4]
    assert center.center.tolist() == [49.5, 49.5]
    assert center.x0 < 49.5 < center.x1
    assert center.y0 < 49.5 < center.y1
