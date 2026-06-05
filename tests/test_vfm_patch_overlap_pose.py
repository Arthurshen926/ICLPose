from __future__ import annotations

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.patch_overlap_pose import (
    PatchOverlapConfig,
    PatchOverlapSupport,
    oracle_patch_overlap_matches,
    patch_overlap_energy_numpy,
    patch_overlap_pose_optimize,
)
from feature_extract.vfm.patch_to_3d_matching import TokenPatchBox
from feature_extract.vfm.query_to_3d_matching import LandmarkMapIndex, QueryTo3DMatch, pnp_pose_error


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=100, height=100, params=(80.0, 80.0, 50.0, 50.0))


def _xyz_from_xy(xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = (xy[:, 0] - 50.0) / 80.0 * z
    y = (xy[:, 1] - 50.0) / 80.0 * z
    return np.stack([x, y, z], axis=1).astype(np.float64)


def _support_and_matches() -> tuple[dict[int, PatchOverlapSupport], list[QueryTo3DMatch], dict[int, TokenPatchBox]]:
    centers = np.asarray(
        [
            [30.0, 30.0],
            [70.0, 30.0],
            [30.0, 70.0],
            [70.0, 70.0],
            [50.0, 50.0],
        ],
        dtype=np.float64,
    )
    xyz = _xyz_from_xy(centers, np.full((centers.shape[0],), 5.0, dtype=np.float64))
    supports = {}
    matches = []
    patches = {}
    for idx, (xy, point) in enumerate(zip(centers, xyz)):
        token_id = int(idx)
        track_id = int(100 + idx)
        offsets = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [-0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, -0.05, 0.0],
            ],
            dtype=np.float64,
        )
        supports[track_id] = PatchOverlapSupport(
            points_xyz=point.reshape(1, 3) + offsets,
            weights=np.ones((5,), dtype=np.float64),
        )
        patches[token_id] = TokenPatchBox(
            token_index=token_id,
            center=xy,
            x0=float(xy[0] - 10.0),
            y0=float(xy[1] - 10.0),
            x1=float(xy[0] + 10.0),
            y1=float(xy[1] + 10.0),
        )
        matches.append(
            QueryTo3DMatch(
                token_index=token_id,
                xy=xy,
                track_id=track_id,
                xyz=point,
                similarity=1.0,
                ratio=0.0,
                landmark_variance=0.0,
                source="oracle_patch_positive",
                similarity_margin=1.0,
            )
        )
    return supports, matches, patches


def test_patch_overlap_energy_prefers_gt_pose_over_bad_translation() -> None:
    supports, matches, patches = _support_and_matches()
    gt_pose = np.eye(4, dtype=np.float64)
    bad_pose = np.eye(4, dtype=np.float64)
    bad_pose[0, 3] = 0.8

    gt_energy = patch_overlap_energy_numpy(matches, patches, supports, _camera(), gt_pose)
    bad_energy = patch_overlap_energy_numpy(matches, patches, supports, _camera(), bad_pose)

    assert gt_energy < bad_energy * 0.25


def test_patch_overlap_optimizer_keeps_flat_containment_solution_when_already_inside_patch() -> None:
    supports, matches, patches = _support_and_matches()
    gt_pose = np.eye(4, dtype=np.float64)
    init_pose = np.eye(4, dtype=np.float64)
    init_pose[0, 3] = 0.25

    result = patch_overlap_pose_optimize(
        matches,
        patches,
        supports,
        _camera(),
        init_pose,
        config=PatchOverlapConfig(iterations=40, lr=0.08, max_anchors_per_patch=5, depth_weight=0.0),
    )

    init_error = pnp_pose_error(init_pose, gt_pose)
    final_error = pnp_pose_error(result.pose_w2c, gt_pose)
    assert result.success
    assert abs(result.final_energy - result.initial_energy) < 1e-10
    assert abs(final_error.translation_m - init_error.translation_m) < 1e-8


def test_patch_moment_regularized_optimizer_reduces_pose_error_from_perturbed_init() -> None:
    supports, matches, patches = _support_and_matches()
    gt_pose = np.eye(4, dtype=np.float64)
    init_pose = np.eye(4, dtype=np.float64)
    init_pose[0, 3] = 0.8

    result = patch_overlap_pose_optimize(
        matches,
        patches,
        supports,
        _camera(),
        init_pose,
        config=PatchOverlapConfig(
            iterations=120,
            lr=0.04,
            sigma=0.5,
            max_anchors_per_patch=5,
            depth_weight=1.0,
            center_weight=1.0,
            init_regularization_weight=1e-4,
            optimize_rotation=False,
        ),
    )

    init_error = pnp_pose_error(init_pose, gt_pose)
    final_error = pnp_pose_error(result.pose_w2c, gt_pose)
    assert result.success
    assert result.final_energy < result.initial_energy
    assert final_error.translation_m < init_error.translation_m * 0.4


def test_oracle_patch_overlap_matches_assigns_patch_from_support_not_centroid() -> None:
    camera = _camera()
    centroid = _xyz_from_xy(np.asarray([[50.0, 50.0]], dtype=np.float64), np.asarray([5.0], dtype=np.float64))[0]
    support_point = _xyz_from_xy(np.asarray([[70.0, 70.0]], dtype=np.float64), np.asarray([5.0], dtype=np.float64))[0]
    index = LandmarkMapIndex(
        track_ids=np.asarray([123], dtype=np.int64),
        xyz=centroid.reshape(1, 3),
        features=np.ones((1, 4), dtype=np.float32),
        mean_variances=np.zeros((1,), dtype=np.float32),
        observation_counts=np.ones((1,), dtype=np.int64),
        observation_image_ids=(("ref.png",),),
    )
    matches, patch_boxes = oracle_patch_overlap_matches(
        index,
        {123: PatchOverlapSupport(points_xyz=support_point.reshape(1, 3), weights=np.ones((1,), dtype=np.float64))},
        np.eye(4, dtype=np.float64),
        camera,
        token_width=5,
        token_height=5,
        min_support_containment=1.0,
    )

    assert len(matches) == 1
    assert patch_boxes[matches[0].token_index].contains(np.asarray([70.0, 70.0], dtype=np.float64))
    assert not patch_boxes[matches[0].token_index].contains(np.asarray([50.0, 50.0], dtype=np.float64))
