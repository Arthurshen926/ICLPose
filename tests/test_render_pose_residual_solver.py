from __future__ import annotations

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import camera_center_from_pose_w2c
from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import QueryTo3DMatch, pnp_pose_error
from feature_extract.vfm.render_pose_residual_solver import solve_render_pose_delta_from_matches


def _camera() -> ColmapCamera:
    return ColmapCamera(camera_id=1, model_id=1, width=640, height=480, params=(500.0, 500.0, 320.0, 240.0))


def _project(points_xyz: np.ndarray, pose_w2c: np.ndarray, camera: ColmapCamera) -> np.ndarray:
    fx, fy, cx, cy = camera.params[:4]
    points_cam = (pose_w2c[:3, :3] @ points_xyz.T + pose_w2c[:3, 3:4]).T
    return np.column_stack(
        [
            fx * points_cam[:, 0] / points_cam[:, 2] + cx,
            fy * points_cam[:, 1] / points_cam[:, 2] + cy,
        ]
    ).astype(np.float64)


def _pose_w2c_from_center(center: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = -np.asarray(center, dtype=np.float64).reshape(3)
    return pose


def _synthetic_matches(pose_w2c: np.ndarray, camera: ColmapCamera) -> list[QueryTo3DMatch]:
    xyz = np.asarray(
        [
            [-1.2, -0.8, 6.0],
            [1.1, -0.7, 6.4],
            [-1.0, 0.9, 7.0],
            [1.3, 1.0, 7.5],
            [-0.4, -1.1, 8.2],
            [0.6, -1.2, 8.8],
            [-1.4, 0.2, 9.1],
            [1.5, -0.1, 9.7],
            [-0.3, 1.4, 10.3],
            [0.9, 1.2, 10.9],
        ],
        dtype=np.float64,
    )
    xys = _project(xyz, pose_w2c, camera)
    return [
        QueryTo3DMatch(
            token_index=idx,
            xy=xy,
            track_id=idx + 1,
            xyz=point,
            similarity=0.9,
            ratio=0.0,
            landmark_variance=0.0,
            pnp_soft_score=0.8,
        )
        for idx, (xy, point) in enumerate(zip(xys, xyz))
    ]


def test_correct_correspondences_with_offset_render_pose_recover_gt_translation() -> None:
    camera = _camera()
    gt_pose_w2c = _pose_w2c_from_center(np.asarray([0.8, -0.4, 0.25], dtype=np.float64))
    render_pose_w2c = _pose_w2c_from_center(np.asarray([1.05, -0.4, 0.25], dtype=np.float64))

    solved_pose = solve_render_pose_delta_from_matches(
        _synthetic_matches(gt_pose_w2c, camera),
        render_pose_w2c,
        camera,
        max_iterations=20,
    )

    error = pnp_pose_error(solved_pose, gt_pose_w2c)
    assert error.translation_m < 1e-4


def test_solver_returns_global_query_pose_not_local_residual_pose() -> None:
    camera = _camera()
    gt_center = np.asarray([0.8, -0.4, 0.25], dtype=np.float64)
    render_center = gt_center + np.asarray([0.25, 0.0, 0.0], dtype=np.float64)
    gt_pose_w2c = _pose_w2c_from_center(gt_center)
    render_pose_w2c = _pose_w2c_from_center(render_center)

    solved_pose = solve_render_pose_delta_from_matches(
        _synthetic_matches(gt_pose_w2c, camera),
        render_pose_w2c,
        camera,
        max_iterations=20,
    )

    solved_center = camera_center_from_pose_w2c(solved_pose)
    np.testing.assert_allclose(solved_center, gt_center, atol=1e-4)
    assert np.linalg.norm(solved_center - (gt_center - render_center)) > 0.5
