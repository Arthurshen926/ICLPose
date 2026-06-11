from __future__ import annotations

from typing import Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    QueryTo3DMatch,
    camera_matrix_and_distortion,
    estimate_pose_pnp_fixed,
    pnp_pose_error,
)
from feature_extract.vfm.rendered_keypoint_matching import backproject_depth_to_world


def project_world_to_image(points_xyz: np.ndarray, pose_w2c: np.ndarray, camera: ColmapCamera) -> np.ndarray:
    """Project world points with a COLMAP camera and a world-to-camera pose."""

    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    matrix, distortion = camera_matrix_and_distortion(camera)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for projection diagnostics") from exc

    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(points, rvec, pose[:3, 3], matrix, distortion)
    return projected.reshape(-1, 2).astype(np.float64)


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values.astype(np.float64), float(percentile)))


def render_depth_roundtrip_stats(
    xy: np.ndarray,
    depth: np.ndarray,
    camera: ColmapCamera,
    pose_w2c: np.ndarray,
) -> dict[str, float | int]:
    """Backproject render pixels with depth, then project them back with the same pose."""

    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    world, valid = backproject_depth_to_world(coords, depth, camera, pose_w2c)
    projected = project_world_to_image(world[valid], pose_w2c, camera) if np.any(valid) else np.zeros((0, 2))
    errors = np.linalg.norm(projected - coords[valid], axis=1) if projected.size else np.zeros((0,), dtype=np.float64)
    count = int(coords.shape[0])
    valid_count = int(np.count_nonzero(valid))
    return {
        "count": count,
        "valid_count": valid_count,
        "valid_fraction": float(valid_count / count) if count else 0.0,
        "median_roundtrip_error_px": _safe_percentile(errors, 50.0),
        "p95_roundtrip_error_px": _safe_percentile(errors, 95.0),
        "max_roundtrip_error_px": float(np.max(errors)) if errors.size else float("nan"),
    }


def _matches_from_render_pixels(
    xy: np.ndarray,
    xyz: np.ndarray,
    valid: Sequence[bool],
) -> list[QueryTo3DMatch]:
    matches: list[QueryTo3DMatch] = []
    for index, is_valid in enumerate(np.asarray(valid, dtype=bool).reshape(-1)):
        if not bool(is_valid):
            continue
        matches.append(
            QueryTo3DMatch(
                token_index=index,
                xy=np.asarray(xy[index], dtype=np.float64),
                track_id=index,
                xyz=np.asarray(xyz[index], dtype=np.float64),
                similarity=1.0,
                ratio=1.0,
                landmark_variance=0.0,
                source="synthetic_render_depth",
            )
        )
    return matches


def synthetic_render_lock_stats(
    xy: np.ndarray,
    depth: np.ndarray,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray,
    min_inliers: int = 4,
    pnp_method: str = "ITERATIVE",
    refine_method: str = "lm",
) -> dict[str, float | int | bool]:
    """Solve PnP from query pixels paired to 3D points backprojected from the same render pose.

    This isolates a structural failure mode: if correspondence pixels are not corrected away from
    the render grid, depth-derived 3D points make a geometrically self-consistent PnP problem whose
    optimum is the render pose.
    """

    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xyz, valid = backproject_depth_to_world(coords, depth, camera, render_pose_w2c)
    matches = _matches_from_render_pixels(coords, xyz, valid)
    result = estimate_pose_pnp_fixed(
        matches,
        camera,
        min_inliers=int(min_inliers),
        pnp_method=pnp_method,
        refine_method=refine_method,
    )
    error = pnp_pose_error(result.pose_w2c, np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4))
    return {
        "match_count": len(matches),
        "pnp_success": bool(result.success),
        "pnp_inlier_count": int(result.inlier_count),
        "pnp_render_translation_delta_m": float(error.translation_m),
        "pnp_render_rotation_delta_deg": float(error.rotation_deg),
    }
