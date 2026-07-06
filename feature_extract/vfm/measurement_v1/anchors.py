from __future__ import annotations

from typing import Any

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.measurement_v1.types import SurfaceAnchor
from feature_extract.vfm.rendered_keypoint_matching import backproject_depth_to_world


def _depth_gradient(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float64)
    gy, gx = np.gradient(values)
    grad = np.sqrt(gx * gx + gy * gy)
    return np.where(np.isfinite(grad), grad, np.inf)


def _local_variance(values: np.ndarray, row: int, col: int, radius: int = 1) -> float:
    r0 = max(int(row) - radius, 0)
    r1 = min(int(row) + radius + 1, int(values.shape[0]))
    c0 = max(int(col) - radius, 0)
    c1 = min(int(col) + radius + 1, int(values.shape[1]))
    patch = np.asarray(values[r0:r1, c0:c1], dtype=np.float64)
    patch = patch[np.isfinite(patch)]
    if patch.size <= 1:
        return 0.0
    return float(np.var(patch))


def build_surface_anchors(
    depth: np.ndarray,
    alpha: np.ndarray,
    camera: ColmapCamera,
    render_pose_w2c: np.ndarray,
    *,
    token_grid_width: int,
    token_grid_height: int,
    subanchors_per_token: int = 2,
    max_anchors: int = 2048,
    min_alpha: float = 0.2,
    border_margin_px: int = 2,
    max_depth_gradient: float = 0.5,
) -> tuple[list[SurfaceAnchor], list[dict[str, Any]]]:
    """Build immutable 3D anchors from stable render depth/alpha pixels."""

    depth_map = np.asarray(depth, dtype=np.float64)
    alpha_map = np.asarray(alpha, dtype=np.float64)
    if depth_map.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    if alpha_map.shape != depth_map.shape:
        raise ValueError("alpha must have the same shape as depth")
    height, width = depth_map.shape
    grid_w = int(token_grid_width)
    grid_h = int(token_grid_height)
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("token grid dimensions must be positive")
    gradients = _depth_gradient(depth_map)
    rows: list[dict[str, Any]] = []
    accepted_by_token: dict[int, list[tuple[float, int, int, dict[str, Any]]]] = {}

    margin = max(int(border_margin_px), 0)
    for row in range(height):
        for col in range(width):
            xy = np.asarray([float(col) + 0.5, float(row) + 0.5], dtype=np.float64)
            token_col = min(int(float(col) * float(grid_w) / max(float(width), 1.0)), grid_w - 1)
            token_row = min(int(float(row) * float(grid_h) / max(float(height), 1.0)), grid_h - 1)
            token_index = int(token_row * grid_w + token_col)
            depth_value = float(depth_map[row, col])
            alpha_value = float(alpha_map[row, col])
            gradient = float(gradients[row, col])
            variance = _local_variance(depth_map, row, col)
            rejection = ""
            if row < margin or col < margin or row >= height - margin or col >= width - margin:
                rejection = "border"
            elif not np.isfinite(depth_value) or depth_value <= 0.0:
                rejection = "depth"
            elif not np.isfinite(alpha_value) or alpha_value < float(min_alpha):
                rejection = "alpha"
            elif not np.isfinite(gradient) or gradient > float(max_depth_gradient):
                rejection = "depth_gradient"
            quality = 0.0 if rejection else float(np.clip(alpha_value / (1.0 + gradient + variance), 0.0, 1.0))
            row_payload: dict[str, Any] = {
                "anchor_id": None,
                "token_index": token_index,
                "subanchor_index": None,
                "render_x": float(xy[0]),
                "render_y": float(xy[1]),
                "depth": depth_value,
                "alpha": alpha_value,
                "depth_gradient": gradient,
                "depth_variance": variance,
                "quality": quality,
                "rejection_reason": rejection,
            }
            rows.append(row_payload)
            if not rejection:
                accepted_by_token.setdefault(token_index, []).append((quality, row, col, row_payload))

    anchors: list[SurfaceAnchor] = []
    for token_index in sorted(accepted_by_token):
        candidates = sorted(accepted_by_token[token_index], key=lambda item: (-item[0], item[1], item[2]))
        for sub_idx, (_quality, row, col, row_payload) in enumerate(candidates[: max(int(subanchors_per_token), 0)]):
            if len(anchors) >= int(max_anchors):
                break
            xy = np.asarray([[float(col) + 0.5, float(row) + 0.5]], dtype=np.float64)
            xyz, valid = backproject_depth_to_world(
                xy,
                np.asarray([float(depth_map[row, col])], dtype=np.float64),
                camera,
                np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4),
            )
            if not bool(valid[0]):
                row_payload["rejection_reason"] = "backproject"
                continue
            variance = float(row_payload["depth_variance"])
            cov_world = np.eye(3, dtype=np.float64) * max(variance, 1e-6)
            anchor = SurfaceAnchor(
                anchor_id=len(anchors),
                token_index=int(token_index),
                subanchor_index=int(sub_idx),
                render_xy_px=xy.reshape(2),
                world_xyz=xyz[0],
                cov_world_3x3=cov_world,
                depth_m=float(depth_map[row, col]),
                alpha=float(alpha_map[row, col]),
                quality=float(row_payload["quality"]),
                normal_world=None,
                surface_id=None,
            )
            row_payload["anchor_id"] = int(anchor.anchor_id)
            row_payload["subanchor_index"] = int(anchor.subanchor_index)
            row_payload["X"] = float(anchor.world_xyz[0])
            row_payload["Y"] = float(anchor.world_xyz[1])
            row_payload["Z"] = float(anchor.world_xyz[2])
            anchors.append(anchor)
        if len(anchors) >= int(max_anchors):
            break
    return anchors, rows
