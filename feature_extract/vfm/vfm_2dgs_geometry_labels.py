"""High-resolution geometry labels rendered from 2DGS surface elements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap
from feature_extract.vfm.vfm_depth_head import RADIO_TOKEN_DEPTH_INVALID, project_points_to_image


@dataclass(frozen=True)
class TwoDgsGeometryRenderConfig:
    sigma_scale: float = 2.0
    max_radius_px: float = 8.0
    min_opacity: float = 0.0
    depth_epsilon: float = 0.05
    min_support: int = 1
    max_depth_m: float | None = None
    face_camera_normals: bool = True

    def __post_init__(self) -> None:
        if float(self.sigma_scale) <= 0.0:
            raise ValueError("sigma_scale must be positive")
        if float(self.max_radius_px) <= 0.0:
            raise ValueError("max_radius_px must be positive")
        if float(self.min_opacity) < 0.0:
            raise ValueError("min_opacity must be non-negative")
        if float(self.depth_epsilon) < 0.0:
            raise ValueError("depth_epsilon must be non-negative")
        if int(self.min_support) <= 0:
            raise ValueError("min_support must be positive")
        if self.max_depth_m is not None and float(self.max_depth_m) <= 0.0:
            raise ValueError("max_depth_m must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "sigma_scale": float(self.sigma_scale),
            "max_radius_px": float(self.max_radius_px),
            "min_opacity": float(self.min_opacity),
            "depth_epsilon": float(self.depth_epsilon),
            "min_support": int(self.min_support),
            "max_depth_m": None if self.max_depth_m is None else float(self.max_depth_m),
            "face_camera_normals": bool(self.face_camera_normals),
        }


@dataclass(frozen=True)
class TwoDgsGeometryLabel:
    depth: np.ndarray
    normal_cam: np.ndarray
    alpha: np.ndarray
    valid: np.ndarray
    support_count: np.ndarray

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=np.float32)
        normal_cam = np.asarray(self.normal_cam, dtype=np.float32)
        alpha = np.asarray(self.alpha, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        support_count = np.asarray(self.support_count, dtype=np.int32)
        if depth.ndim != 2:
            raise ValueError("depth must have shape (H, W)")
        if normal_cam.shape != (*depth.shape, 3):
            raise ValueError("normal_cam must have shape (H, W, 3)")
        if alpha.shape != depth.shape or valid.shape != depth.shape or support_count.shape != depth.shape:
            raise ValueError("alpha, valid, and support_count must match depth shape")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "normal_cam", normal_cam)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "support_count", support_count)


def _camera_facing_normals(
    centers: np.ndarray,
    normals: np.ndarray,
    pose_w2c: np.ndarray,
    face_camera: bool,
) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    n_world = np.asarray(normals, dtype=np.float64)
    norm = np.linalg.norm(n_world, axis=1, keepdims=True)
    n_world = n_world / np.maximum(norm, 1e-8)
    if bool(face_camera):
        r_c2w = pose[:3, :3].T
        camera_center = -r_c2w @ pose[:3, 3]
        view_to_camera = camera_center.reshape(1, 3) - np.asarray(centers, dtype=np.float64)
        flip = np.sum(n_world * view_to_camera, axis=1) < 0.0
        n_world = n_world.copy()
        n_world[flip] *= -1.0
    n_cam = n_world @ pose[:3, :3].T
    n_cam = n_cam / np.maximum(np.linalg.norm(n_cam, axis=1, keepdims=True), 1e-8)
    return n_cam.astype(np.float32, copy=False)


def render_2dgs_geometry_label(
    surface: SurfaceElementMap,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    width: int,
    height: int,
    config: TwoDgsGeometryRenderConfig | None = None,
) -> TwoDgsGeometryLabel:
    """Render high-resolution 2DGS pseudo depth, normal, and alpha labels.

    This is a geometry-label renderer for supervision, not a photometric 2DGS
    renderer. It projects each surface element's center and tangent footprint,
    applies a nearest-depth z-buffer, then composites only the front layer
    within `depth_epsilon`.
    """

    cfg = config or TwoDgsGeometryRenderConfig()
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("render dimensions must be positive")
    depth = np.full((h, w), RADIO_TOKEN_DEPTH_INVALID, dtype=np.float32)
    normal_cam = np.zeros((h, w, 3), dtype=np.float32)
    alpha = np.zeros((h, w), dtype=np.float32)
    support_count = np.zeros((h, w), dtype=np.int32)
    if len(surface) == 0:
        return TwoDgsGeometryLabel(depth=depth, normal_cam=normal_cam, alpha=alpha, valid=np.zeros((h, w), dtype=bool), support_count=support_count)

    centers = np.asarray(surface.centers, dtype=np.float64)
    center_xy, center_z, center_visible = project_points_to_image(centers, pose_w2c, camera)
    opacity = np.asarray(surface.opacity, dtype=np.float32).reshape(-1)
    visible = center_visible & (opacity >= float(cfg.min_opacity))
    if cfg.max_depth_m is not None:
        visible &= center_z <= float(cfg.max_depth_m)
    rows = np.flatnonzero(visible)
    if rows.size == 0:
        return TwoDgsGeometryLabel(depth=depth, normal_cam=normal_cam, alpha=alpha, valid=np.zeros((h, w), dtype=bool), support_count=support_count)

    sx = float(w) / max(float(camera.width), 1.0)
    sy = float(h) / max(float(camera.height), 1.0)
    center_px = np.stack((center_xy[rows, 0] * sx, center_xy[rows, 1] * sy), axis=1).astype(np.float64)
    z_all = center_z[rows].astype(np.float64, copy=False)
    opacity_all = np.clip(opacity[rows].astype(np.float64, copy=False), 0.0, 1.0)

    tangent1_world = np.asarray(surface.tangent1[rows], dtype=np.float64) * np.asarray(
        surface.scale1[rows], dtype=np.float64
    ).reshape(-1, 1) * float(cfg.sigma_scale)
    tangent2_world = np.asarray(surface.tangent2[rows], dtype=np.float64) * np.asarray(
        surface.scale2[rows], dtype=np.float64
    ).reshape(-1, 1) * float(cfg.sigma_scale)
    p1_xy, p1_z, _ = project_points_to_image(centers[rows] + tangent1_world, pose_w2c, camera)
    p2_xy, p2_z, _ = project_points_to_image(centers[rows] + tangent2_world, pose_w2c, camera)
    v1 = np.stack((p1_xy[:, 0] * sx, p1_xy[:, 1] * sy), axis=1).astype(np.float64) - center_px
    v2 = np.stack((p2_xy[:, 0] * sx, p2_xy[:, 1] * sy), axis=1).astype(np.float64) - center_px
    v1[~np.isfinite(v1).all(axis=1) | ~(p1_z > 1e-8)] = 0.0
    v2[~np.isfinite(v2).all(axis=1) | ~(p2_z > 1e-8)] = 0.0

    a = v1[:, 0] * v1[:, 0] + v2[:, 0] * v2[:, 0] + 0.25
    b = v1[:, 0] * v1[:, 1] + v2[:, 0] * v2[:, 1]
    c = v1[:, 1] * v1[:, 1] + v2[:, 1] * v2[:, 1] + 0.25
    eig_max = 0.5 * ((a + c) + np.sqrt(np.maximum((a - c) * (a - c) + 4.0 * b * b, 0.0)))
    radius = np.sqrt(np.maximum(eig_max, 0.25))
    oversized = radius > float(cfg.max_radius_px)
    if np.any(oversized):
        shrink = float(cfg.max_radius_px) / np.maximum(radius[oversized], 1e-8)
        v1[oversized] *= shrink.reshape(-1, 1)
        v2[oversized] *= shrink.reshape(-1, 1)
        a = v1[:, 0] * v1[:, 0] + v2[:, 0] * v2[:, 0] + 0.25
        b = v1[:, 0] * v1[:, 1] + v2[:, 0] * v2[:, 1]
        c = v1[:, 1] * v1[:, 1] + v2[:, 1] * v2[:, 1] + 0.25
    det = np.maximum(a * c - b * b, 1e-12)
    inv00 = c / det
    inv01 = -b / det
    inv11 = a / det

    base_x = np.floor(np.clip(center_px[:, 0], 0.0, float(w - 1))).astype(np.int64)
    base_y = np.floor(np.clip(center_px[:, 1], 0.0, float(h - 1))).astype(np.int64)
    token_depth = np.full((h, w), np.inf, dtype=np.float64)
    flat_depth = token_depth.reshape(-1)
    radius_i = int(np.ceil(float(cfg.max_radius_px)))

    for dy in range(-radius_i, radius_i + 1):
        yy = base_y + int(dy)
        inside_y = (yy >= 0) & (yy < h)
        if not np.any(inside_y):
            continue
        for dx in range(-radius_i, radius_i + 1):
            xx = base_x + int(dx)
            inside = inside_y & (xx >= 0) & (xx < w)
            if not np.any(inside):
                continue
            delta_x = xx.astype(np.float64) + 0.5 - center_px[:, 0]
            delta_y = yy.astype(np.float64) + 0.5 - center_px[:, 1]
            mahal = inv00 * delta_x * delta_x + 2.0 * inv01 * delta_x * delta_y + inv11 * delta_y * delta_y
            covered = inside & (mahal <= 1.0)
            if np.any(covered):
                flat_index = yy[covered] * w + xx[covered]
                np.minimum.at(flat_depth, flat_index, z_all[covered])

    flat_support = support_count.reshape(-1)
    alpha_accum = np.zeros((h, w), dtype=np.float64)
    normal_accum = np.zeros((h * w, 3), dtype=np.float64)
    flat_alpha = alpha_accum.reshape(-1)
    normals_cam = _camera_facing_normals(centers[rows], surface.normals[rows], pose_w2c, bool(cfg.face_camera_normals))
    eps = float(cfg.depth_epsilon)

    for dy in range(-radius_i, radius_i + 1):
        yy = base_y + int(dy)
        inside_y = (yy >= 0) & (yy < h)
        if not np.any(inside_y):
            continue
        for dx in range(-radius_i, radius_i + 1):
            xx = base_x + int(dx)
            inside = inside_y & (xx >= 0) & (xx < w)
            if not np.any(inside):
                continue
            delta_x = xx.astype(np.float64) + 0.5 - center_px[:, 0]
            delta_y = yy.astype(np.float64) + 0.5 - center_px[:, 1]
            mahal = inv00 * delta_x * delta_x + 2.0 * inv01 * delta_x * delta_y + inv11 * delta_y * delta_y
            covered = inside & (mahal <= 1.0)
            if not np.any(covered):
                continue
            flat_index = yy[covered] * w + xx[covered]
            near_front = z_all[covered] <= flat_depth[flat_index] + eps
            if not np.any(near_front):
                continue
            chosen = flat_index[near_front]
            weights = opacity_all[covered][near_front] * np.exp(-0.5 * mahal[covered][near_front])
            source_rows = np.flatnonzero(covered)[near_front]
            np.add.at(flat_support, chosen, 1)
            np.add.at(flat_alpha, chosen, weights)
            np.add.at(normal_accum[:, 0], chosen, weights * normals_cam[source_rows, 0])
            np.add.at(normal_accum[:, 1], chosen, weights * normals_cam[source_rows, 1])
            np.add.at(normal_accum[:, 2], chosen, weights * normals_cam[source_rows, 2])

    valid = (support_count >= int(cfg.min_support)) & np.isfinite(token_depth)
    depth[valid] = token_depth[valid].astype(np.float32)
    alpha[valid] = np.clip(1.0 - np.exp(-alpha_accum[valid]), 0.0, 1.0).astype(np.float32)
    normal_flat = normal_cam.reshape(-1, 3)
    valid_flat = valid.reshape(-1)
    n = normal_accum[valid_flat]
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    normal_flat[valid_flat] = n.astype(np.float32, copy=False)
    return TwoDgsGeometryLabel(depth=depth, normal_cam=normal_cam, alpha=alpha, valid=valid, support_count=support_count)
