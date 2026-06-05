"""Lightweight token-grid depth heads on frozen RADIO/VFM features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap


RADIO_TOKEN_DEPTH_INVALID = np.float32(0.0)


@dataclass(frozen=True)
class TokenDepthRasterConfig:
    depth_epsilon: float = 0.05
    min_points_per_token: int = 1
    max_depth_m: float | None = None

    def __post_init__(self) -> None:
        if float(self.depth_epsilon) < 0.0:
            raise ValueError("depth_epsilon must be non-negative")
        if int(self.min_points_per_token) <= 0:
            raise ValueError("min_points_per_token must be positive")
        if self.max_depth_m is not None and float(self.max_depth_m) <= 0.0:
            raise ValueError("max_depth_m must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "depth_epsilon": float(self.depth_epsilon),
            "min_points_per_token": int(self.min_points_per_token),
            "max_depth_m": None if self.max_depth_m is None else float(self.max_depth_m),
        }


@dataclass(frozen=True)
class TokenDepthLabel:
    depth: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    support_count: np.ndarray
    center_valid: np.ndarray
    filled_by_splat: np.ndarray

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        support_count = np.asarray(self.support_count, dtype=np.int32)
        center_valid = np.asarray(self.center_valid, dtype=bool)
        filled_by_splat = np.asarray(self.filled_by_splat, dtype=bool)
        shape = depth.shape
        for name, value in (
            ("valid", valid),
            ("confidence", confidence),
            ("support_count", support_count),
            ("center_valid", center_valid),
            ("filled_by_splat", filled_by_splat),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must match depth shape")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "support_count", support_count)
        object.__setattr__(self, "center_valid", center_valid)
        object.__setattr__(self, "filled_by_splat", filled_by_splat)


class RadioTokenDepthHead(nn.Module):
    """Small 1x1-conv depth adaptor for frozen RADIO token maps.

    The head predicts metric depth in meters by regressing log-depth and
    exponentiating it. Keeping the model token-local is intentional: this tests
    whether existing VFM tokens contain useful geometry without adding a large
    context network.
    """

    def __init__(self, in_channels: int = 1280, hidden_channels: int = 256, min_depth: float = 1e-3) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        if int(hidden_channels) <= 0:
            raise ValueError("hidden_channels must be positive")
        if float(min_depth) <= 0.0:
            raise ValueError("min_depth must be positive")
        self.min_depth = float(min_depth)
        self.net = nn.Sequential(
            nn.GroupNorm(1, int(in_channels)),
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), max(int(hidden_channels) // 4, 16), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(max(int(hidden_channels) // 4, 16), 1, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape (B, C, H, W)")
        log_depth = self.net(tokens).squeeze(1)
        return torch.exp(torch.clamp(log_depth, min=np.log(self.min_depth), max=np.log(1e4)))


def _camera_params(camera: ColmapCamera) -> tuple[float, float, float, float, float]:
    if camera.model_id == 0:
        f, cx, cy = camera.params[:3]
        return float(f), float(f), float(cx), float(cy), 0.0
    if camera.model_id == 1:
        fx, fy, cx, cy = camera.params[:4]
        return float(fx), float(fy), float(cx), float(cy), 0.0
    if camera.model_id == 2:
        f, cx, cy, k = camera.params[:4]
        return float(f), float(f), float(cx), float(cy), float(k)
    raise ValueError(f"unsupported camera model id: {camera.model_id}")


def project_points_to_image(points_xyz: np.ndarray, pose_w2c: np.ndarray, camera: ColmapCamera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    fx, fy, cx, cy, k = _camera_params(camera)
    cam = points @ pose[:3, :3].T + pose[:3, 3]
    z = cam[:, 2]
    safe_z = np.maximum(z, 1e-8)
    x = cam[:, 0] / safe_z
    y = cam[:, 1] / safe_z
    if abs(float(k)) > 0.0:
        r2 = x * x + y * y
        scale = 1.0 + float(k) * r2
        x = x * scale
        y = y * scale
    xy = np.stack((float(fx) * x + float(cx), float(fy) * y + float(cy)), axis=1)
    visible = z > 1e-8
    visible &= (xy[:, 0] >= 0.0) & (xy[:, 0] <= float(camera.width - 1))
    visible &= (xy[:, 1] >= 0.0) & (xy[:, 1] <= float(camera.height - 1))
    return xy.astype(np.float64), z.astype(np.float32), visible.astype(bool)


def rasterize_surface_token_depth(
    surface: SurfaceElementMap,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    token_height: int,
    token_width: int,
    config: TokenDepthRasterConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 2DGS surface element centers and produce robust token depth.

    For each token cell, we first find the nearest visible depth layer, then use
    a median over points within `depth_epsilon` of that layer. This rejects
    mixed foreground/background tokens more conservatively than a plain average.
    """

    cfg = config or TokenDepthRasterConfig()
    token_h = int(token_height)
    token_w = int(token_width)
    if token_h <= 0 or token_w <= 0:
        raise ValueError("token grid dimensions must be positive")
    depth = np.full((token_h, token_w), RADIO_TOKEN_DEPTH_INVALID, dtype=np.float32)
    valid = np.zeros((token_h, token_w), dtype=bool)
    if len(surface) == 0:
        return depth, valid
    xy, z, visible = project_points_to_image(surface.centers, pose_w2c, camera)
    if cfg.max_depth_m is not None:
        visible &= z <= float(cfg.max_depth_m)
    rows = np.flatnonzero(visible)
    if rows.size == 0:
        return depth, valid
    xs = np.floor(np.clip(xy[rows, 0] / max(float(camera.width), 1.0) * token_w, 0.0, float(token_w - 1))).astype(np.int64)
    ys = np.floor(np.clip(xy[rows, 1] / max(float(camera.height), 1.0) * token_h, 0.0, float(token_h - 1))).astype(np.int64)
    buckets: dict[tuple[int, int], list[float]] = {}
    for row, x_idx, y_idx in zip(rows.tolist(), xs.tolist(), ys.tolist()):
        buckets.setdefault((int(y_idx), int(x_idx)), []).append(float(z[int(row)]))
    for (y_idx, x_idx), values in buckets.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.size < int(cfg.min_points_per_token):
            continue
        front = float(np.min(arr))
        layer = arr[arr <= front + float(cfg.depth_epsilon)]
        if layer.size < int(cfg.min_points_per_token):
            continue
        depth[y_idx, x_idx] = np.float32(np.median(layer))
        valid[y_idx, x_idx] = True
    return depth, valid


def splat_surface_token_depth(
    surface: SurfaceElementMap,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    token_height: int,
    token_width: int,
    config: TokenDepthRasterConfig | None = None,
    sigma_scale: float = 2.0,
    min_opacity: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate dense 2DGS token-depth by splatting projected surface disks.

    This is a token-grid z-buffer approximation, not a differentiable 2DGS
    renderer. Each surface element projects its center and two tangent-scale
    endpoints to estimate an ellipse footprint in token coordinates. Covered
    tokens keep the nearest depth layer.
    """

    cfg = config or TokenDepthRasterConfig()
    if float(sigma_scale) <= 0.0:
        raise ValueError("sigma_scale must be positive")
    if float(min_opacity) < 0.0:
        raise ValueError("min_opacity must be non-negative")
    token_h = int(token_height)
    token_w = int(token_width)
    if token_h <= 0 or token_w <= 0:
        raise ValueError("token grid dimensions must be positive")
    depth = np.full((token_h, token_w), RADIO_TOKEN_DEPTH_INVALID, dtype=np.float32)
    valid = np.zeros((token_h, token_w), dtype=bool)
    if len(surface) == 0:
        return depth, valid

    centers = np.asarray(surface.centers, dtype=np.float64)
    center_xy, center_z, center_visible = project_points_to_image(centers, pose_w2c, camera)
    if cfg.max_depth_m is not None:
        center_visible &= center_z <= float(cfg.max_depth_m)
    center_visible &= np.asarray(surface.opacity, dtype=np.float32) >= float(min_opacity)
    rows = np.flatnonzero(center_visible)
    if rows.size == 0:
        return depth, valid

    token_depth = np.full((token_h, token_w), np.inf, dtype=np.float64)
    token_hits = np.zeros((token_h, token_w), dtype=np.int32)
    # Pixel to token-coordinate scale. Token coordinates use [0, W/H).
    sx = float(token_w) / max(float(camera.width), 1.0)
    sy = float(token_h) / max(float(camera.height), 1.0)
    for row in rows.tolist():
        row = int(row)
        z = float(center_z[row])
        if not np.isfinite(z) or z <= 1e-8:
            continue
        center_token = np.asarray([center_xy[row, 0] * sx, center_xy[row, 1] * sy], dtype=np.float64)
        p1 = centers[row] + np.asarray(surface.tangent1[row], dtype=np.float64) * float(surface.scale1[row]) * float(sigma_scale)
        p2 = centers[row] + np.asarray(surface.tangent2[row], dtype=np.float64) * float(surface.scale2[row]) * float(sigma_scale)
        p1_xy, _p1_z, p1_vis = project_points_to_image(p1.reshape(1, 3), pose_w2c, camera)
        p2_xy, _p2_z, p2_vis = project_points_to_image(p2.reshape(1, 3), pose_w2c, camera)
        if bool(p1_vis[0]):
            radius1 = np.linalg.norm(np.asarray([p1_xy[0, 0] * sx, p1_xy[0, 1] * sy]) - center_token)
        else:
            radius1 = 0.5
        if bool(p2_vis[0]):
            radius2 = np.linalg.norm(np.asarray([p2_xy[0, 0] * sx, p2_xy[0, 1] * sy]) - center_token)
        else:
            radius2 = 0.5
        # Keep the approximation conservative but non-degenerate. Very small
        # disks still supervise their center token.
        rx = max(float(radius1), 0.5)
        ry = max(float(radius2), 0.5)
        max_radius = max(rx, ry)
        x0 = max(0, int(np.floor(center_token[0] - max_radius)))
        x1 = min(token_w - 1, int(np.ceil(center_token[0] + max_radius)))
        y0 = max(0, int(np.floor(center_token[1] - max_radius)))
        y1 = min(token_h - 1, int(np.ceil(center_token[1] + max_radius)))
        if x0 > x1 or y0 > y1:
            continue
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                dx = (float(xx) + 0.5 - center_token[0]) / max(rx, 1e-6)
                dy = (float(yy) + 0.5 - center_token[1]) / max(ry, 1e-6)
                if dx * dx + dy * dy > 1.0:
                    continue
                token_hits[yy, xx] += 1
                if z + float(cfg.depth_epsilon) < token_depth[yy, xx]:
                    token_depth[yy, xx] = z
    hit_valid = token_hits >= int(cfg.min_points_per_token)
    valid[hit_valid & np.isfinite(token_depth)] = True
    depth[valid] = token_depth[valid].astype(np.float32)
    return depth, valid


def render_surface_token_depth_label(
    surface: SurfaceElementMap,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    token_height: int,
    token_width: int,
    config: TokenDepthRasterConfig | None = None,
    sigma_scale: float = 2.0,
    min_opacity: float = 0.0,
    max_radius_tokens: float = 0.0,
) -> TokenDepthLabel:
    """Render token-depth labels from 2DGS surface elements with diagnostics.

    This is the preferred fallback when the original 2DGS renderer is not
    available. It uses oriented token-space ellipses derived from projected
    surfel tangents and records enough metadata to distinguish center-projected
    support from footprint-filled support.
    """

    cfg = config or TokenDepthRasterConfig()
    if float(sigma_scale) <= 0.0:
        raise ValueError("sigma_scale must be positive")
    if float(min_opacity) < 0.0:
        raise ValueError("min_opacity must be non-negative")
    if float(max_radius_tokens) < 0.0:
        raise ValueError("max_radius_tokens must be non-negative")
    token_h = int(token_height)
    token_w = int(token_width)
    if token_h <= 0 or token_w <= 0:
        raise ValueError("token grid dimensions must be positive")
    depth = np.full((token_h, token_w), RADIO_TOKEN_DEPTH_INVALID, dtype=np.float32)
    valid = np.zeros((token_h, token_w), dtype=bool)
    confidence_accum = np.zeros((token_h, token_w), dtype=np.float64)
    support_count = np.zeros((token_h, token_w), dtype=np.int32)
    center_valid = np.zeros((token_h, token_w), dtype=bool)
    if len(surface) == 0:
        return TokenDepthLabel(
            depth=depth,
            valid=valid,
            confidence=np.zeros_like(depth),
            support_count=support_count,
            center_valid=center_valid,
            filled_by_splat=np.zeros_like(valid),
        )

    centers = np.asarray(surface.centers, dtype=np.float64)
    center_xy, center_z, center_visible = project_points_to_image(centers, pose_w2c, camera)
    if cfg.max_depth_m is not None:
        center_visible &= center_z <= float(cfg.max_depth_m)
    opacity = np.asarray(surface.opacity, dtype=np.float32).reshape(-1)
    center_visible &= opacity >= float(min_opacity)
    rows = np.flatnonzero(center_visible)
    if rows.size == 0:
        return TokenDepthLabel(
            depth=depth,
            valid=valid,
            confidence=np.zeros_like(depth),
            support_count=support_count,
            center_valid=center_valid,
            filled_by_splat=np.zeros_like(valid),
        )

    token_depth = np.full((token_h, token_w), np.inf, dtype=np.float64)
    sx = float(token_w) / max(float(camera.width), 1.0)
    sy = float(token_h) / max(float(camera.height), 1.0)
    eps_eye = np.eye(2, dtype=np.float64) * 0.25

    tangent1_world = np.asarray(surface.tangent1[rows], dtype=np.float64) * np.asarray(
        surface.scale1[rows], dtype=np.float64
    ).reshape(-1, 1) * float(sigma_scale)
    tangent2_world = np.asarray(surface.tangent2[rows], dtype=np.float64) * np.asarray(
        surface.scale2[rows], dtype=np.float64
    ).reshape(-1, 1) * float(sigma_scale)
    p1_xy, p1_z, _p1_visible = project_points_to_image(centers[rows] + tangent1_world, pose_w2c, camera)
    p2_xy, p2_z, _p2_visible = project_points_to_image(centers[rows] + tangent2_world, pose_w2c, camera)
    center_token_all = np.stack((center_xy[rows, 0] * sx, center_xy[rows, 1] * sy), axis=1).astype(np.float64)
    v1_all = np.stack((p1_xy[:, 0] * sx, p1_xy[:, 1] * sy), axis=1).astype(np.float64) - center_token_all
    v2_all = np.stack((p2_xy[:, 0] * sx, p2_xy[:, 1] * sy), axis=1).astype(np.float64) - center_token_all
    v1_all[~np.isfinite(v1_all).all(axis=1) | ~(p1_z > 1e-8)] = 0.0
    v2_all[~np.isfinite(v2_all).all(axis=1) | ~(p2_z > 1e-8)] = 0.0

    if float(max_radius_tokens) > 0.0:
        max_radius = float(max_radius_tokens)
        a = v1_all[:, 0] * v1_all[:, 0] + v2_all[:, 0] * v2_all[:, 0] + 0.25
        b = v1_all[:, 0] * v1_all[:, 1] + v2_all[:, 0] * v2_all[:, 1]
        c = v1_all[:, 1] * v1_all[:, 1] + v2_all[:, 1] * v2_all[:, 1] + 0.25
        eig_max = 0.5 * ((a + c) + np.sqrt(np.maximum((a - c) * (a - c) + 4.0 * b * b, 0.0)))
        radius = np.sqrt(np.maximum(eig_max, 0.25))
        oversized = radius > max_radius
        if np.any(oversized):
            shrink = max_radius / np.maximum(radius[oversized], 1e-8)
            v1_all[oversized] *= shrink.reshape(-1, 1)
            v2_all[oversized] *= shrink.reshape(-1, 1)
            a = v1_all[:, 0] * v1_all[:, 0] + v2_all[:, 0] * v2_all[:, 0] + 0.25
            b = v1_all[:, 0] * v1_all[:, 1] + v2_all[:, 0] * v2_all[:, 1]
            c = v1_all[:, 1] * v1_all[:, 1] + v2_all[:, 1] * v2_all[:, 1] + 0.25
        det = np.maximum(a * c - b * b, 1e-12)
        inv00 = c / det
        inv01 = -b / det
        inv11 = a / det

        base_x = np.floor(np.clip(center_token_all[:, 0], 0.0, float(token_w - 1))).astype(np.int64)
        base_y = np.floor(np.clip(center_token_all[:, 1], 0.0, float(token_h - 1))).astype(np.int64)
        center_valid[base_y, base_x] = True
        flat_depth = token_depth.reshape(-1)
        z_all = center_z[rows].astype(np.float64, copy=False)
        opacity_all = np.clip(opacity[rows].astype(np.float64, copy=False), 0.0, 1.0)
        radius_i = int(np.ceil(max_radius))

        for dy in range(-radius_i, radius_i + 1):
            yy = base_y + int(dy)
            inside_y = (yy >= 0) & (yy < token_h)
            if not np.any(inside_y):
                continue
            for dx in range(-radius_i, radius_i + 1):
                xx = base_x + int(dx)
                inside = inside_y & (xx >= 0) & (xx < token_w)
                if not np.any(inside):
                    continue
                delta_x = xx.astype(np.float64) + 0.5 - center_token_all[:, 0]
                delta_y = yy.astype(np.float64) + 0.5 - center_token_all[:, 1]
                mahal = inv00 * delta_x * delta_x + 2.0 * inv01 * delta_x * delta_y + inv11 * delta_y * delta_y
                covered = inside & (mahal <= 1.0)
                if not np.any(covered):
                    continue
                flat_index = yy[covered] * token_w + xx[covered]
                np.minimum.at(flat_depth, flat_index, z_all[covered])

        flat_support = support_count.reshape(-1)
        flat_confidence = confidence_accum.reshape(-1)
        eps = float(cfg.depth_epsilon)
        for dy in range(-radius_i, radius_i + 1):
            yy = base_y + int(dy)
            inside_y = (yy >= 0) & (yy < token_h)
            if not np.any(inside_y):
                continue
            for dx in range(-radius_i, radius_i + 1):
                xx = base_x + int(dx)
                inside = inside_y & (xx >= 0) & (xx < token_w)
                if not np.any(inside):
                    continue
                delta_x = xx.astype(np.float64) + 0.5 - center_token_all[:, 0]
                delta_y = yy.astype(np.float64) + 0.5 - center_token_all[:, 1]
                mahal = inv00 * delta_x * delta_x + 2.0 * inv01 * delta_x * delta_y + inv11 * delta_y * delta_y
                covered = inside & (mahal <= 1.0)
                if not np.any(covered):
                    continue
                flat_index = yy[covered] * token_w + xx[covered]
                near_front = z_all[covered] <= flat_depth[flat_index] + eps
                if not np.any(near_front):
                    continue
                chosen_index = flat_index[near_front]
                local_weight = opacity_all[covered][near_front] * np.exp(-0.5 * mahal[covered][near_front])
                np.add.at(flat_support, chosen_index, 1)
                np.add.at(flat_confidence, chosen_index, local_weight)

        valid = (support_count >= int(cfg.min_points_per_token)) & np.isfinite(token_depth)
        depth[valid] = token_depth[valid].astype(np.float32)
        confidence = np.zeros_like(depth, dtype=np.float32)
        confidence[valid] = np.clip(1.0 - np.exp(-confidence_accum[valid]), 0.0, 1.0).astype(np.float32)
        filled_by_splat = valid & ~center_valid
        return TokenDepthLabel(
            depth=depth,
            valid=valid,
            confidence=confidence,
            support_count=support_count,
            center_valid=center_valid,
            filled_by_splat=filled_by_splat,
        )

    for local_idx, row in enumerate(rows.tolist()):
        row = int(row)
        z = float(center_z[row])
        if not np.isfinite(z) or z <= 1e-8:
            continue
        center_token = center_token_all[local_idx]
        center_x = int(np.floor(np.clip(center_token[0], 0.0, float(token_w - 1))))
        center_y = int(np.floor(np.clip(center_token[1], 0.0, float(token_h - 1))))
        center_valid[center_y, center_x] = True

        v1 = v1_all[local_idx].copy()
        v2 = v2_all[local_idx].copy()
        cov = np.outer(v1, v1) + np.outer(v2, v2) + eps_eye
        eigvals = np.linalg.eigvalsh(cov)
        radius = max(float(np.sqrt(max(float(np.max(eigvals)), 0.25))), 0.5)
        if float(max_radius_tokens) > 0.0 and radius > float(max_radius_tokens):
            shrink = float(max_radius_tokens) / max(radius, 1e-8)
            v1 *= shrink
            v2 *= shrink
            cov = np.outer(v1, v1) + np.outer(v2, v2) + eps_eye
            eigvals = np.linalg.eigvalsh(cov)
            radius = max(float(np.sqrt(max(float(np.max(eigvals)), 0.25))), 0.5)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        x0 = max(0, int(np.floor(center_token[0] - radius)))
        x1 = min(token_w - 1, int(np.ceil(center_token[0] + radius)))
        y0 = max(0, int(np.floor(center_token[1] - radius)))
        y1 = min(token_h - 1, int(np.ceil(center_token[1] + radius)))
        if x0 > x1 or y0 > y1:
            continue
        alpha = float(np.clip(opacity[row], 0.0, 1.0))
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                delta = np.asarray([float(xx) + 0.5 - center_token[0], float(yy) + 0.5 - center_token[1]], dtype=np.float64)
                mahal = float(delta @ inv_cov @ delta)
                if mahal > 1.0:
                    continue
                local_weight = alpha * float(np.exp(-0.5 * mahal))
                if z + float(cfg.depth_epsilon) < token_depth[yy, xx]:
                    token_depth[yy, xx] = z
                    support_count[yy, xx] = 1
                    confidence_accum[yy, xx] = local_weight
                elif abs(z - float(token_depth[yy, xx])) <= float(cfg.depth_epsilon):
                    support_count[yy, xx] += 1
                    confidence_accum[yy, xx] += local_weight

    valid = (support_count >= int(cfg.min_points_per_token)) & np.isfinite(token_depth)
    depth[valid] = token_depth[valid].astype(np.float32)
    confidence = np.zeros_like(depth, dtype=np.float32)
    confidence[valid] = np.clip(1.0 - np.exp(-confidence_accum[valid]), 0.0, 1.0).astype(np.float32)
    filled_by_splat = valid & ~center_valid
    return TokenDepthLabel(
        depth=depth,
        valid=valid,
        confidence=confidence,
        support_count=support_count,
        center_valid=center_valid,
        filled_by_splat=filled_by_splat,
    )


def masked_log_depth_l1(pred_depth: torch.Tensor, target_depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if pred_depth.shape != target_depth.shape or pred_depth.shape != valid_mask.shape:
        raise ValueError("pred_depth, target_depth, and valid_mask must have matching shapes")
    valid = valid_mask.bool()
    if not bool(torch.any(valid)):
        return torch.zeros((), dtype=pred_depth.dtype, device=pred_depth.device)
    pred = torch.clamp(pred_depth[valid], min=1e-6)
    target = torch.clamp(target_depth[valid], min=1e-6)
    return torch.mean(torch.abs(torch.log(pred) - torch.log(target)))


def scale_invariant_log_loss(pred_depth: torch.Tensor, target_depth: torch.Tensor, valid_mask: torch.Tensor, lam: float = 0.85) -> torch.Tensor:
    if pred_depth.shape != target_depth.shape or pred_depth.shape != valid_mask.shape:
        raise ValueError("pred_depth, target_depth, and valid_mask must have matching shapes")
    valid = valid_mask.bool()
    if not bool(torch.any(valid)):
        return torch.zeros((), dtype=pred_depth.dtype, device=pred_depth.device)
    diff = torch.log(torch.clamp(pred_depth[valid], min=1e-6)) - torch.log(torch.clamp(target_depth[valid], min=1e-6))
    return torch.mean(diff * diff) - float(lam) * torch.mean(diff) ** 2


def compute_depth_metrics(pred_depth: np.ndarray, target_depth: np.ndarray, valid_mask: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred_depth, dtype=np.float64)
    target = np.asarray(target_depth, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(pred) & np.isfinite(target) & (pred > 0.0) & (target > 0.0)
    if pred.shape != target.shape or pred.shape != valid.shape:
        raise ValueError("pred_depth, target_depth, and valid_mask must have matching shapes")
    if not np.any(valid):
        return {
            "valid_count": 0,
            "abs_rel": float("nan"),
            "rmse": float("nan"),
            "log_rmse": float("nan"),
            "mae": float("nan"),
            "delta1": float("nan"),
            "delta2": float("nan"),
            "delta3": float("nan"),
        }
    p = pred[valid]
    t = target[valid]
    ratio = np.maximum(p / np.maximum(t, 1e-12), t / np.maximum(p, 1e-12))
    log_diff = np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(t, 1e-12))
    return {
        "valid_count": int(p.size),
        "abs_rel": float(np.mean(np.abs(p - t) / np.maximum(t, 1e-12))),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "log_rmse": float(np.sqrt(np.mean(log_diff**2))),
        "mae": float(np.mean(np.abs(p - t))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def aggregate_depth_metrics(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float | int]:
    valid_rows = [row for row in rows if int(row.get("valid_count", 0)) > 0]
    if not valid_rows:
        return {"image_count": 0, "valid_count": 0}
    total = float(sum(int(row["valid_count"]) for row in valid_rows))
    result: dict[str, float | int] = {"image_count": int(len(valid_rows)), "valid_count": int(total)}
    for key in ("abs_rel", "rmse", "log_rmse", "mae", "delta1", "delta2", "delta3"):
        result[key] = float(sum(float(row[key]) * int(row["valid_count"]) for row in valid_rows) / max(total, 1.0))
    return result
