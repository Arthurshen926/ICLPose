"""Raw VFM feature aggregation for sampled Gaussian localization anchors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from feature_extract.vfm.gaussian_vfm_field import GaussianVFMFeatureView, GaussianVFMSource
from feature_extract.vfm.query_to_3d_matching import normalize_rows
from feature_extract.vfm.semidense_anchor_map import SemiDenseAnchorMap


@dataclass(frozen=True)
class VfmGaussianAnchorVoteConfig:
    top_token_fraction: float = 0.05
    min_saliency: float = 0.0
    saliency_mode: str = "norm"
    min_owner_opacity: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.top_token_fraction) <= 1.0:
            raise ValueError("top_token_fraction must be in (0, 1]")
        if float(self.min_saliency) < 0.0:
            raise ValueError("min_saliency must be non-negative")
        if self.saliency_mode not in {"norm", "local_contrast"}:
            raise ValueError("saliency_mode must be 'norm' or 'local_contrast'")
        if not 0.0 <= float(self.min_owner_opacity) <= 1.0:
            raise ValueError("min_owner_opacity must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "top_token_fraction": float(self.top_token_fraction),
            "min_saliency": float(self.min_saliency),
            "saliency_mode": str(self.saliency_mode),
            "min_owner_opacity": float(self.min_owner_opacity),
        }


@dataclass(frozen=True)
class RawGaussianFeatureAggregationConfig:
    min_observations: int = 2
    l2_normalize_observations: bool = True
    l2_normalize_features: bool = True
    require_token_owner_visibility: bool = False
    owner_min_opacity: float = 0.0
    min_contribution_alpha: float = 0.0
    max_contribution_entropy: float | None = None

    def __post_init__(self) -> None:
        if int(self.min_observations) <= 0:
            raise ValueError("min_observations must be positive")
        if not 0.0 <= float(self.owner_min_opacity) <= 1.0:
            raise ValueError("owner_min_opacity must be in [0, 1]")
        if not 0.0 <= float(self.min_contribution_alpha) <= 1.0:
            raise ValueError("min_contribution_alpha must be in [0, 1]")
        if self.max_contribution_entropy is not None and float(self.max_contribution_entropy) < 0.0:
            raise ValueError("max_contribution_entropy must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "min_observations": int(self.min_observations),
            "l2_normalize_observations": bool(self.l2_normalize_observations),
            "l2_normalize_features": bool(self.l2_normalize_features),
            "require_token_owner_visibility": bool(self.require_token_owner_visibility),
            "owner_min_opacity": float(self.owner_min_opacity),
            "min_contribution_alpha": float(self.min_contribution_alpha),
            "max_contribution_entropy": (
                None if self.max_contribution_entropy is None else float(self.max_contribution_entropy)
            ),
        }


@dataclass(frozen=True)
class GaussianTokenContributionView:
    """Dominant Gaussian contribution for every VFM token in one reference view.

    `top_contributor` stores source-row indices in the same Gaussian source array
    that is passed to aggregation. Invalid/unassigned tokens should be -1.
    """

    image_id: str
    feature_map: np.ndarray
    top_contributor: np.ndarray
    top_alpha: np.ndarray | None = None
    alpha_entropy: np.ndarray | None = None


@dataclass(frozen=True)
class GaussianTokenContributionConfig:
    radius_px: float = 1.5
    depth_epsilon: float = 0.02
    opacity_threshold: float = 0.0
    view_angle_power: float = 0.0

    def __post_init__(self) -> None:
        if float(self.radius_px) <= 0.0:
            raise ValueError("radius_px must be positive")
        if float(self.depth_epsilon) < 0.0:
            raise ValueError("depth_epsilon must be non-negative")
        if not 0.0 <= float(self.opacity_threshold) <= 1.0:
            raise ValueError("opacity_threshold must be in [0, 1]")
        if float(self.view_angle_power) < 0.0:
            raise ValueError("view_angle_power must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "radius_px": float(self.radius_px),
            "depth_epsilon": float(self.depth_epsilon),
            "opacity_threshold": float(self.opacity_threshold),
            "view_angle_power": float(self.view_angle_power),
        }


def _intrinsic_matrix(camera, width: int, height: int) -> np.ndarray:
    params = tuple(float(value) for value in camera.params)
    if int(camera.model_id) == 1 and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    elif int(camera.model_id) in {0, 2, 8} and len(params) >= 3:
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        fx = fy = params[0] if params else float(max(width, height))
        cx, cy = float(width) * 0.5, float(height) * 0.5
    sx = float(width) / max(float(camera.width), 1.0)
    sy = float(height) / max(float(camera.height), 1.0)
    return np.asarray([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _project_normalized_to_pixels(x: np.ndarray, y: np.ndarray, camera) -> np.ndarray:
    params = tuple(float(value) for value in camera.params)
    model_id = int(camera.model_id)
    if model_id == 0 and len(params) >= 3:  # SIMPLE_PINHOLE
        f, cx, cy = params[:3]
        u = f * x + cx
        v = f * y + cy
    elif model_id == 1 and len(params) >= 4:  # PINHOLE
        fx, fy, cx, cy = params[:4]
        u = fx * x + cx
        v = fy * y + cy
    elif model_id == 2 and len(params) >= 4:  # SIMPLE_RADIAL
        f, cx, cy, k1 = params[:4]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2
        u = f * x * radial + cx
        v = f * y * radial + cy
    elif model_id == 3 and len(params) >= 5:  # RADIAL
        f, cx, cy, k1, k2 = params[:5]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        u = f * x * radial + cx
        v = f * y * radial + cy
    elif model_id == 4 and len(params) >= 8:  # OPENCV
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        u = fx * x_distorted + cx
        v = fy * y_distorted + cy
    else:
        matrix = _intrinsic_matrix(camera, int(camera.width), int(camera.height))
        homogeneous = np.stack([x, y, np.ones_like(x)], axis=1)
        uvw = (matrix @ homogeneous.T).T
        u = uvw[:, 0]
        v = uvw[:, 1]
    sx = float(camera.width)
    sy = float(camera.height)
    if sx <= 0.0 or sy <= 0.0:
        raise ValueError("camera width and height must be positive")
    return np.stack([u, v], axis=1).astype(np.float64)


def _project_xyz_to_grid(xyz: np.ndarray, pose_w2c: np.ndarray, camera, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    camera_points = (pose @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2]
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, np.nan)
    normalized_x = camera_points[:, 0] / safe_depth
    normalized_y = camera_points[:, 1] / safe_depth
    uv_original = _project_normalized_to_pixels(normalized_x, normalized_y, camera)
    scale = np.asarray(
        [
            float(width) / max(float(camera.width), 1.0),
            float(height) / max(float(camera.height), 1.0),
        ],
        dtype=np.float64,
    )
    uv = uv_original * scale[None, :]
    return uv.astype(np.float64), depth.astype(np.float64)


def _token_owner_map(
    source: GaussianVFMSource,
    view: GaussianVFMFeatureView,
    height: int,
    width: int,
    min_opacity: float = 0.0,
) -> np.ndarray:
    uv, depth = _project_xyz_to_grid(source.xyz, view.pose_w2c, view.camera, width, height)
    finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(depth)
    finite &= (uv[:, 0] >= 0.0) & (uv[:, 0] < float(width))
    finite &= (uv[:, 1] >= 0.0) & (uv[:, 1] < float(height))
    if float(min_opacity) > 0.0:
        finite &= source.opacity >= float(min_opacity)
    x = np.zeros((uv.shape[0],), dtype=np.int64)
    y = np.zeros((uv.shape[0],), dtype=np.int64)
    finite_rows = np.flatnonzero(finite)
    if finite_rows.size:
        x[finite_rows] = np.rint(uv[finite_rows, 0]).astype(np.int64)
        y[finite_rows] = np.rint(uv[finite_rows, 1]).astype(np.int64)
    valid = finite & (depth > 1e-8) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    owner = np.full((height * width,), -1, dtype=np.int64)
    if not np.any(valid):
        return owner.reshape(height, width)
    valid_rows = np.flatnonzero(valid)
    keys = y[valid_rows] * int(width) + x[valid_rows]
    order = np.argsort(depth[valid_rows], kind="mergesort")
    sorted_keys = keys[order]
    sorted_rows = valid_rows[order]
    unique_keys, first = np.unique(sorted_keys, return_index=True)
    owner[unique_keys] = sorted_rows[first]
    return owner.reshape(height, width)


def vfm_token_saliency(feature_map: np.ndarray, mode: str = "norm") -> np.ndarray:
    features = np.asarray(feature_map, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    if mode == "norm":
        return np.linalg.norm(features, axis=0).astype(np.float32)
    if mode == "local_contrast":
        center = features
        diffs = []
        diffs.append(np.abs(center[:, 1:, :] - center[:, :-1, :]).mean(axis=0))
        diffs.append(np.abs(center[:, :, 1:] - center[:, :, :-1]).mean(axis=0))
        saliency = np.zeros(features.shape[1:], dtype=np.float32)
        saliency[1:, :] += diffs[0]
        saliency[:-1, :] += diffs[0]
        saliency[:, 1:] += diffs[1]
        saliency[:, :-1] += diffs[1]
        return saliency
    raise ValueError("mode must be 'norm' or 'local_contrast'")


def _top_saliency_mask(saliency: np.ndarray, fraction: float, min_saliency: float) -> np.ndarray:
    values = np.asarray(saliency, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros_like(saliency, dtype=bool)
    keep = max(1, int(np.ceil(float(values.size) * float(fraction))))
    threshold = float(np.partition(values, max(0, values.size - keep))[max(0, values.size - keep)])
    threshold = max(threshold, float(min_saliency))
    return np.asarray(saliency >= threshold, dtype=bool)


def _camera_center_from_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return (-rotation.T @ translation).astype(np.float64, copy=False)


def _view_angle_weights(source: GaussianVFMSource, pose_w2c: np.ndarray, power: float) -> np.ndarray:
    if float(power) <= 0.0 or source.normal is None:
        return np.ones((source.xyz.shape[0],), dtype=np.float32)
    camera_center = _camera_center_from_w2c(pose_w2c)
    directions = camera_center[None, :] - np.asarray(source.xyz, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-8)
    normals = np.asarray(source.normal, dtype=np.float64)
    weights = np.abs(np.sum(normals * directions, axis=1))
    weights = np.power(np.clip(weights, 0.0, 1.0), float(power))
    return weights.astype(np.float32, copy=False)


def build_gaussian_token_contribution_view(
    source: GaussianVFMSource,
    view: GaussianVFMFeatureView,
    config: GaussianTokenContributionConfig | None = None,
) -> GaussianTokenContributionView:
    """Approximate dominant Gaussian contribution on the VFM token grid.

    This is a token-grid splat diagnostic, not a full 3DGS rasterizer. It keeps
    projected Gaussians within a local splat radius, selects the front depth
    layer within `depth_epsilon`, and records the highest normalized
    contribution for every token.
    """

    cfg = config or GaussianTokenContributionConfig()
    feature_map = np.asarray(view.feature_map, dtype=np.float32)
    _channels, height, width = feature_map.shape
    top_contributor = np.full((height, width), -1, dtype=np.int64)
    top_alpha = np.zeros((height, width), dtype=np.float32)
    alpha_entropy = np.zeros((height, width), dtype=np.float32)
    if source.xyz.shape[0] == 0:
        return GaussianTokenContributionView(
            image_id=view.image_id,
            feature_map=feature_map,
            top_contributor=top_contributor,
            top_alpha=top_alpha,
            alpha_entropy=alpha_entropy,
        )
    uv, depth = _project_xyz_to_grid(source.xyz, view.pose_w2c, view.camera, width, height)
    radius = float(cfg.radius_px)
    valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(depth)
    valid &= depth > 1e-8
    valid &= (uv[:, 0] >= -radius) & (uv[:, 0] < float(width) + radius)
    valid &= (uv[:, 1] >= -radius) & (uv[:, 1] < float(height) + radius)
    valid &= np.asarray(source.opacity, dtype=np.float32) >= float(cfg.opacity_threshold)
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size == 0:
        return GaussianTokenContributionView(
            image_id=view.image_id,
            feature_map=feature_map,
            top_contributor=top_contributor,
            top_alpha=top_alpha,
            alpha_entropy=alpha_entropy,
        )
    projected = uv[valid_rows].astype(np.float64, copy=False)
    tree = cKDTree(projected)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64), indexing="ij")
    token_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    neighbors = tree.query_ball_point(token_xy, r=radius)
    view_weights = _view_angle_weights(source, view.pose_w2c, float(cfg.view_angle_power))
    sigma_sq = max((radius * 0.5) ** 2, 1e-6)
    for token_idx, local_neighbors in enumerate(neighbors):
        if not local_neighbors:
            continue
        rows = valid_rows[np.asarray(local_neighbors, dtype=np.int64)]
        token = token_xy[token_idx]
        offsets = uv[rows] - token[None, :]
        dist_sq = np.sum(np.square(offsets), axis=1)
        nearest_depth = float(np.min(depth[rows]))
        depth_mask = depth[rows] <= nearest_depth + float(cfg.depth_epsilon)
        if not np.any(depth_mask):
            continue
        rows = rows[depth_mask]
        dist_sq = dist_sq[depth_mask]
        weights = (
            np.asarray(source.opacity, dtype=np.float32)[rows].astype(np.float64)
            * np.exp(-0.5 * dist_sq / sigma_sq)
            * view_weights[rows].astype(np.float64)
        )
        positive = weights > 1e-12
        if not np.any(positive):
            continue
        rows = rows[positive]
        weights = weights[positive]
        total = float(np.sum(weights))
        if total <= 0.0:
            continue
        probs = weights / total
        best_local = int(np.argmax(weights))
        y = token_idx // int(width)
        x = token_idx % int(width)
        top_contributor[y, x] = int(rows[best_local])
        top_alpha[y, x] = np.float32(probs[best_local])
        if probs.size <= 1:
            alpha_entropy[y, x] = 0.0
        else:
            entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
            alpha_entropy[y, x] = np.float32(entropy / max(float(np.log(probs.size)), 1e-12))
    return GaussianTokenContributionView(
        image_id=view.image_id,
        feature_map=feature_map,
        top_contributor=top_contributor,
        top_alpha=top_alpha,
        alpha_entropy=alpha_entropy,
    )


def vote_gaussians_from_token_contribution_views(
    source_gaussian_count: int,
    contribution_views: Sequence[GaussianTokenContributionView],
    vote_config: VfmGaussianAnchorVoteConfig | None = None,
    min_contribution_alpha: float = 0.0,
    max_contribution_entropy: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    cfg = vote_config or VfmGaussianAnchorVoteConfig()
    votes = np.zeros((int(source_gaussian_count),), dtype=np.int64)
    rows = []
    for view in contribution_views:
        feature_map = np.asarray(view.feature_map, dtype=np.float32)
        _channels, height, width = feature_map.shape
        saliency = vfm_token_saliency(feature_map, mode=cfg.saliency_mode)
        token_mask = _top_saliency_mask(saliency, cfg.top_token_fraction, cfg.min_saliency)
        owners = np.asarray(view.top_contributor, dtype=np.int64)
        if owners.shape != (height, width):
            raise ValueError("top_contributor must match feature token grid")
        valid = token_mask & (owners >= 0) & (owners < int(source_gaussian_count))
        if view.top_alpha is not None:
            top_alpha = np.asarray(view.top_alpha, dtype=np.float32)
            if top_alpha.shape != (height, width):
                raise ValueError("top_alpha must match feature token grid")
            valid &= top_alpha >= float(min_contribution_alpha)
        elif float(min_contribution_alpha) > 0.0:
            raise ValueError("top_alpha is required when min_contribution_alpha is positive")
        if max_contribution_entropy is not None:
            if view.alpha_entropy is None:
                raise ValueError("alpha_entropy is required when max_contribution_entropy is set")
            entropy = np.asarray(view.alpha_entropy, dtype=np.float32)
            if entropy.shape != (height, width):
                raise ValueError("alpha_entropy must match feature token grid")
            valid &= entropy <= float(max_contribution_entropy)
        hit_owners = owners[valid]
        unique_hits = np.unique(hit_owners)
        if unique_hits.size:
            votes[unique_hits] += 1
        rows.append(
            {
                "image_id": str(view.image_id),
                "selected_tokens": int(np.sum(token_mask)),
                "valid_contribution_tokens": int(np.sum(valid)),
                "voted_gaussians": int(unique_hits.size),
                "grid_height": int(height),
                "grid_width": int(width),
            }
        )
    return votes, {
        "stage": "vfm_token_saliency_contribution_gaussian_votes",
        "config": cfg.to_dict(),
        "min_contribution_alpha": float(min_contribution_alpha),
        "max_contribution_entropy": (
            None if max_contribution_entropy is None else float(max_contribution_entropy)
        ),
        "view_count": int(len(rows)),
        "voted_gaussian_count": int(np.sum(votes > 0)),
        "max_votes": int(np.max(votes)) if votes.size else 0,
        "views": rows,
    }


def vote_gaussians_from_vfm_token_saliency(
    source: GaussianVFMSource,
    views: Sequence[GaussianVFMFeatureView],
    config: VfmGaussianAnchorVoteConfig | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    cfg = config or VfmGaussianAnchorVoteConfig()
    votes = np.zeros((source.xyz.shape[0],), dtype=np.int64)
    rows = []
    for view in views:
        feature_map = np.asarray(view.feature_map, dtype=np.float32)
        _channels, height, width = feature_map.shape
        saliency = vfm_token_saliency(feature_map, mode=cfg.saliency_mode)
        token_mask = _top_saliency_mask(saliency, cfg.top_token_fraction, cfg.min_saliency)
        owner = _token_owner_map(source, view, height, width, min_opacity=float(cfg.min_owner_opacity))
        hit_owners = owner[token_mask]
        hit_owners = hit_owners[hit_owners >= 0]
        if hit_owners.size:
            votes[np.unique(hit_owners)] += 1
        rows.append(
            {
                "image_id": view.image_id,
                "selected_tokens": int(np.sum(token_mask)),
                "voted_gaussians": int(np.unique(hit_owners).size),
                "grid_height": int(height),
                "grid_width": int(width),
            }
        )
    return votes, {
        "stage": "vfm_token_saliency_gaussian_votes",
        "config": cfg.to_dict(),
        "view_count": int(len(rows)),
        "voted_gaussian_count": int(np.sum(votes > 0)),
        "max_votes": int(np.max(votes)) if votes.size else 0,
        "views": rows,
    }


def vote_gaussians_from_vfm_token_saliency_configs(
    source: GaussianVFMSource,
    views: Sequence[GaussianVFMFeatureView],
    configs: Mapping[str, VfmGaussianAnchorVoteConfig],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Vote Gaussian anchors for several VFM-token saliency configs in one pass."""

    if not configs:
        raise ValueError("configs must be non-empty")
    normalized_configs = {str(name): config for name, config in configs.items()}
    votes_by_name = {
        name: np.zeros((source.xyz.shape[0],), dtype=np.int64)
        for name in normalized_configs
    }
    rows_by_name: dict[str, list[dict[str, object]]] = {name: [] for name in normalized_configs}
    for view in views:
        feature_map = np.asarray(view.feature_map, dtype=np.float32)
        _channels, height, width = feature_map.shape
        saliency_by_mode: dict[str, np.ndarray] = {}
        owner_by_opacity: dict[float, np.ndarray] = {}
        for name, cfg in normalized_configs.items():
            saliency = saliency_by_mode.get(cfg.saliency_mode)
            if saliency is None:
                saliency = vfm_token_saliency(feature_map, mode=cfg.saliency_mode)
                saliency_by_mode[cfg.saliency_mode] = saliency
            opacity_key = float(cfg.min_owner_opacity)
            owner = owner_by_opacity.get(opacity_key)
            if owner is None:
                owner = _token_owner_map(source, view, height, width, min_opacity=opacity_key)
                owner_by_opacity[opacity_key] = owner
            token_mask = _top_saliency_mask(saliency, cfg.top_token_fraction, cfg.min_saliency)
            hit_owners = owner[token_mask]
            hit_owners = hit_owners[hit_owners >= 0]
            unique_hits = np.unique(hit_owners)
            if unique_hits.size:
                votes_by_name[name][unique_hits] += 1
            rows_by_name[name].append(
                {
                    "image_id": view.image_id,
                    "selected_tokens": int(np.sum(token_mask)),
                    "voted_gaussians": int(unique_hits.size),
                    "grid_height": int(height),
                    "grid_width": int(width),
                }
            )
    config_summaries = {}
    for name, cfg in normalized_configs.items():
        votes = votes_by_name[name]
        config_summaries[name] = {
            "config": cfg.to_dict(),
            "view_count": int(len(rows_by_name[name])),
            "voted_gaussian_count": int(np.sum(votes > 0)),
            "max_votes": int(np.max(votes)) if votes.size else 0,
            "views": rows_by_name[name],
        }
    return votes_by_name, {
        "stage": "vfm_token_saliency_gaussian_votes_multi",
        "view_count": int(len(views)),
        "configs": config_summaries,
    }


def sample_gaussian_indices_from_votes(
    source: GaussianVFMSource,
    vote_counts: np.ndarray,
    max_anchors: int,
    min_votes: int = 1,
    nms_voxel_size: float = 0.03,
    opacity_power: float = 0.0,
    min_opacity: float = 0.0,
) -> np.ndarray:
    votes = np.asarray(vote_counts, dtype=np.int64).reshape(-1)
    if votes.shape != (source.xyz.shape[0],):
        raise ValueError("vote_counts must have shape (N,)")
    if int(max_anchors) <= 0:
        raise ValueError("max_anchors must be positive")
    if float(opacity_power) < 0.0:
        raise ValueError("opacity_power must be non-negative")
    if not 0.0 <= float(min_opacity) <= 1.0:
        raise ValueError("min_opacity must be in [0, 1]")
    candidates = np.flatnonzero((votes >= int(min_votes)) & (source.opacity >= float(min_opacity)))
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.int64)
    scores = votes[candidates].astype(np.float64)
    if float(opacity_power) > 0.0:
        scores = scores * np.power(np.maximum(source.opacity[candidates].astype(np.float64), 1e-8), float(opacity_power))
    order = np.lexsort((-source.opacity[candidates], -votes[candidates], -scores))
    ordered = candidates[order]
    if float(nms_voxel_size) <= 0.0:
        return ordered[: int(max_anchors)].astype(np.int64, copy=False)
    selected: list[int] = []
    occupied: set[tuple[int, int, int]] = set()
    inv = 1.0 / float(nms_voxel_size)
    for row in ordered.tolist():
        key = tuple(np.floor(source.xyz[int(row)] * inv).astype(np.int64).tolist())
        if key in occupied:
            continue
        occupied.add(key)
        selected.append(int(row))
        if len(selected) >= int(max_anchors):
            break
    return np.asarray(selected, dtype=np.int64)


def _bilinear_sample_feature(feature_map: np.ndarray, xy: np.ndarray) -> np.ndarray:
    channels, height, width = feature_map.shape
    x = np.clip(xy[:, 0], 0.0, float(width - 1))
    y = np.clip(xy[:, 1], 0.0, float(height - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    f00 = feature_map[:, y0, x0].T
    f01 = feature_map[:, y1, x0].T
    f10 = feature_map[:, y0, x1].T
    f11 = feature_map[:, y1, x1].T
    return (
        f00 * ((1.0 - wx) * (1.0 - wy))[:, None]
        + f10 * (wx * (1.0 - wy))[:, None]
        + f01 * ((1.0 - wx) * wy)[:, None]
        + f11 * (wx * wy)[:, None]
    ).astype(np.float32)


def _validate_contribution_view(
    view: GaussianTokenContributionView,
    feature_dim: int,
    cfg: RawGaussianFeatureAggregationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    feature_map = np.asarray(view.feature_map, dtype=np.float32)
    if feature_map.ndim != 3:
        raise ValueError("contribution view feature_map must have shape (C, H, W)")
    if int(feature_map.shape[0]) != int(feature_dim):
        raise ValueError("all contribution views must have the same feature dimension")
    _channels, height, width = feature_map.shape
    top_contributor = np.asarray(view.top_contributor, dtype=np.int64)
    if top_contributor.shape != (height, width):
        raise ValueError("top_contributor must have shape (H, W) matching feature_map")
    top_alpha = None
    if view.top_alpha is not None:
        top_alpha = np.asarray(view.top_alpha, dtype=np.float32)
        if top_alpha.shape != (height, width):
            raise ValueError("top_alpha must have shape (H, W) matching feature_map")
    elif float(cfg.min_contribution_alpha) > 0.0:
        raise ValueError("top_alpha is required when min_contribution_alpha is positive")
    alpha_entropy = None
    if view.alpha_entropy is not None:
        alpha_entropy = np.asarray(view.alpha_entropy, dtype=np.float32)
        if alpha_entropy.shape != (height, width):
            raise ValueError("alpha_entropy must have shape (H, W) matching feature_map")
    elif cfg.max_contribution_entropy is not None:
        raise ValueError("alpha_entropy is required when max_contribution_entropy is set")
    return feature_map, top_contributor, top_alpha, alpha_entropy


def aggregate_raw_vfm_features_from_token_contributions(
    source: GaussianVFMSource,
    sampled_source_indices: np.ndarray,
    contribution_views: Sequence[GaussianTokenContributionView],
    config: RawGaussianFeatureAggregationConfig | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SemiDenseAnchorMap:
    """Aggregate raw VFM token features using explicit Gaussian contribution maps.

    This is the sparse-map counterpart of render-contribution visibility: a
    reference token contributes to a Gaussian only when the rasterizer marks that
    Gaussian as the token's dominant visible contributor, optionally passing
    top-alpha and alpha-entropy quality gates.
    """

    cfg = config or RawGaussianFeatureAggregationConfig()
    sampled = np.asarray(sampled_source_indices, dtype=np.int64).reshape(-1)
    if sampled.size and (np.min(sampled) < 0 or np.max(sampled) >= source.xyz.shape[0]):
        raise ValueError("sampled_source_indices contains out-of-range source rows")
    feature_dim = int(contribution_views[0].feature_map.shape[0]) if contribution_views else 0
    sums = np.zeros((sampled.size, feature_dim), dtype=np.float32)
    sum_squares = np.zeros((sampled.size, feature_dim), dtype=np.float32)
    counts = np.zeros((sampled.size,), dtype=np.int64)
    observation_image_ids: list[list[str]] = [[] for _ in range(int(sampled.size))]
    source_to_local = np.full((source.xyz.shape[0],), -1, dtype=np.int64)
    if sampled.size:
        source_to_local[sampled] = np.arange(sampled.size, dtype=np.int64)

    for view in contribution_views:
        feature_map, top_contributor, top_alpha, alpha_entropy = _validate_contribution_view(view, feature_dim, cfg)
        _channels, height, width = feature_map.shape
        owners_flat = top_contributor.reshape(-1)
        valid = (owners_flat >= 0) & (owners_flat < source.xyz.shape[0])
        if top_alpha is not None:
            valid &= top_alpha.reshape(-1) >= float(cfg.min_contribution_alpha)
        if alpha_entropy is not None and cfg.max_contribution_entropy is not None:
            valid &= alpha_entropy.reshape(-1) <= float(cfg.max_contribution_entropy)
        token_rows = np.flatnonzero(valid)
        if token_rows.size == 0:
            continue
        local_rows = source_to_local[owners_flat[token_rows]]
        keep_tokens = local_rows >= 0
        if not np.any(keep_tokens):
            continue
        token_rows = token_rows[keep_tokens]
        local_rows = local_rows[keep_tokens]
        token_features = feature_map.reshape(feature_dim, height * width).T[token_rows].astype(np.float32, copy=False)
        view_sums = np.zeros((sampled.size, feature_dim), dtype=np.float32)
        view_counts = np.zeros((sampled.size,), dtype=np.int64)
        np.add.at(view_sums, local_rows, token_features)
        np.add.at(view_counts, local_rows, 1)
        local_valid = np.flatnonzero(view_counts > 0)
        if local_valid.size == 0:
            continue
        view_features = view_sums[local_valid] / np.maximum(view_counts[local_valid, None], 1)
        if cfg.l2_normalize_observations:
            view_features, _valid_norm = normalize_rows(view_features.astype(np.float32, copy=False))
        sums[local_valid] += view_features.astype(np.float32, copy=False)
        sum_squares[local_valid] += np.square(view_features.astype(np.float32, copy=False))
        counts[local_valid] += 1
        for local_row in local_valid.tolist():
            observation_image_ids[int(local_row)].append(str(view.image_id))

    keep_local = np.flatnonzero(counts >= int(cfg.min_observations))
    features = sums[keep_local] / np.maximum(counts[keep_local, None], 1)
    variances = (
        sum_squares[keep_local] / np.maximum(counts[keep_local, None], 1)
        - np.square(features)
    )
    mean_variances = np.maximum(np.mean(variances, axis=1), 0.0).astype(np.float32, copy=False)
    if cfg.l2_normalize_features:
        features, _valid = normalize_rows(features.astype(np.float32, copy=False))
    keep_source = sampled[keep_local]
    return SemiDenseAnchorMap(
        anchor_ids=np.arange(keep_source.size, dtype=np.int64),
        xyz=source.xyz[keep_source].astype(np.float64, copy=False),
        features=features.astype(np.float32, copy=False),
        source_types=np.asarray(["gaussian_raw_vfm"] * int(keep_source.size), dtype=str),
        source_track_ids=np.full((int(keep_source.size),), -1, dtype=np.int64),
        source_gaussian_indices=source.gaussian_indices[keep_source].astype(np.int64, copy=False),
        support_counts=counts[keep_local].astype(np.int64, copy=False),
        mean_distances=np.zeros((int(keep_source.size),), dtype=np.float32),
        feature_variances=mean_variances,
        observation_counts=counts[keep_local].astype(np.int64, copy=False),
        visibility_counts=counts[keep_local].astype(np.int64, copy=False),
        quality_scores=np.clip(counts[keep_local].astype(np.float32) / max(float(np.max(counts)), 1.0), 0.0, 1.0),
        opacity=source.opacity[keep_source].astype(np.float32, copy=False),
        scale=source.scale[keep_source].astype(np.float32, copy=False),
        observation_image_ids=tuple(tuple(observation_image_ids[int(local_idx)]) for local_idx in keep_local.tolist()),
        metadata={
            "stage": "stage_h2_render_contribution_raw_gaussian_vfm_anchor_map",
            "aggregation_config": cfg.to_dict(),
            "input_sampled_gaussian_count": int(sampled.size),
            "source_gaussian_count": int(source.xyz.shape[0]),
            "contribution_view_count": int(len(contribution_views)),
            **dict(metadata or {}),
        },
    )


def aggregate_raw_vfm_features_from_contribution_visibility(
    source: GaussianVFMSource,
    sampled_source_indices: np.ndarray,
    contribution_views: Sequence[GaussianTokenContributionView],
    camera_views: Sequence[GaussianVFMFeatureView],
    config: RawGaussianFeatureAggregationConfig | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SemiDenseAnchorMap:
    """Bilinear-sample raw features at Gaussian projections gated by contribution visibility."""

    cfg = config or RawGaussianFeatureAggregationConfig()
    if len(contribution_views) != len(camera_views):
        raise ValueError("contribution_views and camera_views must have the same length")
    sampled = np.asarray(sampled_source_indices, dtype=np.int64).reshape(-1)
    if sampled.size and (np.min(sampled) < 0 or np.max(sampled) >= source.xyz.shape[0]):
        raise ValueError("sampled_source_indices contains out-of-range source rows")
    feature_dim = int(contribution_views[0].feature_map.shape[0]) if contribution_views else 0
    sums = np.zeros((sampled.size, feature_dim), dtype=np.float32)
    sum_squares = np.zeros((sampled.size, feature_dim), dtype=np.float32)
    counts = np.zeros((sampled.size,), dtype=np.int64)
    observation_image_ids: list[list[str]] = [[] for _ in range(int(sampled.size))]
    for contribution_view, camera_view in zip(contribution_views, camera_views):
        if str(contribution_view.image_id) != str(camera_view.image_id):
            raise ValueError("contribution view and camera view image_id mismatch")
        feature_map = np.asarray(contribution_view.feature_map, dtype=np.float32)
        if int(feature_map.shape[0]) != feature_dim:
            raise ValueError("all contribution views must have the same channel dimension")
        _channels, height, width = feature_map.shape
        top_contributor = np.asarray(contribution_view.top_contributor, dtype=np.int64)
        if top_contributor.shape != (height, width):
            raise ValueError("top_contributor must match feature token grid")
        top_alpha = None if contribution_view.top_alpha is None else np.asarray(contribution_view.top_alpha, dtype=np.float32)
        if top_alpha is not None and top_alpha.shape != (height, width):
            raise ValueError("top_alpha must match feature token grid")
        alpha_entropy = (
            None if contribution_view.alpha_entropy is None else np.asarray(contribution_view.alpha_entropy, dtype=np.float32)
        )
        if alpha_entropy is not None and alpha_entropy.shape != (height, width):
            raise ValueError("alpha_entropy must match feature token grid")
        uv, depth = _project_xyz_to_grid(source.xyz[sampled], camera_view.pose_w2c, camera_view.camera, width, height)
        finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(depth)
        finite &= depth > 1e-8
        finite &= (uv[:, 0] >= 0.0) & (uv[:, 0] < float(width))
        finite &= (uv[:, 1] >= 0.0) & (uv[:, 1] < float(height))
        x = np.zeros((uv.shape[0],), dtype=np.int64)
        y = np.zeros((uv.shape[0],), dtype=np.int64)
        finite_rows = np.flatnonzero(finite)
        if finite_rows.size:
            x[finite_rows] = np.rint(uv[finite_rows, 0]).astype(np.int64)
            y[finite_rows] = np.rint(uv[finite_rows, 1]).astype(np.int64)
        in_grid = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        local_rows = np.flatnonzero(in_grid)
        if local_rows.size == 0:
            continue
        owner_valid = top_contributor[y[local_rows], x[local_rows]] == sampled[local_rows]
        if top_alpha is not None:
            owner_valid &= top_alpha[y[local_rows], x[local_rows]] >= float(cfg.min_contribution_alpha)
        elif float(cfg.min_contribution_alpha) > 0.0:
            raise ValueError("top_alpha is required when min_contribution_alpha is positive")
        if cfg.max_contribution_entropy is not None:
            if alpha_entropy is None:
                raise ValueError("alpha_entropy is required when max_contribution_entropy is set")
            owner_valid &= alpha_entropy[y[local_rows], x[local_rows]] <= float(cfg.max_contribution_entropy)
        local_valid = local_rows[owner_valid]
        if local_valid.size == 0:
            continue
        sampled_features = _bilinear_sample_feature(feature_map, uv[local_valid])
        if cfg.l2_normalize_observations:
            sampled_features, _valid_norm = normalize_rows(sampled_features)
        sums[local_valid] += sampled_features.astype(np.float32, copy=False)
        sum_squares[local_valid] += np.square(sampled_features.astype(np.float32, copy=False))
        counts[local_valid] += 1
        for local_row in local_valid.tolist():
            observation_image_ids[int(local_row)].append(str(contribution_view.image_id))
    keep_local = np.flatnonzero(counts >= int(cfg.min_observations))
    features = sums[keep_local] / np.maximum(counts[keep_local, None], 1)
    variances = (
        sum_squares[keep_local] / np.maximum(counts[keep_local, None], 1)
        - np.square(features)
    )
    mean_variances = np.maximum(np.mean(variances, axis=1), 0.0).astype(np.float32, copy=False)
    if cfg.l2_normalize_features:
        features, _valid = normalize_rows(features.astype(np.float32, copy=False))
    keep_source = sampled[keep_local]
    return SemiDenseAnchorMap(
        anchor_ids=np.arange(keep_source.size, dtype=np.int64),
        xyz=source.xyz[keep_source].astype(np.float64, copy=False),
        features=features.astype(np.float32, copy=False),
        source_types=np.asarray(["gaussian_raw_vfm"] * int(keep_source.size), dtype=str),
        source_track_ids=np.full((int(keep_source.size),), -1, dtype=np.int64),
        source_gaussian_indices=source.gaussian_indices[keep_source].astype(np.int64, copy=False),
        support_counts=counts[keep_local].astype(np.int64, copy=False),
        mean_distances=np.zeros((int(keep_source.size),), dtype=np.float32),
        feature_variances=mean_variances,
        observation_counts=counts[keep_local].astype(np.int64, copy=False),
        visibility_counts=counts[keep_local].astype(np.int64, copy=False),
        quality_scores=np.clip(counts[keep_local].astype(np.float32) / max(float(np.max(counts)), 1.0), 0.0, 1.0),
        opacity=source.opacity[keep_source].astype(np.float32, copy=False),
        scale=source.scale[keep_source].astype(np.float32, copy=False),
        observation_image_ids=tuple(tuple(observation_image_ids[int(local_idx)]) for local_idx in keep_local.tolist()),
        metadata={
            "stage": "stage_h2_contribution_visible_center_sampled_raw_gaussian_vfm_anchor_map",
            "aggregation_config": cfg.to_dict(),
            "input_sampled_gaussian_count": int(sampled.size),
            "source_gaussian_count": int(source.xyz.shape[0]),
            "contribution_view_count": int(len(contribution_views)),
            **dict(metadata or {}),
        },
    )


def aggregate_raw_vfm_features_to_gaussian_anchors(
    source: GaussianVFMSource,
    sampled_source_indices: np.ndarray,
    views: Sequence[GaussianVFMFeatureView],
    config: RawGaussianFeatureAggregationConfig | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SemiDenseAnchorMap:
    cfg = config or RawGaussianFeatureAggregationConfig()
    sampled = np.asarray(sampled_source_indices, dtype=np.int64).reshape(-1)
    if sampled.size and (np.min(sampled) < 0 or np.max(sampled) >= source.xyz.shape[0]):
        raise ValueError("sampled_source_indices contains out-of-range source rows")
    feature_dim = int(views[0].feature_map.shape[0]) if views else 0
    sums = np.zeros((sampled.size, feature_dim), dtype=np.float32)
    sum_squares = np.zeros((sampled.size, feature_dim), dtype=np.float32)
    counts = np.zeros((sampled.size,), dtype=np.int64)
    observation_image_ids: list[list[str]] = [[] for _ in range(int(sampled.size))]
    for view in views:
        feature_map = np.asarray(view.feature_map, dtype=np.float32)
        if int(feature_map.shape[0]) != feature_dim:
            raise ValueError("all views must have the same feature dimension")
        _channels, height, width = feature_map.shape
        uv, depth = _project_xyz_to_grid(source.xyz[sampled], view.pose_w2c, view.camera, width, height)
        valid = (
            np.isfinite(uv[:, 0])
            & np.isfinite(uv[:, 1])
            & np.isfinite(depth)
            & (depth > 1e-8)
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= float(width - 1))
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= float(height - 1))
        )
        if cfg.require_token_owner_visibility:
            owner = _token_owner_map(source, view, height, width, min_opacity=float(cfg.owner_min_opacity))
            finite_grid = (
                np.isfinite(uv[:, 0])
                & np.isfinite(uv[:, 1])
                & (uv[:, 0] >= 0.0)
                & (uv[:, 0] < float(width))
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] < float(height))
            )
            x = np.zeros((uv.shape[0],), dtype=np.int64)
            y = np.zeros((uv.shape[0],), dtype=np.int64)
            finite_rows = np.flatnonzero(finite_grid)
            if finite_rows.size:
                x[finite_rows] = np.rint(uv[finite_rows, 0]).astype(np.int64)
                y[finite_rows] = np.rint(uv[finite_rows, 1]).astype(np.int64)
            in_grid = finite_grid & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            owner_valid = np.zeros_like(valid)
            local_rows = np.flatnonzero(valid & in_grid)
            for local_row in local_rows.tolist():
                owner_valid[local_row] = int(owner[y[local_row], x[local_row]]) == int(sampled[local_row])
            valid &= owner_valid
        local_valid = np.flatnonzero(valid)
        if local_valid.size == 0:
            continue
        sampled_features = _bilinear_sample_feature(feature_map, uv[local_valid])
        if cfg.l2_normalize_observations:
            sampled_features, _valid_norm = normalize_rows(sampled_features)
        sums[local_valid] += sampled_features.astype(np.float32, copy=False)
        sum_squares[local_valid] += np.square(sampled_features.astype(np.float32, copy=False))
        counts[local_valid] += 1
        for local_row in local_valid.tolist():
            observation_image_ids[int(local_row)].append(str(view.image_id))
    keep_local = np.flatnonzero(counts >= int(cfg.min_observations))
    features = sums[keep_local] / np.maximum(counts[keep_local, None], 1)
    variances = (
        sum_squares[keep_local] / np.maximum(counts[keep_local, None], 1)
        - np.square(features)
    )
    mean_variances = np.maximum(np.mean(variances, axis=1), 0.0).astype(np.float32, copy=False)
    if cfg.l2_normalize_features:
        features, _valid = normalize_rows(features.astype(np.float32, copy=False))
    keep_source = sampled[keep_local]
    return SemiDenseAnchorMap(
        anchor_ids=np.arange(keep_source.size, dtype=np.int64),
        xyz=source.xyz[keep_source].astype(np.float64, copy=False),
        features=features.astype(np.float32, copy=False),
        source_types=np.asarray(["gaussian_raw_vfm"] * int(keep_source.size), dtype=str),
        source_track_ids=np.full((int(keep_source.size),), -1, dtype=np.int64),
        source_gaussian_indices=source.gaussian_indices[keep_source].astype(np.int64, copy=False),
        support_counts=counts[keep_local].astype(np.int64, copy=False),
        mean_distances=np.zeros((int(keep_source.size),), dtype=np.float32),
        feature_variances=mean_variances,
        observation_counts=counts[keep_local].astype(np.int64, copy=False),
        visibility_counts=counts[keep_local].astype(np.int64, copy=False),
        quality_scores=np.clip(counts[keep_local].astype(np.float32) / max(float(np.max(counts)), 1.0), 0.0, 1.0),
        opacity=source.opacity[keep_source].astype(np.float32, copy=False),
        scale=source.scale[keep_source].astype(np.float32, copy=False),
        observation_image_ids=tuple(tuple(observation_image_ids[int(local_idx)]) for local_idx in keep_local.tolist()),
        metadata={
            "stage": "stage_h2_raw_gaussian_vfm_anchor_map",
            "aggregation_config": cfg.to_dict(),
            "input_sampled_gaussian_count": int(sampled.size),
            "source_gaussian_count": int(source.xyz.shape[0]),
            **dict(metadata or {}),
        },
    )


def subset_gaussian_anchor_map_by_source_indices(
    anchor_map: SemiDenseAnchorMap,
    source: GaussianVFMSource,
    sampled_source_indices: np.ndarray,
    metadata: Mapping[str, object] | None = None,
) -> SemiDenseAnchorMap:
    """Return anchors whose Gaussian source rows are in `sampled_source_indices`.

    The output order follows `sampled_source_indices`. Missing rows are expected
    when the union aggregation did not meet `min_observations` for that Gaussian.
    """

    sampled = np.asarray(sampled_source_indices, dtype=np.int64).reshape(-1)
    if sampled.size and (np.min(sampled) < 0 or np.max(sampled) >= source.xyz.shape[0]):
        raise ValueError("sampled_source_indices contains out-of-range source rows")
    row_by_gaussian_id = {
        int(gaussian_id): int(row)
        for row, gaussian_id in enumerate(np.asarray(anchor_map.source_gaussian_indices, dtype=np.int64).tolist())
    }
    selected_rows = []
    missing = 0
    for source_row in sampled.tolist():
        gaussian_id = int(source.gaussian_indices[int(source_row)])
        row = row_by_gaussian_id.get(gaussian_id)
        if row is None:
            missing += 1
            continue
        selected_rows.append(row)
    subset = anchor_map.subset(np.asarray(selected_rows, dtype=np.int64))
    subset.metadata.update(
        {
            "requested_sampled_gaussian_count": int(sampled.size),
            "missing_after_union_aggregation": int(missing),
            **dict(metadata or {}),
        }
    )
    return subset


def project_gaussian_anchor_map_features(
    anchor_map: SemiDenseAnchorMap,
    selector,
    output_dim: int,
    device: str = "cpu",
    batch_size: int = 65536,
) -> SemiDenseAnchorMap:
    encoded = selector.encode_rows(anchor_map.features, device=device, batch_size=int(batch_size))
    encoded = np.asarray(encoded, dtype=np.float32)
    if encoded.shape != (len(anchor_map), int(output_dim)):
        raise ValueError("selector returned an unexpected descriptor shape")
    encoded, _valid = normalize_rows(encoded)
    metadata = dict(anchor_map.metadata or {})
    metadata.update(
        {
            "stage": "stage_h2_selector_projected_gaussian_anchor_map",
            "source_feature_dim": int(anchor_map.feature_dim),
            "output_feature_dim": int(output_dim),
        }
    )
    return SemiDenseAnchorMap(
        anchor_ids=anchor_map.anchor_ids,
        xyz=anchor_map.xyz,
        features=encoded.astype(np.float32, copy=False),
        source_types=anchor_map.source_types,
        source_track_ids=anchor_map.source_track_ids,
        source_gaussian_indices=anchor_map.source_gaussian_indices,
        support_counts=anchor_map.support_counts,
        mean_distances=anchor_map.mean_distances,
        feature_variances=anchor_map.feature_variances,
        observation_counts=anchor_map.observation_counts,
        visibility_counts=anchor_map.visibility_counts,
        quality_scores=anchor_map.quality_scores,
        opacity=anchor_map.opacity,
        scale=anchor_map.scale,
        observation_image_ids=anchor_map.observation_image_ids,
        metadata=metadata,
    )
