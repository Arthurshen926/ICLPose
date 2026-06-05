"""Raw VFM-2DGS feature rendering diagnostics.

This module intentionally stays selector-free. It renders already-aggregated
raw VFM anchor features from a VFM-2DGS anchor map into a query camera view
using the anchor's 2DGS surface support footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.gaussian_raw_landmarks import _project_xyz_to_grid
from feature_extract.vfm.query_to_3d_matching import normalize_rows
from feature_extract.vfm.vfm_2dgs_mapping import SurfaceElementMap, Vfm2DgsAnchorMap


@dataclass(frozen=True)
class Vfm2DgsRenderConfig:
    width: int
    height: int
    radius_scale: float = 1.0
    min_radius_px: float = 0.5
    max_radius_px: float = 4.0
    depth_epsilon: float = 0.25
    min_weight: float = 1e-8
    normalize_features: bool = True

    def __post_init__(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("render width and height must be positive")
        if float(self.radius_scale) <= 0.0:
            raise ValueError("radius_scale must be positive")
        if float(self.min_radius_px) < 0.0:
            raise ValueError("min_radius_px must be non-negative")
        if float(self.max_radius_px) <= 0.0:
            raise ValueError("max_radius_px must be positive")
        if float(self.depth_epsilon) < 0.0:
            raise ValueError("depth_epsilon must be non-negative")


@dataclass(frozen=True)
class RenderedVfm2DgsFeatureMap:
    feature_map: np.ndarray
    xyz_map: np.ndarray
    visibility_mask: np.ndarray
    alpha_map: np.ndarray
    depth_map: np.ndarray
    variance_map: np.ndarray
    support_count_map: np.ndarray

    def __post_init__(self) -> None:
        feature_map = np.asarray(self.feature_map, dtype=np.float32)
        if feature_map.ndim != 3:
            raise ValueError("feature_map must have shape (C, H, W)")
        channels, height, width = feature_map.shape
        xyz_map = np.asarray(self.xyz_map, dtype=np.float32)
        if xyz_map.shape != (height, width, 3):
            raise ValueError("xyz_map must have shape (H, W, 3)")
        for name in ("visibility_mask", "alpha_map", "depth_map", "variance_map", "support_count_map"):
            value = np.asarray(getattr(self, name))
            if value.shape != (height, width):
                raise ValueError(f"{name} must have shape (H, W)")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "feature_map", feature_map)
        object.__setattr__(self, "xyz_map", xyz_map)
        object.__setattr__(self, "visibility_mask", np.asarray(self.visibility_mask, dtype=bool))
        object.__setattr__(self, "alpha_map", np.asarray(self.alpha_map, dtype=np.float32))
        object.__setattr__(self, "depth_map", np.asarray(self.depth_map, dtype=np.float32))
        object.__setattr__(self, "variance_map", np.asarray(self.variance_map, dtype=np.float32))
        object.__setattr__(self, "support_count_map", np.asarray(self.support_count_map, dtype=np.int64))


def _camera_focal_for_render_grid(camera, width: int, height: int) -> float:
    params = tuple(float(value) for value in camera.params)
    model_id = int(camera.model_id)
    if model_id in {0, 2, 3} and len(params) >= 1:
        fx = fy = params[0]
    elif model_id in {1, 4} and len(params) >= 2:
        fx, fy = params[:2]
    elif len(params) >= 1:
        fx = fy = params[0]
    else:
        fx = fy = max(float(camera.width), float(camera.height))
    scale_x = float(width) / max(float(camera.width), 1.0)
    scale_y = float(height) / max(float(camera.height), 1.0)
    return float(0.5 * (abs(fx) * scale_x + abs(fy) * scale_y))


def _empty_render(feature_dim: int, cfg: Vfm2DgsRenderConfig) -> RenderedVfm2DgsFeatureMap:
    return RenderedVfm2DgsFeatureMap(
        feature_map=np.zeros((int(feature_dim), int(cfg.height), int(cfg.width)), dtype=np.float32),
        xyz_map=np.zeros((int(cfg.height), int(cfg.width), 3), dtype=np.float32),
        visibility_mask=np.zeros((int(cfg.height), int(cfg.width)), dtype=bool),
        alpha_map=np.zeros((int(cfg.height), int(cfg.width)), dtype=np.float32),
        depth_map=np.full((int(cfg.height), int(cfg.width)), np.inf, dtype=np.float32),
        variance_map=np.zeros((int(cfg.height), int(cfg.width)), dtype=np.float32),
        support_count_map=np.zeros((int(cfg.height), int(cfg.width)), dtype=np.int64),
    )


def render_vfm_2dgs_anchor_features(
    anchor_map: Vfm2DgsAnchorMap,
    surface_elements: SurfaceElementMap,
    pose_w2c: np.ndarray,
    camera,
    config: Vfm2DgsRenderConfig,
) -> RenderedVfm2DgsFeatureMap:
    """Render raw VFM anchor features with 2DGS surface-support footprints."""

    cfg = config
    feature_dim = int(anchor_map.feature_dim)
    if len(anchor_map) == 0 or len(surface_elements) == 0:
        return _empty_render(feature_dim, cfg)
    row_by_element = surface_elements.row_by_element_id
    focal_grid = _camera_focal_for_render_grid(camera, int(cfg.width), int(cfg.height))
    element_uv, element_depth = _project_xyz_to_grid(
        surface_elements.centers,
        pose_w2c,
        camera,
        int(cfg.width),
        int(cfg.height),
    )

    pixel_ids: list[int] = []
    anchor_rows: list[int] = []
    weights: list[float] = []
    depths: list[float] = []
    for anchor_row in range(len(anchor_map)):
        start = int(anchor_map.support_offsets[anchor_row])
        end = int(anchor_map.support_offsets[anchor_row + 1])
        support_ids = anchor_map.support_element_ids[start:end]
        support_weights = anchor_map.support_weights[start:end]
        if support_ids.size == 0:
            continue
        for element_id, support_weight in zip(support_ids.tolist(), support_weights.tolist()):
            element_row = row_by_element.get(int(element_id))
            if element_row is None:
                continue
            depth = float(element_depth[element_row])
            uv = element_uv[element_row]
            if not np.isfinite(depth) or depth <= 1e-8 or not np.all(np.isfinite(uv)):
                continue
            if uv[0] < -float(cfg.max_radius_px) or uv[0] > float(cfg.width - 1) + float(cfg.max_radius_px):
                continue
            if uv[1] < -float(cfg.max_radius_px) or uv[1] > float(cfg.height - 1) + float(cfg.max_radius_px):
                continue
            surface_radius = (
                max(float(surface_elements.scale1[element_row]), float(surface_elements.scale2[element_row]))
                * focal_grid
                / max(depth, 1e-8)
                * float(cfg.radius_scale)
            )
            radius = float(np.clip(surface_radius, float(cfg.min_radius_px), float(cfg.max_radius_px)))
            sigma_sq = max((0.5 * radius) ** 2, 1e-8)
            x0 = max(0, int(np.floor(float(uv[0]) - radius)))
            x1 = min(int(cfg.width) - 1, int(np.ceil(float(uv[0]) + radius)))
            y0 = max(0, int(np.floor(float(uv[1]) - radius)))
            y1 = min(int(cfg.height) - 1, int(np.ceil(float(uv[1]) + radius)))
            if x1 < x0 or y1 < y0:
                continue
            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    dist_sq = (float(xx) - float(uv[0])) ** 2 + (float(yy) - float(uv[1])) ** 2
                    if dist_sq > radius * radius:
                        continue
                    weight = (
                        float(support_weight)
                        * float(surface_elements.opacity[element_row])
                        * float(np.exp(-0.5 * dist_sq / sigma_sq))
                    )
                    if weight <= float(cfg.min_weight):
                        continue
                    pixel_ids.append(int(yy) * int(cfg.width) + int(xx))
                    anchor_rows.append(int(anchor_row))
                    weights.append(weight)
                    depths.append(depth)

    if not pixel_ids:
        return _empty_render(feature_dim, cfg)
    pixel_ids_arr = np.asarray(pixel_ids, dtype=np.int64)
    anchor_rows_arr = np.asarray(anchor_rows, dtype=np.int64)
    weights_arr = np.asarray(weights, dtype=np.float64)
    depths_arr = np.asarray(depths, dtype=np.float64)

    min_depth = np.full((int(cfg.height) * int(cfg.width),), np.inf, dtype=np.float64)
    np.minimum.at(min_depth, pixel_ids_arr, depths_arr)
    keep = depths_arr <= (min_depth[pixel_ids_arr] + float(cfg.depth_epsilon))
    pixel_ids_arr = pixel_ids_arr[keep]
    anchor_rows_arr = anchor_rows_arr[keep]
    weights_arr = weights_arr[keep]
    depths_arr = depths_arr[keep]
    if pixel_ids_arr.size == 0:
        return _empty_render(feature_dim, cfg)

    flat_count = int(cfg.height) * int(cfg.width)
    weight_sum = np.zeros((flat_count,), dtype=np.float64)
    support_count = np.zeros((flat_count,), dtype=np.int64)
    feature_sum = np.zeros((flat_count, feature_dim), dtype=np.float64)
    xyz_sum = np.zeros((flat_count, 3), dtype=np.float64)
    variance_sum = np.zeros((flat_count,), dtype=np.float64)
    depth_sum = np.zeros((flat_count,), dtype=np.float64)
    np.add.at(weight_sum, pixel_ids_arr, weights_arr)
    np.add.at(support_count, pixel_ids_arr, 1)
    np.add.at(feature_sum, pixel_ids_arr, anchor_map.features[anchor_rows_arr].astype(np.float64) * weights_arr[:, None])
    np.add.at(xyz_sum, pixel_ids_arr, anchor_map.centers[anchor_rows_arr].astype(np.float64) * weights_arr[:, None])
    np.add.at(variance_sum, pixel_ids_arr, anchor_map.feature_variances[anchor_rows_arr].astype(np.float64) * weights_arr)
    np.add.at(depth_sum, pixel_ids_arr, depths_arr * weights_arr)
    visible_flat = weight_sum > float(cfg.min_weight)
    feature_flat = np.zeros((flat_count, feature_dim), dtype=np.float32)
    xyz_flat = np.zeros((flat_count, 3), dtype=np.float32)
    variance_flat = np.zeros((flat_count,), dtype=np.float32)
    depth_flat = np.full((flat_count,), np.inf, dtype=np.float32)
    if np.any(visible_flat):
        denom = np.maximum(weight_sum[visible_flat], 1e-12)
        feature_flat[visible_flat] = (feature_sum[visible_flat] / denom[:, None]).astype(np.float32)
        if bool(cfg.normalize_features):
            feature_flat[visible_flat], _valid = normalize_rows(feature_flat[visible_flat])
        xyz_flat[visible_flat] = (xyz_sum[visible_flat] / denom[:, None]).astype(np.float32)
        variance_flat[visible_flat] = (variance_sum[visible_flat] / denom).astype(np.float32)
        depth_flat[visible_flat] = (depth_sum[visible_flat] / denom).astype(np.float32)
    return RenderedVfm2DgsFeatureMap(
        feature_map=feature_flat.reshape(int(cfg.height), int(cfg.width), feature_dim).transpose(2, 0, 1),
        xyz_map=xyz_flat.reshape(int(cfg.height), int(cfg.width), 3),
        visibility_mask=visible_flat.reshape(int(cfg.height), int(cfg.width)),
        alpha_map=np.clip(weight_sum.reshape(int(cfg.height), int(cfg.width)), 0.0, 1.0).astype(np.float32),
        depth_map=depth_flat.reshape(int(cfg.height), int(cfg.width)),
        variance_map=variance_flat.reshape(int(cfg.height), int(cfg.width)),
        support_count_map=support_count.reshape(int(cfg.height), int(cfg.width)),
    )


def evaluate_gt_aligned_render_features(
    query_feature_map: np.ndarray,
    rendered: RenderedVfm2DgsFeatureMap,
    top_k: int = 5,
    block_size: int = 512,
) -> dict[str, float | int | None]:
    """Compare query tokens with rendered map features at the same token position."""

    query = np.asarray(query_feature_map, dtype=np.float32)
    if query.shape != rendered.feature_map.shape:
        raise ValueError("query_feature_map and rendered feature_map must have identical shape")
    channels, height, width = query.shape
    query_flat = query.reshape(channels, -1).T
    render_flat = rendered.feature_map.reshape(channels, -1).T
    query_flat, query_valid = normalize_rows(query_flat)
    render_flat, render_valid = normalize_rows(render_flat)
    visible = rendered.visibility_mask.reshape(-1) & query_valid & render_valid
    visible_indices = np.flatnonzero(visible)
    if visible_indices.size == 0:
        return {
            "visible_token_count": 0,
            "visible_fraction": 0.0,
            "same_cosine_mean": None,
            "same_cosine_median": None,
            "negative_cosine_mean": None,
            "cosine_margin_mean": None,
            "same_gt_negative_win_rate": None,
            "top1_exact": None,
            "top1_within1": None,
            f"top{int(top_k)}_exact": None,
            f"top{int(top_k)}_within1": None,
        }
    same = np.sum(query_flat[visible_indices] * render_flat[visible_indices], axis=1)
    negative_indices = np.roll(visible_indices, max(1, visible_indices.size // 3))
    negative = np.sum(query_flat[visible_indices] * render_flat[negative_indices], axis=1)

    render_visible = render_flat[visible_indices]
    render_positions = np.stack([visible_indices % int(width), visible_indices // int(width)], axis=1)
    top1_exact = []
    top1_within1 = []
    topk_exact = []
    topk_within1 = []
    keep = min(max(int(top_k), 1), int(visible_indices.size))
    for start in range(0, visible_indices.size, int(block_size)):
        end = min(start + int(block_size), visible_indices.size)
        scores = query_flat[visible_indices[start:end]] @ render_visible.T
        if keep == 1:
            cols = np.argmax(scores, axis=1)[:, None]
        else:
            cols = np.argpartition(-scores, kth=keep - 1, axis=1)[:, :keep]
            local_scores = np.take_along_axis(scores, cols, axis=1)
            order = np.argsort(-local_scores, axis=1)
            cols = np.take_along_axis(cols, order, axis=1)
        target = np.arange(start, end, dtype=np.int64)
        top1 = cols[:, 0]
        qpos = render_positions[target]
        top1_pos = render_positions[top1]
        top1_exact.extend((top1 == target).tolist())
        top1_within1.extend((np.max(np.abs(top1_pos - qpos), axis=1) <= 1).tolist())
        topk_exact.extend([bool(row_target in row.tolist()) for row_target, row in zip(target.tolist(), cols)])
        for row_target, row in zip(target.tolist(), cols):
            positions = render_positions[row]
            topk_within1.append(bool(np.any(np.max(np.abs(positions - render_positions[row_target]), axis=1) <= 1)))
    return {
        "visible_token_count": int(visible_indices.size),
        "visible_fraction": float(visible_indices.size / max(height * width, 1)),
        "same_cosine_mean": float(np.mean(same)),
        "same_cosine_median": float(np.median(same)),
        "same_cosine_p10": float(np.percentile(same, 10.0)),
        "same_cosine_p90": float(np.percentile(same, 90.0)),
        "negative_cosine_mean": float(np.mean(negative)),
        "negative_cosine_median": float(np.median(negative)),
        "cosine_margin_mean": float(np.mean(same - negative)),
        "same_gt_negative_win_rate": float(np.mean(same > negative)),
        "top1_exact": float(np.mean(top1_exact)),
        "top1_within1": float(np.mean(top1_within1)),
        f"top{int(top_k)}_exact": float(np.mean(topk_exact)),
        f"top{int(top_k)}_within1": float(np.mean(topk_within1)),
    }
