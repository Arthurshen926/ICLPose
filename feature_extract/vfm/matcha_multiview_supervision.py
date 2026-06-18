"""3DGS multiview geometry supervision for RADIO-adapted MATCHA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.matcha_coarse_supervision import (
    MatchaCoarseSupervision,
    MatchaCoarseSupervisionConfig,
    _cell_indices,
    _grid_cell_centers,
    _patch_overlap_proxy,
    _project_world_to_image,
    _sample_depth,
    _sample_depth_edge,
    _sample_optional_scalar_map,
    cell_offset_labels,
    cell_offset_soft_labels,
)
from feature_extract.vfm.rendered_keypoint_selector_samples import render_keypoints_to_world


@dataclass(frozen=True)
class MatchaGeometryView:
    camera: ColmapCamera
    pose_w2c: np.ndarray
    depth: np.ndarray
    alpha: np.ndarray | None = None
    view_id: str = ""

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose_w2c, dtype=np.float64).reshape(4, 4)
        depth = np.asarray(self.depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError("depth must have shape (H, W)")
        alpha = None if self.alpha is None else np.asarray(self.alpha, dtype=np.float32)
        if alpha is not None and alpha.shape != depth.shape:
            raise ValueError("alpha must have the same shape as depth")
        object.__setattr__(self, "pose_w2c", pose)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "view_id", str(self.view_id))


@dataclass(frozen=True)
class MatchaMultiviewSupervisionConfig:
    min_support_views: int = 1
    support_depth_tolerance_m: float = 0.05
    coarse_config: MatchaCoarseSupervisionConfig | Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if int(self.min_support_views) < 0:
            raise ValueError("min_support_views must be non-negative")
        if float(self.support_depth_tolerance_m) <= 0.0:
            raise ValueError("support_depth_tolerance_m must be positive")
        if self.coarse_config is None:
            coarse = MatchaCoarseSupervisionConfig()
        elif isinstance(self.coarse_config, MatchaCoarseSupervisionConfig):
            coarse = self.coarse_config
        else:
            coarse = MatchaCoarseSupervisionConfig(**dict(self.coarse_config))
        object.__setattr__(self, "min_support_views", int(self.min_support_views))
        object.__setattr__(self, "support_depth_tolerance_m", float(self.support_depth_tolerance_m))
        object.__setattr__(self, "coarse_config", coarse)


def _empty_multiview_supervision(
    *,
    no_match_items: Sequence[tuple[int, float, int, int, int, int, float, bool]] = (),
) -> MatchaCoarseSupervision:
    return MatchaCoarseSupervision(
        query_indices=np.zeros((0,), dtype=np.int64),
        render_indices=np.zeros((0,), dtype=np.int64),
        query_xy=np.zeros((0, 2), dtype=np.float64),
        render_xy=np.zeros((0, 2), dtype=np.float64),
        query_offset_labels=np.zeros((0,), dtype=np.int64),
        render_offset_labels=np.zeros((0,), dtype=np.int64),
        roundtrip_errors_px=np.zeros((0,), dtype=np.float32),
        no_match_query_indices=np.asarray([item[2] for item in no_match_items], dtype=np.int64),
        no_match_render_indices=np.asarray([item[3] for item in no_match_items], dtype=np.int64),
        no_match_query_offset_labels=np.asarray([item[4] for item in no_match_items], dtype=np.int64),
        no_match_render_offset_labels=np.asarray([item[5] for item in no_match_items], dtype=np.int64),
        no_match_roundtrip_errors_px=np.asarray([item[1] for item in no_match_items], dtype=np.float32),
        no_match_reason_ids=np.asarray([item[0] for item in no_match_items], dtype=np.int64),
        no_match_confidence_targets=np.asarray([item[6] for item in no_match_items], dtype=np.float32),
        no_match_confidence_ignore_mask=np.asarray([item[7] for item in no_match_items], dtype=bool),
        source="geometry_3dgs_multiview",
        support_view_counts=np.zeros((0,), dtype=np.int64),
    )


def _camera_depth(points_xyz: np.ndarray, view: MatchaGeometryView) -> np.ndarray:
    pose = np.asarray(view.pose_w2c, dtype=np.float64).reshape(4, 4)
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    camera_xyz = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    return camera_xyz[:, 2].astype(np.float64, copy=False)


def _support_view_agreement(
    points_xyz: np.ndarray,
    support_views: Sequence[MatchaGeometryView],
    *,
    coarse_config: MatchaCoarseSupervisionConfig,
    support_depth_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    counts = np.zeros((points.shape[0],), dtype=np.int64)
    error_sum = np.zeros((points.shape[0],), dtype=np.float32)
    if not support_views:
        return counts, error_sum
    for view in support_views:
        support_xy, support_proj_valid = _project_world_to_image(points, view.camera, view.pose_w2c)
        support_depth, support_depth_valid = _sample_depth(
            view.depth,
            support_xy,
            image_width=int(view.camera.width),
            image_height=int(view.camera.height),
        )
        projected_depth = _camera_depth(points, view)
        depth_delta = np.abs(support_depth - projected_depth).astype(np.float32)
        support_xyz, support_xyz_valid = render_keypoints_to_world(
            support_xy,
            view.depth,
            view.camera,
            view.pose_w2c,
            render_width=int(view.camera.width),
            render_height=int(view.camera.height),
        )
        world_delta = np.linalg.norm(support_xyz - points, axis=1).astype(np.float32)
        support_alpha, support_alpha_valid = _sample_optional_scalar_map(
            view.alpha,
            support_xy,
            image_width=int(view.camera.width),
            image_height=int(view.camera.height),
        )
        alpha_valid = support_alpha_valid
        if float(coarse_config.alpha_threshold) > 0.0:
            alpha_valid &= support_alpha >= float(coarse_config.alpha_threshold)
        edge_valid = np.ones((points.shape[0],), dtype=bool)
        if float(coarse_config.depth_edge_threshold_m) > 0.0:
            support_edge = _sample_depth_edge(
                view.depth,
                support_xy,
                image_width=int(view.camera.width),
                image_height=int(view.camera.height),
            )
            edge_valid &= support_edge <= float(coarse_config.depth_edge_threshold_m)
        valid = (
            support_proj_valid
            & support_depth_valid
            & support_xyz_valid
            & np.isfinite(projected_depth)
            & (projected_depth > float(coarse_config.min_depth))
            & np.isfinite(depth_delta)
            & np.isfinite(world_delta)
            & (depth_delta <= float(support_depth_tolerance_m))
            & (world_delta <= float(support_depth_tolerance_m))
            & alpha_valid
            & edge_valid
        )
        counts += valid.astype(np.int64)
        error_sum += np.where(valid, depth_delta + world_delta, 0.0).astype(np.float32)
    return counts, error_sum


def build_matcha_3dgs_multiview_coarse_supervision(
    *,
    query_view: MatchaGeometryView,
    render_view: MatchaGeometryView,
    support_views: Sequence[MatchaGeometryView],
    query_grid_hw: tuple[int, int],
    render_grid_hw: tuple[int, int],
    render_seed_xy: np.ndarray | None = None,
    config: MatchaMultiviewSupervisionConfig | None = None,
) -> MatchaCoarseSupervision:
    """Build MATCHA cell supervision by aggregating 3DGS-rendered multiview geometry."""

    mv_cfg = config or MatchaMultiviewSupervisionConfig()
    cfg = mv_cfg.coarse_config
    render_grid_h, render_grid_w = int(render_grid_hw[0]), int(render_grid_hw[1])
    query_grid_h, query_grid_w = int(query_grid_hw[0]), int(query_grid_hw[1])
    if render_seed_xy is None:
        render_xy = _grid_cell_centers(
            image_width=int(render_view.camera.width),
            image_height=int(render_view.camera.height),
            grid_width=render_grid_w,
            grid_height=render_grid_h,
        )
        render_indices = np.arange(render_xy.shape[0], dtype=np.int64)
        render_cell_valid = np.ones((render_xy.shape[0],), dtype=bool)
    else:
        render_xy = np.asarray(render_seed_xy, dtype=np.float64).reshape(-1, 2)
        if render_xy.shape[0] == 0:
            return _empty_multiview_supervision()
        render_indices, render_cell_valid = _cell_indices(
            render_xy,
            image_width=int(render_view.camera.width),
            image_height=int(render_view.camera.height),
            grid_width=render_grid_w,
            grid_height=render_grid_h,
        )

    render_xyz, render_depth_valid = render_keypoints_to_world(
        render_xy,
        render_view.depth,
        render_view.camera,
        render_view.pose_w2c,
        render_width=int(render_view.camera.width),
        render_height=int(render_view.camera.height),
    )
    query_xy, query_proj_valid = _project_world_to_image(render_xyz, query_view.camera, query_view.pose_w2c)
    query_indices, query_cell_valid = _cell_indices(
        query_xy,
        image_width=int(query_view.camera.width),
        image_height=int(query_view.camera.height),
        grid_width=query_grid_w,
        grid_height=query_grid_h,
    )
    query_depth_values, query_depth_valid = _sample_depth(
        query_view.depth,
        query_xy,
        image_width=int(query_view.camera.width),
        image_height=int(query_view.camera.height),
    )
    query_depth_valid = query_depth_valid & (query_depth_values > float(cfg.min_depth))
    render_alpha, render_alpha_valid = _sample_optional_scalar_map(
        render_view.alpha,
        render_xy,
        image_width=int(render_view.camera.width),
        image_height=int(render_view.camera.height),
    )
    query_alpha, query_alpha_valid = _sample_optional_scalar_map(
        query_view.alpha,
        query_xy,
        image_width=int(query_view.camera.width),
        image_height=int(query_view.camera.height),
    )
    alpha_valid = render_alpha_valid & query_alpha_valid
    if float(cfg.alpha_threshold) > 0.0:
        alpha_valid &= (render_alpha >= float(cfg.alpha_threshold)) & (query_alpha >= float(cfg.alpha_threshold))

    query_xyz, query_xyz_valid = render_keypoints_to_world(
        query_xy,
        query_view.depth,
        query_view.camera,
        query_view.pose_w2c,
        render_width=int(query_view.camera.width),
        render_height=int(query_view.camera.height),
    )
    render_xy_roundtrip, render_roundtrip_valid = _project_world_to_image(query_xyz, render_view.camera, render_view.pose_w2c)
    roundtrip = np.linalg.norm(render_xy_roundtrip - render_xy, axis=1).astype(np.float32)
    render_depth_edge = _sample_depth_edge(
        render_view.depth,
        render_xy,
        image_width=int(render_view.camera.width),
        image_height=int(render_view.camera.height),
    )
    query_depth_edge = _sample_depth_edge(
        query_view.depth,
        query_xy,
        image_width=int(query_view.camera.width),
        image_height=int(query_view.camera.height),
    )
    depth_edge_valid = np.ones((render_xy.shape[0],), dtype=bool)
    if float(cfg.depth_edge_threshold_m) > 0.0:
        depth_edge_valid &= (render_depth_edge <= float(cfg.depth_edge_threshold_m)) & (
            query_depth_edge <= float(cfg.depth_edge_threshold_m)
        )

    render_labels, render_label_valid = cell_offset_labels(
        render_xy,
        image_width=int(render_view.camera.width),
        image_height=int(render_view.camera.height),
        grid_width=render_grid_w,
        grid_height=render_grid_h,
        offset_bins=int(cfg.offset_bins),
    )
    query_labels, query_label_valid = cell_offset_labels(
        query_xy,
        image_width=int(query_view.camera.width),
        image_height=int(query_view.camera.height),
        grid_width=query_grid_w,
        grid_height=query_grid_h,
        offset_bins=int(cfg.offset_bins),
    )
    render_soft, _render_soft_valid = cell_offset_soft_labels(
        render_xy,
        image_width=int(render_view.camera.width),
        image_height=int(render_view.camera.height),
        grid_width=render_grid_w,
        grid_height=render_grid_h,
        offset_bins=int(cfg.offset_bins),
        sigma_bins=float(cfg.soft_offset_sigma_bins),
    )
    query_soft, _query_soft_valid = cell_offset_soft_labels(
        query_xy,
        image_width=int(query_view.camera.width),
        image_height=int(query_view.camera.height),
        grid_width=query_grid_w,
        grid_height=query_grid_h,
        offset_bins=int(cfg.offset_bins),
        sigma_bins=float(cfg.soft_offset_sigma_bins),
    )
    support_counts, support_error_sum = _support_view_agreement(
        render_xyz,
        tuple(support_views),
        coarse_config=cfg,
        support_depth_tolerance_m=float(mv_cfg.support_depth_tolerance_m),
    )
    support_valid = support_counts >= int(mv_cfg.min_support_views)
    geometry_candidate = (
        render_cell_valid
        & render_depth_valid
        & query_proj_valid
        & query_cell_valid
        & render_label_valid
        & query_label_valid
        & np.isfinite(roundtrip)
    )
    visibility_valid = (
        geometry_candidate
        & query_depth_valid
        & query_xyz_valid
        & render_roundtrip_valid
        & alpha_valid
        & depth_edge_valid
    )
    valid = visibility_valid & support_valid & (roundtrip <= float(cfg.roundtrip_threshold_px))

    no_match_items: list[tuple[int, float, int, int, int, int, float, bool]] = []
    if bool(cfg.collect_no_match) and int(cfg.max_no_match) > 0:
        no_match_mask = geometry_candidate & ~valid
        reason_ids = np.zeros((render_xy.shape[0],), dtype=np.int64)
        reason_ids[~alpha_valid] = 1
        reason_ids[(reason_ids == 0) & ~depth_edge_valid] = 2
        reason_ids[(reason_ids == 0) & (~query_depth_valid | ~query_xyz_valid | ~render_roundtrip_valid | (roundtrip > float(cfg.roundtrip_threshold_px)))] = 3
        reason_ids[(reason_ids == 0) & ~support_valid] = 5
        reason_ids[(reason_ids == 0) & no_match_mask] = 4
        for idx in np.flatnonzero(no_match_mask).tolist():
            no_match_items.append(
                (
                    int(reason_ids[int(idx)]),
                    float(roundtrip[int(idx)]) if np.isfinite(roundtrip[int(idx)]) else float("inf"),
                    int(query_indices[int(idx)]),
                    int(render_indices[int(idx)]),
                    int(query_labels[int(idx)]),
                    int(render_labels[int(idx)]),
                    0.0,
                    False,
                )
            )
        no_match_items.sort(key=lambda item: (item[0], item[2], item[3], item[1]))
        no_match_items = no_match_items[: int(cfg.max_no_match)]
    if not np.any(valid):
        return _empty_multiview_supervision(no_match_items=no_match_items)

    roundtrip_threshold = max(float(cfg.roundtrip_threshold_px), 1e-6)
    query_overlap = _patch_overlap_proxy(
        query_xy,
        image_width=int(query_view.camera.width),
        image_height=int(query_view.camera.height),
        grid_width=query_grid_w,
        grid_height=query_grid_h,
    )
    render_overlap = _patch_overlap_proxy(
        render_xy,
        image_width=int(render_view.camera.width),
        image_height=int(render_view.camera.height),
        grid_width=render_grid_w,
        grid_height=render_grid_h,
    )
    roundtrip_weight = np.clip(1.0 - roundtrip / roundtrip_threshold, 0.0, 1.0)
    edge_weight = np.ones((render_xy.shape[0],), dtype=np.float32)
    if float(cfg.depth_edge_threshold_m) > 0.0:
        edge_weight = np.clip(
            1.0 - np.maximum(render_depth_edge, query_depth_edge) / max(float(cfg.depth_edge_threshold_m), 1e-6),
            0.0,
            1.0,
        ).astype(np.float32)
    if len(support_views) > 0:
        support_fraction = np.clip(support_counts.astype(np.float32) / float(len(support_views)), 0.0, 1.0)
    else:
        support_fraction = np.ones((render_xy.shape[0],), dtype=np.float32)
    confidence = np.clip(
        roundtrip_weight * edge_weight * support_fraction * (0.5 + 0.5 * np.minimum(query_overlap, render_overlap)),
        0.0,
        1.0,
    )
    confidence_ignore = np.zeros((render_xy.shape[0],), dtype=bool)
    if bool(cfg.pose_confidence_labels):
        pose_positive = (roundtrip <= float(cfg.pose_confidence_positive_threshold_px)) & visibility_valid & support_valid
        pose_negative = (roundtrip > float(cfg.pose_confidence_negative_threshold_px)) | ~visibility_valid | ~support_valid
        confidence = pose_positive.astype(np.float32, copy=False)
        confidence_ignore = ~(pose_positive | pose_negative)
    support_mean_error = support_error_sum / np.maximum(support_counts.astype(np.float32), 1.0)
    support_shortfall = np.maximum(int(mv_cfg.min_support_views) - support_counts, 0).astype(np.float32)
    uncertainty = (
        roundtrip.astype(np.float32)
        + np.maximum(render_depth_edge, query_depth_edge).astype(np.float32)
        + (1.0 - np.minimum(query_overlap, render_overlap).astype(np.float32))
        + support_mean_error.astype(np.float32)
        + support_shortfall
    )

    candidates = []
    for idx in np.flatnonzero(valid).tolist():
        candidates.append(
            (
                int(support_counts[int(idx)]),
                float(support_mean_error[int(idx)]),
                float(roundtrip[int(idx)]),
                int(query_indices[int(idx)]),
                int(render_indices[int(idx)]),
                query_xy[int(idx)],
                render_xy[int(idx)],
                int(query_labels[int(idx)]),
                int(render_labels[int(idx)]),
                query_soft[int(idx)],
                render_soft[int(idx)],
                float(confidence[int(idx)]),
                bool(confidence_ignore[int(idx)]),
                float(uncertainty[int(idx)]),
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    used_query: set[int] = set()
    used_render: set[int] = set()
    kept = []
    for item in candidates:
        query_idx, render_idx = int(item[3]), int(item[4])
        if query_idx in used_query or render_idx in used_render:
            continue
        used_query.add(query_idx)
        used_render.add(render_idx)
        kept.append(item)
    if not kept:
        return _empty_multiview_supervision(no_match_items=no_match_items)
    kept.sort(key=lambda item: (item[3], item[4]))
    return MatchaCoarseSupervision(
        query_indices=np.asarray([item[3] for item in kept], dtype=np.int64),
        render_indices=np.asarray([item[4] for item in kept], dtype=np.int64),
        query_xy=np.stack([item[5] for item in kept], axis=0),
        render_xy=np.stack([item[6] for item in kept], axis=0),
        query_offset_labels=np.asarray([item[7] for item in kept], dtype=np.int64),
        render_offset_labels=np.asarray([item[8] for item in kept], dtype=np.int64),
        roundtrip_errors_px=np.asarray([item[2] for item in kept], dtype=np.float32),
        query_offset_soft_labels=np.stack([item[9] for item in kept], axis=0).astype(np.float32, copy=False),
        render_offset_soft_labels=np.stack([item[10] for item in kept], axis=0).astype(np.float32, copy=False),
        confidence_targets=np.asarray([item[11] for item in kept], dtype=np.float32),
        confidence_ignore_mask=np.asarray([item[12] for item in kept], dtype=bool),
        uncertainty_px=np.asarray([item[13] for item in kept], dtype=np.float32),
        no_match_query_indices=np.asarray([item[2] for item in no_match_items], dtype=np.int64),
        no_match_render_indices=np.asarray([item[3] for item in no_match_items], dtype=np.int64),
        no_match_query_offset_labels=np.asarray([item[4] for item in no_match_items], dtype=np.int64),
        no_match_render_offset_labels=np.asarray([item[5] for item in no_match_items], dtype=np.int64),
        no_match_roundtrip_errors_px=np.asarray([item[1] for item in no_match_items], dtype=np.float32),
        no_match_reason_ids=np.asarray([item[0] for item in no_match_items], dtype=np.int64),
        no_match_confidence_targets=np.asarray([item[6] for item in no_match_items], dtype=np.float32),
        no_match_confidence_ignore_mask=np.asarray([item[7] for item in no_match_items], dtype=bool),
        source="geometry_3dgs_multiview",
        support_view_counts=np.asarray([item[0] for item in kept], dtype=np.int64),
    )
