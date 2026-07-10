"""MATCHA-style coarse cell supervision from depth and camera poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion
from feature_extract.vfm.rendered_keypoint_matching import bilinear_sample_feature_map
from feature_extract.vfm.rendered_keypoint_selector_samples import render_keypoints_to_world


_GEOMETRY_SUPERVISION_SOURCES = frozenset({"geometry_depth_pose", "geometry_3dgs_multiview"})


@dataclass(frozen=True)
class MatchaCoarseSupervisionConfig:
    offset_bins: int = 8
    roundtrip_threshold_px: float = 1.5
    min_depth: float = 1e-6
    alpha_threshold: float = 0.0
    depth_edge_threshold_m: float = 0.0
    collect_no_match: bool = False
    max_no_match: int = 256
    soft_offset_sigma_bins: float = 0.75
    pose_confidence_labels: bool = False
    pose_confidence_positive_threshold_px: float = 8.0
    pose_confidence_negative_threshold_px: float = 24.0

    def __post_init__(self) -> None:
        if int(self.offset_bins) <= 0:
            raise ValueError("offset_bins must be positive")
        if float(self.roundtrip_threshold_px) < 0.0:
            raise ValueError("roundtrip_threshold_px must be non-negative")
        if float(self.min_depth) < 0.0:
            raise ValueError("min_depth must be non-negative")
        if float(self.alpha_threshold) < 0.0:
            raise ValueError("alpha_threshold must be non-negative")
        if float(self.depth_edge_threshold_m) < 0.0:
            raise ValueError("depth_edge_threshold_m must be non-negative")
        if int(self.max_no_match) < 0:
            raise ValueError("max_no_match must be non-negative")
        if float(self.soft_offset_sigma_bins) <= 0.0:
            raise ValueError("soft_offset_sigma_bins must be positive")
        if float(self.pose_confidence_positive_threshold_px) < 0.0:
            raise ValueError("pose_confidence_positive_threshold_px must be non-negative")
        if float(self.pose_confidence_negative_threshold_px) < float(self.pose_confidence_positive_threshold_px):
            raise ValueError("pose_confidence_negative_threshold_px must be >= pose_confidence_positive_threshold_px")


@dataclass(frozen=True)
class MatchaCoarseSupervision:
    query_indices: np.ndarray
    render_indices: np.ndarray
    query_xy: np.ndarray
    render_xy: np.ndarray
    query_offset_labels: np.ndarray
    render_offset_labels: np.ndarray
    roundtrip_errors_px: np.ndarray
    query_offset_soft_labels: np.ndarray | None = None
    render_offset_soft_labels: np.ndarray | None = None
    confidence_targets: np.ndarray | None = None
    confidence_ignore_mask: np.ndarray | None = None
    uncertainty_px: np.ndarray | None = None
    no_match_query_indices: np.ndarray | None = None
    no_match_render_indices: np.ndarray | None = None
    no_match_query_offset_labels: np.ndarray | None = None
    no_match_render_offset_labels: np.ndarray | None = None
    no_match_roundtrip_errors_px: np.ndarray | None = None
    no_match_reason_ids: np.ndarray | None = None
    no_match_confidence_targets: np.ndarray | None = None
    no_match_confidence_ignore_mask: np.ndarray | None = None
    source: str = "geometry_depth_pose"
    support_view_counts: np.ndarray | None = None
    track_ids: np.ndarray | None = None
    landmark_xyz: np.ndarray | None = None

    def __post_init__(self) -> None:
        source = str(self.source)
        if "matcher" in source.lower():
            raise ValueError("MATCHA coarse supervision source must be geometry-derived, not matcher-derived")
        if source not in _GEOMETRY_SUPERVISION_SOURCES:
            allowed = ", ".join(sorted(_GEOMETRY_SUPERVISION_SOURCES))
            raise ValueError(f"MATCHA coarse supervision source must be one of: {allowed}")
        object.__setattr__(self, "source", source)

        query_indices = np.asarray(self.query_indices, dtype=np.int64).reshape(-1)
        render_indices = np.asarray(self.render_indices, dtype=np.int64).reshape(-1)
        query_xy = np.asarray(self.query_xy, dtype=np.float64).reshape(-1, 2)
        render_xy = np.asarray(self.render_xy, dtype=np.float64).reshape(-1, 2)
        query_labels = np.asarray(self.query_offset_labels, dtype=np.int64).reshape(-1)
        render_labels = np.asarray(self.render_offset_labels, dtype=np.int64).reshape(-1)
        roundtrip = np.asarray(self.roundtrip_errors_px, dtype=np.float32).reshape(-1)
        count = query_indices.shape[0]
        if render_indices.shape[0] != count or query_xy.shape[0] != count or render_xy.shape[0] != count:
            raise ValueError("all supervision arrays must have the same length")
        if query_labels.shape[0] != count or render_labels.shape[0] != count or roundtrip.shape[0] != count:
            raise ValueError("label and roundtrip arrays must have the same length as indices")
        object.__setattr__(self, "query_indices", query_indices)
        object.__setattr__(self, "render_indices", render_indices)
        object.__setattr__(self, "query_xy", query_xy)
        object.__setattr__(self, "render_xy", render_xy)
        object.__setattr__(self, "query_offset_labels", query_labels)
        object.__setattr__(self, "render_offset_labels", render_labels)
        object.__setattr__(self, "roundtrip_errors_px", roundtrip)
        for name, value in (
            ("query_offset_soft_labels", self.query_offset_soft_labels),
            ("render_offset_soft_labels", self.render_offset_soft_labels),
        ):
            if value is None:
                hard = query_labels if name.startswith("query") else render_labels
                soft = np.zeros((count, 65), dtype=np.float32)
                if count:
                    soft[np.arange(count), np.clip(hard, 0, 64)] = 1.0
                object.__setattr__(self, name, soft)
                continue
            arr = np.asarray(value, dtype=np.float32).reshape(-1, 65)
            if arr.shape[0] != count:
                raise ValueError(f"{name} must contain one distribution per match")
            object.__setattr__(self, name, arr)
        if self.confidence_targets is None:
            confidence = np.ones((count,), dtype=np.float32)
        else:
            confidence = np.asarray(self.confidence_targets, dtype=np.float32).reshape(-1)
            if confidence.shape[0] != count:
                raise ValueError("confidence_targets must contain one value per match")
            confidence = np.clip(confidence, 0.0, 1.0)
        object.__setattr__(self, "confidence_targets", confidence)
        if self.confidence_ignore_mask is None:
            confidence_ignore = np.zeros((count,), dtype=bool)
        else:
            confidence_ignore = np.asarray(self.confidence_ignore_mask, dtype=bool).reshape(-1)
            if confidence_ignore.shape[0] != count:
                raise ValueError("confidence_ignore_mask must contain one value per match")
        object.__setattr__(self, "confidence_ignore_mask", confidence_ignore)
        if self.uncertainty_px is None:
            uncertainty = roundtrip.astype(np.float32, copy=True)
        else:
            uncertainty = np.asarray(self.uncertainty_px, dtype=np.float32).reshape(-1)
            if uncertainty.shape[0] != count:
                raise ValueError("uncertainty_px must contain one value per match")
        object.__setattr__(self, "uncertainty_px", uncertainty)

        no_query = np.zeros((0,), dtype=np.int64) if self.no_match_query_indices is None else np.asarray(self.no_match_query_indices, dtype=np.int64).reshape(-1)
        no_count = int(no_query.shape[0])
        no_render = np.zeros((0,), dtype=np.int64) if self.no_match_render_indices is None else np.asarray(self.no_match_render_indices, dtype=np.int64).reshape(-1)
        no_qlabels = np.full((no_count,), 64, dtype=np.int64) if self.no_match_query_offset_labels is None else np.asarray(self.no_match_query_offset_labels, dtype=np.int64).reshape(-1)
        no_rlabels = np.full((no_count,), 64, dtype=np.int64) if self.no_match_render_offset_labels is None else np.asarray(self.no_match_render_offset_labels, dtype=np.int64).reshape(-1)
        no_roundtrip = np.full((no_count,), np.inf, dtype=np.float32) if self.no_match_roundtrip_errors_px is None else np.asarray(self.no_match_roundtrip_errors_px, dtype=np.float32).reshape(-1)
        no_reason = np.zeros((no_count,), dtype=np.int64) if self.no_match_reason_ids is None else np.asarray(self.no_match_reason_ids, dtype=np.int64).reshape(-1)
        no_confidence = np.zeros((no_count,), dtype=np.float32) if self.no_match_confidence_targets is None else np.asarray(self.no_match_confidence_targets, dtype=np.float32).reshape(-1)
        no_confidence_ignore = np.zeros((no_count,), dtype=bool) if self.no_match_confidence_ignore_mask is None else np.asarray(self.no_match_confidence_ignore_mask, dtype=bool).reshape(-1)
        for name, arr in (
            ("no_match_render_indices", no_render),
            ("no_match_query_offset_labels", no_qlabels),
            ("no_match_render_offset_labels", no_rlabels),
            ("no_match_roundtrip_errors_px", no_roundtrip),
            ("no_match_reason_ids", no_reason),
            ("no_match_confidence_targets", no_confidence),
            ("no_match_confidence_ignore_mask", no_confidence_ignore),
        ):
            if arr.shape[0] != no_count:
                raise ValueError(f"{name} must contain one value per no-match sample")
        object.__setattr__(self, "no_match_query_indices", no_query)
        object.__setattr__(self, "no_match_render_indices", no_render)
        object.__setattr__(self, "no_match_query_offset_labels", no_qlabels)
        object.__setattr__(self, "no_match_render_offset_labels", no_rlabels)
        object.__setattr__(self, "no_match_roundtrip_errors_px", no_roundtrip)
        object.__setattr__(self, "no_match_reason_ids", no_reason)
        object.__setattr__(self, "no_match_confidence_targets", np.clip(no_confidence, 0.0, 1.0))
        object.__setattr__(self, "no_match_confidence_ignore_mask", no_confidence_ignore)
        if self.support_view_counts is None:
            support_counts = np.zeros((count,), dtype=np.int64)
        else:
            support_counts = np.asarray(self.support_view_counts, dtype=np.int64).reshape(-1)
            if support_counts.shape[0] != count:
                raise ValueError("support_view_counts must contain one value per match")
            support_counts = np.maximum(support_counts, 0)
        object.__setattr__(self, "support_view_counts", support_counts)
        if self.track_ids is None:
            track_ids = np.full((count,), -1, dtype=np.int64)
        else:
            track_ids = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
            if track_ids.shape[0] != count:
                raise ValueError("track_ids must contain one value per match")
        object.__setattr__(self, "track_ids", track_ids)
        if self.landmark_xyz is None:
            landmark_xyz = np.full((count, 3), np.nan, dtype=np.float64)
        else:
            landmark_xyz = np.asarray(self.landmark_xyz, dtype=np.float64).reshape(-1, 3)
            if landmark_xyz.shape[0] != count:
                raise ValueError("landmark_xyz must contain one 3D point per match")
        object.__setattr__(self, "landmark_xyz", landmark_xyz)

    @property
    def count(self) -> int:
        return int(self.query_indices.shape[0])

    @property
    def no_match_count(self) -> int:
        return int(np.asarray(self.no_match_query_indices).shape[0])


def _empty_supervision() -> MatchaCoarseSupervision:
    return MatchaCoarseSupervision(
        query_indices=np.zeros((0,), dtype=np.int64),
        render_indices=np.zeros((0,), dtype=np.int64),
        query_xy=np.zeros((0, 2), dtype=np.float64),
        render_xy=np.zeros((0, 2), dtype=np.float64),
        query_offset_labels=np.zeros((0,), dtype=np.int64),
        render_offset_labels=np.zeros((0,), dtype=np.int64),
        roundtrip_errors_px=np.zeros((0,), dtype=np.float32),
    )


def merge_fine_labels_by_cell_pair(
    coarse: MatchaCoarseSupervision,
    fine: MatchaCoarseSupervision,
) -> tuple[MatchaCoarseSupervision, int]:
    """Copy geometry-validated fine labels without changing coarse pairs.

    Sub-cell or detector seeds are useful fine supervision, but using them as
    the coarse correspondence seed changes descriptor positives and can make
    the coarse matcher chase seed jitter.  This helper keeps the stable coarse
    query/render cell pairs.  Query-side labels are copied only when the same
    query/render cell pair survived the fine-seed geometry pass; render-side
    labels are copied by render cell because render refinement drives the 3D
    sample used by PnP and does not require the sub-cell projection to remain
    inside the center-derived query cell.
    """

    coarse_count = int(coarse.count)
    fine_count = int(fine.count)
    if coarse_count == 0 or fine_count == 0:
        return coarse, 0
    fine_by_pair: dict[tuple[int, int], int] = {}
    fine_by_render: dict[int, int] = {}
    for idx, (query_idx, render_idx) in enumerate(zip(fine.query_indices.tolist(), fine.render_indices.tolist())):
        fine_by_pair.setdefault((int(query_idx), int(render_idx)), int(idx))
        fine_by_render.setdefault(int(render_idx), int(idx))

    query_labels = np.asarray(coarse.query_offset_labels, dtype=np.int64).copy()
    render_labels = np.asarray(coarse.render_offset_labels, dtype=np.int64).copy()
    query_soft = np.asarray(coarse.query_offset_soft_labels, dtype=np.float32).copy()
    render_soft = np.asarray(coarse.render_offset_soft_labels, dtype=np.float32).copy()
    transferred = 0
    for idx, (query_idx, render_idx) in enumerate(zip(coarse.query_indices.tolist(), coarse.render_indices.tolist())):
        copied = False
        pair_fine_idx = fine_by_pair.get((int(query_idx), int(render_idx)))
        if pair_fine_idx is not None:
            query_labels[int(idx)] = int(fine.query_offset_labels[pair_fine_idx])
            query_soft[int(idx)] = np.asarray(fine.query_offset_soft_labels[pair_fine_idx], dtype=np.float32)
            copied = True
        render_fine_idx = fine_by_render.get(int(render_idx))
        if render_fine_idx is not None:
            render_labels[int(idx)] = int(fine.render_offset_labels[render_fine_idx])
            render_soft[int(idx)] = np.asarray(fine.render_offset_soft_labels[render_fine_idx], dtype=np.float32)
            copied = True
        if copied:
            transferred += 1

    if transferred == 0:
        return coarse, 0
    return (
        MatchaCoarseSupervision(
            query_indices=coarse.query_indices,
            render_indices=coarse.render_indices,
            query_xy=coarse.query_xy,
            render_xy=coarse.render_xy,
            query_offset_labels=query_labels,
            render_offset_labels=render_labels,
            roundtrip_errors_px=coarse.roundtrip_errors_px,
            query_offset_soft_labels=query_soft,
            render_offset_soft_labels=render_soft,
            confidence_targets=coarse.confidence_targets,
            confidence_ignore_mask=coarse.confidence_ignore_mask,
            uncertainty_px=coarse.uncertainty_px,
            no_match_query_indices=coarse.no_match_query_indices,
            no_match_render_indices=coarse.no_match_render_indices,
            no_match_query_offset_labels=coarse.no_match_query_offset_labels,
            no_match_render_offset_labels=coarse.no_match_render_offset_labels,
            no_match_roundtrip_errors_px=coarse.no_match_roundtrip_errors_px,
            no_match_reason_ids=coarse.no_match_reason_ids,
            no_match_confidence_targets=coarse.no_match_confidence_targets,
            no_match_confidence_ignore_mask=coarse.no_match_confidence_ignore_mask,
            source=coarse.source,
            support_view_counts=coarse.support_view_counts,
            track_ids=coarse.track_ids,
            landmark_xyz=coarse.landmark_xyz,
        ),
        int(transferred),
    )


def _grid_cell_centers(*, image_width: int, image_height: int, grid_width: int, grid_height: int) -> np.ndarray:
    xs = (np.arange(int(grid_width), dtype=np.float64) + 0.5) * float(image_width) / float(grid_width)
    ys = (np.arange(int(grid_height), dtype=np.float64) + 0.5) * float(image_height) / float(grid_height)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)


def _project_world_to_image(points_xyz: np.ndarray, camera: ColmapCamera, pose_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for MATCHA coarse supervision") from exc
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _ = cv2.Rodrigues(pose[:3, :3])
    projected, _ = cv2.projectPoints(points, rvec, pose[:3, 3], camera_matrix, distortion)
    camera_xyz = (pose[:3, :3] @ points.T).T + pose[:3, 3]
    xy = projected.reshape(-1, 2).astype(np.float64, copy=False)
    valid = np.isfinite(xy).all(axis=1) & (camera_xyz[:, 2] > 1e-6)
    return xy, valid


def _cell_indices(xy: np.ndarray, *, image_width: int, image_height: int, grid_width: int, grid_height: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    cell_w = float(image_width) / float(grid_width)
    cell_h = float(image_height) / float(grid_height)
    col = np.floor(coords[:, 0] / cell_w).astype(np.int64)
    row = np.floor(coords[:, 1] / cell_h).astype(np.int64)
    valid = (
        np.isfinite(coords).all(axis=1)
        & (coords[:, 0] >= 0.0)
        & (coords[:, 0] < float(image_width))
        & (coords[:, 1] >= 0.0)
        & (coords[:, 1] < float(image_height))
        & (col >= 0)
        & (col < int(grid_width))
        & (row >= 0)
        & (row < int(grid_height))
    )
    idx = row * int(grid_width) + col
    idx[~valid] = -1
    return idx, valid


def cell_offset_labels(
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
    offset_bins: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize image coordinates into MATCHA's per-cell offset bins."""

    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    cell_w = float(image_width) / float(grid_width)
    cell_h = float(image_height) / float(grid_height)
    cell_idx, valid = _cell_indices(
        coords,
        image_width=int(image_width),
        image_height=int(image_height),
        grid_width=int(grid_width),
        grid_height=int(grid_height),
    )
    col = np.maximum(cell_idx % int(grid_width), 0)
    row = np.maximum(cell_idx // int(grid_width), 0)
    frac_x = np.clip((coords[:, 0] - col.astype(np.float64) * cell_w) / max(cell_w, 1e-12), 0.0, 1.0 - 1e-12)
    frac_y = np.clip((coords[:, 1] - row.astype(np.float64) * cell_h) / max(cell_h, 1e-12), 0.0, 1.0 - 1e-12)
    bin_x = np.floor(frac_x * int(offset_bins)).astype(np.int64)
    bin_y = np.floor(frac_y * int(offset_bins)).astype(np.int64)
    labels = bin_x + int(offset_bins) * bin_y
    labels[~valid] = -1
    return labels.astype(np.int64, copy=False), valid


def cell_offset_soft_labels(
    xy: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_width: int,
    grid_height: int,
    offset_bins: int = 8,
    sigma_bins: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return soft 65-way offset distributions for patch-level supervision."""

    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    labels, valid = cell_offset_labels(
        coords,
        image_width=int(image_width),
        image_height=int(image_height),
        grid_width=int(grid_width),
        grid_height=int(grid_height),
        offset_bins=int(offset_bins),
    )
    output = np.zeros((coords.shape[0], int(offset_bins) * int(offset_bins) + 1), dtype=np.float32)
    if coords.shape[0] == 0:
        return output, valid
    cell_w = float(image_width) / float(grid_width)
    cell_h = float(image_height) / float(grid_height)
    cell_idx, _cell_valid = _cell_indices(
        coords,
        image_width=int(image_width),
        image_height=int(image_height),
        grid_width=int(grid_width),
        grid_height=int(grid_height),
    )
    col = np.maximum(cell_idx % int(grid_width), 0)
    row = np.maximum(cell_idx // int(grid_width), 0)
    frac_x = np.clip((coords[:, 0] - col.astype(np.float64) * cell_w) / max(cell_w, 1e-12), 0.0, 1.0 - 1e-12)
    frac_y = np.clip((coords[:, 1] - row.astype(np.float64) * cell_h) / max(cell_h, 1e-12), 0.0, 1.0 - 1e-12)
    center_x = np.clip(frac_x * int(offset_bins), 0.0, float(int(offset_bins) - 1))
    center_y = np.clip(frac_y * int(offset_bins), 0.0, float(int(offset_bins) - 1))
    bins_x, bins_y = np.meshgrid(np.arange(int(offset_bins), dtype=np.float64), np.arange(int(offset_bins), dtype=np.float64), indexing="xy")
    bins = np.stack([bins_x.reshape(-1), bins_y.reshape(-1)], axis=1)
    sigma = max(float(sigma_bins), 1e-3)
    for idx in range(coords.shape[0]):
        if not bool(valid[idx]):
            output[idx, -1] = 1.0
            continue
        dist2 = (bins[:, 0] - center_x[idx]) ** 2 + (bins[:, 1] - center_y[idx]) ** 2
        probs = np.exp(-0.5 * dist2 / (sigma * sigma)).astype(np.float32)
        probs_sum = float(np.sum(probs))
        if probs_sum <= 0.0:
            output[idx, int(labels[idx])] = 1.0
        else:
            output[idx, : int(offset_bins) * int(offset_bins)] = probs / probs_sum
    return output, valid


def _sample_depth(depth: np.ndarray, xy: np.ndarray, *, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    values, valid = bilinear_sample_feature_map(
        np.asarray(depth, dtype=np.float32).reshape(1, *np.asarray(depth).shape[-2:]),
        xy,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    depth_values = values[:, 0].astype(np.float64, copy=False)
    return depth_values, valid & np.isfinite(depth_values)


def _sample_optional_scalar_map(values: np.ndarray | None, xy: np.ndarray, *, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    if values is None:
        count = np.asarray(xy).reshape(-1, 2).shape[0]
        return np.ones((count,), dtype=np.float64), np.ones((count,), dtype=bool)
    sampled, valid = bilinear_sample_feature_map(
        np.asarray(values, dtype=np.float32).reshape(1, *np.asarray(values).shape[-2:]),
        xy,
        image_width=int(image_width),
        image_height=int(image_height),
    )
    scalars = sampled[:, 0].astype(np.float64, copy=False)
    return scalars, valid & np.isfinite(scalars)


def _sample_depth_edge(depth: np.ndarray, xy: np.ndarray, *, image_width: int, image_height: int) -> np.ndarray:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    arr = np.asarray(depth, dtype=np.float32)
    height, width = int(arr.shape[-2]), int(arr.shape[-1])
    if coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    px = np.clip(np.rint(coords[:, 0] * float(width) / max(float(image_width), 1.0)).astype(np.int64), 0, width - 1)
    py = np.clip(np.rint(coords[:, 1] * float(height) / max(float(image_height), 1.0)).astype(np.int64), 0, height - 1)
    edges = np.zeros((coords.shape[0],), dtype=np.float32)
    for idx, (x, y) in enumerate(zip(px.tolist(), py.tolist())):
        x0, x1 = max(0, x - 1), min(width, x + 2)
        y0, y1 = max(0, y - 1), min(height, y + 2)
        patch = arr[y0:y1, x0:x1]
        patch = patch[np.isfinite(patch) & (patch > 0.0)]
        if patch.size:
            edges[idx] = float(np.max(patch) - np.min(patch))
    return edges


def _patch_overlap_proxy(xy: np.ndarray, *, image_width: int, image_height: int, grid_width: int, grid_height: int) -> np.ndarray:
    coords = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if coords.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    cell_w = float(image_width) / float(grid_width)
    cell_h = float(image_height) / float(grid_height)
    cell_idx, valid = _cell_indices(coords, image_width=image_width, image_height=image_height, grid_width=grid_width, grid_height=grid_height)
    col = np.maximum(cell_idx % int(grid_width), 0)
    row = np.maximum(cell_idx // int(grid_width), 0)
    frac_x = np.clip((coords[:, 0] - col.astype(np.float64) * cell_w) / max(cell_w, 1e-12), 0.0, 1.0)
    frac_y = np.clip((coords[:, 1] - row.astype(np.float64) * cell_h) / max(cell_h, 1e-12), 0.0, 1.0)
    margin = np.minimum.reduce([frac_x, 1.0 - frac_x, frac_y, 1.0 - frac_y])
    overlap = np.clip(2.0 * margin, 0.0, 1.0).astype(np.float32)
    overlap[~valid] = 0.0
    return overlap


def build_matcha_coarse_supervision(
    *,
    render_depth: np.ndarray,
    query_depth: np.ndarray,
    render_alpha: np.ndarray | None = None,
    query_alpha: np.ndarray | None = None,
    render_camera: ColmapCamera,
    query_camera: ColmapCamera,
    render_pose_w2c: np.ndarray,
    query_pose_w2c: np.ndarray,
    render_grid_hw: tuple[int, int],
    query_grid_hw: tuple[int, int],
    render_seed_xy: np.ndarray | None = None,
    config: MatchaCoarseSupervisionConfig | None = None,
) -> MatchaCoarseSupervision:
    """Build coarse query-render cell correspondences with bidirectional depth checks."""

    cfg = config or MatchaCoarseSupervisionConfig()
    render_grid_h, render_grid_w = int(render_grid_hw[0]), int(render_grid_hw[1])
    query_grid_h, query_grid_w = int(query_grid_hw[0]), int(query_grid_hw[1])
    if render_seed_xy is None:
        render_xy = _grid_cell_centers(
            image_width=int(render_camera.width),
            image_height=int(render_camera.height),
            grid_width=render_grid_w,
            grid_height=render_grid_h,
        )
        render_indices = np.arange(render_xy.shape[0], dtype=np.int64)
        render_cell_valid = np.ones((render_xy.shape[0],), dtype=bool)
    else:
        render_xy = np.asarray(render_seed_xy, dtype=np.float64).reshape(-1, 2)
        if render_xy.shape[0] == 0:
            return _empty_supervision()
        render_indices, render_cell_valid = _cell_indices(
            render_xy,
            image_width=int(render_camera.width),
            image_height=int(render_camera.height),
            grid_width=render_grid_w,
            grid_height=render_grid_h,
        )
    render_xyz, render_depth_valid = render_keypoints_to_world(
        render_xy,
        np.asarray(render_depth, dtype=np.float32),
        render_camera,
        render_pose_w2c,
        render_width=int(render_camera.width),
        render_height=int(render_camera.height),
    )
    query_xy, query_proj_valid = _project_world_to_image(render_xyz, query_camera, query_pose_w2c)
    query_indices, query_cell_valid = _cell_indices(
        query_xy,
        image_width=int(query_camera.width),
        image_height=int(query_camera.height),
        grid_width=query_grid_w,
        grid_height=query_grid_h,
    )
    query_depth_values, query_depth_valid = _sample_depth(
        query_depth,
        query_xy,
        image_width=int(query_camera.width),
        image_height=int(query_camera.height),
    )
    query_depth_valid = query_depth_valid & (query_depth_values > float(cfg.min_depth))
    render_alpha_values, render_alpha_valid = _sample_optional_scalar_map(
        render_alpha,
        render_xy,
        image_width=int(render_camera.width),
        image_height=int(render_camera.height),
    )
    query_alpha_values, query_alpha_valid = _sample_optional_scalar_map(
        query_alpha,
        query_xy,
        image_width=int(query_camera.width),
        image_height=int(query_camera.height),
    )
    alpha_valid = render_alpha_valid & query_alpha_valid
    if float(cfg.alpha_threshold) > 0.0:
        alpha_valid &= (render_alpha_values >= float(cfg.alpha_threshold)) & (query_alpha_values >= float(cfg.alpha_threshold))
    query_xyz, query_xyz_valid = render_keypoints_to_world(
        query_xy,
        np.asarray(query_depth, dtype=np.float32),
        query_camera,
        query_pose_w2c,
        render_width=int(query_camera.width),
        render_height=int(query_camera.height),
    )
    render_xy_roundtrip, render_roundtrip_valid = _project_world_to_image(query_xyz, render_camera, render_pose_w2c)
    roundtrip = np.linalg.norm(render_xy_roundtrip - render_xy, axis=1).astype(np.float32)
    render_depth_edge = _sample_depth_edge(
        np.asarray(render_depth, dtype=np.float32),
        render_xy,
        image_width=int(render_camera.width),
        image_height=int(render_camera.height),
    )
    query_depth_edge = _sample_depth_edge(
        np.asarray(query_depth, dtype=np.float32),
        query_xy,
        image_width=int(query_camera.width),
        image_height=int(query_camera.height),
    )
    depth_edge_valid = np.ones((render_xy.shape[0],), dtype=bool)
    if float(cfg.depth_edge_threshold_m) > 0.0:
        depth_edge_valid &= (render_depth_edge <= float(cfg.depth_edge_threshold_m)) & (query_depth_edge <= float(cfg.depth_edge_threshold_m))
    render_labels, render_label_valid = cell_offset_labels(
        render_xy,
        image_width=int(render_camera.width),
        image_height=int(render_camera.height),
        grid_width=render_grid_w,
        grid_height=render_grid_h,
        offset_bins=int(cfg.offset_bins),
    )
    query_labels, query_label_valid = cell_offset_labels(
        query_xy,
        image_width=int(query_camera.width),
        image_height=int(query_camera.height),
        grid_width=query_grid_w,
        grid_height=query_grid_h,
        offset_bins=int(cfg.offset_bins),
    )
    render_soft, _render_soft_valid = cell_offset_soft_labels(
        render_xy,
        image_width=int(render_camera.width),
        image_height=int(render_camera.height),
        grid_width=render_grid_w,
        grid_height=render_grid_h,
        offset_bins=int(cfg.offset_bins),
        sigma_bins=float(cfg.soft_offset_sigma_bins),
    )
    query_soft, _query_soft_valid = cell_offset_soft_labels(
        query_xy,
        image_width=int(query_camera.width),
        image_height=int(query_camera.height),
        grid_width=query_grid_w,
        grid_height=query_grid_h,
        offset_bins=int(cfg.offset_bins),
        sigma_bins=float(cfg.soft_offset_sigma_bins),
    )
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
        render_cell_valid
        & render_depth_valid
        & query_proj_valid
        & query_cell_valid
        & query_depth_valid
        & query_xyz_valid
        & render_roundtrip_valid
        & alpha_valid
        & render_label_valid
        & query_label_valid
        & np.isfinite(roundtrip)
    )
    valid = (
        render_cell_valid
        & render_depth_valid
        & query_proj_valid
        & query_cell_valid
        & query_depth_valid
        & query_xyz_valid
        & render_roundtrip_valid
        & alpha_valid
        & depth_edge_valid
        & render_label_valid
        & query_label_valid
        & np.isfinite(roundtrip)
        & (roundtrip <= float(cfg.roundtrip_threshold_px))
    )
    no_match_items = []
    if bool(cfg.collect_no_match) and int(cfg.max_no_match) > 0:
        no_match_mask = geometry_candidate & ~valid
        reason_ids = np.zeros((render_xy.shape[0],), dtype=np.int64)
        reason_ids[~alpha_valid] = 1
        reason_ids[(reason_ids == 0) & ~depth_edge_valid] = 2
        reason_ids[(reason_ids == 0) & (~query_depth_valid | ~query_xyz_valid | ~render_roundtrip_valid | (roundtrip > float(cfg.roundtrip_threshold_px)))] = 3
        reason_ids[(reason_ids == 0) & no_match_mask] = 4
        for idx in np.flatnonzero(no_match_mask).tolist():
            if bool(cfg.pose_confidence_labels):
                pose_positive = (
                    (roundtrip[int(idx)] <= float(cfg.pose_confidence_positive_threshold_px))
                    and bool(visibility_valid[int(idx)])
                    and bool(depth_edge_valid[int(idx)])
                )
                pose_negative = (
                    (roundtrip[int(idx)] > float(cfg.pose_confidence_negative_threshold_px))
                    or (not bool(visibility_valid[int(idx)]))
                    or (not bool(depth_edge_valid[int(idx)]))
                )
                confidence_target = 1.0 if pose_positive else 0.0
                confidence_ignore = not (pose_positive or pose_negative)
            else:
                confidence_target = 0.0
                confidence_ignore = False
            no_match_items.append(
                (
                    int(reason_ids[int(idx)]),
                    float(roundtrip[int(idx)]) if np.isfinite(roundtrip[int(idx)]) else float("inf"),
                    int(query_indices[int(idx)]),
                    int(render_indices[int(idx)]),
                    int(query_labels[int(idx)]),
                    int(render_labels[int(idx)]),
                    float(confidence_target),
                    bool(confidence_ignore),
                )
            )
        no_match_items.sort(key=lambda item: (item[0], item[2], item[3], item[1]))
        no_match_items = no_match_items[: int(cfg.max_no_match)]
    if not np.any(valid):
        empty = _empty_supervision()
        if no_match_items:
            return MatchaCoarseSupervision(
                query_indices=empty.query_indices,
                render_indices=empty.render_indices,
                query_xy=empty.query_xy,
                render_xy=empty.render_xy,
                query_offset_labels=empty.query_offset_labels,
                render_offset_labels=empty.render_offset_labels,
                roundtrip_errors_px=empty.roundtrip_errors_px,
                no_match_query_indices=np.asarray([item[2] for item in no_match_items], dtype=np.int64),
                no_match_render_indices=np.asarray([item[3] for item in no_match_items], dtype=np.int64),
                no_match_query_offset_labels=np.asarray([item[4] for item in no_match_items], dtype=np.int64),
                no_match_render_offset_labels=np.asarray([item[5] for item in no_match_items], dtype=np.int64),
                no_match_roundtrip_errors_px=np.asarray([item[1] for item in no_match_items], dtype=np.float32),
                no_match_reason_ids=np.asarray([item[0] for item in no_match_items], dtype=np.int64),
                no_match_confidence_targets=np.asarray([item[6] for item in no_match_items], dtype=np.float32),
                no_match_confidence_ignore_mask=np.asarray([item[7] for item in no_match_items], dtype=bool),
            )
        return empty

    candidates = []
    roundtrip_threshold = max(float(cfg.roundtrip_threshold_px), 1e-6)
    query_overlap = _patch_overlap_proxy(
        query_xy,
        image_width=int(query_camera.width),
        image_height=int(query_camera.height),
        grid_width=query_grid_w,
        grid_height=query_grid_h,
    )
    render_overlap = _patch_overlap_proxy(
        render_xy,
        image_width=int(render_camera.width),
        image_height=int(render_camera.height),
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
    confidence = np.clip(roundtrip_weight * edge_weight * (0.5 + 0.5 * np.minimum(query_overlap, render_overlap)), 0.0, 1.0)
    confidence_ignore = np.zeros((render_xy.shape[0],), dtype=bool)
    if bool(cfg.pose_confidence_labels):
        pose_positive = (
            (roundtrip <= float(cfg.pose_confidence_positive_threshold_px))
            & visibility_valid
            & depth_edge_valid
        )
        pose_negative = (roundtrip > float(cfg.pose_confidence_negative_threshold_px)) | ~visibility_valid | ~depth_edge_valid
        confidence = pose_positive.astype(np.float32, copy=False)
        confidence_ignore = ~(pose_positive | pose_negative)
    uncertainty = (
        roundtrip.astype(np.float32)
        + np.maximum(render_depth_edge, query_depth_edge).astype(np.float32)
        + (1.0 - np.minimum(query_overlap, render_overlap).astype(np.float32))
    )
    for idx in np.flatnonzero(valid).tolist():
        candidates.append(
            (
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
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_query: set[int] = set()
    used_render: set[int] = set()
    kept = []
    for item in candidates:
        _err, query_idx, render_idx, *_rest = item
        if query_idx in used_query or render_idx in used_render:
            continue
        used_query.add(query_idx)
        used_render.add(render_idx)
        kept.append(item)
    if not kept:
        return _empty_supervision()
    kept.sort(key=lambda item: (item[1], item[2]))
    return MatchaCoarseSupervision(
        query_indices=np.asarray([item[1] for item in kept], dtype=np.int64),
        render_indices=np.asarray([item[2] for item in kept], dtype=np.int64),
        query_xy=np.stack([item[3] for item in kept], axis=0),
        render_xy=np.stack([item[4] for item in kept], axis=0),
        query_offset_labels=np.asarray([item[5] for item in kept], dtype=np.int64),
        render_offset_labels=np.asarray([item[6] for item in kept], dtype=np.int64),
        roundtrip_errors_px=np.asarray([item[0] for item in kept], dtype=np.float32),
        query_offset_soft_labels=np.stack([item[7] for item in kept], axis=0).astype(np.float32, copy=False),
        render_offset_soft_labels=np.stack([item[8] for item in kept], axis=0).astype(np.float32, copy=False),
        confidence_targets=np.asarray([item[9] for item in kept], dtype=np.float32),
        confidence_ignore_mask=np.asarray([item[10] for item in kept], dtype=bool),
        uncertainty_px=np.asarray([item[11] for item in kept], dtype=np.float32),
        no_match_query_indices=np.asarray([item[2] for item in no_match_items], dtype=np.int64),
        no_match_render_indices=np.asarray([item[3] for item in no_match_items], dtype=np.int64),
        no_match_query_offset_labels=np.asarray([item[4] for item in no_match_items], dtype=np.int64),
        no_match_render_offset_labels=np.asarray([item[5] for item in no_match_items], dtype=np.int64),
        no_match_roundtrip_errors_px=np.asarray([item[1] for item in no_match_items], dtype=np.float32),
        no_match_reason_ids=np.asarray([item[0] for item in no_match_items], dtype=np.int64),
        no_match_confidence_targets=np.asarray([item[6] for item in no_match_items], dtype=np.float32),
        no_match_confidence_ignore_mask=np.asarray([item[7] for item in no_match_items], dtype=bool),
    )
