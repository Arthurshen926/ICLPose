"""Pose-conditioned geometry between fixed query-region supports.

The factor in this module never compares image coordinates with world-frame
coordinates.  Query edges and projected surface edges are both represented in
the normalized camera image, after a pose hypothesis exists.  Edge topology is
constructed once from query measurements and is therefore independent of the
candidate surface assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
)

from .physical_map import DOUBLE_SIDED, GoalMapletPhysicalMap


LOCAL_EDGE = np.uint8(0)
LONG_EDGE = np.uint8(1)


@dataclass(frozen=True)
class QueryEdgeGraph:
    source: np.ndarray
    target: np.ndarray
    kind: np.ndarray
    high_displacement: np.ndarray
    distinctive_context: np.ndarray

    def __post_init__(self) -> None:
        source = np.asarray(self.source, dtype=np.int64).reshape(-1)
        target = np.asarray(self.target, dtype=np.int64).reshape(-1)
        kind = np.asarray(self.kind, dtype=np.uint8).reshape(-1)
        high = np.asarray(self.high_displacement, dtype=bool).reshape(-1)
        distinctive = np.asarray(self.distinctive_context, dtype=bool).reshape(-1)
        if not all(value.shape == source.shape for value in (target, kind, high, distinctive)):
            raise ValueError("query-edge arrays differ")
        if np.any(source < 0) or np.any(target < 0) or np.any(source >= target):
            raise ValueError("query edges must be unique, ordered and non-negative")
        if not set(np.unique(kind).tolist()).issubset({int(LOCAL_EDGE), int(LONG_EDGE)}):
            raise ValueError("unknown query-edge kind")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "high_displacement", high)
        object.__setattr__(self, "distinctive_context", distinctive)


def build_query_edge_graph(
    xy_normalized: np.ndarray,
    context_descriptors: np.ndarray,
    *,
    local_neighbors: int = 4,
    long_neighbors: int = 2,
    minimum_long_displacement: float = 0.30,
    high_displacement: float = 0.35,
    distinctive_cosine: float = 0.75,
) -> QueryEdgeGraph:
    """Build deterministic local and long-range edges from query data only.

    Long edges prefer a combination of image leverage and context contrast.
    All thresholds are fixed representation constants; no map identity, pose,
    phase label, or ground-truth pose participates in edge construction.
    """

    xy = np.asarray(xy_normalized, dtype=np.float64).reshape(-1, 2)
    descriptor = np.asarray(context_descriptors, dtype=np.float64)
    if descriptor.ndim != 2 or descriptor.shape[0] != xy.shape[0]:
        raise ValueError("query edge coordinates and descriptors differ")
    descriptor = descriptor / np.maximum(
        np.linalg.norm(descriptor, axis=1, keepdims=True), 1e-12,
    )
    count = int(xy.shape[0])
    if count < 2:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_b = np.zeros((0,), dtype=bool)
        return QueryEdgeGraph(empty_i, empty_i, empty_i.astype(np.uint8), empty_b, empty_b)
    displacement = xy[None] - xy[:, None]
    distance = np.linalg.norm(displacement, axis=2)
    cosine = np.clip(descriptor @ descriptor.T, -1.0, 1.0)
    np.fill_diagonal(distance, np.inf)
    edge_kind: dict[tuple[int, int], int] = {}
    for source in range(count):
        order = np.argsort(distance[source], kind="stable")
        for target in order[: min(int(local_neighbors), count - 1)].tolist():
            edge = tuple(sorted((int(source), int(target))))
            edge_kind[edge] = int(LOCAL_EDGE)
    for source in range(count):
        valid = np.flatnonzero(
            np.isfinite(distance[source])
            & (distance[source] >= float(minimum_long_displacement))
        )
        if valid.size == 0:
            continue
        priority = distance[source, valid] * (
            0.5 + 0.5 * (1.0 - cosine[source, valid])
        )
        order = valid[np.argsort(-priority, kind="stable")]
        accepted = 0
        for target in order.tolist():
            edge = tuple(sorted((int(source), int(target))))
            if edge in edge_kind:
                continue
            edge_kind[edge] = int(LONG_EDGE)
            accepted += 1
            if accepted >= int(long_neighbors):
                break
    edges = sorted(edge_kind)
    source = np.asarray([edge[0] for edge in edges], dtype=np.int64)
    target = np.asarray([edge[1] for edge in edges], dtype=np.int64)
    kind = np.asarray([edge_kind[edge] for edge in edges], dtype=np.uint8)
    edge_distance = np.linalg.norm(xy[target] - xy[source], axis=1)
    edge_cosine = np.sum(descriptor[source] * descriptor[target], axis=1)
    return QueryEdgeGraph(
        source,
        target,
        kind,
        edge_distance >= float(high_displacement),
        edge_cosine <= float(distinctive_cosine),
    )


def project_child_surfaces(
    child_rows: np.ndarray,
    pose_w2c: np.ndarray,
    physical: GoalMapletPhysicalMap,
    camera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project child centers and tangent rectangles into normalized pixels."""

    children = np.asarray(child_rows, dtype=np.int64).reshape(-1)
    projected_center = np.full((children.size, 2), np.nan, dtype=np.float64)
    projected_extent = np.full((children.size, 2), np.nan, dtype=np.float64)
    visible = children >= 0
    if not np.any(visible):
        return projected_center, projected_extent, visible
    safe = np.maximum(children, 0)
    center = physical.child_centers[safe]
    signs = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    local = np.zeros((children.size, 4, 3), dtype=np.float64)
    local[..., :2] = signs[None] * np.maximum(
        physical.child_extents[safe, None, :2], 1e-4,
    )
    corners = np.einsum("nki,nij->nkj", local, physical.child_frames[safe])
    corners += center[:, None]
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    matrix, distortion = camera_matrix_and_distortion(camera)
    rotation, _ = cv2.Rodrigues(pose[:3, :3])
    center_xy, _ = cv2.projectPoints(center, rotation, pose[:3, 3], matrix, distortion)
    corner_xy, _ = cv2.projectPoints(
        corners.reshape(-1, 3), rotation, pose[:3, 3], matrix, distortion,
    )
    center_depth = center @ pose[:3, :3].T + pose[:3, 3]
    corner_depth = corners @ pose[:3, :3].T + pose[:3, 3]
    scale = np.asarray([float(camera.width), float(camera.height)], dtype=np.float64)
    normalized_center = center_xy.reshape(-1, 2) / scale
    normalized_corner = corner_xy.reshape(children.size, 4, 2) / scale
    projected_center[visible] = normalized_center[visible]
    projected_extent[visible] = 0.5 * (
        np.max(normalized_corner[visible], axis=1)
        - np.min(normalized_corner[visible], axis=1)
    )
    visible &= center_depth[:, 2] > 0.05
    visible &= np.all(corner_depth[..., 2] > 0.05, axis=1)
    visible &= np.all(np.isfinite(projected_center), axis=1)
    visible &= np.all(np.isfinite(projected_extent), axis=1)
    if all(hasattr(physical, name) for name in (
        "child_normals", "child_parent_rows", "maplet_sidedness",
    )):
        camera_center = -pose[:3, :3].T @ pose[:3, 3]
        view = camera_center[None] - center
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-8)
        incidence = np.sum(physical.child_normals[safe] * view, axis=1)
        parent = physical.child_parent_rows[safe]
        front = (physical.maplet_sidedness[parent] == DOUBLE_SIDED) | (incidence >= 0.02)
        visible &= front
    return projected_center, projected_extent, visible


def _huber(residual: np.ndarray, delta: float = 1.0) -> np.ndarray:
    value = np.asarray(residual, dtype=np.float64)
    absolute = np.abs(value)
    return np.where(
        absolute <= float(delta),
        0.5 * np.square(absolute),
        float(delta) * (absolute - 0.5 * float(delta)),
    )


def score_pose_conditioned_query_edges(
    graph: QueryEdgeGraph,
    xy_normalized: np.ndarray,
    extent_normalized: np.ndarray,
    child_rows: np.ndarray,
    pose_w2c: np.ndarray,
    physical: GoalMapletPhysicalMap,
    camera,
    *,
    support_mask: np.ndarray | None = None,
    missing_edge_huber_loss: float = 2.0,
) -> dict[str, dict[str, float | int | None]]:
    """Score signed displacement, order and relative scale under one pose.

    The primary ``vector_score`` is the negative mean analytic Huber loss.
    Relative scale and order are reported separately so this audit cannot hide
    a failed displacement hypothesis behind hand-tuned factor weights.
    """

    xy = np.asarray(xy_normalized, dtype=np.float64).reshape(-1, 2)
    extent = np.asarray(extent_normalized, dtype=np.float64).reshape(-1, 2)
    children = np.asarray(child_rows, dtype=np.int64).reshape(-1)
    if xy.shape != extent.shape or children.shape != (xy.shape[0],):
        raise ValueError("query-edge scoring inputs differ")
    selected_support = (
        np.ones((xy.shape[0],), dtype=bool)
        if support_mask is None
        else np.asarray(support_mask, dtype=bool).reshape(-1)
    )
    if selected_support.shape != (xy.shape[0],):
        raise ValueError("query-edge support mask differs")
    projected, projected_extent, visible = project_child_surfaces(
        children, pose_w2c, physical, camera,
    )
    source, target = graph.source, graph.target
    edge_valid = (
        selected_support[source]
        & selected_support[target]
        & visible[source]
        & visible[target]
        & (children[source] >= 0)
        & (children[target] >= 0)
    )
    observed = xy[target] - xy[source]
    predicted = projected[target] - projected[source]
    sigma = np.maximum(
        np.linalg.norm(extent[source], axis=1)
        + np.linalg.norm(extent[target], axis=1),
        0.02,
    )
    normalized_residual = np.linalg.norm(observed - predicted, axis=1) / sigma
    query_scale = np.maximum(np.linalg.norm(extent, axis=1), 1e-6)
    map_scale = np.maximum(np.linalg.norm(projected_extent, axis=1), 1e-6)
    scale_residual = np.abs(
        np.log(query_scale[source] / query_scale[target])
        - np.log(map_scale[source] / map_scale[target])
    )
    significant_x = np.abs(observed[:, 0]) > sigma
    significant_y = np.abs(observed[:, 1]) > sigma
    correct_x = (~significant_x) | (np.sign(observed[:, 0]) == np.sign(predicted[:, 0]))
    correct_y = (~significant_y) | (np.sign(observed[:, 1]) == np.sign(predicted[:, 1]))
    order_denominator = significant_x.astype(np.int64) + significant_y.astype(np.int64)
    order_numerator = (
        significant_x & correct_x
    ).astype(np.int64) + (significant_y & correct_y).astype(np.int64)

    categories = {
        "all": np.ones(source.shape, dtype=bool),
        "local": graph.kind == LOCAL_EDGE,
        "long_range": graph.kind == LONG_EDGE,
        "high_displacement": graph.high_displacement,
        "distinctive_context": graph.distinctive_context,
    }
    result: dict[str, dict[str, float | int | None]] = {}
    for name, category in categories.items():
        planned = category & selected_support[source] & selected_support[target]
        use = edge_valid & category
        order_use = use & (order_denominator > 0)
        count = int(np.sum(use))
        planned_count = int(np.sum(planned))
        missing_count = int(planned_count - count)
        fixed_vector_score = None
        fixed_relative_scale_score = None
        if planned_count:
            vector_loss = float(np.sum(_huber(normalized_residual[use])))
            scale_loss = float(np.sum(_huber(scale_residual[use])))
            penalty = float(missing_edge_huber_loss) * float(missing_count)
            fixed_vector_score = -(vector_loss + penalty) / float(planned_count)
            fixed_relative_scale_score = -(scale_loss + penalty) / float(planned_count)
        result[name] = {
            "planned_edge_count": planned_count,
            "edge_count": count,
            "missing_edge_count": missing_count,
            "edge_coverage": float(count / planned_count) if planned_count else None,
            "fixed_vector_score": fixed_vector_score,
            "fixed_relative_scale_score": fixed_relative_scale_score,
            "vector_score": (
                float(-np.mean(_huber(normalized_residual[use]))) if count else None
            ),
            "normalized_vector_residual_median": (
                float(np.median(normalized_residual[use])) if count else None
            ),
            "normalized_vector_residual_p90": (
                float(np.percentile(normalized_residual[use], 90.0)) if count else None
            ),
            "relative_scale_score": (
                float(-np.mean(_huber(scale_residual[use]))) if count else None
            ),
            "order_consistency": (
                float(np.sum(order_numerator[order_use]) / np.sum(order_denominator[order_use]))
                if np.any(order_use) else None
            ),
        }
    return result
