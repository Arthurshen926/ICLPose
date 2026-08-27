"""Bounded planar-maplet extraction and oracle plane-pose solvers.

The map extractor refits every existing exact-membership physical parent as a
bounded plane.  This preserves primitive lineage and avoids inventing a new
connectivity partition.  The pose solvers operate on plane correspondences;
they do not use point features, PnP, ALIKE, or a discrete camera-pose lattice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial import cKDTree

from .physical_map import GoalMapletPhysicalMap


PLANARITY_DISTANCE_RMS_M = 0.10
PLANARITY_NORMAL_COSINE = 0.90
MINIMUM_BOUNDARY_AREA_M2 = 0.01
MINIMUM_MEMBER_COUNT = 1
MERGE_NORMAL_COSINE = 0.90
MERGE_PLANE_DISTANCE_M = 0.10
MERGE_BOUNDARY_GAP_M = 0.25


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), 1.0e-15)


def _frame_from_normal(normal: np.ndarray) -> np.ndarray:
    n = _unit(normal)
    axis = np.asarray([1.0, 0.0, 0.0])
    if abs(float(n @ axis)) > 0.9:
        axis = np.asarray([0.0, 1.0, 0.0])
    u = _unit(axis - n * float(n @ axis))
    v = np.cross(n, u)
    return np.stack((u, v, n))


@dataclass(frozen=True)
class PlanarMapletMap:
    parent_rows: np.ndarray
    normals_world: np.ndarray
    offsets_world: np.ndarray
    centers_world: np.ndarray
    frames_world: np.ndarray
    boundary_offsets: np.ndarray
    boundary_uv: np.ndarray
    boundary_area_m2: np.ndarray
    member_counts: np.ndarray
    member_area_m2: np.ndarray
    center_residual_rms_m: np.ndarray
    center_residual_p95_m: np.ndarray
    normal_cosine_p10: np.ndarray
    tangent_rank_ratio: np.ndarray
    eligible: np.ndarray


def _fit_member_groups(
    physical: GoalMapletPhysicalMap,
    member_groups: list[np.ndarray],
    reference_normals: np.ndarray,
) -> PlanarMapletMap:
    count = len(member_groups)
    normals = np.empty((count, 3), np.float64)
    offsets = np.empty((count,), np.float64)
    centers = np.empty((count, 3), np.float64)
    frames = np.empty((count, 3, 3), np.float64)
    areas = np.empty((count,), np.float64)
    member_counts = np.empty((count,), np.int64)
    member_area = np.empty((count,), np.float64)
    rms = np.empty((count,), np.float64)
    p95 = np.empty((count,), np.float64)
    cosine_p10 = np.empty((count,), np.float64)
    rank_ratio = np.empty((count,), np.float64)
    boundary_parts: list[np.ndarray] = []
    boundary_offsets = [0]
    for parent in range(count):
        rows = np.asarray(member_groups[parent], np.int64)
        weight = (
            np.pi * np.asarray(physical.primitive_scale1[rows], np.float64)
            * np.asarray(physical.primitive_scale2[rows], np.float64)
            * np.asarray(physical.primitive_opacity[rows], np.float64)
        )
        points = np.asarray(physical.primitive_centers[rows], np.float64)
        center = np.average(points, axis=0, weights=np.maximum(weight, 1.0e-12))
        covariance = np.einsum(
            "n,ni,nj->ij", weight, points - center, points - center,
        ) / max(float(np.sum(weight)), 1.0e-12)
        # A 2DGS primitive is an oriented ellipse, not a point.  Add its exact
        # uniform-ellipse in-plane covariance so small/linear groups cannot
        # choose an arbitrary PCA normal from centers alone.
        tangent1 = np.asarray(physical.primitive_tangent1[rows], np.float64)
        tangent2 = np.asarray(physical.primitive_tangent2[rows], np.float64)
        scale1 = np.asarray(physical.primitive_scale1[rows], np.float64)
        scale2 = np.asarray(physical.primitive_scale2[rows], np.float64)
        covariance += np.einsum(
            "n,n,ni,nj->ij", weight, scale1 * scale1 / 4.0, tangent1, tangent1,
        ) / max(float(np.sum(weight)), 1.0e-12)
        covariance += np.einsum(
            "n,n,ni,nj->ij", weight, scale2 * scale2 / 4.0, tangent2, tangent2,
        ) / max(float(np.sum(weight)), 1.0e-12)
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        normal = eigenvector[:, 0]
        if float(normal @ reference_normals[parent]) < 0.0:
            normal = -normal
        normal = _unit(normal)
        frame = _frame_from_normal(normal)
        center_signed_residual = (points - center) @ normal
        residual_squared = (
            center_signed_residual * center_signed_residual
            + (scale1 * (tangent1 @ normal)) ** 2 / 4.0
            + (scale2 * (tangent2 @ normal)) ** 2 / 4.0
        )
        residual = np.sqrt(residual_squared)
        primitive_cosine = np.abs(
            np.asarray(physical.primitive_normals[rows], np.float64) @ normal
        )
        signs = np.asarray([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        corners = (
            points[:, None, :]
            + signs[None, :, 0, None]
            * np.asarray(physical.primitive_scale1[rows])[:, None, None]
            * np.asarray(physical.primitive_tangent1[rows])[:, None, :]
            + signs[None, :, 1, None]
            * np.asarray(physical.primitive_scale2[rows])[:, None, None]
            * np.asarray(physical.primitive_tangent2[rows])[:, None, :]
        ).reshape(-1, 3)
        uv = (corners - center) @ frame[:2].T
        try:
            hull = ConvexHull(uv)
            boundary = uv[hull.vertices]
            area = float(hull.volume)
        except QhullError:
            boundary = uv[np.unique(uv, axis=0, return_index=True)[1]]
            area = 0.0
        normals[parent] = normal
        offsets[parent] = float(normal @ center)  # n^T x = rho
        centers[parent] = center
        frames[parent] = frame
        areas[parent] = area
        member_counts[parent] = rows.size
        member_area[parent] = float(np.sum(weight))
        rms[parent] = float(np.sqrt(np.average(residual * residual, weights=np.maximum(weight, 1e-12))))
        p95[parent] = float(np.quantile(residual, 0.95))
        cosine_p10[parent] = float(np.quantile(primitive_cosine, 0.10))
        rank_ratio[parent] = float(eigenvalue[1] / max(eigenvalue[2], 1.0e-15))
        boundary_parts.append(boundary.astype(np.float64))
        boundary_offsets.append(boundary_offsets[-1] + len(boundary))
    eligible = (
        (member_counts >= MINIMUM_MEMBER_COUNT)
        & (areas >= MINIMUM_BOUNDARY_AREA_M2)
        & (rms <= PLANARITY_DISTANCE_RMS_M)
        & (cosine_p10 >= PLANARITY_NORMAL_COSINE)
    )
    return PlanarMapletMap(
        parent_rows=np.arange(count, dtype=np.int64), normals_world=normals,
        offsets_world=offsets, centers_world=centers, frames_world=frames,
        boundary_offsets=np.asarray(boundary_offsets, np.int64),
        boundary_uv=np.concatenate(boundary_parts, axis=0),
        boundary_area_m2=areas, member_counts=member_counts,
        member_area_m2=member_area, center_residual_rms_m=rms,
        center_residual_p95_m=p95, normal_cosine_p10=cosine_p10,
        tangent_rank_ratio=rank_ratio, eligible=eligible,
    )


def fit_planar_maplets(physical: GoalMapletPhysicalMap) -> PlanarMapletMap:
    groups = [
        np.asarray(physical.membership_primitive_rows[physical.member_slice(row)], np.int64)
        for row in range(int(physical.maplet_ids.size))
    ]
    return _fit_member_groups(physical, groups, physical.maplet_normals)


def fit_child_planar_maplets(physical: GoalMapletPhysicalMap) -> PlanarMapletMap:
    groups = [
        np.asarray(
            physical.child_member_primitive_rows[
                int(physical.child_member_offsets[row]):int(physical.child_member_offsets[row + 1])
            ], np.int64,
        )
        for row in range(int(physical.child_parent_rows.size))
    ]
    return _fit_member_groups(physical, groups, physical.child_normals)


def merge_coplanar_child_regions(
    physical: GoalMapletPhysicalMap,
    child_planes: PlanarMapletMap,
    *,
    restrict_to_parent: bool = True,
) -> tuple[PlanarMapletMap, np.ndarray, list[np.ndarray]]:
    """Merge adjacent coplanar child seeds without crossing parent context."""

    child_count = int(physical.child_parent_rows.size)
    if child_planes.parent_rows.shape != (child_count,):
        raise ValueError("child plane inventory differs")
    seed = (
        (child_planes.boundary_area_m2 >= MINIMUM_BOUNDARY_AREA_M2)
        & (child_planes.center_residual_rms_m <= PLANARITY_DISTANCE_RMS_M)
    )
    radius = np.empty((child_count,), np.float64)
    for child in range(child_count):
        lo = int(child_planes.boundary_offsets[child])
        hi = int(child_planes.boundary_offsets[child + 1])
        radius[child] = (
            float(np.max(np.linalg.norm(child_planes.boundary_uv[lo:hi], axis=1)))
            if hi > lo else 0.0
        )
    owner = np.arange(child_count, dtype=np.int64)

    def find(value: int) -> int:
        while owner[value] != value:
            owner[value] = owner[owner[value]]
            value = int(owner[value])
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            owner[max(a, b)] = min(a, b)

    if restrict_to_parent:
        candidate_groups = []
        for parent in range(int(physical.maplet_ids.size)):
            begin = int(physical.maplet_child_offsets[parent])
            end = int(physical.maplet_child_offsets[parent + 1])
            rows = np.arange(begin, end, dtype=np.int64)
            candidate_groups.append(rows[seed[rows]])
        candidate_pairs = (
            (left, int(right))
            for rows in candidate_groups
            for local, left in enumerate(rows.tolist())
            for right in rows[local + 1:].tolist()
        )
    else:
        seed_rows = np.flatnonzero(seed)
        tree = cKDTree(child_planes.centers_world[seed_rows])
        # Atomic child regions are approximately metre-scale.  Exact radius
        # and plane-distance checks below reject unrelated pairs; this query
        # radius only keeps the global graph sparse.
        local_pairs = tree.query_pairs(r=3.0, output_type="ndarray")
        candidate_pairs = (
            (int(seed_rows[left]), int(seed_rows[right]))
            for left, right in local_pairs.tolist()
        )
    for left, right in candidate_pairs:
            right_rows = np.asarray([right], np.int64)
            center_distance = np.linalg.norm(
                child_planes.centers_world[right_rows]
                - child_planes.centers_world[left], axis=1,
            )
            nearby = center_distance <= (
                radius[left] + radius[right_rows] + MERGE_BOUNDARY_GAP_M
            )
            cosine = np.abs(
                child_planes.normals_world[right_rows]
                @ child_planes.normals_world[left]
            )
            left_distance = np.abs(
                child_planes.centers_world[right_rows]
                @ child_planes.normals_world[left]
                - child_planes.offsets_world[left]
            )
            right_distance = np.abs(
                child_planes.normals_world[right_rows]
                @ child_planes.centers_world[left]
                - child_planes.offsets_world[right_rows]
            )
            for accepted_right in right_rows[
                nearby & (cosine >= MERGE_NORMAL_COSINE)
                & (left_distance <= MERGE_PLANE_DISTANCE_M)
                & (right_distance <= MERGE_PLANE_DISTANCE_M)
            ].tolist():
                union(left, int(accepted_right))
    components: dict[int, list[int]] = {}
    for child in np.flatnonzero(seed).tolist():
        components.setdefault(find(int(child)), []).append(int(child))
    component_children = [np.asarray(value, np.int64) for _, value in sorted(components.items())]
    groups, reference, parents = [], [], []
    for children in component_children:
        members = np.concatenate([
            np.asarray(physical.child_member_primitive_rows[
                int(physical.child_member_offsets[child]):
                int(physical.child_member_offsets[child + 1])
            ], np.int64)
            for child in children.tolist()
        ])
        groups.append(np.unique(members))
        reference.append(child_planes.normals_world[int(children[0])])
        parent_values = np.unique(physical.child_parent_rows[children])
        parents.append(int(parent_values[0]) if parent_values.size == 1 else -1)
    merged = _fit_member_groups(physical, groups, np.asarray(reference, np.float64))
    return merged, np.asarray(parents, np.int64), component_children


def fit_weighted_plane(points: np.ndarray, weights: np.ndarray, reference_normal: np.ndarray):
    points = np.asarray(points, np.float64)
    weights = np.asarray(weights, np.float64).reshape(-1)
    if points.shape != (weights.size, 3) or weights.size < 3 or np.any(weights <= 0):
        raise ValueError("weighted plane observations differ")
    center = np.average(points, axis=0, weights=weights)
    covariance = np.einsum(
        "n,ni,nj->ij", weights, points - center, points - center,
    ) / float(np.sum(weights))
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    normal = eigenvector[:, 0]
    if float(normal @ reference_normal) < 0.0:
        normal = -normal
    return _unit(normal), float(_unit(normal) @ center), eigenvalue


def fit_weighted_primitive_plane(
    physical: GoalMapletPhysicalMap,
    primitive_rows: np.ndarray,
    weights: np.ndarray,
    reference_normal: np.ndarray,
):
    """Fit a plane to visible 2DGS ellipses, not only their centers."""

    rows = np.asarray(primitive_rows, np.int64).reshape(-1)
    weight = np.asarray(weights, np.float64).reshape(-1)
    if rows.shape != weight.shape or rows.size == 0 or np.any(weight <= 0.0):
        raise ValueError("weighted primitive plane observations differ")
    points = np.asarray(physical.primitive_centers[rows], np.float64)
    center = np.average(points, axis=0, weights=weight)
    covariance = np.einsum(
        "n,ni,nj->ij", weight, points - center, points - center,
    )
    for tangent_name, scale_name in (
        ("primitive_tangent1", "primitive_scale1"),
        ("primitive_tangent2", "primitive_scale2"),
    ):
        tangent = np.asarray(getattr(physical, tangent_name)[rows], np.float64)
        scale = np.asarray(getattr(physical, scale_name)[rows], np.float64)
        covariance += np.einsum(
            "n,n,ni,nj->ij", weight, scale * scale / 4.0, tangent, tangent,
        )
    covariance /= float(np.sum(weight))
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    normal = eigenvector[:, 0]
    if float(normal @ reference_normal) < 0.0:
        normal = -normal
    normal = _unit(normal)
    return normal, float(normal @ center), eigenvalue


def solve_rotation_from_plane_normals(query_normals: np.ndarray, map_normals: np.ndarray, weights: np.ndarray):
    query = np.asarray(query_normals, np.float64)
    target = np.asarray(map_normals, np.float64)
    weight = np.asarray(weights, np.float64).reshape(-1)
    cross = np.einsum("n,ni,nj->ij", weight, query, target)
    u, singular, vt = np.linalg.svd(cross)
    rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    return rotation, singular


def solve_metric_translation(map_normals: np.ndarray, map_offsets: np.ndarray, query_offsets: np.ndarray, weights: np.ndarray):
    a = np.asarray(map_normals, np.float64)
    b = np.asarray(map_offsets, np.float64) - np.asarray(query_offsets, np.float64)
    w = np.sqrt(np.asarray(weights, np.float64)).reshape(-1, 1)
    solution, _, rank, singular = np.linalg.lstsq(w * a, w[:, 0] * b, rcond=None)
    return solution, int(rank), singular


def solve_scaled_translation(map_normals: np.ndarray, map_offsets: np.ndarray, query_offsets: np.ndarray, weights: np.ndarray):
    a = np.column_stack((
        np.asarray(map_normals, np.float64),
        np.asarray(query_offsets, np.float64),
    ))
    b = np.asarray(map_offsets, np.float64)
    w = np.sqrt(np.asarray(weights, np.float64)).reshape(-1, 1)
    solution, _, rank, singular = np.linalg.lstsq(w * a, w[:, 0] * b, rcond=None)
    return solution[:3], float(solution[3]), int(rank), singular


__all__ = [
    "MERGE_BOUNDARY_GAP_M", "MERGE_NORMAL_COSINE", "MERGE_PLANE_DISTANCE_M",
    "MINIMUM_BOUNDARY_AREA_M2", "MINIMUM_MEMBER_COUNT",
    "PLANARITY_DISTANCE_RMS_M", "PLANARITY_NORMAL_COSINE", "PlanarMapletMap",
    "fit_child_planar_maplets", "fit_planar_maplets", "fit_weighted_plane",
    "fit_weighted_primitive_plane", "solve_metric_translation",
    "merge_coplanar_child_regions", "solve_rotation_from_plane_normals",
    "solve_scaled_translation",
]
