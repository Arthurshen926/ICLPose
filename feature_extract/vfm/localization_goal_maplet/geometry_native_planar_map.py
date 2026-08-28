"""Geometry-native finite planar primitives extracted directly from 2DGS surfels.

This module deliberately does not consume the legacy parent/child identities.
Those identities remain useful retrieval indices, but they are not physical
surface boundaries.  Plane identity is instead determined by local 3D support
connectivity followed by component-level plane fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .lineage import arrays_sha256, canonical_json_sha256
from .physical_map import GoalMapletPhysicalMap


SCHEMA = "goal_maplet_geometry_native_planar_map_v1"


def _unit_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float64)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-15)


def _frame(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-15)
    axis = np.array([1.0, 0.0, 0.0])
    if abs(float(normal @ axis)) > .9:
        axis = np.array([0.0, 1.0, 0.0])
    u = axis - normal * float(normal @ axis)
    u /= max(float(np.linalg.norm(u)), 1e-15)
    return np.stack((u, np.cross(normal, u), normal))


@dataclass(frozen=True)
class PrimitiveSurfaceTable:
    primitive_ids: np.ndarray
    centers: np.ndarray
    tangent1: np.ndarray
    tangent2: np.ndarray
    normals: np.ndarray
    scale1: np.ndarray
    scale2: np.ndarray
    opacity: np.ndarray

    @classmethod
    def from_physical_map(cls, physical: GoalMapletPhysicalMap) -> "PrimitiveSurfaceTable":
        # Do not access maplet_ids, child_parent_rows, or either membership
        # table here.  This is the enforceable seam that keeps physical plane
        # extraction independent of the old voxel hierarchy.
        return cls(
            primitive_ids=np.asarray(physical.primitive_ids, np.int64),
            centers=np.asarray(physical.primitive_centers, np.float64),
            tangent1=np.asarray(physical.primitive_tangent1, np.float64),
            tangent2=np.asarray(physical.primitive_tangent2, np.float64),
            normals=np.asarray(physical.primitive_normals, np.float64),
            scale1=np.asarray(physical.primitive_scale1, np.float64),
            scale2=np.asarray(physical.primitive_scale2, np.float64),
            opacity=np.asarray(physical.primitive_opacity, np.float64),
        ).validated()

    def validated(self) -> "PrimitiveSurfaceTable":
        count = np.asarray(self.primitive_ids).size
        if count == 0 or np.unique(self.primitive_ids).size != count:
            raise ValueError("primitive IDs must be non-empty and unique")
        for name in ("centers", "tangent1", "tangent2", "normals"):
            value = np.asarray(getattr(self, name))
            if value.shape != (count, 3) or not np.isfinite(value).all():
                raise ValueError(f"invalid primitive {name}")
        for name in ("scale1", "scale2", "opacity"):
            value = np.asarray(getattr(self, name))
            if value.shape != (count,) or not np.isfinite(value).all():
                raise ValueError(f"invalid primitive {name}")
        if np.any(self.scale1 <= 0) or np.any(self.scale2 <= 0) or np.any(self.opacity <= 0):
            raise ValueError("primitive support must be positive")
        if np.max(np.abs(np.linalg.norm(self.normals, axis=1) - 1.0)) > 1e-5:
            raise ValueError("primitive normals must be unit length")
        return self


@dataclass(frozen=True)
class GeometryNativePlanarMap:
    plane_ids: np.ndarray
    normals_world: np.ndarray
    offsets_world: np.ndarray
    centers_world: np.ndarray
    frames_world: np.ndarray
    boundary_offsets: np.ndarray
    boundary_uv: np.ndarray
    boundary_area_m2: np.ndarray
    member_offsets: np.ndarray
    member_primitive_rows: np.ndarray
    member_counts: np.ndarray
    support_area_m2: np.ndarray
    residual_rms_m: np.ndarray
    residual_p95_m: np.ndarray
    normal_cosine_p10: np.ndarray
    metadata: dict[str, object]

    def validated(self, primitive_count: int | None = None) -> "GeometryNativePlanarMap":
        count = np.asarray(self.plane_ids).size
        if self.plane_ids.shape != (count,) or not np.array_equal(self.plane_ids, np.arange(count)):
            raise ValueError("plane IDs must be canonical rows")
        for name, shape in (("normals_world", (count, 3)), ("centers_world", (count, 3)),
                            ("frames_world", (count, 3, 3))):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid plane {name}")
        for name in ("offsets_world", "boundary_area_m2", "member_counts", "support_area_m2",
                     "residual_rms_m", "residual_p95_m", "normal_cosine_p10"):
            value = np.asarray(getattr(self, name))
            if value.shape != (count,) or not np.isfinite(value).all():
                raise ValueError(f"invalid plane {name}")
        for name, offsets, total in (
            ("boundary", self.boundary_offsets, self.boundary_uv.shape[0]),
            ("member", self.member_offsets, self.member_primitive_rows.size),
        ):
            offsets = np.asarray(offsets)
            if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != total or np.any(np.diff(offsets) < 0):
                raise ValueError(f"invalid {name} offsets")
        if self.boundary_uv.ndim != 2 or self.boundary_uv.shape[1] != 2:
            raise ValueError("invalid plane boundaries")
        rows = np.asarray(self.member_primitive_rows, np.int64)
        if primitive_count is not None and np.any((rows < 0) | (rows >= primitive_count)):
            raise ValueError("plane member is outside primitive inventory")
        if np.unique(rows).size != rows.size:
            raise ValueError("a primitive may support at most one planar instance")
        if not np.array_equal(np.diff(self.member_offsets), self.member_counts):
            raise ValueError("plane member counts differ")
        if self.metadata.get("artifact_type") != SCHEMA or self.metadata.get("uses_parent_child_partition") is not False:
            raise ValueError("planar map hierarchy-independence contract differs")
        return self

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in (
            "plane_ids", "normals_world", "offsets_world", "centers_world", "frames_world",
            "boundary_offsets", "boundary_uv", "boundary_area_m2", "member_offsets",
            "member_primitive_rows", "member_counts", "support_area_m2", "residual_rms_m",
            "residual_p95_m", "normal_cosine_p10",
        )}

    def save_npz(self, path: Path) -> None:
        arrays = self.arrays()
        metadata = dict(self.metadata)
        metadata["arrays_sha256"] = arrays_sha256(arrays)
        metadata["content_sha256"] = canonical_json_sha256({
            "arrays_sha256": metadata["arrays_sha256"],
            **{key: value for key, value in metadata.items() if key not in ("content_sha256",)},
        })
        np.savez_compressed(Path(path), **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))

    @classmethod
    def load_npz(cls, path: Path, *, primitive_count: int | None = None) -> "GeometryNativePlanarMap":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            names = (
                "plane_ids", "normals_world", "offsets_world", "centers_world", "frames_world",
                "boundary_offsets", "boundary_uv", "boundary_area_m2", "member_offsets",
                "member_primitive_rows", "member_counts", "support_area_m2", "residual_rms_m",
                "residual_p95_m", "normal_cosine_p10",
            )
            values = {name: np.asarray(data[name]) for name in names}
        if arrays_sha256(values) != metadata.get("arrays_sha256"):
            raise ValueError("planar map array hash differs")
        result = cls(metadata=metadata, **values)
        return result.validated(primitive_count)


def _build_local_graph(
    table: PrimitiveSurfaceTable, *, neighbors: int, normal_degrees: float,
    plane_distance_m: float, support_gap_m: float,
) -> csr_matrix:
    count = table.primitive_ids.size
    k = min(max(int(neighbors) + 1, 2), count)
    distance, index = cKDTree(table.centers).query(table.centers, k=k, workers=-1)
    if k == 1:
        return csr_matrix((count, count), dtype=bool)
    left = np.repeat(np.arange(count, dtype=np.int64), k - 1)
    right = np.asarray(index[:, 1:]).reshape(-1)
    center_distance = np.asarray(distance[:, 1:]).reshape(-1)
    unique = left < right
    left, right, center_distance = left[unique], right[unique], center_distance[unique]
    delta = table.centers[right] - table.centers[left]
    signed_cosine = np.einsum("ij,ij->i", table.normals[left], table.normals[right])
    cosine = np.abs(signed_cosine)
    reciprocal_distance = np.maximum(
        np.abs(np.einsum("ij,ij->i", delta, table.normals[left])),
        np.abs(np.einsum("ij,ij->i", delta, table.normals[right])),
    )
    aligned = table.normals[left] + np.where(signed_cosine >= 0.0, 1.0, -1.0)[:, None] * table.normals[right]
    aligned = _unit_rows(aligned)
    tangent_delta = delta - np.einsum("ij,ij->i", delta, aligned)[:, None] * aligned
    tangent_distance = np.linalg.norm(tangent_delta, axis=1)
    direction = tangent_delta / np.maximum(tangent_distance[:, None], 1e-15)
    left_radius = np.sqrt(
        (np.einsum("ij,ij->i", direction, table.tangent1[left]) * table.scale1[left]) ** 2
        + (np.einsum("ij,ij->i", direction, table.tangent2[left]) * table.scale2[left]) ** 2
    )
    right_radius = np.sqrt(
        (np.einsum("ij,ij->i", direction, table.tangent1[right]) * table.scale1[right]) ** 2
        + (np.einsum("ij,ij->i", direction, table.tangent2[right]) * table.scale2[right]) ** 2
    )
    keep = (
        (cosine >= np.cos(np.deg2rad(float(normal_degrees))))
        & (reciprocal_distance <= float(plane_distance_m))
        & (tangent_distance <= left_radius + right_radius + float(support_gap_m))
        & np.isfinite(center_distance)
    )
    row = np.r_[left[keep], right[keep]]
    column = np.r_[right[keep], left[keep]]
    return coo_matrix((np.ones(row.size, bool), (row, column)), shape=(count, count)).tocsr()


def _fit_plane(table: PrimitiveSurfaceTable, rows: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = np.pi * table.scale1[rows] * table.scale2[rows] * table.opacity[rows]
    center = np.average(table.centers[rows], axis=0, weights=np.maximum(weight, 1e-15))
    delta = table.centers[rows] - center
    covariance = np.einsum("n,ni,nj->ij", weight, delta, delta)
    covariance += np.einsum("n,n,ni,nj->ij", weight, table.scale1[rows] ** 2 / 4.0,
                            table.tangent1[rows], table.tangent1[rows])
    covariance += np.einsum("n,n,ni,nj->ij", weight, table.scale2[rows] ** 2 / 4.0,
                            table.tangent2[rows], table.tangent2[rows])
    covariance /= max(float(np.sum(weight)), 1e-15)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    normal = eigenvector[:, 0]
    if float(normal @ reference) < 0.0:
        normal = -normal
    normal /= max(float(np.linalg.norm(normal)), 1e-15)
    return center, normal, eigenvalue


def _component_from_seed(graph: csr_matrix, allowed: np.ndarray, seed: int) -> np.ndarray:
    if not allowed[seed]:
        return np.zeros((0,), np.int64)
    visited = np.zeros(allowed.shape, bool)
    visited[seed] = True
    stack = [int(seed)]
    result = []
    while stack:
        row = stack.pop()
        result.append(row)
        neighbors = graph.indices[graph.indptr[row]:graph.indptr[row + 1]]
        fresh = neighbors[allowed[neighbors] & ~visited[neighbors]]
        if fresh.size:
            visited[fresh] = True
            stack.extend(fresh[::-1].tolist())
    return np.asarray(sorted(result), np.int64)


def _compatible(table: PrimitiveSurfaceTable, candidates: np.ndarray, center: np.ndarray, normal: np.ndarray,
                *, normal_degrees: float, plane_distance_m: float) -> np.ndarray:
    cosine = np.abs(table.normals[candidates] @ normal)
    center_distance = np.abs((table.centers[candidates] - center) @ normal)
    support_distance = np.sqrt(
        center_distance ** 2
        + (table.scale1[candidates] * (table.tangent1[candidates] @ normal)) ** 2 / 4.0
        + (table.scale2[candidates] * (table.tangent2[candidates] @ normal)) ** 2 / 4.0
    )
    return (cosine >= np.cos(np.deg2rad(float(normal_degrees)))) & (support_distance <= float(plane_distance_m))


def _robust_group_quality(
    table: PrimitiveSurfaceTable, rows: np.ndarray, center: np.ndarray, normal: np.ndarray,
) -> tuple[float, float, float]:
    center_distance = (table.centers[rows] - center) @ normal
    residual = np.sqrt(
        center_distance ** 2
        + (table.scale1[rows] * (table.tangent1[rows] @ normal)) ** 2 / 4.0
        + (table.scale2[rows] * (table.tangent2[rows] @ normal)) ** 2 / 4.0
    )
    weight = np.pi * table.scale1[rows] * table.scale2[rows] * table.opacity[rows]
    rms = float(np.sqrt(np.average(residual ** 2, weights=np.maximum(weight, 1e-15))))
    return rms, float(np.quantile(residual, .95)), float(np.quantile(np.abs(table.normals[rows] @ normal), .10))


def extract_geometry_native_planar_map(
    table: PrimitiveSurfaceTable, *, neighbors: int = 16,
    adjacency_normal_degrees: float = 12.0, adjacency_plane_distance_m: float = .05,
    adjacency_support_gap_m: float = .10, fit_normal_degrees: float = 15.0,
    fit_plane_distance_m: float = .06, minimum_members: int = 6,
    minimum_support_area_m2: float = .10, maximum_hypotheses: int = 8,
) -> GeometryNativePlanarMap:
    table = table.validated()
    graph = _build_local_graph(
        table, neighbors=neighbors, normal_degrees=adjacency_normal_degrees,
        plane_distance_m=adjacency_plane_distance_m, support_gap_m=adjacency_support_gap_m,
    )
    component_count, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    breaks = np.flatnonzero(np.r_[True, labels[order][1:] != labels[order][:-1], True])
    area = np.pi * table.scale1 * table.scale2 * table.opacity
    groups: list[np.ndarray] = []
    nonplanar = 0
    for begin, end in zip(breaks[:-1], breaks[1:]):
        component = order[begin:end]
        if component.size < int(minimum_members):
            nonplanar += int(component.size)
            continue
        # The desired common case is a complete, already connected wall or
        # roof face.  Accept it in one fit before invoking sequential model
        # extraction; otherwise large clean planes would paradoxically take
        # the slowest seed-by-seed route.
        whole_center, whole_normal, _ = _fit_plane(
            table, component, table.normals[component[np.argmax(area[component])]],
        )
        whole_compatible = _compatible(
            table, component, whole_center, whole_normal,
            normal_degrees=fit_normal_degrees, plane_distance_m=fit_plane_distance_m,
        )
        if np.all(whole_compatible) and float(np.sum(area[component])) >= minimum_support_area_m2:
            groups.append(component)
            continue
        remaining = np.zeros(table.primitive_ids.size, bool)
        remaining[component] = True
        while np.count_nonzero(remaining[component]) >= int(minimum_members):
            candidates = component[remaining[component]]
            ranked = candidates[np.lexsort((table.primitive_ids[candidates], -area[candidates]))]
            if ranked.size > maximum_hypotheses:
                # Area leaders plus a stable spatially distributed ID sample.
                leaders = ranked[:maximum_hypotheses // 2]
                positions = np.linspace(0, ranked.size - 1, maximum_hypotheses - leaders.size, dtype=np.int64)
                seeds = np.unique(np.r_[leaders, ranked[positions]])
            else:
                seeds = ranked
            best = np.zeros((0,), np.int64)
            best_seed = int(seeds[0])
            best_score = -1.0
            for seed in seeds.tolist():
                allowed = np.zeros(remaining.shape, bool)
                allowed[candidates] = _compatible(
                    table, candidates, table.centers[seed], table.normals[seed],
                    normal_degrees=fit_normal_degrees, plane_distance_m=fit_plane_distance_m,
                )
                group = _component_from_seed(graph, allowed, int(seed))
                score = float(np.sum(area[group]))
                key = (score, group.size, -int(table.primitive_ids[seed]))
                best_key = (best_score, best.size, -int(table.primitive_ids[best_seed]))
                if key > best_key:
                    best, best_seed, best_score = group, int(seed), score
            group = best
            for _ in range(3):
                if group.size < minimum_members:
                    break
                center, normal, _ = _fit_plane(table, group, table.normals[best_seed])
                allowed = np.zeros(remaining.shape, bool)
                allowed[candidates] = _compatible(
                    table, candidates, center, normal,
                    normal_degrees=fit_normal_degrees, plane_distance_m=fit_plane_distance_m,
                )
                updated = _component_from_seed(graph, allowed, best_seed)
                if np.array_equal(updated, group):
                    break
                group = updated
            if group.size >= minimum_members and float(np.sum(area[group])) >= minimum_support_area_m2:
                groups.append(group)
                remaining[group] = False
            else:
                # ``best`` maximizes support among all tested hypotheses.  If
                # it cannot satisfy the minimum plane contract, no smaller
                # overlapping hypothesis can do so.  Retire the entire local
                # fragment rather than degrading to quadratic one-seed-at-a-
                # time rejection on vegetation or reconstruction clutter.
                rejected = group if group.size else np.asarray([best_seed], np.int64)
                remaining[rejected] = False
                nonplanar += int(rejected.size)
        nonplanar += int(np.count_nonzero(remaining[component]))

    # Canonicalize independently of graph component enumeration.
    groups.sort(key=lambda rows: int(np.min(table.primitive_ids[rows])))
    count = len(groups)
    normals, offsets, centers, frames = [], [], [], []
    boundaries, boundary_offsets, boundary_area = [], [0], []
    member_offsets, members, member_counts = [0], [], []
    support_area, rms, p95, cosine_p10 = [], [], [], []
    signs = np.asarray([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    for rows in groups:
        reference = table.normals[rows[np.argmax(area[rows])]]
        center, normal, _ = _fit_plane(table, rows, reference)
        plane_frame = _frame(normal)
        center_signed = (table.centers[rows] - center) @ normal
        residual = np.sqrt(
            center_signed ** 2
            + (table.scale1[rows] * (table.tangent1[rows] @ normal)) ** 2 / 4.0
            + (table.scale2[rows] * (table.tangent2[rows] @ normal)) ** 2 / 4.0
        )
        weight = area[rows]
        corners = (
            table.centers[rows, None, :]
            + signs[None, :, :1] * table.scale1[rows, None, None] * table.tangent1[rows, None, :]
            + signs[None, :, 1:] * table.scale2[rows, None, None] * table.tangent2[rows, None, :]
        ).reshape(-1, 3)
        uv = (corners - center) @ plane_frame[:2].T
        try:
            hull = ConvexHull(uv)
            boundary = uv[hull.vertices]
            hull_area = float(hull.volume)
        except QhullError:
            boundary = uv[np.unique(uv, axis=0, return_index=True)[1]]
            hull_area = 0.0
        normals.append(normal); offsets.append(float(normal @ center)); centers.append(center); frames.append(plane_frame)
        boundaries.append(boundary); boundary_offsets.append(boundary_offsets[-1] + boundary.shape[0]); boundary_area.append(hull_area)
        members.append(rows); member_offsets.append(member_offsets[-1] + rows.size); member_counts.append(rows.size)
        support_area.append(float(np.sum(weight)))
        rms.append(float(np.sqrt(np.average(residual ** 2, weights=np.maximum(weight, 1e-15)))))
        p95.append(float(np.quantile(residual, .95)))
        cosine_p10.append(float(np.quantile(np.abs(table.normals[rows] @ normal), .10)))
    arrays = dict(
        plane_ids=np.arange(count, dtype=np.int64),
        normals_world=np.asarray(normals, np.float64).reshape(-1, 3),
        offsets_world=np.asarray(offsets, np.float64), centers_world=np.asarray(centers, np.float64).reshape(-1, 3),
        frames_world=np.asarray(frames, np.float64).reshape(-1, 3, 3),
        boundary_offsets=np.asarray(boundary_offsets, np.int64),
        boundary_uv=np.concatenate(boundaries, axis=0) if boundaries else np.zeros((0, 2), np.float64),
        boundary_area_m2=np.asarray(boundary_area, np.float64),
        member_offsets=np.asarray(member_offsets, np.int64),
        member_primitive_rows=np.concatenate(members).astype(np.int64) if members else np.zeros((0,), np.int64),
        member_counts=np.asarray(member_counts, np.int64), support_area_m2=np.asarray(support_area, np.float64),
        residual_rms_m=np.asarray(rms, np.float64), residual_p95_m=np.asarray(p95, np.float64),
        normal_cosine_p10=np.asarray(cosine_p10, np.float64),
    )
    metadata = {
        "artifact_type": SCHEMA,
        "representation": "finite_connected_planes_from_raw_2dgs_primitive_geometry",
        "uses_parent_child_partition": False,
        "uses_voxel_identity_or_boundary": False,
        "uses_query_pose_or_ground_truth": False,
        "uses_mapping_rgb": False,
        "primitive_count": int(table.primitive_ids.size),
        "plane_count": count,
        "assigned_primitive_count": int(arrays["member_primitive_rows"].size),
        "unassigned_primitive_count": int(table.primitive_ids.size - arrays["member_primitive_rows"].size),
        "broad_graph_component_count": int(component_count),
        "connectivity": "3d_ellipse_support_knn_graph",
        "boundary": "convex_summary_plus_exact_member_primitive_inventory",
        "configuration": {
            "neighbors": int(neighbors), "adjacency_normal_degrees": float(adjacency_normal_degrees),
            "adjacency_plane_distance_m": float(adjacency_plane_distance_m),
            "adjacency_support_gap_m": float(adjacency_support_gap_m),
            "fit_normal_degrees": float(fit_normal_degrees), "fit_plane_distance_m": float(fit_plane_distance_m),
            "minimum_members": int(minimum_members), "minimum_support_area_m2": float(minimum_support_area_m2),
            "maximum_hypotheses": int(maximum_hypotheses),
        },
    }
    return GeometryNativePlanarMap(metadata=metadata, **arrays).validated(table.primitive_ids.size)


def merge_adjacent_coplanar_planes(
    physical: GoalMapletPhysicalMap,
    planar: GeometryNativePlanarMap,
    *, boundary_gap_m: float = .40, normal_degrees: float = 6.0,
    reciprocal_plane_distance_m: float = .05, fit_normal_degrees: float = 15.0,
    fit_plane_distance_m: float = .06, boundary_sample_step_m: float = .20,
) -> GeometryNativePlanarMap:
    """Merge plane fragments across small reconstruction holes.

    Candidate edges come from finite boundary proximity, not global plane
    equation similarity.  Every accepted union is then refitted using all raw
    member ellipses, so distant parallel facades cannot merge and curved chains
    cannot accumulate silently.
    """
    table = PrimitiveSurfaceTable.from_physical_map(physical)
    planar = planar.validated(table.primitive_ids.size)
    count = planar.plane_ids.size
    samples, sample_owner = [], []
    for row in range(count):
        lo, hi = int(planar.boundary_offsets[row]), int(planar.boundary_offsets[row + 1])
        uv = planar.boundary_uv[lo:hi]
        if uv.shape[0] < 2:
            continue
        world = (
            planar.centers_world[row][None]
            + uv[:, :1] * planar.frames_world[row, 0][None]
            + uv[:, 1:] * planar.frames_world[row, 1][None]
        )
        for begin, end in zip(world, np.roll(world, -1, axis=0)):
            pieces = max(1, int(np.ceil(np.linalg.norm(end - begin) / float(boundary_sample_step_m))))
            alpha = np.arange(pieces, dtype=np.float64) / pieces
            samples.append(begin[None] * (1.0 - alpha[:, None]) + end[None] * alpha[:, None])
            sample_owner.extend([row] * pieces)
    if not samples:
        return planar
    sample_xyz = np.concatenate(samples, axis=0)
    sample_owner = np.asarray(sample_owner, np.int64)
    sample_pairs = cKDTree(sample_xyz).query_pairs(float(boundary_gap_m), output_type="ndarray")
    candidate_gap: dict[tuple[int, int], float] = {}
    if sample_pairs.size:
        left = sample_owner[sample_pairs[:, 0]]; right = sample_owner[sample_pairs[:, 1]]
        valid = left != right
        for a, b, distance in zip(
            left[valid].tolist(), right[valid].tolist(),
            np.linalg.norm(sample_xyz[sample_pairs[valid, 0]] - sample_xyz[sample_pairs[valid, 1]], axis=1).tolist(),
        ):
            key = (min(a, b), max(a, b))
            candidate_gap[key] = min(candidate_gap.get(key, np.inf), float(distance))
    edges = []
    cosine_threshold = np.cos(np.deg2rad(float(normal_degrees)))
    for (left, right), gap in candidate_gap.items():
        if abs(float(planar.normals_world[left] @ planar.normals_world[right])) < cosine_threshold:
            continue
        if abs(float(planar.normals_world[left] @ planar.centers_world[right] - planar.offsets_world[left])) > reciprocal_plane_distance_m:
            continue
        if abs(float(planar.normals_world[right] @ planar.centers_world[left] - planar.offsets_world[right])) > reciprocal_plane_distance_m:
            continue
        edges.append((gap, left, right))
    owner = np.arange(count, dtype=np.int64)
    groups = {
        row: planar.member_primitive_rows[int(planar.member_offsets[row]):int(planar.member_offsets[row + 1])].copy()
        for row in range(count)
    }

    def find(value: int) -> int:
        while owner[value] != value:
            owner[value] = owner[owner[value]]
            value = int(owner[value])
        return value

    accepted = rejected = 0
    for _gap, left, right in sorted(edges):
        a, b = find(left), find(right)
        if a == b:
            continue
        rows = np.unique(np.r_[groups[a], groups[b]])
        reference = planar.normals_world[a]
        center, normal, _ = _fit_plane(table, rows, reference)
        rms, p95, cosine_p10 = _robust_group_quality(table, rows, center, normal)
        if (rms > .05 or p95 > .08 or cosine_p10 < np.cos(np.deg2rad(float(fit_normal_degrees)))):
            rejected += 1
            continue
        root, other = min(a, b), max(a, b)
        owner[other] = root
        groups[root] = rows
        del groups[other]
        accepted += 1
    final_groups = sorted(groups.values(), key=lambda rows: int(np.min(table.primitive_ids[rows])))
    from .planar_maplet_oracle import _fit_member_groups
    reference = np.asarray([
        table.normals[rows[np.argmax(np.pi * table.scale1[rows] * table.scale2[rows] * table.opacity[rows])]]
        for rows in final_groups
    ])
    fitted = _fit_member_groups(physical, final_groups, reference)
    offsets = [0]
    for rows in final_groups:
        offsets.append(offsets[-1] + rows.size)
    metadata = dict(planar.metadata)
    metadata.update({
        "artifact_type": SCHEMA,
        "representation": "finite_connected_planes_from_raw_2dgs_geometry_with_component_refit_merge",
        "plane_count": len(final_groups),
        "premerge_plane_count": int(count),
        "candidate_merge_edge_count": len(edges),
        "accepted_merge_count": int(accepted),
        "rejected_merge_count": int(rejected),
        "merge_uses_parent_child_partition": False,
        "merge_boundary_gap_m": float(boundary_gap_m),
        "merge_normal_degrees": float(normal_degrees),
        "merge_reciprocal_plane_distance_m": float(reciprocal_plane_distance_m),
        "merge_full_component_refit_with_atomic_rollback": True,
    })
    result = GeometryNativePlanarMap(
        plane_ids=np.arange(len(final_groups), dtype=np.int64),
        normals_world=fitted.normals_world, offsets_world=fitted.offsets_world,
        centers_world=fitted.centers_world, frames_world=fitted.frames_world,
        boundary_offsets=fitted.boundary_offsets, boundary_uv=fitted.boundary_uv,
        boundary_area_m2=fitted.boundary_area_m2,
        member_offsets=np.asarray(offsets, np.int64),
        member_primitive_rows=np.concatenate(final_groups).astype(np.int64),
        member_counts=fitted.member_counts, support_area_m2=fitted.member_area_m2,
        residual_rms_m=fitted.center_residual_rms_m,
        residual_p95_m=fitted.center_residual_p95_m,
        normal_cosine_p10=fitted.normal_cosine_p10, metadata=metadata,
    )
    return result.validated(table.primitive_ids.size)


def merge_covisible_coplanar_planes(
    physical: GoalMapletPhysicalMap,
    planar: GeometryNativePlanarMap,
    contributor_paths: list[Path],
    *, minimum_covisible_views: int = 2, normal_degrees: float = 6.0,
    reciprocal_plane_distance_m: float = .05, fit_normal_degrees: float = 15.0,
    fit_plane_distance_m: float = .06,
) -> GeometryNativePlanarMap:
    """Merge fragments with repeated image-space boundary adjacency.

    Only ``topk_ids`` is read from each contributor archive.  Camera pose,
    image appearance, and query data are neither required nor accessed.
    """
    table = PrimitiveSurfaceTable.from_physical_map(physical)
    planar = planar.validated(table.primitive_ids.size)
    primitive_owner = np.full(table.primitive_ids.size, -1, np.int64)
    for plane in range(planar.plane_ids.size):
        lo, hi = int(planar.member_offsets[plane]), int(planar.member_offsets[plane + 1])
        primitive_owner[planar.member_primitive_rows[lo:hi]] = plane
    maximum_id = int(np.max(table.primitive_ids))
    row_by_id = np.full(maximum_id + 1, -1, np.int32)
    row_by_id[table.primitive_ids] = np.arange(table.primitive_ids.size, dtype=np.int32)
    view_count: dict[tuple[int, int], int] = {}
    for path in sorted(map(Path, contributor_paths)):
        with np.load(path, allow_pickle=False) as data:
            ids = np.asarray(data["topk_ids"][:, :, 0], np.int64)
        valid_id = (ids >= 0) & (ids <= maximum_id)
        rows = np.full(ids.shape, -1, np.int64)
        rows[valid_id] = row_by_id[ids[valid_id]]
        owners = np.full(ids.shape, -1, np.int64)
        valid_row = rows >= 0
        owners[valid_row] = primitive_owner[rows[valid_row]]
        pairs = []
        for left, right in ((owners[:, :-1], owners[:, 1:]), (owners[:-1], owners[1:])):
            valid = (left >= 0) & (right >= 0) & (left != right)
            if np.any(valid):
                pair = np.sort(np.stack((left[valid], right[valid]), axis=1), axis=1)
                pairs.append(pair)
        if not pairs:
            continue
        for left, right in np.unique(np.concatenate(pairs), axis=0).tolist():
            key = (int(left), int(right))
            view_count[key] = view_count.get(key, 0) + 1
    cosine_threshold = np.cos(np.deg2rad(float(normal_degrees)))
    edges = []
    for (left, right), views in view_count.items():
        if views < int(minimum_covisible_views):
            continue
        if abs(float(planar.normals_world[left] @ planar.normals_world[right])) < cosine_threshold:
            continue
        if abs(float(planar.normals_world[left] @ planar.centers_world[right] - planar.offsets_world[left])) > reciprocal_plane_distance_m:
            continue
        if abs(float(planar.normals_world[right] @ planar.centers_world[left] - planar.offsets_world[right])) > reciprocal_plane_distance_m:
            continue
        edges.append((-views, left, right))
    count = planar.plane_ids.size
    owner = np.arange(count, dtype=np.int64)
    groups = {
        row: planar.member_primitive_rows[int(planar.member_offsets[row]):int(planar.member_offsets[row + 1])].copy()
        for row in range(count)
    }
    def find(value: int) -> int:
        while owner[value] != value:
            owner[value] = owner[owner[value]]
            value = int(owner[value])
        return value
    accepted = rejected = 0
    for _negative_views, left, right in sorted(edges):
        a, b = find(left), find(right)
        if a == b:
            continue
        rows = np.unique(np.r_[groups[a], groups[b]])
        center, normal, _ = _fit_plane(table, rows, planar.normals_world[a])
        rms, p95, cosine_p10 = _robust_group_quality(table, rows, center, normal)
        if (rms > .05 or p95 > .08 or cosine_p10 < np.cos(np.deg2rad(float(fit_normal_degrees)))):
            rejected += 1
            continue
        root, other = min(a, b), max(a, b)
        owner[other] = root; groups[root] = rows; del groups[other]; accepted += 1
    final_groups = sorted(groups.values(), key=lambda rows: int(np.min(table.primitive_ids[rows])))
    from .planar_maplet_oracle import _fit_member_groups
    area = np.pi * table.scale1 * table.scale2 * table.opacity
    reference = np.asarray([table.normals[rows[np.argmax(area[rows])]] for rows in final_groups])
    fitted = _fit_member_groups(physical, final_groups, reference)
    offsets = np.r_[0, np.cumsum([rows.size for rows in final_groups])].astype(np.int64)
    metadata = dict(planar.metadata)
    metadata.update({
        "representation": "finite_planes_with_mapping_covisibility_connectivity",
        "plane_count": len(final_groups), "pre_covisibility_plane_count": int(count),
        "contributor_view_count": len(contributor_paths),
        "covisibility_candidate_pair_count": len(view_count),
        "covisibility_eligible_edge_count": len(edges),
        "covisibility_accepted_merge_count": int(accepted),
        "covisibility_rejected_merge_count": int(rejected),
        "minimum_covisible_views": int(minimum_covisible_views),
        "covisibility_reads_only_topk_ids": True,
        "covisibility_reads_camera_pose": False,
        "covisibility_reads_rgb_or_radio": False,
    })
    return GeometryNativePlanarMap(
        plane_ids=np.arange(len(final_groups), dtype=np.int64),
        normals_world=fitted.normals_world, offsets_world=fitted.offsets_world,
        centers_world=fitted.centers_world, frames_world=fitted.frames_world,
        boundary_offsets=fitted.boundary_offsets, boundary_uv=fitted.boundary_uv,
        boundary_area_m2=fitted.boundary_area_m2, member_offsets=offsets,
        member_primitive_rows=np.concatenate(final_groups).astype(np.int64),
        member_counts=fitted.member_counts, support_area_m2=fitted.member_area_m2,
        residual_rms_m=fitted.center_residual_rms_m, residual_p95_m=fitted.center_residual_p95_m,
        normal_cosine_p10=fitted.normal_cosine_p10, metadata=metadata,
    ).validated(table.primitive_ids.size)


__all__ = [
    "SCHEMA", "PrimitiveSurfaceTable", "GeometryNativePlanarMap",
    "extract_geometry_native_planar_map", "merge_adjacent_coplanar_planes",
    "merge_covisible_coplanar_planes",
]
