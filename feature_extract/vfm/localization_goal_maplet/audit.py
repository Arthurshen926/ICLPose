"""Physical-map audits that gate all retrieval and pose experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from .physical_map import DOUBLE_SIDED, GoalMapletPhysicalMap


def _stats(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "median": float(np.median(data)),
        "mean": float(np.mean(data)),
        "p90": float(np.percentile(data, 90.0)),
        "max": float(np.max(data)),
    }


def _component_count(points: np.ndarray, radii: np.ndarray) -> int:
    count = int(points.shape[0])
    if count <= 1:
        return count
    search_radius = max(float(np.percentile(radii, 90.0) * 2.5 + 0.03), 0.05)
    pairs = cKDTree(points).query_pairs(search_radius, output_type="ndarray")
    parent = np.arange(count, dtype=np.int64)

    def find(row: int) -> int:
        while parent[row] != row:
            parent[row] = parent[parent[row]]
            row = int(parent[row])
        return row

    for left, right in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
        root_left, root_right = find(int(left)), find(int(right))
        if root_left != root_right:
            parent[root_right] = root_left
    return int(np.unique([find(row) for row in range(count)]).size)


def _compactness(local_uv: np.ndarray, radii: np.ndarray) -> float:
    if local_uv.shape[0] < 3:
        return 1.0
    try:
        hull = ConvexHull(local_uv)
        hull_area = float(hull.volume)
        hull_perimeter = float(hull.area)
    except Exception:
        return 0.0
    # Isoperimetric compactness is overlap-independent and exposes elongated
    # or fragmented envelopes; summed disk area saturated at one on this map.
    return float(np.clip(4.0 * np.pi * hull_area / max(hull_perimeter**2, 1e-12), 0.0, 1.0))


@dataclass(frozen=True)
class PhysicalAuditThresholds:
    maximum_disconnected_fraction: float = 0.10
    maximum_normal_p90_degrees: float = 25.0
    maximum_excess_depth_fraction: float = 0.10
    maximum_double_sided_maplet_fraction: float = 0.35
    minimum_members_per_maplet: int = 2
    minimum_children_per_nontrivial_maplet: int = 3


def audit_physical_map(
    physical_map: GoalMapletPhysicalMap,
    thresholds: PhysicalAuditThresholds = PhysicalAuditThresholds(),
) -> dict[str, object]:
    member_counts = np.diff(physical_map.membership_offsets)
    child_counts = np.diff(physical_map.maplet_child_offsets)
    component_counts: list[int] = []
    normal_p90: list[float] = []
    depth_ratios: list[float] = []
    compactness: list[float] = []
    unique_owned: list[np.ndarray] = []
    for row in range(physical_map.maplet_ids.size):
        link = physical_map.member_slice(row)
        primitive_rows = physical_map.membership_primitive_rows[link]
        unique_owned.append(primitive_rows)
        points = physical_map.primitive_centers[primitive_rows]
        radii = np.maximum(
            physical_map.primitive_scale1[primitive_rows],
            physical_map.primitive_scale2[primitive_rows],
        )
        local = (points - physical_map.maplet_centers[row]) @ physical_map.maplet_frames[row].T
        component_counts.append(_component_count(points, radii))
        cosines = np.clip(
            physical_map.primitive_normals[primitive_rows] @ physical_map.maplet_normals[row],
            -1.0,
            1.0,
        )
        cosines = np.where(
            physical_map.primitive_sidedness[primitive_rows] == DOUBLE_SIDED,
            np.abs(cosines),
            cosines,
        )
        normal_p90.append(float(np.percentile(np.degrees(np.arccos(cosines)), 90.0)))
        depth_ratios.append(float(np.ptp(local[:, 2]) / max(2.0 * np.max(physical_map.maplet_extents[row, :2]), 1e-6)))
        compactness.append(_compactness(local[:, :2], radii))
    child_component_counts: list[int] = []
    child_normal_p90: list[float] = []
    child_depth_ratios: list[float] = []
    for child_row in range(physical_map.child_parent_rows.size):
        start, end = int(physical_map.child_member_offsets[child_row]), int(physical_map.child_member_offsets[child_row + 1])
        primitive_rows = physical_map.child_member_primitive_rows[start:end]
        points = physical_map.primitive_centers[primitive_rows]
        radii = np.maximum(
            physical_map.primitive_scale1[primitive_rows],
            physical_map.primitive_scale2[primitive_rows],
        )
        child_component_counts.append(_component_count(points, radii))
        cosines = np.clip(
            physical_map.primitive_normals[primitive_rows] @ physical_map.child_normals[child_row],
            -1.0,
            1.0,
        )
        cosines = np.where(
            physical_map.primitive_sidedness[primitive_rows] == DOUBLE_SIDED,
            np.abs(cosines),
            cosines,
        )
        child_normal_p90.append(float(np.percentile(np.degrees(np.arccos(cosines)), 90.0)))
        local = (points - physical_map.child_centers[child_row]) @ physical_map.child_frames[child_row].T
        child_depth_ratios.append(float(np.ptp(local[:, 2]) / max(2.0 * np.max(physical_map.child_extents[child_row, :2]), 1e-6)))
    owned = np.concatenate(unique_owned) if unique_owned else np.zeros((0,), dtype=np.int64)
    _, overlap = np.unique(owned, return_counts=True)
    disconnected_fraction = float(np.mean(np.asarray(component_counts) > 1))
    excess_depth_fraction = float(np.mean(np.asarray(depth_ratios) > 0.25))
    child_disconnected_fraction = float(np.mean(np.asarray(child_component_counts) > 1))
    child_excess_depth_fraction = float(np.mean(np.asarray(child_depth_ratios) > 0.25))
    double_sided_fraction = float(np.mean(physical_map.maplet_sidedness == DOUBLE_SIDED))
    gates = {
        "no_empty_or_tiny_maplets": bool(np.min(member_counts, initial=thresholds.minimum_members_per_maplet) >= thresholds.minimum_members_per_maplet),
        "connected_surface_support": bool(child_disconnected_fraction <= thresholds.maximum_disconnected_fraction),
        "normal_purity": bool(np.percentile(child_normal_p90, 90.0) <= thresholds.maximum_normal_p90_degrees),
        "single_depth_layer": bool(child_excess_depth_fraction <= thresholds.maximum_excess_depth_fraction),
        "normal_orientation_resolved": bool(double_sided_fraction <= thresholds.maximum_double_sided_maplet_fraction),
        "metric_child_support": bool(np.min(child_counts[member_counts >= 8], initial=thresholds.minimum_children_per_nontrivial_maplet) >= thresholds.minimum_children_per_nontrivial_maplet),
    }
    return {
        "stage": "goal_maplet_physical_map_audit",
        "artifact_content_sha256": physical_map.content_sha256,
        "maplet_count": int(physical_map.maplet_ids.size),
        "scene_primitive_count": int(physical_map.primitive_ids.size),
        "owned_primitive_count": int(np.unique(owned).size),
        "membership_count": int(owned.size),
        "overlapping_owned_primitive_fraction": float(np.mean(overlap > 1)) if overlap.size else 0.0,
        "child_tile_count": int(physical_map.child_parent_rows.size),
        "members_per_maplet": _stats(member_counts),
        "children_per_maplet": _stats(child_counts),
        "connected_components_per_maplet": _stats(np.asarray(component_counts)),
        "disconnected_maplet_fraction": disconnected_fraction,
        "normal_dispersion_p90_degrees_per_maplet": _stats(np.asarray(normal_p90)),
        "relative_depth_span_per_maplet": _stats(np.asarray(depth_ratios)),
        "excess_depth_maplet_fraction": excess_depth_fraction,
        "boundary_compactness_per_maplet": _stats(np.asarray(compactness)),
        "base_tile_connected_components": _stats(np.asarray(child_component_counts)),
        "disconnected_base_tile_fraction": child_disconnected_fraction,
        "base_tile_normal_dispersion_p90_degrees": _stats(np.asarray(child_normal_p90)),
        "base_tile_relative_depth_span": _stats(np.asarray(child_depth_ratios)),
        "excess_depth_base_tile_fraction": child_excess_depth_fraction,
        "double_sided_maplet_fraction": double_sided_fraction,
        "orientation_confidence": _stats(physical_map.maplet_orientation_confidence),
        "gates": gates,
        "physical_map_ready": bool(all(gates.values())),
    }
