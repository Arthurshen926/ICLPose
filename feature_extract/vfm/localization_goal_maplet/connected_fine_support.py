"""Deterministic connected, variable-scale regions over selected child voxels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fine_support_selection import child_surface_area_m2
from .physical_map import GoalMapletPhysicalMap


COMPONENT_SEMANTICS = "six_connected_metric_voxels_unsigned_normal_axis_v1"


@dataclass(frozen=True)
class ConnectedFineSupportSet:
    component_offsets: np.ndarray
    component_child_rows: np.ndarray
    component_centers: np.ndarray
    component_extents: np.ndarray
    component_surface_area_m2: np.ndarray

    def __post_init__(self) -> None:
        offsets = np.asarray(self.component_offsets, dtype=np.int64).reshape(-1)
        rows = np.asarray(self.component_child_rows, dtype=np.int64).reshape(-1)
        count = max(offsets.size - 1, 0)
        centers = np.asarray(self.component_centers, dtype=np.float64)
        extents = np.asarray(self.component_extents, dtype=np.float64)
        area = np.asarray(self.component_surface_area_m2, dtype=np.float64).reshape(-1)
        if (
            offsets.size == 0
            or offsets[0] != 0
            or offsets[-1] != rows.size
            or np.any(offsets[1:] < offsets[:-1])
            or np.unique(rows).size != rows.size
            or centers.shape != (count, 3)
            or extents.shape != (count, 3)
            or area.shape != (count,)
            or np.any(~np.isfinite(centers))
            or np.any(~np.isfinite(extents))
            or np.any(extents < 0.0)
            or np.any(~np.isfinite(area))
            or np.any(area <= 0.0)
        ):
            raise ValueError("invalid connected fine-support set")
        object.__setattr__(self, "component_offsets", offsets)
        object.__setattr__(self, "component_child_rows", rows)
        object.__setattr__(self, "component_centers", centers)
        object.__setattr__(self, "component_extents", extents)
        object.__setattr__(self, "component_surface_area_m2", area)

    @property
    def component_count(self) -> int:
        return int(self.component_offsets.size - 1)


def connected_fine_support_components(
    selected_child_rows: np.ndarray,
    physical: GoalMapletPhysicalMap,
    *,
    maximum_normal_angle_degrees: float = 30.0,
    precomputed_child_surface_area_m2: np.ndarray | None = None,
) -> ConnectedFineSupportSet:
    """Merge face-adjacent 1m child voxels into variable-scale supports.

    Connectivity is geometry-only: a shared voxel face, the same dominant
    unsigned normal axis, and bounded normal angle.  It may cross arbitrary
    5m parent voxel boundaries, so parent partition seams do not fragment a
    physical surface.  The union of child/primitive support is unchanged.
    """

    rows = np.unique(np.asarray(selected_child_rows, dtype=np.int64).reshape(-1))
    child_count = int(physical.child_parent_rows.size)
    child_size = float(physical.metadata.get("child_voxel_size_m", 0.0))
    angle = float(maximum_normal_angle_degrees)
    if (
        np.any(rows < 0)
        or np.any(rows >= child_count)
        or not np.isfinite(child_size)
        or child_size <= 0.0
        or not np.isfinite(angle)
        or not 0.0 <= angle <= 90.0
    ):
        raise ValueError("invalid connected fine-support input")
    if rows.size == 0:
        return ConnectedFineSupportSet(
            np.asarray([0], dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    centers = np.asarray(physical.child_centers[rows], dtype=np.float64)
    normals = np.asarray(physical.child_normals[rows], dtype=np.float64)
    voxel = np.floor(centers / child_size).astype(np.int64)
    axis = np.argmax(np.abs(normals), axis=1).astype(np.int64)
    by_key: dict[tuple[int, int, int, int], list[int]] = {}
    for local in range(rows.size):
        key = tuple(voxel[local].tolist()) + (int(axis[local]),)
        by_key.setdefault(key, []).append(local)

    parent = np.arange(rows.size, dtype=np.int64)

    def find(value: int) -> int:
        root = int(value)
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            if int(rows[a]) > int(rows[b]):
                a, b = b, a
            parent[b] = a

    threshold = float(np.cos(np.deg2rad(angle)))
    directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for key, left_rows in by_key.items():
        # Parent partition seams can create multiple child rows in the same
        # metric voxel.  They are part of the same support whenever their
        # unsigned normals agree; omitting this zero-distance union leaves an
        # artificial seam even before checking the six neighbouring faces.
        for local_index, left in enumerate(left_rows):
            for right in left_rows[local_index + 1 :]:
                if float(abs(np.dot(normals[left], normals[right]))) >= threshold:
                    union(left, right)
        for dx, dy, dz in directions:
            neighbour = (key[0] + dx, key[1] + dy, key[2] + dz, key[3])
            for left in left_rows:
                for right in by_key.get(neighbour, ()):
                    if float(abs(np.dot(normals[left], normals[right]))) >= threshold:
                        union(left, right)

    groups: dict[int, list[int]] = {}
    for local in range(rows.size):
        groups.setdefault(find(local), []).append(int(rows[local]))
    ordered = sorted(groups.values(), key=lambda values: min(values))
    area = (
        child_surface_area_m2(physical)
        if precomputed_child_surface_area_m2 is None
        else np.asarray(precomputed_child_surface_area_m2, dtype=np.float64).reshape(-1)
    )
    if (
        area.shape != (child_count,)
        or np.any(~np.isfinite(area))
        or np.any(area <= 0.0)
    ):
        raise ValueError("invalid precomputed child-area ledger")
    offsets = [0]
    flat: list[int] = []
    output_centers: list[np.ndarray] = []
    output_extents: list[np.ndarray] = []
    output_area: list[float] = []
    half_voxel = 0.5 * child_size
    for values in ordered:
        component = np.asarray(sorted(values), dtype=np.int64)
        component_area = area[component]
        low = np.min(
            np.floor(physical.child_centers[component] / child_size) * child_size,
            axis=0,
        )
        high = np.max(
            (np.floor(physical.child_centers[component] / child_size) + 1.0)
            * child_size,
            axis=0,
        )
        # The axis-aligned box is only a compact retrieval-region carrier; it
        # never replaces the exact primitive union used by evaluation.
        output_centers.append(0.5 * (low + high))
        output_extents.append(np.maximum(0.5 * (high - low), half_voxel))
        output_area.append(float(np.sum(component_area)))
        flat.extend(component.tolist())
        offsets.append(len(flat))
    return ConnectedFineSupportSet(
        np.asarray(offsets, dtype=np.int64),
        np.asarray(flat, dtype=np.int64),
        np.asarray(output_centers, dtype=np.float64),
        np.asarray(output_extents, dtype=np.float64),
        np.asarray(output_area, dtype=np.float64),
    )
