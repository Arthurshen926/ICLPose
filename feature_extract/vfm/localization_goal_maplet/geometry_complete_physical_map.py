"""Geometry-only, coverage-complete parent/child hierarchy for pure retrieval.

The legacy physical map kept the full clean 2DGS only as an occluder table;
its retrievable parent/child memberships covered a small acquisition-selected
subset.  This module builds a deterministic control hierarchy from every
frozen clean primitive.  No image, pose, route, RADIO feature, or ground-truth
input is accepted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .physical_map import DOUBLE_SIDED, GoalMapletPhysicalMap, SCHEMA


GEOMETRY_COMPLETE_PARTITION = (
    "world_metric_voxel_child_unsigned_dominant_normal_axis_parent_partition_v1"
)


def _unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def _unsigned_normals(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _unit(value)
    dominant = np.argmax(np.abs(normal), axis=1)
    sign = np.where(normal[np.arange(normal.shape[0]), dominant] < 0.0, -1.0, 1.0)
    return normal * sign[:, None], sign


def _frame(normal: np.ndarray) -> np.ndarray:
    n = _unit(np.asarray(normal, dtype=np.float64).reshape(1, 3))[0]
    seed = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(n)))]
    tangent = seed - n * float(seed @ n)
    tangent = _unit(tangent.reshape(1, 3))[0]
    second = _unit(np.cross(n, tangent).reshape(1, 3))[0]
    return np.stack([tangent, second, n], axis=0)


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class GeometryCompleteBuildAudit:
    parent_voxel_keys: np.ndarray
    child_voxel_normal_keys: np.ndarray
    unique_child_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "parent_voxel_keys_sha256": _hash_array(self.parent_voxel_keys),
            "child_voxel_normal_keys_sha256": _hash_array(
                self.child_voxel_normal_keys
            ),
            "parent_count": int(self.parent_voxel_keys.shape[0]),
            "unique_child_count": int(self.unique_child_count),
        }


def build_geometry_complete_physical_map(
    geometry_carrier: GoalMapletPhysicalMap,
    *,
    child_voxel_size_m: float = 1.0,
    parent_voxel_size_m: float = 5.0,
    metadata: Mapping[str, object] | None = None,
) -> tuple[GoalMapletPhysicalMap, GeometryCompleteBuildAudit]:
    """Partition every frozen primitive into one child and one parent.

    Children separate perpendicular surfaces inside the same metric voxel via
    the dominant axis of a canonical unsigned normal.  Parents are coarse
    world-coordinate context cells.  The hard partition is deliberately the
    simplest coverage-complete control; overlapping context neighborhoods can
    be evaluated later without conflating coverage with overlap policy.
    """

    child_size = float(child_voxel_size_m)
    parent_size = float(parent_voxel_size_m)
    if (
        not np.isfinite(child_size)
        or not np.isfinite(parent_size)
        or child_size <= 0.0
        or parent_size < child_size
    ):
        raise ValueError("invalid geometry-complete voxel sizes")
    centers = np.asarray(geometry_carrier.primitive_centers, dtype=np.float64)
    primitive_count = int(centers.shape[0])
    if primitive_count == 0:
        raise ValueError("geometry carrier is empty")
    unsigned_normal, normal_sign = _unsigned_normals(
        geometry_carrier.primitive_normals
    )
    dominant_axis = np.argmax(np.abs(unsigned_normal), axis=1).astype(np.int64)
    child_voxel = np.floor(centers / child_size).astype(np.int64)
    child_key_per_primitive = np.column_stack([child_voxel, dominant_axis])
    child_keys, primitive_child = np.unique(
        child_key_per_primitive, axis=0, return_inverse=True
    )
    preliminary_child_count = int(child_keys.shape[0])
    primitive_area_mass = np.maximum(
        np.pi
        * np.asarray(geometry_carrier.primitive_scale1, dtype=np.float64)
        * np.asarray(geometry_carrier.primitive_scale2, dtype=np.float64)
        * np.maximum(
            np.asarray(geometry_carrier.primitive_opacity, dtype=np.float64),
            1e-3,
        ),
        1e-10,
    )
    child_mass = np.bincount(
        primitive_child, weights=primitive_area_mass, minlength=preliminary_child_count
    )
    preliminary_child_centers = np.column_stack(
        [
            np.bincount(
                primitive_child,
                weights=primitive_area_mass * centers[:, axis],
                minlength=preliminary_child_count,
            )
            / child_mass
            for axis in range(3)
        ]
    )
    parent_key_per_child = np.floor(
        preliminary_child_centers / parent_size
    ).astype(np.int64)
    parent_keys, preliminary_child_parent = np.unique(
        parent_key_per_child, axis=0, return_inverse=True
    )
    parent_count = int(parent_keys.shape[0])

    # GoalMapletPhysicalMap requires all children of a parent to be contiguous.
    child_order = np.lexsort(
        (
            child_keys[:, 3],
            child_keys[:, 2],
            child_keys[:, 1],
            child_keys[:, 0],
            preliminary_child_parent,
        )
    )
    inverse_child_order = np.empty_like(child_order)
    inverse_child_order[child_order] = np.arange(child_order.size, dtype=np.int64)
    primitive_child = inverse_child_order[primitive_child]
    child_keys = child_keys[child_order]
    child_parent = preliminary_child_parent[child_order]
    child_count = int(child_keys.shape[0])
    child_offsets_from_primitives = np.r_[
        0, np.cumsum(np.bincount(primitive_child, minlength=child_count))
    ].astype(np.int64)
    primitive_order = np.argsort(primitive_child, kind="stable")

    child_centers: list[np.ndarray] = []
    child_normals: list[np.ndarray] = []
    child_frames: list[np.ndarray] = []
    child_extents: list[np.ndarray] = []
    child_local_uv: list[np.ndarray] = []
    child_member_weights: list[np.ndarray] = []
    for child_row in range(child_count):
        start, end = (
            int(child_offsets_from_primitives[child_row]),
            int(child_offsets_from_primitives[child_row + 1]),
        )
        rows = primitive_order[start:end]
        mass = primitive_area_mass[rows]
        center = np.average(centers[rows], axis=0, weights=mass)
        normal = _unit(
            np.sum(unsigned_normal[rows] * mass[:, None], axis=0).reshape(1, 3)
        )[0]
        frame = _frame(normal)
        local = (centers[rows] - center) @ frame.T
        radius = np.maximum(
            np.asarray(geometry_carrier.primitive_scale1)[rows],
            np.asarray(geometry_carrier.primitive_scale2)[rows],
        )
        child_centers.append(center)
        child_normals.append(normal)
        child_frames.append(frame)
        child_extents.append(np.max(np.abs(local) + radius[:, None], axis=0))
        child_local_uv.append(local[:, :2].astype(np.float32))
        child_member_weights.append(mass.astype(np.float32))

    # Primitive membership follows the already parent-grouped child order.
    parent_primitive_rows: list[np.ndarray] = []
    parent_membership_weights: list[np.ndarray] = []
    parent_boundary_weights: list[np.ndarray] = []
    parent_centers: list[np.ndarray] = []
    parent_normals: list[np.ndarray] = []
    parent_frames: list[np.ndarray] = []
    parent_extents: list[np.ndarray] = []
    maplet_child_offsets = np.r_[
        0, np.cumsum(np.bincount(child_parent, minlength=parent_count))
    ].astype(np.int64)
    for parent_row in range(parent_count):
        child_start, child_end = (
            int(maplet_child_offsets[parent_row]),
            int(maplet_child_offsets[parent_row + 1]),
        )
        grouped_rows = []
        for child_row in range(child_start, child_end):
            start, end = (
                int(child_offsets_from_primitives[child_row]),
                int(child_offsets_from_primitives[child_row + 1]),
            )
            grouped_rows.append(primitive_order[start:end])
        rows = np.concatenate(grouped_rows)
        mass = primitive_area_mass[rows]
        center = np.average(centers[rows], axis=0, weights=mass)
        normal = _unit(
            np.sum(unsigned_normal[rows] * mass[:, None], axis=0).reshape(1, 3)
        )[0]
        frame = _frame(normal)
        local = (centers[rows] - center) @ frame.T
        radius = np.maximum(
            np.asarray(geometry_carrier.primitive_scale1)[rows],
            np.asarray(geometry_carrier.primitive_scale2)[rows],
        )
        parent_centers.append(center)
        parent_normals.append(normal)
        parent_frames.append(frame)
        parent_extents.append(np.max(np.abs(local) + radius[:, None], axis=0))
        parent_primitive_rows.append(rows.astype(np.int64))
        parent_membership_weights.append(mass.astype(np.float32))
        voxel_low = parent_keys[parent_row].astype(np.float64) * parent_size
        fractional = np.clip((centers[rows] - voxel_low) / parent_size, 0.0, 1.0)
        edge_distance = np.min(
            np.concatenate([fractional, 1.0 - fractional], axis=1), axis=1
        )
        parent_boundary_weights.append(
            np.clip(1.0 - 4.0 * edge_distance, 0.0, 1.0).astype(np.float32)
        )

    membership_offsets = np.r_[
        0, np.cumsum([value.size for value in parent_primitive_rows])
    ].astype(np.int64)
    primitive_normals = unsigned_normal.copy()
    primitive_tangent2 = np.asarray(
        geometry_carrier.primitive_tangent2, dtype=np.float64
    ).copy()
    primitive_tangent2[normal_sign < 0.0] *= -1.0
    output_metadata = {
        "artifact_type": SCHEMA,
        "representation": "geometry_only_coverage_complete_primitive_child_parent_partition",
        "vfm_layer": "radio_final",
        "stores_mapping_rgb": False,
        "stores_mapping_image_paths": False,
        "stores_mapping_image_ids": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_point_correspondences": False,
        "uses_mapping_pose": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "normal_orientation": "canonical_unsigned_axis_double_sided",
        "child_partition": GEOMETRY_COMPLETE_PARTITION,
        "parent_partition": "world_metric_hard_voxel_context_partition_v1",
        "child_voxel_size_m": child_size,
        "parent_voxel_size_m": parent_size,
        "primitive_membership_coverage_fraction": 1.0,
        "source_geometry_carrier_physical_map_sha256": geometry_carrier.content_sha256,
        **dict(metadata or {}),
    }
    result = GoalMapletPhysicalMap(
        maplet_ids=np.arange(parent_count, dtype=np.int64),
        maplet_centers=np.asarray(parent_centers, dtype=np.float64),
        maplet_normals=np.asarray(parent_normals, dtype=np.float64),
        maplet_frames=np.asarray(parent_frames, dtype=np.float64),
        maplet_extents=np.asarray(parent_extents, dtype=np.float64),
        maplet_sidedness=np.full(parent_count, DOUBLE_SIDED, dtype=np.uint8),
        maplet_orientation_confidence=np.zeros(parent_count, dtype=np.float32),
        primitive_ids=np.asarray(geometry_carrier.primitive_ids, dtype=np.int64),
        primitive_centers=centers,
        primitive_tangent1=np.asarray(
            geometry_carrier.primitive_tangent1, dtype=np.float64
        ),
        primitive_tangent2=primitive_tangent2,
        primitive_normals=primitive_normals,
        primitive_scale1=np.asarray(geometry_carrier.primitive_scale1, dtype=np.float64),
        primitive_scale2=np.asarray(geometry_carrier.primitive_scale2, dtype=np.float64),
        primitive_opacity=np.asarray(geometry_carrier.primitive_opacity, dtype=np.float64),
        primitive_sidedness=np.full(primitive_count, DOUBLE_SIDED, dtype=np.uint8),
        primitive_orientation_confidence=np.zeros(primitive_count, dtype=np.float32),
        membership_offsets=membership_offsets,
        membership_primitive_rows=np.concatenate(parent_primitive_rows),
        membership_weights=np.concatenate(parent_membership_weights),
        membership_boundary_weights=np.concatenate(parent_boundary_weights),
        maplet_child_offsets=maplet_child_offsets,
        child_parent_rows=child_parent.astype(np.int64),
        child_centers=np.asarray(child_centers, dtype=np.float64),
        child_normals=np.asarray(child_normals, dtype=np.float64),
        child_frames=np.asarray(child_frames, dtype=np.float64),
        child_extents=np.asarray(child_extents, dtype=np.float64),
        child_member_offsets=child_offsets_from_primitives,
        child_member_primitive_rows=primitive_order.astype(np.int64),
        child_member_weights=np.concatenate(child_member_weights),
        child_member_local_uv=np.concatenate(child_local_uv, axis=0),
        metadata=output_metadata,
    )
    audit = GeometryCompleteBuildAudit(
        parent_voxel_keys=parent_keys.astype(np.int64),
        child_voxel_normal_keys=child_keys.astype(np.int64),
        unique_child_count=child_count,
    )
    return result, audit
