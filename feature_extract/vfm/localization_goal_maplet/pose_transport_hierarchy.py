"""Deterministic local child graph for sparse pose transport.

Children remain the frozen geometry-complete physical partition.  Adjacency
is not learned from query images: two children of the same parent are joined
only when their conservative bounding spheres touch after a fixed metric gap
and their unsigned surface normals are compatible.  A second, coarser support
ID is the deterministic six-connected metric-voxel component with compatible
unsigned normals.  It may cross parent seams but never changes the exact child
union or reads image/query evidence.
"""

from __future__ import annotations

import numpy as np

from .candidate_conditioned_pose_attribution import (
    PoseTransportHierarchy,
    pose_transport_hierarchy_content_sha256,
)
from .connected_fine_support import connected_fine_support_components
from .physical_map import GoalMapletPhysicalMap


HIERARCHY_SEMANTICS = (
    "geometry_child_touching_adjacency_plus_six_connected_unsigned_normal_support_v2"
)


def build_pose_transport_hierarchy(
    physical: GoalMapletPhysicalMap,
    *,
    adjacency_gap_m: float = 0.25,
    minimum_unsigned_normal_cosine: float = 0.5,
) -> PoseTransportHierarchy:
    gap = float(adjacency_gap_m)
    cosine = float(minimum_unsigned_normal_cosine)
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError("adjacency_gap_m must be finite and nonnegative")
    if not np.isfinite(cosine) or not 0.0 <= cosine <= 1.0:
        raise ValueError("minimum_unsigned_normal_cosine must lie in [0,1]")
    parent = np.asarray(physical.child_parent_rows, dtype=np.int64).reshape(-1)
    centers = np.asarray(physical.child_centers, dtype=np.float64)
    normals = np.asarray(physical.child_normals, dtype=np.float64)
    extents = np.asarray(physical.child_extents, dtype=np.float64)
    if (
        centers.shape != (parent.size, 3) or normals.shape != centers.shape
        or extents.shape != centers.shape or np.any(~np.isfinite(centers))
        or np.any(~np.isfinite(normals)) or np.any(~np.isfinite(extents))
        or np.any(extents < 0.0)
    ):
        raise ValueError("invalid physical child geometry")
    radii = 0.5 * np.linalg.norm(extents, axis=1)
    neighbours: list[set[int]] = [set() for _ in range(parent.size)]
    for parent_row in range(int(physical.maplet_ids.size)):
        rows = np.flatnonzero(parent == parent_row)
        if rows.size < 2:
            continue
        delta = centers[rows, None, :] - centers[None, rows, :]
        distance = np.linalg.norm(delta, axis=2)
        touching = distance <= radii[rows, None] + radii[None, rows] + gap
        aligned = np.abs(normals[rows] @ normals[rows].T) >= cosine
        pair = np.triu(touching & aligned, k=1)
        left, right = np.nonzero(pair)
        for a, b in zip(rows[left].tolist(), rows[right].tolist()):
            neighbours[a].add(b)
            neighbours[b].add(a)
    offsets = np.zeros((parent.size + 1,), dtype=np.int64)
    rows_out: list[int] = []
    for child, values in enumerate(neighbours):
        rows_out.extend(sorted(values))
        offsets[child + 1] = len(rows_out)
    adjacency = np.asarray(rows_out, dtype=np.int64)
    connected = connected_fine_support_components(
        np.arange(parent.size, dtype=np.int64), physical,
        maximum_normal_angle_degrees=float(
            np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
        ),
    )
    support = np.full(parent.shape, -1, dtype=np.int64)
    for component in range(connected.component_count):
        begin = int(connected.component_offsets[component])
        end = int(connected.component_offsets[component + 1])
        support[connected.component_child_rows[begin:end]] = component
    if np.any(support < 0):
        raise AssertionError("connected support does not cover every physical child")
    return PoseTransportHierarchy(
        child_parent_ids=parent,
        child_support_ids=support,
        adjacency_offsets=offsets,
        adjacency_child_rows=adjacency,
        content_sha256=pose_transport_hierarchy_content_sha256(
            parent, support, offsets, adjacency
        ),
    )
