"""Query-independent physical-map support cells crossed with analytic SO(3).

This module deliberately builds a *support upper bound*, not a pose ranking.
The position domain is the single world-axis AABB of the union of every exact
physical primitive rectangle, expanded by a frozen constant and discretised
with the same 2 m world lattice as the parent-local proposal.  Positions are
shared by every query and the Cartesian product with analytic60 is implicit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from .lineage import arrays_sha256
from .parent_geometry_pose_proposal import (
    LATTICE_SPACING_M,
    MAXIMUM_UNIQUE_CELLS_PER_QUERY,
    ORIENTATION_COUNT,
    PARENT_AABB_EXPANSION_LINF_M,
    exact_parent_primitive_rectangle_aabbs,
    lattice_origin_from_parent_geometry,
    analytic_orientation_codebook,
    orientation_cover_certificate,
)
from .physical_map import GoalMapletPhysicalMap


SCHEMA = "goal_maplet_query_independent_global_physical_pose_support_audit_v1"
SEMANTICS = (
    "single_full_map_exact_primitive_rectangle_union_world_aabb_linf12_"
    "world_lattice2_x_regular600cell_so3_60_implicit_v1"
)
ALL_PARENT_UNION_SCHEMA = (
    "goal_maplet_query_independent_all_parent_geometry_analytic60_proposal_v1"
)
ALL_PARENT_UNION_SEMANTICS = (
    "all_physical_parent_exact_rectangle_world_aabb_each_linf12_union_dedup_"
    "world_lattice2_x_regular600cell_so3_60_implicit_v1"
)
ALL_PARENT_UNION_ARRAY_NAMES = (
    "maplet_ids",
    "parent_aabb_min_world",
    "parent_aabb_max_world",
    "lattice_origin_world",
    "lattice_spacing_m",
    "cell_indices_world",
    "cell_first_parent_row",
    "cell_parent_support_count",
    "implicit_pose_factor_count",
    "orientation_rotations_w2c",
)


def global_physical_rectangle_aabb(
    physical: GoalMapletPhysicalMap,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the one exact world AABB enclosing all physical rectangles."""

    parent_lower, parent_upper = exact_parent_primitive_rectangle_aabbs(physical)
    lower = np.min(parent_lower, axis=0)
    upper = np.max(parent_upper, axis=0)
    if (
        lower.shape != (3,) or upper.shape != (3,)
        or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
        or np.any(upper < lower)
    ):
        raise ValueError("global physical rectangle AABB differs")
    return lower, upper


def build_global_physical_support_audit(
    physical: GoalMapletPhysicalMap,
    *,
    maximum_position_count: int = MAXIMUM_UNIQUE_CELLS_PER_QUERY,
) -> dict[str, Any]:
    """Count the global factored domain without materialising cells or poses.

    The cap is evaluated solely from physical-map geometry.  In particular,
    this function has no query, retrieval, contributor, pose, or label input.
    """

    cap = int(maximum_position_count)
    if cap <= 0:
        raise ValueError("global position cap must be positive")
    parent_lower, _ = exact_parent_primitive_rectangle_aabbs(physical)
    lower, upper = global_physical_rectangle_aabb(physical)
    origin = lattice_origin_from_parent_geometry(parent_lower)
    expanded_lower = lower - PARENT_AABB_EXPANSION_LINF_M
    expanded_upper = upper + PARENT_AABB_EXPANSION_LINF_M
    first = np.floor(
        (expanded_lower - origin) / LATTICE_SPACING_M
    ).astype(np.int64)
    last = np.floor(
        (expanded_upper - origin) / LATTICE_SPACING_M
    ).astype(np.int64)
    shape = last - first + 1
    if np.any(shape <= 0):
        raise ValueError("global expanded lattice shape differs")
    position_count = int(np.prod(shape, dtype=np.int64))
    implicit_factor_count = position_count * ORIENTATION_COUNT
    cell_index_bytes = position_count * 3 * np.dtype(np.int32).itemsize
    cell_center_bytes = position_count * 3 * np.dtype(np.float64).itemsize
    orientation_bytes = ORIENTATION_COUNT * 3 * 3 * np.dtype(np.float64).itemsize
    within_cap = position_count <= cap
    return {
        "artifact_type": SCHEMA,
        "semantics": SEMANTICS,
        "phase": "phase1_score_before_label_geometry_only_structural_gate",
        "uses_query_image": False,
        "uses_query_retrieval": False,
        "uses_query_pose": False,
        "uses_query_ground_truth": False,
        "uses_mapping_camera_position_seed": False,
        "query_independent_positions_shared": True,
        "parent_retrieval_cuts_domain": False,
        "parent_retrieval_future_role": "priority_only_not_support_pruning",
        "single_global_bounding_box": True,
        "global_aabb_is_union_bounding_box_not_sparse_parent_box_union": True,
        "primitive_rectangle_axis_extent": (
            "abs(tangent1_axis)*scale1+abs(tangent2_axis)*scale2"
        ),
        "support_sigma_multiplier": 1.0,
        "global_rectangle_aabb_min_world": lower.tolist(),
        "global_rectangle_aabb_max_world": upper.tolist(),
        "global_rectangle_aabb_extent_m": (upper - lower).tolist(),
        "expansion_linf_m": PARENT_AABB_EXPANSION_LINF_M,
        "expanded_aabb_min_world": expanded_lower.tolist(),
        "expanded_aabb_max_world": expanded_upper.tolist(),
        "lattice_origin_world": origin.tolist(),
        "lattice_spacing_m": LATTICE_SPACING_M,
        "lattice_first_index": first.tolist(),
        "lattice_last_index_inclusive": last.tolist(),
        "lattice_shape": shape.tolist(),
        "cell_cover_radius_m": float(np.sqrt(3.0) * LATTICE_SPACING_M / 2.0),
        "cell_cover_radius_strictly_below_2m": bool(
            np.sqrt(3.0) * LATTICE_SPACING_M / 2.0 < 2.0
        ),
        "position_count": position_count,
        "position_hard_cap": cap,
        "position_count_within_cap": within_cap,
        "orientation_count": ORIENTATION_COUNT,
        "orientation_cover_certificate": orientation_cover_certificate(),
        "implicit_pose_factor_count": implicit_factor_count,
        "cartesian_product_materialized": False,
        "ram_bytes": {
            "cell_indices_int32_if_materialized": cell_index_bytes,
            "cell_centers_float64_if_materialized": cell_center_bytes,
            "orientation_rotations_float64": orientation_bytes,
            "minimal_factored_indices_plus_orientation": (
                cell_index_bytes + orientation_bytes
            ),
            "factored_float64_centers_plus_orientation": (
                cell_center_bytes + orientation_bytes
            ),
        },
        "structural_gate": {
            "decision": "GO_TO_SEQ10_RAW_SUPPORT" if within_cap else "KILL",
            "reason": (
                "position_count_within_hard_cap"
                if within_cap else "position_hard_cap_exceeded_before_label_read"
            ),
            "seq10_pose_labels_read": False,
        },
        "support_claim": "nonphysical_global_raw_support_upper_bound_only",
        "free_space_or_clearance_certificate": False,
        "collision_certificate": False,
        "ranking_performed": False,
    }


def _expanded_aabb_cell_range(
    lower: np.ndarray, upper: np.ndarray, origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.floor(
        (
            np.asarray(lower, dtype=np.float64)
            - PARENT_AABB_EXPANSION_LINF_M
            - np.asarray(origin, dtype=np.float64)
        ) / LATTICE_SPACING_M
    ).astype(np.int64)
    last = np.floor(
        (
            np.asarray(upper, dtype=np.float64)
            + PARENT_AABB_EXPANSION_LINF_M
            - np.asarray(origin, dtype=np.float64)
        ) / LATTICE_SPACING_M
    ).astype(np.int64)
    if np.any(last < first):
        raise ValueError("expanded physical AABB cell range differs")
    return first, last


def build_all_parent_union_support_arrays(
    physical: GoalMapletPhysicalMap,
    *,
    maximum_position_count: int = MAXIMUM_UNIQUE_CELLS_PER_QUERY,
) -> dict[str, np.ndarray]:
    """Materialise the deduplicated union of every expanded parent box.

    This remains query independent: all parents are included and retrieval
    cannot add or remove a cell.  The hard cap is checked after exact
    deduplication and the function never truncates to satisfy it.
    """

    cap = int(maximum_position_count)
    if cap <= 0:
        raise ValueError("all-parent union position cap must be positive")
    parent_lower, parent_upper = exact_parent_primitive_rectangle_aabbs(physical)
    origin = lattice_origin_from_parent_geometry(parent_lower)
    # cell -> [first supporting parent row, number of supporting parent boxes]
    inventory: dict[tuple[int, int, int], list[int]] = {}
    raw_parent_cell_count = 0
    for parent_row, (lower, upper) in enumerate(zip(parent_lower, parent_upper)):
        first, last = _expanded_aabb_cell_range(lower, upper, origin)
        raw_parent_cell_count += int(np.prod(last - first + 1, dtype=np.int64))
        for x in range(int(first[0]), int(last[0]) + 1):
            for y in range(int(first[1]), int(last[1]) + 1):
                for z in range(int(first[2]), int(last[2]) + 1):
                    key = (x, y, z)
                    current = inventory.get(key)
                    if current is None:
                        inventory[key] = [parent_row, 1]
                    else:
                        current[1] += 1
    if not inventory:
        raise ValueError("all-parent union position inventory is empty")
    position_count = len(inventory)
    if position_count > cap:
        raise ValueError(
            "all-parent union position hard cap exceeded without truncation: "
            f"{position_count}>{cap}"
        )
    ordered = sorted(inventory)
    first_parent = np.asarray(
        [inventory[key][0] for key in ordered], dtype=np.int32,
    )
    support_count_i64 = np.asarray(
        [inventory[key][1] for key in ordered], dtype=np.int64,
    )
    if np.max(support_count_i64) > np.iinfo(np.uint16).max:
        raise ValueError("all-parent cell support count exceeds uint16")
    return {
        "maplet_ids": np.asarray(physical.maplet_ids, dtype=np.int64).copy(),
        "parent_aabb_min_world": parent_lower,
        "parent_aabb_max_world": parent_upper,
        "lattice_origin_world": origin,
        "lattice_spacing_m": np.asarray(LATTICE_SPACING_M, dtype=np.float64),
        "cell_indices_world": np.asarray(ordered, dtype=np.int32),
        "cell_first_parent_row": first_parent,
        "cell_parent_support_count": support_count_i64.astype(np.uint16),
        "implicit_pose_factor_count": np.asarray(
            position_count * ORIENTATION_COUNT, dtype=np.int64,
        ),
        "orientation_rotations_w2c": analytic_orientation_codebook(),
        # The raw pre-dedup sum is intentionally metadata, not a deployment
        # array; the builder records it from this deterministic reconstruction.
        "_raw_parent_cell_count": np.asarray(raw_parent_cell_count, dtype=np.int64),
    }


def all_parent_union_public_arrays(
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Remove deterministic build-only statistics before serialization."""

    if set(arrays) != {*ALL_PARENT_UNION_ARRAY_NAMES, "_raw_parent_cell_count"}:
        raise ValueError("all-parent union internal arrays differ")
    return {name: np.asarray(arrays[name]) for name in ALL_PARENT_UNION_ARRAY_NAMES}


def load_all_parent_union_support(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    artifact = Path(path)
    expected = {
        *(f"{name}.npy" for name in ALL_PARENT_UNION_ARRAY_NAMES),
        "metadata_json.npy",
    }
    try:
        with zipfile.ZipFile(artifact, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("all-parent union proposal is not a valid NPZ") from error
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("all-parent union proposal NPZ members differ")
    with np.load(artifact, allow_pickle=False) as data:
        if set(data.files) != {*ALL_PARENT_UNION_ARRAY_NAMES, "metadata_json"}:
            raise ValueError("all-parent union proposal arrays differ")
        arrays = {
            name: np.asarray(data[name]) for name in ALL_PARENT_UNION_ARRAY_NAMES
        }
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    cells = arrays["cell_indices_world"]
    position_count = int(cells.shape[0])
    parent_count = int(arrays["maplet_ids"].size)
    if (
        metadata.get("artifact_type") != ALL_PARENT_UNION_SCHEMA
        or metadata.get("semantics") != ALL_PARENT_UNION_SEMANTICS
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("uses_query_retrieval") is not False
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or metadata.get("uses_mapping_camera_position_seed") is not False
        or metadata.get("query_independent_positions_shared") is not True
        or metadata.get("parent_retrieval_cuts_domain") is not False
        or metadata.get("cartesian_product_materialized") is not False
        or metadata.get("orientation_cover_certificate")
        != orientation_cover_certificate()
        or arrays["maplet_ids"].ndim != 1
        or parent_count <= 0
        or np.unique(arrays["maplet_ids"]).size != parent_count
        or arrays["parent_aabb_min_world"].shape != (parent_count, 3)
        or arrays["parent_aabb_max_world"].shape != (parent_count, 3)
        or np.any(~np.isfinite(arrays["parent_aabb_min_world"]))
        or np.any(~np.isfinite(arrays["parent_aabb_max_world"]))
        or np.any(
            arrays["parent_aabb_max_world"] < arrays["parent_aabb_min_world"]
        )
        or arrays["lattice_origin_world"].shape != (3,)
        or float(arrays["lattice_spacing_m"]) != LATTICE_SPACING_M
        or cells.ndim != 2 or cells.shape[1:] != (3,)
        or cells.dtype != np.int32 or position_count <= 0
        or position_count > int(metadata.get("position_hard_cap", -1))
        or np.unique(cells, axis=0).shape[0] != position_count
        or not np.array_equal(
            cells,
            np.asarray(sorted(map(tuple, cells.tolist())), dtype=np.int32),
        )
        or arrays["cell_first_parent_row"].shape != (position_count,)
        or np.any(arrays["cell_first_parent_row"] < 0)
        or np.any(arrays["cell_first_parent_row"] >= parent_count)
        or arrays["cell_parent_support_count"].shape != (position_count,)
        or arrays["cell_parent_support_count"].dtype != np.uint16
        or np.any(arrays["cell_parent_support_count"] <= 0)
        or arrays["implicit_pose_factor_count"].shape != ()
        or int(arrays["implicit_pose_factor_count"])
        != position_count * ORIENTATION_COUNT
        or arrays["orientation_rotations_w2c"].shape != (60, 3, 3)
        or not np.array_equal(
            arrays["orientation_rotations_w2c"], analytic_orientation_codebook(),
        )
        or int(metadata.get("position_count", -1)) != position_count
        or int(metadata.get("orientation_count", -1)) != ORIENTATION_COUNT
        or int(metadata.get("implicit_pose_factor_count", -1))
        != position_count * ORIENTATION_COUNT
    ):
        raise ValueError("all-parent union proposal contract differs")
    return arrays, metadata
