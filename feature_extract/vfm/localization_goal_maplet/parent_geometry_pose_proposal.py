"""Query-parent-conditioned position cells and an analytic full-SO(3) cover.

Positions come only from physical parent geometry and the pose-free ranked
scene-parent prefix.  Mapping/query camera centres are not an input.  The
represented support is a union of 2 m world-axis lattice cells crossed with
the fixed 60-rotation regular-600-cell codebook; the Cartesian product is
never materialized.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .factorized_pose_free_proposal import _validate_proper_rotations
from .lineage import arrays_sha256
from .physical_map import GoalMapletPhysicalMap


SCHEMA = "goal_maplet_pose_free_parent_geometry_analytic60_proposal_v1"
SEMANTICS = (
    "top_parent_prefix_exact_primitive_rectangle_world_aabb_linf12_"
    "world_lattice2_x_regular600cell_so3_60_v1"
)
PARENT_PREFIX_BUDGETS = (4, 8, 16)
MAXIMUM_PARENT_PREFIX = 16
LATTICE_SPACING_M = 2.0
PARENT_AABB_EXPANSION_LINF_M = 12.0
MAXIMUM_UNIQUE_CELLS_PER_QUERY = 131072
ORIENTATION_COUNT = 60
ORIENTATION_CONSTRUCTION = "binary_icosahedral_600_cell_antipodal_so3"
ORIENTATION_COVER_RADIUS_DEG = 44.47751218592992
ARRAY_NAMES = (
    "image_ids", "parent_prefix_budgets", "selected_parent_ids",
    "selected_parent_scores", "selected_parent_aabb_min_world",
    "selected_parent_aabb_max_world", "lattice_origin_world",
    "lattice_spacing_m", "cell_offsets", "cell_indices_world",
    "cell_first_parent_rank", "unique_position_count_by_parent_prefix",
    "implicit_pose_factor_count_by_parent_prefix",
    "orientation_rotations_w2c",
)


def _even_permutations() -> tuple[tuple[int, ...], ...]:
    values = []
    for permutation in itertools.permutations(range(4)):
        inversions = sum(
            permutation[first] > permutation[second]
            for first in range(4)
            for second in range(first + 1, 4)
        )
        if inversions % 2 == 0:
            values.append(tuple(int(value) for value in permutation))
    return tuple(values)


def binary_icosahedral_quaternions() -> np.ndarray:
    """Return one unit-wxyz quaternion per antipodal regular-600-cell pair."""

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices: set[tuple[float, float, float, float]] = set()
    for axis in range(4):
        for sign in (-1.0, 1.0):
            value = [0.0, 0.0, 0.0, 0.0]
            value[axis] = sign
            vertices.add(tuple(value))
    for signs in itertools.product((-1.0, 1.0), repeat=4):
        vertices.add(tuple(0.5 * sign for sign in signs))
    base = (0.0, 0.5, phi / 2.0, 1.0 / (2.0 * phi))
    for permutation in _even_permutations():
        permuted = tuple(base[index] for index in permutation)
        nonzero = tuple(index for index, value in enumerate(permuted) if value)
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            value = list(permuted)
            for index, sign in zip(nonzero, signs):
                value[index] *= sign
            vertices.add(tuple(round(item, 15) for item in value))
    full = np.asarray(sorted(vertices), dtype=np.float64)
    if full.shape != (120, 4):
        raise RuntimeError("regular 600-cell construction differs")
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    representatives = []
    for quaternion in full:
        first = int(np.flatnonzero(np.abs(quaternion) > 1.0e-12)[0])
        if quaternion[first] > 0.0:
            representatives.append(quaternion)
    result = np.asarray(representatives, dtype=np.float64)
    if result.shape != (ORIENTATION_COUNT, 4):
        raise RuntimeError("600-cell antipodal quotient differs")
    return result


def analytic_orientation_codebook() -> np.ndarray:
    quaternion = binary_icosahedral_quaternions()
    # Retain byte-for-byte parity with the hierarchical_pose_domain_v4
    # authority, whose quaternion-to-matrix boundary normalizes once more.
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    result = np.empty((ORIENTATION_COUNT, 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[:, 0, 1] = 2.0 * (x * y - z * w)
    result[:, 0, 2] = 2.0 * (x * z + y * w)
    result[:, 1, 0] = 2.0 * (x * y + z * w)
    result[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[:, 1, 2] = 2.0 * (y * z - x * w)
    result[:, 2, 0] = 2.0 * (x * z - y * w)
    result[:, 2, 1] = 2.0 * (y * z + x * w)
    result[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return _validate_proper_rotations(
        result, context="analytic regular-600-cell SO(3) codebook",
        tolerance=1.0e-12,
    )


def orientation_cover_certificate() -> dict[str, object]:
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    pair_dot = phi / 2.0
    centre_dot = math.sqrt((1.0 + 3.0 * pair_dot) / 4.0)
    s3_radius = math.acos(centre_dot)
    return {
        "construction": ORIENTATION_CONSTRUCTION,
        "orientation_count": ORIENTATION_COUNT,
        "s3_covering_radius_deg": math.degrees(s3_radius),
        "so3_covering_radius_deg": math.degrees(2.0 * s3_radius),
        "proof": (
            "regular_600_cell_tetrahedral_facets_tile_s3_and_each_facet_"
            "circumcentre_has_dot_sqrt((1+3*phi/2)/4)_with_its_vertices"
        ),
        "strictly_below_45_degrees": True,
        "monte_carlo_used_as_authority": False,
    }


def exact_parent_primitive_rectangle_aabbs(
    physical: GoalMapletPhysicalMap,
) -> tuple[np.ndarray, np.ndarray]:
    """World AABBs of complete parent membership using oriented rectangles."""

    axis_extent = (
        np.abs(np.asarray(physical.primitive_tangent1, dtype=np.float64))
        * np.asarray(physical.primitive_scale1, dtype=np.float64)[:, None]
        + np.abs(np.asarray(physical.primitive_tangent2, dtype=np.float64))
        * np.asarray(physical.primitive_scale2, dtype=np.float64)[:, None]
    )
    center = np.asarray(physical.primitive_centers, dtype=np.float64)
    lower, upper = [], []
    for parent in range(int(physical.maplet_ids.size)):
        rows = physical.membership_primitive_rows[physical.member_slice(parent)]
        if rows.size <= 0:
            raise ValueError("physical parent has no complete primitive membership")
        lower.append(np.min(center[rows] - axis_extent[rows], axis=0))
        upper.append(np.max(center[rows] + axis_extent[rows], axis=0))
    return np.stack(lower), np.stack(upper)


def lattice_origin_from_parent_geometry(parent_aabb_min: np.ndarray) -> np.ndarray:
    return (
        np.floor(np.min(np.asarray(parent_aabb_min), axis=0) / LATTICE_SPACING_M)
        * LATTICE_SPACING_M
    )


def _parent_cells(
    lower: np.ndarray, upper: np.ndarray, origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.floor(
        (np.asarray(lower) - PARENT_AABB_EXPANSION_LINF_M - origin)
        / LATTICE_SPACING_M
    ).astype(np.int64)
    last = np.floor(
        (np.asarray(upper) + PARENT_AABB_EXPANSION_LINF_M - origin)
        / LATTICE_SPACING_M
    ).astype(np.int64)
    if np.any(last < first):
        raise ValueError("expanded parent AABB lattice range differs")
    return first, last


def build_parent_geometry_proposal_arrays(
    image_ids: np.ndarray,
    scene_parent_ids: np.ndarray,
    scene_parent_scores: np.ndarray,
    physical: GoalMapletPhysicalMap,
) -> dict[str, np.ndarray]:
    image = np.asarray(image_ids)
    parent_ids = np.asarray(scene_parent_ids, dtype=np.int64)
    parent_scores = np.asarray(scene_parent_scores, dtype=np.float64)
    query_count = int(image.size)
    if (
        query_count <= 0 or image.ndim != 1 or image.dtype.kind != "U"
        or np.unique(image).size != query_count
        or parent_ids.shape != (query_count, MAXIMUM_PARENT_PREFIX)
        or parent_scores.shape != parent_ids.shape
        or np.any(~np.isfinite(parent_scores))
        or np.any(np.diff(parent_scores, axis=1) > 1.0e-12)
        or any(np.unique(value).size != MAXIMUM_PARENT_PREFIX for value in parent_ids)
    ):
        raise ValueError("parent-geometry query parent prefix differs")
    map_row = {int(value): row for row, value in enumerate(physical.maplet_ids.tolist())}
    if any(int(value) not in map_row for value in parent_ids.reshape(-1).tolist()):
        raise ValueError("retrieved parent ID is absent from the physical map")
    parent_lower, parent_upper = exact_parent_primitive_rectangle_aabbs(physical)
    origin = lattice_origin_from_parent_geometry(parent_lower)
    selected_rows = np.asarray(
        [[map_row[int(value)] for value in row] for row in parent_ids],
        dtype=np.int64,
    )
    selected_lower = parent_lower[selected_rows]
    selected_upper = parent_upper[selected_rows]
    offsets = [0]
    cell_rows = []
    first_rank_rows = []
    counts = np.zeros((query_count, len(PARENT_PREFIX_BUDGETS)), dtype=np.int32)
    for query in range(query_count):
        cell_first_rank: dict[tuple[int, int, int], int] = {}
        for rank in range(1, MAXIMUM_PARENT_PREFIX + 1):
            first, last = _parent_cells(
                selected_lower[query, rank - 1],
                selected_upper[query, rank - 1], origin,
            )
            for x in range(int(first[0]), int(last[0]) + 1):
                for y in range(int(first[1]), int(last[1]) + 1):
                    for z in range(int(first[2]), int(last[2]) + 1):
                        cell_first_rank.setdefault((x, y, z), rank)
            if rank in PARENT_PREFIX_BUDGETS:
                counts[query, PARENT_PREFIX_BUDGETS.index(rank)] = len(cell_first_rank)
        if len(cell_first_rank) > MAXIMUM_UNIQUE_CELLS_PER_QUERY:
            raise ValueError("parent-geometry position hard cap exceeded without truncation")
        ordered = sorted(cell_first_rank)
        cell_rows.append(np.asarray(ordered, dtype=np.int32))
        first_rank_rows.append(np.asarray(
            [cell_first_rank[value] for value in ordered], dtype=np.uint8,
        ))
        offsets.append(offsets[-1] + len(ordered))
    return {
        "image_ids": image.copy(),
        "parent_prefix_budgets": np.asarray(PARENT_PREFIX_BUDGETS, dtype=np.int16),
        "selected_parent_ids": parent_ids.copy(),
        "selected_parent_scores": parent_scores.copy(),
        "selected_parent_aabb_min_world": selected_lower,
        "selected_parent_aabb_max_world": selected_upper,
        "lattice_origin_world": origin,
        "lattice_spacing_m": np.asarray(LATTICE_SPACING_M, dtype=np.float64),
        "cell_offsets": np.asarray(offsets, dtype=np.int64),
        "cell_indices_world": np.concatenate(cell_rows, axis=0),
        "cell_first_parent_rank": np.concatenate(first_rank_rows, axis=0),
        "unique_position_count_by_parent_prefix": counts,
        "implicit_pose_factor_count_by_parent_prefix": (
            counts.astype(np.int64) * ORIENTATION_COUNT
        ),
        "orientation_rotations_w2c": analytic_orientation_codebook(),
    }


def load_parent_geometry_proposal(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    artifact = Path(path)
    expected = {*(f"{name}.npy" for name in ARRAY_NAMES), "metadata_json.npy"}
    try:
        with zipfile.ZipFile(artifact, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("parent-geometry proposal is not a valid NPZ") from error
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError("parent-geometry proposal NPZ members differ")
    with np.load(artifact, allow_pickle=False) as data:
        if set(data.files) != {*ARRAY_NAMES, "metadata_json"}:
            raise ValueError("parent-geometry proposal arrays differ")
        arrays = {name: np.asarray(data[name]) for name in ARRAY_NAMES}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    certificate = orientation_cover_certificate()
    if (
        metadata.get("artifact_type") != SCHEMA
        or metadata.get("semantics") != SEMANTICS
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or metadata.get("uses_mapping_camera_position_seed") is not False
        or metadata.get("cross_position_orientation_product_materialized") is not False
        or metadata.get("orientation_cover_certificate") != certificate
    ):
        raise ValueError("parent-geometry proposal metadata differs")
    image_ids = arrays["image_ids"]
    query_count = int(image_ids.size)
    offsets = arrays["cell_offsets"]
    cells = arrays["cell_indices_world"]
    first_rank = arrays["cell_first_parent_rank"]
    counts = arrays["unique_position_count_by_parent_prefix"]
    if (
        query_count <= 0 or image_ids.ndim != 1 or image_ids.dtype.kind != "U"
        or np.unique(image_ids).size != query_count
        or not np.array_equal(
            arrays["parent_prefix_budgets"],
            np.asarray(PARENT_PREFIX_BUDGETS, dtype=arrays["parent_prefix_budgets"].dtype),
        )
        or arrays["selected_parent_ids"].shape != (query_count, 16)
        or arrays["selected_parent_scores"].shape != (query_count, 16)
        or arrays["selected_parent_aabb_min_world"].shape != (query_count, 16, 3)
        or arrays["selected_parent_aabb_max_world"].shape != (query_count, 16, 3)
        or arrays["lattice_origin_world"].shape != (3,)
        or float(arrays["lattice_spacing_m"]) != LATTICE_SPACING_M
        or offsets.shape != (query_count + 1,) or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0) or offsets[-1] != cells.shape[0]
        or cells.ndim != 2 or cells.shape[1:] != (3,) or cells.dtype.kind not in "iu"
        or first_rank.shape != (cells.shape[0],) or first_rank.dtype != np.uint8
        or np.any(first_rank < 1) or np.any(first_rank > 16)
        or counts.shape != (query_count, 3)
        or arrays["implicit_pose_factor_count_by_parent_prefix"].shape != counts.shape
        or arrays["orientation_rotations_w2c"].shape != (60, 3, 3)
        or not np.array_equal(
            arrays["orientation_rotations_w2c"], analytic_orientation_codebook(),
        )
    ):
        raise ValueError("parent-geometry proposal shapes differ")
    for query in range(query_count):
        start, end = int(offsets[query]), int(offsets[query + 1])
        local_cells = cells[start:end]
        local_rank = first_rank[start:end]
        if (
            end - start > MAXIMUM_UNIQUE_CELLS_PER_QUERY
            or np.unique(local_cells, axis=0).shape[0] != end - start
            or not np.array_equal(
                local_cells,
                np.asarray(sorted(map(tuple, local_cells.tolist())), dtype=local_cells.dtype),
            )
        ):
            raise ValueError("parent-geometry proposal cell inventory differs")
        expected_counts = np.asarray([
            int(np.sum(local_rank <= budget)) for budget in PARENT_PREFIX_BUDGETS
        ], dtype=counts.dtype)
        if (
            not np.array_equal(counts[query], expected_counts)
            or not np.array_equal(
                arrays["implicit_pose_factor_count_by_parent_prefix"][query],
                expected_counts.astype(np.int64) * ORIENTATION_COUNT,
            )
        ):
            raise ValueError("parent-geometry proposal prefix counts differ")
    if (
        int(metadata.get("query_count", -1)) != query_count
        or metadata.get("parent_prefix_budgets") != list(PARENT_PREFIX_BUDGETS)
        or int(metadata.get("orientation_count", -1)) != ORIENTATION_COUNT
        or float(metadata.get("lattice_spacing_m", -1.0)) != LATTICE_SPACING_M
        or float(metadata.get("parent_aabb_expansion_linf_m", -1.0))
        != PARENT_AABB_EXPANSION_LINF_M
        or int(metadata.get("maximum_unique_cells_per_query", -1))
        != MAXIMUM_UNIQUE_CELLS_PER_QUERY
    ):
        raise ValueError("parent-geometry proposal metadata counts differ")
    return arrays, metadata
