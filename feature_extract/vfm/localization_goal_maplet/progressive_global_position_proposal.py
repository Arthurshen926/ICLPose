"""Progressive query-conditioned prefixes inside frozen global-v2 support.

The global position support is immutable.  Each physical parent's expanded
box receives a deterministic coarse-to-fine cell order, then pose-free scene
parent evidence selects a prefix of parents and ranked round-robin allocates a
fixed unique-position budget.  No camera pose or label is an input.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from .global_physical_pose_support import (
    ORIENTATION_COUNT,
    _expanded_aabb_cell_range,
)
from .lineage import arrays_sha256
from .parent_geometry_pose_proposal import analytic_orientation_codebook


PARENT_ORDER_SCHEMA = "goal_maplet_global_v2_parent_progressive_cell_order_v1"
PARENT_ORDER_SEMANTICS = (
    "exact_expanded_parent_box_one_per_dyadic_bin_checkpoint_"
    "bit_reversed_morton_nested_v1"
)
QUERY_PROPOSAL_SCHEMA = (
    "goal_maplet_query_conditioned_progressive_global_position_proposal_v1"
)
QUERY_PROPOSAL_SEMANTICS = (
    "layout_scene_parent_ranked_round_robin_over_frozen_parent_nested_orders_v1"
)
PARENT_CEILING_SCHEMA = "goal_maplet_parent_cropped_full_union_ceiling_candidates_v1"
PARENT_CEILING_SEMANTICS = (
    "layout_scene_parent_prefix_complete_expanded_box_cell_union_deduplicated_v1"
)
GLOBAL_ORDER_SCHEMA = "goal_maplet_global_v2_sparse_dyadic_cell_order_v1"
GLOBAL_ORDER_SEMANTICS = (
    "global_v2_sparse_cells_one_inherited_or_new_representative_per_nonempty_"
    "dyadic_bin_new_bins_nearest_center_bit_reversed_morton_nested_v1"
)
PARENT_PREFIX_BUDGETS = (4, 8, 16, 32)
POSITION_BUDGETS = (256, 512, 1024, 2048, 4096)
MAXIMUM_PARENT_PREFIX = max(PARENT_PREFIX_BUDGETS)
MAXIMUM_POSITION_BUDGET = max(POSITION_BUDGETS)
PARENT_ORDER_ARRAY_NAMES = (
    "maplet_ids", "parent_cell_offsets", "parent_order_cell_rows",
)
QUERY_PROPOSAL_ARRAY_NAMES = (
    "image_ids", "parent_prefix_budgets", "total_position_budgets",
    "selected_parent_ids", "selected_parent_scores", "candidate_offsets",
    "candidate_cell_rows", "candidate_source_parent_rank",
    "candidate_source_parent_local_rank",
    "candidate_count_by_parent_prefix",
    "implicit_pose_factor_count_by_parent_prefix_position_budget",
    "orientation_rotations_w2c",
)
PARENT_CEILING_ARRAY_NAMES = (
    "image_ids", "parent_prefix_budgets", "selected_parent_ids",
    "selected_parent_scores", "candidate_offsets", "candidate_cell_rows",
    "candidate_count_by_parent_prefix",
)
GLOBAL_ORDER_ARRAY_NAMES = (
    "global_cell_rows", "dyadic_level_bins",
    "dyadic_level_nonempty_bin_counts", "dyadic_level_prefix_counts",
)
PRIORITY_FALLBACK_SCHEMA = (
    "goal_maplet_query_conditioned_parent_priority_global_fallback_proposal_v1"
)
PRIORITY_FALLBACK_SEMANTICS = (
    "top64_parent_ranked_round_robin_priority_exact_rational_interleave_"
    "with_sparse_global_progressive_fallback_v1"
)
FALLBACK_FRACTIONS = ((1, 4), (1, 2), (3, 4))
PRIORITY_FALLBACK_POSITION_BUDGETS = (4096, 8192, 16384, 32768)
PRIORITY_FALLBACK_MAXIMUM_BUDGET = max(PRIORITY_FALLBACK_POSITION_BUDGETS)
PRIORITY_FALLBACK_ARRAY_NAMES = (
    "image_ids", "fallback_fraction_numerators", "fallback_fraction_denominators",
    "total_position_budgets", "selected_parent_ids", "selected_parent_scores",
    "candidate_offsets", "candidate_cell_rows", "candidate_source_queue",
    "candidate_source_parent_rank", "candidate_source_queue_local_rank",
    "candidate_count_by_fallback_fraction",
    "achieved_fallback_count_by_fraction_budget",
    "implicit_pose_factor_count_by_fraction_budget",
    "orientation_rotations_w2c",
)


def _reverse_bits(value: int, width: int) -> int:
    result = 0
    for bit in range(width):
        result |= ((int(value) >> bit) & 1) << (width - 1 - bit)
    return result


def _morton_code(x: int, y: int, z: int, bits: int) -> int:
    code = 0
    for bit in range(bits):
        code |= ((int(x) >> bit) & 1) << (3 * bit)
        code |= ((int(y) >> bit) & 1) << (3 * bit + 1)
        code |= ((int(z) >> bit) & 1) << (3 * bit + 2)
    return code


def progressive_dense_box_cell_order(
    first: np.ndarray, last: np.ndarray,
) -> np.ndarray:
    """Return every integer cell in a box in a nested multiresolution order."""

    start = np.asarray(first, dtype=np.int64)
    stop = np.asarray(last, dtype=np.int64)
    if start.shape != (3,) or stop.shape != (3,) or np.any(stop < start):
        raise ValueError("progressive dense box bounds differ")
    shape = stop - start + 1
    expected = int(np.prod(shape, dtype=np.int64))
    seen: set[tuple[int, int, int]] = set()
    ordered: list[tuple[int, int, int]] = []
    level = 0
    while len(ordered) < expected:
        bins = np.minimum(1 << level, shape).astype(np.int64)
        axis_values: list[list[tuple[int, int]]] = []
        for axis in range(3):
            count = int(bins[axis])
            length = int(shape[axis])
            values = []
            for index in range(count):
                lower = (index * length) // count
                upper = ((index + 1) * length) // count - 1
                values.append((index, int(start[axis]) + (lower + upper) // 2))
            axis_values.append(values)
        bits = max(1, int(math.ceil(math.log2(int(np.max(bins))))))
        occupied_bins = {
            tuple(
                min(
                    int(bins[axis]) - 1,
                    (
                        (cell[axis] - int(start[axis]) + 1) * int(bins[axis])
                        - 1
                    ) // int(shape[axis]),
                )
                for axis in range(3)
            )
            for cell in ordered
        }
        level_candidates = []
        for x, y, z in itertools.product(*axis_values):
            bin_index = (x[0], y[0], z[0])
            cell = (x[1], y[1], z[1])
            morton = _morton_code(*bin_index, bits)
            spread_key = _reverse_bits(morton, 3 * bits)
            level_candidates.append((bin_index in occupied_bins, spread_key, bin_index, cell))
        # A dyadic checkpoint contains exactly one representative per current
        # bin.  Bins already represented by the coarser prefix therefore add
        # nothing at this level; the final singleton level still covers every
        # cell.  Bit-reversed Morton disperses prefixes between checkpoints.
        for occupied, _, _, cell in sorted(level_candidates):
            if occupied:
                continue
            if cell not in seen:
                seen.add(cell)
                ordered.append(cell)
        if np.array_equal(bins, shape):
            break
        level += 1
        if level > 31:
            raise RuntimeError("progressive dense box did not terminate")
    if len(ordered) != expected or len(seen) != expected:
        raise AssertionError("progressive dense box inventory differs")
    return np.asarray(ordered, dtype=np.int32)


def build_sparse_global_progressive_order_arrays(
    cell_indices_world: np.ndarray,
) -> dict[str, np.ndarray]:
    """Order an irregular global support with exact nonempty-bin checkpoints."""

    cells = np.asarray(cell_indices_world, dtype=np.int32)
    if (
        cells.ndim != 2 or cells.shape[1:] != (3,) or cells.size <= 0
        or np.unique(cells, axis=0).shape[0] != cells.shape[0]
    ):
        raise ValueError("sparse global support cells differ")
    first = np.min(cells.astype(np.int64), axis=0)
    last = np.max(cells.astype(np.int64), axis=0)
    shape = last - first + 1
    local = cells.astype(np.int64) - first[None]
    ordered: list[int] = []
    seen: set[int] = set()
    level_bins, nonempty_counts, prefix_counts = [], [], []
    level = 0
    while len(ordered) < cells.shape[0]:
        bins = np.minimum(1 << level, shape).astype(np.int64)
        assignments = ((local + 1) * bins[None] - 1) // shape[None]
        occupied_by_prefix = {
            tuple(assignments[row].tolist()) for row in ordered
        }
        best_by_bin: dict[tuple[int, int, int], tuple[tuple, int]] = {}
        for row in range(cells.shape[0]):
            bin_index = tuple(int(value) for value in assignments[row])
            # Integer twice-centre of the exact floor-partition interval.
            target2 = []
            for axis in range(3):
                index = bin_index[axis]
                lower = (index * int(shape[axis])) // int(bins[axis])
                upper = ((index + 1) * int(shape[axis])) // int(bins[axis]) - 1
                target2.append(2 * int(first[axis]) + lower + upper)
            delta2 = 2 * cells[row].astype(np.int64) - np.asarray(target2)
            key = (
                int(np.dot(delta2, delta2)),
                int(cells[row, 0]), int(cells[row, 1]), int(cells[row, 2]), row,
            )
            current = best_by_bin.get(bin_index)
            if current is None or key < current[0]:
                best_by_bin[bin_index] = (key, row)
        bits = max(1, int(math.ceil(math.log2(int(np.max(bins))))))
        candidates = []
        for bin_index, (_, row) in best_by_bin.items():
            if bin_index in occupied_by_prefix:
                continue
            morton = _morton_code(*bin_index, bits)
            candidates.append((
                _reverse_bits(morton, 3 * bits), bin_index, int(row),
            ))
        for _, _, row in sorted(candidates):
            if row not in seen:
                seen.add(row)
                ordered.append(row)
        level_bins.append(bins.copy())
        nonempty_counts.append(len(best_by_bin))
        prefix_counts.append(len(ordered))
        if len(ordered) != len(best_by_bin):
            raise AssertionError(
                "sparse global dyadic checkpoint is not one-per-nonempty-bin"
            )
        if np.array_equal(bins, shape):
            break
        level += 1
        if level > 31:
            raise RuntimeError("sparse global progressive order did not terminate")
    order = np.asarray(ordered, dtype=np.int32)
    if (
        order.size != cells.shape[0] or np.unique(order).size != order.size
        or not np.array_equal(np.sort(order), np.arange(cells.shape[0]))
    ):
        raise AssertionError("sparse global progressive order is not a permutation")
    return {
        "global_cell_rows": order,
        "dyadic_level_bins": np.asarray(level_bins, dtype=np.int32),
        "dyadic_level_nonempty_bin_counts": np.asarray(
            nonempty_counts, dtype=np.int32,
        ),
        "dyadic_level_prefix_counts": np.asarray(prefix_counts, dtype=np.int32),
    }


def build_parent_progressive_order_arrays(
    global_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    cells = np.asarray(global_arrays["cell_indices_world"], dtype=np.int32)
    maplet_ids = np.asarray(global_arrays["maplet_ids"], dtype=np.int64)
    lower = np.asarray(global_arrays["parent_aabb_min_world"], dtype=np.float64)
    upper = np.asarray(global_arrays["parent_aabb_max_world"], dtype=np.float64)
    origin = np.asarray(global_arrays["lattice_origin_world"], dtype=np.float64)
    parent_count = int(maplet_ids.size)
    if (
        cells.ndim != 2 or cells.shape[1:] != (3,)
        or lower.shape != (parent_count, 3) or upper.shape != lower.shape
    ):
        raise ValueError("global arrays for parent progressive order differ")
    global_row = {tuple(value): row for row, value in enumerate(cells.tolist())}
    if len(global_row) != cells.shape[0]:
        raise ValueError("global support cells are not unique")
    offsets = [0]
    parent_rows = []
    covered = np.zeros((cells.shape[0],), dtype=np.bool_)
    for parent in range(parent_count):
        first, last = _expanded_aabb_cell_range(
            lower[parent], upper[parent], origin,
        )
        progressive = progressive_dense_box_cell_order(first, last)
        try:
            rows = np.asarray(
                [global_row[tuple(value)] for value in progressive.tolist()],
                dtype=np.int32,
            )
        except KeyError as error:
            raise AssertionError("parent progressive cell is outside global-v2") from error
        if np.unique(rows).size != rows.size:
            raise AssertionError("parent progressive order contains duplicate cells")
        parent_rows.append(rows)
        covered[rows] = True
        offsets.append(offsets[-1] + int(rows.size))
    if not np.all(covered):
        raise AssertionError("parent progressive orders do not cover global-v2 support")
    return {
        "maplet_ids": maplet_ids.copy(),
        "parent_cell_offsets": np.asarray(offsets, dtype=np.int64),
        "parent_order_cell_rows": np.concatenate(parent_rows),
    }


def ranked_round_robin_unique_prefix(
    parent_orders: list[np.ndarray], *, maximum_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fair cyclic allocation in evidence-rank order with global deduplication."""

    maximum = int(maximum_count)
    if maximum <= 0 or not parent_orders:
        raise ValueError("ranked round-robin budget differs")
    orders = [np.asarray(value, dtype=np.int32).reshape(-1) for value in parent_orders]
    if any(value.size <= 0 or np.unique(value).size != value.size for value in orders):
        raise ValueError("ranked round-robin parent order differs")
    cursors = np.zeros((len(orders),), dtype=np.int64)
    seen: set[int] = set()
    rows, source_rank, source_local_rank = [], [], []
    while len(rows) < maximum:
        added_in_round = False
        for parent_rank, order in enumerate(orders, start=1):
            cursor = int(cursors[parent_rank - 1])
            while cursor < order.size and int(order[cursor]) in seen:
                cursor += 1
            cursors[parent_rank - 1] = cursor + 1
            if cursor >= order.size:
                continue
            row = int(order[cursor])
            seen.add(row)
            rows.append(row)
            source_rank.append(parent_rank)
            source_local_rank.append(cursor)
            added_in_round = True
            if len(rows) == maximum:
                break
        if not added_in_round:
            break
    if len(rows) < maximum:
        raise ValueError(
            "ranked round-robin union cannot satisfy fixed unique-position budget"
        )
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(source_rank, dtype=np.uint8),
        np.asarray(source_local_rank, dtype=np.int32),
    )


def ranked_round_robin_full_union(
    parent_orders: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    union_count = int(np.unique(np.concatenate(parent_orders)).size)
    return ranked_round_robin_unique_prefix(
        parent_orders, maximum_count=union_count,
    )


def exact_rational_priority_fallback_prefix(
    parent_rows: np.ndarray,
    parent_source_rank: np.ndarray,
    fallback_rows: np.ndarray,
    *,
    fallback_numerator: int,
    fallback_denominator: int,
    maximum_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interleave two queues exactly until a requested source is exhausted."""

    local = np.asarray(parent_rows, dtype=np.int32).reshape(-1)
    local_rank = np.asarray(parent_source_rank, dtype=np.uint8).reshape(-1)
    fallback = np.asarray(fallback_rows, dtype=np.int32).reshape(-1)
    numerator, denominator = int(fallback_numerator), int(fallback_denominator)
    maximum = int(maximum_count)
    if (
        local.shape != local_rank.shape or local.size <= 0 or fallback.size <= 0
        or np.unique(local).size != local.size
        or np.unique(fallback).size != fallback.size
        or numerator <= 0 or numerator >= denominator or denominator <= 0
        or maximum <= 0
    ):
        raise ValueError("priority-fallback interleave inputs differ")
    cursors = [0, 0]
    seen: set[int] = set()
    rows, sources, ranks, source_local = [], [], [], []
    for slot in range(1, maximum + 1):
        fallback_requested = (
            (slot * numerator) // denominator
            > ((slot - 1) * numerator) // denominator
        )
        source = 1 if fallback_requested else 0
        queue = fallback if source else local
        cursor = cursors[source]
        while cursor < queue.size and int(queue[cursor]) in seen:
            cursor += 1
        if cursor >= queue.size:
            # Exact schedule is no longer representable without lying about
            # its source fraction.  The valid nested prefix ends here.
            break
        row = int(queue[cursor])
        cursors[source] = cursor + 1
        seen.add(row)
        rows.append(row)
        sources.append(source)
        ranks.append(0 if source else int(local_rank[cursor]))
        source_local.append(cursor)
    return (
        np.asarray(rows, dtype=np.int32),
        np.asarray(sources, dtype=np.uint8),
        np.asarray(ranks, dtype=np.uint8),
        np.asarray(source_local, dtype=np.int32),
    )


def complete_priority_fallback_order(
    parent_rows: np.ndarray,
    parent_source_rank: np.ndarray,
    fallback_rows: np.ndarray,
    *,
    fallback_numerator: int,
    fallback_denominator: int,
    scheduled_maximum_count: int,
) -> np.ndarray:
    """Materialise the declared implicit tail for contract tests/replay."""

    prefix, _, _, _ = exact_rational_priority_fallback_prefix(
        parent_rows, parent_source_rank, fallback_rows,
        fallback_numerator=fallback_numerator,
        fallback_denominator=fallback_denominator,
        maximum_count=scheduled_maximum_count,
    )
    seen = set(map(int, prefix.tolist()))
    tail = [
        int(row) for row in np.asarray(parent_rows).tolist() if int(row) not in seen
    ]
    seen.update(tail)
    tail.extend(
        int(row) for row in np.asarray(fallback_rows).tolist() if int(row) not in seen
    )
    result = np.concatenate([prefix, np.asarray(tail, dtype=np.int32)])
    if np.unique(result).size != result.size:
        raise AssertionError("priority-fallback implicit completion is not unique")
    return result.astype(np.int32, copy=False)


def build_priority_fallback_query_arrays(
    image_ids: np.ndarray,
    scene_parent_ids: np.ndarray,
    scene_parent_scores: np.ndarray,
    parent_order_arrays: dict[str, np.ndarray],
    sparse_global_order_arrays: dict[str, np.ndarray],
    orientation_rotations_w2c: np.ndarray,
    *,
    global_position_count: int,
) -> dict[str, np.ndarray]:
    image = np.asarray(image_ids)
    parent_ids = np.asarray(scene_parent_ids, dtype=np.int64)
    parent_scores = np.asarray(scene_parent_scores, dtype=np.float64)
    query_count = int(image.size)
    maplet_ids = np.asarray(parent_order_arrays["maplet_ids"], dtype=np.int64)
    parent_offsets = np.asarray(parent_order_arrays["parent_cell_offsets"])
    parent_cells = np.asarray(parent_order_arrays["parent_order_cell_rows"])
    fallback = np.asarray(
        sparse_global_order_arrays["global_cell_rows"], dtype=np.int32,
    )
    if (
        query_count <= 0 or image.ndim != 1 or image.dtype.kind != "U"
        or np.unique(image).size != query_count
        or parent_ids.shape != (query_count, 64)
        or parent_scores.shape != parent_ids.shape
        or np.any(~np.isfinite(parent_scores)) or np.any(parent_scores <= 0.0)
        or np.any(np.diff(parent_scores, axis=1) > 1e-12)
        or any(np.unique(row).size != 64 for row in parent_ids)
        or fallback.shape != (int(global_position_count),)
        or not np.array_equal(np.sort(fallback), np.arange(global_position_count))
    ):
        raise ValueError("priority-fallback query inputs differ")
    maplet_row = {int(value): row for row, value in enumerate(maplet_ids.tolist())}
    if any(int(value) not in maplet_row for value in parent_ids.reshape(-1)):
        raise ValueError("priority parent is absent from parent orders")
    offsets = [0]
    candidate_rows, candidate_sources = [], []
    candidate_parent_ranks, candidate_local_ranks = [], []
    counts = np.zeros((query_count, len(FALLBACK_FRACTIONS)), dtype=np.int32)
    achieved = np.full(
        (
            query_count, len(FALLBACK_FRACTIONS),
            len(PRIORITY_FALLBACK_POSITION_BUDGETS),
        ),
        -1, dtype=np.int32,
    )
    for query in range(query_count):
        orders = []
        for parent_id in parent_ids[query]:
            parent = maplet_row[int(parent_id)]
            orders.append(parent_cells[
                int(parent_offsets[parent]):int(parent_offsets[parent + 1])
            ])
        local_rows, local_source_rank, _ = ranked_round_robin_full_union(orders)
        outside_mask = np.ones((int(global_position_count),), dtype=np.bool_)
        outside_mask[local_rows] = False
        fallback_outside_parent_union = fallback[outside_mask[fallback]]
        if fallback_outside_parent_union.size + local_rows.size != int(
            global_position_count
        ):
            raise AssertionError("priority and fallback queues do not partition global-v2")
        for fraction_index, (numerator, denominator) in enumerate(FALLBACK_FRACTIONS):
            rows, sources, ranks, local = exact_rational_priority_fallback_prefix(
                local_rows, local_source_rank, fallback_outside_parent_union,
                fallback_numerator=numerator,
                fallback_denominator=denominator,
                maximum_count=PRIORITY_FALLBACK_MAXIMUM_BUDGET,
            )
            if (
                np.unique(rows).size != rows.size
                or np.any(rows < 0) or np.any(rows >= int(global_position_count))
            ):
                raise AssertionError("priority-fallback proposal left global-v2")
            for budget_index, budget in enumerate(PRIORITY_FALLBACK_POSITION_BUDGETS):
                if rows.size >= budget:
                    expected = (budget * numerator) // denominator
                    observed = int(np.sum(sources[:budget] == 1))
                    if observed != expected:
                        raise AssertionError("exact rational fallback fraction differs")
                    achieved[query, fraction_index, budget_index] = observed
            candidate_rows.append(rows)
            candidate_sources.append(sources)
            candidate_parent_ranks.append(ranks)
            candidate_local_ranks.append(local)
            counts[query, fraction_index] = rows.size
            offsets.append(offsets[-1] + rows.size)
    implicit = np.broadcast_to(
        np.asarray(PRIORITY_FALLBACK_POSITION_BUDGETS, dtype=np.int64)[None, None]
        * ORIENTATION_COUNT,
        (
            query_count, len(FALLBACK_FRACTIONS),
            len(PRIORITY_FALLBACK_POSITION_BUDGETS),
        ),
    ).copy()
    implicit[achieved < 0] = -1
    return {
        "image_ids": image.copy(),
        "fallback_fraction_numerators": np.asarray(
            [value[0] for value in FALLBACK_FRACTIONS], dtype=np.int16,
        ),
        "fallback_fraction_denominators": np.asarray(
            [value[1] for value in FALLBACK_FRACTIONS], dtype=np.int16,
        ),
        "total_position_budgets": np.asarray(
            PRIORITY_FALLBACK_POSITION_BUDGETS, dtype=np.int32,
        ),
        "selected_parent_ids": parent_ids.copy(),
        "selected_parent_scores": parent_scores.copy(),
        "candidate_offsets": np.asarray(offsets, dtype=np.int64),
        "candidate_cell_rows": np.concatenate(candidate_rows),
        "candidate_source_queue": np.concatenate(candidate_sources),
        "candidate_source_parent_rank": np.concatenate(candidate_parent_ranks),
        "candidate_source_queue_local_rank": np.concatenate(candidate_local_ranks),
        "candidate_count_by_fallback_fraction": counts,
        "achieved_fallback_count_by_fraction_budget": achieved,
        "implicit_pose_factor_count_by_fraction_budget": implicit,
        "orientation_rotations_w2c": np.asarray(
            orientation_rotations_w2c, dtype=np.float64,
        ).copy(),
    }


def build_progressive_query_proposal_arrays(
    image_ids: np.ndarray,
    scene_parent_ids: np.ndarray,
    scene_parent_scores: np.ndarray,
    parent_order_arrays: dict[str, np.ndarray],
    orientation_rotations_w2c: np.ndarray,
    *,
    global_position_count: int,
) -> dict[str, np.ndarray]:
    image = np.asarray(image_ids)
    parent_ids = np.asarray(scene_parent_ids, dtype=np.int64)
    parent_scores = np.asarray(scene_parent_scores, dtype=np.float64)
    query_count = int(image.size)
    maplet_ids = np.asarray(parent_order_arrays["maplet_ids"], dtype=np.int64)
    offsets = np.asarray(parent_order_arrays["parent_cell_offsets"], dtype=np.int64)
    parent_rows = np.asarray(
        parent_order_arrays["parent_order_cell_rows"], dtype=np.int32,
    )
    if (
        query_count <= 0 or image.ndim != 1 or image.dtype.kind != "U"
        or np.unique(image).size != query_count
        or parent_ids.shape != (query_count, MAXIMUM_PARENT_PREFIX)
        or parent_scores.shape != parent_ids.shape
        or np.any(~np.isfinite(parent_scores)) or np.any(parent_scores <= 0.0)
        or np.any(np.diff(parent_scores, axis=1) > 1e-12)
        or any(np.unique(row).size != MAXIMUM_PARENT_PREFIX for row in parent_ids)
        or offsets.shape != (maplet_ids.size + 1,)
        or offsets[0] != 0 or offsets[-1] != parent_rows.size
        or np.any(parent_rows < 0) or np.any(parent_rows >= int(global_position_count))
    ):
        raise ValueError("progressive query proposal inputs differ")
    maplet_row = {int(value): row for row, value in enumerate(maplet_ids.tolist())}
    if any(int(value) not in maplet_row for value in parent_ids.reshape(-1)):
        raise ValueError("retrieved parent is absent from progressive parent orders")
    group_offsets = [0]
    candidate_rows, source_ranks, source_local_ranks = [], [], []
    counts = np.zeros((query_count, len(PARENT_PREFIX_BUDGETS)), dtype=np.int32)
    for query in range(query_count):
        for prefix_index, prefix in enumerate(PARENT_PREFIX_BUDGETS):
            orders = []
            for value in parent_ids[query, :prefix]:
                row = maplet_row[int(value)]
                orders.append(parent_rows[int(offsets[row]):int(offsets[row + 1])])
            rows, ranks, local = ranked_round_robin_unique_prefix(
                orders, maximum_count=MAXIMUM_POSITION_BUDGET,
            )
            if (
                np.unique(rows).size != MAXIMUM_POSITION_BUDGET
                or np.any(rows < 0) or np.any(rows >= int(global_position_count))
            ):
                raise AssertionError("progressive proposal left global-v2 support")
            candidate_rows.append(rows)
            source_ranks.append(ranks)
            source_local_ranks.append(local)
            counts[query, prefix_index] = rows.size
            group_offsets.append(group_offsets[-1] + rows.size)
    implicit = np.broadcast_to(
        np.asarray(POSITION_BUDGETS, dtype=np.int64)[None, None, :]
        * ORIENTATION_COUNT,
        (query_count, len(PARENT_PREFIX_BUDGETS), len(POSITION_BUDGETS)),
    ).copy()
    return {
        "image_ids": image.copy(),
        "parent_prefix_budgets": np.asarray(PARENT_PREFIX_BUDGETS, dtype=np.int16),
        "total_position_budgets": np.asarray(POSITION_BUDGETS, dtype=np.int32),
        "selected_parent_ids": parent_ids.copy(),
        "selected_parent_scores": parent_scores.copy(),
        "candidate_offsets": np.asarray(group_offsets, dtype=np.int64),
        "candidate_cell_rows": np.concatenate(candidate_rows),
        "candidate_source_parent_rank": np.concatenate(source_ranks),
        "candidate_source_parent_local_rank": np.concatenate(source_local_ranks),
        "candidate_count_by_parent_prefix": counts,
        "implicit_pose_factor_count_by_parent_prefix_position_budget": implicit,
        "orientation_rotations_w2c": np.asarray(
            orientation_rotations_w2c, dtype=np.float64,
        ).copy(),
    }


def build_parent_cropped_ceiling_arrays(
    image_ids: np.ndarray,
    scene_parent_ids: np.ndarray,
    scene_parent_scores: np.ndarray,
    parent_order_arrays: dict[str, np.ndarray],
    *,
    global_position_count: int,
    parent_prefix_budgets: tuple[int, ...] = (4, 8, 16, 32, 64),
) -> dict[str, np.ndarray]:
    image = np.asarray(image_ids)
    parent_ids = np.asarray(scene_parent_ids, dtype=np.int64)
    parent_scores = np.asarray(scene_parent_scores, dtype=np.float64)
    budgets = tuple(int(value) for value in parent_prefix_budgets)
    maximum = max(budgets)
    maplet_ids = np.asarray(parent_order_arrays["maplet_ids"], dtype=np.int64)
    offsets = np.asarray(parent_order_arrays["parent_cell_offsets"], dtype=np.int64)
    parent_rows = np.asarray(
        parent_order_arrays["parent_order_cell_rows"], dtype=np.int32,
    )
    query_count = int(image.size)
    if (
        budgets != tuple(sorted(set(budgets))) or budgets[0] <= 0
        or query_count <= 0 or image.ndim != 1 or image.dtype.kind != "U"
        or np.unique(image).size != query_count
        or parent_ids.shape != (query_count, maximum)
        or parent_scores.shape != parent_ids.shape
        or np.any(~np.isfinite(parent_scores)) or np.any(parent_scores <= 0.0)
        or np.any(np.diff(parent_scores, axis=1) > 1e-12)
        or any(np.unique(row).size != maximum for row in parent_ids)
    ):
        raise ValueError("parent-cropped ceiling inputs differ")
    maplet_row = {int(value): row for row, value in enumerate(maplet_ids.tolist())}
    if any(int(value) not in maplet_row for value in parent_ids.reshape(-1)):
        raise ValueError("parent-cropped ceiling parent is absent from map")
    group_offsets = [0]
    groups = []
    counts = np.zeros((query_count, len(budgets)), dtype=np.int32)
    for query in range(query_count):
        for budget_index, budget in enumerate(budgets):
            rows = []
            for value in parent_ids[query, :budget]:
                parent = maplet_row[int(value)]
                rows.append(parent_rows[int(offsets[parent]):int(offsets[parent + 1])])
            union = np.unique(np.concatenate(rows)).astype(np.int32, copy=False)
            if (
                union.size <= 0 or np.any(union < 0)
                or np.any(union >= int(global_position_count))
            ):
                raise AssertionError("parent-cropped ceiling left global-v2 support")
            groups.append(union)
            counts[query, budget_index] = union.size
            group_offsets.append(group_offsets[-1] + int(union.size))
    return {
        "image_ids": image.copy(),
        "parent_prefix_budgets": np.asarray(budgets, dtype=np.int16),
        "selected_parent_ids": parent_ids.copy(),
        "selected_parent_scores": parent_scores.copy(),
        "candidate_offsets": np.asarray(group_offsets, dtype=np.int64),
        "candidate_cell_rows": np.concatenate(groups),
        "candidate_count_by_parent_prefix": counts,
    }


def _load_exact_npz(
    path: Path, array_names: tuple[str, ...], description: str,
) -> tuple[dict[str, np.ndarray], dict]:
    expected = {*(f"{name}.npy" for name in array_names), "metadata_json.npy"}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = [entry.filename for entry in archive.infolist()]
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"{description} is not a valid NPZ") from error
    if len(members) != len(set(members)) or set(members) != expected:
        raise ValueError(f"{description} NPZ members differ")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {*array_names, "metadata_json"}:
            raise ValueError(f"{description} arrays differ")
        arrays = {name: np.asarray(data[name]) for name in array_names}
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
    return arrays, metadata


def load_parent_progressive_order(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    arrays, metadata = _load_exact_npz(
        Path(path), PARENT_ORDER_ARRAY_NAMES, "parent progressive order",
    )
    ids = arrays["maplet_ids"]
    offsets = arrays["parent_cell_offsets"]
    rows = arrays["parent_order_cell_rows"]
    global_count = int(metadata.get("global_support", {}).get("position_count", -1))
    if (
        metadata.get("artifact_type") != PARENT_ORDER_SCHEMA
        or metadata.get("semantics") != PARENT_ORDER_SEMANTICS
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("uses_query_retrieval") is not False
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or ids.ndim != 1 or ids.size <= 0 or np.unique(ids).size != ids.size
        or offsets.shape != (ids.size + 1,) or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0) or offsets[-1] != rows.size
        or rows.ndim != 1 or rows.dtype != np.int32
        or global_count <= 0 or np.any(rows < 0) or np.any(rows >= global_count)
        or np.unique(rows).size != global_count
    ):
        raise ValueError("parent progressive order contract differs")
    for parent in range(ids.size):
        local = rows[int(offsets[parent]):int(offsets[parent + 1])]
        if np.unique(local).size != local.size:
            raise ValueError("parent progressive order contains duplicate cells")
    return arrays, metadata


def load_progressive_query_proposal(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    arrays, metadata = _load_exact_npz(
        Path(path), QUERY_PROPOSAL_ARRAY_NAMES, "progressive query proposal",
    )
    image = arrays["image_ids"]
    query_count = int(image.size)
    offsets = arrays["candidate_offsets"]
    rows = arrays["candidate_cell_rows"]
    group_count = query_count * len(PARENT_PREFIX_BUDGETS)
    if (
        metadata.get("artifact_type") != QUERY_PROPOSAL_SCHEMA
        or metadata.get("semantics") != QUERY_PROPOSAL_SEMANTICS
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or metadata.get("uses_mapping_camera_position_seed") is not False
        or metadata.get("candidate_cells_strictly_from_global_v2") is not True
        or metadata.get("position_budget_prefixes_nested_within_parent_prefix") is not True
        or metadata.get("parent_prefix_domains_required_nested") is not False
        or image.ndim != 1 or image.dtype.kind != "U" or query_count <= 0
        or np.unique(image).size != query_count
        or not np.array_equal(
            arrays["parent_prefix_budgets"], np.asarray(PARENT_PREFIX_BUDGETS),
        )
        or not np.array_equal(
            arrays["total_position_budgets"], np.asarray(POSITION_BUDGETS),
        )
        or arrays["selected_parent_ids"].shape
        != (query_count, MAXIMUM_PARENT_PREFIX)
        or arrays["selected_parent_scores"].shape
        != (query_count, MAXIMUM_PARENT_PREFIX)
        or np.any(~np.isfinite(arrays["selected_parent_scores"]))
        or np.any(arrays["selected_parent_scores"] <= 0.0)
        or np.any(np.diff(arrays["selected_parent_scores"], axis=1) > 1e-12)
        or any(
            np.unique(row).size != MAXIMUM_PARENT_PREFIX
            for row in arrays["selected_parent_ids"]
        )
        or offsets.shape != (group_count + 1,) or offsets[0] != 0
        or np.any(np.diff(offsets) != MAXIMUM_POSITION_BUDGET)
        or offsets[-1] != rows.size
        or rows.dtype != np.int32
        or arrays["candidate_source_parent_rank"].shape != rows.shape
        or arrays["candidate_source_parent_rank"].dtype != np.uint8
        or arrays["candidate_source_parent_local_rank"].shape != rows.shape
        or arrays["candidate_source_parent_local_rank"].dtype != np.int32
        or np.any(arrays["candidate_source_parent_local_rank"] < 0)
        or arrays["candidate_count_by_parent_prefix"].shape
        != (query_count, len(PARENT_PREFIX_BUDGETS))
        or np.any(
            arrays["candidate_count_by_parent_prefix"] != MAXIMUM_POSITION_BUDGET
        )
        or arrays["implicit_pose_factor_count_by_parent_prefix_position_budget"].shape
        != (query_count, len(PARENT_PREFIX_BUDGETS), len(POSITION_BUDGETS))
        or arrays["orientation_rotations_w2c"].shape != (ORIENTATION_COUNT, 3, 3)
        or not np.array_equal(
            arrays["orientation_rotations_w2c"], analytic_orientation_codebook(),
        )
    ):
        raise ValueError("progressive query proposal contract differs")
    global_count = int(metadata.get("global_position_count", -1))
    for group in range(group_count):
        start, end = int(offsets[group]), int(offsets[group + 1])
        local = rows[start:end]
        prefix = PARENT_PREFIX_BUDGETS[group % len(PARENT_PREFIX_BUDGETS)]
        ranks = arrays["candidate_source_parent_rank"][start:end]
        if (
            np.unique(local).size != MAXIMUM_POSITION_BUDGET
            or np.any(local < 0) or np.any(local >= global_count)
            or np.any(ranks < 1) or np.any(ranks > prefix)
        ):
            raise ValueError("progressive proposal group support differs")
        # All smaller B domains are literal prefixes of this one stored order.
        for smaller, larger in zip(POSITION_BUDGETS[:-1], POSITION_BUDGETS[1:]):
            if not np.array_equal(local[:smaller], local[:larger][:smaller]):
                raise ValueError("progressive proposal position prefixes are not nested")
    expected_implicit = np.broadcast_to(
        np.asarray(POSITION_BUDGETS, dtype=np.int64)[None, None, :]
        * ORIENTATION_COUNT,
        (query_count, len(PARENT_PREFIX_BUDGETS), len(POSITION_BUDGETS)),
    )
    if not np.array_equal(
        arrays["implicit_pose_factor_count_by_parent_prefix_position_budget"],
        expected_implicit,
    ):
        raise ValueError("progressive proposal implicit factor counts differ")
    return arrays, metadata


def load_parent_cropped_ceiling(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    arrays, metadata = _load_exact_npz(
        Path(path), PARENT_CEILING_ARRAY_NAMES, "parent-cropped ceiling",
    )
    image = arrays["image_ids"]
    budgets = arrays["parent_prefix_budgets"]
    query_count = int(image.size)
    budget_count = int(budgets.size)
    offsets = arrays["candidate_offsets"]
    rows = arrays["candidate_cell_rows"]
    counts = arrays["candidate_count_by_parent_prefix"]
    global_count = int(metadata.get("global_position_count", -1))
    if (
        metadata.get("artifact_type") != PARENT_CEILING_SCHEMA
        or metadata.get("semantics") != PARENT_CEILING_SEMANTICS
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or metadata.get("candidate_cells_strictly_from_global_v2") is not True
        or image.ndim != 1 or image.dtype.kind != "U" or query_count <= 0
        or np.unique(image).size != query_count
        or not np.array_equal(budgets, np.asarray([4, 8, 16, 32, 64]))
        or arrays["selected_parent_ids"].shape != (query_count, 64)
        or arrays["selected_parent_scores"].shape != (query_count, 64)
        or any(np.unique(row).size != 64 for row in arrays["selected_parent_ids"])
        or np.any(~np.isfinite(arrays["selected_parent_scores"]))
        or np.any(arrays["selected_parent_scores"] <= 0.0)
        or np.any(np.diff(arrays["selected_parent_scores"], axis=1) > 1e-12)
        or any(
            any(
                scores[index] == scores[index + 1]
                and ids[index] > ids[index + 1]
                for index in range(63)
            )
            for ids, scores in zip(
                arrays["selected_parent_ids"], arrays["selected_parent_scores"]
            )
        )
        or offsets.shape != (query_count * budget_count + 1,)
        or offsets[0] != 0 or np.any(np.diff(offsets) <= 0)
        or offsets[-1] != rows.size or rows.dtype != np.int32
        or counts.shape != (query_count, budget_count)
        or global_count <= 0 or np.any(rows < 0) or np.any(rows >= global_count)
    ):
        raise ValueError("parent-cropped ceiling contract differs")
    for group in range(query_count * budget_count):
        start, end = int(offsets[group]), int(offsets[group + 1])
        local = rows[start:end]
        if (
            np.unique(local).size != local.size
            or int(counts.reshape(-1)[group]) != local.size
        ):
            raise ValueError("parent-cropped ceiling group differs")
    return arrays, metadata


def load_sparse_global_progressive_order(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    arrays, metadata = _load_exact_npz(
        Path(path), GLOBAL_ORDER_ARRAY_NAMES, "sparse global progressive order",
    )
    rows = arrays["global_cell_rows"]
    bins = arrays["dyadic_level_bins"]
    nonempty = arrays["dyadic_level_nonempty_bin_counts"]
    prefixes = arrays["dyadic_level_prefix_counts"]
    count = int(metadata.get("global_position_count", -1))
    if (
        metadata.get("artifact_type") != GLOBAL_ORDER_SCHEMA
        or metadata.get("semantics") != GLOBAL_ORDER_SEMANTICS
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("uses_query_retrieval") is not False
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or count <= 0 or rows.shape != (count,) or rows.dtype != np.int32
        or not np.array_equal(np.sort(rows), np.arange(count))
        or bins.ndim != 2 or bins.shape[1:] != (3,) or bins.shape[0] <= 0
        or nonempty.shape != (bins.shape[0],)
        or prefixes.shape != nonempty.shape
        or np.any(np.diff(bins, axis=0) < 0)
        or np.any(np.diff(prefixes) < 0)
        or not np.array_equal(prefixes, nonempty)
        or int(prefixes[-1]) != count
    ):
        raise ValueError("sparse global progressive order contract differs")
    return arrays, metadata


def load_priority_fallback_query_proposal(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict]:
    arrays, metadata = _load_exact_npz(
        Path(path), PRIORITY_FALLBACK_ARRAY_NAMES,
        "priority-fallback query proposal",
    )
    image = arrays["image_ids"]
    query_count = int(image.size)
    fraction_count = len(FALLBACK_FRACTIONS)
    budget_count = len(PRIORITY_FALLBACK_POSITION_BUDGETS)
    group_count = query_count * fraction_count
    offsets = arrays["candidate_offsets"]
    rows = arrays["candidate_cell_rows"]
    sources = arrays["candidate_source_queue"]
    ranks = arrays["candidate_source_parent_rank"]
    source_local = arrays["candidate_source_queue_local_rank"]
    counts = arrays["candidate_count_by_fallback_fraction"]
    achieved = arrays["achieved_fallback_count_by_fraction_budget"]
    global_count = int(metadata.get("global_position_count", -1))
    if (
        metadata.get("artifact_type") != PRIORITY_FALLBACK_SCHEMA
        or metadata.get("semantics") != PRIORITY_FALLBACK_SEMANTICS
        or metadata.get("content_sha256") != arrays_sha256(arrays)
        or metadata.get("uses_query_pose") is not False
        or metadata.get("uses_query_ground_truth") is not False
        or metadata.get("candidate_cells_strictly_from_global_v2") is not True
        or metadata.get("fallback_queue_excludes_complete_top64_parent_union") is not True
        or image.ndim != 1 or image.dtype.kind != "U" or query_count <= 0
        or np.unique(image).size != query_count
        or not np.array_equal(
            arrays["fallback_fraction_numerators"], np.asarray([1, 1, 3]),
        )
        or not np.array_equal(
            arrays["fallback_fraction_denominators"], np.asarray([4, 2, 4]),
        )
        or not np.array_equal(
            arrays["total_position_budgets"],
            np.asarray(PRIORITY_FALLBACK_POSITION_BUDGETS),
        )
        or arrays["selected_parent_ids"].shape != (query_count, 64)
        or arrays["selected_parent_scores"].shape != (query_count, 64)
        or np.any(~np.isfinite(arrays["selected_parent_scores"]))
        or np.any(arrays["selected_parent_scores"] <= 0.0)
        or np.any(np.diff(arrays["selected_parent_scores"], axis=1) > 1e-12)
        or offsets.shape != (group_count + 1,) or offsets[0] != 0
        or np.any(np.diff(offsets) <= 0) or offsets[-1] != rows.size
        or rows.dtype != np.int32 or sources.shape != rows.shape
        or sources.dtype != np.uint8 or not set(np.unique(sources)).issubset({0, 1})
        or ranks.shape != rows.shape or ranks.dtype != np.uint8
        or source_local.shape != rows.shape or source_local.dtype != np.int32
        or np.any(source_local < 0)
        or counts.shape != (query_count, fraction_count)
        or achieved.shape != (query_count, fraction_count, budget_count)
        or arrays["implicit_pose_factor_count_by_fraction_budget"].shape
        != (query_count, fraction_count, budget_count)
        or global_count <= 0 or np.any(rows < 0) or np.any(rows >= global_count)
        or arrays["orientation_rotations_w2c"].shape != (ORIENTATION_COUNT, 3, 3)
        or not np.array_equal(
            arrays["orientation_rotations_w2c"], analytic_orientation_codebook(),
        )
    ):
        raise ValueError("priority-fallback query proposal contract differs")
    for query in range(query_count):
        for fraction_index, (numerator, denominator) in enumerate(FALLBACK_FRACTIONS):
            group = query * fraction_count + fraction_index
            start, end = int(offsets[group]), int(offsets[group + 1])
            local_rows = rows[start:end]
            local_sources = sources[start:end]
            local_source_positions = source_local[start:end]
            parent_source_positions = local_source_positions[local_sources == 0]
            fallback_source_positions = local_source_positions[local_sources == 1]
            if (
                np.unique(local_rows).size != local_rows.size
                or int(counts[query, fraction_index]) != local_rows.size
                or np.any(ranks[start:end][local_sources == 0] < 1)
                or np.any(ranks[start:end][local_sources == 0] > 64)
                or np.any(ranks[start:end][local_sources == 1] != 0)
                or not np.array_equal(
                    parent_source_positions,
                    np.arange(parent_source_positions.size, dtype=np.int32),
                )
                or not np.array_equal(
                    fallback_source_positions,
                    np.arange(fallback_source_positions.size, dtype=np.int32),
                )
            ):
                raise ValueError("priority-fallback proposal group differs")
            for budget_index, budget in enumerate(
                PRIORITY_FALLBACK_POSITION_BUDGETS
            ):
                observed = int(achieved[query, fraction_index, budget_index])
                if local_rows.size >= budget:
                    expected = (budget * numerator) // denominator
                    if observed != expected or int(
                        np.sum(local_sources[:budget] == 1)
                    ) != expected:
                        raise ValueError("priority-fallback exact fraction differs")
                elif observed != -1:
                    raise ValueError("invalid priority-fallback budget is not marked")
    expected_implicit = np.broadcast_to(
        np.asarray(PRIORITY_FALLBACK_POSITION_BUDGETS, dtype=np.int64)[None, None]
        * ORIENTATION_COUNT,
        (query_count, fraction_count, budget_count),
    ).copy()
    expected_implicit[achieved < 0] = -1
    if not np.array_equal(
        arrays["implicit_pose_factor_count_by_fraction_budget"], expected_implicit,
    ):
        raise ValueError("priority-fallback implicit factors differ")
    return arrays, metadata
