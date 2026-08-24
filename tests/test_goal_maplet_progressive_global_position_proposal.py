from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_progressive_global_position_proposal import (
    _select_configuration as select_parent_grid_configuration,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_priority_fallback_position_proposal import (
    _select_configuration as select_priority_fallback_configuration,
)
from feature_extract.vfm.localization_goal_maplet.parent_geometry_pose_proposal import (
    analytic_orientation_codebook,
)
from feature_extract.vfm.localization_goal_maplet.progressive_global_position_proposal import (
    MAXIMUM_POSITION_BUDGET,
    PARENT_PREFIX_BUDGETS,
    POSITION_BUDGETS,
    build_progressive_query_proposal_arrays,
    build_sparse_global_progressive_order_arrays,
    complete_priority_fallback_order,
    exact_rational_priority_fallback_prefix,
    progressive_dense_box_cell_order,
    ranked_round_robin_unique_prefix,
)


def test_dyadic_parent_order_is_complete_unique_nested_and_deterministic():
    first = np.asarray([-2, 4, 7])
    last = np.asarray([5, 9, 10])
    one = progressive_dense_box_cell_order(first, last)
    two = progressive_dense_box_cell_order(first, last)
    assert np.array_equal(one, two)
    assert one.shape == (8 * 6 * 4, 3)
    assert np.unique(one, axis=0).shape[0] == one.shape[0]
    expected = {
        (x, y, z)
        for x in range(-2, 6) for y in range(4, 10) for z in range(7, 11)
    }
    assert set(map(tuple, one.tolist())) == expected
    for shape in ((8, 8, 8), (5, 5, 5), (7, 9, 5)):
        shape_array = np.asarray(shape, dtype=np.int64)
        progressive = progressive_dense_box_cell_order(
            np.zeros(3, dtype=np.int64), shape_array - 1,
        )
        first_eight_octants = np.asarray([
            [((int(cell[axis]) + 1) * 2 - 1) // int(shape_array[axis])
             for axis in range(3)]
            for cell in progressive[:8]
        ], dtype=np.int8)
        assert np.unique(first_eight_octants, axis=0).shape[0] == 8


def test_every_small_box_dyadic_checkpoint_has_one_cell_per_bin():
    for nx in range(1, 7):
        for ny in range(1, 7):
            for nz in range(1, 7):
                shape = np.asarray([nx, ny, nz], dtype=np.int64)
                order = progressive_dense_box_cell_order(
                    np.zeros(3, dtype=np.int64), shape - 1,
                )
                level = 0
                while True:
                    bins = np.minimum(1 << level, shape)
                    checkpoint = int(np.prod(bins))
                    prefix = order[:checkpoint]
                    assigned = np.asarray([
                        [((int(cell[axis]) + 1) * int(bins[axis]) - 1)
                         // int(shape[axis]) for axis in range(3)]
                        for cell in prefix
                    ])
                    assert np.unique(assigned, axis=0).shape[0] == checkpoint
                    if np.array_equal(bins, shape):
                        break
                    level += 1


def test_sparse_global_order_handles_holes_stalled_levels_and_translation():
    cells = np.asarray([[0, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=np.int32)
    arrays = build_sparse_global_progressive_order_arrays(cells)
    assert arrays["global_cell_rows"].tolist() == [1, 0, 2]
    assert arrays["dyadic_level_prefix_counts"].tolist() == [1, 2, 2, 3]
    assert np.array_equal(
        np.sort(arrays["global_cell_rows"]), np.arange(cells.shape[0]),
    )
    shifted = build_sparse_global_progressive_order_arrays(
        cells + np.asarray([-17, 9, 23], dtype=np.int32),
    )
    assert np.array_equal(
        arrays["global_cell_rows"], shifted["global_cell_rows"],
    )


def test_exact_rational_interleave_is_unique_exact_and_fails_closed():
    local = np.arange(0, 40_000, dtype=np.int32)
    ranks = (np.arange(local.size) % 64 + 1).astype(np.uint8)
    fallback = np.arange(40_000, 90_000, dtype=np.int32)
    for numerator, denominator in ((1, 4), (1, 2), (3, 4)):
        rows, sources, parent_ranks, _ = exact_rational_priority_fallback_prefix(
            local, ranks, fallback,
            fallback_numerator=numerator, fallback_denominator=denominator,
            maximum_count=32768,
        )
        assert rows.size == 32768
        assert np.unique(rows).size == rows.size
        for budget in (4096, 8192, 16384, 32768):
            assert int(np.sum(sources[:budget] == 1)) == (
                budget * numerator // denominator
            )
        assert np.all(parent_ranks[sources == 1] == 0)
        assert np.all((parent_ranks[sources == 0] >= 1)
                      & (parent_ranks[sources == 0] <= 64))
    short, _, _, _ = exact_rational_priority_fallback_prefix(
        np.arange(3, dtype=np.int32), np.ones(3, dtype=np.uint8),
        np.arange(100, 120, dtype=np.int32),
        fallback_numerator=1, fallback_denominator=4, maximum_count=8,
    )
    assert short.size < 8

    completed = complete_priority_fallback_order(
        np.asarray([0, 1, 2], dtype=np.int32), np.ones(3, dtype=np.uint8),
        np.asarray([3, 4, 5, 6], dtype=np.int32),
        fallback_numerator=1, fallback_denominator=4,
        scheduled_maximum_count=8,
    )
    assert np.array_equal(np.sort(completed), np.arange(7))


def test_ranked_round_robin_is_fair_deduplicated_and_has_provenance():
    rows, ranks, local = ranked_round_robin_unique_prefix([
        np.asarray([1, 2, 3, 6, 8], dtype=np.int32),
        np.asarray([1, 4, 5, 7, 9], dtype=np.int32),
    ], maximum_count=8)
    assert rows.tolist() == [1, 4, 2, 5, 3, 7, 6, 9]
    assert ranks.tolist() == [1, 2, 1, 2, 1, 2, 1, 2]
    assert local.tolist() == [0, 1, 1, 2, 2, 3, 3, 4]
    assert np.unique(rows).size == rows.size


def test_query_grid_rows_stay_in_global_support_and_budgets_are_prefixes():
    parent_count = 32
    local_count = 5000
    orders = np.tile(np.arange(local_count, dtype=np.int32), parent_count)
    parent_order = {
        "maplet_ids": np.arange(100, 100 + parent_count, dtype=np.int64),
        "parent_cell_offsets": np.arange(
            0, (parent_count + 1) * local_count, local_count, dtype=np.int64,
        ),
        "parent_order_cell_rows": orders,
    }
    arrays = build_progressive_query_proposal_arrays(
        np.asarray(["seq10/frame.png"]),
        parent_order["maplet_ids"][None],
        np.linspace(2.0, 1.0, parent_count, dtype=np.float64)[None],
        parent_order,
        analytic_orientation_codebook(),
        global_position_count=local_count,
    )
    assert arrays["candidate_count_by_parent_prefix"].tolist() == [
        [MAXIMUM_POSITION_BUDGET] * len(PARENT_PREFIX_BUDGETS)
    ]
    for group, parent_prefix in enumerate(PARENT_PREFIX_BUDGETS):
        start = int(arrays["candidate_offsets"][group])
        end = int(arrays["candidate_offsets"][group + 1])
        rows = arrays["candidate_cell_rows"][start:end]
        ranks = arrays["candidate_source_parent_rank"][start:end]
        assert rows.size == MAXIMUM_POSITION_BUDGET
        assert np.unique(rows).size == rows.size
        assert np.all((rows >= 0) & (rows < local_count))
        assert np.all((ranks >= 1) & (ranks <= parent_prefix))
        for smaller, larger in zip(POSITION_BUDGETS[:-1], POSITION_BUDGETS[1:]):
            assert np.array_equal(rows[:smaller], rows[:larger][:smaller])
    expected = np.asarray(POSITION_BUDGETS) * 60
    assert np.all(
        arrays["implicit_pose_factor_count_by_parent_prefix_position_budget"]
        == expected[None, None, :]
    )


def test_seq10_selection_minimizes_total_budget_before_parent_prefix():
    rows = [
        {"parent_prefix_budget": 4, "total_position_budget": 256,
         "position_acquisition_hits": 83},
        {"parent_prefix_budget": 32, "total_position_budget": 256,
         "position_acquisition_hits": 84},
        {"parent_prefix_budget": 4, "total_position_budget": 512,
         "position_acquisition_hits": 88},
    ]
    selected, required = select_parent_grid_configuration(rows, query_count=88)
    assert required == 84
    assert selected == {
        "parent_prefix_budget": 32,
        "total_position_budget": 256,
        "position_acquisition_hits": 84,
    }


def test_priority_fallback_selection_rejects_invalid_and_orders_budget_then_fraction():
    rows = [
        {"configuration_valid_for_all_queries": False,
         "total_position_budget": 4096, "fallback_fraction": 0.25,
         "position_acquisition_hits_2m": None},
        {"configuration_valid_for_all_queries": True,
         "total_position_budget": 8192, "fallback_fraction": 0.5,
         "fallback_fraction_numerator": 1, "fallback_fraction_denominator": 2,
         "position_acquisition_hits_2m": 86},
        {"configuration_valid_for_all_queries": True,
         "total_position_budget": 8192, "fallback_fraction": 0.25,
         "fallback_fraction_numerator": 1, "fallback_fraction_denominator": 4,
         "position_acquisition_hits_2m": 84},
    ]
    selected, required = select_priority_fallback_configuration(rows, 88)
    assert required == 84
    assert selected["total_position_budget"] == 8192
    assert selected["fallback_fraction"] == 0.25
