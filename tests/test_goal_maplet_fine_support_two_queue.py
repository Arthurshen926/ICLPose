import numpy as np

from feature_extract.tools.vfm.merge_goal_maplet_fine_support_two_queue import (
    interleave_equal_area_queues,
)


def test_equal_area_queues_deduplicate_without_double_charging():
    rows, source = interleave_equal_area_queues(
        np.asarray([0, 1, 2]),
        np.asarray([0, 3, 4]),
        np.ones(5),
        maximum_area_m2=4.0,
        maximum_children=10,
    )
    assert rows.tolist() == [0, 3, 1, 4]
    assert source.tolist() == [0, 1, 0, 1]
    assert np.unique(rows).size == rows.size


def test_equal_area_queues_skip_oversized_item_and_remain_deterministic():
    rows, _ = interleave_equal_area_queues(
        np.asarray([0, 1]),
        np.asarray([2, 3]),
        np.asarray([10.0, 1.0, 1.0, 1.0]),
        maximum_area_m2=2.0,
        maximum_children=4,
    )
    assert rows.tolist() == [1, 2]
