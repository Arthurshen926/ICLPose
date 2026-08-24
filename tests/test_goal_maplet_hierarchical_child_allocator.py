import numpy as np

from feature_extract.vfm.localization_goal_maplet.hierarchical_child_allocator import (
    allocate_parent_balanced_scene_children,
)
from test_goal_maplet_pure_retrieval import _physical


def test_parent_balanced_allocator_seeds_parent_prefix_before_global_fill():
    physical = _physical()
    parent_ids = physical.maplet_ids[:1]
    parent_scores = np.asarray([0.8], dtype=np.float32)
    score = np.arange(
        1, physical.child_parent_rows.size + 1, dtype=np.float64
    )
    result = allocate_parent_balanced_scene_children(
        parent_ids,
        parent_scores,
        score,
        physical,
        parent_mass_fraction=1.0,
        maximum_children=2,
    )
    np.testing.assert_array_equal(result.seeded_parent_rows, np.asarray([0]))
    assert result.represented_parent_count == 1


def test_zero_parent_fraction_reproduces_global_score_order_without_overlap():
    physical = _physical()
    score = np.arange(1, physical.child_parent_rows.size + 1, dtype=np.float64)
    result = allocate_parent_balanced_scene_children(
        physical.maplet_ids[:1],
        np.asarray([0.8]),
        score,
        physical,
        parent_mass_fraction=0.0,
        maximum_children=2,
        maximum_primitive_iou=1.0,
    )
    np.testing.assert_array_equal(
        result.child_rows,
        np.asarray([score.size - 1, score.size - 2], dtype=np.int64),
    )
