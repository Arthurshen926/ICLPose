import numpy as np
import pytest

from feature_extract.tools.vfm.fuse_goal_maplet_retrieval_scene_child_branches import (
    stable_rank_paired_scene_child_union,
)


def test_scene_child_union_is_rank_paired_deduplicated_and_provenance_bound():
    rows, scores, provenance = stable_rank_paired_scene_child_union(
        np.asarray([1, 2, 3]), np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        np.asarray([3, 4, 1]), np.asarray([0.7, 0.6, 0.9], dtype=np.float32),
        maximum_children=8,
    )
    np.testing.assert_array_equal(rows, [1, 3, 2, 4])
    np.testing.assert_array_equal(
        scores, np.asarray([0.9, 0.7, 0.8, 0.6], dtype=np.float32),
    )
    assert provenance == [
        {"union_rank": 1, "child_row": 1, "baseline_rank": 1,
         "f50_rank": 3, "branches": ["baseline", "f50"]},
        {"union_rank": 2, "child_row": 3, "baseline_rank": 3,
         "f50_rank": 1, "branches": ["baseline", "f50"]},
        {"union_rank": 3, "child_row": 2, "baseline_rank": 2,
         "f50_rank": None, "branches": ["baseline"]},
        {"union_rank": 4, "child_row": 4, "baseline_rank": None,
         "f50_rank": 2, "branches": ["f50"]},
    ]


def test_scene_child_union_uses_a_fixed_prefix_cap_without_labels():
    rows, _, provenance = stable_rank_paired_scene_child_union(
        np.asarray([1, 2, 3]), np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        np.asarray([4, 5, 6]), np.asarray([0.6, 0.5, 0.4], dtype=np.float32),
        maximum_children=3,
    )
    np.testing.assert_array_equal(rows, [1, 4, 2])
    assert [value["union_rank"] for value in provenance] == [1, 2, 3]


def test_scene_child_union_rejects_score_drift_and_internal_duplicates():
    with pytest.raises(ValueError, match="bit-identical"):
        stable_rank_paired_scene_child_union(
            np.asarray([1]), np.asarray([0.9], dtype=np.float32),
            np.asarray([1]), np.asarray([0.8], dtype=np.float32),
            maximum_children=2,
        )
    with pytest.raises(ValueError, match="invalid"):
        stable_rank_paired_scene_child_union(
            np.asarray([1, 1]), np.asarray([0.9, 0.9], dtype=np.float32),
            np.asarray([2]), np.asarray([0.8], dtype=np.float32),
            maximum_children=2,
        )
