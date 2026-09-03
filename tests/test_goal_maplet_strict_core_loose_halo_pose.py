import numpy as np
import pytest

from feature_extract.tools.vfm.refine_goal_maplet_strict_core_loose_halo_pose import (
    _core_first_hypotheses,
    _exclusive_loose_rows,
)


def _pose_inputs():
    pose = np.eye(4)
    world = np.asarray([[0.0, 0.0, 2.0], [0.02, 0.0, 2.0], [0.0, 0.0, 2.0]])
    token = np.asarray([1, 1, 2])
    pixel = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    K = np.eye(3)
    return pose, world, token, pixel, K


def test_core_hypothesis_is_preferred_even_when_halo_residual_is_smaller():
    pose, world, token, pixel, K = _pose_inputs()
    selected, _ = _core_first_hypotheses(
        pose, world, token, pixel, np.asarray([False, True, True]), K, 0.0,
    )
    assert selected.tolist() == [0, 2]


def test_halo_is_used_when_core_is_not_geometrically_valid():
    pose, world, token, pixel, K = _pose_inputs()
    world[0, 0] = 20.0
    selected, _ = _core_first_hypotheses(
        pose, world, token, pixel, np.asarray([False, True, True]), K, 0.0,
    )
    assert selected.tolist() == [1, 2]


def test_loose_exclusive_rows_deduplicate_physical_match_identity():
    strict = {
        "query_tokens": np.asarray([1]),
        "provenance_region_plane_atlas_row": np.asarray([[2, 3, 4]]),
        "prototype_atlas_row": np.asarray([5]),
    }
    loose = {
        "query_tokens": np.asarray([1, 1]),
        "provenance_region_plane_atlas_row": np.asarray([[2, 3, 4], [2, 3, 6]]),
        "prototype_atlas_row": np.asarray([5, 7]),
    }
    assert _exclusive_loose_rows(strict, 0, 1, loose, 0, 2).tolist() == [1]


def test_core_halo_rejects_mismatched_array_lengths():
    pose, world, token, pixel, K = _pose_inputs()
    with pytest.raises(ValueError, match="arrays differ"):
        _core_first_hypotheses(pose, world, token, pixel, np.ones(2, bool), K, 0.0)
