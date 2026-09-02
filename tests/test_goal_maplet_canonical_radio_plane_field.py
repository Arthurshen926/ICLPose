import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_canonical_radio_plane_field import (
    canonicalize,
    leave_one_view_out_ranks,
)


def test_same_view_fragments_do_not_get_extra_weight():
    offsets = np.asarray([0, 3, 5])
    descriptor = np.asarray([
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.0, 1.0], [1.0, 0.0],
    ], np.float32)
    names = np.asarray(["a", "a", "b", "a", "b"])
    canonical, audit = canonicalize(offsets, descriptor, names)
    assert np.allclose(canonical[0], canonical[1])
    assert audit["same_view_duplicate_observation_count"] == 1


def test_mapping_leave_one_view_out_ranking_is_finite():
    offsets = np.asarray([0, 2, 4])
    descriptor = np.asarray([
        [1.0, 0.0], [.9, .1], [0.0, 1.0], [.1, .9],
    ], np.float32)
    names = np.asarray(["a", "b", "a", "b"])
    canonical, _ = canonicalize(offsets, descriptor, names)
    ranks = leave_one_view_out_ranks(offsets, descriptor, names, canonical)
    assert np.array_equal(ranks, np.ones(4, np.int64))
