import numpy as np
import pytest

from feature_extract.tools.vfm.select_goal_maplet_cross_atlas_geometry_consensus import _select


def test_four_geometry_consensus_and_stable_tie():
    scores = np.asarray([
        [[0.9, 0.8, 0.4, 0.3], [0.6, 0.6, 0.7, 0.7]],
        [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
    ])
    assert _select(scores).tolist() == [1, 0]


def test_rejects_missing_geometry_axis():
    with pytest.raises(ValueError):
        _select(np.ones((3, 2, 1)))
