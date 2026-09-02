from __future__ import annotations

import numpy as np

from feature_extract.tools.vfm.build_goal_maplet_direct_radio_plane_ranking import (
    _aggregate_observation_scores,
)


def test_top2_mean_requires_two_consistent_observations() -> None:
    scores = np.asarray([0.99, 0.10, 0.80, 0.79, 0.78], np.float64)
    offsets = np.asarray([0, 2, 5], np.int64)
    maximum = _aggregate_observation_scores(scores, offsets, "max")
    robust = _aggregate_observation_scores(scores, offsets, "top2_mean")
    assert maximum.tolist() == [0.99, 0.80]
    assert np.allclose(robust, [0.545, 0.795])
    assert int(np.argmax(maximum)) == 0
    assert int(np.argmax(robust)) == 1
