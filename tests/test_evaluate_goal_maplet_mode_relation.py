import numpy as np

from feature_extract.tools.vfm.evaluate_goal_maplet_mode_relation import _rank_percentile


def test_rank_percentile_uses_average_rank_for_ties():
    score = _rank_percentile(np.asarray([2.0, 1.0, 1.0, 0.0]))
    assert score[0] == 1.0
    assert score[1] == score[2] == 0.5
    assert score[3] == 0.0
    assert np.all(_rank_percentile(np.ones(4)) == 0.5)
