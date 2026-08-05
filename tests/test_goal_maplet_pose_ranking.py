import numpy as np

from feature_extract.vfm.localization_goal_maplet.pose_proposal import CoarsePoseModes
from feature_extract.vfm.localization_goal_maplet.pose_ranking import _conditional_child_probability


def test_pose_mode_ordering_contract_keeps_support_aligned():
    modes = CoarsePoseModes(
        poses_w2c=np.stack([np.eye(4), 2.0 * np.eye(4)]),
        scores=np.asarray([0.1, 0.2]),
        supporting_region_count=np.asarray([3, 4]),
    )
    order = np.argsort(-modes.scores)
    ranked = CoarsePoseModes(modes.poses_w2c[order], modes.scores[order], modes.supporting_region_count[order])
    assert ranked.supporting_region_count.tolist() == [4, 3]


def test_child_rank_factor_removes_already_scored_parent_mass():
    assert np.isclose(_conditional_child_probability(0.12, 0.30, 1e-4), 0.4)
    assert np.isclose(_conditional_child_probability(0.30, 0.30, 1e-4), 1.0)
