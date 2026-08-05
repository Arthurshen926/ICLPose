import numpy as np

from feature_extract.vfm.localization_goal_maplet.pose_proposal import CoarsePoseModes
from feature_extract.vfm.localization_goal_maplet.pose_ranking import (
    RenderIdentityRanking,
    _conditional_child_probability,
    merge_cascade_identity_rankings,
)


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


def test_cascade_ranking_keeps_every_candidate_feature_aligned():
    poses = np.stack([np.eye(4) for _ in range(4)])
    for index in range(4):
        poses[index, 0, 3] = index
    cheap_modes = CoarsePoseModes(poses, np.asarray([4.0, 3.0, 2.0, 1.0]), np.asarray([40, 30, 20, 10]))
    cheap = RenderIdentityRanking(
        cheap_modes,
        np.asarray([4.0, 3.0, 2.0, 1.0]),
        np.asarray([14.0, 13.0, 12.0, 11.0]),
        np.asarray([24.0, 23.0, 22.0, 21.0]),
        np.asarray([0.4, 0.3, 0.2, 0.1]),
        np.asarray([104.0, 103.0, 102.0, 101.0]),
        np.asarray([3, 2, 1, 0]),
    )
    exact = RenderIdentityRanking(
        CoarsePoseModes(poses[[1, 0]], np.asarray([9.0, 8.0]), np.asarray([30, 40])),
        np.asarray([9.0, 8.0]),
        np.asarray([19.0, 18.0]),
        np.asarray([29.0, 28.0]),
        np.asarray([0.9, 0.8]),
        np.asarray([3.0, 4.0]),
        np.asarray([1, 0]),
    )
    merged = merge_cascade_identity_rankings(cheap, exact, exact_candidate_count=2)
    assert merged.modes.poses_w2c[:, 0, 3].tolist() == [1.0, 0.0, 2.0, 3.0]
    assert merged.cheap_identity_scores.tolist() == [3.0, 4.0, 2.0, 1.0]
    assert merged.proposal_scores.tolist() == [103.0, 104.0, 102.0, 101.0]
    assert merged.original_indices.tolist() == [2, 3, 1, 0]
    assert merged.exact_evaluated.tolist() == [True, True, False, False]
    np.testing.assert_allclose(merged.exact_identity_scores[:2], [9.0, 8.0])
    np.testing.assert_allclose(merged.exact_identity_scores[2:], 0.0)
