import numpy as np
from sklearn.linear_model import LogisticRegression

from feature_extract.vfm.localization_goal_maplet.child_local_mode_ranker import (
    FEATURE_NAMES,
    ChildLocalPairwiseRankerArtifact,
)


def test_child_local_mode_ranker_feature_contract_has_no_absolute_pose():
    assert len(FEATURE_NAMES) == 11
    assert not any("world" in name or "absolute" in name for name in FEATURE_NAMES)
    assert "normalized_reprojection_residual" in FEATURE_NAMES
    assert "vfm_geometry_joint_score" in FEATURE_NAMES


def test_pairwise_ranker_scores_only_valid_modes():
    estimator = LogisticRegression().fit(
        np.asarray([[1.0] * len(FEATURE_NAMES), [-1.0] * len(FEATURE_NAMES)]),
        np.asarray([1, 0]),
    )
    artifact = ChildLocalPairwiseRankerArtifact(estimator, {
        "artifact_type": "goal_maplet_child_local_pairwise_ranker_v2",
        "feature_names": list(FEATURE_NAMES),
        "score_temperature": 0.5,
    })
    features = np.zeros((2, 3, len(FEATURE_NAMES)), dtype=np.float32)
    features[0, 0] = 1.0
    features[0, 1] = -1.0
    features[1, 2] = 0.5
    valid = np.asarray([[True, True, False], [False, False, True]])
    score, probability = artifact.score_modes(features, valid)
    assert score[0, 0] > score[0, 1]
    assert np.isneginf(score[0, 2])
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert probability[1, 2] == 1.0
