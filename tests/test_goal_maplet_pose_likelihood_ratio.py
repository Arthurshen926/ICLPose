import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from feature_extract.vfm.localization_goal_maplet.child_local_factor import FEATURE_NAMES
from feature_extract.vfm.localization_goal_maplet.pose_likelihood_ratio import PoseLikelihoodRatioArtifact


def _artifact() -> PoseLikelihoodRatioArtifact:
    x = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = [-2.0, -1.0, 1.0, 2.0]
    estimator = make_pipeline(
        StandardScaler(), LogisticRegression(fit_intercept=False, random_state=0)
    ).fit(x, np.asarray([0, 0, 1, 1]))
    return PoseLikelihoodRatioArtifact(
        estimator, 1.5, -0.25,
        {
            "artifact_type": "goal_maplet_pose_likelihood_ratio_v1",
            "feature_names": list(FEATURE_NAMES),
            "pairing_contract": "same_image_same_query_group_exact_v1",
            "feature_input_contract": "deployment_query_group_center_v1",
            "runtime_maximum_children": 4,
        },
    )


def test_pose_likelihood_ratio_is_scalar_additive_evidence():
    artifact = _artifact()
    feature = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    feature[:, 0] = [-1.0, 0.0, 1.0]
    score = artifact.score_log_likelihood_ratio(feature)
    probability = artifact.probability_correct(feature)
    assert score.shape == (3,)
    assert np.all(np.diff(score) > 0.0)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.log(probability[:, 1] / probability[:, 0]), score)


def test_pose_likelihood_ratio_round_trip_and_contract(tmp_path):
    artifact = _artifact()
    path = tmp_path / "factor.joblib"
    artifact.save(path)
    restored = PoseLikelihoodRatioArtifact.load(path)
    feature = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    np.testing.assert_allclose(
        artifact.score_log_likelihood_ratio(feature),
        restored.score_log_likelihood_ratio(feature),
    )
    with pytest.raises(ValueError, match="dimension"):
        restored.score_log_likelihood_ratio(np.zeros((1, 2), dtype=np.float32))
