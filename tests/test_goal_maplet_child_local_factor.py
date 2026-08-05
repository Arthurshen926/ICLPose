import numpy as np
from sklearn.linear_model import LogisticRegression

from feature_extract.vfm.localization_goal_maplet.child_local_factor import (
    FEATURE_NAMES,
    NULL_TYPES,
    ChildLocalFactorCalibratorArtifact,
)


def test_child_local_factor_contract_is_relative_and_typed():
    assert NULL_TYPES == (
        "valid", "wrong_child", "pose_incompatible", "unresolved", "field_missing"
    )
    assert not any("world" in name or "absolute" in name for name in FEATURE_NAMES)
    assert "coverage_null_probability" in FEATURE_NAMES
    assert "minimum_normalized_reprojection" in FEATURE_NAMES


def test_child_local_factor_calibrator_restores_missing_classes():
    estimator = LogisticRegression().fit(
        np.asarray([[0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)]),
        np.asarray([0, 2]),
    )
    artifact = ChildLocalFactorCalibratorArtifact(estimator, {
        "artifact_type": "goal_maplet_child_local_factor_calibrator_v2",
        "feature_names": list(FEATURE_NAMES),
        "null_types": list(NULL_TYPES),
        "calibration_temperature": 1.0,
    })
    probability = artifact.predict_typed_probabilities(
        np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    )
    assert probability.shape == (3, len(NULL_TYPES))
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)
    assert np.all(probability[:, 1] < 1e-8)
