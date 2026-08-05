import numpy as np

from feature_extract.vfm.localization_goal_maplet.configuration_ranker import (
    BASE_FEATURE_NAMES,
    BASE_CONFIGURATION_FEATURE_NAMES,
    EXACT_FEATURE_NAMES,
    FEATURE_NAMES,
    ConfigurationPairwiseRankerArtifact,
    candidate_runtime_features,
)
from feature_extract.vfm.localization_goal_maplet.configuration_evidence import (
    FEATURE_NAMES as CONFIGURATION_EVIDENCE_FEATURE_NAMES,
)
from feature_extract.tools.vfm.evaluate_goal_maplet_pose_modes import _stable_proposal_seed


def test_candidate_runtime_features_exclude_absolute_pose():
    details = []
    for index, x in enumerate((10.0, 10.5)):
        pose = np.eye(4)
        details.append({
            "supporting_region_count": 10 + index,
            "pose_w2c": pose.tolist(),
            "camera_center": [x, 0.0, 0.0],
        })
    row = {
        "mode_details": {"actual_parent_actual_child": details},
        "ranking_diagnostics": {"actual_parent_actual_child": {
            "identity_scores": [-1.0, -2.0],
            "parent_log_likelihood": [-1.5, -2.5],
            "child_log_likelihood": [-0.5, -1.5],
            "rendered_coverage": [0.5, 0.4],
            "proposal_scores": [2.0, 1.0],
            "original_indices": [0, 1],
        }},
    }
    feature = candidate_runtime_features(row)
    assert feature.shape == (2, len(FEATURE_NAMES))
    shifted = {**row, "mode_details": {"actual_parent_actual_child": [
        {**d, "camera_center": [d["camera_center"][0] + 100.0, 0.0, 0.0]} for d in details
    ]}}
    np.testing.assert_allclose(feature, candidate_runtime_features(shifted), atol=1e-6)


def test_exact_candidate_features_require_and_preserve_evaluated_mask():
    details = []
    for index in range(3):
        pose = np.eye(4)
        details.append({
            "supporting_region_count": 10,
            "pose_w2c": pose.tolist(),
            "camera_center": [float(index), 0.0, 0.0],
        })
    diagnostics = {
        "identity_scores": [-1.0, -1.1, -1.2],
        "parent_log_likelihood": [-1.0, -1.1, -1.2],
        "child_log_likelihood": [-1.0, -1.1, -1.2],
        "rendered_coverage": [0.4, 0.5, 0.6],
        "proposal_scores": [3.0, 2.0, 1.0],
        "original_indices": [0, 1, 2],
        "cascade_exact_evaluated": [True, True, False],
        "cascade_exact_parent_log_likelihood": [-0.8, -0.9, 0.0],
        "cascade_exact_child_log_likelihood": [-0.7, -1.0, 0.0],
        "cascade_exact_rendered_coverage": [0.9, 0.8, 0.0],
        "cascade_exact_scores": [-0.75, -0.95, 0.0],
    }
    row = {
        "mode_details": {"actual_parent_actual_child": details},
        "ranking_diagnostics": {"actual_parent_actual_child": diagnostics},
    }
    feature = candidate_runtime_features(row, include_exact=True)
    assert feature.shape == (3, len(EXACT_FEATURE_NAMES))
    assert feature[:, -7].tolist() == [1.0, 1.0, 0.0]
    assert np.isfinite(feature).all()


def test_stable_proposal_seed_fits_signed_c_int():
    image_ids = [f"seq9/frame{index:05d}.png" for index in range(100)]
    values = [_stable_proposal_seed(image_id) for image_id in image_ids]
    assert values == [_stable_proposal_seed(image_id) for image_id in image_ids]
    assert all(0 <= value <= 0x7FFFFFFF for value in values)
    assert len(set(values)) == len(values)


def test_configuration_pairwise_ranker_is_set_relative():
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression().fit(
        np.asarray([[1.0] * len(BASE_FEATURE_NAMES), [-1.0] * len(BASE_FEATURE_NAMES)]),
        np.asarray([1, 0]),
    )
    artifact = ConfigurationPairwiseRankerArtifact(estimator, {
        "artifact_type": "goal_maplet_configuration_pairwise_ranker_v2",
        "feature_names": list(BASE_FEATURE_NAMES),
    })
    features = np.asarray([
        [1.0] * len(BASE_FEATURE_NAMES),
        [0.0] * len(BASE_FEATURE_NAMES),
        [-1.0] * len(BASE_FEATURE_NAMES),
    ], dtype=np.float32)
    score = artifact.score_candidates(features)
    assert score.shape == (3,)
    assert score[0] > score[1] > score[2]


def test_configuration_evidence_columns_are_fail_closed():
    details = [{
        "supporting_region_count": 10,
        "pose_w2c": np.eye(4).tolist(),
        "camera_center": [0.0, 0.0, 0.0],
    }]
    diagnostics = {
        "identity_scores": [-1.0], "parent_log_likelihood": [-1.0],
        "child_log_likelihood": [-1.0], "rendered_coverage": [0.5],
        "proposal_scores": [1.0], "original_indices": [0],
        "configuration_evidence_v2": {
            name: [float(index)] for index, name in enumerate(CONFIGURATION_EVIDENCE_FEATURE_NAMES)
        },
    }
    row = {
        "mode_details": {"actual_parent_actual_child": details},
        "ranking_diagnostics": {"actual_parent_actual_child": diagnostics},
    }
    feature = candidate_runtime_features(row, include_configuration=True)
    assert feature.shape == (1, len(BASE_CONFIGURATION_FEATURE_NAMES))
    np.testing.assert_allclose(
        feature[0, -len(CONFIGURATION_EVIDENCE_FEATURE_NAMES):],
        np.arange(len(CONFIGURATION_EVIDENCE_FEATURE_NAMES)),
    )
    del diagnostics["configuration_evidence_v2"][CONFIGURATION_EVIDENCE_FEATURE_NAMES[0]]
    with np.testing.assert_raises(ValueError):
        candidate_runtime_features(row, include_configuration=True)
