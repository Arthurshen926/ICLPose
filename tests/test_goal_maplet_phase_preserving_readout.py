import numpy as np
import json

from feature_extract.vfm.localization_goal_maplet.phase_preserving_readout import (
    dual_band_phase_evidence,
    load_phase_readout_policy,
)


def _feature(values):
    array = np.asarray(values, dtype=np.float32)[None]
    return np.concatenate([array, 1.0 - array], axis=0)


def test_dual_band_prefers_correct_spatial_phase():
    query = _feature([[0.0, 0.2, 0.8, 1.0], [0.0, 0.2, 0.8, 1.0]])
    correct = query.copy()
    shifted = np.roll(query, 1, axis=2)
    valid = np.ones(query.shape[1:], dtype=bool)
    correct_score = dual_band_phase_evidence(query, correct, query, correct, valid)
    shifted_score = dual_band_phase_evidence(query, shifted, query, shifted, valid)
    assert correct_score.score > shifted_score.score
    assert correct_score.horizontal_phase > shifted_score.horizontal_phase


def test_dual_band_uses_fixed_denominator_for_missing_support():
    query = _feature([[0.0, 0.2, 0.8, 1.0], [0.0, 0.2, 0.8, 1.0]])
    full = np.ones(query.shape[1:], dtype=bool)
    sparse = np.zeros(query.shape[1:], dtype=bool)
    sparse[:, :2] = True
    full_score = dual_band_phase_evidence(query, query, query, query, full)
    sparse_score = dual_band_phase_evidence(query, query, query, query, sparse)
    assert full_score.mapper_cosine > sparse_score.mapper_cosine
    assert full_score.score > sparse_score.score


def test_dual_band_rejects_non_spatial_feature():
    feature = np.ones((4, 8), dtype=np.float32)
    try:
        dual_band_phase_evidence(feature, feature, feature, feature, np.ones((8,), bool))
    except ValueError as error:
        assert "[C,H,W]" in str(error)
    else:
        raise AssertionError("non-spatial feature must fail closed")


def test_phase_readout_policy_scores_only_declared_components(tmp_path):
    path = tmp_path / "phase.json"
    path.write_text(json.dumps({
        "artifact_type": "goal_maplet_phase_readout_policy_v1",
        "component_names": ["horizontal_phase", "vertical_phase"],
        "standardizer_mean": [0.0, 0.0],
        "standardizer_scale": [1.0, 2.0],
        "coefficient": [2.0, 4.0],
    }))
    policy = load_phase_readout_policy(path)
    query = _feature([[0.0, 0.2, 0.8, 1.0], [0.0, 0.2, 0.8, 1.0]])
    evidence = dual_band_phase_evidence(
        query, query, query, query, np.ones(query.shape[1:], dtype=bool),
    )
    expected = 2.0 * evidence.horizontal_phase + 2.0 * evidence.vertical_phase
    assert np.isclose(policy.score(evidence), expected)
