import numpy as np
import json

from feature_extract.vfm.localization_goal_maplet.phase_preserving_readout import (
    directional_phase_statistics,
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
    assert correct_score.legacy_dual_band_score > shifted_score.legacy_dual_band_score
    assert correct_score.horizontal_phase > shifted_score.horizontal_phase
    assert not hasattr(correct_score, "score")


def test_dual_band_uses_fixed_denominator_for_missing_support():
    query = _feature([[0.0, 0.2, 0.8, 1.0], [0.0, 0.2, 0.8, 1.0]])
    full = np.ones(query.shape[1:], dtype=bool)
    sparse = np.zeros(query.shape[1:], dtype=bool)
    sparse[:, :2] = True
    full_score = dual_band_phase_evidence(query, query, query, query, full)
    sparse_score = dual_band_phase_evidence(query, query, query, query, sparse)
    assert full_score.mapper_cosine > sparse_score.mapper_cosine
    assert full_score.legacy_dual_band_score > sparse_score.legacy_dual_band_score
    assert np.isclose(
        full_score.horizontal_phase,
        full_score.horizontal_phase_visible * full_score.horizontal_observability,
    )
    assert np.isclose(
        sparse_score.horizontal_phase,
        sparse_score.horizontal_phase_visible * sparse_score.horizontal_observability,
    )
    assert np.isclose(
        full_score.horizontal_phase_visible,
        sparse_score.horizontal_phase_visible,
    )
    assert full_score.horizontal_observability > sparse_score.horizontal_observability


def test_dual_band_rejects_non_spatial_feature():
    feature = np.ones((4, 8), dtype=np.float32)
    try:
        dual_band_phase_evidence(feature, feature, feature, feature, np.ones((8,), bool))
    except ValueError as error:
        assert "[C,H,W]" in str(error)
    else:
        raise AssertionError("non-spatial feature must fail closed")


def test_directional_phase_region_keeps_conditional_semantics():
    query = _feature([[0.0, 0.2, 0.8, 1.0], [0.0, 0.2, 0.8, 1.0]])
    valid = np.ones(query.shape[1:], dtype=bool)
    left = np.zeros_like(valid)
    left[:, :2] = True
    statistics = directional_phase_statistics(
        query, query, valid, delta_y=0, delta_x=1, edge_selector=left,
    )
    assert statistics.grid_edge_count == 2
    assert statistics.informative_edge_count <= statistics.grid_edge_count
    assert np.isclose(
        statistics.fixed_grid_score,
        statistics.conditional_score * statistics.observability,
    )


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


def test_phase_readout_policy_rejects_removed_implicit_mixture(tmp_path):
    path = tmp_path / "phase.json"
    path.write_text(json.dumps({
        "artifact_type": "goal_maplet_phase_readout_policy_v1",
        "component_names": ["dual_band_score"],
        "standardizer_mean": [0.0],
        "standardizer_scale": [1.0],
        "coefficient": [1.0],
    }))
    try:
        load_phase_readout_policy(path)
    except ValueError as error:
        assert "unknown component" in str(error)
    else:
        raise AssertionError("removed implicit dual-band mixture must fail closed")
