import numpy as np
import json

from feature_extract.vfm.localization_goal_maplet.phase_preserving_readout import (
    directional_phase_statistics,
    dual_band_phase_evidence,
    jacobian_phase_statistics,
    load_phase_readout_policy,
    orientation_equivariant_phase_evidence,
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


def test_phase_policy_v2_rejects_identity_evidence(tmp_path):
    path = tmp_path / "phase.json"
    path.write_text(json.dumps({
        "artifact_type": "goal_maplet_phase_readout_policy_v2",
        "role": "phase_residual_only",
        "component_names": ["context_cosine"],
        "standardizer_mean": [0.0],
        "standardizer_scale": [1.0],
        "coefficient": [1.0],
    }))
    try:
        load_phase_readout_policy(path)
    except ValueError as error:
        assert "identity and observability" in str(error)
    else:
        raise AssertionError("phase policy v2 must reject identity evidence")


def test_phase_policy_v2_accepts_only_jacobian_conditional_phase(tmp_path):
    path = tmp_path / "phase.json"
    path.write_text(json.dumps({
        "artifact_type": "goal_maplet_phase_readout_policy_v2",
        "role": "phase_residual_only",
        "component_names": ["jacobian_phase_visible"],
        "standardizer_mean": [0.0],
        "standardizer_scale": [1.0],
        "coefficient": [1.0],
    }))
    policy = load_phase_readout_policy(path)
    query = _feature([[0.0, 0.2, 0.8], [0.1, 0.4, 1.0], [0.0, 0.3, 0.8]])
    evidence = dual_band_phase_evidence(
        query, query, query, query, np.ones(query.shape[1:], dtype=bool),
    )
    assert np.isclose(policy.score(evidence), evidence.jacobian_phase_visible)


def test_phase_policy_v2_rejects_observability_term(tmp_path):
    path = tmp_path / "phase.json"
    path.write_text(json.dumps({
        "artifact_type": "goal_maplet_phase_readout_policy_v2",
        "role": "phase_residual_only",
        "component_names": ["jacobian_observability"],
        "standardizer_mean": [0.0],
        "standardizer_scale": [1.0],
        "coefficient": [1.0],
    }))
    try:
        load_phase_readout_policy(path)
    except ValueError as error:
        assert "identity and observability" in str(error)
    else:
        raise AssertionError("phase policy must not hide observability in its residual")


def test_phase_policy_v2_rejects_negative_phase_coefficient(tmp_path):
    path = tmp_path / "phase.json"
    path.write_text(json.dumps({
        "artifact_type": "goal_maplet_phase_readout_policy_v2",
        "role": "phase_residual_only",
        "component_names": ["jacobian_phase_visible"],
        "standardizer_mean": [0.0],
        "standardizer_scale": [1.0],
        "coefficient": [-1.0],
    }))
    try:
        load_phase_readout_policy(path)
    except ValueError as error:
        assert "monotonic" in str(error)
    else:
        raise AssertionError("phase policy must reward agreement")


def test_jacobian_phase_is_common_quarter_turn_invariant():
    generator = np.random.default_rng(11)
    query = generator.normal(size=(6, 7, 7)).astype(np.float32)
    rendered = (0.8 * query + 0.2 * generator.normal(size=query.shape)).astype(np.float32)
    fraction = generator.uniform(0.2, 1.0, size=(7, 7)).astype(np.float32)
    base = jacobian_phase_statistics(
        query, rendered, feature_fraction=fraction,
    )
    rotated_query = np.rot90(query, axes=(1, 2)).copy()
    rotated_rendered = np.rot90(rendered, axes=(1, 2)).copy()
    rotated_fraction = np.rot90(fraction).copy()
    turned = jacobian_phase_statistics(
        rotated_query, rotated_rendered, feature_fraction=rotated_fraction,
    )
    assert np.isclose(base.conditional_score, turned.conditional_score, atol=1e-6)
    assert np.isclose(base.observability, turned.observability, atol=1e-6)
    assert np.isclose(base.fixed_grid_score, turned.fixed_grid_score, atol=1e-6)
    assert np.isclose(
        base.log_scale_agreement, turned.log_scale_agreement, atol=1e-6,
    )


def test_jacobian_phase_separates_fractional_observability():
    feature = _feature([
        [0.0, 0.2, 0.8, 1.0],
        [0.0, 0.2, 0.8, 1.0],
        [0.0, 0.2, 0.8, 1.0],
        [0.0, 0.2, 0.8, 1.0],
    ])
    full = jacobian_phase_statistics(feature, feature)
    quarter = jacobian_phase_statistics(
        feature, feature, feature_fraction=np.full((4, 4), 0.25, np.float32),
    )
    assert np.isclose(full.conditional_score, quarter.conditional_score)
    assert np.isclose(quarter.observability, 0.25 * full.observability)
    assert np.isclose(
        quarter.fixed_grid_score,
        quarter.conditional_score * quarter.observability,
    )
    assert np.isclose(full.log_scale_agreement, 0.0, atol=1e-7)


def test_minimal_g20_path_matches_dual_band_jacobian_without_identity_terms():
    query = _feature([
        [0.0, 0.2, 0.8, 1.0],
        [0.1, 0.3, 0.7, 0.9],
        [0.0, 0.2, 0.8, 1.0],
    ])
    valid = np.ones(query.shape[1:], dtype=bool)
    legacy = dual_band_phase_evidence(query, query, query, query, valid)
    minimal = orientation_equivariant_phase_evidence(query, query, valid)
    assert np.isclose(minimal.jacobian_phase_visible, legacy.jacobian_phase_visible)
    assert np.isclose(minimal.jacobian_observability, legacy.jacobian_observability)
    assert np.isclose(
        minimal.jacobian_log_scale_agreement,
        legacy.jacobian_log_scale_agreement,
    )
    assert minimal.mapper_cosine == 0.0
    assert minimal.context_cosine == 0.0
    assert minimal.horizontal_phase == 0.0
