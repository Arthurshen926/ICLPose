from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.local_assignment_linear import (
    IDENTITY_STRATEGIES,
    NO_MATCH_STRATEGIES,
    LinearLogitModel,
    build_identity_candidate_features,
    build_no_match_features,
    resolve_rescue_policy_scores,
    selective_baseline_gain_switch_scores,
    selective_switch_scores,
)


def _payload() -> dict[str, np.ndarray]:
    payload = {
        "candidate_track_ids": np.asarray([[10, 11, 12], [20, 21, -1]], dtype=np.int64),
        "candidate_prototype_ids": np.asarray([[0, 1, 0], [1, 0, -1]], dtype=np.int64),
        "maplet_support_counts": np.asarray([[2, 3, 4], [5, 6, 0]], dtype=np.int64),
        "all_support_counts": np.asarray([[7, 8, 9], [10, 11, 0]], dtype=np.int64),
    }
    for index, name in enumerate(dict.fromkeys((*IDENTITY_STRATEGIES, *NO_MATCH_STRATEGIES))):
        payload[f"strategy__{name}"] = np.asarray(
            [[0.9, 0.7, 0.2], [0.6, 0.5, -np.inf]], dtype=np.float32
        ) - 0.01 * index
    return payload


def test_identity_feature_schema_and_invalid_candidate_handling() -> None:
    features, names = build_identity_candidate_features(_payload())
    assert features.shape == (2, 3, 2 * len(IDENTITY_STRATEGIES) + 4)
    assert names[0:2] == ("coarse_prototype", "coarse_prototype_gap_to_row_max")
    np.testing.assert_allclose(features[0, :, 1], [0.0, 0.2, 0.7], atol=1e-6)
    assert np.all(features[1, 2] == 0.0)


def test_no_match_features_are_row_level_and_finite() -> None:
    features, names = build_no_match_features(_payload(), np.asarray([0.8, 1e-9]))
    assert features.shape == (2, 4 * len(NO_MATCH_STRATEGIES) + 2)
    assert names[-2:] == ("query_detector_score", "query_detector_log10_score")
    assert np.all(np.isfinite(features))


def test_linear_model_round_trip(tmp_path) -> None:
    model = LinearLogitModel(
        coefficients=np.asarray([2.0, -1.0]),
        intercept=0.5,
        feature_names=("a", "b"),
        metadata={"probe_sha256": "abc"},
    )
    path = tmp_path / "model.npz"
    model.save(path)
    restored = LinearLogitModel.load(path)
    values = np.asarray([[1.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(restored.decision_function(values), [-0.5])
    assert restored.metadata["probe_sha256"] == "abc"


def test_selective_switch_preserves_baseline_confidence() -> None:
    reranker = np.asarray([[3.0, 1.0, 0.0], [1.0, 1.5, 1.4]], dtype=np.float32)
    baseline = np.asarray([[0.2, 0.8, 0.1], [0.9, 0.8, 0.7]], dtype=np.float32)
    selected, resolved, switched, margin = selective_switch_scores(
        reranker,
        baseline,
        margin_threshold=0.2,
    )
    np.testing.assert_array_equal(selected, [0, 0])
    np.testing.assert_array_equal(switched, [True, False])
    np.testing.assert_allclose(resolved[0], [0.2, -np.inf, -np.inf])
    np.testing.assert_allclose(margin, [2.0, 0.1], atol=1e-6)


def test_selective_switch_ignores_rows_without_reranker_candidates() -> None:
    selected, resolved, switched, margin = selective_switch_scores(
        np.full((1, 2), -np.inf, dtype=np.float32),
        np.asarray([[0.7, 0.6]], dtype=np.float32),
        margin_threshold=0.0,
    )
    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_array_equal(switched, [False])
    assert margin[0] == -np.inf
    np.testing.assert_allclose(resolved, [[0.7, -np.inf]])


def test_selective_switch_can_decouple_identity_from_row_confidence() -> None:
    selected, resolved, switched, _margin = selective_switch_scores(
        np.asarray([[3.0, 1.0]], dtype=np.float32),
        np.asarray([[0.2, 0.8]], dtype=np.float32),
        margin_threshold=0.2,
        preserve_baseline_row_confidence=True,
    )
    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_array_equal(switched, [True])
    np.testing.assert_allclose(resolved, [[0.8, -np.inf]])


def test_baseline_gain_switch_compares_against_the_actual_baseline_candidate() -> None:
    selected, resolved, switched, gain = selective_baseline_gain_switch_scores(
        np.asarray([[0.9, 0.2, 0.85]], dtype=np.float32),
        np.asarray([[0.2, 0.8, 0.1]], dtype=np.float32),
        min_gain=0.5,
        preserve_baseline_row_confidence=True,
    )
    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_array_equal(switched, [True])
    np.testing.assert_allclose(gain, [0.7], atol=1e-6)
    np.testing.assert_allclose(resolved, [[0.8, -np.inf, -np.inf]])


def test_baseline_validity_gate_protects_confident_coarse_matches() -> None:
    reranker = np.asarray([[0.9, 0.2], [0.9, 0.2]], dtype=np.float32)
    baseline = np.asarray([[0.2, 0.8], [0.2, 0.8]], dtype=np.float32)
    baseline_validity = np.asarray([[0.1, 0.9], [0.1, 0.3]], dtype=np.float32)
    selected, _resolved, switched, _gain = selective_baseline_gain_switch_scores(
        reranker,
        baseline,
        min_gain=0.5,
        baseline_validity_scores=baseline_validity,
        max_baseline_validity=0.5,
    )
    np.testing.assert_array_equal(selected, [1, 0])
    np.testing.assert_array_equal(switched, [False, True])


def test_rescue_policy_resolves_after_probability_fusion() -> None:
    candidate = np.asarray(
        [[0.40, 0.35, 0.05], [0.10, 0.45, 0.20]], dtype=np.float32
    )
    keep = np.asarray([0.20, 0.25], dtype=np.float32)
    baseline = np.asarray([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], dtype=np.float32)
    selected, resolved, switched, margin, action = resolve_rescue_policy_scores(
        candidate,
        keep,
        baseline,
        action_margin_threshold=0.05,
    )
    np.testing.assert_array_equal(selected, [1, 1])
    np.testing.assert_array_equal(switched, [True, True])
    np.testing.assert_allclose(margin, [0.15, 0.20], atol=1e-6)
    np.testing.assert_allclose(resolved[:, 1], [0.9, 0.9], atol=1e-6)
    np.testing.assert_allclose(action[:, 1], [0.35, 0.45], atol=1e-6)


def test_rescue_policy_never_treats_baseline_as_a_rescue_action() -> None:
    selected, _resolved, switched, margin, _action = resolve_rescue_policy_scores(
        np.asarray([[0.8, 0.12, 0.05]], dtype=np.float32),
        np.asarray([0.05], dtype=np.float32),
        np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32),
    )
    np.testing.assert_array_equal(selected, [1])
    np.testing.assert_array_equal(switched, [True])
    np.testing.assert_allclose(margin, [0.07], atol=1e-6)

    selected, _resolved, switched, margin, _action = resolve_rescue_policy_scores(
        np.asarray([[0.8, 0.04, 0.03]], dtype=np.float32),
        np.asarray([0.05], dtype=np.float32),
        np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32),
    )
    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_array_equal(switched, [False])
    np.testing.assert_allclose(margin, [-0.01], atol=1e-6)


def test_pose_safe_rescue_keep_preserves_full_baseline_row() -> None:
    baseline = np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32)
    selected, resolved, switched, _margin, _action = resolve_rescue_policy_scores(
        np.asarray([[0.8, 0.04, 0.03]], dtype=np.float32),
        np.asarray([0.05], dtype=np.float32),
        baseline,
        preserve_baseline_alternatives=True,
    )

    np.testing.assert_array_equal(selected, [0])
    np.testing.assert_array_equal(switched, [False])
    np.testing.assert_array_equal(resolved, baseline)


def test_pose_safe_rescue_update_swaps_ranks_without_dropping_alternatives() -> None:
    selected, resolved, switched, _margin, _action = resolve_rescue_policy_scores(
        np.asarray([[0.8, 0.12, 0.05]], dtype=np.float32),
        np.asarray([0.05], dtype=np.float32),
        np.asarray([[0.9, 0.8, 0.7]], dtype=np.float32),
        preserve_baseline_alternatives=True,
    )

    np.testing.assert_array_equal(selected, [1])
    np.testing.assert_array_equal(switched, [True])
    np.testing.assert_allclose(resolved, [[0.8, 0.9, 0.7]], atol=1e-6)


def test_pose_safe_rescue_can_lock_only_updated_rows() -> None:
    selected, resolved, switched, _margin, _action = resolve_rescue_policy_scores(
        np.asarray([[0.8, 0.12, 0.05], [0.8, 0.04, 0.03]], dtype=np.float32),
        np.asarray([0.05, 0.05], dtype=np.float32),
        np.asarray([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], dtype=np.float32),
        preserve_baseline_alternatives=True,
        lock_rescue_updates=True,
    )

    np.testing.assert_array_equal(selected, [1, 0])
    np.testing.assert_array_equal(switched, [True, False])
    np.testing.assert_allclose(resolved[0], [-np.inf, 0.9, -np.inf])
    np.testing.assert_allclose(resolved[1], [0.9, 0.8, 0.7])
