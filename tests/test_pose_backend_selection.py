from __future__ import annotations

import numpy as np

from feature_extract.vfm.localization.pose_backend_selection import (
    POSE_BACKEND_GATE_FEATURE_NAMES,
    MonotonicPoseBackendPolicy,
    PoseBackendGate,
    PoseBackendSelectionExample,
    fit_pose_backend_gate,
    pose_backend_selection_metrics,
    select_monotonic_pose_backend_policy,
)


def _example(index: int) -> PoseBackendSelectionExample:
    learned_is_better = index % 2 == 0
    features = np.zeros((len(POSE_BACKEND_GATE_FEATURE_NAMES),), dtype=np.float64)
    features[0] = 2.0 if learned_is_better else -2.0
    features[1] = 1.0 if learned_is_better else -1.0
    descriptor_index = POSE_BACKEND_GATE_FEATURE_NAMES.index(
        "chosen_verification_descriptor_score_mean"
    )
    baseline_index = POSE_BACKEND_GATE_FEATURE_NAMES.index(
        "baseline_ransac_inlier_ratio"
    )
    features[descriptor_index] = 0.12 if learned_is_better else 0.02
    features[baseline_index] = 0.8
    return PoseBackendSelectionExample(
        query_id=f"query-{index:02d}",
        features=tuple(float(value) for value in features),
        learned_translation_m=0.05 if learned_is_better else 0.35,
        learned_rotation_deg=0.2 if learned_is_better else 1.0,
        baseline_translation_m=0.20 if learned_is_better else 0.10,
        baseline_rotation_deg=0.5 if learned_is_better else 0.3,
    )


def test_pose_backend_gate_uses_grouped_oof_selection() -> None:
    examples = [_example(index) for index in range(30)]
    folds = {example.query_id: index % 5 for index, example in enumerate(examples)}

    gate, summary = fit_pose_backend_gate(
        examples,
        fold_assignments=folds,
        c_values=(0.03, 0.3),
        thresholds=(0.4, 0.5, 0.6, 1.0),
    )
    features = np.asarray([example.features for example in examples])
    decisions = gate.choose_learned(features)
    metrics = pose_backend_selection_metrics(examples, decisions)
    restored = PoseBackendGate.from_dict(gate.to_dict())

    assert not summary["fallback_only_due_to_failed_oof_gate"]
    assert metrics["decision_accuracy_TARGET_ONLY"] == 1.0
    np.testing.assert_array_equal(restored.choose_learned(features), decisions)


def test_monotonic_pose_backend_policy_requires_both_development_gates() -> None:
    train = [_example(index) for index in range(30)]
    validation = [_example(index + 100) for index in range(20)]

    policy, summary = select_monotonic_pose_backend_policy(train, validation)
    restored = MonotonicPoseBackendPolicy.from_dict(policy.to_dict())
    features = np.asarray([example.features for example in validation])
    expected = np.asarray([index % 2 == 0 for index in range(20)])

    assert summary["candidate_count"] == 24
    np.testing.assert_array_equal(policy.choose_learned(features), expected)
    np.testing.assert_array_equal(restored.choose_learned(features), expected)
