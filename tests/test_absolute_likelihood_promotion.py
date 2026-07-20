import numpy as np

from feature_extract.vfm.localization.absolute_likelihood_promotion import (
    FEATURE_NAMES,
    NO_PROMOTION_THRESHOLD,
    AbsoluteLikelihoodPromotionExample,
    crossfit_probabilities,
    fit_final_model,
    policy_metrics,
    promotion_decisions,
    select_oof_tail_safe_threshold,
    tail_safe_gate,
)


def _example(
    index: int,
    *,
    baseline_translation: float,
    optional_translation: float,
    baseline_rotation: float = 1.0,
    optional_rotation: float = 1.0,
    changed: bool = True,
) -> AbsoluteLikelihoodPromotionExample:
    return AbsoluteLikelihoodPromotionExample(
        query_id=f"q{index}",
        split_name="train",
        evaluation_label="policy",
        baseline_hypothesis_index=0,
        optional_hypothesis_index=1 if changed else 0,
        features=tuple(float(index + offset) for offset in range(len(FEATURE_NAMES))),
        baseline_translation_m=baseline_translation,
        baseline_rotation_deg=baseline_rotation,
        optional_translation_m=optional_translation,
        optional_rotation_deg=optional_rotation,
    )


def test_beneficial_label_requires_a_real_safe_change() -> None:
    helpful = _example(0, baseline_translation=0.30, optional_translation=0.20)
    harmful = _example(1, baseline_translation=0.20, optional_translation=0.50)
    unchanged = _example(
        2, baseline_translation=0.30, optional_translation=0.10, changed=False
    )

    assert helpful.target_beneficial is True
    assert harmful.target_beneficial is False
    assert unchanged.target_beneficial is False


def test_decision_never_promotes_an_identical_hypothesis() -> None:
    examples = [
        _example(0, baseline_translation=0.30, optional_translation=0.10),
        _example(1, baseline_translation=0.30, optional_translation=0.10, changed=False),
    ]

    decisions = promotion_decisions(examples, [0.99, 0.99], threshold=0.5)

    assert decisions.tolist() == [True, False]


def test_tail_gate_rejects_new_catastrophic_promotion() -> None:
    examples = [
        _example(0, baseline_translation=0.40, optional_translation=0.10),
        _example(1, baseline_translation=0.30, optional_translation=1.50),
        _example(2, baseline_translation=0.20, optional_translation=0.10),
    ]

    metrics = policy_metrics(examples, [True, True, False])
    gate = tail_safe_gate(metrics)

    assert metrics["promotion"]["new_catastrophic_count"] == 1
    assert gate["checks"]["no_new_catastrophic_promotion"] is False
    assert gate["passes"] is False


def test_oof_threshold_selection_prefers_the_tail_safe_subset() -> None:
    examples = [
        _example(0, baseline_translation=0.50, optional_translation=0.10),
        _example(1, baseline_translation=0.40, optional_translation=0.10),
        _example(2, baseline_translation=0.30, optional_translation=1.50),
        _example(3, baseline_translation=0.20, optional_translation=0.10),
    ]

    threshold, decisions, audit = select_oof_tail_safe_threshold(
        examples, [0.90, 0.80, 0.30, 0.20]
    )

    assert threshold < NO_PROMOTION_THRESHOLD
    assert decisions.tolist() == [True, True, False, False]
    assert audit["gate"]["passes"] is True
    assert audit["metrics"]["promotion"]["new_catastrophic_count"] == 0


def test_crossfit_and_final_model_keep_all_predictions_finite() -> None:
    examples = [
        _example(
            index,
            baseline_translation=0.40,
            optional_translation=0.10 if index % 2 == 0 else 0.70,
        )
        for index in range(10)
    ]

    probabilities, folds = crossfit_probabilities(examples, fold_count=5, c_value=0.1)
    model = fit_final_model(examples, c_value=0.1)

    assert len(folds) == len(examples)
    assert set(folds) == {0, 1, 2, 3, 4}
    assert np.all(np.isfinite(probabilities))
    assert model is not None
    final_probabilities = model.probabilities(
        np.asarray([example.features for example in examples], dtype=np.float64)
    )
    assert np.all(np.isfinite(final_probabilities))
