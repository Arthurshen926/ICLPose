import numpy as np

from feature_extract.tools.vfm.fit_crossfit_audit_gain_promotion import (
    _examples_from_alignment,
)
from feature_extract.vfm.localization.crossfit_audit_promotion import (
    CrossfitAuditPromotionExample,
    promotion_decisions,
    select_tail_safe_threshold,
    sequence_grouped_oof_decisions,
)


def _example(
    index: int,
    *,
    delta: float,
    source_translation: float,
    optional_translation: float,
    sequence: str = "seq1",
) -> CrossfitAuditPromotionExample:
    return CrossfitAuditPromotionExample(
        query_id=f"{sequence}/frame{index:05d}.png",
        split_name="train",
        evaluation_label="policy",
        source_hypothesis_index=0,
        optional_hypothesis_index=1,
        audit_score_delta=delta,
        eligible_without_audit_gain=True,
        source_translation_m=source_translation,
        source_rotation_deg=1.0,
        optional_translation_m=optional_translation,
        optional_rotation_deg=1.0,
    )


def test_threshold_selection_rejects_a_low_gain_catastrophic_promotion() -> None:
    examples = [
        _example(0, delta=0.9, source_translation=0.50, optional_translation=0.10),
        _example(1, delta=0.8, source_translation=0.40, optional_translation=0.10),
        _example(2, delta=0.2, source_translation=0.30, optional_translation=1.50),
        _example(3, delta=0.1, source_translation=0.25, optional_translation=0.10),
    ]

    threshold, decisions, audit = select_tail_safe_threshold(
        examples, minimum_promotion_count=2
    )

    assert threshold is not None
    assert threshold >= 0.8
    assert decisions.tolist() == [True, True, False, False]
    assert audit["gate"]["passes"] is True


def test_sequence_grouped_oof_holds_entire_capture_sequences_out() -> None:
    examples = [
        _example(
            index,
            delta=0.5,
            source_translation=0.40,
            optional_translation=0.10,
            sequence=f"seq{index // 2}",
        )
        for index in range(6)
    ]

    decisions, assignments, _thresholds, audit = sequence_grouped_oof_decisions(
        examples, fold_count=3, minimum_promotion_count=2
    )

    assert decisions.tolist() == [True] * len(examples)
    assert set(assignments) == {0, 1, 2}
    for fold in audit["folds"]:
        assert not set(fold["fit_sequence_groups"]).intersection(
            fold["heldout_sequence_groups"]
        )


def test_alignment_examples_strip_only_the_audit_gain_failure() -> None:
    arrays = {
        "query_ids": np.asarray(["seq1/frame.png", "seq1/frame.png"]),
        "split_names": np.asarray(["train", "train"]),
        "evaluation_labels": np.asarray(["policy", "policy"]),
        "hypothesis_indices": np.asarray([10, 20], dtype=np.int64),
        "source_chosen": np.asarray([True, False]),
        "optional_rank_top1": np.asarray([False, True]),
        "optional_differs_from_source": np.asarray([True, True]),
        "audit_score_deltas": np.asarray([0.25, 0.25]),
        "promotion_failures": np.asarray(
            ["audit_likelihood_gain", "audit_likelihood_gain"]
        ),
    }
    targets = {
        ("train", "seq1/frame.png", "policy", 10): (0.4, 1.0, 0.4, 1.0),
        ("train", "seq1/frame.png", "policy", 20): (0.4, 1.0, 0.1, 1.0),
    }

    examples = _examples_from_alignment(arrays=arrays, targets=targets)

    assert len(examples) == 1
    assert examples[0].eligible_without_audit_gain is True
    assert examples[0].optional_translation_m == 0.1
    assert promotion_decisions(examples, minimum_audit_gain=0.25).tolist() == [True]
