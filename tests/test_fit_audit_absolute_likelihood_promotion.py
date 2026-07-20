import numpy as np

from feature_extract.tools.vfm.fit_audit_absolute_likelihood_promotion import (
    _examples_from_scores,
)


def _scores(*, hypothesis_indices, selection_scores, selected):
    count = len(hypothesis_indices)
    return {
        "query_ids": np.asarray(["query.png"] * count),
        "split_names": np.asarray(["train"] * count),
        "evaluation_labels": np.asarray(["policy"] * count),
        "hypothesis_indices": np.asarray(hypothesis_indices, dtype=np.int64),
        "independent_score_top1": np.asarray(selected, dtype=bool),
        "independent_selection_scores": np.asarray(selection_scores, dtype=np.float64),
        "independent_log_likelihood_means": np.asarray(selection_scores, dtype=np.float64),
        "independent_log_likelihood_medians": np.asarray(selection_scores, dtype=np.float64),
        "independent_log_likelihood_trimmed_means_10": np.asarray(
            selection_scores, dtype=np.float64
        ),
        "independent_log_likelihood_worst_quartile_means": np.asarray(
            selection_scores, dtype=np.float64
        ),
        "independent_log_likelihood_lcb95s": np.asarray(selection_scores, dtype=np.float64),
        "independent_spatial_median_of_means_2x2": np.asarray(
            selection_scores, dtype=np.float64
        ),
        "independent_effective_point_counts": np.asarray([8] * count, dtype=np.int64),
        "independent_evidence_coverages": np.asarray([0.5] * count, dtype=np.float64),
        "verification_point_counts": np.asarray([8] * count, dtype=np.int64),
        "candidate_spatial_materialized_verification_point_counts": np.asarray(
            [8] * count, dtype=np.int64
        ),
        "candidate_spatial_materialized_candidate_view_counts": np.asarray(
            [16] * count, dtype=np.int64
        ),
    }


def test_optional_gain_aligns_baseline_by_hypothesis_key_not_merged_row_index() -> None:
    # The baseline chooses hypothesis 1 at its merged row 1.  The optional
    # artifact deliberately has the opposite row order, so row 1 is hypothesis
    # 0 rather than the baseline hypothesis.  A direct index reuse would turn
    # the expected 5 - 3 gain into zero.
    baseline = _scores(
        hypothesis_indices=[0, 1], selection_scores=[0.0, 1.0], selected=[False, True]
    )
    optional = _scores(
        hypothesis_indices=[1, 0], selection_scores=[3.0, 5.0], selected=[False, True]
    )
    targets = {
        ("train", "query.png", "policy", 0): (0.1, 0.1),
        ("train", "query.png", "policy", 1): (0.2, 0.2),
    }

    examples = _examples_from_scores(
        baseline=baseline, optional=optional, targets=targets
    )

    assert len(examples) == 1
    assert examples[0].baseline_hypothesis_index == 1
    assert examples[0].optional_hypothesis_index == 0
    assert examples[0].features[0] == 2.0


def test_consensus_features_follow_optional_hypothesis_identity_not_row_order() -> None:
    baseline = _scores(
        hypothesis_indices=[0, 1, 2],
        selection_scores=[0.0, 3.0, 2.0],
        selected=[False, True, False],
    )
    optional = _scores(
        hypothesis_indices=[2, 0, 1],
        selection_scores=[1.0, 5.0, 4.0],
        selected=[False, True, False],
    )
    # The optional selection is hypothesis 0; the immutable baseline is
    # hypothesis 1.  Every robust profile in this fixture preserves that order.
    targets = {
        ("train", "query.png", "policy", 0): (0.1, 0.1),
        ("train", "query.png", "policy", 1): (0.2, 0.2),
        ("train", "query.png", "policy", 2): (0.3, 0.3),
    }

    example = _examples_from_scores(
        baseline=baseline, optional=optional, targets=targets
    )[0]

    # baseline rank under optional mean = 1 / (3 - 1); selected hypothesis is
    # rank one under every robust profile, and beats baseline in every profile.
    assert example.features[8] == 0.5
    assert example.features[9] == 0.0
    assert example.features[10] == 0.0
    assert example.features[11] == 1.0
    assert example.features[12] == 1.0
