from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_candidate_pose_latent_evidence import (
    validate_score_target_coverage,
    paired_best_10cm_rank,
    primary_gate,
    validate_ragged_score_layout,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "query_ids": np.asarray(["validation/a.png", "validation/a.png"]),
        "split_names": np.asarray(["validation", "validation"]),
        "evaluation_labels": np.asarray(["fixed", "fixed"]),
        "hypothesis_indices": np.asarray([3, 7]),
        "source_chosen_for_optional_pose": np.asarray([False, True]),
        "baseline_score_top1": np.asarray([False, True]),
        "baseline_selection_scores": np.asarray([0.1, 0.2]),
        "visual_pose_log_likelihood_ratios": np.asarray([0.5, 0.75]),
        "control_pose_log_likelihood_ratios": np.asarray([0.3, 0.2]),
        "verification_query_ids": np.asarray(["validation/a.png"]),
        "verification_split_names": np.asarray(["validation"]),
        "verification_offsets": np.asarray([0, 2]),
        "hypothesis_verification_offsets": np.asarray([0, 2, 4]),
        "visual_point_log_likelihood_ratios": np.asarray([0.1, 0.2, 0.3, 0.4]),
        "control_point_log_likelihood_ratios": np.asarray([0.0, 0.2, 0.1, 0.4]),
        "visual_point_geometric_candidate_counts": np.asarray([1, 2, 1, 2]),
        "control_point_geometric_candidate_counts": np.asarray([1, 2, 1, 2]),
        "visual_point_geometric_view_masses": np.asarray([0.5, 1.0, 0.5, 1.0]),
        "control_point_geometric_view_masses": np.asarray([0.5, 1.0, 0.5, 1.0]),
        "visual_identity_edge_usable": np.asarray([[[True]], [[False]]]),
        "control_identity_edge_usable": np.asarray([[[True]], [[False]]]),
    }


def test_ragged_score_layout_requires_visual_control_geometry_equality() -> None:
    arrays = _arrays()
    validate_ragged_score_layout(arrays)

    arrays["control_point_geometric_view_masses"] = np.asarray([0.5, 0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="geometry"):
        validate_ragged_score_layout(arrays)


def test_ragged_score_layout_requires_each_hypothesis_to_span_its_query_points() -> None:
    arrays = _arrays()
    arrays["hypothesis_verification_offsets"] = np.asarray([0, 2, 3])

    with pytest.raises(ValueError, match="ragged"):
        validate_ragged_score_layout(arrays)


def test_paired_rank_and_primary_gate_require_rank_and_tail_improvement() -> None:
    visual_rows = [
        {"split_name": "validation", "evaluation_label": "fixed", "query_id": "a", "best_10cm_rank": 4},
        {"split_name": "validation", "evaluation_label": "fixed", "query_id": "b", "best_10cm_rank": 10},
        {"split_name": "validation", "evaluation_label": "fixed", "query_id": "c", "best_10cm_rank": 22},
    ]
    control_rows = [
        {"split_name": "validation", "evaluation_label": "fixed", "query_id": "a", "best_10cm_rank": 8},
        {"split_name": "validation", "evaluation_label": "fixed", "query_id": "b", "best_10cm_rank": 5},
        {"split_name": "validation", "evaluation_label": "fixed", "query_id": "c", "best_10cm_rank": 30},
    ]
    paired = paired_best_10cm_rank(visual_rows, control_rows)

    assert paired["best_10cm_rank_wins"] == 2
    assert paired["best_10cm_rank_losses"] == 1
    gate = primary_gate(
        visual={
            "median_best_10cm_rank": 10.0,
            "p90_best_10cm_rank": 20.0,
            "p90_selected_translation_cm": 25.0,
            "catastrophic_1m_count": 0,
        },
        control={
            "median_best_10cm_rank": 25.0,
            "p90_best_10cm_rank": 30.0,
            "p90_selected_translation_cm": 35.0,
            "catastrophic_1m_count": 1,
        },
        paired_rank=paired,
    )

    assert all(gate.values())


def test_primary_gate_rejects_missing_correct_pose_and_tail_regression() -> None:
    gate = primary_gate(
        visual={
            "median_best_10cm_rank": None,
            "p90_best_10cm_rank": None,
            "p90_selected_translation_cm": 105.0,
            "catastrophic_1m_count": 2,
        },
        control={
            "median_best_10cm_rank": 10.0,
            "p90_best_10cm_rank": 20.0,
            "p90_selected_translation_cm": 30.0,
            "catastrophic_1m_count": 0,
        },
        paired_rank={
            "best_10cm_rank_wins": 0,
            "best_10cm_rank_losses": 1,
        },
    )

    assert not all(gate.values())


def test_score_target_coverage_allows_prefix_only_when_explicitly_requested() -> None:
    keys = (("validation", "fixed", "a", 1),)
    target_keys = (
        ("validation", "fixed", "a", 1),
        ("validation", "fixed", "a", 2),
    )
    metadata = [{"score_splits": ["validation"]}]

    validate_score_target_coverage(
        keys=keys,
        target_keys=target_keys,
        score_metadata=metadata,
        require_complete_scope=False,
    )
    with pytest.raises(ValueError, match="complete frozen target scope"):
        validate_score_target_coverage(
            keys=keys,
            target_keys=target_keys,
            score_metadata=metadata,
            require_complete_scope=True,
        )
