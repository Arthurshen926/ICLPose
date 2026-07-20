"""Unit coverage for paired candidate-rank auditing."""

from __future__ import annotations

import numpy as np
import pytest

from feature_extract.tools.vfm.audit_independent_rgb_candidate_predictions import (
    _coarse_prior_hard_negative_summary,
    _paired_coarse_prior_hard_negative_mass,
    _paired_identity_rank,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_training import (
    IDENTITY_TARGET_APPEARANCE,
    IDENTITY_TARGET_GEOMETRIC,
    _initialization_provenance,
    _identity_training_masks,
    _resolve_identity_target,
)


def test_paired_identity_rank_counts_rescues_harms_and_rank_changes() -> None:
    labels = np.asarray(
        [
            [False, True, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    valid = np.ones_like(labels, dtype=bool)
    valid[2] = False
    baseline = np.asarray(
        [
            [0.9, 0.5, 0.1],
            [0.9, 0.5, 0.1],
            [0.9, 0.5, 0.1],
        ],
        dtype=np.float32,
    )
    probe = np.asarray(
        [
            [0.1, 0.9, 0.5],
            [0.5, 0.9, 0.1],
            [0.1, 0.5, 0.9],
        ],
        dtype=np.float32,
    )

    result = _paired_identity_rank(
        baseline_scores=baseline,
        probe_scores=probe,
        labels=labels,
        valid=valid,
    )

    assert result == {
        "positive_group_count": 2,
        "rank_win_count": 1,
        "rank_loss_count": 1,
        "rank_tie_count": 0,
        "top1_rescue_count": 1,
        "top1_harm_count": 1,
        "median_rank_delta_baseline_minus_probe": 0.0,
    }


def test_geometric_identity_target_is_distinct_from_exact_observation_target() -> None:
    valid = np.asarray([[True, True, True]])
    actual_observation = np.asarray([[True, False, False]])
    center_residual = np.asarray([[0.5, np.inf, np.inf]], dtype=np.float32)
    projection_residual = np.asarray([[8.0, 1.0, 7.0]], dtype=np.float32)

    appearance, appearance_supervised = _identity_training_masks(
        valid=valid,
        actual_query_observation=actual_observation,
        actual_center_residuals=center_residual,
        target_projection_residuals=projection_residual,
        identity_target=IDENTITY_TARGET_APPEARANCE,
        positive_threshold_px=2.0,
        negative_threshold_px=5.0,
    )
    geometric, geometric_supervised = _identity_training_masks(
        valid=valid,
        actual_query_observation=actual_observation,
        actual_center_residuals=center_residual,
        target_projection_residuals=projection_residual,
        identity_target=IDENTITY_TARGET_GEOMETRIC,
        positive_threshold_px=2.0,
        negative_threshold_px=5.0,
    )

    assert appearance.tolist() == [[True, False, False]]
    assert geometric.tolist() == [[False, True, False]]
    assert appearance_supervised.tolist() == [[True, False, True]]
    assert geometric_supervised.tolist() == [[True, True, True]]


def test_identity_target_inherits_checkpoint_lineage_for_evaluation() -> None:
    assert _resolve_identity_target(
        None, initialization_training={"identity_target": IDENTITY_TARGET_GEOMETRIC}
    ) == IDENTITY_TARGET_GEOMETRIC


def test_initialization_provenance_preserves_source_training(tmp_path) -> None:
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"source-checkpoint")

    provenance = _initialization_provenance(
        checkpoint,
        {
            "format": "independent_rgb_candidate_verifier_v8",
            "training": {
                "identity_target": IDENTITY_TARGET_GEOMETRIC,
                "coarse_hard_negative_loss_weight": 8.0,
            },
        },
    )

    assert provenance["initialization_checkpoint"] == str(checkpoint)
    assert provenance["initialization_checkpoint_format"] == (
        "independent_rgb_candidate_verifier_v8"
    )
    assert provenance["initialization_checkpoint_sha256"]
    assert provenance["initialization_training"] == {
        "identity_target": IDENTITY_TARGET_GEOMETRIC,
        "coarse_hard_negative_loss_weight": 8.0,
    }


def test_coarse_prior_hard_negative_audit_uses_deployed_group_mass() -> None:
    labels = np.asarray(
        [
            [True, False, False],
            [True, False, False],
            [False, False, False],
        ]
    )
    supervised = np.asarray(
        [
            [True, True, True],
            [True, True, False],
            [False, False, False],
        ]
    )
    prior = np.asarray(
        [
            [0.2, 0.7, 0.1],
            [0.7, 0.3, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    baseline = np.asarray(
        [
            [-1.0, -0.2, -2.0],
            [-0.2, -1.0, -3.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    probe = np.asarray(
        [
            [-0.1, -2.0, -3.0],
            [-0.5, -0.4, -3.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )

    summary = _coarse_prior_hard_negative_summary(
        baseline,
        labels=labels,
        supervision_valid=supervised,
        prior=prior,
    )
    paired = _paired_coarse_prior_hard_negative_mass(
        baseline_scores=baseline,
        probe_scores=probe,
        labels=labels,
        supervision_valid=supervised,
        prior=prior,
    )

    assert summary["active_group_count"] == 2
    assert summary["coarse_top_hard_negative_group_count"] == 1
    assert summary["positive_over_negative_count"] == 1
    assert summary["coarse_top_hard_negative_positive_over_negative_fraction"] == 0.0
    assert paired["active_group_count"] == 2
    assert paired["margin_win_count"] == 1
    assert paired["margin_loss_count"] == 1
    assert paired["margin_tie_count"] == 0
    assert paired["positive_over_negative_rescue_count"] == 1
    assert paired["positive_over_negative_harm_count"] == 1
    assert paired["mean_margin_delta_probe_minus_baseline"] == pytest.approx(
        0.8198579639504744
    )
    assert paired["coarse_top_hard_negative_margin_win_count"] == 1
    assert paired["coarse_top_hard_negative_margin_loss_count"] == 0
