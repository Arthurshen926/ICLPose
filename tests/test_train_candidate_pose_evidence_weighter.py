from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from feature_extract.tools.vfm.train_candidate_pose_evidence_weighter import (
    PreparedEvidenceWeighterQuery,
    _objective,
    _query_terms,
    aggregate_weighted_point_pose_scores,
    evaluate_prepared_queries,
    inner_gate_decision,
)
from feature_extract.vfm.localization.candidate_pose_evidence_weighter import (
    TargetFreePoseEvidenceWeighter,
    TargetFreePoseEvidenceFeatures,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        pose_margin=0.20,
        soft_hard_temperature=0.35,
        rgb_appearance_control_margin=0.02,
        minimum_effective_sample_size=2.0,
        coverage_grid_rows=2,
        coverage_grid_columns=2,
        soft_hard_loss_weight=0.50,
        rgb_appearance_control_loss_weight=0.25,
        coverage_loss_weight=0.02,
        effective_sample_size_loss_weight=0.10,
        uniform_fallback_loss_weight=0.02,
        uniform_regret_loss_weight=10.0,
        maximum_train_uniform_gap_degradation=0.0,
        uniform_win_safety_loss_weight=10.0,
        minimum_train_uniform_win_gap=0.0,
        minimum_mean_gap_improvement=0.002,
        minimum_median_gap_improvement=0.001,
        maximum_p10_gap_degradation=0.002,
        maximum_minimum_gap_degradation=0.010,
        minimum_rgb_visual_gap_improvement=0.002,
    )


def _prepared() -> PreparedEvidenceWeighterQuery:
    return PreparedEvidenceWeighterQuery(
        query_id="train/q0.png",
        features=TargetFreePoseEvidenceFeatures(
            values=torch.tensor(
                [
                    [0.8, 0.2, 0.4, 0.1, 0.9, 0.3, 0.8, 0.2],
                    [0.5, 0.1, 0.3, 0.1, 0.6, 0.2, 0.5, 0.1],
                    [0.2, 0.0, 0.1, 0.0, 0.2, 0.0, 0.2, 0.0],
                    [0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0],
                ]
            )
        ),
        xy=torch.tensor([[5.0, 5.0], [95.0, 5.0], [5.0, 95.0], [95.0, 95.0]]),
        normal_correct_points=torch.tensor([0.8, 0.5, 0.2, 0.1]),
        normal_wrong_points=torch.tensor([[0.2, 0.3, 0.3, 0.2], [0.1, 0.6, 0.2, 0.2]]),
        rgb_deranged_correct_points=torch.tensor([0.1, 0.1, 0.1, 0.1]),
        rgb_deranged_wrong_points=torch.tensor([[0.2, 0.3, 0.3, 0.2], [0.1, 0.3, 0.2, 0.2]]),
    )


def test_weighted_pose_score_uses_one_normalized_static_distribution() -> None:
    correct, wrong = aggregate_weighted_point_pose_scores(
        correct_points=torch.tensor([1.0, 3.0]),
        wrong_points=torch.tensor([[2.0, 4.0], [4.0, 2.0]]),
        weights=torch.tensor([1.0, 3.0]),
    )
    torch.testing.assert_close(correct, torch.tensor([2.5]))
    torch.testing.assert_close(wrong, torch.tensor([3.5, 2.5]))


def test_weighter_objective_backpropagates_without_pose_inputs_to_the_model() -> None:
    prepared = _prepared()
    logits = torch.tensor([1.0, 0.0, -0.5, -1.0], requires_grad=True)
    weights = torch.softmax(logits, dim=0)
    terms = _query_terms(
        prepared=prepared, weights=weights, image_size=(100, 100), args=_args()
    )
    uniform_mass = torch.tensor([0.75], requires_grad=True)
    loss = _objective(terms=terms, uniform_mass=uniform_mass, args=_args())
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert uniform_mass.grad is not None and torch.isfinite(uniform_mass.grad).all()


def test_initial_weighter_matches_uniform_baseline_on_prepared_scores() -> None:
    prepared = _prepared()
    model = TargetFreePoseEvidenceWeighter(
        feature_center=torch.zeros((8,)),
        feature_scale=torch.ones((8,)),
        hidden_dim=8,
        minimum_uniform_mass=0.50,
        maximum_uniform_mass=0.95,
        initial_uniform_mass=0.75,
    ).eval()
    metrics = evaluate_prepared_queries(
        model=model, queries=[prepared], image_size=(100, 100), args=_args()
    )
    assert metrics["mean_normal_pose_gap"] == pytest.approx(
        metrics["uniform"]["mean_normal_pose_gap"]
    )
    assert metrics["uniform_win_to_selector_loss_count"] == 0


def test_inner_gate_rejects_any_uniform_win_that_becomes_a_learned_loss() -> None:
    metrics = {
        "mean_normal_pose_gap": 0.103,
        "median_normal_pose_gap": 0.102,
        "normal_pose_win_fraction": 0.80,
        "p10_normal_pose_gap": 0.095,
        "minimum_normal_pose_gap": 0.090,
        "uniform_win_to_selector_loss_count": 0,
        "mean_normal_minus_rgb_deranged_pose_gap": 0.103,
        "uniform": {
            "mean_normal_pose_gap": 0.100,
            "median_normal_pose_gap": 0.100,
            "normal_pose_win_fraction": 0.75,
            "p10_normal_pose_gap": 0.094,
            "minimum_normal_pose_gap": 0.085,
            "mean_normal_minus_rgb_deranged_pose_gap": 0.100,
        },
    }
    accepted = inner_gate_decision(metrics=metrics, args=_args())
    assert accepted["passed"] is True
    rejected = inner_gate_decision(
        metrics={**metrics, "uniform_win_to_selector_loss_count": 1}, args=_args()
    )
    assert rejected["no_uniform_win_to_loss"] is False
    assert rejected["passed"] is False
