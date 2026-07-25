from __future__ import annotations

import numpy as np
import pytest
import torch

from feature_extract.tools.vfm.pretrain_candidate_pose_rgb_spatial_hard_pose_likelihood import (
    _context_identity_control_metrics,
    _rank_group_rows,
    _validate_args,
    hard_pose_group_margin_loss,
    hard_pose_pretrain_gate,
    parse_args,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_hard_pose_pairs import (
    CandidatePoseRGBSpatialHardPosePairs,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_likelihood import (
    CandidatePoseRGBSpatialEdgePrediction,
    CandidatePoseRGBSpatialRuntime,
    runtime_from_target_free_layout,
)


def test_hard_pose_group_margin_raises_correct_pose_and_lowers_wrong_pose() -> None:
    correct = torch.tensor([0.10], requires_grad=True)
    wrong = torch.tensor([0.40], requires_grad=True)

    loss, metrics = hard_pose_group_margin_loss(
        correct_scores=correct,
        wrong_scores=wrong,
        margin=0.25,
    )

    assert metrics["group_count"] == pytest.approx(1.0)
    assert metrics["mean_correct_minus_wrong"] == pytest.approx(-0.30)
    assert metrics["correct_win_fraction"] == pytest.approx(0.0)
    assert loss.item() == pytest.approx(0.55)
    loss.backward()
    assert float(correct.grad) < 0.0
    assert float(wrong.grad) > 0.0


def test_hard_pose_pretrain_gate_requires_visual_permutation_separation() -> None:
    passed = hard_pose_pretrain_gate(
        {
            "normal_correct_win_fraction": 0.70,
            "normal_mean_correct_minus_wrong": 0.18,
            "visual_gap_delta": 0.08,
        },
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.05,
    )
    assert passed["passed"] is True
    failed = hard_pose_pretrain_gate(
        {
            "normal_correct_win_fraction": 0.70,
            "normal_mean_correct_minus_wrong": 0.18,
            "visual_gap_delta": 0.01,
        },
        minimum_win_fraction=0.55,
        minimum_normal_gap=0.05,
        minimum_visual_gap_delta=0.05,
    )
    assert failed["passed"] is False
    assert failed["checks"]["visual_gap_delta"] is False


def test_rank_partition_never_splits_a_pose_group() -> None:
    rows_by_group = {
        11: np.asarray([0, 1, 2], dtype=np.int64),
        13: np.asarray([3, 4, 5], dtype=np.int64),
        17: np.asarray([6, 7, 8], dtype=np.int64),
    }
    assigned = []
    for rank in range(2):
        assigned.extend(
            _rank_group_rows(
                group_ids=np.asarray([11, 13, 17], dtype=np.int64),
                rows_by_group=rows_by_group,
                rank=rank,
                world_size=2,
                seed=29,
                epoch=3,
            )
        )
    expected = {tuple(value.tolist()) for value in rows_by_group.values()}
    assert {tuple(value.tolist()) for value in assigned}.issubset(expected)
    assert expected.issubset({tuple(value.tolist()) for value in assigned})


def test_target_bearing_hard_pose_pairs_cannot_build_runtime_layout() -> None:
    with pytest.raises(ValueError, match="target-free layout"):
        runtime_from_target_free_layout(
            object.__new__(CandidatePoseRGBSpatialHardPosePairs),
            image_ids=np.asarray(["query/a.png"]),
        )


def test_full_hard_pose_pretrain_allows_context_support_permutation_loss() -> None:
    """The permutation branch is context-only, even when the main path is full.

    The normal full forward provides RGB spatial and pose gradients.  The
    deranged branch deliberately calls ``context_only=True`` and receives no
    RGB patches, so it is a valid independent context-appearance control in a
    joint objective rather than an L0-only special case.
    """

    args = parse_args(
        [
            "--hard-pose-pairs",
            "pairs.npz",
            "--radio-final-context-cache",
            "final.npz",
            "--radio-intermediate-context-cache",
            "intermediate.npz",
            "--alike-spatial-context-cache",
            "alike.npz",
            "--image-root",
            "images",
            "--output-dir",
            "output",
            "--identity-support-permutation-loss-weight",
            "0.25",
        ]
    )

    windows = _validate_args(args)

    assert windows == {
        "radio_final": 9,
        "radio_intermediate": 9,
        "alike": 13,
    }


def test_context_identity_control_metrics_compare_normal_and_deranged_support() -> None:
    runtime = CandidatePoseRGBSpatialRuntime(
        query_image_indices=torch.tensor([0, 1]),
        query_xy=torch.zeros((2, 2)),
        support_image_indices=torch.tensor([[[2], [3]], [[4], [5]]]),
        support_xy=torch.zeros((2, 2, 1, 2)),
        support_view_valid=torch.ones((2, 2, 1), dtype=torch.bool),
        candidate_view_weights=torch.ones((2, 2, 1)),
        candidate_probabilities=torch.full((2, 2), 0.45),
        null_probabilities=torch.full((2,), 0.10),
    )
    shape = (2, 2, 1, 4)
    joint = torch.full((*shape[:-1], 5), 0.2).log()
    common = {
        "spatial_logits": torch.zeros(shape),
        "non_dustbin_logits": torch.zeros(shape[:-1]),
        "joint_log_probabilities": joint,
        "offsets_xy": torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]
        ),
        "edge_usable": torch.ones(shape[:-1], dtype=torch.bool),
    }
    normal = CandidatePoseRGBSpatialEdgePrediction(
        **{
            **common,
            "context_log_likelihood_ratios": torch.tensor(
                [[[2.0], [0.0]], [[0.0], [2.0]]]
            ),
        }
    )
    deranged = CandidatePoseRGBSpatialEdgePrediction(
        **{
            **common,
            "context_log_likelihood_ratios": torch.zeros((2, 2, 1)),
        }
    )

    metrics = _context_identity_control_metrics(
        runtime=runtime,
        prediction=normal,
        permuted_runtime=runtime,
        permuted_prediction=deranged,
        target_observed=torch.tensor([[True, False], [False, True]]),
    )

    assert metrics["context_identity_active_rows"] == pytest.approx(2.0)
    assert metrics["context_identity_normal_top1_sum"] == pytest.approx(2.0)
    assert metrics["context_identity_normal_margin_sum"] == pytest.approx(4.0)
    assert metrics["context_identity_permuted_top1_sum"] == pytest.approx(1.0)
    assert metrics["context_identity_target_score_gap_sum"] == pytest.approx(4.0)
