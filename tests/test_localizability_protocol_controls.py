from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.localizability.controls import (  # noqa: E402
    counterfactual_channel_mask,
    counterfactual_spatial_mask,
    feature_batch_shuffle_control,
    metadata_baseline_scores,
)
from feature_extract.localizability.hard_cases import build_hard_case_masks, summarize_hard_case_masks  # noqa: E402
from feature_extract.localizability.protocol import (  # noqa: E402
    ArtifactProtocol,
    assert_protocol_claims_compatible,
    validate_protocol_metadata,
)
from feature_extract.localizability.rendered_map_scoring import (  # noqa: E402
    render_selected_track_feature_maps,
    score_projected_selected_track_bank,
)
from feature_extract.localizability.reporting import ProtocolResult, format_protocol_summary_markdown  # noqa: E402
from feature_extract.localizability.scorer import PoseHypothesisScorer  # noqa: E402
from feature_extract.localizability.selected_feature_map import aggregate_selected_track_features  # noqa: E402
from feature_extract.localizability.selector import LocalizationFeatureSelector  # noqa: E402
from feature_extract.tools.eval_localizability_protocol_controls import (  # noqa: E402
    _candidate_table_to_tensors,
    available_metadata_baselines,
    summarize_protocol_controls,
)
from feature_extract.tools.build_localizability_hard_cases import build_hard_case_subset_rows  # noqa: E402


def test_protocol_metadata_marks_controlled_lattice_as_non_deployment_and_rejects_leakage():
    protocol = ArtifactProtocol(
        protocol="controlled_lattice",
        candidate_generator="oldhospital_q50_lattice",
        split="val",
        input_fields=("score", "retrieval_scores_candidates"),
        gt_usage="candidate_generation_and_eval",
        solver_conditioned=False,
    )

    with pytest.raises(ValueError, match="retrieval_scores_candidates"):
        validate_protocol_metadata(protocol, training_input=True)

    row = protocol.to_report_row()
    assert row["protocol"] == "controlled_lattice"
    assert row["deployment_claim_allowed"] is False
    assert row["gt_centered"] is True


def test_protocol_claims_cannot_mix_controlled_and_real_retrieval_without_explicit_opt_in():
    controlled = ArtifactProtocol(
        protocol="controlled_lattice",
        candidate_generator="q50",
        split="val",
        input_fields=("score",),
        gt_usage="candidate_generation_and_eval",
    )
    real = ArtifactProtocol(
        protocol="real_retrieval",
        candidate_generator="netvlad_renderloftr_top20",
        split="test",
        input_fields=("score",),
        gt_usage="eval_only",
    )

    with pytest.raises(ValueError, match="mixed protocol"):
        assert_protocol_claims_compatible([controlled, real])

    assert_protocol_claims_compatible([controlled, real], allow_mixed=True)


def test_metadata_baseline_scores_support_rank_inliers_reprojection_and_delta_pose():
    valid = torch.tensor([[True, True, False]])
    fields = {
        "candidate_rank": torch.tensor([[0.0, 1.0, 2.0]]),
        "retrieval_score": torch.tensor([[0.2, 0.9, 0.1]]),
        "pnp_inliers": torch.tensor([[12.0, 8.0, 99.0]]),
        "reproj_median_px": torch.tensor([[5.0, 2.0, 1.0]]),
        "delta_trans_m": torch.tensor([[0.10, 0.05, 0.0]]),
        "delta_rot_deg": torch.tensor([[1.0, 3.0, 0.0]]),
    }

    rank = metadata_baseline_scores(fields, mode="candidate_rank", valid_mask=valid)
    retrieval = metadata_baseline_scores(fields, mode="retrieval_score", valid_mask=valid)
    inliers = metadata_baseline_scores(fields, mode="pnp_inliers", valid_mask=valid)
    reproj = metadata_baseline_scores(fields, mode="reproj_median", valid_mask=valid)
    delta = metadata_baseline_scores(fields, mode="delta_pose", valid_mask=valid, rot_weight=0.01)

    assert int(rank.argmax(dim=1)[0]) == 0
    assert int(retrieval.argmax(dim=1)[0]) == 1
    assert int(inliers.argmax(dim=1)[0]) == 0
    assert int(reproj.argmax(dim=1)[0]) == 1
    assert int(delta.argmax(dim=1)[0]) == 1
    assert torch.isneginf(retrieval[0, 2])


def test_feature_shuffle_and_counterfactual_masks_destroy_specific_evidence():
    feature = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)

    shuffled, permutation = feature_batch_shuffle_control(feature, permutation=torch.tensor([1, 0]))

    assert permutation.tolist() == [1, 0]
    assert torch.allclose(shuffled[0], feature[1])
    assert torch.allclose(shuffled[1], feature[0])

    channel_masked = counterfactual_channel_mask(
        feature[:1],
        torch.tensor([0.1, 0.9, 0.2]),
        mode="remove_high",
        fraction=1.0 / 3.0,
    )
    assert torch.allclose(channel_masked[:, 1], torch.zeros_like(channel_masked[:, 1]))
    assert torch.allclose(channel_masked[:, 0], feature[:1, 0])

    spatial_masked = counterfactual_spatial_mask(
        feature[:1],
        torch.tensor([[[[0.1, 0.9], [0.2, 0.3]]]]),
        mode="remove_high",
        fraction=0.25,
    )
    assert torch.allclose(spatial_masked[:, :, 0, 1], torch.zeros_like(spatial_masked[:, :, 0, 1]))
    assert torch.allclose(spatial_masked[:, :, 1, 1], feature[:1, :, 1, 1])


def test_hard_case_builder_identifies_false_accept_and_pnp_high_score_wrong_cases():
    scores = torch.tensor([[0.9, 0.2, 0.1], [0.7, 0.6, 0.5], [0.1, 0.8, 0.2]])
    costs = torch.tensor([[0.50, 0.10, 0.20], [0.10, 0.40, 0.50], [0.35, 0.12, 0.60]])
    basin = costs <= 0.15
    pnp_inliers = torch.tensor([[80.0, 20.0, 5.0], [20.0, 50.0, 60.0], [10.0, 90.0, 5.0]])
    delta_trans_m = torch.tensor([[0.03, 0.30, 0.20], [0.02, 0.20, 0.30], [0.10, 0.15, 0.30]])
    delta_rot_deg = torch.tensor([[0.5, 5.0, 4.0], [0.3, 2.0, 3.0], [5.0, 2.0, 4.0]])

    masks = build_hard_case_masks(
        scores,
        costs,
        basin_label=basin,
        pnp_inliers=pnp_inliers,
        delta_trans_m=delta_trans_m,
        delta_rot_deg=delta_rot_deg,
        retrieval_topk=3,
    )
    summary = summarize_hard_case_masks(masks)

    assert masks["score_top1_false_accept"].tolist() == [True, False, False]
    assert masks["retrieval_top1_wrong_but_topk_basin"].tolist() == [True, False, True]
    assert masks["near_identity_false_positive"].tolist() == [True, False, False]
    assert masks["pnp_high_score_wrong"].tolist() == [True, True, False]
    assert summary["score_top1_false_accept"]["count"] == 1


def test_projected_selected_track_bank_scores_visible_map_features_with_same_selector():
    observations = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    track_ids = torch.tensor([1, 1, 2, 2])
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    bank = aggregate_selected_track_features(observations, track_ids, xyz=xyz, l2_normalize=False)
    intrinsics = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    poses = torch.eye(4).reshape(1, 1, 4, 4)

    rendered, valid = render_selected_track_feature_maps(bank, poses, intrinsics, image_hw=(5, 5))

    assert rendered.shape == (1, 1, 2, 5, 5)
    assert valid.shape == (1, 1, 1, 5, 5)
    assert torch.allclose(rendered[0, 0, :, 2, 2], torch.tensor([1.0, 0.0]))
    assert valid[0, 0, 0, 2, 2]

    query = torch.zeros(1, 2, 5, 5)
    query[:, 0, 2, 2] = 1.0
    selector = LocalizationFeatureSelector(in_channels=2, out_channels=2, group_size=1, identity_init=True)
    scorer = PoseHypothesisScorer(mode="same_pixel", temperature=0.1)

    scores, aux = score_projected_selected_track_bank(
        query,
        bank,
        poses,
        intrinsics,
        image_hw=(5, 5),
        selector=selector,
        scorer=scorer,
    )

    assert scores.shape == (1, 1)
    assert scores[0, 0] > 0.0
    assert aux["render_valid_mask"].shape == valid.shape


def test_protocol_summary_markdown_keeps_protocols_separate_and_marks_deployment_claims():
    rows = [
        ProtocolResult(
            label="q50 controlled",
            protocol=ArtifactProtocol(
                protocol="controlled_lattice",
                candidate_generator="q50",
                split="val",
                input_fields=("score",),
                gt_usage="candidate_generation_and_eval",
            ),
            metrics={"pred_cost_m": 0.22, "top1_acc": 0.72, "spearman": 0.58, "basin_recall@5": 0.80},
        ),
        ProtocolResult(
            label="real retrieval",
            protocol=ArtifactProtocol(
                protocol="real_retrieval",
                candidate_generator="netvlad",
                split="test",
                input_fields=("score",),
                gt_usage="eval_only",
            ),
            metrics={"pred_cost_m": 0.34, "top1_acc": 0.20, "spearman": 0.50, "basin_recall@5": 0.76},
        ),
    ]

    md = format_protocol_summary_markdown(rows)

    assert "controlled_lattice" in md
    assert "real_retrieval" in md
    assert "deployment claim" in md
    assert "q50 controlled" in md


def test_protocol_controls_summary_compares_pofd_to_available_metadata_baselines():
    rows = [
        {
            "sample_name": "q0",
            "candidate_idx": 0,
            "score": 0.9,
            "score_rank": 0,
            "pose_cost_m": 0.50,
            "trans_err_m": 0.50,
            "rot_err_deg": 1.0,
            "valid": True,
            "in_basin": False,
            "retrieval_pnp_num_inliers_candidates": 80.0,
            "retrieval_pnp_reproj_median_candidates": 2.0,
        },
        {
            "sample_name": "q0",
            "candidate_idx": 1,
            "score": 0.2,
            "score_rank": 1,
            "pose_cost_m": 0.10,
            "trans_err_m": 0.10,
            "rot_err_deg": 1.0,
            "valid": True,
            "in_basin": True,
            "retrieval_pnp_num_inliers_candidates": 20.0,
            "retrieval_pnp_reproj_median_candidates": 5.0,
        },
    ]

    assert available_metadata_baselines(rows) == ["candidate_rank", "pnp_inliers", "reproj_median"]

    summary = summarize_protocol_controls(
        rows,
        label="unit",
        protocol="real_retrieval",
        candidate_generator="toy",
        split="test",
        baseline_modes=("candidate_rank", "pnp_inliers", "reproj_median"),
    )

    assert summary["label"] == "unit"
    assert summary["protocol"]["deployment_claim_allowed"] is True
    assert summary["pofd"]["top1_acc"] == 0.0
    assert summary["metadata_baselines"]["candidate_rank"]["top1_acc"] == 0.0
    assert summary["metadata_baselines"]["pnp_inliers"]["top1_acc"] == 0.0
    assert summary["metadata_baselines"]["reproj_median"]["top1_acc"] == 0.0
    assert summary["hard_cases"]["score_top1_false_accept"]["count"] == 1
    assert summary["paired_statistics"]["candidate_rank"]["pred_cost_delta_m"]["num_samples"] == 1
    assert summary["paired_statistics"]["candidate_rank"]["mcnemar"]["method_only"] == 0
    assert "wilcoxon_cost" in summary["paired_statistics"]["candidate_rank"]


def test_protocol_controls_candidate_rank_uses_candidate_idx_not_model_score_rank():
    rows = [
        {
            "sample_name": "q0",
            "candidate_idx": 0,
            "score": 0.1,
            "score_rank": 1,
            "pose_cost_m": 0.50,
            "valid": True,
            "in_basin": False,
        },
        {
            "sample_name": "q0",
            "candidate_idx": 1,
            "score": 0.9,
            "score_rank": 0,
            "pose_cost_m": 0.10,
            "valid": True,
            "in_basin": True,
        },
    ]

    _, tensors = _candidate_table_to_tensors(rows)
    assert tensors["candidate_idx"].tolist() == [[0.0, 1.0]]

    summary = summarize_protocol_controls(
        rows,
        label="unit",
        protocol="real_retrieval",
        candidate_generator="toy",
        split="test",
        baseline_modes=("candidate_rank",),
    )

    assert summary["pofd"]["top1_acc"] == 1.0
    assert summary["metadata_baselines"]["candidate_rank"]["top1_acc"] == 0.0


def test_protocol_controls_warns_when_controlled_retrieval_score_is_oracle_like():
    rows = [
        {
            "sample_name": "q0",
            "candidate_idx": 0,
            "score": 0.1,
            "pose_cost_m": 0.50,
            "retrieval_scores_candidates": 0.0,
            "valid": True,
            "in_basin": False,
        },
        {
            "sample_name": "q0",
            "candidate_idx": 1,
            "score": 0.9,
            "pose_cost_m": 0.10,
            "retrieval_scores_candidates": 1.0,
            "valid": True,
            "in_basin": True,
        },
    ]

    summary = summarize_protocol_controls(
        rows,
        label="unit",
        protocol="controlled_lattice",
        candidate_generator="toy_gt_centered",
        split="val",
        gt_usage="candidate_generation_and_eval",
    )

    assert any("retrieval_score" in warning for warning in summary["control_warnings"])


def test_build_hard_case_subset_rows_exports_full_candidate_groups():
    rows = [
        {"sample_name": "a", "candidate_idx": 0, "score": 0.9, "pose_cost_m": 0.50, "valid": True, "in_basin": False},
        {"sample_name": "a", "candidate_idx": 1, "score": 0.1, "pose_cost_m": 0.10, "valid": True, "in_basin": True},
        {"sample_name": "b", "candidate_idx": 0, "score": 0.9, "pose_cost_m": 0.10, "valid": True, "in_basin": True},
        {"sample_name": "b", "candidate_idx": 1, "score": 0.1, "pose_cost_m": 0.50, "valid": True, "in_basin": False},
    ]

    subsets = build_hard_case_subset_rows(rows, retrieval_topk=2)

    assert [row["sample_name"] for row in subsets["score_top1_false_accept"]] == ["a", "a"]
    assert subsets["score_top1_false_accept_summary"]["count"] == 1
