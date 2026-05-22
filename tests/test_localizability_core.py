import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.localizability.candidate_bank import (
    CandidateBank,
    CandidateBankMetadata,
    candidate_bank_from_npz,
)
from feature_extract.localizability.adapter_bundle import collect_pose_adapter_trainable_parameters
from feature_extract.localizability.bank_schema import (
    CandidateRow,
    candidate_rows_from_jsonl,
    validate_no_forbidden_training_inputs,
)
from feature_extract.localizability.losses import (
    basin_bce_loss,
    online_score_hard_negative_loss,
    pose_distance_soft_rank_loss,
)
from feature_extract.localizability.interpretability import (
    channel_group_counterfactual_drop,
    spatial_utility_counterfactual_drop,
)
from feature_extract.localizability.mapability import observation_track_feature_variance, track_feature_variance
from feature_extract.localizability.mapability import (
    render_query_feature_consistency,
    observation_track_feature_separability,
)
from feature_extract.localizability.metrics import (
    candidate_score_table_rows,
    ranking_metrics,
    ranking_row_diagnostics,
    risk_coverage_metrics,
)
from feature_extract.localizability.reference_pose_bank import (
    build_reference_pose_bank,
    parse_hloc_pairs_lines,
)
from feature_extract.localizability.reference_pose_scoring import (
    descriptor_from_dense_feature,
    project_descriptor_bank_pca,
    load_descriptor_bank,
    load_patch_descriptor_bank,
    patch_descriptors_from_dense_feature,
    retrieval_order_scores,
    save_descriptor_bank,
    save_patch_descriptor_bank,
    score_reference_pose_patch_descriptors,
    score_reference_pose_descriptors,
)
from feature_extract.localizability.scorer import PoseHypothesisScorer
from feature_extract.localizability.score_calibrator import HypothesisScoreCalibrator, group_candidate_table_rows
from feature_extract.localizability.selector import LocalizationFeatureSelector
from feature_extract.localizability.solver_handoff import evaluate_handoff_rows
from feature_extract.students.pose_energy_net import PairConditionedLocalMatcher, PoseFeatureDomainAdapter


def test_localization_feature_selector_outputs_compact_feature_and_gates():
    torch.manual_seed(7)
    selector = LocalizationFeatureSelector(
        in_channels=8,
        out_channels=4,
        group_size=2,
        spatial_utility=True,
        uncertainty=True,
    )
    feature = torch.randn(2, 8, 5, 7)

    outputs = selector(feature)

    assert outputs["z"].shape == (2, 4, 5, 7)
    assert outputs["utility"].shape == (2, 1, 5, 7)
    assert outputs["uncertainty"].shape == (2, 1, 5, 7)
    assert outputs["channel_gate"].shape == (2, 4)
    assert torch.allclose(outputs["z"].norm(dim=1).mean(), torch.tensor(1.0), atol=1.0e-5)
    assert outputs["utility"].min() >= 0.0
    assert outputs["utility"].max() <= 1.0


def test_localization_feature_selector_identity_init_preserves_feature_direction():
    selector = LocalizationFeatureSelector(
        in_channels=8,
        out_channels=8,
        group_size=2,
        spatial_utility=True,
        uncertainty=False,
        identity_init=True,
    )
    feature = torch.randn(2, 8, 5, 7)

    outputs = selector(feature)

    expected = torch.nn.functional.normalize(feature.float(), dim=1, eps=1.0e-6)
    cosine = (outputs["z"] * expected).sum(dim=1).mean()
    assert cosine.item() > 0.999


def test_pose_hypothesis_scorer_prefers_aligned_candidate_with_utility_weight():
    query = torch.zeros(1, 4, 9, 9)
    query[:, :, 4, 4] = 1.0
    render = query[:, None].repeat(1, 3, 1, 1, 1)
    render[:, 1] = torch.roll(render[:, 1], shifts=2, dims=-1)
    render[:, 2] = 0.0
    utility = torch.zeros(1, 1, 9, 9)
    utility[:, :, 4, 4] = 1.0

    scorer = PoseHypothesisScorer(mode="local_corr", radius=1, temperature=0.1)
    scores, aux = scorer(query, render, query_utility=utility)

    assert scores.shape == (1, 3)
    assert int(scores.argmax(dim=1)[0]) == 0
    assert aux["score_maps"].shape[:2] == (1, 3)
    assert aux["valid_mask"].all()


def test_pair_matcher_local_uses_center_offset_evidence_not_shift_invariant_max():
    query = torch.zeros(1, 4, 7, 7)
    query[:, :, 3, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    render = torch.zeros(1, 2, 4, 7, 7)
    render[:, 0, :, 3, 3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    render[:, 1, :, 3, 4] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    utility = torch.zeros(1, 1, 7, 7)
    utility[:, :, 3, 3] = 1.0

    local = PoseHypothesisScorer(mode="local_corr", radius=1, temperature=0.1)
    local_scores, _ = local(query, render, query_utility=utility)
    assert torch.allclose(local_scores[:, 0], local_scores[:, 1], atol=1.0e-5)

    matcher = PairConditionedLocalMatcher(
        channels=4,
        hidden_dim=8,
        offset_radius=1,
        zero_init_residual=True,
        base_dot_weight=1.0,
    )
    pair = PoseHypothesisScorer(
        mode="pair_matcher_local",
        radius=1,
        temperature=0.1,
        pair_matcher=matcher,
        pair_matcher_stride=1,
        pair_matcher_score_channel=1,
    )
    pair_scores, aux = pair(query, render, query_utility=utility)

    assert int(pair_scores.argmax(dim=1)[0]) == 0
    assert pair_scores[0, 0] > pair_scores[0, 1]
    assert aux["score_maps"].shape[:3] == (1, 2, 3)


@pytest.mark.parametrize("mode", ["same_pixel", "local_corr", "pair_matcher_local"])
def test_pose_hypothesis_scorer_marks_zero_render_mask_candidates_invalid(mode):
    query = torch.randn(1, 4, 7, 7)
    render = torch.randn(1, 2, 4, 7, 7)
    render_mask = torch.zeros(1, 2, 1, 7, 7)
    pair_matcher = None
    if mode == "pair_matcher_local":
        pair_matcher = PairConditionedLocalMatcher(
            channels=4,
            hidden_dim=8,
            offset_radius=1,
            zero_init_residual=True,
            base_dot_weight=1.0,
        )

    scorer = PoseHypothesisScorer(
        mode=mode,
        radius=1,
        pair_matcher=pair_matcher,
        pair_matcher_stride=1,
        pair_matcher_score_channel=1,
    )
    scores, aux = scorer(query, render, render_valid_mask=render_mask)

    assert torch.allclose(scores, torch.zeros_like(scores))
    assert aux["valid_mask"].tolist() == [[False, False]]
    assert aux["weight_sum"].tolist() == [[0.0, 0.0]]


def test_collect_pose_adapter_trainable_parameters_keeps_backbone_boundaries_explicit():
    adapter = PoseFeatureDomainAdapter(channels=4, hidden_dim=8, rgb_context_enabled=True, rgb_context_channels=2)
    matcher = PairConditionedLocalMatcher(channels=4, hidden_dim=8, offset_radius=1)

    params = collect_pose_adapter_trainable_parameters(
        adapter,
        matcher,
        train_query_adapter=False,
        train_render_adapter=True,
        train_rgb_context=False,
        train_pair_matcher=True,
    )

    assert params
    assert not any(param.requires_grad for param in adapter.query_adapter.parameters())
    assert all(param.requires_grad for param in adapter.render_adapter.parameters())
    assert not any(param.requires_grad for param in adapter.query_rgb_stem.parameters())
    assert all(param.requires_grad for param in matcher.parameters())


def test_pose_rank_losses_focus_on_geometry_and_online_hard_negative():
    scores = torch.tensor([[0.1, 0.7, 0.2]], requires_grad=True)
    costs = torch.tensor([[0.05, 0.40, 0.12]])
    rank_loss, rank_metrics = pose_distance_soft_rank_loss(scores, costs, temperature_m=0.05)
    hard_loss, hard_metrics = online_score_hard_negative_loss(
        scores,
        costs,
        cost_gap_m=0.10,
        margin=0.05,
    )
    basin_loss = basin_bce_loss(scores, torch.tensor([[True, False, True]]))
    total = rank_loss + hard_loss + basin_loss
    total.backward()

    assert rank_loss.item() > 0.0
    assert hard_loss.item() > 0.0
    assert basin_loss.item() > 0.0
    assert rank_metrics["rank_target_entropy"].item() < 1.1
    assert hard_metrics["online_hard_active"].item() == 1.0
    assert scores.grad is not None


def test_ranking_metrics_report_oracle_gap_spearman_and_basin_recall():
    scores = torch.tensor([[0.2, 0.9, 0.1], [0.8, 0.1, 0.2]])
    costs = torch.tensor([[0.30, 0.10, 0.50], [0.40, 0.20, 0.10]])
    basin = costs <= 0.20

    metrics = ranking_metrics(scores, costs, basin_label=basin, topk=(1, 2))

    assert metrics["top1_acc"].item() == 0.5
    assert metrics["pred_cost_m"].item() == 0.25
    assert abs(metrics["oracle_cost_m"].item() - 0.10) < 1.0e-6
    assert abs(metrics["oracle_gap_m"].item() - 0.15) < 1.0e-6
    assert metrics["basin_recall@2"].item() == 1.0
    assert -1.0 <= metrics["spearman"].item() <= 1.0


def test_risk_coverage_metrics_reports_high_confidence_false_accepts():
    confidence = torch.tensor([0.9, 0.8, 0.2, 0.1])
    success = torch.tensor([True, False, True, False])

    metrics = risk_coverage_metrics(confidence, success, coverages=(0.25, 0.5, 1.0))

    assert metrics["success_rate"].item() == 0.5
    assert metrics["risk@25"].item() == 0.0
    assert metrics["risk@50"].item() == 0.5
    assert metrics["risk@100"].item() == 0.5
    assert metrics["high_conf_false_accept@25"].item() == 0.0
    assert metrics["high_conf_false_accept@50"].item() == 0.5
    assert 0.0 <= metrics["risk_coverage_auc"].item() <= 1.0


def test_patch_reference_pose_scorer_prefers_local_patch_alignment(tmp_path):
    query = torch.zeros(4, 4, 4)
    query[:, :2, :2] = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(4, 1, 1)
    query[:, 2:, 2:] = torch.tensor([0.0, 1.0, 0.0, 0.0]).view(4, 1, 1)
    aligned = query.clone()
    wrong = torch.zeros_like(query)
    wrong[:, :2, :2] = torch.tensor([0.0, 0.0, 1.0, 0.0]).view(4, 1, 1)
    wrong[:, 2:, 2:] = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(4, 1, 1)

    patch_bank = {
        "q.png": patch_descriptors_from_dense_feature(query, grid_hw=(2, 2)),
        "aligned.png": patch_descriptors_from_dense_feature(aligned, grid_hw=(2, 2)),
        "wrong.png": patch_descriptors_from_dense_feature(wrong, grid_hw=(2, 2)),
    }
    path = tmp_path / "patch_bank.pt"
    save_patch_descriptor_bank(path, patch_bank, metadata={"grid_hw": [2, 2]})
    loaded, metadata = load_patch_descriptor_bank(path)

    scores, valid = score_reference_pose_patch_descriptors(
        sample_names=["q.png"],
        reference_names=[["wrong.png", "aligned.png"]],
        patch_descriptors=loaded,
        topk=2,
    )

    assert metadata["grid_hw"] == [2, 2]
    assert valid.tolist() == [[True, True]]
    assert int(scores.argmax(dim=1)[0]) == 1
    assert scores[0, 1] > scores[0, 0]


def test_ranking_row_diagnostics_keeps_selected_oracle_and_basin_fields():
    scores = torch.tensor([[0.2, 0.9, 0.1], [0.8, 0.1, 0.2]])
    costs = torch.tensor([[0.30, 0.10, 0.50], [0.40, 0.20, 0.10]])
    basin = costs <= 0.20

    rows = ranking_row_diagnostics(scores, costs, basin_label=basin, sample_names=["a", "b"])

    assert rows[0]["sample_name"] == "a"
    assert rows[0]["selected_idx"] == 1
    assert rows[0]["oracle_idx"] == 1
    assert rows[0]["selected_in_basin"] is True
    assert rows[1]["selected_idx"] == 0
    assert rows[1]["oracle_idx"] == 2
    assert abs(rows[1]["oracle_gap_m"] - 0.3) < 1.0e-6


def test_candidate_score_table_rows_exports_one_row_per_candidate():
    scores = torch.tensor([[0.2, 0.7, -1.0]])
    costs = torch.tensor([[0.3, 0.1, 0.5]])
    trans_err = torch.tensor([[0.29, 0.08, 0.50]])
    rot_err = torch.tensor([[1.0, 2.0, 20.0]])
    valid = torch.tensor([[True, True, False]])
    basin = torch.tensor([[False, True, False]])

    rows = candidate_score_table_rows(
        scores,
        costs,
        trans_err_m=trans_err,
        rot_err_deg=rot_err,
        valid_mask=valid,
        basin_label=basin,
        sample_names=["seq/frame.png"],
        extra_fields={"delta_trans_m": torch.tensor([[0.0, 0.1, 0.2]])},
    )

    assert len(rows) == 3
    assert rows[0]["sample_name"] == "seq/frame.png"
    assert rows[1]["candidate_idx"] == 1
    assert rows[1]["is_oracle"] is True
    assert rows[1]["score_rank"] == 0
    assert abs(rows[1]["delta_trans_m"] - 0.1) < 1.0e-6
    assert rows[2]["valid"] is False


def test_group_candidate_table_rows_stacks_samples_by_candidate_index():
    rows = [
        {"sample_name": "b", "candidate_idx": 1, "score": 0.2, "pose_cost_m": 0.5, "valid": True},
        {"sample_name": "a", "candidate_idx": 0, "score": 0.7, "pose_cost_m": 0.1, "valid": True},
        {"sample_name": "b", "candidate_idx": 0, "score": 0.4, "pose_cost_m": 0.2, "valid": False},
        {"sample_name": "a", "candidate_idx": 1, "score": 0.3, "pose_cost_m": 0.4, "valid": True},
    ]

    table = group_candidate_table_rows(rows, feature_keys=("score",))

    assert table.sample_names == ["b", "a"]
    assert table.features.shape == (2, 2, 1)
    assert torch.allclose(table.features[0, :, 0], torch.tensor([0.4, 0.2]))
    assert torch.allclose(table.pose_cost_m[1], torch.tensor([0.1, 0.4]))
    assert table.valid_mask.tolist() == [[False, True], [True, True]]


def test_group_candidate_table_rows_computes_score_relative_features():
    rows = [
        {"sample_name": "a", "candidate_idx": 0, "score": 1.0, "score_rank": 1, "pose_cost_m": 0.2, "valid": True},
        {"sample_name": "a", "candidate_idx": 1, "score": 3.0, "score_rank": 0, "pose_cost_m": 0.1, "valid": True},
        {"sample_name": "a", "candidate_idx": 2, "score": -5.0, "score_rank": 2, "pose_cost_m": 0.5, "valid": False},
    ]

    table = group_candidate_table_rows(
        rows,
        feature_keys=("score_margin_to_top1", "score_rank_norm", "score_zscore"),
    )

    assert torch.allclose(table.features[0, :, 0], torch.tensor([-2.0, 0.0, 0.0]), atol=1.0e-6)
    assert torch.allclose(table.features[0, :, 1], torch.tensor([0.5, 0.0, 1.0]), atol=1.0e-6)
    assert abs(float(table.features[0, :2, 2].mean())) < 1.0e-6
    assert table.features[0, 2, 2].item() == 0.0


def test_hypothesis_score_calibrator_supports_linear_and_mlp_outputs():
    features = torch.randn(2, 3, 4)

    linear = HypothesisScoreCalibrator(feature_dim=4, model_type="linear")
    mlp = HypothesisScoreCalibrator(
        feature_dim=4,
        model_type="mlp",
        hidden_dim=8,
        num_layers=2,
        score_residual_weight=0.5,
        score_feature_index=0,
    )

    assert linear(features).shape == (2, 3)
    assert mlp(features).shape == (2, 3)


def test_solver_handoff_can_select_topk_by_external_solver_quality():
    rows = [
        {
            "sample_name": "q0",
            "candidate_idx": 0,
            "score": 10.0,
            "pose_cost_m": 0.40,
            "trans_err_m": 0.40,
            "rot_err_deg": 3.0,
            "valid": True,
            "retrieval_pnp_success_candidates": 1.0,
            "retrieval_pnp_num_inliers_candidates": 20.0,
        },
        {
            "sample_name": "q0",
            "candidate_idx": 1,
            "score": 5.0,
            "pose_cost_m": 0.10,
            "trans_err_m": 0.10,
            "rot_err_deg": 1.0,
            "valid": True,
            "retrieval_pnp_success_candidates": 1.0,
            "retrieval_pnp_num_inliers_candidates": 80.0,
        },
        {
            "sample_name": "q1",
            "candidate_idx": 0,
            "score": 9.0,
            "pose_cost_m": 0.20,
            "trans_err_m": 0.20,
            "rot_err_deg": 2.0,
            "valid": True,
            "retrieval_pnp_success_candidates": 1.0,
            "retrieval_pnp_num_inliers_candidates": 30.0,
        },
        {
            "sample_name": "q1",
            "candidate_idx": 1,
            "score": 8.0,
            "pose_cost_m": 0.50,
            "trans_err_m": 0.50,
            "rot_err_deg": 5.0,
            "valid": True,
            "retrieval_pnp_success_candidates": 1.0,
            "retrieval_pnp_num_inliers_candidates": 90.0,
        },
    ]

    top1 = evaluate_handoff_rows(rows, topk=1, selection_mode="pofd_score")
    inliers = evaluate_handoff_rows(rows, topk=2, selection_mode="pnp_inliers")

    assert abs(top1["trans_mean_m"] - 0.30) < 1.0e-6
    assert abs(inliers["trans_mean_m"] - 0.30) < 1.0e-6
    assert inliers["selected_candidate_indices"] == [1, 1]
    assert inliers["success_25cm_10deg"] == 0.5


def test_candidate_bank_from_npz_normalizes_required_fields(tmp_path):
    path = tmp_path / "bank.npz"
    pose_gt = torch.eye(4).view(1, 4, 4).numpy()
    candidates = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1).numpy()
    import numpy as np

    np.savez(
        path,
        sample_names=np.array(["seq/frame.png"]),
        pose_gt=pose_gt,
        candidates=candidates,
        pose_cost_m=np.array([[0.1, 0.5]], dtype=np.float32),
        trans_err_m=np.array([[0.1, 0.5]], dtype=np.float32),
        rot_err_deg=np.array([[1.0, 10.0]], dtype=np.float32),
    )

    bank = candidate_bank_from_npz(path)

    assert isinstance(bank, CandidateBank)
    assert isinstance(bank.metadata, CandidateBankMetadata)
    assert bank.candidate_pose.shape == (1, 2, 4, 4)
    assert bank.basin_label(0.25, 5.0).tolist() == [[True, False]]


def test_bank_schema_loads_candidate_rows_and_rejects_training_leakage(tmp_path):
    rows_path = tmp_path / "candidate_rows.jsonl"
    rows_path.write_text(
        "\n".join(
            [
                '{"sample_name":"q1","candidate_idx":0,"score":0.4,"pose_cost_m":0.1,"valid":true}',
                '{"sample_name":"q1","candidate_idx":1,"score":0.9,"pose_cost_m":0.5,"valid":true}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = candidate_rows_from_jsonl(rows_path)

    assert rows == [
        CandidateRow(sample_name="q1", candidate_idx=0, score=0.4, pose_cost_m=0.1, valid=True),
        CandidateRow(sample_name="q1", candidate_idx=1, score=0.9, pose_cost_m=0.5, valid=True),
    ]
    with pytest.raises(ValueError, match="retrieval_scores_candidates"):
        validate_no_forbidden_training_inputs(["score", "retrieval_scores_candidates"])


def test_reference_pose_bank_from_hloc_pairs_computes_pose_errors_and_valid_mask():
    def pose_at(x: float) -> torch.Tensor:
        pose = torch.eye(4)
        pose[0, 3] = x
        return pose

    pairs = parse_hloc_pairs_lines(
        [
            "query/a.png ref/near.png",
            "query/a.png ref/far.png",
            "query/b.png ref/missing.png",
        ]
    )
    bank = build_reference_pose_bank(
        query_poses={
            "query/a.png": pose_at(0.0).numpy(),
            "query/b.png": pose_at(2.0).numpy(),
        },
        reference_poses={
            "ref/near.png": pose_at(0.1).numpy(),
            "ref/far.png": pose_at(1.0).numpy(),
        },
        query_to_refs=pairs,
        topk=2,
        scene="TinyScene",
    )

    assert isinstance(bank, CandidateBank)
    assert bank.sample_names == ["query/a.png", "query/b.png"]
    assert bank.candidate_pose.shape == (2, 2, 4, 4)
    assert bank.valid_mask.tolist() == [[True, True], [False, False]]
    assert torch.allclose(bank.trans_err_m[0], torch.tensor([0.1, 1.0]), atol=1.0e-6)
    assert torch.isinf(bank.pose_cost_m[1]).all()
    assert bank.metadata.scene == "TinyScene"
    assert bank.metadata.candidate_source == "reference_pose_pairs"


def test_reference_pose_descriptor_scoring_prefers_matching_reference():
    feature = torch.zeros(4, 3, 5)
    feature[0] = 1.0
    desc = descriptor_from_dense_feature(feature)
    assert desc.shape == (4,)
    assert torch.allclose(desc.norm(), torch.tensor(1.0), atol=1.0e-6)

    scores, valid = score_reference_pose_descriptors(
        sample_names=["q/a.png"],
        reference_names=[["r/good.png", "r/bad.png"]],
        descriptors={
            "q/a.png": torch.tensor([1.0, 0.0, 0.0]),
            "r/good.png": torch.tensor([0.9, 0.1, 0.0]),
            "r/bad.png": torch.tensor([0.0, 1.0, 0.0]),
        },
    )

    assert scores.shape == (1, 2)
    assert valid.tolist() == [[True, True]]
    assert int(scores.argmax(dim=1)[0]) == 0


def test_retrieval_order_scores_keep_first_valid_reference_on_top():
    valid_mask = torch.tensor([[True, True, False], [False, True, True]])

    scores = retrieval_order_scores(valid_mask)

    assert scores.shape == (2, 3)
    assert int(scores[0].argmax()) == 0
    assert int(scores[1].argmax()) == 1
    assert scores[0, 0] > scores[0, 1] > scores[0, 2]


def test_descriptor_bank_roundtrip_preserves_names_and_normalized_vectors(tmp_path):
    path = tmp_path / "descriptors.pt"
    descriptors = {
        "q/a.png": torch.tensor([3.0, 4.0]),
        "r/b.png": torch.tensor([0.0, 2.0]),
    }

    save_descriptor_bank(path, descriptors, metadata={"feature_key": "fine"})
    loaded, metadata = load_descriptor_bank(path)

    assert metadata["feature_key"] == "fine"
    assert sorted(loaded) == ["q/a.png", "r/b.png"]
    assert torch.allclose(loaded["q/a.png"].norm(), torch.tensor(1.0), atol=1.0e-6)
    scores, valid = score_reference_pose_descriptors(
        sample_names=["q/a.png"],
        reference_names=[["r/b.png"]],
        descriptors=loaded,
    )
    assert valid.tolist() == [[True]]
    assert scores.shape == (1, 1)


def test_project_descriptor_bank_pca_keeps_descriptor_api_and_compacts_dim():
    descriptors = {
        "q/a.png": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "r/good.png": torch.tensor([0.9, 0.1, 0.0, 0.0]),
        "r/bad.png": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }

    projected, metadata = project_descriptor_bank_pca(descriptors, out_dim=2)
    scores, valid = score_reference_pose_descriptors(
        sample_names=["q/a.png"],
        reference_names=[["r/good.png", "r/bad.png"]],
        descriptors=projected,
    )

    assert metadata["method"] == "pca"
    assert metadata["input_dim"] == 4
    assert metadata["output_dim"] == 2
    assert sorted(projected) == ["q/a.png", "r/bad.png", "r/good.png"]
    assert all(vec.shape == (2,) for vec in projected.values())
    assert all(torch.allclose(vec.norm(), torch.tensor(1.0), atol=1.0e-6) for vec in projected.values())
    assert valid.tolist() == [[True, True]]
    assert int(scores.argmax(dim=1)[0]) == 0


def test_track_feature_variance_ignores_invalid_observations():
    features = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
        ]
    )
    track_ids = torch.tensor([0, 0, 1])
    valid = torch.tensor([True, True, False])

    variance, stats = track_feature_variance(features, track_ids, valid_mask=valid)

    assert variance.item() < 1.0e-6
    assert stats["num_tracks"].item() == 1


def test_observation_track_feature_variance_measures_multi_view_consistency():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.2, 0.0],
            [0.0, 1.0],
            [0.0, 1.4],
            [9.0, 9.0],
        ]
    )
    track_ids = torch.tensor([10, 10, 20, 20, 30])
    valid = torch.tensor([True, True, True, True, False])

    variance, stats = observation_track_feature_variance(features, track_ids, valid_mask=valid)

    assert stats["num_tracks"].item() == 2
    assert stats["num_observations"].item() == 4
    assert variance.item() > 0.0
    assert variance.item() < 0.05


def test_observation_track_feature_separability_reports_between_over_within_ratio():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [1.1, 0.0],
            [0.0, 1.0],
            [0.0, 1.1],
        ]
    )
    track_ids = torch.tensor([10, 10, 20, 20])

    ratio, stats = observation_track_feature_separability(features, track_ids)

    assert stats["num_tracks"].item() == 2
    assert stats["within_track_variance"].item() > 0.0
    assert stats["between_track_distance"].item() > stats["within_track_variance"].item()
    assert ratio.item() > 10.0


def test_render_query_feature_consistency_averages_valid_dense_cosine_matches():
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    render = torch.tensor(
        [
            [
                [[[1.0, 0.0]], [[0.0, 1.0]]],
                [[[0.0, 1.0]], [[1.0, 0.0]]],
            ]
        ]
    )
    valid = torch.tensor([[[[True, True]], [[True, False]]]])

    consistency, stats = render_query_feature_consistency(query, render, valid_mask=valid)

    assert stats["num_valid"].item() == 3
    assert torch.allclose(consistency, torch.tensor(2.0 / 3.0), atol=1.0e-6)


def test_channel_group_counterfactual_drop_reports_high_utility_groups_as_more_important():
    scores = torch.tensor([[3.0, 1.0, 0.0], [1.5, 2.0, 0.1]])
    costs = torch.tensor([[0.1, 0.4, 0.8], [0.2, 0.1, 0.7]])
    group_scores = torch.stack(
        [
            scores - torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            scores - torch.tensor([[0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]),
        ],
        dim=0,
    )

    report = channel_group_counterfactual_drop(scores, group_scores, costs)

    assert report["base_pred_cost_m"].item() == pytest.approx(0.1)
    assert int(report["worst_group_idx"].item()) == 0
    assert report["group_pred_cost_drop_m"][0] > report["group_pred_cost_drop_m"][1]


def test_spatial_utility_counterfactual_drop_masks_high_utility_regions():
    score_maps = torch.zeros(1, 2, 4, 4)
    score_maps[:, 0, 0, 0] = 5.0
    score_maps[:, 1, 3, 3] = 4.0
    utility = torch.zeros(1, 1, 4, 4)
    utility[:, :, 0, 0] = 1.0
    utility[:, :, 3, 3] = 0.1
    costs = torch.tensor([[0.1, 0.5]])

    report = spatial_utility_counterfactual_drop(score_maps, utility, costs, drop_fraction=1.0 / 16.0)

    assert report["base_selected_idx"].tolist() == [0]
    assert report["drop_high_selected_idx"].tolist() == [1]
    assert report["drop_low_selected_idx"].tolist() == [0]
    assert report["drop_high_pred_cost_m"] > report["base_pred_cost_m"]
