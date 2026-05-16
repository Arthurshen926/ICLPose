import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.tools.train_nvs_pose_feature_adapter import (
    apply_config_defaults,
    apply_pose_energy_residual_update,
    build_nvs_candidate_bank,
    candidate_correction_cosines,
    candidate_identity_mask,
    candidate_teacher_quality_listwise_loss,
    candidate_selection_bias_metrics,
    candidate_teacher_quality_scores_from_batch,
    effective_candidate_teacher_quality_weight,
    collect_trainable_parameters,
    pose_energy_factorized_selection_metrics,
    pose_energy_logits_with_base_prior,
    local_flow_nce_loss,
    local_flow_nce_loss_from_corr,
    pair_matcher_local_candidate_score_maps,
    pair_matcher_local_candidate_scores,
    local_zero_offset_scores_from_corr,
    local_zero_offset_correlation_scores,
    masked_dense_alignment_loss,
    masked_dense_cosine,
    load_adapter_checkpoint,
    nvs_pose_energy_vector_dim,
    parse_args as parse_nvs_pose_feature_adapter_args,
    pose_energy_correction_cosine_soft_label_loss,
    score_anti_identity_loss,
    score_pose_improvement_soft_label_loss,
    nvs_teacher_correspondence_loss,
    nvs_teacher_pair_match_loss,
    observability_contrast_loss,
    candidate_observability_margin_loss,
    observability_score_contrast_loss,
    hard_flow_candidate_selection_mask,
    pose_threshold_success_metrics,
    parse_score_feature_hw,
    selected_candidate_pose_metrics,
    pose_energy_direction_pairwise_loss,
    pose_energy_score_monotonicity_loss,
    project_world_positions_to_feature_grid,
    rank_losses_from_scores,
    resize_score_candidate_tensors,
    save_checkpoint,
    _flow_geometry_for_score_hw,
    _scale_intrinsics_between_hw,
    variance_floor_loss,
    warped_candidate_alignment_loss,
)
from feature_extract.tools.train_pose_energy import load_pose_energy_checkpoint
from feature_extract.tools.eval_pose_energy_buckets import build_candidate_bank as build_eval_pose_energy_candidate_bank
from feature_extract.students.pose_energy_net import PairConditionedLocalMatcher, PoseEnergyNet, PoseFeatureDomainAdapter


def _w2c_pose_from_center_and_yaw(center, yaw_deg):
    yaw = torch.tensor(float(yaw_deg) * torch.pi / 180.0)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor(
        [
            [cos_y, -sin_y, 0.0],
            [sin_y, cos_y, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center_t = torch.tensor(center, dtype=torch.float32)
    pose[:3, 3] = -(pose[:3, :3] @ center_t)
    return pose


def test_pose_energy_factorized_selection_metrics_reports_composed_pose_cost():
    pose_gt = _w2c_pose_from_center_and_yaw([1.0, 0.0, 0.0], 30.0).view(1, 4, 4)
    candidate_pose = torch.stack(
        [
            _w2c_pose_from_center_and_yaw([1.0, 0.0, 0.0], 0.0),
            _w2c_pose_from_center_and_yaw([0.0, 0.0, 0.0], 30.0),
            _w2c_pose_from_center_and_yaw([0.0, 0.0, 0.0], 0.0),
        ],
        dim=0,
    ).view(1, 3, 4, 4)
    outputs = {
        "energy_logits": torch.zeros(1, 3),
        "translation_energy_logits": torch.tensor([[3.0, 1.0, 0.0]]),
        "rotation_energy_logits": torch.tensor([[0.0, 4.0, 1.0]]),
    }

    metrics = pose_energy_factorized_selection_metrics(
        outputs,
        candidate_pose,
        pose_gt,
        rot_cost_weight=1.0,
    )

    assert int(metrics["factorized_translation_idx"][0]) == 0
    assert int(metrics["factorized_rotation_idx"][0]) == 1
    assert metrics["factorized_translation_top1_acc"].item() == 1.0
    assert metrics["factorized_rotation_top1_acc"].item() == 1.0
    assert metrics["factorized_pred_cost_m"].item() < 1.0e-3


def test_pose_energy_logits_with_base_prior_preserves_rank_prior_with_zero_residual():
    logits = torch.zeros(1, 3)
    base_scores = torch.tensor([[0.1, 0.9, 0.2]])
    valid = torch.tensor([[True, True, False]])

    combined = pose_energy_logits_with_base_prior(
        logits,
        base_scores,
        valid,
        weight=1.0,
        mode="zscore",
    )

    assert combined[0, 1] > combined[0, 0]
    assert combined[0, 2] < -1.0e5


def test_load_adapter_checkpoint_can_extend_with_new_uncertainty_heads(tmp_path):
    old_adapter = PoseFeatureDomainAdapter(channels=4, uncertainty_enabled=False)
    checkpoint_path = tmp_path / "adapter.pth"
    torch.save({"pose_feature_adapter_state_dict": old_adapter.state_dict()}, checkpoint_path)
    new_adapter = PoseFeatureDomainAdapter(channels=4, uncertainty_enabled=True)

    info = load_adapter_checkpoint(checkpoint_path, new_adapter, strict_adapter=False)

    assert "missing_adapter_keys" in info
    assert any(key.startswith("query_uncertainty.") for key in info["missing_adapter_keys"])
    assert any(key.startswith("render_uncertainty.") for key in info["missing_adapter_keys"])


def test_load_adapter_checkpoint_can_drop_old_uncertainty_heads(tmp_path):
    old_adapter = PoseFeatureDomainAdapter(channels=4, uncertainty_enabled=True)
    checkpoint_path = tmp_path / "adapter_with_uncertainty.pth"
    torch.save({"pose_feature_adapter_state_dict": old_adapter.state_dict()}, checkpoint_path)
    new_adapter = PoseFeatureDomainAdapter(channels=4, uncertainty_enabled=False)

    info = load_adapter_checkpoint(checkpoint_path, new_adapter, strict_adapter=False)

    assert "unexpected_adapter_keys" in info
    assert any(key.startswith("query_uncertainty.") for key in info["unexpected_adapter_keys"])
    assert any(key.startswith("render_uncertainty.") for key in info["unexpected_adapter_keys"])


def test_masked_dense_cosine_prefers_matching_candidate():
    query = torch.zeros(1, 2, 3, 3)
    query[:, 0] = 1.0
    render = torch.zeros(1, 2, 2, 3, 3)
    render[:, 0, 0] = 1.0
    render[:, 1, 1] = 1.0

    scores = masked_dense_cosine(query, render)

    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]


def test_alignment_loss_is_lower_for_identical_features():
    query = torch.randn(2, 4, 5, 5)
    same_loss, same_cos = masked_dense_alignment_loss(query, query)
    diff_loss, diff_cos = masked_dense_alignment_loss(query, -query)

    assert same_loss < diff_loss
    assert same_cos > diff_cos


def test_observability_contrast_loss_prefers_gt_render_over_wrong_pose():
    query = torch.zeros(1, 2, 2, 2)
    query[:, 0] = 1.0
    gt_render = query.clone()
    bad_render = torch.zeros(1, 2, 2, 2, 2)
    bad_render[:, 0, 1] = 1.0
    bad_render[:, 1, 0] = 1.0
    pose_cost = torch.tensor([[0.30, 0.02]])
    valid = torch.ones_like(pose_cost, dtype=torch.bool)

    good = observability_contrast_loss(
        query,
        gt_render,
        bad_render,
        pose_cost,
        valid,
        margin=0.05,
        min_negative_cost_m=0.10,
    )
    bad = observability_contrast_loss(
        -query,
        gt_render,
        bad_render,
        pose_cost,
        valid,
        margin=0.05,
        min_negative_cost_m=0.10,
    )

    assert good["active"].item() == 1.0
    assert good["loss"] < bad["loss"]
    assert good["gt_score"] > good["hard_negative_score"]
    assert bad["gt_score"] < bad["hard_negative_score"]


def test_observability_score_contrast_loss_uses_selector_scores_directly():
    pose_cost = torch.tensor([[0.30, 0.02, 0.45]])
    valid = torch.ones_like(pose_cost, dtype=torch.bool)
    good = observability_score_contrast_loss(
        torch.tensor([2.0]),
        torch.tensor([[0.0, 3.0, 1.0]]),
        pose_cost,
        valid,
        margin=0.1,
        min_negative_cost_m=0.10,
    )
    bad = observability_score_contrast_loss(
        torch.tensor([0.0]),
        torch.tensor([[2.0, 3.0, 1.0]]),
        pose_cost,
        valid,
        margin=0.1,
        min_negative_cost_m=0.10,
    )

    assert good["loss"] < bad["loss"]
    assert good["hard_negative_score"].item() == 1.0
    assert good["gap"].item() == 1.0


def test_observability_score_contrast_loss_handles_rows_without_hard_negatives():
    out = observability_score_contrast_loss(
        torch.tensor([1.0]),
        torch.tensor([[0.0, 0.5]]),
        torch.tensor([[0.01, 0.02]]),
        torch.ones(1, 2, dtype=torch.bool),
        margin=0.1,
        min_negative_cost_m=0.10,
    )

    assert out["active"].item() == 0.0
    assert out["loss"].item() == 0.0


def test_candidate_observability_margin_loss_separates_best_candidate_from_hard_negatives():
    scores_good = torch.tensor([[3.0, 1.0, 0.0]])
    scores_bad = torch.tensor([[0.0, 3.0, 1.0]])
    pose_cost = torch.tensor([[0.04, 0.30, 0.08]])
    valid = torch.ones_like(pose_cost, dtype=torch.bool)

    good = candidate_observability_margin_loss(
        scores_good,
        pose_cost,
        valid,
        margin=0.1,
        min_cost_gap_m=0.10,
    )
    bad = candidate_observability_margin_loss(
        scores_bad,
        pose_cost,
        valid,
        margin=0.1,
        min_cost_gap_m=0.10,
    )

    assert good["active"].item() == 1.0
    assert good["loss"] < bad["loss"]
    assert good["gap"] > bad["gap"]


def test_hard_flow_candidate_selection_mask_keeps_best_and_score_hard_negative():
    scores = torch.tensor([[0.0, 5.0, 4.0, 1.0], [3.0, 2.0, 1.0, 0.0]])
    pose_cost = torch.tensor([[0.05, 0.30, 0.40, 0.07], [0.20, 0.21, 0.22, 0.23]])
    valid = torch.ones_like(scores, dtype=torch.bool)

    out = hard_flow_candidate_selection_mask(
        scores,
        pose_cost,
        valid,
        mode="best_and_hard_negative",
        min_cost_gap_m=0.10,
    )

    expected = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
        ]
    )
    assert torch.equal(out["selection_mask"], expected)
    assert torch.equal(out["best_index"], torch.tensor([0, 0]))
    assert torch.equal(out["hard_negative_index"], torch.tensor([1, 0]))
    assert torch.isclose(out["hard_negative_active"], torch.tensor(0.5))


def test_pose_threshold_success_metrics_reports_refinement_buckets():
    trans_err = torch.tensor([0.04, 0.12, 0.30])
    rot_err = torch.tensor([1.0, 4.0, 12.0]) * torch.pi / 180.0

    metrics = pose_threshold_success_metrics(trans_err, rot_err, prefix="pred_")

    assert torch.isclose(metrics["pred_success_5cm_2deg"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(metrics["pred_success_10cm_5deg"], torch.tensor(1.0 / 3.0))
    assert torch.isclose(metrics["pred_success_25cm_10deg"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(metrics["pred_success_50cm_10deg"], torch.tensor(2.0 / 3.0))


def test_selected_candidate_pose_metrics_follow_actual_selector_index():
    pose_cost = torch.tensor([[0.10, 0.30], [0.40, 0.20]])
    trans_err = torch.tensor([[0.09, 0.29], [0.39, 0.19]])
    rot_err = torch.tensor([[0.01, 0.02], [0.03, 0.04]])
    oracle_cost = torch.tensor([0.10, 0.20])
    selected_idx = torch.tensor([1, 1])

    metrics = selected_candidate_pose_metrics(
        pose_cost,
        trans_err,
        rot_err,
        oracle_cost,
        selected_idx,
    )

    assert torch.allclose(metrics["selected_cost"], torch.tensor([0.30, 0.20]))
    assert torch.allclose(metrics["selected_trans"], torch.tensor([0.29, 0.19]))
    assert torch.allclose(metrics["selected_rot"], torch.tensor([0.02, 0.04]))
    assert torch.allclose(metrics["selected_oracle_gap"], torch.tensor([0.20, 0.00]))


def test_candidate_selection_bias_metrics_reports_identity_and_selected_margins():
    scores = torch.tensor([[0.0, 2.0, 1.5], [1.0, 0.0, 3.0]])
    pose_cost = torch.tensor([[0.30, 0.10, 0.40], [0.40, 0.20, 0.10]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    selected_idx = torch.tensor([2, 0])
    identity_mask = torch.tensor([[True, False, False], [True, False, False]])

    metrics = candidate_selection_bias_metrics(
        scores,
        pose_cost,
        valid,
        selected_idx,
        identity_mask=identity_mask,
    )

    assert torch.isclose(metrics["score_best_minus_score_identity"], torch.tensor(2.0))
    assert torch.isclose(metrics["score_best_minus_score_selected"], torch.tensor(1.25))
    assert torch.isclose(metrics["selected_identity_frac"], torch.tensor(0.5))


def test_parse_score_feature_hw_accepts_config_list_and_empty_values():
    assert parse_score_feature_hw([34, 60]) == (34, 60)
    assert parse_score_feature_hw("34,60") == (34, 60)
    assert parse_score_feature_hw("34x60") == (34, 60)
    assert parse_score_feature_hw(None) is None
    assert parse_score_feature_hw("") is None


def test_resize_score_candidate_tensors_downsamples_render_mask_and_weight():
    query = torch.randn(2, 3, 8, 10)
    render = torch.randn(2, 4, 3, 8, 10)
    mask = torch.ones(2, 4, 1, 8, 10)
    weight = torch.ones(2, 4, 1, 8, 10)

    q_out, r_out, m_out, w_out = resize_score_candidate_tensors(
        query,
        render,
        mask,
        weight,
        score_hw=(4, 5),
    )

    assert q_out.shape[-2:] == (8, 10)
    assert r_out.shape[-2:] == (4, 5)
    assert m_out.shape[-2:] == (4, 5)
    assert w_out.shape[-2:] == (4, 5)


def test_flow_geometry_for_score_hw_resizes_positions_and_scales_intrinsics():
    position = torch.randn(2, 3, 3, 8, 10)
    intrinsics = torch.tensor([[100.0, 80.0, 50.0, 40.0], [120.0, 90.0, 60.0, 45.0]])

    scaled_position, scaled_intrinsics = _flow_geometry_for_score_hw(
        position,
        intrinsics,
        source_hw=(8, 10),
        score_hw=(4, 5),
    )

    assert scaled_position.shape == (2, 3, 3, 4, 5)
    assert torch.allclose(scaled_intrinsics[:, 0], intrinsics[:, 0] * 0.5)
    assert torch.allclose(scaled_intrinsics[:, 1], intrinsics[:, 1] * 0.5)
    assert torch.allclose(scaled_intrinsics[:, 2], intrinsics[:, 2] * 0.5)
    assert torch.allclose(scaled_intrinsics[:, 3], intrinsics[:, 3] * 0.5)
    assert scaled_intrinsics.data_ptr() != intrinsics.data_ptr()


def test_scale_intrinsics_between_hw_supports_matrix_form():
    intrinsics = torch.eye(3).repeat(2, 1, 1)
    intrinsics[:, 0, 0] = 100.0
    intrinsics[:, 1, 1] = 80.0
    intrinsics[:, 0, 2] = 50.0
    intrinsics[:, 1, 2] = 40.0

    scaled = _scale_intrinsics_between_hw(intrinsics, (8, 10), (4, 5))

    assert torch.allclose(scaled[:, 0, 0], torch.full((2,), 50.0))
    assert torch.allclose(scaled[:, 1, 1], torch.full((2,), 40.0))
    assert torch.allclose(scaled[:, 0, 2], torch.full((2,), 25.0))
    assert torch.allclose(scaled[:, 1, 2], torch.full((2,), 20.0))


def test_rank_losses_pick_lowest_pose_cost_when_score_matches():
    scores = torch.tensor([[0.1, 0.8, 0.2]])
    pose_cost = torch.tensor([[0.3, 0.05, 0.2]])
    valid = torch.ones_like(scores, dtype=torch.bool)

    losses = rank_losses_from_scores(
        scores,
        pose_cost,
        valid,
        temperature_m=0.1,
        pairwise_weight=1.0,
        pairwise_min_gap_m=0.05,
        pairwise_logit_margin=0.1,
    )

    assert int(losses["pred_index"][0]) == 1
    assert int(losses["target_index"][0]) == 1
    assert float(losses["top1_acc"]) == 1.0


def test_pose_energy_score_monotonicity_loss_rewards_higher_updated_score():
    before = torch.tensor([1.0, 1.0])
    good_after = torch.tensor([1.3, 1.4])
    bad_after = torch.tensor([0.8, 0.7])
    before_cost = torch.tensor([0.30, 0.20])
    after_cost = torch.tensor([0.10, 0.25])

    good = pose_energy_score_monotonicity_loss(
        before,
        good_after,
        before_cost=before_cost,
        after_cost=after_cost,
        margin=0.05,
        improved_only=True,
    )
    bad = pose_energy_score_monotonicity_loss(
        before,
        bad_after,
        before_cost=before_cost,
        after_cost=after_cost,
        margin=0.05,
        improved_only=True,
    )

    assert good["active"].item() == 0.5
    assert good["loss"] < bad["loss"]
    assert good["score_gain"].item() > 0.0


def test_pose_energy_direction_pairwise_loss_penalizes_wrong_direction():
    init_pose = torch.eye(4).view(1, 4, 4)
    pose_gt = init_pose.clone()
    pose_gt[:, 0, 3] = -0.10
    candidate_pose = init_pose[:, None].repeat(1, 3, 1, 1)
    candidate_pose[:, 0, 0, 3] = -0.10
    candidate_pose[:, 1, 0, 3] = 0.10
    candidate_pose[:, 2, 1, 3] = 0.10
    target_index = torch.tensor([0])

    bad_logits = torch.tensor([[0.0, 2.0, 1.0]])
    good_logits = torch.tensor([[2.0, 0.0, 0.0]])

    bad = pose_energy_direction_pairwise_loss(
        bad_logits,
        candidate_pose,
        init_pose,
        pose_gt,
        target_index,
        min_cos_gap=0.25,
        logit_margin=0.5,
    )
    good = pose_energy_direction_pairwise_loss(
        good_logits,
        candidate_pose,
        init_pose,
        pose_gt,
        target_index,
        min_cos_gap=0.25,
        logit_margin=0.5,
    )

    assert bad["active"].item() > 0.0
    assert bad["loss"] > good["loss"]
    assert bad["pred_cos"] < good["pred_cos"]


def test_pose_energy_correction_cosine_soft_label_loss_prefers_positive_correction_direction():
    correction_cos = torch.tensor([[0.95, -0.95, 0.05]])
    valid = torch.ones_like(correction_cos, dtype=torch.bool)
    bad_logits = torch.tensor([[0.0, 3.0, 1.0]])
    good_logits = torch.tensor([[3.0, 0.0, 1.0]])

    bad = pose_energy_correction_cosine_soft_label_loss(
        bad_logits,
        correction_cos,
        valid_mask=valid,
        target_temperature=0.1,
        min_cos=0.0,
    )
    good = pose_energy_correction_cosine_soft_label_loss(
        good_logits,
        correction_cos,
        valid_mask=valid,
        target_temperature=0.1,
        min_cos=0.0,
    )

    assert int(bad["target_index"][0]) == 0
    assert bad["loss"] > good["loss"]
    assert bad["pred_cos"] < good["pred_cos"]


def test_score_pose_improvement_soft_label_loss_prefers_candidates_that_reduce_pose_cost():
    init_cost = torch.tensor([[0.40]])
    candidate_cost = torch.tensor([[0.10, 0.45, 0.32]])
    valid = torch.ones_like(candidate_cost, dtype=torch.bool)
    bad_logits = torch.tensor([[0.0, 3.0, 1.0]])
    good_logits = torch.tensor([[3.0, 0.0, 1.0]])

    bad = score_pose_improvement_soft_label_loss(
        bad_logits,
        candidate_cost,
        init_cost,
        valid_mask=valid,
        target_temperature_m=0.05,
        min_improvement_m=0.0,
    )
    good = score_pose_improvement_soft_label_loss(
        good_logits,
        candidate_cost,
        init_cost,
        valid_mask=valid,
        target_temperature_m=0.05,
        min_improvement_m=0.0,
    )

    assert int(bad["target_index"][0]) == 0
    assert bad["loss"] > good["loss"]
    assert bad["pred_improvement_m"] < good["pred_improvement_m"]


def test_candidate_teacher_quality_scores_from_render_loftr_pnp_fields():
    valid = torch.tensor([[True, True, True]])
    batch = {
        "retrieval_pnp_success_candidates": torch.tensor([[1.0, 1.0, 0.0]]),
        "retrieval_pnp_num_inliers_candidates": torch.tensor([[120.0, 8.0, 0.0]]),
        "retrieval_pnp_num_matches_candidates": torch.tensor([[160.0, 30.0, 5.0]]),
        "retrieval_pnp_reproj_median_candidates": torch.tensor([[1.0, 9.0, float("inf")]]),
        "retrieval_pnp_inlier_ratio_candidates": torch.tensor([[0.75, 0.1, 0.0]]),
        "retrieval_pnp_inlier_conf_mean_candidates": torch.tensor([[0.8, 0.3, 0.0]]),
    }

    quality, active = candidate_teacher_quality_scores_from_batch(
        batch,
        valid_mask=valid,
        target_mode="pnp_composite",
    )

    assert active.tolist() == [[True, True, True]]
    assert quality[0, 0] > quality[0, 1] > quality[0, 2]


def test_candidate_teacher_quality_listwise_loss_prefers_high_quality_candidate():
    valid = torch.tensor([[True, True, True]])
    batch = {
        "retrieval_pnp_success_candidates": torch.tensor([[1.0, 1.0, 0.0]]),
        "retrieval_pnp_num_inliers_candidates": torch.tensor([[4.0, 80.0, 2.0]]),
        "retrieval_pnp_num_matches_candidates": torch.tensor([[20.0, 120.0, 10.0]]),
        "retrieval_pnp_inlier_ratio_candidates": torch.tensor([[0.2, 0.9, 0.1]]),
        "retrieval_pnp_inlier_conf_mean_candidates": torch.tensor([[0.3, 0.8, 0.1]]),
    }
    bad_logits = torch.tensor([[3.0, 0.0, 1.0]], requires_grad=True)
    good_logits = torch.tensor([[0.0, 3.0, 1.0]], requires_grad=True)

    bad = candidate_teacher_quality_listwise_loss(
        bad_logits,
        batch,
        valid_mask=valid,
        temperature=0.5,
        pairwise_weight=0.5,
        pairwise_min_gap=0.25,
    )
    good = candidate_teacher_quality_listwise_loss(
        good_logits,
        batch,
        valid_mask=valid,
        temperature=0.5,
        pairwise_weight=0.5,
        pairwise_min_gap=0.25,
    )

    assert int(bad["target_index"][0]) == 1
    assert bad["loss"] > good["loss"]
    assert bad["pred_quality"] < good["pred_quality"]
    bad["loss"].backward()
    assert bad_logits.grad is not None
    assert torch.isfinite(bad_logits.grad).all()


def test_effective_candidate_teacher_quality_weight_supports_delayed_warmup():
    args = type(
        "Args",
        (),
        {
            "candidate_teacher_quality_weight": 2.0,
            "candidate_teacher_quality_start_step": 10,
            "candidate_teacher_quality_warmup_steps": 20,
            "current_step": 0,
        },
    )()

    assert effective_candidate_teacher_quality_weight(args) == 0.0
    args.current_step = 20
    assert effective_candidate_teacher_quality_weight(args) == 1.0
    args.current_step = 40
    assert effective_candidate_teacher_quality_weight(args) == 2.0


def test_score_anti_identity_loss_penalizes_identity_when_better_candidate_exists():
    candidate_cost = torch.tensor([[0.30, 0.10, 0.45]])
    valid = torch.ones_like(candidate_cost, dtype=torch.bool)
    bad_scores = torch.tensor([[3.0, 0.0, 1.0]])
    good_scores = torch.tensor([[0.0, 3.0, 1.0]])

    bad = score_anti_identity_loss(
        bad_scores,
        candidate_cost,
        valid_mask=valid,
        identity_index=0,
        min_gap_m=0.03,
        logit_margin=0.5,
    )
    good = score_anti_identity_loss(
        good_scores,
        candidate_cost,
        valid_mask=valid,
        identity_index=0,
        min_gap_m=0.03,
        logit_margin=0.5,
    )

    assert bad["active"].item() > 0.0
    assert bad["loss"] > good["loss"]
    assert bad["selected_identity_frac"] > good["selected_identity_frac"]


def test_nvs_teacher_correspondence_loss_prefers_teacher_matches():
    query = torch.zeros(1, 3, 2, 3)
    render = torch.zeros(1, 3, 2, 3)
    query[0, 0, 0, 0] = 1.0
    query[0, 1, 1, 2] = 1.0
    render.copy_(query)
    query_xy = torch.tensor([[[0.0, 0.0], [2.0, 1.0]]])
    map_xy_good = torch.tensor([[[0.0, 0.0], [2.0, 1.0]]])
    map_xy_bad = torch.tensor([[[2.0, 1.0], [0.0, 0.0]]])
    conf = torch.ones(1, 2)
    valid = torch.ones(1, 2)
    source_hw = torch.tensor([[2.0, 3.0]])

    good_loss, good_metrics = nvs_teacher_correspondence_loss(
        query,
        render,
        {
            "teacher_corr_query_xy": query_xy,
            "teacher_corr_map_xy": map_xy_good,
            "teacher_corr_conf": conf,
            "teacher_corr_valid": valid,
            "teacher_corr_hw": source_hw,
        },
        weight=1.0,
        patch_weight=1.0,
        temperature=0.05,
        min_points=2,
        patch_radius=1,
        patch_temperature=0.05,
    )
    bad_loss, bad_metrics = nvs_teacher_correspondence_loss(
        query,
        render,
        {
            "teacher_corr_query_xy": query_xy,
            "teacher_corr_map_xy": map_xy_bad,
            "teacher_corr_conf": conf,
            "teacher_corr_valid": valid,
            "teacher_corr_hw": source_hw,
        },
        weight=1.0,
        patch_weight=1.0,
        temperature=0.05,
        min_points=2,
        patch_radius=1,
        patch_temperature=0.05,
    )

    assert good_loss < bad_loss
    assert good_metrics["nvs_teacher_corr_missing"].item() == 0.0
    assert good_metrics["nvs_teacher_corr_acc"].item() == 1.0
    assert bad_metrics["nvs_teacher_corr_acc"].item() == 0.0


def test_nvs_teacher_pair_match_loss_prefers_teacher_center_patch_and_backprops():
    matcher = PairConditionedLocalMatcher(
        channels=3,
        hidden_dim=8,
        offset_radius=1,
        zero_init_residual=True,
        base_dot_weight=10.0,
    )
    query = torch.zeros(1, 3, 3, 3)
    render = torch.zeros(1, 3, 3, 3)
    query[0, 0, 1, 1] = 1.0
    render[0, 0, 1, 1] = 1.0
    render[0, 1, 1, 0] = 1.0
    query_xy = torch.tensor([[[1.0, 1.0]]])
    good_map_xy = torch.tensor([[[1.0, 1.0]]])
    bad_map_xy = torch.tensor([[[0.0, 1.0]]])
    conf = torch.ones(1, 1)
    valid = torch.ones(1, 1)
    source_hw = torch.tensor([[3.0, 3.0]])

    good_loss, good_metrics = nvs_teacher_pair_match_loss(
        matcher,
        query,
        render,
        {
            "teacher_corr_query_xy": query_xy,
            "teacher_corr_map_xy": good_map_xy,
            "teacher_corr_conf": conf,
            "teacher_corr_valid": valid,
            "teacher_corr_hw": source_hw,
        },
        weight=1.0,
        radius=1,
        temperature=0.05,
        min_points=1,
        positive_weight=0.5,
    )
    bad_loss, bad_metrics = nvs_teacher_pair_match_loss(
        matcher,
        query,
        render,
        {
            "teacher_corr_query_xy": query_xy,
            "teacher_corr_map_xy": bad_map_xy,
            "teacher_corr_conf": conf,
            "teacher_corr_valid": valid,
            "teacher_corr_hw": source_hw,
        },
        weight=1.0,
        radius=1,
        temperature=0.05,
        min_points=1,
        positive_weight=0.5,
    )

    good_loss.backward()

    assert good_loss < bad_loss
    assert good_loss.item() >= 0.0
    assert good_metrics["nvs_pair_match_acc"].item() == 1.0
    assert bad_metrics["nvs_pair_match_acc"].item() == 0.0
    assert any(param.grad is not None for param in matcher.parameters())


def test_local_zero_offset_scores_can_use_continuous_reliability_weights():
    corr = torch.tensor([[[[[10.0, 0.0]]], [[[0.0, 5.0]]]]])
    valid = torch.ones(1, 2, 1, 1, 2, dtype=torch.bool)
    weight = torch.tensor([[[[0.1, 1.0]], [[0.1, 1.0]]]])

    unweighted, _ = local_zero_offset_scores_from_corr(
        corr,
        valid,
        radius=0,
        temperature=0.1,
        peak_gap_weight=0.0,
        offset_weight=0.0,
    )
    weighted, _ = local_zero_offset_scores_from_corr(
        corr,
        valid,
        radius=0,
        temperature=0.1,
        peak_gap_weight=0.0,
        offset_weight=0.0,
        weight=weight,
    )

    assert unweighted[0, 0] > unweighted[0, 1]
    assert weighted[0, 1] > weighted[0, 0]


def test_pair_matcher_heatmap_score_maps_prefer_aligned_candidate():
    matcher = PairConditionedLocalMatcher(
        channels=3,
        hidden_dim=8,
        offset_radius=1,
        zero_init_residual=True,
        base_dot_weight=10.0,
    )
    query = torch.zeros(1, 3, 5, 5)
    query[0, 0, 1:4, 1:4] = 1.0
    query[0, 1, 2, 2] = 1.0
    query[0, 2, 0, 4] = 1.0
    aligned = query.clone()
    shifted = torch.roll(query, shifts=1, dims=-1)
    render = torch.stack([aligned, shifted], dim=1)

    score_maps, valid = pair_matcher_local_candidate_score_maps(
        matcher,
        query,
        render,
        radius=1,
        stride=1,
        temperature=0.05,
        chunk_points=64,
        candidate_score_mode="center_logprob_margin",
    )

    assert score_maps.shape == (1, 2, 3, 5, 5)
    assert valid.shape == (1, 2, 5, 5)
    scores = (score_maps[:, :, 0] * valid.float()).flatten(2).sum(dim=2) / valid.float().flatten(2).sum(dim=2)
    assert scores[0, 0] > scores[0, 1]


def test_pair_matcher_heatmap_candidate_chunking_matches_full_result():
    matcher = PairConditionedLocalMatcher(
        channels=3,
        hidden_dim=8,
        offset_radius=1,
        zero_init_residual=False,
        base_dot_weight=2.0,
    )
    query = torch.randn(2, 3, 5, 5)
    render = torch.randn(2, 5, 3, 5, 5)
    mask = torch.ones(2, 5, 5, 5, dtype=torch.bool)

    full_maps, full_valid = pair_matcher_local_candidate_score_maps(
        matcher,
        query,
        render,
        mask=mask,
        radius=1,
        stride=1,
        temperature=0.05,
        chunk_points=512,
        candidate_chunk_size=0,
    )
    chunked_maps, chunked_valid = pair_matcher_local_candidate_score_maps(
        matcher,
        query,
        render,
        mask=mask,
        radius=1,
        stride=1,
        temperature=0.05,
        chunk_points=512,
        candidate_chunk_size=2,
    )

    assert torch.equal(chunked_valid, full_valid)
    assert torch.allclose(chunked_maps, full_maps, atol=1.0e-6)


def test_nvs_best_metric_defaults_to_pose_energy_metric_when_enabled():
    args = type(
        "Args",
        (),
        {
            "best_metric": None,
            "best_metric_mode": None,
        },
    )()

    resolved = apply_config_defaults(
        args,
        {"nvs_pose_feature_adapter": {"pose_energy_enabled": True}},
    )

    assert resolved.best_metric == "pose_energy_pred_cost_m"
    assert resolved.best_metric_mode == "min"


def test_nvs_config_overrides_pose_energy_defaults_unless_cli_set():
    args = type(
        "Args",
        (),
        {
            "best_metric": None,
            "best_metric_mode": None,
            "synthetic_ratio": None,
            "topk": None,
            "lattice_trans_cm": None,
            "lattice_rot_deg": None,
            "lattice_direction_mode": None,
        },
    )()

    resolved = apply_config_defaults(
        args,
        {
            "pose_energy": {"synthetic_ratio": 0.5},
            "nvs_pose_feature_adapter": {
                "synthetic_ratio": 0.0,
                "topk": 8,
                "lattice_trans_cm": [0, 5, 10],
                "lattice_rot_deg": [0, 1, 2],
                "lattice_direction_mode": "axis",
            },
        },
    )

    assert resolved.synthetic_ratio == 0.0
    assert resolved.topk == 8
    assert resolved.lattice_trans_cm == [0, 5, 10]
    assert resolved.lattice_rot_deg == [0, 1, 2]
    assert resolved.lattice_direction_mode == "axis"


def test_nvs_config_enables_pose_observability_diagnostics():
    args = type("Args", (), {})()

    resolved = apply_config_defaults(
        args,
        {"nvs_pose_feature_adapter": {"pose_observability_diagnostic_enabled": True}},
    )

    assert resolved.pose_observability_diagnostic_enabled is True


def test_nvs_parser_defers_training_defaults_to_config(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--config", "config.yaml", "--checkpoint", "model.pth", "--out-dir", "out"],
    )

    args = parse_nvs_pose_feature_adapter_args()
    resolved = apply_config_defaults(
        args,
        {"training": {"batch_size": 6, "max_steps": 7, "eval_every": 2, "save_every": 3}},
    )

    assert resolved.batch_size == 6
    assert resolved.max_steps == 7
    assert resolved.eval_every == 2
    assert resolved.save_every == 3


def test_variance_floor_loss_penalizes_collapse():
    collapsed = torch.ones(2, 4, 5, 5)
    varied = torch.randn(2, 4, 5, 5)

    assert variance_floor_loss(collapsed, 0.1) > variance_floor_loss(varied, 0.1)


def test_nvs_candidate_bank_can_prepend_gt_and_drop_identity_shortcut():
    args = type(
        "Args",
        (),
        {
            "topk": 4,
            "lattice_trans_cm": "5,10",
            "lattice_rot_deg": "1",
            "include_identity_candidate": False,
            "limit_strategy": "first",
            "combine_trans_rot": False,
            "lattice_direction_mode": "axis",
            "train_append_gt_candidate": True,
            "eval_append_gt_candidate": False,
            "candidate_bank_mode": "lattice",
        },
    )()
    init_pose = torch.eye(4).view(1, 4, 4)
    pose_gt = init_pose.clone()
    pose_gt[:, 0, 3] = 0.123

    train_bank = build_nvs_candidate_bank(init_pose, pose_gt, args, train=True)
    eval_bank = build_nvs_candidate_bank(init_pose, pose_gt, args, train=False)

    assert torch.allclose(train_bank[:, 0], pose_gt)
    assert not torch.allclose(eval_bank[:, 0], pose_gt)


def test_nvs_balanced_candidate_bank_contains_identity_trans_rot_and_joint():
    args = type(
        "Args",
        (),
        {
            "topk": 16,
            "lattice_trans_cm": "10",
            "lattice_rot_deg": "5",
            "include_identity_candidate": True,
            "limit_strategy": "first",
            "combine_trans_rot": False,
            "lattice_direction_mode": "axis",
            "train_append_gt_candidate": False,
            "eval_append_gt_candidate": False,
            "candidate_bank_mode": "balanced",
        },
    )()
    init_pose = torch.eye(4).view(1, 4, 4)
    pose_gt = init_pose.clone()

    bank = build_nvs_candidate_bank(init_pose, pose_gt, args, train=True)
    delta_t = torch.linalg.norm(bank[:, :, :3, 3] - init_pose[:, None, :3, 3], dim=-1)
    rot_trace = bank[:, :, 0, 0] + bank[:, :, 1, 1] + bank[:, :, 2, 2]
    delta_r = torch.acos(torch.clamp((rot_trace - 1.0) * 0.5, -1.0, 1.0))

    identity = (delta_t < 1e-6) & (delta_r < 1e-4)
    trans_only = (delta_t > 0.05) & (delta_r < 1e-4)
    rot_only = (delta_t < 1e-6) & (delta_r > 0.01)
    joint = (delta_t > 0.05) & (delta_r > 0.01)

    assert bank.shape[1] <= args.topk
    assert identity.any()
    assert trans_only.any()
    assert rot_only.any()
    assert joint.any()


def test_nvs_direction_balanced_candidate_bank_adds_gt_direction_hard_negatives_without_exact_gt():
    args = type(
        "Args",
        (),
        {
            "topk": 16,
            "lattice_trans_cm": "5,10",
            "lattice_rot_deg": "1,2",
            "include_identity_candidate": True,
            "limit_strategy": "first",
            "combine_trans_rot": False,
            "lattice_direction_mode": "axis",
            "train_append_gt_candidate": False,
            "eval_append_gt_candidate": False,
            "candidate_bank_mode": "direction_balanced",
        },
    )()
    init_pose = torch.eye(4).view(1, 4, 4)
    pose_gt = init_pose.clone()
    pose_gt[:, 0, 3] = 0.20

    train_bank = build_nvs_candidate_bank(init_pose, pose_gt, args, train=True)
    eval_bank = build_nvs_candidate_bank(init_pose, pose_gt, args, train=False)
    target_translation = pose_gt[:, 0, 3]
    train_translation = train_bank[0, :, 0, 3]
    eval_translation = eval_bank[0, :, 0, 3]

    assert not torch.isclose(train_translation, target_translation[0], atol=1e-5).any()
    assert torch.isclose(train_translation, target_translation[0] * 0.5, atol=1e-5).any()
    assert (train_translation < -0.05).any()
    assert not torch.isclose(eval_translation, target_translation[0], atol=1e-5).any()


def test_nvs_adaptive_direction_balanced_candidate_bank_adds_same_magnitude_direction_pairs():
    args = type(
        "Args",
        (),
        {
            "topk": 48,
            "lattice_trans_cm": "0,2,5,10,25,50,100",
            "lattice_rot_deg": "0,0.5,1,2,5,10,20",
            "include_identity_candidate": True,
            "limit_strategy": "uniform",
            "combine_trans_rot": False,
            "lattice_direction_mode": "cube",
            "train_append_gt_candidate": False,
            "eval_append_gt_candidate": False,
            "candidate_bank_mode": "adaptive_direction_balanced",
            "direction_fractions": "0.75,0.5,0.25",
        },
    )()
    init_pose = _w2c_pose_from_center_and_yaw([0.0, 0.0, 0.0], 0.0).view(1, 4, 4)
    pose_gt = _w2c_pose_from_center_and_yaw([0.20, 0.0, 0.0], 0.0).view(1, 4, 4)

    train_bank = build_nvs_candidate_bank(init_pose, pose_gt, args, train=True)
    correction_cos = candidate_correction_cosines(train_bank, init_pose, pose_gt)[0]
    init_center = torch.zeros(3)
    cand_centers = train_bank[0, :, :3, :3].transpose(-1, -2).neg() @ train_bank[0, :, :3, 3:4]
    cand_delta_norm = torch.linalg.norm(cand_centers.squeeze(-1) - init_center, dim=-1)

    assert (correction_cos > 0.99).any()
    assert (correction_cos < -0.99).any()
    assert torch.isclose(cand_delta_norm, torch.tensor(0.15), atol=1e-4).any()
    assert not torch.allclose(train_bank[:, 0], pose_gt)


def test_candidate_correction_cosines_reports_direction_relative_to_init_pose():
    init_pose = _w2c_pose_from_center_and_yaw([0.0, 0.0, 0.0], 0.0).view(1, 4, 4)
    pose_gt = _w2c_pose_from_center_and_yaw([1.0, 0.0, 0.0], 0.0).view(1, 4, 4)
    candidate_pose = torch.stack(
        [
            _w2c_pose_from_center_and_yaw([0.5, 0.0, 0.0], 0.0),
            _w2c_pose_from_center_and_yaw([-0.5, 0.0, 0.0], 0.0),
            _w2c_pose_from_center_and_yaw([0.0, 0.0, 0.0], 0.0),
        ],
        dim=0,
    ).view(1, 3, 4, 4)

    correction_cos = candidate_correction_cosines(candidate_pose, init_pose, pose_gt)

    assert correction_cos[0, 0] > 0.99
    assert correction_cos[0, 1] < -0.99
    assert correction_cos[0, 2].abs() < 1e-6


def test_candidate_identity_mask_detects_identity_without_assuming_first_candidate():
    init_pose = _w2c_pose_from_center_and_yaw([0.0, 0.0, 0.0], 0.0).view(1, 4, 4)
    bank = torch.stack(
        [
            _w2c_pose_from_center_and_yaw([0.10, 0.0, 0.0], 0.0),
            _w2c_pose_from_center_and_yaw([0.0, 0.0, 0.0], 0.0),
            _w2c_pose_from_center_and_yaw([0.0, 0.0, 0.0], 2.0),
        ],
        dim=0,
    ).view(1, 3, 4, 4)
    identity = candidate_identity_mask(bank, init_pose)

    assert identity.any()
    assert identity.tolist() == [[False, True, False]]


def test_nvs_adaptive_candidate_bank_scales_lattice_to_init_error_bucket():
    args = type(
        "Args",
        (),
        {
            "topk": 128,
            "lattice_trans_cm": "0,2,5,10,25,50,100",
            "lattice_rot_deg": "0,0.5,1,2,5,10,20",
            "include_identity_candidate": True,
            "limit_strategy": "uniform",
            "combine_trans_rot": False,
            "lattice_direction_mode": "cube",
            "train_append_gt_candidate": False,
            "eval_append_gt_candidate": False,
            "candidate_bank_mode": "adaptive_balanced",
        },
    )()
    init_pose = torch.eye(4).view(1, 4, 4)
    pose_gt_small = init_pose.clone()
    pose_gt_small[:, 0, 3] = -0.10
    pose_gt_medium = init_pose.clone()
    pose_gt_medium[:, 0, 3] = -0.50

    small_bank = build_nvs_candidate_bank(init_pose, pose_gt_small, args, train=True)
    medium_bank = build_nvs_candidate_bank(init_pose, pose_gt_medium, args, train=True)
    small_delta = torch.linalg.norm(small_bank[:, :, :3, 3] - init_pose[:, None, :3, 3], dim=-1)
    medium_delta = torch.linalg.norm(medium_bank[:, :, :3, 3] - init_pose[:, None, :3, 3], dim=-1)

    assert small_delta.max().item() <= 0.251
    assert medium_delta.max().item() >= 0.49


def test_nvs_train_parser_accepts_adaptive_balanced_candidate_bank(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_nvs_pose_feature_adapter.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pth",
            "--out-dir",
            "out",
            "--candidate-bank-mode",
            "adaptive_balanced",
        ],
    )

    args = parse_nvs_pose_feature_adapter_args()

    assert args.candidate_bank_mode == "adaptive_balanced"


def test_nvs_train_parser_accepts_adaptive_direction_balanced_candidate_bank(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_nvs_pose_feature_adapter.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pth",
            "--out-dir",
            "out",
            "--candidate-bank-mode",
            "adaptive_direction_balanced",
        ],
    )

    args = parse_nvs_pose_feature_adapter_args()

    assert args.candidate_bank_mode == "adaptive_direction_balanced"


def test_nvs_train_parser_accepts_score_correction_cosine_loss_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_nvs_pose_feature_adapter.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pth",
            "--out-dir",
            "out",
            "--score-correction-cosine-weight",
            "2.0",
            "--score-correction-cosine-temperature",
            "0.07",
            "--score-correction-cosine-min-cos",
            "0.2",
            "--score-pose-improvement-weight",
            "1.5",
            "--score-pose-improvement-temperature-m",
            "0.04",
            "--score-pose-improvement-min-improvement-m",
            "0.01",
            "--score-anti-identity-weight",
            "0.7",
            "--score-anti-identity-min-gap-m",
            "0.02",
            "--score-anti-identity-logit-margin",
            "0.4",
            "--score-anti-identity-index",
            "2",
            "--score-use-uncertainty",
        ],
    )

    args = parse_nvs_pose_feature_adapter_args()

    assert args.score_correction_cosine_weight == 2.0
    assert args.score_correction_cosine_temperature == 0.07
    assert args.score_correction_cosine_min_cos == 0.2
    assert args.score_pose_improvement_weight == 1.5
    assert args.score_pose_improvement_temperature_m == 0.04
    assert args.score_pose_improvement_min_improvement_m == 0.01
    assert args.score_anti_identity_weight == 0.7
    assert args.score_anti_identity_min_gap_m == 0.02
    assert args.score_anti_identity_logit_margin == 0.4
    assert args.score_anti_identity_index == 2
    assert args.score_use_uncertainty is True


def test_nvs_train_parser_accepts_pofd_observability_args(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_nvs_pose_feature_adapter.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "checkpoint.pth",
            "--out-dir",
            "out",
            "--observability-contrast-weight",
            "1.5",
            "--observability-contrast-margin",
            "0.07",
            "--observability-negative-min-cost-m",
            "0.12",
            "--observability-contrast-score-source",
            "selection_score",
        ],
    )

    args = parse_nvs_pose_feature_adapter_args()

    assert args.observability_contrast_weight == 1.5
    assert args.observability_contrast_margin == 0.07
    assert args.observability_negative_min_cost_m == 0.12
    assert args.observability_contrast_score_source == "selection_score"


def test_eval_pose_energy_candidate_bank_can_use_bucket_adaptive_lattice():
    args = type(
        "Args",
        (),
        {
            "topk": 128,
            "lattice_trans_cm": "0,2,5,10,25,50,100",
            "lattice_rot_deg": "0,0.5,1,2,5,10,20",
            "include_identity_candidate": True,
            "limit_strategy": "uniform",
            "combine_trans_rot": False,
            "lattice_direction_mode": "cube",
            "candidate_bank_mode": "adaptive_balanced",
        },
    )()
    init_pose = torch.eye(4).view(1, 4, 4)

    small_bank = build_eval_pose_energy_candidate_bank(init_pose, args, trans_cm=10.0, rot_deg=2.0)
    medium_bank = build_eval_pose_energy_candidate_bank(init_pose, args, trans_cm=50.0, rot_deg=10.0)
    small_delta = torch.linalg.norm(small_bank[:, :, :3, 3] - init_pose[:, None, :3, 3], dim=-1)
    medium_delta = torch.linalg.norm(medium_bank[:, :, :3, 3] - init_pose[:, None, :3, 3], dim=-1)

    assert small_delta.max().item() <= 0.251
    assert medium_delta.max().item() >= 0.49


def test_eval_pose_energy_balanced_candidate_bank_includes_joint_candidates():
    args = type(
        "Args",
        (),
        {
            "topk": 16,
            "lattice_trans_cm": "10",
            "lattice_rot_deg": "5",
            "limit_strategy": "first",
            "combine_trans_rot": False,
            "lattice_direction_mode": "axis",
            "candidate_bank_mode": "balanced",
        },
    )()
    init_pose = torch.eye(4).view(1, 4, 4)

    bank = build_eval_pose_energy_candidate_bank(init_pose, args)
    delta_t = torch.linalg.norm(bank[:, :, :3, 3] - init_pose[:, None, :3, 3], dim=-1)
    rot_trace = bank[:, :, 0, 0] + bank[:, :, 1, 1] + bank[:, :, 2, 2]
    delta_r = torch.acos(torch.clamp((rot_trace - 1.0) * 0.5, -1.0, 1.0))

    assert ((delta_t > 0.05) & (delta_r > 0.01)).any()


def test_project_world_positions_identity_pose_to_feature_grid():
    position = torch.tensor(
        [[
            [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        ]]
    )
    pose = torch.eye(4).view(1, 4, 4)
    intrinsics = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    grid, valid, depth = project_world_positions_to_feature_grid(
        position,
        pose,
        intrinsics,
        feature_hw=(2, 2),
    )

    expected_grid = torch.tensor([[[[[-1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [1.0, 1.0]]]]])
    assert torch.allclose(grid, expected_grid, atol=1.0e-5)
    assert valid.all()
    assert torch.allclose(depth, torch.ones(1, 1, 1, 2, 2))


def test_warped_candidate_alignment_loss_uses_projected_correspondences():
    query = torch.tensor(
        [[
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]]
    )
    render_good = query[:, None].clone()
    render_bad = -render_good
    position = torch.tensor(
        [[
            [
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        ]]
    )
    pose = torch.eye(4).view(1, 4, 4)
    intrinsics = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = torch.ones(1, 1, 1, 2, 2)
    depth = torch.ones(1, 1, 1, 2, 2)

    good_loss, good_metrics = warped_candidate_alignment_loss(
        query,
        render_good,
        position,
        pose,
        intrinsics,
        candidate_mask=mask,
        target_depth=depth,
        target_mask=depth,
    )
    bad_loss, _bad_metrics = warped_candidate_alignment_loss(
        query,
        render_bad,
        position,
        pose,
        intrinsics,
        candidate_mask=mask,
        target_depth=depth,
        target_mask=torch.ones_like(depth),
    )

    assert good_loss < bad_loss
    assert good_metrics["warp_valid_frac"] > 0.99


def test_local_zero_offset_correlation_scores_prefer_aligned_candidate():
    torch.manual_seed(7)
    query = torch.randn(1, 6, 5, 5)
    aligned = query.clone()
    shifted = torch.roll(query, shifts=1, dims=-1)
    render = torch.stack([aligned, shifted], dim=1)

    scores, stats = local_zero_offset_correlation_scores(
        query,
        render,
        radius=1,
        temperature=0.05,
        peak_gap_weight=0.5,
        offset_weight=0.1,
    )

    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]
    assert stats["local_peak_offset_px"] >= 0.0


def test_local_offset_only_scores_are_available():
    torch.manual_seed(11)
    query = torch.randn(1, 4, 4, 4)
    render = torch.stack([query, torch.roll(query, shifts=1, dims=-1)], dim=1)

    expected_scores, _ = local_zero_offset_correlation_scores(
        query,
        render,
        radius=1,
        score_mode="local_neg_expected_offset",
    )
    peak_scores, _ = local_zero_offset_correlation_scores(
        query,
        render,
        radius=1,
        score_mode="local_neg_peak_offset",
    )

    assert expected_scores.shape == (1, 2)
    assert peak_scores.shape == (1, 2)
    assert torch.isfinite(expected_scores).all()
    assert torch.isfinite(peak_scores).all()


def test_pair_matcher_local_candidate_scores_prefer_aligned_candidate():
    torch.manual_seed(17)
    query = torch.randn(1, 5, 5, 5)
    aligned = query.clone()
    shifted = torch.roll(query, shifts=1, dims=-1)
    render = torch.stack([aligned, shifted], dim=1)
    matcher = PairConditionedLocalMatcher(
        channels=5,
        hidden_dim=8,
        offset_radius=1,
        zero_init_residual=True,
        base_dot_weight=10.0,
    )

    scores, stats = pair_matcher_local_candidate_scores(
        matcher,
        query,
        render,
        radius=1,
        stride=1,
        temperature=0.05,
        chunk_points=16,
    )

    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]
    assert stats["local_peak_offset_px"] >= 0.0


def test_pair_matcher_local_candidate_scores_offset_chunk_matches_full():
    torch.manual_seed(19)
    query = torch.randn(1, 5, 5, 6)
    render = torch.stack([query.clone(), torch.roll(query, shifts=1, dims=-1)], dim=1)
    matcher = PairConditionedLocalMatcher(
        channels=5,
        hidden_dim=8,
        offset_radius=2,
        zero_init_residual=False,
        base_dot_weight=3.0,
    )

    full_scores, full_stats = pair_matcher_local_candidate_scores(
        matcher,
        query,
        render,
        radius=2,
        stride=1,
        temperature=0.07,
        chunk_points=8,
        offset_chunk_size=0,
    )
    chunked_scores, chunked_stats = pair_matcher_local_candidate_scores(
        matcher,
        query,
        render,
        radius=2,
        stride=1,
        temperature=0.07,
        chunk_points=8,
        offset_chunk_size=5,
    )

    assert torch.allclose(chunked_scores, full_scores, atol=1.0e-5, rtol=1.0e-5)
    assert torch.allclose(
        chunked_stats["local_peak_gap"],
        full_stats["local_peak_gap"],
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_local_flow_nce_loss_accepts_zero_offset_identity_projection():
    query = torch.randn(1, 5, 3, 3)
    render = query[:, None].clone()
    xs = torch.arange(3, dtype=torch.float32).view(1, 1, 1, 3).expand(1, 1, 3, 3)
    ys = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1).expand(1, 1, 3, 3)
    zs = torch.ones_like(xs)
    position = torch.stack([xs, ys, zs], dim=2)
    pose = torch.eye(4).view(1, 4, 4)
    intrinsics = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = torch.ones(1, 1, 1, 3, 3)
    depth = torch.ones(1, 1, 1, 3, 3)

    loss, metrics = local_flow_nce_loss(
        query,
        render,
        position,
        pose,
        intrinsics,
        candidate_mask=mask,
        target_depth=depth,
        target_mask=depth,
        radius=1,
        temperature=0.05,
    )

    assert torch.isfinite(loss)
    assert metrics["local_flow_valid_frac"] > 0.99
    assert metrics["local_flow_target_offset_px"] < 0.01


def test_local_flow_nce_loss_from_corr_filters_selected_candidates():
    corr = torch.zeros(1, 2, 9, 2, 2)
    xs = torch.arange(2, dtype=torch.float32).view(1, 1, 1, 2).expand(1, 2, 2, 2)
    ys = torch.arange(2, dtype=torch.float32).view(1, 1, 2, 1).expand(1, 2, 2, 2)
    zs = torch.ones_like(xs)
    position = torch.stack([xs, ys, zs], dim=2)
    pose = torch.eye(4).view(1, 4, 4)
    intrinsics = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = torch.ones(1, 2, 1, 2, 2)
    depth = torch.ones(1, 1, 1, 2, 2)

    _loss, metrics = local_flow_nce_loss_from_corr(
        corr,
        torch.ones_like(corr, dtype=torch.bool),
        position,
        pose,
        intrinsics,
        candidate_mask=mask,
        target_depth=depth,
        target_mask=depth,
        candidate_selection_mask=torch.tensor([[True, False]]),
        radius=1,
        temperature=0.05,
    )

    assert torch.isclose(metrics["local_flow_valid_frac"], torch.tensor(0.5))
    assert torch.isclose(metrics["local_flow_candidate_selected_frac"], torch.tensor(0.5))


def test_nvs_checkpoint_roundtrips_energy_net_and_trainable_query_prefix(tmp_path):
    class TinyQueryModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.local_corr_projector = torch.nn.Conv2d(3, 3, kernel_size=1)
            self.fine_head = torch.nn.Conv2d(3, 3, kernel_size=1)
            self.unrelated = torch.nn.Conv2d(3, 3, kernel_size=1)

    args = type(
        "Args",
        (),
        {
            "train_projector": True,
            "train_model_prefixes": "fine_head.",
            "pose_energy_use_delta_vector": False,
            "pose_energy_use_center_delta_vector": False,
            "pose_energy_use_rgb": False,
            "pose_energy_use_uncertainty": False,
        },
    )()
    model = TinyQueryModel()
    adapter = PoseFeatureDomainAdapter(channels=3, hidden_dim=4)
    energy_net = PoseEnergyNet(vector_dim=nvs_pose_energy_vector_dim(args), score_map_channels=3, hidden_dim=8)
    pair_matcher = PairConditionedLocalMatcher(channels=3, hidden_dim=8, offset_radius=1)
    optimizer = torch.optim.AdamW(
        list(adapter.parameters())
        + list(energy_net.parameters())
        + list(pair_matcher.parameters())
        + list(model.fine_head.parameters()),
        lr=1.0e-3,
    )
    path = tmp_path / "ckpt.pth"

    with torch.no_grad():
        model.fine_head.weight.fill_(0.25)
        model.local_corr_projector.weight.fill_(0.5)
        model.unrelated.weight.fill_(0.75)
        energy_net.energy_head.bias.fill_(1.25)
        pair_matcher.residual[-1].bias.fill_(0.33)

    save_checkpoint(
        path,
        adapter,
        model,
        optimizer,
        7,
        {"pred": 1.0},
        {},
        args,
        energy_net=energy_net,
        pair_matcher=pair_matcher,
    )

    with torch.no_grad():
        model.fine_head.weight.zero_()
        model.local_corr_projector.weight.zero_()
        model.unrelated.weight.zero_()
        energy_net.energy_head.bias.zero_()
        pair_matcher.residual[-1].bias.zero_()

    loaded = load_adapter_checkpoint(
        path,
        adapter,
        model=model,
        optimizer=None,
        energy_net=energy_net,
        pair_matcher=pair_matcher,
    )

    assert loaded["step"] == 7
    assert torch.allclose(model.fine_head.weight, torch.full_like(model.fine_head.weight, 0.25))
    assert torch.allclose(model.local_corr_projector.weight, torch.full_like(model.local_corr_projector.weight, 0.5))
    assert torch.allclose(model.unrelated.weight, torch.zeros_like(model.unrelated.weight))
    assert torch.allclose(energy_net.energy_head.bias, torch.full_like(energy_net.energy_head.bias, 1.25))
    assert torch.allclose(pair_matcher.residual[-1].bias, torch.full_like(pair_matcher.residual[-1].bias, 0.33))


def test_pose_energy_eval_loader_accepts_nvs_checkpoint_key(tmp_path):
    args = type(
        "Args",
        (),
        {
            "train_projector": False,
            "train_model_prefixes": "",
            "pose_energy_use_delta_vector": False,
            "pose_energy_use_center_delta_vector": False,
            "pose_energy_use_rgb": False,
            "pose_energy_use_uncertainty": False,
        },
    )()
    model = torch.nn.Conv2d(3, 3, kernel_size=1)
    adapter = PoseFeatureDomainAdapter(channels=3, hidden_dim=4)
    energy_net = PoseEnergyNet(vector_dim=nvs_pose_energy_vector_dim(args), score_map_channels=3, hidden_dim=8)
    optimizer = torch.optim.AdamW(list(adapter.parameters()) + list(energy_net.parameters()), lr=1.0e-3)
    path = tmp_path / "nvs_ckpt.pth"

    with torch.no_grad():
        energy_net.energy_head.bias.fill_(2.5)

    save_checkpoint(path, adapter, model, optimizer, 11, {"pred": 1.0}, {}, args, energy_net=energy_net)

    with torch.no_grad():
        energy_net.energy_head.bias.zero_()

    loaded = load_pose_energy_checkpoint(
        str(path),
        energy_net,
        model,
        optimizer=None,
        pose_feature_adapter=adapter,
    )

    assert loaded["step"] == 11
    assert torch.allclose(energy_net.energy_head.bias, torch.full_like(energy_net.energy_head.bias, 2.5))


def test_collect_trainable_parameters_can_freeze_pose_feature_adapter():
    adapter = PoseFeatureDomainAdapter(channels=3, hidden_dim=4)
    energy_net = PoseEnergyNet(vector_dim=0, score_map_channels=3, hidden_dim=8, context_layers=0)
    model = torch.nn.Conv2d(3, 3, kernel_size=1)
    for param in model.parameters():
        param.requires_grad_(False)

    params = collect_trainable_parameters(adapter, energy_net, None, model, train_adapter=False)

    assert params
    assert all(not param.requires_grad for param in adapter.parameters())
    assert all(param.requires_grad for param in energy_net.parameters())
    assert all(not param.requires_grad for param in model.parameters())
    adapter_param_ids = {id(param) for param in adapter.parameters()}
    assert all(id(param) not in adapter_param_ids for param in params)


def test_collect_trainable_parameters_can_train_only_render_adapter_domain():
    adapter = PoseFeatureDomainAdapter(
        channels=3,
        hidden_dim=4,
        rgb_context_enabled=True,
        texture_branch_enabled=True,
    )
    model = torch.nn.Conv2d(3, 3, kernel_size=1)
    for param in model.parameters():
        param.requires_grad_(False)

    params = collect_trainable_parameters(
        adapter,
        None,
        None,
        model,
        train_adapter=True,
        adapter_domains="render",
    )

    named = dict(adapter.named_parameters())
    assert params
    assert any(name.startswith("render_") for name, param in named.items() if param.requires_grad)
    assert all(not param.requires_grad for name, param in named.items() if name.startswith("query_"))
    assert all(param.requires_grad for name, param in named.items() if name.startswith("render_"))


def test_collect_trainable_parameters_can_freeze_energy_and_pair_matcher():
    adapter = PoseFeatureDomainAdapter(channels=3, hidden_dim=4)
    energy_net = PoseEnergyNet(vector_dim=0, score_map_channels=3, hidden_dim=8, context_layers=0)
    pair_matcher = PairConditionedLocalMatcher(channels=3, hidden_dim=4, offset_radius=1)
    model = torch.nn.Conv2d(3, 3, kernel_size=1)
    for param in model.parameters():
        param.requires_grad_(False)

    params = collect_trainable_parameters(
        adapter,
        energy_net,
        pair_matcher,
        model,
        train_adapter=True,
        adapter_domains="render",
        train_energy_net=False,
        train_pair_matcher=False,
    )

    assert params
    assert all(not param.requires_grad for param in energy_net.parameters())
    assert all(not param.requires_grad for param in pair_matcher.parameters())
    assert all(param.requires_grad for name, param in adapter.named_parameters() if name.startswith("render_"))


def test_nvs_pose_energy_vector_dim_includes_uncertainty_features():
    base_args = type(
        "Args",
        (),
        {
            "pose_energy_use_delta_vector": False,
            "pose_energy_use_center_delta_vector": False,
            "pose_energy_use_rgb": False,
            "pose_energy_use_uncertainty": False,
        },
    )()
    uncertainty_args = type(
        "Args",
        (),
        {
            "pose_energy_use_delta_vector": False,
            "pose_energy_use_center_delta_vector": False,
            "pose_energy_use_rgb": False,
            "pose_energy_use_uncertainty": True,
        },
    )()

    assert nvs_pose_energy_vector_dim(uncertainty_args) == nvs_pose_energy_vector_dim(base_args) + 3


def test_nvs_pose_energy_vector_dim_matches_feature_builder_optional_flags():
    args = type(
        "Args",
        (),
        {
            "pose_energy_use_delta_vector": True,
            "pose_energy_use_center_delta_vector": False,
            "pose_energy_use_rgb": False,
            "pose_energy_use_uncertainty": True,
        },
    )()

    assert nvs_pose_energy_vector_dim(args) == 37


def test_nvs_pose_energy_vector_dim_includes_center_delta_features():
    base_args = type(
        "Args",
        (),
        {
            "pose_energy_use_delta_vector": True,
            "pose_energy_use_center_delta_vector": False,
            "pose_energy_use_rgb": False,
            "pose_energy_use_uncertainty": True,
        },
    )()
    center_args = type(
        "Args",
        (),
        {
            "pose_energy_use_delta_vector": True,
            "pose_energy_use_center_delta_vector": True,
            "pose_energy_use_rgb": False,
            "pose_energy_use_uncertainty": True,
        },
    )()

    assert nvs_pose_energy_vector_dim(center_args) == nvs_pose_energy_vector_dim(base_args) + 6


def test_pose_energy_residual_update_uses_update_scale():
    pose = torch.eye(4).view(1, 4, 4)
    delta = torch.tensor([[0.20, 0.0, 0.0, 0.0, 0.0, 0.0]])

    half_update = apply_pose_energy_residual_update(pose, delta, update_scale=0.5)
    full_update = apply_pose_energy_residual_update(pose, delta, update_scale=1.0)

    assert torch.allclose(half_update[:, 0, 3], torch.tensor([0.10]), atol=1e-5)
    assert torch.allclose(full_update[:, 0, 3], torch.tensor([0.20]), atol=1e-5)
