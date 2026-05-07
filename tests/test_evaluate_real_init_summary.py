from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch

from feature_retrieval.evaluate_impl import (
    compute_candidate_support_counts,
    compute_loftr_validation_score,
    compute_pnp_quality_prior,
    fuse_candidate_poses,
)
from pose_refine.runtime import apply_pose_delta


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_IMPL = REPO_ROOT / "feature_retrieval" / "evaluate_impl.py"
RETRIEVAL_DATASET = REPO_ROOT / "data" / "radio_loc_retrieval_dataset.py"


def _load_helper(name: str):
    tree = ast.parse(EVALUATE_IMPL.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(module)
            namespace = {
                "apply_pose_delta": apply_pose_delta,
                "Dict": Dict,
                "F": torch.nn.functional,
                "hashlib": hashlib,
                "os": os,
                "Optional": Optional,
                "torch": torch,
            }
            exec(compile(module, str(EVALUATE_IMPL), "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} not found in {EVALUATE_IMPL}")


def test_sha256_file_or_none_hashes_existing_init_cache(tmp_path):
    cache_path = tmp_path / "retrieval_init_poses.npz"
    cache_path.write_bytes(b"fixed init cache")

    sha256_file_or_none = _load_helper("sha256_file_or_none")

    assert sha256_file_or_none(str(cache_path)) == hashlib.sha256(b"fixed init cache").hexdigest()


def test_sha256_file_or_none_returns_none_for_missing_init_cache(tmp_path):
    sha256_file_or_none = _load_helper("sha256_file_or_none")

    assert sha256_file_or_none(str(tmp_path / "missing.npz")) is None


def test_real_init_summary_records_init_cache_sha256():
    source = EVALUATE_IMPL.read_text(encoding="utf-8")

    assert "init_pose_cache_report_path" in source
    assert '"init_cache_sha256": sha256_file_or_none(init_pose_cache_report_path)' in source


def test_feature_retrieval_eval_accepts_localization_manifest():
    source = EVALUATE_IMPL.read_text(encoding="utf-8")

    assert '"--localization_manifest"' in source
    assert "load_mainline_config(args.config, localization_manifest=args.localization_manifest)" in source
    assert "should_restore_pose_checkpoint_map_state(config)" in source
    assert '"localization_manifest": args.localization_manifest' in source
    assert 'manifest_val_init = config.get("dataset", {}).get("val_init_poses_path")' in source
    assert '"query_feature_dir": ds_cfg["feature_dir"]' in source
    assert '"joint_map_checkpoint": config.get("dcff", {}).get("joint_checkpoint")' in source


def test_feature_retrieval_eval_can_override_eval_split():
    source = EVALUATE_IMPL.read_text(encoding="utf-8")

    assert '"--eval_split"' in source
    assert '"--eval_split_file"' in source
    assert '"--eval_start"' in source
    assert 'eval_split_file = args.eval_split_file or ds_cfg[f"{args.eval_split}_split"]' in source
    assert "split=args.eval_split" in source
    assert "split_file=eval_split_file" in source
    assert "eval_indices = list(range(eval_start, eval_stop))" in source


def test_refine_gate_config_keeps_only_enabled_thresholds():
    build_refine_gate_config = _load_helper("build_refine_gate_config")
    args = SimpleNamespace(
        refine_gate_max_step_trans_mm=25.0,
        refine_gate_max_step_rot_deg=0.0,
        refine_gate_max_delta_xi_norm=0.0,
        refine_gate_max_flow_mag_mean=4.5,
        refine_gate_max_flow_mag_max=0.0,
        refine_gate_min_confidence_mean=0.0,
        refine_gate_max_confidence_lowfrac=0.0,
        refine_gate_min_depth_valid_ratio=0.7,
    )

    assert build_refine_gate_config(args) == {
        "max_step_trans_mm": 25.0,
        "max_flow_mag_mean": 4.5,
        "min_depth_valid_ratio": 0.7,
    }


def test_eval_records_refine_diagnostics_and_gate_settings():
    source = EVALUATE_IMPL.read_text(encoding="utf-8")

    assert '"--record_refine_diagnostics"' in source
    assert '"refine_gate": refine_gate' in source
    assert '"refine_diagnostics": summaries["refine_diagnostics"]' in source
    assert 'record[f"top1_diag_{key}"]' in source
    assert 'record[f"selected_diag_{key}"]' in source
    assert 'record[f"oracle_diag_{key}"]' in source
    assert 'record[f"hyp_diag_{key}"]' in source
    assert "compute_gate_keep(stats)" in source


def test_eval_supports_feature_select_and_gain_columns():
    source = EVALUATE_IMPL.read_text(encoding="utf-8")

    assert '"feature_select"' in source
    assert "compute_candidate_feature_residuals" in source
    assert '"top1_trans_gain_mm"' in source
    assert '"fused_trans_worsened"' in source


def test_eval_supports_loftr_render_candidate_validation():
    source = EVALUATE_IMPL.read_text(encoding="utf-8")

    assert '"loftr_render_select"' in source
    assert '"loftr_render_pose"' in source
    assert '"--loftr_select_topn"' in source
    assert '"--loftr_select_score_mode"' in source
    assert "LoFTRInitializer" in source
    assert "_render_rgbd_for_loftr" in source
    assert '"hyp_loftr_render_scores"' in source
    assert '"hyp_loftr_render_inliers"' in source
    assert '"hyp_loftr_render_reproj_rmse"' in source


def test_loftr_validation_pnp_quality_score_uses_reprojection_stats():
    refined_pose = torch.eye(4).numpy()
    validated_pose = torch.eye(4).numpy()
    result = SimpleNamespace(
        num_inliers=100,
        num_confident_matches=200,
        num_raw_matches=240,
        num_depth_valid=180,
        mean_confidence=0.8,
        extra={
            "pnp_quality": {
                "pnp_reproj_rmse": 1.0,
                "pnp_reproj_median": 0.7,
                "pnp_inlier_ratio": 0.5,
                "pnp_inlier_conf_mean": 0.75,
            }
        },
    )

    basic = compute_loftr_validation_score(
        result,
        validated_pose,
        refined_pose,
        score_mode="basic",
    )
    quality = compute_loftr_validation_score(
        result,
        validated_pose,
        refined_pose,
        score_mode="pnp_quality",
    )

    assert basic["score"] == 80.0
    assert quality["score"] == 18.75
    assert quality["reproj_rmse"] == 1.0
    assert quality["reproj_median"] == 0.7
    assert quality["inlier_ratio"] == 0.5
    assert quality["raw_matches"] == 240.0
    assert quality["depth_valid"] == 180.0


def test_score_select_fusion_uses_highest_candidate_score():
    poses = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
    poses[0, 0, 3] = 1.0
    poses[1, 0, 3] = 2.0
    poses[2, 0, 3] = 3.0

    fused, info = fuse_candidate_poses(
        poses,
        method="score_select",
        retrieval_scores=torch.tensor([0.1, 4.0, 2.0]),
    )

    assert torch.allclose(fused, poses[1])
    assert info["selected_idx"] == 1.0
    assert info["used_top1_fallback"] == 0.0


def test_pnp_quality_prior_combines_quality_fields():
    prior = compute_pnp_quality_prior(
        fallback_scores=torch.tensor([0.0, 0.0, 0.0]),
        candidate_quality={
            "retrieval_pnp_success_candidates": torch.tensor([1.0, 1.0, 1.0]),
            "retrieval_pnp_num_inliers_candidates": torch.tensor([100.0, 250.0, 150.0]),
            "retrieval_pnp_inlier_ratio_candidates": torch.tensor([0.4, 0.8, 0.5]),
            "retrieval_pnp_inlier_conf_mean_candidates": torch.tensor([0.5, 0.9, 0.6]),
            "retrieval_pnp_reproj_rmse_candidates": torch.tensor([5.0, 1.0, 3.0]),
        },
    )

    assert int(torch.argmax(prior).item()) == 1


def test_quality_consensus_selects_quality_candidate_when_support_is_small():
    poses = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
    poses[0, 0, 3] = 0.0
    poses[1, 0, 3] = 5.0
    poses[2, 0, 3] = 10.0

    fused, info = fuse_candidate_poses(
        poses,
        method="quality_consensus",
        retrieval_scores=torch.tensor([0.0, 0.0, 0.0]),
        quality_scores=torch.tensor([0.2, 3.0, 1.0]),
        consensus_radius_m=0.5,
        quality_consensus_small_size=2,
        quality_consensus_large_size=3,
    )

    assert torch.allclose(fused, poses[1])
    assert info["selected_idx"] == 1.0


def test_quality_weighted_consensus_centroid_uses_linear_quality_weights():
    poses = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
    poses[0, 0, 3] = 0.0
    poses[1, 0, 3] = -0.2
    poses[2, 0, 3] = -1.0

    fused, info = fuse_candidate_poses(
        poses,
        method="quality_weighted_consensus_centroid",
        retrieval_scores=torch.tensor([1.0, 1.0, 1.0]),
        quality_scores=torch.tensor([0.0, 0.0, 10.0]),
        consensus_radius_m=1.1,
        consensus_min_size=2,
    )

    expected_center_x = (0.1 * 0.0 + 0.1 * 0.2 + 1.1 * 1.0) / 1.3
    assert torch.allclose(fused[0, 3], torch.tensor(-expected_center_x), atol=1e-6)
    assert info["cluster_size"] == 3.0
    assert info["selected_idx"] == 0.0
    assert info["used_top1_fallback"] == 0.0


def test_candidate_support_counts_use_camera_centers():
    poses = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
    poses[0, 0, 3] = 0.0
    poses[1, 0, 3] = -0.25
    poses[2, 0, 3] = -2.0

    support = compute_candidate_support_counts(poses, radius_m=0.5)

    assert support.tolist() == [2.0, 2.0, 1.0]


def test_eval_reports_loaded_candidate_slots_and_effective_topk():
    source = EVALUATE_IMPL.read_text(encoding="utf-8")

    assert "summarize_init_candidate_slots(val_ds)" in source
    assert '"effective_retrieval_topk": effective_retrieval_topk' in source
    assert '"loaded_candidate_slots"' in source
    assert "using only the first" in source


def test_eval_records_per_hypothesis_errors_and_candidate_quality():
    source = EVALUATE_IMPL.read_text(encoding="utf-8")

    assert '"hyp_final_trans_err_mm"' in source
    assert '"hyp_candidate_scores"' in source
    assert '"hyp_candidate_quality_scores"' in source
    assert '"hyp_consensus_support"' in source
    assert '"hyp_camera_centers"' in source
    assert '"hyp_rgb_mse"' in source
    assert '"hyp_feature_residuals"' in source
    assert '"--hyp_chunk_size"' in source
    assert "candidate_poses[h_start:h_end]" in source
    assert "OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS" in source
    assert 'record[f"hyp_{key}"]' in source


def test_retrieval_dataset_preserves_source_dir_for_rgb_select():
    source = RETRIEVAL_DATASET.read_text(encoding="utf-8")

    assert "super().__init__(*args, source_dir=source_dir, **kwargs)" in source


def test_candidate_feature_residuals_expand_single_query_batch():
    compute_candidate_feature_residuals = _load_helper("compute_candidate_feature_residuals")
    query = torch.zeros(1, 2, 2, 2)
    rendered = torch.stack([torch.zeros(2, 2, 2), torch.ones(2, 2, 2)], dim=0)

    scores = compute_candidate_feature_residuals(query, rendered, normalize=False)

    assert torch.allclose(scores, torch.tensor([0.0, 1.0]))


def test_eval_apply_pose_delta_honors_after_first_component_scales():
    class DummyModel:
        pose_update_scale = 1.0
        pose_update_trans_scale = 1.0
        pose_update_rot_scale = 1.0
        pose_update_trans_scale_after_first = 0.25
        pose_update_rot_scale_after_first = 0.0

    eval_apply_pose_delta = _load_helper("eval_apply_pose_delta")
    pose = torch.eye(4).unsqueeze(0)
    delta = torch.tensor([[0.2, 0.0, 0.0, 0.0, 0.0, 0.2]])

    updated = eval_apply_pose_delta(DummyModel(), pose, delta, outer_iter=1)

    assert torch.allclose(updated[0, :3, 3], torch.tensor([0.05, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(updated[0, :3, :3], torch.eye(3), atol=1e-6)
