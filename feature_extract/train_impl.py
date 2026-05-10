#!/usr/bin/env python3
"""
Minimal joint RADIO-DCFF feature-learning scaffold.

Current scope:
  1. Query-side student learns from raw RGB -> dual 64d feature heads
  2. Teacher supervision comes from cached RADIO dual features
  3. Map-side joint supervision path is exposed as a config-gated hook

Usage:
    python -m feature_extract.train \
        --config feature_extract/configs/joint_radio_dcff_oh_v5l_pointwise_featsharp_full.yaml

Smoke test:
    python -m feature_extract.train \
        --config feature_extract/configs/joint_radio_dcff_oh_v5m_pointwise_teacher_anchor_pilot.yaml \
        --smoke-test
"""

import argparse
import copy
import inspect
import json
import logging
import math
import os
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.radio_loc_dataset import (
    camera_params_to_intrinsics,
    colmap_to_w2c,
    read_colmap_cameras,
    read_colmap_images,
)
from data.radio_loc_retrieval_dataset import (
    OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS,
    load_retrieval_init_entries,
)
from feature_field.dcff.losses import (
    channel_standardized_loss,
    cosine_loss,
    feature_gradient_loss,
    infonce_contrastive_loss,
    l1_feature_loss,
)
from feature_extract import load_config as load_feature_extract_config
from feature_extract.students.radio_query_student import (
    CandidateScoreFusionHead,
    CandidateScoreMapFusionHead,
    RadioQueryStudent,
    matrix_to_rotation_6d,
)
from feature_field import build_dcff_runtime, intrinsics_to_K
from feature_field.runtime import _apply_dcff_postprocess
from feature_field.utils.loc_reporting import save_experiment_bundle
from feature_field.utils.feature_track_vis import save_feature_track_visual
from pose_refine import apply_pose_delta, compute_image_jacobian, diff_pose_solve, feature_metric_solve
from pose_refine.utils.lie_algebra import se3_log
from pose_refine.models.concat_pose_net import correlation_confidence_from_probs


CANDIDATE_RENDER_BASE_FEATURE_NAMES = (
    "render_score",
    "render_score_centered",
    "render_valid_fraction",
)
CANDIDATE_RENDER_RICH_FEATURE_NAMES = CANDIDATE_RENDER_BASE_FEATURE_NAMES + (
    "render_score_std",
    "render_score_max",
    "render_score_max_centered",
    "render_score_topk_mean",
    "render_score_topk_centered",
    "render_score_peakiness",
)
CANDIDATE_QUALITY_FEATURE_NAMES = (
    "retrieval_score_log_z",
    "retrieval_original_score",
    "retrieval_pnp_success",
    "retrieval_pnp_num_inliers_log_z",
    "retrieval_pnp_num_matches_log_z",
    "retrieval_pnp_reproj_rmse_quality",
    "retrieval_pnp_reproj_median_quality",
    "retrieval_pnp_inlier_ratio",
    "retrieval_pnp_inlier_conf_mean",
)
DEFAULT_CANDIDATE_SCORE_FUSION_INPUT_DIM = (
    len(CANDIDATE_RENDER_BASE_FEATURE_NAMES) + len(CANDIDATE_QUALITY_FEATURE_NAMES)
)
RICH_CANDIDATE_SCORE_FUSION_INPUT_DIM = (
    len(CANDIDATE_RENDER_RICH_FEATURE_NAMES) + len(CANDIDATE_QUALITY_FEATURE_NAMES)
)


DEFAULT_CONFIG = {
    "exp_name": "joint_radio_dcff_oh_v1",
    "output_dir": "/root/ICLPose/result/feature_extract",
    "dataset": {
        "source_dir": "dataset/OldHospital",
        "feature_dir": "/root/ICLPose/result/feature_extract/features_radio_dual/OldHospital",
        "train_split": "dataset/OldHospital/dataset_train.txt",
        "val_split": "dataset/OldHospital/dataset_test.txt",
        "image_patterns": [
            "seq*/*.png",
            "seq*/*.jpg",
            "images/*.png",
            "images/*.jpg",
            "*.png",
            "*.jpg",
        ],
        "patch_size": 16,
        "input_hw": [1088, 1920],
        "feature_hw": [68, 120],
        "coarse_feature_hw": None,
        "student_feature_hw": None,
        "student_coarse_feature_hw": None,
        "cache_teacher": False,
        "fallback_val_ratio": 0.1,
        "max_train_samples": None,
        "max_val_samples": None,
        "synthetic_if_missing": False,
        "prior_mask_path": None,
        "prior_mask_channels": [0, 1, 2],
        "teacher_correspondence_path": None,
        "teacher_correspondence_train_path": None,
        "teacher_correspondence_val_path": None,
        "teacher_correspondence_max_points": 512,
        "teacher_correspondence_coordinate_space": "auto",
    },
    "model": {
        "feature_dim": 64,
        "fine_feature_dim": None,
        "coarse_feature_dim": None,
        "base_channels": 32,
        "stage_dims": [32, 64, 96, 128],
        "dropout": 0.0,
        "l2_normalize": True,
        "predict_magnitude": False,
        "fine_init_norm": 1.0,
        "coarse_init_norm": 1.0,
        "magnitude_min": 1e-4,
        "warmstart_strict": True,
        "fine_low_level_skip": False,
        "fine_low_level_init": 0.0,
        "fine_highres_skip": False,
        "fine_highres_source": "stage2",
        "fine_highres_init": 0.0,
        "fine_highres_zero_init": False,
        "global_context_enabled": False,
        "global_context_zero_init": True,
        "window_attention_layers": 0,
        "window_attention_heads": 8,
        "window_attention_size": 16,
        "window_attention_mlp_ratio": 2.0,
        "window_attention_dropout": 0.0,
        "window_attention_shift": False,
        "window_attention_zero_init": True,
        "teacher_fine_condition": False,
        "teacher_fine_init": 1.0,
        "teacher_fine_zero_init": True,
        "teacher_fine_detach": True,
        "scene_coord_head": False,
        "scene_coord_zero_init": True,
        "scene_coord_detach_base": False,
        "scene_coord_use_pixel_grid": False,
        "scene_coord_global_context": False,
        "local_matcher_enabled": False,
        "local_matcher_radius": 4,
        "local_matcher_hidden_dim": 64,
        "local_matcher_zero_init": True,
        "local_matcher_residual_scale": 1.0,
        "local_matcher_context_mode": "basic",
        "local_flow_head_enabled": False,
        "local_flow_head_radius": 4,
        "local_flow_head_hidden_dim": 64,
        "local_flow_head_zero_init": True,
        "local_flow_head_max_flow": None,
        "local_flow_head_base_flow_mode": "none",
        "local_flow_head_base_temperature": 0.05,
        "local_flow_head_context_mode": "basic",
        "local_corr_projector_enabled": False,
        "local_corr_projector_hidden_dim": 96,
        "local_corr_projector_output_dim": None,
        "local_corr_projector_zero_init": True,
        "local_corr_projector_l2_normalize": True,
        "local_corr_projector_domain_adapter": False,
        "local_corr_query_projector_zero_init": None,
        "local_corr_render_projector_zero_init": None,
        "query_channel_gate_enabled": False,
        "query_channel_gate_hidden_dim": None,
        "query_channel_gate_zero_init": True,
        "apply_query_channel_gate": False,
    },
    "training": {
        "device": "cuda",
        "seed": 42,
        "epochs": 8,
        "batch_size": 2,
        "num_workers": 2,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "grad_clip": 1.0,
        "amp": True,
        "log_every": 10,
        "save_every_epochs": 1,
        "val_every_epochs": 1,
        "max_steps": None,
        "best_metric": "loss_total",
        "best_metric_mode": "min",
        "model_lr_scales": {},
        "freeze_model_except_prefixes": None,
    },
    "loss": {
        "fine_l1_weight": 1.0,
        "fine_cos_weight": 1.0,
        "fine_channel_std_weight": 0.0,
        "coarse_l1_weight": 1.0,
        "coarse_cos_weight": 1.0,
        "coarse_channel_std_weight": 0.0,
        "fine_coarse_ortho_weight": 0.0,
        "teacher_norm_weight": 0.0,
        "query_teacher_infonce_weight": 0.0,
        "infonce_temperature": 0.07,
        "infonce_samples": 256,
        "infonce_cross_batch": False,
    },
    "retrieval": {
        "enabled": False,
        "feature_dir": None,
        "teacher_subdir": "cls",
        "student_dim": 768,
        "hidden_dim": 256,
        "dropout": 0.0,
        "l2_normalize": True,
        "cache_teacher": False,
        "l1_weight": 0.0,
        "cos_weight": 0.0,
        "infonce_weight": 0.0,
        "similarity_weight": 0.0,
        "temperature": 0.07,
    },
    "map_supervision": {
        "enabled": False,
        "config_path": None,
        "colmap_dir": None,
        "cache_rendered": True,
        "trainable": False,
        "map_lr_scale": 0.1,
        "hash_mlp_lr_scale": 0.05,
        "train_fine_decoder": False,
        "train_coarse_fusion": False,
        "train_feat_sharp": False,
        "train_hash_mlp": False,
        "train_latent": False,
        "train_geometry": False,
        "reset_latent": False,
        "latent_init_std": 0.01,
        "detach_query_features": False,
        "coarse_smoothing_kernel": 1,
        "latent_lr_scale": 0.05,
        "geometry_lr_scale": 0.01,
        "position_lr_scale": None,
        "opacity_lr_scale": None,
        "scaling_lr_scale": None,
        "rotation_lr_scale": None,
        "color_lr_scale": None,
        "coarse_start_epoch": 999999,
        "query_fine_weight": 0.0,
        "query_fine_raw_weight": 0.0,
        "query_coarse_weight": 0.0,
        "coarse_pose_rank_weight": 0.0,
        "coarse_pose_rank_temperature": 0.07,
        "coarse_pose_energy_weight": 0.0,
        "coarse_pose_energy_temperature": 0.07,
        "coarse_pose_local_energy_weight": 0.0,
        "coarse_pose_local_energy_temperature": 0.07,
        "coarse_pose_local_energy_radius": 4,
        "coarse_pose_local_energy_preprocess": "none",
        "coarse_pose_local_energy_highpass_kernel": 5,
        "candidate_render_score_weight": 0.0,
        "candidate_render_score_feature": "coarse",
        "candidate_render_score_mode": "local",
        "candidate_render_score_temperature": 0.07,
        "candidate_render_score_radius": 4,
        "candidate_render_score_preprocess": "none",
        "candidate_render_score_highpass_kernel": 5,
        "candidate_render_score_rot_cost_weight": 0.1,
        "candidate_render_score_max_candidates": 0,
        "candidate_render_include_aux": True,
        "candidate_render_batch_size": 0,
        "candidate_render_score_train_map": False,
        "candidate_render_pose_source": "pose_init",
        "coarse_pose_lattice_base_source": "rendered_map_pose_neg",
        "coarse_pose_lattice_trans_cm": [0.0, 5.0, 10.0, 25.0],
        "coarse_pose_lattice_rot_deg": [0.0, 1.0, 2.0, 5.0],
        "coarse_pose_lattice_include_identity": True,
        "coarse_pose_lattice_max_candidates": 0,
        "coarse_pose_lattice_limit_strategy": "head",
        "coarse_pose_lattice_combine_trans_rot": False,
        "coarse_pose_lattice_oracle_subset_size": 0,
        "coarse_pose_lattice_oracle_subset_extras_strategy": "uniform",
        "coarse_pose_lattice_oracle_subset_rot_cost_weight": 0.1,
        "candidate_score_fusion_weight": 0.0,
        "candidate_score_fusion_feature": "coarse",
        "candidate_score_fusion_mode": "local",
        "candidate_score_fusion_temperature": 0.07,
        "candidate_score_fusion_radius": 4,
        "candidate_score_fusion_preprocess": "none",
        "candidate_score_fusion_highpass_kernel": 5,
        "candidate_score_fusion_rot_cost_weight": 0.1,
        "candidate_score_fusion_target_mode": "hard",
        "candidate_score_fusion_target_temperature_m": 0.25,
        "candidate_score_fusion_render_feature_mode": "basic",
        "candidate_score_fusion_score_map_mode": "peak_offset",
        "candidate_score_fusion_wls_radius": None,
        "candidate_score_fusion_wls_temperature": None,
        "candidate_score_fusion_wls_damping": 1e-3,
        "candidate_score_fusion_wls_update_scale": 1.0,
        "candidate_score_fusion_wls_conf_mode": "max",
        "candidate_score_fusion_wls_conf_variance_scale": 0.5,
        "candidate_score_fusion_wls_downsample": 1,
        "candidate_score_fusion_cost_regression_weight": 0.0,
        "candidate_score_fusion_cost_regression_temperature_m": None,
        "candidate_score_fusion_pairwise_rank_weight": 0.0,
        "candidate_score_fusion_pairwise_rank_temperature": 1.0,
        "candidate_score_fusion_pairwise_rank_min_gap_m": 0.0,
        "candidate_two_stage_enabled": False,
        "candidate_stage1_topk": 0,
        "candidate_stage2_topm": 1,
        "candidate_stage2_selection": "pred",
        "candidate_stage2_train_map": False,
        "candidate_stage2_detach_selection": True,
        "candidate_stage2_prefix": "rendered_map_candidate_refine",
        "candidate_refined_pose_weight": 0.0,
        "candidate_refined_pose_feature": None,
        "candidate_refined_pose_radius": None,
        "candidate_refined_pose_temperature": None,
        "candidate_refined_pose_damping": None,
        "candidate_refined_pose_update_scale": None,
        "candidate_refined_pose_rot_cost_weight": None,
        "candidate_refined_pose_wls_conf_mode": None,
        "candidate_refined_pose_wls_conf_variance_scale": None,
        "candidate_refined_pose_wls_downsample": None,
        "query_fine_infonce_weight": 0.0,
        "query_coarse_infonce_weight": 0.0,
        "fine_coarse_ortho_weight": 0.0,
        "perturb_rank_weight": 0.0,
        "perturb_max_shift_px": 2,
        "perturb_margin": 0.1,
        "perturb_margin_per_m": 0.0,
        "perturb_render_negatives": False,
        "perturb_rot_deg": 0.0,
        "perturb_rot_deg_choices": None,
        "perturb_trans_m": 0.0,
        "perturb_trans_cm_choices": None,
        "perturb_pose_mode": "camera_center",
        "perturb_frame": "camera",
        "perturb_axes": [0, 1, 2],
        "global_render_negative_count": 0,
        "global_render_negative_min_trans_m": 0.0,
        "rendered_teacher_fine_weight": 0.0,
        "rendered_teacher_fine_raw_weight": 0.0,
        "rendered_teacher_coarse_weight": 0.0,
        "rendered_teacher_fine_infonce_weight": 0.0,
        "rendered_teacher_coarse_infonce_weight": 0.0,
        "infonce_cross_batch": True,
        "variance_target_std": 0.05,
        "query_variance_weight": 0.0,
        "map_variance_weight": 0.0,
        "query_covariance_weight": 0.0,
        "map_covariance_weight": 0.0,
        "depth_observability_weight": 0.0,
        "depth_observability_power": 1.0,
        "depth_observability_max": 4.0,
        "translation_observability_weight": 0.0,
        "translation_observability_mode": "xyz",
        "translation_observability_power": 1.0,
        "translation_observability_max": 4.0,
        "feature_metric_pose_weight": 0.0,
        "feature_metric_pose_damping": 1e-3,
        "feature_metric_pose_update_scale": 1.0,
        "feature_metric_pose_rot_weight": 1.0,
        "feature_metric_pose_trans_weight": 50.0,
        "feature_metric_pose_normalize": True,
        "feature_metric_pose_rot_damping_multiplier": 1.0,
        "feature_metric_gradient_direction_weight": 0.0,
        "query_corr_ce_weight": 0.0,
        "query_corr_ce_temperature": None,
        "query_corr_subpixel_weight": 0.0,
        "query_corr_flow_weight": 0.0,
        "query_corr_flow_cosine_weight": 0.0,
        "query_corr_flow_head_conf_weight": 0.0,
        "query_corr_peak_margin_weight": 0.0,
        "query_corr_distill_weight": 0.0,
        "query_corr_distill_temperature": None,
        "query_corr_distill_target_temperature": None,
        "query_corr_peak_margin": 0.05,
        "query_corr_low_peak_gap_threshold": 0.0,
        "query_identity_corr_ce_weight": 0.0,
        "query_identity_corr_subpixel_weight": 0.0,
        "query_identity_corr_flow_weight": 0.0,
        "query_identity_corr_peak_margin_weight": 0.0,
        "query_identity_corr_distill_weight": 0.0,
        "query_identity_corr_distill_temperature": None,
        "query_identity_corr_distill_target_temperature": None,
        "query_identity_corr_peak_margin": None,
        "query_identity_corr_radius": None,
        "query_identity_corr_temperature": None,
        "query_identity_corr_ce_temperature": None,
        "query_identity_corr_feature_preprocess": None,
        "query_identity_corr_highpass_kernel": None,
        "query_identity_corr_highpass_scale": None,
        "query_corr_wls_pose_weight": 0.0,
        "query_corr_wls_pose_damping": 1e-3,
        "query_corr_wls_pose_update_scale": 1.0,
        "query_corr_wls_pose_rot_weight": 1.0,
        "query_corr_wls_pose_trans_weight": 50.0,
        "query_corr_wls_conf_threshold": 0.0,
        "query_corr_wls_conf_power": 1.0,
        "query_corr_wls_conf_mode": "max",
        "query_corr_wls_conf_variance_scale": 0.5,
        "query_corr_wls_min_conf_cov": 0.0,
        "query_corr_wls_accept_min_conf_mean": 0.0,
        "query_corr_wls_accept_min_conf_cov": 0.0,
        "query_corr_wls_accept_min_delta_mm": 0.0,
        "query_corr_wls_accept_max_delta_mm": 0.0,
        "query_corr_pose_gain_weight": 0.0,
        "query_corr_pose_gain_rot_margin_deg": 0.0,
        "query_corr_pose_gain_trans_margin_mm": 0.0,
        "query_corr_pose_gain_rot_weight": 0.0,
        "query_corr_pose_gain_trans_weight": 1.0,
        "query_scene_coord_weight": 0.0,
        "query_scene_coord_warp_weight": 0.0,
        "query_scene_coord_huber_beta": 0.02,
        "scene_coord_center": [0.0, 0.0, 0.0],
        "scene_coord_scale": 20.0,
        "query_corr_scene_coord_weight": 0.0,
        "feature_metric_scene_coord_weight": 0.0,
        "query_corr_radius": 4,
        "query_corr_temperature": 0.05,
        "query_corr_huber_delta": 1.0,
        "query_corr_min_flow_px": 0.0,
        "query_corr_max_flow_px": 0.0,
        "query_corr_flow_decode_mode": "softargmax",
        "query_corr_feature_preprocess": "none",
        "query_corr_highpass_kernel": 3,
        "query_corr_highpass_scale": 1.0,
        "teacher_corr_weight": 0.0,
        "teacher_corr_temperature": 0.07,
        "teacher_corr_min_confidence": 0.0,
        "teacher_corr_min_points": 4,
        "teacher_corr_use_projector": True,
        "teacher_corr_negative_exclusion_px": 0.0,
        "teacher_corr_positive_weight": 0.0,
        "teacher_corr_margin_weight": 0.0,
        "teacher_corr_margin": 0.1,
        "teacher_corr_local_patch_weight": 0.0,
        "teacher_corr_local_patch_radius": 2,
        "teacher_corr_local_patch_temperature": None,
        "teacher_corr_local_patch_min_points": None,
        "teacher_corr_local_patch_positive_weight": 0.0,
        "teacher_corr_local_patch_margin_weight": 0.0,
        "teacher_corr_local_patch_margin": 0.05,
        "query_projected_fine_weight": 0.0,
        "local_corr_projector_apply_to_query": True,
        "query_flow_warp_weight": 0.0,
        "query_flow_warp_contrastive_weight": 0.0,
        "query_flow_warp_contrastive_margin": 0.1,
        "query_flow_warp_contrastive_offsets": None,
        "query_fine_key": "fine",
        "query_local_fine_key": None,
        "query_local_fine_weight": 0.0,
        "query_local_fine_infonce_weight": 0.0,
        "query_local_fine_grad_weight": 0.0,
        "local_matcher_enabled": False,
        "local_flow_head_enabled": False,
        "local_corr_projector_enabled": False,
        "map_self_corr_ce_weight": 0.0,
        "map_self_corr_subpixel_weight": 0.0,
        "map_self_corr_flow_weight": 0.0,
        "map_self_corr_peak_margin_weight": 0.0,
        "map_self_flow_warp_weight": 0.0,
        "map_self_flow_warp_contrastive_weight": 0.0,
        "map_self_feature_metric_pose_weight": 0.0,
    },
    "visualization": {
        "num_val_vis": 4,
        "save_root": "/root/ICLPose/result/feature_extract/visualizations/feature_track",
    },
}


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    config_path = Path(path)
    with open(config_path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    base_config = user_cfg.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        return deep_merge(load_config(base_path), user_cfg)
    return deep_merge(DEFAULT_CONFIG, user_cfg)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_warmstart(model, checkpoint, strict=True, skip_prefixes=None):
    """Load model weights for warmstart, optionally skipping incompatible tensors.

    PyTorch's strict=False still raises on shape mismatches. Warmstarting across
    high-res student variants needs the compatible layers while leaving resized
    heads/skip branches randomly initialized.
    """
    source_state = checkpoint.get("model_state_dict", checkpoint)
    skip_prefixes = tuple(str(prefix) for prefix in (skip_prefixes or []) if str(prefix))
    if strict:
        if skip_prefixes:
            raise ValueError("skip_prefixes requires strict=False warmstart")
        load_result = model.load_state_dict(source_state, strict=True)
        return {
            "load_result": load_result,
            "missing_keys": list(getattr(load_result, "missing_keys", [])),
            "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
            "skipped_mismatched": {},
            "skipped_unexpected": [],
            "skipped_by_prefix": [],
        }

    target_state = model.state_dict()
    filtered_state = {}
    skipped_mismatched = {}
    skipped_unexpected = []
    skipped_by_prefix = []
    for key, value in source_state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped_by_prefix.append(key)
            continue
        if key not in target_state:
            skipped_unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped_mismatched[key] = (tuple(value.shape), tuple(target_state[key].shape))
            continue
        filtered_state[key] = value

    load_result = model.load_state_dict(filtered_state, strict=False)
    return {
        "load_result": load_result,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "skipped_mismatched": skipped_mismatched,
        "skipped_unexpected": skipped_unexpected,
        "skipped_by_prefix": skipped_by_prefix,
    }


def perturb_w2c_camera_center(pose: torch.Tensor, offset: torch.Tensor, frame: str = "camera") -> torch.Tensor:
    single = pose.ndim == 2
    poses = pose.unsqueeze(0) if single else pose
    poses = poses.float()
    offsets = offset.to(device=poses.device, dtype=poses.dtype)
    if offsets.ndim == 1:
        offsets = offsets.view(1, 3).expand(poses.shape[0], -1)
    if offsets.shape[0] != poses.shape[0]:
        raise ValueError(f"offset batch {offsets.shape[0]} does not match pose batch {poses.shape[0]}")

    R = poses[:, :3, :3]
    t = poses[:, :3, 3]
    centers = -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)
    frame_key = str(frame).lower()
    if frame_key == "camera":
        offsets_world = (R.transpose(1, 2) @ offsets.unsqueeze(-1)).squeeze(-1)
    elif frame_key == "world":
        offsets_world = offsets
    else:
        raise ValueError(f"Unknown perturb frame '{frame}'. Use 'camera' or 'world'.")
    new_centers = centers + offsets_world
    result = poses.clone()
    result[:, :3, 3] = -(R @ new_centers.unsqueeze(-1)).squeeze(-1)
    return result[0] if single else result


def axis_angle_rotation_matrix(axis: int, angle_rad: torch.Tensor) -> torch.Tensor:
    """Build a 3x3 rotation matrix for one canonical axis."""
    axis_idx = int(axis)
    if axis_idx not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    angle = torch.as_tensor(angle_rad, dtype=torch.float32)
    c = torch.cos(angle)
    s = torch.sin(angle)
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    if axis_idx == 0:
        rows = ((one, zero, zero), (zero, c, -s), (zero, s, c))
    elif axis_idx == 1:
        rows = ((c, zero, s), (zero, one, zero), (-s, zero, c))
    else:
        rows = ((c, -s, zero), (s, c, zero), (zero, zero, one))
    return torch.stack([torch.stack(row) for row in rows])


def camera_centers_from_w2c(poses_w2c: torch.Tensor) -> torch.Tensor:
    R = poses_w2c[:, :3, :3]
    t = poses_w2c[:, :3, 3]
    return -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)


def pose_error_tensors(pose_pred: torch.Tensor, pose_gt: torch.Tensor):
    R_pred = pose_pred[:, :3, :3].float()
    R_gt = pose_gt[:, :3, :3].float()
    R_rel = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_cos_loss = 1.0 - cos_angle
    rot_err_deg = torch.acos(cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * 180.0 / math.pi
    trans_err_m = torch.linalg.norm(
        camera_centers_from_w2c(pose_pred.float()) - camera_centers_from_w2c(pose_gt.float()),
        dim=1,
    )
    return rot_cos_loss, rot_err_deg, trans_err_m


def _as_float_sequence(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.replace(";", ",").split(",")
    if isinstance(value, (int, float)):
        value = [value]
    return [float(v) for v in value]


def pose_lattice_delta_templates(
    trans_cm,
    rot_deg,
    *,
    include_identity=True,
    combine_trans_rot=False,
    dtype=torch.float32,
):
    """Build axis-aligned local SE(3) perturbations for CPR coarse pose search."""
    zero = torch.zeros(6, dtype=dtype)
    trans_rows = []
    rot_rows = []
    if include_identity:
        rows = [zero]
    else:
        rows = []
    trans_values_m = [abs(v) * 0.01 for v in _as_float_sequence(trans_cm) if abs(float(v)) > 1e-12]
    rot_values_rad = [math.radians(abs(v)) for v in _as_float_sequence(rot_deg) if abs(float(v)) > 1e-12]
    axes = torch.eye(3, dtype=dtype)
    for magnitude in trans_values_m:
        for axis in axes:
            for sign in (-1.0, 1.0):
                delta = torch.zeros(6, dtype=dtype)
                delta[:3] = axis * (sign * float(magnitude))
                trans_rows.append(delta)
    for magnitude in rot_values_rad:
        for axis in axes:
            for sign in (-1.0, 1.0):
                delta = torch.zeros(6, dtype=dtype)
                delta[3:] = axis * (sign * float(magnitude))
                rot_rows.append(delta)
    if combine_trans_rot and (trans_rows or rot_rows):
        trans_basis = [zero] + trans_rows
        rot_basis = [zero] + rot_rows
        rows = []
        seen = set()
        for trans_delta in trans_basis:
            for rot_delta in rot_basis:
                delta = trans_delta + rot_delta
                key = tuple(round(float(v), 9) for v in delta.tolist())
                if key in seen:
                    continue
                seen.add(key)
                rows.append(delta)
    else:
        rows.extend(trans_rows)
        rows.extend(rot_rows)
    if not rows:
        rows.append(zero)
    return torch.stack(rows).to(dtype=dtype)


def build_local_pose_lattice_candidates(
    base_pose_w2c: torch.Tensor,
    *,
    trans_cm,
    rot_deg,
    include_identity=True,
    max_candidates=0,
    limit_strategy="head",
    combine_trans_rot=False,
):
    """Generate a local pose candidate bank around an initial w2c pose."""
    single = base_pose_w2c.ndim == 2
    base = base_pose_w2c.unsqueeze(0) if single else base_pose_w2c
    if base.ndim != 3 or base.shape[-2:] != (4, 4):
        raise ValueError(f"base_pose_w2c must have shape (B,4,4), got {tuple(base_pose_w2c.shape)}")
    deltas = pose_lattice_delta_templates(
        trans_cm,
        rot_deg,
        include_identity=bool(include_identity),
        combine_trans_rot=bool(combine_trans_rot),
        dtype=torch.float32,
    ).to(device=base.device, dtype=base.dtype)
    limit = int(max_candidates or 0)
    if limit > 0 and limit < deltas.shape[0]:
        strategy_key = str(limit_strategy or "head").lower()
        if strategy_key in ("head", "first", "prefix"):
            select_idx = torch.arange(limit, device=deltas.device, dtype=torch.long)
        else:
            select_idx = select_pose_candidate_indices(
                deltas.shape[0],
                limit=limit,
                strategy=strategy_key,
            ).to(device=deltas.device)
        deltas = deltas[select_idx]
    bsz, num_candidates = base.shape[0], deltas.shape[0]
    base_flat = base[:, None].expand(-1, num_candidates, -1, -1).reshape(bsz * num_candidates, 4, 4)
    delta_flat = deltas[None].expand(bsz, -1, -1).reshape(bsz * num_candidates, 6)
    candidates = apply_pose_delta(base_flat.float(), delta_flat.float()).to(dtype=base.dtype)
    candidates = candidates.reshape(bsz, num_candidates, 4, 4)
    return candidates[0] if single else candidates


def _rotation_error_from_mats(rot_pred: torch.Tensor, rot_gt: torch.Tensor):
    rot_gt = rot_gt.to(device=rot_pred.device, dtype=rot_pred.dtype)
    rot_rel = rot_pred.transpose(-1, -2) @ rot_gt
    trace = rot_rel[..., 0, 0] + rot_rel[..., 1, 1] + rot_rel[..., 2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_cos_loss = 1.0 - cos_angle
    rot_err_deg = torch.acos(cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * 180.0 / math.pi
    return rot_cos_loss, rot_err_deg


def farthest_point_anchor_centers(centers: torch.Tensor, num_anchors: int) -> torch.Tensor:
    """Deterministic farthest-point sampling over camera centers."""
    centers = torch.as_tensor(centers, dtype=torch.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")
    if centers.shape[0] <= 0:
        raise ValueError("centers must contain at least one point")
    num_anchors = max(1, min(int(num_anchors), int(centers.shape[0])))
    centroid = centers.mean(dim=0, keepdim=True)
    first_idx = torch.linalg.norm(centers - centroid, dim=1).argmax()
    selected = [int(first_idx.item())]
    min_dist = torch.linalg.norm(centers - centers[first_idx].view(1, 3), dim=1)
    for _ in range(1, num_anchors):
        next_idx = int(min_dist.argmax().item())
        selected.append(next_idx)
        next_dist = torch.linalg.norm(centers - centers[next_idx].view(1, 3), dim=1)
        min_dist = torch.minimum(min_dist, next_dist)
    return centers[selected].clone()


def pose_anchor_centers_from_dataset(dataset, num_anchors: int) -> torch.Tensor:
    centers = []
    for record in dataset.records:
        sample_name = record["sample_name"].replace("\\", "/")
        pose = dataset.name_to_pose.get(sample_name)
        if pose is None:
            pose = dataset.basename_to_pose.get(Path(sample_name).name)
        if pose is not None:
            centers.append(camera_centers_from_w2c(pose.float().unsqueeze(0))[0])
    if not centers:
        raise RuntimeError(
            "Cannot build anchor pose initializer: no COLMAP pose_gt was found for the training split."
        )
    return farthest_point_anchor_centers(torch.stack(centers, dim=0), num_anchors=num_anchors)


def pose_anchors_from_dataset(dataset, num_anchors: int, sampling: str = "fps"):
    poses = []
    centers = []
    for record in dataset.records:
        sample_name = record["sample_name"].replace("\\", "/")
        pose = dataset.name_to_pose.get(sample_name)
        if pose is None:
            pose = dataset.basename_to_pose.get(Path(sample_name).name)
        if pose is not None:
            pose = pose.float()
            poses.append(pose)
            centers.append(camera_centers_from_w2c(pose.unsqueeze(0))[0])
    if not centers:
        raise RuntimeError(
            "Cannot build anchor pose initializer: no COLMAP pose_gt was found for the training split."
        )
    centers = torch.stack(centers, dim=0)
    sampling = str(sampling or "fps").lower()
    if sampling in {"all", "train_all", "pose_bank"}:
        rotmats = torch.stack([pose[:3, :3] for pose in poses], dim=0)
        return centers, rotmats
    if sampling != "fps":
        raise ValueError("pose_init_anchor_sampling must be one of {'fps', 'train_all'}")
    anchors = farthest_point_anchor_centers(centers, num_anchors=num_anchors)
    dist = torch.cdist(anchors, centers)
    nearest = dist.argmin(dim=1)
    rotmats = torch.stack([poses[int(idx.item())][:3, :3] for idx in nearest], dim=0)
    return anchors, rotmats


def _channel_mean_descriptor(feature):
    feature_t = torch.as_tensor(feature).float()
    if feature_t.ndim <= 1:
        return feature_t.reshape(-1)
    if feature_t.shape[0] == 1 and feature_t.ndim >= 3:
        feature_t = feature_t.squeeze(0)
    return feature_t.reshape(feature_t.shape[0], -1).mean(dim=1)


def _global_descriptor_from_teacher_pair(fine, coarse, feature_source: str = "fine_coarse"):
    feature_source = str(feature_source or "fine_coarse").lower()
    if feature_source not in {"fine_coarse", "fine", "coarse"}:
        raise ValueError("pose_init_feature_source must be one of {'fine_coarse', 'fine', 'coarse'}")
    fine_vec = _channel_mean_descriptor(fine)
    coarse_vec = _channel_mean_descriptor(coarse)
    if feature_source == "fine":
        return fine_vec
    if feature_source == "coarse":
        return coarse_vec
    return torch.cat([fine_vec, coarse_vec], dim=0)


def pose_feature_bank_from_dataset(
    dataset,
    teacher_store,
    num_anchors: int,
    sampling: str = "train_all",
    feature_source: str = "fine_coarse",
):
    poses = []
    centers = []
    descriptors = []
    feature_source_key = str(feature_source or "fine_coarse").lower()
    for record in dataset.records:
        sample_name = record["sample_name"].replace("\\", "/")
        pose = dataset.name_to_pose.get(sample_name)
        if pose is None:
            pose = dataset.basename_to_pose.get(Path(sample_name).name)
        if pose is None:
            continue
        if feature_source_key in {"retrieval", "summary", "global_retrieval"}:
            retrieval_store = getattr(dataset, "retrieval_teacher_store", None)
            if retrieval_store is None:
                raise ValueError("feature_source='retrieval' requires dataset.retrieval_teacher_store")
            descriptor = retrieval_store.load_record(record).float().view(-1)
        else:
            if teacher_store is None:
                raise ValueError("teacher_store is required unless feature_source='retrieval'")
            fine, coarse = teacher_store.load_pair(record["teacher_idx"])
            descriptor = _global_descriptor_from_teacher_pair(fine, coarse, feature_source=feature_source_key)
        pose = pose.float()
        poses.append(pose)
        centers.append(camera_centers_from_w2c(pose.unsqueeze(0))[0])
        descriptors.append(descriptor)
    if not centers:
        raise RuntimeError(
            "Cannot build feature-bank pose initializer: no COLMAP pose_gt was found for the training split."
        )
    centers = torch.stack(centers, dim=0)
    descriptors = torch.stack(descriptors, dim=0)
    sampling = str(sampling or "train_all").lower()
    if sampling in {"all", "train_all", "pose_bank", "feature_bank"}:
        rotmats = torch.stack([pose[:3, :3] for pose in poses], dim=0)
        return centers, rotmats, descriptors
    if sampling != "fps":
        raise ValueError("pose_init_anchor_sampling must be one of {'fps', 'train_all'}")
    anchors = farthest_point_anchor_centers(centers, num_anchors=num_anchors)
    dist = torch.cdist(anchors, centers)
    nearest = dist.argmin(dim=1)
    rotmats = torch.stack([poses[int(idx.item())][:3, :3] for idx in nearest], dim=0)
    desc = descriptors[nearest]
    return anchors, rotmats, desc


def pose_feature_bank_from_map_renderer(
    dataset,
    map_renderer,
    num_anchors: int,
    sampling: str = "train_all",
    feature_source: str = "fine_coarse",
):
    poses = []
    centers = []
    descriptors = []
    for record in dataset.records:
        sample_name = record["sample_name"].replace("\\", "/")
        pose = dataset.name_to_pose.get(sample_name)
        if pose is None:
            pose = dataset.basename_to_pose.get(Path(sample_name).name)
        if pose is None:
            continue
        with torch.no_grad():
            _fine_raw, fine, coarse, mask, *_rest = map_renderer._render_single(sample_name, require_grad=False)
            if mask is not None:
                mask_t = mask.float()
                if mask_t.shape[-2:] != fine.shape[-2:]:
                    mask_t = F.interpolate(mask_t, size=fine.shape[-2:], mode="nearest")
                denom = torch.clamp(mask_t.sum(dim=(-1, -2)), min=1.0)
                fine_vec = (fine.float() * mask_t).sum(dim=(-1, -2)) / denom
                if mask_t.shape[-2:] != coarse.shape[-2:]:
                    coarse_mask = F.interpolate(mask_t, size=coarse.shape[-2:], mode="nearest")
                else:
                    coarse_mask = mask_t
                coarse_denom = torch.clamp(coarse_mask.sum(dim=(-1, -2)), min=1.0)
                coarse_vec = (coarse.float() * coarse_mask).sum(dim=(-1, -2)) / coarse_denom
            else:
                fine_vec = fine.float().mean(dim=(-1, -2))
                coarse_vec = coarse.float().mean(dim=(-1, -2))
            descriptors.append(
                _global_descriptor_from_teacher_pair(
                    fine_vec.squeeze(0),
                    coarse_vec.squeeze(0),
                    feature_source=feature_source,
                )
            )
        pose = pose.float()
        poses.append(pose)
        centers.append(camera_centers_from_w2c(pose.unsqueeze(0))[0])
    if not centers:
        raise RuntimeError(
            "Cannot build feature-bank pose initializer: no COLMAP pose_gt was found for the training split."
        )
    centers = torch.stack(centers, dim=0)
    descriptors = torch.stack(descriptors, dim=0)
    sampling = str(sampling or "train_all").lower()
    if sampling in {"all", "train_all", "pose_bank", "feature_bank"}:
        rotmats = torch.stack([pose[:3, :3] for pose in poses], dim=0)
        return centers, rotmats, descriptors
    if sampling != "fps":
        raise ValueError("pose_init_anchor_sampling must be one of {'fps', 'train_all'}")
    anchors = farthest_point_anchor_centers(centers, num_anchors=num_anchors)
    dist = torch.cdist(anchors, centers)
    nearest = dist.argmin(dim=1)
    rotmats = torch.stack([poses[int(idx.item())][:3, :3] for idx in nearest], dim=0)
    desc = descriptors[nearest]
    return anchors, rotmats, desc


def refresh_pose_init_feature_bank_from_map_renderer(
    model,
    dataset,
    map_renderer,
    cfg,
    logger=None,
):
    model_cfg = cfg.get("model", {})
    if str(model_cfg.get("pose_init_mode", "direct")).lower() != "feature_bank":
        return False
    if str(model_cfg.get("pose_init_feature_bank_source", "teacher")).lower() != "map":
        return False
    if map_renderer is None:
        return False
    head = getattr(model, "pose_init_head", None)
    if head is None or not hasattr(head, "anchor_descriptors"):
        return False
    anchors, anchor_rotmats, anchor_desc = pose_feature_bank_from_map_renderer(
        dataset,
        map_renderer,
        num_anchors=int(model_cfg.get("pose_init_num_anchors", 64)),
        sampling=str(model_cfg.get("pose_init_anchor_sampling", "train_all")),
        feature_source=str(model_cfg.get("pose_init_feature_source", "fine_coarse")),
    )
    device = head.anchor_descriptors.device
    dtype = head.anchor_descriptors.dtype
    if tuple(anchor_desc.shape) != tuple(head.anchor_descriptors.shape):
        raise ValueError(
            "Refreshed pose init feature bank shape mismatch: "
            f"new={tuple(anchor_desc.shape)} existing={tuple(head.anchor_descriptors.shape)}"
        )
    if tuple(anchors.shape) != tuple(head.anchor_centers.shape):
        raise ValueError(
            "Refreshed pose init anchor center shape mismatch: "
            f"new={tuple(anchors.shape)} existing={tuple(head.anchor_centers.shape)}"
        )
    if tuple(anchor_rotmats.shape) != tuple(head.anchor_rotmats.shape):
        raise ValueError(
            "Refreshed pose init anchor rotation shape mismatch: "
            f"new={tuple(anchor_rotmats.shape)} existing={tuple(head.anchor_rotmats.shape)}"
        )
    with torch.no_grad():
        head.anchor_descriptors.copy_(F.normalize(anchor_desc.to(device=device, dtype=dtype), dim=1))
        head.anchor_centers.copy_(anchors.to(device=head.anchor_centers.device, dtype=head.anchor_centers.dtype))
        rotmats = anchor_rotmats.to(device=head.anchor_rotmats.device, dtype=head.anchor_rotmats.dtype)
        head.anchor_rotmats.copy_(rotmats)
        head.anchor_rot6d.copy_(matrix_to_rotation_6d(rotmats).to(device=head.anchor_rot6d.device, dtype=head.anchor_rot6d.dtype))
    if logger is not None:
        logger.info(
            "Refreshed pose init feature bank from current map renderer: count=%d desc_dim=%d",
            anchor_desc.shape[0],
            anchor_desc.shape[1],
        )
    return True


def compute_pose_init_losses(outputs, batch, cfg):
    pose_cfg = cfg.get("pose_init", {})
    device = outputs["fine"].device if "fine" in outputs else batch["pose_gt"].device
    if not pose_cfg.get("enabled", False):
        return torch.zeros((), device=device), {}
    pose_init = outputs.get("pose_init")
    pose_gt = batch.get("pose_gt", batch.get("rendered_map_pose_gt"))
    if pose_init is None or pose_gt is None:
        return torch.zeros((), device=device), {}

    top_centers = pose_init["center"].float()
    top_rotmat = pose_init["rotmat"].float()
    top_scores = pose_init["scores"].float()
    top_log_var = pose_init.get("log_var")
    use_all_anchor_pose_loss = bool(pose_cfg.get("use_all_anchor_pose_loss", False))
    if use_all_anchor_pose_loss and "all_center" in pose_init and "all_rotmat" in pose_init:
        centers = pose_init["all_center"].float()
        rotmat = pose_init["all_rotmat"].float()
        scores = pose_init.get("anchor_logits", top_scores).float()
        log_var = pose_init.get("all_log_var")
    else:
        centers = top_centers
        rotmat = top_rotmat
        scores = top_scores
        log_var = top_log_var
    pose_gt = pose_gt.to(device=centers.device, dtype=centers.dtype)
    gt_centers = camera_centers_from_w2c(pose_gt).unsqueeze(1)
    gt_rot = pose_gt[:, :3, :3].unsqueeze(1)

    trans_err_m = torch.linalg.norm(centers - gt_centers, dim=-1)
    rot_cos_loss, rot_err_deg = _rotation_error_from_mats(rotmat, gt_rot)
    cost = trans_err_m + float(pose_cfg.get("rot_cost_weight", 0.1)) * (rot_err_deg / 180.0)
    best_idx = cost.argmin(dim=1)
    batch_idx = torch.arange(centers.shape[0], device=centers.device)

    best_trans = trans_err_m[batch_idx, best_idx]
    best_rot_cos = rot_cos_loss[batch_idx, best_idx]
    best_rot_deg = rot_err_deg[batch_idx, best_idx]
    loss = (
        float(pose_cfg.get("trans_weight", 1.0)) * F.smooth_l1_loss(best_trans, torch.zeros_like(best_trans))
        + float(pose_cfg.get("rot_weight", 1.0)) * best_rot_cos.mean()
    )

    score_weight = float(pose_cfg.get("score_weight", 0.0))
    score_loss = torch.zeros((), device=centers.device)
    if score_weight > 0:
        score_loss = F.cross_entropy(scores, best_idx.detach())
        loss = loss + score_weight * score_loss

    uncertainty_weight = float(pose_cfg.get("uncertainty_weight", 0.0))
    uncertainty_loss = torch.zeros((), device=centers.device)
    if uncertainty_weight > 0 and log_var is not None:
        best_log_var = log_var.float()[batch_idx, best_idx]
        trans_nll = best_trans.square() * torch.exp(-best_log_var[:, 0]) + best_log_var[:, 0]
        rot_rad = best_rot_deg * math.pi / 180.0
        rot_nll = rot_rad.square() * torch.exp(-best_log_var[:, 1]) + best_log_var[:, 1]
        uncertainty_loss = 0.5 * (trans_nll + rot_nll).mean()
        loss = loss + uncertainty_weight * uncertainty_loss

    anchor_loss = torch.zeros((), device=centers.device)
    anchor_target_idx = None
    anchor_acc = torch.zeros((), device=centers.device)
    anchor_topk_recall = torch.zeros((), device=centers.device)
    anchor_target_dist_mm = torch.zeros((), device=centers.device)
    anchor_weight = float(pose_cfg.get("anchor_weight", 0.0))
    if anchor_weight > 0 and "anchor_logits" in pose_init and "anchor_centers" in pose_init:
        anchor_logits = pose_init["anchor_logits"].float()
        anchor_centers = pose_init["anchor_centers"].to(device=centers.device, dtype=centers.dtype)
        anchor_dist = torch.linalg.norm(anchor_centers.unsqueeze(0) - gt_centers, dim=-1)
        anchor_rot_err_deg = torch.zeros_like(anchor_dist)
        anchor_rotmats = pose_init.get("anchor_rotmats")
        if anchor_rotmats is not None:
            anchor_rot = anchor_rotmats.to(device=centers.device, dtype=centers.dtype)
            if anchor_rot.ndim == 3:
                anchor_rot = anchor_rot.unsqueeze(0).expand(anchor_dist.shape[0], -1, -1, -1)
            _anchor_rot_loss, anchor_rot_err_deg = _rotation_error_from_mats(anchor_rot, gt_rot)
        anchor_target_cost = anchor_dist + float(pose_cfg.get("rot_cost_weight", 0.1)) * (anchor_rot_err_deg / 180.0)
        anchor_target_idx = anchor_target_cost.argmin(dim=1)
        anchor_soft_sigma_m = float(pose_cfg.get("anchor_soft_sigma_m", 0.0))
        if anchor_soft_sigma_m > 0:
            sigma_m = max(anchor_soft_sigma_m, 1e-6)
            target_logits = -0.5 * (anchor_dist / sigma_m).square()
            anchor_soft_rot_sigma_deg = float(pose_cfg.get("anchor_soft_rot_sigma_deg", 0.0))
            if anchor_soft_rot_sigma_deg > 0 and anchor_rotmats is not None:
                target_logits = target_logits - 0.5 * (anchor_rot_err_deg / max(anchor_soft_rot_sigma_deg, 1e-6)).square()
            target_prob = F.softmax(target_logits, dim=1).detach()
            anchor_loss = -(target_prob * F.log_softmax(anchor_logits, dim=1)).sum(dim=1).mean()
        else:
            anchor_loss = F.cross_entropy(anchor_logits, anchor_target_idx)
        loss = loss + anchor_weight * anchor_loss
        with torch.no_grad():
            topk_idx = pose_init.get("anchor_indices")
            if topk_idx is None:
                topk = min(centers.shape[1], anchor_logits.shape[1])
                topk_idx = torch.topk(anchor_logits, k=topk, dim=1).indices
            topk_idx = topk_idx.to(device=centers.device)
            anchor_acc = (anchor_logits.argmax(dim=1) == anchor_target_idx).float().mean() * 100.0
            anchor_topk_recall = (topk_idx == anchor_target_idx.unsqueeze(1)).any(dim=1).float().mean() * 100.0
            anchor_target_dist_mm = anchor_dist[batch_idx, anchor_target_idx] * 1000.0
            topk_anchor_dist_mm = anchor_dist.gather(1, topk_idx) * 1000.0
            topk_anchor_rot_deg = anchor_rot_err_deg.gather(1, topk_idx)
            anchor_topk_recall_5deg_1000mm = (
                ((topk_anchor_rot_deg < 5.0) & (topk_anchor_dist_mm < 1000.0)).any(dim=1).float().mean() * 100.0
            )
            anchor_topk_recall_2deg_250mm = (
                ((topk_anchor_rot_deg < 2.0) & (topk_anchor_dist_mm < 250.0)).any(dim=1).float().mean() * 100.0
            )
            anchor_topk_recall_1deg_100mm = (
                ((topk_anchor_rot_deg < 1.0) & (topk_anchor_dist_mm < 100.0)).any(dim=1).float().mean() * 100.0
            )

    with torch.no_grad():
        pred_idx = scores.argmax(dim=1)
        pred_trans = trans_err_m[batch_idx, pred_idx]
        pred_rot = rot_err_deg[batch_idx, pred_idx]
        best_trans_mm = best_trans * 1000.0
        pred_trans_mm = pred_trans * 1000.0
        best_joint_5deg_1000mm = ((best_rot_deg < 5.0) & (best_trans_mm < 1000.0)).float().mean() * 100.0
        best_joint_1deg_100mm = ((best_rot_deg < 1.0) & (best_trans_mm < 100.0)).float().mean() * 100.0
        best_joint_1deg_50mm = ((best_rot_deg < 1.0) & (best_trans_mm < 50.0)).float().mean() * 100.0
        pred_joint_5deg_1000mm = ((pred_rot < 5.0) & (pred_trans_mm < 1000.0)).float().mean() * 100.0
        pred_joint_1deg_100mm = ((pred_rot < 1.0) & (pred_trans_mm < 100.0)).float().mean() * 100.0
        pred_joint_1deg_50mm = ((pred_rot < 1.0) & (pred_trans_mm < 50.0)).float().mean() * 100.0
    metrics = {
        "pose_init_loss": loss.detach(),
        "pose_init_score_loss": score_loss.detach(),
        "pose_init_uncertainty_loss": uncertainty_loss.detach(),
        "pose_init_anchor_loss": anchor_loss.detach(),
        "pose_init_best_idx": best_idx.float().mean().detach(),
        "pose_init_best_trans_mm": best_trans_mm.mean().detach(),
        "pose_init_best_rot_deg": best_rot_deg.mean().detach(),
        "pose_init_pred_trans_mm": pred_trans_mm.mean().detach(),
        "pose_init_pred_rot_deg": pred_rot.mean().detach(),
        "pose_init_best_joint_5deg_1000mm": best_joint_5deg_1000mm.detach(),
        "pose_init_best_joint_1deg_100mm": best_joint_1deg_100mm.detach(),
        "pose_init_best_joint_1deg_50mm": best_joint_1deg_50mm.detach(),
        "pose_init_pred_joint_5deg_1000mm": pred_joint_5deg_1000mm.detach(),
        "pose_init_pred_joint_1deg_100mm": pred_joint_1deg_100mm.detach(),
        "pose_init_pred_joint_1deg_50mm": pred_joint_1deg_50mm.detach(),
        # Backwards-compatible aliases used by older logs/config dashboards. These are top-K oracle recalls.
        "pose_init_joint_5deg_1000mm": best_joint_5deg_1000mm.detach(),
        "pose_init_joint_1deg_100mm": best_joint_1deg_100mm.detach(),
        "pose_init_joint_1deg_50mm": best_joint_1deg_50mm.detach(),
    }
    if anchor_target_idx is not None:
        metrics.update(
            {
                "pose_init_anchor_target_idx": anchor_target_idx.float().mean().detach(),
                "pose_init_anchor_acc": anchor_acc.detach(),
                "pose_init_anchor_topk_recall": anchor_topk_recall.detach(),
                "pose_init_anchor_topk_recall_5deg_1000mm": anchor_topk_recall_5deg_1000mm.detach(),
                "pose_init_anchor_topk_recall_2deg_250mm": anchor_topk_recall_2deg_250mm.detach(),
                "pose_init_anchor_topk_recall_1deg_100mm": anchor_topk_recall_1deg_100mm.detach(),
                "pose_init_anchor_target_dist_mm": anchor_target_dist_mm.mean().detach(),
            }
        )
    return loss, metrics


def descriptor_pose_retrieval_metrics(
    query_desc,
    query_pose,
    bank_desc,
    bank_pose,
    *,
    topk=(1, 5, 10),
    prefix="retrieval",
):
    """Evaluate descriptor retrieval as a pose initializer."""
    query_n = F.normalize(query_desc.float(), dim=1)
    bank_n = F.normalize(bank_desc.float(), dim=1)
    query_pose_f = query_pose.float()
    bank_pose_f = bank_pose.float()
    max_k = min(max(int(k) for k in topk), bank_n.shape[0])
    sim = torch.matmul(query_n, bank_n.t())
    top_idx = torch.topk(sim, k=max_k, dim=1).indices
    q_centers = camera_centers_from_w2c(query_pose_f)
    b_centers = camera_centers_from_w2c(bank_pose_f)
    q_rot = query_pose_f[:, :3, :3]
    b_rot = bank_pose_f[:, :3, :3]
    gathered_centers = b_centers[top_idx]
    gathered_rot = b_rot[top_idx]
    trans_mm = torch.linalg.norm(gathered_centers - q_centers.unsqueeze(1), dim=-1) * 1000.0
    _rot_loss, rot_deg = _rotation_error_from_mats(gathered_rot, q_rot.unsqueeze(1))
    metrics = {}
    for raw_k in topk:
        k = min(int(raw_k), max_k)
        trans_k = trans_mm[:, :k]
        rot_k = rot_deg[:, :k]
        best_cost = trans_k / 1000.0 + 0.1 * (rot_k / 180.0)
        best_idx = best_cost.argmin(dim=1)
        batch_idx = torch.arange(trans_k.shape[0], device=trans_k.device)
        best_trans = trans_k[batch_idx, best_idx]
        best_rot = rot_k[batch_idx, best_idx]
        metrics[f"{prefix}_top{k}_best_trans_mm"] = best_trans.mean().detach()
        metrics[f"{prefix}_top{k}_best_rot_deg"] = best_rot.mean().detach()
        metrics[f"{prefix}_top{k}_recall_5deg_1000mm"] = (
            ((best_rot < 5.0) & (best_trans < 1000.0)).float().mean() * 100.0
        ).detach()
        metrics[f"{prefix}_top{k}_recall_2deg_250mm"] = (
            ((best_rot < 2.0) & (best_trans < 250.0)).float().mean() * 100.0
        ).detach()
        metrics[f"{prefix}_top{k}_recall_1deg_100mm"] = (
            ((best_rot < 1.0) & (best_trans < 100.0)).float().mean() * 100.0
        ).detach()
        if k == 1:
            metrics[f"{prefix}_top1_trans_mm"] = trans_k[:, 0].mean().detach()
            metrics[f"{prefix}_top1_rot_deg"] = rot_k[:, 0].mean().detach()
    return metrics


def topk_descriptor_candidate_indices(query_desc: torch.Tensor, bank_desc: torch.Tensor, k: int):
    """Return top-K bank indices by L2-normalized descriptor cosine."""
    if query_desc.ndim != 2 or bank_desc.ndim != 2:
        raise ValueError("query_desc and bank_desc must be 2D tensors")
    if query_desc.shape[1] != bank_desc.shape[1]:
        raise ValueError(
            f"descriptor dim mismatch: query={query_desc.shape[1]} bank={bank_desc.shape[1]}"
        )
    k_i = max(1, min(int(k), int(bank_desc.shape[0])))
    query_n = F.normalize(query_desc.float(), dim=1)
    bank_n = F.normalize(bank_desc.float(), dim=1)
    sim = query_n @ bank_n.t()
    values, indices = torch.topk(sim, k=k_i, dim=1)
    return indices.long(), values


def select_pose_candidate_indices(num_items, limit=None, scores=None, strategy="all"):
    """Select pose-bank candidate indices for render-score diagnostics/training."""
    num_items = int(num_items)
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if limit is None:
        limit_i = num_items
    else:
        limit_i = max(1, min(int(limit), num_items))
    strategy_key = str(strategy).lower()
    if strategy_key in ("all", "none") or limit_i >= num_items:
        device = scores.device if torch.is_tensor(scores) else torch.device("cpu")
        return torch.arange(num_items, device=device, dtype=torch.long)
    if strategy_key in ("head", "first", "prefix"):
        device = scores.device if torch.is_tensor(scores) else torch.device("cpu")
        return torch.arange(limit_i, device=device, dtype=torch.long)
    if strategy_key in ("score", "scores", "anchor_score", "anchor_logits"):
        if scores is None:
            raise ValueError(f"strategy '{strategy}' requires scores")
        score_t = torch.as_tensor(scores)
        if score_t.numel() != num_items:
            raise ValueError(f"scores length {score_t.numel()} does not match num_items={num_items}")
        return torch.topk(score_t.float().view(-1), k=limit_i, dim=0).indices.long()
    if strategy_key in ("uniform", "linspace"):
        return torch.linspace(0, num_items - 1, steps=limit_i).round().long()
    raise ValueError(f"Unknown candidate selection strategy '{strategy}'")


def _render_score_preprocess(query, candidates, mode="none", highpass_kernel=5):
    mode_key = str(mode or "none").lower()
    if mode_key in ("none", "raw"):
        return query, candidates
    if mode_key in ("spatial_center", "center", "channel_center"):
        query = query - query.mean(dim=(-2, -1), keepdim=True)
        candidates = candidates - candidates.mean(dim=(-2, -1), keepdim=True)
        return query, candidates
    if mode_key in ("spatial_zscore", "zscore", "channel_zscore"):
        query = query - query.mean(dim=(-2, -1), keepdim=True)
        candidates = candidates - candidates.mean(dim=(-2, -1), keepdim=True)
        query = query / query.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        candidates = candidates / candidates.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return query, candidates
    if mode_key in ("highpass", "spatial_highpass"):
        kernel = max(1, int(highpass_kernel))
        if kernel % 2 == 0:
            kernel += 1
        padding = kernel // 2
        query = query - F.avg_pool2d(query, kernel_size=kernel, stride=1, padding=padding)
        candidates = candidates - F.avg_pool2d(candidates, kernel_size=kernel, stride=1, padding=padding)
        return query, candidates
    raise ValueError(f"Unknown render-score preprocess '{mode}'")


def _masked_score_map_stats(score_map, mask=None, *, topk=0.02, entropy_temperature=0.1):
    """Summarize per-candidate score maps without collapsing away peak structure."""
    if score_map.ndim == 3:
        score_map = score_map.unsqueeze(1)
    if score_map.ndim != 4:
        raise ValueError("score_map must have shape (K,1,H,W) or (K,H,W)")
    values = score_map.float()
    if mask is None:
        mask_t = torch.ones_like(values[:, :1], dtype=torch.bool)
    else:
        mask_t = mask.to(device=values.device)
        if mask_t.ndim == 3:
            mask_t = mask_t.unsqueeze(1)
        if mask_t.ndim != 4:
            raise ValueError("mask must have shape (K,1,H,W), (K,H,W), or (1,H,W)")
        if mask_t.shape[0] == 1 and values.shape[0] > 1:
            mask_t = mask_t.expand(values.shape[0], -1, -1, -1)
        if mask_t.shape[-2:] != values.shape[-2:]:
            mask_t = F.interpolate(mask_t.float(), size=values.shape[-2:], mode="nearest")
        mask_t = mask_t.bool()
    mask_f = mask_t.to(dtype=values.dtype)
    denom = mask_f.sum(dim=(1, 2, 3)).clamp(min=1.0)
    mean = (values * mask_f).sum(dim=(1, 2, 3)) / denom
    centered = torch.where(mask_t, values - mean.view(-1, 1, 1, 1), torch.zeros_like(values))
    std = ((centered.square() * mask_f).sum(dim=(1, 2, 3)) / denom).sqrt()

    flat = values.flatten(1)
    flat_mask = mask_t.flatten(1)
    masked_flat = flat.masked_fill(~flat_mask, -1.0e6)
    max_score = masked_flat.max(dim=1).values
    max_score = torch.where(torch.isfinite(max_score) & (max_score > -1.0e5), max_score, torch.zeros_like(max_score))
    topk_f = float(topk)
    if topk_f < 1.0:
        k = int(math.ceil(float(flat.shape[1]) * max(topk_f, 1.0 / float(flat.shape[1]))))
    else:
        k = int(topk_f)
    k = max(1, min(k, flat.shape[1]))
    top_values = torch.topk(masked_flat, k=k, dim=1).values
    top_valid = top_values > -1.0e5
    top_sum = torch.where(top_valid, top_values, torch.zeros_like(top_values)).sum(dim=1)
    top_count = top_valid.float().sum(dim=1).clamp(min=1.0)
    topk_mean = top_sum / top_count

    temp = max(float(entropy_temperature), 1e-6)
    logits = (flat / temp).masked_fill(~flat_mask, -1.0e6)
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1)
    valid_count = flat_mask.float().sum(dim=1).clamp(min=1.0)
    entropy_norm = entropy / torch.log(valid_count.clamp(min=2.0))
    peakiness = (1.0 - entropy_norm).clamp(min=0.0, max=1.0)

    return {
        "mean": mean,
        "valid_fraction": mask_f.mean(dim=(1, 2, 3)),
        "std": std,
        "max": max_score,
        "topk_mean": topk_mean,
        "peakiness": peakiness,
    }


def render_score_feature_candidates(query_feat, candidate_feat, mask=None, preprocess="none", highpass_kernel=5):
    """Score rendered pose candidates by masked query-map feature cosine.

    Args:
        query_feat: (C,H,W) or (1,C,H,W) query feature map.
        candidate_feat: (K,C,H,W) rendered candidate feature maps.
        mask: optional (K,1,H,W), (K,H,W), or (1,H,W) candidate validity mask.
        preprocess: optional raw/spatial_center/spatial_zscore/highpass before cosine.

    Returns:
        Dict with per-candidate cosine scores, best index, and top-1 margin.
    """
    if query_feat.ndim == 3:
        query = query_feat.unsqueeze(0)
    elif query_feat.ndim == 4 and query_feat.shape[0] == 1:
        query = query_feat
    else:
        raise ValueError("query_feat must have shape (C,H,W) or (1,C,H,W)")
    if candidate_feat.ndim != 4:
        raise ValueError("candidate_feat must have shape (K,C,H,W)")
    if candidate_feat.shape[1] != query.shape[1]:
        raise ValueError(
            f"candidate channel dim {candidate_feat.shape[1]} does not match query dim {query.shape[1]}"
        )
    candidates = candidate_feat.float()
    query = query.to(device=candidates.device, dtype=candidates.dtype)
    if candidates.shape[-2:] != query.shape[-2:]:
        candidates = F.interpolate(candidates, size=query.shape[-2:], mode="bilinear", align_corners=False)
    query_expand = query.expand(candidates.shape[0], -1, -1, -1)
    query_expand, candidates = _render_score_preprocess(
        query_expand,
        candidates,
        mode=preprocess,
        highpass_kernel=highpass_kernel,
    )
    query_n = F.normalize(query_expand, dim=1)
    candidate_n = F.normalize(candidates, dim=1)
    cos_map = (query_n * candidate_n).sum(dim=1, keepdim=True)
    score_stats = _masked_score_map_stats(cos_map, mask)
    scores = score_stats["mean"]
    valid_fraction = score_stats["valid_fraction"]
    best_idx = scores.argmax()
    if scores.numel() > 1:
        top2 = torch.topk(scores, k=2).values
        margin = top2[0] - top2[1]
    else:
        margin = torch.zeros((), device=scores.device, dtype=scores.dtype)
    return {
        "scores": scores,
        "best_idx": best_idx,
        "best_score": scores[best_idx],
        "score_margin": margin,
        "valid_fraction": valid_fraction,
        "score_stats": score_stats,
        "score_map": cos_map,
    }


def _local_correlation_peak_score_map(corr, radius):
    peak_map, peak_idx = corr.max(dim=1, keepdim=True)
    radius = int(radius)
    if radius <= 0:
        zero = torch.zeros_like(peak_map)
        return torch.cat([peak_map, zero, zero], dim=1)
    window = 2 * radius + 1
    peak_idx_f = peak_idx.float()
    dx = torch.remainder(peak_idx_f, float(window)) - float(radius)
    dy = torch.div(peak_idx_f, float(window), rounding_mode="floor") - float(radius)
    return torch.cat([peak_map, dx / float(radius), dy / float(radius)], dim=1)


def _local_correlation_score_map(corr, radius, mode="peak_offset"):
    mode_key = str(mode or "peak_offset").lower()
    if mode_key in ("peak", "peak_offset", "peak-offset", "summary"):
        return _local_correlation_peak_score_map(corr, radius=int(radius))
    if mode_key in ("volume", "corr_volume", "correlation_volume"):
        return corr
    if mode_key in (
        "volume_plus_peak_offset",
        "volume+peak_offset",
        "volume_peak_offset",
        "corr_volume_plus_peak_offset",
    ):
        peak_offset = _local_correlation_peak_score_map(corr, radius=int(radius))
        return torch.cat([corr, peak_offset], dim=1)
    raise ValueError(
        "candidate score_map_mode must be 'peak_offset', 'volume', or 'volume_plus_peak_offset'"
    )


def local_render_score_feature_candidates(
    query_feat,
    candidate_feat,
    mask=None,
    *,
    radius=4,
    preprocess="none",
    highpass_kernel=5,
    score_map_mode="peak_offset",
):
    """Score candidates by best local query-map feature correlation."""
    if query_feat.ndim == 3:
        query = query_feat.unsqueeze(0)
    elif query_feat.ndim == 4 and query_feat.shape[0] == 1:
        query = query_feat
    else:
        raise ValueError("query_feat must have shape (C,H,W) or (1,C,H,W)")
    if candidate_feat.ndim != 4:
        raise ValueError("candidate_feat must have shape (K,C,H,W)")
    if candidate_feat.shape[1] != query.shape[1]:
        raise ValueError(
            f"candidate channel dim {candidate_feat.shape[1]} does not match query dim {query.shape[1]}"
        )
    candidates = candidate_feat.float()
    query = query.to(device=candidates.device, dtype=candidates.dtype)
    if candidates.shape[-2:] != query.shape[-2:]:
        candidates = F.interpolate(candidates, size=query.shape[-2:], mode="bilinear", align_corners=False)
    query_expand = query.expand(candidates.shape[0], -1, -1, -1)
    query_expand, candidates = _render_score_preprocess(
        query_expand,
        candidates,
        mode=preprocess,
        highpass_kernel=highpass_kernel,
    )
    query_n = F.normalize(query_expand, dim=1)
    candidate_n = F.normalize(candidates, dim=1)
    corr = shifted_local_correlation(query_n, candidate_n, radius=int(radius))
    score_map = _local_correlation_score_map(corr, radius=int(radius), mode=score_map_mode)
    peak_map = corr.max(dim=1, keepdim=True).values
    score_stats = _masked_score_map_stats(peak_map, mask)
    scores = score_stats["mean"]
    valid_fraction = score_stats["valid_fraction"]
    best_idx = scores.argmax()
    if scores.numel() > 1:
        top2 = torch.topk(scores, k=2).values
        margin = top2[0] - top2[1]
    else:
        margin = torch.zeros((), device=scores.device, dtype=scores.dtype)
    return {
        "scores": scores,
        "best_idx": best_idx,
        "best_score": scores[best_idx],
        "score_margin": margin,
        "valid_fraction": valid_fraction,
        "score_stats": score_stats,
        "score_map": score_map,
    }


def _masked_row_standardize(values, valid_mask):
    values = values.float()
    valid = valid_mask.to(device=values.device).bool()
    finite = torch.isfinite(values)
    values = torch.where(finite, values, torch.zeros_like(values))
    usable = valid & finite
    denom = usable.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (values * usable.float()).sum(dim=1, keepdim=True) / denom
    centered = torch.where(usable, values - mean, torch.zeros_like(values))
    var = (centered.square() * usable.float()).sum(dim=1, keepdim=True) / denom
    std = var.sqrt().clamp(min=1e-6)
    return torch.where(valid, centered / std, torch.zeros_like(values))


def _candidate_batch_value(batch, key, valid_mask):
    value = batch.get(key)
    if value is None:
        return torch.zeros(valid_mask.shape, device=valid_mask.device, dtype=torch.float32)
    value_t = value.to(device=valid_mask.device, dtype=torch.float32)
    if value_t.shape != valid_mask.shape:
        raise ValueError(f"{key} must have shape {tuple(valid_mask.shape)}, got {tuple(value_t.shape)}")
    return torch.where(torch.isfinite(value_t), value_t, torch.zeros_like(value_t))


def candidate_quality_features_from_batch(batch, *, valid_mask=None):
    """Build fixed per-candidate retrieval/PnP prior features from a training batch."""
    if valid_mask is None:
        source = next(
            (batch.get(key) for key in ("retrieval_scores_candidates",) + OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS if key in batch),
            None,
        )
        if source is None:
            raise ValueError("valid_mask is required when batch has no candidate quality fields")
        valid_mask = torch.ones_like(source, dtype=torch.bool)
    valid = valid_mask.bool()
    device = valid.device
    features = []

    retrieval_scores = _candidate_batch_value(batch, "retrieval_scores_candidates", valid)
    retrieval_scores = torch.log1p(torch.clamp(retrieval_scores, min=0.0))
    features.append(_masked_row_standardize(retrieval_scores, valid))

    original_scores = _candidate_batch_value(batch, "retrieval_original_scores_candidates", valid)
    features.append(torch.where(valid, torch.clamp(original_scores, min=-1.0, max=1.0), torch.zeros_like(original_scores)))

    pnp_success = _candidate_batch_value(batch, "retrieval_pnp_success_candidates", valid)
    features.append(torch.where(valid, torch.clamp(pnp_success, min=0.0, max=1.0), torch.zeros_like(pnp_success)))

    num_inliers = torch.log1p(torch.clamp(_candidate_batch_value(batch, "retrieval_pnp_num_inliers_candidates", valid), min=0.0))
    features.append(_masked_row_standardize(num_inliers, valid))

    num_matches = torch.log1p(torch.clamp(_candidate_batch_value(batch, "retrieval_pnp_num_matches_candidates", valid), min=0.0))
    features.append(_masked_row_standardize(num_matches, valid))

    reproj_rmse = batch.get("retrieval_pnp_reproj_rmse_candidates")
    if reproj_rmse is None:
        rmse_quality = torch.zeros(valid.shape, device=device, dtype=torch.float32)
    else:
        rmse = reproj_rmse.to(device=device, dtype=torch.float32)
        if rmse.shape != valid.shape:
            raise ValueError(
                f"retrieval_pnp_reproj_rmse_candidates must have shape {tuple(valid.shape)}, got {tuple(rmse.shape)}"
            )
        rmse_quality = torch.where(torch.isfinite(rmse), 1.0 / (1.0 + torch.clamp(rmse, min=0.0)), torch.zeros_like(rmse))
    features.append(torch.where(valid, rmse_quality, torch.zeros_like(rmse_quality)))

    reproj_median = batch.get("retrieval_pnp_reproj_median_candidates")
    if reproj_median is None:
        median_quality = torch.zeros(valid.shape, device=device, dtype=torch.float32)
    else:
        median = reproj_median.to(device=device, dtype=torch.float32)
        if median.shape != valid.shape:
            raise ValueError(
                f"retrieval_pnp_reproj_median_candidates must have shape {tuple(valid.shape)}, got {tuple(median.shape)}"
            )
        median_quality = torch.where(
            torch.isfinite(median),
            1.0 / (1.0 + torch.clamp(median, min=0.0)),
            torch.zeros_like(median),
        )
    features.append(torch.where(valid, median_quality, torch.zeros_like(median_quality)))

    inlier_ratio = _candidate_batch_value(batch, "retrieval_pnp_inlier_ratio_candidates", valid)
    features.append(torch.where(valid, torch.clamp(inlier_ratio, min=0.0, max=1.0), torch.zeros_like(inlier_ratio)))

    inlier_conf = _candidate_batch_value(batch, "retrieval_pnp_inlier_conf_mean_candidates", valid)
    features.append(torch.where(valid, torch.clamp(inlier_conf, min=0.0, max=1.0), torch.zeros_like(inlier_conf)))

    stacked = torch.stack(features, dim=-1).to(device=device, dtype=torch.float32)
    return stacked, list(CANDIDATE_QUALITY_FEATURE_NAMES)


def _candidate_pose_target(candidate_pose, pose_gt, valid, rot_cost_weight):
    bsz, num_candidates = candidate_pose.shape[:2]
    cand_pose_f = candidate_pose.float()
    pose_gt_f = pose_gt.to(device=cand_pose_f.device, dtype=cand_pose_f.dtype)
    cand_centers = camera_centers_from_w2c(cand_pose_f.reshape(bsz * num_candidates, 4, 4)).reshape(
        bsz,
        num_candidates,
        3,
    )
    gt_centers = camera_centers_from_w2c(pose_gt_f).unsqueeze(1)
    trans_err_m = torch.linalg.norm(cand_centers - gt_centers, dim=-1)
    _rot_loss, rot_err_deg = _rotation_error_from_mats(cand_pose_f[:, :, :3, :3], pose_gt_f[:, None, :3, :3])
    cost = trans_err_m + float(rot_cost_weight) * (rot_err_deg / 180.0)
    cost_for_target = cost.masked_fill(~valid, float("inf"))
    target_idx = cost_for_target.argmin(dim=1)
    return target_idx, trans_err_m, rot_err_deg, cost


def gather_candidate_bank(values, indices):
    """Gather arbitrary candidate-bank entries along dimension 1."""
    if values is None:
        return None
    if indices.ndim != 2:
        raise ValueError(f"indices must have shape (B,M), got {tuple(indices.shape)}")
    if values.ndim < 2:
        raise ValueError(f"candidate bank values must have at least 2 dimensions, got {tuple(values.shape)}")
    if values.shape[0] != indices.shape[0]:
        raise ValueError(
            f"candidate bank batch size {values.shape[0]} does not match indices batch size {indices.shape[0]}"
        )
    gather_idx = indices.to(device=values.device, dtype=torch.long)
    if gather_idx.numel() > 0:
        if gather_idx.min().item() < 0 or gather_idx.max().item() >= values.shape[1]:
            raise IndexError(
                f"candidate gather indices must be in [0, {values.shape[1] - 1}], "
                f"got min={gather_idx.min().item()} max={gather_idx.max().item()}"
            )
    view_shape = tuple(gather_idx.shape) + (1,) * (values.ndim - 2)
    expand_shape = tuple(gather_idx.shape) + tuple(values.shape[2:])
    return values.gather(1, gather_idx.reshape(view_shape).expand(expand_shape))


def select_candidate_stage2_indices(
    *,
    valid,
    topm,
    selection="pred",
    logits=None,
    pose_cost=None,
):
    """Select topM candidate indices for second-stage differentiable refinement."""
    if valid.ndim != 2:
        raise ValueError(f"valid must have shape (B,K), got {tuple(valid.shape)}")
    valid = valid.bool()
    num_candidates = valid.shape[1]
    keep = max(1, min(int(topm or 1), num_candidates))
    mode = str(selection or "pred").lower()
    if mode in ("pred", "score", "scorer"):
        if logits is None:
            raise ValueError("candidate_stage2_selection='pred' requires scorer logits")
        scores = logits.to(device=valid.device, dtype=torch.float32).masked_fill(~valid, -1.0e6)
    elif mode in ("oracle", "target", "teacher_forcing"):
        if pose_cost is None:
            raise ValueError("candidate_stage2_selection='oracle' requires pose costs")
        scores = (-pose_cost.to(device=valid.device, dtype=torch.float32)).masked_fill(~valid, -1.0e6)
    else:
        raise ValueError("candidate_stage2_selection must be 'pred' or 'oracle'")
    return torch.topk(scores, k=keep, dim=1).indices


def _candidate_wls_refined_pose_cost(
    query_feat,
    candidate_feat,
    candidate_pose,
    pose_gt,
    candidate_depth,
    candidate_intrinsics,
    valid,
    *,
    radius=4,
    temperature=0.05,
    damping=1e-3,
    update_scale=1.0,
    rot_cost_weight=0.1,
    wls_conf_mode="max",
    wls_conf_variance_scale=0.5,
    wls_downsample=1,
):
    """Compute per-candidate pose cost after one differentiable local-corr WLS update."""
    bsz, num_candidates, channels, height, width = candidate_feat.shape
    device = candidate_feat.device
    dtype = candidate_feat.dtype
    query = query_feat.to(device=device, dtype=dtype)
    if query.shape[-2:] != (height, width):
        query = F.interpolate(query, size=(height, width), mode="bilinear", align_corners=False)
    query_flat = query[:, None].expand(-1, num_candidates, -1, -1, -1).reshape(
        bsz * num_candidates,
        channels,
        height,
        width,
    )
    candidate_flat = candidate_feat.reshape(bsz * num_candidates, channels, height, width).float()
    query_flat = query_flat.float()

    depth = candidate_depth.to(device=device, dtype=torch.float32)
    if depth.ndim == 5:
        if depth.shape[2] == 1:
            depth = depth[:, :, 0]
        else:
            raise ValueError(
                "candidate_depth with 5 dimensions must have shape (B,K,1,H,W), "
                f"got {tuple(candidate_depth.shape)}"
            )
    elif depth.ndim != 4:
        raise ValueError(f"candidate_depth must have shape (B,K,H,W) or (B,K,1,H,W), got {tuple(depth.shape)}")
    depth_flat = depth.reshape(bsz * num_candidates, depth.shape[-2], depth.shape[-1])
    if depth_flat.shape[-2:] != (height, width):
        depth_flat = F.interpolate(
            depth_flat.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    intr = candidate_intrinsics.to(device=device, dtype=torch.float32)
    if intr.ndim == 2:
        intr = intr[:, None].expand(-1, num_candidates, -1)
    if intr.shape != (bsz, num_candidates, 4):
        raise ValueError(
            f"candidate_intrinsics must have shape {(bsz, num_candidates, 4)} or {(bsz, 4)}, "
            f"got {tuple(intr.shape)}"
        )
    intr_flat = intr.reshape(bsz * num_candidates, 4)
    downsample = int(wls_downsample or 1)
    if downsample > 1:
        target_height = max(1, int(round(height / float(downsample))))
        target_width = max(1, int(round(width / float(downsample))))
        if (target_height, target_width) != (height, width):
            query_flat = F.interpolate(query_flat, size=(target_height, target_width), mode="bilinear", align_corners=False)
            candidate_flat = F.interpolate(
                candidate_flat,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )
            depth_flat = F.interpolate(
                depth_flat.unsqueeze(1),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            scale_x = float(target_width) / float(width)
            scale_y = float(target_height) / float(height)
            intr_flat = intr_flat.clone()
            intr_flat[:, 0] *= scale_x
            intr_flat[:, 2] *= scale_x
            intr_flat[:, 1] *= scale_y
            intr_flat[:, 3] *= scale_y
            height, width = target_height, target_width
    pose_ref_flat = candidate_pose.to(device=device, dtype=torch.float32).reshape(bsz * num_candidates, 4, 4)
    pose_gt_flat = pose_gt.to(device=device, dtype=torch.float32)[:, None].expand(
        -1,
        num_candidates,
        -1,
        -1,
    ).reshape(bsz * num_candidates, 4, 4)
    valid_flat = valid.to(device=device).bool().reshape(bsz * num_candidates)

    with torch.cuda.amp.autocast(enabled=False):
        rendered_n = F.normalize(candidate_flat, dim=1)
        query_n = F.normalize(query_flat, dim=1)
        corr = shifted_local_correlation(rendered_n, query_n, radius=int(radius)).float()
        dx, dy = _local_correlation_offsets(int(radius), corr.device, corr.dtype)
        probs = torch.softmax(corr / max(float(temperature), 1e-6), dim=1)
        flow = torch.cat(
            [
                (probs * dx).sum(dim=1, keepdim=True),
                (probs * dy).sum(dim=1, keepdim=True),
            ],
            dim=1,
        )
        confidence = correlation_confidence_from_probs(
            probs,
            radius=int(radius),
            mode=wls_conf_mode,
            variance_scale=wls_conf_variance_scale,
        )
        valid_weight = (depth_flat > 0.05).unsqueeze(1).float()
        valid_weight = valid_weight * valid_flat.to(dtype=valid_weight.dtype).view(-1, 1, 1, 1)
        Ju, Jv, depth_valid = compute_image_jacobian(depth_flat, intr_flat)
        delta_xi = diff_pose_solve(
            flow,
            (confidence * valid_weight).expand(-1, 2, -1, -1).contiguous(),
            Ju,
            Jv,
            depth_valid,
            damping=float(damping),
        )
        pose_pred = apply_pose_delta(pose_ref_flat.float(), delta_xi.float(), scale=float(update_scale))
        _rot_loss, rot_err_deg, trans_err_m = pose_error_tensors(pose_pred, pose_gt_flat)
        _init_rot_loss, init_rot_err_deg, init_trans_err_m = pose_error_tensors(pose_ref_flat, pose_gt_flat)
        cost = trans_err_m + float(rot_cost_weight) * (rot_err_deg / 180.0)
        cost = cost.reshape(bsz, num_candidates).masked_fill(~valid, float("inf"))
        trans_err_m = trans_err_m.reshape(bsz, num_candidates)
        rot_err_deg = rot_err_deg.reshape(bsz, num_candidates)
        init_trans_err_m = init_trans_err_m.reshape(bsz, num_candidates)
        init_rot_err_deg = init_rot_err_deg.reshape(bsz, num_candidates)
        conf_denom = valid_weight.sum(dim=(-1, -2, -3)).clamp(min=1.0)
        conf_mean = (confidence * valid_weight).sum(dim=(-1, -2, -3)) / conf_denom
        conf_mean = conf_mean.reshape(bsz, num_candidates)
    return {
        "cost": cost,
        "trans_err_m": trans_err_m,
        "rot_err_deg": rot_err_deg,
        "init_trans_err_m": init_trans_err_m,
        "init_rot_err_deg": init_rot_err_deg,
        "conf_mean": conf_mean,
    }


def candidate_refined_pose_loss(
    query_feat,
    candidate_feat,
    candidate_pose,
    pose_gt,
    candidate_depth,
    candidate_intrinsics,
    valid,
    *,
    radius=4,
    temperature=0.05,
    damping=1e-3,
    update_scale=1.0,
    rot_cost_weight=0.1,
    wls_conf_mode="max",
    wls_conf_variance_scale=0.5,
    wls_downsample=1,
):
    """Optimize selected candidates by one differentiable local-corr WLS pose update."""
    refined = _candidate_wls_refined_pose_cost(
        query_feat,
        candidate_feat,
        candidate_pose,
        pose_gt,
        candidate_depth,
        candidate_intrinsics,
        valid.bool(),
        radius=int(radius),
        temperature=float(temperature),
        damping=float(damping),
        update_scale=float(update_scale),
        rot_cost_weight=float(rot_cost_weight),
        wls_conf_mode=wls_conf_mode,
        wls_conf_variance_scale=float(wls_conf_variance_scale),
        wls_downsample=int(wls_downsample or 1),
    )
    valid = valid.to(device=refined["cost"].device).bool()
    usable = valid & torch.isfinite(refined["cost"])
    denom = usable.float().sum().clamp(min=1.0)
    loss = (refined["cost"].masked_fill(~usable, 0.0) * usable.float()).sum() / denom
    metrics = {
        "map_candidate_refined_pose_loss": loss.detach(),
        "map_candidate_refined_pose_trans_mm": (
            refined["trans_err_m"].masked_fill(~usable, 0.0) * usable.float() * 1000.0
        ).sum().detach()
        / denom.detach(),
        "map_candidate_refined_pose_rot_deg": (
            refined["rot_err_deg"].masked_fill(~usable, 0.0) * usable.float()
        ).sum().detach()
        / denom.detach(),
        "map_candidate_refined_pose_selected_init_trans_mm": (
            refined["init_trans_err_m"].masked_fill(~usable, 0.0) * usable.float() * 1000.0
        ).sum().detach()
        / denom.detach(),
        "map_candidate_refined_pose_conf_mean": (
            refined["conf_mean"].masked_fill(~usable, 0.0) * usable.float()
        ).sum().detach()
        / denom.detach(),
        "map_candidate_refined_pose_valid_fraction": valid.float().mean().detach(),
    }
    return loss, metrics


def _candidate_render_score_rows(
    query_feat,
    candidate_feat,
    mask=None,
    *,
    mode="local",
    radius=4,
    preprocess="none",
    highpass_kernel=5,
    score_map_mode="peak_offset",
    return_score_maps=False,
):
    bsz = query_feat.shape[0]
    score_rows = []
    valid_fraction_rows = []
    stats_rows = {key: [] for key in ("std", "max", "topk_mean", "peakiness")}
    score_map_rows = []
    mode_key = str(mode or "local").lower()
    for batch_idx in range(bsz):
        mask_i = mask[batch_idx] if mask is not None else None
        if mode_key in ("global", "mean", "cosine"):
            score_result = render_score_feature_candidates(
                query_feat[batch_idx],
                candidate_feat[batch_idx],
                mask=mask_i,
                preprocess=preprocess,
                highpass_kernel=highpass_kernel,
            )
        elif mode_key in ("local", "local_corr", "correlation"):
            score_result = local_render_score_feature_candidates(
                query_feat[batch_idx],
                candidate_feat[batch_idx],
                mask=mask_i,
                radius=radius,
                preprocess=preprocess,
                highpass_kernel=highpass_kernel,
                score_map_mode=score_map_mode,
            )
        else:
            raise ValueError(f"Unknown candidate render score mode {mode!r}")
        score_rows.append(score_result["scores"])
        valid_fraction_rows.append(score_result.get("valid_fraction", torch.ones_like(score_result["scores"])))
        if return_score_maps:
            score_map_rows.append(score_result["score_map"])
        score_stats = score_result.get("score_stats", {})
        for key in stats_rows:
            stats_rows[key].append(score_stats.get(key, torch.zeros_like(score_result["scores"])))
    stacked_stats = {key: torch.stack(rows, dim=0) for key, rows in stats_rows.items()}
    stacked_score_maps = torch.stack(score_map_rows, dim=0) if return_score_maps else None
    return torch.stack(score_rows, dim=0), torch.stack(valid_fraction_rows, dim=0), stacked_stats, stacked_score_maps


def candidate_score_fusion_listwise_loss(
    query_feat,
    candidate_feat,
    candidate_pose,
    pose_gt,
    scorer,
    *,
    batch=None,
    mask=None,
    candidate_valid_mask=None,
    mode="local",
    temperature=0.07,
    radius=4,
    preprocess="none",
    highpass_kernel=5,
    score_map_mode="peak_offset",
    rot_cost_weight=0.1,
    target_mode="hard",
    target_temperature_m=0.25,
    render_feature_mode="basic",
    wls_radius=None,
    wls_temperature=None,
    wls_damping=1e-3,
    wls_update_scale=1.0,
    wls_conf_mode="max",
    wls_conf_variance_scale=0.5,
    wls_downsample=1,
    cost_regression_weight=0.0,
    cost_regression_temperature_m=None,
    pairwise_rank_weight=0.0,
    pairwise_rank_temperature=1.0,
    pairwise_rank_min_gap_m=0.0,
    basin_trans_m=0.25,
    basin_rot_deg=5.0,
    return_details=False,
):
    """Train a learnable scorer to rank pose candidates using render scores plus priors."""
    if scorer is None:
        raise ValueError("candidate_score_fusion_listwise_loss requires a scorer module")
    if query_feat.ndim != 4:
        raise ValueError("query_feat must have shape (B,C,H,W)")
    if candidate_feat.ndim != 5:
        raise ValueError("candidate_feat must have shape (B,K,C,H,W)")
    if candidate_pose.ndim != 4 or candidate_pose.shape[-2:] != (4, 4):
        raise ValueError("candidate_pose must have shape (B,K,4,4)")
    if pose_gt.ndim != 3 or pose_gt.shape[-2:] != (4, 4):
        raise ValueError("pose_gt must have shape (B,4,4)")
    if candidate_feat.shape[:2] != candidate_pose.shape[:2]:
        raise ValueError("candidate_feat and candidate_pose must have matching B,K dimensions")
    if query_feat.shape[0] != candidate_feat.shape[0] or pose_gt.shape[0] != query_feat.shape[0]:
        raise ValueError("query_feat, candidate_feat, and pose_gt batch sizes must match")

    bsz, num_candidates = candidate_feat.shape[:2]
    score_map_scorer = bool(getattr(scorer, "expects_score_map", False))
    render_scores, render_valid_fraction, render_stats, score_maps = _candidate_render_score_rows(
        query_feat,
        candidate_feat,
        mask=mask,
        mode=mode,
        radius=radius,
        preprocess=preprocess,
        highpass_kernel=highpass_kernel,
        score_map_mode=score_map_mode,
        return_score_maps=score_map_scorer,
    )
    valid = torch.ones((bsz, num_candidates), device=render_scores.device, dtype=torch.bool)
    if candidate_valid_mask is not None:
        valid = candidate_valid_mask.to(device=render_scores.device).bool()
        if valid.shape != (bsz, num_candidates):
            raise ValueError(
                f"candidate_valid_mask must have shape {(bsz, num_candidates)}, got {tuple(valid.shape)}"
            )
    target_idx, trans_err_m, rot_err_deg, pose_cost = _candidate_pose_target(
        candidate_pose.to(device=render_scores.device),
        pose_gt.to(device=render_scores.device),
        valid,
        rot_cost_weight,
    )
    render_scores = torch.where(valid, render_scores, torch.zeros_like(render_scores))
    render_mean = (render_scores * valid.float()).sum(dim=1, keepdim=True) / valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    render_features = torch.stack(
        [
            render_scores,
            render_scores - render_mean,
            torch.where(valid, render_valid_fraction, torch.zeros_like(render_valid_fraction)),
        ],
        dim=-1,
    )
    render_feature_mode_key = str(render_feature_mode or "basic").lower()
    if render_feature_mode_key in ("rich", "stats", "spatial_stats"):
        render_features = torch.stack(
            [
                render_scores,
                render_scores - render_mean,
                torch.where(valid, render_valid_fraction, torch.zeros_like(render_valid_fraction)),
                torch.where(valid, render_stats["std"], torch.zeros_like(render_scores)),
                torch.where(valid, render_stats["max"], torch.zeros_like(render_scores)),
                torch.where(valid, render_stats["max"] - render_mean, torch.zeros_like(render_scores)),
                torch.where(valid, render_stats["topk_mean"], torch.zeros_like(render_scores)),
                torch.where(valid, render_stats["topk_mean"] - render_mean, torch.zeros_like(render_scores)),
                torch.where(valid, render_stats["peakiness"], torch.zeros_like(render_scores)),
            ],
            dim=-1,
        )
    elif render_feature_mode_key not in ("basic", "base", "scalar"):
        raise ValueError("candidate score fusion render_feature_mode must be 'basic' or 'rich'")
    quality_features, _quality_names = candidate_quality_features_from_batch(batch or {}, valid_mask=valid)
    fusion_features = torch.cat(
        [render_features.to(dtype=quality_features.dtype), quality_features.to(device=render_scores.device)],
        dim=-1,
    )
    if score_map_scorer:
        if score_maps is None:
            raise ValueError("score_map scorer requires candidate score maps")
        raw_logits = scorer(score_maps.to(device=render_scores.device), fusion_features).to(
            device=render_scores.device,
            dtype=render_scores.dtype,
        )
    else:
        raw_logits = scorer(fusion_features).to(device=render_scores.device, dtype=render_scores.dtype)
    logits = raw_logits.masked_fill(~valid, -1.0e6) / max(float(temperature), 1e-6)
    target_mode_key = str(target_mode or "hard").lower()
    soft_target_entropy = torch.zeros((), device=render_scores.device, dtype=render_scores.dtype)
    refined_target = None
    if target_mode_key in ("wls", "wls_hard", "wls_soft", "wls_pose_soft", "refined_wls", "refined_wls_soft"):
        candidate_depth = (batch or {}).get("rendered_map_candidate_depth")
        candidate_intrinsics = (batch or {}).get("rendered_map_candidate_intrinsics")
        if candidate_depth is None or candidate_intrinsics is None:
            raise ValueError("WLS candidate targets require rendered_map_candidate_depth and rendered_map_candidate_intrinsics")
        refined_target = _candidate_wls_refined_pose_cost(
            query_feat,
            candidate_feat,
            candidate_pose.to(device=render_scores.device),
            pose_gt.to(device=render_scores.device),
            candidate_depth.to(device=render_scores.device),
            candidate_intrinsics.to(device=render_scores.device),
            valid,
            radius=int(radius if wls_radius is None else wls_radius),
            temperature=float(temperature if wls_temperature is None else wls_temperature),
            damping=float(wls_damping),
            update_scale=float(wls_update_scale),
            rot_cost_weight=float(rot_cost_weight),
            wls_conf_mode=wls_conf_mode,
            wls_conf_variance_scale=float(wls_conf_variance_scale),
            wls_downsample=int(wls_downsample or 1),
        )
        pose_cost = refined_target["cost"].detach()
        target_idx = pose_cost.masked_fill(~valid, float("inf")).argmin(dim=1)
    if target_mode_key in ("hard", "argmin", "ce", "gt_pose_error_hard", "pose_error_hard"):
        loss = F.cross_entropy(logits, target_idx)
    elif target_mode_key in ("wls", "wls_hard", "refined_wls"):
        loss = F.cross_entropy(logits, target_idx)
    elif target_mode_key in (
        "soft",
        "soft_pose",
        "pose_softmax",
        "gt_pose_error_soft",
        "pose_error_soft",
        "wls_soft",
        "wls_pose_soft",
        "refined_wls_soft",
    ):
        target_logits = (-pose_cost / max(float(target_temperature_m), 1e-6)).masked_fill(~valid, -1.0e6)
        target_probs = F.softmax(target_logits, dim=1).detach()
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(target_probs * log_probs).sum(dim=1).mean()
        soft_target_entropy = -(target_probs * torch.log(target_probs.clamp(min=1e-8))).sum(dim=1).mean()
    else:
        raise ValueError("candidate score fusion target_mode must be 'hard', 'soft', 'wls_hard', or 'wls_soft'")
    cost_regression_loss = torch.zeros((), device=render_scores.device, dtype=render_scores.dtype)
    if float(cost_regression_weight) > 0.0:
        cost_temp = max(
            float(target_temperature_m if cost_regression_temperature_m is None else cost_regression_temperature_m),
            1e-6,
        )
        target_values = (-pose_cost / cost_temp).masked_fill(~valid, 0.0)
        target_values = _masked_row_standardize(target_values, valid).detach()
        pred_values = _masked_row_standardize(raw_logits.to(dtype=target_values.dtype), valid)
        reg_per_candidate = F.smooth_l1_loss(pred_values, target_values, reduction="none")
        cost_regression_loss = (reg_per_candidate * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        loss = loss + float(cost_regression_weight) * cost_regression_loss
    pairwise_rank_loss = torch.zeros((), device=render_scores.device, dtype=render_scores.dtype)
    pairwise_rank_acc = torch.zeros((), device=render_scores.device, dtype=render_scores.dtype)
    if float(pairwise_rank_weight) > 0.0 and num_candidates > 1:
        cost_gap = pose_cost[:, None, :] - pose_cost[:, :, None]
        pair_valid = valid[:, :, None] & valid[:, None, :]
        pair_valid = pair_valid & (cost_gap > float(pairwise_rank_min_gap_m))
        if pair_valid.any():
            logit_margin = raw_logits[:, :, None] - raw_logits[:, None, :]
            rank_temp = max(float(pairwise_rank_temperature), 1e-6)
            pair_losses = F.softplus(-logit_margin / rank_temp)
            pairwise_rank_loss = pair_losses[pair_valid].mean()
            pairwise_rank_acc = (logit_margin[pair_valid] > 0.0).float().mean()
            loss = loss + float(pairwise_rank_weight) * pairwise_rank_loss
    pred_idx = logits.argmax(dim=1)
    basin_mask = valid & (trans_err_m <= float(basin_trans_m)) & (rot_err_deg <= float(basin_rot_deg))
    oracle_basin = basin_mask.any(dim=1).float().mean().detach()
    target_basin = basin_mask[torch.arange(bsz, device=render_scores.device), target_idx].float().mean().detach()
    topk_basin = {}
    for topk in (1, 4, 8):
        k_eff = min(int(topk), num_candidates)
        topk_idx = logits.topk(k_eff, dim=1).indices
        topk_basin[topk] = basin_mask.gather(1, topk_idx).any(dim=1).float().mean().detach()
    render_logits = render_scores.masked_fill(~valid, -1.0e6)
    render_pred_idx = render_logits.argmax(dim=1)
    batch_idx = torch.arange(bsz, device=render_scores.device)
    target_scores = raw_logits[batch_idx, target_idx]
    non_target_scores = raw_logits.masked_fill(~valid, -1.0e6)
    non_target_scores = non_target_scores.scatter(1, target_idx[:, None], -1.0e6)
    if num_candidates > 1:
        best_non_target = non_target_scores.max(dim=1).values
    else:
        best_non_target = target_scores.new_zeros(target_scores.shape)
    metrics = {
        "map_candidate_score_fusion_loss": loss.detach(),
        "map_candidate_score_fusion_acc": (pred_idx == target_idx).float().mean().detach(),
        "map_candidate_score_fusion_render_acc": (render_pred_idx == target_idx).float().mean().detach(),
        "map_candidate_score_fusion_margin": (target_scores - best_non_target).mean().detach(),
        "map_candidate_score_fusion_target_idx": target_idx.float().mean().detach(),
        "map_candidate_score_fusion_pred_idx": pred_idx.float().mean().detach(),
        "map_candidate_score_fusion_pred_trans_mm": (trans_err_m[batch_idx, pred_idx] * 1000.0).mean().detach(),
        "map_candidate_score_fusion_pred_rot_deg": rot_err_deg[batch_idx, pred_idx].mean().detach(),
        "map_candidate_score_fusion_top1_basin_recall": topk_basin[1],
        "map_candidate_score_fusion_top4_basin_recall": topk_basin[4],
        "map_candidate_score_fusion_top8_basin_recall": topk_basin[8],
        "map_candidate_score_fusion_oracle_basin_recall": oracle_basin,
        "map_candidate_score_fusion_target_basin_recall": target_basin,
        "map_candidate_score_fusion_target_trans_mm": (trans_err_m[batch_idx, target_idx] * 1000.0).mean().detach(),
        "map_candidate_score_fusion_target_rot_deg": rot_err_deg[batch_idx, target_idx].mean().detach(),
        "map_candidate_score_fusion_num_candidates": torch.tensor(float(num_candidates), device=render_scores.device),
        "map_candidate_score_fusion_soft_target_entropy": soft_target_entropy.detach(),
        "map_candidate_score_fusion_cost_regression_loss": cost_regression_loss.detach(),
        "map_candidate_score_fusion_pairwise_rank_loss": pairwise_rank_loss.detach(),
        "map_candidate_score_fusion_pairwise_rank_acc": pairwise_rank_acc.detach(),
    }
    if refined_target is not None:
        refined_trans = refined_target["trans_err_m"].detach()
        refined_rot = refined_target["rot_err_deg"].detach()
        refined_init_trans = refined_target["init_trans_err_m"].detach()
        refined_conf = refined_target["conf_mean"].detach()
        metrics.update(
            {
                "map_candidate_score_fusion_wls_target_trans_mm": (
                    refined_trans[batch_idx, target_idx] * 1000.0
                ).mean().detach(),
                "map_candidate_score_fusion_wls_pred_trans_mm": (
                    refined_trans[batch_idx, pred_idx] * 1000.0
                ).mean().detach(),
                "map_candidate_score_fusion_wls_target_rot_deg": refined_rot[batch_idx, target_idx].mean().detach(),
                "map_candidate_score_fusion_wls_pred_rot_deg": refined_rot[batch_idx, pred_idx].mean().detach(),
                "map_candidate_score_fusion_wls_init_trans_mm": (
                    refined_init_trans[batch_idx, target_idx] * 1000.0
                ).mean().detach(),
                "map_candidate_score_fusion_wls_conf_mean": refined_conf[valid].mean().detach(),
            }
        )
    if return_details:
        details = {
            "raw_logits": raw_logits,
            "logits": logits,
            "valid": valid,
            "target_idx": target_idx,
            "pose_cost": pose_cost,
            "trans_err_m": trans_err_m,
            "rot_err_deg": rot_err_deg,
        }
        return loss, metrics, details
    return loss, metrics


def render_score_candidate_listwise_loss(
    query_feat,
    candidate_feat,
    candidate_pose,
    pose_gt,
    mask=None,
    *,
    candidate_valid_mask=None,
    mode="local",
    temperature=0.07,
    radius=4,
    preprocess="none",
    highpass_kernel=5,
    rot_cost_weight=0.1,
):
    """Train query/map features to score the nearest rendered pose candidate highest."""
    if query_feat.ndim != 4:
        raise ValueError("query_feat must have shape (B,C,H,W)")
    if candidate_feat.ndim != 5:
        raise ValueError("candidate_feat must have shape (B,K,C,H,W)")
    if candidate_pose.ndim != 4 or candidate_pose.shape[-2:] != (4, 4):
        raise ValueError("candidate_pose must have shape (B,K,4,4)")
    if pose_gt.ndim != 3 or pose_gt.shape[-2:] != (4, 4):
        raise ValueError("pose_gt must have shape (B,4,4)")
    if candidate_feat.shape[:2] != candidate_pose.shape[:2]:
        raise ValueError("candidate_feat and candidate_pose must have matching B,K dimensions")
    if query_feat.shape[0] != candidate_feat.shape[0] or pose_gt.shape[0] != query_feat.shape[0]:
        raise ValueError("query_feat, candidate_feat, and pose_gt batch sizes must match")

    bsz, num_candidates = candidate_feat.shape[:2]
    if num_candidates <= 0:
        raise ValueError("candidate_feat must contain at least one candidate")
    score_rows = []
    mode_key = str(mode or "local").lower()
    for batch_idx in range(bsz):
        mask_i = mask[batch_idx] if mask is not None else None
        if mode_key in ("global", "mean", "cosine"):
            score_result = render_score_feature_candidates(
                query_feat[batch_idx],
                candidate_feat[batch_idx],
                mask=mask_i,
                preprocess=preprocess,
                highpass_kernel=highpass_kernel,
            )
        elif mode_key in ("local", "local_corr", "correlation"):
            score_result = local_render_score_feature_candidates(
                query_feat[batch_idx],
                candidate_feat[batch_idx],
                mask=mask_i,
                radius=radius,
                preprocess=preprocess,
                highpass_kernel=highpass_kernel,
            )
        else:
            raise ValueError(f"Unknown candidate render score mode {mode!r}")
        score_rows.append(score_result["scores"])
    scores = torch.stack(score_rows, dim=0)

    cand_pose_f = candidate_pose.to(device=scores.device, dtype=scores.dtype)
    pose_gt_f = pose_gt.to(device=scores.device, dtype=scores.dtype)
    cand_centers = camera_centers_from_w2c(cand_pose_f.reshape(bsz * num_candidates, 4, 4)).reshape(
        bsz,
        num_candidates,
        3,
    )
    gt_centers = camera_centers_from_w2c(pose_gt_f).unsqueeze(1)
    trans_err_m = torch.linalg.norm(cand_centers - gt_centers, dim=-1)
    _rot_loss, rot_err_deg = _rotation_error_from_mats(cand_pose_f[:, :, :3, :3], pose_gt_f[:, None, :3, :3])
    cost = trans_err_m + float(rot_cost_weight) * (rot_err_deg / 180.0)

    valid = torch.ones((bsz, num_candidates), device=scores.device, dtype=torch.bool)
    if candidate_valid_mask is not None:
        valid = candidate_valid_mask.to(device=scores.device).bool()
        if valid.shape != (bsz, num_candidates):
            raise ValueError(
                f"candidate_valid_mask must have shape {(bsz, num_candidates)}, got {tuple(valid.shape)}"
            )
    cost_for_target = cost.masked_fill(~valid, float("inf"))
    target_idx = cost_for_target.argmin(dim=1)
    logits = scores.masked_fill(~valid, -1.0e6) / max(float(temperature), 1e-6)
    loss = F.cross_entropy(logits, target_idx)
    pred_idx = logits.argmax(dim=1)
    batch_idx = torch.arange(bsz, device=scores.device)
    target_scores = scores[batch_idx, target_idx]
    non_target_scores = scores.masked_fill(~valid, -1.0e6)
    non_target_scores = non_target_scores.scatter(1, target_idx[:, None], -1.0e6)
    if num_candidates > 1:
        best_non_target = non_target_scores.max(dim=1).values
    else:
        best_non_target = target_scores.new_zeros(target_scores.shape)
    metrics = {
        "map_candidate_render_score_loss": loss.detach(),
        "map_candidate_render_score_acc": (pred_idx == target_idx).float().mean().detach(),
        "map_candidate_render_score_margin": (target_scores - best_non_target).mean().detach(),
        "map_candidate_render_score_target_idx": target_idx.float().mean().detach(),
        "map_candidate_render_score_pred_idx": pred_idx.float().mean().detach(),
        "map_candidate_render_score_target_trans_mm": (trans_err_m[batch_idx, target_idx] * 1000.0).mean().detach(),
        "map_candidate_render_score_target_rot_deg": rot_err_deg[batch_idx, target_idx].mean().detach(),
        "map_candidate_render_score_pred_trans_mm": (trans_err_m[batch_idx, pred_idx] * 1000.0).mean().detach(),
        "map_candidate_render_score_pred_rot_deg": rot_err_deg[batch_idx, pred_idx].mean().detach(),
        "map_candidate_render_score_num_candidates": torch.tensor(float(num_candidates), device=scores.device),
    }
    return loss, metrics


def pose_update_gain_loss(
    pose_pred: torch.Tensor,
    pose_ref: torch.Tensor,
    pose_gt: torch.Tensor,
    *,
    trans_margin_m: float = 0.0,
    rot_margin_deg: float = 0.0,
    rot_weight: float = 0.0,
    trans_weight: float = 1.0,
):
    """Penalize pose updates that fail to improve over the initial pose by a margin."""
    _, rot_err_deg, trans_err_m = pose_error_tensors(pose_pred.float(), pose_gt.float())
    _, init_rot_err_deg, init_trans_err_m = pose_error_tensors(pose_ref.float(), pose_gt.float())
    trans_margin = max(float(trans_margin_m), 0.0)
    rot_margin = max(float(rot_margin_deg), 0.0)
    trans_loss_m = F.relu(trans_err_m + trans_margin - init_trans_err_m)
    rot_loss_deg = F.relu(rot_err_deg + rot_margin - init_rot_err_deg)
    loss = float(trans_weight) * trans_loss_m.mean() + float(rot_weight) * rot_loss_deg.mean()
    with torch.no_grad():
        trans_improved = (trans_err_m < init_trans_err_m).float().mean()
        trans_margin_met = (trans_err_m + trans_margin <= init_trans_err_m).float().mean()
        rot_margin_met = (rot_err_deg + rot_margin <= init_rot_err_deg).float().mean()
    return loss, {
        "map_corr_pose_gain_loss": loss.detach(),
        "map_corr_pose_gain_trans_loss_mm": (trans_loss_m.detach().mean() * 1000.0),
        "map_corr_pose_gain_rot_loss_deg": rot_loss_deg.detach().mean(),
        "map_corr_pose_gain_trans_margin_mm": torch.tensor(
            trans_margin * 1000.0,
            device=pose_pred.device,
            dtype=pose_pred.dtype,
        ),
        "map_corr_pose_gain_rot_margin_deg": torch.tensor(
            rot_margin,
            device=pose_pred.device,
            dtype=pose_pred.dtype,
        ),
        "map_corr_pose_gain_trans_improved": trans_improved.detach(),
        "map_corr_pose_gain_trans_margin_met": trans_margin_met.detach(),
        "map_corr_pose_gain_rot_margin_met": rot_margin_met.detach(),
    }


def compute_w2c_flow(
    pose_init: torch.Tensor,
    pose_gt: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: dict,
    target_hw=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project depth from pose_init into pose_gt and return rendered->query flow."""
    with torch.cuda.amp.autocast(enabled=False):
        init = pose_init.float()
        gt = pose_gt.float()
        if init.ndim == 2:
            init = init.unsqueeze(0)
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        depth_f = depth.float()
        if depth_f.ndim == 4:
            depth_f = depth_f.squeeze(1)
        if depth_f.ndim != 3:
            raise ValueError(f"depth must have shape (B,H,W) or (B,1,H,W), got {tuple(depth.shape)}")
        B, H, W = depth_f.shape
        if init.shape[0] != B:
            init = init.expand(B, -1, -1)
        if gt.shape[0] != B:
            gt = gt.expand(B, -1, -1)

        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        device = depth_f.device
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)
        v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)

        X = (u_coords - cx) / fx * depth_f
        Y = (v_coords - cy) / fy * depth_f
        Z = depth_f
        pts = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=-1)
        pts_flat = pts.reshape(B, -1, 4).permute(0, 2, 1)

        T_rel = torch.bmm(gt, torch.linalg.inv(init))
        pts_gt = torch.bmm(T_rel[:, :3, :], pts_flat).reshape(B, 3, H, W)
        Z_gt_raw = pts_gt[:, 2:3]
        Z_gt = Z_gt_raw.clamp(min=0.01)
        u_gt = fx * pts_gt[:, 0:1] / Z_gt + cx
        v_gt = fy * pts_gt[:, 1:2] / Z_gt + cy

        flow = torch.cat(
            [
                u_gt - u_coords.unsqueeze(1),
                v_gt - v_coords.unsqueeze(1),
            ],
            dim=1,
        )
        valid = (
            (depth_f.unsqueeze(1) > 0.05)
            & (Z_gt_raw > 0.1)
            & (u_gt > -0.5)
            & (u_gt < W - 0.5)
            & (v_gt > -0.5)
            & (v_gt < H - 0.5)
        ).float()
        flow = flow * valid

        if target_hw is not None:
            tH, tW = int(target_hw[0]), int(target_hw[1])
            if (tH, tW) != (H, W):
                sx = tW / max(W, 1)
                sy = tH / max(H, 1)
                flow = F.interpolate(flow, (tH, tW), mode="bilinear", align_corners=False)
                flow[:, 0] *= sx
                flow[:, 1] *= sy
                valid = F.interpolate(valid, (tH, tW), mode="nearest")
        return flow, valid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_safe_num_workers(training_cfg: dict, dataset_cfg: dict) -> int:
    """Avoid DataLoader shared-memory crashes for full-resolution RGB tensors."""
    workers = int(training_cfg.get("num_workers", 0))
    if workers <= 0 or bool(training_cfg.get("allow_highres_num_workers", False)):
        return workers
    input_hw = dataset_cfg.get("input_hw") or []
    if len(input_hw) >= 2:
        pixels = int(input_hw[0]) * int(input_hw[1])
        threshold = int(training_cfg.get("highres_worker_pixel_threshold", 1024 * 1024))
        if pixels >= threshold:
            return 0
    return workers


def setup_logger(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("joint_radio_dcff")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", mode="a")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_cambridge_split_ordered(split_path):
    ordered = []
    names = set()
    split_path = Path(split_path)
    if not split_path.is_file():
        return ordered, names
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if (
                not line
                or line.startswith("#")
                or line.startswith("Visual")
                or line.startswith("ImageFile")
            ):
                continue
            image_name = line.split()[0].replace("\\", "/")
            stem = str(Path(image_name).with_suffix(""))
            ordered.append(image_name)
            names.add(image_name)
            names.add(stem + ".png")
            names.add(stem + ".jpg")
    return ordered, names


def parse_cambridge_split(split_path):
    _, names = parse_cambridge_split_ordered(split_path)
    return names


def discover_images(source_dir, patterns):
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        return []
    for pattern in patterns:
        found = sorted(source_dir.glob(pattern))
        if found:
            return found
    return []


def _normalize_record_name(name):
    return Path(str(name).replace("\\", "/")).as_posix()


def _candidate_record_names(name):
    norm = _normalize_record_name(name)
    candidates = [norm]
    if norm.startswith("images/"):
        candidates.append(norm[len("images/"):])
    candidates.append(Path(norm).name)
    stem = str(Path(norm).with_suffix(""))
    candidates.append(stem)
    seen = set()
    ordered = []
    for item in candidates:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def infer_colmap_dir_from_dataset(dataset_cfg):
    explicit = dataset_cfg.get("colmap_dir")
    if explicit:
        return explicit
    source_dir = Path(dataset_cfg.get("source_dir", ""))
    candidates = [
        source_dir / "sparse" / "0",
        source_dir.parent / "sparse" / "0",
    ]
    for candidate in candidates:
        if (candidate / "images.bin").is_file():
            return str(candidate)
    return None


def load_image_id_to_name(dataset_cfg):
    if dataset_cfg.get("image_id_to_name"):
        return {int(k): _normalize_record_name(v) for k, v in dataset_cfg["image_id_to_name"].items()}
    colmap_dir = infer_colmap_dir_from_dataset(dataset_cfg)
    if not colmap_dir:
        return None
    images_bin = Path(colmap_dir) / "images.bin"
    if not images_bin.is_file():
        return None
    images = read_colmap_images(str(images_bin))
    return {int(image_id): _normalize_record_name(meta.name) for image_id, meta in images.items()}


def load_feature_export_id_to_name(dataset_cfg):
    if dataset_cfg.get("feature_id_to_name"):
        return {int(k): _normalize_record_name(v) for k, v in dataset_cfg["feature_id_to_name"].items()}
    feature_dir = dataset_cfg.get("feature_dir")
    if not feature_dir:
        return None
    export_index_path = Path(feature_dir) / "export_index.json"
    if not export_index_path.is_file():
        return None
    with open(export_index_path, "r", encoding="utf-8") as handle:
        export_index = json.load(handle)
    mapping = {}
    for item in export_index:
        if "teacher_idx" not in item or "sample_name" not in item:
            continue
        mapping[int(item["teacher_idx"])] = _normalize_record_name(item["sample_name"])
    return mapping or None


def load_feature_id_to_name(dataset_cfg):
    export_mapping = load_feature_export_id_to_name(dataset_cfg)
    if export_mapping:
        return export_mapping
    if str(dataset_cfg.get("feature_id_space", "")).lower() == "colmap":
        return load_image_id_to_name(dataset_cfg)
    return None


def build_records_from_feature_ids(dataset_cfg, teacher_indices, image_id_to_name=None):
    images = discover_images(dataset_cfg["source_dir"], dataset_cfg["image_patterns"])
    if not images:
        return []

    source_dir = Path(dataset_cfg["source_dir"])
    name_to_path = {}
    for image_path in images:
        rel_name = _normalize_record_name(image_path.relative_to(source_dir))
        for key in _candidate_record_names(rel_name):
            name_to_path.setdefault(key, (image_path, rel_name))

    records = []
    if image_id_to_name:
        normalized_id_to_name = {
            int(image_id): _normalize_record_name(name)
            for image_id, name in image_id_to_name.items()
        }
        for teacher_idx in teacher_indices:
            image_name = normalized_id_to_name.get(int(teacher_idx))
            if image_name is None:
                continue
            match = None
            for key in _candidate_record_names(image_name):
                if key in name_to_path:
                    match = name_to_path[key]
                    break
            if match is None:
                continue
            image_path, rel_name = match
            records.append(
                {
                    "teacher_idx": int(teacher_idx),
                    "image_path": str(image_path),
                    "sample_name": rel_name,
                    "normalized_name": rel_name.replace("\\", "/"),
                }
            )
        return records

    for teacher_idx in teacher_indices:
        if teacher_idx >= len(images):
            continue
        image_path = images[teacher_idx]
        rel_name = _normalize_record_name(image_path.relative_to(source_dir))
        records.append(
            {
                "teacher_idx": int(teacher_idx),
                "image_path": str(image_path),
                "sample_name": rel_name,
                "normalized_name": rel_name.replace("\\", "/"),
            }
        )
    return records


class TeacherFeatureStore:
    def __init__(self, feature_dir, cache_in_memory=False):
        self.feature_dir = Path(feature_dir)
        self.fine_dir = self.feature_dir / "fine_geo"
        self.coarse_dir = self.feature_dir / "coarse_sem"
        if not self.fine_dir.is_dir() or not self.coarse_dir.is_dir():
            raise FileNotFoundError(
                f"Expected fine_geo/ and coarse_sem/ under {self.feature_dir}"
            )

        self.fine_files = self._discover_files(self.fine_dir, "fine_geo")
        self.coarse_files = self._discover_files(self.coarse_dir, "coarse_sem")
        self.indices = sorted(set(self.fine_files) & set(self.coarse_files))
        if not self.indices:
            raise RuntimeError(f"No paired teacher features found in {self.feature_dir}")

        sample = safe_torch_load(self.fine_files[self.indices[0]]).float()
        coarse_sample = safe_torch_load(self.coarse_files[self.indices[0]]).float()
        self.fine_feature_dim = int(sample.shape[0])
        self.coarse_feature_dim = int(coarse_sample.shape[0])
        self.feature_dim = self.fine_feature_dim
        self.feature_hw = (int(sample.shape[1]), int(sample.shape[2]))
        self.coarse_feature_hw = (int(coarse_sample.shape[1]), int(coarse_sample.shape[2]))
        self.cache_in_memory = cache_in_memory
        self._cache = {}

    @staticmethod
    def _discover_files(root_dir, scale_name):
        mapping = {}
        pattern = re.compile(rf"rgb_(\d+)_{re.escape(scale_name)}_.*\.pt$")
        for path in sorted(root_dir.glob("*.pt")):
            match = pattern.match(path.name)
            if match:
                mapping[int(match.group(1))] = path
        return mapping

    def load_pair(self, index):
        if self.cache_in_memory and index in self._cache:
            fine, coarse = self._cache[index]
        else:
            fine = safe_torch_load(self.fine_files[index]).float()
            coarse = safe_torch_load(self.coarse_files[index]).float()
            if self.cache_in_memory:
                self._cache[index] = (fine, coarse)
        return fine, coarse


def sample_name_to_feature_stem(sample_name):
    return Path(sample_name).with_suffix("").as_posix().replace("/", "_")


def _as_numpy_mapping(payload):
    if isinstance(payload, np.lib.npyio.NpzFile):
        return {key: payload[key] for key in payload.files}
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"Unsupported correspondence payload type: {type(payload)!r}")


def _payload_array(payload, *keys, default=None):
    for key in keys:
        if key in payload:
            value = payload[key]
            if torch.is_tensor(value):
                return value.detach().cpu().numpy()
            return np.asarray(value)
    return default


def _scale_xy_array(xy, src_hw, dst_hw):
    xy = np.asarray(xy, dtype=np.float32).copy()
    if xy.size == 0:
        return xy.reshape(0, 2)
    src_h, src_w = float(src_hw[0]), float(src_hw[1])
    dst_h, dst_w = float(dst_hw[0]), float(dst_hw[1])
    sx = (dst_w - 1.0) / max(src_w - 1.0, 1.0)
    sy = (dst_h - 1.0) / max(src_h - 1.0, 1.0)
    xy[:, 0] *= sx
    xy[:, 1] *= sy
    return xy


class TeacherCorrespondenceStore:
    """Load sparse teacher correspondences for query-map local alignment.

    The preferred schema is one ``.npz`` per query with ``query_xy`` and
    optional ``map_xy``/``confidence``/``query_hw``/``map_hw`` fields.  LoFTR
    coordinates are accepted and scaled to the configured feature grid here,
    so the training loop can consume fixed-size tensors through the default
    DataLoader collate path.
    """

    def __init__(
        self,
        path,
        *,
        feature_hw,
        max_points=512,
        coordinate_space="auto",
    ):
        self.path = Path(path)
        self.feature_hw = tuple(int(v) for v in feature_hw)
        self.max_points = int(max_points)
        self.coordinate_space = str(coordinate_space or "auto").lower()
        self._cache = {}
        self._mapping = {}
        if self.path.is_dir():
            for suffix in ("*.npz", "*.pt", "*.pth", "*.pkl", "*.pickle"):
                for file_path in sorted(self.path.glob(suffix)):
                    self._mapping.setdefault(file_path.stem, file_path)
        elif not self.path.is_file():
            raise FileNotFoundError(f"teacher correspondence path not found: {self.path}")

    def _empty(self):
        n = max(self.max_points, 0)
        return {
            "teacher_corr_query_xy": torch.zeros(n, 2, dtype=torch.float32),
            "teacher_corr_map_xy": torch.zeros(n, 2, dtype=torch.float32),
            "teacher_corr_conf": torch.zeros(n, dtype=torch.float32),
            "teacher_corr_valid": torch.zeros(n, dtype=torch.float32),
            "teacher_corr_hw": torch.tensor(self.feature_hw, dtype=torch.float32),
        }

    def _candidate_keys(self, sample_name):
        keys = []
        for name in _candidate_record_names(sample_name):
            keys.append(sample_name_to_feature_stem(name))
            keys.append(Path(name).stem)
            keys.append(name.replace("/", "_"))
        seen = set()
        ordered = []
        for key in keys:
            if key and key not in seen:
                ordered.append(key)
                seen.add(key)
        return ordered

    def _load_file(self, file_path):
        file_path = Path(file_path)
        if file_path in self._cache:
            return self._cache[file_path]
        if file_path.suffix == ".npz":
            with np.load(file_path, allow_pickle=True) as data:
                payload = {key: data[key] for key in data.files}
        elif file_path.suffix in {".pt", ".pth"}:
            payload = safe_torch_load(file_path)
        elif file_path.suffix in {".pkl", ".pickle"}:
            with open(file_path, "rb") as handle:
                payload = pickle.load(handle)
        else:
            raise ValueError(f"Unsupported teacher correspondence file: {file_path}")
        self._cache[file_path] = payload
        return payload

    def _lookup_payload(self, record):
        sample_name = record["sample_name"]
        if self.path.is_dir():
            for key in self._candidate_keys(sample_name):
                file_path = self._mapping.get(key)
                if file_path is not None:
                    return self._load_file(file_path)
            return None

        payload = self._load_file(self.path)
        mapping = _as_numpy_mapping(payload)
        if "query_xy" in mapping or "pts2d_query" in mapping or "keypoints0" in mapping:
            return mapping
        for key in self._candidate_keys(sample_name):
            if key in mapping:
                return mapping[key].item() if isinstance(mapping[key], np.ndarray) and mapping[key].shape == () else mapping[key]
        return None

    def load_record(self, record):
        payload = self._lookup_payload(record)
        if payload is None:
            return self._empty()
        payload = _as_numpy_mapping(payload)
        query_xy = _payload_array(payload, "query_xy", "pts2d_query", "keypoints0")
        if query_xy is None:
            return self._empty()
        query_xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
        map_xy = _payload_array(payload, "map_xy", "render_xy", "projected_xy", default=query_xy)
        map_xy = np.asarray(map_xy, dtype=np.float32).reshape(-1, 2)
        conf = _payload_array(payload, "confidence", "conf", "scores", default=np.ones((len(query_xy),), dtype=np.float32))
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        valid = _payload_array(payload, "valid", "valid_mask", "inlier_mask", default=np.ones((len(query_xy),), dtype=np.float32))
        valid = np.asarray(valid, dtype=np.float32).reshape(-1) > 0

        n = min(len(query_xy), len(map_xy), len(conf), len(valid))
        if n <= 0 or self.max_points <= 0:
            return self._empty()
        query_xy = query_xy[:n]
        map_xy = map_xy[:n]
        conf = conf[:n]
        valid = valid[:n]

        query_hw = _payload_array(payload, "query_hw", "xy_hw", "image_hw", default=None)
        map_hw = _payload_array(payload, "map_hw", "render_hw", "xy_hw", "image_hw", default=None)
        coordinate_space = str(
            _payload_array(payload, "coordinate_space", default=np.asarray(self.coordinate_space)).item()
            if _payload_array(payload, "coordinate_space", default=None) is not None
            else self.coordinate_space
        ).lower()
        if coordinate_space != "feature":
            if query_hw is not None:
                query_xy = _scale_xy_array(query_xy, np.asarray(query_hw).reshape(-1)[:2], self.feature_hw)
            if map_hw is not None:
                map_xy = _scale_xy_array(map_xy, np.asarray(map_hw).reshape(-1)[:2], self.feature_hw)

        finite = (
            np.isfinite(query_xy).all(axis=1)
            & np.isfinite(map_xy).all(axis=1)
            & np.isfinite(conf)
            & valid
        )
        if not finite.any():
            return self._empty()
        query_xy = query_xy[finite]
        map_xy = map_xy[finite]
        conf = conf[finite]

        order = np.argsort(-conf)
        order = order[: self.max_points]
        query_xy = query_xy[order]
        map_xy = map_xy[order]
        conf = conf[order]
        count = len(order)

        item = self._empty()
        item["teacher_corr_query_xy"][:count] = torch.from_numpy(query_xy).float()
        item["teacher_corr_map_xy"][:count] = torch.from_numpy(map_xy).float()
        item["teacher_corr_conf"][:count] = torch.from_numpy(conf).float()
        item["teacher_corr_valid"][:count] = 1.0
        return item


class RetrievalTeacherStore:
    def __init__(self, feature_dir, subdir="cls", cache_in_memory=False):
        root = Path(feature_dir)
        self.root_dir = root / subdir if subdir else root
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Retrieval teacher directory not found: {self.root_dir}")

        descriptor_name = Path(subdir).name if subdir else "cls"
        pattern = re.compile(rf"(.+)_{re.escape(descriptor_name)}_.*\.pt$")
        self.files = {}
        for path in sorted(self.root_dir.glob("*.pt")):
            match = pattern.match(path.name)
            if match:
                self.files[match.group(1)] = path
        if not self.files:
            raise RuntimeError(f"No {descriptor_name} teacher descriptors found in {self.root_dir}")

        sample = safe_torch_load(next(iter(self.files.values()))).float().view(-1)
        self.feature_dim = int(sample.numel())
        self.cache_in_memory = cache_in_memory
        self._cache = {}

    def load(self, sample_name):
        stem = sample_name_to_feature_stem(sample_name)
        if self.cache_in_memory and stem in self._cache:
            return self._cache[stem]

        path = self.files.get(stem)
        if path is None:
            raise KeyError(f"Missing retrieval teacher descriptor for {sample_name} ({stem})")

        descriptor = safe_torch_load(path).float().view(-1)
        if self.cache_in_memory:
            self._cache[stem] = descriptor
        return descriptor

    def load_record(self, record):
        sample_name = record["sample_name"]
        stem = sample_name_to_feature_stem(sample_name)
        path = self.files.get(stem)
        if path is None and "teacher_idx" in record:
            stem = f"rgb_{int(record['teacher_idx'])}"
            path = self.files.get(stem)
        if path is None:
            raise KeyError(f"Missing retrieval teacher descriptor for {sample_name} ({stem})")
        if self.cache_in_memory and stem in self._cache:
            return self._cache[stem]
        descriptor = safe_torch_load(path).float().view(-1)
        if self.cache_in_memory:
            self._cache[stem] = descriptor
        return descriptor


def build_all_records(dataset_cfg, teacher_store, allow_synthetic=False):
    feature_id_to_name = load_feature_id_to_name(dataset_cfg)
    records = build_records_from_feature_ids(dataset_cfg, teacher_store.indices, feature_id_to_name)
    if records:
        return records

    images = discover_images(dataset_cfg["source_dir"], dataset_cfg["image_patterns"])
    if images:
        raise RuntimeError(
            "No teacher feature ids could be matched to RGB images. "
            "Check dataset.colmap_dir/image_id_to_name and feature cache filenames."
        )
    elif allow_synthetic:
        records = []
        for teacher_idx in teacher_store.indices:
            records.append(
                {
                    "teacher_idx": int(teacher_idx),
                    "image_path": None,
                    "sample_name": f"synthetic_{teacher_idx:05d}",
                    "normalized_name": f"synthetic_{teacher_idx:05d}",
                }
            )
    else:
        raise FileNotFoundError(
            "No source RGB images found. Set dataset.synthetic_if_missing=true or use --smoke-test."
        )

    if not records:
        raise RuntimeError("No records could be paired with teacher features.")
    return records


def _records_in_split_order(all_records, split_order, split_names):
    record_by_key = {}
    for record in all_records:
        keys = [record["normalized_name"]]
        keys.extend(_candidate_record_names(record["normalized_name"]))
        for key in keys:
            if key in split_names:
                record_by_key.setdefault(key, record)

    records = []
    used = set()
    for split_name in split_order:
        keys = [split_name]
        keys.extend(_candidate_record_names(split_name))
        for key in keys:
            record = record_by_key.get(key)
            if record is None:
                continue
            record_id = id(record)
            if record_id not in used:
                records.append(record)
                used.add(record_id)
            break

    for record in all_records:
        if id(record) in used:
            continue
        if record["normalized_name"] in split_names:
            records.append(record)
            used.add(id(record))
    return records


def split_records(all_records, dataset_cfg):
    train_order, train_split = parse_cambridge_split_ordered(dataset_cfg.get("train_split"))
    val_order, val_split = parse_cambridge_split_ordered(dataset_cfg.get("val_split"))

    if train_split and val_split:
        train_records = _records_in_split_order(all_records, train_order, train_split)
        val_records = _records_in_split_order(all_records, val_order, val_split)
    else:
        val_ratio = float(dataset_cfg.get("fallback_val_ratio", 0.1))
        split_idx = max(1, int(round(len(all_records) * (1.0 - val_ratio))))
        train_records = all_records[:split_idx]
        val_records = all_records[split_idx:]

    if not val_records:
        val_records = train_records[: max(1, min(8, len(train_records)))]
    if not train_records:
        raise RuntimeError("Training split is empty after pairing images and teacher caches.")

    if dataset_cfg.get("max_train_samples") is not None:
        train_records = train_records[: int(dataset_cfg["max_train_samples"])]
    if dataset_cfg.get("max_val_samples") is not None:
        val_records = val_records[: int(dataset_cfg["max_val_samples"])]

    return train_records, val_records


def load_pose_candidate_cache_index(cache_paths):
    if not cache_paths:
        return {}
    if isinstance(cache_paths, (str, os.PathLike)):
        paths = [cache_paths]
    else:
        paths = list(cache_paths)
    index = {}
    for raw_path in paths:
        if raw_path is None:
            continue
        entries, _stats = load_retrieval_init_entries(str(raw_path))
        for entry in entries:
            keys = _candidate_record_names(entry["query_image_name"])
            keys.extend(_candidate_record_names(entry.get("query_image_stem", "")))
            for key in keys:
                index.setdefault(key, entry)
    return index


class JointRADIOQueryDataset(Dataset):
    def __init__(
        self,
        records,
        teacher_store,
        input_hw,
        feature_hw,
        synthetic_rgb=False,
        retrieval_teacher_store=None,
        teacher_correspondence_store=None,
        prior_mask_path=None,
        prior_mask_channels=None,
        colmap_dir=None,
        pose_candidate_cache_index=None,
        pose_candidate_topk=0,
    ):
        self.records = records
        self.teacher_store = teacher_store
        self.input_hw = tuple(input_hw)
        self.feature_hw = tuple(feature_hw)
        self.synthetic_rgb = synthetic_rgb
        self.retrieval_teacher_store = retrieval_teacher_store
        self.teacher_correspondence_store = teacher_correspondence_store
        self.pose_candidate_cache_index = pose_candidate_cache_index or {}
        self.pose_candidate_topk = max(0, int(pose_candidate_topk or 0))
        self.prior_masks = None
        self.name_to_pose = {}
        self.basename_to_pose = {}
        if colmap_dir:
            images_bin = Path(colmap_dir) / "images.bin"
            if images_bin.is_file():
                for _image_id, image_meta in read_colmap_images(str(images_bin)).items():
                    name = image_meta.name.replace("\\", "/")
                    pose = torch.from_numpy(colmap_to_w2c(image_meta.qvec, image_meta.tvec)).float()
                    self.name_to_pose[name] = pose
                    self.basename_to_pose.setdefault(Path(name).name, pose)
        self.prior_mask_channels = list(prior_mask_channels or [0, 1, 2])
        if prior_mask_path:
            prior_mask_path = Path(prior_mask_path)
            if prior_mask_path.is_file():
                with open(prior_mask_path, "rb") as handle:
                    raw_masks = pickle.load(handle)
                self.prior_masks = {}
                for key, mask_tuple in raw_masks.items():
                    self.prior_masks[key] = tuple(
                        (mask.detach().to("cpu").bool() if torch.is_tensor(mask) else torch.as_tensor(mask).bool())
                        for mask in mask_tuple
                    )
            else:
                raise FileNotFoundError(f"prior_mask_path not found: {prior_mask_path}")

    def __len__(self):
        return len(self.records)

    def _load_rgb(self, record):
        if record["image_path"] is None:
            generator = torch.Generator().manual_seed(record["teacher_idx"])
            return torch.rand(3, self.input_hw[0], self.input_hw[1], generator=generator)

        with Image.open(record["image_path"]) as img:
            img = img.convert("RGB")
            if tuple(reversed(self.input_hw)) != img.size:
                img = img.resize((self.input_hw[1], self.input_hw[0]), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _load_prior_mask(self, record):
        if self.prior_masks is None:
            return None
        sample_name = record["sample_name"].replace("\\", "/")
        mask_tuple = self.prior_masks.get(sample_name)
        if mask_tuple is None:
            mask_tuple = self.prior_masks.get(Path(sample_name).name)
        if mask_tuple is None:
            return torch.ones(1, *self.feature_hw, dtype=torch.float32)

        valid = None
        for channel_idx in self.prior_mask_channels:
            if channel_idx < 0 or channel_idx >= len(mask_tuple):
                continue
            channel = mask_tuple[channel_idx].bool()
            valid = channel if valid is None else (valid & channel)
        if valid is None:
            return torch.ones(1, *self.feature_hw, dtype=torch.float32)
        mask = valid.float().unsqueeze(0).unsqueeze(0)
        if tuple(mask.shape[-2:]) != self.feature_hw:
            mask = F.interpolate(mask, size=self.feature_hw, mode="nearest")
        return mask.squeeze(0).float()

    def __getitem__(self, idx):
        record = self.records[idx]
        rgb = self._load_rgb(record)
        teacher_fine, teacher_coarse = self.teacher_store.load_pair(record["teacher_idx"])
        item = {
            "rgb": rgb,
            "teacher_fine": teacher_fine,
            "teacher_coarse": teacher_coarse,
            "teacher_idx": record["teacher_idx"],
            "sample_name": record["sample_name"],
        }
        sample_name = record["sample_name"].replace("\\", "/")
        pose_gt = self.name_to_pose.get(sample_name)
        if pose_gt is None:
            pose_gt = self.basename_to_pose.get(Path(sample_name).name)
        if pose_gt is not None:
            item["pose_gt"] = pose_gt.clone()
        prior_mask = self._load_prior_mask(record)
        if prior_mask is not None:
            item["prior_mask"] = prior_mask
        if self.retrieval_teacher_store is not None:
            item["teacher_retrieval"] = self.retrieval_teacher_store.load_record(record)
        if self.teacher_correspondence_store is not None:
            item.update(self.teacher_correspondence_store.load_record(record))
        if self.pose_candidate_cache_index:
            candidate_entry = None
            for key in _candidate_record_names(record["sample_name"]):
                candidate_entry = self.pose_candidate_cache_index.get(key)
                if candidate_entry is not None:
                    break
            if candidate_entry is not None:
                poses = np.asarray(candidate_entry["pose_init_candidates"], dtype=np.float32)
                valid = np.asarray(candidate_entry["candidate_valid_mask"], dtype=bool)
                scores = np.asarray(candidate_entry["retrieval_scores_candidates"], dtype=np.float32)
                limit = self.pose_candidate_topk if self.pose_candidate_topk > 0 else len(poses)
                limit = max(1, min(int(limit), len(poses)))
                item["pose_init_candidates"] = torch.from_numpy(poses[:limit].copy()).float()
                item["candidate_valid_mask"] = torch.from_numpy(valid[:limit].copy()).bool()
                item["retrieval_scores_candidates"] = torch.from_numpy(scores[:limit].copy()).float()
                for key in OPTIONAL_RETRIEVAL_CANDIDATE_FLOAT_KEYS:
                    if key in candidate_entry:
                        values = np.asarray(candidate_entry[key], dtype=np.float32)
                        item[key] = torch.from_numpy(values[:limit].copy()).float()
        return item


class MapFeatureRenderer(nn.Module):
    def __init__(self, cfg, feature_hw, device, logger):
        super().__init__()
        map_cfg = cfg.get("map_supervision", {})
        config_path = map_cfg.get("config_path")
        if not config_path:
            raise ValueError("map_supervision.config_path is required when map supervision is enabled")
        with open(config_path, "r", encoding="utf-8") as f:
            render_cfg = yaml.safe_load(f) or {}
        coarse_smoothing_kernel = map_cfg.get("coarse_smoothing_kernel")
        if coarse_smoothing_kernel is not None:
            render_cfg = copy.deepcopy(render_cfg)
            render_cfg.setdefault("dcff", {})["coarse_smoothing_kernel"] = int(coarse_smoothing_kernel)
        fine_decoder_override = map_cfg.get("fine_decoder_override")
        if fine_decoder_override is not None:
            render_cfg = copy.deepcopy(render_cfg)
            render_cfg.setdefault("dcff", {})["fine_decoder_override"] = copy.deepcopy(fine_decoder_override)
        refiner_override = map_cfg.get("refiner_override")
        if refiner_override is not None:
            render_cfg = copy.deepcopy(render_cfg)
            render_cfg.setdefault("dcff", {})["refiner_override"] = copy.deepcopy(refiner_override)

        self.device = device
        self.logger = logger
        self.feature_hw = tuple(feature_hw)
        self.render_width = int(self.feature_hw[1])
        self.render_height = int(self.feature_hw[0])
        self.cache_in_memory = bool(map_cfg.get("cache_rendered", True))
        self.cache_dtype = torch.float16
        self._cache = {}
        self.candidate_render_batch_size = max(0, int(map_cfg.get("candidate_render_batch_size", 0) or 0))
        self.perturb_render_negatives = bool(map_cfg.get("perturb_render_negatives", False))
        self.perturb_rot_deg = float(map_cfg.get("perturb_rot_deg", 0.0))
        rot_choices = map_cfg.get("perturb_rot_deg_choices") or []
        self.perturb_rot_deg_choices = [float(v) for v in rot_choices if float(v) > 0.0]
        self.perturb_trans_m = float(map_cfg.get("perturb_trans_m", 0.0))
        cm_choices = map_cfg.get("perturb_trans_cm_choices") or []
        self.perturb_trans_cm_choices = [float(v) for v in cm_choices if float(v) > 0.0]
        self.perturb_pose_mode = str(map_cfg.get("perturb_pose_mode", "camera_center") or "camera_center").lower()
        self.perturb_frame = str(map_cfg.get("perturb_frame", "camera")).lower()
        self.perturb_axes = [int(v) for v in (map_cfg.get("perturb_axes") or [0, 1, 2]) if int(v) in (0, 1, 2)]
        if not self.perturb_axes:
            self.perturb_axes = [0, 1, 2]
        self.global_render_negative_count = max(0, int(map_cfg.get("global_render_negative_count", 0)))
        self.global_render_negative_min_trans_m = max(
            0.0,
            float(map_cfg.get("global_render_negative_min_trans_m", 0.0)),
        )
        self.alpha_threshold = float(map_cfg.get("alpha_threshold", 0.5))
        self.trainable = bool(map_cfg.get("trainable", False))
        self.train_fine_decoder = self.trainable and bool(map_cfg.get("train_fine_decoder", False))
        self.train_coarse_fusion = self.trainable and bool(
            map_cfg.get("train_coarse_fusion", self.train_fine_decoder)
        )
        self.train_feat_sharp = self.trainable and bool(map_cfg.get("train_feat_sharp", False))
        self.train_fsm = self.trainable and bool(map_cfg.get("train_fsm", False))
        self.train_hash_mlp = self.trainable and bool(map_cfg.get("train_hash_mlp", False))
        self.train_latent = self.trainable and bool(map_cfg.get("train_latent", False))
        self.train_geometry = self.trainable and bool(map_cfg.get("train_geometry", False))
        self.train_color = self.trainable and bool(map_cfg.get("train_color", False))
        self.map_lr_scale = float(map_cfg.get("map_lr_scale", 0.1))
        self.hash_mlp_lr_scale = float(map_cfg.get("hash_mlp_lr_scale", 0.05))
        self.latent_lr_scale = float(map_cfg.get("latent_lr_scale", self.map_lr_scale))
        self.geometry_lr_scale = float(map_cfg.get("geometry_lr_scale", self.map_lr_scale * 0.25))
        self.position_lr_scale = map_cfg.get("position_lr_scale", None)
        self.opacity_lr_scale = map_cfg.get("opacity_lr_scale", None)
        self.scaling_lr_scale = map_cfg.get("scaling_lr_scale", None)
        self.rotation_lr_scale = map_cfg.get("rotation_lr_scale", None)
        self.color_lr_scale = map_cfg.get("color_lr_scale", None)

        runtime = build_dcff_runtime(render_cfg, device, printer=logger.info)
        self.gaussians = runtime.gaussians
        self.dcff_renderer = runtime.renderer
        self.feat_sharp = runtime.refiner
        self.feat_select = runtime.feat_select
        if bool(map_cfg.get("reset_latent", False)):
            latent_std = float(map_cfg.get("latent_init_std", 0.01))
            with torch.no_grad():
                self.gaussians._latent.normal_(mean=0.0, std=latent_std)
            logger.info("  Reset Gaussian latent to N(0, %.4f^2)", latent_std)
        self.gaussians._latent.requires_grad_(self.train_latent)
        self.gaussians._xyz.requires_grad_(self.train_geometry)
        self.gaussians._rotation.requires_grad_(self.train_geometry)
        self.gaussians._scaling.requires_grad_(self.train_geometry)
        self.gaussians._opacity.requires_grad_(self.train_geometry)
        self.gaussians._features_dc.requires_grad_(self.train_color)
        self.gaussians._features_rest.requires_grad_(self.train_color)
        for p in self.dcff_renderer.fine_decoder.parameters():
            p.requires_grad_(self.train_fine_decoder)
        if getattr(self.dcff_renderer, "coarse_carrier_fusion", None) is not None:
            for p in self.dcff_renderer.coarse_carrier_fusion.parameters():
                p.requires_grad_(self.train_coarse_fusion)
        for p in self.feat_sharp.parameters():
            p.requires_grad_(self.train_feat_sharp)
        if self.feat_select is not None:
            for p in self.feat_select.parameters():
                p.requires_grad_(self.train_fsm)
        for p in self.dcff_renderer.hash_grid.mlp.parameters():
            p.requires_grad_(self.train_hash_mlp)
        for p in self.dcff_renderer.hash_grid.hash_encoding.parameters():
            p.requires_grad_(False)
        if getattr(self.dcff_renderer.hash_grid, "sh_encoding", None) is not None:
            for p in self.dcff_renderer.hash_grid.sh_encoding.parameters():
                p.requires_grad_(False)
        self.set_train_mode(False)

        colmap_dir = map_cfg.get("colmap_dir") or render_cfg.get("dataset", {}).get("colmap_dir")
        if not colmap_dir:
            raise ValueError("map_supervision.colmap_dir is required when map supervision is enabled")
        cameras = read_colmap_cameras(os.path.join(colmap_dir, "cameras.bin"))
        images = read_colmap_images(os.path.join(colmap_dir, "images.bin"))

        self.name_to_pose = {}
        self.name_to_intr = {}
        self.basename_to_name = {}
        for image_meta in images.values():
            name = image_meta.name.replace("\\", "/")
            self.name_to_pose[name] = torch.from_numpy(colmap_to_w2c(image_meta.qvec, image_meta.tvec)).float()
            self.name_to_intr[name] = camera_params_to_intrinsics(
                cameras[image_meta.camera_id],
                target_hw=self.feature_hw,
            )
            self.basename_to_name.setdefault(Path(name).name, name)
        self.all_pose_names = sorted(self.name_to_pose.keys())
        self.name_to_center = {
            name: self._camera_center_from_w2c(pose)
            for name, pose in self.name_to_pose.items()
        }

        logger.info(
            "Map renderer loaded: config=%s, views=%d, feature_hw=%s, cache=%s, trainable=%s, fine_decoder=%s, coarse_fusion=%s, feat_sharp=%s, fsm=%s, hash_mlp=%s, latent=%s, geometry=%s, global_neg=%d",
            config_path,
            len(self.name_to_pose),
            self.feature_hw,
            self.cache_in_memory,
            self.trainable,
            self.train_fine_decoder,
            self.train_coarse_fusion,
            self.train_feat_sharp,
            self.train_fsm,
            self.train_hash_mlp,
            self.train_latent,
            self.train_geometry,
            self.global_render_negative_count,
        )

    def _normalize_name(self, sample_name):
        normalized = str(sample_name).replace("\\", "/")
        if normalized in self.name_to_pose:
            return normalized
        basename = Path(normalized).name
        if basename in self.basename_to_name:
            return self.basename_to_name[basename]
        raise KeyError(f"Missing COLMAP pose for sample '{sample_name}'")

    @staticmethod
    def _camera_center_from_w2c(pose):
        pose_f = pose.detach().cpu().float()
        return -(pose_f[:3, :3].T @ pose_f[:3, 3])

    def _sample_global_negative_names(self, normalized, count):
        if count <= 0:
            return []
        query_center = self.name_to_center[normalized]
        candidates = []
        for name in self.all_pose_names:
            if name == normalized:
                continue
            if self.global_render_negative_min_trans_m > 0.0:
                dist = torch.linalg.norm(self.name_to_center[name] - query_center).item()
                if dist < self.global_render_negative_min_trans_m:
                    continue
            candidates.append(name)
        if not candidates:
            candidates = [name for name in self.all_pose_names if name != normalized]
        if not candidates:
            return []
        if len(candidates) >= count:
            return random.sample(candidates, count)
        return [random.choice(candidates) for _ in range(count)]

    def _sample_translation_offset(self):
        offset = torch.zeros(3, dtype=torch.float32)
        if self.perturb_trans_cm_choices:
            axis = random.choice(self.perturb_axes)
            sign = -1.0 if random.random() < 0.5 else 1.0
            offset[axis] = sign * random.choice(self.perturb_trans_cm_choices) / 100.0
            return offset, float(torch.linalg.norm(offset).item())
        trans_sigma = max(0.0, self.perturb_trans_m)
        if trans_sigma > 0:
            offset = torch.from_numpy(np.random.normal(0.0, trans_sigma, size=3).astype(np.float32))
            return offset, float(torch.linalg.norm(offset).item())
        return offset, 0.0

    def _perturb_w2c_pose_with_distance(self, pose):
        pose_tensor = pose.detach().cpu().float().clone()
        center = -(pose_tensor[:3, :3].T @ pose_tensor[:3, 3])
        rot_sigma = np.deg2rad(max(0.0, self.perturb_rot_deg))
        offset, dist_m = self._sample_translation_offset()
        if self.perturb_pose_mode in ("se3", "se3_delta", "delta", "lattice"):
            delta = torch.zeros(1, 6, dtype=torch.float32)
            delta[0, :3] = offset
            if self.perturb_rot_deg_choices:
                axis = random.choice(self.perturb_axes)
                sign = -1.0 if random.random() < 0.5 else 1.0
                delta[0, 3 + axis] = math.radians(sign * random.choice(self.perturb_rot_deg_choices))
            elif rot_sigma > 0:
                delta[0, 3:] = torch.from_numpy(np.random.normal(0.0, rot_sigma, size=3).astype(np.float32))
            return apply_pose_delta(pose_tensor.unsqueeze(0), delta).squeeze(0), dist_m
        if self.perturb_rot_deg_choices:
            axis = random.choice(self.perturb_axes)
            sign = -1.0 if random.random() < 0.5 else 1.0
            angle = torch.tensor(
                math.radians(sign * random.choice(self.perturb_rot_deg_choices)),
                dtype=torch.float32,
            )
            delta_r = axis_angle_rotation_matrix(axis, angle)
            pose_tensor[:3, :3] = delta_r @ pose_tensor[:3, :3]
            pose_tensor[:3, 3] = -(pose_tensor[:3, :3] @ center)
        elif rot_sigma > 0:
            rx, ry, rz = np.random.normal(0.0, rot_sigma, size=3).astype(np.float32)
            cx, sx = np.cos(rx), np.sin(rx)
            cy, sy = np.cos(ry), np.sin(ry)
            cz, sz = np.cos(rz), np.sin(rz)
            rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            delta_r = torch.from_numpy(rot_z @ rot_y @ rot_x).float()
            pose_tensor[:3, :3] = delta_r @ pose_tensor[:3, :3]
            pose_tensor[:3, 3] = -(pose_tensor[:3, :3] @ center)
        if torch.linalg.norm(offset).item() <= 0.0:
            return pose_tensor, dist_m
        return perturb_w2c_camera_center(pose_tensor, offset, frame=self.perturb_frame), dist_m

    def _perturb_w2c_pose(self, pose):
        perturbed, _dist_m = self._perturb_w2c_pose_with_distance(pose)
        return perturbed

    def has_trainable_params(self):
        return (
            self.train_fine_decoder
            or self.train_coarse_fusion
            or self.train_feat_sharp
            or self.train_fsm
            or self.train_hash_mlp
            or self.train_latent
            or self.train_geometry
            or self.train_color
        )

    def set_train_mode(self, enabled):
        if enabled and self.has_trainable_params():
            self.dcff_renderer.train()
            self.feat_sharp.train()
            if self.feat_select is not None:
                self.feat_select.train(self.train_fsm)
        else:
            self.dcff_renderer.eval()
            self.feat_sharp.eval()
            if self.feat_select is not None:
                self.feat_select.eval()

    def get_param_groups(self, base_lr, weight_decay):
        groups = []
        if self.train_fine_decoder:
            groups.append(
                {
                    "params": list(self.dcff_renderer.fine_decoder.parameters()),
                    "lr": base_lr * self.map_lr_scale,
                    "weight_decay": weight_decay,
                }
            )
        if self.train_coarse_fusion and getattr(self.dcff_renderer, "coarse_carrier_fusion", None) is not None:
            groups.append(
                {
                    "params": list(self.dcff_renderer.coarse_carrier_fusion.parameters()),
                    "lr": base_lr * self.map_lr_scale,
                    "weight_decay": weight_decay,
                    "name": "map_coarse_fusion",
                }
            )
        if self.train_feat_sharp:
            groups.append(
                {
                    "params": list(self.feat_sharp.parameters()),
                    "lr": base_lr * self.map_lr_scale,
                    "weight_decay": weight_decay,
                }
            )
        if self.train_fsm and self.feat_select is not None:
            groups.append(
                {
                    "params": list(self.feat_select.parameters()),
                    "lr": base_lr * self.map_lr_scale,
                    "weight_decay": weight_decay,
                }
            )
        if self.train_hash_mlp:
            groups.append(
                {
                    "params": list(self.dcff_renderer.hash_grid.mlp.parameters()),
                    "lr": base_lr * self.hash_mlp_lr_scale,
                    "weight_decay": weight_decay,
                }
            )
        if self.train_latent:
            groups.append(
                {
                    "params": [self.gaussians._latent],
                    "lr": base_lr * self.latent_lr_scale,
                    "weight_decay": weight_decay,
                }
            )
        if self.train_geometry:
            geom_specs = [
                (self.gaussians._xyz, self.position_lr_scale, "map_xyz"),
                (self.gaussians._rotation, self.rotation_lr_scale, "map_rotation"),
                (self.gaussians._scaling, self.scaling_lr_scale, "map_scaling"),
                (self.gaussians._opacity, self.opacity_lr_scale, "map_opacity"),
            ]
            for param, lr_scale, name in geom_specs:
                scale = self.geometry_lr_scale if lr_scale is None else float(lr_scale)
                groups.append(
                    {
                        "params": [param],
                        "lr": base_lr * scale,
                        "weight_decay": weight_decay,
                        "name": name,
                    }
                )
        if self.train_color:
            color_scale = self.geometry_lr_scale if self.color_lr_scale is None else float(self.color_lr_scale)
            groups.append(
                {
                    "params": [
                        self.gaussians._features_dc,
                        self.gaussians._features_rest,
                    ],
                    "lr": base_lr * color_scale,
                    "weight_decay": weight_decay,
                    "name": "map_color",
                }
            )
        return [group for group in groups if group["params"]]

    def export_trainable_state(self):
        state = {}
        if self.train_fine_decoder:
            state["fine_decoder"] = self.dcff_renderer.fine_decoder.state_dict()
        if self.train_coarse_fusion and getattr(self.dcff_renderer, "coarse_carrier_fusion", None) is not None:
            state["coarse_fusion"] = self.dcff_renderer.coarse_carrier_fusion.state_dict()
        if self.train_feat_sharp:
            state["feat_sharp"] = self.feat_sharp.state_dict()
        if self.train_fsm and self.feat_select is not None:
            state["fsm"] = self.feat_select.state_dict()
        if self.train_hash_mlp:
            state["hash_grid_mlp"] = self.dcff_renderer.hash_grid.mlp.state_dict()
        if self.train_latent:
            state["gaussian_latent"] = self.gaussians._latent.detach().cpu()
        if self.train_geometry:
            state["gaussian_geometry"] = {
                "xyz": self.gaussians._xyz.detach().cpu(),
                "rotation": self.gaussians._rotation.detach().cpu(),
                "scaling": self.gaussians._scaling.detach().cpu(),
                "opacity": self.gaussians._opacity.detach().cpu(),
            }
        if self.train_color:
            state["gaussian_color"] = {
                "features_dc": self.gaussians._features_dc.detach().cpu(),
                "features_rest": self.gaussians._features_rest.detach().cpu(),
            }
        return state

    def load_trainable_state(self, state_dict):
        if not state_dict:
            return
        if "fine_decoder" in state_dict:
            try:
                self.dcff_renderer.fine_decoder.load_state_dict(state_dict["fine_decoder"])
            except RuntimeError as e:
                self.logger.info("Skipping map warmstart fine_decoder (architecture changed): %s", e)
        if "coarse_fusion" in state_dict and getattr(self.dcff_renderer, "coarse_carrier_fusion", None) is not None:
            try:
                self.dcff_renderer.coarse_carrier_fusion.load_state_dict(state_dict["coarse_fusion"])
            except RuntimeError as e:
                self.logger.info("Skipping map warmstart coarse_fusion (architecture changed): %s", e)
        if "feat_sharp" in state_dict:
            try:
                self.feat_sharp.load_state_dict(state_dict["feat_sharp"])
            except RuntimeError as e:
                self.logger.info("Skipping map warmstart feat_sharp (architecture changed): %s", e)
        if "fsm" in state_dict and self.feat_select is not None:
            try:
                self.feat_select.load_state_dict(state_dict["fsm"])
            except RuntimeError as e:
                self.logger.info("Skipping map warmstart FSM (architecture changed): %s", e)
        if "hash_grid_mlp" in state_dict:
            try:
                self.dcff_renderer.hash_grid.mlp.load_state_dict(state_dict["hash_grid_mlp"])
            except RuntimeError as e:
                self.logger.info("Skipping map warmstart hash_grid_mlp (architecture changed): %s", e)
        if "gaussian_latent" in state_dict:
            latent = state_dict["gaussian_latent"]
            if latent.shape == self.gaussians._latent.shape:
                self.gaussians._latent.data.copy_(latent.to(self.device))
        if "gaussian_geometry" in state_dict:
            try:
                geom = state_dict["gaussian_geometry"]
                self.gaussians._xyz.data.copy_(geom["xyz"].to(self.device))
                self.gaussians._rotation.data.copy_(geom["rotation"].to(self.device))
                self.gaussians._scaling.data.copy_(geom["scaling"].to(self.device))
                self.gaussians._opacity.data.copy_(geom["opacity"].to(self.device))
            except KeyError as e:
                self.logger.info("Skipping map warmstart geometry (missing key): %s", e)
        if "gaussian_color" in state_dict:
            try:
                color = state_dict["gaussian_color"]
                self.gaussians._features_dc.data.copy_(color["features_dc"].to(self.device))
                self.gaussians._features_rest.data.copy_(color["features_rest"].to(self.device))
            except KeyError as e:
                self.logger.info("Skipping map warmstart color (missing key): %s", e)

    def clear_cache(self):
        self._cache.clear()

    def _render_single(self, sample_name, require_grad=False):
        normalized = self._normalize_name(sample_name)
        use_cache = self.cache_in_memory and not require_grad
        if use_cache and normalized in self._cache:
            fine_raw_cpu, fine_cpu, coarse_cpu, mask_cpu, alpha_cpu, rgb_cpu, depth_cpu, position_cpu = self._cache[normalized]
            return (
                fine_raw_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                fine_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                coarse_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                mask_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                alpha_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                rgb_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                depth_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                position_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
            )

        pose = self.name_to_pose[normalized].to(self.device)
        K = intrinsics_to_K(self.name_to_intr[normalized], self.device)
        result = self.dcff_renderer(
            self.gaussians,
            viewmat=pose,
            K=K,
            width=self.render_width,
            height=self.render_height,
            render_coarse=True,
            feature_height=self.render_height,
            feature_width=self.render_width,
        )
        fine_raw = result["fine_features"].float()
        post_result = _apply_dcff_postprocess(
            result,
            self.render_height,
            self.render_width,
            feat_sharp=self.feat_sharp,
            feat_select=self.feat_select,
            use_coarse_for_fsm=bool(getattr(self.dcff_renderer, '_fsm_use_coarse', False)),
            temperature=0.5,
            hard=False,
        )
        fine = post_result["fine_features"].float()
        coarse = post_result["coarse_features"].float()
        alpha = post_result["alpha"].float()
        alpha_feat = F.interpolate(alpha, size=self.feature_hw, mode="bilinear", align_corners=False)
        mask = (alpha_feat > self.alpha_threshold).float()
        rgb = result.get("rgb")
        if rgb is None:
            rgb = torch.zeros((1, 3, *self.feature_hw), device=self.device, dtype=fine.dtype)
        elif rgb.shape[-2:] != self.feature_hw:
            rgb = F.interpolate(rgb.float(), size=self.feature_hw, mode="bilinear", align_corners=False)
        else:
            rgb = rgb.float()
        depth = result.get("depth")
        if depth is None:
            depth = torch.zeros((1, 1, *self.feature_hw), device=self.device, dtype=fine.dtype)
        elif depth.shape[-2:] != self.feature_hw:
            depth = F.interpolate(depth.float(), size=self.feature_hw, mode="bilinear", align_corners=False)
        else:
            depth = depth.float()
        position = self.dcff_renderer.depth_to_position_map(depth, K, pose)
        if position.ndim == 4:
            position = position.permute(0, 3, 1, 2).contiguous()
        else:
            position = position.permute(2, 0, 1).unsqueeze(0).contiguous()
        position = position.float()

        if use_cache:
            self._cache[normalized] = (
                fine_raw.detach().cpu().to(self.cache_dtype),
                fine.detach().cpu().to(self.cache_dtype),
                coarse.detach().cpu().to(self.cache_dtype),
                mask.detach().cpu().to(self.cache_dtype),
                alpha_feat.detach().cpu().to(self.cache_dtype),
                rgb.detach().cpu().to(self.cache_dtype),
                depth.detach().cpu().to(self.cache_dtype),
                position.detach().cpu().to(self.cache_dtype),
            )
        return fine_raw, fine, coarse, mask, alpha_feat, rgb, depth, position

    def _render_pose(self, sample_name, pose, require_grad=False, feature="all", include_aux=True):
        normalized = self._normalize_name(sample_name)
        K = intrinsics_to_K(self.name_to_intr[normalized], self.device)
        feature = str(feature or "all").lower()
        render_fine = feature not in ("coarse", "query_coarse")
        result = self.dcff_renderer(
            self.gaussians,
            viewmat=pose.to(self.device),
            K=K,
            width=self.render_width,
            height=self.render_height,
            render_coarse=True,
            render_fine=render_fine,
            feature_height=self.render_height,
            feature_width=self.render_width,
        )
        if render_fine:
            fine_raw = result["fine_features"].float()
            post_result = _apply_dcff_postprocess(
                result,
                self.render_height,
                self.render_width,
                feat_sharp=self.feat_sharp,
                feat_select=self.feat_select,
                use_coarse_for_fsm=bool(getattr(self.dcff_renderer, '_fsm_use_coarse', False)),
                temperature=0.5,
                hard=False,
            )
            fine = post_result["fine_features"].float()
            coarse = post_result["coarse_features"].float()
            alpha = post_result["alpha"].float()
        else:
            fine_raw = None
            fine = None
            coarse = result["coarse_features"].float()
            alpha = result["alpha"].float()
        alpha_feat = F.interpolate(alpha, size=self.feature_hw, mode="bilinear", align_corners=False)
        mask = (alpha_feat > self.alpha_threshold).float()
        feature_dtype = coarse.dtype
        if include_aux:
            rgb = result.get("rgb")
            if rgb is None:
                rgb = torch.zeros((1, 3, *self.feature_hw), device=self.device, dtype=feature_dtype)
            elif rgb.shape[-2:] != self.feature_hw:
                rgb = F.interpolate(rgb.float(), size=self.feature_hw, mode="bilinear", align_corners=False)
            else:
                rgb = rgb.float()
            depth = result.get("depth")
            if depth is None:
                depth = torch.zeros((1, 1, *self.feature_hw), device=self.device, dtype=feature_dtype)
            elif depth.shape[-2:] != self.feature_hw:
                depth = F.interpolate(depth.float(), size=self.feature_hw, mode="bilinear", align_corners=False)
            else:
                depth = depth.float()
            position = self.dcff_renderer.depth_to_position_map(depth, K, pose.to(self.device))
            if position.ndim == 4:
                position = position.permute(0, 3, 1, 2).contiguous()
            else:
                position = position.permute(2, 0, 1).unsqueeze(0).contiguous()
        else:
            rgb = torch.zeros((1, 3, *self.feature_hw), device=self.device, dtype=feature_dtype)
            depth = torch.zeros((1, 1, *self.feature_hw), device=self.device, dtype=feature_dtype)
            position = torch.zeros((1, 3, *self.feature_hw), device=self.device, dtype=feature_dtype)
        return fine_raw, fine, coarse, mask, alpha_feat, rgb, depth, position.float()

    def _render_pose_bank_coarse_chunked(self, batch, pose_bank, *, require_grad=False, include_aux=False):
        """Render a coarse-only candidate bank in chunks while preserving candidate order."""
        if include_aux:
            raise ValueError("coarse chunked candidate render only supports include_aux=False")
        bsz, num_candidates = pose_bank.shape[:2]
        flat_poses = pose_bank.reshape(bsz * num_candidates, 4, 4).to(self.device, dtype=torch.float32)
        flat_K = []
        intrinsics_rows = []
        for sample_name in batch["sample_name"]:
            normalized = self._normalize_name(sample_name)
            K = intrinsics_to_K(self.name_to_intr[normalized], self.device)
            flat_K.append(K.expand(num_candidates, -1, -1))
            intr = self.name_to_intr[normalized]
            intrinsics_rows.append(
                torch.tensor(
                    [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])],
                    device=self.device,
                    dtype=torch.float32,
                ).expand(num_candidates, -1)
            )
        flat_K = torch.cat(flat_K, dim=0)
        chunk_size = int(self.candidate_render_batch_size or flat_poses.shape[0])
        chunk_size = max(1, min(chunk_size, flat_poses.shape[0]))
        coarse_chunks = []
        mask_chunks = []
        context = torch.enable_grad if require_grad else torch.no_grad
        with context():
            for start in range(0, flat_poses.shape[0], chunk_size):
                end = min(start + chunk_size, flat_poses.shape[0])
                result = self.dcff_renderer(
                    self.gaussians,
                    viewmat=flat_poses[start:end],
                    K=flat_K[start:end],
                    width=self.render_width,
                    height=self.render_height,
                    render_coarse=True,
                    render_fine=False,
                    feature_height=self.render_height,
                    feature_width=self.render_width,
                )
                coarse = result["coarse_features"].float()
                alpha = result["alpha"].float()
                alpha_feat = F.interpolate(alpha, size=self.feature_hw, mode="bilinear", align_corners=False)
                coarse_chunks.append(coarse)
                mask_chunks.append((alpha_feat > self.alpha_threshold).float())
        coarse = torch.cat(coarse_chunks, dim=0).reshape(
            bsz,
            num_candidates,
            -1,
            self.feature_hw[0],
            self.feature_hw[1],
        )
        mask = torch.cat(mask_chunks, dim=0).reshape(
            bsz,
            num_candidates,
            1,
            self.feature_hw[0],
            self.feature_hw[1],
        )
        depth = torch.zeros(
            (bsz, num_candidates, 1, self.feature_hw[0], self.feature_hw[1]),
            device=self.device,
            dtype=coarse.dtype,
        )
        intrinsics = torch.stack(intrinsics_rows, dim=0)
        return coarse, mask, depth, intrinsics

    def attach_pose_candidate_renders(
        self,
        batch,
        candidate_poses,
        *,
        require_grad=False,
        prefix="rendered_map_candidate",
        candidate_valid_mask=None,
        max_candidates=None,
        feature="all",
        include_aux=True,
    ):
        """Render a post-forward bank of pose candidates for listwise feature scoring."""
        feature = str(feature or "all").lower()
        pose_bank = candidate_poses.to(self.device, dtype=torch.float32)
        if pose_bank.ndim != 4 or pose_bank.shape[-2:] != (4, 4):
            raise ValueError(f"candidate_poses must have shape (B,K,4,4), got {tuple(pose_bank.shape)}")
        bsz, num_candidates = pose_bank.shape[:2]
        if bsz != len(batch["sample_name"]):
            raise ValueError(
                f"candidate pose batch size {bsz} does not match sample_name count {len(batch['sample_name'])}"
            )
        if max_candidates is not None and int(max_candidates) > 0:
            keep = min(num_candidates, int(max_candidates))
            pose_bank = pose_bank[:, :keep]
            num_candidates = keep
            if candidate_valid_mask is not None:
                candidate_valid_mask = candidate_valid_mask[:, :keep]

        if (
            int(getattr(self, "candidate_render_batch_size", 0) or 0) > 0
            and feature in ("coarse", "query_coarse")
            and not include_aux
        ):
            coarse, mask, depth, intrinsics = self._render_pose_bank_coarse_chunked(
                batch,
                pose_bank,
                require_grad=require_grad,
                include_aux=include_aux,
            )
            batch[f"{prefix}_pose"] = pose_bank
            batch[f"{prefix}_coarse"] = coarse
            batch[f"{prefix}_mask"] = mask
            batch[f"{prefix}_depth"] = depth
            batch[f"{prefix}_intrinsics"] = intrinsics
            if candidate_valid_mask is not None:
                batch[f"{prefix}_valid_mask"] = candidate_valid_mask.to(self.device).bool()
            return batch

        fine_rows = []
        coarse_rows = []
        mask_rows = []
        depth_rows = []
        intrinsics_rows = []
        pose_rows = []
        context = torch.enable_grad if require_grad else torch.no_grad
        with context():
            for batch_idx, sample_name in enumerate(batch["sample_name"]):
                sample_fine = []
                sample_coarse = []
                sample_mask = []
                sample_depth = []
                sample_intrinsics = []
                sample_pose = []
                normalized = self._normalize_name(sample_name) if hasattr(self, "_normalize_name") else sample_name
                intr = getattr(self, "name_to_intr", {}).get(normalized)
                if intr is None:
                    intr = getattr(self, "name_to_intr", {}).get(sample_name)
                intr_tensor = None
                if intr is not None:
                    intr_tensor = torch.tensor(
                        [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])],
                        device=self.device,
                        dtype=torch.float32,
                    )
                for cand_idx in range(num_candidates):
                    pose = pose_bank[batch_idx, cand_idx]
                    _fine_raw, fine, coarse, mask, _alpha, _rgb, depth, _position = self._render_pose(
                        sample_name,
                        pose,
                        require_grad=require_grad,
                        feature=feature,
                        include_aux=include_aux,
                    )
                    if fine is not None:
                        sample_fine.append(fine.squeeze(0))
                    sample_coarse.append(coarse.squeeze(0))
                    sample_mask.append(mask.squeeze(0))
                    sample_depth.append(depth.squeeze(0))
                    if intr_tensor is not None:
                        sample_intrinsics.append(intr_tensor)
                    sample_pose.append(pose)
                if sample_fine:
                    fine_rows.append(torch.stack(sample_fine, dim=0))
                coarse_rows.append(torch.stack(sample_coarse, dim=0))
                mask_rows.append(torch.stack(sample_mask, dim=0))
                depth_rows.append(torch.stack(sample_depth, dim=0))
                if sample_intrinsics:
                    intrinsics_rows.append(torch.stack(sample_intrinsics, dim=0))
                pose_rows.append(torch.stack(sample_pose, dim=0))

        batch[f"{prefix}_pose"] = torch.stack(pose_rows, dim=0)
        if fine_rows:
            batch[f"{prefix}_fine"] = torch.stack(fine_rows, dim=0)
        batch[f"{prefix}_coarse"] = torch.stack(coarse_rows, dim=0)
        batch[f"{prefix}_mask"] = torch.stack(mask_rows, dim=0)
        batch[f"{prefix}_depth"] = torch.stack(depth_rows, dim=0)
        if intrinsics_rows:
            batch[f"{prefix}_intrinsics"] = torch.stack(intrinsics_rows, dim=0)
        if candidate_valid_mask is not None:
            batch[f"{prefix}_valid_mask"] = candidate_valid_mask.to(self.device).bool()
        return batch

    def attach_to_batch(self, batch, require_grad=False):
        fine_raw_list = []
        fine_list = []
        coarse_list = []
        mask_list = []
        alpha_list = []
        rgb_list = []
        depth_list = []
        position_list = []
        intrinsics_list = []
        pose_list = []
        context = torch.enable_grad if require_grad else torch.no_grad
        with context():
            for sample_name in batch["sample_name"]:
                normalized = self._normalize_name(sample_name)
                fine_raw, fine, coarse, mask, alpha, rgb, depth, position = self._render_single(
                    sample_name,
                    require_grad=require_grad,
                )
                fine_raw_list.append(fine_raw.squeeze(0))
                fine_list.append(fine.squeeze(0))
                coarse_list.append(coarse.squeeze(0))
                mask_list.append(mask.squeeze(0))
                alpha_list.append(alpha.squeeze(0))
                rgb_list.append(rgb.squeeze(0))
                depth_list.append(depth.squeeze(0))
                position_list.append(position.squeeze(0))
                pose_list.append(self.name_to_pose[normalized].to(self.device))
                intr = self.name_to_intr[normalized]
                intrinsics_list.append(
                    torch.tensor(
                        [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])],
                        device=self.device,
                        dtype=torch.float32,
                    )
                )

        neg_fine_raw_list = []
        neg_fine_list = []
        neg_coarse_list = []
        neg_mask_list = []
        neg_alpha_list = []
        neg_depth_list = []
        neg_position_list = []
        neg_flow_list = []
        neg_flow_valid_list = []
        neg_dist_m_list = []
        neg_pose_list = []
        if self.perturb_render_negatives:
            with context():
                for sample_name in batch["sample_name"]:
                    normalized = self._normalize_name(sample_name)
                    neg_pose, neg_dist_m = self._perturb_w2c_pose_with_distance(self.name_to_pose[normalized])
                    fine_raw, fine, coarse, mask, alpha, _rgb, depth, position = self._render_pose(
                        sample_name,
                        neg_pose,
                        require_grad=require_grad,
                    )
                    with torch.no_grad():
                        gt_pose = self.name_to_pose[normalized].to(self.device)
                        flow, flow_valid = compute_w2c_flow(
                            neg_pose.unsqueeze(0).to(self.device),
                            gt_pose.unsqueeze(0),
                            depth.detach(),
                            self.name_to_intr[normalized],
                            target_hw=self.feature_hw,
                        )
                    neg_fine_raw_list.append(fine_raw.squeeze(0))
                    neg_fine_list.append(fine.squeeze(0))
                    neg_coarse_list.append(coarse.squeeze(0))
                    neg_mask_list.append(mask.squeeze(0))
                    neg_alpha_list.append(alpha.squeeze(0))
                    neg_depth_list.append(depth.squeeze(0))
                    neg_position_list.append(position.squeeze(0))
                    neg_flow_list.append(flow.squeeze(0))
                    neg_flow_valid_list.append(flow_valid.squeeze(0))
                    neg_dist_m_list.append(float(neg_dist_m))
                    neg_pose_list.append(neg_pose.to(self.device))

        global_neg_coarse_list = []
        global_neg_mask_list = []
        global_neg_pose_list = []
        if self.global_render_negative_count > 0:
            with context():
                for sample_name in batch["sample_name"]:
                    normalized = self._normalize_name(sample_name)
                    neg_names = self._sample_global_negative_names(
                        normalized,
                        self.global_render_negative_count,
                    )
                    if not neg_names:
                        continue
                    sample_coarse = []
                    sample_mask = []
                    sample_pose = []
                    for neg_name in neg_names:
                        _fine_raw, _fine, coarse, mask, _alpha, _rgb, _depth, _position = self._render_single(
                            neg_name,
                            require_grad=require_grad,
                        )
                        sample_coarse.append(coarse.squeeze(0))
                        sample_mask.append(mask.squeeze(0))
                        sample_pose.append(self.name_to_pose[neg_name].to(self.device))
                    global_neg_coarse_list.append(torch.stack(sample_coarse, dim=0))
                    global_neg_mask_list.append(torch.stack(sample_mask, dim=0))
                    global_neg_pose_list.append(torch.stack(sample_pose, dim=0))

        batch["rendered_map_fine_raw"] = torch.stack(fine_raw_list, dim=0)
        batch["rendered_map_fine"] = torch.stack(fine_list, dim=0)
        batch["rendered_map_coarse"] = torch.stack(coarse_list, dim=0)
        batch["rendered_map_mask"] = torch.stack(mask_list, dim=0)
        batch["rendered_map_alpha"] = torch.stack(alpha_list, dim=0)
        batch["rendered_map_rgb"] = torch.stack(rgb_list, dim=0)
        batch["rendered_map_depth"] = torch.stack(depth_list, dim=0)
        batch["rendered_map_position"] = torch.stack(position_list, dim=0)
        batch["rendered_map_intrinsics"] = torch.stack(intrinsics_list, dim=0)
        batch["rendered_map_pose_gt"] = torch.stack(pose_list, dim=0)
        if neg_fine_list:
            batch["rendered_map_fine_neg"] = torch.stack(neg_fine_list, dim=0)
            batch["rendered_map_coarse_neg"] = torch.stack(neg_coarse_list, dim=0)
            batch["rendered_map_mask_neg"] = torch.stack(neg_mask_list, dim=0)
            batch["rendered_map_alpha_neg"] = torch.stack(neg_alpha_list, dim=0)
            batch["rendered_map_depth_neg"] = torch.stack(neg_depth_list, dim=0)
            batch["rendered_map_position_neg"] = torch.stack(neg_position_list, dim=0)
            batch["rendered_map_flow_neg_to_gt"] = torch.stack(neg_flow_list, dim=0)
            batch["rendered_map_flow_valid_neg_to_gt"] = torch.stack(neg_flow_valid_list, dim=0)
            batch["rendered_map_pose_neg"] = torch.stack(neg_pose_list, dim=0)
            batch["rendered_map_neg_dist_m"] = torch.tensor(neg_dist_m_list, device=self.device, dtype=torch.float32)
        if len(global_neg_coarse_list) == len(batch["sample_name"]):
            batch["rendered_map_coarse_global_neg"] = torch.stack(global_neg_coarse_list, dim=0)
            batch["rendered_map_mask_global_neg"] = torch.stack(global_neg_mask_list, dim=0)
            batch["rendered_map_pose_global_neg"] = torch.stack(global_neg_pose_list, dim=0)
        return batch


def move_batch_to_device(batch, device):
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def resolve_linear_weight(map_cfg, key, epoch):
    start = float(map_cfg.get(key, 0.0))
    end_key = f"{key}_end"
    anneal_key = f"{key}_anneal_epochs"
    if end_key not in map_cfg or anneal_key not in map_cfg:
        return start
    end = float(map_cfg.get(end_key, start))
    anneal_epochs = max(1, int(map_cfg.get(anneal_key, 1)))
    alpha = min(max(float(epoch), 0.0) / float(anneal_epochs), 1.0)
    return start + (end - start) * alpha


POSE_CANDIDATE_BATCH_SOURCES = ("batch", "cache", "dataset", "pose_init_candidates")
POSE_CANDIDATE_MODEL_SOURCES = ("pose_init", "model_pose_init", "feature_bank_online", "online_feature_bank")
POSE_CANDIDATE_LATTICE_SOURCES = ("coarse_pose_lattice", "local_pose_lattice", "pose_lattice")


def _candidate_render_feature_request(map_cfg, *, epoch=0):
    requested = []
    render_weight = resolve_linear_weight(map_cfg, "candidate_render_score_weight", epoch)
    fusion_weight = resolve_linear_weight(map_cfg, "candidate_score_fusion_weight", epoch)
    if render_weight > 0.0:
        requested.append(str(map_cfg.get("candidate_render_score_feature", "coarse")).lower())
    if fusion_weight > 0.0:
        requested.append(str(map_cfg.get("candidate_score_fusion_feature", "coarse")).lower())
    if requested and all(feature in ("coarse", "query_coarse") for feature in requested):
        return "coarse"
    return "all"


def should_preattach_pose_candidate_renders(cfg, *, epoch=0):
    """Return True when pose candidates are known before the query forward pass."""
    map_cfg = cfg.get("map_supervision", {})
    if not map_cfg.get("enabled", False):
        return False
    render_weight = resolve_linear_weight(map_cfg, "candidate_render_score_weight", epoch)
    fusion_weight = resolve_linear_weight(map_cfg, "candidate_score_fusion_weight", epoch)
    if render_weight <= 0.0 and fusion_weight <= 0.0:
        return False
    source = str(map_cfg.get("candidate_render_pose_source", "pose_init")).lower()
    return source in POSE_CANDIDATE_BATCH_SOURCES or source in POSE_CANDIDATE_LATTICE_SOURCES


def _resolve_pose_lattice_base_pose(batch, outputs, map_cfg):
    source = str(map_cfg.get("coarse_pose_lattice_base_source", "rendered_map_pose_neg"))
    source_key = source.lower()
    if source_key in ("pose_init", "model_pose_init"):
        pose_init = outputs.get("pose_init")
        if not isinstance(pose_init, dict):
            return None
        pose = pose_init.get("pose_w2c")
    elif source_key in ("pose_gt", "gt", "rendered_map_pose_gt"):
        pose = batch.get("pose_gt", batch.get("rendered_map_pose_gt"))
    elif source_key in ("pose_neg", "rendered_map_pose_neg", "init", "t0"):
        pose = batch.get("rendered_map_pose_neg", batch.get("pose_init", batch.get("pose_gt")))
    else:
        pose = batch.get(source)
    if pose is None:
        return None
    pose_t = pose.detach()
    if pose_t.ndim == 4:
        base_idx = int(map_cfg.get("coarse_pose_lattice_base_index", 0) or 0)
        base_idx = max(0, min(base_idx, pose_t.shape[1] - 1))
        pose_t = pose_t[:, base_idx]
    if pose_t.ndim != 3 or pose_t.shape[-2:] != (4, 4):
        raise ValueError(f"local pose lattice base pose must have shape (B,4,4), got {tuple(pose_t.shape)}")
    return pose_t


def _select_pose_lattice_oracle_subset(candidate_poses, candidate_valid_mask, pose_gt, map_cfg):
    """Keep a small render subset while guaranteeing the GT-nearest candidate is present."""
    subset_size = int(map_cfg.get("coarse_pose_lattice_oracle_subset_size", 0) or 0)
    if subset_size <= 0 or pose_gt is None:
        return candidate_poses, candidate_valid_mask, None
    if candidate_poses.ndim != 4 or candidate_poses.shape[-2:] != (4, 4):
        return candidate_poses, candidate_valid_mask, None
    bsz, num_candidates = candidate_poses.shape[:2]
    subset_size = max(1, min(subset_size, num_candidates))
    if subset_size >= num_candidates:
        return candidate_poses, candidate_valid_mask, None
    if candidate_valid_mask is None:
        valid = torch.ones(
            bsz,
            num_candidates,
            device=candidate_poses.device,
            dtype=torch.bool,
        )
    else:
        valid = candidate_valid_mask.to(device=candidate_poses.device, dtype=torch.bool)
    pose_gt_t = pose_gt.to(device=candidate_poses.device, dtype=candidate_poses.dtype)
    if pose_gt_t.ndim != 3 or pose_gt_t.shape[0] != bsz:
        return candidate_poses, candidate_valid_mask, None
    with torch.no_grad():
        _target_idx, _trans_err, _rot_err, pose_cost = _candidate_pose_target(
            candidate_poses,
            pose_gt_t,
            valid,
            float(map_cfg.get("coarse_pose_lattice_oracle_subset_rot_cost_weight", 0.1)),
        )
        extras_count = max(0, subset_size - 1)
        if extras_count > 0:
            extra_idx = select_pose_candidate_indices(
                num_candidates,
                limit=extras_count,
                strategy=str(map_cfg.get("coarse_pose_lattice_oracle_subset_extras_strategy", "uniform")),
            ).to(device=candidate_poses.device)
        else:
            extra_idx = torch.empty(0, device=candidate_poses.device, dtype=torch.long)
        all_idx = torch.arange(num_candidates, device=candidate_poses.device, dtype=torch.long)
        selected_rows = []
        for bidx in range(bsz):
            cost_row = pose_cost[bidx].masked_fill(~valid[bidx], float("inf"))
            best_idx = int(cost_row.argmin().item())
            row = [best_idx]
            for idx in extra_idx.detach().cpu().tolist():
                idx_i = int(idx)
                if bool(valid[bidx, idx_i]) and idx_i not in row:
                    row.append(idx_i)
                if len(row) >= subset_size:
                    break
            if len(row) < subset_size:
                sorted_idx = torch.argsort(cost_row)
                for idx in sorted_idx.detach().cpu().tolist():
                    idx_i = int(idx)
                    if not math.isfinite(float(cost_row[idx_i].item())):
                        continue
                    if idx_i not in row:
                        row.append(idx_i)
                    if len(row) >= subset_size:
                        break
            if len(row) < subset_size:
                for idx in all_idx.detach().cpu().tolist():
                    idx_i = int(idx)
                    if idx_i not in row:
                        row.append(idx_i)
                    if len(row) >= subset_size:
                        break
            selected_rows.append(torch.tensor(row[:subset_size], device=candidate_poses.device, dtype=torch.long))
        selected_idx = torch.stack(selected_rows, dim=0)
    selected_poses = gather_candidate_bank(candidate_poses, selected_idx)
    selected_valid = gather_candidate_bank(valid, selected_idx)
    return selected_poses, selected_valid, selected_idx


def maybe_attach_pose_candidate_renders(batch, outputs, cfg, map_renderer, *, require_grad=False, epoch=0):
    map_cfg = cfg.get("map_supervision", {})
    if map_renderer is None or not map_cfg.get("enabled", False):
        return batch
    render_weight = resolve_linear_weight(map_cfg, "candidate_render_score_weight", epoch)
    fusion_weight = resolve_linear_weight(map_cfg, "candidate_score_fusion_weight", epoch)
    if render_weight <= 0.0 and fusion_weight <= 0.0:
        return batch
    source = str(map_cfg.get("candidate_render_pose_source", "pose_init")).lower()
    candidate_valid_mask = None
    candidate_scores = None
    if source in POSE_CANDIDATE_BATCH_SOURCES:
        candidate_poses = batch.get("pose_init_candidates")
        candidate_valid_mask = batch.get("candidate_valid_mask")
        if candidate_poses is None:
            return batch
    elif source in POSE_CANDIDATE_LATTICE_SOURCES:
        base_pose = _resolve_pose_lattice_base_pose(batch, outputs, map_cfg)
        if base_pose is None:
            return batch
        candidate_poses = build_local_pose_lattice_candidates(
            base_pose,
            trans_cm=map_cfg.get("coarse_pose_lattice_trans_cm", [0.0, 5.0, 10.0, 25.0]),
            rot_deg=map_cfg.get("coarse_pose_lattice_rot_deg", [0.0, 1.0, 2.0, 5.0]),
            include_identity=bool(map_cfg.get("coarse_pose_lattice_include_identity", True)),
            max_candidates=int(map_cfg.get("coarse_pose_lattice_max_candidates", 0) or 0),
            limit_strategy=str(map_cfg.get("coarse_pose_lattice_limit_strategy", "head") or "head"),
            combine_trans_rot=bool(map_cfg.get("coarse_pose_lattice_combine_trans_rot", False)),
        )
        candidate_valid_mask = torch.ones(
            candidate_poses.shape[:2],
            device=candidate_poses.device,
            dtype=torch.bool,
        )
        batch["coarse_pose_lattice_base_pose"] = base_pose
        pose_gt_for_subset = batch.get("rendered_map_pose_gt", batch.get("pose_gt"))
        candidate_poses, candidate_valid_mask, subset_indices = _select_pose_lattice_oracle_subset(
            candidate_poses,
            candidate_valid_mask,
            pose_gt_for_subset,
            map_cfg,
        )
        if subset_indices is not None:
            batch["coarse_pose_lattice_oracle_subset_indices"] = subset_indices.detach()
    else:
        pose_init = outputs.get("pose_init")
        if not isinstance(pose_init, dict) or "pose_w2c" not in pose_init:
            return batch
        candidate_poses = pose_init["pose_w2c"]
        candidate_valid_mask = pose_init.get("candidate_valid_mask", pose_init.get("valid_mask"))
        candidate_scores = pose_init.get("scores")
    candidate_poses = candidate_poses.detach()
    if candidate_poses.ndim != 4 or candidate_poses.shape[1] <= 0:
        return batch
    if source in POSE_CANDIDATE_MODEL_SOURCES:
        if candidate_valid_mask is None:
            candidate_valid_mask = torch.ones(
                candidate_poses.shape[:2],
                device=candidate_poses.device,
                dtype=torch.bool,
            )
        else:
            candidate_valid_mask = candidate_valid_mask.to(device=candidate_poses.device).bool().detach()
            if candidate_valid_mask.shape != candidate_poses.shape[:2]:
                raise ValueError(
                    "pose_init candidate valid mask must have shape "
                    f"{tuple(candidate_poses.shape[:2])}, got {tuple(candidate_valid_mask.shape)}"
                )
        if candidate_scores is not None:
            score_t = candidate_scores.detach().to(device=candidate_poses.device, dtype=torch.float32)
            if score_t.shape != candidate_poses.shape[:2]:
                raise ValueError(
                    "pose_init candidate scores must have shape "
                    f"{tuple(candidate_poses.shape[:2])}, got {tuple(score_t.shape)}"
                )
            valid_for_scores = candidate_valid_mask.to(device=score_t.device).bool()
            valid_min = score_t.masked_fill(~valid_for_scores, float("inf")).amin(dim=1, keepdim=True)
            valid_min = torch.where(torch.isfinite(valid_min), valid_min, torch.zeros_like(valid_min))
            score_quality = torch.clamp(score_t - valid_min, min=0.0)
            score_quality = torch.where(valid_for_scores, score_quality, torch.zeros_like(score_quality))
            batch.setdefault("retrieval_original_scores_candidates", score_t)
            batch.setdefault("retrieval_scores_candidates", score_quality)
    render_with_grad = bool(require_grad and map_cfg.get("candidate_render_score_train_map", False))
    target_mode = str(map_cfg.get("candidate_score_fusion_target_mode", "hard")).lower()
    needs_candidate_aux = (
        "wls" in target_mode
        or "refined" in target_mode
        or float(map_cfg.get("candidate_refined_pose_weight", 0.0) or 0.0) > 0.0
    )
    include_aux = bool(map_cfg.get("candidate_render_include_aux", True)) or needs_candidate_aux
    max_candidates = int(
        map_cfg.get(
            "candidate_stage1_topk",
            map_cfg.get("candidate_render_score_max_candidates", 0),
        )
        or map_cfg.get("candidate_render_score_max_candidates", 0)
        or 0
    )
    return map_renderer.attach_pose_candidate_renders(
        batch,
        candidate_poses,
        require_grad=render_with_grad,
        candidate_valid_mask=candidate_valid_mask,
        max_candidates=max_candidates,
        feature=_candidate_render_feature_request(map_cfg, epoch=epoch),
        include_aux=include_aux,
    )


def resolve_perturb_rank_margin(map_cfg, batch, device):
    margin = torch.tensor(float(map_cfg.get("perturb_margin", 0.1)), device=device)
    margin_per_m = float(map_cfg.get("perturb_margin_per_m", 0.0))
    if margin_per_m != 0.0 and batch.get("rendered_map_neg_dist_m") is not None:
        dist = batch["rendered_map_neg_dist_m"].to(device=device, dtype=torch.float32)
        margin = margin + margin_per_m * dist.mean()
    return margin


def resolve_query_feature_dims(cfg, teacher_store):
    """Resolve fine/coarse query dimensions from config or teacher cache."""
    model_cfg = cfg.setdefault("model", {})
    dataset_cfg = cfg.setdefault("dataset", {})
    fallback_dim = int(model_cfg.get("feature_dim", teacher_store.fine_feature_dim))

    fine_dim_cfg = model_cfg.get("fine_feature_dim")
    coarse_dim_cfg = model_cfg.get("coarse_feature_dim")
    fine_dim = teacher_store.fine_feature_dim if fine_dim_cfg is None else int(fine_dim_cfg)
    coarse_dim = teacher_store.coarse_feature_dim if coarse_dim_cfg is None else int(coarse_dim_cfg)

    if fine_dim != teacher_store.fine_feature_dim:
        raise ValueError(
            f"Query fine_feature_dim={fine_dim} does not match teacher fine dim "
            f"{teacher_store.fine_feature_dim}"
        )
    if coarse_dim != teacher_store.coarse_feature_dim:
        raise ValueError(
            f"Query coarse_feature_dim={coarse_dim} does not match teacher coarse dim "
            f"{teacher_store.coarse_feature_dim}"
        )

    model_cfg["feature_dim"] = fallback_dim
    model_cfg["fine_feature_dim"] = fine_dim
    model_cfg["coarse_feature_dim"] = coarse_dim
    dataset_cfg["teacher_feature_hw"] = list(teacher_store.feature_hw)
    dataset_cfg["teacher_coarse_feature_hw"] = list(teacher_store.coarse_feature_hw)
    dataset_cfg["feature_hw"] = list(
        dataset_cfg.get("student_feature_hw") or teacher_store.feature_hw
    )
    dataset_cfg["coarse_feature_hw"] = list(
        dataset_cfg.get("student_coarse_feature_hw") or teacher_store.coarse_feature_hw
    )
    return fine_dim, coarse_dim


def _candidate_score_map_channel_count(mode, radius):
    mode_key = str(mode or "peak_offset").lower()
    volume_channels = (2 * int(radius) + 1) ** 2
    if mode_key == "volume":
        return volume_channels
    if mode_key in ("volume_plus_peak_offset", "volume+peak_offset"):
        return volume_channels + 3
    return 3


def _candidate_fusion_vector_dim(map_cfg, model_cfg):
    if model_cfg.get("candidate_score_fusion_input_dim") is not None:
        return int(model_cfg["candidate_score_fusion_input_dim"])
    render_mode = str(map_cfg.get("candidate_score_fusion_render_feature_mode", "basic")).lower()
    render_dim = 9 if render_mode in ("rich", "stats", "spatial_stats") else 3
    return render_dim + len(CANDIDATE_QUALITY_FEATURE_NAMES)


def _build_candidate_score_fusion_head(model_cfg, map_cfg):
    if not bool(model_cfg.get("candidate_score_fusion_head", False)):
        return None
    vector_dim = _candidate_fusion_vector_dim(map_cfg, model_cfg)
    hidden_dim = int(model_cfg.get("candidate_score_fusion_hidden_dim", 64))
    zero_init = bool(model_cfg.get("candidate_score_fusion_zero_init", False))
    initial_bias = float(model_cfg.get("candidate_score_fusion_initial_bias", 0.0))
    use_score_map = bool(model_cfg.get("candidate_score_fusion_use_score_map_head", False))
    if use_score_map:
        radius = int(map_cfg.get("candidate_score_fusion_radius", map_cfg.get("candidate_render_score_radius", 4)))
        score_map_mode = str(map_cfg.get("candidate_score_fusion_score_map_mode", "peak_offset"))
        return CandidateScoreMapFusionHead(
            vector_dim=vector_dim,
            score_map_channels=int(
                model_cfg.get(
                    "candidate_score_fusion_score_map_channels",
                    _candidate_score_map_channel_count(score_map_mode, radius),
                )
            ),
            map_channels=int(model_cfg.get("candidate_score_fusion_map_channels", 8)),
            grid_size=int(model_cfg.get("candidate_score_fusion_grid_size", 4)),
            hidden_dim=hidden_dim,
            zero_init=zero_init,
            initial_vector_weights=model_cfg.get("candidate_score_fusion_initial_weights"),
            initial_bias=initial_bias,
            context_layers=int(model_cfg.get("candidate_score_fusion_context_layers", 0)),
            context_heads=int(model_cfg.get("candidate_score_fusion_context_heads", 1)),
            context_feedforward_dim=model_cfg.get("candidate_score_fusion_context_feedforward_dim"),
            context_residual=bool(model_cfg.get("candidate_score_fusion_context_residual", False)),
        )
    return CandidateScoreFusionHead(
        input_dim=vector_dim,
        hidden_dim=hidden_dim,
        zero_init=zero_init,
        initial_weights=model_cfg.get("candidate_score_fusion_initial_weights"),
        initial_bias=initial_bias,
    )


def build_radio_query_student(
    cfg,
    *,
    fine_feature_dim,
    coarse_feature_dim,
    retrieval_dim=None,
    retrieval_hidden_dim=None,
):
    """Build the query student from the same config used by training/eval."""
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    retrieval_cfg = cfg.get("retrieval", {})
    model = RadioQueryStudent(
        in_channels=3,
        feature_dim=int(model_cfg["feature_dim"]),
        fine_feature_dim=int(fine_feature_dim),
        coarse_feature_dim=int(coarse_feature_dim),
        base_channels=int(model_cfg["base_channels"]),
        stage_dims=tuple(model_cfg["stage_dims"]),
        output_hw=tuple(dataset_cfg["feature_hw"]),
        coarse_output_hw=tuple(dataset_cfg.get("coarse_feature_hw") or dataset_cfg["feature_hw"]),
        input_hw=tuple(dataset_cfg["input_hw"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        l2_normalize=bool(model_cfg.get("l2_normalize", True)),
        predict_magnitude=bool(model_cfg.get("predict_magnitude", False)),
        fine_init_norm=float(model_cfg.get("fine_init_norm", 1.0)),
        coarse_init_norm=float(model_cfg.get("coarse_init_norm", 1.0)),
        magnitude_min=float(model_cfg.get("magnitude_min", 1e-4)),
        retrieval_dim=retrieval_dim,
        retrieval_hidden_dim=retrieval_hidden_dim,
        retrieval_dropout=float(retrieval_cfg.get("dropout", 0.0)) if retrieval_dim is not None else 0.0,
        retrieval_l2_normalize=bool(retrieval_cfg.get("l2_normalize", True)),
        fine_low_level_skip=bool(model_cfg.get("fine_low_level_skip", False)),
        fine_low_level_init=float(model_cfg.get("fine_low_level_init", 0.0)),
        fine_highres_skip=bool(model_cfg.get("fine_highres_skip", False)),
        fine_highres_source=str(model_cfg.get("fine_highres_source", "stage2")),
        fine_highres_init=float(model_cfg.get("fine_highres_init", 0.0)),
        fine_highres_zero_init=bool(model_cfg.get("fine_highres_zero_init", False)),
        global_context_enabled=bool(model_cfg.get("global_context_enabled", False)),
        global_context_zero_init=bool(model_cfg.get("global_context_zero_init", True)),
        window_attention_layers=int(model_cfg.get("window_attention_layers", 0)),
        window_attention_heads=int(model_cfg.get("window_attention_heads", 8)),
        window_attention_size=int(model_cfg.get("window_attention_size", 16)),
        window_attention_mlp_ratio=float(model_cfg.get("window_attention_mlp_ratio", 2.0)),
        window_attention_dropout=float(model_cfg.get("window_attention_dropout", 0.0)),
        window_attention_shift=bool(model_cfg.get("window_attention_shift", False)),
        window_attention_zero_init=bool(model_cfg.get("window_attention_zero_init", True)),
        teacher_fine_condition=bool(model_cfg.get("teacher_fine_condition", False)),
        teacher_fine_init=float(model_cfg.get("teacher_fine_init", 1.0)),
        teacher_fine_zero_init=bool(model_cfg.get("teacher_fine_zero_init", True)),
        teacher_fine_detach=bool(model_cfg.get("teacher_fine_detach", True)),
        scene_coord_head=bool(model_cfg.get("scene_coord_head", False)),
        scene_coord_zero_init=bool(model_cfg.get("scene_coord_zero_init", True)),
        scene_coord_detach_base=bool(model_cfg.get("scene_coord_detach_base", False)),
        scene_coord_use_pixel_grid=bool(model_cfg.get("scene_coord_use_pixel_grid", False)),
        scene_coord_global_context=bool(model_cfg.get("scene_coord_global_context", False)),
        local_matcher_enabled=bool(model_cfg.get("local_matcher_enabled", False)),
        local_matcher_radius=int(model_cfg.get("local_matcher_radius", 4)),
        local_matcher_hidden_dim=int(model_cfg.get("local_matcher_hidden_dim", 64)),
        local_matcher_zero_init=bool(model_cfg.get("local_matcher_zero_init", True)),
        local_matcher_residual_scale=float(model_cfg.get("local_matcher_residual_scale", 1.0)),
        local_matcher_context_mode=str(model_cfg.get("local_matcher_context_mode", "basic")),
        local_flow_head_enabled=bool(model_cfg.get("local_flow_head_enabled", False)),
        local_flow_head_radius=int(model_cfg.get("local_flow_head_radius", model_cfg.get("local_matcher_radius", 4))),
        local_flow_head_hidden_dim=int(model_cfg.get("local_flow_head_hidden_dim", 64)),
        local_flow_head_zero_init=bool(model_cfg.get("local_flow_head_zero_init", True)),
        local_flow_head_max_flow=model_cfg.get("local_flow_head_max_flow"),
        local_flow_head_base_flow_mode=str(model_cfg.get("local_flow_head_base_flow_mode", "none")),
        local_flow_head_base_temperature=float(model_cfg.get("local_flow_head_base_temperature", 0.05)),
        local_flow_head_context_mode=str(model_cfg.get("local_flow_head_context_mode", "basic")),
        local_corr_projector_enabled=bool(model_cfg.get("local_corr_projector_enabled", False)),
        local_corr_projector_hidden_dim=int(model_cfg.get("local_corr_projector_hidden_dim", 96)),
        local_corr_projector_output_dim=model_cfg.get("local_corr_projector_output_dim"),
        local_corr_projector_zero_init=bool(model_cfg.get("local_corr_projector_zero_init", True)),
        local_corr_projector_l2_normalize=bool(
            model_cfg.get("local_corr_projector_l2_normalize", True)
        ),
        local_corr_projector_domain_adapter=bool(model_cfg.get("local_corr_projector_domain_adapter", False)),
        local_corr_query_projector_zero_init=model_cfg.get("local_corr_query_projector_zero_init"),
        local_corr_render_projector_zero_init=model_cfg.get("local_corr_render_projector_zero_init"),
        query_channel_gate_enabled=bool(model_cfg.get("query_channel_gate_enabled", False)),
        query_channel_gate_hidden_dim=model_cfg.get("query_channel_gate_hidden_dim"),
        query_channel_gate_zero_init=bool(model_cfg.get("query_channel_gate_zero_init", True)),
        apply_query_channel_gate=bool(model_cfg.get("apply_query_channel_gate", False)),
    )
    model.candidate_score_fusion_head = _build_candidate_score_fusion_head(model_cfg, cfg.get("map_supervision", {}))
    return model


def build_model_param_groups(model, *, base_lr, weight_decay, lr_scales=None):
    """Build optimizer groups with optional name-prefix LR multipliers."""
    scales = {
        str(prefix): float(scale)
        for prefix, scale in (lr_scales or {}).items()
        if str(prefix) and abs(float(scale) - 1.0) > 1e-12
    }
    if not scales:
        return [
            {
                "params": [param for param in model.parameters() if param.requires_grad],
                "lr": float(base_lr),
                "weight_decay": float(weight_decay),
            }
        ]

    ordered_prefixes = sorted(scales, key=len, reverse=True)
    grouped_params = {prefix: [] for prefix in ordered_prefixes}
    default_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = next((prefix for prefix in ordered_prefixes if name.startswith(prefix)), None)
        if matched is None:
            default_params.append(param)
        else:
            grouped_params[matched].append(param)

    groups = []
    if default_params:
        groups.append(
            {
                "name": "model.default",
                "params": default_params,
                "lr": float(base_lr),
                "weight_decay": float(weight_decay),
            }
        )
    for prefix in ordered_prefixes:
        params = grouped_params[prefix]
        if not params:
            continue
        groups.append(
            {
                "name": f"model.{prefix}",
                "params": params,
                "lr": float(base_lr) * scales[prefix],
                "weight_decay": float(weight_decay),
            }
        )
    return groups


def _matches_name_prefix(name, prefixes):
    for prefix in prefixes or []:
        prefix = str(prefix)
        if not prefix:
            continue
        if name == prefix or name.startswith(prefix):
            return True
    return False


def apply_model_trainable_filter(model, *, trainable_prefixes=None):
    """Freeze all model parameters except those matching requested name prefixes."""
    prefixes = [str(prefix) for prefix in (trainable_prefixes or []) if str(prefix)]
    summary = {
        "trainable_tensors": 0,
        "frozen_tensors": 0,
        "trainable_parameters": 0,
        "frozen_parameters": 0,
    }
    if not prefixes:
        for param in model.parameters():
            if param.requires_grad:
                summary["trainable_tensors"] += 1
                summary["trainable_parameters"] += int(param.numel())
            else:
                summary["frozen_tensors"] += 1
                summary["frozen_parameters"] += int(param.numel())
        return summary

    for name, param in model.named_parameters():
        keep_trainable = _matches_name_prefix(name, prefixes)
        param.requires_grad_(keep_trainable)
        if keep_trainable:
            summary["trainable_tensors"] += 1
            summary["trainable_parameters"] += int(param.numel())
        else:
            summary["frozen_tensors"] += 1
            summary["frozen_parameters"] += int(param.numel())
    return summary


def depth_observability_weight(
    depth,
    mask=None,
    *,
    strength=1.0,
    power=1.0,
    max_weight=4.0,
):
    """Build a normalized inverse-depth weight for translation-observable pixels."""
    if strength <= 0:
        if mask is not None:
            return torch.ones_like(mask.float())
        depth_f = depth.float()
        if depth_f.ndim == 3:
            depth_f = depth_f.unsqueeze(1)
        return torch.ones_like(depth_f)

    depth_f = depth.float()
    if depth_f.ndim == 3:
        depth_f = depth_f.unsqueeze(1)
    valid = depth_f > 0.05
    if mask is not None:
        mask_f = mask.float()
        if mask_f.ndim == 3:
            mask_f = mask_f.unsqueeze(1)
        if mask_f.shape[-2:] != depth_f.shape[-2:]:
            mask_f = F.interpolate(mask_f, size=depth_f.shape[-2:], mode="nearest")
        valid = valid & (mask_f > 0)
    else:
        mask_f = valid.float()

    inv_depth = torch.where(
        valid,
        depth_f.clamp(min=0.05).pow(-float(power)),
        torch.zeros_like(depth_f),
    )
    denom = valid.float().sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    mean = inv_depth.sum(dim=(1, 2, 3), keepdim=True) / denom
    normalized = inv_depth / mean.clamp(min=1e-6)
    normalized = normalized.clamp(max=float(max_weight))
    blended = 1.0 + float(strength) * (normalized - 1.0)
    return torch.where(valid, blended.clamp(min=0.0), torch.zeros_like(blended))


def translation_observability_weight(
    depth,
    intrinsics,
    mask=None,
    *,
    strength=1.0,
    mode="xyz",
    power=1.0,
    max_weight=4.0,
):
    """Build normalized pixel weights from the translational image Jacobian norm."""
    if strength <= 0:
        if mask is not None:
            return torch.ones_like(mask.float())
        depth_f = depth.float()
        if depth_f.ndim == 3:
            depth_f = depth_f.unsqueeze(1)
        return torch.ones_like(depth_f)

    depth_f = depth.float()
    if depth_f.ndim == 3:
        depth_f = depth_f.unsqueeze(1)
    B, _C, H, W = depth_f.shape
    device = depth_f.device
    dtype = depth_f.dtype
    valid = depth_f > 0.05
    if mask is not None:
        mask_f = mask.float()
        if mask_f.ndim == 3:
            mask_f = mask_f.unsqueeze(1)
        if mask_f.shape[-2:] != (H, W):
            mask_f = F.interpolate(mask_f, size=(H, W), mode="nearest")
        valid = valid & (mask_f > 0)

    intr = intrinsics
    if isinstance(intr, torch.Tensor):
        intr_t = intr.to(device=device, dtype=dtype)
        if intr_t.ndim == 1:
            intr_t = intr_t.view(1, 4).expand(B, -1)
        fx = intr_t[:, 0].view(B, 1, 1, 1)
        fy = intr_t[:, 1].view(B, 1, 1, 1)
        cx = intr_t[:, 2].view(B, 1, 1, 1)
        cy = intr_t[:, 3].view(B, 1, 1, 1)
    else:
        fx = torch.full((B, 1, 1, 1), float(intr["fx"]), device=device, dtype=dtype)
        fy = torch.full((B, 1, 1, 1), float(intr["fy"]), device=device, dtype=dtype)
        cx = torch.full((B, 1, 1, 1), float(intr["cx"]), device=device, dtype=dtype)
        cy = torch.full((B, 1, 1, 1), float(intr["cy"]), device=device, dtype=dtype)

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    u = u_coords.view(1, 1, H, W)
    v = v_coords.view(1, 1, H, W)
    x = (u - cx) / fx.clamp(min=1e-6)
    y = (v - cy) / fy.clamp(min=1e-6)
    inv_z = depth_f.clamp(min=0.05).reciprocal()

    mode_key = str(mode).lower()
    tx = fx * inv_z
    ty = fy * inv_z
    tz_u = fx * x * inv_z
    tz_v = fy * y * inv_z
    if mode_key == "xy":
        obs = torch.sqrt(tx.square() + ty.square()).clamp(min=0.0)
    elif mode_key == "z":
        obs = torch.sqrt(tz_u.square() + tz_v.square()).clamp(min=0.0)
    elif mode_key == "xyz":
        obs = torch.sqrt(tx.square() + ty.square() + tz_u.square() + tz_v.square()).clamp(min=0.0)
    else:
        raise ValueError(f"Unknown translation observability mode '{mode}'. Use xy, z, or xyz.")

    obs = torch.where(valid, obs.pow(float(power)), torch.zeros_like(obs))
    denom = valid.float().sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    mean = obs.sum(dim=(1, 2, 3), keepdim=True) / denom
    normalized = obs / mean.clamp(min=1e-6)
    normalized = normalized.clamp(max=float(max_weight))
    blended = 1.0 + float(strength) * (normalized - 1.0)
    return torch.where(valid, blended.clamp(min=0.0), torch.zeros_like(blended))


def feature_orthogonality_loss(pred_a, pred_b, mask=None):
    if pred_a.shape[1] != pred_b.shape[1]:
        return pred_a.new_zeros(())
    if pred_a.shape[-2:] != pred_b.shape[-2:]:
        pred_b = F.interpolate(pred_b, pred_a.shape[-2:], mode="bilinear", align_corners=False)
    pred_a_n = F.normalize(pred_a, dim=1)
    pred_b_n = F.normalize(pred_b, dim=1)
    cos = (pred_a_n * pred_b_n).sum(dim=1, keepdim=True)
    penalty = cos.square()
    if mask is not None:
        penalty = penalty * mask
        denom = mask.sum().clamp(min=1.0)
        return penalty.sum() / denom
    return penalty.mean()


def feature_variance_loss(feat, mask=None, target_std=0.05):
    """Penalize collapsed feature channels using a VICReg-style variance floor."""
    B, C, H, W = feat.shape
    x = feat.float().reshape(B, C, -1)
    if mask is not None:
        m = mask.float()
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, (H, W), mode="nearest")
        m = m.reshape(B, 1, -1)
        denom = m.sum(dim=-1).clamp(min=1.0)
        mean = (x * m).sum(dim=-1) / denom
        var = ((x - mean.unsqueeze(-1)) ** 2 * m).sum(dim=-1) / denom
    else:
        var = x.var(dim=-1, unbiased=False)
    std = torch.sqrt(var + 1e-6)
    return F.relu(float(target_std) - std).mean()


def feature_covariance_loss(feat, mask=None, max_samples=1024):
    """Reduce channel redundancy without requiring fine/coarse same dimensionality."""
    B, C, H, W = feat.shape
    x = feat.float().permute(0, 2, 3, 1).reshape(-1, C)
    if mask is not None:
        m = mask.float()
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m, (H, W), mode="nearest")
        valid = (m.reshape(-1) > 0.5).nonzero(as_tuple=True)[0]
        if valid.numel() > 1:
            x = x[valid]
    if x.shape[0] > max_samples:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_samples]
        x = x[idx]
    if x.shape[0] <= 1:
        return feat.new_zeros(())
    x = x - x.mean(dim=0, keepdim=True)
    x = x / x.std(dim=0, keepdim=True).clamp(min=1e-6)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / max(1, C * (C - 1))


def masked_global_feature_descriptor(feat, mask=None):
    feat_f = feat.float()
    if mask is None:
        return feat_f.mean(dim=(-1, -2))
    mask_f = mask.float()
    if mask_f.ndim == 3:
        mask_f = mask_f.unsqueeze(1)
    if mask_f.shape[-2:] != feat_f.shape[-2:]:
        mask_f = F.interpolate(mask_f, size=feat_f.shape[-2:], mode="nearest")
    denom = torch.clamp(mask_f.sum(dim=(-1, -2)), min=1.0)
    return (feat_f * mask_f).sum(dim=(-1, -2)) / denom


def coarse_pose_ranking_loss(query_coarse, pos_coarse, neg_coarse, pos_mask=None, neg_mask=None, temperature=0.07):
    query_desc = F.normalize(masked_global_feature_descriptor(query_coarse, pos_mask), dim=1)
    pos_desc = F.normalize(masked_global_feature_descriptor(pos_coarse, pos_mask), dim=1)
    neg_desc = F.normalize(masked_global_feature_descriptor(neg_coarse, neg_mask), dim=1)
    pos_sim = (query_desc * pos_desc).sum(dim=1)
    neg_sim = (query_desc * neg_desc).sum(dim=1)
    logits = torch.stack([pos_sim, neg_sim], dim=1) / max(float(temperature), 1e-6)
    targets = torch.zeros(query_desc.shape[0], dtype=torch.long, device=query_desc.device)
    loss = F.cross_entropy(logits, targets)
    metrics = {
        "map_coarse_pose_rank_loss": loss.detach(),
        "map_coarse_pose_rank_pos": pos_sim.mean().detach(),
        "map_coarse_pose_rank_neg": neg_sim.mean().detach(),
        "map_coarse_pose_rank_gap": (pos_sim - neg_sim).mean().detach(),
        "map_coarse_pose_rank_acc": (pos_sim > neg_sim).float().mean().detach(),
    }
    return loss, metrics


def coarse_pose_energy_nce_loss(
    query_coarse,
    pos_coarse,
    neg_coarse,
    pos_mask=None,
    neg_mask=None,
    temperature=0.07,
):
    query_desc = F.normalize(masked_global_feature_descriptor(query_coarse, pos_mask), dim=1)
    pos_desc = F.normalize(masked_global_feature_descriptor(pos_coarse, pos_mask), dim=1)
    if neg_coarse.ndim == 4:
        neg_desc = masked_global_feature_descriptor(neg_coarse, neg_mask).unsqueeze(1)
    elif neg_coarse.ndim == 5:
        bsz, neg_count, channels, height, width = neg_coarse.shape
        neg_flat = neg_coarse.reshape(bsz * neg_count, channels, height, width)
        neg_mask_flat = None
        if neg_mask is not None:
            if neg_mask.ndim == 4:
                neg_mask_flat = neg_mask.reshape(bsz * neg_count, 1, height, width)
            elif neg_mask.ndim == 5:
                neg_mask_flat = neg_mask.reshape(bsz * neg_count, *neg_mask.shape[2:])
            else:
                raise ValueError(f"Unsupported neg_mask ndim={neg_mask.ndim}")
        neg_desc = masked_global_feature_descriptor(neg_flat, neg_mask_flat).reshape(bsz, neg_count, -1)
    else:
        raise ValueError(f"Expected neg_coarse to be 4D or 5D, got shape={tuple(neg_coarse.shape)}")
    neg_desc = F.normalize(neg_desc, dim=-1)
    pos_sim = (query_desc * pos_desc).sum(dim=1, keepdim=True)
    neg_sim = torch.einsum("bc,bnc->bn", query_desc, neg_desc)
    logits = torch.cat([pos_sim, neg_sim], dim=1) / max(float(temperature), 1e-6)
    targets = torch.zeros(query_desc.shape[0], dtype=torch.long, device=query_desc.device)
    loss = F.cross_entropy(logits, targets)
    best_neg = neg_sim.max(dim=1).values
    metrics = {
        "map_coarse_pose_energy_loss": loss.detach(),
        "map_coarse_pose_energy_pos": pos_sim.mean().detach(),
        "map_coarse_pose_energy_best_neg": best_neg.mean().detach(),
        "map_coarse_pose_energy_gap": (pos_sim.squeeze(1) - best_neg).mean().detach(),
        "map_coarse_pose_energy_acc": (pos_sim.squeeze(1) > best_neg).float().mean().detach(),
        "map_coarse_pose_energy_neg_count": torch.tensor(float(neg_sim.shape[1]), device=query_desc.device),
    }
    return loss, metrics


def candidate_local_render_score_nce_loss(
    query_feat,
    pos_feat,
    neg_feat,
    pos_mask=None,
    neg_mask=None,
    *,
    temperature=0.07,
    radius=4,
    preprocess="none",
    highpass_kernel=5,
):
    """Listwise pose-energy loss using local render/query feature correlation."""
    if query_feat.ndim != 4 or pos_feat.ndim != 4:
        raise ValueError("query_feat and pos_feat must have shape (B,C,H,W)")
    if neg_feat.ndim not in (4, 5):
        raise ValueError(f"Expected neg_feat to be 4D or 5D, got shape={tuple(neg_feat.shape)}")
    if query_feat.shape[0] != pos_feat.shape[0]:
        raise ValueError("query_feat and pos_feat batch sizes must match")
    bsz = query_feat.shape[0]
    if neg_feat.shape[0] != bsz:
        raise ValueError("neg_feat batch size must match query batch size")

    pos_scores = []
    neg_scores = []
    for batch_idx in range(bsz):
        pos_mask_i = None
        if pos_mask is not None:
            pos_mask_i = pos_mask[batch_idx : batch_idx + 1]
        pos_result = local_render_score_feature_candidates(
            query_feat[batch_idx],
            pos_feat[batch_idx : batch_idx + 1],
            mask=pos_mask_i,
            radius=radius,
            preprocess=preprocess,
            highpass_kernel=highpass_kernel,
        )
        pos_scores.append(pos_result["scores"].view(1))

        if neg_feat.ndim == 4:
            neg_feat_i = neg_feat[batch_idx : batch_idx + 1]
            neg_mask_i = neg_mask[batch_idx : batch_idx + 1] if neg_mask is not None else None
        else:
            neg_feat_i = neg_feat[batch_idx]
            neg_mask_i = neg_mask[batch_idx] if neg_mask is not None else None
        neg_result = local_render_score_feature_candidates(
            query_feat[batch_idx],
            neg_feat_i,
            mask=neg_mask_i,
            radius=radius,
            preprocess=preprocess,
            highpass_kernel=highpass_kernel,
        )
        neg_scores.append(neg_result["scores"])

    pos_sim = torch.stack(pos_scores, dim=0)
    neg_sim = torch.stack(neg_scores, dim=0)
    logits = torch.cat([pos_sim, neg_sim], dim=1) / max(float(temperature), 1e-6)
    targets = torch.zeros(bsz, dtype=torch.long, device=query_feat.device)
    loss = F.cross_entropy(logits, targets)
    best_neg = neg_sim.max(dim=1).values
    metrics = {
        "map_coarse_pose_local_energy_loss": loss.detach(),
        "map_coarse_pose_local_energy_pos": pos_sim.mean().detach(),
        "map_coarse_pose_local_energy_best_neg": best_neg.mean().detach(),
        "map_coarse_pose_local_energy_gap": (pos_sim.squeeze(1) - best_neg).mean().detach(),
        "map_coarse_pose_local_energy_acc": (pos_sim.squeeze(1) > best_neg).float().mean().detach(),
        "map_coarse_pose_local_energy_neg_count": torch.tensor(float(neg_sim.shape[1]), device=query_feat.device),
    }
    return loss, metrics


def _resize_query_flow_valid(query_feat, flow_gt, valid_mask, target_hw):
    h, w = int(target_hw[0]), int(target_hw[1])
    query = query_feat.float()
    flow = flow_gt.to(device=query.device).float()
    valid = valid_mask.float()
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if query.shape[-2:] != (h, w):
        query = F.interpolate(query, (h, w), mode="bilinear", align_corners=False)
    if flow.shape[-2:] != (h, w):
        src_h, src_w = flow.shape[-2:]
        flow = F.interpolate(flow, (h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / max(src_w, 1)
        flow[:, 1] *= h / max(src_h, 1)
    if valid.shape[-2:] != (h, w):
        valid = F.interpolate(valid, (h, w), mode="nearest")
    return query, flow, valid


def _local_correlation_offsets(radius, device, dtype):
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    channels = (2 * radius + 1) ** 2
    return dx.reshape(1, channels, 1, 1), dy.reshape(1, channels, 1, 1)


def local_correlation_feature_preprocess(
    feat,
    *,
    mode="none",
    highpass_kernel=3,
    highpass_scale=1.0,
):
    """Preprocess descriptors for local correlation without changing stored features."""
    mode = str(mode or "none").lower()
    if mode in {"none", "identity", "raw"}:
        return feat
    kernel = int(highpass_kernel)
    if kernel <= 1:
        high = torch.zeros_like(feat)
    else:
        if kernel % 2 == 0:
            raise ValueError(f"highpass_kernel must be odd, got {kernel}")
        pad = kernel // 2
        low = F.avg_pool2d(
            feat,
            kernel_size=kernel,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )
        high = feat - low
    high = high * float(highpass_scale)
    if mode in {"highpass", "hp"}:
        return high
    if mode in {"residual_highpass", "highpass_residual", "residual_hp"}:
        return feat + high
    if mode in {"concat_highpass", "cat_highpass", "concat_hp"}:
        return torch.cat([feat, high], dim=1)
    raise ValueError(
        "local correlation feature preprocess must be one of "
        "{'none', 'highpass', 'residual_highpass', 'concat_highpass'}, "
        f"got {mode!r}"
    )


def shifted_local_correlation(fmap1, fmap2, radius=4):
    """Local dot-product correlation without materializing C*window unfold."""
    B, C, H, W = fmap1.shape
    radius = int(radius)
    fmap2_pad = F.pad(fmap2, [radius, radius, radius, radius], mode="constant", value=0)
    corrs = []
    for dy in range(-radius, radius + 1):
        y0 = dy + radius
        for dx in range(-radius, radius + 1):
            x0 = dx + radius
            sampled = fmap2_pad[:, :, y0 : y0 + H, x0 : x0 + W]
            corrs.append((fmap1 * sampled).sum(dim=1))
    return torch.stack(corrs, dim=1).contiguous()


def local_correlation_distribution_loss(
    rendered_feat,
    query_feat,
    valid_mask=None,
    *,
    target_feat=None,
    radius=4,
    student_temperature=0.05,
    target_temperature=0.05,
    feature_preprocess="none",
    highpass_kernel=3,
    highpass_scale=1.0,
):
    """Distill query-map local correlation toward a map-self correlation distribution."""
    with torch.cuda.amp.autocast(enabled=False):
        rendered = rendered_feat.float()
        query = query_feat.float()
        if query.shape[-2:] != rendered.shape[-2:]:
            query = F.interpolate(query, size=rendered.shape[-2:], mode="bilinear", align_corners=False)
        if target_feat is None:
            target = rendered
        else:
            target = target_feat.float()
            if target.shape[-2:] != rendered.shape[-2:]:
                target = F.interpolate(target, size=rendered.shape[-2:], mode="bilinear", align_corners=False)
        if valid_mask is None:
            valid_weight = rendered.new_ones(rendered.shape[0], 1, rendered.shape[-2], rendered.shape[-1])
        else:
            valid_weight = valid_mask.float()
            if valid_weight.ndim == 3:
                valid_weight = valid_weight.unsqueeze(1)
            if valid_weight.shape[-2:] != rendered.shape[-2:]:
                valid_weight = F.interpolate(valid_weight, size=rendered.shape[-2:], mode="nearest")
            valid_weight = valid_weight.clamp(min=0.0)

        rendered_corr = local_correlation_feature_preprocess(
            rendered,
            mode=feature_preprocess,
            highpass_kernel=highpass_kernel,
            highpass_scale=highpass_scale,
        )
        query_corr = local_correlation_feature_preprocess(
            query,
            mode=feature_preprocess,
            highpass_kernel=highpass_kernel,
            highpass_scale=highpass_scale,
        )
        target_corr = local_correlation_feature_preprocess(
            target,
            mode=feature_preprocess,
            highpass_kernel=highpass_kernel,
            highpass_scale=highpass_scale,
        )

        rendered_n = F.normalize(rendered_corr, dim=1)
        query_n = F.normalize(query_corr, dim=1)
        target_n = F.normalize(target_corr, dim=1)
        student_corr = shifted_local_correlation(rendered_n, query_n, radius=int(radius)).float()
        with torch.no_grad():
            teacher_corr = shifted_local_correlation(rendered_n, target_n, radius=int(radius)).float()
            target_probs = torch.softmax(
                teacher_corr / max(float(target_temperature), 1e-6),
                dim=1,
            )

        log_probs = F.log_softmax(
            student_corr / max(float(student_temperature), 1e-6),
            dim=1,
        )
        loss_map = -(target_probs * log_probs).sum(dim=1, keepdim=True)
        denom = valid_weight.sum().clamp(min=1.0)
        loss = (loss_map * valid_weight).sum() / denom

        with torch.no_grad():
            student_top2 = torch.topk(student_corr, k=2, dim=1).values
            teacher_top2 = torch.topk(teacher_corr, k=2, dim=1).values
            student_argmax = student_corr.argmax(dim=1, keepdim=True)
            teacher_argmax = teacher_corr.argmax(dim=1, keepdim=True)
            argmax_agree = ((student_argmax == teacher_argmax).float() * valid_weight).sum() / denom
            student_gap = ((student_top2[:, 0:1] - student_top2[:, 1:2]) * valid_weight).sum() / denom
            teacher_gap = ((teacher_top2[:, 0:1] - teacher_top2[:, 1:2]) * valid_weight).sum() / denom

    return loss, {
        "map_query_corr_distill_loss": loss.detach(),
        "map_query_corr_distill_argmax_agree": argmax_agree.detach(),
        "map_query_corr_distill_student_gap": student_gap.detach(),
        "map_query_corr_distill_target_gap": teacher_gap.detach(),
    }


def local_correlation_subpixel_loss(
    rendered_feat,
    query_feat,
    flow_gt,
    valid_mask,
    *,
    radius=4,
    temperature=0.05,
):
    """Soft CE over a rendered-centered local correlation window."""
    with torch.cuda.amp.autocast(enabled=False):
        rendered = rendered_feat.float()
        query, flow, valid_weight = _resize_query_flow_valid(
            query_feat,
            flow_gt,
            valid_mask,
            rendered.shape[-2:],
        )
        rendered_n = F.normalize(rendered, dim=1)
        query_n = F.normalize(query, dim=1)
        corr = shifted_local_correlation(rendered_n, query_n, radius=int(radius)).float()

        B, channels, H, W = corr.shape
        radius = int(radius)
        window = 2 * radius + 1
        expected_channels = window * window
        if channels != expected_channels:
            raise ValueError(f"corr has {channels} channels, expected {expected_channels}")

        log_probs = F.log_softmax(corr / max(float(temperature), 1e-6), dim=1)
        fx = flow[:, 0:1]
        fy = flow[:, 1:2]
        x0 = torch.floor(fx)
        y0 = torch.floor(fy)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        wx1 = (fx - x0).clamp(0.0, 1.0)
        wy1 = (fy - y0).clamp(0.0, 1.0)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        valid_positive = valid_weight > 0.0
        loss_map = torch.zeros(B, 1, H, W, device=corr.device, dtype=corr.dtype)
        target_mass = torch.zeros_like(loss_map)
        for yy, wy in ((y0, wy0), (y1, wy1)):
            for xx, wx in ((x0, wx0), (x1, wx1)):
                in_bounds = (
                    valid_positive
                    & (xx >= -radius)
                    & (xx <= radius)
                    & (yy >= -radius)
                    & (yy <= radius)
                )
                mass = (wx * wy) * in_bounds.float()
                idx = ((yy.long() + radius) * window + (xx.long() + radius)).clamp(
                    0,
                    expected_channels - 1,
                )
                loss_map = loss_map - mass * log_probs.gather(1, idx)
                target_mass = target_mass + mass

        in_window = valid_positive & (target_mass > 1e-6)
        pixel_weight = torch.where(in_window, valid_weight.clamp(min=0.0), torch.zeros_like(valid_weight))
        denom = pixel_weight.sum().clamp(min=1.0)
        loss_per_pixel = loss_map / target_mass.clamp(min=1e-6)
        loss = (loss_per_pixel * pixel_weight).sum() / denom

        with torch.no_grad():
            dx, dy = _local_correlation_offsets(radius, corr.device, corr.dtype)
            probs = torch.softmax(corr / max(float(temperature), 1e-6), dim=1)
            pred_flow = torch.cat(
                [
                    (probs * dx).sum(dim=1, keepdim=True),
                    (probs * dy).sum(dim=1, keepdim=True),
                ],
                dim=1,
            )
            epe_map = torch.linalg.norm(pred_flow - flow, dim=1, keepdim=True)
            epe = (epe_map * pixel_weight).sum() / denom
            nearest_dx = torch.round(fx).long()
            nearest_dy = torch.round(fy).long()
            nearest_target = ((nearest_dy + radius) * window + (nearest_dx + radius)).clamp(
                0,
                expected_channels - 1,
            )
            pred = corr.argmax(dim=1, keepdim=True)
            acc = ((pred == nearest_target) & in_window).float().sum() / in_window.float().sum().clamp(min=1.0)
            coverage = in_window.float().mean()

    return loss, {
        "map_query_corr_subpx_loss": loss.detach(),
        "map_query_corr_subpx_flow_epe": epe.detach(),
        "map_query_corr_subpx_acc": acc.detach(),
        "map_query_corr_subpx_cov": coverage.detach(),
    }


def local_correlation_soft_flow_loss(
    rendered_feat,
    query_feat,
    flow_gt,
    valid_mask,
    *,
    radius=4,
    temperature=0.05,
    huber_delta=1.0,
):
    """Regress subpixel flow as the soft expectation of local correlation."""
    with torch.cuda.amp.autocast(enabled=False):
        rendered = rendered_feat.float()
        query, flow, valid_weight = _resize_query_flow_valid(
            query_feat,
            flow_gt,
            valid_mask,
            rendered.shape[-2:],
        )
        rendered_n = F.normalize(rendered, dim=1)
        query_n = F.normalize(query, dim=1)
        corr = shifted_local_correlation(rendered_n, query_n, radius=int(radius)).float()

        B, channels, H, W = corr.shape
        radius = int(radius)
        expected_channels = (2 * radius + 1) ** 2
        if channels != expected_channels:
            raise ValueError(f"corr has {channels} channels, expected {expected_channels}")

        dx, dy = _local_correlation_offsets(radius, corr.device, corr.dtype)
        weights = torch.softmax(corr / max(float(temperature), 1e-6), dim=1)
        pred_flow = torch.cat(
            [
                (weights * dx).sum(dim=1, keepdim=True),
                (weights * dy).sum(dim=1, keepdim=True),
            ],
            dim=1,
        )
        in_window = (
            (valid_weight > 0.0)
            & (flow[:, :1] >= -radius)
            & (flow[:, :1] <= radius)
            & (flow[:, 1:2] >= -radius)
            & (flow[:, 1:2] <= radius)
        )
        pixel_weight = torch.where(in_window, valid_weight.clamp(min=0.0), torch.zeros_like(valid_weight))
        diff = pred_flow - flow
        abs_diff = diff.abs()
        delta = max(float(huber_delta), 1e-6)
        loss_map = torch.where(abs_diff <= delta, 0.5 * diff.pow(2) / delta, abs_diff - 0.5 * delta)
        denom = (pixel_weight.sum() * 2.0).clamp(min=1.0)
        loss = (loss_map * pixel_weight).sum() / denom

        with torch.no_grad():
            pixel_denom = pixel_weight.sum().clamp(min=1.0)
            epe_map = torch.linalg.norm(diff, dim=1, keepdim=True)
            epe = (epe_map * pixel_weight).sum() / pixel_denom
            coverage = in_window.float().mean()

    return loss, {
        "map_query_corr_flow_loss": loss.detach(),
        "map_query_corr_flow_epe": epe.detach(),
        "map_query_corr_flow_cov": coverage.detach(),
    }


def local_correlation_peak_margin_loss(
    rendered_feat,
    query_feat,
    flow_gt,
    valid_mask,
    *,
    radius=4,
    margin=0.05,
):
    """Make the GT local-correlation logit exceed every non-target offset."""
    with torch.cuda.amp.autocast(enabled=False):
        rendered = rendered_feat.float()
        query, flow, valid_weight = _resize_query_flow_valid(
            query_feat,
            flow_gt,
            valid_mask,
            rendered.shape[-2:],
        )
        rendered_n = F.normalize(rendered, dim=1)
        query_n = F.normalize(query, dim=1)
        corr = shifted_local_correlation(rendered_n, query_n, radius=int(radius)).float()

        B, channels, H, W = corr.shape
        radius = int(radius)
        window = 2 * radius + 1
        expected_channels = window * window
        if channels != expected_channels:
            raise ValueError(f"corr has {channels} channels, expected {expected_channels}")

        fx = flow[:, 0:1]
        fy = flow[:, 1:2]
        x0 = torch.floor(fx)
        y0 = torch.floor(fy)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        wx1 = (fx - x0).clamp(0.0, 1.0)
        wy1 = (fy - y0).clamp(0.0, 1.0)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        valid_positive = valid_weight > 0.0
        pos_logit = torch.zeros(B, 1, H, W, device=corr.device, dtype=corr.dtype)
        target_mass = torch.zeros_like(pos_logit)
        target_mask = torch.zeros_like(corr, dtype=torch.bool)
        for yy, wy in ((y0, wy0), (y1, wy1)):
            for xx, wx in ((x0, wx0), (x1, wx1)):
                in_bounds = (
                    valid_positive
                    & (xx >= -radius)
                    & (xx <= radius)
                    & (yy >= -radius)
                    & (yy <= radius)
                )
                mass = (wx * wy) * in_bounds.float()
                idx = ((yy.long() + radius) * window + (xx.long() + radius)).clamp(
                    0,
                    expected_channels - 1,
                )
                pos_logit = pos_logit + mass * corr.gather(1, idx)
                target_mass = target_mass + mass
                target_mask.scatter_(1, idx, target_mask.gather(1, idx) | in_bounds)

        in_window = valid_positive & (target_mass > 1e-6)
        pixel_weight = torch.where(in_window, valid_weight.clamp(min=0.0), torch.zeros_like(valid_weight))
        pos_logit = pos_logit / target_mass.clamp(min=1e-6)
        neg_logits = corr.masked_fill(target_mask, -1e4)
        hard_neg = neg_logits.max(dim=1, keepdim=True).values
        loss_map = F.relu(float(margin) + hard_neg - pos_logit)
        denom = pixel_weight.sum().clamp(min=1.0)
        loss = (loss_map * pixel_weight).sum() / denom

        with torch.no_grad():
            gap = ((pos_logit - hard_neg) * pixel_weight).sum() / denom
            acc = (((pos_logit > hard_neg).float() * pixel_weight).sum() / denom)
            pos_mean = (pos_logit * pixel_weight).sum() / denom
            neg_mean = (hard_neg * pixel_weight).sum() / denom
            coverage = in_window.float().mean()

    return loss, {
        "map_query_corr_peak_loss": loss.detach(),
        "map_query_corr_peak_gap": gap.detach(),
        "map_query_corr_peak_acc": acc.detach(),
        "map_query_corr_peak_pos": pos_mean.detach(),
        "map_query_corr_peak_neg": neg_mean.detach(),
        "map_query_corr_peak_cov": coverage.detach(),
    }


def _apply_wls_accept_gate(
    pose_pred,
    pose_ref,
    delta_xi,
    confidence,
    valid_wls,
    *,
    min_conf_mean=0.0,
    min_conf_cov=0.0,
    min_delta_mm=0.0,
    max_delta_mm=0.0,
):
    """Return WLS pose when confidence passes gate, otherwise fall back to ref pose."""
    B = pose_pred.shape[0]
    valid = valid_wls.float().reshape(B, -1)
    conf = confidence.float().reshape(B, -1)
    denom = valid.sum(dim=1).clamp(min=1.0)
    conf_mean = (conf * valid).sum(dim=1) / denom
    conf_cov = (((conf > 0.0).float() * valid).sum(dim=1) / denom)
    delta_trans_mm = torch.linalg.norm(delta_xi[:, :3].float(), dim=1) * 1000.0

    accept = torch.ones(B, dtype=torch.bool, device=pose_pred.device)
    if float(min_conf_mean) > 0.0:
        accept = accept & (conf_mean >= float(min_conf_mean))
    if float(min_conf_cov) > 0.0:
        accept = accept & (conf_cov >= float(min_conf_cov))
    if float(min_delta_mm) > 0.0:
        accept = accept & (delta_trans_mm >= float(min_delta_mm))
    if float(max_delta_mm) > 0.0:
        accept = accept & (delta_trans_mm <= float(max_delta_mm))

    gate = accept.to(dtype=pose_pred.dtype).view(B, 1, 1)
    pose_gated = pose_pred * gate + pose_ref * (1.0 - gate)
    accepted = accept.to(dtype=pose_pred.dtype)
    metrics = {
        "map_corr_wls_gated_accept_rate": accepted.mean(),
        "map_corr_wls_gated_conf_mean": (conf_mean * accepted).sum() / accepted.sum().clamp(min=1.0),
        "map_corr_wls_gated_conf_cov": (conf_cov * accepted).sum() / accepted.sum().clamp(min=1.0),
        "map_corr_wls_gated_delta_trans_mm": (
            delta_trans_mm * accepted
        ).sum() / accepted.sum().clamp(min=1.0),
    }
    return pose_gated, metrics


def local_correlation_joint_losses(
    rendered_feat,
    query_feat,
    flow_gt,
    valid_mask,
    *,
    radius=4,
    temperature=0.05,
    ce_temperature=None,
    huber_delta=1.0,
    peak_margin=0.05,
    compute_ce=False,
    compute_subpixel=False,
    compute_flow=False,
    compute_flow_cosine=False,
    compute_peak=False,
    compute_wls_pose=False,
    compute_pose_gain=False,
    depth=None,
    pose_ref=None,
    pose_gt=None,
    intrinsics=None,
    damping=1e-3,
    update_scale=1.0,
    rot_weight=1.0,
    trans_weight=50.0,
    wls_conf_threshold=0.0,
    wls_conf_power=1.0,
    wls_conf_mode="max",
    wls_conf_variance_scale=0.5,
    wls_min_conf_cov=0.0,
    wls_accept_min_conf_mean=0.0,
    wls_accept_min_conf_cov=0.0,
    wls_accept_min_delta_mm=0.0,
    wls_accept_max_delta_mm=0.0,
    pose_gain_trans_margin_m=0.0,
    pose_gain_rot_margin_deg=0.0,
    pose_gain_rot_weight=0.0,
    pose_gain_trans_weight=1.0,
    min_flow_px=0.0,
    max_flow_px=0.0,
    matcher=None,
    flow_head=None,
    flow_head_conf_weight=0.0,
    flow_decode_mode="softargmax",
    feature_preprocess="none",
    highpass_kernel=3,
    highpass_scale=1.0,
    low_peak_gap_threshold=0.0,
):
    """Compute all local-correlation losses from a single correlation volume."""

    def _call_corr_context_module(module, corr_tensor, *, depth_tensor, valid_tensor):
        kwargs = {
            "depth": depth_tensor,
            "valid_mask": valid_tensor,
        }
        callable_obj = getattr(module, "forward", module)
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "intrinsics" in signature.parameters:
            kwargs["intrinsics"] = intrinsics
        return module(corr_tensor, **kwargs)

    with torch.cuda.amp.autocast(enabled=False):
        rendered = rendered_feat.float()
        query, flow, valid_weight = _resize_query_flow_valid(
            query_feat,
            flow_gt,
            valid_mask,
            rendered.shape[-2:],
        )
        min_flow_threshold = float(min_flow_px)
        metrics = {}
        max_flow_threshold = float(max_flow_px)
        if min_flow_threshold > 0.0 or max_flow_threshold > 0.0:
            flow_mag = torch.linalg.norm(flow.float(), dim=1, keepdim=True)
        if min_flow_threshold > 0.0:
            min_flow_mask = flow_mag >= min_flow_threshold
            valid_weight = valid_weight * min_flow_mask.float()
            metrics.update(
                {
                    "map_query_corr_min_flow_px": rendered.new_tensor(min_flow_threshold),
                    "map_query_corr_min_flow_cov": (valid_weight > 0.0).float().mean().detach(),
                }
            )
        if max_flow_threshold > 0.0:
            max_flow_mask = flow_mag <= max_flow_threshold
            valid_weight = valid_weight * max_flow_mask.float()
            metrics.update(
                {
                    "map_query_corr_max_flow_px": rendered.new_tensor(max_flow_threshold),
                    "map_query_corr_max_flow_cov": (valid_weight > 0.0).float().mean().detach(),
                }
            )
        rendered_corr = local_correlation_feature_preprocess(
            rendered,
            mode=feature_preprocess,
            highpass_kernel=highpass_kernel,
            highpass_scale=highpass_scale,
        )
        query_corr = local_correlation_feature_preprocess(
            query,
            mode=feature_preprocess,
            highpass_kernel=highpass_kernel,
            highpass_scale=highpass_scale,
        )
        rendered_n = F.normalize(rendered_corr, dim=1)
        query_n = F.normalize(query_corr, dim=1)
        corr = shifted_local_correlation(rendered_n, query_n, radius=int(radius)).float()
        if matcher is not None:
            corr = _call_corr_context_module(
                matcher,
                corr,
                depth_tensor=depth,
                valid_tensor=valid_weight,
            ).float()

        B, channels, H, W = corr.shape
        radius = int(radius)
        window = 2 * radius + 1
        expected_channels = window * window
        if channels != expected_channels:
            raise ValueError(f"corr has {channels} channels, expected {expected_channels}")

        device = corr.device
        dtype = corr.dtype
        losses = {
            "ce": corr.new_zeros(()),
            "subpixel": corr.new_zeros(()),
            "flow": corr.new_zeros(()),
            "flow_cosine": corr.new_zeros(()),
            "peak": corr.new_zeros(()),
            "wls_pose": corr.new_zeros(()),
            "pose_gain": corr.new_zeros(()),
            "flow_conf": corr.new_zeros(()),
        }

        valid_positive = valid_weight > 0.0
        fx = flow[:, 0:1]
        fy = flow[:, 1:2]
        x0 = torch.floor(fx)
        y0 = torch.floor(fy)
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        wx1 = (fx - x0).clamp(0.0, 1.0)
        wy1 = (fy - y0).clamp(0.0, 1.0)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        target_mass = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
        subpx_loss_map = torch.zeros_like(target_mass)
        pos_logit = torch.zeros_like(target_mass)
        target_mask = torch.zeros_like(corr, dtype=torch.bool)
        log_probs = None
        if compute_subpixel:
            log_probs = F.log_softmax(corr / max(float(temperature), 1e-6), dim=1)

        for yy, wy in ((y0, wy0), (y1, wy1)):
            for xx, wx in ((x0, wx0), (x1, wx1)):
                in_bounds = (
                    valid_positive
                    & (xx >= -radius)
                    & (xx <= radius)
                    & (yy >= -radius)
                    & (yy <= radius)
                )
                mass = (wx * wy) * in_bounds.float()
                idx = ((yy.long() + radius) * window + (xx.long() + radius)).clamp(
                    0,
                    expected_channels - 1,
                )
                if compute_subpixel:
                    subpx_loss_map = subpx_loss_map - mass * log_probs.gather(1, idx)
                if compute_peak:
                    pos_logit = pos_logit + mass * corr.gather(1, idx)
                    target_mask.scatter_(1, idx, target_mask.gather(1, idx) | in_bounds)
                target_mass = target_mass + mass

        in_window = valid_positive & (target_mass > 1e-6)
        pixel_weight = torch.where(in_window, valid_weight.clamp(min=0.0), torch.zeros_like(valid_weight))
        denom = pixel_weight.sum().clamp(min=1.0)

        nearest_dx = torch.round(fx).long()
        nearest_dy = torch.round(fy).long()
        nearest_target = ((nearest_dy + radius) * window + (nearest_dx + radius)).clamp(
            0,
            expected_channels - 1,
        )
        nearest_in_window = (
            valid_positive
            & (nearest_dx >= -radius)
            & (nearest_dx <= radius)
            & (nearest_dy >= -radius)
            & (nearest_dy <= radius)
        )
        nearest_weight = torch.where(
            nearest_in_window,
            valid_weight.clamp(min=0.0),
            torch.zeros_like(valid_weight),
        )
        nearest_denom = nearest_weight.sum().clamp(min=1.0)
        with torch.no_grad():
            nearest_pos = corr.gather(1, nearest_target)
            nearest_mask = torch.zeros_like(corr, dtype=torch.bool)
            nearest_mask.scatter_(1, nearest_target, nearest_in_window)
            nearest_hard_neg = corr.masked_fill(nearest_mask, -1e4).max(dim=1, keepdim=True).values
            nearest_gap = nearest_pos - nearest_hard_neg
            low_threshold = float(low_peak_gap_threshold)
            metrics.update(
                {
                    "map_query_corr_peak_gap_diag": (
                        nearest_gap * nearest_weight
                    ).sum()
                    / nearest_denom,
                    "map_query_corr_peak_acc_diag": (
                        ((nearest_gap > 0.0).float() * nearest_weight).sum()
                        / nearest_denom
                    ),
                    "map_query_corr_peak_low_frac": (
                        ((nearest_gap < low_threshold).float() * nearest_weight).sum()
                        / nearest_denom
                    ),
                    "map_query_corr_peak_low_threshold": corr.new_tensor(low_threshold),
                }
            )

        if compute_ce:
            ce_temp = float(ce_temperature if ce_temperature is not None else temperature)
            ce_logits = corr / max(ce_temp, 1e-6)
            ce_flat = F.cross_entropy(
                ce_logits.permute(0, 2, 3, 1).reshape(-1, expected_channels),
                nearest_target.reshape(-1),
                reduction="none",
            ).view(B, 1, H, W)
            ce_loss = (ce_flat * nearest_weight).sum() / nearest_denom
            losses["ce"] = ce_loss
            with torch.no_grad():
                pred = corr.argmax(dim=1, keepdim=True)
                ce_acc = ((pred == nearest_target).float() * nearest_weight).sum() / nearest_denom
                ce_cov = nearest_in_window.float().mean()
            metrics.update(
                {
                    "map_query_corr_ce_loss": ce_loss.detach(),
                    "map_query_corr_ce_acc": ce_acc.detach(),
                    "map_query_corr_ce_cov": ce_cov.detach(),
                }
            )

        use_explicit_flow = flow_head is not None and (
            compute_flow
            or compute_flow_cosine
            or compute_wls_pose
            or compute_pose_gain
            or float(flow_head_conf_weight) > 0.0
        )
        pred_flow = None
        explicit_confidence = None
        explicit_confidence_logits = None
        if use_explicit_flow:
            flow_pred = _call_corr_context_module(
                flow_head,
                corr,
                depth_tensor=depth,
                valid_tensor=valid_weight,
            )
            pred_flow = flow_pred["flow"].float()
            explicit_confidence = flow_pred.get("confidence")
            if explicit_confidence is not None:
                explicit_confidence = explicit_confidence.float()
            explicit_confidence_logits = flow_pred.get("confidence_logits")
            if explicit_confidence_logits is not None:
                explicit_confidence_logits = explicit_confidence_logits.float()
            if pred_flow.shape[-2:] != corr.shape[-2:]:
                pred_flow = F.interpolate(pred_flow, size=corr.shape[-2:], mode="bilinear", align_corners=False)
            if explicit_confidence is None:
                explicit_confidence = corr.new_ones(B, 1, H, W)
            elif explicit_confidence.shape[-2:] != corr.shape[-2:]:
                explicit_confidence = F.interpolate(
                    explicit_confidence,
                    size=corr.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if explicit_confidence_logits is not None and explicit_confidence_logits.shape[-2:] != corr.shape[-2:]:
                explicit_confidence_logits = F.interpolate(
                    explicit_confidence_logits,
                    size=corr.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            metrics["map_query_corr_flow_source_explicit"] = corr.new_tensor(1.0).detach()
        else:
            metrics["map_query_corr_flow_source_explicit"] = corr.new_tensor(0.0).detach()

        flow_decode = str(flow_decode_mode or "softargmax").lower()
        if flow_decode not in {"softargmax", "argmax", "argmax_st"}:
            raise ValueError(
                "flow_decode_mode must be one of {'softargmax', 'argmax', 'argmax_st'}, "
                f"got {flow_decode_mode!r}"
            )
        need_probs = (
            compute_subpixel
            or compute_flow
            or compute_flow_cosine
            or compute_wls_pose
            or compute_pose_gain
        ) and not use_explicit_flow
        probs = None
        if need_probs:
            dx, dy = _local_correlation_offsets(radius, device, dtype)
            probs = torch.softmax(corr / max(float(temperature), 1e-6), dim=1)
            soft_flow = torch.cat(
                [
                    (probs * dx).sum(dim=1, keepdim=True),
                    (probs * dy).sum(dim=1, keepdim=True),
                ],
                dim=1,
            )
            hard_idx = corr.argmax(dim=1, keepdim=True)
            hard_flow = torch.cat(
                [
                    dx.expand(B, -1, H, W).gather(1, hard_idx),
                    dy.expand(B, -1, H, W).gather(1, hard_idx),
                ],
                dim=1,
            )
            if flow_decode == "argmax":
                pred_flow = hard_flow
            elif flow_decode == "argmax_st":
                pred_flow = hard_flow + (soft_flow - soft_flow.detach())
            else:
                pred_flow = soft_flow
            with torch.no_grad():
                flow_metric_weight = valid_weight.clamp(min=0.0)
                flow_metric_denom = flow_metric_weight.sum().clamp(min=1.0)
                gt_flow_mag = torch.linalg.norm(flow.float(), dim=1, keepdim=True)
                pred_flow_mag = torch.linalg.norm(pred_flow.float(), dim=1, keepdim=True)
                hard_flow_mag = torch.linalg.norm(hard_flow.float(), dim=1, keepdim=True)
                hard_epe_map = torch.linalg.norm(hard_flow.float() - flow.float(), dim=1, keepdim=True)
                hard_cosine = (hard_flow.float() * flow.float()).sum(dim=1, keepdim=True) / (
                    hard_flow_mag * gt_flow_mag
                ).clamp(min=1e-6)
                metrics.update(
                    {
                        "map_query_corr_gt_flow_mag_px": (
                            gt_flow_mag * flow_metric_weight
                        ).sum()
                        / flow_metric_denom,
                        "map_query_corr_pred_flow_mag_px": (
                            pred_flow_mag * flow_metric_weight
                        ).sum()
                        / flow_metric_denom,
                        "map_query_corr_argmax_flow_mag_px": (
                            hard_flow_mag * flow_metric_weight
                        ).sum()
                        / flow_metric_denom,
                        "map_query_corr_argmax_flow_epe": (
                            hard_epe_map * flow_metric_weight
                        ).sum()
                        / flow_metric_denom,
                        "map_query_corr_argmax_flow_cosine": (
                            hard_cosine * flow_metric_weight
                        ).sum()
                        / flow_metric_denom,
                    }
                )
        elif pred_flow is not None and (compute_flow or compute_flow_cosine or compute_wls_pose or compute_pose_gain):
            with torch.no_grad():
                flow_metric_weight = valid_weight.clamp(min=0.0)
                flow_metric_denom = flow_metric_weight.sum().clamp(min=1.0)
                gt_flow_mag = torch.linalg.norm(flow.float(), dim=1, keepdim=True)
                pred_flow_mag = torch.linalg.norm(pred_flow.float(), dim=1, keepdim=True)
                metrics.update(
                    {
                        "map_query_corr_gt_flow_mag_px": (
                            gt_flow_mag * flow_metric_weight
                        ).sum()
                        / flow_metric_denom,
                        "map_query_corr_pred_flow_mag_px": (
                            pred_flow_mag * flow_metric_weight
                        ).sum()
                        / flow_metric_denom,
                    }
                )

        if pred_flow is not None and (compute_flow or compute_flow_cosine or compute_wls_pose or compute_pose_gain):
            cosine_weight = pixel_weight.clamp(min=0.0)
            pred_norm = torch.linalg.norm(pred_flow.float(), dim=1, keepdim=True)
            gt_norm = torch.linalg.norm(flow.float(), dim=1, keepdim=True)
            direction_valid = (gt_norm > 1e-6).float()
            cosine_weight = cosine_weight * direction_valid
            cosine_denom = cosine_weight.sum().clamp(min=1.0)
            flow_cosine_metric = (pred_flow.float() * flow.float()).sum(dim=1, keepdim=True) / (
                pred_norm * gt_norm
            ).clamp(min=1e-6)
            pred_unit_for_loss = pred_flow.float() / pred_norm.clamp(min=1.0)
            gt_unit_for_loss = flow.float() / gt_norm.clamp(min=1e-6)
            flow_cosine_for_loss = (pred_unit_for_loss * gt_unit_for_loss).sum(dim=1, keepdim=True)
            flow_cosine_loss = ((1.0 - flow_cosine_for_loss) * cosine_weight).sum() / cosine_denom
            if compute_flow_cosine:
                losses["flow_cosine"] = flow_cosine_loss
            with torch.no_grad():
                metrics["map_query_corr_flow_cosine"] = (
                    flow_cosine_metric * cosine_weight
                ).sum() / cosine_denom
                if compute_flow_cosine:
                    metrics["map_query_corr_flow_cosine_loss"] = flow_cosine_loss.detach()

        if use_explicit_flow and float(flow_head_conf_weight) > 0.0:
            conf_target = (pixel_weight > 0.0).float()
            conf_weight = valid_weight.clamp(min=0.0)
            if explicit_confidence_logits is not None:
                conf_loss_map = F.binary_cross_entropy_with_logits(
                    explicit_confidence_logits,
                    conf_target,
                    reduction="none",
                )
            else:
                conf_prob = explicit_confidence.clamp(min=1e-4, max=1.0 - 1e-4)
                conf_loss_map = F.binary_cross_entropy(conf_prob, conf_target, reduction="none")
            conf_denom = conf_weight.sum().clamp(min=1.0)
            flow_conf_loss = (conf_loss_map * conf_weight).sum() / conf_denom
            losses["flow_conf"] = flow_conf_loss
            with torch.no_grad():
                conf_pos = (explicit_confidence * pixel_weight).sum() / pixel_weight.sum().clamp(min=1.0)
                conf_all = (explicit_confidence * conf_weight).sum() / conf_denom
            metrics.update(
                {
                    "map_query_corr_flow_conf_loss": flow_conf_loss.detach(),
                    "map_query_corr_flow_conf_pos": conf_pos.detach(),
                    "map_query_corr_flow_conf_mean": conf_all.detach(),
                }
            )

        if compute_subpixel:
            loss_per_pixel = subpx_loss_map / target_mass.clamp(min=1e-6)
            subpx_loss = (loss_per_pixel * pixel_weight).sum() / denom
            losses["subpixel"] = subpx_loss
            with torch.no_grad():
                epe_map = torch.linalg.norm(pred_flow - flow, dim=1, keepdim=True)
                epe = (epe_map * pixel_weight).sum() / denom
                pred = corr.argmax(dim=1, keepdim=True)
                acc = ((pred == nearest_target) & in_window).float().sum() / in_window.float().sum().clamp(min=1.0)
                coverage = in_window.float().mean()
            metrics.update(
                {
                    "map_query_corr_subpx_loss": subpx_loss.detach(),
                    "map_query_corr_subpx_flow_epe": epe.detach(),
                    "map_query_corr_subpx_acc": acc.detach(),
                    "map_query_corr_subpx_cov": coverage.detach(),
                }
            )

        if compute_flow:
            diff = pred_flow - flow
            abs_diff = diff.abs()
            delta = max(float(huber_delta), 1e-6)
            loss_map = torch.where(abs_diff <= delta, 0.5 * diff.pow(2) / delta, abs_diff - 0.5 * delta)
            flow_loss = (loss_map * pixel_weight).sum() / (pixel_weight.sum() * 2.0).clamp(min=1.0)
            losses["flow"] = flow_loss
            with torch.no_grad():
                pixel_denom = pixel_weight.sum().clamp(min=1.0)
                epe_map = torch.linalg.norm(diff, dim=1, keepdim=True)
                epe = (epe_map * pixel_weight).sum() / pixel_denom
                coverage = in_window.float().mean()
            metrics.update(
                {
                    "map_query_corr_flow_loss": flow_loss.detach(),
                    "map_query_corr_flow_epe": epe.detach(),
                    "map_query_corr_flow_cov": coverage.detach(),
                }
            )

        if compute_peak:
            pos_logit = pos_logit / target_mass.clamp(min=1e-6)
            neg_logits = corr.masked_fill(target_mask, -1e4)
            hard_neg = neg_logits.max(dim=1, keepdim=True).values
            loss_map = F.relu(float(peak_margin) + hard_neg - pos_logit)
            peak_loss = (loss_map * pixel_weight).sum() / denom
            losses["peak"] = peak_loss
            with torch.no_grad():
                gap = ((pos_logit - hard_neg) * pixel_weight).sum() / denom
                acc = (((pos_logit > hard_neg).float() * pixel_weight).sum() / denom)
                pos_mean = (pos_logit * pixel_weight).sum() / denom
                neg_mean = (hard_neg * pixel_weight).sum() / denom
                coverage = in_window.float().mean()
            metrics.update(
                {
                    "map_query_corr_peak_loss": peak_loss.detach(),
                    "map_query_corr_peak_gap": gap.detach(),
                    "map_query_corr_peak_acc": acc.detach(),
                    "map_query_corr_peak_pos": pos_mean.detach(),
                    "map_query_corr_peak_neg": neg_mean.detach(),
                    "map_query_corr_peak_cov": coverage.detach(),
                }
            )

        if compute_wls_pose or compute_pose_gain:
            if depth is None or pose_ref is None or pose_gt is None or intrinsics is None:
                raise ValueError("pose update losses require depth, pose_ref, pose_gt, and intrinsics")
            if use_explicit_flow:
                confidence = explicit_confidence
            else:
                confidence = correlation_confidence_from_probs(
                    probs,
                    radius=radius,
                    mode=wls_conf_mode,
                    variance_scale=wls_conf_variance_scale,
                )
            conf_threshold = float(wls_conf_threshold)
            if conf_threshold > 0.0:
                confidence = confidence * (confidence >= conf_threshold).float()
            conf_power = float(wls_conf_power)
            if abs(conf_power - 1.0) > 1e-6:
                confidence = confidence.clamp(min=0.0, max=1.0).pow(conf_power)
            depth_s = depth.float()
            if depth_s.ndim == 4:
                depth_s = depth_s.squeeze(1)
            if depth_s.shape[-2:] != rendered.shape[-2:]:
                depth_s = F.interpolate(
                    depth_s.unsqueeze(1),
                    size=rendered.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            # WLS must consume only pixels whose teacher/rendered->query flow
            # lies inside the local search window.  The flow losses already use
            # ``pixel_weight`` for this reason; using the broader valid mask
            # feeds clipped soft flows from out-of-window correspondences into
            # the geometry solver and can systematically worsen the pose.
            valid_wls = pixel_weight
            if valid_wls.shape[-2:] != rendered.shape[-2:]:
                valid_wls = F.interpolate(valid_wls, size=rendered.shape[-2:], mode="nearest")
            valid_wls = valid_wls * (depth_s > 0.05).unsqueeze(1).float()
            min_conf_cov = float(wls_min_conf_cov)
            if min_conf_cov > 0.0:
                flat_valid = valid_wls.reshape(B, -1)
                flat_conf = confidence.reshape(B, -1)
                cov_per_sample = (
                    ((flat_conf > 0.0).float() * flat_valid).sum(dim=1)
                    / flat_valid.sum(dim=1).clamp(min=1.0)
                )
                keep_sample = (cov_per_sample >= min_conf_cov).to(confidence.dtype).view(B, 1, 1, 1)
                confidence = confidence * keep_sample

            Ju, Jv, depth_valid = compute_image_jacobian(depth_s, intrinsics)
            delta_xi = diff_pose_solve(
                pred_flow,
                (confidence * valid_wls).expand(-1, 2, -1, -1).contiguous(),
                Ju,
                Jv,
                depth_valid,
                damping=float(damping),
            )
            pose_pred = apply_pose_delta(pose_ref.float(), delta_xi.float(), scale=float(update_scale))
            rot_loss, rot_err_deg, trans_err_m = pose_error_tensors(pose_pred, pose_gt.float())
            init_rot_loss, init_rot_err_deg, init_trans_err_m = pose_error_tensors(
                pose_ref.float(),
                pose_gt.float(),
            )
            pose_gated, gated_metrics = _apply_wls_accept_gate(
                pose_pred,
                pose_ref.float(),
                delta_xi.float(),
                confidence,
                valid_wls,
                min_conf_mean=wls_accept_min_conf_mean,
                min_conf_cov=wls_accept_min_conf_cov,
                min_delta_mm=wls_accept_min_delta_mm,
                max_delta_mm=wls_accept_max_delta_mm,
            )
            _gated_rot_loss, gated_rot_err_deg, gated_trans_err_m = pose_error_tensors(
                pose_gated,
                pose_gt.float(),
            )
            wls_loss = corr.new_zeros(())
            if compute_wls_pose:
                wls_loss = float(rot_weight) * rot_loss.mean() + float(trans_weight) * trans_err_m.mean()
                losses["wls_pose"] = wls_loss
            if compute_pose_gain:
                pose_gain_loss, pose_gain_metrics = pose_update_gain_loss(
                    pose_pred,
                    pose_ref.float(),
                    pose_gt.float(),
                    trans_margin_m=pose_gain_trans_margin_m,
                    rot_margin_deg=pose_gain_rot_margin_deg,
                    rot_weight=pose_gain_rot_weight,
                    trans_weight=pose_gain_trans_weight,
                )
                losses["pose_gain"] = pose_gain_loss
                metrics.update(pose_gain_metrics)
            delta_trans_mm = torch.linalg.norm(delta_xi[:, :3].float(), dim=1).mean() * 1000.0
            conf_denom = valid_wls.sum().clamp(min=1.0)
            conf_mean = (confidence * valid_wls).sum() / conf_denom
            conf_cov = ((confidence > 0.0).float() * valid_wls).sum() / conf_denom
            metrics.update(
                {
                    "map_corr_wls_pose_loss": wls_loss.detach(),
                    "map_corr_wls_rot_err_deg": rot_err_deg.detach().mean(),
                    "map_corr_wls_trans_err_mm": (trans_err_m.detach() * 1000.0).mean(),
                    "map_corr_wls_init_rot_err_deg": init_rot_err_deg.detach().mean(),
                    "map_corr_wls_init_trans_err_mm": (init_trans_err_m.detach() * 1000.0).mean(),
                    "map_corr_wls_trans_gain_mm": ((init_trans_err_m - trans_err_m).detach() * 1000.0).mean(),
                    "map_corr_wls_delta_trans_mm": delta_trans_mm.detach(),
                    "map_corr_wls_conf_mean": conf_mean.detach(),
                    "map_corr_wls_conf_cov": conf_cov.detach(),
                    "map_corr_wls_gated_rot_err_deg": gated_rot_err_deg.detach().mean(),
                    "map_corr_wls_gated_trans_err_mm": (gated_trans_err_m.detach() * 1000.0).mean(),
                    "map_corr_wls_gated_trans_gain_mm": (
                        (init_trans_err_m - gated_trans_err_m).detach() * 1000.0
                    ).mean(),
                    **{key: value.detach() for key, value in gated_metrics.items()},
                }
            )

    return {"losses": losses, "metrics": metrics}


def local_correlation_wls_pose_loss(
    rendered_feat,
    query_feat,
    depth,
    pose_ref,
    pose_gt,
    intrinsics,
    valid_mask=None,
    *,
    radius=4,
    temperature=0.05,
    damping=1e-3,
    update_scale=1.0,
    rot_weight=1.0,
    trans_weight=50.0,
    wls_conf_mode="max",
    wls_conf_variance_scale=0.5,
):
    """Supervise pose after local-correlation soft flow and depth WLS."""
    with torch.cuda.amp.autocast(enabled=False):
        rendered = rendered_feat.float()
        query, _flow_unused, valid_weight = _resize_query_flow_valid(
            query_feat,
            torch.zeros(
                rendered.shape[0],
                2,
                rendered.shape[-2],
                rendered.shape[-1],
                device=rendered.device,
                dtype=rendered.dtype,
            ),
            valid_mask if valid_mask is not None else torch.ones(
                rendered.shape[0],
                1,
                rendered.shape[-2],
                rendered.shape[-1],
                device=rendered.device,
                dtype=rendered.dtype,
            ),
            rendered.shape[-2:],
        )
        rendered_n = F.normalize(rendered, dim=1)
        query_n = F.normalize(query, dim=1)
        corr = shifted_local_correlation(rendered_n, query_n, radius=int(radius)).float()
        dx, dy = _local_correlation_offsets(int(radius), corr.device, corr.dtype)
        probs = torch.softmax(corr / max(float(temperature), 1e-6), dim=1)
        flow = torch.cat(
            [
                (probs * dx).sum(dim=1, keepdim=True),
                (probs * dy).sum(dim=1, keepdim=True),
            ],
            dim=1,
        )
        confidence = correlation_confidence_from_probs(
            probs,
            radius=radius,
            mode=wls_conf_mode,
            variance_scale=wls_conf_variance_scale,
        )

        depth_s = depth.float()
        if depth_s.ndim == 4:
            depth_s = depth_s.squeeze(1)
        if depth_s.shape[-2:] != rendered.shape[-2:]:
            depth_s = F.interpolate(
                depth_s.unsqueeze(1),
                size=rendered.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        if valid_weight.shape[-2:] != rendered.shape[-2:]:
            valid_weight = F.interpolate(valid_weight, size=rendered.shape[-2:], mode="nearest")
        valid_weight = valid_weight * (depth_s > 0.05).unsqueeze(1).float()

        Ju, Jv, depth_valid = compute_image_jacobian(depth_s, intrinsics)
        delta_xi = diff_pose_solve(
            flow,
            (confidence * valid_weight).expand(-1, 2, -1, -1).contiguous(),
            Ju,
            Jv,
            depth_valid,
            damping=float(damping),
        )
        pose_pred = apply_pose_delta(pose_ref.float(), delta_xi.float(), scale=float(update_scale))
        rot_loss, rot_err_deg, trans_err_m = pose_error_tensors(pose_pred, pose_gt.float())
        init_rot_loss, init_rot_err_deg, init_trans_err_m = pose_error_tensors(
            pose_ref.float(),
            pose_gt.float(),
        )
        loss = float(rot_weight) * rot_loss.mean() + float(trans_weight) * trans_err_m.mean()
        delta_trans_mm = torch.linalg.norm(delta_xi[:, :3].float(), dim=1).mean() * 1000.0
        conf_denom = valid_weight.sum().clamp(min=1.0)
        conf_mean = (confidence * valid_weight).sum() / conf_denom

    return loss, {
        "map_corr_wls_pose_loss": loss.detach(),
        "map_corr_wls_rot_err_deg": rot_err_deg.detach().mean(),
        "map_corr_wls_trans_err_mm": (trans_err_m.detach() * 1000.0).mean(),
        "map_corr_wls_init_rot_err_deg": init_rot_err_deg.detach().mean(),
        "map_corr_wls_init_trans_err_mm": (init_trans_err_m.detach() * 1000.0).mean(),
        "map_corr_wls_trans_gain_mm": ((init_trans_err_m - trans_err_m).detach() * 1000.0).mean(),
        "map_corr_wls_delta_trans_mm": delta_trans_mm.detach(),
        "map_corr_wls_conf_mean": conf_mean.detach(),
    }


def sample_query_feature_by_flow(query_feat, flow_gt, target_hw, offset_xy=None):
    """Sample query features at rendered pixel locations displaced by rendered->query flow."""
    query = query_feat.float()
    flow = flow_gt.float()
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"flow_gt must have shape (B,2,H,W), got {tuple(flow.shape)}")
    B, _C, H, W = flow.shape
    if query.shape[-2:] != (H, W):
        src_h, src_w = query.shape[-2:]
        query = F.interpolate(query, size=(H, W), mode="bilinear", align_corners=False)
        flow = flow.clone()
        flow[:, 0] *= W / max(src_w, 1)
        flow[:, 1] *= H / max(src_h, 1)

    device = query.device
    dtype = query.dtype
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x.view(1, 1, H, W).expand(B, -1, -1, -1)
    y = y.view(1, 1, H, W).expand(B, -1, -1, -1)
    sample_x = x + flow[:, 0:1].to(device=device, dtype=dtype)
    sample_y = y + flow[:, 1:2].to(device=device, dtype=dtype)
    if offset_xy is not None:
        off_x, off_y = float(offset_xy[0]), float(offset_xy[1])
        sample_x = sample_x + off_x
        sample_y = sample_y + off_y
    in_bounds = (
        (sample_x >= 0.0)
        & (sample_x <= max(W - 1, 1))
        & (sample_y >= 0.0)
        & (sample_y <= max(H - 1, 1))
    ).float()
    norm_x = sample_x / max(W - 1, 1) * 2.0 - 1.0
    norm_y = sample_y / max(H - 1, 1) * 2.0 - 1.0
    grid = torch.cat([norm_x, norm_y], dim=1).permute(0, 2, 3, 1)
    sampled = F.grid_sample(
        query,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled, in_bounds


def _scale_xy_tensor(xy, source_hw, target_hw):
    xy = xy.float().clone()
    if torch.is_tensor(source_hw):
        src = source_hw.to(device=xy.device, dtype=xy.dtype)
        if src.ndim == 1:
            src = src.view(1, 1, 2).expand(xy.shape[0], -1, -1)
        elif src.ndim == 2:
            src = src.view(xy.shape[0], 1, 2)
        src_h = src[..., 0].clamp(min=1.0)
        src_w = src[..., 1].clamp(min=1.0)
    else:
        src_h = xy.new_tensor(float(source_hw[0])).view(1, 1)
        src_w = xy.new_tensor(float(source_hw[1])).view(1, 1)
    dst_h, dst_w = float(target_hw[0]), float(target_hw[1])
    xy[..., 0] = xy[..., 0] * ((dst_w - 1.0) / torch.clamp(src_w - 1.0, min=1.0))
    xy[..., 1] = xy[..., 1] * ((dst_h - 1.0) / torch.clamp(src_h - 1.0, min=1.0))
    return xy


def sample_feature_at_xy(feature, xy, valid=None, *, source_hw=None):
    """Sample ``feature`` at sparse xy points and return ``[B,N,C]`` features."""
    feat = feature.float()
    B, C, H, W = feat.shape
    xy = xy.to(device=feat.device, dtype=feat.dtype)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"xy must have shape (B,N,2), got {tuple(xy.shape)}")
    if source_hw is not None:
        xy = _scale_xy_tensor(xy, source_hw, (H, W))
    x = xy[..., 0]
    y = xy[..., 1]
    in_bounds = (x >= 0.0) & (x <= max(W - 1, 1)) & (y >= 0.0) & (y <= max(H - 1, 1))
    if valid is not None:
        in_bounds = in_bounds & (valid.to(device=feat.device) > 0)
    norm_x = x / max(W - 1, 1) * 2.0 - 1.0
    norm_y = y / max(H - 1, 1) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).view(B, 1, xy.shape[1], 2)
    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = sampled.squeeze(2).permute(0, 2, 1).contiguous()
    return sampled, in_bounds


def sparse_teacher_correspondence_loss(
    query_feat,
    map_feat,
    query_xy,
    map_xy,
    confidence,
    valid,
    *,
    xy_source_hw=None,
    temperature=0.07,
    min_confidence=0.0,
    min_points=4,
    negative_exclusion_px=0.0,
    positive_weight=0.0,
    margin_weight=0.0,
    margin=0.1,
):
    """Sparse symmetric InfoNCE over teacher query-map correspondences."""
    with torch.cuda.amp.autocast(enabled=False):
        q_feat = query_feat.float()
        m_feat = map_feat.float()
        query_xy = query_xy.to(device=q_feat.device).float()
        map_xy = map_xy.to(device=q_feat.device).float()
        confidence = confidence.to(device=q_feat.device).float()
        valid = valid.to(device=q_feat.device).float()
        if confidence.ndim == 3:
            confidence = confidence.squeeze(-1)
        if valid.ndim == 3:
            valid = valid.squeeze(-1)

        q_sparse, q_in = sample_feature_at_xy(q_feat, query_xy, valid, source_hw=xy_source_hw)
        m_sparse, m_in = sample_feature_at_xy(m_feat, map_xy, valid, source_hw=xy_source_hw)
        query_xy_feat = query_xy
        map_xy_feat = map_xy
        if xy_source_hw is not None:
            query_xy_feat = _scale_xy_tensor(query_xy, xy_source_hw, q_feat.shape[-2:])
            map_xy_feat = _scale_xy_tensor(map_xy, xy_source_hw, m_feat.shape[-2:])
        usable = (valid > 0) & q_in & m_in & torch.isfinite(confidence) & (confidence >= float(min_confidence))
        usable = usable & torch.isfinite(q_sparse).all(dim=-1) & torch.isfinite(m_sparse).all(dim=-1)

        losses = []
        acc_values = []
        pos_values = []
        neg_values = []
        cos_values = []
        point_counts = []
        temp = max(float(temperature), 1e-6)
        for batch_idx in range(q_sparse.shape[0]):
            mask_b = usable[batch_idx]
            if int(mask_b.sum().item()) < int(min_points):
                continue
            q_b = F.normalize(q_sparse[batch_idx, mask_b], dim=-1)
            m_b = F.normalize(m_sparse[batch_idx, mask_b], dim=-1)
            w_b = confidence[batch_idx, mask_b].clamp(min=0.0)
            if float(w_b.sum().item()) <= 0.0:
                w_b = torch.ones_like(w_b)
            logits = q_b @ m_b.t()
            eye = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
            allowed_neg = ~eye
            exclusion_px = max(float(negative_exclusion_px), 0.0)
            if exclusion_px > 0.0 and logits.shape[0] > 1:
                q_xy_b = query_xy_feat[batch_idx, mask_b].to(device=logits.device, dtype=logits.dtype)
                m_xy_b = map_xy_feat[batch_idx, mask_b].to(device=logits.device, dtype=logits.dtype)
                q_dist = torch.cdist(q_xy_b, q_xy_b, p=2)
                m_dist = torch.cdist(m_xy_b, m_xy_b, p=2)
                close = (q_dist <= exclusion_px) | (m_dist <= exclusion_px)
                allowed_neg = allowed_neg & ~close
            logits_qm = logits.masked_fill(~(allowed_neg | eye), -1e4)
            logits_mq = logits.t().masked_fill(~(allowed_neg.t() | eye), -1e4)
            targets = torch.arange(logits.shape[0], device=logits.device)
            ce_qm = F.cross_entropy(logits_qm / temp, targets, reduction="none")
            ce_mq = F.cross_entropy(logits_mq / temp, targets, reduction="none")
            per_point = (ce_qm + ce_mq) * 0.5
            pos = logits.diag()
            if positive_weight > 0.0:
                per_point = per_point + float(positive_weight) * (1.0 - pos)
            if margin_weight > 0.0 and allowed_neg.any():
                hard_for_margin = logits.masked_fill(~allowed_neg, -1e4).max(dim=1).values
                valid_margin = hard_for_margin > -1e3
                if valid_margin.any():
                    margin_loss = F.relu(hard_for_margin - pos + float(margin))
                    per_point = per_point + float(margin_weight) * margin_loss * valid_margin.float()
            loss_b = (per_point * w_b).sum() / w_b.sum().clamp(min=1e-6)
            losses.append(loss_b)

            with torch.no_grad():
                neg = logits.masked_fill(~allowed_neg, -1e4)
                hard_neg = neg.max(dim=1).values
                if (hard_neg <= -1e3).any():
                    fallback_neg = logits.masked_fill(eye, -1e4).max(dim=1).values
                    hard_neg = torch.where(hard_neg > -1e3, hard_neg, fallback_neg)
                acc_values.append(((logits_qm.argmax(dim=1) == targets).float() * w_b).sum() / w_b.sum().clamp(min=1e-6))
                pos_values.append((pos * w_b).sum() / w_b.sum().clamp(min=1e-6))
                neg_values.append((hard_neg * w_b).sum() / w_b.sum().clamp(min=1e-6))
                cos_values.append((pos * w_b).sum() / w_b.sum().clamp(min=1e-6))
                point_counts.append(mask_b.float().sum())

        if losses:
            loss = torch.stack(losses).mean()
            acc = torch.stack(acc_values).mean()
            pos = torch.stack(pos_values).mean()
            neg = torch.stack(neg_values).mean()
            cosine = torch.stack(cos_values).mean()
            points = torch.stack(point_counts).mean()
            skipped = q_feat.new_tensor(0.0)
        else:
            loss = q_feat.new_zeros(())
            acc = q_feat.new_zeros(())
            pos = q_feat.new_zeros(())
            neg = q_feat.new_zeros(())
            cosine = q_feat.new_zeros(())
            points = q_feat.new_zeros(())
            skipped = q_feat.new_tensor(1.0)
        coverage = usable.float().mean()
    return loss, {
        "map_teacher_corr_loss": loss.detach(),
        "map_teacher_corr_acc": acc.detach(),
        "map_teacher_corr_pos": pos.detach(),
        "map_teacher_corr_neg": neg.detach(),
        "map_teacher_corr_gap": (pos - neg).detach(),
        "map_teacher_corr_cosine": cosine.detach(),
        "map_teacher_corr_points": points.detach(),
        "map_teacher_corr_cov": coverage.detach(),
        "map_teacher_corr_skipped_no_points": skipped.detach(),
    }


def sparse_teacher_local_patch_loss(
    query_feat,
    map_feat,
    query_xy,
    map_xy,
    confidence,
    valid,
    *,
    xy_source_hw=None,
    radius=2,
    temperature=0.07,
    min_confidence=0.0,
    min_points=4,
    positive_weight=0.0,
    margin_weight=0.0,
    margin=0.05,
):
    """Cross-entropy over a local rendered patch centered at each teacher correspondence."""
    with torch.cuda.amp.autocast(enabled=False):
        q_feat = query_feat.float()
        m_feat = map_feat.float()
        query_xy = query_xy.to(device=q_feat.device).float()
        map_xy = map_xy.to(device=q_feat.device).float()
        confidence = confidence.to(device=q_feat.device).float()
        valid = valid.to(device=q_feat.device).float()
        if confidence.ndim == 3:
            confidence = confidence.squeeze(-1)
        if valid.ndim == 3:
            valid = valid.squeeze(-1)

        q_sparse, q_in = sample_feature_at_xy(q_feat, query_xy, valid, source_hw=xy_source_hw)
        _, m_in = sample_feature_at_xy(m_feat, map_xy, valid, source_hw=xy_source_hw)
        if xy_source_hw is not None:
            map_xy_feat = _scale_xy_tensor(map_xy, xy_source_hw, m_feat.shape[-2:])
        else:
            map_xy_feat = map_xy
        usable = (valid > 0) & q_in & m_in & torch.isfinite(confidence) & (confidence >= float(min_confidence))
        usable = usable & torch.isfinite(q_sparse).all(dim=-1) & torch.isfinite(map_xy_feat).all(dim=-1)

        r = max(int(radius), 0)
        offset_axis = torch.arange(-r, r + 1, device=q_feat.device, dtype=q_feat.dtype)
        dy, dx = torch.meshgrid(offset_axis, offset_axis, indexing="ij")
        offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)
        center_matches = (offsets[:, 0] == 0) & (offsets[:, 1] == 0)
        center_index = int(center_matches.nonzero(as_tuple=False)[0].item())

        losses = []
        acc_values = []
        pos_values = []
        neg_values = []
        gap_values = []
        soft_epe_values = []
        point_counts = []
        temp = max(float(temperature), 1e-6)
        for batch_idx in range(q_sparse.shape[0]):
            mask_b = usable[batch_idx]
            if int(mask_b.sum().item()) < int(min_points):
                continue
            q_b = q_sparse[batch_idx, mask_b]
            xy_b = map_xy_feat[batch_idx, mask_b].to(device=q_feat.device, dtype=q_feat.dtype)
            patch_xy = xy_b[:, None, :] + offsets[None, :, :]
            flat_xy = patch_xy.reshape(1, -1, 2)
            flat_valid = torch.ones(1, flat_xy.shape[1], device=q_feat.device, dtype=q_feat.dtype)
            patch_sparse, patch_in = sample_feature_at_xy(
                m_feat[batch_idx : batch_idx + 1],
                flat_xy,
                flat_valid,
                source_hw=None,
            )
            n_points = q_b.shape[0]
            n_offsets = offsets.shape[0]
            patch_sparse = patch_sparse.view(n_points, n_offsets, q_b.shape[-1])
            patch_in = patch_in.view(n_points, n_offsets)
            center_valid = patch_in[:, center_index]
            if int(center_valid.sum().item()) < int(min_points):
                continue
            q_b = F.normalize(q_b[center_valid], dim=-1)
            patch_sparse = F.normalize(patch_sparse[center_valid], dim=-1)
            patch_in = patch_in[center_valid]
            w_b = confidence[batch_idx, mask_b][center_valid].clamp(min=0.0)
            if float(w_b.sum().item()) <= 0.0:
                w_b = torch.ones_like(w_b)

            logits = (q_b[:, None, :] * patch_sparse).sum(dim=-1)
            logits = logits.masked_fill(~patch_in, -1e4)
            targets = torch.full((logits.shape[0],), center_index, device=logits.device, dtype=torch.long)
            per_point = F.cross_entropy(logits / temp, targets, reduction="none")
            pos = logits[:, center_index]
            other_mask = patch_in.clone()
            other_mask[:, center_index] = False
            hard_neg = logits.masked_fill(~other_mask, -1e4).max(dim=1).values
            has_neg = other_mask.any(dim=1)
            hard_neg = torch.where(has_neg, hard_neg, torch.zeros_like(hard_neg))
            if positive_weight > 0.0:
                per_point = per_point + float(positive_weight) * (1.0 - pos)
            if margin_weight > 0.0:
                margin_loss = F.relu(hard_neg - pos + float(margin))
                per_point = per_point + float(margin_weight) * margin_loss * has_neg.float()
            loss_b = (per_point * w_b).sum() / w_b.sum().clamp(min=1e-6)
            losses.append(loss_b)

            with torch.no_grad():
                acc_values.append(((logits.argmax(dim=1) == center_index).float() * w_b).sum() / w_b.sum().clamp(min=1e-6))
                pos_values.append((pos * w_b).sum() / w_b.sum().clamp(min=1e-6))
                neg_values.append((hard_neg * w_b).sum() / w_b.sum().clamp(min=1e-6))
                gap_values.append(((pos - hard_neg) * w_b).sum() / w_b.sum().clamp(min=1e-6))
                probs = F.softmax(logits / temp, dim=1)
                expected_offset = probs @ offsets.to(device=logits.device, dtype=logits.dtype)
                soft_epe = torch.linalg.vector_norm(expected_offset, dim=1)
                soft_epe_values.append((soft_epe * w_b).sum() / w_b.sum().clamp(min=1e-6))
                point_counts.append(center_valid.float().sum())

        if losses:
            loss = torch.stack(losses).mean()
            acc = torch.stack(acc_values).mean()
            pos = torch.stack(pos_values).mean()
            neg = torch.stack(neg_values).mean()
            gap = torch.stack(gap_values).mean()
            soft_epe = torch.stack(soft_epe_values).mean()
            points = torch.stack(point_counts).mean()
            skipped = q_feat.new_tensor(0.0)
        else:
            loss = q_feat.new_zeros(())
            acc = q_feat.new_zeros(())
            pos = q_feat.new_zeros(())
            neg = q_feat.new_zeros(())
            gap = q_feat.new_zeros(())
            soft_epe = q_feat.new_zeros(())
            points = q_feat.new_zeros(())
            skipped = q_feat.new_tensor(1.0)
        coverage = usable.float().mean()
    return loss, {
        "map_teacher_patch_loss": loss.detach(),
        "map_teacher_patch_acc": acc.detach(),
        "map_teacher_patch_pos": pos.detach(),
        "map_teacher_patch_neg": neg.detach(),
        "map_teacher_patch_gap": gap.detach(),
        "map_teacher_patch_soft_epe": soft_epe.detach(),
        "map_teacher_patch_points": points.detach(),
        "map_teacher_patch_cov": coverage.detach(),
        "map_teacher_patch_skipped_no_points": skipped.detach(),
    }


def flow_warp_feature_alignment_loss(rendered_feat, query_feat, flow_gt, valid_mask):
    """Align rendered-pose features with query features sampled at depth-derived correspondences."""
    with torch.cuda.amp.autocast(enabled=False):
        rendered = rendered_feat.float()
        query_warped, in_bounds = sample_query_feature_by_flow(query_feat, flow_gt, rendered.shape[-2:])
        valid = valid_mask.float()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != rendered.shape[-2:]:
            valid = F.interpolate(valid, size=rendered.shape[-2:], mode="nearest")
        valid = valid * in_bounds
        if query_warped.shape[-2:] != rendered.shape[-2:]:
            query_warped = F.interpolate(query_warped, size=rendered.shape[-2:], mode="bilinear", align_corners=False)
        rendered_n = F.normalize(rendered, dim=1)
        query_n = F.normalize(query_warped, dim=1)
        cos_map = (rendered_n * query_n).sum(dim=1, keepdim=True)
        l1_map = (rendered - query_warped).abs().mean(dim=1, keepdim=True)
        denom = valid.sum().clamp(min=1.0)
        cosine_term = ((1.0 - cos_map) * valid).sum() / denom
        l1_term = (l1_map * valid).sum() / denom
        loss = cosine_term + 0.25 * l1_term
        with torch.no_grad():
            coverage = (valid > 0).float().mean()
            cosine = (cos_map * valid).sum() / denom
    return loss, {
        "map_query_flow_warp_loss": loss.detach(),
        "map_query_flow_warp_cosine": cosine.detach(),
        "map_query_flow_warp_cov": coverage.detach(),
    }


def flow_warp_contrastive_loss(
    rendered_feat,
    query_feat,
    flow_gt,
    valid_mask,
    *,
    margin=0.1,
    offsets=((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)),
):
    """Make the exact depth-flow correspondence beat nearby subpixel hard negatives."""
    with torch.cuda.amp.autocast(enabled=False):
        rendered = rendered_feat.float()
        rendered_n = F.normalize(rendered, dim=1)
        pos_feat, pos_in_bounds = sample_query_feature_by_flow(query_feat, flow_gt, rendered.shape[-2:])
        pos_cos = (rendered_n * F.normalize(pos_feat, dim=1)).sum(dim=1, keepdim=True)

        valid = valid_mask.float()
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        if valid.shape[-2:] != rendered.shape[-2:]:
            valid = F.interpolate(valid, size=rendered.shape[-2:], mode="nearest")
        valid = valid * pos_in_bounds

        neg_cosines = []
        for offset in offsets:
            neg_feat, neg_in_bounds = sample_query_feature_by_flow(
                query_feat,
                flow_gt,
                rendered.shape[-2:],
                offset_xy=offset,
            )
            neg_cos = (rendered_n * F.normalize(neg_feat, dim=1)).sum(dim=1, keepdim=True)
            neg_cos = torch.where(neg_in_bounds > 0, neg_cos, torch.full_like(neg_cos, -1.0))
            neg_cosines.append(neg_cos)
        hard_neg = torch.stack(neg_cosines, dim=0).max(dim=0).values
        margin_t = float(margin)
        loss_map = F.relu(margin_t + hard_neg - pos_cos)
        denom = valid.sum().clamp(min=1.0)
        loss = (loss_map * valid).sum() / denom
        with torch.no_grad():
            gap = ((pos_cos - hard_neg) * valid).sum() / denom
            acc = (((pos_cos - hard_neg) > 0.0).float() * valid).sum() / denom
            coverage = (valid > 0).float().mean()
    return loss, {
        "map_query_flow_warp_contrastive_loss": loss.detach(),
        "map_query_flow_warp_hard_gap": gap.detach(),
        "map_query_flow_warp_hard_acc": acc.detach(),
        "map_query_flow_warp_contrastive_cov": coverage.detach(),
    }


def scene_coord_center_scale(map_cfg, device, dtype=torch.float32):
    center_cfg = map_cfg.get("scene_coord_center", [0.0, 0.0, 0.0])
    if not isinstance(center_cfg, (list, tuple)) or len(center_cfg) != 3:
        raise ValueError("map_supervision.scene_coord_center must be a 3-value list")
    center = torch.tensor(center_cfg, device=device, dtype=dtype).view(1, 3, 1, 1)
    scale = max(float(map_cfg.get("scene_coord_scale", 20.0)), 1e-6)
    return center, scale


def normalize_scene_coord_map(position, center, scale, target_hw=None):
    pos = position.float()
    if pos.ndim != 4:
        raise ValueError(f"scene position map must have shape (B,3,H,W) or (B,H,W,3), got {tuple(pos.shape)}")
    if pos.shape[1] != 3 and pos.shape[-1] == 3:
        pos = pos.permute(0, 3, 1, 2).contiguous()
    if pos.shape[1] != 3:
        raise ValueError(f"scene position map channel dimension must be 3, got {tuple(pos.shape)}")
    if target_hw is not None and pos.shape[-2:] != tuple(target_hw):
        pos = F.interpolate(pos, size=tuple(target_hw), mode="bilinear", align_corners=False)
    return (pos - center.to(device=pos.device, dtype=pos.dtype)) / float(scale)


def augment_feature_with_scene_coord(feature, scene_coord, weight):
    if scene_coord is None or float(weight) <= 0.0:
        return feature
    coord = scene_coord.float()
    if coord.shape[-2:] != feature.shape[-2:]:
        coord = F.interpolate(coord, size=feature.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([feature, coord * float(weight)], dim=1)


def _masked_huber_loss(diff, mask, beta=0.02):
    beta = max(float(beta), 1e-6)
    abs_diff = diff.abs()
    loss_map = torch.where(abs_diff < beta, 0.5 * abs_diff.square() / beta, abs_diff - 0.5 * beta)
    denom = (mask.sum() * diff.shape[1]).clamp(min=1.0)
    return (loss_map * mask).sum() / denom


def scene_coord_regression_loss(
    pred_scene_coord,
    target_position,
    valid_mask,
    center,
    scale,
    *,
    beta=0.02,
):
    """Supervise query-side normalized scene coordinates from rendered DCFF depth."""
    pred = pred_scene_coord.float()
    target = normalize_scene_coord_map(target_position, center, scale, target_hw=pred.shape[-2:])
    mask = valid_mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
    finite = torch.isfinite(target).all(dim=1, keepdim=True).float()
    mask = mask * finite
    diff = pred - target
    loss = _masked_huber_loss(diff, mask, beta=beta)
    with torch.no_grad():
        err_m = torch.linalg.norm(diff * float(scale), dim=1, keepdim=True)
        denom = mask.sum().clamp(min=1.0)
        err_cm = (err_m * mask).sum() / denom * 100.0
        coverage = (mask > 0).float().mean()
    return loss, {
        "map_query_scene_coord_loss": loss.detach(),
        "map_query_scene_coord_err_cm": err_cm.detach(),
        "map_query_scene_coord_cov": coverage.detach(),
    }


def scene_coord_flow_warp_loss(
    pred_scene_coord,
    target_position,
    flow_gt,
    valid_mask,
    center,
    scale,
    *,
    beta=0.02,
):
    """Make query scene coordinates agree with rendered 3D points at depth-derived correspondences."""
    target = normalize_scene_coord_map(target_position, center, scale)
    pred_warped, in_bounds = sample_query_feature_by_flow(
        pred_scene_coord.float(),
        flow_gt,
        target.shape[-2:],
    )
    mask = valid_mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != target.shape[-2:]:
        mask = F.interpolate(mask, size=target.shape[-2:], mode="nearest")
    finite = torch.isfinite(target).all(dim=1, keepdim=True).float()
    mask = mask * in_bounds * finite
    diff = pred_warped - target
    loss = _masked_huber_loss(diff, mask, beta=beta)
    with torch.no_grad():
        err_m = torch.linalg.norm(diff * float(scale), dim=1, keepdim=True)
        denom = mask.sum().clamp(min=1.0)
        err_cm = (err_m * mask).sum() / denom * 100.0
        coverage = (mask > 0).float().mean()
    return loss, {
        "map_query_scene_coord_warp_loss": loss.detach(),
        "map_query_scene_coord_warp_err_cm": err_cm.detach(),
        "map_query_scene_coord_warp_cov": coverage.detach(),
    }


def parse_xy_offsets(offsets_cfg):
    if not offsets_cfg:
        return None
    offsets = []
    for item in offsets_cfg:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "query_flow_warp_contrastive_offsets entries must be [dx, dy] pairs"
            )
        offsets.append((float(item[0]), float(item[1])))
    return tuple(offsets)


def prefix_metric_keys(metrics, old_prefix, new_prefix):
    result = {}
    for key, value in metrics.items():
        if key.startswith(old_prefix):
            result[f"{new_prefix}{key[len(old_prefix):]}"] = value
        else:
            result[key] = value
    return result


def feature_metric_pose_update_from_features(
    query_feat,
    rendered_feat,
    depth,
    pose_ref,
    intrinsics,
    valid_mask=None,
    *,
    damping=1e-3,
    normalize_features=True,
    update_scale=1.0,
    rot_damping_multiplier=1.0,
):
    """Apply one differentiable feature-metric pose update from rendered pose to query pose."""
    with torch.cuda.amp.autocast(enabled=False):
        query = query_feat.float()
        rendered = rendered_feat.float()
        if query.shape[-2:] != rendered.shape[-2:]:
            query = F.interpolate(query, rendered.shape[-2:], mode="bilinear", align_corners=False)
        if normalize_features:
            query = F.normalize(query, dim=1)
            rendered = F.normalize(rendered, dim=1)

        depth_s = depth.float()
        if depth_s.ndim == 4:
            depth_s = depth_s.squeeze(1)
        if depth_s.shape[-2:] != rendered.shape[-2:]:
            depth_s = F.interpolate(
                depth_s.unsqueeze(1),
                size=rendered.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        if valid_mask is None:
            valid_mask = (depth_s > 0.05).unsqueeze(1).float()
        elif valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1).float()
        else:
            valid_mask = valid_mask.float()
        if valid_mask.shape[-2:] != rendered.shape[-2:]:
            valid_mask = F.interpolate(valid_mask, rendered.shape[-2:], mode="nearest")

        delta_xi, residual = feature_metric_solve(
            query,
            rendered,
            depth_s,
            intrinsics,
            damping=float(damping),
            valid_mask=valid_mask,
            rot_damping_multiplier=float(rot_damping_multiplier),
        )
        pose_pred = apply_pose_delta(pose_ref.float(), delta_xi.float(), scale=float(update_scale))
        return delta_xi, pose_pred, residual


def feature_metric_localization_loss(
    query_feat,
    rendered_feat,
    depth,
    pose_ref,
    pose_gt,
    intrinsics,
    valid_mask=None,
    *,
    damping=1e-3,
    normalize_features=True,
    update_scale=1.0,
    rot_weight=1.0,
    trans_weight=50.0,
    rot_damping_multiplier=1.0,
):
    """Train features by supervising the pose after one feature-metric GN/WLS step."""
    delta_xi, pose_pred, residual = feature_metric_pose_update_from_features(
        query_feat,
        rendered_feat,
        depth,
        pose_ref,
        intrinsics,
        valid_mask=valid_mask,
        damping=damping,
        normalize_features=normalize_features,
        update_scale=update_scale,
        rot_damping_multiplier=rot_damping_multiplier,
    )
    rot_loss, rot_err_deg, trans_err_m = pose_error_tensors(pose_pred, pose_gt.float())
    init_rot_loss, init_rot_err_deg, init_trans_err_m = pose_error_tensors(pose_ref.float(), pose_gt.float())
    loss = float(rot_weight) * rot_loss.mean() + float(trans_weight) * trans_err_m.mean()
    delta_trans_mm = torch.linalg.norm(delta_xi[:, :3].float(), dim=1).mean() * 1000.0
    return loss, {
        "map_feature_metric_pose_loss": loss.detach(),
        "map_feature_metric_rot_err_deg": rot_err_deg.detach().mean(),
        "map_feature_metric_trans_err_mm": (trans_err_m.detach() * 1000.0).mean(),
        "map_feature_metric_init_rot_err_deg": init_rot_err_deg.detach().mean(),
        "map_feature_metric_init_trans_err_mm": (init_trans_err_m.detach() * 1000.0).mean(),
        "map_feature_metric_trans_gain_mm": ((init_trans_err_m - trans_err_m).detach() * 1000.0).mean(),
        "map_feature_metric_delta_trans_mm": delta_trans_mm.detach(),
        "map_feature_metric_residual_l1": residual.detach().float().abs().mean(),
    }


def feature_metric_gradient_direction_loss(
    query_feat,
    rendered_feat,
    depth,
    pose_current,
    pose_target,
    intrinsics,
    valid_mask=None,
    *,
    rot_weight=1.0,
    trans_weight=50.0,
):
    """Train features so J^T r points toward the true pose correction.

    Instead of backpropagating through the full GN solve (noisy matrix inverse),
    directly optimize the 6-DoF gradient direction J^T r to align with the
    true twist ξ_gt = log(T_target * T_current^{-1}).

    This is a necessary condition for FM-GN success: if the gradient points
    toward GT, then the GN step (which is (J^T J+λI)^{-1} J^T r) will also
    move in approximately the right direction.

    Args:
        query_feat:   (B, C, H, W) student features at GT pose
        rendered_feat:(B, C, H, W) DCFF features at perturbed pose
        depth:        (B, H, W) depth at perturbed pose
        pose_current: (B, 4, 4) perturbed camera-to-world pose
        pose_target:  (B, 4, 4) GT camera-to-world pose
        intrinsics:   {fx, fy, cx, cy}
        valid_mask:   (B, 1, H, W) optional validity mask
        rot_weight:   weight for rotation components in cosine
        trans_weight: weight for translation components in cosine

    Returns:
        loss: scalar, 1 - weighted_cos(Jtr, ξ_gt)
        metrics: dict with diagnostic values
    """
    with torch.cuda.amp.autocast(enabled=False):
        query = query_feat.float()
        rendered = rendered_feat.float()
        if query.shape[-2:] != rendered.shape[-2:]:
            query = F.interpolate(query, rendered.shape[-2:], mode="bilinear", align_corners=False)
        query = F.normalize(query, dim=1)
        rendered = F.normalize(rendered, dim=1)

        # Normalize depth shape to (B, H, W)
        depth_s = depth.float()
        if depth_s.ndim == 4:
            depth_s = depth_s.squeeze(1)

        # Compute Jtr = -J^T r via feature_metric_solve (ignore delta_xi, residual)
        _, _, Jtr = feature_metric_solve(
            query, rendered, depth_s, intrinsics,
            damping=1e-3, valid_mask=valid_mask, return_Jtr=True,
        )

        # True twist: ξ_gt = log(T_target * T_current^{-1})
        T_delta = torch.bmm(pose_target.float(), torch.inverse(pose_current.float()))
        xi_gt = se3_log(T_delta)  # (B, 6)

        # Separate rotation/translation cosine to avoid scale imbalance.
        # Image Jacobian naturally gives rotation components ~Z× larger than
        # translation, so joint cosine would be insensitive to translation direction.
        Jtr_trans = Jtr[:, :3]
        Jtr_rot = Jtr[:, 3:]
        xi_trans = xi_gt[:, :3]
        xi_rot = xi_gt[:, 3:]

        cos_trans = F.cosine_similarity(Jtr_trans, xi_trans, dim=1, eps=1e-8)
        cos_rot = F.cosine_similarity(Jtr_rot, xi_rot, dim=1, eps=1e-8)

        loss = (
            float(trans_weight) * (1.0 - cos_trans).mean()
            + float(rot_weight) * (1.0 - cos_rot).mean()
        )

        return loss, {
            "map_grad_dir_loss": loss.detach(),
            "map_grad_dir_cos_trans": cos_trans.detach().mean(),
            "map_grad_dir_cos_rot": cos_rot.detach().mean(),
            "map_grad_dir_cos": ((cos_trans + cos_rot) / 2).detach().mean(),
            "map_grad_dir_jtr_norm": Jtr.norm(dim=1).detach().mean(),
        }


def compute_main_losses(outputs, batch, cfg):
    loss_cfg = cfg["loss"]
    teacher_fine = batch["teacher_fine"]
    teacher_coarse = batch["teacher_coarse"]
    pred_fine = outputs["fine"]
    pred_coarse = outputs["coarse"]
    if pred_fine.shape[-2:] != teacher_fine.shape[-2:]:
        pred_fine = F.interpolate(pred_fine, teacher_fine.shape[-2:], mode="bilinear", align_corners=False)
    if pred_coarse.shape[-2:] != teacher_coarse.shape[-2:]:
        pred_coarse = F.interpolate(pred_coarse, teacher_coarse.shape[-2:], mode="bilinear", align_corners=False)
    if pred_fine.shape[1] != teacher_fine.shape[1]:
        raise ValueError(f"Student fine dim {pred_fine.shape[1]} does not match teacher fine dim {teacher_fine.shape[1]}")
    if pred_coarse.shape[1] != teacher_coarse.shape[1]:
        raise ValueError(
            f"Student coarse dim {pred_coarse.shape[1]} does not match teacher coarse dim {teacher_coarse.shape[1]}"
        )

    fine_l1 = l1_feature_loss(pred_fine, teacher_fine)
    fine_cos = cosine_loss(pred_fine, teacher_fine)
    fine_cs = channel_standardized_loss(pred_fine, teacher_fine)
    coarse_l1 = l1_feature_loss(pred_coarse, teacher_coarse)
    coarse_cos = cosine_loss(pred_coarse, teacher_coarse)
    coarse_cs = channel_standardized_loss(pred_coarse, teacher_coarse)
    fine_grad = feature_gradient_loss(pred_fine, teacher_fine)
    coarse_grad = feature_gradient_loss(pred_coarse, teacher_coarse)

    total = (
        loss_cfg["fine_l1_weight"] * fine_l1
        + loss_cfg["fine_cos_weight"] * fine_cos
        + float(loss_cfg.get("fine_channel_std_weight", 0.0)) * fine_cs
        + float(loss_cfg.get("fine_grad_weight", 0.0)) * fine_grad
        + loss_cfg["coarse_l1_weight"] * coarse_l1
        + loss_cfg["coarse_cos_weight"] * coarse_cos
        + float(loss_cfg.get("coarse_channel_std_weight", 0.0)) * coarse_cs
        + float(loss_cfg.get("coarse_grad_weight", 0.0)) * coarse_grad
    )
    fine_coarse_ortho = feature_orthogonality_loss(pred_fine, pred_coarse)
    total = total + float(loss_cfg.get("fine_coarse_ortho_weight", 0.0)) * fine_coarse_ortho

    metrics = {
        "loss_total": total.detach(),
        "fine_l1": fine_l1.detach(),
        "fine_cos_loss": fine_cos.detach(),
        "fine_channel_std_loss": fine_cs.detach(),
        "fine_grad_loss": fine_grad.detach(),
        "coarse_l1": coarse_l1.detach(),
        "coarse_cos_loss": coarse_cos.detach(),
        "coarse_channel_std_loss": coarse_cs.detach(),
        "coarse_grad_loss": coarse_grad.detach(),
        "fine_coarse_ortho_loss": fine_coarse_ortho.detach(),
        "fine_cosine": (1.0 - fine_cos).detach(),
        "coarse_cosine": (1.0 - coarse_cos).detach(),
    }

    teacher_norm_weight = float(loss_cfg.get("teacher_norm_weight", 0.0))
    if teacher_norm_weight > 0:
        if "magnitude" not in outputs:
            raise RuntimeError(
                "loss.teacher_norm_weight requires model.predict_magnitude=true so the student can vary output norms."
            )

        pred_fine_norm = torch.linalg.vector_norm(pred_fine.float(), dim=1, keepdim=True).clamp_min(1e-6)
        pred_coarse_norm = torch.linalg.vector_norm(pred_coarse.float(), dim=1, keepdim=True).clamp_min(1e-6)
        teacher_fine_norm = torch.linalg.vector_norm(teacher_fine.float(), dim=1, keepdim=True).clamp_min(1e-6)
        teacher_coarse_norm = torch.linalg.vector_norm(teacher_coarse.float(), dim=1, keepdim=True).clamp_min(1e-6)

        fine_norm_loss = F.smooth_l1_loss(torch.log(pred_fine_norm), torch.log(teacher_fine_norm))
        coarse_norm_loss = F.smooth_l1_loss(torch.log(pred_coarse_norm), torch.log(teacher_coarse_norm))
        norm_total = 0.5 * (fine_norm_loss + coarse_norm_loss)
        total = total + teacher_norm_weight * norm_total
        metrics.update({
            "teacher_norm_loss": norm_total.detach(),
            "fine_log_norm_loss": fine_norm_loss.detach(),
            "coarse_log_norm_loss": coarse_norm_loss.detach(),
            "pred_fine_norm_mean": pred_fine_norm.mean().detach(),
            "pred_coarse_norm_mean": pred_coarse_norm.mean().detach(),
            "teacher_fine_norm_mean": teacher_fine_norm.mean().detach(),
            "teacher_coarse_norm_mean": teacher_coarse_norm.mean().detach(),
        })

    infonce_weight = float(loss_cfg.get("query_teacher_infonce_weight", 0.0))
    if infonce_weight > 0:
        fine_nce = infonce_contrastive_loss(
            pred_fine,
            teacher_fine,
            temperature=float(loss_cfg.get("infonce_temperature", 0.07)),
            n_samples=int(loss_cfg.get("infonce_samples", 256)),
            cross_batch=bool(loss_cfg.get("infonce_cross_batch", False)),
        )
        coarse_nce = infonce_contrastive_loss(
            pred_coarse,
            teacher_coarse,
            temperature=float(loss_cfg.get("infonce_temperature", 0.07)),
            n_samples=int(loss_cfg.get("infonce_samples", 256)),
            cross_batch=bool(loss_cfg.get("infonce_cross_batch", False)),
        )
        nce_total = 0.5 * (fine_nce + coarse_nce)
        total = total + infonce_weight * nce_total
        metrics["query_teacher_nce"] = nce_total.detach()

    retrieval_cfg = cfg.get("retrieval", {})
    if retrieval_cfg.get("enabled", False) and "teacher_retrieval" in batch:
        pred_retrieval = outputs.get("retrieval")
        if pred_retrieval is None:
            raise RuntimeError(
                "Retrieval supervision is enabled, but the model did not return a retrieval descriptor."
            )

        teacher_retrieval = batch["teacher_retrieval"]
        pred_norm = F.normalize(pred_retrieval, dim=1)
        teacher_norm = F.normalize(teacher_retrieval, dim=1)
        retrieval_l1 = F.l1_loss(pred_retrieval, teacher_retrieval)
        retrieval_cos = 1.0 - torch.sum(pred_norm * teacher_norm, dim=1).mean()
        total = (
            total
            + float(retrieval_cfg.get("l1_weight", 0.0)) * retrieval_l1
            + float(retrieval_cfg.get("cos_weight", 0.0)) * retrieval_cos
        )
        metrics["retrieval_l1"] = retrieval_l1.detach()
        metrics["retrieval_cos_loss"] = retrieval_cos.detach()
        metrics["retrieval_cosine"] = (1.0 - retrieval_cos).detach()

        retrieval_nce_weight = float(retrieval_cfg.get("infonce_weight", 0.0))
        if retrieval_nce_weight > 0:
            temperature = float(retrieval_cfg.get("temperature", 0.07))
            logits_qt = torch.matmul(pred_norm, teacher_norm.t()) / temperature
            logits_tq = torch.matmul(teacher_norm, pred_norm.t()) / temperature
            targets = torch.arange(logits_qt.shape[0], device=logits_qt.device)
            retrieval_nce = 0.5 * (
                F.cross_entropy(logits_qt, targets) + F.cross_entropy(logits_tq, targets)
            )
            total = total + retrieval_nce_weight * retrieval_nce
            metrics["retrieval_nce"] = retrieval_nce.detach()

        similarity_weight = float(retrieval_cfg.get("similarity_weight", 0.0))
        if similarity_weight > 0 and pred_norm.shape[0] > 1:
            student_sim = torch.matmul(pred_norm, pred_norm.t())
            teacher_sim = torch.matmul(teacher_norm, teacher_norm.t())
            mask = ~torch.eye(student_sim.shape[0], dtype=torch.bool, device=student_sim.device)
            similarity_loss = F.mse_loss(student_sim[mask], teacher_sim[mask])
            total = total + similarity_weight * similarity_loss
            metrics["retrieval_similarity_loss"] = similarity_loss.detach()

    pose_init_total, pose_init_metrics = compute_pose_init_losses(outputs, batch, cfg)
    if pose_init_metrics:
        total = total + pose_init_total
        metrics.update(pose_init_metrics)

    return total, metrics


def compute_map_supervision(
    batch,
    outputs,
    cfg,
    device,
    epoch=0,
    local_matcher=None,
    local_flow_head=None,
    local_corr_projector=None,
    candidate_score_fusion_head=None,
    map_renderer=None,
):
    map_cfg = cfg.get("map_supervision", {})
    loss_cfg = cfg.get("loss", {})
    zero = torch.zeros((), device=device)
    if not map_cfg.get("enabled", False):
        return zero, {"map_hook_active": zero, "map_hook_loss": zero}

    required = ["rendered_map_fine", "rendered_map_coarse", "rendered_map_mask"]
    if not all(key in batch for key in required):
        return zero, {"map_hook_active": zero, "map_hook_loss": zero}

    rendered_mask = batch["rendered_map_mask"]
    prior_mask = batch.get("prior_mask")
    if prior_mask is not None:
        prior_mask = prior_mask.float()
        if prior_mask.shape[-2:] != rendered_mask.shape[-2:]:
            prior_mask = F.interpolate(prior_mask, size=rendered_mask.shape[-2:], mode="nearest")
        mask = rendered_mask * prior_mask
    else:
        mask = rendered_mask
    alpha = batch.get("rendered_map_alpha")
    rendered_rgb = batch.get("rendered_map_rgb")
    rendered_depth = batch.get("rendered_map_depth")
    rendered_position = batch.get("rendered_map_position")
    rendered_intrinsics = batch.get("rendered_map_intrinsics")
    rendered_fine_raw = batch.get("rendered_map_fine_raw")
    rendered_fine = batch["rendered_map_fine"]
    rendered_coarse = batch["rendered_map_coarse"]
    teacher_fine = batch["teacher_fine"]
    teacher_coarse = batch["teacher_coarse"]
    query_fine_key = str(map_cfg.get("query_fine_key", "fine"))
    if query_fine_key not in outputs:
        raise KeyError(f"map_supervision.query_fine_key={query_fine_key!r} not found in model outputs")
    query_local_fine_key = str(map_cfg.get("query_local_fine_key") or query_fine_key)
    if query_local_fine_key not in outputs:
        raise KeyError(
            f"map_supervision.query_local_fine_key={query_local_fine_key!r} not found in model outputs"
        )
    pred_fine = outputs[query_fine_key]
    pred_local_fine = outputs[query_local_fine_key]
    pred_coarse = outputs["coarse"]
    pred_scene_coord = outputs.get("scene_coord")

    def _resize_feature(feat, spatial_hw):
        if feat is None or feat.shape[-2:] == tuple(spatial_hw):
            return feat
        return F.interpolate(feat, size=spatial_hw, mode="bilinear", align_corners=False)

    def _resize_mask(feat_mask, spatial_hw):
        if feat_mask is None or feat_mask.shape[-2:] == tuple(spatial_hw):
            return feat_mask
        return F.interpolate(feat_mask.float(), size=spatial_hw, mode="nearest")

    def _resize_feature_bank(feat, spatial_hw):
        if feat is None or feat.shape[-2:] == tuple(spatial_hw):
            return feat
        if feat.ndim == 4:
            return _resize_feature(feat, spatial_hw)
        if feat.ndim != 5:
            raise ValueError(f"Expected feature bank to be 4D or 5D, got shape={tuple(feat.shape)}")
        bsz, count, channels, _height, _width = feat.shape
        flat = feat.reshape(bsz * count, channels, feat.shape[-2], feat.shape[-1])
        resized = _resize_feature(flat, spatial_hw)
        return resized.reshape(bsz, count, channels, int(spatial_hw[0]), int(spatial_hw[1]))

    def _resize_mask_bank(feat_mask, spatial_hw):
        if feat_mask is None or feat_mask.shape[-2:] == tuple(spatial_hw):
            return feat_mask
        if feat_mask.ndim == 4:
            return _resize_mask(feat_mask, spatial_hw)
        if feat_mask.ndim != 5:
            raise ValueError(f"Expected mask bank to be 4D or 5D, got shape={tuple(feat_mask.shape)}")
        bsz, count = feat_mask.shape[:2]
        channels = 1 if feat_mask.ndim == 4 else feat_mask.shape[2]
        flat = feat_mask.reshape(bsz * count, channels, feat_mask.shape[-2], feat_mask.shape[-1])
        resized = _resize_mask(flat, spatial_hw)
        return resized.reshape(bsz, count, channels, int(spatial_hw[0]), int(spatial_hw[1]))

    detach_query_features = bool(map_cfg.get("detach_query_features", False))
    pred_fine_target = pred_fine.detach() if detach_query_features else pred_fine
    pred_local_fine_target = pred_local_fine.detach() if detach_query_features else pred_local_fine
    pred_coarse_target = pred_coarse.detach() if detach_query_features else pred_coarse
    coarse_active = int(epoch) >= int(map_cfg.get("coarse_start_epoch", 0))
    query_coarse_weight = float(map_cfg.get("query_coarse_weight", 0.0)) if coarse_active else 0.0
    coarse_pose_rank_weight = resolve_linear_weight(map_cfg, "coarse_pose_rank_weight", epoch) if coarse_active else 0.0
    coarse_pose_rank_temperature = float(map_cfg.get("coarse_pose_rank_temperature", 0.07))
    coarse_pose_energy_weight = (
        resolve_linear_weight(map_cfg, "coarse_pose_energy_weight", epoch) if coarse_active else 0.0
    )
    coarse_pose_energy_temperature = float(map_cfg.get("coarse_pose_energy_temperature", 0.07))
    coarse_pose_local_energy_weight = (
        resolve_linear_weight(map_cfg, "coarse_pose_local_energy_weight", epoch) if coarse_active else 0.0
    )
    coarse_pose_local_energy_temperature = float(map_cfg.get("coarse_pose_local_energy_temperature", 0.07))
    coarse_pose_local_energy_radius = int(map_cfg.get("coarse_pose_local_energy_radius", 4))
    coarse_pose_local_energy_preprocess = str(map_cfg.get("coarse_pose_local_energy_preprocess", "none"))
    coarse_pose_local_energy_highpass_kernel = int(map_cfg.get("coarse_pose_local_energy_highpass_kernel", 5))
    candidate_render_score_weight = resolve_linear_weight(map_cfg, "candidate_render_score_weight", epoch)
    candidate_render_score_feature = str(map_cfg.get("candidate_render_score_feature", "coarse")).lower()
    candidate_render_score_mode = str(map_cfg.get("candidate_render_score_mode", "local"))
    candidate_render_score_temperature = float(map_cfg.get("candidate_render_score_temperature", 0.07))
    candidate_render_score_radius = int(map_cfg.get("candidate_render_score_radius", 4))
    candidate_render_score_preprocess = str(map_cfg.get("candidate_render_score_preprocess", "none"))
    candidate_render_score_highpass_kernel = int(map_cfg.get("candidate_render_score_highpass_kernel", 5))
    candidate_render_score_rot_cost_weight = float(map_cfg.get("candidate_render_score_rot_cost_weight", 0.1))
    candidate_score_fusion_weight = resolve_linear_weight(map_cfg, "candidate_score_fusion_weight", epoch)
    candidate_score_fusion_feature = str(
        map_cfg.get("candidate_score_fusion_feature", candidate_render_score_feature)
    ).lower()
    candidate_score_fusion_mode = str(map_cfg.get("candidate_score_fusion_mode", candidate_render_score_mode))
    candidate_score_fusion_temperature = float(
        map_cfg.get("candidate_score_fusion_temperature", candidate_render_score_temperature)
    )
    candidate_score_fusion_radius = int(map_cfg.get("candidate_score_fusion_radius", candidate_render_score_radius))
    candidate_score_fusion_preprocess = str(
        map_cfg.get("candidate_score_fusion_preprocess", candidate_render_score_preprocess)
    )
    candidate_score_fusion_highpass_kernel = int(
        map_cfg.get("candidate_score_fusion_highpass_kernel", candidate_render_score_highpass_kernel)
    )
    candidate_score_fusion_rot_cost_weight = float(
        map_cfg.get("candidate_score_fusion_rot_cost_weight", candidate_render_score_rot_cost_weight)
    )
    candidate_score_fusion_target_mode = str(map_cfg.get("candidate_score_fusion_target_mode", "hard"))
    candidate_score_fusion_target_temperature_m = float(
        map_cfg.get("candidate_score_fusion_target_temperature_m", 0.25)
    )
    candidate_score_fusion_render_feature_mode = str(
        map_cfg.get("candidate_score_fusion_render_feature_mode", "basic")
    )
    candidate_score_fusion_score_map_mode = str(
        map_cfg.get("candidate_score_fusion_score_map_mode", "peak_offset")
    )
    candidate_score_fusion_wls_radius = map_cfg.get("candidate_score_fusion_wls_radius")
    candidate_score_fusion_wls_temperature = map_cfg.get("candidate_score_fusion_wls_temperature")
    candidate_score_fusion_wls_damping = float(map_cfg.get("candidate_score_fusion_wls_damping", 1e-3))
    candidate_score_fusion_wls_update_scale = float(map_cfg.get("candidate_score_fusion_wls_update_scale", 1.0))
    candidate_score_fusion_wls_conf_mode = str(map_cfg.get("candidate_score_fusion_wls_conf_mode", "max"))
    candidate_score_fusion_wls_conf_variance_scale = float(
        map_cfg.get("candidate_score_fusion_wls_conf_variance_scale", 0.5)
    )
    candidate_score_fusion_wls_downsample = int(map_cfg.get("candidate_score_fusion_wls_downsample", 1) or 1)
    candidate_score_fusion_cost_regression_weight = float(
        map_cfg.get("candidate_score_fusion_cost_regression_weight", 0.0)
    )
    candidate_score_fusion_cost_regression_temperature_m = map_cfg.get(
        "candidate_score_fusion_cost_regression_temperature_m"
    )
    candidate_score_fusion_pairwise_rank_weight = float(
        map_cfg.get("candidate_score_fusion_pairwise_rank_weight", 0.0)
    )
    candidate_score_fusion_pairwise_rank_temperature = float(
        map_cfg.get("candidate_score_fusion_pairwise_rank_temperature", 1.0)
    )
    candidate_score_fusion_pairwise_rank_min_gap_m = float(
        map_cfg.get("candidate_score_fusion_pairwise_rank_min_gap_m", 0.0)
    )
    candidate_two_stage_enabled = bool(map_cfg.get("candidate_two_stage_enabled", False))
    candidate_stage2_topm = int(map_cfg.get("candidate_stage2_topm", 1) or 1)
    candidate_stage2_selection = str(map_cfg.get("candidate_stage2_selection", "pred"))
    candidate_stage2_train_map = bool(map_cfg.get("candidate_stage2_train_map", False))
    candidate_stage2_detach_selection = bool(map_cfg.get("candidate_stage2_detach_selection", True))
    candidate_stage2_prefix = str(map_cfg.get("candidate_stage2_prefix", "rendered_map_candidate_refine"))
    candidate_refined_pose_weight = resolve_linear_weight(map_cfg, "candidate_refined_pose_weight", epoch)
    candidate_refined_pose_feature = str(
        map_cfg.get("candidate_refined_pose_feature") or candidate_score_fusion_feature
    ).lower()
    candidate_refined_pose_radius = map_cfg.get("candidate_refined_pose_radius")
    candidate_refined_pose_temperature = map_cfg.get("candidate_refined_pose_temperature")
    candidate_refined_pose_damping = map_cfg.get("candidate_refined_pose_damping")
    candidate_refined_pose_update_scale = map_cfg.get("candidate_refined_pose_update_scale")
    candidate_refined_pose_rot_cost_weight = map_cfg.get("candidate_refined_pose_rot_cost_weight")
    candidate_refined_pose_wls_conf_mode = map_cfg.get("candidate_refined_pose_wls_conf_mode")
    candidate_refined_pose_wls_conf_variance_scale = map_cfg.get(
        "candidate_refined_pose_wls_conf_variance_scale"
    )
    candidate_refined_pose_wls_downsample = map_cfg.get("candidate_refined_pose_wls_downsample")
    rendered_teacher_coarse_weight = (
        float(map_cfg.get("rendered_teacher_coarse_weight", 0.0)) if coarse_active else 0.0
    )
    depth_weight_strength = float(map_cfg.get("depth_observability_weight", 0.0))
    if rendered_depth is not None and depth_weight_strength > 0:
        depth_weight = depth_observability_weight(
            rendered_depth,
            mask=mask,
            strength=depth_weight_strength,
            power=float(map_cfg.get("depth_observability_power", 1.0)),
            max_weight=float(map_cfg.get("depth_observability_max", 4.0)),
        )
        mask = mask * depth_weight
    trans_obs_strength = float(map_cfg.get("translation_observability_weight", 0.0))
    if rendered_depth is not None and rendered_intrinsics is not None and trans_obs_strength > 0:
        trans_weight = translation_observability_weight(
            rendered_depth,
            rendered_intrinsics,
            mask=mask,
            strength=trans_obs_strength,
            mode=str(map_cfg.get("translation_observability_mode", "xyz")),
            power=float(map_cfg.get("translation_observability_power", 1.0)),
            max_weight=float(map_cfg.get("translation_observability_max", 4.0)),
        )
        mask = mask * trans_weight

    query_fine_weight = resolve_linear_weight(map_cfg, "query_fine_weight", epoch)
    query_fine_raw_weight = resolve_linear_weight(map_cfg, "query_fine_raw_weight", epoch)
    query_local_fine_weight = resolve_linear_weight(map_cfg, "query_local_fine_weight", epoch)
    rendered_teacher_fine_weight = resolve_linear_weight(map_cfg, "rendered_teacher_fine_weight", epoch)
    rendered_teacher_fine_raw_weight = resolve_linear_weight(
        map_cfg, "rendered_teacher_fine_raw_weight", epoch
    )
    query_fine_infonce_weight = resolve_linear_weight(map_cfg, "query_fine_infonce_weight", epoch)
    query_local_fine_infonce_weight = resolve_linear_weight(
        map_cfg,
        "query_local_fine_infonce_weight",
        epoch,
    )
    query_coarse_infonce_weight = (
        resolve_linear_weight(map_cfg, "query_coarse_infonce_weight", epoch) if coarse_active else 0.0
    )
    query_fine_grad_weight = resolve_linear_weight(map_cfg, "query_fine_grad_weight", epoch)
    query_local_fine_grad_weight = resolve_linear_weight(map_cfg, "query_local_fine_grad_weight", epoch)
    query_coarse_grad_weight = (
        resolve_linear_weight(map_cfg, "query_coarse_grad_weight", epoch) if coarse_active else 0.0
    )
    rendered_teacher_fine_infonce_weight = resolve_linear_weight(
        map_cfg, "rendered_teacher_fine_infonce_weight", epoch
    )
    rendered_teacher_coarse_infonce_weight = (
        resolve_linear_weight(map_cfg, "rendered_teacher_coarse_infonce_weight", epoch)
        if coarse_active
        else 0.0
    )
    rendered_teacher_fine_grad_weight = resolve_linear_weight(
        map_cfg, "rendered_teacher_fine_grad_weight", epoch
    )
    rendered_teacher_coarse_grad_weight = (
        resolve_linear_weight(map_cfg, "rendered_teacher_coarse_grad_weight", epoch)
        if coarse_active
        else 0.0
    )
    infonce_temperature = float(map_cfg.get("infonce_temperature", loss_cfg.get("infonce_temperature", 0.07)))
    infonce_samples = int(map_cfg.get("infonce_samples", loss_cfg.get("infonce_samples", 256)))
    infonce_cross_batch = bool(map_cfg.get("infonce_cross_batch", True))
    fine_coarse_ortho_weight = float(map_cfg.get("fine_coarse_ortho_weight", 0.0))
    alpha_coverage_weight = resolve_linear_weight(map_cfg, "alpha_coverage_weight", epoch)
    rgb_l1_weight = resolve_linear_weight(map_cfg, "rgb_l1_weight", epoch)
    variance_target_std = float(map_cfg.get("variance_target_std", 0.05))
    variance_weight_query = float(map_cfg.get("query_variance_weight", 0.0))
    variance_weight_map = float(map_cfg.get("map_variance_weight", 0.0))
    covariance_weight_query = float(map_cfg.get("query_covariance_weight", 0.0))
    covariance_weight_map = float(map_cfg.get("map_covariance_weight", 0.0))
    query_corr_ce_weight = resolve_linear_weight(map_cfg, "query_corr_ce_weight", epoch)
    query_corr_ce_temperature = map_cfg.get("query_corr_ce_temperature")
    query_corr_subpixel_weight = resolve_linear_weight(map_cfg, "query_corr_subpixel_weight", epoch)
    query_corr_flow_weight = resolve_linear_weight(map_cfg, "query_corr_flow_weight", epoch)
    query_corr_flow_cosine_weight = resolve_linear_weight(
        map_cfg,
        "query_corr_flow_cosine_weight",
        epoch,
    )
    query_corr_flow_head_conf_weight = resolve_linear_weight(
        map_cfg,
        "query_corr_flow_head_conf_weight",
        epoch,
    )
    query_corr_peak_margin_weight = resolve_linear_weight(map_cfg, "query_corr_peak_margin_weight", epoch)
    query_corr_distill_weight = resolve_linear_weight(map_cfg, "query_corr_distill_weight", epoch)
    query_corr_peak_margin = float(map_cfg.get("query_corr_peak_margin", 0.05))
    query_corr_low_peak_gap_threshold = float(
        map_cfg.get("query_corr_low_peak_gap_threshold", query_corr_peak_margin)
    )
    query_corr_wls_pose_weight = resolve_linear_weight(map_cfg, "query_corr_wls_pose_weight", epoch)
    query_corr_wls_pose_damping = float(map_cfg.get("query_corr_wls_pose_damping", 1e-3))
    query_corr_wls_pose_update_scale = float(map_cfg.get("query_corr_wls_pose_update_scale", 1.0))
    query_corr_wls_pose_rot_weight = float(map_cfg.get("query_corr_wls_pose_rot_weight", 1.0))
    query_corr_wls_pose_trans_weight = float(map_cfg.get("query_corr_wls_pose_trans_weight", 50.0))
    query_corr_wls_conf_threshold = float(map_cfg.get("query_corr_wls_conf_threshold", 0.0))
    query_corr_wls_conf_power = float(map_cfg.get("query_corr_wls_conf_power", 1.0))
    query_corr_wls_conf_mode = str(map_cfg.get("query_corr_wls_conf_mode", "max"))
    query_corr_wls_conf_variance_scale = float(map_cfg.get("query_corr_wls_conf_variance_scale", 0.5))
    query_corr_wls_min_conf_cov = float(map_cfg.get("query_corr_wls_min_conf_cov", 0.0))
    query_corr_wls_accept_min_conf_mean = float(
        map_cfg.get("query_corr_wls_accept_min_conf_mean", 0.0)
    )
    query_corr_wls_accept_min_conf_cov = float(
        map_cfg.get("query_corr_wls_accept_min_conf_cov", 0.0)
    )
    query_corr_wls_accept_min_delta_mm = float(
        map_cfg.get("query_corr_wls_accept_min_delta_mm", 0.0)
    )
    query_corr_wls_accept_max_delta_mm = float(
        map_cfg.get("query_corr_wls_accept_max_delta_mm", 0.0)
    )
    query_corr_pose_gain_weight = resolve_linear_weight(map_cfg, "query_corr_pose_gain_weight", epoch)
    query_corr_pose_gain_rot_margin_deg = float(map_cfg.get("query_corr_pose_gain_rot_margin_deg", 0.0))
    query_corr_pose_gain_trans_margin_m = float(map_cfg.get("query_corr_pose_gain_trans_margin_mm", 0.0)) / 1000.0
    query_corr_pose_gain_rot_weight = float(map_cfg.get("query_corr_pose_gain_rot_weight", 0.0))
    query_corr_pose_gain_trans_weight = float(map_cfg.get("query_corr_pose_gain_trans_weight", 1.0))
    query_scene_coord_weight = resolve_linear_weight(map_cfg, "query_scene_coord_weight", epoch)
    query_scene_coord_warp_weight = resolve_linear_weight(map_cfg, "query_scene_coord_warp_weight", epoch)
    query_scene_coord_huber_beta = float(map_cfg.get("query_scene_coord_huber_beta", 0.02))
    query_corr_scene_coord_weight = float(map_cfg.get("query_corr_scene_coord_weight", 0.0))
    feature_metric_scene_coord_weight = float(map_cfg.get("feature_metric_scene_coord_weight", 0.0))
    query_corr_radius = int(map_cfg.get("query_corr_radius", 4))
    query_corr_temperature = float(map_cfg.get("query_corr_temperature", 0.05))
    query_corr_huber_delta = float(map_cfg.get("query_corr_huber_delta", 1.0))
    query_corr_min_flow_px = float(map_cfg.get("query_corr_min_flow_px", 0.0))
    query_corr_max_flow_px = float(map_cfg.get("query_corr_max_flow_px", 0.0))
    query_corr_flow_decode_mode = str(map_cfg.get("query_corr_flow_decode_mode", "softargmax"))
    query_corr_feature_preprocess = str(map_cfg.get("query_corr_feature_preprocess", "none"))
    query_corr_highpass_kernel = int(map_cfg.get("query_corr_highpass_kernel", 3))
    query_corr_highpass_scale = float(map_cfg.get("query_corr_highpass_scale", 1.0))
    teacher_corr_weight = resolve_linear_weight(map_cfg, "teacher_corr_weight", epoch)
    teacher_corr_temperature = float(map_cfg.get("teacher_corr_temperature", 0.07))
    teacher_corr_min_confidence = float(map_cfg.get("teacher_corr_min_confidence", 0.0))
    teacher_corr_min_points = int(map_cfg.get("teacher_corr_min_points", 4))
    teacher_corr_use_projector = bool(map_cfg.get("teacher_corr_use_projector", True))
    teacher_corr_negative_exclusion_px = float(map_cfg.get("teacher_corr_negative_exclusion_px", 0.0))
    teacher_corr_positive_weight = float(map_cfg.get("teacher_corr_positive_weight", 0.0))
    teacher_corr_margin_weight = float(map_cfg.get("teacher_corr_margin_weight", 0.0))
    teacher_corr_margin = float(map_cfg.get("teacher_corr_margin", 0.1))
    teacher_corr_local_patch_weight = resolve_linear_weight(map_cfg, "teacher_corr_local_patch_weight", epoch)
    teacher_corr_local_patch_radius = int(map_cfg.get("teacher_corr_local_patch_radius", 2))
    teacher_corr_local_patch_temperature = float(
        map_cfg.get("teacher_corr_local_patch_temperature")
        if map_cfg.get("teacher_corr_local_patch_temperature") is not None
        else teacher_corr_temperature
    )
    teacher_corr_local_patch_min_points = int(
        map_cfg.get("teacher_corr_local_patch_min_points")
        if map_cfg.get("teacher_corr_local_patch_min_points") is not None
        else teacher_corr_min_points
    )
    teacher_corr_local_patch_positive_weight = float(
        map_cfg.get("teacher_corr_local_patch_positive_weight", 0.0)
    )
    teacher_corr_local_patch_margin_weight = float(
        map_cfg.get("teacher_corr_local_patch_margin_weight", 0.0)
    )
    teacher_corr_local_patch_margin = float(map_cfg.get("teacher_corr_local_patch_margin", 0.05))
    query_projected_fine_weight = resolve_linear_weight(
        map_cfg,
        "query_projected_fine_weight",
        epoch,
    )
    query_corr_distill_temperature = float(
        map_cfg.get("query_corr_distill_temperature")
        if map_cfg.get("query_corr_distill_temperature") is not None
        else query_corr_temperature
    )
    query_corr_distill_target_temperature = float(
        map_cfg.get("query_corr_distill_target_temperature")
        if map_cfg.get("query_corr_distill_target_temperature") is not None
        else query_corr_distill_temperature
    )
    query_identity_corr_ce_weight = resolve_linear_weight(map_cfg, "query_identity_corr_ce_weight", epoch)
    query_identity_corr_subpixel_weight = resolve_linear_weight(
        map_cfg,
        "query_identity_corr_subpixel_weight",
        epoch,
    )
    query_identity_corr_flow_weight = resolve_linear_weight(
        map_cfg,
        "query_identity_corr_flow_weight",
        epoch,
    )
    query_identity_corr_peak_margin_weight = resolve_linear_weight(
        map_cfg,
        "query_identity_corr_peak_margin_weight",
        epoch,
    )
    query_identity_corr_distill_weight = resolve_linear_weight(
        map_cfg,
        "query_identity_corr_distill_weight",
        epoch,
    )
    query_identity_corr_distill_temperature = float(
        map_cfg.get("query_identity_corr_distill_temperature")
        if map_cfg.get("query_identity_corr_distill_temperature") is not None
        else query_corr_temperature
    )
    query_identity_corr_distill_target_temperature = float(
        map_cfg.get("query_identity_corr_distill_target_temperature")
        if map_cfg.get("query_identity_corr_distill_target_temperature") is not None
        else query_identity_corr_distill_temperature
    )
    query_identity_corr_peak_margin = float(
        map_cfg.get("query_identity_corr_peak_margin")
        if map_cfg.get("query_identity_corr_peak_margin") is not None
        else query_corr_peak_margin
    )
    query_identity_corr_radius = int(
        map_cfg.get("query_identity_corr_radius")
        if map_cfg.get("query_identity_corr_radius") is not None
        else query_corr_radius
    )
    query_identity_corr_temperature = float(
        map_cfg.get("query_identity_corr_temperature")
        if map_cfg.get("query_identity_corr_temperature") is not None
        else query_corr_temperature
    )
    query_identity_corr_ce_temperature = (
        map_cfg.get("query_identity_corr_ce_temperature")
        if map_cfg.get("query_identity_corr_ce_temperature") is not None
        else query_corr_ce_temperature
    )
    query_identity_corr_feature_preprocess = str(
        map_cfg.get("query_identity_corr_feature_preprocess")
        if map_cfg.get("query_identity_corr_feature_preprocess") is not None
        else query_corr_feature_preprocess
    )
    query_identity_corr_highpass_kernel = int(
        map_cfg.get("query_identity_corr_highpass_kernel")
        if map_cfg.get("query_identity_corr_highpass_kernel") is not None
        else query_corr_highpass_kernel
    )
    query_identity_corr_highpass_scale = float(
        map_cfg.get("query_identity_corr_highpass_scale")
        if map_cfg.get("query_identity_corr_highpass_scale") is not None
        else query_corr_highpass_scale
    )
    query_local_matcher = local_matcher if bool(map_cfg.get("local_matcher_enabled", False)) else None
    query_local_flow_head = local_flow_head if bool(map_cfg.get("local_flow_head_enabled", False)) else None
    query_local_corr_projector = (
        local_corr_projector if bool(map_cfg.get("local_corr_projector_enabled", False)) else None
    )
    project_query_corr_features = bool(map_cfg.get("local_corr_projector_apply_to_query", True))

    def _project_corr_feature(feat, *, is_query=False):
        if query_local_corr_projector is None or feat is None:
            return feat
        if is_query and not project_query_corr_features:
            return feat
        if hasattr(query_local_corr_projector, "project_query"):
            return (
                query_local_corr_projector.project_query(feat)
                if is_query
                else query_local_corr_projector.project_render(feat)
            )
        return query_local_corr_projector(feat)
    query_flow_warp_weight = resolve_linear_weight(map_cfg, "query_flow_warp_weight", epoch)
    query_flow_warp_contrastive_weight = resolve_linear_weight(
        map_cfg, "query_flow_warp_contrastive_weight", epoch
    )
    query_flow_warp_contrastive_margin = float(map_cfg.get("query_flow_warp_contrastive_margin", 0.1))
    query_flow_warp_contrastive_offsets = parse_xy_offsets(
        map_cfg.get("query_flow_warp_contrastive_offsets")
    )
    flow_warp_contrastive_offsets = (
        query_flow_warp_contrastive_offsets
        if query_flow_warp_contrastive_offsets is not None
        else (
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (2.0, 0.0),
            (-2.0, 0.0),
            (0.0, 2.0),
            (0.0, -2.0),
        )
    )
    map_self_flow_warp_weight = resolve_linear_weight(map_cfg, "map_self_flow_warp_weight", epoch)
    map_self_flow_warp_contrastive_weight = resolve_linear_weight(
        map_cfg, "map_self_flow_warp_contrastive_weight", epoch
    )
    map_self_corr_ce_weight = resolve_linear_weight(map_cfg, "map_self_corr_ce_weight", epoch)
    map_self_corr_subpixel_weight = resolve_linear_weight(
        map_cfg, "map_self_corr_subpixel_weight", epoch
    )
    map_self_corr_flow_weight = resolve_linear_weight(
        map_cfg, "map_self_corr_flow_weight", epoch
    )
    map_self_corr_peak_margin_weight = resolve_linear_weight(
        map_cfg, "map_self_corr_peak_margin_weight", epoch
    )
    map_self_feature_metric_pose_weight = resolve_linear_weight(
        map_cfg, "map_self_feature_metric_pose_weight", epoch
    )
    feature_metric_pose_weight = resolve_linear_weight(map_cfg, "feature_metric_pose_weight", epoch)
    feature_metric_pose_damping = float(map_cfg.get("feature_metric_pose_damping", 1e-3))
    feature_metric_pose_update_scale = float(map_cfg.get("feature_metric_pose_update_scale", 1.0))
    feature_metric_pose_rot_weight = float(map_cfg.get("feature_metric_pose_rot_weight", 1.0))
    feature_metric_pose_rot_damping_multiplier = float(map_cfg.get("feature_metric_pose_rot_damping_multiplier", 1.0))
    feature_metric_pose_trans_weight = float(map_cfg.get("feature_metric_pose_trans_weight", 50.0))
    feature_metric_pose_normalize = bool(map_cfg.get("feature_metric_pose_normalize", True))
    feature_metric_use_projector = bool(map_cfg.get("feature_metric_use_projector", False))
    feature_metric_gradient_direction_weight = resolve_linear_weight(
        map_cfg, "feature_metric_gradient_direction_weight", epoch
    )
    feature_metric_gradient_direction_rot_weight = float(
        map_cfg.get("feature_metric_gradient_direction_rot_weight", 1.0)
    )
    feature_metric_gradient_direction_trans_weight = float(
        map_cfg.get("feature_metric_gradient_direction_trans_weight", 50.0)
    )

    fine_query_mask = _resize_mask(mask, pred_fine_target.shape[-2:])
    local_fine_query_mask = _resize_mask(mask, pred_local_fine_target.shape[-2:])
    coarse_query_mask = _resize_mask(mask, pred_coarse_target.shape[-2:])
    rendered_fine_query = _resize_feature(rendered_fine, pred_fine_target.shape[-2:])
    rendered_fine_local_query = _resize_feature(rendered_fine, pred_local_fine_target.shape[-2:])
    rendered_coarse_query = _resize_feature(rendered_coarse, pred_coarse_target.shape[-2:])
    rendered_fine_raw_query = _resize_feature(rendered_fine_raw, pred_fine_target.shape[-2:])

    fine_teacher_mask = _resize_mask(mask, teacher_fine.shape[-2:])
    coarse_teacher_mask = _resize_mask(mask, teacher_coarse.shape[-2:])
    rendered_fine_teacher = _resize_feature(rendered_fine, teacher_fine.shape[-2:])
    rendered_coarse_teacher = _resize_feature(rendered_coarse, teacher_coarse.shape[-2:])
    rendered_fine_raw_teacher = _resize_feature(rendered_fine_raw, teacher_fine.shape[-2:])

    query_fine_loss = (
        l1_feature_loss(pred_fine_target, rendered_fine_query, fine_query_mask)
        + cosine_loss(pred_fine_target, rendered_fine_query, fine_query_mask)
    ) * query_fine_weight
    query_local_fine_loss = (
        l1_feature_loss(pred_local_fine_target, rendered_fine_local_query, local_fine_query_mask)
        + cosine_loss(pred_local_fine_target, rendered_fine_local_query, local_fine_query_mask)
    ) * query_local_fine_weight
    query_fine_raw_loss = zero
    rendered_teacher_fine_raw_loss = zero
    if rendered_fine_raw_query is not None:
        query_fine_raw_loss = (
            l1_feature_loss(pred_fine_target, rendered_fine_raw_query, fine_query_mask)
            + cosine_loss(pred_fine_target, rendered_fine_raw_query, fine_query_mask)
        ) * query_fine_raw_weight
        rendered_teacher_fine_raw_loss = (
            l1_feature_loss(rendered_fine_raw_teacher, teacher_fine, fine_teacher_mask)
            + cosine_loss(rendered_fine_raw_teacher, teacher_fine, fine_teacher_mask)
        ) * rendered_teacher_fine_raw_weight
    query_coarse_loss = (
        l1_feature_loss(pred_coarse_target, rendered_coarse_query, coarse_query_mask)
        + cosine_loss(pred_coarse_target, rendered_coarse_query, coarse_query_mask)
    ) * query_coarse_weight
    coarse_pose_rank_loss = zero
    coarse_pose_rank_metrics = {}
    neg_coarse_for_rank = batch.get("rendered_map_coarse_neg")
    if coarse_pose_rank_weight > 0 and neg_coarse_for_rank is not None:
        neg_coarse_query_for_rank = _resize_feature(neg_coarse_for_rank, pred_coarse_target.shape[-2:])
        neg_mask_for_rank = _resize_mask(batch.get("rendered_map_mask_neg"), pred_coarse_target.shape[-2:])
        coarse_pose_rank_loss, coarse_pose_rank_metrics = coarse_pose_ranking_loss(
            pred_coarse_target,
            rendered_coarse_query,
            neg_coarse_query_for_rank,
            pos_mask=coarse_query_mask,
            neg_mask=neg_mask_for_rank,
            temperature=coarse_pose_rank_temperature,
        )
    coarse_pose_energy_loss = zero
    coarse_pose_energy_metrics = {}
    neg_coarse_for_energy = batch.get("rendered_map_coarse_global_neg")
    if coarse_pose_energy_weight > 0 and neg_coarse_for_energy is not None:
        neg_coarse_query_for_energy = _resize_feature_bank(
            neg_coarse_for_energy,
            pred_coarse_target.shape[-2:],
        )
        neg_mask_for_energy = _resize_mask_bank(
            batch.get("rendered_map_mask_global_neg"),
            pred_coarse_target.shape[-2:],
        )
        coarse_pose_energy_loss, coarse_pose_energy_metrics = coarse_pose_energy_nce_loss(
            pred_coarse_target,
            rendered_coarse_query,
            neg_coarse_query_for_energy,
            pos_mask=coarse_query_mask,
            neg_mask=neg_mask_for_energy,
            temperature=coarse_pose_energy_temperature,
        )
    coarse_pose_local_energy_loss = zero
    coarse_pose_local_energy_metrics = {}
    if coarse_pose_local_energy_weight > 0 and neg_coarse_for_energy is not None:
        neg_coarse_query_for_local = _resize_feature_bank(
            neg_coarse_for_energy,
            pred_coarse_target.shape[-2:],
        )
        neg_mask_for_local = _resize_mask_bank(
            batch.get("rendered_map_mask_global_neg"),
            pred_coarse_target.shape[-2:],
        )
        coarse_pose_local_energy_loss, coarse_pose_local_energy_metrics = candidate_local_render_score_nce_loss(
            pred_coarse_target,
            rendered_coarse_query,
            neg_coarse_query_for_local,
            pos_mask=coarse_query_mask,
            neg_mask=neg_mask_for_local,
            temperature=coarse_pose_local_energy_temperature,
            radius=coarse_pose_local_energy_radius,
            preprocess=coarse_pose_local_energy_preprocess,
            highpass_kernel=coarse_pose_local_energy_highpass_kernel,
        )
    candidate_render_score_loss = zero
    candidate_render_score_metrics = {}
    if candidate_render_score_weight > 0:
        candidate_pose = batch.get("rendered_map_candidate_pose")
        candidate_mask = batch.get("rendered_map_candidate_mask")
        candidate_valid_mask = batch.get("rendered_map_candidate_valid_mask")
        if candidate_render_score_feature in ("fine", "query_fine"):
            candidate_feat = batch.get("rendered_map_candidate_fine")
            query_candidate_feat = pred_fine_target
        elif candidate_render_score_feature in ("local_fine", "fine_local"):
            candidate_feat = batch.get("rendered_map_candidate_fine")
            query_candidate_feat = pred_local_fine_target
        elif candidate_render_score_feature in ("coarse", "query_coarse"):
            candidate_feat = batch.get("rendered_map_candidate_coarse")
            query_candidate_feat = pred_coarse_target
        else:
            raise ValueError(
                "map_supervision.candidate_render_score_feature must be one of "
                "{'fine', 'local_fine', 'coarse'}"
            )
        pose_gt_for_candidates = batch.get("rendered_map_pose_gt", batch.get("pose_gt"))
        if candidate_feat is not None and candidate_pose is not None and pose_gt_for_candidates is not None:
            candidate_feat = _resize_feature_bank(candidate_feat, query_candidate_feat.shape[-2:])
            candidate_mask = _resize_mask_bank(candidate_mask, query_candidate_feat.shape[-2:])
            candidate_render_score_loss, candidate_render_score_metrics = render_score_candidate_listwise_loss(
                query_candidate_feat,
                candidate_feat,
                candidate_pose,
                pose_gt_for_candidates,
                mask=candidate_mask,
                candidate_valid_mask=candidate_valid_mask,
                mode=candidate_render_score_mode,
                temperature=candidate_render_score_temperature,
                radius=candidate_render_score_radius,
                preprocess=candidate_render_score_preprocess,
                highpass_kernel=candidate_render_score_highpass_kernel,
                rot_cost_weight=candidate_render_score_rot_cost_weight,
            )
        else:
            candidate_render_score_metrics = {
                "map_candidate_render_score_loss": zero.detach(),
                "map_candidate_render_score_acc": zero.detach(),
            }
    candidate_score_fusion_loss = zero
    candidate_score_fusion_metrics = {}
    candidate_score_fusion_details = None
    if candidate_score_fusion_weight > 0:
        candidate_pose = batch.get("rendered_map_candidate_pose")
        candidate_mask = batch.get("rendered_map_candidate_mask")
        candidate_valid_mask = batch.get("rendered_map_candidate_valid_mask")
        if candidate_score_fusion_feature in ("fine", "query_fine"):
            candidate_feat = batch.get("rendered_map_candidate_fine")
            query_candidate_feat = pred_fine_target
        elif candidate_score_fusion_feature in ("local_fine", "fine_local"):
            candidate_feat = batch.get("rendered_map_candidate_fine")
            query_candidate_feat = pred_local_fine_target
        elif candidate_score_fusion_feature in ("coarse", "query_coarse"):
            candidate_feat = batch.get("rendered_map_candidate_coarse")
            query_candidate_feat = pred_coarse_target
        else:
            raise ValueError(
                "map_supervision.candidate_score_fusion_feature must be one of "
                "{'fine', 'local_fine', 'coarse'}"
            )
        candidate_score_fusion_metrics = {
            "map_candidate_score_fusion_loss": zero.detach(),
            "map_candidate_score_fusion_acc": zero.detach(),
            "map_candidate_score_fusion_render_acc": zero.detach(),
        }
        pose_gt_for_candidates = batch.get("rendered_map_pose_gt", batch.get("pose_gt"))
        if (
            candidate_feat is not None
            and candidate_pose is not None
            and pose_gt_for_candidates is not None
            and candidate_score_fusion_head is not None
        ):
            candidate_feat = _resize_feature_bank(candidate_feat, query_candidate_feat.shape[-2:])
            candidate_mask = _resize_mask_bank(candidate_mask, query_candidate_feat.shape[-2:])
            (
                candidate_score_fusion_loss,
                candidate_score_fusion_metrics,
                candidate_score_fusion_details,
            ) = candidate_score_fusion_listwise_loss(
                query_candidate_feat,
                candidate_feat,
                candidate_pose,
                pose_gt_for_candidates,
                candidate_score_fusion_head,
                batch=batch,
                mask=candidate_mask,
                candidate_valid_mask=candidate_valid_mask,
                mode=candidate_score_fusion_mode,
                temperature=candidate_score_fusion_temperature,
                radius=candidate_score_fusion_radius,
                preprocess=candidate_score_fusion_preprocess,
                highpass_kernel=candidate_score_fusion_highpass_kernel,
                score_map_mode=candidate_score_fusion_score_map_mode,
                rot_cost_weight=candidate_score_fusion_rot_cost_weight,
                target_mode=candidate_score_fusion_target_mode,
                target_temperature_m=candidate_score_fusion_target_temperature_m,
                render_feature_mode=candidate_score_fusion_render_feature_mode,
                wls_radius=candidate_score_fusion_wls_radius,
                wls_temperature=candidate_score_fusion_wls_temperature,
                wls_damping=candidate_score_fusion_wls_damping,
                wls_update_scale=candidate_score_fusion_wls_update_scale,
                wls_conf_mode=candidate_score_fusion_wls_conf_mode,
                wls_conf_variance_scale=candidate_score_fusion_wls_conf_variance_scale,
                wls_downsample=candidate_score_fusion_wls_downsample,
                cost_regression_weight=candidate_score_fusion_cost_regression_weight,
                cost_regression_temperature_m=candidate_score_fusion_cost_regression_temperature_m,
                pairwise_rank_weight=candidate_score_fusion_pairwise_rank_weight,
                pairwise_rank_temperature=candidate_score_fusion_pairwise_rank_temperature,
                pairwise_rank_min_gap_m=candidate_score_fusion_pairwise_rank_min_gap_m,
                basin_trans_m=float(map_cfg.get("candidate_score_fusion_basin_trans_m", 0.25)),
                basin_rot_deg=float(map_cfg.get("candidate_score_fusion_basin_rot_deg", 5.0)),
                return_details=True,
            )
        elif candidate_score_fusion_head is None:
            candidate_score_fusion_metrics["map_candidate_score_fusion_missing_head"] = torch.ones(
                (), device=device
            )
    candidate_refined_pose_loss_value = zero
    candidate_refined_pose_metrics = {}
    if candidate_refined_pose_weight > 0:
        pose_gt_for_candidates = batch.get("pose_gt", batch.get("rendered_map_pose_gt"))
        if candidate_two_stage_enabled and map_renderer is not None and candidate_score_fusion_details is not None:
            source_pose = batch.get("rendered_map_candidate_pose")
            if source_pose is not None:
                selected_indices = select_candidate_stage2_indices(
                    valid=candidate_score_fusion_details["valid"],
                    topm=candidate_stage2_topm,
                    selection=candidate_stage2_selection,
                    logits=candidate_score_fusion_details.get("raw_logits"),
                    pose_cost=candidate_score_fusion_details.get("pose_cost"),
                )
                selected_pose = gather_candidate_bank(source_pose, selected_indices)
                source_valid = batch.get("rendered_map_candidate_valid_mask")
                if source_valid is None:
                    source_valid = torch.ones(
                        source_pose.shape[:2],
                        device=source_pose.device,
                        dtype=torch.bool,
                    )
                selected_valid = gather_candidate_bank(source_valid, selected_indices)
                if candidate_stage2_detach_selection:
                    selected_pose = selected_pose.detach()
                    selected_valid = selected_valid.detach()
                renderer_has_trainable = (
                    bool(map_renderer.has_trainable_params())
                    if hasattr(map_renderer, "has_trainable_params")
                    else True
                )
                map_renderer.attach_pose_candidate_renders(
                    batch,
                    selected_pose,
                    require_grad=bool(candidate_stage2_train_map and renderer_has_trainable),
                    prefix=candidate_stage2_prefix,
                    candidate_valid_mask=selected_valid,
                    max_candidates=0,
                    feature=candidate_refined_pose_feature,
                )
                batch[f"{candidate_stage2_prefix}_source_indices"] = selected_indices.detach()
        refined_pose = batch.get(f"{candidate_stage2_prefix}_pose")
        refined_mask = batch.get(f"{candidate_stage2_prefix}_valid_mask")
        refined_depth = batch.get(f"{candidate_stage2_prefix}_depth")
        refined_intrinsics = batch.get(f"{candidate_stage2_prefix}_intrinsics")
        if candidate_refined_pose_feature in ("fine", "query_fine"):
            refined_feat = batch.get(f"{candidate_stage2_prefix}_fine")
            refined_query_feat = pred_fine_target
        elif candidate_refined_pose_feature in ("local_fine", "fine_local"):
            refined_feat = batch.get(f"{candidate_stage2_prefix}_fine")
            refined_query_feat = pred_local_fine_target
        elif candidate_refined_pose_feature in ("coarse", "query_coarse"):
            refined_feat = batch.get(f"{candidate_stage2_prefix}_coarse")
            refined_query_feat = pred_coarse_target
        else:
            raise ValueError(
                "map_supervision.candidate_refined_pose_feature must be one of "
                "{'fine', 'local_fine', 'coarse'}"
            )
        if (
            refined_feat is not None
            and refined_pose is not None
            and refined_depth is not None
            and refined_intrinsics is not None
            and pose_gt_for_candidates is not None
        ):
            if refined_mask is None:
                refined_mask = torch.ones(
                    refined_pose.shape[:2],
                    device=refined_pose.device,
                    dtype=torch.bool,
                )
            refined_feat = _resize_feature_bank(refined_feat, refined_query_feat.shape[-2:])
            candidate_refined_pose_loss_value, candidate_refined_pose_metrics = candidate_refined_pose_loss(
                refined_query_feat,
                refined_feat,
                refined_pose,
                pose_gt_for_candidates,
                refined_depth,
                refined_intrinsics,
                refined_mask,
                radius=int(
                    candidate_refined_pose_radius
                    if candidate_refined_pose_radius is not None
                    else candidate_score_fusion_wls_radius
                    if candidate_score_fusion_wls_radius is not None
                    else candidate_score_fusion_radius
                ),
                temperature=float(
                    candidate_refined_pose_temperature
                    if candidate_refined_pose_temperature is not None
                    else candidate_score_fusion_wls_temperature
                    if candidate_score_fusion_wls_temperature is not None
                    else candidate_score_fusion_temperature
                ),
                damping=float(
                    candidate_refined_pose_damping
                    if candidate_refined_pose_damping is not None
                    else candidate_score_fusion_wls_damping
                ),
                update_scale=float(
                    candidate_refined_pose_update_scale
                    if candidate_refined_pose_update_scale is not None
                    else candidate_score_fusion_wls_update_scale
                ),
                rot_cost_weight=float(
                    candidate_refined_pose_rot_cost_weight
                    if candidate_refined_pose_rot_cost_weight is not None
                    else candidate_score_fusion_rot_cost_weight
                ),
                wls_conf_mode=str(
                    candidate_refined_pose_wls_conf_mode
                    if candidate_refined_pose_wls_conf_mode is not None
                    else candidate_score_fusion_wls_conf_mode
                ),
                wls_conf_variance_scale=float(
                    candidate_refined_pose_wls_conf_variance_scale
                    if candidate_refined_pose_wls_conf_variance_scale is not None
                    else candidate_score_fusion_wls_conf_variance_scale
                ),
                wls_downsample=int(
                    candidate_refined_pose_wls_downsample
                    if candidate_refined_pose_wls_downsample is not None
                    else candidate_score_fusion_wls_downsample
                ),
            )
        else:
            candidate_refined_pose_metrics = {
                "map_candidate_refined_pose_loss": zero.detach(),
                "map_candidate_refined_pose_valid_fraction": zero.detach(),
            }
    query_fine_nce_loss = zero
    if query_fine_infonce_weight > 0:
        query_fine_nce_loss = infonce_contrastive_loss(
            pred_fine_target,
            rendered_fine_query,
            mask=fine_query_mask,
            temperature=infonce_temperature,
            n_samples=infonce_samples,
            cross_batch=infonce_cross_batch,
        ) * query_fine_infonce_weight
    query_local_fine_nce_loss = zero
    if query_local_fine_infonce_weight > 0:
        query_local_fine_nce_loss = infonce_contrastive_loss(
            pred_local_fine_target,
            rendered_fine_local_query,
            mask=local_fine_query_mask,
            temperature=infonce_temperature,
            n_samples=infonce_samples,
            cross_batch=infonce_cross_batch,
        ) * query_local_fine_infonce_weight
    query_coarse_nce_loss = zero
    if query_coarse_infonce_weight > 0:
        query_coarse_nce_loss = infonce_contrastive_loss(
            pred_coarse_target,
            rendered_coarse_query,
            mask=coarse_query_mask,
            temperature=infonce_temperature,
            n_samples=infonce_samples,
            cross_batch=infonce_cross_batch,
        ) * query_coarse_infonce_weight
    query_fine_grad_loss = zero
    if query_fine_grad_weight > 0:
        query_fine_grad_loss = feature_gradient_loss(
            pred_fine_target,
            rendered_fine_query,
            fine_query_mask,
        ) * query_fine_grad_weight
    query_local_fine_grad_loss = zero
    if query_local_fine_grad_weight > 0:
        query_local_fine_grad_loss = feature_gradient_loss(
            pred_local_fine_target,
            rendered_fine_local_query,
            local_fine_query_mask,
        ) * query_local_fine_grad_weight
    query_coarse_grad_loss = zero
    if query_coarse_grad_weight > 0:
        query_coarse_grad_loss = feature_gradient_loss(
            pred_coarse_target,
            rendered_coarse_query,
            coarse_query_mask,
        ) * query_coarse_grad_weight
    rendered_teacher_fine_loss = (
        l1_feature_loss(rendered_fine_teacher, teacher_fine, fine_teacher_mask)
        + cosine_loss(rendered_fine_teacher, teacher_fine, fine_teacher_mask)
    ) * rendered_teacher_fine_weight
    rendered_teacher_fine_nce_loss = zero
    if rendered_teacher_fine_infonce_weight > 0:
        rendered_teacher_fine_nce_loss = infonce_contrastive_loss(
            rendered_fine_teacher,
            teacher_fine,
            mask=fine_teacher_mask,
            temperature=infonce_temperature,
            n_samples=infonce_samples,
            cross_batch=infonce_cross_batch,
        ) * rendered_teacher_fine_infonce_weight
    rendered_teacher_fine_grad_loss = zero
    if rendered_teacher_fine_grad_weight > 0:
        rendered_teacher_fine_grad_loss = feature_gradient_loss(
            rendered_fine_teacher,
            teacher_fine,
            fine_teacher_mask,
        ) * rendered_teacher_fine_grad_weight
    rendered_teacher_coarse_loss = (
        l1_feature_loss(rendered_coarse_teacher, teacher_coarse, coarse_teacher_mask)
        + cosine_loss(rendered_coarse_teacher, teacher_coarse, coarse_teacher_mask)
    ) * rendered_teacher_coarse_weight
    rendered_teacher_coarse_nce_loss = zero
    if rendered_teacher_coarse_infonce_weight > 0:
        rendered_teacher_coarse_nce_loss = infonce_contrastive_loss(
            rendered_coarse_teacher,
            teacher_coarse,
            mask=coarse_teacher_mask,
            temperature=infonce_temperature,
            n_samples=infonce_samples,
            cross_batch=infonce_cross_batch,
        ) * rendered_teacher_coarse_infonce_weight
    rendered_teacher_coarse_grad_loss = zero
    if rendered_teacher_coarse_grad_weight > 0:
        rendered_teacher_coarse_grad_loss = feature_gradient_loss(
            rendered_coarse_teacher,
            teacher_coarse,
            coarse_teacher_mask,
        ) * rendered_teacher_coarse_grad_weight
    query_projected_fine_loss = zero
    query_projected_fine_cos = zero
    if query_projected_fine_weight > 0:
        if query_local_corr_projector is None:
            raise KeyError(
                "map_supervision.query_projected_fine_weight > 0 requires "
                "map_supervision.local_corr_projector_enabled=true"
            )
        projected_query = _project_corr_feature(pred_local_fine_target, is_query=True)
        projected_map = _project_corr_feature(rendered_fine).detach()
        projected_map_query = _resize_feature(projected_map, projected_query.shape[-2:])
        projected_mask = _resize_mask(mask, projected_query.shape[-2:])
        query_projected_fine_loss = (
            l1_feature_loss(projected_query, projected_map_query, projected_mask)
            + cosine_loss(projected_query, projected_map_query, projected_mask)
        )
        with torch.no_grad():
            query_projected_fine_cos = 1.0 - cosine_loss(
                projected_query,
                projected_map_query,
                projected_mask,
            )
    map_fine_coarse_ortho_loss = feature_orthogonality_loss(rendered_fine, rendered_coarse, mask) * fine_coarse_ortho_weight

    query_variance_loss = zero
    map_variance_loss = zero
    query_covariance_loss = zero
    map_covariance_loss = zero
    if variance_weight_query > 0:
        query_variance_loss = 0.5 * (
            feature_variance_loss(pred_fine_target, fine_query_mask, variance_target_std)
            + feature_variance_loss(pred_coarse_target, coarse_query_mask, variance_target_std)
        )
    if variance_weight_map > 0:
        map_variance_loss = 0.5 * (
            feature_variance_loss(rendered_fine_query, fine_query_mask, variance_target_std)
            + feature_variance_loss(rendered_coarse_query, coarse_query_mask, variance_target_std)
        )
    if covariance_weight_query > 0:
        query_covariance_loss = 0.5 * (
            feature_covariance_loss(pred_fine_target, fine_query_mask)
            + feature_covariance_loss(pred_coarse_target, coarse_query_mask)
        )
    if covariance_weight_map > 0:
        map_covariance_loss = 0.5 * (
            feature_covariance_loss(rendered_fine_query, fine_query_mask)
            + feature_covariance_loss(rendered_coarse_query, coarse_query_mask)
        )

    scene_center, scene_scale = scene_coord_center_scale(map_cfg, device, dtype=pred_fine_target.dtype)
    query_scene_coord_loss = zero
    query_scene_coord_metrics = {}
    if query_scene_coord_weight > 0:
        if pred_scene_coord is None:
            raise KeyError("map_supervision.query_scene_coord_weight > 0 requires model.scene_coord_head=true")
        if rendered_position is None:
            raise KeyError("query_scene_coord_weight requires rendered_map_position in the batch")
        query_scene_coord_loss, query_scene_coord_metrics = scene_coord_regression_loss(
            pred_scene_coord,
            rendered_position,
            fine_query_mask,
            scene_center,
            scene_scale,
            beta=query_scene_coord_huber_beta,
        )

    query_scene_coord_warp_loss = zero
    query_scene_coord_warp_metrics = {}
    if query_scene_coord_warp_weight > 0:
        if pred_scene_coord is None:
            raise KeyError("map_supervision.query_scene_coord_warp_weight > 0 requires model.scene_coord_head=true")
        neg_position_for_scene = batch.get("rendered_map_position_neg")
        flow_neg_to_gt_for_scene = batch.get("rendered_map_flow_neg_to_gt")
        flow_valid_for_scene = batch.get("rendered_map_flow_valid_neg_to_gt")
        if (
            neg_position_for_scene is not None
            and flow_neg_to_gt_for_scene is not None
            and flow_valid_for_scene is not None
        ):
            scene_valid = flow_valid_for_scene.float()
            neg_mask_for_scene = batch.get("rendered_map_mask_neg")
            if neg_mask_for_scene is not None:
                if neg_mask_for_scene.shape[-2:] != scene_valid.shape[-2:]:
                    neg_mask_for_scene = F.interpolate(
                        neg_mask_for_scene.float(),
                        size=scene_valid.shape[-2:],
                        mode="nearest",
                    )
                scene_valid = scene_valid * neg_mask_for_scene.float()
            if prior_mask is not None:
                prior_for_scene = prior_mask.float()
                if prior_for_scene.shape[-2:] != scene_valid.shape[-2:]:
                    prior_for_scene = F.interpolate(
                        prior_for_scene,
                        size=scene_valid.shape[-2:],
                        mode="nearest",
                    )
                scene_valid = scene_valid * prior_for_scene
            query_scene_coord_warp_loss, query_scene_coord_warp_metrics = scene_coord_flow_warp_loss(
                pred_scene_coord,
                neg_position_for_scene,
                flow_neg_to_gt_for_scene,
                scene_valid,
                scene_center,
                scene_scale,
                beta=query_scene_coord_huber_beta,
            )

    teacher_corr_loss = zero
    teacher_corr_local_patch_loss = zero
    teacher_corr_metrics = {}
    if teacher_corr_weight > 0 or teacher_corr_local_patch_weight > 0:
        required_teacher_corr = [
            "teacher_corr_query_xy",
            "teacher_corr_map_xy",
            "teacher_corr_conf",
            "teacher_corr_valid",
        ]
        missing_teacher_corr = [key for key in required_teacher_corr if key not in batch]
        teacher_corr_metrics["map_teacher_corr_missing"] = torch.tensor(
            1.0 if missing_teacher_corr else 0.0,
            device=device,
        )
        if not missing_teacher_corr:
            teacher_query_feat = pred_local_fine_target
            teacher_map_feat = rendered_fine
            if teacher_corr_use_projector and query_local_corr_projector is not None:
                teacher_query_feat = _project_corr_feature(teacher_query_feat, is_query=True)
                teacher_map_feat = _project_corr_feature(teacher_map_feat)
            if teacher_corr_weight > 0:
                teacher_corr_loss, teacher_corr_metrics_impl = sparse_teacher_correspondence_loss(
                    teacher_query_feat,
                    teacher_map_feat,
                    batch["teacher_corr_query_xy"],
                    batch["teacher_corr_map_xy"],
                    batch["teacher_corr_conf"],
                    batch["teacher_corr_valid"],
                    xy_source_hw=batch.get("teacher_corr_hw"),
                    temperature=teacher_corr_temperature,
                    min_confidence=teacher_corr_min_confidence,
                    min_points=teacher_corr_min_points,
                    negative_exclusion_px=teacher_corr_negative_exclusion_px,
                    positive_weight=teacher_corr_positive_weight,
                    margin_weight=teacher_corr_margin_weight,
                    margin=teacher_corr_margin,
                )
                teacher_corr_metrics.update(teacher_corr_metrics_impl)
            if teacher_corr_local_patch_weight > 0:
                teacher_corr_local_patch_loss, teacher_corr_patch_metrics = sparse_teacher_local_patch_loss(
                    teacher_query_feat,
                    teacher_map_feat,
                    batch["teacher_corr_query_xy"],
                    batch["teacher_corr_map_xy"],
                    batch["teacher_corr_conf"],
                    batch["teacher_corr_valid"],
                    xy_source_hw=batch.get("teacher_corr_hw"),
                    radius=teacher_corr_local_patch_radius,
                    temperature=teacher_corr_local_patch_temperature,
                    min_confidence=teacher_corr_min_confidence,
                    min_points=teacher_corr_local_patch_min_points,
                    positive_weight=teacher_corr_local_patch_positive_weight,
                    margin_weight=teacher_corr_local_patch_margin_weight,
                    margin=teacher_corr_local_patch_margin,
                )
                teacher_corr_metrics.update(teacher_corr_patch_metrics)
                with torch.no_grad():
                    _, teacher_corr_patch_self_metrics = sparse_teacher_local_patch_loss(
                        teacher_map_feat.detach(),
                        teacher_map_feat.detach(),
                        batch["teacher_corr_map_xy"],
                        batch["teacher_corr_map_xy"],
                        batch["teacher_corr_conf"],
                        batch["teacher_corr_valid"],
                        xy_source_hw=batch.get("teacher_corr_hw"),
                        radius=teacher_corr_local_patch_radius,
                        temperature=teacher_corr_local_patch_temperature,
                        min_confidence=teacher_corr_min_confidence,
                        min_points=teacher_corr_local_patch_min_points,
                    )
                teacher_corr_metrics.update(
                    {
                        key.replace("map_teacher_patch_", "map_teacher_patch_self_"): value
                        for key, value in teacher_corr_patch_self_metrics.items()
                    }
                )
                with torch.no_grad():
                    _, teacher_corr_patch_radio_metrics = sparse_teacher_local_patch_loss(
                        teacher_fine.detach(),
                        rendered_fine_teacher.detach(),
                        batch["teacher_corr_query_xy"],
                        batch["teacher_corr_map_xy"],
                        batch["teacher_corr_conf"],
                        batch["teacher_corr_valid"],
                        xy_source_hw=batch.get("teacher_corr_hw"),
                        radius=teacher_corr_local_patch_radius,
                        temperature=teacher_corr_local_patch_temperature,
                        min_confidence=teacher_corr_min_confidence,
                        min_points=teacher_corr_local_patch_min_points,
                    )
                teacher_corr_metrics.update(
                    {
                        key.replace("map_teacher_patch_", "map_teacher_patch_radio_"): value
                        for key, value in teacher_corr_patch_radio_metrics.items()
                    }
                )

    query_corr_ce_loss = zero
    query_corr_subpx_loss = zero
    query_corr_flow_loss = zero
    query_corr_flow_cosine_loss = zero
    query_corr_peak_margin_loss = zero
    query_corr_distill_loss = zero
    query_corr_wls_pose_loss = zero
    query_corr_pose_gain_loss = zero
    query_corr_flow_conf_loss = zero
    query_corr_wls_pose_metrics = {}
    query_corr_metrics = {}
    if (
        query_corr_ce_weight > 0
        or
        query_corr_subpixel_weight > 0
        or query_corr_flow_weight > 0
        or query_corr_flow_cosine_weight > 0
        or query_corr_flow_head_conf_weight > 0
        or query_corr_peak_margin_weight > 0
        or query_corr_distill_weight > 0
        or query_corr_wls_pose_weight > 0
        or query_corr_pose_gain_weight > 0
    ):
        neg_fine_for_corr = batch.get("rendered_map_fine_neg")
        neg_mask_for_corr = batch.get("rendered_map_mask_neg")
        neg_depth_for_corr = batch.get("rendered_map_depth_neg")
        neg_position_for_corr = batch.get("rendered_map_position_neg")
        flow_neg_to_gt = batch.get("rendered_map_flow_neg_to_gt")
        flow_valid_neg_to_gt = batch.get("rendered_map_flow_valid_neg_to_gt")
        missing_negatives = {
            "rendered_map_fine_neg": neg_fine_for_corr is None,
            "rendered_map_flow_neg_to_gt": flow_neg_to_gt is None,
            "rendered_map_flow_valid_neg_to_gt": flow_valid_neg_to_gt is None,
        }
        skipped_missing_negatives = any(missing_negatives.values())
        query_corr_metrics.update(
            {
                "map_query_corr_skipped_missing_negatives": torch.tensor(
                    1.0 if skipped_missing_negatives else 0.0,
                    device=device,
                ),
                "map_query_corr_missing_rendered_map_fine_neg": torch.tensor(
                    1.0 if missing_negatives["rendered_map_fine_neg"] else 0.0,
                    device=device,
                ),
                "map_query_corr_missing_rendered_map_flow_neg_to_gt": torch.tensor(
                    1.0 if missing_negatives["rendered_map_flow_neg_to_gt"] else 0.0,
                    device=device,
                ),
                "map_query_corr_missing_rendered_map_flow_valid_neg_to_gt": torch.tensor(
                    1.0 if missing_negatives["rendered_map_flow_valid_neg_to_gt"] else 0.0,
                    device=device,
                ),
            }
        )
        if (
            neg_fine_for_corr is not None
            and flow_neg_to_gt is not None
            and flow_valid_neg_to_gt is not None
        ):
            corr_rendered_feat = neg_fine_for_corr
            corr_query_feat = pred_local_fine_target
            if query_corr_scene_coord_weight > 0:
                if pred_scene_coord is None:
                    raise KeyError("query_corr_scene_coord_weight > 0 requires model.scene_coord_head=true")
                if neg_position_for_corr is None:
                    raise KeyError("query_corr_scene_coord_weight requires rendered_map_position_neg")
                neg_scene_for_corr = normalize_scene_coord_map(
                    neg_position_for_corr,
                    scene_center,
                    scene_scale,
                    target_hw=neg_fine_for_corr.shape[-2:],
                )
                corr_rendered_feat = augment_feature_with_scene_coord(
                    corr_rendered_feat,
                    neg_scene_for_corr,
                    query_corr_scene_coord_weight,
                )
                corr_query_feat = augment_feature_with_scene_coord(
                    corr_query_feat,
                    pred_scene_coord,
                    query_corr_scene_coord_weight,
                )
            if query_local_corr_projector is not None and query_corr_scene_coord_weight <= 0:
                corr_rendered_feat = _project_corr_feature(corr_rendered_feat)
                corr_query_feat = _project_corr_feature(corr_query_feat, is_query=True)
            corr_valid = flow_valid_neg_to_gt.float()
            if neg_mask_for_corr is not None:
                if neg_mask_for_corr.shape[-2:] != corr_valid.shape[-2:]:
                    neg_mask_for_corr = F.interpolate(
                        neg_mask_for_corr.float(),
                        size=corr_valid.shape[-2:],
                        mode="nearest",
                    )
                corr_valid = corr_valid * neg_mask_for_corr.float()
            if prior_mask is not None:
                prior_for_corr = prior_mask.float()
                if prior_for_corr.shape[-2:] != corr_valid.shape[-2:]:
                    prior_for_corr = F.interpolate(
                        prior_for_corr,
                        size=corr_valid.shape[-2:],
                        mode="nearest",
                    )
                corr_valid = corr_valid * prior_for_corr
            if neg_depth_for_corr is not None and depth_weight_strength > 0:
                corr_valid = corr_valid * depth_observability_weight(
                    neg_depth_for_corr,
                    mask=corr_valid,
                    strength=depth_weight_strength,
                    power=float(map_cfg.get("depth_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("depth_observability_max", 4.0)),
                )
            if (
                neg_depth_for_corr is not None
                and rendered_intrinsics is not None
                and trans_obs_strength > 0
            ):
                corr_valid = corr_valid * translation_observability_weight(
                    neg_depth_for_corr,
                    rendered_intrinsics,
                    mask=corr_valid,
                    strength=trans_obs_strength,
                    mode=str(map_cfg.get("translation_observability_mode", "xyz")),
                    power=float(map_cfg.get("translation_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("translation_observability_max", 4.0)),
                )
            pose_neg = batch.get("rendered_map_pose_neg")
            pose_gt = batch.get("rendered_map_pose_gt")
            compute_corr_wls = (
                (query_corr_wls_pose_weight > 0 or query_corr_pose_gain_weight > 0)
                and neg_depth_for_corr is not None
                and pose_neg is not None
                and pose_gt is not None
                and rendered_intrinsics is not None
            )
            query_corr_metrics["map_query_corr_wls_pose_skipped_missing_inputs"] = torch.tensor(
                1.0
                if (
                    (query_corr_wls_pose_weight > 0 or query_corr_pose_gain_weight > 0)
                    and not compute_corr_wls
                )
                else 0.0,
                device=device,
            )
            if (
                query_corr_ce_weight > 0
                or query_corr_subpixel_weight > 0
                or query_corr_flow_weight > 0
                or query_corr_flow_cosine_weight > 0
                or query_corr_flow_head_conf_weight > 0
                or query_corr_peak_margin_weight > 0
                or query_corr_wls_pose_weight > 0
                or query_corr_pose_gain_weight > 0
            ):
                joint_corr = local_correlation_joint_losses(
                    corr_rendered_feat,
                    corr_query_feat,
                    flow_neg_to_gt,
                    corr_valid,
                    radius=query_corr_radius,
                    temperature=query_corr_temperature,
                    ce_temperature=query_corr_ce_temperature,
                    huber_delta=query_corr_huber_delta,
                    peak_margin=query_corr_peak_margin,
                    compute_ce=query_corr_ce_weight > 0,
                    compute_subpixel=query_corr_subpixel_weight > 0,
                    compute_flow=query_corr_flow_weight > 0,
                    compute_flow_cosine=query_corr_flow_cosine_weight > 0,
                    compute_peak=query_corr_peak_margin_weight > 0,
                    compute_wls_pose=compute_corr_wls and query_corr_wls_pose_weight > 0,
                    compute_pose_gain=compute_corr_wls and query_corr_pose_gain_weight > 0,
                    matcher=query_local_matcher,
                    flow_head=query_local_flow_head,
                    flow_head_conf_weight=query_corr_flow_head_conf_weight,
                    depth=neg_depth_for_corr,
                    pose_ref=pose_neg,
                    pose_gt=pose_gt,
                    intrinsics=rendered_intrinsics,
                    damping=query_corr_wls_pose_damping,
                    update_scale=query_corr_wls_pose_update_scale,
                    rot_weight=query_corr_wls_pose_rot_weight,
                    trans_weight=query_corr_wls_pose_trans_weight,
                    wls_conf_threshold=query_corr_wls_conf_threshold,
                    wls_conf_power=query_corr_wls_conf_power,
                    wls_conf_mode=query_corr_wls_conf_mode,
                    wls_conf_variance_scale=query_corr_wls_conf_variance_scale,
                    wls_min_conf_cov=query_corr_wls_min_conf_cov,
                    wls_accept_min_conf_mean=query_corr_wls_accept_min_conf_mean,
                    wls_accept_min_conf_cov=query_corr_wls_accept_min_conf_cov,
                    wls_accept_min_delta_mm=query_corr_wls_accept_min_delta_mm,
                    wls_accept_max_delta_mm=query_corr_wls_accept_max_delta_mm,
                    pose_gain_trans_margin_m=query_corr_pose_gain_trans_margin_m,
                    pose_gain_rot_margin_deg=query_corr_pose_gain_rot_margin_deg,
                    pose_gain_rot_weight=query_corr_pose_gain_rot_weight,
                    pose_gain_trans_weight=query_corr_pose_gain_trans_weight,
                    min_flow_px=query_corr_min_flow_px,
                    max_flow_px=query_corr_max_flow_px,
                    flow_decode_mode=query_corr_flow_decode_mode,
                    feature_preprocess=query_corr_feature_preprocess,
                    highpass_kernel=query_corr_highpass_kernel,
                    highpass_scale=query_corr_highpass_scale,
                    low_peak_gap_threshold=query_corr_low_peak_gap_threshold,
                )
                query_corr_ce_loss = joint_corr["losses"]["ce"]
                query_corr_subpx_loss = joint_corr["losses"]["subpixel"]
                query_corr_flow_loss = joint_corr["losses"]["flow"]
                query_corr_flow_cosine_loss = joint_corr["losses"].get("flow_cosine", zero)
                query_corr_peak_margin_loss = joint_corr["losses"]["peak"]
                query_corr_pose_gain_loss = joint_corr["losses"]["pose_gain"]
                query_corr_flow_conf_loss = joint_corr["losses"]["flow_conf"]
                query_corr_metrics.update(joint_corr["metrics"])
                if compute_corr_wls:
                    query_corr_wls_pose_loss = joint_corr["losses"]["wls_pose"]
                    query_corr_wls_pose_metrics = {
                        key: value
                        for key, value in joint_corr["metrics"].items()
                        if key.startswith("map_corr_wls_")
                    }
            if query_corr_distill_weight > 0:
                corr_target_feat = rendered_fine
                if query_corr_scene_coord_weight > 0:
                    if rendered_position is None:
                        raise KeyError("query_corr_distill_weight with scene_coord requires rendered_map_position")
                    gt_scene_for_corr = normalize_scene_coord_map(
                        rendered_position,
                        scene_center,
                        scene_scale,
                        target_hw=rendered_fine.shape[-2:],
                    )
                    corr_target_feat = augment_feature_with_scene_coord(
                        rendered_fine,
                        gt_scene_for_corr,
                        query_corr_scene_coord_weight,
                    )
                elif query_local_corr_projector is not None:
                    corr_target_feat = _project_corr_feature(corr_target_feat)
                query_corr_distill_loss, query_corr_distill_metrics = local_correlation_distribution_loss(
                    corr_rendered_feat,
                    corr_query_feat,
                    corr_valid,
                    target_feat=corr_target_feat,
                    radius=query_corr_radius,
                    student_temperature=query_corr_distill_temperature,
                    target_temperature=query_corr_distill_target_temperature,
                    feature_preprocess=query_corr_feature_preprocess,
                    highpass_kernel=query_corr_highpass_kernel,
                    highpass_scale=query_corr_highpass_scale,
                )
                query_corr_metrics.update(query_corr_distill_metrics)

    query_identity_corr_ce_loss = zero
    query_identity_corr_subpx_loss = zero
    query_identity_corr_flow_loss = zero
    query_identity_corr_peak_margin_loss = zero
    query_identity_corr_distill_loss = zero
    query_identity_corr_metrics = {}
    if (
        query_identity_corr_ce_weight > 0
        or query_identity_corr_subpixel_weight > 0
        or query_identity_corr_flow_weight > 0
        or query_identity_corr_peak_margin_weight > 0
        or query_identity_corr_distill_weight > 0
    ):
        identity_valid = _resize_mask(mask, rendered_fine.shape[-2:])
        if identity_valid is None:
            identity_valid = torch.ones(
                pred_local_fine_target.shape[0],
                1,
                rendered_fine.shape[-2],
                rendered_fine.shape[-1],
                device=device,
                dtype=pred_local_fine_target.dtype,
            )
        identity_flow = pred_local_fine_target.new_zeros(
            pred_local_fine_target.shape[0],
            2,
            rendered_fine.shape[-2],
            rendered_fine.shape[-1],
        )
        identity_depth = _resize_feature(rendered_depth, rendered_fine.shape[-2:])
        if (
            query_identity_corr_ce_weight > 0
            or query_identity_corr_subpixel_weight > 0
            or query_identity_corr_flow_weight > 0
            or query_identity_corr_peak_margin_weight > 0
        ):
            identity_corr = local_correlation_joint_losses(
                _project_corr_feature(rendered_fine),
                _project_corr_feature(pred_local_fine_target, is_query=True),
                identity_flow,
                identity_valid,
                radius=query_identity_corr_radius,
                temperature=query_identity_corr_temperature,
                ce_temperature=query_identity_corr_ce_temperature,
                huber_delta=query_corr_huber_delta,
                peak_margin=query_identity_corr_peak_margin,
                compute_ce=query_identity_corr_ce_weight > 0,
                compute_subpixel=query_identity_corr_subpixel_weight > 0,
                compute_flow=query_identity_corr_flow_weight > 0,
                compute_peak=query_identity_corr_peak_margin_weight > 0,
                matcher=query_local_matcher,
                depth=identity_depth,
                flow_decode_mode=query_corr_flow_decode_mode,
                feature_preprocess=query_identity_corr_feature_preprocess,
                highpass_kernel=query_identity_corr_highpass_kernel,
                highpass_scale=query_identity_corr_highpass_scale,
            )
            query_identity_corr_ce_loss = identity_corr["losses"]["ce"]
            query_identity_corr_subpx_loss = identity_corr["losses"]["subpixel"]
            query_identity_corr_flow_loss = identity_corr["losses"]["flow"]
            query_identity_corr_peak_margin_loss = identity_corr["losses"]["peak"]
            query_identity_corr_metrics = prefix_metric_keys(
                identity_corr["metrics"],
                "map_query_corr_",
                "map_query_identity_corr_",
            )
        if query_identity_corr_distill_weight > 0:
            query_identity_corr_distill_loss, identity_distill_metrics = local_correlation_distribution_loss(
                _project_corr_feature(rendered_fine),
                _project_corr_feature(pred_local_fine_target, is_query=True),
                identity_valid,
                target_feat=_project_corr_feature(rendered_fine),
                radius=query_identity_corr_radius,
                student_temperature=query_identity_corr_distill_temperature,
                target_temperature=query_identity_corr_distill_target_temperature,
                feature_preprocess=query_identity_corr_feature_preprocess,
                highpass_kernel=query_identity_corr_highpass_kernel,
                highpass_scale=query_identity_corr_highpass_scale,
            )
            query_identity_corr_metrics.update(
                prefix_metric_keys(
                    identity_distill_metrics,
                    "map_query_corr_",
                    "map_query_identity_corr_",
                )
            )

    query_flow_warp_loss = zero
    query_flow_warp_metrics = {}
    query_flow_warp_contrastive_loss = zero
    query_flow_warp_contrastive_metrics = {}
    if query_flow_warp_weight > 0 or query_flow_warp_contrastive_weight > 0:
        neg_fine_for_warp = batch.get("rendered_map_fine_neg")
        neg_mask_for_warp = batch.get("rendered_map_mask_neg")
        neg_depth_for_warp = batch.get("rendered_map_depth_neg")
        flow_neg_to_gt = batch.get("rendered_map_flow_neg_to_gt")
        flow_valid_neg_to_gt = batch.get("rendered_map_flow_valid_neg_to_gt")
        if (
            neg_fine_for_warp is not None
            and flow_neg_to_gt is not None
            and flow_valid_neg_to_gt is not None
        ):
            warp_valid = flow_valid_neg_to_gt.float()
            if warp_valid.ndim == 3:
                warp_valid = warp_valid.unsqueeze(1)
            if neg_mask_for_warp is not None:
                neg_mask_w = neg_mask_for_warp.float()
                if neg_mask_w.shape[-2:] != warp_valid.shape[-2:]:
                    neg_mask_w = F.interpolate(neg_mask_w, size=warp_valid.shape[-2:], mode="nearest")
                warp_valid = warp_valid * neg_mask_w
            if prior_mask is not None:
                prior_for_warp = prior_mask.float()
                if prior_for_warp.shape[-2:] != warp_valid.shape[-2:]:
                    prior_for_warp = F.interpolate(prior_for_warp, size=warp_valid.shape[-2:], mode="nearest")
                warp_valid = warp_valid * prior_for_warp
            if neg_depth_for_warp is not None and depth_weight_strength > 0:
                warp_valid = warp_valid * depth_observability_weight(
                    neg_depth_for_warp,
                    mask=warp_valid,
                    strength=depth_weight_strength,
                    power=float(map_cfg.get("depth_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("depth_observability_max", 4.0)),
                )
            if (
                neg_depth_for_warp is not None
                and rendered_intrinsics is not None
                and trans_obs_strength > 0
            ):
                warp_valid = warp_valid * translation_observability_weight(
                    neg_depth_for_warp,
                    rendered_intrinsics,
                    mask=warp_valid,
                    strength=trans_obs_strength,
                    mode=str(map_cfg.get("translation_observability_mode", "xyz")),
                    power=float(map_cfg.get("translation_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("translation_observability_max", 4.0)),
                )
            if query_flow_warp_weight > 0:
                query_flow_warp_loss, query_flow_warp_metrics = flow_warp_feature_alignment_loss(
                    neg_fine_for_warp,
                    pred_local_fine_target,
                    flow_neg_to_gt,
                    warp_valid,
                )
            if query_flow_warp_contrastive_weight > 0:
                (
                    query_flow_warp_contrastive_loss,
                    query_flow_warp_contrastive_metrics,
                ) = flow_warp_contrastive_loss(
                    neg_fine_for_warp,
                    pred_local_fine_target,
                    flow_neg_to_gt,
                    warp_valid,
                    margin=query_flow_warp_contrastive_margin,
                    offsets=flow_warp_contrastive_offsets,
                )

    map_self_flow_warp_loss = zero
    map_self_flow_warp_metrics = {}
    map_self_flow_warp_contrastive_loss = zero
    map_self_flow_warp_contrastive_metrics = {}
    map_self_corr_ce_loss = zero
    map_self_corr_subpx_loss = zero
    map_self_corr_flow_loss = zero
    map_self_corr_peak_margin_loss = zero
    map_self_corr_metrics = {}
    if (
        map_self_flow_warp_weight > 0
        or map_self_flow_warp_contrastive_weight > 0
        or map_self_corr_ce_weight > 0
        or map_self_corr_subpixel_weight > 0
        or map_self_corr_flow_weight > 0
        or map_self_corr_peak_margin_weight > 0
    ):
        neg_fine_for_self = batch.get("rendered_map_fine_neg")
        neg_mask_for_self = batch.get("rendered_map_mask_neg")
        neg_depth_for_self = batch.get("rendered_map_depth_neg")
        flow_neg_to_gt = batch.get("rendered_map_flow_neg_to_gt")
        flow_valid_neg_to_gt = batch.get("rendered_map_flow_valid_neg_to_gt")
        if (
            neg_fine_for_self is not None
            and flow_neg_to_gt is not None
            and flow_valid_neg_to_gt is not None
        ):
            self_valid = flow_valid_neg_to_gt.float()
            if self_valid.ndim == 3:
                self_valid = self_valid.unsqueeze(1)
            if neg_mask_for_self is not None:
                neg_mask_s = neg_mask_for_self.float()
                if neg_mask_s.shape[-2:] != self_valid.shape[-2:]:
                    neg_mask_s = F.interpolate(neg_mask_s, size=self_valid.shape[-2:], mode="nearest")
                self_valid = self_valid * neg_mask_s
            if prior_mask is not None:
                prior_for_self = prior_mask.float()
                if prior_for_self.shape[-2:] != self_valid.shape[-2:]:
                    prior_for_self = F.interpolate(prior_for_self, size=self_valid.shape[-2:], mode="nearest")
                self_valid = self_valid * prior_for_self
            if neg_depth_for_self is not None and depth_weight_strength > 0:
                self_valid = self_valid * depth_observability_weight(
                    neg_depth_for_self,
                    mask=self_valid,
                    strength=depth_weight_strength,
                    power=float(map_cfg.get("depth_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("depth_observability_max", 4.0)),
                )
            if (
                neg_depth_for_self is not None
                and rendered_intrinsics is not None
                and trans_obs_strength > 0
            ):
                self_valid = self_valid * translation_observability_weight(
                    neg_depth_for_self,
                    rendered_intrinsics,
                    mask=self_valid,
                    strength=trans_obs_strength,
                    mode=str(map_cfg.get("translation_observability_mode", "xyz")),
                    power=float(map_cfg.get("translation_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("translation_observability_max", 4.0)),
                )
            if (
                map_self_corr_ce_weight > 0
                or map_self_corr_subpixel_weight > 0
                or map_self_corr_flow_weight > 0
                or map_self_corr_peak_margin_weight > 0
            ):
                self_joint_corr = local_correlation_joint_losses(
                    _project_corr_feature(neg_fine_for_self),
                    _project_corr_feature(
                        rendered_fine,
                        is_query=bool(map_cfg.get("map_self_corr_target_uses_query_projector", False)),
                    ),
                    flow_neg_to_gt,
                    self_valid,
                    radius=query_corr_radius,
                    temperature=query_corr_temperature,
                    huber_delta=query_corr_huber_delta,
                    peak_margin=query_corr_peak_margin,
                    compute_ce=map_self_corr_ce_weight > 0,
                    compute_subpixel=map_self_corr_subpixel_weight > 0,
                    compute_flow=map_self_corr_flow_weight > 0,
                    compute_peak=map_self_corr_peak_margin_weight > 0,
                )
                map_self_corr_ce_loss = self_joint_corr["losses"]["ce"]
                map_self_corr_subpx_loss = self_joint_corr["losses"]["subpixel"]
                map_self_corr_flow_loss = self_joint_corr["losses"]["flow"]
                map_self_corr_peak_margin_loss = self_joint_corr["losses"]["peak"]
                map_self_corr_metrics.update(
                    prefix_metric_keys(self_joint_corr["metrics"], "map_query_", "map_self_")
                )
            if map_self_flow_warp_weight > 0:
                map_self_flow_warp_loss, self_warp_metrics = flow_warp_feature_alignment_loss(
                    neg_fine_for_self,
                    rendered_fine,
                    flow_neg_to_gt,
                    self_valid,
                )
                map_self_flow_warp_metrics = prefix_metric_keys(
                    self_warp_metrics,
                    "map_query_",
                    "map_self_",
                )
            if map_self_flow_warp_contrastive_weight > 0:
                (
                    map_self_flow_warp_contrastive_loss,
                    self_contrast_metrics,
                ) = flow_warp_contrastive_loss(
                    neg_fine_for_self,
                    rendered_fine,
                    flow_neg_to_gt,
                    self_valid,
                    margin=query_flow_warp_contrastive_margin,
                    offsets=flow_warp_contrastive_offsets,
                )
                map_self_flow_warp_contrastive_metrics = prefix_metric_keys(
                    self_contrast_metrics,
                    "map_query_",
                    "map_self_",
                )

    map_self_feature_metric_pose_loss = zero
    map_self_feature_metric_pose_metrics = {}
    if map_self_feature_metric_pose_weight > 0:
        neg_fine_for_self_fm = batch.get("rendered_map_fine_neg")
        neg_mask_for_self_fm = batch.get("rendered_map_mask_neg")
        neg_depth_for_self_fm = batch.get("rendered_map_depth_neg")
        pose_neg = batch.get("rendered_map_pose_neg")
        pose_gt = batch.get("rendered_map_pose_gt")
        flow_valid_neg_to_gt = batch.get("rendered_map_flow_valid_neg_to_gt")
        if (
            neg_fine_for_self_fm is not None
            and neg_depth_for_self_fm is not None
            and pose_neg is not None
            and pose_gt is not None
            and rendered_intrinsics is not None
        ):
            if flow_valid_neg_to_gt is not None:
                self_fm_valid = flow_valid_neg_to_gt.float()
            else:
                self_fm_valid = (neg_depth_for_self_fm.float() > 0.05).float()
            if self_fm_valid.ndim == 3:
                self_fm_valid = self_fm_valid.unsqueeze(1)
            if neg_mask_for_self_fm is not None:
                neg_mask_sfm = neg_mask_for_self_fm.float()
                if neg_mask_sfm.shape[-2:] != self_fm_valid.shape[-2:]:
                    neg_mask_sfm = F.interpolate(neg_mask_sfm, size=self_fm_valid.shape[-2:], mode="nearest")
                self_fm_valid = self_fm_valid * neg_mask_sfm
            if prior_mask is not None:
                prior_for_self_fm = prior_mask.float()
                if prior_for_self_fm.shape[-2:] != self_fm_valid.shape[-2:]:
                    prior_for_self_fm = F.interpolate(prior_for_self_fm, size=self_fm_valid.shape[-2:], mode="nearest")
                self_fm_valid = self_fm_valid * prior_for_self_fm
            if depth_weight_strength > 0:
                self_fm_valid = self_fm_valid * depth_observability_weight(
                    neg_depth_for_self_fm,
                    mask=self_fm_valid,
                    strength=depth_weight_strength,
                    power=float(map_cfg.get("depth_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("depth_observability_max", 4.0)),
                )
            if trans_obs_strength > 0:
                self_fm_valid = self_fm_valid * translation_observability_weight(
                    neg_depth_for_self_fm,
                    rendered_intrinsics,
                    mask=self_fm_valid,
                    strength=trans_obs_strength,
                    mode=str(map_cfg.get("translation_observability_mode", "xyz")),
                    power=float(map_cfg.get("translation_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("translation_observability_max", 4.0)),
                )
            map_self_feature_metric_pose_loss, self_fm_metrics = feature_metric_localization_loss(
                rendered_fine,
                neg_fine_for_self_fm,
                neg_depth_for_self_fm,
                pose_neg,
                pose_gt,
                rendered_intrinsics,
                valid_mask=self_fm_valid,
                damping=feature_metric_pose_damping,
                normalize_features=feature_metric_pose_normalize,
                update_scale=feature_metric_pose_update_scale,
                rot_weight=feature_metric_pose_rot_weight,
                trans_weight=feature_metric_pose_trans_weight,
                rot_damping_multiplier=feature_metric_pose_rot_damping_multiplier,
            )
            map_self_feature_metric_pose_metrics = prefix_metric_keys(
                self_fm_metrics,
                "map_feature_metric_",
                "map_self_feature_metric_",
            )

    feature_metric_pose_loss = zero
    feature_metric_pose_metrics = {}
    if feature_metric_pose_weight > 0 or feature_metric_gradient_direction_weight > 0:
        neg_fine_for_fm = batch.get("rendered_map_fine_neg")
        neg_mask_for_fm = batch.get("rendered_map_mask_neg")
        neg_depth_for_fm = batch.get("rendered_map_depth_neg")
        neg_position_for_fm = batch.get("rendered_map_position_neg")
        pose_neg = batch.get("rendered_map_pose_neg")
        pose_gt = batch.get("rendered_map_pose_gt")
        flow_valid_neg_to_gt = batch.get("rendered_map_flow_valid_neg_to_gt")
        if (
            neg_fine_for_fm is not None
            and neg_depth_for_fm is not None
            and pose_neg is not None
            and pose_gt is not None
            and rendered_intrinsics is not None
        ):
            if flow_valid_neg_to_gt is not None:
                fm_valid = flow_valid_neg_to_gt.float()
            else:
                fm_valid = (neg_depth_for_fm.float() > 0.05).float()
            if fm_valid.ndim == 3:
                fm_valid = fm_valid.unsqueeze(1)
            if neg_mask_for_fm is not None:
                neg_mask_f = neg_mask_for_fm.float()
                if neg_mask_f.shape[-2:] != fm_valid.shape[-2:]:
                    neg_mask_f = F.interpolate(neg_mask_f, size=fm_valid.shape[-2:], mode="nearest")
                fm_valid = fm_valid * neg_mask_f
            if prior_mask is not None:
                prior_for_fm = prior_mask.float()
                if prior_for_fm.shape[-2:] != fm_valid.shape[-2:]:
                    prior_for_fm = F.interpolate(prior_for_fm, size=fm_valid.shape[-2:], mode="nearest")
                fm_valid = fm_valid * prior_for_fm
            if depth_weight_strength > 0:
                fm_valid = fm_valid * depth_observability_weight(
                    neg_depth_for_fm,
                    mask=fm_valid,
                    strength=depth_weight_strength,
                    power=float(map_cfg.get("depth_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("depth_observability_max", 4.0)),
                )
            if trans_obs_strength > 0:
                fm_valid = fm_valid * translation_observability_weight(
                    neg_depth_for_fm,
                    rendered_intrinsics,
                    mask=fm_valid,
                    strength=trans_obs_strength,
                    mode=str(map_cfg.get("translation_observability_mode", "xyz")),
                    power=float(map_cfg.get("translation_observability_power", 1.0)),
                    max_weight=float(map_cfg.get("translation_observability_max", 4.0)),
                )
            fm_query_feat = pred_local_fine_target
            fm_rendered_feat = neg_fine_for_fm
            if feature_metric_scene_coord_weight > 0:
                if pred_scene_coord is None:
                    raise KeyError("feature_metric_scene_coord_weight > 0 requires model.scene_coord_head=true")
                if neg_position_for_fm is None:
                    raise KeyError("feature_metric_scene_coord_weight requires rendered_map_position_neg")
                neg_scene_for_fm = normalize_scene_coord_map(
                    neg_position_for_fm,
                    scene_center,
                    scene_scale,
                    target_hw=neg_fine_for_fm.shape[-2:],
                )
                fm_query_feat = augment_feature_with_scene_coord(
                    fm_query_feat,
                    pred_scene_coord,
                    feature_metric_scene_coord_weight,
                )
                fm_rendered_feat = augment_feature_with_scene_coord(
                    fm_rendered_feat,
                    neg_scene_for_fm,
                    feature_metric_scene_coord_weight,
                )
            elif feature_metric_use_projector and query_local_corr_projector is not None:
                fm_query_feat = _project_corr_feature(fm_query_feat, is_query=True)
                fm_rendered_feat = _project_corr_feature(fm_rendered_feat)
            feature_metric_pose_loss, feature_metric_pose_metrics = feature_metric_localization_loss(
                fm_query_feat,
                fm_rendered_feat,
                neg_depth_for_fm,
                pose_neg,
                pose_gt,
                rendered_intrinsics,
                valid_mask=fm_valid,
                damping=feature_metric_pose_damping,
                normalize_features=feature_metric_pose_normalize,
                update_scale=feature_metric_pose_update_scale,
                rot_weight=feature_metric_pose_rot_weight,
                trans_weight=feature_metric_pose_trans_weight,
                rot_damping_multiplier=feature_metric_pose_rot_damping_multiplier,
            )

    feature_metric_gradient_direction_loss_val = zero
    feature_metric_gradient_direction_metrics = {}
    if feature_metric_gradient_direction_weight > 0:
        try:
            _neg_fine = neg_fine_for_fm
        except UnboundLocalError:
            _neg_fine = None
        if (
            _neg_fine is not None
            and neg_depth_for_fm is not None
            and pose_neg is not None
            and pose_gt is not None
            and rendered_intrinsics is not None
        ):
            fm_gd_query = pred_local_fine_target
            fm_gd_rendered = _neg_fine
            fm_gd_valid = fm_valid if fm_valid is not None else (neg_depth_for_fm.float() > 0.05).float()
            if fm_gd_valid.ndim == 3:
                fm_gd_valid = fm_gd_valid.unsqueeze(1)
            feature_metric_gradient_direction_loss_val, feature_metric_gradient_direction_metrics = \
                feature_metric_gradient_direction_loss(
                    fm_gd_query,
                    fm_gd_rendered,
                    neg_depth_for_fm,
                    pose_neg,
                    pose_gt,
                    rendered_intrinsics,
                    valid_mask=fm_gd_valid,
                    rot_weight=feature_metric_gradient_direction_rot_weight,
                    trans_weight=feature_metric_gradient_direction_trans_weight,
                )

    alpha_coverage_loss = zero
    if alpha is not None and alpha_coverage_weight > 0:
        alpha_target = float(map_cfg.get("alpha_target", 0.9))
        alpha_gap = F.relu(alpha_target - alpha.float())
        if prior_mask is not None and bool(map_cfg.get("alpha_use_prior_mask", True)):
            alpha_gap = alpha_gap * prior_mask
            alpha_coverage_loss = alpha_gap.sum() / torch.clamp(prior_mask.sum(), min=1.0)
        else:
            alpha_coverage_loss = alpha_gap.mean()
        alpha_coverage_loss = alpha_coverage_loss * alpha_coverage_weight

    rgb_reconstruction_loss = zero
    if rendered_rgb is not None and rgb_l1_weight > 0:
        query_rgb = F.interpolate(
            batch["rgb"].float(),
            size=rendered_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        rgb_loss_mask_mode = str(map_cfg.get("rgb_loss_mask", "all")).lower()
        if rgb_loss_mask_mode == "alpha":
            rgb_mask = mask
            rgb_reconstruction_loss = (
                (rendered_rgb.float() - query_rgb).abs() * rgb_mask
            ).sum() / torch.clamp(rgb_mask.sum() * rendered_rgb.shape[1], min=1.0)
        elif rgb_loss_mask_mode == "prior" and prior_mask is not None:
            rgb_reconstruction_loss = (
                (rendered_rgb.float() - query_rgb).abs() * prior_mask
            ).sum() / torch.clamp(prior_mask.sum() * rendered_rgb.shape[1], min=1.0)
        elif rgb_loss_mask_mode in {"alpha_prior", "prior_alpha"} and prior_mask is not None:
            rgb_reconstruction_loss = (
                (rendered_rgb.float() - query_rgb).abs() * mask
            ).sum() / torch.clamp(mask.sum() * rendered_rgb.shape[1], min=1.0)
        else:
            rgb_reconstruction_loss = F.l1_loss(rendered_rgb.float(), query_rgb)
        rgb_reconstruction_loss = rgb_reconstruction_loss * rgb_l1_weight

    perturb_rank_weight = float(map_cfg.get("perturb_rank_weight", 0.0))
    perturb_rank_loss = zero
    perturb_margin = torch.tensor(0.0, device=device)
    if perturb_rank_weight > 0:
        perturb_margin = resolve_perturb_rank_margin(map_cfg, batch, device)
        neg_fine = batch.get("rendered_map_fine_neg")
        neg_coarse = batch.get("rendered_map_coarse_neg")
        neg_mask = batch.get("rendered_map_mask_neg")
        if neg_fine is None or neg_coarse is None:
            max_shift_px = max(0, int(map_cfg.get("perturb_max_shift_px", 2)))
            if max_shift_px > 0:
                shift_x = random.randint(-max_shift_px, max_shift_px)
                shift_y = random.randint(-max_shift_px, max_shift_px)
                if shift_x == 0 and shift_y == 0:
                    shift_x = 1
                neg_fine = torch.roll(rendered_fine, shifts=(shift_y, shift_x), dims=(2, 3))
                neg_coarse = torch.roll(rendered_coarse, shifts=(shift_y, shift_x), dims=(2, 3))
                neg_mask = torch.roll(mask, shifts=(shift_y, shift_x), dims=(2, 3)) if mask is not None else None
        elif prior_mask is not None:
            neg_mask = neg_mask * prior_mask
        if neg_fine is not None and neg_coarse is not None:
            neg_fine_query = _resize_feature(neg_fine, pred_local_fine_target.shape[-2:])
            neg_coarse_query = _resize_feature(neg_coarse, pred_coarse_target.shape[-2:])
            neg_fine_mask = _resize_mask(neg_mask, pred_local_fine_target.shape[-2:])
            neg_coarse_mask = _resize_mask(neg_mask, pred_coarse_target.shape[-2:])
            pos_fine = l1_feature_loss(
                pred_local_fine_target,
                rendered_fine_local_query,
                local_fine_query_mask,
            ) + cosine_loss(
                pred_local_fine_target,
                rendered_fine_local_query,
                local_fine_query_mask,
            )
            neg_fine_loss = l1_feature_loss(
                pred_local_fine_target,
                neg_fine_query,
                neg_fine_mask,
            ) + cosine_loss(
                pred_local_fine_target,
                neg_fine_query,
                neg_fine_mask,
            )
            pos_coarse = l1_feature_loss(pred_coarse_target, rendered_coarse_query, coarse_query_mask) + cosine_loss(
                pred_coarse_target, rendered_coarse_query, coarse_query_mask
            )
            neg_coarse_loss = l1_feature_loss(pred_coarse_target, neg_coarse_query, neg_coarse_mask) + cosine_loss(
                pred_coarse_target, neg_coarse_query, neg_coarse_mask
            )
            perturb_rank_loss = F.relu(
                perturb_margin + 0.5 * (pos_fine + pos_coarse) - 0.5 * (neg_fine_loss + neg_coarse_loss)
            )
    total = (
        query_fine_loss
        + query_fine_raw_loss
        + query_local_fine_loss
        + query_coarse_loss
        + coarse_pose_rank_weight * coarse_pose_rank_loss
        + coarse_pose_energy_weight * coarse_pose_energy_loss
        + coarse_pose_local_energy_weight * coarse_pose_local_energy_loss
        + candidate_render_score_weight * candidate_render_score_loss
        + candidate_score_fusion_weight * candidate_score_fusion_loss
        + candidate_refined_pose_weight * candidate_refined_pose_loss_value
        + query_fine_nce_loss
        + query_local_fine_nce_loss
        + query_coarse_nce_loss
        + query_fine_grad_loss
        + query_local_fine_grad_loss
        + query_coarse_grad_loss
        + rendered_teacher_fine_loss
        + rendered_teacher_fine_raw_loss
        + rendered_teacher_coarse_loss
        + rendered_teacher_fine_nce_loss
        + rendered_teacher_coarse_nce_loss
        + rendered_teacher_fine_grad_loss
        + rendered_teacher_coarse_grad_loss
        + query_projected_fine_weight * query_projected_fine_loss
        + map_fine_coarse_ortho_loss
        + variance_weight_query * query_variance_loss
        + variance_weight_map * map_variance_loss
        + covariance_weight_query * query_covariance_loss
        + covariance_weight_map * map_covariance_loss
        + query_scene_coord_weight * query_scene_coord_loss
        + query_scene_coord_warp_weight * query_scene_coord_warp_loss
        + teacher_corr_weight * teacher_corr_loss
        + teacher_corr_local_patch_weight * teacher_corr_local_patch_loss
        + query_corr_ce_weight * query_corr_ce_loss
        + query_corr_subpixel_weight * query_corr_subpx_loss
        + query_corr_flow_weight * query_corr_flow_loss
        + query_corr_flow_cosine_weight * query_corr_flow_cosine_loss
        + query_corr_flow_head_conf_weight * query_corr_flow_conf_loss
        + query_corr_peak_margin_weight * query_corr_peak_margin_loss
        + query_corr_distill_weight * query_corr_distill_loss
        + query_corr_wls_pose_weight * query_corr_wls_pose_loss
        + query_corr_pose_gain_weight * query_corr_pose_gain_loss
        + query_identity_corr_ce_weight * query_identity_corr_ce_loss
        + query_identity_corr_subpixel_weight * query_identity_corr_subpx_loss
        + query_identity_corr_flow_weight * query_identity_corr_flow_loss
        + query_identity_corr_peak_margin_weight * query_identity_corr_peak_margin_loss
        + query_identity_corr_distill_weight * query_identity_corr_distill_loss
        + query_flow_warp_weight * query_flow_warp_loss
        + query_flow_warp_contrastive_weight * query_flow_warp_contrastive_loss
        + map_self_corr_ce_weight * map_self_corr_ce_loss
        + map_self_corr_subpixel_weight * map_self_corr_subpx_loss
        + map_self_corr_flow_weight * map_self_corr_flow_loss
        + map_self_corr_peak_margin_weight * map_self_corr_peak_margin_loss
        + map_self_flow_warp_weight * map_self_flow_warp_loss
        + map_self_flow_warp_contrastive_weight * map_self_flow_warp_contrastive_loss
        + map_self_feature_metric_pose_weight * map_self_feature_metric_pose_loss
        + feature_metric_pose_weight * feature_metric_pose_loss
        + feature_metric_gradient_direction_weight * feature_metric_gradient_direction_loss_val
        + alpha_coverage_loss
        + rgb_reconstruction_loss
        + perturb_rank_weight * perturb_rank_loss
    )

    with torch.no_grad():
        fine_map_teacher_cos = 1.0 - cosine_loss(rendered_fine_teacher, teacher_fine, fine_teacher_mask)
        coarse_map_teacher_cos = 1.0 - cosine_loss(rendered_coarse_teacher, teacher_coarse, coarse_teacher_mask)
        fine_query_map_cos = 1.0 - cosine_loss(pred_fine, rendered_fine_query, fine_query_mask)
        local_fine_query_map_cos = 1.0 - cosine_loss(
            pred_local_fine,
            rendered_fine_local_query,
            local_fine_query_mask,
        )
        coarse_query_map_cos = 1.0 - cosine_loss(pred_coarse, rendered_coarse_query, coarse_query_mask)
        if rendered_fine_raw_query is not None:
            fine_map_raw_teacher_cos = 1.0 - cosine_loss(rendered_fine_raw_teacher, teacher_fine, fine_teacher_mask)
            fine_query_raw_map_cos = 1.0 - cosine_loss(pred_fine, rendered_fine_raw_query, fine_query_mask)
        else:
            fine_map_raw_teacher_cos = zero
            fine_query_raw_map_cos = zero
        if alpha is not None:
            alpha_float = alpha.float()
            map_alpha_mean = alpha_float.mean()
            map_alpha_coverage = (alpha_float > float(map_cfg.get("alpha_threshold", 0.5))).float().mean()
            if prior_mask is not None:
                alpha_binary = (alpha_float > float(map_cfg.get("alpha_threshold", 0.5))).float()
                map_alpha_coverage_valid = (alpha_binary * prior_mask).sum() / torch.clamp(prior_mask.sum(), min=1.0)
            else:
                map_alpha_coverage_valid = map_alpha_coverage
        else:
            map_alpha_mean = zero
            map_alpha_coverage = zero
            map_alpha_coverage_valid = zero

    return total, {
        "map_hook_active": torch.ones((), device=device),
        "map_hook_loss": total.detach(),
        "map_query_fine_loss": query_fine_loss.detach(),
        "map_query_fine_raw_loss": query_fine_raw_loss.detach(),
        "map_query_local_fine_loss": query_local_fine_loss.detach(),
        "map_query_coarse_loss": query_coarse_loss.detach(),
        **coarse_pose_rank_metrics,
        **coarse_pose_energy_metrics,
        **coarse_pose_local_energy_metrics,
        **candidate_render_score_metrics,
        **candidate_score_fusion_metrics,
        **candidate_refined_pose_metrics,
        "map_query_fine_nce_loss": query_fine_nce_loss.detach(),
        "map_query_local_fine_nce_loss": query_local_fine_nce_loss.detach(),
        "map_query_coarse_nce_loss": query_coarse_nce_loss.detach(),
        "map_query_fine_grad_loss": query_fine_grad_loss.detach(),
        "map_query_local_fine_grad_loss": query_local_fine_grad_loss.detach(),
        "map_query_coarse_grad_loss": query_coarse_grad_loss.detach(),
        "map_rendered_teacher_fine_loss": rendered_teacher_fine_loss.detach(),
        "map_rendered_teacher_fine_raw_loss": rendered_teacher_fine_raw_loss.detach(),
        "map_rendered_teacher_coarse_loss": rendered_teacher_coarse_loss.detach(),
        "map_rendered_teacher_fine_nce_loss": rendered_teacher_fine_nce_loss.detach(),
        "map_rendered_teacher_coarse_nce_loss": rendered_teacher_coarse_nce_loss.detach(),
        "map_rendered_teacher_fine_grad_loss": rendered_teacher_fine_grad_loss.detach(),
        "map_rendered_teacher_coarse_grad_loss": rendered_teacher_coarse_grad_loss.detach(),
        "map_query_projected_fine_loss": query_projected_fine_loss.detach(),
        "map_fine_coarse_ortho_loss": map_fine_coarse_ortho_loss.detach(),
        "map_query_variance_loss": query_variance_loss.detach(),
        "map_variance_loss": map_variance_loss.detach(),
        "map_query_covariance_loss": query_covariance_loss.detach(),
        "map_covariance_loss": map_covariance_loss.detach(),
        "map_query_scene_coord_loss": query_scene_coord_loss.detach(),
        "map_query_scene_coord_warp_loss": query_scene_coord_warp_loss.detach(),
        **query_scene_coord_metrics,
        **query_scene_coord_warp_metrics,
        "map_teacher_corr_weight": torch.tensor(teacher_corr_weight, device=device),
        "map_teacher_corr_weighted_loss": (teacher_corr_weight * teacher_corr_loss).detach(),
        "map_teacher_patch_weight": torch.tensor(teacher_corr_local_patch_weight, device=device),
        "map_teacher_patch_weighted_loss": (
            teacher_corr_local_patch_weight * teacher_corr_local_patch_loss
        ).detach(),
        **teacher_corr_metrics,
        "map_query_corr_ce_loss": query_corr_ce_loss.detach(),
        "map_query_corr_subpx_loss": query_corr_subpx_loss.detach(),
        "map_query_corr_flow_loss": query_corr_flow_loss.detach(),
        "map_query_corr_flow_cosine_loss": query_corr_flow_cosine_loss.detach(),
        "map_query_corr_flow_conf_loss": query_corr_flow_conf_loss.detach(),
        "map_query_corr_peak_margin_loss": query_corr_peak_margin_loss.detach(),
        "map_query_corr_distill_loss": query_corr_distill_loss.detach(),
        "map_query_corr_wls_pose_loss": query_corr_wls_pose_loss.detach(),
        "map_query_corr_pose_gain_loss": query_corr_pose_gain_loss.detach(),
        "map_query_corr_projector_active": torch.tensor(
            1.0 if query_local_corr_projector is not None else 0.0,
            device=device,
        ),
        **query_corr_metrics,
        **query_corr_wls_pose_metrics,
        "map_query_identity_corr_ce_loss": query_identity_corr_ce_loss.detach(),
        "map_query_identity_corr_subpx_loss": query_identity_corr_subpx_loss.detach(),
        "map_query_identity_corr_flow_loss": query_identity_corr_flow_loss.detach(),
        "map_query_identity_corr_peak_margin_loss": query_identity_corr_peak_margin_loss.detach(),
        "map_query_identity_corr_distill_loss": query_identity_corr_distill_loss.detach(),
        **query_identity_corr_metrics,
        "map_query_flow_warp_loss": query_flow_warp_loss.detach(),
        **query_flow_warp_metrics,
        "map_query_flow_warp_contrastive_loss": query_flow_warp_contrastive_loss.detach(),
        **query_flow_warp_contrastive_metrics,
        "map_self_corr_ce_loss": map_self_corr_ce_loss.detach(),
        "map_self_corr_subpx_loss": map_self_corr_subpx_loss.detach(),
        "map_self_corr_flow_loss": map_self_corr_flow_loss.detach(),
        "map_self_corr_peak_margin_loss": map_self_corr_peak_margin_loss.detach(),
        **map_self_corr_metrics,
        "map_self_flow_warp_loss": map_self_flow_warp_loss.detach(),
        **map_self_flow_warp_metrics,
        "map_self_flow_warp_contrastive_loss": map_self_flow_warp_contrastive_loss.detach(),
        **map_self_flow_warp_contrastive_metrics,
        "map_self_feature_metric_pose_loss": map_self_feature_metric_pose_loss.detach(),
        **map_self_feature_metric_pose_metrics,
        **feature_metric_pose_metrics,
        "map_feature_metric_pose_weighted_loss": (feature_metric_pose_weight * feature_metric_pose_loss).detach(),
        "map_self_feature_metric_pose_weighted_loss": (
            map_self_feature_metric_pose_weight * map_self_feature_metric_pose_loss
        ).detach(),
        "map_gradient_direction_loss": feature_metric_gradient_direction_loss_val.detach(),
        **feature_metric_gradient_direction_metrics,
        "map_alpha_coverage_loss": alpha_coverage_loss.detach(),
        "map_rgb_reconstruction_loss": rgb_reconstruction_loss.detach(),
        "map_perturb_rank_loss": perturb_rank_loss.detach(),
        "map_perturb_margin": perturb_margin.detach(),
        "map_query_fine_weight": torch.tensor(query_fine_weight, device=device),
        "map_query_fine_raw_weight": torch.tensor(query_fine_raw_weight, device=device),
        "map_query_local_fine_weight": torch.tensor(query_local_fine_weight, device=device),
        "map_rendered_teacher_fine_weight": torch.tensor(rendered_teacher_fine_weight, device=device),
        "map_rendered_teacher_fine_raw_weight": torch.tensor(
            rendered_teacher_fine_raw_weight, device=device
        ),
        "map_query_fine_infonce_weight": torch.tensor(query_fine_infonce_weight, device=device),
        "map_query_local_fine_infonce_weight": torch.tensor(
            query_local_fine_infonce_weight,
            device=device,
        ),
        "map_query_coarse_infonce_weight": torch.tensor(query_coarse_infonce_weight, device=device),
        "map_coarse_pose_rank_weight": torch.tensor(coarse_pose_rank_weight, device=device),
        "map_coarse_pose_energy_weight": torch.tensor(coarse_pose_energy_weight, device=device),
        "map_coarse_pose_local_energy_weight": torch.tensor(coarse_pose_local_energy_weight, device=device),
        "map_candidate_render_score_weight": torch.tensor(candidate_render_score_weight, device=device),
        "map_candidate_score_fusion_weight": torch.tensor(candidate_score_fusion_weight, device=device),
        "map_candidate_refined_pose_weight": torch.tensor(candidate_refined_pose_weight, device=device),
        "map_query_fine_grad_weight": torch.tensor(query_fine_grad_weight, device=device),
        "map_query_local_fine_grad_weight": torch.tensor(query_local_fine_grad_weight, device=device),
        "map_query_coarse_grad_weight": torch.tensor(query_coarse_grad_weight, device=device),
        "map_rendered_teacher_fine_infonce_weight": torch.tensor(
            rendered_teacher_fine_infonce_weight, device=device
        ),
        "map_rendered_teacher_coarse_infonce_weight": torch.tensor(
            rendered_teacher_coarse_infonce_weight, device=device
        ),
        "map_rendered_teacher_fine_grad_weight": torch.tensor(
            rendered_teacher_fine_grad_weight, device=device
        ),
        "map_rendered_teacher_coarse_grad_weight": torch.tensor(
            rendered_teacher_coarse_grad_weight, device=device
        ),
        "map_query_projected_fine_weight": torch.tensor(query_projected_fine_weight, device=device),
        "map_fine_coarse_ortho_weight": torch.tensor(fine_coarse_ortho_weight, device=device),
        "map_query_variance_weight": torch.tensor(variance_weight_query, device=device),
        "map_variance_weight": torch.tensor(variance_weight_map, device=device),
        "map_query_covariance_weight": torch.tensor(covariance_weight_query, device=device),
        "map_covariance_weight": torch.tensor(covariance_weight_map, device=device),
        "map_query_scene_coord_weight": torch.tensor(query_scene_coord_weight, device=device),
        "map_query_scene_coord_warp_weight": torch.tensor(query_scene_coord_warp_weight, device=device),
        "map_query_corr_scene_coord_weight": torch.tensor(query_corr_scene_coord_weight, device=device),
        "map_teacher_patch_radius": torch.tensor(float(teacher_corr_local_patch_radius), device=device),
        "map_feature_metric_scene_coord_weight": torch.tensor(feature_metric_scene_coord_weight, device=device),
        "map_feature_metric_use_projector": torch.tensor(
            1.0 if feature_metric_use_projector and query_local_corr_projector is not None else 0.0,
            device=device,
        ),
        "map_query_corr_ce_weight": torch.tensor(query_corr_ce_weight, device=device),
        "map_query_corr_subpixel_weight": torch.tensor(query_corr_subpixel_weight, device=device),
        "map_query_corr_flow_weight": torch.tensor(query_corr_flow_weight, device=device),
        "map_query_corr_flow_cosine_weight": torch.tensor(
            query_corr_flow_cosine_weight,
            device=device,
        ),
        "map_query_corr_flow_head_conf_weight": torch.tensor(
            query_corr_flow_head_conf_weight,
            device=device,
        ),
        "map_query_corr_peak_margin_weight": torch.tensor(query_corr_peak_margin_weight, device=device),
        "map_query_corr_distill_weight": torch.tensor(query_corr_distill_weight, device=device),
        "map_query_corr_wls_pose_weight": torch.tensor(query_corr_wls_pose_weight, device=device),
        "map_query_corr_pose_gain_weight": torch.tensor(query_corr_pose_gain_weight, device=device),
        "map_query_identity_corr_ce_weight": torch.tensor(query_identity_corr_ce_weight, device=device),
        "map_query_identity_corr_subpixel_weight": torch.tensor(
            query_identity_corr_subpixel_weight,
            device=device,
        ),
        "map_query_identity_corr_flow_weight": torch.tensor(query_identity_corr_flow_weight, device=device),
        "map_query_identity_corr_peak_margin_weight": torch.tensor(
            query_identity_corr_peak_margin_weight,
            device=device,
        ),
        "map_query_identity_corr_distill_weight": torch.tensor(
            query_identity_corr_distill_weight,
            device=device,
        ),
        "map_query_flow_warp_weight": torch.tensor(query_flow_warp_weight, device=device),
        "map_query_flow_warp_contrastive_weight": torch.tensor(
            query_flow_warp_contrastive_weight, device=device
        ),
        "map_self_corr_ce_weight": torch.tensor(map_self_corr_ce_weight, device=device),
        "map_self_corr_subpixel_weight": torch.tensor(map_self_corr_subpixel_weight, device=device),
        "map_self_corr_flow_weight": torch.tensor(map_self_corr_flow_weight, device=device),
        "map_self_corr_peak_margin_weight": torch.tensor(
            map_self_corr_peak_margin_weight,
            device=device,
        ),
        "map_self_flow_warp_weight": torch.tensor(map_self_flow_warp_weight, device=device),
        "map_self_flow_warp_contrastive_weight": torch.tensor(
            map_self_flow_warp_contrastive_weight, device=device
        ),
        "map_self_feature_metric_pose_weight": torch.tensor(
            map_self_feature_metric_pose_weight, device=device
        ),
        "map_feature_metric_pose_weight": torch.tensor(feature_metric_pose_weight, device=device),
        "map_alpha_coverage_weight": torch.tensor(alpha_coverage_weight, device=device),
        "map_rgb_l1_weight": torch.tensor(rgb_l1_weight, device=device),
        "map_teacher_fine_cosine": fine_map_teacher_cos.detach(),
        "map_teacher_fine_raw_cosine": fine_map_raw_teacher_cos.detach(),
        "map_teacher_coarse_cosine": coarse_map_teacher_cos.detach(),
        "map_query_fine_cosine": fine_query_map_cos.detach(),
        "map_query_local_fine_cosine": local_fine_query_map_cos.detach(),
        "map_query_fine_raw_cosine": fine_query_raw_map_cos.detach(),
        "map_query_projected_fine_cosine": query_projected_fine_cos.detach(),
        "map_query_coarse_cosine": coarse_query_map_cos.detach(),
        "map_alpha_mean": map_alpha_mean.detach(),
        "map_alpha_coverage": map_alpha_coverage.detach(),
        "map_alpha_coverage_valid": map_alpha_coverage_valid.detach(),
        "map_coarse_active": torch.tensor(float(coarse_active), device=device),
    }


def save_validation_visuals(batch, outputs, qual_dir, feature_track_root, step, limit, student_fine_key="fine"):
    if student_fine_key not in outputs:
        raise KeyError(f"student_fine_key={student_fine_key!r} not found in model outputs")
    limit = min(limit, batch["rgb"].shape[0])
    for idx in range(limit):
        sample_name = Path(batch["sample_name"][idx]).with_suffix("").as_posix().replace("/", "_")
        filename = f"step{step:06d}_{sample_name}.png"
        for root in [qual_dir, feature_track_root]:
            save_feature_track_visual(
                Path(root) / filename,
                query_rgb=batch["rgb"][idx].detach().cpu(),
                teacher_fine=batch["teacher_fine"][idx].detach().cpu(),
                student_fine=outputs[student_fine_key][idx].detach().cpu(),
                teacher_coarse=batch["teacher_coarse"][idx].detach().cpu(),
                student_coarse=outputs["coarse"][idx].detach().cpu(),
                rendered_map_fine_raw=batch.get("rendered_map_fine_raw", [None] * limit)[idx].detach().cpu()
                if "rendered_map_fine_raw" in batch
                else None,
                rendered_map_fine=batch.get("rendered_map_fine", [None] * limit)[idx].detach().cpu()
                if "rendered_map_fine" in batch
                else None,
                rendered_map_coarse=batch.get("rendered_map_coarse", [None] * limit)[idx].detach().cpu()
                if "rendered_map_coarse" in batch
                else None,
                rendered_map_mask=batch.get("rendered_map_mask", [None] * limit)[idx].detach().cpu()
                if "rendered_map_mask" in batch
                else None,
                rendered_map_alpha=batch.get("rendered_map_alpha", [None] * limit)[idx].detach().cpu()
                if "rendered_map_alpha" in batch
                else None,
                prior_mask=batch.get("prior_mask", [None] * limit)[idx].detach().cpu()
                if "prior_mask" in batch
                else None,
                sample_name=batch["sample_name"][idx],
            )


def mean_metrics(metric_list):
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    merged = {}
    for key in keys:
        values = []
        for metrics in metric_list:
            val = metrics[key]
            values.append(float(val.item() if isinstance(val, torch.Tensor) else val))
        merged[key] = sum(values) / max(1, len(values))
    return merged


def validation_selection_score(val_metrics, cfg):
    train_cfg = cfg.get("training", {})
    metric_name = str(train_cfg.get("best_metric", "loss_total"))
    mode = str(train_cfg.get("best_metric_mode", "min")).lower()
    if mode not in ("min", "max"):
        raise ValueError("training.best_metric_mode must be 'min' or 'max'")
    if metric_name not in val_metrics:
        metric_name = "loss_total"
    metric_value = float(val_metrics.get(metric_name, float("inf")))
    score = metric_value if mode == "min" else -metric_value
    return metric_name, metric_value, score


def validate(model, loader, cfg, device, qual_dir, feature_track_root, step, logger, map_renderer=None, epoch=0):
    model.eval()
    metrics = []
    saved_visuals = False
    use_amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    if map_renderer is not None:
        if map_renderer.has_trainable_params():
            map_renderer.clear_cache()
        map_renderer.set_train_mode(False)

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if map_renderer is not None:
                batch = map_renderer.attach_to_batch(batch, require_grad=False)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if bool(cfg["model"].get("teacher_fine_condition", False)):
                    outputs = model(batch["rgb"], teacher_fine=batch.get("teacher_fine"))
                else:
                    outputs = model(batch["rgb"])
                batch = maybe_attach_pose_candidate_renders(
                    batch,
                    outputs,
                    cfg,
                    map_renderer,
                    require_grad=False,
                    epoch=epoch,
                )
                main_total, batch_metrics = compute_main_losses(outputs, batch, cfg)
                map_total, map_metrics = compute_map_supervision(
                    batch,
                    outputs,
                    cfg,
                    device,
                    epoch=epoch,
                    local_matcher=getattr(model, "local_matcher", None),
                    local_flow_head=getattr(model, "local_flow_head", None),
                    local_corr_projector=getattr(model, "local_corr_projector", None),
                    candidate_score_fusion_head=getattr(model, "candidate_score_fusion_head", None),
                    map_renderer=map_renderer,
                )
                total = main_total + map_total
            batch_metrics.update(map_metrics)
            batch_metrics["loss_total"] = total.detach()
            metrics.append(batch_metrics)

            if not saved_visuals:
                save_validation_visuals(
                    batch,
                    outputs,
                    qual_dir=qual_dir,
                    feature_track_root=feature_track_root,
                    step=step,
                    limit=int(cfg["visualization"].get("num_val_vis", 4)),
                    student_fine_key=str(cfg.get("map_supervision", {}).get("query_fine_key", "fine")),
                )
                saved_visuals = True

    result = mean_metrics(metrics)
    log_msg = "Val step=%d total=%.4f fine_cos=%.4f coarse_cos=%.4f"
    log_args = [
        step,
        result.get("loss_total", 0.0),
        result.get("fine_cosine", 0.0),
        result.get("coarse_cosine", 0.0),
    ]
    if "retrieval_cosine" in result:
        log_msg += " retrieval_cos=%.4f"
        log_args.append(result.get("retrieval_cosine", 0.0))
    log_msg += " map_hook=%.4f"
    log_args.append(result.get("map_hook_active", 0.0))
    if "map_query_fine_cosine" in result:
        log_msg += " map_q_f=%.4f map_q_loc=%.4f map_q_c=%.4f map_t_f=%.4f map_t_c=%.4f"
        log_args.extend(
            [
                result.get("map_query_fine_cosine", 0.0),
                result.get("map_query_local_fine_cosine", 0.0),
                result.get("map_query_coarse_cosine", 0.0),
                result.get("map_teacher_fine_cosine", 0.0),
                result.get("map_teacher_coarse_cosine", 0.0),
            ]
        )
    if "map_teacher_fine_raw_cosine" in result:
        log_msg += " map_q_f_raw=%.4f map_t_f_raw=%.4f"
        log_args.extend(
            [
                result.get("map_query_fine_raw_cosine", 0.0),
                result.get("map_teacher_fine_raw_cosine", 0.0),
            ]
        )
    if "map_query_projected_fine_cosine" in result:
        log_msg += " qproj=%.4f"
        log_args.append(result.get("map_query_projected_fine_cosine", 0.0))
    if "map_coarse_pose_rank_gap" in result:
        log_msg += " c_rank=%.4f/%.3f"
        log_args.extend(
            [
                result.get("map_coarse_pose_rank_gap", 0.0),
                result.get("map_coarse_pose_rank_acc", 0.0),
            ]
        )
    if "map_coarse_pose_energy_gap" in result:
        log_msg += " c_energy=%.4f/%.3f"
        log_args.extend(
            [
                result.get("map_coarse_pose_energy_gap", 0.0),
                result.get("map_coarse_pose_energy_acc", 0.0),
            ]
        )
    if "map_coarse_pose_local_energy_gap" in result:
        log_msg += " c_local=%.4f/%.3f"
        log_args.extend(
            [
                result.get("map_coarse_pose_local_energy_gap", 0.0),
                result.get("map_coarse_pose_local_energy_acc", 0.0),
            ]
        )
    if "map_candidate_render_score_acc" in result:
        log_msg += " cand=%.3f/%.4f"
        log_args.extend(
            [
                result.get("map_candidate_render_score_acc", 0.0),
                result.get("map_candidate_render_score_margin", 0.0),
            ]
        )
    if "map_candidate_score_fusion_acc" in result:
        log_msg += " cfuse=%.3f/%.4f/%.0fmm"
        log_args.extend(
            [
                result.get("map_candidate_score_fusion_acc", 0.0),
                result.get("map_candidate_score_fusion_margin", 0.0),
                result.get("map_candidate_score_fusion_pred_trans_mm", 0.0),
            ]
        )
        if "map_candidate_score_fusion_top4_basin_recall" in result:
            log_msg += " basin@1/4/8=%.3f/%.3f/%.3f oracle=%.3f"
            log_args.extend(
                [
                    result.get("map_candidate_score_fusion_top1_basin_recall", 0.0),
                    result.get("map_candidate_score_fusion_top4_basin_recall", 0.0),
                    result.get("map_candidate_score_fusion_top8_basin_recall", 0.0),
                    result.get("map_candidate_score_fusion_oracle_basin_recall", 0.0),
                ]
            )
        if result.get("map_candidate_score_fusion_cost_regression_loss", 0.0) > 0:
            log_msg += " cfreg=%.3f"
            log_args.append(result.get("map_candidate_score_fusion_cost_regression_loss", 0.0))
        if result.get("map_candidate_score_fusion_pairwise_rank_loss", 0.0) > 0:
            log_msg += " cfrank=%.3f/%.3f"
            log_args.extend(
                [
                    result.get("map_candidate_score_fusion_pairwise_rank_loss", 0.0),
                    result.get("map_candidate_score_fusion_pairwise_rank_acc", 0.0),
                ]
            )
    if "map_candidate_refined_pose_trans_mm" in result:
        log_msg += " cref=%.0fmm init=%.0fmm"
        log_args.extend(
            [
                result.get("map_candidate_refined_pose_trans_mm", 0.0),
                result.get("map_candidate_refined_pose_selected_init_trans_mm", 0.0),
            ]
        )
    if "map_alpha_coverage" in result:
        log_msg += " alpha_cov=%.4f alpha_cov_valid=%.4f alpha_mean=%.4f"
        log_args.extend(
            [
                result.get("map_alpha_coverage", 0.0),
                result.get("map_alpha_coverage_valid", result.get("map_alpha_coverage", 0.0)),
                result.get("map_alpha_mean", 0.0),
            ]
        )
    if "map_query_corr_subpx_flow_epe" in result:
        log_msg += " corr_epe=%.3f corr_acc=%.3f corr_cov=%.3f"
        log_args.extend(
            [
                result.get("map_query_corr_subpx_flow_epe", 0.0),
                result.get("map_query_corr_subpx_acc", 0.0),
                result.get("map_query_corr_subpx_cov", 0.0),
            ]
        )
    if "map_query_corr_pred_flow_mag_px" in result:
        log_msg += " flow=%.2f/%.2f fcos=%.3f src=%.0f"
        log_args.extend(
            [
                result.get("map_query_corr_pred_flow_mag_px", 0.0),
                result.get("map_query_corr_gt_flow_mag_px", 0.0),
                result.get("map_query_corr_flow_cosine", 0.0),
                result.get("map_query_corr_flow_source_explicit", 0.0),
            ]
        )
    if "map_query_corr_argmax_flow_mag_px" in result:
        log_msg += " aflow=%.2f aepe=%.2f afcos=%.3f"
        log_args.extend(
            [
                result.get("map_query_corr_argmax_flow_mag_px", 0.0),
                result.get("map_query_corr_argmax_flow_epe", 0.0),
                result.get("map_query_corr_argmax_flow_cosine", 0.0),
            ]
        )
    if "map_query_corr_flow_conf_mean" in result:
        log_msg += " fconf=%.3f"
        log_args.append(result.get("map_query_corr_flow_conf_mean", 0.0))
    if result.get("map_query_corr_skipped_missing_negatives", 0.0) > 0:
        log_msg += " corr_skip_neg=%.0f"
        log_args.append(result.get("map_query_corr_skipped_missing_negatives", 0.0))
    if "map_query_scene_coord_err_cm" in result:
        log_msg += " scene=%.1fcm"
        log_args.append(result.get("map_query_scene_coord_err_cm", 0.0))
    if "map_query_scene_coord_warp_err_cm" in result:
        log_msg += " scene_warp=%.1fcm"
        log_args.append(result.get("map_query_scene_coord_warp_err_cm", 0.0))
    if "map_query_corr_ce_acc" in result:
        log_msg += " ce_acc=%.3f ce_cov=%.3f"
        log_args.extend(
            [
                result.get("map_query_corr_ce_acc", 0.0),
                result.get("map_query_corr_ce_cov", 0.0),
            ]
        )
    if "map_query_corr_peak_gap" in result:
        log_msg += " peak_gap=%.4f peak_acc=%.3f"
        log_args.extend(
            [
                result.get("map_query_corr_peak_gap", 0.0),
                result.get("map_query_corr_peak_acc", 0.0),
            ]
        )
    if "map_teacher_corr_gap" in result:
        log_msg += " tgap=%.4f tacc=%.3f tcos=%.4f tpts=%.0f tcov=%.3f"
        log_args.extend(
            [
                result.get("map_teacher_corr_gap", 0.0),
                result.get("map_teacher_corr_acc", 0.0),
                result.get("map_teacher_corr_cosine", 0.0),
                result.get("map_teacher_corr_points", 0.0),
                result.get("map_teacher_corr_cov", 0.0),
            ]
        )
    if "map_teacher_patch_gap" in result:
        log_msg += " tpatch=%.4f/%.3f/%.2f"
        log_args.extend(
            [
                result.get("map_teacher_patch_gap", 0.0),
                result.get("map_teacher_patch_acc", 0.0),
                result.get("map_teacher_patch_soft_epe", 0.0),
            ]
        )
    if "map_teacher_patch_self_gap" in result:
        log_msg += " self_tpatch=%.4f/%.3f"
        log_args.extend(
            [
                result.get("map_teacher_patch_self_gap", 0.0),
                result.get("map_teacher_patch_self_acc", 0.0),
            ]
        )
    if "map_teacher_patch_radio_gap" in result:
        log_msg += " radio_tpatch=%.4f/%.3f"
        log_args.extend(
            [
                result.get("map_teacher_patch_radio_gap", 0.0),
                result.get("map_teacher_patch_radio_acc", 0.0),
            ]
        )
    if "map_query_corr_distill_argmax_agree" in result:
        log_msg += " dist=%.3f/%.4f"
        log_args.extend(
            [
                result.get("map_query_corr_distill_argmax_agree", 0.0),
                result.get("map_query_corr_distill_student_gap", 0.0),
            ]
        )
    if "map_query_identity_corr_peak_gap" in result:
        log_msg += " id_peak=%.4f/%.3f"
        log_args.extend(
            [
                result.get("map_query_identity_corr_peak_gap", 0.0),
                result.get("map_query_identity_corr_peak_acc", 0.0),
            ]
        )
    if "map_query_identity_corr_distill_argmax_agree" in result:
        log_msg += " id_dist=%.3f/%.4f"
        log_args.extend(
            [
                result.get("map_query_identity_corr_distill_argmax_agree", 0.0),
                result.get("map_query_identity_corr_distill_student_gap", 0.0),
            ]
        )
    if "map_corr_wls_trans_err_mm" in result:
        log_msg += " corr_wls=%.1fmm gain=%.1fmm wconf=%.3f/%.3f"
        log_args.extend(
            [
                result.get("map_corr_wls_trans_err_mm", 0.0),
                result.get("map_corr_wls_trans_gain_mm", 0.0),
                result.get("map_corr_wls_conf_mean", 0.0),
                result.get("map_corr_wls_conf_cov", 0.0),
            ]
        )
        if "map_corr_wls_gated_trans_err_mm" in result:
            log_msg += " gated=%.1fmm ggain=%.1fmm acc=%.2f"
            log_args.extend(
                [
                    result.get("map_corr_wls_gated_trans_err_mm", 0.0),
                    result.get("map_corr_wls_gated_trans_gain_mm", 0.0),
                    result.get("map_corr_wls_gated_accept_rate", 0.0),
                ]
            )
    if "map_self_corr_argmax_flow_epe" in result or "map_self_corr_subpx_flow_epe" in result:
        log_msg += " self_corr_epe=%.3f self_corr_acc=%.3f"
        log_args.extend(
            [
                result.get(
                    "map_self_corr_argmax_flow_epe",
                    result.get("map_self_corr_subpx_flow_epe", 0.0),
                ),
                result.get("map_self_corr_subpx_acc", 0.0),
            ]
        )
    if "map_query_flow_warp_cosine" in result:
        log_msg += " warp_cos=%.4f warp_cov=%.3f"
        log_args.extend(
            [
                result.get("map_query_flow_warp_cosine", 0.0),
                result.get("map_query_flow_warp_cov", 0.0),
            ]
        )
    if "map_query_flow_warp_hard_gap" in result:
        log_msg += " warp_gap=%.4f warp_acc=%.3f"
        log_args.extend(
            [
                result.get("map_query_flow_warp_hard_gap", 0.0),
                result.get("map_query_flow_warp_hard_acc", 0.0),
            ]
        )
    if "map_self_flow_warp_hard_gap" in result:
        log_msg += " self_gap=%.4f self_acc=%.3f"
        log_args.extend(
            [
                result.get("map_self_flow_warp_hard_gap", 0.0),
                result.get("map_self_flow_warp_hard_acc", 0.0),
            ]
        )
    if "map_feature_metric_trans_err_mm" in result:
        log_msg += " fm_init=%.1fmm fm_t=%.1fmm fm_gain=%.1fmm fm_dt=%.1fmm"
        log_args.extend(
            [
                result.get("map_feature_metric_init_trans_err_mm", 0.0),
                result.get("map_feature_metric_trans_err_mm", 0.0),
                result.get("map_feature_metric_trans_gain_mm", 0.0),
                result.get("map_feature_metric_delta_trans_mm", 0.0),
            ]
        )
    if "map_self_feature_metric_trans_err_mm" in result:
        log_msg += " self_fm_t=%.1fmm self_fm_gain=%.1fmm"
        log_args.extend(
            [
                result.get("map_self_feature_metric_trans_err_mm", 0.0),
                result.get("map_self_feature_metric_trans_gain_mm", 0.0),
            ]
        )
    if "map_grad_dir_cos" in result:
        log_msg += " grad_cos=%.3f"
        log_args.append(result.get("map_grad_dir_cos", 0.0))
    logger.info(log_msg, *log_args)
    return result


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, step, best_val, map_renderer=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
    }
    if map_renderer is not None:
        payload["map_renderer_state_dict"] = map_renderer.export_trainable_state()
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="Train joint RADIO-DCFF query student scaffold")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path")
    parser.add_argument("--warmstart", default=None, help="Optional checkpoint path to load model weights only")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny dry-run")
    parser.add_argument("--eval-only", action="store_true", help="Run validation once and exit without training")
    parser.add_argument("--eval-output-json", default=None, help="Optional JSON path for eval-only metrics")
    args = parser.parse_args()
    if args.resume and args.warmstart:
        raise ValueError("Use either --resume or --warmstart, not both")

    cfg = load_feature_extract_config(args.config)
    if args.smoke_test:
        cfg["dataset"]["synthetic_if_missing"] = True
        cfg["dataset"]["max_train_samples"] = min(8, cfg["dataset"].get("max_train_samples") or 8)
        cfg["dataset"]["max_val_samples"] = min(4, cfg["dataset"].get("max_val_samples") or 4)
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 1
        cfg["training"]["num_workers"] = 0
        cfg["training"]["max_steps"] = 2

    set_seed(int(cfg["training"].get("seed", 42)))

    output_dir = Path(cfg["output_dir"]) / cfg["exp_name"]
    ckpt_dir = output_dir / "checkpoints"
    qual_dir = output_dir / "qual"
    feature_track_root = Path(cfg["visualization"]["save_root"])
    logger = setup_logger(output_dir)
    qual_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    feature_track_root.mkdir(parents=True, exist_ok=True)

    requested_device = cfg["training"].get("device", "cuda")
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if (
        float(cfg["loss"].get("teacher_norm_weight", 0.0)) > 0.0
        and not bool(cfg["model"].get("predict_magnitude", False))
    ):
        raise ValueError(
            "loss.teacher_norm_weight > 0 requires model.predict_magnitude=true."
        )

    safe_num_workers = resolve_safe_num_workers(cfg["training"], cfg["dataset"])
    if safe_num_workers != int(cfg["training"].get("num_workers", 0)):
        logger.info(
            "Reducing DataLoader num_workers from %d to %d for input_hw=%s to avoid shared-memory worker crashes. "
            "Set training.allow_highres_num_workers=true to override.",
            int(cfg["training"].get("num_workers", 0)),
            safe_num_workers,
            cfg["dataset"].get("input_hw"),
        )
        cfg["training"]["num_workers"] = safe_num_workers

    teacher_store = TeacherFeatureStore(
        cfg["dataset"]["feature_dir"],
        cache_in_memory=bool(cfg["dataset"].get("cache_teacher", False)),
    )
    fine_feature_dim, coarse_feature_dim = resolve_query_feature_dims(cfg, teacher_store)
    logger.info(
        "Teacher feature space: fine=%dd@%s coarse=%dd@%s",
        fine_feature_dim,
        teacher_store.feature_hw,
        coarse_feature_dim,
        teacher_store.coarse_feature_hw,
    )

    retrieval_cfg = cfg.get("retrieval", {})
    retrieval_store = None
    if retrieval_cfg.get("enabled", False):
        retrieval_feature_dir = retrieval_cfg.get("feature_dir")
        if not retrieval_feature_dir:
            raise ValueError("retrieval.feature_dir is required when retrieval.enabled=true")
        retrieval_store = RetrievalTeacherStore(
            retrieval_feature_dir,
            subdir=retrieval_cfg.get("teacher_subdir", "cls"),
            cache_in_memory=bool(retrieval_cfg.get("cache_teacher", False)),
        )
        if int(retrieval_cfg.get("student_dim", 0)) != retrieval_store.feature_dim:
            logger.info(
                "Overriding retrieval student_dim from %s -> teacher dim %d",
                retrieval_cfg.get("student_dim"),
                retrieval_store.feature_dim,
            )
            cfg["retrieval"]["student_dim"] = retrieval_store.feature_dim

    all_records = build_all_records(
        cfg["dataset"],
        teacher_store,
        allow_synthetic=bool(cfg["dataset"].get("synthetic_if_missing", False)),
    )
    train_records, val_records = split_records(all_records, cfg["dataset"])

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    logger.info(
        "Paired records: train=%d val=%d teacher_hw=%s synthetic=%s",
        len(train_records),
        len(val_records),
        teacher_store.feature_hw,
        train_records[0]["image_path"] is None,
    )
    if retrieval_store is not None:
        logger.info(
            "Retrieval supervision enabled: dim=%d dir=%s",
            retrieval_store.feature_dim,
            retrieval_store.root_dir,
        )

    teacher_corr_train_store = None
    teacher_corr_val_store = None
    teacher_corr_path = cfg["dataset"].get("teacher_correspondence_path")
    teacher_corr_train_path = cfg["dataset"].get("teacher_correspondence_train_path") or teacher_corr_path
    teacher_corr_val_path = cfg["dataset"].get("teacher_correspondence_val_path") or teacher_corr_path
    if teacher_corr_train_path:
        teacher_corr_train_store = TeacherCorrespondenceStore(
            teacher_corr_train_path,
            feature_hw=cfg["dataset"]["feature_hw"],
            max_points=int(cfg["dataset"].get("teacher_correspondence_max_points", 512)),
            coordinate_space=str(cfg["dataset"].get("teacher_correspondence_coordinate_space", "auto")),
        )
        logger.info(
            "Train teacher sparse correspondences enabled: path=%s max_points=%d",
            teacher_corr_train_path,
            int(cfg["dataset"].get("teacher_correspondence_max_points", 512)),
        )
    if teacher_corr_val_path:
        teacher_corr_val_store = (
            teacher_corr_train_store
            if teacher_corr_val_path == teacher_corr_train_path
            else TeacherCorrespondenceStore(
                teacher_corr_val_path,
                feature_hw=cfg["dataset"]["feature_hw"],
                max_points=int(cfg["dataset"].get("teacher_correspondence_max_points", 512)),
                coordinate_space=str(cfg["dataset"].get("teacher_correspondence_coordinate_space", "auto")),
            )
        )
        logger.info(
            "Val teacher sparse correspondences enabled: path=%s max_points=%d",
            teacher_corr_val_path,
            int(cfg["dataset"].get("teacher_correspondence_max_points", 512)),
        )

    map_renderer = None
    if cfg.get("map_supervision", {}).get("enabled", False):
        map_renderer = MapFeatureRenderer(
            cfg,
            feature_hw=tuple(cfg["dataset"]["feature_hw"]),
            device=device,
            logger=logger,
        )
    colmap_dir = (
        cfg["dataset"].get("colmap_dir")
        or cfg.get("map_supervision", {}).get("colmap_dir")
        or str(Path(cfg["dataset"]["source_dir"]) / "sparse" / "0")
    )
    train_pose_candidate_index = load_pose_candidate_cache_index(
        cfg["dataset"].get("train_pose_candidate_cache")
        or cfg["dataset"].get("train_pose_candidate_caches")
    )
    val_pose_candidate_index = load_pose_candidate_cache_index(
        cfg["dataset"].get("val_pose_candidate_cache")
        or cfg["dataset"].get("val_pose_candidate_caches")
    )
    if train_pose_candidate_index:
        logger.info("Train pose candidate cache enabled: records=%d", len(train_pose_candidate_index))
    if val_pose_candidate_index:
        logger.info("Val pose candidate cache enabled: records=%d", len(val_pose_candidate_index))
    pose_candidate_topk = int(
        cfg["dataset"].get(
            "pose_candidate_topk",
            cfg.get("map_supervision", {}).get(
                "candidate_stage1_topk",
                cfg.get("map_supervision", {}).get("candidate_render_score_max_candidates", 0),
            ),
        )
        or 0
    )

    train_dataset = JointRADIOQueryDataset(
        train_records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        retrieval_teacher_store=retrieval_store,
        teacher_correspondence_store=teacher_corr_train_store,
        prior_mask_path=cfg["dataset"].get("prior_mask_path"),
        prior_mask_channels=cfg["dataset"].get("prior_mask_channels"),
        colmap_dir=colmap_dir,
        pose_candidate_cache_index=train_pose_candidate_index,
        pose_candidate_topk=pose_candidate_topk,
    )
    val_dataset = JointRADIOQueryDataset(
        val_records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        retrieval_teacher_store=retrieval_store,
        teacher_correspondence_store=teacher_corr_val_store,
        prior_mask_path=cfg["dataset"].get("prior_mask_path"),
        prior_mask_channels=cfg["dataset"].get("prior_mask_channels"),
        colmap_dir=colmap_dir,
        pose_candidate_cache_index=val_pose_candidate_index,
        pose_candidate_topk=pose_candidate_topk,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_radio_query_student(
        cfg,
        fine_feature_dim=fine_feature_dim,
        coarse_feature_dim=coarse_feature_dim,
        retrieval_dim=int(cfg["retrieval"]["student_dim"]) if retrieval_store is not None else None,
        retrieval_hidden_dim=int(cfg["retrieval"].get("hidden_dim", 0)) if retrieval_store is not None else None,
    ).to(device)
    trainable_prefixes = cfg["training"].get("freeze_model_except_prefixes")
    if trainable_prefixes:
        trainable_summary = apply_model_trainable_filter(
            model,
            trainable_prefixes=trainable_prefixes,
        )
        logger.info(
            "Model trainable filter active: prefixes=%s trainable=%d tensors/%d params frozen=%d tensors/%d params",
            trainable_prefixes,
            trainable_summary["trainable_tensors"],
            trainable_summary["trainable_parameters"],
            trainable_summary["frozen_tensors"],
            trainable_summary["frozen_parameters"],
        )

    base_lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    optimizer_groups = build_model_param_groups(
        model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        lr_scales=cfg["training"].get("model_lr_scales"),
    )
    if map_renderer is not None and map_renderer.has_trainable_params():
        optimizer_groups.extend(map_renderer.get_param_groups(base_lr=base_lr, weight_decay=weight_decay))
    optimizer = torch.optim.AdamW(optimizer_groups, lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg["training"]["epochs"])),
    )
    scaler = GradScaler(enabled=bool(cfg["training"].get("amp", True) and device.type == "cuda"))
    trainable_params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]

    start_epoch = 0
    step = 0
    best_val = float("inf")

    resume_path = args.resume
    if resume_path:
        checkpoint = safe_torch_load(resume_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if map_renderer is not None:
            map_renderer.load_trainable_state(checkpoint.get("map_renderer_state_dict"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("best_val", best_val))
        logger.info("Resumed from %s at epoch=%d step=%d", resume_path, start_epoch, step)
    elif args.warmstart:
        checkpoint = safe_torch_load(args.warmstart)
        warmstart_strict = bool(cfg["model"].get("warmstart_strict", True))
        warmstart_skip_prefixes = cfg["model"].get("warmstart_skip_prefixes") or []
        load_result = load_model_warmstart(
            model,
            checkpoint,
            strict=warmstart_strict,
            skip_prefixes=warmstart_skip_prefixes,
        )
        if map_renderer is not None:
            map_renderer.load_trainable_state(checkpoint.get("map_renderer_state_dict"))
        if warmstart_strict:
            logger.info("Warmstarted weights from %s", args.warmstart)
        else:
            logger.info(
                "Warmstarted weights from %s (strict=%s, missing=%d, unexpected=%d, mismatched=%d)",
                args.warmstart,
                warmstart_strict,
                len(load_result["missing_keys"]),
                len(load_result["skipped_unexpected"]) + len(load_result["unexpected_keys"]),
                len(load_result["skipped_mismatched"]) + len(load_result.get("skipped_by_prefix", [])),
            )
            if load_result["skipped_mismatched"]:
                logger.info("Skipped mismatched warmstart tensors: %s", load_result["skipped_mismatched"])
            if load_result.get("skipped_by_prefix"):
                logger.info("Skipped warmstart tensors by prefix: %s", load_result["skipped_by_prefix"])

    if args.eval_only:
        val_metrics = validate(
            model,
            val_loader,
            cfg,
            device,
            qual_dir,
            feature_track_root,
            step,
            logger,
            map_renderer,
            epoch=max(0, start_epoch),
        )
        if args.eval_output_json:
            output_json = Path(args.eval_output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w") as f:
                json.dump({key: float(value) for key, value in val_metrics.items()}, f, indent=2, sort_keys=True)
            logger.info("Wrote eval-only metrics to %s", output_json)
        logger.info("Eval-only done.")
        return

    max_steps = cfg["training"].get("max_steps")
    use_amp = bool(cfg["training"].get("amp", True) and device.type == "cuda")
    latest_val_metrics = {}

    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        model.train()
        if map_renderer is not None:
            map_renderer.set_train_mode(True)
        epoch_metrics = []

        for batch in train_loader:
            step += 1
            batch = move_batch_to_device(batch, device)
            if map_renderer is not None:
                batch = map_renderer.attach_to_batch(
                    batch,
                    require_grad=bool(map_renderer.has_trainable_params()),
                )
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                preattached_pose_candidates = False
                if should_preattach_pose_candidate_renders(cfg, epoch=epoch):
                    batch = maybe_attach_pose_candidate_renders(
                        batch,
                        {},
                        cfg,
                        map_renderer,
                        require_grad=bool(map_renderer is not None and map_renderer.has_trainable_params()),
                        epoch=epoch,
                    )
                    preattached_pose_candidates = "rendered_map_candidate_pose" in batch
                if bool(cfg["model"].get("teacher_fine_condition", False)):
                    outputs = model(batch["rgb"], teacher_fine=batch.get("teacher_fine"))
                else:
                    outputs = model(batch["rgb"])
                if not preattached_pose_candidates:
                    batch = maybe_attach_pose_candidate_renders(
                        batch,
                        outputs,
                        cfg,
                        map_renderer,
                        require_grad=bool(map_renderer is not None and map_renderer.has_trainable_params()),
                        epoch=epoch,
                    )
                main_total, metrics = compute_main_losses(outputs, batch, cfg)
                map_total, map_metrics = compute_map_supervision(
                    batch,
                    outputs,
                    cfg,
                    device,
                    epoch=epoch,
                    local_matcher=getattr(model, "local_matcher", None),
                    local_flow_head=getattr(model, "local_flow_head", None),
                    local_corr_projector=getattr(model, "local_corr_projector", None),
                    candidate_score_fusion_head=getattr(model, "candidate_score_fusion_head", None),
                    map_renderer=map_renderer,
                )
                total_loss = main_total + map_total

            scaler.scale(total_loss).backward()
            if cfg["training"].get("grad_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, float(cfg["training"]["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()

            metrics.update(map_metrics)
            metrics["loss_total"] = total_loss.detach()
            epoch_metrics.append(metrics)

            if step % int(cfg["training"].get("log_every", 10)) == 0:
                mean_train = mean_metrics(epoch_metrics[-int(cfg["training"].get("log_every", 10)):])
                log_msg = "Train epoch=%d step=%d total=%.4f fine_cos=%.4f coarse_cos=%.4f"
                log_args = [
                    epoch,
                    step,
                    mean_train.get("loss_total", 0.0),
                    mean_train.get("fine_cosine", 0.0),
                    mean_train.get("coarse_cosine", 0.0),
                ]
                if "retrieval_cosine" in mean_train:
                    log_msg += " retrieval_cos=%.4f"
                    log_args.append(mean_train.get("retrieval_cosine", 0.0))
                if "map_query_fine_cosine" in mean_train:
                    log_msg += " map_q_f=%.4f map_q_loc=%.4f map_q_c=%.4f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_fine_cosine", 0.0),
                            mean_train.get("map_query_local_fine_cosine", 0.0),
                            mean_train.get("map_query_coarse_cosine", 0.0),
                        ]
                    )
                if "map_query_fine_raw_cosine" in mean_train:
                    log_msg += " map_q_f_raw=%.4f"
                    log_args.append(mean_train.get("map_query_fine_raw_cosine", 0.0))
                if "map_query_projected_fine_cosine" in mean_train:
                    log_msg += " qproj=%.4f"
                    log_args.append(mean_train.get("map_query_projected_fine_cosine", 0.0))
                if "map_coarse_pose_rank_gap" in mean_train:
                    log_msg += " c_rank=%.4f/%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_coarse_pose_rank_gap", 0.0),
                            mean_train.get("map_coarse_pose_rank_acc", 0.0),
                        ]
                    )
                if "map_coarse_pose_energy_gap" in mean_train:
                    log_msg += " c_energy=%.4f/%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_coarse_pose_energy_gap", 0.0),
                            mean_train.get("map_coarse_pose_energy_acc", 0.0),
                        ]
                    )
                if "map_coarse_pose_local_energy_gap" in mean_train:
                    log_msg += " c_local=%.4f/%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_coarse_pose_local_energy_gap", 0.0),
                            mean_train.get("map_coarse_pose_local_energy_acc", 0.0),
                        ]
                    )
                if "map_candidate_render_score_acc" in mean_train:
                    log_msg += " cand=%.3f/%.4f"
                    log_args.extend(
                        [
                            mean_train.get("map_candidate_render_score_acc", 0.0),
                            mean_train.get("map_candidate_render_score_margin", 0.0),
                        ]
                    )
                if "map_candidate_score_fusion_acc" in mean_train:
                    log_msg += " cfuse=%.3f/%.4f/%.0fmm"
                    log_args.extend(
                        [
                            mean_train.get("map_candidate_score_fusion_acc", 0.0),
                            mean_train.get("map_candidate_score_fusion_margin", 0.0),
                            mean_train.get("map_candidate_score_fusion_pred_trans_mm", 0.0),
                        ]
                    )
                    if "map_candidate_score_fusion_top4_basin_recall" in mean_train:
                        log_msg += " basin@1/4/8=%.3f/%.3f/%.3f oracle=%.3f"
                        log_args.extend(
                            [
                                mean_train.get("map_candidate_score_fusion_top1_basin_recall", 0.0),
                                mean_train.get("map_candidate_score_fusion_top4_basin_recall", 0.0),
                                mean_train.get("map_candidate_score_fusion_top8_basin_recall", 0.0),
                                mean_train.get("map_candidate_score_fusion_oracle_basin_recall", 0.0),
                            ]
                        )
                    if mean_train.get("map_candidate_score_fusion_cost_regression_loss", 0.0) > 0:
                        log_msg += " cfreg=%.3f"
                        log_args.append(mean_train.get("map_candidate_score_fusion_cost_regression_loss", 0.0))
                    if mean_train.get("map_candidate_score_fusion_pairwise_rank_loss", 0.0) > 0:
                        log_msg += " cfrank=%.3f/%.3f"
                        log_args.extend(
                            [
                                mean_train.get("map_candidate_score_fusion_pairwise_rank_loss", 0.0),
                                mean_train.get("map_candidate_score_fusion_pairwise_rank_acc", 0.0),
                            ]
                        )
                if "map_candidate_refined_pose_trans_mm" in mean_train:
                    log_msg += " cref=%.0fmm init=%.0fmm"
                    log_args.extend(
                        [
                            mean_train.get("map_candidate_refined_pose_trans_mm", 0.0),
                            mean_train.get("map_candidate_refined_pose_selected_init_trans_mm", 0.0),
                        ]
                    )
                if "map_alpha_coverage" in mean_train:
                    log_msg += " alpha_cov=%.4f alpha_cov_valid=%.4f alpha_mean=%.4f"
                    log_args.extend(
                        [
                            mean_train.get("map_alpha_coverage", 0.0),
                            mean_train.get("map_alpha_coverage_valid", mean_train.get("map_alpha_coverage", 0.0)),
                            mean_train.get("map_alpha_mean", 0.0),
                        ]
                    )
                if "map_query_corr_subpx_flow_epe" in mean_train:
                    log_msg += " corr_epe=%.3f corr_acc=%.3f corr_cov=%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_corr_subpx_flow_epe", 0.0),
                            mean_train.get("map_query_corr_subpx_acc", 0.0),
                            mean_train.get("map_query_corr_subpx_cov", 0.0),
                        ]
                    )
                if "map_query_corr_pred_flow_mag_px" in mean_train:
                    log_msg += " flow=%.2f/%.2f fcos=%.3f src=%.0f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_corr_pred_flow_mag_px", 0.0),
                            mean_train.get("map_query_corr_gt_flow_mag_px", 0.0),
                            mean_train.get("map_query_corr_flow_cosine", 0.0),
                            mean_train.get("map_query_corr_flow_source_explicit", 0.0),
                        ]
                    )
                if "map_query_corr_argmax_flow_mag_px" in mean_train:
                    log_msg += " aflow=%.2f aepe=%.2f afcos=%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_corr_argmax_flow_mag_px", 0.0),
                            mean_train.get("map_query_corr_argmax_flow_epe", 0.0),
                            mean_train.get("map_query_corr_argmax_flow_cosine", 0.0),
                        ]
                    )
                if "map_query_corr_flow_conf_mean" in mean_train:
                    log_msg += " fconf=%.3f"
                    log_args.append(mean_train.get("map_query_corr_flow_conf_mean", 0.0))
                if mean_train.get("map_query_corr_skipped_missing_negatives", 0.0) > 0:
                    log_msg += " corr_skip_neg=%.0f"
                    log_args.append(mean_train.get("map_query_corr_skipped_missing_negatives", 0.0))
                if "map_query_scene_coord_err_cm" in mean_train:
                    log_msg += " scene=%.1fcm"
                    log_args.append(mean_train.get("map_query_scene_coord_err_cm", 0.0))
                if "map_query_scene_coord_warp_err_cm" in mean_train:
                    log_msg += " scene_warp=%.1fcm"
                    log_args.append(mean_train.get("map_query_scene_coord_warp_err_cm", 0.0))
                if "map_query_corr_ce_acc" in mean_train:
                    log_msg += " ce_acc=%.3f ce_cov=%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_corr_ce_acc", 0.0),
                            mean_train.get("map_query_corr_ce_cov", 0.0),
                        ]
                    )
                if "map_query_corr_peak_gap" in mean_train:
                    log_msg += " peak_gap=%.4f peak_acc=%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_corr_peak_gap", 0.0),
                            mean_train.get("map_query_corr_peak_acc", 0.0),
                        ]
                    )
                if "map_teacher_corr_gap" in mean_train:
                    log_msg += " tgap=%.4f tacc=%.3f tcos=%.4f tpts=%.0f tcov=%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_teacher_corr_gap", 0.0),
                            mean_train.get("map_teacher_corr_acc", 0.0),
                            mean_train.get("map_teacher_corr_cosine", 0.0),
                            mean_train.get("map_teacher_corr_points", 0.0),
                            mean_train.get("map_teacher_corr_cov", 0.0),
                        ]
                    )
                if "map_teacher_patch_gap" in mean_train:
                    log_msg += " tpatch=%.4f/%.3f/%.2f"
                    log_args.extend(
                        [
                            mean_train.get("map_teacher_patch_gap", 0.0),
                            mean_train.get("map_teacher_patch_acc", 0.0),
                            mean_train.get("map_teacher_patch_soft_epe", 0.0),
                        ]
                    )
                if "map_teacher_patch_self_gap" in mean_train:
                    log_msg += " self_tpatch=%.4f/%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_teacher_patch_self_gap", 0.0),
                            mean_train.get("map_teacher_patch_self_acc", 0.0),
                        ]
                    )
                if "map_teacher_patch_radio_gap" in mean_train:
                    log_msg += " radio_tpatch=%.4f/%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_teacher_patch_radio_gap", 0.0),
                            mean_train.get("map_teacher_patch_radio_acc", 0.0),
                        ]
                    )
                if "map_query_corr_distill_argmax_agree" in mean_train:
                    log_msg += " dist=%.3f/%.4f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_corr_distill_argmax_agree", 0.0),
                            mean_train.get("map_query_corr_distill_student_gap", 0.0),
                        ]
                    )
                if "map_query_identity_corr_peak_gap" in mean_train:
                    log_msg += " id_peak=%.4f/%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_identity_corr_peak_gap", 0.0),
                            mean_train.get("map_query_identity_corr_peak_acc", 0.0),
                        ]
                    )
                if "map_query_identity_corr_distill_argmax_agree" in mean_train:
                    log_msg += " id_dist=%.3f/%.4f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_identity_corr_distill_argmax_agree", 0.0),
                            mean_train.get("map_query_identity_corr_distill_student_gap", 0.0),
                        ]
                    )
                if "map_corr_wls_trans_err_mm" in mean_train:
                    log_msg += " corr_wls=%.1fmm gain=%.1fmm wconf=%.3f/%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_corr_wls_trans_err_mm", 0.0),
                            mean_train.get("map_corr_wls_trans_gain_mm", 0.0),
                            mean_train.get("map_corr_wls_conf_mean", 0.0),
                            mean_train.get("map_corr_wls_conf_cov", 0.0),
                        ]
                    )
                    if "map_corr_wls_gated_trans_err_mm" in mean_train:
                        log_msg += " gated=%.1fmm ggain=%.1fmm acc=%.2f"
                        log_args.extend(
                            [
                                mean_train.get("map_corr_wls_gated_trans_err_mm", 0.0),
                                mean_train.get("map_corr_wls_gated_trans_gain_mm", 0.0),
                                mean_train.get("map_corr_wls_gated_accept_rate", 0.0),
                            ]
                        )
                if "map_self_corr_argmax_flow_epe" in mean_train or "map_self_corr_subpx_flow_epe" in mean_train:
                    log_msg += " self_corr_epe=%.3f self_corr_acc=%.3f"
                    log_args.extend(
                        [
                            mean_train.get(
                                "map_self_corr_argmax_flow_epe",
                                mean_train.get("map_self_corr_subpx_flow_epe", 0.0),
                            ),
                            mean_train.get("map_self_corr_subpx_acc", 0.0),
                        ]
                    )
                if "map_self_corr_peak_gap" in mean_train:
                    log_msg += " self_peak=%.4f/%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_self_corr_peak_gap", 0.0),
                            mean_train.get("map_self_corr_peak_acc", 0.0),
                        ]
                    )
                if "map_query_flow_warp_cosine" in mean_train:
                    log_msg += " warp_cos=%.4f warp_cov=%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_flow_warp_cosine", 0.0),
                            mean_train.get("map_query_flow_warp_cov", 0.0),
                        ]
                    )
                if "map_query_flow_warp_hard_gap" in mean_train:
                    log_msg += " warp_gap=%.4f warp_acc=%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_flow_warp_hard_gap", 0.0),
                            mean_train.get("map_query_flow_warp_hard_acc", 0.0),
                        ]
                    )
                if "map_self_flow_warp_hard_gap" in mean_train:
                    log_msg += " self_gap=%.4f self_acc=%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_self_flow_warp_hard_gap", 0.0),
                            mean_train.get("map_self_flow_warp_hard_acc", 0.0),
                        ]
                    )
                if "map_feature_metric_trans_err_mm" in mean_train:
                    log_msg += " fm_init=%.1fmm fm_t=%.1fmm fm_gain=%.1fmm fm_dt=%.1fmm"
                    log_args.extend(
                        [
                            mean_train.get("map_feature_metric_init_trans_err_mm", 0.0),
                            mean_train.get("map_feature_metric_trans_err_mm", 0.0),
                            mean_train.get("map_feature_metric_trans_gain_mm", 0.0),
                            mean_train.get("map_feature_metric_delta_trans_mm", 0.0),
                        ]
                    )
                if "map_self_feature_metric_trans_err_mm" in mean_train:
                    log_msg += " self_fm_t=%.1fmm self_fm_gain=%.1fmm"
                    log_args.extend(
                        [
                            mean_train.get("map_self_feature_metric_trans_err_mm", 0.0),
                            mean_train.get("map_self_feature_metric_trans_gain_mm", 0.0),
                        ]
                    )
                if "map_grad_dir_cos" in mean_train:
                    log_msg += " grad_cos=%.3f"
                    log_args.append(mean_train.get("map_grad_dir_cos", 0.0))
                log_msg += " lr=%.2e"
                log_args.append(optimizer.param_groups[0]["lr"])
                logger.info(log_msg, *log_args)

            if max_steps is not None and step >= int(max_steps):
                break

        if (epoch + 1) % int(cfg["training"].get("val_every_epochs", 1)) == 0:
            val_metrics = validate(
                model,
                val_loader,
                cfg,
                device,
                qual_dir,
                feature_track_root,
                step,
                logger,
                map_renderer,
                epoch=epoch,
            )
            latest_val_metrics = dict(val_metrics)
            latest_path = ckpt_dir / "latest.pth"
            save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, step, best_val, map_renderer)
            best_metric_name, best_metric_value, selection_score = validation_selection_score(val_metrics, cfg)
            if selection_score < best_val:
                best_val = selection_score
                logger.info(
                    "New best checkpoint by %s=%.4f (selection_score=%.4f)",
                    best_metric_name,
                    best_metric_value,
                    selection_score,
                )
                save_checkpoint(
                    ckpt_dir / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    step,
                    best_val,
                    map_renderer,
                )
        elif (epoch + 1) % int(cfg["training"].get("save_every_epochs", 1)) == 0:
            save_checkpoint(
                ckpt_dir / "latest.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                step,
                best_val,
                map_renderer,
            )

        scheduler.step()
        if max_steps is not None and step >= int(max_steps):
            logger.info("Stopping early at max_steps=%s", max_steps)
            break

    logger.info("Done. best_val=%.4f final_step=%d", best_val, step)

    final_metrics = {
        "best_val": float(best_val),
        "final_step": int(step),
        "epochs_completed": int(epoch + 1 if 'epoch' in locals() else start_epoch),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "map_supervision_enabled": bool(cfg.get("map_supervision", {}).get("enabled", False)),
        "retrieval_enabled": bool(cfg.get("retrieval", {}).get("enabled", False)),
    }
    if latest_val_metrics:
        final_metrics["latest_val"] = latest_val_metrics

    summary_lines = [
        f"best_val={best_val:.4f}",
        f"train_records={len(train_records)} val_records={len(val_records)}",
        f"map_supervision={'on' if cfg.get('map_supervision', {}).get('enabled', False) else 'off'}",
    ]
    if latest_val_metrics:
        summary_lines.append(
            "latest val fine_cos={:.4f} coarse_cos={:.4f}".format(
                float(latest_val_metrics.get("fine_cosine", 0.0)),
                float(latest_val_metrics.get("coarse_cosine", 0.0)),
            )
        )
        if "map_query_coarse_cosine" in latest_val_metrics:
            summary_lines.append(
                "map q_f={:.4f} q_loc={:.4f} q_c={:.4f} t_f={:.4f} t_c={:.4f}".format(
                    float(latest_val_metrics.get("map_query_fine_cosine", 0.0)),
                    float(latest_val_metrics.get("map_query_local_fine_cosine", 0.0)),
                    float(latest_val_metrics.get("map_query_coarse_cosine", 0.0)),
                    float(latest_val_metrics.get("map_teacher_fine_cosine", 0.0)),
                    float(latest_val_metrics.get("map_teacher_coarse_cosine", 0.0)),
                )
            )

    notes = [
        f"config={args.config}",
        f"resume={args.resume}" if args.resume else "resume=None",
        f"warmstart={args.warmstart}" if args.warmstart else "warmstart=None",
        f"smoke_test={bool(args.smoke_test)}",
    ]
    artifact_paths = [
        output_dir / "config.yaml",
        ckpt_dir / "best.pth",
        ckpt_dir / "latest.pth",
        qual_dir,
        feature_track_root,
        output_dir / "train.log",
    ]
    save_experiment_bundle(
        exp_name=cfg["exp_name"],
        output_dir=output_dir,
        metrics=final_metrics,
        summary_lines=summary_lines,
        notes=notes,
        artifact_paths=artifact_paths,
        results_json_name="results.json",
        results_text_name="results.txt",
        report_markdown_name="report.md",
        report_text_name="report.txt",
    )


if __name__ == "__main__":
    main()
