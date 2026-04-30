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
from feature_field.dcff.losses import (
    channel_standardized_loss,
    cosine_loss,
    feature_gradient_loss,
    infonce_contrastive_loss,
    l1_feature_loss,
)
from feature_extract import load_config as load_feature_extract_config
from feature_extract.students.radio_query_student import RadioQueryStudent
from feature_field import build_dcff_runtime, intrinsics_to_K
from feature_field.runtime import _apply_dcff_postprocess
from feature_field.utils.loc_reporting import save_experiment_bundle
from feature_field.utils.feature_track_vis import save_feature_track_visual
from pose_refine import apply_pose_delta, compute_image_jacobian, diff_pose_solve, feature_metric_solve


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
        "fine_loc_head": False,
        "fine_loc_init": 1.0,
        "fine_loc_zero_init": True,
        "fine_loc_detach_base": False,
        "fine_loc_highres_source": None,
        "fine_loc_highres_init": 1.0,
        "fine_loc_highres_zero_init": True,
        "fine_loc_highres_detach": True,
        "teacher_fine_condition": False,
        "teacher_fine_init": 1.0,
        "teacher_fine_zero_init": True,
        "teacher_fine_detach": True,
        "scene_coord_head": False,
        "scene_coord_zero_init": True,
        "scene_coord_detach_base": False,
        "scene_coord_use_pixel_grid": False,
        "scene_coord_global_context": False,
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
        "query_fine_infonce_weight": 0.0,
        "query_coarse_infonce_weight": 0.0,
        "fine_coarse_ortho_weight": 0.0,
        "perturb_rank_weight": 0.0,
        "perturb_max_shift_px": 2,
        "perturb_margin": 0.1,
        "perturb_margin_per_m": 0.0,
        "perturb_render_negatives": False,
        "perturb_rot_deg": 0.0,
        "perturb_trans_m": 0.0,
        "perturb_trans_cm_choices": None,
        "perturb_frame": "camera",
        "perturb_axes": [0, 1, 2],
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
        "query_corr_subpixel_weight": 0.0,
        "query_corr_flow_weight": 0.0,
        "query_corr_peak_margin_weight": 0.0,
        "query_corr_peak_margin": 0.05,
        "query_corr_wls_pose_weight": 0.0,
        "query_corr_wls_pose_damping": 1e-3,
        "query_corr_wls_pose_update_scale": 1.0,
        "query_corr_wls_pose_rot_weight": 1.0,
        "query_corr_wls_pose_trans_weight": 50.0,
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
        "query_flow_warp_weight": 0.0,
        "query_flow_warp_contrastive_weight": 0.0,
        "query_flow_warp_contrastive_margin": 0.1,
        "query_flow_warp_contrastive_offsets": None,
        "query_fine_key": "fine",
        "map_self_corr_subpixel_weight": 0.0,
        "map_self_corr_flow_weight": 0.0,
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
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_merge(DEFAULT_CONFIG, user_cfg)


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def parse_cambridge_split(split_path):
    names = set()
    split_path = Path(split_path)
    if not split_path.is_file():
        return names
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
            names.add(image_name)
            names.add(stem + ".png")
            names.add(stem + ".jpg")
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


class RetrievalTeacherStore:
    def __init__(self, feature_dir, subdir="cls", cache_in_memory=False):
        root = Path(feature_dir)
        self.root_dir = root / subdir if subdir else root
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Retrieval teacher directory not found: {self.root_dir}")

        pattern = re.compile(r"(.+)_cls_.*\.pt$")
        self.files = {}
        for path in sorted(self.root_dir.glob("*.pt")):
            match = pattern.match(path.name)
            if match:
                self.files[match.group(1)] = path
        if not self.files:
            raise RuntimeError(f"No CLS teacher descriptors found in {self.root_dir}")

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


def split_records(all_records, dataset_cfg):
    train_split = parse_cambridge_split(dataset_cfg.get("train_split"))
    val_split = parse_cambridge_split(dataset_cfg.get("val_split"))

    if train_split and val_split:
        train_records = [r for r in all_records if r["normalized_name"] in train_split]
        val_records = [r for r in all_records if r["normalized_name"] in val_split]
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


class JointRADIOQueryDataset(Dataset):
    def __init__(
        self,
        records,
        teacher_store,
        input_hw,
        feature_hw,
        synthetic_rgb=False,
        retrieval_teacher_store=None,
        prior_mask_path=None,
        prior_mask_channels=None,
    ):
        self.records = records
        self.teacher_store = teacher_store
        self.input_hw = tuple(input_hw)
        self.feature_hw = tuple(feature_hw)
        self.synthetic_rgb = synthetic_rgb
        self.retrieval_teacher_store = retrieval_teacher_store
        self.prior_masks = None
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
        prior_mask = self._load_prior_mask(record)
        if prior_mask is not None:
            item["prior_mask"] = prior_mask
        if self.retrieval_teacher_store is not None:
            item["teacher_retrieval"] = self.retrieval_teacher_store.load(record["sample_name"])
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
        self.perturb_render_negatives = bool(map_cfg.get("perturb_render_negatives", False))
        self.perturb_rot_deg = float(map_cfg.get("perturb_rot_deg", 0.0))
        self.perturb_trans_m = float(map_cfg.get("perturb_trans_m", 0.0))
        cm_choices = map_cfg.get("perturb_trans_cm_choices") or []
        self.perturb_trans_cm_choices = [float(v) for v in cm_choices if float(v) > 0.0]
        self.perturb_frame = str(map_cfg.get("perturb_frame", "camera")).lower()
        self.perturb_axes = [int(v) for v in (map_cfg.get("perturb_axes") or [0, 1, 2]) if int(v) in (0, 1, 2)]
        if not self.perturb_axes:
            self.perturb_axes = [0, 1, 2]
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

        logger.info(
            "Map renderer loaded: config=%s, views=%d, feature_hw=%s, cache=%s, trainable=%s, fine_decoder=%s, coarse_fusion=%s, feat_sharp=%s, fsm=%s, hash_mlp=%s, latent=%s, geometry=%s",
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
        )

    def _normalize_name(self, sample_name):
        normalized = str(sample_name).replace("\\", "/")
        if normalized in self.name_to_pose:
            return normalized
        basename = Path(normalized).name
        if basename in self.basename_to_name:
            return self.basename_to_name[basename]
        raise KeyError(f"Missing COLMAP pose for sample '{sample_name}'")

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
        if rot_sigma > 0:
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

    def _render_pose(self, sample_name, pose, require_grad=False):
        normalized = self._normalize_name(sample_name)
        K = intrinsics_to_K(self.name_to_intr[normalized], self.device)
        result = self.dcff_renderer(
            self.gaussians,
            viewmat=pose.to(self.device),
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
        position = self.dcff_renderer.depth_to_position_map(depth, K, pose.to(self.device))
        if position.ndim == 4:
            position = position.permute(0, 3, 1, 2).contiguous()
        else:
            position = position.permute(2, 0, 1).unsqueeze(0).contiguous()
        return fine_raw, fine, coarse, mask, alpha_feat, rgb, depth, position.float()

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


def _resize_query_flow_valid(query_feat, flow_gt, valid_mask, target_hw):
    h, w = int(target_hw[0]), int(target_hw[1])
    query = query_feat.float()
    flow = flow_gt.to(device=query.device).float()
    valid = valid_mask.float()
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if query.shape[-2:] != (h, w):
        src_h, src_w = query.shape[-2:]
        query = F.interpolate(query, (h, w), mode="bilinear", align_corners=False)
        flow = F.interpolate(flow, (h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / max(src_w, 1)
        flow[:, 1] *= h / max(src_h, 1)
        valid = F.interpolate(valid, (h, w), mode="nearest")
    elif flow.shape[-2:] != (h, w):
        src_h, src_w = flow.shape[-2:]
        flow = F.interpolate(flow, (h, w), mode="bilinear", align_corners=False)
        flow[:, 0] *= w / max(src_w, 1)
        flow[:, 1] *= h / max(src_h, 1)
        valid = F.interpolate(valid, (h, w), mode="nearest")
    elif valid.shape[-2:] != (h, w):
        valid = F.interpolate(valid, (h, w), mode="nearest")
    return query, flow, valid


def _local_correlation_offsets(radius, device, dtype):
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    channels = (2 * radius + 1) ** 2
    return dx.reshape(1, channels, 1, 1), dy.reshape(1, channels, 1, 1)


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


def local_correlation_joint_losses(
    rendered_feat,
    query_feat,
    flow_gt,
    valid_mask,
    *,
    radius=4,
    temperature=0.05,
    huber_delta=1.0,
    peak_margin=0.05,
    compute_subpixel=False,
    compute_flow=False,
    compute_peak=False,
    compute_wls_pose=False,
    depth=None,
    pose_ref=None,
    pose_gt=None,
    intrinsics=None,
    damping=1e-3,
    update_scale=1.0,
    rot_weight=1.0,
    trans_weight=50.0,
):
    """Compute all local-correlation losses from a single correlation volume."""
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

        device = corr.device
        dtype = corr.dtype
        losses = {
            "subpixel": corr.new_zeros(()),
            "flow": corr.new_zeros(()),
            "peak": corr.new_zeros(()),
            "wls_pose": corr.new_zeros(()),
        }
        metrics = {}

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

        need_probs = compute_subpixel or compute_flow or compute_wls_pose
        probs = None
        pred_flow = None
        if need_probs:
            dx, dy = _local_correlation_offsets(radius, device, dtype)
            probs = torch.softmax(corr / max(float(temperature), 1e-6), dim=1)
            pred_flow = torch.cat(
                [
                    (probs * dx).sum(dim=1, keepdim=True),
                    (probs * dy).sum(dim=1, keepdim=True),
                ],
                dim=1,
            )

        if compute_subpixel:
            loss_per_pixel = subpx_loss_map / target_mass.clamp(min=1e-6)
            subpx_loss = (loss_per_pixel * pixel_weight).sum() / denom
            losses["subpixel"] = subpx_loss
            with torch.no_grad():
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

        if compute_wls_pose:
            if depth is None or pose_ref is None or pose_gt is None or intrinsics is None:
                raise ValueError("compute_wls_pose=True requires depth, pose_ref, pose_gt, and intrinsics")
            confidence = probs.max(dim=1, keepdim=True).values
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
            valid_wls = valid_weight
            if valid_wls.shape[-2:] != rendered.shape[-2:]:
                valid_wls = F.interpolate(valid_wls, size=rendered.shape[-2:], mode="nearest")
            valid_wls = valid_wls * (depth_s > 0.05).unsqueeze(1).float()

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
            wls_loss = float(rot_weight) * rot_loss.mean() + float(trans_weight) * trans_err_m.mean()
            losses["wls_pose"] = wls_loss
            delta_trans_mm = torch.linalg.norm(delta_xi[:, :3].float(), dim=1).mean() * 1000.0
            conf_denom = valid_wls.sum().clamp(min=1.0)
            conf_mean = (confidence * valid_wls).sum() / conf_denom
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
        confidence = probs.max(dim=1, keepdim=True).values

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

    return total, metrics


def compute_map_supervision(batch, outputs, cfg, device, epoch=0):
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
    pred_fine = outputs[query_fine_key]
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

    detach_query_features = bool(map_cfg.get("detach_query_features", False))
    pred_fine_target = pred_fine.detach() if detach_query_features else pred_fine
    pred_coarse_target = pred_coarse.detach() if detach_query_features else pred_coarse
    coarse_active = int(epoch) >= int(map_cfg.get("coarse_start_epoch", 0))
    query_coarse_weight = float(map_cfg.get("query_coarse_weight", 0.0)) if coarse_active else 0.0
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
    rendered_teacher_fine_weight = resolve_linear_weight(map_cfg, "rendered_teacher_fine_weight", epoch)
    rendered_teacher_fine_raw_weight = resolve_linear_weight(
        map_cfg, "rendered_teacher_fine_raw_weight", epoch
    )
    query_fine_infonce_weight = resolve_linear_weight(map_cfg, "query_fine_infonce_weight", epoch)
    query_coarse_infonce_weight = (
        resolve_linear_weight(map_cfg, "query_coarse_infonce_weight", epoch) if coarse_active else 0.0
    )
    query_fine_grad_weight = resolve_linear_weight(map_cfg, "query_fine_grad_weight", epoch)
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
    query_corr_subpixel_weight = resolve_linear_weight(map_cfg, "query_corr_subpixel_weight", epoch)
    query_corr_flow_weight = resolve_linear_weight(map_cfg, "query_corr_flow_weight", epoch)
    query_corr_peak_margin_weight = resolve_linear_weight(map_cfg, "query_corr_peak_margin_weight", epoch)
    query_corr_peak_margin = float(map_cfg.get("query_corr_peak_margin", 0.05))
    query_corr_wls_pose_weight = resolve_linear_weight(map_cfg, "query_corr_wls_pose_weight", epoch)
    query_corr_wls_pose_damping = float(map_cfg.get("query_corr_wls_pose_damping", 1e-3))
    query_corr_wls_pose_update_scale = float(map_cfg.get("query_corr_wls_pose_update_scale", 1.0))
    query_corr_wls_pose_rot_weight = float(map_cfg.get("query_corr_wls_pose_rot_weight", 1.0))
    query_corr_wls_pose_trans_weight = float(map_cfg.get("query_corr_wls_pose_trans_weight", 50.0))
    query_scene_coord_weight = resolve_linear_weight(map_cfg, "query_scene_coord_weight", epoch)
    query_scene_coord_warp_weight = resolve_linear_weight(map_cfg, "query_scene_coord_warp_weight", epoch)
    query_scene_coord_huber_beta = float(map_cfg.get("query_scene_coord_huber_beta", 0.02))
    query_corr_scene_coord_weight = float(map_cfg.get("query_corr_scene_coord_weight", 0.0))
    feature_metric_scene_coord_weight = float(map_cfg.get("feature_metric_scene_coord_weight", 0.0))
    query_corr_radius = int(map_cfg.get("query_corr_radius", 4))
    query_corr_temperature = float(map_cfg.get("query_corr_temperature", 0.05))
    query_corr_huber_delta = float(map_cfg.get("query_corr_huber_delta", 1.0))
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
    map_self_corr_subpixel_weight = resolve_linear_weight(
        map_cfg, "map_self_corr_subpixel_weight", epoch
    )
    map_self_corr_flow_weight = resolve_linear_weight(
        map_cfg, "map_self_corr_flow_weight", epoch
    )
    map_self_feature_metric_pose_weight = resolve_linear_weight(
        map_cfg, "map_self_feature_metric_pose_weight", epoch
    )
    feature_metric_pose_weight = resolve_linear_weight(map_cfg, "feature_metric_pose_weight", epoch)
    feature_metric_pose_damping = float(map_cfg.get("feature_metric_pose_damping", 1e-3))
    feature_metric_pose_update_scale = float(map_cfg.get("feature_metric_pose_update_scale", 1.0))
    feature_metric_pose_rot_weight = float(map_cfg.get("feature_metric_pose_rot_weight", 1.0))
    feature_metric_pose_trans_weight = float(map_cfg.get("feature_metric_pose_trans_weight", 50.0))
    feature_metric_pose_normalize = bool(map_cfg.get("feature_metric_pose_normalize", True))

    fine_query_mask = _resize_mask(mask, pred_fine_target.shape[-2:])
    coarse_query_mask = _resize_mask(mask, pred_coarse_target.shape[-2:])
    rendered_fine_query = _resize_feature(rendered_fine, pred_fine_target.shape[-2:])
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

    query_corr_subpx_loss = zero
    query_corr_flow_loss = zero
    query_corr_peak_margin_loss = zero
    query_corr_wls_pose_loss = zero
    query_corr_wls_pose_metrics = {}
    query_corr_metrics = {}
    if (
        query_corr_subpixel_weight > 0
        or query_corr_flow_weight > 0
        or query_corr_peak_margin_weight > 0
        or query_corr_wls_pose_weight > 0
    ):
        neg_fine_for_corr = batch.get("rendered_map_fine_neg")
        neg_mask_for_corr = batch.get("rendered_map_mask_neg")
        neg_depth_for_corr = batch.get("rendered_map_depth_neg")
        neg_position_for_corr = batch.get("rendered_map_position_neg")
        flow_neg_to_gt = batch.get("rendered_map_flow_neg_to_gt")
        flow_valid_neg_to_gt = batch.get("rendered_map_flow_valid_neg_to_gt")
        if (
            neg_fine_for_corr is not None
            and flow_neg_to_gt is not None
            and flow_valid_neg_to_gt is not None
        ):
            corr_rendered_feat = neg_fine_for_corr
            corr_query_feat = pred_fine_target
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
                query_corr_wls_pose_weight > 0
                and neg_depth_for_corr is not None
                and pose_neg is not None
                and pose_gt is not None
                and rendered_intrinsics is not None
            )
            joint_corr = local_correlation_joint_losses(
                corr_rendered_feat,
                corr_query_feat,
                flow_neg_to_gt,
                corr_valid,
                radius=query_corr_radius,
                temperature=query_corr_temperature,
                huber_delta=query_corr_huber_delta,
                peak_margin=query_corr_peak_margin,
                compute_subpixel=query_corr_subpixel_weight > 0,
                compute_flow=query_corr_flow_weight > 0,
                compute_peak=query_corr_peak_margin_weight > 0,
                compute_wls_pose=compute_corr_wls,
                depth=neg_depth_for_corr,
                pose_ref=pose_neg,
                pose_gt=pose_gt,
                intrinsics=rendered_intrinsics,
                damping=query_corr_wls_pose_damping,
                update_scale=query_corr_wls_pose_update_scale,
                rot_weight=query_corr_wls_pose_rot_weight,
                trans_weight=query_corr_wls_pose_trans_weight,
            )
            query_corr_subpx_loss = joint_corr["losses"]["subpixel"]
            query_corr_flow_loss = joint_corr["losses"]["flow"]
            query_corr_peak_margin_loss = joint_corr["losses"]["peak"]
            query_corr_metrics.update(joint_corr["metrics"])
            if compute_corr_wls:
                query_corr_wls_pose_loss = joint_corr["losses"]["wls_pose"]
                query_corr_wls_pose_metrics = {
                    key: value
                    for key, value in joint_corr["metrics"].items()
                    if key.startswith("map_corr_wls_")
                }

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
                    pred_fine_target,
                    flow_neg_to_gt,
                    warp_valid,
                )
            if query_flow_warp_contrastive_weight > 0:
                (
                    query_flow_warp_contrastive_loss,
                    query_flow_warp_contrastive_metrics,
                ) = flow_warp_contrastive_loss(
                    neg_fine_for_warp,
                    pred_fine_target,
                    flow_neg_to_gt,
                    warp_valid,
                    margin=query_flow_warp_contrastive_margin,
                    offsets=flow_warp_contrastive_offsets,
                )

    map_self_flow_warp_loss = zero
    map_self_flow_warp_metrics = {}
    map_self_flow_warp_contrastive_loss = zero
    map_self_flow_warp_contrastive_metrics = {}
    map_self_corr_subpx_loss = zero
    map_self_corr_flow_loss = zero
    map_self_corr_metrics = {}
    if (
        map_self_flow_warp_weight > 0
        or map_self_flow_warp_contrastive_weight > 0
        or map_self_corr_subpixel_weight > 0
        or map_self_corr_flow_weight > 0
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
            if map_self_corr_subpixel_weight > 0 or map_self_corr_flow_weight > 0:
                self_joint_corr = local_correlation_joint_losses(
                    neg_fine_for_self,
                    rendered_fine,
                    flow_neg_to_gt,
                    self_valid,
                    radius=query_corr_radius,
                    temperature=query_corr_temperature,
                    huber_delta=query_corr_huber_delta,
                    compute_subpixel=map_self_corr_subpixel_weight > 0,
                    compute_flow=map_self_corr_flow_weight > 0,
                )
                map_self_corr_subpx_loss = self_joint_corr["losses"]["subpixel"]
                map_self_corr_flow_loss = self_joint_corr["losses"]["flow"]
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
            )
            map_self_feature_metric_pose_metrics = prefix_metric_keys(
                self_fm_metrics,
                "map_feature_metric_",
                "map_self_feature_metric_",
            )

    feature_metric_pose_loss = zero
    feature_metric_pose_metrics = {}
    if feature_metric_pose_weight > 0:
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
            fm_query_feat = pred_fine_target
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
            neg_fine_query = _resize_feature(neg_fine, pred_fine_target.shape[-2:])
            neg_coarse_query = _resize_feature(neg_coarse, pred_coarse_target.shape[-2:])
            neg_fine_mask = _resize_mask(neg_mask, pred_fine_target.shape[-2:])
            neg_coarse_mask = _resize_mask(neg_mask, pred_coarse_target.shape[-2:])
            pos_fine = l1_feature_loss(pred_fine_target, rendered_fine_query, fine_query_mask) + cosine_loss(
                pred_fine_target, rendered_fine_query, fine_query_mask
            )
            neg_fine_loss = l1_feature_loss(pred_fine_target, neg_fine_query, neg_fine_mask) + cosine_loss(
                pred_fine_target, neg_fine_query, neg_fine_mask
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
        + query_coarse_loss
        + query_fine_nce_loss
        + query_coarse_nce_loss
        + query_fine_grad_loss
        + query_coarse_grad_loss
        + rendered_teacher_fine_loss
        + rendered_teacher_fine_raw_loss
        + rendered_teacher_coarse_loss
        + rendered_teacher_fine_nce_loss
        + rendered_teacher_coarse_nce_loss
        + rendered_teacher_fine_grad_loss
        + rendered_teacher_coarse_grad_loss
        + map_fine_coarse_ortho_loss
        + variance_weight_query * query_variance_loss
        + variance_weight_map * map_variance_loss
        + covariance_weight_query * query_covariance_loss
        + covariance_weight_map * map_covariance_loss
        + query_scene_coord_weight * query_scene_coord_loss
        + query_scene_coord_warp_weight * query_scene_coord_warp_loss
        + query_corr_subpixel_weight * query_corr_subpx_loss
        + query_corr_flow_weight * query_corr_flow_loss
        + query_corr_peak_margin_weight * query_corr_peak_margin_loss
        + query_corr_wls_pose_weight * query_corr_wls_pose_loss
        + query_flow_warp_weight * query_flow_warp_loss
        + query_flow_warp_contrastive_weight * query_flow_warp_contrastive_loss
        + map_self_corr_subpixel_weight * map_self_corr_subpx_loss
        + map_self_corr_flow_weight * map_self_corr_flow_loss
        + map_self_flow_warp_weight * map_self_flow_warp_loss
        + map_self_flow_warp_contrastive_weight * map_self_flow_warp_contrastive_loss
        + map_self_feature_metric_pose_weight * map_self_feature_metric_pose_loss
        + feature_metric_pose_weight * feature_metric_pose_loss
        + alpha_coverage_loss
        + rgb_reconstruction_loss
        + perturb_rank_weight * perturb_rank_loss
    )

    with torch.no_grad():
        fine_map_teacher_cos = 1.0 - cosine_loss(rendered_fine_teacher, teacher_fine, fine_teacher_mask)
        coarse_map_teacher_cos = 1.0 - cosine_loss(rendered_coarse_teacher, teacher_coarse, coarse_teacher_mask)
        fine_query_map_cos = 1.0 - cosine_loss(pred_fine, rendered_fine_query, fine_query_mask)
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
        "map_query_coarse_loss": query_coarse_loss.detach(),
        "map_query_fine_nce_loss": query_fine_nce_loss.detach(),
        "map_query_coarse_nce_loss": query_coarse_nce_loss.detach(),
        "map_query_fine_grad_loss": query_fine_grad_loss.detach(),
        "map_query_coarse_grad_loss": query_coarse_grad_loss.detach(),
        "map_rendered_teacher_fine_loss": rendered_teacher_fine_loss.detach(),
        "map_rendered_teacher_fine_raw_loss": rendered_teacher_fine_raw_loss.detach(),
        "map_rendered_teacher_coarse_loss": rendered_teacher_coarse_loss.detach(),
        "map_rendered_teacher_fine_nce_loss": rendered_teacher_fine_nce_loss.detach(),
        "map_rendered_teacher_coarse_nce_loss": rendered_teacher_coarse_nce_loss.detach(),
        "map_rendered_teacher_fine_grad_loss": rendered_teacher_fine_grad_loss.detach(),
        "map_rendered_teacher_coarse_grad_loss": rendered_teacher_coarse_grad_loss.detach(),
        "map_fine_coarse_ortho_loss": map_fine_coarse_ortho_loss.detach(),
        "map_query_variance_loss": query_variance_loss.detach(),
        "map_variance_loss": map_variance_loss.detach(),
        "map_query_covariance_loss": query_covariance_loss.detach(),
        "map_covariance_loss": map_covariance_loss.detach(),
        "map_query_scene_coord_loss": query_scene_coord_loss.detach(),
        "map_query_scene_coord_warp_loss": query_scene_coord_warp_loss.detach(),
        **query_scene_coord_metrics,
        **query_scene_coord_warp_metrics,
        "map_query_corr_subpx_loss": query_corr_subpx_loss.detach(),
        "map_query_corr_flow_loss": query_corr_flow_loss.detach(),
        "map_query_corr_peak_margin_loss": query_corr_peak_margin_loss.detach(),
        "map_query_corr_wls_pose_loss": query_corr_wls_pose_loss.detach(),
        **query_corr_metrics,
        **query_corr_wls_pose_metrics,
        "map_query_flow_warp_loss": query_flow_warp_loss.detach(),
        **query_flow_warp_metrics,
        "map_query_flow_warp_contrastive_loss": query_flow_warp_contrastive_loss.detach(),
        **query_flow_warp_contrastive_metrics,
        "map_self_corr_subpx_loss": map_self_corr_subpx_loss.detach(),
        "map_self_corr_flow_loss": map_self_corr_flow_loss.detach(),
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
        "map_alpha_coverage_loss": alpha_coverage_loss.detach(),
        "map_rgb_reconstruction_loss": rgb_reconstruction_loss.detach(),
        "map_perturb_rank_loss": perturb_rank_loss.detach(),
        "map_perturb_margin": perturb_margin.detach(),
        "map_query_fine_weight": torch.tensor(query_fine_weight, device=device),
        "map_query_fine_raw_weight": torch.tensor(query_fine_raw_weight, device=device),
        "map_rendered_teacher_fine_weight": torch.tensor(rendered_teacher_fine_weight, device=device),
        "map_rendered_teacher_fine_raw_weight": torch.tensor(
            rendered_teacher_fine_raw_weight, device=device
        ),
        "map_query_fine_infonce_weight": torch.tensor(query_fine_infonce_weight, device=device),
        "map_query_coarse_infonce_weight": torch.tensor(query_coarse_infonce_weight, device=device),
        "map_query_fine_grad_weight": torch.tensor(query_fine_grad_weight, device=device),
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
        "map_fine_coarse_ortho_weight": torch.tensor(fine_coarse_ortho_weight, device=device),
        "map_query_variance_weight": torch.tensor(variance_weight_query, device=device),
        "map_variance_weight": torch.tensor(variance_weight_map, device=device),
        "map_query_covariance_weight": torch.tensor(covariance_weight_query, device=device),
        "map_covariance_weight": torch.tensor(covariance_weight_map, device=device),
        "map_query_scene_coord_weight": torch.tensor(query_scene_coord_weight, device=device),
        "map_query_scene_coord_warp_weight": torch.tensor(query_scene_coord_warp_weight, device=device),
        "map_query_corr_scene_coord_weight": torch.tensor(query_corr_scene_coord_weight, device=device),
        "map_feature_metric_scene_coord_weight": torch.tensor(feature_metric_scene_coord_weight, device=device),
        "map_query_corr_subpixel_weight": torch.tensor(query_corr_subpixel_weight, device=device),
        "map_query_corr_flow_weight": torch.tensor(query_corr_flow_weight, device=device),
        "map_query_corr_peak_margin_weight": torch.tensor(query_corr_peak_margin_weight, device=device),
        "map_query_corr_wls_pose_weight": torch.tensor(query_corr_wls_pose_weight, device=device),
        "map_query_flow_warp_weight": torch.tensor(query_flow_warp_weight, device=device),
        "map_query_flow_warp_contrastive_weight": torch.tensor(
            query_flow_warp_contrastive_weight, device=device
        ),
        "map_self_corr_subpixel_weight": torch.tensor(map_self_corr_subpixel_weight, device=device),
        "map_self_corr_flow_weight": torch.tensor(map_self_corr_flow_weight, device=device),
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
        "map_query_fine_raw_cosine": fine_query_raw_map_cos.detach(),
        "map_query_coarse_cosine": coarse_query_map_cos.detach(),
        "map_alpha_mean": map_alpha_mean.detach(),
        "map_alpha_coverage": map_alpha_coverage.detach(),
        "map_alpha_coverage_valid": map_alpha_coverage_valid.detach(),
        "map_coarse_active": torch.tensor(float(coarse_active), device=device),
    }


def save_validation_visuals(batch, outputs, qual_dir, feature_track_root, step, limit):
    limit = min(limit, batch["rgb"].shape[0])
    for idx in range(limit):
        sample_name = Path(batch["sample_name"][idx]).with_suffix("").as_posix().replace("/", "_")
        filename = f"step{step:06d}_{sample_name}.png"
        for root in [qual_dir, feature_track_root]:
            save_feature_track_visual(
                Path(root) / filename,
                query_rgb=batch["rgb"][idx].detach().cpu(),
                teacher_fine=batch["teacher_fine"][idx].detach().cpu(),
                student_fine=outputs["fine"][idx].detach().cpu(),
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
                main_total, batch_metrics = compute_main_losses(outputs, batch, cfg)
                map_total, map_metrics = compute_map_supervision(batch, outputs, cfg, device, epoch=epoch)
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
        log_msg += " map_q_f=%.4f map_q_c=%.4f map_t_f=%.4f map_t_c=%.4f"
        log_args.extend(
            [
                result.get("map_query_fine_cosine", 0.0),
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
    if "map_query_scene_coord_err_cm" in result:
        log_msg += " scene=%.1fcm"
        log_args.append(result.get("map_query_scene_coord_err_cm", 0.0))
    if "map_query_scene_coord_warp_err_cm" in result:
        log_msg += " scene_warp=%.1fcm"
        log_args.append(result.get("map_query_scene_coord_warp_err_cm", 0.0))
    if "map_query_corr_peak_gap" in result:
        log_msg += " peak_gap=%.4f peak_acc=%.3f"
        log_args.extend(
            [
                result.get("map_query_corr_peak_gap", 0.0),
                result.get("map_query_corr_peak_acc", 0.0),
            ]
        )
    if "map_corr_wls_trans_err_mm" in result:
        log_msg += " corr_wls=%.1fmm gain=%.1fmm"
        log_args.extend(
            [
                result.get("map_corr_wls_trans_err_mm", 0.0),
                result.get("map_corr_wls_trans_gain_mm", 0.0),
            ]
        )
    if "map_self_corr_subpx_flow_epe" in result:
        log_msg += " self_corr_epe=%.3f self_corr_acc=%.3f"
        log_args.extend(
            [
                result.get("map_self_corr_subpx_flow_epe", 0.0),
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

    map_renderer = None
    if cfg.get("map_supervision", {}).get("enabled", False):
        map_renderer = MapFeatureRenderer(
            cfg,
            feature_hw=tuple(cfg["dataset"]["feature_hw"]),
            device=device,
            logger=logger,
        )

    train_dataset = JointRADIOQueryDataset(
        train_records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        retrieval_teacher_store=retrieval_store,
        prior_mask_path=cfg["dataset"].get("prior_mask_path"),
        prior_mask_channels=cfg["dataset"].get("prior_mask_channels"),
    )
    val_dataset = JointRADIOQueryDataset(
        val_records,
        teacher_store,
        input_hw=cfg["dataset"]["input_hw"],
        feature_hw=cfg["dataset"]["feature_hw"],
        synthetic_rgb=bool(cfg["dataset"].get("synthetic_if_missing", False)),
        retrieval_teacher_store=retrieval_store,
        prior_mask_path=cfg["dataset"].get("prior_mask_path"),
        prior_mask_channels=cfg["dataset"].get("prior_mask_channels"),
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

    model = RadioQueryStudent(
        in_channels=3,
        feature_dim=int(cfg["model"]["feature_dim"]),
        fine_feature_dim=fine_feature_dim,
        coarse_feature_dim=coarse_feature_dim,
        base_channels=int(cfg["model"]["base_channels"]),
        stage_dims=tuple(cfg["model"]["stage_dims"]),
        output_hw=tuple(cfg["dataset"]["feature_hw"]),
        coarse_output_hw=tuple(cfg["dataset"].get("coarse_feature_hw") or cfg["dataset"]["feature_hw"]),
        input_hw=tuple(cfg["dataset"]["input_hw"]),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        l2_normalize=bool(cfg["model"].get("l2_normalize", True)),
        predict_magnitude=bool(cfg["model"].get("predict_magnitude", False)),
        fine_init_norm=float(cfg["model"].get("fine_init_norm", 1.0)),
        coarse_init_norm=float(cfg["model"].get("coarse_init_norm", 1.0)),
        magnitude_min=float(cfg["model"].get("magnitude_min", 1e-4)),
        retrieval_dim=int(cfg["retrieval"]["student_dim"]) if retrieval_store is not None else None,
        retrieval_hidden_dim=int(cfg["retrieval"].get("hidden_dim", 0)) if retrieval_store is not None else None,
        retrieval_dropout=float(cfg["retrieval"].get("dropout", 0.0)) if retrieval_store is not None else 0.0,
        retrieval_l2_normalize=bool(cfg["retrieval"].get("l2_normalize", True)),
        fine_low_level_skip=bool(cfg["model"].get("fine_low_level_skip", False)),
        fine_low_level_init=float(cfg["model"].get("fine_low_level_init", 0.0)),
        fine_highres_skip=bool(cfg["model"].get("fine_highres_skip", False)),
        fine_highres_source=str(cfg["model"].get("fine_highres_source", "stage2")),
        fine_highres_init=float(cfg["model"].get("fine_highres_init", 0.0)),
        fine_highres_zero_init=bool(cfg["model"].get("fine_highres_zero_init", False)),
        fine_loc_head=bool(cfg["model"].get("fine_loc_head", False)),
        fine_loc_init=float(cfg["model"].get("fine_loc_init", 1.0)),
        fine_loc_zero_init=bool(cfg["model"].get("fine_loc_zero_init", True)),
        fine_loc_detach_base=bool(cfg["model"].get("fine_loc_detach_base", False)),
        fine_loc_highres_source=cfg["model"].get("fine_loc_highres_source"),
        fine_loc_highres_init=float(cfg["model"].get("fine_loc_highres_init", 1.0)),
        fine_loc_highres_zero_init=bool(cfg["model"].get("fine_loc_highres_zero_init", True)),
        fine_loc_highres_detach=bool(cfg["model"].get("fine_loc_highres_detach", True)),
        teacher_fine_condition=bool(cfg["model"].get("teacher_fine_condition", False)),
        teacher_fine_init=float(cfg["model"].get("teacher_fine_init", 1.0)),
        teacher_fine_zero_init=bool(cfg["model"].get("teacher_fine_zero_init", True)),
        teacher_fine_detach=bool(cfg["model"].get("teacher_fine_detach", True)),
        scene_coord_head=bool(cfg["model"].get("scene_coord_head", False)),
        scene_coord_zero_init=bool(cfg["model"].get("scene_coord_zero_init", True)),
        scene_coord_detach_base=bool(cfg["model"].get("scene_coord_detach_base", False)),
        scene_coord_use_pixel_grid=bool(cfg["model"].get("scene_coord_use_pixel_grid", False)),
        scene_coord_global_context=bool(cfg["model"].get("scene_coord_global_context", False)),
    ).to(device)

    base_lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    optimizer_groups = [
        {
            "params": list(model.parameters()),
            "lr": base_lr,
            "weight_decay": weight_decay,
        }
    ]
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
        load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=warmstart_strict)
        if map_renderer is not None:
            map_renderer.load_trainable_state(checkpoint.get("map_renderer_state_dict"))
        if warmstart_strict:
            logger.info("Warmstarted weights from %s", args.warmstart)
        else:
            logger.info(
                "Warmstarted weights from %s (strict=%s, missing=%s, unexpected=%s)",
                args.warmstart,
                warmstart_strict,
                load_result.missing_keys,
                load_result.unexpected_keys,
            )

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
                if bool(cfg["model"].get("teacher_fine_condition", False)):
                    outputs = model(batch["rgb"], teacher_fine=batch.get("teacher_fine"))
                else:
                    outputs = model(batch["rgb"])
                main_total, metrics = compute_main_losses(outputs, batch, cfg)
                map_total, map_metrics = compute_map_supervision(batch, outputs, cfg, device, epoch=epoch)
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
                    log_msg += " map_q_f=%.4f map_q_c=%.4f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_fine_cosine", 0.0),
                            mean_train.get("map_query_coarse_cosine", 0.0),
                        ]
                    )
                if "map_query_fine_raw_cosine" in mean_train:
                    log_msg += " map_q_f_raw=%.4f"
                    log_args.append(mean_train.get("map_query_fine_raw_cosine", 0.0))
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
                if "map_query_scene_coord_err_cm" in mean_train:
                    log_msg += " scene=%.1fcm"
                    log_args.append(mean_train.get("map_query_scene_coord_err_cm", 0.0))
                if "map_query_scene_coord_warp_err_cm" in mean_train:
                    log_msg += " scene_warp=%.1fcm"
                    log_args.append(mean_train.get("map_query_scene_coord_warp_err_cm", 0.0))
                if "map_query_corr_peak_gap" in mean_train:
                    log_msg += " peak_gap=%.4f peak_acc=%.3f"
                    log_args.extend(
                        [
                            mean_train.get("map_query_corr_peak_gap", 0.0),
                            mean_train.get("map_query_corr_peak_acc", 0.0),
                        ]
                    )
                if "map_corr_wls_trans_err_mm" in mean_train:
                    log_msg += " corr_wls=%.1fmm gain=%.1fmm"
                    log_args.extend(
                        [
                            mean_train.get("map_corr_wls_trans_err_mm", 0.0),
                            mean_train.get("map_corr_wls_trans_gain_mm", 0.0),
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
            if val_metrics.get("loss_total", float("inf")) < best_val:
                best_val = float(val_metrics["loss_total"])
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
                "map q_f={:.4f} q_c={:.4f} t_f={:.4f} t_c={:.4f}".format(
                    float(latest_val_metrics.get("map_query_fine_cosine", 0.0)),
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
