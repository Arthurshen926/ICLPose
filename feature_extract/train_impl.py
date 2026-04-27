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
import logging
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
    infonce_contrastive_loss,
    l1_feature_loss,
)
from feature_extract import load_config as load_feature_extract_config
from feature_extract.students.radio_query_student import RadioQueryStudent
from feature_field import build_dcff_runtime, intrinsics_to_K
from feature_field.runtime import _apply_dcff_postprocess
from feature_field.utils.loc_reporting import save_experiment_bundle
from feature_field.utils.feature_track_vis import save_feature_track_visual


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
        "base_channels": 32,
        "stage_dims": [32, 64, 96, 128],
        "dropout": 0.0,
        "l2_normalize": True,
        "predict_magnitude": False,
        "fine_init_norm": 1.0,
        "coarse_init_norm": 1.0,
        "magnitude_min": 1e-4,
        "warmstart_strict": True,
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
        "perturb_render_negatives": False,
        "perturb_rot_deg": 0.0,
        "perturb_trans_m": 0.0,
        "rendered_teacher_fine_weight": 0.0,
        "rendered_teacher_fine_raw_weight": 0.0,
        "rendered_teacher_coarse_weight": 0.0,
        "rendered_teacher_fine_infonce_weight": 0.0,
        "rendered_teacher_coarse_infonce_weight": 0.0,
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
        self.feature_dim = int(sample.shape[0])
        self.feature_hw = (int(sample.shape[1]), int(sample.shape[2]))
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
    images = discover_images(dataset_cfg["source_dir"], dataset_cfg["image_patterns"])
    records = []
    if images:
        for teacher_idx in teacher_store.indices:
            if teacher_idx >= len(images):
                continue
            image_path = images[teacher_idx]
            rel_name = image_path.relative_to(dataset_cfg["source_dir"]).as_posix()
            records.append(
                {
                    "teacher_idx": teacher_idx,
                    "image_path": str(image_path),
                    "sample_name": rel_name,
                    "normalized_name": rel_name.replace("\\", "/"),
                }
            )
    elif allow_synthetic:
        for teacher_idx in teacher_store.indices:
            records.append(
                {
                    "teacher_idx": teacher_idx,
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
        self.alpha_threshold = float(map_cfg.get("alpha_threshold", 0.5))
        self.trainable = bool(map_cfg.get("trainable", False))
        self.train_fine_decoder = self.trainable and bool(map_cfg.get("train_fine_decoder", False))
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
            "Map renderer loaded: config=%s, views=%d, feature_hw=%s, cache=%s, trainable=%s, fine_decoder=%s, feat_sharp=%s, fsm=%s, hash_mlp=%s, latent=%s, geometry=%s",
            config_path,
            len(self.name_to_pose),
            self.feature_hw,
            self.cache_in_memory,
            self.trainable,
            self.train_fine_decoder,
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

    def _perturb_w2c_pose(self, pose):
        pose_np = pose.detach().cpu().numpy().astype(np.float32, copy=True)
        rot_sigma = np.deg2rad(max(0.0, self.perturb_rot_deg))
        trans_sigma = max(0.0, self.perturb_trans_m)
        if rot_sigma <= 0 and trans_sigma <= 0:
            return torch.tensor(pose_np.tolist(), dtype=torch.float32)
        rx, ry, rz = np.random.normal(0.0, rot_sigma, size=3).astype(np.float32)
        tx, ty, tz = np.random.normal(0.0, trans_sigma, size=3).astype(np.float32)

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        delta_r = rot_z @ rot_y @ rot_x
        pose_np[:3, :3] = delta_r @ pose_np[:3, :3]
        pose_np[:3, 3] = pose_np[:3, 3] + np.array([tx, ty, tz], dtype=np.float32)
        return torch.tensor(pose_np.tolist(), dtype=torch.float32)

    def has_trainable_params(self):
        return (
            self.train_fine_decoder
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
            fine_raw_cpu, fine_cpu, coarse_cpu, mask_cpu, alpha_cpu, rgb_cpu, depth_cpu = self._cache[normalized]
            return (
                fine_raw_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                fine_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                coarse_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                mask_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                alpha_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                rgb_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
                depth_cpu.to(device=self.device, dtype=torch.float32, non_blocking=True),
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

        if use_cache:
            self._cache[normalized] = (
                fine_raw.detach().cpu().to(self.cache_dtype),
                fine.detach().cpu().to(self.cache_dtype),
                coarse.detach().cpu().to(self.cache_dtype),
                mask.detach().cpu().to(self.cache_dtype),
                alpha_feat.detach().cpu().to(self.cache_dtype),
                rgb.detach().cpu().to(self.cache_dtype),
                depth.detach().cpu().to(self.cache_dtype),
            )
        return fine_raw, fine, coarse, mask, alpha_feat, rgb, depth

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
        return fine_raw, fine, coarse, mask, alpha_feat, rgb, depth

    def attach_to_batch(self, batch, require_grad=False):
        fine_raw_list = []
        fine_list = []
        coarse_list = []
        mask_list = []
        alpha_list = []
        rgb_list = []
        depth_list = []
        context = torch.enable_grad if require_grad else torch.no_grad
        with context():
            for sample_name in batch["sample_name"]:
                fine_raw, fine, coarse, mask, alpha, rgb, depth = self._render_single(sample_name, require_grad=require_grad)
                fine_raw_list.append(fine_raw.squeeze(0))
                fine_list.append(fine.squeeze(0))
                coarse_list.append(coarse.squeeze(0))
                mask_list.append(mask.squeeze(0))
                alpha_list.append(alpha.squeeze(0))
                rgb_list.append(rgb.squeeze(0))
                depth_list.append(depth.squeeze(0))

        neg_fine_raw_list = []
        neg_fine_list = []
        neg_coarse_list = []
        neg_mask_list = []
        neg_alpha_list = []
        if self.perturb_render_negatives:
            with context():
                for sample_name in batch["sample_name"]:
                    normalized = self._normalize_name(sample_name)
                    neg_pose = self._perturb_w2c_pose(self.name_to_pose[normalized])
                    fine_raw, fine, coarse, mask, alpha, _rgb, _depth = self._render_pose(
                        sample_name,
                        neg_pose,
                        require_grad=require_grad,
                    )
                    neg_fine_raw_list.append(fine_raw.squeeze(0))
                    neg_fine_list.append(fine.squeeze(0))
                    neg_coarse_list.append(coarse.squeeze(0))
                    neg_mask_list.append(mask.squeeze(0))
                    neg_alpha_list.append(alpha.squeeze(0))

        batch["rendered_map_fine_raw"] = torch.stack(fine_raw_list, dim=0)
        batch["rendered_map_fine"] = torch.stack(fine_list, dim=0)
        batch["rendered_map_coarse"] = torch.stack(coarse_list, dim=0)
        batch["rendered_map_mask"] = torch.stack(mask_list, dim=0)
        batch["rendered_map_alpha"] = torch.stack(alpha_list, dim=0)
        batch["rendered_map_rgb"] = torch.stack(rgb_list, dim=0)
        batch["rendered_map_depth"] = torch.stack(depth_list, dim=0)
        if neg_fine_list:
            batch["rendered_map_fine_neg"] = torch.stack(neg_fine_list, dim=0)
            batch["rendered_map_coarse_neg"] = torch.stack(neg_coarse_list, dim=0)
            batch["rendered_map_mask_neg"] = torch.stack(neg_mask_list, dim=0)
            batch["rendered_map_alpha_neg"] = torch.stack(neg_alpha_list, dim=0)
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


def feature_orthogonality_loss(pred_a, pred_b, mask=None):
    pred_a_n = F.normalize(pred_a, dim=1)
    pred_b_n = F.normalize(pred_b, dim=1)
    cos = (pred_a_n * pred_b_n).sum(dim=1, keepdim=True)
    penalty = cos.square()
    if mask is not None:
        penalty = penalty * mask
        denom = mask.sum().clamp(min=1.0)
        return penalty.sum() / denom
    return penalty.mean()


def compute_main_losses(outputs, batch, cfg):
    loss_cfg = cfg["loss"]
    teacher_fine = batch["teacher_fine"]
    teacher_coarse = batch["teacher_coarse"]
    pred_fine = outputs["fine"]
    pred_coarse = outputs["coarse"]

    fine_l1 = l1_feature_loss(pred_fine, teacher_fine)
    fine_cos = cosine_loss(pred_fine, teacher_fine)
    fine_cs = channel_standardized_loss(pred_fine, teacher_fine)
    coarse_l1 = l1_feature_loss(pred_coarse, teacher_coarse)
    coarse_cos = cosine_loss(pred_coarse, teacher_coarse)
    coarse_cs = channel_standardized_loss(pred_coarse, teacher_coarse)

    total = (
        loss_cfg["fine_l1_weight"] * fine_l1
        + loss_cfg["fine_cos_weight"] * fine_cos
        + float(loss_cfg.get("fine_channel_std_weight", 0.0)) * fine_cs
        + loss_cfg["coarse_l1_weight"] * coarse_l1
        + loss_cfg["coarse_cos_weight"] * coarse_cos
        + float(loss_cfg.get("coarse_channel_std_weight", 0.0)) * coarse_cs
    )
    fine_coarse_ortho = feature_orthogonality_loss(pred_fine, pred_coarse)
    total = total + float(loss_cfg.get("fine_coarse_ortho_weight", 0.0)) * fine_coarse_ortho

    metrics = {
        "loss_total": total.detach(),
        "fine_l1": fine_l1.detach(),
        "fine_cos_loss": fine_cos.detach(),
        "fine_channel_std_loss": fine_cs.detach(),
        "coarse_l1": coarse_l1.detach(),
        "coarse_cos_loss": coarse_cos.detach(),
        "coarse_channel_std_loss": coarse_cs.detach(),
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
        )
        coarse_nce = infonce_contrastive_loss(
            pred_coarse,
            teacher_coarse,
            temperature=float(loss_cfg.get("infonce_temperature", 0.07)),
            n_samples=int(loss_cfg.get("infonce_samples", 256)),
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
    rendered_fine_raw = batch.get("rendered_map_fine_raw")
    rendered_fine = batch["rendered_map_fine"]
    rendered_coarse = batch["rendered_map_coarse"]
    teacher_fine = batch["teacher_fine"]
    teacher_coarse = batch["teacher_coarse"]
    pred_fine = outputs["fine"]
    pred_coarse = outputs["coarse"]
    detach_query_features = bool(map_cfg.get("detach_query_features", False))
    pred_fine_target = pred_fine.detach() if detach_query_features else pred_fine
    pred_coarse_target = pred_coarse.detach() if detach_query_features else pred_coarse
    coarse_active = int(epoch) >= int(map_cfg.get("coarse_start_epoch", 0))
    query_coarse_weight = float(map_cfg.get("query_coarse_weight", 0.0)) if coarse_active else 0.0
    rendered_teacher_coarse_weight = (
        float(map_cfg.get("rendered_teacher_coarse_weight", 0.0)) if coarse_active else 0.0
    )

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
    rendered_teacher_fine_infonce_weight = resolve_linear_weight(
        map_cfg, "rendered_teacher_fine_infonce_weight", epoch
    )
    rendered_teacher_coarse_infonce_weight = (
        resolve_linear_weight(map_cfg, "rendered_teacher_coarse_infonce_weight", epoch)
        if coarse_active
        else 0.0
    )
    infonce_temperature = float(map_cfg.get("infonce_temperature", loss_cfg.get("infonce_temperature", 0.07)))
    infonce_samples = int(map_cfg.get("infonce_samples", loss_cfg.get("infonce_samples", 256)))
    fine_coarse_ortho_weight = float(map_cfg.get("fine_coarse_ortho_weight", 0.0))
    alpha_coverage_weight = resolve_linear_weight(map_cfg, "alpha_coverage_weight", epoch)
    rgb_l1_weight = resolve_linear_weight(map_cfg, "rgb_l1_weight", epoch)

    query_fine_loss = (
        l1_feature_loss(pred_fine_target, rendered_fine, mask)
        + cosine_loss(pred_fine_target, rendered_fine, mask)
    ) * query_fine_weight
    query_fine_raw_loss = zero
    rendered_teacher_fine_raw_loss = zero
    if rendered_fine_raw is not None:
        query_fine_raw_loss = (
            l1_feature_loss(pred_fine_target, rendered_fine_raw, mask)
            + cosine_loss(pred_fine_target, rendered_fine_raw, mask)
        ) * query_fine_raw_weight
        rendered_teacher_fine_raw_loss = (
            l1_feature_loss(rendered_fine_raw, teacher_fine, mask)
            + cosine_loss(rendered_fine_raw, teacher_fine, mask)
        ) * rendered_teacher_fine_raw_weight
    query_coarse_loss = (
        l1_feature_loss(pred_coarse_target, rendered_coarse, mask)
        + cosine_loss(pred_coarse_target, rendered_coarse, mask)
    ) * query_coarse_weight
    query_fine_nce_loss = zero
    if query_fine_infonce_weight > 0:
        query_fine_nce_loss = infonce_contrastive_loss(
            pred_fine_target,
            rendered_fine,
            mask=mask,
            temperature=infonce_temperature,
            n_samples=infonce_samples,
        ) * query_fine_infonce_weight
    query_coarse_nce_loss = zero
    if query_coarse_infonce_weight > 0:
        query_coarse_nce_loss = infonce_contrastive_loss(
            pred_coarse_target,
            rendered_coarse,
            mask=mask,
            temperature=infonce_temperature,
            n_samples=infonce_samples,
        ) * query_coarse_infonce_weight
    rendered_teacher_fine_loss = (
        l1_feature_loss(rendered_fine, teacher_fine, mask)
        + cosine_loss(rendered_fine, teacher_fine, mask)
    ) * rendered_teacher_fine_weight
    rendered_teacher_fine_nce_loss = zero
    if rendered_teacher_fine_infonce_weight > 0:
        rendered_teacher_fine_nce_loss = infonce_contrastive_loss(
            rendered_fine,
            teacher_fine,
            mask=mask,
            temperature=infonce_temperature,
            n_samples=infonce_samples,
        ) * rendered_teacher_fine_infonce_weight
    rendered_teacher_coarse_loss = (
        l1_feature_loss(rendered_coarse, teacher_coarse, mask)
        + cosine_loss(rendered_coarse, teacher_coarse, mask)
    ) * rendered_teacher_coarse_weight
    rendered_teacher_coarse_nce_loss = zero
    if rendered_teacher_coarse_infonce_weight > 0:
        rendered_teacher_coarse_nce_loss = infonce_contrastive_loss(
            rendered_coarse,
            teacher_coarse,
            mask=mask,
            temperature=infonce_temperature,
            n_samples=infonce_samples,
        ) * rendered_teacher_coarse_infonce_weight
    map_fine_coarse_ortho_loss = feature_orthogonality_loss(rendered_fine, rendered_coarse, mask) * fine_coarse_ortho_weight

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
    if perturb_rank_weight > 0:
        margin = float(map_cfg.get("perturb_margin", 0.1))
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
            pos_fine = l1_feature_loss(pred_fine_target, rendered_fine, mask) + cosine_loss(
                pred_fine_target, rendered_fine, mask
            )
            neg_fine_loss = l1_feature_loss(pred_fine_target, neg_fine, neg_mask) + cosine_loss(
                pred_fine_target, neg_fine, neg_mask
            )
            pos_coarse = l1_feature_loss(pred_coarse_target, rendered_coarse, mask) + cosine_loss(
                pred_coarse_target, rendered_coarse, mask
            )
            neg_coarse_loss = l1_feature_loss(pred_coarse_target, neg_coarse, neg_mask) + cosine_loss(
                pred_coarse_target, neg_coarse, neg_mask
            )
            perturb_rank_loss = F.relu(margin + 0.5 * (pos_fine + pos_coarse) - 0.5 * (neg_fine_loss + neg_coarse_loss))
    total = (
        query_fine_loss
        + query_fine_raw_loss
        + query_coarse_loss
        + query_fine_nce_loss
        + query_coarse_nce_loss
        + rendered_teacher_fine_loss
        + rendered_teacher_fine_raw_loss
        + rendered_teacher_coarse_loss
        + rendered_teacher_fine_nce_loss
        + rendered_teacher_coarse_nce_loss
        + map_fine_coarse_ortho_loss
        + alpha_coverage_loss
        + rgb_reconstruction_loss
        + perturb_rank_weight * perturb_rank_loss
    )

    fine_map_teacher_cos = 1.0 - cosine_loss(rendered_fine, teacher_fine, mask)
    coarse_map_teacher_cos = 1.0 - cosine_loss(rendered_coarse, teacher_coarse, mask)
    fine_query_map_cos = 1.0 - cosine_loss(pred_fine, rendered_fine, mask)
    coarse_query_map_cos = 1.0 - cosine_loss(pred_coarse, rendered_coarse, mask)
    if rendered_fine_raw is not None:
        fine_map_raw_teacher_cos = 1.0 - cosine_loss(rendered_fine_raw, teacher_fine, mask)
        fine_query_raw_map_cos = 1.0 - cosine_loss(pred_fine, rendered_fine_raw, mask)
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
        "map_rendered_teacher_fine_loss": rendered_teacher_fine_loss.detach(),
        "map_rendered_teacher_fine_raw_loss": rendered_teacher_fine_raw_loss.detach(),
        "map_rendered_teacher_coarse_loss": rendered_teacher_coarse_loss.detach(),
        "map_rendered_teacher_fine_nce_loss": rendered_teacher_fine_nce_loss.detach(),
        "map_rendered_teacher_coarse_nce_loss": rendered_teacher_coarse_nce_loss.detach(),
        "map_fine_coarse_ortho_loss": map_fine_coarse_ortho_loss.detach(),
        "map_alpha_coverage_loss": alpha_coverage_loss.detach(),
        "map_rgb_reconstruction_loss": rgb_reconstruction_loss.detach(),
        "map_perturb_rank_loss": perturb_rank_loss.detach(),
        "map_query_fine_weight": torch.tensor(query_fine_weight, device=device),
        "map_query_fine_raw_weight": torch.tensor(query_fine_raw_weight, device=device),
        "map_rendered_teacher_fine_weight": torch.tensor(rendered_teacher_fine_weight, device=device),
        "map_rendered_teacher_fine_raw_weight": torch.tensor(
            rendered_teacher_fine_raw_weight, device=device
        ),
        "map_query_fine_infonce_weight": torch.tensor(query_fine_infonce_weight, device=device),
        "map_query_coarse_infonce_weight": torch.tensor(query_coarse_infonce_weight, device=device),
        "map_rendered_teacher_fine_infonce_weight": torch.tensor(
            rendered_teacher_fine_infonce_weight, device=device
        ),
        "map_rendered_teacher_coarse_infonce_weight": torch.tensor(
            rendered_teacher_coarse_infonce_weight, device=device
        ),
        "map_fine_coarse_ortho_weight": torch.tensor(fine_coarse_ortho_weight, device=device),
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
    if tuple(cfg["dataset"]["feature_hw"]) != teacher_store.feature_hw:
        logger.info(
            "Overriding feature_hw from config %s -> teacher cache %s",
            cfg["dataset"]["feature_hw"],
            teacher_store.feature_hw,
        )
        cfg["dataset"]["feature_hw"] = list(teacher_store.feature_hw)

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
        base_channels=int(cfg["model"]["base_channels"]),
        stage_dims=tuple(cfg["model"]["stage_dims"]),
        output_hw=tuple(cfg["dataset"]["feature_hw"]),
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
