from __future__ import annotations

"""Teacher target providers used by DCFF evaluation and visualization."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from feature_field.dcff.radio_teacher import CachedFeatureTeacher, OnlineRadioTeacher
from feature_field.utils.scene_colmap import build_da3_image_order


def infer_feature_dims(cfg: dict[str, Any]) -> tuple[int, int]:
    """Return fine/coarse feature dimensions from a DCFF checkpoint config."""
    mcfg = cfg.get("model", {})
    fallback = int(mcfg.get("feature_dim", 64))
    fine_dim = int(mcfg.get("fine_feature_dim", fallback))
    coarse_dim = int(mcfg.get("coarse_feature_dim", fallback))
    return fine_dim, coarse_dim


def infer_teacher_mode(cfg: dict[str, Any]) -> str:
    return str(cfg.get("teacher", {}).get("mode", "cached"))


def resize_image_tensor(cam, longest_edge: int, device: torch.device | str) -> torch.Tensor:
    """Load one camera image as [3,H,W] float tensor after longest-edge resize."""
    img = Image.open(cam.image).convert("RGB")
    width, height = img.size
    scale = float(longest_edge) / max(width, height)
    if scale < 1.0:
        img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device=device)


def load_image_batch(cams, longest_edge: int, device: torch.device | str) -> torch.Tensor:
    return torch.stack([resize_image_tensor(cam, longest_edge, device) for cam in cams], dim=0)


class CachedTeacherTargetProvider:
    """Adapter around the legacy cached PCA teacher features."""

    def __init__(self, feature_dir: str | os.PathLike[str], images_dir: str | os.PathLike[str], device: torch.device | str):
        self.mode = "cached"
        self.device = torch.device(device)
        self.cache = CachedFeatureTeacher(str(feature_dir))
        self.name_to_fid = build_da3_image_order(str(images_dir))
        self.feature_resolution = (self.cache.feat_h, self.cache.feat_w)
        self.fine_dim = self.cache.feature_dim
        self.coarse_dim = self.cache.feature_dim

    def camera_fid(self, cam) -> int | None:
        fid = self.name_to_fid.get(cam.image_name)
        if fid is None or fid not in self.cache.frame_ids:
            return None
        return fid

    def filter_cameras(self, cams):
        return [cam for cam in cams if self.camera_fid(cam) is not None]

    def get_batch(self, cams) -> tuple[torch.Tensor, torch.Tensor, list[int | None]]:
        fids = [self.camera_fid(cam) for cam in cams]
        if any(fid is None for fid in fids):
            missing = [cam.image_name for cam, fid in zip(cams, fids) if fid is None]
            raise KeyError(f"Missing cached teacher features for: {missing[:4]}")

        fine_batch, coarse_batch = [], []
        for fid in fids:
            fine, coarse = self.cache.get(fid)
            fine_batch.append(fine.to(self.device))
            coarse_batch.append(coarse.to(self.device))
        return torch.stack(fine_batch, dim=0), torch.stack(coarse_batch, dim=0), fids


class OnlineTeacherTargetProvider:
    """Online RADIO target provider for learned projection/bottleneck checkpoints."""

    def __init__(
        self,
        cfg: dict[str, Any],
        ckpt: dict[str, Any],
        device: torch.device | str,
        teacher_factory=OnlineRadioTeacher,
    ):
        self.mode = infer_teacher_mode(cfg)
        self.device = torch.device(device)
        self.cfg = cfg
        self.ckpt = ckpt
        self.fine_dim, self.coarse_dim = infer_feature_dims(cfg)

        teacher_cfg = cfg.get("teacher", {})
        compress_cfg = teacher_cfg.get("compress", {})
        pca_init_dir = teacher_cfg.get("pca_init_dir")
        if pca_init_dir is None:
            feature_dir = cfg.get("dataset", {}).get("feature_dir", "")
            candidate = Path(feature_dir) / "pca_params"
            pca_init_dir = str(candidate) if candidate.is_dir() else None

        self.input_longest_edge = int(
            teacher_cfg.get("input_longest_edge", cfg.get("training", {}).get("longest_edge", 960))
        )
        self.teacher = teacher_factory(
            target_dim=int(cfg.get("model", {}).get("feature_dim", self.fine_dim)),
            fine_dim=self.fine_dim,
            coarse_dim=self.coarse_dim,
            bottleneck=(self.mode == "online_bottleneck"),
            shallow_block=teacher_cfg.get("shallow_block", 10),
            radio_repo=teacher_cfg.get("radio_repo", "feature_extract/checkpoints/RADIO"),
            pca_init_dir=pca_init_dir,
            compress_hidden_dim=int(compress_cfg.get("hidden_dim", 256)),
            sample_pixels=int(compress_cfg.get("sample_pixels", 1024)),
            recon_chunk_pixels=int(compress_cfg.get("recon_chunk_pixels", 4096)),
            recon_cos_weight=float(compress_cfg.get("recon_cos_weight", 1.0)),
            recon_l1_weight=float(compress_cfg.get("recon_l1_weight", 0.25)),
            min_spatial_std=float(compress_cfg.get("min_spatial_std", 0.0)),
            std_weight=float(compress_cfg.get("std_weight", 0.0)),
            decorrelation_weight=float(compress_cfg.get("decorrelation_weight", 0.0)),
            fine_raw_highpass_kernel=int(compress_cfg.get("fine_raw_highpass_kernel", 0)),
            coarse_raw_highpass_kernel=int(compress_cfg.get("coarse_raw_highpass_kernel", 0)),
            fine_adapter_highpass_kernel=int(compress_cfg.get("fine_adapter_highpass_kernel", 0)),
            coarse_adapter_highpass_kernel=int(compress_cfg.get("coarse_adapter_highpass_kernel", 0)),
            normalize_output=bool(compress_cfg.get("normalize_output", True)),
        ).to(self.device)
        if "projection_state" in ckpt:
            self.teacher.load_projection_state(ckpt["projection_state"])
        self.teacher.eval()
        self.feature_resolution: tuple[int, int] | None = None

    def _prepare_resolution(self, cam) -> None:
        width, height = Image.open(cam.image).size
        scale = float(self.input_longest_edge) / max(width, height)
        if scale < 1.0:
            width, height = int(width * scale), int(height * scale)
        self.teacher.set_image_size(height, width)
        self.feature_resolution = self.teacher.feature_resolution

    def filter_cameras(self, cams):
        return list(cams)

    def get_batch(self, cams) -> tuple[torch.Tensor, torch.Tensor, list[None]]:
        if not cams:
            raise ValueError("get_batch() requires at least one camera")
        self._prepare_resolution(cams[0])
        images = load_image_batch(cams, self.input_longest_edge, self.device)
        with torch.no_grad():
            fine_raw, coarse_raw = self.teacher.extract_raw(images)
            if self.mode == "online_bottleneck":
                fine, coarse, _ = self.teacher.project(fine_raw, coarse_raw, return_loss=True)
            else:
                fine, coarse = self.teacher.project(fine_raw, coarse_raw)
        self.feature_resolution = (fine.shape[-2], fine.shape[-1])
        return fine.detach(), coarse.detach(), [None for _ in cams]


def default_images_dir(source_dir: str | os.PathLike[str], images_subdir: str = "") -> str:
    source_dir = str(source_dir)
    if images_subdir:
        return os.path.join(source_dir, images_subdir)
    images_dir = os.path.join(source_dir, "images")
    return images_dir if os.path.isdir(images_dir) else source_dir


def build_teacher_target_provider(
    cfg: dict[str, Any],
    ckpt: dict[str, Any],
    source_dir: str | os.PathLike[str],
    feature_dir: str | os.PathLike[str] | None,
    device: torch.device | str,
    images_subdir: str = "",
):
    mode = infer_teacher_mode(cfg)
    if mode in {"online", "online_bottleneck"}:
        return OnlineTeacherTargetProvider(cfg=cfg, ckpt=ckpt, device=device)
    if not feature_dir:
        raise ValueError("feature_dir is required for cached teacher evaluation")
    return CachedTeacherTargetProvider(
        feature_dir=feature_dir,
        images_dir=default_images_dir(source_dir, images_subdir),
        device=device,
    )
