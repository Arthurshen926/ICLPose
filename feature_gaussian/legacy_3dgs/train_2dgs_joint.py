#!/usr/bin/env python3
"""
Joint 2DGS Geometry + DA3 Feature Embedding Training
=====================================================
训练2DGS几何参数(xyz, scaling, rotation, opacity, SH)和DA3特征嵌入
在单一训练循环中同时优化，而非解耦的两阶段训练。

关键优势:
  - 特征loss梯度回传到几何参数 → 几何自动适应特征需求
  - 避免两阶段误差累积 (冻结几何的误差不再被特征训练继承)
  - 密度化(clone/split/prune)自动维护特征嵌入

架构:
  GaussianModel2DGSJoint 继承 GaussianModel2DGS，添加:
    - 每个scale的可学习特征嵌入 _feat_{scale}: [N, D]
    - 密度化操作自动处理特征嵌入的复制/分裂/剪枝
    - 特征通过 rasterization_2dgs 渲染 + channel chunking

用法:
  CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_joint \\
    --source_dir dataset/OldHospital \\
    --images processed \\
    --feature_dir output/features_da3/OldHospital_indexed \\
    --feature_scales coarse,mid,fine \\
    --model_dir output/2dgs_joint/OldHospital/v1 \\
    --iterations 30000 \\
    --feature_weight 0.1 \\
    --feature_embedding_lr 0.01
"""

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from random import randint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from gsplat import rasterization_2dgs, rasterization

# Import base utilities (no global args dependency in these)
from feature_gaussian.legacy_3dgs.train_2dgs_geometry import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
    qvec2rotmat,
    focal2fov,
    RGB2SH,
    inverse_sigmoid,
    build_rotation,
    GaussianModel2DGS,
    CameraData,
    ssim,
    load_image_tensor,
)


# ════════════════════════════════════════════════════════════════════════════
# DA3 Feature Cache
# ════════════════════════════════════════════════════════════════════════════

class DA3FeatureCache:
    """Pre-loads multi-scale DA3 features into GPU memory."""

    def __init__(self, feature_dir, scales, device='cuda'):
        self.data = {}
        feature_dir = Path(feature_dir)

        for scale in scales:
            scale_dir = feature_dir / scale
            if not scale_dir.is_dir():
                raise FileNotFoundError(f"Feature dir not found: {scale_dir}")

            files = {}
            for fpath in sorted(scale_dir.glob(f'rgb_*_{scale}_*.pt')):
                m = re.search(r'rgb_(\d+)_', fpath.name)
                if m:
                    files[int(m.group(1))] = fpath

            if not files:
                raise FileNotFoundError(f"No features in {scale_dir}")

            sample = torch.load(str(files[min(files.keys())]),
                                map_location='cpu', weights_only=True).float()
            dim, h, w = sample.shape
            print(f"  [{scale}] {len(files)} frames, {dim}d @ {w}×{h}")

            feats = {}
            for fid, fpath in sorted(files.items()):
                feat = torch.load(str(fpath), map_location='cpu',
                                  weights_only=True).float()
                feat = F.normalize(feat, p=2, dim=0)
                feats[fid] = feat.to(device)

            self.data[scale] = {'feats': feats, 'dim': dim, 'h': h, 'w': w}
            mem_mb = sum(f.nelement() * 4 for f in feats.values()) / 1024**2
            print(f"         GPU cache: {mem_mb:.0f} MB")

    def get(self, scale, frame_id):
        return self.data[scale]['feats'].get(frame_id)

    def info(self, scale):
        d = self.data[scale]
        return d['dim'], d['h'], d['w']

    def frame_ids(self, scale):
        return set(self.data[scale]['feats'].keys())


def build_da3_image_order(image_dir):
    """
    Reconstruct the DA3 feature extraction image ordering.

    DA3 features are indexed by the order images were discovered by
    find_images() in extract_da3_features.py:
      - seq*/frame*.png (Cambridge): sorted seq dirs, sorted frames within
      - Sequence_*/rgb/ (Replica): sorted sequence dirs, sorted frames
      - Flat directory: sorted filenames

    Returns:
        dict: image_name (relative to image_dir) → DA3 frame_id
    """
    import glob

    name_to_fid = {}
    global_idx = 0

    # Cambridge-style: seq*/frame*.png
    seq_dirs = sorted(glob.glob(os.path.join(image_dir, "seq*")))
    if seq_dirs and all(os.path.isdir(d) for d in seq_dirs):
        for seq_dir in seq_dirs:
            for fname in sorted(os.listdir(seq_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    rel = os.path.join(
                        os.path.basename(seq_dir), fname
                    )
                    name_to_fid[rel] = global_idx
                    global_idx += 1
        if name_to_fid:
            return name_to_fid

    # Replica-style: Sequence_*/rgb/
    seq_dirs = sorted(glob.glob(os.path.join(image_dir, "Sequence_*")))
    if seq_dirs:
        for seq_dir in seq_dirs:
            rgb_dir = os.path.join(seq_dir, "rgb")
            if not os.path.isdir(rgb_dir):
                rgb_dir = seq_dir
            for fname in sorted(os.listdir(rgb_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    rel = os.path.relpath(
                        os.path.join(rgb_dir, fname), image_dir
                    )
                    name_to_fid[rel] = global_idx
                    global_idx += 1
        return name_to_fid

    # Flat directory
    for fname in sorted(os.listdir(image_dir)):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            name_to_fid[fname] = global_idx
            global_idx += 1

    return name_to_fid


# ════════════════════════════════════════════════════════════════════════════
# Joint Gaussian Model (Geometry + Features)
# ════════════════════════════════════════════════════════════════════════════

class GaussianModel2DGSJoint(GaussianModel2DGS):
    """
    2DGS with trainable feature embeddings for joint geometry+feature training.

    Extends GaussianModel2DGS with per-scale feature embeddings that are:
    - Jointly optimized alongside geometry parameters
    - Automatically handled during densification (clone/split/prune)
    - Saved separately for downstream pose estimation
    """

    def __init__(self, sh_degree=3, feature_scales=None):
        """
        Args:
            feature_scales: dict {scale_name: dim},
                e.g. {'coarse': 32, 'mid': 64, 'fine': 64}
        """
        super().__init__(sh_degree)
        self.feature_scales = feature_scales or {}

    def create_from_pcd(self, xyz, colors, spatial_lr_scale):
        super().create_from_pcd(xyz, colors, spatial_lr_scale)
        self.init_feature_embeddings()

    def init_feature_embeddings(self):
        N = self.num_points
        for sn, dim in self.feature_scales.items():
            feat = F.normalize(torch.randn(N, dim, device="cuda"), p=2, dim=-1)
            setattr(self, f'_feat_{sn}',
                    nn.Parameter(feat.requires_grad_(True)))
            print(f"  Feature [{sn}]: [{N}, {dim}]")

    def training_setup(self, args):
        super().training_setup(args)
        feat_lr = getattr(args, 'feature_embedding_lr', 0.01)
        for sn in self.feature_scales:
            self.optimizer.add_param_group({
                'params': [getattr(self, f'_feat_{sn}')],
                'lr': feat_lr,
                'name': f'feat_{sn}',
            })

    def get_feature(self, scale_name):
        """Return L2-normalized feature embedding [N, D]."""
        return F.normalize(getattr(self, f'_feat_{scale_name}'), p=2, dim=-1)

    # ── Densification overrides ──

    def _densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected = torch.norm(grads, dim=-1) >= grad_threshold
        selected = selected & (
            self.get_scaling.max(dim=1).values
            <= self.percent_dense * scene_extent
        )

        feat_new = {
            sn: getattr(self, f'_feat_{sn}')[selected]
            for sn in self.feature_scales
        }
        self._densification_postfix(
            self._xyz[selected],
            self._features_dc[selected],
            self._features_rest[selected],
            self._opacity[selected],
            self._scaling[selected],
            self._rotation[selected],
            feat_new,
        )

    def _densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init = self.num_points
        padded_grad = torch.zeros(n_init, device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected = (padded_grad >= grad_threshold) & (
            self.get_scaling.max(dim=1).values
            > self.percent_dense * scene_extent
        )

        stds = self.get_scaling[selected].repeat(N, 1)
        stds = torch.cat([stds, torch.zeros_like(stds[:, :1])], dim=-1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rots = build_rotation(self._rotation[selected]).repeat(N, 1, 1)

        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected].repeat(N, 1)
        )
        new_dc = self._features_dc[selected].repeat(N, 1, 1)
        new_rest = self._features_rest[selected].repeat(N, 1, 1)
        new_opacity = self._opacity[selected].repeat(N, 1)
        new_scaling = torch.log(
            self.get_scaling[selected].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected].repeat(N, 1)

        feat_new = {
            sn: getattr(self, f'_feat_{sn}')[selected].repeat(N, 1)
            for sn in self.feature_scales
        }
        self._densification_postfix(
            new_xyz, new_dc, new_rest, new_opacity,
            new_scaling, new_rotation, feat_new,
        )

        prune_filter = torch.cat([
            selected,
            torch.zeros(N * selected.sum(), device="cuda", dtype=bool),
        ])
        self._prune_points(prune_filter)

    def _densification_postfix(self, new_xyz, new_dc, new_rest,
                               new_opacity, new_scaling, new_rotation,
                               feat_new=None):
        d = {
            "xyz": new_xyz, "f_dc": new_dc, "f_rest": new_rest,
            "opacity": new_opacity, "scaling": new_scaling,
            "rotation": new_rotation,
        }
        if feat_new:
            for sn, ft in feat_new.items():
                d[f'feat_{sn}'] = ft

        optimizable_tensors = self._cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        for sn in self.feature_scales:
            k = f'feat_{sn}'
            if k in optimizable_tensors:
                setattr(self, f'_feat_{sn}', optimizable_tensors[k])

        self.xyz_gradient_accum = torch.zeros(
            (self.num_points, 1), device="cuda"
        )
        self.denom = torch.zeros((self.num_points, 1), device="cuda")
        self.max_radii2D = torch.zeros(self.num_points, device="cuda")

    def _prune_points(self, mask):
        valid = ~mask
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(
                group["params"][0], None
            )
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid].requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][valid].requires_grad_(True)
                )
            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        for sn in self.feature_scales:
            k = f'feat_{sn}'
            if k in optimizable_tensors:
                setattr(self, f'_feat_{sn}', optimizable_tensors[k])

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]

    # ── Save ──

    def save_features(self, directory, iteration=None):
        """Save per-scale feature embeddings (compatible with MultiScaleRenderer)."""
        os.makedirs(directory, exist_ok=True)
        for sn, dim in self.feature_scales.items():
            scale_dir = os.path.join(directory, sn)
            os.makedirs(scale_dir, exist_ok=True)
            feat = getattr(self, f'_feat_{sn}').detach()
            torch.save({
                'loc_feature': feat,
                'scale': sn,
                'feature_dim': dim,
                'num_gaussians': feat.shape[0],
                'iteration': iteration,
            }, os.path.join(scale_dir, 'best_model.pth'))
            print(f"    Saved [{sn}]: {feat.shape}")

    def load_features(self, directory):
        """Load per-scale feature embeddings saved by save_features()."""
        directory = Path(directory)
        for sn, dim in self.feature_scales.items():
            ckpt_path = directory / sn / 'best_model.pth'
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Missing feature checkpoint for scale '{sn}': {ckpt_path}")
            try:
                checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
            except TypeError:
                checkpoint = torch.load(str(ckpt_path), map_location='cpu')
            feat = checkpoint['loc_feature'].float().to(device='cuda')
            expected_shape = (self.num_points, dim)
            if tuple(feat.shape) != expected_shape:
                raise ValueError(
                    f"Feature shape mismatch for scale '{sn}': got {tuple(feat.shape)}, expected {expected_shape}"
                )
            setattr(self, f'_feat_{sn}', nn.Parameter(feat.requires_grad_(True)))
            print(f"  Loaded feature [{sn}]: {tuple(feat.shape)} from {ckpt_path}")


# ════════════════════════════════════════════════════════════════════════════
# Rendering
# ════════════════════════════════════════════════════════════════════════════

def render_rgb_2dgs(gaussians, cam, bg_color, longest_edge=0):
    """Render RGB + geometry outputs from the joint model."""
    means3D = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scales_2d = gaussians.get_scaling
    rotations = gaussians.get_rotation
    colors = gaussians.get_features  # [N, K, 3] SH
    sh_deg = gaussians.active_sh_degree

    scales = torch.cat([
        scales_2d,
        torch.ones(scales_2d.shape[0], 1, device=scales_2d.device),
    ], dim=-1)

    viewmat = cam.get_world_view_transform()
    width, height = cam.width, cam.height
    if longest_edge > 0 and max(width, height) > longest_edge:
        factor = longest_edge / max(width, height)
        width, height = int(width * factor), int(height * factor)

    tanfovx = math.tan(cam.FovX * 0.5)
    tanfovy = math.tan(cam.FovY * 0.5)
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    K = torch.tensor(
        [[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]],
        device="cuda",
    )

    bg4 = (
        torch.cat([bg_color, bg_color[:1]])
        if bg_color.shape[0] == 3
        else bg_color
    )

    (render_colors, alphas, normals, surf_normals,
     distort, median_depth, info) = rasterization_2dgs(
        means=means3D, quats=rotations, scales=scales,
        opacities=opacity.squeeze(-1), colors=colors,
        viewmats=viewmat[None], Ks=K[None],
        width=width, height=height,
        packed=False, sh_degree=sh_deg,
        backgrounds=bg4[None],
        near_plane=0.01, far_plane=500,
        render_mode="RGB+ED", absgrad=True,
    )

    rendered_image = render_colors[0].permute(2, 0, 1)
    color = rendered_image[:3]
    depth = rendered_image[3:]
    radii = info["radii"].squeeze(0)
    try:
        info["gradient_2dgs"].retain_grad()
    except Exception:
        pass

    return {
        "render": color,
        "rend_alpha": alphas,
        "rend_normal": normals,
        "surf_normal": surf_normals,
        "rend_dist": distort,
        "depth": depth,
        "viewspace_points": info["gradient_2dgs"],
        "visibility_filter": radii > 0,
        "radii": radii,
        "width": width,
        "height": height,
    }


def render_rgb_2dgs_batch(gaussians, cams, bg_color, longest_edge=0):
    """Render RGB + geometry outputs for multiple cameras in one 2DGS call."""
    if not cams:
        return []
    if len(cams) == 1:
        return [render_rgb_2dgs(gaussians, cams[0], bg_color, longest_edge)]

    means3D = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scales_2d = gaussians.get_scaling
    rotations = gaussians.get_rotation
    colors = gaussians.get_features
    sh_deg = gaussians.active_sh_degree

    scales = torch.cat([
        scales_2d,
        torch.ones(scales_2d.shape[0], 1, device=scales_2d.device),
    ], dim=-1)

    width, height = cams[0].width, cams[0].height
    if longest_edge > 0 and max(width, height) > longest_edge:
        factor = longest_edge / max(width, height)
        width, height = int(width * factor), int(height * factor)

    viewmats = []
    Ks = []
    for cam in cams:
        viewmats.append(cam.get_world_view_transform())
        tanfovx = math.tan(cam.FovX * 0.5)
        tanfovy = math.tan(cam.FovY * 0.5)
        fx = width / (2 * tanfovx)
        fy = height / (2 * tanfovy)
        Ks.append(torch.tensor(
            [[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]],
            device="cuda",
        ))

    bg4 = (
        torch.cat([bg_color, bg_color[:1]])
        if bg_color.shape[0] == 3
        else bg_color
    )
    backgrounds = bg4[None].expand(len(cams), -1).contiguous()

    (render_colors, alphas, normals, surf_normals,
     distort, median_depth, info) = rasterization_2dgs(
        means=means3D, quats=rotations, scales=scales,
        opacities=opacity.squeeze(-1), colors=colors,
        viewmats=torch.stack(viewmats), Ks=torch.stack(Ks),
        width=width, height=height,
        packed=False, sh_degree=sh_deg,
        backgrounds=backgrounds,
        near_plane=0.01, far_plane=500,
        render_mode="RGB+ED", absgrad=True,
    )

    try:
        info["gradient_2dgs"].retain_grad()
    except Exception:
        pass

    radii_all = info["radii"]
    results = []
    for b in range(len(cams)):
        rendered_image = render_colors[b].permute(2, 0, 1)
        radii = radii_all[b]
        results.append({
            "render": rendered_image[:3],
            "rend_alpha": alphas[b:b + 1],
            "rend_normal": normals[b:b + 1],
            "surf_normal": surf_normals[b:b + 1],
            "rend_dist": distort[b:b + 1],
            "depth": rendered_image[3:],
            "viewspace_points": info["gradient_2dgs"],
            "visibility_filter": radii > 0,
            "radii": radii,
            "width": width,
            "height": height,
            "_batch_idx": b,
        })
    return results


def render_rgb_3dgs(gaussians, cam, bg_color, longest_edge=0):
    """Render RGB using standard 3DGS (ellipsoid) rasterization.

    Returns dict with same keys as render_rgb_2dgs where possible,
    but rend_normal/surf_normal/rend_dist are None (3DGS doesn't produce them).
    """
    means3D = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scales_2d = gaussians.get_scaling
    rotations = gaussians.get_rotation
    colors = gaussians.get_features
    sh_deg = gaussians.active_sh_degree

    # Use 3D scales: for 2DGS models, pad with learned or fixed third scale
    scales = torch.cat([
        scales_2d,
        torch.zeros(scales_2d.shape[0], 1, device=scales_2d.device),
    ], dim=-1)

    viewmat = cam.get_world_view_transform()
    width, height = cam.width, cam.height
    if longest_edge > 0 and max(width, height) > longest_edge:
        factor = longest_edge / max(width, height)
        width, height = int(width * factor), int(height * factor)

    tanfovx = math.tan(cam.FovX * 0.5)
    tanfovy = math.tan(cam.FovY * 0.5)
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    K = torch.tensor(
        [[fx, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]],
        device="cuda",
    )

    render_colors, render_alphas, info = rasterization(
        means=means3D, quats=rotations, scales=scales,
        opacities=opacity.squeeze(-1), colors=colors,
        viewmats=viewmat[None], Ks=K[None],
        width=width, height=height,
        packed=False, sh_degree=sh_deg,
        backgrounds=bg_color[None],
        near_plane=0.01, far_plane=500,
        render_mode="RGB+ED", absgrad=True,
    )

    rendered_image = render_colors[0].permute(2, 0, 1)
    color = rendered_image[:3]
    depth = rendered_image[3:]
    radii = info["radii"].squeeze(0)
    try:
        means2d = info["means2d"]
        means2d.retain_grad()
    except Exception:
        means2d = None

    return {
        "render": color,
        "rend_alpha": render_alphas,
        "rend_normal": None,
        "surf_normal": None,
        "rend_dist": None,
        "depth": depth,
        "viewspace_points": means2d,
        "visibility_filter": radii > 0,
        "radii": radii,
        "width": width,
        "height": height,
    }


def render_features_2dgs(gaussians, viewmat, feature_colors,
                         feat_h, feat_w, K, chunk_size=32):
    """
    Render feature map using 2DGS rasterization with channel chunking.

    Args:
        gaussians: GaussianModel2DGSJoint
        viewmat: [4, 4] world-to-camera matrix
        feature_colors: [N, D] L2-normalized feature embeddings
        feat_h, feat_w: rendering resolution
        K: [3, 3] intrinsic matrix at feature resolution
        chunk_size: max channels per rasterization call

    Returns:
        [D, feat_h, feat_w] rendered + L2 normalized feature map
    """
    means3D = gaussians.get_xyz
    opacity = gaussians.get_opacity
    scales_2d = gaussians.get_scaling
    rotations = gaussians.get_rotation
    scales = torch.cat([
        scales_2d,
        torch.ones(scales_2d.shape[0], 1, device=scales_2d.device),
    ], dim=-1)

    D = feature_colors.shape[1]
    n_chunks = (D + chunk_size - 1) // chunk_size

    chunks = []
    for i in range(n_chunks):
        c_start = i * chunk_size
        c_end = min((i + 1) * chunk_size, D)
        render_colors, _, _, _, _, _, _ = rasterization_2dgs(
            means=means3D, quats=rotations, scales=scales,
            opacities=opacity.squeeze(-1),
            colors=feature_colors[:, c_start:c_end],
            viewmats=viewmat[None], Ks=K[None],
            width=feat_w, height=feat_h,
            packed=False, near_plane=0.01, far_plane=500,
            render_mode='RGB',
        )
        chunks.append(render_colors)

    feature_map = torch.cat(chunks, dim=-1)[0]  # [H, W, D]
    feature_map = feature_map.permute(2, 0, 1)  # [D, H, W]
    return F.normalize(feature_map, p=2, dim=0)


# ════════════════════════════════════════════════════════════════════════════
# Scene Loading (no global args dependency)
# ════════════════════════════════════════════════════════════════════════════

def load_scene_joint(source_dir, images_subdir=''):
    """Load COLMAP scene data without global args dependency."""
    sparse_dir = os.path.join(source_dir, "sparse", "0")
    cam_intrinsics = read_cameras_binary(
        os.path.join(sparse_dir, "cameras.bin")
    )
    cam_extrinsics = read_images_binary(
        os.path.join(sparse_dir, "images.bin")
    )
    images_dir = os.path.join(
        source_dir, images_subdir if images_subdir else "images"
    )

    all_cams = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        R = qvec2rotmat(extr.qvec).T
        T = extr.tvec
        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            FovX = focal2fov(intr.params[0], intr.width)
            FovY = focal2fov(intr.params[0], intr.height)
        elif intr.model in ("PINHOLE", "OPENCV"):
            FovX = focal2fov(intr.params[0], intr.width)
            FovY = focal2fov(intr.params[1], intr.height)
        else:
            raise ValueError(f"Unsupported camera: {intr.model}")

        img_path = os.path.join(images_dir, extr.name)
        if not os.path.exists(img_path):
            alt = os.path.join(source_dir, extr.name)
            if os.path.exists(alt):
                img_path = alt

        cam = CameraData(
            uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
            image=img_path, image_name=extr.name,
            width=intr.width, height=intr.height,
        )
        all_cams.append(cam)

    # Point cloud
    ply_path = os.path.join(sparse_dir, "points3D.ply")
    bin_path = os.path.join(sparse_dir, "points3D.bin")
    if os.path.exists(ply_path):
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        pcd_xyz = np.stack(
            [v['x'], v['y'], v['z']], axis=1
        ).astype(np.float32)
        pcd_rgb = np.stack(
            [v['red'], v['green'], v['blue']], axis=1
        ).astype(np.float32) / 255.0
    elif os.path.exists(bin_path):
        pcd_xyz, pcd_rgb = read_points3d_binary(bin_path)
    else:
        raise FileNotFoundError(f"No point cloud in {sparse_dir}")

    # Camera extent
    centers = []
    for c in all_cams:
        W2C = np.eye(4)
        W2C[:3, :3] = c.R.T
        W2C[:3, 3] = c.T
        centers.append(np.linalg.inv(W2C)[:3, 3])
    centers = np.array(centers)
    cameras_extent = (
        np.max(np.linalg.norm(centers - centers.mean(0), axis=1)) * 1.1
    )

    print(f"  Cameras: {len(all_cams)}, Points: {pcd_xyz.shape[0]:,}, "
          f"Extent: {cameras_extent:.2f}")
    return all_cams, pcd_xyz, pcd_rgb, cameras_extent


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

def train(args):
    print(f"\n{'='*60}")
    print(f"  Joint 2DGS Geometry + DA3 Feature Training")
    print(f"{'='*60}")
    print(f"  Source:      {args.source_dir}")
    print(f"  Features:    {args.feature_dir}")
    print(f"  Scales:      {args.feature_scales}")
    print(f"  Model:       {args.model_dir}")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Feat weight: {args.feature_weight}")
    print(f"  Feat LR:     {args.feature_embedding_lr}")
    le_str = (f"longest_edge={args.longest_edge}"
              if args.longest_edge > 0 else "FULL")
    print(f"  Resolution:  {le_str}")
    print(f"  Losses:      λ_dist={args.lambda_dist}, "
          f"λ_normal={args.lambda_normal}, "
          f"λ_dssim={args.lambda_dssim}")
    feat_detach = getattr(args, 'detach_feat_geometry', False)
    if feat_detach:
        print(f"  Mode:        DETACH (feature loss does NOT update geometry)")
    else:
        print(f"  Mode:        JOINT (feature loss updates geometry)")
    print(f"{'='*60}\n")

    # ── 1. Load scene ──
    print("Loading scene...")
    all_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene_joint(
        args.source_dir, args.images
    )
    train_cams = all_cams

    # ── 2. Load DA3 features ──
    print("\nLoading DA3 features...")
    scales = [s.strip() for s in args.feature_scales.split(',')]
    feat_cache = DA3FeatureCache(args.feature_dir, scales)

    # Map camera uid → DA3 frame_id using image name matching.
    # COLMAP and DA3 have DIFFERENT orderings (COLMAP sorts by image ID,
    # DA3 sorts alphabetically by seq/frame). Match by image filename.
    images_dir = os.path.join(
        args.source_dir,
        args.images if args.images else "images",
    )
    da3_name_to_fid = build_da3_image_order(images_dir)
    available_fids = feat_cache.frame_ids(scales[0])

    cam_to_fid = {}
    for cam in train_cams:
        fid = da3_name_to_fid.get(cam.image_name)
        if fid is not None and fid in available_fids:
            cam_to_fid[cam.uid] = fid
    print(f"  Matched {len(cam_to_fid)}/{len(train_cams)} cameras to features")

    if len(cam_to_fid) == 0:
        print("  ERROR: No cameras matched to features!")
        print(f"  Camera UIDs: {[c.uid for c in train_cams[:5]]}...")
        print(f"  Feature IDs: {sorted(list(available_fids))[:5]}...")
        return

    # ── 3. Feature scale info ──
    scale_infos = {}
    cam0 = train_cams[0]
    tanfovx = math.tan(cam0.FovX * 0.5)
    tanfovy = math.tan(cam0.FovY * 0.5)
    img_fx = cam0.width / (2 * tanfovx)
    img_fy = cam0.height / (2 * tanfovy)

    for scale in scales:
        dim, h, w = feat_cache.info(scale)
        sx = w / cam0.width
        sy = h / cam0.height
        feat_K = torch.tensor([
            [img_fx * sx, 0, cam0.width * sx / 2.0],
            [0, img_fy * sy, cam0.height * sy / 2.0],
            [0, 0, 1],
        ], device="cuda", dtype=torch.float32)
        scale_infos[scale] = {'dim': dim, 'h': h, 'w': w, 'K': feat_K}
        print(f"  [{scale}] {dim}d @ {w}×{h}, "
              f"fx={img_fx*sx:.2f} fy={img_fy*sy:.2f}")

    # ── 4. Create model ──
    print("\nInitializing model...")
    feature_dims = {s: scale_infos[s]['dim'] for s in scales}
    gaussians = GaussianModel2DGSJoint(
        sh_degree=args.sh_degree, feature_scales=feature_dims
    )
    gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)
    gaussians.training_setup(args)

    bg_color = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    # ── 5. Pre-cache images ──
    print("Pre-caching images...")
    from PIL import Image as PILImage
    cache_count = 0
    for cam in train_cams:
        if isinstance(cam.image, str) and os.path.exists(cam.image):
            cam._cached_np = np.array(
                PILImage.open(cam.image).convert("RGB"), dtype=np.uint8
            )
            cache_count += 1
    print(f"  Cached {cache_count}/{len(train_cams)} images")

    # ── 6. Output setup ──
    os.makedirs(args.model_dir, exist_ok=True)

    # Feature warmup: delay feature training to let geometry stabilize
    feat_start_iter = getattr(args, 'feature_start_iter', 0)

    # ── 7. Training loop ──
    viewpoint_stack = None
    ema_loss = 0.0
    ema_rgb_loss = 0.0
    ema_feat_loss = {s: 0.0 for s in scales}
    best_loss = float('inf')

    pbar = tqdm(range(1, args.iterations + 1), desc="Joint Training")
    for iteration in pbar:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Sample camera
        if not viewpoint_stack:
            viewpoint_stack = list(train_cams)
        idx = randint(0, len(viewpoint_stack) - 1)
        cam = viewpoint_stack.pop(idx)
        fid = cam_to_fid.get(cam.uid)

        # ── RGB render ──
        render_pkg = render_rgb_2dgs(
            gaussians, cam, bg_color, longest_edge=args.longest_edge
        )
        image = render_pkg["render"]
        rw, rh = render_pkg["width"], render_pkg["height"]

        gt_image = load_image_tensor(cam)
        gt_image = F.interpolate(
            gt_image.unsqueeze(0), size=(rh, rw),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        # ── RGB loss ──
        Ll1 = F.l1_loss(image, gt_image)
        ssim_val = ssim(image, gt_image)
        rgb_loss = (
            (1.0 - args.lambda_dssim) * Ll1
            + args.lambda_dssim * (1.0 - ssim_val)
        )
        loss = rgb_loss

        # ── 2DGS regularization ──
        normal_start = getattr(args, 'normal_start_iter', 500)
        dist_start = getattr(args, 'dist_start_iter', 500)
        lambda_normal = args.lambda_normal if iteration > normal_start else 0.0
        lambda_dist = args.lambda_dist if iteration > dist_start else 0.0

        if lambda_normal > 0 or lambda_dist > 0:
            rend_dist = render_pkg["rend_dist"]
            rend_normal = render_pkg["rend_normal"]
            surf_normal = render_pkg["surf_normal"]
            rend_alpha = render_pkg["rend_alpha"]

            surf_normal_proc = surf_normal * rend_alpha.squeeze(0).detach()
            rend_normal_proc = rend_normal.squeeze(0).permute(2, 0, 1)
            if len(surf_normal_proc.shape) == 4:
                surf_normal_proc = surf_normal_proc.squeeze(0)
            surf_normal_proc = surf_normal_proc.permute(2, 0, 1)

            normal_error = (
                1 - (rend_normal_proc * surf_normal_proc).sum(dim=0)
            )[None]

            if lambda_normal > 0:
                loss = loss + lambda_normal * normal_error.mean()
            if lambda_dist > 0:
                rend_dist_val = rend_dist.squeeze(-1)
                loss = loss + lambda_dist * rend_dist_val.mean()

        # ── Scale regularization ──
        lambda_scale = getattr(args, 'lambda_scale', 0.0)
        if lambda_scale > 0 and iteration > dist_start:
            threshold = getattr(args, 'scale_reg_threshold', 0.3)
            log_threshold = math.log(max(threshold, 1e-6))
            max_log_scales_grad = gaussians._scaling.max(dim=1).values
            excess_log = torch.clamp(max_log_scales_grad - log_threshold, min=0)
            loss = loss + lambda_scale * (excess_log ** 2).mean()

        # ── Feature losses ──
        total_feat_loss = torch.tensor(0.0, device="cuda")
        viewmat = cam.get_world_view_transform()

        if fid is not None and iteration >= feat_start_iter:
            for scale in scales:
                gt_feat = feat_cache.get(scale, fid)
                if gt_feat is None:
                    continue

                si = scale_infos[scale]
                feat_colors = gaussians.get_feature(scale)  # [N, D]

                # Optionally detach geometry from feature loss:
                # feature loss only trains embeddings, not geometry
                if feat_detach:
                    # Detach geometry inputs so gradients only flow to
                    # feat_colors (feature embeddings), not xyz/scale/etc.
                    det_means = gaussians.get_xyz.detach()
                    det_opacity = gaussians.get_opacity.detach()
                    det_scales = gaussians.get_scaling.detach()
                    det_rots = gaussians.get_rotation.detach()
                    det_scales3 = torch.cat([
                        det_scales,
                        torch.ones(det_scales.shape[0], 1,
                                   device=det_scales.device),
                    ], dim=-1)

                    D = feat_colors.shape[1]
                    chunks = []
                    for ci in range((D + 31) // 32):
                        cs, ce = ci * 32, min((ci + 1) * 32, D)
                        rc, _, _, _, _, _, _ = rasterization_2dgs(
                            means=det_means, quats=det_rots,
                            scales=det_scales3,
                            opacities=det_opacity.squeeze(-1),
                            colors=feat_colors[:, cs:ce],
                            viewmats=viewmat[None], Ks=si['K'][None],
                            width=si['w'], height=si['h'],
                            packed=False, near_plane=0.01,
                            far_plane=500, render_mode='RGB',
                        )
                        chunks.append(rc)
                    fm = torch.cat(chunks, dim=-1)[0].permute(2, 0, 1)
                    rendered_feat = F.normalize(fm, p=2, dim=0)
                else:
                    rendered_feat = render_features_2dgs(
                        gaussians, viewmat, feat_colors,
                        si['h'], si['w'], si['K'],
                    )

                feat_l1 = F.l1_loss(rendered_feat, gt_feat)
                feat_cos = (
                    1.0 - F.cosine_similarity(
                        rendered_feat, gt_feat, dim=0
                    ).mean()
                )
                scale_loss = feat_l1 + 0.5 * feat_cos
                total_feat_loss = total_feat_loss + scale_loss

                ema_feat_loss[scale] = (
                    0.9 * ema_feat_loss[scale] + 0.1 * scale_loss.item()
                )

            loss = loss + args.feature_weight * total_feat_loss

        # ── Backward ──
        loss_val = loss.item()
        if (torch.isnan(loss) or torch.isinf(loss) or loss_val < -0.01):
            tqdm.write(
                f"  [Iter {iteration}] NaN/Inf loss={loss_val:.4g}, skipping"
            )
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue

        if loss_val > 10.0:
            loss = loss.clamp(max=10.0)

        loss.backward()

        # Gradient clipping (geometry + features)
        clip_params = [
            gaussians._xyz, gaussians._features_dc,
            gaussians._features_rest, gaussians._scaling,
            gaussians._rotation, gaussians._opacity,
        ]
        for sn in scales:
            clip_params.append(getattr(gaussians, f'_feat_{sn}'))
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)

        # ── Densification stats (RGB render only) ──
        with torch.no_grad():
            if iteration < args.densify_until_iter:
                vp = render_pkg["viewspace_points"]
                grad_data = vp.grad if vp.grad is not None else vp
                radii = render_pkg["radii"]
                vis = render_pkg["visibility_filter"]
                gaussians.max_radii2D[vis] = torch.max(
                    gaussians.max_radii2D[vis], radii[vis]
                )
                gaussians.add_densification_stats(grad_data, vis, rw, rh)

        with torch.no_grad():
            # EMA tracking
            ema_loss = 0.4 * loss_val + 0.6 * ema_loss
            ema_rgb_loss = 0.4 * rgb_loss.item() + 0.6 * ema_rgb_loss

            if iteration % 10 == 0:
                feat_str = " ".join(
                    f"{s[0]}={ema_feat_loss[s]:.3f}" for s in scales
                )
                pbar.set_postfix({
                    "L": f"{ema_loss:.4f}",
                    "RGB": f"{ema_rgb_loss:.4f}",
                    "F": feat_str,
                    "N": f"{gaussians.num_points:,}",
                })

            # ── Checkpoints ──
            if (iteration % args.save_interval == 0
                    or iteration == args.iterations):
                save_dir = os.path.join(
                    args.model_dir, "point_cloud",
                    f"iteration_{iteration}",
                )
                print(f"\n  [Iter {iteration}] Saving... "
                      f"({gaussians.num_points:,} Gaussians)")
                gaussians.save_ply(
                    os.path.join(save_dir, "point_cloud.ply")
                )
                feat_dir = os.path.join(args.model_dir, "features")
                gaussians.save_features(feat_dir, iteration=iteration)

                if ema_loss < best_loss:
                    best_loss = ema_loss
                    best_dir = os.path.join(
                        args.model_dir, "point_cloud", "best"
                    )
                    gaussians.save_ply(
                        os.path.join(best_dir, "point_cloud.ply")
                    )
                    best_feat_dir = os.path.join(
                        args.model_dir, "features_best"
                    )
                    gaussians.save_features(best_feat_dir)
                    tqdm.write(f"  ★ New best loss: {best_loss:.5f}")

            # ── Densification ──
            if iteration < args.densify_until_iter:
                if (iteration > args.densify_from_iter
                        and iteration % args.densification_interval == 0):
                    size_threshold = (
                        20 if iteration > args.opacity_reset_interval
                        else None
                    )
                    gaussians.densify_and_prune(
                        args.densify_grad_threshold, 0.005,
                        cameras_extent, size_threshold,
                    )

                if iteration % args.opacity_reset_interval == 0:
                    reset_val = getattr(args, 'opacity_reset_value', 0.01)
                    gaussians.reset_opacity(reset_value=reset_val)

            # ── Optimizer step ──
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # ── Logging ──
        if iteration % 500 == 1:
            feat_detail = " | ".join(
                f"{s}={ema_feat_loss[s]:.4f}" for s in scales
            )
            tqdm.write(
                f"  [Iter {iteration}] "
                f"RGB={ema_rgb_loss:.4f} "
                f"Feat=[{feat_detail}] "
                f"Total={ema_loss:.4f} "
                f"N={gaussians.num_points:,}"
            )

    # ── Final save ──
    print(f"\n  Training complete.")
    save_dir = os.path.join(
        args.model_dir, "point_cloud",
        f"iteration_{args.iterations}",
    )
    gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
    gaussians.save_features(os.path.join(args.model_dir, "features"))
    print(f"  Output: {args.model_dir}")


# ════════════════════════════════════════════════════════════════════════════
# Arguments
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Joint 2DGS Geometry + DA3 Feature Training'
    )

    # Scene
    p.add_argument('--source_dir', type=str, required=True,
                   help='COLMAP scene directory')
    p.add_argument('--images', type=str, default='',
                   help='Image subdirectory (e.g. "processed")')
    p.add_argument('--model_dir', type=str, required=True,
                   help='Output directory')

    # Feature data
    p.add_argument('--feature_dir', type=str, required=True,
                   help='DA3 feature directory (with coarse/mid/fine subdirs)')
    p.add_argument('--feature_scales', type=str, default='coarse,mid,fine',
                   help='Comma-separated scale names to train')

    # Training
    p.add_argument('--iterations', type=int, default=30000)
    p.add_argument('--sh_degree', type=int, default=3)
    p.add_argument('--white_background', action='store_true')
    p.add_argument('--longest_edge', type=int, default=0,
                   help='Downsample longest edge for RGB render (0=full)')

    # Feature training
    p.add_argument('--feature_weight', type=float, default=0.1,
                   help='Weight for feature loss (relative to RGB)')
    p.add_argument('--feature_embedding_lr', type=float, default=0.01,
                   help='Learning rate for feature embeddings')
    p.add_argument('--feature_start_iter', type=int, default=0,
                   help='Iteration to start feature training (0=from start)')
    p.add_argument('--detach_feat_geometry', action='store_true',
                   help='Detach: feature loss only trains embeddings, '
                        'not geometry')

    # Geometry learning rates (match train_2dgs_geometry defaults)
    p.add_argument('--position_lr_init', type=float, default=0.00016)
    p.add_argument('--position_lr_final', type=float, default=0.0000016)
    p.add_argument('--feature_lr', type=float, default=0.0025,
                   help='SH feature LR (not DA3 features)')
    p.add_argument('--opacity_lr', type=float, default=0.05)
    p.add_argument('--scaling_lr', type=float, default=0.005)
    p.add_argument('--rotation_lr', type=float, default=0.001)
    p.add_argument('--percent_dense', type=float, default=0.01)

    # 2DGS losses
    p.add_argument('--lambda_dssim', type=float, default=0.2)
    p.add_argument('--lambda_dist', type=float, default=0.01)
    p.add_argument('--lambda_normal', type=float, default=0.05)
    p.add_argument('--lambda_scale', type=float, default=0.0,
                   help='Scale regularization weight')
    p.add_argument('--scale_reg_threshold', type=float, default=0.3)

    # Reg start iterations
    p.add_argument('--normal_start_iter', type=int, default=500)
    p.add_argument('--dist_start_iter', type=int, default=500)

    # Densification
    p.add_argument('--densify_from_iter', type=int, default=500)
    p.add_argument('--densify_until_iter', type=int, default=20000)
    p.add_argument('--densification_interval', type=int, default=100)
    p.add_argument('--densify_grad_threshold', type=float, default=0.0002)
    p.add_argument('--opacity_reset_interval', type=int, default=3000)
    p.add_argument('--opacity_reset_value', type=float, default=0.01)

    # Save
    p.add_argument('--save_interval', type=int, default=5000)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
