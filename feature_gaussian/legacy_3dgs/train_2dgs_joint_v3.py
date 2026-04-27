#!/usr/bin/env python3
"""
Joint 2DGS Geometry + Feature Training v3
==========================================
基于 v2 的深度改进版，整合消融实验结论和专家建议:

新增特性 (vs v2):
  1. Sky masking: 使用 masks.pkl 排除天空区域的 RGB/geometry/feature loss
  2. AppearanceNetwork: 逐图像仿射颜色校正 (scale+bias)，吸收曝光变化
  3. Transient mask: 可学习的逐像素权重，降低动态物体/曝光异常的影响
  4. 语义引导正则化: 植被区域松绑 normal/dist/scale 正则化
  5. 保留有效改进: depth supervision (λ=0.05), feature_weight=0.5
  6. 回退 densification 至 v1 参数 (消融证明 aggressive 有害)
  7. 移除 random_background (消融证明有害)

用法:
  CUDA_VISIBLE_DEVICES=0 python -m feature_3dgs.train_2dgs_joint_v3 \
    --config configs/joint_oh_v3.yaml
"""

import argparse
import math
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from random import randint, shuffle

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from gsplat import rasterization_2dgs, rasterization

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
    pearson_depth_loss,
    load_mono_depth,
    AppearanceNetwork,
)

from feature_gaussian.legacy_3dgs.train_2dgs_joint import (
    DA3FeatureCache,
    GaussianModel2DGSJoint,
    render_rgb_2dgs,
    render_rgb_2dgs_batch,
    render_rgb_3dgs,
    render_features_2dgs,
    build_da3_image_order,
)

# Reuse v2 utilities
from feature_gaussian.legacy_3dgs.train_2dgs_joint_v2 import (
    load_scene_colmap,
    _compute_extent,
    pca_colorize,
    evaluate_psnr,
)


# ════════════════════════════════════════════════════════════════════════════
# Config (extends v2 defaults)
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    'exp_name': 'joint_v3',

    'dataset': {
        'type': 'colmap',
        'source_dir': '',
        'images': '',
        'feature_dir': '',
        'feature_scales': ['fine'],
        'traj_path': '',
    },

    'model': {
        'sh_degree': 3,
        'white_background': False,
        'random_background': False,   # v3: DISABLED (ablation: -0.18 dB)
    },

    'training': {
        'iterations': 30000,
        'longest_edge': 0,
        'eval_interval': 5000,
        'save_interval': 5000,
        'vis_frames': [0, 200, 500, 800],
        'init_ply': '',
        'freeze_geometry': False,

        # Learning rates
        'position_lr_init': 0.00016,
        'position_lr_final': 0.0000016,
        'feature_lr': 0.0025,
        'opacity_lr': 0.05,
        'scaling_lr': 0.005,
        'rotation_lr': 0.001,
        'percent_dense': 0.01,

        # Feature embedding
        'feature_embedding_lr': 0.01,
        'feature_weight': 0.5,      # v3: 0.5 (ablation: better feat loss)
        'feature_cos_weight': 0.5,
        'feature_start_iter': 500,
        'detach_feat_geometry': False,

        # RGB losses
        'lambda_dssim': 0.2,

        # 2DGS geometry regularization
        'lambda_normal': 0.05,
        'lambda_dist': 0.01,
        'lambda_scale': 0.001,
        'scale_reg_threshold': 0.3,
        'reg_start_iter': 500,

        # Monocular depth supervision (v3: ENABLED, ablation: +0.05 dB)
        'mono_depth_dir': '',
        'lambda_depth': 0.05,
        'depth_start_iter': 3000,
        'depth_warmup_iters': 2000,

        # Densification (v3: v1 params, ablation: aggressive is harmful)
        'densify_from_iter': 500,
        'densify_until_iter': 20000,
        'densification_interval': 100,
        'densify_grad_threshold': 0.0002,
        'opacity_reset_interval': 3000,
        'opacity_reset_value': 0.01,

        # ── v3 新增 ──

        # Sky/object masking
        'use_mask': False,
        'mask_path': '',         # path to masks.pkl

        # Appearance network
        'use_appearance': False,
        'appearance_embed_dim': 32,
        'appearance_lr': 0.001,
        'appearance_reg': 0.0,       # individual identity reg
        'appearance_mean_reg': 0.01, # mean-centering reg

        # Transient mask (learnable per-pixel confidence)
        'use_transient': False,
        'transient_lr': 0.001,
        'transient_reg': 0.01,       # regularize toward 0.5

        # Semantic-guided regularization
        'semantic_mask_dir': '',      # path to semantic_masks/ from generate_semantic_masks.py
        'vegetation_reg_scale': 0.1,  # multiply normal/dist reg in vegetation areas
        'vegetation_scale_boost': 1.0, # boost scale reg in vegetation areas (>1 = stricter)
        'veg_prune_opacity': 0.0,     # >0 enables aggressive vegetation pruning
        'veg_prune_scale_percentile': 0, # >0 prunes veg Gaussians with scale > this percentile (among veg only)
        'veg_max_scale': 0.0,         # >0 clamps veg Gaussians' max scale to this * scene_extent
        'veg_sample_cams': 50,        # cameras to sample for vegetation identification
        'veg_vote_threshold': 0.3,    # fraction of votes to classify as vegetation
        'max_gaussians': 0,           # >0 caps total Gaussian count (0 = unlimited)
        'depth_exclude_vegetation': False,  # exclude vegetation from depth supervision

        # 3DGS comparison mode
        'use_3dgs': False,            # Use 3DGS (ellipsoid) instead of 2DGS (surfel)

        # Multi-camera batching. batch_size uses one batched rasterization call
        # for 2DGS; grad_accum_steps is kept only as a legacy fallback alias.
        'batch_size': 1,
        'grad_accum_steps': 1,
        'cache_resized_images': False,
        'cache_resized_masks': False,
        'image_cache_dtype': 'float16',
        'max_consecutive_bad_steps': 50,
    },

    'output_dir': 'output/2dgs_joint',
}


def _freeze_base_gaussians(gaussians):
    frozen_groups = {'xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation'}
    for group in gaussians.optimizer.param_groups:
        if group['name'] in frozen_groups:
            group['lr'] = 0.0
            for param in group['params']:
                param.requires_grad_(False)
    gaussians.xyz_scheduler_args['lr_init'] = 0.0
    gaussians.xyz_scheduler_args['lr_final'] = 0.0
    for name in frozen_groups:
        if name in gaussians._initial_lrs:
            gaussians._initial_lrs[name] = 0.0


def load_config(config_path):
    """Load YAML config with defaults."""
    with open(config_path) as f:
        user_cfg = yaml.safe_load(f)
    cfg = _deep_merge(DEFAULT_CONFIG.copy(), user_cfg)
    return cfg


def _deep_merge(base, override):
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _init_distributed():
    """Initialize torchrun/NCCL if this process is part of a multi-GPU job."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )
    return is_distributed, rank, local_rank, world_size


def _cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def _all_reduce_optimizer_grads(optimizers, world_size):
    """Average gradients across ranks for non-DDP Parameter containers."""
    if world_size <= 1:
        return
    seen = set()
    for optimizer in optimizers:
        if optimizer is None:
            continue
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if param.grad is None or id(param) in seen:
                    continue
                seen.add(id(param))
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)


def _sync_densification_stats(gaussians, before_accum, before_denom, world_size):
    """Synchronize per-rank densification evidence while preserving history once."""
    if world_size <= 1 or before_accum is None or before_denom is None:
        return
    delta_accum = gaussians.xyz_gradient_accum - before_accum
    delta_denom = gaussians.denom - before_denom
    dist.all_reduce(delta_accum, op=dist.ReduceOp.SUM)
    dist.all_reduce(delta_denom, op=dist.ReduceOp.SUM)
    gaussians.xyz_gradient_accum.copy_(before_accum + delta_accum)
    gaussians.denom.copy_(before_denom + delta_denom)
    dist.all_reduce(gaussians.max_radii2D, op=dist.ReduceOp.MAX)


def _target_render_size(cam, longest_edge):
    width, height = cam.width, cam.height
    if longest_edge > 0 and max(width, height) > longest_edge:
        factor = longest_edge / max(width, height)
        width, height = int(width * factor), int(height * factor)
    return width, height


def _cache_resized_images(train_cams, longest_edge, dtype, show_progress):
    image_cache = {}
    with torch.no_grad():
        for cam in tqdm(train_cams, desc="Image cache", disable=not show_progress):
            width, height = _target_render_size(cam, longest_edge)
            gt = load_image_tensor(cam)
            gt = F.interpolate(
                gt.unsqueeze(0), size=(height, width),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            image_cache[cam.uid] = gt.to(dtype=dtype).contiguous()
    return image_cache


def _cache_resized_masks(train_cams, masks, longest_edge, show_progress):
    mask_cache = {}
    with torch.no_grad():
        for cam in tqdm(train_cams, desc="Mask cache", disable=not show_progress):
            if cam.image_name not in masks:
                continue
            width, height = _target_render_size(cam, longest_edge)
            obj_mask = torch.as_tensor(masks[cam.image_name][0], device="cuda")[None].bool()
            sky_mask = torch.as_tensor(masks[cam.image_name][1], device="cuda")[None].bool()
            distort_mask = torch.as_tensor(masks[cam.image_name][2], device="cuda")[None].bool()
            if obj_mask.shape[1] != height or obj_mask.shape[2] != width:
                obj_mask = F.interpolate(
                    obj_mask[None].float(), size=(height, width), mode="nearest"
                ).squeeze(0) > 0.5
                sky_mask = F.interpolate(
                    sky_mask[None].float(), size=(height, width), mode="nearest"
                ).squeeze(0) > 0.5
                distort_mask = F.interpolate(
                    distort_mask[None].float(), size=(height, width), mode="nearest"
                ).squeeze(0) > 0.5
            mask_cache[cam.uid] = (
                obj_mask.contiguous(),
                sky_mask.contiguous(),
                distort_mask.contiguous(),
            )
    return mask_cache


# ════════════════════════════════════════════════════════════════════════════
# Transient Head (learnable per-pixel confidence)
# ════════════════════════════════════════════════════════════════════════════

class TransientHead(nn.Module):
    """Tiny CNN that predicts per-pixel confidence from rendered features.

    Produces a soft mask [1, H, W] where ~1 means "static/reliable" and
    ~0 means "transient/ignore". Used to down-weight RGB loss on dynamic
    objects, exposure flares, or other per-view artifacts.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        # Init: output ~0.5 = neutral weight
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, rgb):
        """
        Args:
            rgb: [3, H, W] rendered image
        Returns:
            confidence: [1, H, W] in (0, 1)
        """
        return self.net(rgb.unsqueeze(0)).squeeze(0)


# ════════════════════════════════════════════════════════════════════════════
# Semantic mask loading
# ════════════════════════════════════════════════════════════════════════════

def load_semantic_masks(sem_dir, train_cams, device='cuda', cam_to_fid=None):
    """Load precomputed semantic masks for all training cameras.

    Supports two formats:
      1. DA3 unified: rgb_{fid}_mask.pt files (6-class, indexed by frame ID)
      2. Legacy: {image_name}_sem.pt files (5-class)

    Returns:
        dict: cam.image_name → uint8 tensor [H_p, W_p] with values 0-5
              0=other 1=building 2=vegetation 3=sky 4=ground 5=dynamic
    """
    sem_masks = {}
    sem_dir = Path(sem_dir)
    if not sem_dir.is_dir():
        print(f"  WARNING: semantic_mask_dir not found: {sem_dir}")
        return sem_masks

    # Try DA3 unified format first: rgb_{idx}_mask.pt
    da3_test = sem_dir / 'rgb_0_mask.pt'
    if da3_test.exists() and cam_to_fid is not None:
        for cam in train_cams:
            fid = cam_to_fid.get(cam.uid)
            if fid is not None:
                fpath = sem_dir / f'rgb_{fid}_mask.pt'
                if fpath.exists():
                    sem_masks[cam.image_name] = torch.load(
                        str(fpath), map_location=device, weights_only=True)
        print(f"  Loaded {len(sem_masks)}/{len(train_cams)} semantic masks (DA3 unified 6-class)")
        return sem_masks

    # Fallback: legacy _sem.pt format
    for cam in train_cams:
        save_name = cam.image_name.replace('/', '_').replace('\\', '_')
        save_name = os.path.splitext(save_name)[0] + '_sem.pt'
        fpath = sem_dir / save_name
        if fpath.exists():
            sem_masks[cam.image_name] = torch.load(str(fpath),
                                                    map_location=device,
                                                    weights_only=True)
    print(f"  Loaded {len(sem_masks)}/{len(train_cams)} semantic masks (legacy format)")
    return sem_masks


# ════════════════════════════════════════════════════════════════════════════
# Visualization (v3: adds mask overlay)
# ════════════════════════════════════════════════════════════════════════════

def visualize_comparison_v3(gaussians, train_cams, cam_to_fid, feat_cache,
                            scale_infos, scales, cfg, iteration, output_dir,
                            appearance_net=None, masks=None):
    """Generate multi-channel visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vis_frames = cfg['training']['vis_frames']
    longest_edge = cfg['training']['longest_edge']
    bg_color = torch.tensor(
        [1, 1, 1] if cfg['model']['white_background'] else [0, 0, 0],
        dtype=torch.float32, device="cuda"
    )

    vis_cams = []
    for idx in vis_frames:
        if idx < len(train_cams):
            vis_cams.append((idx, train_cams[idx]))
    if not vis_cams:
        return

    n_cols = len(vis_cams)
    row_labels = ['GT Image', 'Rendered RGB', 'Alpha', 'Depth', 'Normal']
    for s in scales:
        row_labels.append(f'GT Feat ({s})')
        row_labels.append(f'Rend Feat ({s})')
        row_labels.append(f'Feat Err ({s})')
    n_rows = len(row_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for j, (cam_idx, cam) in enumerate(vis_cams):
        fid = cam_to_fid.get(cam.uid)

        with torch.no_grad():
            render_pkg = render_rgb_2dgs(gaussians, cam, bg_color, longest_edge)
            rendered_rgb = render_pkg["render"].clamp(0, 1)
            rw, rh = render_pkg["width"], render_pkg["height"]

            # Apply appearance correction for visualization
            if appearance_net is not None:
                rendered_rgb = appearance_net(rendered_rgb, cam_idx).clamp(0, 1)
            rendered_rgb = rendered_rgb.cpu()

            depth = render_pkg["depth"].cpu()
            # "rend_alpha" is the correct key returned by render_rgb_2dgs
            alpha = render_pkg.get("rend_alpha", render_pkg.get("alpha"))
            if alpha is not None:
                alpha_vis = alpha.squeeze().cpu()
            else:
                alpha_vis = None
            rend_normal = render_pkg.get("rend_normal")
            if rend_normal is not None:
                rn = rend_normal.squeeze(0).permute(2, 0, 1)
                normal_vis = (rn * 0.5 + 0.5).clamp(0, 1).cpu()
            else:
                normal_vis = torch.zeros(3, rh, rw)

        gt_image = load_image_tensor(cam)
        gt_resized = F.interpolate(
            gt_image.unsqueeze(0), size=(rh, rw),
            mode="bilinear", align_corners=False
        ).squeeze(0).cpu()

        axes[0, j].imshow(gt_resized.permute(1, 2, 0).numpy())
        axes[0, j].set_title(f'{cam.image_name}', fontsize=9)
        axes[0, j].axis('off')

        axes[1, j].imshow(rendered_rgb.permute(1, 2, 0).numpy())
        psnr_val = -10 * np.log10(((rendered_rgb - gt_resized) ** 2).mean().item() + 1e-10)
        axes[1, j].set_title(f'PSNR={psnr_val:.1f}dB', fontsize=9)
        axes[1, j].axis('off')

        if alpha_vis is not None:
            axes[2, j].imshow(alpha_vis.numpy(), cmap='gray', vmin=0, vmax=1)
        else:
            axes[2, j].imshow(np.zeros((rh, rw)), cmap='gray')
        axes[2, j].axis('off')

        d = depth.squeeze().numpy()
        valid = d > 0
        if valid.any():
            vmin, vmax = np.percentile(d[valid], [2, 98])
        else:
            vmin, vmax = 0, 1
        axes[3, j].imshow(d, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[3, j].axis('off')

        axes[4, j].imshow(normal_vis.permute(1, 2, 0).numpy())
        axes[4, j].axis('off')

        # Feature rows
        row_offset = 5
        viewmat = cam.get_world_view_transform()
        for si, scale in enumerate(scales):
            gt_row   = row_offset + si * 3
            rend_row = row_offset + si * 3 + 1
            err_row  = row_offset + si * 3 + 2

            gt_feat_vis = None
            if fid is not None:
                gt_feat = feat_cache.get(scale, fid)
                if gt_feat is not None:
                    gt_feat_vis = gt_feat.cpu()
                    axes[gt_row, j].imshow(pca_colorize(gt_feat_vis))
            axes[gt_row, j].axis('off')

            with torch.no_grad():
                si_info = scale_infos[scale]
                feat_colors = gaussians.get_feature(scale)
                rendered_feat = render_features_2dgs(
                    gaussians, viewmat, feat_colors,
                    si_info['h'], si_info['w'], si_info['K']
                )
            rend_feat_vis = rendered_feat.cpu()
            axes[rend_row, j].imshow(pca_colorize(rend_feat_vis))
            axes[rend_row, j].axis('off')

            # Cosine error map: 1 - cos_sim per pixel (brighter = worse)
            if gt_feat_vis is not None:
                import torch.nn.functional as _F
                rf_n = _F.normalize(rend_feat_vis, p=2, dim=0)
                gf_n = _F.normalize(gt_feat_vis, p=2, dim=0)
                cos_map = _F.cosine_similarity(rf_n, gf_n, dim=0)  # [H, W]
                err_map = (1 - cos_map).clamp(0, 1).numpy()
                mean_cos = cos_map.mean().item()
                im = axes[err_row, j].imshow(err_map, cmap='hot', vmin=0, vmax=0.5)
                axes[err_row, j].set_title(f'cos={mean_cos:.3f}', fontsize=8)
            else:
                axes[err_row, j].imshow(np.zeros((si_info['h'], si_info['w'])), cmap='hot')
            axes[err_row, j].axis('off')

    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=90, labelpad=15)

    fig.suptitle(f'{cfg["exp_name"]} — Iter {iteration}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    vis_path = os.path.join(output_dir, f'vis_iter{iteration:06d}.png')
    fig.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved visualization: {vis_path}")


# ════════════════════════════════════════════════════════════════════════════
# Vegetation-aware Gaussian pruning
# ════════════════════════════════════════════════════════════════════════════

def prune_vegetation_gaussians(gaussians, train_cams, sem_masks,
                                veg_opacity_threshold=0.05,
                                veg_scale_percentile=0,
                                veg_max_scale=0.0,
                                scene_extent=1.0,
                                n_sample_cams=50,
                                veg_vote_threshold=0.3,
                                log_fn=None):
    """Prune and clamp vegetation Gaussians.

    Projects Gaussian centers to training cameras, checks semantic label, and:
      1) Prunes vegetation (class 2) Gaussians with opacity < veg_opacity_threshold
      2) Prunes vegetation Gaussians with scale > veg_scale_percentile (among veg only)
      3) Clamps remaining vegetation Gaussians' max scale to veg_max_scale * scene_extent

    Args:
        n_sample_cams: number of cameras to sample for voting (default 50)
        veg_vote_threshold: fraction of votes needed to label as vegetation (default 0.3)
        log_fn: callable for logging to train.log (optional)

    Returns (n_opacity_pruned, n_scale_pruned, n_clamped, n_veg_identified).
    """
    if not sem_masks:
        return 0, 0, 0, 0

    # Sample cameras that have semantic masks
    cams_with_masks = [c for c in train_cams if c.image_name in sem_masks]
    if not cams_with_masks:
        return 0, 0, 0, 0

    import random
    sample_cams = random.sample(cams_with_masks, min(n_sample_cams, len(cams_with_masks)))

    xyz = gaussians.get_xyz.detach()  # [N, 3]
    N = xyz.shape[0]
    veg_votes = torch.zeros(N, device='cuda')
    total_votes = torch.zeros(N, device='cuda')

    for cam in sample_cams:
        sem_label = sem_masks[cam.image_name].to('cuda')  # [H_p, W_p]
        H_s, W_s = sem_label.shape

        # World-to-camera transform (already row-major w2c from get_world_view_transform)
        w2c = cam.get_world_view_transform()  # [4, 4]
        xyz_h = torch.cat([xyz, torch.ones(N, 1, device='cuda')], dim=-1)  # [N, 4]
        xyz_cam = (w2c @ xyz_h.T).T[:, :3]  # [N, 3]

        # Filter points behind camera
        valid_depth = xyz_cam[:, 2] > 0.01
        if not valid_depth.any():
            continue

        # Project to image plane using FoV-based intrinsics
        tanfovx = math.tan(cam.FovX * 0.5)
        tanfovy = math.tan(cam.FovY * 0.5)
        fx = cam.width / (2 * tanfovx)
        fy = cam.height / (2 * tanfovy)
        cx = cam.width / 2.0
        cy = cam.height / 2.0

        # Scale intrinsics to semantic mask resolution
        sx = W_s / cam.width
        sy = H_s / cam.height

        px = (fx * xyz_cam[:, 0] / xyz_cam[:, 2] + cx) * sx
        py = (fy * xyz_cam[:, 1] / xyz_cam[:, 2] + cy) * sy

        # Check bounds
        in_bounds = valid_depth & (px >= 0) & (px < W_s) & (py >= 0) & (py < H_s)
        if not in_bounds.any():
            continue

        px_int = px[in_bounds].long()
        py_int = py[in_bounds].long()
        labels = sem_label[py_int, px_int]

        total_votes[in_bounds] += 1
        veg_votes[in_bounds] += (labels == 2).float()

    # Gaussians are "vegetation" if enough votes say so
    has_votes = total_votes > 0
    is_veg = has_votes & (veg_votes / total_votes.clamp(min=1) > veg_vote_threshold)
    n_veg = is_veg.sum().item()

    # Criterion 1: Prune vegetation Gaussians with low opacity
    opacity = gaussians.get_opacity.squeeze()
    opacity_prune = is_veg & (opacity < veg_opacity_threshold)
    n_opacity = opacity_prune.sum().item()

    # Criterion 2: Prune vegetation Gaussians with large scales (vegetation-local percentile)
    scales = gaussians.get_scaling.detach()  # [N, 2] or [N, 3]
    max_scale = scales.max(dim=-1).values     # [N]
    n_scale = 0
    if veg_scale_percentile > 0 and n_veg > 0:
        veg_scales = max_scale[is_veg]
        veg_threshold = torch.quantile(veg_scales, veg_scale_percentile / 100.0)
        scale_prune = is_veg & (max_scale > veg_threshold)
        n_scale = scale_prune.sum().item()
    else:
        scale_prune = torch.zeros(N, dtype=torch.bool, device='cuda')

    # Log vegetation statistics
    _log = log_fn if log_fn else print
    if n_veg > 0:
        veg_scales_stats = max_scale[is_veg]
        _log(f"    Veg identified: {n_veg}/{N} ({100*n_veg/N:.1f}%), "
             f"scale med={veg_scales_stats.median():.4f} "
             f"p90={torch.quantile(veg_scales_stats, 0.9):.4f} "
             f"p95={torch.quantile(veg_scales_stats, 0.95):.4f} "
             f"max={veg_scales_stats.max():.4f} "
             f"(sampled {len(sample_cams)} cams, vote>{veg_vote_threshold})")
    else:
        _log(f"    Veg identified: 0/{N} (sampled {len(sample_cams)} cams, vote>{veg_vote_threshold})")

    # Combine prune criteria and prune
    combined_prune = opacity_prune | scale_prune
    n_total = combined_prune.sum().item()
    if n_total > 0:
        gaussians._prune_points(combined_prune)
        torch.cuda.empty_cache()
        # Re-identify vegetation after pruning for clamping
        is_veg = is_veg[~combined_prune]

    # Criterion 3: Clamp remaining vegetation Gaussians' max scale
    n_clamped = 0
    if veg_max_scale > 0 and scene_extent > 0:
        abs_max = veg_max_scale * scene_extent
        log_max = math.log(max(abs_max, 1e-7))
        # Work in log-space (_scaling stores log of actual scale)
        with torch.no_grad():
            veg_idx = is_veg.nonzero(as_tuple=True)[0]
            if len(veg_idx) > 0:
                veg_log_scales = gaussians._scaling[veg_idx]  # [n_veg, D]
                exceeds = (veg_log_scales > log_max).any(dim=-1)
                n_clamped = exceeds.sum().item()
                if n_clamped > 0:
                    gaussians._scaling[veg_idx] = veg_log_scales.clamp(max=log_max)

    return n_opacity, n_scale, n_clamped, n_veg


# ════════════════════════════════════════════════════════════════════════════
# Contrastive Feature Loss
# ════════════════════════════════════════════════════════════════════════════

def contrastive_feature_loss(
    rendered_feat: torch.Tensor,
    gt_feat: torch.Tensor,
    n_anchors: int = 256,
    n_negatives: int = 128,
    temperature: float = 0.07,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss on rendered vs GT feature maps.

    Pushes same-position features together and different-position features apart,
    countering feature collapse from L1+cosine loss (which has no negative push).

    Args:
        rendered_feat: (C, H, W) L2-normalized rendered feature map
        gt_feat:       (C, H, W) L2-normalized GT feature map
        n_anchors:     number of anchor pixels to sample
        n_negatives:   number of negative pixels per anchor
        temperature:   InfoNCE temperature (lower = sharper)
        mask:          (1, H, W) optional valid pixel mask

    Returns:
        scalar InfoNCE loss
    """
    C, H, W = rendered_feat.shape
    N = H * W

    # Flatten to (N, C)
    r_flat = rendered_feat.reshape(C, N).t()  # (N, C)
    g_flat = gt_feat.reshape(C, N).t()        # (N, C)

    # Get valid pixel indices
    if mask is not None:
        valid_idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
        if valid_idx.numel() < 4:
            return rendered_feat.new_tensor(0.0)
    else:
        valid_idx = torch.arange(N, device=rendered_feat.device)

    # Sample anchors
    n_anc = min(n_anchors, valid_idx.numel())
    perm = torch.randperm(valid_idx.numel(), device=rendered_feat.device)[:n_anc]
    anchor_idx = valid_idx[perm]

    # For each anchor, sample negatives (different positions)
    n_neg = min(n_negatives, valid_idx.numel() - 1)
    if n_neg < 1:
        return rendered_feat.new_tensor(0.0)

    anchors = r_flat[anchor_idx]  # (n_anc, C)
    positives = g_flat[anchor_idx]  # (n_anc, C)

    # Sample shared negative pool from GT features (efficient)
    neg_perm = torch.randperm(valid_idx.numel(), device=rendered_feat.device)[:n_neg]
    neg_idx = valid_idx[neg_perm]
    negatives = g_flat[neg_idx]  # (n_neg, C)

    # Positive similarity: (n_anc,)
    pos_sim = (anchors * positives).sum(dim=1) / temperature

    # Negative similarity: (n_anc, n_neg)
    neg_sim = torch.mm(anchors, negatives.t()) / temperature

    # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (n_anc, 1+n_neg)
    labels = torch.zeros(n_anc, dtype=torch.long, device=rendered_feat.device)
    loss = F.cross_entropy(logits, labels)

    return loss


# ════════════════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════════════════

def train(cfg):
    is_distributed, rank, local_rank, world_size = _init_distributed()
    is_main_process = rank == 0
    exp_name = cfg['exp_name']
    output_dir = os.path.join(cfg['output_dir'], exp_name)
    os.makedirs(output_dir, exist_ok=True)

    dcfg = cfg['dataset']
    mcfg = cfg['model']
    tcfg = cfg['training']

    seed = int(tcfg.get('seed', 12345)) + rank * 100003
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)

    if is_main_process:
        print(f"\n{'='*70}")
        print(f"  Joint 2DGS Geometry + Feature Training v3")
        print(f"  Experiment: {exp_name}")
        print(f"{'='*70}")
        print(f"  Dataset type:      {dcfg['type']}")
        print(f"  Source:             {dcfg['source_dir']}")
        print(f"  Features:           {dcfg['feature_dir']}")
        print(f"  Scales:             {dcfg['feature_scales']}")
        print(f"  Iterations:         {tcfg['iterations']}")
        le = tcfg['longest_edge']
        print(f"  Resolution:         {'FULL' if le == 0 else f'longest_edge={le}'}")
        print(f"  Feature weight:     {tcfg['feature_weight']}")
        print(f"  Depth supervision:  λ={tcfg['lambda_depth']}")
        print(f"  Use mask:           {tcfg['use_mask']}")
        print(f"  Use appearance:     {tcfg['use_appearance']}")
        print(f"  Use transient:      {tcfg['use_transient']}")
        print(f"  Use 3DGS:           {tcfg.get('use_3dgs', False)}")
        print(f"  Semantic reg:       veg_scale={tcfg['vegetation_reg_scale']}")
        print(f"  Veg prune opacity:  {tcfg.get('veg_prune_opacity', 0.0)}")
        print(f"  Veg prune scale %: {tcfg.get('veg_prune_scale_percentile', 0)} (veg-local)")
        print(f"  Veg max scale:      {tcfg.get('veg_max_scale', 0.0)} * extent")
        print(f"  Batch size / GPU:   {tcfg.get('batch_size', tcfg.get('grad_accum_steps', 1))}")
        print(f"  Distributed:        {world_size} GPU(s), local_rank={local_rank}")
        print(f"  Output:             {output_dir}")
        print(f"{'='*70}\n")

    # Save config
    if is_main_process:
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    if is_distributed:
        dist.barrier()

    # ── 1. Load scene ──
    if is_main_process:
        print("Loading scene...")
    ds_type = dcfg['type']
    if ds_type in ('colmap', 'cambridge'):
        train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
            load_scene_colmap(dcfg['source_dir'], dcfg.get('images', ''))
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")

    # ── 2. Load features only when explicit feature supervision is active ──
    configured_scales = dcfg['feature_scales']
    if isinstance(configured_scales, str):
        configured_scales = [s.strip() for s in configured_scales.split(',') if s.strip()]
    feature_training_enabled = (
        float(tcfg.get('feature_weight', 0.0)) > 0.0
        and int(tcfg.get('feature_start_iter', 0)) <= int(tcfg['iterations'])
    )

    feat_cache = None
    cam_to_fid = {}
    scale_infos = {}
    scales = []

    if feature_training_enabled:
        scales = configured_scales
        if is_main_process:
            print(f"\nLoading features ({scales})...")
        feat_cache = DA3FeatureCache(dcfg['feature_dir'], scales)

        # Map cameras to feature frame IDs
        images_subdir = dcfg.get('images', '')
        if images_subdir:
            images_dir = os.path.join(dcfg['source_dir'], images_subdir)
        else:
            import glob as _glob
            if _glob.glob(os.path.join(dcfg['source_dir'], 'seq*')):
                images_dir = dcfg['source_dir']
            elif os.path.isdir(os.path.join(dcfg['source_dir'], 'images')):
                images_dir = os.path.join(dcfg['source_dir'], 'images')
            else:
                images_dir = dcfg['source_dir']
        da3_name_to_fid = build_da3_image_order(images_dir)
        available_fids = feat_cache.frame_ids(scales[0])

        for cam in train_cams:
            fid = da3_name_to_fid.get(cam.image_name)
            if fid is not None and fid in available_fids:
                cam_to_fid[cam.uid] = fid
        if is_main_process:
            print(f"  Matched {len(cam_to_fid)}/{len(train_cams)} cameras to features")
        if len(cam_to_fid) == 0:
            print("ERROR: No cameras matched to features!")
            return
    else:
        if is_main_process:
            print(
            "\nFeature training disabled "
            "(feature_weight=0 or feature_start_iter > iterations); "
            "skipping teacher feature cache and explicit feature embeddings"
            )

    # ── 3. Feature scale info ──
    cam0 = train_cams[0]
    tanfovx = math.tan(cam0.FovX * 0.5)
    tanfovy = math.tan(cam0.FovY * 0.5)
    img_fx = cam0.width / (2 * tanfovx)
    img_fy = cam0.height / (2 * tanfovy)

    for scale in scales:
        dim, h, w = feat_cache.info(scale)
        sx, sy = w / cam0.width, h / cam0.height
        feat_K = torch.tensor([
            [img_fx * sx, 0, cam0.width * sx / 2.0],
            [0, img_fy * sy, cam0.height * sy / 2.0],
            [0, 0, 1],
        ], device="cuda", dtype=torch.float32)
        scale_infos[scale] = {'dim': dim, 'h': h, 'w': w, 'K': feat_K}
        if is_main_process:
            print(f"  [{scale}] {dim}d @ {w}×{h}")

    # ── 4. Create model ──
    if is_main_process:
        print("\nInitializing model...")
    feature_dims = {s: scale_infos[s]['dim'] for s in scales}
    gaussians = GaussianModel2DGSJoint(
        sh_degree=mcfg['sh_degree'],
        feature_scales=feature_dims,
    )
    init_ply = tcfg.get('init_ply', '')
    if init_ply:
        gaussians.load_ply(init_ply)
        gaussians.spatial_lr_scale = cameras_extent
        gaussians.init_feature_embeddings()
        if is_main_process:
            print(f"  Warmstarted Gaussians from {init_ply}")
            print(f"  Loaded {gaussians.num_points:,} Gaussians")
    else:
        gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)

    train_args = argparse.Namespace(**{
        'position_lr_init': tcfg['position_lr_init'],
        'position_lr_final': tcfg['position_lr_final'],
        'feature_lr': tcfg['feature_lr'],
        'opacity_lr': tcfg['opacity_lr'],
        'scaling_lr': tcfg['scaling_lr'],
        'rotation_lr': tcfg['rotation_lr'],
        'percent_dense': tcfg['percent_dense'],
        'feature_embedding_lr': tcfg['feature_embedding_lr'],
        'iterations': tcfg['iterations'],
    })
    gaussians.training_setup(train_args)
    if tcfg.get('freeze_geometry', False):
        _freeze_base_gaussians(gaussians)
        if is_main_process:
            print("  Base Gaussian geometry/appearance frozen; training feature embeddings only")

    bg_color = torch.tensor(
        [1, 1, 1] if mcfg['white_background'] else [0, 0, 0],
        dtype=torch.float32, device="cuda"
    )

    # ── 5. Load masks (masks.pkl) ──
    masks = None
    if tcfg.get('use_mask', False):
        mask_candidates = [
            tcfg.get('mask_path', ''),
            os.path.join(dcfg['source_dir'], 'masks.pkl'),
        ]
        for mp in mask_candidates:
            if mp and os.path.exists(mp):
                if is_main_process:
                    print(f"  Loading masks from {mp}")
                with open(mp, 'rb') as f:
                    masks = pickle.load(f)
                matched = sum(1 for c in train_cams if c.image_name in masks)
                if is_main_process:
                    print(f"  Loaded masks for {len(masks)} images, matched {matched}/{len(train_cams)}")
                break
        if masks is None and is_main_process:
            print("  WARNING: use_mask=True but no masks.pkl found")

    # ── 6. Load semantic masks ──
    sem_masks = {}
    sem_dir = tcfg.get('semantic_mask_dir', '')
    if sem_dir:
        sem_masks = load_semantic_masks(sem_dir, train_cams, cam_to_fid=cam_to_fid)

    # ── 7. AppearanceNetwork ──
    appearance_net = None
    app_optimizer = None
    if tcfg.get('use_appearance', False):
        n_images = len(train_cams)
        appearance_net = AppearanceNetwork(
            n_images=n_images,
            embed_dim=tcfg.get('appearance_embed_dim', 32),
        ).cuda()
        app_optimizer = torch.optim.Adam(
            appearance_net.parameters(),
            lr=tcfg.get('appearance_lr', 0.001),
        )
        if is_main_process:
            print(f"  AppearanceNetwork: {n_images} images, embed_dim={tcfg.get('appearance_embed_dim', 32)}")

    # ── 8. TransientHead ──
    transient_head = None
    transient_optimizer = None
    if tcfg.get('use_transient', False):
        transient_head = TransientHead(in_channels=3).cuda()
        transient_optimizer = torch.optim.Adam(
            transient_head.parameters(),
            lr=tcfg.get('transient_lr', 0.001),
        )
        if is_main_process:
            print(f"  TransientHead enabled")

    # ── 9. Pre-cache mono depth ──
    mono_depth_dir = tcfg.get('mono_depth_dir', '')
    mono_depth_cache = {}
    if mono_depth_dir and os.path.isdir(mono_depth_dir):
        if is_main_process:
            print(f"Pre-caching monocular depth from {mono_depth_dir}...")
        for cam in train_cams:
            depth_name = os.path.splitext(cam.image_name)[0] + ".npy"
            depth_path = os.path.join(mono_depth_dir, depth_name)
            if os.path.exists(depth_path):
                d = np.load(depth_path)
                mono_depth_cache[cam.image_name] = 1.0 - d  # invert DPT disparity
        if is_main_process:
            print(f"  Cached {len(mono_depth_cache)} depth maps")
    elif tcfg.get('lambda_depth', 0) > 0 and is_main_process:
        print(f"  WARNING: lambda_depth={tcfg['lambda_depth']} but mono_depth_dir not found")

    # ── 10. Optional fixed-resolution GPU caches ──
    image_cache = {}
    mask_cache = {}
    if tcfg.get('cache_resized_images', False):
        cache_dtype_name = str(tcfg.get('image_cache_dtype', 'float16')).lower()
        cache_dtype = torch.float32 if cache_dtype_name == 'float32' else torch.float16
        if is_main_process:
            print(f"Pre-caching resized train images on GPU ({cache_dtype_name})...")
        try:
            image_cache = _cache_resized_images(
                train_cams, tcfg['longest_edge'], cache_dtype, is_main_process
            )
            if is_main_process:
                print(f"  Cached {len(image_cache)} resized images")
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            image_cache.clear()
            torch.cuda.empty_cache()
            if is_main_process:
                print("  WARNING: image cache OOM; falling back to on-demand image loads")
    elif is_main_process:
        print("Skipping image pre-cache (on-demand loading enabled)")

    if masks is not None and tcfg.get('cache_resized_masks', False):
        if is_main_process:
            print("Pre-caching resized train masks on GPU...")
        try:
            mask_cache = _cache_resized_masks(
                train_cams, masks, tcfg['longest_edge'], is_main_process
            )
            if is_main_process:
                print(f"  Cached {len(mask_cache)} resized masks")
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            mask_cache.clear()
            torch.cuda.empty_cache()
            if is_main_process:
                print("  WARNING: mask cache OOM; falling back to on-demand mask resize")

    # Build cam_uid → sequential index mapping for AppearanceNetwork
    cam_uid_to_idx = {cam.uid: i for i, cam in enumerate(train_cams)}

    # ════════════════════════════════════════════════════════════════════════
    # Training loop
    # ════════════════════════════════════════════════════════════════════════
    iterations = tcfg['iterations']
    feat_start = tcfg['feature_start_iter']
    feat_weight = tcfg['feature_weight']
    feat_cos_w = tcfg['feature_cos_weight']
    feat_detach = tcfg['detach_feat_geometry']
    lambda_dssim = tcfg['lambda_dssim']
    lambda_normal = tcfg['lambda_normal']
    lambda_dist = tcfg['lambda_dist']
    lambda_scale = tcfg['lambda_scale']
    reg_start = tcfg['reg_start_iter']
    longest_edge = tcfg['longest_edge']

    # Select render function: 2DGS (surfel) vs 3DGS (ellipsoid)
    use_3dgs = tcfg.get('use_3dgs', False)
    render_fn = render_rgb_3dgs if use_3dgs else render_rgb_2dgs
    if use_3dgs:
        # 3DGS doesn't produce normal/dist, disable those losses
        lambda_normal = 0.0
        lambda_dist = 0.0
        log("  [3DGS mode] Disabled normal/dist regularization")
    veg_reg_scale = tcfg.get('vegetation_reg_scale', 0.1)

    grad_accum_steps = max(1, int(tcfg.get('grad_accum_steps', 1)))
    batch_size = max(1, int(tcfg.get('batch_size', grad_accum_steps)))
    if batch_size == 1 and grad_accum_steps > 1:
        batch_size = grad_accum_steps

    viewpoint_stack = []
    ema_loss = 0.0
    ema_rgb = 0.0
    ema_depth = 0.0
    ema_feat = {s: 0.0 for s in scales}
    best_psnr = 0.0

    log_file = open(os.path.join(output_dir, 'train.log'), 'w') if is_main_process else None

    def log(msg):
        if not is_main_process:
            return
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    if batch_size > 1 and not use_3dgs:
        log(f"  Batched rasterization: {batch_size} cameras per optimizer step per GPU (global batch={batch_size * world_size})")
    elif batch_size > 1:
        log(f"  Batch size: {batch_size} cameras per step (sequential 3DGS fallback)")
    else:
        log("  Batch size: 1 camera per optimizer step")

    pbar = tqdm(range(1, iterations + 1), desc="Joint v3", disable=not is_main_process)
    consecutive_bad_steps = 0
    max_consecutive_bad_steps = max(1, int(tcfg.get('max_consecutive_bad_steps', 50)))
    for iteration in pbar:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ── Render a multi-camera batch, then average per-view losses ──
        accum_loss_val = 0.0
        accum_rgb_val = 0.0
        accum_depth_val = 0.0
        bad_step = False
        losses = []

        cams_batch = []
        for _ in range(batch_size):
            if not viewpoint_stack:
                viewpoint_stack = list(train_cams)
                shuffle(viewpoint_stack)
            cams_batch.append(viewpoint_stack.pop())

        if batch_size > 1 and not use_3dgs:
            try:
                render_pkgs = render_rgb_2dgs_batch(
                    gaussians, cams_batch, bg_color, longest_edge
                )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                if is_main_process:
                    tqdm.write(
                        f"  [Iter {iteration}] OOM in batched render; "
                        "falling back to sequential views"
                    )
                render_pkgs = [
                    render_fn(gaussians, cam, bg_color, longest_edge)
                    for cam in cams_batch
                ]
        else:
            render_pkgs = [
                render_fn(gaussians, cam, bg_color, longest_edge)
                for cam in cams_batch
            ]

        for accum_step, (cam, render_pkg) in enumerate(zip(cams_batch, render_pkgs)):
            fid = cam_to_fid.get(cam.uid)
            cam_idx = cam_uid_to_idx[cam.uid]

            # ── RGB render ──
            image = render_pkg["render"]  # [3, H, W]
            rw, rh = render_pkg["width"], render_pkg["height"]

            # ── Apply AppearanceNetwork ──
            if appearance_net is not None:
                image = appearance_net(image, cam_idx)

            gt_image = image_cache.get(cam.uid) if image_cache else None
            if gt_image is not None and (gt_image.shape[1] != rh or gt_image.shape[2] != rw):
                gt_image = None
            if gt_image is None:
                gt_image = load_image_tensor(cam)
                gt_image = F.interpolate(
                    gt_image.unsqueeze(0), size=(rh, rw),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
            else:
                gt_image = gt_image.to(dtype=image.dtype)

            # ── Sky / object masking ──
            rgb_mask = None
            sky_mask_2d = None
            cached_masks = mask_cache.get(cam.uid) if mask_cache else None
            if cached_masks is not None:
                obj_mask, sky_mask_raw, distort_mask = cached_masks
            elif masks is not None and cam.image_name in masks:
                obj_mask = masks[cam.image_name][0].cuda()[None]
                sky_mask_raw = masks[cam.image_name][1].cuda()[None]
                distort_mask = masks[cam.image_name][2].cuda()[None]

                if obj_mask.shape[1] != rh or obj_mask.shape[2] != rw:
                    obj_mask = F.interpolate(obj_mask[None].float(), size=(rh, rw),
                                             mode="nearest").squeeze(0) > 0.5
                    sky_mask_raw = F.interpolate(sky_mask_raw[None].float(), size=(rh, rw),
                                                  mode="nearest").squeeze(0) > 0.5
                    distort_mask = F.interpolate(distort_mask[None].float(), size=(rh, rw),
                                                  mode="nearest").squeeze(0) > 0.5
            else:
                obj_mask = sky_mask_raw = distort_mask = None

            if obj_mask is not None:
                rgb_mask = (obj_mask & distort_mask).float()
                sky_mask_2d = ~sky_mask_raw

                image = image * rgb_mask + bg_color[:, None, None] * (1.0 - rgb_mask)
                gt_image = gt_image * rgb_mask + bg_color[:, None, None] * (1.0 - rgb_mask)

            # ── Transient confidence mask ──
            transient_weight = None
            if transient_head is not None:
                transient_weight = transient_head(render_pkg["render"].detach())

            # ── RGB loss: L1 + SSIM ──
            # Always compute RAW (unweighted) L1 for densification-healthy gradients
            Ll1_raw = F.l1_loss(image, gt_image)
            ssim_val = ssim(image, gt_image)
            rgb_loss_raw = (1.0 - lambda_dssim) * Ll1_raw + lambda_dssim * (1.0 - ssim_val)

            if transient_weight is not None:
                pixel_l1 = (image - gt_image).abs().mean(dim=0, keepdim=True)
                Ll1 = (pixel_l1 * transient_weight).mean()
                rgb_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_val)
            else:
                rgb_loss = rgb_loss_raw

            loss = rgb_loss

            # ── Transient regularization ──
            if transient_weight is not None and tcfg.get('transient_reg', 0) > 0:
                t_reg = ((transient_weight - 0.5) ** 2).mean()
                loss = loss + tcfg['transient_reg'] * t_reg

            # ── Semantic-aware geometry regularization + dynamic masking ──
            sem_weight = None
            dynamic_mask_2d = None
            if sem_masks and cam.image_name in sem_masks:
                sem_label = sem_masks[cam.image_name]
                sem_resized = F.interpolate(
                    sem_label.float()[None, None], size=(rh, rw), mode='nearest'
                ).squeeze().long()
                is_veg = (sem_resized == 2)
                is_dynamic = (sem_resized == 5)
                sem_weight = torch.ones(1, rh, rw, device='cuda')
                sem_weight[0, is_veg] = veg_reg_scale
                sem_weight[0, is_dynamic] = 0.0  # zero-out dynamic regions in geometry reg

                # Build dynamic object mask: exclude from RGB and feature losses
                if is_dynamic.any():
                    dynamic_mask_2d = (~is_dynamic).float()[None]  # [1, H, W], 1=valid 0=dynamic

            # ── Apply dynamic mask to RGB loss (recompute excluding dynamic pixels) ──
            if dynamic_mask_2d is not None:
                pixel_l1 = (image - gt_image).abs().mean(dim=0, keepdim=True)
                if transient_weight is not None:
                    pixel_l1 = pixel_l1 * transient_weight
                n_valid = dynamic_mask_2d.sum().clamp(min=1)
                masked_l1 = (pixel_l1 * dynamic_mask_2d).sum() / n_valid
                rgb_loss_masked = (1.0 - lambda_dssim) * masked_l1 + lambda_dssim * (1.0 - ssim_val)
                loss = loss - rgb_loss + rgb_loss_masked
                rgb_loss = rgb_loss_masked

            # ── 2DGS geometry regularization ──
            if iteration > reg_start:
                rend_dist = render_pkg["rend_dist"]
                rend_normal = render_pkg["rend_normal"]
                surf_normal = render_pkg["surf_normal"]
                rend_alpha = render_pkg["rend_alpha"]

                if lambda_normal > 0 and rend_normal is not None and surf_normal is not None:
                    surf_n = surf_normal * rend_alpha.squeeze(0).detach()
                    rend_n = rend_normal.squeeze(0).permute(2, 0, 1)
                    if len(surf_n.shape) == 4:
                        surf_n = surf_n.squeeze(0)
                    surf_n = surf_n.permute(2, 0, 1)
                    normal_error = (1 - (rend_n * surf_n).sum(dim=0))[None]
                    normal_error = torch.nan_to_num(
                        normal_error,
                        nan=0.0,
                        posinf=tcfg.get('normal_error_max', 2.0),
                        neginf=0.0,
                    ).clamp(min=0.0, max=tcfg.get('normal_error_max', 2.0))
                    if sem_weight is not None:
                        normal_error = normal_error * sem_weight
                    loss = loss + lambda_normal * normal_error.mean()

                if lambda_dist > 0 and rend_dist is not None:
                    dist_loss = rend_dist.squeeze(-1)
                    dist_loss = torch.nan_to_num(
                        dist_loss,
                        nan=0.0,
                        posinf=tcfg.get('dist_loss_max', 10.0),
                        neginf=0.0,
                    ).clamp(min=0.0, max=tcfg.get('dist_loss_max', 10.0))
                    if len(dist_loss.shape) == 3:
                        if sem_weight is not None:
                            dist_loss = dist_loss * sem_weight
                        loss = loss + lambda_dist * dist_loss.mean()
                    else:
                        loss = loss + lambda_dist * dist_loss.mean()

                if lambda_scale > 0:
                    log_threshold = math.log(max(tcfg['scale_reg_threshold'], 1e-6))
                    excess = torch.clamp(
                        gaussians._scaling.max(dim=1).values - log_threshold, min=0
                    )
                    scale_penalty = torch.clamp(
                        excess ** 2,
                        max=tcfg.get('scale_loss_max', 10.0),
                    ).mean()
                    # Boost scale regularization in vegetation areas
                    veg_scale_boost = tcfg.get('vegetation_scale_boost', 1.0)
                    if veg_scale_boost > 1.0 and sem_masks and cam.image_name in sem_masks:
                        # Per-Gaussian scale weight: project to check if in vegetation
                        # Approximate: use per-pixel weight on rendered scale loss
                        scale_sem_weight = torch.ones(1, rh, rw, device='cuda')
                        if sem_weight is not None:
                            # sem_weight has veg_reg_scale in veg areas; invert and apply boost
                            is_veg_px = (sem_weight < 1.0)
                            scale_sem_weight[is_veg_px] = veg_scale_boost
                        # Scale reg is per-Gaussian, not per-pixel, so apply global boost
                        # based on vegetation fraction in current view
                        veg_frac = (scale_sem_weight > 1.0).float().mean()
                        effective_boost = 1.0 + (veg_scale_boost - 1.0) * veg_frac.item()
                        loss = loss + lambda_scale * effective_boost * scale_penalty
                    else:
                        loss = loss + lambda_scale * scale_penalty

            # ── Monocular depth supervision ──
            depth_loss_val = torch.tensor(0.0, device="cuda")
            lambda_depth_cfg = tcfg.get('lambda_depth', 0.0)
            depth_start = tcfg.get('depth_start_iter', 3000)
            depth_warmup = tcfg.get('depth_warmup_iters', 2000)
            if lambda_depth_cfg > 0 and mono_depth_cache and iteration > depth_start:
                warmup_progress = min(1.0, (iteration - depth_start) / max(1, depth_warmup))
                lambda_depth_now = lambda_depth_cfg * warmup_progress

                rendered_depth = render_pkg["depth"]
                if cam.image_name in mono_depth_cache:
                    mono_d = torch.from_numpy(mono_depth_cache[cam.image_name]).float().cuda()
                    if mono_d.shape[0] != rh or mono_d.shape[1] != rw:
                        mono_d = F.interpolate(
                            mono_d[None, None], size=(rh, rw),
                            mode="bilinear", align_corners=False
                        ).squeeze()

                    depth_valid_mask = None
                    if sky_mask_2d is not None:
                        depth_valid_mask = (~sky_mask_2d).squeeze(0)

                    # Exclude vegetation from depth supervision (unreliable monocular depth)
                    depth_exclude_veg = tcfg.get('depth_exclude_vegetation', False)
                    if depth_exclude_veg and sem_masks and cam.image_name in sem_masks:
                        sem_label = sem_masks[cam.image_name]
                        sem_for_depth = F.interpolate(
                            sem_label.float()[None, None], size=(rh, rw), mode='nearest'
                        ).squeeze().long()
                        veg_mask = (sem_for_depth == 2)
                        if depth_valid_mask is not None:
                            depth_valid_mask = depth_valid_mask & (~veg_mask)
                        else:
                            depth_valid_mask = ~veg_mask

                    depth_loss_val = lambda_depth_now * pearson_depth_loss(
                        rendered_depth, mono_d, valid_mask=depth_valid_mask
                    )
                    loss = loss + depth_loss_val

            # ── Feature losses ──
            total_feat_loss = torch.tensor(0.0, device="cuda")
            if fid is not None and iteration >= feat_start:
                viewmat = cam.get_world_view_transform()
                for scale in scales:
                    gt_feat = feat_cache.get(scale, fid)
                    if gt_feat is None:
                        continue
                    si = scale_infos[scale]
                    feat_colors = gaussians.get_feature(scale)

                    if feat_detach:
                        det_means = gaussians.get_xyz.detach()
                        det_opacity = gaussians.get_opacity.detach()
                        det_scales2d = gaussians.get_scaling.detach()
                        det_rots = gaussians.get_rotation.detach()
                        det_scales3 = torch.cat([
                            det_scales2d,
                            torch.ones(det_scales2d.shape[0], 1, device="cuda"),
                        ], dim=-1)
                        rast_fn = rasterization if use_3dgs else rasterization_2dgs
                        D = feat_colors.shape[1]
                        chunks = []
                        for ci in range((D + 31) // 32):
                            cs, ce = ci * 32, min((ci + 1) * 32, D)
                            rc, *_ = rast_fn(
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
                        if use_3dgs:
                            # 3DGS: inline rendering with rasterization
                            _means = gaussians.get_xyz
                            _opacity = gaussians.get_opacity
                            _scales2d = gaussians.get_scaling
                            _rots = gaussians.get_rotation
                            _scales3 = torch.cat([_scales2d, torch.ones(_scales2d.shape[0], 1, device="cuda")], dim=-1)
                            _D = feat_colors.shape[1]
                            _chunks = []
                            for ci in range((_D + 31) // 32):
                                cs, ce = ci * 32, min((ci + 1) * 32, _D)
                                rc, *_ = rasterization(
                                    means=_means, quats=_rots, scales=_scales3,
                                    opacities=_opacity.squeeze(-1),
                                    colors=feat_colors[:, cs:ce],
                                    viewmats=viewmat[None], Ks=si['K'][None],
                                    width=si['w'], height=si['h'],
                                    packed=False, near_plane=0.01,
                                    far_plane=500, render_mode='RGB',
                                )
                                _chunks.append(rc)
                            fm = torch.cat(_chunks, dim=-1)[0].permute(2, 0, 1)
                            rendered_feat = F.normalize(fm, p=2, dim=0)
                        else:
                            rendered_feat = render_features_2dgs(
                                gaussians, viewmat, feat_colors,
                                si['h'], si['w'], si['K'],
                            )

                    feat_l1_map = (rendered_feat - gt_feat).abs().mean(dim=0, keepdim=True)
                    cos_map = F.cosine_similarity(rendered_feat, gt_feat, dim=0).unsqueeze(0)

                    if dynamic_mask_2d is not None:
                        dyn_mask_feat = F.interpolate(
                            dynamic_mask_2d[None], size=(si['h'], si['w']), mode='nearest'
                        )[0]  # [1, h, w]
                        n_valid_f = dyn_mask_feat.sum().clamp(min=1)
                        feat_l1 = (feat_l1_map * dyn_mask_feat).sum() / n_valid_f
                        feat_cos = 1.0 - (cos_map * dyn_mask_feat).sum() / n_valid_f
                    else:
                        feat_l1 = feat_l1_map.mean()
                        feat_cos = 1.0 - cos_map.mean()
                    scale_loss = feat_l1 + feat_cos_w * feat_cos
                    total_feat_loss = total_feat_loss + scale_loss
                    ema_feat[scale] = 0.9 * ema_feat[scale] + 0.1 * scale_loss.item()

                    # ── Contrastive loss: push different-position features apart ──
                    contrastive_w = tcfg.get('contrastive_weight', 0.0)
                    if contrastive_w > 0:
                        c_mask = dyn_mask_feat if dynamic_mask_2d is not None else None
                        c_loss = contrastive_feature_loss(
                            rendered_feat, gt_feat,
                            n_anchors=tcfg.get('contrastive_anchors', 256),
                            n_negatives=tcfg.get('contrastive_negatives', 128),
                            temperature=tcfg.get('contrastive_temperature', 0.07),
                            mask=c_mask,
                        )
                        total_feat_loss = total_feat_loss + contrastive_w * c_loss

                loss = loss + feat_weight * total_feat_loss

            # ── Appearance regularization (once per batch, on last step) ──
            if accum_step == batch_size - 1 and appearance_net is not None:
                app_reg_w = tcfg.get('appearance_reg', 0.0)
                app_mean_reg = tcfg.get('appearance_mean_reg', 0.0)
                if app_reg_w > 0 or app_mean_reg > 0:
                    n_reg = min(64, appearance_net.n_images)
                    reg_idx = torch.randint(0, appearance_net.n_images, (n_reg,), device='cuda')
                    reg_embeds = appearance_net.embedding(reg_idx)
                    reg_params = appearance_net.mlp(reg_embeds)
                    sr = appearance_net.scale_range
                    br = appearance_net.bias_range
                    reg_scale = torch.sigmoid(reg_params[:, :3]) * sr + (1.0 - sr / 2)
                    reg_bias = torch.tanh(reg_params[:, 3:]) * br

                    if app_reg_w > 0:
                        identity_loss = ((reg_scale - 1.0) ** 2).mean() + (reg_bias ** 2).mean()
                        loss = loss + app_reg_w * identity_loss

                    if app_mean_reg > 0:
                        mean_scale = reg_scale.mean(dim=0)
                        mean_bias = reg_bias.mean(dim=0)
                        mean_loss = ((mean_scale - 1.0) ** 2).sum() + (mean_bias ** 2).sum()
                        loss = loss + app_mean_reg * mean_loss

            # ── Safety check ──
            loss_val = loss.item()
            if torch.isnan(loss) or torch.isinf(loss) or loss_val < -0.01:
                if is_main_process:
                    tqdm.write(f"  [Iter {iteration}] Bad loss={loss_val:.4g}, skipping batch")
                bad_step = True
                break
            if loss_val > 10.0:
                loss = loss.clamp(max=10.0)
                loss_val = loss.item()

            # ── Accumulate EMA values ──
            losses.append(loss)
            accum_loss_val += loss_val / batch_size
            accum_rgb_val += rgb_loss_raw.item() / batch_size
            depth_v = depth_loss_val.item() if torch.is_tensor(depth_loss_val) else depth_loss_val
            accum_depth_val += depth_v / batch_size

        # ── End of per-view loss loop ──

        bad_step_tensor = torch.tensor(
            [1 if bad_step or not losses else 0],
            device="cuda",
            dtype=torch.int32,
        )
        if is_distributed:
            dist.all_reduce(bad_step_tensor, op=dist.ReduceOp.MAX)

        if bad_step_tensor.item() > 0:
            consecutive_bad_steps += 1
            gaussians.optimizer.zero_grad(set_to_none=True)
            if app_optimizer is not None:
                app_optimizer.zero_grad(set_to_none=True)
            if transient_optimizer is not None:
                transient_optimizer.zero_grad(set_to_none=True)
            if consecutive_bad_steps >= max_consecutive_bad_steps:
                raise RuntimeError(
                    f"Aborting after {consecutive_bad_steps} consecutive bad steps; "
                    "Gaussian parameters are likely unstable."
                )
            continue

        consecutive_bad_steps = 0

        total_loss = sum(losses) / batch_size
        total_loss.backward()

        # ── Densification stats from every view in the rasterization batch ──
        with torch.no_grad():
            if iteration < tcfg['densify_until_iter']:
                before_accum = gaussians.xyz_gradient_accum.clone() if is_distributed else None
                before_denom = gaussians.denom.clone() if is_distributed else None
                for view_idx, render_pkg in enumerate(render_pkgs):
                    vp = render_pkg["viewspace_points"]
                    grad_data = vp.grad if vp.grad is not None else vp
                    if grad_data.dim() == 3 and grad_data.size(0) > 1:
                        grad_data = grad_data[view_idx:view_idx + 1]
                    radii = render_pkg["radii"]
                    vis = render_pkg["visibility_filter"]
                    gaussians.max_radii2D[vis] = torch.max(
                        gaussians.max_radii2D[vis], radii[vis]
                    )
                    gaussians.add_densification_stats(
                        grad_data, vis, render_pkg["width"], render_pkg["height"]
                    )
                _sync_densification_stats(
                    gaussians, before_accum, before_denom, world_size
                )

        _all_reduce_optimizer_grads(
            [gaussians.optimizer, app_optimizer, transient_optimizer],
            world_size,
        )

        # ── Gradient clipping ──
        clip_params = [
            gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
            gaussians._scaling, gaussians._rotation, gaussians._opacity,
        ]
        for sn in scales:
            clip_params.append(getattr(gaussians, f'_feat_{sn}'))
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)

        # ── EMA update ──
        ema_loss = 0.4 * accum_loss_val + 0.6 * ema_loss
        ema_rgb = 0.4 * accum_rgb_val + 0.6 * ema_rgb
        ema_depth = 0.4 * accum_depth_val + 0.6 * ema_depth

        with torch.no_grad():
            if is_main_process and iteration % 10 == 0:
                feat_str = " ".join(f"{s[0]}={ema_feat[s]:.3f}" for s in scales)
                pbar.set_postfix({
                    "L": f"{ema_loss:.4f}", "RGB": f"{ema_rgb:.4f}",
                    "D": f"{ema_depth:.4f}",
                    "F": feat_str, "N": f"{gaussians.num_points:,}",
                })

            # ── Periodic eval + visualization ──
            eval_interval = tcfg['eval_interval']
            if is_main_process and (iteration % eval_interval == 0 or iteration == iterations):
                eval_cams = test_cams if test_cams else train_cams
                psnr_val = evaluate_psnr(
                    gaussians, eval_cams, bg_color, longest_edge
                )
                feat_str = " | ".join(f"{s}={ema_feat[s]:.4f}" for s in scales)
                msg = (f"  [Iter {iteration}] PSNR={psnr_val:.2f}dB | "
                       f"RGB={ema_rgb:.4f} | Depth={ema_depth:.4f} | "
                       f"Feat=[{feat_str}] | "
                       f"N={gaussians.num_points:,}")
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    msg += " ★ BEST"
                    best_dir = os.path.join(output_dir, "point_cloud", "best")
                    gaussians.save_ply(os.path.join(best_dir, "point_cloud.ply"))
                    gaussians.save_features(
                        os.path.join(output_dir, "features_best")
                    )
                log(msg)

                vis_dir = os.path.join(output_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)
                visualize_comparison_v3(
                    gaussians, train_cams, cam_to_fid, feat_cache,
                    scale_infos, scales, cfg, iteration, vis_dir,
                    appearance_net=appearance_net, masks=masks,
                )

            # ── Checkpoints ──
            save_interval = tcfg['save_interval']
            if is_main_process and (iteration % save_interval == 0 or iteration == iterations):
                save_dir = os.path.join(
                    output_dir, "point_cloud", f"iteration_{iteration}"
                )
                gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
                gaussians.save_features(
                    os.path.join(output_dir, "features"), iteration
                )

            # ── Densification ──
            max_gaussians = tcfg.get('max_gaussians', 0)
            if iteration < tcfg['densify_until_iter']:
                if (iteration > tcfg['densify_from_iter'] and
                        iteration % tcfg['densification_interval'] == 0):
                    # size_threshold: configurable (default None = no screen-space pruning)
                    # Old hardcoded: 20 after first opacity reset → kills large outdoor Gaussians
                    size_threshold = tcfg.get('size_threshold', None)
                    # Skip densification if at Gaussian budget
                    if max_gaussians > 0 and gaussians.num_points >= max_gaussians:
                        # Only prune, no densification
                        prune_mask = (gaussians.get_opacity < 0.005).squeeze()
                        if size_threshold:
                            big_vs = gaussians.max_radii2D > size_threshold
                            big_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * cameras_extent
                            prune_mask = prune_mask | big_vs | big_ws
                        if prune_mask.any():
                            gaussians._prune_points(prune_mask)
                            torch.cuda.empty_cache()
                    else:
                        if is_distributed:
                            torch.manual_seed(1000000 + iteration)
                        gaussians.densify_and_prune(
                            tcfg['densify_grad_threshold'], 0.005,
                            cameras_extent, size_threshold,
                        )
                # Vegetation-aware pruning + clamping (BEFORE opacity reset)
                veg_prune_opacity = tcfg.get('veg_prune_opacity', 0.0)
                veg_prune_scale_pct = tcfg.get('veg_prune_scale_percentile', 0)
                veg_max_scale = tcfg.get('veg_max_scale', 0.0)
                if ((veg_prune_opacity > 0 or veg_prune_scale_pct > 0 or veg_max_scale > 0)
                        and sem_masks and iteration % tcfg['opacity_reset_interval'] == 0):
                    n_opa, n_scl, n_clamp, n_veg = prune_vegetation_gaussians(
                        gaussians, train_cams, sem_masks,
                        veg_opacity_threshold=veg_prune_opacity,
                        veg_scale_percentile=veg_prune_scale_pct,
                        veg_max_scale=veg_max_scale,
                        scene_extent=cameras_extent,
                        n_sample_cams=tcfg.get('veg_sample_cams', 50),
                        veg_vote_threshold=tcfg.get('veg_vote_threshold', 0.3),
                        log_fn=log,
                    )
                    log(f"  [Iter {iteration}] Veg: {n_veg} identified, pruned {n_opa} opacity"
                        f" + {n_scl} scale, clamped {n_clamp}")

                if iteration % tcfg['opacity_reset_interval'] == 0:
                    gaussians.reset_opacity(
                        reset_value=tcfg['opacity_reset_value']
                    )

            # ── Optimizer steps ──
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # AppearanceNetwork + TransientHead optimizer steps
        if app_optimizer is not None:
            app_optimizer.step()
            app_optimizer.zero_grad(set_to_none=True)
        if transient_optimizer is not None:
            transient_optimizer.step()
            transient_optimizer.zero_grad(set_to_none=True)

        # ── Logging ──
        if iteration % 1000 == 1:
            feat_detail = " | ".join(
                f"{s}={ema_feat[s]:.4f}" for s in scales
            )
            log(f"  [Iter {iteration}] "
                f"RGB={ema_rgb:.4f} Depth={ema_depth:.4f} Feat=[{feat_detail}] "
                f"Total={ema_loss:.4f} N={gaussians.num_points:,}")

    # ── Final ──
    log(f"\n  Training complete. Best PSNR: {best_psnr:.2f}dB")
    log(f"  Output: {output_dir}")
    if log_file is not None:
        log_file.close()
    _cleanup_distributed(is_distributed)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 2DGS + Feature Training v3')
    parser.add_argument('--config', required=True, help='YAML config path')
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
