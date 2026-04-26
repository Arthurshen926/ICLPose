#!/usr/bin/env python3
"""
Joint 2DGS + GSFF Triplane + Raw RADIO Feature Training
========================================================
联合训练三个组件:
  1. 2DGS 几何+外观重建 (v7 recipe)
  2. GSFF triplane + encoder 自监督对比特征学习
  3. 原始 RADIO 1280d 特征监督 (通过 projection head)

Training schedule:
  Phase 1 (0 ~ gsff_start_iter): 仅训练 2DGS 几何 (RGB + depth + reg)
  Phase 2 (gsff_start_iter ~ end): 联合训练几何 + triplane + encoder + RADIO

Usage:
    CUDA_VISIBLE_DEVICES=2 python legacy/gsff_baseline/scripts/train_joint_gsff_radio.py \
        --config legacy/gsff_baseline/configs/joint_oh_v8_gsff_radio.yaml
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from random import randint, shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not locate repository root from script path")

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))


from gsplat import rasterization_2dgs, spherical_harmonics

# 2DGS components
from feature_3dgs.train_2dgs_geometry import (
    read_cameras_binary, read_images_binary, read_points3d_binary,
    qvec2rotmat, focal2fov, RGB2SH, inverse_sigmoid, build_rotation,
    GaussianModel2DGS, CameraData, ssim, load_image_tensor,
    pearson_depth_loss, load_mono_depth, AppearanceNetwork,
)
from feature_3dgs.train_2dgs_joint_v2 import (
    load_scene_colmap, _compute_extent, pca_colorize, evaluate_psnr,
)
from feature_3dgs.train_2dgs_joint_v3 import (
    TransientHead, load_semantic_masks, prune_vegetation_gaussians,
)
from feature_3dgs.train_2dgs_joint import (
    render_rgb_2dgs, build_da3_image_order,
)

# GSFF components
from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder, SegmentationHead
from gsff.losses import (
    info_nce_loss, prototypical_loss, segmentation_ce_loss,
    compute_pseudo_logits, get_pixel_proto_labels,
)
from gsff.clustering import spectral_clustering, compute_prototypes


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    'exp_name': 'joint_gsff_radio',
    'dataset': {
        'type': 'cambridge',
        'source_dir': '',
        'images': '',
        'radio_feature_dir': '',
        'radio_feature_scale': 'fine_radio',
    },
    'model': {
        'sh_degree': 3,
        'white_background': False,
    },
    'gsff': {
        'feature_dim': 16,
        'coarse_resolution': 256,
        'fine_resolution': 1024,
        'n_clusters': 34,
        'temperature': 0.07,
        'nce_samples': 1024,
        'cosine_loss_weight': 0.5,
        'prototype_update_freq': 500,
        'radio_proj_hidden': [128, 256],
        'weight_nce': 0.5,
        'weight_pro': 0.5,
        'weight_ce': 0.5,
        'weight_radio': 0.3,
        'weight_tvl': 0.1,
        'gsff_start_iter': 2000,
        'fine_start_iter': 8000,
        'lr_triplane': 1e-3,
        'lr_encoder': 1e-4,
        'lr_radio_proj': 5e-4,
    },
    'training': {
        'iterations': 40000,
        'longest_edge': 0,
        'eval_interval': 2000,
        'save_interval': 5000,
        'vis_frames': [0, 200, 500, 800],
        'position_lr_init': 0.00016,
        'position_lr_final': 0.0000016,
        'feature_lr': 0.0025,
        'opacity_lr': 0.05,
        'scaling_lr': 0.005,
        'rotation_lr': 0.001,
        'percent_dense': 0.01,
        'lambda_dssim': 0.2,
        'lambda_normal': 0.05,
        'lambda_dist': 0.01,
        'lambda_scale': 0.001,
        'scale_reg_threshold': 0.3,
        'reg_start_iter': 500,
        'mono_depth_dir': '',
        'lambda_depth': 0.05,
        'depth_start_iter': 3000,
        'depth_warmup_iters': 2000,
        'depth_exclude_vegetation': False,
        'densify_from_iter': 500,
        'densify_until_iter': 25000,
        'densification_interval': 100,
        'densify_grad_threshold': 0.0002,
        'opacity_reset_interval': 6000,
        'opacity_reset_value': 0.01,
        'use_mask': False,
        'mask_path': '',
        'use_appearance': False,
        'appearance_embed_dim': 32,
        'appearance_lr': 0.001,
        'appearance_reg': 0.0,
        'appearance_mean_reg': 0.01,
        'use_transient': False,
        'transient_lr': 0.001,
        'transient_reg': 0.01,
        'semantic_mask_dir': '',
        'vegetation_reg_scale': 0.1,
        'vegetation_scale_boost': 1.0,
        'veg_prune_opacity': 0.0,
        'veg_prune_scale_percentile': 0,
        'grad_accum_steps': 1,
    },
    'output_dir': 'output/joint_gsff_radio',
}


def _deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    with open(path) as f:
        user_cfg = yaml.safe_load(f)
    return _deep_merge(DEFAULT_CONFIG.copy(), user_cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Radio Feature Cache (raw 1280d)
# ═══════════════════════════════════════════════════════════════════════════

class RadioFeatureCache:
    """Lazy-loading cache for raw 1280d RADIO features.
    
    Caches on CPU to avoid GPU OOM (1077 frames × ~40MB = 43GB).
    Only transfers to GPU on demand.
    """

    def __init__(self, feature_dir, scale='fine_radio', max_gpu_cache=32):
        self.feature_dir = Path(feature_dir) / scale
        self._cpu_cache = {}
        self._gpu_cache = {}   # small LRU on GPU
        self._gpu_order = []   # access order for LRU eviction
        self._max_gpu = max_gpu_cache
        self._info = None

        # Discover available frame IDs
        self._frame_ids = set()
        if self.feature_dir.is_dir():
            import re
            for f in self.feature_dir.iterdir():
                m = re.match(r'rgb_(\d+)_fine_radio_(\d+)x(\d+)x(\d+)\.pt', f.name)
                if m:
                    self._frame_ids.add(int(m.group(1)))
                    if self._info is None:
                        self._info = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

        if self._info:
            print(f"  [RadioFeatureCache] {len(self._frame_ids)} frames, "
                  f"{self._info[0]}d @ {self._info[2]}x{self._info[1]} "
                  f"(GPU LRU={max_gpu_cache})")
        else:
            print(f"  [RadioFeatureCache] WARNING: no features found in {self.feature_dir}")

    @property
    def frame_ids(self):
        return self._frame_ids

    @property
    def dim(self):
        return self._info[0] if self._info else 0

    @property
    def height(self):
        return self._info[1] if self._info else 0

    @property
    def width(self):
        return self._info[2] if self._info else 0

    def get(self, fid):
        """Get feature tensor [C, H, W] on GPU for frame ID."""
        # Check GPU cache first
        if fid in self._gpu_cache:
            # Move to end (most recently used)
            if fid in self._gpu_order:
                self._gpu_order.remove(fid)
            self._gpu_order.append(fid)
            return self._gpu_cache[fid]

        # Load from CPU cache or disk
        if fid in self._cpu_cache:
            feat_cpu = self._cpu_cache[fid]
        else:
            if self._info is None:
                return None
            d, h, w = self._info
            fname = f'rgb_{fid}_fine_radio_{d}x{h}x{w}.pt'
            fpath = self.feature_dir / fname
            if not fpath.exists():
                return None
            feat_cpu = torch.load(str(fpath), map_location='cpu', weights_only=True).float()
            self._cpu_cache[fid] = feat_cpu

        # Transfer to GPU with LRU eviction
        feat_gpu = feat_cpu.cuda()
        self._gpu_cache[fid] = feat_gpu
        self._gpu_order.append(fid)

        # Evict oldest if over limit
        while len(self._gpu_cache) > self._max_gpu:
            old_fid = self._gpu_order.pop(0)
            if old_fid in self._gpu_cache:
                del self._gpu_cache[old_fid]

        return feat_gpu

    def clear_cache(self):
        """Free memory."""
        self._gpu_cache.clear()
        self._gpu_order.clear()
        self._cpu_cache.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Radio Projection Head
# ═══════════════════════════════════════════════════════════════════════════

class RadioProjectionHead(nn.Module):
    """MLP that projects triplane features (16d) to RADIO space (1280d).

    Applied per-pixel on rendered feature maps:
        [B, 16, H, W] → [B, 1280, H, W]

    Implemented as 1x1 convolutions for spatial efficiency.
    """

    def __init__(self, input_dim=16, output_dim=1280, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256]

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Conv2d(prev_dim, hd, 1),
                nn.GELU(),
            ])
            prev_dim = hd
        layers.append(nn.Conv2d(prev_dim, output_dim, 1))
        self.net = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  [RadioProjectionHead] {input_dim}d → {output_dim}d, "
              f"hidden={hidden_dims}, {n_params:,} params")

    def forward(self, x):
        """x: [B, D_in, H, W] → [B, D_out, H, W]"""
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════
# Feature rendering through 2DGS
# ═══════════════════════════════════════════════════════════════════════════

def render_features_through_gaussians(means3d, quats, scales, opacities,
                                       features, viewmats, Ks, width, height,
                                       chunk_size=16):
    """Render feature maps by alpha-blending per-Gaussian features.

    Args:
        features: [N, D] per-Gaussian features (from triplane)
        Returns: [C, D, H, W]
    """
    D = features.shape[1]
    C = viewmats.shape[0]
    n_chunks = (D + chunk_size - 1) // chunk_size

    all_results = []
    for ci in range(C):
        cam_chunks = []
        for i in range(n_chunks):
            c_start = i * chunk_size
            c_end = min((i + 1) * chunk_size, D)
            rc, ra, *_ = rasterization_2dgs(
                means=means3d,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=features[:, c_start:c_end],
                viewmats=viewmats[ci:ci+1],
                Ks=Ks[ci:ci+1],
                width=width,
                height=height,
                packed=False,
                near_plane=0.01,
                far_plane=1e5,
                render_mode='RGB',
            )
            cam_chunks.append(rc)
        all_results.append(torch.cat(cam_chunks, dim=-1))

    feature_maps = torch.cat(all_results, dim=0)  # [C, H, W, D]
    return feature_maps.permute(0, 3, 1, 2)  # [C, D, H, W]


# ═══════════════════════════════════════════════════════════════════════════
# SSIM Loss
# ═══════════════════════════════════════════════════════════════════════════

def _fspecial_gauss(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return g / g.sum()


def ssim_loss(img1, img2, window_size=11):
    C = img1.shape[1]
    window = _fspecial_gauss(window_size, 1.5).to(img1.device)
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = F.conv2d(img1**2, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=pad, groups=C) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return 1.0 - ssim_map.mean()


# ═══════════════════════════════════════════════════════════════════════════
# Main Training
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg):
    exp_name = cfg['exp_name']
    output_dir = os.path.join(cfg['output_dir'], exp_name)
    os.makedirs(output_dir, exist_ok=True)

    dcfg = cfg['dataset']
    mcfg = cfg['model']
    tcfg = cfg['training']
    gcfg = cfg['gsff']

    print(f"\n{'='*70}")
    print(f"  Joint 2DGS + GSFF Triplane + Raw RADIO Training")
    print(f"  Experiment: {exp_name}")
    print(f"{'='*70}")
    print(f"  Dataset:        {dcfg['source_dir']}")
    print(f"  RADIO features: {dcfg['radio_feature_dir']}")
    print(f"  GSFF dim:       {gcfg['feature_dim']}d, coarse={gcfg['coarse_resolution']}, fine={gcfg['fine_resolution']}")
    print(f"  Iterations:     {tcfg['iterations']}")
    print(f"  GSFF start:     {gcfg['gsff_start_iter']}")
    print(f"  Fine start:     {gcfg['fine_start_iter']}")
    print(f"  RADIO weight:   {gcfg['weight_radio']}")
    print(f"{'='*70}\n")

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── 1. Load scene ──
    print("Loading scene...")
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
        load_scene_colmap(dcfg['source_dir'], dcfg.get('images', ''))

    # ── 2. Load raw RADIO features ──
    print("\nLoading raw RADIO features...")
    radio_cache = RadioFeatureCache(
        dcfg['radio_feature_dir'],
        dcfg['radio_feature_scale'],
    )
    radio_dim = radio_cache.dim
    radio_h = radio_cache.height
    radio_w = radio_cache.width
    print(f"  RADIO: {radio_dim}d @ {radio_w}x{radio_h}")

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

    cam_to_fid = {}
    for cam in train_cams:
        fid = da3_name_to_fid.get(cam.image_name)
        if fid is not None and fid in radio_cache.frame_ids:
            cam_to_fid[cam.uid] = fid
    print(f"  Matched {len(cam_to_fid)}/{len(train_cams)} cameras to RADIO features")
    if len(cam_to_fid) == 0:
        print("ERROR: No cameras matched to RADIO features!")
        return

    # RADIO feature rendering intrinsics
    cam0 = train_cams[0]
    tanfovx = math.tan(cam0.FovX * 0.5)
    tanfovy = math.tan(cam0.FovY * 0.5)
    img_fx = cam0.width / (2 * tanfovx)
    img_fy = cam0.height / (2 * tanfovy)

    K_radio = torch.tensor([
        [img_fx * radio_w / cam0.width, 0, radio_w / 2.0],
        [0, img_fy * radio_h / cam0.height, radio_h / 2.0],
        [0, 0, 1],
    ], device="cuda", dtype=torch.float32)

    # ── 3. Create 2DGS model ──
    print("\nInitializing 2DGS model...")
    gaussians = GaussianModel2DGS(sh_degree=mcfg['sh_degree'])
    gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)

    train_args = argparse.Namespace(**{
        'position_lr_init': tcfg['position_lr_init'],
        'position_lr_final': tcfg['position_lr_final'],
        'feature_lr': tcfg['feature_lr'],
        'opacity_lr': tcfg['opacity_lr'],
        'scaling_lr': tcfg['scaling_lr'],
        'rotation_lr': tcfg['rotation_lr'],
        'percent_dense': tcfg['percent_dense'],
        'iterations': tcfg['iterations'],
    })
    gaussians.training_setup(train_args)

    bg_color = torch.tensor(
        [1, 1, 1] if mcfg['white_background'] else [0, 0, 0],
        dtype=torch.float32, device="cuda"
    )

    # ── 4. Create GSFF models ──
    print("\nInitializing GSFF models...")

    # Compute scene extent for triplane normalization
    xyz_init = gaussians.get_xyz.detach().cpu().numpy()
    norms = np.linalg.norm(xyz_init, axis=1)
    scene_extent_tri = float(np.percentile(norms, 99)) * 1.2
    print(f"  Scene extent (triplane): {scene_extent_tri:.2f}")

    triplane = DualScaleTriplane(
        coarse_resolution=gcfg['coarse_resolution'],
        fine_resolution=gcfg['fine_resolution'],
        feature_dim=gcfg['feature_dim'],
        scene_extent=scene_extent_tri,
    ).cuda()

    encoder = DualScaleEncoder(
        feature_dim=gcfg['feature_dim'],
        freeze_backbone=True,
    ).cuda()

    seg_head_coarse = SegmentationHead(
        feature_dim=gcfg['feature_dim'],
        num_classes=gcfg['n_clusters'],
    ).cuda()

    seg_head_fine = SegmentationHead(
        feature_dim=gcfg['feature_dim'],
        num_classes=gcfg['n_clusters'],
    ).cuda()

    radio_proj = RadioProjectionHead(
        input_dim=gcfg['feature_dim'],
        output_dim=radio_dim,
        hidden_dims=gcfg['radio_proj_hidden'],
    ).cuda()

    # Print param counts
    n_tri = sum(p.numel() for p in triplane.parameters())
    n_enc = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_seg = (sum(p.numel() for p in seg_head_coarse.parameters()) +
             sum(p.numel() for p in seg_head_fine.parameters()))
    n_proj = sum(p.numel() for p in radio_proj.parameters())
    print(f"  Triplane:      {n_tri:,} params")
    print(f"  Encoder:       {n_enc:,} trainable params")
    print(f"  Seg heads:     {n_seg:,} params")
    print(f"  Radio proj:    {n_proj:,} params")
    print(f"  Total GSFF:    {n_tri + n_enc + n_seg + n_proj:,}")

    # GSFF optimizer (separate from 2DGS geometry optimizer)
    gsff_param_groups = [
        {'params': triplane.parameters(), 'lr': float(gcfg['lr_triplane'])},
        {'params': encoder.coarse_encoder.proj.parameters(), 'lr': float(gcfg['lr_encoder'])},
        {'params': encoder.fine_encoder.parameters(), 'lr': float(gcfg['lr_encoder'])},
        {'params': seg_head_coarse.parameters(), 'lr': float(gcfg['lr_encoder'])},
        {'params': seg_head_fine.parameters(), 'lr': float(gcfg['lr_encoder'])},
        {'params': radio_proj.parameters(), 'lr': float(gcfg['lr_radio_proj'])},
    ]
    gsff_optimizer = torch.optim.Adam(gsff_param_groups, eps=1e-15)
    gsff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        gsff_optimizer,
        T_max=tcfg['iterations'] - gcfg['gsff_start_iter'],
        eta_min=1e-6,
    )

    # ── 5. Run spectral clustering ──
    print("\nRunning clustering...")
    t0 = time.time()
    cluster_labels = spectral_clustering(
        xyz_init, n_clusters=gcfg['n_clusters'],
        subsample=min(20000, len(xyz_init)),
    )
    cluster_labels_tensor = torch.from_numpy(cluster_labels).long().cuda()
    print(f"  Clustering: {gcfg['n_clusters']} clusters in {time.time()-t0:.1f}s")

    prototypes_coarse = None
    prototypes_fine = None

    def update_prototypes(scale='coarse'):
        with torch.no_grad():
            means3d = gaussians.get_xyz.detach()
            if scale == 'coarse':
                features = triplane.extract_coarse(means3d)
            else:
                features = triplane.extract_fine(means3d)
            features = F.normalize(features, p=2, dim=1)
        # Re-cluster if Gaussian count changed
        nonlocal cluster_labels_tensor
        if features.shape[0] != cluster_labels_tensor.shape[0]:
            xyz_np = means3d.cpu().numpy()
            labels = spectral_clustering(
                xyz_np, n_clusters=gcfg['n_clusters'],
                subsample=min(20000, len(xyz_np)),
            )
            cluster_labels_tensor = torch.from_numpy(labels).long().cuda()
        return compute_prototypes(
            features, cluster_labels_tensor, gcfg['n_clusters'],
        )

    # ── 6. Load masks, semantic masks, mono depth, appearance ──
    longest_edge = tcfg['longest_edge']

    masks = None
    if tcfg.get('use_mask', False):
        mask_candidates = [
            tcfg.get('mask_path', ''),
            os.path.join(dcfg['source_dir'], 'masks.pkl'),
        ]
        for mp in mask_candidates:
            if mp and os.path.exists(mp):
                print(f"  Loading masks from {mp}")
                with open(mp, 'rb') as f:
                    masks = pickle.load(f)
                matched = sum(1 for c in train_cams if c.image_name in masks)
                print(f"  Loaded masks for {len(masks)} images, matched {matched}/{len(train_cams)}")
                break

    sem_masks = {}
    sem_dir = tcfg.get('semantic_mask_dir', '')
    if sem_dir:
        sem_masks = load_semantic_masks(sem_dir, train_cams, cam_to_fid=cam_to_fid)

    appearance_net = None
    app_optimizer = None
    if tcfg.get('use_appearance', False):
        n_images = len(train_cams)
        appearance_net = AppearanceNetwork(
            n_images=n_images,
            embed_dim=tcfg.get('appearance_embed_dim', 32),
        ).cuda()
        app_optimizer = torch.optim.Adam(
            appearance_net.parameters(), lr=tcfg.get('appearance_lr', 0.001),
        )
        print(f"  AppearanceNetwork: {n_images} images")

    transient_head = None
    transient_optimizer = None
    if tcfg.get('use_transient', False):
        transient_head = TransientHead(in_channels=3).cuda()
        transient_optimizer = torch.optim.Adam(
            transient_head.parameters(), lr=tcfg.get('transient_lr', 0.001),
        )
        print("  TransientHead enabled")

    mono_depth_cache = {}
    mono_depth_dir = tcfg.get('mono_depth_dir', '')
    if mono_depth_dir and os.path.isdir(mono_depth_dir):
        print(f"Pre-caching monocular depth from {mono_depth_dir}...")
        for cam in train_cams:
            depth_name = os.path.splitext(cam.image_name)[0] + ".npy"
            depth_path = os.path.join(mono_depth_dir, depth_name)
            if os.path.exists(depth_path):
                d = np.load(depth_path)
                mono_depth_cache[cam.image_name] = 1.0 - d
        print(f"  Cached {len(mono_depth_cache)} depth maps")

    cam_uid_to_idx = {cam.uid: i for i, cam in enumerate(train_cams)}

    # ═══════════════════════════════════════════════════════════════════════
    # DINOv2 image normalization
    # ═══════════════════════════════════════════════════════════════════════
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    # ═══════════════════════════════════════════════════════════════════════
    # Compute coarse feature map size
    # ═══════════════════════════════════════════════════════════════════════
    # For encoder: we need to know render size for DINOv2 patch layout
    # Use the camera's native resolution (or longest_edge downscaled)
    if longest_edge > 0:
        scale_factor = longest_edge / max(cam0.width, cam0.height)
        render_w = int(cam0.width * scale_factor)
        render_h = int(cam0.height * scale_factor)
    else:
        render_w = cam0.width
        render_h = cam0.height

    coarse_h = render_h // 14
    coarse_w = render_w // 14
    K_coarse = torch.tensor([
        [img_fx * coarse_w / cam0.width, 0, coarse_w / 2.0],
        [0, img_fy * coarse_h / cam0.height, coarse_h / 2.0],
        [0, 0, 1],
    ], device="cuda", dtype=torch.float32)

    # Fine feature rendering at RADIO resolution (reuse RADIO K)
    fine_h, fine_w = radio_h, radio_w

    print(f"\n  Render:  {render_w}x{render_h}")
    print(f"  Coarse:  {coarse_w}x{coarse_h} (DINOv2 patches)")
    print(f"  Fine:    {fine_w}x{fine_h} (RADIO resolution)")
    print(f"  RADIO:   {radio_dim}d @ {radio_w}x{radio_h}")

    # ═══════════════════════════════════════════════════════════════════════
    # Pre-cache DINOv2 tokens (backbone is frozen, never changes)
    # ═══════════════════════════════════════════════════════════════════════
    print("\nPre-caching DINOv2 tokens for all training images...")
    dino_token_cache = {}  # uid → [1, N, 768] on CPU, fp16
    ps = encoder.coarse_encoder.patch_size
    dino_backbone = encoder.coarse_encoder.backbone

    with torch.no_grad():
        for ci, cam in enumerate(tqdm(train_cams, desc="DINOv2 cache")):
            gt_img = load_image_tensor(cam)
            gt_img = F.interpolate(
                gt_img.unsqueeze(0), size=(render_h, render_w),
                mode="bilinear", align_corners=False
            ).cuda()
            img_norm = (gt_img - IMAGENET_MEAN) / IMAGENET_STD
            # Pad for DINOv2
            pad_h = (ps - render_h % ps) % ps
            pad_w = (ps - render_w % ps) % ps
            if pad_h > 0 or pad_w > 0:
                img_norm = F.pad(img_norm, (0, pad_w, 0, pad_h), mode='reflect')
            feats = dino_backbone.forward_features(img_norm)
            tokens = feats['x_norm_patchtokens']  # [1, N, 768]
            dino_token_cache[cam.uid] = tokens.cpu().half()

    h_patches = (render_h + (ps - render_h % ps) % ps) // ps
    w_patches = (render_w + (ps - render_w % ps) % ps) // ps
    cache_mb = sum(t.numel() * 2 for t in dino_token_cache.values()) / 1024**2
    print(f"  Cached {len(dino_token_cache)} DINOv2 token sets "
          f"({h_patches}x{w_patches} patches, {cache_mb:.0f} MB CPU)")

    def encoder_with_cache(img_norm, cam_uid):
        """Run encoder using pre-cached DINOv2 tokens (skips backbone forward)."""
        tokens = dino_token_cache[cam_uid].cuda().float()  # [1, N, 768]
        # Coarse: projection only
        coarse_feat = encoder.coarse_encoder.proj(tokens)
        coarse_feat = coarse_feat.permute(0, 2, 1).reshape(
            1, encoder.coarse_encoder.feature_dim, h_patches, w_patches)
        # Fine: CNN + upsampled DINOv2
        fine_feat = encoder.fine_encoder(img_norm, tokens.detach(), h_patches, w_patches)
        return coarse_feat, fine_feat

    # ═══════════════════════════════════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════════════════════════════════
    iterations = tcfg['iterations']
    gsff_start = gcfg['gsff_start_iter']
    fine_start = gcfg['fine_start_iter']
    lambda_dssim = tcfg['lambda_dssim']
    lambda_normal = tcfg['lambda_normal']
    lambda_dist = tcfg['lambda_dist']
    lambda_scale = tcfg['lambda_scale']
    reg_start = tcfg['reg_start_iter']
    veg_reg_scale = tcfg.get('vegetation_reg_scale', 0.1)
    grad_accum_steps = max(1, tcfg.get('grad_accum_steps', 1))

    viewpoint_stack = []
    ema_loss = 0.0
    ema_rgb = 0.0
    ema_depth = 0.0
    ema_nce = 0.0
    ema_radio = 0.0
    ema_cos_c = 0.0
    ema_cos_f = 0.0
    best_psnr = 0.0

    log_file = open(os.path.join(output_dir, 'train.log'), 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"  Grad accumulation: {grad_accum_steps} cameras per step")

    pbar = tqdm(range(1, iterations + 1), desc="Joint GSFF+2DGS")
    for iteration in pbar:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        accum_loss_val = 0.0
        accum_rgb_val = 0.0
        bad_step = False

        for accum_step in range(grad_accum_steps):
            # Sample camera
            if not viewpoint_stack:
                viewpoint_stack = list(train_cams)
                shuffle(viewpoint_stack)
            cam = viewpoint_stack.pop()
            fid = cam_to_fid.get(cam.uid)
            cam_idx = cam_uid_to_idx[cam.uid]

            # ═══ RGB rendering ═══
            render_pkg = render_rgb_2dgs(gaussians, cam, bg_color, longest_edge)
            image = render_pkg["render"]  # [3, H, W]
            rw, rh = render_pkg["width"], render_pkg["height"]

            if appearance_net is not None:
                image = appearance_net(image, cam_idx)

            gt_image = load_image_tensor(cam)
            gt_image = F.interpolate(
                gt_image.unsqueeze(0), size=(rh, rw),
                mode="bilinear", align_corners=False
            ).squeeze(0)

            # ═══ Sky / object masking ═══
            rgb_mask = None
            sky_mask_2d = None
            if masks is not None and cam.image_name in masks:
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
                rgb_mask = (obj_mask & distort_mask).float()
                sky_mask_2d = ~sky_mask_raw
                image = image * rgb_mask + bg_color[:, None, None] * (1.0 - rgb_mask)
                gt_image = gt_image * rgb_mask + bg_color[:, None, None] * (1.0 - rgb_mask)

            # ═══ Transient confidence ═══
            transient_weight = None
            if transient_head is not None:
                transient_weight = transient_head(render_pkg["render"].detach())

            # ═══ RGB loss ═══
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

            # ═══ Transient regularization ═══
            if transient_weight is not None and tcfg.get('transient_reg', 0) > 0:
                t_reg = ((transient_weight - 0.5) ** 2).mean()
                loss = loss + tcfg['transient_reg'] * t_reg

            # ═══ Semantic-aware geometry regularization ═══
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
                sem_weight[0, is_dynamic] = 0.0
                if is_dynamic.any():
                    dynamic_mask_2d = (~is_dynamic).float()[None]

            # Dynamic mask on RGB
            if dynamic_mask_2d is not None:
                pixel_l1 = (image - gt_image).abs().mean(dim=0, keepdim=True)
                if transient_weight is not None:
                    pixel_l1 = pixel_l1 * transient_weight
                n_valid = dynamic_mask_2d.sum().clamp(min=1)
                masked_l1 = (pixel_l1 * dynamic_mask_2d).sum() / n_valid
                rgb_loss_masked = (1.0 - lambda_dssim) * masked_l1 + lambda_dssim * (1.0 - ssim_val)
                loss = loss - rgb_loss + rgb_loss_masked
                rgb_loss = rgb_loss_masked

            # ═══ 2DGS geometry regularization ═══
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
                    if sem_weight is not None:
                        normal_error = normal_error * sem_weight
                    loss = loss + lambda_normal * normal_error.mean()

                if lambda_dist > 0 and rend_dist is not None:
                    dist_loss = rend_dist.squeeze(-1)
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
                    veg_scale_boost = tcfg.get('vegetation_scale_boost', 1.0)
                    if veg_scale_boost > 1.0 and sem_masks and cam.image_name in sem_masks:
                        scale_sem_weight = torch.ones(1, rh, rw, device='cuda')
                        if sem_weight is not None:
                            is_veg_px = (sem_weight < 1.0)
                            scale_sem_weight[is_veg_px] = veg_scale_boost
                        veg_frac = (scale_sem_weight > 1.0).float().mean()
                        effective_boost = 1.0 + (veg_scale_boost - 1.0) * veg_frac.item()
                        loss = loss + lambda_scale * effective_boost * (excess ** 2).mean()
                    else:
                        loss = loss + lambda_scale * (excess ** 2).mean()

            # ═══ Monocular depth supervision ═══
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
                    if tcfg.get('depth_exclude_vegetation', False) and sem_masks and cam.image_name in sem_masks:
                        sem_for_depth = F.interpolate(
                            sem_masks[cam.image_name].float()[None, None], size=(rh, rw), mode='nearest'
                        ).squeeze().long()
                        veg_mask = (sem_for_depth == 2)
                        if depth_valid_mask is not None:
                            depth_valid_mask = depth_valid_mask & (~veg_mask)
                        else:
                            depth_valid_mask = ~veg_mask
                    loss = loss + lambda_depth_now * pearson_depth_loss(
                        rendered_depth, mono_d, valid_mask=depth_valid_mask
                    )

            # ═══ GSFF losses (Phase 2 only) ═══
            gsff_loss = torch.tensor(0.0, device='cuda')
            gsff_loss_dict = {}

            if iteration >= gsff_start and fid is not None:
                means3d = gaussians.get_xyz
                quats = gaussians.get_rotation
                scales_2d = gaussians.get_scaling
                opacities = gaussians.get_opacity.squeeze(-1)
                scales_3d = torch.cat([scales_2d, torch.ones_like(scales_2d[:, :1])], dim=-1)

                # ── Prepare image for encoder ──
                gt_img_for_enc = load_image_tensor(cam)
                gt_img_for_enc = F.interpolate(
                    gt_img_for_enc.unsqueeze(0), size=(rh, rw),
                    mode="bilinear", align_corners=False
                )  # [1, 3, rh, rw]
                img_norm = (gt_img_for_enc - IMAGENET_MEAN) / IMAGENET_STD

                # ── 2D encoder features (using pre-cached DINOv2 tokens) ──
                coarse_feat_2d, fine_feat_2d = encoder_with_cache(img_norm, cam.uid)
                # coarse: [1, D, rh/14, rw/14], fine: [1, D, rh, rw]

                # ── Extract triplane features ──
                coarse_colors = triplane.extract_coarse(means3d.detach())  # [N, D]
                coarse_colors_norm = F.normalize(coarse_colors, p=2, dim=1)

                # ── Render coarse 3D features ──
                viewmat = cam.get_world_view_transform().unsqueeze(0)  # [1, 4, 4]
                coarse_feat_3d = render_features_through_gaussians(
                    means3d.detach(), quats.detach(), scales_3d.detach(),
                    opacities.detach(), coarse_colors_norm,
                    viewmat, K_coarse.unsqueeze(0),
                    coarse_w, coarse_h,
                )  # [1, D, coarse_h, coarse_w]

                coarse_feat_2d_norm = F.normalize(coarse_feat_2d, p=2, dim=1)
                coarse_feat_3d_norm = F.normalize(coarse_feat_3d, p=2, dim=1)

                if coarse_feat_2d_norm.shape[2:] != coarse_feat_3d_norm.shape[2:]:
                    coarse_feat_2d_norm = F.interpolate(
                        coarse_feat_2d_norm, coarse_feat_3d_norm.shape[2:],
                        mode='bilinear', align_corners=False,
                    )

                # ── Coarse NCE ──
                l_nce = info_nce_loss(
                    coarse_feat_3d_norm, coarse_feat_2d_norm,
                    temperature=gcfg['temperature'],
                    max_samples=gcfg['nce_samples'],
                )
                gsff_loss = gsff_loss + gcfg['weight_nce'] * l_nce
                gsff_loss_dict['nce'] = l_nce.item()

                # Direct cosine alignment
                cos_c = (coarse_feat_2d_norm * coarse_feat_3d_norm).sum(dim=1).mean()
                gsff_loss_dict['cos_c'] = cos_c.item()
                if gcfg['cosine_loss_weight'] > 0:
                    gsff_loss = gsff_loss + gcfg['cosine_loss_weight'] * (1.0 - cos_c)

                # ── Prototypical + Segmentation (coarse) ──
                if prototypes_coarse is not None:
                    l_pro = prototypical_loss(
                        coarse_feat_3d_norm, coarse_feat_2d_norm,
                        prototypes_coarse, temperature=gcfg['temperature'],
                        max_samples=gcfg['nce_samples'],
                    )
                    gsff_loss = gsff_loss + gcfg['weight_pro'] * l_pro
                    gsff_loss_dict['pro'] = l_pro.item()

                    proto_labels = get_pixel_proto_labels(coarse_feat_3d_norm, prototypes_coarse)
                    logits_3d = compute_pseudo_logits(coarse_feat_3d_norm, prototypes_coarse)
                    seg_2d = seg_head_coarse(coarse_feat_2d_norm)
                    l_ce = segmentation_ce_loss(seg_2d, logits_3d, proto_labels)
                    gsff_loss = gsff_loss + gcfg['weight_ce'] * l_ce
                    gsff_loss_dict['ce'] = l_ce.item()

                # ── TV regularization ──
                l_tvl = triplane.total_variation_loss()
                gsff_loss = gsff_loss + gcfg['weight_tvl'] * l_tvl
                gsff_loss_dict['tvl'] = l_tvl.item()

                # ═══ Fine level + RADIO supervision ═══
                if iteration >= fine_start:
                    fine_colors = triplane.extract_fine(means3d.detach())
                    fine_colors_norm = F.normalize(fine_colors, p=2, dim=1)

                    # Render fine features at RADIO resolution
                    fine_feat_3d = render_features_through_gaussians(
                        means3d.detach(), quats.detach(), scales_3d.detach(),
                        opacities.detach(), fine_colors_norm,
                        viewmat, K_radio.unsqueeze(0),
                        fine_w, fine_h,
                    )  # [1, D, fine_h, fine_w]

                    fine_feat_2d_resized = F.interpolate(
                        fine_feat_2d, (fine_h, fine_w),
                        mode='bilinear', align_corners=False,
                    )
                    fine_feat_2d_norm = F.normalize(fine_feat_2d_resized, p=2, dim=1)
                    fine_feat_3d_norm = F.normalize(fine_feat_3d, p=2, dim=1)

                    # Fine NCE
                    l_nce_fine = info_nce_loss(
                        fine_feat_3d_norm, fine_feat_2d_norm,
                        temperature=gcfg['temperature'],
                        max_samples=gcfg['nce_samples'],
                    )
                    gsff_loss = gsff_loss + gcfg['weight_nce'] * l_nce_fine
                    gsff_loss_dict['nce_f'] = l_nce_fine.item()

                    cos_f = (fine_feat_2d_norm * fine_feat_3d_norm).sum(dim=1).mean()
                    gsff_loss_dict['cos_f'] = cos_f.item()
                    if gcfg['cosine_loss_weight'] > 0:
                        gsff_loss = gsff_loss + gcfg['cosine_loss_weight'] * (1.0 - cos_f)

                    # Fine prototypical + segmentation
                    if prototypes_fine is not None:
                        l_pro_f = prototypical_loss(
                            fine_feat_3d_norm, fine_feat_2d_norm,
                            prototypes_fine, temperature=gcfg['temperature'],
                            max_samples=gcfg['nce_samples'],
                        )
                        gsff_loss = gsff_loss + gcfg['weight_pro'] * l_pro_f
                        gsff_loss_dict['pro_f'] = l_pro_f.item()

                        fine_proto_labels = get_pixel_proto_labels(fine_feat_3d_norm, prototypes_fine)
                        fine_logits_3d = compute_pseudo_logits(fine_feat_3d_norm, prototypes_fine)
                        fine_seg_2d = seg_head_fine(fine_feat_2d_norm)
                        l_ce_f = segmentation_ce_loss(fine_seg_2d, fine_logits_3d, fine_proto_labels)
                        gsff_loss = gsff_loss + gcfg['weight_ce'] * l_ce_f
                        gsff_loss_dict['ce_f'] = l_ce_f.item()

                    # ═══ RADIO supervision: projection head + L1 + cosine ═══
                    gt_radio = radio_cache.get(fid)
                    if gt_radio is not None:
                        # Project triplane features to RADIO space
                        proj_feat = radio_proj(fine_feat_3d)  # [1, 1280, fine_h, fine_w]

                        # Ensure GT RADIO matches spatial resolution
                        gt_radio_4d = gt_radio.unsqueeze(0)  # [1, 1280, H_r, W_r]
                        if gt_radio_4d.shape[2:] != proj_feat.shape[2:]:
                            gt_radio_4d = F.interpolate(
                                gt_radio_4d, proj_feat.shape[2:],
                                mode='bilinear', align_corners=False,
                            )

                        # L1 loss on projected features
                        radio_l1 = (proj_feat - gt_radio_4d).abs().mean()
                        # Cosine loss
                        proj_norm = F.normalize(proj_feat, p=2, dim=1)
                        gt_norm = F.normalize(gt_radio_4d, p=2, dim=1)
                        radio_cos = (proj_norm * gt_norm).sum(dim=1).mean()
                        radio_loss = radio_l1 + 0.5 * (1.0 - radio_cos)

                        gsff_loss = gsff_loss + gcfg['weight_radio'] * radio_loss
                        gsff_loss_dict['radio_l1'] = radio_l1.item()
                        gsff_loss_dict['radio_cos'] = radio_cos.item()

                loss = loss + gsff_loss

            # ═══ Appearance regularization ═══
            if accum_step == grad_accum_steps - 1 and appearance_net is not None:
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
                        loss = loss + app_reg_w * (((reg_scale - 1.0) ** 2).mean() + (reg_bias ** 2).mean())
                    if app_mean_reg > 0:
                        mean_loss = ((reg_scale.mean(0) - 1.0) ** 2).sum() + (reg_bias.mean(0) ** 2).sum()
                        loss = loss + app_mean_reg * mean_loss

            # ═══ Safety check ═══
            loss_val = loss.item()
            if torch.isnan(loss) or torch.isinf(loss) or loss_val < -0.01:
                tqdm.write(f"  [Iter {iteration}] Bad loss={loss_val:.4g}, skip")
                bad_step = True
                break
            if loss_val > 10.0:
                loss = loss.clamp(max=10.0)

            # ═══ Backward ═══
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            # Densification stats
            with torch.no_grad():
                if iteration < tcfg['densify_until_iter']:
                    vp = render_pkg["viewspace_points"]
                    grad_data = vp.grad if vp.grad is not None else vp
                    radii = render_pkg["radii"]
                    vis = render_pkg["visibility_filter"]
                    gaussians.max_radii2D[vis] = torch.max(
                        gaussians.max_radii2D[vis], radii[vis]
                    )
                    gaussians.add_densification_stats(grad_data, vis, rw, rh)

            accum_loss_val += loss_val / grad_accum_steps
            accum_rgb_val += rgb_loss_raw.item() / grad_accum_steps

        # ── End accumulation ──
        if bad_step:
            gaussians.optimizer.zero_grad(set_to_none=True)
            gsff_optimizer.zero_grad(set_to_none=True)
            if app_optimizer:
                app_optimizer.zero_grad(set_to_none=True)
            if transient_optimizer:
                transient_optimizer.zero_grad(set_to_none=True)
            continue

        # ── Gradient clipping ──
        gs_params = [
            gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
            gaussians._scaling, gaussians._rotation, gaussians._opacity,
        ]
        torch.nn.utils.clip_grad_norm_(gs_params, max_norm=1.0)

        gsff_params = (
            list(triplane.parameters()) +
            list(encoder.parameters()) +
            list(seg_head_coarse.parameters()) +
            list(seg_head_fine.parameters()) +
            list(radio_proj.parameters())
        )
        torch.nn.utils.clip_grad_norm_(gsff_params, max_norm=1.0)

        # ── EMA ──
        ema_loss = 0.4 * accum_loss_val + 0.6 * ema_loss
        ema_rgb = 0.4 * accum_rgb_val + 0.6 * ema_rgb
        if 'nce' in gsff_loss_dict:
            ema_nce = 0.1 * gsff_loss_dict['nce'] + 0.9 * ema_nce
        if 'radio_cos' in gsff_loss_dict:
            ema_radio = 0.1 * gsff_loss_dict.get('radio_cos', 0) + 0.9 * ema_radio
        if 'cos_c' in gsff_loss_dict:
            ema_cos_c = 0.1 * gsff_loss_dict['cos_c'] + 0.9 * ema_cos_c
        if 'cos_f' in gsff_loss_dict:
            ema_cos_f = 0.1 * gsff_loss_dict['cos_f'] + 0.9 * ema_cos_f

        with torch.no_grad():
            if iteration % 10 == 0:
                pbar.set_postfix({
                    "L": f"{ema_loss:.4f}", "RGB": f"{ema_rgb:.4f}",
                    "NCE": f"{ema_nce:.3f}", "cC": f"{ema_cos_c:.3f}",
                    "cF": f"{ema_cos_f:.3f}", "rCos": f"{ema_radio:.3f}",
                    "N": f"{gaussians.num_points:,}",
                })

            # ── Update prototypes ──
            if (iteration >= gsff_start and iteration > 0 and
                    iteration % gcfg['prototype_update_freq'] == 0):
                prototypes_coarse = update_prototypes('coarse')
                if iteration >= fine_start:
                    prototypes_fine = update_prototypes('fine')

            # ── Periodic status log (every 500 iters) ──
            if iteration % 500 == 0:
                phase = "Phase1-Geo" if iteration < gsff_start else (
                    "Phase2-Coarse" if iteration < fine_start else "Phase2-Full")
                mem_mb = torch.cuda.max_memory_allocated() / 1024**2
                log(f"  [Iter {iteration}] {phase} | L={ema_loss:.4f} RGB={ema_rgb:.4f} "
                    f"NCE={ema_nce:.3f} cos_c={ema_cos_c:.3f} cos_f={ema_cos_f:.3f} "
                    f"rCos={ema_radio:.3f} N={gaussians.num_points:,} GPU={mem_mb:.0f}MB")

            # ── Eval + save ──
            eval_interval = tcfg['eval_interval']
            if iteration % eval_interval == 0 or iteration == iterations:
                eval_cams = test_cams if test_cams else train_cams
                psnr_val = evaluate_psnr(gaussians, eval_cams, bg_color, longest_edge)
                msg = (f"  [Iter {iteration}] PSNR={psnr_val:.2f}dB | "
                       f"RGB={ema_rgb:.4f} | NCE={ema_nce:.3f} | "
                       f"cos_c={ema_cos_c:.3f} cos_f={ema_cos_f:.3f} | "
                       f"radio_cos={ema_radio:.3f} | N={gaussians.num_points:,}")
                if psnr_val > best_psnr:
                    best_psnr = psnr_val
                    msg += " ★ BEST"
                    # Save best geometry
                    best_dir = os.path.join(output_dir, "point_cloud", "best")
                    os.makedirs(best_dir, exist_ok=True)
                    gaussians.save_ply(os.path.join(best_dir, "point_cloud.ply"))
                    # Save best GSFF
                    best_gsff = os.path.join(output_dir, "checkpoints", "best_gsff.pth")
                    os.makedirs(os.path.dirname(best_gsff), exist_ok=True)
                    torch.save({
                        'iteration': iteration,
                        'psnr': psnr_val,
                        'triplane': triplane.state_dict(),
                        'encoder': encoder.state_dict(),
                        'seg_head_coarse': seg_head_coarse.state_dict(),
                        'seg_head_fine': seg_head_fine.state_dict(),
                        'radio_proj': radio_proj.state_dict(),
                        'scene_extent': scene_extent_tri,
                    }, best_gsff)
                log(msg)

            save_interval = tcfg['save_interval']
            if iteration % save_interval == 0 or iteration == iterations:
                save_dir = os.path.join(output_dir, "point_cloud", f"iteration_{iteration}")
                os.makedirs(save_dir, exist_ok=True)
                gaussians.save_ply(os.path.join(save_dir, "point_cloud.ply"))
                # Save GSFF checkpoint
                ckpt_path = os.path.join(output_dir, "checkpoints", f"gsff_iter_{iteration:06d}.pth")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    'iteration': iteration,
                    'triplane': triplane.state_dict(),
                    'encoder': encoder.state_dict(),
                    'seg_head_coarse': seg_head_coarse.state_dict(),
                    'seg_head_fine': seg_head_fine.state_dict(),
                    'radio_proj': radio_proj.state_dict(),
                    'gsff_optimizer': gsff_optimizer.state_dict(),
                    'prototypes_coarse': prototypes_coarse.cpu() if prototypes_coarse is not None else None,
                    'prototypes_fine': prototypes_fine.cpu() if prototypes_fine is not None else None,
                    'scene_extent': scene_extent_tri,
                }, ckpt_path)
                # Latest alias
                torch.save(torch.load(ckpt_path, weights_only=False),
                           os.path.join(output_dir, "checkpoints", "latest_gsff.pth"))

            # ── Densification ──
            if iteration < tcfg['densify_until_iter']:
                if (iteration > tcfg['densify_from_iter'] and
                        iteration % tcfg['densification_interval'] == 0):
                    gaussians.densify_and_prune(
                        tcfg['densify_grad_threshold'], 0.005,
                        cameras_extent, None,
                    )

                # Vegetation pruning
                veg_prune_opacity = tcfg.get('veg_prune_opacity', 0.0)
                veg_prune_scale_pct = tcfg.get('veg_prune_scale_percentile', 0)
                if ((veg_prune_opacity > 0 or veg_prune_scale_pct > 0)
                        and sem_masks
                        and iteration % tcfg['opacity_reset_interval'] == 0):
                    n_opa, n_scl, n_clamp, n_veg = prune_vegetation_gaussians(
                        gaussians, train_cams, sem_masks,
                        veg_opacity_threshold=veg_prune_opacity,
                        veg_scale_percentile=veg_prune_scale_pct,
                        scene_extent=cameras_extent,
                        n_sample_cams=tcfg.get('veg_sample_cams', 50),
                        veg_vote_threshold=tcfg.get('veg_vote_threshold', 0.3),
                        log_fn=log,
                    )
                    log(f"  [Iter {iteration}] Veg: {n_veg} id, prune {n_opa}+{n_scl}, clamp {n_clamp}")

                if iteration % tcfg['opacity_reset_interval'] == 0:
                    gaussians.reset_opacity(reset_value=tcfg['opacity_reset_value'])

            # ── Optimizer steps ──
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        # GSFF optimizer step
        if iteration >= gsff_start:
            gsff_optimizer.step()
            gsff_scheduler.step()
        gsff_optimizer.zero_grad(set_to_none=True)

        if app_optimizer:
            app_optimizer.step()
            app_optimizer.zero_grad(set_to_none=True)
        if transient_optimizer:
            transient_optimizer.step()
            transient_optimizer.zero_grad(set_to_none=True)

        # ── Periodic logging ──
        if iteration % 1000 == 1:
            gsff_str = " | ".join(f"{k}={v:.4f}" for k, v in gsff_loss_dict.items())
            log(f"  [Iter {iteration}] RGB={ema_rgb:.4f} GSFF=[{gsff_str}] "
                f"Total={ema_loss:.4f} N={gaussians.num_points:,}")

    # ── Final ──
    log(f"\n  Training complete. Best PSNR: {best_psnr:.2f}dB")
    log(f"  Output: {output_dir}")
    log_file.close()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Joint 2DGS + GSFF Triplane + Raw RADIO Training')
    parser.add_argument('--config', required=True, help='YAML config path')
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
