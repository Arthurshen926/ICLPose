#!/usr/bin/env python3
"""
GSFFs Training Script for OldHospital (Cambridge Landmarks).

Reproduces the GSFFs paper: self-supervised contrastive learning of
triplane feature fields + 2D encoder for 6-DOF pose refinement.

Training schedule (50K iterations on single GPU):
  - Phase 1 (0-15K): Photometric loss only (L_PHO) — train Gaussian RGB
  - Phase 2 (15K-50K): Joint loss:
      L = L_PHO + 0.5*L_NCE + 0.5*L_PRO + 0.5*L_CE + 0.1*L_TVL

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/train_gsff.py \\
        --source_dir dataset/OldHospital \\
        --model_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \\
        --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json \\
        --output_dir output/gsff/OldHospital
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsplat import rasterization_2dgs, spherical_harmonics
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel

from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder, SegmentationHead
from gsff.losses import (
    info_nce_loss,
    prototypical_loss,
    segmentation_ce_loss,
    compute_pseudo_logits,
    get_pixel_proto_labels,
)
from gsff.clustering import spectral_clustering, compute_prototypes


# ── Dataset ─────────────────────────────────────────────────────────────────

class CambridgeDataset(Dataset):
    """Cambridge Landmarks dataset loader (train or test split)."""

    # ImageNet normalization for DINOv2
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, source_dir, cameras_json, split='train', target_size=None):
        self.source_dir = Path(source_dir)
        self.split = split
        self.target_size = target_size  # (H, W) or None for original

        # Load cameras
        with open(cameras_json) as f:
            all_cams = json.load(f)
        self.cam_by_name = {c['img_name']: c for c in all_cams}

        # Load split list
        split_file = self.source_dir / f'dataset_{split}.txt'
        self.samples = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('Visual') or line.startswith('ImageFile'):
                    continue
                parts = line.split()
                img_name = parts[0]
                # Cambridge pose format: X Y Z W P Q R (position + quaternion WPQR)
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                w, p, q, r = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                self.samples.append({
                    'img_name': img_name,
                    'position': np.array([x, y, z], dtype=np.float32),
                    'quat_wpqr': np.array([w, p, q, r], dtype=np.float32),
                })

        # Get shared intrinsics from first camera
        first_cam = all_cams[0]
        self.fx = first_cam['fx']
        self.fy = first_cam['fy']
        self.width = first_cam['width']
        self.height = first_cam['height']
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        print(f"[CambridgeDataset] {split}: {len(self.samples)} images, "
              f"{self.width}x{self.height}, fx={self.fx:.1f}")

    def __len__(self):
        return len(self.samples)

    def _quat_to_rotmat(self, q):
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float32)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = sample['img_name']

        # Load image
        img_path = self.source_dir / img_name
        if not img_path.exists():
            # Try processed subdir
            img_path = self.source_dir / 'processed' / img_name
        img = Image.open(img_path).convert('RGB')

        if self.target_size is not None:
            img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)

        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0

        # Normalize for DINOv2
        img_norm = (img_tensor - self.MEAN) / self.STD

        # Build c2w from Cambridge pose (position + quaternion)
        quat = sample['quat_wpqr']
        R_c2w = self._quat_to_rotmat(quat)
        t_c2w = sample['position']

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w
        c2w[:3, 3] = t_c2w

        # Use COLMAP-derived poses from cameras.json if available (more accurate)
        if img_name in self.cam_by_name:
            cam = self.cam_by_name[img_name]
            R_c2w_col = np.array(cam['rotation'], dtype=np.float32)
            t_c2w_col = np.array(cam['position'], dtype=np.float32)
            c2w[:3, :3] = R_c2w_col
            c2w[:3, 3] = t_c2w_col

        # w2c = inv(c2w)
        w2c = np.linalg.inv(c2w).astype(np.float32)

        return {
            'image': img_tensor,       # [3, H, W] in [0, 1]
            'image_norm': img_norm,    # [3, H, W] ImageNet normalized
            'w2c': torch.from_numpy(w2c),  # [4, 4]
            'c2w': torch.from_numpy(c2w),  # [4, 4]
            'img_name': img_name,
        }


# ── Rendering Utilities ─────────────────────────────────────────────────────

def render_rgb_2dgs(gaussian_model, viewmats, Ks, width, height):
    """Render RGB images from 2DGS model via gsplat."""
    means3d = gaussian_model.get_xyz
    quats = gaussian_model.get_rotation
    scales_2d = gaussian_model.get_scaling  # [N, 2]
    opacities = gaussian_model.get_opacity.squeeze(-1)

    # Pad scales for 2DGS: [N, 2] -> [N, 3]
    scales = torch.cat([scales_2d, torch.ones_like(scales_2d[:, :1])], dim=-1)

    # SH to RGB (degree 0)
    sh0 = gaussian_model._features_dc.reshape(-1, 1, 3)  # [N, 1, 3]
    colors = spherical_harmonics(0, means3d, viewmats, sh0)  # [C, N, 3]
    colors = torch.clamp(colors + 0.5, 0.0, 1.0)

    C = viewmats.shape[0]
    render_colors_list = []
    for c in range(C):
        rc, ra, *_ = rasterization_2dgs(
            means=means3d,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors[c],  # [N, 3]
            viewmats=viewmats[c:c+1],
            Ks=Ks[c:c+1],
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e5,
            render_mode='RGB',
        )
        render_colors_list.append(rc)

    render_colors = torch.cat(render_colors_list, dim=0)  # [C, H, W, 3]
    return render_colors.permute(0, 3, 1, 2)  # [C, 3, H, W]


def render_features_2dgs(means3d, quats, scales, opacities, features,
                         viewmats, Ks, width, height, chunk_size=16):
    """
    Render feature maps by alpha-blending triplane features through gsplat.

    Args:
        features: [N, D] per-Gaussian features (from triplane)
        viewmats: [C, 4, 4]
        Returns: [C, D, H, W]
    """
    D = features.shape[1]
    C = viewmats.shape[0]
    n_chunks = (D + chunk_size - 1) // chunk_size

    all_chunks = []
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
        all_chunks.append(torch.cat(cam_chunks, dim=-1))

    feature_maps = torch.cat(all_chunks, dim=0)  # [C, H, W, D]
    return feature_maps.permute(0, 3, 1, 2)  # [C, D, H, W]


# ── SSIM Loss ────────────────────────────────────────────────────────────────

def _fspecial_gauss(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return g / g.sum()


def ssim_loss(img1, img2, window_size=11):
    """Compute 1 - SSIM between [B, C, H, W] images."""
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


# ── Training ─────────────────────────────────────────────────────────────────

class GSFFTrainer:
    """GSFFs training pipeline following the paper."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda')
        self.setup_output_dir()
        self.load_gaussian_model()
        self.build_models()
        self.build_data()
        self.setup_optimizer()
        self.start_iteration = 0
        if args.warmstart:
            self.load_warmstart(args.warmstart)
        if args.resume:
            self.load_resume(args.resume)

    def load_resume(self, path):
        """Resume training from a full checkpoint (model + optimizer + scheduler)."""
        print(f"Resuming from: {path}")
        ckpt = torch.load(path, map_location='cpu')
        self.triplane.load_state_dict(ckpt['triplane'])
        self.encoder.load_state_dict(ckpt['encoder'])
        if 'seg_head_coarse' in ckpt:
            self.seg_head_coarse.load_state_dict(ckpt['seg_head_coarse'])
        if 'seg_head_fine' in ckpt:
            self.seg_head_fine.load_state_dict(ckpt['seg_head_fine'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_iteration = ckpt.get('iteration', 0)
        print(f"  Resumed at iteration {self.start_iteration}")

    def load_warmstart(self, path):
        """Load triplane and coarse encoder weights from a previous checkpoint."""
        print(f"Warmstarting from: {path}")
        ckpt = torch.load(path, map_location='cpu')
        # Load triplane (coarse planes only; fine will be retrained)
        tri_state = ckpt['triplane']
        current_tri = self.triplane.state_dict()
        loaded_keys = []
        for k, v in tri_state.items():
            if k in current_tri and current_tri[k].shape == v.shape:
                current_tri[k] = v
                loaded_keys.append(k)
        self.triplane.load_state_dict(current_tri)
        print(f"  Triplane: loaded {len(loaded_keys)}/{len(current_tri)} keys: {loaded_keys}")
        # Load coarse encoder projection weights
        enc_state = ckpt['encoder']
        current_enc = self.encoder.state_dict()
        loaded_enc = []
        for k, v in enc_state.items():
            if k in current_enc and current_enc[k].shape == v.shape:
                current_enc[k] = v
                loaded_enc.append(k)
        self.encoder.load_state_dict(current_enc)
        print(f"  Encoder: loaded {len(loaded_enc)}/{len(current_enc)} compatible keys")
        # Load coarse seg head if shapes match
        if 'seg_head_coarse' in ckpt:
            try:
                self.seg_head_coarse.load_state_dict(ckpt['seg_head_coarse'])
                print("  Coarse seg head: loaded")
            except Exception:
                print("  Coarse seg head: shape mismatch, skipped")

    def setup_output_dir(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, 'checkpoints'), exist_ok=True)

    def load_gaussian_model(self):
        """Load pre-trained 2DGS model (frozen geometry)."""
        print("Loading 2DGS model...")
        self.gs_model = GaussianFeatureModel(feature_dim=16)
        self.gs_model.load_ply(self.args.model_path)
        self.gs_model = self.gs_model.to(self.device)
        self.gs_model.eval()

        # Freeze all Gaussian parameters
        for p in self.gs_model.parameters():
            p.requires_grad = False

        self.N_gaussians = self.gs_model.num_gaussians
        self.means3d = self.gs_model.get_xyz.detach()
        self.quats = self.gs_model.get_rotation.detach()

        scales_raw = self.gs_model.get_scaling  # [N, 2] for 2DGS
        self.scales = torch.cat([
            scales_raw,
            torch.ones_like(scales_raw[:, :1])
        ], dim=-1).detach()  # [N, 3]

        self.opacities = self.gs_model.get_opacity.squeeze(-1).detach()

        # Compute scene extent for triplane normalization (robust to outliers)
        xyz_np = self.means3d.cpu().numpy()
        norms = np.linalg.norm(xyz_np, axis=1)
        # Use 99th percentile to avoid floater Gaussians
        self.scene_extent = float(np.percentile(norms, 99)) * 1.2
        print(f"  Scene extent: {self.scene_extent:.2f} (99th pctile of norms)")
        print(f"  N Gaussians: {self.N_gaussians:,}")

    def build_models(self):
        """Initialize triplane, encoder, and segmentation head."""
        args = self.args

        # Triplane feature field
        self.triplane = DualScaleTriplane(
            coarse_resolution=args.coarse_resolution,
            fine_resolution=args.fine_resolution,
            feature_dim=args.feature_dim,
            scene_extent=self.scene_extent,
        ).to(self.device)

        # 2D encoder
        self.encoder = DualScaleEncoder(
            feature_dim=args.feature_dim,
            freeze_backbone=True,
            use_old_fine_encoder=getattr(args, 'use_old_fine_encoder', False),
        ).to(self.device)

        # Segmentation head
        self.seg_head_coarse = SegmentationHead(
            feature_dim=args.feature_dim,
            num_classes=args.n_clusters,
        ).to(self.device)

        self.seg_head_fine = SegmentationHead(
            feature_dim=args.feature_dim,
            num_classes=args.n_clusters,
        ).to(self.device)

        # Print parameter counts
        n_tri = sum(p.numel() for p in self.triplane.parameters())
        n_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        n_seg = (sum(p.numel() for p in self.seg_head_coarse.parameters()) +
                 sum(p.numel() for p in self.seg_head_fine.parameters()))
        print(f"  Triplane params: {n_tri:,}")
        print(f"  Encoder trainable params: {n_enc:,}")
        print(f"  Segmentation head params: {n_seg:,}")

    def build_data(self):
        """Setup dataset and dataloader."""
        args = self.args

        # Determine render size
        self.render_h = args.render_height
        self.render_w = args.render_width

        target_size = (self.render_h, self.render_w) if (self.render_h != 1080 or self.render_w != 1920) else None

        self.train_dataset = CambridgeDataset(
            args.source_dir, args.cameras_json,
            split='train', target_size=target_size,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,  # Single image per iteration (like 3DGS training)
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        self.fx_render = self.train_dataset.fx * self.render_w / self.train_dataset.width
        self.fy_render = self.train_dataset.fy * self.render_h / self.train_dataset.height
        self.cx_render = self.render_w / 2.0
        self.cy_render = self.render_h / 2.0

        # Build K matrix for rendering
        self.K_render = torch.zeros(3, 3, device=self.device)
        self.K_render[0, 0] = self.fx_render
        self.K_render[1, 1] = self.fy_render
        self.K_render[0, 2] = self.cx_render
        self.K_render[1, 2] = self.cy_render
        self.K_render[2, 2] = 1.0

        # Coarse feature map size: H/14 x W/14 (ViT patch size)
        self.coarse_h = self.render_h // 14
        self.coarse_w = self.render_w // 14

        print(f"  Render size: {self.render_w}x{self.render_h}")
        print(f"  Coarse feature size: {self.coarse_w}x{self.coarse_h}")

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        args = self.args

        param_groups = [
            {'params': self.triplane.parameters(), 'lr': args.lr_triplane},
            {'params': self.encoder.coarse_encoder.proj.parameters(), 'lr': args.lr_encoder},
            {'params': self.encoder.fine_encoder.parameters(), 'lr': args.lr_encoder},
            {'params': self.seg_head_coarse.parameters(), 'lr': args.lr_encoder},
            {'params': self.seg_head_fine.parameters(), 'lr': args.lr_encoder},
        ]

        self.optimizer = torch.optim.Adam(param_groups, eps=1e-15)

        # Learning rate scheduler
        if args.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.total_iters - args.phase1_iters,
                eta_min=1e-6,
            )
        elif args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.lr_step_size,
                gamma=0.5,
            )
        else:
            self.scheduler = None

    def run_clustering(self):
        """Compute spectral clustering on Gaussian centers."""
        print("Running spectral clustering on Gaussian centers...")
        t0 = time.time()
        xyz_np = self.means3d.cpu().numpy()

        # Subsample for tractability (300K is large for spectral clustering)
        self.cluster_labels = spectral_clustering(
            xyz_np,
            n_clusters=self.args.n_clusters,
            n_eigenvectors=min(50, self.args.n_clusters + 10),
            subsample=min(20000, len(xyz_np)),
        )
        self.cluster_labels_tensor = torch.from_numpy(
            self.cluster_labels
        ).long().to(self.device)

        print(f"  Clustering done in {time.time()-t0:.1f}s, "
              f"{self.args.n_clusters} clusters")

    def update_prototypes(self, scale='coarse'):
        """Recompute cluster prototypes from current triplane features."""
        with torch.no_grad():
            if scale == 'coarse':
                features = self.triplane.extract_coarse(self.means3d)
            else:
                features = self.triplane.extract_fine(self.means3d)
            features = F.normalize(features, p=2, dim=1)

        prototypes = compute_prototypes(
            features, self.cluster_labels_tensor, self.args.n_clusters,
        )
        return prototypes

    def train_step_phase2(self, batch, iteration):
        """Phase 2 training step: joint contrastive + photometric loss."""
        args = self.args
        image_norm = batch['image_norm'].to(self.device)  # [1, 3, H, W]
        image_raw = batch['image'].to(self.device)
        viewmat = batch['w2c'].to(self.device)  # [1, 4, 4]

        Ks = self.K_render.unsqueeze(0)  # [1, 3, 3]

        # --- Extract 2D features ---
        coarse_feat_2d, fine_feat_2d = self.encoder(image_norm)
        # coarse: [1, D, H/14, W/14], fine: [1, D, H, W]

        # --- Extract triplane features for all Gaussians ---
        coarse_colors = self.triplane.extract_coarse(self.means3d)  # [N, D]
        coarse_colors_norm = F.normalize(coarse_colors, p=2, dim=1)

        # --- Render 3D features at coarse resolution ---
        K_coarse = self.K_render.clone()
        K_coarse[0] *= self.coarse_w / self.render_w
        K_coarse[1] *= self.coarse_h / self.render_h

        coarse_feat_3d = render_features_2dgs(
            self.means3d, self.quats, self.scales, self.opacities,
            coarse_colors_norm,
            viewmat, K_coarse.unsqueeze(0),
            self.coarse_w, self.coarse_h,
            chunk_size=16,
        )  # [1, D, H_c, W_c]

        # L2 normalize feature maps
        coarse_feat_2d_norm = F.normalize(coarse_feat_2d, p=2, dim=1)
        coarse_feat_3d_norm = F.normalize(coarse_feat_3d, p=2, dim=1)

        # Ensure spatial dimensions match (DINOv2 pads, so sizes may differ)
        if coarse_feat_2d_norm.shape[2:] != coarse_feat_3d_norm.shape[2:]:
            coarse_feat_2d_norm = F.interpolate(
                coarse_feat_2d_norm, coarse_feat_3d_norm.shape[2:],
                mode='bilinear', align_corners=False,
            )

        # --- Losses ---
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}

        # 1. NCE loss (coarse)
        l_nce = info_nce_loss(
            coarse_feat_3d_norm, coarse_feat_2d_norm,
            temperature=args.temperature,
            max_samples=args.nce_samples,
        )
        total_loss = total_loss + 0.5 * l_nce
        loss_dict['nce'] = l_nce.item()

        # 1b. Direct cosine similarity loss (aligns features for pose refinement)
        if args.cosine_loss_weight > 0:
            cos_sim_c = (coarse_feat_2d_norm * coarse_feat_3d_norm).sum(dim=1).mean()
            l_cos_c = 1.0 - cos_sim_c
            total_loss = total_loss + args.cosine_loss_weight * l_cos_c
            loss_dict['cos_c'] = cos_sim_c.item()

        # 2. Prototypical loss (coarse)
        if hasattr(self, 'prototypes_coarse') and self.prototypes_coarse is not None:
            l_pro = prototypical_loss(
                coarse_feat_3d_norm, coarse_feat_2d_norm,
                self.prototypes_coarse,
                temperature=args.temperature,
                max_samples=args.nce_samples,
            )
            total_loss = total_loss + 0.5 * l_pro
            loss_dict['pro'] = l_pro.item()

            # 3. Segmentation CE loss (coarse)
            proto_labels = get_pixel_proto_labels(
                coarse_feat_3d_norm, self.prototypes_coarse,
            )  # [1, H_c, W_c]

            logits_3d = compute_pseudo_logits(
                coarse_feat_3d_norm, self.prototypes_coarse,
            )  # [1, K, H_c, W_c]

            seg_2d = self.seg_head_coarse(coarse_feat_2d_norm)  # [1, K, H_c, W_c]

            l_ce = segmentation_ce_loss(seg_2d, logits_3d, proto_labels)
            total_loss = total_loss + 0.5 * l_ce
            loss_dict['ce'] = l_ce.item()

        # 4. Total variation loss on triplane
        l_tvl = self.triplane.total_variation_loss()
        total_loss = total_loss + 0.1 * l_tvl
        loss_dict['tvl'] = l_tvl.item()

        # --- Fine level: full losses (NCE + PRO + CE) ---
        if iteration > args.fine_start_iter:
            fine_colors = self.triplane.extract_fine(self.means3d)
            fine_colors_norm = F.normalize(fine_colors, p=2, dim=1)

            fine_render_h = min(self.render_h, 270)
            fine_render_w = min(self.render_w, 480)

            K_fine = self.K_render.clone()
            K_fine[0] *= fine_render_w / self.render_w
            K_fine[1] *= fine_render_h / self.render_h

            fine_feat_3d = render_features_2dgs(
                self.means3d, self.quats, self.scales, self.opacities,
                fine_colors_norm,
                viewmat, K_fine.unsqueeze(0),
                fine_render_w, fine_render_h,
                chunk_size=16,
            )

            fine_feat_2d_resized = F.interpolate(
                fine_feat_2d, (fine_render_h, fine_render_w),
                mode='bilinear', align_corners=False,
            )

            fine_feat_2d_norm = F.normalize(fine_feat_2d_resized, p=2, dim=1)
            fine_feat_3d_norm = F.normalize(fine_feat_3d, p=2, dim=1)

            # Fine NCE loss
            l_nce_fine = info_nce_loss(
                fine_feat_3d_norm, fine_feat_2d_norm,
                temperature=args.temperature,
                max_samples=args.nce_samples,
            )
            total_loss = total_loss + 0.5 * l_nce_fine
            loss_dict['nce_fine'] = l_nce_fine.item()

            # Fine prototypical loss
            if hasattr(self, 'prototypes_fine') and self.prototypes_fine is not None:
                l_pro_fine = prototypical_loss(
                    fine_feat_3d_norm, fine_feat_2d_norm,
                    self.prototypes_fine,
                    temperature=args.temperature,
                    max_samples=args.nce_samples,
                )
                total_loss = total_loss + 0.5 * l_pro_fine
                loss_dict['pro_fine'] = l_pro_fine.item()

                # Fine segmentation CE loss
                fine_proto_labels = get_pixel_proto_labels(
                    fine_feat_3d_norm, self.prototypes_fine,
                )
                fine_logits_3d = compute_pseudo_logits(
                    fine_feat_3d_norm, self.prototypes_fine,
                )
                fine_seg_2d = self.seg_head_fine(fine_feat_2d_norm)
                l_ce_fine = segmentation_ce_loss(fine_seg_2d, fine_logits_3d, fine_proto_labels)
                total_loss = total_loss + 0.5 * l_ce_fine
                loss_dict['ce_fine'] = l_ce_fine.item()

            # Direct cosine similarity loss for fine features
            fine_cos = (fine_feat_2d_norm * fine_feat_3d_norm).sum(dim=1).mean()
            loss_dict['cos_f'] = fine_cos.item()
            if args.cosine_loss_weight > 0:
                l_cos_f = 1.0 - fine_cos
                total_loss = total_loss + args.cosine_loss_weight * l_cos_f

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict

    def train(self):
        """Main training loop: 50K iterations."""
        args = self.args
        print(f"\n{'='*60}")
        print(f"Starting GSFFs training: {args.total_iters} iterations")
        print(f"  Phase 1 (photometric only): 0-{args.phase1_iters}")
        print(f"  Phase 2 (joint contrastive): {args.phase1_iters}-{args.total_iters}")
        print(f"{'='*60}\n")

        # Run clustering before phase 2
        self.run_clustering()
        self.prototypes_coarse = None
        self.prototypes_fine = None

        iteration = self.start_iteration
        epoch = 0
        best_loss = -float('inf')  # For fine_cos (higher is better)
        args.fine_start_iter = args.phase1_iters  # Start fine immediately after warmup
        log_mode = 'a' if self.start_iteration > 0 else 'w'
        log_file = open(os.path.join(args.output_dir, 'train.log'), log_mode)

        while iteration < args.total_iters:
            epoch += 1
            for batch in self.train_loader:
                if iteration >= args.total_iters:
                    break

                self.optimizer.zero_grad()

                if iteration < args.phase1_iters:
                    # Phase 1: Skip (we use pre-trained 2DGS, no RGB training needed)
                    # The paper trains Gaussian RGB for 15K iters first, but
                    # since we already have a trained 2DGS model, we skip to phase 2.
                    # Just warm up with TVL + weak NCE
                    if iteration == 0:
                        print("Phase 1: Warm-up with TV loss + weak NCE...")
                    loss, loss_dict = self.train_step_phase2(batch, iteration)
                    loss = loss * 0.1  # Weaker gradients during warmup
                else:
                    # Phase 2: Full joint training
                    if iteration == args.phase1_iters:
                        print(f"\nPhase 2 started at iteration {iteration}")

                    loss, loss_dict = self.train_step_phase2(batch, iteration)

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.triplane.parameters()) +
                    list(self.encoder.parameters()) +
                    list(self.seg_head_coarse.parameters()) +
                    list(self.seg_head_fine.parameters()),
                    max_norm=1.0,
                )

                self.optimizer.step()
                if iteration >= args.phase1_iters and self.scheduler is not None:
                    self.scheduler.step()

                # Update prototypes periodically
                if iteration > 0 and iteration % args.prototype_update_freq == 0:
                    self.prototypes_coarse = self.update_prototypes('coarse')
                    if iteration > args.fine_start_iter:
                        self.prototypes_fine = self.update_prototypes('fine')

                # Logging
                if iteration % 100 == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    msg = (f"[Iter {iteration:6d}/{args.total_iters}] "
                           f"loss={loss_dict.get('total', 0):.4f} "
                           f"nce={loss_dict.get('nce', 0):.4f} "
                           f"pro={loss_dict.get('pro', 0):.4f} "
                           f"ce={loss_dict.get('ce', 0):.4f} "
                           f"tvl={loss_dict.get('tvl', 0):.4f} "
                           f"nce_f={loss_dict.get('nce_fine', 0):.4f} "
                           f"cos_c={loss_dict.get('cos_c', 0):.3f} "
                           f"cos_f={loss_dict.get('cos_f', 0):.3f} "
                           f"lr={lr:.6f}")
                    print(msg)
                    log_file.write(msg + '\n')
                    log_file.flush()

                # Save checkpoint
                if iteration > 0 and iteration % args.save_freq == 0:
                    self.save_checkpoint(iteration, loss_dict.get('total', 0))
                    # Use fine cos_sim as best metric (higher = better)
                    if 'cos_f' in loss_dict:
                        metric = loss_dict['cos_f']
                        if metric > best_loss:  # best_loss stores best fine_cos
                            best_loss = metric
                            self.save_checkpoint(iteration, loss_dict.get('total', 0), name='best.pth')
                            print(f'  ★ New best fine_cos={metric:.4f}')
                    elif loss_dict.get('nce', float('inf')) < best_loss:
                        best_loss = loss_dict['nce']
                        self.save_checkpoint(iteration, best_loss, name='best.pth')

                iteration += 1

        # Final save
        self.save_checkpoint(iteration, loss_dict.get('total', 0), name='final.pth')
        log_file.close()
        print(f"\nTraining complete. Output: {args.output_dir}")

    def save_checkpoint(self, iteration, loss, name=None):
        if name is None:
            name = f'iter_{iteration:06d}.pth'

        path = os.path.join(self.args.output_dir, 'checkpoints', name)
        state = {
            'iteration': iteration,
            'loss': loss,
            'triplane': self.triplane.state_dict(),
            'encoder': self.encoder.state_dict(),
            'seg_head_coarse': self.seg_head_coarse.state_dict(),
            'seg_head_fine': self.seg_head_fine.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'cluster_labels': self.cluster_labels,
            'scene_extent': self.scene_extent,
            'args': vars(self.args),
        }
        if hasattr(self, 'prototypes_coarse') and self.prototypes_coarse is not None:
            state['prototypes_coarse'] = self.prototypes_coarse.cpu()
        if hasattr(self, 'prototypes_fine') and self.prototypes_fine is not None:
            state['prototypes_fine'] = self.prototypes_fine.cpu()

        torch.save(state, path)
        # Also save as latest
        latest_path = os.path.join(self.args.output_dir, 'checkpoints', 'latest.pth')
        torch.save(state, latest_path)


def parse_args():
    parser = argparse.ArgumentParser(description='GSFFs Training')

    # Paths
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help='2DGS PLY path')
    parser.add_argument('--cameras_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--warmstart', type=str, default=None,
                        help='Load triplane + coarse encoder weights only (reset optimizer)')

    # Model
    parser.add_argument('--feature_dim', type=int, default=16)
    parser.add_argument('--coarse_resolution', type=int, default=256)
    parser.add_argument('--fine_resolution', type=int, default=1024)
    parser.add_argument('--n_clusters', type=int, default=34)

    # Rendering
    parser.add_argument('--render_height', type=int, default=540)
    parser.add_argument('--render_width', type=int, default=960)

    # Training
    parser.add_argument('--total_iters', type=int, default=50000)
    parser.add_argument('--phase1_iters', type=int, default=5000,
                        help='Warmup iters (paper uses 15K but we have pre-trained 2DGS)')
    parser.add_argument('--lr_triplane', type=float, default=1e-3)
    parser.add_argument('--lr_encoder', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--nce_samples', type=int, default=1024)
    parser.add_argument('--prototype_update_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--use_old_fine_encoder', action='store_true',
                        help='Use old CNN-only fine encoder (v1 compatible)')
    parser.add_argument('--cosine_loss_weight', type=float, default=0.0,
                        help='Weight for direct cosine similarity loss (0=disabled)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='LR scheduler type')
    parser.add_argument('--lr_step_size', type=int, default=50000,
                        help='Step size for StepLR scheduler')

    return parser.parse_args()


def main():
    args = parse_args()
    trainer = GSFFTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
