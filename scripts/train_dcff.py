"""
Train Deferred Cascaded Feature Field (DCFF).

Joint training of 2DGS geometry + 16d latent + hash grid + decoders,
supervised by pre-extracted dual-scale RADIO features.

Phase 1 (0 → fine_start):       Geometry only (RGB + depth)
Phase 2 (fine_start → coarse_start): + Fine features (latent → decoder vs RADIO_geo)
Phase 3 (coarse_start → end):   + Coarse features (hash grid + MLP vs RADIO_sem)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_dcff.py \
        --config configs/dcff_oldhospital.yaml

    # Resume from checkpoint
    python scripts/train_dcff.py --config configs/dcff_oldhospital.yaml \
        --resume output/dcff_oldhospital/checkpoints/latest.pth
"""

import os
import sys
import math
import time
import yaml
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcff.hybrid_gaussian import HybridGaussianModel
from dcff.hash_grid import SpatialHashGrid
from dcff.deferred_renderer import DeferredCascadedRenderer
from dcff.losses import DCFFLoss

# Reuse scene loading from existing codebase
from feature_3dgs.train_2dgs_joint_v2 import (
    load_scene_colmap, build_da3_image_order, CameraData,
)

from gsplat import rasterization_2dgs


# ═══════════════════════════════════════════════════════════════════════════
# Feature Cache (loads pre-extracted RADIO dual-scale features)
# ═══════════════════════════════════════════════════════════════════════════

class DualRadioFeatureCache:
    """Lazy loader for pre-extracted dual-scale RADIO features.

    Loads both fine_geo (shallow RADIO) and coarse_sem (deep RADIO) features
    with CPU storage + GPU LRU cache.
    """

    def __init__(self, feature_dir, max_gpu_cache=32):
        self.feature_dir = feature_dir
        self._max_gpu = max_gpu_cache
        self._gpu_cache = {}
        self._gpu_order = []
        self._cpu_cache = {}

        # Discover available features
        geo_dir = os.path.join(feature_dir, 'fine_geo')
        sem_dir = os.path.join(feature_dir, 'coarse_sem')

        if not os.path.isdir(geo_dir) or not os.path.isdir(sem_dir):
            raise FileNotFoundError(
                f"Expected fine_geo/ and coarse_sem/ in {feature_dir}"
            )

        # Parse frame IDs from filenames
        self.frame_ids = set()
        self._geo_paths = {}
        self._sem_paths = {}

        for fn in sorted(os.listdir(geo_dir)):
            if fn.endswith('.pt') and fn.startswith('rgb_'):
                parts = fn.replace('.pt', '').split('_')
                fid = int(parts[1])
                self.frame_ids.add(fid)
                self._geo_paths[fid] = os.path.join(geo_dir, fn)

        for fn in sorted(os.listdir(sem_dir)):
            if fn.endswith('.pt') and fn.startswith('rgb_'):
                parts = fn.replace('.pt', '').split('_')
                fid = int(parts[1])
                self._sem_paths[fid] = os.path.join(sem_dir, fn)

        # Read shape from first file
        sample = torch.load(self._geo_paths[min(self.frame_ids)], map_location='cpu')
        self.geo_dim, self.geo_h, self.geo_w = sample.shape
        sample = torch.load(self._sem_paths[min(self.frame_ids)], map_location='cpu')
        self.sem_dim, self.sem_h, self.sem_w = sample.shape

        print(f"  [DualRadioCache] {len(self.frame_ids)} frames")
        print(f"    fine_geo:   {self.geo_dim}d @ {self.geo_w}x{self.geo_h}")
        print(f"    coarse_sem: {self.sem_dim}d @ {self.sem_w}x{self.sem_h}")

    def get(self, fid):
        """Get (geo, sem) feature tensors on GPU for frame fid.

        Returns:
            geo: [geo_dim, geo_h, geo_w] float32 on GPU
            sem: [sem_dim, sem_h, sem_w] float32 on GPU
        """
        if fid in self._gpu_cache:
            return self._gpu_cache[fid]

        # Load from CPU cache or disk
        if fid not in self._cpu_cache:
            geo = torch.load(self._geo_paths[fid], map_location='cpu').float()
            sem = torch.load(self._sem_paths[fid], map_location='cpu').float()
            self._cpu_cache[fid] = (geo, sem)

        geo_cpu, sem_cpu = self._cpu_cache[fid]
        result = (geo_cpu.cuda(), sem_cpu.cuda())
        self._gpu_cache[fid] = result
        self._gpu_order.append(fid)

        # Evict oldest
        while len(self._gpu_cache) > self._max_gpu:
            old = self._gpu_order.pop(0)
            if old in self._gpu_cache:
                del self._gpu_cache[old]

        return result


# ═══════════════════════════════════════════════════════════════════════════
# Image loading utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_image_tensor(cam, longest_edge=960):
    """Load and resize image to tensor [1, 3, H, W] on GPU."""
    from PIL import Image
    from torchvision import transforms

    img = Image.open(cam.image).convert('RGB')

    # Resize to longest edge
    W, H = img.size
    scale = longest_edge / max(W, H)
    if scale < 1.0:
        new_W = int(W * scale)
        new_H = int(H * scale)
        img = img.resize((new_W, new_H), Image.LANCZOS)

    tensor = transforms.ToTensor()(img).unsqueeze(0).cuda()  # [1, 3, H, W]
    return tensor


def load_image_batch(cams, longest_edge=960):
    """Load a batch of images and stack them to [B, 3, H, W]."""
    images = [load_image_tensor(cam, longest_edge) for cam in cams]
    return torch.cat(images, dim=0)


def cam_to_viewmat(cam):
    """Convert CameraData to 4x4 world-to-camera matrix on GPU."""
    W2C = np.eye(4)
    W2C[:3, :3] = cam.R.T
    W2C[:3, 3] = cam.T
    return torch.tensor(W2C, dtype=torch.float32, device="cuda")


def cam_to_K(cam, width, height):
    """Compute intrinsics matrix scaled to target resolution."""
    tanfovx = math.tan(cam.FovX * 0.5)
    tanfovy = math.tan(cam.FovY * 0.5)
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    return torch.tensor([
        [fx, 0, width / 2.0],
        [0, fy, height / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device="cuda")


# ═══════════════════════════════════════════════════════════════════════════
# Main Training
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg, resume_path=None, init_ply=None):
    exp_name = cfg['exp_name']
    output_dir = os.path.join(cfg['output_dir'], exp_name)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    dcfg = cfg['dataset']
    mcfg = cfg['model']
    tcfg = cfg['training']
    hcfg = cfg['hash_grid']
    fcfg = cfg.get('fine_decoder', {})

    print(f"\n{'='*70}")
    print(f"  DCFF: Deferred Cascaded Feature Field Training")
    print(f"  Experiment: {exp_name}")
    print(f"{'='*70}")
    print(f"  Dataset:        {dcfg['source_dir']}")
    print(f"  Features:       {dcfg['feature_dir']}")
    print(f"  Latent dim:     {mcfg['latent_dim']}")
    print(f"  Feature dim:    {mcfg['feature_dim']}")
    print(f"  Iterations:     {tcfg['iterations']}")
    print(f"  Fine start:     {tcfg['fine_start_iter']}")
    print(f"  Coarse start:   {tcfg['coarse_start_iter']}")
    print(f"  Hash mode:      {hcfg.get('input_mode', 'legacy')}")
    print(f"{'='*70}\n")

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── 1. Load scene ──
    print("Loading scene...")
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
        load_scene_colmap(dcfg['source_dir'], dcfg.get('images', ''))

    # ── 2. Load dual-scale RADIO features ──
    print("\nLoading dual-scale RADIO features...")
    radio_cache = DualRadioFeatureCache(dcfg['feature_dir'])

    # Build camera → frame ID mapping
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
    print(f"  Matched {len(cam_to_fid)}/{len(train_cams)} cameras to features")

    # ── 3. Create models ──
    print("\nInitializing models...")

    # 3a. HybridGaussianModel (2DGS + latent)
    gaussians = HybridGaussianModel(
        sh_degree=mcfg['sh_degree'],
        latent_dim=mcfg['latent_dim'],
    )
    geometry_lr_scale = float(tcfg.get('geometry_finetune_lr_scale', 1.0)) if init_ply is not None else 1.0
    train_args = argparse.Namespace(**{
        'position_lr_init': float(tcfg['position_lr_init']) * geometry_lr_scale,
        'position_lr_final': float(tcfg['position_lr_final']) * geometry_lr_scale,
        'feature_lr': float(tcfg['feature_lr']) * geometry_lr_scale,
        'opacity_lr': float(tcfg['opacity_lr']) * geometry_lr_scale,
        'scaling_lr': float(tcfg['scaling_lr']) * geometry_lr_scale,
        'rotation_lr': float(tcfg['rotation_lr']) * geometry_lr_scale,
        'latent_lr': float(tcfg['latent_lr']),
        'percent_dense': float(tcfg['percent_dense']),
        'iterations': tcfg['iterations'],
    })

    # Resume: load Gaussians from PLY, otherwise init from point cloud
    start_iteration = 0
    freeze_geometry = False
    geometry_frozen = False
    resume_meta = None
    if resume_path is not None:
        resume_meta = torch.load(resume_path, map_location='cpu')
        ply_path = resume_path.replace('.pth', '.ply')
        if os.path.exists(ply_path):
            freeze_geometry = bool(resume_meta.get('geometry_frozen', False))
            gaussians.load_ply(ply_path, freeze_geometry=freeze_geometry)
            geometry_frozen = freeze_geometry
            print(f"  Resumed Gaussians from {ply_path}: {gaussians.num_points:,}")
        else:
            raise FileNotFoundError(f"PLY not found for resume: {ply_path}")
    elif init_ply is not None:
        # Initialize from pretrained geometry, freeze it, only train latent
        freeze_geometry = tcfg.get('freeze_geometry', True)
        gaussians.load_ply(init_ply, freeze_geometry=freeze_geometry)
        geometry_frozen = freeze_geometry
        gaussians.spatial_lr_scale = cameras_extent
        print(f"  Initialized from pretrained PLY: {init_ply}")
        print(f"  Geometry frozen: {freeze_geometry}")
    else:
        gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)
    gaussians.training_setup(train_args)

    bg_color = torch.tensor(
        [1, 1, 1] if mcfg.get('white_background', False) else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    # 3b. Compute scene extent for hash grid normalization
    xyz_init = gaussians.get_xyz.detach().cpu().numpy()
    norms = np.linalg.norm(xyz_init, axis=1)
    scene_extent = float(np.percentile(norms, 99)) * 1.2
    print(f"  Scene extent: {scene_extent:.2f}")

    # 3c. Hash Grid (implicit coarse decoder)
    hash_grid = SpatialHashGrid(
        scene_extent=scene_extent,
        feature_dim=mcfg['feature_dim'],
        input_mode=hcfg.get('input_mode', 'legacy'),
        latent_dim=mcfg['latent_dim'],
        scale_dim=hcfg.get('scale_dim', 2),
        scale_pe_freqs=hcfg.get('scale_pe_freqs', 4),
        include_raw_scale=hcfg.get('include_raw_scale', True),
        n_levels=hcfg['n_levels'],
        n_features_per_level=hcfg['n_features_per_level'],
        log2_hashmap_size=hcfg['log2_hashmap_size'],
        base_resolution=hcfg['base_resolution'],
        max_resolution=hcfg['max_resolution'],
        sh_degree=hcfg.get('sh_degree', 3),
        mlp_hidden=hcfg.get('mlp_hidden', 128),
        mlp_layers=hcfg.get('mlp_layers', 2),
    ).cuda()

    # 3d. Deferred Cascaded Renderer
    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid,
        latent_dim=mcfg['latent_dim'],
        fine_feature_dim=mcfg['feature_dim'],
        coarse_feature_dim=mcfg['feature_dim'],
        fine_hidden_dim=fcfg.get('hidden_dim'),
        fine_num_layers=fcfg.get('num_layers', 3),
        fine_use_viewdirs=fcfg.get('use_viewdirs', False),
        fine_view_degree=fcfg.get('view_degree', 2),
    ).cuda()

    # 3e. Loss
    loss_cfg = tcfg.get('loss', {})
    loss_fn = DCFFLoss(
        lambda_dssim=float(loss_cfg.get('lambda_dssim', 0.2)),
        lambda_fine_cos=float(loss_cfg.get('lambda_fine_cos', 1.0)),
        lambda_fine_l1=float(loss_cfg.get('lambda_fine_l1', 0.5)),
        lambda_coarse_cos=float(loss_cfg.get('lambda_coarse_cos', 1.0)),
        lambda_coarse_l1=float(loss_cfg.get('lambda_coarse_l1', 0.5)),
        lambda_tv=float(loss_cfg.get('lambda_tv', 0.01)),
        lambda_normal=float(loss_cfg.get('lambda_normal', 0.05)),
        lambda_dist=float(loss_cfg.get('lambda_dist', 0.01)),
    )

    # Print param counts
    n_hash = sum(p.numel() for p in hash_grid.parameters())
    n_fine = sum(p.numel() for p in renderer.fine_decoder.parameters())
    n_gauss = gaussians.num_points * (3 + 4 + 2 + 1 + 3 + mcfg['latent_dim'])
    print(f"\n  Gaussians:   {gaussians.num_points:,} ({n_gauss:,} parameters)")
    print(f"  Hash grid:   {n_hash:,} params")
    print(f"  Fine decoder: {n_fine:,} params")
    print(f"  Total DCFF:  {n_hash + n_fine:,} trainable (excl. Gaussians)\n")

    # ── 4. Optimizers ──
    # Separate optimizer for DCFF modules (hash grid + fine decoder)
    dcff_params = [
        {'params': hash_grid.parameters(), 'lr': float(tcfg['lr_hash_grid'])},
        {'params': renderer.fine_decoder.parameters(), 'lr': float(tcfg['lr_fine_decoder'])},
    ]
    dcff_optimizer = torch.optim.Adam(dcff_params, eps=1e-15)
    dcff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dcff_optimizer,
        T_max=tcfg['iterations'] - tcfg['fine_start_iter'],
        eta_min=1e-6,
    )

    # ── 4b. Resume checkpoint state ──
    if resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cuda')
        hash_grid.load_state_dict(ckpt['hash_grid_state'])
        renderer.fine_decoder.load_state_dict(ckpt['fine_decoder_state'])
        dcff_optimizer.load_state_dict(ckpt['dcff_optimizer_state'])
        dcff_scheduler.load_state_dict(ckpt['dcff_scheduler_state'])
        start_iteration = ckpt['iteration']
        best_metrics = ckpt.get('best_metrics', {'total_loss': float('inf')})
        geometry_frozen = bool(ckpt.get('geometry_frozen', geometry_frozen))
        print(f"  Resumed from iteration {start_iteration}")
        print(f"  Best loss so far: {best_metrics.get('total_loss', 'N/A')}")

    # ── 5. Training loop ──
    longest_edge = tcfg.get('longest_edge', 960)
    batch_size = int(tcfg.get('batch_size', 1))
    total_iters = tcfg['iterations']
    fine_start = tcfg['fine_start_iter']
    coarse_start = tcfg['coarse_start_iter']
    densify_until = tcfg.get('densify_until_iter', 25000)
    densify_from = tcfg.get('densify_from_iter', 500)
    densify_every = tcfg.get('densify_every', 100)
    opacity_reset_every = tcfg.get('opacity_reset_every', 3000)
    geometry_unfreeze_iter = int(tcfg.get('geometry_unfreeze_iter', 0) or 0)
    geometry_densify_after_unfreeze = bool(tcfg.get('geometry_densify_after_unfreeze', False))
    grad_threshold = float(tcfg.get('densify_grad_threshold', 0.0002))
    min_opacity = float(tcfg.get('min_opacity', 0.005))
    max_num_gaussians = tcfg.get('max_num_gaussians', 500000)
    log_every = tcfg.get('log_every', 100)
    save_every = tcfg.get('save_every', 5000)
    sh_up_every = tcfg.get('sh_degree_up_every', 1000)

    # Feature resolution (matching RADIO output)
    feat_h, feat_w = radio_cache.geo_h, radio_cache.geo_w

    # Cameras with matched features only
    valid_cams = [c for c in train_cams if c.uid in cam_to_fid]
    n_cams = len(valid_cams)

    # Setup logging
    log_path = os.path.join(output_dir, 'train.log')
    log_f = open(log_path, 'a' if start_iteration > 0 else 'w')
    if start_iteration > 0:
        log_f.write(f"\n--- Resumed from iteration {start_iteration} ---\n")

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f"Training {total_iters} iterations, {n_cams} cameras")
    log(f"Batch size: {batch_size}")
    log(f"Feature target: {feat_w}x{feat_h}")
    if init_ply is not None:
        log(f"Geometry LR scale: {geometry_lr_scale:.4f}")
        if geometry_unfreeze_iter > 0:
            log(f"Geometry unfreeze scheduled at iter {geometry_unfreeze_iter}")

    if start_iteration == 0:
        best_metrics = {'total_loss': float('inf')}
    start_time = time.time()

    for iteration in range(start_iteration + 1, total_iters + 1):
        if geometry_frozen and geometry_unfreeze_iter > 0 and iteration == geometry_unfreeze_iter:
            gaussians.set_geometry_trainable(True)
            gaussians.training_setup(train_args)
            geometry_frozen = False
            log(f"[Geometry] Unfroze geometry at iter {iteration} with LR scale {geometry_lr_scale:.4f}")

        # Determine phase
        if iteration < fine_start:
            phase = 1
        elif iteration < coarse_start:
            phase = 2
        else:
            phase = 3

        gaussians.update_learning_rate(iteration)

        # SH degree ramp-up
        if iteration % sh_up_every == 0:
            gaussians.oneupSHdegree()

        # Sample random batch of cameras
        batch_indices = np.random.randint(n_cams, size=batch_size)
        cams = [valid_cams[idx] for idx in batch_indices]
        fids = [cam_to_fid[cam.uid] for cam in cams]

        # Load GT image batch
        gt_image = load_image_batch(cams, longest_edge)
        _, _, img_h, img_w = gt_image.shape

        # Camera matrices
        viewmat = torch.stack([cam_to_viewmat(cam) for cam in cams], dim=0)
        K = torch.stack([cam_to_K(cam, img_w, img_h) for cam in cams], dim=0)

        # ── Forward pass ──
        render_result = renderer(
            gaussians,
            viewmat=viewmat,
            K=K,
            width=img_w,
            height=img_h,
            render_coarse=(phase >= 3),
            feature_height=feat_h,
            feature_width=feat_w,
        )

        # Load RADIO targets
        radio_geo, radio_sem = None, None
        if phase >= 2:
            geo_batch = []
            sem_batch = []
            for fid in fids:
                geo, sem = radio_cache.get(fid)
                geo_batch.append(geo)
                sem_batch.append(sem)
            radio_geo = torch.stack(geo_batch, dim=0)
            if phase >= 3:
                radio_sem = torch.stack(sem_batch, dim=0)

        # ── Loss computation ──
        losses = loss_fn.compute(
            render_result=render_result,
            gt_rgb=gt_image,
            radio_geo=radio_geo,
            radio_sem=radio_sem,
            hash_grid=hash_grid if phase >= 3 else None,
            phase=phase,
        )

        total_loss = losses['total']

        # ── Backward + optimize ──
        total_loss.backward()

        with torch.no_grad():
            # Gaussian optimizer step (geometry + latent)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # DCFF optimizer step (hash grid + fine decoder)
            if phase >= 2:
                dcff_optimizer.step()
                dcff_optimizer.zero_grad(set_to_none=True)
                dcff_scheduler.step()

            # ── Densification (skip if geometry is frozen) ──
            allow_geometry_updates = not geometry_frozen
            allow_densify = allow_geometry_updates and (
                iteration < densify_until and iteration >= densify_from and geometry_densify_after_unfreeze
            )
            if allow_densify:
                meta = render_result.get('meta', {})
                if 'means2d' in meta:
                    grad_2d = meta['means2d'].grad
                    if grad_2d is not None:
                        radii = meta.get('radii', None)
                        if radii is not None:
                            radii = radii.squeeze(0)  # [1, N] → [N]
                            visibility = radii > 0
                            # Update max radii for pruning
                            gaussians.max_radii2D[visibility] = torch.max(
                                gaussians.max_radii2D[visibility],
                                radii[visibility],
                            )
                        else:
                            visibility = torch.ones(gaussians.num_points, dtype=torch.bool, device="cuda")
                        gaussians.add_densification_stats(
                            grad_2d, visibility, img_w, img_h,
                        )

                if iteration % densify_every == 0:
                    # Skip densification if already at cap
                    if gaussians.num_points < max_num_gaussians:
                        gaussians.densify_and_prune(
                            max_grad=grad_threshold,
                            min_opacity=min_opacity,
                            extent=cameras_extent,
                            max_screen_size=20,
                        )
                    else:
                        # Only prune, no cloning/splitting
                        prune_mask = (gaussians.get_opacity < min_opacity).squeeze()
                        if prune_mask.sum() > 0:
                            gaussians._prune_points(prune_mask)

            # Opacity reset (only during densification phase, not after)
            if iteration % opacity_reset_every == 0 and iteration < densify_until and not geometry_frozen and geometry_densify_after_unfreeze:
                gaussians.reset_opacity()

        # ── Logging ──
        if iteration % log_every == 0:
            elapsed = time.time() - start_time
            elapsed_iters = max(1, iteration - start_iteration)
            it_per_sec = elapsed_iters / max(elapsed, 1e-6)
            img_per_sec = it_per_sec * batch_size

            parts = [f"[Iter {iteration}]"]
            parts.append(f"P{phase}")
            parts.append(f"L={losses['total']:.4f}")
            parts.append(f"RGB={losses['rgb']:.4f}")

            if 'fine_cos' in losses:
                parts.append(f"fine_cos={1-losses['fine_cos'].item():.3f}")
            if 'coarse_cos' in losses:
                parts.append(f"coarse_cos={1-losses['coarse_cos'].item():.3f}")
            if 'tv' in losses:
                parts.append(f"TV={losses['tv']:.4f}")

            parts.append(f"B={batch_size}")
            parts.append(f"N={gaussians.num_points:,}")
            parts.append(f"({it_per_sec:.1f} it/s, {img_per_sec:.1f} img/s)")

            msg = ' | '.join(parts)

            if losses['total'].item() < best_metrics['total_loss']:
                best_metrics['total_loss'] = losses['total'].item()
                best_metrics['iteration'] = iteration
                msg += ' ★ BEST'

            log(msg)

        # ── Checkpointing ──
        if iteration % save_every == 0 or iteration == total_iters:
            ckpt = {
                'iteration': iteration,
                'config': cfg,
                'gaussian_state': {
                    'num_points': gaussians.num_points,
                },
                'hash_grid_state': hash_grid.state_dict(),
                'fine_decoder_state': renderer.fine_decoder.state_dict(),
                'dcff_optimizer_state': dcff_optimizer.state_dict(),
                'dcff_scheduler_state': dcff_scheduler.state_dict(),
                'best_metrics': best_metrics,
                'geometry_frozen': geometry_frozen,
            }
            torch.save(ckpt, os.path.join(ckpt_dir, 'latest.pth'))
            gaussians.save_ply(os.path.join(ckpt_dir, 'latest.ply'))

            if losses['total'].item() <= best_metrics.get('total_loss', float('inf')):
                torch.save(ckpt, os.path.join(ckpt_dir, 'best.pth'))
                gaussians.save_ply(os.path.join(ckpt_dir, 'best.ply'))

            log(f"  [Checkpoint] Saved at iter {iteration}")

    elapsed = time.time() - start_time
    log(f"\nTraining complete in {elapsed/3600:.1f}h")
    log(f"Best total loss: {best_metrics['total_loss']:.4f} at iter {best_metrics.get('iteration', 0)}")
    log_f.close()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='DCFF Training')
    parser.add_argument('--config', required=True, help='Config YAML path')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    parser.add_argument('--init_ply', default=None,
                        help='Initialize geometry from pretrained PLY (freeze geom, train latent only)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    init_ply = args.init_ply
    if init_ply is None:
        init_ply = cfg.get('training', {}).get('init_ply')

    train(cfg, resume_path=args.resume, init_ply=init_ply)


if __name__ == '__main__':
    main()
