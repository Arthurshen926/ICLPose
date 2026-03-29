"""
DCFF v5 Training — Feature Field with Online/Cached RADIO Teacher.

Key improvements over v4:
  1. Online RADIO teacher mode (no PCA bottleneck, learned projection)
  2. Cached PCA mode with existing features (backward compatible)
  3. Multi-resolution supervision (fine @ full, coarse @ half)
  4. Gradient clipping + LR warmup for training stability
  5. Larger latent_dim (32d default) for better fine feature capacity
  6. Validation every N iterations

Usage:
    # Online RADIO teacher (no PCA)
    CUDA_VISIBLE_DEVICES=1 python scripts/train_dcff_v5.py \
        --config configs/dcff_oldhospital_v5a.yaml

    # Cached PCA features
    CUDA_VISIBLE_DEVICES=5 python scripts/train_dcff_v5.py \
        --config configs/dcff_oldhospital_v5b.yaml
"""

import os
import sys
import math
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcff.hybrid_gaussian import HybridGaussianModel
from dcff.hash_grid import SpatialHashGrid
from dcff.deferred_renderer import DeferredCascadedRenderer
from dcff.losses import DCFFLoss
from dcff.radio_teacher import OnlineRadioTeacher, CachedFeatureTeacher

from feature_3dgs.train_2dgs_joint_v2 import (
    load_scene_colmap, build_da3_image_order, CameraData,
)


# ═════════════════════════════════════════════════════════════════
# Image loading
# ═════════════════════════════════════════════════════════════════

def load_image_tensor(cam, longest_edge=960):
    """Load and resize to longest edge → [1, 3, H, W] on GPU."""
    img = Image.open(cam.image).convert('RGB')
    W, H = img.size
    scale = longest_edge / max(W, H)
    if scale < 1.0:
        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).cuda()


def load_image_fullres(cam):
    """Load at full resolution → [1, 3, H, W] on GPU."""
    img = Image.open(cam.image).convert('RGB')
    return transforms.ToTensor()(img).unsqueeze(0).cuda()


def load_image_batch(cams, longest_edge=960):
    return torch.cat([load_image_tensor(c, longest_edge) for c in cams], dim=0)


def load_fullres_batch(cams):
    return torch.cat([load_image_fullres(c) for c in cams], dim=0)


def cam_to_viewmat(cam):
    W2C = np.eye(4)
    W2C[:3, :3] = cam.R.T
    W2C[:3, 3] = cam.T
    return torch.tensor(W2C, dtype=torch.float32, device="cuda")


def cam_to_K(cam, width, height):
    tanfovx = math.tan(cam.FovX * 0.5)
    tanfovy = math.tan(cam.FovY * 0.5)
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    return torch.tensor([
        [fx, 0, width / 2.0],
        [0, fy, height / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device="cuda")


# ═════════════════════════════════════════════════════════════════
# LR warmup helper
# ═════════════════════════════════════════════════════════════════

def warmup_lr_scale(iteration, warmup_iters):
    """Linear warmup from 0.1 to 1.0 over warmup_iters."""
    if warmup_iters <= 0 or iteration >= warmup_iters:
        return 1.0
    return 0.1 + 0.9 * (iteration / warmup_iters)


# ═════════════════════════════════════════════════════════════════
# Main Training
# ═════════════════════════════════════════════════════════════════

def train(cfg, resume_path=None):
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
    teacher_cfg = cfg.get('teacher', {})

    teacher_mode = teacher_cfg.get('mode', 'cached')

    print(f"\n{'='*70}")
    print(f"  DCFF v5: Feature Field Training")
    print(f"  Experiment: {exp_name}")
    print(f"{'='*70}")
    print(f"  Teacher:        {teacher_mode}")
    print(f"  Dataset:        {dcfg['source_dir']}")
    print(f"  Latent dim:     {mcfg['latent_dim']}")
    print(f"  Feature dim:    {mcfg['feature_dim']}")
    print(f"  Iterations:     {tcfg['iterations']}")
    print(f"  Fine start:     {tcfg['fine_start_iter']}")
    print(f"  Coarse start:   {tcfg['coarse_start_iter']}")
    print(f"{'='*70}\n")

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── 1. Load scene ──
    print("Loading scene...")
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
        load_scene_colmap(dcfg['source_dir'], dcfg.get('images', ''))

    # ── 2. Setup teacher ──
    print("\nSetting up teacher...")

    if teacher_mode == 'online':
        teacher = OnlineRadioTeacher(
            target_dim=mcfg['feature_dim'],
            shallow_block=teacher_cfg.get('shallow_block', 10),
            radio_repo=teacher_cfg.get('radio_repo', '/root/RADIO'),
            pca_init_dir=teacher_cfg.get('pca_init_dir', None),
        ).cuda()
        # Compute feature resolution from actual image dimensions
        longest_edge_cfg = int(tcfg.get('longest_edge', 960))
        sample_cam = train_cams[0]
        img_w, img_h = sample_cam.width, sample_cam.height
        scale = longest_edge_cfg / max(img_w, img_h)
        if scale < 1.0:
            img_w, img_h = int(img_w * scale), int(img_h * scale)
        teacher.set_image_size(img_h, img_w)
        feat_h, feat_w = teacher.feature_resolution
        print(f"  Image size: {img_w}×{img_h} → RADIO features: {feat_w}×{feat_h}")
        radio_cache = None
        cam_to_fid = None
    else:
        teacher = None
        radio_cache = CachedFeatureTeacher(dcfg['feature_dir'])
        feat_h, feat_w = radio_cache.feat_h, radio_cache.feat_w

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

        # Also map test cameras for validation
        for cam in test_cams:
            fid = da3_name_to_fid.get(cam.image_name)
            if fid is not None and fid in radio_cache.frame_ids:
                cam_to_fid[cam.uid] = fid
        n_test_mapped = sum(1 for c in test_cams if c.uid in cam_to_fid)
        print(f"  Matched {len(cam_to_fid) - n_test_mapped}/{len(train_cams)} train + "
              f"{n_test_mapped}/{len(test_cams)} test cameras to features")

    # Coarse feature resolution (half of fine)
    coarse_h = feat_h // 2
    coarse_w = feat_w // 2
    print(f"  Fine resolution:   {feat_w}×{feat_h}")
    print(f"  Coarse resolution: {coarse_w}×{coarse_h}")

    # ── 3. Create models ──
    print("\nInitializing models...")
    init_ply = tcfg.get('init_ply', None)

    gaussians = HybridGaussianModel(
        sh_degree=mcfg['sh_degree'],
        latent_dim=mcfg['latent_dim'],
    )

    train_args = argparse.Namespace(**{
        'position_lr_init': float(tcfg['position_lr_init']),
        'position_lr_final': float(tcfg['position_lr_final']),
        'feature_lr': float(tcfg['feature_lr']),
        'opacity_lr': float(tcfg['opacity_lr']),
        'scaling_lr': float(tcfg['scaling_lr']),
        'rotation_lr': float(tcfg['rotation_lr']),
        'latent_lr': float(tcfg['latent_lr']),
        'percent_dense': float(tcfg['percent_dense']),
        'iterations': tcfg['iterations'],
    })

    # Load pretrained geometry
    freeze_geometry = tcfg.get('freeze_geometry', True)
    if init_ply:
        gaussians.load_ply(init_ply, freeze_geometry=freeze_geometry)
        gaussians.spatial_lr_scale = cameras_extent
        print(f"  Loaded PLY: {init_ply}")
        print(f"  Gaussians: {gaussians.num_points:,}, geometry frozen={freeze_geometry}")
    else:
        gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)
    gaussians.training_setup(train_args)

    bg_color = torch.tensor(
        [1, 1, 1] if mcfg.get('white_background', False) else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    # Scene extent for hash grid
    xyz_init = gaussians.get_xyz.detach().cpu().numpy()
    scene_extent = float(np.percentile(np.linalg.norm(xyz_init, axis=1), 99)) * 1.2
    print(f"  Scene extent: {scene_extent:.2f}")

    # Hash Grid
    hash_grid = SpatialHashGrid(
        scene_extent=scene_extent,
        feature_dim=mcfg['feature_dim'],
        input_mode=hcfg.get('input_mode', 'implicit_scale'),
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
        mlp_hidden=hcfg.get('mlp_hidden', 256),
        mlp_layers=hcfg.get('mlp_layers', 4),
    ).cuda()

    # Deferred Cascaded Renderer
    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid,
        latent_dim=mcfg['latent_dim'],
        fine_feature_dim=mcfg['feature_dim'],
        coarse_feature_dim=mcfg['feature_dim'],
        fine_hidden_dim=fcfg.get('hidden_dim', 256),
        fine_num_layers=fcfg.get('num_layers', 5),
        fine_use_viewdirs=fcfg.get('use_viewdirs', True),
        fine_view_degree=fcfg.get('view_degree', 2),
    ).cuda()

    # Loss
    loss_cfg = tcfg.get('loss', {})
    loss_fn = DCFFLoss(
        lambda_dssim=float(loss_cfg.get('lambda_dssim', 0.2)),
        lambda_fine_cos=float(loss_cfg.get('lambda_fine_cos', 0.65)),
        lambda_fine_l1=float(loss_cfg.get('lambda_fine_l1', 0.2)),
        lambda_coarse_cos=float(loss_cfg.get('lambda_coarse_cos', 0.45)),
        lambda_coarse_l1=float(loss_cfg.get('lambda_coarse_l1', 0.15)),
        lambda_tv=float(loss_cfg.get('lambda_tv', 0.02)),
        lambda_normal=float(loss_cfg.get('lambda_normal', 0.0)),
        lambda_dist=float(loss_cfg.get('lambda_dist', 0.0)),
    )

    # Print param counts
    n_hash = sum(p.numel() for p in hash_grid.parameters())
    n_fine = sum(p.numel() for p in renderer.fine_decoder.parameters())
    n_proj = 0
    if teacher is not None:
        n_proj = sum(p.numel() for p in teacher.get_projection_params())
    print(f"\n  Gaussians:    {gaussians.num_points:,}")
    print(f"  Latent dim:   {mcfg['latent_dim']}")
    print(f"  Hash grid:    {n_hash:,} params")
    print(f"  Fine decoder: {n_fine:,} params")
    if n_proj > 0:
        print(f"  Projections:  {n_proj:,} params")
    print(f"  Total DCFF:   {n_hash + n_fine + n_proj:,} trainable\n")

    # ── 4. Optimizers ──
    dcff_params = [
        {'params': hash_grid.parameters(), 'lr': float(tcfg['lr_hash_grid'])},
        {'params': renderer.fine_decoder.parameters(), 'lr': float(tcfg['lr_fine_decoder'])},
    ]
    if teacher is not None:
        dcff_params.append({
            'params': teacher.get_projection_params(),
            'lr': float(teacher_cfg.get('lr_projection', 0.0005)),
        })

    dcff_optimizer = torch.optim.Adam(dcff_params, eps=1e-15)
    dcff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dcff_optimizer,
        T_max=tcfg['iterations'] - tcfg['fine_start_iter'],
        eta_min=1e-6,
    )

    # ── 5. Resume ──
    start_iteration = 0
    best_metrics = {'total_loss': float('inf')}

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cuda')
        hash_grid.load_state_dict(ckpt['hash_grid_state'])
        renderer.fine_decoder.load_state_dict(ckpt['fine_decoder_state'])
        if 'projection_state' in ckpt and teacher is not None:
            teacher.proj_fine.load_state_dict(ckpt['projection_state']['fine'])
            teacher.proj_coarse.load_state_dict(ckpt['projection_state']['coarse'])
        dcff_optimizer.load_state_dict(ckpt['dcff_optimizer_state'])
        dcff_scheduler.load_state_dict(ckpt['dcff_scheduler_state'])
        start_iteration = ckpt['iteration']
        best_metrics = ckpt.get('best_metrics', best_metrics)
        print(f"  Resumed from iteration {start_iteration}")

    # ── 6. Training loop ──
    longest_edge = tcfg.get('longest_edge', 960)
    batch_size = int(tcfg.get('batch_size', 4))
    total_iters = tcfg['iterations']
    fine_start = tcfg['fine_start_iter']
    coarse_start = tcfg['coarse_start_iter']
    warmup_iters = int(tcfg.get('warmup_iters', 1000))
    grad_clip = float(tcfg.get('grad_clip', 1.0))
    log_every = tcfg.get('log_every', 100)
    save_every = tcfg.get('save_every', 5000)
    val_every = tcfg.get('val_every', 5000)
    sh_up_every = tcfg.get('sh_degree_up_every', 1000)
    coarse_downsample = tcfg.get('coarse_downsample', True)

    # Valid cameras
    if teacher_mode == 'cached':
        valid_cams = [c for c in train_cams if c.uid in cam_to_fid]
    else:
        valid_cams = list(train_cams)
    n_cams = len(valid_cams)

    log_path = os.path.join(output_dir, 'train.log')
    log_f = open(log_path, 'a' if start_iteration > 0 else 'w')

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f"Training {total_iters} iters, {n_cams} cameras, batch={batch_size}")
    log(f"Teacher: {teacher_mode}, feature_dim={mcfg['feature_dim']}, "
        f"latent_dim={mcfg['latent_dim']}")
    log(f"Fine: {feat_w}×{feat_h}, Coarse: {coarse_w}×{coarse_h}")
    log(f"Warmup: {warmup_iters} iters, grad_clip: {grad_clip}")

    start_time = time.time()

    for iteration in range(start_iteration + 1, total_iters + 1):
        # Phase
        if iteration < fine_start:
            phase = 1
        elif iteration < coarse_start:
            phase = 2
        else:
            phase = 3

        gaussians.update_learning_rate(iteration)

        if iteration % sh_up_every == 0:
            gaussians.oneupSHdegree()

        # Store initial LR on first iteration
        if iteration == start_iteration + 1:
            for pg in dcff_optimizer.param_groups:
                pg['initial_lr'] = pg['lr']

        # LR warmup (applied before scheduler)
        if warmup_iters > 0 and iteration <= warmup_iters + start_iteration:
            scale = warmup_lr_scale(iteration - start_iteration, warmup_iters)
            for pg in dcff_optimizer.param_groups:
                if 'initial_lr' in pg:
                    pg['lr'] = pg['initial_lr'] * scale

        # Sample batch
        batch_indices = np.random.randint(n_cams, size=batch_size)
        cams = [valid_cams[idx] for idx in batch_indices]

        # Load GT image for rendering
        gt_image = load_image_batch(cams, longest_edge)
        _, _, img_h, img_w = gt_image.shape

        # Camera matrices
        viewmat = torch.stack([cam_to_viewmat(c) for c in cams], dim=0)
        K = torch.stack([cam_to_K(c, img_w, img_h) for c in cams], dim=0)

        # ── Forward: Render ──
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

        # ── Get teacher targets ──
        fine_target = None
        coarse_target = None

        if phase >= 2:
            if teacher_mode == 'online':
                # Load full-res images for RADIO
                full_images = load_fullres_batch(cams)
                fine_raw, coarse_raw = teacher.extract_raw(full_images)
                fine_proj, coarse_proj = teacher.project(fine_raw, coarse_raw)
                del full_images, fine_raw, coarse_raw

                fine_target = fine_proj  # [B, feat_dim, Hp, Wp]
                if phase >= 3:
                    if coarse_downsample:
                        coarse_target = F.interpolate(
                            coarse_proj, (coarse_h, coarse_w),
                            mode='bilinear', align_corners=False)
                    else:
                        coarse_target = coarse_proj
            else:
                fids = [cam_to_fid[c.uid] for c in cams]
                geo_batch, sem_batch = [], []
                for fid in fids:
                    geo, sem = radio_cache.get(fid)
                    geo_batch.append(geo)
                    sem_batch.append(sem)
                fine_target = torch.stack(geo_batch, dim=0)
                if phase >= 3:
                    coarse_sem = torch.stack(sem_batch, dim=0)
                    if coarse_downsample:
                        coarse_target = F.interpolate(
                            coarse_sem, (coarse_h, coarse_w),
                            mode='bilinear', align_corners=False)
                    else:
                        coarse_target = coarse_sem

        # ── Multi-resolution coarse: downsample prediction too ──
        coarse_pred_for_loss = None
        if phase >= 3 and coarse_downsample:
            coarse_features = render_result.get('coarse_features')
            if coarse_features is not None:
                coarse_pred_for_loss = F.interpolate(
                    coarse_features, (coarse_h, coarse_w),
                    mode='bilinear', align_corners=False)

        # ── Loss computation ──
        # For multi-res coarse: temporarily replace coarse features
        if coarse_pred_for_loss is not None:
            render_result_for_loss = dict(render_result)
            render_result_for_loss['coarse_features'] = coarse_pred_for_loss
        else:
            render_result_for_loss = render_result

        losses = loss_fn.compute(
            render_result=render_result_for_loss,
            gt_rgb=gt_image,
            radio_geo=fine_target,
            radio_sem=coarse_target,
            hash_grid=hash_grid if phase >= 3 else None,
            phase=phase,
        )

        total_loss = losses['total']

        # ── NaN protection ──
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            log(f"  [WARN] NaN/Inf at iter {iteration}, skipping step. "
                f"Components: " + ", ".join(
                    f"{k}={v.item():.4f}" if isinstance(v, torch.Tensor)
                    else f"{k}={v:.4f}" for k, v in losses.items()
                    if k != 'total'))
            gaussians.optimizer.zero_grad(set_to_none=True)
            if phase >= 2:
                dcff_optimizer.zero_grad(set_to_none=True)
            nan_count = nan_count + 1 if 'nan_count' in dir() else 1
            continue

        # ── Backward + optimize ──
        total_loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            all_params = list(hash_grid.parameters()) + \
                         list(renderer.fine_decoder.parameters())
            if teacher is not None:
                all_params += teacher.get_projection_params()
            torch.nn.utils.clip_grad_norm_(all_params, grad_clip)

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if phase >= 2:
                dcff_optimizer.step()
                dcff_optimizer.zero_grad(set_to_none=True)
                dcff_scheduler.step()

        # ── Track best ──
        is_best = losses['total'].item() < best_metrics['total_loss']
        if is_best:
            best_metrics['total_loss'] = losses['total'].item()
            best_metrics['iteration'] = iteration

        # ── Logging ──
        if iteration % log_every == 0:
            elapsed = time.time() - start_time
            elapsed_iters = max(1, iteration - start_iteration)
            it_s = elapsed_iters / max(elapsed, 1e-6)
            img_s = it_s * batch_size

            parts = [f"[Iter {iteration}]", f"P{phase}",
                     f"L={losses['total']:.4f}",
                     f"RGB={losses['rgb']:.4f}"]

            if 'fine_cos' in losses:
                parts.append(f"fine_cos={1 - losses['fine_cos'].item():.3f}")
            if 'coarse_cos' in losses:
                parts.append(f"coarse_cos={1 - losses['coarse_cos'].item():.3f}")
            if 'tv' in losses:
                parts.append(f"TV={losses['tv']:.4f}")

            parts.append(f"B={batch_size}")
            parts.append(f"N={gaussians.num_points:,}")
            parts.append(f"({it_s:.1f} it/s, {img_s:.1f} img/s)")

            msg = ' | '.join(parts)
            if is_best:
                msg += ' ★ BEST'
            log(msg)

        # ── Save best independently (only after coarse starts) ──
        if is_best and iteration >= cfg.get('training', {}).get('coarse_start', 2000):
            best_ckpt = {
                'iteration': iteration,
                'config': cfg,
                'hash_grid_state': hash_grid.state_dict(),
                'fine_decoder_state': renderer.fine_decoder.state_dict(),
                'dcff_optimizer_state': dcff_optimizer.state_dict(),
                'dcff_scheduler_state': dcff_scheduler.state_dict(),
                'best_metrics': best_metrics,
            }
            if teacher is not None:
                best_ckpt['projection_state'] = {
                    'fine': teacher.proj_fine.state_dict(),
                    'coarse': teacher.proj_coarse.state_dict(),
                }
            torch.save(best_ckpt, os.path.join(ckpt_dir, 'best.pth'))
            gaussians.save_ply(os.path.join(ckpt_dir, 'best.ply'))

        # ── Checkpointing ──
        if iteration % save_every == 0 or iteration == total_iters:
            ckpt = {
                'iteration': iteration,
                'config': cfg,
                'hash_grid_state': hash_grid.state_dict(),
                'fine_decoder_state': renderer.fine_decoder.state_dict(),
                'dcff_optimizer_state': dcff_optimizer.state_dict(),
                'dcff_scheduler_state': dcff_scheduler.state_dict(),
                'best_metrics': best_metrics,
            }
            if teacher is not None:
                ckpt['projection_state'] = {
                    'fine': teacher.proj_fine.state_dict(),
                    'coarse': teacher.proj_coarse.state_dict(),
                }
            torch.save(ckpt, os.path.join(ckpt_dir, 'latest.pth'))
            gaussians.save_ply(os.path.join(ckpt_dir, 'latest.ply'))
            log(f"  [Checkpoint] Saved at iter {iteration}")

        # ── Simple validation ──
        if val_every > 0 and iteration % val_every == 0 and test_cams:
            _run_validation(
                iteration, test_cams[:10], renderer, gaussians, hash_grid,
                loss_fn, teacher, radio_cache, cam_to_fid, teacher_mode,
                feat_h, feat_w, coarse_h, coarse_w, longest_edge,
                coarse_downsample, mcfg['feature_dim'], log,
            )

    elapsed = time.time() - start_time
    log(f"\nTraining complete in {elapsed / 3600:.1f}h")
    log(f"Best total loss: {best_metrics['total_loss']:.4f} "
        f"at iter {best_metrics.get('iteration', '?')}")
    log_f.close()


def _run_validation(iteration, test_cams, renderer, gaussians, hash_grid,
                    loss_fn, teacher, radio_cache, cam_to_fid, teacher_mode,
                    feat_h, feat_w, coarse_h, coarse_w, longest_edge,
                    coarse_downsample, feature_dim, log):
    """Quick validation on a subset of test cameras."""
    renderer.eval()
    hash_grid.eval()

    val_losses = []
    n_valid = 0

    with torch.no_grad():
        for cam in test_cams:
            # Skip cams without features in cached mode
            if teacher_mode == 'cached' and cam.uid not in cam_to_fid:
                continue

            gt_image = load_image_tensor(cam, longest_edge)
            _, _, img_h, img_w = gt_image.shape
            viewmat = cam_to_viewmat(cam).unsqueeze(0)
            K_mat = cam_to_K(cam, img_w, img_h).unsqueeze(0)

            result = renderer(
                gaussians, viewmat=viewmat, K=K_mat,
                width=img_w, height=img_h,
                render_coarse=True,
                feature_height=feat_h, feature_width=feat_w,
            )

            if teacher_mode == 'online' and teacher is not None:
                full_img = load_image_fullres(cam)
                fine_raw, coarse_raw = teacher.extract_raw(full_img)
                fine_target, coarse_target = teacher.project(fine_raw, coarse_raw)
                if coarse_downsample:
                    coarse_target = F.interpolate(
                        coarse_target, (coarse_h, coarse_w),
                        mode='bilinear', align_corners=False)
            elif radio_cache is not None and cam.uid in cam_to_fid:
                fid = cam_to_fid[cam.uid]
                geo, sem = radio_cache.get(fid)
                fine_target = geo.unsqueeze(0)
                coarse_sem = sem.unsqueeze(0)
                if coarse_downsample:
                    coarse_target = F.interpolate(
                        coarse_sem, (coarse_h, coarse_w),
                        mode='bilinear', align_corners=False)
                else:
                    coarse_target = coarse_sem
            else:
                continue

            result_for_loss = dict(result)
            if coarse_downsample and 'coarse_features' in result:
                result_for_loss['coarse_features'] = F.interpolate(
                    result['coarse_features'], (coarse_h, coarse_w),
                    mode='bilinear', align_corners=False)

            losses = loss_fn.compute(
                render_result=result_for_loss, gt_rgb=gt_image,
                radio_geo=fine_target, radio_sem=coarse_target,
                hash_grid=None, phase=3,
            )
            val_losses.append(losses['total'].item())
            n_valid += 1

    if val_losses:
        avg = sum(val_losses) / len(val_losses)
        log(f"  [Val {iteration}] avg_loss={avg:.4f} ({n_valid} frames)")

    renderer.train()
    hash_grid.train()


# ═════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    init_ply = cfg.get('training', {}).get('init_ply', None)
    if init_ply:
        cfg['training']['init_ply'] = init_ply

    train(cfg, resume_path=args.resume)
