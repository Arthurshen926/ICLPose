"""
Joint End-to-End Training for ICLPose-loc: Map + Query + Pose.

Three-stage training:
  Stage 1 (Map-only): Train DCFF map features with RADIO teacher (already done)
  Stage 2 (Query-only): Train query network with frozen map features
  Stage 3 (Joint E2E): Joint optimization of map + query + pose

This script handles Stage 2 and 3, resuming from a pretrained DCFF map.

Usage:
    # Stage 2: Query-only (frozen map)
    python -m feature_field.train_joint \
        --config feature_field/configs/dcff_oldhospital_v12_joint.yaml \
        --map_checkpoint feature_field/output/dcff_oldhospital_v11_joint_geo/checkpoints/best.pth \
        --stage query_only

    # Stage 3: Joint E2E
    python -m feature_field.train_joint \
        --config feature_field/configs/dcff_oldhospital_v12_joint.yaml \
        --map_checkpoint feature_field/output/dcff_oldhospital_v11_joint_geo/checkpoints/best.pth \
        --stage joint
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_field.dcff.hybrid_gaussian import HybridGaussianModel
from feature_field.dcff.hash_grid import SpatialHashGrid
from feature_field.dcff.deferred_renderer import DeferredCascadedRenderer
from feature_field.dcff.losses import DCFFLoss
from feature_field.dcff.radio_teacher import CachedFeatureTeacher
from feature_field.models.query_pose_network import (
    QueryPoseNetwork, pose_loss, quaternion_to_rotation_matrix,
    geodesic_rotation_loss,
)
from feature_field.utils.scene_colmap import (
    CameraData, load_scene_colmap, build_da3_image_order,
)
from feature_field.utils.project_config import load_feature_field_config


def load_image_tensor(cam, longest_edge=960):
    img = Image.open(cam.image).convert('RGB')
    W, H = img.size
    scale = longest_edge / max(W, H)
    if scale < 1.0:
        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0).cuda()


def load_fullres_batch(cams):
    return torch.cat([load_image_tensor(c, longest_edge=None) for c in cams], dim=0)


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


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def train_joint(cfg, map_ckpt_path=None, stage='query_only', resume_path=None):
    exp_name = cfg['exp_name']
    output_dir = os.path.join(cfg['output_dir'], exp_name)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    dcfg = cfg['dataset']
    mcfg = cfg['model']
    tcfg = cfg['training']
    qcfg = cfg.get('query', {})
    jcfg = cfg.get('joint', {})

    print(f"\n{'='*70}")
    print(f"  Joint Training: ICLPose-loc")
    print(f"  Experiment: {exp_name}")
    print(f"  Stage: {stage}")
    print(f"{'='*70}")

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # ── 1. Load scene ──
    print("Loading scene...")
    train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = \
        load_scene_colmap(dcfg['source_dir'], dcfg.get('images', ''))

    # Load cached features
    radio_cache = CachedFeatureTeacher(dcfg['feature_dir'])
    feat_h, feat_w = radio_cache.feat_h, radio_cache.feat_w
    coarse_h, coarse_w = feat_h // 2, feat_w // 2

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
    for cam in test_cams:
        fid = da3_name_to_fid.get(cam.image_name)
        if fid is not None and fid in radio_cache.frame_ids:
            cam_to_fid[cam.uid] = fid

    valid_cams = [c for c in train_cams if c.uid in cam_to_fid]
    n_cams = len(valid_cams)
    print(f"  Cameras: {n_cams} train, {len(test_cams)} test")
    print(f"  Features: {feat_w}×{feat_h} fine, {coarse_w}×{coarse_h} coarse")

    # ── 2. Load DCFF map model ──
    print("\nLoading DCFF map model...")
    gaussians = HybridGaussianModel(
        sh_degree=mcfg['sh_degree'],
        latent_dim=mcfg['latent_dim'],
    )

    train_args = argparse.Namespace(**{
        'position_lr_init': float(tcfg.get('position_lr_init', 0.00016)),
        'position_lr_final': float(tcfg.get('position_lr_final', 0.0000016)),
        'feature_lr': float(tcfg.get('feature_lr', 0.0025)),
        'opacity_lr': float(tcfg.get('opacity_lr', 0.05)),
        'scaling_lr': float(tcfg.get('scaling_lr', 0.005)),
        'rotation_lr': float(tcfg.get('rotation_lr', 0.001)),
        'latent_lr': float(tcfg.get('latent_lr', 0.0003)),
        'percent_dense': float(tcfg.get('percent_dense', 0.01)),
        'iterations': tcfg['iterations'],
    })

    freeze_geometry = tcfg.get('freeze_geometry', True)
    init_ply = tcfg.get('init_ply')
    if init_ply and os.path.exists(init_ply):
        gaussians.load_ply(init_ply, freeze_geometry=freeze_geometry)
        gaussians.spatial_lr_scale = cameras_extent
        print(f"  Loaded PLY: {init_ply}")
    else:
        gaussians.create_from_pcd(pcd_xyz, pcd_rgb, cameras_extent)
    gaussians.training_setup(train_args)

    bg_color = torch.tensor(
        [1, 1, 1] if mcfg.get('white_background', False) else [0, 0, 0],
        dtype=torch.float32, device="cuda",
    )

    # Hash Grid
    hash_grid = SpatialHashGrid(
        scene_extent=cameras_extent * 1.2,
        feature_dim=mcfg['feature_dim'],
        input_mode='implicit_scale',
        latent_dim=mcfg['latent_dim'],
        scale_dim=2,
        scale_pe_freqs=4,
        include_raw_scale=True,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=20,
        base_resolution=16,
        max_resolution=4096,
        sh_degree=3,
        mlp_hidden=128,
        mlp_layers=2,
    ).cuda()

    # Renderer
    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid,
        latent_dim=mcfg['latent_dim'],
        fine_feature_dim=mcfg['feature_dim'],
        coarse_feature_dim=mcfg['feature_dim'],
        fine_hidden_dim=128,
        fine_num_layers=5,
        fine_use_viewdirs=False,
        fine_decoder_type='spatial',
        coarse_mode='carrier_residual',
    ).cuda()

    # DCFF Loss
    loss_fn = DCFFLoss(
        lambda_dssim=0.2,
        lambda_fine_cos=1.0,
        lambda_fine_l1=1.0,
        lambda_coarse_cos=1.0,
        lambda_coarse_l1=1.0,
        lambda_tv=0.03,
        lambda_channel_std=0.1,
    )

    # ── 3. Load pretrained map checkpoint ──
    if map_ckpt_path and os.path.exists(map_ckpt_path):
        print(f"  Loading map checkpoint: {map_ckpt_path}")
        ckpt = torch.load(map_ckpt_path, map_location='cuda')
        hash_grid.load_state_dict(ckpt['hash_grid_state'])
        renderer.fine_decoder.load_state_dict(ckpt['fine_decoder_state'])
        if 'coarse_fusion_state' in ckpt:
            renderer.coarse_carrier_fusion.load_state_dict(ckpt['coarse_fusion_state'])
        if 'latent' in ckpt:
            latent = ckpt['latent']
            if isinstance(latent, torch.Tensor):
                gs_latent = gaussians._latent
                n_copy = min(latent.shape[0], gs_latent.shape[0])
                gs_latent.data[:n_copy].copy_(latent[:n_copy].cuda())
        print(f"  Map checkpoint loaded")

    # Freeze map for query-only stage
    map_frozen = (stage == 'query_only')
    if map_frozen:
        print("  Stage: QUERY-ONLY - Map features FROZEN")
        hash_grid.eval()
        for p in hash_grid.parameters():
            p.requires_grad = False
        renderer.eval()
        for p in renderer.parameters():
            p.requires_grad = False
        gaussians.eval()
        for p in gaussians.parameters():
            p.requires_grad = False
    else:
        print("  Stage: JOINT E2E - Map + Query + Pose")

    # ── 4. Create query network ──
    print("\nInitializing query network...")
    query_net = QueryPoseNetwork(
        feature_dim=mcfg['feature_dim'],
        img_size=qcfg.get('img_size', 224),
        matcher_num_heads=qcfg.get('matcher_num_heads', 4),
        matcher_num_layers=qcfg.get('matcher_num_layers', 2),
        pose_hidden_dim=qcfg.get('pose_hidden_dim', 256),
        pose_num_layers=qcfg.get('pose_num_layers', 3),
    ).cuda()

    n_map = sum(p.numel() for p in hash_grid.parameters() if p.requires_grad)
    n_qnet = sum(p.numel() for p in query_net.parameters() if p.requires_grad)
    print(f"  Trainable: Map={n_map:,}, Query={n_qnet:,}, Total={n_map + n_qnet:,}")

    # ── 5. Optimizers ──
    params = []
    if not map_frozen:
        params.extend([
            {'params': hash_grid.parameters(), 'lr': float(tcfg.get('lr_hash_grid', 0.0002))},
            {'params': renderer.fine_decoder.parameters(), 'lr': float(tcfg.get('lr_fine_decoder', 0.0006))},
        ])
    params.append({
        'params': query_net.parameters(),
        'lr': float(qcfg.get('lr', 0.0001)),
    })

    optimizer = torch.optim.AdamW(params, weight_decay=0.01, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg['iterations'], eta_min=1e-6,
    )

    use_amp = tcfg.get('use_amp', True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 6. Resume ──
    start_iter = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location='cuda')
        if 'query_net_state' in ckpt:
            query_net.load_state_dict(ckpt['query_net_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_iter = ckpt.get('iteration', 0)
        print(f"  Resumed from iter {start_iter}")

    # ── 7. Training loop ──
    total_iters = tcfg['iterations']
    batch_size = int(tcfg.get('batch_size', 4))
    log_every = tcfg.get('log_every', 100)
    save_every = tcfg.get('save_every', 5000)
    longest_edge = tcfg.get('longest_edge', 960)
    coarse_downsample = tcfg.get('coarse_downsample', True)

    log_path = os.path.join(output_dir, 'train.log')
    log_f = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_f.write(msg + '\n')
        log_f.flush()

    log(f"Joint training {total_iters} iters, stage={stage}")
    log(f"Map frozen: {map_frozen}, batch={batch_size}")

    # Precompute ground truth poses for all cameras
    print("Precomputing ground truth poses...")
    cam_to_quat = {}
    cam_to_trans = {}
    for cam in train_cams + test_cams:
        W2C = np.eye(4)
        W2C[:3, :3] = cam.R.T
        W2C[:3, 3] = cam.T
        C2W = np.linalg.inv(W2C)
        R_c2w = C2W[:3, :3]
        t_c2w = C2W[:3, 3]
        cam_to_quat[cam.uid] = rotation_to_quaternion(R_c2w)
        cam_to_trans[cam.uid] = t_c2w

    start_time = time.time()

    for iteration in range(start_iter + 1, total_iters + 1):
        query_net.train()
        if not map_frozen:
            hash_grid.train()
            renderer.train()
            gaussians.update_learning_rate(iteration)

        # Sample map and query cameras (different views)
        batch_idx = np.random.randint(n_cams, size=batch_size)
        map_cams = [valid_cams[idx] for idx in batch_idx]

        # For joint training, sample query from different positions
        if not map_frozen and iteration > tcfg.get('joint_start', 5000):
            # Joint: sample query from different camera
            query_idx = np.random.randint(n_cams, size=batch_size)
            # Avoid same camera
            for i in range(batch_size):
                while query_idx[i] == batch_idx[i]:
                    query_idx[i] = np.random.randint(n_cams)
            query_cams = [valid_cams[idx] for idx in query_idx]
        else:
            # Query-only: same cameras but different augmentation
            query_cams = map_cams

        # ── Map rendering ──
        map_imgs = load_image_tensor(map_cams[0], longest_edge) if len(map_cams) == 1 else \
                   torch.cat([load_image_tensor(c, longest_edge) for c in map_cams])
        _, _, img_h, img_w = map_imgs.shape

        with torch.no_grad() if map_frozen else torch.cuda.amp.autocast(enabled=use_amp):
            map_viewmat = torch.stack([cam_to_viewmat(c) for c in map_cams], dim=0)
            map_K = torch.stack([cam_to_K(c, img_w, img_h) for c in map_cams], dim=0)

            map_result = renderer(
                gaussians,
                viewmat=map_viewmat,
                K=map_K,
                width=img_w,
                height=img_h,
                render_coarse=True,
                feature_height=feat_h,
                feature_width=feat_w,
            )

            # Get fine features at feature resolution
            map_feat = map_result['fine_features']
            map_alpha = F.interpolate(map_result['alpha'], map_feat.shape[-2:],
                                      mode='bilinear', align_corners=False)

            # Coarse features
            if coarse_downsample:
                map_coarse = F.interpolate(
                    map_result['coarse_features'], (coarse_h, coarse_w),
                    mode='bilinear', align_corners=False)
            else:
                map_coarse = map_result['coarse_features']

        # ── Query processing ──
        # Resize map features to query image resolution
        q_img_size = qcfg.get('img_size', 224)
        map_feat_resized = F.interpolate(
            map_feat, (q_img_size, q_img_size),
            mode='bilinear', align_corners=False)
        map_mask_resized = (map_alpha > 0.5).float()
        map_mask_resized = F.interpolate(
            map_mask_resized, (q_img_size, q_img_size),
            mode='bilinear', align_corners=False)

        # Load and resize query images
        query_imgs = []
        for c in query_cams:
            img = Image.open(c.image).convert('RGB')
            img = img.resize((q_img_size, q_img_size), Image.LANCZOS)
            query_imgs.append(transforms.ToTensor()(img))
        query_imgs = torch.stack(query_imgs, dim=0).cuda()

        # ── Pose prediction ──
        pose_result = query_net(
            query_imgs,
            map_feat_resized,
            map_mask=map_mask_resized,
        )

        # ── Pose loss ──
        rot_gt = torch.tensor(
            np.stack([cam_to_quat[c.uid] for c in query_cams], axis=0),
            dtype=torch.float32, device='cuda',
        )
        trans_gt = torch.tensor(
            np.stack([cam_to_trans[c.uid] for c in query_cams], axis=0),
            dtype=torch.float32, device='cuda',
        )

        p_loss = pose_loss(
            pose_result['rotation'],
            pose_result['translation'],
            rot_gt,
            trans_gt,
            rotation_weight=jcfg.get('rotation_weight', 1.0),
            translation_weight=jcfg.get('translation_weight', 1.0),
        )

        # ── Feature consistency loss (optional) ──
        feat_cons_loss = torch.tensor(0.0, device='cuda')
        if jcfg.get('use_feature_consistency', False):
            # Query features should be similar to corresponding map features
            matched = pose_result['matched_features']
            query_cls = pose_result['query_cls']
            feat_cons_loss = F.l1_loss(query_cls, matched) * jcfg.get('feat_cons_weight', 0.1)

        total_loss = p_loss['total'] + feat_cons_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            log(f"  [WARN] NaN at iter {iteration}")
            optimizer.zero_grad(set_to_none=True)
            continue

        # ── Backward ──
        scaler.scale(total_loss).backward()

        if tcfg.get('grad_clip', 0) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in query_net.parameters() if p.requires_grad], tcfg['grad_clip'])

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        # ── Logging ──
        if iteration % log_every == 0:
            elapsed = time.time() - start_time
            it_s = max(1, iteration - start_iter) / max(elapsed, 1e-6)
            rot_err = p_loss['rotation_loss'].item()
            trans_err = p_loss['translation_loss'].item()
            rot_geodesic = geodesic_rotation_loss(pose_result['rotation'], rot_gt).item()
            trans_l1 = F.l1_loss(pose_result['translation'], trans_gt).item()

            log(
                f"[Iter {iteration}] L={total_loss.item():.4f} | "
                f"rot={rot_err:.4f} (geo={rot_geodesic:.3f}°) | "
                f"trans={trans_err:.4f} (L1={trans_l1:.3f}m) | "
                f"({it_s:.1f} it/s)"
            )

        # ── Save ──
        if iteration % save_every == 0 or iteration == total_iters:
            ckpt = {
                'iteration': iteration,
                'config': cfg,
                'query_net_state': query_net.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(ckpt_dir, 'latest.pth'))
            log(f"  [Checkpoint] Saved at iter {iteration}")

    elapsed = time.time() - start_time
    log(f"\nTraining complete in {elapsed/3600:.1f}h")
    log_f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--map_checkpoint', default=None)
    parser.add_argument('--stage', default='query_only',
                       choices=['query_only', 'joint'])
    parser.add_argument('--resume', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = load_feature_field_config(args.config)
    train_joint(cfg, map_ckpt_path=args.map_checkpoint,
                stage=args.stage, resume_path=args.resume)
