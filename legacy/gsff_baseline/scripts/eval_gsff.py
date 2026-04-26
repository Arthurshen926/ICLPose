#!/usr/bin/env python3
"""
GSFFs Pose Refinement Evaluation on OldHospital.

For each test image:
  1. Initialize pose from nearest training image (by image name proximity)
  2. Extract 2D features with trained encoder
  3. Coarse refinement (R=256 triplane, ViT-level features)
  4. Fine refinement (R=1024 triplane, pixel-level features)
  5. Report median position/rotation errors

Paper target: 21cm / 0.41° (Feature), 18cm / 0.36° (Feature tuned)

Usage:
    CUDA_VISIBLE_DEVICES=3 python legacy/gsff_baseline/scripts/eval_gsff.py \\
        --checkpoint output/gsff/OldHospital/checkpoints/best.pth \\
        --source_dir dataset/OldHospital \\
        --model_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \\
        --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder
from gsff.pose_refine import refine_pose, refine_pose_with_rgb


# ── Pose utilities ──────────────────────────────────────────────────────────

def pose_error(pred_w2c: torch.Tensor, gt_w2c: torch.Tensor):
    """
    Compute position (cm) and rotation (deg) errors between predicted and GT w2c poses.

    Returns:
        pos_err_cm: position error in centimeters
        rot_err_deg: rotation error in degrees
    """
    pred_c2w = torch.inverse(pred_w2c)
    gt_c2w = torch.inverse(gt_w2c)

    # Position error: ||t_pred - t_gt|| in meters → cm
    pos_err = torch.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3]).item() * 100.0

    # Rotation error: arccos((tr(R_rel) - 1) / 2)
    R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
    trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    rot_err = torch.acos(cos_angle).item() * 180.0 / math.pi

    return pos_err, rot_err


def find_nearest_train_pose(test_cam, train_cams, train_positions):
    """Find nearest training camera by 3D position (proxy for DenseVLAD).
    
    Args:
        test_cam: dict with 'position' key
        train_cams: list of camera dicts
        train_positions: [N_train, 3] numpy array of training positions
        
    Returns:
        nearest camera dict
    """
    test_pos = np.array(test_cam['position'])
    dists = np.linalg.norm(train_positions - test_pos[None, :], axis=1)
    nearest_idx = np.argmin(dists)
    return train_cams[nearest_idx]


def find_topk_nearest_train_poses(test_cam, train_cams, train_positions, k=5):
    """Find k nearest training cameras by 3D position."""
    test_pos = np.array(test_cam['position'])
    dists = np.linalg.norm(train_positions - test_pos[None, :], axis=1)
    top_indices = np.argsort(dists)[:k]
    return [train_cams[i] for i in top_indices]


def build_dino_retrieval_db(backbone, train_images, train_cams, source_dir, 
                             render_w, render_h, device, MEAN, STD):
    """
    Pre-compute DINOv2 CLS token descriptors for all training images.
    Returns: (descriptors [N, 768], train_cams list)
    """
    print("Building DINOv2 retrieval database...")
    descriptors = []
    for cam in tqdm(train_cams, desc="  Encoding train images"):
        img_name = cam['img_name']
        img_path = source_dir / img_name
        if not img_path.exists():
            img_path = source_dir / 'processed' / img_name
        img = Image.open(img_path).convert('RGB').resize((render_w, render_h), Image.BILINEAR)
        img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        img_norm = (img_t.unsqueeze(0).to(device) - MEAN) / STD
        
        with torch.no_grad():
            # Pad to patch_size=14 multiples
            ps = 14
            H, W = img_norm.shape[2], img_norm.shape[3]
            pad_h = (ps - H % ps) % ps
            pad_w = (ps - W % ps) % ps
            if pad_h > 0 or pad_w > 0:
                img_norm = F.pad(img_norm, (0, pad_w, 0, pad_h), mode='reflect')
            out = backbone.forward_features(img_norm)
            cls_token = out['x_norm_clstoken']  # [1, 768]
            descriptors.append(cls_token.cpu())
    
    descriptors = torch.cat(descriptors, dim=0)  # [N, 768]
    descriptors = F.normalize(descriptors, p=2, dim=1)
    print(f"  Built DB: {descriptors.shape[0]} images, {descriptors.shape[1]}-d")
    return descriptors


def retrieve_nearest_by_dino(query_img_norm, backbone, db_descriptors, train_cams, device):
    """Find nearest training camera using DINOv2 CLS token similarity.
    Uses top-K visual matches and picks the one with median position to be robust.
    """
    ps = 14
    H, W = query_img_norm.shape[2], query_img_norm.shape[3]
    pad_h = (ps - H % ps) % ps
    pad_w = (ps - W % ps) % ps
    img_padded = query_img_norm
    if pad_h > 0 or pad_w > 0:
        img_padded = F.pad(query_img_norm, (0, pad_w, 0, pad_h), mode='reflect')
    
    with torch.no_grad():
        out = backbone.forward_features(img_padded)
        q_desc = F.normalize(out['x_norm_clstoken'], p=2, dim=1)  # [1, 768]
    
    sims = (q_desc.cpu() @ db_descriptors.T).squeeze(0)  # [N]
    # Use top-10 visual matches, pick closest by position to be robust
    top_k = min(10, len(train_cams))
    top_indices = sims.topk(top_k).indices.tolist()
    top_cams = [train_cams[i] for i in top_indices]
    # Return the one whose position is closest to the median position of top matches
    positions = np.array([c['position'] for c in top_cams])
    median_pos = np.median(positions, axis=0)
    dists_to_median = np.linalg.norm(positions - median_pos[None, :], axis=1)
    best_local = np.argmin(dists_to_median)
    return top_cams[best_local]


# ── Main evaluation ─────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device('cuda')

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_args = argparse.Namespace(**ckpt['args'])

    # Load Gaussian model
    print("Loading 2DGS model...")
    gs_model = GaussianFeatureModel(feature_dim=ckpt_args.feature_dim)
    gs_model.load_ply(args.model_path)
    gs_model = gs_model.to(device)
    gs_model.eval()

    means3d = gs_model.get_xyz.detach()
    quats = gs_model.get_rotation.detach()
    scales_raw = gs_model.get_scaling
    scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1).detach()
    opacities = gs_model.get_opacity.squeeze(-1).detach()

    scene_extent = ckpt['scene_extent']

    # Load triplane
    triplane = DualScaleTriplane(
        coarse_resolution=ckpt_args.coarse_resolution,
        fine_resolution=ckpt_args.fine_resolution,
        feature_dim=ckpt_args.feature_dim,
        scene_extent=scene_extent,
    ).to(device)
    triplane.load_state_dict(ckpt['triplane'])
    triplane.eval()

    # Load encoder (detect old vs new architecture)
    encoder_state = ckpt['encoder']
    use_old = any('fine_encoder.encoder.' in k for k in encoder_state.keys())
    encoder = DualScaleEncoder(
        feature_dim=ckpt_args.feature_dim,
        freeze_backbone=True,
        use_old_fine_encoder=use_old,
    ).to(device)
    encoder.load_state_dict(encoder_state)
    if use_old:
        print("  Using old FineEncoder (v1 checkpoint)")
    encoder.eval()

    # Load cameras
    with open(args.cameras_json) as f:
        all_cams = json.load(f)

    # Load test split
    source_dir = Path(args.source_dir)
    test_file = source_dir / 'dataset_test.txt'
    test_samples = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Visual') or line.startswith('ImageFile'):
                continue
            parts = line.split()
            test_samples.append({
                'img_name': parts[0],
                'position': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                'quat': np.array([float(parts[4]), float(parts[5]),
                                  float(parts[6]), float(parts[7])]),
            })

    # Build train camera list for retrieval
    train_file = source_dir / 'dataset_train.txt'
    train_names = set()
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Visual') or line.startswith('ImageFile'):
                continue
            train_names.add(line.split()[0])

    train_cams = [c for c in all_cams if c['img_name'] in train_names]
    train_positions = np.array([c['position'] for c in train_cams])
    cam_by_name = {c['img_name']: c for c in all_cams}

    # Camera intrinsics
    first_cam = all_cams[0]
    orig_w, orig_h = first_cam['width'], first_cam['height']
    render_h = args.render_height
    render_w = args.render_width

    fx = first_cam['fx'] * render_w / orig_w
    fy = first_cam['fy'] * render_h / orig_h

    K_mat = torch.zeros(3, 3, device=device)
    K_mat[0, 0] = fx
    K_mat[1, 1] = fy
    K_mat[0, 2] = render_w / 2.0
    K_mat[1, 2] = render_h / 2.0
    K_mat[2, 2] = 1.0

    # Coarse intrinsics
    coarse_h = render_h // 14
    coarse_w = render_w // 14
    K_coarse = K_mat.clone()
    K_coarse[0] *= coarse_w / render_w
    K_coarse[1] *= coarse_h / render_h

    # Fine intrinsics
    fine_h = min(render_h, args.fine_render_height)
    fine_w = min(render_w, args.fine_render_width)
    K_fine = K_mat.clone()
    K_fine[0] *= fine_w / render_w
    K_fine[1] *= fine_h / render_h

    # ImageNet normalization
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Build DINOv2 retrieval database (replaces position-NN)
    if args.use_dino_retrieval:
        dino_backbone = encoder.coarse_encoder.backbone
        db_descriptors = build_dino_retrieval_db(
            dino_backbone, None, train_cams, source_dir,
            render_w, render_h, device, MEAN, STD,
        )

    # Pre-extract triplane features
    print("Extracting triplane features...")
    with torch.no_grad():
        coarse_colors = triplane.extract_coarse(means3d)
        coarse_colors = F.normalize(coarse_colors, p=2, dim=1)
        fine_colors = triplane.extract_fine(means3d)
        fine_colors = F.normalize(fine_colors, p=2, dim=1)

    # Pre-compute RGB colors for photometric loss (SH degree 0 = constant)
    rgb_colors = None
    if args.rgb_weight > 0:
        print("Extracting Gaussian RGB colors (SH degree 0)...")
        sh0 = gs_model._features_dc.reshape(-1, 1, 3)  # [N, 1, 3]
        rgb_colors = (sh0.squeeze(1) + 0.5).clamp(0, 1).detach()  # [N, 3]

    # Evaluate
    print(f"\nEvaluating {len(test_samples)} test images...")
    pos_errors = []
    rot_errors = []

    for sample in tqdm(test_samples):
        img_name = sample['img_name']

        # Load and preprocess image
        img_path = source_dir / img_name
        if not img_path.exists():
            img_path = source_dir / 'processed' / img_name
        img = Image.open(img_path).convert('RGB')
        img = img.resize((render_w, render_h), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        img_norm = ((img_tensor.unsqueeze(0).to(device)) - MEAN) / STD

        # GT pose (from cameras.json which uses COLMAP c2w convention)
        gt_c2w = np.eye(4, dtype=np.float32)
        cam = cam_by_name[img_name]
        gt_c2w[:3, :3] = np.array(cam['rotation'], dtype=np.float32)
        gt_c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
        gt_w2c = torch.from_numpy(np.linalg.inv(gt_c2w).astype(np.float32)).to(device)

        # Initial pose: DINOv2 retrieval or position-NN
        if args.use_dino_retrieval:
            nn_cam = retrieve_nearest_by_dino(
                img_norm, dino_backbone, db_descriptors, train_cams, device,
            )
            init_c2w = np.eye(4, dtype=np.float32)
            init_c2w[:3, :3] = np.array(nn_cam['rotation'], dtype=np.float32)
            init_c2w[:3, 3] = np.array(nn_cam['position'], dtype=np.float32)
            init_w2c = torch.from_numpy(np.linalg.inv(init_c2w).astype(np.float32)).to(device)
            init_candidates = [init_w2c]
        elif args.multi_start > 1:
            top_cams = find_topk_nearest_train_poses(
                cam, train_cams, train_positions, k=args.multi_start)
            init_candidates = []
            for nc in top_cams:
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = np.array(nc['rotation'], dtype=np.float32)
                c2w[:3, 3] = np.array(nc['position'], dtype=np.float32)
                init_candidates.append(
                    torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device))
        else:
            nn_cam = find_nearest_train_pose(cam, train_cams, train_positions)
            init_c2w = np.eye(4, dtype=np.float32)
            init_c2w[:3, :3] = np.array(nn_cam['rotation'], dtype=np.float32)
            init_c2w[:3, 3] = np.array(nn_cam['position'], dtype=np.float32)
            init_w2c = torch.from_numpy(np.linalg.inv(init_c2w).astype(np.float32)).to(device)
            init_candidates = [init_w2c]

        # Extract 2D features
        with torch.no_grad():
            coarse_feat_2d, fine_feat_2d = encoder(img_norm)
            coarse_feat_2d = F.normalize(coarse_feat_2d, p=2, dim=1)
            fine_feat_2d = F.normalize(fine_feat_2d, p=2, dim=1)

        # Resize to render resolution
        coarse_feat_2d = F.interpolate(
            coarse_feat_2d, (coarse_h, coarse_w),
            mode='bilinear', align_corners=False,
        )
        fine_feat_2d = F.interpolate(
            fine_feat_2d, (fine_h, fine_w),
            mode='bilinear', align_corners=False,
        )

        # Multi-start: run refinement from each candidate, pick lowest fine loss
        best_refined = None
        best_fine_loss = float('inf')

        # Prepare RGB target for photometric loss
        rgb_fine_target = None
        if args.rgb_weight > 0:
            rgb_fine_target = F.interpolate(
                img_tensor.unsqueeze(0).to(device),
                (fine_h, fine_w), mode='bilinear', align_corners=False,
            )

        for init_w2c in init_candidates:
            current_pose = init_w2c
            for rnd in range(args.rounds):
                lr_scale = 1.0  # keep constant LR across rounds
                # Stage 1: Coarse refinement
                current_pose = refine_pose(
                    coarse_feat_2d, means3d, quats, scales, opacities,
                    coarse_colors, current_pose, K_coarse, coarse_w, coarse_h,
                    n_iters=args.coarse_iters, lr=args.coarse_lr * lr_scale,
                    chunk_size=16, lr_decay=args.lr_decay,
                    loss_type=args.loss_type,
                    reset_interval=args.reset_interval,
                    trans_lr_scale=args.trans_lr_scale,
                )
                # Stage 2: Fine refinement (with optional RGB loss)
                if args.rgb_weight > 0 and rgb_colors is not None:
                    current_pose = refine_pose_with_rgb(
                        fine_feat_2d, rgb_fine_target,
                        means3d, quats, scales, opacities,
                        fine_colors, rgb_colors,
                        current_pose, K_fine, fine_w, fine_h,
                        n_iters=args.fine_iters, lr=args.fine_lr * lr_scale,
                        rgb_weight=args.rgb_weight,
                        chunk_size=16, loss_type=args.loss_type,
                    )
                else:
                    current_pose = refine_pose(
                        fine_feat_2d, means3d, quats, scales, opacities,
                        fine_colors, current_pose, K_fine, fine_w, fine_h,
                        n_iters=args.fine_iters, lr=args.fine_lr * lr_scale,
                        chunk_size=16, lr_decay=args.lr_decay,
                    loss_type=args.loss_type,
                    reset_interval=args.reset_interval,
                    tune_features=args.tune_features,
                    feature_lr=args.feature_lr * lr_scale,
                    trans_lr_scale=args.trans_lr_scale,
                )

            # Evaluate fine loss for this candidate
            with torch.no_grad():
                from gsff.pose_refine import render_features_for_pose
                rendered = render_features_for_pose(
                    means3d, quats, scales, opacities, fine_colors,
                    current_pose.unsqueeze(0), K_fine.unsqueeze(0), fine_w, fine_h)
                rendered_n = F.normalize(rendered, p=2, dim=1)
                fine_loss = F.mse_loss(rendered_n, fine_feat_2d).item()

            if fine_loss < best_fine_loss:
                best_fine_loss = fine_loss
                best_refined = current_pose

        refined_w2c = best_refined

        # Compute error
        pos_err, rot_err = pose_error(refined_w2c, gt_w2c)
        pos_errors.append(pos_err)
        rot_errors.append(rot_err)

    # Report results
    pos_errors = np.array(pos_errors)
    rot_errors = np.array(rot_errors)

    print(f"\n{'='*60}")
    print(f"GSFFs-PR Feature Results on OldHospital")
    print(f"{'='*60}")
    print(f"  Median position error: {np.median(pos_errors):.1f} cm")
    print(f"  Median rotation error: {np.median(rot_errors):.2f}°")
    print(f"  Mean position error:   {np.mean(pos_errors):.1f} cm")
    print(f"  Mean rotation error:   {np.mean(rot_errors):.2f}°")
    print(f"  90th %ile position:    {np.percentile(pos_errors, 90):.1f} cm")
    print(f"  90th %ile rotation:    {np.percentile(rot_errors, 90):.2f}°")
    print(f"{'='*60}")
    print(f"  Paper target: 21 cm / 0.41°")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(os.path.dirname(args.checkpoint), '..', 'eval_results.json')
    results = {
        'median_pos_cm': float(np.median(pos_errors)),
        'median_rot_deg': float(np.median(rot_errors)),
        'mean_pos_cm': float(np.mean(pos_errors)),
        'mean_rot_deg': float(np.mean(rot_errors)),
        'p90_pos_cm': float(np.percentile(pos_errors, 90)),
        'p90_rot_deg': float(np.percentile(rot_errors, 90)),
        'n_test': len(test_samples),
        'per_image': [{'pos_cm': float(p), 'rot_deg': float(r)} 
                      for p, r in zip(pos_errors, rot_errors)],
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='GSFFs Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cameras_json', type=str, required=True)
    parser.add_argument('--render_height', type=int, default=540)
    parser.add_argument('--render_width', type=int, default=960)
    parser.add_argument('--fine_render_height', type=int, default=270)
    parser.add_argument('--fine_render_width', type=int, default=480)
    parser.add_argument('--coarse_iters', type=int, default=300)
    parser.add_argument('--fine_iters', type=int, default=300)
    parser.add_argument('--coarse_lr', type=float, default=0.01)
    parser.add_argument('--fine_lr', type=float, default=0.005)
    parser.add_argument('--lr_decay', type=float, default=1.0,
                        help='Per-step LR decay factor (e.g. 0.998)')
    parser.add_argument('--multi_start', type=int, default=1,
                        help='Number of nearest-neighbor init poses to try (pick best)')
    parser.add_argument('--rounds', type=int, default=1,
                        help='Number of coarse-fine iteration rounds')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'huber'],
                        help='Loss function for refinement')
    parser.add_argument('--use_dino_retrieval', action='store_true',
                        help='Use DINOv2 CLS token retrieval instead of position-NN')
    parser.add_argument('--reset_interval', type=int, default=0,
                        help='Fold delta_xi into viewmat every N steps (0=disabled)')
    parser.add_argument('--tune_features', action='store_true',
                        help='Enable "Feature tuned" mode: jointly optimize features + pose')
    parser.add_argument('--feature_lr', type=float, default=0.001,
                        help='Learning rate for feature residual in tune_features mode')
    parser.add_argument('--rgb_weight', type=float, default=0.0,
                        help='Weight for RGB photometric loss during fine refinement (0=disabled)')
    parser.add_argument('--trans_lr_scale', type=float, default=1.0,
                        help='Scale factor for translation LR relative to rotation LR')
    return parser.parse_args()


if __name__ == '__main__':
    main()
