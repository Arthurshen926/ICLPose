#!/usr/bin/env python3
"""
Visualize GSFF SOTA intermediate quantities for a test image:
  1. Query image + initial pose rendering
  2. Coarse 2D encoder features (PCA colorized)
  3. Fine 2D encoder features (PCA colorized)
  4. Triplane-rendered coarse 3D features @ GT pose
  5. Triplane-rendered fine 3D features @ GT pose
  6. Coarse feature cosine similarity map (2D vs 3D @ GT)
  7. Fine feature cosine similarity map
  8. RGB rendering @ GT pose
  9. Optimization trajectory: rendered features over refinement steps

Usage:
    CUDA_VISIBLE_DEVICES=2 python legacy/gsff_baseline/scripts/visualize_gsff_intermediates.py \
        --checkpoint output/gsff/OldHospital/checkpoints/final.pth \
        --source_dir dataset/OldHospital \
        --model_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json \
        --sample_indices 0 9 47 100 \
        --output_dir output/gsff_vis
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

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
from gsff.pose_refine import (
    refine_pose, render_features_for_pose, se3_exp,
    _transform_gaussians, _render_with_transformed_gaussians
)
from gsplat import rasterization_2dgs


def pca_colorize(features, n_components=3):
    """
    PCA-based colorization of feature maps.
    features: [D, H, W] tensor -> returns [H, W, 3] numpy array in [0,1]
    """
    D, H, W = features.shape
    feat_flat = features.reshape(D, -1).T.cpu().numpy()  # [H*W, D]
    pca = PCA(n_components=n_components)
    rgb = pca.fit_transform(feat_flat)  # [H*W, 3]
    # normalize to [0, 1]
    rgb = (rgb - rgb.min(axis=0)) / (rgb.max(axis=0) - rgb.min(axis=0) + 1e-8)
    return rgb.reshape(H, W, 3)


def cosine_sim_map(feat_2d, feat_3d):
    """
    Compute per-pixel cosine similarity.
    feat_2d, feat_3d: [1, D, H, W]
    Returns: [H, W] numpy array
    """
    sim = F.cosine_similarity(feat_2d, feat_3d, dim=1)  # [1, H, W]
    return sim.squeeze(0).cpu().numpy()


def render_rgb_at_pose(gs_model, viewmat, K_mat, width, height):
    """Render RGB from 2DGS at given pose."""
    from gsplat import spherical_harmonics
    means3d = gs_model.get_xyz
    quats = gs_model.get_rotation
    scales_raw = gs_model.get_scaling
    scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1)
    opacities = gs_model.get_opacity.squeeze(-1)
    
    sh0 = gs_model._features_dc.reshape(-1, 1, 3)
    colors = (sh0.squeeze(1) + 0.5).clamp(0, 1)
    
    render_colors, render_alphas, *_ = rasterization_2dgs(
        means=means3d, quats=quats, scales=scales, opacities=opacities,
        colors=colors, viewmats=viewmat.unsqueeze(0), Ks=K_mat.unsqueeze(0),
        width=width, height=height, packed=False,
        near_plane=0.01, far_plane=1e5, render_mode='RGB',
    )
    rgb = render_colors[0].permute(2, 0, 1).clamp(0, 1)  # [3, H, W]
    return rgb


def find_nearest_train_pose(test_cam, train_cams, train_positions):
    test_pos = np.array(test_cam['position'])
    dists = np.linalg.norm(train_positions - test_pos[None, :], axis=1)
    return train_cams[np.argmin(dists)]


def make_viewmat(cam):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(cam['rotation'], dtype=np.float32)
    c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
    return torch.from_numpy(np.linalg.inv(c2w).astype(np.float32))


def pose_error(pred_w2c, gt_w2c):
    pred_c2w = torch.inverse(pred_w2c)
    gt_c2w = torch.inverse(gt_w2c)
    pos_err = torch.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3]).item() * 100.0
    R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
    trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_err = torch.acos(cos_angle).item() * 180.0 / math.pi
    return pos_err, rot_err


def visualize_sample(
    idx, sample, gs_model, triplane, encoder, train_cams, train_positions,
    cam_by_name, source_dir, device, K_mat, K_coarse, K_fine,
    render_w, render_h, coarse_h, coarse_w, fine_h, fine_w,
    coarse_colors, fine_colors, output_dir, MEAN, STD, args,
):
    img_name = sample['img_name']
    print(f"\n[{idx}] Processing {img_name}...")
    
    # Load image
    img_path = source_dir / img_name
    if not img_path.exists():
        img_path = source_dir / 'processed' / img_name
    img = Image.open(img_path).convert('RGB').resize((render_w, render_h), Image.BILINEAR)
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    img_norm = ((img_tensor.unsqueeze(0).to(device)) - MEAN) / STD
    
    # GT pose
    cam = cam_by_name[img_name]
    gt_w2c = make_viewmat(cam).to(device)
    
    # Initial pose
    nn_cam = find_nearest_train_pose(cam, train_cams, train_positions)
    init_w2c = make_viewmat(nn_cam).to(device)
    init_pos, init_rot = pose_error(init_w2c, gt_w2c)
    
    means3d = gs_model.get_xyz.detach()
    quats = gs_model.get_rotation.detach()
    scales_raw = gs_model.get_scaling
    scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1).detach()
    opacities = gs_model.get_opacity.squeeze(-1).detach()
    
    # 2D encoder features
    with torch.no_grad():
        coarse_2d, fine_2d = encoder(img_norm)
        coarse_2d_norm = F.normalize(coarse_2d, p=2, dim=1)
        fine_2d_norm = F.normalize(fine_2d, p=2, dim=1)
    
    coarse_2d_resized = F.interpolate(coarse_2d_norm, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
    fine_2d_resized = F.interpolate(fine_2d_norm, (fine_h, fine_w), mode='bilinear', align_corners=False)
    
    # 3D features at GT pose
    with torch.no_grad():
        coarse_3d_gt = render_features_for_pose(
            means3d, quats, scales, opacities, coarse_colors,
            gt_w2c.unsqueeze(0), K_coarse.unsqueeze(0), coarse_w, coarse_h)
        coarse_3d_gt_norm = F.normalize(coarse_3d_gt, p=2, dim=1)
        
        fine_3d_gt = render_features_for_pose(
            means3d, quats, scales, opacities, fine_colors,
            gt_w2c.unsqueeze(0), K_fine.unsqueeze(0), fine_w, fine_h)
        fine_3d_gt_norm = F.normalize(fine_3d_gt, p=2, dim=1)
    
    # 3D features at init pose
    with torch.no_grad():
        coarse_3d_init = render_features_for_pose(
            means3d, quats, scales, opacities, coarse_colors,
            init_w2c.unsqueeze(0), K_coarse.unsqueeze(0), coarse_w, coarse_h)
        coarse_3d_init_norm = F.normalize(coarse_3d_init, p=2, dim=1)
    
    # RGB at GT and init pose
    with torch.no_grad():
        rgb_gt = render_rgb_at_pose(gs_model, gt_w2c, K_mat, render_w, render_h)
        rgb_init = render_rgb_at_pose(gs_model, init_w2c, K_mat, render_w, render_h)
    
    # Run coarse refinement and track trajectory
    coarse_refined = refine_pose(
        coarse_2d_resized, means3d, quats, scales, opacities,
        coarse_colors, init_w2c, K_coarse, coarse_w, coarse_h,
        n_iters=300, lr=0.01, chunk_size=16, trans_lr_scale=args.trans_lr_scale,
    )
    coarse_pos, coarse_rot = pose_error(coarse_refined, gt_w2c)
    
    # Fine refinement
    fine_2d_resized_for_refine = F.interpolate(fine_2d_norm, (fine_h, fine_w), mode='bilinear', align_corners=False)
    
    fine_refined = refine_pose(
        fine_2d_resized_for_refine, means3d, quats, scales, opacities,
        fine_colors, coarse_refined, K_fine, fine_w, fine_h,
        n_iters=300, lr=0.005, chunk_size=16, trans_lr_scale=args.trans_lr_scale,
    )
    final_pos, final_rot = pose_error(fine_refined, gt_w2c)
    
    # 3D features at refined pose
    with torch.no_grad():
        fine_3d_refined = render_features_for_pose(
            means3d, quats, scales, opacities, fine_colors,
            fine_refined.unsqueeze(0), K_fine.unsqueeze(0), fine_w, fine_h)
        fine_3d_refined_norm = F.normalize(fine_3d_refined, p=2, dim=1)
        
        rgb_refined = render_rgb_at_pose(gs_model, fine_refined, K_mat, render_w, render_h)
    
    # Cosine similarity maps
    cos_coarse_gt = cosine_sim_map(coarse_2d_resized, coarse_3d_gt_norm)
    cos_fine_gt = cosine_sim_map(fine_2d_resized, fine_3d_gt_norm)
    cos_coarse_init = cosine_sim_map(coarse_2d_resized, coarse_3d_init_norm)
    cos_fine_refined = cosine_sim_map(fine_2d_resized, fine_3d_refined_norm)
    
    # PCA colorization
    coarse_2d_pca = pca_colorize(coarse_2d_resized.squeeze(0))
    fine_2d_pca = pca_colorize(fine_2d_resized.squeeze(0))
    coarse_3d_gt_pca = pca_colorize(coarse_3d_gt_norm.squeeze(0))
    fine_3d_gt_pca = pca_colorize(fine_3d_gt_norm.squeeze(0))
    coarse_3d_init_pca = pca_colorize(coarse_3d_init_norm.squeeze(0))
    fine_3d_refined_pca = pca_colorize(fine_3d_refined_norm.squeeze(0))
    
    # ── Figure 1: Main panel (4x3 grid) ──
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.25, wspace=0.15)
    
    # Row 1: Query image, RGB@GT, RGB@Init, RGB@Refined
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_np); ax.set_title('Query Image', fontsize=11)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(rgb_gt.permute(1,2,0).cpu().numpy()); ax.set_title('RGB @ GT Pose', fontsize=11)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(rgb_init.permute(1,2,0).cpu().numpy())
    ax.set_title(f'RGB @ Init ({init_pos:.0f}cm, {init_rot:.1f}°)', fontsize=11)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(rgb_refined.permute(1,2,0).cpu().numpy())
    ax.set_title(f'RGB @ Refined ({final_pos:.1f}cm, {final_rot:.2f}°)', fontsize=11)
    ax.axis('off')
    
    # Row 2: Coarse features (2D encoder, 3D@GT, 3D@Init, CosSim)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(coarse_2d_pca); ax.set_title(f'Coarse 2D Encoder\n{coarse_2d_resized.shape[1]}d @ {coarse_h}x{coarse_w}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(coarse_3d_gt_pca); ax.set_title('Coarse 3D @ GT', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(coarse_3d_init_pca); ax.set_title('Coarse 3D @ Init', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(cos_coarse_gt, vmin=0, vmax=1, cmap='viridis')
    ax.set_title(f'Coarse CosSim @ GT\nmean={cos_coarse_gt.mean():.3f}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 3: Fine features (2D encoder, 3D@GT, 3D@Refined, CosSim)
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(fine_2d_pca); ax.set_title(f'Fine 2D Encoder\n{fine_2d_resized.shape[1]}d @ {fine_h}x{fine_w}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(fine_3d_gt_pca); ax.set_title('Fine 3D @ GT', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(fine_3d_refined_pca); ax.set_title('Fine 3D @ Refined', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 3])
    im = ax.imshow(cos_fine_gt, vmin=0, vmax=1, cmap='viridis')
    ax.set_title(f'Fine CosSim @ GT\nmean={cos_fine_gt.mean():.3f}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 4: Cosine sim comparison (Init vs GT, Refined vs GT)
    ax = fig.add_subplot(gs[3, 0])
    im = ax.imshow(cos_coarse_init, vmin=0, vmax=1, cmap='RdYlGn')
    ax.set_title(f'Coarse CosSim @ Init\nmean={cos_coarse_init.mean():.3f}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(gs[3, 1])
    im = ax.imshow(cos_coarse_gt, vmin=0, vmax=1, cmap='RdYlGn')
    ax.set_title(f'Coarse CosSim @ GT\nmean={cos_coarse_gt.mean():.3f}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(gs[3, 2])
    im = ax.imshow(cos_fine_refined, vmin=0, vmax=1, cmap='RdYlGn')
    ax.set_title(f'Fine CosSim @ Refined\nmean={cos_fine_refined.mean():.3f}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(gs[3, 3])
    im = ax.imshow(cos_fine_gt, vmin=0, vmax=1, cmap='RdYlGn')
    ax.set_title(f'Fine CosSim @ GT\nmean={cos_fine_gt.mean():.3f}', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(
        f'{img_name}\n'
        f'Init: {init_pos:.0f}cm/{init_rot:.1f}° → Coarse: {coarse_pos:.1f}cm/{coarse_rot:.2f}° → Final: {final_pos:.1f}cm/{final_rot:.2f}°',
        fontsize=14, fontweight='bold'
    )
    
    safe_name = img_name.replace('/', '_').replace('.png', '').replace('.jpg', '')
    out_path = os.path.join(output_dir, f'vis_{idx:03d}_{safe_name}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Init: {init_pos:.0f}cm → Coarse: {coarse_pos:.1f}cm → Final: {final_pos:.1f}cm / {final_rot:.2f}°")
    
    return {
        'img_name': img_name,
        'init_pos': init_pos, 'init_rot': init_rot,
        'coarse_pos': coarse_pos, 'coarse_rot': coarse_rot,
        'final_pos': final_pos, 'final_rot': final_rot,
        'cos_coarse_gt': float(cos_coarse_gt.mean()),
        'cos_fine_gt': float(cos_fine_gt.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cameras_json', type=str, required=True)
    parser.add_argument('--render_height', type=int, default=540)
    parser.add_argument('--render_width', type=int, default=960)
    parser.add_argument('--fine_render_height', type=int, default=270)
    parser.add_argument('--fine_render_width', type=int, default=480)
    parser.add_argument('--sample_indices', type=int, nargs='+', default=[0, 9, 47, 85, 100, 150])
    parser.add_argument('--output_dir', type=str, default='output/gsff_vis')
    parser.add_argument('--trans_lr_scale', type=float, default=10.0)
    args = parser.parse_args()
    
    device = torch.device('cuda')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_args = argparse.Namespace(**ckpt['args'])
    
    # Load Gaussian model
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
    
    # Load encoder
    encoder_state = ckpt['encoder']
    use_old = any('fine_encoder.encoder.' in k for k in encoder_state.keys())
    encoder = DualScaleEncoder(
        feature_dim=ckpt_args.feature_dim,
        freeze_backbone=True,
        use_old_fine_encoder=use_old,
    ).to(device)
    encoder.load_state_dict(encoder_state)
    encoder.eval()
    
    # Cameras
    with open(args.cameras_json) as f:
        all_cams = json.load(f)
    cam_by_name = {c['img_name']: c for c in all_cams}
    
    source_dir = Path(args.source_dir)
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
            })
    
    render_w, render_h = args.render_width, args.render_height
    first_cam = all_cams[0]
    orig_w, orig_h = first_cam['width'], first_cam['height']
    fx = first_cam['fx'] * render_w / orig_w
    fy = first_cam['fy'] * render_h / orig_h
    
    K_mat = torch.zeros(3, 3, device=device)
    K_mat[0, 0] = fx; K_mat[1, 1] = fy
    K_mat[0, 2] = render_w / 2.0; K_mat[1, 2] = render_h / 2.0
    K_mat[2, 2] = 1.0
    
    coarse_h, coarse_w = render_h // 14, render_w // 14
    fine_h = min(render_h, args.fine_render_height)
    fine_w = min(render_w, args.fine_render_width)
    
    K_coarse = K_mat.clone()
    K_coarse[0] *= coarse_w / render_w
    K_coarse[1] *= coarse_h / render_h
    
    K_fine = K_mat.clone()
    K_fine[0] *= fine_w / render_w
    K_fine[1] *= fine_h / render_h
    
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    # Extract triplane features once
    print("Extracting triplane features...")
    with torch.no_grad():
        coarse_colors = triplane.extract_coarse(means3d)
        coarse_colors = F.normalize(coarse_colors, p=2, dim=1)
        fine_colors = triplane.extract_fine(means3d)
        fine_colors = F.normalize(fine_colors, p=2, dim=1)
    
    # Process samples
    results = []
    for idx in args.sample_indices:
        if idx >= len(test_samples):
            print(f"  Skipping index {idx} (only {len(test_samples)} test samples)")
            continue
        r = visualize_sample(
            idx, test_samples[idx], gs_model, triplane, encoder,
            train_cams, train_positions, cam_by_name, source_dir, device,
            K_mat, K_coarse, K_fine, render_w, render_h,
            coarse_h, coarse_w, fine_h, fine_w,
            coarse_colors, fine_colors, args.output_dir, MEAN, STD, args,
        )
        results.append(r)
    
    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['img_name']}: {r['init_pos']:.0f}cm → {r['final_pos']:.1f}cm/{r['final_rot']:.2f}° "
              f"(cos_c={r['cos_coarse_gt']:.3f}, cos_f={r['cos_fine_gt']:.3f})")
    print(f"\nVisualization outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
