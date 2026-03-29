#!/usr/bin/env python3
"""
Visualize GSFFs feature field renderings vs 2D encoder features.

Produces a grid comparing:
  Row 1: Query image | Coarse 2D (encoder) | Coarse 3D (rendered) | Cosine similarity
  Row 2: Query image | Fine 2D (encoder)   | Fine 3D (rendered)   | Cosine similarity

For 4 sample test images at GT pose and NN-init pose, saved as PNG.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from gsff.triplane import DualScaleTriplane
from gsff.encoder import DualScaleEncoder
from gsff.pose_refine import render_features_for_pose


def pca_colorize(feat_map, n_components=3):
    """
    PCA-based colorization of a feature map [D, H, W] -> [3, H, W] in [0,1].
    """
    D, H, W = feat_map.shape
    feat_flat = feat_map.reshape(D, -1).T  # [HW, D]
    feat_flat = feat_flat - feat_flat.mean(0)
    
    U, S, Vt = torch.linalg.svd(feat_flat, full_matrices=False)
    proj = U[:, :n_components] * S[:n_components]  # [HW, 3]
    
    # Normalize to [0, 1]
    for c in range(n_components):
        mn, mx = proj[:, c].min(), proj[:, c].max()
        if mx - mn > 1e-8:
            proj[:, c] = (proj[:, c] - mn) / (mx - mn)
        else:
            proj[:, c] = 0.5
    
    return proj.T.reshape(3, H, W)


def cosine_sim_map(feat_2d, feat_3d):
    """
    Pixel-wise cosine similarity [D, H, W] x [D, H, W] -> [H, W] in [-1, 1].
    """
    f2 = F.normalize(feat_2d, p=2, dim=0)
    f3 = F.normalize(feat_3d, p=2, dim=0)
    return (f2 * f3).sum(dim=0)


def make_vis_grid(query_img, coarse_2d, coarse_3d, fine_2d, fine_3d, title=""):
    """Build visualization grid as numpy array [H, W, 3] in uint8."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, hspace=0.25, wspace=0.1)

    # -- Row 1: Coarse --
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(query_img)
    ax.set_title("Query Image", fontsize=11)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    c2d_vis = pca_colorize(coarse_2d).permute(1, 2, 0).cpu().numpy()
    ax.imshow(c2d_vis)
    ax.set_title(f"Coarse 2D (encoder) {list(coarse_2d.shape[1:])}", fontsize=11)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    c3d_vis = pca_colorize(coarse_3d).permute(1, 2, 0).cpu().numpy()
    ax.imshow(c3d_vis)
    ax.set_title(f"Coarse 3D (rendered) {list(coarse_3d.shape[1:])}", fontsize=11)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 3])
    c_sim = cosine_sim_map(coarse_2d, coarse_3d).cpu().numpy()
    im = ax.imshow(c_sim, vmin=-0.2, vmax=1.0, cmap='RdYlGn')
    ax.set_title(f"Coarse cos_sim (mean={c_sim.mean():.3f})", fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # -- Row 2: Fine --
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(query_img)
    ax.set_title("Query Image", fontsize=11)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 1])
    f2d_vis = pca_colorize(fine_2d).permute(1, 2, 0).cpu().numpy()
    ax.imshow(f2d_vis)
    ax.set_title(f"Fine 2D (encoder) {list(fine_2d.shape[1:])}", fontsize=11)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 2])
    f3d_vis = pca_colorize(fine_3d).permute(1, 2, 0).cpu().numpy()
    ax.imshow(f3d_vis)
    ax.set_title(f"Fine 3D (rendered) {list(fine_3d.shape[1:])}", fontsize=11)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 3])
    f_sim = cosine_sim_map(fine_2d, fine_3d).cpu().numpy()
    im = ax.imshow(f_sim, vmin=-0.2, vmax=1.0, cmap='RdYlGn')
    ax.set_title(f"Fine cos_sim (mean={f_sim.mean():.3f})", fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='output/gsff/OldHospital/checkpoints/best.pth')
    parser.add_argument('--source_dir', type=str, default='dataset/OldHospital')
    parser.add_argument('--model_path', type=str, default='output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply')
    parser.add_argument('--cameras_json', type=str, default='output/2dgs_models/OldHospital/v7_depth/cameras.json')
    parser.add_argument('--output_dir', type=str, default='output/gsff/OldHospital/vis')
    parser.add_argument('--n_samples', type=int, default=4)
    parser.add_argument('--render_height', type=int, default=540)
    parser.add_argument('--render_width', type=int, default=960)
    args = parser.parse_args()

    device = torch.device('cuda')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    print("Loading checkpoint...")
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

    # Load triplane + encoder
    triplane = DualScaleTriplane(
        coarse_resolution=ckpt_args.coarse_resolution,
        fine_resolution=ckpt_args.fine_resolution,
        feature_dim=ckpt_args.feature_dim,
        scene_extent=scene_extent,
    ).to(device)
    triplane.load_state_dict(ckpt['triplane'])
    triplane.eval()

    encoder = DualScaleEncoder(feature_dim=ckpt_args.feature_dim, freeze_backbone=True).to(device)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    # Load cameras
    with open(args.cameras_json) as f:
        all_cams = json.load(f)
    cam_by_name = {c['img_name']: c for c in all_cams}

    # Load test split
    source_dir = Path(args.source_dir)
    test_samples = []
    with open(source_dir / 'dataset_test.txt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Visual') or line.startswith('ImageFile'):
                continue
            parts = line.split()
            test_samples.append({'img_name': parts[0]})

    # Also load train cameras for NN-init
    train_names = set()
    with open(source_dir / 'dataset_train.txt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Visual') or line.startswith('ImageFile'):
                continue
            train_names.add(line.split()[0])
    train_cams = [c for c in all_cams if c['img_name'] in train_names]
    train_positions = np.array([c['position'] for c in train_cams])

    # Intrinsics
    render_h, render_w = args.render_height, args.render_width
    first_cam = all_cams[0]
    orig_w, orig_h = first_cam['width'], first_cam['height']
    fx = first_cam['fx'] * render_w / orig_w
    fy = first_cam['fy'] * render_h / orig_h

    K_mat = torch.zeros(3, 3, device=device)
    K_mat[0, 0] = fx; K_mat[1, 1] = fy
    K_mat[0, 2] = render_w / 2.0; K_mat[1, 2] = render_h / 2.0; K_mat[2, 2] = 1.0

    coarse_h, coarse_w = render_h // 14, render_w // 14
    fine_h, fine_w = min(render_h, 270), min(render_w, 480)

    K_coarse = K_mat.clone()
    K_coarse[0] *= coarse_w / render_w
    K_coarse[1] *= coarse_h / render_h

    K_fine = K_mat.clone()
    K_fine[0] *= fine_w / render_w
    K_fine[1] *= fine_h / render_h

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Pre-extract triplane features
    print("Extracting triplane features...")
    with torch.no_grad():
        coarse_colors = triplane.extract_coarse(means3d)
        coarse_colors_norm = F.normalize(coarse_colors, p=2, dim=1)
        fine_colors = triplane.extract_fine(means3d)
        fine_colors_norm = F.normalize(fine_colors, p=2, dim=1)

    # Select samples (spread evenly)
    step = max(1, len(test_samples) // args.n_samples)
    selected = [test_samples[i * step] for i in range(args.n_samples)]

    print(f"\nRendering {len(selected)} samples...")
    all_grids = []

    for idx, sample in enumerate(selected):
        img_name = sample['img_name']
        cam = cam_by_name[img_name]

        # Load image
        img_path = source_dir / img_name
        if not img_path.exists():
            img_path = source_dir / 'processed' / img_name
        img_pil = Image.open(img_path).convert('RGB')
        img_pil_resized = img_pil.resize((render_w, render_h), Image.BILINEAR)
        img_np = np.array(img_pil_resized)

        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        img_norm = ((img_tensor.unsqueeze(0).to(device)) - MEAN) / STD

        # GT pose
        gt_c2w = np.eye(4, dtype=np.float32)
        gt_c2w[:3, :3] = np.array(cam['rotation'], dtype=np.float32)
        gt_c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
        gt_w2c = torch.from_numpy(np.linalg.inv(gt_c2w).astype(np.float32)).to(device)

        # NN init pose
        test_pos = np.array(cam['position'])
        dists = np.linalg.norm(train_positions - test_pos[None, :], axis=1)
        nn_cam = train_cams[np.argmin(dists)]
        nn_c2w = np.eye(4, dtype=np.float32)
        nn_c2w[:3, :3] = np.array(nn_cam['rotation'], dtype=np.float32)
        nn_c2w[:3, 3] = np.array(nn_cam['position'], dtype=np.float32)
        nn_w2c = torch.from_numpy(np.linalg.inv(nn_c2w).astype(np.float32)).to(device)

        init_pos_err = np.linalg.norm(test_pos - np.array(nn_cam['position'])) * 100

        with torch.no_grad():
            # 2D features
            coarse_feat_2d, fine_feat_2d = encoder(img_norm)

            # Resize 2D features
            coarse_feat_2d_r = F.interpolate(coarse_feat_2d, (coarse_h, coarse_w),
                                             mode='bilinear', align_corners=False)
            fine_feat_2d_r = F.interpolate(fine_feat_2d, (fine_h, fine_w),
                                           mode='bilinear', align_corners=False)

            # === GT pose rendering ===
            coarse_feat_3d_gt = render_features_for_pose(
                means3d, quats, scales, opacities, coarse_colors_norm,
                gt_w2c.unsqueeze(0), K_coarse.unsqueeze(0), coarse_w, coarse_h,
            )
            fine_feat_3d_gt = render_features_for_pose(
                means3d, quats, scales, opacities, fine_colors_norm,
                gt_w2c.unsqueeze(0), K_fine.unsqueeze(0), fine_w, fine_h,
            )

            # === NN-init pose rendering ===
            coarse_feat_3d_nn = render_features_for_pose(
                means3d, quats, scales, opacities, coarse_colors_norm,
                nn_w2c.unsqueeze(0), K_coarse.unsqueeze(0), coarse_w, coarse_h,
            )
            fine_feat_3d_nn = render_features_for_pose(
                means3d, quats, scales, opacities, fine_colors_norm,
                nn_w2c.unsqueeze(0), K_fine.unsqueeze(0), fine_w, fine_h,
            )

        # Make GT pose grid
        grid_gt = make_vis_grid(
            img_np,
            coarse_feat_2d_r[0], coarse_feat_3d_gt[0],
            fine_feat_2d_r[0], fine_feat_3d_gt[0],
            title=f"[{idx}] {img_name} @ GT pose"
        )

        # Make NN-init pose grid
        grid_nn = make_vis_grid(
            img_np,
            coarse_feat_2d_r[0], coarse_feat_3d_nn[0],
            fine_feat_2d_r[0], fine_feat_3d_nn[0],
            title=f"[{idx}] {img_name} @ NN-init pose (init err={init_pos_err:.0f}cm)"
        )

        all_grids.append(grid_gt)
        all_grids.append(grid_nn)

        # Save individual
        Image.fromarray(grid_gt).save(os.path.join(args.output_dir, f'sample_{idx:02d}_gt.png'))
        Image.fromarray(grid_nn).save(os.path.join(args.output_dir, f'sample_{idx:02d}_nn.png'))
        print(f"  [{idx}] {img_name}: init_err={init_pos_err:.0f}cm")

    # Stack all into one big image
    combined = np.vstack(all_grids)
    Image.fromarray(combined).save(os.path.join(args.output_dir, 'feature_comparison_all.png'))
    print(f"\nAll visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
