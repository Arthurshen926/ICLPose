#!/usr/bin/env python3
"""
GSFFs 中间量可视化脚本。

对选定的测试图像运行 SOTA 配置的位姿优化，保存以下中间量可视化：

1. 输入图像 + 初始/优化后/GT 位姿渲染对比
2. 2D encoder coarse/fine 特征图 (PCA 降维到 RGB)
3. 3D triplane 渲染特征图（初始位姿 vs GT 位姿）
4. 特征余弦相似度热力图
5. 优化轨迹（loss 曲线 + 位姿误差曲线）
6. Coarse→Fine 多轮迭代的位姿改进过程

Usage:
    CUDA_VISIBLE_DEVICES=2 python legacy/gsff_baseline/scripts/visualize_gsff.py \
        --checkpoint output/gsff/OldHospital/checkpoints/final.pth \
        --source_dir dataset/OldHospital \
        --model_path output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply \
        --cameras_json output/2dgs_models/OldHospital/v7_depth/cameras.json \
        --output_dir output/gsff/OldHospital/visualizations \
        --num_samples 6
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
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
from gsff.pose_refine import (
    se3_exp, render_features_for_pose, render_features_transformed,
    _transform_gaussians, _render_with_transformed_gaussians,
)
from gsplat import rasterization_2dgs


# ── Utilities ────────────────────────────────────────────────────────────────

def pose_error(pred_w2c, gt_w2c):
    pred_c2w = torch.inverse(pred_w2c)
    gt_c2w = torch.inverse(gt_w2c)
    pos_err = torch.norm(pred_c2w[:3, 3] - gt_c2w[:3, 3]).item() * 100.0
    R_rel = pred_c2w[:3, :3] @ gt_c2w[:3, :3].T
    trace = R_rel[0, 0] + R_rel[1, 1] + R_rel[2, 2]
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rot_err = torch.acos(cos_angle).item() * 180.0 / math.pi
    return pos_err, rot_err


def features_to_rgb(feat_map, pca_model=None, fit=False):
    """Convert [1, D, H, W] feature map to [H, W, 3] RGB via PCA."""
    B, D, H, W = feat_map.shape
    pixels = feat_map[0].reshape(D, -1).T.cpu().numpy()  # [HW, D]
    if pca_model is None:
        pca_model = PCA(n_components=3)
        fit = True
    if fit:
        rgb = pca_model.fit_transform(pixels)
    else:
        rgb = pca_model.transform(pixels)
    # Normalize to [0, 1]
    rgb = rgb - rgb.min(axis=0)
    maxv = rgb.max(axis=0)
    maxv[maxv == 0] = 1
    rgb = rgb / maxv
    return rgb.reshape(H, W, 3), pca_model


def render_rgb(means3d, quats, scales, opacities, sh_dc, viewmat, K, width, height):
    """Render RGB image from 2DGS model (SH degree 0)."""
    rgb_colors = (sh_dc.squeeze(1) + 0.5).clamp(0, 1)
    render_colors, _, *_ = rasterization_2dgs(
        means=means3d, quats=quats, scales=scales, opacities=opacities,
        colors=rgb_colors, viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0), width=width, height=height,
        packed=False, near_plane=0.01, far_plane=1e5, render_mode='RGB',
    )
    return render_colors[0].clamp(0, 1).cpu().numpy()  # [H, W, 3]


def refine_pose_with_trajectory(
    feat_2d, means3d, quats, scales, opacities, colors,
    init_viewmat, K_mat, width, height, gt_w2c,
    n_iters=300, lr=0.01, trans_lr_scale=10.0, loss_type='mse',
):
    """Refine pose and record per-step loss + pose error trajectory."""
    device = init_viewmat.device
    
    delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
    delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
    
    param_groups = [
        {'params': [delta_t], 'lr': lr * trans_lr_scale},
        {'params': [delta_r], 'lr': lr},
    ]
    optimizer = torch.optim.Adam(param_groups)
    K_unsq = K_mat.unsqueeze(0).detach()
    
    trajectory = {'loss': [], 'pos_err': [], 'rot_err': [], 'delta_xi_norm': []}
    best_loss = float('inf')
    best_viewmat = init_viewmat.clone()
    
    for i in range(n_iters):
        optimizer.zero_grad()
        delta_xi = torch.cat([delta_t, delta_r])
        delta_T = se3_exp(delta_xi)
        
        feat_3d = render_features_transformed(
            means3d, quats, scales, opacities, colors,
            delta_T, init_viewmat, K_unsq, width, height, 16)
        feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
        
        loss = F.mse_loss(feat_3d_norm, feat_2d) if loss_type == 'mse' else F.smooth_l1_loss(feat_3d_norm, feat_2d)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_t, delta_r], 0.5)
        optimizer.step()
        
        with torch.no_grad():
            current_vm = se3_exp(delta_xi.detach()) @ init_viewmat
            pos_e, rot_e = pose_error(current_vm, gt_w2c)
            trajectory['loss'].append(loss.item())
            trajectory['pos_err'].append(pos_e)
            trajectory['rot_err'].append(rot_e)
            trajectory['delta_xi_norm'].append(delta_xi.detach().norm().item())
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_viewmat = current_vm.clone()
    
    return best_viewmat, trajectory


def cosine_sim_map(feat_2d, feat_3d):
    """Compute per-pixel cosine similarity [1, 1, H, W]."""
    f2 = F.normalize(feat_2d, p=2, dim=1)
    f3 = F.normalize(feat_3d, p=2, dim=1)
    return (f2 * f3).sum(dim=1, keepdim=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device('cuda')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ──
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_args = argparse.Namespace(**ckpt['args'])

    gs_model = GaussianFeatureModel(feature_dim=ckpt_args.feature_dim)
    gs_model.load_ply(args.model_path)
    gs_model = gs_model.to(device)
    gs_model.eval()

    means3d = gs_model.get_xyz.detach()
    quats = gs_model.get_rotation.detach()
    scales_raw = gs_model.get_scaling
    scales = torch.cat([scales_raw, torch.ones_like(scales_raw[:, :1])], dim=-1).detach()
    opacities = gs_model.get_opacity.squeeze(-1).detach()
    sh_dc = gs_model._features_dc.reshape(-1, 1, 3).detach()

    scene_extent = ckpt['scene_extent']

    triplane = DualScaleTriplane(
        coarse_resolution=ckpt_args.coarse_resolution,
        fine_resolution=ckpt_args.fine_resolution,
        feature_dim=ckpt_args.feature_dim,
        scene_extent=scene_extent,
    ).to(device)
    triplane.load_state_dict(ckpt['triplane'])
    triplane.eval()

    encoder_state = ckpt['encoder']
    use_old = any('fine_encoder.encoder.' in k for k in encoder_state.keys())
    encoder = DualScaleEncoder(
        feature_dim=ckpt_args.feature_dim, freeze_backbone=True,
        use_old_fine_encoder=use_old,
    ).to(device)
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    # ── Camera setup ──
    with open(args.cameras_json) as f:
        all_cams = json.load(f)

    source_dir = Path(args.source_dir)
    render_h, render_w = args.render_height, args.render_width
    first_cam = all_cams[0]
    orig_w, orig_h = first_cam['width'], first_cam['height']

    fx = first_cam['fx'] * render_w / orig_w
    fy = first_cam['fy'] * render_h / orig_h
    K_mat = torch.zeros(3, 3, device=device)
    K_mat[0, 0] = fx; K_mat[1, 1] = fy
    K_mat[0, 2] = render_w / 2.0; K_mat[1, 2] = render_h / 2.0; K_mat[2, 2] = 1.0

    coarse_h, coarse_w = render_h // 14, render_w // 14
    K_coarse = K_mat.clone()
    K_coarse[0] *= coarse_w / render_w; K_coarse[1] *= coarse_h / render_h

    fine_h, fine_w = min(render_h, 270), min(render_w, 480)
    K_fine = K_mat.clone()
    K_fine[0] *= fine_w / render_w; K_fine[1] *= fine_h / render_h

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # ── Pre-extract triplane features ──
    with torch.no_grad():
        coarse_colors = F.normalize(triplane.extract_coarse(means3d), p=2, dim=1)
        fine_colors = F.normalize(triplane.extract_fine(means3d), p=2, dim=1)

    # ── Load test samples ──
    cam_by_name = {c['img_name']: c for c in all_cams}
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

    # ── Select samples: mix of good and challenging ──
    # Run quick eval to find error distribution
    print(f"Quick pre-scan to select {args.num_samples} representative samples...")
    all_errors = []
    for sample in tqdm(test_samples[:], desc="Pre-scan"):
        cam = cam_by_name[sample['img_name']]
        gt_c2w = np.eye(4, dtype=np.float32)
        gt_c2w[:3, :3] = np.array(cam['rotation'], dtype=np.float32)
        gt_c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
        gt_w2c = torch.from_numpy(np.linalg.inv(gt_c2w).astype(np.float32)).to(device)
        
        # Init pose
        test_pos = np.array(cam['position'])
        dists = np.linalg.norm(train_positions - test_pos[None, :], axis=1)
        nn_idx = np.argmin(dists)
        nn_cam = train_cams[nn_idx]
        init_c2w = np.eye(4, dtype=np.float32)
        init_c2w[:3, :3] = np.array(nn_cam['rotation'], dtype=np.float32)
        init_c2w[:3, 3] = np.array(nn_cam['position'], dtype=np.float32)
        init_w2c = torch.from_numpy(np.linalg.inv(init_c2w).astype(np.float32)).to(device)
        
        init_pos_err, _ = pose_error(init_w2c, gt_w2c)
        all_errors.append((sample['img_name'], init_pos_err))
    
    # Sort by init error and pick diverse samples
    all_errors.sort(key=lambda x: x[1])
    n = len(all_errors)
    indices = [0, n//5, n//3, n//2, int(n*0.75), n-1]  # easy → hard
    indices = indices[:args.num_samples]
    selected = [all_errors[i][0] for i in indices]
    print(f"Selected samples (init error): {[(n, f'{e:.0f}cm') for n, e in all_errors if n in selected]}")

    # ── Visualize each sample ──
    pca_coarse = None  # Shared PCA across samples for consistency
    pca_fine = None
    summary_rows = []

    for idx, img_name in enumerate(selected):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(selected)}] Visualizing {img_name}")
        print(f"{'='*60}")
        
        sample_dir = os.path.join(args.output_dir, img_name.replace('/', '_').replace('.png', ''))
        os.makedirs(sample_dir, exist_ok=True)

        # Load image
        img_path = source_dir / img_name
        if not img_path.exists():
            img_path = source_dir / 'processed' / img_name
        img = Image.open(img_path).convert('RGB')
        img_for_display = img.resize((render_w, render_h), Image.BILINEAR)
        img_np = np.array(img_for_display).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(np.array(img_for_display)).float().permute(2, 0, 1) / 255.0
        img_norm = (img_tensor.unsqueeze(0).to(device) - MEAN) / STD

        # GT pose
        cam = cam_by_name[img_name]
        gt_c2w = np.eye(4, dtype=np.float32)
        gt_c2w[:3, :3] = np.array(cam['rotation'], dtype=np.float32)
        gt_c2w[:3, 3] = np.array(cam['position'], dtype=np.float32)
        gt_w2c = torch.from_numpy(np.linalg.inv(gt_c2w).astype(np.float32)).to(device)

        # Init pose (NN)
        test_pos = np.array(cam['position'])
        dists = np.linalg.norm(train_positions - test_pos[None, :], axis=1)
        nn_idx = np.argmin(dists)
        nn_cam = train_cams[nn_idx]
        init_c2w = np.eye(4, dtype=np.float32)
        init_c2w[:3, :3] = np.array(nn_cam['rotation'], dtype=np.float32)
        init_c2w[:3, 3] = np.array(nn_cam['position'], dtype=np.float32)
        init_w2c = torch.from_numpy(np.linalg.inv(init_c2w).astype(np.float32)).to(device)

        init_pos_err, init_rot_err = pose_error(init_w2c, gt_w2c)

        # ── 1. Extract 2D features ──
        with torch.no_grad():
            coarse_feat_2d, fine_feat_2d = encoder(img_norm)
            coarse_feat_2d = F.normalize(coarse_feat_2d, p=2, dim=1)
            fine_feat_2d_full = F.normalize(fine_feat_2d, p=2, dim=1)
        coarse_feat_2d_resized = F.interpolate(coarse_feat_2d, (coarse_h, coarse_w), mode='bilinear', align_corners=False)
        fine_feat_2d_resized = F.interpolate(fine_feat_2d_full, (fine_h, fine_w), mode='bilinear', align_corners=False)

        # ── 2. Render features at GT / init poses ──
        with torch.no_grad():
            feat_3d_gt_coarse = render_features_for_pose(
                means3d, quats, scales, opacities, coarse_colors,
                gt_w2c.unsqueeze(0), K_coarse.unsqueeze(0), coarse_w, coarse_h)
            feat_3d_gt_fine = render_features_for_pose(
                means3d, quats, scales, opacities, fine_colors,
                gt_w2c.unsqueeze(0), K_fine.unsqueeze(0), fine_w, fine_h)
            feat_3d_init_coarse = render_features_for_pose(
                means3d, quats, scales, opacities, coarse_colors,
                init_w2c.unsqueeze(0), K_coarse.unsqueeze(0), coarse_w, coarse_h)
            feat_3d_init_fine = render_features_for_pose(
                means3d, quats, scales, opacities, fine_colors,
                init_w2c.unsqueeze(0), K_fine.unsqueeze(0), fine_w, fine_h)

        # ── 3. Run multi-round refinement with trajectory ──
        current_pose = init_w2c.clone()
        all_trajectories = []
        round_poses = [init_w2c.clone()]
        
        for rnd in range(args.rounds):
            # Coarse
            refined_coarse, traj_coarse = refine_pose_with_trajectory(
                coarse_feat_2d_resized, means3d, quats, scales, opacities,
                coarse_colors, current_pose, K_coarse, coarse_w, coarse_h, gt_w2c,
                n_iters=args.coarse_iters, lr=0.01, trans_lr_scale=args.trans_lr_scale,
            )
            # Fine
            refined_fine, traj_fine = refine_pose_with_trajectory(
                fine_feat_2d_resized, means3d, quats, scales, opacities,
                fine_colors, refined_coarse, K_fine, fine_w, fine_h, gt_w2c,
                n_iters=args.fine_iters, lr=0.005, trans_lr_scale=args.trans_lr_scale,
            )
            current_pose = refined_fine
            all_trajectories.append(('coarse', rnd, traj_coarse))
            all_trajectories.append(('fine', rnd, traj_fine))
            round_poses.append(current_pose.clone())

        final_pos_err, final_rot_err = pose_error(current_pose, gt_w2c)
        print(f"  Init: {init_pos_err:.1f}cm / {init_rot_err:.2f}°")
        print(f"  Final: {final_pos_err:.1f}cm / {final_rot_err:.2f}°")
        summary_rows.append({
            'img': img_name, 'init_pos': init_pos_err, 'init_rot': init_rot_err,
            'final_pos': final_pos_err, 'final_rot': final_rot_err,
        })

        # ── Render at final pose ──
        with torch.no_grad():
            feat_3d_final_coarse = render_features_for_pose(
                means3d, quats, scales, opacities, coarse_colors,
                current_pose.unsqueeze(0), K_coarse.unsqueeze(0), coarse_w, coarse_h)
            feat_3d_final_fine = render_features_for_pose(
                means3d, quats, scales, opacities, fine_colors,
                current_pose.unsqueeze(0), K_fine.unsqueeze(0), fine_w, fine_h)

        # ═══════════════════════════════════════════════════════════════
        # VISUALIZATION 1: Feature maps (PCA → RGB)
        # ═══════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'{img_name} — Feature Maps (PCA→RGB)', fontsize=14)

        # Coarse row
        coarse_2d_rgb, pca_coarse = features_to_rgb(coarse_feat_2d_resized, pca_coarse, fit=(pca_coarse is None))
        coarse_gt_rgb, _ = features_to_rgb(F.normalize(feat_3d_gt_coarse, p=2, dim=1), pca_coarse)
        coarse_init_rgb, _ = features_to_rgb(F.normalize(feat_3d_init_coarse, p=2, dim=1), pca_coarse)
        coarse_final_rgb, _ = features_to_rgb(F.normalize(feat_3d_final_coarse, p=2, dim=1), pca_coarse)

        axes[0, 0].imshow(coarse_2d_rgb); axes[0, 0].set_title('Coarse 2D (Encoder)')
        axes[0, 1].imshow(coarse_gt_rgb); axes[0, 1].set_title('Coarse 3D @ GT')
        axes[0, 2].imshow(coarse_init_rgb); axes[0, 2].set_title('Coarse 3D @ Init')
        axes[0, 3].imshow(coarse_final_rgb); axes[0, 3].set_title('Coarse 3D @ Final')

        # Fine row
        fine_2d_rgb, pca_fine = features_to_rgb(fine_feat_2d_resized, pca_fine, fit=(pca_fine is None))
        fine_gt_rgb, _ = features_to_rgb(F.normalize(feat_3d_gt_fine, p=2, dim=1), pca_fine)
        fine_init_rgb, _ = features_to_rgb(F.normalize(feat_3d_init_fine, p=2, dim=1), pca_fine)
        fine_final_rgb, _ = features_to_rgb(F.normalize(feat_3d_final_fine, p=2, dim=1), pca_fine)

        axes[1, 0].imshow(fine_2d_rgb); axes[1, 0].set_title('Fine 2D (Encoder)')
        axes[1, 1].imshow(fine_gt_rgb); axes[1, 1].set_title('Fine 3D @ GT')
        axes[1, 2].imshow(fine_init_rgb); axes[1, 2].set_title('Fine 3D @ Init')
        axes[1, 3].imshow(fine_final_rgb); axes[1, 3].set_title('Fine 3D @ Final')

        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '1_feature_maps.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # ═══════════════════════════════════════════════════════════════
        # VISUALIZATION 2: Cosine similarity heatmaps
        # ═══════════════════════════════════════════════════════════════
        with torch.no_grad():
            cos_coarse_gt = cosine_sim_map(coarse_feat_2d_resized, F.normalize(feat_3d_gt_coarse, p=2, dim=1))
            cos_coarse_init = cosine_sim_map(coarse_feat_2d_resized, F.normalize(feat_3d_init_coarse, p=2, dim=1))
            cos_coarse_final = cosine_sim_map(coarse_feat_2d_resized, F.normalize(feat_3d_final_coarse, p=2, dim=1))
            cos_fine_gt = cosine_sim_map(fine_feat_2d_resized, F.normalize(feat_3d_gt_fine, p=2, dim=1))
            cos_fine_init = cosine_sim_map(fine_feat_2d_resized, F.normalize(feat_3d_init_fine, p=2, dim=1))
            cos_fine_final = cosine_sim_map(fine_feat_2d_resized, F.normalize(feat_3d_final_fine, p=2, dim=1))

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{img_name} — Cosine Similarity (2D vs 3D)', fontsize=14)

        vmin, vmax = -0.2, 1.0
        for row, (s_gt, s_init, s_final, label) in enumerate([
            (cos_coarse_gt, cos_coarse_init, cos_coarse_final, 'Coarse'),
            (cos_fine_gt, cos_fine_init, cos_fine_final, 'Fine'),
        ]):
            im0 = axes[row, 0].imshow(s_gt[0, 0].cpu().numpy(), cmap='RdYlGn', vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f'{label} @ GT (mean={s_gt.mean():.3f})')
            axes[row, 1].imshow(s_init[0, 0].cpu().numpy(), cmap='RdYlGn', vmin=vmin, vmax=vmax)
            axes[row, 1].set_title(f'{label} @ Init (mean={s_init.mean():.3f})')
            axes[row, 2].imshow(s_final[0, 0].cpu().numpy(), cmap='RdYlGn', vmin=vmin, vmax=vmax)
            axes[row, 2].set_title(f'{label} @ Final (mean={s_final.mean():.3f})')
        
        for ax in axes.flat:
            ax.axis('off')
        fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.6, label='Cosine Similarity')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '2_cosine_similarity.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # ═══════════════════════════════════════════════════════════════
        # VISUALIZATION 3: Optimization trajectory
        # ═══════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'{img_name} — Optimization Trajectory\n'
                     f'Init: {init_pos_err:.1f}cm/{init_rot_err:.2f}° → '
                     f'Final: {final_pos_err:.1f}cm/{final_rot_err:.2f}°', fontsize=13)

        colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        step_offset = 0
        for i, (stage, rnd, traj) in enumerate(all_trajectories):
            color = colors_list[i % len(colors_list)]
            label = f'R{rnd+1} {stage}'
            steps = list(range(step_offset, step_offset + len(traj['loss'])))
            axes[0, 0].plot(steps, traj['loss'], color=color, label=label, linewidth=1.5)
            axes[0, 1].plot(steps, traj['pos_err'], color=color, label=label, linewidth=1.5)
            axes[1, 0].plot(steps, traj['rot_err'], color=color, label=label, linewidth=1.5)
            axes[1, 1].plot(steps, traj['delta_xi_norm'], color=color, label=label, linewidth=1.5)
            step_offset += len(traj['loss'])

        axes[0, 0].set_ylabel('MSE Loss'); axes[0, 0].set_xlabel('Step')
        axes[0, 0].legend(fontsize=8); axes[0, 0].set_title('Loss')
        axes[0, 1].set_ylabel('Position Error (cm)'); axes[0, 1].set_xlabel('Step')
        axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].legend(fontsize=8); axes[0, 1].set_title('Position Error')
        axes[1, 0].set_ylabel('Rotation Error (°)'); axes[1, 0].set_xlabel('Step')
        axes[1, 0].legend(fontsize=8); axes[1, 0].set_title('Rotation Error')
        axes[1, 1].set_ylabel('||Δξ||'); axes[1, 1].set_xlabel('Step')
        axes[1, 1].legend(fontsize=8); axes[1, 1].set_title('Update Magnitude')

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '3_optimization_trajectory.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # ═══════════════════════════════════════════════════════════════
        # VISUALIZATION 4: RGB renderings (Init / Final / GT)
        # ═══════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'{img_name} — Pose Comparison', fontsize=14)

        axes[0].imshow(img_np); axes[0].set_title('Query Image')
        
        with torch.no_grad():
            rgb_init = render_rgb(means3d, quats, scales, opacities, sh_dc,
                                  init_w2c, K_mat, render_w, render_h)
            rgb_final = render_rgb(means3d, quats, scales, opacities, sh_dc,
                                   current_pose, K_mat, render_w, render_h)
            rgb_gt = render_rgb(means3d, quats, scales, opacities, sh_dc,
                                gt_w2c, K_mat, render_w, render_h)

        axes[1].imshow(rgb_init); axes[1].set_title(f'Init ({init_pos_err:.0f}cm/{init_rot_err:.1f}°)')
        axes[2].imshow(rgb_final); axes[2].set_title(f'Final ({final_pos_err:.1f}cm/{final_rot_err:.2f}°)')
        axes[3].imshow(rgb_gt); axes[3].set_title('GT Render')

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '4_rgb_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # ═══════════════════════════════════════════════════════════════
        # VISUALIZATION 5: Per-round pose improvement
        # ═══════════════════════════════════════════════════════════════
        round_errors = []
        for rp in round_poses:
            pe, re = pose_error(rp, gt_w2c)
            round_errors.append((pe, re))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{img_name} — Per-Round Improvement', fontsize=14)
        rounds_x = list(range(len(round_errors)))
        ax1.bar(rounds_x, [e[0] for e in round_errors], color=['#e74c3c'] + ['#3498db'] * args.rounds)
        ax1.set_xticks(rounds_x)
        ax1.set_xticklabels(['Init'] + [f'R{i+1}' for i in range(args.rounds)])
        ax1.set_ylabel('Position Error (cm)'); ax1.set_title('Position')
        for i, (pe, _) in enumerate(round_errors):
            ax1.text(i, pe + 1, f'{pe:.1f}', ha='center', fontsize=10)

        ax2.bar(rounds_x, [e[1] for e in round_errors], color=['#e74c3c'] + ['#2ecc71'] * args.rounds)
        ax2.set_xticks(rounds_x)
        ax2.set_xticklabels(['Init'] + [f'R{i+1}' for i in range(args.rounds)])
        ax2.set_ylabel('Rotation Error (°)'); ax2.set_title('Rotation')
        for i, (_, re) in enumerate(round_errors):
            ax2.text(i, re + 0.05, f'{re:.2f}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, '5_per_round.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Saved visualizations to {sample_dir}/")

    # ── Summary figure ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('GSFFs Pose Refinement — Sample Overview', fontsize=14)

    names = [r['img'].split('/')[-1].replace('.png', '') for r in summary_rows]
    init_pos = [r['init_pos'] for r in summary_rows]
    final_pos = [r['final_pos'] for r in summary_rows]
    init_rot = [r['init_rot'] for r in summary_rows]
    final_rot = [r['final_rot'] for r in summary_rows]

    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, init_pos, w, label='Init', color='#e74c3c', alpha=0.8)
    axes[0].bar(x + w/2, final_pos, w, label='Final', color='#3498db', alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('Position Error (cm)'); axes[0].legend()
    axes[0].set_title('Position Error: Init vs Final')

    axes[1].bar(x - w/2, init_rot, w, label='Init', color='#e74c3c', alpha=0.8)
    axes[1].bar(x + w/2, final_rot, w, label='Final', color='#2ecc71', alpha=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Rotation Error (°)'); axes[1].legend()
    axes[1].set_title('Rotation Error: Init vs Final')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSummary figure saved: {args.output_dir}/summary.png")


def parse_args():
    p = argparse.ArgumentParser(description='GSFFs Visualization')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--source_dir', type=str, required=True)
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--cameras_json', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='output/gsff/OldHospital/visualizations')
    p.add_argument('--render_height', type=int, default=540)
    p.add_argument('--render_width', type=int, default=960)
    p.add_argument('--num_samples', type=int, default=6)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--coarse_iters', type=int, default=300)
    p.add_argument('--fine_iters', type=int, default=300)
    p.add_argument('--trans_lr_scale', type=float, default=10.0)
    return p.parse_args()


if __name__ == '__main__':
    main()
