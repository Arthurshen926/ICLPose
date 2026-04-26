#!/usr/bin/env python3
"""
PnP验证脚本 - 检验隐式对应关系的质量

用途：
1. 加载训练好的模型
2. 对验证集推理，获取2D-3D keypoints
3. 用OpenCV PnP求解位姿，与网络位姿对比
4. 可视化heatmap和对应关系

结论判读：
- PnP位姿 << 网络位姿 → 对应关系好，回归器是瓶颈
- PnP位姿 >> 网络位姿 → 对应关系差，需改进融合模块
- PnP完全失败 → 对应关系无意义
"""

import os
import sys
import yaml
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import CorrespondenceDataset, collate_fn
from torch.utils.data import DataLoader
from utils.model_factory import create_model
from utils.training_utils import compute_pose_error
from splatloc_modules.gaussian_splatting.scene.gaussian_model import GaussianModel
from splatloc_modules.models.decoders import FeatureDecoder


def load_model_and_config(checkpoint_path, config_path=None):
    """加载模型和配置"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 尝试从checkpoint目录找config
    if config_path is None:
        ckpt_dir = Path(checkpoint_path).parent.parent
        config_path = ckpt_dir / 'config.yaml'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return checkpoint, config


def solve_pnp(img_keypoints_np, pcd_keypoints_np, K, method=cv2.SOLVEPNP_EPNP):
    """
    使用OpenCV PnP求解位姿
    
    Args:
        img_keypoints_np: (N, 2) 2D关键点
        pcd_keypoints_np: (N, 3) 3D关键点
        K: (3, 3) 相机内参
    
    Returns:
        success: 是否成功
        pose_c2w: (4, 4) 位姿矩阵 (camera-to-world)
    """
    N = img_keypoints_np.shape[0]
    if N < 4:
        return False, np.eye(4)
    
    # PnP求解
    success, rvec, tvec = cv2.solvePnP(
        pcd_keypoints_np.astype(np.float64),
        img_keypoints_np.astype(np.float64),
        K.astype(np.float64),
        None,
        flags=method
    )
    
    if not success:
        return False, np.eye(4)
    
    # 可选：LM精炼
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            pcd_keypoints_np.astype(np.float64),
            img_keypoints_np.astype(np.float64),
            K.astype(np.float64),
            None,
            rvec, tvec
        )
    except:
        pass
    
    # 转换为位姿矩阵
    R_w2c, _ = cv2.Rodrigues(rvec)
    t_w2c = tvec.flatten()
    
    # w2c → c2w
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c
    
    pose_c2w = np.eye(4)
    pose_c2w[:3, :3] = R_c2w
    pose_c2w[:3, 3] = t_c2w
    
    return True, pose_c2w


def solve_pnp_ransac(img_keypoints_np, pcd_keypoints_np, K):
    """使用RANSAC PnP求解位姿（更鲁棒）"""
    N = img_keypoints_np.shape[0]
    if N < 6:
        return False, np.eye(4), None
    
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pcd_keypoints_np.astype(np.float64),
        img_keypoints_np.astype(np.float64),
        K.astype(np.float64),
        None,
        iterationsCount=1000,
        reprojectionError=8.0,
        flags=cv2.SOLVEPNP_EPNP
    )
    
    if not success:
        return False, np.eye(4), None
    
    R_w2c, _ = cv2.Rodrigues(rvec)
    t_w2c = tvec.flatten()
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c
    
    pose_c2w = np.eye(4)
    pose_c2w[:3, :3] = R_c2w
    pose_c2w[:3, 3] = t_c2w
    
    return True, pose_c2w, inliers


def visualize_heatmap_and_keypoints(
    image, img_keypoints, pcd_keypoints, 
    gt_pose, K, heatmap, img_pixels,
    save_path=None
):
    """
    可视化heatmap和keypoint对应关系
    
    Args:
        image: (3, H, W) 图像tensor  
        img_keypoints: (N_query, 2) 2D关键点
        pcd_keypoints: (N_query, 3) 3D关键点
        gt_pose: (4, 4) GT位姿
        K: (3, 3) 内参
        heatmap: (N_query, N_img) attention heatmap
        img_pixels: (N_img, 2) 采样的2D像素坐标
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 原始图像
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    # 1. 图像 + 预测的2D关键点
    axes[0, 0].imshow(img_np)
    axes[0, 0].scatter(
        img_keypoints[:, 0].cpu().numpy(),
        img_keypoints[:, 1].cpu().numpy(),
        c='red', s=30, marker='x', linewidths=1,
    )
    axes[0, 0].set_title(f'Predicted 2D Keypoints (N={img_keypoints.shape[0]})')
    axes[0, 0].set_xlim(0, 640)
    axes[0, 0].set_ylim(480, 0)
    
    # 2. 图像 + GT投影的3D关键点
    R_c2w = gt_pose[:3, :3].cpu().numpy()
    t_c2w = gt_pose[:3, 3].cpu().numpy()
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    
    pts3d = pcd_keypoints.cpu().numpy()
    pts_cam = (R_w2c @ pts3d.T + t_w2c.reshape(3, 1)).T
    K_np = K.cpu().numpy()
    
    valid = pts_cam[:, 2] > 0.05
    pts_proj = np.zeros((pts3d.shape[0], 2))
    if valid.sum() > 0:
        pts_proj[valid, 0] = K_np[0, 0] * pts_cam[valid, 0] / pts_cam[valid, 2] + K_np[0, 2]
        pts_proj[valid, 1] = K_np[1, 1] * pts_cam[valid, 1] / pts_cam[valid, 2] + K_np[1, 2]
    
    axes[0, 1].imshow(img_np)
    axes[0, 1].scatter(
        pts_proj[valid, 0], pts_proj[valid, 1],
        c='blue', s=30, marker='o',
    )
    axes[0, 1].scatter(
        img_keypoints[:, 0].cpu().numpy(),
        img_keypoints[:, 1].cpu().numpy(),
        c='red', s=30, marker='x', linewidths=1,
    )
    axes[0, 1].set_title('Red=pred 2D, Blue=GT-projected 3D')
    axes[0, 1].set_xlim(0, 640)
    axes[0, 1].set_ylim(480, 0)
    
    # 3. Heatmap统计
    heatmap_np = heatmap.cpu().numpy()
    max_vals = heatmap_np.max(axis=1)
    entropy = -(heatmap_np * np.log(heatmap_np + 1e-10)).sum(axis=1)
    
    axes[1, 0].hist(max_vals, bins=30, alpha=0.7)
    axes[1, 0].axvline(x=1.0/heatmap_np.shape[1], color='r', linestyle='--', 
                        label=f'Uniform ({1.0/heatmap_np.shape[1]:.4f})')
    axes[1, 0].set_xlabel('Max attention probability')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Heatmap Max Prob Distribution\nmean={max_vals.mean():.4f}, max={max_vals.max():.4f}')
    axes[1, 0].legend()
    
    # 4. Keypoint分布
    kp2d = img_keypoints.cpu().numpy()
    axes[1, 1].scatter(kp2d[:, 0], kp2d[:, 1], c='red', s=20, alpha=0.7, label='2D keypoints')
    axes[1, 1].set_xlim(0, 640)
    axes[1, 1].set_ylim(480, 0)
    axes[1, 1].set_title(f'Keypoint Spatial Distribution\nSpread: x={kp2d[:, 0].std():.1f}px, y={kp2d[:, 1].std():.1f}px')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存可视化: {save_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                        default='output/exp025_synthetic_data/checkpoints/best.pth')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=20,
                        help='验证样本数')
    parser.add_argument('--vis', action='store_true', default=True,
                        help='可视化')
    parser.add_argument('--output_dir', type=str, default='output/pnp_validation')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    print("=" * 60)
    print("PnP 对应关系验证")
    print("=" * 60)
    
    checkpoint, config = load_model_and_config(args.checkpoint, args.config)
    
    # 创建模型
    model = create_model(config['model'], device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    # 创建FeatureDecoder
    splatloc_cfg = config.get('splatloc', {})
    scene_cfg = config.get('scene', config.get('splatloc', {}))
    decoder_cfg = config.get('decoder', {})
    
    combined_cfg = {
        'scene': {'bound': scene_cfg.get('bound', config.get('scene', {}).get('bound')),
                  'voxel_sdf': scene_cfg.get('voxel_sdf', config.get('scene', {}).get('voxel_sdf', 0.06))},
        'decoder': decoder_cfg
    }
    
    feat_decoder = FeatureDecoder(combined_cfg)
    decoder_ckpt = torch.load(splatloc_cfg['decoder_path'], map_location='cpu')
    feat_decoder.load_state_dict(decoder_ckpt)
    feat_decoder = feat_decoder.to(device)
    feat_decoder.bounding_box = feat_decoder.bounding_box.to(device)
    feat_decoder.eval()
    
    # 2. 加载验证数据集 
    data_cfg = config['dataset']
    val_dataset = CorrespondenceDataset(
        data_root=data_cfg['data_root'],
        scene_name=data_cfg['val_scene'],
        image_size=tuple(data_cfg['image_size']),
        augment=False,
        gaussian_path=data_cfg.get('gaussian_path'),
        fx=data_cfg['fx'], fy=data_cfg['fy'],
        cx=data_cfg['cx'], cy=data_cfg['cy'],
        sample_step=data_cfg.get('val_step', 1),
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    
    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 推理 + PnP
    net_rot_errors = []
    net_trans_errors = []
    pnp_rot_errors = []
    pnp_trans_errors = []
    pnp_ransac_rot_errors = []
    pnp_ransac_trans_errors = []
    heatmap_stats = []
    
    feature_dim = config['model']['feature_dim']
    img_width = data_cfg['image_size'][0]
    img_height = data_cfg['image_size'][1]
    
    print(f"\n验证集: {len(val_dataset)} 样本")
    print(f"测试样本: {min(args.num_samples, len(val_dataset))}")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.num_samples:
                break
            
            # 特征提取
            images = batch['image'].to(device)
            poses = batch['pose'].to(device)
            K = batch['intrinsics'].to(device)
            pts_2d = batch['points_2d'].to(device)
            pts_3d = batch['points_3d'].to(device)
            sample_indices = batch['sample_indices'].to(device)
            batch_size = batch['batch_size']
            
            fused_features = batch.get('fused_feature')
            if fused_features is not None:
                fused_features = fused_features.to(device)
            
            # 3D特征
            pcd_feats_flat = feat_decoder(pts_3d)
            
            pts_per_sample = [(sample_indices == b).sum().item() for b in range(batch_size)]
            max_pts = max(pts_per_sample)
            
            pcd_feats = torch.zeros(batch_size, max_pts, feature_dim, device=device)
            pcd_points = torch.zeros(batch_size, max_pts, 3, device=device)
            img_pixels = torch.zeros(batch_size, max_pts, 2, device=device)
            img_feats = torch.zeros(batch_size, max_pts, feature_dim, device=device)
            
            for b in range(batch_size):
                mask = sample_indices == b
                n_pts = pts_per_sample[b]
                pcd_feats[b, :n_pts] = pcd_feats_flat[mask]
                pcd_points[b, :n_pts] = pts_3d[mask]
                img_pixels[b, :n_pts] = pts_2d[mask]
            
            # 2D特征
            if fused_features is not None:
                for b in range(batch_size):
                    mask = sample_indices == b
                    if not mask.any():
                        continue
                    pts_2d_b = pts_2d[mask]
                    n_pts = pts_2d_b.shape[0]
                    H_feat, W_feat = fused_features.shape[2], fused_features.shape[3]
                    u_feat = pts_2d_b[:, 0] * (W_feat / img_width)
                    v_feat = pts_2d_b[:, 1] * (H_feat / img_height)
                    grid_x = 2.0 * u_feat / (W_feat - 1) - 1.0
                    grid_y = 2.0 * v_feat / (H_feat - 1) - 1.0
                    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
                    sampled_feats = torch.nn.functional.grid_sample(
                        fused_features[b:b+1], grid,
                        mode='bilinear', padding_mode='border', align_corners=True
                    ).squeeze(2).squeeze(0).permute(1, 0)
                    img_feats[b, :n_pts] = sampled_feats
            else:
                img_feats = pcd_feats.clone()
            
            # 位置编码
            img_pixels_norm = img_pixels.clone()
            img_pixels_norm[:, :, 0] = img_pixels_norm[:, :, 0] / img_width
            img_pixels_norm[:, :, 1] = img_pixels_norm[:, :, 1] / img_height
            img_pos_embeds = model.pos_enc_2d(img_pixels_norm)
            pcd_pos_embeds = model.pos_enc_3d(pcd_points)
            pcd_pos_embeds = model.pos_enc_3d_proj(pcd_pos_embeds)
            
            # 前向传播
            outputs = model(
                img_feats=img_feats,
                pcd_feats=pcd_feats,
                img_pixels=img_pixels,
                pcd_points=pcd_points,
                img_pos_embeds=img_pos_embeds,
                pcd_pos_embeds=pcd_pos_embeds,
            )
            
            pose_pred, _, _, _, img_heatmap, kp_2d, kp_3d = outputs
            
            # GT位姿
            gt_pose = poses[0]
            
            # 网络位姿误差
            rot_err, trans_err = compute_pose_error(
                pose_pred, gt_pose.unsqueeze(0)
            )
            net_rot_errors.append(rot_err[0].item())
            net_trans_errors.append(trans_err[0].item())
            
            # PnP求解
            kp_2d_np = kp_2d[0].cpu().numpy()
            kp_3d_np = kp_3d[0].cpu().numpy()
            K_np = K[0].cpu().numpy()
            
            # EPnP
            success, pose_pnp = solve_pnp(kp_2d_np, kp_3d_np, K_np)
            if success:
                pose_pnp_t = torch.from_numpy(pose_pnp).float().unsqueeze(0).to(device)
                pnp_rot, pnp_trans = compute_pose_error(pose_pnp_t, gt_pose.unsqueeze(0))
                pnp_rot_errors.append(pnp_rot[0].item())
                pnp_trans_errors.append(pnp_trans[0].item())
            else:
                pnp_rot_errors.append(float('nan'))
                pnp_trans_errors.append(float('nan'))
            
            # RANSAC PnP
            success_r, pose_pnp_r, inliers = solve_pnp_ransac(kp_2d_np, kp_3d_np, K_np)
            if success_r:
                pose_pnp_r_t = torch.from_numpy(pose_pnp_r).float().unsqueeze(0).to(device)
                pnp_r_rot, pnp_r_trans = compute_pose_error(pose_pnp_r_t, gt_pose.unsqueeze(0))
                pnp_ransac_rot_errors.append(pnp_r_rot[0].item())
                pnp_ransac_trans_errors.append(pnp_r_trans[0].item())
                n_inliers = len(inliers) if inliers is not None else 0
            else:
                pnp_ransac_rot_errors.append(float('nan'))
                pnp_ransac_trans_errors.append(float('nan'))
                n_inliers = 0
            
            # Heatmap统计
            hm = img_heatmap[0].cpu().numpy()
            hm_max = hm.max(axis=1).mean()
            hm_entropy = -(hm * np.log(hm + 1e-10)).sum(axis=1).mean()
            uniform_entropy = np.log(hm.shape[1])
            heatmap_stats.append({
                'max_prob': hm_max,
                'entropy': hm_entropy,
                'uniform_entropy': uniform_entropy,
                'entropy_ratio': hm_entropy / uniform_entropy,
            })
            
            # 打印
            pnp_rot_str = f"{pnp_rot_errors[-1]:.2f}°" if not np.isnan(pnp_rot_errors[-1]) else "FAIL"
            pnp_trans_str = f"{pnp_trans_errors[-1]:.3f}m" if not np.isnan(pnp_trans_errors[-1]) else "FAIL"
            pnp_r_str = f"{pnp_ransac_rot_errors[-1]:.2f}°/{pnp_ransac_trans_errors[-1]:.3f}m" if not np.isnan(pnp_ransac_rot_errors[-1]) else "FAIL"
            
            print(f"[{i+1:3d}] Net: {rot_err[0].item():.2f}° / {trans_err[0].item():.3f}m | "
                  f"PnP: {pnp_rot_str} / {pnp_trans_str} | "
                  f"RANSAC-PnP: {pnp_r_str} | "
                  f"Inliers: {n_inliers}/{kp_2d_np.shape[0]} | "
                  f"HM_max: {hm_max:.4f}")
            
            # 可视化
            if args.vis and i < 10:
                visualize_heatmap_and_keypoints(
                    images[0], kp_2d[0], kp_3d[0],
                    gt_pose, K[0], img_heatmap[0], img_pixels[0],
                    save_path=str(output_dir / f'sample_{i:03d}.png'),
                )
    
    # 4. 汇总统计
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    
    net_rot = np.array(net_rot_errors)
    net_trans = np.array(net_trans_errors)
    pnp_rot = np.array(pnp_rot_errors)
    pnp_trans = np.array(pnp_trans_errors)
    pnp_r_rot = np.array(pnp_ransac_rot_errors)
    pnp_r_trans = np.array(pnp_ransac_trans_errors)
    
    print(f"\n网络位姿:")
    print(f"  旋转误差: mean={net_rot.mean():.2f}°, median={np.median(net_rot):.2f}°")
    print(f"  平移误差: mean={net_trans.mean():.3f}m, median={np.median(net_trans):.3f}m")
    
    valid_pnp = ~np.isnan(pnp_rot)
    if valid_pnp.sum() > 0:
        print(f"\nEPnP位姿 ({valid_pnp.sum()}/{len(pnp_rot)} 成功):")
        print(f"  旋转误差: mean={pnp_rot[valid_pnp].mean():.2f}°, median={np.median(pnp_rot[valid_pnp]):.2f}°")
        print(f"  平移误差: mean={pnp_trans[valid_pnp].mean():.3f}m, median={np.median(pnp_trans[valid_pnp]):.3f}m")
    
    valid_r = ~np.isnan(pnp_r_rot)
    if valid_r.sum() > 0:
        print(f"\nRANSAC-PnP位姿 ({valid_r.sum()}/{len(pnp_r_rot)} 成功):")
        print(f"  旋转误差: mean={pnp_r_rot[valid_r].mean():.2f}°, median={np.median(pnp_r_rot[valid_r]):.2f}°")
        print(f"  平移误差: mean={pnp_r_trans[valid_r].mean():.3f}m, median={np.median(pnp_r_trans[valid_r]):.3f}m")
    
    # Heatmap统计
    print(f"\nHeatmap统计:")
    avg_max = np.mean([s['max_prob'] for s in heatmap_stats])
    avg_entropy_ratio = np.mean([s['entropy_ratio'] for s in heatmap_stats])
    print(f"  平均max prob: {avg_max:.6f} (理想: >0.1)")
    print(f"  平均entropy ratio: {avg_entropy_ratio:.4f} (理想: <0.5, 当前1.0=均匀分布)")
    
    # 结论
    print("\n" + "=" * 70)
    print("诊断结论")
    print("=" * 70)
    
    if avg_max < 0.01:
        print("⚠️  Heatmap过于平坦! max_prob < 0.01")
        print("   → 对应关系可能无意义，每个keypoint ≈ 所有点的均值")
        print("   → 建议: 大幅降低temperature (当前τ_img=4.0 → 0.1~0.5)")
    elif avg_max < 0.05:
        print("⚠️  Heatmap偏平! max_prob < 0.05")
        print("   → 对应关系有一定区分度但不够")
        print("   → 建议: 降低temperature")
    else:
        print("✅ Heatmap分布合理")
    
    if valid_pnp.sum() > 0:
        pnp_mean_rot = pnp_rot[valid_pnp].mean()
        net_mean_rot = net_rot.mean()
        
        if pnp_mean_rot < net_mean_rot * 0.5:
            print(f"\n✅ PnP ({pnp_mean_rot:.1f}°) << Network ({net_mean_rot:.1f}°)")
            print("   → 对应关系质量好! 回归器是瓶颈")
            print("   → 建议: 用PnP替代或辅助回归器")
        elif pnp_mean_rot > net_mean_rot * 2:
            print(f"\n❌ PnP ({pnp_mean_rot:.1f}°) >> Network ({net_mean_rot:.1f}°)")
            print("   → 对应关系质量差! 网络靠全局特征而非几何对应")
            print("   → 建议: 改进融合模块，或改为端到端回归")
        else:
            print(f"\n🟡 PnP ({pnp_mean_rot:.1f}°) ≈ Network ({net_mean_rot:.1f}°)")
            print("   → 对应关系有些作用但不是决定性因素")
    
    print()


if __name__ == '__main__':
    main()
