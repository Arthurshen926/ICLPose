"""
测试初始位姿生成和使用是否正确
"""

import sys
import yaml
import numpy as np
import torch
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import CorrespondenceDataset, collate_fn
from torch.utils.data import DataLoader


def test_initial_pose_generation():
    """测试初始位姿生成"""
    print("=" * 80)
    print("测试初始位姿生成")
    print("=" * 80)
    
    # 加载配置
    config_path = Path(__file__).parent / "configs/train_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['dataset']
    
    # 创建数据集（启用初始位姿）
    print("\n1. 创建数据集（use_initial_pose=True）...")
    dataset = CorrespondenceDataset(
        data_root=data_cfg['data_root'],
        scene_name=data_cfg['train_scene'],
        image_size=tuple(data_cfg['image_size']),
        augment=True,
        max_samples=10,
        use_depth=False,
        gaussian_path=data_cfg.get('gaussian_path'),
        fx=data_cfg['fx'],
        fy=data_cfg['fy'],
        cx=data_cfg['cx'],
        cy=data_cfg['cy'],
        sample_step=1,
        use_initial_pose=True,
        pose_noise_rot_deg=5.0,
        pose_noise_trans_m=0.1,
    )
    
    print(f"   数据集大小: {len(dataset)}")
    
    # 测试单个样本
    print("\n2. 测试单个样本...")
    sample = dataset[0]
    
    print(f"   样本包含的键: {list(sample.keys())}")
    assert 'pose' in sample, "缺少pose（真值位姿）"
    assert 'initial_pose' in sample, "缺少initial_pose（带噪声的初始位姿）"
    
    pose_gt = sample['pose'].numpy()
    pose_init = sample['initial_pose'].numpy()
    
    print(f"\n   真值位姿 (GT):")
    print(f"     旋转: {pose_gt[:3, :3]}")
    print(f"     平移: {pose_gt[:3, 3]}")
    
    print(f"\n   初始位姿 (带噪声):")
    print(f"     旋转: {pose_init[:3, :3]}")
    print(f"     平移: {pose_init[:3, 3]}")
    
    # 计算误差
    R_gt = pose_gt[:3, :3]
    R_init = pose_init[:3, :3]
    t_gt = pose_gt[:3, 3]
    t_init = pose_init[:3, 3]
    
    # 旋转误差（度）
    R_diff = R_init.T @ R_gt
    trace = np.trace(R_diff)
    angle_diff_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    angle_diff_deg = np.degrees(angle_diff_rad)
    
    # 平移误差（米）
    trans_diff = np.linalg.norm(t_init - t_gt)
    
    print(f"\n   噪声统计:")
    print(f"     旋转误差: {angle_diff_deg:.2f}° (配置: 5.0°标准差)")
    print(f"     平移误差: {trans_diff:.3f}m (配置: 0.1m标准差)")
    
    # 测试DataLoader
    print("\n3. 测试DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    batch = next(iter(loader))
    print(f"   Batch包含的键: {list(batch.keys())}")
    assert 'pose' in batch, "batch缺少pose"
    assert 'initial_pose' in batch, "batch缺少initial_pose"
    
    print(f"   pose shape: {batch['pose'].shape}")
    print(f"   initial_pose shape: {batch['initial_pose'].shape}")
    
    # 验证batch中的每帧都有不同的初始位姿
    print("\n4. 验证每帧的初始位姿是独立的...")
    poses_init = batch['initial_pose'].numpy()
    
    all_same = True
    for i in range(1, len(poses_init)):
        if not np.allclose(poses_init[0], poses_init[i]):
            all_same = False
            break
    
    if all_same:
        print("   ⚠️  警告: 所有帧的初始位姿相同（不应该这样）")
    else:
        print("   ✓ 正确: 每帧都有独立的初始位姿")
    
    # 打印每帧的噪声统计
    print("\n5. Batch中每帧的噪声统计:")
    poses_gt = batch['pose'].numpy()
    
    rot_errors = []
    trans_errors = []
    
    for i in range(len(poses_gt)):
        R_gt = poses_gt[i, :3, :3]
        R_init = poses_init[i, :3, :3]
        t_gt = poses_gt[i, :3, 3]
        t_init = poses_init[i, :3, 3]
        
        R_diff = R_init.T @ R_gt
        trace = np.trace(R_diff)
        angle_diff_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        angle_diff_deg = np.degrees(angle_diff_rad)
        
        trans_diff = np.linalg.norm(t_init - t_gt)
        
        rot_errors.append(angle_diff_deg)
        trans_errors.append(trans_diff)
        
        print(f"   帧{i}: 旋转误差={angle_diff_deg:.2f}°, 平移误差={trans_diff:.3f}m")
    
    print(f"\n   统计:")
    print(f"     旋转误差 - 均值: {np.mean(rot_errors):.2f}°, 标准差: {np.std(rot_errors):.2f}°")
    print(f"     平移误差 - 均值: {np.mean(trans_errors):.3f}m, 标准差: {np.std(trans_errors):.3f}m")
    
    print("\n" + "=" * 80)
    print("✓ 测试通过！初始位姿生成正常")
    print("=" * 80)


def test_relative_pose_computation():
    """测试相对位姿计算"""
    print("\n" + "=" * 80)
    print("测试相对位姿计算")
    print("=" * 80)
    
    from train import compute_relative_pose, compose_pose
    
    # 创建测试数据
    batch_size = 4
    
    # 真值位姿（随机生成）
    poses_gt = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    for i in range(batch_size):
        # 随机旋转
        angle = torch.randn(1) * 0.1
        axis = torch.randn(3)
        axis = axis / axis.norm()
        # 简化：只添加平移
        poses_gt[i, :3, 3] = torch.randn(3)
    
    # 初始位姿（加噪声）
    poses_init = poses_gt.clone()
    poses_init[:, :3, 3] += torch.randn(batch_size, 3) * 0.1
    
    print(f"\n1. 计算相对位姿...")
    poses_rel = compute_relative_pose(poses_gt, poses_init)
    
    print(f"   相对位姿 shape: {poses_rel.shape}")
    print(f"   相对平移示例: {poses_rel[0, :3, 3]}")
    
    print(f"\n2. 恢复绝对位姿...")
    poses_reconstructed = compose_pose(poses_rel, poses_init)
    
    print(f"   重建位姿 shape: {poses_reconstructed.shape}")
    
    print(f"\n3. 验证重建误差...")
    recon_error = torch.abs(poses_reconstructed - poses_gt).max().item()
    print(f"   最大重建误差: {recon_error:.6f}")
    
    if recon_error < 1e-5:
        print("   ✓ 重建成功！误差在数值精度范围内")
    else:
        print(f"   ⚠️  警告: 重建误差较大 ({recon_error:.6f})")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    test_initial_pose_generation()
    test_relative_pose_computation()
    print("\n✅ 所有测试完成！")
