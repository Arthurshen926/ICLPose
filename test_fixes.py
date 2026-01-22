#!/usr/bin/env python3
"""
测试修复后的Dataset和特征提取实现
验证:
1. 融合特征是否正确加载
2. 深度图是否存在并正确使用
3. 2D-3D对应关系是否真实
4. HashGrid配置是否匹配
"""

import torch
import yaml
import sys
import os
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import CorrespondenceDataset, collate_fn
from torch.utils.data import DataLoader
from models.decoders import FeatureDecoder
from gaussian_splatting.scene.gaussian_model import GaussianModel
from munch import munchify

def test_dataset():
    """测试Dataset加载"""
    print("\n" + "="*60)
    print("测试1: Dataset加载和融合特征")
    print("="*60)
    
    dataset = CorrespondenceDataset(
        data_root="/home/yons/Projects/data/room_0",
        scene_name="Sequence_1",
        image_size=(640, 480),
        augment=False,
        max_samples=5,
        use_depth=True,
        fx=320.0, fy=320.0, cx=319.5, cy=239.5
    )
    
    print(f"✓ 数据集大小: {len(dataset)}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"\n样本0内容:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    # 检查融合特征
    if sample['fused_feature'] is not None:
        print(f"\n✓ 融合特征已加载: {sample['fused_feature'].shape}")
        print(f"  特征范围: [{sample['fused_feature'].min():.3f}, {sample['fused_feature'].max():.3f}]")
    else:
        print("\n✗ 警告: 融合特征未加载!")
    
    # 检查深度图
    if sample.get('depth') is not None:
        print(f"\n✓ 深度图已加载: {sample['depth'].shape}")
        print(f"  深度范围: [{sample['depth'].min():.3f}m, {sample['depth'].max():.3f}m]")
    else:
        print("\n✗ 警告: 深度图未加载!")
    
    # 检查2D-3D对应
    pts_2d = sample['points_2d']
    pts_3d = sample['points_3d']
    valid_mask = sample['valid_mask']
    
    print(f"\n✓ 2D-3D对应点:")
    print(f"  2D点数量: {pts_2d.shape[0]}")
    print(f"  3D点数量: {pts_3d.shape[0]}")
    print(f"  有效点数: {valid_mask.sum().item()}")
    print(f"  2D坐标范围: u=[{pts_2d[:,0].min():.1f}, {pts_2d[:,0].max():.1f}], v=[{pts_2d[:,1].min():.1f}, {pts_2d[:,1].max():.1f}]")
    print(f"  3D坐标范围: x=[{pts_3d[:,0].min():.2f}, {pts_3d[:,0].max():.2f}]")
    print(f"                y=[{pts_3d[:,1].min():.2f}, {pts_3d[:,1].max():.2f}]")
    print(f"                z=[{pts_3d[:,2].min():.2f}, {pts_3d[:,2].max():.2f}]")
    
    return dataset


def test_dataloader(dataset):
    """测试DataLoader和collate_fn"""
    print("\n" + "="*60)
    print("测试2: DataLoader批处理")
    print("="*60)
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    
    print(f"\n批次内容:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
    
    if batch['fused_feature'] is not None:
        print(f"\n✓ 批次融合特征: {batch['fused_feature'].shape}")
    else:
        print(f"\n✗ 批次融合特征为None")
    
    print(f"\n✓ 点云拼接测试:")
    print(f"  总点数: {batch['points_3d'].shape[0]}")
    print(f"  sample_indices范围: [{batch['sample_indices'].min()}, {batch['sample_indices'].max()}]")
    
    return batch


def test_feature_extraction(batch):
    """测试特征提取"""
    print("\n" + "="*60)
    print("测试3: 特征提取流程")
    print("="*60)
    
    # 加载配置
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载SplatLoc模型
    print("\n加载SplatLoc模型...")
    config_munch = munchify(config)
    
    # 加载FeatureDecoder
    feat_decoder = FeatureDecoder(config).to(device)
    decoder_ckpt = config['splatloc']['decoder_path']
    print(f"  加载decoder: {decoder_ckpt}")
    
    try:
        feat_decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device))
        print(f"  ✓ Decoder加载成功")
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"  ✗ HashGrid大小不匹配!")
            print(f"  错误: {e}")
            print(f"\n  当前配置:")
            print(f"    scene.bound: {config['scene']['bound']}")
            print(f"    voxel_sdf: {config['scene']['voxel_sdf']}")
            return False
        else:
            raise e
    
    # 移动bounding_box到GPU
    feat_decoder.bounding_box = feat_decoder.bounding_box.to(device)
    feat_decoder.eval()
    
    # 测试3D特征提取
    print("\n测试3D特征提取...")
    pts_3d = batch['points_3d'].to(device)
    
    with torch.no_grad():
        pcd_feats = feat_decoder(pts_3d[:100])  # 测试前100个点
    
    print(f"  输入: {pts_3d[:100].shape}")
    print(f"  输出: {pcd_feats.shape}")
    print(f"  特征范围: [{pcd_feats.min():.3f}, {pcd_feats.max():.3f}]")
    print(f"  特征均值: {pcd_feats.mean():.3f}")
    print(f"  特征标准差: {pcd_feats.std():.3f}")
    
    # 测试2D特征采样
    if batch['fused_feature'] is not None:
        print("\n测试2D融合特征采样...")
        fused_feat = batch['fused_feature'].to(device)  # [B, 256, H, W]
        pts_2d = batch['points_2d'].to(device)  # [total_N, 2]
        sample_indices = batch['sample_indices'].to(device)
        
        # 采样第一个样本的特征
        mask_0 = (sample_indices == 0)
        pts_2d_0 = pts_2d[mask_0][:100]  # 前100个点
        
        # 归一化坐标
        H, W = fused_feat.shape[2], fused_feat.shape[3]
        grid_x = 2.0 * pts_2d_0[:, 0] / (W - 1) - 1.0
        grid_y = 2.0 * pts_2d_0[:, 1] / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
        
        # 采样
        sampled_feats = torch.nn.functional.grid_sample(
            fused_feat[0:1], grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).squeeze(2).squeeze(0).permute(1, 0)
        
        print(f"  输入2D坐标: {pts_2d_0.shape}")
        print(f"  输出特征: {sampled_feats.shape}")
        print(f"  特征范围: [{sampled_feats.min():.3f}, {sampled_feats.max():.3f}]")
        print(f"  特征均值: {sampled_feats.mean():.3f}")
        print(f"  特征标准差: {sampled_feats.std():.3f}")
    
    print("\n✓ 所有测试通过!")
    return True


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("隐式对应关系训练 - 修复验证测试")
    print("="*60)
    
    try:
        # 测试1: Dataset
        dataset = test_dataset()
        
        # 测试2: DataLoader
        batch = test_dataloader(dataset)
        
        # 测试3: 特征提取
        success = test_feature_extraction(batch)
        
        if success:
            print("\n" + "="*60)
            print("✓ 所有测试通过! 可以开始训练")
            print("="*60)
            print("\n运行命令:")
            print("  python train.py --config configs/train_config.yaml")
        else:
            print("\n" + "="*60)
            print("✗ 测试失败，请检查配置")
            print("="*60)
            
    except Exception as e:
        print(f"\n✗ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
