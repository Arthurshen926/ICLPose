"""
快速测试特征提取和归一化
"""
import torch
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from data.dataset import CorrespondenceDataset
from torch.utils.data import DataLoader

def test_features():
    print("=" * 60)
    print("测试特征提取和归一化")
    print("=" * 60)
    
    # 加载配置
    with open("configs/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建数据集（只取1个样本）
    dataset = CorrespondenceDataset(
        data_root=config['dataset']['data_root'],
        scene_name=config['dataset']['val_scene'],
        image_size=tuple(config['dataset']['image_size']),
        max_samples=1
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    
    print(f"\n[Batch Info]")
    print(f"  Image: {batch['image'].shape}")
    print(f"  Fused feature: {batch['fused_feature'].shape if batch['fused_feature'] is not None else 'None'}")
    print(f"  Points 2D: {batch['points_2d'].shape}")
    print(f"  Points 3D: {batch['points_3d'].shape}")
    
    if batch['fused_feature'] is not None:
        feat = batch['fused_feature'][0]  # [256, 35, 46]
        print(f"\n[2D Fused Feature Stats (before normalization)]")
        print(f"  Shape: {feat.shape}")
        print(f"  Mean: {feat.mean():.4f}")
        print(f"  Std: {feat.std():.4f}")
        print(f"  Range: [{feat.min():.4f}, {feat.max():.4f}]")
        
        # 模拟采样后的特征
        sampled = feat[:, 0, 0]  # 取一个点的特征 [256]
        print(f"\n[Sample point feature]")
        print(f"  Before L2 norm: ||f|| = {torch.norm(sampled):.4f}")
        
        # L2归一化
        sampled_norm = torch.nn.functional.normalize(sampled.unsqueeze(0), p=2, dim=-1)
        print(f"  After L2 norm: ||f|| = {torch.norm(sampled_norm):.4f}")
        print(f"  Mean: {sampled_norm.mean():.4f}, Std: {sampled_norm.std():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ 特征统计检查完成")
    print("=" * 60)
    
    print("\n[Expected behavior]")
    print("  - 2D和3D特征都应该L2归一化到单位长度 (||f|| = 1.0)")
    print("  - 归一化后的特征余弦相似度 = 点积")
    print("  - 这样能确保2D-3D匹配只关注方向，不受特征幅度影响")

if __name__ == '__main__':
    test_features()
