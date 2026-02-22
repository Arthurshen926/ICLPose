"""
诊断特征提取问题
检查2D和3D特征是否正常
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from gaussian_splatting.scene.gaussian_model import GaussianModel
from models.decoders import FeatureDecoder
from data.dataset import CorrespondenceDataset
from torch.utils.data import DataLoader

def diagnose_features():
    print("=" * 80)
    print("特征诊断工具")
    print("=" * 80)
    
    # 加载配置
    config_path = "configs/train_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载SplatLoc模型
    print("\n[1] 加载SplatLoc模型...")
    splatloc_cfg = config['splatloc']
    
    # 加载Gaussians
    gaussians = GaussianModel(sh_degree=0, config={'Training': {'primitive_reg': False}})
    gaussians.load_ply(splatloc_cfg['gaussians_path'])
    print(f"  ✓ Gaussian点数: {gaussians.get_xyz.shape[0]}")
    
    # 加载Decoder
    decoder_config = {
        'scene': config['scene'],
        'decoder': config['decoder'],
        'Training': {'primitive_reg': False}
    }
    feat_decoder = FeatureDecoder(config=decoder_config, input_ch=3).to(device)
    feat_decoder.bounding_box = feat_decoder.bounding_box.to(device)
    
    checkpoint = torch.load(splatloc_cfg['decoder_path'], map_location=device)
    if 'decoder_state_dict' in checkpoint:
        feat_decoder.load_state_dict(checkpoint['decoder_state_dict'], strict=False)
    else:
        feat_decoder.load_state_dict(checkpoint, strict=False)
    
    feat_decoder.eval()
    print(f"  ✓ Decoder加载完成")
    
    # 2. 加载数据集
    print("\n[2] 加载验证数据集...")
    val_dataset = CorrespondenceDataset(
        data_root=config['dataset']['data_root'],
        scene_name=config['dataset']['val_scene'],
        image_size=tuple(config['dataset']['image_size']),
        sample_step=config['dataset'].get('val_step', 1),
        max_samples=10,  # 只取10个样本用于诊断
        augment=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"  ✓ 验证集样本数: {len(val_dataset)}")
    
    # 3. 测试特征提取
    print("\n[3] 测试特征提取...")
    
    batch = next(iter(val_loader))
    
    print(f"  Batch keys: {batch.keys()}")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Pose shape: {batch['pose'].shape}")
    
    # 检查图像
    img = batch['image'][0].cpu().numpy()
    print(f"\n  图像统计:")
    print(f"    Shape: {img.shape}")
    print(f"    Mean: {img.mean():.4f}")
    print(f"    Std: {img.std():.4f}")
    print(f"    Range: [{img.min():.4f}, {img.max():.4f}]")
    
    # 检查是否有融合特征
    if 'fused_feature' in batch and batch['fused_feature'] is not None:
        fused_feat = batch['fused_feature'][0]
        print(f"\n  融合特征统计:")
        print(f"    Shape: {fused_feat.shape}")
        print(f"    Mean: {fused_feat.mean():.4f}")
        print(f"    Std: {fused_feat.std():.4f}")
        print(f"    Range: [{fused_feat.min():.4f}, {fused_feat.max():.4f}]")
    else:
        print(f"\n  ⚠️ 没有找到融合特征!")
    
    # 检查3D点云特征
    if 'pcd_xyz' in batch:
        pcd_xyz = batch['pcd_xyz'][0].to(device)
        print(f"\n  3D点云:")
        print(f"    Shape: {pcd_xyz.shape}")
        print(f"    Range: X[{pcd_xyz[:, 0].min():.2f}, {pcd_xyz[:, 0].max():.2f}], "
              f"Y[{pcd_xyz[:, 1].min():.2f}, {pcd_xyz[:, 1].max():.2f}], "
              f"Z[{pcd_xyz[:, 2].min():.2f}, {pcd_xyz[:, 2].max():.2f}]")
        
        # 查询3D特征
        with torch.no_grad():
            pcd_feats = feat_decoder.query_feature(pcd_xyz)
        
        print(f"\n  3D特征 (从decoder查询):")
        print(f"    Shape: {pcd_feats.shape}")
        print(f"    Mean: {pcd_feats.mean():.4f}")
        print(f"    Std: {pcd_feats.std():.4f}")
        print(f"    Range: [{pcd_feats.min():.4f}, {pcd_feats.max():.4f}]")
        
        # 计算特征范数
        norms = torch.norm(pcd_feats, dim=-1)
        print(f"    L2 Norm: mean={norms.mean():.4f}, std={norms.std():.4f}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)
    
    # 4. 建议
    print("\n[建议]")
    
    if img.min() < -2 or img.max() > 3:
        print("  ⚠️ 图像归一化可能有问题（值超出正常范围）")
    
    if 'fused_feature' not in batch or batch['fused_feature'] is None:
        print("  🔴 缺少融合特征！这可能是主要问题！")
        print("     需要先运行SplatLoc的特征提取生成fused_feature.npy文件")
    else:
        fused_feat = batch['fused_feature'][0]
        if fused_feat.std() < 0.1:
            print("  ⚠️ 融合特征方差太小，可能未正确生成")
        if abs(fused_feat.mean()) < 0.01 and fused_feat.std() < 0.01:
            print("  🔴 融合特征接近0，可能是空白数据！")

if __name__ == '__main__':
    diagnose_features()
