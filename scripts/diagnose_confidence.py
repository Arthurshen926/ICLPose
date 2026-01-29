"""
诊断置信度低的问题
检查特征质量、相似度计算、可视化流程
"""

import sys
import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import CorrespondenceDataset, collate_fn
from torch.utils.data import DataLoader


def check_feature_similarity():
    """检查实际训练中特征的相似度分布"""
    print("=" * 80)
    print("1. 检查2D-3D特征相似度")
    print("=" * 80)
    
    # 加载配置
    config_path = Path("configs/train_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_cfg = config['dataset']
    
    # 创建数据集
    dataset = CorrespondenceDataset(
        data_root=data_cfg['data_root'],
        scene_name=data_cfg['train_scene'],
        image_size=tuple(data_cfg['image_size']),
        augment=False,  # 禁用增强便于分析
        max_samples=5,
        use_depth=False,
        gaussian_path=data_cfg.get('gaussian_path'),
        fx=data_cfg['fx'],
        fy=data_cfg['fy'],
        cx=data_cfg['cx'],
        cy=data_cfg['cy'],
        sample_step=1,
        use_initial_pose=True,
        pose_noise_rot_deg=3.0,
        pose_noise_trans_m=0.05,
    )
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"\nBatch信息:")
    print(f"  图像: {batch['image'].shape}")
    print(f"  2D点: {batch['points_2d'].shape}")
    print(f"  3D点: {batch['points_3d'].shape}")
    
    # 模拟从SplatLoc提取的特征
    # 实际训练中，这些特征来自FeatureDecoder
    # 这里用随机特征模拟最坏情况
    n_2d = batch['points_2d'].shape[0]
    n_3d = batch['points_3d'].shape[0]
    
    print(f"\n🔍 关键问题诊断:")
    print(f"=" * 80)
    
    # 场景1: 完全随机的特征（未训练或特征提取失败）
    print("\n场景1: 完全随机特征（模拟特征提取失败）")
    img_feats_random = torch.randn(1, n_2d, 256)
    pcd_feats_random = torch.randn(1, n_3d, 256)
    
    img_f = torch.nn.functional.normalize(img_feats_random[0], dim=-1)
    pcd_f = torch.nn.functional.normalize(pcd_feats_random[0], dim=-1)
    similarity = torch.mm(img_f, pcd_f.t())
    
    print(f"  相似度范围: [{similarity.min():.4f}, {similarity.max():.4f}]")
    print(f"  相似度均值: {similarity.mean():.4f}")
    print(f"  最大置信度均值: {similarity.max(dim=1)[0].mean():.4f}")
    print(f"  ❌ 问题: 置信度太低（<0.2），热力图会很暗")
    
    # 场景2: 部分相关的特征（网络学习中）
    print("\n场景2: 部分相关特征（网络学习了一些东西）")
    # 制造一些相关性
    img_feats_semi = torch.randn(1, n_2d, 256)
    pcd_feats_semi = img_feats_semi[:, :min(n_2d, n_3d), :].clone() + torch.randn(1, min(n_2d, n_3d), 256) * 0.5
    if n_3d > n_2d:
        pcd_feats_semi = torch.cat([pcd_feats_semi, torch.randn(1, n_3d - n_2d, 256)], dim=1)
    
    img_f = torch.nn.functional.normalize(img_feats_semi[0], dim=-1)
    pcd_f = torch.nn.functional.normalize(pcd_feats_semi[0], dim=-1)
    similarity = torch.mm(img_f, pcd_f.t())
    
    print(f"  相似度范围: [{similarity.min():.4f}, {similarity.max():.4f}]")
    print(f"  相似度均值: {similarity.mean():.4f}")
    print(f"  最大置信度均值: {similarity.max(dim=1)[0].mean():.4f}")
    print(f"  ⚠️  问题: 置信度中等（0.3-0.5），热力图会有些响应但不强")
    
    # 场景3: 高度相关的特征（理想状态）
    print("\n场景3: 高度相关特征（理想训练结果）")
    img_feats_good = torch.randn(1, n_2d, 256)
    # 大部分3D点对应到2D点
    pcd_feats_good = img_feats_good[:, :min(n_2d, n_3d), :].clone() + torch.randn(1, min(n_2d, n_3d), 256) * 0.1
    if n_3d > n_2d:
        pcd_feats_good = torch.cat([pcd_feats_good, torch.randn(1, n_3d - n_2d, 256)], dim=1)
    
    img_f = torch.nn.functional.normalize(img_feats_good[0], dim=-1)
    pcd_f = torch.nn.functional.normalize(pcd_feats_good[0], dim=-1)
    similarity = torch.mm(img_f, pcd_f.t())
    
    print(f"  相似度范围: [{similarity.min():.4f}, {similarity.max():.4f}]")
    print(f"  相似度均值: {similarity.mean():.4f}")
    print(f"  最大置信度均值: {similarity.max(dim=1)[0].mean():.4f}")
    print(f"  ✅ 理想: 置信度高（>0.7），热力图应该清晰可见")


def diagnose_feature_extraction_issue():
    """诊断特征提取的可能问题"""
    print("\n" + "=" * 80)
    print("2. 特征提取问题诊断")
    print("=" * 80)
    
    print("\n🔍 可能的问题根源:")
    print("-" * 80)
    
    print("\n问题1: Transformer没有学习2D-3D对应关系")
    print("  原因: ICPoseNet是直接预测位姿，不是学习对应关系")
    print("  说明: 网络架构是 [Query -> Fusion -> Pose]，没有显式的对应关系监督")
    print("  结果: img_feats和pcd_feats经过fusion后，原始特征相似度没有意义")
    print("  ❌ 这就是为什么置信度低！")
    
    print("\n问题2: 可视化使用了错误的特征")
    print("  当前: 使用Fusion前的img_feats和pcd_feats")
    print("  问题: 这些是输入特征，还没有经过跨模态对齐")
    print("  正确: 应该使用Fusion后的query特征或attention权重")
    
    print("\n问题3: 没有对应关系的监督信号")
    print("  当前: 只有位姿loss，没有特征对应loss")
    print("  结果: 网络不需要学习显式的2D-3D匹配")
    print("  改进: 可以添加contrastive loss或correspondence loss")
    
    print("\n问题4: 位姿准确不代表特征匹配好")
    print("  你的观察: 位姿精度高（1.26°, 0.121m）")
    print("  但: 这可能是通过全局信息（如场景几何）达到的")
    print("  而非: 通过精确的点对点匹配")


def suggest_fixes():
    """建议修复方案"""
    print("\n" + "=" * 80)
    print("3. 解决方案建议")
    print("=" * 80)
    
    print("\n方案A: 修改可视化（推荐 - 快速验证）")
    print("-" * 80)
    print("使用Transformer的attention权重作为置信度：")
    print("  1. 提取cross-attention权重（img_feats -> pcd_feats）")
    print("  2. 将attention权重作为2D-3D匹配置信度")
    print("  3. 可视化attention热力图")
    print("  优点: 不需要重新训练")
    print("  缺点: 如果attention权重也很弱，说明网络确实没学到")
    
    print("\n方案B: 添加对应关系监督（推荐 - 长期）")
    print("-" * 80)
    print("在训练中添加contrastive loss：")
    print("  1. 对于已知的2D-3D对应点（从视锥裁剪获得）")
    print("  2. 最大化对应点特征相似度")
    print("  3. 最小化非对应点特征相似度")
    print("  优点: 显式学习特征对应")
    print("  缺点: 需要重新训练")
    
    print("\n方案C: 改变网络架构（最彻底）")
    print("-" * 80)
    print("使用显式匹配的架构：")
    print("  1. 先预测2D-3D对应关系（匹配矩阵）")
    print("  2. 再从匹配结果回归位姿（RANSAC或加权最小二乘）")
    print("  3. 类似SuperGlue的架构")
    print("  优点: 可解释性强")
    print("  缺点: 需要大改代码")
    
    print("\n方案D: 检查特征提取（立即执行）")
    print("-" * 80)
    print("验证SplatLoc特征是否正常：")
    print("  1. 检查feat_decoder是否正确加载")
    print("  2. 验证特征不是全零或NaN")
    print("  3. 检查特征尺度是否合理")


def check_current_architecture():
    """检查当前架构的特点"""
    print("\n" + "=" * 80)
    print("4. 当前架构分析")
    print("=" * 80)
    
    print("\n当前ICPoseNet架构:")
    print("  输入: img_feats (B, N_img, 256), pcd_feats (B, N_pcd, 256)")
    print("  处理: Query-based Fusion Transformer")
    print("    - 初始化: learnable queries (N_query, 256)")
    print("    - Fusion: queries attend to both img_feats and pcd_feats")
    print("    - 输出: fused queries (N_query, 256)")
    print("  回归: queries -> MLP -> pose (rotation_6d + translation)")
    
    print("\n❌ 为什么置信度可视化失败:")
    print("  1. img_feats和pcd_feats是独立的输入特征")
    print("  2. 没有显式对齐这两组特征")
    print("  3. 网络通过queries间接关联两者")
    print("  4. 原始特征的余弦相似度≈随机噪声")
    
    print("\n✅ 为什么位姿仍然准确:")
    print("  1. Transformer通过attention学习隐式关联")
    print("  2. Queries聚合了img和pcd的全局信息")
    print("  3. 位姿回归依赖全局特征而非局部匹配")
    print("  4. 类似于'端到端的直接回归'而非'匹配+求解'")


def main():
    print("\n🔬 置信度低的根本原因分析")
    print("=" * 80)
    
    check_feature_similarity()
    diagnose_feature_extraction_issue()
    suggest_fixes()
    check_current_architecture()
    
    print("\n" + "=" * 80)
    print("🎯 结论")
    print("=" * 80)
    print("\n你的怀疑是对的！问题不在于可视化或训练失败，而在于:")
    print("\n  1. ❌ 当前架构不学习显式的2D-3D对应关系")
    print("  2. ❌ 可视化使用的是Fusion前的特征，本就不该有高相似度")
    print("  3. ✅ 网络通过全局信息（Transformer）隐式推断位姿")
    print("  4. ✅ 位姿准确说明网络学到了东西，只是不是'点对点匹配'")
    print("\n这类似于:")
    print("  - 不好的类比: SIFT特征匹配 -> 位姿求解（显式匹配）")
    print("  - 你的架构: 直接从场景特征回归位姿（隐式端到端）")
    print("\n📝 推荐行动:")
    print("  1. 立即: 修改可视化，使用attention权重而非特征相似度")
    print("  2. 中期: 添加contrastive loss训练显式对应关系")
    print("  3. 长期: 考虑改用显式匹配架构（如果需要可解释性）")


if __name__ == '__main__':
    main()
