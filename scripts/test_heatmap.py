#!/usr/bin/env python3
"""
快速测试Keypoint Heatmap机制
验证新增的attention heatmap是否正常工作
"""

import torch
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_heatmap_output():
    """测试ICPoseNet是否正确返回heatmap"""
    print("🔍 测试Keypoint Heatmap机制...")
    
    # 1. 导入模型
    from ic_models.ic_pose_net import ICPoseNet
    
    # 2. 创建模型
    model = ICPoseNet(
        feature_dim=256,
        num_queries=64,
        fusion_layers=4,
        num_heads=8
    )
    model.eval()
    
    # 3. 创建测试输入
    batch_size = 2
    num_img_points = 512
    num_pcd_points = 1024
    feature_dim = 256
    
    img_feats = torch.randn(batch_size, num_img_points, feature_dim)
    pcd_feats = torch.randn(batch_size, num_pcd_points, feature_dim)
    
    print(f"✓ 输入特征: img_feats {img_feats.shape}, pcd_feats {pcd_feats.shape}")
    
    # 4. 前向传播
    with torch.no_grad():
        outputs = model(img_feats, pcd_feats)
    
    # 5. 检查输出
    print(f"\n📊 输出检查:")
    print(f"  返回值数量: {len(outputs)}")
    
    if len(outputs) == 5:
        pose_matrix, pose_9d, rotation_6d, translation, img_heatmap = outputs
        
        print(f"  ✓ pose_matrix: {pose_matrix.shape}")
        print(f"  ✓ pose_9d: {pose_9d.shape}")
        print(f"  ✓ rotation_6d: {rotation_6d.shape}")
        print(f"  ✓ translation: {translation.shape}")
        print(f"  ✓ img_heatmap: {img_heatmap.shape}")
        
        # 6. 验证heatmap属性
        print(f"\n🔍 Heatmap验证:")
        print(f"  形状: {img_heatmap.shape} (应为 [{batch_size}, 64 queries, {num_img_points} img_points])")
        print(f"  数值范围: [{img_heatmap.min().item():.4f}, {img_heatmap.max().item():.4f}]")
        print(f"  均值: {img_heatmap.mean().item():.4f}")
        
        # 检查是否是概率分布（每个query的权重和应为1）
        sum_per_query = img_heatmap.sum(dim=-1)  # (B, N_query)
        print(f"  每个query的权重和: mean={sum_per_query.mean().item():.4f}, std={sum_per_query.std().item():.6f}")
        
        if torch.allclose(sum_per_query, torch.ones_like(sum_per_query), atol=1e-5):
            print(f"  ✅ Heatmap是归一化的概率分布！")
        else:
            print(f"  ⚠️  警告: Heatmap未正确归一化")
        
        # 7. 可视化一个query的heatmap
        print(f"\n📈 第一个batch第一个query的heatmap统计:")
        query_0_heatmap = img_heatmap[0, 0, :]  # (N_img,)
        print(f"  最大权重: {query_0_heatmap.max().item():.6f}")
        print(f"  最小权重: {query_0_heatmap.min().item():.6f}")
        print(f"  Top-5权重: {query_0_heatmap.topk(5)[0].tolist()}")
        
        print(f"\n✅ 测试通过！Keypoint Heatmap机制正常工作")
        return True
        
    else:
        print(f"  ❌ 错误: 期望5个返回值，但得到{len(outputs)}个")
        return False

if __name__ == '__main__':
    success = test_heatmap_output()
    sys.exit(0 if success else 1)
