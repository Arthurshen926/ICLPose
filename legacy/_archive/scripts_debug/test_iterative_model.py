#!/usr/bin/env python
"""测试迭代精化模型"""
import torch
import sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')

from ic_models.ic_pose_net_iterative import ICPoseNetIterative, ICPoseNetIterativeLite

def test_model():
    print("=" * 60)
    print("测试 ICPoseNetIterative")
    print("=" * 60)
    
    # 测试模型实例化
    model = ICPoseNetIterative(
        feature_dim=256,
        num_queries=64,
        num_stages=3,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1
    )
    print(f'✓ ICPoseNetIterative 创建成功')
    print(f'  总参数量: {sum(p.numel() for p in model.parameters()):,}')
    
    # 测试前向传播
    batch_size = 2
    n_img = 1024
    n_pcd = 2048
    
    img_feats = torch.randn(batch_size, n_img, 256)
    pcd_feats = torch.randn(batch_size, n_pcd, 256)
    img_pixels = torch.rand(batch_size, n_img, 2) * 640
    pcd_points = torch.randn(batch_size, n_pcd, 3)
    img_pos = torch.randn(batch_size, n_img, 256)
    pcd_pos = torch.randn(batch_size, n_pcd, 256)
    initial_pose = torch.eye(4).unsqueeze(0).expand(batch_size, 4, 4).clone()
    
    # 添加一些噪声到初始位姿
    initial_pose[:, :3, 3] = initial_pose[:, :3, 3] + torch.randn(batch_size, 3) * 0.1
    
    outputs = model(img_feats, pcd_feats, img_pixels, pcd_points, 
                    img_pos, pcd_pos, initial_pose=initial_pose)
    
    print(f'✓ 前向传播成功')
    print(f'  输出位姿形状: {outputs[0].shape}')
    print(f'  阶段位姿数量: {len(outputs[7])}')
    
    # 检查位姿是否在迭代中变化
    stage_poses = outputs[7]
    print(f'\n各阶段位姿变化:')
    for i, pose in enumerate(stage_poses):
        t = pose[0, :3, 3]
        print(f'  Stage {i}: t = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]')
    
    print("\n" + "=" * 60)
    print("测试 ICPoseNetIterativeLite")
    print("=" * 60)
    
    model_lite = ICPoseNetIterativeLite(
        feature_dim=256,
        num_queries=64,
        num_stages=3,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1
    )
    print(f'✓ ICPoseNetIterativeLite 创建成功')
    print(f'  总参数量: {sum(p.numel() for p in model_lite.parameters()):,}')
    print(f'  参数量减少: {100*(1-sum(p.numel() for p in model_lite.parameters())/sum(p.numel() for p in model.parameters())):.1f}%')
    
    print("\n✅ 所有测试通过!")

if __name__ == "__main__":
    test_model()
