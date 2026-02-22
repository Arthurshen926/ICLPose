"""
测试可视化功能
快速验证visualization.py中的函数是否正常工作
"""

import numpy as np
import torch
from pathlib import Path

# 添加路径
import sys
sys.path.append(str(Path(__file__).parent))

from utils.visualization import (
    visualize_pose_prediction,
    visualize_feature_similarity_matrix,
    visualize_2d3d_correspondence
)

def test_pose_visualization():
    """测试位姿预测可视化"""
    print("测试位姿预测可视化...")
    
    # 创建虚拟数据
    gt_pose = np.eye(4)
    gt_pose[:3, 3] = [1.0, 0.5, 0.2]  # GT平移
    
    pred_pose = np.eye(4)
    pred_pose[:3, 3] = [1.1, 0.55, 0.18]  # 预测平移（稍有偏差）
    # 添加小的旋转偏差
    theta = np.deg2rad(5)
    pred_pose[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # 创建虚拟图像
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 保存可视化
    save_path = "/tmp/test_pose_vis.png"
    visualize_pose_prediction(gt_pose, pred_pose, img, save_path)
    print(f"  ✓ 位姿可视化已保存: {save_path}")

def test_similarity_visualization():
    """测试特征相似度矩阵可视化"""
    print("测试特征相似度矩阵可视化...")
    
    # 创建虚拟特征
    img_feats = np.random.randn(100, 256).astype(np.float32)
    pcd_feats = np.random.randn(200, 256).astype(np.float32)
    
    # 保存可视化
    save_path = "/tmp/test_similarity_vis.png"
    visualize_feature_similarity_matrix(img_feats, pcd_feats, save_path)
    print(f"  ✓ 相似度矩阵已保存: {save_path}")

def test_correspondence_visualization():
    """测试2D-3D对应关系可视化"""
    print("测试2D-3D对应关系可视化...")
    
    # 创建虚拟数据
    img_feats = np.random.randn(100, 256).astype(np.float32)
    pcd_feats = np.random.randn(200, 256).astype(np.float32)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pcd_xyz = np.random.randn(200, 3).astype(np.float32) * 2  # 3D点云坐标
    
    # 保存可视化
    save_path = "/tmp/test_correspondence_vis.png"
    visualize_2d3d_correspondence(img_feats, pcd_feats, img, pcd_xyz, save_path)
    print(f"  ✓ 对应关系可视化已保存: {save_path}")

if __name__ == '__main__':
    print("=" * 60)
    print("可视化功能测试")
    print("=" * 60)
    
    try:
        test_pose_visualization()
        print()
        test_similarity_visualization()
        print()
        test_correspondence_visualization()
        print()
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
