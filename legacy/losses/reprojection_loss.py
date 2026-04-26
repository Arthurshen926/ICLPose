"""
Reprojection Loss for Keypoint-based Pose Estimation

参考: ICL-I2PReg/kitti/stage_2/loss.py

核心思想:
    将检测到的3D关键点使用GT位姿投影到2D图像平面，
    与检测到的2D关键点计算像素距离，作为几何一致性约束。

作用:
    1. 提供直接的几何监督信号
    2. 加速keypoint对应关系的学习
    3. 与pose loss互补：pose loss监督最终位姿，reprojection loss监督中间表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def project_3d_to_2d(
    points_3d: torch.Tensor,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int] = (640, 480),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将3D点投影到2D图像平面
    
    Args:
        points_3d: (B, N, 3) 世界坐标系下的3D点
        pose: (B, 4, 4) 相机位姿矩阵 (camera-to-world, c2w)
        intrinsics: (B, 3, 3) 相机内参矩阵
        image_size: (width, height) 图像尺寸
        
    Returns:
        points_2d: (B, N, 2) 投影后的2D像素坐标 (u, v)
        valid_mask: (B, N) 有效点掩码（在图像范围内且深度>0）
    """
    B, N, _ = points_3d.shape
    device = points_3d.device
    width, height = image_size
    
    # 1. 计算world-to-camera变换
    # pose是c2w (camera-to-world)，需要求逆得到w2c
    R_c2w = pose[:, :3, :3]  # (B, 3, 3)
    t_c2w = pose[:, :3, 3]   # (B, 3)
    
    # w2c: R_w2c = R_c2w^T, t_w2c = -R_c2w^T @ t_c2w
    R_w2c = R_c2w.transpose(1, 2)  # (B, 3, 3)
    t_w2c = -torch.bmm(R_w2c, t_c2w.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    
    # 2. 将3D点从世界坐标系转换到相机坐标系
    # points_cam = R_w2c @ points_world + t_w2c
    points_cam = torch.bmm(points_3d, R_w2c.transpose(1, 2)) + t_w2c.unsqueeze(1)  # (B, N, 3)
    
    # 3. 投影到图像平面
    # 获取相机内参
    fx = intrinsics[:, 0, 0].unsqueeze(-1)  # (B, 1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1)  # (B, 1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1)  # (B, 1)
    cy = intrinsics[:, 1, 2].unsqueeze(-1)  # (B, 1)
    
    # 透视投影
    z = points_cam[:, :, 2:3]  # (B, N, 1)
    z = torch.clamp(z, min=1e-6)  # 避免除零
    
    x = points_cam[:, :, 0:1]  # (B, N, 1)
    y = points_cam[:, :, 1:2]  # (B, N, 1)
    
    u = fx.unsqueeze(1) * x / z + cx.unsqueeze(1)  # (B, N, 1)
    v = fy.unsqueeze(1) * y / z + cy.unsqueeze(1)  # (B, N, 1)
    
    points_2d = torch.cat([u, v], dim=-1)  # (B, N, 2)
    
    # 4. 计算有效掩码
    # 条件：深度 > 0 且 像素坐标在图像范围内
    valid_mask = (
        (points_cam[:, :, 2] > 0.05) &  # 深度 > 5cm
        (points_2d[:, :, 0] >= 0) & (points_2d[:, :, 0] < width) &
        (points_2d[:, :, 1] >= 0) & (points_2d[:, :, 1] < height)
    )  # (B, N)
    
    return points_2d, valid_mask


def reprojection_loss(
    img_keypoints: torch.Tensor,
    pcd_keypoints: torch.Tensor,
    gt_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int] = (640, 480),
    reduction: str = 'mean',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算重投影损失
    
    将3D关键点使用GT位姿投影到2D，与检测到的2D关键点计算距离
    
    Args:
        img_keypoints: (B, N_query, 2) 检测到的2D关键点坐标 (u, v) 像素坐标
        pcd_keypoints: (B, N_query, 3) 检测到的3D关键点坐标 (x, y, z) 世界坐标
        gt_pose: (B, 4, 4) GT相机位姿 (camera-to-world)
        intrinsics: (B, 3, 3) 相机内参矩阵
        image_size: (width, height) 图像尺寸
        reduction: 'mean', 'sum', 或 'none'
        
    Returns:
        loss: 重投影损失（像素距离）
        valid_ratio: 有效关键点比例
    """
    B, N, _ = pcd_keypoints.shape
    device = pcd_keypoints.device
    
    # 1. 将3D关键点投影到2D
    projected_2d, valid_mask = project_3d_to_2d(
        pcd_keypoints, gt_pose, intrinsics, image_size
    )  # (B, N, 2), (B, N)
    
    # 2. 计算像素距离
    # reprojection error = ||projected_2d - img_keypoints||_2
    diff = projected_2d - img_keypoints  # (B, N, 2)
    pixel_dist = torch.norm(diff, dim=-1)  # (B, N)
    
    # 3. 只对有效点计算损失（过滤掉投影到图像外的点）
    # 同时过滤掉距离过大的异常点（可能是错误的对应）
    max_dist_threshold = 200.0  # 最大200像素
    valid_mask = valid_mask & (pixel_dist < max_dist_threshold)
    
    # 4. 计算损失
    if valid_mask.sum() > 0:
        valid_dist = pixel_dist[valid_mask]
        
        if reduction == 'mean':
            loss = valid_dist.mean()
        elif reduction == 'sum':
            loss = valid_dist.sum()
        else:
            loss = pixel_dist  # 保持原始形状
    else:
        # 没有有效点，返回0损失
        loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # 5. 计算有效比例（用于监控）
    valid_ratio = valid_mask.float().mean()
    
    return loss, valid_ratio


class ReprojectionLoss(nn.Module):
    """
    Reprojection Loss 模块（包装版本）
    
    用于训练时计算2D-3D keypoint几何一致性损失
    
    Args:
        weight: 损失权重
        image_size: (width, height) 图像尺寸
        max_dist_threshold: 最大像素距离阈值（过滤异常点）
    """
    
    def __init__(
        self,
        weight: float = 0.1,
        image_size: Tuple[int, int] = (640, 480),
        max_dist_threshold: float = 200.0,
    ):
        super().__init__()
        self.weight = weight
        self.image_size = image_size
        self.max_dist_threshold = max_dist_threshold
        
        print(f"[ReprojectionLoss] weight={weight}, image_size={image_size}")
    
    def forward(
        self,
        img_keypoints: torch.Tensor,
        pcd_keypoints: torch.Tensor,
        gt_pose: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            img_keypoints: (B, N_query, 2) 检测到的2D关键点坐标
            pcd_keypoints: (B, N_query, 3) 检测到的3D关键点坐标
            gt_pose: (B, 4, 4) GT相机位姿
            intrinsics: (B, 3, 3) 相机内参矩阵
            
        Returns:
            weighted_loss: 加权后的损失
            info: 包含详细信息的字典
        """
        loss, valid_ratio = reprojection_loss(
            img_keypoints=img_keypoints,
            pcd_keypoints=pcd_keypoints,
            gt_pose=gt_pose,
            intrinsics=intrinsics,
            image_size=self.image_size,
            reduction='mean',
        )
        
        weighted_loss = self.weight * loss
        
        info = {
            'reprojection_loss': loss.item() if loss.requires_grad else loss,
            'reprojection_loss_weighted': weighted_loss.item() if weighted_loss.requires_grad else weighted_loss,
            'reprojection_valid_ratio': valid_ratio.item(),
        }
        
        return weighted_loss, info


if __name__ == "__main__":
    """测试 Reprojection Loss"""
    print("测试 Reprojection Loss\n")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟数据
    B, N = 2, 16  # batch_size=2, 16个关键点
    
    # 3D关键点（世界坐标系）
    pcd_keypoints = torch.randn(B, N, 3, device=device)
    pcd_keypoints[:, :, 2] = torch.abs(pcd_keypoints[:, :, 2]) + 1.0  # 确保z>0
    
    # 相机位姿（单位矩阵，相机在原点）
    gt_pose = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
    gt_pose[:, :3, 3] = torch.randn(B, 3, device=device) * 0.5  # 添加小平移
    
    # 相机内参
    fx, fy = 320.0, 320.0
    cx, cy = 319.5, 239.5
    intrinsics = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=device).unsqueeze(0).expand(B, 3, 3)
    
    # 使用真实投影生成"完美"的2D关键点
    projected_2d, valid_mask = project_3d_to_2d(
        pcd_keypoints, gt_pose, intrinsics, (640, 480)
    )
    
    print(f"3D keypoints shape: {pcd_keypoints.shape}")
    print(f"Projected 2D shape: {projected_2d.shape}")
    print(f"Valid mask sum: {valid_mask.sum().item()}")
    
    # 测试1: 完美匹配（2D = 投影后的3D）
    print("\n--- 测试1: 完美匹配 ---")
    img_keypoints_perfect = projected_2d.clone()
    loss1, info1 = ReprojectionLoss(weight=0.1)(
        img_keypoints_perfect, pcd_keypoints, gt_pose, intrinsics
    )
    print(f"Loss (perfect): {info1['reprojection_loss']:.4f}")
    print(f"Valid ratio: {info1['reprojection_valid_ratio']:.4f}")
    
    # 测试2: 添加噪声
    print("\n--- 测试2: 添加噪声 ---")
    noise = torch.randn_like(projected_2d) * 10.0  # 10像素噪声
    img_keypoints_noisy = projected_2d + noise
    loss2, info2 = ReprojectionLoss(weight=0.1)(
        img_keypoints_noisy, pcd_keypoints, gt_pose, intrinsics
    )
    print(f"Loss (noisy, ~10px): {info2['reprojection_loss']:.4f}")
    print(f"Valid ratio: {info2['reprojection_valid_ratio']:.4f}")
    
    # 测试3: 大噪声
    print("\n--- 测试3: 大噪声 ---")
    noise_large = torch.randn_like(projected_2d) * 50.0  # 50像素噪声
    img_keypoints_large_noise = projected_2d + noise_large
    loss3, info3 = ReprojectionLoss(weight=0.1)(
        img_keypoints_large_noise, pcd_keypoints, gt_pose, intrinsics
    )
    print(f"Loss (noisy, ~50px): {info3['reprojection_loss']:.4f}")
    print(f"Valid ratio: {info3['reprojection_valid_ratio']:.4f}")
    
    print("\n✓ Reprojection Loss 测试完成!")
