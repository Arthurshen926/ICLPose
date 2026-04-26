"""
Diversity Loss for Keypoint Detection

防止所有keypoints坍缩到同一位置，确保检测到的关键点分布在整个图像/点云中
参考: ICL-I2PReg/kitti/stage_2/loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def diversity_loss(keypoints, margin):
    """
    计算keypoint的diversity loss
    
    惩罚距离过近的keypoints，鼓励它们分散分布
    
    Args:
        keypoints: (B, N, 2 or 3) keypoint坐标
        margin: float, 最小距离阈值 (像素或米)
                - 2D: 通常使用10-16像素
                - 3D: 通常使用0.1-0.16米 (10-16cm)
    
    Returns:
        loss: scalar, diversity loss
    
    Implementation:
        1. 计算所有keypoint对之间的距离矩阵
        2. 对于距离 < margin的对，计算loss = relu(margin - distance)
        3. 平均所有对的loss（排除对角线）
    """
    B, N, D = keypoints.shape  # D=2 for 2D, D=3 for 3D
    
    # 计算pairwise距离矩阵 (B, N, N)
    # keypoints: (B, N, D)
    # keypoints.unsqueeze(2): (B, N, 1, D)
    # keypoints.unsqueeze(1): (B, 1, N, D)
    diff = keypoints.unsqueeze(2) - keypoints.unsqueeze(1)  # (B, N, N, D)
    pairwise_dist = torch.norm(diff, dim=-1)  # (B, N, N)
    
    # 惩罚距离 < margin的对
    # loss_matrix[i,j] = max(0, margin - dist(i,j))
    loss_matrix = F.relu(margin - pairwise_dist)  # (B, N, N)
    
    # 排除对角线（自己和自己的距离=0）
    # 创建mask，对角线为0，其余为1
    mask = (1 - torch.eye(N, device=keypoints.device)).unsqueeze(0)  # (1, N, N)
    loss_matrix = loss_matrix * mask  # (B, N, N)
    
    # 平均所有对的loss
    # 总对数: B * N * (N-1)
    loss = loss_matrix.sum() / (B * N * (N - 1) + 1e-8)
    
    return loss


class DiversityLoss(nn.Module):
    """
    Diversity Loss模块（包装版本）
    
    Args:
        margin_2d: 2D keypoint的最小距离阈值（像素）
        margin_3d: 3D keypoint的最小距离阈值（米）
        weight: diversity loss的权重
    """
    def __init__(self, margin_2d=10.0, margin_3d=0.1, weight=0.01):
        super().__init__()
        self.margin_2d = margin_2d
        self.margin_3d = margin_3d
        self.weight = weight
    
    def forward(self, img_keypoints, pcd_keypoints):
        """
        计算总diversity loss
        
        Args:
            img_keypoints: (B, N_query, 2) 2D关键点坐标
            pcd_keypoints: (B, N_query, 3) 3D关键点坐标
            
        Returns:
            loss: scalar
        """
        loss_2d = diversity_loss(img_keypoints, self.margin_2d)
        loss_3d = diversity_loss(pcd_keypoints, self.margin_3d)
        
        return self.weight * (loss_2d + loss_3d)


if __name__ == "__main__":
    """测试diversity loss"""
    # 测试2D keypoints
    keypoints_2d = torch.tensor([
        [[0, 0], [5, 5], [20, 20]],  # batch 0
        [[0, 0], [0, 0], [30, 30]],  # batch 1 - 两个重合点
    ], dtype=torch.float32)
    
    print("2D Keypoints:")
    print(keypoints_2d)
    
    loss_2d = diversity_loss(keypoints_2d, margin=10.0)
    print(f"Diversity Loss (margin=10): {loss_2d.item():.4f}")
    
    # 测试3D keypoints
    keypoints_3d = torch.tensor([
        [[0, 0, 0], [0.05, 0.05, 0.05], [0.2, 0.2, 0.2]],  # batch 0
        [[0, 0, 0], [0, 0, 0], [0.3, 0.3, 0.3]],  # batch 1
    ], dtype=torch.float32)
    
    print("\n3D Keypoints:")
    print(keypoints_3d)
    
    loss_3d = diversity_loss(keypoints_3d, margin=0.1)
    print(f"Diversity Loss (margin=0.1): {loss_3d.item():.4f}")
    
    # 测试模块
    div_loss_module = DiversityLoss(margin_2d=10.0, margin_3d=0.1, weight=0.01)
    total_loss = div_loss_module(keypoints_2d, keypoints_3d)
    print(f"\nTotal Diversity Loss: {total_loss.item():.6f}")
