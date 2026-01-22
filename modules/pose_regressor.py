"""
位姿回归模块
从ICL-I2PReg的model.py改编，从融合特征回归相机位姿

参考: ICL-I2PReg/kitti/stage_2/model.py (FeatureFusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    将6D旋转表示转换为旋转矩阵（连续且稳定）
    
    参考: Zhou et al. "On the Continuity of Rotation Representations in Neural Networks" CVPR 2019
    
    Args:
        d6: (B, 6) 6D旋转表示，表示旋转矩阵的前两列
            d6[:, :3] = 第一列向量 a1
            d6[:, 3:] = 第二列向量 a2
    
    Returns:
        R: (B, 3, 3) 旋转矩阵
    
    算法:
        1. 归一化a1: b1 = a1 / ||a1||
        2. 正交化a2: u2 = a2 - (a2·b1)b1
        3. 归一化b2: b2 = u2 / ||u2||
        4. 第三列: b3 = b1 × b2
        5. 旋转矩阵: R = [b1, b2, b3]
    """
    a1 = d6[:, :3]  # (B, 3)
    a2 = d6[:, 3:]  # (B, 3)
    
    # 归一化第一列
    b1 = F.normalize(a1, dim=1)  # (B, 3)
    
    # 正交化第二列
    dot = (a2 * b1).sum(dim=1, keepdim=True)  # (B, 1)
    u2 = a2 - dot * b1  # (B, 3)
    b2 = F.normalize(u2, dim=1)  # (B, 3)
    
    # 第三列：叉积
    b3 = torch.cross(b1, b2, dim=1)  # (B, 3)
    
    # 组合为旋转矩阵
    R = torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)
    
    return R


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵转换为6D表示
    
    Args:
        R: (B, 3, 3) 旋转矩阵
    
    Returns:
        d6: (B, 6) 6D旋转表示（前两列的元素）
    """
    # R的形状是(B, 3, 3)，我们需要前两列
    # R[:, :, 0]是第一列，R[:, :, 1]是第二列
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)  # (B, 6)


class PoseRegressor(nn.Module):
    """
    位姿回归器（使用6D旋转表示和相对位姿）
    
    从融合后的查询特征回归6-DOF相机位姿
    输入: fused_feats (B, N_query, 256)
    输出: pose (B, 9) - [tx, ty, tz, r1, r2, r3, r4, r5, r6] (相对平移+6D旋转)
    
    改进:
        1. 使用6D旋转表示（Zhou et al. CVPR 2019），替代轴角表示
        2. 预测相对位姿（相对初始位姿），而非绝对位姿
        3. 更稳定的梯度和收敛特性
    
    架构:
        1. 全局平均池化聚合查询特征
        2. MLP回归位姿参数
    """
    
    def __init__(self, feature_dim=256, hidden_dim=512, output_dim=9):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（9-DOF: 3平移+6旋转）
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 特征聚合
        self.feature_aggregation = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 旋转回归器（输出6D旋转表示）
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 6)  # 6D旋转表示
        )
        
        # 平移回归器（输出相对平移）
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # 3D平移向量（相对）
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, fused_feats, query_padding_mask=None):
        """
        前向传播
        
        Args:
            fused_feats: (B, N_query, C) 融合后的查询特征
            query_padding_mask: (B, N_query) padding掩码，True表示无效位置
            
        Returns:
            pose: (B, 9) 位姿参数 [tx, ty, tz, r1, r2, r3, r4, r5, r6]
            rotation_6d: (B, 6) 6D旋转表示
            translation: (B, 3) 平移向量（相对）
        """
        # 全局平均池化（考虑padding mask）
        if query_padding_mask is not None:
            # 将padding位置的特征置零
            mask = (~query_padding_mask).float().unsqueeze(-1)  # (B, N, 1)
            masked_feats = fused_feats * mask  # (B, N, C)
            global_feat = masked_feats.sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, C)
        else:
            global_feat = fused_feats.mean(dim=1)  # (B, C)
        
        # 特征聚合
        feat = self.feature_aggregation(global_feat)  # (B, hidden_dim)
        
        # 回归旋转和平移
        rotation_6d = self.rotation_head(feat)  # (B, 6) 6D旋转表示
        translation = self.translation_head(feat)  # (B, 3) 相对平移
        
        # 拼接为完整位姿
        pose = torch.cat([translation, rotation_6d], dim=-1)  # (B, 9)
        
        return pose, rotation_6d, translation


class PoseRegressorQuaternion(nn.Module):
    """
    位姿回归器（四元数版本）
    
    输出四元数表示的旋转（更稳定）
    输出: pose (B, 7) - [tx, ty, tz, qw, qx, qy, qz]
    """
    
    def __init__(self, feature_dim=256, hidden_dim=512):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 特征聚合
        self.feature_aggregation = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 旋转回归器（输出四元数）
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # 四元数 [qw, qx, qy, qz]
        )
        
        # 平移回归器
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # 3D平移向量
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, fused_feats, query_padding_mask=None):
        """
        前向传播
        
        Args:
            fused_feats: (B, N_query, C) 融合后的查询特征
            query_padding_mask: (B, N_query) padding掩码
            
        Returns:
            pose: (B, 7) 位姿参数 [tx, ty, tz, qw, qx, qy, qz]
            rotation: (B, 4) 四元数（归一化后）
            translation: (B, 3) 平移向量
        """
        # 全局平均池化
        if query_padding_mask is not None:
            mask = (~query_padding_mask).float().unsqueeze(-1)
            masked_feats = fused_feats * mask
            global_feat = masked_feats.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        else:
            global_feat = fused_feats.mean(dim=1)
        
        # 特征聚合
        feat = self.feature_aggregation(global_feat)
        
        # 回归旋转和平移
        rotation = self.rotation_head(feat)  # (B, 4) 未归一化的四元数
        translation = self.translation_head(feat)  # (B, 3)
        
        # 归一化四元数
        rotation = F.normalize(rotation, p=2, dim=-1)  # (B, 4)
        
        # 拼接为完整位姿
        pose = torch.cat([translation, rotation], dim=-1)  # (B, 7)
        
        return pose, rotation, translation


def rotation_vector_to_matrix(rvec):
    """
    将旋转向量（轴角表示）转换为旋转矩阵
    
    Args:
        rvec: (B, 3) 旋转向量
        
    Returns:
        R: (B, 3, 3) 旋转矩阵
    """
    batch_size = rvec.shape[0]
    device = rvec.device
    
    # 计算旋转角度
    angle = torch.norm(rvec, dim=-1, keepdim=True)  # (B, 1)
    
    # 处理零旋转的情况
    angle_safe = torch.where(angle < 1e-8, torch.ones_like(angle), angle)
    
    # 归一化旋转轴
    axis = rvec / angle_safe  # (B, 3)
    
    # Rodrigues公式
    cos_angle = torch.cos(angle)  # (B, 1)
    sin_angle = torch.sin(angle)  # (B, 1)
    
    # 构造反对称矩阵
    zeros = torch.zeros(batch_size, device=device)
    K = torch.stack([
        zeros, -axis[:, 2], axis[:, 1],
        axis[:, 2], zeros, -axis[:, 0],
        -axis[:, 1], axis[:, 0], zeros
    ], dim=-1).reshape(batch_size, 3, 3)
    
    # R = I + sin(θ)K + (1-cos(θ))K^2
    I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, 3, 3)
    R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1) * torch.bmm(K, K)
    
    # 处理零旋转
    R = torch.where(angle.unsqueeze(-1) < 1e-8, I, R)
    
    return R


def quaternion_to_matrix(quaternion):
    """
    将四元数转换为旋转矩阵
    
    Args:
        quaternion: (B, 4) 四元数 [qw, qx, qy, qz]
        
    Returns:
        R: (B, 3, 3) 旋转矩阵
    """
    qw, qx, qy, qz = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
    
    R = torch.stack([
        1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy),
        2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx),
        2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)
    ], dim=-1).reshape(-1, 3, 3)
    
    return R
