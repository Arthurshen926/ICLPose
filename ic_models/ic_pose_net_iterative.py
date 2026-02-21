"""
ICPoseNetIterative: 迭代精化版位姿估计网络

关键改进点（参考ICL-I2PReg的多阶段设计）：
1. 多阶段迭代精化：从粗到细逐步改进位姿估计
2. 位姿感知特征：将当前位姿估计编码并传递给下一阶段
3. 残差位姿学习：每阶段学习位姿增量而非绝对值
4. 更大模型容量：更多融合层和更大隐藏维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.fusion_module import CrossModalFusionModule
from modules.pose_regressor import PoseRegressor, rotation_6d_to_matrix, matrix_to_rotation_6d
from ic_models.positional_encoding import PositionalEncoding2D, PositionalEncoding3D


class PoseEncoder(nn.Module):
    """
    位姿编码器：将4x4位姿矩阵编码为特征向量
    
    用于将当前位姿估计传递给下一阶段
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        # 位姿矩阵有12个有效参数（3x3旋转 + 3平移）
        self.encoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        
    def forward(self, pose_matrix):
        """
        Args:
            pose_matrix: (B, 4, 4) 位姿矩阵
        Returns:
            pose_feat: (B, feature_dim) 位姿特征
        """
        # 提取3x3旋转和3平移
        R = pose_matrix[:, :3, :3]  # (B, 3, 3)
        t = pose_matrix[:, :3, 3]   # (B, 3)
        
        # 展平为12维向量
        pose_vec = torch.cat([R.reshape(-1, 9), t], dim=-1)  # (B, 12)
        
        return self.encoder(pose_vec)  # (B, feature_dim)


class RefinementStage(nn.Module):
    """
    单个精化阶段
    
    接收：
    - Query特征
    - 图像特征 + 位置编码
    - 点云特征 + 位置编码
    - 当前位姿估计（用于位姿感知）
    
    输出：
    - 位姿增量（相对于当前位姿）
    - 更新后的Query特征
    - Keypoint坐标
    """
    def __init__(self, feature_dim=256, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # 位姿编码器
        self.pose_encoder = PoseEncoder(feature_dim)
        
        # Query位姿融合：将位姿特征融入Query
        self.query_pose_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        
        # 跨模态融合模块（每阶段2层：1个img block + 1个pcd block）
        self.fusion_module = CrossModalFusionModule(
            feature_dim=feature_dim,
            num_layers=2,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # 位姿回归器（预测位姿增量）
        self.pose_regressor = PoseRegressor(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=9  # 3平移 + 6D旋转
        )
        
    def forward(self, query_feats, img_feats, pcd_feats, 
                img_pos_embeds, pcd_pos_embeds, 
                img_pixels, pcd_points,
                current_pose):
        """
        Args:
            query_feats: (B, N_query, C) Query特征
            img_feats: (B, N_img, C) 图像特征
            pcd_feats: (B, N_pcd, C) 点云特征
            img_pos_embeds: (B, N_img, C) 图像位置编码
            pcd_pos_embeds: (B, N_pcd, C) 点云位置编码
            img_pixels: (B, N_img, 2) 像素坐标
            pcd_points: (B, N_pcd, 3) 点云坐标
            current_pose: (B, 4, 4) 当前位姿估计
            
        Returns:
            delta_pose: (B, 4, 4) 位姿增量矩阵
            updated_query: (B, N_query, C) 更新后的Query
            img_keypoints: (B, N_query, 2) 2D关键点
            pcd_keypoints: (B, N_query, 3) 3D关键点
            img_heatmap: (B, N_query, N_img) 2D注意力图
        """
        batch_size = query_feats.shape[0]
        device = query_feats.device
        
        # 1. 编码当前位姿
        pose_feat = self.pose_encoder(current_pose)  # (B, C)
        pose_feat = pose_feat.unsqueeze(1).expand(-1, query_feats.shape[1], -1)  # (B, N_query, C)
        
        # 2. 将位姿特征融入Query
        query_with_pose = torch.cat([query_feats, pose_feat], dim=-1)  # (B, N_query, 2C)
        query_conditioned = self.query_pose_fusion(query_with_pose)  # (B, N_query, C)
        
        # 3. 跨模态融合
        query_list, img_tokens, pcd_tokens = self.fusion_module(
            query_conditioned, img_feats, pcd_feats,
            query_pos_embeds=None,
            img_pos_embeds=img_pos_embeds,
            pcd_pos_embeds=pcd_pos_embeds,
        )
        
        query_img_feats = query_list[0]  # 经过img cross-attention的query
        query_pcd_feats = query_list[1]  # 经过pcd cross-attention的query
        query_output = query_list[2]     # 最终query
        
        # 4. 计算Keypoint Heatmap
        img_keypoint_heatmap = torch.matmul(
            query_img_feats, img_tokens.transpose(1, 2)
        ) / (self.feature_dim ** 0.5)
        img_keypoint_heatmap = F.softmax(img_keypoint_heatmap, dim=-1)
        
        pcd_keypoint_heatmap = torch.matmul(
            query_pcd_feats, pcd_tokens.transpose(1, 2)
        ) / (self.feature_dim ** 0.5)
        pcd_keypoint_heatmap = F.softmax(pcd_keypoint_heatmap, dim=-1)
        
        # 5. 提取Keypoint坐标（soft-argmax）
        img_keypoints = torch.matmul(img_keypoint_heatmap, img_pixels)
        pcd_keypoints = torch.matmul(pcd_keypoint_heatmap, pcd_points)
        
        # 6. 回归位姿增量
        pose_9d, rotation_6d, translation = self.pose_regressor(
            query_output, img_keypoints, pcd_keypoints
        )
        
        # 7. 构建位姿增量矩阵
        R_delta = rotation_6d_to_matrix(rotation_6d)
        delta_pose = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, 4, 4).clone()
        delta_pose[:, :3, :3] = R_delta
        delta_pose[:, :3, 3] = translation
        
        return delta_pose, query_output, img_keypoints, pcd_keypoints, img_keypoint_heatmap


class ICPoseNetIterative(nn.Module):
    """
    迭代精化位姿估计网络
    
    架构流程:
    1. 使用初始位姿（带噪声）作为起点
    2. 通过多个RefinementStage逐步精化位姿
    3. 每阶段学习位姿增量并累积到当前估计
    4. 最终输出精化后的位姿
    
    关键设计：
    - 位姿感知：每阶段都知道当前位姿估计，可以针对性地修正
    - 残差学习：学习位姿增量而非绝对值，更容易收敛
    - 共享Query：Query特征在阶段间传递和更新
    """
    
    def __init__(self,
                 feature_dim=256,
                 num_queries=64,
                 num_stages=3,      # 精化阶段数
                 hidden_dim=512,
                 num_heads=8,
                 dropout=0.1):
        """
        Args:
            feature_dim: 特征维度
            num_queries: Query数量
            num_stages: 精化阶段数（推荐2-4）
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_stages = num_stages
        
        # 可学习的Query Embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        nn.init.xavier_uniform_(self.query_embed)
        
        # 位置编码模块
        self.pos_enc_2d = PositionalEncoding2D(embed_dim=feature_dim, temperature=10000)
        self.pos_enc_3d = PositionalEncoding3D(embed_dim=258, temperature=10000, normalize=True, scale_factor=0.1)
        self.pos_enc_3d_proj = nn.Linear(258, feature_dim)
        
        # 多阶段精化模块（每阶段独立参数）
        self.stages = nn.ModuleList([
            RefinementStage(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_stages)
        ])
        
        print(f"📐 ICPoseNetIterative initialized:")
        print(f"   - Feature dim: {feature_dim}")
        print(f"   - Num queries: {num_queries}")
        print(f"   - Num stages: {num_stages}")
        print(f"   - Hidden dim: {hidden_dim}")
        
    def forward(self, img_feats, pcd_feats, img_pixels, pcd_points,
                img_pos_embeds=None, pcd_pos_embeds=None,
                initial_pose=None):
        """
        前向传播
        
        Args:
            img_feats: (B, N_img, C) 图像特征
            pcd_feats: (B, N_pcd, C) 点云特征
            img_pixels: (B, N_img, 2) 像素坐标
            pcd_points: (B, N_pcd, 3) 点云坐标
            img_pos_embeds: (B, N_img, C) 图像位置编码（可选）
            pcd_pos_embeds: (B, N_pcd, C) 点云位置编码（可选）
            initial_pose: (B, 4, 4) 初始位姿（必须提供！）
            
        Returns:
            pose_matrix: (B, 4, 4) 最终位姿估计（绝对坐标系）
            pose_9d: (B, 9) 9D位姿 [tx, ty, tz, r1-r6]
            rotation_6d: (B, 6) 6D旋转
            translation: (B, 3) 平移
            img_keypoint_heatmap: (B, N_query, N_img) 最后阶段的2D heatmap
            img_keypoints: (B, N_query, 2) 最后阶段的2D关键点
            pcd_keypoints: (B, N_query, 3) 最后阶段的3D关键点
            stage_poses: List[(B, 4, 4)] 每阶段的位姿估计
        """
        batch_size = img_feats.shape[0]
        device = img_feats.device
        
        # 必须提供初始位姿
        if initial_pose is None:
            raise ValueError("ICPoseNetIterative requires initial_pose!")
        
        # 生成位置编码（如果未提供）
        if img_pos_embeds is None:
            # 假设输入是展平的图像特征，需要根据像素坐标生成
            img_pos_embeds = self._generate_2d_pos_embeds(img_pixels, device)
        
        if pcd_pos_embeds is None:
            pcd_pos_embeds = self._generate_3d_pos_embeds(pcd_points)
        
        # 初始化
        query_feats = self.query_embed.expand(batch_size, -1, -1)
        current_pose = initial_pose.clone()
        stage_poses = [current_pose.clone()]
        
        # 多阶段迭代精化
        for stage_idx, stage in enumerate(self.stages):
            delta_pose, query_feats, img_keypoints, pcd_keypoints, img_heatmap = stage(
                query_feats, img_feats, pcd_feats,
                img_pos_embeds, pcd_pos_embeds,
                img_pixels, pcd_points,
                current_pose
            )
            
            # 累积位姿增量: new_pose = current_pose @ delta_pose
            current_pose = torch.bmm(current_pose, delta_pose)
            stage_poses.append(current_pose.clone())
        
        # 提取最终的旋转和平移
        R_final = current_pose[:, :3, :3]
        t_final = current_pose[:, :3, 3]
        
        # 转换为6D旋转表示
        rotation_6d = matrix_to_rotation_6d(R_final)
        
        # 计算相对于初始位姿的变换（用于loss计算）
        relative_pose = torch.bmm(torch.inverse(initial_pose), current_pose)
        t_relative = relative_pose[:, :3, 3]
        R_relative = relative_pose[:, :3, :3]
        rotation_6d_relative = matrix_to_rotation_6d(R_relative)
        
        # 9D位姿（相对）
        pose_9d = torch.cat([t_relative, rotation_6d_relative], dim=-1)
        
        return (current_pose, pose_9d, rotation_6d_relative, t_relative,
                img_heatmap, img_keypoints, pcd_keypoints, stage_poses)
    
    def _generate_2d_pos_embeds(self, img_pixels, device):
        """从像素坐标生成2D位置编码"""
        # 简化版本：直接使用线性编码
        batch_size, n_pixels, _ = img_pixels.shape
        
        # 归一化到[-1, 1]
        h, w = 480, 640  # 假设固定图像大小
        x_norm = (img_pixels[..., 0] / w) * 2 - 1
        y_norm = (img_pixels[..., 1] / h) * 2 - 1
        coords = torch.stack([x_norm, y_norm], dim=-1)
        
        # 使用正弦位置编码
        pos_embeds = self.pos_enc_2d._generate_from_coords(coords, device)
        return pos_embeds
    
    def _generate_3d_pos_embeds(self, pcd_points):
        """从点云坐标生成3D位置编码"""
        pos_embeds_raw = self.pos_enc_3d(pcd_points)
        return self.pos_enc_3d_proj(pos_embeds_raw)


class ICPoseNetIterativeLite(nn.Module):
    """
    轻量级迭代精化版本
    
    与ICPoseNetIterative相比：
    - 所有阶段共享参数（参数效率更高）
    - 更少的计算开销
    """
    
    def __init__(self,
                 feature_dim=256,
                 num_queries=64,
                 num_stages=3,
                 hidden_dim=512,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_stages = num_stages
        
        # Query Embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        nn.init.xavier_uniform_(self.query_embed)
        
        # 位置编码
        self.pos_enc_2d = PositionalEncoding2D(embed_dim=feature_dim, temperature=10000)
        self.pos_enc_3d = PositionalEncoding3D(embed_dim=258, temperature=10000, normalize=True, scale_factor=0.1)
        self.pos_enc_3d_proj = nn.Linear(258, feature_dim)
        
        # 共享的精化阶段（所有迭代共享参数）
        self.shared_stage = RefinementStage(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        print(f"📐 ICPoseNetIterativeLite initialized:")
        print(f"   - Feature dim: {feature_dim}")
        print(f"   - Num queries: {num_queries}")
        print(f"   - Num stages: {num_stages} (shared params)")
        
    def forward(self, img_feats, pcd_feats, img_pixels, pcd_points,
                img_pos_embeds=None, pcd_pos_embeds=None,
                initial_pose=None):
        """与ICPoseNetIterative相同的接口"""
        batch_size = img_feats.shape[0]
        device = img_feats.device
        
        if initial_pose is None:
            raise ValueError("ICPoseNetIterativeLite requires initial_pose!")
        
        # 生成位置编码
        if pcd_pos_embeds is None:
            pcd_pos_embeds_raw = self.pos_enc_3d(pcd_points)
            pcd_pos_embeds = self.pos_enc_3d_proj(pcd_pos_embeds_raw)
        
        # 初始化
        query_feats = self.query_embed.expand(batch_size, -1, -1)
        current_pose = initial_pose.clone()
        stage_poses = [current_pose.clone()]
        
        # 多阶段迭代（共享参数）
        for stage_idx in range(self.num_stages):
            delta_pose, query_feats, img_keypoints, pcd_keypoints, img_heatmap = self.shared_stage(
                query_feats, img_feats, pcd_feats,
                img_pos_embeds, pcd_pos_embeds,
                img_pixels, pcd_points,
                current_pose
            )
            
            current_pose = torch.bmm(current_pose, delta_pose)
            stage_poses.append(current_pose.clone())
        
        # 提取输出
        R_final = current_pose[:, :3, :3]
        t_final = current_pose[:, :3, 3]
        rotation_6d = matrix_to_rotation_6d(R_final)
        
        relative_pose = torch.bmm(torch.inverse(initial_pose), current_pose)
        t_relative = relative_pose[:, :3, 3]
        R_relative = relative_pose[:, :3, :3]
        rotation_6d_relative = matrix_to_rotation_6d(R_relative)
        pose_9d = torch.cat([t_relative, rotation_6d_relative], dim=-1)
        
        return (current_pose, pose_9d, rotation_6d_relative, t_relative,
                img_heatmap, img_keypoints, pcd_keypoints, stage_poses)
