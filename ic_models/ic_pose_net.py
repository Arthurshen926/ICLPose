"""
ICPoseNet: 隐式对应关系位姿估计网络
整合SplatLoc特征提取 + 跨模态融合 + 位姿回归
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.fusion_module import CrossModalFusionModule
from modules.pose_regressor import PoseRegressor, rotation_vector_to_matrix
from ic_models.positional_encoding import PositionalEncoding2D, PositionalEncoding3D


class ICPoseNet(nn.Module):
    """
    隐式对应关系位姿估计网络
    
    架构流程:
    1. 从SplatLoc获取2D图像特征和3D场景特征
    2. 使用可学习的query embeddings
    3. 通过CrossModalFusionModule融合2D-3D特征
    4. 使用PoseRegressor回归6-DOF位姿
    
    输入:
        - image: (B, 3, H, W) RGB图像
        - gaussians: GaussianModel (3DGS场景)
        - feat_decoder: FeatureDecoder (特征解码器)
        
    输出:
        - pose: (B, 4, 4) 位姿矩阵
        - rotation: (B, 3) 旋转向量
        - translation: (B, 3) 平移向量
    """
    
    def __init__(self,
                 feature_dim=256,
                 num_queries=128,
                 fusion_layers=6,
                 num_heads=8,
                 dropout=0.1,
                 attention_temperature=None):  # None means sqrt(feature_dim)
        """
        Args:
            feature_dim: 特征维度（应与SplatLoc的decoder输出维度一致）
            num_queries: 可学习query的数量
            fusion_layers: 跨模态融合层数
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        # Attention temperature for heatmap computation
        # Lower temperature = sharper distribution
        self.attention_temperature = attention_temperature if attention_temperature is not None else (feature_dim ** 0.5)
        
        # 可学习的query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        nn.init.xavier_uniform_(self.query_embed)
        
        # 位置编码模块
        self.pos_enc_2d = PositionalEncoding2D(embed_dim=feature_dim, temperature=10000)
        self.pos_enc_3d = PositionalEncoding3D(embed_dim=258, temperature=10000, normalize=True, scale_factor=0.1)
        # 3D位置编码投影层（258 -> 256）
        self.pos_enc_3d_proj = nn.Linear(258, feature_dim)
        
        # 跨模态融合模块
        self.fusion_module = CrossModalFusionModule(
            feature_dim=feature_dim,
            num_layers=fusion_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation='ReLU'
        )
        
        # 位姿回归器
        self.pose_regressor = PoseRegressor(
            feature_dim=feature_dim,
            hidden_dim=512,
            output_dim=6
        )
        
    def forward(self, img_feats, pcd_feats, 
                img_pixels, pcd_points,
                img_pos_embeds=None, pcd_pos_embeds=None,
                img_padding_mask=None, pcd_padding_mask=None):
        """
        前向传播
        
        Args:
            img_feats: (B, N_img, C) 2D图像特征
            pcd_feats: (B, N_pcd, C) 3D点云特征
            img_pixels: (B, N_img, 2) 2D像素坐标
            pcd_points: (B, N_pcd, 3) 3D点坐标
            img_pos_embeds: (B, N_img, C) 2D位置编码（可选）
            pcd_pos_embeds: (B, N_pcd, C) 3D位置编码（可选）
            img_padding_mask: (B, N_img) 图像特征padding掩码
            pcd_padding_mask: (B, N_pcd) 点云特征padding掩码
            
        Returns:
            pose_matrix: (B, 4, 4) 位姿矩阵
            pose_9d: (B, 9) 9-DOF位姿 [tx, ty, tz, r1...r6] (3平移+6D旋转)
            rotation_6d: (B, 6) 6D旋转表示
            translation: (B, 3) 平移向量
            img_keypoint_heatmap: (B, N_query, N_img) 2D attention heatmap
            img_keypoints: (B, N_query, 2) 检测到的2D关键点坐标
            pcd_keypoints: (B, N_query, 3) 检测到的3D关键点坐标
        """
        batch_size = img_feats.shape[0]
        device = img_feats.device
        
        # 扩展query embeddings到batch
        query_feats = self.query_embed.expand(batch_size, -1, -1)  # (B, N_query, C)
        
        # 跨模态融合（通过embeds参数传递位置编码）
        query_list, img_tokens, pcd_tokens = self.fusion_module(
            query_feats, img_feats, pcd_feats,
            query_pos_embeds=None,  # Query不需要位置编码
            img_pos_embeds=img_pos_embeds,
            pcd_pos_embeds=pcd_pos_embeds,
        )
        
        # 🆕 完全对齐ICL-I2PReg: 使用分离的query特征
        query_img_feats = query_list[0]  # 经过img cross-attention的query
        query_pcd_feats = query_list[1]  # 经过pcd cross-attention的query
        query_output = query_list[2]     # 最终query（用于位姿回归）
        
        # 🆕 计算Keypoint Heatmap（完全对齐ICL-I2PReg）
        # 使用img-specific query和处理后的img_tokens
        img_keypoint_heatmap = torch.matmul(
            query_img_feats,  # (B, N_query, C) - img-specific query
            img_tokens.transpose(1, 2)  # (B, C, N_img) - 处理后的tokens
        ) / self.attention_temperature  # Temperature scaling (lower = sharper)
        
        # Softmax归一化 → 概率分布
        img_keypoint_heatmap = F.softmax(img_keypoint_heatmap, dim=-1)  # (B, N_query, N_img)
        
        # 🆕 计算3D Keypoint Heatmap（统一使用相同temperature - 对齐ICL-I2PReg）
        pcd_keypoint_heatmap = torch.matmul(
            query_pcd_feats,  # (B, N_query, C) - pcd-specific query
            pcd_tokens.transpose(1, 2)  # (B, C, N_pcd)
        ) / self.attention_temperature  # 与img heatmap使用相同的temperature
        pcd_keypoint_heatmap = F.softmax(pcd_keypoint_heatmap, dim=-1)  # (B, N_query, N_pcd)
        
        # 🆕 提取Keypoint坐标（soft-argmax）
        img_keypoints = torch.matmul(img_keypoint_heatmap, img_pixels)  # (B, N_query, 2)
        pcd_keypoints = torch.matmul(pcd_keypoint_heatmap, pcd_points)  # (B, N_query, 3)
        
        # 回归位姿（使用query_output + keypoints）
        pose_9d, rotation_6d, translation = self.pose_regressor(
            query_output, img_keypoints, pcd_keypoints
        )
        
        # 转换为位姿矩阵（使用6D旋转表示）
        from modules.pose_regressor import rotation_6d_to_matrix
        R = rotation_6d_to_matrix(rotation_6d)
        pose_matrix = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, 4, 4).clone()
        pose_matrix[:, :3, :3] = R
        pose_matrix[:, :3, 3] = translation
        
        return pose_matrix, pose_9d, rotation_6d, translation, \
               img_keypoint_heatmap, img_keypoints, pcd_keypoints
    
    def pose_6d_to_matrix(self, rotation, translation):
        """
        将6-DOF位姿转换为4x4位姿矩阵
        
        Args:
            rotation: (B, 3) 旋转向量
            translation: (B, 3) 平移向量
            
        Returns:
            pose_matrix: (B, 4, 4) 位姿矩阵
        """
        batch_size = rotation.shape[0]
        device = rotation.device
        
        # 旋转向量转旋转矩阵
        R = rotation_vector_to_matrix(rotation)  # (B, 3, 3)
        
        # 构造4x4位姿矩阵
        pose_matrix = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, 4, 4).clone()
        pose_matrix[:, :3, :3] = R
        pose_matrix[:, :3, 3] = translation
        
        return pose_matrix


class ICPoseNetSimple(nn.Module):
    """
    简化版ICPoseNet
    
    直接从预提取的2D-3D特征对进行位姿估计，
    不包含SplatLoc的特征提取步骤
    """
    
    def __init__(self,
                 feature_dim=256,
                 hidden_dim=512,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1):
        """
        Args:
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # 2D特征编码器
        self.img_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 3D特征编码器
        self.pcd_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 跨模态融合
        self.fusion_module = CrossModalFusionModule(
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            d_feedforward=hidden_dim,
            dropout=dropout
        )
        
        # 位姿回归
        self.pose_regressor = PoseRegressor(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=6
        )
        
        # Query embeddings
        num_queries = 64
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        nn.init.xavier_uniform_(self.query_embed)
        
    def forward(self, img_feats, pcd_feats):
        """
        前向传播
        
        Args:
            img_feats: (B, N_img, C) 2D图像特征
            pcd_feats: (B, N_pcd, C) 3D点云特征
            
        Returns:
            pose_matrix: (B, 4, 4) 位姿矩阵
            pose_9d: (B, 9) 9D位姿 [tx, ty, tz, r1-r6]
            rotation_6d: (B, 6) 6D旋转表示
            translation: (B, 3) 平移向量（相对）
        """
        from modules.pose_regressor import rotation_6d_to_matrix
        
        batch_size = img_feats.shape[0]
        
        # 编码特征
        img_feats = self.img_encoder(img_feats)
        pcd_feats = self.pcd_encoder(pcd_feats)
        
        # Query embeddings
        query_feats = self.query_embed.expand(batch_size, -1, -1)
        
        # 跨模态融合
        query_list, _, _ = self.fusion_module(query_feats, img_feats, pcd_feats)
        fused_feats = query_list[-1]
        
        # 位姿回归（输出9D: 3平移 + 6旋转）
        pose_9d, rotation_6d, translation = self.pose_regressor(fused_feats)
        
        # 6D旋转 -> 旋转矩阵
        R = rotation_6d_to_matrix(rotation_6d)  # (B, 3, 3)
        
        # 构建位姿矩阵
        pose_matrix = torch.eye(4, device=rotation_6d.device).unsqueeze(0).expand(batch_size, 4, 4).clone()
        pose_matrix[:, :3, :3] = R
        pose_matrix[:, :3, 3] = translation
        
        return pose_matrix, pose_9d, rotation_6d, translation
