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
                 dropout=0.1):
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
            d_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='relu'
        )
        
        # 位姿回归器
        self.pose_regressor = PoseRegressor(
            feature_dim=feature_dim,
            hidden_dim=512,
            output_dim=6
        )
        
    def forward(self, img_feats, pcd_feats, 
                img_padding_mask=None, pcd_padding_mask=None):
        """
        前向传播
        
        Args:
            img_feats: (B, N_img, C) 2D图像特征
            pcd_feats: (B, N_pcd, C) 3D点云特征
            img_padding_mask: (B, N_img) 图像特征padding掩码
            pcd_padding_mask: (B, N_pcd) 点云特征padding掩码
            
        Returns:
            pose_matrix: (B, 4, 4) 位姿矩阵
            pose_6d: (B, 6) 6-DOF位姿 [tx, ty, tz, rx, ry, rz]
            rotation: (B, 3) 旋转向量
            translation: (B, 3) 平移向量
        """
        batch_size = img_feats.shape[0]
        device = img_feats.device
        
        # 扩展query embeddings到batch
        query_feats = self.query_embed.expand(batch_size, -1, -1)  # (B, N_query, C)
        
        # 跨模态融合
        query_list, img_tokens, pcd_tokens = self.fusion_module(
            query_feats, img_feats, pcd_feats,
            img_padding_mask=img_padding_mask,
            pcd_padding_mask=pcd_padding_mask
        )
        
        # 使用最后一层的query特征进行位姿回归
        fused_feats = query_list[-1]  # (B, N_query, C)
        
        # 回归位姿
        pose_6d, rotation, translation = self.pose_regressor(fused_feats)
        
        # 转换为位姿矩阵
        pose_matrix = self.pose_6d_to_matrix(rotation, translation)
        
        return pose_matrix, pose_6d, rotation, translation
    
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
    
    def extract_features_from_splatloc(self, images, gaussians, feat_decoder, 
                                       num_2d_samples=512, num_3d_samples=2048):
        """
        从SplatLoc提取2D和3D特征
        
        Args:
            images: (B, 3, H, W) 输入图像
            gaussians: GaussianModel 3DGS场景
            feat_decoder: FeatureDecoder 特征解码器
            num_2d_samples: 采样的2D特征点数量
            num_3d_samples: 采样的3D特征点数量
            
        Returns:
            img_feats: (B, N_2d, C) 2D特征
            pcd_feats: (B, N_3d, C) 3D特征
            coords_2d: (B, N_2d, 2) 2D坐标
            coords_3d: (B, N_3d, 3) 3D坐标
        """
        batch_size, _, height, width = images.shape
        device = images.device
        
        # 1. 提取2D图像特征
        # 这里需要根据SplatLoc的实际实现来提取特征
        # 假设从渲染的特征图中采样
        with torch.no_grad():
            # TODO: 从SplatLoc渲染特征图
            # feature_map = render_features(gaussians, feat_decoder, camera_params)
            # 这里用占位符
            feature_map = torch.randn(batch_size, self.feature_dim, height, width, device=device)
        
        # 在图像上均匀采样2D点
        coords_2d = self.sample_2d_points(batch_size, height, width, num_2d_samples, device)
        
        # 从特征图中提取对应的特征
        img_feats = self.sample_features_from_map(feature_map, coords_2d)
        
        # 2. 提取3D场景特征
        # 从3DGS场景中采样3D点并查询特征
        with torch.no_grad():
            coords_3d = self.sample_3d_points(gaussians, num_3d_samples, device)
            # TODO: 使用feat_decoder查询3D点的特征
            # pcd_feats = feat_decoder.query_features(coords_3d)
            # 这里用占位符
            pcd_feats = torch.randn(batch_size, num_3d_samples, self.feature_dim, device=device)
        
        return img_feats, pcd_feats, coords_2d, coords_3d
    
    def sample_2d_points(self, batch_size, height, width, num_samples, device):
        """
        在图像上均匀采样2D点
        
        Args:
            batch_size: 批大小
            height: 图像高度
            width: 图像宽度
            num_samples: 采样点数
            device: 设备
            
        Returns:
            coords_2d: (B, N, 2) 归一化的2D坐标 [x, y] in [0, 1]
        """
        # 均匀网格采样
        grid_size = int(np.sqrt(num_samples))
        y = torch.linspace(0, height - 1, grid_size, device=device)
        x = torch.linspace(0, width - 1, grid_size, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (grid_size^2, 2)
        
        # 如果采样点数不是完全平方数，随机选择
        if coords.shape[0] > num_samples:
            indices = torch.randperm(coords.shape[0], device=device)[:num_samples]
            coords = coords[indices]
        elif coords.shape[0] < num_samples:
            # 补充随机点
            extra = num_samples - coords.shape[0]
            random_coords = torch.rand(extra, 2, device=device)
            random_coords[:, 0] *= width - 1
            random_coords[:, 1] *= height - 1
            coords = torch.cat([coords, random_coords], dim=0)
        
        # 归一化到[0, 1]
        coords[:, 0] /= (width - 1)
        coords[:, 1] /= (height - 1)
        
        # 扩展到batch
        coords_2d = coords.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 2)
        
        return coords_2d
    
    def sample_3d_points(self, gaussians, num_samples, device):
        """
        从3DGS场景中采样3D点
        
        Args:
            gaussians: GaussianModel
            num_samples: 采样点数
            device: 设备
            
        Returns:
            coords_3d: (1, N, 3) 3D坐标
        """
        # 从Gaussian中心采样
        xyz = gaussians.get_xyz  # (M, 3)
        
        if xyz.shape[0] > num_samples:
            # 随机采样
            indices = torch.randperm(xyz.shape[0], device=device)[:num_samples]
            sampled_xyz = xyz[indices]
        else:
            # 重复采样
            sampled_xyz = xyz[torch.randint(0, xyz.shape[0], (num_samples,), device=device)]
        
        coords_3d = sampled_xyz.unsqueeze(0)  # (1, N, 3)
        
        return coords_3d
    
    def sample_features_from_map(self, feature_map, coords_2d):
        """
        从特征图中采样特征
        
        Args:
            feature_map: (B, C, H, W) 特征图
            coords_2d: (B, N, 2) 归一化坐标 [x, y] in [0, 1]
            
        Returns:
            features: (B, N, C) 采样的特征
        """
        # 转换为grid_sample所需的格式: [-1, 1]
        grid = coords_2d * 2.0 - 1.0  # (B, N, 2)
        grid = grid.unsqueeze(2)  # (B, N, 1, 2)
        
        # 使用grid_sample进行双线性插值
        features = F.grid_sample(
            feature_map, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )  # (B, C, N, 1)
        
        features = features.squeeze(-1).transpose(1, 2)  # (B, N, C)
        
        return features


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
