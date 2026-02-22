"""
跨模态融合模块
从ICL-I2PReg的fusion_module.py改编，用于图像和点云特征的融合

参考: ICL-I2PReg/kitti/stage_2/fusion_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerLayer


class CrossModalFusionModule(nn.Module):
    """
    跨模态融合模块
    
    使用Transformer进行2D图像特征和3D点云特征的融合
    输入: 
        - img_feats: (B, N_img, 256) 2D图像特征
        - pcd_feats: (B, N_pcd, 256) 3D点云特征
        - query_feats: (B, N_query, 256) 查询特征（可学习）
    
    输出:
        - fused_feats: (B, N_query, 256) 融合后的查询特征
        - img_tokens: (B, N_img, 256) 更新后的图像特征
        - pcd_tokens: (B, N_pcd, 256) 更新后的点云特征
    
    融合策略:
        1. Query自注意力
        2. Query与Image交叉注意力
        3. Query自注意力
        4. Query与Point Cloud交叉注意力
        重复多层
    """
    
    def __init__(self, 
                 feature_dim=256,
                 num_layers=6,
                 num_heads=8,
                 d_feedforward=1024,
                 dropout=0.1,
                 activation='relu'):
        """
        Args:
            feature_dim: 特征维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            d_feedforward: 前馈网络维度
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # Transformer层列表
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                d_model=feature_dim,
                num_heads=num_heads,
                d_feedforward=d_feedforward,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # 图像和点云特征的独立Transformer编码器（可选）
        self.img_encoder = TransformerLayer(
            d_model=feature_dim,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        self.pcd_encoder = TransformerLayer(
            d_model=feature_dim,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            dropout=dropout,
            activation=activation
        )
        
    def forward(self, query_feats, img_feats, pcd_feats, 
                img_padding_mask=None, pcd_padding_mask=None):
        """
        前向传播
        
        Args:
            query_feats: (B, N_query, C) 查询特征（可学习）
            img_feats: (B, N_img, C) 2D图像特征
            pcd_feats: (B, N_pcd, C) 3D点云特征
            img_padding_mask: (B, N_img) 图像特征padding掩码，True表示无效位置
            pcd_padding_mask: (B, N_pcd) 点云特征padding掩码，True表示无效位置
            
        Returns:
            query_list: list of (B, N_query, C) 每层的query特征（用于多尺度监督）
            img_tokens: (B, N_img, C) 更新后的图像特征
            pcd_tokens: (B, N_pcd, C) 更新后的点云特征
        """
        # 首先对图像和点云特征进行自注意力编码
        img_tokens = self.img_encoder(
            img_feats, img_feats, img_feats,
            query_padding_mask=img_padding_mask,
            key_padding_mask=img_padding_mask
        )
        
        pcd_tokens = self.pcd_encoder(
            pcd_feats, pcd_feats, pcd_feats,
            query_padding_mask=pcd_padding_mask,
            key_padding_mask=pcd_padding_mask
        )
        
        # 多层跨模态融合
        query = query_feats
        query_list = []
        
        for i, layer in enumerate(self.transformer_layers):
            # 奇数层: Query与Image交叉注意力
            if i % 2 == 0:
                query = layer(
                    query, img_tokens, img_tokens,
                    key_padding_mask=img_padding_mask
                )
            # 偶数层: Query与Point Cloud交叉注意力
            else:
                query = layer(
                    query, pcd_tokens, pcd_tokens,
                    key_padding_mask=pcd_padding_mask
                )
            
            query_list.append(query)
        
        return query_list, img_tokens, pcd_tokens


class OverlapEstimator(nn.Module):
    """
    重叠区域估计器（可选）
    
    用于估计2D-3D特征之间的重叠程度
    这可以帮助过滤掉低置信度的对应关系
    """
    
    def __init__(self, feature_dim=256, hidden_dim=128):
        """
        Args:
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img_feats, pcd_feats):
        """
        前向传播
        
        Args:
            img_feats: (B, N_img, C) 图像特征
            pcd_feats: (B, N_pcd, C) 点云特征
            
        Returns:
            overlap_scores: (B,) 重叠分数 [0, 1]
        """
        # 全局平均池化
        img_global = img_feats.mean(dim=1)  # (B, C)
        pcd_global = pcd_feats.mean(dim=1)  # (B, C)
        
        # 拼接并预测重叠分数
        concat_feats = torch.cat([img_global, pcd_global], dim=-1)  # (B, 2C)
        overlap_scores = self.mlp(concat_feats).squeeze(-1)  # (B,)
        
        return overlap_scores
