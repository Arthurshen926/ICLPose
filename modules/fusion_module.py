"""
跨模态融合模块 - 基于ICL-I2PReg架构的实现
使用Self-Attention和Cross-Attention交替融合

参考: ICL-I2PReg/kitti/stage_2/fusion_module.py

架构说明:
    - 每个"block"包含2层：
        1. Self-Attn + Cross-Attn(Image)
        2. Self-Attn + Cross-Attn(PointCloud)
    - num_layers参数控制block数量（必须是2的倍数）
    - ICL-I2PReg原版使用1个block（2层），但可配置多个block以增强融合
"""

import torch
import torch.nn as nn
from typing import Optional, List
from .transformer import TransformerLayer


class CrossModalFusionModule(nn.Module):
    """
    跨模态融合模块 - Self-Cross交替模式
    
    架构:
    每个block包含:
    1. Query Self-Attention → Query-Image Cross-Attention
    2. Query Self-Attention → Query-PointCloud Cross-Attention
    
    可配置多个block堆叠，增强跨模态融合能力
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = 'ReLU',
    ):
        """
        Args:
            feature_dim: 特征维度
            num_layers: 总层数（必须是2的倍数，每2层构成1个block）
                        - 2层 = 1个block（ICL-I2PReg原版）
                        - 4层 = 2个blocks
                        - 6层 = 3个blocks
            num_heads: 注意力头数
            dropout: Dropout概率
            activation: 激活函数
        """
        super().__init__()
        
        # 确保层数是2的倍数
        if num_layers % 2 != 0:
            num_layers = (num_layers // 2) * 2
            if num_layers < 2:
                num_layers = 2
            print(f"⚠️  fusion_layers必须是2的倍数，已调整为{num_layers}")
        
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_blocks = num_layers // 2  # 每2层构成1个block
        
        # Token投影层
        self.img_in_proj = nn.Linear(feature_dim, feature_dim)
        self.pcd_in_proj = nn.Linear(feature_dim, feature_dim)
        
        # Self-Attention层: queries与自己交互
        self.self_attn_layers = nn.ModuleList([
            TransformerLayer(
                d_model=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        
        # Cross-Attention层: queries与img/pcd交互
        self.cross_attn_layers = nn.ModuleList([
            TransformerLayer(
                d_model=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        
        # Output投影层
        self.query_out_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(
        self,
        query_feats: torch.Tensor,  # (B, N_query, C)
        img_feats: torch.Tensor,    # (B, N_img, C)
        pcd_feats: torch.Tensor,    # (B, N_pcd, C)
        query_pos_embeds: Optional[torch.Tensor] = None,  # (B, N_query, C)
        img_pos_embeds: Optional[torch.Tensor] = None,    # (B, N_img, C)
        pcd_pos_embeds: Optional[torch.Tensor] = None,    # (B, N_pcd, C)
    ):
        """
        前向传播
        
        Args:
            query_feats: 查询特征 [B, N_query, C]
            img_feats: 2D图像特征 [B, N_img, C]
            pcd_feats: 3D点云特征 [B, N_pcd, C]
            query_pos_embeds: Query位置编码（可选）
            img_pos_embeds: Image位置编码（可选）
            pcd_pos_embeds: PointCloud位置编码（可选）
        
        Returns:
            query_list: [query_img, query_pcd, query_output] - 最后一个block的分离query特征
            img_tokens: 处理后的图像tokens
            pcd_tokens: 处理后的点云tokens
        """
        # 投影tokens
        img_tokens = self.img_in_proj(img_feats)
        pcd_tokens = self.pcd_in_proj(pcd_feats)
        
        query = query_feats
        query_img_final = None
        query_pcd_final = None
        
        # 遍历所有blocks
        for block_idx in range(self.num_blocks):
            layer_idx = block_idx * 2
            
            # Layer 1: Self-Attention + Cross-Attention with Image
            query_s1 = self.self_attn_layers[layer_idx](
                q=query,
                k=query,
                v=query,
                q_embeds=query_pos_embeds,
                k_embeds=query_pos_embeds,
            )
            query_c1 = self.cross_attn_layers[layer_idx](
                q=query_s1,
                k=img_tokens,
                v=img_tokens,
                q_embeds=query_pos_embeds,
                k_embeds=img_pos_embeds,
            )
            
            # 保存最后一个block的img query
            if block_idx == self.num_blocks - 1:
                query_img_final = query_c1
            
            # Layer 2: Self-Attention + Cross-Attention with PointCloud
            query_s2 = self.self_attn_layers[layer_idx + 1](
                q=query_c1,
                k=query_c1,
                v=query_c1,
                q_embeds=query_pos_embeds,
                k_embeds=query_pos_embeds,
            )
            query_c2 = self.cross_attn_layers[layer_idx + 1](
                q=query_s2,
                k=pcd_tokens,
                v=pcd_tokens,
                q_embeds=query_pos_embeds,
                k_embeds=pcd_pos_embeds,
            )
            
            # 保存最后一个block的pcd query
            if block_idx == self.num_blocks - 1:
                query_pcd_final = query_c2
            
            # 更新query用于下一个block
            query = query_c2
        
        # Output投影
        query_output = self.query_out_proj(query)
        
        # 返回最后一个block的query特征
        query_list = [query_img_final, query_pcd_final, query_output]
        
        return query_list, img_tokens, pcd_tokens


class LearnableQueryEmbedding(nn.Module):
    """可学习的Query Embedding"""
    
    def __init__(self, num_queries: int, feature_dim: int):
        super().__init__()
        self.num_queries = num_queries
        self.feature_dim = feature_dim
        
        # 可学习的query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        
        # 初始化
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        
    def forward(self, batch_size: int):
        """
        Args:
            batch_size: Batch大小
        Returns:
            query_embeds: [B, N_query, C]
        """
        return self.query_embed.expand(batch_size, -1, -1)
