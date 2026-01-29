"""
跨模态融合模块 - 完全对齐ICL-I2PReg的实现
使用Self-Attention和Cross-Attention交替融合

参考: ICL-I2PReg/kitti/stage_2/fusion_module.py
"""

import torch
import torch.nn as nn
from typing import Optional
from .transformer import TransformerLayer


class CrossModalFusionModule(nn.Module):
    """
    跨模态融合模块 - Self-Cross交替模式
    
    架构对齐ICL-I2PReg:
    每一层包含:
    1. Query Self-Attention (queries与自己交互)
    2. Query-Image Cross-Attention (queries与image交互)
    然后下一层:
    3. Query Self-Attention
    4. Query-PointCloud Cross-Attention (queries与pcd交互)
    
    这样保证queries可以在每次cross-attention前后聚合信息
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = 'ReLU',
    ):
        """
        Args:
            feature_dim: 特征维度
            num_layers: Transformer层数（🆕 对齐ICL-I2PReg：固定使用2层）
            num_heads: 注意力头数
            dropout: Dropout概率
            activation: 激活函数
        """
        super().__init__()
        
        # 🆕 对齐ICL-I2PReg: 固定使用2层（1个img block + 1个pcd block）
        # 即使配置要求更多层，我们也只使用2层以匹配ICL-I2PReg架构
        if num_layers != 2:
            print(f"⚠️  警告: fusion_layers配置为{num_layers}，但ICL-I2PReg架构固定使用2层")
            print(f"    将自动使用2层（1个img block + 1个pcd block）")
        
        self.feature_dim = feature_dim
        self.num_layers = 2  # 固定为2层
        self.num_blocks = 1  # 1个完整block（img + pcd）
        
        # 🆕 Token投影层（对齐ICL-I2PReg）
        self.img_in_proj = nn.Linear(feature_dim, feature_dim)
        self.pcd_in_proj = nn.Linear(feature_dim, feature_dim)
        
        # Self-Attention层: queries与自己交互（只需要2层）
        self.self_attn_layers = nn.ModuleList([
            TransformerLayer(
                d_model=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(2)  # 🆕 固定2层
        ])
        
        # Cross-Attention层: queries与img/pcd交互（只需要2层）
        self.cross_attn_layers = nn.ModuleList([
            TransformerLayer(
                d_model=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(2)  # 🆕 固定2层
        ])
        
        # 🆕 Output投影层
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
            query_list: [query_feats_c1, query_feats_c2, query_output] - 分离的query特征
            img_tokens: 处理后的图像tokens
            pcd_tokens: 处理后的点云tokens
        """
        # 🆕 投影tokens（对齐ICL-I2PReg）
        img_tokens = self.img_in_proj(img_feats)
        pcd_tokens = self.pcd_in_proj(pcd_feats)
        
        query = query_feats
        query_list = []
        
        # 🆕 对齐ICL-I2PReg: 只做2层（1个img block + 1个pcd block）
        # Block 1: Self-Attention + Cross-Attention with Image
        query_feats_s1 = self.self_attn_layers[0](
            q=query,
            k=query,
            v=query,
            q_embeds=query_pos_embeds,
            k_embeds=query_pos_embeds,
        )
        query_feats_c1 = self.cross_attn_layers[0](
            q=query_feats_s1,
            k=img_tokens,
            v=img_tokens,
            q_embeds=query_pos_embeds,
            k_embeds=img_pos_embeds,
        )
        query_list.append(query_feats_c1)  # 保存img query
        
        # Block 2: Self-Attention + Cross-Attention with PointCloud
        query_feats_s2 = self.self_attn_layers[1](
            q=query_feats_c1,
            k=query_feats_c1,
            v=query_feats_c1,
            q_embeds=query_pos_embeds,
            k_embeds=query_pos_embeds,
        )
        query_feats_c2 = self.cross_attn_layers[1](
            q=query_feats_s2,
            k=pcd_tokens,
            v=pcd_tokens,
            q_embeds=query_pos_embeds,
            k_embeds=pcd_pos_embeds,
        )
        query_list.append(query_feats_c2)  # 保存pcd query
        
        # 🆕 Output投影
        query_output = self.query_out_proj(query_feats_c2)
        query_list.append(query_output)  # 保存最终query
        
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
