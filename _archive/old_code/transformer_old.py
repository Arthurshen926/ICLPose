"""
Transformer层实现
完全对齐ICL-I2PReg的vision3d实现

参考: ICL-I2PReg/vision3d-main/vision3d/layers/transformer.py
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def build_dropout_layer(dropout: Optional[float]):
    """构建Dropout层"""
    if dropout is not None and dropout > 0:
        return nn.Dropout(dropout)
    return nn.Identity()


def build_act_layer(act_cfg: Union[str, dict]):
    """构建激活层"""
    if isinstance(act_cfg, dict):
        act_cfg = act_cfg.get('type', 'ReLU')
    
    if act_cfg == 'ReLU' or act_cfg == 'relu':
        return nn.ReLU(inplace=True)
    elif act_cfg == 'GELU' or act_cfg == 'gelu':
        return nn.GELU()
    elif act_cfg == 'LeakyReLU' or act_cfg == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    else:
        raise ValueError(f'Unsupported activation: {act_cfg}')


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Query, Key, Value投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = build_dropout_layer(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None, attention_mask=None):
        """
        前向传播
        
        Args:
            query: (B, N, C) 查询特征
            key: (B, M, C) 键特征
            value: (B, M, C) 值特征
            key_padding_mask: (B, M) padding掩码，True表示需要mask的位置
            attention_mask: (N, M) 注意力掩码
            
        Returns:
            output: (B, N, C) 输出特征
            attention_weights: (B, num_heads, N, M) 注意力权重
        """
        batch_size = query.shape[0]
        
        # 线性投影并reshape为多头格式
        q = self.q_proj(query)  # (B, N, C)
        k = self.k_proj(key)    # (B, M, C)
        v = self.v_proj(value)  # (B, M, C)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)  # (B, H, N, D)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.num_heads)  # (B, H, M, D)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.num_heads)  # (B, H, M, D)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, H, N, M)
        
        # 应用padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, M), True表示需要mask的位置
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, M)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 应用attention mask
        if attention_mask is not None:
            # attention_mask: (N, M)
            scores = scores + attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, N, M)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, v)  # (B, H, N, D)
        output = rearrange(output, 'b h n d -> b n (h d)')  # (B, N, C)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output, attention_weights


class AttentionLayer(nn.Module):
    """注意力层（多头注意力 + 残差连接 + LayerNorm）"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, key_padding_mask=None, attention_mask=None):
        """
        前向传播
        
        Args:
            query: (B, N, C) 查询特征
            key: (B, M, C) 键特征
            value: (B, M, C) 值特征
            key_padding_mask: (B, M) padding掩码
            attention_mask: (N, M) 注意力掩码
            
        Returns:
            output: (B, N, C) 输出特征
        """
        # 多头注意力
        attn_output, _ = self.attention(query, key, value, key_padding_mask, attention_mask)
        
        # Dropout + 残差连接 + LayerNorm
        output = self.norm(query + self.dropout(attn_output))
        
        return output


class AttentionOutput(nn.Module):
    """注意力输出层（FFN + 残差连接 + LayerNorm）"""
    
    def __init__(self, d_model, d_feedforward, dropout=0.1, activation='relu'):
        """
        Args:
            d_model: 模型维度
            d_feedforward: 前馈网络维度
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.linear2 = nn.Linear(d_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = build_dropout_layer(dropout)
        self.activation = build_act_layer(activation)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, N, C) 输入特征
            
        Returns:
            output: (B, N, C) 输出特征
        """
        # 前馈网络
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        # Dropout + 残差连接 + LayerNorm
        output = self.norm(x + self.dropout(ffn_output))
        
        return output


class TransformerLayer(nn.Module):
    """Transformer层（自注意力 + 交叉注意力 + FFN）"""
    
    def __init__(self, d_model, num_heads, d_feedforward=None, dropout=0.1, activation='relu'):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_feedforward: 前馈网络维度（默认为4*d_model）
            dropout: Dropout概率
            activation: 激活函数类型
        """
        super().__init__()
        if d_feedforward is None:
            d_feedforward = 4 * d_model
        
        # 自注意力层
        self.self_attn = AttentionLayer(d_model, num_heads, dropout)
        
        # 交叉注意力层
        self.cross_attn = AttentionLayer(d_model, num_heads, dropout)
        
        # 前馈网络
        self.ffn = AttentionOutput(d_model, d_feedforward, dropout, activation)
        
    def forward(self, query, key, value, 
                query_padding_mask=None, key_padding_mask=None,
                self_attention_mask=None, cross_attention_mask=None):
        """
        前向传播
        
        Args:
            query: (B, N, C) 查询特征
            key: (B, M, C) 键特征
            value: (B, M, C) 值特征
            query_padding_mask: (B, N) 查询padding掩码
            key_padding_mask: (B, M) 键padding掩码
            self_attention_mask: (N, N) 自注意力掩码
            cross_attention_mask: (N, M) 交叉注意力掩码
            
        Returns:
            output: (B, N, C) 输出特征
        """
        # 自注意力
        query = self.self_attn(
            query, query, query,
            key_padding_mask=query_padding_mask,
            attention_mask=self_attention_mask
        )
        
        # 交叉注意力
        query = self.cross_attn(
            query, key, value,
            key_padding_mask=key_padding_mask,
            attention_mask=cross_attention_mask
        )
        
        # 前馈网络
        output = self.ffn(query)
        
        return output
