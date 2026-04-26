"""
Transformer层实现 - 完全对齐ICL-I2PReg的vision3d实现
参考: ICL-I2PReg/vision3d-main/vision3d/layers/transformer.py
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """
    多头注意力层，完全对齐ICL-I2PReg的vision3d实现
    支持position/rotation embeddings和attention weights
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V线性变换
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = build_dropout_layer(dropout)
        
    def _forward_impl(
        self,
        q: Tensor,  # (B, N_q, C)
        k: Tensor,  # (B, N_k, C)
        v: Tensor,  # (B, N_v, C)
        q_embeds: Optional[Tensor] = None,  # (B, N_q, C)
        k_embeds: Optional[Tensor] = None,  # (B, N_k, C)
        qk_embeds: Optional[Tensor] = None,  # (B, N_q, N_k, H)
        v_embeds: Optional[Tensor] = None,  # (B, N_v, C)
        weights: Optional[Tensor] = None,  # (B, N_q, N_k)
        masks: Optional[Tensor] = None,  # (B, N_q, N_k)
        return_attn: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            q: Query features [B, N_q, C]
            k: Key features [B, N_k, C]
            v: Value features [B, N_v, C]
            q_embeds: Query position embeddings
            k_embeds: Key position embeddings
            qk_embeds: Query-Key pair embeddings (用于RPE)
            v_embeds: Value embeddings
            weights: Attention weights (用于加权)
            masks: Attention masks (True=keep, False=mask)
            return_attn: 是否返回attention scores
        """
        B, N_q, _ = q.shape
        N_k = k.shape[1]
        H = self.num_heads
        C_per_head = self.d_k
        
        # Linear projections
        q_out = self.q_proj(q)  # (B, N_q, C)
        k_out = self.k_proj(k)  # (B, N_k, C)
        v_out = self.v_proj(v)  # (B, N_v, C)
        
        # Add embeddings to Q and K (用于APE - Absolute Position Encoding)
        if q_embeds is not None:
            q_out = q_out + self.q_proj(q_embeds)
        if k_embeds is not None:
            k_out = k_out + self.k_proj(k_embeds)
        if v_embeds is not None:
            v_out = v_out + self.v_proj(v_embeds)
        
        # Reshape to multi-head: (B, N, C) -> (B, H, N, C_per_head)
        q_out = q_out.view(B, N_q, H, C_per_head).transpose(1, 2)  # (B, H, N_q, C_per_head)
        k_out = k_out.view(B, N_k, H, C_per_head).transpose(1, 2)  # (B, H, N_k, C_per_head)
        v_out = v_out.view(B, N_k, H, C_per_head).transpose(1, 2)  # (B, H, N_v, C_per_head)
        
        # Compute attention scores: Q @ K^T
        attn_scores = torch.einsum('bhqc,bhkc->bhqk', q_out, k_out)  # (B, H, N_q, N_k)
        attn_scores = attn_scores / (C_per_head ** 0.5)
        
        # Add qk_embeds (用于RPE - Relative Position Encoding)
        if qk_embeds is not None:
            # qk_embeds: (B, N_q, N_k, H) -> (B, H, N_q, N_k)
            attn_scores = attn_scores + qk_embeds.permute(0, 3, 1, 2)
        
        # Apply masks (mask out invalid positions)
        if masks is not None:
            # masks: (B, N_q, N_k), True=keep, False=mask
            attn_scores = attn_scores.masked_fill(~masks.unsqueeze(1), float('-inf'))
        
        # Apply weights (用于加权attention)
        if weights is not None:
            # weights: (B, N_q, N_k)
            attn_scores = attn_scores + weights.unsqueeze(1).log()
        
        # Softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)  # (B, H, N_q, N_k)
        attn_probs = self.dropout(attn_probs)
        
        # Weighted sum: attn_probs @ V
        output = torch.einsum('bhqk,bhkc->bhqc', attn_probs, v_out)  # (B, H, N_q, C_per_head)
        
        # Reshape back: (B, H, N_q, C_per_head) -> (B, N_q, C)
        output = output.transpose(1, 2).contiguous().view(B, N_q, self.d_model)
        
        # Output projection
        output = self.o_proj(output)
        output = self.dropout(output)
        
        if return_attn:
            # Return attention scores (before softmax, after masking)
            return output, attn_scores
        return output
    
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_embeds: Optional[Tensor] = None,
        k_embeds: Optional[Tensor] = None,
        qk_embeds: Optional[Tensor] = None,
        v_embeds: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        return_attn: bool = False,
    ):
        return self._forward_impl(
            q, k, v,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            qk_embeds=qk_embeds,
            v_embeds=v_embeds,
            weights=weights,
            masks=masks,
            return_attn=return_attn,
        )


class AttentionLayer(nn.Module):
    """注意力层 = MultiHeadAttention + Dropout"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: Optional[float] = None):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.dropout = build_dropout_layer(dropout)
        
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_embeds: Optional[Tensor] = None,
        k_embeds: Optional[Tensor] = None,
        qk_embeds: Optional[Tensor] = None,
        v_embeds: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        return_attn: bool = False,
    ):
        output = self.attention(
            q, k, v,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            qk_embeds=qk_embeds,
            v_embeds=v_embeds,
            weights=weights,
            masks=masks,
            return_attn=return_attn,
        )
        if return_attn:
            hidden, attn_scores = output
            hidden = self.dropout(hidden)
            return hidden, attn_scores
        else:
            hidden = self.dropout(output)
            return hidden


class AttentionOutput(nn.Module):
    """注意力输出层 = Linear + Dropout + LayerNorm + Residual"""
    
    def __init__(self, d_model: int, dropout: Optional[float] = None):
        super().__init__()
        self.expand = nn.Linear(d_model, d_model, bias=False)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, hidden: Tensor, input_tensor: Tensor):
        """
        Args:
            hidden: Attention输出 [B, N, C]
            input_tensor: 残差连接的输入 [B, N, C]
        """
        hidden = self.expand(hidden)
        hidden = self.dropout(hidden)
        hidden = self.norm(hidden + input_tensor)
        return hidden


class TransformerLayer(nn.Module):
    """
    Transformer层 - 完全对齐ICL-I2PReg的vision3d实现
    包含: Attention + Output + FFN
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: Optional[float] = None,
        activation: Union[str, dict] = 'ReLU',
        **kwargs
    ):
        super().__init__()
        
        # Attention层
        self.attention = AttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout)
        
        # FFN层
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout1 = build_dropout_layer(dropout)
        self.act = build_act_layer(activation)
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout2 = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_embeds: Optional[Tensor] = None,
        k_embeds: Optional[Tensor] = None,
        qk_embeds: Optional[Tensor] = None,
        v_embeds: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        return_attn: bool = False,
    ):
        """
        Args:
            q, k, v: Query, Key, Value features
            q_embeds, k_embeds: Position embeddings for Q and K
            qk_embeds: Pair-wise embeddings (RPE)
            v_embeds: Value embeddings
            weights: Attention weights
            masks: Attention masks
            return_attn: Whether to return attention scores
        """
        # 1. Multi-Head Attention
        attn_output = self.attention(
            q, k, v,
            q_embeds=q_embeds,
            k_embeds=k_embeds,
            qk_embeds=qk_embeds,
            v_embeds=v_embeds,
            weights=weights,
            masks=masks,
            return_attn=return_attn,
        )
        
        if return_attn:
            hidden, attn_scores = attn_output
        else:
            hidden = attn_output
            attn_scores = None
        
        # 2. Attention Output (Residual + Norm)
        hidden = self.output(hidden, q)
        
        # 3. Feed-Forward Network
        output = self.linear1(hidden)
        output = self.dropout1(output)
        output = self.act(output)
        output = self.linear2(output)
        output = self.dropout2(output)
        output = self.norm(output + hidden)
        
        if return_attn:
            return output, attn_scores
        return output
