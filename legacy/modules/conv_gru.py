"""
Convolutional GRU Module
========================
用于迭代位姿更新的循环单元。接收融合残差特征图，
维护空间隐藏状态，输出更新后的隐藏状态。

参考:
  - Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow," ECCV 2020
  - Teed & Deng, "DROID-SLAM," NeurIPS 2021
  - Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder," EMNLP 2014
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """
    卷积GRU单元 (2D spatial)
    
    标准GRU公式但用卷积替代全连接:
      z = σ(Conv([h_{t-1}, x_t]))       # 更新门
      r = σ(Conv([h_{t-1}, x_t]))       # 重置门
      h̃ = tanh(Conv([r ⊙ h_{t-1}, x_t])) # 候选隐藏状态
      h_t = (1-z) ⊙ h_{t-1} + z ⊙ h̃    # 最终隐藏状态
    
    Args:
        input_dim: 输入特征维度 (E_fused 的通道数)
        hidden_dim: 隐藏状态维度
        kernel_size: 卷积核大小 (默认3)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # 更新门 z 和重置门 r 合并计算
        self.conv_gates = nn.Conv2d(
            input_dim + hidden_dim, 2 * hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True
        )
        
        # 候选隐藏状态
        self.conv_candidate = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size=kernel_size, padding=padding, bias=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) 输入特征图 (融合残差)
            h: (B, C_h, H, W) 上一步隐藏状态 (None时自动初始化为0)
        Returns:
            h_new: (B, C_h, H, W) 更新后的隐藏状态
        """
        B, _, H, W = x.shape
        
        if h is None:
            h = torch.zeros(B, self.hidden_dim, H, W,
                          device=x.device, dtype=x.dtype)
        
        # 拼接 [h, x]
        combined = torch.cat([h, x], dim=1)    # (B, C_h + C_in, H, W)
        
        # 更新门 + 重置门
        gates = self.conv_gates(combined)       # (B, 2*C_h, H, W)
        z, r = gates.chunk(2, dim=1)            # 各 (B, C_h, H, W)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        
        # 候选隐藏状态
        combined_r = torch.cat([r * h, x], dim=1)  # (B, C_h + C_in, H, W)
        h_candidate = torch.tanh(self.conv_candidate(combined_r))
        
        # 更新
        h_new = (1 - z) * h + z * h_candidate
        
        return h_new


class ConvGRUBlock(nn.Module):
    """
    ConvGRU + 额外卷积处理，构成完整的更新块。
    
    参考 RAFT 的 UpdateBlock:
      1. ConvGRU 更新隐藏状态
      2. 从隐藏状态分别解码 pose delta 和 flow field
    
    Args:
        input_dim: 融合残差的通道数
        hidden_dim: GRU隐藏状态维度
        context_dim: 上下文特征维度 (可选，用于注入额外信息)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        context_dim: int = 0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 输入预处理: 将融合残差映射到合适维度
        total_input = input_dim + context_dim
        self.input_encoder = nn.Sequential(
            nn.Conv2d(total_input, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # ConvGRU核心
        self.gru = ConvGRUCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=3
        )
    
    def forward(
        self,
        residual: torch.Tensor,
        hidden: torch.Tensor = None,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            residual: (B, C_in, H, W) 融合残差特征
            hidden: (B, C_h, H, W) 上次隐藏状态
            context: (B, C_ctx, H, W) 可选上下文 (如查询图像编码)
        Returns:
            hidden: (B, C_h, H, W) 更新后的隐藏状态
        """
        if context is not None:
            x = torch.cat([residual, context], dim=1)
        else:
            x = residual
        
        x = self.input_encoder(x)
        hidden = self.gru(x, hidden)
        
        return hidden
