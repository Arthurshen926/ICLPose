"""
Dynamic Feature Selector (Scale-Gated Residual Fusion)
=======================================================
接收多尺度残差特征图，通过可学习的尺度门控机制，
自适应地融合为统一的残差特征图。

核心设计:
  1. Scale Gate: 对每个尺度的残差做 GlobalAvgPool → MLP → Sigmoid，
     得到标量门控值 g_i，表示该尺度的信息量
  2. Softmax 归一化门控值，确保尺度之间的竞争
  3. 上采样所有残差到统一分辨率，按门控值加权
  4. 通道融合: Conv1×1 降维 + Conv3×3 空间融合

物理直觉:
  - 位姿偏差大时: coarse 残差信号强 → g_coarse 高
  - 位姿偏差中等: mid 残差信号强 → g_mid 高
  - 位姿偏差小时: fine 残差信号强 → g_fine 高

参考:
  - SENet (Hu et al., CVPR 2018): Squeeze-and-Excitation 通道注意力
  - FPN (Lin et al., CVPR 2017): 多尺度特征金字塔
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class ScaleGate(nn.Module):
    """
    尺度门控模块: 从残差特征图中估计该尺度的重要性分数
    
    GlobalAvgPool → MLP → Sigmoid → 标量 ∈ [0, 1]
    """
    
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # (B, C, 1, 1)
            nn.Flatten(),                  # (B, C)
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),      # (B, 1) 标量分数
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 残差特征图
        Returns:
            (B, 1) 重要性分数 (未归一化)
        """
        return self.gate(x)


class ResidualComputer(nn.Module):
    """
    残差计算模块: 计算查询特征与渲染特征之间的差异
    
    支持多种残差计算方式:
      - 'subtract': 简单相减 + L2_norm
      - 'concat': 拼接 [Q, R, Q-R]
      - 'correlation': 逐通道correlation (类似RAFT的cost volume简化版)
    """
    
    def __init__(self, feat_dim: int, mode: str = 'concat', out_dim: int = None):
        super().__init__()
        self.mode = mode
        self.feat_dim = feat_dim
        
        if mode == 'subtract':
            # 简单差值: output = Q - R, 维度不变
            self.out_dim = feat_dim
        elif mode == 'concat':
            # 拼接 [Q, R, Q-R] → Conv降维
            self.out_dim = out_dim or feat_dim
            self.proj = nn.Sequential(
                nn.Conv2d(feat_dim * 3, self.out_dim, 1, bias=False),
                nn.GroupNorm(min(8, self.out_dim), self.out_dim),
                nn.ReLU(inplace=True),
            )
        elif mode == 'correlation':
            # 逐通道cos相似度 + 差值拼接 → 降维
            self.out_dim = out_dim or feat_dim
            self.proj = nn.Sequential(
                nn.Conv2d(feat_dim + 1, self.out_dim, 1, bias=False),
                nn.GroupNorm(min(8, self.out_dim), self.out_dim),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(f"Unknown residual mode: {mode}")
    
    def forward(self, query: torch.Tensor, rendered: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, C, H, W) 查询图像特征
            rendered: (B, C, H, W) 渲染特征图 (同分辨率同维度)
        Returns:
            (B, C_out, H, W) 残差特征图
        """
        if self.mode == 'subtract':
            diff = query - rendered
            return F.normalize(diff, p=2, dim=1)
        
        elif self.mode == 'concat':
            diff = query - rendered
            concat = torch.cat([query, rendered, diff], dim=1)
            return self.proj(concat)
        
        elif self.mode == 'correlation':
            # 逐通道余弦相似度
            q_norm = F.normalize(query, p=2, dim=1)
            r_norm = F.normalize(rendered, p=2, dim=1)
            cosine = (q_norm * r_norm).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            diff = query - rendered
            concat = torch.cat([diff, cosine], dim=1)
            return self.proj(concat)


class DynamicFeatureSelector(nn.Module):
    """
    动态特征选择器: 多尺度残差 → Scale Gate → 加权融合 → 统一残差
    
    管理 N 个尺度的残差计算和自适应融合。
    
    Args:
        scale_configs: 每个尺度的配置
            [{'name': 'coarse', 'feat_dim': 32, 'resolution': (7, 10)}, ...]
        output_resolution: 融合后的统一分辨率 (H, W)
        hidden_dim: 融合后的通道数
        residual_mode: 残差计算方式 ('subtract', 'concat', 'correlation')
        residual_out_dim: 每个尺度残差的输出维度 (统一)
        gate_hidden: Scale Gate 的 MLP 隐藏维度
    """
    
    def __init__(
        self,
        scale_configs: List[Dict],
        output_resolution: Tuple[int, int] = (35, 46),
        hidden_dim: int = 128,
        residual_mode: str = 'concat',
        residual_out_dim: int = 64,
        gate_hidden: int = 32,
    ):
        super().__init__()
        
        self.scale_configs = scale_configs
        self.num_scales = len(scale_configs)
        self.output_resolution = output_resolution
        self.hidden_dim = hidden_dim
        self.residual_out_dim = residual_out_dim
        
        # 每个尺度的残差计算器
        self.residual_computers = nn.ModuleDict()
        for cfg in scale_configs:
            name = cfg['name']
            self.residual_computers[name] = ResidualComputer(
                feat_dim=cfg['feat_dim'],
                mode=residual_mode,
                out_dim=residual_out_dim,
            )
        
        # 每个尺度的 Scale Gate
        self.scale_gates = nn.ModuleDict()
        for cfg in scale_configs:
            name = cfg['name']
            self.scale_gates[name] = ScaleGate(
                in_channels=residual_out_dim,
                hidden_dim=gate_hidden,
            )
        
        # 通道融合: 所有尺度的残差拼接 → 降维
        total_channels = residual_out_dim * self.num_scales
        self.channel_fusion = nn.Sequential(
            nn.Conv2d(total_channels, hidden_dim, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 打印配置
        scale_names = [c['name'] for c in scale_configs]
        print(f"[DynamicFeatureSelector] scales={scale_names}, "
              f"out_res={output_resolution}, hidden={hidden_dim}, "
              f"residual_mode={residual_mode}")
    
    def forward(
        self,
        query_feats: Dict[str, torch.Tensor],
        rendered_feats: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            query_feats: {scale_name: (B, C_i, H_i, W_i)} 查询特征 (N个尺度)
            rendered_feats: {scale_name: (B, C_i, H_i, W_i)} 渲染特征 (N个尺度)
            
        Returns:
            fused: (B, hidden_dim, H_out, W_out) 融合残差特征
            gate_values: {scale_name: float} 各尺度的门控值 (用于可视化/监控)
        """
        B = next(iter(query_feats.values())).shape[0]
        H_out, W_out = self.output_resolution
        
        # Step 1: 计算每个尺度的残差
        residuals = {}
        for cfg in self.scale_configs:
            name = cfg['name']
            q = query_feats[name]     # (B, C_i, H_i, W_i)
            r = rendered_feats[name]  # (B, C_i, H_i, W_i)
            residuals[name] = self.residual_computers[name](q, r)
            # → (B, residual_out_dim, H_i, W_i)
        
        # Step 2: 计算 Scale Gate 值
        raw_gates = []
        gate_names = []
        for cfg in self.scale_configs:
            name = cfg['name']
            g = self.scale_gates[name](residuals[name])  # (B, 1)
            raw_gates.append(g)
            gate_names.append(name)
        
        # Softmax 归一化 (尺度间竞争)
        raw_gates = torch.cat(raw_gates, dim=1)   # (B, N_scales)
        gate_weights = F.softmax(raw_gates, dim=1) # (B, N_scales)
        
        # 记录 gate 值 (batch 平均, 用于监控)
        gate_values = {}
        for i, name in enumerate(gate_names):
            gate_values[name] = gate_weights[:, i].mean().item()
        
        # Step 3: 上采样 + 门控加权
        weighted_residuals = []
        for i, cfg in enumerate(self.scale_configs):
            name = cfg['name']
            res = residuals[name]  # (B, C_r, H_i, W_i)
            
            # 上采样到统一分辨率
            if res.shape[2] != H_out or res.shape[3] != W_out:
                res = F.interpolate(
                    res, size=(H_out, W_out),
                    mode='bilinear', align_corners=False
                )
            
            # 门控加权
            g = gate_weights[:, i].view(B, 1, 1, 1)  # (B, 1, 1, 1)
            weighted_residuals.append(res * g)
        
        # Step 4: 拼接 + 通道融合
        concat = torch.cat(weighted_residuals, dim=1)  # (B, N*C_r, H_out, W_out)
        fused = self.channel_fusion(concat)             # (B, hidden_dim, H_out, W_out)
        
        return fused, gate_values
