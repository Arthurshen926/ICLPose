"""
Dual-Head Output Module
========================
从 ConvGRU 隐藏状态解码两个输出:
  1. Flow Head: Dense flow field (Δu, Δv) + 逐像素置信度
  2. Pose: 通过 FlowToPose 几何层从光流推导位姿 (替代原有 PoseHead)

设计原则:
  - Flow Head 提供显式的 dense 像素级对应 (密集梯度信号)
  - DifferentiableFlowToPose 通过 Image Jacobian 将光流转为 6DoF 位姿
  - 位姿梯度自动反传到 FlowHead → 姿态误差驱动更好的光流预测

V8 核心改进:
  FlowHead → predicted flow → (geometry WLS) → pose xi
  位姿不再依赖独立的 PoseHead 回归, 而是严格从几何关系推导

参考:
  - RAFT (Teed & Deng, ECCV 2020): flow head 设计
  - DROID-SLAM (Teed & Deng, NeurIPS 2021): confidence-weighted flow + differentiable BA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from modules.flow_to_pose import DifferentiableFlowToPose


class PoseHead(nn.Module):
    """
    位姿增量预测头: hidden state → Δξ ∈ se(3)
    
    流程: ConvGRU hidden (B, C_h, H, W)
      → 卷积逐步降维 (保留空间结构) → flatten → MLP → (B, 6)
    
    关键改进: 不用 GAP (会丢失空间信息, 导致旋转方向不可区分)
    而是用步进卷积逐步缩小空间维度, FC 层直接看到空间分布的特征
    
    例:  (B, 128, 35, 46)
      → Conv stride=2 → (B, 64, 18, 23)
      → Conv stride=2 → (B, 32, 9, 12)
      → Conv stride=2 → (B, 16, 5, 6)
      → flatten → (B, 480)
      → FC → (B, 6)
    """
    
    def __init__(self, hidden_dim: int, mlp_dim: int = 256):
        super().__init__()
        
        # Spatial downsampling (preserves direction info)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, stride=2, padding=1),
            nn.GroupNorm(4, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
        )
        
        # Compute flattened dim: 35→18→9→5, 46→23→12→6 → 16*5*6=480
        self._spatial_dim = 16 * 5 * 6  # For 35×46 input
        
        # FC layers after flatten
        self.fc = nn.Sequential(
            nn.Linear(self._spatial_dim, mlp_dim),
            nn.ReLU(inplace=True),
        )
        
        # 分离旋转和平移头
        self.rotation_head = nn.Linear(mlp_dim, 3)
        self.translation_head = nn.Linear(mlp_dim, 3)
        
        # 适中初始化: 确保初始 xi 有足够幅度产生有意义的 loss 变化
        nn.init.normal_(self.rotation_head.weight, std=0.01)
        nn.init.zeros_(self.rotation_head.bias)
        nn.init.normal_(self.translation_head.weight, std=0.01)
        nn.init.zeros_(self.translation_head.bias)
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (B, C_h, H, W) GRU 隐藏状态
        Returns:
            xi: (B, 6) se(3) 增量 [v_x, v_y, v_z, ω_x, ω_y, ω_z]
        """
        x = self.spatial_encoder(hidden)         # (B, 16, 5, 6)
        x = x.flatten(1)                         # (B, 480)
        feat = self.fc(x)                         # (B, mlp_dim)
        
        v = self.translation_head(feat)           # (B, 3)
        omega = self.rotation_head(feat)          # (B, 3)
        
        xi = torch.cat([v, omega], dim=-1)        # (B, 6)
        return xi


class FlowHead(nn.Module):
    """
    Dense Flow 预测头: hidden state → per-pixel (Δu, Δv, confidence)
    
    流程: ConvGRU hidden (B, C_h, H, W)
      → Conv layers → (B, 3, H, W) [Δu, Δv, log_conf]
    
    输出:
      - flow: (B, 2, H, W) 像素偏移
      - log_confidence: (B, 1, H, W) 对数置信度 (用于 confidence-weighted loss)
    
    参考 DROID-SLAM 的 confidence weight 设计:
      L_flow = Σ conf * ||flow_pred - flow_gt||₁ - λ * log(conf)
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int = 64):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, intermediate_dim, 3, padding=1),
            nn.GroupNorm(8, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_dim, intermediate_dim, 3, padding=1),
            nn.GroupNorm(8, intermediate_dim),
            nn.ReLU(inplace=True),
        )
        
        # 分离 flow 和 confidence
        self.flow_conv = nn.Conv2d(intermediate_dim, 2, 3, padding=1)
        self.conf_conv = nn.Conv2d(intermediate_dim, 1, 3, padding=1)
        
        # 小随机初始化 flow (非零以确保梯度穿透到前层)
        nn.init.normal_(self.flow_conv.weight, std=0.001)
        nn.init.zeros_(self.flow_conv.bias)
        
        # confidence 初始化为 0 (sigmoid → 0.5, exp → 1.0)
        nn.init.zeros_(self.conf_conv.weight)
        nn.init.zeros_(self.conf_conv.bias)
    
    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: (B, C_h, H, W) GRU 隐藏状态
        Returns:
            flow: (B, 2, H, W) 像素偏移 [Δu, Δv]
            log_conf: (B, 1, H, W) 对数置信度
        """
        feat = self.conv(hidden)                # (B, inter_dim, H, W)
        flow = self.flow_conv(feat)             # (B, 2, H, W)
        log_conf = self.conf_conv(feat)         # (B, 1, H, W)
        
        return flow, log_conf


class DualHead(nn.Module):
    """
    双头输出模块: FlowHead (dense flow) + FlowToPose (geometric pose)
    
    核心改进 (V8):
      PoseHead 被替换为 DifferentiableFlowToPose 几何层:
      FlowHead → (Δu, Δv, confidence) → 加权最小二乘 → xi ∈ se(3)
      
      位姿梯度直接反传到 FlowHead, 让 FlowHead 学习产生几何一致的光流
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        pose_mlp_dim: int = 256,
        flow_intermediate_dim: int = 64,
        # FlowToPose 参数
        flow_to_pose_intrinsics: Optional[Dict[str, float]] = None,
        flow_to_pose_damping: float = 1e-3,
    ):
        super().__init__()
        
        self.flow_head = FlowHead(hidden_dim, flow_intermediate_dim)
        
        # FlowToPose 几何层 (如果提供了 intrinsics)
        self.flow_to_pose = None
        if flow_to_pose_intrinsics is not None:
            self.flow_to_pose = DifferentiableFlowToPose(
                intrinsics=flow_to_pose_intrinsics,
                damping=flow_to_pose_damping,
                negate_output=True,  # flow方向: GT→perturbed, 需取反得到修正
            )
            print(f"  [DualHead] Using FlowToPose (geometric), "
                  f"fx={flow_to_pose_intrinsics['fx']:.1f}")
        
        # 保留 PoseHead 作为 fallback (当没有深度时)
        self.pose_head = PoseHead(hidden_dim, pose_mlp_dim)
    
    def forward(
        self, 
        hidden: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden: (B, C_h, H, W) GRU 隐藏状态
            depth: (B, H, W) 深度图 (用于 FlowToPose)
                   如果提供且 flow_to_pose 已初始化, 使用几何方法推导位姿
                   否则退化为 PoseHead 回归
        Returns:
            dict:
                'xi': (B, 6) se(3) 位姿增量
                'flow': (B, 2, H, W) dense flow
                'log_confidence': (B, 1, H, W) 对数置信度
        """
        # Flow 预测 (始终需要)
        flow, log_conf = self.flow_head(hidden)
        
        # 位姿: 优先使用 FlowToPose 几何方法
        if self.flow_to_pose is not None and depth is not None:
            xi = self.flow_to_pose(flow, log_conf, depth)
        else:
            # Fallback: PoseHead 回归
            xi = self.pose_head(hidden)
        
        return {
            'xi': xi,
            'flow': flow,
            'log_confidence': log_conf,
        }
