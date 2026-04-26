"""
Differentiable Flow-to-Pose Conversion via Weighted Least Squares
==================================================================

核心思路:
  FlowHead 预测的光流场 + 深度图 + 相机内参 → 通过几何关系直接推导位姿变化量

原理 (小角度线性化):
  对于像素 (u, v) 在深度 Z 下, 相机运动 (ω, t) 产生的光流为:

    Δu = fx[-ωx·x·y + ωy(1+x²) - ωz·y + tx/Z - x·tz/Z]
    Δv = fy[-ωx(1+y²) + ωy·x·y + ωz·x + ty/Z - y·tz/Z]

  其中 x = (u-cx)/fx, y = (v-cy)/fy 是归一化像素坐标

  这是关于 ξ = [tx, ty, tz, ωx, ωy, ωz] 的 **线性方程**!
  每个像素给出 2 个方程, 6 个未知数.
  N 个像素 → 2N × 6 超定系统 → 加权最小二乘求解

优势:
  - 几何严格: 利用投影几何而非学习的回归
  - 完全可微: 梯度流过 flow → Jacobian → pose
  - 密集信号: 1610 个像素的 flow 信号聚合为 6DoF 位姿
  - 鲁棒性: 置信度加权天然抑制遮挡/错误区域

参考:
  - Visual Servoing 中的 Image Jacobian / Interaction Matrix
  - DROID-SLAM: 类似的 differentiable BA layer
  - Direct Sparse Odometry (DSO): 光度法 + Jacobian
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


def flow_to_pose_weighted_lstsq(
    flow_pred: torch.Tensor,
    log_confidence: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: Dict[str, float],
    damping: float = 1e-3,
) -> torch.Tensor:
    """
    可微分的光流→位姿转换 (加权最小二乘)
    
    通过 Image Jacobian (Interaction Matrix) 将预测的光流场线性化为
    关于相机运动参数的超定方程组, 然后用加权最小二乘求解.
    
    Args:
        flow_pred: (B, 2, H, W) 预测的光流 [Δu, Δv]
        log_confidence: (B, 1, H, W) 对数置信度 (FlowHead 输出)
        depth: (B, H, W) 深度图 (单位: 米)
        intrinsics: {'fx':, 'fy':, 'cx':, 'cy':} 缩放后的相机内参
        damping: Levenberg-Marquardt 阻尼因子 (数值稳定性)
    
    Returns:
        xi: (B, 6) se(3) 增量 [tx, ty, tz, ωx, ωy, ωz]
            注意: 此 xi 描述的是 GT→perturbed 方向的运动, 
            需要在外部取反得到 perturbed→GT 的修正量
    """
    B, _, H, W = flow_pred.shape
    N = H * W
    device = flow_pred.device
    
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    
    # 处理深度维度
    if depth.dim() == 4:
        depth = depth.squeeze(1)  # (B, 1, H, W) → (B, H, W)
    
    # ---- 像素网格 ----
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )  # (H, W)
    
    # 归一化像素坐标
    x = (u_coords - cx) / fx  # (H, W)
    y = (v_coords - cy) / fy  # (H, W)
    x = x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    y = y.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    
    # ---- 有效像素掩码 + 置信度 ----
    Z = depth  # (B, H, W)
    valid = (Z > 0.05).float()  # (B, H, W)
    Z_safe = Z.clamp(min=0.05)
    inv_Z = 1.0 / Z_safe  # (B, H, W)
    
    # 置信度权重
    conf = torch.exp(log_confidence.squeeze(1))  # (B, H, W)
    w = conf * valid  # (B, H, W), 无效像素权重为 0
    
    # ---- 光流值 ----
    fu = flow_pred[:, 0]  # (B, H, W) — Δu
    fv = flow_pred[:, 1]  # (B, H, W) — Δv
    
    # ---- Image Jacobian (Interaction Matrix) ----
    # 对于 Δu 行: J_u = [fx/Z, 0, -fx·x/Z, -fx·x·y, fx(1+x²), -fx·y]
    # 对于 Δv 行: J_v = [0, fy/Z, -fy·y/Z, -fy(1+y²), fy·x·y, fy·x]
    
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Ju columns: (B, H, W) each
    Ju_tx = fx * inv_Z
    Ju_ty = torch.zeros_like(x)
    Ju_tz = -fx * x * inv_Z
    Ju_wx = -fx * xy
    Ju_wy = fx * (1.0 + x2)
    Ju_wz = -fx * y
    
    # Jv columns: (B, H, W) each
    Jv_tx = torch.zeros_like(y)
    Jv_ty = fy * inv_Z
    Jv_tz = -fy * y * inv_Z
    Jv_wx = -fy * (1.0 + y2)
    Jv_wy = fy * xy
    Jv_wz = fy * x
    
    # Stack: (B, N, 6)
    Ju = torch.stack([Ju_tx, Ju_ty, Ju_tz, Ju_wx, Ju_wy, Ju_wz], dim=-1)  # (B, H, W, 6)
    Jv = torch.stack([Jv_tx, Jv_ty, Jv_tz, Jv_wx, Jv_wy, Jv_wz], dim=-1)  # (B, H, W, 6)
    
    Ju = Ju.reshape(B, N, 6)  # (B, N, 6)
    Jv = Jv.reshape(B, N, 6)  # (B, N, 6)
    
    w_flat = w.reshape(B, N)           # (B, N)
    fu_flat = fu.reshape(B, N)         # (B, N)
    fv_flat = fv.reshape(B, N)         # (B, N)
    
    # ---- 构建法方程 J^T W J ξ = J^T W f ----
    # 高效计算: 不需要显式构建 (2N, 6) 矩阵
    
    # J^T_u · diag(w) · J_u + J^T_v · diag(w) · J_v → (B, 6, 6)
    Ju_w = Ju * w_flat.unsqueeze(-1)   # (B, N, 6)
    Jv_w = Jv * w_flat.unsqueeze(-1)   # (B, N, 6)
    
    JtWJ = (torch.bmm(Ju_w.transpose(1, 2), Ju) + 
            torch.bmm(Jv_w.transpose(1, 2), Jv))  # (B, 6, 6)
    
    # J^T W f → (B, 6, 1)
    JtWf = (torch.bmm(Ju_w.transpose(1, 2), fu_flat.unsqueeze(-1)) + 
            torch.bmm(Jv_w.transpose(1, 2), fv_flat.unsqueeze(-1)))  # (B, 6, 1)
    
    # ---- LM 阻尼 + 求解 ----
    # (J^T W J + λI) ξ = J^T W f
    damping_mat = damping * torch.eye(6, device=device, dtype=JtWJ.dtype).unsqueeze(0)
    JtWJ_reg = JtWJ + damping_mat
    
    # Solve the 6×6 system
    xi = torch.linalg.solve(JtWJ_reg, JtWf).squeeze(-1)  # (B, 6)
    
    return xi


class DifferentiableFlowToPose(nn.Module):
    """
    可微分光流→位姿层 (作为 nn.Module 方便集成)
    
    将 FlowHead 的稠密光流预测 + 深度图 → 通过加权最小二乘得到 6DoF 位姿增量
    
    关键: 替代 PoseHead 的神经回归, 用几何方法直接从光流推导位姿
    
    梯度流:
      loss → xi → (J^T W J)^{-1} J^T W f → flow_pred, confidence → FlowHead → backbone
      位姿误差的梯度直接指导 FlowHead 产生更好的光流预测
    """
    
    def __init__(
        self,
        intrinsics: Dict[str, float],
        damping: float = 1e-3,
        negate_output: bool = True,
    ):
        """
        Args:
            intrinsics: {'fx':, 'fy':, 'cx':, 'cy':} 缩放后的相机内参
            damping: LM 阻尼因子
            negate_output: 如果为 True, 输出 -xi (从 flow方向==GT→perturbed 
                          转换为 correction方向==perturbed→GT)
        """
        super().__init__()
        self.intrinsics = intrinsics
        self.damping = damping
        self.negate_output = negate_output
    
    def forward(
        self,
        flow_pred: torch.Tensor,
        log_confidence: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            flow_pred: (B, 2, H, W) 预测光流
            log_confidence: (B, 1, H, W) 对数置信度
            depth: (B, H, W) or (B, 1, H, W) 深度图
        
        Returns:
            xi: (B, 6) se(3) 修正量 [tx, ty, tz, ωx, ωy, ωz]
        """
        xi = flow_to_pose_weighted_lstsq(
            flow_pred, log_confidence, depth, 
            self.intrinsics, self.damping,
        )
        
        if self.negate_output:
            xi = -xi
        
        return xi
