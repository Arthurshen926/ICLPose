"""
Geometry Solver
===============
可微分的加权最小二乘位姿求解器，从 dense flow + Image Jacobian 求解 6-DOF 相机运动增量。

从 ic_models/corr_pose_net.py 提取并独立化，供 MSFlowPoseNet 等多种架构复用。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_image_jacobian(
    depth: torch.Tensor,
    intrinsics: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 Image Jacobian: 像素位移关于相机运动 ξ 的 Jacobian.

    对于像素 (u,v) 在深度 Z 处，相机运动 ξ=[tx,ty,tz,ωx,ωy,ωz] 产生的像素位移:
      Δu = [fx/Z, 0, -fx·x/Z, -fx·xy, fx(1+x²), -fx·y] · ξ
      Δv = [0, fy/Z, -fy·y/Z, -fy(1+y²), fy·xy, fy·x] · ξ

    Args:
        depth: (B, H, W) 深度图
        intrinsics: {'fx', 'fy', 'cx', 'cy'}

    Returns:
        Ju: (B, N, 6) u方向的 Jacobian
        Jv: (B, N, 6) v方向的 Jacobian
        valid: (B, N) 有效像素 mask
    """
    B, H, W = depth.shape
    N = H * W
    device = depth.device

    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    x = (u_coords - cx) / fx  # (H, W)
    y = (v_coords - cy) / fy
    x = x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    y = y.unsqueeze(0).expand(B, -1, -1)

    Z = depth.clamp(min=0.05)
    inv_Z = 1.0 / Z
    valid = (depth > 0.05).reshape(B, N)

    x2, y2, xy = x * x, y * y, x * y

    # Ju: (B, H, W, 6)
    Ju = torch.stack([
        fx * inv_Z,                     # tx
        torch.zeros_like(x),            # ty
        -fx * x * inv_Z,                # tz
        -fx * xy,                        # ωx
        fx * (1.0 + x2),                # ωy
        -fx * y,                         # ωz
    ], dim=-1)

    Jv = torch.stack([
        torch.zeros_like(y),            # tx
        fy * inv_Z,                     # ty
        -fy * y * inv_Z,                # tz
        -fy * (1.0 + y2),               # ωx
        fy * xy,                         # ωy
        fy * x,                          # ωz
    ], dim=-1)

    return Ju.reshape(B, N, 6), Jv.reshape(B, N, 6), valid


def diff_pose_solve(
    flow: torch.Tensor,
    confidence: torch.Tensor,
    Ju: torch.Tensor,
    Jv: torch.Tensor,
    valid: torch.Tensor,
    damping: float = 1e-3,
) -> torch.Tensor:
    """
    可微分的加权最小二乘位姿求解器.

    给定预测的 dense flow (像素位移) 和 Image Jacobian (像素位移→相机运动的映射),
    通过加权正规方程求解 6-DOF 相机运动增量。

    数学:
      对每个像素 i, 我们希望: Ju[i]·δξ ≈ flow_u[i], Jv[i]·δξ ≈ flow_v[i]
      加权目标: min Σ w[i] · (||Ju[i]·δξ - flow_u[i]||² + ||Jv[i]·δξ - flow_v[i]||²)
      正规方程: (Σ J^T W J) δξ = Σ J^T W flow

    Args:
        flow: (B, 2, H, W) 预测的像素位移 [Δu, Δv]
        confidence: (B, 1, H, W) 每像素权重 [0, 1]
        Ju: (B, N, 6) Image Jacobian ∂u/∂ξ
        Jv: (B, N, 6) Image Jacobian ∂v/∂ξ
        valid: (B, N) 有效像素 mask
        damping: LM 阻尼项

    Returns:
        delta_xi: (B, 6) se(3) 更新向量 [vx, vy, vz, ωx, ωy, ωz]
    """
    B = flow.shape[0]
    device = flow.device

    # Flatten spatial dims
    flow_u = flow[:, 0].reshape(B, -1)              # (B, N)
    flow_v = flow[:, 1].reshape(B, -1)              # (B, N)
    w = confidence[:, 0].reshape(B, -1)             # (B, N)
    w = w * valid.float()                            # zero out invalid pixels

    # Weighted Jacobians
    w_unsq = w.unsqueeze(-1)                         # (B, N, 1)
    wJu = Ju * w_unsq                                # (B, N, 6)
    wJv = Jv * w_unsq                                # (B, N, 6)

    # JtWJ: (B, 6, 6)
    JtWJ = torch.bmm(Ju.transpose(1, 2), wJu) + \
           torch.bmm(Jv.transpose(1, 2), wJv)

    # LM damping
    JtWJ = JtWJ + damping * torch.eye(6, device=device).unsqueeze(0)

    # JtWr: (B, 6)
    JtWr = torch.bmm(
        Ju.transpose(1, 2), (w * flow_u).unsqueeze(-1)
    ).squeeze(-1) + torch.bmm(
        Jv.transpose(1, 2), (w * flow_v).unsqueeze(-1)
    ).squeeze(-1)

    # Solve 6×6 system
    delta_xi = torch.linalg.solve(JtWJ, JtWr)  # (B, 6)

    return delta_xi
