"""
SE(3) Lie Algebra Utilities
============================
se(3) ↔ SE(3) 之间的转换：指数映射 (exp) 和对数映射 (log)。

参考:
  - Teed & Deng, "DROID-SLAM", NeurIPS 2021
  - Sola et al., "A micro Lie theory for state estimation in robotics", 2018
  - https://github.com/princeton-vl/DROID-SLAM/blob/main/droid_slam/lietorch/

约定:
  - se(3) ξ = [v, ω] ∈ ℝ⁶, 其中 v ∈ ℝ³ (平移), ω ∈ ℝ³ (旋转)
  - SE(3) T = [[R, t], [0, 1]] ∈ ℝ⁴ˣ⁴
  - 使用左乘约定: T_new = exp(ξ) · T_old
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def hat(omega: torch.Tensor) -> torch.Tensor:
    """
    Hat operator: ω ∈ ℝ³ → [ω]× ∈ so(3) (skew-symmetric matrix)
    
    Args:
        omega: (..., 3) rotation vector
    Returns:
        (..., 3, 3) skew-symmetric matrix
    """
    *batch, _ = omega.shape
    O = torch.zeros(*batch, 3, 3, device=omega.device, dtype=omega.dtype)
    O[..., 0, 1] = -omega[..., 2]
    O[..., 0, 2] = omega[..., 1]
    O[..., 1, 0] = omega[..., 2]
    O[..., 1, 2] = -omega[..., 0]
    O[..., 2, 0] = -omega[..., 1]
    O[..., 2, 1] = omega[..., 0]
    return O


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """
    SO(3) 指数映射: ω ∈ ℝ³ → R ∈ SO(3)
    Rodrigues' rotation formula.
    
    注意: 使用 gradient-safe 实现，避免 theta=0 时 sqrt 的梯度为 inf。
    
    Args:
        omega: (..., 3) rotation vector (角轴表示, 模长=旋转角度)
    Returns:
        (..., 3, 3) rotation matrix
    """
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # (..., 1)
    
    # 使用 clamp+sqrt 避免 sqrt(0) 的 inf 梯度
    # 对于 small angles，结果会被 torch.where 替换，所以 sqrt(1) 的值不影响结果
    theta_sq_safe = torch.clamp(theta_sq, min=1e-10)
    theta = torch.sqrt(theta_sq_safe)  # (..., 1) always > 0
    
    small = (theta_sq < 1e-8).squeeze(-1)
    
    # 正常情况: Rodrigues formula
    # sinc(θ) = sin(θ)/θ,  (1-cos(θ))/θ²
    sinc_theta = (torch.sin(theta) / theta).unsqueeze(-1)   # (..., 1, 1)
    one_minus_cos_over_sq = ((1 - torch.cos(theta)) / theta_sq_safe).unsqueeze(-1)   # (..., 1, 1)
    
    K = hat(omega)  # (..., 3, 3) skew of full omega
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    R = I + sinc_theta * K + one_minus_cos_over_sq * (K @ K)
    
    # small angle (Taylor): R ≈ I + [ω]×
    if small.any():
        R_small = I + hat(omega)
        R = torch.where(small[..., None, None], R_small, R)
    
    return R


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """
    SE(3) 指数映射: ξ = [v, ω] ∈ ℝ⁶ → T ∈ SE(3)
    
    使用完整的 Rodrigues 公式 (不是简单地把 R 和 v 组合):
    T = [[R, V·v], [0, 1]]
    其中 V = I + (1-cos(θ))/θ² · [ω]× + (θ-sin(θ))/θ³ · [ω]×²
    
    Args:
        xi: (..., 6) se(3) 向量, [v_x, v_y, v_z, ω_x, ω_y, ω_z]
    Returns:
        (..., 4, 4) SE(3) 变换矩阵
    """
    v = xi[..., :3]    # (..., 3) translation part
    omega = xi[..., 3:]  # (..., 3) rotation part
    
    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # (..., 1)
    
    # Gradient-safe sqrt: clamp then sqrt, use torch.where for small angles
    theta_sq_safe = torch.clamp(theta_sq, min=1e-10)
    theta = torch.sqrt(theta_sq_safe)
    
    small = (theta_sq < 1e-8).squeeze(-1)
    
    # Rotation
    R = so3_exp(omega)  # (..., 3, 3)
    
    # Translation: t = V · v
    # V = I + (1-cos(θ))/θ² · [ω]× + (θ-sin(θ))/θ³ · [ω]×²
    K = hat(omega)  # (..., 3, 3) skew of full omega (NOT unit axis)
    
    sin_theta = torch.sin(theta)      # (..., 1)
    cos_theta = torch.cos(theta)      # (..., 1)
    
    # 系数 (使用 theta_sq_safe 避免除零)
    a = ((1 - cos_theta) / theta_sq_safe).unsqueeze(-1)  # (..., 1, 1)
    b = ((theta - sin_theta) / (theta_sq_safe * theta)).unsqueeze(-1)  # (..., 1, 1)
    
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand_as(K)
    V = I + a * K + b * (K @ K)
    
    t = (V @ v.unsqueeze(-1)).squeeze(-1)  # (..., 3)
    
    # small angle: V ≈ I, t ≈ v
    if small.any():
        t = torch.where(small.unsqueeze(-1), v, t)
    
    # 组装 4x4 矩阵
    *batch, _ = xi.shape
    T = torch.zeros(*batch, 4, 4, device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    
    return T


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """
    SE(3) 对数映射: T ∈ SE(3) → ξ ∈ ℝ⁶
    
    Args:
        T: (..., 4, 4) SE(3) 变换矩阵
    Returns:
        (..., 6) se(3) 向量 [v_x, v_y, v_z, ω_x, ω_y, ω_z]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    
    # SO(3) log: 从旋转矩阵提取旋转向量
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # (...)
    
    eps = 1e-8
    small = (theta.abs() < eps)
    
    # 旋转向量
    # ω = θ / (2·sin(θ)) · [R - R^T]_vee
    sin_theta = torch.sin(theta)
    factor = theta / (2 * sin_theta + eps)  # (...)
    
    omega = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1) * factor.unsqueeze(-1)  # (..., 3)
    
    # small angle
    if small.any():
        omega_small = 0.5 * torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ], dim=-1)
        omega = torch.where(small.unsqueeze(-1), omega_small, omega)
    
    # V^{-1} 求解 v = V^{-1} · t
    theta_unsq = theta.unsqueeze(-1)  # (..., 1)
    theta_sq = theta_unsq * theta_unsq
    
    K = hat(omega)  # full omega hat (NOT unit axis)
    half_theta = 0.5 * theta_unsq
    
    cot_half = torch.cos(half_theta) / (torch.sin(half_theta) + eps)
    
    # V^{-1} = I - 0.5·[ω]× + (1 - θ/2·cot(θ/2))/θ² · [ω]×²
    a = 1.0 - half_theta * cot_half  # (..., 1)
    a = (a / (theta_sq + eps)).unsqueeze(-1)  # (..., 1, 1)
    
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand_as(K)
    V_inv = I - 0.5 * K + a * (K @ K)
    
    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)  # (..., 3)
    
    # small angle: V^{-1} ≈ I
    if small.any():
        v = torch.where(small.unsqueeze(-1), t, v)
    
    return torch.cat([v, omega], dim=-1)  # (..., 6)


def pose_compose(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """
    位姿复合: T_out = T1 · T2
    
    Args:
        T1, T2: (..., 4, 4) SE(3) matrices
    Returns:
        (..., 4, 4) composed transform
    """
    return T1 @ T2


def pose_inverse(T: torch.Tensor) -> torch.Tensor:
    """
    位姿求逆: T^{-1}
    
    对于 SE(3): T^{-1} = [[R^T, -R^T·t], [0, 1]]
    
    Args:
        T: (..., 4, 4) SE(3) matrix
    Returns:
        (..., 4, 4) inverse transform
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv
    T_inv[..., 3, 3] = 1.0
    
    return T_inv


def compute_gt_flow(
    depth_gt: torch.Tensor,
    pose_gt: torch.Tensor,
    pose_current: torch.Tensor,
    intrinsics: dict,
) -> dict:
    """
    计算 GT flow field: 从 pose_gt 视角到 pose_current 视角的像素偏移
    
    支持单帧和 batch:
      - 单帧: depth (H,W), pose (4,4)
      - Batch: depth (B,H,W), pose (B,4,4)
    
    对于查询图像中的每个像素 (u, v):
      1. 用 pose_gt 的深度图反投影到3D世界点
      2. 用 pose_current 将3D世界点投影回图像 → (u', v')
      3. flow = (u' - u, v' - v)
    
    Args:
        depth_gt: (B, H, W) or (H, W) GT位姿下的渲染深度图
        pose_gt: (B, 4, 4) or (4, 4) GT相机位姿 (w2c)
        pose_current: (B, 4, 4) or (4, 4) 当前估计位姿 (w2c)
        intrinsics: {'fx':, 'fy':, 'cx':, 'cy':}
        
    Returns:
        dict:
            'flow': (B, 2, H, W) or (2, H, W) GT flow field [Δu, Δv]
            'valid_mask': (B, 1, H, W) or (1, H, W) 有效像素mask
    """
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # 处理 batch 维度
    batched = depth_gt.dim() == 3
    if not batched:
        depth_gt = depth_gt.unsqueeze(0)
        pose_gt = pose_gt.unsqueeze(0)
        pose_current = pose_current.unsqueeze(0)
    
    B, H, W = depth_gt.shape
    device = depth_gt.device
    
    # 创建像素网格
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )  # (H, W) each
    
    flows = []
    valids = []
    
    for b in range(B):
        z = depth_gt[b]  # (H, W)
        valid_depth = z > 0.05
        
        # 反投影到相机坐标系
        x_c = (u_coords - cx) / fx * z
        y_c = (v_coords - cy) / fy * z
        points_cam = torch.stack([x_c, y_c, z], dim=-1)  # (H, W, 3)
        
        # 相机坐标 → 世界坐标 (pose_gt 是 w2c: p_c = R·p_w + t)
        R_gt = pose_gt[b, :3, :3]
        t_gt = pose_gt[b, :3, 3]
        R_gt_inv = R_gt.T
        t_gt_inv = -R_gt.T @ t_gt
        
        points_flat = points_cam.reshape(-1, 3)  # (HW, 3)
        points_world = (R_gt_inv @ points_flat.T).T + t_gt_inv  # (HW, 3)
        
        # 世界坐标 → 当前位姿的相机坐标
        R_cur = pose_current[b, :3, :3]
        t_cur = pose_current[b, :3, 3]
        points_cur_cam = (R_cur @ points_world.T).T + t_cur  # (HW, 3)
        
        # 投影到像素坐标
        z_cur = points_cur_cam[:, 2]
        valid_z = z_cur > 0.05
        
        u_proj = fx * points_cur_cam[:, 0] / (z_cur + 1e-8) + cx
        v_proj = fy * points_cur_cam[:, 1] / (z_cur + 1e-8) + cy
        
        u_proj = u_proj.reshape(H, W)
        v_proj = v_proj.reshape(H, W)
        
        flow_u = u_proj - u_coords
        flow_v = v_proj - v_coords
        flow = torch.stack([flow_u, flow_v], dim=0)  # (2, H, W)
        
        valid_z = valid_z.reshape(H, W)
        valid_proj = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)
        valid = valid_depth & valid_z & valid_proj  # (H, W)
        
        flows.append(flow)
        valids.append(valid)
    
    flow_out = torch.stack(flows, dim=0)                  # (B, 2, H, W)
    valid_out = torch.stack(valids, dim=0).unsqueeze(1)    # (B, 1, H, W)
    
    if not batched:
        flow_out = flow_out.squeeze(0)    # (2, H, W)
        valid_out = valid_out.squeeze(0)  # (1, H, W)
    
    return {'flow': flow_out, 'valid_mask': valid_out}
