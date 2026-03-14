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


def _solve_weighted_normal_eq(
    flow_u: torch.Tensor,
    flow_v: torch.Tensor,
    w_u: torch.Tensor,
    w_v: torch.Tensor,
    Ju: torch.Tensor,
    Jv: torch.Tensor,
    damping: float,
    adaptive_damping: bool = False,
    adaptive_damping_max: float = 0.1,
    adaptive_damping_cond_thresh: float = 1e4,
) -> torch.Tensor:
    """Solve (J^T W J + λD) ξ = J^T W f.

    Supports directional weights (separate w_u, w_v) and adaptive damping
    that scales with the condition number of J^T W J.
    """
    w_u_unsq = w_u.unsqueeze(-1)                     # (B, N, 1)
    w_v_unsq = w_v.unsqueeze(-1)
    wJu = Ju * w_u_unsq
    wJv = Jv * w_v_unsq

    JtWJ = torch.bmm(Ju.transpose(1, 2), wJu) + \
           torch.bmm(Jv.transpose(1, 2), wJv)

    diag = torch.diagonal(JtWJ, dim1=-2, dim2=-1)

    if adaptive_damping:
        # Scale damping by condition number: higher cond → stronger damping
        # cond ≈ max(diag) / min(diag) as cheap approximation
        diag_clamped = diag.clamp(min=1e-8)
        cond_approx = diag_clamped.max(dim=1, keepdim=True).values / \
                      diag_clamped.min(dim=1, keepdim=True).values
        # Linear ramp: damping scales from base to max when cond > threshold
        scale = (cond_approx / adaptive_damping_cond_thresh).clamp(min=1.0)
        eff_damping = (damping * scale).clamp(max=adaptive_damping_max)
        diag_damping = eff_damping * diag_clamped
    else:
        diag_damping = damping * diag.clamp(min=1e-6)

    JtWJ = JtWJ + torch.diag_embed(diag_damping)

    JtWr = torch.bmm(
        Ju.transpose(1, 2), (w_u * flow_u).unsqueeze(-1)
    ).squeeze(-1) + torch.bmm(
        Jv.transpose(1, 2), (w_v * flow_v).unsqueeze(-1)
    ).squeeze(-1)

    return torch.linalg.solve(JtWJ, JtWr)


def diff_pose_solve(
    flow: torch.Tensor,
    confidence: torch.Tensor,
    Ju: torch.Tensor,
    Jv: torch.Tensor,
    valid: torch.Tensor,
    damping: float = 1e-3,
    irls_iters: int = 0,
    irls_huber_k: float = 1.345,
    pixel_stride: int = 1,
    adaptive_damping: bool = False,
    adaptive_damping_max: float = 0.1,
    adaptive_damping_cond_thresh: float = 1e4,
) -> torch.Tensor:
    """
    可微分的加权最小二乘位姿求解器, 支持 IRLS 鲁棒估计和方向性置信度.

    给定预测的 dense flow 和 Image Jacobian, 求解 6-DOF 位姿增量.
    可选 IRLS: 初始 WLS 求解后, 基于残差用 Huber 权重迭代 re-weight,
    抑制 flow 预测中的离群像素 (遮挡/误匹配).

    Args:
        flow: (B, 2, H, W) 预测的像素位移 [Δu, Δv]
        confidence: (B, 1, H, W) or (B, 2, H, W) 每像素权重 [0, 1]
            When shape is (B, 2, H, W), channel 0 = u-direction weight,
            channel 1 = v-direction weight (directional confidence).
        Ju: (B, N, 6) Image Jacobian ∂u/∂ξ
        Jv: (B, N, 6) Image Jacobian ∂v/∂ξ
        valid: (B, N) 有效像素 mask
        damping: LM 阻尼系数
        irls_iters: IRLS 迭代次数 (0 = 纯 WLS, 无 IRLS)
        irls_huber_k: Huber loss 阈值 (以 MAD 的倍数表示, 默认 1.345)
        pixel_stride: spatial stride for pixel subsampling (>1 reduces
                      correlated flow errors in repetitive-texture scenes)
        adaptive_damping: enable condition-number-adaptive damping
        adaptive_damping_max: maximum damping coefficient
        adaptive_damping_cond_thresh: condition number threshold for scaling

    Returns:
        delta_xi: (B, 6) se(3) 更新向量
    """
    B = flow.shape[0]
    device = flow.device

    flow_u = flow[:, 0].reshape(B, -1)
    flow_v = flow[:, 1].reshape(B, -1)

    # Support both scalar and directional confidence
    if confidence.shape[1] == 2:
        w_base_u = confidence[:, 0].reshape(B, -1) * valid.float()
        w_base_v = confidence[:, 1].reshape(B, -1) * valid.float()
    else:
        w_scalar = confidence[:, 0].reshape(B, -1) * valid.float()
        w_base_u = w_scalar
        w_base_v = w_scalar

    # Pixel subsampling: select every `pixel_stride` pixel in each spatial dim
    # to decorrelate flow errors in repetitive-texture scenes
    if pixel_stride > 1:
        H, W = flow.shape[-2:]
        # Build a stride mask on the 2D grid
        mask_2d = torch.zeros(H, W, device=device, dtype=torch.bool)
        mask_2d[::pixel_stride, ::pixel_stride] = True
        mask_1d = mask_2d.reshape(-1)  # (H*W,)
        # Index into flattened arrays
        idx = mask_1d.nonzero(as_tuple=True)[0]
        flow_u = flow_u[:, idx]
        flow_v = flow_v[:, idx]
        w_base_u = w_base_u[:, idx]
        w_base_v = w_base_v[:, idx]
        Ju = Ju[:, idx]
        Jv = Jv[:, idx]
        valid = valid[:, idx]

    # Initial WLS solve
    delta_xi = _solve_weighted_normal_eq(
        flow_u, flow_v, w_base_u, w_base_v, Ju, Jv, damping,
        adaptive_damping=adaptive_damping,
        adaptive_damping_max=adaptive_damping_max,
        adaptive_damping_cond_thresh=adaptive_damping_cond_thresh)

    # IRLS: iteratively re-weight based on residuals
    for _ in range(irls_iters):
        # Compute residuals: r_u = J_u @ xi - flow_u, r_v = J_v @ xi - flow_v
        pred_u = torch.bmm(Ju, delta_xi.unsqueeze(-1)).squeeze(-1)  # (B, N)
        pred_v = torch.bmm(Jv, delta_xi.unsqueeze(-1)).squeeze(-1)
        res = torch.sqrt((pred_u - flow_u)**2 + (pred_v - flow_v)**2 + 1e-8)

        # Adaptive Huber threshold: median(res) * irls_huber_k
        # Use weighted median (only valid pixels) for robustness
        res_valid = res + (1.0 - valid.float()) * 1e6  # push invalid to high
        median_res = res_valid.median(dim=1, keepdim=True).values
        threshold = (median_res * irls_huber_k).clamp(min=0.1)

        # Huber weight: w = 1 if |r| < k, else k/|r|
        huber_w = torch.where(res < threshold, torch.ones_like(res),
                              threshold / res.clamp(min=1e-6))

        # Combined weights: network confidence × Huber robustness × validity
        w_irls_u = w_base_u * huber_w
        w_irls_v = w_base_v * huber_w

        delta_xi = _solve_weighted_normal_eq(
            flow_u, flow_v, w_irls_u, w_irls_v, Ju, Jv, damping,
            adaptive_damping=adaptive_damping,
            adaptive_damping_max=adaptive_damping_max,
            adaptive_damping_cond_thresh=adaptive_damping_cond_thresh)

    # Soft safety clamp using tanh — preserves gradients unlike hard clamp
    # tanh(x/limit)*limit ≈ x for small x, saturates smoothly at ±limit
    trans_limit, rot_limit = 2.0, 1.5708
    delta_trans = torch.tanh(delta_xi[:, :3] / trans_limit) * trans_limit
    delta_rot = torch.tanh(delta_xi[:, 3:] / rot_limit) * rot_limit
    delta_xi = torch.cat([delta_trans, delta_rot], dim=1)

    return delta_xi
