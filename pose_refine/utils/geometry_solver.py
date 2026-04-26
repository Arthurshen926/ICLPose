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


def pnp_ransac_solve(
    flow: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: dict,
    confidence: torch.Tensor = None,
    reprojection_threshold: float = 8.0,
    n_iters: int = 1000,
) -> torch.Tensor:
    """Non-differentiable PnP+RANSAC solver for robust pose estimation.

    Converts predicted flow + depth into 3D-2D correspondences and uses
    OpenCV PnP+RANSAC.  Much more robust to outlier flow than WLS.

    Args:
        flow: (B, 2, H, W) predicted optical flow
        depth: (B, 1, H, W) or (B, H, W) rendered depth
        intrinsics: {'fx', 'fy', 'cx', 'cy'}
        confidence: (B, 1, H, W) optional per-pixel confidence
        reprojection_threshold: RANSAC inlier threshold (pixels)
        n_iters: RANSAC iterations

    Returns:
        delta_xi: (B, 6) se(3) pose update vector
    """
    import cv2
    import numpy as np

    B, _, H, W = flow.shape
    device = flow.device

    if depth.dim() == 4:
        depth = depth.squeeze(1)  # (B, H, W)

    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )

    results = []
    for b in range(B):
        z = depth[b]  # (H, W)
        valid = z > 0.05

        # 3D points at current (rendered) pose in camera frame
        x3d = (u_coords - cx) * z / fx
        y3d = (v_coords - cy) * z / fy
        pts_3d = torch.stack([x3d, y3d, z], dim=-1)  # (H, W, 3)

        # Target 2D pixels = original pixel + flow
        fu = flow[b, 0]  # (H, W)
        fv = flow[b, 1]
        u_target = u_coords + fu
        v_target = v_coords + fv
        pts_2d = torch.stack([u_target, v_target], dim=-1)  # (H, W, 2)

        # Filter by valid depth + optional confidence
        mask = valid
        if confidence is not None:
            conf = confidence[b, 0] if confidence.dim() == 4 else confidence[b]
            mask = mask & (conf > 0.1)

        flat_3d = pts_3d[mask].detach().cpu().numpy().astype(np.float64)
        flat_2d = pts_2d[mask].detach().cpu().numpy().astype(np.float64)

        if len(flat_3d) < 6:
            results.append(torch.zeros(6, device=device))
            continue

        # Subsample if too many points (PnP is O(n))
        if len(flat_3d) > 5000:
            idx = np.random.choice(len(flat_3d), 5000, replace=False)
            flat_3d = flat_3d[idx]
            flat_2d = flat_2d[idx]

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                flat_3d, flat_2d, camera_matrix, None,
                iterationsCount=n_iters,
                reprojectionError=reprojection_threshold,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success or inliers is None or len(inliers) < 4:
                results.append(torch.zeros(6, device=device))
                continue

            # Refine with inliers
            if len(inliers) >= 6:
                success, rvec, tvec = cv2.solvePnP(
                    flat_3d[inliers.flatten()],
                    flat_2d[inliers.flatten()],
                    camera_matrix, None,
                    rvec=rvec, tvec=tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

            # PnP gives w2c of the "target" frame relative to camera frame
            # We need the delta: T_new = exp(xi) @ T_current
            # PnP result is T_target_from_current
            R_pnp, _ = cv2.Rodrigues(rvec)
            T_pnp = np.eye(4)
            T_pnp[:3, :3] = R_pnp
            T_pnp[:3, 3] = tvec.flatten()

            # Convert to se(3) via matrix logarithm
            from pose_refine.utils.lie_algebra import se3_log
            T_tensor = torch.from_numpy(T_pnp).float().unsqueeze(0).to(device)
            xi = se3_log(T_tensor).squeeze(0)  # (6,)
            results.append(xi)

        except Exception:
            results.append(torch.zeros(6, device=device))

    return torch.stack(results, dim=0)  # (B, 6)


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
    robust_kernel: str = 'huber',
    gnc_mu_init: float = 1.0,
    gnc_mu_step: float = 1.4,
) -> torch.Tensor:
    """
    可微分的加权最小二乘位姿求解器, 支持 IRLS 鲁棒估计和方向性置信度.

    给定预测的 dense flow 和 Image Jacobian, 求解 6-DOF 位姿增量.
    可选 IRLS: 初始 WLS 求解后, 基于残差用 Huber 权重迭代 re-weight,
    抑制 flow 预测中的离群像素 (遮挡/误匹配).

    Robust kernels:
      - 'huber': Huber loss — linear penalty beyond threshold (default, IRLS classic)
      - 'gm': Geman-McClure — smooth redescending kernel, fully suppresses gross outliers
      - 'gnc_gm': Graduated Non-Convexity with GM kernel — starts convex (L2),
        gradually transitions to GM over IRLS iterations for global convergence

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
        robust_kernel: 'huber' | 'gm' | 'gnc_gm' — which robust kernel to use in IRLS
        gnc_mu_init: GNC initial mu (1.0 = start from L2)
        gnc_mu_step: GNC mu growth factor per iteration (>1, typically 1.4)

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
    for irls_i in range(irls_iters):
        # Compute residuals: r_u = J_u @ xi - flow_u, r_v = J_v @ xi - flow_v
        pred_u = torch.bmm(Ju, delta_xi.unsqueeze(-1)).squeeze(-1)  # (B, N)
        pred_v = torch.bmm(Jv, delta_xi.unsqueeze(-1)).squeeze(-1)
        res = torch.sqrt((pred_u - flow_u)**2 + (pred_v - flow_v)**2 + 1e-8)

        # Adaptive threshold: median(res) * irls_huber_k
        res_valid = res + (1.0 - valid.float()) * 1e6  # push invalid to high
        median_res = res_valid.median(dim=1, keepdim=True).values
        threshold = (median_res * irls_huber_k).clamp(min=0.1)

        if robust_kernel == 'gm':
            # Geman-McClure: w = c² / (c² + r²)²  (redescending: fully suppresses gross outliers)
            c2 = threshold ** 2
            r2 = res ** 2
            robust_w = c2 / (c2 + r2).clamp(min=1e-8)
        elif robust_kernel == 'gnc_gm':
            # Graduated Non-Convexity with GM kernel
            # mu starts at gnc_mu_init (convex, ~L2) and grows toward GM
            mu = gnc_mu_init * (gnc_mu_step ** irls_i)
            c2 = threshold ** 2
            r2 = res ** 2
            # GNC-GM weight: interpolates between L2 (mu=0) and GM (mu→∞)
            # w = (mu * c2) / (mu * c2 + r2)^2  → approaches GM as mu grows
            mc2 = mu * c2
            robust_w = mc2 / (mc2 + r2).clamp(min=1e-8)
        else:
            # Huber weight: w = 1 if |r| < k, else k/|r|
            robust_w = torch.where(res < threshold, torch.ones_like(res),
                                   threshold / res.clamp(min=1e-6))

        # Combined weights: network confidence × robust kernel × validity
        w_irls_u = w_base_u * robust_w
        w_irls_v = w_base_v * robust_w

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


def feature_metric_solve(
    query_feat: torch.Tensor,
    ref_feat: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: Dict[str, float],
    damping: float = 1e-3,
    valid_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feature-metric direct pose solver (Inverse Compositional).

    Instead of predicting flow via correlation, directly solves for the
    camera motion that minimizes the feature residual between query and
    reference. Works even with low feature similarity (cos_sim ≈ 0.3-0.4)
    because it only needs LOCALLY consistent features, not precise correspondences.

    Derivation:
      Feature residual: r(ξ) = F_query(p) - F_ref(warp(p, ξ))
      Linearize: r(ξ) ≈ r(0) + J_feat * ξ
      where J_feat[c, :] = (dF_c/du) * Ju + (dF_c/dv) * Jv
      WLS: ξ = (J^T J + λI)^{-1} J^T r

    Args:
        query_feat: (B, C, H, W) query features (RADIO)
        ref_feat:   (B, C, H, W) rendered reference features (DCFF)
        depth:      (B, H, W) depth at reference viewpoint
        intrinsics: {fx, fy, cx, cy}
        damping:    LM damping coefficient
        valid_mask: (B, 1, H, W) optional validity mask

    Returns:
        delta_xi:   (B, 6) se(3) update vector
        residual:   (B, C, H, W) feature residual (for monitoring)
    """
    B, C, H, W = query_feat.shape
    N = H * W
    device = query_feat.device

    # 1. Feature residual
    residual = query_feat - ref_feat  # (B, C, H, W)

    # 2. Spatial gradients of reference features (Sobel-like)
    # Using central differences for smoothness
    pad_ref = F.pad(ref_feat, (1, 1, 1, 1), mode='replicate')
    grad_u = (pad_ref[:, :, 1:-1, 2:] - pad_ref[:, :, 1:-1, :-2]) / 2.0  # (B,C,H,W)
    grad_v = (pad_ref[:, :, 2:, 1:-1] - pad_ref[:, :, :-2, 1:-1]) / 2.0

    # 3. Image Jacobian
    Ju, Jv, depth_valid = compute_image_jacobian(depth, intrinsics)  # (B,N,6)

    # 4. Build feature Jacobian: J_feat[b, n*c, :] = grad_u[b,c,n] * Ju[b,n,:] + grad_v[b,c,n] * Jv[b,n,:]
    # Instead of materializing the full (B, N*C, 6) matrix, compute J^T J and J^T r directly
    grad_u_flat = grad_u.reshape(B, C, N)  # (B, C, N)
    grad_v_flat = grad_v.reshape(B, C, N)
    r_flat = residual.reshape(B, C, N)     # (B, C, N)

    # Validity mask
    if valid_mask is not None:
        vmask = (valid_mask.reshape(B, 1, N) * depth_valid.float().unsqueeze(1))  # (B, 1, N)
    else:
        vmask = depth_valid.float().unsqueeze(1)  # (B, 1, N)

    # J^T J = sum_c sum_n [grad_u_c * Ju_n + grad_v_c * Jv_n]^T [grad_u_c * Ju_n + grad_v_c * Jv_n]
    # J^T r = sum_c sum_n [grad_u_c * Ju_n + grad_v_c * Jv_n]^T r_c_n
    # Efficient: accumulate over channels
    JtJ = torch.zeros(B, 6, 6, device=device)
    Jtr = torch.zeros(B, 6, device=device)

    for c in range(C):
        gu = grad_u_flat[:, c, :] * vmask[:, 0, :]  # (B, N)
        gv = grad_v_flat[:, c, :] * vmask[:, 0, :]
        rc = r_flat[:, c, :] * vmask[:, 0, :]

        # J_c = gu[:,:,None] * Ju + gv[:,:,None] * Jv  -> (B, N, 6)
        J_c = gu.unsqueeze(-1) * Ju + gv.unsqueeze(-1) * Jv

        JtJ += torch.bmm(J_c.transpose(1, 2), J_c)
        Jtr += torch.bmm(J_c.transpose(1, 2), rc.unsqueeze(-1)).squeeze(-1)

    # Damping
    diag = JtJ.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6)
    JtJ = JtJ + damping * torch.diag_embed(diag)

    # Solve
    delta_xi = torch.linalg.solve(JtJ, Jtr)

    # Safety clamp
    trans_limit, rot_limit = 2.0, 1.5708
    delta_trans = torch.tanh(delta_xi[:, :3] / trans_limit) * trans_limit
    delta_rot = torch.tanh(delta_xi[:, 3:] / rot_limit) * rot_limit
    delta_xi = torch.cat([delta_trans, delta_rot], dim=1)

    return delta_xi, residual


def diff_pose_solve_sequential(
    flow: torch.Tensor,
    confidence: torch.Tensor,
    Ju: torch.Tensor,
    Jv: torch.Tensor,
    valid: torch.Tensor,
    damping: float = 1e-3,
) -> torch.Tensor:
    """
    Sequential rotation-first then translation solve.

    For outdoor scenes with large depths, translation and rotation Jacobians
    differ by orders of magnitude (rotation Jacobian ∝ fx, translation ∝ fx/Z).
    A joint 6-DOF solve is ill-conditioned and rotation noise leaks into
    translation estimates.

    This function:
      1. Solves rotation (3-DOF) from the full flow
      2. Computes residual flow after removing the rotation component
      3. Solves translation (3-DOF) from the residual

    Args:
        flow: (B, 2, H, W) predicted flow
        confidence: (B, 1, H, W) per-pixel weights
        Ju, Jv: (B, N, 6) Image Jacobian [tx,ty,tz,wx,wy,wz]
        valid: (B, N) valid pixel mask
        damping: LM damping coefficient

    Returns:
        delta_xi: (B, 6) se(3) update [tx,ty,tz,wx,wy,wz]
    """
    B = flow.shape[0]
    device = flow.device

    flow_u = flow[:, 0].reshape(B, -1)
    flow_v = flow[:, 1].reshape(B, -1)

    w = confidence[:, 0].reshape(B, -1) * valid.float()
    w_unsq = w.unsqueeze(-1)

    # Split Jacobian into rotation (cols 3:6) and translation (cols 0:3)
    Ju_rot = Ju[:, :, 3:6]   # (B, N, 3)
    Jv_rot = Jv[:, :, 3:6]
    Ju_trans = Ju[:, :, 0:3]
    Jv_trans = Jv[:, :, 0:3]

    # ── Step 1: Solve rotation only ──
    wJu_rot = Ju_rot * w_unsq
    wJv_rot = Jv_rot * w_unsq
    JtWJ_rot = torch.bmm(Ju_rot.transpose(1, 2), wJu_rot) + \
               torch.bmm(Jv_rot.transpose(1, 2), wJv_rot)   # (B, 3, 3)
    diag_rot = JtWJ_rot.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6)
    JtWJ_rot = JtWJ_rot + damping * torch.diag_embed(diag_rot)

    JtWr_rot = torch.bmm(
        Ju_rot.transpose(1, 2), (w * flow_u).unsqueeze(-1)
    ).squeeze(-1) + torch.bmm(
        Jv_rot.transpose(1, 2), (w * flow_v).unsqueeze(-1)
    ).squeeze(-1)   # (B, 3)

    omega = torch.linalg.solve(JtWJ_rot, JtWr_rot)  # (B, 3)

    # ── Step 2: Compute residual flow after rotation ──
    rot_flow_u = torch.bmm(Ju_rot, omega.unsqueeze(-1)).squeeze(-1)
    rot_flow_v = torch.bmm(Jv_rot, omega.unsqueeze(-1)).squeeze(-1)
    resid_u = flow_u - rot_flow_u
    resid_v = flow_v - rot_flow_v

    # ── Step 3: Solve translation from residual ──
    wJu_trans = Ju_trans * w_unsq
    wJv_trans = Jv_trans * w_unsq
    JtWJ_trans = torch.bmm(Ju_trans.transpose(1, 2), wJu_trans) + \
                 torch.bmm(Jv_trans.transpose(1, 2), wJv_trans)
    diag_trans = JtWJ_trans.diagonal(dim1=-2, dim2=-1).clamp(min=1e-6)
    JtWJ_trans = JtWJ_trans + damping * torch.diag_embed(diag_trans)

    JtWr_trans = torch.bmm(
        Ju_trans.transpose(1, 2), (w * resid_u).unsqueeze(-1)
    ).squeeze(-1) + torch.bmm(
        Jv_trans.transpose(1, 2), (w * resid_v).unsqueeze(-1)
    ).squeeze(-1)

    v = torch.linalg.solve(JtWJ_trans, JtWr_trans)  # (B, 3)

    # Combine: xi = [v, omega]
    delta_xi = torch.cat([v, omega], dim=1)

    # Soft safety clamp
    trans_limit, rot_limit = 2.0, 1.5708
    delta_trans = torch.tanh(delta_xi[:, :3] / trans_limit) * trans_limit
    delta_rot = torch.tanh(delta_xi[:, 3:] / rot_limit) * rot_limit
    delta_xi = torch.cat([delta_trans, delta_rot], dim=1)

    return delta_xi
