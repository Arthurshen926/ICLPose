"""
Direct pose refinement through differentiable rasterizer for GSFFs.

Implements Eq. 4 from the paper:
  P* = argmin_{P ∈ SE(3)} ||F2D - F3D(P, G)||²₂

Updates are on the Lie algebra se(3) by backpropagating through the
gsplat rasterizer.
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from se(3) to SE(3).
    
    Args:
        xi: [6] or [B, 6] se(3) vector [tx, ty, tz, rx, ry, rz]
        
    Returns:
        T: [4, 4] or [B, 4, 4] transformation matrix
    """
    if xi.dim() == 1:
        xi = xi.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    B = xi.shape[0]
    t = xi[:, :3]  # [B, 3]
    omega = xi[:, 3:]  # [B, 3]
    
    theta = torch.norm(omega, dim=1, keepdim=True)  # [B, 1]
    theta = theta.clamp(min=1e-8)
    
    # Skew-symmetric matrix
    K = _skew(omega / theta)  # [B, 3, 3]
    
    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(B, -1, -1)
    sin_t = torch.sin(theta).unsqueeze(-1)  # [B, 1, 1]
    cos_t = torch.cos(theta).unsqueeze(-1)
    
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)  # [B, 3, 3]
    
    # V matrix for translation
    V = eye + ((1.0 - cos_t) / (theta.unsqueeze(-1) ** 2 + 1e-10)) * K + \
        ((theta.unsqueeze(-1) - sin_t) / (theta.unsqueeze(-1) ** 3 + 1e-10)) * (K @ K)
    
    Vt = (V @ t.unsqueeze(-1)).squeeze(-1)  # [B, 3]
    
    T = torch.eye(4, device=xi.device, dtype=xi.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    T[:, :3, :3] = R
    T[:, :3, 3] = Vt
    
    if squeeze:
        T = T.squeeze(0)
    return T


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix from [B, 3] vector."""
    B = v.shape[0]
    zero = torch.zeros(B, device=v.device, dtype=v.dtype)
    K = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero
    ], dim=1).reshape(B, 3, 3)
    return K


def render_features_for_pose(
    means3d: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    chunk_size: int = 16,
) -> torch.Tensor:
    """
    Render feature map from current pose using 2DGS rasterizer (non-differentiable viewmat).
    For pose refinement with gradient support, use render_features_transformed().
    """
    D = colors.shape[1]
    n_chunks = (D + chunk_size - 1) // chunk_size
    
    chunks = []
    for i in range(n_chunks):
        c_start = i * chunk_size
        c_end = min((i + 1) * chunk_size, D)
        
        render_colors, render_alphas, *_ = rasterization_2dgs(
            means=means3d,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors[:, c_start:c_end],
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e5,
            render_mode='RGB',
        )
        chunks.append(render_colors)
    
    feature_map = torch.cat(chunks, dim=-1).permute(0, 3, 1, 2)
    return feature_map


def _rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert [3, 3] rotation matrix to [4] quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return torch.stack([w, x, y, z])


def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two quaternions.
    q1, q2: [N, 4] (w, x, y, z)
    Returns: [N, 4]
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=1)


def _transform_gaussians(means3d, quats, delta_T, fixed_viewmat):
    """Transform Gaussians by delta_T to implement differentiable pose change.
    
    Returns:
        means_transformed: [N, 3]
        quats_transformed: [N, 4]
    """
    fixed_inv = torch.inverse(fixed_viewmat)
    delta_T_world = fixed_inv @ delta_T @ fixed_viewmat
    
    R_dw = delta_T_world[:3, :3]
    t_dw = delta_T_world[:3, 3]
    means_transformed = means3d @ R_dw.T + t_dw.unsqueeze(0)
    
    q_delta = _rotation_matrix_to_quaternion(R_dw)
    q_delta_expanded = q_delta.unsqueeze(0).expand(quats.shape[0], -1)
    quats_transformed = _quaternion_multiply(q_delta_expanded, quats)
    quats_transformed = quats_transformed / (quats_transformed.norm(dim=1, keepdim=True) + 1e-8)
    
    return means_transformed, quats_transformed


def _render_with_transformed_gaussians(
    means_transformed, quats_transformed, scales, opacities,
    colors, fixed_viewmat, K, width, height, chunk_size=16,
):
    """Render an arbitrary channel-count from transformed Gaussians."""
    D = colors.shape[1]
    n_chunks = (D + chunk_size - 1) // chunk_size
    fixed_vm = fixed_viewmat.unsqueeze(0)
    
    chunks = []
    for i in range(n_chunks):
        c_start = i * chunk_size
        c_end = min((i + 1) * chunk_size, D)
        
        render_colors, render_alphas, *_ = rasterization_2dgs(
            means=means_transformed,
            quats=quats_transformed,
            scales=scales,
            opacities=opacities,
            colors=colors[:, c_start:c_end],
            viewmats=fixed_vm,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e5,
            render_mode='RGB',
        )
        chunks.append(render_colors)
    
    feature_map = torch.cat(chunks, dim=-1).permute(0, 3, 1, 2)
    return feature_map


def render_features_transformed(
    means3d: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    delta_T: torch.Tensor,
    fixed_viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    chunk_size: int = 16,
) -> torch.Tensor:
    """
    Render features using 2DGS with differentiable Gaussian transformation.
    """
    means_transformed, quats_transformed = _transform_gaussians(
        means3d, quats, delta_T, fixed_viewmat)
    return _render_with_transformed_gaussians(
        means_transformed, quats_transformed, scales, opacities,
        colors, fixed_viewmat, K, width, height, chunk_size)


def refine_pose(
    feat_2d: torch.Tensor,
    means3d: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    init_viewmat: torch.Tensor,
    K_mat: torch.Tensor,
    width: int,
    height: int,
    n_iters: int = 100,
    lr: float = 0.01,
    chunk_size: int = 16,
    lr_decay: float = 1.0,
    loss_type: str = 'mse',
    reset_interval: int = 0,
    tune_features: bool = False,
    feature_lr: float = 0.001,
    trans_lr_scale: float = 1.0,
) -> torch.Tensor:
    """
    Iterative pose refinement via differentiable Gaussian transformation.
    
    Uses 2DGS rasterizer (same as training) with gradients flowing through
    the Gaussian mean/quaternion transformation instead of viewmat.
    
    P* = argmin_{ΔP} ||F2D - F3D(ΔP · P_init, G)||²₂
    
    Args:
        reset_interval: if > 0, fold delta_xi into viewmat every N steps
                       to keep optimization in the tangent space
        tune_features: if True, jointly optimize per-Gaussian feature residuals
                      (paper's "Feature tuned" variant)
        feature_lr: learning rate for feature residual optimization
        trans_lr_scale: multiplier for translation LR relative to rotation LR
    """
    device = init_viewmat.device
    
    # Split translation and rotation for separate learning rates
    use_split_lr = (trans_lr_scale != 1.0)
    if use_split_lr:
        delta_t = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
        delta_r = torch.zeros(3, device=device, dtype=torch.float32, requires_grad=True)
    else:
        delta_xi = torch.zeros(6, device=device, dtype=torch.float32, requires_grad=True)
    
    means3d = means3d.detach()
    quats = quats.detach()
    scales = scales.detach()
    opacities = opacities.detach()
    colors_base = colors.detach()
    feat_2d = feat_2d.detach()
    init_viewmat = init_viewmat.detach().clone()
    K_unsq = K_mat.unsqueeze(0).detach()
    
    # Optional: jointly optimize per-Gaussian feature residuals
    if use_split_lr:
        param_groups = [
            {'params': [delta_t], 'lr': lr * trans_lr_scale},
            {'params': [delta_r], 'lr': lr},
        ]
    else:
        param_groups = [{'params': [delta_xi], 'lr': lr}]
    if tune_features:
        feat_residual = torch.zeros_like(colors_base, requires_grad=True)
        param_groups.append({'params': [feat_residual], 'lr': feature_lr})
    
    optimizer = torch.optim.Adam(param_groups)
    
    if lr_decay < 1.0:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    else:
        scheduler = None
    
    best_loss = float('inf')
    best_viewmat = init_viewmat.clone()
    
    for i in range(n_iters):
        optimizer.zero_grad()
        
        # Build delta_xi from split or unified parameters
        if use_split_lr:
            delta_xi_val = torch.cat([delta_t, delta_r])
        else:
            delta_xi_val = delta_xi
        
        # Periodically fold delta into viewmat to keep updates small
        if reset_interval > 0 and i > 0 and i % reset_interval == 0:
            with torch.no_grad():
                init_viewmat = se3_exp(delta_xi_val) @ init_viewmat
                if use_split_lr:
                    delta_t.zero_()
                    delta_r.zero_()
                else:
                    delta_xi.zero_()
        
        delta_T = se3_exp(delta_xi_val)
        
        colors = colors_base
        if tune_features:
            colors = F.normalize(colors_base + feat_residual, p=2, dim=1)
        
        feat_3d = render_features_transformed(
            means3d, quats, scales, opacities, colors,
            delta_T, init_viewmat, K_unsq, width, height, chunk_size,
        )
        
        feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
        if loss_type == 'huber':
            loss = F.smooth_l1_loss(feat_3d_norm, feat_2d)
        else:
            loss = F.mse_loss(feat_3d_norm, feat_2d)
        
        loss.backward()
        if use_split_lr:
            torch.nn.utils.clip_grad_norm_([delta_t, delta_r], 0.5)
        else:
            torch.nn.utils.clip_grad_norm_([delta_xi], 0.5)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            final_viewmat = (se3_exp(delta_xi_val.detach()) @ init_viewmat)
            best_viewmat = final_viewmat.clone()
    
    return best_viewmat


def refine_pose_with_rgb(
    feat_2d: torch.Tensor,
    rgb_target: torch.Tensor,
    means3d: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors_feat: torch.Tensor,
    colors_rgb: torch.Tensor,
    init_viewmat: torch.Tensor,
    K_mat: torch.Tensor,
    width: int,
    height: int,
    n_iters: int = 300,
    lr: float = 0.005,
    rgb_weight: float = 0.1,
    chunk_size: int = 16,
    loss_type: str = 'mse',
) -> torch.Tensor:
    """
    Fine refinement with combined feature + photometric RGB loss.
    
    feature loss provides wide convergence basin,
    RGB loss provides precise gradient signal near GT.
    
    Args:
        feat_2d: [1, D, H, W] L2-normalized 2D encoder features
        rgb_target: [1, 3, H, W] target RGB image (in [0,1])
        colors_feat: [N, D] triplane features (L2-normalized)
        colors_rgb: [N, 3] Gaussian RGB colors (from SH evaluation at current viewpoint)
        rgb_weight: weight for RGB loss (0.1 = 10% of total)
    """
    device = init_viewmat.device
    
    delta_xi = torch.zeros(6, device=device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([delta_xi], lr=lr)
    
    means3d = means3d.detach()
    quats = quats.detach()
    scales = scales.detach()
    opacities = opacities.detach()
    colors_feat = colors_feat.detach()
    colors_rgb = colors_rgb.detach()
    feat_2d = feat_2d.detach()
    rgb_target = rgb_target.detach()
    init_viewmat = init_viewmat.detach().clone()
    K_unsq = K_mat.unsqueeze(0).detach()
    
    best_loss = float('inf')
    best_viewmat = init_viewmat.clone()
    
    for i in range(n_iters):
        optimizer.zero_grad()
        
        delta_T = se3_exp(delta_xi)
        means_t, quats_t = _transform_gaussians(
            means3d, quats, delta_T, init_viewmat)
        
        # Feature loss
        feat_3d = _render_with_transformed_gaussians(
            means_t, quats_t, scales, opacities,
            colors_feat, init_viewmat, K_unsq, width, height, chunk_size)
        feat_3d_norm = F.normalize(feat_3d, p=2, dim=1)
        
        if loss_type == 'huber':
            feat_loss = F.smooth_l1_loss(feat_3d_norm, feat_2d)
        else:
            feat_loss = F.mse_loss(feat_3d_norm, feat_2d)
        
        # RGB photometric loss  
        rgb_3d = _render_with_transformed_gaussians(
            means_t, quats_t, scales, opacities,
            colors_rgb, init_viewmat, K_unsq, width, height, chunk_size=3)
        rgb_loss = F.mse_loss(rgb_3d, rgb_target)
        
        # Progressive RGB weight: increase as features converge
        progress = min(i / max(n_iters * 0.5, 1), 1.0)
        current_rgb_weight = rgb_weight * progress
        
        loss = feat_loss + current_rgb_weight * rgb_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_xi], 0.5)
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            final_viewmat = (se3_exp(delta_xi.detach()) @ init_viewmat)
            best_viewmat = final_viewmat.clone()
    
    return best_viewmat


def refine_pose_hierarchical(
    image: torch.Tensor,
    encoder,
    triplane,
    gaussian_params: dict,
    init_viewmat: torch.Tensor,
    K_mat: torch.Tensor,
    coarse_size: tuple,
    fine_size: tuple,
    coarse_iters: int = 100,
    fine_iters: int = 100,
    coarse_lr: float = 0.01,
    fine_lr: float = 0.005,
) -> torch.Tensor:
    """
    Hierarchical coarse-to-fine pose refinement.
    
    1. Coarse refinement with ViT-level features (large convergence basin)
    2. Fine refinement with pixel-level features (high precision)
    
    Args:
        image: [1, 3, H, W] query image (normalized)
        encoder: DualScaleEncoder
        triplane: DualScaleTriplane
        gaussian_params: dict with keys 'means3d', 'quats', 'scales', 'opacities'
        init_viewmat: [4, 4] initial w2c pose
        K_mat: [3, 3] camera intrinsic matrix
        coarse_size: (H_coarse, W_coarse) for coarse level
        fine_size: (H_fine, W_fine) for fine level
        coarse_iters, fine_iters: iterations per stage
        coarse_lr, fine_lr: learning rates
        
    Returns:
        refined_viewmat: [4, 4]
    """
    device = init_viewmat.device
    means3d = gaussian_params['means3d']
    quats = gaussian_params['quats']
    scales = gaussian_params['scales']
    opacities = gaussian_params['opacities']
    
    with torch.no_grad():
        coarse_feat_2d, fine_feat_2d = encoder(image)
        coarse_feat_2d = F.normalize(coarse_feat_2d, p=2, dim=1)
        fine_feat_2d = F.normalize(fine_feat_2d, p=2, dim=1)
    
    # Scale intrinsics for coarse level
    H_orig, W_orig = image.shape[2], image.shape[3]
    H_c, W_c = coarse_size
    H_f, W_f = fine_size
    
    K_coarse = K_mat.clone()
    K_coarse[0] *= W_c / W_orig
    K_coarse[1] *= H_c / H_orig
    
    K_fine = K_mat.clone()
    K_fine[0] *= W_f / W_orig
    K_fine[1] *= H_f / H_orig
    
    # Resize 2D features to match render resolution
    coarse_feat_2d = F.interpolate(coarse_feat_2d, (H_c, W_c), mode='bilinear', align_corners=False)
    fine_feat_2d = F.interpolate(fine_feat_2d, (H_f, W_f), mode='bilinear', align_corners=False)
    
    # Stage 1: Coarse refinement
    with torch.no_grad():
        coarse_colors = triplane.extract_coarse(means3d)
        coarse_colors = F.normalize(coarse_colors, p=2, dim=1)
    
    viewmat = refine_pose(
        coarse_feat_2d, means3d, quats, scales, opacities,
        coarse_colors, init_viewmat, K_coarse, W_c, H_c,
        n_iters=coarse_iters, lr=coarse_lr, chunk_size=16,
    )
    
    # Stage 2: Fine refinement
    with torch.no_grad():
        fine_colors = triplane.extract_fine(means3d)
        fine_colors = F.normalize(fine_colors, p=2, dim=1)
    
    viewmat = refine_pose(
        fine_feat_2d, means3d, quats, scales, opacities,
        fine_colors, viewmat, K_fine, W_f, H_f,
        n_iters=fine_iters, lr=fine_lr, chunk_size=16,
    )
    
    return viewmat
