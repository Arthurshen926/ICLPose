"""
Depth-based backward warping for feature rendering.

Instead of rendering features from 3DGS (which smooths spatial information),
this module warps stored query features from the GT pose to the estimated pose
using 3DGS-rendered depth. This preserves the full spatial distinctiveness of
stored features while using 3DGS only for geometry.

Usage in training:
    depth = renderer.render_depth_only(pose_est)
    warped = backward_warp_features(query_feats, depth, pose_gt, pose_est, scale_intrinsics)
"""

import torch
import torch.nn.functional as F


def backward_warp_features(
    query_feats: dict,
    depth: torch.Tensor,
    pose_gt: torch.Tensor,
    pose_est: torch.Tensor,
    scale_intrinsics: dict,
) -> dict:
    """
    Backward-warp query features from GT pose to estimated pose using depth.

    For each pixel in the estimated view:
      1. Unproject to 3D using depth at estimated pose
      2. Transform to GT camera frame
      3. Project to GT image coordinates
      4. Sample query features at those coordinates

    Args:
        query_feats: {scale_name: (B, C, H, W)} features at GT pose
        depth: (B, H_d, W_d) depth map rendered at estimated pose
        pose_gt: (B, 4, 4) GT world-to-camera matrix
        pose_est: (B, 4, 4) estimated world-to-camera matrix
        scale_intrinsics: {scale_name: {'fx', 'fy', 'cx', 'cy'}}

    Returns:
        warped_feats: {scale_name: (B, C, H, W)} warped features at estimated pose
    """
    B = pose_gt.shape[0]
    device = pose_gt.device

    # Relative transform: maps points from estimated camera to GT camera
    # X_gt = T_gt @ T_est^{-1} @ X_est
    try:
        T_est_inv = torch.linalg.inv(pose_est)
    except RuntimeError:
        T_est_inv = torch.linalg.pinv(pose_est)
    T_rel = pose_gt @ T_est_inv  # (B, 4, 4)

    warped = {}
    for scale, feats in query_feats.items():
        if scale not in scale_intrinsics:
            warped[scale] = feats  # passthrough if no intrinsics
            continue

        _, C, H, W = feats.shape
        K = scale_intrinsics[scale]
        fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']

        # Resize depth to this scale's resolution
        d = F.interpolate(
            depth.unsqueeze(1), size=(H, W), mode='nearest'
        ).squeeze(1)  # (B, H, W)

        # Replace NaN/Inf depth with 0 (will be masked out)
        d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))

        # Create pixel grid for estimated view
        u = torch.arange(W, device=device, dtype=torch.float32)
        v = torch.arange(H, device=device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing='ij')  # (H, W)

        # Unproject to 3D in estimated camera frame
        x_est = (uu.unsqueeze(0) - cx) / fx * d  # (B, H, W)
        y_est = (vv.unsqueeze(0) - cy) / fy * d
        z_est = d
        ones = torch.ones_like(z_est)

        # (B, 4, H*W)
        pts = torch.stack([x_est, y_est, z_est, ones], dim=1).reshape(B, 4, H * W)

        # Transform to GT camera frame
        pts_gt = T_rel @ pts  # (B, 4, H*W)

        # Project to GT image
        Z_gt = pts_gt[:, 2].clamp(min=1e-6)
        u_gt = fx * pts_gt[:, 0] / Z_gt + cx  # (B, H*W)
        v_gt = fy * pts_gt[:, 1] / Z_gt + cy

        # Normalize to [-1, 1] for grid_sample
        u_norm = 2.0 * u_gt / max(W - 1, 1) - 1.0
        v_norm = 2.0 * v_gt / max(H - 1, 1) - 1.0

        grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B, H, W, 2)
        # Replace any NaN in grid with 0 (maps to padding_mode='zeros')
        grid = torch.where(torch.isfinite(grid), grid, torch.zeros_like(grid))

        # Mask invalid pixels (depth <= 0 or behind camera)
        valid = (d > 0) & (Z_gt.reshape(B, H, W) > 0)

        warped_feat = F.grid_sample(
            feats, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )
        # Zero out invalid pixels
        warped_feat = warped_feat * valid.unsqueeze(1).float()
        # Final NaN guard
        warped_feat = torch.where(torch.isfinite(warped_feat), warped_feat, torch.zeros_like(warped_feat))

        warped[scale] = warped_feat

    return warped
