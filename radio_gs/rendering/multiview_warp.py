"""Depth-based multi-view feature warping utilities for RADIO-GS.

Used by:
- FeatSharp-3D: warp features from source views to reference for consistency checking
- Multi-view consistency loss: ensure rendered features agree across viewpoints
- Data augmentation: create pseudo ground-truth from nearby views
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def create_pixel_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create pixel coordinate grid.

    Returns:
        grid: [1, 2, H, W] with (u, v) pixel coordinates where u=column, v=row.
    """
    v, u = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    # Stack as (u, v) and add batch dim
    return torch.stack([u, v], dim=0).unsqueeze(0)  # [1, 2, H, W]


def unproject_pixels(
    depth: torch.Tensor,
    K: torch.Tensor,
    viewmat: torch.Tensor,
    return_world: bool = True,
) -> torch.Tensor:
    """Unproject pixels to 3D points.

    Args:
        depth: [B, H, W] per-pixel depth (camera-space z).
        K: [3, 3] camera intrinsics.
        viewmat: [B, 4, 4] world-to-camera transform.
        return_world: if True return world coords, else camera coords.

    Returns:
        points: [B, 3, H, W] 3D coordinates.
    """
    B, H, W = depth.shape
    device = depth.device

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    grid = create_pixel_grid(H, W, device)  # [1, 2, H, W]
    u = grid[:, 0]  # [1, H, W]
    v = grid[:, 1]  # [1, H, W]

    # Unproject to camera coordinates: x = (u - cx) / fx * z, etc.
    z = depth  # [B, H, W]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    points_cam = torch.stack([x, y, z], dim=1)  # [B, 3, H, W]

    if not return_world:
        return points_cam

    # Camera → world: P_world = R^T @ (P_cam - t) = inv(viewmat) @ P_cam_h
    R = viewmat[:, :3, :3]  # [B, 3, 3]
    t = viewmat[:, :3, 3]   # [B, 3]

    # inv(viewmat) rotation is R^T, translation is -R^T @ t
    R_inv = R.transpose(1, 2)  # [B, 3, 3]
    t_inv = -torch.bmm(R_inv, t.unsqueeze(-1)).squeeze(-1)  # [B, 3]

    pts_flat = points_cam.reshape(B, 3, -1)  # [B, 3, H*W]
    world_flat = torch.bmm(R_inv, pts_flat) + t_inv.unsqueeze(-1)  # [B, 3, H*W]
    return world_flat.reshape(B, 3, H, W)


def compute_view_directions(
    depth: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Compute per-pixel view directions in world space (from camera origin to point).

    Args:
        depth: [B, H, W] per-pixel depth.
        viewmat: [B, 4, 4] world-to-camera transform.
        K: [3, 3] intrinsics.

    Returns:
        directions: [B, 3, H, W] unit view-direction vectors in world space.
    """
    points_world = unproject_pixels(depth, K, viewmat, return_world=True)  # [B, 3, H, W]

    # Camera origin in world coords: -R^T @ t
    R = viewmat[:, :3, :3]
    t = viewmat[:, :3, 3]
    cam_origin = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1))  # [B, 3, 1]

    dirs = points_world - cam_origin.unsqueeze(-1)  # [B, 3, H, W]
    dirs = F.normalize(dirs, dim=1, eps=1e-8)
    return dirs


def warp_features(
    feat_src: torch.Tensor,
    depth_ref: torch.Tensor,
    viewmat_ref: torch.Tensor,
    viewmat_src: torch.Tensor,
    K: torch.Tensor,
    padding_mode: str = "zeros",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp source features to reference view using depth-based reprojection.

    Args:
        feat_src: [B, C, H, W] source view features.
        depth_ref: [B, H, W] reference view depth.
        viewmat_ref: [B, 4, 4] reference world-to-camera.
        viewmat_src: [B, 4, 4] source world-to-camera.
        K: [3, 3] intrinsics.
        padding_mode: padding for grid_sample ('zeros', 'border', 'reflection').

    Returns:
        warped_feat: [B, C, H, W] source features warped to reference frame.
        valid_mask: [B, 1, H, W] binary mask (1 where warp is valid).
    """
    B, C, H, W = feat_src.shape
    device = feat_src.device

    # 1. Unproject reference pixels to world coordinates
    points_world = unproject_pixels(depth_ref, K, viewmat_ref, return_world=True)  # [B, 3, H, W]

    # 2. Project world points into source camera
    R_src = viewmat_src[:, :3, :3]  # [B, 3, 3]
    t_src = viewmat_src[:, :3, 3]   # [B, 3]

    pts_flat = points_world.reshape(B, 3, -1)  # [B, 3, H*W]
    pts_src = torch.bmm(R_src, pts_flat) + t_src.unsqueeze(-1)  # [B, 3, H*W]

    # 3. Perspective projection using intrinsics
    z_src = pts_src[:, 2:3, :]  # [B, 1, H*W]
    # Avoid division by zero / negative depth
    z_safe = z_src.clamp(min=1e-6)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u_src = fx * pts_src[:, 0:1, :] / z_safe + cx  # [B, 1, H*W]
    v_src = fy * pts_src[:, 1:2, :] / z_safe + cy  # [B, 1, H*W]

    # 4. Normalize to [-1, 1] for grid_sample (align_corners=True)
    u_norm = 2.0 * u_src / (W - 1) - 1.0
    v_norm = 2.0 * v_src / (H - 1) - 1.0

    grid = torch.cat([u_norm, v_norm], dim=1)  # [B, 2, H*W]
    grid = grid.reshape(B, 2, H, W).permute(0, 2, 3, 1)  # [B, H, W, 2]

    # 5. Sample source features
    warped_feat = F.grid_sample(
        feat_src,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )

    # 6. Validity mask: valid depth, positive z in source, within source image bounds
    valid_depth = (depth_ref > 0) & torch.isfinite(depth_ref)  # [B, H, W]
    valid_z = (z_src > 1e-6).reshape(B, H, W)
    valid_bounds = (
        (u_src >= 0).reshape(B, H, W)
        & (u_src <= W - 1).reshape(B, H, W)
        & (v_src >= 0).reshape(B, H, W)
        & (v_src <= H - 1).reshape(B, H, W)
    )
    valid_mask = (valid_depth & valid_z & valid_bounds).unsqueeze(1).float()  # [B, 1, H, W]

    return warped_feat, valid_mask


def compute_consistency_map(
    feat_ref: torch.Tensor,
    feat_sources: List[torch.Tensor],
    depth_ref: torch.Tensor,
    viewmat_ref: torch.Tensor,
    viewmats_src: List[torch.Tensor],
    K: torch.Tensor,
    metric: str = "cosine",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-pixel multi-view feature consistency.

    For each source view, warps source features to the reference frame and
    computes per-pixel similarity with the reference features.

    Args:
        feat_ref: [B, C, H, W] reference features.
        feat_sources: list of [B, C, H, W] source features.
        depth_ref: [B, H, W] reference depth.
        viewmat_ref: [B, 4, 4] reference world-to-camera.
        viewmats_src: list of [B, 4, 4] source world-to-camera matrices.
        K: [3, 3] intrinsics.
        metric: 'cosine' for cosine similarity or 'l2' for negative L2 distance.

    Returns:
        consistency: [B, 1, H, W] mean consistency score (0–1 for cosine).
        num_valid: [B, 1, H, W] count of valid source views per pixel.
    """
    B, C, H, W = feat_ref.shape
    device = feat_ref.device

    score_sum = torch.zeros(B, 1, H, W, device=device)
    num_valid = torch.zeros(B, 1, H, W, device=device)

    feat_ref_norm = F.normalize(feat_ref, dim=1, eps=1e-8) if metric == "cosine" else None

    for feat_src, viewmat_src in zip(feat_sources, viewmats_src):
        warped, mask = warp_features(feat_src, depth_ref, viewmat_ref, viewmat_src, K)

        if metric == "cosine":
            warped_norm = F.normalize(warped, dim=1, eps=1e-8)
            sim = (feat_ref_norm * warped_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            # Map from [-1, 1] to [0, 1]
            sim = (sim + 1.0) * 0.5
        elif metric == "l2":
            dist = (feat_ref - warped).pow(2).sum(dim=1, keepdim=True).sqrt()
            # Convert distance to similarity: exp(-dist)
            sim = torch.exp(-dist)
        else:
            raise ValueError(f"Unknown metric '{metric}', expected 'cosine' or 'l2'")

        score_sum = score_sum + sim * mask
        num_valid = num_valid + mask

    consistency = score_sum / num_valid.clamp(min=1.0)
    return consistency, num_valid


def select_source_views(
    viewmat_ref: torch.Tensor,
    viewmats_all: torch.Tensor,
    num_sources: int = 2,
    min_baseline: float = 0.05,
    max_baseline: float = 1.0,
    exclude_idx: int = -1,
) -> List[int]:
    """Select good source views for multi-view consistency.

    Prefers views with moderate baseline and similar viewing direction.

    Args:
        viewmat_ref: [4, 4] reference world-to-camera.
        viewmats_all: [N, 4, 4] all candidate world-to-camera matrices.
        num_sources: number of source views to select.
        min_baseline: minimum camera translation distance.
        max_baseline: maximum camera translation distance.
        exclude_idx: index in viewmats_all to exclude (typically the reference).

    Returns:
        List of indices into viewmats_all.
    """
    N = viewmats_all.shape[0]
    device = viewmats_all.device

    # Camera positions in world = -R^T @ t
    def _cam_pos(vm: torch.Tensor) -> torch.Tensor:
        R = vm[:3, :3]
        t = vm[:3, 3]
        return -R.T @ t  # [3]

    pos_ref = _cam_pos(viewmat_ref)  # [3]
    forward_ref = viewmat_ref[2, :3]  # 3rd row of R = forward direction in world

    baselines = torch.zeros(N, device=device)
    dir_scores = torch.zeros(N, device=device)

    for i in range(N):
        pos_i = _cam_pos(viewmats_all[i])
        baselines[i] = (pos_i - pos_ref).norm()
        forward_i = viewmats_all[i, 2, :3]
        dir_scores[i] = F.cosine_similarity(
            forward_ref.unsqueeze(0), forward_i.unsqueeze(0)
        ).squeeze()

    # Mask: within baseline range and not excluded
    valid = (baselines >= min_baseline) & (baselines <= max_baseline)
    if 0 <= exclude_idx < N:
        valid[exclude_idx] = False

    if valid.sum() == 0:
        # Fallback: relax constraints, just pick closest non-excluded views
        fallback_baselines = baselines.clone()
        if 0 <= exclude_idx < N:
            fallback_baselines[exclude_idx] = float("inf")
        _, indices = fallback_baselines.topk(min(num_sources, N), largest=False)
        return indices.tolist()

    # Score: prefer higher direction similarity, penalise extreme baselines
    mid_baseline = (min_baseline + max_baseline) / 2.0
    baseline_score = 1.0 - (baselines - mid_baseline).abs() / (max_baseline - min_baseline + 1e-8)
    combined_score = 0.6 * dir_scores + 0.4 * baseline_score

    # Zero out invalid
    combined_score[~valid] = -float("inf")

    k = min(num_sources, int(valid.sum().item()))
    _, indices = combined_score.topk(k, largest=True)
    return indices.tolist()
