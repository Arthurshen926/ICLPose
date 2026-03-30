"""FeatSharp-3D: Feature sharpening for 3DGS rendered features.

3DGS alpha-blending inherently over-smooths features (each pixel is a weighted
average of overlapping Gaussians). FeatSharp-3D addresses this via:
  1. Detecting over-smoothed regions through multi-view consistency
  2. Applying learned or analytical sharpening to recover high-frequency detail

Inspired by FeatSharp (ICML 2025), adapted for 3DGS rendered feature fields.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AnalyticalSharpener(nn.Module):
    """Unsharp masking in feature space — no learnable parameters."""

    def __init__(self, sigma: float = 1.0, strength: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.strength = strength

        # Pre-compute Gaussian kernel
        kernel_size = max(3, 2 * round(3 * sigma) + 1)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        # Shape: [1, 1, K, K] — expanded per-channel via groups in forward
        self.register_buffer("kernel", kernel_2d.unsqueeze(0).unsqueeze(0))

    def forward(self, feature_map: Tensor) -> Tensor:
        """Apply unsharp masking.

        Args:
            feature_map: [B, C, H, W] input features.

        Returns:
            [B, C, H, W] sharpened features.
        """
        B, C, H, W = feature_map.shape
        # Depthwise blur: expand kernel to [C, 1, K, K]
        kernel = self.kernel.expand(C, -1, -1, -1)
        blurred = F.conv2d(feature_map, kernel, padding=self.padding, groups=C)
        high_freq = feature_map - blurred
        return feature_map + self.strength * high_freq


class _DepthwiseSeparableBlock(nn.Module):
    """Depthwise separable conv block with residual connection."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        return residual + x


class LearnedSharpener(nn.Module):
    """Lightweight learned feature sharpener using depthwise separable convolutions."""

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 32, kernel_size: int = 3):
        super().__init__()
        self.block1 = _DepthwiseSeparableBlock(feature_dim, kernel_size)
        self.block2 = _DepthwiseSeparableBlock(feature_dim, kernel_size)

    def forward(self, feature_map: Tensor) -> Tensor:
        """Apply learned sharpening.

        Args:
            feature_map: [B, C, H, W] input features.

        Returns:
            [B, C, H, W] sharpened features.
        """
        x = self.block1(feature_map)
        x = self.block2(x)
        return x


def warp_to_reference(
    feat_src: Tensor,
    depth_ref: Tensor,
    viewmat_ref: Tensor,
    viewmat_src: Tensor,
    K: Tensor,
) -> tuple[Tensor, Tensor]:
    """Warp source features to reference view using depth-based reprojection.

    Steps:
        1. Unproject reference pixels to 3D using depth
        2. Transform to source camera frame
        3. Project to source pixel coordinates
        4. Grid sample source features at projected coordinates

    Args:
        feat_src: [B, C, H, W] source view features.
        depth_ref: [B, H, W] reference view depth.
        viewmat_ref: [B, 4, 4] reference world-to-camera.
        viewmat_src: [B, 4, 4] source world-to-camera.
        K: [3, 3] camera intrinsics.

    Returns:
        warped_feat: [B, C, H, W] source features warped into reference frame.
        valid_mask: [B, 1, H, W] binary mask of valid (in-bounds, positive-depth) pixels.
    """
    B, C, H, W = feat_src.shape
    device = feat_src.device

    # --- 1. Build reference pixel grid ----------------------------------------
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(u)
    # [3, H*W]
    pixel_coords = torch.stack([u.reshape(-1), v.reshape(-1), ones.reshape(-1)], dim=0)

    # --- 2. Unproject to 3D in reference camera frame -------------------------
    K_inv = torch.inverse(K)  # [3, 3]
    # [3, H*W]
    rays = K_inv @ pixel_coords
    # [B, H*W]
    depth_flat = depth_ref.reshape(B, -1)
    # [B, 3, H*W]
    pts_ref = rays.unsqueeze(0) * depth_flat.unsqueeze(1)

    # --- 3. Reference camera → world → source camera -------------------------
    # ref_cam_to_world = inv(viewmat_ref), then world_to_src = viewmat_src
    # Combined: T_src_from_ref = viewmat_src @ inv(viewmat_ref)
    ref_to_world = torch.inverse(viewmat_ref)  # [B, 4, 4]
    T_src_from_ref = viewmat_src @ ref_to_world  # [B, 4, 4]

    R = T_src_from_ref[:, :3, :3]  # [B, 3, 3]
    t = T_src_from_ref[:, :3, 3:]  # [B, 3, 1]
    # [B, 3, H*W]
    pts_src = R @ pts_ref + t

    # --- 4. Project to source pixel coordinates -------------------------------
    z_src = pts_src[:, 2:3, :]  # [B, 1, H*W]
    # Avoid division by zero / negative depth
    safe_z = z_src.clamp(min=1e-6)
    pts_src_norm = pts_src / safe_z  # [B, 3, H*W]

    # [B, 3, H*W]
    proj = K.unsqueeze(0) @ pts_src_norm
    u_src = proj[:, 0, :]  # [B, H*W]
    v_src = proj[:, 1, :]  # [B, H*W]

    # --- 5. Normalise to [-1, 1] for grid_sample -----------------------------
    u_norm = 2.0 * u_src / (W - 1) - 1.0
    v_norm = 2.0 * v_src / (H - 1) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B, H, W, 2)

    # --- 6. Sample source features --------------------------------------------
    warped_feat = F.grid_sample(
        feat_src, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    # --- 7. Validity mask -----------------------------------------------------
    depth_valid = (depth_flat > 0).unsqueeze(1).reshape(B, 1, H, W)
    z_positive = (z_src.reshape(B, 1, H, W) > 1e-6)
    in_bounds = (
        (u_norm.reshape(B, 1, H, W).abs() <= 1.0)
        & (v_norm.reshape(B, 1, H, W).abs() <= 1.0)
    )
    valid_mask = (depth_valid & z_positive & in_bounds).float()

    return warped_feat, valid_mask


class MultiViewConsistencySharpener(nn.Module):
    """FeatSharp-3D: multi-view consistency-guided feature refinement.

    High consistency → feature is reliable, keep as-is.
    Low consistency  → over-smoothed or view-dependent artifact, apply correction.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        num_source_views: int = 2,
        consistency_threshold: float = 0.8,
        sharpening_strength: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_source_views = num_source_views
        self.consistency_threshold = consistency_threshold
        self.sharpening_strength = sharpening_strength

        # Learned residual correction conditioned on consistency
        self.correction = nn.Sequential(
            nn.Conv2d(feature_dim + 1, feature_dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, feature_dim), feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
        )

    def forward(
        self,
        features_ref: Tensor,
        features_sources: List[Tensor],
        depths_ref: Tensor,
        viewmats_ref: Tensor,
        viewmats_sources: List[Tensor],
        K: Tensor,
    ) -> Tensor:
        """Sharpen reference features using multi-view consistency.

        Args:
            features_ref: [B, C, H, W] reference view features.
            features_sources: list of [B, C, H, W] from nearby source views.
            depths_ref: [B, H, W] depth from reference view.
            viewmats_ref: [B, 4, 4] reference world-to-camera.
            viewmats_sources: list of [B, 4, 4] source world-to-camera matrices.
            K: [3, 3] camera intrinsics.

        Returns:
            [B, C, H, W] refined features.
        """
        B, C, H, W = features_ref.shape

        # --- 1. Warp each source view's features to the reference frame -------
        cos_sims: List[Tensor] = []
        total_valid = torch.zeros(B, 1, H, W, device=features_ref.device)

        ref_norm = F.normalize(features_ref, dim=1)  # [B, C, H, W]

        for feat_src, vmat_src in zip(features_sources, viewmats_sources):
            warped, valid = warp_to_reference(feat_src, depths_ref, viewmats_ref, vmat_src, K)
            warped_norm = F.normalize(warped, dim=1)
            # Per-pixel cosine similarity: sum over C
            sim = (ref_norm * warped_norm).sum(dim=1, keepdim=True)  # [B, 1, H, W]
            sim = sim * valid  # zero out invalid pixels
            cos_sims.append(sim)
            total_valid = total_valid + valid

        # --- 2. Mean consistency score ----------------------------------------
        stacked = torch.stack(cos_sims, dim=0).sum(dim=0)  # [B, 1, H, W]
        # Avoid div-by-zero where no source view is valid
        consistency = stacked / total_valid.clamp(min=1.0)  # [B, 1, H, W]

        # --- 3. Learned residual correction weighted by inconsistency ---------
        correction_input = torch.cat([features_ref, consistency], dim=1)  # [B, C+1, H, W]
        residual = self.correction(correction_input)  # [B, C, H, W]

        # Weight correction: stronger where consistency is low
        weight = (1.0 - consistency.clamp(0.0, 1.0)) * self.sharpening_strength
        refined = features_ref + weight * residual

        return refined


class FeatSharp3D(nn.Module):
    """Unified wrapper combining sharpening strategies for 3DGS features.

    Modes:
        'analytical' — parameter-free unsharp masking (fast baseline).
        'learned'    — lightweight depthwise-separable sharpener.
        'multiview'  — FeatSharp-3D with multi-view consistency guidance.
        'none'       — identity pass-through.
    """

    MODES = ("analytical", "learned", "multiview", "none")

    def __init__(self, mode: str = "analytical", feature_dim: int = 64, **kwargs):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.MODES}")
        self.mode = mode

        if mode == "analytical":
            self.sharpener = AnalyticalSharpener(
                sigma=kwargs.get("sigma", 1.0),
                strength=kwargs.get("strength", 0.5),
            )
        elif mode == "learned":
            self.sharpener = LearnedSharpener(
                feature_dim=feature_dim,
                hidden_dim=kwargs.get("hidden_dim", 32),
                kernel_size=kwargs.get("kernel_size", 3),
            )
        elif mode == "multiview":
            self.sharpener = MultiViewConsistencySharpener(
                feature_dim=feature_dim,
                num_source_views=kwargs.get("num_source_views", 2),
                consistency_threshold=kwargs.get("consistency_threshold", 0.8),
                sharpening_strength=kwargs.get("sharpening_strength", 1.0),
            )
        else:  # none
            self.sharpener = nn.Identity()

    def forward(self, feature_map: Tensor, **kwargs) -> Tensor:
        """Dispatch to the configured sharpener.

        Args:
            feature_map: [B, C, H, W] input features.
            **kwargs: additional arguments forwarded to the multiview sharpener
                (features_sources, depths_ref, viewmats_ref, viewmats_sources, K).

        Returns:
            [B, C, H, W] sharpened features.
        """
        if self.mode == "multiview":
            return self.sharpener(
                features_ref=feature_map,
                features_sources=kwargs["features_sources"],
                depths_ref=kwargs["depths_ref"],
                viewmats_ref=kwargs["viewmats_ref"],
                viewmats_sources=kwargs["viewmats_sources"],
                K=kwargs["K"],
            )
        return self.sharpener(feature_map)
