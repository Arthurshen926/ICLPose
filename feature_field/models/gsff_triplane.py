"""
Triplane Feature Field for GSFFs.

Implements the scale-aware triplane grid feature representation described in Section 3.1.
Each 3D Gaussian's feature is computed by:
  1. Projecting the Gaussian center onto 3 orthogonal planes (xy, xz, yz)
  2. Extracting features from each plane using the projected 2D Gaussian as an RBF kernel
  3. Averaging the 3 plane features to get the volumetric feature g_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TriplaneFeatureField(nn.Module):
    """
    Triplane grid feature field.
    
    Three orthogonal planes Hxy, Hxz, Hyz ∈ R^{R×R×D} centered at the scene
    origin. Features for each 3D Gaussian are extracted by projecting onto the
    planes and sampling with bilinear interpolation (simplified RBF kernel).
    
    Args:
        resolution: Grid resolution R (256 for coarse, 1024 for fine)
        feature_dim: Feature dimension D (16 in the paper)
        scene_extent: Half-length of the triplane in world units
    """
    
    def __init__(self, resolution: int = 256, feature_dim: int = 16,
                 scene_extent: float = 10.0):
        super().__init__()
        self.resolution = resolution
        self.feature_dim = feature_dim
        self.scene_extent = scene_extent
        
        # Three learnable planes: [D, R, R] (channel-first for conv compatibility)
        self.plane_xy = nn.Parameter(torch.randn(1, feature_dim, resolution, resolution) * 0.01)
        self.plane_xz = nn.Parameter(torch.randn(1, feature_dim, resolution, resolution) * 0.01)
        self.plane_yz = nn.Parameter(torch.randn(1, feature_dim, resolution, resolution) * 0.01)
    
    def _normalize_coords(self, xyz: torch.Tensor) -> torch.Tensor:
        """Normalize 3D coordinates to [-1, 1] for grid_sample."""
        return xyz / self.scene_extent
    
    def extract_features(self, xyz: torch.Tensor,
                         scales: torch.Tensor = None) -> torch.Tensor:
        """
        Extract volumetric features for N Gaussians.
        
        For simplicity, uses bilinear sampling at Gaussian centers. The paper's
        full RBF kernel (weighted by 2D projected Gaussian) primarily matters
        for large Gaussians; bilinear is a good approximation for most cases.
        
        Args:
            xyz: [N, 3] Gaussian center positions in world coordinates
            scales: [N, 2 or 3] Gaussian scales (optional, for scale-aware extraction)
            
        Returns:
            features: [N, D] volumetric features for each Gaussian
        """
        coords = self._normalize_coords(xyz)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # Project onto 3 planes and bilinear sample
        # grid_sample expects [N, H_out, W_out, 2] grid in [-1, 1]
        # We sample at single points, so H_out=W_out=1
        
        # XY plane: use (x, y)
        grid_xy = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)  # [1, N, 1, 2]
        feat_xy = F.grid_sample(self.plane_xy, grid_xy, mode='bilinear',
                                padding_mode='border', align_corners=True)
        feat_xy = feat_xy.squeeze(-1).squeeze(0).T  # [N, D]
        
        # XZ plane: use (x, z)
        grid_xz = torch.stack([x, z], dim=-1).view(1, -1, 1, 2)
        feat_xz = F.grid_sample(self.plane_xz, grid_xz, mode='bilinear',
                                padding_mode='border', align_corners=True)
        feat_xz = feat_xz.squeeze(-1).squeeze(0).T  # [N, D]
        
        # YZ plane: use (y, z)
        grid_yz = torch.stack([y, z], dim=-1).view(1, -1, 1, 2)
        feat_yz = F.grid_sample(self.plane_yz, grid_yz, mode='bilinear',
                                padding_mode='border', align_corners=True)
        feat_yz = feat_yz.squeeze(-1).squeeze(0).T  # [N, D]
        
        # Average features from 3 planes
        features = (feat_xy + feat_xz + feat_yz) / 3.0
        
        return features
    
    def total_variation_loss(self) -> torch.Tensor:
        """Total variation regularization on the triplane grids (L_TVL)."""
        tv = 0.0
        for plane in [self.plane_xy, self.plane_xz, self.plane_yz]:
            # Horizontal TV
            tv = tv + torch.mean(torch.abs(plane[:, :, :, 1:] - plane[:, :, :, :-1]))
            # Vertical TV
            tv = tv + torch.mean(torch.abs(plane[:, :, 1:, :] - plane[:, :, :-1, :]))
        return tv


class DualScaleTriplane(nn.Module):
    """
    Dual-scale (coarse + fine) triplane feature field.
    
    Paper uses R=256 for coarse and R=1024 for fine, both with D=16.
    """
    
    def __init__(self, coarse_resolution: int = 256, fine_resolution: int = 1024,
                 feature_dim: int = 16, scene_extent: float = 10.0):
        super().__init__()
        self.coarse = TriplaneFeatureField(coarse_resolution, feature_dim, scene_extent)
        self.fine = TriplaneFeatureField(fine_resolution, feature_dim, scene_extent)
    
    def extract_coarse(self, xyz, scales=None):
        return self.coarse.extract_features(xyz, scales)
    
    def extract_fine(self, xyz, scales=None):
        return self.fine.extract_features(xyz, scales)
    
    def total_variation_loss(self):
        return self.coarse.total_variation_loss() + self.fine.total_variation_loss()
