"""
Tri-Plane Feature Model for 3DGS
=================================
GSFF-inspired spatial feature encoding using tri-plane representation.

Instead of per-Gaussian feature vectors, features are decoded from three
axis-aligned 2D feature planes (XY, XZ, YZ) via bilinear interpolation
and a shared MLP with multi-head output.

For any 3D Gaussian center (x, y, z):
  1. Normalize to [-1, 1] using scene bounding box
  2. Sample from each plane: f_xy = grid_sample(plane_xy, (x, y))
                             f_xz = grid_sample(plane_xz, (x, z))
                             f_yz = grid_sample(plane_yz, (y, z))
  3. Concatenate: f = [f_xy; f_xz; f_yz]  (3 * plane_channels)
  4. MLP shared trunk → per-scale heads → L2 normalized features

Advantages over per-Gaussian features:
  - Spatial continuity: neighboring Gaussians get correlated features
  - Parameter efficiency: ~6M vs ~94M for 417K Gaussians × 224d
  - Sub-pixel interpolation capability from continuous tri-plane field
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from feature_gaussian.models.gaussian_feature_model import GaussianFeatureModel


class TriPlaneDecoder(nn.Module):
    """Shared MLP trunk + multi-head output for decoding tri-plane features."""

    def __init__(
        self,
        in_dim: int,           # 3 * plane_channels
        trunk_dim: int = 128,
        head_dims: Dict[str, int] = None,
    ):
        super().__init__()
        if head_dims is None:
            head_dims = {"coarse": 32, "mid": 64, "fine": 64}
        self.head_names = sorted(head_dims.keys())

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_dim),
            nn.GELU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.GELU(),
        )

        self.heads = nn.ModuleDict()
        for name, dim in head_dims.items():
            self.heads[name] = nn.Linear(trunk_dim, dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (N, in_dim) concatenated tri-plane features
        Returns:
            Dict[str, (N, head_dim)] L2-normalized per-scale features
        """
        h = self.trunk(x)
        out = {}
        for name in self.head_names:
            out[name] = F.normalize(self.heads[name](h), p=2, dim=-1)
        return out


class TriPlaneFeatureModel(GaussianFeatureModel):
    """
    3DGS model with tri-plane spatial feature encoding.

    Inherits frozen geometry from GaussianFeatureModel, replaces
    per-Gaussian _loc_feature with tri-plane + MLP decoder.

    The total feature_dim is sum of all head dims (for compatibility
    with the rendering pipeline that expects get_loc_feature → [N, D]).
    """

    def __init__(
        self,
        plane_resolution: int = 256,
        plane_channels: int = 32,
        trunk_dim: int = 128,
        head_dims: Dict[str, int] = None,
    ):
        if head_dims is None:
            head_dims = {"coarse": 32, "mid": 64, "fine": 64}

        total_dim = sum(head_dims.values())
        super().__init__(feature_dim=total_dim)

        self.plane_resolution = plane_resolution
        self.plane_channels = plane_channels
        self.head_dims = head_dims

        # Store ordered head info for consistent feature concatenation
        self._head_names = sorted(head_dims.keys())
        self._head_offsets = {}
        offset = 0
        for name in self._head_names:
            self._head_offsets[name] = (offset, offset + head_dims[name])
            offset += head_dims[name]

        # Three learnable feature planes: XY, XZ, YZ
        R = plane_resolution
        C = plane_channels
        self.plane_xy = nn.Parameter(torch.randn(1, C, R, R) * 0.01)
        self.plane_xz = nn.Parameter(torch.randn(1, C, R, R) * 0.01)
        self.plane_yz = nn.Parameter(torch.randn(1, C, R, R) * 0.01)

        # MLP decoder
        self.decoder = TriPlaneDecoder(
            in_dim=3 * C,
            trunk_dim=trunk_dim,
            head_dims=head_dims,
        )

        # Scene bounding box (set after loading PLY)
        self.register_buffer("bbox_min", torch.zeros(3))
        self.register_buffer("bbox_max", torch.ones(3))

    def _compute_bbox(self, padding: float = 0.1, percentile: float = 0.5):
        """Compute scene bounding box from Gaussian centers with padding.
        
        Uses percentile-based bounds to exclude extreme outlier Gaussians,
        then adds padding as a fraction of the extent.
        """
        xyz = self._xyz.float()  # (N, 3)
        lo = percentile / 100.0
        hi = 1.0 - lo
        bmin = torch.quantile(xyz, lo, dim=0)
        bmax = torch.quantile(xyz, hi, dim=0)
        extent = bmax - bmin
        self.bbox_min = bmin - padding * extent
        self.bbox_max = bmax + padding * extent

    def load_ply(self, ply_path: str):
        """Load PLY and compute scene bounding box."""
        super().load_ply(ply_path)
        self._compute_bbox()
        # The inherited _loc_feature is unused; override with dummy
        self._loc_feature = nn.Parameter(torch.empty(0), requires_grad=False)
        print(f"  [TriPlane] BBox: {self.bbox_min.tolist()} → {self.bbox_max.tolist()}")
        print(f"  [TriPlane] Planes: {self.plane_resolution}² × {self.plane_channels}d × 3")

    def _normalize_coords(self, xyz: torch.Tensor) -> torch.Tensor:
        """Normalize Gaussian centers to [-1, 1] for grid_sample."""
        # xyz: (N, 3)
        normalized = 2.0 * (xyz - self.bbox_min) / (self.bbox_max - self.bbox_min + 1e-8) - 1.0
        return normalized.clamp(-1.0, 1.0)

    def _sample_planes(self, xyz: torch.Tensor) -> torch.Tensor:
        """Sample features from tri-planes at Gaussian center locations.

        Args:
            xyz: (N, 3) Gaussian centers in world coordinates

        Returns:
            (N, 3*C) concatenated features from XY, XZ, YZ planes
        """
        coords = self._normalize_coords(xyz)  # (N, 3) in [-1, 1]
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        # grid_sample expects (B, C, H, W) input and (B, H_out, W_out, 2) grid
        # For N points: grid shape (1, 1, N, 2)
        def sample_plane(plane, u, v):
            grid = torch.stack([v, u], dim=-1)  # (N, 2), grid_sample uses (x=W, y=H)
            grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
            out = F.grid_sample(
                plane, grid, mode="bilinear", padding_mode="border", align_corners=True
            )  # (1, C, 1, N)
            return out.squeeze(0).squeeze(1).T  # (N, C)

        f_xy = sample_plane(self.plane_xy, x, y)
        f_xz = sample_plane(self.plane_xz, x, z)
        f_yz = sample_plane(self.plane_yz, y, z)

        return torch.cat([f_xy, f_xz, f_yz], dim=-1)  # (N, 3*C)

    def decode_features(self, xyz: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Decode multi-scale features for all Gaussians.

        Args:
            xyz: Optional (N, 3) coordinates. Defaults to self._xyz.

        Returns:
            Dict[str, (N, D_head)] per-scale L2-normalized features
        """
        if xyz is None:
            xyz = self._xyz
        tri_feats = self._sample_planes(xyz)  # (N, 3*C)
        return self.decoder(tri_feats)

    @property
    def get_loc_feature(self) -> torch.Tensor:
        """Compatibility: return concatenated multi-scale features [N, total_dim].

        This is called by the rendering pipeline (FeatureRenderer).
        Features are L2-normalized per-head, then concatenated.
        """
        scale_feats = self.decode_features()
        parts = [scale_feats[name] for name in self._head_names]
        return torch.cat(parts, dim=-1)  # (N, sum(head_dims))

    def get_scale_feature(self, scale_name: str) -> torch.Tensor:
        """Get features for a specific scale only.

        Args:
            scale_name: 'coarse', 'mid', or 'fine'

        Returns:
            (N, D_head) L2-normalized features
        """
        return self.decode_features()[scale_name]

    def get_scale_slice(self, scale_name: str) -> Tuple[int, int]:
        """Get the (start, end) indices of a scale in the concatenated feature vector."""
        return self._head_offsets[scale_name]

    def split_feature_map(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split a rendered concatenated feature map into per-scale maps.

        Args:
            feature_map: (total_dim, H, W) rendered feature map

        Returns:
            Dict[str, (D_head, H, W)]
        """
        out = {}
        for name in self._head_names:
            s, e = self._head_offsets[name]
            out[name] = feature_map[s:e]
        return out

    def trainable_parameters(self):
        """Return parameter groups for optimizer (planes + decoder)."""
        return [
            {"name": "planes", "params": [self.plane_xy, self.plane_xz, self.plane_yz]},
            {"name": "decoder", "params": list(self.decoder.parameters())},
        ]
