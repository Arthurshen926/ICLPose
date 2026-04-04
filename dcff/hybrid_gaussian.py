"""
HybridGaussianModel — 2DGS with per-Gaussian learnable latent codes.

Extends the standard 2DGS Gaussian model with a `_latent` parameter per splat,
enabling deferred feature decoding in screen space.  Geometry (xyz, rotation,
scaling, opacity, SH) can be frozen while only the latent codes are trained.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class HybridGaussianModel(nn.Module):
    """2DGS Gaussian model augmented with learnable per-Gaussian latent codes.

    Properties mirror standard GaussianModel interface expected by gsplat-based
    renderers: get_xyz, get_rotation, get_scaling, get_opacity, etc.
    """

    def __init__(self, sh_degree: int = 3, latent_dim: int = 32):
        super().__init__()
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.latent_dim = latent_dim

        # Gaussian parameters (all nn.Parameter)
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3))
        self._features_rest = nn.Parameter(torch.empty(0, 0, 3))
        self._opacity = nn.Parameter(torch.empty(0, 1))
        self._scaling = nn.Parameter(torch.empty(0, 2))  # 2DGS: 2 scales
        self._rotation = nn.Parameter(torch.empty(0, 4))
        self._latent = nn.Parameter(torch.empty(0, latent_dim))

        # Densification stats
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.spatial_lr_scale = 1.0

    @property
    def num_points(self):
        return self._xyz.shape[0]

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_scaling_for_render(self):
        """Return [N, 3] scales padded with a near-zero third dimension for 2DGS."""
        s = torch.exp(self._scaling)  # [N, 2]
        pad = torch.full((s.shape[0], 1), 1e-8, device=s.device, dtype=s.dtype)
        return torch.cat([s, pad], dim=-1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_latent(self):
        return self._latent

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def load_ply(self, path: str, freeze_geometry: bool = True):
        """Load Gaussian parameters from a PLY file.

        PLY must contain: x,y,z, f_dc_*, f_rest_*, opacity, scale_*, rot_*, loc_*
        where loc_* are the per-Gaussian latent codes.
        """
        plydata = PlyData.read(path)
        vertex = plydata.elements[0]

        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        N = xyz.shape[0]

        # SH coefficients
        f_dc = np.stack([vertex[f'f_dc_{i}'] for i in range(3)], axis=1)
        f_dc = f_dc.reshape(N, 1, 3)

        extra_f_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith('f_rest_')],
            key=lambda x: int(x.split('_')[-1])
        )
        if len(extra_f_names) > 0:
            f_rest = np.stack([vertex[n] for n in extra_f_names], axis=1)
            n_rest = f_rest.shape[1]
            f_rest = f_rest.reshape(N, n_rest // 3, 3)
        else:
            f_rest = np.zeros((N, 0, 3), dtype=np.float32)

        opacity = vertex['opacity'].reshape(N, 1)

        # 2DGS scales
        scale_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith('scale_')],
            key=lambda x: int(x.split('_')[-1])
        )
        scales = np.stack([vertex[n] for n in scale_names], axis=1)

        rot_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith('rot_')],
            key=lambda x: int(x.split('_')[-1])
        )
        rotations = np.stack([vertex[n] for n in rot_names], axis=1)

        # Latent codes (loc_*)
        loc_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith('loc_')],
            key=lambda x: int(x.split('_')[-1])
        )
        if len(loc_names) > 0:
            latent = np.stack([vertex[n] for n in loc_names], axis=1)
        else:
            latent = np.zeros((N, self.latent_dim), dtype=np.float32)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device),
                                 requires_grad=not freeze_geometry)
        self._features_dc = nn.Parameter(torch.tensor(f_dc, dtype=torch.float32, device=device),
                                         requires_grad=not freeze_geometry)
        self._features_rest = nn.Parameter(torch.tensor(f_rest, dtype=torch.float32, device=device),
                                           requires_grad=not freeze_geometry)
        self._opacity = nn.Parameter(torch.tensor(opacity, dtype=torch.float32, device=device),
                                     requires_grad=not freeze_geometry)
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device),
                                     requires_grad=not freeze_geometry)
        self._rotation = nn.Parameter(torch.tensor(rotations, dtype=torch.float32, device=device),
                                      requires_grad=not freeze_geometry)
        self._latent = nn.Parameter(torch.tensor(latent, dtype=torch.float32, device=device),
                                    requires_grad=True)

        self.max_radii2D = torch.zeros(N, device=device)

        # Infer active SH degree from number of rest coefficients
        n_sh_rest = f_rest.shape[1]
        num_sh_total = n_sh_rest + 1  # +1 for DC
        self.active_sh_degree = int(math.sqrt(num_sh_total)) - 1

        print(f"[HybridGaussianModel] Loaded {N} Gaussians from {path}")
        print(f"  SH degree: {self.active_sh_degree}, Latent dim: {latent.shape[1]}")
        print(f"  Geometry frozen: {freeze_geometry}")

        return self

    def create_from_pcd(self, xyz, rgb, extent=1.0):
        """Initialize Gaussians from a point cloud."""
        N = xyz.shape[0]
        device = 'cuda'

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device))

        # SH DC from RGB (C0 normalization)
        C0 = 0.28209479177387814
        f_dc = (torch.tensor(rgb, dtype=torch.float32, device=device) - 0.5) / C0
        self._features_dc = nn.Parameter(f_dc.reshape(N, 1, 3))

        num_sh_rest = (self.max_sh_degree + 1) ** 2 - 1
        self._features_rest = nn.Parameter(
            torch.zeros(N, num_sh_rest, 3, dtype=torch.float32, device=device))

        self._opacity = nn.Parameter(
            inverse_sigmoid(0.1 * torch.ones(N, 1, dtype=torch.float32, device=device)))

        # Initialize 2DGS scales
        dist = torch.clamp_min(
            torch.sqrt(torch.sum((torch.tensor(xyz, device=device).unsqueeze(0) -
                                   torch.tensor(xyz, device=device).unsqueeze(1)) ** 2, dim=-1)),
            1e-7)
        # Use average nearest-neighbor distance
        dist[dist == 0] = 1e7
        avg_dist = dist.topk(4, largest=False).values[:, 1:].mean(dim=1)
        log_scale = torch.log(torch.clamp(avg_dist * 0.5, min=1e-7))
        self._scaling = nn.Parameter(
            log_scale.unsqueeze(-1).repeat(1, 2))

        self._rotation = nn.Parameter(
            torch.zeros(N, 4, dtype=torch.float32, device=device))
        self._rotation.data[:, 0] = 1.0

        self._latent = nn.Parameter(
            torch.randn(N, self.latent_dim, dtype=torch.float32, device=device) * 0.01)

        self.max_radii2D = torch.zeros(N, device=device)
        self.spatial_lr_scale = extent

        print(f"[HybridGaussianModel] Created {N} Gaussians from point cloud")
        return self

    def training_setup(self, lr_dict=None):
        """Setup optimizer for training. lr_dict maps param names to learning rates."""
        if lr_dict is None:
            lr_dict = {}

        param_groups = []
        defaults = {
            'xyz': 0.00016, 'features_dc': 0.0025, 'features_rest': 0.000125,
            'opacity': 0.05, 'scaling': 0.005, 'rotation': 0.001, 'latent': 0.001
        }
        for name, default_lr in defaults.items():
            param = getattr(self, f'_{name}')
            if param.requires_grad:
                lr = lr_dict.get(name, default_lr)
                param_groups.append({'params': [param], 'lr': lr, 'name': name})

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        return self.optimizer

    def update_learning_rate(self, iteration):
        """Placeholder for LR scheduling."""
        pass

    def set_geometry_trainable(self, trainable=True):
        """Toggle gradient computation for geometry parameters."""
        for name in ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']:
            getattr(self, name).requires_grad_(trainable)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """Track gradient statistics for densification."""
        pass

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """Densification and pruning step."""
        pass

    def reset_opacity(self):
        """Reset opacity to initial values."""
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        self._opacity.data = opacities_new

    def _prune_points(self, mask):
        """Remove points indicated by mask."""
        valid = ~mask
        self._xyz = nn.Parameter(self._xyz[valid])
        self._features_dc = nn.Parameter(self._features_dc[valid])
        self._features_rest = nn.Parameter(self._features_rest[valid])
        self._opacity = nn.Parameter(self._opacity[valid])
        self._scaling = nn.Parameter(self._scaling[valid])
        self._rotation = nn.Parameter(self._rotation[valid])
        self._latent = nn.Parameter(self._latent[valid])
        self.max_radii2D = self.max_radii2D[valid]

