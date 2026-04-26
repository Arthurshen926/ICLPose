"""
Spatial Hash Grid + MLP Decoder for Coarse Semantic Features.

Supports two modes:

  1. legacy:
      pos_3d [0,1]^3 → HashGrid → hash_feat
      view_dir → SH → sh_feat
      latent → passthrough
      cat(hash_feat, latent, sh_feat) → MLP → coarse_feat

  2. implicit_scale:
      pos_3d [0,1]^3 → HashGrid → hash_feat
      scale_2d → positional encoding → scale_feat
      cat(hash_feat, scale_feat) → MLP → coarse_feat

The second mode matches the intended DCFF coarse branch: purely implicit,
low-frequency, and independent from explicit latent/view-dependent signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False


class FourierFeatureEncoder(nn.Module):
    """Parameter-free sinusoidal encoder used as a tiny-cudann fallback."""

    def __init__(self, input_dim: int, n_frequencies: int = 8, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.include_input = include_input
        self.register_buffer(
            'freq_bands',
            2.0 ** torch.arange(n_frequencies, dtype=torch.float32),
            persistent=False,
        )
        self.n_output_dims = input_dim * ((1 if include_input else 0) + 2 * n_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = []
        if self.include_input:
            encoded.append(x)
        for freq in self.freq_bands:
            scaled = x * freq * torch.pi
            encoded.append(torch.sin(scaled))
            encoded.append(torch.cos(scaled))
        return torch.cat(encoded, dim=-1)


class ScalePositionalEncoder(nn.Module):
    """Sinusoidal encoding for 2D Gaussian scales."""

    def __init__(self, input_dim: int = 2, n_frequencies: int = 4, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.n_frequencies = n_frequencies
        self.include_input = include_input
        self.register_buffer(
            'freq_bands',
            2.0 ** torch.arange(n_frequencies, dtype=torch.float32),
            persistent=False,
        )
        self.output_dim = input_dim * ((1 if include_input else 0) + 2 * n_frequencies)

    def forward(self, scales: torch.Tensor) -> torch.Tensor:
        scales = torch.log1p(scales.clamp_min(0.0))
        encoded = []
        if self.include_input:
            encoded.append(scales)
        for freq in self.freq_bands:
            scaled = scales * freq * torch.pi
            encoded.append(torch.sin(scaled))
            encoded.append(torch.cos(scaled))
        return torch.cat(encoded, dim=-1)


class SpatialHashGrid(nn.Module):
    """Multi-resolution hash grid with MLP decoder for coarse semantic features."""

    def __init__(
        self,
        scene_extent: float = 10.0,
        feature_dim: int = 64,
        input_mode: str = 'legacy',
        latent_dim: int = 16,
        scale_dim: int = 2,
        scale_pe_freqs: int = 4,
        include_raw_scale: bool = True,
        # Hash grid params
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        # View encoding
        sh_degree: int = 3,
        # MLP params
        mlp_hidden: int = 128,
        mlp_layers: int = 2,
    ):
        super().__init__()
        self.scene_extent = scene_extent
        self.feature_dim = feature_dim
        self.input_mode = input_mode
        self.latent_dim = latent_dim
        self.uses_tcnn = TCNN_AVAILABLE

        if self.input_mode not in {'legacy', 'implicit_scale'}:
            raise ValueError(f"Unsupported input_mode={input_mode}")

        if self.uses_tcnn:
            # Multi-resolution hash grid: positions [0, 1]^3 → hash features
            per_level_scale = np.exp2(
                np.log2(max_resolution / base_resolution) / (n_levels - 1)
            )
            self.hash_encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": n_levels,
                    "n_features_per_level": n_features_per_level,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                },
                dtype=torch.float32,
            )
        else:
            fallback_freqs = max(4, min(10, n_levels // 2))
            self.hash_encoding = FourierFeatureEncoder(3, n_frequencies=fallback_freqs)
        hash_dim = self.hash_encoding.n_output_dims

        if self.input_mode == 'legacy':
            self.sh_degree = sh_degree
            if self.uses_tcnn:
                self.sh_encoding = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "SphericalHarmonics",
                        "degree": sh_degree,
                    },
                    dtype=torch.float32,
                )
            else:
                self.sh_encoding = FourierFeatureEncoder(3, n_frequencies=max(2, sh_degree + 1))
            sh_dim = self.sh_encoding.n_output_dims
            self.scale_encoder = None
            mlp_input_dim = hash_dim + latent_dim + sh_dim
        else:
            self.sh_degree = 0
            self.sh_encoding = None
            self.scale_encoder = ScalePositionalEncoder(
                input_dim=scale_dim,
                n_frequencies=scale_pe_freqs,
                include_input=include_raw_scale,
            )
            sh_dim = 0
            mlp_input_dim = hash_dim + self.scale_encoder.output_dim

        layers = []
        prev = mlp_input_dim
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Linear(prev, mlp_hidden), nn.GELU()])
            prev = mlp_hidden
        layers.append(nn.Linear(prev, feature_dim))
        self.mlp = nn.Sequential(*layers)

        n_hash = sum(p.numel() for p in self.hash_encoding.parameters())
        n_mlp = sum(p.numel() for p in self.mlp.parameters())
        print(f"  [SpatialHashGrid] hash={n_hash:,} + mlp={n_mlp:,} = {n_hash+n_mlp:,} params")
        if not self.uses_tcnn:
            print("    tinycudann unavailable, using Fourier fallback encoder")
        if self.input_mode == 'legacy':
            print(f"    mode=legacy, hash_dim={hash_dim}, sh_dim={sh_dim}, latent_dim={latent_dim}")
        else:
            print(
                f"    mode=implicit_scale, hash_dim={hash_dim}, "
                f"scale_dim={scale_dim}, scale_pe_dim={self.scale_encoder.output_dim}"
            )
        print(f"    mlp: {mlp_input_dim} → {'→'.join([str(mlp_hidden)]*(mlp_layers-1))} → {feature_dim}")

    def normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Map world positions from [-extent, extent] to [0, 1]."""
        return ((positions / self.scene_extent) + 1.0) * 0.5

    def forward(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor = None,
        latent: torch.Tensor = None,
        view_dirs: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode coarse semantic features.

        Args:
            positions: [N, 3] world-space 3D positions
            scales: [N, scale_dim] per-pixel rasterized Gaussian scales
            latent: [N, latent_dim] per-pixel latent from rasterized z_map (legacy)
            view_dirs: [N, 3] normalized view direction vectors (legacy)
            valid_mask: [N] bool mask for valid (visible) pixels

        Returns:
            features: [N, feature_dim] coarse semantic feature vectors
        """
        if valid_mask is not None and not valid_mask.all():
            out = torch.zeros(positions.shape[0], self.feature_dim,
                              device=positions.device, dtype=positions.dtype)
            if valid_mask.sum() == 0:
                return out
            pos_v = positions[valid_mask]
            scale_v = scales[valid_mask] if scales is not None else None
            lat_v = latent[valid_mask] if latent is not None else None
            vd_v = view_dirs[valid_mask] if view_dirs is not None else None
            out[valid_mask] = self._forward_impl(pos_v, scale_v, lat_v, vd_v)
            return out
        return self._forward_impl(positions, scales, latent, view_dirs)

    def _forward_impl(self, positions, scales, latent, view_dirs):
        """Core forward without masking."""
        # Normalize positions to [0, 1] for hash grid
        pos_norm = self.normalize_positions(positions).clamp(0.0, 1.0)

        # Encode position via hash grid
        hash_feat = self.hash_encoding(pos_norm)  # [N, hash_dim]

        if self.input_mode == 'legacy':
            if latent is None or view_dirs is None:
                raise ValueError("legacy SpatialHashGrid requires latent and view_dirs")
            vd_norm = F.normalize(view_dirs, dim=-1)
            sh_feat = self.sh_encoding(vd_norm)
            mlp_input = torch.cat([hash_feat, latent, sh_feat], dim=-1)
        else:
            if scales is None:
                raise ValueError("implicit_scale SpatialHashGrid requires scales")
            scale_feat = self.scale_encoder(scales)
            mlp_input = torch.cat([hash_feat, scale_feat], dim=-1)
        return self.mlp(mlp_input).float()  # [N, feature_dim], ensure fp32

    def total_variation_loss(self):
        """TV regularization on hash grid parameters for spatial smoothness.

        Samples random 3D points and computes finite-difference gradients.
        """
        n_samples = 4096
        pts = torch.rand(n_samples, 3, device=next(self.mlp.parameters()).device)
        eps = 1e-3

        tv = 0.0
        for dim in range(3):
            pts_plus = pts.clone()
            pts_plus[:, dim] = (pts_plus[:, dim] + eps).clamp(0, 1)
            feat0 = self.hash_encoding(pts)
            feat1 = self.hash_encoding(pts_plus)
            tv = tv + (feat1 - feat0).abs().mean()

        return tv / 3.0
