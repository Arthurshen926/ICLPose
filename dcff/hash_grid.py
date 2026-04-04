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
import tinycudann as tcnn


class SpatialHashGrid(nn.Module):
    """Multi-resolution spatial hash grid with MLP decoder for coarse features.

    Uses tinycudann HashGrid encoding for fast multi-resolution hash lookup.
    """

    def __init__(
        self,
        input_mode: str = "implicit_scale",
        output_dim: int = 128,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        max_resolution: int = 2048,
        mlp_hidden: int = 256,
        mlp_layers: int = 4,
        # legacy mode params
        latent_dim: int = 32,
        view_sh_degree: int = 2,
        # implicit_scale mode params
        scale_dim: int = 2,
        scale_pe_freqs: int = 4,
        include_raw_scale: bool = True,
        # scene bounds
        pos_min: float = 0.0,
        pos_max: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.input_mode = input_mode
        self.output_dim = output_dim
        self.pos_min = pos_min
        self.pos_max = pos_max

        # Compute per_level_scale
        import math
        if n_levels > 1:
            per_level_scale = math.exp(
                math.log(max_resolution / base_resolution) / (n_levels - 1)
            )
        else:
            per_level_scale = 1.0

        # tinycudann hash encoding
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
        )
        hash_out_dim = n_levels * n_features_per_level

        # Compute extra input dim based on mode
        if input_mode == "legacy":
            # SH view direction encoding
            self.view_sh_degree = view_sh_degree
            sh_dim = sum(2 * l + 1 for l in range(view_sh_degree + 1))
            extra_dim = latent_dim + sh_dim
            self.latent_dim = latent_dim
        elif input_mode == "implicit_scale":
            self.scale_dim = scale_dim
            self.scale_pe_freqs = scale_pe_freqs
            self.include_raw_scale = include_raw_scale
            # PE: scale_dim * scale_pe_freqs * 2 (sin + cos) + optional raw
            pe_dim = scale_dim * scale_pe_freqs * 2
            if include_raw_scale:
                pe_dim += scale_dim
            extra_dim = pe_dim
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        mlp_in = hash_out_dim + extra_dim

        # MLP decoder
        layers = []
        prev_dim = mlp_in
        for i in range(mlp_layers - 1):
            layers.append(nn.Linear(prev_dim, mlp_hidden))
            layers.append(nn.GELU())
            prev_dim = mlp_hidden
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def encode_scale_pe(self, scales):
        """Positional encoding for 2D Gaussian scales."""
        freqs = 2.0 ** torch.arange(
            self.scale_pe_freqs, device=scales.device, dtype=scales.dtype
        ) * np.pi
        # scales: [N, scale_dim]
        x = scales.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # [N, scale_dim, freqs]
        pe = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [N, scale_dim, 2*freqs]
        pe = pe.reshape(scales.shape[0], -1)  # [N, scale_dim * 2 * freqs]
        if self.include_raw_scale:
            pe = torch.cat([scales, pe], dim=-1)
        return pe

    def normalize_positions(self, positions):
        """Normalize positions to [0, 1] range for hash grid."""
        return (positions - self.pos_min) / (self.pos_max - self.pos_min + 1e-8)

    def forward(self, positions, scales=None, latent=None, view_dirs=None,
                valid_mask=None):
        """
        Args:
            positions: [N, 3] world-space positions
            scales: [N, 2] per-pixel Gaussian scales (implicit_scale mode)
            latent: [N, latent_dim] per-pixel latent (legacy mode)
            view_dirs: [N, 3] unit view directions (legacy mode)
            valid_mask: [N] bool, only process valid pixels

        Returns:
            features: [N, output_dim]
        """
        N = positions.shape[0]
        device = positions.device

        # Output tensor
        out = torch.zeros(N, self.output_dim, device=device, dtype=torch.float32)

        if valid_mask is not None:
            valid_idx = valid_mask.nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                return out
            pos = positions[valid_idx]
            if scales is not None:
                scales = scales[valid_idx]
            if latent is not None:
                latent = latent[valid_idx]
            if view_dirs is not None:
                view_dirs = view_dirs[valid_idx]
        else:
            pos = positions
            valid_idx = None

        # Normalize positions
        pos_norm = self.normalize_positions(pos).clamp(0.0, 1.0).float()

        # Hash encoding
        hash_feat = self.hash_encoding(pos_norm).float()

        # Build extra features
        if self.input_mode == "legacy":
            sh_feat = self._encode_view_sh(view_dirs)
            extra = torch.cat([latent, sh_feat], dim=-1)
        elif self.input_mode == "implicit_scale":
            extra = self.encode_scale_pe(scales)

        # MLP
        mlp_in = torch.cat([hash_feat, extra], dim=-1)
        feat = self.mlp(mlp_in)

        if valid_idx is not None:
            out[valid_idx] = feat
        else:
            out = feat

        return out

    def _encode_view_sh(self, view_dirs):
        """SH encoding for view directions (legacy mode)."""
        x = view_dirs[:, 0:1]
        y = view_dirs[:, 1:2]
        z = view_dirs[:, 2:3]
        basis = [torch.ones_like(x)]
        if self.view_sh_degree >= 1:
            basis.extend([y, z, x])
        if self.view_sh_degree >= 2:
            basis.extend([x*y, y*z, 3*z*z - 1, x*z, x*x - y*y])
        return torch.cat(basis, dim=-1)

    def total_variation_loss(self):
        """Regularization loss on hash grid parameters."""
        params = self.hash_encoding.params
        if params.dim() == 1:
            # Approximate TV: penalize differences between adjacent entries
            tv = (params[1:] - params[:-1]).abs().mean()
        else:
            tv = torch.tensor(0.0, device=params.device)
        return tv

