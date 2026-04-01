"""
Explicit Per-Gaussian Feature Embedding (Architecture A)
========================================================
Stores a compact latent vector (default 64-d) on every Gaussian, renders it
via alpha-blending, then decodes to RADIO 1280-d in screen space.

Geometry is loaded from a pretrained 3DGS PLY and frozen; only the per-Gaussian
feature embeddings are optimized during distillation training.
"""

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData


class ExplicitFeatureGaussian(nn.Module):
    """Per-Gaussian learnable feature embedding with frozen 3DGS geometry.

    Frozen buffers (from PLY):
        _xyz            [N, 3]       Gaussian centres
        _rotation       [N, 4]       quaternions
        _scaling        [N, 2|3]     log-space scales (3DGS=3, 2DGS=2)
        _opacity        [N, 1]       logit opacities
        _features_dc    [N, 1, 3]    SH DC coefficients

    Learnable:
        _feature        [N, latent_dim]  per-Gaussian feature embedding
    """

    def __init__(self, latent_dim: int = 64, train_sh: bool = False) -> None:
        super().__init__()
        self._latent_dim = latent_dim
        self._train_sh = train_sh

        # Frozen geometry (registered as buffers → no grad, saved in state_dict)
        self.register_buffer("_xyz", torch.empty(0))
        self.register_buffer("_rotation", torch.empty(0))
        self.register_buffer("_scaling", torch.empty(0))
        self.register_buffer("_opacity", torch.empty(0))
        self.register_buffer("_features_dc", torch.empty(0))
        self.register_buffer("_features_rest", torch.empty(0))
        self._sh_degree = 0

        # Learnable feature embedding
        self._feature = nn.Parameter(torch.empty(0))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_gaussians(self) -> int:
        return self._xyz.shape[0]

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    # ------------------------------------------------------------------
    # Accessors (apply stored activations)
    # ------------------------------------------------------------------

    def get_features(self) -> torch.Tensor:
        """Return L2-normalised per-Gaussian features [N, latent_dim]."""
        return F.normalize(self._feature, p=2, dim=-1)

    def get_xyz(self) -> torch.Tensor:
        """Return Gaussian centres [N, 3]."""
        return self._xyz

    def get_opacity(self) -> torch.Tensor:
        """Return activated opacities [N, 1] via sigmoid."""
        return torch.sigmoid(self._opacity)

    def get_scaling(self) -> torch.Tensor:
        """Return activated scales [N, 2|3] via exp."""
        return torch.exp(self._scaling)

    def get_rotation(self) -> torch.Tensor:
        """Return unit quaternions [N, 4]."""
        return F.normalize(self._rotation, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def load_from_ply(self, ply_path: str) -> None:
        """Load frozen geometry from a pretrained 3DGS PLY and init features.

        Supports both 3DGS (3-component scaling) and 2DGS (2-component scaling),
        as well as PLYs with varying numbers of SH coefficients.
        """
        ply_path = str(ply_path)
        plydata = PlyData.read(ply_path)
        vertex = plydata.elements[0]
        N = vertex.count

        # -- xyz --------------------------------------------------------
        xyz = np.stack(
            [np.asarray(vertex["x"]),
             np.asarray(vertex["y"]),
             np.asarray(vertex["z"])],
            axis=1,
        )  # [N, 3]

        # -- opacity ----------------------------------------------------
        opacity = np.asarray(vertex["opacity"])[..., np.newaxis]  # [N, 1]

        # -- SH DC (only DC band; ignore higher-order coefficients) -----
        features_dc = np.zeros((N, 1, 3), dtype=np.float32)
        for i in range(3):
            prop_name = f"f_dc_{i}"
            if prop_name in [p.name for p in vertex.properties]:
                features_dc[:, 0, i] = np.asarray(vertex[prop_name])

        # -- SH rest (higher-order coefficients for RGB rendering) ------
        rest_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        n_rest = len(rest_names)
        if n_rest > 0:
            n_coeffs_per_ch = n_rest // 3
            features_rest = np.zeros((N, n_coeffs_per_ch, 3), dtype=np.float32)
            # Original 3DGS saves as transpose(1,2).flatten():
            # [ch0_c0, ch0_c1, ..., ch0_cK, ch1_c0, ..., ch2_cK]
            for i, rn in enumerate(rest_names):
                ch = i // n_coeffs_per_ch
                coeff_idx = i % n_coeffs_per_ch
                features_rest[:, coeff_idx, ch] = np.asarray(vertex[rn])
            import math
            sh_degree = int(math.sqrt(n_coeffs_per_ch + 1)) - 1
        else:
            features_rest = np.zeros((N, 0, 3), dtype=np.float32)
            sh_degree = 0

        # -- scaling (2 or 3 components) --------------------------------
        scale_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        scales = np.stack(
            [np.asarray(vertex[n]) for n in scale_names], axis=1
        )  # [N, 2|3]

        # -- rotation ---------------------------------------------------
        rot_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("rot_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        rots = np.stack(
            [np.asarray(vertex[n]) for n in rot_names], axis=1
        )  # [N, 4]

        # -- Store as frozen buffers (use .data to preserve buffer registration)
        device = self._xyz.device
        self._xyz = self._xyz.new_tensor(xyz)
        self._rotation = self._rotation.new_tensor(rots)
        self._scaling = self._scaling.new_tensor(scales)
        self._opacity = self._opacity.new_tensor(opacity)
        self._features_dc = self._features_dc.new_tensor(features_dc)
        self._features_rest = self._features_rest.new_tensor(features_rest)
        self._sh_degree = sh_degree
        # Re-register buffers since new_tensor creates new tensors
        self.register_buffer("_xyz", self._xyz)
        self.register_buffer("_rotation", self._rotation)
        self.register_buffer("_scaling", self._scaling)
        self.register_buffer("_opacity", self._opacity)
        self.register_buffer("_features_dc", self._features_dc)
        self.register_buffer("_features_rest", self._features_rest)

        # -- Default feature init ---------------------------------------
        self.init_features_random()

        print(
            f"[ExplicitFeatureGaussian] Loaded {N:,} Gaussians from {ply_path}\n"
            f"  scaling components: {scales.shape[1]}  |  latent_dim: {self._latent_dim}"
            f"  |  SH degree: {sh_degree}\n"
            f"  trainable params: {N * self._latent_dim:,}"
        )

    def init_features_random(self) -> None:
        """Initialise feature embeddings with small random values then L2-normalise."""
        N = self.num_gaussians
        if N == 0:
            raise RuntimeError("Load geometry (load_from_ply) before initialising features.")
        feat = torch.randn(N, self._latent_dim, device=self._xyz.device) * 0.01
        feat = F.normalize(feat, p=2, dim=-1)
        self._feature = nn.Parameter(feat)

    def init_features_from_data(self, feature_data: torch.Tensor) -> None:
        """Initialise features from pre-computed per-Gaussian vectors.

        Args:
            feature_data: [N, D] tensor. If D != latent_dim it is truncated /
                          zero-padded (e.g. PCA-projected RADIO features).
        """
        N = self.num_gaussians
        if N == 0:
            raise RuntimeError("Load geometry (load_from_ply) before initialising features.")
        if feature_data.shape[0] != N:
            raise ValueError(
                f"feature_data has {feature_data.shape[0]} rows but model has {N} Gaussians"
            )

        D = feature_data.shape[1]
        if D == self._latent_dim:
            feat = feature_data.clone()
        elif D > self._latent_dim:
            feat = feature_data[:, : self._latent_dim].clone()
        else:
            feat = torch.zeros(N, self._latent_dim, device=feature_data.device)
            feat[:, :D] = feature_data

        feat = F.normalize(feat.float(), p=2, dim=-1).to(self._xyz.device)
        self._feature = nn.Parameter(feat)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return learnable parameters (feature embeddings + optionally SH)."""
        params = [self._feature]
        if self._train_sh and hasattr(self, "_sh_dc_param"):
            params.append(self._sh_dc_param)
            if hasattr(self, "_sh_rest_param") and self._sh_rest_param is not None:
                params.append(self._sh_rest_param)
        return params

    def enable_sh_training(self) -> None:
        """Convert frozen SH buffers to trainable parameters."""
        self._train_sh = True
        self._sh_dc_param = nn.Parameter(self._features_dc.clone())
        if self._features_rest.numel() > 0:
            self._sh_rest_param = nn.Parameter(self._features_rest.clone())
        else:
            self._sh_rest_param = None
        print(f"[ExplicitFeatureGaussian] SH coefficients unfrozen "
              f"(DC: {self._sh_dc_param.shape}, "
              f"rest: {self._sh_rest_param.shape if self._sh_rest_param is not None else 'none'})")

    def get_sh_colors(self) -> torch.Tensor:
        """Return SH coefficients [N, K, 3] for RGB rendering.

        Uses trainable parameters if SH training is enabled, else frozen buffers.
        """
        if self._train_sh and hasattr(self, "_sh_dc_param"):
            dc = self._sh_dc_param
            rest = self._sh_rest_param
        else:
            dc = self._features_dc
            rest = self._features_rest

        if rest is not None and rest.numel() > 0:
            return torch.cat([dc, rest], dim=1)  # [N, K, 3]
        return dc  # [N, 1, 3]

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save feature weights and metadata."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "latent_dim": self._latent_dim,
                "feature": self._feature.data,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load feature weights from a checkpoint.

        Geometry must already be loaded via ``load_from_ply`` so that the
        Gaussian count is known and buffers are populated.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        saved_dim = ckpt["latent_dim"]
        if saved_dim != self._latent_dim:
            raise ValueError(
                f"Checkpoint latent_dim={saved_dim} but model latent_dim={self._latent_dim}"
            )
        feat = ckpt["feature"].to(self._xyz.device)
        if feat.shape[0] != self.num_gaussians:
            raise ValueError(
                f"Checkpoint has {feat.shape[0]} Gaussians but model has {self.num_gaussians}"
            )
        self._feature = nn.Parameter(feat)
