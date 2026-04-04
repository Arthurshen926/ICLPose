"""
gsplat-based feature field renderer for RADIO-GS.

Renders compact feature maps (e.g. 64-d) from 3D Gaussians via alpha-blending.
Uses channel chunking to stay within gsplat's CUDA shared-memory limits (~32
channels per rasterisation pass).  Supports both 3DGS and 2DGS surfel modes.

Requires gsplat >= 1.4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from gsplat import rasterization, rasterization_2dgs


class FeatureFieldRenderer(nn.Module):
    """Differentiable feature-field renderer backed by gsplat.

    The renderer is camera-aware: intrinsics are stored once at construction and
    reused across frames.  Call :meth:`render_features` for single-view or
    :meth:`render_features_batch` for multi-view rendering.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        max_channels_per_chunk: int = 32,
        use_2dgs: bool = False,
        background_color: float = 0.0,
        near_plane: float = 0.01,
        far_plane: float = 100.0,
    ) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_channels_per_chunk = max_channels_per_chunk
        self.use_2dgs = use_2dgs
        self.background_color = background_color
        self.near_plane = near_plane
        self.far_plane = far_plane

        # Intrinsic matrix stored as a buffer so it moves with .to(device)
        K = self._build_K_matrix(fx, fy, cx, cy)
        self.register_buffer("K", K)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_features(
        self,
        gaussian_model: nn.Module,
        viewmat: Tensor,
        feature_height: int | None = None,
        feature_width: int | None = None,
    ) -> dict[str, Tensor]:
        """Render a feature map from a single viewpoint.

        Args:
            gaussian_model: A model exposing ``get_xyz()``, ``get_rotation()``,
                ``get_scaling()``, ``get_opacity()``, and ``get_features()``.
            viewmat: [4, 4] world-to-camera matrix.
            feature_height: Output feature-map height (defaults to image_height).
            feature_width:  Output feature-map width  (defaults to image_width).

        Returns:
            dict with
                ``feature_map`` [D, fH, fW],
                ``depth_map``   [fH, fW],
                ``alpha_map``   [fH, fW].
        """
        fH = feature_height or self.image_height
        fW = feature_width or self.image_width

        means = gaussian_model.get_xyz()        # [N, 3]
        quats = gaussian_model.get_rotation()   # [N, 4]
        scales = gaussian_model.get_scaling()   # [N, 2|3]
        opacities = gaussian_model.get_opacity()  # [N, 1]
        features = gaussian_model.get_features()  # [N, D]

        N = means.shape[0]
        if N == 0:
            D = features.shape[1] if features.dim() == 2 else 0
            device = self.K.device
            return {
                "feature_map": torch.zeros(D, fH, fW, device=device),
                "depth_map": torch.zeros(fH, fW, device=device),
                "alpha_map": torch.zeros(fH, fW, device=device),
            }

        opacities = opacities.squeeze(-1)  # [N]

        # gsplat expects batched camera tensors: [C, 4, 4] and [C, 3, 3]
        viewmats = viewmat.unsqueeze(0)    # [1, 4, 4]
        Ks = self.K.unsqueeze(0)           # [1, 3, 3]

        feat_render, depth, alpha = self._chunk_render(
            means, quats, scales, opacities, features,
            viewmats, Ks, fW, fH,
        )
        # feat_render: [1, fH, fW, D], depth: [1, fH, fW, 1], alpha: [1, fH, fW, 1]

        feature_map = feat_render[0].permute(2, 0, 1)  # [D, fH, fW]
        depth_map = depth[0, :, :, 0]                  # [fH, fW]
        alpha_map = alpha[0, :, :, 0]                  # [fH, fW]

        return {
            "feature_map": feature_map,
            "depth_map": depth_map,
            "alpha_map": alpha_map,
        }

    def render_features_batch(
        self,
        gaussian_model: nn.Module,
        viewmats: Tensor,
        feature_height: int | None = None,
        feature_width: int | None = None,
    ) -> dict[str, Tensor]:
        """Batch-render feature maps from multiple viewpoints.

        Args:
            gaussian_model: Same interface as :meth:`render_features`.
            viewmats: [B, 4, 4] world-to-camera matrices.
            feature_height: Output height (defaults to image_height).
            feature_width:  Output width  (defaults to image_width).

        Returns:
            dict with
                ``feature_map`` [B, D, fH, fW],
                ``depth_map``   [B, fH, fW],
                ``alpha_map``   [B, fH, fW].
        """
        B = viewmats.shape[0]
        fH = feature_height or self.image_height
        fW = feature_width or self.image_width

        means = gaussian_model.get_xyz()
        quats = gaussian_model.get_rotation()
        scales = gaussian_model.get_scaling()
        opacities = gaussian_model.get_opacity().squeeze(-1)
        features = gaussian_model.get_features()

        N = means.shape[0]
        if N == 0:
            D = features.shape[1] if features.dim() == 2 else 0
            device = self.K.device
            return {
                "feature_map": torch.zeros(B, D, fH, fW, device=device),
                "depth_map": torch.zeros(B, fH, fW, device=device),
                "alpha_map": torch.zeros(B, fH, fW, device=device),
            }

        Ks = self.K.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 3]

        feat_render, depth, alpha = self._chunk_render(
            means, quats, scales, opacities, features,
            viewmats, Ks, fW, fH,
        )
        # feat_render: [B, fH, fW, D]

        feature_map = feat_render.permute(0, 3, 1, 2)  # [B, D, fH, fW]
        depth_map = depth[:, :, :, 0]                   # [B, fH, fW]
        alpha_map = alpha[:, :, :, 0]                    # [B, fH, fW]

        return {
            "feature_map": feature_map,
            "depth_map": depth_map,
            "alpha_map": alpha_map,
        }

    def render_features_and_rgb(
        self,
        gaussian_model: nn.Module,
        viewmats: Tensor,
        feature_height: int | None = None,
        feature_width: int | None = None,
    ) -> dict[str, Tensor]:
        """Render feature maps AND RGB simultaneously from the same model.

        Renders features at ``feature_height × feature_width`` resolution
        and RGB at ``image_height × image_width`` (or same as feature if None).

        Args:
            gaussian_model: Must expose ``get_sh_colors()`` in addition to
                the standard feature accessors.
            viewmats: [B, 4, 4] world-to-camera matrices.

        Returns:
            dict with ``feature_map``, ``rgb``, ``depth_map``, ``alpha_map``.
        """
        # Feature rendering (same as render_features_batch)
        feat_result = self.render_features_batch(
            gaussian_model, viewmats, feature_height, feature_width
        )

        # RGB rendering at feature resolution for guide signal
        B = viewmats.shape[0]
        fH = feature_height or self.image_height
        fW = feature_width or self.image_width

        means = gaussian_model.get_xyz()
        quats = gaussian_model.get_rotation()
        scales = gaussian_model.get_scaling()
        opacities = gaussian_model.get_opacity().squeeze(-1)

        sh_degree = getattr(gaussian_model, "_sh_degree", 0)
        if hasattr(gaussian_model, "get_sh_colors"):
            colors = gaussian_model.get_sh_colors()  # [N, K, 3]
        else:
            C0 = 0.28209479177387814
            colors = (gaussian_model._features_dc[:, 0, :] * C0 + 0.5).clamp(0.0, 1.0)
            sh_degree = None

        if self.use_2dgs and scales.shape[-1] == 2:
            pad = torch.full(
                (scales.shape[0], 1), -10.0,
                device=scales.device, dtype=scales.dtype,
            )
            scales = torch.cat([scales, pad], dim=-1)

        Ks = self.K.unsqueeze(0).expand(B, -1, -1)
        bg = torch.zeros(B, 3, device=means.device)
        bg_rgbed = torch.zeros(B, 4, device=means.device) if self.use_2dgs else None

        # Render RGB per-view (gsplat SH eval requires per-view)
        rgb_list = []
        geom_depth_list = []
        geom_alpha_list = []
        for b in range(B):
            raster_fn = rasterization_2dgs if self.use_2dgs else rasterization
            if self.use_2dgs:
                renders, alphas, _normals, _nd, _distort, _median, info = raster_fn(
                    means=means, quats=quats,
                    scales=torch.exp(scales) if not hasattr(gaussian_model, 'get_scaling') else scales,
                    opacities=opacities, colors=colors,
                    viewmats=viewmats[b:b+1], Ks=Ks[b:b+1],
                    width=fW, height=fH,
                    near_plane=self.near_plane, far_plane=self.far_plane,
                    backgrounds=bg_rgbed[b:b+1],
                    render_mode="RGB+ED",
                    sh_degree=sh_degree if sh_degree and sh_degree > 0 else None,
                )
            else:
                renders, alphas, info = raster_fn(
                    means=means, quats=quats, scales=scales,
                    opacities=opacities, colors=colors,
                    viewmats=viewmats[b:b+1], Ks=Ks[b:b+1],
                    width=fW, height=fH,
                    near_plane=self.near_plane, far_plane=self.far_plane,
                    backgrounds=bg[b:b+1],
                    sh_degree=sh_degree if sh_degree and sh_degree > 0 else None,
                )
            rgb_render = renders[0, ..., :3] if self.use_2dgs else renders[0]
            rgb_list.append(rgb_render.permute(2, 0, 1).clamp(0.0, 1.0))
            geom_alpha_list.append(alphas[0, :, :, 0])
            if self.use_2dgs:
                geom_depth_list.append(renders[0, :, :, 3])

        feat_result["rgb"] = torch.stack(rgb_list, dim=0)  # [B, 3, fH, fW]
        feat_result["geom_alpha"] = torch.stack(geom_alpha_list, dim=0)  # [B, fH, fW]
        if geom_depth_list:
            feat_result["geom_depth"] = torch.stack(geom_depth_list, dim=0)  # [B, fH, fW]
        return feat_result

    def render_rgb(
        self,
        gaussian_model: nn.Module,
        viewmat: Tensor,
    ) -> dict[str, Tensor]:
        """Standard RGB rendering using SH coefficients.

        Uses full spherical harmonics if available (degree > 0), otherwise
        falls back to DC-only rendering.

        Args:
            gaussian_model: Must expose ``_features_dc`` [N, 1, 3] and optionally
                ``_features_rest`` [N, K, 3] and ``_sh_degree`` int.
            viewmat: [4, 4] world-to-camera matrix.

        Returns:
            dict with ``rgb`` [3, H, W], ``depth`` [H, W], ``alpha`` [H, W].
        """
        means = gaussian_model.get_xyz()
        quats = gaussian_model.get_rotation()
        scales = gaussian_model.get_scaling()
        opacities = gaussian_model.get_opacity().squeeze(-1)

        N = means.shape[0]
        H, W = self.image_height, self.image_width
        device = self.K.device

        if N == 0:
            return {
                "rgb": torch.zeros(3, H, W, device=device),
                "depth": torch.zeros(H, W, device=device),
                "alpha": torch.zeros(H, W, device=device),
            }

        # Build SH colors: [N, K, 3] where K = (degree+1)^2
        sh_degree = getattr(gaussian_model, "_sh_degree", 0)
        features_rest = getattr(gaussian_model, "_features_rest", None)
        if sh_degree > 0 and features_rest is not None and features_rest.numel() > 0:
            colors = torch.cat([gaussian_model._features_dc, features_rest], dim=1)  # [N, K, 3]
        else:
            # DC-only fallback
            C0 = 0.28209479177387814
            colors = (gaussian_model._features_dc[:, 0, :] * C0 + 0.5).clamp(0.0, 1.0)  # [N, 3]
            sh_degree = None  # tell gsplat no SH eval needed

        viewmats_b = viewmat.unsqueeze(0)  # [1, 4, 4]
        Ks = self.K.unsqueeze(0)           # [1, 3, 3]

        bg = torch.ones(3, device=device)
        bg_rgbed = torch.tensor([1.0, 1.0, 1.0, 0.0], device=device) if self.use_2dgs else None

        raster_fn = rasterization_2dgs if self.use_2dgs else rasterization

        if self.use_2dgs and scales.shape[-1] == 2:
            pad = torch.full(
                (scales.shape[0], 1), -10.0,
                device=scales.device, dtype=scales.dtype,
            )
            scales = torch.cat([scales, pad], dim=-1)

        if self.use_2dgs:
            renders, alphas, _normals, _nd, _distort, _median, info = raster_fn(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats_b,
                Ks=Ks,
                width=W,
                height=H,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                backgrounds=bg_rgbed.unsqueeze(0),
                render_mode="RGB+ED",
                sh_degree=sh_degree,
            )
        else:
            renders, alphas, info = raster_fn(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats_b,
                Ks=Ks,
                width=W,
                height=H,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                backgrounds=bg.unsqueeze(0),
                sh_degree=sh_degree,
            )
        # renders: [1, H, W, 3], alphas: [1, H, W, 1]

        rgb_render = renders[0, ..., :3] if self.use_2dgs else renders[0]
        rgb = rgb_render.permute(2, 0, 1).clamp(0.0, 1.0)   # [3, H, W]
        alpha = alphas[0, :, :, 0]           # [H, W]

        depth_map = torch.zeros(H, W, device=device)
        if self.use_2dgs:
            depth_map = renders[0, :, :, 3]
        elif "depths" in info:
            depth_map = info["depths"][0, :, :, 0]

        return {"rgb": rgb, "depth": depth_map, "alpha": alpha}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_render(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        features: Tensor,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Render high-dimensional features via channel chunking.

        Splits the D-dimensional feature vectors into chunks of at most
        ``max_channels_per_chunk`` channels, rasterises each chunk
        independently, and concatenates the results.

        Returns:
            feature_render: [C, H, W, D]  (C = number of cameras)
            depth:          [C, H, W, 1]
            alpha:          [C, H, W, 1]
        """
        D = features.shape[1]
        chunk_size = self.max_channels_per_chunk
        n_chunks = (D + chunk_size - 1) // chunk_size

        raster_fn = rasterization_2dgs if self.use_2dgs else rasterization

        # gsplat rasterization_2dgs requires [N, 3] scales; 2DGS PLY files
        # store only 2 components.  Pad with a small third scale if needed.
        if self.use_2dgs and scales.shape[-1] == 2:
            pad = torch.full(
                (scales.shape[0], 1), -10.0,
                device=scales.device, dtype=scales.dtype,
            )
            scales = torch.cat([scales, pad], dim=-1)

        rendered_chunks: list[Tensor] = []
        depth_out: Tensor | None = None
        alpha_out: Tensor | None = None

        for i in range(n_chunks):
            c_start = i * chunk_size
            c_end = min(c_start + chunk_size, D)
            chunk_feat = features[:, c_start:c_end]  # [N, c_dim]

            bg = torch.full(
                (chunk_feat.shape[1],),
                self.background_color,
                device=features.device,
            )

            if self.use_2dgs:
                renders, alphas, _normals, _nd, _distort, _median, info = raster_fn(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=chunk_feat,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=width,
                    height=height,
                    near_plane=self.near_plane,
                    far_plane=self.far_plane,
                    backgrounds=bg.unsqueeze(0).expand(viewmats.shape[0], -1),
                )
            else:
                _median = None
                renders, alphas, info = raster_fn(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=chunk_feat,
                    viewmats=viewmats,
                    Ks=Ks,
                    width=width,
                    height=height,
                    near_plane=self.near_plane,
                    far_plane=self.far_plane,
                    backgrounds=bg.unsqueeze(0).expand(viewmats.shape[0], -1),
                )
            # renders: [C, H, W, c_dim], alphas: [C, H, W, 1]

            rendered_chunks.append(renders)

            # Capture depth and alpha from the first chunk only
            if i == 0:
                alpha_out = alphas
                if _median is not None:
                    depth_out = _median
                elif "depths" in info:
                    depth_out = info["depths"]

        # Concatenate feature chunks along the channel dimension
        feature_render = torch.cat(rendered_chunks, dim=-1)  # [C, H, W, D]

        C = viewmats.shape[0]
        if depth_out is None:
            depth_out = torch.zeros(C, height, width, 1, device=features.device)
        if alpha_out is None:
            alpha_out = torch.zeros(C, height, width, 1, device=features.device)

        return feature_render, depth_out, alpha_out

    @staticmethod
    def _build_K_matrix(fx: float, fy: float, cx: float, cy: float) -> Tensor:
        """Construct a [3, 3] camera intrinsic matrix."""
        return torch.tensor(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
