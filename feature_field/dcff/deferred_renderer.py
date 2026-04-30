from __future__ import annotations

"""
Deferred Cascaded Renderer — Screen-space feature decode pipeline.

This is the computational core of the DCFF architecture.
Instead of querying 3D features for ALL Gaussians (O(N_splats)),
we first rasterize to screen space, then decode features only for
visible pixels (O(N_pixels)). This is dramatically more efficient.

Pipeline:
  Step 1. Rasterize 2DGS → RGB, depth, alpha, normals (standard)
  Step 2. Rasterize 16d latent z_i → Z_map (inherits sharp 2DGS edges)
  Step 3. Fine features  = Conv1x1(Z_map) → 64d (explicit, sharp boundaries)
  Step 4. Coarse features = HashGrid(Pos_map) + MLP(hash, z, view_dir) → 64d (implicit, smooth)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat import rasterization
try:
    from gsplat import rasterization_2dgs
except ImportError:
    rasterization_2dgs = rasterization


def encode_view_directions_sh(view_dirs: torch.Tensor, degree: int = 2) -> torch.Tensor:
    """Compact real SH-like encoding up to degree 2 for view directions."""
    if degree < 0 or degree > 2:
        raise ValueError(f"Only degree 0-2 supported, got degree={degree}")

    x = view_dirs[..., 0:1]
    y = view_dirs[..., 1:2]
    z = view_dirs[..., 2:3]

    basis = [torch.ones_like(x)]
    if degree >= 1:
        basis.extend([y, z, x])
    if degree >= 2:
        basis.extend([
            x * y,
            y * z,
            3.0 * z * z - 1.0,
            x * z,
            x * x - y * y,
        ])
    return torch.cat(basis, dim=-1)


class FineDecoder(nn.Module):
    """Explicit fine decoder for geometric features.

    By default this matches the legacy latent-only 1×1 Conv MLP. When
    `use_viewdirs=True`, it decodes from explicit latent + per-pixel view SH.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        feature_dim: int = 64,
        hidden_dim: int = None,
        num_layers: int = 3,
        use_viewdirs: bool = False,
        view_degree: int = 2,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feature_dim
        self.use_viewdirs = use_viewdirs
        self.view_degree = view_degree
        self.view_dim = 0 if not use_viewdirs else sum(2 * order + 1 for order in range(view_degree + 1))
        input_dim = latent_dim + self.view_dim

        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")

        layers = []
        prev = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Conv2d(prev, hidden_dim, 1), nn.GELU()])
            prev = hidden_dim
        layers.append(nn.Conv2d(prev, feature_dim, 1))
        self.decoder = nn.Sequential(*layers)

        n_params = sum(p.numel() for p in self.parameters())
        mode = 'latent+view' if use_viewdirs else 'latent-only'
        print(
            f"  [FineDecoder] mode={mode}, input={input_dim}d, "
            f"hidden={hidden_dim}, layers={num_layers}, params={n_params:,}"
        )

    def forward(self, z_map: torch.Tensor, view_dir_map: torch.Tensor = None) -> torch.Tensor:
        """Decode fine features from explicit latent map and optional view encoding."""
        if self.use_viewdirs:
            if view_dir_map is None:
                raise ValueError("FineDecoder requires view_dir_map when use_viewdirs=True")
            z_map = torch.cat([z_map, view_dir_map], dim=1)
        return self.decoder(z_map)


class SpatialFineDecoder(nn.Module):
    """Fine decoder with 3×3 spatial convolutions for neighborhood context.

    Unlike the 1×1 FineDecoder, this uses dilated 3×3 convolutions to give
    each pixel access to its spatial neighborhood (5×5 effective receptive field).
    This helps produce smoother, more coherent feature maps.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        use_viewdirs: bool = False,
        view_degree: int = 2,
    ):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.view_degree = view_degree
        self.view_dim = 0 if not use_viewdirs else sum(2 * order + 1 for order in range(view_degree + 1))
        input_dim = latent_dim + self.view_dim

        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feature_dim, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        mode = 'latent+view' if use_viewdirs else 'latent-only'
        print(
            f"  [SpatialFineDecoder] mode={mode}, input={input_dim}d, "
            f"hidden={hidden_dim}, receptive_field=5×5, params={n_params:,}"
        )

    def forward(self, z_map: torch.Tensor, view_dir_map: torch.Tensor = None) -> torch.Tensor:
        if self.use_viewdirs:
            if view_dir_map is None:
                raise ValueError("SpatialFineDecoder requires view_dir_map when use_viewdirs=True")
            z_map = torch.cat([z_map, view_dir_map], dim=1)
        return self.decoder(z_map)


def _group_norm_groups(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualSpatialBlock(nn.Module):
    """Small residual conv block for preserving high-frequency feature detail."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = int(dilation)
        groups = _group_norm_groups(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=padding, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResidualSpatialFineDecoder(nn.Module):
    """Higher-capacity fine decoder with residual 3x3 spatial context."""

    def __init__(
        self,
        latent_dim: int = 32,
        feature_dim: int = 64,
        hidden_dim: int = 192,
        num_blocks: int = 3,
        use_viewdirs: bool = False,
        view_degree: int = 2,
    ):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.view_degree = view_degree
        self.view_dim = 0 if not use_viewdirs else sum(2 * order + 1 for order in range(view_degree + 1))
        input_dim = latent_dim + self.view_dim
        groups = _group_norm_groups(hidden_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        dilations = [1, 2, 1, 2]
        self.blocks = nn.Sequential(*[
            ResidualSpatialBlock(hidden_dim, dilation=dilations[i % len(dilations)])
            for i in range(max(1, int(num_blocks)))
        ])
        self.head = nn.Conv2d(hidden_dim, feature_dim, 1)

        n_params = sum(p.numel() for p in self.parameters())
        mode = 'latent+view' if use_viewdirs else 'latent-only'
        print(
            f"  [ResidualSpatialFineDecoder] mode={mode}, input={input_dim}d, "
            f"hidden={hidden_dim}, blocks={max(1, int(num_blocks))}, params={n_params:,}"
        )

    def forward(self, z_map: torch.Tensor, view_dir_map: torch.Tensor = None) -> torch.Tensor:
        if self.use_viewdirs:
            if view_dir_map is None:
                raise ValueError("ResidualSpatialFineDecoder requires view_dir_map when use_viewdirs=True")
            z_map = torch.cat([z_map, view_dir_map], dim=1)
        x = self.stem(z_map)
        x = self.blocks(x)
        return self.head(x)


class CarrierResidualCoarseFusion(nn.Module):
    """Carrier-conditioned residual on top of the implicit coarse field.

    Starts as a conservative, trainable residual on top of the legacy implicit
    coarse path. The gate is biased low instead of zeroing both factors; a
    product of two zero-initialized branches is a dead path with no gradient.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        feature_dim: int = 64,
        carrier_hidden_dim: int = 128,
        gate_hidden_dim: int = 64,
        forward_batch_chunk_size: int = 0,
    ):
        super().__init__()
        self.forward_batch_chunk_size = int(forward_batch_chunk_size or 0)
        self.carrier_proj = nn.Sequential(
            nn.Conv2d(latent_dim, carrier_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(carrier_hidden_dim, feature_dim, 1),
        )
        self.residual_gate = nn.Sequential(
            nn.Conv2d(feature_dim * 2, gate_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden_dim, 1, 1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.residual_gate[-2].weight)
        nn.init.constant_(self.residual_gate[-2].bias, -2.0)

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"  [CarrierResidualCoarseFusion] latent={latent_dim}d, feature={feature_dim}d, "
            f"carrier_hidden={carrier_hidden_dim}, gate_hidden={gate_hidden_dim}, params={n_params:,}"
        )

    def forward(
        self,
        z_map: torch.Tensor,
        implicit_coarse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk = self.forward_batch_chunk_size
        if chunk > 0 and z_map.shape[0] > chunk:
            fused_out = None
            for start in range(0, z_map.shape[0], chunk):
                end = min(start + chunk, z_map.shape[0])
                fused, carrier, gate = self.forward(
                    z_map[start:end],
                    implicit_coarse[start:end],
                )
                if fused_out is None:
                    fused_out = fused.new_empty(
                        z_map.shape[0],
                        fused.shape[1],
                        fused.shape[2],
                        fused.shape[3],
                    )
                fused_out[start:end] = fused
            return fused_out, None, None

        carrier_coarse = self.carrier_proj(z_map)
        gate = self.residual_gate(torch.cat([carrier_coarse, implicit_coarse], dim=1))
        fused = implicit_coarse + self.residual_scale * gate * carrier_coarse
        return fused, carrier_coarse, gate


class SpatialDirectCoarseDecoder(nn.Module):
    """Direct spatial decoder for coarse features from rasterized coarse latent.

    This mode is useful when an implicit position MLP collapses to low-frequency
    view gradients and the coarse target still contains object/layout structure.
    """

    def __init__(
        self,
        latent_dim: int = 24,
        feature_dim: int = 64,
        hidden_dim: int = 160,
    ):
        super().__init__()
        self.decoder = SpatialFineDecoder(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            use_viewdirs=False,
        )

    def forward(
        self,
        z_map: torch.Tensor,
        implicit_coarse: torch.Tensor = None,
    ) -> tuple[torch.Tensor, None, None]:
        return self.decoder(z_map), None, None


class DeferredCascadedRenderer(nn.Module):
    """Screen-space deferred rendering for dual-scale feature extraction.

    Renders 2DGS surfels to screen, then decodes fine (explicit) and
    coarse (implicit) features in 2D screen space.
    """

    def __init__(
        self,
        hash_grid: nn.Module,
        latent_dim: int = 16,
        fine_latent_dim: int | None = None,
        coarse_latent_dim: int | None = None,
        fine_feature_dim: int = 64,
        coarse_feature_dim: int = 64,
        fine_hidden_dim: int = None,
        fine_num_layers: int = 3,
        fine_use_viewdirs: bool = False,
        fine_view_degree: int = 2,
        fine_decoder_type: str = 'pointwise',
        coarse_mode: str = 'implicit_only',
        coarse_carrier_hidden_dim: int | None = None,
        coarse_gate_hidden_dim: int | None = None,
        coarse_forward_batch_chunk_size: int = 0,
        chunk_size: int = 16,
        coarse_smoothing_kernel: int = 1,
    ):
        super().__init__()
        self.hash_grid = hash_grid
        self.latent_dim = latent_dim
        if fine_latent_dim is None and coarse_latent_dim is None:
            self.fine_latent_dim = latent_dim
            self.coarse_latent_dim = latent_dim
            self.split_latent = False
        else:
            if fine_latent_dim is None:
                fine_latent_dim = latent_dim - int(coarse_latent_dim)
            if coarse_latent_dim is None:
                coarse_latent_dim = latent_dim - int(fine_latent_dim)
            self.fine_latent_dim = int(fine_latent_dim)
            self.coarse_latent_dim = int(coarse_latent_dim)
            if self.fine_latent_dim <= 0 or self.coarse_latent_dim <= 0:
                raise ValueError(
                    f"Invalid split latent dims: fine={self.fine_latent_dim}, "
                    f"coarse={self.coarse_latent_dim}"
                )
            if self.fine_latent_dim + self.coarse_latent_dim != latent_dim:
                raise ValueError(
                    f"fine_latent_dim + coarse_latent_dim must equal latent_dim "
                    f"({self.fine_latent_dim}+{self.coarse_latent_dim}!={latent_dim})"
                )
            self.split_latent = True
        if fine_decoder_type == 'spatial':
            self.fine_decoder = SpatialFineDecoder(
                latent_dim=self.fine_latent_dim,
                feature_dim=fine_feature_dim,
                hidden_dim=fine_hidden_dim or 128,
                use_viewdirs=fine_use_viewdirs,
                view_degree=fine_view_degree,
            )
        elif fine_decoder_type == 'residual_spatial':
            self.fine_decoder = ResidualSpatialFineDecoder(
                latent_dim=self.fine_latent_dim,
                feature_dim=fine_feature_dim,
                hidden_dim=fine_hidden_dim or 192,
                num_blocks=fine_num_layers,
                use_viewdirs=fine_use_viewdirs,
                view_degree=fine_view_degree,
            )
        else:
            self.fine_decoder = FineDecoder(
                latent_dim=self.fine_latent_dim,
                feature_dim=fine_feature_dim,
                hidden_dim=fine_hidden_dim,
                num_layers=fine_num_layers,
                use_viewdirs=fine_use_viewdirs,
                view_degree=fine_view_degree,
            )
        self.fine_feature_dim = fine_feature_dim
        self.coarse_feature_dim = coarse_feature_dim
        self.coarse_mode = coarse_mode
        self.coarse_direct_uses_full_latent = coarse_mode == 'spatial_full_direct'
        self.chunk_size = chunk_size
        kernel = max(1, int(coarse_smoothing_kernel))
        if kernel % 2 == 0:
            kernel += 1
        self.coarse_smoothing_kernel = kernel

        self.coarse_carrier_fusion = None
        if coarse_mode == 'carrier_residual':
            self.coarse_carrier_fusion = CarrierResidualCoarseFusion(
                latent_dim=self.coarse_latent_dim,
                feature_dim=coarse_feature_dim,
                carrier_hidden_dim=coarse_carrier_hidden_dim or max(coarse_feature_dim, self.coarse_latent_dim),
                gate_hidden_dim=coarse_gate_hidden_dim or max(coarse_feature_dim, self.coarse_latent_dim),
                forward_batch_chunk_size=coarse_forward_batch_chunk_size,
            )
        elif coarse_mode == 'spatial_direct':
            self.coarse_carrier_fusion = SpatialDirectCoarseDecoder(
                latent_dim=self.coarse_latent_dim,
                feature_dim=coarse_feature_dim,
                hidden_dim=coarse_carrier_hidden_dim or max(128, coarse_feature_dim),
            )
        elif coarse_mode == 'spatial_full_direct':
            self.coarse_carrier_fusion = SpatialDirectCoarseDecoder(
                latent_dim=latent_dim,
                feature_dim=coarse_feature_dim,
                hidden_dim=coarse_carrier_hidden_dim or max(128, coarse_feature_dim),
            )
        elif coarse_mode != 'implicit_only':
            raise ValueError(f"Unsupported coarse_mode '{coarse_mode}'")

    def render_rgb_depth(
        self,
        means3d: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        sh_degree: int = 0,
    ):
        """Standard 2DGS RGB + depth rendering.

        Args:
            means3d: [N, 3]
            quats: [N, 4] (normalized)
            scales: [N, 3] (padded for 2DGS)
            opacities: [N, 1]
            colors: SH coefficients [N, K, 3]
            viewmat: [1, 4, 4] or [4, 4]
            K: [1, 3, 3] or [3, 3]

        Returns:
            dict with 'rgb' [1,3,H,W], 'depth' [1,1,H,W], 'alpha' [1,1,H,W],
            'normals' [1,3,H,W], 'meta' dict
        """
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)
        if K.dim() == 2:
            K = K.unsqueeze(0)

        if rasterization_2dgs is rasterization:
            render_colors, render_alphas, meta = rasterization(
                means=means3d,
                quats=quats,
                scales=scales,
                opacities=opacities.squeeze(-1) if opacities.dim() == 2 else opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K,
                width=width,
                height=height,
                packed=True,
                near_plane=0.01,
                far_plane=1e5,
                render_mode='RGB+ED',
                sh_degree=sh_degree,
                absgrad=True,
            )
            normals = surf_normals = distort = median_depth = None
        else:
            render_colors, render_alphas, normals, surf_normals, distort, median_depth, meta = \
                rasterization_2dgs(
                    means=means3d,
                    quats=quats,
                    scales=scales,
                    opacities=opacities.squeeze(-1) if opacities.dim() == 2 else opacities,
                    colors=colors,
                    viewmats=viewmat,
                    Ks=K,
                    width=width,
                    height=height,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e5,
                    render_mode='RGB+ED',
                    sh_degree=sh_degree,
                    absgrad=True,
                )

        # Retain grad on means2d for densification
        if 'means2d' in meta:
            try:
                meta['means2d'].retain_grad()
            except Exception:
                pass

        # render_colors: [1, H, W, 4] → last channel is expected depth
        rgb = render_colors[..., :3].permute(0, 3, 1, 2)   # [1, 3, H, W]
        depth = render_colors[..., 3:4].permute(0, 3, 1, 2)  # [1, 1, H, W]
        alpha = render_alphas.permute(0, 3, 1, 2)            # [1, 1, H, W]
        norm = normals.permute(0, 3, 1, 2) if normals is not None else None

        return {
            'rgb': rgb,
            'depth': depth,
            'alpha': alpha,
            'normals': norm,
            'surf_normals': surf_normals,
            'distort': distort,
            'median_depth': median_depth,
            'meta': meta,
        }

    def render_attribute_map(
        self,
        means3d: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        attributes: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
    ):
        """Rasterize arbitrary per-Gaussian attributes to screen space."""
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)
        if K.dim() == 2:
            K = K.unsqueeze(0)

        D = attributes.shape[1]
        n_chunks = (D + self.chunk_size - 1) // self.chunk_size
        chunks = []

        for i in range(n_chunks):
            c_start = i * self.chunk_size
            c_end = min((i + 1) * self.chunk_size, D)
            if rasterization_2dgs is rasterization:
                rc, ra, _meta = rasterization(
                    means=means3d,
                    quats=quats,
                    scales=scales,
                    opacities=opacities.squeeze(-1) if opacities.dim() == 2 else opacities,
                    colors=attributes[:, c_start:c_end],
                    viewmats=viewmat,
                    Ks=K,
                    width=width,
                    height=height,
                    packed=True,
                    near_plane=0.01,
                    far_plane=1e5,
                    render_mode='RGB',
                )
            else:
                rc, ra, *_ = rasterization_2dgs(
                    means=means3d,
                    quats=quats,
                    scales=scales,
                    opacities=opacities.squeeze(-1) if opacities.dim() == 2 else opacities,
                    colors=attributes[:, c_start:c_end],
                    viewmats=viewmat,
                    Ks=K,
                    width=width,
                    height=height,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e5,
                    render_mode='RGB',
                )
            chunks.append(rc)

        z_map = torch.cat(chunks, dim=-1)  # [1, H, W, D]
        z_map = z_map.permute(0, 3, 1, 2)  # [1, D, H, W]
        return z_map

    def render_latent(
        self,
        means3d: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        latent: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
    ):
        """Rasterize per-Gaussian latent vectors to screen space."""
        return self.render_attribute_map(
            means3d, quats, scales, opacities, latent, viewmat, K, width, height,
        )

    def render_scales(
        self,
        means3d: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        opacities: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
    ):
        """Rasterize per-Gaussian 2D scales to screen space."""
        return self.render_attribute_map(
            means3d, quats, scales, opacities, scales[:, :2], viewmat, K, width, height,
        )

    @staticmethod
    def depth_to_position_map(
        depth: torch.Tensor,
        K: torch.Tensor,
        viewmat: torch.Tensor,
    ) -> torch.Tensor:
        """Unproject depth map to world-space 3D position map.

        Args:
            depth: [B, 1, H, W], [B, H, W], or [H, W] depth values
            K: [B, 3, 3] or [3, 3] camera intrinsics
            viewmat: [B, 4, 4] or [4, 4] world-to-camera transforms

        Returns:
            position_map: [B, H, W, 3] or [H, W, 3] world-space coordinates
        """
        squeeze_batch = False
        if depth.dim() == 4:
            depth = depth.squeeze(1)
        elif depth.dim() == 2:
            depth = depth.unsqueeze(0)
            squeeze_batch = True

        if K.dim() == 2:
            K = K.unsqueeze(0)
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)

        B, H, W = depth.shape
        device = depth.device

        # Create pixel grid
        v_coords, u_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)
        v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)

        # Unproject to camera space
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)
        z = depth
        x_cam = (u_coords - cx) / fx * z
        y_cam = (v_coords - cy) / fy * z
        cam_pts = torch.stack([x_cam, y_cam, z], dim=-1)  # [B, H, W, 3]

        # Camera-to-world transform
        c2w = torch.inverse(viewmat)  # [4, 4]
        R = c2w[:, :3, :3]  # [B, 3, 3]
        t = c2w[:, :3, 3]   # [B, 3]

        # Transform to world space
        world_pts = torch.einsum('bhwj,bij->bhwi', cam_pts, R) + t[:, None, None, :]
        return world_pts[0] if squeeze_batch else world_pts

    @staticmethod
    def compute_view_directions(
        position_map: torch.Tensor,
        viewmat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-pixel view directions from camera center to 3D points.

        Args:
            position_map: [B, H, W, 3] or [H, W, 3] world-space positions
            viewmat: [B, 4, 4] or [4, 4] world-to-camera transforms

        Returns:
            view_dirs: same spatial shape as position_map
        """
        squeeze_batch = False
        if position_map.dim() == 3:
            position_map = position_map.unsqueeze(0)
            squeeze_batch = True
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)

        c2w = torch.inverse(viewmat)
        cam_center = c2w[:, :3, 3]  # [B, 3]
        view_dirs = position_map - cam_center[:, None, None, :]
        view_dirs = F.normalize(view_dirs, dim=-1)
        return view_dirs[0] if squeeze_batch else view_dirs

    def decode_fine(
        self,
        z_map: torch.Tensor,
        position_map: torch.Tensor = None,
        viewmat: torch.Tensor = None,
    ) -> torch.Tensor:
        """Decode fine geometric features from rasterized latent map.

        Args:
            z_map: [B, latent_dim, H, W]
        Returns:
            fine_features: [B, fine_feature_dim, H, W]
        """
        view_dir_map = None
        if self.fine_decoder.use_viewdirs:
            if position_map is None or viewmat is None:
                raise ValueError("decode_fine requires position_map and viewmat when use_viewdirs=True")
            view_dirs = self.compute_view_directions(position_map, viewmat)
            view_dir_map = encode_view_directions_sh(
                view_dirs,
                degree=self.fine_decoder.view_degree,
            ).permute(0, 3, 1, 2)
        return self.fine_decoder(z_map, view_dir_map=view_dir_map)

    def decode_coarse(
        self,
        position_map: torch.Tensor,
        alpha: torch.Tensor,
        z_map: torch.Tensor = None,
        scale_map: torch.Tensor = None,
        viewmat: torch.Tensor = None,
        alpha_threshold: float = 0.5,
    ) -> torch.Tensor:
        """Decode coarse semantic features via hash grid + MLP.

        Only queries the hash grid for visible pixels (alpha > threshold),
        saving computation on sky/background regions.

        Args:
            position_map: [H, W, 3] world-space positions
            alpha: [1, 1, H, W] rendered alpha
            z_map: [1, latent_dim, H, W] rasterized latent (legacy mode)
            scale_map: [1, 2, H, W] rasterized Gaussian scales (implicit_scale mode)
            viewmat: [4, 4] world-to-camera (legacy mode)
            alpha_threshold: minimum alpha to consider pixel valid

        Returns:
            coarse_features: [1, coarse_feature_dim, H, W]
        """
        B, H, W = position_map.shape[:3]

        # Validity mask: only decode for visible pixels
        valid = alpha.squeeze(1) > alpha_threshold  # [B, H, W]

        # Flatten spatial dims
        pos_flat = position_map.reshape(-1, 3)       # [BHW, 3]
        valid_flat = valid.reshape(-1)               # [BHW]

        if getattr(self.hash_grid, 'input_mode', 'legacy') == 'legacy':
            if z_map is None or viewmat is None:
                raise ValueError("Legacy coarse decoding requires z_map and viewmat")
            view_dirs = self.compute_view_directions(position_map, viewmat)
            z_flat = z_map.permute(0, 2, 3, 1).reshape(-1, z_map.shape[1])
            vd_flat = view_dirs.reshape(-1, 3)
            coarse_flat = self.hash_grid(
                positions=pos_flat,
                latent=z_flat,
                view_dirs=vd_flat,
                valid_mask=valid_flat,
            )
        else:
            if scale_map is None:
                raise ValueError("implicit_scale coarse decoding requires scale_map")
            scale_flat = scale_map.permute(0, 2, 3, 1).reshape(-1, scale_map.shape[1])
            coarse_flat = self.hash_grid(
                positions=pos_flat,
                scales=scale_flat,
                valid_mask=valid_flat,
            )

        # Reshape to spatial
        coarse = coarse_flat.reshape(B, H, W, self.coarse_feature_dim)
        coarse = coarse.permute(0, 3, 1, 2)  # [B, C, H, W]
        if self.coarse_smoothing_kernel > 1:
            pad = self.coarse_smoothing_kernel // 2
            coarse = F.avg_pool2d(
                coarse,
                kernel_size=self.coarse_smoothing_kernel,
                stride=1,
                padding=pad,
            )
        return coarse

    def forward(
        self,
        gaussians,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        render_coarse: bool = True,
        feature_height: int = None,
        feature_width: int = None,
    ):
        """Full deferred cascaded rendering pipeline.

        Args:
            gaussians: HybridGaussianModel instance
            viewmat: [4, 4] world-to-camera transform
            K: [3, 3] camera intrinsics
            width, height: rendering resolution
            render_coarse: whether to decode coarse features (Phase 3)
            feature_height, feature_width: target resolution for features
                (if different from rendering resolution)

        Returns:
            dict with:
                'rgb': [1, 3, H, W]
                'depth': [1, 1, H, W]
                'alpha': [1, 1, H, W]
                'normals': [1, 3, H, W]
                'z_map': [1, latent_dim, H, W]
                'fine_features': [1, fine_feat_dim, fH, fW]
                'coarse_features': [1, coarse_feat_dim, fH, fW] (if render_coarse)
                'meta': rasterization metadata
        """
        means3d = gaussians.get_xyz
        quats = gaussians.get_rotation
        scales = gaussians.get_scaling_for_render
        opacities = gaussians.get_opacity
        sh_colors = gaussians.get_features  # [N, K, 3]
        latent = gaussians.get_latent       # [N, latent_dim]

        # Step 1: RGB + depth + normals
        rgb_result = self.render_rgb_depth(
            means3d, quats, scales, opacities, sh_colors,
            viewmat, K, width, height,
            sh_degree=gaussians.active_sh_degree,
        )

        # Step 2: Latent rasterization
        feat_h = feature_height or height
        feat_w = feature_width or width

        # Scale K for feature resolution if different
        if feat_h != height or feat_w != width:
            K_feat = K.clone()
            if K_feat.dim() == 2:
                K_feat[0, 0] *= feat_w / width
                K_feat[1, 1] *= feat_h / height
                K_feat[0, 2] *= feat_w / width
                K_feat[1, 2] *= feat_h / height
            else:
                K_feat[:, 0, 0] *= feat_w / width
                K_feat[:, 1, 1] *= feat_h / height
                K_feat[:, 0, 2] *= feat_w / width
                K_feat[:, 1, 2] *= feat_h / height
            z_map = self.render_latent(
                means3d, quats, scales, opacities, latent,
                viewmat, K_feat, feat_w, feat_h,
            )
            # Also render alpha at feature resolution for coarse masking
            alpha_feat = F.interpolate(
                rgb_result['alpha'], (feat_h, feat_w), mode='bilinear', align_corners=False
            )
            depth_feat = F.interpolate(
                rgb_result['depth'], (feat_h, feat_w), mode='bilinear', align_corners=False
            )
        else:
            K_feat = K
            z_map = self.render_latent(
                means3d, quats, scales, opacities, latent,
                viewmat, K_feat, feat_w, feat_h,
            )
            alpha_feat = rgb_result['alpha']
            depth_feat = rgb_result['depth']

        if self.split_latent:
            z_fine_map = z_map[:, :self.fine_latent_dim]
            z_coarse_map = z_map[:, self.fine_latent_dim:self.fine_latent_dim + self.coarse_latent_dim]
        else:
            z_fine_map = z_map
            z_coarse_map = z_map

        scale_map = None
        if render_coarse and getattr(self.hash_grid, 'input_mode', 'legacy') == 'implicit_scale':
            scale_map = self.render_scales(
                means3d, quats, scales, opacities, viewmat, K_feat, feat_w, feat_h,
            )

        position_map = None
        needs_position_map = (
            render_coarse
            and self.coarse_mode not in {'spatial_direct', 'spatial_full_direct'}
        ) or self.fine_decoder.use_viewdirs
        if needs_position_map:
            position_map = self.depth_to_position_map(depth_feat, K_feat, viewmat)

        # Step 3: Fine features (explicit decode)
        fine_features = self.decode_fine(z_fine_map, position_map=position_map, viewmat=viewmat)

        # Step 4: Coarse features (implicit decode via hash grid)
        coarse_features = None
        coarse_implicit_features = None
        coarse_carrier_features = None
        coarse_fusion_gate = None
        if render_coarse:
            if self.coarse_mode in {'spatial_direct', 'spatial_full_direct'}:
                coarse_input_map = z_map if self.coarse_direct_uses_full_latent else z_coarse_map
                coarse_features, coarse_carrier_features, coarse_fusion_gate = self.coarse_carrier_fusion(
                    coarse_input_map,
                    None,
                )
            else:
                coarse_implicit_features = self.decode_coarse(
                    position_map=position_map,
                    alpha=alpha_feat,
                    z_map=z_coarse_map,
                    scale_map=scale_map,
                    viewmat=viewmat,
                )
                coarse_features = coarse_implicit_features
                if self.coarse_carrier_fusion is not None:
                    coarse_features, coarse_carrier_features, coarse_fusion_gate = self.coarse_carrier_fusion(
                        z_coarse_map,
                        coarse_implicit_features,
                    )

        return {
            'rgb': rgb_result['rgb'],
            'depth': rgb_result['depth'],
            'alpha': rgb_result['alpha'],
            'normals': rgb_result['normals'],
            'surf_normals': rgb_result['surf_normals'],
            'distort': rgb_result['distort'],
            'z_map': z_map,
            'z_fine_map': z_fine_map,
            'z_coarse_map': z_coarse_map,
            'scale_map': scale_map,
            'fine_features': fine_features,
            'coarse_features': coarse_features,
            'coarse_implicit_features': coarse_implicit_features,
            'coarse_carrier_features': coarse_carrier_features,
            'coarse_fusion_gate': coarse_fusion_gate,
            'meta': rgb_result['meta'],
        }
