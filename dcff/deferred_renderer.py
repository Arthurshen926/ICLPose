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

from gsplat import rasterization_2dgs


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


class DeferredCascadedRenderer(nn.Module):
    """Screen-space deferred rendering for dual-scale feature extraction.

    Renders 2DGS surfels to screen, then decodes fine (explicit) and
    coarse (implicit) features in 2D screen space.
    """

    def __init__(
        self,
        hash_grid: nn.Module,
        latent_dim: int = 16,
        fine_feature_dim: int = 64,
        coarse_feature_dim: int = 64,
        fine_hidden_dim: int = None,
        fine_num_layers: int = 3,
        fine_use_viewdirs: bool = False,
        fine_view_degree: int = 2,
        chunk_size: int = 16,
    ):
        super().__init__()
        self.hash_grid = hash_grid
        self.fine_decoder = FineDecoder(
            latent_dim=latent_dim,
            feature_dim=fine_feature_dim,
            hidden_dim=fine_hidden_dim,
            num_layers=fine_num_layers,
            use_viewdirs=fine_use_viewdirs,
            view_degree=fine_view_degree,
        )
        self.latent_dim = latent_dim
        self.fine_feature_dim = fine_feature_dim
        self.coarse_feature_dim = coarse_feature_dim
        self.chunk_size = chunk_size

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
            z_flat = z_map.permute(0, 2, 3, 1).reshape(-1, self.latent_dim)
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

        scale_map = None
        if render_coarse and getattr(self.hash_grid, 'input_mode', 'legacy') == 'implicit_scale':
            scale_map = self.render_scales(
                means3d, quats, scales, opacities, viewmat, K_feat, feat_w, feat_h,
            )

        position_map = None
        if render_coarse or self.fine_decoder.use_viewdirs:
            position_map = self.depth_to_position_map(depth_feat, K_feat, viewmat)

        # Step 3: Fine features (explicit decode)
        fine_features = self.decode_fine(z_map, position_map=position_map, viewmat=viewmat)

        # Step 4: Coarse features (implicit decode via hash grid)
        coarse_features = None
        if render_coarse:
            coarse_features = self.decode_coarse(
                position_map=position_map,
                alpha=alpha_feat,
                z_map=z_map,
                scale_map=scale_map,
                viewmat=viewmat,
            )

        return {
            'rgb': rgb_result['rgb'],
            'depth': rgb_result['depth'],
            'alpha': rgb_result['alpha'],
            'normals': rgb_result['normals'],
            'surf_normals': rgb_result['surf_normals'],
            'distort': rgb_result['distort'],
            'z_map': z_map,
            'scale_map': scale_map,
            'fine_features': fine_features,
            'coarse_features': coarse_features,
            'meta': rgb_result['meta'],
        }
