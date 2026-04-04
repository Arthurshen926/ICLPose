# Source Generated with Decompyle++
# File: deferred_renderer.cpython-39.pyc (Python 3.9)

'''
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
'''
import torch
from torch.nn import nn
import torch.nn.functional
F = functional
nn
from gsplat import rasterization_2dgs

def encode_view_directions_sh(view_dirs = None, degree = None):
    '''Compact real SH-like encoding up to degree 2 for view directions.'''
    if degree < 0 or degree > 2:
        raise ValueError(f'''Only degree 0-2 supported, got degree={degree}''')
    x = view_dirs[(..., 0:1)]
    y = view_dirs[(..., 1:2)]
    z = view_dirs[(..., 2:3)]
    basis = [
        torch.ones_like(x)]
    if degree >= 1:
        basis.extend([
            y,
            z,
            x])
    if degree >= 2:
        basis.extend([
            x * y,
            y * z,
            3 * z * z - 1,
            x * z,
            x * x - y * y])
    return torch.cat(basis, -1, **('dim',))


class FineDecoder(nn.Module):
    '''Explicit fine decoder for geometric features.

    By default this matches the legacy latent-only 1×1 Conv MLP. When
    `use_viewdirs=True`, it decodes from explicit latent + per-pixel view SH.
    '''
    
    def __init__(self = None, latent_dim = None, feature_dim = None, hidden_dim = None, num_layers = None, use_viewdirs = None, view_degree = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feature_dim
        self.use_viewdirs = use_viewdirs
        self.view_degree = view_degree
        self.view_dim = 0 if not use_viewdirs else sum((lambda .0: for order in .0:
2 * order + 1)(range(view_degree + 1)))
        input_dim = latent_dim + self.view_dim
        if num_layers < 2:
            raise ValueError(f'''num_layers must be >= 2, got {num_layers}''')
        layers = []
        prev = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(prev, hidden_dim, 1),
                nn.GELU()])
            prev = hidden_dim
        layers.append(nn.Conv2d(prev, feature_dim, 1))
    # WARNING: Decompyle incomplete

    
    def forward(self = None, z_map = None, view_dir_map = None):
        '''Decode fine features from explicit latent map and optional view encoding.'''
        if self.use_viewdirs:
            if view_dir_map is None:
                raise ValueError('FineDecoder requires view_dir_map when use_viewdirs=True')
            z_map = torch.cat([
                z_map,
                view_dir_map], 1, **('dim',))
        return self.decoder(z_map)

    __classcell__ = None


class DeferredCascadedRenderer(nn.Module):
    '''Screen-space deferred rendering for dual-scale feature extraction.

    Renders 2DGS surfels to screen, then decodes fine (explicit) and
    coarse (implicit) features in 2D screen space.
    '''
    
    def __init__(self = None, hash_grid = None, latent_dim = None, fine_feature_dim = None, coarse_feature_dim = None, fine_hidden_dim = None, fine_num_layers = None, fine_use_viewdirs = None, fine_view_degree = None, chunk_size = None, normalize_features = None):
        super().__init__()
        self.hash_grid = hash_grid
        self.fine_decoder = FineDecoder(latent_dim, fine_feature_dim, fine_hidden_dim, fine_num_layers, fine_use_viewdirs, fine_view_degree, **('latent_dim', 'feature_dim', 'hidden_dim', 'num_layers', 'use_viewdirs', 'view_degree'))
        self.latent_dim = latent_dim
        self.fine_feature_dim = fine_feature_dim
        self.coarse_feature_dim = coarse_feature_dim
        self.chunk_size = chunk_size
        self.normalize_features = normalize_features

    
    def render_rgb_depth(self, means3d, quats, scales, opacities, colors, viewmat = None, K = None, width = None, height = (0,), sh_degree = {
        'means3d': torch.Tensor,
        'quats': torch.Tensor,
        'scales': torch.Tensor,
        'opacities': torch.Tensor,
        'colors': torch.Tensor,
        'viewmat': torch.Tensor,
        'K': torch.Tensor,
        'width': int,
        'height': int,
        'sh_degree': int }):
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
        (render_colors, render_alphas, normals, surf_normals, distort, median_depth, meta) = rasterization_2dgs(means3d, quats, scales, opacities.squeeze(-1) if opacities.dim() == 2 else opacities, colors, viewmat, K, width, height, False, 0.01, 100000, 'RGB+ED', sh_degree, True, **('means', 'quats', 'scales', 'opacities', 'colors', 'viewmats', 'Ks', 'width', 'height', 'packed', 'near_plane', 'far_plane', 'render_mode', 'sh_degree', 'absgrad'))
    # WARNING: Decompyle incomplete

    
    def render_attribute_map(self, means3d, quats, scales, opacities, attributes, viewmat = None, K = None, width = None, height = {
        'means3d': torch.Tensor,
        'quats': torch.Tensor,
        'scales': torch.Tensor,
        'opacities': torch.Tensor,
        'attributes': torch.Tensor,
        'viewmat': torch.Tensor,
        'K': torch.Tensor,
        'width': int,
        'height': int }):
        '''Rasterize arbitrary per-Gaussian attributes to screen space.'''
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)
        if K.dim() == 2:
            K = K.unsqueeze(0)
        D = attributes.shape[1]
        n_chunks = (D + self.chunk_size - 1) // self.chunk_size
        chunks = []
    # WARNING: Decompyle incomplete

    
    def render_latent(self, means3d, quats, scales, opacities, latent, viewmat = None, K = None, width = None, height = {
        'means3d': torch.Tensor,
        'quats': torch.Tensor,
        'scales': torch.Tensor,
        'opacities': torch.Tensor,
        'latent': torch.Tensor,
        'viewmat': torch.Tensor,
        'K': torch.Tensor,
        'width': int,
        'height': int }):
        '''Rasterize per-Gaussian latent vectors to screen space.'''
        return self.render_attribute_map(means3d, quats, scales, opacities, latent, viewmat, K, width, height)

    
    def render_scales(self, means3d, quats, scales, opacities, viewmat = None, K = None, width = None, height = {
        'means3d': torch.Tensor,
        'quats': torch.Tensor,
        'scales': torch.Tensor,
        'opacities': torch.Tensor,
        'viewmat': torch.Tensor,
        'K': torch.Tensor,
        'width': int,
        'height': int }):
        '''Rasterize per-Gaussian 2D scales to screen space.'''
        return self.render_attribute_map(means3d, quats, scales, opacities, scales[(:, :2)], viewmat, K, width, height)

    
    def depth_to_position_map(depth = None, K = None, viewmat = staticmethod):
        '''Unproject depth map to world-space 3D position map.

        Args:
            depth: [B, 1, H, W], [B, H, W], or [H, W] depth values
            K: [B, 3, 3] or [3, 3] camera intrinsics
            viewmat: [B, 4, 4] or [4, 4] world-to-camera transforms

        Returns:
            position_map: [B, H, W, 3] or [H, W, 3] world-space coordinates
        '''
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
        (B, H, W) = depth.shape
        device = depth.device
        (v_coords, u_coords) = torch.meshgrid(torch.arange(H, device, torch.float32, **('device', 'dtype')), torch.arange(W, device, torch.float32, **('device', 'dtype')), 'ij', **('indexing',))
        u_coords = u_coords.unsqueeze(0).expand(B, -1, -1)
        v_coords = v_coords.unsqueeze(0).expand(B, -1, -1)
        fx = K[(:, 0, 0)].view(B, 1, 1)
        fy = K[(:, 1, 1)].view(B, 1, 1)
        cx = K[(:, 0, 2)].view(B, 1, 1)
        cy = K[(:, 1, 2)].view(B, 1, 1)
        z = depth
        x_cam = ((u_coords - cx) / fx) * z
        y_cam = ((v_coords - cy) / fy) * z
        cam_pts = torch.stack([
            x_cam,
            y_cam,
            z], -1, **('dim',))
        c2w = torch.inverse(viewmat)
        R = c2w[(:, :3, :3)]
        t = c2w[(:, :3, 3)]
        world_pts = torch.einsum('bhwj,bij->bhwi', cam_pts, R) + t[(:, None, None, :)]
        if squeeze_batch:
            return world_pts[0]

    depth_to_position_map = None(depth_to_position_map)
    
    def compute_view_directions(position_map = None, viewmat = None):
        '''Compute per-pixel view directions from camera center to 3D points.

        Args:
            position_map: [B, H, W, 3] or [H, W, 3] world-space positions
            viewmat: [B, 4, 4] or [4, 4] world-to-camera transforms

        Returns:
            view_dirs: same spatial shape as position_map
        '''
        squeeze_batch = False
        if position_map.dim() == 3:
            position_map = position_map.unsqueeze(0)
            squeeze_batch = True
        if viewmat.dim() == 2:
            viewmat = viewmat.unsqueeze(0)
        c2w = torch.inverse(viewmat)
        cam_center = c2w[(:, :3, 3)]
        view_dirs = position_map - cam_center[(:, None, None, :)]
        view_dirs = F.normalize(view_dirs, -1, **('dim',))
        if squeeze_batch:
            return view_dirs[0]

    compute_view_directions = None(compute_view_directions)
    
    def decode_fine(self = None, z_map = None, position_map = None, viewmat = (None, None)):
        '''Decode fine geometric features from rasterized latent map.

        Args:
            z_map: [B, latent_dim, H, W]
        Returns:
            fine_features: [B, fine_feature_dim, H, W]
        '''
        view_dir_map = None
        if self.fine_decoder.use_viewdirs:
            if position_map is None or viewmat is None:
                raise ValueError('decode_fine requires position_map and viewmat when use_viewdirs=True')
            view_dirs = self.compute_view_directions(position_map, viewmat)
            view_dir_map = encode_view_directions_sh(view_dirs, self.fine_decoder.view_degree, **('degree',)).permute(0, 3, 1, 2)
        return self.fine_decoder(z_map, view_dir_map, **('view_dir_map',))

    
    def decode_coarse(self, position_map, alpha = None, z_map = None, scale_map = None, viewmat = (None, None, None, 0.5), alpha_threshold = {
        'position_map': torch.Tensor,
        'alpha': torch.Tensor,
        'z_map': torch.Tensor,
        'scale_map': torch.Tensor,
        'viewmat': torch.Tensor,
        'alpha_threshold': float,
        'return': torch.Tensor }):
        '''Decode coarse semantic features via hash grid + MLP.

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
        '''
        (B, H, W) = position_map.shape[:3]
        valid = alpha.squeeze(1) > alpha_threshold
        pos_flat = position_map.reshape(-1, 3)
        valid_flat = valid.reshape(-1)
        if getattr(self.hash_grid, 'input_mode', 'legacy') == 'legacy':
            if z_map is None or viewmat is None:
                raise ValueError('Legacy coarse decoding requires z_map and viewmat')
            view_dirs = self.compute_view_directions(position_map, viewmat)
            z_flat = z_map.permute(0, 2, 3, 1).reshape(-1, self.latent_dim)
            vd_flat = view_dirs.reshape(-1, 3)
            coarse_flat = self.hash_grid(pos_flat, z_flat, vd_flat, valid_flat, **('positions', 'latent', 'view_dirs', 'valid_mask'))
        elif scale_map is None:
            raise ValueError('implicit_scale coarse decoding requires scale_map')
        scale_flat = scale_map.permute(0, 2, 3, 1).reshape(-1, scale_map.shape[1])
        coarse_flat = self.hash_grid(pos_flat, scale_flat, valid_flat, **('positions', 'scales', 'valid_mask'))
        coarse = coarse_flat.reshape(B, H, W, self.coarse_feature_dim)
        coarse = coarse.permute(0, 3, 1, 2)
        return coarse

    
    def forward(self, gaussians, viewmat, K, width = None, height = None, render_coarse = None, feature_height = (True, None, None), feature_width = {
        'viewmat': torch.Tensor,
        'K': torch.Tensor,
        'width': int,
        'height': int,
        'render_coarse': bool,
        'feature_height': int,
        'feature_width': int }):
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
        sh_colors = gaussians.get_features
        latent = gaussians.get_latent
        rgb_result = self.render_rgb_depth(means3d, quats, scales, opacities, sh_colors, viewmat, K, width, height, gaussians.active_sh_degree, **('sh_degree',))
        if not feature_height:
            pass
        feat_h = height
        if not feature_width:
            pass
        feat_w = width
        if feat_h != height or feat_w != width:
            K_feat = K.clone()
            if K_feat.dim() == 2:
                K_feat[(0, 0)] *= feat_w / width
                K_feat[(1, 1)] *= feat_h / height
                K_feat[(0, 2)] *= feat_w / width
                K_feat[(1, 2)] *= feat_h / height
            else:
                K_feat[(:, 0, 0)] *= feat_w / width
                K_feat[(:, 1, 1)] *= feat_h / height
                K_feat[(:, 0, 2)] *= feat_w / width
                K_feat[(:, 1, 2)] *= feat_h / height
            z_map = self.render_latent(means3d, quats, scales, opacities, latent, viewmat, K_feat, feat_w, feat_h)
            alpha_feat = F.interpolate(rgb_result['alpha'], (feat_h, feat_w), 'bilinear', False, **('mode', 'align_corners'))
            depth_feat = F.interpolate(rgb_result['depth'], (feat_h, feat_w), 'bilinear', False, **('mode', 'align_corners'))
        else:
            K_feat = K
            z_map = self.render_latent(means3d, quats, scales, opacities, latent, viewmat, K_feat, feat_w, feat_h)
            alpha_feat = rgb_result['alpha']
            depth_feat = rgb_result['depth']
        scale_map = None
        if render_coarse and getattr(self.hash_grid, 'input_mode', 'legacy') == 'implicit_scale':
            scale_map = self.render_scales(means3d, quats, scales, opacities, viewmat, K_feat, feat_w, feat_h)
        position_map = None
        if render_coarse or self.fine_decoder.use_viewdirs:
            position_map = self.depth_to_position_map(depth_feat, K_feat, viewmat)
        fine_features = self.decode_fine(z_map, position_map, viewmat, **('position_map', 'viewmat'))
        coarse_features = None
        if render_coarse:
            coarse_features = self.decode_coarse(position_map, alpha_feat, z_map, scale_map, viewmat, **('position_map', 'alpha', 'z_map', 'scale_map', 'viewmat'))
        if self.normalize_features:
            fine_features = F.normalize(fine_features, 2, 1, **('p', 'dim'))
            if coarse_features is not None:
                coarse_features = F.normalize(coarse_features, 2, 1, **('p', 'dim'))
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
            'meta': rgb_result['meta'] }

    __classcell__ = None

