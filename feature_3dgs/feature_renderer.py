"""
Feature Renderer (gsplat v1.0+)
===============================
使用gsplat v1.0+ 统一API进行可微分渲染。
支持批量渲染 (多视角一次调用) 和手动分块渲染 (channel chunking)。

gsplat v1.0 API:
  rasterization(means, quats, scales, opacities, colors, viewmats, Ks, ...)
  返回: render_colors [C, H, W, D], render_alphas [C, H, W, 1], meta

注意: gsplat 1.0.0 没有内置 channel_chunk, 需要手动分块渲染高维特征。
"""

import torch
import torch.nn.functional as F
from gsplat import rasterization


def _build_K(fx: float, fy: float, cx: float, cy: float, device: torch.device) -> torch.Tensor:
    """构造3x3相机内参矩阵"""
    K = torch.zeros(3, 3, device=device)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    return K


class FeatureRenderer:
    """
    使用gsplat v1.0+进行特征图渲染。
    
    核心流程 (单次调用):
      rasterization(means, quats, scales, opacities, colors, viewmats, Ks, ...)
    支持:
      - 批量渲染: viewmats [C, 4, 4] 一次渲染C个视角
      - N-D特征: 手动channel chunking (gsplat 1.0.0不支持内置channel_chunk)
      - 深度渲染: render_mode='D'
    """
    
    DEFAULT_CHANNEL_CHUNK = 32  # gsplat CUDA shared memory限制
    
    @staticmethod
    def render_features(
        gaussian_model,
        viewmat: torch.Tensor,
        fx: float, fy: float,
        cx: float, cy: float,
        img_height: int, img_width: int,
        feature_height: int = None,
        feature_width: int = None,
        norm_feat_before_render: bool = True,
        norm_feat_after_render: bool = True,
        max_channels_per_chunk: int = None,
    ) -> dict:
        """
        渲染特征图 (单视角, 向后兼容接口)。
        
        Args:
            gaussian_model: GaussianFeatureModel实例
            viewmat: [4, 4] world-to-camera变换矩阵 (w2c)
            fx, fy, cx, cy: 相机内参
            img_height, img_width: 渲染分辨率
            feature_height, feature_width: 特征图目标分辨率 (若不同则resize)
            norm_feat_before_render: 渲染前是否L2归一化每个Gaussian特征
            norm_feat_after_render: 渲染后是否L2归一化特征图
            max_channels_per_chunk: 每个chunk的最大通道数
            
        Returns:
            dict: {
                'feature_map': [D, fH, fW] 渲染的特征图,
                'alpha': [1, H, W] alpha通道,
            }
        """
        result = FeatureRenderer.render_features_batch(
            gaussian_model=gaussian_model,
            viewmats=viewmat.unsqueeze(0),  # [1, 4, 4]
            fx=fx, fy=fy, cx=cx, cy=cy,
            img_height=img_height, img_width=img_width,
            feature_height=feature_height, feature_width=feature_width,
            norm_feat_before_render=norm_feat_before_render,
            norm_feat_after_render=norm_feat_after_render,
            max_channels_per_chunk=max_channels_per_chunk,
        )
        
        return {
            'feature_map': result['feature_map'][0],  # [D, fH, fW]
            'alpha': result['alpha'][0],               # [1, H, W]
        }
    
    @staticmethod
    def render_features_batch(
        gaussian_model,
        viewmats: torch.Tensor,
        fx: float, fy: float,
        cx: float, cy: float,
        img_height: int, img_width: int,
        feature_height: int = None,
        feature_width: int = None,
        norm_feat_before_render: bool = True,
        norm_feat_after_render: bool = True,
        max_channels_per_chunk: int = None,
    ) -> dict:
        """
        批量渲染特征图 (多视角一次调用)。
        
        Args:
            gaussian_model: GaussianFeatureModel实例
            viewmats: [C, 4, 4] batch of w2c变换矩阵
            fx, fy, cx, cy: 相机内参 (共享)
            img_height, img_width: 渲染分辨率 
            feature_height, feature_width: 特征图目标分辨率
            norm_feat_before_render: 渲染前L2归一化
            norm_feat_after_render: 渲染后L2归一化
            max_channels_per_chunk: channel_chunk大小
            
        Returns:
            dict: {
                'feature_map': [C, D, fH, fW] 批量特征图,
                'alpha': [C, 1, H, W] alpha通道,
            }
        """
        device = gaussian_model.get_xyz.device
        C = viewmats.shape[0]
        
        if feature_height is None:
            feature_height = img_height
        if feature_width is None:
            feature_width = img_width
        if max_channels_per_chunk is None:
            max_channels_per_chunk = FeatureRenderer.DEFAULT_CHANNEL_CHUNK
        
        # --- Gaussian参数 ---
        means3d = gaussian_model.get_xyz          # [N, 3]
        scales = gaussian_model.get_scaling        # [N, 3]
        quats = gaussian_model.get_rotation        # [N, 4]
        opacities = gaussian_model.get_opacity.squeeze(-1)  # [N] (v1.x要求1D)
        
        # --- 特征嵌入 ---
        if norm_feat_before_render:
            colors = gaussian_model.get_loc_feature  # L2 normalized [N, D]
        else:
            colors = gaussian_model._loc_feature     # raw [N, D]
        
        D = colors.shape[1]
        
        # --- 内参矩阵 [C, 3, 3] ---
        K = _build_K(fx, fy, cx, cy, device)
        Ks = K.unsqueeze(0).expand(C, -1, -1)  # [C, 3, 3]
        
        # --- 分块渲染 (gsplat 1.0.0 不支持内置 channel_chunk) ---
        chunk_size = max_channels_per_chunk
        n_chunks = (D + chunk_size - 1) // chunk_size
        
        feature_chunks = []
        alpha_out = None
        
        for i in range(n_chunks):
            c_start = i * chunk_size
            c_end = min((i + 1) * chunk_size, D)
            chunk_colors = colors[:, c_start:c_end]  # [N, c_dim]
            
            render_colors, render_alphas, meta = rasterization(
                means=means3d,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=chunk_colors,        # [N, c_dim]
                viewmats=viewmats,          # [C, 4, 4]
                Ks=Ks,                      # [C, 3, 3]
                width=img_width,
                height=img_height,
                packed=True,
                render_mode='RGB',
                near_plane=0.01,
                far_plane=1e5,
            )
            # render_colors: [C, H, W, c_dim]
            feature_chunks.append(render_colors)
            
            if i == 0:
                alpha_out = render_alphas  # [C, H, W, 1]
        
        # 拼接所有chunks: [C, H, W, D]
        if n_chunks == 1:
            feature_map_hwd = feature_chunks[0]
        else:
            feature_map_hwd = torch.cat(feature_chunks, dim=-1)
        
        # [C, H, W, D] -> [C, D, H, W]
        feature_map = feature_map_hwd.permute(0, 3, 1, 2)
        # [C, H, W, 1] -> [C, 1, H, W]
        alpha = alpha_out.permute(0, 3, 1, 2)
        
        # 如果特征分辨率与渲染分辨率不同, resize
        if feature_height != img_height or feature_width != img_width:
            feature_map = F.interpolate(
                feature_map,
                size=(feature_height, feature_width),
                mode='bilinear',
                align_corners=False,
            )
        
        # 渲染后L2归一化 (per-pixel, dim=1 即通道维)
        if norm_feat_after_render:
            feature_map = F.normalize(feature_map, p=2, dim=1)
        
        return {
            'feature_map': feature_map,   # [C, D, fH, fW]
            'alpha': alpha,               # [C, 1, H, W]
        }
    
    @staticmethod
    def render_rgb(
        gaussian_model,
        viewmat: torch.Tensor,
        fx: float, fy: float,
        cx: float, cy: float,
        img_height: int, img_width: int,
    ) -> dict:
        """
        渲染RGB图像 (用于可视化验证)。
        使用SH DC系数 (sh_degree=0) 直接作为颜色。
        
        Returns:
            dict: {'rgb': [3, H, W], 'alpha': [1, H, W]}
        """
        device = gaussian_model.get_xyz.device
        means3d = gaussian_model.get_xyz
        scales = gaussian_model.get_scaling
        quats = gaussian_model.get_rotation
        opacities = gaussian_model.get_opacity.squeeze(-1)  # [N]
        
        # SH DC -> RGB
        C0 = 0.28209479177387814
        colors = gaussian_model._features_dc * C0 + 0.5  # [N, 3]
        colors = torch.clamp(colors, 0.0, 1.0)
        
        K = _build_K(fx, fy, cx, cy, device)
        Ks = K.unsqueeze(0)         # [1, 3, 3]
        viewmats = viewmat.unsqueeze(0)  # [1, 4, 4]
        background = torch.ones(3, device=device)
        
        render_colors, render_alphas, meta = rasterization(
            means=means3d,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=img_width,
            height=img_height,
            packed=True,
            backgrounds=background.unsqueeze(0),  # [1, 3]
            render_mode='RGB',
        )
        # render_colors: [1, H, W, 3], render_alphas: [1, H, W, 1]
        
        rgb = render_colors[0].permute(2, 0, 1)       # [3, H, W]
        alpha = render_alphas[0].permute(2, 0, 1)      # [1, H, W]
        
        return {
            'rgb': rgb,
            'alpha': alpha,
        }
    
    @staticmethod
    def render_depth(
        gaussian_model,
        viewmat: torch.Tensor,
        fx: float, fy: float,
        cx: float, cy: float,
        img_height: int, img_width: int,
    ) -> torch.Tensor:
        """
        渲染深度图 (使用gsplat内置的depth渲染模式)。
        
        Returns:
            (H, W) depth map (accumulated z-depth)
        """
        device = gaussian_model.get_xyz.device
        means3d = gaussian_model.get_xyz
        scales = gaussian_model.get_scaling
        quats = gaussian_model.get_rotation
        opacities = gaussian_model.get_opacity.squeeze(-1)
        
        # 深度渲染模式需要colors占位 (不参与运算，但API要求)
        # 使用1通道假颜色即可
        dummy_colors = torch.zeros(means3d.shape[0], 1, device=device)
        
        K = _build_K(fx, fy, cx, cy, device)
        Ks = K.unsqueeze(0)
        viewmats = viewmat.unsqueeze(0)
        
        render_colors, render_alphas, meta = rasterization(
            means=means3d,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=dummy_colors,
            viewmats=viewmats,
            Ks=Ks,
            width=img_width,
            height=img_height,
            packed=True,
            render_mode='D',      # 直接渲染深度
            near_plane=0.01,
            far_plane=1e5,
        )
        # render_colors: [1, H, W, 1] (depth)
        return render_colors[0, :, :, 0]  # [H, W]
