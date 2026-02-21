"""
Feature Renderer
================
使用gsplat对带特征嵌入的3DGS进行可微分渲染。
两遍渲染: 第一遍渲染RGB (可选), 第二遍渲染特征图。
参考STDLoc的渲染策略。

注意: 当特征维度很大 (如256) 时, gsplat的CUDA kernel会超出共享内存限制,
因此采用分块渲染策略 (chunked rendering): 将D维特征分成若干小块, 
每块独立渲染后拼接。
"""

import torch
import torch.nn.functional as F
from gsplat import project_gaussians, rasterize_gaussians


class FeatureRenderer:
    """
    使用gsplat进行特征图渲染。
    
    核心流程:
    1. project_gaussians: 将3D Gaussians投影到2D
    2. rasterize_gaussians: 分块渲染 (chunked) 特征图, 避免共享内存溢出
    3. 拼接所有chunk得到完整特征图
    """
    
    BLOCK_WIDTH = 16   # gsplat tile size
    MAX_CHANNELS = 32  # 每次渲染的最大通道数 (受CUDA shared memory限制)
    
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
        渲染特征图 (分块渲染以支持高维特征)。
        
        Args:
            gaussian_model: GaussianFeatureModel实例
            viewmat: [4, 4] world-to-camera变换矩阵 (w2c)
            fx, fy, cx, cy: 相机内参
            img_height, img_width: 图像分辨率 (用于投影)
            feature_height, feature_width: 特征图分辨率 (如与image不同则resize)
            norm_feat_before_render: 渲染前是否L2归一化每个Gaussian特征
            norm_feat_after_render: 渲染后是否L2归一化特征图
            max_channels_per_chunk: 每个chunk的最大通道数
            
        Returns:
            dict: {
                'feature_map': [D, fH, fW] 渲染的特征图,
                'alpha': [1, H, W] alpha通道,
                'radii': [N] 屏幕空间半径,
                'visible_mask': [N] 可见性mask,
            }
        """
        device = gaussian_model.get_xyz.device
        
        if feature_height is None:
            feature_height = img_height
        if feature_width is None:
            feature_width = img_width
        if max_channels_per_chunk is None:
            max_channels_per_chunk = FeatureRenderer.MAX_CHANNELS
        
        # --- Step 1: 投影所有Gaussians到2D (一次投影, 复用) ---
        means3d = gaussian_model.get_xyz          # [N, 3]
        scales = gaussian_model.get_scaling        # [N, 3] (activated)
        quats = gaussian_model.get_rotation        # [N, 4] (normalized)
        opacities = gaussian_model.get_opacity     # [N, 1]
        
        (xys, depths, radii, conics, 
         compensation, num_tiles_hit, cov3d) = project_gaussians(
            means3d=means3d,
            scales=scales,
            glob_scale=1.0,
            quats=quats,
            viewmat=viewmat,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_height=img_height,
            img_width=img_width,
            block_width=FeatureRenderer.BLOCK_WIDTH,
        )
        
        # 可见性过滤
        visible_mask = radii.squeeze(-1) > 0  # [N]
        
        # --- Step 2: 获取特征嵌入 ---
        if norm_feat_before_render:
            loc_features = gaussian_model.get_loc_feature  # L2 normalized [N, D]
        else:
            loc_features = gaussian_model._loc_feature  # raw [N, D]
        
        D = gaussian_model.feature_dim
        
        # --- Step 3: 分块渲染特征图 ---
        # 将D维特征分成多个chunk, 每个chunk独立渲染
        chunk_size = max_channels_per_chunk
        n_chunks = (D + chunk_size - 1) // chunk_size
        
        feature_chunks = []
        alpha_out = None
        
        for i in range(n_chunks):
            c_start = i * chunk_size
            c_end = min((i + 1) * chunk_size, D)
            c_dim = c_end - c_start
            
            chunk_colors = loc_features[:, c_start:c_end]        # [N, c_dim]
            chunk_bg = torch.zeros(c_dim, device=device)
            
            chunk_out = rasterize_gaussians(
                xys=xys,
                depths=depths,
                radii=radii,
                conics=conics,
                num_tiles_hit=num_tiles_hit,
                colors=chunk_colors,
                opacity=opacities.squeeze(-1),
                img_height=img_height,
                img_width=img_width,
                block_width=FeatureRenderer.BLOCK_WIDTH,
                background=chunk_bg,
                return_alpha=(i == 0),  # 只在第一个chunk取alpha
            )
            
            if i == 0 and isinstance(chunk_out, tuple):
                chunk_img, alpha = chunk_out
                # gsplat 0.1.x: alpha可能是 [H, W] 或 [H, W, 1]
                if alpha.dim() == 2:
                    alpha_out = alpha.unsqueeze(0)   # [1, H, W]
                elif alpha.dim() == 3:
                    alpha_out = alpha.permute(2, 0, 1)  # [1, H, W]
                else:
                    alpha_out = alpha
            elif isinstance(chunk_out, tuple):
                chunk_img, _ = chunk_out
            else:
                chunk_img = chunk_out
            
            # chunk_img: [H, W, c_dim] -> [c_dim, H, W]
            feature_chunks.append(chunk_img.permute(2, 0, 1))
        
        # 拼接所有chunks
        feature_map = torch.cat(feature_chunks, dim=0)  # [D, H, W]
        
        # 如果特征分辨率与图像分辨率不同, resize
        if feature_height != img_height or feature_width != img_width:
            feature_map = F.interpolate(
                feature_map.unsqueeze(0),
                size=(feature_height, feature_width),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        
        # 渲染后L2归一化 (参考STDLoc: per-pixel归一化)
        if norm_feat_after_render:
            feature_map = F.normalize(feature_map, p=2, dim=0)
        
        return {
            'feature_map': feature_map,      # [D, fH, fW]
            'alpha': alpha_out,              # [1, H, W] or None
            'radii': radii,                   # [N, 1]
            'visible_mask': visible_mask,     # [N]
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
            dict: {'rgb': [3, H, W], 'alpha': [1, H, W], ...}
        """
        from splatloc_modules.gaussian_splatting.utils.sh_utils import eval_sh
        
        device = gaussian_model.get_xyz.device
        means3d = gaussian_model.get_xyz
        scales = gaussian_model.get_scaling
        quats = gaussian_model.get_rotation
        opacities = gaussian_model.get_opacity
        
        (xys, depths, radii, conics,
         compensation, num_tiles_hit, cov3d) = project_gaussians(
            means3d=means3d,
            scales=scales,
            glob_scale=1.0,
            quats=quats,
            viewmat=viewmat,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_height=img_height,
            img_width=img_width,
            block_width=FeatureRenderer.BLOCK_WIDTH,
        )
        
        # SH DC -> RGB (sh_degree=0: color = SH_coeff * C0 + 0.5)
        C0 = 0.28209479177387814  # 1/(2*sqrt(pi))
        colors = gaussian_model._features_dc * C0 + 0.5  # [N, 3]
        colors = torch.clamp(colors, 0.0, 1.0)
        
        background = torch.ones(3, device=device)  # 白色背景
        
        rgb = rasterize_gaussians(
            xys=xys,
            depths=depths,
            radii=radii,
            conics=conics,
            num_tiles_hit=num_tiles_hit,
            colors=colors,
            opacity=opacities.squeeze(-1),
            img_height=img_height,
            img_width=img_width,
            block_width=FeatureRenderer.BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )
        
        if isinstance(rgb, tuple):
            rgb, alpha = rgb
            if alpha.dim() == 2:
                alpha = alpha.unsqueeze(0)
            elif alpha.dim() == 3:
                alpha = alpha.permute(2, 0, 1)
        else:
            alpha = None
        
        rgb = rgb.permute(2, 0, 1)  # [3, H, W]
        
        return {
            'rgb': rgb,
            'alpha': alpha,
        }
