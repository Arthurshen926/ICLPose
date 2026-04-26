"""
Multi-Scale Feature Renderer
=============================
管理多个尺度的 3DGS 特征模型，按需渲染特征图和深度图。

每个尺度有独立的 GaussianFeatureModel (shared geometry, different features)。
所有尺度共享同一套 Gaussian 几何参数 (xyz, rotation, scaling, opacity)，
仅特征嵌入 (_loc_feature) 不同。

尺度定义:
  - coarse: 1280d→32d (压缩后), 7×10, SD stage 5
  - mid:    1280d→64d, 15×20, SD stage 4
  - fine_sd:  640d→64d, 35×46, SD stage 3
  - fine_dino: 768d→64d, 35×46, DINO patches

用法:
    renderer = MultiScaleRenderer(ply_path, scale_model_paths, device)
    result = renderer.render(pose_w2c, intrinsics, scales=['coarse', 'fine_sd'])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer


def _load_triplane_renderer(
    ply_path: str,
    triplane_path: str,
    scale_resolutions: Dict[str, Tuple[int, int]],
    device: str,
    img_height: int,
    img_width: int,
    fx: float, fy: float, cx: float, cy: float,
):
    """Load a TriPlaneFeatureModel and build per-scale rendering info.
    
    Returns (models_dict, scale_info_dict, scale_names) where models_dict has a single
    'triplane' entry and scale_info has per-scale resolution + dim info.
    """
    from feature_3dgs.triplane_feature_model import TriPlaneFeatureModel
    
    ckpt = torch.load(triplane_path, map_location='cpu')
    
    # Extract config from checkpoint (flat keys, not nested)
    head_dims = ckpt['head_dims']  # e.g. {'coarse': 32, 'mid': 64, 'fine': 64}
    
    model = TriPlaneFeatureModel(
        plane_resolution=ckpt['plane_resolution'],
        plane_channels=ckpt['plane_channels'],
        trunk_dim=ckpt.get('trunk_dim', 128),
        head_dims=head_dims,
    )
    model.load_ply(ply_path)
    
    # Load tri-plane parameters and decoder weights
    model.plane_xy.data = ckpt['plane_xy'].to(device)
    model.plane_xz.data = ckpt['plane_xz'].to(device)
    model.plane_yz.data = ckpt['plane_yz'].to(device)
    model.decoder.load_state_dict(ckpt['decoder_state_dict'])
    model.bbox_min = ckpt['bbox_min'].to(device)
    model.bbox_max = ckpt['bbox_max'].to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device).eval()
    
    # Build scale info from head_dims
    scale_names = sorted(head_dims.keys())
    scale_info = {}
    dim_offset = 0
    for name in scale_names:
        dim = head_dims[name]
        res = scale_resolutions.get(name, (35, 46))
        scale_info[name] = {
            'feat_dim': dim,
            'resolution': tuple(res),
            'dim_offset': dim_offset,
            'num_gaussians': model.get_xyz.shape[0],
        }
        dim_offset += dim
        print(f"  [{name}] feat_dim={dim}, res={res} (triplane)")
    
    return {'triplane': model}, scale_info, scale_names


class MultiScaleRenderer(nn.Module):
    """
    多尺度 3DGS 特征渲染器
    
    管理 N 个尺度的特征模型，提供统一的渲染接口。
    
    Args:
        ply_path: 原始 3DGS PLY 文件路径 (包含几何参数)
        scale_model_paths: {scale_name: pth_path} 各尺度的训练好的模型
        device: 设备
        img_height, img_width: 图像分辨率 (用于 Gaussian 投影)
        fx, fy, cx, cy: 相机内参 (默认值来自 room_0)
    """
    
    # 各尺度的渲染分辨率配置
    SCALE_RESOLUTIONS = {
        'coarse': (7, 10),
        'mid': (15, 20),
        'fine_sd': (35, 46),
        'fine_dino': (35, 46),
    }
    
    # v2: SD 保留 UNet 零填充后的原生分辨率, 2× 层级
    SCALE_RESOLUTIONS_V2 = {
        'coarse': (8, 10),
        'mid': (16, 20),
        'fine_sd': (32, 40),
        'fine_dino': (35, 46),
    }
    
    def __init__(
        self,
        ply_path: str,
        scale_model_paths: Dict[str, str],
        device: str = 'cuda',
        img_height: int = 480,
        img_width: int = 640,
        fx: float = 320.0, fy: float = 320.0,
        cx: float = 319.5, cy: float = 239.5,
        triplane_model_path: str = None,
        scale_resolutions: Dict[str, Tuple[int, int]] = None,
        # Feature sharpening: unsharp mask to counteract alpha-blending low-pass
        # filter. Applied after rendering + L2 norm. 0.0 = disabled.
        sharpen_strength: float = 0.0,
        sharpen_kernel_size: int = 3,
    ):
        super().__init__()
        
        self.device = device
        self.img_height = img_height
        self.img_width = img_width
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.triplane_mode = triplane_model_path is not None
        self.sharpen_strength = sharpen_strength
        self.sharpen_kernel_size = sharpen_kernel_size
        
        if self.triplane_mode:
            # Tri-plane mode: single shared model, per-scale feature slicing
            res_config = scale_resolutions or {}
            self.models, self.scale_info, self.scale_names = _load_triplane_renderer(
                ply_path, triplane_model_path, res_config,
                device, img_height, img_width, fx, fy, cx, cy)
            self.models = nn.ModuleDict(self.models)
        else:
            # Legacy per-scale models mode
            self.scale_names = sorted(scale_model_paths.keys())
            self.models = nn.ModuleDict()
            self.scale_info = {}
            
            for name, pth_path in scale_model_paths.items():
                pth_path = Path(pth_path)
                if not pth_path.exists():
                    print(f"  [Warning] {name} model not found: {pth_path}, skipping")
                    continue
                
                # 加载 checkpoint
                ckpt = torch.load(str(pth_path), map_location='cpu')
                feat_dim = ckpt['feature_dim']
                loc_feature = ckpt['loc_feature']  # (N, D)
                resolution = ckpt.get('resolution', self.SCALE_RESOLUTIONS.get(name, (35, 46)))
                # Explicit scale_resolutions override (e.g. joint-trained models without resolution in ckpt)
                if scale_resolutions and name in scale_resolutions:
                    resolution = tuple(scale_resolutions[name])
                
                # 创建 model 并加载几何参数
                model = GaussianFeatureModel(feature_dim=feat_dim)
                model.load_ply(ply_path)
                
                # 覆盖特征嵌入
                with torch.no_grad():
                    model._loc_feature = nn.Parameter(loc_feature.to(device))
                
                # 冻结所有参数 (推理模式)
                for param in model.parameters():
                    param.requires_grad = False
                
                model = model.to(device)
                model.eval()
                
                self.models[name] = model
                self.scale_info[name] = {
                    'feat_dim': feat_dim,
                    'resolution': tuple(resolution) if hasattr(resolution, '__iter__') else resolution,
                    'num_gaussians': loc_feature.shape[0],
                }
                
                print(f"  [{name}] feat_dim={feat_dim}, "
                      f"res={resolution}, N={loc_feature.shape[0]}")
        
        # 深度渲染用最细分辨率
        fine_res = None
        for name in ['fine', 'fine_sd', 'fine_dino']:
            if name in self.scale_info:
                fine_res = self.scale_info[name]['resolution']
                break
        if fine_res is None:
            fine_res = (35, 46)
        self.depth_H, self.depth_W = fine_res
        self.depth_fx = fx * (self.depth_W / img_width)
        self.depth_fy = fy * (self.depth_H / img_height)
        self.depth_cx = cx * (self.depth_W / img_width)
        self.depth_cy = cy * (self.depth_H / img_height)
        
        print(f"[MultiScaleRenderer] Loaded {len(self.models)} scale models: "
              f"{list(self.models.keys())}, depth_res={self.depth_H}×{self.depth_W}"
              + (f", sharpen={self.sharpen_strength}" if self.sharpen_strength > 0 else ""))

    def _sharpen_features(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Unsharp mask to counteract alpha-blending low-pass filtering.

        Alpha-blending averages ~10-50 overlapping Gaussians per pixel,
        producing blurred features. This sharpens them:
          feat' = feat + strength * (feat - blur(feat))
        then re-L2-normalizes per pixel.

        Args:
            feat: (C, H, W) or (B, C, H, W) L2-normalized feature map

        Returns:
            same shape, sharpened and re-L2-normalized
        """
        if self.sharpen_strength <= 0:
            return feat

        squeeze = feat.ndim == 3
        if squeeze:
            feat = feat.unsqueeze(0)  # (1, C, H, W)

        k = self.sharpen_kernel_size
        padding = k // 2
        # Average-pool blur (depthwise, per-channel)
        blurred = F.avg_pool2d(feat, k, stride=1, padding=padding)
        # Unsharp mask
        sharpened = feat + self.sharpen_strength * (feat - blurred)
        # Re-L2-normalize per pixel
        sharpened = F.normalize(sharpened, p=2, dim=1)

        if squeeze:
            sharpened = sharpened.squeeze(0)
        return sharpened
    
    @torch.no_grad()
    def render_scale(
        self,
        scale_name: str,
        pose_w2c: torch.Tensor,
        return_depth: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        渲染单个尺度的特征图
        
        Args:
            scale_name: 尺度名称
            pose_w2c: (4, 4) world-to-camera 变换矩阵
            return_depth: 是否同时渲染深度图
            
        Returns:
            dict:
                'feature_map': (D, fH, fW) 特征图 (L2 normalized)
                'depth_map': (H, W) 深度图 (仅当 return_depth=True)
        """
        assert scale_name in self.scale_info, \
            f"Scale '{scale_name}' not loaded. Available: {list(self.scale_info.keys())}"
        
        info = self.scale_info[scale_name]
        fH, fW = info['resolution']

        if self.triplane_mode:
            model = self.models['triplane']
            # Render full concatenated feature at this scale's resolution
            total_dim = sum(self.scale_info[s]['feat_dim'] for s in self.scale_names)
        else:
            model = self.models[scale_name]
            total_dim = info['feat_dim']
        
        # 缩放内参以匹配特征分辨率
        scale_x = fW / self.img_width
        scale_y = fH / self.img_height
        render_fx = self.fx * scale_x
        render_fy = self.fy * scale_y
        render_cx = self.cx * scale_x
        render_cy = self.cy * scale_y
        
        result = FeatureRenderer.render_features(
            gaussian_model=model,
            viewmat=pose_w2c,
            fx=render_fx, fy=render_fy,
            cx=render_cx, cy=render_cy,
            img_height=fH,
            img_width=fW,
            feature_height=fH,
            feature_width=fW,
            norm_feat_before_render=True,
            norm_feat_after_render=True,
            max_channels_per_chunk=128,
        )
        
        feat_map = result['feature_map']  # (D_total, fH, fW)
        
        if self.triplane_mode:
            # Slice the scale's channels from the concatenated output
            offset = info['dim_offset']
            dim = info['feat_dim']
            feat_map = feat_map[offset:offset + dim]
            feat_map = F.normalize(feat_map, p=2, dim=0)
        
        # Apply feature sharpening (counteract alpha-blending blur)
        feat_map = self._sharpen_features(feat_map)
        
        out = {'feature_map': feat_map}
        
        if return_depth:
            out['depth_map'] = self._render_depth(model, pose_w2c)
        
        return out
    
    @torch.no_grad()
    def render_all_scales(
        self,
        pose_w2c: torch.Tensor,
        scales: List[str] = None,
        return_depth: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        渲染所有(或指定)尺度的特征图
        
        Args:
            pose_w2c: (4, 4) world-to-camera
            scales: 要渲染的尺度列表 (None = 全部)
            return_depth: 是否渲染深度图 (只渲染一次，来自第一个模型)
        
        Returns:
            dict:
                '{scale_name}_feat': (D_i, H_i, W_i) per-scale 特征图
                'depth_map': (H, W) 深度图 (若 return_depth)
        """
        if scales is None:
            scales = list(self.scale_info.keys())
        
        result = {}
        
        for i, name in enumerate(scales):
            need_depth = return_depth and (i == 0)
            r = self.render_scale(name, pose_w2c, return_depth=need_depth)
            result[f'{name}_feat'] = r['feature_map']
            if 'depth_map' in r:
                result['depth_map'] = r['depth_map']
        
        return result
    
    @torch.no_grad()
    def render_batch(
        self,
        poses_w2c: torch.Tensor,
        scales: List[str] = None,
        return_depth: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        批量渲染 (利用gsplat v1.5+ 原生batch rendering，一次CUDA调用渲染多视角)
        
        Args:
            poses_w2c: (B, 4, 4) batch of w2c poses
            scales: 尺度列表
            return_depth: 是否返回深度
            
        Returns:
            dict:
                '{scale_name}_feat': (B, D_i, H_i, W_i)
                'depth_map': (B, H, W) 如果 return_depth
        """
        if scales is None:
            scales = list(self.scale_info.keys())
        
        B = poses_w2c.shape[0]
        out = {}
        
        for name in scales:
            model = self.models['triplane'] if self.triplane_mode else self.models[name]
            info = self.scale_info[name]
            fH, fW = info['resolution']
            
            scale_x = fW / self.img_width
            scale_y = fH / self.img_height
            render_fx = self.fx * scale_x
            render_fy = self.fy * scale_y
            render_cx = self.cx * scale_x
            render_cy = self.cy * scale_y
            
            result = FeatureRenderer.render_features_batch(
                gaussian_model=model,
                viewmats=poses_w2c,          # [B, 4, 4] — 原生batch!
                fx=render_fx, fy=render_fy,
                cx=render_cx, cy=render_cy,
                img_height=fH,
                img_width=fW,
                feature_height=fH,
                feature_width=fW,
                norm_feat_before_render=True,
                norm_feat_after_render=True,
                max_channels_per_chunk=128,
            )
            
            feat = result['feature_map']  # [B, D_total, fH, fW]
            
            if self.triplane_mode:
                # Slice this scale's channels from concatenated output
                offset = info['dim_offset']
                dim = info['feat_dim']
                feat = feat[:, offset:offset + dim]
                feat = F.normalize(feat, p=2, dim=1)
            
            # Apply feature sharpening (counteract alpha-blending blur)
            feat = self._sharpen_features(feat)
            
            out[f'{name}_feat'] = feat
        
        if return_depth:
            # 使用第一个模型渲染深度 (批量)
            first_model = self.models['triplane'] if self.triplane_mode else self.models[scales[0]]
            from feature_3dgs.feature_renderer import _build_K
            K = _build_K(self.depth_fx, self.depth_fy, 
                        self.depth_cx, self.depth_cy, poses_w2c.device)
            Ks = K.unsqueeze(0).expand(B, -1, -1)
            
            dummy_colors = torch.zeros(
                first_model.get_xyz.shape[0], 1, device=poses_w2c.device)
            
            is_2dgs = getattr(first_model, 'is_2dgs', False)
            from gsplat import rasterization
            gauss_scales = first_model.get_scaling_for_render if is_2dgs else first_model.get_scaling
            render_colors, _, _ = rasterization(
                means=first_model.get_xyz,
                quats=first_model.get_rotation,
                scales=gauss_scales,
                opacities=first_model.get_opacity.squeeze(-1),
                colors=dummy_colors,
                viewmats=poses_w2c,
                Ks=Ks,
                width=self.depth_W,
                height=self.depth_H,
                packed=True,
                render_mode='D',
                near_plane=0.01,
                far_plane=1e5,
            )
            out['depth_map'] = render_colors[:, :, :, 0]  # [B, H, W]
        
        return out

    @torch.no_grad()
    def render_depth_batch(
        self,
        poses_w2c: torch.Tensor,
    ) -> torch.Tensor:
        """Render depth only (no features) for a batch of poses.
        
        Returns: (B, H, W) depth map at fine resolution.
        """
        first_model = list(self.models.values())[0]
        B = poses_w2c.shape[0]
        from feature_3dgs.feature_renderer import _build_K
        K = _build_K(self.depth_fx, self.depth_fy,
                     self.depth_cx, self.depth_cy, poses_w2c.device)
        Ks = K.unsqueeze(0).expand(B, -1, -1)
        dummy_colors = torch.zeros(
            first_model.get_xyz.shape[0], 1, device=poses_w2c.device)

        is_2dgs = getattr(first_model, 'is_2dgs', False)
        # Use regular rasterization for depth (works for both 3DGS and 2DGS with padded scales)
        from gsplat import rasterization
        scales = first_model.get_scaling_for_render if is_2dgs else first_model.get_scaling
        render_colors, _, _ = rasterization(
            means=first_model.get_xyz,
            quats=first_model.get_rotation,
            scales=scales,
            opacities=first_model.get_opacity.squeeze(-1),
            colors=dummy_colors,
            viewmats=poses_w2c,
            Ks=Ks,
            width=self.depth_W,
            height=self.depth_H,
            packed=True,
            render_mode='D',
            near_plane=0.01, far_plane=1e5,
        )
        return render_colors[:, :, :, 0]  # [B, H, W]

    def get_scale_intrinsics(self) -> dict:
        """Get per-scale camera intrinsics for depth warping."""
        result = {}
        for name, info in self.scale_info.items():
            H, W = info['resolution']
            result[name] = {
                'fx': self.fx * W / self.img_width,
                'fy': self.fy * H / self.img_height,
                'cx': self.cx * W / self.img_width,
                'cy': self.cy * H / self.img_height,
            }
        return result
    
    def _render_depth(
        self,
        model: GaussianFeatureModel,
        pose_w2c: torch.Tensor,
    ) -> torch.Tensor:
        """
        渲染深度图 (使用 gsplat v1.5+ 内置深度渲染模式)
        
        Returns:
            (H, W) depth map
        """
        return FeatureRenderer.render_depth(
            gaussian_model=model,
            viewmat=pose_w2c,
            fx=self.depth_fx, fy=self.depth_fy,
            cx=self.depth_cx, cy=self.depth_cy,
            img_height=self.depth_H,
            img_width=self.depth_W,
        )
    
    def get_scale_info(self) -> Dict[str, Dict]:
        """返回所有已加载尺度的信息"""
        return self.scale_info
    
    def get_available_scales(self) -> List[str]:
        """返回可用的尺度名称列表"""
        return list(self.scale_info.keys())
