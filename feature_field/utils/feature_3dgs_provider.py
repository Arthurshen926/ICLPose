"""
Feature3DGS Provider
====================
为位姿估计训练提供基于特征3DGS的2D-3D对应数据。

核心思想:
  当前方案 (SplatLoc模式):
    - 3D点坐标: 随机采样的Gaussian中心（稀疏）
    - 3D特征:   FeatureDecoder(xyz) —— HashGrid隐式查询
    - 2D特征:   预提取的DINO+SD特征图采样

  新方案 (Feature3DGS模式):
    - 3D点坐标: 渲染深度图 → 反投影 → 准确的可见3D点
    - 3D特征:   渲染特征图 → 在像素位置采样（与3D点一一对应）
    - 2D特征:   预提取的DINO+SD特征图采样（不变）

优势:
  1. 3D点全部来自可见表面（深度渲染保证准确的遮挡处理）
  2. 2D位置与3D位置严格对应（通过深度反投影）
  3. 3D特征和FeatureDecoder特征来自同一嵌入空间（已对齐DINO+SD）
  4. 特征连续且稠密（渲染而非离散采样）
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple

from feature_gaussian.models.gaussian_feature_model import GaussianFeatureModel
from feature_field.models.feature_renderer import FeatureRenderer, _build_K


BLOCK_WIDTH = 16  # gsplat tile size


class Feature3DGSProvider:
    """
    基于特征3DGS的2D-3D对应数据提供器。

    使用流程:
        1. 构造时加载预训练的GaussianFeatureModel（含特征嵌入）
        2. 在每个训练step中，给定相机位姿，调用 render_and_backproject():
             - 渲染深度图 → 反投影得到世界坐标3D点和对应像素坐标
             - 渲染特征图 → 在上述像素坐标处采样得到3D特征
           返回的2D坐标直接与fused_feature对应，3D坐标/特征由渲染保证
    """

    def __init__(
        self,
        feature_ply_path: str,
        feature_dim: int = 256,
        device: str = 'cuda',
        depth_min: float = 0.01,
        depth_max: float = 20.0,
        max_channels_per_chunk: int = 32,
    ):
        """
        Args:
            feature_ply_path: 带特征嵌入的PLY文件路径
                              （由 feature_3dgs/train_feature_embedding.py 训练产出）
            feature_dim: 特征维度（需与训练时一致）
            device: 计算设备
            depth_min: 深度图有效最小值 (m)
            depth_max: 深度图有效最大值 (m)
            max_channels_per_chunk: 渲染时每chunk最大通道数（受CUDA共享内存限制）
        """
        self.feature_dim = feature_dim
        self.device = device
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.max_channels_per_chunk = max_channels_per_chunk

        # 加载带特征嵌入的3DGS
        print(f"[Feature3DGSProvider] 加载特征3DGS: {feature_ply_path}")
        self.gaussian_model = GaussianFeatureModel(feature_dim=feature_dim)

        # 使用 load_ply_with_features() 加载已训练的特征嵌入（loc_* 属性）
        # 与 load_ply() 的区别: load_ply_with_features 读取 PLY 中的 loc_0..loc_D 属性
        self.gaussian_model.load_ply_with_features(feature_ply_path)
        self.gaussian_model = self.gaussian_model.to(device)
        self.gaussian_model.eval()

        # 以模型实际加载的特征维度为准
        self.feature_dim = self.gaussian_model.feature_dim

        # 冻结所有参数（推理时不训练特征嵌入）
        for param in self.gaussian_model.parameters():
            param.requires_grad = False

        print(f"  ✓ Gaussian数量: {self.gaussian_model.num_gaussians}")
        print(f"  ✓ 特征维度: {self.feature_dim}")
        print(f"  ✓ 所有参数已冻结")

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def render_and_backproject(
        self,
        c2w: torch.Tensor,
        fx: float, fy: float,
        cx: float, cy: float,
        img_height: int, img_width: int,
        num_samples: int = 1024,
        sample_stride: int = 1,
        norm_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        给定相机位姿，渲染深度图和特征图，反投影得到3D点及其特征。

        Args:
            c2w: (4, 4) camera-to-world矩阵（与数据集格式一致）
            fx, fy, cx, cy: 相机内参
            img_height, img_width: 图像分辨率
            num_samples: 期望采样的有效点数量
            sample_stride: 像素采样步长（>1 可加速，减少点数）
            norm_features: 是否L2归一化特征

        Returns:
            dict:
                'points_3d':     (N, 3) 世界坐标系3D点
                'pixel_coords':  (N, 2) 对应像素坐标 (u, v)
                'pcd_feats':     (N, feature_dim) 3D特征（从特征图采样）
                'depth_map':     (H, W) 渲染深度图
                'feature_map':   (feature_dim, H, W) 渲染特征图
                'valid_mask':    (N,) bool，标记有效点
        """
        device = self.device

        # C2W -> W2C（gsplat使用world-to-camera矩阵）
        if not isinstance(c2w, torch.Tensor):
            c2w = torch.tensor(c2w, dtype=torch.float32, device=device)
        c2w = c2w.to(device)
        w2c = torch.inverse(c2w)  # (4, 4)

        # ====== 1. 渲染深度图 (gsplat v1.5+ 内置depth模式) ======
        depth_map = FeatureRenderer.render_depth(
            gaussian_model=self.gaussian_model,
            viewmat=w2c,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_height=img_height,
            img_width=img_width,
        )  # (H, W)

        # ====== 2. 渲染特征图 ======
        feat_result = FeatureRenderer.render_features(
            gaussian_model=self.gaussian_model,
            viewmat=w2c,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_height=img_height,
            img_width=img_width,
            norm_feat_before_render=norm_features,
            norm_feat_after_render=norm_features,
            max_channels_per_chunk=self.max_channels_per_chunk,
        )
        feature_map = feat_result['feature_map']  # (feature_dim, H, W)

        # ====== 4. 有效深度像素 → 采样 ======
        valid_depth = (depth_map > self.depth_min) & (depth_map < self.depth_max)
        pixel_vs, pixel_us = torch.where(valid_depth)  # (M,), (M,)

        if pixel_vs.numel() == 0:
            # 没有有效深度点（位姿可能完全离开场景）
            empty = torch.zeros(0, device=device)
            return {
                'points_3d': torch.zeros(0, 3, device=device),
                'pixel_coords': torch.zeros(0, 2, device=device),
                'pcd_feats': torch.zeros(0, self.feature_dim, device=device),
                'depth_map': depth_map,
                'feature_map': feature_map,
                'valid_mask': empty.bool(),
            }

        # 均匀降采样到 num_samples
        M = pixel_vs.numel()
        if M > num_samples:
            perm = torch.randperm(M, device=device)[:num_samples]
            pixel_vs = pixel_vs[perm]
            pixel_us = pixel_us[perm]

        # ====== 5. 深度反投影 → 世界坐标3D点 ======
        z = depth_map[pixel_vs, pixel_us]  # (N,), camera-space depth
        x_c = (pixel_us.float() - cx) / fx * z
        y_c = (pixel_vs.float() - cy) / fy * z
        points_cam = torch.stack([x_c, y_c, z], dim=-1)  # (N, 3)

        # Camera → World: P_world = R_c2w * P_cam + t_c2w
        R_c2w = c2w[:3, :3]  # (3, 3)
        t_c2w = c2w[:3, 3]   # (3,)
        points_world = (R_c2w @ points_cam.T).T + t_c2w  # (N, 3)

        # ====== 6. 在特征图上采样3D点的特征 ======
        pixel_coords = torch.stack([pixel_us.float(), pixel_vs.float()], dim=-1)  # (N, 2) -> (u, v)
        pcd_feats = self._sample_features_at_pixels(
            feature_map, pixel_us, pixel_vs, img_height, img_width
        )  # (N, feature_dim)

        if norm_features:
            pcd_feats = F.normalize(pcd_feats, p=2, dim=-1)

        valid = torch.ones(points_world.shape[0], dtype=torch.bool, device=device)

        return {
            'points_3d': points_world,       # (N, 3)
            'pixel_coords': pixel_coords,    # (N, 2) (u, v)
            'pcd_feats': pcd_feats,          # (N, feature_dim)
            'depth_map': depth_map,          # (H, W)
            'feature_map': feature_map,      # (feature_dim, H, W)
            'valid_mask': valid,             # (N,)
        }

    @torch.no_grad()
    def render_feature_map_only(
        self,
        c2w: torch.Tensor,
        fx: float, fy: float,
        cx: float, cy: float,
        img_height: int, img_width: int,
        norm_features: bool = True,
    ) -> torch.Tensor:
        """
        仅渲染特征图，不做反投影。
        用于: 先拿到特征图，再到训练循环中按需采样。

        Returns:
            feature_map: (feature_dim, H, W)
        """
        if not isinstance(c2w, torch.Tensor):
            c2w = torch.tensor(c2w, dtype=torch.float32, device=self.device)
        w2c = torch.inverse(c2w.to(self.device))

        result = FeatureRenderer.render_features(
            gaussian_model=self.gaussian_model,
            viewmat=w2c,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_height=img_height,
            img_width=img_width,
            norm_feat_before_render=norm_features,
            norm_feat_after_render=norm_features,
            max_channels_per_chunk=self.max_channels_per_chunk,
        )
        return result['feature_map']  # (feature_dim, H, W)

    # ------------------------------------------------------------------
    # 批量接口（供 train.py 使用）
    # ------------------------------------------------------------------

    @torch.no_grad()
    def render_batch(
        self,
        c2w_batch: torch.Tensor,
        fx: float, fy: float,
        cx: float, cy: float,
        img_height: int, img_width: int,
        num_samples: int = 1024,
        norm_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        批量渲染并反投影，返回 padded 的 batch 张量。

        Args:
            c2w_batch: (B, 4, 4) 每帧的 camera-to-world 位姿
            num_samples: 每帧期望采样的3D点数量

        Returns:
            dict:
                'points_3d':    (B, N, 3)
                'pixel_coords': (B, N, 2)
                'pcd_feats':    (B, N, feature_dim)
                'feature_maps': (B, feature_dim, H, W) 渲染特征图（可供2D采样）
                'depth_maps':   (B, H, W)
                'pts_per_sample': list[int] 每帧实际有效点数
        """
        B = c2w_batch.shape[0]
        device = self.device

        all_points_3d = []
        all_pixel_coords = []
        all_pcd_feats = []
        all_feature_maps = []
        all_depth_maps = []
        pts_per_sample = []

        for b in range(B):
            result = self.render_and_backproject(
                c2w=c2w_batch[b],
                fx=fx, fy=fy,
                cx=cx, cy=cy,
                img_height=img_height,
                img_width=img_width,
                num_samples=num_samples,
                norm_features=norm_features,
            )
            N_b = result['points_3d'].shape[0]
            all_points_3d.append(result['points_3d'])
            all_pixel_coords.append(result['pixel_coords'])
            all_pcd_feats.append(result['pcd_feats'])
            all_feature_maps.append(result['feature_map'])
            all_depth_maps.append(result['depth_map'])
            pts_per_sample.append(N_b)

        # Pad 到 max_N
        max_N = max(pts_per_sample) if pts_per_sample else num_samples

        def pad_to(tensor, max_n, fill=0.0):
            n = tensor.shape[0]
            if n == max_n:
                return tensor
            pad = torch.full(
                (max_n - n, *tensor.shape[1:]),
                fill, dtype=tensor.dtype, device=device
            )
            return torch.cat([tensor, pad], dim=0)

        points_3d  = torch.stack([pad_to(p, max_N)   for p in all_points_3d],  dim=0)  # (B, N, 3)
        pixel_coords = torch.stack([pad_to(p, max_N)  for p in all_pixel_coords], dim=0)  # (B, N, 2)
        pcd_feats  = torch.stack([pad_to(f, max_N)   for f in all_pcd_feats],   dim=0)  # (B, N, D)
        feature_maps = torch.stack(all_feature_maps, dim=0)  # (B, D, H, W)
        depth_maps   = torch.stack(all_depth_maps,   dim=0)  # (B, H, W)

        return {
            'points_3d':    points_3d,
            'pixel_coords': pixel_coords,
            'pcd_feats':    pcd_feats,
            'feature_maps': feature_maps,
            'depth_maps':   depth_maps,
            'pts_per_sample': pts_per_sample,
        }

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _render_depth(
        self,
        w2c, fx, fy, cx, cy, img_height, img_width, device,
    ) -> torch.Tensor:
        """
        渲染深度图 (gsplat v1.5+ 内置depth模式)。

        Returns:
            depth_map: (H, W) camera-space depth
        """
        return FeatureRenderer.render_depth(
            gaussian_model=self.gaussian_model,
            viewmat=w2c,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_height=img_height,
            img_width=img_width,
        )

    def _render_feature_map(
        self,
        w2c, fx, fy, cx, cy, img_height, img_width, device,
        norm_features: bool = True,
    ) -> torch.Tensor:
        """
        渲染特征图 (gsplat v1.5+ 内置channel_chunk)。

        Returns:
            feature_map: (feature_dim, H, W)
        """
        result = FeatureRenderer.render_features(
            gaussian_model=self.gaussian_model,
            viewmat=w2c,
            fx=fx, fy=fy,
            cx=cx, cy=cy,
            img_height=img_height,
            img_width=img_width,
            norm_feat_before_render=norm_features,
            norm_feat_after_render=norm_features,
            max_channels_per_chunk=self.max_channels_per_chunk,
        )
        return result['feature_map']

    def _sample_features_at_pixels(
        self,
        feature_map: torch.Tensor,
        pixel_us: torch.Tensor,
        pixel_vs: torch.Tensor,
        img_height: int,
        img_width: int,
    ) -> torch.Tensor:
        """
        在整数像素位置采样特征（双线性插值）。

        Args:
            feature_map: (D, H, W)
            pixel_us:   (N,) u坐标（列，float）
            pixel_vs:   (N,) v坐标（行，float）

        Returns:
            feats: (N, D)
        """
        D, H, W = feature_map.shape
        N = pixel_us.shape[0]

        # 归一化到 [-1, 1] for F.grid_sample
        grid_x = 2.0 * pixel_us.float() / (W - 1) - 1.0  # (N,)
        grid_y = 2.0 * pixel_vs.float() / (H - 1) - 1.0  # (N,)
        grid = torch.stack([grid_x, grid_y], dim=-1)       # (N, 2)
        grid = grid.unsqueeze(0).unsqueeze(0)              # (1, 1, N, 2)

        # grid_sample: (1, D, H, W) + (1, 1, N, 2) -> (1, D, 1, N)
        feats = F.grid_sample(
            feature_map.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # (1, D, 1, N)

        feats = feats.squeeze(2).squeeze(0).T  # (N, D)
        return feats

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @property
    def num_gaussians(self) -> int:
        return self.gaussian_model.num_gaussians
