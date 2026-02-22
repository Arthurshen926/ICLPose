"""
Raw (Per-Scale) Gaussian Feature Model
=======================================
支持按尺度独立训练的 3DGS 特征嵌入模型。

每个实例只负责一个尺度的特征嵌入:
  - fine_sd:   640d per Gaussian
  - fine_dino: 768d per Gaussian
  - mid:       1280d per Gaussian
  - coarse:    1280d per Gaussian

内存分析 (417K Gaussians):
  - fine_sd(640d):   param=1.00GB, Adam(m+v)=2.00GB → ~4GB
  - fine_dino(768d): param=1.20GB, Adam(m+v)=2.40GB → ~5GB
  - mid(1280d):      param=1.99GB, Adam(m+v)=3.98GB → ~8GB
  - coarse(1280d):   param=1.99GB, Adam(m+v)=3.98GB → ~8GB

全部 3968d 需要 ~25GB → 超出 24GB 显存，必须按尺度独立训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.raw_multiscale_dataset import RAW_SCALE_CONFIGS


class RawScaleGaussianModel(GaussianFeatureModel):
    """
    单尺度原始维度 Gaussian 特征模型。

    继承 GaussianFeatureModel，根据 scale 名称自动设置特征维度。
    """

    def __init__(self, scale: str = 'fine_sd'):
        """
        Args:
            scale: 尺度名 ('fine_sd', 'fine_dino', 'mid', 'coarse')
        """
        if scale not in RAW_SCALE_CONFIGS:
            raise ValueError(f"未知尺度 '{scale}', 可选: {list(RAW_SCALE_CONFIGS.keys())}")

        self.scale = scale
        self.scale_dim = RAW_SCALE_CONFIGS[scale]['dim']
        self.scale_resolution = RAW_SCALE_CONFIGS[scale]['resolution']

        super().__init__(feature_dim=self.scale_dim)

    def summary(self) -> str:
        return (f"RawScaleGaussianModel(scale={self.scale}, "
                f"dim={self.scale_dim}, "
                f"resolution={self.scale_resolution[1]}×{self.scale_resolution[0]}, "
                f"gaussians={self.num_gaussians:,})")


class RawCombinedGaussianModel(GaussianFeatureModel):
    """
    多尺度组合 Gaussian 特征模型 (合并多个尺度的特征)。

    仅在 GPU 显存足够时使用 (需要 >25GB)。
    组合方式: [fine_sd(640) | fine_dino(768) | mid(1280) | coarse(1280)] = 3968d

    注意: 渲染时需要 124 个 gsplat chunks，训练会非常慢。
          通常建议使用 RawScaleGaussianModel 按尺度独立训练。
    """

    def __init__(self, scales=None):
        if scales is None:
            scales = ['fine_sd', 'fine_dino', 'mid', 'coarse']
        self.scales = scales
        self.scale_dims = {s: RAW_SCALE_CONFIGS[s]['dim'] for s in scales}
        total_dim = sum(self.scale_dims.values())

        # 计算各尺度在总特征向量中的偏移量
        self.scale_offsets = {}
        offset = 0
        for s in scales:
            d = self.scale_dims[s]
            self.scale_offsets[s] = (offset, offset + d)
            offset += d

        super().__init__(feature_dim=total_dim)

    def get_scale_feature(self, scale: str) -> torch.Tensor:
        """获取指定尺度的子特征 [N, D_scale], L2 normalized"""
        start, end = self.scale_offsets[scale]
        feat = self._loc_feature[:, start:end]
        return F.normalize(feat, p=2, dim=-1)

    def split_feature_map(self, feature_map: torch.Tensor) -> dict:
        """
        将渲染后的 [total_dim, H, W] 特征图拆分为各尺度。

        Args:
            feature_map: [total_dim, H, W] 完整渲染特征图

        Returns:
            dict: {scale_name: [D_scale, H, W], ...}
        """
        result = {}
        for s in self.scales:
            start, end = self.scale_offsets[s]
            result[s] = feature_map[start:end]
        return result
