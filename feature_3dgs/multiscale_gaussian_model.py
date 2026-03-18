"""
Multi-Scale Gaussian Feature Model
====================================
在 GaussianFeatureModel 基础上，支持多尺度特征嵌入的拆分和渲染。

每个 Gaussian 存储一个 total_dim 的特征向量 (默认224):
  [fine_sd(D_fine) | fine_dino(D_fine) | mid(D_mid) | coarse(D_coarse)]

渲染后按偏移量拆分出 3 个尺度的特征图。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel


class MultiScaleGaussianModel(GaussianFeatureModel):
    """
    多尺度特征嵌入的 3DGS 模型。支持可配置维度。
    """

    # 默认维度 (兼容旧模型)
    FINE_SD_DIM = 64
    FINE_DINO_DIM = 64
    MID_DIM = 64
    COARSE_DIM = 32
    TOTAL_DIM = FINE_SD_DIM + FINE_DINO_DIM + MID_DIM + COARSE_DIM  # 224

    # 默认偏移量
    FINE_SD_START = 0
    FINE_SD_END = FINE_SD_DIM
    FINE_DINO_START = FINE_SD_END
    FINE_DINO_END = FINE_DINO_START + FINE_DINO_DIM
    FINE_END = FINE_DINO_END          # 128
    MID_START = FINE_END
    MID_END = MID_START + MID_DIM     # 192
    COARSE_START = MID_END
    COARSE_END = COARSE_START + COARSE_DIM  # 224

    def __init__(self, fine_sd_dim=64, fine_dino_dim=64, mid_dim=64, coarse_dim=32):
        total_dim = fine_sd_dim + fine_dino_dim + mid_dim + coarse_dim
        super().__init__(feature_dim=total_dim)
        
        # Store instance-level dimensions (override class defaults)
        self._fine_sd_dim = fine_sd_dim
        self._fine_dino_dim = fine_dino_dim
        self._mid_dim = mid_dim
        self._coarse_dim = coarse_dim
        self._total_dim = total_dim
        
        # Instance-level offsets
        self._fine_sd_start = 0
        self._fine_sd_end = fine_sd_dim
        self._fine_dino_start = fine_sd_dim
        self._fine_dino_end = fine_sd_dim + fine_dino_dim
        self._fine_end = fine_sd_dim + fine_dino_dim
        self._mid_start = self._fine_end
        self._mid_end = self._mid_start + mid_dim
        self._coarse_start = self._mid_end
        self._coarse_end = self._coarse_start + coarse_dim

    @property
    def get_fine_feature(self) -> torch.Tensor:
        feat = self._loc_feature[:, :self._fine_end]
        return F.normalize(feat, p=2, dim=-1)

    @property
    def get_mid_feature(self) -> torch.Tensor:
        feat = self._loc_feature[:, self._mid_start:self._mid_end]
        return F.normalize(feat, p=2, dim=-1)

    @property
    def get_coarse_feature(self) -> torch.Tensor:
        feat = self._loc_feature[:, self._coarse_start:self._coarse_end]
        return F.normalize(feat, p=2, dim=-1)

    def split_feature_map(self, feature_map: torch.Tensor) -> dict:
        return {
            'fine': feature_map[self._fine_sd_start:self._fine_end],
            'mid': feature_map[self._mid_start:self._mid_end],
            'coarse': feature_map[self._coarse_start:self._coarse_end],
        }
