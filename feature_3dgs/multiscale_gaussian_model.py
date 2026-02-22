"""
Multi-Scale Gaussian Feature Model
====================================
在 GaussianFeatureModel 基础上，支持多尺度特征嵌入的拆分和渲染。

每个 Gaussian 存储一个 total_dim=224 的特征向量:
  [fine_sd(64) | fine_dino(64) | mid(64) | coarse(32)]

渲染后按偏移量拆分出 3 个尺度的特征图:
  - fine:   [128, fH, fW] = fine_sd + fine_dino
  - mid:    [64, mH, mW]
  - coarse: [32, cH, cW]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel


class MultiScaleGaussianModel(GaussianFeatureModel):
    """
    多尺度特征嵌入的 3DGS 模型。
    
    继承 GaussianFeatureModel，添加多尺度 split 信息。
    """

    # 各尺度维度 (压缩后)
    FINE_SD_DIM = 64
    FINE_DINO_DIM = 64
    MID_DIM = 64
    COARSE_DIM = 32
    TOTAL_DIM = FINE_SD_DIM + FINE_DINO_DIM + MID_DIM + COARSE_DIM  # 224

    # 偏移量
    FINE_SD_START = 0
    FINE_SD_END = FINE_SD_DIM
    FINE_DINO_START = FINE_SD_END
    FINE_DINO_END = FINE_DINO_START + FINE_DINO_DIM
    FINE_END = FINE_DINO_END          # 128
    MID_START = FINE_END
    MID_END = MID_START + MID_DIM     # 192
    COARSE_START = MID_END
    COARSE_END = COARSE_START + COARSE_DIM  # 224

    def __init__(self):
        super().__init__(feature_dim=self.TOTAL_DIM)

    @property
    def get_fine_feature(self) -> torch.Tensor:
        """[N, 128] fine_sd + fine_dino (L2 normalized)"""
        feat = self._loc_feature[:, :self.FINE_END]
        return F.normalize(feat, p=2, dim=-1)

    @property
    def get_mid_feature(self) -> torch.Tensor:
        """[N, 64] mid (L2 normalized)"""
        feat = self._loc_feature[:, self.MID_START:self.MID_END]
        return F.normalize(feat, p=2, dim=-1)

    @property
    def get_coarse_feature(self) -> torch.Tensor:
        """[N, 32] coarse (L2 normalized)"""
        feat = self._loc_feature[:, self.COARSE_START:self.COARSE_END]
        return F.normalize(feat, p=2, dim=-1)

    @staticmethod
    def split_feature_map(feature_map: torch.Tensor) -> dict:
        """
        将渲染后的 [224, H, W] 特征图拆分为多尺度。
        
        Args:
            feature_map: [224, H, W] 完整渲染特征图
            
        Returns:
            dict: {
                'fine': [128, H, W],
                'mid':  [64, H, W],
                'coarse': [32, H, W],
            }
        """
        S = MultiScaleGaussianModel
        return {
            'fine': feature_map[S.FINE_SD_START:S.FINE_END],
            'mid': feature_map[S.MID_START:S.MID_END],
            'coarse': feature_map[S.COARSE_START:S.COARSE_END],
        }
