"""
Feature 3DGS Module
===================
实现3DGS几何/外观重建与特征重建的解耦。
在预训练好的3DGS基础上，冻结几何和外观参数，仅训练每个Gaussian的特征嵌入。

参考: STDLoc (https://github.com/...)

核心思路:
1. 加载预训练好的3DGS (PLY文件)，冻结 xyz, rotation, scaling, opacity, SH
2. 为每个Gaussian添加可学习的特征嵌入 (feature embedding)
3. 通过gsplat渲染特征图，与预提取的GT特征图做L1 loss
4. 仅优化特征嵌入参数
"""

from .gaussian_feature_model import GaussianFeatureModel
from .feature_renderer import FeatureRenderer
from .feature_dataset import FeatureEmbeddingDataset
from .feature_3dgs_provider import Feature3DGSProvider

__all__ = [
    'GaussianFeatureModel',
    'FeatureRenderer',
    'FeatureEmbeddingDataset',
    'Feature3DGSProvider',
]
