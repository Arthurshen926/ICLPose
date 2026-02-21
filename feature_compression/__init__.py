# Feature Compression Module
# 提供特征压缩功能

from .autoencoder import AutoencoderFlexible, FEATURE_CONFIGS
from .compressor import FeatureCompressor

__all__ = ['AutoencoderFlexible', 'FEATURE_CONFIGS', 'FeatureCompressor']
