"""
Models module from SplatLoc

包含特征解码器和编码器。
"""

from .decoders import FeatureDecoder
from .encoding import get_encoder

__all__ = ['FeatureDecoder', 'get_encoder']
