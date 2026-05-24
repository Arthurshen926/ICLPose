"""Feature extractor implementations."""

from feature_extract.extractors.extractor_dino import ViTExtractor
from feature_extract.extractors.extractor_radio import RADIOFeatureExtractor


__all__ = [
    "RADIOFeatureExtractor",
    "ViTExtractor",
]
