"""Legacy compatibility facade for FeatureExtract.

Active code should import from feature_extract.*.
"""

from feature_extract.extractors.fused_feature_extractor import FusedFeatureExtractor


__all__ = ["FusedFeatureExtractor"]
