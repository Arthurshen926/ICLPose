"""Legacy compatibility facade for FeatureGaussian / FeatureField.

Active code should import from feature_gaussian.* and feature_field.*.
"""

from feature_gaussian.legacy_3dgs.feature_dataset import FeatureEmbeddingDataset
from feature_gaussian.models.gaussian_feature_model import GaussianFeatureModel
from feature_field.models.feature_renderer import FeatureRenderer
from feature_field.utils.feature_3dgs_provider import Feature3DGSProvider


__all__ = [
    "Feature3DGSProvider",
    "FeatureEmbeddingDataset",
    "FeatureRenderer",
    "GaussianFeatureModel",
]
