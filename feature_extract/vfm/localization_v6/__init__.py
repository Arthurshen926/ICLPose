"""V6 maplet-atlas correlation localization.

This package is intentionally independent from the V5 point-similarity
alignment implementation.  V6 predicts image displacement distributions first
and converts them to pose updates with explicit projective geometry.
"""

from .atlas_baking import bake_feature_atlas
from .maplet_atlas import MapletFeatureAtlasBank

__all__ = [
    "bake_feature_atlas",
    "MapletFeatureAtlasBank",
]
