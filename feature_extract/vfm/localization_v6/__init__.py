"""V6 maplet-atlas correlation localization.

This package is intentionally independent from the V5 point-similarity
alignment implementation.  V6 predicts image displacement distributions first
and converts them to pose updates with explicit projective geometry.
"""

from .atlas_baking import bake_feature_atlas
from .atlas_pose_alignment import (
    AtlasAlignmentLevel,
    AtlasPoseAlignmentResult,
    refine_pose_with_maplet_atlases,
)
from .map_entities import (
    ChartExpansionPosterior,
    MetricSurfaceChartBank,
    RegionChartIndex,
    RetrievalRegionBank,
    expand_region_posterior_to_charts,
)
from .maplet_atlas import MapletFeatureAtlasBank
from .structured_frame_adapter import (
    StructuredFrameAdapter,
    StructuredFrameAdapterConfig,
    load_structured_frame_adapter,
    save_structured_frame_adapter,
)

__all__ = [
    "bake_feature_atlas",
    "AtlasAlignmentLevel",
    "AtlasPoseAlignmentResult",
    "refine_pose_with_maplet_atlases",
    "ChartExpansionPosterior",
    "expand_region_posterior_to_charts",
    "MapletFeatureAtlasBank",
    "MetricSurfaceChartBank",
    "RegionChartIndex",
    "RetrievalRegionBank",
    "StructuredFrameAdapter",
    "StructuredFrameAdapterConfig",
    "load_structured_frame_adapter",
    "save_structured_frame_adapter",
]
