"""Real-image-first RADIO localization components."""

from feature_extract.vfm.localization.coarse_matcher import CoarseMatcher, MatchaCoarseMatcher
from feature_extract.vfm.localization.feature_mapper import AdapterFeatureMapper, FeatureMapper
from feature_extract.vfm.localization.model import MeasurementBranch, SelectorCoarseMeasurementModel
from feature_extract.vfm.localization.schemas import (
    CoarseProposal,
    FeatureMapPair,
    LocalizationMatchResult,
    MappedFeatureMap,
    MeasurementResult,
)

__all__ = [
    "AdapterFeatureMapper",
    "CoarseMatcher",
    "CoarseProposal",
    "FeatureMapper",
    "FeatureMapPair",
    "LocalizationMatchResult",
    "MappedFeatureMap",
    "MatchaCoarseMatcher",
    "MeasurementBranch",
    "MeasurementResult",
    "SelectorCoarseMeasurementModel",
]
