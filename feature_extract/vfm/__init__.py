"""VFM-MapLoc interfaces.

This package contains the clean mainline for localizable foundation feature
selection. It deliberately avoids the legacy POFD/CPR/q50-stage vocabulary.
"""

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost, assign_basin_labels
from feature_extract.vfm.map_lifting import (
    SelectedTrackFeatureBank,
    TrackFeature,
    TrackObservation,
    aggregate_selected_tracks,
)
from feature_extract.vfm.protocols import EvaluationProtocol, ProtocolKind, validate_no_leakage
from feature_extract.vfm.selector import LocalizableFeatureSelector, SelectorOutput

__all__ = [
    "CandidateHypothesis",
    "EvaluationProtocol",
    "LocalizableFeatureSelector",
    "PoseCost",
    "ProtocolKind",
    "SelectedTrackFeatureBank",
    "SelectorOutput",
    "TrackFeature",
    "TrackObservation",
    "aggregate_selected_tracks",
    "assign_basin_labels",
    "validate_no_leakage",
]
