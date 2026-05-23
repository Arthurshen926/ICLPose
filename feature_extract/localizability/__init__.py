"""POFD-FS utilities for localization-usable foundation feature selection."""

from .candidate_bank import CandidateBank, CandidateBankMetadata, candidate_bank_from_npz
from .bank_schema import CandidateRow, validate_no_forbidden_training_inputs
from .losses import basin_bce_loss, online_score_hard_negative_loss, pose_distance_soft_rank_loss
from .mapability import track_feature_variance
from .metrics import ranking_metrics
from .protocol import ArtifactProtocol, assert_protocol_claims_compatible, validate_protocol_metadata
from .refinement_policy import build_guarded_refinement_entries, pose_delta_trans_rot
from .reporting import ProtocolResult, format_protocol_summary_markdown
from .rendered_map_scoring import render_selected_track_feature_maps, score_projected_selected_track_bank
from .scorer import PoseHypothesisScorer
from .selected_feature_map import (
    SelectedTrackFeatureBank,
    aggregate_selected_track_features,
    hard_hypothesis_selection_targets,
    load_selected_track_feature_bank,
    localization_feature_utility,
    save_selected_track_feature_bank,
    score_query_with_selected_map_features,
    weak_joint_selected_feature_loss,
)
from .selector import LocalizationFeatureSelector

__all__ = [
    "CandidateBank",
    "CandidateBankMetadata",
    "CandidateRow",
    "LocalizationFeatureSelector",
    "PoseHypothesisScorer",
    "ArtifactProtocol",
    "ProtocolResult",
    "SelectedTrackFeatureBank",
    "aggregate_selected_track_features",
    "assert_protocol_claims_compatible",
    "basin_bce_loss",
    "build_guarded_refinement_entries",
    "candidate_bank_from_npz",
    "format_protocol_summary_markdown",
    "hard_hypothesis_selection_targets",
    "load_selected_track_feature_bank",
    "localization_feature_utility",
    "online_score_hard_negative_loss",
    "pose_delta_trans_rot",
    "pose_distance_soft_rank_loss",
    "ranking_metrics",
    "render_selected_track_feature_maps",
    "save_selected_track_feature_bank",
    "score_query_with_selected_map_features",
    "score_projected_selected_track_bank",
    "track_feature_variance",
    "validate_protocol_metadata",
    "validate_no_forbidden_training_inputs",
    "weak_joint_selected_feature_loss",
]
