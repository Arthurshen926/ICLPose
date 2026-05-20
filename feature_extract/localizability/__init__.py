"""POFD-FS utilities for localization-usable foundation feature selection."""

from .candidate_bank import CandidateBank, CandidateBankMetadata, candidate_bank_from_npz
from .losses import basin_bce_loss, online_score_hard_negative_loss, pose_distance_soft_rank_loss
from .mapability import track_feature_variance
from .metrics import ranking_metrics
from .scorer import PoseHypothesisScorer
from .selector import LocalizationFeatureSelector

__all__ = [
    "CandidateBank",
    "CandidateBankMetadata",
    "LocalizationFeatureSelector",
    "PoseHypothesisScorer",
    "basin_bce_loss",
    "candidate_bank_from_npz",
    "online_score_hard_negative_loss",
    "pose_distance_soft_rank_loss",
    "ranking_metrics",
    "track_feature_variance",
]
