"""VFM-MapLoc interfaces.

This package contains the clean mainline for localizable foundation feature
selection and map-conditioned hypothesis verification.
"""

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost, assign_basin_labels
from feature_extract.vfm.candidate_adapters import (
    candidate_from_normalized_record,
    load_candidate_records,
    load_pose_init_npz_records,
    load_reference_pose_bank_npz_records,
    load_score_table_jsonl_records,
)
from feature_extract.vfm.candidate_scoring import score_candidate_bank_by_metadata
from feature_extract.vfm.candidate_labeling import label_candidate_bank_from_score_table
from feature_extract.vfm.cambridge_pose_lattice import (
    CambridgePoseRecord,
    build_cambridge_pose_lattice_bank,
    parse_cambridge_pose_file,
    parse_world_offsets,
    pose_w2c_from_center_rotation,
    quaternion_wxyz_to_rotation_matrix,
)
from feature_extract.vfm.colmap_tracks import (
    ColmapCamera,
    ColmapPoint3D,
    ColmapTrackObservation,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
    read_colmap_points3d_binary,
    load_colmap_track_observations,
)
from feature_extract.vfm.controls import (
    mask_channels_by_utility,
    metadata_only_scores,
    pca_channel_projection,
    random_channel_projection,
    shuffle_features,
)
from feature_extract.vfm.datasets import (
    TokenCandidateExample,
    TokenCandidateIndex,
    build_token_candidate_index,
)
from feature_extract.vfm.descriptor_selector_training import (
    DescriptorSelectorTrainingConfig,
    DescriptorSelectorTrainingRun,
    DescriptorSelectorTrainingSummary,
    run_descriptor_selector_training,
    train_descriptor_selector,
)
from feature_extract.vfm.dense_selector_training import (
    DenseSelectorTrainingConfig,
    DenseSelectorTrainingRun,
    DenseSelectorTrainingSummary,
    run_dense_selector_training,
    train_dense_selector,
)
from feature_extract.vfm.dense_selector_scoring import (
    dense_selected_descriptor,
    score_candidate_bank_by_dense_selector,
)
from feature_extract.vfm.experiments import GateExpectation, VFMExperimentManifest
from feature_extract.vfm.hard_cases import HardCaseCandidate, HardCaseSplits, build_hard_case_splits
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.map_lifting import (
    SelectedTrackFeatureBank,
    TrackFeature,
    TrackBankMapabilitySummary,
    TrackObservation,
    aggregate_selected_tracks,
    load_selected_track_bank_npz,
    mapability_summary,
    save_selected_track_bank_npz,
)
from feature_extract.vfm.mapability_metrics import (
    TrackBankMapabilityReport,
    compare_track_bank_mapability,
    estimate_track_bank_storage_bytes,
)
from feature_extract.vfm.protocols import EvaluationProtocol, ProtocolKind, validate_no_leakage
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    PnPResult,
    PoseError,
    QueryTo3DMatch,
    QueryTo3DMatchingConfig,
    estimate_pose_pnp_ransac,
    filter_landmarks_by_reference_images,
    match_query_tokens_to_landmarks,
    pnp_pose_error,
    reprojection_precision,
    token_grid_xy,
)
from feature_extract.vfm.rendered_map_scoring import (
    RenderedSelectedMapFeature,
    render_selected_track_bank,
    score_rendered_selected_features,
)
from feature_extract.vfm.rendered_map_verifier import (
    RenderedMapEvidenceScore,
    build_track_observation_index,
    build_track_visibility_index,
    build_track_xyz_index,
    project_xyz_to_image,
    projected_selected_map_token_grid,
    rendered_selected_map_token_grid,
    score_candidate_bank_by_projected_rendered_selected_map,
    score_candidate_bank_by_sparse_rendered_selected_map,
    sparse_rendered_map_evidence,
    sparse_rendered_map_score,
)
from feature_extract.vfm.score_table import ScoreRow, ScoreTableReport, evaluate_score_table
from feature_extract.vfm.selected_descriptor_bank import build_selected_descriptor_bank
from feature_extract.vfm.selector import LocalizableFeatureSelector, SelectorOutput
from feature_extract.vfm.selector_descriptor_scoring import (
    infer_selector_dims_from_state_dict,
    load_selector_from_checkpoint,
    score_candidate_bank_by_selector_descriptor_cosine,
)
from feature_extract.vfm.statistics import (
    WilcoxonSignedRankResult,
    mcnemar_exact_pvalue,
    paired_bootstrap_delta_ci,
    wilcoxon_signed_rank,
)
from feature_extract.vfm.tokens import TokenBankManifest, TokenBankRecord, TokenLayerSpec
from feature_extract.vfm.token_manifest_summary import TokenManifestSummary, summarize_token_manifest
from feature_extract.vfm.token_candidate_scoring import score_candidate_bank_by_token_cosine
from feature_extract.vfm.token_descriptor_bank import (
    TokenDescriptorBank,
    build_token_descriptor_bank,
    score_candidate_bank_by_descriptor_cosine,
)
from feature_extract.vfm.track_feature_sampling import (
    load_colmap_track_observations_jsonl,
    sample_token_track_observations,
)
from feature_extract.vfm.training import (
    SyntheticSelectorTrainingConfig,
    SyntheticSelectorTrainingResult,
    run_synthetic_selector_training,
)

__all__ = [
    "CandidateHypothesis",
    "CandidateHypothesisBank",
    "CambridgePoseRecord",
    "ColmapCamera",
    "ColmapPoint3D",
    "ColmapTrackObservation",
    "DenseSelectorTrainingConfig",
    "DenseSelectorTrainingRun",
    "DenseSelectorTrainingSummary",
    "DescriptorSelectorTrainingConfig",
    "DescriptorSelectorTrainingRun",
    "DescriptorSelectorTrainingSummary",
    "EvaluationProtocol",
    "GateExpectation",
    "HardCaseCandidate",
    "HardCaseSplits",
    "LandmarkMapIndex",
    "LocalizableFeatureSelector",
    "PnPResult",
    "PoseCost",
    "PoseError",
    "ProtocolKind",
    "QueryTo3DMatch",
    "QueryTo3DMatchingConfig",
    "RenderedSelectedMapFeature",
    "RenderedMapEvidenceScore",
    "ScoreRow",
    "ScoreTableReport",
    "SelectedTrackFeatureBank",
    "SelectorOutput",
    "SyntheticSelectorTrainingConfig",
    "SyntheticSelectorTrainingResult",
    "TokenBankManifest",
    "TokenBankRecord",
    "TokenCandidateExample",
    "TokenCandidateIndex",
    "TokenDescriptorBank",
    "TokenManifestSummary",
    "TrackBankMapabilitySummary",
    "TrackBankMapabilityReport",
    "TokenLayerSpec",
    "TrackFeature",
    "TrackObservation",
    "VFMExperimentManifest",
    "WilcoxonSignedRankResult",
    "aggregate_selected_tracks",
    "assign_basin_labels",
    "build_cambridge_pose_lattice_bank",
    "build_hard_case_splits",
    "build_track_observation_index",
    "build_track_visibility_index",
    "build_selected_descriptor_bank",
    "build_track_xyz_index",
    "candidate_from_normalized_record",
    "build_token_candidate_index",
    "build_token_descriptor_bank",
    "compare_track_bank_mapability",
    "dense_selected_descriptor",
    "estimate_track_bank_storage_bytes",
    "evaluate_score_table",
    "estimate_pose_pnp_ransac",
    "filter_landmarks_by_reference_images",
    "infer_selector_dims_from_state_dict",
    "load_selected_track_bank_npz",
    "load_selector_from_checkpoint",
    "load_candidate_records",
    "load_colmap_track_observations",
    "load_pose_init_npz_records",
    "load_colmap_track_observations_jsonl",
    "load_reference_pose_bank_npz_records",
    "load_score_table_jsonl_records",
    "label_candidate_bank_from_score_table",
    "mask_channels_by_utility",
    "mapability_summary",
    "metadata_only_scores",
    "mcnemar_exact_pvalue",
    "match_query_tokens_to_landmarks",
    "paired_bootstrap_delta_ci",
    "pca_channel_projection",
    "parse_cambridge_pose_file",
    "parse_world_offsets",
    "pnp_pose_error",
    "pose_w2c_from_center_rotation",
    "project_xyz_to_image",
    "projected_selected_map_token_grid",
    "quaternion_wxyz_to_rotation_matrix",
    "random_channel_projection",
    "read_colmap_cameras_binary",
    "read_colmap_images_binary",
    "read_colmap_points3d_binary",
    "render_selected_track_bank",
    "rendered_selected_map_token_grid",
    "reprojection_precision",
    "save_selected_track_bank_npz",
    "sample_token_track_observations",
    "score_rendered_selected_features",
    "score_candidate_bank_by_metadata",
    "score_candidate_bank_by_descriptor_cosine",
    "score_candidate_bank_by_dense_selector",
    "score_candidate_bank_by_projected_rendered_selected_map",
    "score_candidate_bank_by_sparse_rendered_selected_map",
    "score_candidate_bank_by_selector_descriptor_cosine",
    "score_candidate_bank_by_token_cosine",
    "sparse_rendered_map_score",
    "sparse_rendered_map_evidence",
    "shuffle_features",
    "summarize_token_manifest",
    "token_grid_xy",
    "run_dense_selector_training",
    "run_descriptor_selector_training",
    "run_synthetic_selector_training",
    "train_dense_selector",
    "train_descriptor_selector",
    "validate_no_leakage",
    "wilcoxon_signed_rank",
]
