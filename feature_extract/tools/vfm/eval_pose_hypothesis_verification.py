"""Evaluate held-out multi-hypothesis PnP on frozen global top-L proposals.

All pose hypotheses are selected without ground-truth pose access. Ground
truth is used only after a final pose has been returned, to report validation
metrics. A reused late block is evaluated only under the explicit development
cross-block flag and can never produce a production claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from feature_extract.tools.vfm.eval_global_partial_assignment import (
    _load_frozen_baseline_policy,
    _validate_frozen_baseline_pose,
)
from feature_extract.tools.vfm.probe_detector_maplet_geometry import _pose_gate
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import (
    camera_center_from_qvec_tvec,
    qvec_to_rotmat,
    read_colmap_cameras_binary,
    read_colmap_images_binary,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.candidate_pose_evidence import (
    CANDIDATE_POSE_EVIDENCE_VERSION,
)
from feature_extract.vfm.localization.candidate_relation_features import (
    RELATION_CHANNELS,
)
from feature_extract.vfm.local_maplet_matching import (
    build_disjoint_maplet_cluster_ids,
    load_local_maplet_support_index_npz,
)
from feature_extract.vfm.localization.latent_correspondence_pnp import LatentEMConfig
from feature_extract.vfm.localization.local_maplet_geometry_probe import (
    canonical_rows_for_track_candidates,
)
from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    CandidateSpatialLikelihood,
    DIAGNOSTIC_GROUPED_HYPOTHESIS_SELECTION_POLICIES,
    GROUPED_HYPOTHESIS_SELECTION_POLICIES,
    GroupedCandidatePnPConfig,
    GroupedProsacProfile,
    PoseVerificationCandidatePool,
    VerifiedPnPConfig,
    estimate_pose_from_grouped_candidate_pool,
    estimate_pose_with_heldout_verification,
    pose_information_diagnostics,
    resolve_pose_guided_candidate_pool,
    select_crossfit_likelihood_with_immutable_baseline,
    select_geometry_guided_generation_with_immutable_baseline,
    wrap_immutable_pose_on_grouped_denominator,
)
from feature_extract.vfm.localization.pose_safe_selection import (
    global_assignment_score_matrix,
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.measurement_v1.spatial_likelihood_calibration import (
    CandidateSpatialLikelihoodCalibration,
    load_candidate_spatial_likelihood_calibration,
)
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    POSE_VIEW_MIXTURE_SEMANTICS,
)
from feature_extract.vfm.measurement_v1.candidate_measurement_utility import (
    UTILITY_APPLY_STAGE,
    UTILITY_MODEL_FORMAT,
)
from feature_extract.vfm.query_to_3d_matching import (
    PnPResult,
    QueryTo3DMatch,
    estimate_pose_pnp_ransac,
    pnp_pose_error,
)
from feature_extract.vfm.statistics import paired_bootstrap_delta_ci


def _pad_relation_edge_rows(
    rows: list[dict[str, object]],
    key: str,
    *,
    trailing_size: int | None = None,
) -> np.ndarray:
    """Pad variable edge features without inventing evidence for absent edges."""
    values = [np.asarray(row[key], dtype=np.float64) for row in rows]
    max_edges = max((len(value) for value in values), default=0)
    shape = (len(values), max_edges)
    if trailing_size is not None:
        shape += (int(trailing_size),)
    output = np.full(shape, np.nan, dtype=np.float64)
    for row_index, value in enumerate(values):
        if not len(value):
            continue
        expected_shape = (
            (len(value),)
            if trailing_size is None
            else (len(value), trailing_size)
        )
        output[row_index, : len(value)] = value.reshape(expected_shape)
    return output


def _positive_int_list(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or min(output) <= 0:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return output


def _positive_float_list(value: str) -> tuple[float, ...]:
    output = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not output or min(output) <= 0.0 or not np.all(np.isfinite(output)):
        raise argparse.ArgumentTypeError("expected positive comma-separated floats")
    return output


def _integer_list(value: str) -> tuple[int, ...]:
    output = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not output:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return output


def _grouped_prosac_profiles_json(value: str) -> tuple[GroupedProsacProfile, ...]:
    try:
        text = str(value)
        if text.startswith("@"):
            text = Path(text[1:]).read_text()
        payload = json.loads(text)
        if not isinstance(payload, list) or not payload:
            raise ValueError("profile payload must be a non-empty list")
        return tuple(
            GroupedProsacProfile(
                name=str(item["name"]),
                hypotheses_per_limit=int(item["hypotheses_per_limit"]),
                minimal_set_sizes=tuple(
                    int(size) for size in item["minimal_set_sizes"]
                ),
                candidate_probability_power=float(
                    item.get("candidate_probability_power", 1.0)
                ),
                candidate_uniform_mix=float(
                    item.get("candidate_uniform_mix", 0.0)
                ),
                local_optimization=bool(item.get("local_optimization", False)),
                use_spatial_modes=bool(item.get("use_spatial_modes", False)),
            )
            for item in payload
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(
            f"invalid grouped PROSAC profile JSON: {exc}"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--candidate_artifact", required=True)
    parser.add_argument("--score_artifact", required=True)
    parser.add_argument(
        "--candidate_posterior_overlays",
        default="",
        help=(
            "comma-separated target-free graph or candidate-maplet posterior "
            "artifacts averaged as an identity overlay"
        ),
    )
    parser.add_argument("--projected_landmark_bank", required=True)
    parser.add_argument("--maplet_support_index", default=None)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--query_shard_count",
        type=int,
        default=1,
        help="number of deterministic execution shards; does not change inference",
    )
    parser.add_argument(
        "--query_shard_index",
        type=int,
        default=0,
        help="zero-based deterministic execution shard index",
    )
    parser.add_argument(
        "--score_keys",
        default="ensemble__geometry_p05px",
        help="comma-separated compact matcher score arrays",
    )
    parser.add_argument("--baseline_score_key", default="baseline_scores")
    parser.add_argument("--frozen_baseline_summary", default=None)
    parser.add_argument(
        "--frozen_baseline_source_score_key",
        default="strategy__alike_support_top2_mean",
        help="baseline score identity recorded by the frozen global sweep",
    )
    parser.add_argument(
        "--assignment_modes", default="row_argmax,global_bipartite"
    )
    parser.add_argument(
        "--fit_match_counts", type=_positive_int_list, default=(24, 32, 48, 64)
    )
    parser.add_argument(
        "--hypothesis_selection_modes",
        default="score_topk,spatial_round_robin,geometry_diverse",
    )
    parser.add_argument(
        "--ransac_thresholds_px", type=_positive_float_list, default=(2.0, 4.0, 8.0)
    )
    parser.add_argument("--rng_seed_offsets", type=_integer_list, default=(0, 1))
    parser.add_argument("--ransac_iterations", type=int, default=3000)
    parser.add_argument("--holdout_folds", type=int, default=4)
    parser.add_argument("--holdout_fold", type=int, default=0)
    parser.add_argument("--verification_strict_px", type=float, default=2.0)
    parser.add_argument("--verification_loose_px", type=float, default=5.0)
    parser.add_argument("--final_consensus_px", type=float, default=4.0)
    parser.add_argument("--final_refine_f_scale_px", type=float, default=2.0)
    parser.add_argument("--min_final_inliers", type=int, default=6)
    parser.add_argument("--enable_final_refine", action="store_true")
    parser.add_argument(
        "--disable_topl_candidate_pool_verification", action="store_true"
    )
    parser.add_argument("--candidate_pool_residual_sigma_px", type=float, default=2.0)
    parser.add_argument(
        "--candidate_pose_outlier_likelihood", type=float, default=1e-3
    )
    parser.add_argument(
        "--candidate_pose_null_likelihood", type=float, default=1e-3
    )
    parser.add_argument(
        "--candidate_identity_prior_temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature-calibrate candidate identity probabilities within "
            "their retained mass; explicit null mass remains unchanged."
        ),
    )
    parser.add_argument(
        "--candidate_pose_relation_neighbor_k",
        type=int,
        default=0,
        help=(
            "Enable diagnostic candidate-marginalized projected-displacement "
            "factors over this many query-space neighbors; zero disables it."
        ),
    )
    parser.add_argument(
        "--candidate_pose_relation_sigma_px", type=float, default=4.0
    )
    parser.add_argument(
        "--candidate_pose_relation_outlier_likelihood", type=float, default=1e-3
    )
    parser.add_argument(
        "--candidate_relation_feature_neighbor_k",
        type=int,
        default=0,
        help=(
            "Export calibrated-relation input features over this query KNN; "
            "zero keeps the feature path disabled and cannot change pose ranking."
        ),
    )
    parser.add_argument(
        "--candidate_relation_feature_max_modes", type=int, default=1
    )
    parser.add_argument("--candidate_pool_hard_threshold_px", type=float, default=8.0)
    parser.add_argument(
        "--candidate_pool_descriptor_rank_weight", type=float, default=0.02
    )
    parser.add_argument("--candidate_pool_refine_iterations", type=int, default=2)
    parser.add_argument("--candidate_evidence", default="")
    parser.add_argument(
        "--candidate_spatial_likelihood_train",
        default="",
        help="comma-separated target-free train spatial likelihood shards",
    )
    parser.add_argument("--candidate_spatial_likelihood_validation", default="")
    parser.add_argument("--candidate_spatial_likelihood_test", default="")
    parser.add_argument(
        "--candidate_spatial_view_mixture_policy",
        choices=(
            "frozen_candidate_posterior",
            "uniform_all_measured_views",
            "artifact_pose_view_posterior",
        ),
        default="frozen_candidate_posterior",
    )
    parser.add_argument(
        "--candidate_generation_spatial_likelihood_validation", default=""
    )
    parser.add_argument("--candidate_generation_spatial_likelihood_test", default="")
    parser.add_argument(
        "--candidate_generation_spatial_view_mixture_policy",
        choices=(
            "frozen_candidate_posterior",
            "uniform_all_measured_views",
            "artifact_pose_view_posterior",
        ),
        default="frozen_candidate_posterior",
    )
    parser.add_argument(
        "--candidate_spatial_pose_view_geometry_sigma_deg",
        type=float,
        default=0.0,
        help=(
            "Pose-conditioned support-view direction sigma for scoring only; "
            "zero disables geometry reweighting."
        ),
    )
    parser.add_argument(
        "--candidate_generation_spatial_pose_view_geometry_sigma_deg",
        type=float,
        default=0.0,
        help=(
            "Pose-conditioned support-view direction sigma for hypothesis "
            "generation; keep zero when generation is frozen."
        ),
    )
    parser.add_argument("--candidate_spatial_calibration", default="")
    parser.add_argument(
        "--allow_diagnostic_candidate_spatial_calibration", action="store_true"
    )
    parser.add_argument(
        "--allow_legacy_candidate_spatial_dustbin",
        action="store_true",
        help="diagnostic only: allow v3 identity-head dustbin spatial artifacts",
    )
    parser.add_argument(
        "--allow_uncalibrated_candidate_spatial_v4",
        action="store_true",
        help="diagnostic only: run v4 true-residual validity before calibration",
    )
    parser.add_argument(
        "--candidate_spatial_log_evidence_weight", type=float, default=1.0
    )
    parser.add_argument("--candidate_geometry_probabilities_validation", default="")
    parser.add_argument("--candidate_geometry_probabilities_test", default="")
    parser.add_argument("--candidate_geometry_probabilities_train_oof", default="")
    parser.add_argument("--candidate_update_predictions_validation", default="")
    parser.add_argument("--candidate_update_predictions_test", default="")
    parser.add_argument("--candidate_measurement_utility_validation", default="")
    parser.add_argument("--candidate_measurement_utility_test", default="")
    parser.add_argument(
        "--candidate_spatial_utility_gate_weight",
        type=float,
        default=0.0,
        help=(
            "Deprecated semantic violation; measurement utility may only gate "
            "selected-identity coordinate updates and this value must remain zero."
        ),
    )
    parser.add_argument(
        "--enable_optional_candidate_coordinate_refine", action="store_true"
    )
    parser.add_argument("--candidate_coordinate_min_updates", type=int, default=4)
    parser.add_argument(
        "--candidate_coordinate_min_grid_cells", type=int, default=2
    )
    parser.add_argument(
        "--export_train_grouped_hypotheses", action="store_true"
    )
    parser.add_argument(
        "--export_grouped_hypothesis_artifact", action="store_true"
    )
    parser.add_argument(
        "--export_selected_pose_artifact",
        action="store_true",
        help=(
            "export the final inference-selected grouped pose for immutable "
            "cross-run baseline replay"
        ),
    )
    parser.add_argument(
        "--immutable_baseline_pose_artifact",
        default="",
        help=(
            "frozen selected-pose artifact replayed without matcher, assignment, "
            "PnP, or pose refinement"
        ),
    )
    parser.add_argument(
        "--immutable_baseline_pose_evaluation_label",
        default="",
        help=(
            "source policy label when the immutable pose artifact contains more "
            "than one grouped evaluation"
        ),
    )
    parser.add_argument(
        "--candidate_geometry_prior_mix_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_geometry_generation_mix_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--enable_grouped_generation_grid_fallback",
        action="store_true",
        help=(
            "Run an immutable unmixed grouped baseline and promote geometry-guided "
            "generation only when held-out strict grid coverage does not regress."
        ),
    )
    parser.add_argument(
        "--grouped_generation_min_strict_grid_delta", type=int, default=0
    )
    parser.add_argument(
        "--enable_grouped_crossfit_likelihood_fallback",
        action="store_true",
        help=(
            "Promote grouped/latent generation over an immutable assignment "
            "baseline only on a shared component-disjoint likelihood fold."
        ),
    )
    parser.add_argument(
        "--grouped_immutable_baseline_generation_mode",
        choices=("assignment_ransac", "grouped_prosac"),
        default="assignment_ransac",
    )
    parser.add_argument(
        "--grouped_likelihood_min_mean_delta", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_likelihood_min_effective_groups", type=int, default=8
    )
    parser.add_argument(
        "--grouped_observability_min_information_matches", type=int, default=0
    )
    parser.add_argument(
        "--grouped_observability_min_translation_eigenvalue",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--grouped_observability_max_translation_condition",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--grouped_observability_max_joint_condition", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_observability_min_bearing_span_deg", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_observability_min_depth_span_ratio", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_observability_min_xyz_second_ratio", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_observability_min_xyz_third_ratio", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_spatial_geometry_calibration_weight",
        type=float,
        default=0.0,
        help=(
            "Interpolate raw spatial dustbin reliability with the calibrated "
            "true-residual geometry probability without changing identity mass."
        ),
    )
    parser.add_argument("--enable_grouped_candidate_pnp", action="store_true")
    parser.add_argument(
        "--grouped_only",
        action="store_true",
        help="skip unrelated legacy policy sweeps while retaining immutable baselines",
    )
    parser.add_argument(
        "--grouped_null_score_key",
        default="ensemble__set_dustbin_probability_DIAGNOSTIC_ONLY",
    )
    parser.add_argument(
        "--grouped_candidate_limits", type=_positive_int_list, default=(1, 3, 5, 10, 20)
    )
    parser.add_argument("--grouped_samples_per_limit", type=int, default=2)
    parser.add_argument(
        "--grouped_sampling_temperatures",
        type=_positive_float_list,
        default=(0.5, 1.0),
    )
    parser.add_argument(
        "--grouped_fit_match_counts", type=_positive_int_list, default=(32, 64)
    )
    parser.add_argument(
        "--grouped_ransac_thresholds_px",
        type=_positive_float_list,
        default=(2.0, 4.0),
    )
    parser.add_argument("--grouped_ransac_iterations", type=int, default=2000)
    parser.add_argument("--grouped_rank_fold_count", type=int, default=1)
    parser.add_argument(
        "--grouped_crossfit_mode",
        choices=(
            "token_spatial",
            "token_spatial_track_purged",
            "token_spatial_track_maplet_purged",
            "token_track_component",
            "token_track_voxel_component",
            "token_track_maplet_component",
        ),
        default="token_spatial",
    )
    parser.add_argument(
        "--grouped_crossfit_role_assignment",
        choices=("fixed", "adaptive_balanced"),
        default="fixed",
        help=(
            "Assign fixed logical fold roles or balance fit/verify/audit over "
            "strictly disjoint component folds. Adaptive assignment uses only "
            "candidate-graph structure and never pose or ground truth."
        ),
    )
    parser.add_argument(
        "--grouped_crossfit_spatial_fold_policy",
        choices=("legacy_local_modulo", "cell_rotated_balanced"),
        default="legacy_local_modulo",
        help=(
            "Assign each grid cell from fold zero for artifact replay, or rotate "
            "cell offsets to balance non-divisible fold counts."
        ),
    )
    parser.add_argument(
        "--grouped_crossfit_maplet_voxel_size_m", type=float, default=0.5
    )
    parser.add_argument(
        "--enable_grouped_independent_shortlist_pool",
        action="store_true",
        help=(
            "split rank folds into track/maplet-disjoint shortlist and final "
            "hypothesis-selection evidence pools"
        ),
    )
    parser.add_argument(
        "--grouped_final_refine_acceptance_policy",
        choices=(
            "fixed_posterior_likelihood_gain",
            "legacy_rank_key",
            "strict_count_gain_with_grid_nondecrease",
        ),
        default="fixed_posterior_likelihood_gain",
    )
    parser.add_argument(
        "--optional_grouped_final_refine_acceptance_policy",
        choices=(
            "same_as_immutable_baseline",
            "fixed_posterior_likelihood_gain",
            "legacy_rank_key",
            "strict_count_gain_with_grid_nondecrease",
        ),
        default="same_as_immutable_baseline",
    )
    parser.add_argument(
        "--grouped_hypothesis_selection_policy",
        choices=GROUPED_HYPOTHESIS_SELECTION_POLICIES,
        default="fixed_posterior_likelihood_only",
        help=(
            "production held-out mean likelihood or an explicitly "
            "development-only robust/legacy selector"
        ),
    )
    parser.add_argument(
        "--grouped_final_refine_mode",
        choices=("auto", "hard_assignment", "latent_em"),
        default="auto",
        help="auto keeps top-L soft whenever grouped latent EM is enabled",
    )
    parser.add_argument("--grouped_min_fit_matches", type=int, default=8)
    parser.add_argument("--grouped_min_fit_grid_cells", type=int, default=4)
    parser.add_argument(
        "--grouped_min_xyz_second_singular_ratio", type=float, default=1e-3
    )
    parser.add_argument(
        "--grouped_generation_mode",
        choices=(
            "assignment_ransac",
            "grouped_prosac",
            "assignment_plus_grouped_prosac",
        ),
        default="assignment_ransac",
    )
    parser.add_argument(
        "--grouped_prosac_hypotheses_per_limit", type=int, default=64
    )
    parser.add_argument(
        "--grouped_prosac_minimal_set_size", type=int, default=6
    )
    parser.add_argument(
        "--grouped_prosac_minimal_set_sizes",
        type=_positive_int_list,
        default=(),
        help="optional 4-8 point sizes cycled within each PROSAC budget",
    )
    parser.add_argument(
        "--grouped_prosac_group_probability_power", type=float, default=1.0
    )
    parser.add_argument(
        "--grouped_prosac_candidate_probability_power", type=float, default=0.5
    )
    parser.add_argument(
        "--grouped_prosac_max_sample_attempts", type=int, default=64
    )
    parser.add_argument(
        "--grouped_prosac_min_bearing_span_deg", type=float, default=3.0
    )
    parser.add_argument(
        "--grouped_prosac_min_translation_eigenvalue", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_prosac_max_translation_condition", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_prosac_max_joint_condition", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_prosac_min_depth_span_ratio", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_prosac_min_xyz_third_ratio", type=float, default=0.0
    )
    parser.add_argument(
        "--grouped_prosac_observability_evidence_mode",
        choices=("minimal_sample", "resolved_fit_consensus"),
        default="minimal_sample",
        help=(
            "Evaluate active P24 gates on the sampled minimal set or on the "
            "pose-resolved fit consensus."
        ),
    )
    parser.add_argument(
        "--disable_grouped_prosac_local_optimization", action="store_true"
    )
    parser.add_argument(
        "--grouped_prosac_local_consensus_px", type=float, default=4.0
    )
    parser.add_argument(
        "--grouped_prosac_local_min_matches", type=int, default=8
    )
    parser.add_argument("--grouped_prosac_use_spatial_modes", action="store_true")
    parser.add_argument(
        "--grouped_prosac_verification_top_k",
        type=int,
        default=0,
        help="fully verify only the top-k generated poses; zero verifies all",
    )
    parser.add_argument(
        "--grouped_prosac_shortlist_evidence_mode",
        choices=("full_spatial", "base_coordinate_then_full_spatial"),
        default="full_spatial",
        help=(
            "Use RGB spatial evidence for every generated pose or reserve it "
            "for the verified shortlist and latent refinement."
        ),
    )
    parser.add_argument(
        "--grouped_prosac_spatial_rescore_top_k",
        type=int,
        default=0,
        help=(
            "In selective mode, evaluate full RGB spatial likelihood only for "
            "this many base-coordinate shortlist entries before final top-k."
        ),
    )
    parser.add_argument(
        "--grouped_prosac_shortlist_selection_mode",
        choices=("score_topk", "profile_pose_diverse"),
        default="score_topk",
        help=(
            "Keep the legacy score top-k or reserve shortlist capacity for "
            "generation-profile and pose-space diversity."
        ),
    )
    parser.add_argument(
        "--grouped_prosac_shortlist_diverse_count", type=int, default=0
    )
    parser.add_argument(
        "--grouped_prosac_shortlist_min_per_profile", type=int, default=0
    )
    parser.add_argument(
        "--grouped_prosac_shortlist_translation_diversity_m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--grouped_prosac_shortlist_rotation_diversity_deg",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--grouped_prosac_profiles_json",
        type=_grouped_prosac_profiles_json,
        default=(),
        help=(
            "JSON list of named generation profiles; all profiles run inside "
            "one deduplicated hypothesis ensemble"
        ),
    )
    parser.add_argument(
        "--max_hypothesis_information_audit",
        type=int,
        default=-1,
        help="cap expensive per-hypothesis diagnostics; -1 keeps all",
    )
    parser.add_argument("--enable_grouped_latent_em", action="store_true")
    parser.add_argument("--grouped_latent_em_seed_count", type=int, default=16)
    parser.add_argument(
        "--grouped_latent_em_seed_min_per_profile", type=int, default=0
    )
    parser.add_argument(
        "--grouped_latent_em_seed_evidence_mode",
        choices=("crossfit_shortlist", "fit_pool_DIAGNOSTIC_ONLY"),
        default="crossfit_shortlist",
        help=(
            "select EM seeds from the held-out shortlist evidence; the fit-pool "
            "self-consistency policy is retained only as an explicit diagnostic"
        ),
    )
    parser.add_argument(
        "--grouped_latent_em_translation_diversity_m", type=float, default=0.05
    )
    parser.add_argument(
        "--grouped_latent_em_rotation_diversity_deg", type=float, default=0.5
    )
    parser.add_argument("--grouped_latent_em_iterations", type=int, default=3)
    parser.add_argument(
        "--grouped_latent_em_identity_temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--grouped_latent_em_identity_temperature_floor", type=float, default=0.75
    )
    parser.add_argument(
        "--grouped_latent_em_null_mass_floor", type=float, default=0.02
    )
    parser.add_argument(
        "--grouped_latent_em_null_likelihood", type=float, default=1e-3
    )
    parser.add_argument(
        "--grouped_latent_em_outlier_likelihood", type=float, default=1e-3
    )
    parser.add_argument(
        "--grouped_latent_em_residual_sigma_px", type=float, default=3.0
    )
    parser.add_argument(
        "--grouped_latent_em_spatial_evidence_weight", type=float, default=-1.0
    )
    parser.add_argument(
        "--grouped_latent_em_coordinate_update_policy",
        choices=(
            "posterior_mean",
            "concentrated_map",
            "calibrated_mixture_map",
        ),
        default="posterior_mean",
    )
    parser.add_argument(
        "--grouped_latent_em_minimum_coordinate_mode_probability",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--grouped_latent_em_max_responsibility_change", type=float, default=0.25
    )
    parser.add_argument(
        "--grouped_latent_em_min_candidate_weight", type=float, default=2e-3
    )
    parser.add_argument(
        "--grouped_latent_em_min_effective_group_mass", type=float, default=0.05
    )
    parser.add_argument(
        "--grouped_latent_em_min_effective_groups", type=int, default=8
    )
    parser.add_argument(
        "--disable_grouped_latent_em_track_capacity", action="store_true"
    )
    parser.add_argument(
        "--grouped_latent_em_robust_f_scale_px", type=float, default=2.0
    )
    parser.add_argument(
        "--grouped_latent_em_max_nfev", type=int, default=50
    )
    parser.add_argument(
        "--grouped_latent_em_max_translation_step_m", type=float, default=1.0
    )
    parser.add_argument(
        "--grouped_latent_em_max_rotation_step_deg", type=float, default=10.0
    )
    parser.add_argument("--single_ransac_match_count", type=int, default=32)
    parser.add_argument("--single_ransac_selection_mode", default="score_topk")
    parser.add_argument("--single_ransac_threshold_px", type=float, default=8.0)
    parser.add_argument("--single_ransac_iterations", type=int, default=5000)
    parser.add_argument(
        "--evaluation_role",
        choices=("development", "untouched_test"),
        default="development",
    )
    parser.add_argument(
        "--development_cross_block_audit",
        action="store_true",
        help="replay validation-frozen policies on the reused late block",
    )
    return parser.parse_args(argv)


def _train_grouped_export_requires_geometry(args: argparse.Namespace) -> bool:
    """Return whether an enabled train export actually consumes geometry scores."""

    if not bool(args.export_train_grouped_hypotheses):
        return False
    return bool(
        float(args.candidate_geometry_prior_mix_weight) > 0.0
        or float(args.candidate_geometry_generation_mix_weight) > 0.0
        or float(args.candidate_spatial_geometry_calibration_weight) > 0.0
    )


def _immutable_baseline_required_splits(
    args: argparse.Namespace,
) -> tuple[str, ...]:
    """Return inference splits that can actually invoke baseline promotion.

    Train hypotheses are exported only as target-free calibration episodes;
    they never enter immutable-baseline selection and therefore must not make
    a validation/test-only baseline artifact fail its coverage contract.
    """

    return (
        ("validation", "test")
        if bool(args.development_cross_block_audit)
        else ("validation",)
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


SELECTED_POSE_ARTIFACT_FORMAT = "selected_pose_inference_only_v1"


def _canonical_manifest_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_selected_pose_artifact(
    path: Path,
    rows: Sequence[dict[str, object]],
    *,
    source_manifest: dict[str, object],
) -> None:
    """Write final inference poses with enough provenance for strict replay."""

    if not rows:
        raise ValueError("selected pose artifact requires at least one row")
    identities: set[tuple[str, str, str]] = set()
    poses = np.full((len(rows), 4, 4), np.nan, dtype=np.float64)
    success = np.zeros((len(rows),), dtype=bool)
    match_counts = np.zeros((len(rows),), dtype=np.int64)
    inlier_counts = np.zeros((len(rows),), dtype=np.int64)
    for index, row in enumerate(rows):
        identity = (
            str(row["split_name"]),
            str(row["evaluation_label"]),
            str(row["query_id"]),
        )
        if identity in identities:
            raise ValueError(f"duplicate selected pose identity: {identity}")
        identities.add(identity)
        row_success = bool(row["success"])
        match_count = int(row["match_count"])
        inlier_count = int(row["inlier_count"])
        if match_count < 0 or not 0 <= inlier_count <= match_count:
            raise ValueError("selected pose match/inlier counts are invalid")
        success[index] = row_success
        match_counts[index] = match_count
        inlier_counts[index] = inlier_count
        if row_success:
            pose = np.asarray(row["pose_w2c"], dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(pose)):
                raise ValueError("successful selected pose is non-finite")
            poses[index] = pose
        elif row.get("pose_w2c") is not None:
            raise ValueError("failed selected pose must not carry a pose matrix")
    source_hash = _canonical_manifest_sha256(source_manifest)
    metadata = {
        "format": SELECTED_POSE_ARTIFACT_FORMAT,
        "row_count": int(len(rows)),
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_reestimated_during_replay": False,
        "source_manifest_sha256": source_hash,
        "source_manifest": source_manifest,
    }
    np.savez_compressed(
        path,
        query_ids=np.asarray([str(row["query_id"]) for row in rows]),
        split_names=np.asarray([str(row["split_name"]) for row in rows]),
        evaluation_labels=np.asarray(
            [str(row["evaluation_label"]) for row in rows]
        ),
        success=success,
        poses_w2c=poses,
        match_counts=match_counts,
        inlier_counts=inlier_counts,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _load_immutable_baseline_pose_artifact(
    path: Path,
    *,
    expected_colmap_cameras_sha256: str,
    expected_colmap_images_sha256: str,
    evaluation_label: str = "",
) -> dict[str, object]:
    """Load frozen poses and reject ambiguous, stale, or cross-scene replay."""

    payload = _load_npz(path)
    required = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "success",
        "poses_w2c",
        "match_counts",
        "inlier_counts",
        "metadata_json",
    }
    if set(payload) != required:
        raise ValueError(
            "immutable baseline pose artifact fields differ: "
            f"missing={sorted(required - set(payload))}, "
            f"extra={sorted(set(payload) - required)}"
        )
    metadata = json.loads(str(payload["metadata_json"].item()))
    if metadata.get("format") != SELECTED_POSE_ARTIFACT_FORMAT:
        raise ValueError("unsupported immutable baseline pose artifact format")
    source_manifest = metadata.get("source_manifest")
    if not isinstance(source_manifest, dict) or metadata.get(
        "source_manifest_sha256"
    ) != _canonical_manifest_sha256(source_manifest):
        raise ValueError("immutable baseline pose source manifest is stale")
    source_inputs = source_manifest.get("inputs")
    if not isinstance(source_inputs, dict):
        raise ValueError("immutable baseline pose source inputs are missing")
    scene_mismatches = {
        "colmap_cameras_bin_sha256": (
            expected_colmap_cameras_sha256,
            source_inputs.get("colmap_cameras_bin_sha256"),
        ),
        "colmap_images_bin_sha256": (
            expected_colmap_images_sha256,
            source_inputs.get("colmap_images_bin_sha256"),
        ),
    }
    scene_mismatches = {
        key: {"expected": expected, "actual": actual}
        for key, (expected, actual) in scene_mismatches.items()
        if actual != expected
    }
    if scene_mismatches:
        raise ValueError(
            "immutable baseline pose belongs to a different COLMAP scene: "
            f"{json.dumps(scene_mismatches, sort_keys=True)}"
        )
    query_ids = payload["query_ids"].astype(str).reshape(-1)
    split_names = payload["split_names"].astype(str).reshape(-1)
    labels = payload["evaluation_labels"].astype(str).reshape(-1)
    success = np.asarray(payload["success"], dtype=bool).reshape(-1)
    poses = np.asarray(payload["poses_w2c"], dtype=np.float64)
    match_counts = np.asarray(payload["match_counts"], dtype=np.int64).reshape(-1)
    inlier_counts = np.asarray(payload["inlier_counts"], dtype=np.int64).reshape(-1)
    row_count = len(query_ids)
    if (
        any(len(values) != row_count for values in (split_names, labels, success, match_counts, inlier_counts))
        or poses.shape != (row_count, 4, 4)
        or int(metadata.get("row_count", -1)) != row_count
    ):
        raise ValueError("immutable baseline pose artifact dimensions differ")
    available_labels = tuple(sorted(set(labels.tolist())))
    selected_label = str(evaluation_label)
    if selected_label:
        if selected_label not in available_labels:
            raise ValueError(
                f"immutable baseline pose label is missing: {selected_label}"
            )
    elif len(available_labels) == 1:
        selected_label = available_labels[0]
    else:
        raise ValueError(
            "immutable baseline pose artifact contains multiple policies; "
            "select --immutable_baseline_pose_evaluation_label"
        )
    records: dict[tuple[str, str], PnPResult] = {}
    for index in np.flatnonzero(labels == selected_label):
        identity = (str(split_names[index]), str(query_ids[index]))
        if identity in records:
            raise ValueError(f"duplicate immutable baseline pose: {identity}")
        match_count = int(match_counts[index])
        inlier_count = int(inlier_counts[index])
        if match_count < 0 or not 0 <= inlier_count <= match_count:
            raise ValueError("immutable baseline match/inlier counts are invalid")
        row_success = bool(success[index])
        pose = np.asarray(poses[index], dtype=np.float64).reshape(4, 4)
        if row_success and not np.all(np.isfinite(pose)):
            raise ValueError("successful immutable baseline pose is non-finite")
        if not row_success and np.any(np.isfinite(pose)):
            raise ValueError("failed immutable baseline pose unexpectedly has finite values")
        inlier_mask = np.zeros((match_count,), dtype=bool)
        inlier_mask[:inlier_count] = True
        records[identity] = PnPResult(
            success=row_success,
            pose_w2c=pose.copy() if row_success else None,
            inlier_mask=inlier_mask,
            match_count=match_count,
            inlier_count=inlier_count,
        )
    if not records:
        raise ValueError("immutable baseline pose policy has no rows")
    return {
        "path": str(path),
        "sha256": file_sha256_short(path),
        "evaluation_label": selected_label,
        "metadata": metadata,
        "records": records,
    }


POSTERIOR_OVERLAY_CANDIDATE_KEY = "posterior_overlay_candidate_probability"
POSTERIOR_OVERLAY_NULL_KEY = (
    "posterior_overlay_null_probability_DIAGNOSTIC_ONLY"
)


def _load_candidate_posterior_ensemble(
    paths: Sequence[Path],
    *,
    expected_image_ids: np.ndarray,
    expected_split: dict[str, list[str]],
    query_count: int,
    candidate_count: int,
    candidate_path: Path,
    proposals_path: Path,
    bank_path: Path,
    split_path: Path,
    descriptor_space_id: str | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Load aligned target-free posteriors without replacing evidence arrays."""

    if not paths:
        raise ValueError("candidate posterior ensemble requires at least one artifact")
    image_ids = np.asarray(expected_image_ids).astype(str).reshape(-1)
    expected_hashes = {
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "landmark_bank_sha256": file_sha256_short(bank_path),
        "split_json_sha256": file_sha256_short(split_path),
    }
    full_posteriors = []
    manifests = []
    for path in paths:
        artifact_path = Path(path)
        payload = _load_npz(artifact_path)
        metadata = json.loads(str(payload["metadata_json"].item()))
        artifact_format = str(metadata.get("format", ""))
        graph_posterior = (
            artifact_format.endswith("candidate_graph_v3_posterior")
            or artifact_format.endswith("candidate_graph_v2_posterior")
            or "candidate_graph_posterior" in artifact_format
        )
        maplet_posterior = artifact_format == "candidate_maplet_prior_overlay_v1"
        if not graph_posterior and not maplet_posterior:
            raise ValueError("unsupported candidate posterior artifact")

        if maplet_posterior:
            required_fields = {
                "candidate_track_ids",
                "candidate_probabilities",
                "null_probabilities",
                "metadata_json",
            }
            if set(payload) != required_fields:
                raise ValueError("candidate-maplet posterior fields differ from contract")
            if metadata.get("contains_ground_truth") is not False or metadata.get(
                "contains_target_errors"
            ) is not False:
                raise ValueError("candidate-maplet posterior is not target-free")
            if str(metadata.get("probability_semantics")) != (
                "candidate_identity_probability_plus_explicit_null_equals_one"
            ):
                raise ValueError("candidate-maplet posterior semantics differ")
            if str(metadata.get("proposals_sha256")) != str(
                expected_hashes["proposals_sha256"]
            ):
                raise ValueError("candidate-maplet posterior references different proposals")
            lineage = metadata.get("inference_data_manifest")
            if not isinstance(lineage, dict):
                raise ValueError("candidate-maplet posterior inference lineage is missing")
            query_set = metadata.get("inference_query_set")
            if not isinstance(query_set, dict):
                raise ValueError("candidate-maplet posterior query-set lineage is missing")
            lineage_mismatches = {
                key: {"expected": value, "actual": lineage.get(key)}
                for key, value in {
                    "proposals_sha256": expected_hashes["proposals_sha256"],
                    "projected_landmark_bank_sha256": expected_hashes[
                        "landmark_bank_sha256"
                    ],
                }.items()
                if str(lineage.get(key)) != str(value)
            }

            candidate_payload = _load_npz(candidate_path)
            candidate_metadata = json.loads(
                str(candidate_payload["metadata_json"].item())
            )
            maplet_hash = candidate_metadata.get("maplet_support_index_sha256")
            if str(lineage.get("maplet_support_index_sha256")) != str(maplet_hash):
                lineage_mismatches["maplet_support_index_sha256"] = {
                    "expected": maplet_hash,
                    "actual": lineage.get("maplet_support_index_sha256"),
                }
            candidate_contract = {
                "proposals_sha256": expected_hashes["proposals_sha256"],
                "projected_landmark_bank_sha256": expected_hashes[
                    "landmark_bank_sha256"
                ],
                "maplet_support_index_sha256": maplet_hash,
            }
            for key, expected_value in candidate_contract.items():
                if str(candidate_metadata.get(key)) != str(expected_value):
                    lineage_mismatches[f"candidate_artifact::{key}"] = {
                        "expected": expected_value,
                        "actual": candidate_metadata.get(key),
                    }
            expected_source_rows = int(query_set.get("query_point_count", -1))
            expected_source_top_k = int(query_set.get("candidate_top_k", -1))
            if expected_source_rows <= 0 or expected_source_top_k <= 0:
                lineage_mismatches["inference_query_set"] = {
                    "expected": "positive query_point_count and candidate_top_k",
                    "actual": query_set,
                }
            if lineage_mismatches:
                raise ValueError(
                    "candidate-maplet posterior is stale or misaligned: "
                    f"{json.dumps(lineage_mismatches, sort_keys=True)}"
                )

            with np.load(proposals_path, allow_pickle=False) as proposal_payload:
                proposal_tracks = np.asarray(
                    proposal_payload["candidate_track_ids"], dtype=np.int64
                )
                proposal_query_ids = np.asarray(
                    proposal_payload["query_ids"]
                ).astype(str)
            overlay_tracks = np.asarray(
                payload["candidate_track_ids"], dtype=np.int64
            )
            source_probabilities = np.asarray(
                payload["candidate_probabilities"], dtype=np.float64
            )
            source_null = np.asarray(
                payload["null_probabilities"], dtype=np.float64
            ).reshape(-1)
            if not np.array_equal(overlay_tracks, proposal_tracks):
                raise ValueError("candidate-maplet posterior track identities differ")
            if proposal_tracks.shape != (
                expected_source_rows,
                expected_source_top_k,
            ):
                raise ValueError("candidate-maplet posterior query-set dimensions differ")
            if source_probabilities.shape != proposal_tracks.shape or source_null.shape != (
                proposal_tracks.shape[0],
            ):
                raise ValueError("candidate-maplet posterior arrays have incompatible shapes")
            source_valid = proposal_tracks >= 0
            if np.any(~np.isfinite(source_probabilities)) or np.any(
                ~np.isfinite(source_null)
            ):
                raise ValueError("candidate-maplet posterior contains non-finite values")
            if np.any((source_probabilities < 0.0) | (source_probabilities > 1.0)) or np.any(
                (source_null < 0.0) | (source_null > 1.0)
            ):
                raise ValueError("candidate-maplet posterior values must be probabilities")
            if np.any(np.abs(source_probabilities[~source_valid]) > 1e-6):
                raise ValueError("invalid candidate-maplet columns carry posterior mass")
            source_mass = np.sum(
                np.where(source_valid, source_probabilities, 0.0), axis=1
            ) + source_null
            if np.any(np.abs(source_mass - 1.0) > 1e-5):
                raise ValueError("candidate-maplet posterior probability mass differs from one")

            selected_rows = np.asarray(
                candidate_payload["selected_rows"], dtype=np.int64
            ).reshape(-1)
            selected_columns = np.asarray(
                candidate_payload["selected_columns"], dtype=np.int64
            )
            selected_valid = np.asarray(
                candidate_payload["valid_edges"], dtype=bool
            )
            expected_row_count = len(image_ids) * int(query_count)
            if (
                selected_rows.shape != (expected_row_count,)
                or selected_columns.shape != (expected_row_count, int(candidate_count))
                or selected_valid.shape != selected_columns.shape
            ):
                raise ValueError("candidate-maplet posterior target pool dimensions differ")
            if np.any((selected_rows < 0) | (selected_rows >= len(proposal_tracks))):
                raise ValueError("candidate-maplet posterior target rows are out of range")
            expected_valid = selected_columns >= 0
            if not np.array_equal(selected_valid, expected_valid):
                raise ValueError("candidate-maplet posterior target validity differs")
            if np.any(selected_columns[selected_valid] >= proposal_tracks.shape[1]):
                raise ValueError("candidate-maplet posterior target columns are out of range")
            selected_query_ids = proposal_query_ids[selected_rows].reshape(
                len(image_ids), int(query_count)
            )
            if not np.all(selected_query_ids == image_ids[:, None]):
                raise ValueError("candidate-maplet posterior image ordering differs")
            safe_columns = np.maximum(selected_columns, 0)
            compact = np.take_along_axis(
                source_probabilities[selected_rows], safe_columns, axis=1
            )
            compact = np.where(selected_valid, compact, 0.0)
            retained_mass = np.sum(compact, axis=1)
            compact_null = 1.0 - retained_mass
            if np.any(compact_null < -1e-6):
                raise ValueError("candidate-maplet compact posterior exceeds unit mass")
            compact_null = np.clip(compact_null, 0.0, 1.0)
            full = np.concatenate(
                [compact, compact_null[:, None]], axis=1
            ).reshape(
                len(image_ids), int(query_count), int(candidate_count) + 1
            )
            full_posteriors.append(full.astype(np.float32))
            manifests.append(
                {
                    "path": str(artifact_path),
                    "sha256": file_sha256_short(artifact_path),
                    "checkpoint_sha256": list(
                        metadata.get("checkpoint_sha256") or []
                    ),
                    "format": artifact_format,
                    "score_source_sha256": metadata.get(
                        "inference_scores_sha256"
                    ),
                    "omitted_candidate_mass_transferred_to_null": True,
                }
            )
            continue

        input_hashes = metadata.get("input_hashes")
        if not isinstance(input_hashes, dict):
            raise ValueError("candidate posterior input manifest is missing")
        mismatches = {
            key: {"expected": value, "actual": input_hashes.get(key)}
            for key, value in expected_hashes.items()
            if input_hashes.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "candidate posterior is stale or misaligned: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        if metadata.get("descriptor_space_id") != descriptor_space_id:
            raise ValueError("candidate posterior descriptor space differs")
        if int(metadata.get("query_count", -1)) != int(query_count) or int(
            metadata.get("candidate_count", -1)
        ) != int(candidate_count):
            raise ValueError("candidate posterior dimensions differ from candidate pool")
        if not np.array_equal(payload["image_ids"].astype(str), image_ids):
            raise ValueError("candidate posterior image ordering differs")
        checkpoint_path = artifact_path.parent / "best.pt"
        if not checkpoint_path.exists() or file_sha256_short(
            checkpoint_path
        ) != metadata.get("checkpoint_sha256"):
            raise ValueError("candidate posterior checkpoint is missing or stale")

        full = np.full(
            (len(image_ids), int(query_count), int(candidate_count) + 1),
            np.nan,
            dtype=np.float32,
        )
        assigned = np.zeros((len(image_ids),), dtype=bool)
        for split_name in ("train", "validation", "test"):
            indices = np.asarray(payload[f"{split_name}_indices"], dtype=np.int64)
            expected_indices = np.flatnonzero(
                np.isin(image_ids, np.asarray(expected_split[split_name]).astype(str))
            )
            if not np.array_equal(indices, expected_indices):
                raise ValueError("candidate posterior split indices differ")
            values = np.asarray(
                payload[f"{split_name}_posterior"], dtype=np.float32
            )
            if values.shape != (
                len(indices),
                int(query_count),
                int(candidate_count) + 1,
            ):
                raise ValueError("candidate posterior split tensor has an invalid shape")
            full[indices] = values
            assigned[indices] = True
        if not np.all(assigned) or not np.all(np.isfinite(full)):
            raise ValueError("candidate posterior does not cover every image")
        if np.any(full < 0.0) or not np.allclose(
            np.sum(full, axis=2), 1.0, atol=2e-5, rtol=2e-5
        ):
            raise ValueError("candidate posterior probability mass is invalid")
        full_posteriors.append(full)
        manifests.append(
            {
                "path": str(artifact_path),
                "sha256": file_sha256_short(artifact_path),
                "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
                "format": str(metadata["format"]),
                "score_source_sha256": input_hashes.get("score_artifact_sha256"),
            }
        )
    ensemble = np.mean(np.stack(full_posteriors, axis=0), axis=0)
    candidates = ensemble[:, :, :-1].reshape(-1, int(candidate_count))
    null = np.repeat(
        ensemble[:, :, -1].reshape(-1, 1), int(candidate_count), axis=1
    )
    return candidates, null, manifests


def _compact(values: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    valid = columns >= 0
    safe_columns = np.maximum(columns, 0)
    output = np.take_along_axis(np.asarray(values)[rows], safe_columns, axis=1).copy()
    if np.issubdtype(output.dtype, np.floating):
        output[~valid] = -np.inf
    else:
        output[~valid] = -1
    return output


def _selected_columns(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    safe = np.where(valid & np.isfinite(scores), scores, -np.inf)
    selected = np.argmax(safe, axis=1).astype(np.int64)
    selected[~np.any(np.isfinite(safe), axis=1)] = -1
    return selected


def _score_array(
    payload: dict[str, np.ndarray],
    proposals: dict[str, np.ndarray],
    key: str,
    *,
    selected_rows: np.ndarray,
    selected_columns: np.ndarray,
) -> np.ndarray:
    source = payload.get(str(key), proposals.get(str(key)))
    if source is None:
        raise ValueError(f"score array is missing: {key}")
    values = np.asarray(source)
    if values.shape == selected_columns.shape:
        return values.astype(np.float32)
    proposal_shape = np.asarray(proposals["candidate_track_ids"]).shape
    if values.shape == proposal_shape:
        return _compact(values, selected_rows, selected_columns).astype(np.float32)
    raise ValueError(
        f"score array {key} has shape {values.shape}; expected {selected_columns.shape} "
        f"or {proposal_shape}"
    )


def _query_seed(query_id: str) -> int:
    return int.from_bytes(hashlib.sha256(str(query_id).encode("utf8")).digest()[:4], "little")


def _query_execution_shard(
    query_ids: Sequence[str], *, shard_count: int, shard_index: int
) -> tuple[str, ...]:
    """Select an exactly reproducible, balanced execution-only query shard."""

    if int(shard_count) <= 0:
        raise ValueError("query_shard_count must be positive")
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("query_shard_index must be in [0, query_shard_count)")
    return tuple(
        str(query_id)
        for position, query_id in enumerate(query_ids)
        if position % int(shard_count) == int(shard_index)
    )


def _hypothesis_is_chosen_for_audit(result: object, hypothesis_index: int) -> bool:
    chosen = getattr(result, "chosen_hypothesis_index", None)
    return bool(chosen is not None and int(hypothesis_index) == int(chosen))


def _validate_spatial_likelihood_artifact(
    path: Path,
    *,
    split_name: str,
    candidate_evidence_path: Path,
    candidate_evidence_metadata: dict[str, object],
    score_path: Path,
    proposals_path: Path,
    candidate_path: Path,
    bank_path: Path,
    allow_legacy_identity_dustbin: bool = False,
) -> dict[str, np.ndarray]:
    payload = _load_npz(path)
    metadata = json.loads(str(payload["metadata_json"].item()))
    artifact_format = str(metadata.get("format", ""))
    if artifact_format not in {
        "candidate_spatial_likelihood_v3",
        "candidate_spatial_likelihood_v4",
        "candidate_spatial_likelihood_v5",
        "candidate_spatial_likelihood_v6",
        "candidate_spatial_likelihood_v7",
    }:
        raise ValueError("unsupported candidate spatial likelihood artifact")
    if (
        artifact_format == "candidate_spatial_likelihood_v3"
        and not bool(allow_legacy_identity_dustbin)
    ):
        raise ValueError(
            "v3 candidate spatial dustbin is identity-derived and diagnostic-only"
        )
    if artifact_format == "candidate_spatial_likelihood_v4" and metadata.get(
        "dustbin_probability_semantics"
    ) != (
        "one_minus_probability_predicted_spatial_mode_gt_pose_projection_residual_le_2px"
    ):
        raise ValueError("v4 candidate spatial likelihood has incompatible dustbin semantics")
    if artifact_format in {
        "candidate_spatial_likelihood_v5",
        "candidate_spatial_likelihood_v6",
        "candidate_spatial_likelihood_v7",
    }:
        if metadata.get("dustbin_probability_semantics") != (
            "normalized_k_plus_dustbin_gt_pose_projected_offset_outside_local_support"
        ) or not bool(metadata.get("joint_probability_mass_normalized")):
            raise ValueError(
                "v5 candidate spatial likelihood lacks its normalized K+1 contract"
            )
        local_log = np.asarray(payload["local_log_probabilities"], dtype=np.float64)
        dustbin = np.asarray(payload["dustbin_probabilities"], dtype=np.float64)
        if local_log.ndim != 2 or dustbin.shape != (len(local_log),):
            raise ValueError("v5 candidate spatial probability arrays have invalid shapes")
        if np.any(~np.isfinite(local_log)) or np.any(~np.isfinite(dustbin)):
            raise ValueError("v5 candidate spatial probability arrays are non-finite")
        if np.any((dustbin < 0.0) | (dustbin > 1.0)):
            raise ValueError("v5 candidate spatial dustbin is outside [0, 1]")
        conditional_mass = np.sum(np.exp(local_log), axis=1)
        joint_mass = (1.0 - dustbin) * conditional_mass + dustbin
        if (
            np.max(np.abs(conditional_mass - 1.0), initial=0.0) > 2e-3
            or np.max(np.abs(joint_mass - 1.0), initial=0.0) > 2e-3
        ):
            raise ValueError("v5 candidate spatial K+1 probability mass is invalid")
        if artifact_format in {
            "candidate_spatial_likelihood_v6",
            "candidate_spatial_likelihood_v7",
        }:
            if metadata.get("support_view_probability_semantics") != (
                POSE_VIEW_MIXTURE_SEMANTICS
            ):
                raise ValueError(
                    "v6 candidate spatial likelihood lacks its learned "
                    "pose-view mixture contract"
                )
            view_probability = np.asarray(
                payload["support_view_probabilities"], dtype=np.float64
            )
            if view_probability.shape != (len(local_log),) or np.any(
                ~np.isfinite(view_probability)
            ) or np.any((view_probability < 0.0) | (view_probability > 1.0)):
                raise ValueError("v6 pose-view probabilities are invalid")
    if artifact_format == "candidate_spatial_likelihood_v7":
        forbidden = {
            key
            for key in payload
            if key.startswith("target_")
            or "ground_truth" in key
            or key
            in {
                "measurement_validity_supervision_weight",
                "dustbin_supervision_weight",
            }
        }
        if forbidden:
            raise ValueError(
                "v7 target-free spatial likelihood exposes targets: "
                f"{sorted(forbidden)}"
            )
        if (
            bool(metadata.get("contains_ground_truth_arrays"))
            or bool(metadata.get("ground_truth_loaded_by_inference_process"))
            or not bool(metadata.get("prediction_frozen_before_target_join"))
            or not bool(metadata.get("input_schema_allowlisted"))
        ):
            raise ValueError("v7 spatial likelihood lacks its target-free contract")
    if str(metadata.get("query_source")) != "real_pair" or bool(
        metadata.get("pose_or_ground_truth_used_for_inference")
    ):
        raise ValueError("candidate spatial likelihood is not pose-free real-image evidence")
    rows_csv = Path(str(metadata.get("rows_csv", "")))
    if not rows_csv.exists() or file_sha256_short(rows_csv) != metadata.get(
        "rows_csv_sha256"
    ):
        raise ValueError("candidate spatial likelihood rows CSV is stale")
    rows_summary_path = rows_csv.with_suffix(".summary.json")
    if not rows_summary_path.exists():
        raise ValueError("candidate spatial likelihood rows summary is missing")
    rows_summary = json.loads(rows_summary_path.read_text())
    if str(rows_summary.get("split")) != str(split_name):
        raise ValueError("candidate spatial likelihood split differs")
    if artifact_format == "candidate_spatial_likelihood_v7" and rows_summary.get(
        "stage"
    ) != "candidate_specific_real_rgb_inference_rows":
        raise ValueError("v7 spatial likelihood does not use sanitized inference rows")
    row_inputs = rows_summary.get("inputs")
    if not isinstance(row_inputs, dict) or row_inputs.get(
        "selection_artifact_sha256"
    ) != file_sha256_short(candidate_evidence_path):
        raise ValueError("candidate spatial likelihood references different evidence")
    if metadata.get("candidate_evidence_sha256") != file_sha256_short(
        candidate_evidence_path
    ):
        raise ValueError("candidate spatial likelihood evidence hash differs")
    if artifact_format == "candidate_spatial_likelihood_v7":
        inference_evidence_path = Path(
            str(metadata.get("candidate_inference_evidence", ""))
        )
        if (
            not inference_evidence_path.exists()
            or metadata.get("candidate_inference_evidence_sha256")
            != file_sha256_short(inference_evidence_path)
            or not isinstance(row_inputs, dict)
            or row_inputs.get("inference_evidence_sha256")
            != file_sha256_short(inference_evidence_path)
        ):
            raise ValueError("v7 target-free inference evidence is stale or misaligned")
    expected_evidence = {
        "score_artifact_sha256": file_sha256_short(score_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
    }
    mismatches = {
        key: {"expected": value, "actual": candidate_evidence_metadata.get(key)}
        for key, value in expected_evidence.items()
        if candidate_evidence_metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "candidate evidence is stale or misaligned: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    return payload


def _validate_spatial_likelihood_artifact_set(
    paths: Sequence[Path],
    **kwargs: object,
) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("candidate spatial likelihood shard set is empty")
    payloads = [
        _validate_spatial_likelihood_artifact(path, **kwargs) for path in paths
    ]
    if len(payloads) == 1:
        return payloads[0]
    keys = set(payloads[0])
    if any(set(payload) != keys for payload in payloads[1:]):
        raise ValueError("candidate spatial likelihood shards expose different schemas")
    metadata = [json.loads(str(payload["metadata_json"].item())) for payload in payloads]
    shard_counts = {int(item.get("query_shard_count", -1)) for item in metadata}
    shard_indices = [int(item.get("query_shard_index", -1)) for item in metadata]
    if len(shard_counts) != 1 or shard_counts != {len(paths)} or sorted(
        shard_indices
    ) != list(range(len(paths))):
        raise ValueError("candidate spatial likelihood shard set is incomplete")
    ignored = {"query_shard_count", "query_shard_index"}
    canonical = [
        {key: value for key, value in item.items() if key not in ignored}
        for item in metadata
    ]
    if any(item != canonical[0] for item in canonical[1:]):
        raise ValueError("candidate spatial likelihood shard manifests differ")
    merged: dict[str, np.ndarray] = {}
    row_counts = [len(np.asarray(payload["query_ids"])) for payload in payloads]
    for key in sorted(keys - {"metadata_json"}):
        arrays = [np.asarray(payload[key]) for payload in payloads]
        row_aligned = all(
            array.ndim > 0 and int(array.shape[0]) == row_count
            for array, row_count in zip(arrays, row_counts)
        )
        if not row_aligned:
            if not all(np.array_equal(arrays[0], array) for array in arrays[1:]):
                raise ValueError(f"spatial shard shared array {key!r} differs")
            merged[key] = arrays[0].copy()
        else:
            merged[key] = np.concatenate(arrays, axis=0)
    query_ids = np.asarray(merged["query_ids"]).astype(str)
    source_rows = np.asarray(merged["source_query_rows"], dtype=np.int64)
    identity = np.asarray(merged["candidate_identity_keys"]).astype(str)
    support = np.asarray(merged["support_image_ids"]).astype(str)
    row_keys = list(zip(query_ids.tolist(), source_rows.tolist(), identity.tolist(), support.tolist()))
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("candidate spatial likelihood shards contain duplicate rows")
    merged_metadata = dict(canonical[0])
    merged_metadata["merged_query_shard_count"] = len(paths)
    merged_metadata["merged_query_shard_indices"] = sorted(shard_indices)
    merged_metadata["source_shard_sha256"] = [
        file_sha256_short(path) for path in paths
    ]
    merged["metadata_json"] = np.asarray(
        json.dumps(merged_metadata, sort_keys=True)
    )
    return merged


def _resolve_spatial_view_mixture_probabilities(
    frozen_probabilities: np.ndarray,
    valid_mask: np.ndarray,
    *,
    policy: str,
) -> np.ndarray:
    """Resolve support-view mass without changing candidate identity mass."""

    frozen = np.asarray(frozen_probabilities, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if frozen.shape != valid.shape or frozen.ndim != 3:
        raise ValueError("spatial support-view arrays must have shape (N,L,V)")
    if np.any(~np.isfinite(frozen[valid])) or np.any(frozen[valid] < 0.0):
        raise ValueError("spatial support-view probabilities are invalid")
    if str(policy) in {
        "frozen_candidate_posterior",
        "artifact_pose_view_posterior",
    }:
        output = np.where(valid, frozen, 0.0)
    elif str(policy) == "uniform_all_measured_views":
        counts = np.sum(valid, axis=2, keepdims=True)
        output = np.divide(
            valid.astype(np.float64),
            counts,
            out=np.zeros_like(frozen, dtype=np.float64),
            where=counts > 0,
        )
    else:
        raise ValueError(f"unsupported spatial support-view policy: {policy}")
    available_mass = np.sum(output, axis=2)
    if np.any(available_mass > 1.0 + 2e-5):
        raise ValueError("spatial support-view probability mass exceeds one")
    if str(policy) == "artifact_pose_view_posterior":
        measured = np.any(valid, axis=2)
        if np.any(np.abs(available_mass[measured] - 1.0) > 2e-5):
            raise ValueError(
                "learned artifact pose-view probability mass must equal one"
            )
    return output.astype(np.float32)


def _load_candidate_geometry_probability_rows(
    path: Path,
    *,
    spatial_path: Path,
    spatial_payload: dict[str, np.ndarray],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists() and (path.parent / "summary.json").exists():
        summary_path = path.parent / "summary.json"
    if not path.exists() or not summary_path.exists():
        raise ValueError("candidate geometry probability artifact is incomplete")
    summary = json.loads(summary_path.read_text())
    if summary.get("stage") != "candidate_geometry_verifier_apply":
        raise ValueError("unsupported candidate geometry probability artifact")
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict) or outputs.get(
        "probabilities_sha256"
    ) != file_sha256_short(path):
        raise ValueError("candidate geometry probability CSV is stale")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("candidate geometry probability inputs are missing")
    model_path = Path(str(inputs.get("model", "")))
    if not model_path.exists() or inputs.get("model_sha256") != file_sha256_short(
        model_path
    ):
        raise ValueError("candidate geometry verifier model is stale")
    model = json.loads(model_path.read_text())
    spatial_metadata = json.loads(str(spatial_payload["metadata_json"].item()))
    if model.get("measurement_checkpoint_sha256") != spatial_metadata.get(
        "measurement_checkpoint_sha256"
    ):
        raise ValueError("candidate geometry verifier checkpoint differs from spatial RGB")
    spatial_summary_path = spatial_path.parent / "summary.json"
    if not spatial_summary_path.exists():
        raise ValueError("candidate spatial diagnostic summary is missing")
    spatial_summary = json.loads(spatial_summary_path.read_text())
    spatial_outputs = spatial_summary.get("outputs")
    if not isinstance(spatial_outputs, dict) or inputs.get(
        "diagnostic_rows_sha256"
    ) != spatial_outputs.get("diagnostic_rows_sha256"):
        raise ValueError("candidate geometry probabilities use different RGB diagnostics")
    required = {
        "query_id",
        "source_query_row",
        "candidate_measurement_rank",
        "track_id",
        "prototype_id",
        "geometry_probability",
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("candidate geometry probability CSV schema is incomplete")
        if any("target" in str(name).lower() for name in reader.fieldnames):
            raise ValueError("candidate geometry probability CSV exposes target fields")
        rows = list(reader)
    if not rows:
        raise ValueError("candidate geometry probability CSV is empty")
    return rows, {
        "path": str(path),
        "sha256": file_sha256_short(path),
        "model": str(model_path),
        "model_sha256": file_sha256_short(model_path),
        "geometry_threshold_px": float(model["geometry_threshold_px"]),
    }


def _load_candidate_update_rows(
    path: Path,
    *,
    spatial_path: Path,
    spatial_payload: dict[str, np.ndarray],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists() and (path.parent / "summary.json").exists():
        summary_path = path.parent / "summary.json"
    if not path.exists() or not summary_path.exists():
        raise ValueError("candidate coordinate update artifact is incomplete")
    summary = json.loads(summary_path.read_text())
    stage = str(summary.get("stage", ""))
    if stage not in {
        "candidate_coordinate_update_verifier_fit",
        "candidate_coordinate_update_verifier_apply",
        UTILITY_APPLY_STAGE,
    }:
        raise ValueError("unsupported candidate coordinate update artifact")
    is_utility = stage == UTILITY_APPLY_STAGE
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("candidate coordinate update outputs are missing")
    recorded_hash = outputs.get(
        "validation_predictions_sha256"
        if stage == "candidate_coordinate_update_verifier_fit"
        else "predictions_sha256"
    )
    if recorded_hash != file_sha256_short(path):
        raise ValueError("candidate coordinate update CSV is stale")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("candidate coordinate update inputs are missing")
    model_path = Path(
        str(
            outputs.get("model", "")
            if stage == "candidate_coordinate_update_verifier_fit"
            else inputs.get("model", "")
        )
    )
    model_sha = (
        outputs.get("model_sha256")
        if stage == "candidate_coordinate_update_verifier_fit"
        else inputs.get("model_sha256")
    )
    if not model_path.exists() or model_sha != file_sha256_short(model_path):
        raise ValueError("candidate coordinate update verifier model is stale")
    model = json.loads(model_path.read_text())
    expected_model_format = (
        UTILITY_MODEL_FORMAT
        if is_utility
        else "candidate_coordinate_update_verifier_v1"
    )
    if model.get("format") != expected_model_format:
        raise ValueError("unsupported candidate coordinate update model")
    spatial_metadata = json.loads(str(spatial_payload["metadata_json"].item()))
    if model.get("measurement_checkpoint_sha256") != spatial_metadata.get(
        "measurement_checkpoint_sha256"
    ):
        raise ValueError("candidate coordinate update checkpoint differs from spatial RGB")
    if is_utility:
        offset_hash = hashlib.sha256(
            np.ascontiguousarray(
                np.asarray(spatial_payload["offsets_xy"], dtype="<f4")
            ).tobytes()
        ).hexdigest()[:16]
        expected_contract = {
            "measurement_checkpoint_sha256": spatial_metadata.get(
                "measurement_checkpoint_sha256"
            ),
            "candidate_evidence_sha256": spatial_metadata.get(
                "candidate_evidence_sha256"
            ),
            "candidate_inference_evidence_sha256": spatial_metadata.get(
                "candidate_inference_evidence_sha256"
            ),
            "coordinate_space_id": spatial_metadata.get("coordinate_space_id"),
            "offsets_sha256": offset_hash,
            "coordinate_proposal_policy": inputs.get(
                "coordinate_proposal_policy"
            ),
        }
        mismatches = {
            key: {"expected": value, "actual": model.get(key)}
            for key, value in expected_contract.items()
            if str(model.get(key, "")) != str(value)
        }
        if mismatches:
            raise ValueError(
                "candidate measurement utility differs from spatial RGB: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if inputs.get("spatial_sha256") != [file_sha256_short(spatial_path)]:
            raise ValueError(
                "candidate measurement utility was applied to different spatial RGB"
            )
        if str(model.get("coordinate_proposal_policy", "")) != str(
            inputs.get("coordinate_proposal_policy", "")
        ):
            raise ValueError(
                "candidate measurement utility coordinate policy is inconsistent"
            )
        protocol = summary.get("protocol", {})
        if (
            bool(protocol.get("ground_truth_loaded"))
            or bool(protocol.get("pose_loaded"))
            or not bool(protocol.get("probability_is_action_gate_not_identity_likelihood"))
        ):
            raise ValueError("candidate measurement utility apply is not target-free")
    else:
        spatial_summary_path = spatial_path.parent / "summary.json"
        if not spatial_summary_path.exists():
            raise ValueError("candidate spatial diagnostic summary is missing")
        spatial_summary = json.loads(spatial_summary_path.read_text())
        spatial_outputs = spatial_summary.get("outputs")
        diagnostic_hash = inputs.get(
            "validation_diagnostic_rows_sha256"
            if stage == "candidate_coordinate_update_verifier_fit"
            else "diagnostic_rows_sha256"
        )
        if not isinstance(spatial_outputs, dict) or diagnostic_hash != spatial_outputs.get(
            "diagnostic_rows_sha256"
        ):
            raise ValueError("candidate coordinate updates use different RGB diagnostics")
    required = {
        "candidate_identity_key",
        "query_id",
        "source_query_row",
        "candidate_measurement_rank",
        "track_id",
        "prototype_id",
        "update_beneficial_probability",
        "center_x",
        "center_y",
        "refined_x",
        "refined_y",
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("candidate coordinate update CSV schema is incomplete")
        if any("target" in str(name).lower() for name in reader.fieldnames):
            raise ValueError("candidate coordinate update CSV exposes target fields")
        rows = list(reader)
    if not rows:
        raise ValueError("candidate coordinate update CSV is empty")
    return rows, {
        "path": str(path),
        "sha256": file_sha256_short(path),
        "model": str(model_path),
        "model_sha256": file_sha256_short(model_path),
        "update_threshold": float(model["update_threshold"]),
        "candidate_geometry_verifier_sha256": str(
            model.get("candidate_geometry_verifier_sha256", "")
        ),
        "requires_candidate_geometry_probabilities": not is_utility,
        "artifact_kind": (
            "measurement_utility_action_gate"
            if is_utility
            else "legacy_coordinate_update_verifier"
        ),
        "coordinate_proposal_policy": str(
            model.get("coordinate_proposal_policy", "")
        ),
    }


def _load_train_oof_candidate_geometry_probability_rows(
    path: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    summary_path = path.parent / "summary.json"
    if not path.exists() or not summary_path.exists():
        raise ValueError("train OOF candidate geometry artifact is incomplete")
    summary = json.loads(summary_path.read_text())
    if summary.get("stage") != "candidate_geometry_verifier_fit":
        raise ValueError("unsupported train OOF candidate geometry artifact")
    outputs = summary.get("outputs")
    if not isinstance(outputs, dict) or outputs.get(
        "train_oof_probabilities_sha256"
    ) != file_sha256_short(path):
        raise ValueError("train OOF candidate geometry CSV is stale")
    model_path = Path(str(outputs.get("model", "")))
    if not model_path.exists() or outputs.get("model_sha256") != file_sha256_short(
        model_path
    ):
        raise ValueError("train OOF candidate geometry verifier model is stale")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("train OOF candidate geometry inputs are missing")
    diagnostic_path = Path(str(inputs.get("train_diagnostic_rows_csv", "")))
    if not diagnostic_path.exists() or inputs.get(
        "train_diagnostic_rows_sha256"
    ) != file_sha256_short(diagnostic_path):
        raise ValueError("train OOF candidate geometry diagnostics are stale")
    model = json.loads(model_path.read_text())
    required = {
        "query_id",
        "source_query_row",
        "candidate_measurement_rank",
        "track_id",
        "prototype_id",
        "geometry_probability",
    }
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("train OOF candidate geometry CSV schema is incomplete")
        if any("target" in str(name).lower() for name in reader.fieldnames):
            raise ValueError("train OOF candidate geometry CSV exposes target fields")
        rows = list(reader)
    if not rows:
        raise ValueError("train OOF candidate geometry CSV is empty")
    return rows, {
        "path": str(path),
        "sha256": file_sha256_short(path),
        "model": str(model_path),
        "model_sha256": file_sha256_short(model_path),
        "geometry_threshold_px": float(model["geometry_threshold_px"]),
        "probability_source": "query_grouped_out_of_fold",
        "diagnostic_rows": str(diagnostic_path),
        "diagnostic_rows_sha256": file_sha256_short(diagnostic_path),
    }


def _set_cv2_seed(seed: int) -> None:
    try:
        import cv2

        cv2.setRNGSeed(int(int(seed) % (2**31 - 1)))
    except ImportError:  # pragma: no cover
        return


def _bootstrap_percentile_ci(
    values: np.ndarray,
    *,
    percentile: float,
    seed: int,
    resamples: int = 2000,
) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(resamples), array.size))
    statistics = np.percentile(array[indices], float(percentile), axis=1)
    low, high = np.percentile(statistics, (2.5, 97.5))
    return [float(low), float(high)]


def _pose_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    success = [row for row in rows if bool(row.get("success"))]
    translations = np.asarray(
        [float(row["translation_m"]) for row in success], dtype=np.float64
    )
    rotations = np.asarray(
        [float(row["rotation_deg"]) for row in success], dtype=np.float64
    )
    output: dict[str, object] = {
        "query_count": int(len(rows)),
        "success_count": int(len(success)),
        "success_rate": 0.0 if not rows else float(len(success) / len(rows)),
        "median_translation_m_success": (
            None if translations.size == 0 else float(np.median(translations))
        ),
        "p90_translation_m_success": (
            None if translations.size == 0 else float(np.percentile(translations, 90))
        ),
        "median_rotation_deg_success": (
            None if rotations.size == 0 else float(np.median(rotations))
        ),
        "p90_rotation_deg_success": (
            None if rotations.size == 0 else float(np.percentile(rotations, 90))
        ),
        "median_translation_m_success_bootstrap95_ci": _bootstrap_percentile_ci(
            translations, percentile=50.0, seed=1729
        ),
        "p90_translation_m_success_bootstrap95_ci": _bootstrap_percentile_ci(
            translations, percentile=90.0, seed=1730
        ),
        "median_rotation_deg_success_bootstrap95_ci": _bootstrap_percentile_ci(
            rotations, percentile=50.0, seed=1731
        ),
        "p90_rotation_deg_success_bootstrap95_ci": _bootstrap_percentile_ci(
            rotations, percentile=90.0, seed=1732
        ),
        "median_matches": (
            None
            if not rows
            else float(np.median([int(row.get("match_count", 0)) for row in rows]))
        ),
        "median_inliers_success": (
            None
            if not success
            else float(np.median([int(row.get("inlier_count", 0)) for row in success]))
        ),
    }
    for distance, angle, name in (
        (0.25, 2.0, "25cm_2deg"),
        (0.10, 5.0, "10cm_5deg"),
        (0.05, 5.0, "5cm_5deg"),
        (0.03, 5.0, "3cm_5deg"),
    ):
        output[f"recall_{name}"] = (
            0.0
            if not rows
            else float(
                np.mean(
                    [
                        bool(row.get("success"))
                        and float(row["translation_m"]) <= distance
                        and float(row["rotation_deg"]) <= angle
                        for row in rows
                    ]
                )
            )
        )
    pre_refine = [
        row
        for row in rows
        if row.get("pre_refine_translation_m") is not None
        and row.get("pre_refine_rotation_deg") is not None
    ]
    oracle = [
        row
        for row in rows
        if row.get("hypothesis_oracle_translation_m") is not None
        and row.get("hypothesis_oracle_rotation_deg") is not None
    ]
    if pre_refine:
        output["pre_refine_median_translation_m"] = float(
            np.median([float(row["pre_refine_translation_m"]) for row in pre_refine])
        )
        output["pre_refine_p90_translation_m"] = float(
            np.percentile(
                [float(row["pre_refine_translation_m"]) for row in pre_refine], 90
            )
        )
        output["pre_refine_median_rotation_deg"] = float(
            np.median([float(row["pre_refine_rotation_deg"]) for row in pre_refine])
        )
        output["median_final_minus_pre_refine_translation_m"] = float(
            np.median(
                [
                    float(row["translation_m"])
                    - float(row["pre_refine_translation_m"])
                    for row in pre_refine
                    if row.get("translation_m") is not None
                ]
            )
        )
    if oracle:
        output["hypothesis_oracle_median_translation_m"] = float(
            np.median(
                [float(row["hypothesis_oracle_translation_m"]) for row in oracle]
            )
        )
        output["hypothesis_oracle_p90_translation_m"] = float(
            np.percentile(
                [float(row["hypothesis_oracle_translation_m"]) for row in oracle], 90
            )
        )
        output["hypothesis_oracle_median_rotation_deg"] = float(
            np.median([float(row["hypothesis_oracle_rotation_deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_10cm_5deg"] = float(
            np.mean([bool(row["hypothesis_oracle_10cm_5deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_5cm_5deg"] = float(
            np.mean([bool(row["hypothesis_oracle_5cm_5deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_3cm_5deg"] = float(
            np.mean([bool(row["hypothesis_oracle_3cm_5deg"]) for row in oracle])
        )
        output["hypothesis_oracle_recall_25cm_2deg"] = float(
            np.mean([bool(row["hypothesis_oracle_25cm_2deg"]) for row in oracle])
        )
        output["median_chosen_hypothesis_translation_rank"] = float(
            np.median(
                [float(row["chosen_hypothesis_translation_rank"]) for row in oracle]
            )
        )
        for key in (
            "valid_hypothesis_count",
            "hypothesis_3cm_5deg_count",
            "hypothesis_10cm_5deg_count",
            "hypothesis_25cm_2deg_count",
            "catastrophic_hypothesis_count",
        ):
            output[f"median_{key}"] = float(
                np.median([int(row[key]) for row in oracle])
            )
        selection_regrets = np.asarray(
            [
                float(row["translation_m"])
                - float(row["hypothesis_oracle_translation_m"])
                for row in oracle
                if row.get("translation_m") is not None
            ],
            dtype=np.float64,
        )
        if len(selection_regrets):
            output["median_hypothesis_selection_regret_m"] = float(
                np.median(selection_regrets)
            )
            output["p90_hypothesis_selection_regret_m"] = float(
                np.percentile(selection_regrets, 90)
            )
            output["max_hypothesis_selection_regret_m"] = float(
                np.max(selection_regrets)
            )
    verified_oracle_rows = [
        row
        for row in rows
        if row.get("verified_hypothesis_oracle_translation_m_TARGET_ONLY")
        is not None
    ]
    if verified_oracle_rows:
        verified_translation = np.asarray(
            [
                float(row["verified_hypothesis_oracle_translation_m_TARGET_ONLY"])
                for row in verified_oracle_rows
            ],
            dtype=np.float64,
        )
        shortlist_gaps = np.asarray(
            [
                float(row["verified_hypothesis_oracle_translation_m_TARGET_ONLY"])
                - float(row["hypothesis_oracle_translation_m"])
                for row in verified_oracle_rows
                if row.get("hypothesis_oracle_translation_m") is not None
            ],
            dtype=np.float64,
        )
        verified_selection_regrets = np.asarray(
            [
                float(row["translation_m"])
                - float(row["verified_hypothesis_oracle_translation_m_TARGET_ONLY"])
                for row in verified_oracle_rows
                if row.get("translation_m") is not None
            ],
            dtype=np.float64,
        )
        output.update(
            {
                "verified_hypothesis_query_count": int(len(verified_oracle_rows)),
                "median_verified_hypothesis_count": float(
                    np.median(
                        [
                            int(row["verified_hypothesis_count"])
                            for row in verified_oracle_rows
                        ]
                    )
                ),
                "verified_hypothesis_oracle_median_translation_m_TARGET_ONLY": float(
                    np.median(verified_translation)
                ),
                "verified_hypothesis_oracle_p90_translation_m_TARGET_ONLY": float(
                    np.percentile(verified_translation, 90)
                ),
                "verified_hypothesis_oracle_recall_3cm_5deg_TARGET_ONLY": float(
                    np.mean(
                        [
                            bool(row["verified_hypothesis_oracle_3cm_5deg_TARGET_ONLY"])
                            for row in verified_oracle_rows
                        ]
                    )
                ),
                "verified_hypothesis_oracle_recall_5cm_5deg_TARGET_ONLY": float(
                    np.mean(
                        [
                            bool(row["verified_hypothesis_oracle_5cm_5deg_TARGET_ONLY"])
                            for row in verified_oracle_rows
                        ]
                    )
                ),
                "verified_hypothesis_oracle_recall_10cm_5deg_TARGET_ONLY": float(
                    np.mean(
                        [
                            bool(row["verified_hypothesis_oracle_10cm_5deg_TARGET_ONLY"])
                            for row in verified_oracle_rows
                        ]
                    )
                ),
                "median_shortlist_oracle_gap_m_TARGET_ONLY": float(
                    np.median(shortlist_gaps)
                ),
                "p90_shortlist_oracle_gap_m_TARGET_ONLY": float(
                    np.percentile(shortlist_gaps, 90)
                ),
                "max_shortlist_oracle_gap_m_TARGET_ONLY": float(
                    np.max(shortlist_gaps)
                ),
                "shortlist_oracle_degradation_rate_TARGET_ONLY": float(
                    np.mean(shortlist_gaps > 1e-12)
                ),
                "median_verified_selection_regret_m_TARGET_ONLY": float(
                    np.median(verified_selection_regrets)
                ),
                "p90_verified_selection_regret_m_TARGET_ONLY": float(
                    np.percentile(verified_selection_regrets, 90)
                ),
                "max_verified_selection_regret_m_TARGET_ONLY": float(
                    np.max(verified_selection_regrets)
                ),
            }
        )
    spatial_rescored_rows = [
        row
        for row in rows
        if row.get(
            "spatial_rescored_hypothesis_oracle_translation_m_TARGET_ONLY"
        )
        is not None
    ]
    if spatial_rescored_rows:
        spatial_rescored_translation = np.asarray(
            [
                float(
                    row[
                        "spatial_rescored_hypothesis_oracle_translation_m_TARGET_ONLY"
                    ]
                )
                for row in spatial_rescored_rows
            ],
            dtype=np.float64,
        )
        output.update(
            {
                "spatial_rescored_hypothesis_query_count": int(
                    len(spatial_rescored_rows)
                ),
                "median_spatial_rescored_hypothesis_count": float(
                    np.median(
                        [
                            int(row["spatial_rescored_hypothesis_count"])
                            for row in spatial_rescored_rows
                        ]
                    )
                ),
                "spatial_rescored_hypothesis_oracle_median_translation_m_TARGET_ONLY": float(
                    np.median(spatial_rescored_translation)
                ),
                "spatial_rescored_hypothesis_oracle_p90_translation_m_TARGET_ONLY": float(
                    np.percentile(spatial_rescored_translation, 90)
                ),
            }
        )
    raw_oracle_rows = [
        row
        for row in rows
        if row.get("raw_hypothesis_oracle_translation_m_TARGET_ONLY") is not None
        and row.get("raw_hypothesis_oracle_rotation_deg_TARGET_ONLY") is not None
    ]
    if raw_oracle_rows:
        raw_oracle_translation = np.asarray(
            [
                float(row["raw_hypothesis_oracle_translation_m_TARGET_ONLY"])
                for row in raw_oracle_rows
            ],
            dtype=np.float64,
        )
        raw_oracle_rotation = np.asarray(
            [
                float(row["raw_hypothesis_oracle_rotation_deg_TARGET_ONLY"])
                for row in raw_oracle_rows
            ],
            dtype=np.float64,
        )
        output.update(
            {
                "raw_hypothesis_oracle_query_count": int(len(raw_oracle_rows)),
                "raw_hypothesis_oracle_median_translation_m_TARGET_ONLY": float(
                    np.median(raw_oracle_translation)
                ),
                "raw_hypothesis_oracle_p90_translation_m_TARGET_ONLY": float(
                    np.percentile(raw_oracle_translation, 90)
                ),
                "raw_hypothesis_oracle_median_rotation_deg_TARGET_ONLY": float(
                    np.median(raw_oracle_rotation)
                ),
                "raw_hypothesis_oracle_p90_rotation_deg_TARGET_ONLY": float(
                    np.percentile(raw_oracle_rotation, 90)
                ),
                "raw_hypothesis_oracle_recall_3cm_5deg_TARGET_ONLY": float(
                    np.mean(
                        (raw_oracle_translation <= 0.03)
                        & (raw_oracle_rotation <= 5.0)
                    )
                ),
                "raw_hypothesis_oracle_recall_5cm_5deg_TARGET_ONLY": float(
                    np.mean(
                        (raw_oracle_translation <= 0.05)
                        & (raw_oracle_rotation <= 5.0)
                    )
                ),
                "raw_hypothesis_oracle_recall_10cm_5deg_TARGET_ONLY": float(
                    np.mean(
                        (raw_oracle_translation <= 0.10)
                        & (raw_oracle_rotation <= 5.0)
                    )
                ),
                "raw_hypothesis_oracle_recall_25cm_2deg_TARGET_ONLY": float(
                    np.mean(
                        (raw_oracle_translation <= 0.25)
                        & (raw_oracle_rotation <= 2.0)
                    )
                ),
            }
        )
    latent_rows = [
        row
        for row in rows
        if row.get("latent_em_hypothesis_oracle_translation_m_TARGET_ONLY")
        is not None
        and row.get("raw_hypothesis_oracle_translation_m_TARGET_ONLY") is not None
    ]
    if latent_rows:
        paired_raw_oracle_translation = np.asarray(
            [
                float(row["raw_hypothesis_oracle_translation_m_TARGET_ONLY"])
                for row in latent_rows
            ],
            dtype=np.float64,
        )
        latent_oracle_translation = np.asarray(
            [
                float(
                    row[
                        "latent_em_hypothesis_oracle_translation_m_TARGET_ONLY"
                    ]
                )
                for row in latent_rows
            ],
            dtype=np.float64,
        )
        output.update(
            {
                "latent_em_query_count": int(len(latent_rows)),
                "latent_em_median_hypothesis_count": float(
                    np.median(
                        [
                            int(row["latent_em_hypothesis_count"])
                            for row in latent_rows
                        ]
                    )
                ),
                "latent_em_only_oracle_median_translation_m_TARGET_ONLY": float(
                    np.median(latent_oracle_translation)
                ),
                "latent_em_only_oracle_p90_translation_m_TARGET_ONLY": float(
                    np.percentile(latent_oracle_translation, 90)
                ),
                "latent_em_query_win_rate_over_raw_oracle_TARGET_ONLY": float(
                    np.mean(
                        latent_oracle_translation < paired_raw_oracle_translation
                    )
                ),
            }
        )
    latent_seed_rows = [
        row
        for row in rows
        if row.get(
            "latent_em_seed_parent_oracle_translation_m_TARGET_ONLY"
        )
        is not None
        and row.get("latent_em_seed_parent_raw_rank_TARGET_ONLY") is not None
        and row.get("raw_hypothesis_oracle_translation_m_TARGET_ONLY") is not None
    ]
    if latent_seed_rows:
        seed_oracle_translation = np.asarray(
            [
                float(
                    row[
                        "latent_em_seed_parent_oracle_translation_m_TARGET_ONLY"
                    ]
                )
                for row in latent_seed_rows
            ],
            dtype=np.float64,
        )
        raw_oracle_translation = np.asarray(
            [
                float(row["raw_hypothesis_oracle_translation_m_TARGET_ONLY"])
                for row in latent_seed_rows
            ],
            dtype=np.float64,
        )
        seed_raw_ranks = np.asarray(
            [
                int(row["latent_em_seed_parent_raw_rank_TARGET_ONLY"])
                for row in latent_seed_rows
            ],
            dtype=np.int64,
        )
        refined_oracle_rows = [
            row
            for row in latent_seed_rows
            if row.get(
                "latent_em_refined_minus_seed_oracle_translation_m_TARGET_ONLY"
            )
            is not None
        ]
        output.update(
            {
                "latent_em_seed_audit_query_count_TARGET_ONLY": int(
                    len(latent_seed_rows)
                ),
                "latent_em_seed_parent_count_median_TARGET_ONLY": float(
                    np.median(
                        [
                            int(row["latent_em_seed_parent_count_TARGET_ONLY"])
                            for row in latent_seed_rows
                        ]
                    )
                ),
                "latent_em_seed_parent_oracle_median_translation_m_TARGET_ONLY": float(
                    np.median(seed_oracle_translation)
                ),
                "latent_em_seed_parent_oracle_p90_translation_m_TARGET_ONLY": float(
                    np.percentile(seed_oracle_translation, 90)
                ),
                "latent_em_seed_parent_raw_rank_median_TARGET_ONLY": float(
                    np.median(seed_raw_ranks)
                ),
                "latent_em_seed_parent_raw_rank_p90_TARGET_ONLY": float(
                    np.percentile(seed_raw_ranks, 90)
                ),
                "latent_em_seed_selection_oracle_gap_median_m_TARGET_ONLY": float(
                    np.median(seed_oracle_translation - raw_oracle_translation)
                ),
                "latent_em_seed_parent_recall_5cm_5deg_TARGET_ONLY": float(
                    np.mean(seed_oracle_translation <= 0.05)
                ),
                "latent_em_seed_parent_recall_10cm_5deg_TARGET_ONLY": float(
                    np.mean(seed_oracle_translation <= 0.10)
                ),
            }
        )
        if refined_oracle_rows:
            refinement_delta = np.asarray(
                [
                    float(
                        row[
                            "latent_em_refined_minus_seed_oracle_translation_m_TARGET_ONLY"
                        ]
                    )
                    for row in refined_oracle_rows
                ],
                dtype=np.float64,
            )
            output.update(
                {
                    "latent_em_refined_minus_seed_oracle_median_translation_m_TARGET_ONLY": float(
                        np.median(refinement_delta)
                    ),
                    "latent_em_refined_beats_seed_oracle_rate_TARGET_ONLY": float(
                        np.mean(refinement_delta < 0.0)
                    ),
                }
            )
    latent_parent_rows = [
        row
        for row in rows
        if int(row.get("latent_em_parent_pair_count_TARGET_ONLY", 0)) > 0
    ]
    if latent_parent_rows:
        pair_count = int(
            np.sum(
                [
                    int(row["latent_em_parent_pair_count_TARGET_ONLY"])
                    for row in latent_parent_rows
                ]
            )
        )
        win_count = int(
            np.sum(
                [
                    int(row["latent_em_parent_win_count_TARGET_ONLY"])
                    for row in latent_parent_rows
                ]
            )
        )
        loss_count = int(
            np.sum(
                [
                    int(row["latent_em_parent_loss_count_TARGET_ONLY"])
                    for row in latent_parent_rows
                ]
            )
        )
        output.update(
            {
                "latent_em_parent_pair_count_TARGET_ONLY": pair_count,
                "latent_em_parent_win_count_TARGET_ONLY": win_count,
                "latent_em_parent_loss_count_TARGET_ONLY": loss_count,
                "latent_em_parent_win_rate_TARGET_ONLY": float(
                    win_count / max(pair_count, 1)
                ),
                "latent_em_parent_loss_rate_TARGET_ONLY": float(
                    loss_count / max(pair_count, 1)
                ),
                "latent_em_parent_median_query_median_translation_delta_m_TARGET_ONLY": float(
                    np.median(
                        [
                            float(
                                row[
                                    "latent_em_parent_median_translation_delta_m_TARGET_ONLY"
                                ]
                            )
                            for row in latent_parent_rows
                        ]
                    )
                ),
                "latent_em_parent_worst_translation_regression_m_TARGET_ONLY": float(
                    np.max(
                        [
                            float(
                                row[
                                    "latent_em_parent_max_translation_delta_m_TARGET_ONLY"
                                ]
                            )
                            for row in latent_parent_rows
                        ]
                    )
                ),
            }
        )
    grouped_samples = [
        row for row in rows if int(row.get("grouped_minimal_sample_count", 0)) > 0
    ]
    if grouped_samples:
        sample_count = int(
            np.sum([int(row["grouped_minimal_sample_count"]) for row in grouped_samples])
        )
        pair_count = int(
            np.sum(
                [
                    int(row["grouped_minimal_sample_pair_count_TARGET_ONLY"])
                    for row in grouped_samples
                ]
            )
        )
        all_correct_2px = int(
            np.sum(
                [
                    int(
                        row[
                            "grouped_minimal_sample_all_correct_2px_count_TARGET_ONLY"
                        ]
                    )
                    for row in grouped_samples
                ]
            )
        )
        all_correct_5px = int(
            np.sum(
                [
                    int(
                        row[
                            "grouped_minimal_sample_all_correct_5px_count_TARGET_ONLY"
                        ]
                    )
                    for row in grouped_samples
                ]
            )
        )
        correct_2px_pairs = int(
            np.sum(
                [
                    int(
                        row[
                            "grouped_minimal_sample_correct_2px_pair_count_TARGET_ONLY"
                        ]
                    )
                    for row in grouped_samples
                ]
            )
        )
        correct_5px_pairs = int(
            np.sum(
                [
                    int(
                        row[
                            "grouped_minimal_sample_correct_5px_pair_count_TARGET_ONLY"
                        ]
                    )
                    for row in grouped_samples
                ]
            )
        )
        output.update(
            {
                "grouped_minimal_sample_count": sample_count,
                "grouped_minimal_sample_all_correct_2px_rate_TARGET_ONLY": float(
                    all_correct_2px / max(sample_count, 1)
                ),
                "grouped_minimal_sample_all_correct_5px_rate_TARGET_ONLY": float(
                    all_correct_5px / max(sample_count, 1)
                ),
                "grouped_minimal_sample_correct_2px_pair_rate_TARGET_ONLY": float(
                    correct_2px_pairs / max(pair_count, 1)
                ),
                "grouped_minimal_sample_correct_5px_pair_rate_TARGET_ONLY": float(
                    correct_5px_pairs / max(pair_count, 1)
                ),
            }
        )
    profile_names = sorted(
        {
            str(profile_name)
            for row in rows
            for profile_name in dict(
                row.get("hypothesis_profile_audit_TARGET_ONLY", {})
            )
        }
    )
    if profile_names:
        profile_summary: dict[str, dict[str, object]] = {}
        for profile_name in profile_names:
            values = [
                dict(row["hypothesis_profile_audit_TARGET_ONLY"])[profile_name]
                for row in rows
                if profile_name
                in dict(row.get("hypothesis_profile_audit_TARGET_ONLY", {}))
            ]
            oracle_values = [
                value
                for value in values
                if value.get("oracle_translation_m") is not None
                and value.get("oracle_rotation_deg") is not None
            ]
            sample_count = int(
                np.sum([int(value.get("minimal_sample_count", 0)) for value in values])
            )
            pair_count = int(
                np.sum(
                    [int(value.get("minimal_sample_pair_count", 0)) for value in values]
                )
            )
            item: dict[str, object] = {
                "query_count": int(len(values)),
                "query_count_with_valid_hypothesis": int(len(oracle_values)),
                "median_valid_hypothesis_count": float(
                    np.median(
                        [int(value.get("valid_hypothesis_count", 0)) for value in values]
                    )
                ),
                "median_raw_hypothesis_count": float(
                    np.median(
                        [int(value.get("raw_hypothesis_count", 0)) for value in values]
                    )
                ),
                "median_latent_em_hypothesis_count": float(
                    np.median(
                        [
                            int(value.get("latent_em_hypothesis_count", 0))
                            for value in values
                        ]
                    )
                ),
                "minimal_sample_count": sample_count,
                "minimal_sample_all_correct_2px_rate": float(
                    np.sum(
                        [
                            int(value.get("minimal_sample_all_correct_2px_count", 0))
                            for value in values
                        ]
                    )
                    / max(sample_count, 1)
                ),
                "minimal_sample_all_correct_5px_rate": float(
                    np.sum(
                        [
                            int(value.get("minimal_sample_all_correct_5px_count", 0))
                            for value in values
                        ]
                    )
                    / max(sample_count, 1)
                ),
                "minimal_sample_correct_2px_pair_rate": float(
                    np.sum(
                        [
                            int(value.get("minimal_sample_correct_2px_pair_count", 0))
                            for value in values
                        ]
                    )
                    / max(pair_count, 1)
                ),
                "minimal_sample_correct_5px_pair_rate": float(
                    np.sum(
                        [
                            int(value.get("minimal_sample_correct_5px_pair_count", 0))
                            for value in values
                        ]
                    )
                    / max(pair_count, 1)
                ),
            }
            verified_count_values = [
                int(value["verified_hypothesis_count"])
                for value in values
                if "verified_hypothesis_count" in value
            ]
            if verified_count_values:
                item["median_verified_hypothesis_count"] = float(
                    np.median(verified_count_values)
                )
            if oracle_values:
                translations = np.asarray(
                    [float(value["oracle_translation_m"]) for value in oracle_values],
                    dtype=np.float64,
                )
                item.update(
                    {
                        "oracle_median_translation_m": float(np.median(translations)),
                        "oracle_p90_translation_m": float(
                            np.percentile(translations, 90)
                        ),
                        "oracle_median_rotation_deg": float(
                            np.median(
                                [
                                    float(value["oracle_rotation_deg"])
                                    for value in oracle_values
                                ]
                            )
                        ),
                        "oracle_recall_3cm_5deg": float(
                            np.mean(
                                [
                                    bool(value["oracle_3cm_5deg"])
                                    for value in oracle_values
                                ]
                            )
                        ),
                        "oracle_recall_5cm_5deg": float(
                            np.mean(
                                [
                                    bool(value["oracle_5cm_5deg"])
                                    for value in oracle_values
                                ]
                            )
                        ),
                        "oracle_recall_10cm_5deg": float(
                            np.mean(
                                [
                                    bool(value["oracle_10cm_5deg"])
                                    for value in oracle_values
                                ]
                            )
                        ),
                        "oracle_recall_25cm_2deg": float(
                            np.mean(
                                [
                                    bool(value["oracle_25cm_2deg"])
                                    for value in oracle_values
                                ]
                            )
                        ),
                        "median_catastrophic_hypothesis_count": float(
                            np.median(
                                [
                                    int(value["catastrophic_hypothesis_count"])
                                    for value in oracle_values
                                ]
                            )
                        ),
                    }
                )
            verified_oracle_values = [
                value
                for value in values
                if value.get("verified_oracle_translation_m") is not None
            ]
            if verified_oracle_values:
                verified_translations = np.asarray(
                    [
                        float(value["verified_oracle_translation_m"])
                        for value in verified_oracle_values
                    ],
                    dtype=np.float64,
                )
                item.update(
                    {
                        "verified_oracle_median_translation_m": float(
                            np.median(verified_translations)
                        ),
                        "verified_oracle_p90_translation_m": float(
                            np.percentile(verified_translations, 90)
                        ),
                        "verified_oracle_recall_3cm_5deg": float(
                            np.mean(
                                [
                                    bool(value["verified_oracle_3cm_5deg"])
                                    for value in verified_oracle_values
                                ]
                            )
                        ),
                        "verified_oracle_recall_5cm_5deg": float(
                            np.mean(
                                [
                                    bool(value["verified_oracle_5cm_5deg"])
                                    for value in verified_oracle_values
                                ]
                            )
                        ),
                        "verified_oracle_recall_10cm_5deg": float(
                            np.mean(
                                [
                                    bool(value["verified_oracle_10cm_5deg"])
                                    for value in verified_oracle_values
                                ]
                            )
                        ),
                    }
                )
            profile_summary[profile_name] = item
        output["hypothesis_profile_audit_TARGET_ONLY"] = profile_summary
    observability_audited = [
        row
        for row in rows
        if row.get("grouped_observability_rejected_hypothesis_count") is not None
    ]
    if observability_audited:
        rejection_counts = np.asarray(
            [
                int(row["grouped_observability_rejected_hypothesis_count"])
                for row in observability_audited
            ],
            dtype=np.int64,
        )
        output.update(
            {
                "grouped_observability_audited_query_count": int(
                    len(observability_audited)
                ),
                "grouped_observability_rejected_hypothesis_count": int(
                    np.sum(rejection_counts)
                ),
                "grouped_observability_median_rejected_hypotheses_per_query": float(
                    np.median(rejection_counts)
                ),
            }
        )
    promotion_rows = [
        row
        for row in rows
        if row.get("crossfit_promotion_promoted") is not None
    ]
    if promotion_rows:
        deltas = np.asarray(
            [
                float(row["crossfit_promotion_likelihood_mean_delta"])
                for row in promotion_rows
                if row.get("crossfit_promotion_likelihood_mean_delta") is not None
            ],
            dtype=np.float64,
        )
        comparable = [
            row
            for row in promotion_rows
            if row.get("crossfit_baseline_translation_m_TARGET_ONLY") is not None
            and row.get("crossfit_optional_translation_m_TARGET_ONLY") is not None
            and row.get("translation_m") is not None
        ]
        promotion_output: dict[str, object] = {
            "crossfit_promotion_query_count": int(len(promotion_rows)),
            "crossfit_promotion_count": int(
                np.sum(
                    [bool(row["crossfit_promotion_promoted"]) for row in promotion_rows]
                )
            ),
            "crossfit_promotion_rate": float(
                np.mean(
                    [bool(row["crossfit_promotion_promoted"]) for row in promotion_rows]
                )
            ),
            "crossfit_abstain_count": int(
                np.sum(
                    [bool(row["crossfit_promotion_abstained"]) for row in promotion_rows]
                )
            ),
            "crossfit_likelihood_mean_delta_median": (
                None if deltas.size == 0 else float(np.median(deltas))
            ),
        }
        if comparable:
            baseline_values = np.asarray(
                [
                    float(row["crossfit_baseline_translation_m_TARGET_ONLY"])
                    for row in comparable
                ],
                dtype=np.float64,
            )
            optional_values = np.asarray(
                [
                    float(row["crossfit_optional_translation_m_TARGET_ONLY"])
                    for row in comparable
                ],
                dtype=np.float64,
            )
            selected_values = np.asarray(
                [float(row["translation_m"]) for row in comparable],
                dtype=np.float64,
            )
            promotion_output.update(
                {
                    "crossfit_baseline_median_translation_m_TARGET_ONLY": float(
                        np.median(baseline_values)
                    ),
                    "crossfit_baseline_p90_translation_m_TARGET_ONLY": float(
                        np.percentile(baseline_values, 90)
                    ),
                    "crossfit_optional_median_translation_m_TARGET_ONLY": float(
                        np.median(optional_values)
                    ),
                    "crossfit_optional_p90_translation_m_TARGET_ONLY": float(
                        np.percentile(optional_values, 90)
                    ),
                    "crossfit_selected_vs_baseline_win_count_TARGET_ONLY": int(
                        np.sum(selected_values < baseline_values - 1e-12)
                    ),
                    "crossfit_selected_vs_baseline_loss_count_TARGET_ONLY": int(
                        np.sum(selected_values > baseline_values + 1e-12)
                    ),
                    "crossfit_optional_vs_baseline_win_count_TARGET_ONLY": int(
                        np.sum(optional_values < baseline_values - 1e-12)
                    ),
                    "crossfit_optional_vs_baseline_loss_count_TARGET_ONLY": int(
                        np.sum(optional_values > baseline_values + 1e-12)
                    ),
                }
            )
            selected_delta, selected_ci_low, selected_ci_high = (
                paired_bootstrap_delta_ci(
                    selected_values, baseline_values, resamples=10000, seed=1729
                )
            )
            optional_delta, optional_ci_low, optional_ci_high = (
                paired_bootstrap_delta_ci(
                    optional_values, baseline_values, resamples=10000, seed=1730
                )
            )
            promotion_output.update(
                {
                    "crossfit_selected_minus_baseline_mean_translation_m_TARGET_ONLY": selected_delta,
                    "crossfit_selected_minus_baseline_mean_translation_m_bootstrap95_ci_TARGET_ONLY": [
                        selected_ci_low,
                        selected_ci_high,
                    ],
                    "crossfit_optional_minus_baseline_mean_translation_m_TARGET_ONLY": optional_delta,
                    "crossfit_optional_minus_baseline_mean_translation_m_bootstrap95_ci_TARGET_ONLY": [
                        optional_ci_low,
                        optional_ci_high,
                    ],
                }
            )
        output.update(promotion_output)
    return output


def _policy_key(trial: dict[str, object]) -> tuple[float, ...]:
    pose = dict(trial["verified_pose"])
    values = np.asarray(
        [
            float(pose["median_translation_m_success"]),
            float(pose["p90_translation_m_success"]),
            float(pose["median_rotation_deg_success"]),
        ],
        dtype=np.float64,
    )
    geomean = float(np.exp(np.mean(np.log(np.maximum(values, 1e-12)))))
    return (
        float(pose["success_rate"]),
        -geomean,
        -float(pose["p90_translation_m_success"]),
        -float(pose["median_translation_m_success"]),
        -float(pose["median_rotation_deg_success"]),
        float(pose["recall_10cm_5deg"]),
        float(pose["recall_5cm_5deg"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if bool(args.development_cross_block_audit) and str(args.evaluation_role) != "development":
        raise ValueError("cross-block replay is development-only")
    if (
        str(args.grouped_hypothesis_selection_policy)
        in DIAGNOSTIC_GROUPED_HYPOTHESIS_SELECTION_POLICIES
        and str(args.evaluation_role) != "development"
    ):
        raise ValueError("diagnostic grouped selectors are development-only")
    if int(args.single_ransac_match_count) < 4:
        raise ValueError("single_ransac_match_count must be at least four")
    if int(args.max_hypothesis_information_audit) < -1:
        raise ValueError("max_hypothesis_information_audit must be >= -1")
    if int(args.query_shard_count) <= 0:
        raise ValueError("query_shard_count must be positive")
    if (
        not np.isfinite(float(args.candidate_identity_prior_temperature))
        or float(args.candidate_identity_prior_temperature) <= 0.0
    ):
        raise ValueError("candidate identity prior temperature must be positive")
    if not 0 <= int(args.query_shard_index) < int(args.query_shard_count):
        raise ValueError("query_shard_index must be in [0, query_shard_count)")
    if bool(args.grouped_only) and not bool(args.enable_grouped_candidate_pnp):
        raise ValueError("grouped_only requires grouped candidate PnP")
    if bool(args.grouped_only) and args.frozen_baseline_summary is not None:
        raise ValueError("grouped_only cannot perform external baseline policy replay")
    if str(args.immutable_baseline_pose_artifact) and not bool(
        args.enable_grouped_crossfit_likelihood_fallback
    ):
        raise ValueError(
            "immutable baseline pose replay requires grouped crossfit fallback"
        )
    if bool(args.export_selected_pose_artifact) and not bool(
        args.enable_grouped_candidate_pnp
    ):
        raise ValueError("selected pose export requires grouped candidate PnP")
    if str(args.single_ransac_selection_mode) not in {
        "score_topk",
        "spatial_round_robin",
    }:
        raise ValueError("unsupported single-RANSAC selection mode")
    assignment_modes = tuple(
        item.strip() for item in str(args.assignment_modes).split(",") if item.strip()
    )
    if not assignment_modes or set(assignment_modes) - {
        "row_argmax",
        "global_bipartite",
    }:
        raise ValueError("unsupported assignment mode")
    hypothesis_modes = tuple(
        item.strip()
        for item in str(args.hypothesis_selection_modes).split(",")
        if item.strip()
    )
    score_keys = tuple(
        item.strip() for item in str(args.score_keys).split(",") if item.strip()
    )
    if not score_keys:
        raise ValueError("score_keys cannot be empty")

    proposals_path = Path(args.proposals)
    candidate_path = Path(args.candidate_artifact)
    score_path = Path(args.score_artifact)
    bank_path = Path(args.projected_landmark_bank)
    split_path = Path(args.split_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_hypothesis_export_rows: list[dict[str, object]] = []
    selected_pose_export_rows: list[dict[str, object]] = []
    proposals = _load_npz(proposals_path)
    candidate = _load_npz(candidate_path)
    score_payload = _load_npz(score_path)
    spatial_paths = {
        "train": tuple(
            Path(value)
            for value in str(args.candidate_spatial_likelihood_train).split(",")
            if value
        ),
        "validation": (
            ()
            if not str(args.candidate_spatial_likelihood_validation)
            else (Path(args.candidate_spatial_likelihood_validation),)
        ),
        "test": (
            ()
            if not str(args.candidate_spatial_likelihood_test)
            else (Path(args.candidate_spatial_likelihood_test),)
        ),
    }
    generation_spatial_paths = {
        "validation": (
            None
            if not str(args.candidate_generation_spatial_likelihood_validation)
            else Path(args.candidate_generation_spatial_likelihood_validation)
        ),
        "test": (
            None
            if not str(args.candidate_generation_spatial_likelihood_test)
            else Path(args.candidate_generation_spatial_likelihood_test)
        ),
    }
    spatial_requested = any(paths for paths in spatial_paths.values())
    generation_spatial_requested = any(
        path is not None for path in generation_spatial_paths.values()
    )
    if spatial_requested != bool(str(args.candidate_evidence)):
        raise ValueError(
            "candidate_evidence and candidate spatial likelihoods must be provided together"
        )
    if spatial_requested and any(path is None for path in spatial_paths.values()):
        raise ValueError("both validation and test spatial likelihoods are required")
    if generation_spatial_requested and any(
        path is None for path in generation_spatial_paths.values()
    ):
        raise ValueError(
            "both validation and test generation spatial likelihoods are required"
        )
    if generation_spatial_requested and not (
        spatial_requested and bool(args.enable_grouped_candidate_pnp)
    ):
        raise ValueError(
            "frozen generation spatial likelihoods require grouped PnP and "
            "scoring spatial likelihoods"
        )
    if (
        spatial_requested
        and not generation_spatial_requested
        and str(args.candidate_generation_spatial_view_mixture_policy)
        != str(args.candidate_spatial_view_mixture_policy)
    ):
        raise ValueError(
            "different scoring/generation support-view policies require explicit "
            "generation spatial likelihood artifacts"
        )
    scoring_view_geometry_sigma = float(
        args.candidate_spatial_pose_view_geometry_sigma_deg
    )
    generation_view_geometry_sigma = float(
        args.candidate_generation_spatial_pose_view_geometry_sigma_deg
    )
    if (
        not np.isfinite(scoring_view_geometry_sigma)
        or scoring_view_geometry_sigma < 0.0
        or not np.isfinite(generation_view_geometry_sigma)
        or generation_view_geometry_sigma < 0.0
    ):
        raise ValueError(
            "candidate spatial pose-view geometry sigmas must be finite and non-negative"
        )
    if not spatial_requested and (
        scoring_view_geometry_sigma > 0.0
        or generation_view_geometry_sigma > 0.0
    ):
        raise ValueError(
            "pose-view geometry weighting requires candidate spatial likelihoods"
        )
    if (
        spatial_requested
        and not generation_spatial_requested
        and not np.isclose(
            scoring_view_geometry_sigma,
            generation_view_geometry_sigma,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise ValueError(
            "different scoring/generation view geometry requires explicit "
            "generation spatial likelihood artifacts"
        )
    spatial_calibration_path = (
        None
        if not str(args.candidate_spatial_calibration)
        else Path(args.candidate_spatial_calibration)
    )
    if spatial_calibration_path is not None and not spatial_requested:
        raise ValueError("candidate spatial calibration requires spatial likelihoods")
    if not 0.0 <= float(args.candidate_spatial_log_evidence_weight) <= 1.0:
        raise ValueError("candidate spatial log evidence weight must be in [0, 1]")
    if not 0.0 <= float(args.candidate_geometry_prior_mix_weight) <= 1.0:
        raise ValueError("candidate geometry prior mix weight must be in [0, 1]")
    if not 0.0 <= float(args.candidate_geometry_generation_mix_weight) <= 1.0:
        raise ValueError("candidate geometry generation mix weight must be in [0, 1]")
    if int(args.grouped_generation_min_strict_grid_delta) < 0:
        raise ValueError("grouped generation strict grid delta must be non-negative")
    if (
        not np.isfinite(float(args.grouped_likelihood_min_mean_delta))
        or float(args.grouped_likelihood_min_mean_delta) < 0.0
    ):
        raise ValueError("grouped likelihood delta must be finite and non-negative")
    if int(args.grouped_likelihood_min_effective_groups) < 1:
        raise ValueError("grouped likelihood gate requires effective groups")
    observability_thresholds = (
        float(args.grouped_observability_min_information_matches),
        float(args.grouped_observability_min_translation_eigenvalue),
        float(args.grouped_observability_max_translation_condition),
        float(args.grouped_observability_max_joint_condition),
        float(args.grouped_observability_min_bearing_span_deg),
        float(args.grouped_observability_min_depth_span_ratio),
        float(args.grouped_observability_min_xyz_second_ratio),
        float(args.grouped_observability_min_xyz_third_ratio),
    )
    if any(
        not np.isfinite(value) or value < 0.0
        for value in observability_thresholds
    ):
        raise ValueError("grouped observability thresholds must be non-negative")
    if float(args.grouped_crossfit_maplet_voxel_size_m) <= 0.0:
        raise ValueError("grouped cross-fit maplet voxel size must be positive")
    if (
        str(args.grouped_crossfit_mode) == "token_track_maplet_component"
        and args.maplet_support_index is None
    ):
        raise ValueError("explicit maplet cross-fit requires --maplet_support_index")
    if bool(args.enable_grouped_generation_grid_fallback) and bool(
        args.enable_grouped_crossfit_likelihood_fallback
    ):
        raise ValueError("grouped grid and cross-fit likelihood fallbacks are exclusive")
    if bool(args.enable_grouped_generation_grid_fallback) and (
        not bool(args.enable_grouped_candidate_pnp)
        or float(args.candidate_geometry_generation_mix_weight) <= 0.0
    ):
        raise ValueError(
            "grouped generation grid fallback requires grouped candidate PnP and "
            "positive geometry generation mixing"
        )
    if bool(args.enable_grouped_crossfit_likelihood_fallback) and (
        not bool(args.enable_grouped_candidate_pnp)
        or str(args.grouped_crossfit_mode) == "token_spatial"
    ):
        raise ValueError(
            "cross-fit likelihood fallback requires grouped PnP and a "
            "track-disjoint component cross-fit mode"
        )
    if not 0.0 <= float(
        args.candidate_spatial_geometry_calibration_weight
    ) <= 1.0:
        raise ValueError(
            "candidate spatial geometry calibration weight must be in [0, 1]"
        )
    candidate_evidence_path = (
        None if not spatial_requested else Path(args.candidate_evidence)
    )
    candidate_evidence = (
        None if candidate_evidence_path is None else _load_npz(candidate_evidence_path)
    )
    candidate_evidence_metadata = (
        {}
        if candidate_evidence is None
        else json.loads(str(candidate_evidence["metadata_json"].item()))
    )
    spatial_payloads = (
        {}
        if candidate_evidence_path is None
        else {
            split_name: _validate_spatial_likelihood_artifact_set(
                paths,
                split_name=split_name,
                candidate_evidence_path=candidate_evidence_path,
                candidate_evidence_metadata=candidate_evidence_metadata,
                score_path=score_path,
                proposals_path=proposals_path,
                candidate_path=candidate_path,
                bank_path=bank_path,
                allow_legacy_identity_dustbin=bool(
                    args.allow_legacy_candidate_spatial_dustbin
                ),
            )
            for split_name, paths in spatial_paths.items()
            if paths
        }
    )
    generation_spatial_payloads = (
        {}
        if not generation_spatial_requested
        else {
            split_name: _validate_spatial_likelihood_artifact(
                path,
                split_name=split_name,
                candidate_evidence_path=candidate_evidence_path,
                candidate_evidence_metadata=candidate_evidence_metadata,
                score_path=score_path,
                proposals_path=proposals_path,
                candidate_path=candidate_path,
                bank_path=bank_path,
                allow_legacy_identity_dustbin=False,
            )
            for split_name, path in generation_spatial_paths.items()
            if path is not None
        }
    )
    if any(
        json.loads(str(payload["metadata_json"].item())).get("format")
        not in {
            "candidate_spatial_likelihood_v5",
            "candidate_spatial_likelihood_v7",
        }
        for payload in generation_spatial_payloads.values()
    ):
        raise ValueError("frozen generation spatial likelihoods must use v5 or v7")
    spatial_calibration: CandidateSpatialLikelihoodCalibration | None = None
    has_v4_spatial = any(
        json.loads(str(payload["metadata_json"].item())).get("format")
        == "candidate_spatial_likelihood_v4"
        for payload in spatial_payloads.values()
    )
    if (
        has_v4_spatial
        and spatial_calibration_path is None
        and not bool(args.allow_uncalibrated_candidate_spatial_v4)
    ):
        raise ValueError(
            "production measurement density requires "
            "a query-disjoint spatial calibration for production pose evaluation"
        )
    if spatial_calibration_path is not None:
        spatial_calibration = load_candidate_spatial_likelihood_calibration(
            spatial_calibration_path,
            require_production=not bool(
                args.allow_diagnostic_candidate_spatial_calibration
            ),
        )
        calibrated_payloads: dict[str, dict[str, np.ndarray]] = {}
        for split_name, payload in spatial_payloads.items():
            metadata = json.loads(str(payload["metadata_json"].item()))
            spatial_calibration.validate_spatial_metadata(metadata)
            local, dustbin = spatial_calibration.apply(
                payload["local_log_probabilities"],
                payload["dustbin_probabilities"],
            )
            calibrated = dict(payload)
            calibrated["local_log_probabilities"] = local
            calibrated["dustbin_probabilities"] = dustbin
            calibrated_payloads[split_name] = calibrated
        spatial_payloads = calibrated_payloads
    geometry_probability_paths = {
        "validation": (
            None
            if not str(args.candidate_geometry_probabilities_validation)
            else Path(args.candidate_geometry_probabilities_validation)
        ),
        "test": (
            None
            if not str(args.candidate_geometry_probabilities_test)
            else Path(args.candidate_geometry_probabilities_test)
        ),
        "train": (
            None
            if not str(args.candidate_geometry_probabilities_train_oof)
            else Path(args.candidate_geometry_probabilities_train_oof)
        ),
    }
    legacy_update_prediction_paths = {
        "validation": (
            None
            if not str(args.candidate_update_predictions_validation)
            else Path(args.candidate_update_predictions_validation)
        ),
        "test": (
            None
            if not str(args.candidate_update_predictions_test)
            else Path(args.candidate_update_predictions_test)
        ),
    }
    utility_prediction_paths = {
        "validation": (
            None
            if not str(args.candidate_measurement_utility_validation)
            else Path(args.candidate_measurement_utility_validation)
        ),
        "test": (
            None
            if not str(args.candidate_measurement_utility_test)
            else Path(args.candidate_measurement_utility_test)
        ),
    }
    if any(path is not None for path in legacy_update_prediction_paths.values()) and any(
        path is not None for path in utility_prediction_paths.values()
    ):
        raise ValueError(
            "legacy candidate updates and measurement utility artifacts are mutually exclusive"
        )
    update_prediction_paths = (
        utility_prediction_paths
        if any(path is not None for path in utility_prediction_paths.values())
        else legacy_update_prediction_paths
    )
    update_predictions_requested = any(
        path is not None for path in update_prediction_paths.values()
    )
    if update_predictions_requested and any(
        path is None for path in update_prediction_paths.values()
    ):
        raise ValueError("both validation and test candidate updates are required")
    if bool(args.enable_optional_candidate_coordinate_refine) and not (
        update_predictions_requested
        and spatial_requested
        and bool(args.enable_grouped_candidate_pnp)
    ):
        raise ValueError(
            "candidate coordinate refinement requires grouped PnP plus aligned "
            "spatial and update artifacts"
        )
    if int(args.candidate_coordinate_min_updates) < 4:
        raise ValueError("candidate coordinate refinement needs at least four updates")
    if int(args.candidate_coordinate_min_grid_cells) <= 0:
        raise ValueError("candidate coordinate update grid coverage must be positive")
    if update_predictions_requested and not spatial_requested:
        raise ValueError("candidate coordinate updates require spatial evidence")
    if float(args.candidate_spatial_utility_gate_weight) != 0.0:
        raise ValueError(
            "measurement-update utility cannot gate candidate spatial pose "
            "likelihood; keep --candidate_spatial_utility_gate_weight=0 and use "
            "--enable_optional_candidate_coordinate_refine"
        )
    evaluation_geometry_probabilities_requested = any(
        geometry_probability_paths[name] is not None
        for name in ("validation", "test")
    )
    if evaluation_geometry_probabilities_requested and any(
        geometry_probability_paths[name] is None
        for name in ("validation", "test")
    ):
        raise ValueError("both validation and test geometry probabilities are required")
    train_geometry_probabilities_requested = (
        geometry_probability_paths["train"] is not None
    )
    geometry_probabilities_requested = bool(
        evaluation_geometry_probabilities_requested
        or train_geometry_probabilities_requested
    )
    if bool(args.export_train_grouped_hypotheses) and not bool(
        args.enable_grouped_candidate_pnp
    ):
        raise ValueError(
            "train grouped hypothesis export requires grouped candidate PnP"
        )
    if (
        _train_grouped_export_requires_geometry(args)
        and not train_geometry_probabilities_requested
    ):
        raise ValueError(
            "geometry-guided train grouped export requires train OOF geometry "
            "probabilities"
        )
    if float(args.candidate_geometry_prior_mix_weight) > 0.0 and not (
        evaluation_geometry_probabilities_requested and spatial_requested
    ):
        raise ValueError(
            "geometry prior mixing requires aligned geometry and spatial artifacts"
        )
    if float(args.candidate_geometry_generation_mix_weight) > 0.0 and not (
        evaluation_geometry_probabilities_requested and spatial_requested
    ):
        raise ValueError(
            "geometry-guided generation requires aligned geometry and spatial artifacts"
        )
    if float(args.candidate_spatial_geometry_calibration_weight) > 0.0 and not (
        evaluation_geometry_probabilities_requested and spatial_requested
    ):
        raise ValueError(
            "spatial geometry calibration requires aligned geometry and spatial artifacts"
        )
    if evaluation_geometry_probabilities_requested and not spatial_requested:
        raise ValueError("evaluation geometry probabilities require spatial evidence")
    geometry_probability_rows: dict[str, list[dict[str, str]]] = {}
    geometry_probability_metadata: dict[str, dict[str, object]] = {}
    if geometry_probabilities_requested:
        for split_name, path in geometry_probability_paths.items():
            if path is None:
                continue
            if split_name == "train":
                rows, metadata = (
                    _load_train_oof_candidate_geometry_probability_rows(path)
                )
            else:
                split_spatial_paths = spatial_paths[split_name]
                if len(split_spatial_paths) != 1:
                    raise RuntimeError(
                        "candidate geometry artifact configuration is incomplete"
                    )
                spatial_path = split_spatial_paths[0]
                rows, metadata = _load_candidate_geometry_probability_rows(
                    path,
                    spatial_path=spatial_path,
                    spatial_payload=spatial_payloads[split_name],
                )
            geometry_probability_rows[split_name] = rows
            geometry_probability_metadata[split_name] = metadata
    update_prediction_rows: dict[str, list[dict[str, str]]] = {}
    update_prediction_metadata: dict[str, dict[str, object]] = {}
    if update_predictions_requested:
        for split_name, path in update_prediction_paths.items():
            split_spatial_paths = spatial_paths[split_name]
            if path is None or len(split_spatial_paths) != 1:
                raise RuntimeError("candidate update artifact configuration is incomplete")
            spatial_path = split_spatial_paths[0]
            rows, metadata = _load_candidate_update_rows(
                path,
                spatial_path=spatial_path,
                spatial_payload=spatial_payloads[split_name],
            )
            if bool(metadata["requires_candidate_geometry_probabilities"]):
                geometry_metadata = geometry_probability_metadata.get(split_name)
                if geometry_metadata is None or metadata[
                    "candidate_geometry_verifier_sha256"
                ] != geometry_metadata.get("model_sha256"):
                    raise ValueError(
                        "candidate updates and candidate geometry probabilities differ"
                    )
            update_prediction_rows[split_name] = rows
            update_prediction_metadata[split_name] = metadata
        if len(
            {
                (
                    str(metadata["model_sha256"]),
                    float(metadata["update_threshold"]),
                    str(metadata["artifact_kind"]),
                )
                for metadata in update_prediction_metadata.values()
            }
        ) != 1:
            raise ValueError("validation/test candidate updates use different models")
    selected_rows = np.asarray(candidate["selected_rows"], dtype=np.int64)
    selected_columns = np.asarray(candidate["selected_columns"], dtype=np.int64)
    valid_edges = np.asarray(candidate["valid_edges"], dtype=bool)
    if candidate_evidence is not None:
        if not np.array_equal(candidate_evidence["selected_rows"], selected_rows):
            raise ValueError("candidate evidence rows differ from candidate artifact")
        for split_name, spatial_payload in spatial_payloads.items():
            source_rows = np.asarray(
                spatial_payload["source_query_rows"], dtype=np.int64
            )
            if not np.all(np.isin(source_rows, selected_rows)):
                raise ValueError(
                    f"{split_name} spatial likelihood contains unknown proposal rows"
                )
    if selected_columns.ndim != 2 or valid_edges.shape != selected_columns.shape:
        raise ValueError("candidate artifact arrays have incompatible shapes")
    candidate_metadata = json.loads(str(candidate["metadata_json"].item()))
    expected_candidate = {
        "proposals_sha256": file_sha256_short(proposals_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
    }
    candidate_mismatches = {
        key: {"expected": value, "actual": candidate_metadata.get(key)}
        for key, value in expected_candidate.items()
        if candidate_metadata.get(key) != value
    }
    if candidate_mismatches:
        raise ValueError(
            "stale candidate artifact: "
            f"{json.dumps(candidate_mismatches, sort_keys=True)}"
        )
    score_summary_path = score_path.parent / "summary.json"
    if not score_summary_path.exists():
        raise ValueError("score artifact requires sibling summary.json")
    score_summary = json.loads(score_summary_path.read_text())
    score_manifest = score_summary.get("data_manifest")
    if not isinstance(score_manifest, dict):
        raise ValueError("score summary is missing data_manifest")
    score_outputs = score_summary.get("outputs")
    recorded_score_hash = (
        None
        if not isinstance(score_outputs, dict)
        else score_outputs.get("scores_sha256")
    )
    actual_score_hash = file_sha256_short(score_path)
    if not recorded_score_hash or str(recorded_score_hash) != actual_score_hash:
        raise ValueError("score artifact hash differs from its sibling summary")
    expected_score = {
        "proposals_sha256": file_sha256_short(proposals_path),
        "feature_artifact_sha256": file_sha256_short(candidate_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
    }
    score_mismatches = {
        key: {"expected": value, "actual": score_manifest.get(key)}
        for key, value in expected_score.items()
        if score_manifest.get(key) != value
    }
    if score_mismatches:
        raise ValueError(
            "stale or misaligned score artifact: "
            f"{json.dumps(score_mismatches, sort_keys=True)}"
        )
    frozen_baseline_source = (
        None
        if args.frozen_baseline_summary is None
        else _load_frozen_baseline_policy(
            Path(args.frozen_baseline_summary),
            proposals_path=proposals_path,
            candidate_path=candidate_path,
            bank_path=bank_path,
            split_path=split_path,
            baseline_score_key=str(args.frozen_baseline_source_score_key),
        )
    )
    if frozen_baseline_source is not None:
        score_protocol = score_summary.get("protocol")
        score_baseline = (
            None
            if not isinstance(score_protocol, dict)
            else score_protocol.get("baseline_strategy")
        )
        normalized_score_baseline = str(score_baseline)
        if not normalized_score_baseline.startswith("strategy__"):
            normalized_score_baseline = f"strategy__{normalized_score_baseline}"
        if normalized_score_baseline != str(args.frozen_baseline_source_score_key):
            raise ValueError(
                "score artifact baseline identity differs from frozen baseline"
            )
        if (
            int(args.single_ransac_match_count)
            != int(frozen_baseline_source["max_matches"])
            or str(args.single_ransac_selection_mode)
            != str(frozen_baseline_source["selection_mode"])
        ):
            raise ValueError(
                "single-RANSAC policy must replay the frozen baseline K/mode"
            )

    landmark_index, landmark_metadata = load_landmark_index_npz(bank_path)
    maplet_path = (
        None
        if args.maplet_support_index is None
        else Path(args.maplet_support_index)
    )
    maplet_metadata = None
    maplet_index = None
    maplet_cluster_ids_by_bank_row = None
    if maplet_path is not None:
        maplet_index, maplet_metadata = load_local_maplet_support_index_npz(
            maplet_path
        )
        if not np.array_equal(
            np.asarray(maplet_index.anchor_track_ids, dtype=np.int64),
            np.asarray(landmark_index.track_ids, dtype=np.int64),
        ):
            raise ValueError("maplet support index and landmark bank rows differ")
        source_hash = str(maplet_metadata.get("source_landmark_index_sha256", ""))
        if source_hash and source_hash != file_sha256_short(bank_path):
            raise ValueError("maplet support index references a different landmark bank")
        maplet_cluster_ids_by_bank_row = build_disjoint_maplet_cluster_ids(
            maplet_index
        )
    compact_tracks = _compact(
        proposals["candidate_track_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    compact_prototypes = _compact(
        proposals["candidate_prototype_ids"], selected_rows, selected_columns
    ).astype(np.int64)
    canonical_rows = canonical_rows_for_track_candidates(
        compact_tracks, landmark_index.track_ids
    )
    valid_edges &= canonical_rows >= 0
    query_ids = np.asarray(proposals["query_ids"])[selected_rows].astype(str)
    query_xy = np.asarray(proposals["xy"], dtype=np.float32)[selected_rows]
    if candidate_evidence is not None and (
        not np.array_equal(candidate_evidence["query_ids"].astype(str), query_ids)
        or not np.allclose(candidate_evidence["query_xy"], query_xy, rtol=0.0, atol=1e-5)
    ):
        raise ValueError("candidate evidence query rows differ from evaluator rows")
    split = json.loads(split_path.read_text())
    for name in ("train", "validation", "test"):
        if name not in split or not isinstance(split[name], list) or not split[name]:
            raise ValueError("split JSON requires non-empty train/validation/test lists")
    split_masks = {
        name: np.isin(query_ids, np.asarray(split[name], dtype=np.str_))
        for name in ("train", "validation", "test")
    }
    execution_split = {
        name: _query_execution_shard(
            split[name],
            shard_count=int(args.query_shard_count),
            shard_index=int(args.query_shard_index),
        )
        for name in ("train", "validation", "test")
    }
    if any(
        np.any(split_masks[left] & split_masks[right])
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("train/validation/late splits overlap")
    posterior_overlay_paths = tuple(
        Path(value.strip())
        for value in str(args.candidate_posterior_overlays).split(",")
        if value.strip()
    )
    posterior_overlay_manifest: list[dict[str, object]] = []
    if posterior_overlay_paths:
        query_count = int(candidate_metadata.get("query_points_per_image", 0))
        if query_count <= 0 or len(query_ids) % query_count:
            raise ValueError("candidate rows do not form fixed whole-image blocks")
        image_id_blocks = query_ids.reshape(-1, query_count)
        if not np.all(image_id_blocks == image_id_blocks[:, :1]):
            raise ValueError("candidate rows are not contiguous whole-image blocks")
        overlay_candidates, overlay_null, posterior_overlay_manifest = (
            _load_candidate_posterior_ensemble(
                posterior_overlay_paths,
                expected_image_ids=image_id_blocks[:, 0],
                expected_split=split,
                query_count=query_count,
                candidate_count=selected_columns.shape[1],
                candidate_path=candidate_path,
                proposals_path=proposals_path,
                bank_path=bank_path,
                split_path=split_path,
                descriptor_space_id=landmark_metadata.get("descriptor_space_id"),
            )
        )
        if (
            POSTERIOR_OVERLAY_CANDIDATE_KEY in score_payload
            or POSTERIOR_OVERLAY_NULL_KEY in score_payload
        ):
            raise ValueError("posterior overlay keys collide with score artifact")
        score_payload[POSTERIOR_OVERLAY_CANDIDATE_KEY] = overlay_candidates
        score_payload[POSTERIOR_OVERLAY_NULL_KEY] = overlay_null
    geometry_probability_matrices: dict[str, np.ndarray] = {}
    if geometry_probabilities_requested:
        if candidate_evidence is None:
            raise RuntimeError("candidate geometry probabilities require candidate evidence")
        evidence_columns = np.asarray(
            candidate_evidence["candidate_compact_columns"], dtype=np.int64
        )
        compact_row_by_source = {
            int(source_row): int(compact_row)
            for compact_row, source_row in enumerate(selected_rows.tolist())
        }
        for split_name, probability_rows in geometry_probability_rows.items():
            matrix = np.full(valid_edges.shape, np.nan, dtype=np.float64)
            seen: set[tuple[int, int]] = set()
            for probability_row in probability_rows:
                source_row = int(probability_row["source_query_row"])
                compact_row = compact_row_by_source.get(source_row)
                if compact_row is None or not bool(split_masks[split_name][compact_row]):
                    raise ValueError(
                        "candidate geometry probability row has an unknown split/source row"
                    )
                candidate_rank = int(
                    probability_row["candidate_measurement_rank"]
                ) - 1
                if not 0 <= candidate_rank < evidence_columns.shape[1]:
                    raise ValueError("candidate geometry rank is outside evidence top-M")
                column = int(evidence_columns[compact_row, candidate_rank])
                if column < 0 or not bool(valid_edges[compact_row, column]):
                    raise ValueError("candidate geometry probability slot is invalid")
                identity = (compact_row, column)
                if identity in seen:
                    raise ValueError("duplicate candidate geometry probability slot")
                seen.add(identity)
                if (
                    str(probability_row["query_id"]) != str(query_ids[compact_row])
                    or int(probability_row["track_id"])
                    != int(compact_tracks[compact_row, column])
                    or int(probability_row["prototype_id"])
                    != int(compact_prototypes[compact_row, column])
                ):
                    raise ValueError("candidate geometry probability identity differs")
                probability = float(probability_row["geometry_probability"])
                if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError("candidate geometry probability is outside [0, 1]")
                matrix[compact_row, column] = probability
            if not seen:
                raise ValueError("candidate geometry probability artifact aligned no rows")
            geometry_probability_matrices[split_name] = matrix
    update_probability_matrices: dict[str, np.ndarray] = {}
    update_refined_xy_matrices: dict[str, np.ndarray] = {}
    if update_predictions_requested:
        if candidate_evidence is None:
            raise RuntimeError("candidate updates require candidate evidence")
        evidence_columns = np.asarray(
            candidate_evidence["candidate_compact_columns"], dtype=np.int64
        )
        compact_row_by_source = {
            int(source_row): int(compact_row)
            for compact_row, source_row in enumerate(selected_rows.tolist())
        }
        for split_name, prediction_rows in update_prediction_rows.items():
            spatial_payload = spatial_payloads[split_name]
            spatial_identity_by_slot: dict[tuple[int, int], str] = {}
            for source_row, measurement_rank, identity in zip(
                np.asarray(spatial_payload["source_query_rows"], dtype=np.int64),
                np.asarray(
                    spatial_payload["candidate_measurement_ranks"], dtype=np.int64
                ),
                np.asarray(spatial_payload["candidate_identity_keys"]).astype(str),
            ):
                key = (int(source_row), int(measurement_rank) - 1)
                previous = spatial_identity_by_slot.setdefault(key, str(identity))
                if previous != str(identity):
                    raise ValueError("spatial support views disagree on candidate identity")
            probability_matrix = np.full(valid_edges.shape, np.nan, dtype=np.float64)
            refined_matrix = np.full(
                (*valid_edges.shape, 2), np.nan, dtype=np.float64
            )
            seen: set[tuple[int, int]] = set()
            for prediction_row in prediction_rows:
                source_row = int(prediction_row["source_query_row"])
                compact_row = compact_row_by_source.get(source_row)
                if compact_row is None or not bool(split_masks[split_name][compact_row]):
                    raise ValueError(
                        "candidate update row has an unknown split/source row"
                    )
                candidate_rank = int(
                    prediction_row["candidate_measurement_rank"]
                ) - 1
                if not 0 <= candidate_rank < evidence_columns.shape[1]:
                    raise ValueError("candidate update rank is outside evidence top-M")
                column = int(evidence_columns[compact_row, candidate_rank])
                if column < 0 or not bool(valid_edges[compact_row, column]):
                    raise ValueError("candidate update probability slot is invalid")
                slot = (compact_row, column)
                if slot in seen:
                    raise ValueError("duplicate candidate update probability slot")
                seen.add(slot)
                expected_identity = spatial_identity_by_slot.get(
                    (source_row, candidate_rank)
                )
                if expected_identity != str(
                    prediction_row["candidate_identity_key"]
                ):
                    raise ValueError("candidate update identity differs from spatial RGB")
                if (
                    str(prediction_row["query_id"]) != str(query_ids[compact_row])
                    or int(prediction_row["track_id"])
                    != int(compact_tracks[compact_row, column])
                    or int(prediction_row["prototype_id"])
                    != int(compact_prototypes[compact_row, column])
                ):
                    raise ValueError("candidate update candidate identity differs")
                center = np.asarray(
                    [
                        float(prediction_row["center_x"]),
                        float(prediction_row["center_y"]),
                    ],
                    dtype=np.float64,
                )
                if not np.allclose(
                    center, query_xy[compact_row], rtol=0.0, atol=1e-4
                ):
                    raise ValueError("candidate update center differs from coarse query xy")
                probability = float(
                    prediction_row["update_beneficial_probability"]
                )
                refined = np.asarray(
                    [
                        float(prediction_row["refined_x"]),
                        float(prediction_row["refined_y"]),
                    ],
                    dtype=np.float64,
                )
                if (
                    not np.isfinite(probability)
                    or not 0.0 <= probability <= 1.0
                    or not np.all(np.isfinite(refined))
                ):
                    raise ValueError("candidate coordinate update values are invalid")
                probability_matrix[compact_row, column] = probability
                refined_matrix[compact_row, column] = refined
            if not seen:
                raise ValueError("candidate update artifact aligned no rows")
            update_probability_matrices[split_name] = probability_matrix
            update_refined_xy_matrices[split_name] = refined_matrix
    model_dir = Path(args.colmap_model_dir)
    cameras = read_colmap_cameras_binary(model_dir / "cameras.bin")
    images = read_colmap_images_binary(model_dir / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    camera_centers_by_name = {
        str(image.image_name): camera_center_from_qvec_tvec(
            image.qvec, image.tvec
        )
        for image in images.values()
    }
    immutable_baseline_pose_source = (
        None
        if not str(args.immutable_baseline_pose_artifact)
        else _load_immutable_baseline_pose_artifact(
            Path(args.immutable_baseline_pose_artifact),
            expected_colmap_cameras_sha256=file_sha256_short(
                model_dir / "cameras.bin"
            ),
            expected_colmap_images_sha256=file_sha256_short(
                model_dir / "images.bin"
            ),
            evaluation_label=str(
                args.immutable_baseline_pose_evaluation_label
            ),
        )
    )
    if immutable_baseline_pose_source is not None:
        baseline_records = immutable_baseline_pose_source["records"]
        required_splits = _immutable_baseline_required_splits(args)
        missing = [
            (split_name, str(query_id))
            for split_name in required_splits
            for query_id in execution_split[split_name]
            if (split_name, str(query_id)) not in baseline_records
        ]
        if missing:
            raise ValueError(
                "immutable baseline pose artifact does not cover this execution: "
                f"{missing[:5]}"
            )

    config = VerifiedPnPConfig(
        fit_match_counts=tuple(args.fit_match_counts),
        selection_modes=hypothesis_modes,
        ransac_thresholds_px=tuple(args.ransac_thresholds_px),
        rng_seed_offsets=tuple(args.rng_seed_offsets),
        ransac_iterations=int(args.ransac_iterations),
        holdout_folds=int(args.holdout_folds),
        holdout_fold=int(args.holdout_fold),
        verification_strict_px=float(args.verification_strict_px),
        verification_loose_px=float(args.verification_loose_px),
        final_consensus_px=float(args.final_consensus_px),
        final_refine_f_scale_px=float(args.final_refine_f_scale_px),
        min_final_inliers=int(args.min_final_inliers),
        enable_final_refine=bool(args.enable_final_refine),
        candidate_pool_residual_sigma_px=float(
            args.candidate_pool_residual_sigma_px
        ),
        candidate_pool_hard_threshold_px=float(
            args.candidate_pool_hard_threshold_px
        ),
        candidate_pool_descriptor_rank_weight=float(
            args.candidate_pool_descriptor_rank_weight
        ),
        candidate_pool_refine_iterations=int(args.candidate_pool_refine_iterations),
    )
    grouped_config = None if not bool(args.enable_grouped_candidate_pnp) else GroupedCandidatePnPConfig(
        candidate_limits=tuple(args.grouped_candidate_limits),
        samples_per_limit=int(args.grouped_samples_per_limit),
        sampling_temperatures=tuple(args.grouped_sampling_temperatures),
        fit_match_counts=tuple(args.grouped_fit_match_counts),
        ransac_thresholds_px=tuple(args.grouped_ransac_thresholds_px),
        ransac_iterations=int(args.grouped_ransac_iterations),
        holdout_folds=int(args.holdout_folds),
        verification_fold=int(args.holdout_fold),
        verification_fold_count=int(args.grouped_rank_fold_count),
        final_audit_fold=(
            int(args.holdout_fold) + int(args.grouped_rank_fold_count)
        )
        % int(args.holdout_folds),
        crossfit_mode=str(args.grouped_crossfit_mode),
        crossfit_role_assignment=str(args.grouped_crossfit_role_assignment),
        crossfit_spatial_fold_policy=str(
            args.grouped_crossfit_spatial_fold_policy
        ),
        crossfit_maplet_voxel_size_m=float(
            args.grouped_crossfit_maplet_voxel_size_m
        ),
        independent_shortlist_pool=bool(
            args.enable_grouped_independent_shortlist_pool
        ),
        min_fit_matches=int(args.grouped_min_fit_matches),
        min_fit_grid_cells=int(args.grouped_min_fit_grid_cells),
        min_xyz_second_singular_ratio=float(
            args.grouped_min_xyz_second_singular_ratio
        ),
        generation_mode=str(args.grouped_generation_mode),
        prosac_hypotheses_per_limit=int(
            args.grouped_prosac_hypotheses_per_limit
        ),
        prosac_minimal_set_size=int(args.grouped_prosac_minimal_set_size),
        prosac_minimal_set_sizes=tuple(args.grouped_prosac_minimal_set_sizes),
        prosac_group_probability_power=float(
            args.grouped_prosac_group_probability_power
        ),
        prosac_candidate_probability_power=float(
            args.grouped_prosac_candidate_probability_power
        ),
        prosac_max_sample_attempts=int(args.grouped_prosac_max_sample_attempts),
        prosac_min_bearing_span_deg=float(
            args.grouped_prosac_min_bearing_span_deg
        ),
        prosac_min_translation_information_eigenvalue=float(
            args.grouped_prosac_min_translation_eigenvalue
        ),
        prosac_max_translation_information_condition=float(
            args.grouped_prosac_max_translation_condition
        ),
        prosac_max_joint_information_condition=float(
            args.grouped_prosac_max_joint_condition
        ),
        prosac_min_depth_span_ratio=float(
            args.grouped_prosac_min_depth_span_ratio
        ),
        prosac_min_xyz_third_singular_ratio=float(
            args.grouped_prosac_min_xyz_third_ratio
        ),
        prosac_observability_evidence_mode=str(
            args.grouped_prosac_observability_evidence_mode
        ),
        prosac_local_optimization=not bool(
            args.disable_grouped_prosac_local_optimization
        ),
        prosac_local_consensus_px=float(args.grouped_prosac_local_consensus_px),
        prosac_local_min_matches=int(args.grouped_prosac_local_min_matches),
        prosac_use_spatial_modes=bool(args.grouped_prosac_use_spatial_modes),
        prosac_verification_top_k=int(
            args.grouped_prosac_verification_top_k
        ),
        prosac_shortlist_evidence_mode=str(
            args.grouped_prosac_shortlist_evidence_mode
        ),
        prosac_spatial_rescore_top_k=int(
            args.grouped_prosac_spatial_rescore_top_k
        ),
        prosac_shortlist_selection_mode=str(
            args.grouped_prosac_shortlist_selection_mode
        ),
        prosac_shortlist_diverse_count=int(
            args.grouped_prosac_shortlist_diverse_count
        ),
        prosac_shortlist_min_per_profile=int(
            args.grouped_prosac_shortlist_min_per_profile
        ),
        prosac_shortlist_translation_diversity_m=float(
            args.grouped_prosac_shortlist_translation_diversity_m
        ),
        prosac_shortlist_rotation_diversity_deg=float(
            args.grouped_prosac_shortlist_rotation_diversity_deg
        ),
        prosac_profiles=tuple(args.grouped_prosac_profiles_json),
        latent_em_enabled=bool(args.enable_grouped_latent_em),
        latent_em_seed_count=int(args.grouped_latent_em_seed_count),
        latent_em_seed_min_per_profile=int(
            args.grouped_latent_em_seed_min_per_profile
        ),
        latent_em_seed_evidence_mode=str(
            args.grouped_latent_em_seed_evidence_mode
        ),
        latent_em_translation_diversity_m=float(
            args.grouped_latent_em_translation_diversity_m
        ),
        latent_em_rotation_diversity_deg=float(
            args.grouped_latent_em_rotation_diversity_deg
        ),
        latent_em_config=LatentEMConfig(
            iterations=int(args.grouped_latent_em_iterations),
            identity_temperature=float(
                args.grouped_latent_em_identity_temperature
            ),
            identity_temperature_floor=float(
                args.grouped_latent_em_identity_temperature_floor
            ),
            null_mass_floor=float(args.grouped_latent_em_null_mass_floor),
            null_likelihood=float(args.grouped_latent_em_null_likelihood),
            outlier_likelihood=float(
                args.grouped_latent_em_outlier_likelihood
            ),
            residual_sigma_px=float(
                args.grouped_latent_em_residual_sigma_px
            ),
            spatial_evidence_weight=(
                None
                if float(args.grouped_latent_em_spatial_evidence_weight) < 0.0
                else float(args.grouped_latent_em_spatial_evidence_weight)
            ),
            coordinate_update_policy=str(
                args.grouped_latent_em_coordinate_update_policy
            ),
            minimum_coordinate_mode_probability=float(
                args.grouped_latent_em_minimum_coordinate_mode_probability
            ),
            max_responsibility_change=float(
                args.grouped_latent_em_max_responsibility_change
            ),
            min_candidate_weight=float(
                args.grouped_latent_em_min_candidate_weight
            ),
            min_effective_group_mass=float(
                args.grouped_latent_em_min_effective_group_mass
            ),
            min_effective_groups=int(
                args.grouped_latent_em_min_effective_groups
            ),
            enforce_track_capacity=not bool(
                args.disable_grouped_latent_em_track_capacity
            ),
            robust_f_scale_px=float(
                args.grouped_latent_em_robust_f_scale_px
            ),
            max_nfev=int(args.grouped_latent_em_max_nfev),
            max_translation_step_m=float(
                args.grouped_latent_em_max_translation_step_m
            ),
            max_rotation_step_deg=float(
                args.grouped_latent_em_max_rotation_step_deg
            ),
        ),
        verification_strict_px=float(args.verification_strict_px),
        verification_loose_px=float(args.verification_loose_px),
        candidate_pool_residual_sigma_px=float(
            args.candidate_pool_residual_sigma_px
        ),
        candidate_pose_outlier_likelihood=float(
            args.candidate_pose_outlier_likelihood
        ),
        candidate_pose_null_likelihood=float(
            args.candidate_pose_null_likelihood
        ),
        candidate_pose_relation_neighbor_k=int(
            args.candidate_pose_relation_neighbor_k
        ),
        candidate_pose_relation_sigma_px=float(
            args.candidate_pose_relation_sigma_px
        ),
        candidate_pose_relation_outlier_likelihood=float(
            args.candidate_pose_relation_outlier_likelihood
        ),
        candidate_relation_feature_neighbor_k=int(
            args.candidate_relation_feature_neighbor_k
        ),
        candidate_relation_feature_max_modes=int(
            args.candidate_relation_feature_max_modes
        ),
        candidate_pool_hard_threshold_px=float(
            args.candidate_pool_hard_threshold_px
        ),
        candidate_pool_descriptor_rank_weight=float(
            args.candidate_pool_descriptor_rank_weight
        ),
        final_consensus_px=float(args.final_consensus_px),
        final_refine_f_scale_px=float(args.final_refine_f_scale_px),
        min_final_inliers=int(args.min_final_inliers),
        enable_final_refine=bool(args.enable_final_refine),
        final_refine_mode=str(args.grouped_final_refine_mode),
        final_refine_acceptance_policy=str(
            args.grouped_final_refine_acceptance_policy
        ),
        hypothesis_selection_policy=str(
            args.grouped_hypothesis_selection_policy
        ),
    )
    optional_grouped_config = grouped_config
    if (
        grouped_config is not None
        and str(args.optional_grouped_final_refine_acceptance_policy)
        != "same_as_immutable_baseline"
    ):
        optional_grouped_config = replace(
            grouped_config,
            final_refine_acceptance_policy=str(
                args.optional_grouped_final_refine_acceptance_policy
            ),
        )
    immutable_grouped_config = (
        None
        if grouped_config is None
        else replace(
            grouped_config,
            generation_mode=str(args.grouped_immutable_baseline_generation_mode),
            latent_em_enabled=False,
            final_refine_mode="hard_assignment",
            enable_candidate_coordinate_refine=False,
        )
    )
    if grouped_config is not None and bool(
        args.enable_optional_candidate_coordinate_refine
    ):
        optional_grouped_config = replace(
            optional_grouped_config,
            enable_candidate_coordinate_refine=True,
            min_candidate_coordinate_updates=int(
                args.candidate_coordinate_min_updates
            ),
            min_candidate_coordinate_update_grid_cells=int(
                args.candidate_coordinate_min_grid_cells
            ),
        )
    raw_scores = {
        key: _score_array(
            score_payload,
            proposals,
            key,
            selected_rows=selected_rows,
            selected_columns=selected_columns,
        )
        for key in (*score_keys, str(args.baseline_score_key))
    }
    grouped_null_scores = (
        None
        if not bool(args.enable_grouped_candidate_pnp)
        else _score_array(
            score_payload,
            proposals,
            str(args.grouped_null_score_key),
            selected_rows=selected_rows,
            selected_columns=selected_columns,
        )
    )
    for values in raw_scores.values():
        values[~valid_edges] = -np.inf

    policy_scores: dict[str, np.ndarray] = {}
    policy_metadata: dict[str, dict[str, object]] = {}
    for score_key, values in raw_scores.items():
        if "row_argmax" in assignment_modes:
            key = f"row_argmax__{score_key}"
            policy_scores[key] = values
            policy_metadata[key] = {
                "assignment_mode": "row_argmax_then_conflict_resolution",
                "score_key": score_key,
            }
        if "global_bipartite" in assignment_modes:
            key = f"global_bipartite__{score_key}"
            resolved, _selected = global_assignment_score_matrix(
                compact_tracks,
                values,
                query_ids,
                valid_mask=valid_edges,
                dustbin_score=None,
            )
            policy_scores[key] = resolved
            policy_metadata[key] = {
                "assignment_mode": "whole_image_sparse_bipartite_per_query_dustbin",
                "score_key": score_key,
            }

    def matches_by_query(scores: np.ndarray, split_name: str) -> dict[str, list[QueryTo3DMatch]]:
        selected = _selected_columns(scores, valid_edges)
        output: dict[str, list[QueryTo3DMatch]] = {}
        for row in np.flatnonzero(split_masks[split_name]).tolist():
            column = int(selected[row])
            if column < 0:
                continue
            query_id = str(query_ids[row])
            output.setdefault(query_id, []).append(
                QueryTo3DMatch(
                    token_index=int(selected_rows[row]),
                    xy=np.asarray(query_xy[row], dtype=np.float64),
                    track_id=int(compact_tracks[row, column]),
                    xyz=np.asarray(
                        landmark_index.xyz[int(canonical_rows[row, column])],
                        dtype=np.float64,
                    ),
                    similarity=float(scores[row, column]),
                    ratio=0.0,
                    landmark_variance=float(
                        landmark_index.mean_variances[
                            int(canonical_rows[row, column])
                        ]
                    ),
                    source="heldout_pose_hypothesis_eval",
                    prototype_id=int(compact_prototypes[row, column]),
                )
            )
        return output

    def candidate_pools_by_query(
        scores: np.ndarray,
        split_name: str,
        *,
        null_scores: np.ndarray | None = None,
        spatial_payload: dict[str, np.ndarray] | None = None,
        geometry_probabilities: np.ndarray | None = None,
        update_probabilities: np.ndarray | None = None,
        update_refined_xy: np.ndarray | None = None,
        update_threshold: float = 0.5,
        spatial_utility_gate_weight: float = 0.0,
        spatial_view_mixture_policy: str = "frozen_candidate_posterior",
        pose_view_geometry_sigma_deg: float = 0.0,
    ) -> dict[str, PoseVerificationCandidatePool]:
        output: dict[str, PoseVerificationCandidatePool] = {}
        for query_id in execution_split[split_name]:
            rows = np.flatnonzero(
                split_masks[split_name] & (query_ids == str(query_id))
            )
            xyz = np.zeros((*canonical_rows[rows].shape, 3), dtype=np.float64)
            local_valid = valid_edges[rows]
            xyz[local_valid] = landmark_index.xyz[
                canonical_rows[rows][local_valid]
            ]
            local_null = None
            if null_scores is not None:
                repeated = np.asarray(null_scores[rows], dtype=np.float64)
                if repeated.ndim != 2 or repeated.shape[1] == 0:
                    raise ValueError("grouped null score array has an invalid shape")
                if not np.allclose(
                    repeated, repeated[:, :1], rtol=0.0, atol=2e-5
                ):
                    raise ValueError(
                        "grouped null score must be constant inside each candidate set"
                    )
                local_null = repeated[:, 0]
            spatial_likelihood = None
            if spatial_payload is not None:
                if candidate_evidence is None:
                    raise RuntimeError("spatial likelihood requires candidate evidence")
                spatial_metadata = json.loads(
                    str(spatial_payload["metadata_json"].item())
                )
                support_probability_semantics = str(
                    spatial_metadata.get(
                        "support_view_probability_semantics", ""
                    )
                )
                if (
                    str(spatial_view_mixture_policy)
                    == "artifact_pose_view_posterior"
                    and support_probability_semantics
                    != POSE_VIEW_MIXTURE_SEMANTICS
                ):
                    raise ValueError(
                        "artifact pose-view policy requires a learned RGB "
                        "pose-view mixture artifact"
                    )
                offsets = np.asarray(spatial_payload["offsets_xy"], dtype=np.float64)
                source_rows = np.asarray(
                    spatial_payload["source_query_rows"], dtype=np.int64
                )
                source_ranks = np.asarray(
                    spatial_payload["candidate_measurement_ranks"], dtype=np.int64
                ) - 1
                view_ranks = np.asarray(
                    spatial_payload["support_view_ranks"], dtype=np.int64
                )
                max_views = max(
                    1,
                    int(
                        np.max(view_ranks, initial=-1)
                        + 1
                    ),
                )
                shape = (len(rows), valid_edges.shape[1], max_views)
                log_maps = np.full(
                    (*shape, len(offsets)), np.nan, dtype=np.float16
                )
                view_probabilities = np.zeros(shape, dtype=np.float32)
                dustbin_probabilities = np.zeros(shape, dtype=np.float32)
                spatial_valid = np.zeros(shape, dtype=bool)
                support_camera_centers = (
                    None
                    if float(pose_view_geometry_sigma_deg) <= 0.0
                    else np.full((*shape, 3), np.nan, dtype=np.float64)
                )
                query_global_rows = selected_rows[rows]
                local_by_source = {
                    int(source_row): int(local_row)
                    for local_row, source_row in enumerate(query_global_rows.tolist())
                }
                compact_by_source = {
                    int(source_row): int(compact_row)
                    for compact_row, source_row in zip(
                        rows.tolist(), query_global_rows.tolist()
                    )
                }
                relevant = np.flatnonzero(
                    np.isin(source_rows, query_global_rows)
                )
                if len(relevant) == 0:
                    raise ValueError(
                        f"query {query_id} has no aligned spatial likelihood rows"
                    )
                for spatial_row in relevant.tolist():
                    source_row = int(source_rows[spatial_row])
                    local_row = local_by_source[source_row]
                    compact_row = compact_by_source[source_row]
                    if str(spatial_payload["query_ids"][spatial_row]) != str(query_id):
                        raise ValueError("spatial likelihood query identity differs")
                    candidate_rank = int(source_ranks[spatial_row])
                    if not 0 <= candidate_rank < candidate_evidence[
                        "candidate_compact_columns"
                    ].shape[1]:
                        raise ValueError("spatial candidate rank is outside evidence top-M")
                    column = int(
                        candidate_evidence["candidate_compact_columns"][
                            compact_row, candidate_rank
                        ]
                    )
                    view_rank = int(view_ranks[spatial_row])
                    if column < 0 or not 0 <= view_rank < max_views:
                        raise ValueError("spatial likelihood candidate slot is invalid")
                    expected_track = int(compact_tracks[compact_row, column])
                    expected_prototype = int(compact_prototypes[compact_row, column])
                    if (
                        int(spatial_payload["candidate_track_ids"][spatial_row])
                        != expected_track
                        or int(
                            spatial_payload["candidate_prototype_ids"][spatial_row]
                        )
                        != expected_prototype
                    ):
                        raise ValueError("spatial likelihood candidate identity differs")
                    if not np.allclose(
                        spatial_payload["center_xy"][spatial_row],
                        query_xy[compact_row],
                        rtol=0.0,
                        atol=1e-4,
                    ):
                        raise ValueError("spatial likelihood center differs from query xy")
                    if spatial_valid[local_row, column, view_rank]:
                        raise ValueError("duplicate spatial likelihood candidate/view slot")
                    log_maps[local_row, column, view_rank] = spatial_payload[
                        "local_log_probabilities"
                    ][spatial_row]
                    support_priors = np.asarray(
                        candidate_evidence[
                            "candidate_support_view_probabilities"
                        ][compact_row, candidate_rank],
                        dtype=np.float64,
                    )
                    expected_view_probability = (
                        0.0
                        if not 0 <= view_rank < len(support_priors)
                        else float(support_priors[view_rank])
                    )
                    if (
                        not np.isfinite(expected_view_probability)
                        or expected_view_probability < 0.0
                    ):
                        raise ValueError("candidate evidence support prior is invalid")
                    observed_view_probability = float(
                        spatial_payload["support_view_probabilities"][spatial_row]
                    )
                    learned_artifact_probability = (
                        support_probability_semantics
                        == POSE_VIEW_MIXTURE_SEMANTICS
                    )
                    if learned_artifact_probability and (
                        not np.isfinite(observed_view_probability)
                        or not 0.0 <= observed_view_probability <= 1.0
                    ):
                        raise ValueError(
                            "spatial likelihood support probability is invalid"
                        )
                    if support_probability_semantics == (
                        "frozen_candidate_maplet_view_posterior"
                    ) and (
                        0 <= view_rank < len(support_priors)
                        and np.isfinite(observed_view_probability)
                        and not np.isclose(
                            observed_view_probability,
                            expected_view_probability,
                            rtol=0.0,
                            atol=2e-5,
                        )
                    ):
                        raise ValueError(
                            "spatial likelihood support prior differs from evidence"
                        )
                    view_probabilities[
                        local_row, column, view_rank
                    ] = (
                        observed_view_probability
                        if str(spatial_view_mixture_policy)
                        == "artifact_pose_view_posterior"
                        else expected_view_probability
                    )
                    dustbin_probabilities[local_row, column, view_rank] = float(
                        spatial_payload["dustbin_probabilities"][spatial_row]
                    )
                    if support_camera_centers is not None:
                        support_image_id = str(
                            spatial_payload["support_image_ids"][spatial_row]
                        )
                        support_camera_center = camera_centers_by_name.get(
                            support_image_id
                        )
                        if support_camera_center is None:
                            raise ValueError(
                                "spatial support image is absent from the mapping model"
                            )
                        support_camera_centers[
                            local_row, column, view_rank
                        ] = support_camera_center
                    spatial_valid[local_row, column, view_rank] = True
                if not np.any(spatial_valid):
                    raise ValueError(
                        f"query {query_id} produced no valid spatial likelihood slots"
                    )
                view_probabilities = _resolve_spatial_view_mixture_probabilities(
                    view_probabilities,
                    spatial_valid,
                    policy=str(spatial_view_mixture_policy),
                )
                spatial_likelihood = CandidateSpatialLikelihood(
                    offsets_xy=offsets,
                    local_log_probabilities=log_maps,
                    view_probabilities=view_probabilities,
                    dustbin_probabilities=dustbin_probabilities,
                    valid_mask=spatial_valid,
                    log_evidence_weight=float(
                        args.candidate_spatial_log_evidence_weight
                    ),
                    support_camera_centers=support_camera_centers,
                    pose_view_geometry_sigma_deg=float(
                        pose_view_geometry_sigma_deg
                    ),
                )
            output[str(query_id)] = PoseVerificationCandidatePool(
                token_indices=selected_rows[rows],
                xy=query_xy[rows],
                track_ids=compact_tracks[rows],
                prototype_ids=compact_prototypes[rows],
                xyz=xyz,
                descriptor_scores=scores[rows],
                valid_mask=local_valid,
                measurement_geometry_probabilities=(
                    None
                    if geometry_probabilities is None
                    else geometry_probabilities[rows]
                ),
                null_scores=local_null,
                spatial_likelihood=spatial_likelihood,
                identity_prior_temperature=float(
                    args.candidate_identity_prior_temperature
                ),
                geometry_prior_mix_weight=float(
                    args.candidate_geometry_prior_mix_weight
                ),
                spatial_geometry_calibration_weight=float(
                    args.candidate_spatial_geometry_calibration_weight
                ),
                geometry_generation_mix_weight=float(
                    args.candidate_geometry_generation_mix_weight
                ),
                candidate_update_probabilities=(
                    None if update_probabilities is None else update_probabilities[rows]
                ),
                candidate_refined_xy=(
                    None if update_refined_xy is None else update_refined_xy[rows]
                ),
                maplet_cluster_ids=(
                    None
                    if maplet_cluster_ids_by_bank_row is None
                    else np.where(
                        local_valid,
                        maplet_cluster_ids_by_bank_row[
                            np.maximum(canonical_rows[rows], 0)
                        ],
                        -1,
                    )
                ),
                topology_neighbor_track_ids=(
                    None
                    if maplet_index is None
                    else np.where(
                        local_valid[:, :, None],
                        maplet_index.neighbor_track_ids[
                            np.maximum(canonical_rows[rows], 0)
                        ],
                        -1,
                    )
                ),
                topology_support_image_indices=(
                    None
                    if maplet_index is None
                    else np.where(
                        local_valid[:, :, None],
                        maplet_index.support_image_indices[
                            np.maximum(canonical_rows[rows], 0)
                        ],
                        -1,
                    )
                ),
                topology_support_coverage_counts=(
                    None
                    if maplet_index is None
                    else np.where(
                        local_valid[:, :, None],
                        maplet_index.support_coverage_counts[
                            np.maximum(canonical_rows[rows], 0)
                        ],
                        0,
                    )
                ),
                candidate_update_threshold=float(update_threshold),
                spatial_utility_gate_weight=float(spatial_utility_gate_weight),
            )
        return output

    def matched_single_ransac(matches, camera, query_id):
        selected_matches = select_pose_safe_matches(
            matches,
            max_matches=int(args.single_ransac_match_count),
            image_width=int(camera.width),
            image_height=int(camera.height),
            mode=str(args.single_ransac_selection_mode),
        )
        selected_matches = stable_uniform_ransac_order(selected_matches)
        _set_cv2_seed(_query_seed(query_id))
        return estimate_pose_pnp_ransac(
            selected_matches,
            camera,
            reprojection_error_px=float(args.single_ransac_threshold_px),
            iterations=int(args.single_ransac_iterations),
            refine_method="LM",
        )

    def append_selected_pose_export(
        result,
        *,
        query_id: str,
        split_name: str,
        evaluation_label: str,
    ) -> None:
        if not bool(args.export_selected_pose_artifact):
            return
        selected_pose_export_rows.append(
            {
                "query_id": str(query_id),
                "split_name": str(split_name),
                "evaluation_label": str(evaluation_label),
                "success": bool(result.success),
                "pose_w2c": (
                    None
                    if result.pose_w2c is None
                    else np.asarray(result.pose_w2c, dtype=np.float64).reshape(4, 4)
                ),
                "match_count": int(result.match_count),
                "inlier_count": int(result.inlier_count),
            }
        )

    def append_grouped_hypothesis_export(
        result,
        *,
        query_id: str,
        split_name: str,
        evaluation_label: str,
    ) -> None:
        if not bool(args.export_grouped_hypothesis_artifact):
            return
        chosen_index = result.chosen_hypothesis_index
        for hypothesis_index, (record, pose_w2c) in enumerate(
            zip(result.hypotheses, result.hypothesis_poses_w2c)
        ):
            if pose_w2c is None:
                continue
            verification = record.verification
            sample_tokens = np.full((8,), -1, dtype=np.int64)
            sample_tracks = np.full((8,), -1, dtype=np.int64)
            sample_count = min(len(record.sample_token_indices), 8)
            sample_tokens[:sample_count] = np.asarray(
                record.sample_token_indices[:sample_count], dtype=np.int64
            )
            sample_tracks[:sample_count] = np.asarray(
                record.sample_track_ids[:sample_count], dtype=np.int64
            )
            grouped_hypothesis_export_rows.append(
                {
                    "query_id": str(query_id),
                    "split_name": str(split_name),
                    "evaluation_label": str(evaluation_label),
                    "hypothesis_index": int(hypothesis_index),
                    "pose_w2c": np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
                    "generation_profile": str(record.generation_profile),
                    "selection_mode": str(record.selection_mode),
                    "latent_em_applied": bool(record.latent_em_applied),
                    "latent_em_parent_hypothesis_index": (
                        -1
                        if record.latent_em_parent_hypothesis_index is None
                        else int(record.latent_em_parent_hypothesis_index)
                    ),
                    "latent_em_seed_evidence_mode": (
                        ""
                        if record.latent_em_seed_evidence_mode is None
                        else str(record.latent_em_seed_evidence_mode)
                    ),
                    "local_optimization_applied": bool(
                        record.local_optimization_applied
                    ),
                    "sample_count": int(sample_count),
                    "sample_token_indices": sample_tokens,
                    "sample_track_ids": sample_tracks,
                    "sampling_log_probability": (
                        np.nan
                        if record.sampling_log_probability is None
                        else float(record.sampling_log_probability)
                    ),
                    "preliminary_log_likelihood_mean": (
                        np.nan
                        if record.preliminary_log_likelihood_mean is None
                        else float(record.preliminary_log_likelihood_mean)
                    ),
                    "shortlist_log_likelihood_mean": (
                        np.nan
                        if record.shortlist_log_likelihood_mean is None
                        else float(record.shortlist_log_likelihood_mean)
                    ),
                    "shortlist_evidence_mode": (
                        ""
                        if record.shortlist_evidence_mode is None
                        else str(record.shortlist_evidence_mode)
                    ),
                    "shortlisted_for_verification": verification is not None,
                    "verification_log_likelihood_mean": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_log_likelihood_mean is None
                        else float(
                            verification.fixed_posterior_log_likelihood_mean
                        )
                    ),
                    "verification_log_likelihood_std": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_log_likelihood_std is None
                        else float(verification.fixed_posterior_log_likelihood_std)
                    ),
                    "verification_log_likelihood_standard_error": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_log_likelihood_standard_error
                        is None
                        else float(
                            verification.fixed_posterior_log_likelihood_standard_error
                        )
                    ),
                    "verification_log_likelihood_median": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_log_likelihood_median is None
                        else float(
                            verification.fixed_posterior_log_likelihood_median
                        )
                    ),
                    "verification_log_likelihood_trimmed_mean_10": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_log_likelihood_trimmed_mean_10
                        is None
                        else float(
                            verification.fixed_posterior_log_likelihood_trimmed_mean_10
                        )
                    ),
                    "verification_log_likelihood_worst_quartile_mean": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_log_likelihood_worst_quartile_mean
                        is None
                        else float(
                            verification.fixed_posterior_log_likelihood_worst_quartile_mean
                        )
                    ),
                    "verification_log_likelihood_lcb95": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_log_likelihood_lcb95 is None
                        else float(
                            verification.fixed_posterior_log_likelihood_lcb95
                        )
                    ),
                    "verification_spatial_median_of_means_2x2": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_spatial_median_of_means_2x2
                        is None
                        else float(
                            verification.fixed_posterior_spatial_median_of_means_2x2
                        )
                    ),
                    "verification_spatial_mom_cell_count": (
                        0
                        if verification is None
                        else int(
                            verification.fixed_posterior_spatial_mom_cell_count
                        )
                    ),
                    "verification_effective_group_count": (
                        0
                        if verification is None
                        else int(
                            verification.fixed_posterior_effective_group_count
                        )
                    ),
                    "verification_identity_prior_temperature": (
                        np.nan
                        if verification is None
                        else float(
                            verification.fixed_posterior_identity_prior_temperature
                        )
                    ),
                    "verification_null_evidence_fraction_mean": (
                        np.nan
                        if verification is None
                        else verification.fixed_posterior_null_evidence_fraction_mean
                    ),
                    "verification_candidate_inlier_evidence_fraction_mean": (
                        np.nan
                        if verification is None
                        else verification.fixed_posterior_candidate_inlier_evidence_fraction_mean
                    ),
                    "verification_spatial_log_likelihood_gain_mean": (
                        np.nan
                        if verification is None
                        else verification.fixed_posterior_spatial_log_likelihood_gain_mean
                    ),
                    "verification_relation_pair_count": (
                        0
                        if verification is None
                        else int(
                            verification.fixed_posterior_relation_pair_count
                        )
                    ),
                    "verification_relation_effective_pair_count": (
                        0
                        if verification is None
                        else int(
                            verification.fixed_posterior_relation_effective_pair_count
                        )
                    ),
                    "verification_relation_log_likelihood_ratio_mean": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_relation_log_likelihood_ratio_mean
                        is None
                        else float(
                            verification.fixed_posterior_relation_log_likelihood_ratio_mean
                        )
                    ),
                    "verification_relation_log_likelihood_ratio_sum": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_relation_log_likelihood_ratio_sum
                        is None
                        else float(
                            verification.fixed_posterior_relation_log_likelihood_ratio_sum
                        )
                    ),
                    "verification_relation_feature_edge_count": (
                        0
                        if verification is None
                        else int(
                            verification.fixed_posterior_relation_feature_edge_count
                        )
                    ),
                    "verification_relation_feature_candidate_pair_mass_mean": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_relation_feature_candidate_pair_mass_mean
                        is None
                        else float(
                            verification.fixed_posterior_relation_feature_candidate_pair_mass_mean
                        )
                    ),
                    "verification_relation_feature_null_touching_mass_mean": (
                        np.nan
                        if verification is None
                        or verification.fixed_posterior_relation_feature_null_touching_mass_mean
                        is None
                        else float(
                            verification.fixed_posterior_relation_feature_null_touching_mass_mean
                        )
                    ),
                    "verification_relation_feature_histogram": (
                        (np.nan,)
                        * (
                            len(RELATION_CHANNELS)
                            * (
                                (
                                    len((0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1e6))
                                    - 1
                                )
                                if optional_grouped_config is None
                                else (
                                    len(optional_grouped_config.candidate_relation_feature_bin_edges_px)
                                    - 1
                                )
                            )
                        )
                        if verification is None
                        or not verification.fixed_posterior_relation_feature_histogram
                        else tuple(
                            verification.fixed_posterior_relation_feature_histogram
                        )
                    ),
                    "verification_relation_feature_edge_histograms": (
                        ()
                        if verification is None
                        else verification.fixed_posterior_relation_feature_edge_histograms
                    ),
                    "verification_relation_feature_edge_null_touching_masses": (
                        ()
                        if verification is None
                        else verification.fixed_posterior_relation_feature_edge_null_touching_masses
                    ),
                    "verification_relation_feature_edge_graph_sha256": (
                        ""
                        if verification is None
                        or verification.fixed_posterior_relation_feature_edge_graph_sha256
                        is None
                        else str(
                            verification.fixed_posterior_relation_feature_edge_graph_sha256
                        )
                    ),
                    "verification_positive_depth_ratio": (
                        np.nan
                        if verification is None
                        else float(verification.positive_depth_ratio)
                    ),
                    "verification_strict_inlier_count": (
                        0
                        if verification is None
                        else int(verification.strict_inlier_count)
                    ),
                    "verification_loose_inlier_count": (
                        0
                        if verification is None
                        else int(verification.loose_inlier_count)
                    ),
                    "verification_strict_grid_cell_count": (
                        0
                        if verification is None
                        else int(verification.strict_grid_cell_count)
                    ),
                    "verification_loose_grid_cell_count": (
                        0
                        if verification is None
                        else int(verification.loose_grid_cell_count)
                    ),
                    "verification_soft_consensus": (
                        np.nan
                        if verification is None
                        else float(verification.soft_consensus)
                    ),
                    "verification_median_residual_px": (
                        np.nan
                        if verification is None
                        else float(verification.clipped_median_residual_px)
                    ),
                    "translation_information_min_eigenvalue": (
                        np.nan
                        if verification is None
                        or verification.translation_information_min_eigenvalue
                        is None
                        else float(
                            verification.translation_information_min_eigenvalue
                        )
                    ),
                    "translation_information_condition": (
                        np.nan
                        if verification is None
                        or verification.translation_information_condition is None
                        else float(verification.translation_information_condition)
                    ),
                    "rotation_information_min_eigenvalue": (
                        np.nan
                        if verification is None
                        or verification.rotation_information_min_eigenvalue is None
                        else float(
                            verification.rotation_information_min_eigenvalue
                        )
                    ),
                    "joint_information_condition": (
                        np.nan
                        if verification is None
                        or verification.joint_information_condition is None
                        else float(verification.joint_information_condition)
                    ),
                    "bearing_max_angle_deg": (
                        np.nan
                        if verification is None
                        or verification.bearing_max_angle_deg is None
                        else float(verification.bearing_max_angle_deg)
                    ),
                    "camera_depth_span_ratio": (
                        np.nan
                        if verification is None
                        or verification.camera_depth_span_ratio is None
                        else float(verification.camera_depth_span_ratio)
                    ),
                    "xyz_second_singular_ratio": (
                        np.nan
                        if verification is None
                        or verification.xyz_second_singular_ratio is None
                        else float(verification.xyz_second_singular_ratio)
                    ),
                    "xyz_third_singular_ratio": (
                        np.nan
                        if verification is None
                        or verification.xyz_third_singular_ratio is None
                        else float(verification.xyz_third_singular_ratio)
                    ),
                    "chosen_for_optional_pose": bool(
                        chosen_index is not None
                        and int(hypothesis_index) == int(chosen_index)
                    ),
                }
            )

    def evaluate(
        scores: np.ndarray,
        split_name: str,
        backend: str,
        *,
        candidate_pool_scores: np.ndarray | None = None,
        candidate_pool_null_scores: np.ndarray | None = None,
        candidate_pool_spatial_payload: dict[str, np.ndarray] | None = None,
        candidate_pool_generation_spatial_payload: dict[str, np.ndarray]
        | None = None,
        candidate_pool_geometry_probabilities: np.ndarray | None = None,
        candidate_pool_update_probabilities: np.ndarray | None = None,
        candidate_pool_update_refined_xy: np.ndarray | None = None,
        candidate_pool_update_threshold: float = 0.5,
        evaluation_label: str = "",
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        grouped = matches_by_query(scores, split_name)
        grouped_candidate_pools = (
            {}
            if candidate_pool_scores is None
            else candidate_pools_by_query(
                candidate_pool_scores,
                split_name,
                null_scores=candidate_pool_null_scores,
                spatial_payload=candidate_pool_spatial_payload,
                geometry_probabilities=candidate_pool_geometry_probabilities,
                update_probabilities=candidate_pool_update_probabilities,
                update_refined_xy=candidate_pool_update_refined_xy,
                update_threshold=float(candidate_pool_update_threshold),
                spatial_utility_gate_weight=float(
                    args.candidate_spatial_utility_gate_weight
                ),
                spatial_view_mixture_policy=str(
                    args.candidate_spatial_view_mixture_policy
                ),
                pose_view_geometry_sigma_deg=float(
                    args.candidate_spatial_pose_view_geometry_sigma_deg
                ),
            )
        )
        grouped_generation_candidate_pools = (
            {}
            if candidate_pool_generation_spatial_payload is None
            else candidate_pools_by_query(
                candidate_pool_scores,
                split_name,
                null_scores=candidate_pool_null_scores,
                spatial_payload=candidate_pool_generation_spatial_payload,
                geometry_probabilities=candidate_pool_geometry_probabilities,
                update_probabilities=candidate_pool_update_probabilities,
                update_refined_xy=candidate_pool_update_refined_xy,
                update_threshold=float(candidate_pool_update_threshold),
                spatial_utility_gate_weight=float(
                    args.candidate_spatial_utility_gate_weight
                ),
                spatial_view_mixture_policy=str(
                    args.candidate_generation_spatial_view_mixture_policy
                ),
                pose_view_geometry_sigma_deg=float(
                    args.candidate_generation_spatial_pose_view_geometry_sigma_deg
                ),
            )
        )
        expected_ids = [str(value) for value in execution_split[split_name]]
        rows: list[dict[str, object]] = []
        for query_id in expected_ids:
            image = images_by_name.get(query_id)
            if image is None:
                rows.append(
                    {
                        "query_id": query_id,
                        "success": False,
                        "failure_reason": "missing_colmap_image",
                        "match_count": 0,
                        "inlier_count": 0,
                    }
                )
                continue
            camera = cameras[int(image.camera_id)]
            matches = grouped.get(query_id, [])
            verified_result = None
            crossfit_promotion_audit = None
            promotion_baseline_result = None
            promotion_optional_result = None
            if backend == "grouped_candidate_pool":
                pool = grouped_candidate_pools.get(query_id)
                if pool is None:
                    raise ValueError("grouped candidate PnP requires a candidate pool")
                if grouped_config is None or optional_grouped_config is None:
                    raise RuntimeError("grouped candidate PnP configuration is missing")
                optional_result = estimate_pose_from_grouped_candidate_pool(
                    pool,
                    camera,
                    config=optional_grouped_config,
                    query_seed=_query_seed(query_id),
                    generation_candidate_pool=(
                        grouped_generation_candidate_pools.get(query_id)
                    ),
                )
                promotion_optional_result = optional_result
                append_grouped_hypothesis_export(
                    optional_result,
                    query_id=query_id,
                    split_name=split_name,
                    evaluation_label=evaluation_label,
                )
                generation_promotion_audit = None
                if bool(args.enable_grouped_crossfit_likelihood_fallback) and (
                    split_name != "train"
                ):
                    if immutable_grouped_config is None:
                        raise RuntimeError("immutable grouped baseline is missing")
                    frozen_baseline_result = (
                        matched_single_ransac(matches, camera, query_id)
                        if immutable_baseline_pose_source is None
                        else immutable_baseline_pose_source["records"][
                            (split_name, query_id)
                        ]
                    )
                    baseline_result = wrap_immutable_pose_on_grouped_denominator(
                        frozen_baseline_result,
                        pool,
                        camera,
                        config=immutable_grouped_config,
                        query_seed=_query_seed(query_id),
                    )
                    promotion_baseline_result = baseline_result
                    verified_result, crossfit_promotion_audit = (
                        select_crossfit_likelihood_with_immutable_baseline(
                            baseline_result,
                            optional_result,
                            min_log_likelihood_mean_delta=float(
                                args.grouped_likelihood_min_mean_delta
                            ),
                            min_effective_group_count=int(
                                args.grouped_likelihood_min_effective_groups
                            ),
                            min_information_match_count=int(
                                args.grouped_observability_min_information_matches
                            ),
                            min_translation_information_eigenvalue=float(
                                args.grouped_observability_min_translation_eigenvalue
                            ),
                            max_translation_information_condition=float(
                                args.grouped_observability_max_translation_condition
                            ),
                            max_joint_information_condition=float(
                                args.grouped_observability_max_joint_condition
                            ),
                            min_bearing_span_deg=float(
                                args.grouped_observability_min_bearing_span_deg
                            ),
                            min_depth_span_ratio=float(
                                args.grouped_observability_min_depth_span_ratio
                            ),
                            min_xyz_second_singular_ratio=float(
                                args.grouped_observability_min_xyz_second_ratio
                            ),
                            min_xyz_third_singular_ratio=float(
                                args.grouped_observability_min_xyz_third_ratio
                            ),
                        )
                    )
                elif bool(args.enable_grouped_generation_grid_fallback):
                    baseline_result = estimate_pose_from_grouped_candidate_pool(
                        pool.with_geometry_generation_mix_weight(0.0),
                        camera,
                        config=grouped_config,
                        query_seed=_query_seed(query_id),
                        generation_candidate_pool=(
                            None
                            if query_id not in grouped_generation_candidate_pools
                            else grouped_generation_candidate_pools[
                                query_id
                            ].with_geometry_generation_mix_weight(0.0)
                        ),
                    )
                    verified_result, generation_promotion_audit = (
                        select_geometry_guided_generation_with_immutable_baseline(
                            baseline_result,
                            optional_result,
                            min_strict_grid_cell_delta=int(
                                args.grouped_generation_min_strict_grid_delta
                            ),
                        )
                    )
                    promotion_baseline_result = baseline_result
                else:
                    verified_result = optional_result
                pose = verified_result.pose_w2c
                solver_success = bool(verified_result.success)
                match_count = int(verified_result.match_count)
                inlier_count = int(verified_result.inlier_count)
                backend_summary = verified_result.summary()
                if generation_promotion_audit is not None:
                    backend_summary["geometry_guided_generation_promotion"] = (
                        generation_promotion_audit
                    )
                if crossfit_promotion_audit is not None:
                    backend_summary["crossfit_likelihood_promotion"] = (
                        crossfit_promotion_audit
                    )
                    backend_summary["crossfit_immutable_baseline"] = (
                        promotion_baseline_result.summary()
                        if promotion_baseline_result is not None
                        else None
                    )
                    backend_summary["crossfit_optional_generation"] = (
                        optional_result.summary()
                    )
                append_selected_pose_export(
                    verified_result,
                    query_id=query_id,
                    split_name=split_name,
                    evaluation_label=evaluation_label,
                )
            elif backend == "verified":
                verified_result = estimate_pose_with_heldout_verification(
                    matches,
                    camera,
                    config=config,
                    query_seed=_query_seed(query_id),
                    candidate_pool=grouped_candidate_pools.get(query_id),
                )
                pose = verified_result.pose_w2c
                solver_success = bool(verified_result.success)
                match_count = int(verified_result.match_count)
                inlier_count = int(verified_result.inlier_count)
                backend_summary = verified_result.summary()
            elif backend == "single_ransac":
                result = matched_single_ransac(matches, camera, query_id)
                pose = result.pose_w2c
                solver_success = bool(result.success)
                match_count = int(result.match_count)
                inlier_count = int(result.inlier_count)
                backend_summary = None
            else:
                raise ValueError(f"unsupported backend: {backend}")
            gt_pose = np.eye(4, dtype=np.float64)
            gt_pose[:3, :3] = qvec_to_rotmat(image.qvec)
            gt_pose[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
            promotion_baseline_error = (
                None
                if promotion_baseline_result is None
                else pnp_pose_error(promotion_baseline_result.pose_w2c, gt_pose)
            )
            promotion_optional_error = (
                None
                if promotion_optional_result is None
                else pnp_pose_error(promotion_optional_result.pose_w2c, gt_pose)
            )
            error = pnp_pose_error(pose, gt_pose)
            pre_refine_error = None
            oracle_error = None
            chosen_translation_rank = None
            valid_hypothesis_count = 0
            hypothesis_3cm_count = 0
            hypothesis_10cm_count = 0
            hypothesis_25cm_count = 0
            catastrophic_hypothesis_count = 0
            raw_hypothesis_oracle_translation_m = None
            raw_hypothesis_oracle_rotation_deg = None
            latent_em_hypothesis_oracle_translation_m = None
            latent_em_hypothesis_oracle_rotation_deg = None
            latent_em_hypothesis_count = 0
            latent_em_parent_translation_deltas: list[float] = []
            latent_em_seed_parent_count = 0
            latent_em_seed_parent_oracle_translation_m = None
            latent_em_seed_parent_oracle_rotation_deg = None
            latent_em_seed_parent_raw_rank = None
            latent_em_refined_minus_seed_oracle_translation_m = None
            verified_hypothesis_oracle_translation_m = None
            verified_hypothesis_oracle_rotation_deg = None
            verified_hypothesis_count = 0
            spatial_rescored_hypothesis_oracle_translation_m = None
            spatial_rescored_hypothesis_oracle_rotation_deg = None
            spatial_rescored_hypothesis_count = 0
            hypothesis_information_audit: list[dict[str, object]] = []
            grouped_minimal_sample_count = 0
            grouped_minimal_sample_pair_count = 0
            grouped_minimal_sample_correct_2px_pair_count = 0
            grouped_minimal_sample_correct_5px_pair_count = 0
            grouped_minimal_sample_all_correct_2px_count = 0
            grouped_minimal_sample_all_correct_5px_count = 0
            grouped_observability_rejected_hypothesis_count = 0
            hypothesis_profile_audit: dict[str, dict[str, object]] = {}
            if verified_result is not None:
                pre_refine_error = pnp_pose_error(
                    verified_result.pre_refine_pose_w2c, gt_pose
                )
                hypothesis_audit_result = (
                    promotion_optional_result
                    if promotion_optional_result is not None
                    else verified_result
                )
                grouped_observability_rejected_hypothesis_count = int(
                    np.sum(
                        [
                            bool(record.observability_gate_failures)
                            for record in hypothesis_audit_result.hypotheses
                        ]
                    )
                )
                hypothesis_errors = [
                    pnp_pose_error(hypothesis_pose, gt_pose)
                    for hypothesis_pose in hypothesis_audit_result.hypothesis_poses_w2c
                    if hypothesis_pose is not None
                ]
                finite_hypothesis_errors = [
                    value
                    for value in hypothesis_errors
                    if np.isfinite(value.translation_m)
                    and np.isfinite(value.rotation_deg)
                ]
                raw_hypothesis_errors = []
                raw_hypothesis_errors_by_index: dict[int, object] = {}
                latent_em_hypothesis_errors = []
                latent_em_parent_errors_by_index: dict[int, object] = {}
                verified_hypothesis_errors = []
                spatial_rescored_hypothesis_errors = []
                profile_hypothesis_errors: dict[
                    str, list[tuple[object, bool, bool]]
                ] = {}
                for record_index, (record, hypothesis_pose) in enumerate(
                    zip(
                        hypothesis_audit_result.hypotheses,
                        hypothesis_audit_result.hypothesis_poses_w2c,
                    )
                ):
                    if hypothesis_pose is None:
                        continue
                    source_error = pnp_pose_error(hypothesis_pose, gt_pose)
                    if not (
                        np.isfinite(source_error.translation_m)
                        and np.isfinite(source_error.rotation_deg)
                    ):
                        continue
                    if record.latent_em_applied:
                        latent_em_hypothesis_errors.append(source_error)
                        parent_index = record.latent_em_parent_hypothesis_index
                        if parent_index is not None:
                            if not 0 <= int(parent_index) < len(
                                hypothesis_audit_result.hypothesis_poses_w2c
                            ):
                                raise RuntimeError(
                                    "latent EM parent hypothesis index is invalid"
                                )
                            parent_pose = hypothesis_audit_result.hypothesis_poses_w2c[
                                int(parent_index)
                            ]
                            if parent_pose is not None:
                                parent_error = pnp_pose_error(parent_pose, gt_pose)
                                if np.isfinite(
                                    parent_error.translation_m
                                ) and np.isfinite(parent_error.rotation_deg):
                                    latent_em_parent_errors_by_index[
                                        int(parent_index)
                                    ] = parent_error
                                    latent_em_parent_translation_deltas.append(
                                        float(source_error.translation_m)
                                        - float(parent_error.translation_m)
                                    )
                    else:
                        raw_hypothesis_errors.append(source_error)
                        raw_hypothesis_errors_by_index[int(record_index)] = (
                            source_error
                        )
                    if record.verification is not None:
                        verified_hypothesis_errors.append(source_error)
                    if record.spatial_rescored_for_shortlist:
                        spatial_rescored_hypothesis_errors.append(source_error)
                    profile_hypothesis_errors.setdefault(
                        str(record.generation_profile), []
                    ).append(
                        (
                            source_error,
                            bool(record.latent_em_applied),
                            record.verification is not None,
                        )
                    )
                latent_em_hypothesis_count = len(latent_em_hypothesis_errors)
                verified_hypothesis_count = len(verified_hypothesis_errors)
                spatial_rescored_hypothesis_count = len(
                    spatial_rescored_hypothesis_errors
                )
                if raw_hypothesis_errors:
                    raw_oracle = min(
                        raw_hypothesis_errors,
                        key=lambda value: (
                            float(value.translation_m),
                            float(value.rotation_deg),
                        ),
                    )
                    raw_hypothesis_oracle_translation_m = float(
                        raw_oracle.translation_m
                    )
                    raw_hypothesis_oracle_rotation_deg = float(
                        raw_oracle.rotation_deg
                    )
                if latent_em_hypothesis_errors:
                    latent_oracle = min(
                        latent_em_hypothesis_errors,
                        key=lambda value: (
                            float(value.translation_m),
                            float(value.rotation_deg),
                        ),
                    )
                    latent_em_hypothesis_oracle_translation_m = float(
                        latent_oracle.translation_m
                    )
                    latent_em_hypothesis_oracle_rotation_deg = float(
                        latent_oracle.rotation_deg
                    )
                latent_em_parent_errors_by_index = {
                    int(index): error
                    for index, error in latent_em_parent_errors_by_index.items()
                    if int(index) in raw_hypothesis_errors_by_index
                }
                if latent_em_parent_errors_by_index:
                    latent_em_seed_parent_count = int(
                        len(latent_em_parent_errors_by_index)
                    )
                    seed_parent_index, seed_parent_oracle = min(
                        latent_em_parent_errors_by_index.items(),
                        key=lambda item: (
                            float(item[1].translation_m),
                            float(item[1].rotation_deg),
                        ),
                    )
                    latent_em_seed_parent_oracle_translation_m = float(
                        seed_parent_oracle.translation_m
                    )
                    latent_em_seed_parent_oracle_rotation_deg = float(
                        seed_parent_oracle.rotation_deg
                    )
                    raw_order = sorted(
                        raw_hypothesis_errors_by_index,
                        key=lambda index: (
                            float(
                                raw_hypothesis_errors_by_index[
                                    index
                                ].translation_m
                            ),
                            float(
                                raw_hypothesis_errors_by_index[index].rotation_deg
                            ),
                            int(index),
                        ),
                    )
                    latent_em_seed_parent_raw_rank = int(
                        raw_order.index(int(seed_parent_index)) + 1
                    )
                    if latent_em_hypothesis_oracle_translation_m is not None:
                        latent_em_refined_minus_seed_oracle_translation_m = float(
                            latent_em_hypothesis_oracle_translation_m
                            - latent_em_seed_parent_oracle_translation_m
                        )
                if verified_hypothesis_errors:
                    verified_oracle = min(
                        verified_hypothesis_errors,
                        key=lambda value: (
                            float(value.translation_m),
                            float(value.rotation_deg),
                        ),
                    )
                    verified_hypothesis_oracle_translation_m = float(
                        verified_oracle.translation_m
                    )
                    verified_hypothesis_oracle_rotation_deg = float(
                        verified_oracle.rotation_deg
                    )
                if spatial_rescored_hypothesis_errors:
                    spatial_rescored_oracle = min(
                        spatial_rescored_hypothesis_errors,
                        key=lambda value: (
                            float(value.translation_m),
                            float(value.rotation_deg),
                        ),
                    )
                    spatial_rescored_hypothesis_oracle_translation_m = float(
                        spatial_rescored_oracle.translation_m
                    )
                    spatial_rescored_hypothesis_oracle_rotation_deg = float(
                        spatial_rescored_oracle.rotation_deg
                    )
                if finite_hypothesis_errors:
                    valid_hypothesis_count = int(len(finite_hypothesis_errors))
                    hypothesis_3cm_count = int(
                        np.sum(
                            [
                                value.translation_m <= 0.03
                                and value.rotation_deg <= 5.0
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                    hypothesis_10cm_count = int(
                        np.sum(
                            [
                                value.translation_m <= 0.10
                                and value.rotation_deg <= 5.0
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                    hypothesis_25cm_count = int(
                        np.sum(
                            [
                                value.translation_m <= 0.25
                                and value.rotation_deg <= 2.0
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                    catastrophic_hypothesis_count = int(
                        np.sum(
                            [
                                value.translation_m > 1.0
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                    oracle_error = min(
                        finite_hypothesis_errors,
                        key=lambda value: (
                            float(value.translation_m), float(value.rotation_deg)
                        ),
                    )
                    chosen_translation_rank = 1 + int(
                        np.sum(
                            [
                                float(value.translation_m)
                                < float(pre_refine_error.translation_m)
                                for value in finite_hypothesis_errors
                            ]
                        )
                    )
                for profile_name, profile_values in sorted(
                    profile_hypothesis_errors.items()
                ):
                    errors = [item[0] for item in profile_values]
                    oracle = min(
                        errors,
                        key=lambda value: (
                            float(value.translation_m),
                            float(value.rotation_deg),
                        ),
                    )
                    hypothesis_profile_audit[profile_name] = {
                        "valid_hypothesis_count": int(len(errors)),
                        "raw_hypothesis_count": int(
                            np.sum([not item[1] for item in profile_values])
                        ),
                        "latent_em_hypothesis_count": int(
                            np.sum([item[1] for item in profile_values])
                        ),
                        "verified_hypothesis_count": int(
                            np.sum([item[2] for item in profile_values])
                        ),
                        "oracle_translation_m": float(oracle.translation_m),
                        "oracle_rotation_deg": float(oracle.rotation_deg),
                        "oracle_3cm_5deg": bool(
                            oracle.translation_m <= 0.03
                            and oracle.rotation_deg <= 5.0
                        ),
                        "oracle_5cm_5deg": bool(
                            oracle.translation_m <= 0.05
                            and oracle.rotation_deg <= 5.0
                        ),
                        "oracle_10cm_5deg": bool(
                            oracle.translation_m <= 0.10
                            and oracle.rotation_deg <= 5.0
                        ),
                        "oracle_25cm_2deg": bool(
                            oracle.translation_m <= 0.25
                            and oracle.rotation_deg <= 2.0
                        ),
                        "hypothesis_3cm_5deg_count": int(
                            np.sum(
                                [
                                    value.translation_m <= 0.03
                                    and value.rotation_deg <= 5.0
                                    for value in errors
                                ]
                            )
                        ),
                        "hypothesis_10cm_5deg_count": int(
                            np.sum(
                                [
                                    value.translation_m <= 0.10
                                    and value.rotation_deg <= 5.0
                                    for value in errors
                                ]
                            )
                        ),
                        "hypothesis_25cm_2deg_count": int(
                            np.sum(
                                [
                                    value.translation_m <= 0.25
                                    and value.rotation_deg <= 2.0
                                    for value in errors
                                ]
                            )
                        ),
                        "catastrophic_hypothesis_count": int(
                            np.sum(
                                [value.translation_m > 1.0 for value in errors]
                            )
                        ),
                        "minimal_sample_count": 0,
                        "minimal_sample_pair_count": 0,
                        "minimal_sample_correct_2px_pair_count": 0,
                        "minimal_sample_correct_5px_pair_count": 0,
                        "minimal_sample_all_correct_2px_count": 0,
                        "minimal_sample_all_correct_5px_count": 0,
                    }
                    verified_profile_errors = [
                        item[0] for item in profile_values if item[2]
                    ]
                    if verified_profile_errors:
                        verified_profile_oracle = min(
                            verified_profile_errors,
                            key=lambda value: (
                                float(value.translation_m),
                                float(value.rotation_deg),
                            ),
                        )
                        hypothesis_profile_audit[profile_name].update(
                            {
                                "verified_oracle_translation_m": float(
                                    verified_profile_oracle.translation_m
                                ),
                                "verified_oracle_rotation_deg": float(
                                    verified_profile_oracle.rotation_deg
                                ),
                                "verified_oracle_3cm_5deg": bool(
                                    verified_profile_oracle.translation_m <= 0.03
                                    and verified_profile_oracle.rotation_deg <= 5.0
                                ),
                                "verified_oracle_5cm_5deg": bool(
                                    verified_profile_oracle.translation_m <= 0.05
                                    and verified_profile_oracle.rotation_deg <= 5.0
                                ),
                                "verified_oracle_10cm_5deg": bool(
                                    verified_profile_oracle.translation_m <= 0.10
                                    and verified_profile_oracle.rotation_deg <= 5.0
                                ),
                            }
                        )
                audited_minimal_samples: set[tuple[tuple[int, int], ...]] = set()
                audited_minimal_samples_by_profile: dict[
                    str, set[tuple[tuple[int, int], ...]]
                ] = {}
                for record in hypothesis_audit_result.hypotheses:
                    if not record.sample_token_indices:
                        continue
                    sample_identity = tuple(
                        sorted(
                            zip(
                                record.sample_token_indices,
                                record.sample_track_ids,
                            )
                        )
                    )
                    profile_name = str(record.generation_profile)
                    profile_seen = audited_minimal_samples_by_profile.setdefault(
                        profile_name, set()
                    )
                    is_global_new = sample_identity not in audited_minimal_samples
                    is_profile_new = sample_identity not in profile_seen
                    if not is_global_new and not is_profile_new:
                        continue
                    sample_residuals: list[float] = []
                    for token_index, track_id in zip(
                        record.sample_token_indices, record.sample_track_ids
                    ):
                        proposal_row = int(token_index)
                        tracks = np.asarray(
                            proposals["candidate_track_ids"][proposal_row],
                            dtype=np.int64,
                        )
                        columns = np.flatnonzero(tracks == int(track_id))
                        residual = (
                            float("inf")
                            if len(columns) == 0
                            else float(
                                proposals["candidate_gt_residuals_px"][
                                    proposal_row, int(columns[0])
                                ]
                            )
                        )
                        sample_residuals.append(residual)
                    correct_2px = int(
                        np.sum(np.asarray(sample_residuals) <= 2.0)
                    )
                    correct_5px = int(
                        np.sum(np.asarray(sample_residuals) <= 5.0)
                    )
                    all_correct_2px = int(
                        bool(sample_residuals)
                        and correct_2px == len(sample_residuals)
                    )
                    all_correct_5px = int(
                        bool(sample_residuals)
                        and correct_5px == len(sample_residuals)
                    )
                    if is_global_new:
                        audited_minimal_samples.add(sample_identity)
                        grouped_minimal_sample_count += 1
                        grouped_minimal_sample_pair_count += len(sample_residuals)
                        grouped_minimal_sample_correct_2px_pair_count += correct_2px
                        grouped_minimal_sample_correct_5px_pair_count += correct_5px
                        grouped_minimal_sample_all_correct_2px_count += all_correct_2px
                        grouped_minimal_sample_all_correct_5px_count += all_correct_5px
                    if is_profile_new:
                        profile_seen.add(sample_identity)
                        profile_stats = hypothesis_profile_audit.setdefault(
                            profile_name,
                            {
                                "valid_hypothesis_count": 0,
                                "raw_hypothesis_count": 0,
                                "latent_em_hypothesis_count": 0,
                                "minimal_sample_count": 0,
                                "minimal_sample_pair_count": 0,
                                "minimal_sample_correct_2px_pair_count": 0,
                                "minimal_sample_correct_5px_pair_count": 0,
                                "minimal_sample_all_correct_2px_count": 0,
                                "minimal_sample_all_correct_5px_count": 0,
                            },
                        )
                        for key, increment in (
                            ("minimal_sample_count", 1),
                            ("minimal_sample_pair_count", len(sample_residuals)),
                            ("minimal_sample_correct_2px_pair_count", correct_2px),
                            ("minimal_sample_correct_5px_pair_count", correct_5px),
                            ("minimal_sample_all_correct_2px_count", all_correct_2px),
                            ("minimal_sample_all_correct_5px_count", all_correct_5px),
                        ):
                            profile_stats[key] = int(profile_stats[key]) + int(
                                increment
                            )
                diagnostic_pool = grouped_candidate_pools.get(query_id)
                audit_limit = int(args.max_hypothesis_information_audit)
                audit_written = 0
                for hypothesis_index, (record, hypothesis_pose) in enumerate(
                    zip(
                        hypothesis_audit_result.hypotheses,
                        hypothesis_audit_result.hypothesis_poses_w2c,
                    )
                ):
                    if hypothesis_pose is None:
                        continue
                    is_chosen = _hypothesis_is_chosen_for_audit(
                        hypothesis_audit_result,
                        hypothesis_index,
                    )
                    if audit_limit == 0:
                        continue
                    if (
                        audit_limit > 0
                        and audit_written >= audit_limit
                        and not is_chosen
                    ):
                        continue
                    audit_written += 1
                    if diagnostic_pool is None:
                        diagnostic_matches = matches
                    else:
                        diagnostic_matches, _selected, _residuals = (
                            resolve_pose_guided_candidate_pool(
                                diagnostic_pool,
                                hypothesis_pose,
                                camera,
                                residual_sigma_px=float(
                                    args.candidate_pool_residual_sigma_px
                                ),
                                hard_threshold_px=float(
                                    args.candidate_pool_hard_threshold_px
                                ),
                                descriptor_rank_weight=float(
                                    args.candidate_pool_descriptor_rank_weight
                                ),
                            )
                        )
                    hypothesis_error = pnp_pose_error(hypothesis_pose, gt_pose)
                    verification = record.verification
                    hypothesis_information_audit.append(
                        {
                            "hypothesis_index": int(hypothesis_index),
                            "chosen": is_chosen,
                            "fit_match_count_limit": int(
                                record.fit_match_count_limit
                            ),
                            "fit_match_count": int(record.fit_match_count),
                            "fit_inlier_count": int(record.fit_inlier_count),
                            "selection_mode": str(record.selection_mode),
                            "generation_profile": str(
                                record.generation_profile
                            ),
                            "prosac_prefix_size": (
                                None
                                if record.prosac_prefix_size is None
                                else int(record.prosac_prefix_size)
                            ),
                            "sampling_log_probability": (
                                None
                                if record.sampling_log_probability is None
                                else float(record.sampling_log_probability)
                            ),
                            "ransac_threshold_px": float(
                                record.ransac_threshold_px
                            ),
                            "rng_seed_offset": int(record.rng_seed_offset),
                            "local_optimization_applied": bool(
                                record.local_optimization_applied
                            ),
                            "latent_em_applied": bool(record.latent_em_applied),
                            "latent_em_iterations": int(
                                record.latent_em_iterations
                            ),
                            "latent_em_final_log_likelihood": (
                                None
                                if record.latent_em_final_log_likelihood is None
                                else float(record.latent_em_final_log_likelihood)
                            ),
                            "latent_em_parent_hypothesis_index": (
                                None
                                if record.latent_em_parent_hypothesis_index is None
                                else int(record.latent_em_parent_hypothesis_index)
                            ),
                            "latent_em_seed_evidence_mode": (
                                None
                                if record.latent_em_seed_evidence_mode is None
                                else str(record.latent_em_seed_evidence_mode)
                            ),
                            "resolved_match_count": int(len(diagnostic_matches)),
                            "preliminary_log_likelihood_mean": (
                                None
                                if record.preliminary_log_likelihood_mean is None
                                else float(
                                    record.preliminary_log_likelihood_mean
                                )
                            ),
                            "shortlist_log_likelihood_mean": (
                                None
                                if record.shortlist_log_likelihood_mean is None
                                else float(record.shortlist_log_likelihood_mean)
                            ),
                            "shortlist_evidence_mode": (
                                None
                                if record.shortlist_evidence_mode is None
                                else str(record.shortlist_evidence_mode)
                            ),
                            "spatial_rescored_for_shortlist": bool(
                                record.spatial_rescored_for_shortlist
                            ),
                            "verified_for_final_selection": (
                                verification is not None
                            ),
                            "fixed_posterior_log_likelihood_mean": (
                                None
                                if verification is None
                                else verification.fixed_posterior_log_likelihood_mean
                            ),
                            "fixed_posterior_log_likelihood_std": (
                                None
                                if verification is None
                                else verification.fixed_posterior_log_likelihood_std
                            ),
                            "fixed_posterior_log_likelihood_standard_error": (
                                None
                                if verification is None
                                else verification.fixed_posterior_log_likelihood_standard_error
                            ),
                            "fixed_posterior_log_likelihood_median": (
                                None
                                if verification is None
                                else verification.fixed_posterior_log_likelihood_median
                            ),
                            "fixed_posterior_log_likelihood_trimmed_mean_10": (
                                None
                                if verification is None
                                else verification.fixed_posterior_log_likelihood_trimmed_mean_10
                            ),
                            "fixed_posterior_log_likelihood_worst_quartile_mean": (
                                None
                                if verification is None
                                else verification.fixed_posterior_log_likelihood_worst_quartile_mean
                            ),
                            "fixed_posterior_log_likelihood_lcb95": (
                                None
                                if verification is None
                                else verification.fixed_posterior_log_likelihood_lcb95
                            ),
                            "fixed_posterior_spatial_median_of_means_2x2": (
                                None
                                if verification is None
                                else verification.fixed_posterior_spatial_median_of_means_2x2
                            ),
                            "fixed_posterior_spatial_mom_cell_count": (
                                None
                                if verification is None
                                else int(
                                    verification.fixed_posterior_spatial_mom_cell_count
                                )
                            ),
                            "fixed_posterior_identity_prior_temperature": (
                                None
                                if verification is None
                                else float(
                                    verification.fixed_posterior_identity_prior_temperature
                                )
                            ),
                            "fixed_posterior_identity_prior_entropy_mean": (
                                None
                                if verification is None
                                else verification.fixed_posterior_identity_prior_entropy_mean
                            ),
                            "fixed_posterior_identity_prior_effective_candidate_count_mean": (
                                None
                                if verification is None
                                else verification.fixed_posterior_identity_prior_effective_candidate_count_mean
                            ),
                            "fixed_posterior_retained_candidate_mass_mean": (
                                None
                                if verification is None
                                else verification.fixed_posterior_retained_candidate_mass_mean
                            ),
                            "fixed_posterior_null_evidence_fraction_mean": (
                                None
                                if verification is None
                                else verification.fixed_posterior_null_evidence_fraction_mean
                            ),
                            "fixed_posterior_candidate_inlier_evidence_fraction_mean": (
                                None
                                if verification is None
                                else verification.fixed_posterior_candidate_inlier_evidence_fraction_mean
                            ),
                            "fixed_posterior_spatial_log_likelihood_gain_mean": (
                                None
                                if verification is None
                                else verification.fixed_posterior_spatial_log_likelihood_gain_mean
                            ),
                            "fixed_posterior_spatial_calibrated_candidate_count": (
                                None
                                if verification is None
                                else int(
                                    verification.fixed_posterior_spatial_calibrated_candidate_count
                                )
                            ),
                            "fixed_posterior_spatial_geometry_calibration_weight": (
                                None
                                if verification is None
                                else float(
                                    verification.fixed_posterior_spatial_geometry_calibration_weight
                                )
                            ),
                            "strict_inlier_count": (
                                None
                                if verification is None
                                else int(verification.strict_inlier_count)
                            ),
                            "strict_grid_cell_count": (
                                None
                                if verification is None
                                else int(verification.strict_grid_cell_count)
                            ),
                            "soft_consensus": (
                                None
                                if verification is None
                                else float(verification.soft_consensus)
                            ),
                            "selected_candidate_fraction": (
                                None
                                if verification is None
                                else float(verification.selected_candidate_fraction)
                            ),
                            "selected_descriptor_score_mean": (
                                None
                                if verification is None
                                else verification.selected_descriptor_score_mean
                            ),
                            "selected_descriptor_margin_mean": (
                                None
                                if verification is None
                                else verification.selected_descriptor_margin_mean
                            ),
                            "selected_assignment_utility_mean": (
                                None
                                if verification is None
                                else verification.selected_assignment_utility_mean
                            ),
                            "selected_reprojection_mean_px": (
                                None
                                if verification is None
                                else verification.selected_reprojection_mean_px
                            ),
                            "selected_reprojection_p90_px": (
                                None
                                if verification is None
                                else verification.selected_reprojection_p90_px
                            ),
                            "measurement_probability_mean": (
                                None
                                if verification is None
                                else verification.measurement_probability_mean
                            ),
                            "measurement_high_confidence_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_high_confidence_fraction
                                )
                            ),
                            "measurement_strict_probability_mass_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_strict_probability_mass_fraction
                                )
                            ),
                            "measurement_loose_probability_mass_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_loose_probability_mass_fraction
                                )
                            ),
                            "measurement_soft_consensus_ratio": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_soft_consensus_ratio
                                )
                            ),
                            "measurement_high_confidence_strict_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_high_confidence_strict_fraction
                                )
                            ),
                            "measurement_high_confidence_loose_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_high_confidence_loose_fraction
                                )
                            ),
                            "measurement_high_confidence_contradiction_fraction": (
                                None
                                if verification is None
                                else float(
                                    verification.measurement_high_confidence_contradiction_fraction
                                )
                            ),
                            # These target errors are written only after inference
                            # and must never be consumed by a production selector.
                            "translation_m_TARGET_ONLY": (
                                None
                                if not np.isfinite(hypothesis_error.translation_m)
                                else float(hypothesis_error.translation_m)
                            ),
                            "rotation_deg_TARGET_ONLY": (
                                None
                                if not np.isfinite(hypothesis_error.rotation_deg)
                                else float(hypothesis_error.rotation_deg)
                            ),
                            **pose_information_diagnostics(
                                hypothesis_pose,
                                diagnostic_matches,
                                camera,
                                residual_sigma_px=float(
                                    args.candidate_pool_residual_sigma_px
                                ),
                            ),
                        }
                    )
            success = bool(
                solver_success
                and np.isfinite(error.translation_m)
                and np.isfinite(error.rotation_deg)
            )
            rows.append(
                {
                    "query_id": query_id,
                    "success": success,
                    "solver_success": solver_success,
                    "failure_reason": None if success else "pnp_solver_failure",
                    "match_count": match_count,
                    "inlier_count": inlier_count,
                    "translation_m": (
                        None if not np.isfinite(error.translation_m) else float(error.translation_m)
                    ),
                    "rotation_deg": (
                        None if not np.isfinite(error.rotation_deg) else float(error.rotation_deg)
                    ),
                    "pre_refine_translation_m": (
                        None
                        if pre_refine_error is None
                        or not np.isfinite(pre_refine_error.translation_m)
                        else float(pre_refine_error.translation_m)
                    ),
                    "pre_refine_rotation_deg": (
                        None
                        if pre_refine_error is None
                        or not np.isfinite(pre_refine_error.rotation_deg)
                        else float(pre_refine_error.rotation_deg)
                    ),
                    "hypothesis_oracle_translation_m": (
                        None if oracle_error is None else float(oracle_error.translation_m)
                    ),
                    "hypothesis_oracle_rotation_deg": (
                        None if oracle_error is None else float(oracle_error.rotation_deg)
                    ),
                    "hypothesis_oracle_10cm_5deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.10
                            and oracle_error.rotation_deg <= 5.0
                        )
                    ),
                    "hypothesis_oracle_3cm_5deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.03
                            and oracle_error.rotation_deg <= 5.0
                        )
                    ),
                    "hypothesis_oracle_25cm_2deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.25
                            and oracle_error.rotation_deg <= 2.0
                        )
                    ),
                    "hypothesis_oracle_5cm_5deg": (
                        None
                        if oracle_error is None
                        else bool(
                            oracle_error.translation_m <= 0.05
                            and oracle_error.rotation_deg <= 5.0
                        )
                    ),
                    "chosen_hypothesis_translation_rank": chosen_translation_rank,
                    "valid_hypothesis_count": int(valid_hypothesis_count),
                    "hypothesis_3cm_5deg_count": int(hypothesis_3cm_count),
                    "hypothesis_10cm_5deg_count": int(hypothesis_10cm_count),
                    "hypothesis_25cm_2deg_count": int(hypothesis_25cm_count),
                    "catastrophic_hypothesis_count": int(
                        catastrophic_hypothesis_count
                    ),
                    "raw_hypothesis_oracle_translation_m_TARGET_ONLY": (
                        raw_hypothesis_oracle_translation_m
                    ),
                    "raw_hypothesis_oracle_rotation_deg_TARGET_ONLY": (
                        raw_hypothesis_oracle_rotation_deg
                    ),
                    "latent_em_hypothesis_count": int(
                        latent_em_hypothesis_count
                    ),
                    "grouped_observability_rejected_hypothesis_count": int(
                        grouped_observability_rejected_hypothesis_count
                    ),
                    "latent_em_hypothesis_oracle_translation_m_TARGET_ONLY": (
                        latent_em_hypothesis_oracle_translation_m
                    ),
                    "latent_em_hypothesis_oracle_rotation_deg_TARGET_ONLY": (
                        latent_em_hypothesis_oracle_rotation_deg
                    ),
                    "latent_em_seed_parent_count_TARGET_ONLY": int(
                        latent_em_seed_parent_count
                    ),
                    "latent_em_seed_parent_oracle_translation_m_TARGET_ONLY": (
                        latent_em_seed_parent_oracle_translation_m
                    ),
                    "latent_em_seed_parent_oracle_rotation_deg_TARGET_ONLY": (
                        latent_em_seed_parent_oracle_rotation_deg
                    ),
                    "latent_em_seed_parent_raw_rank_TARGET_ONLY": (
                        latent_em_seed_parent_raw_rank
                    ),
                    "latent_em_refined_minus_seed_oracle_translation_m_TARGET_ONLY": (
                        latent_em_refined_minus_seed_oracle_translation_m
                    ),
                    "verified_hypothesis_count": int(verified_hypothesis_count),
                    "spatial_rescored_hypothesis_count": int(
                        spatial_rescored_hypothesis_count
                    ),
                    "spatial_rescored_hypothesis_oracle_translation_m_TARGET_ONLY": (
                        spatial_rescored_hypothesis_oracle_translation_m
                    ),
                    "spatial_rescored_hypothesis_oracle_rotation_deg_TARGET_ONLY": (
                        spatial_rescored_hypothesis_oracle_rotation_deg
                    ),
                    "verified_hypothesis_oracle_translation_m_TARGET_ONLY": (
                        verified_hypothesis_oracle_translation_m
                    ),
                    "verified_hypothesis_oracle_rotation_deg_TARGET_ONLY": (
                        verified_hypothesis_oracle_rotation_deg
                    ),
                    "verified_hypothesis_oracle_3cm_5deg_TARGET_ONLY": (
                        None
                        if verified_hypothesis_oracle_translation_m is None
                        else bool(
                            verified_hypothesis_oracle_translation_m <= 0.03
                            and verified_hypothesis_oracle_rotation_deg <= 5.0
                        )
                    ),
                    "verified_hypothesis_oracle_5cm_5deg_TARGET_ONLY": (
                        None
                        if verified_hypothesis_oracle_translation_m is None
                        else bool(
                            verified_hypothesis_oracle_translation_m <= 0.05
                            and verified_hypothesis_oracle_rotation_deg <= 5.0
                        )
                    ),
                    "verified_hypothesis_oracle_10cm_5deg_TARGET_ONLY": (
                        None
                        if verified_hypothesis_oracle_translation_m is None
                        else bool(
                            verified_hypothesis_oracle_translation_m <= 0.10
                            and verified_hypothesis_oracle_rotation_deg <= 5.0
                        )
                    ),
                    "latent_em_parent_pair_count_TARGET_ONLY": int(
                        len(latent_em_parent_translation_deltas)
                    ),
                    "latent_em_parent_win_count_TARGET_ONLY": int(
                        np.sum(
                            np.asarray(
                                latent_em_parent_translation_deltas,
                                dtype=np.float64,
                            )
                            < 0.0
                        )
                    ),
                    "latent_em_parent_loss_count_TARGET_ONLY": int(
                        np.sum(
                            np.asarray(
                                latent_em_parent_translation_deltas,
                                dtype=np.float64,
                            )
                            > 0.0
                        )
                    ),
                    "latent_em_parent_median_translation_delta_m_TARGET_ONLY": (
                        None
                        if not latent_em_parent_translation_deltas
                        else float(np.median(latent_em_parent_translation_deltas))
                    ),
                    "latent_em_parent_max_translation_delta_m_TARGET_ONLY": (
                        None
                        if not latent_em_parent_translation_deltas
                        else float(np.max(latent_em_parent_translation_deltas))
                    ),
                    # Minimal-set correctness is a target-only post-inference
                    # audit and is never exposed to generation or selection.
                    "grouped_minimal_sample_count": int(
                        grouped_minimal_sample_count
                    ),
                    "grouped_minimal_sample_pair_count_TARGET_ONLY": int(
                        grouped_minimal_sample_pair_count
                    ),
                    "grouped_minimal_sample_correct_2px_pair_count_TARGET_ONLY": int(
                        grouped_minimal_sample_correct_2px_pair_count
                    ),
                    "grouped_minimal_sample_correct_5px_pair_count_TARGET_ONLY": int(
                        grouped_minimal_sample_correct_5px_pair_count
                    ),
                    "grouped_minimal_sample_all_correct_2px_count_TARGET_ONLY": int(
                        grouped_minimal_sample_all_correct_2px_count
                    ),
                    "grouped_minimal_sample_all_correct_5px_count_TARGET_ONLY": int(
                        grouped_minimal_sample_all_correct_5px_count
                    ),
                    "hypothesis_information_audit_with_TARGET_ONLY_errors": (
                        hypothesis_information_audit
                    ),
                    "hypothesis_profile_audit_TARGET_ONLY": hypothesis_profile_audit,
                    "crossfit_promotion_promoted": (
                        None
                        if crossfit_promotion_audit is None
                        else bool(crossfit_promotion_audit["promoted"])
                    ),
                    "crossfit_promotion_abstained": (
                        None
                        if crossfit_promotion_audit is None
                        else bool(crossfit_promotion_audit["abstained"])
                    ),
                    "crossfit_promotion_likelihood_mean_delta": (
                        None
                        if crossfit_promotion_audit is None
                        else crossfit_promotion_audit[
                            "optional_minus_baseline_log_likelihood_mean"
                        ]
                    ),
                    "crossfit_baseline_translation_m_TARGET_ONLY": (
                        None
                        if promotion_baseline_error is None
                        or not np.isfinite(promotion_baseline_error.translation_m)
                        else float(promotion_baseline_error.translation_m)
                    ),
                    "crossfit_optional_translation_m_TARGET_ONLY": (
                        None
                        if promotion_optional_error is None
                        or not np.isfinite(promotion_optional_error.translation_m)
                        else float(promotion_optional_error.translation_m)
                    ),
                    "crossfit_baseline_rotation_deg_TARGET_ONLY": (
                        None
                        if promotion_baseline_error is None
                        or not np.isfinite(promotion_baseline_error.rotation_deg)
                        else float(promotion_baseline_error.rotation_deg)
                    ),
                    "crossfit_optional_rotation_deg_TARGET_ONLY": (
                        None
                        if promotion_optional_error is None
                        or not np.isfinite(promotion_optional_error.rotation_deg)
                        else float(promotion_optional_error.rotation_deg)
                    ),
                    "inference_verification": backend_summary,
                }
            )
        return _pose_summary(rows), rows

    validation_trials: list[dict[str, object]] = []
    validation_rows: dict[str, dict[str, object]] = {}
    legacy_policy_items = () if bool(args.grouped_only) else policy_scores.items()
    for policy_key, scores in legacy_policy_items:
        source_scores = raw_scores[str(policy_metadata[policy_key]["score_key"])]
        candidate_pool_scores = (
            None
            if bool(args.disable_topl_candidate_pool_verification)
            else source_scores
        )
        verified_pose, verified_rows = evaluate(
            scores,
            "validation",
            "verified",
            candidate_pool_scores=candidate_pool_scores,
        )
        single_pose, single_rows = evaluate(scores, "validation", "single_ransac")
        trial = {
            "trial_id": int(len(validation_trials)),
            "policy_key": policy_key,
            **policy_metadata[policy_key],
            "verified_pose": verified_pose,
            "matched_single_ransac_pose": single_pose,
            "passes_matched_pose_gate": _pose_gate(verified_pose, single_pose),
            "passes_frozen_baseline_pose_gate": (
                None
                if frozen_baseline_source is None
                else _pose_gate(verified_pose, frozen_baseline_source["pose"])
            ),
        }
        validation_trials.append(trial)
        validation_rows[policy_key] = {
            "verified": verified_rows,
            "matched_single_ransac": single_rows,
        }
    if bool(args.enable_grouped_candidate_pnp):
        if grouped_null_scores is None:
            raise RuntimeError("grouped null scores were not loaded")
        for score_key in score_keys:
            scores = raw_scores[str(score_key)]
            grouped_pose, grouped_rows = evaluate(
                scores,
                "validation",
                "grouped_candidate_pool",
                candidate_pool_scores=scores,
                candidate_pool_null_scores=grouped_null_scores,
                candidate_pool_spatial_payload=spatial_payloads.get("validation"),
                candidate_pool_generation_spatial_payload=(
                    generation_spatial_payloads.get("validation")
                ),
                candidate_pool_geometry_probabilities=(
                    geometry_probability_matrices.get("validation")
                ),
                candidate_pool_update_probabilities=(
                    update_probability_matrices.get("validation")
                ),
                candidate_pool_update_refined_xy=(
                    update_refined_xy_matrices.get("validation")
                ),
                candidate_pool_update_threshold=float(
                    update_prediction_metadata.get("validation", {}).get(
                        "update_threshold", 0.5
                    )
                ),
                evaluation_label=(
                    f"grouped_candidate_pool__{str(score_key)}"
                ),
            )
            single_pose, single_rows = evaluate(
                scores, "validation", "single_ransac"
            )
            policy_key = f"grouped_candidate_pool__{str(score_key)}"
            trial = {
                "trial_id": int(len(validation_trials)),
                "policy_key": policy_key,
                "assignment_mode": "grouped_progressive_topl_with_explicit_null",
                "score_key": str(score_key),
                "verified_pose": grouped_pose,
                "matched_single_ransac_pose": single_pose,
                "passes_matched_pose_gate": _pose_gate(grouped_pose, single_pose),
                "passes_frozen_baseline_pose_gate": (
                    None
                    if frozen_baseline_source is None
                    else _pose_gate(grouped_pose, frozen_baseline_source["pose"])
                ),
            }
            validation_trials.append(trial)
            policy_metadata[policy_key] = {
                "assignment_mode": trial["assignment_mode"],
                "score_key": str(score_key),
            }
            validation_rows[policy_key] = {
                "verified": grouped_rows,
                "matched_single_ransac": single_rows,
            }
    if frozen_baseline_source is not None:
        baseline_policy_key = f"global_bipartite__{str(args.baseline_score_key)}"
        baseline_trials = [
            trial
            for trial in validation_trials
            if str(trial["policy_key"]) == baseline_policy_key
        ]
        if len(baseline_trials) != 1:
            raise ValueError(
                "frozen baseline replay requires global_bipartite baseline scores"
            )
        _validate_frozen_baseline_pose(
            baseline_trials[0]["matched_single_ransac_pose"],
            frozen_baseline_source,
        )
    finite_trials = [
        trial
        for trial in validation_trials
        if trial["verified_pose"]["median_translation_m_success"] is not None
        and trial["verified_pose"]["p90_translation_m_success"] is not None
    ]
    if not finite_trials:
        raise RuntimeError("no verification policy produced a finite validation pose")
    gate_key = (
        "passes_matched_pose_gate"
        if frozen_baseline_source is None
        else "passes_frozen_baseline_pose_gate"
    )
    gated = [trial for trial in finite_trials if bool(trial[gate_key])]
    chosen = max(gated or finite_trials, key=_policy_key)
    chosen["selection_fallback_without_strict_pose_gate"] = not bool(gated)

    late_trials: list[dict[str, object]] = []
    late_rows: dict[str, dict[str, object]] = {}
    if bool(args.development_cross_block_audit):
        for trial in validation_trials:
            policy_key = str(trial["policy_key"])
            source_scores = raw_scores[str(policy_metadata[policy_key]["score_key"])]
            candidate_pool_scores = (
                None
                if bool(args.disable_topl_candidate_pool_verification)
                else source_scores
            )
            is_grouped = str(policy_metadata[policy_key]["assignment_mode"]).startswith(
                "grouped_progressive"
            )
            evaluation_scores = (
                source_scores if is_grouped else policy_scores[policy_key]
            )
            verified_pose, verified_rows = evaluate(
                evaluation_scores,
                "test",
                "grouped_candidate_pool" if is_grouped else "verified",
                candidate_pool_scores=candidate_pool_scores,
                candidate_pool_null_scores=(
                    grouped_null_scores if is_grouped else None
                ),
                candidate_pool_spatial_payload=(
                    spatial_payloads.get("test") if is_grouped else None
                ),
                candidate_pool_generation_spatial_payload=(
                    generation_spatial_payloads.get("test") if is_grouped else None
                ),
                candidate_pool_geometry_probabilities=(
                    geometry_probability_matrices.get("test") if is_grouped else None
                ),
                candidate_pool_update_probabilities=(
                    update_probability_matrices.get("test") if is_grouped else None
                ),
                candidate_pool_update_refined_xy=(
                    update_refined_xy_matrices.get("test") if is_grouped else None
                ),
                candidate_pool_update_threshold=float(
                    update_prediction_metadata.get("test", {}).get(
                        "update_threshold", 0.5
                    )
                ),
                evaluation_label=policy_key,
            )
            single_pose, single_rows = evaluate(
                evaluation_scores, "test", "single_ransac"
            )
            late_trials.append(
                {
                    "trial_id": int(trial["trial_id"]),
                    "policy_key": policy_key,
                    "verified_pose": verified_pose,
                    "matched_single_ransac_pose": single_pose,
                    "passes_matched_pose_gate": _pose_gate(verified_pose, single_pose),
                    "passes_frozen_baseline_pose_gate": (
                        None
                        if frozen_baseline_source is None
                        or not isinstance(
                            frozen_baseline_source.get("late_development_pose"),
                            dict,
                        )
                        else _pose_gate(
                            verified_pose,
                            frozen_baseline_source["late_development_pose"],
                        )
                    ),
                    "selected_by_late_metrics": False,
                }
            )
            late_rows[policy_key] = {
                "verified": verified_rows,
                "matched_single_ransac": single_rows,
            }
        if frozen_baseline_source is not None:
            expected_late_pose = frozen_baseline_source.get("late_development_pose")
            if not isinstance(expected_late_pose, dict):
                raise ValueError(
                    "frozen baseline summary has no late development replay"
                )
            baseline_policy_key = f"global_bipartite__{str(args.baseline_score_key)}"
            baseline_late_trials = [
                trial
                for trial in late_trials
                if str(trial["policy_key"]) == baseline_policy_key
            ]
            if len(baseline_late_trials) != 1:
                raise ValueError("late replay is missing the frozen baseline policy")
            _validate_frozen_baseline_pose(
                baseline_late_trials[0]["matched_single_ransac_pose"],
                {"pose": expected_late_pose},
            )

    train_trials: list[dict[str, object]] = []
    train_rows: dict[str, dict[str, object]] = {}
    if bool(args.export_train_grouped_hypotheses):
        if grouped_null_scores is None:
            raise RuntimeError("train grouped export requires grouped null scores")
        train_geometry = geometry_probability_matrices.get("train")
        if _train_grouped_export_requires_geometry(args) and train_geometry is None:
            raise RuntimeError(
                "geometry-guided train grouped export requires aligned OOF geometry"
            )
        for score_key in score_keys:
            scores = raw_scores[str(score_key)]
            grouped_pose, grouped_rows = evaluate(
                scores,
                "train",
                "grouped_candidate_pool",
                candidate_pool_scores=scores,
                candidate_pool_null_scores=grouped_null_scores,
                candidate_pool_geometry_probabilities=train_geometry,
                candidate_pool_spatial_payload=spatial_payloads.get("train"),
                evaluation_label=(
                    f"grouped_candidate_pool__{str(score_key)}"
                ),
            )
            policy_key = f"grouped_candidate_pool__{str(score_key)}"
            train_trials.append(
                {
                    "policy_key": policy_key,
                    "score_key": str(score_key),
                    "verified_pose": grouped_pose,
                    "target_labels_exported_only_after_inference": True,
                }
            )
            train_rows[policy_key] = {"verified": grouped_rows}

    rows_path = output_dir / "pose_rows.json"
    rows_path.write_text(
        json.dumps(
            {
                "train_oof_geometry": train_rows,
                "validation": validation_rows,
                "late_development": late_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    manifest = {
        "proposals": str(proposals_path),
        "proposals_sha256": file_sha256_short(proposals_path),
        "candidate_artifact": str(candidate_path),
        "candidate_artifact_sha256": file_sha256_short(candidate_path),
        "score_artifact": str(score_path),
        "score_artifact_sha256": file_sha256_short(score_path),
        "candidate_posterior_overlays": posterior_overlay_manifest,
        "projected_landmark_bank": str(bank_path),
        "projected_landmark_bank_sha256": file_sha256_short(bank_path),
        "maplet_support_index": None if maplet_path is None else str(maplet_path),
        "maplet_support_index_sha256": (
            None if maplet_path is None else file_sha256_short(maplet_path)
        ),
        "maplet_cluster_construction": (
            None
            if maplet_path is None
            else "deterministic_mutual_neighbor_disjoint_packing_v1"
        ),
        "colmap_model_dir": str(model_dir),
        "colmap_cameras_bin": str(model_dir / "cameras.bin"),
        "colmap_cameras_bin_sha256": file_sha256_short(
            model_dir / "cameras.bin"
        ),
        "colmap_images_bin": str(model_dir / "images.bin"),
        "colmap_images_bin_sha256": file_sha256_short(model_dir / "images.bin"),
        "split_json": str(split_path),
        "split_json_sha256": file_sha256_short(split_path),
        "frozen_baseline_summary": (
            None
            if frozen_baseline_source is None
            else str(frozen_baseline_source["source_path"])
        ),
        "frozen_baseline_summary_sha256": (
            None
            if frozen_baseline_source is None
            else str(frozen_baseline_source["source_sha256"])
        ),
        "immutable_baseline_pose_artifact": (
            None
            if immutable_baseline_pose_source is None
            else str(immutable_baseline_pose_source["path"])
        ),
        "immutable_baseline_pose_artifact_sha256": (
            None
            if immutable_baseline_pose_source is None
            else str(immutable_baseline_pose_source["sha256"])
        ),
        "immutable_baseline_pose_evaluation_label": (
            None
            if immutable_baseline_pose_source is None
            else str(immutable_baseline_pose_source["evaluation_label"])
        ),
        "descriptor_space_id": landmark_metadata.get("descriptor_space_id"),
        "candidate_evidence": (
            None if candidate_evidence_path is None else str(candidate_evidence_path)
        ),
        "candidate_evidence_sha256": (
            None
            if candidate_evidence_path is None
            else file_sha256_short(candidate_evidence_path)
        ),
        "candidate_spatial_likelihood_train": [
            str(path) for path in spatial_paths["train"]
        ],
        "candidate_spatial_likelihood_train_sha256": [
            file_sha256_short(path) for path in spatial_paths["train"]
        ],
        "candidate_spatial_likelihood_validation": (
            None if not spatial_paths["validation"] else str(spatial_paths["validation"][0])
        ),
        "candidate_spatial_likelihood_validation_sha256": (
            None if not spatial_paths["validation"] else file_sha256_short(spatial_paths["validation"][0])
        ),
        "candidate_spatial_likelihood_test": (
            None if not spatial_paths["test"] else str(spatial_paths["test"][0])
        ),
        "candidate_spatial_likelihood_test_sha256": (
            None if not spatial_paths["test"] else file_sha256_short(spatial_paths["test"][0])
        ),
        "candidate_generation_spatial_likelihood_validation": (
            None
            if generation_spatial_paths["validation"] is None
            else str(generation_spatial_paths["validation"])
        ),
        "candidate_generation_spatial_likelihood_validation_sha256": (
            None
            if generation_spatial_paths["validation"] is None
            else file_sha256_short(generation_spatial_paths["validation"])
        ),
        "candidate_generation_spatial_likelihood_test": (
            None
            if generation_spatial_paths["test"] is None
            else str(generation_spatial_paths["test"])
        ),
        "candidate_generation_spatial_likelihood_test_sha256": (
            None
            if generation_spatial_paths["test"] is None
            else file_sha256_short(generation_spatial_paths["test"])
        ),
        "candidate_spatial_calibration": (
            None
            if spatial_calibration_path is None
            else str(spatial_calibration_path)
        ),
        "candidate_spatial_calibration_sha256": (
            None
            if spatial_calibration_path is None
            else file_sha256_short(spatial_calibration_path)
        ),
        "candidate_geometry_probabilities": geometry_probability_metadata,
        "candidate_coordinate_updates": update_prediction_metadata,
    }
    grouped_hypothesis_artifact_path: Path | None = None
    if bool(args.export_grouped_hypothesis_artifact):
        if not grouped_hypothesis_export_rows:
            raise RuntimeError(
                "grouped hypothesis artifact was requested but no poses were generated"
            )
        grouped_hypothesis_artifact_path = (
            output_dir / "grouped_hypotheses_inference_only_v1.npz"
        )
        scalar_fields = {
            "query_ids": ("query_id", str),
            "split_names": ("split_name", str),
            "evaluation_labels": ("evaluation_label", str),
            "hypothesis_indices": ("hypothesis_index", np.int64),
            "generation_profiles": ("generation_profile", str),
            "selection_modes": ("selection_mode", str),
            "latent_em_applied": ("latent_em_applied", bool),
            "latent_em_parent_hypothesis_indices": (
                "latent_em_parent_hypothesis_index",
                np.int64,
            ),
            "latent_em_seed_evidence_modes": (
                "latent_em_seed_evidence_mode",
                str,
            ),
            "local_optimization_applied": (
                "local_optimization_applied",
                bool,
            ),
            "sample_counts": ("sample_count", np.int64),
            "sampling_log_probabilities": (
                "sampling_log_probability",
                np.float64,
            ),
            "preliminary_log_likelihood_means": (
                "preliminary_log_likelihood_mean",
                np.float64,
            ),
            "shortlist_log_likelihood_means": (
                "shortlist_log_likelihood_mean",
                np.float64,
            ),
            "shortlist_evidence_modes": ("shortlist_evidence_mode", str),
            "shortlisted_for_verification": (
                "shortlisted_for_verification",
                bool,
            ),
            "verification_log_likelihood_means": (
                "verification_log_likelihood_mean",
                np.float64,
            ),
            "verification_effective_group_counts": (
                "verification_effective_group_count",
                np.int64,
            ),
            "verification_identity_prior_temperatures": (
                "verification_identity_prior_temperature",
                np.float64,
            ),
            "verification_null_evidence_fraction_means": (
                "verification_null_evidence_fraction_mean",
                np.float64,
            ),
            "verification_candidate_inlier_evidence_fraction_means": (
                "verification_candidate_inlier_evidence_fraction_mean",
                np.float64,
            ),
            "verification_spatial_log_likelihood_gain_means": (
                "verification_spatial_log_likelihood_gain_mean",
                np.float64,
            ),
            "verification_relation_pair_counts": (
                "verification_relation_pair_count",
                np.int64,
            ),
            "verification_relation_effective_pair_counts": (
                "verification_relation_effective_pair_count",
                np.int64,
            ),
            "verification_relation_log_likelihood_ratio_means": (
                "verification_relation_log_likelihood_ratio_mean",
                np.float64,
            ),
            "verification_relation_log_likelihood_ratio_sums": (
                "verification_relation_log_likelihood_ratio_sum",
                np.float64,
            ),
            "verification_relation_feature_edge_counts": (
                "verification_relation_feature_edge_count",
                np.int64,
            ),
            "verification_relation_feature_candidate_pair_mass_means": (
                "verification_relation_feature_candidate_pair_mass_mean",
                np.float64,
            ),
            "verification_relation_feature_null_touching_mass_means": (
                "verification_relation_feature_null_touching_mass_mean",
                np.float64,
            ),
            "verification_relation_feature_edge_graph_sha256": (
                "verification_relation_feature_edge_graph_sha256",
                str,
            ),
            "verification_positive_depth_ratios": (
                "verification_positive_depth_ratio",
                np.float64,
            ),
            "verification_strict_inlier_counts": (
                "verification_strict_inlier_count",
                np.int64,
            ),
            "verification_loose_inlier_counts": (
                "verification_loose_inlier_count",
                np.int64,
            ),
            "verification_strict_grid_cell_counts": (
                "verification_strict_grid_cell_count",
                np.int64,
            ),
            "verification_loose_grid_cell_counts": (
                "verification_loose_grid_cell_count",
                np.int64,
            ),
            "verification_soft_consensus": (
                "verification_soft_consensus",
                np.float64,
            ),
            "verification_median_residual_px": (
                "verification_median_residual_px",
                np.float64,
            ),
            "translation_information_min_eigenvalues": (
                "translation_information_min_eigenvalue",
                np.float64,
            ),
            "translation_information_conditions": (
                "translation_information_condition",
                np.float64,
            ),
            "rotation_information_min_eigenvalues": (
                "rotation_information_min_eigenvalue",
                np.float64,
            ),
            "joint_information_conditions": (
                "joint_information_condition",
                np.float64,
            ),
            "bearing_max_angles_deg": ("bearing_max_angle_deg", np.float64),
            "camera_depth_span_ratios": (
                "camera_depth_span_ratio",
                np.float64,
            ),
            "xyz_second_singular_ratios": (
                "xyz_second_singular_ratio",
                np.float64,
            ),
            "xyz_third_singular_ratios": (
                "xyz_third_singular_ratio",
                np.float64,
            ),
            "chosen_for_optional_pose": ("chosen_for_optional_pose", bool),
        }
        arrays = {
            output_key: np.asarray(
                [row[source_key] for row in grouped_hypothesis_export_rows],
                dtype=dtype,
            )
            for output_key, (source_key, dtype) in scalar_fields.items()
        }
        arrays.update(
            {
                "poses_w2c": np.stack(
                    [
                        np.asarray(row["pose_w2c"], dtype=np.float64)
                        for row in grouped_hypothesis_export_rows
                    ],
                    axis=0,
                ),
                "sample_token_indices": np.stack(
                    [
                        np.asarray(row["sample_token_indices"], dtype=np.int64)
                        for row in grouped_hypothesis_export_rows
                    ],
                    axis=0,
                ),
                "sample_track_ids": np.stack(
                    [
                        np.asarray(row["sample_track_ids"], dtype=np.int64)
                        for row in grouped_hypothesis_export_rows
                    ],
                    axis=0,
                ),
                "verification_relation_feature_histograms": np.asarray(
                    [
                        row["verification_relation_feature_histogram"]
                        for row in grouped_hypothesis_export_rows
                    ],
                    dtype=np.float64,
                ),
                "verification_relation_feature_edge_histograms": _pad_relation_edge_rows(
                    grouped_hypothesis_export_rows,
                    "verification_relation_feature_edge_histograms",
                    trailing_size=(
                        len(RELATION_CHANNELS)
                        * (
                            len(
                                optional_grouped_config.candidate_relation_feature_bin_edges_px
                            )
                            - 1
                        )
                    ),
                ),
                "verification_relation_feature_edge_null_touching_masses": (
                    _pad_relation_edge_rows(
                        grouped_hypothesis_export_rows,
                        "verification_relation_feature_edge_null_touching_masses",
                    )
                ),
                "metadata_json": np.asarray(
                    json.dumps(
                        {
                            "format": "grouped_pose_hypotheses_inference_only_v1",
                            "pose_or_ground_truth_used_for_generation": False,
                            "contains_target_fields": False,
                            "row_count": int(len(grouped_hypothesis_export_rows)),
                            "inputs": manifest,
                            "grouped_config": (
                                None
                                if optional_grouped_config is None
                                else asdict(optional_grouped_config)
                            ),
                        },
                        sort_keys=True,
                    )
                ),
            }
        )
        np.savez_compressed(grouped_hypothesis_artifact_path, **arrays)
    summary = {
        "stage": "heldout_multi_hypothesis_pose_verification",
        "inputs": manifest,
        "protocol": {
            "proposal_scope": "full_bank_global_faiss_top_l",
            "identity_posterior_overlay": bool(posterior_overlay_paths),
            "identity_posterior_overlay_ensemble_size": int(
                len(posterior_overlay_paths)
            ),
            "identity_posterior_overlay_candidate_key": (
                POSTERIOR_OVERLAY_CANDIDATE_KEY
                if posterior_overlay_paths
                else None
            ),
            "identity_posterior_overlay_null_key": (
                POSTERIOR_OVERLAY_NULL_KEY if posterior_overlay_paths else None
            ),
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "measurement": bool(spatial_requested),
            "candidate_specific_rgb_spatial_likelihood": bool(spatial_requested),
            "candidate_spatial_view_mixture_policy": (
                None
                if not spatial_requested
                else str(args.candidate_spatial_view_mixture_policy)
            ),
            "generation_spatial_likelihood_frozen": bool(
                generation_spatial_requested
            ),
            "generation_and_scoring_spatial_likelihood_roles_separated": bool(
                generation_spatial_requested
            ),
            "candidate_generation_spatial_view_mixture_policy": (
                None
                if not spatial_requested
                else str(
                    args.candidate_generation_spatial_view_mixture_policy
                    if generation_spatial_requested
                    else args.candidate_spatial_view_mixture_policy
                )
            ),
            "candidate_spatial_log_evidence_weight": (
                None
                if not spatial_requested
                else float(args.candidate_spatial_log_evidence_weight)
            ),
            "candidate_spatial_pose_view_geometry_sigma_deg": (
                None if not spatial_requested else scoring_view_geometry_sigma
            ),
            "candidate_generation_spatial_pose_view_geometry_sigma_deg": (
                None
                if not spatial_requested
                else (
                    generation_view_geometry_sigma
                    if generation_spatial_requested
                    else scoring_view_geometry_sigma
                )
            ),
            "pose_view_geometry_changes_identity_or_null_mass": False,
            "candidate_geometry_prior_mix_weight": float(
                args.candidate_geometry_prior_mix_weight
            ),
            "candidate_geometry_generation_mix_weight": float(
                args.candidate_geometry_generation_mix_weight
            ),
            "candidate_coordinate_update_refine": bool(
                args.enable_optional_candidate_coordinate_refine
            ),
            "candidate_coordinate_update_policy": (
                "selected_identity_only_then_independent_final_audit"
                if bool(args.enable_optional_candidate_coordinate_refine)
                else None
            ),
            "candidate_geometry_generation_policy": (
                "pose_free_true_residual_probability_within_retained_candidate_mass"
                if float(args.candidate_geometry_generation_mix_weight) > 0.0
                else "original_fixed_candidate_posterior"
            ),
            "immutable_unmixed_generation_baseline": bool(
                args.enable_grouped_generation_grid_fallback
            ),
            "geometry_guided_generation_promotion_rule": (
                "optional_strict_grid_cells_ge_baseline_plus_min_delta"
                if bool(args.enable_grouped_generation_grid_fallback)
                else None
            ),
            "grouped_generation_min_strict_grid_delta": int(
                args.grouped_generation_min_strict_grid_delta
            ),
            "component_disjoint_crossfit": bool(
                str(args.grouped_crossfit_mode) != "token_spatial"
            ),
            "grouped_crossfit_mode": str(args.grouped_crossfit_mode),
            "grouped_crossfit_role_assignment": str(
                args.grouped_crossfit_role_assignment
            ),
            "grouped_crossfit_spatial_fold_policy": str(
                args.grouped_crossfit_spatial_fold_policy
            ),
            "grouped_prosac_shortlist_evidence_mode": str(
                args.grouped_prosac_shortlist_evidence_mode
            ),
            "grouped_prosac_spatial_rescore_top_k": int(
                args.grouped_prosac_spatial_rescore_top_k
            ),
            "grouped_prosac_observability_evidence_mode": str(
                args.grouped_prosac_observability_evidence_mode
            ),
            "grouped_hypothesis_selection_policy": str(
                args.grouped_hypothesis_selection_policy
            ),
            "grouped_hypothesis_selection_production_eligible": bool(
                str(args.grouped_hypothesis_selection_policy)
                == "fixed_posterior_likelihood_only"
            ),
            "crossfit_likelihood_immutable_baseline": bool(
                args.enable_grouped_crossfit_likelihood_fallback
            ),
            "crossfit_likelihood_promotion_rule": (
                "same_denominator_heldout_log_likelihood_ratio"
                if bool(args.enable_grouped_crossfit_likelihood_fallback)
                else None
            ),
            "crossfit_likelihood_min_mean_delta": float(
                args.grouped_likelihood_min_mean_delta
            ),
            "crossfit_likelihood_min_effective_groups": int(
                args.grouped_likelihood_min_effective_groups
            ),
            "crossfit_observability_gate": {
                "min_information_matches": int(
                    args.grouped_observability_min_information_matches
                ),
                "min_translation_eigenvalue": float(
                    args.grouped_observability_min_translation_eigenvalue
                ),
                "max_translation_condition": float(
                    args.grouped_observability_max_translation_condition
                ),
                "max_joint_condition": float(
                    args.grouped_observability_max_joint_condition
                ),
                "min_bearing_span_deg": float(
                    args.grouped_observability_min_bearing_span_deg
                ),
                "min_depth_span_ratio": float(
                    args.grouped_observability_min_depth_span_ratio
                ),
                "min_xyz_second_ratio": float(
                    args.grouped_observability_min_xyz_second_ratio
                ),
                "min_xyz_third_ratio": float(
                    args.grouped_observability_min_xyz_third_ratio
                ),
            },
            "immutable_baseline_generation_mode": (
                "external_selected_pose_artifact_bit_exact"
                if immutable_baseline_pose_source is not None
                else "matched_single_ransac_current_run_DIAGNOSTIC_ONLY"
                if bool(args.enable_grouped_crossfit_likelihood_fallback)
                else str(args.grouped_immutable_baseline_generation_mode)
            ),
            "crossfit_immutable_pose_is_reestimated": bool(
                args.enable_grouped_crossfit_likelihood_fallback
                and immutable_baseline_pose_source is None
            ),
            "immutable_baseline_final_refine_acceptance_policy": (
                None
                if grouped_config is None
                else str(grouped_config.final_refine_acceptance_policy)
            ),
            "optional_final_refine_acceptance_policy": (
                None
                if optional_grouped_config is None
                else str(optional_grouped_config.final_refine_acceptance_policy)
            ),
            "immutable_baseline_final_refine_mode": (
                None
                if immutable_grouped_config is None
                else str(immutable_grouped_config.final_refine_mode)
            ),
            "optional_final_refine_mode": (
                None
                if optional_grouped_config is None
                else str(optional_grouped_config.final_refine_mode)
            ),
            "candidate_pose_evidence_version": str(
                CANDIDATE_POSE_EVIDENCE_VERSION
            ),
            "candidate_spatial_geometry_calibration_weight": float(
                args.candidate_spatial_geometry_calibration_weight
            ),
            "candidate_spatial_utility_gate_weight": float(
                args.candidate_spatial_utility_gate_weight
            ),
            "candidate_spatial_reliability_policy": (
                (
                    "true_residual_geometry_probability_interpolated_with_calibrated_dustbin"
                    if spatial_calibration is not None
                    else "true_residual_calibrated_geometry_probability_interpolated_with_raw_dustbin"
                )
                if float(args.candidate_spatial_geometry_calibration_weight) > 0.0
                else (
                    "query_grouped_platt_calibrated_measurement_dustbin"
                    if spatial_calibration is not None
                    else "raw_measurement_dustbin"
                )
            ),
            "candidate_spatial_calibration_production_eligible": (
                None
                if spatial_calibration is None
                else bool(spatial_calibration.production_eligible)
            ),
            "candidate_spatial_calibration_identity_mass_modified": False,
            "candidate_measurement_utility_action_gate": (
                None
                if not any(
                    metadata.get("artifact_kind")
                    == "measurement_utility_action_gate"
                    for metadata in update_prediction_metadata.values()
                )
                else {
                    "decision": "hard_threshold_from_train_OOF",
                    "identity_mass_modified": False,
                    "pose_likelihood_modified": False,
                    "role": "selected_identity_RGB_coordinate_update_only",
                }
            ),
            "candidate_identity_prior_temperature": float(
                args.candidate_identity_prior_temperature
            ),
            "candidate_identity_prior_mass_policy": (
                "temperature_calibrate_within_retained_mass_null_immutable"
            ),
            "candidate_geometry_prior_mass_policy": (
                "preserve_explicit_null_and_retained_candidate_mass"
                if geometry_probabilities_requested
                else None
            ),
            "ground_truth_available_to_pose_selector": False,
            "pre_pnp_depth_proxy": False,
            "pre_pnp_geometry": "image_bearing_coverage_plus_world_xyz_covariance",
            "post_pose_geometry": (
                "heldout_topl_pose_guided_bipartite_assignment_plus_cheirality_plus_camera_depth"
                if not bool(args.disable_topl_candidate_pool_verification)
                else "heldout_hard_assignment_reprojection_plus_cheirality_plus_camera_depth"
            ),
            "topl_candidate_pool_verification": not bool(
                args.disable_topl_candidate_pool_verification
            ),
            "grouped_only_execution": bool(args.grouped_only),
            "policy_selected_on_validation_only": True,
            "validation_gate_reference": (
                "matched_single_ransac"
                if frozen_baseline_source is None
                else "externally_frozen_global_baseline_exact_replay"
            ),
            "late_block_is_untouched_test": False,
            "production_promoted": False,
        },
        "execution": {
            "query_shard_count": int(args.query_shard_count),
            "query_shard_index": int(args.query_shard_index),
            "query_shard_policy": "split_order_position_modulo_v1",
            "query_ids": {
                name: list(execution_split[name])
                for name in ("train", "validation", "test")
            },
            "inference_is_query_independent": True,
            "query_seed_policy": "sha256_query_id_v1",
        },
        "config": {
            "verified_pnp": {
                **asdict(config),
                "fit_match_counts": list(config.fit_match_counts),
                "selection_modes": list(config.selection_modes),
                "ransac_thresholds_px": list(config.ransac_thresholds_px),
                "rng_seed_offsets": list(config.rng_seed_offsets),
            },
            "grouped_candidate_pnp": {
                "enabled": bool(args.enable_grouped_candidate_pnp),
                "null_score_key": str(args.grouped_null_score_key),
                "candidate_spatial_log_evidence_weight": float(
                    args.candidate_spatial_log_evidence_weight
                ),
                "candidate_identity_prior_temperature": float(
                    args.candidate_identity_prior_temperature
                ),
                "candidate_spatial_pose_view_geometry_sigma_deg": float(
                    scoring_view_geometry_sigma
                ),
                "candidate_generation_spatial_pose_view_geometry_sigma_deg": float(
                    generation_view_geometry_sigma
                    if generation_spatial_requested
                    else scoring_view_geometry_sigma
                ),
                "candidate_geometry_prior_mix_weight": float(
                    args.candidate_geometry_prior_mix_weight
                ),
                "candidate_geometry_generation_mix_weight": float(
                    args.candidate_geometry_generation_mix_weight
                ),
                "immutable_unmixed_generation_baseline": bool(
                    args.enable_grouped_generation_grid_fallback
                ),
                "generation_promotion_rule": (
                    "optional_strict_grid_cells_ge_baseline_plus_min_delta"
                    if bool(args.enable_grouped_generation_grid_fallback)
                    else None
                ),
                "generation_min_strict_grid_cell_delta": int(
                    args.grouped_generation_min_strict_grid_delta
                ),
                "crossfit_likelihood_immutable_baseline": bool(
                    args.enable_grouped_crossfit_likelihood_fallback
                ),
                "crossfit_likelihood_promotion_rule": (
                    "same_denominator_heldout_log_likelihood_ratio"
                    if bool(args.enable_grouped_crossfit_likelihood_fallback)
                    else None
                ),
                "crossfit_likelihood_min_mean_delta": float(
                    args.grouped_likelihood_min_mean_delta
                ),
                "crossfit_likelihood_min_effective_groups": int(
                    args.grouped_likelihood_min_effective_groups
                ),
                "crossfit_observability_gate": {
                    "min_information_matches": int(
                        args.grouped_observability_min_information_matches
                    ),
                    "min_translation_eigenvalue": float(
                        args.grouped_observability_min_translation_eigenvalue
                    ),
                    "max_translation_condition": float(
                        args.grouped_observability_max_translation_condition
                    ),
                    "max_joint_condition": float(
                        args.grouped_observability_max_joint_condition
                    ),
                    "min_bearing_span_deg": float(
                        args.grouped_observability_min_bearing_span_deg
                    ),
                    "min_depth_span_ratio": float(
                        args.grouped_observability_min_depth_span_ratio
                    ),
                    "min_xyz_second_ratio": float(
                        args.grouped_observability_min_xyz_second_ratio
                    ),
                    "min_xyz_third_ratio": float(
                        args.grouped_observability_min_xyz_third_ratio
                    ),
                },
                "immutable_baseline_generation_mode": (
                    "external_selected_pose_artifact_bit_exact"
                    if immutable_baseline_pose_source is not None
                    else "matched_single_ransac_current_run_DIAGNOSTIC_ONLY"
                    if bool(args.enable_grouped_crossfit_likelihood_fallback)
                    else str(args.grouped_immutable_baseline_generation_mode)
                ),
                "crossfit_immutable_pose_is_reestimated": bool(
                    args.enable_grouped_crossfit_likelihood_fallback
                    and immutable_baseline_pose_source is None
                ),
                "immutable_baseline_final_refine_acceptance_policy": (
                    None
                    if grouped_config is None
                    else str(grouped_config.final_refine_acceptance_policy)
                ),
                "optional_final_refine_acceptance_policy": (
                    None
                    if optional_grouped_config is None
                    else str(optional_grouped_config.final_refine_acceptance_policy)
                ),
                "immutable_baseline_final_refine_mode": (
                    None
                    if immutable_grouped_config is None
                    else str(immutable_grouped_config.final_refine_mode)
                ),
                "optional_final_refine_mode": (
                    None
                    if optional_grouped_config is None
                    else str(optional_grouped_config.final_refine_mode)
                ),
                "candidate_pose_evidence_version": str(
                    CANDIDATE_POSE_EVIDENCE_VERSION
                ),
                "optional_candidate_coordinate_refine": bool(
                    args.enable_optional_candidate_coordinate_refine
                ),
                "candidate_spatial_geometry_calibration_weight": float(
                    args.candidate_spatial_geometry_calibration_weight
                ),
                "candidate_spatial_utility_gate_weight": float(
                    args.candidate_spatial_utility_gate_weight
                ),
                **(
                    {}
                    if grouped_config is None
                    else {
                        **asdict(grouped_config),
                        "candidate_limits": list(grouped_config.candidate_limits),
                        "sampling_temperatures": list(
                            grouped_config.sampling_temperatures
                        ),
                        "fit_match_counts": list(grouped_config.fit_match_counts),
                        "ransac_thresholds_px": list(
                            grouped_config.ransac_thresholds_px
                        ),
                    }
                ),
            },
            "single_ransac": {
                "match_count": int(args.single_ransac_match_count),
                "selection_mode": str(args.single_ransac_selection_mode),
                "threshold_px": float(args.single_ransac_threshold_px),
                "iterations": int(args.single_ransac_iterations),
            },
        },
        "validation": {
            "trial_count": int(len(validation_trials)),
            "chosen": chosen,
            "trials": validation_trials,
        },
        "late_development_replay": {
            "enabled": bool(args.development_cross_block_audit),
            "trials": late_trials,
            "development_only": True,
            "used_for_policy_selection": False,
        },
        "train_hypothesis_export": {
            "enabled": bool(args.export_train_grouped_hypotheses),
            "geometry_probabilities_are_query_grouped_oof": bool(
                train_geometry_probabilities_requested
            ),
            "target_labels_written_only_after_pose_inference": True,
            "trials": train_trials,
        },
        "outputs": {
            "pose_rows": str(rows_path),
            "pose_rows_sha256": file_sha256_short(rows_path),
            "grouped_hypothesis_artifact": (
                None
                if grouped_hypothesis_artifact_path is None
                else str(grouped_hypothesis_artifact_path)
            ),
            "grouped_hypothesis_artifact_sha256": (
                None
                if grouped_hypothesis_artifact_path is None
                else file_sha256_short(grouped_hypothesis_artifact_path)
            ),
            "selected_pose_artifact": None,
            "selected_pose_artifact_sha256": None,
            "summary": str(output_dir / "summary.json"),
        },
    }
    if bool(args.export_selected_pose_artifact):
        if not selected_pose_export_rows:
            raise RuntimeError(
                "selected pose artifact was requested but no grouped poses were evaluated"
            )
        selected_pose_artifact_path = (
            output_dir / "selected_pose_inference_only_v1.npz"
        )
        selected_pose_source_manifest = {
            "stage": summary["stage"],
            "inputs": summary["inputs"],
            "protocol": summary["protocol"],
            "execution": summary["execution"],
            "config": summary["config"],
        }
        _write_selected_pose_artifact(
            selected_pose_artifact_path,
            selected_pose_export_rows,
            source_manifest=selected_pose_source_manifest,
        )
        summary["outputs"]["selected_pose_artifact"] = str(
            selected_pose_artifact_path
        )
        summary["outputs"]["selected_pose_artifact_sha256"] = file_sha256_short(
            selected_pose_artifact_path
        )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
