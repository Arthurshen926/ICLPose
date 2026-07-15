"""Inference-only multi-hypothesis PnP with held-out geometric verification.

The verifier deliberately has no ground-truth pose input. A fixed subset of
the correspondences is withheld from hypothesis fitting and is used only to
rank poses by reprojection consistency, image coverage, and cheirality.
"""

from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import product
from typing import Callable, Optional, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.pose_safe_selection import (
    resolve_global_query_track_assignment,
    resolve_pose_match_conflicts,
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.localization.candidate_pose_evidence import (
    CANDIDATE_POSE_EVIDENCE_VERSION,
    candidate_pose_evidence,
    measurement_reliability,
    project_candidate_xyz,
)
from feature_extract.vfm.localization.latent_correspondence_pnp import (
    LatentEMConfig,
    refine_pose_latent_em,
)
from feature_extract.vfm.query_to_3d_matching import (
    PnPResult,
    QueryTo3DMatch,
    camera_matrix_and_distortion,
    estimate_pose_pnp_fixed,
    estimate_pose_pnp_fixed_hypotheses,
    estimate_pose_pnp_fixed_robust,
    estimate_pose_pnp_ransac,
    match_reprojection_errors,
)


GROUPED_HYPOTHESIS_SELECTION_POLICIES = (
    "fixed_posterior_likelihood_only",
    "fixed_posterior_median_DIAGNOSTIC_ONLY",
    "fixed_posterior_trimmed_mean_10_DIAGNOSTIC_ONLY",
    "fixed_posterior_worst_quartile_mean_DIAGNOSTIC_ONLY",
    "fixed_posterior_lcb95_DIAGNOSTIC_ONLY",
    "fixed_posterior_spatial_mom_2x2_DIAGNOSTIC_ONLY",
    "legacy_self_consistency_DIAGNOSTIC_ONLY",
)

DIAGNOSTIC_GROUPED_HYPOTHESIS_SELECTION_POLICIES = frozenset(
    policy
    for policy in GROUPED_HYPOTHESIS_SELECTION_POLICIES
    if policy.endswith("_DIAGNOSTIC_ONLY")
)


@dataclass(frozen=True)
class VerifiedPnPConfig:
    fit_match_counts: tuple[int, ...] = (24, 32, 48, 64)
    selection_modes: tuple[str, ...] = (
        "score_topk",
        "spatial_round_robin",
        "geometry_diverse",
    )
    ransac_thresholds_px: tuple[float, ...] = (2.0, 4.0, 8.0)
    rng_seed_offsets: tuple[int, ...] = (0, 1)
    ransac_iterations: int = 3000
    holdout_folds: int = 4
    holdout_fold: int = 0
    final_audit_fold: int | None = 1
    grid_rows: int = 4
    grid_cols: int = 4
    geometry_prefilter_multiplier: int = 3
    verification_strict_px: float = 2.0
    verification_loose_px: float = 5.0
    final_consensus_px: float = 4.0
    final_refine_f_scale_px: float = 2.0
    min_final_inliers: int = 6
    enable_final_refine: bool = False
    candidate_pool_residual_sigma_px: float = 2.0
    candidate_pool_hard_threshold_px: float = 8.0
    candidate_pool_descriptor_rank_weight: float = 0.02
    candidate_pool_refine_iterations: int = 2
    measurement_verified_threshold: float = 0.64
    measurement_verified_min_matches: int = 8
    measurement_verified_min_grid_cells: int = 6

    def __post_init__(self) -> None:
        if not self.fit_match_counts or min(self.fit_match_counts) < 4:
            raise ValueError("fit_match_counts must contain values >= 4")
        supported = {
            "score_topk",
            "spatial_round_robin",
            "geometry_diverse",
            "measurement_verified",
            "measurement_verified_refined",
        }
        if not self.selection_modes or set(self.selection_modes) - supported:
            raise ValueError("unsupported hypothesis selection mode")
        if not self.ransac_thresholds_px or min(self.ransac_thresholds_px) <= 0.0:
            raise ValueError("ransac thresholds must be positive")
        if not self.rng_seed_offsets:
            raise ValueError("at least one RNG seed offset is required")
        if int(self.ransac_iterations) <= 0:
            raise ValueError("ransac_iterations must be positive")
        if int(self.holdout_folds) < 2:
            raise ValueError("holdout_folds must be at least two")
        if not 0 <= int(self.holdout_fold) < int(self.holdout_folds):
            raise ValueError("holdout_fold is outside holdout_folds")
        if self.final_audit_fold is not None:
            if int(self.holdout_folds) < 3:
                raise ValueError("final audit requires at least three holdout folds")
            if not 0 <= int(self.final_audit_fold) < int(self.holdout_folds):
                raise ValueError("final_audit_fold is outside holdout_folds")
            if int(self.final_audit_fold) == int(self.holdout_fold):
                raise ValueError("rank verification and final audit folds must differ")
        if int(self.grid_rows) <= 0 or int(self.grid_cols) <= 0:
            raise ValueError("grid dimensions must be positive")
        if int(self.geometry_prefilter_multiplier) <= 0:
            raise ValueError("geometry_prefilter_multiplier must be positive")
        if not 0.0 < float(self.verification_strict_px) <= float(
            self.verification_loose_px
        ):
            raise ValueError("verification thresholds are inconsistent")
        if float(self.final_consensus_px) <= 0.0:
            raise ValueError("final_consensus_px must be positive")
        if float(self.final_refine_f_scale_px) <= 0.0:
            raise ValueError("final_refine_f_scale_px must be positive")
        if int(self.min_final_inliers) < 4:
            raise ValueError("min_final_inliers must be at least four")
        if float(self.candidate_pool_residual_sigma_px) <= 0.0:
            raise ValueError("candidate-pool residual sigma must be positive")
        if float(self.candidate_pool_hard_threshold_px) <= 0.0:
            raise ValueError("candidate-pool hard threshold must be positive")
        if float(self.candidate_pool_descriptor_rank_weight) < 0.0:
            raise ValueError("candidate-pool descriptor rank weight must be non-negative")
        if int(self.candidate_pool_refine_iterations) <= 0:
            raise ValueError("candidate-pool refine iterations must be positive")
        if not 0.0 <= float(self.measurement_verified_threshold) <= 1.0:
            raise ValueError("measurement verified threshold must be in [0, 1]")
        if int(self.measurement_verified_min_matches) < 4:
            raise ValueError("measurement verified minimum matches must be at least four")
        if int(self.measurement_verified_min_grid_cells) <= 0:
            raise ValueError("measurement verified minimum grid cells must be positive")


@dataclass(frozen=True)
class GroupedProsacProfile:
    """One deterministic hypothesis-generation profile in an ensemble."""

    name: str
    hypotheses_per_limit: int
    minimal_set_sizes: tuple[int, ...]
    candidate_probability_power: float = 1.0
    candidate_uniform_mix: float = 0.0
    local_optimization: bool = False
    use_spatial_modes: bool = False

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("grouped PROSAC profile name cannot be empty")
        if int(self.hypotheses_per_limit) <= 0:
            raise ValueError("grouped PROSAC profile hypothesis count must be positive")
        sizes = tuple(int(value) for value in self.minimal_set_sizes)
        if not sizes or tuple(sorted(set(sizes))) != sizes:
            raise ValueError(
                "grouped PROSAC profile minimal sizes must be strictly increasing"
            )
        if not all(4 <= value <= 8 for value in sizes):
            raise ValueError("grouped PROSAC profile minimal sizes must be in [4, 8]")
        if float(self.candidate_probability_power) <= 0.0:
            raise ValueError(
                "grouped PROSAC profile candidate probability power must be positive"
            )
        if not 0.0 <= float(self.candidate_uniform_mix) <= 1.0:
            raise ValueError(
                "grouped PROSAC profile candidate uniform mix must be in [0, 1]"
            )


@dataclass(frozen=True)
class GroupedCandidatePnPConfig:
    """Progressive top-L hypothesis generation without collapsing query groups."""

    candidate_limits: tuple[int, ...] = (1, 3, 5, 10, 20)
    samples_per_limit: int = 2
    sampling_temperatures: tuple[float, ...] = (0.5, 1.0)
    fit_match_counts: tuple[int, ...] = (32, 64)
    ransac_thresholds_px: tuple[float, ...] = (2.0, 4.0)
    ransac_iterations: int = 2000
    holdout_folds: int = 4
    verification_fold: int = 0
    verification_fold_count: int = 1
    final_audit_fold: int = 1
    crossfit_mode: str = "token_spatial"
    crossfit_role_assignment: str = "fixed"
    crossfit_spatial_fold_policy: str = "legacy_local_modulo"
    crossfit_maplet_voxel_size_m: float = 0.5
    independent_shortlist_pool: bool = False
    grid_rows: int = 4
    grid_cols: int = 4
    min_fit_matches: int = 8
    min_fit_grid_cells: int = 4
    min_xyz_second_singular_ratio: float = 1e-3
    verification_strict_px: float = 2.0
    verification_loose_px: float = 5.0
    candidate_pool_residual_sigma_px: float = 2.0
    candidate_pose_outlier_likelihood: float = 1e-3
    candidate_pose_null_likelihood: float = 1e-3
    candidate_pose_relation_neighbor_k: int = 0
    candidate_pose_relation_sigma_px: float = 4.0
    candidate_pose_relation_outlier_likelihood: float = 1e-3
    candidate_pool_hard_threshold_px: float = 8.0
    candidate_pool_descriptor_rank_weight: float = 0.02
    final_consensus_px: float = 4.0
    final_refine_f_scale_px: float = 2.0
    min_final_inliers: int = 6
    enable_final_refine: bool = True
    final_refine_mode: str = "auto"
    final_refine_acceptance_policy: str = "fixed_posterior_likelihood_gain"
    hypothesis_selection_policy: str = "fixed_posterior_likelihood_only"
    enable_candidate_coordinate_refine: bool = False
    min_candidate_coordinate_updates: int = 4
    min_candidate_coordinate_update_grid_cells: int = 2
    generation_mode: str = "assignment_ransac"
    prosac_hypotheses_per_limit: int = 64
    prosac_minimal_set_size: int = 6
    prosac_minimal_set_sizes: tuple[int, ...] = ()
    prosac_group_probability_power: float = 1.0
    prosac_candidate_probability_power: float = 0.5
    prosac_max_sample_attempts: int = 64
    prosac_min_bearing_span_deg: float = 3.0
    prosac_min_translation_information_eigenvalue: float = 0.0
    prosac_max_translation_information_condition: float = 0.0
    prosac_max_joint_information_condition: float = 0.0
    prosac_min_depth_span_ratio: float = 0.0
    prosac_min_xyz_third_singular_ratio: float = 0.0
    prosac_observability_evidence_mode: str = "minimal_sample"
    prosac_local_optimization: bool = True
    prosac_local_consensus_px: float = 4.0
    prosac_local_min_matches: int = 8
    prosac_use_spatial_modes: bool = False
    prosac_verification_top_k: int = 0
    prosac_shortlist_evidence_mode: str = "full_spatial"
    prosac_spatial_rescore_top_k: int = 0
    prosac_shortlist_selection_mode: str = "score_topk"
    prosac_shortlist_diverse_count: int = 0
    prosac_shortlist_min_per_profile: int = 0
    prosac_shortlist_translation_diversity_m: float = 0.05
    prosac_shortlist_rotation_diversity_deg: float = 0.5
    prosac_profiles: tuple[GroupedProsacProfile, ...] = ()
    latent_em_enabled: bool = False
    latent_em_seed_count: int = 16
    latent_em_seed_min_per_profile: int = 0
    latent_em_seed_evidence_mode: str = "crossfit_shortlist"
    latent_em_translation_diversity_m: float = 0.05
    latent_em_rotation_diversity_deg: float = 0.5
    latent_em_config: LatentEMConfig = LatentEMConfig()

    def __post_init__(self) -> None:
        if not self.candidate_limits or min(self.candidate_limits) <= 0:
            raise ValueError("candidate_limits must be positive")
        if tuple(sorted(set(self.candidate_limits))) != self.candidate_limits:
            raise ValueError("candidate_limits must be strictly increasing")
        if int(self.samples_per_limit) <= 0:
            raise ValueError("samples_per_limit must be positive")
        if not self.sampling_temperatures or min(self.sampling_temperatures) <= 0.0:
            raise ValueError("sampling temperatures must be positive")
        if not self.fit_match_counts or min(self.fit_match_counts) < 4:
            raise ValueError("fit_match_counts must contain values >= 4")
        if not self.ransac_thresholds_px or min(self.ransac_thresholds_px) <= 0.0:
            raise ValueError("RANSAC thresholds must be positive")
        if int(self.ransac_iterations) <= 0:
            raise ValueError("ransac_iterations must be positive")
        if int(self.holdout_folds) < 3:
            raise ValueError("grouped candidate PnP requires at least three folds")
        folds = int(self.holdout_folds)
        if not 0 <= int(self.verification_fold) < folds:
            raise ValueError("verification_fold is outside holdout_folds")
        if int(self.verification_fold_count) <= 0:
            raise ValueError("verification_fold_count must be positive")
        verification_folds = tuple(
            range(
                int(self.verification_fold),
                int(self.verification_fold) + int(self.verification_fold_count),
            )
        )
        if not verification_folds or max(verification_folds) >= folds:
            raise ValueError("verification folds are outside holdout_folds")
        if not 0 <= int(self.final_audit_fold) < folds:
            raise ValueError("final_audit_fold is outside holdout_folds")
        if int(self.final_audit_fold) in verification_folds:
            raise ValueError("verification and final audit folds must differ")
        if len(verification_folds) + 1 >= folds:
            raise ValueError("grouped candidate PnP requires at least one fit fold")
        if str(self.crossfit_mode) not in {
            "token_spatial",
            "token_spatial_track_purged",
            "token_spatial_track_maplet_purged",
            "token_track_component",
            "token_track_voxel_component",
            "token_track_maplet_component",
        }:
            raise ValueError("unsupported grouped cross-fit mode")
        if str(self.crossfit_role_assignment) not in {
            "fixed",
            "adaptive_balanced",
        }:
            raise ValueError("unsupported grouped cross-fit role assignment")
        if str(self.crossfit_spatial_fold_policy) not in {
            "legacy_local_modulo",
            "cell_rotated_balanced",
        }:
            raise ValueError("unsupported grouped spatial fold policy")
        if float(self.crossfit_maplet_voxel_size_m) <= 0.0:
            raise ValueError("cross-fit maplet voxel size must be positive")
        if bool(self.independent_shortlist_pool):
            if (
                str(self.crossfit_mode)
                != "token_spatial_track_maplet_purged"
            ):
                raise ValueError(
                    "independent shortlist pool requires strict token/track/maplet purging"
                )
            if int(self.verification_fold_count) < 2:
                raise ValueError(
                    "independent shortlist pool requires at least two rank folds"
                )
            if str(self.crossfit_spatial_fold_policy) != "cell_rotated_balanced":
                raise ValueError(
                    "independent shortlist pool requires balanced spatial folds"
                )
        if int(self.grid_rows) <= 0 or int(self.grid_cols) <= 0:
            raise ValueError("grid dimensions must be positive")
        if int(self.min_fit_matches) < 4:
            raise ValueError("min_fit_matches must be at least four")
        if int(self.min_fit_grid_cells) <= 0:
            raise ValueError("min_fit_grid_cells must be positive")
        if not 0.0 <= float(self.min_xyz_second_singular_ratio) < 1.0:
            raise ValueError("min_xyz_second_singular_ratio must be in [0, 1)")
        if not 0.0 < float(self.verification_strict_px) <= float(
            self.verification_loose_px
        ):
            raise ValueError("verification thresholds are inconsistent")
        if float(self.candidate_pool_residual_sigma_px) <= 0.0:
            raise ValueError("candidate-pool residual sigma must be positive")
        if not 0.0 < float(self.candidate_pose_outlier_likelihood) <= 1.0:
            raise ValueError("candidate pose outlier likelihood must be in (0, 1]")
        if not 0.0 < float(self.candidate_pose_null_likelihood) <= 1.0:
            raise ValueError("candidate pose null likelihood must be in (0, 1]")
        if int(self.candidate_pose_relation_neighbor_k) < 0:
            raise ValueError("candidate pose relation neighbor count must be non-negative")
        if float(self.candidate_pose_relation_sigma_px) <= 0.0:
            raise ValueError("candidate pose relation sigma must be positive")
        if not 0.0 < float(
            self.candidate_pose_relation_outlier_likelihood
        ) <= 1.0:
            raise ValueError(
                "candidate pose relation outlier likelihood must be in (0, 1]"
            )
        if float(self.candidate_pool_hard_threshold_px) <= 0.0:
            raise ValueError("candidate-pool hard threshold must be positive")
        if float(self.candidate_pool_descriptor_rank_weight) < 0.0:
            raise ValueError("candidate-pool descriptor rank weight must be non-negative")
        if float(self.final_consensus_px) <= 0.0:
            raise ValueError("final_consensus_px must be positive")
        if float(self.final_refine_f_scale_px) <= 0.0:
            raise ValueError("final_refine_f_scale_px must be positive")
        if int(self.min_final_inliers) < 4:
            raise ValueError("min_final_inliers must be at least four")
        if str(self.final_refine_mode) not in {
            "auto",
            "hard_assignment",
            "latent_em",
        }:
            raise ValueError("unsupported grouped final refine mode")
        if str(self.final_refine_mode) == "latent_em" and not bool(
            self.latent_em_enabled
        ):
            raise ValueError("latent final refine requires latent EM")
        if str(self.final_refine_acceptance_policy) not in {
            "fixed_posterior_likelihood_gain",
            "legacy_rank_key",
            "strict_count_gain_with_grid_nondecrease",
        }:
            raise ValueError("unsupported grouped final refine acceptance policy")
        if (
            str(self.hypothesis_selection_policy)
            not in GROUPED_HYPOTHESIS_SELECTION_POLICIES
        ):
            raise ValueError("unsupported grouped hypothesis selection policy")
        if int(self.min_candidate_coordinate_updates) < 4:
            raise ValueError("candidate coordinate refinement requires at least four updates")
        if int(self.min_candidate_coordinate_update_grid_cells) <= 0:
            raise ValueError("candidate coordinate update grid coverage must be positive")
        if str(self.generation_mode) not in {
            "assignment_ransac",
            "grouped_prosac",
            "assignment_plus_grouped_prosac",
        }:
            raise ValueError("unsupported grouped hypothesis generation mode")
        if int(self.prosac_hypotheses_per_limit) <= 0:
            raise ValueError("PROSAC hypotheses per candidate limit must be positive")
        if not 4 <= int(self.prosac_minimal_set_size) <= 8:
            raise ValueError("PROSAC minimal set size must be in [4, 8]")
        if self.prosac_minimal_set_sizes:
            if tuple(sorted(set(self.prosac_minimal_set_sizes))) != tuple(
                self.prosac_minimal_set_sizes
            ):
                raise ValueError(
                    "PROSAC minimal set sizes must be strictly increasing"
                )
            if not all(
                4 <= int(value) <= 8 for value in self.prosac_minimal_set_sizes
            ):
                raise ValueError("PROSAC minimal set sizes must be in [4, 8]")
        if float(self.prosac_group_probability_power) < 0.0:
            raise ValueError("PROSAC group probability power must be non-negative")
        if float(self.prosac_candidate_probability_power) <= 0.0:
            raise ValueError("PROSAC candidate probability power must be positive")
        if int(self.prosac_max_sample_attempts) <= 0:
            raise ValueError("PROSAC sample attempts must be positive")
        if float(self.prosac_min_bearing_span_deg) < 0.0:
            raise ValueError("PROSAC bearing span must be non-negative")
        prosac_observability_thresholds = (
            float(self.prosac_min_translation_information_eigenvalue),
            float(self.prosac_max_translation_information_condition),
            float(self.prosac_max_joint_information_condition),
            float(self.prosac_min_depth_span_ratio),
            float(self.prosac_min_xyz_third_singular_ratio),
        )
        if any(
            not np.isfinite(value) or value < 0.0
            for value in prosac_observability_thresholds
        ):
            raise ValueError("PROSAC observability thresholds must be finite and non-negative")
        if str(self.prosac_observability_evidence_mode) not in {
            "minimal_sample",
            "resolved_fit_consensus",
        }:
            raise ValueError("unsupported PROSAC observability evidence mode")
        if float(self.prosac_local_consensus_px) <= 0.0:
            raise ValueError("PROSAC local consensus threshold must be positive")
        if int(self.prosac_local_min_matches) < 4:
            raise ValueError("PROSAC local optimization requires at least four matches")
        if int(self.prosac_verification_top_k) < 0:
            raise ValueError("PROSAC verification top-k must be non-negative")
        if str(self.prosac_shortlist_evidence_mode) not in {
            "full_spatial",
            "base_coordinate_then_full_spatial",
        }:
            raise ValueError("unsupported PROSAC shortlist evidence mode")
        if int(self.prosac_spatial_rescore_top_k) < 0:
            raise ValueError("PROSAC spatial rescore top-k must be non-negative")
        if str(self.prosac_shortlist_selection_mode) not in {
            "score_topk",
            "profile_pose_diverse",
        }:
            raise ValueError("unsupported PROSAC shortlist selection mode")
        if int(self.prosac_shortlist_diverse_count) < 0:
            raise ValueError("PROSAC shortlist diverse count must be non-negative")
        if int(self.prosac_shortlist_min_per_profile) < 0:
            raise ValueError(
                "PROSAC shortlist minimum per profile must be non-negative"
            )
        if (
            float(self.prosac_shortlist_translation_diversity_m) < 0.0
            or float(self.prosac_shortlist_rotation_diversity_deg) < 0.0
        ):
            raise ValueError(
                "PROSAC shortlist pose diversity thresholds must be non-negative"
            )
        if str(self.prosac_shortlist_selection_mode) == "score_topk" and (
            int(self.prosac_shortlist_diverse_count) > 0
            or int(self.prosac_shortlist_min_per_profile) > 0
        ):
            raise ValueError(
                "PROSAC shortlist diversity parameters require profile_pose_diverse"
            )
        if str(self.prosac_shortlist_selection_mode) == "profile_pose_diverse":
            if int(self.prosac_verification_top_k) <= 0:
                raise ValueError(
                    "diverse PROSAC shortlist requires a finite verification top-k"
                )
            if not 0 < int(self.prosac_shortlist_diverse_count) <= int(
                self.prosac_verification_top_k
            ):
                raise ValueError(
                    "PROSAC shortlist diverse count must be in verification top-k"
                )
        if int(self.prosac_spatial_rescore_top_k) > 0:
            if (
                str(self.prosac_shortlist_evidence_mode)
                != "base_coordinate_then_full_spatial"
            ):
                raise ValueError(
                    "PROSAC spatial rescore requires selective shortlist evidence"
                )
            if int(self.prosac_verification_top_k) <= 0:
                raise ValueError(
                    "PROSAC spatial rescore requires a finite verification top-k"
                )
            if (
                int(self.prosac_verification_top_k) > 0
                and int(self.prosac_spatial_rescore_top_k)
                < int(self.prosac_verification_top_k)
            ):
                raise ValueError(
                    "PROSAC spatial rescore top-k must cover the verification top-k"
                )
        profile_names = [str(profile.name) for profile in self.prosac_profiles]
        if len(set(profile_names)) != len(profile_names):
            raise ValueError("grouped PROSAC profile names must be unique")
        if int(self.latent_em_seed_count) <= 0:
            raise ValueError("latent EM seed count must be positive")
        if int(self.latent_em_seed_min_per_profile) < 0:
            raise ValueError(
                "latent EM seed minimum per profile must be non-negative"
            )
        if str(self.latent_em_seed_evidence_mode) not in {
            "crossfit_shortlist",
            "fit_pool_DIAGNOSTIC_ONLY",
        }:
            raise ValueError("unsupported latent EM seed evidence mode")
        if float(self.latent_em_translation_diversity_m) < 0.0:
            raise ValueError("latent EM translation diversity must be non-negative")
        if float(self.latent_em_rotation_diversity_deg) < 0.0:
            raise ValueError("latent EM rotation diversity must be non-negative")
        if bool(self.latent_em_enabled):
            if not np.isclose(
                float(self.latent_em_config.outlier_likelihood),
                float(self.candidate_pose_outlier_likelihood),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    "latent EM and fixed verification must share outlier likelihood"
                )
            if not np.isclose(
                float(self.latent_em_config.null_likelihood),
                float(self.candidate_pose_null_likelihood),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    "latent EM and fixed verification must share null likelihood"
                )


@dataclass(frozen=True)
class CandidateSpatialLikelihood:
    """Candidate/view-specific local likelihood maps aligned to a top-L pool."""

    offsets_xy: np.ndarray
    local_log_probabilities: np.ndarray
    view_probabilities: np.ndarray
    dustbin_probabilities: np.ndarray
    valid_mask: np.ndarray
    log_evidence_weight: float = 1.0
    support_camera_centers: np.ndarray | None = None
    pose_view_geometry_sigma_deg: float = 0.0

    def __post_init__(self) -> None:
        offsets = np.array(self.offsets_xy, dtype=np.float64, copy=True).reshape(-1, 2)
        log_probabilities = np.array(
            self.local_log_probabilities, dtype=np.float64, copy=True
        )
        view_probabilities = np.array(
            self.view_probabilities, dtype=np.float64, copy=True
        )
        dustbin_probabilities = np.array(
            self.dustbin_probabilities, dtype=np.float64, copy=True
        )
        valid = np.array(self.valid_mask, dtype=bool, copy=True)
        log_evidence_weight = float(self.log_evidence_weight)
        support_camera_centers = (
            None
            if self.support_camera_centers is None
            else np.array(
                self.support_camera_centers, dtype=np.float64, copy=True
            )
        )
        pose_view_geometry_sigma_deg = float(
            self.pose_view_geometry_sigma_deg
        )
        if log_probabilities.ndim != 4:
            raise ValueError(
                "spatial log probabilities must have shape (N, L, V, K)"
            )
        if log_probabilities.shape[3] != len(offsets):
            raise ValueError("spatial likelihood bins and offsets differ")
        expected = log_probabilities.shape[:3]
        if (
            view_probabilities.shape != expected
            or dustbin_probabilities.shape != expected
            or valid.shape != expected
        ):
            raise ValueError("spatial likelihood view arrays are misaligned")
        if support_camera_centers is not None:
            if support_camera_centers.shape != (*expected, 3):
                raise ValueError(
                    "spatial support camera centers must have shape (N,L,V,3)"
                )
            if np.any(
                ~np.all(np.isfinite(support_camera_centers), axis=3) & valid
            ):
                raise ValueError(
                    "valid spatial support camera centers must be finite"
                )
        if (
            not np.isfinite(pose_view_geometry_sigma_deg)
            or pose_view_geometry_sigma_deg < 0.0
        ):
            raise ValueError(
                "pose-view geometry sigma must be finite and non-negative"
            )
        if (
            pose_view_geometry_sigma_deg > 0.0
            and support_camera_centers is None
        ):
            raise ValueError(
                "pose-view geometry weighting requires support camera centers"
            )
        if np.any(~np.isfinite(offsets)):
            raise ValueError("spatial likelihood offsets must be finite")
        if np.any(~np.isfinite(view_probabilities[valid])) or np.any(
            view_probabilities[valid] < 0.0
        ):
            raise ValueError("spatial view probabilities must be finite and non-negative")
        available_view_mass = np.sum(
            np.where(valid, view_probabilities, 0.0), axis=2
        )
        if np.any(available_view_mass > 1.0 + 2e-5):
            raise ValueError(
                "available spatial support-view probability mass exceeds one"
            )
        if np.any(~np.isfinite(dustbin_probabilities[valid])) or np.any(
            (dustbin_probabilities[valid] < 0.0)
            | (dustbin_probabilities[valid] > 1.0)
        ):
            raise ValueError("spatial dustbin probabilities must be in [0, 1]")
        if np.any(~np.all(np.isfinite(log_probabilities), axis=3) & valid):
            raise ValueError("valid spatial likelihood maps must be finite")
        if not 0.0 <= log_evidence_weight <= 1.0:
            raise ValueError("spatial log evidence weight must be in [0, 1]")
        xs = np.unique(offsets[:, 0])
        ys = np.unique(offsets[:, 1])
        if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) != len(offsets):
            raise ValueError("spatial offsets must form a complete regular 2D grid")
        if not np.allclose(np.diff(xs), np.diff(xs)[0], rtol=0.0, atol=1e-8) or not np.allclose(
            np.diff(ys), np.diff(ys)[0], rtol=0.0, atol=1e-8
        ):
            raise ValueError("spatial likelihood grid spacing must be regular")
        grid_order = np.lexsort((offsets[:, 0], offsets[:, 1]))
        expected_offsets = np.stack(
            np.meshgrid(xs, ys, indexing="xy"), axis=-1
        ).reshape(-1, 2)
        if not np.allclose(offsets[grid_order], expected_offsets, rtol=0.0, atol=1e-8):
            raise ValueError("spatial likelihood offset grid is incomplete")
        object.__setattr__(self, "offsets_xy", offsets)
        object.__setattr__(self, "local_log_probabilities", log_probabilities)
        object.__setattr__(self, "view_probabilities", view_probabilities)
        object.__setattr__(self, "dustbin_probabilities", dustbin_probabilities)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "log_evidence_weight", log_evidence_weight)
        object.__setattr__(
            self, "support_camera_centers", support_camera_centers
        )
        object.__setattr__(
            self,
            "pose_view_geometry_sigma_deg",
            pose_view_geometry_sigma_deg,
        )
        object.__setattr__(self, "_grid_x", xs)
        object.__setattr__(self, "_grid_y", ys)
        object.__setattr__(self, "_grid_order", grid_order)
        ordered_log_probabilities = log_probabilities[:, :, :, grid_order]
        ordered_log_probabilities = ordered_log_probabilities - np.max(
            ordered_log_probabilities, axis=3, keepdims=True
        )
        probability_maps = np.exp(ordered_log_probabilities)
        probability_maps /= np.maximum(
            np.sum(probability_maps, axis=3, keepdims=True), 1e-12
        )
        probability_maps = probability_maps.reshape(
            *valid.shape, len(ys), len(xs)
        )
        probability_maps = probability_maps.astype(np.float32)
        for array in (
            offsets,
            log_probabilities,
            view_probabilities,
            dustbin_probabilities,
            valid,
            xs,
            ys,
            grid_order,
            probability_maps,
        ):
            array.setflags(write=False)
        if support_camera_centers is not None:
            support_camera_centers.setflags(write=False)
        object.__setattr__(
            self, "_probability_maps", probability_maps
        )

    @property
    def candidate_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.valid_mask.shape[:2])

    def subset(self, rows: np.ndarray) -> "CandidateSpatialLikelihood":
        indices = np.asarray(rows, dtype=np.int64).reshape(-1)
        return CandidateSpatialLikelihood(
            self.offsets_xy,
            self.local_log_probabilities[indices],
            self.view_probabilities[indices],
            self.dustbin_probabilities[indices],
            self.valid_mask[indices],
            self.log_evidence_weight,
            (
                None
                if self.support_camera_centers is None
                else self.support_camera_centers[indices]
            ),
            self.pose_view_geometry_sigma_deg,
        )

    def mask_candidates(self, keep_mask: np.ndarray) -> "CandidateSpatialLikelihood":
        keep = np.array(keep_mask, dtype=bool, copy=True)
        if keep.shape != self.candidate_shape:
            raise ValueError("spatial candidate mask has incompatible shape")
        return CandidateSpatialLikelihood(
            self.offsets_xy,
            self.local_log_probabilities,
            self.view_probabilities,
            self.dustbin_probabilities,
            self.valid_mask & keep[:, :, None],
            self.log_evidence_weight,
            self.support_camera_centers,
            self.pose_view_geometry_sigma_deg,
        )


@dataclass(frozen=True)
class PoseVerificationCandidatePool:
    """Top-L 3D candidates for a set of unique 2D query measurements."""

    token_indices: np.ndarray
    xy: np.ndarray
    track_ids: np.ndarray
    prototype_ids: np.ndarray
    xyz: np.ndarray
    descriptor_scores: np.ndarray
    valid_mask: np.ndarray
    measurement_geometry_probabilities: np.ndarray | None = None
    measurement_verification_threshold: float = 0.5
    null_scores: np.ndarray | None = None
    spatial_likelihood: CandidateSpatialLikelihood | None = None
    identity_prior_temperature: float = 1.0
    geometry_prior_mix_weight: float = 0.0
    spatial_geometry_calibration_weight: float = 0.0
    geometry_generation_mix_weight: float = 0.0
    candidate_update_probabilities: np.ndarray | None = None
    candidate_refined_xy: np.ndarray | None = None
    candidate_update_threshold: float = 0.5
    spatial_utility_gate_weight: float = 0.0
    maplet_cluster_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        explicit_null = self.null_scores is not None
        tokens = np.array(self.token_indices, dtype=np.int64, copy=True).reshape(-1)
        xy = np.array(self.xy, dtype=np.float64, copy=True).reshape(-1, 2)
        tracks = np.array(self.track_ids, dtype=np.int64, copy=True)
        prototypes = np.array(self.prototype_ids, dtype=np.int64, copy=True)
        xyz = np.array(self.xyz, dtype=np.float64, copy=True)
        scores = np.array(self.descriptor_scores, dtype=np.float64, copy=True)
        valid = np.array(self.valid_mask, dtype=bool, copy=True)
        if tracks.ndim != 2 or tracks.shape[0] != len(tokens):
            raise ValueError("candidate track ids must have shape (N, L)")
        if (
            len(xy) != len(tokens)
            or prototypes.shape != tracks.shape
            or xyz.shape != (*tracks.shape, 3)
            or scores.shape != tracks.shape
            or valid.shape != tracks.shape
        ):
            raise ValueError("candidate-pool arrays have incompatible shapes")
        if len(np.unique(tokens)) != len(tokens):
            raise ValueError("candidate-pool token indices must be unique")
        valid &= (tracks >= 0) & np.isfinite(scores) & np.all(np.isfinite(xyz), axis=2)
        measurement = (
            np.full(tracks.shape, np.nan, dtype=np.float64)
            if self.measurement_geometry_probabilities is None
            else np.array(
                self.measurement_geometry_probabilities, dtype=np.float64, copy=True
            )
        )
        if measurement.shape != tracks.shape:
            raise ValueError(
                "measurement geometry probabilities must match candidate-pool shape"
            )
        finite_measurement = np.isfinite(measurement)
        if np.any(
            (measurement[finite_measurement] < 0.0)
            | (measurement[finite_measurement] > 1.0)
        ):
            raise ValueError("measurement geometry probabilities must be in [0, 1]")
        measurement[~valid] = np.nan
        threshold = float(self.measurement_verification_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("measurement verification threshold must be in [0, 1]")
        null_scores = (
            np.zeros((len(tokens),), dtype=np.float64)
            if self.null_scores is None
            else np.array(
                self.null_scores, dtype=np.float64, copy=True
            ).reshape(-1)
        )
        if null_scores.shape != (len(tokens),):
            raise ValueError("null scores must contain one value per query group")
        if np.any(~np.isfinite(null_scores)) or np.any(null_scores < 0.0):
            raise ValueError("null scores must be finite and non-negative")
        if explicit_null:
            candidate_mass = np.where(valid, scores, 0.0)
            if np.any(candidate_mass < 0.0) or np.any(candidate_mass > 1.0):
                raise ValueError(
                    "explicit-null candidate scores must be probabilities in [0, 1]"
                )
            if np.any(null_scores > 1.0):
                raise ValueError("explicit null scores must be probabilities in [0, 1]")
            joint_mass = np.sum(candidate_mass, axis=1) + null_scores
            if np.any(joint_mass <= 0.0) or np.any(np.abs(joint_mass - 1.0) > 2e-4):
                raise ValueError(
                    "explicit-null candidate and null probabilities must sum to one"
                )
            # Joint-softmax artifacts are commonly float32. Canonicalize only
            # their tolerated simplex drift without reallocating unknown mass.
            scores = np.where(valid, scores / joint_mass[:, None], scores)
            null_scores = null_scores / joint_mass
        spatial_likelihood = self.spatial_likelihood
        if (
            spatial_likelihood is not None
            and spatial_likelihood.candidate_shape != tracks.shape
        ):
            raise ValueError("spatial likelihood and candidate pool shapes differ")
        identity_prior_temperature = float(self.identity_prior_temperature)
        if (
            not np.isfinite(identity_prior_temperature)
            or identity_prior_temperature <= 0.0
        ):
            raise ValueError("identity prior temperature must be finite and positive")
        geometry_prior_mix_weight = float(self.geometry_prior_mix_weight)
        if not 0.0 <= geometry_prior_mix_weight <= 1.0:
            raise ValueError("geometry prior mix weight must be in [0, 1]")
        spatial_geometry_calibration_weight = float(
            self.spatial_geometry_calibration_weight
        )
        if not 0.0 <= spatial_geometry_calibration_weight <= 1.0:
            raise ValueError(
                "spatial geometry calibration weight must be in [0, 1]"
            )
        geometry_generation_mix_weight = float(
            self.geometry_generation_mix_weight
        )
        if not 0.0 <= geometry_generation_mix_weight <= 1.0:
            raise ValueError("geometry generation mix weight must be in [0, 1]")
        if (self.candidate_update_probabilities is None) != (
            self.candidate_refined_xy is None
        ):
            raise ValueError(
                "candidate update probabilities and refined coordinates must be paired"
            )
        update_probability = (
            np.full(tracks.shape, np.nan, dtype=np.float64)
            if self.candidate_update_probabilities is None
            else np.array(
                self.candidate_update_probabilities, dtype=np.float64, copy=True
            )
        )
        refined_xy = (
            np.full((*tracks.shape, 2), np.nan, dtype=np.float64)
            if self.candidate_refined_xy is None
            else np.array(self.candidate_refined_xy, dtype=np.float64, copy=True)
        )
        if update_probability.shape != tracks.shape or refined_xy.shape != (
            *tracks.shape,
            2,
        ):
            raise ValueError("candidate coordinate update arrays have incompatible shapes")
        finite_update = np.isfinite(update_probability)
        if np.any(
            (update_probability[finite_update] < 0.0)
            | (update_probability[finite_update] > 1.0)
        ):
            raise ValueError("candidate update probabilities must be in [0, 1]")
        if np.any(finite_update & ~np.all(np.isfinite(refined_xy), axis=2)):
            raise ValueError("candidate update probability lacks finite refined coordinates")
        update_probability[~valid] = np.nan
        refined_xy[~valid] = np.nan
        update_threshold = float(self.candidate_update_threshold)
        if not 0.0 <= update_threshold <= 1.0:
            raise ValueError("candidate update threshold must be in [0, 1]")
        spatial_utility_gate_weight = float(self.spatial_utility_gate_weight)
        if not np.isfinite(spatial_utility_gate_weight):
            raise ValueError("spatial utility gate weight must be finite")
        if spatial_utility_gate_weight != 0.0:
            raise ValueError(
                "measurement-update utility is an action posterior and cannot gate "
                "candidate spatial pose likelihood; use selected-identity coordinate "
                "refinement instead"
            )
        explicit_maplet_clusters = self.maplet_cluster_ids is not None
        maplet_clusters = (
            np.full(tracks.shape, -1, dtype=np.int64)
            if self.maplet_cluster_ids is None
            else np.asarray(self.maplet_cluster_ids, dtype=np.int64).copy()
        )
        if maplet_clusters.shape != tracks.shape:
            raise ValueError("maplet cluster ids must match candidate-pool shape")
        if explicit_maplet_clusters and np.any(maplet_clusters[valid] < 0):
            raise ValueError("valid candidates require explicit maplet cluster ids")
        maplet_clusters[~valid] = -1
        for array in (
            tokens,
            xy,
            tracks,
            prototypes,
            xyz,
            scores,
            valid,
            measurement,
            null_scores,
            update_probability,
            refined_xy,
            maplet_clusters,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "token_indices", tokens)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "track_ids", tracks)
        object.__setattr__(self, "prototype_ids", prototypes)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "descriptor_scores", scores)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "measurement_geometry_probabilities", measurement)
        object.__setattr__(self, "measurement_verification_threshold", threshold)
        object.__setattr__(self, "null_scores", null_scores)
        object.__setattr__(self, "_has_explicit_null", bool(explicit_null))
        object.__setattr__(self, "spatial_likelihood", spatial_likelihood)
        object.__setattr__(
            self, "identity_prior_temperature", identity_prior_temperature
        )
        object.__setattr__(
            self, "geometry_prior_mix_weight", geometry_prior_mix_weight
        )
        object.__setattr__(
            self,
            "spatial_geometry_calibration_weight",
            spatial_geometry_calibration_weight,
        )
        object.__setattr__(
            self,
            "geometry_generation_mix_weight",
            geometry_generation_mix_weight,
        )
        object.__setattr__(self, "candidate_update_probabilities", update_probability)
        object.__setattr__(self, "candidate_refined_xy", refined_xy)
        object.__setattr__(self, "candidate_update_threshold", update_threshold)
        object.__setattr__(
            self, "spatial_utility_gate_weight", spatial_utility_gate_weight
        )
        object.__setattr__(self, "maplet_cluster_ids", maplet_clusters)
        object.__setattr__(
            self, "_has_explicit_maplet_clusters", bool(explicit_maplet_clusters)
        )

    @property
    def query_count(self) -> int:
        return int(len(self.token_indices))

    @property
    def has_explicit_null(self) -> bool:
        return bool(self._has_explicit_null)

    @property
    def has_explicit_maplet_clusters(self) -> bool:
        return bool(self._has_explicit_maplet_clusters)

    def subset_by_token_indices(
        self, token_indices: Sequence[int]
    ) -> "PoseVerificationCandidatePool":
        wanted = {int(value) for value in token_indices}
        rows = np.asarray(
            [index for index, token in enumerate(self.token_indices) if int(token) in wanted],
            dtype=np.int64,
        )
        if len(rows) != len(wanted):
            available = set(int(value) for value in self.token_indices.tolist())
            raise ValueError(
                f"candidate pool is missing token indices: {sorted(wanted - available)}"
            )
        return PoseVerificationCandidatePool(
            token_indices=self.token_indices[rows],
            xy=self.xy[rows],
            track_ids=self.track_ids[rows],
            prototype_ids=self.prototype_ids[rows],
            xyz=self.xyz[rows],
            descriptor_scores=self.descriptor_scores[rows],
            valid_mask=self.valid_mask[rows],
            measurement_geometry_probabilities=self.measurement_geometry_probabilities[
                rows
            ],
            measurement_verification_threshold=self.measurement_verification_threshold,
            null_scores=None if not self.has_explicit_null else self.null_scores[rows],
            spatial_likelihood=(
                None
                if self.spatial_likelihood is None
                else self.spatial_likelihood.subset(rows)
            ),
            identity_prior_temperature=self.identity_prior_temperature,
            geometry_prior_mix_weight=self.geometry_prior_mix_weight,
            spatial_geometry_calibration_weight=self.spatial_geometry_calibration_weight,
            geometry_generation_mix_weight=self.geometry_generation_mix_weight,
            candidate_update_probabilities=self.candidate_update_probabilities[rows],
            candidate_refined_xy=self.candidate_refined_xy[rows],
            candidate_update_threshold=self.candidate_update_threshold,
            spatial_utility_gate_weight=self.spatial_utility_gate_weight,
            maplet_cluster_ids=(
                self.maplet_cluster_ids[rows]
                if self.has_explicit_maplet_clusters
                else None
            ),
        )

    def with_spatial_likelihood(
        self,
        spatial_likelihood: CandidateSpatialLikelihood | None,
    ) -> "PoseVerificationCandidatePool":
        """Replace only spatial evidence without rebuilding identity state."""

        if (
            spatial_likelihood is not None
            and spatial_likelihood.candidate_shape != self.valid_mask.shape
        ):
            raise ValueError("spatial likelihood and candidate pool shapes differ")
        # ``dataclasses.replace`` reruns ``__post_init__``. Besides perturbing
        # normalized probability mass, that turns canonical internal maplet
        # sentinels into an explicit maplet input. A shallow immutable copy is
        # the only operation that preserves all non-spatial state exactly.
        output = copy(self)
        object.__setattr__(output, "spatial_likelihood", spatial_likelihood)
        return output

    def with_geometry_generation_mix_weight(
        self, weight: float
    ) -> "PoseVerificationCandidatePool":
        """Replace only the pose-free generation weight without recanonicalizing."""

        resolved = float(weight)
        if not np.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
            raise ValueError("geometry generation mix weight must be in [0, 1]")
        output = copy(self)
        object.__setattr__(output, "geometry_generation_mix_weight", resolved)
        return output

    def mask_candidates_to_null(
        self, keep_mask: np.ndarray
    ) -> "PoseVerificationCandidatePool":
        """Remove candidate identities while preserving joint-softmax mass."""

        if not self.has_explicit_null:
            raise ValueError("candidate masking requires an explicit null posterior")
        keep = np.array(keep_mask, dtype=bool, copy=True)
        if keep.shape != self.valid_mask.shape:
            raise ValueError("candidate keep mask has incompatible shape")
        keep &= self.valid_mask
        scores = np.asarray(self.descriptor_scores, dtype=np.float64)
        if np.any(scores[self.valid_mask] < 0.0):
            raise ValueError("candidate masking requires non-negative probability mass")
        removed = self.valid_mask & ~keep
        removed_mass = np.sum(np.where(removed, scores, 0.0), axis=1)
        masked_scores = np.where(keep, scores, 0.0)
        masked_null = np.asarray(self.null_scores, dtype=np.float64) + removed_mass
        row_total = np.sum(masked_scores, axis=1) + masked_null
        if np.any(row_total <= 0.0) or np.any(np.abs(row_total - 1.0) > 2e-4):
            raise ValueError("candidate masking requires normalized joint probability mass")
        # Source posteriors are commonly stored as float32. Canonicalize their
        # sub-ULP simplex drift before a fully purged row becomes explicit
        # null=1+epsilon. Scaling preserves every relative candidate/null mass.
        masked_scores = masked_scores / row_total[:, None]
        masked_null = masked_null / row_total
        if np.any(masked_null > 1.0) or np.any(masked_null < 0.0):
            raise RuntimeError("candidate masking failed to canonicalize null mass")
        return PoseVerificationCandidatePool(
            token_indices=self.token_indices,
            xy=self.xy,
            track_ids=self.track_ids,
            prototype_ids=self.prototype_ids,
            xyz=self.xyz,
            descriptor_scores=masked_scores,
            valid_mask=keep,
            measurement_geometry_probabilities=(
                self.measurement_geometry_probabilities
            ),
            measurement_verification_threshold=(
                self.measurement_verification_threshold
            ),
            null_scores=masked_null,
            spatial_likelihood=(
                None
                if self.spatial_likelihood is None
                else self.spatial_likelihood.mask_candidates(keep)
            ),
            identity_prior_temperature=self.identity_prior_temperature,
            geometry_prior_mix_weight=self.geometry_prior_mix_weight,
            spatial_geometry_calibration_weight=(
                self.spatial_geometry_calibration_weight
            ),
            geometry_generation_mix_weight=self.geometry_generation_mix_weight,
            candidate_update_probabilities=self.candidate_update_probabilities,
            candidate_refined_xy=self.candidate_refined_xy,
            candidate_update_threshold=self.candidate_update_threshold,
            spatial_utility_gate_weight=self.spatial_utility_gate_weight,
            maplet_cluster_ids=(
                self.maplet_cluster_ids
                if self.has_explicit_maplet_clusters
                else None
            ),
        )


def candidate_pool_likelihood_manifest_sha256(
    pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    residual_sigma_px: float,
    candidate_outlier_likelihood: float = 1e-4,
    null_likelihood: float = 1.0,
) -> str:
    """Hash every pose-independent input to the held-out likelihood."""

    digest = hashlib.sha256()

    def update_array(name: str, value: np.ndarray) -> None:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(int(item) for item in array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))

    update_array("token_indices", pool.token_indices)
    update_array("xy", pool.xy)
    update_array("track_ids", pool.track_ids)
    update_array("prototype_ids", pool.prototype_ids)
    update_array("xyz", pool.xyz)
    update_array("descriptor_scores", pool.descriptor_scores)
    update_array("valid_mask", pool.valid_mask)
    update_array("null_scores", pool.null_scores)
    digest.update(
        b"maplet_clusters:present"
        if pool.has_explicit_maplet_clusters
        else b"maplet_clusters:none"
    )
    if pool.has_explicit_maplet_clusters:
        update_array("maplet_cluster_ids", pool.maplet_cluster_ids)
    update_array(
        "measurement_geometry_probabilities",
        pool.measurement_geometry_probabilities,
    )
    update_array("candidate_update_probabilities", pool.candidate_update_probabilities)
    update_array("candidate_refined_xy", pool.candidate_refined_xy)
    digest.update(
        repr(
            (
                str(CANDIDATE_POSE_EVIDENCE_VERSION),
                float(pool.geometry_prior_mix_weight),
                float(pool.spatial_geometry_calibration_weight),
                float(pool.spatial_utility_gate_weight),
                float(pool.candidate_update_threshold),
                float(pool.identity_prior_temperature),
                float(residual_sigma_px),
                float(candidate_outlier_likelihood),
                float(null_likelihood),
                int(camera.model_id),
                int(camera.width),
                int(camera.height),
                tuple(float(value) for value in camera.params),
            )
        ).encode("ascii")
    )
    spatial = pool.spatial_likelihood
    digest.update(b"spatial:none" if spatial is None else b"spatial:present")
    if spatial is not None:
        update_array("spatial_offsets_xy", spatial.offsets_xy)
        update_array(
            "spatial_local_log_probabilities", spatial.local_log_probabilities
        )
        update_array("spatial_view_probabilities", spatial.view_probabilities)
        update_array(
            "spatial_dustbin_probabilities", spatial.dustbin_probabilities
        )
        update_array("spatial_valid_mask", spatial.valid_mask)
        digest.update(repr(float(spatial.log_evidence_weight)).encode("ascii"))
        if (
            spatial.pose_view_geometry_sigma_deg > 0.0
            or spatial.support_camera_centers is not None
        ):
            digest.update(b"pose_view_geometry:track_to_camera_direction_gaussian_v1")
            digest.update(
                repr(float(spatial.pose_view_geometry_sigma_deg)).encode("ascii")
            )
            digest.update(
                b"spatial_support_camera_centers:none"
                if spatial.support_camera_centers is None
                else b"spatial_support_camera_centers:present"
            )
            if spatial.support_camera_centers is not None:
                update_array(
                    "spatial_support_camera_centers",
                    spatial.support_camera_centers,
                )
    return digest.hexdigest()


@dataclass(frozen=True)
class HypothesisVerification:
    verification_count: int
    finite_count: int
    positive_depth_count: int
    positive_depth_ratio: float
    strict_inlier_count: int
    loose_inlier_count: int
    strict_grid_cell_count: int
    loose_grid_cell_count: int
    soft_consensus: float
    clipped_median_residual_px: float
    depth_range_m: float | None
    selected_candidate_count: int = 0
    selected_candidate_fraction: float = 0.0
    selected_descriptor_score_mean: float | None = None
    selected_descriptor_score_median: float | None = None
    selected_descriptor_margin_mean: float | None = None
    selected_descriptor_rank_score_mean: float | None = None
    selected_assignment_utility_mean: float | None = None
    selected_reprojection_mean_px: float | None = None
    selected_reprojection_p90_px: float | None = None
    measurement_evidence_count: int = 0
    measurement_evidence_fraction: float = 0.0
    measurement_probability_mean: float | None = None
    measurement_high_confidence_fraction: float = 0.0
    measurement_strict_probability_mass_fraction: float = 0.0
    measurement_loose_probability_mass_fraction: float = 0.0
    measurement_soft_consensus_ratio: float = 0.0
    measurement_high_confidence_strict_fraction: float = 0.0
    measurement_high_confidence_loose_fraction: float = 0.0
    measurement_high_confidence_contradiction_fraction: float = 0.0
    fixed_posterior_log_likelihood_sum: float | None = None
    fixed_posterior_log_likelihood_mean: float | None = None
    fixed_posterior_log_likelihood_std: float | None = None
    fixed_posterior_log_likelihood_standard_error: float | None = None
    fixed_posterior_log_likelihood_median: float | None = None
    fixed_posterior_log_likelihood_trimmed_mean_10: float | None = None
    fixed_posterior_log_likelihood_worst_quartile_mean: float | None = None
    fixed_posterior_log_likelihood_lcb95: float | None = None
    fixed_posterior_spatial_median_of_means_2x2: float | None = None
    fixed_posterior_spatial_mom_cell_count: int = 0
    fixed_posterior_effective_group_count: int = 0
    fixed_posterior_mass_max_abs_error: float | None = None
    fixed_posterior_spatial_candidate_count: int = 0
    fixed_posterior_geometry_prior_evidence_count: int = 0
    fixed_posterior_geometry_prior_mix_weight: float = 0.0
    fixed_posterior_identity_prior_temperature: float = 1.0
    fixed_posterior_identity_prior_entropy_mean: float | None = None
    fixed_posterior_identity_prior_effective_candidate_count_mean: float | None = None
    fixed_posterior_retained_candidate_mass_mean: float | None = None
    fixed_posterior_null_evidence_fraction_mean: float | None = None
    fixed_posterior_candidate_inlier_evidence_fraction_mean: float | None = None
    fixed_posterior_spatial_log_likelihood_gain_mean: float | None = None
    fixed_posterior_spatial_calibrated_candidate_count: int = 0
    fixed_posterior_spatial_geometry_calibration_weight: float = 0.0
    fixed_posterior_relation_pair_count: int = 0
    fixed_posterior_relation_effective_pair_count: int = 0
    fixed_posterior_relation_log_likelihood_ratio_sum: float | None = None
    fixed_posterior_relation_log_likelihood_ratio_mean: float | None = None
    information_match_count: int = 0
    translation_information_min_eigenvalue: float | None = None
    translation_information_condition: float | None = None
    rotation_information_min_eigenvalue: float | None = None
    rotation_information_condition: float | None = None
    joint_information_min_eigenvalue: float | None = None
    joint_information_condition: float | None = None
    bearing_max_angle_deg: float | None = None
    camera_depth_span_m: float | None = None
    camera_depth_span_ratio: float | None = None
    xyz_second_singular_ratio: float | None = None
    xyz_third_singular_ratio: float | None = None

    def rank_key(self) -> tuple[float, ...]:
        # All hypotheses for an image use the same held-out set. Strict
        # consensus leads because centimeter pose accuracy is not represented
        # well by a permissive 8 px RANSAC inlier count.
        return (
            float(self.positive_depth_ratio >= 0.9),
            float(self.strict_inlier_count),
            float(self.strict_grid_cell_count),
            float(self.loose_inlier_count),
            float(self.loose_grid_cell_count),
            float(self.soft_consensus),
            -float(self.clipped_median_residual_px),
        )

    def fixed_posterior_rank_key(self) -> tuple[float, ...]:
        """Rank poses only by one immutable top-L likelihood denominator."""

        likelihood = self.fixed_posterior_log_likelihood_mean
        if likelihood is None:
            return self.rank_key()
        # Inlier counts, grid coverage and residual summaries are all computed
        # from the pose being ranked. Using them as tie-breaks would reintroduce
        # self-consistency evidence after the fixed-denominator likelihood.
        return (float(likelihood),)

    def hypothesis_selection_rank_key(self, policy: str) -> tuple[float, ...]:
        """Return the explicit selector statistic without hidden tie-breaks."""

        policy_name = str(policy)
        if policy_name == "fixed_posterior_likelihood_only":
            return self.fixed_posterior_rank_key()
        if policy_name == "legacy_self_consistency_DIAGNOSTIC_ONLY":
            return self.legacy_fixed_posterior_rank_key_diagnostic_only()
        field_by_policy = {
            "fixed_posterior_median_DIAGNOSTIC_ONLY": (
                "fixed_posterior_log_likelihood_median"
            ),
            "fixed_posterior_trimmed_mean_10_DIAGNOSTIC_ONLY": (
                "fixed_posterior_log_likelihood_trimmed_mean_10"
            ),
            "fixed_posterior_worst_quartile_mean_DIAGNOSTIC_ONLY": (
                "fixed_posterior_log_likelihood_worst_quartile_mean"
            ),
            "fixed_posterior_lcb95_DIAGNOSTIC_ONLY": (
                "fixed_posterior_log_likelihood_lcb95"
            ),
            "fixed_posterior_spatial_mom_2x2_DIAGNOSTIC_ONLY": (
                "fixed_posterior_spatial_median_of_means_2x2"
            ),
        }
        field_name = field_by_policy.get(policy_name)
        if field_name is None:
            raise ValueError("unsupported grouped hypothesis selection policy")
        value = getattr(self, field_name)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(
                f"grouped hypothesis selector {policy_name} lacks {field_name}"
            )
        # Robust diagnostic selectors remain immutable-likelihood-only. Pose
        # self-consistency quantities are deliberately excluded from ties.
        return (float(value),)

    def legacy_fixed_posterior_rank_key_diagnostic_only(
        self,
    ) -> tuple[float, ...]:
        """Replay the historical S43 selector; never use for promotion claims."""

        likelihood = self.fixed_posterior_log_likelihood_mean
        if likelihood is None:
            return self.rank_key()
        return (
            float(likelihood),
            float(self.positive_depth_ratio >= 0.9),
            float(self.strict_grid_cell_count),
            float(self.strict_inlier_count),
            float(self.soft_consensus),
            -float(self.clipped_median_residual_px),
        )


@dataclass(frozen=True)
class PoseHypothesisRecord:
    fit_match_count_limit: int
    fit_match_count: int
    selection_mode: str
    ransac_threshold_px: float
    rng_seed_offset: int
    solver_success: bool
    fit_inlier_count: int
    verification: HypothesisVerification | None
    sample_token_indices: tuple[int, ...] = ()
    sample_track_ids: tuple[int, ...] = ()
    prosac_prefix_size: int | None = None
    sampling_log_probability: float | None = None
    local_optimization_match_count: int = 0
    local_optimization_applied: bool = False
    latent_em_applied: bool = False
    latent_em_iterations: int = 0
    latent_em_final_log_likelihood: float | None = None
    latent_em_parent_hypothesis_index: int | None = None
    latent_em_seed_evidence_mode: str | None = None
    observability_gate_failures: tuple[str, ...] = ()
    generation_profile: str = "legacy"
    spatial_rescored_for_shortlist: bool = False
    preliminary_log_likelihood_mean: float | None = None
    shortlist_log_likelihood_mean: float | None = None
    shortlist_evidence_mode: str | None = None


@dataclass(frozen=True)
class GroupedMinimalSet:
    """One group-aware PnP sample with explicit candidate identities."""

    matches: tuple[QueryTo3DMatch, ...]
    row_indices: tuple[int, ...]
    candidate_columns: tuple[int, ...]
    prosac_prefix_size: int
    sampling_log_probability: float
    spatial_mode_count: int = 0
    generation_profile: str = "legacy"


@dataclass(frozen=True)
class GroupedGeneratedPose:
    """A pose generated from one grouped minimal set before held-out ranking."""

    pose_w2c: np.ndarray | None
    sample: GroupedMinimalSet
    solver_success: bool
    local_optimization_match_count: int = 0
    local_optimization_inlier_count: int = 0
    local_optimization_applied: bool = False
    latent_em_applied: bool = False
    latent_em_iterations: int = 0
    latent_em_final_log_likelihood: float | None = None
    latent_em_parent_generated_index: int | None = None
    latent_em_seed_evidence_mode: str | None = None
    observability_gate_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupedCandidateCrossfitPools:
    fit: PoseVerificationCandidatePool
    shortlist: PoseVerificationCandidatePool
    verification: PoseVerificationCandidatePool
    audit: PoseVerificationCandidatePool
    fit_tokens: tuple[int, ...]
    shortlist_tokens: tuple[int, ...]
    verification_tokens: tuple[int, ...]
    audit_tokens: tuple[int, ...]
    partition_audit: dict[str, object]


@dataclass(frozen=True)
class ComponentPartitionPlan:
    fit_indices: np.ndarray
    verification_indices: np.ndarray
    audit_indices: np.ndarray
    audit: dict[str, object]


HypothesisSelector = Callable[
    [
        Sequence[PoseHypothesisRecord],
        Sequence[Optional[np.ndarray]],
        Sequence[int],
    ],
    int,
]


@dataclass(frozen=True)
class VerifiedPnPResult:
    success: bool
    pose_w2c: np.ndarray | None
    inlier_mask: np.ndarray
    match_count: int
    inlier_count: int
    fit_count: int
    verification_count: int
    final_audit_count: int
    chosen_hypothesis_index: int | None
    hypotheses: tuple[PoseHypothesisRecord, ...]
    hypothesis_poses_w2c: tuple[np.ndarray | None, ...]
    pre_refine_pose_w2c: np.ndarray | None
    pre_refine_verification: HypothesisVerification | None
    pre_refine_final_audit_verification: HypothesisVerification | None
    final_verification: HypothesisVerification | None
    final_refine_mode_used: str | None = None
    final_refine_attempted: bool = False
    final_refine_accepted: bool = False
    candidate_coordinate_update_count: int = 0
    candidate_coordinate_update_grid_cell_count: int = 0
    candidate_coordinate_refine_attempted: bool = False
    candidate_coordinate_refine_accepted: bool = False
    verification_denominator_sha256: str | None = None
    final_audit_denominator_sha256: str | None = None
    crossfit_partition_audit: dict[str, object] | None = None

    def summary(self) -> dict[str, object]:
        return {
            "success": bool(self.success),
            "match_count": int(self.match_count),
            "inlier_count": int(self.inlier_count),
            "fit_count": int(self.fit_count),
            "verification_count": int(self.verification_count),
            "final_audit_count": int(self.final_audit_count),
            "chosen_hypothesis_index": self.chosen_hypothesis_index,
            "hypothesis_count": int(len(self.hypotheses)),
            "final_refine_mode_used": self.final_refine_mode_used,
            "final_refine_attempted": bool(self.final_refine_attempted),
            "final_refine_accepted": bool(self.final_refine_accepted),
            "candidate_coordinate_update_count": int(
                self.candidate_coordinate_update_count
            ),
            "candidate_coordinate_update_grid_cell_count": int(
                self.candidate_coordinate_update_grid_cell_count
            ),
            "candidate_coordinate_refine_attempted": bool(
                self.candidate_coordinate_refine_attempted
            ),
            "candidate_coordinate_refine_accepted": bool(
                self.candidate_coordinate_refine_accepted
            ),
            "verification_denominator_sha256": self.verification_denominator_sha256,
            "final_audit_denominator_sha256": self.final_audit_denominator_sha256,
            "crossfit_partition_audit": self.crossfit_partition_audit,
            "chosen_hypothesis": (
                None
                if self.chosen_hypothesis_index is None
                else _record_json(self.hypotheses[self.chosen_hypothesis_index])
            ),
            "pre_refine_verification": (
                None
                if self.pre_refine_verification is None
                else asdict(self.pre_refine_verification)
            ),
            "pre_refine_final_audit_verification": (
                None
                if self.pre_refine_final_audit_verification is None
                else asdict(self.pre_refine_final_audit_verification)
            ),
            "final_verification": (
                None
                if self.final_verification is None
                else asdict(self.final_verification)
            ),
        }


def select_geometry_guided_generation_with_immutable_baseline(
    baseline: VerifiedPnPResult,
    optional: VerifiedPnPResult,
    *,
    min_strict_grid_cell_delta: int = 0,
) -> tuple[VerifiedPnPResult, dict[str, object]]:
    """Promote optional generation only with non-decreasing held-out coverage.

    The baseline result is returned unchanged whenever the optional result does
    not pass the inference-only gate. The gate intentionally uses the chosen
    hypothesis verification fold, not the final audit fold or ground truth.
    """

    minimum_delta = int(min_strict_grid_cell_delta)
    if minimum_delta < 0:
        raise ValueError("min_strict_grid_cell_delta must be non-negative")

    baseline_verification = baseline.pre_refine_verification
    optional_verification = optional.pre_refine_verification
    baseline_grid_cells = (
        None
        if baseline_verification is None
        else int(baseline_verification.strict_grid_cell_count)
    )
    optional_grid_cells = (
        None
        if optional_verification is None
        else int(optional_verification.strict_grid_cell_count)
    )

    promoted = False
    if not baseline.success:
        if optional.success and optional_verification is not None:
            promoted = True
            reason = "baseline_failed_optional_succeeded"
        else:
            reason = "baseline_and_optional_unusable"
    elif not optional.success:
        reason = "optional_failed"
    elif baseline_verification is None:
        reason = "baseline_verification_missing"
    elif optional_verification is None:
        reason = "optional_verification_missing"
    elif optional_grid_cells >= baseline_grid_cells + minimum_delta:
        promoted = True
        reason = "strict_grid_coverage_gate_passed"
    else:
        reason = "strict_grid_coverage_regressed"

    selected = optional if promoted else baseline
    audit = {
        "selected_source": (
            "optional_geometry_guided_generation"
            if promoted
            else "immutable_unmixed_baseline"
        ),
        "promoted": bool(promoted),
        "fallback_reason": str(reason),
        "rule": "optional_strict_grid_cells_ge_baseline_plus_min_delta",
        "min_strict_grid_cell_delta": minimum_delta,
        "baseline_success": bool(baseline.success),
        "optional_success": bool(optional.success),
        "baseline_strict_grid_cell_count": baseline_grid_cells,
        "optional_strict_grid_cell_count": optional_grid_cells,
    }
    return selected, audit


def select_crossfit_likelihood_with_immutable_baseline(
    baseline: VerifiedPnPResult,
    optional: VerifiedPnPResult,
    *,
    min_log_likelihood_mean_delta: float = 0.0,
    min_effective_group_count: int = 8,
    min_information_match_count: int = 0,
    min_translation_information_eigenvalue: float = 0.0,
    max_translation_information_condition: float = 0.0,
    max_joint_information_condition: float = 0.0,
    min_bearing_span_deg: float = 0.0,
    min_depth_span_ratio: float = 0.0,
    min_xyz_second_singular_ratio: float = 0.0,
    min_xyz_third_singular_ratio: float = 0.0,
) -> tuple[VerifiedPnPResult, dict[str, object]]:
    """Promote only with stronger evidence on one immutable held-out pool."""

    minimum_delta = float(min_log_likelihood_mean_delta)
    minimum_groups = int(min_effective_group_count)
    if not np.isfinite(minimum_delta) or minimum_delta < 0.0:
        raise ValueError("likelihood promotion delta must be finite and non-negative")
    if minimum_groups < 1:
        raise ValueError("likelihood promotion requires at least one effective group")
    observability_thresholds = {
        "min_information_match_count": float(min_information_match_count),
        "min_translation_information_eigenvalue": float(
            min_translation_information_eigenvalue
        ),
        "max_translation_information_condition": float(
            max_translation_information_condition
        ),
        "max_joint_information_condition": float(max_joint_information_condition),
        "min_bearing_span_deg": float(min_bearing_span_deg),
        "min_depth_span_ratio": float(min_depth_span_ratio),
        "min_xyz_second_singular_ratio": float(min_xyz_second_singular_ratio),
        "min_xyz_third_singular_ratio": float(min_xyz_third_singular_ratio),
    }
    if any(
        not np.isfinite(value) or value < 0.0
        for value in observability_thresholds.values()
    ):
        raise ValueError("observability thresholds must be finite and non-negative")

    # Hypothesis generation and internal refinement may consume the verification
    # fold. Promotion must score the actual returned poses on the untouched audit
    # fold, rather than scoring their pre-refine ancestors on reused evidence.
    baseline_verification = baseline.final_verification
    optional_verification = optional.final_verification
    baseline_hash = baseline.final_audit_denominator_sha256
    optional_hash = optional.final_audit_denominator_sha256
    if baseline.success and optional.success:
        if baseline_hash is None or optional_hash is None:
            raise ValueError("likelihood promotion requires denominator manifests")
        if baseline_hash != optional_hash:
            raise ValueError(
                "baseline and optional use different held-out likelihood denominators"
            )
        if baseline_verification is not None and optional_verification is not None and (
            baseline_verification.verification_count
            != optional_verification.verification_count
            or baseline_verification.fixed_posterior_effective_group_count
            != optional_verification.fixed_posterior_effective_group_count
        ):
            raise ValueError(
                "baseline and optional held-out group denominators differ"
            )

    baseline_likelihood = (
        None
        if baseline_verification is None
        else baseline_verification.fixed_posterior_log_likelihood_mean
    )
    optional_likelihood = (
        None
        if optional_verification is None
        else optional_verification.fixed_posterior_log_likelihood_mean
    )
    effective_groups = (
        0
        if optional_verification is None
        else int(optional_verification.fixed_posterior_effective_group_count)
    )
    likelihood_delta = (
        None
        if baseline_likelihood is None or optional_likelihood is None
        else float(optional_likelihood - baseline_likelihood)
    )
    observability_failures: list[str] = []
    if optional_verification is not None:
        minimum_checks = (
            (
                "information_match_count",
                float(optional_verification.information_match_count),
                observability_thresholds["min_information_match_count"],
            ),
            (
                "translation_information_min_eigenvalue",
                optional_verification.translation_information_min_eigenvalue,
                observability_thresholds[
                    "min_translation_information_eigenvalue"
                ],
            ),
            (
                "bearing_max_angle_deg",
                optional_verification.bearing_max_angle_deg,
                observability_thresholds["min_bearing_span_deg"],
            ),
            (
                "camera_depth_span_ratio",
                optional_verification.camera_depth_span_ratio,
                observability_thresholds["min_depth_span_ratio"],
            ),
            (
                "xyz_second_singular_ratio",
                optional_verification.xyz_second_singular_ratio,
                observability_thresholds["min_xyz_second_singular_ratio"],
            ),
            (
                "xyz_third_singular_ratio",
                optional_verification.xyz_third_singular_ratio,
                observability_thresholds["min_xyz_third_singular_ratio"],
            ),
        )
        for name, value, threshold in minimum_checks:
            if threshold > 0.0 and (
                value is None
                or not np.isfinite(float(value))
                or float(value) < threshold
            ):
                observability_failures.append(str(name))
        maximum_checks = (
            (
                "translation_information_condition",
                optional_verification.translation_information_condition,
                observability_thresholds["max_translation_information_condition"],
            ),
            (
                "joint_information_condition",
                optional_verification.joint_information_condition,
                observability_thresholds["max_joint_information_condition"],
            ),
        )
        for name, value, threshold in maximum_checks:
            if threshold > 0.0 and (
                value is None
                or not np.isfinite(float(value))
                or float(value) > threshold
            ):
                observability_failures.append(str(name))

    promoted = False
    if not baseline.success:
        if (
            optional.success
            and optional_likelihood is not None
            and effective_groups >= minimum_groups
            and not observability_failures
        ):
            promoted = True
            reason = "baseline_failed_optional_has_heldout_evidence"
        elif observability_failures:
            reason = "observability_gate_veto"
        else:
            reason = "baseline_and_optional_unusable"
    elif not optional.success:
        reason = "optional_failed"
    elif baseline_likelihood is None or optional_likelihood is None:
        reason = "fixed_posterior_likelihood_missing"
    elif effective_groups < minimum_groups:
        reason = "insufficient_effective_groups"
    elif observability_failures:
        reason = "observability_gate_veto"
    elif likelihood_delta is not None and likelihood_delta >= minimum_delta:
        promoted = True
        reason = "crossfit_likelihood_ratio_gate_passed"
    else:
        reason = "crossfit_likelihood_ratio_gate_abstained"

    selected = optional if promoted else baseline
    audit = {
        "selected_source": (
            "optional_latent_generation" if promoted else "immutable_baseline"
        ),
        "promoted": bool(promoted),
        "abstained": bool(baseline.success and not promoted),
        "fallback_reason": str(reason),
        "rule": "same_denominator_final_audit_log_likelihood_ratio",
        "evidence_partition": "final_audit",
        "final_audit_denominator_sha256": (
            optional_hash if promoted else baseline_hash
        ),
        "min_log_likelihood_mean_delta": float(minimum_delta),
        "min_effective_group_count": int(minimum_groups),
        "effective_group_count": int(effective_groups),
        "baseline_log_likelihood_mean": baseline_likelihood,
        "optional_log_likelihood_mean": optional_likelihood,
        "optional_minus_baseline_log_likelihood_mean": likelihood_delta,
        "baseline_success": bool(baseline.success),
        "optional_success": bool(optional.success),
        "observability_gate_passed": not bool(observability_failures),
        "observability_gate_failures": list(observability_failures),
        "observability_thresholds": dict(observability_thresholds),
        "optional_observability": (
            None
            if optional_verification is None
            else {
                "information_match_count": int(
                    optional_verification.information_match_count
                ),
                "translation_information_min_eigenvalue": (
                    optional_verification.translation_information_min_eigenvalue
                ),
                "translation_information_condition": (
                    optional_verification.translation_information_condition
                ),
                "joint_information_condition": (
                    optional_verification.joint_information_condition
                ),
                "bearing_max_angle_deg": optional_verification.bearing_max_angle_deg,
                "camera_depth_span_ratio": (
                    optional_verification.camera_depth_span_ratio
                ),
                "xyz_second_singular_ratio": (
                    optional_verification.xyz_second_singular_ratio
                ),
                "xyz_third_singular_ratio": (
                    optional_verification.xyz_third_singular_ratio
                ),
            }
        ),
    }
    return selected, audit


def _accept_grouped_final_refine(
    baseline: HypothesisVerification | None,
    proposed: HypothesisVerification | None,
    *,
    policy: str,
) -> bool:
    if proposed is None:
        return False
    if baseline is None:
        return True
    if str(policy) == "fixed_posterior_likelihood_gain":
        baseline_likelihood = baseline.fixed_posterior_log_likelihood_mean
        proposed_likelihood = proposed.fixed_posterior_log_likelihood_mean
        if baseline_likelihood is None or proposed_likelihood is None:
            return False
        return bool(float(proposed_likelihood) > float(baseline_likelihood))
    if str(policy) == "legacy_rank_key":
        return proposed.rank_key() >= baseline.rank_key()
    if str(policy) == "strict_count_gain_with_grid_nondecrease":
        return bool(
            proposed.strict_inlier_count > baseline.strict_inlier_count
            and proposed.strict_grid_cell_count >= baseline.strict_grid_cell_count
        )
    raise ValueError("unsupported grouped final refine acceptance policy")


def _record_json(record: PoseHypothesisRecord) -> dict[str, object]:
    output = asdict(record)
    return output


def _stable_mix(token_index: int, track_id: int, salt: int) -> int:
    payload = f"{int(token_index)}:{int(track_id)}:{int(salt)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def deterministic_spatial_holdout(
    matches: Sequence[QueryTo3DMatch],
    *,
    image_width: int,
    image_height: int,
    folds: int = 4,
    fold: int = 0,
    grid_rows: int = 4,
    grid_cols: int = 4,
    salt: int = 0,
    fold_policy: str = "legacy_local_modulo",
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, spatially distributed fit and verification rows."""

    values = list(matches)
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if int(folds) < 2 or not 0 <= int(fold) < int(folds):
        raise ValueError("invalid holdout fold configuration")
    if int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("grid dimensions must be positive")
    if str(fold_policy) not in {
        "legacy_local_modulo",
        "cell_rotated_balanced",
    }:
        raise ValueError("unsupported spatial holdout fold policy")
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, match in enumerate(values):
        x, y = np.asarray(match.xy, dtype=np.float64).reshape(2)
        col = int(
            np.clip(
                np.floor(x / float(image_width) * int(grid_cols)),
                0,
                int(grid_cols) - 1,
            )
        )
        row = int(
            np.clip(
                np.floor(y / float(image_height) * int(grid_rows)),
                0,
                int(grid_rows) - 1,
            )
        )
        buckets.setdefault((row, col), []).append(index)

    verification: list[int] = []
    balanced_fold_counts = np.zeros((int(folds),), dtype=np.int64)
    for cell_index, cell in enumerate(sorted(buckets)):
        ordered = sorted(
            buckets[cell],
            key=lambda index: (
                _stable_mix(
                    values[index].token_index,
                    values[index].track_id,
                    int(salt),
                ),
                int(values[index].token_index),
                int(values[index].track_id),
            ),
        )
        if str(fold_policy) == "legacy_local_modulo":
            fold_order = tuple(range(int(folds)))
        else:
            rotation = (int(cell_index) + int(salt)) % int(folds)
            fold_order = tuple(
                sorted(
                    range(int(folds)),
                    key=lambda candidate_fold: (
                        int(balanced_fold_counts[int(candidate_fold)]),
                        (int(candidate_fold) - rotation) % int(folds),
                    ),
                )
            )
        assigned_folds = [
            int(fold_order[local_index % int(folds)])
            for local_index in range(len(ordered))
        ]
        if str(fold_policy) == "cell_rotated_balanced":
            balanced_fold_counts += np.bincount(
                np.asarray(assigned_folds, dtype=np.int64),
                minlength=int(folds),
            )
        verification.extend(
            index
            for index, assigned_fold in zip(ordered, assigned_folds)
            if int(assigned_fold) == int(fold)
        )
    verify = np.asarray(sorted(set(verification)), dtype=np.int64)
    fit_mask = np.ones((len(values),), dtype=bool)
    fit_mask[verify] = False
    fit = np.flatnonzero(fit_mask).astype(np.int64)
    return fit, verify


def deterministic_spatial_partitions(
    matches: Sequence[QueryTo3DMatch],
    *,
    image_width: int,
    image_height: int,
    folds: int = 4,
    verification_fold: int = 0,
    verification_fold_count: int = 1,
    final_audit_fold: int = 1,
    grid_rows: int = 4,
    grid_cols: int = 4,
    salt: int = 0,
    fold_policy: str = "legacy_local_modulo",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint hypothesis-fit, rank-verification, and final-audit rows."""

    verification_folds = tuple(
        range(
            int(verification_fold),
            int(verification_fold) + int(verification_fold_count),
        )
    )
    if not verification_folds or min(verification_folds) < 0 or max(
        verification_folds
    ) >= int(folds):
        raise ValueError("verification folds are outside folds")
    if int(final_audit_fold) in verification_folds:
        raise ValueError("verification and final audit folds must differ")
    verification_parts = []
    for fold in verification_folds:
        _fit, fold_indices = deterministic_spatial_holdout(
            matches,
            image_width=int(image_width),
            image_height=int(image_height),
            folds=int(folds),
            fold=int(fold),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            salt=int(salt),
            fold_policy=str(fold_policy),
        )
        verification_parts.append(fold_indices)
    verification = np.asarray(
        sorted(
            set(
                np.concatenate(verification_parts).astype(np.int64).tolist()
            )
        ),
        dtype=np.int64,
    )
    _fit, final_audit = deterministic_spatial_holdout(
        matches,
        image_width=int(image_width),
        image_height=int(image_height),
        folds=int(folds),
        fold=int(final_audit_fold),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        salt=int(salt),
        fold_policy=str(fold_policy),
    )
    if np.intersect1d(verification, final_audit).size:
        raise RuntimeError("deterministic spatial folds unexpectedly overlap")
    excluded = np.zeros((len(matches),), dtype=bool)
    excluded[verification] = True
    excluded[final_audit] = True
    fit = np.flatnonzero(~excluded).astype(np.int64)
    return fit, verification, final_audit


def _deterministic_component_partition_plan(
    pool: PoseVerificationCandidatePool,
    *,
    image_width: int,
    image_height: int,
    folds: int = 4,
    verification_fold: int = 0,
    verification_fold_count: int = 1,
    final_audit_fold: int = 1,
    grid_rows: int = 4,
    grid_cols: int = 4,
    maplet_voxel_size_m: float | None = None,
    use_explicit_maplet_clusters: bool = False,
    role_assignment: str = "fixed",
    minimum_fit_count: int = 8,
    minimum_heldout_count: int = 4,
    salt: int = 0,
) -> ComponentPartitionPlan:
    """Partition token-track-maplet components without identity leakage.

    A sliding KNN maplet graph percolates across a facade and is unsuitable as
    a cross-fit unit. Maplet isolation therefore consumes fixed disjoint maplet
    IDs. The legacy voxel grouping remains available only as a diagnostic.
    """

    fold_count = int(folds)
    verification_folds = tuple(
        range(
            int(verification_fold),
            int(verification_fold) + int(verification_fold_count),
        )
    )
    if fold_count < 3:
        raise ValueError("component cross-fit requires at least three folds")
    if not verification_folds or min(verification_folds) < 0 or max(
        verification_folds
    ) >= fold_count:
        raise ValueError("verification folds are outside folds")
    if not 0 <= int(final_audit_fold) < fold_count:
        raise ValueError("final audit fold is outside folds")
    if int(final_audit_fold) in verification_folds:
        raise ValueError("verification and final audit folds must differ")
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("grid dimensions must be positive")
    if maplet_voxel_size_m is not None and float(maplet_voxel_size_m) <= 0.0:
        raise ValueError("maplet voxel size must be positive")
    if bool(use_explicit_maplet_clusters) and maplet_voxel_size_m is not None:
        raise ValueError("explicit maplet clusters and voxel grouping are exclusive")
    if bool(use_explicit_maplet_clusters) and not pool.has_explicit_maplet_clusters:
        raise ValueError("explicit maplet component cross-fit requires cluster ids")
    if str(role_assignment) not in {"fixed", "adaptive_balanced"}:
        raise ValueError("unsupported component cross-fit role assignment")
    if int(minimum_fit_count) < 4 or int(minimum_heldout_count) < 1:
        raise ValueError("component cross-fit role minima are invalid")

    count = int(pool.query_count)
    parent = np.arange(count, dtype=np.int64)

    def find(row: int) -> int:
        value = int(row)
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    track_owner: dict[int, int] = {}
    voxel_owner: dict[tuple[int, int, int], int] = {}
    maplet_owner: dict[int, int] = {}
    valid = np.asarray(pool.valid_mask, dtype=bool)
    tracks = np.asarray(pool.track_ids, dtype=np.int64)
    xyz = np.asarray(pool.xyz, dtype=np.float64)
    for row in range(count):
        for column in np.flatnonzero(valid[row]).tolist():
            track_id = int(tracks[row, column])
            previous = track_owner.setdefault(track_id, int(row))
            union(row, previous)
            if bool(use_explicit_maplet_clusters):
                cluster_id = int(pool.maplet_cluster_ids[row, column])
                previous_maplet = maplet_owner.setdefault(cluster_id, int(row))
                union(row, previous_maplet)
            if maplet_voxel_size_m is not None:
                voxel = tuple(
                    np.floor(
                        xyz[row, column] / float(maplet_voxel_size_m)
                    ).astype(np.int64).tolist()
                )
                previous_voxel = voxel_owner.setdefault(voxel, int(row))
                union(row, previous_voxel)

    components: dict[int, list[int]] = {}
    for row in range(count):
        components.setdefault(find(row), []).append(int(row))
    if len(components) < fold_count:
        raise ValueError(
            "candidate graph has fewer connected components than cross-fit folds"
        )

    cell_count = int(grid_rows) * int(grid_cols)
    row_cells = np.zeros((count,), dtype=np.int64)
    for row, (x, y) in enumerate(np.asarray(pool.xy, dtype=np.float64)):
        grid_row = int(
            np.clip(
                np.floor(y / float(image_height) * int(grid_rows)),
                0,
                int(grid_rows) - 1,
            )
        )
        grid_column = int(
            np.clip(
                np.floor(x / float(image_width) * int(grid_cols)),
                0,
                int(grid_cols) - 1,
            )
        )
        row_cells[row] = grid_row * int(grid_cols) + grid_column

    component_rows = list(components.values())
    component_rows.sort(
        key=lambda rows: (
            -len(rows),
            int.from_bytes(
                hashlib.sha256(
                    (
                        f"{int(salt)}:"
                        + ",".join(
                            str(value)
                            for value in sorted(
                                int(pool.token_indices[row]) for row in rows
                            )
                        )
                    ).encode("ascii")
                ).digest()[:8],
                "little",
            ),
        )
    )
    target_size = float(count) / float(fold_count)
    total_cell_histogram = np.bincount(
        row_cells, minlength=cell_count
    ).astype(np.float64)
    target_cell_histogram = total_cell_histogram / float(fold_count)
    fold_sizes = np.zeros((fold_count,), dtype=np.float64)
    fold_cells = np.zeros((fold_count, cell_count), dtype=np.float64)
    fold_component_counts = np.zeros((fold_count,), dtype=np.int64)
    assignment = np.full((count,), -1, dtype=np.int64)
    for component_index, rows in enumerate(component_rows):
        histogram = np.bincount(
            row_cells[np.asarray(rows, dtype=np.int64)], minlength=cell_count
        ).astype(np.float64)
        size = float(len(rows))

        def fold_cost(fold: int) -> tuple[float, float, int]:
            size_error = (fold_sizes[fold] + size) / max(target_size, 1.0)
            cell_error = float(
                np.mean(
                    np.abs(
                        fold_cells[fold] + histogram - target_cell_histogram
                    )
                    / np.maximum(target_cell_histogram, 1.0)
                )
            )
            tie = int(
                (
                    component_index
                    + int(salt)
                    + 104729 * int(fold)
                )
                % fold_count
            )
            return float(size_error + 0.25 * cell_error), float(
                fold_sizes[fold]
            ), tie

        empty_folds = np.flatnonzero(fold_component_counts == 0).tolist()
        eligible_folds = (
            empty_folds
            if component_index < fold_count and empty_folds
            else range(fold_count)
        )
        selected_fold = min(eligible_folds, key=fold_cost)
        assignment[np.asarray(rows, dtype=np.int64)] = int(selected_fold)
        fold_sizes[selected_fold] += size
        fold_cells[selected_fold] += histogram
        fold_component_counts[selected_fold] += 1
    if np.any(assignment < 0):
        raise RuntimeError("component partition left rows unassigned")

    fixed_verification_folds = tuple(int(value) for value in verification_folds)
    fixed_audit_fold = int(final_audit_fold)
    fixed_fit_folds = tuple(
        fold
        for fold in range(fold_count)
        if fold not in (*fixed_verification_folds, fixed_audit_fold)
    )
    if str(role_assignment) == "adaptive_balanced":
        target_counts = (0.5 * count, 0.25 * count, 0.25 * count)

        def role_cost(roles: tuple[int, ...]) -> tuple[object, ...]:
            role_counts = tuple(
                int(
                    sum(
                        int(fold_sizes[fold])
                        for fold, role in enumerate(roles)
                        if int(role) == role_index
                    )
                )
                for role_index in range(3)
            )
            fit_count, verification_count, audit_count = role_counts
            deficit = (
                max(0, int(minimum_fit_count) - fit_count)
                + max(0, int(minimum_heldout_count) - verification_count)
                + max(0, int(minimum_heldout_count) - audit_count)
            )
            target_error = float(
                sum(
                    abs(value - target) / max(float(count), 1.0)
                    for value, target in zip(role_counts, target_counts)
                )
            )
            verification_smaller_than_audit = int(
                verification_count < audit_count
            )
            role_grid_cells = tuple(
                int(
                    np.count_nonzero(
                        np.sum(
                            [
                                fold_cells[fold]
                                for fold, role in enumerate(roles)
                                if int(role) == role_index
                            ],
                            axis=0,
                        )
                    )
                )
                for role_index in range(3)
            )
            tie = tuple(
                int((int(salt) + 104729 * fold + role) % (3 * fold_count))
                for fold, role in enumerate(roles)
            )
            return (
                int(deficit > 0),
                int(deficit),
                target_error,
                verification_smaller_than_audit,
                -min(verification_count, audit_count),
                -min(role_grid_cells[1], role_grid_cells[2]),
                tie,
            )

        role_candidates = (
            tuple(int(value) for value in roles)
            for roles in product(range(3), repeat=fold_count)
            if set(roles) == {0, 1, 2}
        )
        selected_roles = min(role_candidates, key=role_cost)
        fit_folds = tuple(
            fold for fold, role in enumerate(selected_roles) if role == 0
        )
        verification_role_folds = tuple(
            fold for fold, role in enumerate(selected_roles) if role == 1
        )
        audit_role_folds = tuple(
            fold for fold, role in enumerate(selected_roles) if role == 2
        )
    else:
        fit_folds = fixed_fit_folds
        verification_role_folds = fixed_verification_folds
        audit_role_folds = (fixed_audit_fold,)

    verification = np.flatnonzero(
        np.isin(
            assignment,
            np.asarray(verification_role_folds, dtype=np.int64),
        )
    ).astype(np.int64)
    final_audit = np.flatnonzero(
        np.isin(assignment, np.asarray(audit_role_folds, dtype=np.int64))
    ).astype(np.int64)
    fit = np.flatnonzero(
        np.isin(assignment, np.asarray(fit_folds, dtype=np.int64))
    ).astype(np.int64)

    partition_tracks = []
    for indices in (fit, verification, final_audit):
        local_valid = valid[indices]
        partition_tracks.append(
            set(int(value) for value in tracks[indices][local_valid].tolist())
        )
    if any(
        partition_tracks[first] & partition_tracks[second]
        for first in range(3)
        for second in range(first + 1, 3)
    ):
        raise RuntimeError("physical track crosses component cross-fit partitions")
    if maplet_voxel_size_m is not None:
        partition_voxels: list[set[tuple[int, int, int]]] = []
        for indices in (fit, verification, final_audit):
            local_voxels: set[tuple[int, int, int]] = set()
            for row in indices.tolist():
                for column in np.flatnonzero(valid[row]).tolist():
                    local_voxels.add(
                        tuple(
                            np.floor(
                                xyz[row, column] / float(maplet_voxel_size_m)
                            ).astype(np.int64).tolist()
                        )
                    )
            partition_voxels.append(local_voxels)
        if any(
            partition_voxels[first] & partition_voxels[second]
            for first in range(3)
            for second in range(first + 1, 3)
        ):
            raise RuntimeError("maplet voxel crosses component cross-fit partitions")
    partition_maplets: list[set[int]] | None = None
    maplet_overlap_counts: list[int] | None = None
    if pool.has_explicit_maplet_clusters:
        partition_maplets: list[set[int]] = []
        clusters = np.asarray(pool.maplet_cluster_ids, dtype=np.int64)
        for indices in (fit, verification, final_audit):
            local_valid = valid[indices]
            partition_maplets.append(
                set(int(value) for value in clusters[indices][local_valid].tolist())
            )
        maplet_overlap_counts = [
            int(len(partition_maplets[first] & partition_maplets[second]))
            for first in range(3)
            for second in range(first + 1, 3)
        ]
        if bool(use_explicit_maplet_clusters) and any(maplet_overlap_counts):
            raise RuntimeError("explicit maplet crosses component cross-fit partitions")
    component_sizes = sorted(
        (int(len(rows)) for rows in component_rows), reverse=True
    )
    token_fold_pairs = sorted(
        (
            int(pool.token_indices[row]),
            int(assignment[row]),
        )
        for row in range(count)
    )
    token_role_pairs = [
        (int(pool.token_indices[row]), "fit") for row in fit.tolist()
    ]
    token_role_pairs.extend(
        (int(pool.token_indices[row]), "verification")
        for row in verification.tolist()
    )
    token_role_pairs.extend(
        (int(pool.token_indices[row]), "audit")
        for row in final_audit.tolist()
    )
    token_role_pairs.sort()
    audit = {
        "partition_type": "token_track_maplet_connected_component",
        "role_assignment": str(role_assignment),
        "query_group_count": int(count),
        "component_count": int(len(component_rows)),
        "component_sizes_desc": component_sizes,
        "largest_component_count": int(component_sizes[0]),
        "largest_component_fraction": float(component_sizes[0] / max(count, 1)),
        "fold_sizes": [int(value) for value in fold_sizes.tolist()],
        "fold_component_counts": [
            int(value) for value in fold_component_counts.tolist()
        ],
        "fold_grid_cell_counts": [
            int(np.count_nonzero(fold_cells[fold])) for fold in range(fold_count)
        ],
        "fit_folds": [int(value) for value in fit_folds],
        "verification_folds": [
            int(value) for value in verification_role_folds
        ],
        "audit_folds": [int(value) for value in audit_role_folds],
        "fit_count": int(fit.size),
        "verification_count": int(verification.size),
        "audit_count": int(final_audit.size),
        "minimum_heldout_count": int(min(verification.size, final_audit.size)),
        "all_folds_nonempty": bool(np.all(fold_component_counts > 0)),
        "strict_track_disjoint": True,
        "strict_maplet_disjoint": bool(use_explicit_maplet_clusters),
        "maplet_overlap_counts_fit_verify_fit_audit_verify_audit": (
            maplet_overlap_counts
        ),
        "assignment_sha256": hashlib.sha256(
            repr(token_fold_pairs).encode("ascii")
        ).hexdigest(),
        "role_manifest_sha256": hashlib.sha256(
            repr(token_role_pairs).encode("ascii")
        ).hexdigest(),
    }
    return ComponentPartitionPlan(
        fit_indices=fit,
        verification_indices=verification,
        audit_indices=final_audit,
        audit=audit,
    )


def deterministic_component_partitions(
    pool: PoseVerificationCandidatePool,
    *,
    image_width: int,
    image_height: int,
    folds: int = 4,
    verification_fold: int = 0,
    verification_fold_count: int = 1,
    final_audit_fold: int = 1,
    grid_rows: int = 4,
    grid_cols: int = 4,
    maplet_voxel_size_m: float | None = None,
    use_explicit_maplet_clusters: bool = False,
    role_assignment: str = "fixed",
    minimum_fit_count: int = 8,
    minimum_heldout_count: int = 4,
    salt: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plan = _deterministic_component_partition_plan(
        pool,
        image_width=int(image_width),
        image_height=int(image_height),
        folds=int(folds),
        verification_fold=int(verification_fold),
        verification_fold_count=int(verification_fold_count),
        final_audit_fold=int(final_audit_fold),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        maplet_voxel_size_m=maplet_voxel_size_m,
        use_explicit_maplet_clusters=bool(use_explicit_maplet_clusters),
        role_assignment=str(role_assignment),
        minimum_fit_count=int(minimum_fit_count),
        minimum_heldout_count=int(minimum_heldout_count),
        salt=int(salt),
    )
    return plan.fit_indices, plan.verification_indices, plan.audit_indices


def _camera_bearings(
    matches: Sequence[QueryTo3DMatch], camera: ColmapCamera
) -> np.ndarray:
    if not matches:
        return np.empty((0, 3), dtype=np.float64)
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for bearing construction") from exc
    xy = np.stack([match.xy for match in matches], axis=0).astype(np.float64)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    normalized = cv2.undistortPoints(
        xy.reshape(-1, 1, 2), camera_matrix, distortion
    ).reshape(-1, 2)
    bearings = np.column_stack(
        [normalized, np.ones((len(normalized),), dtype=np.float64)]
    )
    bearings /= np.maximum(np.linalg.norm(bearings, axis=1, keepdims=True), 1e-12)
    return bearings


def _pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.empty((0, 0), dtype=np.float64)
    delta = values[:, None, :] - values[None, :, :]
    return np.linalg.norm(delta, axis=2)


def _pairwise_bearing_angles(bearings: np.ndarray) -> np.ndarray:
    if len(bearings) == 0:
        return np.empty((0, 0), dtype=np.float64)
    cosine = np.clip(bearings @ bearings.T, -1.0, 1.0)
    return np.arccos(cosine)


def _robust_distance_scale(distances: np.ndarray) -> float:
    upper = distances[np.triu_indices(len(distances), k=1)]
    finite = upper[np.isfinite(upper) & (upper > 0.0)]
    return 1.0 if finite.size == 0 else max(float(np.percentile(finite, 90)), 1e-12)


def select_geometry_diverse_matches(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    max_matches: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    prefilter_multiplier: int = 3,
) -> list[QueryTo3DMatch]:
    """Select confident matches with 2D-bearing and 3D-structure coverage.

    Camera-frame depth is intentionally absent here because pose is unknown.
    World-coordinate z is not used as a proxy for depth.
    """

    limit = int(max_matches)
    if limit <= 0:
        raise ValueError("max_matches must be positive")
    if int(prefilter_multiplier) <= 0:
        raise ValueError("prefilter_multiplier must be positive")
    unique = resolve_pose_match_conflicts(matches)
    if len(unique) <= limit:
        return unique
    ordered = sorted(
        unique,
        key=lambda match: (
            -float(match.similarity),
            int(match.token_index),
            int(match.track_id),
        ),
    )
    pool_count = min(len(ordered), max(limit, limit * int(prefilter_multiplier)))
    pool = ordered[:pool_count]
    bearings = _camera_bearings(pool, camera)
    xyz = np.stack([match.xyz for match in pool], axis=0).astype(np.float64)
    bearing_distance = _pairwise_bearing_angles(bearings)
    xyz_distance = _pairwise_euclidean(xyz)
    bearing_scale = _robust_distance_scale(bearing_distance)
    xyz_scale = _robust_distance_scale(xyz_distance)
    # Rank confidence avoids treating logits, probabilities, and cosine scores
    # as calibrated to the same numerical scale.
    confidence = np.linspace(1.0, 0.0, len(pool), dtype=np.float64)
    width = max(float(camera.width), 1.0)
    height = max(float(camera.height), 1.0)
    cells = []
    for match in pool:
        x, y = np.asarray(match.xy, dtype=np.float64).reshape(2)
        cells.append(
            (
                int(np.clip(np.floor(y / height * grid_rows), 0, grid_rows - 1)),
                int(np.clip(np.floor(x / width * grid_cols), 0, grid_cols - 1)),
            )
        )

    chosen = [0]
    chosen_cells = {cells[0]}
    remaining = set(range(1, len(pool)))
    while remaining and len(chosen) < limit:
        def utility(index: int) -> tuple[float, float, float, int]:
            bearing_gain = min(float(bearing_distance[index, other]) for other in chosen)
            xyz_gain = min(float(xyz_distance[index, other]) for other in chosen)
            new_cell = float(cells[index] not in chosen_cells)
            value = (
                0.35 * confidence[index]
                + 0.25 * min(bearing_gain / bearing_scale, 1.0)
                + 0.25 * min(xyz_gain / xyz_scale, 1.0)
                + 0.15 * new_cell
            )
            return value, confidence[index], new_cell, -index

        best = max(remaining, key=utility)
        remaining.remove(best)
        chosen.append(best)
        chosen_cells.add(cells[best])
    return [pool[index] for index in chosen]


def _positive_depth_mask(
    matches: Sequence[QueryTo3DMatch], pose_w2c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not matches:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=np.float64)
    xyz = np.stack([match.xyz for match in matches], axis=0).astype(np.float64)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    depth = (xyz @ pose[:3, :3].T + pose[:3, 3])[:, 2]
    finite_positive = np.isfinite(depth) & (depth > 1e-6)
    return finite_positive, depth


def _information_spectrum(matrix: np.ndarray) -> tuple[float | None, float | None]:
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))
    if eigenvalues.size == 0 or not np.all(np.isfinite(eigenvalues)):
        return None, None
    largest = max(float(eigenvalues[-1]), 0.0)
    smallest = max(float(eigenvalues[0]), 0.0)
    if largest <= 0.0:
        return 0.0, None
    return smallest, float(largest / max(smallest, 1e-15))


def _marginal_information_block(
    primary: np.ndarray,
    cross: np.ndarray,
    nuisance: np.ndarray,
) -> np.ndarray:
    """Return information remaining after marginalizing nuisance parameters."""

    primary_matrix = np.asarray(primary, dtype=np.float64)
    cross_matrix = np.asarray(cross, dtype=np.float64)
    nuisance_matrix = np.asarray(nuisance, dtype=np.float64)
    marginalized = (
        primary_matrix
        - cross_matrix @ np.linalg.pinv(nuisance_matrix, rcond=1e-12) @ cross_matrix.T
    )
    return 0.5 * (marginalized + marginalized.T)


def pose_information_diagnostics(
    pose_w2c: np.ndarray | None,
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    residual_sigma_px: float = 2.0,
) -> dict[str, float | int | None]:
    """Audit local pose observability for one resolved correspondence set.

    The information matrix uses a pinhole first-order Jacobian at the supplied
    pose. It is intentionally an inference-safe geometry diagnostic: no target
    pose or target residual enters any returned quantity.
    """

    if pose_w2c is None:
        return {"information_match_count": 0}
    sigma = float(residual_sigma_px)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("residual_sigma_px must be positive")
    values = list(matches)
    if not values:
        return {"information_match_count": 0}

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    xyz_world = np.stack([match.xyz for match in values], axis=0).astype(np.float64)
    xyz_camera = xyz_world @ pose[:3, :3].T + pose[:3, 3]
    finite_positive = np.all(np.isfinite(xyz_camera), axis=1) & (
        xyz_camera[:, 2] > 1e-6
    )
    xyz_world = xyz_world[finite_positive]
    xyz_camera = xyz_camera[finite_positive]
    valid_matches = [
        match for match, accepted in zip(values, finite_positive) if bool(accepted)
    ]
    count = int(len(xyz_camera))
    output: dict[str, float | int | None] = {
        "information_match_count": count,
        "translation_information_min_eigenvalue": None,
        "translation_information_condition": None,
        "rotation_information_min_eigenvalue": None,
        "rotation_information_condition": None,
        "joint_information_min_eigenvalue": None,
        "joint_information_condition": None,
        "bearing_max_angle_deg": None,
        "camera_depth_span_m": None,
        "camera_depth_span_ratio": None,
        "xyz_second_singular_ratio": None,
        "xyz_third_singular_ratio": None,
    }
    if count == 0:
        return output

    depths = xyz_camera[:, 2]
    depth_span = float(np.max(depths) - np.min(depths))
    output["camera_depth_span_m"] = depth_span
    output["camera_depth_span_ratio"] = float(
        depth_span / max(float(np.median(depths)), 1e-12)
    )
    bearings = _camera_bearings(valid_matches, camera)
    if len(bearings) >= 2:
        angles = _pairwise_bearing_angles(bearings)
        output["bearing_max_angle_deg"] = float(np.degrees(np.max(angles)))

    if count >= 2:
        centered_xyz = xyz_world - np.mean(xyz_world, axis=0, keepdims=True)
        singular = np.linalg.svd(centered_xyz, compute_uv=False)
        largest = float(singular[0]) if singular.size else 0.0
        if largest > 1e-12:
            second = float(singular[1]) if singular.size >= 2 else 0.0
            third = float(singular[2]) if singular.size >= 3 else 0.0
            output["xyz_second_singular_ratio"] = second / largest
            output["xyz_third_singular_ratio"] = third / largest

    if count < 4:
        return output

    camera_matrix, _distortion = camera_matrix_and_distortion(camera)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    jacobian = np.empty((2 * count, 6), dtype=np.float64)
    for index, point in enumerate(xyz_camera):
        x, y, z = point.tolist()
        projection = np.asarray(
            [[fx / z, 0.0, -fx * x / (z * z)],
             [0.0, fy / z, -fy * y / (z * z)]],
            dtype=np.float64,
        )
        skew = np.asarray(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )
        jacobian[2 * index : 2 * index + 2, :3] = projection / sigma
        jacobian[2 * index : 2 * index + 2, 3:] = (
            projection @ (-skew) / sigma
        )

    joint_information = jacobian.T @ jacobian
    translation_conditional = joint_information[:3, :3]
    rotation_conditional = joint_information[3:, 3:]
    cross_information = joint_information[:3, 3:]
    translation_information = _marginal_information_block(
        translation_conditional,
        cross_information,
        rotation_conditional,
    )
    rotation_information = _marginal_information_block(
        rotation_conditional,
        cross_information.T,
        translation_conditional,
    )
    translation_min, translation_condition = _information_spectrum(
        translation_information
    )
    rotation_min, rotation_condition = _information_spectrum(rotation_information)
    joint_min, joint_condition = _information_spectrum(joint_information)
    output.update(
        {
            "translation_information_min_eigenvalue": translation_min,
            "translation_information_condition": translation_condition,
            "rotation_information_min_eigenvalue": rotation_min,
            "rotation_information_condition": rotation_condition,
            "joint_information_min_eigenvalue": joint_min,
            "joint_information_condition": joint_condition,
        }
    )
    return output


def grouped_pose_observability_gate_failures(
    pose_w2c: np.ndarray | None,
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    config: GroupedCandidatePnPConfig,
) -> tuple[str, ...]:
    """Return inference-only P24 veto reasons for one generated pose."""

    thresholds = (
        float(config.prosac_min_translation_information_eigenvalue),
        float(config.prosac_max_translation_information_condition),
        float(config.prosac_max_joint_information_condition),
        float(config.prosac_min_depth_span_ratio),
        float(config.prosac_min_xyz_third_singular_ratio),
    )
    if not any(value > 0.0 for value in thresholds):
        return ()
    diagnostics = pose_information_diagnostics(
        pose_w2c,
        matches,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
    )
    failures: list[str] = []
    minimum_checks = (
        (
            "translation_information_min_eigenvalue",
            float(config.prosac_min_translation_information_eigenvalue),
        ),
        ("camera_depth_span_ratio", float(config.prosac_min_depth_span_ratio)),
        (
            "xyz_third_singular_ratio",
            float(config.prosac_min_xyz_third_singular_ratio),
        ),
    )
    maximum_checks = (
        (
            "translation_information_condition",
            float(config.prosac_max_translation_information_condition),
        ),
        (
            "joint_information_condition",
            float(config.prosac_max_joint_information_condition),
        ),
    )
    for name, threshold in minimum_checks:
        value = diagnostics.get(name)
        if threshold > 0.0 and (
            value is None
            or not np.isfinite(float(value))
            or float(value) < threshold
        ):
            failures.append(name)
    for name, threshold in maximum_checks:
        value = diagnostics.get(name)
        if threshold > 0.0 and (
            value is None
            or not np.isfinite(float(value))
            or float(value) > threshold
        ):
            failures.append(name)
    return tuple(failures)


def _grid_cell_count(
    matches: Sequence[QueryTo3DMatch],
    mask: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    grid_rows: int,
    grid_cols: int,
) -> int:
    cells: set[tuple[int, int]] = set()
    for match, accepted in zip(matches, np.asarray(mask, dtype=bool)):
        if not bool(accepted):
            continue
        x, y = np.asarray(match.xy, dtype=np.float64).reshape(2)
        cells.add(
            (
                int(
                    np.clip(
                        np.floor(y / float(image_height) * int(grid_rows)),
                        0,
                        int(grid_rows) - 1,
                    )
                ),
                int(
                    np.clip(
                        np.floor(x / float(image_width) * int(grid_cols)),
                        0,
                        int(grid_cols) - 1,
                    )
                ),
            )
        )
    return len(cells)


def verify_pose_hypothesis(
    pose_w2c: np.ndarray | None,
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    strict_threshold_px: float = 2.0,
    loose_threshold_px: float = 5.0,
    grid_rows: int = 4,
    grid_cols: int = 4,
    expected_count: int | None = None,
) -> HypothesisVerification | None:
    if pose_w2c is None or len(matches) == 0:
        return None
    residuals = match_reprojection_errors(matches, pose_w2c, camera)
    positive, depths = _positive_depth_mask(matches, pose_w2c)
    finite = np.isfinite(residuals) & np.isfinite(depths)
    strict = finite & positive & (residuals <= float(strict_threshold_px))
    loose = finite & positive & (residuals <= float(loose_threshold_px))
    total_count = len(matches) if expected_count is None else int(expected_count)
    if total_count < len(matches):
        raise ValueError("expected_count cannot be smaller than the supplied matches")
    clipped = np.minimum(
        np.where(finite & positive, residuals, float(loose_threshold_px) * 4.0),
        float(loose_threshold_px) * 4.0,
    )
    if total_count > len(matches):
        clipped = np.concatenate(
            [
                clipped,
                np.full(
                    (total_count - len(matches),),
                    float(loose_threshold_px) * 4.0,
                    dtype=np.float64,
                ),
            ]
        )
    soft = np.where(
        finite & positive,
        np.exp(-0.5 * np.square(residuals / float(strict_threshold_px))),
        0.0,
    )
    positive_depths = depths[finite & positive]
    return HypothesisVerification(
        verification_count=int(total_count),
        finite_count=int(np.sum(finite)),
        positive_depth_count=int(np.sum(finite & positive)),
        positive_depth_ratio=float(np.sum(finite & positive) / max(total_count, 1)),
        strict_inlier_count=int(np.sum(strict)),
        loose_inlier_count=int(np.sum(loose)),
        strict_grid_cell_count=_grid_cell_count(
            matches,
            strict,
            image_width=int(camera.width),
            image_height=int(camera.height),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
        ),
        loose_grid_cell_count=_grid_cell_count(
            matches,
            loose,
            image_width=int(camera.width),
            image_height=int(camera.height),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
        ),
        soft_consensus=float(np.sum(soft)),
        clipped_median_residual_px=float(np.median(clipped)),
        depth_range_m=(
            None
            if positive_depths.size == 0
            else float(np.max(positive_depths) - np.min(positive_depths))
        ),
        **pose_information_diagnostics(
            pose_w2c,
            matches,
            camera,
            residual_sigma_px=float(strict_threshold_px),
        ),
    )


def _candidate_pool_projected_xy(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    return project_candidate_xyz(
        pool.xyz,
        pool.valid_mask,
        pose_w2c,
        camera,
    )


def _candidate_pool_reprojection_residuals(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    projected, valid = _candidate_pool_projected_xy(pool, pose_w2c, camera)
    residuals = np.linalg.norm(projected - pool.xy[:, None, :], axis=2)
    residuals[~valid] = np.inf
    return residuals, valid


def _mass_preserving_geometry_mixed_probabilities(
    pool: PoseVerificationCandidatePool,
    *,
    mix_weight: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mix pose-free geometry evidence within retained candidate mass only."""

    weight = float(mix_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("geometry mix weight must be in [0, 1]")
    probabilities = np.where(
        pool.valid_mask, np.asarray(pool.descriptor_scores, dtype=np.float64), 0.0
    ).copy()
    geometry_probability = np.asarray(
        pool.measurement_geometry_probabilities, dtype=np.float64
    )
    geometry_evidence = np.isfinite(geometry_probability) & pool.valid_mask
    if weight <= 0.0:
        return probabilities, geometry_evidence, int(np.sum(geometry_evidence))
    for row in range(pool.query_count):
        columns = np.flatnonzero(pool.valid_mask[row])
        retained_mass = float(np.sum(probabilities[row, columns]))
        if len(columns) == 0 or retained_mass <= 0.0:
            continue
        relative_prior = probabilities[row, columns] / retained_mass
        evidence_values = relative_prior.copy()
        measured = geometry_evidence[row, columns]
        evidence_values[measured] = geometry_probability[row, columns][measured]
        log_weight = (
            (1.0 - weight) * np.log(np.maximum(relative_prior, 1e-12))
            + weight * np.log(np.maximum(evidence_values, 1e-12))
        )
        log_weight -= float(np.max(log_weight))
        normalized = np.exp(log_weight)
        normalized /= max(float(np.sum(normalized)), 1e-12)
        probabilities[row, columns] = retained_mass * normalized
    return probabilities, geometry_evidence, int(np.sum(geometry_evidence))


def mass_preserving_tempered_identity_probabilities(
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Calibrate conditional identity mass without changing explicit null mass."""

    values = np.asarray(probabilities, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("identity probabilities and validity mask differ")
    value = float(temperature)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("identity prior temperature must be finite and positive")
    if np.any(values[valid] < 0.0) or np.any(~np.isfinite(values[valid])):
        raise ValueError("identity probabilities must be finite and non-negative")

    output = np.where(valid, values, 0.0).copy()
    inverse_temperature = 1.0 / value
    for row in range(len(output)):
        columns = np.flatnonzero(valid[row] & (output[row] > 0.0))
        retained_mass = float(np.sum(output[row, columns]))
        if len(columns) == 0 or retained_mass <= 0.0:
            continue
        logits = np.log(np.maximum(output[row, columns], 1e-300))
        logits *= inverse_temperature
        logits -= float(np.max(logits))
        conditional = np.exp(logits)
        conditional /= max(float(np.sum(conditional)), 1e-300)
        output[row, columns] = retained_mass * conditional
    return output


def fixed_posterior_group_log_likelihood_statistics(
    log_likelihood: np.ndarray,
    xy: np.ndarray,
) -> dict[str, float | int]:
    """Summarize one immutable likelihood value per query group.

    Every statistic uses the same pose-independent group denominator. The
    spatial median-of-means partitions groups by fixed query coordinates, not
    by pose residuals or inlier membership.
    """

    values = np.asarray(log_likelihood, dtype=np.float64).reshape(-1)
    coordinates = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(values) == 0:
        raise ValueError("fixed-posterior statistics require at least one group")
    if len(coordinates) != len(values):
        raise ValueError("fixed-posterior likelihood and query coordinates differ")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(coordinates)):
        raise ValueError("fixed-posterior group statistics require finite inputs")

    ordered = np.sort(values)
    count = int(len(ordered))
    sample_std = float(np.std(ordered, ddof=1)) if count > 1 else 0.0
    standard_error = float(sample_std / np.sqrt(float(count)))

    trim_count = int(np.floor(0.1 * count))
    trimmed = (
        ordered[trim_count : count - trim_count]
        if trim_count > 0 and 2 * trim_count < count
        else ordered
    )
    worst_quartile_count = max(1, int(np.ceil(0.25 * count)))

    median_xy = np.median(coordinates, axis=0)
    spatial_cells = (
        (coordinates[:, 0] > median_xy[0]).astype(np.int64)
        + 2 * (coordinates[:, 1] > median_xy[1]).astype(np.int64)
    )
    spatial_cell_means = np.asarray(
        [
            np.mean(values[spatial_cells == cell])
            for cell in range(4)
            if np.any(spatial_cells == cell)
        ],
        dtype=np.float64,
    )

    mean = float(np.mean(ordered))
    return {
        "log_likelihood_std": sample_std,
        "log_likelihood_standard_error": standard_error,
        "log_likelihood_median": float(np.median(ordered)),
        "log_likelihood_trimmed_mean_10": float(np.mean(trimmed)),
        "log_likelihood_worst_quartile_mean": float(
            np.mean(ordered[:worst_quartile_count])
        ),
        "log_likelihood_lcb95": float(mean - 1.96 * standard_error),
        "spatial_median_of_means_2x2": float(np.median(spatial_cell_means)),
        "spatial_mom_cell_count": int(len(spatial_cell_means)),
    }


def fixed_posterior_pose_log_likelihood(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    residual_sigma_px: float = 2.0,
    candidate_outlier_likelihood: float = 1e-4,
    null_likelihood: float = 1.0,
    probability_mass_atol: float = 2e-4,
    spatial_evidence_weight: float | None = None,
) -> dict[str, float | int]:
    """Marginalize immutable candidate and null probabilities under a pose."""

    sigma = float(residual_sigma_px)
    outlier = float(candidate_outlier_likelihood)
    null_value = float(null_likelihood)
    if sigma <= 0.0:
        raise ValueError("residual_sigma_px must be positive")
    if not 0.0 < outlier <= 1.0:
        raise ValueError("candidate_outlier_likelihood must be in (0, 1]")
    if not 0.0 < null_value <= 1.0:
        raise ValueError("null_likelihood must be in (0, 1]")
    geometry_mix_weight = float(pool.geometry_prior_mix_weight)
    probabilities, geometry_evidence, geometry_evidence_count = (
        _mass_preserving_geometry_mixed_probabilities(
            pool,
            mix_weight=geometry_mix_weight,
        )
    )
    identity_prior_temperature = float(pool.identity_prior_temperature)
    probabilities = mass_preserving_tempered_identity_probabilities(
        probabilities,
        pool.valid_mask,
        temperature=identity_prior_temperature,
    )
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(
            "fixed-posterior likelihood requires candidate probabilities in [0, 1]"
        )
    null_scores = np.asarray(pool.null_scores, dtype=np.float64)
    if np.any(null_scores < 0.0) or np.any(null_scores > 1.0):
        raise ValueError("fixed-posterior likelihood requires null probabilities in [0, 1]")
    spatial_calibration_weight = float(
        pool.spatial_geometry_calibration_weight
    )
    total_mass = np.sum(probabilities, axis=1) + null_scores
    mass_error = np.abs(total_mass - 1.0)
    if np.any(mass_error > float(probability_mass_atol)):
        raise ValueError(
            "candidate and null probabilities do not preserve joint-softmax mass"
        )
    pose_evidence = candidate_pose_evidence(
        pool,
        pose_w2c,
        camera,
        residual_sigma_px=sigma,
        outlier_likelihood=outlier,
        spatial_evidence_weight=spatial_evidence_weight,
    )
    candidate_likelihood = pose_evidence.candidate_likelihoods
    spatial_candidate_count = int(np.sum(pose_evidence.spatial_candidate_mask))
    spatial_calibrated_candidate_count = int(
        np.sum(pose_evidence.spatial_calibrated_mask & geometry_evidence)
    )
    group_likelihood = (
        null_scores * null_value
        + np.sum(probabilities * candidate_likelihood, axis=1)
    )
    log_likelihood = np.log(np.maximum(group_likelihood, 1e-12))
    base_group_likelihood = (
        null_scores * null_value
        + np.sum(
            probabilities * pose_evidence.base_candidate_likelihoods,
            axis=1,
        )
    )
    spatial_log_likelihood_gain = log_likelihood - np.log(
        np.maximum(base_group_likelihood, 1e-12)
    )
    null_evidence_fraction = np.divide(
        null_scores * null_value,
        group_likelihood,
        out=np.zeros_like(group_likelihood),
        where=group_likelihood > 1e-12,
    )
    candidate_inlier_evidence = np.sum(
        probabilities
        * candidate_likelihood
        * pose_evidence.candidate_inlier_probabilities,
        axis=1,
    )
    candidate_inlier_evidence_fraction = np.divide(
        candidate_inlier_evidence,
        group_likelihood,
        out=np.zeros_like(group_likelihood),
        where=group_likelihood > 1e-12,
    )
    retained_candidate_mass = np.sum(probabilities, axis=1)
    conditional = np.divide(
        probabilities,
        retained_candidate_mass[:, None],
        out=np.zeros_like(probabilities),
        where=retained_candidate_mass[:, None] > 1e-12,
    )
    identity_entropy = -np.sum(
        np.where(
            conditional > 0.0,
            conditional * np.log(np.maximum(conditional, 1e-300)),
            0.0,
        ),
        axis=1,
    )
    informative = (1.0 - null_scores) > 1e-3
    distribution_statistics = fixed_posterior_group_log_likelihood_statistics(
        log_likelihood,
        pool.xy,
    )
    return {
        "log_likelihood_sum": float(np.sum(log_likelihood)),
        "log_likelihood_mean": float(np.mean(log_likelihood)),
        **distribution_statistics,
        "effective_group_count": int(np.sum(informative)),
        "mass_max_abs_error": float(np.max(mass_error)),
        "spatial_candidate_count": int(spatial_candidate_count),
        "geometry_prior_evidence_count": int(geometry_evidence_count),
        "geometry_prior_mix_weight": float(geometry_mix_weight),
        "identity_prior_temperature": float(identity_prior_temperature),
        "identity_prior_entropy_mean": float(np.mean(identity_entropy)),
        "identity_prior_effective_candidate_count_mean": float(
            np.mean(np.exp(identity_entropy))
        ),
        "retained_candidate_mass_mean": float(
            np.mean(retained_candidate_mass)
        ),
        "null_evidence_fraction_mean": float(
            np.mean(null_evidence_fraction)
        ),
        "candidate_inlier_evidence_fraction_mean": float(
            np.mean(candidate_inlier_evidence_fraction)
        ),
        "spatial_log_likelihood_gain_mean": float(
            np.mean(spatial_log_likelihood_gain)
        ),
        "spatial_calibrated_candidate_count": int(
            spatial_calibrated_candidate_count
        ),
        "spatial_geometry_calibration_weight": float(
            spatial_calibration_weight
        ),
    }


def candidate_relation_neighbor_edges(
    pool: PoseVerificationCandidatePool,
    *,
    neighbor_k: int,
) -> np.ndarray:
    """Build a deterministic undirected query-neighbor graph once per pool."""

    count = int(pool.query_count)
    requested = int(neighbor_k)
    if requested < 0:
        raise ValueError("relation neighbor count must be non-negative")
    if count < 2 or requested == 0:
        return np.empty((0, 2), dtype=np.int64)
    k = min(requested, count - 1)
    xy = np.asarray(pool.xy, dtype=np.float64).reshape(count, 2)
    squared_distance = np.sum(
        np.square(xy[:, None, :] - xy[None, :, :]), axis=2
    )
    np.fill_diagonal(squared_distance, np.inf)
    nearest = np.argsort(squared_distance, axis=1, kind="mergesort")[:, :k]
    sources = np.repeat(np.arange(count, dtype=np.int64), k)
    targets = nearest.reshape(-1).astype(np.int64, copy=False)
    edges = np.column_stack(
        [np.minimum(sources, targets), np.maximum(sources, targets)]
    )
    return np.unique(edges, axis=0)


def fixed_posterior_pairwise_relation_log_likelihood(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    residual_sigma_px: float = 2.0,
    candidate_outlier_likelihood: float = 1e-4,
    null_likelihood: float = 1.0,
    relation_sigma_px: float = 4.0,
    relation_outlier_likelihood: float = 1e-3,
    neighbor_k: int = 4,
    neighbor_edges: np.ndarray | None = None,
    probability_mass_atol: float = 2e-4,
    spatial_evidence_weight: float | None = None,
) -> dict[str, float | int]:
    """Evaluate a candidate-marginalized pairwise relation pseudolikelihood.

    The returned value is a log likelihood ratio against the same immutable
    unary candidate/null evidence with every pair factor set to one. Candidate
    identity is therefore marginalized rather than selected. Edges that touch
    the explicit null state are neutral, and invalid or duplicate-track pairs
    receive the configured outlier floor.

    This is deliberately an audit statistic, not an independent calibrated
    likelihood: neighboring edges share query groups and the projected
    displacement is derived from the same pose as the unary reprojection term.
    """

    relation_sigma = float(relation_sigma_px)
    relation_outlier = float(relation_outlier_likelihood)
    null_value = float(null_likelihood)
    if relation_sigma <= 0.0:
        raise ValueError("relation_sigma_px must be positive")
    if not 0.0 < relation_outlier <= 1.0:
        raise ValueError("relation_outlier_likelihood must be in (0, 1]")
    if not 0.0 < null_value <= 1.0:
        raise ValueError("null_likelihood must be in (0, 1]")

    probabilities, _geometry_evidence, _geometry_count = (
        _mass_preserving_geometry_mixed_probabilities(
            pool,
            mix_weight=float(pool.geometry_prior_mix_weight),
        )
    )
    probabilities = mass_preserving_tempered_identity_probabilities(
        probabilities,
        pool.valid_mask,
        temperature=float(pool.identity_prior_temperature),
    )
    null_scores = np.asarray(pool.null_scores, dtype=np.float64)
    total_mass = np.sum(probabilities, axis=1) + null_scores
    mass_error = np.abs(total_mass - 1.0)
    if np.any(mass_error > float(probability_mass_atol)):
        raise ValueError(
            "candidate and null probabilities do not preserve joint-softmax mass"
        )

    edges = (
        candidate_relation_neighbor_edges(pool, neighbor_k=int(neighbor_k))
        if neighbor_edges is None
        else np.asarray(neighbor_edges, dtype=np.int64)
    )
    if edges.size == 0:
        return {
            "pair_count": 0,
            "effective_pair_count": 0,
            "log_likelihood_ratio_sum": 0.0,
            "log_likelihood_ratio_mean": 0.0,
            "mass_max_abs_error": float(np.max(mass_error)),
        }
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("relation neighbor edges must have shape [E, 2]")
    if np.any(edges < 0) or np.any(edges >= int(pool.query_count)):
        raise ValueError("relation neighbor edge index is outside the candidate pool")
    if np.any(edges[:, 0] >= edges[:, 1]):
        raise ValueError("relation neighbor edges must be unique ordered pairs")
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("relation neighbor edges must not contain duplicates")

    pose_evidence = candidate_pose_evidence(
        pool,
        pose_w2c,
        camera,
        residual_sigma_px=float(residual_sigma_px),
        outlier_likelihood=float(candidate_outlier_likelihood),
        spatial_evidence_weight=spatial_evidence_weight,
    )
    candidate_weight = probabilities * pose_evidence.candidate_likelihoods
    null_weight = null_scores * null_value
    candidate_sum = np.sum(candidate_weight, axis=1)
    group_evidence = candidate_sum + null_weight

    first = edges[:, 0]
    second = edges[:, 1]
    projected_first = pose_evidence.projected_xy[first, :, None, :]
    projected_second = pose_evidence.projected_xy[second, None, :, :]
    observed_delta = (
        np.asarray(pool.xy, dtype=np.float64)[second]
        - np.asarray(pool.xy, dtype=np.float64)[first]
    )[:, None, None, :]
    predicted_delta = projected_second - projected_first
    relation_residual = np.linalg.norm(predicted_delta - observed_delta, axis=3)

    pair_projection_valid = (
        pose_evidence.projection_valid[first, :, None]
        & pose_evidence.projection_valid[second, None, :]
    )
    distinct_track = (
        pool.track_ids[first, :, None] != pool.track_ids[second, None, :]
    )
    pair_valid = pair_projection_valid & distinct_track
    relation_factor = np.full(
        relation_residual.shape, relation_outlier, dtype=np.float64
    )
    relation_factor[pair_valid] = relation_outlier + (1.0 - relation_outlier) * np.exp(
        -0.5 * np.square(relation_residual[pair_valid] / relation_sigma)
    )

    candidate_pair_weight = (
        candidate_weight[first, :, None] * candidate_weight[second, None, :]
    )
    related_candidate_evidence = np.sum(
        candidate_pair_weight * relation_factor, axis=(1, 2)
    )
    null_touching_evidence = (
        null_weight[first] * null_weight[second]
        + null_weight[first] * candidate_sum[second]
        + candidate_sum[first] * null_weight[second]
    )
    related_evidence = related_candidate_evidence + null_touching_evidence
    independent_evidence = group_evidence[first] * group_evidence[second]
    related_evidence = np.minimum(related_evidence, independent_evidence)
    log_ratio = np.log(np.maximum(related_evidence, 1e-12)) - np.log(
        np.maximum(independent_evidence, 1e-12)
    )
    candidate_pair_mass = candidate_sum[first] * candidate_sum[second]
    effective = (candidate_pair_mass > 1e-12) & np.any(pair_valid, axis=(1, 2))
    effective_log_ratio = log_ratio[effective]
    return {
        "pair_count": int(len(edges)),
        "effective_pair_count": int(np.sum(effective)),
        "log_likelihood_ratio_sum": float(np.sum(effective_log_ratio)),
        "log_likelihood_ratio_mean": (
            0.0
            if len(effective_log_ratio) == 0
            else float(np.mean(effective_log_ratio))
        ),
        "mass_max_abs_error": float(np.max(mass_error)),
    }


def _descriptor_rank_scores(pool: PoseVerificationCandidatePool) -> np.ndarray:
    ranks = np.zeros(pool.track_ids.shape, dtype=np.float64)
    for row in range(pool.query_count):
        columns = np.flatnonzero(pool.valid_mask[row])
        if len(columns) == 0:
            continue
        order = columns[
            np.argsort(-pool.descriptor_scores[row, columns], kind="mergesort")
        ]
        values = (
            np.ones((1,), dtype=np.float64)
            if len(order) == 1
            else np.linspace(1.0, 0.0, len(order), dtype=np.float64)
        )
        ranks[row, order] = values
    return ranks


def _resolve_pose_guided_candidate_pool_with_matrices(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    residual_sigma_px: float = 2.0,
    hard_threshold_px: float = 8.0,
    descriptor_rank_weight: float = 0.02,
) -> tuple[list[QueryTo3DMatch], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve top-L candidates under a pose with one physical track per image."""

    sigma = float(residual_sigma_px)
    hard = float(hard_threshold_px)
    descriptor_weight = float(descriptor_rank_weight)
    if sigma <= 0.0 or hard <= 0.0 or descriptor_weight < 0.0:
        raise ValueError("invalid pose-guided candidate-pool configuration")
    if pool.query_count == 0:
        shape = pool.track_ids.shape
        return (
            [],
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            np.full(shape, np.inf, dtype=np.float64),
            np.zeros(shape, dtype=bool),
        )
    residuals, projection_valid = _candidate_pool_reprojection_residuals(
        pool, pose_w2c, camera
    )
    valid = projection_valid & (residuals <= hard)
    geometric = np.exp(-0.5 * np.square(residuals / sigma))
    utility = geometric + descriptor_weight * _descriptor_rank_scores(pool)
    utility[~valid] = -np.inf
    selected = resolve_global_query_track_assignment(
        pool.track_ids,
        utility,
        valid_mask=valid,
        dustbin_score=0.0,
    )
    matches: list[QueryTo3DMatch] = []
    selected_residuals = np.full((pool.query_count,), np.inf, dtype=np.float64)
    for row, column in enumerate(selected.tolist()):
        if int(column) < 0:
            continue
        selected_residuals[row] = float(residuals[row, column])
        matches.append(
            QueryTo3DMatch(
                token_index=int(pool.token_indices[row]),
                xy=np.asarray(pool.xy[row], dtype=np.float64),
                track_id=int(pool.track_ids[row, column]),
                xyz=np.asarray(pool.xyz[row, column], dtype=np.float64),
                similarity=float(utility[row, column]),
                ratio=0.0,
                landmark_variance=0.0,
                source="pose_guided_topl_candidate_pool",
                prototype_id=int(pool.prototype_ids[row, column]),
            )
        )
    return matches, selected, selected_residuals, residuals, projection_valid


def resolve_pose_guided_candidate_pool(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    residual_sigma_px: float = 2.0,
    hard_threshold_px: float = 8.0,
    descriptor_rank_weight: float = 0.02,
) -> tuple[list[QueryTo3DMatch], np.ndarray, np.ndarray]:
    """Resolve top-L candidates under a pose with one physical track per image."""

    matches, selected, residuals, _all_residuals, _projection_valid = (
        _resolve_pose_guided_candidate_pool_with_matrices(
            pool,
            pose_w2c,
            camera,
            residual_sigma_px=float(residual_sigma_px),
            hard_threshold_px=float(hard_threshold_px),
            descriptor_rank_weight=float(descriptor_rank_weight),
        )
    )
    return matches, selected, residuals


def _generated_pose_observability_failures(
    pose_w2c: np.ndarray | None,
    sample_matches: Sequence[QueryTo3DMatch],
    fit_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    config: GroupedCandidatePnPConfig,
) -> tuple[str, ...]:
    """Evaluate P24 on the configured, explicitly named evidence set."""

    thresholds = (
        float(config.prosac_min_translation_information_eigenvalue),
        float(config.prosac_max_translation_information_condition),
        float(config.prosac_max_joint_information_condition),
        float(config.prosac_min_depth_span_ratio),
        float(config.prosac_min_xyz_third_singular_ratio),
    )
    if pose_w2c is None or not any(value > 0.0 for value in thresholds):
        return ()
    if str(config.prosac_observability_evidence_mode) == "minimal_sample":
        evidence_matches = list(sample_matches)
    else:
        resolved, selected_columns, selected_residuals = (
            resolve_pose_guided_candidate_pool(
                fit_pool,
                pose_w2c,
                camera,
                residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
                hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
                descriptor_rank_weight=float(
                    config.candidate_pool_descriptor_rank_weight
                ),
            )
        )
        accepted_residuals = selected_residuals[selected_columns >= 0]
        consensus_mask = accepted_residuals <= float(
            config.prosac_local_consensus_px
        )
        evidence_matches = [
            match
            for match, accepted in zip(resolved, consensus_mask)
            if bool(accepted)
        ]
    return grouped_pose_observability_gate_failures(
        pose_w2c,
        evidence_matches,
        camera,
        config,
    )


def _measurement_verification_statistics(
    pool: PoseVerificationCandidatePool,
    residuals: np.ndarray,
    projection_valid: np.ndarray,
    *,
    strict_threshold_px: float,
    loose_threshold_px: float,
    residual_sigma_px: float,
) -> dict[str, float | int | None]:
    probabilities = np.asarray(
        pool.measurement_geometry_probabilities, dtype=np.float64
    )
    evidence = np.isfinite(probabilities) & pool.valid_mask
    evidence_count = int(np.sum(evidence))
    evidence_rows = int(np.sum(np.any(evidence, axis=1)))
    if evidence_count == 0:
        return {
            "measurement_evidence_count": 0,
            "measurement_evidence_fraction": 0.0,
            "measurement_probability_mean": None,
            "measurement_high_confidence_fraction": 0.0,
            "measurement_strict_probability_mass_fraction": 0.0,
            "measurement_loose_probability_mass_fraction": 0.0,
            "measurement_soft_consensus_ratio": 0.0,
            "measurement_high_confidence_strict_fraction": 0.0,
            "measurement_high_confidence_loose_fraction": 0.0,
            "measurement_high_confidence_contradiction_fraction": 0.0,
        }
    values = probabilities[evidence]
    valid_projection = projection_valid[evidence]
    evidence_residuals = residuals[evidence]
    strict = valid_projection & (
        evidence_residuals <= float(strict_threshold_px)
    )
    loose = valid_projection & (
        evidence_residuals <= float(loose_threshold_px)
    )
    probability_mass = max(float(np.sum(values)), 1e-12)
    soft = np.where(
        valid_projection,
        np.exp(
            -0.5
            * np.square(
                evidence_residuals / max(float(residual_sigma_px), 1e-8)
            )
        ),
        0.0,
    )
    high = values >= float(pool.measurement_verification_threshold)
    high_count = int(np.sum(high))
    if high_count == 0:
        high_strict = high_loose = high_contradiction = 0.0
    else:
        high_strict = float(np.sum(high & strict) / high_count)
        high_loose = float(np.sum(high & loose) / high_count)
        high_contradiction = float(np.sum(high & ~loose) / high_count)
    return {
        "measurement_evidence_count": evidence_count,
        "measurement_evidence_fraction": float(
            evidence_rows / max(pool.query_count, 1)
        ),
        "measurement_probability_mean": float(np.mean(values)),
        "measurement_high_confidence_fraction": float(high_count / evidence_count),
        "measurement_strict_probability_mass_fraction": float(
            np.sum(values * strict) / probability_mass
        ),
        "measurement_loose_probability_mass_fraction": float(
            np.sum(values * loose) / probability_mass
        ),
        "measurement_soft_consensus_ratio": float(
            np.sum(values * soft) / probability_mass
        ),
        "measurement_high_confidence_strict_fraction": high_strict,
        "measurement_high_confidence_loose_fraction": high_loose,
        "measurement_high_confidence_contradiction_fraction": high_contradiction,
    }


def verify_pose_candidate_pool(
    pose_w2c: np.ndarray | None,
    pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    residual_sigma_px: float = 2.0,
    candidate_outlier_likelihood: float = 1e-4,
    null_likelihood: float = 1.0,
    relation_neighbor_k: int = 0,
    relation_sigma_px: float = 4.0,
    relation_outlier_likelihood: float = 1e-3,
    relation_neighbor_edges: np.ndarray | None = None,
    hard_threshold_px: float = 8.0,
    descriptor_rank_weight: float = 0.02,
    strict_threshold_px: float = 2.0,
    loose_threshold_px: float = 5.0,
    grid_rows: int = 4,
    grid_cols: int = 4,
) -> HypothesisVerification | None:
    if pose_w2c is None or pool.query_count == 0:
        return None
    matches, selected, residuals, all_residuals, projection_valid = (
        _resolve_pose_guided_candidate_pool_with_matrices(
            pool,
            pose_w2c,
            camera,
            residual_sigma_px=float(residual_sigma_px),
            hard_threshold_px=float(hard_threshold_px),
            descriptor_rank_weight=float(descriptor_rank_weight),
        )
    )
    verification = verify_pose_hypothesis(
        pose_w2c,
        matches,
        camera,
        strict_threshold_px=float(strict_threshold_px),
        loose_threshold_px=float(loose_threshold_px),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        expected_count=int(pool.query_count),
    )
    if verification is None:
        return None
    measurement_statistics = _measurement_verification_statistics(
        pool,
        all_residuals,
        projection_valid,
        strict_threshold_px=float(strict_threshold_px),
        loose_threshold_px=float(loose_threshold_px),
        residual_sigma_px=float(residual_sigma_px),
    )
    fixed_posterior_statistics: dict[str, float | int] = {}
    if pool.has_explicit_null:
        fixed_posterior = fixed_posterior_pose_log_likelihood(
            pool,
            np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
            camera,
            residual_sigma_px=float(residual_sigma_px),
            candidate_outlier_likelihood=float(candidate_outlier_likelihood),
            null_likelihood=float(null_likelihood),
        )
        fixed_posterior_statistics = {
            "fixed_posterior_log_likelihood_sum": float(
                fixed_posterior["log_likelihood_sum"]
            ),
            "fixed_posterior_log_likelihood_mean": float(
                fixed_posterior["log_likelihood_mean"]
            ),
            "fixed_posterior_log_likelihood_std": float(
                fixed_posterior["log_likelihood_std"]
            ),
            "fixed_posterior_log_likelihood_standard_error": float(
                fixed_posterior["log_likelihood_standard_error"]
            ),
            "fixed_posterior_log_likelihood_median": float(
                fixed_posterior["log_likelihood_median"]
            ),
            "fixed_posterior_log_likelihood_trimmed_mean_10": float(
                fixed_posterior["log_likelihood_trimmed_mean_10"]
            ),
            "fixed_posterior_log_likelihood_worst_quartile_mean": float(
                fixed_posterior["log_likelihood_worst_quartile_mean"]
            ),
            "fixed_posterior_log_likelihood_lcb95": float(
                fixed_posterior["log_likelihood_lcb95"]
            ),
            "fixed_posterior_spatial_median_of_means_2x2": float(
                fixed_posterior["spatial_median_of_means_2x2"]
            ),
            "fixed_posterior_spatial_mom_cell_count": int(
                fixed_posterior["spatial_mom_cell_count"]
            ),
            "fixed_posterior_effective_group_count": int(
                fixed_posterior["effective_group_count"]
            ),
            "fixed_posterior_mass_max_abs_error": float(
                fixed_posterior["mass_max_abs_error"]
            ),
            "fixed_posterior_spatial_candidate_count": int(
                fixed_posterior["spatial_candidate_count"]
            ),
            "fixed_posterior_geometry_prior_evidence_count": int(
                fixed_posterior["geometry_prior_evidence_count"]
            ),
            "fixed_posterior_geometry_prior_mix_weight": float(
                fixed_posterior["geometry_prior_mix_weight"]
            ),
            "fixed_posterior_identity_prior_temperature": float(
                fixed_posterior["identity_prior_temperature"]
            ),
            "fixed_posterior_identity_prior_entropy_mean": float(
                fixed_posterior["identity_prior_entropy_mean"]
            ),
            "fixed_posterior_identity_prior_effective_candidate_count_mean": float(
                fixed_posterior[
                    "identity_prior_effective_candidate_count_mean"
                ]
            ),
            "fixed_posterior_retained_candidate_mass_mean": float(
                fixed_posterior["retained_candidate_mass_mean"]
            ),
            "fixed_posterior_null_evidence_fraction_mean": float(
                fixed_posterior["null_evidence_fraction_mean"]
            ),
            "fixed_posterior_candidate_inlier_evidence_fraction_mean": float(
                fixed_posterior[
                    "candidate_inlier_evidence_fraction_mean"
                ]
            ),
            "fixed_posterior_spatial_log_likelihood_gain_mean": float(
                fixed_posterior["spatial_log_likelihood_gain_mean"]
            ),
            "fixed_posterior_spatial_calibrated_candidate_count": int(
                fixed_posterior["spatial_calibrated_candidate_count"]
            ),
            "fixed_posterior_spatial_geometry_calibration_weight": float(
                fixed_posterior["spatial_geometry_calibration_weight"]
            ),
        }
        if int(relation_neighbor_k) > 0 or relation_neighbor_edges is not None:
            relation = fixed_posterior_pairwise_relation_log_likelihood(
                pool,
                np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4),
                camera,
                residual_sigma_px=float(residual_sigma_px),
                candidate_outlier_likelihood=float(
                    candidate_outlier_likelihood
                ),
                null_likelihood=float(null_likelihood),
                relation_sigma_px=float(relation_sigma_px),
                relation_outlier_likelihood=float(
                    relation_outlier_likelihood
                ),
                neighbor_k=int(relation_neighbor_k),
                neighbor_edges=relation_neighbor_edges,
            )
            fixed_posterior_statistics.update(
                {
                    "fixed_posterior_relation_pair_count": int(
                        relation["pair_count"]
                    ),
                    "fixed_posterior_relation_effective_pair_count": int(
                        relation["effective_pair_count"]
                    ),
                    "fixed_posterior_relation_log_likelihood_ratio_sum": float(
                        relation["log_likelihood_ratio_sum"]
                    ),
                    "fixed_posterior_relation_log_likelihood_ratio_mean": float(
                        relation["log_likelihood_ratio_mean"]
                    ),
                }
            )
    accepted_rows = np.flatnonzero(selected >= 0)
    if len(accepted_rows) == 0:
        return replace(
            verification,
            **measurement_statistics,
            **fixed_posterior_statistics,
        )
    accepted_columns = selected[accepted_rows]
    descriptor_scores = pool.descriptor_scores[accepted_rows, accepted_columns]
    rank_scores = _descriptor_rank_scores(pool)[accepted_rows, accepted_columns]
    selected_residuals = residuals[accepted_rows]
    geometric = np.exp(
        -0.5
        * np.square(
            selected_residuals / max(float(residual_sigma_px), 1e-8)
        )
    )
    utilities = geometric + float(descriptor_rank_weight) * rank_scores
    margins: list[float] = []
    for row, column in zip(accepted_rows.tolist(), accepted_columns.tolist()):
        alternatives = np.flatnonzero(pool.valid_mask[row])
        alternatives = alternatives[alternatives != int(column)]
        next_score = (
            float(pool.descriptor_scores[row, column])
            if len(alternatives) == 0
            else float(np.max(pool.descriptor_scores[row, alternatives]))
        )
        margins.append(float(pool.descriptor_scores[row, column]) - next_score)
    return replace(
        verification,
        selected_candidate_count=int(len(accepted_rows)),
        selected_candidate_fraction=float(len(accepted_rows) / max(pool.query_count, 1)),
        selected_descriptor_score_mean=float(np.mean(descriptor_scores)),
        selected_descriptor_score_median=float(np.median(descriptor_scores)),
        selected_descriptor_margin_mean=float(np.mean(margins)),
        selected_descriptor_rank_score_mean=float(np.mean(rank_scores)),
        selected_assignment_utility_mean=float(np.mean(utilities)),
        selected_reprojection_mean_px=float(np.mean(selected_residuals)),
        selected_reprojection_p90_px=float(np.quantile(selected_residuals, 0.9)),
        **measurement_statistics,
        **fixed_posterior_statistics,
    )


def _set_cv2_seed(seed: int) -> None:
    try:
        import cv2

        cv2.setRNGSeed(int(int(seed) % (2**31 - 1)))
    except ImportError:  # pragma: no cover
        return


def _select_fit_matches(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    max_matches: int,
    mode: str,
    config: VerifiedPnPConfig,
) -> list[QueryTo3DMatch]:
    if str(mode) in {"measurement_verified", "measurement_verified_refined"}:
        verified = []
        for match in matches:
            if (
                match.geometry_probability is None
                or not np.isfinite(float(match.geometry_probability))
                or float(match.geometry_probability)
                < float(config.measurement_verified_threshold)
            ):
                continue
            xy = np.asarray(match.xy, dtype=np.float64)
            if (
                str(mode) == "measurement_verified_refined"
                and match.measurement_refined_xy is not None
                and np.all(
                    np.isfinite(
                        np.asarray(match.measurement_refined_xy, dtype=np.float64)
                    )
                )
            ):
                xy = np.asarray(match.measurement_refined_xy, dtype=np.float64)
            verified.append(
                replace(
                    match,
                    xy=xy,
                    similarity=float(match.geometry_probability),
                )
            )
        cells = {
            (
                int(
                    np.clip(
                        np.floor(
                            float(match.xy[1])
                            / max(float(camera.height), 1.0)
                            * int(config.grid_rows)
                        ),
                        0,
                        int(config.grid_rows) - 1,
                    )
                ),
                int(
                    np.clip(
                        np.floor(
                            float(match.xy[0])
                            / max(float(camera.width), 1.0)
                            * int(config.grid_cols)
                        ),
                        0,
                        int(config.grid_cols) - 1,
                    )
                ),
            )
            for match in verified
        }
        if len(verified) < int(config.measurement_verified_min_matches) or len(
            cells
        ) < int(config.measurement_verified_min_grid_cells):
            return []
        return select_pose_safe_matches(
            verified,
            max_matches=min(int(max_matches), len(verified)),
            image_width=int(camera.width),
            image_height=int(camera.height),
            mode="spatial_round_robin",
        )
    if str(mode) == "geometry_diverse":
        return select_geometry_diverse_matches(
            matches,
            camera,
            max_matches=int(max_matches),
            grid_rows=int(config.grid_rows),
            grid_cols=int(config.grid_cols),
            prefilter_multiplier=int(config.geometry_prefilter_multiplier),
        )
    return select_pose_safe_matches(
        matches,
        max_matches=int(max_matches),
        image_width=int(camera.width),
        image_height=int(camera.height),
        mode=str(mode),
    )


def _empty_result(
    match_count: int,
    *,
    fit_count: int,
    verification_count: int,
    final_audit_count: int = 0,
    hypotheses: Sequence[PoseHypothesisRecord] = (),
    hypothesis_poses_w2c: Sequence[np.ndarray | None] | None = None,
    crossfit_partition_audit: dict[str, object] | None = None,
) -> VerifiedPnPResult:
    poses = (
        tuple(None for _record in hypotheses)
        if hypothesis_poses_w2c is None
        else tuple(hypothesis_poses_w2c)
    )
    if len(poses) != len(hypotheses):
        raise ValueError("hypothesis records and poses must have equal length")
    return VerifiedPnPResult(
        success=False,
        pose_w2c=None,
        inlier_mask=np.zeros((int(match_count),), dtype=bool),
        match_count=int(match_count),
        inlier_count=0,
        fit_count=int(fit_count),
        verification_count=int(verification_count),
        final_audit_count=int(final_audit_count),
        chosen_hypothesis_index=None,
        hypotheses=tuple(hypotheses),
        hypothesis_poses_w2c=poses,
        pre_refine_pose_w2c=None,
        pre_refine_verification=None,
        pre_refine_final_audit_verification=None,
        final_verification=None,
        crossfit_partition_audit=crossfit_partition_audit,
    )


def estimate_pose_with_heldout_verification(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    config: VerifiedPnPConfig = VerifiedPnPConfig(),
    query_seed: int = 0,
    candidate_pool: PoseVerificationCandidatePool | None = None,
    hypothesis_selector: HypothesisSelector | None = None,
) -> VerifiedPnPResult:
    """Fit, verify, choose, and robustly refit a pose without GT access."""

    unique = resolve_pose_match_conflicts(matches)
    if len(unique) < max(8, int(config.min_final_inliers)):
        return _empty_result(len(unique), fit_count=0, verification_count=0)
    if config.final_audit_fold is None:
        fit_indices, verification_indices = deterministic_spatial_holdout(
            unique,
            image_width=int(camera.width),
            image_height=int(camera.height),
            folds=int(config.holdout_folds),
            fold=int(config.holdout_fold),
            grid_rows=int(config.grid_rows),
            grid_cols=int(config.grid_cols),
            salt=int(query_seed),
        )
        final_audit_indices = np.empty((0,), dtype=np.int64)
    else:
        fit_indices, verification_indices, final_audit_indices = (
            deterministic_spatial_partitions(
                unique,
                image_width=int(camera.width),
                image_height=int(camera.height),
                folds=int(config.holdout_folds),
                verification_fold=int(config.holdout_fold),
                final_audit_fold=int(config.final_audit_fold),
                grid_rows=int(config.grid_rows),
                grid_cols=int(config.grid_cols),
                salt=int(query_seed),
            )
        )
    fit_pool = [unique[int(index)] for index in fit_indices]
    verification_matches = [unique[int(index)] for index in verification_indices]
    final_audit_matches = [unique[int(index)] for index in final_audit_indices]
    verification_candidate_pool = (
        None
        if candidate_pool is None
        else candidate_pool.subset_by_token_indices(
            [match.token_index for match in verification_matches]
        )
    )
    final_audit_candidate_pool = (
        None
        if candidate_pool is None or len(final_audit_matches) == 0
        else candidate_pool.subset_by_token_indices(
            [match.token_index for match in final_audit_matches]
        )
    )
    if (
        len(fit_pool) < 4
        or len(verification_matches) < 4
        or (
            config.final_audit_fold is not None
            and len(final_audit_matches) < 4
        )
    ):
        return _empty_result(
            len(unique),
            fit_count=len(fit_pool),
            verification_count=len(verification_matches),
            final_audit_count=len(final_audit_matches),
        )

    hypotheses: list[PoseHypothesisRecord] = []
    successful_poses: list[np.ndarray | None] = []
    for fit_count in config.fit_match_counts:
        for selection_mode in config.selection_modes:
            selected = _select_fit_matches(
                fit_pool,
                camera,
                max_matches=min(int(fit_count), len(fit_pool)),
                mode=str(selection_mode),
                config=config,
            )
            selected = stable_uniform_ransac_order(selected)
            for threshold in config.ransac_thresholds_px:
                for seed_offset in config.rng_seed_offsets:
                    _set_cv2_seed(int(query_seed) + int(seed_offset))
                    result = estimate_pose_pnp_ransac(
                        selected,
                        camera,
                        reprojection_error_px=float(threshold),
                        iterations=int(config.ransac_iterations),
                        refine_method="LM",
                    )
                    if verification_candidate_pool is None:
                        verification = verify_pose_hypothesis(
                            result.pose_w2c,
                            verification_matches,
                            camera,
                            strict_threshold_px=float(config.verification_strict_px),
                            loose_threshold_px=float(config.verification_loose_px),
                            grid_rows=int(config.grid_rows),
                            grid_cols=int(config.grid_cols),
                        )
                    else:
                        verification = verify_pose_candidate_pool(
                            result.pose_w2c,
                            verification_candidate_pool,
                            camera,
                            residual_sigma_px=float(
                                config.candidate_pool_residual_sigma_px
                            ),
                            hard_threshold_px=float(
                                config.candidate_pool_hard_threshold_px
                            ),
                            descriptor_rank_weight=float(
                                config.candidate_pool_descriptor_rank_weight
                            ),
                            strict_threshold_px=float(config.verification_strict_px),
                            loose_threshold_px=float(config.verification_loose_px),
                            grid_rows=int(config.grid_rows),
                            grid_cols=int(config.grid_cols),
                        )
                    hypotheses.append(
                        PoseHypothesisRecord(
                            fit_match_count_limit=int(fit_count),
                            fit_match_count=int(len(selected)),
                            selection_mode=str(selection_mode),
                            ransac_threshold_px=float(threshold),
                            rng_seed_offset=int(seed_offset),
                            solver_success=bool(result.success),
                            fit_inlier_count=int(result.inlier_count),
                            verification=verification,
                        )
                    )
                    successful_poses.append(
                        None
                        if result.pose_w2c is None
                        else np.asarray(result.pose_w2c, dtype=np.float64).reshape(4, 4)
                    )
    eligible = [
        index
        for index, record in enumerate(hypotheses)
        if record.solver_success and record.verification is not None
    ]
    if not eligible:
        return _empty_result(
            len(unique),
            fit_count=len(fit_pool),
            verification_count=len(verification_matches),
            final_audit_count=len(final_audit_matches),
            hypotheses=hypotheses,
            hypothesis_poses_w2c=successful_poses,
        )
    if hypothesis_selector is None:
        chosen_index = max(
            eligible,
            key=lambda index: (
                hypotheses[index].verification.fixed_posterior_rank_key(),  # type: ignore[union-attr]
                -int(index),
            ),
        )
    else:
        chosen_index = int(
            hypothesis_selector(hypotheses, successful_poses, eligible)
        )
        if chosen_index not in set(eligible):
            raise ValueError(
                "hypothesis_selector must return an eligible hypothesis index"
            )
    chosen_pose = successful_poses[chosen_index]
    if chosen_pose is None:
        raise RuntimeError("eligible hypothesis has no pose")
    pre_refine = hypotheses[chosen_index].verification
    audit_matches = (
        final_audit_matches if final_audit_matches else verification_matches
    )
    audit_candidate_pool = (
        final_audit_candidate_pool
        if final_audit_candidate_pool is not None
        else verification_candidate_pool
    )

    def audit_verification(pose_w2c: np.ndarray) -> HypothesisVerification | None:
        if audit_candidate_pool is None:
            return verify_pose_hypothesis(
                pose_w2c,
                audit_matches,
                camera,
                strict_threshold_px=float(config.verification_strict_px),
                loose_threshold_px=float(config.verification_loose_px),
                grid_rows=int(config.grid_rows),
                grid_cols=int(config.grid_cols),
            )
        return verify_pose_candidate_pool(
            pose_w2c,
            audit_candidate_pool,
            camera,
            residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
            hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
            descriptor_rank_weight=float(
                config.candidate_pool_descriptor_rank_weight
            ),
            strict_threshold_px=float(config.verification_strict_px),
            loose_threshold_px=float(config.verification_loose_px),
            grid_rows=int(config.grid_rows),
            grid_cols=int(config.grid_cols),
        )

    def internal_verification(
        pose_w2c: np.ndarray,
    ) -> HypothesisVerification | None:
        if verification_candidate_pool is None:
            return verify_pose_hypothesis(
                pose_w2c,
                verification_matches,
                camera,
                strict_threshold_px=float(config.verification_strict_px),
                loose_threshold_px=float(config.verification_loose_px),
                grid_rows=int(config.grid_rows),
                grid_cols=int(config.grid_cols),
            )
        return verify_pose_candidate_pool(
            pose_w2c,
            verification_candidate_pool,
            camera,
            residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
            hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
            descriptor_rank_weight=float(
                config.candidate_pool_descriptor_rank_weight
            ),
            strict_threshold_px=float(config.verification_strict_px),
            loose_threshold_px=float(config.verification_loose_px),
            grid_rows=int(config.grid_rows),
            grid_cols=int(config.grid_cols),
        )

    def full_inlier_mask(pose_w2c: np.ndarray) -> np.ndarray:
        if candidate_pool is None:
            residuals = match_reprojection_errors(unique, pose_w2c, camera)
            positive, _depths = _positive_depth_mask(unique, pose_w2c)
            return (
                np.isfinite(residuals)
                & positive
                & (residuals <= float(config.final_consensus_px))
            )
        _matches, _selected, residuals = resolve_pose_guided_candidate_pool(
            candidate_pool,
            pose_w2c,
            camera,
            residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
            hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
            descriptor_rank_weight=float(
                config.candidate_pool_descriptor_rank_weight
            ),
        )
        return np.isfinite(residuals) & (
            residuals <= float(config.final_consensus_px)
        )

    pre_refine_final_audit = audit_verification(chosen_pose)
    refinement_indices = np.asarray(
        sorted(set(fit_indices.tolist()) | set(verification_indices.tolist())),
        dtype=np.int64,
    )
    refinement_matches = [unique[int(index)] for index in refinement_indices]
    refinement_candidate_pool = (
        None
        if candidate_pool is None
        else candidate_pool.subset_by_token_indices(
            [match.token_index for match in refinement_matches]
        )
    )
    if candidate_pool is None:
        final_source_matches = refinement_matches
        final_selected_columns = None
        all_residuals = match_reprojection_errors(
            refinement_matches, chosen_pose, camera
        )
        positive, _depths = _positive_depth_mask(refinement_matches, chosen_pose)
        consensus = (
            np.isfinite(all_residuals)
            & positive
            & (all_residuals <= float(config.final_consensus_px))
        )
        result_match_count = len(unique)
    else:
        if refinement_candidate_pool is None:
            raise RuntimeError("candidate-pool refinement subset is missing")
        final_source_matches, final_selected_columns, all_residuals = (
            resolve_pose_guided_candidate_pool(
                refinement_candidate_pool,
                chosen_pose,
                camera,
                residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
                hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
                descriptor_rank_weight=float(
                    config.candidate_pool_descriptor_rank_weight
                ),
            )
        )
        consensus = np.isfinite(all_residuals) & (
            all_residuals <= float(config.final_consensus_px)
        )
        result_match_count = int(candidate_pool.query_count)
    if int(np.sum(consensus)) < int(config.min_final_inliers):
        output_inliers = full_inlier_mask(chosen_pose)
        return VerifiedPnPResult(
            success=True,
            pose_w2c=chosen_pose,
            inlier_mask=output_inliers,
            match_count=result_match_count,
            inlier_count=int(np.sum(output_inliers)),
            fit_count=len(fit_pool),
            verification_count=len(verification_matches),
            final_audit_count=len(final_audit_matches),
            chosen_hypothesis_index=int(chosen_index),
            hypotheses=tuple(hypotheses),
            hypothesis_poses_w2c=tuple(successful_poses),
            pre_refine_pose_w2c=chosen_pose,
            pre_refine_verification=pre_refine,
            pre_refine_final_audit_verification=pre_refine_final_audit,
            final_verification=pre_refine_final_audit,
        )

    if not bool(config.enable_final_refine):
        output_inliers = full_inlier_mask(chosen_pose)
        return VerifiedPnPResult(
            success=True,
            pose_w2c=chosen_pose,
            inlier_mask=output_inliers,
            match_count=result_match_count,
            inlier_count=int(np.sum(output_inliers)),
            fit_count=len(fit_pool),
            verification_count=len(verification_matches),
            final_audit_count=len(final_audit_matches),
            chosen_hypothesis_index=int(chosen_index),
            hypotheses=tuple(hypotheses),
            hypothesis_poses_w2c=tuple(successful_poses),
            pre_refine_pose_w2c=chosen_pose,
            pre_refine_verification=pre_refine,
            pre_refine_final_audit_verification=pre_refine_final_audit,
            final_verification=pre_refine_final_audit,
        )

    final_pose = chosen_pose
    refine_iterations = (
        1
        if candidate_pool is None
        else int(config.candidate_pool_refine_iterations)
    )
    for _iteration in range(refine_iterations):
        if candidate_pool is not None:
            if refinement_candidate_pool is None:
                raise RuntimeError("candidate-pool refinement subset is missing")
            final_source_matches, final_selected_columns, all_residuals = (
                resolve_pose_guided_candidate_pool(
                    refinement_candidate_pool,
                    final_pose,
                    camera,
                    residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
                    hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
                    descriptor_rank_weight=float(
                        config.candidate_pool_descriptor_rank_weight
                    ),
                )
            )
            consensus = np.isfinite(all_residuals) & (
                all_residuals <= float(config.final_consensus_px)
            )
        if candidate_pool is None:
            consensus_matches = [
                match
                for match, accepted in zip(final_source_matches, consensus)
                if bool(accepted)
            ]
            consensus_residuals = all_residuals[consensus]
        else:
            if final_selected_columns is None:
                raise RuntimeError("candidate-pool resolution lost selected columns")
            accepted_residuals = all_residuals[final_selected_columns >= 0]
            accepted_consensus = accepted_residuals <= float(
                config.final_consensus_px
            )
            consensus_matches = [
                match
                for match, accepted in zip(
                    final_source_matches, accepted_consensus
                )
                if bool(accepted)
            ]
            consensus_residuals = accepted_residuals[accepted_consensus]
        if len(consensus_matches) < int(config.min_final_inliers):
            break
        weights = np.exp(
            -0.5
            * np.square(
                consensus_residuals
                / max(float(config.final_refine_f_scale_px), 1e-6)
            )
        )
        refined: PnPResult = estimate_pose_pnp_fixed_robust(
            consensus_matches,
            camera,
            weights=weights,
            min_inliers=int(config.min_final_inliers),
            initial_pose_w2c=final_pose,
            loss="huber",
            f_scale_px=float(config.final_refine_f_scale_px),
            max_nfev=100,
        )
        if not refined.success or refined.pose_w2c is None:
            break
        final_pose = np.asarray(refined.pose_w2c, dtype=np.float64).reshape(4, 4)
    final_internal_verification = internal_verification(final_pose)
    # Refinement is an internal model-selection step. It may use the ranking
    # fold, but the final-audit fold must not decide which pose is returned.
    if (
        final_internal_verification is None
        or pre_refine is None
        or (
            verification_candidate_pool is not None
            and not _accept_grouped_final_refine(
                pre_refine,
                final_internal_verification,
                policy="fixed_posterior_likelihood_gain",
            )
        )
        or (
            verification_candidate_pool is None
            and final_internal_verification.rank_key() < pre_refine.rank_key()
        )
    ):
        final_pose = chosen_pose
    final_verification = audit_verification(final_pose)
    final_inliers = full_inlier_mask(final_pose)
    return VerifiedPnPResult(
        success=True,
        pose_w2c=final_pose,
        inlier_mask=final_inliers,
        match_count=result_match_count,
        inlier_count=int(np.sum(final_inliers)),
        fit_count=len(fit_pool),
        verification_count=len(verification_matches),
        final_audit_count=len(final_audit_matches),
        chosen_hypothesis_index=int(chosen_index),
        hypotheses=tuple(hypotheses),
        hypothesis_poses_w2c=tuple(successful_poses),
        pre_refine_pose_w2c=chosen_pose,
        pre_refine_verification=pre_refine,
        pre_refine_final_audit_verification=pre_refine_final_audit,
        final_verification=final_verification,
    )


def _candidate_pool_representatives(
    pool: PoseVerificationCandidatePool,
) -> list[QueryTo3DMatch]:
    """Build one placeholder per group for deterministic spatial partitioning."""

    matches: list[QueryTo3DMatch] = []
    for row in range(pool.query_count):
        columns = np.flatnonzero(pool.valid_mask[row])
        if columns.size == 0:
            continue
        column = int(
            columns[
                np.argmax(pool.descriptor_scores[row, columns])
            ]
        )
        matches.append(
            QueryTo3DMatch(
                token_index=int(pool.token_indices[row]),
                xy=np.asarray(pool.xy[row], dtype=np.float64),
                track_id=int(pool.track_ids[row, column]),
                xyz=np.asarray(pool.xyz[row, column], dtype=np.float64),
                similarity=float(pool.descriptor_scores[row, column]),
                ratio=0.0,
                landmark_variance=0.0,
                source="grouped_candidate_partition_representative",
                prototype_id=int(pool.prototype_ids[row, column]),
            )
        )
    return matches


def _sample_grouped_candidate_assignment(
    pool: PoseVerificationCandidatePool,
    *,
    candidate_limit: int,
    mode: str,
    temperature: float,
    seed: int,
) -> tuple[list[QueryTo3DMatch], int]:
    """Sample at most one candidate or explicit null from each query group."""

    if int(candidate_limit) <= 0:
        raise ValueError("candidate_limit must be positive")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if str(mode) not in {
        "forced_top1",
        "null_argmax",
        "geometry_mixed_argmax",
        "posterior_sample",
    }:
        raise ValueError("unsupported grouped assignment mode")
    rng = np.random.default_rng(int(seed))
    generation_scores, _geometry_evidence, _geometry_evidence_count = (
        _mass_preserving_geometry_mixed_probabilities(
            pool,
            mix_weight=float(pool.geometry_generation_mix_weight),
        )
    )
    chosen: list[tuple[int, int, float]] = []
    null_count = 0
    for row in range(pool.query_count):
        columns = np.flatnonzero(pool.valid_mask[row])
        if columns.size == 0:
            null_count += 1
            continue
        order = np.argsort(
            -pool.descriptor_scores[row, columns], kind="mergesort"
        )
        columns = columns[order[: min(int(candidate_limit), len(order))]]
        scores = np.asarray(pool.descriptor_scores[row, columns], dtype=np.float64)
        sampling_scores = np.asarray(
            generation_scores[row, columns], dtype=np.float64
        )
        null_score = float(pool.null_scores[row])
        if str(mode) == "forced_top1":
            column = int(columns[0])
            selected_score = float(scores[0])
        elif str(mode) == "null_argmax":
            if null_score >= float(scores[0]):
                null_count += 1
                continue
            column = int(columns[0])
            selected_score = float(scores[0])
        elif str(mode) == "geometry_mixed_argmax":
            selected_index = int(np.argmax(sampling_scores))
            if null_score >= float(sampling_scores[selected_index]):
                null_count += 1
                continue
            column = int(columns[selected_index])
            selected_score = float(sampling_scores[selected_index])
        else:
            if np.any(sampling_scores < 0.0):
                raise ValueError(
                    "posterior sampling requires non-negative candidate scores"
                )
            masses = np.concatenate(
                [sampling_scores, np.asarray([null_score])]
            )
            power = 1.0 / float(temperature)
            masses = np.power(np.maximum(masses, 0.0), power)
            if not np.any(masses > 0.0):
                masses[:-1] = 1.0
            probabilities = masses / np.sum(masses)
            sampled = int(rng.choice(len(probabilities), p=probabilities))
            if sampled == len(columns):
                null_count += 1
                continue
            column = int(columns[sampled])
            selected_score = float(sampling_scores[sampled])
        chosen.append((row, column, selected_score))

    # A single physical landmark cannot explain two image measurements in one
    # PnP hypothesis. Resolve these collisions before entering the solver.
    chosen.sort(
        key=lambda item: (
            -float(item[2]),
            int(pool.token_indices[item[0]]),
            int(pool.track_ids[item[0], item[1]]),
        )
    )
    unique: list[tuple[int, int, float]] = []
    used_tracks: set[int] = set()
    for item in chosen:
        track_id = int(pool.track_ids[item[0], item[1]])
        if track_id in used_tracks:
            null_count += 1
            continue
        used_tracks.add(track_id)
        unique.append(item)
    unique.sort(key=lambda item: int(pool.token_indices[item[0]]))
    matches = [
        QueryTo3DMatch(
            token_index=int(pool.token_indices[row]),
            xy=np.asarray(pool.xy[row], dtype=np.float64),
            track_id=int(pool.track_ids[row, column]),
            xyz=np.asarray(pool.xyz[row, column], dtype=np.float64),
            similarity=float(score),
            ratio=0.0,
            landmark_variance=0.0,
            source=f"grouped_candidate:{mode}",
            prototype_id=int(pool.prototype_ids[row, column]),
        )
        for row, column, score in unique
    ]
    return matches, int(null_count)


def _grouped_fit_is_degenerate(
    matches: Sequence[QueryTo3DMatch],
    camera: ColmapCamera,
    *,
    min_matches: int,
    min_grid_cells: int,
    grid_rows: int,
    grid_cols: int,
    min_xyz_second_singular_ratio: float,
) -> bool:
    values = list(matches)
    if len(values) < int(min_matches):
        return True
    occupied = _grid_cell_count(
        values,
        np.ones((len(values),), dtype=bool),
        image_width=int(camera.width),
        image_height=int(camera.height),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
    )
    if occupied < int(min_grid_cells):
        return True
    xyz = np.stack([match.xyz for match in values], axis=0).astype(np.float64)
    singular = np.linalg.svd(xyz - np.mean(xyz, axis=0), compute_uv=False)
    if len(singular) < 2 or float(singular[0]) <= 1e-12:
        return True
    return bool(
        float(singular[1] / singular[0])
        < float(min_xyz_second_singular_ratio)
    )


def _sample_candidate_spatial_xy(
    pool: PoseVerificationCandidatePool,
    row: int,
    column: int,
    rng: np.random.Generator,
    *,
    enabled: bool,
) -> tuple[np.ndarray, bool]:
    base_xy = np.asarray(pool.xy[int(row)], dtype=np.float64)
    spatial = pool.spatial_likelihood
    if not bool(enabled) or spatial is None:
        return base_xy, False
    valid_views = np.flatnonzero(spatial.valid_mask[int(row), int(column)])
    if valid_views.size == 0:
        return base_xy, False
    raw_reliability = 1.0 - spatial.dustbin_probabilities[
        int(row), int(column), valid_views
    ]
    reliability = measurement_reliability(
        raw_reliability,
        float(pool.measurement_geometry_probabilities[int(row), int(column)]),
        float(pool.spatial_geometry_calibration_weight),
    )
    view_prior = np.maximum(
        np.asarray(
            spatial.view_probabilities[int(row), int(column), valid_views],
            dtype=np.float64,
        ),
        0.0,
    )
    spatial_probability = float(spatial.log_evidence_weight) * float(
        np.sum(view_prior * reliability)
    )
    if spatial_probability <= 0.0 or float(rng.random()) >= spatial_probability:
        return base_xy, False
    view_mass = (view_prior * reliability).astype(np.float64)
    view_mass = np.maximum(view_mass, 0.0)
    if float(np.sum(view_mass)) <= 1e-12:
        return base_xy, False
    view_probability = view_mass / np.sum(view_mass)
    view = int(rng.choice(valid_views, p=view_probability))
    log_probability = np.array(
        spatial.local_log_probabilities[int(row), int(column), view],
        dtype=np.float64,
        copy=True,
    )
    log_probability -= float(np.max(log_probability))
    mode_probability = np.exp(log_probability)
    mode_probability /= max(float(np.sum(mode_probability)), 1e-12)
    mode = int(rng.choice(len(mode_probability), p=mode_probability))
    return base_xy + spatial.offsets_xy[mode], True


def _prosac_group_order_and_quality(
    pool: PoseVerificationCandidatePool,
    *,
    candidate_limit: int,
    temperature: float,
    group_probability_power: float,
    candidate_probability_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PROSAC group order from factorized non-null and identity mass.

    ``descriptor_scores`` and ``null_scores`` form a joint simplex.  Recover
    the conditional identity distribution before applying the candidate
    exponent so the non-null probability is counted exactly once:
    ``q_i^alpha * sum_j r_ij^beta``.  Truncating to a candidate limit keeps the
    omitted conditional mass omitted; it is never renormalized against null.
    """

    inverse_temperature = 1.0 / float(temperature)
    quality = np.zeros((pool.query_count,), dtype=np.float64)
    for row in range(pool.query_count):
        columns = np.flatnonzero(pool.valid_mask[row])
        if columns.size == 0:
            continue
        order = np.argsort(
            -pool.descriptor_scores[row, columns], kind="mergesort"
        )
        columns = columns[order[: min(int(candidate_limit), len(order))]]
        candidate_joint_mass = np.maximum(
            np.asarray(pool.descriptor_scores[row, columns], dtype=np.float64),
            0.0,
        )
        positive = candidate_joint_mass > 0.0
        if not np.any(positive):
            continue
        nonnull_probability = max(1.0 - float(pool.null_scores[row]), 0.0)
        if nonnull_probability <= 1e-12:
            continue
        conditional_mass = candidate_joint_mass / nonnull_probability
        pair_mass = np.zeros_like(conditional_mass)
        pair_mass[positive] = np.power(
            conditional_mass[positive],
            float(candidate_probability_power) * inverse_temperature,
        )
        quality[row] = float(
            nonnull_probability
            ** (float(group_probability_power) * inverse_temperature)
            * np.sum(pair_mass)
        )
    rows = np.flatnonzero(quality > 0.0)
    order = sorted(
        rows.tolist(),
        key=lambda row: (
            -float(quality[int(row)]),
            int(pool.token_indices[int(row)]),
        ),
    )
    return np.asarray(order, dtype=np.int64), quality


def _prosac_prefix_size(
    group_count: int,
    sample_size: int,
    iteration: int,
    iteration_count: int,
) -> int:
    if int(group_count) < int(sample_size):
        return int(group_count)
    progress = float(int(iteration) + 1) / max(float(iteration_count), 1.0)
    # Quadratic growth keeps early samples concentrated on the highest posterior
    # groups while still exposing the full pool by the final iteration.
    grown = int(
        np.floor(
            (int(group_count) - int(sample_size))
            * min(max(progress, 0.0), 1.0) ** 2
        )
    )
    return min(int(group_count), int(sample_size) + grown)


def sample_grouped_minimal_set(
    pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    candidate_limit: int,
    sample_size: int,
    iteration: int,
    iteration_count: int,
    temperature: float,
    group_probability_power: float,
    candidate_probability_power: float,
    min_grid_cells: int,
    grid_rows: int,
    grid_cols: int,
    min_xyz_second_singular_ratio: float,
    min_bearing_span_deg: float,
    max_attempts: int,
    use_spatial_modes: bool,
    seed: int,
    generation_profile: str = "legacy",
    candidate_uniform_mix: float = 0.0,
) -> GroupedMinimalSet | None:
    """Sample one geometrically valid minimal set directly from query groups.

    Query groups and physical tracks are unique by construction. Explicit null
    mass reduces a group's sampling probability instead of being redistributed
    over its retained candidates.
    """

    if not pool.has_explicit_null:
        raise ValueError("group-aware minimal sampling requires explicit null mass")
    if int(candidate_limit) <= 0:
        raise ValueError("candidate_limit must be positive")
    if int(sample_size) < 4:
        raise ValueError("sample_size must be at least four")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= float(candidate_uniform_mix) <= 1.0:
        raise ValueError("candidate_uniform_mix must be in [0, 1]")
    ordered_rows, quality = _prosac_group_order_and_quality(
        pool,
        candidate_limit=int(candidate_limit),
        temperature=float(temperature),
        group_probability_power=float(group_probability_power),
        candidate_probability_power=float(candidate_probability_power),
    )
    if len(ordered_rows) < int(sample_size):
        return None
    prefix_size = _prosac_prefix_size(
        len(ordered_rows), int(sample_size), int(iteration), int(iteration_count)
    )
    prefix_rows = ordered_rows[:prefix_size]
    rng = np.random.default_rng(int(seed))
    # Spatial-mode availability must not perturb the sampled token/track identities.
    spatial_seed = int.from_bytes(
        hashlib.sha256(
            f"{int(seed)}:candidate_spatial_mode".encode("ascii")
        ).digest()[:8],
        "little",
    )
    spatial_rng = np.random.default_rng(spatial_seed)
    width = max(float(camera.width), 1.0)
    height = max(float(camera.height), 1.0)
    image_diagonal = max(float(np.hypot(width, height)), 1.0)

    for _attempt in range(int(max_attempts)):
        available = prefix_rows.tolist()
        selected_rows: list[int] = []
        selected_columns: list[int] = []
        selected_matches: list[QueryTo3DMatch] = []
        selected_cells: set[tuple[int, int]] = set()
        used_tracks: set[int] = set()
        log_probability = 0.0
        spatial_mode_count = 0
        while available and len(selected_matches) < int(sample_size):
            row_weights = np.asarray(
                [max(float(quality[int(row)]), 0.0) for row in available],
                dtype=np.float64,
            )
            if selected_matches:
                selected_xy = np.stack(
                    [match.xy for match in selected_matches], axis=0
                ).astype(np.float64)
                diversity = []
                for row in available:
                    xy = np.asarray(pool.xy[int(row)], dtype=np.float64)
                    min_distance = float(
                        np.min(np.linalg.norm(selected_xy - xy[None], axis=1))
                    )
                    cell = (
                        int(
                            np.clip(
                                np.floor(xy[1] / height * int(grid_rows)),
                                0,
                                int(grid_rows) - 1,
                            )
                        ),
                        int(
                            np.clip(
                                np.floor(xy[0] / width * int(grid_cols)),
                                0,
                                int(grid_cols) - 1,
                            )
                        ),
                    )
                    distance_gain = min(min_distance / (0.2 * image_diagonal), 1.0)
                    diversity.append(
                        (0.25 + 0.75 * distance_gain)
                        * (1.5 if cell not in selected_cells else 1.0)
                    )
                row_weights *= np.asarray(diversity, dtype=np.float64)
            if float(np.sum(row_weights)) <= 1e-12:
                break
            row_probabilities = row_weights / np.sum(row_weights)
            available_index = int(
                rng.choice(len(available), p=row_probabilities)
            )
            row = int(available.pop(available_index))
            log_probability += float(
                np.log(max(row_probabilities[available_index], 1e-12))
            )

            columns = np.flatnonzero(pool.valid_mask[row])
            order = np.argsort(
                -pool.descriptor_scores[row, columns], kind="mergesort"
            )
            columns = columns[order[: min(int(candidate_limit), len(order))]]
            candidate_joint_mass = np.maximum(
                np.asarray(pool.descriptor_scores[row, columns], dtype=np.float64),
                0.0,
            )
            nonnull_probability = max(1.0 - float(pool.null_scores[row]), 0.0)
            if nonnull_probability <= 1e-12:
                continue
            candidate_mass = candidate_joint_mass / nonnull_probability
            positive = candidate_mass > 0.0
            candidate_mass[positive] = np.power(
                candidate_mass[positive],
                float(candidate_probability_power) / float(temperature),
            )
            candidate_mass[~positive] = 0.0
            for local_index, column in enumerate(columns.tolist()):
                if int(pool.track_ids[row, int(column)]) in used_tracks:
                    candidate_mass[local_index] = 0.0
            if float(np.sum(candidate_mass)) <= 1e-12:
                continue
            candidate_probabilities = candidate_mass / np.sum(candidate_mass)
            uniform_mix = float(candidate_uniform_mix)
            if uniform_mix > 0.0:
                uniform = (candidate_mass > 0.0).astype(np.float64)
                uniform /= np.sum(uniform)
                candidate_probabilities = (
                    (1.0 - uniform_mix) * candidate_probabilities
                    + uniform_mix * uniform
                )
            local_column = int(
                rng.choice(len(columns), p=candidate_probabilities)
            )
            column = int(columns[local_column])
            log_probability += float(
                np.log(max(candidate_probabilities[local_column], 1e-12))
            )
            xy, used_spatial_mode = _sample_candidate_spatial_xy(
                pool,
                row,
                column,
                spatial_rng,
                enabled=bool(use_spatial_modes),
            )
            track_id = int(pool.track_ids[row, column])
            match = QueryTo3DMatch(
                token_index=int(pool.token_indices[row]),
                xy=xy,
                track_id=track_id,
                xyz=np.asarray(pool.xyz[row, column], dtype=np.float64),
                similarity=float(pool.descriptor_scores[row, column]),
                ratio=0.0,
                landmark_variance=0.0,
                source="grouped_prosac_minimal_set",
                prototype_id=int(pool.prototype_ids[row, column]),
            )
            selected_rows.append(row)
            selected_columns.append(column)
            selected_matches.append(match)
            used_tracks.add(track_id)
            selected_cells.add(
                (
                    int(
                        np.clip(
                            np.floor(xy[1] / height * int(grid_rows)),
                            0,
                            int(grid_rows) - 1,
                        )
                    ),
                    int(
                        np.clip(
                            np.floor(xy[0] / width * int(grid_cols)),
                            0,
                            int(grid_cols) - 1,
                        )
                    ),
                )
            )
            spatial_mode_count += int(used_spatial_mode)

        if len(selected_matches) != int(sample_size):
            continue
        if _grouped_fit_is_degenerate(
            selected_matches,
            camera,
            min_matches=int(sample_size),
            min_grid_cells=min(int(min_grid_cells), int(sample_size)),
            grid_rows=int(grid_rows),
            grid_cols=int(grid_cols),
            min_xyz_second_singular_ratio=float(min_xyz_second_singular_ratio),
        ):
            continue
        bearings = _camera_bearings(selected_matches, camera)
        bearing_span_deg = 0.0
        if len(bearings) >= 2:
            bearing_span_deg = float(
                np.degrees(np.max(_pairwise_bearing_angles(bearings)))
            )
        if bearing_span_deg < float(min_bearing_span_deg):
            continue
        return GroupedMinimalSet(
            matches=tuple(selected_matches),
            row_indices=tuple(selected_rows),
            candidate_columns=tuple(selected_columns),
            prosac_prefix_size=int(prefix_size),
            sampling_log_probability=float(log_probability),
            spatial_mode_count=int(spatial_mode_count),
            generation_profile=str(generation_profile),
        )
    return None


def _grouped_minimal_set_signature(
    sample: GroupedMinimalSet,
) -> tuple[tuple[int, int, float, float], ...]:
    """Identify both correspondence identity and sampled spatial mode."""

    return tuple(
        sorted(
            (
                int(match.token_index),
                int(match.track_id),
                round(float(match.xy[0]), 8),
                round(float(match.xy[1]), 8),
            )
            for match in sample.matches
        )
    )


def _resolved_grouped_prosac_profiles(
    config: GroupedCandidatePnPConfig,
) -> tuple[GroupedProsacProfile, ...]:
    if config.prosac_profiles:
        return tuple(config.prosac_profiles)
    sample_sizes = (
        tuple(int(value) for value in config.prosac_minimal_set_sizes)
        if config.prosac_minimal_set_sizes
        else (int(config.prosac_minimal_set_size),)
    )
    return (
        GroupedProsacProfile(
            name="legacy",
            hypotheses_per_limit=int(config.prosac_hypotheses_per_limit),
            minimal_set_sizes=sample_sizes,
            candidate_probability_power=float(
                config.prosac_candidate_probability_power
            ),
            candidate_uniform_mix=0.0,
            local_optimization=bool(config.prosac_local_optimization),
            use_spatial_modes=bool(config.prosac_use_spatial_modes),
        ),
    )


def generate_grouped_prosac_hypotheses(
    fit_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    config: GroupedCandidatePnPConfig,
    query_seed: int,
) -> tuple[GroupedGeneratedPose, ...]:
    """Generate hypotheses with group-aware posterior-guided minimal samples."""

    generated: list[GroupedGeneratedPose] = []
    seen_raw_samples: set[tuple[tuple[int, int, float, float], ...]] = set()
    seen_local_samples: set[tuple[tuple[int, int, float, float], ...]] = set()
    profiles: list[GroupedProsacProfile] = []
    for profile in _resolved_grouped_prosac_profiles(config):
        if bool(profile.use_spatial_modes) and fit_pool.spatial_likelihood is not None:
            profiles.append(
                replace(
                    profile,
                    name=f"{profile.name}__base_coordinate",
                    use_spatial_modes=False,
                )
            )
        profiles.append(profile)

    for profile in profiles:
        sample_sizes = tuple(int(value) for value in profile.minimal_set_sizes)
        for candidate_limit in config.candidate_limits:
            effective_limit = min(int(candidate_limit), fit_pool.track_ids.shape[1])
            for temperature_index, temperature in enumerate(
                config.sampling_temperatures
            ):
                for iteration in range(int(profile.hypotheses_per_limit)):
                    size_index = int(iteration % len(sample_sizes))
                    sample_size = int(sample_sizes[size_index])
                    size_iteration = int(iteration // len(sample_sizes))
                    size_iteration_count = int(
                        np.ceil(
                            max(
                                int(profile.hypotheses_per_limit) - size_index,
                                0,
                            )
                            / len(sample_sizes)
                        )
                    )
                    seed_payload = (
                        f"{int(query_seed)}:{effective_limit}:"
                        f"{float(temperature):.8g}:{temperature_index}:"
                        f"{sample_size}:{size_iteration}:grouped_prosac"
                    ).encode("ascii")
                    sample_seed = int.from_bytes(
                        hashlib.sha256(seed_payload).digest()[:8], "little"
                    )
                    sample = sample_grouped_minimal_set(
                        fit_pool,
                        camera,
                        candidate_limit=effective_limit,
                        sample_size=sample_size,
                        iteration=size_iteration,
                        iteration_count=max(size_iteration_count, 1),
                        temperature=float(temperature),
                        group_probability_power=float(
                            config.prosac_group_probability_power
                        ),
                        candidate_probability_power=float(
                            profile.candidate_probability_power
                        ),
                        min_grid_cells=int(config.min_fit_grid_cells),
                        grid_rows=int(config.grid_rows),
                        grid_cols=int(config.grid_cols),
                        min_xyz_second_singular_ratio=float(
                            config.min_xyz_second_singular_ratio
                        ),
                        min_bearing_span_deg=float(
                            config.prosac_min_bearing_span_deg
                        ),
                        max_attempts=int(config.prosac_max_sample_attempts),
                        use_spatial_modes=bool(profile.use_spatial_modes),
                        seed=sample_seed,
                        generation_profile=str(profile.name),
                        candidate_uniform_mix=float(
                            profile.candidate_uniform_mix
                        ),
                    )
                    if sample is None:
                        continue
                    signature = _grouped_minimal_set_signature(sample)
                    needs_raw = signature not in seen_raw_samples
                    needs_local = bool(profile.local_optimization) and (
                        signature not in seen_local_samples
                    )
                    if not needs_raw and not needs_local:
                        continue
                    if len(sample.matches) == 4:
                        sample_results = estimate_pose_pnp_fixed_hypotheses(
                            sample.matches,
                            camera,
                            min_inliers=len(sample.matches),
                            pnp_method="AP3P",
                        )
                    else:
                        sample_results = (
                            estimate_pose_pnp_fixed(
                                sample.matches,
                                camera,
                                min_inliers=len(sample.matches),
                                pnp_method="EPNP",
                                refine_method="none",
                            ),
                        )
                    for result in sample_results:
                        pose = result.pose_w2c
                        solver_success = bool(result.success and pose is not None)
                        pose_array = (
                            None
                            if pose is None
                            else np.asarray(pose, dtype=np.float64).reshape(4, 4)
                        )
                        observability_failures = (
                            ()
                            if not solver_success
                            else _generated_pose_observability_failures(
                                pose_array,
                                sample.matches,
                                fit_pool,
                                camera,
                                config,
                            )
                        )
                        accepted = bool(
                            solver_success and not observability_failures
                        )
                        if needs_raw:
                            generated.append(
                                GroupedGeneratedPose(
                                    pose_w2c=pose_array if accepted else None,
                                    sample=sample,
                                    solver_success=accepted,
                                    observability_gate_failures=(
                                        observability_failures
                                    ),
                                )
                            )
                        if not needs_local or not accepted or pose_array is None:
                            continue
                        resolved, selected_columns, selected_residuals = (
                            resolve_pose_guided_candidate_pool(
                                fit_pool,
                                pose_array,
                                camera,
                                residual_sigma_px=float(
                                    config.candidate_pool_residual_sigma_px
                                ),
                                hard_threshold_px=float(
                                    config.prosac_local_consensus_px
                                ),
                                descriptor_rank_weight=float(
                                    config.candidate_pool_descriptor_rank_weight
                                ),
                            )
                        )
                        accepted_residuals = selected_residuals[
                            selected_columns >= 0
                        ]
                        consensus_mask = accepted_residuals <= float(
                            config.prosac_local_consensus_px
                        )
                        consensus = [
                            match
                            for match, accepted_match in zip(
                                resolved, consensus_mask
                            )
                            if bool(accepted_match)
                        ]
                        local_match_count = int(len(resolved))
                        local_inlier_count = int(len(consensus))
                        if len(consensus) < int(config.prosac_local_min_matches):
                            continue
                        weights = np.exp(
                            -0.5
                            * np.square(
                                accepted_residuals[consensus_mask]
                                / float(config.prosac_local_consensus_px)
                            )
                        )
                        refined = estimate_pose_pnp_fixed_robust(
                            consensus,
                            camera,
                            weights=weights,
                            min_inliers=int(config.prosac_local_min_matches),
                            initial_pose_w2c=pose_array,
                            loss="huber",
                            f_scale_px=float(config.prosac_local_consensus_px),
                            max_nfev=50,
                        )
                        if refined.success and refined.pose_w2c is not None:
                            refined_pose = np.asarray(
                                refined.pose_w2c, dtype=np.float64
                            ).reshape(4, 4)
                            refined_observability_failures = (
                                _generated_pose_observability_failures(
                                    refined_pose,
                                    sample.matches,
                                    fit_pool,
                                    camera,
                                    config,
                                )
                            )
                            refined_accepted = not refined_observability_failures
                            generated.append(
                                GroupedGeneratedPose(
                                    pose_w2c=(
                                        refined_pose
                                        if refined_accepted
                                        else None
                                    ),
                                    sample=sample,
                                    solver_success=bool(refined_accepted),
                                    local_optimization_match_count=(
                                        local_match_count
                                    ),
                                    local_optimization_inlier_count=(
                                        local_inlier_count
                                    ),
                                    local_optimization_applied=True,
                                    observability_gate_failures=(
                                        refined_observability_failures
                                    ),
                                )
                            )
                    if needs_raw:
                        seen_raw_samples.add(signature)
                    if needs_local:
                        seen_local_samples.add(signature)
    return tuple(generated)


def _pose_hypotheses_are_near(
    first_pose: np.ndarray,
    second_pose: np.ndarray,
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> bool:
    first = np.asarray(first_pose, dtype=np.float64).reshape(4, 4)
    second = np.asarray(second_pose, dtype=np.float64).reshape(4, 4)
    first_center = -first[:3, :3].T @ first[:3, 3]
    second_center = -second[:3, :3].T @ second[:3, 3]
    translation = float(np.linalg.norm(first_center - second_center))
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    rotation = float(np.degrees(np.arccos(cosine)))
    return bool(
        translation <= float(translation_threshold_m)
        and rotation <= float(rotation_threshold_deg)
    )


def select_grouped_hypothesis_shortlist(
    generated: Sequence[GroupedGeneratedPose],
    ranked_scores: Sequence[tuple[float, float, int]],
    *,
    top_k: int,
    selection_mode: str = "score_topk",
    diverse_count: int = 0,
    min_per_profile: int = 0,
    translation_diversity_m: float = 0.05,
    rotation_diversity_deg: float = 0.5,
) -> tuple[int, ...]:
    """Select a deterministic shortlist without letting one pose mode crowd it.

    The diverse policy preserves the highest-scoring head verbatim. Its
    remaining budget first covers underrepresented generation profiles with
    pose-space NMS, then adds any other distinct modes, and finally backfills by
    score so the requested budget is not reduced by aggressive NMS.
    """

    limit = int(top_k)
    if limit < 0:
        raise ValueError("grouped hypothesis shortlist top-k must be non-negative")
    if str(selection_mode) not in {"score_topk", "profile_pose_diverse"}:
        raise ValueError("unsupported grouped hypothesis shortlist mode")
    if int(diverse_count) < 0 or int(min_per_profile) < 0:
        raise ValueError("grouped hypothesis shortlist counts must be non-negative")
    if (
        float(translation_diversity_m) < 0.0
        or float(rotation_diversity_deg) < 0.0
    ):
        raise ValueError("grouped hypothesis diversity thresholds must be non-negative")

    ranked: list[tuple[float, float, int]] = []
    seen_indices: set[int] = set()
    for score, sample_score, index in ranked_scores:
        candidate_index = int(index)
        if candidate_index in seen_indices:
            continue
        if not 0 <= candidate_index < len(generated):
            raise ValueError("grouped hypothesis shortlist index is out of range")
        candidate = generated[candidate_index]
        if not candidate.solver_success or candidate.pose_w2c is None:
            continue
        seen_indices.add(candidate_index)
        ranked.append((float(score), float(sample_score), candidate_index))
    if limit == 0 or not ranked:
        return ()
    limit = min(limit, len(ranked))
    if str(selection_mode) == "score_topk":
        return tuple(int(item[2]) for item in ranked[:limit])
    if not 0 < int(diverse_count) <= limit:
        raise ValueError("diverse shortlist count must be in (0, top_k]")

    head_count = max(limit - int(diverse_count), 0)
    selected = [int(item[2]) for item in ranked[:head_count]]
    selected_set = set(selected)

    def is_pose_distinct(candidate_index: int) -> bool:
        candidate_pose = generated[int(candidate_index)].pose_w2c
        if candidate_pose is None:
            return False
        return not any(
            _pose_hypotheses_are_near(
                candidate_pose,
                generated[int(previous_index)].pose_w2c,  # type: ignore[arg-type]
                translation_threshold_m=float(translation_diversity_m),
                rotation_threshold_deg=float(rotation_diversity_deg),
            )
            for previous_index in selected
        )

    profile_order: list[str] = []
    profile_candidates: dict[str, list[int]] = {}
    for _score, _sample_score, candidate_index in ranked:
        profile = str(generated[candidate_index].sample.generation_profile)
        if profile not in profile_candidates:
            profile_order.append(profile)
            profile_candidates[profile] = []
        profile_candidates[profile].append(int(candidate_index))
    profile_counts = {profile: 0 for profile in profile_order}
    for candidate_index in selected:
        profile = str(generated[candidate_index].sample.generation_profile)
        profile_counts[profile] = int(profile_counts.get(profile, 0)) + 1

    while len(selected) < limit and any(
        int(profile_counts[profile]) < int(min_per_profile)
        for profile in profile_order
    ):
        made_progress = False
        for profile in profile_order:
            if len(selected) >= limit:
                break
            if int(profile_counts[profile]) >= int(min_per_profile):
                continue
            candidate_index = next(
                (
                    index
                    for index in profile_candidates[profile]
                    if index not in selected_set and is_pose_distinct(index)
                ),
                None,
            )
            if candidate_index is None:
                profile_counts[profile] = int(min_per_profile)
                continue
            selected.append(int(candidate_index))
            selected_set.add(int(candidate_index))
            profile_counts[profile] += 1
            made_progress = True
        if not made_progress:
            break

    for _score, _sample_score, candidate_index in ranked:
        if len(selected) >= limit:
            break
        if candidate_index in selected_set or not is_pose_distinct(candidate_index):
            continue
        selected.append(int(candidate_index))
        selected_set.add(int(candidate_index))

    for _score, _sample_score, candidate_index in ranked:
        if len(selected) >= limit:
            break
        if candidate_index in selected_set:
            continue
        selected.append(int(candidate_index))
        selected_set.add(int(candidate_index))
    return tuple(selected)


def _score_grouped_hypothesis_indices(
    generated: Sequence[GroupedGeneratedPose],
    candidate_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    config: GroupedCandidatePnPConfig,
    indices: Sequence[int],
    *,
    spatial_evidence_weight: float | None,
) -> list[tuple[float, float, int]]:
    """Score fixed poses on one immutable candidate-pool denominator."""

    scored: list[tuple[float, float, int]] = []
    seen: set[int] = set()
    for raw_index in indices:
        index = int(raw_index)
        if index in seen:
            continue
        if not 0 <= index < len(generated):
            raise ValueError("grouped hypothesis score index is out of range")
        seen.add(index)
        item = generated[index]
        if (
            not item.solver_success
            or item.pose_w2c is None
        ):
            continue
        likelihood = fixed_posterior_pose_log_likelihood(
            candidate_pool,
            item.pose_w2c,
            camera,
            residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
            candidate_outlier_likelihood=float(
                config.candidate_pose_outlier_likelihood
            ),
            null_likelihood=float(config.candidate_pose_null_likelihood),
            spatial_evidence_weight=spatial_evidence_weight,
        )
        scored.append(
            (
                float(likelihood["log_likelihood_mean"]),
                float(item.sample.sampling_log_probability),
                int(index),
            )
        )
    scored.sort(reverse=True)
    return scored


def _select_latent_em_seeds_from_ranked(
    generated: Sequence[GroupedGeneratedPose],
    ranked_scores: Sequence[tuple[float, float, int]],
    config: GroupedCandidatePnPConfig,
) -> tuple[int, ...]:
    scored: list[tuple[float, float, int]] = []
    for score, sample_score, raw_index in ranked_scores:
        index = int(raw_index)
        if not 0 <= index < len(generated):
            raise ValueError("latent EM seed index is out of range")
        if generated[index].latent_em_applied:
            continue
        scored.append((float(score), float(sample_score), index))
    scored.sort(reverse=True)
    limit = min(int(config.latent_em_seed_count), len(scored))
    if limit <= 0:
        return ()
    return select_grouped_hypothesis_shortlist(
        generated,
        scored,
        top_k=int(limit),
        selection_mode="profile_pose_diverse",
        diverse_count=int(limit),
        min_per_profile=int(config.latent_em_seed_min_per_profile),
        translation_diversity_m=float(
            config.latent_em_translation_diversity_m
        ),
        rotation_diversity_deg=float(config.latent_em_rotation_diversity_deg),
    )


def _select_latent_em_seeds_fit_pool_diagnostic_only(
    generated: Sequence[GroupedGeneratedPose],
    fit_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    config: GroupedCandidatePnPConfig,
) -> tuple[int, ...]:
    """Reproduce the old self-consistency seed policy for explicit ablations."""

    shortlist_spatial_weight = (
        0.0
        if str(config.prosac_shortlist_evidence_mode)
        == "base_coordinate_then_full_spatial"
        else None
    )
    scored = _score_grouped_hypothesis_indices(
        generated,
        fit_pool,
        camera,
        config,
        tuple(range(len(generated))),
        spatial_evidence_weight=shortlist_spatial_weight,
    )
    return _select_latent_em_seeds_from_ranked(generated, scored, config)


def _select_configured_grouped_shortlist(
    generated: Sequence[GroupedGeneratedPose],
    ranked_scores: Sequence[tuple[float, float, int]],
    config: GroupedCandidatePnPConfig,
    *,
    top_k: int,
) -> tuple[int, ...]:
    limit = min(int(top_k), len(ranked_scores))
    if limit <= 0:
        return ()
    diverse_count = int(config.prosac_shortlist_diverse_count)
    if str(config.prosac_shortlist_selection_mode) == "profile_pose_diverse":
        diverse_count = min(diverse_count, limit)
    return select_grouped_hypothesis_shortlist(
        generated,
        ranked_scores,
        top_k=limit,
        selection_mode=str(config.prosac_shortlist_selection_mode),
        diverse_count=diverse_count,
        min_per_profile=int(config.prosac_shortlist_min_per_profile),
        translation_diversity_m=float(
            config.prosac_shortlist_translation_diversity_m
        ),
        rotation_diversity_deg=float(
            config.prosac_shortlist_rotation_diversity_deg
        ),
    )


def _apply_selected_candidate_coordinate_updates(
    matches: Sequence[QueryTo3DMatch],
    selected_columns: np.ndarray,
    pool: PoseVerificationCandidatePool,
) -> tuple[list[QueryTo3DMatch], int]:
    """Update only identities already selected by the frozen pose assignment."""

    selected = np.asarray(selected_columns, dtype=np.int64).reshape(-1)
    if selected.shape != (pool.query_count,):
        raise ValueError("selected candidate columns and pool rows differ")
    accepted_rows = np.flatnonzero(selected >= 0)
    if len(matches) != len(accepted_rows):
        raise ValueError("selected candidate matches and columns differ")
    output: list[QueryTo3DMatch] = []
    update_count = 0
    probabilities = np.asarray(
        pool.candidate_update_probabilities, dtype=np.float64
    )
    refined_xy = np.asarray(pool.candidate_refined_xy, dtype=np.float64)
    for match, row in zip(matches, accepted_rows.tolist()):
        column = int(selected[row])
        if (
            int(match.token_index) != int(pool.token_indices[row])
            or int(match.track_id) != int(pool.track_ids[row, column])
            or int(match.prototype_id or 0) != int(pool.prototype_ids[row, column])
        ):
            raise ValueError("selected candidate match identity changed before refinement")
        probability = float(probabilities[row, column])
        coordinate = refined_xy[row, column]
        if (
            np.isfinite(probability)
            and probability >= float(pool.candidate_update_threshold)
            and np.all(np.isfinite(coordinate))
        ):
            output.append(
                replace(
                    match,
                    xy=np.asarray(coordinate, dtype=np.float64),
                    source="pose_guided_topl_candidate_pool:rgb_coordinate_update",
                )
            )
            update_count += 1
        else:
            output.append(match)
    return output, int(update_count)


def _identity_owner_by_role(
    identity_ids: np.ndarray,
    candidate_mask: np.ndarray,
    row_roles: np.ndarray,
    *,
    namespace: str,
    salt: int,
) -> dict[int, int]:
    identities = np.asarray(identity_ids, dtype=np.int64)
    valid = np.asarray(candidate_mask, dtype=bool)
    roles = np.asarray(row_roles, dtype=np.int64).reshape(-1)
    if identities.shape != valid.shape or identities.shape[0] != len(roles):
        raise ValueError("identity ownership arrays are not aligned")
    output: dict[int, int] = {}
    for identity in np.unique(identities[valid]).tolist():
        rows, _columns = np.nonzero(valid & (identities == int(identity)))
        available_roles = sorted(set(int(roles[row]) for row in rows.tolist()))
        if not available_roles or available_roles[0] < 0:
            raise RuntimeError("candidate identity has no cross-fit role")
        payload = f"{namespace}:{int(identity)}:{int(salt)}".encode("ascii")
        owner_index = int.from_bytes(
            hashlib.sha256(payload).digest()[:8], "little"
        ) % len(available_roles)
        output[int(identity)] = int(available_roles[owner_index])
    return output


def _mask_to_identity_owners(
    identity_ids: np.ndarray,
    candidate_mask: np.ndarray,
    row_roles: np.ndarray,
    owners: dict[int, int],
) -> np.ndarray:
    identities = np.asarray(identity_ids, dtype=np.int64)
    keep = np.asarray(candidate_mask, dtype=bool).copy()
    roles = np.asarray(row_roles, dtype=np.int64).reshape(-1)
    rows, columns = np.nonzero(keep)
    for row, column in zip(rows.tolist(), columns.tolist()):
        identity = int(identities[row, column])
        if int(owners[identity]) != int(roles[row]):
            keep[row, column] = False
    return keep


def _retained_identity_set(
    pool: PoseVerificationCandidatePool,
    identity_ids: np.ndarray,
) -> set[int]:
    identities = np.asarray(identity_ids, dtype=np.int64)
    if identities.shape != pool.valid_mask.shape:
        raise ValueError("retained identity arrays are not aligned")
    return set(int(value) for value in identities[pool.valid_mask].tolist())


def _independent_shortlist_spatial_partitions(
    matches: Sequence[QueryTo3DMatch],
    *,
    camera: ColmapCamera,
    config: GroupedCandidatePnPConfig,
    query_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split rank folds into shortlist and selection without sharing rows."""

    rank_folds = tuple(
        range(
            int(config.verification_fold),
            int(config.verification_fold) + int(config.verification_fold_count),
        )
    )
    if len(rank_folds) < 2:
        raise ValueError("independent shortlist partition requires two rank folds")

    heldout_by_fold: dict[int, np.ndarray] = {}
    for fold in rank_folds + (int(config.final_audit_fold),):
        _fit, heldout = deterministic_spatial_holdout(
            matches,
            image_width=int(camera.width),
            image_height=int(camera.height),
            folds=int(config.holdout_folds),
            fold=int(fold),
            grid_rows=int(config.grid_rows),
            grid_cols=int(config.grid_cols),
            salt=int(query_seed),
            fold_policy=str(config.crossfit_spatial_fold_policy),
        )
        heldout_by_fold[int(fold)] = heldout
    shortlist = heldout_by_fold[int(rank_folds[0])]
    verification = np.asarray(
        sorted(
            set(
                np.concatenate(
                    [heldout_by_fold[int(fold)] for fold in rank_folds[1:]]
                ).astype(np.int64).tolist()
            )
        ),
        dtype=np.int64,
    )
    audit = heldout_by_fold[int(config.final_audit_fold)]
    if (
        np.intersect1d(shortlist, verification).size
        or np.intersect1d(shortlist, audit).size
        or np.intersect1d(verification, audit).size
    ):
        raise RuntimeError("independent shortlist spatial folds overlap")
    excluded = np.zeros((len(matches),), dtype=bool)
    excluded[shortlist] = True
    excluded[verification] = True
    excluded[audit] = True
    fit = np.flatnonzero(~excluded).astype(np.int64)
    return fit, shortlist, verification, audit


def _partition_spatial_pool_with_identity_purging(
    candidate_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    config: GroupedCandidatePnPConfig,
    query_seed: int,
    purge_maplets: bool,
) -> GroupedCandidateCrossfitPools:
    """Keep balanced token folds while making candidate identities disjoint."""

    if bool(purge_maplets) and not candidate_pool.has_explicit_maplet_clusters:
        raise ValueError("maplet-purged cross-fit requires explicit cluster ids")
    representatives = _candidate_pool_representatives(candidate_pool)
    if bool(config.independent_shortlist_pool):
        (
            fit_indices,
            shortlist_indices,
            verification_indices,
            audit_indices,
        ) = _independent_shortlist_spatial_partitions(
            representatives,
            camera=camera,
            config=config,
            query_seed=int(query_seed),
        )
        role_indices = (
            fit_indices,
            shortlist_indices,
            verification_indices,
            audit_indices,
        )
        role_names = ("fit", "shortlist", "verification", "audit")
    else:
        fit_indices, verification_indices, audit_indices = (
            deterministic_spatial_partitions(
                representatives,
                image_width=int(camera.width),
                image_height=int(camera.height),
                folds=int(config.holdout_folds),
                verification_fold=int(config.verification_fold),
                verification_fold_count=int(config.verification_fold_count),
                final_audit_fold=int(config.final_audit_fold),
                grid_rows=int(config.grid_rows),
                grid_cols=int(config.grid_cols),
                salt=int(query_seed),
                fold_policy=str(config.crossfit_spatial_fold_policy),
            )
        )
        shortlist_indices = verification_indices
        role_indices = (fit_indices, verification_indices, audit_indices)
        role_names = ("fit", "verification", "audit")
    row_roles = np.full((candidate_pool.query_count,), -1, dtype=np.int64)
    for role, indices in enumerate(role_indices):
        row_roles[np.asarray(indices, dtype=np.int64)] = int(role)
    if np.any(row_roles < 0):
        raise RuntimeError("spatial cross-fit left query groups unassigned")

    keep = np.asarray(candidate_pool.valid_mask, dtype=bool).copy()
    maplet_owner_count = 0
    if bool(purge_maplets):
        maplet_owners = _identity_owner_by_role(
            candidate_pool.maplet_cluster_ids,
            keep,
            row_roles,
            namespace="maplet",
            salt=int(query_seed),
        )
        maplet_owner_count = int(len(maplet_owners))
        keep = _mask_to_identity_owners(
            candidate_pool.maplet_cluster_ids,
            keep,
            row_roles,
            maplet_owners,
        )
    track_owners = _identity_owner_by_role(
        candidate_pool.track_ids,
        keep,
        row_roles,
        namespace="track_after_maplet" if purge_maplets else "track",
        salt=int(query_seed),
    )
    keep = _mask_to_identity_owners(
        candidate_pool.track_ids,
        keep,
        row_roles,
        track_owners,
    )
    masked_pool = candidate_pool.mask_candidates_to_null(keep)

    role_pools: list[PoseVerificationCandidatePool] = []
    role_tokens: list[tuple[int, ...]] = []
    retention: dict[str, object] = {}
    for role_name, indices in zip(role_names, role_indices):
        tokens = tuple(
            int(candidate_pool.token_indices[int(index)])
            for index in np.asarray(indices, dtype=np.int64).tolist()
        )
        role_tokens.append(tokens)
        role_pool = masked_pool.subset_by_token_indices(tokens)
        role_pools.append(role_pool)
        original_valid = int(
            np.sum(candidate_pool.valid_mask[np.asarray(indices, dtype=np.int64)])
        )
        retained_valid = int(np.sum(role_pool.valid_mask))
        retention[role_name] = {
            "query_group_count": int(role_pool.query_count),
            "original_candidate_count": original_valid,
            "retained_candidate_count": retained_valid,
            "retained_candidate_fraction": float(
                retained_valid / max(original_valid, 1)
            ),
            "non_null_query_group_count": int(
                np.sum(np.any(role_pool.valid_mask, axis=1))
            ),
            "mean_null_probability": (
                None
                if role_pool.query_count == 0
                else float(np.mean(role_pool.null_scores))
            ),
        }

    track_sets = [
        _retained_identity_set(pool, pool.track_ids) for pool in role_pools
    ]
    role_pairs = tuple(
        (first, second)
        for first in range(len(role_pools))
        for second in range(first + 1, len(role_pools))
    )
    track_overlaps = [
        int(len(track_sets[first] & track_sets[second]))
        for first, second in role_pairs
    ]
    if any(track_overlaps):
        raise RuntimeError("physical track crosses a purged cross-fit partition")
    maplet_overlaps: list[int] | None = None
    if candidate_pool.has_explicit_maplet_clusters:
        maplet_sets = [
            _retained_identity_set(pool, pool.maplet_cluster_ids)
            for pool in role_pools
        ]
        maplet_overlaps = [
            int(len(maplet_sets[first] & maplet_sets[second]))
            for first, second in role_pairs
        ]
        if bool(purge_maplets) and any(maplet_overlaps):
            raise RuntimeError("maplet crosses a purged cross-fit partition")

    original_candidate_count = int(np.sum(candidate_pool.valid_mask))
    retained_candidate_count = int(np.sum(masked_pool.valid_mask))
    partition_audit: dict[str, object] = {
        "partition_type": (
            "token_spatial_track_maplet_purged"
            if purge_maplets
            else "token_spatial_track_purged"
        ),
        "role_assignment": "deterministic_identity_owner_over_spatial_folds",
        "spatial_fold_policy": str(config.crossfit_spatial_fold_policy),
        "query_group_count": int(candidate_pool.query_count),
        "fit_count": int(len(fit_indices)),
        "shortlist_count": int(len(shortlist_indices)),
        "verification_count": int(len(verification_indices)),
        "audit_count": int(len(audit_indices)),
        "minimum_heldout_count": int(
            min(
                len(shortlist_indices),
                len(verification_indices),
                len(audit_indices),
            )
        ),
        "independent_shortlist_pool": bool(config.independent_shortlist_pool),
        "strict_track_disjoint": True,
        "strict_maplet_disjoint": bool(purge_maplets),
        "track_owner_count": int(len(track_owners)),
        "maplet_owner_count": int(maplet_owner_count),
        "track_overlap_counts_fit_verify_fit_audit_verify_audit": track_overlaps,
        "identity_overlap_role_pairs": [
            [str(role_names[first]), str(role_names[second])]
            for first, second in role_pairs
        ],
        "maplet_overlap_counts_fit_verify_fit_audit_verify_audit": (
            maplet_overlaps
        ),
        "original_candidate_count": original_candidate_count,
        "retained_candidate_count": retained_candidate_count,
        "retained_candidate_fraction": float(
            retained_candidate_count / max(original_candidate_count, 1)
        ),
        "removed_probability_mass_mean": float(
            np.mean(masked_pool.null_scores - candidate_pool.null_scores)
        ),
        "retention_by_role": retention,
    }
    pools_by_role = dict(zip(role_names, role_pools))
    tokens_by_role = dict(zip(role_names, role_tokens))
    return GroupedCandidateCrossfitPools(
        fit=pools_by_role["fit"],
        shortlist=(
            pools_by_role["shortlist"]
            if bool(config.independent_shortlist_pool)
            else pools_by_role["verification"]
        ),
        verification=pools_by_role["verification"],
        audit=pools_by_role["audit"],
        fit_tokens=tokens_by_role["fit"],
        shortlist_tokens=(
            tokens_by_role["shortlist"]
            if bool(config.independent_shortlist_pool)
            else tokens_by_role["verification"]
        ),
        verification_tokens=tokens_by_role["verification"],
        audit_tokens=tokens_by_role["audit"],
        partition_audit=partition_audit,
    )


def partition_grouped_candidate_pool(
    candidate_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    config: GroupedCandidatePnPConfig,
    query_seed: int,
) -> GroupedCandidateCrossfitPools:
    """Build the one fit/verification/audit split used by all pose variants."""

    representatives = _candidate_pool_representatives(candidate_pool)
    if str(config.crossfit_mode) in {
        "token_spatial_track_purged",
        "token_spatial_track_maplet_purged",
    }:
        return _partition_spatial_pool_with_identity_purging(
            candidate_pool,
            camera,
            config=config,
            query_seed=int(query_seed),
            purge_maplets=(
                str(config.crossfit_mode)
                == "token_spatial_track_maplet_purged"
            ),
        )
    if str(config.crossfit_mode) == "token_spatial":
        fit_indices, verification_indices, audit_indices = (
            deterministic_spatial_partitions(
                representatives,
                image_width=int(camera.width),
                image_height=int(camera.height),
                folds=int(config.holdout_folds),
                verification_fold=int(config.verification_fold),
                verification_fold_count=int(config.verification_fold_count),
                final_audit_fold=int(config.final_audit_fold),
                grid_rows=int(config.grid_rows),
                grid_cols=int(config.grid_cols),
                salt=int(query_seed),
                fold_policy=str(config.crossfit_spatial_fold_policy),
            )
        )
        partition_audit: dict[str, object] = {
            "partition_type": "token_spatial",
            "role_assignment": "fixed",
            "spatial_fold_policy": str(config.crossfit_spatial_fold_policy),
            "query_group_count": int(candidate_pool.query_count),
            "fit_count": int(fit_indices.size),
            "verification_count": int(verification_indices.size),
            "audit_count": int(audit_indices.size),
            "minimum_heldout_count": int(
                min(verification_indices.size, audit_indices.size)
            ),
            "strict_track_disjoint": False,
            "strict_maplet_disjoint": False,
        }
    else:
        component_plan = _deterministic_component_partition_plan(
            candidate_pool,
            image_width=int(camera.width),
            image_height=int(camera.height),
            folds=int(config.holdout_folds),
            verification_fold=int(config.verification_fold),
            verification_fold_count=int(config.verification_fold_count),
            final_audit_fold=int(config.final_audit_fold),
            grid_rows=int(config.grid_rows),
            grid_cols=int(config.grid_cols),
            maplet_voxel_size_m=(
                float(config.crossfit_maplet_voxel_size_m)
                if str(config.crossfit_mode) == "token_track_voxel_component"
                else None
            ),
            use_explicit_maplet_clusters=(
                str(config.crossfit_mode) == "token_track_maplet_component"
            ),
            role_assignment=str(config.crossfit_role_assignment),
            minimum_fit_count=int(config.min_fit_matches),
            minimum_heldout_count=4,
            salt=int(query_seed),
        )
        fit_indices = component_plan.fit_indices
        verification_indices = component_plan.verification_indices
        audit_indices = component_plan.audit_indices
        partition_audit = {
            **component_plan.audit,
            "partition_type": str(config.crossfit_mode),
        }
    fit_tokens = tuple(
        int(representatives[int(index)].token_index) for index in fit_indices
    )
    verification_tokens = tuple(
        int(representatives[int(index)].token_index)
        for index in verification_indices
    )
    audit_tokens = tuple(
        int(representatives[int(index)].token_index) for index in audit_indices
    )
    return GroupedCandidateCrossfitPools(
        fit=candidate_pool.subset_by_token_indices(fit_tokens),
        shortlist=candidate_pool.subset_by_token_indices(verification_tokens),
        verification=candidate_pool.subset_by_token_indices(verification_tokens),
        audit=candidate_pool.subset_by_token_indices(audit_tokens),
        fit_tokens=fit_tokens,
        shortlist_tokens=verification_tokens,
        verification_tokens=verification_tokens,
        audit_tokens=audit_tokens,
        partition_audit=partition_audit,
    )


def _grouped_relation_verification_kwargs(
    config: GroupedCandidatePnPConfig,
    pool: PoseVerificationCandidatePool,
) -> dict[str, object]:
    neighbor_k = int(config.candidate_pose_relation_neighbor_k)
    if neighbor_k <= 0:
        return {}
    return {
        "relation_neighbor_k": neighbor_k,
        "relation_sigma_px": float(config.candidate_pose_relation_sigma_px),
        "relation_outlier_likelihood": float(
            config.candidate_pose_relation_outlier_likelihood
        ),
        "relation_neighbor_edges": candidate_relation_neighbor_edges(
            pool, neighbor_k=neighbor_k
        ),
    }


def _validate_generation_candidate_pool_compatibility(
    scoring_pool: PoseVerificationCandidatePool,
    generation_pool: PoseVerificationCandidatePool,
) -> None:
    """Require frozen generation to differ only in its spatial likelihood."""

    array_fields = (
        "token_indices",
        "xy",
        "track_ids",
        "prototype_ids",
        "xyz",
        "descriptor_scores",
        "valid_mask",
        "measurement_geometry_probabilities",
        "null_scores",
        "candidate_update_probabilities",
        "candidate_refined_xy",
        "maplet_cluster_ids",
    )
    for name in array_fields:
        scoring = np.asarray(getattr(scoring_pool, name))
        generation = np.asarray(getattr(generation_pool, name))
        if np.issubdtype(scoring.dtype, np.floating) or np.issubdtype(
            generation.dtype, np.floating
        ):
            equal = np.array_equal(scoring, generation, equal_nan=True)
        else:
            equal = np.array_equal(scoring, generation)
        if not equal:
            raise ValueError(
                f"generation and scoring candidate pools differ in {name}"
            )
    scalar_fields = (
        "measurement_verification_threshold",
        "geometry_prior_mix_weight",
        "spatial_geometry_calibration_weight",
        "geometry_generation_mix_weight",
        "candidate_update_threshold",
        "spatial_utility_gate_weight",
        "has_explicit_null",
        "has_explicit_maplet_clusters",
    )
    for name in scalar_fields:
        if getattr(scoring_pool, name) != getattr(generation_pool, name):
            raise ValueError(
                f"generation and scoring candidate pools differ in {name}"
            )


def estimate_pose_from_grouped_candidate_pool(
    candidate_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    config: GroupedCandidatePnPConfig = GroupedCandidatePnPConfig(),
    query_seed: int = 0,
    hypothesis_selector: HypothesisSelector | None = None,
    generation_candidate_pool: PoseVerificationCandidatePool | None = None,
) -> VerifiedPnPResult:
    """Generate PnP hypotheses while retaining top-L candidate uncertainty.

    Each fit hypothesis contains at most one 3D candidate per query token and
    at most one query token per physical track. Candidate groups used for
    hypothesis ranking and final audit are disjoint from fit groups. When the
    independent shortlist pool is enabled, shortlist/EM-seed evidence is also
    disjoint from final hypothesis-selection evidence.
    """

    if not candidate_pool.has_explicit_null:
        raise ValueError("grouped candidate PnP requires an explicit null posterior")
    generation_pool = (
        candidate_pool
        if generation_candidate_pool is None
        else generation_candidate_pool
    )
    _validate_generation_candidate_pool_compatibility(
        candidate_pool, generation_pool
    )
    representatives = _candidate_pool_representatives(candidate_pool)
    if len(representatives) < max(12, int(config.min_final_inliers)):
        return _empty_result(
            candidate_pool.query_count, fit_count=0, verification_count=0
        )
    partitions = partition_grouped_candidate_pool(
        candidate_pool,
        camera,
        config=config,
        query_seed=int(query_seed),
    )
    fit_pool = partitions.fit
    shortlist_pool = partitions.shortlist
    verification_pool = partitions.verification
    audit_pool = partitions.audit
    verification_relation_kwargs = _grouped_relation_verification_kwargs(
        config, verification_pool
    )
    audit_relation_kwargs = _grouped_relation_verification_kwargs(
        config, audit_pool
    )
    fit_tokens = list(partitions.fit_tokens)
    verification_tokens = list(partitions.verification_tokens)
    audit_tokens = list(partitions.audit_tokens)
    if generation_candidate_pool is None:
        hypothesis_fit_pool = fit_pool
    else:
        generation_fit_source = generation_pool.subset_by_token_indices(fit_tokens)
        generation_spatial_likelihood = (
            None
            if generation_fit_source.spatial_likelihood is None
            else generation_fit_source.spatial_likelihood.mask_candidates(
                fit_pool.valid_mask
            )
        )
        # Cross-fit identity purging has already transferred removed candidate
        # mass into the scoring pool's null posterior. Recomputing that transfer
        # from the generation pool can introduce rounding drift and, more
        # importantly, would let generation replace non-spatial identity state.
        hypothesis_fit_pool = fit_pool.with_spatial_likelihood(
            generation_spatial_likelihood
        )
        _validate_generation_candidate_pool_compatibility(
            fit_pool, hypothesis_fit_pool
        )
    if (
        fit_pool.query_count < int(config.min_fit_matches)
        or shortlist_pool.query_count < 4
        or verification_pool.query_count < 4
        or audit_pool.query_count < 4
    ):
        return _empty_result(
            candidate_pool.query_count,
            fit_count=fit_pool.query_count,
            verification_count=verification_pool.query_count,
            final_audit_count=audit_pool.query_count,
            crossfit_partition_audit=partitions.partition_audit,
        )
    verification_denominator_sha256 = candidate_pool_likelihood_manifest_sha256(
        verification_pool,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
    )
    final_audit_denominator_sha256 = candidate_pool_likelihood_manifest_sha256(
        audit_pool,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
    )

    hypotheses: list[PoseHypothesisRecord] = []
    poses: list[np.ndarray | None] = []
    if str(config.generation_mode) in {
        "assignment_ransac",
        "assignment_plus_grouped_prosac",
    }:
        assignment_specs: list[tuple[int, str, float, int]] = [
            (1, "forced_top1", 1.0, -2),
            (1, "null_argmax", 1.0, -1),
        ]
        if float(candidate_pool.geometry_generation_mix_weight) > 0.0:
            assignment_specs.extend(
                (
                    min(int(candidate_limit), candidate_pool.track_ids.shape[1]),
                    "geometry_mixed_argmax",
                    1.0,
                    -3,
                )
                for candidate_limit in config.candidate_limits
            )
        for candidate_limit in config.candidate_limits:
            effective_limit = min(
                int(candidate_limit), hypothesis_fit_pool.track_ids.shape[1]
            )
            for temperature_index, temperature in enumerate(
                config.sampling_temperatures
            ):
                for sample_index in range(int(config.samples_per_limit)):
                    assignment_specs.append(
                        (
                            effective_limit,
                            "posterior_sample",
                            float(temperature),
                            int(
                                temperature_index * int(config.samples_per_limit)
                                + sample_index
                            ),
                        )
                    )

        for spec_index, (
            candidate_limit,
            mode,
            temperature,
            sample_index,
        ) in enumerate(assignment_specs):
            seed_payload = (
                f"{int(query_seed)}:{int(candidate_limit)}:{mode}:"
                f"{float(temperature):.8g}:{int(sample_index)}"
            ).encode("ascii")
            assignment_seed = int.from_bytes(
                hashlib.sha256(seed_payload).digest()[:8], "little"
            )
            assigned, null_count = _sample_grouped_candidate_assignment(
                hypothesis_fit_pool,
                candidate_limit=int(candidate_limit),
                mode=str(mode),
                temperature=float(temperature),
                seed=assignment_seed,
            )
            for fit_count in config.fit_match_counts:
                if len(assigned) < int(config.min_fit_matches):
                    continue
                selected = select_geometry_diverse_matches(
                    assigned,
                    camera,
                    max_matches=min(int(fit_count), len(assigned)),
                    grid_rows=int(config.grid_rows),
                    grid_cols=int(config.grid_cols),
                )
                if _grouped_fit_is_degenerate(
                    selected,
                    camera,
                    min_matches=int(config.min_fit_matches),
                    min_grid_cells=int(config.min_fit_grid_cells),
                    grid_rows=int(config.grid_rows),
                    grid_cols=int(config.grid_cols),
                    min_xyz_second_singular_ratio=float(
                        config.min_xyz_second_singular_ratio
                    ),
                ):
                    continue
                selected = stable_uniform_ransac_order(selected)
                for threshold_index, threshold in enumerate(
                    config.ransac_thresholds_px
                ):
                    solver_seed = int(
                        (
                            assignment_seed
                            + 104729 * int(fit_count)
                            + threshold_index
                        )
                        % (2**31 - 1)
                    )
                    _set_cv2_seed(solver_seed)
                    result = estimate_pose_pnp_ransac(
                        selected,
                        camera,
                        reprojection_error_px=float(threshold),
                        iterations=int(config.ransac_iterations),
                        refine_method="LM",
                    )
                    verification = verify_pose_candidate_pool(
                        result.pose_w2c,
                        verification_pool,
                        camera,
                        residual_sigma_px=float(
                            config.candidate_pool_residual_sigma_px
                        ),
                        candidate_outlier_likelihood=float(
                            config.candidate_pose_outlier_likelihood
                        ),
                        null_likelihood=float(
                            config.candidate_pose_null_likelihood
                        ),
                        hard_threshold_px=float(
                            config.candidate_pool_hard_threshold_px
                        ),
                        descriptor_rank_weight=float(
                            config.candidate_pool_descriptor_rank_weight
                        ),
                        strict_threshold_px=float(config.verification_strict_px),
                        loose_threshold_px=float(config.verification_loose_px),
                        grid_rows=int(config.grid_rows),
                        grid_cols=int(config.grid_cols),
                        **verification_relation_kwargs,
                    )
                    hypotheses.append(
                        PoseHypothesisRecord(
                            fit_match_count_limit=int(fit_count),
                            fit_match_count=int(len(selected)),
                            selection_mode=(
                                "grouped_progressive_"
                                f"{mode}_L{int(candidate_limit)}_"
                                f"T{float(temperature):g}_"
                                f"S{int(sample_index)}_N{int(null_count)}"
                            ),
                            ransac_threshold_px=float(threshold),
                            rng_seed_offset=int(spec_index),
                            solver_success=bool(result.success),
                            fit_inlier_count=int(result.inlier_count),
                            verification=verification,
                        )
                    )
                    poses.append(
                        None
                        if result.pose_w2c is None
                        else np.asarray(result.pose_w2c, dtype=np.float64).reshape(
                            4, 4
                        )
                    )

    if str(config.generation_mode) in {
        "grouped_prosac",
        "assignment_plus_grouped_prosac",
    }:
        generated = list(
            generate_grouped_prosac_hypotheses(
                hypothesis_fit_pool,
                camera,
                config=config,
                query_seed=int(query_seed),
            )
        )
        raw_generated_count = len(generated)
        verification_top_k = int(config.prosac_verification_top_k)
        generated_to_verify: set[int] = set()
        spatial_rescored_indices: set[int] = set()
        preliminary_score_by_index: dict[int, float] = {}
        shortlist_score_by_index: dict[int, float] = {}
        shortlist_evidence_by_index: dict[int, str] = {}
        shortlist_spatial_weight = (
            0.0
            if str(config.prosac_shortlist_evidence_mode)
            == "base_coordinate_then_full_spatial"
            else None
        )
        raw_indices = tuple(range(raw_generated_count))
        raw_valid_indices = {
            int(index)
            for index, item in enumerate(generated)
            if item.solver_success and item.pose_w2c is not None
        }
        needs_raw_shortlist_scores = bool(
            (
                verification_top_k > 0
                and len(raw_valid_indices) > verification_top_k
            )
            or (
                bool(config.latent_em_enabled)
                and str(config.latent_em_seed_evidence_mode)
                == "crossfit_shortlist"
            )
        )
        raw_preliminary_scores: list[tuple[float, float, int]] = []
        raw_shortlist_scores: list[tuple[float, float, int]] = []
        if needs_raw_shortlist_scores:
            raw_preliminary_scores = _score_grouped_hypothesis_indices(
                generated,
                shortlist_pool,
                camera,
                config,
                raw_indices,
                spatial_evidence_weight=shortlist_spatial_weight,
            )
            preliminary_evidence_mode = (
                "base_coordinate"
                if shortlist_spatial_weight == 0.0
                else "full_spatial"
            )
            for score, _sample_score, generated_index in raw_preliminary_scores:
                preliminary_score_by_index[int(generated_index)] = float(score)
                shortlist_score_by_index[int(generated_index)] = float(score)
                shortlist_evidence_by_index[int(generated_index)] = (
                    preliminary_evidence_mode
                )
            spatial_rescore_top_k = int(config.prosac_spatial_rescore_top_k)
            if spatial_rescore_top_k > 0:
                raw_rescore_indices = tuple(
                    int(item[2])
                    for item in raw_preliminary_scores[:spatial_rescore_top_k]
                )
                raw_shortlist_scores = _score_grouped_hypothesis_indices(
                    generated,
                    shortlist_pool,
                    camera,
                    config,
                    raw_rescore_indices,
                    spatial_evidence_weight=None,
                )
                for score, _sample_score, generated_index in raw_shortlist_scores:
                    index = int(generated_index)
                    spatial_rescored_indices.add(index)
                    shortlist_score_by_index[index] = float(score)
                    shortlist_evidence_by_index[index] = "full_spatial"
            else:
                raw_shortlist_scores = list(raw_preliminary_scores)

        if verification_top_k > 0 and len(raw_valid_indices) > verification_top_k:
            generated_to_verify.update(
                _select_configured_grouped_shortlist(
                    generated,
                    raw_shortlist_scores,
                    config,
                    top_k=verification_top_k,
                )
            )
        else:
            generated_to_verify.update(raw_valid_indices)

        if bool(config.latent_em_enabled):
            if (
                str(config.latent_em_seed_evidence_mode)
                == "fit_pool_DIAGNOSTIC_ONLY"
            ):
                seed_indices = _select_latent_em_seeds_fit_pool_diagnostic_only(
                    generated, hypothesis_fit_pool, camera, config
                )
            else:
                seed_indices = _select_latent_em_seeds_from_ranked(
                    generated, raw_shortlist_scores, config
                )
            em_variants: list[GroupedGeneratedPose] = []
            for seed_index in seed_indices:
                seed = generated[int(seed_index)]
                if seed.pose_w2c is None:
                    continue
                refined = refine_pose_latent_em(
                    hypothesis_fit_pool,
                    seed.pose_w2c,
                    camera,
                    config.latent_em_config,
                )
                if not refined.success:
                    continue
                refined_pose = np.asarray(
                    refined.pose_w2c, dtype=np.float64
                ).reshape(4, 4)
                resolved_matches, _selected_columns, _residuals = (
                    resolve_pose_guided_candidate_pool(
                        hypothesis_fit_pool,
                        refined_pose,
                        camera,
                        residual_sigma_px=float(
                            config.candidate_pool_residual_sigma_px
                        ),
                        hard_threshold_px=float(
                            config.candidate_pool_hard_threshold_px
                        ),
                        descriptor_rank_weight=float(
                            config.candidate_pool_descriptor_rank_weight
                        ),
                    )
                )
                observability_failures = grouped_pose_observability_gate_failures(
                    refined_pose, resolved_matches, camera, config
                )
                em_variants.append(
                    GroupedGeneratedPose(
                        pose_w2c=(
                            refined_pose if not observability_failures else None
                        ),
                        sample=seed.sample,
                        solver_success=not bool(observability_failures),
                        latent_em_applied=True,
                        latent_em_iterations=int(
                            refined.accepted_iterations
                        ),
                        latent_em_final_log_likelihood=float(
                            refined.final_log_likelihood_sum
                        ),
                        latent_em_parent_generated_index=int(seed_index),
                        latent_em_seed_evidence_mode=str(
                            config.latent_em_seed_evidence_mode
                        ),
                        observability_gate_failures=observability_failures,
                    )
                )
            generated.extend(em_variants)
            em_indices = tuple(range(raw_generated_count, len(generated)))
            em_valid_indices = {
                int(index)
                for index in em_indices
                if generated[int(index)].solver_success
                and generated[int(index)].pose_w2c is not None
            }
            # EM variants are optional additions. They must never evict an
            # immutable raw hypothesis from either shortlist stage.
            generated_to_verify.update(em_valid_indices)
            if verification_top_k > 0 and em_valid_indices:
                em_preliminary_scores = _score_grouped_hypothesis_indices(
                    generated,
                    shortlist_pool,
                    camera,
                    config,
                    tuple(sorted(em_valid_indices)),
                    spatial_evidence_weight=shortlist_spatial_weight,
                )
                preliminary_evidence_mode = (
                    "base_coordinate"
                    if shortlist_spatial_weight == 0.0
                    else "full_spatial"
                )
                for score, _sample_score, generated_index in em_preliminary_scores:
                    index = int(generated_index)
                    preliminary_score_by_index[index] = float(score)
                    shortlist_score_by_index[index] = float(score)
                    shortlist_evidence_by_index[index] = preliminary_evidence_mode
                if int(config.prosac_spatial_rescore_top_k) > 0:
                    em_spatial_scores = _score_grouped_hypothesis_indices(
                        generated,
                        shortlist_pool,
                        camera,
                        config,
                        tuple(sorted(em_valid_indices)),
                        spatial_evidence_weight=None,
                    )
                    for score, _sample_score, generated_index in em_spatial_scores:
                        index = int(generated_index)
                        spatial_rescored_indices.add(index)
                        shortlist_score_by_index[index] = float(score)
                        shortlist_evidence_by_index[index] = "full_spatial"
        generated_hypothesis_offset = len(hypotheses)
        for generated_index, item in enumerate(generated):
            verification = (
                None
                if generated_index not in generated_to_verify
                else verify_pose_candidate_pool(
                    item.pose_w2c,
                    verification_pool,
                    camera,
                    residual_sigma_px=float(
                        config.candidate_pool_residual_sigma_px
                    ),
                    candidate_outlier_likelihood=float(
                        config.candidate_pose_outlier_likelihood
                    ),
                    null_likelihood=float(
                        config.candidate_pose_null_likelihood
                    ),
                    hard_threshold_px=float(
                        config.candidate_pool_hard_threshold_px
                    ),
                    descriptor_rank_weight=float(
                        config.candidate_pool_descriptor_rank_weight
                    ),
                    strict_threshold_px=float(config.verification_strict_px),
                    loose_threshold_px=float(config.verification_loose_px),
                    grid_rows=int(config.grid_rows),
                    grid_cols=int(config.grid_cols),
                    **verification_relation_kwargs,
                )
            )
            sample = item.sample
            sample_tracks = tuple(int(match.track_id) for match in sample.matches)
            sample_tokens = tuple(int(match.token_index) for match in sample.matches)
            hypotheses.append(
                PoseHypothesisRecord(
                    fit_match_count_limit=int(len(sample.matches)),
                    fit_match_count=int(len(sample.matches)),
                    selection_mode=(
                        "grouped_prosac_minimal_set_"
                        f"{str(sample.generation_profile)}_"
                        f"P{int(sample.prosac_prefix_size)}_"
                        f"M{int(sample.spatial_mode_count)}_"
                        f"{'EM' if item.latent_em_applied else ('LO' if item.local_optimization_applied else 'RAW')}"
                    ),
                    ransac_threshold_px=float(config.prosac_local_consensus_px),
                    rng_seed_offset=int(generated_index),
                    solver_success=bool(item.solver_success),
                    fit_inlier_count=int(item.local_optimization_inlier_count),
                    verification=verification,
                    sample_token_indices=sample_tokens,
                    sample_track_ids=sample_tracks,
                    prosac_prefix_size=int(sample.prosac_prefix_size),
                    sampling_log_probability=float(
                        sample.sampling_log_probability
                    ),
                    local_optimization_match_count=int(
                        item.local_optimization_match_count
                    ),
                    local_optimization_applied=bool(
                        item.local_optimization_applied
                    ),
                    latent_em_applied=bool(item.latent_em_applied),
                    latent_em_iterations=int(item.latent_em_iterations),
                    latent_em_final_log_likelihood=(
                        None
                        if item.latent_em_final_log_likelihood is None
                        else float(item.latent_em_final_log_likelihood)
                    ),
                    latent_em_parent_hypothesis_index=(
                        None
                        if item.latent_em_parent_generated_index is None
                        else int(generated_hypothesis_offset)
                        + int(item.latent_em_parent_generated_index)
                    ),
                    latent_em_seed_evidence_mode=(
                        item.latent_em_seed_evidence_mode
                    ),
                    observability_gate_failures=tuple(
                        item.observability_gate_failures
                    ),
                    generation_profile=str(sample.generation_profile),
                    spatial_rescored_for_shortlist=(
                        int(generated_index) in spatial_rescored_indices
                    ),
                    preliminary_log_likelihood_mean=(
                        preliminary_score_by_index.get(int(generated_index))
                    ),
                    shortlist_log_likelihood_mean=(
                        shortlist_score_by_index.get(int(generated_index))
                    ),
                    shortlist_evidence_mode=(
                        shortlist_evidence_by_index.get(int(generated_index))
                    ),
                )
            )
            poses.append(item.pose_w2c)

    eligible = [
        index
        for index, record in enumerate(hypotheses)
        if record.solver_success and record.verification is not None
    ]
    if not eligible:
        return _empty_result(
            candidate_pool.query_count,
            fit_count=fit_pool.query_count,
            verification_count=verification_pool.query_count,
            final_audit_count=audit_pool.query_count,
            hypotheses=hypotheses,
            hypothesis_poses_w2c=poses,
            crossfit_partition_audit=partitions.partition_audit,
        )
    if hypothesis_selector is None:
        chosen_index = max(
            eligible,
            key=lambda index: (
                (
                    hypotheses[index].verification.hypothesis_selection_rank_key(  # type: ignore[union-attr]
                        str(config.hypothesis_selection_policy)
                    )
                ),
                -int(index),
            ),
        )
    else:
        chosen_index = int(hypothesis_selector(hypotheses, poses, eligible))
        if chosen_index not in set(eligible):
            raise ValueError(
                "hypothesis_selector must return an eligible hypothesis index"
            )
    chosen_pose = poses[chosen_index]
    if chosen_pose is None:
        raise RuntimeError("eligible grouped hypothesis has no pose")
    pre_refine_verification = hypotheses[chosen_index].verification
    pre_refine_audit = verify_pose_candidate_pool(
        chosen_pose,
        audit_pool,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
        hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
        descriptor_rank_weight=float(config.candidate_pool_descriptor_rank_weight),
        strict_threshold_px=float(config.verification_strict_px),
        loose_threshold_px=float(config.verification_loose_px),
        grid_rows=int(config.grid_rows),
        grid_cols=int(config.grid_cols),
        **audit_relation_kwargs,
    )

    final_pose = chosen_pose
    refinement_verification = pre_refine_verification
    candidate_coordinate_update_count = 0
    candidate_coordinate_update_grid_cell_count = 0
    candidate_coordinate_refine_attempted = False
    candidate_coordinate_refine_accepted = False
    final_refine_mode_used: str | None = None
    final_refine_attempted = False
    final_refine_accepted = False
    if bool(config.enable_final_refine):
        refinement_tokens = [*fit_tokens, *verification_tokens]
        refinement_pool = candidate_pool.subset_by_token_indices(refinement_tokens)
        refine_mode = str(config.final_refine_mode)
        if refine_mode == "auto":
            refine_mode = (
                "latent_em" if bool(config.latent_em_enabled) else "hard_assignment"
            )
        final_refine_mode_used = refine_mode
        proposed_pose: np.ndarray | None = None
        if refine_mode == "latent_em":
            final_refine_attempted = True
            refined_latent = refine_pose_latent_em(
                refinement_pool,
                chosen_pose,
                camera,
                config.latent_em_config,
            )
            if refined_latent.success:
                proposed_pose = np.asarray(
                    refined_latent.pose_w2c, dtype=np.float64
                ).reshape(4, 4)
        else:
            refinement_matches, selected_columns, selected_residuals = (
                resolve_pose_guided_candidate_pool(
                    refinement_pool,
                    chosen_pose,
                    camera,
                    residual_sigma_px=float(
                        config.candidate_pool_residual_sigma_px
                    ),
                    hard_threshold_px=float(
                        config.candidate_pool_hard_threshold_px
                    ),
                    descriptor_rank_weight=float(
                        config.candidate_pool_descriptor_rank_weight
                    ),
                )
            )
            accepted_residuals = selected_residuals[selected_columns >= 0]
            consensus_mask = accepted_residuals <= float(config.final_consensus_px)
            consensus_matches = [
                match
                for match, accepted in zip(refinement_matches, consensus_mask)
                if bool(accepted)
            ]
            if len(consensus_matches) >= int(config.min_final_inliers):
                final_refine_attempted = True
                weights = np.exp(
                    -0.5
                    * np.square(
                        accepted_residuals[consensus_mask]
                        / float(config.final_refine_f_scale_px)
                    )
                )
                refined = estimate_pose_pnp_fixed_robust(
                    consensus_matches,
                    camera,
                    weights=weights,
                    min_inliers=int(config.min_final_inliers),
                    initial_pose_w2c=chosen_pose,
                    loss="huber",
                    f_scale_px=float(config.final_refine_f_scale_px),
                    max_nfev=100,
                )
                if refined.success and refined.pose_w2c is not None:
                    proposed_pose = np.asarray(
                        refined.pose_w2c, dtype=np.float64
                    ).reshape(4, 4)
        if proposed_pose is not None:
            proposed_verification = verify_pose_candidate_pool(
                proposed_pose,
                verification_pool,
                camera,
                residual_sigma_px=float(
                    config.candidate_pool_residual_sigma_px
                ),
                candidate_outlier_likelihood=float(
                    config.candidate_pose_outlier_likelihood
                ),
                null_likelihood=float(config.candidate_pose_null_likelihood),
                hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
                descriptor_rank_weight=float(
                    config.candidate_pool_descriptor_rank_weight
                ),
                strict_threshold_px=float(config.verification_strict_px),
                loose_threshold_px=float(config.verification_loose_px),
                grid_rows=int(config.grid_rows),
                grid_cols=int(config.grid_cols),
                **verification_relation_kwargs,
            )
            if _accept_grouped_final_refine(
                refinement_verification,
                proposed_verification,
                policy=str(config.final_refine_acceptance_policy),
            ):
                final_pose = proposed_pose
                refinement_verification = proposed_verification
                final_refine_accepted = True

    if bool(config.enable_candidate_coordinate_refine):
        refinement_tokens = [*fit_tokens, *verification_tokens]
        refinement_pool = candidate_pool.subset_by_token_indices(refinement_tokens)
        refinement_matches, selected_columns, selected_residuals = (
            resolve_pose_guided_candidate_pool(
                refinement_pool,
                final_pose,
                camera,
                residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
                hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
                descriptor_rank_weight=float(
                    config.candidate_pool_descriptor_rank_weight
                ),
            )
        )
        coordinate_matches, candidate_coordinate_update_count = (
            _apply_selected_candidate_coordinate_updates(
                refinement_matches, selected_columns, refinement_pool
            )
        )
        accepted_residuals = selected_residuals[selected_columns >= 0]
        consensus_mask = accepted_residuals <= float(config.final_consensus_px)
        consensus_matches = [
            match
            for match, accepted in zip(coordinate_matches, consensus_mask)
            if bool(accepted)
        ]
        applied_consensus_updates = sum(
            match.source.endswith(":rgb_coordinate_update")
            for match in consensus_matches
        )
        candidate_coordinate_update_count = int(applied_consensus_updates)
        updated_consensus_matches = [
            match
            for match in consensus_matches
            if match.source.endswith(":rgb_coordinate_update")
        ]
        candidate_coordinate_update_grid_cell_count = _grid_cell_count(
            updated_consensus_matches,
            np.ones((len(updated_consensus_matches),), dtype=bool),
            image_width=int(camera.width),
            image_height=int(camera.height),
            grid_rows=int(config.grid_rows),
            grid_cols=int(config.grid_cols),
        )
        if (
            candidate_coordinate_update_count
            >= int(config.min_candidate_coordinate_updates)
            and candidate_coordinate_update_grid_cell_count
            >= int(config.min_candidate_coordinate_update_grid_cells)
            and len(consensus_matches) >= int(config.min_final_inliers)
        ):
            candidate_coordinate_refine_attempted = True
            weights = np.exp(
                -0.5
                * np.square(
                    accepted_residuals[consensus_mask]
                    / float(config.final_refine_f_scale_px)
                )
            )
            refined = estimate_pose_pnp_fixed_robust(
                consensus_matches,
                camera,
                weights=weights,
                min_inliers=int(config.min_final_inliers),
                initial_pose_w2c=final_pose,
                loss="huber",
                f_scale_px=float(config.final_refine_f_scale_px),
                max_nfev=100,
            )
            if refined.success and refined.pose_w2c is not None:
                proposed_pose = np.asarray(
                    refined.pose_w2c, dtype=np.float64
                ).reshape(4, 4)
                proposed_verification = verify_pose_candidate_pool(
                    proposed_pose,
                    verification_pool,
                    camera,
                    residual_sigma_px=float(
                        config.candidate_pool_residual_sigma_px
                    ),
                    candidate_outlier_likelihood=float(
                        config.candidate_pose_outlier_likelihood
                    ),
                    null_likelihood=float(
                        config.candidate_pose_null_likelihood
                    ),
                    hard_threshold_px=float(
                        config.candidate_pool_hard_threshold_px
                    ),
                    descriptor_rank_weight=float(
                        config.candidate_pool_descriptor_rank_weight
                    ),
                    strict_threshold_px=float(config.verification_strict_px),
                    loose_threshold_px=float(config.verification_loose_px),
                    grid_rows=int(config.grid_rows),
                    grid_cols=int(config.grid_cols),
                    **verification_relation_kwargs,
                )
                if _accept_grouped_final_refine(
                    refinement_verification,
                    proposed_verification,
                    policy=str(config.final_refine_acceptance_policy),
                ):
                    final_pose = proposed_pose
                    refinement_verification = proposed_verification
                    candidate_coordinate_refine_accepted = True

    final_audit = verify_pose_candidate_pool(
        final_pose,
        audit_pool,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
        hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
        descriptor_rank_weight=float(config.candidate_pool_descriptor_rank_weight),
        strict_threshold_px=float(config.verification_strict_px),
        loose_threshold_px=float(config.verification_loose_px),
        grid_rows=int(config.grid_rows),
        grid_cols=int(config.grid_cols),
        **audit_relation_kwargs,
    )

    _final_matches, _final_columns, final_residuals = resolve_pose_guided_candidate_pool(
        candidate_pool,
        final_pose,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
        descriptor_rank_weight=float(config.candidate_pool_descriptor_rank_weight),
    )
    final_inliers = np.isfinite(final_residuals) & (
        final_residuals <= float(config.final_consensus_px)
    )
    return VerifiedPnPResult(
        success=True,
        pose_w2c=final_pose,
        inlier_mask=final_inliers,
        match_count=int(candidate_pool.query_count),
        inlier_count=int(np.sum(final_inliers)),
        fit_count=int(fit_pool.query_count),
        verification_count=int(verification_pool.query_count),
        final_audit_count=int(audit_pool.query_count),
        chosen_hypothesis_index=int(chosen_index),
        hypotheses=tuple(hypotheses),
        hypothesis_poses_w2c=tuple(poses),
        pre_refine_pose_w2c=chosen_pose,
        pre_refine_verification=pre_refine_verification,
        pre_refine_final_audit_verification=pre_refine_audit,
        final_verification=final_audit,
        final_refine_mode_used=final_refine_mode_used,
        final_refine_attempted=bool(final_refine_attempted),
        final_refine_accepted=bool(final_refine_accepted),
        candidate_coordinate_update_count=int(candidate_coordinate_update_count),
        candidate_coordinate_update_grid_cell_count=int(
            candidate_coordinate_update_grid_cell_count
        ),
        candidate_coordinate_refine_attempted=bool(
            candidate_coordinate_refine_attempted
        ),
        candidate_coordinate_refine_accepted=bool(
            candidate_coordinate_refine_accepted
        ),
        verification_denominator_sha256=verification_denominator_sha256,
        final_audit_denominator_sha256=final_audit_denominator_sha256,
        crossfit_partition_audit=partitions.partition_audit,
    )


def reverify_grouped_result_on_shared_denominator(
    result: VerifiedPnPResult,
    candidate_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    config: GroupedCandidatePnPConfig,
    query_seed: int,
) -> VerifiedPnPResult:
    """Keep a baseline pose fixed while replacing only held-out evidence."""

    if not result.success:
        return result
    if result.pose_w2c is None or result.pre_refine_pose_w2c is None:
        raise ValueError("successful grouped result is missing a pose")
    partitions = partition_grouped_candidate_pool(
        candidate_pool,
        camera,
        config=config,
        query_seed=int(query_seed),
    )

    def verify(pose: np.ndarray, pool: PoseVerificationCandidatePool):
        relation_kwargs = _grouped_relation_verification_kwargs(config, pool)
        return verify_pose_candidate_pool(
            pose,
            pool,
            camera,
            residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
            candidate_outlier_likelihood=float(
                config.candidate_pose_outlier_likelihood
            ),
            null_likelihood=float(config.candidate_pose_null_likelihood),
            hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
            descriptor_rank_weight=float(
                config.candidate_pool_descriptor_rank_weight
            ),
            strict_threshold_px=float(config.verification_strict_px),
            loose_threshold_px=float(config.verification_loose_px),
            grid_rows=int(config.grid_rows),
            grid_cols=int(config.grid_cols),
            **relation_kwargs,
        )

    pre_refine_verification = verify(
        result.pre_refine_pose_w2c, partitions.verification
    )
    pre_refine_audit = verify(result.pre_refine_pose_w2c, partitions.audit)
    final_audit = verify(result.pose_w2c, partitions.audit)
    if pre_refine_verification is None:
        raise RuntimeError("shared verification denominator produced no evidence")
    hypotheses = list(result.hypotheses)
    if result.chosen_hypothesis_index is not None:
        chosen_index = int(result.chosen_hypothesis_index)
        hypotheses[chosen_index] = replace(
            hypotheses[chosen_index], verification=pre_refine_verification
        )
    denominator_sha256 = candidate_pool_likelihood_manifest_sha256(
        partitions.verification,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
    )
    final_audit_denominator_sha256 = candidate_pool_likelihood_manifest_sha256(
        partitions.audit,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
    )
    return replace(
        result,
        fit_count=int(partitions.fit.query_count),
        verification_count=int(partitions.verification.query_count),
        final_audit_count=int(partitions.audit.query_count),
        hypotheses=tuple(hypotheses),
        pre_refine_verification=pre_refine_verification,
        pre_refine_final_audit_verification=pre_refine_audit,
        final_verification=final_audit,
        verification_denominator_sha256=denominator_sha256,
        final_audit_denominator_sha256=final_audit_denominator_sha256,
        crossfit_partition_audit=partitions.partition_audit,
    )


def wrap_immutable_pose_on_grouped_denominator(
    source: PnPResult,
    candidate_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    config: GroupedCandidatePnPConfig,
    query_seed: int,
) -> VerifiedPnPResult:
    """Wrap an already-frozen pose without ever re-estimating it.

    Only held-out likelihood fields are added. On fallback, ``pose_w2c`` is the
    exact source array and no grouped assignment, refine, or solver is run.
    """

    if not source.success or source.pose_w2c is None:
        return _empty_result(
            int(source.match_count),
            fit_count=0,
            verification_count=0,
            final_audit_count=0,
        )
    pose = np.asarray(source.pose_w2c, dtype=np.float64).reshape(4, 4)
    partitions = partition_grouped_candidate_pool(
        candidate_pool,
        camera,
        config=config,
        query_seed=int(query_seed),
    )
    verification = verify_pose_candidate_pool(
        pose,
        partitions.verification,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
        hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
        descriptor_rank_weight=float(config.candidate_pool_descriptor_rank_weight),
        strict_threshold_px=float(config.verification_strict_px),
        loose_threshold_px=float(config.verification_loose_px),
        grid_rows=int(config.grid_rows),
        grid_cols=int(config.grid_cols),
        **_grouped_relation_verification_kwargs(
            config, partitions.verification
        ),
    )
    audit = verify_pose_candidate_pool(
        pose,
        partitions.audit,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
        hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
        descriptor_rank_weight=float(config.candidate_pool_descriptor_rank_weight),
        strict_threshold_px=float(config.verification_strict_px),
        loose_threshold_px=float(config.verification_loose_px),
        grid_rows=int(config.grid_rows),
        grid_cols=int(config.grid_cols),
        **_grouped_relation_verification_kwargs(config, partitions.audit),
    )
    verification_hash = candidate_pool_likelihood_manifest_sha256(
        partitions.verification,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
    )
    audit_hash = candidate_pool_likelihood_manifest_sha256(
        partitions.audit,
        camera,
        residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
        candidate_outlier_likelihood=float(
            config.candidate_pose_outlier_likelihood
        ),
        null_likelihood=float(config.candidate_pose_null_likelihood),
    )
    return VerifiedPnPResult(
        success=True,
        pose_w2c=pose,
        inlier_mask=np.asarray(source.inlier_mask, dtype=bool).copy(),
        match_count=int(source.match_count),
        inlier_count=int(source.inlier_count),
        fit_count=int(partitions.fit.query_count),
        verification_count=int(partitions.verification.query_count),
        final_audit_count=int(partitions.audit.query_count),
        chosen_hypothesis_index=None,
        hypotheses=(),
        hypothesis_poses_w2c=(),
        pre_refine_pose_w2c=pose,
        pre_refine_verification=verification,
        pre_refine_final_audit_verification=audit,
        final_verification=audit,
        verification_denominator_sha256=verification_hash,
        final_audit_denominator_sha256=audit_hash,
        crossfit_partition_audit=partitions.partition_audit,
    )
