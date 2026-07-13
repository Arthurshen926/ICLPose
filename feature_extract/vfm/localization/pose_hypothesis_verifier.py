"""Inference-only multi-hypothesis PnP with held-out geometric verification.

The verifier deliberately has no ground-truth pose input. A fixed subset of
the correspondences is withheld from hypothesis fitting and is used only to
rank poses by reprojection consistency, image coverage, and cheirality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from typing import Callable, Optional, Sequence

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.pose_safe_selection import (
    resolve_global_query_track_assignment,
    resolve_pose_match_conflicts,
    select_pose_safe_matches,
    stable_uniform_ransac_order,
)
from feature_extract.vfm.query_to_3d_matching import (
    PnPResult,
    QueryTo3DMatch,
    camera_matrix_and_distortion,
    estimate_pose_pnp_fixed_robust,
    estimate_pose_pnp_ransac,
    match_reprojection_errors,
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
    grid_rows: int = 4
    grid_cols: int = 4
    min_fit_matches: int = 8
    min_fit_grid_cells: int = 4
    min_xyz_second_singular_ratio: float = 1e-3
    verification_strict_px: float = 2.0
    verification_loose_px: float = 5.0
    candidate_pool_residual_sigma_px: float = 2.0
    candidate_pool_hard_threshold_px: float = 8.0
    candidate_pool_descriptor_rank_weight: float = 0.02
    final_consensus_px: float = 4.0
    final_refine_f_scale_px: float = 2.0
    min_final_inliers: int = 6
    enable_final_refine: bool = True
    final_refine_acceptance_policy: str = "legacy_rank_key"
    enable_candidate_coordinate_refine: bool = False
    min_candidate_coordinate_updates: int = 4
    min_candidate_coordinate_update_grid_cells: int = 2

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
        if str(self.final_refine_acceptance_policy) not in {
            "legacy_rank_key",
            "strict_count_gain_with_grid_nondecrease",
        }:
            raise ValueError("unsupported grouped final refine acceptance policy")
        if int(self.min_candidate_coordinate_updates) < 4:
            raise ValueError("candidate coordinate refinement requires at least four updates")
        if int(self.min_candidate_coordinate_update_grid_cells) <= 0:
            raise ValueError("candidate coordinate update grid coverage must be positive")


@dataclass(frozen=True)
class CandidateSpatialLikelihood:
    """Candidate/view-specific local likelihood maps aligned to a top-L pool."""

    offsets_xy: np.ndarray
    local_log_probabilities: np.ndarray
    view_probabilities: np.ndarray
    dustbin_probabilities: np.ndarray
    valid_mask: np.ndarray
    log_evidence_weight: float = 1.0

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets_xy, dtype=np.float64).reshape(-1, 2)
        log_probabilities = np.asarray(
            self.local_log_probabilities, dtype=np.float64
        )
        view_probabilities = np.asarray(self.view_probabilities, dtype=np.float64)
        dustbin_probabilities = np.asarray(
            self.dustbin_probabilities, dtype=np.float64
        )
        valid = np.asarray(self.valid_mask, dtype=bool)
        log_evidence_weight = float(self.log_evidence_weight)
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
        if np.any(~np.isfinite(offsets)):
            raise ValueError("spatial likelihood offsets must be finite")
        if np.any(~np.isfinite(view_probabilities[valid])) or np.any(
            view_probabilities[valid] < 0.0
        ):
            raise ValueError("spatial view probabilities must be finite and non-negative")
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
        object.__setattr__(self, "_grid_x", xs)
        object.__setattr__(self, "_grid_y", ys)
        object.__setattr__(self, "_grid_order", grid_order)
        probability_maps = np.exp(log_probabilities[:, :, :, grid_order]).reshape(
            *valid.shape, len(ys), len(xs)
        )
        object.__setattr__(
            self, "_probability_maps", probability_maps.astype(np.float32)
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
    geometry_prior_mix_weight: float = 0.0
    spatial_geometry_calibration_weight: float = 0.0
    geometry_generation_mix_weight: float = 0.0
    candidate_update_probabilities: np.ndarray | None = None
    candidate_refined_xy: np.ndarray | None = None
    candidate_update_threshold: float = 0.5

    def __post_init__(self) -> None:
        explicit_null = self.null_scores is not None
        tokens = np.asarray(self.token_indices, dtype=np.int64).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float64).reshape(-1, 2)
        tracks = np.asarray(self.track_ids, dtype=np.int64)
        prototypes = np.asarray(self.prototype_ids, dtype=np.int64)
        xyz = np.asarray(self.xyz, dtype=np.float64)
        scores = np.asarray(self.descriptor_scores, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool).copy()
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
            else np.asarray(
                self.measurement_geometry_probabilities, dtype=np.float64
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
            else np.asarray(self.null_scores, dtype=np.float64).reshape(-1)
        )
        if null_scores.shape != (len(tokens),):
            raise ValueError("null scores must contain one value per query group")
        if np.any(~np.isfinite(null_scores)) or np.any(null_scores < 0.0):
            raise ValueError("null scores must be finite and non-negative")
        spatial_likelihood = self.spatial_likelihood
        if (
            spatial_likelihood is not None
            and spatial_likelihood.candidate_shape != tracks.shape
        ):
            raise ValueError("spatial likelihood and candidate pool shapes differ")
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
            else np.asarray(self.candidate_update_probabilities, dtype=np.float64)
        )
        refined_xy = (
            np.full((*tracks.shape, 2), np.nan, dtype=np.float64)
            if self.candidate_refined_xy is None
            else np.asarray(self.candidate_refined_xy, dtype=np.float64)
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

    @property
    def query_count(self) -> int:
        return int(len(self.token_indices))

    @property
    def has_explicit_null(self) -> bool:
        return bool(self._has_explicit_null)

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
            self.token_indices[rows],
            self.xy[rows],
            self.track_ids[rows],
            self.prototype_ids[rows],
            self.xyz[rows],
            self.descriptor_scores[rows],
            self.valid_mask[rows],
            self.measurement_geometry_probabilities[rows],
            self.measurement_verification_threshold,
            None if not self.has_explicit_null else self.null_scores[rows],
            (
                None
                if self.spatial_likelihood is None
                else self.spatial_likelihood.subset(rows)
            ),
            self.geometry_prior_mix_weight,
            self.spatial_geometry_calibration_weight,
            self.geometry_generation_mix_weight,
            self.candidate_update_probabilities[rows],
            self.candidate_refined_xy[rows],
            self.candidate_update_threshold,
        )


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
    fixed_posterior_effective_group_count: int = 0
    fixed_posterior_mass_max_abs_error: float | None = None
    fixed_posterior_spatial_candidate_count: int = 0
    fixed_posterior_geometry_prior_evidence_count: int = 0
    fixed_posterior_geometry_prior_mix_weight: float = 0.0
    fixed_posterior_spatial_calibrated_candidate_count: int = 0
    fixed_posterior_spatial_geometry_calibration_weight: float = 0.0

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
        """Rank poses by one immutable top-L likelihood denominator."""

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
    candidate_coordinate_update_count: int = 0
    candidate_coordinate_update_grid_cell_count: int = 0
    candidate_coordinate_refine_attempted: bool = False
    candidate_coordinate_refine_accepted: bool = False

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
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, spatially distributed fit and verification rows."""

    values = list(matches)
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if int(folds) < 2 or not 0 <= int(fold) < int(folds):
        raise ValueError("invalid holdout fold configuration")
    if int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("grid dimensions must be positive")
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
    for cell in sorted(buckets):
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
        verification.extend(
            index
            for local_index, index in enumerate(ordered)
            if local_index % int(folds) == int(fold)
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
    )
    if np.intersect1d(verification, final_audit).size:
        raise RuntimeError("deterministic spatial folds unexpectedly overlap")
    excluded = np.zeros((len(matches),), dtype=bool)
    excluded[verification] = True
    excluded[final_audit] = True
    fit = np.flatnonzero(~excluded).astype(np.int64)
    return fit, verification, final_audit


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

    translation_information = jacobian[:, :3].T @ jacobian[:, :3]
    rotation_information = jacobian[:, 3:].T @ jacobian[:, 3:]
    joint_information = jacobian.T @ jacobian
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
    )


def _candidate_pool_projected_xy(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    if pool.query_count == 0:
        return (
            np.full((*pool.track_ids.shape, 2), np.nan, dtype=np.float64),
            np.zeros(pool.track_ids.shape, dtype=bool),
        )
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for candidate-pool projection") from exc
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    flat_xyz = pool.xyz.reshape(-1, 3)
    camera_matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(
        flat_xyz, rvec, pose[:3, 3], camera_matrix, distortion
    )
    projected = projected.reshape(*pool.track_ids.shape, 2)
    camera_xyz = flat_xyz @ pose[:3, :3].T + pose[:3, 3]
    positive = camera_xyz[:, 2].reshape(pool.track_ids.shape) > 1e-6
    in_bounds = (
        (projected[:, :, 0] >= 0.0)
        & (projected[:, :, 0] < float(camera.width))
        & (projected[:, :, 1] >= 0.0)
        & (projected[:, :, 1] < float(camera.height))
    )
    valid = pool.valid_mask & positive & in_bounds & np.all(np.isfinite(projected), axis=2)
    projected[~valid] = np.nan
    return projected, valid


def _candidate_pool_reprojection_residuals(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    projected, valid = _candidate_pool_projected_xy(pool, pose_w2c, camera)
    residuals = np.linalg.norm(projected - pool.xy[:, None, :], axis=2)
    residuals[~valid] = np.inf
    return residuals, valid


def _spatial_candidate_likelihood_ratios(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    outside_window_ratio: float,
) -> np.ndarray:
    spatial = pool.spatial_likelihood
    if spatial is None:
        return np.ones(pool.track_ids.shape, dtype=np.float64)
    outside = float(outside_window_ratio)
    if not 0.0 < outside <= 1.0:
        raise ValueError("outside_window_ratio must be in (0, 1]")
    projected, projection_valid = _candidate_pool_projected_xy(
        pool, pose_w2c, camera
    )
    offsets = projected - pool.xy[:, None, :]
    xs = np.asarray(spatial._grid_x, dtype=np.float64)
    ys = np.asarray(spatial._grid_y, dtype=np.float64)
    step_x = float(xs[1] - xs[0])
    step_y = float(ys[1] - ys[0])
    maps = spatial._probability_maps
    ratios = np.ones(pool.track_ids.shape, dtype=np.float64)
    calibration_weight = float(pool.spatial_geometry_calibration_weight)
    geometry_probability = np.asarray(
        pool.measurement_geometry_probabilities, dtype=np.float64
    )
    for row, column in np.argwhere(np.any(spatial.valid_mask, axis=2)).tolist():
        view_mask = np.asarray(spatial.valid_mask[row, column], dtype=bool)
        weights = np.asarray(
            spatial.view_probabilities[row, column, view_mask], dtype=np.float64
        )
        if not np.any(weights > 0.0):
            weights = np.ones_like(weights)
        weights /= np.sum(weights)
        dx, dy = offsets[row, column]
        inside = bool(
            projection_valid[row, column]
            and xs[0] <= dx <= xs[-1]
            and ys[0] <= dy <= ys[-1]
        )
        if inside:
            gx = np.clip((dx - xs[0]) / step_x, 0.0, len(xs) - 1.0)
            gy = np.clip((dy - ys[0]) / step_y, 0.0, len(ys) - 1.0)
            x0 = min(int(np.floor(gx)), len(xs) - 2)
            y0 = min(int(np.floor(gy)), len(ys) - 2)
            tx = float(gx - x0)
            ty = float(gy - y0)
            view_maps = maps[row, column, view_mask]
            probability = (
                (1.0 - tx) * (1.0 - ty) * view_maps[:, y0, x0]
                + tx * (1.0 - ty) * view_maps[:, y0, x0 + 1]
                + (1.0 - tx) * ty * view_maps[:, y0 + 1, x0]
                + tx * ty * view_maps[:, y0 + 1, x0 + 1]
            )
            # A uniform K-bin map has likelihood ratio one. This keeps missing
            # RGB neutral while preserving calibrated multimodal shape.
            view_ratios = probability * float(len(spatial.offsets_xy))
        else:
            view_ratios = np.full((int(np.sum(view_mask)),), outside)
        dustbin = spatial.dustbin_probabilities[row, column, view_mask]
        raw_reliability = 1.0 - dustbin
        calibrated_probability = float(geometry_probability[row, column])
        if calibration_weight > 0.0 and np.isfinite(calibrated_probability):
            # The calibrated probability is learned from true pose-projection
            # residuals. It controls whether the local RGB likelihood is
            # informative, but deliberately does not alter identity/null mass.
            reliability = (
                (1.0 - calibration_weight) * raw_reliability
                + calibration_weight * calibrated_probability
            )
        else:
            reliability = raw_reliability
        view_evidence = (1.0 - reliability) + reliability * view_ratios
        ratios[row, column] = float(np.sum(weights * view_evidence))
    return ratios


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


def fixed_posterior_pose_log_likelihood(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    residual_sigma_px: float = 2.0,
    candidate_outlier_likelihood: float = 1e-4,
    null_likelihood: float = 1.0,
    probability_mass_atol: float = 2e-4,
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
    residuals, projection_valid = _candidate_pool_reprojection_residuals(
        pool, pose_w2c, camera
    )
    gaussian = np.zeros_like(residuals, dtype=np.float64)
    gaussian[projection_valid] = np.exp(
        -0.5 * np.square(residuals[projection_valid] / sigma)
    )
    candidate_likelihood = outlier + (1.0 - outlier) * gaussian
    spatial_candidate_count = 0
    spatial_calibrated_candidate_count = 0
    if pool.spatial_likelihood is not None:
        spatial_ratio = _spatial_candidate_likelihood_ratios(
            pool,
            pose_w2c,
            camera,
            outside_window_ratio=outlier,
        )
        weight = float(pool.spatial_likelihood.log_evidence_weight)
        # The RGB branch exports a likelihood ratio against a uniform local
        # map, whereas the generic reprojection term is a bounded geometric
        # compatibility. Keep geometry as the anchor and add RGB only as a
        # tempered Bayes factor. Missing RGB has ratio one and is neutral;
        # weight zero is therefore an exact regression baseline.
        candidate_likelihood *= np.exp(
            weight * np.log(np.maximum(spatial_ratio, 1e-12))
        )
        spatial_candidate_count = int(
            np.sum(np.any(pool.spatial_likelihood.valid_mask, axis=2))
        )
        spatial_calibrated_candidate_count = int(
            np.sum(
                np.any(pool.spatial_likelihood.valid_mask, axis=2)
                & geometry_evidence
            )
        )
    group_likelihood = (
        null_scores * null_value
        + np.sum(probabilities * candidate_likelihood, axis=1)
    )
    log_likelihood = np.log(np.maximum(group_likelihood, 1e-12))
    informative = (1.0 - null_scores) > 1e-3
    return {
        "log_likelihood_sum": float(np.sum(log_likelihood)),
        "log_likelihood_mean": float(np.mean(log_likelihood)),
        "effective_group_count": int(np.sum(informative)),
        "mass_max_abs_error": float(np.max(mass_error)),
        "spatial_candidate_count": int(spatial_candidate_count),
        "geometry_prior_evidence_count": int(geometry_evidence_count),
        "geometry_prior_mix_weight": float(geometry_mix_weight),
        "spatial_calibrated_candidate_count": int(
            spatial_calibrated_candidate_count
        ),
        "spatial_geometry_calibration_weight": float(
            spatial_calibration_weight
        ),
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
        )
        fixed_posterior_statistics = {
            "fixed_posterior_log_likelihood_sum": float(
                fixed_posterior["log_likelihood_sum"]
            ),
            "fixed_posterior_log_likelihood_mean": float(
                fixed_posterior["log_likelihood_mean"]
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
            "fixed_posterior_spatial_calibrated_candidate_count": int(
                fixed_posterior["spatial_calibrated_candidate_count"]
            ),
            "fixed_posterior_spatial_geometry_calibration_weight": float(
                fixed_posterior["spatial_geometry_calibration_weight"]
            ),
        }
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
) -> VerifiedPnPResult:
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
        hypothesis_poses_w2c=tuple(None for _record in hypotheses),
        pre_refine_pose_w2c=None,
        pre_refine_verification=None,
        pre_refine_final_audit_verification=None,
        final_verification=None,
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
        )
    if hypothesis_selector is None:
        chosen_index = max(
            eligible,
            key=lambda index: (
                hypotheses[index].verification.fixed_posterior_rank_key(),  # type: ignore[union-attr]
                int(hypotheses[index].fit_inlier_count),
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
    final_verification = audit_verification(final_pose)
    # The final-audit fold has not participated in hypothesis fitting,
    # hypothesis ranking, candidate reassignment, or robust refitting.
    if (
        final_verification is None
        or pre_refine_final_audit is None
        or final_verification.rank_key()
        < pre_refine_final_audit.rank_key()
    ):
        final_pose = chosen_pose
        final_verification = pre_refine_final_audit
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


def estimate_pose_from_grouped_candidate_pool(
    candidate_pool: PoseVerificationCandidatePool,
    camera: ColmapCamera,
    *,
    config: GroupedCandidatePnPConfig = GroupedCandidatePnPConfig(),
    query_seed: int = 0,
    hypothesis_selector: HypothesisSelector | None = None,
) -> VerifiedPnPResult:
    """Generate PnP hypotheses while retaining top-L candidate uncertainty.

    Each fit hypothesis contains at most one 3D candidate per query token and
    at most one query token per physical track. Candidate groups used for
    hypothesis ranking and final audit are disjoint from fit groups.
    """

    if not candidate_pool.has_explicit_null:
        raise ValueError("grouped candidate PnP requires an explicit null posterior")
    representatives = _candidate_pool_representatives(candidate_pool)
    if len(representatives) < max(12, int(config.min_final_inliers)):
        return _empty_result(
            candidate_pool.query_count, fit_count=0, verification_count=0
        )
    fit_indices, verification_indices, audit_indices = deterministic_spatial_partitions(
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
    )
    fit_tokens = [representatives[int(index)].token_index for index in fit_indices]
    verification_tokens = [
        representatives[int(index)].token_index for index in verification_indices
    ]
    audit_tokens = [representatives[int(index)].token_index for index in audit_indices]
    fit_pool = candidate_pool.subset_by_token_indices(fit_tokens)
    verification_pool = candidate_pool.subset_by_token_indices(verification_tokens)
    audit_pool = candidate_pool.subset_by_token_indices(audit_tokens)
    if (
        fit_pool.query_count < int(config.min_fit_matches)
        or verification_pool.query_count < 4
        or audit_pool.query_count < 4
    ):
        return _empty_result(
            candidate_pool.query_count,
            fit_count=fit_pool.query_count,
            verification_count=verification_pool.query_count,
            final_audit_count=audit_pool.query_count,
        )

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
        effective_limit = min(int(candidate_limit), fit_pool.track_ids.shape[1])
        for temperature_index, temperature in enumerate(
            config.sampling_temperatures
        ):
            for sample_index in range(int(config.samples_per_limit)):
                assignment_specs.append(
                    (
                        effective_limit,
                        "posterior_sample",
                        float(temperature),
                        int(temperature_index * int(config.samples_per_limit) + sample_index),
                    )
                )

    hypotheses: list[PoseHypothesisRecord] = []
    poses: list[np.ndarray | None] = []
    for spec_index, (candidate_limit, mode, temperature, sample_index) in enumerate(
        assignment_specs
    ):
        seed_payload = (
            f"{int(query_seed)}:{int(candidate_limit)}:{mode}:"
            f"{float(temperature):.8g}:{int(sample_index)}"
        ).encode("ascii")
        assignment_seed = int.from_bytes(
            hashlib.sha256(seed_payload).digest()[:8], "little"
        )
        assigned, null_count = _sample_grouped_candidate_assignment(
            fit_pool,
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
                    (assignment_seed + 104729 * int(fit_count) + threshold_index)
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
                        selection_mode=(
                            "grouped_progressive_"
                            f"{mode}_L{int(candidate_limit)}_T{float(temperature):g}_"
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
                    else np.asarray(result.pose_w2c, dtype=np.float64).reshape(4, 4)
                )

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
        )
    if hypothesis_selector is None:
        chosen_index = max(
            eligible,
            key=lambda index: (
                hypotheses[index].verification.fixed_posterior_rank_key(),  # type: ignore[union-attr]
                int(hypotheses[index].fit_inlier_count),
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
        hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
        descriptor_rank_weight=float(config.candidate_pool_descriptor_rank_weight),
        strict_threshold_px=float(config.verification_strict_px),
        loose_threshold_px=float(config.verification_loose_px),
        grid_rows=int(config.grid_rows),
        grid_cols=int(config.grid_cols),
    )

    final_pose = chosen_pose
    final_audit = pre_refine_audit
    candidate_coordinate_update_count = 0
    candidate_coordinate_update_grid_cell_count = 0
    candidate_coordinate_refine_attempted = False
    candidate_coordinate_refine_accepted = False
    if bool(config.enable_final_refine):
        refinement_tokens = [*fit_tokens, *verification_tokens]
        refinement_pool = candidate_pool.subset_by_token_indices(refinement_tokens)
        refinement_matches, selected_columns, selected_residuals = (
            resolve_pose_guided_candidate_pool(
                refinement_pool,
                chosen_pose,
                camera,
                residual_sigma_px=float(config.candidate_pool_residual_sigma_px),
                hard_threshold_px=float(config.candidate_pool_hard_threshold_px),
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
                proposed_pose = np.asarray(refined.pose_w2c, dtype=np.float64).reshape(
                    4, 4
                )
                proposed_audit = verify_pose_candidate_pool(
                    proposed_pose,
                    audit_pool,
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
                if _accept_grouped_final_refine(
                    pre_refine_audit,
                    proposed_audit,
                    policy=str(config.final_refine_acceptance_policy),
                ):
                    final_pose = proposed_pose
                    final_audit = proposed_audit

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
                proposed_audit = verify_pose_candidate_pool(
                    proposed_pose,
                    audit_pool,
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
                if _accept_grouped_final_refine(
                    final_audit,
                    proposed_audit,
                    policy=str(config.final_refine_acceptance_policy),
                ):
                    final_pose = proposed_pose
                    final_audit = proposed_audit
                    candidate_coordinate_refine_accepted = True

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
    )
