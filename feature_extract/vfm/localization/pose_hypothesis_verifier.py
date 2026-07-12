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

    def __post_init__(self) -> None:
        tokens = np.asarray(self.token_indices, dtype=np.int64).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float64).reshape(-1, 2)
        tracks = np.asarray(self.track_ids, dtype=np.int64)
        prototypes = np.asarray(self.prototype_ids, dtype=np.int64)
        xyz = np.asarray(self.xyz, dtype=np.float64)
        scores = np.asarray(self.descriptor_scores, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
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
        object.__setattr__(self, "token_indices", tokens)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "track_ids", tracks)
        object.__setattr__(self, "prototype_ids", prototypes)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "descriptor_scores", scores)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "measurement_geometry_probabilities", measurement)
        object.__setattr__(self, "measurement_verification_threshold", threshold)

    @property
    def query_count(self) -> int:
        return int(len(self.token_indices))

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
    final_audit_fold: int = 1,
    grid_rows: int = 4,
    grid_cols: int = 4,
    salt: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint hypothesis-fit, rank-verification, and final-audit rows."""

    if int(verification_fold) == int(final_audit_fold):
        raise ValueError("verification and final audit folds must differ")
    _fit, verification = deterministic_spatial_holdout(
        matches,
        image_width=int(image_width),
        image_height=int(image_height),
        folds=int(folds),
        fold=int(verification_fold),
        grid_rows=int(grid_rows),
        grid_cols=int(grid_cols),
        salt=int(salt),
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


def _candidate_pool_reprojection_residuals(
    pool: PoseVerificationCandidatePool,
    pose_w2c: np.ndarray,
    camera: ColmapCamera,
) -> tuple[np.ndarray, np.ndarray]:
    if pool.query_count == 0:
        shape = pool.track_ids.shape
        return np.full(shape, np.inf, dtype=np.float64), np.zeros(shape, dtype=bool)
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
    residuals = np.linalg.norm(projected - pool.xy[:, None, :], axis=2)
    camera_xyz = flat_xyz @ pose[:3, :3].T + pose[:3, 3]
    positive = camera_xyz[:, 2].reshape(pool.track_ids.shape) > 1e-6
    valid = pool.valid_mask & positive & np.isfinite(residuals)
    residuals[~valid] = np.inf
    return residuals, valid


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
    accepted_rows = np.flatnonzero(selected >= 0)
    if len(accepted_rows) == 0:
        return replace(verification, **measurement_statistics)
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
                hypotheses[index].verification.rank_key(),  # type: ignore[union-attr]
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
