"""Pose likelihood from query points unused by correspondence fitting.

This verifier deliberately stays outside the candidate pool used to generate a
pose.  A pose is evaluated by projecting the global landmark bank, applying an
SfM observation-view gate, and comparing the projected descriptors with a
fixed set of held-out query descriptors.  Ground-truth pose is not part of the
API.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.query_to_3d_matching import (
    LandmarkMapIndex,
    camera_matrix_and_distortion,
)
from feature_extract.vfm.render_pose_diagnostics import project_world_to_image


INDEPENDENT_LANDMARK_POSE_LIKELIHOOD_VERSION = (
    "independent_landmark_pose_likelihood_v2"
)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("descriptor arrays must have shape (N, C)")
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-6)


@dataclass(frozen=True)
class IndependentVerificationPoints:
    xy: np.ndarray
    descriptors: np.ndarray
    descriptor_reference_scores: np.ndarray
    source_row_indices: np.ndarray | None = None
    candidate_track_ids: np.ndarray | None = None
    candidate_descriptor_scores: np.ndarray | None = None
    candidate_null_probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        xy = np.asarray(self.xy, dtype=np.float64).reshape(-1, 2)
        descriptors = np.asarray(self.descriptors, dtype=np.float32)
        reference = np.asarray(
            self.descriptor_reference_scores, dtype=np.float64
        ).reshape(-1)
        if descriptors.ndim != 2 or descriptors.shape[0] != xy.shape[0]:
            raise ValueError("descriptors must have shape (N, C) aligned with xy")
        if reference.shape != (xy.shape[0],):
            raise ValueError("descriptor_reference_scores must have shape (N,)")
        if np.any(~np.isfinite(xy)) or np.any(~np.isfinite(descriptors)):
            raise ValueError("verification coordinates and descriptors must be finite")
        if np.any(~np.isfinite(reference)):
            raise ValueError("descriptor reference scores must be finite")
        source_rows = (
            np.arange(xy.shape[0], dtype=np.int64)
            if self.source_row_indices is None
            else np.asarray(self.source_row_indices, dtype=np.int64).reshape(-1)
        )
        if source_rows.shape != (xy.shape[0],):
            raise ValueError("source_row_indices must have shape (N,)")
        if len(np.unique(source_rows)) != len(source_rows):
            raise ValueError("verification source rows must be unique")
        candidate_tracks = self.candidate_track_ids
        candidate_scores = self.candidate_descriptor_scores
        candidate_null = self.candidate_null_probabilities
        if (candidate_tracks is None) != (candidate_scores is None):
            raise ValueError(
                "candidate track IDs and descriptor scores must be supplied together"
            )
        if candidate_tracks is not None:
            candidate_tracks = np.asarray(candidate_tracks, dtype=np.int64)
            candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
            if (
                candidate_tracks.ndim != 2
                or candidate_tracks.shape[0] != xy.shape[0]
                or candidate_tracks.shape != candidate_scores.shape
                or candidate_tracks.shape[1] == 0
            ):
                raise ValueError(
                    "fixed candidates must have aligned non-empty shape (N, L)"
                )
            valid_candidates = candidate_tracks >= 0
            if np.any(~np.isfinite(candidate_scores[valid_candidates])):
                raise ValueError("valid fixed candidate scores must be finite")
            for row in range(candidate_tracks.shape[0]):
                valid_row = candidate_tracks[row, candidate_tracks[row] >= 0]
                if len(np.unique(valid_row)) != len(valid_row):
                    raise ValueError("fixed candidate tracks must be unique per point")
            if candidate_null is not None:
                candidate_null = np.asarray(
                    candidate_null, dtype=np.float32
                ).reshape(-1)
                if candidate_null.shape != (xy.shape[0],):
                    raise ValueError(
                        "candidate null probabilities must have shape (N,)"
                    )
                if np.any(~np.isfinite(candidate_scores)) or np.any(
                    ~np.isfinite(candidate_null)
                ):
                    raise ValueError("candidate posterior probabilities must be finite")
                if np.any((candidate_scores < 0.0) | (candidate_scores > 1.0)) or np.any(
                    (candidate_null < 0.0) | (candidate_null > 1.0)
                ):
                    raise ValueError("candidate posterior probabilities must be in [0, 1]")
                if np.any(np.abs(candidate_scores[~valid_candidates]) > 1e-6):
                    raise ValueError("invalid candidate columns must have zero probability")
                candidate_mass = np.sum(
                    np.where(valid_candidates, candidate_scores, 0.0), axis=1
                )
                if np.any(np.abs(candidate_mass + candidate_null - 1.0) > 1e-4):
                    raise ValueError(
                        "candidate and null posterior probabilities must sum to one"
                    )
        elif candidate_null is not None:
            raise ValueError("candidate null probabilities require fixed candidates")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "descriptors", _normalize_rows(descriptors))
        object.__setattr__(self, "descriptor_reference_scores", reference)
        object.__setattr__(self, "source_row_indices", source_rows)
        object.__setattr__(self, "candidate_track_ids", candidate_tracks)
        object.__setattr__(self, "candidate_descriptor_scores", candidate_scores)
        object.__setattr__(self, "candidate_null_probabilities", candidate_null)

    def __len__(self) -> int:
        return int(self.xy.shape[0])

    def subset(self, rows: np.ndarray) -> "IndependentVerificationPoints":
        indices = np.asarray(rows)
        return IndependentVerificationPoints(
            xy=self.xy[indices],
            descriptors=self.descriptors[indices],
            descriptor_reference_scores=self.descriptor_reference_scores[indices],
            source_row_indices=self.source_row_indices[indices],
            candidate_track_ids=(
                None
                if self.candidate_track_ids is None
                else self.candidate_track_ids[indices]
            ),
            candidate_descriptor_scores=(
                None
                if self.candidate_descriptor_scores is None
                else self.candidate_descriptor_scores[indices]
            ),
            candidate_null_probabilities=(
                None
                if self.candidate_null_probabilities is None
                else self.candidate_null_probabilities[indices]
            ),
        )


def _mixed_uint64(values: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic SplitMix64 finalizer used for artifact-stable folds."""

    values_u64 = np.asarray(values, dtype=np.int64).view(np.uint64).copy()
    values_u64 += np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    values_u64 = (values_u64 ^ (values_u64 >> np.uint64(30))) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values_u64 = (values_u64 ^ (values_u64 >> np.uint64(27))) * np.uint64(
        0x94D049BB133111EB
    )
    return values_u64 ^ (values_u64 >> np.uint64(31))


def deterministic_identity_folds(
    identity_ids: np.ndarray,
    *,
    fold_count: int,
    seed: int = 0,
) -> np.ndarray:
    """Assign duplicate physical identities to the same deterministic fold."""

    identities = np.asarray(identity_ids, dtype=np.int64).reshape(-1)
    if int(fold_count) < 2:
        raise ValueError("identity cross-fit requires at least two folds")
    unique, inverse = np.unique(identities, return_inverse=True)
    unique_folds = (
        _mixed_uint64(unique, int(seed)) % np.uint64(int(fold_count))
    ).astype(np.int64)
    return unique_folds[inverse]


def spatially_balanced_point_folds(
    points: IndependentVerificationPoints,
    *,
    image_width: int,
    image_height: int,
    fold_count: int = 3,
    grid_rows: int = 4,
    grid_cols: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Split fixed query points while balancing every image grid cell."""

    if int(fold_count) < 2:
        raise ValueError("point cross-fit requires at least two folds")
    if int(image_width) <= 0 or int(image_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if int(grid_rows) <= 0 or int(grid_cols) <= 0:
        raise ValueError("point cross-fit grid dimensions must be positive")
    normalized_x = np.clip(points.xy[:, 0] / float(image_width), 0.0, 1.0 - 1e-9)
    normalized_y = np.clip(points.xy[:, 1] / float(image_height), 0.0, 1.0 - 1e-9)
    cells = (
        np.floor(normalized_y * int(grid_rows)).astype(np.int64)
        * int(grid_cols)
        + np.floor(normalized_x * int(grid_cols)).astype(np.int64)
    )
    folds = np.full((len(points),), -1, dtype=np.int64)
    row_hashes = _mixed_uint64(points.source_row_indices, int(seed))
    for cell in np.unique(cells):
        members = np.flatnonzero(cells == int(cell))
        order = members[np.argsort(row_hashes[members], kind="mergesort")]
        offset = int(
            _mixed_uint64(np.asarray([cell], dtype=np.int64), int(seed) + 1)[0]
            % np.uint64(int(fold_count))
        )
        folds[order] = (
            np.arange(len(order), dtype=np.int64) + offset
        ) % int(fold_count)
    if np.any(folds < 0):
        raise RuntimeError("point cross-fit left unassigned rows")
    return folds


@dataclass(frozen=True)
class LandmarkObservationViewIndex:
    """Sparse observation rays aligned to rows of a unique-track landmark bank."""

    landmark_row_indices: np.ndarray
    viewing_rays: np.ndarray
    landmark_count: int
    ray_convention: str = "camera_to_landmark_world"

    def __post_init__(self) -> None:
        rows = np.asarray(self.landmark_row_indices, dtype=np.int64).reshape(-1)
        rays = np.asarray(self.viewing_rays, dtype=np.float32).reshape(-1, 3)
        if rows.shape[0] != rays.shape[0]:
            raise ValueError("one landmark row is required per viewing ray")
        if int(self.landmark_count) < 0:
            raise ValueError("landmark_count must be non-negative")
        if np.any((rows < 0) | (rows >= int(self.landmark_count))):
            raise ValueError("observation view index contains an invalid landmark row")
        if str(self.ray_convention) != "camera_to_landmark_world":
            raise ValueError("unsupported observation ray convention")
        if np.any(~np.isfinite(rays)):
            raise ValueError("observation viewing rays must be finite")
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        if np.any(norms <= 1e-8):
            raise ValueError("observation viewing rays must be non-zero")
        object.__setattr__(self, "landmark_row_indices", rows)
        object.__setattr__(self, "viewing_rays", rays / norms)
        order = np.argsort(rows, kind="mergesort")
        sorted_rows = rows[order]
        counts = np.bincount(sorted_rows, minlength=int(self.landmark_count))
        offsets = np.concatenate(
            [np.zeros((1,), dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
        )
        object.__setattr__(self, "_sorted_viewing_rays", (rays / norms)[order])
        object.__setattr__(self, "_landmark_offsets", offsets)

    @classmethod
    def from_track_observations(
        cls,
        bank_track_ids: np.ndarray,
        observation_track_ids: np.ndarray,
        observation_viewing_rays: np.ndarray,
    ) -> "LandmarkObservationViewIndex":
        tracks = np.asarray(bank_track_ids, dtype=np.int64).reshape(-1)
        if len(np.unique(tracks)) != len(tracks):
            raise ValueError(
                "view index construction requires one bank row per physical track"
            )
        observation_tracks = np.asarray(
            observation_track_ids, dtype=np.int64
        ).reshape(-1)
        rays = np.asarray(observation_viewing_rays, dtype=np.float32).reshape(-1, 3)
        if observation_tracks.shape[0] != rays.shape[0]:
            raise ValueError("observation track IDs and rays must be aligned")
        if tracks.size == 0:
            return cls(
                landmark_row_indices=np.zeros((0,), dtype=np.int64),
                viewing_rays=np.zeros((0, 3), dtype=np.float32),
                landmark_count=0,
            )
        order = np.argsort(tracks, kind="mergesort")
        sorted_tracks = tracks[order]
        positions = np.searchsorted(sorted_tracks, observation_tracks)
        clipped = np.minimum(positions, len(sorted_tracks) - 1)
        found = (
            (positions < len(sorted_tracks))
            & (sorted_tracks[clipped] == observation_tracks)
        )
        return cls(
            landmark_row_indices=order[positions[found]],
            viewing_rays=rays[found],
            landmark_count=int(len(tracks)),
        )

    def maximum_view_cosines(self, query_viewing_rays: np.ndarray) -> np.ndarray:
        query_rays = np.asarray(query_viewing_rays, dtype=np.float32).reshape(-1, 3)
        if query_rays.shape[0] != int(self.landmark_count):
            raise ValueError("query viewing rays must contain one row per landmark")
        query_rays = query_rays / np.maximum(
            np.linalg.norm(query_rays, axis=1, keepdims=True), 1e-8
        )
        cosines = np.sum(
            query_rays[self.landmark_row_indices] * self.viewing_rays, axis=1
        )
        maximum = np.full((int(self.landmark_count),), -1.0, dtype=np.float32)
        np.maximum.at(maximum, self.landmark_row_indices, cosines)
        return maximum

    def minimum_view_angles_deg(
        self, query_viewing_rays: np.ndarray
    ) -> np.ndarray:
        maximum = self.maximum_view_cosines(query_viewing_rays)
        angles = np.full(maximum.shape, np.inf, dtype=np.float32)
        valid = maximum >= -1.0
        angles[valid] = np.degrees(
            np.arccos(np.clip(maximum[valid], -1.0, 1.0))
        ).astype(np.float32)
        return angles

    def minimum_view_angles_deg_for_rows(
        self,
        landmark_rows: np.ndarray,
        query_viewing_rays: np.ndarray,
    ) -> np.ndarray:
        """Return the full-bank view gate restricted to immutable row IDs."""

        rows = np.asarray(landmark_rows, dtype=np.int64).reshape(-1)
        query_rays = np.asarray(query_viewing_rays, dtype=np.float32).reshape(-1, 3)
        if query_rays.shape != (len(rows), 3):
            raise ValueError("subset query rays must align with landmark rows")
        if np.any((rows < 0) | (rows >= int(self.landmark_count))):
            raise ValueError("subset view rows are outside the landmark bank")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("subset view rows must be unique")
        query_rays = query_rays / np.maximum(
            np.linalg.norm(query_rays, axis=1, keepdims=True), 1e-8
        )
        starts = self._landmark_offsets[rows]
        counts = self._landmark_offsets[rows + 1] - starts
        output = np.full((len(rows),), np.inf, dtype=np.float32)
        total = int(np.sum(counts))
        if total == 0:
            return output
        local_rows = np.repeat(np.arange(len(rows), dtype=np.int64), counts)
        local_offsets = np.repeat(
            np.cumsum(counts, dtype=np.int64) - counts, counts
        )
        observation_rows = (
            np.arange(total, dtype=np.int64)
            - local_offsets
            + np.repeat(starts, counts)
        )
        cosines = np.sum(
            query_rays[local_rows] * self._sorted_viewing_rays[observation_rows],
            axis=1,
        )
        maximum = np.full((len(rows),), -1.0, dtype=np.float32)
        np.maximum.at(maximum, local_rows, cosines)
        valid = counts > 0
        output[valid] = np.degrees(
            np.arccos(np.clip(maximum[valid], -1.0, 1.0))
        ).astype(np.float32)
        return output


@dataclass(frozen=True)
class LandmarkPrototypeViewIndex:
    """One descriptor prototype and one observation-view distribution per row."""

    track_ids: np.ndarray
    prototype_ids: np.ndarray
    mean_viewing_rays: np.ndarray
    viewing_ray_concentrations: np.ndarray
    viewing_angle_p90_deg: np.ndarray
    valid_mask: np.ndarray
    ray_convention: str = "camera_to_landmark_world"

    def __post_init__(self) -> None:
        track_ids = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        prototype_ids = np.asarray(self.prototype_ids, dtype=np.int64).reshape(-1)
        mean_rays = np.asarray(self.mean_viewing_rays, dtype=np.float32).reshape(-1, 3)
        concentrations = np.asarray(
            self.viewing_ray_concentrations, dtype=np.float32
        ).reshape(-1)
        p90 = np.asarray(self.viewing_angle_p90_deg, dtype=np.float32).reshape(-1)
        valid = np.asarray(self.valid_mask, dtype=bool).reshape(-1)
        count = len(track_ids)
        if not (
            prototype_ids.shape == concentrations.shape == p90.shape == valid.shape == (count,)
            and mean_rays.shape == (count, 3)
        ):
            raise ValueError("prototype view geometry arrays are not row-aligned")
        if str(self.ray_convention) != "camera_to_landmark_world":
            raise ValueError("unsupported prototype ray convention")
        if np.any(~np.isfinite(concentrations)) or np.any(
            (concentrations < 0.0) | (concentrations > 1.0)
        ):
            raise ValueError("prototype view concentrations must be in [0, 1]")
        if np.any(~np.isfinite(p90)) or np.any((p90 < 0.0) | (p90 > 180.0)):
            raise ValueError("prototype viewing angle p90 must be in [0, 180]")
        if np.any(~np.isfinite(mean_rays[valid])):
            raise ValueError("valid prototype mean viewing rays must be finite")
        norms = np.linalg.norm(mean_rays[valid], axis=1)
        if np.any(norms <= 1e-8):
            raise ValueError("valid prototype mean viewing rays must be non-zero")
        normalized = mean_rays.copy()
        normalized[valid] /= norms[:, None]
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "prototype_ids", prototype_ids)
        object.__setattr__(self, "mean_viewing_rays", normalized)
        object.__setattr__(self, "viewing_ray_concentrations", concentrations)
        object.__setattr__(self, "viewing_angle_p90_deg", p90)
        object.__setattr__(self, "valid_mask", valid)

    @property
    def landmark_count(self) -> int:
        return int(len(self.track_ids))

    def minimum_view_angles_deg(
        self, query_viewing_rays: np.ndarray
    ) -> np.ndarray:
        query_rays = np.asarray(query_viewing_rays, dtype=np.float32).reshape(-1, 3)
        if query_rays.shape != self.mean_viewing_rays.shape:
            raise ValueError("query viewing rays and prototype rows differ")
        query_rays = query_rays / np.maximum(
            np.linalg.norm(query_rays, axis=1, keepdims=True), 1e-8
        )
        output = np.full((self.landmark_count,), np.inf, dtype=np.float32)
        cosine = np.sum(
            query_rays[self.valid_mask]
            * self.mean_viewing_rays[self.valid_mask],
            axis=1,
        )
        center_angle = np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )
        output[self.valid_mask] = np.maximum(
            center_angle - self.viewing_angle_p90_deg[self.valid_mask], 0.0
        )
        return output

    def minimum_view_angles_deg_for_rows(
        self,
        landmark_rows: np.ndarray,
        query_viewing_rays: np.ndarray,
    ) -> np.ndarray:
        rows = np.asarray(landmark_rows, dtype=np.int64).reshape(-1)
        query_rays = np.asarray(query_viewing_rays, dtype=np.float32).reshape(-1, 3)
        if query_rays.shape != (len(rows), 3):
            raise ValueError("subset query rays must align with landmark rows")
        if np.any((rows < 0) | (rows >= self.landmark_count)):
            raise ValueError("subset view rows are outside the landmark bank")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("subset view rows must be unique")
        query_rays = query_rays / np.maximum(
            np.linalg.norm(query_rays, axis=1, keepdims=True), 1e-8
        )
        output = np.full((len(rows),), np.inf, dtype=np.float32)
        valid = self.valid_mask[rows]
        cosine = np.sum(
            query_rays[valid] * self.mean_viewing_rays[rows[valid]], axis=1
        )
        center_angle = np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )
        output[valid] = np.maximum(
            center_angle - self.viewing_angle_p90_deg[rows[valid]], 0.0
        )
        return output


def load_landmark_prototype_view_index_npz(
    path: Path,
    landmark_index: LandmarkMapIndex,
    *,
    expected_descriptor_space_id: str | None = None,
) -> tuple[LandmarkPrototypeViewIndex, dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "track_ids",
            "prototype_ids",
            "mean_viewing_rays",
            "viewing_ray_concentrations",
            "viewing_angle_p90_deg",
            "view_geometry_valid",
            "metadata_json",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"prototype view sidecar is missing fields: {missing}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        index = LandmarkPrototypeViewIndex(
            track_ids=payload["track_ids"],
            prototype_ids=payload["prototype_ids"],
            mean_viewing_rays=payload["mean_viewing_rays"],
            viewing_ray_concentrations=payload["viewing_ray_concentrations"],
            viewing_angle_p90_deg=payload["viewing_angle_p90_deg"],
            valid_mask=payload["view_geometry_valid"],
            ray_convention=str(
                metadata.get("ray_convention", "camera_to_landmark_world")
            ),
        )
    if not np.array_equal(index.track_ids, landmark_index.track_ids):
        raise ValueError("prototype view sidecar track rows differ from landmark bank")
    if not np.array_equal(index.prototype_ids, landmark_index.prototype_ids):
        raise ValueError("prototype view sidecar IDs differ from landmark bank")
    if expected_descriptor_space_id is not None and str(
        metadata.get("descriptor_space_id")
    ) != str(expected_descriptor_space_id):
        raise ValueError("prototype view sidecar descriptor space differs")
    return index, metadata


@dataclass(frozen=True)
class IndependentLandmarkPoseLikelihoodConfig:
    nearest_landmarks: int = 4
    maximum_reprojection_distance_px: float = 8.0
    spatial_sigma_px: float = 3.0
    descriptor_temperature: float = 0.04
    outlier_likelihood: float = 0.01
    minimum_observation_count: int = 2
    maximum_view_angle_deg: float | None = 15.0
    kdtree_workers: int = 1
    candidate_mode: str = "pose_local_knn"
    fixed_candidate_prior_source: str = "prototype_similarity"

    def __post_init__(self) -> None:
        if int(self.nearest_landmarks) <= 0:
            raise ValueError("nearest_landmarks must be positive")
        if float(self.maximum_reprojection_distance_px) <= 0.0:
            raise ValueError("maximum_reprojection_distance_px must be positive")
        if float(self.spatial_sigma_px) <= 0.0:
            raise ValueError("spatial_sigma_px must be positive")
        if float(self.descriptor_temperature) <= 0.0:
            raise ValueError("descriptor_temperature must be positive")
        if not 0.0 < float(self.outlier_likelihood) < 1.0:
            raise ValueError("outlier_likelihood must be in (0, 1)")
        if int(self.minimum_observation_count) <= 0:
            raise ValueError("minimum_observation_count must be positive")
        if self.maximum_view_angle_deg is not None and not (
            0.0 < float(self.maximum_view_angle_deg) <= 180.0
        ):
            raise ValueError("maximum_view_angle_deg must be in (0, 180]")
        if int(self.kdtree_workers) == 0 or int(self.kdtree_workers) < -1:
            raise ValueError("kdtree_workers must be -1 or a positive integer")
        if str(self.candidate_mode) not in {"pose_local_knn", "fixed_global_topl"}:
            raise ValueError("unsupported independent pose candidate mode")
        if str(self.fixed_candidate_prior_source) not in {
            "prototype_similarity",
            "prototype_similarity_with_learned_null",
            "coarse_score",
            "learned_probability",
        }:
            raise ValueError("unsupported fixed candidate prior source")

    def to_dict(self) -> dict[str, object]:
        return {
            "nearest_landmarks": int(self.nearest_landmarks),
            "maximum_reprojection_distance_px": float(
                self.maximum_reprojection_distance_px
            ),
            "spatial_sigma_px": float(self.spatial_sigma_px),
            "descriptor_temperature": float(self.descriptor_temperature),
            "outlier_likelihood": float(self.outlier_likelihood),
            "minimum_observation_count": int(self.minimum_observation_count),
            "maximum_view_angle_deg": (
                None
                if self.maximum_view_angle_deg is None
                else float(self.maximum_view_angle_deg)
            ),
            "kdtree_workers": int(self.kdtree_workers),
            "candidate_mode": str(self.candidate_mode),
            "fixed_candidate_prior_source": str(
                self.fixed_candidate_prior_source
            ),
        }


@dataclass(frozen=True)
class IndependentLandmarkPoseLikelihoodScore:
    log_likelihood_sum: float
    log_likelihood_mean: float
    log_likelihood_median: float
    log_likelihood_trimmed_mean_10: float
    log_likelihood_worst_quartile_mean: float
    log_likelihood_lcb95: float
    spatial_median_of_means_2x2: float
    verification_point_count: int
    effective_point_count: int
    evidence_coverage: float
    projected_landmark_count: int
    view_eligible_landmark_count: int
    point_evidence: np.ndarray

    def statistic(self, name: str) -> float:
        fields = {
            "mean": "log_likelihood_mean",
            "median": "log_likelihood_median",
            "trimmed_mean_10": "log_likelihood_trimmed_mean_10",
            "worst_quartile_mean": "log_likelihood_worst_quartile_mean",
            "lcb95": "log_likelihood_lcb95",
            "spatial_median_of_means_2x2": "spatial_median_of_means_2x2",
        }
        field = fields.get(str(name))
        if field is None:
            raise ValueError(f"unsupported pose likelihood statistic: {name}")
        return float(getattr(self, field))


def _fixed_group_log_likelihood_statistics(
    log_likelihood: np.ndarray,
    xy: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(log_likelihood, dtype=np.float64).reshape(-1)
    coordinates = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(values) == 0:
        return {
            "log_likelihood_median": float("-inf"),
            "log_likelihood_trimmed_mean_10": float("-inf"),
            "log_likelihood_worst_quartile_mean": float("-inf"),
            "log_likelihood_lcb95": float("-inf"),
            "spatial_median_of_means_2x2": float("-inf"),
        }
    if len(coordinates) != len(values):
        raise ValueError("fixed likelihood values and query coordinates differ")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(coordinates)):
        raise ValueError("fixed likelihood statistics require finite inputs")
    ordered = np.sort(values)
    count = len(ordered)
    trim_count = int(np.floor(0.1 * count))
    trimmed = (
        ordered[trim_count : count - trim_count]
        if trim_count > 0 and 2 * trim_count < count
        else ordered
    )
    worst_count = max(1, int(np.ceil(0.25 * count)))
    standard_error = (
        float(np.std(ordered, ddof=1) / np.sqrt(float(count)))
        if count > 1
        else 0.0
    )
    median_xy = np.median(coordinates, axis=0)
    cells = (
        (coordinates[:, 0] > median_xy[0]).astype(np.int64)
        + 2 * (coordinates[:, 1] > median_xy[1]).astype(np.int64)
    )
    cell_means = np.asarray(
        [
            np.mean(values[cells == cell])
            for cell in range(4)
            if np.any(cells == cell)
        ],
        dtype=np.float64,
    )
    mean = float(np.mean(ordered))
    return {
        "log_likelihood_median": float(np.median(ordered)),
        "log_likelihood_trimmed_mean_10": float(np.mean(trimmed)),
        "log_likelihood_worst_quartile_mean": float(
            np.mean(ordered[:worst_count])
        ),
        "log_likelihood_lcb95": float(mean - 1.96 * standard_error),
        "spatial_median_of_means_2x2": float(np.median(cell_means)),
    }


@dataclass(frozen=True)
class IndependentPoseCorrespondences:
    point_indices: np.ndarray
    landmark_row_indices: np.ndarray
    track_ids: np.ndarray
    xy: np.ndarray
    xyz: np.ndarray
    weights: np.ndarray
    descriptor_similarities: np.ndarray
    initial_reprojection_distances_px: np.ndarray

    def __post_init__(self) -> None:
        point_indices = np.asarray(self.point_indices, dtype=np.int64).reshape(-1)
        landmark_rows = np.asarray(
            self.landmark_row_indices, dtype=np.int64
        ).reshape(-1)
        track_ids = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float64).reshape(-1, 2)
        xyz = np.asarray(self.xyz, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        similarities = np.asarray(
            self.descriptor_similarities, dtype=np.float64
        ).reshape(-1)
        distances = np.asarray(
            self.initial_reprojection_distances_px, dtype=np.float64
        ).reshape(-1)
        count = len(point_indices)
        if not (
            landmark_rows.shape
            == track_ids.shape
            == weights.shape
            == similarities.shape
            == distances.shape
            == (count,)
            and xy.shape == (count, 2)
            and xyz.shape == (count, 3)
        ):
            raise ValueError("pose-conditioned correspondences are not aligned")
        if len(np.unique(point_indices)) != count:
            raise ValueError("one correspondence is allowed per query point")
        if len(np.unique(track_ids)) != count:
            raise ValueError("one correspondence is allowed per physical track")
        if np.any(~np.isfinite(xy)) or np.any(~np.isfinite(xyz)):
            raise ValueError("pose-conditioned coordinates must be finite")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("pose-conditioned weights must be finite and positive")
        object.__setattr__(self, "point_indices", point_indices)
        object.__setattr__(self, "landmark_row_indices", landmark_rows)
        object.__setattr__(self, "track_ids", track_ids)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "descriptor_similarities", similarities)
        object.__setattr__(self, "initial_reprojection_distances_px", distances)

    def __len__(self) -> int:
        return int(len(self.point_indices))


@dataclass(frozen=True)
class IndependentPoseRefinementConfig:
    nearest_landmarks: int = 8
    maximum_reprojection_distance_px: float = 12.0
    spatial_sigma_px: float = 4.0
    descriptor_temperature: float = 0.04
    minimum_match_evidence: float = 0.05
    minimum_correspondences: int = 8
    iterations: int = 2
    robust_loss: str = "huber"
    robust_f_scale_px: float = 2.0
    max_nfev: int = 40
    maximum_translation_step_m: float = 0.25
    maximum_rotation_step_deg: float = 3.0
    minimum_fit_log_likelihood_gain: float = 0.0

    def __post_init__(self) -> None:
        if int(self.nearest_landmarks) <= 0:
            raise ValueError("refinement nearest_landmarks must be positive")
        if float(self.maximum_reprojection_distance_px) <= 0.0:
            raise ValueError("refinement reprojection radius must be positive")
        if float(self.spatial_sigma_px) <= 0.0:
            raise ValueError("refinement spatial sigma must be positive")
        if float(self.descriptor_temperature) <= 0.0:
            raise ValueError("refinement descriptor temperature must be positive")
        if not 0.0 < float(self.minimum_match_evidence) <= 1.0:
            raise ValueError("minimum match evidence must be in (0, 1]")
        if int(self.minimum_correspondences) < 4:
            raise ValueError("pose refinement requires at least four correspondences")
        if not 0 <= int(self.iterations) <= 5:
            raise ValueError("pose refinement iterations must be in [0, 5]")
        if str(self.robust_loss).lower() not in {
            "linear",
            "soft_l1",
            "huber",
            "cauchy",
            "arctan",
        }:
            raise ValueError("unsupported pose refinement robust loss")
        if float(self.robust_f_scale_px) <= 0.0:
            raise ValueError("pose refinement robust scale must be positive")
        if int(self.max_nfev) <= 0:
            raise ValueError("pose refinement max_nfev must be positive")
        if float(self.maximum_translation_step_m) <= 0.0:
            raise ValueError("maximum pose refinement translation must be positive")
        if float(self.maximum_rotation_step_deg) <= 0.0:
            raise ValueError("maximum pose refinement rotation must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class IndependentPoseRefinementResult:
    success: bool
    pose_w2c: np.ndarray
    accepted_iterations: int
    final_correspondence_count: int
    fit_log_likelihood_before: float
    fit_log_likelihood_after: float
    translation_step_m: float
    rotation_step_deg: float
    used_track_ids: np.ndarray
    failure_reason: str | None = None


class IndependentLandmarkPoseVerifier:
    """Cache immutable bank state and score poses on a fixed held-out set."""

    def __init__(
        self,
        landmark_index: LandmarkMapIndex,
        observation_views: LandmarkObservationViewIndex
        | LandmarkPrototypeViewIndex,
        config: IndependentLandmarkPoseLikelihoodConfig | None = None,
    ) -> None:
        if int(observation_views.landmark_count) != len(landmark_index):
            raise ValueError("observation view index and landmark bank differ")
        self.landmark_index = landmark_index
        self.observation_views = observation_views
        self.config = config or IndependentLandmarkPoseLikelihoodConfig()
        self._features = _normalize_rows(landmark_index.features)
        self._base_eligible = np.asarray(
            landmark_index.observation_counts >= int(self.config.minimum_observation_count),
            dtype=bool,
        )
        track_order = np.argsort(
            np.asarray(landmark_index.track_ids, dtype=np.int64), kind="mergesort"
        )
        self._track_order = track_order
        self._sorted_track_ids = np.asarray(
            landmark_index.track_ids, dtype=np.int64
        )[track_order]
        self._fixed_candidate_cache: dict[tuple[int, int], object] = {}

    def _static_eligible_mask(
        self, eligible_landmark_mask: np.ndarray | None
    ) -> np.ndarray:
        eligible = (
            self._base_eligible.copy()
            if eligible_landmark_mask is None
            else np.asarray(eligible_landmark_mask, dtype=bool).reshape(-1).copy()
        )
        if eligible.shape != (len(self.landmark_index),):
            raise ValueError("eligible_landmark_mask must have shape (M,)")
        eligible &= self._base_eligible
        return eligible

    def _rows_for_track(self, track_id: int) -> np.ndarray:
        left = int(np.searchsorted(self._sorted_track_ids, int(track_id), side="left"))
        right = int(
            np.searchsorted(self._sorted_track_ids, int(track_id), side="right")
        )
        return self._track_order[left:right]

    def _prepare_fixed_candidates(
        self,
        points: IndependentVerificationPoints,
        eligible_landmark_mask: np.ndarray | None,
    ) -> tuple[tuple[tuple[object, ...], ...], ...]:
        if points.candidate_track_ids is None:
            raise ValueError(
                "fixed_global_topl scoring requires fixed candidate tracks"
            )
        mask_key = 0 if eligible_landmark_mask is None else id(eligible_landmark_mask)
        cache_key = (id(points), mask_key)
        cached = self._fixed_candidate_cache.get(cache_key)
        if cached is not None:
            cached_points, cached_mask, cached_prepared = cached  # type: ignore[misc]
            if cached_points is points and cached_mask is eligible_landmark_mask:
                return cached_prepared  # type: ignore[return-value]

        static_eligible = self._static_eligible_mask(eligible_landmark_mask)
        temperature = float(self.config.descriptor_temperature)
        prior_source = str(self.config.fixed_candidate_prior_source)
        learned_probability = prior_source == "learned_probability"
        learned_null = prior_source in {
            "learned_probability",
            "prototype_similarity_with_learned_null",
        }
        if learned_null and points.candidate_null_probabilities is None:
            raise ValueError(
                "learned candidate availability requires explicit null probabilities"
            )
        prepared_points: list[tuple[tuple[object, ...], ...]] = []
        for point_index in range(len(points)):
            raw_entries: list[tuple[int, np.ndarray, np.ndarray, float]] = []
            for candidate_column, track_id in enumerate(
                points.candidate_track_ids[point_index].tolist()
            ):
                if int(track_id) < 0:
                    continue
                all_rows = self._rows_for_track(int(track_id))
                if all_rows.size:
                    similarities = (
                        self._features[all_rows] @ points.descriptors[point_index]
                    ).astype(np.float64)
                    maximum_similarity = float(np.max(similarities))
                    all_view_evidence = np.exp(
                        np.minimum(
                            (similarities - maximum_similarity) / temperature,
                            0.0,
                        )
                    )
                    retained = static_eligible[all_rows]
                    rows = all_rows[retained]
                    relative_view_evidence = all_view_evidence[retained]
                else:
                    if prior_source in {
                        "prototype_similarity",
                        "prototype_similarity_with_learned_null",
                    }:
                        raise ValueError(
                            f"fixed candidate track {int(track_id)} is absent from the bank"
                        )
                    maximum_similarity = float("-inf")
                    rows = np.zeros((0,), dtype=np.int64)
                    relative_view_evidence = np.zeros((0,), dtype=np.float64)
                prior_logit = (
                    maximum_similarity
                    if prior_source
                    in {
                        "prototype_similarity",
                        "prototype_similarity_with_learned_null",
                    }
                    else float(
                        points.candidate_descriptor_scores[
                            point_index, candidate_column
                        ]
                    )
                )
                raw_entries.append(
                    (
                        int(track_id),
                        rows.astype(np.int64, copy=False),
                        relative_view_evidence,
                        prior_logit,
                    )
                )
            if not raw_entries:
                prepared_points.append(tuple())
                continue
            if learned_probability:
                prepared_points.append(
                    tuple(
                        (
                            entry[0],
                            entry[1],
                            entry[2],
                            float(entry[3]),
                            float(entry[3]),
                        )
                        for entry in raw_entries
                    )
                )
                continue
            logits = np.asarray([entry[3] for entry in raw_entries], dtype=np.float64)
            relative_priors = np.exp(
                np.minimum((logits - float(np.max(logits))) / temperature, 0.0)
            )
            normalized_priors = relative_priors / max(
                float(np.sum(relative_priors)), 1e-12
            )
            if prior_source == "prototype_similarity_with_learned_null":
                normalized_priors *= 1.0 - float(
                    points.candidate_null_probabilities[point_index]
                )
            prepared_points.append(
                tuple(
                    (
                        entry[0],
                        entry[1],
                        entry[2],
                        float(relative_prior),
                        float(normalized_prior),
                    )
                    for entry, relative_prior, normalized_prior in zip(
                        raw_entries, relative_priors, normalized_priors
                    )
                )
            )
        prepared = tuple(prepared_points)
        # Keep object references with the cache entry so CPython ID reuse cannot
        # alias candidates from a completed query onto a later query.
        self._fixed_candidate_cache[cache_key] = (
            points,
            eligible_landmark_mask,
            prepared,
        )
        return prepared

    def _fixed_candidate_point_evidence(
        self,
        projected: np.ndarray,
        pose_eligible: np.ndarray,
        points: IndependentVerificationPoints,
        eligible_landmark_mask: np.ndarray | None,
    ) -> np.ndarray:
        prepared = self._prepare_fixed_candidates(points, eligible_landmark_mask)
        evidence = np.zeros((len(points),), dtype=np.float64)
        radius = float(self.config.maximum_reprojection_distance_px)
        sigma = float(self.config.spatial_sigma_px)
        for point_index, candidates in enumerate(prepared):
            point_evidence = 0.0
            for _track_id, rows, view_evidence, _relative_prior, prior in candidates:
                rows_array = np.asarray(rows, dtype=np.int64)
                visible = pose_eligible[rows_array]
                if not np.any(visible):
                    continue
                visible_rows = rows_array[visible]
                distances = np.linalg.norm(
                    projected[visible_rows] - points.xy[point_index], axis=1
                )
                within = distances <= radius
                if not np.any(within):
                    continue
                spatial = np.exp(-0.5 * np.square(distances[within] / sigma))
                view_values = np.asarray(view_evidence, dtype=np.float64)[visible][
                    within
                ]
                point_evidence += float(prior) * float(
                    np.max(view_values * spatial)
                )
            evidence[point_index] = min(max(point_evidence, 0.0), 1.0)
        return evidence

    def _fixed_candidate_landmark_rows(
        self,
        points: IndependentVerificationPoints,
        eligible_landmark_mask: np.ndarray | None,
    ) -> np.ndarray:
        prepared = self._prepare_fixed_candidates(points, eligible_landmark_mask)
        row_blocks = [
            np.asarray(candidate[1], dtype=np.int64)
            for candidates in prepared
            for candidate in candidates
            if len(np.asarray(candidate[1]).reshape(-1))
        ]
        if not row_blocks:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.concatenate(row_blocks))

    def eligible_mask_excluding_tracks(
        self, excluded_track_ids: np.ndarray | None
    ) -> np.ndarray:
        eligible = self._base_eligible.copy()
        if excluded_track_ids is not None:
            excluded = np.asarray(excluded_track_ids, dtype=np.int64).reshape(-1)
            if excluded.size:
                eligible &= ~np.isin(self.landmark_index.track_ids, excluded)
        return eligible

    def _projected_eligible_landmarks(
        self,
        pose_w2c: np.ndarray,
        camera: ColmapCamera,
        eligible_landmark_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
        if np.any(~np.isfinite(pose)):
            raise ValueError("pose must be finite")
        eligible = self._static_eligible_mask(eligible_landmark_mask)

        camera_points = (
            pose[:3, :3] @ self.landmark_index.xyz.T
        ).T + pose[:3, 3]
        projected = project_world_to_image(self.landmark_index.xyz, pose, camera)
        eligible &= camera_points[:, 2] > 0.0
        eligible &= (
            (projected[:, 0] >= 0.0)
            & (projected[:, 0] < float(camera.width))
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < float(camera.height))
        )

        camera_center = -(pose[:3, :3].T @ pose[:3, 3])
        query_viewing_rays = self.landmark_index.xyz - camera_center[None, :]
        minimum_view_angles = self.observation_views.minimum_view_angles_deg(
            query_viewing_rays
        )
        if self.config.maximum_view_angle_deg is not None:
            eligible &= minimum_view_angles <= float(
                self.config.maximum_view_angle_deg
            )
        else:
            eligible &= np.isfinite(minimum_view_angles)
        return projected, eligible

    def _projected_fixed_candidate_landmarks(
        self,
        pose_w2c: np.ndarray,
        camera: ColmapCamera,
        points: IndependentVerificationPoints,
        eligible_landmark_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project only pose-independent top-L rows used by this query role."""

        pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
        if np.any(~np.isfinite(pose)):
            raise ValueError("pose must be finite")
        rows = self._fixed_candidate_landmark_rows(
            points, eligible_landmark_mask
        )
        projected = np.full((len(self.landmark_index), 2), np.nan, dtype=np.float64)
        eligible = np.zeros((len(self.landmark_index),), dtype=bool)
        if rows.size == 0:
            return projected, eligible, rows

        xyz = np.asarray(self.landmark_index.xyz[rows], dtype=np.float64)
        selected_projected = project_world_to_image(xyz, pose, camera)
        camera_points = xyz @ pose[:3, :3].T + pose[:3, 3]
        selected_eligible = camera_points[:, 2] > 0.0
        selected_eligible &= (
            (selected_projected[:, 0] >= 0.0)
            & (selected_projected[:, 0] < float(camera.width))
            & (selected_projected[:, 1] >= 0.0)
            & (selected_projected[:, 1] < float(camera.height))
        )
        camera_center = -(pose[:3, :3].T @ pose[:3, 3])
        query_viewing_rays = xyz - camera_center[None, :]
        minimum_view_angles = self.observation_views.minimum_view_angles_deg_for_rows(
            rows, query_viewing_rays
        )
        if self.config.maximum_view_angle_deg is not None:
            selected_eligible &= minimum_view_angles <= float(
                self.config.maximum_view_angle_deg
            )
        else:
            selected_eligible &= np.isfinite(minimum_view_angles)
        projected[rows] = selected_projected
        eligible[rows] = selected_eligible
        return projected, eligible, rows

    def pose_conditioned_correspondences(
        self,
        pose_w2c: np.ndarray,
        camera: ColmapCamera,
        points: IndependentVerificationPoints,
        refinement_config: IndependentPoseRefinementConfig,
        *,
        eligible_landmark_mask: np.ndarray | None = None,
    ) -> IndependentPoseCorrespondences:
        """Resolve a one-query/one-track assignment inside a local pose basin."""

        if points.descriptors.shape[1] != self._features.shape[1]:
            raise ValueError("query and landmark descriptor dimensions differ")
        projected, eligible = self._projected_eligible_landmarks(
            pose_w2c, camera, eligible_landmark_mask
        )
        landmark_rows = np.flatnonzero(eligible)
        empty = IndependentPoseCorrespondences(
            point_indices=np.zeros((0,), dtype=np.int64),
            landmark_row_indices=np.zeros((0,), dtype=np.int64),
            track_ids=np.zeros((0,), dtype=np.int64),
            xy=np.zeros((0, 2), dtype=np.float64),
            xyz=np.zeros((0, 3), dtype=np.float64),
            weights=np.zeros((0,), dtype=np.float64),
            descriptor_similarities=np.zeros((0,), dtype=np.float64),
            initial_reprojection_distances_px=np.zeros((0,), dtype=np.float64),
        )
        if len(points) == 0 or landmark_rows.size == 0:
            return empty
        try:
            from scipy.spatial import cKDTree
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("SciPy is required for landmark pose refinement") from exc
        tree = cKDTree(projected[landmark_rows])
        distances, neighbors = tree.query(
            points.xy,
            k=int(refinement_config.nearest_landmarks),
            distance_upper_bound=float(
                refinement_config.maximum_reprojection_distance_px
            ),
            workers=int(self.config.kdtree_workers),
        )
        if distances.ndim == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        point_grid = np.broadcast_to(
            np.arange(len(points), dtype=np.int64)[:, None], distances.shape
        )
        valid = np.isfinite(distances) & (neighbors < landmark_rows.size)
        if not np.any(valid):
            return empty
        point_indices = point_grid[valid]
        matched_rows = landmark_rows[neighbors[valid]]
        matched_distances = np.asarray(distances[valid], dtype=np.float64)
        similarities = np.sum(
            points.descriptors[point_indices] * self._features[matched_rows], axis=1
        ).astype(np.float64)
        descriptor_evidence = np.exp(
            np.minimum(
                (
                    similarities
                    - points.descriptor_reference_scores[point_indices]
                )
                / float(refinement_config.descriptor_temperature),
                0.0,
            )
        )
        spatial_evidence = np.exp(
            -0.5
            * np.square(
                matched_distances / float(refinement_config.spatial_sigma_px)
            )
        )
        weights = descriptor_evidence * spatial_evidence
        keep = weights >= float(refinement_config.minimum_match_evidence)
        if not np.any(keep):
            return empty
        point_indices = point_indices[keep]
        matched_rows = matched_rows[keep]
        matched_distances = matched_distances[keep]
        similarities = similarities[keep]
        weights = weights[keep]
        tracks = np.asarray(self.landmark_index.track_ids, dtype=np.int64)[
            matched_rows
        ]

        # Greedy maximum-weight matching is deterministic and prevents duplicate
        # descriptor prototypes of one physical track from inflating support.
        order = np.lexsort(
            (
                matched_rows,
                tracks,
                point_indices,
                -weights,
            )
        )
        used_points: set[int] = set()
        used_tracks: set[int] = set()
        selected: list[int] = []
        for edge in order.tolist():
            point = int(point_indices[edge])
            track = int(tracks[edge])
            if point in used_points or track in used_tracks:
                continue
            used_points.add(point)
            used_tracks.add(track)
            selected.append(int(edge))
        selected_array = np.asarray(selected, dtype=np.int64)
        return IndependentPoseCorrespondences(
            point_indices=point_indices[selected_array],
            landmark_row_indices=matched_rows[selected_array],
            track_ids=tracks[selected_array],
            xy=points.xy[point_indices[selected_array]],
            xyz=self.landmark_index.xyz[matched_rows[selected_array]],
            weights=weights[selected_array],
            descriptor_similarities=similarities[selected_array],
            initial_reprojection_distances_px=matched_distances[selected_array],
        )

    @staticmethod
    def _pose_step(
        initial_pose_w2c: np.ndarray, proposed_pose_w2c: np.ndarray
    ) -> tuple[float, float]:
        initial = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
        proposed = np.asarray(proposed_pose_w2c, dtype=np.float64).reshape(4, 4)
        initial_center = -(initial[:3, :3].T @ initial[:3, 3])
        proposed_center = -(proposed[:3, :3].T @ proposed[:3, 3])
        translation = float(np.linalg.norm(initial_center - proposed_center))
        relative_rotation = proposed[:3, :3] @ initial[:3, :3].T
        cosine = float(
            np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        return translation, float(np.degrees(np.arccos(cosine)))

    @staticmethod
    def _optimize_correspondences(
        correspondences: IndependentPoseCorrespondences,
        initial_pose_w2c: np.ndarray,
        camera: ColmapCamera,
        config: IndependentPoseRefinementConfig,
    ) -> np.ndarray | None:
        try:
            import cv2
            from scipy.optimize import least_squares
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV and SciPy are required for pose refinement") from exc
        pose = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
        rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
        params = np.concatenate([rvec.reshape(3), pose[:3, 3]])
        matrix, distortion = camera_matrix_and_distortion(camera)
        normalized_weights = correspondences.weights / max(
            float(np.mean(correspondences.weights)), 1e-12
        )
        sqrt_weights = np.sqrt(normalized_weights)

        def residuals(value: np.ndarray) -> np.ndarray:
            projected, _ = cv2.projectPoints(
                correspondences.xyz,
                value[:3].reshape(3, 1),
                value[3:6].reshape(3, 1),
                matrix,
                distortion,
            )
            error = projected.reshape(-1, 2) - correspondences.xy
            return (error * sqrt_weights[:, None]).reshape(-1)

        try:
            result = least_squares(
                residuals,
                params,
                loss=str(config.robust_loss).lower(),
                f_scale=float(config.robust_f_scale_px),
                max_nfev=int(config.max_nfev),
                method="trf",
            )
        except Exception:
            return None
        if not bool(result.success) or not np.all(np.isfinite(result.x)):
            return None
        rotation, _jacobian = cv2.Rodrigues(result.x[:3].reshape(3, 1))
        output = np.eye(4, dtype=np.float64)
        output[:3, :3] = rotation
        output[:3, 3] = result.x[3:6]
        return output

    def refine_pose(
        self,
        initial_pose_w2c: np.ndarray,
        camera: ColmapCamera,
        points: IndependentVerificationPoints,
        refinement_config: IndependentPoseRefinementConfig | None = None,
        *,
        eligible_landmark_mask: np.ndarray | None = None,
    ) -> IndependentPoseRefinementResult:
        """Refine only on the caller-provided fit fold and bounded local basin."""

        config = refinement_config or IndependentPoseRefinementConfig()
        initial = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4)
        pose = initial.copy()
        initial_score = self.score_pose(
            pose, camera, points, eligible_landmark_mask=eligible_landmark_mask
        ).log_likelihood_mean
        current_score = float(initial_score)
        accepted = 0
        last_correspondences = IndependentPoseCorrespondences(
            point_indices=np.zeros((0,), dtype=np.int64),
            landmark_row_indices=np.zeros((0,), dtype=np.int64),
            track_ids=np.zeros((0,), dtype=np.int64),
            xy=np.zeros((0, 2), dtype=np.float64),
            xyz=np.zeros((0, 3), dtype=np.float64),
            weights=np.zeros((0,), dtype=np.float64),
            descriptor_similarities=np.zeros((0,), dtype=np.float64),
            initial_reprojection_distances_px=np.zeros((0,), dtype=np.float64),
        )
        failure_reason: str | None = None
        for _iteration in range(int(config.iterations)):
            correspondences = self.pose_conditioned_correspondences(
                pose,
                camera,
                points,
                config,
                eligible_landmark_mask=eligible_landmark_mask,
            )
            last_correspondences = correspondences
            if len(correspondences) < int(config.minimum_correspondences):
                failure_reason = "insufficient_correspondences"
                break
            proposed = self._optimize_correspondences(
                correspondences, pose, camera, config
            )
            if proposed is None:
                failure_reason = "robust_pnp_failure"
                break
            translation_step, rotation_step = self._pose_step(initial, proposed)
            if (
                translation_step > float(config.maximum_translation_step_m)
                or rotation_step > float(config.maximum_rotation_step_deg)
            ):
                failure_reason = "pose_step_safety_limit"
                break
            proposed_score = self.score_pose(
                proposed,
                camera,
                points,
                eligible_landmark_mask=eligible_landmark_mask,
            ).log_likelihood_mean
            if proposed_score < current_score + float(
                config.minimum_fit_log_likelihood_gain
            ):
                failure_reason = "non_improving_fit_likelihood"
                break
            pose = proposed
            current_score = float(proposed_score)
            accepted += 1
        translation_step, rotation_step = self._pose_step(initial, pose)
        return IndependentPoseRefinementResult(
            success=bool(accepted > 0),
            pose_w2c=pose,
            accepted_iterations=int(accepted),
            final_correspondence_count=int(len(last_correspondences)),
            fit_log_likelihood_before=float(initial_score),
            fit_log_likelihood_after=float(current_score),
            translation_step_m=float(translation_step),
            rotation_step_deg=float(rotation_step),
            used_track_ids=np.unique(last_correspondences.track_ids),
            failure_reason=failure_reason,
        )

    def score_pose(
        self,
        pose_w2c: np.ndarray,
        camera: ColmapCamera,
        points: IndependentVerificationPoints,
        *,
        eligible_landmark_mask: np.ndarray | None = None,
    ) -> IndependentLandmarkPoseLikelihoodScore:
        if points.descriptors.shape[1] != self._features.shape[1]:
            raise ValueError("query and landmark descriptor dimensions differ")
        if str(self.config.candidate_mode) == "fixed_global_topl":
            projected, eligible, _candidate_rows = (
                self._projected_fixed_candidate_landmarks(
                    pose_w2c,
                    camera,
                    points,
                    eligible_landmark_mask,
                )
            )
            evidence = self._fixed_candidate_point_evidence(
                projected,
                eligible,
                points,
                eligible_landmark_mask,
            )
        else:
            projected, eligible = self._projected_eligible_landmarks(
                pose_w2c, camera, eligible_landmark_mask
            )
            evidence = np.zeros((len(points),), dtype=np.float64)
        view_eligible_count = int(np.count_nonzero(eligible))
        landmark_rows = np.flatnonzero(eligible)
        if (
            str(self.config.candidate_mode) == "pose_local_knn"
            and len(points)
            and landmark_rows.size
        ):
            try:
                from scipy.spatial import cKDTree
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("SciPy is required for landmark pose likelihood") from exc
            tree = cKDTree(projected[landmark_rows])
            distances, neighbors = tree.query(
                points.xy,
                k=int(self.config.nearest_landmarks),
                distance_upper_bound=float(
                    self.config.maximum_reprojection_distance_px
                ),
                workers=int(self.config.kdtree_workers),
            )
            if distances.ndim == 1:
                distances = distances[:, None]
                neighbors = neighbors[:, None]
            for neighbor_rank in range(distances.shape[1]):
                valid = np.isfinite(distances[:, neighbor_rank]) & (
                    neighbors[:, neighbor_rank] < landmark_rows.size
                )
                if not np.any(valid):
                    continue
                matched_rows = landmark_rows[neighbors[valid, neighbor_rank]]
                similarity = np.sum(
                    points.descriptors[valid] * self._features[matched_rows], axis=1
                )
                descriptor_evidence = np.exp(
                    np.minimum(
                        (
                            similarity
                            - points.descriptor_reference_scores[valid]
                        )
                        / float(self.config.descriptor_temperature),
                        0.0,
                    )
                )
                spatial_evidence = np.exp(
                    -0.5
                    * np.square(
                        distances[valid, neighbor_rank]
                        / float(self.config.spatial_sigma_px)
                    )
                )
                evidence[valid] = np.maximum(
                    evidence[valid], descriptor_evidence * spatial_evidence
                )

        likelihood = float(self.config.outlier_likelihood) + (
            1.0 - float(self.config.outlier_likelihood)
        ) * evidence
        log_likelihood = np.log(np.maximum(likelihood, 1e-12))
        statistics = _fixed_group_log_likelihood_statistics(
            log_likelihood, points.xy
        )
        return IndependentLandmarkPoseLikelihoodScore(
            log_likelihood_sum=float(np.sum(log_likelihood)),
            log_likelihood_mean=(
                float(np.mean(log_likelihood)) if len(points) else float("-inf")
            ),
            **statistics,
            verification_point_count=int(len(points)),
            effective_point_count=int(np.count_nonzero(evidence > 0.0)),
            evidence_coverage=(
                float(np.mean(evidence > 0.05)) if len(points) else 0.0
            ),
            projected_landmark_count=int(landmark_rows.size),
            view_eligible_landmark_count=view_eligible_count,
            point_evidence=evidence,
        )
