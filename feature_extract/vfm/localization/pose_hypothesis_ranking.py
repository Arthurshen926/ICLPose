"""Train-only supervision for inference-safe PnP hypothesis ranking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from feature_extract.vfm.localization.pose_hypothesis_verifier import (
    PoseHypothesisRecord,
)


LEGACY_POSE_HYPOTHESIS_FEATURE_NAMES: tuple[str, ...] = (
    "fit_match_count_limit_log1p",
    "fit_match_count_fraction",
    "fit_inlier_count_log1p",
    "fit_inlier_ratio",
    "ransac_threshold_log1p",
    "selection_score_topk",
    "selection_spatial_round_robin",
    "selection_geometry_diverse",
    "verification_finite_ratio",
    "verification_positive_depth_ratio",
    "verification_strict_inlier_ratio",
    "verification_loose_inlier_ratio",
    "verification_strict_grid_fraction",
    "verification_loose_grid_fraction",
    "verification_soft_consensus_ratio",
    "verification_strict_given_loose_ratio",
    "verification_median_residual_log1p",
    "verification_depth_range_log1p",
    "verification_selected_candidate_fraction",
    "verification_selected_descriptor_score_mean",
    "verification_selected_descriptor_score_median",
    "verification_selected_descriptor_margin_mean",
    "verification_selected_descriptor_rank_score_mean",
    "verification_selected_assignment_utility_mean",
    "verification_selected_reprojection_mean_log1p",
    "verification_selected_reprojection_p90_log1p",
    "pose_cluster_5cm_0p5deg_fraction",
    "pose_cluster_10cm_1deg_fraction",
    "pose_cluster_25cm_2deg_fraction",
    "pose_cluster_50cm_5deg_fraction",
    "nearest_camera_center_distance_log1p",
    "median_camera_center_distance_log1p",
    "nearest_rotation_distance_log1p",
    "median_rotation_distance_log1p",
)

MEASUREMENT_POSE_HYPOTHESIS_FEATURE_NAMES: tuple[str, ...] = (
    "selection_measurement_verified",
    "measurement_evidence_fraction",
    "measurement_probability_mean",
    "measurement_high_confidence_fraction",
    "measurement_strict_probability_mass_fraction",
    "measurement_loose_probability_mass_fraction",
    "measurement_soft_consensus_ratio",
    "measurement_high_confidence_strict_fraction",
    "measurement_high_confidence_loose_fraction",
    "measurement_high_confidence_contradiction_fraction",
)

POSE_HYPOTHESIS_FEATURE_NAMES: tuple[str, ...] = (
    LEGACY_POSE_HYPOTHESIS_FEATURE_NAMES
    + MEASUREMENT_POSE_HYPOTHESIS_FEATURE_NAMES
)


@dataclass(frozen=True)
class HypothesisRankingGroup:
    query_id: str
    records: tuple[PoseHypothesisRecord, ...]
    poses_w2c: tuple[np.ndarray | None, ...]
    translation_errors_m: tuple[float, ...]
    rotation_errors_deg: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.records)
        if not (
            len(self.poses_w2c) == count
            and len(self.translation_errors_m) == count
            and len(self.rotation_errors_deg) == count
        ):
            raise ValueError("hypothesis ranking group arrays must have equal length")


@dataclass(frozen=True)
class PoseHypothesisRanker:
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    c_value: float
    rotation_equivalent_m_per_deg: float
    minimum_pair_gap_m: float

    def __post_init__(self) -> None:
        feature_count = len(POSE_HYPOTHESIS_FEATURE_NAMES)
        if len(self.scale) != feature_count or len(self.weights) != feature_count:
            raise ValueError("pose hypothesis ranker dimension is incompatible")
        if min(self.scale) <= 0.0:
            raise ValueError("pose hypothesis ranker scales must be positive")

    def scores(
        self,
        records: Sequence[PoseHypothesisRecord],
        poses_w2c: Sequence[np.ndarray | None],
        eligible_indices: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices, features = pose_hypothesis_features(
            records, poses_w2c, eligible_indices=eligible_indices
        )
        standardized = features / np.asarray(self.scale, dtype=np.float64)[None, :]
        scores = standardized @ np.asarray(self.weights, dtype=np.float64)
        return indices, scores

    def select(
        self,
        records: Sequence[PoseHypothesisRecord],
        poses_w2c: Sequence[np.ndarray | None],
        eligible_indices: Sequence[int],
    ) -> int:
        indices, scores = self.scores(
            records, poses_w2c, eligible_indices=eligible_indices
        )
        if len(indices) == 0:
            raise ValueError("pose hypothesis ranker received no eligible hypotheses")
        order = np.lexsort((indices, -scores))
        return int(indices[int(order[0])])

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(POSE_HYPOTHESIS_FEATURE_NAMES),
            "scale": list(self.scale),
            "weights": list(self.weights),
            "c_value": float(self.c_value),
            "rotation_equivalent_m_per_deg": float(
                self.rotation_equivalent_m_per_deg
            ),
            "minimum_pair_gap_m": float(self.minimum_pair_gap_m),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PoseHypothesisRanker":
        feature_names = tuple(str(value) for value in payload["feature_names"])
        scale = tuple(float(value) for value in payload["scale"])
        weights = tuple(float(value) for value in payload["weights"])
        if feature_names == LEGACY_POSE_HYPOTHESIS_FEATURE_NAMES:
            missing = len(MEASUREMENT_POSE_HYPOTHESIS_FEATURE_NAMES)
            scale = scale + tuple(1.0 for _index in range(missing))
            weights = weights + tuple(0.0 for _index in range(missing))
        elif feature_names != POSE_HYPOTHESIS_FEATURE_NAMES:
            raise ValueError("pose hypothesis ranker feature schema is incompatible")
        return cls(
            scale=scale,
            weights=weights,
            c_value=float(payload["c_value"]),
            rotation_equivalent_m_per_deg=float(
                payload["rotation_equivalent_m_per_deg"]
            ),
            minimum_pair_gap_m=float(payload["minimum_pair_gap_m"]),
        )


def eligible_hypothesis_indices(
    records: Sequence[PoseHypothesisRecord],
    poses_w2c: Sequence[np.ndarray | None],
) -> np.ndarray:
    if len(records) != len(poses_w2c):
        raise ValueError("hypothesis records and poses must have equal length")
    return np.asarray(
        [
            index
            for index, (record, pose) in enumerate(zip(records, poses_w2c))
            if bool(record.solver_success)
            and record.verification is not None
            and pose is not None
            and np.all(np.isfinite(np.asarray(pose, dtype=np.float64)))
        ],
        dtype=np.int64,
    )


def legacy_hypothesis_index(
    records: Sequence[PoseHypothesisRecord],
    poses_w2c: Sequence[np.ndarray | None],
) -> int:
    eligible = eligible_hypothesis_indices(records, poses_w2c)
    if len(eligible) == 0:
        raise ValueError("no eligible pose hypothesis")
    return int(
        max(
            eligible.tolist(),
            key=lambda index: (
                records[index].verification.rank_key(),  # type: ignore[union-attr]
                int(records[index].fit_inlier_count),
                -int(index),
            ),
        )
    )


def _camera_center(pose_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    return -pose[:3, :3].T @ pose[:3, 3]


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64) @ np.asarray(
        second, dtype=np.float64
    ).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _cross_hypothesis_distances(poses_w2c: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    count = len(poses_w2c)
    centers = np.stack([_camera_center(pose) for pose in poses_w2c])
    rotations = np.stack(
        [np.asarray(pose, dtype=np.float64).reshape(4, 4)[:3, :3] for pose in poses_w2c]
    )
    center_distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2
    )
    rotation_distances = np.zeros((count, count), dtype=np.float64)
    for first in range(count):
        for second in range(first + 1, count):
            distance = _rotation_distance_deg(rotations[first], rotations[second])
            rotation_distances[first, second] = distance
            rotation_distances[second, first] = distance
    return center_distances, rotation_distances


def pose_hypothesis_features(
    records: Sequence[PoseHypothesisRecord],
    poses_w2c: Sequence[np.ndarray | None],
    *,
    eligible_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inference-only features for eligible hypotheses.

    No ground-truth pose, residual, query identity, or split information is an
    input to this function.
    """

    all_eligible = eligible_hypothesis_indices(records, poses_w2c)
    if eligible_indices is None:
        indices = all_eligible
    else:
        requested = np.asarray(eligible_indices, dtype=np.int64).reshape(-1)
        allowed = set(int(value) for value in all_eligible.tolist())
        if any(int(value) not in allowed for value in requested.tolist()):
            raise ValueError("feature request contains an ineligible hypothesis")
        indices = requested
    if len(indices) == 0:
        return indices, np.empty((0, len(POSE_HYPOTHESIS_FEATURE_NAMES)))

    eligible_poses = [
        np.asarray(poses_w2c[int(index)], dtype=np.float64).reshape(4, 4)
        for index in all_eligible.tolist()
    ]
    center_distances, rotation_distances = _cross_hypothesis_distances(
        eligible_poses
    )
    local_index = {
        int(global_index): local for local, global_index in enumerate(all_eligible)
    }
    features: list[list[float]] = []
    for global_index in indices.tolist():
        record = records[int(global_index)]
        verification = record.verification
        if verification is None:
            raise RuntimeError("eligible hypothesis has no verification")
        count = max(int(verification.verification_count), 1)
        fit_count = max(int(record.fit_match_count), 1)
        local = local_index[int(global_index)]
        other_mask = np.arange(len(all_eligible)) != int(local)
        center_other = center_distances[local, other_mask]
        rotation_other = rotation_distances[local, other_mask]
        if center_other.size == 0:
            center_other = np.asarray([100.0], dtype=np.float64)
            rotation_other = np.asarray([180.0], dtype=np.float64)
        cluster_fractions = [
            float(
                np.mean(
                    (center_other <= translation_threshold)
                    & (rotation_other <= rotation_threshold)
                )
            )
            for translation_threshold, rotation_threshold in (
                (0.05, 0.5),
                (0.10, 1.0),
                (0.25, 2.0),
                (0.50, 5.0),
            )
        ]
        depth_range = (
            0.0
            if verification.depth_range_m is None
            else max(float(verification.depth_range_m), 0.0)
        )
        selected_score_mean = (
            0.0
            if verification.selected_descriptor_score_mean is None
            else float(verification.selected_descriptor_score_mean)
        )
        selected_score_median = (
            0.0
            if verification.selected_descriptor_score_median is None
            else float(verification.selected_descriptor_score_median)
        )
        selected_margin_mean = (
            0.0
            if verification.selected_descriptor_margin_mean is None
            else float(verification.selected_descriptor_margin_mean)
        )
        selected_rank_mean = (
            0.0
            if verification.selected_descriptor_rank_score_mean is None
            else float(verification.selected_descriptor_rank_score_mean)
        )
        selected_utility_mean = (
            0.0
            if verification.selected_assignment_utility_mean is None
            else float(verification.selected_assignment_utility_mean)
        )
        selected_reprojection_mean = (
            20.0
            if verification.selected_reprojection_mean_px is None
            else max(float(verification.selected_reprojection_mean_px), 0.0)
        )
        selected_reprojection_p90 = (
            20.0
            if verification.selected_reprojection_p90_px is None
            else max(float(verification.selected_reprojection_p90_px), 0.0)
        )
        values = [
            math.log1p(max(int(record.fit_match_count_limit), 0)),
            float(record.fit_match_count)
            / max(float(record.fit_match_count_limit), 1.0),
            math.log1p(max(int(record.fit_inlier_count), 0)),
            float(record.fit_inlier_count) / float(fit_count),
            math.log1p(max(float(record.ransac_threshold_px), 0.0)),
            float(record.selection_mode == "score_topk"),
            float(record.selection_mode == "spatial_round_robin"),
            float(record.selection_mode == "geometry_diverse"),
            float(verification.finite_count) / float(count),
            float(verification.positive_depth_ratio),
            float(verification.strict_inlier_count) / float(count),
            float(verification.loose_inlier_count) / float(count),
            float(verification.strict_grid_cell_count) / 16.0,
            float(verification.loose_grid_cell_count) / 16.0,
            float(verification.soft_consensus) / float(count),
            float(verification.strict_inlier_count)
            / max(float(verification.loose_inlier_count), 1.0),
            math.log1p(max(float(verification.clipped_median_residual_px), 0.0)),
            math.log1p(depth_range),
            float(verification.selected_candidate_fraction),
            selected_score_mean,
            selected_score_median,
            selected_margin_mean,
            selected_rank_mean,
            selected_utility_mean,
            math.log1p(selected_reprojection_mean),
            math.log1p(selected_reprojection_p90),
            *cluster_fractions,
            math.log1p(min(float(np.min(center_other)), 100.0)),
            math.log1p(min(float(np.median(center_other)), 100.0)),
            math.log1p(min(float(np.min(rotation_other)), 180.0)),
            math.log1p(min(float(np.median(rotation_other)), 180.0)),
            float(record.selection_mode.startswith("measurement_verified")),
            float(verification.measurement_evidence_fraction),
            (
                0.0
                if verification.measurement_probability_mean is None
                else float(verification.measurement_probability_mean)
            ),
            float(verification.measurement_high_confidence_fraction),
            float(verification.measurement_strict_probability_mass_fraction),
            float(verification.measurement_loose_probability_mass_fraction),
            float(verification.measurement_soft_consensus_ratio),
            float(verification.measurement_high_confidence_strict_fraction),
            float(verification.measurement_high_confidence_loose_fraction),
            float(verification.measurement_high_confidence_contradiction_fraction),
        ]
        features.append(values)
    output = np.asarray(features, dtype=np.float64)
    if output.shape[1] != len(POSE_HYPOTHESIS_FEATURE_NAMES):
        raise RuntimeError("pose hypothesis feature schema is inconsistent")
    if not np.all(np.isfinite(output)):
        raise ValueError("pose hypothesis features contain non-finite values")
    return indices, output


def _quality(
    group: HypothesisRankingGroup,
    *,
    rotation_equivalent_m_per_deg: float,
) -> np.ndarray:
    translation = np.asarray(group.translation_errors_m, dtype=np.float64)
    rotation = np.asarray(group.rotation_errors_deg, dtype=np.float64)
    return translation + float(rotation_equivalent_m_per_deg) * rotation


def _pairwise_training_arrays(
    groups: Sequence[HypothesisRankingGroup],
    *,
    rotation_equivalent_m_per_deg: float,
    minimum_pair_gap_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    differences: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    used_queries = 0
    for group in groups:
        indices, features = pose_hypothesis_features(group.records, group.poses_w2c)
        quality = _quality(
            group,
            rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
        )[indices]
        finite = np.isfinite(quality)
        indices = indices[finite]
        features = features[finite]
        quality = quality[finite]
        query_pair_count = 0
        for first in range(len(indices)):
            for second in range(first + 1, len(indices)):
                gap = float(quality[second] - quality[first])
                if abs(gap) < float(minimum_pair_gap_m):
                    continue
                difference = features[first] - features[second]
                first_is_better = int(gap > 0.0)
                pair_weight = float(np.clip(abs(gap) / 0.10, 0.25, 4.0))
                differences.extend((difference, -difference))
                labels.extend((first_is_better, 1 - first_is_better))
                weights.extend((pair_weight, pair_weight))
                query_pair_count += 1
        used_queries += int(query_pair_count > 0)
    if not differences or len(set(labels)) < 2:
        raise ValueError("pose hypothesis ranker requires non-tied train pairs")
    return (
        np.stack(differences),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        int(used_queries),
    )


def _fit_ranker(
    groups: Sequence[HypothesisRankingGroup],
    *,
    c_value: float,
    rotation_equivalent_m_per_deg: float,
    minimum_pair_gap_m: float,
) -> tuple[PoseHypothesisRanker, dict[str, object]]:
    differences, labels, pair_weights, used_queries = _pairwise_training_arrays(
        groups,
        rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
        minimum_pair_gap_m=minimum_pair_gap_m,
    )
    scale = np.std(differences, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    classifier = LogisticRegression(
        C=float(c_value),
        fit_intercept=False,
        solver="lbfgs",
        max_iter=3000,
        random_state=0,
    )
    classifier.fit(
        differences / scale[None, :], labels, sample_weight=pair_weights
    )
    ranker = PoseHypothesisRanker(
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in classifier.coef_[0]),
        c_value=float(c_value),
        rotation_equivalent_m_per_deg=float(rotation_equivalent_m_per_deg),
        minimum_pair_gap_m=float(minimum_pair_gap_m),
    )
    return ranker, {
        "pair_row_count": int(len(labels)),
        "pair_count": int(len(labels) // 2),
        "query_count_with_pairs": int(used_queries),
    }


def _selection_metrics(
    groups: Sequence[HypothesisRankingGroup],
    selector,
) -> dict[str, object]:
    translation: list[float] = []
    rotation: list[float] = []
    ranks: list[int] = []
    query_rows: list[dict[str, object]] = []
    for group in groups:
        index = int(selector(group))
        error_t = float(group.translation_errors_m[index])
        error_r = float(group.rotation_errors_deg[index])
        finite_errors = np.asarray(group.translation_errors_m, dtype=np.float64)
        finite_errors = finite_errors[np.isfinite(finite_errors)]
        if not np.isfinite(error_t) or not np.isfinite(error_r):
            continue
        rank = 1 + int(np.sum(finite_errors < error_t))
        translation.append(error_t)
        rotation.append(error_r)
        ranks.append(rank)
        query_rows.append(
            {
                "query_id": str(group.query_id),
                "chosen_hypothesis_index": index,
                "translation_m": error_t,
                "rotation_deg": error_r,
                "translation_rank": rank,
            }
        )
    values_t = np.asarray(translation, dtype=np.float64)
    values_r = np.asarray(rotation, dtype=np.float64)
    if values_t.size == 0:
        raise ValueError("pose hypothesis evaluation has no finite selected pose")
    return {
        "query_count": int(len(groups)),
        "finite_count": int(len(values_t)),
        "median_translation_m": float(np.median(values_t)),
        "p90_translation_m": float(np.quantile(values_t, 0.9)),
        "median_rotation_deg": float(np.median(values_r)),
        "recall_25cm_2deg": float(np.mean((values_t <= 0.25) & (values_r <= 2.0))),
        "recall_10cm_5deg": float(np.mean((values_t <= 0.10) & (values_r <= 5.0))),
        "recall_5cm_5deg": float(np.mean((values_t <= 0.05) & (values_r <= 5.0))),
        "median_translation_rank": float(np.median(np.asarray(ranks))),
        "rows": query_rows,
    }


def evaluate_pose_hypothesis_ranker(
    groups: Sequence[HypothesisRankingGroup],
    ranker: PoseHypothesisRanker,
) -> dict[str, object]:
    return _selection_metrics(
        groups,
        lambda group: ranker.select(
            group.records,
            group.poses_w2c,
            eligible_hypothesis_indices(group.records, group.poses_w2c),
        ),
    )


def evaluate_legacy_hypothesis_ranking(
    groups: Sequence[HypothesisRankingGroup],
) -> dict[str, object]:
    return _selection_metrics(
        groups,
        lambda group: legacy_hypothesis_index(group.records, group.poses_w2c),
    )


def _fold_assignments(
    groups: Sequence[HypothesisRankingGroup], folds: int
) -> dict[str, int]:
    if int(folds) < 2:
        raise ValueError("pose hypothesis ranker needs at least two folds")
    ordered = sorted(
        (str(group.query_id) for group in groups),
        key=lambda value: hashlib.sha256(value.encode("utf8")).digest(),
    )
    return {query_id: index % int(folds) for index, query_id in enumerate(ordered)}


def _metric_key(
    metrics: Mapping[str, object], *, rotation_equivalent_m_per_deg: float
) -> tuple[float, ...]:
    median_translation = float(metrics["median_translation_m"])
    p90_translation = float(metrics["p90_translation_m"])
    median_rotation = float(metrics["median_rotation_deg"])
    balanced_risk = (
        median_translation
        + 0.5 * p90_translation
        + float(rotation_equivalent_m_per_deg) * median_rotation
    )
    return (
        balanced_risk,
        p90_translation,
        median_translation,
        median_rotation,
        -float(metrics["recall_10cm_5deg"]),
        -float(metrics["recall_5cm_5deg"]),
    )


def fit_pose_hypothesis_ranker(
    groups: Sequence[HypothesisRankingGroup],
    *,
    c_values: Sequence[float] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    cross_validation_folds: int = 5,
    rotation_equivalent_m_per_deg: float = 0.02,
    minimum_pair_gap_m: float = 0.005,
) -> tuple[PoseHypothesisRanker, dict[str, object]]:
    values = tuple(float(value) for value in c_values)
    if not groups or not values or min(values) <= 0.0:
        raise ValueError("pose hypothesis ranker requires train groups and positive C values")
    fold_count = min(int(cross_validation_folds), len(groups))
    assignments = _fold_assignments(groups, fold_count)
    candidates: list[dict[str, object]] = []
    for c_value in values:
        fold_metrics: list[dict[str, object]] = []
        rows: list[dict[str, object]] = []
        for fold in range(fold_count):
            train = [
                group
                for group in groups
                if assignments[str(group.query_id)] != int(fold)
            ]
            heldout = [
                group
                for group in groups
                if assignments[str(group.query_id)] == int(fold)
            ]
            ranker, _fit = _fit_ranker(
                train,
                c_value=c_value,
                rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
                minimum_pair_gap_m=minimum_pair_gap_m,
            )
            metrics = evaluate_pose_hypothesis_ranker(heldout, ranker)
            fold_metrics.append({key: value for key, value in metrics.items() if key != "rows"})
            rows.extend(metrics["rows"])
        by_query = {str(row["query_id"]): row for row in rows}
        ordered_rows = [by_query[str(group.query_id)] for group in groups]
        selected_t = np.asarray(
            [float(row["translation_m"]) for row in ordered_rows], dtype=np.float64
        )
        selected_r = np.asarray(
            [float(row["rotation_deg"]) for row in ordered_rows], dtype=np.float64
        )
        aggregate = {
            "query_count": int(len(ordered_rows)),
            "median_translation_m": float(np.median(selected_t)),
            "p90_translation_m": float(np.quantile(selected_t, 0.9)),
            "median_rotation_deg": float(np.median(selected_r)),
            "recall_25cm_2deg": float(
                np.mean((selected_t <= 0.25) & (selected_r <= 2.0))
            ),
            "recall_10cm_5deg": float(
                np.mean((selected_t <= 0.10) & (selected_r <= 5.0))
            ),
            "recall_5cm_5deg": float(
                np.mean((selected_t <= 0.05) & (selected_r <= 5.0))
            ),
            "median_translation_rank": float(
                np.median([float(row["translation_rank"]) for row in ordered_rows])
            ),
        }
        candidates.append(
            {
                "c_value": float(c_value),
                "aggregate": aggregate,
                "folds": fold_metrics,
            }
        )
    chosen = min(
        candidates,
        key=lambda item: _metric_key(
            item["aggregate"],
            rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
        ),
    )
    ranker, fit_details = _fit_ranker(
        groups,
        c_value=float(chosen["c_value"]),
        rotation_equivalent_m_per_deg=rotation_equivalent_m_per_deg,
        minimum_pair_gap_m=minimum_pair_gap_m,
    )
    legacy = evaluate_legacy_hypothesis_ranking(groups)
    learned_in_sample = evaluate_pose_hypothesis_ranker(groups, ranker)
    importance = sorted(
        (
            {
                "feature": name,
                "weight": float(weight),
                "absolute_weight": abs(float(weight)),
            }
            for name, weight in zip(POSE_HYPOTHESIS_FEATURE_NAMES, ranker.weights)
        ),
        key=lambda item: -float(item["absolute_weight"]),
    )
    return ranker, {
        "cross_validation_folds": int(fold_count),
        "fold_assignments": assignments,
        "candidates": candidates,
        "chosen_c_value": float(chosen["c_value"]),
        "c_selection_objective": (
            "median_translation_m + 0.5 * p90_translation_m + "
            "rotation_equivalent_m_per_deg * median_rotation_deg"
        ),
        "chosen_oof_metrics": chosen["aggregate"],
        "legacy_train_metrics": {key: value for key, value in legacy.items() if key != "rows"},
        "learned_train_in_sample_metrics": {
            key: value for key, value in learned_in_sample.items() if key != "rows"
        },
        "fit": fit_details,
        "feature_importance": importance,
    }


def fit_pose_hypothesis_ranker_fixed(
    groups: Sequence[HypothesisRankingGroup],
    *,
    c_value: float,
    rotation_equivalent_m_per_deg: float = 0.02,
    minimum_pair_gap_m: float = 0.005,
) -> PoseHypothesisRanker:
    """Fit one fixed ranker without any held-out hyperparameter selection."""

    ranker, _details = _fit_ranker(
        groups,
        c_value=float(c_value),
        rotation_equivalent_m_per_deg=float(rotation_equivalent_m_per_deg),
        minimum_pair_gap_m=float(minimum_pair_gap_m),
    )
    return ranker
