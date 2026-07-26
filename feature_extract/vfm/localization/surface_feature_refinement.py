"""Trust-region featuremetric refinement on fixed 2DGS surface anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logsumexp

from feature_extract.vfm.localization.real_image_observation_features import (
    AlikeDenseObservationExtractor,
)
from feature_extract.vfm.localization.surface_localization import (
    AnchorLocalDescriptorBank,
)
from feature_extract.vfm.query_to_3d_matching import (
    camera_matrix_and_distortion,
)
from feature_extract.vfm.surface_maplet_bank import StableSurfaceAnchorMap


@dataclass(frozen=True)
class SurfaceFeatureRefinementConfig:
    maximum_anchors: int = 192
    maximum_anchors_per_cell: int = 4
    grid_rows: int = 8
    grid_cols: int = 8
    minimum_normal_cosine: float = 0.15
    search_radius_px: float = 24.0
    coarse_step_px: float = 2.0
    fine_radius_px: float = 2.0
    fine_step_px: float = 0.5
    similarity_center: float = 0.65
    similarity_scale: float = 0.08
    null_probability: float = 0.5
    view_direction_temperature: float = 0.10
    minimum_measurement_llr: float = 0.0
    minimum_peak_margin: float = 0.05
    minimum_query_keypoint_score: float = 0.01
    query_keypoint_score_center: float = 0.05
    query_keypoint_log_weight: float = 1.0
    displacement_prior_sigma_px: float = 10.0
    measurement_nms_radius_px: float = 2.0
    minimum_fit_anchors: int = 12
    minimum_holdout_anchors: int = 4
    holdout_stride: int = 5
    ransac_reprojection_px: float = 4.0
    maximum_translation_update_m: float = 0.35
    maximum_rotation_update_deg: float = 1.5
    minimum_initial_holdout_residual_px: float = 8.0
    minimum_holdout_evidence_gain: float = 0.4
    minimum_holdout_residual_gain_px: float = 3.0

    def __post_init__(self) -> None:
        if int(self.maximum_anchors) <= 0:
            raise ValueError("maximum_anchors must be positive")
        if int(self.maximum_anchors_per_cell) <= 0:
            raise ValueError("maximum_anchors_per_cell must be positive")
        if int(self.grid_rows) <= 0 or int(self.grid_cols) <= 0:
            raise ValueError("grid dimensions must be positive")
        if float(self.search_radius_px) <= 0.0:
            raise ValueError("search radius must be positive")
        if (
            float(self.coarse_step_px) <= 0.0
            or float(self.fine_step_px) <= 0.0
        ):
            raise ValueError("search steps must be positive")
        if float(self.similarity_scale) <= 0.0:
            raise ValueError("similarity_scale must be positive")
        if float(self.minimum_query_keypoint_score) < 0.0:
            raise ValueError(
                "minimum_query_keypoint_score must be non-negative"
            )
        if float(self.query_keypoint_score_center) <= 0.0:
            raise ValueError(
                "query_keypoint_score_center must be positive"
            )
        if float(self.displacement_prior_sigma_px) <= 0.0:
            raise ValueError(
                "displacement_prior_sigma_px must be positive"
            )
        if not 0.0 < float(self.null_probability) < 1.0:
            raise ValueError("null_probability must be in (0, 1)")
        if float(self.view_direction_temperature) <= 0.0:
            raise ValueError("view_direction_temperature must be positive")
        if int(self.holdout_stride) < 2:
            raise ValueError("holdout_stride must be at least two")
        if int(self.minimum_fit_anchors) < 4:
            raise ValueError("minimum_fit_anchors must be at least four")
        if int(self.minimum_holdout_anchors) < 1:
            raise ValueError("minimum_holdout_anchors must be positive")
        if (
            float(self.minimum_initial_holdout_residual_px) < 0.0
            or float(self.minimum_holdout_evidence_gain) < 0.0
            or float(self.minimum_holdout_residual_gain_px) < 0.0
        ):
            raise ValueError("holdout acceptance gains must be non-negative")


@dataclass(frozen=True)
class SurfaceFeatureRefinementResult:
    success: bool
    accepted: bool
    pose_w2c: np.ndarray
    measurement_count: int
    fit_count: int
    holdout_count: int
    initial_holdout_evidence: float | None
    refined_holdout_evidence: float | None
    initial_holdout_residual_px: float | None
    refined_holdout_residual_px: float | None
    translation_update_m: float
    rotation_update_deg: float
    failure_reason: str | None

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose_w2c, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError("pose_w2c must have shape (4, 4)")
        object.__setattr__(self, "pose_w2c", pose)


def _pose_distance(
    first_w2c: np.ndarray,
    second_w2c: np.ndarray,
) -> tuple[float, float]:
    first = np.asarray(first_w2c, dtype=np.float64).reshape(4, 4)
    second = np.asarray(second_w2c, dtype=np.float64).reshape(4, 4)
    first_center = -first[:3, :3].T @ first[:3, 3]
    second_center = -second[:3, :3].T @ second[:3, 3]
    translation = float(np.linalg.norm(first_center - second_center))
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = float(
        np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    )
    return translation, float(np.degrees(np.arccos(cosine)))


def _project(
    xyz: np.ndarray,
    pose_w2c: np.ndarray,
    camera,
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    camera_xyz = points @ pose[:3, :3].T + pose[:3, 3]
    matrix, distortion = camera_matrix_and_distortion(camera)
    rvec, _jacobian = cv2.Rodrigues(pose[:3, :3])
    projected, _jacobian = cv2.projectPoints(
        points,
        rvec,
        pose[:3, 3],
        matrix,
        distortion,
    )
    return projected.reshape(-1, 2), camera_xyz[:, 2]


def _descriptor_lookup(
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
) -> tuple[np.ndarray, np.ndarray]:
    bank_row_by_id = {
        int(anchor_id): int(row)
        for row, anchor_id in enumerate(
            descriptor_bank.anchor_ids.tolist()
        )
    }
    offsets = np.full((len(anchors), 2), -1, dtype=np.int64)
    quality = np.zeros((len(anchors),), dtype=np.float64)
    for anchor_row, anchor_id in enumerate(anchors.anchor_ids.tolist()):
        bank_row = bank_row_by_id.get(int(anchor_id))
        if bank_row is None:
            continue
        start = int(descriptor_bank.descriptor_offsets[bank_row])
        end = int(descriptor_bank.descriptor_offsets[bank_row + 1])
        if end <= start:
            continue
        offsets[anchor_row] = (start, end)
        quality[anchor_row] = (
            max(float(anchors.quality_scores[anchor_row]), 1e-8)
            * np.log1p(end - start)
        )
    return offsets, quality


def _select_visible_anchor_rows(
    *,
    pose_w2c: np.ndarray,
    anchors: StableSurfaceAnchorMap,
    descriptor_quality: np.ndarray,
    camera,
    config: SurfaceFeatureRefinementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    projected, depth = _project(anchors.xyz, pose_w2c, camera)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    direction = camera_center[None, :] - anchors.xyz
    direction /= np.maximum(
        np.linalg.norm(direction, axis=1, keepdims=True),
        1e-12,
    )
    normal_cosine = np.sum(anchors.normals * direction, axis=1)
    valid = (
        (descriptor_quality > 0.0)
        & (depth > 1e-6)
        & (projected[:, 0] >= float(config.search_radius_px))
        & (
            projected[:, 0]
            <= float(camera.width - 1) - float(config.search_radius_px)
        )
        & (projected[:, 1] >= float(config.search_radius_px))
        & (
            projected[:, 1]
            <= float(camera.height - 1) - float(config.search_radius_px)
        )
        & (
            normal_cosine
            >= float(config.minimum_normal_cosine)
        )
    )
    candidates = np.flatnonzero(valid)
    score = descriptor_quality[candidates] * (
        0.5 + 0.5 * normal_cosine[candidates]
    )
    order = candidates[np.lexsort((candidates, -score))]
    cell_counts = np.zeros(
        (int(config.grid_rows), int(config.grid_cols)),
        dtype=np.int64,
    )
    selected: list[int] = []
    for anchor_row in order.tolist():
        col = int(
            np.clip(
                projected[anchor_row, 0]
                / max(float(camera.width), 1.0)
                * int(config.grid_cols),
                0,
                int(config.grid_cols) - 1,
            )
        )
        row = int(
            np.clip(
                projected[anchor_row, 1]
                / max(float(camera.height), 1.0)
                * int(config.grid_rows),
                0,
                int(config.grid_rows) - 1,
            )
        )
        if (
            cell_counts[row, col]
            >= int(config.maximum_anchors_per_cell)
        ):
            continue
        selected.append(anchor_row)
        cell_counts[row, col] += 1
        if len(selected) >= int(config.maximum_anchors):
            break
    rows = np.asarray(selected, dtype=np.int64)
    return rows, projected[rows].astype(np.float32)


def _view_log_weights(
    support_image_ids: Sequence[str],
    *,
    support_view_directions: np.ndarray,
    support_quality: np.ndarray,
    query_view_direction: np.ndarray,
    excluded_image_id: str,
    view_mode_scores: Mapping[str, float],
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    retained = np.asarray(
        [
            index
            for index, image_id in enumerate(support_image_ids)
            if str(image_id) != str(excluded_image_id)
        ],
        dtype=np.int64,
    )
    if len(retained) == 0:
        return retained, np.zeros((0,), dtype=np.float64)
    quality = np.maximum(
        np.asarray(support_quality, dtype=np.float64)[retained],
        1e-8,
    )
    logits = np.log(quality)
    directions = np.asarray(
        support_view_directions, dtype=np.float64
    )[retained]
    direction_valid = np.linalg.norm(directions, axis=1) > 0.5
    query_direction = np.asarray(
        query_view_direction, dtype=np.float64
    ).reshape(3)
    query_norm = float(np.linalg.norm(query_direction))
    if np.any(direction_valid) and query_norm > 0.5:
        query_direction /= query_norm
        view_cosine = directions @ query_direction
        logits += np.where(
            direction_valid,
            view_cosine / float(temperature),
            np.min(view_cosine[direction_valid]) / float(temperature),
        )
    retained_ids = [support_image_ids[index] for index in retained.tolist()]
    if view_mode_scores and any(
        image_id in view_mode_scores for image_id in retained_ids
    ):
        minimum = min(view_mode_scores.values())
        logits += np.asarray(
            [
                (
                    view_mode_scores.get(
                        image_id,
                        minimum - float(temperature),
                    )
                    / float(temperature)
                )
                for image_id in retained_ids
            ],
            dtype=np.float64,
        )
    return retained, logits - logsumexp(logits)


def _mixture_llr(
    query_descriptors: np.ndarray,
    support_descriptors: np.ndarray,
    log_weights: np.ndarray,
    config: SurfaceFeatureRefinementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    similarity = (
        np.asarray(query_descriptors, dtype=np.float32)
        @ np.asarray(support_descriptors, dtype=np.float32).T
    ).astype(np.float64)
    edge_llr = np.clip(
        (
            similarity - float(config.similarity_center)
        )
        / float(config.similarity_scale),
        -12.0,
        12.0,
    )
    mixture = logsumexp(
        edge_llr + np.asarray(log_weights)[None, :],
        axis=1,
    )
    null = float(config.null_probability)
    evidence = np.logaddexp(
        np.log(null),
        np.log1p(-null) + mixture,
    )
    return evidence, np.max(similarity, axis=1)


def _measurement_objective(
    evidence: np.ndarray,
    keypoint_score: np.ndarray,
    displacement_xy: np.ndarray,
    config: SurfaceFeatureRefinementConfig,
) -> np.ndarray:
    detector_log_ratio = np.log(
        np.maximum(np.asarray(keypoint_score, dtype=np.float64), 1e-8)
        / float(config.query_keypoint_score_center)
    )
    detector_log_ratio = np.clip(detector_log_ratio, -6.0, 3.0)
    displacement = np.asarray(displacement_xy, dtype=np.float64)
    spatial_penalty = 0.5 * np.sum(
        displacement * displacement, axis=-1
    ) / float(config.displacement_prior_sigma_px) ** 2
    return (
        np.asarray(evidence, dtype=np.float64)
        + float(config.query_keypoint_log_weight) * detector_log_ratio
        - spatial_penalty
    )


def _measure_anchor_positions(
    *,
    image_path: Path,
    image_id: str,
    camera,
    anchor_rows: np.ndarray,
    predicted_xy: np.ndarray,
    query_view_directions: np.ndarray,
    descriptor_offsets: np.ndarray,
    descriptor_bank: AnchorLocalDescriptorBank,
    alike: AlikeDenseObservationExtractor,
    view_mode_scores: Mapping[str, float],
    config: SurfaceFeatureRefinementConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coarse_axis = np.arange(
        -float(config.search_radius_px),
        float(config.search_radius_px) + 0.5 * float(config.coarse_step_px),
        float(config.coarse_step_px),
        dtype=np.float32,
    )
    coarse_offsets = np.stack(
        np.meshgrid(coarse_axis, coarse_axis, indexing="xy"),
        axis=-1,
    ).reshape(-1, 2)
    candidate_xy = (
        np.asarray(predicted_xy, dtype=np.float32)[:, None, :]
        + coarse_offsets[None, :, :]
    )
    sampled, coarse_scores_flat, _image_hash = alike.sample_points(
        image_path,
        candidate_xy.reshape(-1, 2),
        image_width=int(camera.width),
        image_height=int(camera.height),
    )
    sampled = sampled.reshape(
        len(anchor_rows),
        len(coarse_offsets),
        -1,
    )
    coarse_scores = coarse_scores_flat.reshape(
        len(anchor_rows),
        len(coarse_offsets),
    )
    coarse_best = np.zeros((len(anchor_rows),), dtype=np.int64)
    coarse_evidence: list[np.ndarray] = []
    valid_support = np.zeros((len(anchor_rows),), dtype=bool)
    for row, anchor_row in enumerate(anchor_rows.tolist()):
        start, end = descriptor_offsets[int(anchor_row)]
        retained, log_weights = _view_log_weights(
            descriptor_bank.support_image_ids[int(start) : int(end)],
            support_view_directions=descriptor_bank.support_view_directions[
                int(start) : int(end)
            ],
            support_quality=descriptor_bank.descriptor_quality[
                int(start) : int(end)
            ],
            query_view_direction=query_view_directions[row],
            excluded_image_id=image_id,
            view_mode_scores=view_mode_scores,
            temperature=float(config.view_direction_temperature),
        )
        if len(retained) == 0:
            coarse_evidence.append(
                np.full(
                    (len(coarse_offsets),),
                    -float("inf"),
                    dtype=np.float64,
                )
            )
            continue
        evidence, _similarity = _mixture_llr(
            sampled[row],
            descriptor_bank.descriptors[
                int(start) : int(end)
            ][retained],
            log_weights,
            config,
        )
        objective = _measurement_objective(
            evidence,
            coarse_scores[row],
            coarse_offsets,
            config,
        )
        coarse_evidence.append(objective)
        coarse_best[row] = int(np.argmax(objective))
        valid_support[row] = True
    coarse_evidence_array = np.stack(coarse_evidence, axis=0)
    coarse_xy = candidate_xy[
        np.arange(len(anchor_rows)),
        coarse_best,
    ]
    fine_axis = np.arange(
        -float(config.fine_radius_px),
        float(config.fine_radius_px) + 0.5 * float(config.fine_step_px),
        float(config.fine_step_px),
        dtype=np.float32,
    )
    fine_offsets = np.stack(
        np.meshgrid(fine_axis, fine_axis, indexing="xy"),
        axis=-1,
    ).reshape(-1, 2)
    fine_xy = coarse_xy[:, None, :] + fine_offsets[None, :, :]
    fine_sampled, fine_scores_flat, _image_hash = alike.sample_points(
        image_path,
        fine_xy.reshape(-1, 2),
        image_width=int(camera.width),
        image_height=int(camera.height),
    )
    fine_sampled = fine_sampled.reshape(
        len(anchor_rows),
        len(fine_offsets),
        -1,
    )
    fine_scores = fine_scores_flat.reshape(
        len(anchor_rows),
        len(fine_offsets),
    )
    measured_xy = coarse_xy.copy()
    confidence = np.zeros((len(anchor_rows),), dtype=np.float64)
    peak_evidence = np.full(
        (len(anchor_rows),),
        -float("inf"),
        dtype=np.float64,
    )
    peak_keypoint_score = np.zeros(
        (len(anchor_rows),),
        dtype=np.float64,
    )
    for row, anchor_row in enumerate(anchor_rows.tolist()):
        if not valid_support[row]:
            continue
        start, end = descriptor_offsets[int(anchor_row)]
        retained, log_weights = _view_log_weights(
            descriptor_bank.support_image_ids[int(start) : int(end)],
            support_view_directions=descriptor_bank.support_view_directions[
                int(start) : int(end)
            ],
            support_quality=descriptor_bank.descriptor_quality[
                int(start) : int(end)
            ],
            query_view_direction=query_view_directions[row],
            excluded_image_id=image_id,
            view_mode_scores=view_mode_scores,
            temperature=float(config.view_direction_temperature),
        )
        evidence, _similarity = _mixture_llr(
            fine_sampled[row],
            descriptor_bank.descriptors[
                int(start) : int(end)
            ][retained],
            log_weights,
            config,
        )
        total_displacement = (
            fine_xy[row] - np.asarray(predicted_xy[row])
        )
        objective = _measurement_objective(
            evidence,
            fine_scores[row],
            total_displacement,
            config,
        )
        best = int(np.argmax(objective))
        peak_evidence[row] = float(evidence[best])
        peak_keypoint_score[row] = float(fine_scores[row, best])
        measured_xy[row] = fine_xy[row, best]
        coarse_values = coarse_evidence_array[row]
        coarse_delta = (
            coarse_offsets
            - coarse_offsets[int(coarse_best[row])]
        )
        competing = coarse_values[
            np.sum(coarse_delta * coarse_delta, axis=1) >= 9.0
        ]
        second = (
            float(np.max(competing))
            if len(competing)
            else -float("inf")
        )
        margin = float(objective[best] - second)
        confidence[row] = float(
            expit(objective[best])
            * expit(
                (
                    margin - float(config.minimum_peak_margin)
                )
                / 0.05
            )
        )
    valid = (
        valid_support
        & np.isfinite(peak_evidence)
        & (
            peak_evidence
            >= float(config.minimum_measurement_llr)
        )
        & (
            peak_keypoint_score
            >= float(config.minimum_query_keypoint_score)
        )
    )
    order = np.flatnonzero(valid)[
        np.argsort(-confidence[valid], kind="mergesort")
    ]
    keep: list[int] = []
    radius2 = float(config.measurement_nms_radius_px) ** 2
    for row in order.tolist():
        if keep:
            delta = measured_xy[np.asarray(keep)] - measured_xy[row]
            if np.any(np.sum(delta * delta, axis=1) <= radius2):
                continue
        keep.append(row)
    kept = np.asarray(keep, dtype=np.int64)
    return (
        anchor_rows[kept],
        measured_xy[kept].astype(np.float32),
        confidence[kept].astype(np.float32),
        peak_evidence[kept].astype(np.float32),
    )


def _projected_feature_evidence(
    *,
    pose_w2c: np.ndarray,
    image_path: Path,
    image_id: str,
    camera,
    anchor_rows: np.ndarray,
    descriptor_offsets: np.ndarray,
    descriptor_bank: AnchorLocalDescriptorBank,
    anchors: StableSurfaceAnchorMap,
    alike: AlikeDenseObservationExtractor,
    view_mode_scores: Mapping[str, float],
    config: SurfaceFeatureRefinementConfig,
) -> float | None:
    if len(anchor_rows) == 0:
        return None
    xy, depth = _project(anchors.xyz[anchor_rows], pose_w2c, camera)
    valid = (
        (depth > 1e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= float(camera.width - 1))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= float(camera.height - 1))
    )
    if not np.any(valid):
        return None
    query, _scores, _hash = alike.sample_points(
        image_path,
        xy[valid].astype(np.float32),
        image_width=int(camera.width),
        image_height=int(camera.height),
    )
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    camera_center = -pose[:3, :3].T @ pose[:3, 3]
    query_view_directions = (
        camera_center[None, :] - anchors.xyz[anchor_rows[valid]]
    )
    query_view_directions /= np.maximum(
        np.linalg.norm(query_view_directions, axis=1, keepdims=True),
        1e-12,
    )
    values: list[float] = []
    for query_row, anchor_row in enumerate(anchor_rows[valid].tolist()):
        start, end = descriptor_offsets[int(anchor_row)]
        retained, log_weights = _view_log_weights(
            descriptor_bank.support_image_ids[int(start) : int(end)],
            support_view_directions=descriptor_bank.support_view_directions[
                int(start) : int(end)
            ],
            support_quality=descriptor_bank.descriptor_quality[
                int(start) : int(end)
            ],
            query_view_direction=query_view_directions[query_row],
            excluded_image_id=image_id,
            view_mode_scores=view_mode_scores,
            temperature=float(config.view_direction_temperature),
        )
        if len(retained) == 0:
            continue
        evidence, _similarity = _mixture_llr(
            query[query_row : query_row + 1],
            descriptor_bank.descriptors[
                int(start) : int(end)
            ][retained],
            log_weights,
            config,
        )
        values.append(float(evidence[0]))
    return float(np.mean(values)) if values else None


def _weighted_pose_fit(
    *,
    initial_pose_w2c: np.ndarray,
    xyz: np.ndarray,
    xy: np.ndarray,
    weights: np.ndarray,
    camera,
    config: SurfaceFeatureRefinementConfig,
) -> np.ndarray | None:
    pose = np.asarray(
        initial_pose_w2c, dtype=np.float64
    ).reshape(4, 4).copy()
    matrix, distortion = camera_matrix_and_distortion(camera)
    initial_rotation = pose[:3, :3].copy()
    initial_center = -initial_rotation.T @ pose[:3, 3]
    initial = np.zeros((6,), dtype=np.float64)
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
    initial_rvec, _jacobian = cv2.Rodrigues(initial_rotation)
    success, _rvec, _tvec, inliers = cv2.solvePnPRansac(
        points,
        pixels,
        matrix,
        distortion,
        rvec=initial_rvec,
        tvec=pose[:3, 3].reshape(3, 1).copy(),
        useExtrinsicGuess=True,
        iterationsCount=128,
        reprojectionError=float(config.ransac_reprojection_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if (
        not bool(success)
        or inliers is None
        or len(inliers) < int(config.minimum_fit_anchors)
    ):
        return None
    inlier_rows = np.asarray(inliers, dtype=np.int64).reshape(-1)
    points = points[inlier_rows]
    pixels = pixels[inlier_rows]
    weight_values = weight_values[inlier_rows]
    values = np.sqrt(
        np.maximum(weight_values, 1e-4)
    )

    def pose_from_increment(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta_rotation, _jacobian = cv2.Rodrigues(
            parameters[:3].reshape(3, 1)
        )
        rotation = delta_rotation @ initial_rotation
        center = initial_center + parameters[3:6]
        return rotation, -rotation @ center

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation, translation = pose_from_increment(parameters)
        rvec, _jacobian = cv2.Rodrigues(rotation)
        projected, _jacobian = cv2.projectPoints(
            points,
            rvec,
            translation,
            matrix,
            distortion,
        )
        return (
            (projected.reshape(-1, 2) - pixels)
            * values[:, None]
        ).reshape(-1)

    rotation_bound = np.radians(
        float(config.maximum_rotation_update_deg)
    )
    translation_bound = float(config.maximum_translation_update_m)
    bound = np.asarray(
        [
            rotation_bound,
            rotation_bound,
            rotation_bound,
            translation_bound,
            translation_bound,
            translation_bound,
        ],
        dtype=np.float64,
    )
    optimized = least_squares(
        residual,
        initial,
        bounds=(-bound, bound),
        method="trf",
        loss="huber",
        f_scale=float(config.ransac_reprojection_px),
        max_nfev=80,
    )
    if not bool(optimized.success) or not np.isfinite(optimized.x).all():
        return None
    rotation, translation = pose_from_increment(optimized.x)
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation
    output[:3, 3] = translation
    return output


def refine_surface_pose_featuremetric(
    *,
    initial_pose_w2c: np.ndarray,
    image_path: Path,
    image_id: str,
    camera,
    anchors: StableSurfaceAnchorMap,
    descriptor_bank: AnchorLocalDescriptorBank,
    alike: AlikeDenseObservationExtractor,
    view_mode_scores: Mapping[str, float] | None = None,
    config: SurfaceFeatureRefinementConfig | None = None,
) -> SurfaceFeatureRefinementResult:
    policy = config or SurfaceFeatureRefinementConfig()
    initial = np.asarray(
        initial_pose_w2c, dtype=np.float64
    ).reshape(4, 4).copy()
    descriptor_offsets, descriptor_quality = _descriptor_lookup(
        anchors,
        descriptor_bank,
    )
    anchor_rows, predicted_xy = _select_visible_anchor_rows(
        pose_w2c=initial,
        anchors=anchors,
        descriptor_quality=descriptor_quality,
        camera=camera,
        config=policy,
    )
    initial_center = -initial[:3, :3].T @ initial[:3, 3]
    query_view_directions = (
        initial_center[None, :] - anchors.xyz[anchor_rows]
    )
    query_view_directions /= np.maximum(
        np.linalg.norm(query_view_directions, axis=1, keepdims=True),
        1e-12,
    )
    if len(anchor_rows) < (
        int(policy.minimum_fit_anchors)
        + int(policy.minimum_holdout_anchors)
    ):
        return SurfaceFeatureRefinementResult(
            success=False,
            accepted=False,
            pose_w2c=initial,
            measurement_count=0,
            fit_count=0,
            holdout_count=0,
            initial_holdout_evidence=None,
            refined_holdout_evidence=None,
            initial_holdout_residual_px=None,
            refined_holdout_residual_px=None,
            translation_update_m=0.0,
            rotation_update_deg=0.0,
            failure_reason="insufficient_visible_anchors",
        )
    measured_rows, measured_xy, confidence, _evidence = (
        _measure_anchor_positions(
            image_path=Path(image_path),
            image_id=str(image_id),
            camera=camera,
            anchor_rows=anchor_rows,
            predicted_xy=predicted_xy,
            query_view_directions=query_view_directions,
            descriptor_offsets=descriptor_offsets,
            descriptor_bank=descriptor_bank,
            alike=alike,
            view_mode_scores=dict(view_mode_scores or {}),
            config=policy,
        )
    )
    holdout = (
        np.mod(
            np.abs(anchors.anchor_ids[measured_rows]),
            int(policy.holdout_stride),
        )
        == 0
    )
    fit = ~holdout
    if (
        int(np.sum(fit)) < int(policy.minimum_fit_anchors)
        or int(np.sum(holdout))
        < int(policy.minimum_holdout_anchors)
    ):
        return SurfaceFeatureRefinementResult(
            success=False,
            accepted=False,
            pose_w2c=initial,
            measurement_count=len(measured_rows),
            fit_count=int(np.sum(fit)),
            holdout_count=int(np.sum(holdout)),
            initial_holdout_evidence=None,
            refined_holdout_evidence=None,
            initial_holdout_residual_px=None,
            refined_holdout_residual_px=None,
            translation_update_m=0.0,
            rotation_update_deg=0.0,
            failure_reason="insufficient_confident_measurements",
        )
    refined = _weighted_pose_fit(
        initial_pose_w2c=initial,
        xyz=anchors.xyz[measured_rows[fit]],
        xy=measured_xy[fit],
        weights=confidence[fit],
        camera=camera,
        config=policy,
    )
    if refined is None:
        return SurfaceFeatureRefinementResult(
            success=False,
            accepted=False,
            pose_w2c=initial,
            measurement_count=len(measured_rows),
            fit_count=int(np.sum(fit)),
            holdout_count=int(np.sum(holdout)),
            initial_holdout_evidence=None,
            refined_holdout_evidence=None,
            initial_holdout_residual_px=None,
            refined_holdout_residual_px=None,
            translation_update_m=0.0,
            rotation_update_deg=0.0,
            failure_reason="pose_optimization_failed",
        )
    translation_update, rotation_update = _pose_distance(
        initial,
        refined,
    )
    holdout_rows = measured_rows[holdout]
    initial_projection, _depth = _project(
        anchors.xyz[holdout_rows],
        initial,
        camera,
    )
    refined_projection, _depth = _project(
        anchors.xyz[holdout_rows],
        refined,
        camera,
    )
    initial_residual = float(
        np.median(
            np.linalg.norm(
                initial_projection - measured_xy[holdout],
                axis=1,
            )
        )
    )
    refined_residual = float(
        np.median(
            np.linalg.norm(
                refined_projection - measured_xy[holdout],
                axis=1,
            )
        )
    )
    initial_feature = _projected_feature_evidence(
        pose_w2c=initial,
        image_path=Path(image_path),
        image_id=str(image_id),
        camera=camera,
        anchor_rows=holdout_rows,
        descriptor_offsets=descriptor_offsets,
        descriptor_bank=descriptor_bank,
        anchors=anchors,
        alike=alike,
        view_mode_scores=dict(view_mode_scores or {}),
        config=policy,
    )
    refined_feature = _projected_feature_evidence(
        pose_w2c=refined,
        image_path=Path(image_path),
        image_id=str(image_id),
        camera=camera,
        anchor_rows=holdout_rows,
        descriptor_offsets=descriptor_offsets,
        descriptor_bank=descriptor_bank,
        anchors=anchors,
        alike=alike,
        view_mode_scores=dict(view_mode_scores or {}),
        config=policy,
    )
    within_trust_region = (
        translation_update
        <= float(policy.maximum_translation_update_m) + 1e-9
        and rotation_update
        <= float(policy.maximum_rotation_update_deg) + 1e-9
    )
    evidence_improves = (
        initial_feature is not None
        and refined_feature is not None
        and refined_feature
        >= initial_feature
        + float(policy.minimum_holdout_evidence_gain)
    )
    residual_improves = (
        initial_residual
        >= float(policy.minimum_initial_holdout_residual_px)
        and
        refined_residual
        <= initial_residual
        - float(policy.minimum_holdout_residual_gain_px)
    )
    accepted = bool(
        within_trust_region
        and evidence_improves
        and residual_improves
    )
    return SurfaceFeatureRefinementResult(
        success=True,
        accepted=accepted,
        pose_w2c=refined if accepted else initial,
        measurement_count=len(measured_rows),
        fit_count=int(np.sum(fit)),
        holdout_count=int(np.sum(holdout)),
        initial_holdout_evidence=initial_feature,
        refined_holdout_evidence=refined_feature,
        initial_holdout_residual_px=initial_residual,
        refined_holdout_residual_px=refined_residual,
        translation_update_m=translation_update,
        rotation_update_deg=rotation_update,
        failure_reason=(
            None if accepted else "heldout_acceptance_rejected"
        ),
    )
