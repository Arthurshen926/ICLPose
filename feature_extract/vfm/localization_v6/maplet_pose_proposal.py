"""Coarse pose modes from probabilistic query-region/maplet groups."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from feature_extract.vfm.colmap_tracks import ColmapCamera
from feature_extract.vfm.localization.continuous_surface_alignment import (
    project_world_points,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    QueryMapletGroup,
)
from feature_extract.vfm.localization.surface_retrieval_maplets import (
    SurfaceRetrievalMapletBank,
)
from feature_extract.vfm.query_to_3d_matching import camera_matrix_and_distortion


@dataclass(frozen=True)
class MapletPoseHypothesis:
    pose_w2c: np.ndarray
    score: float
    supporting_group_count: int
    sampled_maplet_ids: np.ndarray


def _score_hypothesis(
    pose: np.ndarray,
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
    camera: ColmapCamera,
) -> tuple[float, int]:
    row_by_id = {
        int(value): int(row) for row, value in enumerate(bank.maplet_ids.tolist())
    }
    total = 0.0
    support = 0
    for group in groups:
        rows = np.asarray(
            [row_by_id.get(int(value), -1) for value in group.maplet_ids],
            dtype=np.int64,
        )
        valid = rows >= 0
        if not np.any(valid):
            total += np.log(max(float(group.null_probability), 1e-8))
            continue
        pixels, depth = project_world_points(
            bank.centers[rows[valid]], pose, camera
        )
        residual = pixels - np.asarray(group.query_region_xy)[None]
        maplet_radius = np.max(bank.extents[rows[valid], :2], axis=1)
        projected_radius = (
            float(camera.params[0])
            * maplet_radius
            / np.maximum(depth, 1e-3)
        )
        region_radius = float(np.linalg.norm(group.query_region_extent))
        sigma = np.maximum(projected_radius + region_radius, 2.0)
        likelihood = np.asarray(group.probabilities)[valid] * np.exp(
            -0.5 * np.sum(residual * residual, axis=1) / (sigma * sigma)
        )
        mass = float(np.sum(likelihood))
        combined = mass + float(group.null_probability) * 0.1
        total += np.log(max(combined, 1e-8))
        if mass > float(group.null_probability) * 0.1:
            support += 1
    return total, support


def propose_maplet_center_poses(
    groups: tuple[QueryMapletGroup, ...],
    bank: SurfaceRetrievalMapletBank,
    camera: ColmapCamera,
    *,
    trials: int = 2048,
    sample_size: int = 6,
    maximum_modes: int = 16,
    seed: int = 73,
) -> tuple[MapletPoseHypothesis, ...]:
    """Grouped probabilistic PnP for basin entry, never final precision."""

    usable = tuple(group for group in groups if group.maplet_ids.size)
    if len(usable) < max(int(sample_size), 4):
        return tuple()
    matrix, distortion = camera_matrix_and_distortion(camera)
    row_by_id = {
        int(value): int(row) for row, value in enumerate(bank.maplet_ids.tolist())
    }
    rng = np.random.default_rng(int(seed))
    hypotheses: list[MapletPoseHypothesis] = []
    for _ in range(int(trials)):
        selected_groups = rng.choice(
            len(usable), size=int(sample_size), replace=False
        )
        object_points = []
        image_points = []
        sampled_ids = []
        for group_row in selected_groups.tolist():
            group = usable[group_row]
            probability = np.asarray(group.probabilities, dtype=np.float64)
            probability /= max(float(np.sum(probability)), 1e-12)
            choice = int(rng.choice(group.maplet_ids.size, p=probability))
            maplet_id = int(group.maplet_ids[choice])
            row = row_by_id.get(maplet_id)
            if row is None or maplet_id in sampled_ids:
                break
            sampled_ids.append(maplet_id)
            object_points.append(bank.centers[row])
            image_points.append(group.query_region_xy)
        if len(object_points) < int(sample_size):
            continue
        success, rotation, translation = cv2.solvePnP(
            np.asarray(object_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
            matrix,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            continue
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cv2.Rodrigues(rotation)[0]
        pose[:3, 3] = translation.reshape(3)
        score, support = _score_hypothesis(pose, usable, bank, camera)
        hypotheses.append(
            MapletPoseHypothesis(
                pose_w2c=pose,
                score=score,
                supporting_group_count=support,
                sampled_maplet_ids=np.asarray(sampled_ids, dtype=np.int64),
            )
        )
    hypotheses.sort(key=lambda item: item.score, reverse=True)
    modes: list[MapletPoseHypothesis] = []
    for candidate in hypotheses:
        center = -candidate.pose_w2c[:3, :3].T @ candidate.pose_w2c[:3, 3]
        duplicate = False
        for kept in modes:
            kept_center = -kept.pose_w2c[:3, :3].T @ kept.pose_w2c[:3, 3]
            rotation = candidate.pose_w2c[:3, :3] @ kept.pose_w2c[:3, :3].T
            angle = np.degrees(
                np.arccos(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
            )
            if np.linalg.norm(center - kept_center) < 0.20 and angle < 2.0:
                duplicate = True
                break
        if not duplicate:
            modes.append(candidate)
        if len(modes) >= int(maximum_modes):
            break
    return tuple(modes)
