"""Bounded factorized position--orientation initialization for pose basins.

The initializer addresses the dimensionality failure of a sparse joint SE(3)
cloud.  It first searches camera centres on a deterministic coarse-to-fine
world grid while preserving each retrieved seed orientation, then expands the
frozen orientation codebook only at a small beam of surviving centres.  It
does not consume a target pose, keypoints, correspondences, PnP, or absolute
pose regression.  The original retrieval seeds remain external protected
hypotheses; this module only proposes additional local-search initializers.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np

from .factorized_pose_basin import build_default_factorized_pose_basin


FACTORIZED_HIERARCHICAL_INITIALIZATION_SEMANTICS = (
    "protected_seed_world_position_beam_then_left_orientation_codebook_v1"
)


@dataclass(frozen=True)
class FactorizedPoseInitializationResult:
    poses_w2c: np.ndarray
    scores: np.ndarray
    source_seed_indices: np.ndarray
    evaluated_pose_count: int
    position_stage_pose_counts: tuple[int, ...]
    orientation_stage_pose_count: int
    position_beam_width_per_seed: int
    survivors_per_seed: int
    semantics: str = FACTORIZED_HIERARCHICAL_INITIALIZATION_SEMANTICS


def _centers_from_w2c(poses_w2c: np.ndarray) -> np.ndarray:
    pose = np.asarray(poses_w2c, dtype=np.float64)
    return (-np.swapaxes(pose[:, :3, :3], 1, 2) @ pose[:, :3, 3, None])[:, :, 0]


def _pose_from_center(rotation_w2c: np.ndarray, center_world: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_w2c
    result[:3, 3] = -rotation_w2c @ center_world
    return result


def _score_checked(
    score_batch: Callable[[np.ndarray], np.ndarray], poses: np.ndarray,
) -> np.ndarray:
    score = np.asarray(score_batch(poses), dtype=np.float64).reshape(-1)
    if score.shape != (poses.shape[0],) or np.any(~np.isfinite(score)):
        raise ValueError("factorized initializer scorer output differs from pose batch")
    return score


def _select_per_source(
    poses: np.ndarray,
    scores: np.ndarray,
    sources: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected: list[int] = []
    for source in range(int(np.max(sources, initial=-1)) + 1):
        rows = np.flatnonzero(sources == source)
        order = rows[np.lexsort((rows, -scores[rows]))]
        selected.extend(order[: int(count)].tolist())
    index = np.asarray(selected, dtype=np.int64)
    return poses[index], scores[index], sources[index]


def factorized_hierarchical_pose_initialization(
    seed_poses_w2c: np.ndarray,
    position_score_batch: Callable[[np.ndarray], np.ndarray],
    orientation_score_batch: Callable[[np.ndarray], np.ndarray],
    *,
    translation_half_extent_m: float = 8.0,
    coarse_translation_step_m: float = 4.0,
    refinement_translation_steps_m: tuple[float, ...] = (2.0, 1.0, 0.5),
    rotation_radius_deg: float = 45.0,
    position_beam_width_per_seed: int = 2,
    survivors_per_seed: int = 2,
) -> FactorizedPoseInitializationResult:
    """Return score-ranked factorized initializers inside every seed domain."""

    seeds = np.asarray(seed_poses_w2c, dtype=np.float64)
    extent = float(translation_half_extent_m)
    coarse = float(coarse_translation_step_m)
    steps = tuple(float(value) for value in refinement_translation_steps_m)
    position_beam = int(position_beam_width_per_seed)
    survivors = int(survivors_per_seed)
    rotation_radius = float(rotation_radius_deg)
    if (
        seeds.ndim != 3 or seeds.shape[1:] != (4, 4) or seeds.shape[0] == 0
        or np.any(~np.isfinite(seeds))
        or not np.isfinite(extent) or extent <= 0.0
        or not np.isfinite(coarse) or coarse <= 0.0
        or abs(extent / coarse - round(extent / coarse)) > 1.0e-12
        or not steps or any(not np.isfinite(value) or value <= 0.0 for value in steps)
        or any(steps[index] >= (coarse if index == 0 else steps[index - 1])
               for index in range(len(steps)))
        or not np.isfinite(rotation_radius) or not 0.0 < rotation_radius < 180.0
        or position_beam <= 0 or survivors <= 0
    ):
        raise ValueError("invalid factorized hierarchical initialization contract")

    seed_centers = _centers_from_w2c(seeds)
    rotations = seeds[:, :3, :3]
    coarse_axis = np.arange(-extent, extent + 0.5 * coarse, coarse, dtype=np.float64)
    coarse_offsets = np.asarray(list(product(coarse_axis, repeat=3)), dtype=np.float64)
    pose_rows: list[np.ndarray] = []
    source_rows: list[int] = []
    for source in range(seeds.shape[0]):
        for offset in coarse_offsets:
            pose_rows.append(_pose_from_center(rotations[source], seed_centers[source] + offset))
            source_rows.append(source)
    position_poses = np.asarray(pose_rows, dtype=np.float64)
    position_sources = np.asarray(source_rows, dtype=np.int64)
    position_scores = _score_checked(position_score_batch, position_poses)
    evaluated = int(position_poses.shape[0])
    stage_counts = [int(position_poses.shape[0])]
    position_poses, position_scores, position_sources = _select_per_source(
        position_poses, position_scores, position_sources, position_beam,
    )

    for step in steps:
        offsets = np.asarray(list(product((-step, 0.0, step), repeat=3)), dtype=np.float64)
        current_centers = _centers_from_w2c(position_poses)
        candidates: list[np.ndarray] = []
        sources: list[int] = []
        seen: set[tuple[int, float, float, float]] = set()
        for row, source in enumerate(position_sources.tolist()):
            for offset in offsets:
                center = current_centers[row] + offset
                if np.any(np.abs(center - seed_centers[source]) > extent + 1.0e-10):
                    continue
                key = (source, *np.round(center, 10).tolist())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(_pose_from_center(rotations[source], center))
                sources.append(source)
        candidate_pose = np.asarray(candidates, dtype=np.float64)
        candidate_source = np.asarray(sources, dtype=np.int64)
        candidate_score = _score_checked(position_score_batch, candidate_pose)
        evaluated += int(candidate_pose.shape[0])
        stage_counts.append(int(candidate_pose.shape[0]))
        position_poses, position_scores, position_sources = _select_per_source(
            candidate_pose, candidate_score, candidate_source, position_beam,
        )

    orientation = build_default_factorized_pose_basin().rotation_offsets_left
    orientation_angle = np.degrees(np.arccos(np.clip(
        (np.trace(orientation, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0,
    )))
    orientation = orientation[orientation_angle <= rotation_radius + 1.0e-8]
    if orientation.shape[0] == 0:
        raise AssertionError("factorized orientation domain lost its identity")
    position_centers = _centers_from_w2c(position_poses)
    orientation_poses: list[np.ndarray] = []
    orientation_sources: list[int] = []
    for row, source in enumerate(position_sources.tolist()):
        for rotation_offset in orientation:
            orientation_poses.append(_pose_from_center(
                rotation_offset @ rotations[source], position_centers[row],
            ))
            orientation_sources.append(source)
    final_pose = np.asarray(orientation_poses, dtype=np.float64)
    final_source = np.asarray(orientation_sources, dtype=np.int64)
    final_score = _score_checked(orientation_score_batch, final_pose)
    evaluated += int(final_pose.shape[0])
    final_pose, final_score, final_source = _select_per_source(
        final_pose, final_score, final_source, survivors,
    )
    return FactorizedPoseInitializationResult(
        poses_w2c=final_pose,
        scores=final_score,
        source_seed_indices=final_source,
        evaluated_pose_count=evaluated,
        position_stage_pose_counts=tuple(stage_counts),
        orientation_stage_pose_count=len(orientation_poses),
        position_beam_width_per_seed=position_beam,
        survivors_per_seed=survivors,
    )
