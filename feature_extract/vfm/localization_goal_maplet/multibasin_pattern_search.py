"""Batched, derivative-free multi-basin trust-region search on SE(3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .se3_local_quadratic import left_retract_pose_w2c


@dataclass(frozen=True)
class PoseBasinState:
    pose_w2c: np.ndarray
    score: float
    translation_radius_m: float
    rotation_radius_deg: float
    source_basin_index: int
    accepted_updates: int


@dataclass(frozen=True)
class MultiBasinPatternSearchResult:
    basins: tuple[PoseBasinState, ...]
    evaluator_calls: int
    evaluated_pose_count: int
    completed_sweeps: int


def _probe_basin(
    pose_w2c: np.ndarray,
    translation_radius_m: float,
    rotation_radius_deg: float,
) -> np.ndarray:
    """Return center and +/- six axes under one left-SE(3) retraction."""

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    probes = [pose]
    for axis in np.eye(6, dtype=np.float64):
        probes.append(left_retract_pose_w2c(
            pose, axis,
            translation_step_m=float(translation_radius_m),
            rotation_step_degrees=float(rotation_radius_deg),
        ))
        probes.append(left_retract_pose_w2c(
            pose, -axis,
            translation_step_m=float(translation_radius_m),
            rotation_step_degrees=float(rotation_radius_deg),
        ))
    return np.asarray(probes, dtype=np.float64)


def batched_multibasin_pattern_search(
    initial_poses_w2c: np.ndarray,
    score_batch: Callable[[np.ndarray], np.ndarray],
    *,
    translation_radius_m: float = 1.0,
    rotation_radius_deg: float = 12.0,
    shrink_factor: float = 0.5,
    minimum_translation_radius_m: float = 0.0625,
    minimum_rotation_radius_deg: float = 0.75,
    maximum_sweeps: int = 8,
    maximum_basins: int | None = None,
    score_improvement_epsilon: float = 1e-8,
) -> MultiBasinPatternSearchResult:
    """Refine all basins with one batched 13-probe evaluator call per sweep.

    Every basin survives until the optional explicit basin budget is applied;
    a temporarily weaker mode is never silently replaced by another mode's
    probe.  The scorer is maximize-oriented and may be non-differentiable.
    """

    initial = np.asarray(initial_poses_w2c, dtype=np.float64)
    if initial.ndim != 3 or initial.shape[1:] != (4, 4) or initial.shape[0] == 0:
        raise ValueError("initial poses must have shape [basin,4,4]")
    scalars = (
        translation_radius_m,
        rotation_radius_deg,
        shrink_factor,
        minimum_translation_radius_m,
        minimum_rotation_radius_deg,
    )
    if (
        any(not np.isfinite(value) or float(value) <= 0.0 for value in scalars)
        or not 0.0 < float(shrink_factor) < 1.0
        or int(maximum_sweeps) <= 0
    ):
        raise ValueError("invalid pattern-search trust-region configuration")
    initial_score = np.asarray(score_batch(initial), dtype=np.float64).reshape(-1)
    if initial_score.shape != (initial.shape[0],) or np.any(~np.isfinite(initial_score)):
        raise ValueError("initial scorer output differs from basin count")
    evaluator_calls = 1
    evaluated_pose_count = int(initial.shape[0])
    states = [
        PoseBasinState(
            pose_w2c=initial[index].copy(), score=float(initial_score[index]),
            translation_radius_m=float(translation_radius_m),
            rotation_radius_deg=float(rotation_radius_deg),
            source_basin_index=index, accepted_updates=0,
        )
        for index in range(initial.shape[0])
    ]
    if maximum_basins is not None and len(states) > int(maximum_basins):
        order = sorted(range(len(states)), key=lambda i: (-states[i].score, i))
        states = [states[index] for index in order[: int(maximum_basins)]]
    completed = 0
    for _ in range(int(maximum_sweeps)):
        active = [
            state.translation_radius_m >= float(minimum_translation_radius_m)
            or state.rotation_radius_deg >= float(minimum_rotation_radius_deg)
            for state in states
        ]
        if not any(active):
            break
        probe_batches = [
            _probe_basin(
                state.pose_w2c,
                max(state.translation_radius_m, float(minimum_translation_radius_m)),
                max(state.rotation_radius_deg, float(minimum_rotation_radius_deg)),
            )
            for state, enabled in zip(states, active) if enabled
        ]
        flat = np.concatenate(probe_batches, axis=0)
        value = np.asarray(score_batch(flat), dtype=np.float64).reshape(-1)
        if value.shape != (flat.shape[0],) or np.any(~np.isfinite(value)):
            raise ValueError("probe scorer output differs from pose batch")
        evaluator_calls += 1
        evaluated_pose_count += int(flat.shape[0])
        cursor = 0
        updated: list[PoseBasinState] = []
        for state, enabled in zip(states, active):
            if not enabled:
                updated.append(state)
                continue
            local_pose = flat[cursor : cursor + 13]
            local_score = value[cursor : cursor + 13]
            cursor += 13
            best = int(np.argmax(local_score))
            improved = float(local_score[best]) > state.score + float(score_improvement_epsilon)
            updated.append(PoseBasinState(
                pose_w2c=(local_pose[best].copy() if improved else state.pose_w2c),
                score=(float(local_score[best]) if improved else state.score),
                translation_radius_m=(
                    state.translation_radius_m
                    if improved else state.translation_radius_m * float(shrink_factor)
                ),
                rotation_radius_deg=(
                    state.rotation_radius_deg
                    if improved else state.rotation_radius_deg * float(shrink_factor)
                ),
                source_basin_index=state.source_basin_index,
                accepted_updates=state.accepted_updates + int(improved),
            ))
        states = updated
        completed += 1
    states.sort(key=lambda state: (-state.score, state.source_basin_index))
    return MultiBasinPatternSearchResult(
        basins=tuple(states), evaluator_calls=evaluator_calls,
        evaluated_pose_count=evaluated_pose_count, completed_sweeps=completed,
    )
