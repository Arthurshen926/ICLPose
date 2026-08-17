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
    translation_radii_m: np.ndarray
    rotation_radii_deg: np.ndarray
    source_basin_index: int
    accepted_updates: int

    @property
    def translation_radius_m(self) -> float:
        """Legacy diagnostic: largest still-active translation radius."""

        return float(np.max(self.translation_radii_m))

    @property
    def rotation_radius_deg(self) -> float:
        """Legacy diagnostic: largest still-active rotation radius."""

        return float(np.max(self.rotation_radii_deg))


@dataclass(frozen=True)
class MultiBasinPatternSearchResult:
    basins: tuple[PoseBasinState, ...]
    evaluator_calls: int
    evaluated_pose_count: int
    completed_sweeps: int
    pruning_applied_after_sweep: int | None


def _vector3(value: float | Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full((3,), float(array), dtype=np.float64)
    else:
        array = array.reshape(-1)
        if array.shape != (3,):
            raise ValueError(f"{name} must be a positive scalar or length-three vector")
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return array


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


def _axis_probe_pair(
    pose_w2c: np.ndarray,
    axis: int,
    translation_radii_m: np.ndarray,
    rotation_radii_deg: np.ndarray,
) -> np.ndarray:
    coordinate = np.zeros((6,), dtype=np.float64)
    coordinate[int(axis)] = 1.0
    translation_step = (
        float(translation_radii_m[int(axis)]) if int(axis) < 3 else 1.0
    )
    rotation_step = (
        float(rotation_radii_deg[int(axis) - 3]) if int(axis) >= 3 else 1.0
    )
    return np.asarray([
        left_retract_pose_w2c(
            pose_w2c, sign * coordinate,
            translation_step_m=translation_step,
            rotation_step_degrees=rotation_step,
        )
        for sign in (1.0, -1.0)
    ], dtype=np.float64)


def _prune_after_coarse_sweep(
    states: list[PoseBasinState],
    maximum_basins: int,
    location_ids: np.ndarray | None,
) -> list[PoseBasinState]:
    order = sorted(range(len(states)), key=lambda row: (-states[row].score, states[row].source_basin_index))
    if location_ids is None:
        return [states[row] for row in order[:maximum_basins]]

    selected: list[int] = []
    covered: set[int] = set()
    # First round: best post-coarse survivor from every atlas location.
    for row in order:
        location = int(location_ids[states[row].source_basin_index])
        if location not in covered:
            selected.append(row)
            covered.add(location)
    if len(selected) > int(maximum_basins):
        raise ValueError("maximum_basins cannot retain one survivor per location")
    selected_set = set(selected)
    for row in order:
        if len(selected) >= int(maximum_basins):
            break
        if row not in selected_set:
            selected.append(row)
            selected_set.add(row)
    selected.sort(key=lambda row: (-states[row].score, states[row].source_basin_index))
    return [states[row] for row in selected]


def batched_multibasin_pattern_search(
    initial_poses_w2c: np.ndarray,
    score_batch: Callable[[np.ndarray], np.ndarray],
    *,
    translation_radius_m: float | Sequence[float] = 1.0,
    rotation_radius_deg: float | Sequence[float] = 12.0,
    shrink_factor: float = 0.5,
    minimum_translation_radius_m: float | Sequence[float] = 0.0625,
    minimum_rotation_radius_deg: float | Sequence[float] = 0.75,
    maximum_sweeps: int = 8,
    maximum_basins: int | None = None,
    basin_location_ids: Sequence[int] | None = None,
    score_improvement_epsilon: float = 1e-8,
) -> MultiBasinPatternSearchResult:
    """Refine independent basins using cached-center, anisotropic probes.

    Each of the six tangent axes owns its trust radius.  A successful poll
    moves to its best improving direction without shrinking any other axis:
    those directions must be re-tested at the new center.  Radii shrink only
    when the *entire* poll fails.  The center is evaluated once initially and
    after that is never rendered redundantly.  Optional
    basin pruning occurs only after the first completed coarse sweep; with
    ``basin_location_ids`` it preserves at least one post-coarse survivor for
    every location.
    """

    initial = np.asarray(initial_poses_w2c, dtype=np.float64)
    if initial.ndim != 3 or initial.shape[1:] != (4, 4) or initial.shape[0] == 0:
        raise ValueError("initial poses must have shape [basin,4,4]")
    translation = _vector3(translation_radius_m, name="translation_radius_m")
    rotation = _vector3(rotation_radius_deg, name="rotation_radius_deg")
    minimum_translation = _vector3(
        minimum_translation_radius_m, name="minimum_translation_radius_m"
    )
    minimum_rotation = _vector3(
        minimum_rotation_radius_deg, name="minimum_rotation_radius_deg"
    )
    if (
        not np.isfinite(shrink_factor)
        or not 0.0 < float(shrink_factor) < 1.0
        or int(maximum_sweeps) <= 0
        or not np.isfinite(score_improvement_epsilon)
        or float(score_improvement_epsilon) < 0.0
    ):
        raise ValueError("invalid pattern-search trust-region configuration")
    if maximum_basins is not None and int(maximum_basins) <= 0:
        raise ValueError("maximum_basins must be positive")
    location_ids = None
    if basin_location_ids is not None:
        location_ids = np.asarray(basin_location_ids, dtype=np.int64).reshape(-1)
        if location_ids.shape != (initial.shape[0],):
            raise ValueError("basin_location_ids must align with initial poses")
        if maximum_basins is not None and np.unique(location_ids).size > int(maximum_basins):
            raise ValueError("maximum_basins cannot retain one survivor per location")

    initial_score = np.asarray(score_batch(initial), dtype=np.float64).reshape(-1)
    if initial_score.shape != (initial.shape[0],) or np.any(~np.isfinite(initial_score)):
        raise ValueError("initial scorer output differs from basin count")
    evaluator_calls = 1
    evaluated_pose_count = int(initial.shape[0])
    states = [
        PoseBasinState(
            pose_w2c=initial[index].copy(), score=float(initial_score[index]),
            translation_radii_m=translation.copy(),
            rotation_radii_deg=rotation.copy(),
            source_basin_index=index, accepted_updates=0,
        )
        for index in range(initial.shape[0])
    ]
    completed = 0
    pruning_after: int | None = None
    for _ in range(int(maximum_sweeps)):
        batches: list[np.ndarray] = []
        state_axes: list[list[int]] = []
        for state in states:
            active_axes = [
                axis for axis in range(6)
                if (
                    state.translation_radii_m[axis] >= minimum_translation[axis]
                    if axis < 3 else
                    state.rotation_radii_deg[axis - 3] >= minimum_rotation[axis - 3]
                )
            ]
            state_axes.append(active_axes)
            for axis in active_axes:
                batches.append(_axis_probe_pair(
                    state.pose_w2c, axis,
                    state.translation_radii_m, state.rotation_radii_deg,
                ))
        if not batches:
            break
        flat = np.concatenate(batches, axis=0)
        values = np.asarray(score_batch(flat), dtype=np.float64).reshape(-1)
        if values.shape != (flat.shape[0],) or np.any(~np.isfinite(values)):
            raise ValueError("probe scorer output differs from pose batch")
        evaluator_calls += 1
        evaluated_pose_count += int(flat.shape[0])

        cursor = 0
        updated: list[PoseBasinState] = []
        for state, active_axes in zip(states, state_axes):
            translation_next = state.translation_radii_m.copy()
            rotation_next = state.rotation_radii_deg.copy()
            best_score = state.score
            best_pose = state.pose_w2c
            failed_axes: list[int] = []
            for axis in active_axes:
                pair_pose = flat[cursor : cursor + 2]
                pair_score = values[cursor : cursor + 2]
                cursor += 2
                axis_best = int(np.argmax(pair_score))
                axis_improves = (
                    float(pair_score[axis_best])
                    > state.score + float(score_improvement_epsilon)
                )
                if not axis_improves:
                    failed_axes.append(axis)
                if float(pair_score[axis_best]) > best_score + float(score_improvement_epsilon):
                    best_score = float(pair_score[axis_best])
                    best_pose = pair_pose[axis_best].copy()
            improved = best_score > state.score + float(score_improvement_epsilon)
            # Standard poll semantics: a move changes the local landscape, so
            # failures measured at the old center cannot justify shrinking
            # non-selected axes.  Only a wholly unsuccessful poll contracts
            # its active trust radii.
            if not improved:
                for axis in failed_axes:
                    if axis < 3:
                        translation_next[axis] *= float(shrink_factor)
                    else:
                        rotation_next[axis - 3] *= float(shrink_factor)
            updated.append(PoseBasinState(
                pose_w2c=best_pose.copy() if improved else state.pose_w2c,
                score=best_score if improved else state.score,
                translation_radii_m=translation_next,
                rotation_radii_deg=rotation_next,
                source_basin_index=state.source_basin_index,
                accepted_updates=state.accepted_updates + int(improved),
            ))
        states = updated
        completed += 1
        if maximum_basins is not None and len(states) > int(maximum_basins):
            states = _prune_after_coarse_sweep(
                states, int(maximum_basins), location_ids
            )
            pruning_after = completed
            maximum_basins = None  # The one explicit budget application is final.

    states.sort(key=lambda state: (-state.score, state.source_basin_index))
    return MultiBasinPatternSearchResult(
        basins=tuple(states), evaluator_calls=evaluator_calls,
        evaluated_pose_count=evaluated_pose_count, completed_sweeps=completed,
        pruning_applied_after_sweep=pruning_after,
    )
