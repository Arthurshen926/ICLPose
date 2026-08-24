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


@dataclass(frozen=True)
class MultiBasinBeamSearchResult:
    basins: tuple[PoseBasinState, ...]
    evaluator_calls: int
    evaluated_pose_count: int
    completed_levels: int
    beam_width_per_source: int


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


def _poses_are_duplicate(
    left: np.ndarray,
    right: np.ndarray,
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> bool:
    left_rotation, right_rotation = left[:3, :3], right[:3, :3]
    left_center = -left_rotation.T @ left[:3, 3]
    right_center = -right_rotation.T @ right[:3, 3]
    translation = float(np.linalg.norm(left_center - right_center))
    relative = left_rotation @ right_rotation.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    rotation = float(np.degrees(np.arccos(cosine)))
    return bool(
        translation <= float(translation_threshold_m) + 1.0e-12
        and rotation <= float(rotation_threshold_deg) + 1.0e-10
    )


def batched_multibasin_beam_pattern_search(
    initial_poses_w2c: np.ndarray,
    score_batch: Callable[[np.ndarray], np.ndarray],
    *,
    translation_radii_m: Sequence[float] = (4.0, 2.0, 1.0, 0.5, 0.25),
    rotation_radii_deg: Sequence[float] = (22.5, 11.25, 5.625, 2.8125, 1.40625),
    beam_width_per_source: int = 4,
    duplicate_translation_m: float = 0.5,
    duplicate_rotation_deg: float = 5.0,
) -> MultiBasinBeamSearchResult:
    """Retain multiple score-diverse paths through a bounded SE(3) poll tree.

    The ordinary pattern search commits to one improving axis at each sweep.
    That is efficient inside a genuinely unimodal local basin, but it cannot
    traverse a coarse acquisition domain whose score contains false peaks.
    This bounded beam keeps up to ``beam_width_per_source`` physically
    distinct hypotheses for every original basin at every frozen scale.  It
    is still a heuristic search: no completeness or continuous-domain claim
    follows from the retained beam.
    """

    initial = np.asarray(initial_poses_w2c, dtype=np.float64)
    translation = np.asarray(translation_radii_m, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation_radii_deg, dtype=np.float64).reshape(-1)
    if initial.ndim != 3 or initial.shape[1:] != (4, 4) or initial.shape[0] == 0:
        raise ValueError("beam initial poses must have shape [basin,4,4]")
    if (
        translation.size == 0 or translation.shape != rotation.shape
        or np.any(~np.isfinite(translation)) or np.any(translation <= 0.0)
        or np.any(~np.isfinite(rotation)) or np.any(rotation <= 0.0)
        or np.any(translation[1:] >= translation[:-1])
        or np.any(rotation[1:] >= rotation[:-1])
        or int(beam_width_per_source) <= 0
        or not np.isfinite(duplicate_translation_m)
        or float(duplicate_translation_m) < 0.0
        or not np.isfinite(duplicate_rotation_deg)
        or float(duplicate_rotation_deg) < 0.0
    ):
        raise ValueError("beam-search schedule or budget is invalid")
    initial_score = np.asarray(score_batch(initial), dtype=np.float64).reshape(-1)
    if initial_score.shape != (initial.shape[0],) or np.any(~np.isfinite(initial_score)):
        raise ValueError("beam initial scorer output differs")
    states = [
        PoseBasinState(
            pose_w2c=initial[row].copy(), score=float(initial_score[row]),
            translation_radii_m=np.full(3, translation[0]),
            rotation_radii_deg=np.full(3, rotation[0]),
            source_basin_index=row, accepted_updates=0,
        )
        for row in range(initial.shape[0])
    ]
    evaluator_calls = 1
    evaluated = int(initial.shape[0])
    for level, (translation_radius, rotation_radius) in enumerate(
        zip(translation.tolist(), rotation.tolist())
    ):
        probes = np.concatenate([
            _probe_basin(state.pose_w2c, translation_radius, rotation_radius)[1:]
            for state in states
        ], axis=0)
        probe_score = np.asarray(score_batch(probes), dtype=np.float64).reshape(-1)
        if probe_score.shape != (probes.shape[0],) or np.any(~np.isfinite(probe_score)):
            raise ValueError("beam probe scorer output differs")
        evaluator_calls += 1
        evaluated += int(probes.shape[0])
        expanded = list(states)
        cursor = 0
        for state in states:
            for _ in range(12):
                expanded.append(PoseBasinState(
                    pose_w2c=probes[cursor].copy(), score=float(probe_score[cursor]),
                    translation_radii_m=np.full(3, translation_radius),
                    rotation_radii_deg=np.full(3, rotation_radius),
                    source_basin_index=state.source_basin_index,
                    accepted_updates=state.accepted_updates + 1,
                ))
                cursor += 1
        retained: list[PoseBasinState] = []
        for source in range(initial.shape[0]):
            rows = [state for state in expanded if state.source_basin_index == source]
            rows.sort(key=lambda state: (-state.score, state.accepted_updates))
            source_rows: list[PoseBasinState] = []
            for state in rows:
                if any(_poses_are_duplicate(
                    state.pose_w2c, previous.pose_w2c,
                    translation_threshold_m=float(duplicate_translation_m),
                    rotation_threshold_deg=float(duplicate_rotation_deg),
                ) for previous in source_rows):
                    continue
                source_rows.append(state)
                if len(source_rows) == int(beam_width_per_source):
                    break
            retained.extend(source_rows)
        states = retained
        if not states:
            raise RuntimeError(f"beam search lost every state at level {level}")
    states.sort(key=lambda state: (-state.score, state.source_basin_index))
    return MultiBasinBeamSearchResult(
        basins=tuple(states), evaluator_calls=evaluator_calls,
        evaluated_pose_count=evaluated, completed_levels=int(translation.size),
        beam_width_per_source=int(beam_width_per_source),
    )
