"""Derivative-free SE(3) refinement under a pose-conditioned VFM score.

The refiner deliberately knows nothing about point correspondences.  Its
objective is supplied by the caller and is evaluated on a fixed query grid;
the production caller uses the single canonical primitive VFM field.  A
bounded coordinate trust region is useful here because VFM token features are
too coarse for a stable photometric-style Jacobian, while batched primitive
rendering makes a small, explicit SE(3) stencil inexpensive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from feature_extract.vfm.localization_v6.se3_update import se3_exp


@dataclass(frozen=True)
class PrimitiveVFMRefinementResult:
    pose_w2c: np.ndarray
    initial_score: float
    final_score: float
    accepted_steps: int
    history: tuple[dict[str, float | int | bool], ...]


def refine_pose_with_primitive_vfm_score(
    initial_pose_w2c: np.ndarray,
    score_poses: Callable[[np.ndarray], np.ndarray],
    *,
    translation_steps_m: Sequence[float] = (0.60, 0.40, 0.25, 0.12),
    rotation_steps_deg: Sequence[float] = (5.0, 3.0, 2.0, 1.0),
    iterations_per_scale: int = 2,
    minimum_score_improvement: float = 1.0e-6,
) -> PrimitiveVFMRefinementResult:
    """Maximize a batched fixed-denominator score in a bounded SE(3) stencil.

    Updates are left-multiplicative camera-frame twists.  At every iteration
    the current pose and both signs of all six coordinate axes are evaluated
    in one batch.  Only a strictly improving update is accepted, so the
    returned objective can never regress even when the VFM surface is flat.
    """

    translation = tuple(float(value) for value in translation_steps_m)
    rotation = tuple(float(value) for value in rotation_steps_deg)
    if len(translation) != len(rotation) or not translation:
        raise ValueError("translation and rotation trust-region schedules differ")
    if any(value <= 0.0 for value in translation + rotation):
        raise ValueError("trust-region steps must be positive")
    if int(iterations_per_scale) <= 0:
        raise ValueError("iterations per trust-region scale must be positive")

    pose = np.asarray(initial_pose_w2c, dtype=np.float64).reshape(4, 4).copy()
    initial = np.asarray(score_poses(pose[None]), dtype=np.float64).reshape(-1)
    if initial.shape != (1,) or not np.isfinite(initial[0]):
        raise ValueError("initial primitive VFM score is not finite")
    initial_score = current_score = float(initial[0])
    accepted_steps = 0
    history: list[dict[str, float | int | bool]] = []
    for scale_index, (translation_step, rotation_step) in enumerate(
        zip(translation, rotation)
    ):
        for iteration in range(int(iterations_per_scale)):
            deltas = [np.zeros((6,), dtype=np.float64)]
            for axis in range(6):
                magnitude = (
                    np.deg2rad(rotation_step) if axis < 3 else translation_step
                )
                for sign in (-1.0, 1.0):
                    delta = np.zeros((6,), dtype=np.float64)
                    delta[axis] = sign * magnitude
                    deltas.append(delta)
            candidates = np.stack([se3_exp(delta) @ pose for delta in deltas])
            scores = np.asarray(score_poses(candidates), dtype=np.float64).reshape(-1)
            if scores.shape != (len(candidates),) or np.any(~np.isfinite(scores)):
                raise ValueError("primitive VFM trust-region scores are invalid")
            selected = int(np.argmax(scores))
            selected_score = float(scores[selected])
            accepted = bool(
                selected > 0
                and selected_score >= current_score + float(minimum_score_improvement)
            )
            if accepted:
                pose = candidates[selected]
                current_score = selected_score
                accepted_steps += 1
            history.append(
                {
                    "scale_index": int(scale_index),
                    "iteration": int(iteration),
                    "translation_step_m": float(translation_step),
                    "rotation_step_deg": float(rotation_step),
                    "selected_stencil_index": int(selected),
                    "candidate_score": float(selected_score),
                    "accepted": accepted,
                    "score": float(current_score),
                }
            )
    return PrimitiveVFMRefinementResult(
        pose_w2c=pose,
        initial_score=float(initial_score),
        final_score=float(current_score),
        accepted_steps=int(accepted_steps),
        history=tuple(history),
    )
