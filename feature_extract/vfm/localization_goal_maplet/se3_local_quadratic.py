"""Finite-difference 6D local quadratic audit for a pose score.

Coordinates are normalized by caller-selected translation/rotation steps, so
translation--rotation cross terms and conditioning are explicit rather than
being hidden by incompatible physical units.  The fitted Hessian is for
``loss = -score``; positive eigenvalues therefore mean a locally concave pose
score in the complete six-dimensional neighborhood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


AXIS_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")


@dataclass(frozen=True)
class LocalSE3Quadratic:
    loss_gradient: np.ndarray
    loss_hessian: np.ndarray
    hessian_eigenvalues: np.ndarray
    hessian_condition_number: float
    predicted_bias_normalized: np.ndarray
    positive_definite: bool
    fit_root_mean_square_error: float
    fit_maximum_absolute_error: float


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def left_retract_pose_w2c(
    pose_w2c: np.ndarray,
    coordinate: np.ndarray,
    *,
    translation_step_m: float,
    rotation_step_degrees: float,
) -> np.ndarray:
    """Apply one coupled left SE(3) exponential to a world-to-camera pose.

    ``coordinate`` is dimensionless ``[rho_x,rho_y,rho_z,omega_x,...]``.
    Translation is scaled in metres and rotation in radians before a *single*
    exponential is left-multiplied onto ``pose_w2c``.  This avoids the former
    order-dependent world-translation/local-rotation probe construction.
    """

    pose = np.asarray(pose_w2c, dtype=np.float64)
    value = np.asarray(coordinate, dtype=np.float64).reshape(6)
    if pose.shape != (4, 4) or np.any(~np.isfinite(pose)) or np.any(~np.isfinite(value)):
        raise ValueError("SE(3) retraction inputs must be finite")
    if not np.array_equal(pose[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise ValueError("pose_w2c must be a homogeneous transform")
    if float(translation_step_m) <= 0.0 or float(rotation_step_degrees) <= 0.0:
        raise ValueError("SE(3) coordinate scales must be positive")
    rotation = pose[:3, :3]
    if (
        np.linalg.norm(rotation @ rotation.T - np.eye(3)) > 1e-7
        or np.linalg.det(rotation) <= 0.0
    ):
        raise ValueError("pose_w2c must contain a proper rotation")
    rho = value[:3] * float(translation_step_m)
    omega = value[3:] * np.radians(float(rotation_step_degrees))
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    omega_hat2 = omega_hat @ omega_hat
    if theta < 1e-8:
        # Stable series through fourth order.  This is also exact at omega=0.
        rotation_delta = np.eye(3) + omega_hat + 0.5 * omega_hat2
        left_jacobian = np.eye(3) + 0.5 * omega_hat + (1.0 / 6.0) * omega_hat2
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
        c = (theta - np.sin(theta)) / (theta * theta * theta)
        rotation_delta = np.eye(3) + a * omega_hat + b * omega_hat2
        left_jacobian = np.eye(3) + b * omega_hat + c * omega_hat2
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = rotation_delta
    delta[:3, 3] = left_jacobian @ rho
    result = delta @ pose
    result[3] = np.asarray([0.0, 0.0, 0.0, 1.0])
    return result


def complete_quadratic_probe_coordinates() -> dict[str, np.ndarray]:
    """Return the 73 central-difference points needed for a full 6D Hessian."""

    probes = {"center": np.zeros((6,), dtype=np.float64)}
    eye = np.eye(6, dtype=np.float64)
    for axis, name in enumerate(AXIS_NAMES):
        probes[name + "+"] = eye[axis]
        probes[name + "-"] = -eye[axis]
    for left in range(6):
        for right in range(left + 1, 6):
            for left_sign, left_label in ((1.0, "+"), (-1.0, "-")):
                for right_sign, right_label in ((1.0, "+"), (-1.0, "-")):
                    coordinate = np.zeros((6,), dtype=np.float64)
                    coordinate[left] = left_sign
                    coordinate[right] = right_sign
                    probes[
                        f"{AXIS_NAMES[left]}{left_label}_{AXIS_NAMES[right]}{right_label}"
                    ] = coordinate
    if len(probes) != 73:
        raise AssertionError("complete 6D central-difference design must have 73 probes")
    return probes


def minimal_quadratic_probe_coordinates() -> dict[str, np.ndarray]:
    """Return a 28-point exactly determined quadratic design.

    This uses the center, both signs of each axis, and one positive corner for
    every axis pair.  It is cheaper than the 73-point symmetric design but more
    sensitive to cubic terms, so reports must identify it as a minimal audit.
    """

    probes = {"center": np.zeros((6,), dtype=np.float64)}
    eye = np.eye(6, dtype=np.float64)
    for axis, name in enumerate(AXIS_NAMES):
        probes[name + "+"] = eye[axis]
        probes[name + "-"] = -eye[axis]
    for left in range(6):
        for right in range(left + 1, 6):
            probes[f"{AXIS_NAMES[left]}+_{AXIS_NAMES[right]}+"] = (
                eye[left] + eye[right]
            )
    if len(probes) != 28:
        raise AssertionError("minimal 6D quadratic design must have 28 probes")
    return probes


def fit_local_se3_quadratic_least_squares(
    coordinates: Mapping[str, np.ndarray],
    scores: Mapping[str, float],
    *,
    eigenvalue_tolerance: float = 1e-9,
) -> LocalSE3Quadratic:
    """Fit ``loss=-score`` from any full-rank normalized 6D design."""

    if set(coordinates) != set(scores) or len(scores) < 28:
        raise ValueError("quadratic coordinates/scores differ or are underdetermined")
    design = []
    target = []
    for name in coordinates:
        x = np.asarray(coordinates[name], dtype=np.float64).reshape(6)
        score = float(scores[name])
        if np.any(~np.isfinite(x)) or not np.isfinite(score):
            raise ValueError("quadratic observations must be finite")
        row = [1.0, *x.tolist()]
        row.extend((0.5 * x * x).tolist())
        row.extend(
            float(x[left] * x[right])
            for left in range(6) for right in range(left + 1, 6)
        )
        design.append(row)
        target.append(-score)
    matrix = np.asarray(design, dtype=np.float64)
    if np.linalg.matrix_rank(matrix) != 28:
        raise ValueError("quadratic probe design is rank deficient")
    coefficient, _, _, _ = np.linalg.lstsq(
        matrix, np.asarray(target, dtype=np.float64), rcond=None
    )
    gradient = coefficient[1:7]
    hessian = np.zeros((6, 6), dtype=np.float64)
    hessian[np.diag_indices(6)] = coefficient[7:13]
    offset = 13
    for left in range(6):
        for right in range(left + 1, 6):
            hessian[left, right] = hessian[right, left] = coefficient[offset]
            offset += 1
    eigenvalues = np.linalg.eigvalsh(hessian)
    positive = bool(np.all(eigenvalues > float(eigenvalue_tolerance)))
    condition = float(eigenvalues[-1] / eigenvalues[0]) if positive else float("inf")
    predicted = (
        -np.linalg.solve(hessian, gradient)
        if positive else np.full((6,), np.nan, dtype=np.float64)
    )
    fitted_target = matrix @ coefficient
    residual = fitted_target - np.asarray(target, dtype=np.float64)
    return LocalSE3Quadratic(
        loss_gradient=gradient,
        loss_hessian=hessian,
        hessian_eigenvalues=eigenvalues,
        hessian_condition_number=condition,
        predicted_bias_normalized=predicted,
        positive_definite=positive,
        fit_root_mean_square_error=float(np.sqrt(np.mean(residual * residual))),
        fit_maximum_absolute_error=float(np.max(np.abs(residual))),
    )


def fit_complete_local_se3_quadratic(
    scores: Mapping[str, float],
    *,
    eigenvalue_tolerance: float = 1e-9,
) -> LocalSE3Quadratic:
    """Fit the exact central-difference quadratic and report ``H(-score)``."""

    expected = complete_quadratic_probe_coordinates()
    if set(scores) != set(expected):
        missing = sorted(set(expected) - set(scores))
        extra = sorted(set(scores) - set(expected))
        raise ValueError(f"quadratic probe set differs; missing={missing}, extra={extra}")
    value = {key: float(scores[key]) for key in expected}
    if any(not np.isfinite(item) for item in value.values()):
        raise ValueError("quadratic scores must be finite")
    center = value["center"]
    score_gradient = np.zeros((6,), dtype=np.float64)
    score_hessian = np.zeros((6, 6), dtype=np.float64)
    for axis, name in enumerate(AXIS_NAMES):
        plus, minus = value[name + "+"], value[name + "-"]
        score_gradient[axis] = 0.5 * (plus - minus)
        score_hessian[axis, axis] = plus - 2.0 * center + minus
    for left in range(6):
        for right in range(left + 1, 6):
            prefix = (AXIS_NAMES[left], AXIS_NAMES[right])
            cross = 0.25 * (
                value[f"{prefix[0]}+_{prefix[1]}+"]
                - value[f"{prefix[0]}+_{prefix[1]}-"]
                - value[f"{prefix[0]}-_{prefix[1]}+"]
                + value[f"{prefix[0]}-_{prefix[1]}-"]
            )
            score_hessian[left, right] = score_hessian[right, left] = cross
    loss_gradient = -score_gradient
    loss_hessian = -0.5 * (score_hessian + score_hessian.T)
    eigenvalues = np.linalg.eigvalsh(loss_hessian)
    positive = bool(np.all(eigenvalues > float(eigenvalue_tolerance)))
    condition = (
        float(eigenvalues[-1] / eigenvalues[0])
        if positive
        else float("inf")
    )
    predicted = (
        -np.linalg.solve(loss_hessian, loss_gradient)
        if positive
        else np.full((6,), np.nan, dtype=np.float64)
    )
    return LocalSE3Quadratic(
        loss_gradient=loss_gradient,
        loss_hessian=loss_hessian,
        hessian_eigenvalues=eigenvalues,
        hessian_condition_number=condition,
        predicted_bias_normalized=predicted,
        positive_definite=positive,
        fit_root_mean_square_error=0.0,
        fit_maximum_absolute_error=0.0,
    )
