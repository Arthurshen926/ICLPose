"""Deterministic candidate-relative SE(3) supervision around a training pose."""

from __future__ import annotations

import numpy as np

from .se3_local_quadratic import AXIS_NAMES, left_retract_pose_w2c


LOCAL_POSE_SUPERVISION_SEMANTICS_V2 = (
    "gt_anchor_plus_left_camera_se3_axis_and_pair_multiscale_v2"
)
LOCAL_POSE_SUPERVISION_SEMANTICS = (
    "gt_anchor_plus_left_camera_se3_axis_pair_and_global_joint_v3"
)
TRANSLATION_RADII_M = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
ROTATION_RADII_DEG = (2.5, 5.0, 10.0, 20.0, 40.0)
PAIR_SCALES = (0.5, 1.0)
GLOBAL_JOINT_SAMPLE_PAIRS = 64


def _halton(index: int, base: int) -> float:
    value, fraction, row = 0.0, 1.0, int(index)
    while row > 0:
        fraction /= float(base)
        value += fraction * float(row % base)
        row //= base
    return value


def deterministic_global_joint_coordinates() -> np.ndarray:
    """Return 128 symmetric low-discrepancy coordinates in the 8 m/45° domain."""

    rows = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for sample in range(1, GLOBAL_JOINT_SAMPLE_PAIRS + 1):
        coordinate = np.zeros((6,), dtype=np.float64)
        coordinate[:3] = [
            2.0 * _halton(sample, base) - 1.0 for base in (2, 3, 5)
        ]
        z = 1.0 - 2.0 * (sample - 0.5) / GLOBAL_JOINT_SAMPLE_PAIRS
        radial = np.sqrt(max(0.0, 1.0 - z * z))
        axis = np.asarray([
            radial * np.cos(sample * golden_angle),
            radial * np.sin(sample * golden_angle), z,
        ])
        coordinate[3:] = axis * (_halton(sample, 7) ** (1.0 / 3.0))
        rows.extend((coordinate, -coordinate))
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (128, 6) or np.any(np.max(np.abs(result[:, :3]), axis=1) > 1.0):
        raise AssertionError("global joint supervision design differs")
    if np.any(np.linalg.norm(result[:, 3:], axis=1) > 1.0 + 1.0e-12):
        raise AssertionError("global joint rotation lies outside the frozen ball")
    return result


def complete_quadratic_candidate_indices() -> dict[str, int]:
    """Map the embedded normalized 1 m/10° complete design to row indices."""

    result = {"center": 0}
    translation_base = 1 + TRANSLATION_RADII_M.index(1.0) * 6
    for axis in range(3):
        result[AXIS_NAMES[axis] + "-"] = translation_base + 2 * axis
        result[AXIS_NAMES[axis] + "+"] = translation_base + 2 * axis + 1
    rotation_base = 1 + 6 * len(TRANSLATION_RADII_M) + ROTATION_RADII_DEG.index(10.0) * 6
    for local_axis, axis in enumerate(range(3, 6)):
        result[AXIS_NAMES[axis] + "-"] = rotation_base + 2 * local_axis
        result[AXIS_NAMES[axis] + "+"] = rotation_base + 2 * local_axis + 1
    pair_base = (
        1 + 6 * len(TRANSLATION_RADII_M) + 6 * len(ROTATION_RADII_DEG)
        + PAIR_SCALES.index(1.0) * 60
    )
    row = pair_base
    for left in range(6):
        for right in range(left + 1, 6):
            for left_label in ("-", "+"):
                for right_label in ("-", "+"):
                    result[
                        f"{AXIS_NAMES[left]}{left_label}_{AXIS_NAMES[right]}{right_label}"
                    ] = row
                    row += 1
    if len(result) != 73 or row != 187:
        raise AssertionError("local v2 complete quadratic design differs")
    return result


def build_local_pose_supervision_candidates(
    target_pose_w2c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GT plus signed multiscale translation and rotation probes.

    Every offset uses the same single left-camera SE(3) exponential as the
    actual pattern search.  In addition to signed axes, signed pairs at two
    scales expose translation--rotation and cross-axis interactions that a
    sequential search can otherwise exploit out of distribution.  No image
    evidence or learned score participates in candidate generation.
    """

    target = np.asarray(target_pose_w2c, dtype=np.float64).reshape(4, 4)
    if np.any(~np.isfinite(target)):
        raise ValueError("local supervision target pose must be finite")
    poses = [target.copy()]
    for radius in TRANSLATION_RADII_M:
        for axis in range(3):
            coordinate = np.zeros((6,), dtype=np.float64)
            coordinate[axis] = 1.0
            for sign in (-1.0, 1.0):
                poses.append(left_retract_pose_w2c(
                    target, sign * coordinate,
                    translation_step_m=float(radius),
                    rotation_step_degrees=1.0,
                ))
    for radius in ROTATION_RADII_DEG:
        for axis in range(3, 6):
            coordinate = np.zeros((6,), dtype=np.float64)
            coordinate[axis] = 1.0
            for sign in (-1.0, 1.0):
                poses.append(left_retract_pose_w2c(
                    target, sign * coordinate,
                    translation_step_m=1.0,
                    rotation_step_degrees=float(radius),
                ))
    for scale in PAIR_SCALES:
        for left in range(6):
            for right in range(left + 1, 6):
                for left_sign in (-1.0, 1.0):
                    for right_sign in (-1.0, 1.0):
                        coordinate = np.zeros((6,), dtype=np.float64)
                        coordinate[left] = left_sign * float(scale)
                        coordinate[right] = right_sign * float(scale)
                        poses.append(left_retract_pose_w2c(
                            target, coordinate,
                            translation_step_m=1.0,
                            rotation_step_degrees=10.0,
                        ))
    for coordinate in deterministic_global_joint_coordinates():
        poses.append(left_retract_pose_w2c(
            target, coordinate,
            translation_step_m=8.0,
            rotation_step_degrees=45.0,
        ))
    pose_array = np.asarray(poses, dtype=np.float64)
    rotations = pose_array[:, :3, :3]
    target_rotation = target[:3, :3]
    centers = (-np.swapaxes(rotations, 1, 2) @ pose_array[:, :3, 3, None])[:, :, 0]
    target_center = -target_rotation.T @ target[:3, 3]
    translation_error = np.linalg.norm(centers - target_center[None], axis=1)
    relative = rotations @ target_rotation.T
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error = np.degrees(np.arccos(cosine))
    # Row zero is the exact input matrix.  Remove trace/arccos roundoff so the
    # serialized supervision preserves the mathematical anchor identity.
    translation_error[0] = 0.0
    rotation_error[0] = 0.0
    return (
        pose_array,
        np.asarray(translation_error, dtype=np.float32),
        np.asarray(rotation_error, dtype=np.float32),
    )
