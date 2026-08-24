"""Deterministic GT-relative SE(3) stencils for pose-energy diagnostics.

The stencils in this module are supervision instruments, not deployable pose
proposals.  A candidate is always obtained with the repository convention

``T_candidate = Exp(delta_camera) @ T_target_w2c``.

The quadratic-complete stencil evaluates every signed radial direction at two
scales.  Its six coordinate directions and fifteen pair-sum directions make
the outer-product design span all 21 entries of a symmetric 6x6 local
curvature matrix.  This is the minimum structural property needed to rule out
an unobserved SE(3) axis or pairwise coupling in a locally quadratic energy.
It does not, by itself, establish basin capture or end-to-end localization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from feature_extract.vfm.localization_v6.se3_update import se3_exp


QUADRATIC_COMPLETE_6DOF_STENCIL_SEMANTICS = (
    "gt_relative_medium_6dof_quadratic_complete_radial_stencil_v1"
)
CONTROLLED_MEDIUM_6DOF_CANDIDATE_SEMANTICS = (
    "controlled_medium_6dof_quadratic_oracle_v1"
)
TWIST_ORDER = ("r_x", "r_y", "r_z", "t_x", "t_y", "t_z")


@dataclass(frozen=True)
class ControlledPoseStencil:
    """Immutable candidate twists and their radial-path supervision contract."""

    twists_left_camera: np.ndarray
    normalized_directions: np.ndarray
    candidate_direction_ids: np.ndarray
    candidate_signs: np.ndarray
    candidate_radius_fractions: np.ndarray
    radial_paths: np.ndarray
    direction_axis_pairs: np.ndarray
    translation_radius_m: float
    rotation_radius_deg: float
    semantics: str
    content_sha256: str

    @property
    def candidate_count(self) -> int:
        return int(self.twists_left_camera.shape[0])

    @property
    def direction_count(self) -> int:
        return int(self.normalized_directions.shape[0])

    @property
    def radial_path_count(self) -> int:
        return int(self.radial_paths.shape[0])


def _content_sha256(
    *,
    twists: np.ndarray,
    directions: np.ndarray,
    candidate_direction_ids: np.ndarray,
    candidate_signs: np.ndarray,
    candidate_radius_fractions: np.ndarray,
    radial_paths: np.ndarray,
    direction_axis_pairs: np.ndarray,
    translation_radius_m: float,
    rotation_radius_deg: float,
    semantics: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(semantics).encode("utf-8") + b"\0")
    for scalar in (translation_radius_m, rotation_radius_deg):
        digest.update(np.asarray(float(scalar), dtype=np.float64).tobytes())
    for name, value in (
        ("twists_left_camera", twists),
        ("normalized_directions", directions),
        ("candidate_direction_ids", candidate_direction_ids),
        ("candidate_signs", candidate_signs),
        ("candidate_radius_fractions", candidate_radius_fractions),
        ("radial_paths", radial_paths),
        ("direction_axis_pairs", direction_axis_pairs),
    ):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _quadratic_design_rank(directions: np.ndarray) -> int:
    """Return rank of ``d d^T`` in the 21D symmetric-matrix basis."""

    value = np.asarray(directions, dtype=np.float64).reshape(-1, 6)
    upper = np.triu_indices(6)
    design = np.stack([np.outer(row, row)[upper] for row in value], axis=0)
    return int(np.linalg.matrix_rank(design, tol=1.0e-10))


def build_medium_quadratic_complete_6dof_stencil() -> ControlledPoseStencil:
    """Return an 85-candidate, locally quadratic-complete medium stencil.

    Candidate zero is the GT anchor.  The remaining candidates form 42
    independent signed radial paths ``GT -> 0.5 radius -> 1.0 radius``:

    * six coordinate directions cover every SE(3) tangent axis;
    * fifteen pair-sum directions expose every symmetric cross-axis term;
    * both signs prevent a one-sided slope from masquerading as curvature.

    Rotational and translational sub-vectors are normalized separately to the
    medium-stage radii (15 degrees and 1 metre).  Hence a mixed rotation--
    translation direction reaches both declared radii at fraction one, while
    a same-modality pair has unit norm within that modality.
    """

    translation_radius_m = 1.0
    rotation_radius_deg = 15.0
    rotation_radius_rad = np.deg2rad(rotation_radius_deg)
    radius_fractions = (0.5, 1.0)

    raw_directions: list[np.ndarray] = []
    axis_pairs: list[tuple[int, int]] = []
    for axis in range(6):
        direction = np.zeros(6, dtype=np.float64)
        direction[axis] = 1.0
        raw_directions.append(direction)
        axis_pairs.append((axis, -1))
    for first in range(6):
        for second in range(first + 1, 6):
            direction = np.zeros(6, dtype=np.float64)
            direction[first] = 1.0
            direction[second] = 1.0
            raw_directions.append(direction)
            axis_pairs.append((first, second))

    normalized_rows = []
    twist_direction_rows = []
    for raw in raw_directions:
        normalized = np.zeros(6, dtype=np.float64)
        twist = np.zeros(6, dtype=np.float64)
        rotation_norm = float(np.linalg.norm(raw[:3]))
        translation_norm = float(np.linalg.norm(raw[3:]))
        if rotation_norm > 0.0:
            normalized[:3] = raw[:3] / rotation_norm
            twist[:3] = normalized[:3] * rotation_radius_rad
        if translation_norm > 0.0:
            normalized[3:] = raw[3:] / translation_norm
            twist[3:] = normalized[3:] * translation_radius_m
        normalized_rows.append(normalized)
        twist_direction_rows.append(twist)
    directions = np.stack(normalized_rows)
    twist_directions = np.stack(twist_direction_rows)
    if _quadratic_design_rank(directions) != 21:
        raise AssertionError("controlled 6DoF directions do not span Sym(6)")

    twists = [np.zeros(6, dtype=np.float64)]
    candidate_direction_ids = [-1]
    candidate_signs = [0]
    candidate_radius_fractions = [0.0]
    paths: list[tuple[int, int, int]] = []
    for direction_id, direction in enumerate(twist_directions):
        for sign in (-1, 1):
            indices = [0]
            for fraction in radius_fractions:
                indices.append(len(twists))
                twists.append(float(sign) * float(fraction) * direction)
                candidate_direction_ids.append(direction_id)
                candidate_signs.append(sign)
                candidate_radius_fractions.append(float(fraction))
            paths.append(tuple(indices))

    twist_array = np.stack(twists).astype(np.float64)
    direction_ids = np.asarray(candidate_direction_ids, dtype=np.int16)
    signs = np.asarray(candidate_signs, dtype=np.int8)
    fractions = np.asarray(candidate_radius_fractions, dtype=np.float32)
    radial_paths = np.asarray(paths, dtype=np.int16)
    pair_array = np.asarray(axis_pairs, dtype=np.int8)
    semantics = QUADRATIC_COMPLETE_6DOF_STENCIL_SEMANTICS
    content = _content_sha256(
        twists=twist_array,
        directions=directions,
        candidate_direction_ids=direction_ids,
        candidate_signs=signs,
        candidate_radius_fractions=fractions,
        radial_paths=radial_paths,
        direction_axis_pairs=pair_array,
        translation_radius_m=translation_radius_m,
        rotation_radius_deg=rotation_radius_deg,
        semantics=semantics,
    )
    return ControlledPoseStencil(
        twists_left_camera=twist_array,
        normalized_directions=directions.astype(np.float64),
        candidate_direction_ids=direction_ids,
        candidate_signs=signs,
        candidate_radius_fractions=fractions,
        radial_paths=radial_paths,
        direction_axis_pairs=pair_array,
        translation_radius_m=translation_radius_m,
        rotation_radius_deg=rotation_radius_deg,
        semantics=semantics,
        content_sha256=content,
    )


def stencil_candidate_poses(
    target_pose_w2c: np.ndarray, stencil: ControlledPoseStencil,
) -> np.ndarray:
    """Apply every frozen left-camera twist to one target pose."""

    target = np.asarray(target_pose_w2c, dtype=np.float64)
    if target.shape != (4, 4) or np.any(~np.isfinite(target)):
        raise ValueError("target pose must be one finite 4x4 matrix")
    if not np.allclose(target[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError("target pose has an invalid homogeneous row")
    return np.stack([
        se3_exp(delta) @ target
        for delta in np.asarray(stencil.twists_left_camera, dtype=np.float64)
    ])


def controlled_pose_stencil_audit(
    stencil: ControlledPoseStencil,
) -> dict[str, object]:
    """Return JSON-safe structural evidence for an artifact sidecar."""

    directions = np.asarray(stencil.normalized_directions, dtype=np.float64)
    paths = np.asarray(stencil.radial_paths, dtype=np.int64)
    axis_pairs = np.asarray(stencil.direction_axis_pairs, dtype=np.int64)
    quadratic_rank = _quadratic_design_rank(directions)
    return {
        "semantics": str(stencil.semantics),
        "content_sha256": str(stencil.content_sha256),
        "twist_order": list(TWIST_ORDER),
        "left_multiplicative_camera_frame": True,
        "stage": "medium",
        "translation_radius_m": float(stencil.translation_radius_m),
        "rotation_radius_deg": float(stencil.rotation_radius_deg),
        "radial_fractions": [0.5, 1.0],
        "candidate_count": int(stencil.candidate_count),
        "direction_count": int(stencil.direction_count),
        "coordinate_axis_direction_count": int(np.sum(axis_pairs[:, 1] < 0)),
        "pair_coupling_direction_count": int(np.sum(axis_pairs[:, 1] >= 0)),
        "signed_radial_path_count": int(stencil.radial_path_count),
        "adjacent_monotonic_pair_count": int(
            stencil.radial_path_count * (paths.shape[1] - 1)
        ),
        "covers_all_six_tangent_axes": bool(
            set(axis_pairs[axis_pairs[:, 1] < 0, 0].tolist()) == set(range(6))
        ),
        "covers_all_fifteen_axis_pairs": bool(
            set(map(tuple, axis_pairs[axis_pairs[:, 1] >= 0].tolist()))
            == {(first, second) for first in range(6) for second in range(first + 1, 6)}
        ),
        "symmetric_quadratic_parameter_count": 21,
        "symmetric_quadratic_design_rank": int(quadratic_rank),
        "local_quadratic_6dof_identifiable": bool(quadratic_rank == 21),
        "gt_relative_oracle_diagnostic": True,
        "basin_capture_claim_eligible": False,
        "end_to_end_localization_claim_eligible": False,
    }
