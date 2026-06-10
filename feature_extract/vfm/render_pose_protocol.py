"""Render-pose protocol helpers for query-to-render localization experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    camera_center_from_pose_w2c,
    pose_w2c_from_center_rotation,
    rotation_angle_deg,
)
from feature_extract.vfm.hypotheses import CandidateHypothesis


@dataclass(frozen=True)
class RenderPoseSelection:
    pose_w2c: np.ndarray
    label: str
    candidate_id: str | None = None
    reference_image: str | None = None
    render_translation_error_m: float | None = None
    render_rotation_error_deg: float | None = None

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose_w2c, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError("pose_w2c must have shape (4, 4)")
        object.__setattr__(self, "pose_w2c", pose)


@dataclass(frozen=True)
class SE3Perturbation:
    pose_w2c: np.ndarray
    translation_world: np.ndarray
    rotation_deg_xyz: np.ndarray
    translation_error_m: float
    rotation_error_deg: float

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose_w2c, dtype=np.float64).reshape(4, 4)
        translation = np.asarray(self.translation_world, dtype=np.float64).reshape(3)
        rotation = np.asarray(self.rotation_deg_xyz, dtype=np.float64).reshape(3)
        object.__setattr__(self, "pose_w2c", pose)
        object.__setattr__(self, "translation_world", translation)
        object.__setattr__(self, "rotation_deg_xyz", rotation)


def parse_world_offset(text: str) -> np.ndarray:
    parts = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("world offset must be formatted as dx,dy,dz")
    return np.asarray(parts, dtype=np.float64)


def translate_pose_world(pose_w2c: np.ndarray, offset_world: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    offset = np.asarray(offset_world, dtype=np.float64).reshape(3)
    center = camera_center_from_pose_w2c(pose) + offset
    return pose_w2c_from_center_rotation(center, pose[:3, :3])


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{str(key)}".encode("utf8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) & 0x7FFFFFFF


def _euler_xyz_to_rotation(rotation_deg_xyz: Sequence[float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(rotation_deg_xyz, dtype=np.float64).reshape(3))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def sample_se3_perturbation(
    pose_w2c: np.ndarray,
    *,
    max_translation_m: float,
    max_rotation_deg: float = 0.0,
    seed: int = 0,
    key: str = "",
) -> SE3Perturbation:
    """Deterministically perturb a pose in camera-center/world coordinates."""

    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rng = np.random.default_rng(_stable_seed(int(seed), str(key)))
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    radius = float(rng.uniform(0.0, max(float(max_translation_m), 0.0)))
    translation = direction * radius
    rotation_deg_xyz = rng.uniform(
        -max(float(max_rotation_deg), 0.0),
        max(float(max_rotation_deg), 0.0),
        size=3,
    ).astype(np.float64)
    center = camera_center_from_pose_w2c(pose) + translation
    delta_rotation = _euler_xyz_to_rotation(rotation_deg_xyz)
    perturbed_rotation = delta_rotation @ pose[:3, :3]
    perturbed = pose_w2c_from_center_rotation(center, perturbed_rotation)
    translation_error, rotation_error = render_pose_error_fields(perturbed, pose)
    return SE3Perturbation(
        pose_w2c=perturbed,
        translation_world=translation,
        rotation_deg_xyz=rotation_deg_xyz,
        translation_error_m=translation_error,
        rotation_error_deg=rotation_error,
    )


def render_pose_error_fields(render_pose_w2c: np.ndarray, gt_pose_w2c: np.ndarray) -> tuple[float, float]:
    render = np.asarray(render_pose_w2c, dtype=np.float64).reshape(4, 4)
    gt = np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4)
    t_error = float(np.linalg.norm(camera_center_from_pose_w2c(render) - camera_center_from_pose_w2c(gt)))
    r_error = rotation_angle_deg(render[:3, :3], gt[:3, :3])
    return t_error, r_error


def _candidate_rank(candidate: CandidateHypothesis, fallback: int) -> tuple[float, int, str]:
    metadata = dict(candidate.metadata)
    for key in ("retrieval_rank", "reference_rank", "score_rank"):
        value = metadata.get(key)
        if value is not None:
            try:
                return (float(value), fallback, str(candidate.candidate_id))
            except (TypeError, ValueError):
                pass
    if candidate.prior_score is not None:
        return (-float(candidate.prior_score), fallback, str(candidate.candidate_id))
    return (float(fallback), fallback, str(candidate.candidate_id))


def group_top_reference_poses(candidates: Sequence[CandidateHypothesis]) -> dict[str, CandidateHypothesis]:
    grouped: dict[str, tuple[int, CandidateHypothesis]] = {}
    for idx, candidate in enumerate(candidates):
        if candidate.query_id is None or candidate.pose is None:
            continue
        query_id = str(candidate.query_id)
        current = grouped.get(query_id)
        if current is None or _candidate_rank(candidate, idx) < _candidate_rank(current[1], current[0]):
            grouped[query_id] = (idx, candidate)
    return {query_id: item[1] for query_id, item in grouped.items()}


def group_topk_reference_poses(candidates: Sequence[CandidateHypothesis], *, top_k: int) -> dict[str, list[CandidateHypothesis]]:
    grouped: dict[str, list[tuple[tuple[float, int, str], CandidateHypothesis]]] = {}
    for idx, candidate in enumerate(candidates):
        if candidate.query_id is None or candidate.pose is None:
            continue
        query_id = str(candidate.query_id)
        grouped.setdefault(query_id, []).append((_candidate_rank(candidate, idx), candidate))
    output: dict[str, list[CandidateHypothesis]] = {}
    limit = max(1, int(top_k))
    for query_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        output[query_id] = [candidate for _rank, candidate in items[:limit]]
    return output


def select_render_pose(
    query_id: str,
    gt_pose_w2c: np.ndarray,
    *,
    mode: str,
    world_offset: np.ndarray | None = None,
    reference_top1: Mapping[str, CandidateHypothesis] | None = None,
) -> RenderPoseSelection:
    gt = np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4)
    mode = str(mode)
    if mode == "gt":
        return RenderPoseSelection(
            pose_w2c=gt,
            label="gt",
            render_translation_error_m=0.0,
            render_rotation_error_deg=0.0,
        )
    if mode == "gt_offset":
        if world_offset is None:
            raise ValueError("mode=gt_offset requires world_offset")
        offset = np.asarray(world_offset, dtype=np.float64).reshape(3)
        pose = translate_pose_world(gt, offset)
        t_error, r_error = render_pose_error_fields(pose, gt)
        label = f"gt_offset:{offset[0]:.3f},{offset[1]:.3f},{offset[2]:.3f}"
        return RenderPoseSelection(
            pose_w2c=pose,
            label=label,
            render_translation_error_m=t_error,
            render_rotation_error_deg=r_error,
        )
    if mode == "reference_top1":
        reference_top1 = reference_top1 or {}
        candidate = reference_top1.get(str(query_id))
        if candidate is None or candidate.pose is None:
            raise KeyError(f"reference top1 pose not found for query {query_id!r}")
        pose = np.asarray(candidate.pose, dtype=np.float64).reshape(4, 4)
        t_error, r_error = render_pose_error_fields(pose, gt)
        return RenderPoseSelection(
            pose_w2c=pose,
            label="reference_top1",
            candidate_id=str(candidate.candidate_id),
            reference_image=candidate.reference_image,
            render_translation_error_m=t_error,
            render_rotation_error_deg=r_error,
        )
    raise ValueError("render pose mode must be one of: gt, gt_offset, reference_top1")
