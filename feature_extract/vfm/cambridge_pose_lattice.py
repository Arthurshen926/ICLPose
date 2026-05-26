"""Build controlled rendered-pose lattices from Cambridge pose files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import re

import numpy as np

from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind


@dataclass(frozen=True)
class CambridgePoseRecord:
    image_id: str
    camera_center: np.ndarray
    rotation_w2c: np.ndarray
    pose_w2c: np.ndarray


def safe_image_id_key(image_id: str) -> str:
    """Return a stable filesystem/path-safe image key without extension."""

    path = Path(str(image_id))
    without_suffix = str(path.with_suffix(""))
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", without_suffix).strip("_")


def quaternion_wxyz_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Return the COLMAP-style world-to-camera rotation matrix for q=[w,x,y,z]."""

    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    qw, qx, qy, qz = (q / norm).tolist()
    return np.asarray(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qz * qw,
                2.0 * qz * qx + 2.0 * qy * qw,
            ],
            [
                2.0 * qx * qy + 2.0 * qz * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qx * qw,
            ],
            [
                2.0 * qz * qx - 2.0 * qy * qw,
                2.0 * qy * qz + 2.0 * qx * qw,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def pose_w2c_from_center_rotation(camera_center: np.ndarray, rotation_w2c: np.ndarray) -> np.ndarray:
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_w2c, dtype=np.float64).reshape(3, 3)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ center
    return pose


def camera_center_from_pose_w2c(pose_w2c: np.ndarray) -> np.ndarray:
    """Return world-space camera center from a world-to-camera pose."""

    pose = np.asarray(pose_w2c, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("pose_w2c must have shape (4, 4)")
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return (-rotation.T @ translation).astype(np.float64, copy=False)


def parse_cambridge_pose_file(path: Path) -> list[CambridgePoseRecord]:
    records: list[CambridgePoseRecord] = []
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("Visual Landmark") or text.startswith("ImageFile"):
            continue
        parts = text.split()
        if len(parts) != 8:
            raise ValueError(f"invalid Cambridge pose line: {line}")
        image_id = parts[0]
        center = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        rotation = quaternion_wxyz_to_rotation_matrix(
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
            float(parts[7]),
        )
        records.append(
            CambridgePoseRecord(
                image_id=image_id,
                camera_center=center,
                rotation_w2c=rotation,
                pose_w2c=pose_w2c_from_center_rotation(center, rotation),
            )
        )
    if not records:
        raise ValueError(f"no Cambridge poses found in {path}")
    return records


def parse_world_offsets(text: str) -> tuple[tuple[float, float, float], ...]:
    offsets = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [float(value) for value in item.split(",")]
        if len(parts) != 3:
            raise ValueError("each offset must be formatted as dx,dy,dz")
        offsets.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not offsets:
        raise ValueError("at least one offset is required")
    return tuple(offsets)


def q_level_world_offsets(q_level: str) -> tuple[tuple[float, float, float], ...]:
    """Return a compact axis-aligned translation lattice for a q-level radius."""

    radii = {
        "q10": 0.10,
        "q25": 0.25,
        "q50": 0.50,
    }
    if q_level not in radii:
        raise ValueError("q_level must be one of: q10, q25, q50")
    radius = float(radii[q_level])
    half = 0.5 * radius
    offsets: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for scale in (half, radius):
        offsets.extend(
            [
                (scale, 0.0, 0.0),
                (-scale, 0.0, 0.0),
                (0.0, scale, 0.0),
                (0.0, -scale, 0.0),
                (0.0, 0.0, scale),
                (0.0, 0.0, -scale),
            ]
        )
    return tuple(offsets)


def build_cambridge_pose_lattice_bank(
    pose_file: Path,
    protocol_name: str,
    offsets: Sequence[Sequence[float]],
) -> CandidateHypothesisBank:
    candidates: list[CandidateHypothesis] = []
    parsed_offsets = tuple(tuple(float(value) for value in offset) for offset in offsets)
    for record in parse_cambridge_pose_file(Path(pose_file)):
        for idx, offset in enumerate(parsed_offsets):
            offset_vector = np.asarray(offset, dtype=np.float64).reshape(3)
            center = record.camera_center + offset_vector
            pose = pose_w2c_from_center_rotation(center, record.rotation_w2c)
            translation_m = float(np.linalg.norm(offset_vector))
            candidates.append(
                CandidateHypothesis(
                    query_id=record.image_id,
                    candidate_id=f"{safe_image_id_key(record.image_id)}:rendered_lattice:{idx:03d}",
                    candidate_type="rendered_pose_lattice",
                    pose=pose.tolist(),
                    pose_error=PoseCost(translation_m=translation_m, rotation_deg=0.0),
                    metadata={
                        "candidate_generator": "gt_centered_world_translation_lattice",
                        "candidate_uses_gt": True,
                        "offset_world_m": [float(value) for value in offset],
                    },
                )
            )
    return CandidateHypothesisBank.from_candidates(
        protocol_name=protocol_name,
        protocol_kind=ProtocolKind.CONTROLLED_LATTICE,
        candidates=candidates,
    )


def _gt_pose_index(gt_pose_file: Path) -> dict[str, CambridgePoseRecord]:
    return {record.image_id: record for record in parse_cambridge_pose_file(Path(gt_pose_file))}


def rotation_angle_deg(rotation_a_w2c: np.ndarray, rotation_b_w2c: np.ndarray) -> float:
    """Return angular distance between two world-to-camera rotations."""

    relative = np.asarray(rotation_a_w2c, dtype=np.float64) @ np.asarray(rotation_b_w2c, dtype=np.float64).T
    cos_angle = float((np.trace(relative) - 1.0) * 0.5)
    cos_angle = min(1.0, max(-1.0, cos_angle))
    return float(np.degrees(np.arccos(cos_angle)))


def _source_rank(candidate: CandidateHypothesis, fallback: int) -> tuple[float, int, str]:
    rank = candidate.metadata.get("retrieval_rank", candidate.metadata.get("reference_rank"))
    if rank is not None:
        try:
            return (float(rank), fallback, candidate.candidate_id)
        except (TypeError, ValueError):
            pass
    if candidate.prior_score is not None:
        return (-float(candidate.prior_score), fallback, candidate.candidate_id)
    return (float(fallback), fallback, candidate.candidate_id)


def build_init_pose_lattice_bank(
    init_bank: CandidateHypothesisBank,
    gt_pose_file: Path,
    protocol_name: str,
    offsets: Sequence[Sequence[float]],
    max_inits_per_query: int = 1,
) -> CandidateHypothesisBank:
    """Build rendered-pose proposals around non-oracle initialization poses.

    The initialization pose defines the candidate distribution. Cambridge GT
    poses are used only to label candidate pose error for evaluation.
    """

    if max_inits_per_query <= 0:
        raise ValueError("max_inits_per_query must be positive")
    parsed_offsets = tuple(tuple(float(value) for value in offset) for offset in offsets)
    if not parsed_offsets:
        raise ValueError("at least one offset is required")
    gt_by_query = _gt_pose_index(Path(gt_pose_file))
    grouped: dict[str, list[tuple[int, CandidateHypothesis]]] = {}
    for idx, candidate in enumerate(init_bank.candidates):
        if candidate.query_id is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing query_id")
        if candidate.pose is None:
            raise ValueError(f"candidate {candidate.candidate_id} is missing pose")
        if candidate.query_id not in gt_by_query:
            raise ValueError(f"GT pose not found for query {candidate.query_id!r}")
        grouped.setdefault(candidate.query_id, []).append((idx, candidate))

    candidates: list[CandidateHypothesis] = []
    for query_id in sorted(grouped):
        selected_inits = sorted(
            grouped[query_id],
            key=lambda item: _source_rank(item[1], item[0]),
        )[:max_inits_per_query]
        gt = gt_by_query[query_id]
        for init_rank, (_source_index, source) in enumerate(selected_inits):
            init_pose = np.asarray(source.pose, dtype=np.float64)
            if init_pose.shape != (4, 4):
                raise ValueError(f"candidate {source.candidate_id} pose must have shape (4, 4)")
            init_rotation = init_pose[:3, :3]
            init_center = camera_center_from_pose_w2c(init_pose)
            for offset_rank, offset in enumerate(parsed_offsets):
                offset_vector = np.asarray(offset, dtype=np.float64).reshape(3)
                center = init_center + offset_vector
                pose = pose_w2c_from_center_rotation(center, init_rotation)
                translation_m = float(np.linalg.norm(center - gt.camera_center))
                rotation_deg = rotation_angle_deg(init_rotation, gt.rotation_w2c)
                metadata = dict(source.metadata)
                metadata.update(
                    {
                        "candidate_generator": "init_centered_world_translation_lattice",
                        "candidate_uses_gt": False,
                        "gt_used_for_label_only": True,
                        "source_protocol_name": init_bank.protocol_name,
                        "source_protocol_kind": init_bank.protocol_kind.value,
                        "source_candidate_id": source.candidate_id,
                        "source_candidate_type": source.candidate_type,
                        "source_init_rank": init_rank + 1,
                        "offset_rank": offset_rank,
                        "offset_world_m": [float(value) for value in offset],
                    }
                )
                candidates.append(
                    CandidateHypothesis(
                        query_id=query_id,
                        candidate_id=f"{safe_image_id_key(query_id)}:init_lattice:{init_rank:03d}:{offset_rank:03d}",
                        candidate_type="rendered_pose_lattice",
                        pose=pose.tolist(),
                        reference_image=source.reference_image,
                        prior_score=source.prior_score,
                        pose_error=PoseCost(translation_m=translation_m, rotation_deg=rotation_deg),
                        metadata=metadata,
                    )
                )
    return CandidateHypothesisBank.from_candidates(
        protocol_name=protocol_name,
        protocol_kind=ProtocolKind.RENDERED_POSE,
        candidates=candidates,
    )


def build_cambridge_reference_pose_neighbor_bank(
    query_pose_file: Path,
    reference_pose_file: Path,
    protocol_name: str,
    top_k: int = 10,
    exclude_same_image: bool = False,
    rot_cost_weight: float = 0.05,
) -> CandidateHypothesisBank:
    """Build a fixed reference-pose candidate bank from Cambridge pose files.

    This is intended for supervised selector training on train-query images.
    It uses poses to define nearest reference candidates, so artifacts must be
    reported as reference-pose/GT-assisted candidate generation rather than as
    deployable retrieval.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rot_cost_weight < 0.0:
        raise ValueError("rot_cost_weight must be non-negative")
    queries = parse_cambridge_pose_file(Path(query_pose_file))
    references = parse_cambridge_pose_file(Path(reference_pose_file))
    candidates: list[CandidateHypothesis] = []
    for query in queries:
        scored = []
        for reference in references:
            if exclude_same_image and query.image_id == reference.image_id:
                continue
            translation_m = float(np.linalg.norm(query.camera_center - reference.camera_center))
            rotation_deg = rotation_angle_deg(query.rotation_w2c, reference.rotation_w2c)
            pose_cost_m = translation_m + rot_cost_weight * rotation_deg
            scored.append((pose_cost_m, translation_m, rotation_deg, reference))
        if not scored:
            raise ValueError(f"query {query.image_id} has no valid reference candidates")
        scored.sort(key=lambda item: (item[0], item[3].image_id))
        for rank, (pose_cost_m, translation_m, rotation_deg, reference) in enumerate(scored[:top_k], start=1):
            candidates.append(
                CandidateHypothesis(
                    query_id=query.image_id,
                    candidate_id=f"{Path(query.image_id).stem}:reference_pose:{rank - 1:03d}",
                    candidate_type="reference_pose",
                    pose=reference.pose_w2c.tolist(),
                    reference_image=reference.image_id,
                    pose_error=PoseCost(translation_m=translation_m, rotation_deg=rotation_deg),
                    prior_score=-float(pose_cost_m),
                    metadata={
                        "candidate_generator": "pose_nearest_reference",
                        "candidate_uses_gt": True,
                        "reference_rank": rank,
                        "pose_cost_m": float(pose_cost_m),
                        "rot_cost_weight": float(rot_cost_weight),
                    },
                )
            )
    return CandidateHypothesisBank.from_candidates(
        protocol_name=protocol_name,
        protocol_kind=ProtocolKind.REFERENCE_POSE,
        candidates=candidates,
    )
