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


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Return a unit quaternion [w, x, y, z] from a rotation matrix."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diag = np.diag(matrix)
        axis = int(np.argmax(diag))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
    if quat[0] < 0.0:
        quat = -quat
    return tuple(float(value) for value in quat)


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


def write_cambridge_pose_file(records: Sequence[CambridgePoseRecord], path: Path) -> None:
    values = tuple(records)
    if not values:
        raise ValueError("at least one pose record is required")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Visual Landmark Dataset V1",
        "ImageFile, Camera Position [X Y Z W P Q R]",
        "",
    ]
    for record in values:
        qw, qx, qy, qz = rotation_matrix_to_quaternion_wxyz(record.rotation_w2c)
        center = np.asarray(record.camera_center, dtype=np.float64).reshape(3)
        lines.append(
            f"{record.image_id} "
            f"{center[0]:.9f} {center[1]:.9f} {center[2]:.9f} "
            f"{qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f}"
        )
    output.write_text("\n".join(lines) + "\n")


def _world_y_yaw_rotation(degrees: float) -> np.ndarray:
    angle = np.radians(float(degrees))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _world_x_pitch_rotation(degrees: float) -> np.ndarray:
    angle = np.radians(float(degrees))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _world_z_roll_rotation(degrees: float) -> np.ndarray:
    angle = np.radians(float(degrees))
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _world_yaw_pitch_roll_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    return (
        _world_z_roll_rotation(float(roll_deg))
        @ _world_y_yaw_rotation(float(yaw_deg))
        @ _world_x_pitch_rotation(float(pitch_deg))
    )


def _parse_yaw_offsets(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("at least one yaw offset is required")
    return values


def build_virtual_reference_pose_records(
    reference_pose_file: Path,
    offsets: Sequence[Sequence[float]],
    yaw_offsets_deg: Sequence[float] = (0.0,),
    image_prefix: str = "virtual_reference",
) -> list[CambridgePoseRecord]:
    """Build renderable virtual reference poses from real reference poses.

    The generated records use only reference poses and synthetic offsets. Query
    GT is intentionally not an input; query poses should only be used later for
    offline coverage labels or retrieval benchmarks.
    """

    parsed_offsets = tuple(tuple(float(value) for value in offset) for offset in offsets)
    if not parsed_offsets:
        raise ValueError("at least one offset is required")
    parsed_yaw = tuple(float(value) for value in yaw_offsets_deg)
    if not parsed_yaw:
        raise ValueError("at least one yaw offset is required")
    prefix = str(image_prefix).strip().strip("/")
    if not prefix:
        raise ValueError("image_prefix must be non-empty")
    records: list[CambridgePoseRecord] = []
    for reference in parse_cambridge_pose_file(Path(reference_pose_file)):
        ref_key = safe_image_id_key(reference.image_id)
        for offset_rank, offset in enumerate(parsed_offsets):
            offset_vector = np.asarray(offset, dtype=np.float64).reshape(3)
            center = reference.camera_center + offset_vector
            for yaw_rank, yaw_deg in enumerate(parsed_yaw):
                yaw_world = _world_y_yaw_rotation(float(yaw_deg))
                rotation = reference.rotation_w2c @ yaw_world.T
                image_id = f"{prefix}/{ref_key}/t{offset_rank:03d}_y{yaw_rank:03d}.png"
                records.append(
                    CambridgePoseRecord(
                        image_id=image_id,
                        camera_center=center,
                        rotation_w2c=rotation,
                        pose_w2c=pose_w2c_from_center_rotation(center, rotation),
                    )
                )
    return records


def _inclusive_axis_values(min_value: float, max_value: float, step: float) -> tuple[float, ...]:
    if float(step) <= 0.0:
        raise ValueError("grid_step_m must be positive")
    start = float(min_value)
    end = float(max_value)
    if end < start:
        raise ValueError("grid axis max must be greater than or equal to min")
    values: list[float] = []
    current = start
    epsilon = max(abs(float(step)) * 1e-6, 1e-9)
    while current <= end + epsilon:
        values.append(float(current))
        current += float(step)
    if values and values[-1] < end - epsilon:
        values.append(end)
    return tuple(values)


def build_virtual_reference_pose_grid_records(
    reference_pose_file: Path,
    grid_step_m: float,
    yaw_offsets_deg: Sequence[float] = (0.0,),
    image_prefix: str = "virtual_reference_grid",
    margin_m: float = 0.0,
    height_mode: str = "nearest",
    height_knn: int = 4,
    height_offsets_m: Sequence[float] = (0.0,),
    orientation_knn: int = 1,
) -> list[CambridgePoseRecord]:
    """Build a dense renderable virtual pose grid from reference poses only.

    The grid samples world x/z locations inside the reference-pose footprint.
    Each virtual pose inherits height and base orientation from the nearest
    real reference pose, then applies optional world-y yaw offsets. Query poses
    are intentionally not used.
    """

    if float(grid_step_m) <= 0.0:
        raise ValueError("grid_step_m must be positive")
    if float(margin_m) < 0.0:
        raise ValueError("margin_m must be non-negative")
    parsed_height_mode = str(height_mode)
    if parsed_height_mode not in {"nearest", "idw"}:
        raise ValueError("height_mode must be one of: nearest, idw")
    if int(height_knn) <= 0:
        raise ValueError("height_knn must be positive")
    if int(orientation_knn) <= 0:
        raise ValueError("orientation_knn must be positive")
    parsed_yaw = tuple(float(value) for value in yaw_offsets_deg)
    if not parsed_yaw:
        raise ValueError("at least one yaw offset is required")
    parsed_height_offsets = tuple(float(value) for value in height_offsets_m)
    if not parsed_height_offsets:
        raise ValueError("at least one height offset is required")
    prefix = str(image_prefix).strip().strip("/")
    if not prefix:
        raise ValueError("image_prefix must be non-empty")

    references = parse_cambridge_pose_file(Path(reference_pose_file))
    centers = np.stack([record.camera_center for record in references], axis=0).astype(np.float64)
    margin = float(margin_m)
    x_values = _inclusive_axis_values(
        float(np.min(centers[:, 0]) - margin),
        float(np.max(centers[:, 0]) + margin),
        float(grid_step_m),
    )
    z_values = _inclusive_axis_values(
        float(np.min(centers[:, 2]) - margin),
        float(np.max(centers[:, 2]) + margin),
        float(grid_step_m),
    )
    reference_xz = centers[:, [0, 2]]
    records: list[CambridgePoseRecord] = []
    for x_rank, x_value in enumerate(x_values):
        for z_rank, z_value in enumerate(z_values):
            xz = np.asarray([x_value, z_value], dtype=np.float64)
            dists2 = np.sum((reference_xz - xz[None, :]) ** 2, axis=1)
            nearest_idx = int(np.argmin(dists2))
            reference = references[nearest_idx]
            orientation_order = np.argsort(dists2)[: min(int(orientation_knn), len(references))]
            if parsed_height_mode == "idw":
                order = np.argsort(dists2)[: min(int(height_knn), len(references))]
                if float(dists2[order[0]]) <= 1e-12:
                    height = float(centers[order[0], 1])
                else:
                    weights = 1.0 / np.maximum(dists2[order], 1e-12)
                    height = float(np.sum(weights * centers[order, 1]) / np.sum(weights))
            else:
                height = float(reference.camera_center[1])
            include_height_rank = len(parsed_height_offsets) != 1 or abs(parsed_height_offsets[0]) > 1e-12
            include_orientation_rank = len(orientation_order) != 1
            for orientation_rank, orientation_idx in enumerate(orientation_order):
                orientation_reference = references[int(orientation_idx)]
                for height_rank, height_offset in enumerate(parsed_height_offsets):
                    center = np.asarray([x_value, height + float(height_offset), z_value], dtype=np.float64)
                    for yaw_rank, yaw_deg in enumerate(parsed_yaw):
                        yaw_world = _world_y_yaw_rotation(float(yaw_deg))
                        rotation = orientation_reference.rotation_w2c @ yaw_world.T
                        id_parts = [f"x{x_rank:03d}", f"z{z_rank:03d}"]
                        if include_orientation_rank:
                            id_parts.append(f"o{orientation_rank:03d}")
                        if include_height_rank:
                            id_parts.append(f"h{height_rank:03d}")
                        id_parts.append(f"y{yaw_rank:03d}")
                        image_id = f"{prefix}/{'_'.join(id_parts)}.png"
                        records.append(
                            CambridgePoseRecord(
                                image_id=image_id,
                                camera_center=center,
                                rotation_w2c=rotation,
                                pose_w2c=pose_w2c_from_center_rotation(center, rotation),
                            )
                        )
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


def fine_lattice_world_offsets(
    radius_m: float,
    step_m: float,
    height_offsets_m: Sequence[float] = (0.0,),
) -> tuple[tuple[float, float, float], ...]:
    """Return a disk-shaped local x/z lattice with explicit height offsets."""

    if float(radius_m) < 0.0:
        raise ValueError("radius_m must be non-negative")
    if float(step_m) <= 0.0:
        raise ValueError("step_m must be positive")
    height_offsets = tuple(float(value) for value in height_offsets_m)
    if not height_offsets:
        raise ValueError("at least one height offset is required")
    radius = float(radius_m)
    steps = np.arange(-radius, radius + float(step_m) * 0.5, float(step_m), dtype=np.float64)
    offsets: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for dx in steps:
        for dz in steps:
            if float(np.hypot(dx, dz)) > radius + 1e-9:
                continue
            for dy in height_offsets:
                offset = (
                    float(np.round(dx, 9)),
                    float(np.round(dy, 9)),
                    float(np.round(dz, 9)),
                )
                if offset not in seen:
                    offsets.append(offset)
                    seen.add(offset)
    offsets.sort(key=lambda item: (float(np.linalg.norm(item)), item[0], item[1], item[2]))
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
    if abs(1.0 - cos_angle) <= 1e-12:
        return 0.0
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
    yaw_offsets_deg: Sequence[float] = (0.0,),
    pitch_offsets_deg: Sequence[float] = (0.0,),
    roll_offsets_deg: Sequence[float] = (0.0,),
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
    parsed_yaw = tuple(float(value) for value in yaw_offsets_deg)
    if not parsed_yaw:
        raise ValueError("at least one yaw offset is required")
    parsed_pitch = tuple(float(value) for value in pitch_offsets_deg)
    if not parsed_pitch:
        raise ValueError("at least one pitch offset is required")
    parsed_roll = tuple(float(value) for value in roll_offsets_deg)
    if not parsed_roll:
        raise ValueError("at least one roll offset is required")
    orientation_offsets: list[tuple[int, int, int, float, float, float]] = []
    for yaw_rank, yaw_deg in enumerate(parsed_yaw):
        for pitch_rank, pitch_deg in enumerate(parsed_pitch):
            for roll_rank, roll_deg in enumerate(parsed_roll):
                orientation_offsets.append((yaw_rank, pitch_rank, roll_rank, yaw_deg, pitch_deg, roll_deg))
    include_orientation_rank = (
        len(orientation_offsets) != 1
        or abs(orientation_offsets[0][3]) > 1e-12
        or abs(orientation_offsets[0][4]) > 1e-12
        or abs(orientation_offsets[0][5]) > 1e-12
    )
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
                for orientation_rank, (yaw_rank, pitch_rank, roll_rank, yaw_deg, pitch_deg, roll_deg) in enumerate(
                    orientation_offsets
                ):
                    orientation_world = _world_yaw_pitch_roll_rotation(
                        yaw_deg=float(yaw_deg),
                        pitch_deg=float(pitch_deg),
                        roll_deg=float(roll_deg),
                    )
                    candidate_rotation = init_rotation @ orientation_world.T
                    pose = pose_w2c_from_center_rotation(center, candidate_rotation)
                    translation_m = float(np.linalg.norm(center - gt.camera_center))
                    rotation_deg = rotation_angle_deg(candidate_rotation, gt.rotation_w2c)
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
                            "orientation_rank": orientation_rank,
                            "orientation_lattice_kind": "world_yaw_pitch_roll",
                            "yaw_rank": yaw_rank,
                            "yaw_offset_deg": float(yaw_deg),
                            "pitch_rank": pitch_rank,
                            "pitch_offset_deg": float(pitch_deg),
                            "roll_rank": roll_rank,
                            "roll_offset_deg": float(roll_deg),
                        }
                    )
                    if not include_orientation_rank:
                        candidate_id = f"{safe_image_id_key(query_id)}:init_lattice:{init_rank:03d}:{offset_rank:03d}"
                    else:
                        candidate_id = (
                            f"{safe_image_id_key(query_id)}:init_lattice:"
                            f"{init_rank:03d}:{offset_rank:03d}:{orientation_rank:03d}"
                        )
                    candidates.append(
                        CandidateHypothesis(
                            query_id=query_id,
                            candidate_id=candidate_id,
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
    reference_centers = np.stack([reference.camera_center for reference in references], axis=0).astype(np.float64)
    reference_rotations = np.stack([reference.rotation_w2c for reference in references], axis=0).astype(np.float64)
    reference_image_ids = tuple(reference.image_id for reference in references)
    candidates: list[CandidateHypothesis] = []
    for query in queries:
        translations = np.linalg.norm(reference_centers - query.camera_center[None, :], axis=1)
        cos_angles = (np.einsum("ij,nij->n", query.rotation_w2c, reference_rotations) - 1.0) * 0.5
        rotations = np.degrees(np.arccos(np.clip(cos_angles, -1.0, 1.0)))
        pose_costs = translations + float(rot_cost_weight) * rotations
        if exclude_same_image:
            same = np.asarray([query.image_id == reference_id for reference_id in reference_image_ids], dtype=bool)
            pose_costs = pose_costs.copy()
            pose_costs[same] = np.inf
        valid_count = int(np.count_nonzero(np.isfinite(pose_costs)))
        if valid_count <= 0:
            raise ValueError(f"query {query.image_id} has no valid reference candidates")
        keep = min(int(top_k), valid_count)
        if keep < len(references):
            top_indices = np.argpartition(pose_costs, kth=keep - 1)[:keep]
        else:
            top_indices = np.arange(len(references), dtype=np.int64)
        sorted_indices = sorted(
            (int(idx) for idx in top_indices if np.isfinite(pose_costs[int(idx)])),
            key=lambda idx: (float(pose_costs[idx]), reference_image_ids[idx]),
        )[:keep]
        for rank, ref_idx in enumerate(sorted_indices, start=1):
            reference = references[ref_idx]
            translation_m = float(translations[ref_idx])
            rotation_deg = float(rotations[ref_idx])
            pose_cost_m = float(pose_costs[ref_idx])
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
