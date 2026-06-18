"""Streaming VPR top-k utilities for large virtual pose databases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

from feature_extract.vfm.cambridge_pose_lattice import (
    CambridgePoseRecord,
    _inclusive_axis_values,
    _world_y_yaw_rotation,
    camera_center_from_pose_w2c,
    parse_cambridge_pose_file,
    pose_w2c_from_center_rotation,
    rotation_angle_deg,
    safe_image_id_key,
)
from feature_extract.vfm.hypotheses import CandidateHypothesis, PoseCost
from feature_extract.vfm.hypothesis_io import CandidateHypothesisBank
from feature_extract.vfm.protocols import ProtocolKind


@dataclass(frozen=True)
class StreamedGridRecord:
    ordinal: int
    record: CambridgePoseRecord


@dataclass(frozen=True)
class StreamingVPRMatch:
    score: float
    ordinal: int
    record: CambridgePoseRecord


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError("descriptors must have shape (N, D)")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-6)


class StreamingVPRTopK:
    """Maintain per-query top-k descriptor matches over streamed candidates."""

    def __init__(self, query_ids: Sequence[str], query_descriptors: np.ndarray, top_k: int) -> None:
        if int(top_k) <= 0:
            raise ValueError("top_k must be positive")
        self.query_ids: Tuple[str, ...] = tuple(str(query_id) for query_id in query_ids)
        if len(self.query_ids) == 0:
            raise ValueError("at least one query descriptor is required")
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("duplicate query id")
        self.query_descriptors = _normalize_rows(query_descriptors)
        if self.query_descriptors.shape[0] != len(self.query_ids):
            raise ValueError("query_ids length must match query descriptor rows")
        self.top_k = int(top_k)
        self._matches: dict[str, list[StreamingVPRMatch]] = {query_id: [] for query_id in self.query_ids}
        self.processed_candidate_count = 0

    def update(
        self,
        *,
        records: Sequence[CambridgePoseRecord],
        descriptors: np.ndarray,
        start_ordinal: int,
        ordinals: Sequence[int] | None = None,
    ) -> None:
        record_values = tuple(records)
        candidate_descriptors = _normalize_rows(descriptors)
        if candidate_descriptors.shape[0] != len(record_values):
            raise ValueError("record count must match candidate descriptor rows")
        if candidate_descriptors.shape[1] != self.query_descriptors.shape[1]:
            raise ValueError("candidate descriptor dimension must match query descriptors")
        if len(record_values) == 0:
            return
        if ordinals is None:
            ordinal_values = tuple(int(start_ordinal) + int(local_idx) for local_idx in range(len(record_values)))
        else:
            ordinal_values = tuple(int(ordinal) for ordinal in ordinals)
            if len(ordinal_values) != len(record_values):
                raise ValueError("ordinals length must match record count")
        scores = np.asarray(self.query_descriptors @ candidate_descriptors.T, dtype=np.float32)
        for query_idx, query_id in enumerate(self.query_ids):
            merged = list(self._matches[query_id])
            for local_idx, record in enumerate(record_values):
                merged.append(
                    StreamingVPRMatch(
                        score=float(scores[query_idx, local_idx]),
                        ordinal=ordinal_values[local_idx],
                        record=record,
                    )
                )
            merged.sort(key=lambda item: (-item.score, item.ordinal, item.record.image_id))
            self._matches[query_id] = merged[: self.top_k]
        self.processed_candidate_count += len(record_values)

    def results_for_query(self, query_id: str) -> Tuple[StreamingVPRMatch, ...]:
        if query_id not in self._matches:
            raise ValueError(f"query id not found: {query_id}")
        return tuple(self._matches[query_id])

    def all_results(self) -> Mapping[str, Tuple[StreamingVPRMatch, ...]]:
        return {query_id: tuple(matches) for query_id, matches in self._matches.items()}


def _height_and_order(
    *,
    reference_centers: np.ndarray,
    reference_xz: np.ndarray,
    xz: np.ndarray,
    height_mode: str,
    height_knn: int,
) -> tuple[float, np.ndarray]:
    dists2 = np.sum((reference_xz - xz[None, :]) ** 2, axis=1)
    order = np.argsort(dists2)
    if str(height_mode) == "nearest":
        return float(reference_centers[int(order[0]), 1]), order
    if str(height_mode) != "idw":
        raise ValueError("height_mode must be one of: nearest, idw")
    keep = order[: min(int(height_knn), int(order.shape[0]))]
    if float(dists2[int(keep[0])]) <= 1e-12:
        return float(reference_centers[int(keep[0]), 1]), order
    weights = 1.0 / np.maximum(dists2[keep], 1e-12)
    return float(np.sum(weights * reference_centers[keep, 1]) / np.sum(weights)), order


def iter_virtual_reference_grid_records(
    *,
    reference_pose_file: Path,
    grid_step_m: float,
    grid_margin_m: float,
    height_mode: str,
    height_knn: int,
    height_offsets_m: Sequence[float],
    orientation_knn: int,
    yaw_offsets_deg: Sequence[float],
    image_prefix: str,
    start_ordinal: int = 0,
    max_records: int = 0,
) -> Iterable[StreamedGridRecord]:
    """Yield virtual reference grid records without materializing the full DB."""

    if float(grid_step_m) <= 0.0:
        raise ValueError("grid_step_m must be positive")
    if float(grid_margin_m) < 0.0:
        raise ValueError("grid_margin_m must be non-negative")
    if int(height_knn) <= 0:
        raise ValueError("height_knn must be positive")
    if int(orientation_knn) <= 0:
        raise ValueError("orientation_knn must be positive")
    parsed_height_offsets = tuple(float(value) for value in height_offsets_m)
    parsed_yaw_offsets = tuple(float(value) for value in yaw_offsets_deg)
    if not parsed_height_offsets:
        raise ValueError("at least one height offset is required")
    if not parsed_yaw_offsets:
        raise ValueError("at least one yaw offset is required")
    prefix = str(image_prefix).strip().strip("/")
    if not prefix:
        raise ValueError("image_prefix must be non-empty")

    references = parse_cambridge_pose_file(Path(reference_pose_file))
    centers = np.stack([record.camera_center for record in references], axis=0).astype(np.float64)
    reference_xz = centers[:, [0, 2]]
    margin = float(grid_margin_m)
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
    yaw_rotations = tuple(_world_y_yaw_rotation(yaw_deg) for yaw_deg in parsed_yaw_offsets)
    include_height_rank = len(parsed_height_offsets) != 1 or abs(parsed_height_offsets[0]) > 1e-12
    yielded = 0
    ordinal = 0
    start = max(0, int(start_ordinal))
    limit = int(max_records)
    for x_rank, x_value in enumerate(x_values):
        for z_rank, z_value in enumerate(z_values):
            xz = np.asarray([float(x_value), float(z_value)], dtype=np.float64)
            base_height, order = _height_and_order(
                reference_centers=centers,
                reference_xz=reference_xz,
                xz=xz,
                height_mode=str(height_mode),
                height_knn=int(height_knn),
            )
            orientation_order = order[: min(int(orientation_knn), len(references))]
            include_orientation_rank = len(orientation_order) != 1
            for orientation_rank, orientation_idx in enumerate(orientation_order):
                orientation_reference = references[int(orientation_idx)]
                for height_rank, height_offset in enumerate(parsed_height_offsets):
                    center = np.asarray(
                        [float(x_value), base_height + float(height_offset), float(z_value)],
                        dtype=np.float64,
                    )
                    for yaw_rank, yaw_rotation in enumerate(yaw_rotations):
                        if ordinal >= start:
                            id_parts = [f"x{x_rank:03d}", f"z{z_rank:03d}"]
                            if include_orientation_rank:
                                id_parts.append(f"o{orientation_rank:03d}")
                            if include_height_rank:
                                id_parts.append(f"h{height_rank:03d}")
                            id_parts.append(f"y{yaw_rank:03d}")
                            rotation = orientation_reference.rotation_w2c @ yaw_rotation.T
                            record = CambridgePoseRecord(
                                image_id=f"{prefix}/{'_'.join(id_parts)}.png",
                                camera_center=center,
                                rotation_w2c=rotation,
                                pose_w2c=pose_w2c_from_center_rotation(center, rotation),
                            )
                            yield StreamedGridRecord(ordinal=ordinal, record=record)
                            yielded += 1
                            if limit > 0 and yielded >= limit:
                                return
                        ordinal += 1


def estimate_virtual_reference_grid_record_count(
    *,
    reference_pose_file: Path,
    grid_step_m: float,
    grid_margin_m: float,
    height_offsets_m: Sequence[float],
    orientation_knn: int,
    yaw_offsets_deg: Sequence[float],
) -> int:
    references = parse_cambridge_pose_file(Path(reference_pose_file))
    centers = np.stack([record.camera_center for record in references], axis=0).astype(np.float64)
    margin = float(grid_margin_m)
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
    return int(len(x_values) * len(z_values) * len(tuple(height_offsets_m)) * int(orientation_knn) * len(tuple(yaw_offsets_deg)))


def build_streaming_vpr_candidate_bank(
    topk: StreamingVPRTopK,
    *,
    query_pose_file: Path,
    protocol_name: str,
    descriptor_pooling: str,
) -> CandidateHypothesisBank:
    query_pose_by_id = {record.image_id: record for record in parse_cambridge_pose_file(Path(query_pose_file))}
    candidates: list[CandidateHypothesis] = []
    for query_id in topk.query_ids:
        if query_id not in query_pose_by_id:
            raise ValueError(f"query pose not found: {query_id}")
        query_pose = query_pose_by_id[query_id]
        for rank, match in enumerate(topk.results_for_query(query_id), start=1):
            center = camera_center_from_pose_w2c(match.record.pose_w2c)
            translation_m = float(np.linalg.norm(center - query_pose.camera_center))
            rotation_deg = rotation_angle_deg(match.record.rotation_w2c, query_pose.rotation_w2c)
            candidates.append(
                CandidateHypothesis(
                    query_id=query_id,
                    candidate_id=f"{safe_image_id_key(query_id)}:stream_vpr:{rank - 1:03d}",
                    candidate_type="sharded_streaming_virtual_vpr",
                    pose=match.record.pose_w2c.tolist(),
                    reference_image=match.record.image_id,
                    prior_score=float(match.score),
                    pose_error=PoseCost(translation_m=translation_m, rotation_deg=rotation_deg),
                    metadata={
                        "candidate_generator": "sharded_streaming_virtual_vpr",
                        "candidate_uses_gt": False,
                        "pose_label_uses_gt": True,
                        "streaming_vpr_rank": int(rank),
                        "streaming_vpr_score": float(match.score),
                        "streaming_vpr_candidate_ordinal": int(match.ordinal),
                        "descriptor_pooling": str(descriptor_pooling),
                    },
                )
            )
    if not candidates:
        raise ValueError("streaming VPR produced no candidates")
    return CandidateHypothesisBank.from_candidates(
        protocol_name=str(protocol_name),
        protocol_kind=ProtocolKind.RENDERED_POSE,
        candidates=candidates,
    )
