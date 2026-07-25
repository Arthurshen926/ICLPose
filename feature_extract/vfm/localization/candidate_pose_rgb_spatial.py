"""Strict target-free layout for raw-RGB candidate spatial evidence.

The layout is the immutable join between a frozen query point, its fixed
global top-L 3-D landmark candidates, and fixed real-SfM support observations.
It deliberately contains no pose, projection residual, registered identity,
or learned prediction.  Those values may only be joined in train-only or
post-scoring audit artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


CANDIDATE_POSE_RGB_SPATIAL_LAYOUT_FORMAT = "candidate_pose_rgb_spatial_layout_v1"

_REQUIRED_METADATA_FIELDS = (
    "verification_points_sha256",
    "maplet_support_index_sha256",
    "support_geometry_index_sha256",
    "projection_space_id",
    "descriptor_space_id",
)
_REQUIRED_ARRAY_FIELDS = (
    "source_point_ids",
    "query_ids",
    "split_names",
    "xy",
    "point_sources",
    "candidate_track_ids",
    "candidate_bank_rows",
    "candidate_coarse_similarities",
    "candidate_prior_probabilities",
    "null_probabilities",
    "support_image_ids",
    "support_xy",
    "support_view_valid",
    "support_view_weights",
    "support_coverage_counts",
    "metadata_json",
)
_FORBIDDEN_ARRAY_TOKENS = ("target", "residual", "pose", "label", "ground_truth")


def _metadata_is_target_free(metadata: Mapping[str, object]) -> bool:
    return (
        metadata.get("format") == CANDIDATE_POSE_RGB_SPATIAL_LAYOUT_FORMAT
        and metadata.get("contains_ground_truth") is False
        and metadata.get("contains_target_errors") is False
        and metadata.get("pose_or_ground_truth_used") is False
        and metadata.get("image_retrieval_or_submap_used") is False
        and metadata.get("render") is False
        and all(str(metadata.get(field, "")) for field in _REQUIRED_METADATA_FIELDS)
    )


@dataclass(frozen=True)
class CandidatePoseRGBSpatialLayout:
    """Target-free P1 point/candidate/support-view layout for RGB evidence."""

    source_point_ids: np.ndarray
    query_ids: np.ndarray
    split_names: np.ndarray
    xy: np.ndarray
    point_sources: np.ndarray
    candidate_track_ids: np.ndarray
    candidate_bank_rows: np.ndarray
    candidate_coarse_similarities: np.ndarray
    candidate_prior_probabilities: np.ndarray
    null_probabilities: np.ndarray
    support_image_ids: np.ndarray
    support_xy: np.ndarray
    support_view_valid: np.ndarray
    support_view_weights: np.ndarray
    support_coverage_counts: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        split_names = np.asarray(self.split_names).astype(str).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float32)
        point_sources = np.asarray(self.point_sources).astype(str).reshape(-1)
        tracks = np.asarray(self.candidate_track_ids, dtype=np.int64)
        bank_rows = np.asarray(self.candidate_bank_rows, dtype=np.int64)
        coarse = np.asarray(self.candidate_coarse_similarities, dtype=np.float32)
        candidate = np.asarray(self.candidate_prior_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        support_ids = np.asarray(self.support_image_ids).astype(str)
        support_xy = np.asarray(self.support_xy, dtype=np.float32)
        support_valid = np.asarray(self.support_view_valid, dtype=bool)
        support_weights = np.asarray(self.support_view_weights, dtype=np.float32)
        coverage = np.asarray(self.support_coverage_counts, dtype=np.int32)
        metadata = dict(self.metadata)

        count = len(source_ids)
        if (
            count == 0
            or len(np.unique(source_ids)) != count
            or np.any(query_ids == "")
            or np.any(point_sources == "")
            or query_ids.shape != split_names.shape != point_sources.shape != (count,)
            or set(split_names.tolist()) - {"train", "validation", "test"}
            or xy.shape != (count, 2)
            or tracks.ndim != 2
            or tracks.shape[0] != count
            or tracks.shape[1] == 0
            or bank_rows.shape != tracks.shape
            or coarse.shape != tracks.shape
            or candidate.shape != tracks.shape
            or null.shape != (count,)
            or support_ids.ndim != 3
            or support_ids.shape[:2] != tracks.shape
            or support_ids.shape[2] == 0
            or support_xy.shape != (*support_ids.shape, 2)
            or support_valid.shape != support_ids.shape
            or support_weights.shape != support_ids.shape
            or coverage.shape != support_ids.shape
            or not np.isfinite(xy).all()
            or not np.isfinite(coarse).all()
            or not np.isfinite(candidate).all()
            or not np.isfinite(null).all()
            or not np.isfinite(support_xy).all()
            or np.any(candidate < 0.0)
            or np.any(null < 0.0)
            or np.any(support_weights < 0.0)
            or np.any(coverage < 0)
        ):
            raise ValueError("candidate RGB spatial layout arrays are invalid")
        if not _metadata_is_target_free(metadata):
            raise ValueError("candidate RGB spatial layout metadata is not target-free")

        candidate_valid = tracks >= 0
        invalid_candidate_views = ~candidate_valid[:, :, None]
        if (
            np.any(candidate_valid & (bank_rows < 0))
            or np.any(~candidate_valid & (bank_rows != -1))
            or np.any(~candidate_valid & (candidate > 1e-7))
            or np.any(~candidate_valid & np.any(support_valid, axis=2))
            or np.any(invalid_candidate_views & (support_ids != ""))
            or np.any(invalid_candidate_views & (coverage != 0))
            or np.any(invalid_candidate_views & (support_weights > 1e-7))
        ):
            raise ValueError("candidate RGB spatial layout candidate slots are invalid")
        if np.any(np.abs(candidate.sum(axis=1) + null - 1.0) > 1e-4):
            raise ValueError("candidate RGB spatial layout probability mass is invalid")

        valid_ids = support_ids[support_valid]
        if (
            np.any(valid_ids == "")
            or np.any(support_ids[~support_valid] != "")
            or np.any(coverage[~support_valid] != 0)
            or np.any(support_weights[~support_valid] > 1e-7)
        ):
            raise ValueError("candidate RGB spatial layout support slots are invalid")
        query_by_view = np.broadcast_to(query_ids[:, None, None], support_ids.shape)
        if np.any(support_ids[support_valid] == query_by_view[support_valid]):
            raise ValueError("candidate RGB spatial layout uses a query image as its support")

        view_mass = support_weights.sum(axis=2)
        has_support = np.any(support_valid, axis=2)
        if (
            np.any(np.abs(view_mass[has_support] - 1.0) > 1e-4)
            or np.any(view_mass[~has_support] > 1e-7)
            or np.any(candidate_valid & has_support & (candidate <= 0.0))
        ):
            raise ValueError("candidate RGB spatial layout view weights are invalid")

        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "split_names", split_names)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "point_sources", point_sources)
        object.__setattr__(self, "candidate_track_ids", tracks)
        object.__setattr__(self, "candidate_bank_rows", bank_rows)
        object.__setattr__(self, "candidate_coarse_similarities", coarse)
        object.__setattr__(self, "candidate_prior_probabilities", candidate)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "support_image_ids", support_ids)
        object.__setattr__(self, "support_xy", support_xy)
        object.__setattr__(self, "support_view_valid", support_valid)
        object.__setattr__(self, "support_view_weights", support_weights)
        object.__setattr__(self, "support_coverage_counts", coverage)
        object.__setattr__(self, "metadata", metadata)

    @property
    def row_count(self) -> int:
        return int(len(self.source_point_ids))

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_track_ids.shape[1])

    @property
    def support_view_count(self) -> int:
        return int(self.support_image_ids.shape[2])


def select_fixed_rgb_support_views(
    *,
    query_ids: np.ndarray,
    candidate_track_ids: np.ndarray,
    candidate_bank_rows: np.ndarray,
    maplet_support_image_ids: np.ndarray,
    maplet_support_image_indices: np.ndarray,
    maplet_support_coverage_counts: np.ndarray,
    support_xy_by_image_track: Mapping[tuple[str, int], np.ndarray],
    support_views_per_candidate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Attach fixed, real support observations to frozen candidate tracks.

    ``support_xy_by_image_track`` is built from the SfM support-observation
    index.  An unavailable observation is omitted rather than approximated,
    and each selected view mixture is normalized after query-image exclusion.
    """

    ids = np.asarray(query_ids).astype(str).reshape(-1)
    tracks = np.asarray(candidate_track_ids, dtype=np.int64)
    bank_rows = np.asarray(candidate_bank_rows, dtype=np.int64)
    image_ids = np.asarray(maplet_support_image_ids).astype(str).reshape(-1)
    maplet_indices = np.asarray(maplet_support_image_indices, dtype=np.int64)
    maplet_coverage = np.asarray(maplet_support_coverage_counts, dtype=np.int32)
    view_count = int(support_views_per_candidate)
    if (
        len(ids) == 0
        or np.any(ids == "")
        or tracks.ndim != 2
        or bank_rows.shape != tracks.shape
        or image_ids.size == 0
        or maplet_indices.ndim != 2
        or maplet_coverage.shape != maplet_indices.shape
        or view_count <= 0
        or np.any((maplet_indices < -1) | (maplet_indices >= len(image_ids)))
        or np.any(maplet_coverage < 0)
    ):
        raise ValueError("fixed RGB support-view inputs are invalid")
    valid_candidate = tracks >= 0
    if (
        tracks.shape[0] != len(ids)
        or np.any(valid_candidate & ((bank_rows < 0) | (bank_rows >= len(maplet_indices))))
        or np.any(~valid_candidate & (bank_rows != -1))
    ):
        raise ValueError("fixed RGB support-view candidate rows are invalid")

    all_ids = np.full(
        (*tracks.shape, maplet_indices.shape[1]),
        "",
        dtype=f"<U{max(1, max(len(value) for value in image_ids.tolist()))}",
    )
    all_xy = np.zeros((*all_ids.shape, 2), dtype=np.float32)
    all_valid = np.zeros(all_ids.shape, dtype=bool)
    all_coverage = np.zeros(all_ids.shape, dtype=np.int32)
    for point_index, query_id in enumerate(ids.tolist()):
        for candidate_index, track_id in enumerate(tracks[point_index].tolist()):
            if int(track_id) < 0:
                continue
            canonical_row = int(bank_rows[point_index, candidate_index])
            for source_slot, (support_index, coverage) in enumerate(zip(
                maplet_indices[canonical_row].tolist(),
                maplet_coverage[canonical_row].tolist(),
            )):
                if int(support_index) < 0:
                    continue
                support_id = str(image_ids[int(support_index)])
                if support_id == str(query_id):
                    continue
                coordinate = support_xy_by_image_track.get((support_id, int(track_id)))
                if coordinate is None:
                    continue
                xy = np.asarray(coordinate, dtype=np.float32).reshape(-1)
                if xy.shape != (2,) or not np.isfinite(xy).all():
                    raise ValueError("fixed RGB support-view geometry is invalid")
                all_ids[point_index, candidate_index, source_slot] = support_id
                all_xy[point_index, candidate_index, source_slot] = xy
                all_valid[point_index, candidate_index, source_slot] = True
                all_coverage[point_index, candidate_index, source_slot] = int(coverage)
    return _truncate_resolved_rgb_support_views(
        query_ids=ids,
        support_image_ids=all_ids,
        support_xy=all_xy,
        support_view_valid=all_valid,
        support_coverage_counts=all_coverage,
        support_views_per_candidate=view_count,
    )


def _truncate_resolved_rgb_support_views(
    *,
    query_ids: np.ndarray,
    support_image_ids: np.ndarray,
    support_xy: np.ndarray,
    support_view_valid: np.ndarray,
    support_coverage_counts: np.ndarray,
    support_views_per_candidate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep the first fixed usable maplet views and normalize their weights."""

    query = np.asarray(query_ids).astype(str).reshape(-1)
    source_ids = np.asarray(support_image_ids).astype(str)
    source_xy = np.asarray(support_xy, dtype=np.float32)
    source_valid = np.asarray(support_view_valid, dtype=bool)
    source_coverage = np.asarray(support_coverage_counts, dtype=np.int32)
    count = int(support_views_per_candidate)
    if (
        len(query) == 0
        or source_ids.ndim != 3
        or source_ids.shape[0] != len(query)
        or source_xy.shape != (*source_ids.shape, 2)
        or source_valid.shape != source_ids.shape
        or source_coverage.shape != source_ids.shape
        or count <= 0
        or not np.isfinite(source_xy).all()
        or np.any(source_coverage < 0)
        or np.any(source_ids[source_valid] == "")
    ):
        raise ValueError("resolved RGB support-view inputs are invalid")
    shape = (*source_ids.shape[:2], count)
    selected_ids = np.full(
        shape,
        "",
        dtype=f"<U{max(1, source_ids.dtype.itemsize // np.dtype('<U1').itemsize)}",
    )
    selected_xy = np.zeros((*shape, 2), dtype=np.float32)
    selected_valid = np.zeros(shape, dtype=bool)
    selected_weights = np.zeros(shape, dtype=np.float32)
    selected_coverage = np.zeros(shape, dtype=np.int32)
    for point_index, query_id in enumerate(query.tolist()):
        for candidate_index in range(source_ids.shape[1]):
            valid_slots = np.flatnonzero(
                source_valid[point_index, candidate_index]
                & (source_ids[point_index, candidate_index] != str(query_id))
            )[:count]
            if not len(valid_slots):
                continue
            selected_ids[point_index, candidate_index, : len(valid_slots)] = source_ids[
                point_index, candidate_index, valid_slots
            ]
            selected_xy[point_index, candidate_index, : len(valid_slots)] = source_xy[
                point_index, candidate_index, valid_slots
            ]
            selected_valid[point_index, candidate_index, : len(valid_slots)] = True
            selected_coverage[point_index, candidate_index, : len(valid_slots)] = source_coverage[
                point_index, candidate_index, valid_slots
            ]
            weights = selected_coverage[point_index, candidate_index, : len(valid_slots)].astype(
                np.float32
            )
            if float(weights.sum()) <= 0.0:
                weights.fill(1.0)
            selected_weights[point_index, candidate_index, : len(valid_slots)] = (
                weights / weights.sum()
            )
    return selected_ids, selected_xy, selected_valid, selected_weights, selected_coverage


def save_candidate_pose_rgb_spatial_layout(
    layout: CandidatePoseRGBSpatialLayout, path: Path
) -> None:
    """Serialize a validated target-free RGB spatial layout atomically."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_point_ids=layout.source_point_ids,
            query_ids=layout.query_ids,
            split_names=layout.split_names,
            xy=layout.xy,
            point_sources=layout.point_sources,
            candidate_track_ids=layout.candidate_track_ids,
            candidate_bank_rows=layout.candidate_bank_rows,
            candidate_coarse_similarities=layout.candidate_coarse_similarities,
            candidate_prior_probabilities=layout.candidate_prior_probabilities,
            null_probabilities=layout.null_probabilities,
            support_image_ids=layout.support_image_ids,
            support_xy=layout.support_xy,
            support_view_valid=layout.support_view_valid,
            support_view_weights=layout.support_view_weights,
            support_coverage_counts=layout.support_coverage_counts,
            metadata_json=np.asarray(json.dumps(dict(layout.metadata), sort_keys=True)),
        )
    temporary.replace(output)


def load_candidate_pose_rgb_spatial_layout(path: Path) -> CandidatePoseRGBSpatialLayout:
    """Load a target-free RGB layout and reject accidental target-side fields."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        fields = set(payload.files)
        missing = set(_REQUIRED_ARRAY_FIELDS) - fields
        if missing:
            raise ValueError(f"candidate RGB spatial layout lacks {sorted(missing)}")
        forbidden = sorted(
            field
            for field in fields - {"metadata_json"}
            if any(token in field.lower() for token in _FORBIDDEN_ARRAY_TOKENS)
        )
        if forbidden:
            raise ValueError(
                f"candidate RGB spatial layout exposes target-side arrays: {forbidden}"
            )
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("candidate RGB spatial layout metadata is invalid") from error
        if not isinstance(metadata, dict):
            raise ValueError("candidate RGB spatial layout metadata is invalid")
        arrays = {
            field: np.asarray(payload[field]).copy()
            for field in _REQUIRED_ARRAY_FIELDS
            if field != "metadata_json"
        }
    return CandidatePoseRGBSpatialLayout(metadata=metadata, **arrays)
