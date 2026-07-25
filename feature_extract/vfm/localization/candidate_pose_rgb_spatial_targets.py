"""Train-only targets for target-free candidate RGB spatial layouts.

The runtime RGB layout deliberately contains only fixed query points, global
top-L candidates, and real support observations.  This module defines the
separate target-bearing join used to train a spatial density against correct
and coherent-wrong poses.  Runtime scorers must never load this artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


CANDIDATE_POSE_RGB_SPATIAL_TARGET_FORMAT = "candidate_pose_rgb_spatial_targets_v1"
CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT = (
    "candidate_pose_rgb_spatial_targets_v2"
)
_SUPPORTED_TARGET_FORMATS = frozenset(
    {
        CANDIDATE_POSE_RGB_SPATIAL_TARGET_FORMAT,
        CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
    }
)

_REQUIRED_METADATA_FIELDS = (
    "rgb_spatial_layout_sha256",
    "train_pairs_sha256",
    "support_geometry_index_sha256",
    "projected_landmark_bank_sha256",
    "projection_space_id",
    "descriptor_space_id",
)
_REQUIRED_ARRAY_FIELDS = (
    "source_point_ids",
    "query_ids",
    "spatial_target_offsets_xy",
    "spatial_target_observed",
    "spatial_target_dustbin",
    "pair_query_ids",
    "pair_ids",
    "pair_point_offsets",
    "pair_source_point_ids",
    "correct_projection_offsets_xy",
    "correct_projection_valid",
    "coherent_wrong_projection_offsets_xy",
    "coherent_wrong_projection_valid",
    "metadata_json",
)


def _metadata_is_train_only(metadata: Mapping[str, object]) -> bool:
    try:
        radius = float(metadata.get("spatial_search_radius_px", 0.0))
    except (TypeError, ValueError):
        return False
    base_valid = (
        metadata.get("format") in _SUPPORTED_TARGET_FORMATS
        and metadata.get("training_only_target_artifact") is True
        and metadata.get("contains_ground_truth") is True
        and metadata.get("contains_validation_or_test_targets") is False
        and metadata.get("runtime_layout_is_target_free") is True
        and metadata.get("pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer")
        is True
        and metadata.get("render") is False
        and metadata.get("image_retrieval_or_submap_used") is False
        and np.isfinite(radius)
        and radius > 0.0
        and all(str(metadata.get(field, "")).strip() for field in _REQUIRED_METADATA_FIELDS)
    )
    if not base_valid:
        return False
    if metadata.get("format") == CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT:
        try:
            identity_radius = float(metadata.get("registered_identity_radius_px", 0.0))
        except (TypeError, ValueError):
            return False
        return (
            metadata.get("spatial_supervision_mode") == "registered_exact_identity"
            and metadata.get("spatial_target_semantics")
            == "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
            and metadata.get("spatial_class_balance")
            == "per_batch_observed_dustbin_mean_v1"
            and np.isfinite(identity_radius)
            and identity_radius > 0.0
            and bool(str(metadata.get("colmap_images_sha256", "")).strip())
        )
    return True


@dataclass(frozen=True)
class CandidatePoseRGBSpatialTrainingTargets:
    """Correct/wrong pose offsets and train-only spatial supervision.

    ``source_point_ids`` identify the immutable target-free runtime layout.
    ``pair_*`` arrays expand those points once for every coherent-wrong pose
    mined for a train query.  Pose matrices are intentionally not retained:
    the artifact contains only their projected candidate offsets.
    """

    source_point_ids: np.ndarray
    query_ids: np.ndarray
    spatial_target_offsets_xy: np.ndarray
    spatial_target_observed: np.ndarray
    spatial_target_dustbin: np.ndarray
    pair_query_ids: np.ndarray
    pair_ids: np.ndarray
    pair_point_offsets: np.ndarray
    pair_source_point_ids: np.ndarray
    correct_projection_offsets_xy: np.ndarray
    correct_projection_valid: np.ndarray
    coherent_wrong_projection_offsets_xy: np.ndarray
    coherent_wrong_projection_valid: np.ndarray
    metadata: Mapping[str, object]
    spatial_target_supervised: np.ndarray | None = None

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        target_offsets = np.asarray(self.spatial_target_offsets_xy, dtype=np.float32)
        target_observed = np.asarray(self.spatial_target_observed, dtype=bool)
        target_dustbin = np.asarray(self.spatial_target_dustbin, dtype=bool)
        if self.spatial_target_supervised is None:
            target_supervised = np.ones(target_offsets.shape[:2], dtype=bool)
        else:
            target_supervised = np.asarray(self.spatial_target_supervised, dtype=bool)
        pair_queries = np.asarray(self.pair_query_ids).astype(str).reshape(-1)
        pair_ids = np.asarray(self.pair_ids, dtype=np.int64).reshape(-1)
        pair_offsets = np.asarray(self.pair_point_offsets, dtype=np.int64).reshape(-1)
        pair_source_ids = np.asarray(self.pair_source_point_ids, dtype=np.int64).reshape(-1)
        correct_offsets = np.asarray(
            self.correct_projection_offsets_xy, dtype=np.float32
        )
        correct_valid = np.asarray(self.correct_projection_valid, dtype=bool)
        wrong_offsets = np.asarray(
            self.coherent_wrong_projection_offsets_xy, dtype=np.float32
        )
        wrong_valid = np.asarray(self.coherent_wrong_projection_valid, dtype=bool)
        metadata = dict(self.metadata)

        count = int(len(source_ids))
        pair_count = int(len(pair_ids))
        if (
            count == 0
            or len(np.unique(source_ids)) != count
            or query_ids.shape != (count,)
            or np.any(query_ids == "")
            or target_offsets.ndim != 3
            or target_offsets.shape[0] != count
            or target_offsets.shape[2] != 2
            or target_observed.shape != target_offsets.shape[:2]
            or target_dustbin.shape != target_offsets.shape[:2]
            or target_supervised.shape != target_offsets.shape[:2]
            or np.any(target_observed & ~target_supervised)
            or np.any(target_dustbin & ~target_supervised)
            or not np.isfinite(target_offsets).all()
            or pair_count == 0
            or pair_queries.shape != (pair_count,)
            or np.any(pair_queries == "")
            or len(np.unique(pair_ids)) != pair_count
            or pair_offsets.shape != (pair_count + 1,)
            or pair_offsets[0] != 0
            or pair_offsets[-1] != len(pair_source_ids)
            or np.any(pair_offsets[1:] <= pair_offsets[:-1])
            or correct_offsets.shape
            != (len(pair_source_ids), target_offsets.shape[1], 2)
            or wrong_offsets.shape != correct_offsets.shape
            or correct_valid.shape != correct_offsets.shape[:2]
            or wrong_valid.shape != correct_valid.shape
            or not np.isfinite(correct_offsets).all()
            or not np.isfinite(wrong_offsets).all()
            or not _metadata_is_train_only(metadata)
        ):
            raise ValueError("candidate RGB spatial train-only targets are invalid")

        radius = float(metadata["spatial_search_radius_px"])
        expected_dustbin = target_supervised & (
            (~target_observed) | (np.max(np.abs(target_offsets), axis=2) > radius)
        )
        if not np.array_equal(target_dustbin, expected_dustbin):
            raise ValueError("candidate RGB spatial target dustbin semantics are invalid")

        source_query_by_id = {
            int(source_id): str(query_id)
            for source_id, query_id in zip(source_ids.tolist(), query_ids.tolist())
        }
        if any(int(source_id) not in source_query_by_id for source_id in pair_source_ids):
            raise ValueError("candidate RGB spatial pair references an unknown source point")
        for pair_index, query_id in enumerate(pair_queries.tolist()):
            start, stop = pair_offsets[pair_index : pair_index + 2].tolist()
            group_ids = pair_source_ids[int(start) : int(stop)]
            if (
                len(np.unique(group_ids)) != len(group_ids)
                or any(source_query_by_id[int(source_id)] != str(query_id) for source_id in group_ids)
            ):
                raise ValueError("candidate RGB spatial pair points do not match its query")

        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "spatial_target_offsets_xy", target_offsets)
        object.__setattr__(self, "spatial_target_observed", target_observed)
        object.__setattr__(self, "spatial_target_dustbin", target_dustbin)
        object.__setattr__(self, "spatial_target_supervised", target_supervised)
        object.__setattr__(self, "pair_query_ids", pair_queries)
        object.__setattr__(self, "pair_ids", pair_ids)
        object.__setattr__(self, "pair_point_offsets", pair_offsets)
        object.__setattr__(self, "pair_source_point_ids", pair_source_ids)
        object.__setattr__(self, "correct_projection_offsets_xy", correct_offsets)
        object.__setattr__(self, "correct_projection_valid", correct_valid)
        object.__setattr__(
            self,
            "coherent_wrong_projection_offsets_xy",
            wrong_offsets,
        )
        object.__setattr__(self, "coherent_wrong_projection_valid", wrong_valid)
        object.__setattr__(self, "metadata", metadata)

    @property
    def source_point_count(self) -> int:
        return int(len(self.source_point_ids))

    @property
    def candidate_count(self) -> int:
        return int(self.spatial_target_offsets_xy.shape[1])

    @property
    def pair_count(self) -> int:
        return int(len(self.pair_ids))


def save_candidate_pose_rgb_spatial_training_targets(
    targets: CandidatePoseRGBSpatialTrainingTargets, path: Path
) -> None:
    """Serialize validated train-only targets atomically."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_point_ids": targets.source_point_ids,
        "query_ids": targets.query_ids,
        "spatial_target_offsets_xy": targets.spatial_target_offsets_xy,
        "spatial_target_observed": targets.spatial_target_observed,
        "spatial_target_dustbin": targets.spatial_target_dustbin,
        "spatial_target_supervised": targets.spatial_target_supervised,
        "pair_query_ids": targets.pair_query_ids,
        "pair_ids": targets.pair_ids,
        "pair_point_offsets": targets.pair_point_offsets,
        "pair_source_point_ids": targets.pair_source_point_ids,
        "correct_projection_offsets_xy": targets.correct_projection_offsets_xy,
        "correct_projection_valid": targets.correct_projection_valid,
        "coherent_wrong_projection_offsets_xy": targets.coherent_wrong_projection_offsets_xy,
        "coherent_wrong_projection_valid": targets.coherent_wrong_projection_valid,
        "metadata_json": np.asarray(json.dumps(dict(targets.metadata), sort_keys=True)),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, suffix=".npz", delete=False
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(handle, **payload)
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_candidate_pose_rgb_spatial_training_targets(
    path: Path,
) -> CandidatePoseRGBSpatialTrainingTargets:
    """Load a target artifact and reject a malformed train-only boundary."""

    with np.load(Path(path), allow_pickle=False) as payload:
        missing = set(_REQUIRED_ARRAY_FIELDS).difference(payload.files)
        if missing:
            raise ValueError(f"candidate RGB spatial targets lack {sorted(missing)}")
        arrays = {
            field: np.asarray(payload[field]).copy()
            for field in _REQUIRED_ARRAY_FIELDS
            if field != "metadata_json"
        }
        if "spatial_target_supervised" in payload.files:
            arrays["spatial_target_supervised"] = np.asarray(
                payload["spatial_target_supervised"]
            ).copy()
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("candidate RGB spatial target metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("candidate RGB spatial target metadata is invalid")
    return CandidatePoseRGBSpatialTrainingTargets(metadata=metadata, **arrays)


def project_simple_radial_offsets(
    *,
    xyz: np.ndarray,
    poses_w2c: np.ndarray,
    query_xy: np.ndarray,
    focal_length: float,
    principal_x: float,
    principal_y: float,
    radial_k: float,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project fixed tracks under poses and express them relative to query points."""

    points = np.asarray(xyz, dtype=np.float64)
    poses = np.asarray(poses_w2c, dtype=np.float64)
    query = np.asarray(query_xy, dtype=np.float64)
    if (
        points.ndim not in {2, 3}
        or (points.ndim == 2 and points.shape[1] != 3)
        or (points.ndim == 3 and points.shape[2] != 3)
        or len(points) == 0
        or poses.ndim != 3
        or poses.shape[1:] != (4, 4)
        or len(poses) == 0
        or query.shape != (len(poses), 2)
        or not np.isfinite(points).all()
        or not np.isfinite(poses).all()
        or not np.isfinite(query).all()
        or not np.isfinite(
            [focal_length, principal_x, principal_y, radial_k]
        ).all()
        or float(focal_length) <= 0.0
        or int(image_width) <= 1
        or int(image_height) <= 1
    ):
        raise ValueError("simple-radial projection inputs are invalid")
    if points.ndim == 2:
        points_by_pose = np.broadcast_to(points[None], (len(poses), *points.shape))
    elif points.shape[0] == len(poses):
        points_by_pose = points
    else:
        raise ValueError("per-point simple-radial landmarks must match pose count")
    camera = np.einsum("pij,plj->pli", poses[:, :3, :3], points_by_pose)
    camera += poses[:, None, :3, 3]
    depth = camera[..., 2]
    positive_depth = depth > 1e-8
    safe_depth = np.where(positive_depth, depth, 1.0)
    normalized_x = camera[..., 0] / safe_depth
    normalized_y = camera[..., 1] / safe_depth
    radial = 1.0 + float(radial_k) * (
        normalized_x * normalized_x + normalized_y * normalized_y
    )
    projected = np.stack(
        [
            float(focal_length) * radial * normalized_x + float(principal_x),
            float(focal_length) * radial * normalized_y + float(principal_y),
        ],
        axis=2,
    )
    finite = np.isfinite(projected).all(axis=2)
    in_image = (
        (projected[..., 0] >= 0.0)
        & (projected[..., 0] <= float(int(image_width) - 1))
        & (projected[..., 1] >= 0.0)
        & (projected[..., 1] <= float(int(image_height) - 1))
    )
    valid = positive_depth & finite & in_image
    offsets = projected - query[:, None, :]
    offsets = np.where(np.isfinite(offsets), offsets, 0.0).astype(np.float32)
    return offsets, valid
