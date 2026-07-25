"""Train-only direct coherent-wrong candidate targets for identity LLR.

The paired runtime layout contains only fixed real-image query/support inputs.
This sidecar records which *different* candidates become locally plausible
under a mined coherent-wrong pose.  It intentionally contains no pose matrix,
projection offset, reprojection residual, or track ID; those were consumed by
the train-only builder before this compact target contract was written.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_IDENTITY_TARGET_FORMAT = (
    "candidate_pose_rgb_spatial_hard_pose_identity_targets_v1"
)
_REQUIRED_METADATA_FIELDS = (
    "hard_pose_identity_pairs_sha256",
    "hard_pose_pairs_sha256",
    "candidate_count",
    "coherent_wrong_local_radius_px",
)


def _metadata_is_train_only(metadata: Mapping[str, object]) -> bool:
    try:
        candidate_count = int(metadata.get("candidate_count", 0))
        radius = float(metadata.get("coherent_wrong_local_radius_px", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        metadata.get("format") == CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_IDENTITY_TARGET_FORMAT
        and metadata.get("training_only_target_artifact") is True
        and metadata.get("contains_ground_truth") is True
        and metadata.get("contains_validation_or_test_targets") is False
        and metadata.get("runtime_scorer_must_not_load_this_artifact") is True
        and metadata.get("runtime_layout_is_target_free") is True
        and metadata.get("pose_projection_or_residual_serialized") is False
        and metadata.get("positive_candidate_is_original_slot_zero") is True
        and metadata.get("render") is False
        and metadata.get("image_retrieval_or_submap_used") is False
        and candidate_count >= 2
        and np.isfinite(radius)
        and radius > 0.0
        and all(str(metadata.get(field, "")).strip() for field in _REQUIRED_METADATA_FIELDS)
    )


@dataclass(frozen=True)
class CandidatePoseRGBSpatialHardPoseIdentityTargets:
    """Different coherent-wrong candidate masks joined only after scoring."""

    anchor_ids: np.ndarray
    query_image_ids: np.ndarray
    hard_negative_candidate_mask: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchor_ids, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_image_ids).astype(str).reshape(-1)
        hard_negative = np.asarray(self.hard_negative_candidate_mask, dtype=bool)
        metadata = dict(self.metadata)
        candidate_count = int(metadata.get("candidate_count", 0))
        count = int(len(anchors))
        if (
            count == 0
            or len(np.unique(anchors)) != count
            or query_ids.shape != (count,)
            or np.any(query_ids == "")
            or hard_negative.shape != (count, candidate_count)
            or np.any(hard_negative[:, 0])
            or np.any(~np.any(hard_negative[:, 1:], axis=1))
            or not _metadata_is_train_only(metadata)
        ):
            raise ValueError("hard-pose identity target arrays or metadata are invalid")
        object.__setattr__(self, "anchor_ids", anchors)
        object.__setattr__(self, "query_image_ids", query_ids)
        object.__setattr__(self, "hard_negative_candidate_mask", hard_negative)
        object.__setattr__(self, "metadata", metadata)

    @property
    def count(self) -> int:
        return int(len(self.anchor_ids))

    @property
    def candidate_count(self) -> int:
        return int(self.hard_negative_candidate_mask.shape[1])


def save_candidate_pose_rgb_spatial_hard_pose_identity_targets(
    targets: CandidatePoseRGBSpatialHardPoseIdentityTargets, path: Path
) -> None:
    """Atomically persist a validated train-only direct hard-negative sidecar."""

    if not isinstance(targets, CandidatePoseRGBSpatialHardPoseIdentityTargets):
        raise TypeError("hard-pose identity target save requires a validated artifact")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".npz", dir=str(output.parent)
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            anchor_ids=targets.anchor_ids,
            query_image_ids=np.asarray(targets.query_image_ids, dtype=np.str_),
            hard_negative_candidate_mask=targets.hard_negative_candidate_mask,
            metadata_json=np.asarray(json.dumps(dict(targets.metadata), sort_keys=True), dtype=np.str_),
        )
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_candidate_pose_rgb_spatial_hard_pose_identity_targets(
    path: Path,
) -> CandidatePoseRGBSpatialHardPoseIdentityTargets:
    """Load the train-only sidecar and reject malformed target boundaries."""

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "anchor_ids",
            "query_image_ids",
            "hard_negative_candidate_mask",
            "metadata_json",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"hard-pose identity targets lack {sorted(missing)}")
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("hard-pose identity target metadata is invalid") from error
        if not isinstance(metadata, dict):
            raise ValueError("hard-pose identity target metadata is invalid")
        return CandidatePoseRGBSpatialHardPoseIdentityTargets(
            anchor_ids=np.asarray(payload["anchor_ids"]),
            query_image_ids=np.asarray(payload["query_image_ids"]),
            hard_negative_candidate_mask=np.asarray(payload["hard_negative_candidate_mask"]),
            metadata=metadata,
        )
