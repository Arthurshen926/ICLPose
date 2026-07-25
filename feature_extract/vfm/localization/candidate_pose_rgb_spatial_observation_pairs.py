"""Train-only observation-pair supervision for RGB spatial likelihood pretraining.

The P1 layout deliberately contains only a few hundred exact top-L identity
hits.  This artifact widens supervision to registered SfM observations from
the train-query images while keeping all targets outside the runtime scorer.
Each row contains one real same-track support observation and a fixed set of
different-track visual hard negatives.  The candidate model receives only
image IDs and image coordinates; track IDs and split labels remain target
fields used by the pretraining command alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PAIR_FORMAT = (
    "candidate_pose_rgb_spatial_observation_pairs_v1"
)

_REQUIRED_METADATA_FIELDS = (
    "train_query_layout_sha256",
    "colmap_images_bin_sha256",
    "support_observation_index_sha256",
    "hard_negative_context_cache_sha256",
    "hard_negative_landmark_bank_sha256",
    "train_query_image_list_sha256",
)


def _validate_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    value = dict(metadata)
    required = (
        value.get("format") == CANDIDATE_POSE_RGB_SPATIAL_OBSERVATION_PAIR_FORMAT
        and value.get("training_only_target_artifact") is True
        and value.get("contains_ground_truth") is True
        and value.get("contains_validation_or_test_targets") is False
        and value.get("runtime_scorer_must_not_load_this_artifact") is True
        and value.get("render") is False
        and value.get("image_retrieval_or_submap_used") is False
        and value.get("candidate_set")
        == "fixed_positive_same_track_plus_radio_pca_global_landmark_hard_negatives"
        and value.get("query_split") == "train_only_inner_partition_v1"
        and value.get("hard_negative_semantics")
        == "radio_intermediate_pca_global_landmark_ann_distinct_track_v1"
        and all(str(value.get(field, "")).strip() for field in _REQUIRED_METADATA_FIELDS)
    )
    if not required:
        raise ValueError("observation-pair metadata violates the train-only contract")
    try:
        negative_count = int(value.get("negative_count", 0))
        fold_count = int(value.get("inner_validation_fold_count", 0))
        fold_index = int(value.get("inner_validation_fold_index", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("observation-pair metadata is malformed") from error
    if negative_count <= 0 or fold_count < 2 or not 0 <= fold_index < fold_count:
        raise ValueError("observation-pair metadata has invalid dimensions")
    return value


@dataclass(frozen=True)
class CandidatePoseRGBSpatialObservationPairs:
    """One fixed positive plus visual hard negatives per query observation."""

    anchor_ids: np.ndarray
    query_image_ids: np.ndarray
    query_xy: np.ndarray
    positive_support_image_ids: np.ndarray
    positive_support_xy: np.ndarray
    positive_track_ids: np.ndarray
    negative_support_image_ids: np.ndarray
    negative_support_xy: np.ndarray
    negative_track_ids: np.ndarray
    negative_sources: np.ndarray
    split_names: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchor_ids, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_image_ids).astype(str).reshape(-1)
        query_xy = np.asarray(self.query_xy, dtype=np.float32)
        positive_ids = np.asarray(self.positive_support_image_ids).astype(str).reshape(-1)
        positive_xy = np.asarray(self.positive_support_xy, dtype=np.float32)
        positive_tracks = np.asarray(self.positive_track_ids, dtype=np.int64).reshape(-1)
        negative_ids = np.asarray(self.negative_support_image_ids).astype(str)
        negative_xy = np.asarray(self.negative_support_xy, dtype=np.float32)
        negative_tracks = np.asarray(self.negative_track_ids, dtype=np.int64)
        negative_sources = np.asarray(self.negative_sources).astype(str)
        splits = np.asarray(self.split_names).astype(str).reshape(-1)
        metadata = _validate_metadata(self.metadata)
        count = int(len(anchors))
        negative_count = int(metadata["negative_count"])
        if (
            count == 0
            or len(np.unique(anchors)) != count
            or query_ids.shape != (count,)
            or query_xy.shape != (count, 2)
            or positive_ids.shape != (count,)
            or positive_xy.shape != (count, 2)
            or positive_tracks.shape != (count,)
            or negative_ids.shape != (count, negative_count)
            or negative_xy.shape != (count, negative_count, 2)
            or negative_tracks.shape != (count, negative_count)
            or negative_sources.shape != (count, negative_count)
            or splits.shape != (count,)
            or np.any(query_ids == "")
            or np.any(positive_ids == "")
            or np.any(negative_ids == "")
            or np.any(negative_sources == "")
            or np.any(positive_tracks < 0)
            or np.any(negative_tracks < 0)
            or np.any(negative_tracks == positive_tracks[:, None])
            or np.any(~np.isfinite(query_xy))
            or np.any(~np.isfinite(positive_xy))
            or np.any(~np.isfinite(negative_xy))
            or set(splits.tolist()) != {"inner_train", "inner_validation"}
        ):
            raise ValueError("observation-pair arrays are invalid")
        object.__setattr__(self, "anchor_ids", anchors)
        object.__setattr__(self, "query_image_ids", query_ids)
        object.__setattr__(self, "query_xy", query_xy)
        object.__setattr__(self, "positive_support_image_ids", positive_ids)
        object.__setattr__(self, "positive_support_xy", positive_xy)
        object.__setattr__(self, "positive_track_ids", positive_tracks)
        object.__setattr__(self, "negative_support_image_ids", negative_ids)
        object.__setattr__(self, "negative_support_xy", negative_xy)
        object.__setattr__(self, "negative_track_ids", negative_tracks)
        object.__setattr__(self, "negative_sources", negative_sources)
        object.__setattr__(self, "split_names", splits)
        object.__setattr__(self, "metadata", metadata)

    @property
    def row_count(self) -> int:
        return int(len(self.anchor_ids))

    @property
    def negative_count(self) -> int:
        return int(self.negative_track_ids.shape[1])


def save_candidate_pose_rgb_spatial_observation_pairs(
    pairs: CandidatePoseRGBSpatialObservationPairs,
    path: Path,
) -> None:
    """Atomically serialize a target-bearing pretraining artifact."""

    if not isinstance(pairs, CandidatePoseRGBSpatialObservationPairs):
        raise TypeError("observation-pair save requires a validated artifact")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".npz", dir=str(output.parent)
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            anchor_ids=pairs.anchor_ids,
            query_image_ids=np.asarray(pairs.query_image_ids, dtype=np.str_),
            query_xy=pairs.query_xy,
            positive_support_image_ids=np.asarray(
                pairs.positive_support_image_ids, dtype=np.str_
            ),
            positive_support_xy=pairs.positive_support_xy,
            positive_track_ids=pairs.positive_track_ids,
            negative_support_image_ids=np.asarray(
                pairs.negative_support_image_ids, dtype=np.str_
            ),
            negative_support_xy=pairs.negative_support_xy,
            negative_track_ids=pairs.negative_track_ids,
            negative_sources=np.asarray(pairs.negative_sources, dtype=np.str_),
            split_names=np.asarray(pairs.split_names, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(pairs.metadata, sort_keys=True), dtype=np.str_),
        )
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_candidate_pose_rgb_spatial_observation_pairs(
    path: Path,
) -> CandidatePoseRGBSpatialObservationPairs:
    """Load and validate a train-only observation-pair artifact."""

    source = Path(path)
    required = {
        "anchor_ids",
        "query_image_ids",
        "query_xy",
        "positive_support_image_ids",
        "positive_support_xy",
        "positive_track_ids",
        "negative_support_image_ids",
        "negative_support_xy",
        "negative_track_ids",
        "negative_sources",
        "split_names",
        "metadata_json",
    }
    with np.load(source, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"observation-pair artifact lacks {sorted(missing)}")
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("observation-pair metadata is invalid") from error
        if not isinstance(metadata, dict):
            raise ValueError("observation-pair metadata must be an object")
        return CandidatePoseRGBSpatialObservationPairs(
            anchor_ids=np.asarray(payload["anchor_ids"]),
            query_image_ids=np.asarray(payload["query_image_ids"]),
            query_xy=np.asarray(payload["query_xy"]),
            positive_support_image_ids=np.asarray(payload["positive_support_image_ids"]),
            positive_support_xy=np.asarray(payload["positive_support_xy"]),
            positive_track_ids=np.asarray(payload["positive_track_ids"]),
            negative_support_image_ids=np.asarray(payload["negative_support_image_ids"]),
            negative_support_xy=np.asarray(payload["negative_support_xy"]),
            negative_track_ids=np.asarray(payload["negative_track_ids"]),
            negative_sources=np.asarray(payload["negative_sources"]),
            split_names=np.asarray(payload["split_names"]),
            metadata=metadata,
        )
