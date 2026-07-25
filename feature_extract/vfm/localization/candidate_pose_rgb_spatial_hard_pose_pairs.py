"""Train-only hard-pose rows for candidate-specific RGB spatial likelihood.

Each row contains fixed visual inputs for one query anchor and a small,
candidate-specific support set.  Correct and coherent-wrong pose projections
remain in this train-only artifact; the runtime model receives neither poses
nor track identities and never loads this file.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PAIR_FORMAT = (
    "candidate_pose_rgb_spatial_hard_pose_pairs_v1"
)


def _metadata_is_train_only(metadata: Mapping[str, object]) -> bool:
    try:
        candidate_count = int(metadata.get("candidate_count", 0))
        group_size = int(metadata.get("anchors_per_pose_pair", 0))
        search_radius = float(metadata.get("spatial_search_radius_px", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        metadata.get("format") == CANDIDATE_POSE_RGB_SPATIAL_HARD_POSE_PAIR_FORMAT
        and metadata.get("training_only_target_artifact") is True
        and metadata.get("contains_ground_truth") is True
        and metadata.get("contains_validation_or_test_targets") is False
        and metadata.get("runtime_scorer_must_not_load_this_artifact") is True
        and metadata.get("render") is False
        and metadata.get("image_retrieval_or_submap_used") is False
        and metadata.get("candidate_set")
        == "same_track_mapping_support_plus_radio_pca_global_landmark_ann_hard_negatives"
        and metadata.get("pose_target_semantics")
        == "correct_vs_coherent_wrong_simple_radial_candidate_projection_offsets_v1"
        and candidate_count >= 2
        and group_size >= 2
        and np.isfinite(search_radius)
        and search_radius > 0.0
        and all(
            str(metadata.get(name, "")).strip()
            for name in (
                "source_observation_pairs_sha256",
                "train_pairs_sha256",
                "colmap_images_sha256",
                "colmap_points3d_sha256",
            )
        )
    )


@dataclass(frozen=True)
class CandidatePoseRGBSpatialHardPosePairs:
    """Fixed visual rows and train-only correct/coherent-wrong projections."""

    row_ids: np.ndarray
    pose_pair_ids: np.ndarray
    query_image_ids: np.ndarray
    query_xy: np.ndarray
    candidate_track_ids: np.ndarray
    support_image_ids: np.ndarray
    support_xy: np.ndarray
    correct_projection_offsets_xy: np.ndarray
    correct_projection_valid: np.ndarray
    coherent_wrong_projection_offsets_xy: np.ndarray
    coherent_wrong_projection_valid: np.ndarray
    spatial_target_observed: np.ndarray
    spatial_target_dustbin: np.ndarray
    split_names: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        row_ids = np.asarray(self.row_ids, dtype=np.int64).reshape(-1)
        pose_pair_ids = np.asarray(self.pose_pair_ids, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_image_ids).astype(str).reshape(-1)
        query_xy = np.asarray(self.query_xy, dtype=np.float32)
        tracks = np.asarray(self.candidate_track_ids, dtype=np.int64)
        support_ids = np.asarray(self.support_image_ids).astype(str)
        support_xy = np.asarray(self.support_xy, dtype=np.float32)
        correct_offsets = np.asarray(self.correct_projection_offsets_xy, dtype=np.float32)
        correct_valid = np.asarray(self.correct_projection_valid, dtype=bool)
        wrong_offsets = np.asarray(self.coherent_wrong_projection_offsets_xy, dtype=np.float32)
        wrong_valid = np.asarray(self.coherent_wrong_projection_valid, dtype=bool)
        observed = np.asarray(self.spatial_target_observed, dtype=bool)
        dustbin = np.asarray(self.spatial_target_dustbin, dtype=bool)
        splits = np.asarray(self.split_names).astype(str).reshape(-1)
        metadata = dict(self.metadata)
        count = int(len(row_ids))
        candidate_count = int(metadata.get("candidate_count", 0))
        group_size = int(metadata.get("anchors_per_pose_pair", 0))
        radius = float(metadata.get("spatial_search_radius_px", 0.0))
        if (
            count == 0
            or len(np.unique(row_ids)) != count
            or query_ids.shape != (count,)
            or query_xy.shape != (count, 2)
            or tracks.shape != (count, candidate_count)
            or support_ids.shape != tracks.shape
            or support_xy.shape != (*tracks.shape, 2)
            or correct_offsets.shape != (*tracks.shape, 2)
            or correct_valid.shape != tracks.shape
            or wrong_offsets.shape != (*tracks.shape, 2)
            or wrong_valid.shape != tracks.shape
            or observed.shape != tracks.shape
            or dustbin.shape != tracks.shape
            or splits.shape != (count,)
            or np.any(query_ids == "")
            or np.any(support_ids == "")
            or np.any(tracks < 0)
            or np.any(~np.isfinite(query_xy))
            or np.any(~np.isfinite(support_xy))
            or np.any(~np.isfinite(correct_offsets))
            or np.any(~np.isfinite(wrong_offsets))
            or set(splits.tolist()) != {"inner_train", "inner_validation"}
            or not _metadata_is_train_only(metadata)
        ):
            raise ValueError("hard-pose pair arrays or metadata are invalid")
        negative_duplicates = (
            tracks[:, 1:, None] == tracks[:, None, 1:]
        ) & ~np.eye(candidate_count - 1, dtype=bool)[None]
        if (
            np.any(tracks[:, 1:] == tracks[:, :1])
            or np.any(negative_duplicates)
            or np.any(support_ids == query_ids[:, None])
            or np.any(~observed[:, 0])
            or np.any(observed[:, 1:])
            or np.any(dustbin != ~observed)
            or np.any(~correct_valid[:, 0])
            or np.any(np.max(np.abs(correct_offsets[:, 0]), axis=1) > radius + 1e-4)
        ):
            raise ValueError("hard-pose pair candidate/target contract is invalid")
        groups, group_counts = np.unique(pose_pair_ids, return_counts=True)
        if len(groups) == 0 or np.any(groups < 0) or np.any(group_counts != group_size):
            raise ValueError("hard-pose pair group cardinality is invalid")
        for group_id in groups.tolist():
            rows = np.flatnonzero(pose_pair_ids == int(group_id))
            if len(np.unique(query_ids[rows])) != 1 or len(np.unique(splits[rows])) != 1:
                raise ValueError("hard-pose pair group crosses query or split boundaries")
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "pose_pair_ids", pose_pair_ids)
        object.__setattr__(self, "query_image_ids", query_ids)
        object.__setattr__(self, "query_xy", query_xy)
        object.__setattr__(self, "candidate_track_ids", tracks)
        object.__setattr__(self, "support_image_ids", support_ids)
        object.__setattr__(self, "support_xy", support_xy)
        object.__setattr__(self, "correct_projection_offsets_xy", correct_offsets)
        object.__setattr__(self, "correct_projection_valid", correct_valid)
        object.__setattr__(self, "coherent_wrong_projection_offsets_xy", wrong_offsets)
        object.__setattr__(self, "coherent_wrong_projection_valid", wrong_valid)
        object.__setattr__(self, "spatial_target_observed", observed)
        object.__setattr__(self, "spatial_target_dustbin", dustbin)
        object.__setattr__(self, "split_names", splits)
        object.__setattr__(self, "metadata", metadata)

    @property
    def row_count(self) -> int:
        return int(len(self.row_ids))

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_track_ids.shape[1])

    @property
    def group_count(self) -> int:
        return int(len(np.unique(self.pose_pair_ids)))


def save_candidate_pose_rgb_spatial_hard_pose_pairs(
    pairs: CandidatePoseRGBSpatialHardPosePairs, path: Path
) -> None:
    """Atomically persist the train-only hard-pose contract."""

    if not isinstance(pairs, CandidatePoseRGBSpatialHardPosePairs):
        raise TypeError("hard-pose pair save requires a validated artifact")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".npz", dir=str(output.parent)
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            row_ids=pairs.row_ids,
            pose_pair_ids=pairs.pose_pair_ids,
            query_image_ids=np.asarray(pairs.query_image_ids, dtype=np.str_),
            query_xy=pairs.query_xy,
            candidate_track_ids=pairs.candidate_track_ids,
            support_image_ids=np.asarray(pairs.support_image_ids, dtype=np.str_),
            support_xy=pairs.support_xy,
            correct_projection_offsets_xy=pairs.correct_projection_offsets_xy,
            correct_projection_valid=pairs.correct_projection_valid,
            coherent_wrong_projection_offsets_xy=pairs.coherent_wrong_projection_offsets_xy,
            coherent_wrong_projection_valid=pairs.coherent_wrong_projection_valid,
            spatial_target_observed=pairs.spatial_target_observed,
            spatial_target_dustbin=pairs.spatial_target_dustbin,
            split_names=np.asarray(pairs.split_names, dtype=np.str_),
            metadata_json=np.asarray(json.dumps(pairs.metadata, sort_keys=True), dtype=np.str_),
        )
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_candidate_pose_rgb_spatial_hard_pose_pairs(
    path: Path,
) -> CandidatePoseRGBSpatialHardPosePairs:
    """Load and validate a hard-pose training artifact."""

    source = Path(path)
    required = {
        "row_ids",
        "pose_pair_ids",
        "query_image_ids",
        "query_xy",
        "candidate_track_ids",
        "support_image_ids",
        "support_xy",
        "correct_projection_offsets_xy",
        "correct_projection_valid",
        "coherent_wrong_projection_offsets_xy",
        "coherent_wrong_projection_valid",
        "spatial_target_observed",
        "spatial_target_dustbin",
        "split_names",
        "metadata_json",
    }
    with np.load(source, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"hard-pose pair artifact lacks {sorted(missing)}")
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("hard-pose pair metadata is invalid") from error
        return CandidatePoseRGBSpatialHardPosePairs(
            row_ids=np.asarray(payload["row_ids"]),
            pose_pair_ids=np.asarray(payload["pose_pair_ids"]),
            query_image_ids=np.asarray(payload["query_image_ids"]),
            query_xy=np.asarray(payload["query_xy"]),
            candidate_track_ids=np.asarray(payload["candidate_track_ids"]),
            support_image_ids=np.asarray(payload["support_image_ids"]),
            support_xy=np.asarray(payload["support_xy"]),
            correct_projection_offsets_xy=np.asarray(payload["correct_projection_offsets_xy"]),
            correct_projection_valid=np.asarray(payload["correct_projection_valid"]),
            coherent_wrong_projection_offsets_xy=np.asarray(
                payload["coherent_wrong_projection_offsets_xy"]
            ),
            coherent_wrong_projection_valid=np.asarray(
                payload["coherent_wrong_projection_valid"]
            ),
            spatial_target_observed=np.asarray(payload["spatial_target_observed"]),
            spatial_target_dustbin=np.asarray(payload["spatial_target_dustbin"]),
            split_names=np.asarray(payload["split_names"]),
            metadata=metadata,
        )
