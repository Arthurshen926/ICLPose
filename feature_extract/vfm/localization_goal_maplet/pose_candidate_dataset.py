"""Strict loader for frozen Goal-Maplet pose-candidate datasets.

The full-token pose scorers consume rendered candidate grids, not the sparse
transport hierarchy.  Keeping their loader separate from the trainable
transport loader lets old, hash-sealed v1 candidate grids remain replayable
without weakening the v2-only training contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .lineage import arrays_sha256


POSE_CANDIDATE_DATASET_SCHEMAS = frozenset({
    "goal_maplet_real_sparse_pose_transport_dataset_v1",
    "goal_maplet_real_sparse_pose_transport_dataset_v2",
    "goal_maplet_direct_pose_candidate_dataset_v1",
    "goal_maplet_controlled_pose_training_inventory_v1",
})

DIRECT_POSE_CANDIDATE_DATASET_SCHEMA = "goal_maplet_direct_pose_candidate_dataset_v1"
CONTROLLED_POSE_TRAINING_INVENTORY_SCHEMA = (
    "goal_maplet_controlled_pose_training_inventory_v1"
)

_REQUIRED_ARRAYS = (
    "image_ids",
    "radio_final",
    "source_child_rows",
    "source_child_probabilities",
    "query_reliability",
    "token_xy",
    "candidate_poses_w2c",
    "translation_m",
    "rotation_deg",
    "candidate_valid",
    "target_child_rows",
    "target_child_weights",
    "target_canonical_features",
    "target_modality_valid",
)


def load_pose_candidate_dataset(
    path: Path,
    *,
    require_rendered_targets: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load a replayable candidate grid while preserving version boundaries."""

    dataset_path = Path(path)
    with np.load(dataset_path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("pose candidate dataset lacks metadata")
        metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    if metadata.get("artifact_type") not in POSE_CANDIDATE_DATASET_SCHEMAS:
        raise ValueError("not a replayable pose candidate dataset")
    if metadata.get("content_sha256") != arrays_sha256(arrays):
        raise ValueError("pose candidate dataset content hash differs")
    direct = metadata.get("artifact_type") in {
        DIRECT_POSE_CANDIDATE_DATASET_SCHEMA,
        CONTROLLED_POSE_TRAINING_INVENTORY_SCHEMA,
    }
    if direct and require_rendered_targets:
        raise ValueError("direct pose candidate dataset has no rendered target grids")
    required = (
        (
            "image_ids", "radio_token_paths", "radio_file_sha256",
            "contributor_paths", "contributor_file_sha256",
            "candidate_poses_w2c", "translation_m", "rotation_deg",
            "candidate_valid",
        )
        if direct else _REQUIRED_ARRAYS
    )
    if any(name not in arrays for name in required):
        raise ValueError("pose candidate dataset lacks required arrays")
    required_false = (
        "uses_alike",
        "uses_point_correspondences",
        "uses_pnp",
        "uses_absolute_pose_regression",
    )
    if any(metadata.get(key) is not False for key in required_false):
        raise ValueError("pose candidate dataset violates the method boundary")
    if metadata.get("canonical_map_excludes_query_route") is not True:
        raise ValueError("pose candidate dataset is not map-disjoint")

    image_ids = np.asarray(arrays["image_ids"])
    candidate_valid = np.asarray(arrays["candidate_valid"])
    poses = np.asarray(arrays["candidate_poses_w2c"])
    translation = np.asarray(arrays["translation_m"])
    rotation = np.asarray(arrays["rotation_deg"])
    query_count = int(image_ids.size)
    if candidate_valid.ndim != 2 or candidate_valid.shape[0] != query_count:
        raise ValueError("pose candidate validity shape differs")
    candidate_shape = candidate_valid.shape
    if poses.shape != candidate_shape + (4, 4):
        raise ValueError("pose candidate matrices differ from the inventory")
    if translation.shape != candidate_shape or rotation.shape != candidate_shape:
        raise ValueError("pose candidate error arrays differ from the inventory")
    if np.any(~np.isfinite(poses)) or np.any(~np.isfinite(translation)) or np.any(~np.isfinite(rotation)):
        raise ValueError("pose candidate geometry must be finite")
    if np.any(translation < 0.0) or np.any(rotation < 0.0):
        raise ValueError("pose candidate errors must be nonnegative")
    if not np.all(candidate_valid[:, 0]):
        raise ValueError("diagnostic candidate zero must be present for every query")

    if direct:
        for name in (
            "radio_token_paths", "radio_file_sha256", "contributor_paths",
            "contributor_file_sha256",
        ):
            value = np.asarray(arrays[name])
            if value.shape != (query_count,) or any(not str(item) for item in value.tolist()):
                raise ValueError("direct pose candidate file inventory differs")
        if metadata.get("artifact_type") == DIRECT_POSE_CANDIDATE_DATASET_SCHEMA:
            if metadata.get("candidate_pool_frozen_before_target_pose_opened") is not True:
                raise ValueError("direct pose candidates were not frozen before labels")
        elif (
            metadata.get("uses_gt_for_training_candidate_generation") is not True
            or metadata.get("deployment_candidate_pool") is not False
            or metadata.get("production_eligible") is not False
        ):
            raise ValueError("controlled pose training inventory semantics differ")
        return arrays, metadata

    target_rows = np.asarray(arrays["target_child_rows"])
    target_mass = np.asarray(arrays["target_child_weights"])
    target_features = np.asarray(arrays["target_canonical_features"])
    target_valid = np.asarray(arrays["target_modality_valid"])
    if target_rows.shape[:2] != candidate_shape or target_mass.shape != target_rows.shape:
        raise ValueError("target child grids differ from the candidate inventory")
    if target_features.ndim != target_rows.ndim + 1 or target_features.shape[:-1] != target_rows.shape:
        raise ValueError("target feature grids differ from target children")
    if target_valid.ndim != target_rows.ndim + 1 or target_valid.shape[:-1] != target_rows.shape:
        raise ValueError("target validity grids differ from target children")
    if np.any(~np.isfinite(target_mass)) or np.any(~np.isfinite(target_features)):
        raise ValueError("target pose evidence must be finite")
    if np.any(target_mass < 0.0):
        raise ValueError("target child mass must be nonnegative")
    return arrays, metadata
