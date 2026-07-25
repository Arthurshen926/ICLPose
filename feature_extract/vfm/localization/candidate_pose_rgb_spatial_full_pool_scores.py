"""Target-free train-pool RGB spatial score artifact contract.

The artifact records only a frozen hypothesis row identity and a score emitted
by the target-free RGB/RADIO likelihood.  It deliberately omits pose matrices,
residuals, target labels, and track identities.  A later train-only builder may
join it to those fields after the score ranking has already been frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


CANDIDATE_POSE_RGB_SPATIAL_TRAIN_FULL_POOL_SCORE_FORMAT = (
    "candidate_pose_rgb_spatial_train_full_pool_scores_v1"
)

_REQUIRED_METADATA = {
    "format": CANDIDATE_POSE_RGB_SPATIAL_TRAIN_FULL_POOL_SCORE_FORMAT,
    "contains_target_fields": False,
    "pose_or_ground_truth_used_for_scoring": False,
    "supervision_arrays_loaded": False,
    "runtime_layout_is_target_free": True,
    "projection_after_network_only": True,
    "train_rows_only": True,
    "render": False,
    "image_retrieval_or_submap_used": False,
    "diagnostic_only": True,
    "promotion_allowed": False,
    "raw_scores_must_not_feed_pnp": True,
}


def validate_candidate_pose_rgb_spatial_train_full_pool_score_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Reject score artifacts that could leak targets into hard-mode mining."""

    values = dict(metadata)
    if any(values.get(key) != expected for key, expected in _REQUIRED_METADATA.items()):
        raise ValueError("train full-pool RGB score metadata violates the target-free contract")
    artifacts = values.get("hypothesis_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("train full-pool RGB score metadata lacks hypothesis lineage")
    for entry in artifacts:
        if (
            not isinstance(entry, Mapping)
            or not str(entry.get("path", ""))
            or not str(entry.get("sha256", ""))
        ):
            raise ValueError("train full-pool RGB score hypothesis lineage is invalid")
    if (
        not str(values.get("hypothesis_semantic_hash", ""))
        or not str(values.get("rgb_spatial_layout_sha256", ""))
        or not isinstance(values.get("checkpoint"), Mapping)
    ):
        raise ValueError("train full-pool RGB score metadata is incomplete")
    # A limited scorer run is useful for validating the distributed inference
    # path, but must never be mistaken for a complete hard-mode mining input.
    if not isinstance(values.get("complete_train_coverage"), bool):
        raise ValueError("train full-pool RGB score coverage declaration is invalid")
    return values


@dataclass(frozen=True)
class CandidatePoseRGBSpatialTrainFullPoolScores:
    """One target-free full-pool score per frozen train hypothesis row."""

    source_artifact_indices: np.ndarray
    source_row_indices: np.ndarray
    query_ids: np.ndarray
    split_names: np.ndarray
    evaluation_labels: np.ndarray
    hypothesis_indices: np.ndarray
    pose_log_likelihood_ratios: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        artifacts = np.asarray(self.source_artifact_indices, dtype=np.int64).reshape(-1)
        rows = np.asarray(self.source_row_indices, dtype=np.int64).reshape(-1)
        queries = np.asarray(self.query_ids).astype(str).reshape(-1)
        splits = np.asarray(self.split_names).astype(str).reshape(-1)
        labels = np.asarray(self.evaluation_labels).astype(str).reshape(-1)
        hypotheses = np.asarray(self.hypothesis_indices, dtype=np.int64).reshape(-1)
        scores = np.asarray(self.pose_log_likelihood_ratios, dtype=np.float32).reshape(-1)
        metadata = validate_candidate_pose_rgb_spatial_train_full_pool_score_metadata(
            self.metadata
        )
        count = len(artifacts)
        if (
            count == 0
            or any(
                value.shape != (count,)
                for value in (rows, queries, splits, labels, hypotheses, scores)
            )
            or np.any(artifacts < 0)
            or np.any(rows < 0)
            or np.any(hypotheses < 0)
            or np.any(queries == "")
            or np.any(labels == "")
            or not np.all(splits == "train")
            or not np.isfinite(scores).all()
        ):
            raise ValueError("train full-pool RGB score arrays are invalid")
        keys = list(zip(artifacts.tolist(), rows.tolist()))
        if len(keys) != len(set(keys)):
            raise ValueError("train full-pool RGB score rows are not unique")
        object.__setattr__(self, "source_artifact_indices", artifacts)
        object.__setattr__(self, "source_row_indices", rows)
        object.__setattr__(self, "query_ids", queries)
        object.__setattr__(self, "split_names", splits)
        object.__setattr__(self, "evaluation_labels", labels)
        object.__setattr__(self, "hypothesis_indices", hypotheses)
        object.__setattr__(self, "pose_log_likelihood_ratios", scores)
        object.__setattr__(self, "metadata", metadata)

    @property
    def row_count(self) -> int:
        return int(len(self.source_artifact_indices))


def save_candidate_pose_rgb_spatial_train_full_pool_scores(
    scores: CandidatePoseRGBSpatialTrainFullPoolScores, path: Path
) -> None:
    """Serialize a train-only score overlay without pose or target arrays."""

    if not isinstance(scores, CandidatePoseRGBSpatialTrainFullPoolScores):
        raise ValueError("train full-pool RGB score artifact type is invalid")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_artifact_indices": scores.source_artifact_indices,
        "source_row_indices": scores.source_row_indices,
        "query_ids": scores.query_ids,
        "split_names": scores.split_names,
        "evaluation_labels": scores.evaluation_labels,
        "hypothesis_indices": scores.hypothesis_indices,
        "pose_log_likelihood_ratios": scores.pose_log_likelihood_ratios,
        "metadata_json": np.asarray(json.dumps(dict(scores.metadata), sort_keys=True), dtype=np.str_),
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


def load_candidate_pose_rgb_spatial_train_full_pool_scores(
    path: Path,
) -> CandidatePoseRGBSpatialTrainFullPoolScores:
    """Load and validate a target-free full-pool train score artifact."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"train full-pool RGB score artifact is absent: {source}")
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "source_artifact_indices",
            "source_row_indices",
            "query_ids",
            "split_names",
            "evaluation_labels",
            "hypothesis_indices",
            "pose_log_likelihood_ratios",
            "metadata_json",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"train full-pool RGB score artifact lacks {sorted(missing)}")
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("train full-pool RGB score metadata is invalid") from error
        return CandidatePoseRGBSpatialTrainFullPoolScores(
            source_artifact_indices=np.asarray(payload["source_artifact_indices"]),
            source_row_indices=np.asarray(payload["source_row_indices"]),
            query_ids=np.asarray(payload["query_ids"]),
            split_names=np.asarray(payload["split_names"]),
            evaluation_labels=np.asarray(payload["evaluation_labels"]),
            hypothesis_indices=np.asarray(payload["hypothesis_indices"]),
            pose_log_likelihood_ratios=np.asarray(payload["pose_log_likelihood_ratios"]),
            metadata=metadata,
        )
