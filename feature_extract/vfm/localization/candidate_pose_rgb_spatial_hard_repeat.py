"""Train-only coherent-repeat edge targets for RGB candidate likelihoods.

The frozen runtime layout remains deliberately target-free.  This artifact is
constructed only after current coherent-wrong pose modes have been mined on
the training split.  Each row pairs one candidate that lands at a query anchor
under the correct pose with a *different* candidate that lands at the same
anchor under a coherent wrong pose.  It is therefore a hard identity target,
not a random support-image permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT = (
    "candidate_pose_rgb_spatial_hard_repeat_targets_v1"
)
CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT = (
    "candidate_pose_rgb_spatial_hard_repeat_targets_v2"
)
_SUPPORTED_HARD_REPEAT_FORMATS = frozenset(
    {
        CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT,
        CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT,
    }
)

_REQUIRED_METADATA_FIELDS = (
    "rgb_spatial_layout_sha256",
    "rgb_spatial_targets_sha256",
    "projection_space_id",
    "descriptor_space_id",
)
_REQUIRED_ARRAY_FIELDS = (
    "source_point_ids",
    "query_ids",
    "pair_ids",
    "positive_candidate_indices",
    "negative_candidate_indices",
    "positive_offsets_xy",
    "negative_offsets_xy",
    "metadata_json",
)


def _metadata_is_train_only(metadata: Mapping[str, object]) -> bool:
    try:
        positive_radius = float(metadata.get("positive_radius_px", 0.0))
        negative_radius = float(metadata.get("negative_radius_px", 0.0))
        candidate_count = int(metadata.get("candidate_count", 0))
        negative_cap = int(metadata.get("max_negatives_per_source_pair", 1))
    except (TypeError, ValueError):
        return False
    artifact_format = str(metadata.get("format", ""))
    multi_negative_contract = (
        artifact_format != CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT
        or (
            metadata.get("negative_selection")
            == "all_or_capped_distinct_coherent_wrong_local_candidates_v1"
            and negative_cap >= 0
        )
    )
    return (
        artifact_format in _SUPPORTED_HARD_REPEAT_FORMATS
        and metadata.get("training_only_target_artifact") is True
        and metadata.get("contains_ground_truth") is True
        and metadata.get("contains_validation_or_test_targets") is False
        and metadata.get("runtime_layout_is_target_free") is True
        and metadata.get("pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer")
        is True
        and metadata.get("render") is False
        and metadata.get("image_retrieval_or_submap_used") is False
        and np.isfinite(positive_radius)
        and np.isfinite(negative_radius)
        and positive_radius > 0.0
        and negative_radius > 0.0
        and candidate_count > 1
        and multi_negative_contract
        and all(str(metadata.get(field, "")).strip() for field in _REQUIRED_METADATA_FIELDS)
    )


@dataclass(frozen=True)
class CandidatePoseRGBSpatialHardRepeatTargets:
    """Train-only correct-edge/coherent-wrong-edge targets.

    V1 contains at most one wrong candidate for every ``(pose-pair, point)``
    tuple.  V2 permits several distinct wrong candidates for that tuple while
    preserving one exact registered positive.  Runtime scoring never loads
    either target artifact.
    """

    source_point_ids: np.ndarray
    query_ids: np.ndarray
    pair_ids: np.ndarray
    positive_candidate_indices: np.ndarray
    negative_candidate_indices: np.ndarray
    positive_offsets_xy: np.ndarray
    negative_offsets_xy: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        pair_ids = np.asarray(self.pair_ids, dtype=np.int64).reshape(-1)
        positive = np.asarray(self.positive_candidate_indices, dtype=np.int64).reshape(-1)
        negative = np.asarray(self.negative_candidate_indices, dtype=np.int64).reshape(-1)
        positive_offsets = np.asarray(self.positive_offsets_xy, dtype=np.float32)
        negative_offsets = np.asarray(self.negative_offsets_xy, dtype=np.float32)
        metadata = dict(self.metadata)
        count = int(len(source_ids))
        candidate_count = int(metadata.get("candidate_count", 0))
        artifact_format = str(metadata.get("format", ""))
        pair_sources = (
            np.stack([pair_ids, source_ids], axis=1)
            if count
            else np.zeros((0, 2), dtype=np.int64)
        )
        edges = (
            np.stack([pair_ids, source_ids, positive, negative], axis=1)
            if count
            else np.zeros((0, 4), dtype=np.int64)
        )
        if (
            count == 0
            or query_ids.shape != (count,)
            or np.any(query_ids == "")
            or pair_ids.shape != (count,)
            or positive.shape != (count,)
            or negative.shape != (count,)
            or positive_offsets.shape != (count, 2)
            or negative_offsets.shape != (count, 2)
            or np.any(positive < 0)
            or np.any(negative < 0)
            or np.any(positive >= candidate_count)
            or np.any(negative >= candidate_count)
            or np.any(positive == negative)
            or (
                artifact_format == CANDIDATE_POSE_RGB_SPATIAL_HARD_REPEAT_FORMAT
                and len(np.unique(pair_sources, axis=0)) != count
            )
            or (
                artifact_format == CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT
                and len(np.unique(edges, axis=0)) != count
            )
            or not np.isfinite(positive_offsets).all()
            or not np.isfinite(negative_offsets).all()
            or not _metadata_is_train_only(metadata)
        ):
            raise ValueError("candidate RGB spatial hard-repeat targets are invalid")
        for pair_id in np.unique(pair_ids).tolist():
            if len(np.unique(query_ids[pair_ids == int(pair_id)])) != 1:
                raise ValueError("hard-repeat pose pair crosses query boundaries")
        if artifact_format == CANDIDATE_POSE_RGB_SPATIAL_MULTI_HARD_REPEAT_FORMAT:
            unique_pair_sources, inverse = np.unique(pair_sources, axis=0, return_inverse=True)
            for group_index in range(len(unique_pair_sources)):
                if len(np.unique(positive[inverse == int(group_index)])) != 1:
                    raise ValueError("multi-negative hard-repeat target changes its positive identity")
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "pair_ids", pair_ids)
        object.__setattr__(self, "positive_candidate_indices", positive)
        object.__setattr__(self, "negative_candidate_indices", negative)
        object.__setattr__(self, "positive_offsets_xy", positive_offsets)
        object.__setattr__(self, "negative_offsets_xy", negative_offsets)
        object.__setattr__(self, "metadata", metadata)

    @property
    def count(self) -> int:
        return int(len(self.source_point_ids))


def select_coherent_hard_repeat_candidates(
    *,
    correct_offsets_xy: np.ndarray,
    correct_valid: np.ndarray,
    wrong_offsets_xy: np.ndarray,
    wrong_valid: np.ndarray,
    candidate_prior_probabilities: np.ndarray,
    positive_radius_px: float,
    negative_radius_px: float,
    positive_candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one true and one different coherent-wrong candidate per point.

    Candidate geometry comes only from a train-only correct/wrong pose join.
    Within each valid set, nearest anchor projection wins; frozen candidate
    prior breaks exact geometric ties without becoming a runtime encoder input.
    """

    source_rows, selected_positive, selected_negative = (
        select_coherent_hard_repeat_candidate_edges(
            correct_offsets_xy=correct_offsets_xy,
            correct_valid=correct_valid,
            wrong_offsets_xy=wrong_offsets_xy,
            wrong_valid=wrong_valid,
            candidate_prior_probabilities=candidate_prior_probabilities,
            positive_radius_px=positive_radius_px,
            negative_radius_px=negative_radius_px,
            positive_candidate_mask=positive_candidate_mask,
            max_negatives_per_source_pair=1,
        )
    )
    row_count = int(np.asarray(correct_offsets_xy).shape[0])
    positive = np.full((row_count,), -1, dtype=np.int64)
    negative = np.full((row_count,), -1, dtype=np.int64)
    positive[source_rows] = selected_positive
    negative[source_rows] = selected_negative
    return positive, negative


def select_coherent_hard_repeat_candidate_edges(
    *,
    correct_offsets_xy: np.ndarray,
    correct_valid: np.ndarray,
    wrong_offsets_xy: np.ndarray,
    wrong_valid: np.ndarray,
    candidate_prior_probabilities: np.ndarray,
    positive_radius_px: float,
    negative_radius_px: float,
    positive_candidate_mask: np.ndarray | None = None,
    max_negatives_per_source_pair: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select exact positives against all or capped coherent-wrong candidates.

    The returned rows refer only to train-only projection joins.  ``0`` for
    ``max_negatives_per_source_pair`` means retain every distinct local wrong
    candidate; a positive cap keeps the closest candidates under the frozen
    distance/prior tie-break used by V1.
    """

    correct_offsets = np.asarray(correct_offsets_xy, dtype=np.float32)
    wrong_offsets = np.asarray(wrong_offsets_xy, dtype=np.float32)
    correct_mask = np.asarray(correct_valid, dtype=bool)
    wrong_mask = np.asarray(wrong_valid, dtype=bool)
    priors = np.asarray(candidate_prior_probabilities, dtype=np.float32)
    positive_mask = (
        None
        if positive_candidate_mask is None
        else np.asarray(positive_candidate_mask, dtype=bool)
    )
    positive_radius = float(positive_radius_px)
    negative_radius = float(negative_radius_px)
    negative_cap = int(max_negatives_per_source_pair)
    if (
        correct_offsets.ndim != 3
        or correct_offsets.shape[2] != 2
        or wrong_offsets.shape != correct_offsets.shape
        or correct_mask.shape != correct_offsets.shape[:2]
        or wrong_mask.shape != correct_mask.shape
        or priors.shape != correct_mask.shape
        or (positive_mask is not None and positive_mask.shape != correct_mask.shape)
        or not np.isfinite(correct_offsets).all()
        or not np.isfinite(wrong_offsets).all()
        or not np.isfinite(priors).all()
        or np.any(priors < 0.0)
        or not np.isfinite(positive_radius)
        or not np.isfinite(negative_radius)
        or positive_radius <= 0.0
        or negative_radius <= 0.0
        or negative_cap < 0
    ):
        raise ValueError("coherent hard-repeat candidate selection inputs are invalid")
    selected_rows: list[int] = []
    selected_positive: list[int] = []
    selected_negative: list[int] = []
    positive_distance = np.max(np.abs(correct_offsets), axis=2)
    negative_distance = np.max(np.abs(wrong_offsets), axis=2)
    for row in range(len(correct_offsets)):
        correct_local = correct_mask[row] & (positive_distance[row] <= positive_radius)
        if positive_mask is None:
            positive_rows = np.flatnonzero(correct_local)
        else:
            # Exact-identity mode requires the registered query observation's
            # own top-L track.  A merely nearby geometric projection is not a
            # candidate-positive for this identity LLR.
            positive_rows = np.flatnonzero(correct_local & positive_mask[row])
        if not len(positive_rows):
            continue
        # np.lexsort uses its last key first: min distance, then max frozen
        # prior, then stable candidate index.
        pos_order = np.lexsort(
            (positive_rows, -priors[row, positive_rows], positive_distance[row, positive_rows])
        )
        pos = int(positive_rows[int(pos_order[0])])
        negative_mask = (
            wrong_mask[row]
            & (negative_distance[row] <= negative_radius)
            & (np.arange(correct_offsets.shape[1]) != pos)
        )
        if positive_mask is None:
            # Legacy geometric targets do not tell us which local candidate is
            # the identity, so retain the conservative exclusion of all
            # correct-pose local candidates.
            negative_mask &= ~correct_local
        else:
            # With exact track labels, another geometrically nearby candidate
            # is precisely the hard repeat we want to reject.  Exclude only
            # registered identity positives, never the whole correct-pose set.
            negative_mask &= ~positive_mask[row]
        negative_rows = np.flatnonzero(negative_mask)
        if not len(negative_rows):
            continue
        neg_order = np.lexsort(
            (negative_rows, -priors[row, negative_rows], negative_distance[row, negative_rows])
        )
        ordered_negative = negative_rows[neg_order]
        if negative_cap > 0:
            ordered_negative = ordered_negative[:negative_cap]
        for negative in ordered_negative.tolist():
            selected_rows.append(int(row))
            selected_positive.append(pos)
            selected_negative.append(int(negative))
    return (
        np.asarray(selected_rows, dtype=np.int64),
        np.asarray(selected_positive, dtype=np.int64),
        np.asarray(selected_negative, dtype=np.int64),
    )


def save_candidate_pose_rgb_spatial_hard_repeat_targets(
    targets: CandidatePoseRGBSpatialHardRepeatTargets, path: Path
) -> None:
    """Atomically serialize a validated train-only hard-repeat artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, suffix=".npz", delete=False
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(
                handle,
                source_point_ids=targets.source_point_ids,
                query_ids=targets.query_ids,
                pair_ids=targets.pair_ids,
                positive_candidate_indices=targets.positive_candidate_indices,
                negative_candidate_indices=targets.negative_candidate_indices,
                positive_offsets_xy=targets.positive_offsets_xy,
                negative_offsets_xy=targets.negative_offsets_xy,
                metadata_json=np.asarray(json.dumps(dict(targets.metadata), sort_keys=True)),
            )
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_candidate_pose_rgb_spatial_hard_repeat_targets(
    path: Path,
) -> CandidatePoseRGBSpatialHardRepeatTargets:
    """Load a hard-repeat artifact while preserving the train-only boundary."""

    with np.load(Path(path), allow_pickle=False) as payload:
        missing = set(_REQUIRED_ARRAY_FIELDS).difference(payload.files)
        if missing:
            raise ValueError(f"candidate RGB spatial hard-repeat targets lack {sorted(missing)}")
        arrays = {
            field: np.asarray(payload[field]).copy()
            for field in _REQUIRED_ARRAY_FIELDS
            if field != "metadata_json"
        }
        try:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("candidate RGB spatial hard-repeat metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError("candidate RGB spatial hard-repeat metadata is invalid")
    return CandidatePoseRGBSpatialHardRepeatTargets(metadata=metadata, **arrays)
