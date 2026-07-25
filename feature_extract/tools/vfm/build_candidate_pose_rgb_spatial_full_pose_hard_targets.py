"""Expand frozen RGB-spatial train targets with diverse full-pose hard modes.

The runtime layout stays target-free and fixed.  This builder is deliberately
train-only: it first validates the frozen global hypothesis shards, then joins
their train rows to a target-side pose-error artifact to identify coherent,
high-score *wrong* poses.  The output serializes only projected local offsets
for those selected poses.  It never serializes pose matrices, pose errors,
target labels, or hypothesis row IDs, so the runtime scorer cannot consume
them by accident.

Existing coherent modes from the base target are retained.  Additional modes
are selected from the complete frozen pose pool by target-free score, subject
to target-side incorrectness and geometric pose diversity.  This closes the
gap between training against a few hand-mined modes and validation against a
large global pose pool.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


# Direct invocation is used by the experiment harness as well as pytest.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from feature_extract.tools.vfm.build_candidate_pose_rgb_spatial_targets import (
    _candidate_xyz_for_layout,
    _load_bank_xyz,
    _query_camera_parameters,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.candidate_pose_llr import (
    grouped_hypothesis_semantic_manifest,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial import (
    CandidatePoseRGBSpatialLayout,
    load_candidate_pose_rgb_spatial_layout,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_full_pool_scores import (
    load_candidate_pose_rgb_spatial_train_full_pool_scores,
)
from feature_extract.vfm.localization.candidate_pose_rgb_spatial_targets import (
    CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT,
    CANDIDATE_POSE_RGB_SPATIAL_TARGET_FORMAT,
    CandidatePoseRGBSpatialTrainingTargets,
    load_candidate_pose_rgb_spatial_training_targets,
    project_simple_radial_offsets,
    save_candidate_pose_rgb_spatial_training_targets,
)


_HYPOTHESIS_FORMAT = "grouped_pose_hypotheses_inference_only_v1"
_TARGET_JOIN_FORMAT = "grouped_pose_hypothesis_targets_v1"
_FROZEN_SCORE_FIELDS = (
    "preliminary_log_likelihood_means",
    "shortlist_log_likelihood_means",
    "verification_log_likelihood_means",
)
_GEOMETRY_PROJECTED_SPATIAL_SEMANTICS = (
    "correct_pose_projected_offset_inside_fixed_local_support_or_dustbin_v1"
)
_REGISTERED_IDENTITY_SPATIAL_SEMANTICS = (
    "registered_query_observation_exact_track_local_offset_or_explicit_null_dustbin_v1"
)


def resolve_full_pose_hard_base_spatial_semantics(
    metadata: Mapping[str, object],
) -> tuple[str, str]:
    """Accept only explicit correct-pose projection target contracts.

    Full-pose supervision consumes the correct-pose and coherent-wrong local
    projection tables, not a particular per-candidate density label.  Both
    current target contracts provide those tables: the dense geometry-projected
    target used by the high-resolution likelihood and the older registered
    exact-identity target.  Keeping their formats distinct matters because the
    latter has sparse identity labels while the former does not.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("full-pose hard target base metadata is invalid")
    target_format = str(metadata.get("format", "")).strip()
    target_semantics = str(metadata.get("spatial_target_semantics", "")).strip()
    target_mode = str(metadata.get("spatial_supervision_mode", "")).strip()
    if (
        target_format == CANDIDATE_POSE_RGB_SPATIAL_TARGET_FORMAT
        and target_mode in ("", "geometry_projected")
        and target_semantics == _GEOMETRY_PROJECTED_SPATIAL_SEMANTICS
    ):
        return "geometry_projected", target_semantics
    if (
        target_format == CANDIDATE_POSE_RGB_SPATIAL_IDENTITY_TARGET_FORMAT
        and target_mode == "registered_exact_identity"
        and target_semantics == _REGISTERED_IDENTITY_SPATIAL_SEMANTICS
    ):
        return "registered_exact_identity", target_semantics
    raise ValueError(
        "full-pose hard targets require either the geometry-projected local-density "
        "contract or the registered exact-identity contract"
    )


@dataclass(frozen=True)
class _CanonicalQueryProjection:
    """Correct-pose projection table in one stable source-point order."""

    query_id: str
    source_point_ids: np.ndarray
    layout_rows: np.ndarray
    correct_offsets_xy: np.ndarray
    correct_valid: np.ndarray

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        layout_rows = np.asarray(self.layout_rows, dtype=np.int64).reshape(-1)
        offsets = np.asarray(self.correct_offsets_xy, dtype=np.float32)
        valid = np.asarray(self.correct_valid, dtype=bool)
        if (
            not str(self.query_id)
            or len(source_ids) == 0
            or len(np.unique(source_ids)) != len(source_ids)
            or layout_rows.shape != source_ids.shape
            or np.any(layout_rows < 0)
            or offsets.ndim != 3
            or offsets.shape[0] != len(source_ids)
            or offsets.shape[2] != 2
            or valid.shape != offsets.shape[:2]
            or not np.isfinite(offsets).all()
        ):
            raise ValueError("canonical RGB spatial query projection is invalid")
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "layout_rows", layout_rows)
        object.__setattr__(self, "correct_offsets_xy", offsets)
        object.__setattr__(self, "correct_valid", valid)


@dataclass(frozen=True)
class _TargetJoinedTrainHypotheses:
    """One train query's target-free hypotheses plus train-only target join."""

    query_id: str
    source_artifact_index: int
    source_row_indices: np.ndarray
    poses_w2c: np.ndarray
    scores: np.ndarray
    translation_errors_m: np.ndarray
    rotation_errors_deg: np.ndarray
    correct_10cm_5deg: np.ndarray

    def __post_init__(self) -> None:
        rows = np.asarray(self.source_row_indices, dtype=np.int64).reshape(-1)
        poses = np.asarray(self.poses_w2c, dtype=np.float64)
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        translation = np.asarray(self.translation_errors_m, dtype=np.float64).reshape(-1)
        rotation = np.asarray(self.rotation_errors_deg, dtype=np.float64).reshape(-1)
        correct = np.asarray(self.correct_10cm_5deg, dtype=bool).reshape(-1)
        count = len(rows)
        if (
            not str(self.query_id)
            or int(self.source_artifact_index) < 0
            or count == 0
            or len(np.unique(rows)) != count
            or np.any(rows < 0)
            or poses.shape != (count, 4, 4)
            or scores.shape != (count,)
            or translation.shape != (count,)
            or rotation.shape != (count,)
            or correct.shape != (count,)
            or not np.isfinite(poses).all()
            or not np.isfinite(translation).all()
            or not np.isfinite(rotation).all()
            or np.any(translation < 0.0)
            or np.any(rotation < 0.0)
        ):
            raise ValueError("target-joined train hypotheses are invalid")
        object.__setattr__(self, "source_row_indices", rows)
        object.__setattr__(self, "poses_w2c", poses)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "translation_errors_m", translation)
        object.__setattr__(self, "rotation_errors_deg", rotation)
        object.__setattr__(self, "correct_10cm_5deg", correct)


def _camera_centers_w2c(poses_w2c: np.ndarray) -> np.ndarray:
    """Return camera centers for finite world-to-camera pose matrices."""

    poses = np.asarray(poses_w2c, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("pose matrices are invalid")
    centers = -np.einsum("nij,nj->ni", np.swapaxes(poses[:, :3, :3], 1, 2), poses[:, :3, 3])
    if not np.isfinite(centers).all():
        raise ValueError("pose camera centers are invalid")
    return centers


def _rotation_distance_degrees(first_w2c: np.ndarray, second_w2c: np.ndarray) -> float:
    """Return the angular distance between two valid world-to-camera rotations."""

    first = np.asarray(first_w2c, dtype=np.float64)
    second = np.asarray(second_w2c, dtype=np.float64)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("pose rotation distance inputs are invalid")
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = float((np.trace(relative) - 1.0) * 0.5)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def select_diverse_full_pose_hard_mode_indices(
    *,
    poses_w2c: np.ndarray,
    scores: np.ndarray,
    translation_errors_m: np.ndarray,
    rotation_errors_deg: np.ndarray,
    correct_10cm_5deg: np.ndarray,
    source_row_indices: np.ndarray,
    max_modes: int,
    score_pool_size: int,
    minimum_translation_error_m: float,
    minimum_rotation_error_deg: float,
    minimum_pose_translation_diversity_m: float,
    minimum_pose_rotation_diversity_deg: float,
) -> np.ndarray:
    """Choose target-side wrong modes without using target fields for ranking.

    Scores and pose matrices come from the frozen inference artifact.  Target
    values only exclude correct or nearly-correct poses from training negatives.
    The greedy diversity rule prevents a dense cluster of almost identical PnP
    outputs from consuming the entire mode budget.
    """

    poses = np.asarray(poses_w2c, dtype=np.float64)
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    translation = np.asarray(translation_errors_m, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation_errors_deg, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct_10cm_5deg, dtype=bool).reshape(-1)
    source_rows = np.asarray(source_row_indices, dtype=np.int64).reshape(-1)
    count = len(score_values)
    limit = int(max_modes)
    pool_size = int(score_pool_size)
    thresholds = (
        float(minimum_translation_error_m),
        float(minimum_rotation_error_deg),
        float(minimum_pose_translation_diversity_m),
        float(minimum_pose_rotation_diversity_deg),
    )
    if (
        poses.shape != (count, 4, 4)
        or translation.shape != (count,)
        or rotation.shape != (count,)
        or correct.shape != (count,)
        or source_rows.shape != (count,)
        or count == 0
        or len(np.unique(source_rows)) != count
        or np.any(source_rows < 0)
        or not np.isfinite(poses).all()
        or not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
        or np.any(translation < 0.0)
        or np.any(rotation < 0.0)
        or limit <= 0
        or pool_size <= 0
        or any(not math.isfinite(value) or value < 0.0 for value in thresholds)
    ):
        raise ValueError("full-pose hard-mode selection inputs are invalid")
    eligible = (
        np.isfinite(score_values)
        & ~correct
        & (
            (translation >= float(minimum_translation_error_m))
            | (rotation >= float(minimum_rotation_error_deg))
        )
    )
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        return np.zeros((0,), dtype=np.int64)
    # Last lexsort key wins: descending frozen score, then immutable source row.
    ordered = candidates[
        np.lexsort((source_rows[candidates], -score_values[candidates]))
    ][:pool_size]
    centers = _camera_centers_w2c(poses)
    selected: list[int] = []
    for index in ordered.tolist():
        is_diverse = True
        for previous in selected:
            translation_distance = float(np.linalg.norm(centers[index] - centers[previous]))
            rotation_distance = _rotation_distance_degrees(poses[index], poses[previous])
            if (
                translation_distance < float(minimum_pose_translation_diversity_m)
                and rotation_distance < float(minimum_pose_rotation_diversity_deg)
            ):
                is_diverse = False
                break
        if is_diverse:
            selected.append(int(index))
            if len(selected) >= limit:
                break
    return np.asarray(selected, dtype=np.int64)


def _load_metadata(payload: Mapping[str, Any], *, name: str) -> dict[str, object]:
    try:
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{name} metadata is invalid")
    return metadata


def _load_target_free_score_overlay(
    *,
    path: Path,
    hypothesis_artifacts: Sequence[Path],
    expected_semantic_hash: str,
    layout_sha256: str,
    evaluation_label: str,
) -> tuple[dict[tuple[int, int], tuple[str, str, int, float]], dict[str, object]]:
    """Load a current-model score overlay without exposing target fields.

    The overlay is keyed solely by the immutable frozen artifact/row identity.
    It may rank train hypotheses, but does not contain the corresponding pose
    matrices or target-side error labels.  Those values remain unavailable
    until the target join below.
    """

    overlay_path = Path(path)
    overlay = load_candidate_pose_rgb_spatial_train_full_pool_scores(overlay_path)
    metadata = dict(overlay.metadata)
    if (
        str(metadata.get("rgb_spatial_layout_sha256", "")) != str(layout_sha256)
        or str(metadata.get("hypothesis_semantic_hash", "")) != str(expected_semantic_hash)
        or str(metadata.get("evaluation_label", "")) != str(evaluation_label)
    ):
        raise ValueError("target-free full-pool score overlay lineage differs")
    if (
        metadata.get("complete_train_coverage") is not True
        or metadata.get("score_component") != "combined"
        or metadata.get("fixed_global_topl") is not True
        or metadata.get("explicit_null") is not True
    ):
        raise ValueError("target-free full-pool score overlay is not a complete combined likelihood")
    recorded = metadata.get("hypothesis_artifacts")
    paths = tuple(Path(value) for value in hypothesis_artifacts)
    if not isinstance(recorded, list) or len(recorded) != len(paths):
        raise ValueError("target-free full-pool score overlay source manifest is invalid")
    for index, source in enumerate(paths):
        entry = recorded[index]
        if (
            not isinstance(entry, Mapping)
            or Path(str(entry.get("path", ""))).resolve() != source.resolve()
            or str(entry.get("sha256", "")) != file_sha256_short(source)
        ):
            raise ValueError("target-free full-pool score overlay source is stale")
    output: dict[tuple[int, int], tuple[str, str, int, float]] = {}
    for artifact_index, row_index, query_id, label, hypothesis_index, score in zip(
        overlay.source_artifact_indices.tolist(),
        overlay.source_row_indices.tolist(),
        overlay.query_ids.astype(str).tolist(),
        overlay.evaluation_labels.astype(str).tolist(),
        overlay.hypothesis_indices.tolist(),
        overlay.pose_log_likelihood_ratios.tolist(),
    ):
        key = (int(artifact_index), int(row_index))
        if key in output:
            raise ValueError("target-free full-pool score overlay has duplicate source rows")
        output[key] = (str(query_id), str(label), int(hypothesis_index), float(score))
    if not output:
        raise ValueError("target-free full-pool score overlay is empty")
    return output, {
        "path": str(overlay_path),
        "sha256": file_sha256_short(overlay_path),
        "format": str(metadata["format"]),
        "checkpoint": dict(metadata["checkpoint"]),
        "score_component": str(metadata.get("score_component", "")),
        "row_count": int(overlay.row_count),
    }


def _canonical_query_projections(
    *,
    layout: CandidatePoseRGBSpatialLayout,
    targets: CandidatePoseRGBSpatialTrainingTargets,
) -> dict[str, _CanonicalQueryProjection]:
    """Validate that all legacy modes share an identical correct projection."""

    layout_row_by_source = {
        int(source_id): row
        for row, source_id in enumerate(np.asarray(layout.source_point_ids, dtype=np.int64).tolist())
    }
    output: dict[str, _CanonicalQueryProjection] = {}
    for pair_index, query_id in enumerate(np.asarray(targets.pair_query_ids).astype(str).tolist()):
        start, stop = targets.pair_point_offsets[pair_index : pair_index + 2].tolist()
        source_ids = np.asarray(targets.pair_source_point_ids[int(start) : int(stop)], dtype=np.int64)
        if len(source_ids) == 0 or len(np.unique(source_ids)) != len(source_ids):
            raise ValueError("base RGB spatial target pair source IDs are invalid")
        try:
            layout_rows = np.asarray(
                [layout_row_by_source[int(source_id)] for source_id in source_ids.tolist()], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError("base RGB spatial target pair source is absent from layout") from error
        if (
            np.any(np.asarray(layout.split_names[layout_rows]).astype(str) != "train")
            or np.any(np.asarray(layout.query_ids[layout_rows]).astype(str) != str(query_id))
        ):
            raise ValueError("base RGB spatial target pair is not train-only")
        offsets = np.asarray(targets.correct_projection_offsets_xy[int(start) : int(stop)], dtype=np.float32)
        valid = np.asarray(targets.correct_projection_valid[int(start) : int(stop)], dtype=bool)
        existing = output.get(str(query_id))
        if existing is None:
            output[str(query_id)] = _CanonicalQueryProjection(
                query_id=str(query_id),
                source_point_ids=source_ids,
                layout_rows=layout_rows,
                correct_offsets_xy=offsets,
                correct_valid=valid,
            )
            continue
        if set(existing.source_point_ids.tolist()) != set(source_ids.tolist()):
            raise ValueError("base RGB spatial wrong modes cover different source points")
        pair_position = {int(source_id): position for position, source_id in enumerate(source_ids.tolist())}
        reorder = np.asarray(
            [pair_position[int(source_id)] for source_id in existing.source_point_ids.tolist()], dtype=np.int64
        )
        if not (
            np.allclose(existing.correct_offsets_xy, offsets[reorder], atol=1e-5, rtol=0.0)
            and np.array_equal(existing.correct_valid, valid[reorder])
        ):
            raise ValueError("base RGB spatial wrong modes disagree on correct projection")
    if not output:
        raise ValueError("base RGB spatial targets have no train query projections")
    return output


def _load_target_joined_train_hypotheses(
    *,
    hypothesis_artifacts: Sequence[Path],
    hypothesis_target_join: Path,
    evaluation_label: str,
    expected_semantic_hash: str,
    frozen_score_field: str,
    target_free_score_overlay: Path | None = None,
    layout_sha256: str = "",
) -> tuple[dict[str, _TargetJoinedTrainHypotheses], dict[str, object]]:
    """Read frozen inference rows and verify their target-side join row by row."""

    paths = tuple(Path(path) for path in hypothesis_artifacts)
    join_path = Path(hypothesis_target_join)
    label = str(evaluation_label)
    score_field = str(frozen_score_field)
    if (
        not paths
        or len(paths) != len(set(paths))
        or not label
        or score_field not in _FROZEN_SCORE_FIELDS
    ):
        raise ValueError("full-pose hard-target sources are invalid")
    overlay_scores: dict[tuple[int, int], tuple[str, str, int, float]] | None = None
    overlay_lineage: dict[str, object] | None = None
    if target_free_score_overlay is not None:
        if not expected_semantic_hash or not layout_sha256:
            raise ValueError("target-free score overlay requires explicit source lineage")
        overlay_scores, overlay_lineage = _load_target_free_score_overlay(
            path=Path(target_free_score_overlay),
            hypothesis_artifacts=paths,
            expected_semantic_hash=expected_semantic_hash,
            layout_sha256=layout_sha256,
            evaluation_label=label,
        )
    if not join_path.is_file():
        raise FileNotFoundError(f"hypothesis target join is absent: {join_path}")
    with np.load(join_path, allow_pickle=False) as payload:
        required = {
            "query_ids",
            "split_names",
            "hypothesis_indices",
            "source_artifact_indices",
            "source_row_indices",
            "translation_errors_m",
            "rotation_errors_deg",
            "correct_10cm_5deg",
            "metadata_json",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"hypothesis target join lacks {sorted(missing)}")
        joined = {name: np.asarray(payload[name]).copy() for name in required if name != "metadata_json"}
        join_metadata = _load_metadata(payload, name="hypothesis target join")
    if (
        join_metadata.get("format") != _TARGET_JOIN_FORMAT
        or join_metadata.get("targets_joined_after_inference") is not True
    ):
        raise ValueError("hypothesis target join does not preserve after-inference targets")
    recorded_paths = join_metadata.get("inference_artifacts")
    recorded_hashes = join_metadata.get("inference_artifact_sha256")
    if (
        not isinstance(recorded_paths, list)
        or not isinstance(recorded_hashes, list)
        or len(recorded_paths) != len(paths)
        or len(recorded_hashes) != len(paths)
    ):
        raise ValueError("hypothesis target join source manifest is invalid")
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"frozen hypothesis artifact is absent: {path}")
        if Path(str(recorded_paths[index])).resolve() != path.resolve():
            raise ValueError("hypothesis target join source artifact order differs")
        if str(recorded_hashes[index]) != file_sha256_short(path):
            raise ValueError("hypothesis target join source artifact is stale")

    joined_query = np.asarray(joined["query_ids"]).astype(str)
    joined_split = np.asarray(joined["split_names"]).astype(str)
    joined_hypothesis = np.asarray(joined["hypothesis_indices"], dtype=np.int64)
    joined_artifact = np.asarray(joined["source_artifact_indices"], dtype=np.int64)
    joined_row = np.asarray(joined["source_row_indices"], dtype=np.int64)
    joined_translation = np.asarray(joined["translation_errors_m"], dtype=np.float64)
    joined_rotation = np.asarray(joined["rotation_errors_deg"], dtype=np.float64)
    joined_correct = np.asarray(joined["correct_10cm_5deg"], dtype=bool)
    row_count = len(joined_query)
    if (
        row_count == 0
        or any(value.shape != (row_count,) for value in (
            joined_split,
            joined_hypothesis,
            joined_artifact,
            joined_row,
            joined_translation,
            joined_rotation,
            joined_correct,
        ))
        or np.any(joined_query == "")
        or np.any(joined_artifact < 0)
        or np.any(joined_row < 0)
        or not np.isfinite(joined_translation).all()
        or not np.isfinite(joined_rotation).all()
        or np.any(joined_translation < 0.0)
        or np.any(joined_rotation < 0.0)
    ):
        raise ValueError("hypothesis target join arrays are invalid")

    pools: dict[str, _TargetJoinedTrainHypotheses] = {}
    source_manifest: list[dict[str, object]] = []
    expected_overlay_keys: set[tuple[int, int]] = set()
    for artifact_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "query_ids",
                "split_names",
                "evaluation_labels",
                "hypothesis_indices",
                "poses_w2c",
                score_field,
                "metadata_json",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"frozen hypothesis artifact lacks {sorted(missing)}")
            source = {name: np.asarray(payload[name]).copy() for name in required if name != "metadata_json"}
            source_metadata = _load_metadata(payload, name="frozen hypothesis artifact")
        if (
            source_metadata.get("format") != _HYPOTHESIS_FORMAT
            or source_metadata.get("contains_target_fields") is not False
        ):
            raise ValueError("full-pose source is not a target-free hypothesis artifact")
        manifest = grouped_hypothesis_semantic_manifest(source_metadata)
        if expected_semantic_hash and str(manifest["semantic_hash"]) != expected_semantic_hash:
            raise ValueError("full-pose source semantic lineage differs from base targets")
        query_ids = np.asarray(source["query_ids"]).astype(str)
        split_names = np.asarray(source["split_names"]).astype(str)
        labels = np.asarray(source["evaluation_labels"]).astype(str)
        hypothesis_indices = np.asarray(source["hypothesis_indices"], dtype=np.int64)
        poses = np.asarray(source["poses_w2c"], dtype=np.float64)
        scores = np.asarray(source[score_field], dtype=np.float64)
        count = len(query_ids)
        if (
            count == 0
            or split_names.shape != (count,)
            or labels.shape != (count,)
            or hypothesis_indices.shape != (count,)
            or poses.shape != (count, 4, 4)
            or scores.shape != (count,)
            or np.any(query_ids == "")
            or not np.isfinite(poses).all()
        ):
            raise ValueError("frozen hypothesis source arrays are invalid")
        target_rows = np.flatnonzero(joined_artifact == artifact_index)
        target_source_rows = joined_row[target_rows]
        if (
            len(target_rows) != count
            or np.any(target_source_rows >= count)
            or len(np.unique(target_source_rows)) != count
        ):
            raise ValueError("hypothesis target join does not cover source rows exactly once")
        target_position_by_source_row = np.full((count,), -1, dtype=np.int64)
        target_position_by_source_row[target_source_rows] = target_rows
        target_positions = target_position_by_source_row[np.arange(count, dtype=np.int64)]
        if (
            np.any(target_positions < 0)
            or not np.array_equal(joined_query[target_positions], query_ids)
            or not np.array_equal(joined_split[target_positions], split_names)
            or not np.array_equal(joined_hypothesis[target_positions], hypothesis_indices)
        ):
            raise ValueError("hypothesis target join/source rows do not agree")
        source_manifest.append(
            {
                "path": str(path),
                "sha256": file_sha256_short(path),
                "semantic_hash": str(manifest["semantic_hash"]),
            }
        )
        train_mask = (split_names == "train") & (labels == label)
        if overlay_scores is not None:
            current_scores = np.full((count,), np.nan, dtype=np.float64)
            for row_index in np.flatnonzero(train_mask).tolist():
                key = (int(artifact_index), int(row_index))
                expected_overlay_keys.add(key)
                try:
                    scored_query, scored_label, scored_hypothesis, scored_value = overlay_scores[key]
                except KeyError as error:
                    raise ValueError("target-free full-pool score overlay lacks a train source row") from error
                if (
                    scored_query != str(query_ids[row_index])
                    or scored_label != str(labels[row_index])
                    or scored_hypothesis != int(hypothesis_indices[row_index])
                    or not math.isfinite(float(scored_value))
                ):
                    raise ValueError("target-free full-pool score overlay row identity differs")
                current_scores[row_index] = float(scored_value)
            scores = current_scores
        for query_id in np.unique(query_ids[train_mask]).tolist():
            rows = np.flatnonzero(train_mask & (query_ids == str(query_id)))
            if len(rows) == 0:
                continue
            if str(query_id) in pools:
                raise ValueError("frozen train query appears in more than one hypothesis shard")
            targets_for_rows = target_positions[rows]
            pools[str(query_id)] = _TargetJoinedTrainHypotheses(
                query_id=str(query_id),
                source_artifact_index=artifact_index,
                source_row_indices=rows,
                poses_w2c=poses[rows],
                scores=scores[rows],
                translation_errors_m=joined_translation[targets_for_rows],
                rotation_errors_deg=joined_rotation[targets_for_rows],
                correct_10cm_5deg=joined_correct[targets_for_rows],
            )
    if not pools:
        raise ValueError("full-pose target join has no train hypotheses for evaluation label")
    if overlay_scores is not None and set(overlay_scores) != expected_overlay_keys:
        raise ValueError("target-free full-pool score overlay does not exactly cover train source rows")
    return pools, {
        "hypothesis_target_join": str(join_path),
        "hypothesis_target_join_sha256": file_sha256_short(join_path),
        "hypothesis_artifacts": source_manifest,
        "evaluation_label": label,
        "frozen_score_field": score_field,
        "score_source": (
            "current_target_free_rgb_full_pool_score_overlay_v1"
            if overlay_scores is not None
            else f"frozen_hypothesis_{score_field}_v1"
        ),
        "target_free_score_overlay": overlay_lineage,
    }


def _quantiles(values: Sequence[float]) -> list[float] | None:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return None
    if not np.isfinite(array).all():
        raise ValueError("hard-mode audit values are non-finite")
    return [float(value) for value in np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist()]


def build_candidate_pose_rgb_spatial_full_pose_hard_targets(
    *,
    rgb_spatial_layout: Path,
    base_training_targets: Path,
    hypothesis_artifacts: Sequence[Path],
    hypothesis_target_join: Path,
    projected_landmark_bank: Path,
    colmap_model_dir: Path,
    evaluation_label: str,
    frozen_score_field: str,
    target_free_score_overlay: Path | None,
    additional_modes_per_query: int,
    minimum_additional_modes_per_query: int,
    score_pool_size: int,
    minimum_translation_error_m: float,
    minimum_rotation_error_deg: float,
    minimum_pose_translation_diversity_m: float,
    minimum_pose_rotation_diversity_deg: float,
    output: Path,
    summary_json: Path,
    force: bool,
) -> dict[str, object]:
    """Append full-pool train-only hard modes to a compatible projection target."""

    layout_path = Path(rgb_spatial_layout)
    base_path = Path(base_training_targets)
    bank_path = Path(projected_landmark_bank)
    output_path = Path(output)
    summary_path = Path(summary_json)
    if (output_path.exists() or summary_path.exists()) and not bool(force):
        raise FileExistsError("refusing to overwrite full-pose hard target output")
    if int(additional_modes_per_query) <= 0 or int(minimum_additional_modes_per_query) <= 0:
        raise ValueError("full-pose hard-mode counts must be positive")
    if int(minimum_additional_modes_per_query) > int(additional_modes_per_query):
        raise ValueError("minimum full-pose hard-mode count exceeds requested count")
    if int(score_pool_size) <= 0:
        raise ValueError("full-pose hard-mode score pool must be positive")

    layout = load_candidate_pose_rgb_spatial_layout(layout_path)
    base = load_candidate_pose_rgb_spatial_training_targets(base_path)
    layout_sha256 = file_sha256_short(layout_path)
    if str(base.metadata.get("rgb_spatial_layout_sha256", "")) != layout_sha256:
        raise ValueError("full-pose hard targets require a current frozen-layout base target")
    base_spatial_mode, base_spatial_semantics = resolve_full_pose_hard_base_spatial_semantics(
        base.metadata
    )
    if file_sha256_short(bank_path) != str(base.metadata.get("projected_landmark_bank_sha256", "")):
        raise ValueError("full-pose hard targets would use a stale projected landmark bank")
    train_layout_rows = np.flatnonzero(np.asarray(layout.split_names).astype(str) == "train")
    if (
        len(train_layout_rows) == 0
        or set(np.asarray(layout.source_point_ids[train_layout_rows], dtype=np.int64).tolist())
        != set(np.asarray(base.source_point_ids, dtype=np.int64).tolist())
        or set(np.asarray(base.query_ids).astype(str).tolist())
        != set(np.asarray(layout.query_ids[train_layout_rows]).astype(str).tolist())
    ):
        raise ValueError("base targets do not exactly cover frozen train layout rows")
    canonical_queries = _canonical_query_projections(layout=layout, targets=base)
    expected_semantic_hash = str(
        dict(base.metadata.get("hypothesis_semantic_lineage", {})).get("semantic_hash", "")
    )
    pools, source_lineage = _load_target_joined_train_hypotheses(
        hypothesis_artifacts=hypothesis_artifacts,
        hypothesis_target_join=Path(hypothesis_target_join),
        evaluation_label=str(evaluation_label),
        expected_semantic_hash=expected_semantic_hash,
        frozen_score_field=str(frozen_score_field),
        target_free_score_overlay=target_free_score_overlay,
        layout_sha256=layout_sha256,
    )
    if set(canonical_queries) != set(pools):
        missing = sorted(set(canonical_queries).difference(pools))
        extra = sorted(set(pools).difference(canonical_queries))
        raise ValueError(
            "full-pose target pool/train layout query coverage differs: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    selected_candidate_indices_by_query: dict[str, np.ndarray] = {}
    selected_ranks: list[float] = []
    selected_translation: list[float] = []
    selected_rotation: list[float] = []
    selected_counts: list[float] = []
    for query_id in sorted(canonical_queries):
        pool = pools[query_id]
        selected = select_diverse_full_pose_hard_mode_indices(
            poses_w2c=pool.poses_w2c,
            scores=pool.scores,
            translation_errors_m=pool.translation_errors_m,
            rotation_errors_deg=pool.rotation_errors_deg,
            correct_10cm_5deg=pool.correct_10cm_5deg,
            source_row_indices=pool.source_row_indices,
            # Retain extra score-ranked candidates here because a subset can
            # exactly duplicate legacy coherent modes after projection.  The
            # final per-query count is enforced only after that canonical
            # projection-level de-duplication below.
            max_modes=min(
                int(score_pool_size),
                max(
                    int(additional_modes_per_query) * 2,
                    int(additional_modes_per_query) + 8,
                ),
            ),
            score_pool_size=int(score_pool_size),
            minimum_translation_error_m=float(minimum_translation_error_m),
            minimum_rotation_error_deg=float(minimum_rotation_error_deg),
            minimum_pose_translation_diversity_m=float(minimum_pose_translation_diversity_m),
            minimum_pose_rotation_diversity_deg=float(minimum_pose_rotation_diversity_deg),
        )
        if len(selected) < int(minimum_additional_modes_per_query):
            raise ValueError(
                f"train query {query_id!r} has only {len(selected)} diverse full-pose hard modes"
            )
        finite_order = np.flatnonzero(np.isfinite(pool.scores))
        finite_order = finite_order[
            np.lexsort((pool.source_row_indices[finite_order], -pool.scores[finite_order]))
        ]
        rank_by_local = np.full((len(pool.scores),), -1, dtype=np.int64)
        rank_by_local[finite_order] = np.arange(1, len(finite_order) + 1, dtype=np.int64)
        if np.any(rank_by_local[selected] <= 0):
            raise RuntimeError("selected full-pose hard mode has no finite frozen score rank")
        selected_candidate_indices_by_query[query_id] = selected

    bank_tracks, bank_xyz, _bank_metadata = _load_bank_xyz(bank_path)
    candidate_xyz = _candidate_xyz_for_layout(
        layout=layout,
        bank_tracks=bank_tracks,
        bank_xyz=bank_xyz,
    )
    cameras = _query_camera_parameters(
        colmap_model_dir=Path(colmap_model_dir), query_ids=tuple(sorted(canonical_queries))
    )
    source_parts: list[np.ndarray] = []
    correct_offset_parts: list[np.ndarray] = []
    correct_valid_parts: list[np.ndarray] = []
    wrong_offset_parts: list[np.ndarray] = []
    wrong_valid_parts: list[np.ndarray] = []
    pair_queries: list[str] = []
    pair_ids: list[int] = []
    pair_offsets = [0]
    legacy_mode_keys_by_query: dict[str, set[tuple[bytes, bytes]]] = {
        query_id: set() for query_id in canonical_queries
    }

    # Preserve all current coherent modes exactly; they are known difficult
    # near-miss poses and remain useful alongside broader full-pool modes.
    for pair_index, (pair_id, query_id) in enumerate(
        zip(base.pair_ids.tolist(), base.pair_query_ids.astype(str).tolist())
    ):
        start, stop = base.pair_point_offsets[pair_index : pair_index + 2].tolist()
        source_ids = np.asarray(base.pair_source_point_ids[int(start) : int(stop)], dtype=np.int64)
        source_parts.append(source_ids)
        correct_offset_parts.append(
            np.asarray(base.correct_projection_offsets_xy[int(start) : int(stop)], dtype=np.float32)
        )
        correct_valid_parts.append(
            np.asarray(base.correct_projection_valid[int(start) : int(stop)], dtype=bool)
        )
        wrong_offset_parts.append(
            np.asarray(base.coherent_wrong_projection_offsets_xy[int(start) : int(stop)], dtype=np.float32)
        )
        wrong_valid_parts.append(
            np.asarray(base.coherent_wrong_projection_valid[int(start) : int(stop)], dtype=bool)
        )
        canonical = canonical_queries[str(query_id)]
        pair_position = {int(source_id): position for position, source_id in enumerate(source_ids.tolist())}
        reorder = np.asarray(
            [pair_position[int(source_id)] for source_id in canonical.source_point_ids.tolist()], dtype=np.int64
        )
        legacy_offsets = np.asarray(
            base.coherent_wrong_projection_offsets_xy[int(start) : int(stop)], dtype=np.float32
        )[reorder]
        legacy_valid = np.asarray(
            base.coherent_wrong_projection_valid[int(start) : int(stop)], dtype=bool
        )[reorder]
        legacy_mode_keys_by_query[str(query_id)].add(
            (legacy_offsets.tobytes(), legacy_valid.tobytes())
        )
        pair_queries.append(str(query_id))
        pair_ids.append(int(pair_id))
        pair_offsets.append(pair_offsets[-1] + len(source_ids))
    next_pair_id = max(pair_ids) + 1 if pair_ids else 0
    added_local_projection_counts: list[float] = []
    skipped_duplicate_full_pool_modes = 0
    search_radius = float(base.metadata["spatial_search_radius_px"])
    for query_id in sorted(canonical_queries):
        canonical = canonical_queries[query_id]
        pool = pools[query_id]
        focal, principal_x, principal_y, radial_k, width, height = cameras[query_id]
        query_xy = np.asarray(layout.xy[canonical.layout_rows], dtype=np.float32)
        xyz = np.asarray(candidate_xyz[canonical.layout_rows], dtype=np.float64)
        candidate_valid = np.asarray(
            layout.candidate_track_ids[canonical.layout_rows] >= 0, dtype=bool
        )
        selected_effective: list[int] = []
        for local_index in selected_candidate_indices_by_query[query_id].tolist():
            pose = np.broadcast_to(
                np.asarray(pool.poses_w2c[int(local_index)], dtype=np.float64),
                (len(canonical.layout_rows), 4, 4),
            ).copy()
            wrong_offsets, wrong_valid = project_simple_radial_offsets(
                xyz=xyz,
                poses_w2c=pose,
                query_xy=query_xy,
                focal_length=float(focal),
                principal_x=float(principal_x),
                principal_y=float(principal_y),
                radial_k=float(radial_k),
                image_width=int(width),
                image_height=int(height),
            )
            wrong_valid &= candidate_valid
            mode_key = (wrong_offsets.tobytes(), wrong_valid.tobytes())
            if mode_key in legacy_mode_keys_by_query[query_id]:
                skipped_duplicate_full_pool_modes += 1
                continue
            source_parts.append(canonical.source_point_ids)
            correct_offset_parts.append(canonical.correct_offsets_xy)
            correct_valid_parts.append(canonical.correct_valid)
            wrong_offset_parts.append(wrong_offsets)
            wrong_valid_parts.append(wrong_valid)
            pair_queries.append(query_id)
            pair_ids.append(next_pair_id)
            next_pair_id += 1
            pair_offsets.append(pair_offsets[-1] + len(canonical.source_point_ids))
            legacy_mode_keys_by_query[query_id].add(mode_key)
            selected_effective.append(int(local_index))
            added_local_projection_counts.append(
                float(
                    np.count_nonzero(
                        wrong_valid
                        & (np.max(np.abs(wrong_offsets), axis=2) <= search_radius)
                    )
                )
            )
            if len(selected_effective) >= int(additional_modes_per_query):
                break
        if len(selected_effective) < int(minimum_additional_modes_per_query):
            raise ValueError(
                f"train query {query_id!r} retains only {len(selected_effective)} nonduplicate "
                "full-pose hard modes"
            )
        if len(selected_effective) != int(additional_modes_per_query):
            raise ValueError(
                f"train query {query_id!r} cannot fill the requested nonduplicate full-pose mode budget"
            )
        finite_order = np.flatnonzero(np.isfinite(pool.scores))
        finite_order = finite_order[
            np.lexsort((pool.source_row_indices[finite_order], -pool.scores[finite_order]))
        ]
        rank_by_local = np.full((len(pool.scores),), -1, dtype=np.int64)
        rank_by_local[finite_order] = np.arange(1, len(finite_order) + 1, dtype=np.int64)
        effective = np.asarray(selected_effective, dtype=np.int64)
        selected_counts.append(float(len(effective)))
        selected_ranks.extend(rank_by_local[effective].astype(np.float64).tolist())
        selected_translation.extend(pool.translation_errors_m[effective].astype(np.float64).tolist())
        selected_rotation.extend(pool.rotation_errors_deg[effective].astype(np.float64).tolist())

    metadata = dict(base.metadata)
    metadata.update(
        {
            # Preserve the base contract.  A geometry-projected v1 target must
            # not be re-labelled as a sparse registered-identity v2 target.
            "format": str(base.metadata["format"]),
            "training_only_target_artifact": True,
            "contains_ground_truth": True,
            "contains_validation_or_test_targets": False,
            "runtime_layout_is_target_free": True,
            "pose_or_ground_truth_must_not_be_loaded_by_runtime_scorer": True,
            "full_pose_hard_target_format": "candidate_pose_rgb_spatial_full_pool_hard_modes_v1",
            "base_training_targets_sha256": file_sha256_short(base_path),
            "full_pose_hard_base_spatial_supervision_mode": base_spatial_mode,
            "full_pose_hard_base_spatial_target_semantics": base_spatial_semantics,
            "full_pose_hard_mode_source": "frozen_global_hypotheses_then_train_only_target_join_v1",
            "full_pose_hard_mode_selection": {
                "score_source": str(source_lineage["score_source"]),
                "evaluation_label": str(evaluation_label),
                "additional_modes_per_query": int(additional_modes_per_query),
                "minimum_additional_modes_per_query": int(minimum_additional_modes_per_query),
                "score_pool_size": int(score_pool_size),
                "wrong_pose_filter": {
                    "exclude_correct_10cm_5deg": True,
                    "minimum_translation_error_m": float(minimum_translation_error_m),
                    "minimum_rotation_error_deg": float(minimum_rotation_error_deg),
                },
                "pose_diversity": {
                    "minimum_translation_m": float(minimum_pose_translation_diversity_m),
                    "minimum_rotation_deg": float(minimum_pose_rotation_diversity_deg),
                },
                "selection_order": "descending_target_free_score_then_source_row_before_target_projection_v1",
                "preserved_legacy_coherent_modes": int(base.pair_count),
                "projection_duplicate_policy": "skip_legacy_or_prior_added_mode_then_fill_same_score_order_v1",
            },
            "full_pose_hard_mode_source_lineage": source_lineage,
            "serialized_pose_or_residual": False,
            "inputs": {
                **dict(base.metadata.get("inputs", {})),
                "base_training_targets": str(base_path),
                "hypothesis_target_join": str(Path(hypothesis_target_join)),
                "hypothesis_artifacts": [str(Path(path)) for path in hypothesis_artifacts],
                "target_free_score_overlay": (
                    "" if target_free_score_overlay is None else str(target_free_score_overlay)
                ),
                "projected_landmark_bank": str(bank_path),
            },
        }
    )
    targets = CandidatePoseRGBSpatialTrainingTargets(
        source_point_ids=np.asarray(base.source_point_ids, dtype=np.int64),
        query_ids=np.asarray(base.query_ids).astype(str),
        spatial_target_offsets_xy=np.asarray(base.spatial_target_offsets_xy, dtype=np.float32),
        spatial_target_observed=np.asarray(base.spatial_target_observed, dtype=bool),
        spatial_target_dustbin=np.asarray(base.spatial_target_dustbin, dtype=bool),
        spatial_target_supervised=np.asarray(base.spatial_target_supervised, dtype=bool),
        pair_query_ids=np.asarray(pair_queries),
        pair_ids=np.asarray(pair_ids, dtype=np.int64),
        pair_point_offsets=np.asarray(pair_offsets, dtype=np.int64),
        pair_source_point_ids=np.concatenate(source_parts, axis=0),
        correct_projection_offsets_xy=np.concatenate(correct_offset_parts, axis=0),
        correct_projection_valid=np.concatenate(correct_valid_parts, axis=0),
        coherent_wrong_projection_offsets_xy=np.concatenate(wrong_offset_parts, axis=0),
        coherent_wrong_projection_valid=np.concatenate(wrong_valid_parts, axis=0),
        metadata=metadata,
    )
    save_candidate_pose_rgb_spatial_training_targets(targets, output_path)
    summary = {
        "stage": "build_candidate_pose_rgb_spatial_full_pose_hard_targets",
        "output": str(output_path),
        "output_sha256": file_sha256_short(output_path),
        "source_point_count": targets.source_point_count,
        "candidate_count": targets.candidate_count,
        "base_spatial_supervision_mode": base_spatial_mode,
        "base_spatial_target_semantics": base_spatial_semantics,
        "train_query_count": int(len(canonical_queries)),
        "legacy_pair_count": int(base.pair_count),
        "additional_full_pool_pair_count": int(len(pair_ids) - base.pair_count),
        "total_pair_count": targets.pair_count,
        "skipped_duplicate_full_pool_mode_count": int(skipped_duplicate_full_pool_modes),
        "additional_modes_per_query_quantiles": _quantiles(selected_counts),
        "selected_frozen_score_rank_quantiles": _quantiles(selected_ranks),
        "selected_translation_error_m_quantiles_TARGET_ONLY": _quantiles(selected_translation),
        "selected_rotation_error_deg_quantiles_TARGET_ONLY": _quantiles(selected_rotation),
        "selected_wrong_local_projection_count_quantiles": _quantiles(added_local_projection_counts),
        "selection": metadata["full_pose_hard_mode_selection"],
        "protocol": {
            "train_only_target_join": True,
            "target_free_runtime_layout_unchanged": True,
            "validation_or_test_target_rows_not_serialized": True,
            "pose_or_residual_not_serialized": True,
            "runtime_scorer_must_not_load_output": True,
            "render": False,
            "image_retrieval_or_submap_used": False,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-spatial-layout", required=True)
    parser.add_argument("--base-training-targets", required=True)
    parser.add_argument(
        "--hypothesis-artifact",
        required=True,
        action="append",
        help="Repeat in the exact order recorded by the target-side join.",
    )
    parser.add_argument("--hypothesis-target-join", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--evaluation-label", required=True)
    parser.add_argument(
        "--frozen-score-field",
        choices=_FROZEN_SCORE_FIELDS,
        default="preliminary_log_likelihood_means",
        help=(
            "Target-free frozen score used to order the complete pose pool. "
            "The preliminary field covers every generated hypothesis; verification "
            "is retained only as an explicit shortlist ablation."
        ),
    )
    parser.add_argument(
        "--target-free-score-overlay",
        default="",
        help=(
            "Optional current-model train full-pool score artifact. When supplied it "
            "replaces the historical frozen score field after strict target-free lineage checks."
        ),
    )
    parser.add_argument("--additional-modes-per-query", type=int, default=16)
    parser.add_argument("--minimum-additional-modes-per-query", type=int, default=8)
    parser.add_argument("--score-pool-size", type=int, default=512)
    parser.add_argument("--minimum-translation-error-m", type=float, default=0.25)
    parser.add_argument("--minimum-rotation-error-deg", type=float, default=2.0)
    parser.add_argument("--minimum-pose-translation-diversity-m", type=float, default=0.20)
    parser.add_argument("--minimum-pose-rotation-diversity-deg", type=float, default=0.75)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_candidate_pose_rgb_spatial_full_pose_hard_targets(
        rgb_spatial_layout=Path(args.rgb_spatial_layout),
        base_training_targets=Path(args.base_training_targets),
        hypothesis_artifacts=[Path(path) for path in args.hypothesis_artifact],
        hypothesis_target_join=Path(args.hypothesis_target_join),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        colmap_model_dir=Path(args.colmap_model_dir),
        evaluation_label=str(args.evaluation_label),
        frozen_score_field=str(args.frozen_score_field),
        target_free_score_overlay=(
            None
            if not str(args.target_free_score_overlay).strip()
            else Path(args.target_free_score_overlay)
        ),
        additional_modes_per_query=int(args.additional_modes_per_query),
        minimum_additional_modes_per_query=int(args.minimum_additional_modes_per_query),
        score_pool_size=int(args.score_pool_size),
        minimum_translation_error_m=float(args.minimum_translation_error_m),
        minimum_rotation_error_deg=float(args.minimum_rotation_error_deg),
        minimum_pose_translation_diversity_m=float(args.minimum_pose_translation_diversity_m),
        minimum_pose_rotation_diversity_deg=float(args.minimum_pose_rotation_diversity_deg),
        output=Path(args.output),
        summary_json=Path(args.summary_json),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
