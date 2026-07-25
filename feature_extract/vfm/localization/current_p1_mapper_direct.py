"""Direct P1-token supervision for a frozen projected-observation landmark bank.

The ordinary mapper trainer samples sparse SfM observations.  That is useful
for broad image-to-image supervision, but it is not the same population as the
frozen P1 points used by grouped pose inference.  In particular, a coherent
wrong-pose mode is defined on exact P1 token/candidate groups and must not be
attached to the nearest unrelated SfM observation merely because it is nearby.

This module keeps that boundary explicit:

* :class:`CurrentP1MapperRuntime` contains only target-free P1 inputs used by
  inference: query coordinates, frozen bank rows, candidate priors, and null
  mass.
* :class:`CurrentP1MapperDirectTargets` contains train-only registered-positive
  and coherent-wrong masks after target-free proposal generation is complete.
* The loss samples a mapper full feature map at the exact P1 coordinates, then
  compares its descriptor with the fixed projected-observation bank.

The frozen bank is a teacher descriptor space during this focused fine-tune.
Any resulting mapper checkpoint therefore requires a fresh projected-
observation bank rebuild before production retrieval or pose evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz


CURRENT_P1_MAPPER_DIRECT_PROPOSAL_FORMAT = (
    "current_p1_coherent_mapper_proposals_inference_only_v1"
)
CURRENT_P1_MAPPER_DIRECT_TARGET_FORMAT = "pose_conditioned_system_hard_modes_v2"
CURRENT_P1_MAPPER_DIRECT_LOSS_FORMAT = "current_p1_mapper_direct_loss_v1"

_TARGET_FIELD_TOKENS = ("target", "residual", "label", "ground_truth", "pose")


def _canonical_query_id(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("./")


def _metadata_scalar(payload: Mapping[str, np.ndarray], key: str) -> dict[str, object]:
    if key not in payload:
        raise ValueError(f"P1 artifact is missing {key!r}")
    try:
        value = json.loads(str(np.asarray(payload[key]).item()))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"P1 artifact has invalid {key!r}") from error
    if not isinstance(value, dict):
        raise ValueError(f"P1 artifact {key!r} must encode an object")
    return dict(value)


def _read_npz_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as data:
        arrays = {str(name): np.asarray(data[name]) for name in data.files if name != "metadata_json"}
        metadata = _metadata_scalar(data, "metadata_json")
    return arrays, metadata


@dataclass(frozen=True)
class CurrentP1MapperRuntime:
    """Target-free exact P1 query/candidate layout used by direct mapper scoring."""

    source_point_ids: np.ndarray
    query_ids: np.ndarray
    xy: np.ndarray
    candidate_bank_rows: np.ndarray
    candidate_prior_probabilities: np.ndarray
    null_probabilities: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        source_ids = np.asarray(self.source_point_ids, dtype=np.int64).reshape(-1)
        query_ids = np.asarray(self.query_ids).astype(str).reshape(-1)
        xy = np.asarray(self.xy, dtype=np.float32)
        bank_rows = np.asarray(self.candidate_bank_rows, dtype=np.int64)
        priors = np.asarray(self.candidate_prior_probabilities, dtype=np.float32)
        null = np.asarray(self.null_probabilities, dtype=np.float32).reshape(-1)
        count = int(len(source_ids))
        if (
            count == 0
            or len(np.unique(source_ids)) != count
            or query_ids.shape != (count,)
            or np.any(query_ids == "")
            or xy.shape != (count, 2)
            or bank_rows.ndim != 2
            or bank_rows.shape[0] != count
            or bank_rows.shape[1] < 2
            or priors.shape != bank_rows.shape
            or null.shape != (count,)
            or not np.isfinite(xy).all()
            or not np.isfinite(priors).all()
            or not np.isfinite(null).all()
            or np.any(priors < 0.0)
            or np.any(null <= 0.0)
        ):
            raise ValueError("current P1 mapper runtime arrays are invalid")
        valid = bank_rows >= 0
        if (
            np.any(valid & (priors <= 0.0))
            or np.any(~valid & (priors > 1e-7))
            or np.any(np.abs(priors.sum(axis=1) + null - 1.0) > 2e-4)
        ):
            raise ValueError("current P1 mapper runtime candidate/null mass is invalid")
        object.__setattr__(self, "source_point_ids", source_ids)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "candidate_bank_rows", bank_rows)
        object.__setattr__(self, "candidate_prior_probabilities", priors)
        object.__setattr__(self, "null_probabilities", null)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def row_count(self) -> int:
        return int(len(self.source_point_ids))

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_bank_rows.shape[1])

    def subset(self, rows: np.ndarray) -> "CurrentP1MapperRuntime":
        indices = np.asarray(rows, dtype=np.int64).reshape(-1)
        if (
            len(indices) == 0
            or np.any(indices < 0)
            or np.any(indices >= self.row_count)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("current P1 mapper runtime subset is invalid")
        return CurrentP1MapperRuntime(
            source_point_ids=self.source_point_ids[indices],
            query_ids=self.query_ids[indices],
            xy=self.xy[indices],
            candidate_bank_rows=self.candidate_bank_rows[indices],
            candidate_prior_probabilities=self.candidate_prior_probabilities[indices],
            null_probabilities=self.null_probabilities[indices],
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class CurrentP1MapperDirectTargets:
    """Train-only exact P1 positive and coherent-wrong candidate memberships."""

    positive_mask: np.ndarray
    hard_negative_mask: np.ndarray
    group_hard_mask: np.ndarray
    hard_mode_ids: np.ndarray
    hard_mode_candidate_mask: np.ndarray
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        positive = np.asarray(self.positive_mask, dtype=bool)
        hard = np.asarray(self.hard_negative_mask, dtype=bool)
        group_hard = np.asarray(self.group_hard_mask, dtype=bool).reshape(-1)
        mode_ids = np.asarray(self.hard_mode_ids, dtype=np.int64)
        mode_candidates = np.asarray(self.hard_mode_candidate_mask, dtype=bool)
        if (
            positive.ndim != 2
            or positive.shape[0] == 0
            or hard.shape != positive.shape
            or group_hard.shape != (positive.shape[0],)
            or mode_ids.ndim != 2
            or mode_ids.shape[0] != positive.shape[0]
            or mode_candidates.shape != (*mode_ids.shape, positive.shape[1])
            or np.any(positive & hard)
            or np.any((mode_ids < 0) & np.any(mode_candidates, axis=2))
            or np.any((mode_ids >= 0) & ~np.any(mode_candidates, axis=2))
            or np.any(group_hard & ~np.any(hard, axis=1))
        ):
            raise ValueError("current P1 mapper direct targets are invalid")
        object.__setattr__(self, "positive_mask", positive)
        object.__setattr__(self, "hard_negative_mask", hard)
        object.__setattr__(self, "group_hard_mask", group_hard)
        object.__setattr__(self, "hard_mode_ids", mode_ids)
        object.__setattr__(self, "hard_mode_candidate_mask", mode_candidates)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def row_count(self) -> int:
        return int(self.positive_mask.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.positive_mask.shape[1])

    @property
    def mode_slot_count(self) -> int:
        return int(self.hard_mode_ids.shape[1])

    def subset(self, rows: np.ndarray) -> "CurrentP1MapperDirectTargets":
        indices = np.asarray(rows, dtype=np.int64).reshape(-1)
        if (
            len(indices) == 0
            or np.any(indices < 0)
            or np.any(indices >= self.row_count)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("current P1 mapper direct target subset is invalid")
        return CurrentP1MapperDirectTargets(
            positive_mask=self.positive_mask[indices],
            hard_negative_mask=self.hard_negative_mask[indices],
            group_hard_mask=self.group_hard_mask[indices],
            hard_mode_ids=self.hard_mode_ids[indices],
            hard_mode_candidate_mask=self.hard_mode_candidate_mask[indices],
            metadata=self.metadata,
        )


def _validate_proposal_schema(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> None:
    names = {str(name).lower() for name in arrays}
    forbidden = sorted(
        name for name in names if any(token in name for token in _TARGET_FIELD_TOKENS)
    )
    if forbidden:
        raise ValueError(f"P1 proposal artifact exposes target fields: {forbidden}")
    expected = {
        "layout_row_indices",
        "source_point_ids",
        "query_ids",
        "xy",
        "candidate_track_ids",
        "candidate_bank_rows",
        "candidate_prior_probabilities",
        "null_probabilities",
    }
    if set(arrays) != expected:
        raise ValueError("P1 proposal artifact has an unexpected schema")
    if str(metadata.get("format", "")) != CURRENT_P1_MAPPER_DIRECT_PROPOSAL_FORMAT:
        raise ValueError("P1 proposal artifact has an unsupported format")
    if bool(metadata.get("contains_target_fields", True)):
        raise ValueError("P1 proposal artifact is not marked target-free")
    if bool(metadata.get("pose_or_ground_truth_used_for_generation", True)):
        raise ValueError("P1 proposal generation used target-side pose or ground truth")


def _aligned_target_columns(
    *,
    values: np.ndarray,
    selected_columns: np.ndarray,
    candidate_count: int,
    name: str,
) -> np.ndarray:
    source = np.asarray(values, dtype=bool)
    columns = np.asarray(selected_columns, dtype=np.int64)
    if source.shape != columns.shape:
        raise ValueError(f"P1 target {name} does not align with selected columns")
    output = np.zeros((source.shape[0], int(candidate_count)), dtype=bool)
    for row in range(source.shape[0]):
        valid = columns[row] >= 0
        selected = columns[row, valid]
        if (
            np.any(selected >= int(candidate_count))
            or len(np.unique(selected)) != len(selected)
        ):
            raise ValueError(f"P1 target {name} has invalid selected columns")
        output[row, selected] = source[row, valid]
    return output


def _aligned_mode_target_columns(
    *,
    values: np.ndarray,
    selected_columns: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    source = np.asarray(values, dtype=bool)
    columns = np.asarray(selected_columns, dtype=np.int64)
    if (
        source.ndim != 3
        or source.shape[0] != columns.shape[0]
        or source.shape[2] != columns.shape[1]
    ):
        raise ValueError("P1 hard-mode candidate mask does not align with selected columns")
    output = np.zeros((source.shape[0], source.shape[1], int(candidate_count)), dtype=bool)
    for row in range(source.shape[0]):
        valid = columns[row] >= 0
        selected = columns[row, valid]
        if (
            np.any(selected >= int(candidate_count))
            or len(np.unique(selected)) != len(selected)
        ):
            raise ValueError("P1 hard-mode candidate mask has invalid selected columns")
        output[row][:, selected] = source[row][:, valid]
    return output


def load_current_p1_mapper_direct_supervision(
    *,
    proposals_path: Path,
    targets_path: Path,
    projected_landmark_bank_path: Path,
) -> tuple[CurrentP1MapperRuntime, CurrentP1MapperDirectTargets, np.ndarray, dict[str, object]]:
    """Load exact-P1 train supervision with strict proposal/bank lineage checks.

    The returned descriptor matrix is fixed target data for the trainer.  It is
    intentionally separate from ``CurrentP1MapperRuntime`` so a runtime scorer
    cannot receive positive masks or coherent mode memberships by accident.
    """

    proposals_file = Path(proposals_path)
    targets_file = Path(targets_path)
    bank_file = Path(projected_landmark_bank_path)
    proposal_arrays, proposal_metadata = _read_npz_arrays(proposals_file)
    _validate_proposal_schema(proposal_arrays, proposal_metadata)
    target_arrays, target_metadata = _read_npz_arrays(targets_file)
    if str(target_metadata.get("format", "")) != CURRENT_P1_MAPPER_DIRECT_TARGET_FORMAT:
        raise ValueError("P1 direct target artifact has an unsupported format")
    if not bool(target_metadata.get("training_only_target_artifact", False)):
        raise ValueError("P1 direct target artifact is not marked training-only")
    if bool(target_metadata.get("pose_or_ground_truth_used_for_hypothesis_generation", True)):
        raise ValueError("P1 direct target hypotheses used target-side pose or ground truth")
    if set(map(str, target_metadata.get("split_names", []))) != {"train"}:
        raise ValueError("P1 direct target artifact must contain only train queries")
    target_inputs = dict(target_metadata.get("inputs", {}))
    if str(target_inputs.get("proposals_sha256", "")) != file_sha256_short(proposals_file):
        raise ValueError("P1 direct target/proposal lineage mismatch")
    if str(target_inputs.get("projected_landmark_bank_sha256", "")) != file_sha256_short(bank_file):
        raise ValueError("P1 direct target/projected-bank lineage mismatch")

    required_target_fields = {
        "selected_rows",
        "selected_columns",
        "query_ids",
        "positive_mask_TARGET_ONLY",
        "hard_negative_mask_TARGET_ONLY",
        "group_hard_mask_TARGET_ONLY",
        "hard_mode_ids_TARGET_ONLY",
        "hard_mode_candidate_mask_TARGET_ONLY",
    }
    missing_target_fields = sorted(required_target_fields.difference(target_arrays))
    if missing_target_fields:
        raise ValueError(f"P1 direct target artifact is missing fields: {missing_target_fields}")

    source_ids = np.asarray(proposal_arrays["source_point_ids"], dtype=np.int64).reshape(-1)
    query_ids = np.asarray(
        [_canonical_query_id(value) for value in proposal_arrays["query_ids"].tolist()]
    )
    xy = np.asarray(proposal_arrays["xy"], dtype=np.float32)
    candidate_tracks = np.asarray(proposal_arrays["candidate_track_ids"], dtype=np.int64)
    candidate_rows = np.asarray(proposal_arrays["candidate_bank_rows"], dtype=np.int64)
    priors = np.asarray(proposal_arrays["candidate_prior_probabilities"], dtype=np.float32)
    null = np.asarray(proposal_arrays["null_probabilities"], dtype=np.float32)
    runtime = CurrentP1MapperRuntime(
        source_point_ids=source_ids,
        query_ids=query_ids,
        xy=xy,
        candidate_bank_rows=candidate_rows,
        candidate_prior_probabilities=priors,
        null_probabilities=null,
        metadata=proposal_metadata,
    )

    selected_rows = np.asarray(target_arrays["selected_rows"], dtype=np.int64).reshape(-1)
    selected_columns = np.asarray(target_arrays["selected_columns"], dtype=np.int64)
    target_query_ids = np.asarray(
        [_canonical_query_id(value) for value in target_arrays["query_ids"].tolist()]
    )
    if (
        selected_rows.shape != (runtime.row_count,)
        or len(np.unique(selected_rows)) != runtime.row_count
        or np.any(selected_rows < 0)
        or np.any(selected_rows >= runtime.row_count)
        or selected_columns.shape[0] != runtime.row_count
        or target_query_ids.shape != (runtime.row_count,)
    ):
        raise ValueError("P1 direct target rows do not cover the proposal layout exactly")
    inverse = np.empty((runtime.row_count,), dtype=np.int64)
    inverse[selected_rows] = np.arange(runtime.row_count, dtype=np.int64)
    order = inverse
    if not np.array_equal(target_query_ids[order], runtime.query_ids):
        raise ValueError("P1 direct target query IDs do not align with proposals")
    selected_columns = selected_columns[order]
    positive = _aligned_target_columns(
        values=np.asarray(target_arrays["positive_mask_TARGET_ONLY"], dtype=bool)[order],
        selected_columns=selected_columns,
        candidate_count=runtime.candidate_count,
        name="positive_mask",
    )
    hard = _aligned_target_columns(
        values=np.asarray(target_arrays["hard_negative_mask_TARGET_ONLY"], dtype=bool)[order],
        selected_columns=selected_columns,
        candidate_count=runtime.candidate_count,
        name="hard_negative_mask",
    )
    mode_ids = np.asarray(target_arrays["hard_mode_ids_TARGET_ONLY"], dtype=np.int64)[order]
    mode_candidates = _aligned_mode_target_columns(
        values=np.asarray(target_arrays["hard_mode_candidate_mask_TARGET_ONLY"], dtype=bool)[order],
        selected_columns=selected_columns,
        candidate_count=runtime.candidate_count,
    )
    targets = CurrentP1MapperDirectTargets(
        positive_mask=positive,
        hard_negative_mask=hard,
        group_hard_mask=np.asarray(target_arrays["group_hard_mask_TARGET_ONLY"], dtype=bool)[order],
        hard_mode_ids=mode_ids,
        hard_mode_candidate_mask=mode_candidates,
        metadata=target_metadata,
    )
    if not np.any(targets.hard_mode_ids >= 0):
        raise ValueError("P1 direct target artifact contains no coherent modes")
    if targets.row_count != runtime.row_count or targets.candidate_count != runtime.candidate_count:
        raise RuntimeError("P1 direct runtime/target cardinalities diverged")

    bank, bank_metadata = load_landmark_index_npz(bank_file)
    descriptor_space_id = str(bank_metadata.get("descriptor_space_id", ""))
    expected_space_id = str(target_inputs.get("descriptor_space_id", ""))
    if not descriptor_space_id or descriptor_space_id != expected_space_id:
        raise ValueError("P1 direct target/projected-bank descriptor space mismatch")
    expected_projection = str(target_inputs.get("projection_space_id", ""))
    actual_projection = str(bank_metadata.get("projection_space_id", expected_projection))
    if expected_projection and actual_projection != expected_projection:
        raise ValueError("P1 direct target/projected-bank projection space mismatch")
    expected_checkpoint_hash = str(target_inputs.get("matcha_joint_checkpoint_sha256", ""))
    bank_checkpoint_hash = str(bank_metadata.get("matcha_joint_checkpoint_sha256", ""))
    if expected_checkpoint_hash and bank_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("P1 direct target/projected-bank mapper checkpoint mismatch")
    valid_rows = runtime.candidate_bank_rows >= 0
    if np.any(runtime.candidate_bank_rows[valid_rows] >= len(bank.track_ids)):
        raise ValueError("P1 direct proposal references a missing bank row")
    if np.any(bank.track_ids[runtime.candidate_bank_rows[valid_rows]] != candidate_tracks[valid_rows]):
        raise ValueError("P1 direct proposal bank rows resolve to different tracks")
    descriptors = np.asarray(bank.features, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[0] != len(bank.track_ids) or descriptors.shape[1] <= 0:
        raise ValueError("P1 direct projected landmark bank descriptors are invalid")
    norms = np.linalg.norm(descriptors, axis=1)
    if not np.isfinite(descriptors).all() or np.any(norms <= 1e-8):
        raise ValueError("P1 direct projected landmark bank contains invalid descriptors")
    summary = {
        "format": "current_p1_mapper_direct_supervision_v1",
        "proposal_path": str(proposals_file),
        "proposal_sha256": file_sha256_short(proposals_file),
        "target_path": str(targets_file),
        "target_sha256": file_sha256_short(targets_file),
        "projected_landmark_bank_path": str(bank_file),
        "projected_landmark_bank_sha256": file_sha256_short(bank_file),
        "descriptor_space_id": descriptor_space_id,
        "descriptor_dimension": int(descriptors.shape[1]),
        "bank_mapper_checkpoint_sha256": bank_checkpoint_hash,
        "row_count": int(runtime.row_count),
        "candidate_count": int(runtime.candidate_count),
        "positive_row_count": int(np.any(targets.positive_mask, axis=1).sum()),
        "positive_edge_count": int(targets.positive_mask.sum()),
        "hard_group_count": int(targets.group_hard_mask.sum()),
        "hard_edge_count": int(targets.hard_negative_mask.sum()),
        "coherent_mode_count": int(np.sum(np.unique(targets.hard_mode_ids[targets.hard_mode_ids >= 0]) >= 0)),
        "bank_is_frozen_teacher": True,
        "post_training_requirement": "rebuild_projected_observation_bank_before_runtime_eval",
    }
    return runtime, targets, descriptors, summary


def group_current_p1_runtime_rows_by_query(
    runtime: CurrentP1MapperRuntime,
    targets: CurrentP1MapperDirectTargets | None = None,
) -> dict[str, np.ndarray]:
    """Group exact P1 rows by query without using target identities for grouping."""

    if targets is not None and targets.row_count != runtime.row_count:
        raise ValueError("P1 runtime/target row counts do not match")
    groups = {
        str(query_id): np.flatnonzero(runtime.query_ids == str(query_id)).astype(np.int64)
        for query_id in sorted(set(runtime.query_ids.tolist()))
    }
    if not groups or sum(len(rows) for rows in groups.values()) != runtime.row_count:
        raise RuntimeError("P1 runtime query grouping is incomplete")
    for query_id, rows in groups.items():
        if len(rows) == 0 or len(np.unique(runtime.source_point_ids[rows])) != len(rows):
            raise ValueError(f"P1 runtime query group is invalid: {query_id!r}")
    return groups


def sample_mapper_descriptors_at_p1_xy(
    descriptor_maps: torch.Tensor,
    xy: torch.Tensor,
    image_sizes: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly sample full-map descriptors in the feature image coordinate frame.

    This uses the same endpoint-coordinate / ``align_corners=True`` convention
    as projected-observation bank construction.  It rejects out-of-image P1
    anchors rather than quietly producing border evidence in a different frame.
    """

    maps = torch.as_tensor(descriptor_maps)
    points = torch.as_tensor(xy, dtype=maps.dtype, device=maps.device)
    sizes = torch.as_tensor(image_sizes, dtype=maps.dtype, device=maps.device)
    if (
        maps.ndim != 4
        or points.ndim != 3
        or points.shape[0] != maps.shape[0]
        or points.shape[2] != 2
        or sizes.shape != (maps.shape[0], 2)
        or maps.shape[0] == 0
        or maps.shape[1] == 0
        or not torch.isfinite(maps).all()
        or not torch.isfinite(points).all()
        or not torch.isfinite(sizes).all()
        or torch.any(sizes <= 1.0)
    ):
        raise ValueError("P1 mapper descriptor sampling inputs are invalid")
    max_xy = sizes[:, None, :] - 1.0
    if torch.any(points < 0.0) or torch.any(points > max_xy):
        raise ValueError("P1 mapper coordinates fall outside the feature image frame")
    normalized = 2.0 * points / max_xy - 1.0
    grid = normalized.reshape(maps.shape[0], points.shape[1], 1, 2)
    sampled = F.grid_sample(
        maps,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[:, :, :, 0].transpose(1, 2)
    return F.normalize(sampled.float(), p=2, dim=-1)


def dense_descriptor_anchor_distillation_loss(
    *,
    student_descriptor_maps: torch.Tensor,
    teacher_descriptor_maps: torch.Tensor,
    p1_xy: torch.Tensor,
    image_sizes: torch.Tensor,
    exclusion_radius_tokens: int,
) -> tuple[torch.Tensor, Mapping[str, float]]:
    """Keep non-P1 descriptor maps anchored to the frozen mapper during fine-tuning.

    Direct P1 supervision covers only a small set of exact query points.  The
    mapper itself is pointwise, however, so unconstrained updates can change
    descriptors for every other token and silently degrade global landmark
    retrieval.  This training-only loss preserves the frozen mapper's output
    over every token outside a small neighborhood of the supervised P1 cells.

    ``teacher_descriptor_maps`` is always detached.  It is not available to a
    runtime scorer and it never changes the target-free P1 proposal contract.
    """

    student = torch.as_tensor(student_descriptor_maps)
    teacher = torch.as_tensor(teacher_descriptor_maps, dtype=student.dtype, device=student.device)
    points = torch.as_tensor(p1_xy, dtype=student.dtype, device=student.device)
    sizes = torch.as_tensor(image_sizes, dtype=student.dtype, device=student.device)
    radius = int(exclusion_radius_tokens)
    if (
        student.ndim != 4
        or teacher.shape != student.shape
        or student.shape[0] == 0
        or student.shape[1] == 0
        or points.ndim != 3
        or points.shape[0] != student.shape[0]
        or points.shape[2] != 2
        or sizes.shape != (student.shape[0], 2)
        or radius < 0
        or not torch.isfinite(student).all()
        or not torch.isfinite(teacher).all()
        or not torch.isfinite(points).all()
        or not torch.isfinite(sizes).all()
        or torch.any(sizes <= 1.0)
    ):
        raise ValueError("P1 mapper anchor-distillation inputs are invalid")
    max_xy = sizes[:, None, :] - 1.0
    if torch.any(points < 0.0) or torch.any(points > max_xy):
        raise ValueError("P1 mapper anchor coordinates fall outside the feature image frame")

    batch, _channels, height, width = student.shape
    token_x = torch.round(points[..., 0] * float(width - 1) / max_xy[..., 0]).to(torch.long)
    token_y = torch.round(points[..., 1] * float(height - 1) / max_xy[..., 1]).to(torch.long)
    token_x = token_x.clamp(0, int(width - 1))
    token_y = token_y.clamp(0, int(height - 1))
    grid_y = torch.arange(height, device=student.device).view(1, 1, height, 1)
    grid_x = torch.arange(width, device=student.device).view(1, 1, 1, width)
    excluded = (
        (grid_x - token_x.view(batch, -1, 1, 1)).abs() <= radius
    ) & ((grid_y - token_y.view(batch, -1, 1, 1)).abs() <= radius)
    anchor_mask = ~excluded.any(dim=1)
    anchor_count = int(anchor_mask.sum().detach().cpu().item())
    if anchor_count <= 0:
        raise ValueError("P1 mapper anchor-distillation mask contains no non-P1 token")

    student_normalized = F.normalize(student.float(), p=2, dim=1)
    teacher_normalized = F.normalize(teacher.detach().float(), p=2, dim=1)
    cosine = (student_normalized * teacher_normalized).sum(dim=1).clamp(-1.0, 1.0)
    loss = (1.0 - cosine)[anchor_mask].mean()
    metrics = {
        "anchor_distillation_loss": float(loss.detach().cpu().item()),
        "anchor_descriptor_cosine_mean": float(cosine[anchor_mask].mean().detach().cpu().item()),
        "anchor_token_count": float(anchor_count),
        "anchor_excluded_token_count": float((~anchor_mask).sum().detach().cpu().item()),
    }
    return loss, metrics


def fixed_p1_candidate_log_posteriors(
    *,
    query_descriptors: torch.Tensor,
    candidate_descriptors: torch.Tensor,
    candidate_prior_probabilities: torch.Tensor,
    null_probabilities: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate log posterior and cosine scores with an explicit fixed null.

    The null component is deliberately not a learned dustbin here.  It is the
    frozen P1 mixture mass, so a direct mapper loss cannot make an out-of-bank
    candidate look valid by learning to manipulate the null logit.
    """

    query = torch.as_tensor(query_descriptors)
    candidates = torch.as_tensor(candidate_descriptors, dtype=query.dtype, device=query.device)
    priors = torch.as_tensor(candidate_prior_probabilities, dtype=query.dtype, device=query.device)
    null = torch.as_tensor(null_probabilities, dtype=query.dtype, device=query.device).reshape(-1)
    value = float(temperature)
    if (
        query.ndim != 2
        or candidates.ndim != 3
        or candidates.shape[0] != query.shape[0]
        or candidates.shape[2] != query.shape[1]
        or priors.shape != candidates.shape[:2]
        or null.shape != (query.shape[0],)
        or query.shape[0] == 0
        or not np.isfinite(value)
        or value <= 0.0
        or torch.any(priors < 0.0)
        or torch.any(null <= 0.0)
        or torch.any(torch.abs(priors.sum(dim=1) + null - 1.0) > 2e-4)
    ):
        raise ValueError("fixed P1 candidate posterior inputs are invalid")
    query = F.normalize(query.float(), p=2, dim=1)
    candidates = F.normalize(candidates.float(), p=2, dim=2)
    similarities = torch.einsum("pc,plc->pl", query, candidates)
    negative_infinity = torch.full_like(priors, float("-inf"))
    candidate_log_mass = torch.where(
        priors > 0.0,
        torch.log(priors.clamp_min(torch.finfo(priors.dtype).tiny)) + similarities / value,
        negative_infinity,
    )
    null_log_mass = torch.log(null.clamp_min(torch.finfo(null.dtype).tiny))
    normalizer = torch.logsumexp(
        torch.cat([candidate_log_mass, null_log_mass.unsqueeze(1)], dim=1), dim=1
    )
    log_posterior = candidate_log_mass - normalizer.unsqueeze(1)
    return log_posterior, similarities


@dataclass(frozen=True)
class CurrentP1MapperDirectLoss:
    """Differentiable direct-P1 loss and detached audit metrics."""

    total_loss: torch.Tensor
    positive_nll_loss: torch.Tensor
    coherent_margin_loss: torch.Tensor
    metrics: Mapping[str, float]


def current_p1_mapper_direct_loss(
    *,
    query_descriptors: torch.Tensor,
    candidate_descriptors: torch.Tensor,
    runtime: CurrentP1MapperRuntime,
    targets: CurrentP1MapperDirectTargets,
    temperature: float,
    coherent_margin: float,
    coherent_margin_weight: float,
    coherent_min_mode_rows: int,
) -> CurrentP1MapperDirectLoss:
    """Score exact P1 candidates and contrast correct against coherent-wrong modes."""

    if (
        runtime.row_count != targets.row_count
        or runtime.candidate_count != targets.candidate_count
        or int(coherent_min_mode_rows) < 2
        or not np.isfinite(float(coherent_margin))
        or float(coherent_margin) < 0.0
        or not np.isfinite(float(coherent_margin_weight))
        or float(coherent_margin_weight) < 0.0
    ):
        raise ValueError("current P1 mapper direct loss configuration is invalid")
    log_posteriors, similarities = fixed_p1_candidate_log_posteriors(
        query_descriptors=query_descriptors,
        candidate_descriptors=candidate_descriptors,
        candidate_prior_probabilities=torch.as_tensor(runtime.candidate_prior_probabilities),
        null_probabilities=torch.as_tensor(runtime.null_probabilities),
        temperature=float(temperature),
    )
    device = log_posteriors.device
    positive = torch.as_tensor(targets.positive_mask, dtype=torch.bool, device=device)
    hard = torch.as_tensor(targets.hard_negative_mask, dtype=torch.bool, device=device)
    modes = torch.as_tensor(targets.hard_mode_ids, dtype=torch.long, device=device)
    mode_candidates = torch.as_tensor(
        targets.hard_mode_candidate_mask, dtype=torch.bool, device=device
    )
    valid_positive = torch.any(positive, dim=1)
    if not bool(valid_positive.any()):
        raise ValueError("current P1 mapper direct batch has no registered positive candidate")
    negative_infinity = torch.full_like(log_posteriors, float("-inf"))
    positive_log_probability = torch.logsumexp(
        torch.where(positive, log_posteriors, negative_infinity), dim=1
    )
    positive_nll = -positive_log_probability[valid_positive].mean()

    mode_positive_terms: dict[int, list[torch.Tensor]] = {}
    mode_wrong_terms: dict[int, list[torch.Tensor]] = {}
    for row in range(runtime.row_count):
        if not bool(valid_positive[row]):
            continue
        correct_value = positive_log_probability[row]
        for slot in range(targets.mode_slot_count):
            mode_id = int(modes[row, slot].item())
            if mode_id < 0:
                continue
            wrong_mask = mode_candidates[row, slot] & hard[row]
            if not bool(wrong_mask.any()):
                continue
            wrong_value = torch.logsumexp(
                torch.where(wrong_mask, log_posteriors[row], negative_infinity[row]), dim=0
            )
            mode_positive_terms.setdefault(mode_id, []).append(correct_value)
            mode_wrong_terms.setdefault(mode_id, []).append(wrong_value)
    mode_losses: list[torch.Tensor] = []
    mode_gaps: list[torch.Tensor] = []
    active_mode_row_count = 0
    for mode_id in sorted(mode_positive_terms):
        positive_terms = mode_positive_terms[mode_id]
        wrong_terms = mode_wrong_terms[mode_id]
        if len(positive_terms) < int(coherent_min_mode_rows):
            continue
        positive_mean = torch.stack(positive_terms).mean()
        wrong_mean = torch.stack(wrong_terms).mean()
        gap = positive_mean - wrong_mean
        mode_gaps.append(gap)
        mode_losses.append(F.relu(float(coherent_margin) - gap))
        active_mode_row_count += len(positive_terms)
    if mode_losses:
        coherent_loss = torch.stack(mode_losses).mean()
        coherent_gap = torch.stack(mode_gaps).mean()
        coherent_violation_fraction = torch.stack(
            [value > 0.0 for value in mode_losses]
        ).float().mean()
    else:
        coherent_loss = positive_nll.new_zeros(())
        coherent_gap = positive_nll.new_zeros(())
        coherent_violation_fraction = positive_nll.new_zeros(())
    total = positive_nll + float(coherent_margin_weight) * coherent_loss

    candidate_top1 = torch.argmax(log_posteriors, dim=1)
    top1_correct = positive.gather(1, candidate_top1.unsqueeze(1)).squeeze(1)
    metrics = {
        "format": 1.0,
        "direct_positive_row_count": float(valid_positive.sum().detach().cpu().item()),
        "direct_positive_edge_count": float(positive.sum().detach().cpu().item()),
        "direct_positive_nll_loss": float(positive_nll.detach().cpu().item()),
        "direct_candidate_top1_positive_rate": float(
            top1_correct[valid_positive].float().mean().detach().cpu().item()
        ),
        "direct_positive_similarity_mean": float(
            similarities[positive].mean().detach().cpu().item()
        ),
        "direct_coherent_mode_count": float(len(mode_losses)),
        "direct_coherent_mode_row_count": float(active_mode_row_count),
        "direct_coherent_margin_loss": float(coherent_loss.detach().cpu().item()),
        "direct_coherent_mean_log_posterior_gap": float(coherent_gap.detach().cpu().item()),
        "direct_coherent_violation_fraction": float(
            coherent_violation_fraction.detach().cpu().item()
        ),
        "direct_total_loss": float(total.detach().cpu().item()),
    }
    return CurrentP1MapperDirectLoss(
        total_loss=total,
        positive_nll_loss=positive_nll,
        coherent_margin_loss=coherent_loss,
        metrics=metrics,
    )
