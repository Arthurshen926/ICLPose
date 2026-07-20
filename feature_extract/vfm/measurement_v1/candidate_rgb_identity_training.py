"""Train independent RGB identity and spatial evidence on frozen system top-L rows."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.correspondence_confidence import confidence_metrics
from feature_extract.vfm.measurement_v1.candidate_rgb_identity_verifier import (
    GT_POSE_SPATIAL_DENSITY_SEMANTICS,
    IndependentRGBCandidateVerifier,
    MEASUREMENT_MODE_SUCCESS_SEMANTICS,
    POSE_VIEW_MIXTURE_SEMANTICS,
    RGBCandidateIdentityPrediction,
    aggregate_pose_view_spatial_log_probabilities,
    fuse_candidate_log_likelihood_ratios,
    measurement_mode_residual_and_success,
    normalize_pose_view_mixture_logits,
    normalized_spatial_log_probabilities_with_dustbin,
)
from feature_extract.vfm.measurement_v1.rgb_patch_training import (
    TensorImageLRUCache,
    _read_csv,
    _stack_patch_batch,
)
from feature_extract.vfm.measurement_v1.rgb_data_contract import (
    coordinate_space_from_evidence,
    image_root_manifest,
    require_compatible_contracts,
    validate_sampling_dimensions,
)


def _bool_text(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _float_text(value: object, *, default: float = float("nan")) -> float:
    text = str(value).strip()
    return float(default) if not text else float(text)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    target = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.count_nonzero(target))
    if positives == 0:
        return float("nan")
    order = np.argsort(-values, kind="stable")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / positives)


IDENTITY_TARGET_APPEARANCE = "appearance_observation"
IDENTITY_TARGET_GEOMETRIC = "geometric_projection"
IDENTITY_TARGET_MODES = (
    IDENTITY_TARGET_APPEARANCE,
    IDENTITY_TARGET_GEOMETRIC,
)


def _identity_training_masks(
    *,
    valid: np.ndarray,
    actual_query_observation: np.ndarray,
    actual_center_residuals: np.ndarray,
    target_projection_residuals: np.ndarray,
    identity_target: str,
    positive_threshold_px: float,
    negative_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the groupwise target and hard-negative mask for RGB evidence.

    ``appearance_observation`` is the original exact-SfM-observation objective.
    ``geometric_projection`` is deliberately separate: it learns whether a
    frozen candidate can serve the downstream PnP geometry at the GT pose,
    including near-correct tracks that were not the exact observed SfM track.
    """

    target_mode = str(identity_target)
    if target_mode not in IDENTITY_TARGET_MODES:
        raise ValueError(
            "identity_target must be one of " f"{list(IDENTITY_TARGET_MODES)}"
        )
    candidate_valid = np.asarray(valid, dtype=bool)
    observed = np.asarray(actual_query_observation, dtype=bool)
    center_residual = np.asarray(actual_center_residuals, dtype=np.float32)
    projection_residual = np.asarray(target_projection_residuals, dtype=np.float32)
    if not (
        candidate_valid.shape == observed.shape
        == center_residual.shape
        == projection_residual.shape
    ):
        raise ValueError("identity training target arrays are incompatible")
    if float(positive_threshold_px) <= 0.0 or float(negative_threshold_px) <= float(
        positive_threshold_px
    ):
        raise ValueError("identity target thresholds are invalid")
    if target_mode == IDENTITY_TARGET_APPEARANCE:
        labels = (
            candidate_valid
            & observed
            & np.isfinite(center_residual)
            & (center_residual <= float(positive_threshold_px))
        )
    else:
        labels = (
            candidate_valid
            & np.isfinite(projection_residual)
            & (projection_residual <= float(positive_threshold_px))
        )
    supervision_valid = candidate_valid & (
        labels
        | (
            np.isfinite(projection_residual)
            & (projection_residual >= float(negative_threshold_px))
        )
    )
    return labels, supervision_valid


def _identity_target_semantics(identity_target: str) -> str:
    if str(identity_target) == IDENTITY_TARGET_APPEARANCE:
        return "actual_query_sfm_observation_center_residual_le_threshold"
    if str(identity_target) == IDENTITY_TARGET_GEOMETRIC:
        return "candidate_target_gt_projection_residual_le_threshold"
    raise ValueError(f"unsupported identity target: {identity_target}")


def _resolve_identity_target(
    requested: str | None, *, initialization_training: Mapping[str, Any]
) -> str:
    """Use checkpoint lineage for evaluation unless the caller overrides it."""

    target = (
        initialization_training.get("identity_target", IDENTITY_TARGET_APPEARANCE)
        if requested is None
        else requested
    )
    resolved = str(target)
    if resolved not in IDENTITY_TARGET_MODES:
        raise ValueError(
            "identity_target must be one of " f"{list(IDENTITY_TARGET_MODES)}"
        )
    return resolved


def _initialization_provenance(
    path: Path, payload: Mapping[str, Any]
) -> dict[str, object]:
    """Preserve the source model's actual training lineage in eval-only runs."""

    training = dict(payload.get("training", {}))
    return {
        "initialization_checkpoint": str(Path(path)),
        "initialization_checkpoint_sha256": file_sha256_short(Path(path)),
        "initialization_checkpoint_format": str(payload.get("format", "")),
        "initialization_training": training,
    }


def _resolve_spatial_support_config(
    initial_config: Mapping[str, Any],
    *,
    search_radius_px_override: float | None = None,
    context_radius_px_override: float | None = None,
    step_px_override: float | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Resolve an explicit spatial-support ablation without changing defaults.

    A different crop/template support changes the meaning of a learned spatial
    density.  Callers therefore receive both the resolved values and a compact
    manifest of every override, rather than silently inheriting a checkpoint's
    calibrated support.
    """

    required = {"search_radius_px", "context_radius_px", "step_px"}
    missing = required.difference(initial_config)
    if missing:
        raise ValueError(
            "measurement checkpoint lacks spatial support config: "
            f"{sorted(missing)}"
        )
    resolved = {
        name: float(initial_config[name])
        for name in ("search_radius_px", "context_radius_px", "step_px")
    }
    overrides: dict[str, dict[str, float]] = {}
    requested = {
        "search_radius_px": search_radius_px_override,
        "context_radius_px": context_radius_px_override,
        "step_px": step_px_override,
    }
    for name, value in requested.items():
        if value is None:
            continue
        resolved_value = float(value)
        if not math.isfinite(resolved_value):
            raise ValueError(f"{name}_override must be finite")
        if name == "context_radius_px":
            valid = resolved_value >= 0.0
        else:
            valid = resolved_value > 0.0
        if not valid:
            comparison = "non-negative" if name == "context_radius_px" else "positive"
            raise ValueError(f"{name}_override must be {comparison}")
        if not math.isclose(
            resolved_value,
            resolved[name],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            overrides[name] = {
                "checkpoint_value": float(resolved[name]),
                "resolved_value": resolved_value,
            }
            resolved[name] = resolved_value
    return resolved, overrides


def _resolve_identity_context_config(
    initial_config: Mapping[str, Any],
    *,
    identity_context_radius_px: float | None = None,
    identity_context_step_px: float | None = None,
) -> tuple[dict[str, float | bool | None], dict[str, dict[str, float | None]]]:
    """Resolve a separately declared broad identity-context branch.

    Unlike the local spatial support, this branch provides no offset or
    dustbin density. It contributes only per-view candidate identity evidence.
    """

    checkpoint_enabled = bool(initial_config.get("identity_context_enabled", False))
    checkpoint_radius = initial_config.get("identity_context_radius_px")
    checkpoint_step = initial_config.get("identity_context_step_px")
    if checkpoint_enabled:
        if checkpoint_radius is None or checkpoint_step is None:
            raise ValueError("identity-context checkpoint lacks radius or step")
        checkpoint_radius = float(checkpoint_radius)
        checkpoint_step = float(checkpoint_step)
    else:
        checkpoint_radius = None
        checkpoint_step = None

    if (identity_context_radius_px is None) != (identity_context_step_px is None):
        raise ValueError(
            "identity_context_radius_px and identity_context_step_px must be provided together"
        )
    if identity_context_radius_px is None:
        return {
            "enabled": checkpoint_enabled,
            "radius_px": checkpoint_radius,
            "step_px": checkpoint_step,
        }, {}

    resolved_radius = float(identity_context_radius_px)
    resolved_step = float(identity_context_step_px)
    if (
        not math.isfinite(resolved_radius)
        or not math.isfinite(resolved_step)
        or resolved_radius <= 0.0
        or resolved_step <= 0.0
    ):
        raise ValueError("identity context radius and step must be finite and positive")
    overrides: dict[str, dict[str, float | None]] = {}
    if checkpoint_radius is None or not math.isclose(
        resolved_radius, checkpoint_radius, rel_tol=0.0, abs_tol=1e-12
    ):
        overrides["radius_px"] = {
            "checkpoint_value": checkpoint_radius,
            "resolved_value": resolved_radius,
        }
    if checkpoint_step is None or not math.isclose(
        resolved_step, checkpoint_step, rel_tol=0.0, abs_tol=1e-12
    ):
        overrides["step_px"] = {
            "checkpoint_value": checkpoint_step,
            "resolved_value": resolved_step,
        }
    return {
        "enabled": True,
        "radius_px": resolved_radius,
        "step_px": resolved_step,
    }, overrides


def _resolve_identity_context_layout_config(
    initial_config: Mapping[str, Any],
    *,
    identity_context_layout_grid_size: int | None = None,
) -> tuple[dict[str, int | bool | None], dict[str, dict[str, int | None]]]:
    """Resolve an opt-in broad-context layout residual without changing v6.

    The pooled broad context introduced in v6 and the layout-preserving branch
    are separate evidence paths.  A checkpoint that already declares a layout
    grid must retain it; enabling a grid on a v6 checkpoint explicitly creates
    a new v8 branch whose missing parameters are initialized from scratch.
    """

    checkpoint_enabled = bool(
        initial_config.get("identity_context_layout_enabled", False)
    )
    checkpoint_grid = initial_config.get("identity_context_layout_grid_size")
    if checkpoint_enabled:
        if checkpoint_grid is None:
            raise ValueError("identity-context layout checkpoint lacks grid size")
        checkpoint_grid = int(checkpoint_grid)
    else:
        checkpoint_grid = None
    if identity_context_layout_grid_size is None:
        return {"enabled": checkpoint_enabled, "grid_size": checkpoint_grid}, {}
    resolved_grid = int(identity_context_layout_grid_size)
    if resolved_grid < 2 or resolved_grid > 8:
        raise ValueError("identity_context_layout_grid_size must be in [2, 8]")
    overrides: dict[str, dict[str, int | None]] = {}
    if checkpoint_grid is None or resolved_grid != checkpoint_grid:
        overrides["grid_size"] = {
            "checkpoint_value": checkpoint_grid,
            "resolved_value": resolved_grid,
        }
    return {"enabled": True, "grid_size": resolved_grid}, overrides


def _set_identity_context_residual_training_mode(
    model: IndependentRGBCandidateVerifier,
) -> None:
    """Train broad identity context without updating local evidence buffers.

    ``requires_grad_(False)`` does not stop BatchNorm-style buffers or dropout
    behavior from changing when the parent model is switched to train mode.
    Context-only calibration must therefore keep every local evidence module in
    eval mode while the explicitly trainable broad-context modules remain in
    train mode.
    """

    if (
        model.identity_context_encoder is None
        or model.identity_context_evidence is None
        or model.identity_context_residual is None
    ):
        raise ValueError("identity-context residual training requires context modules")
    model.train()
    for module in (
        model.encoder,
        model.view_evidence,
        model.view_identity_head,
        model.view_measurement_validity_head,
        model.view_pose_mixture_head,
        model.set_candidate_encoder,
        model.measured_set_availability_head,
    ):
        module.eval()
    for module in (
        model.identity_context_encoder,
        model.identity_context_evidence,
        model.identity_context_residual,
    ):
        module.train()


def _set_identity_context_layout_residual_training_mode(
    model: IndependentRGBCandidateVerifier,
) -> None:
    """Train only the isolated layout path while preserving local/v6 paths."""

    if (
        model.identity_context_encoder is None
        or model.identity_context_evidence is None
        or model.identity_context_residual is None
        or model.identity_context_layout_encoder is None
        or model.identity_context_layout_fusion is None
        or model.identity_context_layout_evidence is None
        or model.identity_context_layout_residual is None
    ):
        raise ValueError("identity-context layout training requires v8 context modules")
    model.train()
    for module in (
        model.encoder,
        model.view_evidence,
        model.view_identity_head,
        model.view_measurement_validity_head,
        model.view_pose_mixture_head,
        model.set_candidate_encoder,
        model.measured_set_availability_head,
        model.identity_context_encoder,
        model.identity_context_evidence,
        model.identity_context_residual,
    ):
        module.eval()
    for module in (
        model.identity_context_layout_encoder,
        model.identity_context_layout_fusion,
        model.identity_context_layout_evidence,
        model.identity_context_layout_residual,
    ):
        module.train()


def _load_evidence(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    required = {
        "selected_rows",
        "query_ids",
        "split_names",
        "candidate_valid",
        "candidate_track_ids",
        "candidate_prior_probabilities",
        "unknown_probability",
        "candidate_target_gt_residuals_px",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"candidate evidence lacks arrays: {sorted(missing)}")
    if metadata.get("format") != "candidate_evidence_v3":
        raise ValueError("independent RGB training requires candidate evidence V3")
    if metadata.get("candidate_probability_semantics") != (
        "factorized_top_l_availability_times_conditional_identity"
    ):
        raise ValueError("candidate evidence does not use the factorized identity contract")
    return arrays, metadata


def _row_group_key(row: Mapping[str, object]) -> tuple[str, int]:
    return str(row.get("query_id", "")), int(row.get("source_query_row", -1))


def _load_rows_summary(path: Path, *, split_name: str) -> dict[str, Any]:
    summary_path = Path(path).with_suffix(".summary.json")
    if not summary_path.exists():
        raise ValueError(f"candidate RGB rows lack provenance summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("stage") != "candidate_specific_real_rgb_measurement_rows":
        raise ValueError(f"candidate RGB rows summary has the wrong stage: {summary_path}")
    if str(summary.get("split", "")) != str(split_name):
        raise ValueError(f"candidate RGB rows summary split mismatch: {summary_path}")
    expected_hash = str(dict(summary.get("outputs", {})).get("rows_csv_sha256", ""))
    actual_hash = file_sha256_short(Path(path))
    if expected_hash != actual_hash:
        raise ValueError(
            f"candidate RGB rows are stale for {split_name}: "
            f"summary={expected_hash}, actual={actual_hash}"
        )
    return summary


def _validate_row_coordinate(
    row: Mapping[str, object], key: str, *, upper_bound: int, source: Path
) -> None:
    text = str(row.get(key, "")).strip()
    if not text:
        return
    value = float(text)
    if not math.isfinite(value) or not 0.0 <= value < float(upper_bound):
        raise ValueError(
            f"candidate RGB row {key}={value} lies outside the frozen coordinate "
            f"space [0,{upper_bound}) in {source}"
        )


class CandidateRGBTrainingData:
    def __init__(
        self,
        *,
        candidate_evidence: Path,
        availability_evidence: Path,
        train_rows_csv: Path,
        validation_rows_csv: Path,
        test_rows_csv: Path,
        max_views: int,
    ) -> None:
        self.arrays, self.metadata = _load_evidence(Path(candidate_evidence))
        availability_arrays, self.availability_metadata = _load_evidence(
            Path(availability_evidence)
        )
        self.candidate_evidence_path = Path(candidate_evidence)
        self.availability_evidence_path = Path(availability_evidence)
        self.max_views = int(max_views)
        if self.max_views <= 0:
            raise ValueError("max_views must be positive")
        self.query_ids = np.asarray(self.arrays["query_ids"]).astype(str)
        self.selected_rows = np.asarray(self.arrays["selected_rows"], dtype=np.int64)
        self.split_names = np.asarray(self.arrays["split_names"]).astype(str)
        self.candidate_valid = np.asarray(self.arrays["candidate_valid"], dtype=bool)
        self.track_ids = np.asarray(self.arrays["candidate_track_ids"], dtype=np.int64)
        self.prior = np.asarray(self.arrays["candidate_prior_probabilities"], dtype=np.float32)
        self.unknown = np.asarray(self.arrays["unknown_probability"], dtype=np.float32)
        self.residuals = np.asarray(
            self.arrays["candidate_target_gt_residuals_px"], dtype=np.float32
        )
        for key in ("selected_rows", "query_ids", "split_names"):
            if not np.array_equal(self.arrays[key], availability_arrays[key]):
                raise ValueError(
                    f"candidate and availability evidence differ in {key}"
                )
        availability_valid = np.asarray(
            availability_arrays["candidate_valid"], dtype=bool
        )
        self.availability_residuals = np.asarray(
            availability_arrays["candidate_target_gt_residuals_px"],
            dtype=np.float32,
        )
        if availability_valid.shape != self.availability_residuals.shape:
            raise ValueError("availability evidence arrays have incompatible shapes")
        self.availability_valid = availability_valid
        if self.candidate_valid.shape != self.track_ids.shape or self.prior.shape != self.track_ids.shape:
            raise ValueError("candidate evidence arrays have incompatible shapes")
        if self.residuals.shape != self.track_ids.shape:
            raise ValueError("candidate target residuals have incompatible shape")
        expected_mass = np.sum(
            np.where(self.candidate_valid, self.prior, 0.0), axis=1
        ) + self.unknown
        if not np.allclose(expected_mass, 1.0, atol=2e-5, rtol=0.0):
            raise ValueError("candidate evidence probability mass is not conserved")

        key_to_index: dict[tuple[str, int], int] = {}
        for index, (query_id, selected_row) in enumerate(
            zip(self.query_ids.tolist(), self.selected_rows.tolist())
        ):
            key = (str(query_id), int(selected_row))
            if key in key_to_index:
                raise ValueError(f"candidate evidence group key is not unique: {key}")
            key_to_index[key] = int(index)
        self.rows_by_group: dict[int, list[list[dict[str, str]]]] = {
            index: [[] for _ in range(int(self.track_ids.shape[1]))]
            for index in range(len(self.query_ids))
        }
        self.actual_query_observation = np.zeros_like(
            self.candidate_valid, dtype=bool
        )
        self.actual_center_residuals = np.full(
            self.candidate_valid.shape, np.inf, dtype=np.float32
        )
        self.rows_paths = {
            "train": Path(train_rows_csv),
            "validation": Path(validation_rows_csv),
            "test": Path(test_rows_csv),
        }
        self.rows_summaries = {
            split: _load_rows_summary(path, split_name=split)
            for split, path in self.rows_paths.items()
        }
        self.coordinate_space = coordinate_space_from_evidence(self.metadata)
        availability_coordinate_space = coordinate_space_from_evidence(
            self.availability_metadata
        )
        if (
            availability_coordinate_space["coordinate_space_id"]
            != self.coordinate_space["coordinate_space_id"]
        ):
            raise ValueError(
                "candidate and availability evidence use different coordinate spaces"
            )
        for split, summary in self.rows_summaries.items():
            summary_coordinate = dict(summary.get("coordinate_space", {}))
            if summary_coordinate and (
                summary_coordinate.get("coordinate_space_id")
                != self.coordinate_space["coordinate_space_id"]
            ):
                raise ValueError(
                    f"{split} RGB rows use a different coordinate space than candidate evidence"
                )
        self.image_ids: set[str] = set()
        for split_name, path in self.rows_paths.items():
            rows = _read_csv(Path(path))
            for source_row_index, row in enumerate(rows):
                if str(row.get("split", "")) != split_name:
                    raise ValueError(f"RGB row split mismatch in {path}")
                for key in ("center_x", "support_x", "render_x"):
                    _validate_row_coordinate(
                        row,
                        key,
                        upper_bound=int(self.coordinate_space["image_width"]),
                        source=Path(path),
                    )
                for key in ("center_y", "support_y", "render_y"):
                    _validate_row_coordinate(
                        row,
                        key,
                        upper_bound=int(self.coordinate_space["image_height"]),
                        source=Path(path),
                    )
                self.image_ids.add(str(row.get("query_id", "")).strip())
                self.image_ids.add(str(row.get("support_image_id", "")).strip())
                index = key_to_index.get(_row_group_key(row))
                if index is None:
                    raise ValueError("RGB row does not map to the frozen candidate evidence")
                if self.split_names[index] != split_name:
                    raise ValueError("RGB row crosses the frozen query split")
                candidate = int(row.get("candidate_measurement_rank", 0)) - 1
                if candidate < 0 or candidate >= int(self.track_ids.shape[1]):
                    raise ValueError("candidate measurement rank is out of range")
                if int(row.get("track_id", -1)) != int(self.track_ids[index, candidate]):
                    raise ValueError("RGB row track identity differs from candidate evidence")
                actual = _bool_text(row.get("actual_query_observation"))
                center_residual = _float_text(
                    row.get("center_residual_px"), default=float("inf")
                )
                previous_rows = self.rows_by_group[index][candidate]
                if previous_rows:
                    if bool(self.actual_query_observation[index, candidate]) != bool(actual):
                        raise ValueError("support views disagree on query observation target")
                    previous_residual = float(
                        self.actual_center_residuals[index, candidate]
                    )
                    if not (
                        (math.isinf(previous_residual) and math.isinf(center_residual))
                        or math.isclose(
                            previous_residual,
                            center_residual,
                            rel_tol=0.0,
                            abs_tol=1e-4,
                        )
                    ):
                        raise ValueError("support views disagree on center residual")
                self.actual_query_observation[index, candidate] = bool(actual)
                self.actual_center_residuals[index, candidate] = float(center_residual)
                stored_row = dict(row)
                stored_row["__source_row_index__"] = str(source_row_index)
                self.rows_by_group[index][candidate].append(stored_row)
        for candidate_rows in self.rows_by_group.values():
            for rows in candidate_rows:
                rows.sort(
                    key=lambda row: (
                        int(row.get("support_view_rank", 10**9)),
                        str(row.get("support_image_id", "")),
                    )
                )
                del rows[self.max_views :]
        self.indices_by_split = {
            split: np.flatnonzero(self.split_names == split).astype(np.int64)
            for split in ("train", "validation", "test")
        }
        self.has_any_rgb = np.asarray(
            [any(self.rows_by_group[index]) for index in range(len(self.query_ids))],
            dtype=bool,
        )
        for split, indices in self.indices_by_split.items():
            if int(indices.size) == 0:
                raise ValueError(f"candidate RGB data has no {split} groups")

    def runtime_contract(
        self, *, image_root: Path, image_width: int, image_height: int
    ) -> dict[str, Any]:
        validate_sampling_dimensions(
            self.coordinate_space,
            image_width=int(image_width),
            image_height=int(image_height),
        )
        return {
            "version": 1,
            "coordinate_space": dict(self.coordinate_space),
            "image_source": image_root_manifest(
                Path(image_root), sorted(self.image_ids)
            ),
            "candidate_evidence_sha256": file_sha256_short(
                self.candidate_evidence_path
            ),
            "availability_evidence_sha256": file_sha256_short(
                self.availability_evidence_path
            ),
            "rows_sha256": {
                split: file_sha256_short(path)
                for split, path in sorted(self.rows_paths.items())
            },
        }

    def summary(self) -> dict[str, object]:
        split_summary: dict[str, object] = {}
        for split, indices in self.indices_by_split.items():
            valid = self.candidate_valid[indices]
            measured = np.zeros_like(valid)
            view_counts = np.zeros_like(valid, dtype=np.int64)
            for local_row, index in enumerate(indices.tolist()):
                for candidate, rows in enumerate(self.rows_by_group[int(index)]):
                    measured[local_row, candidate] = bool(rows)
                    view_counts[local_row, candidate] = len(rows)
            split_summary[split] = {
                "group_count": int(len(indices)),
                "groups_with_any_rgb": int(np.count_nonzero(self.has_any_rgb[indices])),
                "groups_without_rgb": int(np.count_nonzero(~self.has_any_rgb[indices])),
                "candidate_count": int(np.count_nonzero(valid)),
                "measured_candidate_count": int(np.count_nonzero(valid & measured)),
                "measured_candidate_rate": float(
                    np.count_nonzero(valid & measured) / max(np.count_nonzero(valid), 1)
                ),
                "mean_views_per_measured_candidate": float(
                    np.mean(view_counts[measured]) if np.any(measured) else 0.0
                ),
                "actual_query_observation_candidate_count": int(
                    np.count_nonzero(self.actual_query_observation[indices] & valid)
                ),
            }
        return split_summary

    def view_identity_positive_prior(
        self,
        *,
        split_name: str,
        positive_threshold_px: float,
        negative_threshold_px: float,
        identity_target: str = IDENTITY_TARGET_APPEARANCE,
    ) -> float:
        positive_views = 0
        total_views = 0
        for index in self.indices_by_split[str(split_name)].tolist():
            for candidate, rows in enumerate(self.rows_by_group[int(index)]):
                if not bool(self.candidate_valid[int(index), candidate]):
                    continue
                count = len(rows)
                labels, supervised = _identity_training_masks(
                    valid=self.candidate_valid[int(index), candidate : candidate + 1],
                    actual_query_observation=self.actual_query_observation[
                        int(index), candidate : candidate + 1
                    ],
                    actual_center_residuals=self.actual_center_residuals[
                        int(index), candidate : candidate + 1
                    ],
                    target_projection_residuals=self.residuals[
                        int(index), candidate : candidate + 1
                    ],
                    identity_target=str(identity_target),
                    positive_threshold_px=float(positive_threshold_px),
                    negative_threshold_px=float(negative_threshold_px),
                )
                if bool(supervised[0]):
                    total_views += count
                if bool(labels[0]):
                    positive_views += count
        if positive_views <= 0 or positive_views >= total_views:
            raise ValueError("view identity prior requires both natural target classes")
        return float(positive_views / total_views)

    def availability_positive_prior(
        self,
        *,
        split_name: str,
        threshold_px: float,
        require_any_rgb: bool,
    ) -> float:
        indices = self.indices_by_split[str(split_name)]
        if bool(require_any_rgb):
            indices = indices[self.has_any_rgb[indices]]
        labels = self.availability_valid[indices] & np.isfinite(
            self.availability_residuals[indices]
        ) & (self.availability_residuals[indices] <= float(threshold_px))
        positive = int(np.count_nonzero(np.any(labels, axis=1)))
        total = int(len(indices))
        if positive <= 0 or positive >= total:
            raise ValueError("availability prior requires both natural target classes")
        return float(positive / total)


def _prepare_batch(
    data: CandidateRGBTrainingData,
    evidence_indices: Sequence[int],
    *,
    image_root: Path,
    image_width: int,
    image_height: int,
    image_cache: TensorImageLRUCache,
    image_cache_device: torch.device | None,
    image_crop_device: torch.device | None = None,
    crop_radius_px: float,
    step_px: float,
    identity_context_crop_radius_px: float | None = None,
    identity_context_step_px: float | None = None,
    identity_threshold_px: float,
    identity_negative_threshold_px: float,
    identity_target: str = IDENTITY_TARGET_APPEARANCE,
    spatial_radius_px: float,
    natural_supervision_mask: Sequence[bool] | None = None,
) -> dict[str, object]:
    flat_rows: list[dict[str, str]] = []
    pair_groups: list[int] = []
    pair_candidates: list[int] = []
    pair_slots: list[int] = []
    pair_view_probabilities: list[float] = []
    for group, evidence_index in enumerate(evidence_indices):
        for candidate, rows in enumerate(data.rows_by_group[int(evidence_index)]):
            for slot, row in enumerate(rows):
                flat_rows.append(row)
                pair_groups.append(int(group))
                pair_candidates.append(int(candidate))
                pair_slots.append(int(slot))
                pair_view_probabilities.append(
                    _float_text(row.get("support_view_probability"), default=0.0)
                )
    if not flat_rows:
        raise ValueError("candidate RGB batch has no measurable views")
    query_patch, support_patch, _target, _baseline, _dustbin = _stack_patch_batch(
        flat_rows,
        image_root=Path(image_root),
        render_cache_by_query={},
        image_width=int(image_width),
        image_height=int(image_height),
        crop_radius_px=float(crop_radius_px),
        step_px=float(step_px),
        query_cache=image_cache,
        render_cache=image_cache,
        query_source="real_pair",
        render_patch_augmentation="none",
        support_patch_warp="none",
        image_cache_device=image_cache_device,
        crop_device=image_crop_device,
    )
    first_pair_by_group = [-1] * len(evidence_indices)
    for pair_row, group in enumerate(pair_groups):
        if first_pair_by_group[group] < 0:
            first_pair_by_group[group] = int(pair_row)
    if any(value < 0 for value in first_pair_by_group):
        raise ValueError("candidate RGB batch contains a group without a query patch")
    if (identity_context_crop_radius_px is None) != (
        identity_context_step_px is None
    ):
        raise ValueError(
            "identity context crop radius and step must be provided together"
        )
    identity_context_query_patch: torch.Tensor | None = None
    identity_context_support_patch: torch.Tensor | None = None
    if identity_context_crop_radius_px is not None:
        (
            context_query_patch,
            context_support_patch,
            _context_target,
            _context_baseline,
            _context_dustbin,
        ) = _stack_patch_batch(
            flat_rows,
            image_root=Path(image_root),
            render_cache_by_query={},
            image_width=int(image_width),
            image_height=int(image_height),
            crop_radius_px=float(identity_context_crop_radius_px),
            step_px=float(identity_context_step_px),
            query_cache=image_cache,
            render_cache=image_cache,
            query_source="real_pair",
            render_patch_augmentation="none",
            support_patch_warp="none",
            image_cache_device=image_cache_device,
            crop_device=image_crop_device,
        )
        if int(context_query_patch.shape[0]) != len(flat_rows) or int(
            context_support_patch.shape[0]
        ) != len(flat_rows):
            raise ValueError("identity context crops are misaligned with RGB pairs")
        identity_context_query_patch = context_query_patch[
            torch.tensor(
                first_pair_by_group,
                dtype=torch.long,
                device=context_query_patch.device,
            )
        ]
        identity_context_support_patch = context_support_patch
    indices = np.asarray(evidence_indices, dtype=np.int64)
    natural_mask = np.ones((len(indices),), dtype=bool)
    if natural_supervision_mask is not None:
        natural_mask = np.asarray(natural_supervision_mask, dtype=bool).reshape(-1)
        if natural_mask.shape != (len(indices),) or not np.any(natural_mask):
            raise ValueError("natural supervision mask must select at least one batch group")
    valid = data.candidate_valid[indices]
    residuals = data.residuals[indices]
    labels, identity_supervision_valid = _identity_training_masks(
        valid=valid,
        actual_query_observation=data.actual_query_observation[indices],
        actual_center_residuals=data.actual_center_residuals[indices],
        target_projection_residuals=residuals,
        identity_target=str(identity_target),
        positive_threshold_px=float(identity_threshold_px),
        negative_threshold_px=float(identity_negative_threshold_px),
    )
    availability_labels = data.availability_valid[indices] & np.isfinite(
        data.availability_residuals[indices]
    ) & (data.availability_residuals[indices] <= float(identity_threshold_px))
    spatial_target = np.asarray(
        [
            [
                _float_text(row.get("query_gt_x")) - _float_text(row.get("center_x")),
                _float_text(row.get("query_gt_y")) - _float_text(row.get("center_y")),
            ]
            for row in flat_rows
        ],
        dtype=np.float32,
    )
    spatial_valid = np.asarray(
        [
            _bool_text(row.get("actual_query_observation"))
            and float(row.get("center_residual_px", float("inf")))
            <= float(spatial_radius_px)
            for row in flat_rows
        ],
        dtype=bool,
    )
    target_gt_projected_offset = np.asarray(
        [
            [
                _float_text(row.get("target_gt_projected_x"))
                - _float_text(row.get("center_x")),
                _float_text(row.get("target_gt_projected_y"))
                - _float_text(row.get("center_y")),
            ]
            for row in flat_rows
        ],
        dtype=np.float32,
    )
    measurement_validity_supervision_weight = np.asarray(
        [
            _float_text(row.get("geometry_supervision_weight"), default=0.0)
            for row in flat_rows
        ],
        dtype=np.float32,
    )
    if np.any(~np.isfinite(measurement_validity_supervision_weight)) or np.any(
        measurement_validity_supervision_weight < 0.0
    ):
        raise ValueError(
            "measurement validity supervision weights must be finite and non-negative"
        )
    target_gt_projection_evaluable = (
        np.all(np.isfinite(target_gt_projected_offset), axis=1)
        & (measurement_validity_supervision_weight > 0.0)
    )
    target_gt_projection_physical_valid = target_gt_projection_evaluable & np.asarray(
        [
            _bool_text(row.get("target_gt_projection_in_front"))
            and _bool_text(row.get("target_gt_projection_in_image"))
            for row in flat_rows
        ],
        dtype=bool,
    )
    stored_gt_residual = np.asarray(
        [
            _float_text(row.get("target_gt_projected_residual_px"))
            for row in flat_rows
        ],
        dtype=np.float32,
    )
    recomputed_gt_residual = np.linalg.norm(target_gt_projected_offset, axis=1)
    comparable = target_gt_projection_evaluable & np.isfinite(stored_gt_residual)
    if np.any(
        ~np.isclose(
            recomputed_gt_residual[comparable],
            stored_gt_residual[comparable],
            rtol=1e-4,
            atol=1e-3,
        )
    ):
        raise ValueError("GT projected offsets disagree with stored projection residuals")
    candidate_shape = (len(indices), int(valid.shape[1]))
    candidate_target_gt_projected_offset = np.full(
        (*candidate_shape, 2), np.nan, dtype=np.float32
    )
    candidate_target_gt_projection_evaluable = np.zeros(
        candidate_shape, dtype=bool
    )
    candidate_target_gt_projection_physical_valid = np.zeros(
        candidate_shape, dtype=bool
    )
    candidate_measurement_supervision_weight = np.zeros(
        candidate_shape, dtype=np.float32
    )
    candidate_target_assigned = np.zeros(candidate_shape, dtype=bool)
    for pair_row, (group, candidate) in enumerate(
        zip(pair_groups, pair_candidates)
    ):
        if not bool(valid[group, candidate]):
            raise ValueError("RGB row references an invalid candidate")
        if candidate_target_assigned[group, candidate]:
            if not np.allclose(
                candidate_target_gt_projected_offset[group, candidate],
                target_gt_projected_offset[pair_row],
                rtol=0.0,
                atol=1e-4,
                equal_nan=True,
            ):
                raise ValueError(
                    "support views disagree on candidate GT projected offset"
                )
            if (
                bool(candidate_target_gt_projection_evaluable[group, candidate])
                != bool(target_gt_projection_evaluable[pair_row])
                or bool(
                    candidate_target_gt_projection_physical_valid[
                        group, candidate
                    ]
                )
                != bool(target_gt_projection_physical_valid[pair_row])
                or not math.isclose(
                    float(
                        candidate_measurement_supervision_weight[
                            group, candidate
                        ]
                    ),
                    float(measurement_validity_supervision_weight[pair_row]),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError(
                    "support views disagree on candidate GT projection contract"
                )
            continue
        candidate_target_gt_projected_offset[group, candidate] = (
            target_gt_projected_offset[pair_row]
        )
        candidate_target_gt_projection_evaluable[group, candidate] = (
            target_gt_projection_evaluable[pair_row]
        )
        candidate_target_gt_projection_physical_valid[group, candidate] = (
            target_gt_projection_physical_valid[pair_row]
        )
        candidate_measurement_supervision_weight[group, candidate] = (
            measurement_validity_supervision_weight[pair_row]
        )
        candidate_target_assigned[group, candidate] = True
    return {
        "evidence_indices": indices,
        "query_patches_by_group": query_patch[
            torch.tensor(first_pair_by_group, dtype=torch.long, device=query_patch.device)
        ],
        "support_patches": support_patch,
        "identity_context_query_patches_by_group": identity_context_query_patch,
        "identity_context_support_patches": identity_context_support_patch,
        "pair_group_indices": torch.tensor(pair_groups, dtype=torch.long),
        "pair_candidate_indices": torch.tensor(pair_candidates, dtype=torch.long),
        "pair_view_slots": torch.tensor(pair_slots, dtype=torch.long),
        "pair_view_probabilities": torch.tensor(
            pair_view_probabilities, dtype=torch.float32
        ),
        "candidate_valid": torch.from_numpy(valid),
        "candidate_labels": torch.from_numpy(labels),
        "identity_supervision_valid": torch.from_numpy(identity_supervision_valid),
        "availability_target": torch.from_numpy(np.any(availability_labels, axis=1)),
        "natural_supervision_mask": torch.from_numpy(natural_mask),
        "candidate_prior": torch.from_numpy(data.prior[indices]),
        "unknown_probability": torch.from_numpy(data.unknown[indices]),
        "spatial_target_xy": torch.from_numpy(spatial_target),
        "spatial_valid": torch.from_numpy(spatial_valid),
        "target_gt_projected_offset_xy": torch.from_numpy(
            target_gt_projected_offset
        ),
        "target_gt_projection_evaluable": torch.from_numpy(
            target_gt_projection_evaluable
        ),
        "target_gt_projection_physical_valid": torch.from_numpy(
            target_gt_projection_physical_valid
        ),
        "measurement_validity_supervision_weight": torch.from_numpy(
            measurement_validity_supervision_weight
        ),
        "candidate_target_gt_projected_offset_xy": torch.from_numpy(
            candidate_target_gt_projected_offset
        ),
        "candidate_target_gt_projection_evaluable": torch.from_numpy(
            candidate_target_gt_projection_evaluable
        ),
        "candidate_target_gt_projection_physical_valid": torch.from_numpy(
            candidate_target_gt_projection_physical_valid
        ),
        "candidate_measurement_supervision_weight": torch.from_numpy(
            candidate_measurement_supervision_weight
        ),
        "flat_rows": flat_rows,
    }


def _forward_batch(
    model: IndependentRGBCandidateVerifier,
    batch: Mapping[str, object],
    *,
    device: torch.device,
    use_amp: bool,
    identity_context_residual_scale: float = 1.0,
    identity_context_layout_residual_scale: float = 0.0,
) -> RGBCandidateIdentityPrediction:
    with torch.autocast(
        device_type=device.type,
        dtype=(torch.float16 if device.type == "cuda" else torch.bfloat16),
        enabled=bool(use_amp and device.type == "cuda"),
    ):
        kwargs: dict[str, torch.Tensor] = {
            "query_patches_by_group": batch["query_patches_by_group"].to(device),
            "support_patches": batch["support_patches"].to(device),
            "pair_group_indices": batch["pair_group_indices"].to(device),
            "pair_candidate_indices": batch["pair_candidate_indices"].to(device),
            "pair_view_slots": batch["pair_view_slots"].to(device),
            "pair_view_probabilities": batch["pair_view_probabilities"].to(device),
            "candidate_valid": batch["candidate_valid"].to(device),
            "identity_context_residual_scale": float(
                identity_context_residual_scale
            ),
            "identity_context_layout_residual_scale": float(
                identity_context_layout_residual_scale
            ),
        }
        context_requested = (
            float(identity_context_residual_scale) > 0.0
            or float(identity_context_layout_residual_scale) > 0.0
        )
        if model.identity_context_enabled and context_requested:
            context_query = batch["identity_context_query_patches_by_group"]
            context_support = batch["identity_context_support_patches"]
            if context_query is None or context_support is None:
                raise ValueError("dual-scale verifier batch lacks identity context crops")
            kwargs["identity_context_query_patches_by_group"] = context_query.to(device)
            kwargs["identity_context_support_patches"] = context_support.to(device)
        return model(**kwargs)


def gt_pose_spatial_density_nll(
    spatial_logits: torch.Tensor,
    non_dustbin_logits: torch.Tensor,
    offsets_xy: torch.Tensor,
    target_offsets_xy: torch.Tensor,
    target_evaluable: torch.Tensor,
    target_physical_valid: torch.Tensor,
    *,
    target_sigma_px: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate normalized K+1 density against target-only GT projections.

    A projection inside the finite local support receives a Gaussian soft-bin
    target. A projection outside the support, behind the camera, or outside the
    image receives dustbin. ``target_evaluable`` is returned separately so
    callers can apply split and provenance-safe supervision weights.
    """

    if float(target_sigma_px) <= 0.0:
        raise ValueError("spatial density target sigma must be positive")
    if spatial_logits.ndim != 2:
        raise ValueError("spatial logits must have shape (N,K)")
    offsets = offsets_xy.to(
        device=spatial_logits.device, dtype=torch.float32
    ).reshape(-1, 2)
    targets = target_offsets_xy.to(
        device=spatial_logits.device, dtype=torch.float32
    ).reshape(-1, 2)
    evaluable = target_evaluable.to(
        device=spatial_logits.device, dtype=torch.bool
    ).reshape(-1)
    physical = target_physical_valid.to(
        device=spatial_logits.device, dtype=torch.bool
    ).reshape(-1)
    row_count = int(spatial_logits.shape[0])
    if (
        int(offsets.shape[0]) != int(spatial_logits.shape[1])
        or int(targets.shape[0]) != row_count
        or int(evaluable.numel()) != row_count
        or int(physical.numel()) != row_count
    ):
        raise ValueError("spatial density targets and predictions are misaligned")
    finite_target = torch.all(torch.isfinite(targets), dim=1)
    minimum = torch.min(offsets, dim=0).values
    maximum = torch.max(offsets, dim=0).values
    inside = (
        evaluable
        & physical
        & finite_target
        & torch.all(targets >= minimum[None], dim=1)
        & torch.all(targets <= maximum[None], dim=1)
    )
    safe_targets = torch.where(
        finite_target[:, None], targets, torch.zeros_like(targets)
    )
    dist2 = torch.sum(
        (offsets[None, :, :] - safe_targets[:, None, :]) ** 2,
        dim=2,
    )
    soft_target = F.softmax(
        -0.5 * dist2 / max(float(target_sigma_px) ** 2, 1e-8), dim=1
    ).detach()
    joint_log_probability = normalized_spatial_log_probabilities_with_dustbin(
        spatial_logits.float(), non_dustbin_logits.float()
    )
    local_nll = -torch.sum(
        soft_target * joint_log_probability[:, :-1], dim=1
    )
    dustbin_nll = -joint_log_probability[:, -1]
    per_row_nll = torch.where(inside, local_nll, dustbin_nll)
    return per_row_nll, inside, evaluable


def gt_pose_candidate_view_mixture_nll(
    spatial_logits: torch.Tensor,
    non_dustbin_logits: torch.Tensor,
    view_probabilities: torch.Tensor,
    offsets_xy: torch.Tensor,
    target_offsets_xy: torch.Tensor,
    target_evaluable: torch.Tensor,
    target_physical_valid: torch.Tensor,
    *,
    pair_group_indices: torch.Tensor,
    pair_candidate_indices: torch.Tensor,
    pair_view_slots: torch.Tensor,
    max_views: int,
    target_sigma_px: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Candidate-level GT density NLL after marginalizing support views."""

    targets = target_offsets_xy.to(
        device=spatial_logits.device, dtype=torch.float32
    )
    if targets.ndim != 3 or int(targets.shape[2]) != 2:
        raise ValueError("candidate target offsets must have shape (B,L,2)")
    batch_size, candidate_count = int(targets.shape[0]), int(targets.shape[1])
    joint_log_probability = normalized_spatial_log_probabilities_with_dustbin(
        spatial_logits.float(), non_dustbin_logits.float()
    )
    candidate_log_probability, measured = (
        aggregate_pose_view_spatial_log_probabilities(
            joint_log_probability,
            view_probabilities,
            pair_group_indices=pair_group_indices,
            pair_candidate_indices=pair_candidate_indices,
            pair_view_slots=pair_view_slots,
            batch_size=batch_size,
            candidate_count=candidate_count,
            max_views=int(max_views),
        )
    )
    flat_nll, flat_inside, flat_evaluable = gt_pose_spatial_density_nll(
        candidate_log_probability[..., :-1].reshape(
            batch_size * candidate_count, -1
        ),
        (
            torch.logsumexp(candidate_log_probability[..., :-1], dim=2)
            - candidate_log_probability[..., -1]
        ).reshape(-1),
        offsets_xy,
        targets.reshape(-1, 2),
        target_evaluable.to(device=spatial_logits.device).reshape(-1),
        target_physical_valid.to(device=spatial_logits.device).reshape(-1),
        target_sigma_px=float(target_sigma_px),
    )
    # The helper above reconstructs the same K+1 probability object from its
    # conditional local map and log-odds. Check this contract before training.
    reconstructed = normalized_spatial_log_probabilities_with_dustbin(
        candidate_log_probability[..., :-1].reshape(
            batch_size * candidate_count, -1
        ),
        (
            torch.logsumexp(candidate_log_probability[..., :-1], dim=2)
            - candidate_log_probability[..., -1]
        ).reshape(-1),
    ).reshape_as(candidate_log_probability)
    if not bool(
        torch.allclose(
            reconstructed[measured],
            candidate_log_probability[measured],
            atol=2e-5,
            rtol=2e-5,
        )
    ):
        raise RuntimeError("candidate support-view mixture lost K+1 probability mass")
    shape = (batch_size, candidate_count)
    evaluable = flat_evaluable.reshape(shape) & measured
    return flat_nll.reshape(shape), flat_inside.reshape(shape), evaluable


def _weighted_selected_mean(
    values: torch.Tensor,
    selected: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Maximum-likelihood mean over the unmodified natural row distribution."""

    if not bool(torch.any(selected)):
        return values.new_zeros(())
    return (
        torch.sum(values[selected] * weights[selected])
        / weights[selected].sum().clamp_min(1e-8)
    )


def coarse_prior_hard_negative_ranking_loss(
    candidate_log_likelihood_ratios: torch.Tensor,
    *,
    labels: torch.Tensor,
    supervision_valid: torch.Tensor,
    candidate_prior: torch.Tensor,
    margin: float,
    prior_power: float,
    evidence_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank geometric positives above frozen high-prior hard negatives.

    The frozen coarse probabilities are used only to weight the training
    comparison.  They are not an RGB-model input: at inference the verifier
    still emits a target-free candidate log-likelihood ratio.  This loss
    mirrors the declared conditional candidate fusion used downstream, so a
    visual residual must overcome a confusable candidate's coarse prior rather
    than merely win an unweighted listwise comparison.
    """

    logits = candidate_log_likelihood_ratios.float()
    positive = labels.to(device=logits.device, dtype=torch.bool)
    supervised = supervision_valid.to(device=logits.device, dtype=torch.bool)
    prior = candidate_prior.to(device=logits.device, dtype=torch.float32)
    if logits.ndim != 2:
        raise ValueError("candidate log-likelihood ratios must be [groups,candidates]")
    if positive.shape != logits.shape or supervised.shape != logits.shape:
        raise ValueError("hard-negative labels do not match candidate logits")
    if prior.shape != logits.shape:
        raise ValueError("hard-negative candidate priors do not match candidate logits")
    if not bool(torch.all(torch.isfinite(prior))) or bool(torch.any(prior < 0.0)):
        raise ValueError("hard-negative candidate priors must be finite and non-negative")
    if not math.isfinite(float(margin)) or float(margin) < 0.0:
        raise ValueError("coarse hard-negative margin must be finite and non-negative")
    if not math.isfinite(float(prior_power)) or float(prior_power) <= 0.0:
        raise ValueError("coarse hard-negative prior power must be finite and positive")
    if not math.isfinite(float(evidence_weight)) or float(evidence_weight) <= 0.0:
        raise ValueError("coarse hard-negative evidence weight must be finite and positive")

    positive = positive & supervised
    negative = ~positive & supervised
    active = torch.any(positive, dim=1) & torch.any(negative, dim=1)
    if not bool(torch.any(active)):
        return torch.zeros((), device=logits.device), active

    # Candidate priors normally have positive retained mass.  The uniform
    # fallback makes malformed all-zero supervised rows neutral rather than
    # silently producing an infinite target preference.
    supervised_prior_mass = torch.sum(
        torch.where(supervised, prior, torch.zeros_like(prior)), dim=1
    )
    log_prior = float(prior_power) * torch.log(
        prior.clamp_min(torch.finfo(torch.float32).tiny)
    )
    uniform_log_prior = torch.zeros_like(log_prior)
    log_prior = torch.where(
        (supervised_prior_mass <= 0.0)[:, None], uniform_log_prior, log_prior
    )
    positive_log_mass = torch.logsumexp(
        torch.where(
            positive,
            log_prior + float(evidence_weight) * logits,
            torch.full_like(logits, -torch.inf),
        ),
        dim=1,
    )
    negative_log_mass = torch.logsumexp(
        torch.where(
            negative,
            log_prior + float(evidence_weight) * logits,
            torch.full_like(logits, -torch.inf),
        ),
        dim=1,
    )
    per_group = F.softplus(
        float(margin) + negative_log_mass - positive_log_mass
    )
    return torch.mean(per_group[active]), active


def _training_loss(
    prediction: RGBCandidateIdentityPrediction,
    batch: Mapping[str, object],
    *,
    identity_loss_weight: float,
    availability_loss_weight: float,
    pair_loss_weight: float,
    coarse_hard_negative_loss_weight: float,
    coarse_hard_negative_margin: float,
    coarse_hard_negative_prior_power: float,
    coarse_hard_negative_evidence_weight: float,
    spatial_loss_weight: float,
    spatial_target_sigma_px: float,
    measurement_validity_loss_weight: float,
    measurement_success_threshold_px: float,
    spatial_density_loss_weight: float = 0.0,
    pose_view_mixture_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = prediction.candidate_log_likelihood_ratios.device
    labels = batch["candidate_labels"].to(device=device, dtype=torch.bool)
    supervision_valid = batch["identity_supervision_valid"].to(
        device=device, dtype=torch.bool
    )
    positive_groups = torch.any(labels, dim=1)
    supervised_logits = torch.where(
        supervision_valid,
        prediction.candidate_log_likelihood_ratios,
        torch.full_like(prediction.candidate_log_likelihood_ratios, -1e9),
    )
    conditional_log_probs = F.log_softmax(supervised_logits, dim=1)
    correct_log_mass = torch.logsumexp(
        torch.where(labels, conditional_log_probs, torch.full_like(conditional_log_probs, -1e9)),
        dim=1,
    )
    identity_loss = (
        -torch.mean(correct_log_mass[positive_groups])
        if bool(torch.any(positive_groups))
        else torch.zeros((), device=device)
    )
    q_target = batch["availability_target"].to(device=device, dtype=torch.float32)
    natural_groups = batch["natural_supervision_mask"].to(
        device=device, dtype=torch.bool
    )
    availability_loss = F.binary_cross_entropy_with_logits(
        prediction.measured_set_availability_logit[natural_groups],
        q_target[natural_groups],
    )
    pair_groups = batch["pair_group_indices"].to(device=device, dtype=torch.long)
    pair_candidates = batch["pair_candidate_indices"].to(device=device, dtype=torch.long)
    pair_target = labels[pair_groups, pair_candidates].float()
    pair_supervised = (
        supervision_valid[pair_groups, pair_candidates]
        & natural_groups[pair_groups]
    )
    pair_loss = (
        F.binary_cross_entropy_with_logits(
            prediction.view_identity_logits[pair_supervised],
            pair_target[pair_supervised],
        )
        if bool(torch.any(pair_supervised))
        else torch.zeros((), device=device)
    )
    coarse_hard_negative_loss, coarse_hard_negative_groups = (
        coarse_prior_hard_negative_ranking_loss(
            prediction.candidate_log_likelihood_ratios,
            labels=labels,
            supervision_valid=supervision_valid,
            candidate_prior=batch["candidate_prior"],
            margin=float(coarse_hard_negative_margin),
            prior_power=float(coarse_hard_negative_prior_power),
            evidence_weight=float(coarse_hard_negative_evidence_weight),
        )
    )

    spatial_valid = batch["spatial_valid"].to(device=device, dtype=torch.bool)
    spatial_loss = torch.zeros((), device=device)
    if bool(torch.any(spatial_valid)):
        logits = prediction.view_spatial_logits[spatial_valid]
        target = batch["spatial_target_xy"].to(device=device, dtype=torch.float32)[
            spatial_valid
        ]
        offsets = prediction.view_spatial_offsets_xy.to(
            device=device, dtype=torch.float32
        )
        dist2 = torch.sum((offsets[None, :, :] - target[:, None, :]) ** 2, dim=2)
        target_probability = F.softmax(
            -0.5 * dist2 / max(float(spatial_target_sigma_px) ** 2, 1e-8), dim=1
        ).detach()
        spatial_loss = -torch.mean(
            torch.sum(target_probability * F.log_softmax(logits, dim=1), dim=1)
        )

    measurement_residual, measurement_success = (
        measurement_mode_residual_and_success(
            prediction.view_spatial_logits,
            prediction.view_spatial_offsets_xy,
            batch["target_gt_projected_offset_xy"],
            batch["target_gt_projection_physical_valid"],
            success_threshold_px=float(measurement_success_threshold_px),
        )
    )
    measurement_supervision = batch["target_gt_projection_evaluable"].to(
        device=device, dtype=torch.bool
    ) & natural_groups[pair_groups]
    measurement_weight = batch["measurement_validity_supervision_weight"].to(
        device=device, dtype=torch.float32
    )
    measurement_validity_loss = torch.zeros((), device=device)
    if bool(torch.any(measurement_supervision)):
        per_view_loss = F.binary_cross_entropy_with_logits(
            prediction.view_measurement_validity_logits[measurement_supervision],
            measurement_success[measurement_supervision].float(),
            reduction="none",
        )
        selected_weight = measurement_weight[measurement_supervision]
        selected_target = measurement_success[measurement_supervision]
        positive = selected_target
        negative = ~selected_target
        class_losses: list[torch.Tensor] = []
        for class_mask in (positive, negative):
            if bool(torch.any(class_mask)):
                class_losses.append(
                    torch.sum(
                        per_view_loss[class_mask] * selected_weight[class_mask]
                    )
                    / selected_weight[class_mask].sum().clamp_min(1e-8)
                )
        measurement_validity_loss = torch.mean(torch.stack(class_losses))

    spatial_density_per_view_nll, spatial_density_non_dustbin, (
        spatial_density_evaluable
    ) = gt_pose_spatial_density_nll(
        prediction.view_spatial_logits,
        prediction.view_measurement_validity_logits,
        prediction.view_spatial_offsets_xy,
        batch["target_gt_projected_offset_xy"],
        batch["target_gt_projection_evaluable"],
        batch["target_gt_projection_physical_valid"],
        target_sigma_px=float(spatial_target_sigma_px),
    )
    spatial_density_supervision = (
        spatial_density_evaluable & natural_groups[pair_groups]
    )
    spatial_density_loss = _weighted_selected_mean(
        spatial_density_per_view_nll,
        spatial_density_supervision,
        measurement_weight,
    )
    (
        pose_view_mixture_per_candidate_nll,
        pose_view_mixture_non_dustbin,
        pose_view_mixture_evaluable,
    ) = gt_pose_candidate_view_mixture_nll(
        prediction.view_spatial_logits,
        prediction.view_measurement_validity_logits,
        prediction.view_pose_mixture_probabilities,
        prediction.view_spatial_offsets_xy,
        batch["candidate_target_gt_projected_offset_xy"],
        batch["candidate_target_gt_projection_evaluable"],
        batch["candidate_target_gt_projection_physical_valid"],
        pair_group_indices=batch["pair_group_indices"],
        pair_candidate_indices=batch["pair_candidate_indices"],
        pair_view_slots=batch["pair_view_slots"],
        max_views=int(prediction.candidate_pose_view_probabilities.shape[2]),
        target_sigma_px=float(spatial_target_sigma_px),
    )
    pose_view_mixture_supervision = (
        pose_view_mixture_evaluable & natural_groups[:, None]
    )
    pose_view_mixture_loss = _weighted_selected_mean(
        pose_view_mixture_per_candidate_nll,
        pose_view_mixture_supervision,
        batch["candidate_measurement_supervision_weight"].to(
            device=device, dtype=torch.float32
        ),
    )
    loss = (
        float(identity_loss_weight) * identity_loss
        + float(availability_loss_weight) * availability_loss
        + float(pair_loss_weight) * pair_loss
        + float(coarse_hard_negative_loss_weight) * coarse_hard_negative_loss
        + float(spatial_loss_weight) * spatial_loss
        + float(measurement_validity_loss_weight) * measurement_validity_loss
        + float(spatial_density_loss_weight) * spatial_density_loss
        + float(pose_view_mixture_loss_weight) * pose_view_mixture_loss
    )
    return loss, {
        "total": float(loss.detach().cpu()),
        "identity": float(identity_loss.detach().cpu()),
        "availability": float(availability_loss.detach().cpu()),
        "pair": float(pair_loss.detach().cpu()),
        "coarse_hard_negative": float(coarse_hard_negative_loss.detach().cpu()),
        "coarse_hard_negative_group_rate": float(
            torch.mean(coarse_hard_negative_groups.float()).detach().cpu()
        ),
        "spatial": float(spatial_loss.detach().cpu()),
        "measurement_validity": float(
            measurement_validity_loss.detach().cpu()
        ),
        "spatial_density": float(spatial_density_loss.detach().cpu()),
        "pose_view_mixture_density": float(
            pose_view_mixture_loss.detach().cpu()
        ),
        "pose_view_mixture_non_dustbin_rate": float(
            torch.mean(
                pose_view_mixture_non_dustbin[
                    pose_view_mixture_supervision
                ].float()
            )
            .detach()
            .cpu()
            if bool(torch.any(pose_view_mixture_supervision))
            else 0.0
        ),
        "spatial_density_non_dustbin_rate": float(
            torch.mean(
                spatial_density_non_dustbin[spatial_density_supervision].float()
            )
            .detach()
            .cpu()
            if bool(torch.any(spatial_density_supervision))
            else 0.0
        ),
        "availability_positive_group_rate": float(
            torch.mean(q_target[natural_groups]).detach().cpu()
        ),
        "appearance_positive_group_rate": float(
            torch.mean(positive_groups.float()).detach().cpu()
        ),
        "positive_pair_rate": float(
            torch.mean(pair_target[pair_supervised]).detach().cpu()
            if bool(torch.any(pair_supervised))
            else 0.0
        ),
        "measurement_success_rate": float(
            torch.mean(measurement_success[measurement_supervision].float())
            .detach()
            .cpu()
            if bool(torch.any(measurement_supervision))
            else 0.0
        ),
        "measurement_success_residual_median_px": float(
            torch.median(measurement_residual[measurement_success]).detach().cpu()
            if bool(torch.any(measurement_success))
            else 0.0
        ),
    }


def _identity_metrics(
    scores: np.ndarray,
    *,
    labels: np.ndarray,
    valid: np.ndarray,
    prior: np.ndarray,
) -> dict[str, object]:
    masked = np.where(valid, scores, -1e9)
    shifted = masked - np.max(masked, axis=1, keepdims=True)
    probability = np.exp(shifted) * valid
    probability /= np.sum(probability, axis=1, keepdims=True).clip(min=1e-30)
    any_positive = np.any(labels, axis=1)
    positive_mass = np.sum(probability * labels, axis=1)
    predicted = np.argmax(masked, axis=1)
    predicted_correct = labels[np.arange(len(labels)), predicted]
    prior_predicted = np.argmax(np.where(valid, prior, -np.inf), axis=1)
    prior_correct = labels[np.arange(len(labels)), prior_predicted]
    rescue = any_positive & ~prior_correct
    preserve = any_positive & prior_correct
    return {
        "conditional_identity_nll": (
            float(np.mean(-np.log(positive_mass[any_positive].clip(min=1e-12))))
            if np.any(any_positive)
            else float("nan")
        ),
        "conditional_top1_correct_rate": (
            float(np.mean(predicted_correct[any_positive]))
            if np.any(any_positive)
            else float("nan")
        ),
        "pair_average_precision": _average_precision(labels[valid], scores[valid]),
        "rank2_to_5_rescue_rate": (
            float(np.mean(predicted_correct[rescue])) if np.any(rescue) else float("nan")
        ),
        "wrong_switch_rate": (
            float(np.mean(~predicted_correct[preserve]))
            if np.any(preserve)
            else float("nan")
        ),
        "positive_group_count": int(np.count_nonzero(any_positive)),
        "group_count": int(len(labels)),
    }


def _evaluate(
    model: IndependentRGBCandidateVerifier,
    data: CandidateRGBTrainingData,
    *,
    split_name: str,
    image_root: Path,
    image_width: int,
    image_height: int,
    image_cache: TensorImageLRUCache,
    image_cache_device: torch.device | None,
    image_crop_device: torch.device | None,
    device: torch.device,
    batch_size: int,
    identity_threshold_px: float,
    identity_negative_threshold_px: float,
    measurement_success_threshold_px: float,
    spatial_density_target_sigma_px: float,
    use_amp: bool,
    collect_spatial_diagnostics: bool,
    identity_context_residual_scale: float = 1.0,
    identity_context_layout_residual_scale: float = 0.0,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    indices = data.indices_by_split[str(split_name)]
    candidate_count = int(data.candidate_valid.shape[1])
    llr = np.zeros((len(indices), candidate_count), dtype=np.float32)
    measured = np.zeros((len(indices), candidate_count), dtype=bool)
    coverage = np.zeros((len(indices), candidate_count), dtype=np.float32)
    q_logit = np.zeros((len(indices),), dtype=np.float32)
    q_available = data.has_any_rgb[indices].copy()
    measurable_positions = np.flatnonzero(q_available).astype(np.int64)
    spatial_nll: list[np.ndarray] = []
    spatial_epe: list[np.ndarray] = []
    measurement_validity_targets: list[np.ndarray] = []
    measurement_validity_probabilities: list[np.ndarray] = []
    measurement_mode_residuals: list[np.ndarray] = []
    spatial_density_nll_values: list[np.ndarray] = []
    spatial_density_non_dustbin_targets: list[np.ndarray] = []
    spatial_density_non_dustbin_probabilities: list[np.ndarray] = []
    pose_view_mixture_nll_values: list[np.ndarray] = []
    uniform_pose_view_mixture_nll_values: list[np.ndarray] = []
    pose_view_mixture_entropy_values: list[np.ndarray] = []
    pose_view_mixture_view_counts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(measurable_positions), int(batch_size)):
            positions = measurable_positions[start : start + int(batch_size)]
            batch_indices = indices[positions].tolist()
            batch = _prepare_batch(
                data,
                batch_indices,
                image_root=Path(image_root),
                image_width=int(image_width),
                image_height=int(image_height),
                image_cache=image_cache,
                image_cache_device=image_cache_device,
                image_crop_device=image_crop_device,
                crop_radius_px=model.crop_radius_px,
                step_px=model.step_px,
                identity_context_crop_radius_px=(
                    model.identity_context_crop_radius_px
                    if (
                        float(identity_context_residual_scale) > 0.0
                        or float(identity_context_layout_residual_scale) > 0.0
                    )
                    else None
                ),
                identity_context_step_px=(
                    model.identity_context_step_px
                    if (
                        float(identity_context_residual_scale) > 0.0
                        or float(identity_context_layout_residual_scale) > 0.0
                    )
                    else None
                ),
                identity_threshold_px=float(identity_threshold_px),
                identity_negative_threshold_px=float(
                    identity_negative_threshold_px
                ),
                spatial_radius_px=model.search_radius_px,
            )
            prediction = _forward_batch(
                model,
                batch,
                device=device,
                use_amp=bool(use_amp),
                identity_context_residual_scale=float(
                    identity_context_residual_scale
                ),
                identity_context_layout_residual_scale=float(
                    identity_context_layout_residual_scale
                ),
            )
            llr[positions] = prediction.candidate_log_likelihood_ratios.cpu().numpy()
            measured[positions] = prediction.candidate_measured.cpu().numpy()
            coverage[positions] = prediction.candidate_view_coverage.cpu().numpy()
            q_logit[positions] = prediction.measured_set_availability_logit.cpu().numpy()
            if not collect_spatial_diagnostics:
                continue
            spatial_valid = batch["spatial_valid"].to(device=device, dtype=torch.bool)
            if bool(torch.any(spatial_valid)):
                logits = prediction.view_spatial_logits[spatial_valid]
                targets = batch["spatial_target_xy"].to(device=device)[spatial_valid]
                offsets = prediction.view_spatial_offsets_xy.to(device=device)
                nearest = torch.argmin(
                    torch.sum((offsets[None] - targets[:, None]) ** 2, dim=2), dim=1
                )
                log_probs = F.log_softmax(logits, dim=1)
                spatial_nll.append(
                    (-log_probs[torch.arange(len(nearest), device=device), nearest])
                    .cpu()
                    .numpy()
                )
                mode = offsets[torch.argmax(logits, dim=1)]
                spatial_epe.append(torch.linalg.norm(mode - targets, dim=1).cpu().numpy())
            measurement_residual, measurement_success = (
                measurement_mode_residual_and_success(
                    prediction.view_spatial_logits,
                    prediction.view_spatial_offsets_xy,
                    batch["target_gt_projected_offset_xy"],
                    batch["target_gt_projection_physical_valid"],
                    success_threshold_px=float(measurement_success_threshold_px),
                )
            )
            measurement_supervision = batch[
                "target_gt_projection_evaluable"
            ].to(device=device, dtype=torch.bool)
            if bool(torch.any(measurement_supervision)):
                measurement_validity_targets.append(
                    measurement_success[measurement_supervision].cpu().numpy()
                )
                measurement_validity_probabilities.append(
                    prediction.view_measurement_validity_probabilities[
                        measurement_supervision
                    ]
                    .cpu()
                    .numpy()
                )
                measurement_mode_residuals.append(
                    measurement_residual[measurement_supervision].cpu().numpy()
                )
            density_nll, density_non_dustbin, density_evaluable = (
                gt_pose_spatial_density_nll(
                    prediction.view_spatial_logits,
                    prediction.view_measurement_validity_logits,
                    prediction.view_spatial_offsets_xy,
                    batch["target_gt_projected_offset_xy"],
                    batch["target_gt_projection_evaluable"],
                    batch["target_gt_projection_physical_valid"],
                    target_sigma_px=float(spatial_density_target_sigma_px),
                )
            )
            if bool(torch.any(density_evaluable)):
                spatial_density_nll_values.append(
                    density_nll[density_evaluable].cpu().numpy()
                )
                spatial_density_non_dustbin_targets.append(
                    density_non_dustbin[density_evaluable].cpu().numpy()
                )
                spatial_density_non_dustbin_probabilities.append(
                    prediction.view_measurement_validity_probabilities[
                        density_evaluable
                    ]
                    .cpu()
                    .numpy()
                )
            learned_candidate_nll, _, learned_candidate_evaluable = (
                gt_pose_candidate_view_mixture_nll(
                    prediction.view_spatial_logits,
                    prediction.view_measurement_validity_logits,
                    prediction.view_pose_mixture_probabilities,
                    prediction.view_spatial_offsets_xy,
                    batch["candidate_target_gt_projected_offset_xy"],
                    batch["candidate_target_gt_projection_evaluable"],
                    batch[
                        "candidate_target_gt_projection_physical_valid"
                    ],
                    pair_group_indices=batch["pair_group_indices"],
                    pair_candidate_indices=batch["pair_candidate_indices"],
                    pair_view_slots=batch["pair_view_slots"],
                    max_views=int(model.max_views),
                    target_sigma_px=float(spatial_density_target_sigma_px),
                )
            )
            uniform_pair_probabilities, uniform_candidate_probabilities = (
                normalize_pose_view_mixture_logits(
                    torch.zeros_like(prediction.view_pose_mixture_logits),
                    pair_group_indices=batch["pair_group_indices"],
                    pair_candidate_indices=batch["pair_candidate_indices"],
                    pair_view_slots=batch["pair_view_slots"],
                    batch_size=int(
                        prediction.candidate_pose_view_probabilities.shape[0]
                    ),
                    candidate_count=int(
                        prediction.candidate_pose_view_probabilities.shape[1]
                    ),
                    max_views=int(model.max_views),
                )
            )
            uniform_candidate_nll, _, uniform_candidate_evaluable = (
                gt_pose_candidate_view_mixture_nll(
                    prediction.view_spatial_logits,
                    prediction.view_measurement_validity_logits,
                    uniform_pair_probabilities,
                    prediction.view_spatial_offsets_xy,
                    batch["candidate_target_gt_projected_offset_xy"],
                    batch["candidate_target_gt_projection_evaluable"],
                    batch[
                        "candidate_target_gt_projection_physical_valid"
                    ],
                    pair_group_indices=batch["pair_group_indices"],
                    pair_candidate_indices=batch["pair_candidate_indices"],
                    pair_view_slots=batch["pair_view_slots"],
                    max_views=int(model.max_views),
                    target_sigma_px=float(spatial_density_target_sigma_px),
                )
            )
            if not torch.equal(
                learned_candidate_evaluable, uniform_candidate_evaluable
            ):
                raise RuntimeError(
                    "learned and uniform pose-view mixtures cover different candidates"
                )
            if bool(torch.any(learned_candidate_evaluable)):
                pose_view_mixture_nll_values.append(
                    learned_candidate_nll[learned_candidate_evaluable]
                    .cpu()
                    .numpy()
                )
                uniform_pose_view_mixture_nll_values.append(
                    uniform_candidate_nll[uniform_candidate_evaluable]
                    .cpu()
                    .numpy()
                )
                learned_probabilities = (
                    prediction.candidate_pose_view_probabilities
                )
                entropy = -torch.sum(
                    torch.where(
                        learned_probabilities > 0.0,
                        learned_probabilities
                        * torch.log(learned_probabilities.clamp_min(1e-30)),
                        torch.zeros_like(learned_probabilities),
                    ),
                    dim=2,
                )
                pose_view_mixture_entropy_values.append(
                    entropy[learned_candidate_evaluable].cpu().numpy()
                )
                pose_view_mixture_view_counts.append(
                    torch.sum(
                        uniform_candidate_probabilities > 0.0, dim=2
                    )[learned_candidate_evaluable]
                    .cpu()
                    .numpy()
                )

    valid = data.candidate_valid[indices]
    geometric_labels = valid & np.isfinite(data.residuals[indices]) & (
        data.residuals[indices] <= float(identity_threshold_px)
    )
    appearance_labels = (
        valid
        & data.actual_query_observation[indices]
        & np.isfinite(data.actual_center_residuals[indices])
        & (data.actual_center_residuals[indices] <= float(identity_threshold_px))
    )
    appearance_supervision_valid = valid & (
        appearance_labels
        | (
            np.isfinite(data.residuals[indices])
            & (data.residuals[indices] >= float(identity_negative_threshold_px))
        )
    )
    prior = data.prior[indices]
    unknown = data.unknown[indices]
    availability_labels = data.availability_valid[indices] & np.isfinite(
        data.availability_residuals[indices]
    ) & (data.availability_residuals[indices] <= float(identity_threshold_px))
    any_available = np.any(availability_labels, axis=1)
    q_probability = 1.0 / (1.0 + np.exp(-np.clip(q_logit, -30.0, 30.0)))
    measurement_target = (
        np.concatenate(measurement_validity_targets).astype(bool)
        if measurement_validity_targets
        else np.zeros((0,), dtype=bool)
    )
    measurement_probability = (
        np.concatenate(measurement_validity_probabilities).astype(np.float64)
        if measurement_validity_probabilities
        else np.zeros((0,), dtype=np.float64)
    )
    measurement_residual = (
        np.concatenate(measurement_mode_residuals).astype(np.float64)
        if measurement_mode_residuals
        else np.zeros((0,), dtype=np.float64)
    )
    finite_measurement_residual = measurement_residual[
        np.isfinite(measurement_residual)
    ]
    density_nll_values = (
        np.concatenate(spatial_density_nll_values).astype(np.float64)
        if spatial_density_nll_values
        else np.zeros((0,), dtype=np.float64)
    )
    density_non_dustbin_target = (
        np.concatenate(spatial_density_non_dustbin_targets).astype(bool)
        if spatial_density_non_dustbin_targets
        else np.zeros((0,), dtype=bool)
    )
    density_non_dustbin_probability = (
        np.concatenate(spatial_density_non_dustbin_probabilities).astype(
            np.float64
        )
        if spatial_density_non_dustbin_probabilities
        else np.zeros((0,), dtype=np.float64)
    )
    candidate_mixture_nll = (
        np.concatenate(pose_view_mixture_nll_values).astype(np.float64)
        if pose_view_mixture_nll_values
        else np.zeros((0,), dtype=np.float64)
    )
    uniform_candidate_mixture_nll = (
        np.concatenate(uniform_pose_view_mixture_nll_values).astype(
            np.float64
        )
        if uniform_pose_view_mixture_nll_values
        else np.zeros((0,), dtype=np.float64)
    )
    candidate_mixture_entropy = (
        np.concatenate(pose_view_mixture_entropy_values).astype(np.float64)
        if pose_view_mixture_entropy_values
        else np.zeros((0,), dtype=np.float64)
    )
    candidate_mixture_view_count = (
        np.concatenate(pose_view_mixture_view_counts).astype(np.int64)
        if pose_view_mixture_view_counts
        else np.zeros((0,), dtype=np.int64)
    )
    mode_success_metric_name = (
        "measurement_geometric_validity_TARGET_ONLY"
        if model.measurement_validity_semantics
        == MEASUREMENT_MODE_SUCCESS_SEMANTICS
        else "legacy_mode_success_correlation_DIAGNOSTIC_ONLY"
    )
    metrics: dict[str, object] = {
        "spatial_diagnostics_collected": bool(collect_spatial_diagnostics),
        "rgb_only": _identity_metrics(
            llr, labels=geometric_labels, valid=valid, prior=prior
        ),
        "coarse_prior": _identity_metrics(
            np.log(prior.clip(min=1e-30)),
            labels=geometric_labels,
            valid=valid,
            prior=prior,
        ),
        "geometric_rgb_only": _identity_metrics(
            llr, labels=geometric_labels, valid=valid, prior=prior
        ),
        "geometric_coarse_prior": _identity_metrics(
            np.log(prior.clip(min=1e-30)),
            labels=geometric_labels,
            valid=valid,
            prior=prior,
        ),
        "appearance_supervised_rgb_only": _identity_metrics(
            llr,
            labels=appearance_labels,
            valid=appearance_supervision_valid,
            prior=prior,
        ),
        "appearance_supervised_coarse_prior": _identity_metrics(
            np.log(prior.clip(min=1e-30)),
            labels=appearance_labels,
            valid=appearance_supervision_valid,
            prior=prior,
        ),
        "full_top_l_availability": confidence_metrics(
            any_available[q_available], q_probability[q_available]
        ),
        "full_top_l_availability_available_group_count": int(
            np.count_nonzero(q_available)
        ),
        "full_top_l_availability_unavailable_group_count": int(
            np.count_nonzero(~q_available)
        ),
        "measured_candidate_rate": float(np.mean(measured[valid])),
        "mean_view_coverage": float(np.mean(coverage[valid])),
        "spatial": {
            "sample_count": int(sum(len(value) for value in spatial_nll)),
            "nearest_grid_nll_mean": (
                float(np.mean(np.concatenate(spatial_nll))) if spatial_nll else float("nan")
            ),
            "mode_epe_median_px": (
                float(np.median(np.concatenate(spatial_epe))) if spatial_epe else float("nan")
            ),
            "mode_epe_p90_px": (
                float(np.quantile(np.concatenate(spatial_epe), 0.9))
                if spatial_epe
                else float("nan")
            ),
        },
        mode_success_metric_name: {
            **confidence_metrics(measurement_target, measurement_probability),
            "success_threshold_px": float(measurement_success_threshold_px),
            "mode_residual_median_px": (
                float(np.median(finite_measurement_residual))
                if len(finite_measurement_residual)
                else float("nan")
            ),
            "mode_residual_p90_px": (
                float(np.quantile(finite_measurement_residual, 0.9))
                if len(finite_measurement_residual)
                else float("nan")
            ),
            "physically_invalid_projection_count": int(
                np.count_nonzero(~np.isfinite(measurement_residual))
            ),
            "probability_semantics": (
                "P(predicted_spatial_mode_gt_pose_projection_residual_le_threshold)"
            ),
        },
        "gt_pose_spatial_density_TARGET_ONLY": {
            "sample_count": int(len(density_nll_values)),
            "joint_k_plus_dustbin_nll_mean": (
                float(np.mean(density_nll_values))
                if len(density_nll_values)
                else float("nan")
            ),
            "joint_k_plus_dustbin_nll_median": (
                float(np.median(density_nll_values))
                if len(density_nll_values)
                else float("nan")
            ),
            "non_dustbin_calibration": (
                confidence_metrics(
                    density_non_dustbin_target,
                    density_non_dustbin_probability,
                )
                if len(density_non_dustbin_target)
                else {}
            ),
            "target_semantics": (
                "gt_pose_projection_inside_local_support_vs_dustbin"
            ),
        },
        "candidate_pose_view_mixture_TARGET_ONLY": {
            "enabled": bool(model.pose_view_mixture_enabled),
            "semantics": (
                model.config()["pose_view_mixture_semantics"]
            ),
            "sample_count": int(len(candidate_mixture_nll)),
            "joint_k_plus_dustbin_nll_mean": (
                float(np.mean(candidate_mixture_nll))
                if len(candidate_mixture_nll)
                else float("nan")
            ),
            "uniform_all_views_nll_mean": (
                float(np.mean(uniform_candidate_mixture_nll))
                if len(uniform_candidate_mixture_nll)
                else float("nan")
            ),
            "nll_delta_vs_uniform_mean": (
                float(
                    np.mean(
                        candidate_mixture_nll
                        - uniform_candidate_mixture_nll
                    )
                )
                if len(candidate_mixture_nll)
                else float("nan")
            ),
            "view_entropy_mean": (
                float(np.mean(candidate_mixture_entropy))
                if len(candidate_mixture_entropy)
                else float("nan")
            ),
            "view_count_mean": (
                float(np.mean(candidate_mixture_view_count))
                if len(candidate_mixture_view_count)
                else float("nan")
            ),
            "multi_view_candidate_rate": (
                float(np.mean(candidate_mixture_view_count >= 2))
                if len(candidate_mixture_view_count)
                else float("nan")
            ),
        },
        "fusion_weight_sweep": {},
        "appearance_fusion_weight_sweep": {},
    }
    prior_tensor = torch.from_numpy(prior)
    unknown_tensor = torch.from_numpy(unknown)
    llr_tensor = torch.from_numpy(llr)
    measured_tensor = torch.from_numpy(measured)
    valid_tensor = torch.from_numpy(valid)
    for weight in (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0):
        fused, _ = fuse_candidate_log_likelihood_ratios(
            prior_tensor,
            unknown_tensor,
            llr_tensor,
            measured_mask=measured_tensor,
            candidate_valid=valid_tensor,
            evidence_weight=float(weight),
        )
        metrics["fusion_weight_sweep"][f"{weight:g}"] = _identity_metrics(
            np.log(fused.numpy().clip(min=1e-30)),
            labels=geometric_labels,
            valid=valid,
            prior=prior,
        )
        metrics["appearance_fusion_weight_sweep"][f"{weight:g}"] = (
            _identity_metrics(
                np.log(fused.numpy().clip(min=1e-30)),
                labels=appearance_labels,
                valid=appearance_supervision_valid,
                prior=prior,
            )
        )
    predictions = {
        "evidence_row_indices": indices,
        "candidate_rgb_log_likelihood_ratios": llr.astype(np.float32),
        "candidate_rgb_measured": measured.astype(bool),
        "candidate_rgb_view_coverage": coverage.astype(np.float32),
        "rgb_full_top_l_availability_available": q_available.astype(bool),
        "rgb_full_top_l_availability_logit": q_logit.astype(np.float32),
        "rgb_full_top_l_availability_probability": q_probability.astype(np.float32),
    }
    return metrics, predictions


def train_independent_rgb_candidate_verifier(
    *,
    candidate_evidence: Path,
    availability_evidence: Path,
    train_rows_csv: Path,
    validation_rows_csv: Path,
    test_rows_csv: Path,
    image_root: Path,
    init_measurement_checkpoint: Path,
    init_independent_checkpoint: Path | None = None,
    search_radius_px_override: float | None = None,
    context_radius_px_override: float | None = None,
    step_px_override: float | None = None,
    identity_context_radius_px: float | None = None,
    identity_context_step_px: float | None = None,
    identity_context_layout_grid_size: int | None = None,
    prediction_identity_context_residual_scale: float = 1.0,
    prediction_identity_context_layout_residual_scale: float | None = None,
    pair_forward_batch_size: int = 0,
    pair_forward_gradient_checkpointing: bool = False,
    freeze_for_measurement_validity: bool = False,
    freeze_for_spatial_density: bool = False,
    freeze_for_pose_view_mixture: bool = False,
    freeze_for_identity_context_residual: bool = False,
    freeze_for_identity_context_layout_residual: bool = False,
    output_dir: Path,
    image_width: int,
    image_height: int,
    steps: int = 500,
    group_batch_size: int = 32,
    query_images_per_batch: int = 8,
    appearance_positive_group_fraction: float = 0.5,
    eval_group_batch_size: int = 32,
    encoder_learning_rate: float = 1e-4,
    head_learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    identity_threshold_px: float = 2.0,
    identity_negative_threshold_px: float = 5.0,
    identity_target: str | None = None,
    identity_loss_weight: float = 1.0,
    availability_loss_weight: float = 0.5,
    pair_loss_weight: float = 0.25,
    coarse_hard_negative_loss_weight: float = 0.0,
    coarse_hard_negative_margin: float = 0.0,
    coarse_hard_negative_prior_power: float = 1.0,
    coarse_hard_negative_evidence_weight: float = 1.0,
    spatial_loss_weight: float = 0.25,
    spatial_target_sigma_px: float = 0.75,
    measurement_validity_loss_weight: float = 0.5,
    measurement_success_threshold_px: float = 2.0,
    spatial_density_loss_weight: float = 0.0,
    pose_view_mixture_loss_weight: float = 0.0,
    max_views: int = 4,
    seed: int = 0,
    device: str = "cuda",
    image_cache_max_gb: float = 12.0,
    gpu_non_cache_reserve_gb: float = 14.0,
    image_cache_dtype: str = "float16",
    image_cache_location: str = "gpu",
    use_amp: bool = True,
    log_every: int = 25,
    prediction_splits: Sequence[str] = ("validation", "test"),
    collect_spatial_diagnostics: bool | None = None,
) -> dict[str, Any]:
    prediction_context_scale = float(prediction_identity_context_residual_scale)
    if not math.isfinite(prediction_context_scale) or not (
        0.0 <= prediction_context_scale <= 1.0
    ):
        raise ValueError(
            "prediction_identity_context_residual_scale must be finite and in [0, 1]"
        )
    if prediction_context_scale != 1.0 and int(steps) != 0:
        raise ValueError(
            "prediction_identity_context_residual_scale is evaluation-only and requires steps=0"
        )
    requested_layout_prediction_scale = (
        None
        if prediction_identity_context_layout_residual_scale is None
        else float(prediction_identity_context_layout_residual_scale)
    )
    if requested_layout_prediction_scale is not None and (
        not math.isfinite(requested_layout_prediction_scale)
        or not 0.0 <= requested_layout_prediction_scale <= 1.0
    ):
        raise ValueError(
            "prediction_identity_context_layout_residual_scale must be finite and in [0, 1]"
        )
    if float(measurement_validity_loss_weight) < 0.0:
        raise ValueError("measurement_validity_loss_weight must be non-negative")
    if not math.isfinite(float(coarse_hard_negative_loss_weight)) or float(
        coarse_hard_negative_loss_weight
    ) < 0.0:
        raise ValueError("coarse_hard_negative_loss_weight must be finite and non-negative")
    if not math.isfinite(float(coarse_hard_negative_margin)) or float(
        coarse_hard_negative_margin
    ) < 0.0:
        raise ValueError("coarse_hard_negative_margin must be finite and non-negative")
    if not math.isfinite(float(coarse_hard_negative_prior_power)) or float(
        coarse_hard_negative_prior_power
    ) <= 0.0:
        raise ValueError("coarse_hard_negative_prior_power must be finite and positive")
    if not math.isfinite(float(coarse_hard_negative_evidence_weight)) or float(
        coarse_hard_negative_evidence_weight
    ) <= 0.0:
        raise ValueError(
            "coarse_hard_negative_evidence_weight must be finite and positive"
        )
    if float(spatial_density_loss_weight) < 0.0:
        raise ValueError("spatial_density_loss_weight must be non-negative")
    if float(pose_view_mixture_loss_weight) < 0.0:
        raise ValueError("pose_view_mixture_loss_weight must be non-negative")
    if float(measurement_success_threshold_px) <= 0.0:
        raise ValueError("measurement_success_threshold_px must be positive")
    spatial_diagnostics_enabled = (
        bool(collect_spatial_diagnostics)
        if collect_spatial_diagnostics is not None
        else any(
            float(weight) > 0.0
            for weight in (
                spatial_loss_weight,
                measurement_validity_loss_weight,
                spatial_density_loss_weight,
                pose_view_mixture_loss_weight,
            )
        )
    )
    if sum(
        bool(value)
        for value in (
            freeze_for_measurement_validity,
            freeze_for_spatial_density,
            freeze_for_pose_view_mixture,
            freeze_for_identity_context_residual,
            freeze_for_identity_context_layout_residual,
        )
    ) > 1:
        raise ValueError(
            "measurement-validity, spatial-density, pose-view, pooled-context, and layout-context "
            "freeze modes are exclusive"
        )
    if bool(freeze_for_identity_context_residual) or bool(
        freeze_for_identity_context_layout_residual
    ):
        non_identity_weights = {
            "availability_loss_weight": availability_loss_weight,
            "spatial_loss_weight": spatial_loss_weight,
            "measurement_validity_loss_weight": measurement_validity_loss_weight,
            "spatial_density_loss_weight": spatial_density_loss_weight,
            "pose_view_mixture_loss_weight": pose_view_mixture_loss_weight,
        }
        enabled_non_identity = [
            name for name, weight in non_identity_weights.items() if float(weight) != 0.0
        ]
        if enabled_non_identity:
            raise ValueError(
                "frozen identity-context residual training permits only identity and "
                "pair losses; set these weights to zero: "
                f"{enabled_non_identity}"
            )
        if (
            float(identity_loss_weight) <= 0.0
            and float(pair_loss_weight) <= 0.0
            and float(coarse_hard_negative_loss_weight) <= 0.0
        ):
            raise ValueError(
                "frozen identity-context residual training requires a positive "
                "identity_loss_weight, pair_loss_weight, or "
                "coarse_hard_negative_loss_weight"
            )
    torch_device = torch.device(
        device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu"
    )
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.benchmark = True
    data = CandidateRGBTrainingData(
        candidate_evidence=Path(candidate_evidence),
        availability_evidence=Path(availability_evidence),
        train_rows_csv=Path(train_rows_csv),
        validation_rows_csv=Path(validation_rows_csv),
        test_rows_csv=Path(test_rows_csv),
        max_views=int(max_views),
    )
    runtime_data_contract = data.runtime_contract(
        image_root=Path(image_root),
        image_width=int(image_width),
        image_height=int(image_height),
    )
    initialization_path = (
        Path(init_measurement_checkpoint)
        if init_independent_checkpoint is None
        else Path(init_independent_checkpoint)
    )
    initial_payload = torch.load(initialization_path, map_location="cpu")
    initial_config = dict(initial_payload.get("config", {}))
    initialization_provenance = _initialization_provenance(
        initialization_path, initial_payload
    )
    resolved_identity_target = _resolve_identity_target(
        identity_target,
        initialization_training=dict(
            initialization_provenance["initialization_training"]
        ),
    )
    if (
        float(coarse_hard_negative_loss_weight) > 0.0
        and resolved_identity_target != IDENTITY_TARGET_GEOMETRIC
    ):
        raise ValueError(
            "coarse hard-negative ranking requires identity_target=geometric_projection"
        )
    spatial_support, spatial_support_overrides = _resolve_spatial_support_config(
        initial_config,
        search_radius_px_override=search_radius_px_override,
        context_radius_px_override=context_radius_px_override,
        step_px_override=step_px_override,
    )
    identity_context, identity_context_overrides = _resolve_identity_context_config(
        initial_config,
        identity_context_radius_px=identity_context_radius_px,
        identity_context_step_px=identity_context_step_px,
    )
    identity_context_layout, identity_context_layout_overrides = (
        _resolve_identity_context_layout_config(
            initial_config,
            identity_context_layout_grid_size=identity_context_layout_grid_size,
        )
    )
    if bool(identity_context_layout["enabled"]) and not bool(
        identity_context["enabled"]
    ):
        raise ValueError("identity-context layout requires an enabled broad context branch")
    default_layout_prediction_scale = (
        1.0 if bool(identity_context_layout["enabled"]) else 0.0
    )
    prediction_layout_scale = (
        default_layout_prediction_scale
        if requested_layout_prediction_scale is None
        else requested_layout_prediction_scale
    )
    if prediction_layout_scale != default_layout_prediction_scale and int(steps) != 0:
        raise ValueError(
            "prediction_identity_context_layout_residual_scale is evaluation-only and requires steps=0"
        )
    if spatial_support_overrides and init_independent_checkpoint is not None:
        raise ValueError(
            "a spatial-support override must initialize from the measurement "
            "encoder, not a fully calibrated independent RGB checkpoint"
        )
    if spatial_support_overrides and any(
        (
            freeze_for_measurement_validity,
            freeze_for_spatial_density,
            freeze_for_pose_view_mixture,
            freeze_for_identity_context_residual,
            freeze_for_identity_context_layout_residual,
        )
    ):
        raise ValueError(
            "a spatial-support override requires joint retraining; frozen-head "
            "modes would reuse a likelihood calibrated for a different support"
        )
    pose_view_mixture_enabled = bool(
        initial_config.get("pose_view_mixture_enabled", False)
        or freeze_for_pose_view_mixture
        or float(pose_view_mixture_loss_weight) > 0.0
    )
    required_config = {
        "search_radius_px",
        "context_radius_px",
        "step_px",
        "feature_dim",
        "hidden_dim",
        "input_mode",
        "encoder_arch",
        "template_scale_factors",
    }
    missing_config = required_config - set(initial_config)
    if missing_config:
        raise ValueError(
            f"measurement checkpoint lacks RGB encoder config: {sorted(missing_config)}"
        )
    model = IndependentRGBCandidateVerifier(
        search_radius_px=float(spatial_support["search_radius_px"]),
        context_radius_px=float(spatial_support["context_radius_px"]),
        step_px=float(spatial_support["step_px"]),
        feature_dim=int(initial_config["feature_dim"]),
        hidden_dim=int(initial_config["hidden_dim"]),
        input_mode=str(initial_config["input_mode"]),
        encoder_arch=str(initial_config["encoder_arch"]),
        template_scale_factors=tuple(initial_config["template_scale_factors"]),
        max_views=int(max_views),
        measurement_validity_semantics=(
            GT_POSE_SPATIAL_DENSITY_SEMANTICS
            if float(spatial_density_loss_weight) > 0.0
            else str(
                initial_config.get(
                    "measurement_validity_semantics",
                    MEASUREMENT_MODE_SUCCESS_SEMANTICS,
                )
            )
        ),
        pose_view_mixture_enabled=pose_view_mixture_enabled,
        pair_forward_batch_size=int(pair_forward_batch_size),
        pair_forward_gradient_checkpointing=bool(
            pair_forward_gradient_checkpointing
        ),
        identity_context_radius_px=(
            None
            if not bool(identity_context["enabled"])
            else float(identity_context["radius_px"])
        ),
        identity_context_step_px=(
            None
            if not bool(identity_context["enabled"])
            else float(identity_context["step_px"])
        ),
        identity_context_layout_grid_size=(
            None
            if not bool(identity_context_layout["enabled"])
            else int(identity_context_layout["grid_size"])
        ),
    )
    initialized_from_independent = init_independent_checkpoint is not None
    if initialized_from_independent:
        checkpoint_format = str(initial_payload.get("format", ""))
        if checkpoint_format not in {
            "independent_rgb_candidate_verifier_v2",
            "independent_rgb_candidate_verifier_v3",
            "independent_rgb_candidate_verifier_v4",
            "independent_rgb_candidate_verifier_v5",
            "independent_rgb_candidate_verifier_v6",
            "independent_rgb_candidate_verifier_v8",
        }:
            raise ValueError(
                "independent initialization requires verifier v2 through v6 or v8"
            )
        if checkpoint_format in {
            "independent_rgb_candidate_verifier_v3",
            "independent_rgb_candidate_verifier_v4",
            "independent_rgb_candidate_verifier_v5",
            "independent_rgb_candidate_verifier_v6",
            "independent_rgb_candidate_verifier_v8",
        }:
            initial_data_contract = dict(initial_payload.get("data_contract", {}))
            if not initial_data_contract:
                raise ValueError("verifier v3 initialization lacks its RGB data contract")
            require_compatible_contracts(
                initial_data_contract,
                runtime_data_contract,
                context="independent verifier initialization",
            )
        incompatible = model.load_state_dict(initial_payload["model"], strict=False)
        allowed_missing = {
            "view_measurement_validity_head.weight",
            "view_measurement_validity_head.bias",
            "view_pose_mixture_head.weight",
            "view_pose_mixture_head.bias",
        }
        if model.identity_context_enabled and checkpoint_format not in {
            "independent_rgb_candidate_verifier_v6",
            "independent_rgb_candidate_verifier_v8",
        }:
            allowed_missing.update(
                {
                    key
                    for key in model.state_dict()
                    if key.startswith("identity_context_")
                }
            )
        if model.identity_context_layout_enabled and checkpoint_format != (
            "independent_rgb_candidate_verifier_v8"
        ):
            allowed_missing.update(
                {
                    key
                    for key in model.state_dict()
                    if key.startswith("identity_context_layout_")
                }
            )
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise ValueError("independent verifier initialization is incompatible")
        if checkpoint_format in {
            "independent_rgb_candidate_verifier_v6",
            "independent_rgb_candidate_verifier_v8",
        }:
            if identity_context_overrides:
                raise ValueError(
                    "a dual-scale verifier checkpoint must retain its declared identity context config"
                )
            if bool(model.identity_context_enabled) != bool(
                initial_config.get("identity_context_enabled", False)
            ):
                raise ValueError("dual-scale verifier identity context is incompatible")
            if checkpoint_format == "independent_rgb_candidate_verifier_v8" and (
                identity_context_layout_overrides
            ):
                raise ValueError(
                    "a layout verifier checkpoint must retain its declared layout config"
                )
            if checkpoint_format == "independent_rgb_candidate_verifier_v8" and (
                bool(model.identity_context_layout_enabled)
                != bool(initial_config.get("identity_context_layout_enabled", False))
            ):
                raise ValueError("layout verifier identity context is incompatible")
            if set(incompatible.missing_keys) - allowed_missing:
                raise ValueError("verifier v6 lacks its dual-scale identity branch")
        elif model.identity_context_enabled:
            model.initialize_identity_context_from_local_encoder()
        if (
            model.identity_context_layout_enabled
            and checkpoint_format != "independent_rgb_candidate_verifier_v8"
        ):
            model.initialize_identity_context_layout_from_pooled_encoder()
        if checkpoint_format == "independent_rgb_candidate_verifier_v5":
            missing_non_context = {
                key
                for key in incompatible.missing_keys
                if not key.startswith("identity_context_")
            }
            if missing_non_context:
                raise ValueError("verifier v5 lacks its pose-view mixture head")
            if initial_config.get("pose_view_mixture_semantics") != (
                POSE_VIEW_MIXTURE_SEMANTICS
            ):
                raise ValueError("verifier v5 pose-view mixture contract is incompatible")
    else:
        model.initialize_encoder_from_measurement_checkpoint(
            Path(init_measurement_checkpoint)
        )
    natural_view_positive_prior = data.view_identity_positive_prior(
        split_name="train",
        positive_threshold_px=float(identity_threshold_px),
        negative_threshold_px=float(identity_negative_threshold_px),
        identity_target=resolved_identity_target,
    )
    natural_group_positive_prior = data.availability_positive_prior(
        split_name="train",
        threshold_px=float(identity_threshold_px),
        require_any_rgb=True,
    )
    if not initialized_from_independent:
        model.set_view_identity_prior(
            natural_view_positive_prior, initialize_head_bias=True
        )
        model.set_measured_set_availability_prior(
            natural_group_positive_prior, initialize_head_bias=True
        )
    model.to(torch_device)
    if bool(freeze_for_measurement_validity):
        if not initialized_from_independent:
            raise ValueError(
                "frozen measurement-validity training requires an independent checkpoint"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.view_measurement_validity_head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            model.view_measurement_validity_head.parameters(),
            lr=float(head_learning_rate),
            weight_decay=float(weight_decay),
        )
    elif bool(freeze_for_spatial_density):
        if not initialized_from_independent:
            raise ValueError(
                "frozen spatial-density training requires an independent checkpoint"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        trainable_modules = (
            model.encoder,
            model.view_evidence,
            model.view_measurement_validity_head,
        )
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        model.logit_scale.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.encoder.parameters(),
                    "lr": float(encoder_learning_rate),
                },
                {
                    "params": [
                        model.logit_scale,
                        *model.view_evidence.parameters(),
                        *model.view_measurement_validity_head.parameters(),
                    ],
                    "lr": float(head_learning_rate),
                },
            ],
            weight_decay=float(weight_decay),
        )
    elif bool(freeze_for_pose_view_mixture):
        if not initialized_from_independent:
            raise ValueError(
                "frozen pose-view mixture training requires an independent checkpoint"
            )
        if model.measurement_validity_semantics != (
            GT_POSE_SPATIAL_DENSITY_SEMANTICS
        ):
            raise ValueError(
                "pose-view mixture training requires normalized K+1 view densities"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.view_pose_mixture_head.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            model.view_pose_mixture_head.parameters(),
            lr=float(head_learning_rate),
            weight_decay=float(weight_decay),
        )
    elif bool(freeze_for_identity_context_layout_residual):
        if not initialized_from_independent:
            raise ValueError(
                "frozen identity-context layout training requires an independent checkpoint"
            )
        if (
            model.identity_context_encoder is None
            or model.identity_context_layout_encoder is None
            or model.identity_context_layout_fusion is None
            or model.identity_context_layout_evidence is None
            or model.identity_context_layout_residual is None
        ):
            raise ValueError(
                "frozen identity-context layout training requires an enabled "
                "layout branch"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        layout_encoder_parameters = list(
            model.identity_context_layout_encoder.parameters()
        )
        layout_head_parameters = [
            *model.identity_context_layout_fusion.parameters(),
            *model.identity_context_layout_evidence.parameters(),
            *model.identity_context_layout_residual.parameters(),
        ]
        for parameter in [*layout_encoder_parameters, *layout_head_parameters]:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": layout_encoder_parameters,
                    "lr": float(encoder_learning_rate),
                },
                {
                    "params": layout_head_parameters,
                    "lr": float(head_learning_rate),
                },
            ],
            weight_decay=float(weight_decay),
        )
    elif bool(freeze_for_identity_context_residual):
        if not initialized_from_independent:
            raise ValueError(
                "frozen identity-context residual training requires an independent checkpoint"
            )
        if (
            model.identity_context_encoder is None
            or model.identity_context_evidence is None
            or model.identity_context_residual is None
        ):
            raise ValueError(
                "frozen identity-context residual training requires an enabled "
                "identity-context branch"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        context_encoder_parameters = list(model.identity_context_encoder.parameters())
        context_head_parameters = [
            *model.identity_context_evidence.parameters(),
            *model.identity_context_residual.parameters(),
        ]
        for parameter in [*context_encoder_parameters, *context_head_parameters]:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": context_encoder_parameters,
                    "lr": float(encoder_learning_rate),
                },
                {
                    "params": context_head_parameters,
                    "lr": float(head_learning_rate),
                },
            ],
            weight_decay=float(weight_decay),
        )
    else:
        encoder_modules = [model.encoder]
        if model.identity_context_encoder is not None:
            encoder_modules.append(model.identity_context_encoder)
        if model.identity_context_layout_encoder is not None:
            encoder_modules.append(model.identity_context_layout_encoder)
        encoder_parameters = [
            parameter
            for module in encoder_modules
            for parameter in module.parameters()
        ]
        encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
        head_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in encoder_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": float(encoder_learning_rate)},
                {"params": head_parameters, "lr": float(head_learning_rate)},
            ],
            weight_decay=float(weight_decay),
        )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(use_amp and torch_device.type == "cuda")
    )
    requested_cache_bytes = (
        None
        if float(image_cache_max_gb) <= 0.0
        else int(float(image_cache_max_gb) * 1024**3)
    )
    if float(gpu_non_cache_reserve_gb) < 0.0:
        raise ValueError("gpu_non_cache_reserve_gb must be non-negative")
    cache_location = str(image_cache_location).strip().lower()
    if cache_location not in {"cpu", "gpu"}:
        raise ValueError("image_cache_location must be cpu or gpu")
    cache_bytes = requested_cache_bytes
    if (
        cache_location == "gpu"
        and torch_device.type == "cuda"
        and requested_cache_bytes is not None
    ):
        total_device_bytes = int(
            torch.cuda.get_device_properties(torch_device).total_memory
        )
        non_cache_reserve_bytes = int(float(gpu_non_cache_reserve_gb) * 1024**3)
        cache_bytes = min(
            requested_cache_bytes,
            max(0, total_device_bytes - non_cache_reserve_bytes),
        )
    cache_dtype_name = str(image_cache_dtype).strip().lower()
    cache_dtype_by_name = {
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if cache_dtype_name not in cache_dtype_by_name:
        raise ValueError("image_cache_dtype must be float16 or float32")
    cache_storage_dtype = (
        cache_dtype_by_name[cache_dtype_name]
        if torch_device.type == "cuda"
        else torch.float32
    )
    image_cache = TensorImageLRUCache(
        max_bytes=cache_bytes, storage_dtype=cache_storage_dtype
    )
    cache_storage_device = torch_device if cache_location == "gpu" else None
    train_indices = data.indices_by_split["train"]
    train_indices = train_indices[data.has_any_rgb[train_indices]].tolist()
    if not train_indices:
        raise ValueError("training split has no groups with RGB evidence")
    train_indices_by_query: dict[str, list[int]] = defaultdict(list)
    for index in train_indices:
        train_indices_by_query[str(data.query_ids[int(index)])].append(int(index))
    train_query_ids = sorted(train_indices_by_query)
    identity_positive, _identity_supervision = _identity_training_masks(
        valid=data.candidate_valid[train_indices],
        actual_query_observation=data.actual_query_observation[train_indices],
        actual_center_residuals=data.actual_center_residuals[train_indices],
        target_projection_residuals=data.residuals[train_indices],
        identity_target=resolved_identity_target,
        positive_threshold_px=float(identity_threshold_px),
        negative_threshold_px=float(identity_negative_threshold_px),
    )
    identity_positive_indices = [
        int(index)
        for index, positive in zip(
            train_indices, np.any(identity_positive, axis=1).tolist()
        )
        if bool(positive)
    ]
    if not identity_positive_indices:
        raise ValueError("training split has no identity-target positive candidates")
    positive_fraction = float(appearance_positive_group_fraction)
    if not 0.0 <= positive_fraction < 1.0:
        raise ValueError("appearance_positive_group_fraction must be in [0, 1)")
    positive_group_count = min(
        int(group_batch_size) - 1,
        max(0, int(round(int(group_batch_size) * positive_fraction))),
    )
    natural_group_count = int(group_batch_size) - positive_group_count
    queries_per_batch = max(
        1, min(int(query_images_per_batch), natural_group_count, len(train_query_ids))
    )
    rng = random.Random(int(seed))
    rolling: dict[str, list[float]] = defaultdict(list)
    started = time.perf_counter()
    if bool(freeze_for_identity_context_layout_residual):
        _set_identity_context_layout_residual_training_mode(model)
    elif bool(freeze_for_identity_context_residual):
        _set_identity_context_residual_training_mode(model)
    else:
        model.train()
    training_context_scale = (
        0.0 if bool(freeze_for_identity_context_layout_residual) else 1.0
    )
    training_layout_scale = (
        1.0
        if model.identity_context_layout_enabled
        and not bool(freeze_for_identity_context_residual)
        else 0.0
    )
    for step in range(1, int(steps) + 1):
        selected_queries = [
            train_query_ids[rng.randrange(len(train_query_ids))]
            for _ in range(queries_per_batch)
        ]
        batch_indices: list[int] = []
        natural_supervision_mask: list[bool] = []
        for row in range(natural_group_count):
            query_id = selected_queries[row % queries_per_batch]
            query_indices = train_indices_by_query[query_id]
            batch_indices.append(query_indices[rng.randrange(len(query_indices))])
            natural_supervision_mask.append(True)
        for _ in range(positive_group_count):
            batch_indices.append(
                identity_positive_indices[
                    rng.randrange(len(identity_positive_indices))
                ]
            )
            natural_supervision_mask.append(False)
        order = list(range(len(batch_indices)))
        rng.shuffle(order)
        batch_indices = [batch_indices[index] for index in order]
        natural_supervision_mask = [
            natural_supervision_mask[index] for index in order
        ]
        batch = _prepare_batch(
            data,
            batch_indices,
            image_root=Path(image_root),
            image_width=int(image_width),
            image_height=int(image_height),
            image_cache=image_cache,
            image_cache_device=cache_storage_device,
            image_crop_device=torch_device,
            crop_radius_px=model.crop_radius_px,
            step_px=model.step_px,
            identity_context_crop_radius_px=(
                model.identity_context_crop_radius_px
                if training_context_scale > 0.0 or training_layout_scale > 0.0
                else None
            ),
            identity_context_step_px=(
                model.identity_context_step_px
                if training_context_scale > 0.0 or training_layout_scale > 0.0
                else None
            ),
            identity_threshold_px=float(identity_threshold_px),
            identity_negative_threshold_px=float(identity_negative_threshold_px),
            identity_target=resolved_identity_target,
            spatial_radius_px=model.search_radius_px,
            natural_supervision_mask=natural_supervision_mask,
        )
        prediction = _forward_batch(
            model,
            batch,
            device=torch_device,
            use_amp=bool(use_amp),
            identity_context_residual_scale=training_context_scale,
            identity_context_layout_residual_scale=training_layout_scale,
        )
        loss, components = _training_loss(
            prediction,
            batch,
            identity_loss_weight=float(identity_loss_weight),
            availability_loss_weight=float(availability_loss_weight),
            pair_loss_weight=float(pair_loss_weight),
            coarse_hard_negative_loss_weight=float(
                coarse_hard_negative_loss_weight
            ),
            coarse_hard_negative_margin=float(coarse_hard_negative_margin),
            coarse_hard_negative_prior_power=float(
                coarse_hard_negative_prior_power
            ),
            coarse_hard_negative_evidence_weight=float(
                coarse_hard_negative_evidence_weight
            ),
            spatial_loss_weight=float(spatial_loss_weight),
            spatial_target_sigma_px=float(spatial_target_sigma_px),
            measurement_validity_loss_weight=float(
                measurement_validity_loss_weight
            ),
            measurement_success_threshold_px=float(
                measurement_success_threshold_px
            ),
            spatial_density_loss_weight=float(spatial_density_loss_weight),
            pose_view_mixture_loss_weight=float(
                pose_view_mixture_loss_weight
            ),
        )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        for key, value in components.items():
            rolling[key].append(float(value))
        if int(log_every) > 0 and (step % int(log_every) == 0 or step == int(steps)):
            elapsed = time.perf_counter() - started
            report = {
                key: float(np.mean(values[-int(log_every) :]))
                for key, values in rolling.items()
            }
            print(
                json.dumps(
                    {
                        "step": int(step),
                        "elapsed_seconds": elapsed,
                        "groups_per_second": float(step * group_batch_size / max(elapsed, 1e-9)),
                        "loss": report,
                        "image_cache": image_cache.summary(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "independent_rgb_candidate_verifier.pt"
    checkpoint_format = (
        "independent_rgb_candidate_verifier_v8"
        if model.identity_context_layout_enabled
        else "independent_rgb_candidate_verifier_v6"
        if model.identity_context_enabled
        else "independent_rgb_candidate_verifier_v5"
        if pose_view_mixture_enabled
        and float(pose_view_mixture_loss_weight) > 0.0
        else "independent_rgb_candidate_verifier_v4"
        if model.measurement_validity_semantics
        == GT_POSE_SPATIAL_DENSITY_SEMANTICS
        else "independent_rgb_candidate_verifier_v3"
    )
    checkpoint_payload = {
        "format": checkpoint_format,
        "model": model.state_dict(),
        "config": model.config(),
        "data_contract": runtime_data_contract,
        "training": {
            "seed": int(seed),
            "identity_threshold_px": float(identity_threshold_px),
            "identity_negative_threshold_px": float(
                identity_negative_threshold_px
            ),
            "identity_target": resolved_identity_target,
            "identity_target_semantics": _identity_target_semantics(
                resolved_identity_target
            ),
            "identity_loss_weight": float(identity_loss_weight),
            "availability_loss_weight": float(availability_loss_weight),
            "pair_loss_weight": float(pair_loss_weight),
            "coarse_hard_negative_loss_weight": float(
                coarse_hard_negative_loss_weight
            ),
            "coarse_hard_negative_margin": float(coarse_hard_negative_margin),
            "coarse_hard_negative_prior_power": float(
                coarse_hard_negative_prior_power
            ),
            "coarse_hard_negative_evidence_weight": float(
                coarse_hard_negative_evidence_weight
            ),
            "coarse_hard_negative_semantics": (
                "target_only_frozen_candidate_prior_weighted_geometric_group_ranking_"
                "at_declared_fusion_evidence_weight"
            ),
            "spatial_loss_weight": float(spatial_loss_weight),
            "spatial_target_sigma_px": float(spatial_target_sigma_px),
            "measurement_validity_loss_weight": float(
                measurement_validity_loss_weight
            ),
            "measurement_success_threshold_px": float(
                measurement_success_threshold_px
            ),
            "spatial_density_loss_weight": float(
                spatial_density_loss_weight
            ),
            "pose_view_mixture_loss_weight": float(
                pose_view_mixture_loss_weight
            ),
            "initialized_from_independent_checkpoint": bool(
                initialized_from_independent
            ),
            "freeze_for_measurement_validity": bool(
                freeze_for_measurement_validity
            ),
            "freeze_for_spatial_density": bool(
                freeze_for_spatial_density
            ),
            "freeze_for_pose_view_mixture": bool(
                freeze_for_pose_view_mixture
            ),
            "freeze_for_identity_context_residual": bool(
                freeze_for_identity_context_residual
            ),
            "freeze_for_identity_context_layout_residual": bool(
                freeze_for_identity_context_layout_residual
            ),
            "collect_spatial_diagnostics": bool(spatial_diagnostics_enabled),
            **initialization_provenance,
        },
    }
    torch.save(checkpoint_payload, checkpoint)
    split_metrics: dict[str, object] = {}
    prediction_path: Path | None = None
    if prediction_splits:
        prediction_blocks: list[dict[str, np.ndarray]] = []
        for split_name in prediction_splits:
            metrics, predictions = _evaluate(
                model,
                data,
                split_name=str(split_name),
                image_root=Path(image_root),
                image_width=int(image_width),
                image_height=int(image_height),
                image_cache=image_cache,
                image_cache_device=cache_storage_device,
                image_crop_device=torch_device,
                device=torch_device,
                batch_size=int(eval_group_batch_size),
                identity_threshold_px=float(identity_threshold_px),
                identity_negative_threshold_px=float(identity_negative_threshold_px),
                measurement_success_threshold_px=float(
                    measurement_success_threshold_px
                ),
                spatial_density_target_sigma_px=float(
                    spatial_target_sigma_px
                ),
                use_amp=bool(use_amp),
                collect_spatial_diagnostics=bool(spatial_diagnostics_enabled),
                identity_context_residual_scale=prediction_context_scale,
                identity_context_layout_residual_scale=prediction_layout_scale,
            )
            split_metrics[str(split_name)] = metrics
            prediction_blocks.append(predictions)
        prediction_path = output / "predictions.npz"
        prediction_arrays = {
            key: np.concatenate([block[key] for block in prediction_blocks], axis=0)
            for key in prediction_blocks[0]
        }
        order = np.argsort(prediction_arrays["evidence_row_indices"], kind="stable")
        prediction_arrays = {
            key: value[order] for key, value in prediction_arrays.items()
        }
        prediction_metadata = {
            "format": "independent_rgb_candidate_predictions_v1",
            "checkpoint_sha256": file_sha256_short(checkpoint),
            **initialization_provenance,
            "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence)),
            "availability_evidence_sha256": file_sha256_short(
                Path(availability_evidence)
            ),
            "candidate_evidence_format": data.metadata.get("format"),
            "candidate_probability_semantics": data.metadata.get(
                "candidate_probability_semantics"
            ),
            "data_contract": runtime_data_contract,
            "rgb_evidence_semantics": "candidate_log_likelihood_ratio_missing_is_zero",
            "support_view_mixture": model.config()["support_view_mixture"],
            "view_likelihood_ratio_conversion": (
                "raw_binary_logit_minus_logit_natural_train_positive_prior"
            ),
            "identity_context_semantics": model.config()["identity_context_semantics"],
            "identity_context_changes_spatial_density": False,
            "identity_context_residual_scale": prediction_context_scale,
            "identity_context_residual_scale_semantics": (
                "evaluation_only_zero_is_same_checkpoint_local_branch_counterfactual"
            ),
            "identity_context_layout_semantics": model.config()[
                "identity_context_layout_semantics"
            ],
            "identity_context_layout_changes_spatial_density": False,
            "identity_context_layout_residual_scale": prediction_layout_scale,
            "identity_context_layout_residual_scale_semantics": (
                "evaluation_only_zero_is_same_checkpoint_without_layout_residual"
            ),
            "natural_train_view_positive_prior": natural_view_positive_prior,
            "natural_train_group_positive_prior": natural_group_positive_prior,
            "identity_training_target": resolved_identity_target,
            "identity_training_target_semantics": _identity_target_semantics(
                resolved_identity_target
            ),
            "rgb_full_top_l_availability_semantics": (
                "P(at_least_one_2px_candidate_in_frozen_full_top_l)"
            ),
            "rgb_availability_changes_coarse_candidate_mass": (
                "only_after_validation_calibrated_likelihood_ratio_fusion"
            ),
            "prediction_splits": [str(value) for value in prediction_splits],
        }
        np.savez_compressed(
            prediction_path,
            **prediction_arrays,
            metadata_json=np.asarray(json.dumps(prediction_metadata, sort_keys=True)),
        )
    summary = {
        "stage": "independent_rgb_candidate_identity_train",
        "protocol": {
            "query_groups_natural_distribution": (
                "q_and_pair_calibration_loss_only"
            ),
            "identity_positive_group_enrichment": positive_fraction,
            "candidate_pool": (
                "frozen_current_system_top"
                f"{int(data.candidate_valid.shape[1])}"
            ),
            "hard_negatives": (
                "same_query_current_system_candidates_with_gt_projection_residual_ge_5px"
            ),
            "coarse_hard_negative_ranking": (
                "disabled"
                if float(coarse_hard_negative_loss_weight) == 0.0
                else "frozen_candidate_prior_weighted_geometric_positive_vs_hard_negative"
            ),
            "coarse_hard_negative_prior_is_model_input": False,
            "identity_training_target": resolved_identity_target,
            "identity_training_target_semantics": _identity_target_semantics(
                resolved_identity_target
            ),
            "appearance_positive_diagnostic": (
                "actual_query_sfm_observation_with_center_residual_le_2px"
            ),
            "ambiguous_unobserved_near_candidates_in_identity_loss": bool(
                resolved_identity_target == IDENTITY_TARGET_APPEARANCE
            ),
            "spatial_target": (
                "gt_pose_projected_candidate_offset_for_every_evaluable_view"
                if model.measurement_validity_semantics
                == GT_POSE_SPATIAL_DENSITY_SEMANTICS
                else "actual_query_sfm_observation_xy_only"
            ),
            "measurement_validity_target": (
                "gt_pose_projected_offset_inside_local_support_or_dustbin"
                if model.measurement_validity_semantics
                == GT_POSE_SPATIAL_DENSITY_SEMANTICS
                else "predicted_spatial_mode_gt_pose_projection_residual_le_threshold"
            ),
            "measurement_validity_sampling": (
                "natural_query_groups_only_no_positive_enrichment"
            ),
            "measurement_validity_training_loss": (
                "normalized_k_plus_dustbin_natural_distribution_mle"
                if model.measurement_validity_semantics
                == GT_POSE_SPATIAL_DENSITY_SEMANTICS
                else "equal_positive_negative_class_mean_then_query_disjoint_calibration"
            ),
            "measurement_validity_target_stationary": bool(
                freeze_for_measurement_validity
                or freeze_for_pose_view_mixture
                or float(spatial_density_loss_weight) > 0.0
            ),
            "pose_view_mixture_target": (
                "candidate_gt_pose_projected_normalized_k_plus_dustbin_density"
                if pose_view_mixture_enabled
                else "disabled"
            ),
            "pose_view_mixture_changes_identity_prior": False,
            "pose_view_mixture_reuses_identity_or_dustbin_posterior": False,
            "pose_view_mixture_permutation_equivariant": True,
            "measurement_validity_changes_identity_prior": False,
            "measurement_validity_used_as_independent_identity_likelihood": False,
            "view_binary_logit_semantics": "natural_posterior_log_odds",
            "view_likelihood_ratio_conversion": (
                "raw_logit_minus_logit_natural_train_positive_prior"
            ),
            "natural_train_view_positive_prior": natural_view_positive_prior,
            "natural_train_group_positive_prior": natural_group_positive_prior,
            "coarse_score_input": False,
            "retrieval_rank_input": False,
            "pose_statistic_input": False,
            "geometry_posterior_input": False,
            "ground_truth_pose_target_only": True,
            "render": False,
            "image_retrieval": False,
            "submap": False,
            "per_view_mixture_retained": True,
            "support_view_mixture": model.config()["support_view_mixture"],
            "missing_view_likelihood_ratio": 1.0,
            "missing_candidate_log_likelihood_ratio": 0.0,
            "all_rgb_missing_group_policy": "candidate_llr_zero_q_unavailable",
            "coarse_candidate_availability_mass_preserved": True,
            "availability_target": "frozen_full_top_l_2px_geometric_availability",
            "coordinate_space_runtime_validated": True,
            "real_rgb_source_manifest_runtime_validated": True,
            "spatial_support_override_requires_fresh_joint_training": bool(
                spatial_support_overrides
            ),
            "pair_forward_microbatch_changes_likelihood_semantics": False,
            "identity_context_residual_scale_evaluation_only": prediction_context_scale,
            "identity_context_residual_training": bool(
                freeze_for_identity_context_residual
            ),
            "identity_context_layout_residual_scale_evaluation_only": (
                prediction_layout_scale
            ),
            "identity_context_layout_residual_training": bool(
                freeze_for_identity_context_layout_residual
            ),
            "identity_context_changes_measured_set_availability": False,
            "prediction_export": (
                "disabled_explicitly_for_training_throughput"
                if not prediction_splits
                else "completed"
            ),
        },
        "config": {
            **model.config(),
            "spatial_support_overrides": spatial_support_overrides,
            "identity_context_overrides": identity_context_overrides,
            "identity_context_layout_overrides": identity_context_layout_overrides,
            "steps": int(steps),
            "group_batch_size": int(group_batch_size),
            "query_images_per_batch": int(queries_per_batch),
            "appearance_positive_group_fraction": positive_fraction,
            "natural_groups_per_batch": int(natural_group_count),
            "enriched_positive_groups_per_batch": int(positive_group_count),
            "eval_group_batch_size": int(eval_group_batch_size),
            "encoder_learning_rate": float(encoder_learning_rate),
            "head_learning_rate": float(head_learning_rate),
            "weight_decay": float(weight_decay),
            "identity_threshold_px": float(identity_threshold_px),
            "identity_negative_threshold_px": float(
                identity_negative_threshold_px
            ),
            "identity_target": resolved_identity_target,
            "identity_target_semantics": _identity_target_semantics(
                resolved_identity_target
            ),
            "identity_loss_weight": float(identity_loss_weight),
            "availability_loss_weight": float(availability_loss_weight),
            "pair_loss_weight": float(pair_loss_weight),
            "coarse_hard_negative_loss_weight": float(
                coarse_hard_negative_loss_weight
            ),
            "coarse_hard_negative_margin": float(coarse_hard_negative_margin),
            "coarse_hard_negative_prior_power": float(
                coarse_hard_negative_prior_power
            ),
            "coarse_hard_negative_evidence_weight": float(
                coarse_hard_negative_evidence_weight
            ),
            "spatial_loss_weight": float(spatial_loss_weight),
            "spatial_target_sigma_px": float(spatial_target_sigma_px),
            "measurement_validity_loss_weight": float(
                measurement_validity_loss_weight
            ),
            "spatial_density_loss_weight": float(
                spatial_density_loss_weight
            ),
            "pose_view_mixture_loss_weight": float(
                pose_view_mixture_loss_weight
            ),
            "measurement_success_threshold_px": float(
                measurement_success_threshold_px
            ),
            "seed": int(seed),
            "prediction_identity_context_residual_scale": prediction_context_scale,
            "prediction_identity_context_layout_residual_scale": prediction_layout_scale,
            "use_amp": bool(use_amp),
            "image_cache_dtype": cache_dtype_name,
            "image_cache_location": cache_location,
            "collect_spatial_diagnostics": bool(spatial_diagnostics_enabled),
            "coordinate_image_width": int(image_width),
            "coordinate_image_height": int(image_height),
            "freeze_for_measurement_validity": bool(
                freeze_for_measurement_validity
            ),
            "freeze_for_spatial_density": bool(
                freeze_for_spatial_density
            ),
            "freeze_for_pose_view_mixture": bool(
                freeze_for_pose_view_mixture
            ),
            "freeze_for_identity_context_residual": bool(
                freeze_for_identity_context_residual
            ),
            "freeze_for_identity_context_layout_residual": bool(
                freeze_for_identity_context_layout_residual
            ),
        },
        "data": data.summary(),
        "metrics": split_metrics,
        "runtime_seconds": float(time.perf_counter() - started),
        "image_cache": image_cache.summary(),
        "image_cache_budget": {
            "requested_max_gb": float(image_cache_max_gb),
            "gpu_non_cache_reserve_gb": float(gpu_non_cache_reserve_gb),
            "effective_max_bytes": cache_bytes,
            "storage_location": cache_location,
        },
        "inputs": {
            "candidate_evidence": str(candidate_evidence),
            "candidate_evidence_sha256": file_sha256_short(Path(candidate_evidence)),
            "availability_evidence": str(availability_evidence),
            "availability_evidence_sha256": file_sha256_short(
                Path(availability_evidence)
            ),
            "train_rows_csv": str(train_rows_csv),
            "train_rows_sha256": file_sha256_short(Path(train_rows_csv)),
            "validation_rows_csv": str(validation_rows_csv),
            "validation_rows_sha256": file_sha256_short(Path(validation_rows_csv)),
            "test_rows_csv": str(test_rows_csv),
            "test_rows_sha256": file_sha256_short(Path(test_rows_csv)),
            "init_measurement_checkpoint": str(init_measurement_checkpoint),
            "init_measurement_checkpoint_sha256": file_sha256_short(
                Path(init_measurement_checkpoint)
            ),
            "init_independent_checkpoint": (
                None
                if init_independent_checkpoint is None
                else str(init_independent_checkpoint)
            ),
            "init_independent_checkpoint_sha256": (
                None
                if init_independent_checkpoint is None
                else file_sha256_short(Path(init_independent_checkpoint))
            ),
            **initialization_provenance,
            "image_root": str(Path(image_root).resolve()),
            "image_source_manifest_sha256": runtime_data_contract[
                "image_source"
            ]["sampled_content_manifest_sha256"],
            "coordinate_space_id": runtime_data_contract[
                "coordinate_space"
            ]["coordinate_space_id"],
        },
        "outputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256_short(checkpoint),
            "predictions": (
                None if prediction_path is None else str(prediction_path)
            ),
            "predictions_sha256": (
                None
                if prediction_path is None
                else file_sha256_short(prediction_path)
            ),
            "summary": str(output / "summary.json"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
