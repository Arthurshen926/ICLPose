"""Independently audit ragged latent candidate-pose evidence scores.

The scorer is target-free and emits one paired visual/control score artifact.
This program is deliberately the first point at which pose targets are read.
It validates every ragged query/hypothesis segment before joining targets, then
reports standalone visual and descriptor-permutation-control ranking quality.
Raw evidence remains diagnostic-only regardless of the audit outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_multiscale_candidate_pose_evidence import (
    _baseline_rows,
    _paired,
    _per_query_rows,
    _split_summary,
    _validate_alpha_zero,
    _write_csv,
)
from feature_extract.tools.vfm.score_candidate_pose_latent_evidence import (
    SCORE_FORMAT,
    validate_target_free_score_metadata,
)
from feature_extract.vfm.artifacts import file_sha256_short


AUDIT_FORMAT = "candidate_pose_latent_evidence_target_audit_v1"
TARGET_FORMAT = "grouped_pose_hypothesis_targets_v1"
_EPSILON = 1e-12

_ROW_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
    "source_chosen_for_optional_pose",
    "baseline_selection_scores",
    "baseline_score_top1",
    "visual_pose_log_likelihood_ratios",
    "visual_source_log_likelihood_means",
    "visual_source_effective_point_counts",
    "control_pose_log_likelihood_ratios",
    "control_source_log_likelihood_means",
    "control_source_effective_point_counts",
)
_STATIC_FIELDS = (
    "verification_source_point_ids",
    "verification_point_sources",
    "verification_source_detector_rows",
    "verification_xy",
    "candidate_track_ids",
    "candidate_prior_probabilities",
    "null_probabilities",
    "candidate_view_weights",
    "candidate_support_image_ids",
    "visual_identity_candidate_probabilities",
    "visual_identity_conditional_probabilities",
    "visual_identity_candidate_residual",
    "visual_identity_selector_weights",
    "visual_identity_edge_usable",
    "control_identity_candidate_probabilities",
    "control_identity_conditional_probabilities",
    "control_identity_candidate_residual",
    "control_identity_selector_weights",
    "control_identity_edge_usable",
)
_POINT_FIELDS = (
    "visual_point_log_likelihood_ratios",
    "visual_point_geometric_candidate_counts",
    "visual_point_geometric_view_masses",
    "control_point_log_likelihood_ratios",
    "control_point_geometric_candidate_counts",
    "control_point_geometric_view_masses",
)
_LAYOUT_FIELDS = (
    "source_names",
    "verification_query_ids",
    "verification_split_names",
    "verification_offsets",
    "hypothesis_verification_offsets",
    "verification_static_digests",
)
_REQUIRED_SCORE_FIELDS = set(_ROW_FIELDS) | set(_STATIC_FIELDS) | set(_POINT_FIELDS) | set(
    _LAYOUT_FIELDS
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-artifacts",
        required=True,
        help="comma-separated full frozen target-free latent score shards",
    )
    parser.add_argument(
        "--target-artifact",
        required=True,
        help="post-inference grouped pose target join",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-development-prefix",
        action="store_true",
        help="allow a nonzero per-query hypothesis prefix for a smoke audit only",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("latent audit score paths must be non-empty and unique")
    return paths


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _row_keys(arrays: Mapping[str, np.ndarray]) -> tuple[tuple[str, str, str, int], ...]:
    columns = (
        np.asarray(arrays["split_names"]).astype(str).reshape(-1),
        np.asarray(arrays["evaluation_labels"]).astype(str).reshape(-1),
        np.asarray(arrays["query_ids"]).astype(str).reshape(-1),
        np.asarray(arrays["hypothesis_indices"], dtype=np.int64).reshape(-1),
    )
    if len({len(column) for column in columns}) != 1:
        raise ValueError("latent audit query/hypothesis keys are misaligned")
    return tuple(
        (str(split), str(label), str(query_id), int(hypothesis_index))
        for split, label, query_id, hypothesis_index in zip(*columns)
    )


def _ragged_offsets(value: np.ndarray, *, segment_count: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"ragged {name} offsets must be one-dimensional integers")
    offsets = np.asarray(raw, dtype=np.int64)
    if (
        offsets.shape != (int(segment_count) + 1,)
        or offsets[0] != 0
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"ragged {name} offsets are invalid")
    return offsets


def _offsets_from_lengths(lengths: Sequence[int]) -> np.ndarray:
    values = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if np.any(values < 0):
        raise ValueError("ragged segment lengths are invalid")
    return np.concatenate((np.zeros((1,), dtype=np.int64), np.cumsum(values)))


def validate_ragged_score_layout(arrays: Mapping[str, np.ndarray]) -> None:
    """Check score rows, static query points, and flat hypothesis segments.

    The function intentionally accepts the compact unit-test schema.  The
    loader below applies the additional full production schema requirements.
    """

    required = {
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "source_chosen_for_optional_pose",
        "baseline_score_top1",
        "baseline_selection_scores",
        "visual_pose_log_likelihood_ratios",
        "control_pose_log_likelihood_ratios",
        "verification_query_ids",
        "verification_split_names",
        "verification_offsets",
        "hypothesis_verification_offsets",
        "visual_point_log_likelihood_ratios",
        "control_point_log_likelihood_ratios",
        "visual_point_geometric_candidate_counts",
        "control_point_geometric_candidate_counts",
        "visual_point_geometric_view_masses",
        "control_point_geometric_view_masses",
        "visual_identity_edge_usable",
        "control_identity_edge_usable",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"ragged latent score lacks fields: {missing}")
    query_ids = np.asarray(arrays["query_ids"]).astype(str).reshape(-1)
    split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
    labels = np.asarray(arrays["evaluation_labels"]).astype(str).reshape(-1)
    hypothesis_indices = np.asarray(arrays["hypothesis_indices"], dtype=np.int64).reshape(-1)
    row_count = len(query_ids)
    if (
        row_count == 0
        or split_names.shape != (row_count,)
        or labels.shape != (row_count,)
        or hypothesis_indices.shape != (row_count,)
        or np.any(query_ids == "")
        or np.any(split_names == "")
        or np.any(labels == "")
        or len(set(_row_keys(arrays))) != row_count
    ):
        raise ValueError("ragged latent score row keys are invalid")
    for name in (
        "source_chosen_for_optional_pose",
        "baseline_score_top1",
        "baseline_selection_scores",
        "visual_pose_log_likelihood_ratios",
        "control_pose_log_likelihood_ratios",
    ):
        values = np.asarray(arrays[name]).reshape(-1)
        if values.shape != (row_count,) or not np.isfinite(values).all():
            raise ValueError(f"ragged latent score row field is invalid: {name}")

    verification_query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    verification_split_names = np.asarray(arrays["verification_split_names"]).astype(str).reshape(-1)
    query_count = len(verification_query_ids)
    if (
        query_count == 0
        or verification_split_names.shape != (query_count,)
        or np.any(verification_query_ids == "")
        or np.any(verification_split_names == "")
        or len(
            set(zip(verification_split_names.tolist(), verification_query_ids.tolist()))
        )
        != query_count
    ):
        raise ValueError("ragged latent score verification query keys are invalid")
    verification_offsets = _ragged_offsets(
        arrays["verification_offsets"],
        segment_count=query_count,
        name="verification",
    )
    hypothesis_offsets = _ragged_offsets(
        arrays["hypothesis_verification_offsets"],
        segment_count=row_count,
        name="hypothesis verification",
    )
    point_count = int(verification_offsets[-1])
    hypothesis_point_count = int(hypothesis_offsets[-1])
    if point_count <= 0:
        raise ValueError("ragged latent score has no verification points")
    point_fields = (
        "visual_point_log_likelihood_ratios",
        "control_point_log_likelihood_ratios",
        "visual_point_geometric_candidate_counts",
        "control_point_geometric_candidate_counts",
        "visual_point_geometric_view_masses",
        "control_point_geometric_view_masses",
    )
    for name in point_fields:
        values = np.asarray(arrays[name]).reshape(-1)
        if values.shape != (hypothesis_point_count,) or not np.isfinite(values).all():
            raise ValueError(f"ragged latent point field is invalid: {name}")
    visual_counts = np.asarray(
        arrays["visual_point_geometric_candidate_counts"], dtype=np.int64
    ).reshape(-1)
    control_counts = np.asarray(
        arrays["control_point_geometric_candidate_counts"], dtype=np.int64
    ).reshape(-1)
    visual_masses = np.asarray(
        arrays["visual_point_geometric_view_masses"], dtype=np.float64
    ).reshape(-1)
    control_masses = np.asarray(
        arrays["control_point_geometric_view_masses"], dtype=np.float64
    ).reshape(-1)
    if (
        np.any(visual_counts < 0)
        or np.any(control_counts < 0)
        or np.any(visual_masses < 0.0)
        or np.any(control_masses < 0.0)
        or not np.array_equal(visual_counts, control_counts)
        or not np.array_equal(visual_masses, control_masses)
    ):
        raise ValueError("latent visual/control geometry differs")

    visual_edges = np.asarray(arrays["visual_identity_edge_usable"], dtype=bool)
    control_edges = np.asarray(arrays["control_identity_edge_usable"], dtype=bool)
    if (
        visual_edges.ndim < 1
        or visual_edges.shape[0] != point_count
        or control_edges.shape != visual_edges.shape
        or not np.array_equal(visual_edges, control_edges)
    ):
        raise ValueError("latent visual/control identity geometry differs")

    points_by_query = {
        (str(split_name), str(query_id)): int(verification_offsets[index + 1] - verification_offsets[index])
        for index, (split_name, query_id) in enumerate(
            zip(verification_split_names.tolist(), verification_query_ids.tolist())
        )
    }
    if any(count <= 0 for count in points_by_query.values()):
        raise ValueError("ragged verification query segment is empty")
    for row, (split_name, query_id) in enumerate(zip(split_names.tolist(), query_ids.tolist())):
        key = (str(split_name), str(query_id))
        expected = points_by_query.get(key)
        actual = int(hypothesis_offsets[row + 1] - hypothesis_offsets[row])
        if expected is None or actual != expected:
            raise ValueError("ragged hypothesis point segment differs from its query")


def _validate_complete_score_schema(arrays: Mapping[str, np.ndarray]) -> None:
    missing = sorted(_REQUIRED_SCORE_FIELDS.difference(arrays))
    if missing:
        raise ValueError(f"latent score is incomplete: {missing}")
    validate_ragged_score_layout(arrays)
    row_count = len(np.asarray(arrays["query_ids"]).reshape(-1))
    point_count = int(np.asarray(arrays["verification_offsets"], dtype=np.int64)[-1])
    source_names = np.asarray(arrays["source_names"]).astype(str).reshape(-1)
    if (
        len(source_names) == 0
        or len(set(source_names.tolist())) != len(source_names)
        or np.any(source_names == "")
    ):
        raise ValueError("latent score source names are invalid")
    source_count = len(source_names)
    for name in (
        "visual_source_log_likelihood_means",
        "control_source_log_likelihood_means",
        "visual_source_effective_point_counts",
        "control_source_effective_point_counts",
    ):
        values = np.asarray(arrays[name])
        if (
            values.shape != (row_count, source_count)
            or not np.isfinite(values).all()
            or (
                name.endswith("effective_point_counts")
                and np.any(np.asarray(values, dtype=np.int64) < 0)
            )
        ):
            raise ValueError(f"latent score source statistic is invalid: {name}")
    if (
        np.asarray(arrays["verification_static_digests"]).astype(str).shape
        != np.asarray(arrays["verification_query_ids"]).shape
        or np.any(np.asarray(arrays["verification_static_digests"]).astype(str) == "")
        or np.asarray(arrays["verification_source_point_ids"]).shape != (point_count,)
        or np.asarray(arrays["verification_point_sources"]).shape != (point_count,)
        or np.asarray(arrays["verification_source_detector_rows"]).shape != (point_count,)
        or np.asarray(arrays["verification_xy"]).shape != (point_count, 2)
    ):
        raise ValueError("latent score static point layout is invalid")
    candidate_track_ids = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    candidate_prior = np.asarray(arrays["candidate_prior_probabilities"], dtype=np.float64)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float64).reshape(-1)
    view_weights = np.asarray(arrays["candidate_view_weights"], dtype=np.float64)
    support_ids = np.asarray(arrays["candidate_support_image_ids"]).astype(str)
    if (
        candidate_track_ids.shape != (point_count, 20)
        or candidate_prior.shape != candidate_track_ids.shape
        or null.shape != (point_count,)
        or view_weights.shape != (point_count, 20, 2)
        or support_ids.shape != view_weights.shape
        or not np.isfinite(candidate_prior).all()
        or not np.isfinite(null).all()
        or not np.isfinite(view_weights).all()
        or np.any(candidate_prior < 0.0)
        or np.any(null < 0.0)
        or np.any(view_weights < 0.0)
        or not np.allclose(candidate_prior.sum(axis=1) + null, 1.0, atol=1e-4)
    ):
        raise ValueError("latent score fixed candidate layout is invalid")
    for prefix in ("visual", "control"):
        candidate = np.asarray(
            arrays[f"{prefix}_identity_candidate_probabilities"], dtype=np.float64
        )
        conditional = np.asarray(
            arrays[f"{prefix}_identity_conditional_probabilities"], dtype=np.float64
        )
        residual = np.asarray(
            arrays[f"{prefix}_identity_candidate_residual"], dtype=np.float64
        )
        selector = np.asarray(
            arrays[f"{prefix}_identity_selector_weights"], dtype=np.float64
        ).reshape(-1)
        if (
            candidate.shape != (point_count, 20)
            or conditional.shape != candidate.shape
            or residual.shape != candidate.shape
            or selector.shape != (point_count,)
            or not np.isfinite(candidate).all()
            or not np.isfinite(conditional).all()
            or not np.isfinite(residual).all()
            or not np.isfinite(selector).all()
            or np.any(candidate < 0.0)
            or np.any(conditional < 0.0)
            or np.any(selector < 0.0)
            or not np.allclose(candidate.sum(axis=1) + null, 1.0, atol=1e-4)
            or not np.allclose(conditional.sum(axis=1), 1.0, atol=1e-4)
        ):
            raise ValueError(f"latent score identity layout is invalid: {prefix}")


def _strict_contract(metadata: Mapping[str, object]) -> Mapping[str, object]:
    strict = metadata.get("strict_candidate_pose_latent_evidence_contract")
    required = {
        "heldout_query_rows": True,
        "formal_p1_mixed_multiscale_points": True,
        "fixed_global_topl": True,
        "fixed_candidate_top_k": 20,
        "identity_candidate_posterior_soft_topl": True,
        "identity_evaluated_before_pose": True,
        "identity_posterior_fixed_across_hypotheses": True,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "fixed_support_view_count": 2,
        "explicit_null": True,
        "candidate_projection_is_only_pose_dependent_encoder_input": True,
        "candidate_pose_matrix_excluded_from_encoder": True,
        "residual_and_target_excluded_from_encoder": True,
        "support_descriptor_permutation_control": True,
        "ragged_query_point_layout": True,
        "no_pnp": True,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    if not isinstance(strict, Mapping) or any(strict.get(key) != value for key, value in required.items()):
        raise ValueError("latent score has an invalid strict contract")
    return strict


def _load_score(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(_REQUIRED_SCORE_FIELDS.difference(payload.files))
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: incomplete latent score artifact ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in _REQUIRED_SCORE_FIELDS}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: latent score metadata is invalid")
    validate_target_free_score_metadata(metadata)
    if (
        metadata.get("format") != SCORE_FORMAT
        or set(metadata.get("evidence_variants", []))
        != {"visual", "support_descriptor_permutation_control"}
        or metadata.get("raw_score_is_calibrated_independent_pose_likelihood") is not False
    ):
        raise ValueError(f"{path}: latent score metadata has the wrong semantics")
    _strict_contract(metadata)
    _validate_complete_score_schema(arrays)
    return arrays, metadata


def _production_scope(metadata: Mapping[str, object], *, allow_development_prefix: bool) -> None:
    scope = metadata.get("hypothesis_scope")
    if not isinstance(scope, Mapping):
        raise ValueError("latent score has no hypothesis scope")
    full = scope.get("all_frozen_hypotheses") is True
    limit = int(scope.get("development_prefix_limit_per_query", -1))
    if full != (limit == 0):
        raise ValueError("latent score hypothesis scope is internally inconsistent")
    if not full and not bool(allow_development_prefix):
        raise ValueError("latent audit rejects a development hypothesis prefix")
    if full and metadata.get("baseline_top1_source_equivalent") is not True:
        raise ValueError("full latent score does not reproduce its frozen baseline top-1")


def _merge_compatibility(metadata: Mapping[str, object]) -> dict[str, object]:
    checkpoint = metadata.get("model_checkpoint")
    if not isinstance(checkpoint, Mapping) or not str(checkpoint.get("sha256", "")):
        raise ValueError("latent score has no checkpoint identity")
    return {
        "format": metadata.get("format"),
        "version": metadata.get("version"),
        "strict_contract": metadata.get("strict_candidate_pose_latent_evidence_contract"),
        "model_checkpoint_sha256": checkpoint.get("sha256"),
        "model_checkpoint_contract": metadata.get("model_checkpoint_contract"),
        "support_descriptor_derangement": metadata.get("support_descriptor_derangement"),
        "raw_score_semantics": metadata.get("raw_score_semantics"),
        "raw_score_is_calibrated_independent_pose_likelihood": metadata.get(
            "raw_score_is_calibrated_independent_pose_likelihood"
        ),
        "verification_layout": metadata.get("verification_layout"),
    }


def _merge_scores(
    paths: Sequence[Path], *, allow_development_prefix: bool
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    loaded = [_load_score(path) for path in paths]
    if not loaded:
        raise ValueError("latent audit has no score artifacts")
    compatibility = _canonical_hash(_merge_compatibility(loaded[0][1]))
    source_names = np.asarray(loaded[0][0]["source_names"]).astype(str)
    row_parts: dict[str, list[np.ndarray]] = {field: [] for field in _ROW_FIELDS}
    static_parts: dict[str, list[np.ndarray]] = {field: [] for field in _STATIC_FIELDS}
    point_parts: dict[str, list[np.ndarray]] = {field: [] for field in _POINT_FIELDS}
    verification_query_ids: list[np.ndarray] = []
    verification_split_names: list[np.ndarray] = []
    verification_static_digests: list[np.ndarray] = []
    verification_lengths: list[int] = []
    hypothesis_lengths: list[int] = []
    row_keys: set[tuple[str, str, str, int]] = set()
    query_keys: set[tuple[str, str]] = set()
    metadata_items: list[dict[str, object]] = []
    for arrays, metadata in loaded:
        _production_scope(metadata, allow_development_prefix=allow_development_prefix)
        if _canonical_hash(_merge_compatibility(metadata)) != compatibility:
            raise ValueError("latent score shards have incompatible model or frozen contracts")
        if not np.array_equal(np.asarray(arrays["source_names"]).astype(str), source_names):
            raise ValueError("latent score shards have different source orders")
        current_row_keys = set(_row_keys(arrays))
        if row_keys.intersection(current_row_keys):
            raise ValueError("latent score shards repeat query/hypothesis rows")
        row_keys.update(current_row_keys)
        current_query_keys = set(
            zip(
                np.asarray(arrays["verification_split_names"]).astype(str).tolist(),
                np.asarray(arrays["verification_query_ids"]).astype(str).tolist(),
            )
        )
        if query_keys.intersection(current_query_keys):
            raise ValueError("latent score shards repeat static query points")
        query_keys.update(current_query_keys)
        for field in _ROW_FIELDS:
            row_parts[field].append(np.asarray(arrays[field]))
        for field in _STATIC_FIELDS:
            static_parts[field].append(np.asarray(arrays[field]))
        for field in _POINT_FIELDS:
            point_parts[field].append(np.asarray(arrays[field]))
        verification_query_ids.append(np.asarray(arrays["verification_query_ids"]))
        verification_split_names.append(np.asarray(arrays["verification_split_names"]))
        verification_static_digests.append(np.asarray(arrays["verification_static_digests"]))
        verification_lengths.extend(
            np.diff(np.asarray(arrays["verification_offsets"], dtype=np.int64)).tolist()
        )
        hypothesis_lengths.extend(
            np.diff(
                np.asarray(arrays["hypothesis_verification_offsets"], dtype=np.int64)
            ).tolist()
        )
        metadata_items.append(metadata)
    merged: dict[str, np.ndarray] = {
        field: np.concatenate(parts, axis=0) for field, parts in row_parts.items()
    }
    merged.update({field: np.concatenate(parts, axis=0) for field, parts in static_parts.items()})
    merged.update({field: np.concatenate(parts, axis=0) for field, parts in point_parts.items()})
    merged["source_names"] = source_names.copy()
    merged["verification_query_ids"] = np.concatenate(verification_query_ids, axis=0)
    merged["verification_split_names"] = np.concatenate(verification_split_names, axis=0)
    merged["verification_static_digests"] = np.concatenate(verification_static_digests, axis=0)
    merged["verification_offsets"] = _offsets_from_lengths(verification_lengths)
    merged["hypothesis_verification_offsets"] = _offsets_from_lengths(hypothesis_lengths)
    _validate_complete_score_schema(merged)
    return merged, metadata_items


def _load_targets(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    fields = (
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "translation_errors_m",
        "rotation_errors_deg",
    )
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(fields).difference(payload.files))
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: latent audit target is incomplete ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != TARGET_FORMAT
        or metadata.get("contains_target_fields") is not True
        or metadata.get("targets_joined_after_inference") is not True
    ):
        raise ValueError("latent audit target is not a post-inference grouped target join")
    count = int(metadata.get("row_count", -1))
    if count <= 0 or any(np.asarray(arrays[field]).shape != (count,) for field in fields):
        raise ValueError("latent audit target rows are misaligned")
    translation = np.asarray(arrays["translation_errors_m"], dtype=np.float64)
    rotation = np.asarray(arrays["rotation_errors_deg"], dtype=np.float64)
    keys = _row_keys(arrays)
    if (
        not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
        or np.any(translation < 0.0)
        or np.any(rotation < 0.0)
        or len(keys) != len(set(keys))
    ):
        raise ValueError("latent audit target values are invalid")
    return arrays, metadata


def _validate_target_lineage(
    *, score_metadata: Sequence[Mapping[str, object]], target_metadata: Mapping[str, object]
) -> None:
    allowed = {
        str(value) for value in target_metadata.get("inference_artifact_sha256", []) if str(value)
    }
    if not allowed:
        raise ValueError("latent audit target has no frozen hypothesis lineage")
    for metadata in score_metadata:
        inputs = metadata.get("inputs")
        hypothesis = inputs.get("hypothesis_artifact") if isinstance(inputs, Mapping) else None
        if (
            not isinstance(hypothesis, Mapping)
            or str(hypothesis.get("sha256", "")) not in allowed
        ):
            raise ValueError("latent score/target hypothesis lineage differs")


def paired_best_10cm_rank(
    visual_rows: Sequence[Mapping[str, object]], control_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Pair best-correct ranks without treating absent correct poses as wins."""

    def key(row: Mapping[str, object]) -> tuple[str, str, str]:
        return (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"]))

    visual = {key(row): row for row in visual_rows}
    control = {key(row): row for row in control_rows}
    if not visual or set(visual) != set(control) or len(visual) != len(visual_rows):
        raise ValueError("latent visual/control per-query rows are unpaired")
    deltas: list[float] = []
    unavailable = 0
    for item in sorted(visual):
        visual_rank = visual[item].get("best_10cm_rank")
        control_rank = control[item].get("best_10cm_rank")
        if (visual_rank is None) != (control_rank is None):
            raise ValueError("latent visual/control correct-pose coverage differs")
        if visual_rank is None:
            unavailable += 1
            continue
        deltas.append(float(visual_rank) - float(control_rank))
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "best_10cm_rank_pair_count": int(len(values)),
        "best_10cm_rank_unavailable_count": int(unavailable),
        "best_10cm_rank_wins": int(np.count_nonzero(values < -_EPSILON)),
        "best_10cm_rank_losses": int(np.count_nonzero(values > _EPSILON)),
        "best_10cm_rank_ties": int(np.count_nonzero(np.abs(values) <= _EPSILON)),
        "median_best_10cm_rank_delta": (
            None if len(values) == 0 else float(np.median(values))
        ),
    }


def primary_gate(
    *,
    visual: Mapping[str, object],
    control: Mapping[str, object],
    paired_rank: Mapping[str, object],
) -> dict[str, bool]:
    """Require rank gain without degrading a held-out pose-error tail."""

    visual_median_rank = visual.get("median_best_10cm_rank")
    control_median_rank = control.get("median_best_10cm_rank")
    visual_p90_rank = visual.get("p90_best_10cm_rank")
    control_p90_rank = control.get("p90_best_10cm_rank")

    def finite(value: object) -> bool:
        return value is not None and bool(np.isfinite(float(value)))

    return {
        "visual_median_best_10cm_rank_within_top20": finite(visual_median_rank)
        and float(visual_median_rank) <= 20.0,
        "visual_median_best_10cm_rank_below_control": finite(visual_median_rank)
        and finite(control_median_rank)
        and float(visual_median_rank) < float(control_median_rank),
        "visual_p90_best_10cm_rank_not_worse": finite(visual_p90_rank)
        and finite(control_p90_rank)
        and float(visual_p90_rank) <= float(control_p90_rank),
        "visual_p90_translation_not_worse": finite(
            visual.get("p90_selected_translation_cm")
        )
        and finite(control.get("p90_selected_translation_cm"))
        and float(visual["p90_selected_translation_cm"])
        <= float(control["p90_selected_translation_cm"]),
        "visual_catastrophic_tail_not_worse": int(
            visual.get("catastrophic_1m_count", 1 << 30)
        )
        <= int(control.get("catastrophic_1m_count", -(1 << 30))),
        "paired_best_10cm_rank_wins_exceed_losses": int(
            paired_rank.get("best_10cm_rank_wins", 0)
        )
        > int(paired_rank.get("best_10cm_rank_losses", 0)),
    }


def _source_summary(values: Mapping[str, np.ndarray], *, prefix: str) -> dict[str, object]:
    names = np.asarray(values["source_names"]).astype(str).reshape(-1)
    logs = np.asarray(values[f"{prefix}_source_log_likelihood_means"], dtype=np.float64)
    counts = np.asarray(values[f"{prefix}_source_effective_point_counts"], dtype=np.float64)
    return {
        str(name): {
            "mean_log_likelihood_ratio": float(np.mean(logs[:, index])),
            "mean_effective_point_count": float(np.mean(counts[:, index])),
        }
        for index, name in enumerate(names.tolist())
    }


def validate_score_target_coverage(
    *,
    keys: Sequence[tuple[str, str, str, int]],
    target_keys: Sequence[tuple[str, str, str, int]],
    score_metadata: Sequence[Mapping[str, object]],
    require_complete_scope: bool,
) -> None:
    requested_splits = {
        str(split)
        for metadata in score_metadata
        for split in metadata.get("score_splits", [])
    }
    if not requested_splits or requested_splits - {"validation", "test"}:
        raise ValueError("latent score metadata has invalid held-out split scope")
    expected = {key for key in target_keys if key[0] in requested_splits}
    observed = set(keys)
    if not observed or not observed.issubset(expected):
        missing = len(expected.difference(observed))
        extra = len(observed.difference(expected))
        raise ValueError(
            "latent audit score is outside its frozen target scope "
            f"(missing={missing}, extra={extra})"
        )
    if bool(require_complete_scope) and observed != expected:
        missing = len(expected.difference(observed))
        extra = len(observed.difference(expected))
        raise ValueError(
            "latent audit score does not cover the complete frozen target scope "
            f"(missing={missing}, extra={extra})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite existing latent audit: {summary_path}")
    score_paths = _paths(str(args.score_artifacts))
    scores, score_metadata = _merge_scores(
        score_paths, allow_development_prefix=bool(args.allow_development_prefix)
    )
    targets, target_metadata = _load_targets(Path(args.target_artifact))
    _validate_target_lineage(
        score_metadata=score_metadata, target_metadata=target_metadata
    )
    keys = _row_keys(scores)
    target_keys = _row_keys(targets)
    validate_score_target_coverage(
        keys=keys,
        target_keys=target_keys,
        score_metadata=score_metadata,
        require_complete_scope=not bool(args.allow_development_prefix),
    )
    target_positions = {key: row for row, key in enumerate(target_keys)}
    try:
        target_rows = np.asarray([target_positions[key] for key in keys], dtype=np.int64)
    except KeyError as error:  # Defensive after complete-scope validation.
        raise ValueError("latent score row is absent from its target join") from error
    translation = np.asarray(targets["translation_errors_m"], dtype=np.float64)[target_rows]
    rotation = np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target_rows]
    baseline_scores = np.asarray(scores["baseline_selection_scores"], dtype=np.float64)
    source_top1 = np.asarray(scores["baseline_score_top1"], dtype=bool)
    if (
        not np.isfinite(baseline_scores).all()
        or translation.shape != baseline_scores.shape
        or rotation.shape != baseline_scores.shape
    ):
        raise ValueError("latent score and target rows are misaligned")
    tie_break_orders = np.asarray(scores["hypothesis_indices"], dtype=np.int64)
    baseline_rows = _baseline_rows(
        keys=keys,
        baseline_scores=baseline_scores,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_break_orders,
    )
    _validate_alpha_zero(
        source_top1=source_top1, keys=keys, baseline_rows=baseline_rows
    )
    visual_rows = _per_query_rows(
        keys=keys,
        scores=np.asarray(scores["visual_pose_log_likelihood_ratios"], dtype=np.float64),
        baseline_scores=baseline_scores,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_break_orders,
        family="candidate_pose_latent_evidence_visual",
        score_mode="standalone_raw_latent_log_likelihood_ratio_diagnostic_only",
        alpha=None,
    )
    control_rows = _per_query_rows(
        keys=keys,
        scores=np.asarray(scores["control_pose_log_likelihood_ratios"], dtype=np.float64),
        baseline_scores=baseline_scores,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_break_orders,
        family="candidate_pose_latent_evidence_support_descriptor_permutation_control",
        score_mode="standalone_raw_latent_log_likelihood_ratio_diagnostic_only",
        alpha=None,
    )
    visual_summary = _split_summary(visual_rows)
    control_summary = _split_summary(control_rows)
    common_splits = sorted(set(visual_summary).intersection(control_summary))
    if not common_splits:
        raise ValueError("latent visual/control audit has no common held-out split")
    paired_rank = paired_best_10cm_rank(visual_rows, control_rows)
    split_paired_rank = {
        split: paired_best_10cm_rank(
            [row for row in visual_rows if str(row["split_name"]) == split],
            [row for row in control_rows if str(row["split_name"]) == split],
        )
        for split in common_splits
    }
    split_gates = {
        split: primary_gate(
            visual=visual_summary[split],
            control=control_summary[split],
            paired_rank=split_paired_rank[split],
        )
        for split in common_splits
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = output_dir / "per_query.csv"
    _write_csv(per_query_path, [*baseline_rows, *visual_rows, *control_rows])
    summary: dict[str, Any] = {
        "stage": "candidate_pose_latent_evidence_target_audit",
        "format": AUDIT_FORMAT,
        "protocol": {
            "target_join_isolated_from_scoring": True,
            "visual_and_support_descriptor_permutation_control_paired": True,
            "ragged_query_and_hypothesis_layout_validated": True,
            "complete_frozen_target_scope_required": not bool(args.allow_development_prefix),
            "fixed_global_topl": True,
            "no_image_retrieval_or_submap": True,
            "no_render": True,
            "no_pnp": True,
            "raw_scores_must_not_feed_pnp": True,
            "promotion_allowed": False,
        },
        "baseline": {"splits": _split_summary(baseline_rows)},
        "visual": {
            "splits": visual_summary,
            "paired_vs_baseline": _paired(baseline_rows, visual_rows),
            "source_evidence": _source_summary(scores, prefix="visual"),
        },
        "support_descriptor_permutation_control": {
            "splits": control_summary,
            "paired_vs_baseline": _paired(baseline_rows, control_rows),
            "source_evidence": _source_summary(scores, prefix="control"),
        },
        "visual_vs_control": {
            "paired_best_10cm_rank": paired_rank,
            "split_paired_best_10cm_rank": split_paired_rank,
            "split_primary_gate": split_gates,
            "all_split_primary_gates_pass": bool(split_gates)
            and all(all(gate.values()) for gate in split_gates.values()),
        },
        "inputs": {
            "score_artifacts": [str(path) for path in score_paths],
            "score_artifact_sha256": [file_sha256_short(path) for path in score_paths],
            "target_artifact": str(args.target_artifact),
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
            "score_contract_sha256": _canonical_hash(
                _merge_compatibility(score_metadata[0])
            ),
            "target_inference_artifact_sha256": list(
                target_metadata.get("inference_artifact_sha256", [])
            ),
        },
        "outputs": {"per_query": str(per_query_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
