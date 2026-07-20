"""Audit frozen candidate appearance features against post-hoc SfM targets.

This is intentionally separate from
``build_frozen_multiscale_candidate_appearance.py``.  The builder reads only
target-free S0 inputs and real-image feature caches.  This audit is the first
place where proposal residuals and registered query-track identities are
loaded, exclusively to measure whether the already-frozen raw appearance
features distinguish correct landmark candidates from repeated alternatives.

It does not fit a calibration, choose a profile, alter a posterior, or score a
pose.  In particular, an apparent validation win here is not a promotion
decision.  A family must later pass an independently frozen cross-split gate
before it is allowed into pose-level calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


ARTIFACT_FORMAT = "frozen_multiscale_candidate_absolute_appearance_v1"
REGION_LAYOUT_ARTIFACT_FORMAT = "frozen_multiscale_candidate_region_layout_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--appearance_artifacts",
        required=True,
        help="comma-separated target-free per-query appearance shards",
    )
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--colmap_model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--geometric_thresholds_px", default="2,5")
    parser.add_argument("--exact_identity_radius_px", type=float, default=2.0)
    parser.add_argument("--audit_splits", default="validation,test")
    parser.add_argument(
        "--allow_diagnostic_artifact",
        action="store_true",
        help="allow a partial verification-point smoke artifact only for code checks",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("appearance artifact paths must be non-empty and unique")
    return paths


def _splits(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    allowed = {"validation", "test"}
    if not result or len(set(result)) != len(result) or set(result) - allowed:
        raise ValueError("audit splits must be unique validation/test names")
    return result


def _thresholds(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    except ValueError as error:
        raise ValueError("geometric thresholds must be numbers") from error
    if (
        not result
        or len(set(result)) != len(result)
        or any(not np.isfinite(item) or item <= 0.0 for item in result)
    ):
        raise ValueError("geometric thresholds must be unique positive finite values")
    return tuple(sorted(result))


def _metadata(data: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    if "metadata_json" not in data:
        raise ValueError(f"{context} lacks metadata_json")
    payload = json.loads(str(np.asarray(data["metadata_json"]).item()))
    if not isinstance(payload, dict):
        raise ValueError(f"{context} metadata is not an object")
    return payload


def _load_appearance(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    base_fields = (
        "verification_query_ids",
        "split_names",
        "verification_source_row_indices",
        "verification_xy",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_view_weights",
        "family_names",
        "candidate_view_usable",
        "candidate_usable_view_weight_mass",
    )
    with np.load(Path(path), allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path}: missing frozen appearance metadata")
        metadata_value = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        if not isinstance(metadata_value, dict):
            raise ValueError(f"{path}: frozen appearance metadata is not an object")
        artifact_format = str(metadata_value.get("format", ""))
        if artifact_format == ARTIFACT_FORMAT:
            score_fields = (
                "candidate_view_aligned_ncc",
                "candidate_view_overlap_fraction",
                "candidate_view_support_fraction",
                "candidate_coverage_weighted_ncc",
                "candidate_max_view_ncc",
            )
        elif artifact_format == REGION_LAYOUT_ARTIFACT_FORMAT:
            score_fields = (
                "candidate_view_region_similarity",
                "candidate_view_region_pair_count",
                "candidate_coverage_weighted_region_similarity",
                "candidate_max_view_region_similarity",
            )
        else:
            raise ValueError(f"{path}: unsupported frozen appearance format {artifact_format!r}")
        fields = (*base_fields, *score_fields)
        missing = sorted(set(fields).difference(payload.files))
        if missing:
            raise ValueError(f"{path}: missing frozen appearance fields {missing}")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        arrays["metadata_json"] = np.asarray(payload["metadata_json"]).copy()
    metadata = _metadata(arrays, context=str(path))
    strict = metadata.get("strict_frozen_appearance_contract")
    required = {
        "heldout_s0_verification_rows": True,
        "fixed_global_topl": True,
        "candidate_identity_fixed": True,
        "candidate_3d_projection_or_pose_used": False,
        "candidate_reselection": False,
        "support_reselection": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
    }
    if (
        metadata.get("format") not in {ARTIFACT_FORMAT, REGION_LAYOUT_ARTIFACT_FORMAT}
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or not isinstance(strict, Mapping)
        or any(strict.get(key) is not expected for key, expected in required.items())
    ):
        raise ValueError(f"{path}: invalid target-free frozen appearance contract")
    count = int(metadata.get("row_count", -1))
    if metadata.get("format") == REGION_LAYOUT_ARTIFACT_FORMAT:
        arrays["candidate_view_aligned_ncc"] = arrays[
            "candidate_view_region_similarity"
        ]
        arrays["candidate_coverage_weighted_ncc"] = arrays[
            "candidate_coverage_weighted_region_similarity"
        ]
        arrays["candidate_max_view_ncc"] = arrays[
            "candidate_max_view_region_similarity"
        ]
    row_fields = (
        "verification_query_ids",
        "split_names",
        "verification_source_row_indices",
        "verification_xy",
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_view_weights",
        "candidate_view_aligned_ncc",
        "candidate_view_usable",
        "candidate_coverage_weighted_ncc",
        "candidate_max_view_ncc",
        "candidate_usable_view_weight_mass",
    )
    if count <= 0 or any(np.asarray(arrays[field]).shape[0] != count for field in row_fields):
        raise ValueError(f"{path}: frozen appearance row fields are not aligned")
    families = np.asarray(arrays["family_names"]).astype(str).reshape(-1)
    if len(families) == 0 or len(set(families.tolist())) != len(families):
        raise ValueError(f"{path}: invalid appearance family names")
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    weights = np.asarray(arrays["candidate_view_weights"], dtype=np.float32)
    view_score = np.asarray(arrays["candidate_view_aligned_ncc"], dtype=np.float32)
    view_usable = np.asarray(arrays["candidate_view_usable"], dtype=bool)
    mean = np.asarray(arrays["candidate_coverage_weighted_ncc"], dtype=np.float32)
    maximum = np.asarray(arrays["candidate_max_view_ncc"], dtype=np.float32)
    mass = np.asarray(arrays["candidate_usable_view_weight_mass"], dtype=np.float32)
    if (
        tracks.ndim != 2
        or probabilities.shape != tracks.shape
        or null.shape != (count,)
        or weights.shape[:2] != tracks.shape
        or view_score.shape != (*weights.shape, len(families))
        or view_usable.shape != view_score.shape
        or mean.shape != (*tracks.shape, len(families))
        or maximum.shape != mean.shape
        or mass.shape != mean.shape
        or np.any(probabilities < 0.0)
        or np.any(null < 0.0)
        or np.max(np.abs(probabilities.sum(axis=1) + null - 1.0)) > 2e-4
        or np.any(~np.isfinite(view_score[view_usable]))
        or np.any(~np.isfinite(mean[mass > 0.0]))
        or np.any(~np.isfinite(maximum[mass > 0.0]))
        or np.any((mass < 0.0) | (mass > 1.0001))
    ):
        raise ValueError(f"{path}: frozen appearance arrays have invalid values")
    keys = _row_keys(arrays)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: repeated query/source-row identities")
    return arrays, metadata


def _row_keys(arrays: Mapping[str, np.ndarray]) -> tuple[tuple[str, int], ...]:
    ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
    rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64).reshape(-1)
    if len(ids) != len(rows):
        raise ValueError("appearance row keys are misaligned")
    return tuple((str(query_id), int(row)) for query_id, row in zip(ids, rows))


def _merge_appearance(
    paths: Sequence[Path], *, allow_diagnostic_artifact: bool
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], tuple[tuple[str, int], ...]]:
    loaded = [_load_appearance(path) for path in paths]
    families = np.asarray(loaded[0][0]["family_names"]).astype(str)
    reference = loaded[0][1]
    compatibility = {
        "format": reference.get("format"),
        "version": reference.get("version"),
        "strict_frozen_appearance_contract": reference.get(
            "strict_frozen_appearance_contract"
        ),
        "profiles": reference.get("profiles"),
        "appearance_config": reference.get("appearance_config"),
        "implementation_hash": reference.get("implementation_hash"),
    }
    row_count = int(reference.get("row_count", 0))
    if row_count != 192 and not bool(allow_diagnostic_artifact):
        raise ValueError("partial appearance artifact requires --allow_diagnostic_artifact")
    fields = tuple(
        field
        for field in loaded[0][0]
        if field not in {"metadata_json", "family_names"}
    )
    merged_parts: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    keys: list[tuple[str, int]] = []
    metadata: list[dict[str, Any]] = []
    for path, (arrays, item_metadata) in zip(paths, loaded):
        item_compatibility = {
            "format": item_metadata.get("format"),
            "version": item_metadata.get("version"),
            "strict_frozen_appearance_contract": item_metadata.get(
                "strict_frozen_appearance_contract"
            ),
            "profiles": item_metadata.get("profiles"),
            "appearance_config": item_metadata.get("appearance_config"),
            "implementation_hash": item_metadata.get("implementation_hash"),
        }
        if item_compatibility != compatibility or not np.array_equal(
            np.asarray(arrays["family_names"]).astype(str), families
        ):
            raise ValueError(f"{path}: frozen appearance shard configuration differs")
        if int(item_metadata.get("row_count", 0)) != 192 and not bool(
            allow_diagnostic_artifact
        ):
            raise ValueError(f"{path}: partial frozen appearance shard is not auditable")
        keys.extend(_row_keys(arrays))
        for field in fields:
            merged_parts[field].append(np.asarray(arrays[field]))
        metadata.append(item_metadata)
    if len(keys) != len(set(keys)):
        raise ValueError("appearance shards overlap query/source-row identities")
    merged = {field: np.concatenate(parts, axis=0) for field, parts in merged_parts.items()}
    merged["family_names"] = families
    return merged, metadata, tuple(keys)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    target = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if target.shape != values.shape or np.any(~np.isfinite(values)):
        raise ValueError("average precision inputs are invalid")
    positive_count = int(np.sum(target))
    if positive_count == 0:
        return None
    order = np.argsort(-values, kind="stable")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / positive_count)


def _top_and_positive_rank(
    scores: np.ndarray, labels: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    candidate_valid = np.asarray(valid, dtype=bool)
    if values.shape != positive.shape or positive.shape != candidate_valid.shape:
        raise ValueError("rank inputs are incompatible")
    if np.any(~np.isfinite(values[candidate_valid])):
        raise ValueError("candidate score is non-finite where direct evidence is usable")
    if np.any(positive & ~candidate_valid):
        raise ValueError("positive candidate cannot be ranked under the supplied validity")
    ranked = np.where(candidate_valid, values, -np.inf)
    order = np.argsort(-ranked, axis=1, kind="stable")
    ranked_positive = np.take_along_axis(positive, order, axis=1)
    has_positive = np.any(ranked_positive, axis=1)
    rank = np.argmax(ranked_positive, axis=1).astype(np.int64) + 1
    rank[~has_positive | ~np.any(candidate_valid, axis=1)] = -1
    return order[:, 0].astype(np.int64), rank


def _rank_metrics(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    candidate_valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1).copy()
    if values.shape != positive.shape or positive.shape != valid.shape or selected.shape != (
        len(values),
    ):
        raise ValueError("rank metric inputs are incompatible")
    selected &= np.any(valid, axis=1)
    if not np.any(selected):
        return {"row_count": 0}
    # Ranking functions require that positives are in the explicit candidate set.
    positives_present = np.any(positive & valid, axis=1)
    top, rank = _top_and_positive_rank(
        values,
        positive & valid,
        valid,
    )
    selected_positive = selected & positives_present
    flat_valid = valid & selected[:, None]
    return {
        "row_count": int(np.sum(selected)),
        "candidate_edge_count": int(np.sum(flat_valid)),
        "candidate_edge_positive_rate": float(np.mean(positive[flat_valid])),
        "candidate_pair_average_precision": _average_precision(
            positive[flat_valid], values[flat_valid]
        ),
        "positive_row_count": int(np.sum(selected_positive)),
        "positive_row_rate": float(np.mean(positives_present[selected])),
        "top1_positive_rate_given_positive": (
            None
            if not np.any(selected_positive)
            else float(np.mean(positive[selected_positive, top[selected_positive]]))
        ),
        "median_first_positive_rank": (
            None
            if not np.any(selected_positive)
            else float(np.median(rank[selected_positive]))
        ),
        "p90_first_positive_rank": (
            None
            if not np.any(selected_positive)
            else float(np.quantile(rank[selected_positive], 0.9))
        ),
    }


def _paired_rank(
    *,
    baseline_scores: np.ndarray,
    appearance_scores: np.ndarray,
    labels: np.ndarray,
    candidate_valid: np.ndarray,
    row_mask: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    appearance = np.asarray(appearance_scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    selected = np.asarray(row_mask, dtype=bool).reshape(-1).copy()
    if (
        baseline.shape != appearance.shape
        or baseline.shape != positive.shape
        or positive.shape != valid.shape
        or selected.shape != (len(baseline),)
    ):
        raise ValueError("paired rank inputs are incompatible")
    label_in_valid = positive & valid
    candidate_rows = selected & np.any(valid, axis=1) & np.any(label_in_valid, axis=1)
    if not np.any(candidate_rows):
        return {
            "positive_row_count": 0,
            "rank_win_count": 0,
            "rank_loss_count": 0,
            "rank_tie_count": 0,
            "median_rank_delta_baseline_minus_appearance": None,
            "top1_rescue_count": 0,
            "top1_harm_count": 0,
        }
    _base_top, base_rank = _top_and_positive_rank(baseline, label_in_valid, valid)
    _appearance_top, appearance_rank = _top_and_positive_rank(
        appearance, label_in_valid, valid
    )
    base = base_rank[candidate_rows]
    probe = appearance_rank[candidate_rows]
    return {
        "positive_row_count": int(len(base)),
        "rank_win_count": int(np.sum(probe < base)),
        "rank_loss_count": int(np.sum(probe > base)),
        "rank_tie_count": int(np.sum(probe == base)),
        "median_rank_delta_baseline_minus_appearance": float(np.median(base - probe)),
        "top1_rescue_count": int(np.sum((base > 1) & (probe == 1))),
        "top1_harm_count": int(np.sum((base == 1) & (probe > 1))),
    }


def _predeclared_identity_gate(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Apply a deliberately conservative, non-promotion candidate gate.

    Every comparison is restricted to the same rows/candidates with usable
    appearance evidence.  This avoids treating crop availability as an
    improvement.  Passing the gate only justifies a later *independent* split
    check; it never authorizes fitting or pose selection on this audit split.
    """

    exact = audit.get("exact_registered_identity")
    if not isinstance(exact, Mapping):
        raise ValueError("candidate identity gate lacks exact identity metrics")
    baseline = exact.get("baseline_common_coverage")
    appearance = exact.get("appearance")
    paired = exact.get("paired_rank")
    if not isinstance(baseline, Mapping) or not isinstance(appearance, Mapping) or not isinstance(
        paired, Mapping
    ):
        raise ValueError("candidate identity gate has incomplete metrics")
    baseline_median = baseline.get("median_first_positive_rank")
    appearance_median = appearance.get("median_first_positive_rank")
    baseline_p90 = baseline.get("p90_first_positive_rank")
    appearance_p90 = appearance.get("p90_first_positive_rank")
    baseline_top1 = baseline.get("top1_positive_rate_given_positive")
    appearance_top1 = appearance.get("top1_positive_rate_given_positive")
    comparable = all(
        value is not None
        for value in (
            baseline_median,
            appearance_median,
            baseline_p90,
            appearance_p90,
            baseline_top1,
            appearance_top1,
        )
    )
    checks = {
        "comparable_exact_identity_rows": bool(comparable),
        "median_rank_strictly_improved": bool(
            comparable and float(appearance_median) < float(baseline_median)
        ),
        "p90_rank_not_worse": bool(
            comparable and float(appearance_p90) <= float(baseline_p90)
        ),
        "top1_not_worse": bool(
            comparable and float(appearance_top1) >= float(baseline_top1)
        ),
        "paired_wins_exceed_losses": int(paired.get("rank_win_count", 0))
        > int(paired.get("rank_loss_count", 0)),
        "top1_rescues_exceed_harms": int(paired.get("top1_rescue_count", 0))
        > int(paired.get("top1_harm_count", 0)),
    }
    return {
        "policy": (
            "validation-only raw-feature gate; pass is necessary but not sufficient "
            "for train-only calibration or any pose-level promotion"
        ),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _family_audit(
    *,
    name: str,
    aggregation: str,
    baseline_scores: np.ndarray,
    appearance_scores: np.ndarray,
    usable_mass: np.ndarray,
    track_valid: np.ndarray,
    geometry_labels: Mapping[float, np.ndarray],
    exact_labels: np.ndarray,
    exact_supervised: np.ndarray,
    split_mask: np.ndarray,
) -> dict[str, Any]:
    raw = np.asarray(appearance_scores, dtype=np.float64)
    mass = np.asarray(usable_mass, dtype=np.float64)
    tracks = np.asarray(track_valid, dtype=bool)
    evidence_valid = tracks & (mass > 0.0) & np.isfinite(raw)
    selected = np.asarray(split_mask, dtype=bool)
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    output: dict[str, Any] = {
        "family": str(name),
        "aggregation": str(aggregation),
        "coverage": {
            "row_count": int(np.sum(selected)),
            "candidate_usable_rate": float(np.mean(evidence_valid[selected])),
            "row_all_topl_usable_rate": float(np.mean(np.all(evidence_valid[selected], axis=1))),
            "row_any_usable_rate": float(np.mean(np.any(evidence_valid[selected], axis=1))),
            "mean_usable_view_weight_mass": float(np.mean(mass[selected & np.any(tracks, axis=1)])),
        },
        "geometry_by_threshold_px": {},
    }
    for threshold, labels in geometry_labels.items():
        raw_labels = np.asarray(labels, dtype=bool)
        usable_labels = raw_labels & evidence_valid
        metric_rows = selected & np.any(evidence_valid, axis=1)
        output["geometry_by_threshold_px"][str(threshold)] = {
            "baseline_common_coverage": _rank_metrics(
                scores=baseline,
                labels=usable_labels,
                candidate_valid=evidence_valid,
                row_mask=metric_rows,
            ),
            "appearance": _rank_metrics(
                scores=raw,
                labels=usable_labels,
                candidate_valid=evidence_valid,
                row_mask=metric_rows,
            ),
            "paired_rank": _paired_rank(
                baseline_scores=baseline,
                appearance_scores=raw,
                labels=raw_labels,
                candidate_valid=evidence_valid,
                row_mask=metric_rows,
            ),
        }
    exact_rows = selected & exact_supervised & np.any(evidence_valid, axis=1)
    exact_usable = np.asarray(exact_labels, dtype=bool) & evidence_valid
    output["exact_registered_identity"] = {
        "baseline_common_coverage": _rank_metrics(
            scores=baseline,
            labels=exact_usable,
            candidate_valid=evidence_valid,
            row_mask=exact_rows,
        ),
        "appearance": _rank_metrics(
            scores=raw,
            labels=exact_usable,
            candidate_valid=evidence_valid,
            row_mask=exact_rows,
        ),
        "paired_rank": _paired_rank(
            baseline_scores=baseline,
            appearance_scores=raw,
            labels=np.asarray(exact_labels, dtype=bool),
            candidate_valid=evidence_valid,
            row_mask=exact_rows,
        ),
    }
    rank2_rows = exact_rows.copy()
    if np.any(rank2_rows):
        _top, baseline_rank = _top_and_positive_rank(
            baseline,
            exact_usable,
            evidence_valid,
        )
        rank2_rows &= baseline_rank > 1
    output["exact_registered_rank2_to_l"] = {
        "paired_rank": _paired_rank(
            baseline_scores=baseline,
            appearance_scores=raw,
            labels=np.asarray(exact_labels, dtype=bool),
            candidate_valid=evidence_valid,
            row_mask=rank2_rows,
        )
    }
    output["predeclared_identity_gate"] = _predeclared_identity_gate(output)
    return output


def audit_frozen_multiscale_candidate_appearance(
    *,
    appearance_artifacts: Sequence[Path],
    proposals: Path,
    colmap_model_dir: Path,
    output_dir: Path,
    geometric_thresholds_px: Sequence[float],
    exact_identity_radius_px: float,
    audit_splits: Sequence[str],
    allow_diagnostic_artifact: bool,
    force: bool,
) -> dict[str, Any]:
    """Measure frozen appearance ranks; target data is confined to this function."""

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()) and not bool(force):
        raise FileExistsError("refusing to overwrite frozen appearance audit output")
    if not (0.0 < float(exact_identity_radius_px)):
        raise ValueError("exact identity radius must be positive")
    arrays, shard_metadata, keys = _merge_appearance(
        tuple(Path(path) for path in appearance_artifacts),
        allow_diagnostic_artifact=bool(allow_diagnostic_artifact),
    )
    proposal_path = Path(proposals)
    with np.load(proposal_path, allow_pickle=False) as payload:
        required = ("query_ids", "candidate_track_ids", "candidate_gt_residuals_px")
        missing = sorted(set(required).difference(payload.files))
        if missing:
            raise ValueError(f"proposal target artifact lacks {missing}")
        proposal_ids = np.asarray(payload["query_ids"]).astype(str)
        proposal_tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64)
        residuals = np.asarray(payload["candidate_gt_residuals_px"], dtype=np.float32)
    source_rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64)
    query_ids = np.asarray(arrays["verification_query_ids"]).astype(str)
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    if (
        np.any(source_rows < 0)
        or np.any(source_rows >= len(proposal_ids))
        or not np.array_equal(proposal_ids[source_rows], query_ids)
        or not np.array_equal(proposal_tracks[source_rows], tracks)
        or residuals.shape != proposal_tracks.shape
    ):
        raise ValueError("frozen appearance tracks do not match proposal target rows")
    split_names = np.asarray(arrays["split_names"]).astype(str)
    selected_splits = tuple(str(value) for value in audit_splits)
    if set(split_names.tolist()) - {"validation", "test"}:
        raise ValueError("appearance artifacts contain a non-held-out split")
    if not set(split_names.tolist()) & set(selected_splits):
        raise ValueError("appearance artifacts have no requested audit split")
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    targets = registered_query_observation_targets(
        query_ids=query_ids,
        query_xy=np.asarray(arrays["verification_xy"], dtype=np.float32),
        images_by_name=images_by_name,
        max_distance_px=float(exact_identity_radius_px),
    )
    exact_labels = registered_candidate_identity_labels(tracks, targets)
    identity_coverage = summarize_registered_candidate_identity(exact_labels, targets)
    geometry_labels = {
        float(threshold): np.isfinite(residuals[source_rows])
        & (residuals[source_rows] <= float(threshold))
        & (tracks >= 0)
        for threshold in geometric_thresholds_px
    }
    baseline_scores = np.log(
        np.maximum(np.asarray(arrays["candidate_probabilities"], dtype=np.float64), 1e-12)
    )
    track_valid = tracks >= 0
    families = np.asarray(arrays["family_names"]).astype(str)
    weighted = np.asarray(arrays["candidate_coverage_weighted_ncc"], dtype=np.float32)
    maximum = np.asarray(arrays["candidate_max_view_ncc"], dtype=np.float32)
    mass = np.asarray(arrays["candidate_usable_view_weight_mass"], dtype=np.float32)
    result: dict[str, Any] = {
        "stage": "audit_frozen_multiscale_candidate_absolute_appearance",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "appearance_artifacts": [
            {"path": str(path), "sha256": file_sha256_short(path)}
            for path in appearance_artifacts
        ],
        "proposals": {"path": str(proposal_path), "sha256": file_sha256_short(proposal_path)},
        "colmap_model_dir": str(Path(colmap_model_dir)),
        "row_count": int(len(query_ids)),
        "query_count": int(len(set(query_ids.tolist()))),
        "audit_splits": list(selected_splits),
        "geometric_thresholds_px": [float(value) for value in geometric_thresholds_px],
        "exact_identity_radius_px": float(exact_identity_radius_px),
        "registered_identity_target_coverage": identity_coverage,
        "families": {},
        "predeclared_gate_summary": {},
        "protocol": {
            "targets_loaded_only_in_audit": True,
            "fitting_or_calibration": False,
            "candidate_reselection": False,
            "posterior_update": False,
            "pose_scoring": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    per_rows: list[dict[str, Any]] = []
    for split_name in selected_splits:
        split_mask = split_names == str(split_name)
        if not np.any(split_mask):
            continue
        for family_index, family in enumerate(families.tolist()):
            for aggregation, scores in (
                ("coverage_weighted_ncc", weighted[..., family_index]),
                ("max_view_ncc", maximum[..., family_index]),
            ):
                audit = _family_audit(
                    name=str(family),
                    aggregation=aggregation,
                    baseline_scores=baseline_scores,
                    appearance_scores=scores,
                    usable_mass=mass[..., family_index],
                    track_valid=track_valid,
                    geometry_labels=geometry_labels,
                    exact_labels=exact_labels,
                    exact_supervised=targets.supervised,
                    split_mask=split_mask,
                )
                result["families"].setdefault(str(family), {}).setdefault(
                    aggregation, {}
                )[str(split_name)] = audit
                result["predeclared_gate_summary"].setdefault(str(family), {}).setdefault(
                    aggregation, {}
                )[str(split_name)] = audit["predeclared_identity_gate"]
                exact = audit["exact_registered_identity"]
                paired = exact["paired_rank"]
                per_rows.append(
                    {
                        "split_name": str(split_name),
                        "family": str(family),
                        "aggregation": aggregation,
                        "exact_baseline_top1": exact["baseline_common_coverage"].get(
                            "top1_positive_rate_given_positive"
                        ),
                        "exact_appearance_top1": exact["appearance"].get(
                            "top1_positive_rate_given_positive"
                        ),
                        "exact_baseline_median_rank": exact["baseline_common_coverage"].get(
                            "median_first_positive_rank"
                        ),
                        "exact_appearance_median_rank": exact["appearance"].get(
                            "median_first_positive_rank"
                        ),
                        "exact_rank_wins": paired.get("rank_win_count"),
                        "exact_rank_losses": paired.get("rank_loss_count"),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    csv_path = output / "family_summary.csv"
    fieldnames = list(per_rows[0]) if per_rows else ["split_name", "family", "aggregation"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_rows)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_frozen_multiscale_candidate_appearance(
        appearance_artifacts=_paths(args.appearance_artifacts),
        proposals=Path(args.proposals),
        colmap_model_dir=Path(args.colmap_model_dir),
        output_dir=Path(args.output_dir),
        geometric_thresholds_px=_thresholds(args.geometric_thresholds_px),
        exact_identity_radius_px=float(args.exact_identity_radius_px),
        audit_splits=_splits(args.audit_splits),
        allow_diagnostic_artifact=bool(args.allow_diagnostic_artifact),
        force=bool(args.force),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
