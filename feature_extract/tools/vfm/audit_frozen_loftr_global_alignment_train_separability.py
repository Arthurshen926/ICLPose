"""Measure raw LoFTR global-alignment separability on train identities only.

This is a smoke-stage diagnostic, not a model fit and not a validation result.
It first verifies that the alignment shards are exact target-free overlays of
their frozen S0 rows.  Only then does it join *train* registered SfM identities
to report whether a predeclared fixed-view aggregate of each raw error feature
contains even a minimally plausible candidate-level signal.  It never writes
per-row labels and never accesses validation or pose targets.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_multiscale_candidate_appearance_residual import (
    _paired_rank,
    _rank_metrics,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.frozen_loftr_global_alignment import (
    FROZEN_LOFTR_GLOBAL_ALIGNMENT_EVIDENCE_FORMAT,
    LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES,
)
from feature_extract.vfm.localization.frozen_multiscale_candidate_appearance_residual_probe import (
    FROZEN_APPEARANCE_ARTIFACT_FORMAT,
)
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


FROZEN_LOFTR_GLOBAL_ALIGNMENT_TRAIN_AUDIT_FORMAT = (
    "frozen_loftr_global_alignment_train_separability_audit_v1"
)
_BASE_FIELDS = (
    "verification_query_ids",
    "split_names",
    "verification_source_row_indices",
    "verification_xy",
    "candidate_track_ids",
    "candidate_probabilities",
    "null_probabilities",
    "candidate_view_weights",
)
_RAW_ERROR_FEATURES = (
    "loftr_homography_anchor_forward_error_px",
    "loftr_homography_anchor_symmetric_error_px",
    "loftr_homography_anchor_nearest_inlier_support_px",
)


@dataclass(frozen=True)
class FrozenGlobalAlignmentEvidence:
    paths: tuple[Path, ...]
    query_ids: np.ndarray
    xy: np.ndarray
    candidate_track_ids: np.ndarray
    candidate_probabilities: np.ndarray
    candidate_view_weights: np.ndarray
    candidate_view_usable: np.ndarray
    candidate_view_features: np.ndarray
    feature_names: tuple[str, ...]
    artifacts: tuple[dict[str, Any], ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-artifacts", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--registered-identity-radius-px", type=float, default=2.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths) or any(not path.is_file() for path in paths):
        raise ValueError("alignment artifacts must be unique existing files")
    return paths


def _metadata(payload: Mapping[str, np.ndarray], *, path: Path) -> dict[str, Any]:
    value = payload.get("metadata_json")
    if value is None:
        raise ValueError(f"{path}: alignment artifact lacks metadata")
    metadata = json.loads(str(np.asarray(value).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: alignment metadata is invalid")
    return metadata


def _validate_direct_source(*, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], path: Path) -> None:
    inputs = metadata.get("inputs")
    source = inputs.get("appearance_artifact") if isinstance(inputs, Mapping) else None
    if not isinstance(source, Mapping):
        raise ValueError(f"{path}: frozen direct S0 source provenance is absent")
    source_path = Path(str(source.get("path", "")))
    if not source_path.is_file() or str(source.get("sha256", "")) != file_sha256_short(source_path):
        raise ValueError(f"{path}: frozen direct S0 source is stale or absent")
    with np.load(source_path, allow_pickle=False) as payload:
        source_metadata = _metadata(payload, path=source_path)
        source_arrays = {name: np.asarray(payload[name]).copy() for name in _BASE_FIELDS}
    if (
        source_metadata.get("format") != FROZEN_APPEARANCE_ARTIFACT_FORMAT
        or source_metadata.get("contains_target_fields") is not False
        or source_metadata.get("pose_or_ground_truth_used") is not False
        or source_metadata.get("supervision_arrays_loaded") is not False
        or any(not np.array_equal(arrays[name], source_arrays[name]) for name in _BASE_FIELDS)
    ):
        raise ValueError(f"{path}: global alignment artifact changed its frozen direct S0 rows")


def _load_alignment_artifacts(paths: Sequence[Path]) -> FrozenGlobalAlignmentEvidence:
    """Fail closed on any non-target-free or misaligned smoke artifact."""

    loaded: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "verification_query_ids",
            "verification_xy",
            "candidate_track_ids",
            "candidate_probabilities",
            "candidate_view_weights",
            "candidate_view_usable",
            "candidate_view_features",
        )
    }
    artifacts: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    reference_names: tuple[str, ...] | None = None
    reference_config: Mapping[str, Any] | None = None
    for path in tuple(Path(item) for item in paths):
        required = (
            *_BASE_FIELDS,
            "feature_names",
            "candidate_view_usable",
            "candidate_view_features",
            "candidate_view_pair_match_counts",
            "candidate_view_homography_model_valid",
            "candidate_usable_view_weight_mass",
            "metadata_json",
        )
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted(set(required).difference(payload.files))
            if missing:
                raise ValueError(f"{path}: alignment artifact lacks {missing}")
            arrays = {name: np.asarray(payload[name]).copy() for name in required if name != "metadata_json"}
            metadata = _metadata(payload, path=path)
        contract = metadata.get("strict_frozen_loftr_global_alignment_contract")
        expected_contract = {
            "heldout_s0_verification_rows": True,
            "fixed_global_topl": True,
            "candidate_identity_fixed": True,
            "candidate_3d_projection_or_pose_used": False,
            "candidate_reselection": False,
            "support_reselection": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "all_mapping_images_pair_cached": True,
            "pair_cache_image_level_selection": False,
            "global_alignment_pose_free": True,
            "global_alignment_model_per_candidate": False,
            "homography_fits_all_cached_pair_matches": True,
        }
        if (
            metadata.get("format") != FROZEN_LOFTR_GLOBAL_ALIGNMENT_EVIDENCE_FORMAT
            or metadata.get("contains_target_fields") is not False
            or metadata.get("pose_or_ground_truth_used") is not False
            or metadata.get("supervision_arrays_loaded") is not False
            or int(metadata.get("fixed_candidate_top_k", -1)) != 20
            or metadata.get("feature_field") != "candidate_view_features"
            or not isinstance(contract, Mapping)
            or any(contract.get(key) is not expected for key, expected in expected_contract.items())
        ):
            raise ValueError(f"{path}: global alignment target-free contract is invalid")
        query_ids = np.asarray(arrays["verification_query_ids"]).astype(str).reshape(-1)
        split_names = np.asarray(arrays["split_names"]).astype(str).reshape(-1)
        source_rows = np.asarray(arrays["verification_source_row_indices"], dtype=np.int64).reshape(-1)
        xy = np.asarray(arrays["verification_xy"], dtype=np.float32)
        tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
        probabilities = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
        null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
        weights = np.asarray(arrays["candidate_view_weights"], dtype=np.float32)
        usable = np.asarray(arrays["candidate_view_usable"], dtype=bool)
        values = np.asarray(arrays["candidate_view_features"], dtype=np.float32)
        pair_counts = np.asarray(arrays["candidate_view_pair_match_counts"], dtype=np.int32)
        model_valid = np.asarray(arrays["candidate_view_homography_model_valid"], dtype=bool)
        usable_mass = np.asarray(arrays["candidate_usable_view_weight_mass"], dtype=np.float32)
        names = tuple(np.asarray(arrays["feature_names"]).astype(str).reshape(-1).tolist())
        if (
            len(query_ids) != 192
            or len(set(query_ids.tolist())) != 1
            or query_ids[0] in seen_queries
            or set(split_names.tolist()) != {"train"}
            or source_rows.shape != (192,)
            or xy.shape != (192, 2)
            or tracks.shape != (192, 20)
            or probabilities.shape != tracks.shape
            or null.shape != (192,)
            or weights.ndim != 3
            or weights.shape[:2] != tracks.shape
            or usable.shape != weights.shape
            or values.shape != (*weights.shape, len(LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES))
            or pair_counts.shape != weights.shape
            or model_valid.shape != weights.shape
            or usable_mass.shape != tracks.shape
            or names != LOFTR_GLOBAL_ALIGNMENT_FEATURE_NAMES
            or np.any(~np.isfinite(xy))
            or np.any(~np.isfinite(probabilities))
            or np.any(~np.isfinite(null))
            or np.any(~np.isfinite(weights))
            or np.any(probabilities < 0.0)
            or np.any(null <= 0.0)
            or np.max(np.abs(probabilities.sum(axis=1, dtype=np.float64) + null - 1.0)) > 2e-4
            or np.any(~np.isfinite(values[usable]))
            or np.any(usable & ~model_valid)
            or np.any(pair_counts[model_valid] <= 0)
            or not np.allclose(
                usable_mass, (weights * usable.astype(np.float32)).sum(axis=2), atol=2e-5
            )
        ):
            raise ValueError(f"{path}: global alignment tensors are invalid")
        _validate_direct_source(metadata=metadata, arrays=arrays, path=path)
        config = metadata.get("global_alignment_config")
        if reference_names is None:
            reference_names = names
            reference_config = config
        elif names != reference_names or config != reference_config:
            raise ValueError(f"{path}: global alignment configuration differs across artifacts")
        seen_queries.add(str(query_ids[0]))
        for name in loaded:
            loaded[name].append(
                {
                    "verification_query_ids": query_ids,
                    "verification_xy": xy,
                    "candidate_track_ids": tracks,
                    "candidate_probabilities": probabilities,
                    "candidate_view_weights": weights,
                    "candidate_view_usable": usable,
                    "candidate_view_features": values,
                }[name]
            )
        artifacts.append(
            {
                "path": str(path),
                "sha256": file_sha256_short(path),
                "query_id": str(query_ids[0]),
                "source_direct_artifact": dict(metadata["inputs"]["appearance_artifact"]),
            }
        )
    if reference_names is None:
        raise RuntimeError("global alignment audit received no artifacts")
    return FrozenGlobalAlignmentEvidence(
        paths=tuple(Path(item) for item in paths),
        query_ids=np.concatenate(loaded["verification_query_ids"], axis=0),
        xy=np.concatenate(loaded["verification_xy"], axis=0),
        candidate_track_ids=np.concatenate(loaded["candidate_track_ids"], axis=0),
        candidate_probabilities=np.concatenate(loaded["candidate_probabilities"], axis=0),
        candidate_view_weights=np.concatenate(loaded["candidate_view_weights"], axis=0),
        candidate_view_usable=np.concatenate(loaded["candidate_view_usable"], axis=0),
        candidate_view_features=np.concatenate(loaded["candidate_view_features"], axis=0),
        feature_names=reference_names,
        artifacts=tuple(artifacts),
    )


def fixed_view_weighted_negative_log1p_error(
    *,
    values: np.ndarray,
    usable: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predeclared raw score: fixed-mixture mean of negative log error.

    Unknown views retain no mass in this diagnostic score.  The returned mass
    makes that fact visible; this function neither selects a view nor changes
    any inference posterior.
    """

    error = np.asarray(values, dtype=np.float32)
    available = np.asarray(usable, dtype=bool)
    fixed_weights = np.asarray(weights, dtype=np.float32)
    if (
        error.shape != available.shape
        or fixed_weights.shape != error.shape
        or np.any(error[available] < 0.0)
        or np.any(~np.isfinite(error[available]))
        or np.any(fixed_weights < 0.0)
    ):
        raise ValueError("global alignment raw-score arrays are invalid")
    masked_weights = fixed_weights * available.astype(np.float32)
    mass = masked_weights.sum(axis=2)
    safe_error = np.where(available, error, 0.0)
    score = -np.sum(
        masked_weights * np.log1p(safe_error), axis=2
    ) / np.maximum(mass, 1e-12)
    score = np.where(mass > 0.0, score, np.nan)
    return score.astype(np.float32), mass.astype(np.float32)


def _score_quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"count": 0, "p10": None, "median": None, "p90": None}
    return {
        "count": int(len(finite)),
        "p10": float(np.quantile(finite, 0.1)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
    }


def audit_frozen_loftr_global_alignment_train_separability(
    *,
    alignment_artifacts: Sequence[Path],
    colmap_model_dir: Path,
    registered_identity_radius_px: float,
    output_json: Path,
) -> dict[str, Any]:
    """Run a train-only raw-evidence smoke audit after target-free export."""

    output = Path(output_json)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite train-only audit: {output}")
    if float(registered_identity_radius_px) <= 0.0:
        raise ValueError("registered identity radius must be positive")
    evidence = _load_alignment_artifacts(tuple(Path(item) for item in alignment_artifacts))
    images = read_colmap_images_binary(Path(colmap_model_dir) / "images.bin")
    images_by_name = {str(image.image_name): image for image in images.values()}
    targets = registered_query_observation_targets(
        query_ids=evidence.query_ids,
        query_xy=evidence.xy,
        images_by_name=images_by_name,
        max_distance_px=float(registered_identity_radius_px),
    )
    labels = registered_candidate_identity_labels(evidence.candidate_track_ids, targets)
    identity = summarize_registered_candidate_identity(labels, targets)
    base_valid = (
        (evidence.candidate_track_ids >= 0) & (evidence.candidate_probabilities > 0.0)
    )
    positive_rows = targets.supervised & np.any(labels, axis=1)
    baseline = np.log(np.maximum(evidence.candidate_probabilities, 1e-30))
    result: dict[str, Any] = {
        "format": FROZEN_LOFTR_GLOBAL_ALIGNMENT_TRAIN_AUDIT_FORMAT,
        "stage": "audit_frozen_loftr_global_alignment_train_separability",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "validation_or_test_targets_used": False,
        "training_model_fit": False,
        "artifacts": list(evidence.artifacts),
        "query_count": int(len(set(evidence.query_ids.tolist()))),
        "row_count": int(len(evidence.query_ids)),
        "registered_identity_radius_px": float(registered_identity_radius_px),
        "registered_identity_target_coverage": identity,
        "raw_score_definition": "fixed_view_weighted_negative_log1p_error_v1",
        "metrics": {},
        "protocol": {
            "target_free_export_verified_before_target_join": True,
            "targets_loaded_only_from_train_colmap_observations": True,
            "fixed_global_top_l": True,
            "candidate_reselection": False,
            "support_reselection": False,
            "support_view_marginalization": "fixed_maplet_weights_over_usable_views_diagnostic_only_v1",
            "pose_scoring": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
    }
    for name in _RAW_ERROR_FEATURES:
        feature_index = evidence.feature_names.index(name)
        raw_score, usable_mass = fixed_view_weighted_negative_log1p_error(
            values=evidence.candidate_view_features[..., feature_index],
            usable=evidence.candidate_view_usable,
            weights=evidence.candidate_view_weights,
        )
        candidate_valid = base_valid & np.isfinite(raw_score) & (usable_mass > 0.0)
        selected = positive_rows & np.any(candidate_valid, axis=1)
        probe_metrics = _rank_metrics(
            scores=raw_score,
            labels=labels,
            candidate_valid=candidate_valid,
            row_mask=selected,
        )
        baseline_metrics = _rank_metrics(
            scores=baseline,
            labels=labels,
            candidate_valid=candidate_valid,
            row_mask=selected,
        )
        paired = _paired_rank(
            baseline_scores=baseline,
            probe_scores=raw_score,
            labels=labels,
            candidate_valid=candidate_valid,
            row_mask=selected,
        )
        flat = candidate_valid & selected[:, None]
        result["metrics"][name] = {
            "feature_index": int(feature_index),
            "candidate_usable_view_weight_mass": _score_quantiles(usable_mass[base_valid]),
            "candidate_with_usable_evidence_rate": float(np.mean(candidate_valid[base_valid])),
            "correct_candidate_raw_score": _score_quantiles(raw_score[labels & candidate_valid]),
            "incorrect_candidate_raw_score": _score_quantiles(
                raw_score[(~labels) & candidate_valid]
            ),
            "baseline_under_same_coverage": baseline_metrics,
            "raw_feature_rank": probe_metrics,
            "paired_rank_against_mapper_posterior": paired,
            "edge_count": int(np.sum(flat)),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit_frozen_loftr_global_alignment_train_separability(
        alignment_artifacts=_paths(args.alignment_artifacts),
        colmap_model_dir=Path(args.colmap_model_dir),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        output_json=Path(args.output_json),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
