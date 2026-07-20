"""Map a frozen full-track candidate probe back onto pose-scoring rows.

The full-track appearance probe is evaluated on fixed held-out detector rows,
whereas independent pose scoring consumes one full proposal-row posterior.  A
bridge must therefore be stricter than a generic posterior export: it may only
replace the exact validation rows that were frozen before labels were joined,
and every replacement must agree with the immutable proposal track identity
and baseline posterior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_PREDICTION_FORMAT,
    FrozenFulltrackPerViewAppearanceFeatures,
    fulltrack_per_view_feature_granularity,
)


FROZEN_FULLTRACK_PER_VIEW_POSE_OVERLAY_FORMAT = (
    "frozen_fulltrack_per_view_candidate_probe_prior_overlay_v1"
)
IDENTITY_PROBABILITY_SEMANTICS = (
    "candidate_identity_probability_plus_explicit_null_equals_one"
)


def _array_sha256_short(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    value = payload.get("metadata_json")
    if value is None:
        raise ValueError(f"{context} lacks metadata_json")
    metadata = json.loads(str(np.asarray(value).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} metadata_json must contain an object")
    return metadata


def _load_prediction_family(
    *,
    path: Path,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    family: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load one train-frozen family without touching audit labels."""

    required = {
        "query_ids",
        "split_names",
        "source_row_indices",
        "candidate_track_ids",
        "family_names",
        "candidate_probabilities",
        "null_probabilities",
        "candidate_residuals",
        "baseline_candidate_probabilities",
        "baseline_null_probabilities",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path}: full-track prediction lacks {missing}")
        arrays = {key: np.asarray(payload[key]).copy() for key in required}
    metadata = _metadata(arrays, context=str(path))
    if (
        metadata.get("format") != FULLTRACK_PER_VIEW_PREDICTION_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("training_supervision_split") != "train"
        or metadata.get("validation_or_test_labels_used_by_fit") is not False
        or metadata.get("prediction_frozen_before_validation_target_join") is not True
        or metadata.get("fixed_global_top_l") != 20
        or metadata.get("candidate_reselection") is not False
        or metadata.get("support_reselection") is not False
        or metadata.get("support_view_selection") is not False
        or metadata.get("fulltrack_compatibility") != dict(features.compatibility)
        or metadata.get("feature_granularity")
        != fulltrack_per_view_feature_granularity(features)
        or not str(metadata.get("identity_supervision_colmap_images_sha256", "")).strip()
    ):
        raise ValueError("full-track prediction violates the frozen pose-overlay protocol")
    if (
        not np.array_equal(np.asarray(arrays["query_ids"]).astype(str), features.query_ids)
        or not np.array_equal(
            np.asarray(arrays["split_names"]).astype(str), features.split_names
        )
        or not np.array_equal(
            np.asarray(arrays["source_row_indices"], dtype=np.int64),
            features.source_row_indices,
        )
        or not np.array_equal(
            np.asarray(arrays["candidate_track_ids"], dtype=np.int64),
            features.candidate_track_ids,
        )
    ):
        raise ValueError("full-track prediction is not row-aligned with appearance data")
    baseline_candidate = np.asarray(
        arrays["baseline_candidate_probabilities"], dtype=np.float32
    )
    baseline_null = np.asarray(arrays["baseline_null_probabilities"], dtype=np.float32)
    if (
        baseline_candidate.shape != features.candidate_probabilities.shape
        or baseline_null.shape != features.null_probabilities.shape
        or np.max(np.abs(baseline_candidate - features.candidate_probabilities)) > 2e-6
        or np.max(np.abs(baseline_null - features.null_probabilities)) > 2e-6
    ):
        raise ValueError("full-track prediction baseline differs from frozen appearance")
    family_names = tuple(str(value) for value in np.asarray(arrays["family_names"]).tolist())
    if (
        str(family) not in FULLTRACK_PER_VIEW_FAMILIES
        or len(set(family_names)) != len(family_names)
        or str(family) not in family_names
    ):
        raise ValueError("requested full-track family is absent from prediction")
    family_contracts = metadata.get("family_evidence_contracts")
    family_architectures = metadata.get("family_architectures")
    family_edge_feature_semantics = metadata.get("family_edge_feature_semantics")
    if (
        not isinstance(family_contracts, Mapping)
        or not isinstance(family_architectures, Mapping)
        or not isinstance(family_edge_feature_semantics, Mapping)
        or set(family_contracts) != set(family_names)
        or set(family_architectures) != set(family_names)
        or set(family_edge_feature_semantics) != set(family_names)
        or family_architectures.get(str(family))
        != FULLTRACK_PER_VIEW_FAMILIES[str(family)].architecture
        or family_edge_feature_semantics.get(str(family))
        != FULLTRACK_PER_VIEW_FAMILIES[str(family)].edge_feature_semantics
        or family_edge_feature_semantics.get(str(family))
        != features.compatibility.get("per_view_edge_feature_semantics")
        or not isinstance(family_contracts.get(str(family)), Mapping)
    ):
        raise ValueError("full-track prediction lacks the requested family contract")
    family_index = family_names.index(str(family))
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32)
    residual = np.asarray(arrays["candidate_residuals"], dtype=np.float32)
    expected_shape = (len(family_names), *features.candidate_probabilities.shape)
    if (
        candidate.shape != expected_shape
        or null.shape != (len(family_names), len(features.query_ids))
        or residual.shape != expected_shape
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(null))
        or np.any(candidate < 0.0)
        or np.any(null < 0.0)
        or np.max(np.abs(candidate.sum(axis=2) + null - 1.0)) > 2e-5
        or np.max(np.abs(null - features.null_probabilities[None, :])) > 0.0
        or np.any(candidate[:, features.candidate_probabilities <= 0.0] > 2e-6)
    ):
        raise ValueError("full-track prediction posterior is invalid")
    return (
        candidate[family_index],
        null[family_index],
        metadata,
        dict(family_contracts[str(family)]),
    )


def _load_base_overlay(
    *, path: Path, proposals_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Load the immutable full proposal posterior and its exact row identity."""

    required = {
        "candidate_track_ids",
        "candidate_probabilities",
        "null_probabilities",
        "metadata_json",
    }
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path}: base prior overlay lacks {missing}")
        arrays = {key: np.asarray(payload[key]).copy() for key in required}
    metadata = _metadata(arrays, context=str(path))
    if (
        metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("probability_semantics") != IDENTITY_PROBABILITY_SEMANTICS
        or str(metadata.get("proposals_sha256")) != str(file_sha256_short(proposals_path))
    ):
        raise ValueError("base prior overlay violates the immutable proposal contract")
    with np.load(Path(proposals_path), allow_pickle=False) as payload:
        required_proposal = {"query_ids", "candidate_track_ids"}
        missing = sorted(required_proposal.difference(payload.files))
        if missing:
            raise ValueError(f"{proposals_path}: proposal artifact lacks {missing}")
        proposal_query_ids = np.asarray(payload["query_ids"]).astype(str).copy()
        proposal_tracks = np.asarray(payload["candidate_track_ids"], dtype=np.int64).copy()
    tracks = np.asarray(arrays["candidate_track_ids"], dtype=np.int64)
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32).reshape(-1)
    valid = tracks >= 0
    if (
        tracks.shape != proposal_tracks.shape
        or not np.array_equal(tracks, proposal_tracks)
        or proposal_query_ids.shape != (len(tracks),)
        or candidate.shape != tracks.shape
        or null.shape != (len(tracks),)
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(null))
        or np.any((candidate < 0.0) | (candidate > 1.0))
        or np.any((null < 0.0) | (null > 1.0))
        or np.any(np.abs(candidate[~valid]) > 1e-6)
        or np.max(np.abs(candidate.sum(axis=1) + null - 1.0)) > 1e-4
    ):
        raise ValueError("base prior overlay posterior is invalid")
    return tracks, candidate, null, metadata, proposal_query_ids


def build_frozen_fulltrack_per_view_validation_pose_overlay(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    predictions_path: Path,
    base_prior_overlay_path: Path,
    proposals_path: Path,
    family: str,
    output_path: Path,
) -> dict[str, Any]:
    """Copy the base posterior and replace only frozen validation rows.

    The function intentionally has no target, hypothesis, pose, image, or
    measurement inputs.  It is an identity-posterior bridge only.
    """

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing pose overlay: {output}")
    predicted_candidate, predicted_null, prediction_metadata, family_contract = (
        _load_prediction_family(
            path=Path(predictions_path), features=features, family=str(family)
        )
    )
    base_tracks, base_candidate, base_null, base_metadata, proposal_query_ids = (
        _load_base_overlay(
            path=Path(base_prior_overlay_path), proposals_path=Path(proposals_path)
        )
    )
    validation = np.asarray(features.split_names == "validation", dtype=bool)
    if not np.any(validation) or np.any(features.split_names == "test"):
        raise ValueError("full-track pose overlay requires train/validation-only features")
    update_rows = np.asarray(features.source_row_indices[validation], dtype=np.int64)
    if (
        len(np.unique(update_rows)) != len(update_rows)
        or np.any(update_rows < 0)
        or np.any(update_rows >= len(base_tracks))
        or not np.array_equal(proposal_query_ids[update_rows], features.query_ids[validation])
        or not np.array_equal(base_tracks[update_rows], features.candidate_track_ids[validation])
        or np.max(
            np.abs(base_candidate[update_rows] - features.candidate_probabilities[validation])
        )
        > 2e-6
        or np.max(np.abs(base_null[update_rows] - features.null_probabilities[validation]))
        > 2e-6
    ):
        raise ValueError("validation full-track rows do not map exactly to base proposals")
    output_candidate = np.array(base_candidate, copy=True)
    output_null = np.array(base_null, copy=True)
    output_candidate[update_rows] = predicted_candidate[validation]
    output_null[update_rows] = predicted_null[validation]
    valid = base_tracks >= 0
    maximum_mass_error = float(
        np.max(np.abs(output_candidate.sum(axis=1, dtype=np.float64) + output_null - 1.0))
    )
    unchanged = np.ones((len(base_tracks),), dtype=bool)
    unchanged[update_rows] = False
    if (
        maximum_mass_error > 3e-5
        or np.any(np.abs(output_candidate[~valid]) > 1e-6)
        or not np.array_equal(output_candidate[unchanged], base_candidate[unchanged])
        or not np.array_equal(output_null[unchanged], base_null[unchanged])
    ):
        raise RuntimeError("full-track pose overlay changed immutable proposal rows")
    metadata: dict[str, Any] = {
        "format": FROZEN_FULLTRACK_PER_VIEW_POSE_OVERLAY_FORMAT,
        "contains_ground_truth": False,
        "contains_target_errors": False,
        "pose_or_ground_truth_used_for_prediction": False,
        "probability_semantics": IDENTITY_PROBABILITY_SEMANTICS,
        "proposals_sha256": file_sha256_short(proposals_path),
        "base_prior_overlay": str(Path(base_prior_overlay_path)),
        "base_prior_overlay_sha256": file_sha256_short(base_prior_overlay_path),
        "base_prior_format": base_metadata.get("format"),
        "predictions": str(Path(predictions_path)),
        "predictions_sha256": file_sha256_short(predictions_path),
        "prediction_format": prediction_metadata.get("format"),
        "family": str(family),
        "family_contract": family_contract,
        "fulltrack_compatibility": dict(features.compatibility),
        "feature_granularity": fulltrack_per_view_feature_granularity(features),
        "identity_supervision_colmap_images_bin": prediction_metadata.get(
            "identity_supervision_colmap_images_bin"
        ),
        "identity_supervision_colmap_images_sha256": prediction_metadata.get(
            "identity_supervision_colmap_images_sha256"
        ),
        "training_supervision_split": prediction_metadata.get(
            "training_supervision_split"
        ),
        "validation_or_test_labels_used_by_fit": False,
        "prediction_frozen_before_validation_target_join": True,
        "fixed_global_top_l": True,
        "candidate_retrieval_or_reselection": False,
        "updated_split_names": ["validation"],
        "updated_source_row_count": int(len(update_rows)),
        "updated_source_rows_sha256": _array_sha256_short(update_rows),
        "train_prediction_rows_overwritten": False,
        "test_prediction_rows_materialized": False,
        "nonvalidation_base_rows_unchanged": True,
        "maximum_probability_mass_error": maximum_mass_error,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "pose_scoring": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            candidate_track_ids=base_tracks,
            candidate_probabilities=output_candidate.astype(np.float32),
            null_probabilities=output_null.astype(np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    summary = {
        "stage": "build_frozen_fulltrack_per_view_validation_pose_overlay",
        "protocol": {
            "validation_rows_only": True,
            "train_rows_left_at_base_posterior": True,
            "test_rows_not_materialized": True,
            "fixed_global_top_l": True,
            "candidate_retrieval_or_reselection": False,
            "pose_or_ground_truth_used": False,
            "render": False,
        },
        "inputs": metadata,
        "outputs": {
            "overlay": str(output),
            "overlay_sha256": file_sha256_short(output),
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
