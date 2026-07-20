"""Bridge a frozen summary-top-four identity probe into pose scoring.

This bridge deliberately accepts only the bounded all-observation summary
diagnostic.  It copies an immutable full proposal posterior and replaces the
exact validation rows that were scored before validation identities were read.
It has no pose, hypothesis, RGB, or target inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.frozen_fulltrack_candidate_appearance_residual_probe import (
    FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_PREDICTION_FORMAT,
    FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES,
    FrozenFulltrackAppearanceFeatures,
    SUMMARY_TOP4_BALANCED_ARCHITECTURE,
    SUMMARY_TOP4_FEATURE_GRANULARITY,
)
from feature_extract.vfm.localization.frozen_fulltrack_per_view_pose_overlay import (
    IDENTITY_PROBABILITY_SEMANTICS,
    _array_sha256_short,
    _load_base_overlay,
)


FROZEN_FULLTRACK_SUMMARY_TOP4_POSE_OVERLAY_FORMAT = (
    "frozen_fulltrack_summary_top4_candidate_probe_prior_overlay_v1"
)


def _metadata(payload: Mapping[str, np.ndarray], *, context: str) -> dict[str, Any]:
    value = payload.get("metadata_json")
    if value is None:
        raise ValueError(f"{context} lacks metadata_json")
    metadata = json.loads(str(np.asarray(value).item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"{context} metadata_json must contain an object")
    return metadata


def _same_float(value: object, expected: float) -> bool:
    try:
        return float(value) == float(expected)
    except (TypeError, ValueError):
        return False


def _assert_summary_family_contract(
    *, family: str, architecture: object, contract: object
) -> dict[str, Any]:
    spec = FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES.get(str(family))
    if spec is None or not isinstance(contract, Mapping):
        raise ValueError("summary top-four prediction family is unsupported")
    expected_feature_names = [
        f"{profile}__uniform_top4_mean_ncc" for profile in spec.profile_names
    ]
    if (
        architecture != SUMMARY_TOP4_BALANCED_ARCHITECTURE
        or contract.get("architecture") != SUMMARY_TOP4_BALANCED_ARCHITECTURE
        or contract.get("candidate_evidence_transform")
        != "relu_positive_relative_summary_top4_uplift_tanh_bounded_traincal_v1"
        or contract.get("summary_statistic") != "uniform_top4_mean_ncc"
        or contract.get("profile_names") != list(spec.profile_names)
        or contract.get("profile_feature_names") != expected_feature_names
        or contract.get("training_objective")
        != "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1"
        or not _same_float(
            contract.get("rank2_hard_pair_weight"), spec.rank2_hard_pair_weight
        )
        or not _same_float(
            contract.get("coarse_top1_stability_weight"),
            spec.coarse_top1_stability_weight,
        )
        or not _same_float(contract.get("residual_scale"), spec.residual_scale)
        or not _same_float(contract.get("residual_cap"), spec.residual_cap)
        or contract.get("missing_evidence_semantics")
        != "common_top1_profile_missing_zero_residual_v1"
        or contract.get("null_handling")
        != "input_null_probability_exactly_preserved_v1"
        or contract.get("candidate_mass_handling")
        != "input_nonnull_mass_exactly_preserved_v1"
        or contract.get("per_view_model") is not False
    ):
        raise ValueError("summary top-four prediction family contract differs")
    return dict(contract)


def _load_prediction_family(
    *,
    path: Path,
    features: FrozenFulltrackAppearanceFeatures,
    family: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load exactly one train-frozen summary family without target joins."""

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
            raise ValueError(f"{path}: summary prediction lacks {missing}")
        arrays = {key: np.asarray(payload[key]).copy() for key in required}
    metadata = _metadata(arrays, context=str(path))
    if (
        metadata.get("format") != FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_PREDICTION_FORMAT
        or metadata.get("contains_ground_truth") is not False
        or metadata.get("contains_target_errors") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("training_supervision_split") != "train"
        or metadata.get("validation_or_test_labels_used_by_fit") is not False
        or metadata.get("prediction_frozen_before_validation_target_join") is not True
        or not str(metadata.get("identity_supervision_colmap_images_bin", "")).strip()
        or not str(metadata.get("identity_supervision_colmap_images_sha256", "")).strip()
        or metadata.get("fixed_global_top_l") != 20
        or metadata.get("candidate_reselection") is not False
        or metadata.get("support_reselection") is not False
        or metadata.get("support_view_selection") is not False
        or metadata.get("all_observation_aggregation_preserved") is not True
        or metadata.get("feature_granularity") != SUMMARY_TOP4_FEATURE_GRANULARITY
        or metadata.get("per_view_model") is not False
        or metadata.get("null_handling")
        != "input_null_probability_exactly_preserved_v1"
        or metadata.get("candidate_mass_handling")
        != "input_nonnull_mass_exactly_preserved_v1"
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("render") is not False
        or metadata.get("pose_scoring") is not False
        or metadata.get("fulltrack_compatibility") != dict(features.compatibility)
    ):
        raise ValueError("summary top-four prediction violates frozen overlay protocol")
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
        raise ValueError("summary top-four prediction is not row-aligned with features")
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
        raise ValueError("summary top-four prediction baseline differs from features")
    family_names = tuple(str(value) for value in np.asarray(arrays["family_names"]).tolist())
    architectures = metadata.get("family_architectures")
    contracts = metadata.get("family_evidence_contracts")
    if (
        str(family) not in FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES
        or not family_names
        or len(set(family_names)) != len(family_names)
        or not set(family_names).issubset(FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES)
        or str(family) not in family_names
        or not isinstance(architectures, Mapping)
        or not isinstance(contracts, Mapping)
        or set(architectures) != set(family_names)
        or set(contracts) != set(family_names)
    ):
        raise ValueError("summary top-four family is absent or contract is incomplete")
    family_contract = _assert_summary_family_contract(
        family=str(family),
        architecture=architectures.get(str(family)),
        contract=contracts.get(str(family)),
    )
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
        or np.any(~np.isfinite(residual))
        or np.any(candidate < 0.0)
        or np.any(null < 0.0)
        or np.max(np.abs(candidate.sum(axis=2) + null - 1.0)) > 2e-5
        or np.max(np.abs(null - features.null_probabilities[None, :])) > 0.0
        or np.any(candidate[:, features.candidate_probabilities <= 0.0] > 2e-6)
    ):
        raise ValueError("summary top-four prediction posterior is invalid")
    family_index = family_names.index(str(family))
    return candidate[family_index], null[family_index], metadata, family_contract


def build_frozen_fulltrack_summary_top4_validation_pose_overlay(
    *,
    features: FrozenFulltrackAppearanceFeatures,
    predictions_path: Path,
    base_prior_overlay_path: Path,
    proposals_path: Path,
    family: str,
    output_path: Path,
) -> dict[str, Any]:
    """Copy the base posterior and replace exact frozen validation rows only."""

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
        raise ValueError("summary top-four pose overlay requires train/validation-only features")
    update_rows = np.asarray(features.source_row_indices[validation], dtype=np.int64)
    if (
        len(np.unique(update_rows)) != len(update_rows)
        or np.any(update_rows < 0)
        or np.any(update_rows >= len(base_tracks))
        or not np.array_equal(
            proposal_query_ids[update_rows], features.query_ids[validation]
        )
        or not np.array_equal(
            base_tracks[update_rows], features.candidate_track_ids[validation]
        )
        or np.max(
            np.abs(
                base_candidate[update_rows]
                - features.candidate_probabilities[validation]
            )
        )
        > 2e-6
        or np.max(
            np.abs(base_null[update_rows] - features.null_probabilities[validation])
        )
        > 2e-6
    ):
        raise ValueError("summary top-four validation rows do not map to base proposals")
    output_candidate = np.array(base_candidate, copy=True)
    output_null = np.array(base_null, copy=True)
    output_candidate[update_rows] = predicted_candidate[validation]
    output_null[update_rows] = predicted_null[validation]
    valid = base_tracks >= 0
    maximum_mass_error = float(
        np.max(
            np.abs(output_candidate.sum(axis=1, dtype=np.float64) + output_null - 1.0)
        )
    )
    unchanged = np.ones((len(base_tracks),), dtype=bool)
    unchanged[update_rows] = False
    if (
        maximum_mass_error > 3e-5
        or np.any(np.abs(output_candidate[~valid]) > 1e-6)
        or not np.array_equal(output_candidate[unchanged], base_candidate[unchanged])
        or not np.array_equal(output_null[unchanged], base_null[unchanged])
    ):
        raise RuntimeError("summary top-four pose overlay changed immutable rows")
    metadata: dict[str, Any] = {
        "format": FROZEN_FULLTRACK_SUMMARY_TOP4_POSE_OVERLAY_FORMAT,
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
        "feature_granularity": SUMMARY_TOP4_FEATURE_GRANULARITY,
        "per_view_model": False,
        "training_supervision_split": prediction_metadata.get(
            "training_supervision_split"
        ),
        "identity_supervision_colmap_images_bin": prediction_metadata.get(
            "identity_supervision_colmap_images_bin"
        ),
        "identity_supervision_colmap_images_sha256": prediction_metadata.get(
            "identity_supervision_colmap_images_sha256"
        ),
        "validation_or_test_labels_used_by_fit": False,
        "prediction_frozen_before_validation_target_join": True,
        "fixed_global_top_l": 20,
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
        "stage": "build_frozen_fulltrack_summary_top4_validation_pose_overlay",
        "protocol": {
            "validation_rows_only": True,
            "train_rows_left_at_base_posterior": True,
            "test_rows_not_materialized": True,
            "fixed_global_top_l": 20,
            "candidate_retrieval_or_reselection": False,
            "pose_or_ground_truth_used": False,
            "render": False,
            "per_view_s2_claimed": False,
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
