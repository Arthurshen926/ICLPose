"""Audit train-frozen sparse per-view predictions on held-out identities.

The matching fit command is deliberately target-free for validation rows.  This
program first verifies immutable candidates, posterior mass, source lineage,
and prediction provenance, then joins audit-split identities only to answer an
S1 question: does a per-real-view mixture improve fixed top-20 identity rank?
Validation is the only split that can receive a candidate gate.  The optional
train mode is a diagnostic for overfitting versus missing capacity and is
hard-coded as non-promotable.  Neither mode forms or scores a pose hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_fulltrack_candidate_appearance import (
    paired_rank,
    rank_metrics,
)
from feature_extract.tools.vfm.audit_frozen_fulltrack_candidate_appearance_residual import (
    fulltrack_residual_candidate_gate,
)
from feature_extract.tools.vfm.audit_frozen_fulltrack_hard_pairs import (
    _validate_projected_bank_lineage,
    coherent_shift_mask,
    pairwise_feature_metrics,
    select_rank2_to_top1_wrong_pairs,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.frozen_fulltrack_per_view_candidate_probe import (
    FULLTRACK_PER_VIEW_FAMILIES,
    FULLTRACK_PER_VIEW_PREDICTION_FORMAT,
    FrozenFulltrackPerViewAppearanceFeatures,
    RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
    RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
    RAW_TOP4_SIGNED_ARCHITECTURE,
    RAW_TOPK_SUPPORT_VIEW_MARGINALIZATIONS,
    fulltrack_per_view_feature_granularity,
    load_frozen_fulltrack_per_view_appearance_features,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


MINIMUM_COHERENT_SHIFT_PAIRS_FOR_CLAIM = 50
_SPARSE_PER_VIEW_ARCHITECTURES = frozenset(
    {
        "sparse_per_view_mlp_learned_logsumexp_mixture_v1",
        "sparse_per_view_mlp_bounded_residual_learned_logsumexp_mixture_v1",
        "sparse_per_view_linear_learned_logsumexp_mixture_v1",
        "sparse_per_view_linear_bounded_residual_learned_logsumexp_mixture_v1",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-split", choices=("validation", "train"), default="validation")
    parser.add_argument("--registered-identity-radius-px", type=float, default=2.0)
    parser.add_argument("--minimum-shift-pairs", type=int, default=3)
    parser.add_argument("--minimum-shift-m", type=float, default=0.10)
    parser.add_argument("--maximum-shift-dispersion-m", type=float, default=0.20)
    parser.add_argument("--maximum-shift-relative-dispersion", type=float, default=0.35)
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("per-view appearance artifacts must be non-empty and unique")
    return paths


def _artifact_entries(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [{"path": str(path), "sha256": file_sha256_short(path)} for path in paths]


def _validate_raw_top4_evidence_contract(
    *,
    architecture: str,
    raw_topk_aggregation: str,
    contract: Mapping[str, Any],
) -> None:
    """Validate calibration state for every raw top-4 family.

    Bounded families used to bypass this check because it was accidentally
    nested below the unbounded-family condition.  Keep the validation in one
    helper so the cap is part of the frozen prediction contract regardless of
    whether calibration happened during or after training.
    """

    raw_architectures = {
        RAW_TOP4_SIGNED_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
    }
    if str(architecture) not in raw_architectures:
        raise ValueError("raw top-4 evidence contract has an unsupported architecture")
    scale = float(contract.get("postfit_residual_scale", 1.0))
    provenance = str(
        contract.get("postfit_scale_provenance", "unit_scale_not_recorded_v1")
    ).strip()
    if not np.isfinite(scale) or scale <= 0.0 or (scale != 1.0 and not provenance):
        raise ValueError("per-view raw top-4 post-fit scale is invalid")
    # v1 uniform-mean artifacts predate the explicit aggregation key.  They
    # remain unambiguous, whereas every non-default aggregation must state its
    # semantics in the frozen prediction contract.
    contract_aggregation = str(contract.get("raw_topk_aggregation", "uniform_mean"))
    if contract_aggregation != str(raw_topk_aggregation):
        raise ValueError("per-view raw top-k aggregation is inconsistent")
    cap = contract.get("postfit_residual_cap")
    cap_provenance = str(contract.get("postfit_cap_provenance", "")).strip()
    bounded_architectures = {
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
        RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
    }
    if str(architecture) in bounded_architectures:
        if (
            cap is None
            or not np.isfinite(float(cap))
            or float(cap) <= 0.0
            or not cap_provenance
        ):
            raise ValueError("bounded raw top-4 cap contract is invalid")
    elif cap is not None:
        raise ValueError("unbounded raw top-4 family unexpectedly has a cap")


def _load_predictions(
    *,
    features: FrozenFulltrackPerViewAppearanceFeatures,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Reject a prediction that could have changed frozen S0 semantics."""

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
            raise ValueError(f"{path}: per-view prediction lacks {missing}")
        arrays = {name: np.asarray(payload[name]).copy() for name in required}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != FULLTRACK_PER_VIEW_PREDICTION_FORMAT
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("training_supervision_split") != "train"
        or metadata.get("validation_or_test_labels_used_by_fit") is not False
        or metadata.get("prediction_frozen_before_validation_target_join") is not True
        or metadata.get("fixed_global_top_l") != 20
        or metadata.get("candidate_reselection") is not False
        or metadata.get("support_reselection") is not False
        or metadata.get("support_view_selection") is not False
        or metadata.get("all_real_sfm_support_observations_retained") is not True
        or metadata.get("feature_granularity")
        != fulltrack_per_view_feature_granularity(features)
        or metadata.get("per_view_model") is not True
        or metadata.get("missing_evidence_semantics")
        not in {
            "joint_profile_missing_edge_omitted_neutral_v1",
            "family_specific_no_availability_cue_v1",
        }
        or metadata.get("null_handling")
        != "input_null_probability_exactly_preserved_v1"
        or metadata.get("candidate_mass_handling")
        != "input_nonnull_mass_exactly_preserved_v1"
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("render") is not False
        or metadata.get("pose_scoring") is not False
        or not str(metadata.get("identity_supervision_colmap_images_sha256", "")).strip()
    ):
        raise ValueError(f"{path}: per-view prediction violates its contract")
    if metadata.get("appearance_artifacts") != _artifact_entries(features.paths):
        raise ValueError("per-view prediction was fit from different artifacts")
    if metadata.get("fulltrack_compatibility") != dict(features.compatibility):
        raise ValueError("per-view prediction has incompatible source contract")
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
        raise ValueError("per-view prediction identities differ from source")
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
        raise ValueError("per-view prediction baseline is not the immutable input posterior")
    family_names = tuple(str(value) for value in np.asarray(arrays["family_names"]).tolist())
    family_architectures = metadata.get("family_architectures")
    family_edge_feature_semantics = metadata.get("family_edge_feature_semantics")
    family_contracts = metadata.get("family_evidence_contracts")
    if (
        not isinstance(family_architectures, Mapping)
        or not isinstance(family_edge_feature_semantics, Mapping)
        or not isinstance(family_contracts, Mapping)
        or set(family_architectures) != set(family_names)
        or set(family_edge_feature_semantics) != set(family_names)
        or set(family_contracts) != set(family_names)
    ):
        raise ValueError("per-view prediction lacks family-specific evidence contracts")
    for family in family_names:
        spec = FULLTRACK_PER_VIEW_FAMILIES[family]
        contract = family_contracts[family]
        if (
            family_architectures.get(family) != spec.architecture
            or family_edge_feature_semantics.get(family)
            != spec.edge_feature_semantics
            or spec.edge_feature_semantics
            != features.compatibility.get("per_view_edge_feature_semantics")
            or not isinstance(contract, Mapping)
            or contract.get("architecture")
            not in {
                *_SPARSE_PER_VIEW_ARCHITECTURES,
                "monotonic_top1_relative_raw_top4_overlay_v1",
                "monotonic_top1_relative_positive_uplift_top4_overlay_v1",
                "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_overlay_v1",
                "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_traincal_overlay_v1",
                "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_traincal_balanced_overlay_v1",
            }
        ):
            raise ValueError("per-view prediction family architecture is inconsistent")
        if spec.architecture == "sparse_per_view_mixture":
            expected = {
                "support_view_marginalization": (
                    "learned_logsumexp_over_retained_real_observation_edges_v1"
                ),
                "missing_evidence_semantics": (
                    "joint_profile_missing_edge_omitted_neutral_v1"
                ),
                "zero_residual_reproduces_fixed_posterior": True,
            }
            if float(spec.coarse_top1_stability_weight) > 0.0:
                expected.update(
                    {
                        "training_objective": (
                            "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1"
                        ),
                        "coarse_top1_stability_weight": float(
                            spec.coarse_top1_stability_weight
                        ),
                    }
                )
            sparse_architecture = str(contract.get("architecture", ""))
            if sparse_architecture not in _SPARSE_PER_VIEW_ARCHITECTURES:
                raise ValueError("per-view sparse residual architecture is inconsistent")
            bounded = "bounded_residual" in sparse_architecture
            expected_architecture = (
                "linear"
                if "sparse_per_view_linear" in sparse_architecture
                else "mlp"
            )
            recorded_architecture = str(
                contract.get("residual_architecture", expected_architecture)
            )
            if recorded_architecture != expected_architecture:
                raise ValueError("per-view sparse residual capacity contract is inconsistent")
            cap = contract.get("residual_cap")
            cap_semantics = str(contract.get("residual_cap_semantics", ""))
            cap_provenance = str(contract.get("residual_cap_provenance", "")).strip()
            if bounded:
                if (
                    cap is None
                    or not np.isfinite(float(cap))
                    or float(cap) <= 0.0
                    or cap_semantics
                    != "symmetric_tanh_bounded_log_likelihood_ratio_v1"
                    or not cap_provenance
                    or cap_provenance == "unbounded_legacy_log_likelihood_ratio_v1"
                ):
                    raise ValueError("bounded per-view residual cap contract is invalid")
            elif cap is not None or (
                cap_semantics
                and cap_semantics != "unbounded_legacy_log_likelihood_ratio_v1"
            ):
                raise ValueError("unbounded per-view residual cap contract is invalid")
        elif spec.architecture == RAW_TOP4_SIGNED_ARCHITECTURE:
            expected = {
                "architecture": "monotonic_top1_relative_raw_top4_overlay_v1",
                "support_view_marginalization": (
                    RAW_TOPK_SUPPORT_VIEW_MARGINALIZATIONS[
                        spec.raw_topk_aggregation
                    ]
                ),
                "missing_evidence_semantics": (
                    "common_top1_profile_missing_zero_residual_v1"
                ),
                "zero_residual_reproduces_fixed_posterior": False,
            }
        elif spec.architecture == RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE:
            expected = {
                "architecture": (
                    "monotonic_top1_relative_positive_uplift_top4_overlay_v1"
                ),
                "support_view_marginalization": (
                    RAW_TOPK_SUPPORT_VIEW_MARGINALIZATIONS[
                        spec.raw_topk_aggregation
                    ]
                ),
                "missing_evidence_semantics": (
                    "common_top1_profile_missing_zero_residual_v1"
                ),
                "zero_residual_reproduces_fixed_posterior": False,
            }
        elif spec.architecture == RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE:
            expected = {
                "architecture": (
                    "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_overlay_v1"
                ),
                "support_view_marginalization": (
                    RAW_TOPK_SUPPORT_VIEW_MARGINALIZATIONS[
                        spec.raw_topk_aggregation
                    ]
                ),
                "missing_evidence_semantics": (
                    "common_top1_profile_missing_zero_residual_v1"
                ),
                "zero_residual_reproduces_fixed_posterior": False,
                "candidate_evidence_transform": (
                    "relu_positive_relative_uplift_tanh_bounded_v1"
                ),
                "postfit_cap_applied_after_train": True,
            }
        elif spec.architecture == RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE:
            expected = {
                "architecture": (
                    "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_traincal_overlay_v1"
                ),
                "support_view_marginalization": (
                    RAW_TOPK_SUPPORT_VIEW_MARGINALIZATIONS[
                        spec.raw_topk_aggregation
                    ]
                ),
                "missing_evidence_semantics": (
                    "common_top1_profile_missing_zero_residual_v1"
                ),
                "zero_residual_reproduces_fixed_posterior": False,
                "candidate_evidence_transform": (
                    "relu_positive_relative_uplift_tanh_bounded_traincal_v1"
                ),
                "postfit_cap_applied_after_train": False,
                "residual_calibration_applied_during_train": True,
            }
        elif (
            spec.architecture
            == RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE
        ):
            expected = {
                "architecture": (
                    "monotonic_top1_relative_positive_uplift_top4_tanh_bounded_traincal_balanced_overlay_v1"
                ),
                "support_view_marginalization": (
                    RAW_TOPK_SUPPORT_VIEW_MARGINALIZATIONS[
                        spec.raw_topk_aggregation
                    ]
                ),
                "missing_evidence_semantics": (
                    "common_top1_profile_missing_zero_residual_v1"
                ),
                "zero_residual_reproduces_fixed_posterior": False,
                "candidate_evidence_transform": (
                    "relu_positive_relative_uplift_tanh_bounded_traincal_v1"
                ),
                "postfit_cap_applied_after_train": False,
                "residual_calibration_applied_during_train": True,
                "training_objective": (
                    "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1"
                ),
                "coarse_top1_stability_weight": float(
                    spec.coarse_top1_stability_weight
                ),
            }
        else:
            raise ValueError("per-view prediction has an unsupported architecture")
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError("per-view prediction family evidence contract is inconsistent")
        if spec.architecture in {
            RAW_TOP4_SIGNED_ARCHITECTURE,
            RAW_TOP4_POSITIVE_UPLIFT_ARCHITECTURE,
            RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_ARCHITECTURE,
            RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_ARCHITECTURE,
            RAW_TOP4_POSITIVE_UPLIFT_BOUNDED_TRAINCAL_BALANCED_ARCHITECTURE,
        }:
            _validate_raw_top4_evidence_contract(
                architecture=spec.architecture,
                raw_topk_aggregation=spec.raw_topk_aggregation,
                contract=contract,
            )
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32)
    residual = np.asarray(arrays["candidate_residuals"], dtype=np.float32)
    if (
        not family_names
        or len(set(family_names)) != len(family_names)
        or set(family_names).difference(FULLTRACK_PER_VIEW_FAMILIES)
        or candidate.shape != (len(family_names), *features.candidate_probabilities.shape)
        or null.shape != (len(family_names), len(features.query_ids))
        or residual.shape != candidate.shape
        or np.any(~np.isfinite(candidate))
        or np.any(~np.isfinite(null))
        or np.any(~np.isfinite(residual))
        or np.any(candidate < 0.0)
        or np.any(null < 0.0)
        or np.max(np.abs(candidate.sum(axis=2) + null - 1.0)) > 2e-5
        or np.max(np.abs(null - features.null_probabilities[None, :])) > 0.0
        or np.any(candidate[:, features.candidate_probabilities <= 0.0] > 2e-6)
    ):
        raise ValueError("per-view prediction arrays are invalid")
    return candidate, null, residual, family_names, metadata


def _top_and_rank(
    *, scores: np.ndarray, labels: np.ndarray, candidate_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze the baseline rank-2 subset before probe scores are inspected."""

    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    if values.shape != positive.shape or positive.shape != valid.shape:
        raise ValueError("per-view rank arrays are incompatible")
    order = np.argsort(-np.where(valid, values, -np.inf), axis=1, kind="stable")
    ranked_positive = np.take_along_axis(positive & valid, order, axis=1)
    rank = np.argmax(ranked_positive, axis=1).astype(np.int64) + 1
    rank[~np.any(ranked_positive, axis=1)] = -1
    return order[:, 0].astype(np.int64), rank


def fulltrack_per_view_candidate_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Use the predeclared strict gate for frozen per-view candidate evidence."""

    result = dict(fulltrack_residual_candidate_gate(metrics))
    result["policy"] = (
        "validation-only frozen per-view candidate-evidence gate; a pass permits only a "
        "separate frozen-hypothesis pose audit, never direct promotion"
    )
    return result


def audit_frozen_fulltrack_per_view_candidate_probe(
    *,
    appearance_artifacts: Sequence[Path],
    predictions: Path,
    colmap_model_dir: Path,
    projected_landmark_bank: Path,
    output_dir: Path,
    audit_split: str = "validation",
    registered_identity_radius_px: float,
    minimum_shift_pairs: int,
    minimum_shift_m: float,
    maximum_shift_dispersion_m: float,
    maximum_shift_relative_dispersion: float,
) -> dict[str, Any]:
    """Join audit identities only after the per-view prediction is frozen."""

    if (
        float(registered_identity_radius_px) <= 0.0
        or int(minimum_shift_pairs) <= 0
        or float(minimum_shift_m) < 0.0
        or float(maximum_shift_dispersion_m) < 0.0
        or float(maximum_shift_relative_dispersion) < 0.0
        or str(audit_split) not in {"train", "validation"}
    ):
        raise ValueError("per-view audit arguments are invalid")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_fulltrack_per_view_appearance_features(paths)
    candidate, null, residual, families, prediction_metadata = _load_predictions(
        features=features, path=Path(predictions)
    )
    selected_rows = np.flatnonzero(features.split_names == str(audit_split))
    if not len(selected_rows):
        raise ValueError(f"per-view audit has no {audit_split} rows")
    _validate_projected_bank_lineage(
        features.artifact_metadata, projected_landmark_bank=Path(projected_landmark_bank)
    )
    bank, _bank_metadata = load_landmark_index_npz(Path(projected_landmark_bank))
    xyz_by_track = {
        int(track): np.asarray(bank.xyz[index], dtype=np.float64)
        for index, track in enumerate(np.asarray(bank.track_ids, dtype=np.int64))
    }
    identity_images_path = Path(colmap_model_dir) / "images.bin"
    identity_images_sha256 = file_sha256_short(identity_images_path)
    if (
        identity_images_sha256
        != str(prediction_metadata["identity_supervision_colmap_images_sha256"])
    ):
        raise ValueError(
            "per-view audit COLMAP identity model differs from the train-frozen "
            "prediction contract"
        )
    images = read_colmap_images_binary(identity_images_path)
    images_by_name = {str(image.image_name): image for image in images.values()}
    targets = registered_query_observation_targets(
        query_ids=features.query_ids[selected_rows],
        query_xy=features.xy[selected_rows],
        images_by_name=images_by_name,
        max_distance_px=float(registered_identity_radius_px),
    )
    labels = registered_candidate_identity_labels(
        features.candidate_track_ids[selected_rows], targets
    )
    identity_coverage = summarize_registered_candidate_identity(labels, targets)
    candidate_valid = (
        (features.candidate_track_ids[selected_rows] >= 0)
        & (features.candidate_probabilities[selected_rows] > 0.0)
    )
    if np.any(labels & ~candidate_valid):
        raise ValueError("audit identity target is absent from frozen posterior")
    baseline_scores = np.log(
        np.maximum(features.candidate_probabilities[selected_rows], 1e-30)
    )
    row_mask = targets.supervised & np.any(labels, axis=1)
    hard_selection = select_rank2_to_top1_wrong_pairs(
        query_ids=features.query_ids[selected_rows],
        candidate_track_ids=features.candidate_track_ids[selected_rows],
        candidate_probabilities=features.candidate_probabilities[selected_rows],
        correct_labels=labels,
        xyz_by_track=xyz_by_track,
    )
    coherent_mask, coherent_diagnostics = coherent_shift_mask(
        hard_selection,
        minimum_pairs=int(minimum_shift_pairs),
        minimum_shift_m=float(minimum_shift_m),
        maximum_dispersion_m=float(maximum_shift_dispersion_m),
        maximum_relative_dispersion=float(maximum_shift_relative_dispersion),
    )
    hard_row_mask = np.zeros((len(selected_rows),), dtype=bool)
    hard_row_mask[hard_selection.row_indices] = True
    _top, baseline_rank = _top_and_rank(
        scores=baseline_scores, labels=labels, candidate_valid=candidate_valid
    )
    rank2_rows = row_mask & (baseline_rank > 1)
    result: dict[str, Any] = {
        "stage": "audit_frozen_fulltrack_per_view_candidate_probe",
        "audit_split": str(audit_split),
        "diagnostic_only": True,
        "promotion_allowed": False,
        "appearance_artifacts": _artifact_entries(paths),
        "predictions": {
            "path": str(Path(predictions)),
            "sha256": file_sha256_short(Path(predictions)),
            "metadata": prediction_metadata,
        },
        "projected_landmark_bank": {
            "path": str(Path(projected_landmark_bank)),
            "sha256": file_sha256_short(Path(projected_landmark_bank)),
        },
        "row_count": int(len(selected_rows)),
        "query_count": int(len(set(features.query_ids[selected_rows].tolist()))),
        "registered_identity_radius_px": float(registered_identity_radius_px),
        "identity_supervision_colmap_images_bin": str(identity_images_path),
        "identity_supervision_colmap_images_sha256": identity_images_sha256,
        "registered_identity_target_coverage": identity_coverage,
        "hard_pair_scope": {
            "pair_count": hard_selection.pair_count,
            "query_count": int(len(set(hard_selection.query_ids.tolist()))),
            "coherent_shift_pair_count": int(np.count_nonzero(coherent_mask)),
            "coherent_shift_query_count": int(
                sum(1 for item in coherent_diagnostics if bool(item["coherent"]))
            ),
            "coherent_shift_minimum_pair_count_for_claim": (
                MINIMUM_COHERENT_SHIFT_PAIRS_FOR_CLAIM
            ),
            "coherent_shift_sufficient_for_claim": bool(
                np.count_nonzero(coherent_mask)
                >= MINIMUM_COHERENT_SHIFT_PAIRS_FOR_CLAIM
            ),
            "coherent_shift_queries": coherent_diagnostics,
        },
        "families": {},
        "predeclared_gate_summary": {},
        "protocol": {
            "targets_loaded_only_in_audit": True,
            "audit_split_is_train_development_diagnostic": bool(
                str(audit_split) == "train"
            ),
            "model_fit_or_selection": False,
            "prediction_frozen_before_validation_target_join": True,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "all_real_sfm_support_observations_retained": True,
            "per_view_s2_claimed": False,
            "input_null_probability_exactly_preserved": True,
            "image_retrieval_or_submap_used": False,
            "render": False,
            "pose_scoring": False,
        },
    }
    rows_for_csv: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        probe_scores = np.log(
            np.maximum(candidate[family_index, selected_rows], 1e-30)
        )
        exact_identity = {
            "baseline": rank_metrics(
                scores=baseline_scores,
                labels=labels,
                valid=candidate_valid,
                rows=row_mask,
            ),
            "probe": rank_metrics(
                scores=probe_scores,
                labels=labels,
                valid=candidate_valid,
                rows=row_mask,
            ),
            "paired_rank": paired_rank(
                baseline_scores=baseline_scores,
                probe_scores=probe_scores,
                labels=labels,
                valid=candidate_valid,
                rows=row_mask,
            ),
        }
        rank2 = {
            "baseline": rank_metrics(
                scores=baseline_scores,
                labels=labels,
                valid=candidate_valid,
                rows=rank2_rows,
            ),
            "probe": rank_metrics(
                scores=probe_scores,
                labels=labels,
                valid=candidate_valid,
                rows=rank2_rows,
            ),
            "paired_rank": paired_rank(
                baseline_scores=baseline_scores,
                probe_scores=probe_scores,
                labels=labels,
                valid=candidate_valid,
                rows=rank2_rows,
            ),
        }
        hard = {
            "baseline_pairwise": pairwise_feature_metrics(
                scores=baseline_scores,
                feature_valid=candidate_valid,
                selection=hard_selection,
            ),
            "probe_pairwise": pairwise_feature_metrics(
                scores=probe_scores,
                feature_valid=candidate_valid,
                selection=hard_selection,
            ),
            "coherent_shift_probe_pairwise": pairwise_feature_metrics(
                scores=probe_scores,
                feature_valid=candidate_valid,
                selection=hard_selection,
                subset_mask=coherent_mask,
            ),
            "paired_rank": paired_rank(
                baseline_scores=baseline_scores,
                probe_scores=probe_scores,
                labels=labels,
                valid=candidate_valid,
                rows=hard_row_mask,
            ),
        }
        active_residual = residual[family_index, selected_rows][candidate_valid]
        payload = {
            "family": family,
            "exact_registered_identity": exact_identity,
            "exact_registered_rank2_to_l": rank2,
            "rank2_to_top1_wrong_hard_pairs": hard,
            "residual_statistics": {
                "mean": None if not len(active_residual) else float(np.mean(active_residual)),
                "std": None if not len(active_residual) else float(np.std(active_residual)),
                "maximum_abs": (
                    None if not len(active_residual) else float(np.max(np.abs(active_residual)))
                ),
            },
        }
        payload["predeclared_incremental_gate"] = (
            fulltrack_per_view_candidate_gate(payload)
            if str(audit_split) == "validation"
            else {
                "policy": (
                    "train-only frozen per-view candidate-evidence diagnostic; it cannot pass a "
                    "promotion gate or authorize a pose audit"
                ),
                "checks": {},
                "passed": False,
                "not_applicable": True,
            }
        )
        result["families"][family] = payload
        result["predeclared_gate_summary"][family] = payload[
            "predeclared_incremental_gate"
        ]
        rows_for_csv.append(
            {
                "family": family,
                "overall_baseline_top1": exact_identity["baseline"].get(
                    "top1_positive_rate_given_positive"
                ),
                "overall_probe_top1": exact_identity["probe"].get(
                    "top1_positive_rate_given_positive"
                ),
                "overall_baseline_p90_rank": exact_identity["baseline"].get(
                    "p90_first_positive_rank"
                ),
                "overall_probe_p90_rank": exact_identity["probe"].get(
                    "p90_first_positive_rank"
                ),
                "rank2_baseline_median": rank2["baseline"].get(
                    "median_first_positive_rank"
                ),
                "rank2_probe_median": rank2["probe"].get("median_first_positive_rank"),
                "hard_pair_probe_win_rate": hard["probe_pairwise"].get("win_rate"),
                "hard_pair_probe_wilson95_lower": hard["probe_pairwise"].get(
                    "win_rate_wilson95_lower"
                ),
                "hard_pair_probe_median_gap": hard["probe_pairwise"].get(
                    "median_correct_minus_wrong"
                ),
                "gate_pass": payload["predeclared_incremental_gate"]["passed"],
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "family_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_for_csv[0]))
        writer.writeheader()
        writer.writerows(rows_for_csv)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_frozen_fulltrack_per_view_candidate_probe(
        appearance_artifacts=_paths(args.appearance_artifacts),
        predictions=Path(args.predictions),
        colmap_model_dir=Path(args.colmap_model_dir),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        output_dir=Path(args.output_dir),
        audit_split=str(args.audit_split),
        registered_identity_radius_px=float(args.registered_identity_radius_px),
        minimum_shift_pairs=int(args.minimum_shift_pairs),
        minimum_shift_m=float(args.minimum_shift_m),
        maximum_shift_dispersion_m=float(args.maximum_shift_dispersion_m),
        maximum_shift_relative_dispersion=float(
            args.maximum_shift_relative_dispersion
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
