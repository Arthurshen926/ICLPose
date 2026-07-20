"""Audit train-frozen full-track residual predictions on validation identities.

The fit command emits predictions before validation labels are materialized.
This command verifies that provenance and fixed-mass invariants, then asks the
only question this aggregate diagnostic is allowed to answer: do predeclared
appearance residuals improve frozen candidate identity ranking, especially for
rank-2-to-top-20 correct tracks?  It never forms or scores a pose.
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
from feature_extract.tools.vfm.audit_frozen_fulltrack_hard_pairs import (
    _validate_projected_bank_lineage,
    coherent_shift_mask,
    pairwise_feature_metrics,
    select_rank2_to_top1_wrong_pairs,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.colmap_tracks import read_colmap_images_binary
from feature_extract.vfm.localization.frozen_fulltrack_candidate_appearance_residual_probe import (
    FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_PREDICTION_FORMAT,
    FROZEN_FULLTRACK_RESIDUAL_FAMILIES,
    FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES,
    FrozenFulltrackAppearanceFeatures,
    SUMMARY_TOP4_BALANCED_ARCHITECTURE,
    SUMMARY_TOP4_FEATURE_GRANULARITY,
    load_frozen_fulltrack_appearance_features,
)
from feature_extract.vfm.localization.landmark_hybrid import load_landmark_index_npz
from feature_extract.vfm.localization.query_observation_identity import (
    registered_candidate_identity_labels,
    registered_query_observation_targets,
    summarize_registered_candidate_identity,
)


MINIMUM_RANK2_ROWS = 50
MINIMUM_HARD_PAIR_ROWS = 50
MINIMUM_COHERENT_SHIFT_PAIRS_FOR_CLAIM = 50


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance-artifacts", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--colmap-model-dir", required=True)
    parser.add_argument("--projected-landmark-bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--registered-identity-radius-px", type=float, default=2.0)
    parser.add_argument("--minimum-shift-pairs", type=int, default=3)
    parser.add_argument("--minimum-shift-m", type=float, default=0.10)
    parser.add_argument("--maximum-shift-dispersion-m", type=float, default=0.20)
    parser.add_argument("--maximum-shift-relative-dispersion", type=float, default=0.35)
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("full-track appearance artifacts must be non-empty and unique")
    return paths


def _artifact_entries(paths: Sequence[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path), "sha256": file_sha256_short(path)} for path in paths
    ]


def _same_float(value: object, expected: float) -> bool:
    try:
        return float(value) == float(expected)
    except (TypeError, ValueError):
        return False


def _load_predictions(
    *,
    features: FrozenFulltrackAppearanceFeatures,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Verify a frozen prediction before target-side validation identity joins."""

    with np.load(Path(path), allow_pickle=False) as payload:
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
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path}: full-track residual prediction lacks {missing}")
        arrays = {name: np.asarray(payload[name]).copy() for name in required}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    family_names = tuple(str(value) for value in np.asarray(arrays["family_names"]).tolist())
    summary_families = set(FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES)
    supported_families = set(FROZEN_FULLTRACK_RESIDUAL_FAMILIES).union(
        summary_families
    )
    summary_top4_mode = bool(family_names) and set(family_names).issubset(
        summary_families
    )
    if (
        not family_names
        or len(set(family_names)) != len(family_names)
        or set(family_names).difference(supported_families)
        or (
            set(family_names).intersection(summary_families)
            and not summary_top4_mode
        )
    ):
        raise ValueError("full-track residual prediction families are invalid")
    expected_granularity = (
        SUMMARY_TOP4_FEATURE_GRANULARITY
        if summary_top4_mode
        else "candidate_summary_aggregate_not_per_view_v1"
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != FROZEN_FULLTRACK_APPEARANCE_RESIDUAL_PREDICTION_FORMAT
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or metadata.get("validation_or_test_labels_used_by_fit") is not False
        or metadata.get("prediction_frozen_before_validation_target_join") is not True
        or not str(metadata.get("identity_supervision_colmap_images_bin", "")).strip()
        or not str(metadata.get("identity_supervision_colmap_images_sha256", "")).strip()
        or metadata.get("fixed_global_top_l") != 20
        or metadata.get("candidate_reselection") is not False
        or metadata.get("support_reselection") is not False
        or metadata.get("support_view_selection") is not False
        or metadata.get("all_observation_aggregation_preserved") is not True
        or metadata.get("feature_granularity") != expected_granularity
        or metadata.get("per_view_model") is not False
        or metadata.get("null_handling")
        != "input_null_probability_exactly_preserved_v1"
        or metadata.get("candidate_mass_handling")
        != "input_nonnull_mass_exactly_preserved_v1"
        or metadata.get("image_retrieval_or_submap_used") is not False
        or metadata.get("render") is not False
        or metadata.get("pose_scoring") is not False
    ):
        raise ValueError(f"{path}: full-track residual prediction violates its contract")
    if metadata.get("appearance_artifacts") != _artifact_entries(features.paths):
        raise ValueError("full-track residual prediction was fit from different artifacts")
    if metadata.get("fulltrack_compatibility") != dict(features.compatibility):
        raise ValueError("full-track residual prediction has incompatible source contract")
    if (
        not np.array_equal(np.asarray(arrays["query_ids"]).astype(str), features.query_ids)
        or not np.array_equal(np.asarray(arrays["split_names"]).astype(str), features.split_names)
        or not np.array_equal(
            np.asarray(arrays["source_row_indices"], dtype=np.int64),
            features.source_row_indices,
        )
        or not np.array_equal(
            np.asarray(arrays["candidate_track_ids"], dtype=np.int64),
            features.candidate_track_ids,
        )
    ):
        raise ValueError("full-track residual prediction identities differ from source")
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
        raise ValueError("full-track residual baseline is not the immutable input posterior")
    candidate = np.asarray(arrays["candidate_probabilities"], dtype=np.float32)
    null = np.asarray(arrays["null_probabilities"], dtype=np.float32)
    residual = np.asarray(arrays["candidate_residuals"], dtype=np.float32)
    if (
        candidate.shape != (len(family_names), *features.candidate_probabilities.shape)
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
        raise ValueError("full-track residual prediction arrays are invalid")
    if summary_top4_mode:
        architectures = metadata.get("family_architectures")
        contracts = metadata.get("family_evidence_contracts")
        if (
            not isinstance(architectures, Mapping)
            or not isinstance(contracts, Mapping)
            or set(architectures) != set(family_names)
            or set(contracts) != set(family_names)
        ):
            raise ValueError("summary top-four prediction lacks family contracts")
        for family in family_names:
            spec = FROZEN_FULLTRACK_SUMMARY_TOP4_FAMILIES[family]
            contract = contracts.get(family)
            expected_feature_names = [
                f"{profile}__uniform_top4_mean_ncc"
                for profile in spec.profile_names
            ]
            if (
                architectures.get(family) != SUMMARY_TOP4_BALANCED_ARCHITECTURE
                or not isinstance(contract, Mapping)
                or contract.get("architecture") != SUMMARY_TOP4_BALANCED_ARCHITECTURE
                or contract.get("candidate_evidence_transform")
                != "relu_positive_relative_summary_top4_uplift_tanh_bounded_traincal_v1"
                or contract.get("summary_statistic") != "uniform_top4_mean_ncc"
                or contract.get("profile_names") != list(spec.profile_names)
                or contract.get("profile_feature_names") != expected_feature_names
                or contract.get("training_objective")
                != "identity_nll_plus_rank2_rescue_and_coarse_top1_stability_v1"
                or not _same_float(
                    contract.get("rank2_hard_pair_weight"),
                    spec.rank2_hard_pair_weight,
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
    return candidate, null, residual, family_names, metadata


def fulltrack_residual_candidate_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Conservative candidate-only gate for a later frozen pose audit.

    It deliberately checks whole-set ranking, rank-2 rescue behavior, and the
    fixed hard pair separately.  Coherent periodic-shift samples are reported
    but excluded from this gate until their holdout count is adequate.
    """

    overall = metrics.get("exact_registered_identity")
    rank2 = metrics.get("exact_registered_rank2_to_l")
    hard = metrics.get("rank2_to_top1_wrong_hard_pairs")
    if not all(isinstance(item, Mapping) for item in (overall, rank2, hard)):
        raise ValueError("full-track residual gate lacks metrics")
    base = overall.get("baseline")
    probe = overall.get("probe")
    paired = overall.get("paired_rank")
    rank2_base = rank2.get("baseline")
    rank2_probe = rank2.get("probe")
    rank2_paired = rank2.get("paired_rank")
    hard_base = hard.get("baseline_pairwise")
    hard_probe = hard.get("probe_pairwise")
    hard_paired = hard.get("paired_rank")
    required = (
        base,
        probe,
        paired,
        rank2_base,
        rank2_probe,
        rank2_paired,
        hard_base,
        hard_probe,
        hard_paired,
    )
    if not all(isinstance(item, Mapping) for item in required):
        raise ValueError("full-track residual gate metrics are incomplete")
    comparable = all(
        value is not None
        for value in (
            base.get("candidate_pair_average_precision"),
            probe.get("candidate_pair_average_precision"),
            base.get("top1_positive_rate_given_positive"),
            probe.get("top1_positive_rate_given_positive"),
            base.get("p90_first_positive_rank"),
            probe.get("p90_first_positive_rank"),
            rank2_base.get("median_first_positive_rank"),
            rank2_probe.get("median_first_positive_rank"),
            rank2_base.get("p90_first_positive_rank"),
            rank2_probe.get("p90_first_positive_rank"),
            hard_base.get("median_correct_minus_wrong"),
            hard_probe.get("median_correct_minus_wrong"),
            hard_probe.get("win_rate_wilson95_lower"),
        )
    )
    checks = {
        "comparable_validation_identity_rows": bool(comparable),
        "overall_candidate_ap_not_worse": bool(
            comparable
            and float(probe["candidate_pair_average_precision"])
            >= float(base["candidate_pair_average_precision"])
        ),
        "overall_top1_not_worse": bool(
            comparable
            and float(probe["top1_positive_rate_given_positive"])
            >= float(base["top1_positive_rate_given_positive"])
        ),
        "overall_p90_rank_not_worse": bool(
            comparable
            and float(probe["p90_first_positive_rank"])
            <= float(base["p90_first_positive_rank"])
        ),
        "overall_rank_wins_exceed_losses": int(paired.get("rank_win_count", 0))
        > int(paired.get("rank_loss_count", 0)),
        "overall_top1_rescues_exceed_harms": int(paired.get("top1_rescue_count", 0))
        > int(paired.get("top1_harm_count", 0)),
        "rank2_to_l_minimum_positive_rows": int(
            rank2_paired.get("positive_row_count", 0)
        )
        >= MINIMUM_RANK2_ROWS,
        "rank2_to_l_median_strictly_improved": bool(
            comparable
            and float(rank2_probe["median_first_positive_rank"])
            < float(rank2_base["median_first_positive_rank"])
        ),
        "rank2_to_l_p90_not_worse": bool(
            comparable
            and float(rank2_probe["p90_first_positive_rank"])
            <= float(rank2_base["p90_first_positive_rank"])
        ),
        "rank2_to_l_wins_exceed_losses": int(rank2_paired.get("rank_win_count", 0))
        > int(rank2_paired.get("rank_loss_count", 0)),
        "rank2_to_l_top1_rescues_exceed_harms": int(
            rank2_paired.get("top1_rescue_count", 0)
        )
        > int(rank2_paired.get("top1_harm_count", 0)),
        "hard_pair_minimum_usable_rows": int(hard_probe.get("usable_pair_count", 0))
        >= MINIMUM_HARD_PAIR_ROWS,
        "hard_pair_wilson95_lower_above_chance": bool(
            comparable and float(hard_probe["win_rate_wilson95_lower"]) > 0.5
        ),
        "hard_pair_median_gap_positive": bool(
            comparable and float(hard_probe["median_correct_minus_wrong"]) > 0.0
        ),
        "hard_pair_gap_strictly_improved": bool(
            comparable
            and float(hard_probe["median_correct_minus_wrong"])
            > float(hard_base["median_correct_minus_wrong"])
        ),
        "hard_pair_rank_wins_exceed_losses": int(
            hard_paired.get("rank_win_count", 0)
        )
        > int(hard_paired.get("rank_loss_count", 0)),
        "hard_pair_top1_rescues_exceed_harms": int(
            hard_paired.get("top1_rescue_count", 0)
        )
        > int(hard_paired.get("top1_harm_count", 0)),
    }
    return {
        "policy": (
            "validation-only aggregate-residual candidate gate; a pass permits only "
            "a separate frozen-hypothesis pose audit, never direct promotion"
        ),
        "coherent_shift_subset_excluded_until_minimum_pair_count": (
            MINIMUM_COHERENT_SHIFT_PAIRS_FOR_CLAIM
        ),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def audit_frozen_fulltrack_candidate_appearance_residual(
    *,
    appearance_artifacts: Sequence[Path],
    predictions: Path,
    colmap_model_dir: Path,
    projected_landmark_bank: Path,
    output_dir: Path,
    registered_identity_radius_px: float,
    minimum_shift_pairs: int,
    minimum_shift_m: float,
    maximum_shift_dispersion_m: float,
    maximum_shift_relative_dispersion: float,
) -> dict[str, Any]:
    """Join validation identities only after the fit prediction has frozen."""

    if (
        float(registered_identity_radius_px) <= 0.0
        or int(minimum_shift_pairs) <= 0
        or float(minimum_shift_m) < 0.0
        or float(maximum_shift_dispersion_m) < 0.0
        or float(maximum_shift_relative_dispersion) < 0.0
    ):
        raise ValueError("full-track residual audit arguments are invalid")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = tuple(Path(path) for path in appearance_artifacts)
    features = load_frozen_fulltrack_appearance_features(paths)
    candidate, null, residual, families, prediction_metadata = _load_predictions(
        features=features, path=Path(predictions)
    )
    validation_rows = np.flatnonzero(features.split_names == "validation")
    if not len(validation_rows):
        raise ValueError("full-track residual audit has no validation rows")
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
            "full-track residual audit COLMAP identity model differs from the "
            "train-frozen prediction contract"
        )
    images = read_colmap_images_binary(identity_images_path)
    images_by_name = {str(image.image_name): image for image in images.values()}
    targets = registered_query_observation_targets(
        query_ids=features.query_ids[validation_rows],
        query_xy=features.xy[validation_rows],
        images_by_name=images_by_name,
        max_distance_px=float(registered_identity_radius_px),
    )
    labels = registered_candidate_identity_labels(
        features.candidate_track_ids[validation_rows], targets
    )
    identity_coverage = summarize_registered_candidate_identity(labels, targets)
    candidate_valid = (
        (features.candidate_track_ids[validation_rows] >= 0)
        & (features.candidate_probabilities[validation_rows] > 0.0)
    )
    if np.any(labels & ~candidate_valid):
        raise ValueError("validation identity target is absent from frozen posterior")
    baseline_scores = np.log(
        np.maximum(features.candidate_probabilities[validation_rows], 1e-30)
    )
    row_mask = targets.supervised & np.any(labels, axis=1)
    hard_selection = select_rank2_to_top1_wrong_pairs(
        query_ids=features.query_ids[validation_rows],
        candidate_track_ids=features.candidate_track_ids[validation_rows],
        candidate_probabilities=features.candidate_probabilities[validation_rows],
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
    hard_row_mask = np.zeros((len(validation_rows),), dtype=bool)
    hard_row_mask[hard_selection.row_indices] = True
    _top, baseline_rank = _top_and_rank(
        scores=baseline_scores, labels=labels, candidate_valid=candidate_valid
    )
    rank2_rows = row_mask & (baseline_rank > 1)
    result: dict[str, Any] = {
        "stage": "audit_frozen_fulltrack_candidate_appearance_conditional_residual",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "appearance_artifacts": _artifact_entries(paths),
        "predictions": {
            "path": str(Path(predictions)),
            "sha256": file_sha256_short(Path(predictions)),
        },
        "projected_landmark_bank": {
            "path": str(Path(projected_landmark_bank)),
            "sha256": file_sha256_short(Path(projected_landmark_bank)),
        },
        "row_count": int(len(validation_rows)),
        "query_count": int(len(set(features.query_ids[validation_rows].tolist()))),
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
            "targets_loaded_only_in_validation_audit": True,
            "model_fit_or_selection": False,
            "prediction_frozen_before_validation_target_join": True,
            "fixed_global_top_l": 20,
            "candidate_reselection": False,
            "all_observation_aggregation_preserved": True,
            "per_view_s2_claimed": False,
            "input_null_probability_exactly_preserved": True,
            "pose_scoring": False,
            "image_retrieval_or_submap_used": False,
            "render": False,
        },
        "prediction_contract": {
            "training_supervision_split": prediction_metadata.get(
                "training_supervision_split"
            ),
            "null_handling": prediction_metadata.get("null_handling"),
            "candidate_mass_handling": prediction_metadata.get(
                "candidate_mass_handling"
            ),
            "feature_granularity": prediction_metadata.get("feature_granularity"),
            "identity_supervision_colmap_images_sha256": prediction_metadata.get(
                "identity_supervision_colmap_images_sha256"
            ),
        },
    }
    rows_for_csv: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        probe_scores = np.log(
            np.maximum(candidate[family_index, validation_rows], 1e-30)
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
            "baseline_coherent_shift_pairwise": pairwise_feature_metrics(
                scores=baseline_scores,
                feature_valid=candidate_valid,
                selection=hard_selection,
                subset_mask=coherent_mask,
            ),
            "probe_coherent_shift_pairwise": pairwise_feature_metrics(
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
        active_residual = residual[family_index, validation_rows][candidate_valid]
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
        payload["predeclared_incremental_gate"] = fulltrack_residual_candidate_gate(
            payload
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


def _top_and_rank(
    *, scores: np.ndarray, labels: np.ndarray, candidate_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Private rank helper used only to freeze the baseline rank-2 subset."""

    values = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(labels, dtype=bool)
    valid = np.asarray(candidate_valid, dtype=bool)
    if values.shape != positive.shape or positive.shape != valid.shape:
        raise ValueError("full-track residual rank arrays are incompatible")
    ranked = np.where(valid, values, -np.inf)
    order = np.argsort(-ranked, axis=1, kind="stable")
    ranked_positive = np.take_along_axis(positive & valid, order, axis=1)
    rank = np.argmax(ranked_positive, axis=1).astype(np.int64) + 1
    rank[~np.any(ranked_positive, axis=1)] = -1
    return order[:, 0].astype(np.int64), rank


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_frozen_fulltrack_candidate_appearance_residual(
        appearance_artifacts=_paths(args.appearance_artifacts),
        predictions=Path(args.predictions),
        colmap_model_dir=Path(args.colmap_model_dir),
        projected_landmark_bank=Path(args.projected_landmark_bank),
        output_dir=Path(args.output_dir),
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
