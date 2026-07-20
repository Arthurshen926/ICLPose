"""Fit a train-only safe promotion policy for normalized RGB pose likelihoods.

The command consumes frozen no-RGB and normalized candidate-specific RGB score
artifacts.  It never regenerates candidates or hypotheses.  Ground-truth pose
errors are loaded only from post-hoc target artifacts to fit a query-grouped
OOF abstention policy and to audit the frozen policy on validation and test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.absolute_likelihood_promotion import (
    FEATURE_NAMES,
    NO_PROMOTION_THRESHOLD,
    PROMOTION_FEATURE_SCHEMA,
    AbsoluteLikelihoodPromotionExample,
    AbsoluteLikelihoodPromotionModel,
    crossfit_probabilities,
    fit_final_model,
    policy_metrics,
    promotion_decisions,
    select_oof_tail_safe_threshold,
    tail_safe_gate,
)


_SCORE_FORMAT = "independent_landmark_hypothesis_scores_v1"
_TARGET_FORMAT = "independent_landmark_hypothesis_score_targets_v1"
_ROW_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
)
_SCORE_FIELDS = _ROW_FIELDS + (
    "independent_score_top1",
    "independent_selection_scores",
    "independent_log_likelihood_means",
    "independent_log_likelihood_medians",
    "independent_log_likelihood_trimmed_means_10",
    "independent_log_likelihood_worst_quartile_means",
    "independent_log_likelihood_lcb95s",
    "independent_spatial_median_of_means_2x2",
    "independent_effective_point_counts",
    "independent_evidence_coverages",
    "verification_point_counts",
    "candidate_spatial_materialized_verification_point_counts",
    "candidate_spatial_materialized_candidate_view_counts",
)
_TARGET_FIELDS = _ROW_FIELDS + (
    "translation_errors_m",
    "rotation_errors_deg",
)
_IMMUTABLE_INPUT_HASHES = (
    "candidate_artifact_sha256",
    "proposals_sha256",
    "fixed_candidate_prior_overlay_sha256",
    "projected_landmark_bank_sha256",
    "independent_verification_landmark_bank_sha256",
    "support_geometry_index_sha256",
    "detector_query_cache_sha256",
)

_OPTIONAL_PROFILE_SCORE_FIELDS = (
    "independent_log_likelihood_means",
    "independent_log_likelihood_medians",
    "independent_log_likelihood_trimmed_means_10",
    "independent_log_likelihood_worst_quartile_means",
    "independent_log_likelihood_lcb95s",
    "independent_spatial_median_of_means_2x2",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_score_artifacts", required=True)
    parser.add_argument("--optional_train_score_artifacts", required=True)
    parser.add_argument("--optional_heldout_score_artifacts", required=True)
    parser.add_argument("--optional_train_targets", required=True)
    parser.add_argument("--optional_heldout_targets", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--c_value", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths:
        raise ValueError("at least one artifact path is required")
    return paths


def _row_key(arrays: Mapping[str, np.ndarray], row: int) -> tuple[str, str, str, int]:
    return (
        str(arrays["split_names"][row]),
        str(arrays["query_ids"][row]),
        str(arrays["evaluation_labels"][row]),
        int(arrays["hypothesis_indices"][row]),
    )


def _group_key(arrays: Mapping[str, np.ndarray], row: int) -> tuple[str, str, str]:
    return (
        str(arrays["split_names"][row]),
        str(arrays["query_ids"][row]),
        str(arrays["evaluation_labels"][row]),
    )


def _load_score(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path}: score artifact has no metadata")
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    missing = [key for key in _SCORE_FIELDS if key not in arrays]
    if missing:
        raise ValueError(f"{path}: score artifact lacks required fields {missing}")
    if metadata.get("format") != _SCORE_FORMAT:
        raise ValueError(f"{path}: unsupported score format")
    if metadata.get("contains_target_fields") is not False:
        raise ValueError(f"{path}: score artifact contains targets")
    if metadata.get("pose_or_ground_truth_used_for_scoring") is not False:
        raise ValueError(f"{path}: score generation used pose or ground truth")
    if metadata.get("supervision_arrays_loaded") is not False:
        raise ValueError(f"{path}: score generation loaded supervision arrays")
    row_count = int(metadata.get("row_count", -1))
    if row_count <= 0:
        raise ValueError(f"{path}: score artifact has an invalid row count")
    for key, values in arrays.items():
        if values.ndim == 0 or values.shape[0] != row_count:
            raise ValueError(f"{path}: {key} is not row aligned")
    keys = [_row_key(arrays, row) for row in range(row_count)]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: score artifact has duplicate hypothesis rows")
    return arrays, metadata


def _merge_scores(paths: Sequence[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    loaded = [_load_score(path) for path in paths]
    schemas = [set(arrays) for arrays, _metadata in loaded]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("score artifacts do not share an array schema")
    arrays = {
        key: np.concatenate([payload[key] for payload, _metadata in loaded], axis=0)
        for key in sorted(schemas[0])
    }
    keys = [_row_key(arrays, row) for row in range(len(arrays["query_ids"]))]
    if len(keys) != len(set(keys)):
        raise ValueError("merged score artifacts contain duplicate hypothesis rows")
    return arrays, [metadata for _payload, metadata in loaded]


def _strict_contract(metadata: Mapping[str, object]) -> Mapping[str, object]:
    strict = metadata.get("strict_absolute_evidence_contract")
    if not isinstance(strict, Mapping):
        raise ValueError("score metadata lacks a strict absolute-evidence contract")
    common = {
        "heldout_query_rows": True,
        "fixed_global_topl": True,
        "explicit_null_mass": True,
        "identity_prior_fixed_across_hypotheses": True,
        "support_appearance_posterior_pose_independent": True,
        "pose_local_candidate_reselection": False,
        "pose_conditioned_refinement": False,
    }
    if any(strict.get(key) is not expected for key, expected in common.items()):
        raise ValueError("score metadata violates immutable absolute-evidence semantics")
    return strict


def _immutable_contract(metadata: Mapping[str, object]) -> dict[str, object]:
    inputs = metadata.get("inputs")
    selection = metadata.get("selection")
    query_selection = metadata.get("query_point_selection")
    if not all(
        isinstance(value, Mapping) for value in (inputs, selection, query_selection)
    ):
        raise ValueError("score metadata lacks an immutable input contract")
    hashes = {key: inputs.get(key) for key in _IMMUTABLE_INPUT_HASHES}
    if any(value is None for value in hashes.values()):
        raise ValueError("score metadata lacks a required immutable input hash")
    return {
        "hypothesis_compatibility_sha256": metadata.get(
            "hypothesis_compatibility_sha256"
        ),
        "config": metadata.get("config"),
        "selection": {
            key: selection.get(key)
            for key in ("statistic", "score_field", "tie_break")
        },
        "query_point_selection": {
            key: query_selection.get(key)
            for key in (
                "verification_point_count",
                "detector_log_merit_weight",
                "merit",
                "source",
            )
        },
        "hypothesis_scope": metadata.get("hypothesis_scope"),
        "crossfit": metadata.get("crossfit"),
        "input_hashes": hashes,
    }


def _validate_score_contracts(
    *,
    baseline_metadata: Sequence[Mapping[str, object]],
    optional_metadata: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not baseline_metadata or not optional_metadata:
        raise ValueError("baseline and optional score metadata are both required")
    baseline_contracts = [_immutable_contract(row) for row in baseline_metadata]
    optional_contracts = [_immutable_contract(row) for row in optional_metadata]
    if len({json.dumps(row, sort_keys=True) for row in baseline_contracts}) != 1:
        raise ValueError("baseline score shards change immutable inputs")
    if len({json.dumps(row, sort_keys=True) for row in optional_contracts}) != 1:
        raise ValueError("optional score shards change immutable inputs")
    if baseline_contracts[0] != optional_contracts[0]:
        raise ValueError("baseline and optional scores change frozen inputs")
    baseline_strict = [_strict_contract(row) for row in baseline_metadata]
    optional_strict = [_strict_contract(row) for row in optional_metadata]
    if any(row.get("candidate_specific_rgb_spatial_modes") is not False for row in baseline_strict):
        raise ValueError("baseline must be the no-RGB score")
    required_rgb = {
        "candidate_specific_rgb_spatial_modes": True,
        "candidate_spatial_dustbin_and_missing_pose_independent": True,
        "candidate_spatial_omitted_topk_mass_is_null": True,
        "candidate_spatial_query_materialization": (
            "required_at_least_one_heldout_verification_point"
        ),
        "candidate_spatial_semantics": (
            "per_view_normalized_continuous_gaussian_mixture_relative_to_"
            "grid_uniform_null_v1"
        ),
    }
    if any(
        any(row.get(key) != expected for key, expected in required_rgb.items())
        for row in optional_strict
    ):
        raise ValueError("optional score does not provide normalized RGB likelihood")
    return baseline_contracts[0]


def _load_targets(
    paths: Sequence[Path], *, expected_score_paths: Sequence[Path]
) -> dict[tuple[str, str, str, int], tuple[float, float]]:
    expected_hashes = [file_sha256_short(path) for path in expected_score_paths]
    index: dict[tuple[str, str, str, int], tuple[float, float]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if "metadata_json" not in payload.files:
                raise ValueError(f"{path}: target artifact has no metadata")
            arrays = {
                key: np.asarray(payload[key]).copy()
                for key in payload.files
                if key != "metadata_json"
            }
            metadata = json.loads(str(payload["metadata_json"].item()))
        missing = [key for key in _TARGET_FIELDS if key not in arrays]
        if missing:
            raise ValueError(f"{path}: target artifact lacks {missing}")
        if metadata.get("format") != _TARGET_FORMAT:
            raise ValueError(f"{path}: unsupported target format")
        if metadata.get("targets_joined_after_independent_score_selection") is not True:
            raise ValueError(f"{path}: targets were not joined after frozen scoring")
        listed_hashes = metadata.get("score_artifact_sha256")
        if not isinstance(listed_hashes, list) or set(listed_hashes) != set(expected_hashes):
            raise ValueError(f"{path}: target artifact does not match optional scores")
        row_count = len(arrays["query_ids"])
        if any(values.ndim == 0 or values.shape[0] != row_count for values in arrays.values()):
            raise ValueError(f"{path}: target fields are not row aligned")
        for row in range(row_count):
            key = _row_key(arrays, row)
            if key in index:
                raise ValueError(f"duplicate target hypothesis row: {key}")
            translation = float(arrays["translation_errors_m"][row])
            rotation = float(arrays["rotation_errors_deg"][row])
            if not np.isfinite(translation) or not np.isfinite(rotation):
                raise ValueError(f"{path}: target errors are not finite")
            index[key] = (translation, rotation)
    return index


def _selected_row(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray, *, context: str
) -> int:
    selected = rows[np.asarray(arrays["independent_score_top1"][rows], dtype=bool)]
    if len(selected) != 1:
        raise ValueError(f"{context}: frozen score must select exactly one hypothesis")
    return int(selected[0])


def _top_gap(values: np.ndarray, selected_row: int, rows: np.ndarray) -> float:
    selected = float(values[selected_row])
    other = rows[rows != int(selected_row)]
    return 0.0 if not len(other) else float(selected - np.max(values[other]))


def _rank_fraction(values: np.ndarray, selected_row: int, rows: np.ndarray) -> float:
    if len(rows) <= 1:
        return 0.0
    rank = 1 + int(np.count_nonzero(values[rows] > values[int(selected_row)]))
    return float((rank - 1) / (len(rows) - 1))


def _profile_consensus_features(
    *,
    optional: Mapping[str, np.ndarray],
    optional_rows: np.ndarray,
    optional_selected: int,
    optional_baseline_row: int,
) -> tuple[float, ...]:
    """Describe whether the mean-selected RGB pose survives robust summaries.

    This is intentionally a rank-only comparison.  The score fields have
    different units, so comparing their raw values would let one aggregation
    dominate merely through scale.  The baseline pose is already fixed before
    these values are read; no candidate or support view is reselected here.
    """

    rows = np.asarray(optional_rows, dtype=np.int64).reshape(-1)
    if len(rows) < 2:
        raise ValueError("profile consensus needs at least two frozen hypotheses")
    selected = int(optional_selected)
    baseline = int(optional_baseline_row)
    if selected not in set(rows.tolist()) or baseline not in set(rows.tolist()):
        raise ValueError("profile consensus rows are absent from the frozen group")
    selected_ranks: list[float] = []
    baseline_ranks: list[float] = []
    selected_margins: list[float] = []
    for field in _OPTIONAL_PROFILE_SCORE_FIELDS:
        values = np.asarray(optional[field], dtype=np.float64)
        local = values[rows]
        if np.any(~np.isfinite(local)):
            raise ValueError(f"{field}: RGB profile scores must be finite")
        selected_ranks.append(_rank_fraction(values, selected, rows))
        baseline_ranks.append(_rank_fraction(values, baseline, rows))
        selected_margins.append(_top_gap(values, selected, rows))
    ranks = np.asarray(selected_ranks, dtype=np.float64)
    baseline_ranks_array = np.asarray(baseline_ranks, dtype=np.float64)
    margins = np.asarray(selected_margins, dtype=np.float64)
    return (
        _rank_fraction(
            np.asarray(optional["independent_selection_scores"], dtype=np.float64),
            baseline,
            rows,
        ),
        float(np.max(ranks)),
        float(np.mean(ranks)),
        float(np.mean(ranks <= 1e-12)),
        float(np.mean(ranks < baseline_ranks_array)),
        float(np.mean(margins)),
        float(np.min(margins)),
    )


def _examples_from_scores(
    *,
    baseline: Mapping[str, np.ndarray],
    optional: Mapping[str, np.ndarray],
    targets: Mapping[tuple[str, str, str, int], tuple[float, float]],
) -> list[AbsoluteLikelihoodPromotionExample]:
    baseline_index = {
        _row_key(baseline, row): row for row in range(len(baseline["query_ids"]))
    }
    optional_index = {
        _row_key(optional, row): row for row in range(len(optional["query_ids"]))
    }
    if set(baseline_index) != set(optional_index):
        raise ValueError("baseline and optional scores do not cover the same rows")
    if set(optional_index) != set(targets):
        raise ValueError("optional scores and post-hoc targets do not cover the same rows")
    groups: dict[tuple[str, str, str], list[int]] = {}
    for row in range(len(optional["query_ids"])):
        groups.setdefault(_group_key(optional, row), []).append(row)
    examples: list[AbsoluteLikelihoodPromotionExample] = []
    for group_key in sorted(groups):
        optional_rows = np.asarray(groups[group_key], dtype=np.int64)
        keys = [_row_key(optional, int(row)) for row in optional_rows.tolist()]
        baseline_rows = np.asarray(
            [baseline_index[key] for key in keys], dtype=np.int64
        )
        if set(optional["hypothesis_indices"][optional_rows].tolist()) != set(
            baseline["hypothesis_indices"][baseline_rows].tolist()
        ):
            raise ValueError(f"{group_key}: hypothesis identities differ")
        optional_selected = _selected_row(
            optional, optional_rows, context=f"optional {group_key}"
        )
        baseline_selected = _selected_row(
            baseline, baseline_rows, context=f"baseline {group_key}"
        )
        optional_baseline_row = optional_index[_row_key(baseline, baseline_selected)]
        baseline_hypothesis = int(baseline["hypothesis_indices"][baseline_selected])
        optional_hypothesis = int(optional["hypothesis_indices"][optional_selected])
        baseline_target = targets[_row_key(baseline, baseline_selected)]
        optional_target = targets[_row_key(optional, optional_selected)]
        selection_scores = np.asarray(
            optional["independent_selection_scores"], dtype=np.float64
        )
        robust_scores = np.asarray(
            optional["independent_spatial_median_of_means_2x2"], dtype=np.float64
        )
        verification_count = float(optional["verification_point_counts"][optional_selected])
        effective_count = float(
            optional["independent_effective_point_counts"][optional_selected]
        )
        materialized_points = float(
            optional[
                "candidate_spatial_materialized_verification_point_counts"
            ][optional_selected]
        )
        materialized_views = float(
            optional["candidate_spatial_materialized_candidate_view_counts"]
            [optional_selected]
        )
        if verification_count <= 0.0 or effective_count < 0.0 or materialized_points < 0.0:
            raise ValueError(f"{group_key}: invalid target-free coverage counts")
        features = (
            float(
                selection_scores[optional_selected]
                - selection_scores[optional_baseline_row]
            ),
            _top_gap(selection_scores, optional_selected, optional_rows),
            _top_gap(robust_scores, optional_selected, optional_rows),
            _rank_fraction(robust_scores, optional_selected, optional_rows),
            float(optional["independent_evidence_coverages"][optional_selected]),
            float(effective_count / verification_count),
            float(materialized_points / verification_count),
            float(np.log1p(materialized_views / max(materialized_points, 1.0))),
            *_profile_consensus_features(
                optional=optional,
                optional_rows=optional_rows,
                optional_selected=optional_selected,
                optional_baseline_row=optional_baseline_row,
            ),
        )
        examples.append(
            AbsoluteLikelihoodPromotionExample(
                query_id=str(group_key[1]),
                split_name=str(group_key[0]),
                evaluation_label=str(group_key[2]),
                baseline_hypothesis_index=baseline_hypothesis,
                optional_hypothesis_index=optional_hypothesis,
                features=features,
                baseline_translation_m=float(baseline_target[0]),
                baseline_rotation_deg=float(baseline_target[1]),
                optional_translation_m=float(optional_target[0]),
                optional_rotation_deg=float(optional_target[1]),
            )
        )
    return examples


def _decision_rows(
    examples: Sequence[AbsoluteLikelihoodPromotionExample],
    probabilities: np.ndarray,
    decisions: np.ndarray,
    *,
    fold_assignments: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    if len(probabilities) != len(examples) or len(decisions) != len(examples):
        raise ValueError("decision rows are not example aligned")
    if fold_assignments is not None and len(fold_assignments) != len(examples):
        raise ValueError("fold assignments are not example aligned")
    rows: list[dict[str, object]] = []
    for index, (example, probability, promoted) in enumerate(
        zip(examples, probabilities.tolist(), decisions.tolist())
    ):
        row: dict[str, object] = {
            "query_id": str(example.query_id),
            "split_name": str(example.split_name),
            "evaluation_label": str(example.evaluation_label),
            "baseline_hypothesis_index": int(example.baseline_hypothesis_index),
            "optional_hypothesis_index": int(example.optional_hypothesis_index),
            "changed_hypothesis": bool(example.changed_hypothesis),
            "target_free_features": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, example.features)
            },
            "promotion_probability": float(probability),
            "promoted": bool(promoted),
            "selected_hypothesis_index": int(
                example.optional_hypothesis_index
                if promoted
                else example.baseline_hypothesis_index
            ),
            "target_beneficial_TARGET_ONLY": bool(example.target_beneficial),
            "baseline_translation_m_TARGET_ONLY": float(example.baseline_translation_m),
            "baseline_rotation_deg_TARGET_ONLY": float(example.baseline_rotation_deg),
            "optional_translation_m_TARGET_ONLY": float(example.optional_translation_m),
            "optional_rotation_deg_TARGET_ONLY": float(example.optional_rotation_deg),
        }
        if fold_assignments is not None:
            row["crossfit_fold"] = int(fold_assignments[index])
        rows.append(row)
    return rows


def _split_examples(
    examples: Sequence[AbsoluteLikelihoodPromotionExample], split_name: str
) -> list[AbsoluteLikelihoodPromotionExample]:
    return [example for example in examples if example.split_name == str(split_name)]


def _predict(
    model: AbsoluteLikelihoodPromotionModel | None,
    examples: Sequence[AbsoluteLikelihoodPromotionExample],
) -> np.ndarray:
    if model is None:
        return np.zeros((len(examples),), dtype=np.float64)
    features = np.asarray([example.features for example in examples], dtype=np.float64)
    return model.probabilities(features)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.fold_count) < 2:
        raise ValueError("fold_count must be at least two")
    if float(args.c_value) <= 0.0:
        raise ValueError("c_value must be positive")
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {summary_path}")

    baseline_paths = _paths(args.baseline_score_artifacts)
    optional_train_paths = _paths(args.optional_train_score_artifacts)
    optional_heldout_paths = _paths(args.optional_heldout_score_artifacts)
    train_target_paths = _paths(args.optional_train_targets)
    heldout_target_paths = _paths(args.optional_heldout_targets)
    baseline_scores, baseline_metadata = _merge_scores(baseline_paths)
    optional_train_scores, optional_train_metadata = _merge_scores(optional_train_paths)
    optional_heldout_scores, optional_heldout_metadata = _merge_scores(
        optional_heldout_paths
    )
    optional_scores = {
        key: np.concatenate(
            [optional_train_scores[key], optional_heldout_scores[key]], axis=0
        )
        for key in optional_train_scores
    }
    optional_keys = [
        _row_key(optional_scores, row)
        for row in range(len(optional_scores["query_ids"]))
    ]
    if len(optional_keys) != len(set(optional_keys)):
        raise ValueError("optional train and held-out artifacts overlap")
    immutable_contract = _validate_score_contracts(
        baseline_metadata=baseline_metadata,
        optional_metadata=[*optional_train_metadata, *optional_heldout_metadata],
    )
    targets = _load_targets(
        train_target_paths, expected_score_paths=optional_train_paths
    )
    heldout_targets = _load_targets(
        heldout_target_paths, expected_score_paths=optional_heldout_paths
    )
    if set(targets).intersection(heldout_targets):
        raise ValueError("train and held-out target artifacts overlap")
    targets.update(heldout_targets)
    examples = _examples_from_scores(
        baseline=baseline_scores,
        optional=optional_scores,
        targets=targets,
    )
    train = _split_examples(examples, "train")
    validation = _split_examples(examples, "validation")
    test = _split_examples(examples, "test")
    if not train or not validation or not test:
        raise ValueError("expected non-empty train, validation, and test examples")
    if len(train) + len(validation) + len(test) != len(examples):
        unknown = sorted({example.split_name for example in examples} - {"train", "validation", "test"})
        raise ValueError(f"unsupported split names in promotion audit: {unknown}")

    oof_probabilities, fold_assignments = crossfit_probabilities(
        train, fold_count=int(args.fold_count), c_value=float(args.c_value)
    )
    threshold, oof_decisions, oof_selection = select_oof_tail_safe_threshold(
        train, oof_probabilities
    )
    final_model = fit_final_model(train, c_value=float(args.c_value))
    train_replay_probabilities = _predict(final_model, train)
    train_replay_decisions = promotion_decisions(
        train, train_replay_probabilities, threshold=float(threshold)
    )
    validation_probabilities = _predict(final_model, validation)
    validation_decisions = promotion_decisions(
        validation, validation_probabilities, threshold=float(threshold)
    )
    test_probabilities = _predict(final_model, test)
    test_decisions = promotion_decisions(
        test, test_probabilities, threshold=float(threshold)
    )
    validation_metrics = policy_metrics(validation, validation_decisions)
    test_metrics = policy_metrics(test, test_decisions)
    validation_gate = tail_safe_gate(validation_metrics)
    test_gate = tail_safe_gate(test_metrics)
    promoted = bool(
        final_model is not None
        and float(threshold) < NO_PROMOTION_THRESHOLD
        and bool(validation_gate["passes"])
        and bool(test_gate["passes"])
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "absolute_likelihood_promotion_policy_v2.json"
    model_payload = {
        "format": "absolute_likelihood_promotion_policy_v2",
        "feature_schema": PROMOTION_FEATURE_SCHEMA,
        "model": None if final_model is None else final_model.to_dict(),
        "promotion_threshold": float(threshold),
        "feature_names": list(FEATURE_NAMES),
        "fit": {
            "split": "train_only",
            "query_grouped_oof": True,
            "fold_count": int(args.fold_count),
            "c_value": float(args.c_value),
            "oof_threshold_selection": oof_selection,
        },
        "immutable_contract": immutable_contract,
    }
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    decision_path = output_dir / "promotion_decisions.json"
    decision_payload = {
        "train_oof": _decision_rows(
            train,
            oof_probabilities,
            oof_decisions,
            fold_assignments=fold_assignments,
        ),
        "train_final_replay_TARGET_ONLY": _decision_rows(
            train, train_replay_probabilities, train_replay_decisions
        ),
        "validation": _decision_rows(
            validation, validation_probabilities, validation_decisions
        ),
        "test": _decision_rows(test, test_probabilities, test_decisions),
    }
    decision_path.write_text(json.dumps(decision_payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "train_oof_rgb_spatial_mode_consensus_promotion",
        "protocol": {
            "immutable_baseline_hypothesis": True,
            "optional_hypothesis_from_fixed_normalized_rgb_likelihood": True,
            "same_frozen_hypotheses_candidates_and_verification_budget": True,
            "target_free_inference_features_only": list(FEATURE_NAMES),
            "rgb_profile_consensus_is_rank_only": True,
            "rgb_profile_score_fields": list(_OPTIONAL_PROFILE_SCORE_FIELDS),
            "pnP_inlier_or_residual_feature_used": False,
            "pose_or_ground_truth_used_for_score_generation": False,
            "ground_truth_used_only_after_frozen_score_join": True,
            "gate_fit_split": "train_only",
            "threshold_selection": "query_grouped_oof_train_only",
            "validation_used_for_model_or_threshold_selection": False,
            "test_used_for_model_or_threshold_selection": False,
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "production_promoted": promoted,
        },
        "inputs": {
            "baseline_score_artifacts": [str(path) for path in baseline_paths],
            "baseline_score_artifact_sha256": [
                file_sha256_short(path) for path in baseline_paths
            ],
            "optional_train_score_artifacts": [str(path) for path in optional_train_paths],
            "optional_train_score_artifact_sha256": [
                file_sha256_short(path) for path in optional_train_paths
            ],
            "optional_heldout_score_artifacts": [
                str(path) for path in optional_heldout_paths
            ],
            "optional_heldout_score_artifact_sha256": [
                file_sha256_short(path) for path in optional_heldout_paths
            ],
            "optional_train_targets": [str(path) for path in train_target_paths],
            "optional_heldout_targets": [str(path) for path in heldout_target_paths],
            "immutable_contract": immutable_contract,
        },
        "train_oof": oof_selection,
        "train_final_replay_TARGET_ONLY": {
            "metrics": policy_metrics(train, train_replay_decisions),
            "gate": tail_safe_gate(policy_metrics(train, train_replay_decisions)),
        },
        "validation": {"metrics": validation_metrics, "gate": validation_gate},
        "test": {"metrics": test_metrics, "gate": test_gate},
        "outputs": {
            "policy": str(model_path),
            "policy_sha256": file_sha256_short(model_path),
            "decisions": str(decision_path),
            "decisions_sha256": file_sha256_short(decision_path),
            "summary": str(summary_path),
        },
        "next_action": (
            "eligible_for_production_safe_promotion_replay"
            if promoted
            else "diagnostic_only_do_not_promote"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
