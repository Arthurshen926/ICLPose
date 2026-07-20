"""Fit a train-only abstention threshold for independent cross-fit audit gain.

The command consumes target-free alignment artifacts and their post-hoc target
join.  It never regenerates proposals, identities, RGB modes, or poses.  The
threshold is evaluated with capture-sequence-grouped outer OOF before one
frozen deployment threshold is emitted for a held-out replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.eval_independent_crossfit_pose_alignment import (
    TARGET_ARTIFACT_FORMAT,
)
from feature_extract.tools.vfm.run_independent_crossfit_pose_alignment import (
    ARTIFACT_FORMAT,
)
from feature_extract.vfm.artifacts import file_sha256_short
from feature_extract.vfm.localization.crossfit_audit_promotion import (
    POLICY_FORMAT,
    CrossfitAuditPromotionExample,
    policy_metrics,
    promotion_decisions,
    select_tail_safe_threshold,
    sequence_grouped_oof_decisions,
    tail_safe_gate,
)


_ROW_FIELDS = (
    "query_ids",
    "split_names",
    "evaluation_labels",
    "hypothesis_indices",
)
_ALIGNMENT_FIELDS = _ROW_FIELDS + (
    "source_chosen",
    "optional_rank_top1",
    "optional_differs_from_source",
    "audit_score_deltas",
    "promotion_failures",
)
_TARGET_FIELDS = _ROW_FIELDS + (
    "source_translation_errors_m",
    "source_rotation_errors_deg",
    "candidate_translation_errors_m",
    "candidate_rotation_errors_deg",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_alignment_artifacts", required=True)
    parser.add_argument("--train_targets", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fold_count", type=int, default=5)
    parser.add_argument("--minimum_promotion_count", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths:
        raise ValueError("at least one artifact path is required")
    return paths


def _row_key(
    arrays: Mapping[str, np.ndarray], row: int
) -> tuple[str, str, str, int]:
    return (
        str(arrays["split_names"][row]),
        str(arrays["query_ids"][row]),
        str(arrays["evaluation_labels"][row]),
        int(arrays["hypothesis_indices"][row]),
    )


def _group_key(
    arrays: Mapping[str, np.ndarray], row: int
) -> tuple[str, str, str]:
    return (
        str(arrays["split_names"][row]),
        str(arrays["query_ids"][row]),
        str(arrays["evaluation_labels"][row]),
    )


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        if "metadata_json" not in payload.files:
            raise ValueError(f"{path}: artifact has no metadata")
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(payload["metadata_json"].item()))
    return arrays, metadata


def _runtime_contract(metadata: Mapping[str, object]) -> dict[str, object]:
    """Keep only query-independent scoring semantics in the policy manifest."""

    inputs = metadata.get("inputs")
    descriptor_evidence = metadata.get("descriptor_evidence")
    crossfit = metadata.get("crossfit")
    if not all(
        isinstance(value, Mapping) for value in (inputs, descriptor_evidence, crossfit)
    ):
        raise ValueError("alignment metadata lacks a runtime contract")
    input_hash_keys = (
        "hypothesis_artifact_sha256",
        "immutable_source_pose_artifact_sha256",
        "detector_query_cache_sha256",
        "proposals_sha256",
        "candidate_artifact_sha256",
        "fixed_candidate_prior_overlay_sha256",
        "projected_landmark_bank_sha256",
        "independent_verification_landmark_bank_sha256",
        "support_geometry_index_sha256",
        "maplet_support_index_sha256",
        "colmap_cameras_bin_sha256",
        "colmap_images_bin_sha256",
    )
    return {
        "alignment_format": metadata.get("format"),
        "alignment_version": metadata.get("version"),
        "hypothesis_compatibility_sha256": metadata.get(
            "hypothesis_compatibility_sha256"
        ),
        "score_config": metadata.get("score_config"),
        "refinement_config": metadata.get("refinement_config"),
        "observability_thresholds": metadata.get("observability_thresholds"),
        "hypothesis_scope": metadata.get("hypothesis_scope"),
        "descriptor_evidence": {
            key: descriptor_evidence.get(key)
            for key in (
                "source",
                "view_geometry_mode",
                "candidate_prior_source",
                "explicit_candidate_null_probability",
                "candidate_specific_rgb_spatial_modes",
                "candidate_spatial_semantics",
            )
        },
        "crossfit": {
            key: crossfit.get(key)
            for key in (
                "query_token_disjoint",
                "physical_track_disjoint",
                "maplet_cluster_disjoint",
                "point_fold_seed",
                "landmark_fold_seed",
                "rank_selects_hypothesis_only",
                "audit_compares_frozen_optional_to_immutable_source_only",
                "denominator_fixed_across_hypotheses_within_each_role",
                "candidate_spatial_role_materialization_required",
                "candidate_spatial_missing_is_neutral_unknown",
                "promotion_requires_distinct_optional_pose",
                "source_pose_policy",
                "audit_compares_frozen_optional_to_fixed_source_only",
                "immutable_source_override",
            )
        },
        "input_hashes": {key: inputs.get(key) for key in input_hash_keys},
    }


def _load_alignment(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays, metadata = _load_npz(path)
    missing = [key for key in _ALIGNMENT_FIELDS if key not in arrays]
    if missing:
        raise ValueError(f"{path}: alignment artifact lacks {missing}")
    if metadata.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"{path}: unsupported alignment format")
    if metadata.get("contains_target_fields") is not False:
        raise ValueError(f"{path}: alignment artifact already contains targets")
    if metadata.get("pose_or_ground_truth_used_for_scoring") is not False:
        raise ValueError(f"{path}: alignment scorer used target pose")
    if str(metadata.get("split_filter")) != "train":
        raise ValueError(f"{path}: calibration artifacts must be train-only")
    descriptor_evidence = metadata.get("descriptor_evidence")
    if not isinstance(descriptor_evidence, Mapping) or (
        descriptor_evidence.get("candidate_specific_rgb_spatial_modes") is not True
    ):
        raise ValueError(f"{path}: calibration requires candidate RGB spatial modes")
    crossfit = metadata.get("crossfit")
    if not isinstance(crossfit, Mapping) or (
        crossfit.get("rank_selects_hypothesis_only") is not True
        or crossfit.get("audit_compares_frozen_optional_to_fixed_source_only")
        is not True
        or crossfit.get("candidate_spatial_role_materialization_required") is not True
    ):
        raise ValueError(f"{path}: alignment violates independent RGB audit contract")
    thresholds = metadata.get("score_thresholds")
    if not isinstance(thresholds, Mapping) or float(
        thresholds.get("minimum_audit_gain", float("nan"))
    ) != 0.0:
        raise ValueError(f"{path}: calibration must start from zero audit threshold")
    row_count = int(metadata.get("row_count", -1))
    if row_count <= 0 or any(
        values.ndim == 0 or values.shape[0] != row_count
        for values in arrays.values()
    ):
        raise ValueError(f"{path}: alignment arrays are not row aligned")
    keys = [_row_key(arrays, row) for row in range(row_count)]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate hypothesis rows")
    if set(arrays["split_names"].astype(str).tolist()) != {"train"}:
        raise ValueError(f"{path}: alignment contains a non-train query")
    return arrays, metadata


def _merge_alignment(
    paths: Sequence[Path],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, object]]:
    loaded = [_load_alignment(path) for path in paths]
    schemas = [set(arrays) for arrays, _metadata in loaded]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ValueError("alignment artifacts do not share an array schema")
    contracts = [_runtime_contract(metadata) for _arrays, metadata in loaded]
    canonical = [json.dumps(contract, sort_keys=True) for contract in contracts]
    if len(set(canonical)) != 1:
        raise ValueError("alignment shards change query-independent semantics")
    arrays = {
        key: np.concatenate([payload[key] for payload, _metadata in loaded], axis=0)
        for key in sorted(schemas[0])
    }
    keys = [_row_key(arrays, row) for row in range(len(arrays["query_ids"]))]
    if len(keys) != len(set(keys)):
        raise ValueError("merged alignment artifacts contain duplicate rows")
    return arrays, [metadata for _payload, metadata in loaded], contracts[0]


def _load_targets(
    paths: Sequence[Path], *, expected_alignment_paths: Sequence[Path]
) -> dict[tuple[str, str, str, int], tuple[float, float, float, float]]:
    expected_hashes = {file_sha256_short(path) for path in expected_alignment_paths}
    targets: dict[tuple[str, str, str, int], tuple[float, float, float, float]] = {}
    for path in paths:
        arrays, metadata = _load_npz(path)
        missing = [key for key in _TARGET_FIELDS if key not in arrays]
        if missing:
            raise ValueError(f"{path}: target artifact lacks {missing}")
        if metadata.get("format") != TARGET_ARTIFACT_FORMAT:
            raise ValueError(f"{path}: unsupported target format")
        if metadata.get("targets_joined_after_selection") is not True:
            raise ValueError(f"{path}: targets were not joined after selection")
        listed_hashes = metadata.get("alignment_artifact_sha256")
        if not isinstance(listed_hashes, list) or set(listed_hashes) != expected_hashes:
            raise ValueError(f"{path}: targets do not match calibration alignments")
        row_count = len(arrays["query_ids"])
        if any(
            values.ndim == 0 or values.shape[0] != row_count
            for values in arrays.values()
        ):
            raise ValueError(f"{path}: target arrays are not row aligned")
        for row in range(row_count):
            key = _row_key(arrays, row)
            values = tuple(
                float(arrays[field][row])
                for field in (
                    "source_translation_errors_m",
                    "source_rotation_errors_deg",
                    "candidate_translation_errors_m",
                    "candidate_rotation_errors_deg",
                )
            )
            if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
                raise ValueError(f"{path}: target errors are not finite")
            if key in targets:
                raise ValueError(f"{path}: duplicate target hypothesis row")
            targets[key] = values
    return targets


def _non_audit_failures(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in str(value).split(",")
        if token and token != "audit_likelihood_gain"
    )


def _examples_from_alignment(
    *,
    arrays: Mapping[str, np.ndarray],
    targets: Mapping[tuple[str, str, str, int], tuple[float, float, float, float]],
) -> list[CrossfitAuditPromotionExample]:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for row in range(len(arrays["query_ids"])):
        groups.setdefault(_group_key(arrays, row), []).append(row)
    examples: list[CrossfitAuditPromotionExample] = []
    for group_key in sorted(groups):
        rows = np.asarray(groups[group_key], dtype=np.int64)
        source_rows = rows[np.asarray(arrays["source_chosen"][rows], dtype=bool)]
        optional_rows = rows[
            np.asarray(arrays["optional_rank_top1"][rows], dtype=bool)
        ]
        if len(source_rows) != 1 or len(optional_rows) != 1:
            raise ValueError(f"{group_key}: source/optional choice is not unique")
        source_row = int(source_rows[0])
        optional_row = int(optional_rows[0])
        deltas = np.asarray(arrays["audit_score_deltas"][rows], dtype=np.float64)
        if not np.all(np.isfinite(deltas)) or not np.allclose(
            deltas, deltas[0], rtol=0.0, atol=0.0
        ):
            raise ValueError(f"{group_key}: audit delta differs by hypothesis row")
        changed = np.asarray(arrays["optional_differs_from_source"][rows], dtype=bool)
        if not np.all(changed == changed[0]):
            raise ValueError(f"{group_key}: optional/source identity differs by row")
        failures = arrays["promotion_failures"][rows].astype(str)
        if not np.all(failures == failures[0]):
            raise ValueError(f"{group_key}: promotion failures differ by row")
        source_key = _row_key(arrays, source_row)
        optional_key = _row_key(arrays, optional_row)
        if source_key not in targets or optional_key not in targets:
            raise ValueError(f"{group_key}: target join lacks source or optional pose")
        source_target = targets[source_key]
        optional_target = targets[optional_key]
        examples.append(
            CrossfitAuditPromotionExample(
                query_id=str(group_key[1]),
                split_name=str(group_key[0]),
                evaluation_label=str(group_key[2]),
                source_hypothesis_index=int(arrays["hypothesis_indices"][source_row]),
                optional_hypothesis_index=int(
                    arrays["hypothesis_indices"][optional_row]
                ),
                audit_score_delta=float(deltas[0]),
                eligible_without_audit_gain=bool(changed[0])
                and not _non_audit_failures(failures[0]),
                source_translation_m=float(source_target[0]),
                source_rotation_deg=float(source_target[1]),
                optional_translation_m=float(optional_target[2]),
                optional_rotation_deg=float(optional_target[3]),
            )
        )
    if set(targets) != {
        _row_key(arrays, row) for row in range(len(arrays["query_ids"]))
    }:
        raise ValueError("targets and alignment rows do not cover the same hypotheses")
    return examples


def _decision_rows(
    examples: Sequence[CrossfitAuditPromotionExample],
    decisions: Sequence[bool],
    *,
    fold_assignments: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    selected = np.asarray(decisions, dtype=bool).reshape(-1)
    if len(selected) != len(examples):
        raise ValueError("decision rows are not example aligned")
    if fold_assignments is not None and len(fold_assignments) != len(examples):
        raise ValueError("fold assignments are not example aligned")
    rows: list[dict[str, object]] = []
    for index, (example, promoted) in enumerate(zip(examples, selected.tolist())):
        row = {
            "query_id": str(example.query_id),
            "split_name": str(example.split_name),
            "evaluation_label": str(example.evaluation_label),
            "sequence_group": str(example.sequence_group),
            "source_hypothesis_index": int(example.source_hypothesis_index),
            "optional_hypothesis_index": int(example.optional_hypothesis_index),
            "optional_differs_from_source": bool(example.changed_hypothesis),
            "eligible_without_audit_gain": bool(
                example.eligible_without_audit_gain
            ),
            "audit_score_delta": float(example.audit_score_delta),
            "promoted": bool(promoted),
            "source_translation_m_TARGET_ONLY": float(example.source_translation_m),
            "source_rotation_deg_TARGET_ONLY": float(example.source_rotation_deg),
            "optional_translation_m_TARGET_ONLY": float(
                example.optional_translation_m
            ),
            "optional_rotation_deg_TARGET_ONLY": float(example.optional_rotation_deg),
        }
        if fold_assignments is not None:
            row["outer_oof_fold"] = int(fold_assignments[index])
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.fold_count) < 2:
        raise ValueError("fold count must be at least two")
    if int(args.minimum_promotion_count) <= 0:
        raise ValueError("minimum promotion count must be positive")
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite {summary_path}")

    alignment_paths = _paths(args.train_alignment_artifacts)
    target_paths = _paths(args.train_targets)
    arrays, _metadata, runtime_contract = _merge_alignment(alignment_paths)
    targets = _load_targets(target_paths, expected_alignment_paths=alignment_paths)
    examples = _examples_from_alignment(arrays=arrays, targets=targets)
    if {example.split_name for example in examples} != {"train"}:
        raise ValueError("calibration examples must contain only train queries")

    oof_decisions, fold_assignments, fold_thresholds, oof_audit = (
        sequence_grouped_oof_decisions(
            examples,
            fold_count=int(args.fold_count),
            minimum_promotion_count=int(args.minimum_promotion_count),
        )
    )
    deployment_threshold, deployment_decisions, deployment_audit = (
        select_tail_safe_threshold(
            examples,
            minimum_promotion_count=int(args.minimum_promotion_count),
        )
    )
    oof_gate = oof_audit["gate"]
    deployment_gate = deployment_audit["gate"]
    promoted = bool(
        deployment_threshold is not None
        and oof_gate["passes"]
        and deployment_gate["passes"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "crossfit_audit_gain_promotion_policy_v1.json"
    policy = {
        "format": POLICY_FORMAT,
        "minimum_audit_gain": deployment_threshold,
        "fit": {
            "split": "train_only",
            "capture_sequence_grouped_outer_oof": True,
            "fold_count": int(args.fold_count),
            "minimum_promotion_count": int(args.minimum_promotion_count),
            "fold_selected_minimum_audit_gains": list(fold_thresholds),
            "outer_oof": oof_audit,
            "full_train_threshold_selection_TARGET_ONLY": deployment_audit,
        },
        "runtime_contract": runtime_contract,
        "eligible_for_heldout_replay": promoted,
    }
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
    decisions_path = output_dir / "train_oof_decisions.json"
    decisions = {
        "outer_oof_TARGET_ONLY": _decision_rows(
            examples, oof_decisions, fold_assignments=fold_assignments
        ),
        "full_train_replay_TARGET_ONLY": _decision_rows(
            examples, deployment_decisions
        ),
    }
    decisions_path.write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n")
    summary = {
        "stage": "train_only_crossfit_audit_gain_promotion",
        "protocol": {
            "pose_or_ground_truth_used_for_scoring": False,
            "ground_truth_joined_only_after_target_free_alignment": True,
            "identity_and_pose_hypotheses_frozen_before_calibration": True,
            "threshold_feature": "crossfit_audit_score_delta_only",
            "outer_oof_group": "capture_sequence",
            "validation_used_for_threshold_selection": False,
            "test_used_for_threshold_selection": False,
            "image_retrieval": False,
            "submap": False,
            "render": False,
            "eligible_for_heldout_replay": promoted,
        },
        "inputs": {
            "train_alignment_artifacts": [str(path) for path in alignment_paths],
            "train_alignment_artifact_sha256": [
                file_sha256_short(path) for path in alignment_paths
            ],
            "train_targets": [str(path) for path in target_paths],
        },
        "train_outer_oof_TARGET_ONLY": oof_audit,
        "full_train_replay_TARGET_ONLY": {
            "minimum_audit_gain": deployment_threshold,
            "metrics": policy_metrics(examples, deployment_decisions),
            "gate": tail_safe_gate(
                policy_metrics(examples, deployment_decisions),
                minimum_promotion_count=int(args.minimum_promotion_count),
            ),
        },
        "outputs": {
            "policy": str(policy_path),
            "policy_sha256": file_sha256_short(policy_path),
            "decisions": str(decisions_path),
            "decisions_sha256": file_sha256_short(decisions_path),
            "summary": str(summary_path),
        },
        "next_action": (
            "run_one_frozen_validation_replay"
            if promoted
            else "diagnostic_only_do_not_promote"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
