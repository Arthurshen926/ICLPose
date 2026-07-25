"""Target-side audit for V5 candidate-projected absolute-context diagnostics.

The scorer writes no labels and marks either separately trained V5 branch as
uncalibrated raw scores.  This program is the only component that joins those
target-free artifacts to frozen pose errors.  It compares the visual V5
checkpoint against the matched position-only control under the same fixed
hypotheses, candidates, support views, query rows, and explicit null
denominator.

Passing this audit is only evidence for a later train-only likelihood
calibration.  It never promotes the raw score into PnP or test-time selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feature_extract.tools.vfm.audit_frozen_multiscale_candidate_pose_evidence import (
    TARGET_FORMAT,
    _baseline_rows,
    _groups,
    _paired,
    _per_query_rows,
    _row_keys,
    _split_summary,
    _validate_alpha_zero,
    _write_csv,
)
from feature_extract.vfm.artifacts import file_sha256_short


SCORE_FORMAT = "v5_dynamic_absolute_context_pose_scores_v1"
_EPSILON = 1e-12
_STATISTIC_FIELDS = {
    "mean": "raw_score_means",
    "median": "raw_score_medians",
    "worst_quartile_mean": "raw_score_worst_quartile_means",
    "spatial_median_of_means_2x2": "raw_score_spatial_median_of_means_2x2",
}
_FORMAL_P1_POINT_COUNTS = {
    "alike_high_detail": 64,
    "radio_intermediate_uniform_context": 64,
    "radio_final_uniform_context": 64,
}
_EVIDENCE_BRANCHES = ("geometry", "strict_identity")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual_score_artifacts", required=True)
    parser.add_argument("--position_score_artifacts", required=True)
    parser.add_argument("--target_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--score_statistic",
        choices=tuple(_STATISTIC_FIELDS),
        default="spatial_median_of_means_2x2",
    )
    parser.add_argument(
        "--verification_protocol",
        choices=("any", "formal_p1_mixed"),
        default="any",
        help="reject P1.5 ALIKE-only artifacts when auditing the formal P1 protocol",
    )
    parser.add_argument(
        "--evidence_branch",
        choices=_EVIDENCE_BRANCHES,
        default="geometry",
        help="require one separately scored V5 branch; visual and control artifacts cannot mix branches",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _paths(value: str) -> tuple[Path, ...]:
    paths = tuple(Path(item.strip()) for item in str(value).split(",") if item.strip())
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("score artifact paths must be non-empty and unique")
    return paths


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _normalized_strict_contract(metadata: Mapping[str, object]) -> object:
    """Make the post-P1.5 formal-point marker backward-compatible.

    The marker was added after the first P1.5 shards had already been scored.
    Its historical absence means ``False`` rather than a different evidence
    protocol.  A formal P1 artifact remains distinct because it sets it True.
    """

    strict = metadata.get("strict_frozen_evidence_contract")
    if not isinstance(strict, Mapping):
        return strict
    output = dict(strict)
    output["formal_p1_mixed_multiscale_points"] = bool(
        output.get("formal_p1_mixed_multiscale_points", False)
    )
    output["strict_identity_raw_residual_only"] = bool(
        output.get("strict_identity_raw_residual_only", False)
    )
    return output


def _load_score(
    path: Path,
    *,
    expected_family: str,
    verification_protocol: str = "any",
    evidence_branch: str = "geometry",
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    fields = (
        "query_ids",
        "split_names",
        "evaluation_labels",
        "hypothesis_indices",
        "source_chosen_for_optional_pose",
        "baseline_score_top1",
        "baseline_selection_scores",
        "score_names",
        "raw_score_means",
        "raw_score_medians",
        "raw_score_worst_quartile_means",
        "raw_score_spatial_median_of_means_2x2",
        "effective_point_counts",
        "effective_view_masses",
    )
    with np.load(Path(path), allow_pickle=False) as payload:
        missing = sorted(set(fields).difference(payload.files))
        if missing or "metadata_json" not in payload.files:
            raise ValueError(f"{path}: incomplete V5 dynamic score artifact ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        formal_arrays = {
            field: np.asarray(payload[field]).copy()
            for field in ("verification_point_sources", "verification_source_detector_rows")
            if field in payload.files
        }
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    branch = str(evidence_branch)
    if branch not in _EVIDENCE_BRANCHES:
        raise ValueError(f"unsupported V5 evidence branch: {branch}")
    strict_identity_branch = branch == "strict_identity"
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != SCORE_FORMAT
        or metadata.get("contains_target_fields") is not False
        or metadata.get("pose_or_ground_truth_used_for_scoring") is not False
        or metadata.get("supervision_arrays_loaded") is not False
        or metadata.get("diagnostic_only") is not True
        or metadata.get("promotion_allowed") is not False
        or str(metadata.get("model_family")) != str(expected_family)
        or metadata.get("raw_score_is_calibrated_independent_pose_likelihood") is not False
        or str(metadata.get("evidence_branch", "geometry")) != branch
        or bool(metadata.get("identity_head_used")) is not strict_identity_branch
    ):
        raise ValueError(f"{path}: not the declared target-free V5 diagnostic")
    semantics = str(metadata.get("raw_score_semantics", ""))
    expected_semantic_prefix = (
        "strict_exact_track_identity_raw_candidate_residual"
        if strict_identity_branch
        else "geometry_head_raw_candidate_reranking_residual"
    )
    if not semantics.startswith(expected_semantic_prefix):
        raise ValueError(f"{path}: V5 raw score semantics do not match the requested branch")
    strict = _normalized_strict_contract(metadata)
    required = {
        "heldout_query_rows": True,
        "fixed_global_topl": True,
        "explicit_null_mass": True,
        "candidate_identity_fixed_across_hypotheses": True,
        "candidate_reselection_per_pose": False,
        "support_reselection_per_pose": False,
        "support_view_descriptor_averaging": False,
        "candidate_specific_query_projection": True,
        "query_center_gaussian_fallback": False,
        "image_retrieval_or_submap_used": False,
        "render": False,
        "geometry_head_only": not strict_identity_branch,
        "identity_head_not_used": not strict_identity_branch,
        "strict_identity_raw_residual_only": strict_identity_branch,
        "raw_scores_calibrated_as_independent_pose_likelihood": False,
    }
    if not isinstance(strict, Mapping) or any(strict.get(key) is not value for key, value in required.items()):
        raise ValueError(f"{path}: V5 frozen evidence contract is incomplete")
    hypothesis_scope = metadata.get("hypothesis_scope")
    if (
        not isinstance(hypothesis_scope, Mapping)
        or hypothesis_scope.get("all_frozen_hypotheses") is not True
        or int(hypothesis_scope.get("development_prefix_limit", -1)) != 0
    ):
        raise ValueError(f"{path}: V5 audit requires every frozen hypothesis, not a prefix")
    count = int(metadata.get("row_count", -1))
    if count <= 0 or any(np.asarray(arrays[field]).shape[0] != count for field in fields if field != "score_names"):
        raise ValueError(f"{path}: V5 score rows are inconsistent")
    names = np.asarray(arrays["score_names"]).astype(str).reshape(-1)
    if names.tolist() != ["all_scales", "radio_final", "radio_intermediate", "alike"]:
        raise ValueError(f"{path}: V5 raw score family order is not immutable")
    for field in _STATISTIC_FIELDS.values():
        values = np.asarray(arrays[field])
        if values.shape != (count, len(names)) or not np.isfinite(values).all():
            raise ValueError(f"{path}: {field} is non-finite or family-misaligned")
    for field in ("effective_point_counts", "effective_view_masses"):
        values = np.asarray(arrays[field])
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"{path}: {field} is invalid")
    keys = _row_keys(arrays)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: V5 dynamic score repeats query/hypothesis rows")
    if not str(metadata.get("frozen_query_evidence_sha256", "")):
        raise ValueError(f"{path}: V5 score lacks its immutable evidence digest")
    if str(verification_protocol) == "formal_p1_mixed":
        selection = metadata.get("verification_point_selection")
        if (
            strict.get("formal_p1_mixed_multiscale_points") is not True
            or not isinstance(selection, Mapping)
            or selection.get("source")
            != "formal_p1_mixed_multiscale_verification_points_v1"
            or int(selection.get("point_count", -1)) != sum(_FORMAL_P1_POINT_COUNTS.values())
            or selection.get("point_sources") != _FORMAL_P1_POINT_COUNTS
            or selection.get("candidate_fit_rows_excluded_from_alike") is not True
            or selection.get("dense_point_detector_rows_are_sentinel_minus_one") is not True
            or selection.get("development_test_source_points_excluded") is not True
        ):
            raise ValueError(f"{path}: not a formal 64/64/64 P1 mixed-evidence score")
        sources = formal_arrays.get("verification_point_sources")
        detector_rows = formal_arrays.get("verification_source_detector_rows")
        if (
            sources is None
            or detector_rows is None
            or sources.shape != detector_rows.shape
            or sources.shape != (sum(_FORMAL_P1_POINT_COUNTS.values()),)
        ):
            raise ValueError(f"{path}: formal P1 score lacks its source-level evidence arrays")
        source_values = np.asarray(sources).astype(str)
        observed = {
            source: int(np.count_nonzero(source_values == source))
            for source in sorted(set(source_values.tolist()))
        }
        alike = source_values == "alike_high_detail"
        if (
            observed != _FORMAL_P1_POINT_COUNTS
            or np.any(np.asarray(detector_rows, dtype=np.int64)[alike] < 0)
            or np.any(np.asarray(detector_rows, dtype=np.int64)[~alike] != -1)
        ):
            raise ValueError(f"{path}: formal P1 point-source provenance is inconsistent")
    return arrays, metadata


def _merge_scores(
    paths: Sequence[Path],
    *,
    expected_family: str,
    verification_protocol: str = "any",
    evidence_branch: str = "geometry",
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], tuple[tuple[str, str, str, int], ...]]:
    loaded = [
        _load_score(
            path,
            expected_family=expected_family,
            verification_protocol=verification_protocol,
            evidence_branch=evidence_branch,
        )
        for path in paths
    ]
    names = np.asarray(loaded[0][0]["score_names"]).astype(str)
    fields = tuple(loaded[0][0])
    compatibility = {
        "score_names": names.tolist(),
        "strict_frozen_evidence_contract": _normalized_strict_contract(loaded[0][1]),
        "model_family": loaded[0][1].get("model_family"),
        "evidence_branch": loaded[0][1].get("evidence_branch", "geometry"),
        "identity_head_used": loaded[0][1].get("identity_head_used"),
        "model_checkpoint_contract": loaded[0][1].get("model_checkpoint_contract"),
        "source_cache_lineage": loaded[0][1].get("source_cache_lineage"),
        "raw_score_semantics": loaded[0][1].get("raw_score_semantics"),
        "raw_score_is_calibrated_independent_pose_likelihood": loaded[0][1].get(
            "raw_score_is_calibrated_independent_pose_likelihood"
        ),
    }
    fingerprint = _canonical_hash(compatibility)
    row_fields = tuple(field for field in fields if field != "score_names")
    merged_parts: dict[str, list[np.ndarray]] = {field: [] for field in row_fields}
    metadata: list[dict[str, object]] = []
    for arrays, item_metadata in loaded:
        item_compatibility = {
            "score_names": np.asarray(arrays["score_names"]).astype(str).tolist(),
            "strict_frozen_evidence_contract": _normalized_strict_contract(item_metadata),
            "model_family": item_metadata.get("model_family"),
            "evidence_branch": item_metadata.get("evidence_branch", "geometry"),
            "identity_head_used": item_metadata.get("identity_head_used"),
            "model_checkpoint_contract": item_metadata.get("model_checkpoint_contract"),
            "source_cache_lineage": item_metadata.get("source_cache_lineage"),
            "raw_score_semantics": item_metadata.get("raw_score_semantics"),
            "raw_score_is_calibrated_independent_pose_likelihood": item_metadata.get(
                "raw_score_is_calibrated_independent_pose_likelihood"
            ),
        }
        if _canonical_hash(item_compatibility) != fingerprint:
            raise ValueError("V5 dynamic score artifacts have incompatible model/evidence contracts")
        for field in row_fields:
            merged_parts[field].append(np.asarray(arrays[field]))
        metadata.append(item_metadata)
    merged = {field: np.concatenate(values, axis=0) for field, values in merged_parts.items()}
    merged["score_names"] = names
    keys = _row_keys(merged)
    if len(keys) != len(set(keys)):
        raise ValueError("merged V5 dynamic score shards repeat query/hypothesis rows")
    return merged, metadata, keys


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
            raise ValueError(f"{path}: incomplete target artifact ({missing})")
        arrays = {field: np.asarray(payload[field]).copy() for field in fields}
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != TARGET_FORMAT
        or metadata.get("contains_target_fields") is not True
    ):
        raise ValueError("target artifact is not the post-score GT join")
    count = int(metadata.get("row_count", -1))
    if count <= 0 or any(np.asarray(value).shape[0] != count for value in arrays.values()):
        raise ValueError("target artifact arrays are not row aligned")
    translation = np.asarray(arrays["translation_errors_m"], dtype=np.float64)
    rotation = np.asarray(arrays["rotation_errors_deg"], dtype=np.float64)
    if (
        not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
        or np.any(translation < 0.0)
        or np.any(rotation < 0.0)
    ):
        raise ValueError("target artifact errors are invalid")
    return arrays, metadata


def _validate_target_lineage(
    *, score_metadata: Sequence[Mapping[str, object]], target_metadata: Mapping[str, object]
) -> None:
    target_hypotheses = set(str(value) for value in target_metadata.get("hypothesis_artifact_sha256", []))
    target_baselines = set(str(value) for value in target_metadata.get("score_artifact_sha256", []))
    for metadata in score_metadata:
        inputs = metadata.get("inputs")
        if not isinstance(inputs, Mapping):
            raise ValueError("V5 score metadata lacks an input manifest")
        hypothesis_sha = str(
            (inputs.get("hypothesis_artifact") or {}).get("sha256", "")
        )
        baseline_sha = str(
            (inputs.get("baseline_score_artifact") or {}).get("sha256", "")
        )
        if hypothesis_sha not in target_hypotheses or baseline_sha not in target_baselines:
            raise ValueError("V5 dynamic score/target lineage differs from frozen S0")


def _validate_visual_control_pairing(
    *,
    visual_metadata: Sequence[Mapping[str, object]],
    position_metadata: Sequence[Mapping[str, object]],
) -> None:
    visual = {str(item.get("query_id")): item for item in visual_metadata}
    position = {str(item.get("query_id")): item for item in position_metadata}
    if not visual or set(visual) != set(position) or len(visual) != len(visual_metadata):
        raise ValueError("visual/control V5 score shards do not cover the same unique queries")
    for query_id in sorted(visual):
        left = visual[query_id]
        right = position[query_id]
        if (
            left.get("frozen_query_evidence_sha256")
            != right.get("frozen_query_evidence_sha256")
            or left.get("verification_point_selection") != right.get("verification_point_selection")
            or left.get("strict_frozen_evidence_contract")
            != right.get("strict_frozen_evidence_contract")
        ):
            raise ValueError("visual/control V5 scores differ in frozen query evidence")


def _paired_rank(
    visual_rows: Sequence[Mapping[str, object]], position_rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    def key(row: Mapping[str, object]) -> tuple[str, str, str]:
        return (str(row["split_name"]), str(row["evaluation_label"]), str(row["query_id"]))

    visual = {key(row): row for row in visual_rows}
    position = {key(row): row for row in position_rows}
    if not visual or set(visual) != set(position):
        raise ValueError("visual/control V5 per-query rows are not aligned")
    oracle_delta = np.asarray(
        [
            float(visual[item]["oracle_score_rank"]) - float(position[item]["oracle_score_rank"])
            for item in sorted(visual)
        ],
        dtype=np.float64,
    )
    best10_pairs = [
        (
            visual[item].get("best_10cm_rank"),
            position[item].get("best_10cm_rank"),
        )
        for item in sorted(visual)
    ]
    valid_best10 = [(left, right) for left, right in best10_pairs if left is not None and right is not None]
    best10_delta = np.asarray(
        [float(left) - float(right) for left, right in valid_best10], dtype=np.float64
    )
    return {
        "oracle_rank_wins": int(np.count_nonzero(oracle_delta < -_EPSILON)),
        "oracle_rank_losses": int(np.count_nonzero(oracle_delta > _EPSILON)),
        "oracle_rank_ties": int(np.count_nonzero(np.abs(oracle_delta) <= _EPSILON)),
        "median_oracle_rank_delta": float(np.median(oracle_delta)),
        "mean_oracle_rank_delta": float(np.mean(oracle_delta)),
        "best_10cm_rank_pair_count": int(len(best10_delta)),
        "best_10cm_rank_wins": int(np.count_nonzero(best10_delta < -_EPSILON)),
        "best_10cm_rank_losses": int(np.count_nonzero(best10_delta > _EPSILON)),
        "best_10cm_rank_ties": int(np.count_nonzero(np.abs(best10_delta) <= _EPSILON)),
        "median_best_10cm_rank_delta": (
            None if len(best10_delta) == 0 else float(np.median(best10_delta))
        ),
    }


def _primary_gate(
    *, visual: Mapping[str, object], position: Mapping[str, object], paired_rank: Mapping[str, object]
) -> dict[str, object]:
    visual_translation_p90 = float(visual["p90_selected_translation_cm"])
    position_translation_p90 = float(position["p90_selected_translation_cm"])
    visual_median_rank = float(visual["median_oracle_score_rank"])
    position_median_rank = float(position["median_oracle_score_rank"])
    visual_p90_rank = float(visual["p90_oracle_score_rank"])
    position_p90_rank = float(position["p90_oracle_score_rank"])
    visual_catastrophic = int(visual["catastrophic_1m_count"])
    position_catastrophic = int(position["catastrophic_1m_count"])
    control_signal = {
        "visual_median_oracle_rank_below_position": visual_median_rank < position_median_rank,
        "visual_p90_oracle_rank_not_worse": visual_p90_rank <= position_p90_rank,
        "visual_p90_translation_not_worse": visual_translation_p90 <= position_translation_p90,
        "visual_catastrophic_tail_not_worse": visual_catastrophic <= position_catastrophic,
        "visual_oracle_rank_wins_exceed_losses": int(paired_rank["oracle_rank_wins"])
        > int(paired_rank["oracle_rank_losses"]),
    }
    return {
        **control_signal,
        "validation_rank_median_under_35": visual_median_rank < 35.0,
        "uncalibrated_probe_signal_passed": bool(all(control_signal.values())),
        "eligible_for_pose_or_pnp_promotion": False,
        "reason": (
            "raw V5 branch scores are intentionally uncalibrated candidate residuals; "
            "even a signal pass only authorizes train-only likelihood calibration"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not bool(args.force):
        raise FileExistsError(f"refusing to overwrite existing audit: {summary_path}")
    visual_paths = _paths(args.visual_score_artifacts)
    position_paths = _paths(args.position_score_artifacts)
    visual, visual_metadata, visual_keys = _merge_scores(
        visual_paths,
        expected_family="bidirectional_absolute_dual_head_raw_layout_visual_v5",
        verification_protocol=str(args.verification_protocol),
        evidence_branch=str(args.evidence_branch),
    )
    position, position_metadata, position_keys = _merge_scores(
        position_paths,
        expected_family="bidirectional_absolute_dual_head_raw_layout_position_control_v5",
        verification_protocol=str(args.verification_protocol),
        evidence_branch=str(args.evidence_branch),
    )
    if visual_keys != position_keys:
        raise ValueError("visual/control V5 score rows have different frozen hypothesis identities")
    _validate_visual_control_pairing(
        visual_metadata=visual_metadata, position_metadata=position_metadata
    )
    targets, target_metadata = _load_targets(Path(args.target_artifact))
    _validate_target_lineage(score_metadata=visual_metadata, target_metadata=target_metadata)
    _validate_target_lineage(score_metadata=position_metadata, target_metadata=target_metadata)
    target_positions = {key: row for row, key in enumerate(_row_keys(targets))}
    try:
        target_rows = np.asarray([target_positions[key] for key in visual_keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError("a V5 dynamic score row is absent from the target artifact") from error
    translation = np.asarray(targets["translation_errors_m"], dtype=np.float64)[target_rows]
    rotation = np.asarray(targets["rotation_errors_deg"], dtype=np.float64)[target_rows]
    baseline = np.asarray(visual["baseline_selection_scores"], dtype=np.float64)
    if (
        not np.array_equal(baseline, np.asarray(position["baseline_selection_scores"], dtype=np.float64))
        or not np.array_equal(
            np.asarray(visual["baseline_score_top1"], dtype=bool),
            np.asarray(position["baseline_score_top1"], dtype=bool),
        )
        or translation.shape != baseline.shape
        or rotation.shape != baseline.shape
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("visual/control V5 baselines differ or targets are misaligned")
    tie_orders = np.arange(len(visual_keys), dtype=np.int64)
    baseline_rows = _baseline_rows(
        keys=visual_keys,
        baseline_scores=baseline,
        translation_m=translation,
        rotation_deg=rotation,
        tie_break_orders=tie_orders,
    )
    _validate_alpha_zero(
        source_top1=np.asarray(visual["baseline_score_top1"], dtype=bool),
        keys=visual_keys,
        baseline_rows=baseline_rows,
    )
    statistic_field = _STATISTIC_FIELDS[str(args.score_statistic)]
    visual_values = np.asarray(visual[statistic_field], dtype=np.float64)
    position_values = np.asarray(position[statistic_field], dtype=np.float64)
    names = np.asarray(visual["score_names"]).astype(str)
    all_rows: list[dict[str, object]] = list(baseline_rows)
    families: dict[str, object] = {}
    for index, name in enumerate(names.tolist()):
        visual_rows = _per_query_rows(
            keys=visual_keys,
            scores=visual_values[:, index],
            baseline_scores=baseline,
            translation_m=translation,
            rotation_deg=rotation,
            tie_break_orders=tie_orders,
            family=f"visual:{name}",
            score_mode=f"uncalibrated_{args.score_statistic}",
            alpha=None,
        )
        position_rows = _per_query_rows(
            keys=visual_keys,
            scores=position_values[:, index],
            baseline_scores=baseline,
            translation_m=translation,
            rotation_deg=rotation,
            tie_break_orders=tie_orders,
            family=f"position_control:{name}",
            score_mode=f"uncalibrated_{args.score_statistic}",
            alpha=None,
        )
        all_rows.extend(visual_rows)
        all_rows.extend(position_rows)
        visual_summary = _split_summary(visual_rows)
        position_summary = _split_summary(position_rows)
        paired_rank = _paired_rank(visual_rows, position_rows)
        report: dict[str, object] = {
            "visual": {
                "splits": visual_summary,
                "paired_vs_s0": _paired(baseline_rows, visual_rows),
            },
            "position_control": {
                "splits": position_summary,
                "paired_vs_s0": _paired(baseline_rows, position_rows),
            },
            "visual_vs_position_control": {
                "paired_rank": paired_rank,
                "paired_selected_translation": _paired(position_rows, visual_rows),
            },
            "effective_evidence": {
                "visual_mean_effective_point_count": float(
                    np.mean(np.asarray(visual["effective_point_counts"]))
                ),
                "position_mean_effective_point_count": float(
                    np.mean(np.asarray(position["effective_point_counts"]))
                ),
                "visual_mean_effective_view_mass": float(
                    np.mean(np.asarray(visual["effective_view_masses"]))
                ),
                "position_mean_effective_view_mass": float(
                    np.mean(np.asarray(position["effective_view_masses"]))
                ),
            },
        }
        if name == "all_scales":
            report["predeclared_primary_validation_gate"] = _primary_gate(
                visual=visual_summary["validation"],
                position=position_summary["validation"],
                paired_rank=paired_rank,
            )
        families[name] = report
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_query.csv", all_rows)
    summary: dict[str, Any] = {
        "stage": "v5_dynamic_absolute_context_pose_target_audit",
        "format": "v5_dynamic_absolute_context_pose_target_audit_v1",
        "protocol": {
            "target_join_isolated_from_scoring": True,
            "visual_position_control_paired": True,
            "fixed_global_topl": True,
            "explicit_null": True,
            "fixed_support_views": True,
            "candidate_specific_query_projection": True,
            "no_image_retrieval_or_submap": True,
            "no_render": True,
            "evidence_branch": str(args.evidence_branch),
            "raw_scores_are_uncalibrated": True,
            "promotion_allowed": False,
            "verification_protocol": str(args.verification_protocol),
        },
        "primary_score_statistic": str(args.score_statistic),
        "baseline": {"splits": _split_summary(baseline_rows)},
        "families": families,
        "inputs": {
            "visual_score_artifacts": [str(path) for path in visual_paths],
            "visual_score_artifact_sha256": [file_sha256_short(path) for path in visual_paths],
            "position_score_artifacts": [str(path) for path in position_paths],
            "position_score_artifact_sha256": [file_sha256_short(path) for path in position_paths],
            "target_artifact": str(args.target_artifact),
            "target_artifact_sha256": file_sha256_short(Path(args.target_artifact)),
            "visual_score_contract_hash": _canonical_hash(
                {
                    "strict": visual_metadata[0].get("strict_frozen_evidence_contract"),
                    "model": visual_metadata[0].get("model_checkpoint_contract"),
                    "sources": visual_metadata[0].get("source_cache_lineage"),
                }
            ),
            "position_score_contract_hash": _canonical_hash(
                {
                    "strict": position_metadata[0].get("strict_frozen_evidence_contract"),
                    "model": position_metadata[0].get("model_checkpoint_contract"),
                    "sources": position_metadata[0].get("source_cache_lineage"),
                }
            ),
        },
        "outputs": {"per_query": str(output_dir / "per_query.csv")},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
